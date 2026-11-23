# Stage 5: text -> handwritten image.
# Picks a corpus glyph per char, places them with measured letter and word gaps, optionally rescales
# them to a typical size, and rasterizes the result. choose, place and scale are swappable.

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage

from preprocess.glyphs import ASC_DESC, DESCENDERS, char_width, glyph_alpha, glyph_mask, graphemes, offset_ok, window

# Gaps in x-heights, used only when the corpus has no neighbouring letters or words to measure
LETTER_GAP = 0.1
WORD_GAP = 0.8
# Scales outside this range would mostly blow up mislabelled samples, so size_scaler clamps to it
MIN_SCALE, MAX_SCALE = 0.67, 1.5
# A char needs this many samples before its median size is trusted as a target
MIN_SIZE_SAMPLES = 3


# 1 glyph in the output: ink at the glyph's frame point p lands at scale * p + (x, y) x-heights; output baseline is
# y = 0. glyph_id is None when the corpus has no sample of char, which leaves a blank.
@dataclass
class Placement:
    char: str
    glyph_id: Optional[str]
    x: float
    y: float
    scale: float = 1.0


# Picks 1 random sample per character, skipping glyphs with an implausible baseline offset unless a char has
# nothing else. Takes the corpus, text and optional seed; returns per word [(char, glyph id or None)].
def choose(corpus, text, seed=None):
    rng = np.random.default_rng(seed)
    good, flagged = {}, {}
    # Sorted so a seed always picks the same glyphs from the same corpus
    for g in sorted(corpus.glyphs.values(), key=lambda g: g.id):
        (good if offset_ok(g) else flagged).setdefault(g.char, []).append(g.id)
    samples = {**flagged, **good}
    pick = lambda ch: samples[ch][rng.integers(len(samples[ch]))] if ch in samples else None
    return [[(ch, pick(ch)) for ch in graphemes(word)] for word in text.split()]


# Left and right ink edge in x-heights with the line's slant undone. Only the core band (baseline to x-height) counts,
# so ascender loops, descender tails and crossbars overhang their neighbours as in real writing.
def core_extent(line, rows, cols):
    xs = (cols - (line.base_row - rows) * line.shear) / line.xheight_px
    y = (line.base_row - rows) / line.xheight_px
    core = (y >= 0) & (y <= 1)
    xs = xs[core] if core.any() else xs
    return float(xs.min()), float(xs.max())


# core_extent of 1 glyph's own ink.
def glyph_extent(corpus, glyph_id):
    glyph = corpus.glyphs[glyph_id]
    line = corpus.lines[glyph.line_id]
    win = window(line, glyph.bbox)
    rows, cols = np.nonzero(glyph_mask(line, glyph, win))
    return core_extent(line, rows + win[0], cols + win[2])


# Median core-band gap between neighbouring letters and between neighbouring words in the corpus' segments.
# Returns (letter gap, word gap) in x-heights; negative means the neighbours overlap.
def spacing(corpus):
    letter, word = [], []
    for line in corpus.lines.values():
        prev = None
        for seg in (corpus.segments[s] for s in line.segment_ids):
            rows, cols = np.nonzero(np.isin(line.labels, seg.comp_ids))
            lo, hi = core_extent(line, rows, cols)
            if prev is not None:
                word.append(lo - prev)
            prev = hi
            ext = [glyph_extent(corpus, g) for g in seg.glyph_ids]
            letter += [b[0] - a[1] for a, b in zip(ext, ext[1:])]
    return (float(np.median(letter)) if letter else LETTER_GAP, float(np.median(word)) if word else WORD_GAP)


# Where a glyph's anchor sits in its own frame and where it belongs in the output, both as heights in x-heights.
# dy is measured from each letter's anchor (glyphs._measure); punctuation has none, so it keeps its source height.
def anchor(glyph):
    c = glyph.char[0]
    if not c.isalnum():
        return 0.0, 0.0
    if c in DESCENDERS:
        return 1 + glyph.dy, 1.0
    if c in ASC_DESC:
        return 0.5 + glyph.dy, 0.5
    return glyph.dy, 0.0


# Lays chosen glyphs out left to right, core band to core band at the corpus' typical gaps.
# Takes the corpus, choose() output, optionally spacing() output and a scale function (glyph id -> factor, e.g.
# size_scaler); returns [Placement].
def place(corpus, chosen, gaps=None, scale=None):
    letter_gap, word_gap = gaps or spacing(corpus)
    out, pen = [], 0.0
    for word in chosen:
        for ch, gid in word:
            if gid is None:
                out.append(Placement(ch, None, pen, 0.0))
                pen += char_width(ch) + letter_gap
                continue
            glyph = corpus.glyphs[gid]
            s = scale(gid) if scale else 1.0
            lo, hi = glyph_extent(corpus, gid)
            # Scales about the frame origin, then moves the anchor to its height
            ay, target = anchor(glyph)
            y = target - s * ay
            # Moving y slides the slant-undone edge by y * shear, so it is added back to keep the core edge at pen
            out.append(Placement(ch, gid, pen - s * lo + y * corpus.lines[glyph.line_id].shear, y, s))
            pen += s * (hi - lo) + letter_gap
        pen += word_gap - letter_gap
    return out


# Size measures: (glyph id -> value, power), where scaling a glyph by s multiplies its value by s ** power.

# Total ink (alpha) in x-height^2; follows both size and stroke thickness.
def ink_mass(corpus, glyph_id):
    glyph = corpus.glyphs[glyph_id]
    alpha, _ = glyph_alpha(corpus.lines[glyph.line_id], glyph)
    return float(alpha.sum()) / corpus.lines[glyph.line_id].xheight_px ** 2


# Ink mass over stroke width, roughly the length of pen path in x-heights; thickness cancels out.
# Width = 2 x the median distance to paper along the stroke centres (local maxima of the distance transform).
def stroke_length(corpus, glyph_id):
    glyph = corpus.glyphs[glyph_id]
    line = corpus.lines[glyph.line_id]
    alpha, _ = glyph_alpha(line, glyph)
    dist = ndimage.distance_transform_edt(alpha > 0.3)
    ridge = (dist > 0) & (dist == ndimage.maximum_filter(dist, 3))
    width = 2 * float(np.median(dist[ridge])) / line.xheight_px if ridge.any() else 0.0
    return ink_mass(corpus, glyph_id) / width if width > 0 else 0.0


# Height of the glyph's ink box in x-heights.
def ink_height(corpus, glyph_id):
    x0, y0, x1, y1 = corpus.glyphs[glyph_id].bbox
    return y1 - y0


MEASURES = {"mass": (ink_mass, 2), "length": (stroke_length, 1), "height": (ink_height, 1)}


# Scale function that brings each glyph's size measure to its char's median, clamped to [MIN_SCALE, MAX_SCALE].
# Takes the corpus and a MEASURES key; chars with too few samples, or a glyph measuring 0, stay at scale 1.
def size_scaler(corpus, measure):
    fn, power = MEASURES[measure]
    value = {gid: fn(corpus, gid) for gid in corpus.glyphs}
    by_char = {}
    for gid, v in value.items():
        by_char.setdefault(corpus.glyphs[gid].char, []).append(v)
    target = {ch: float(np.median(vs)) for ch, vs in by_char.items() if len(vs) >= MIN_SIZE_SAMPLES}

    def scale(gid):
        t, v = target.get(corpus.glyphs[gid].char), value[gid]
        return float(np.clip((t / v) ** (1 / power), MIN_SCALE, MAX_SCALE)) if t and v > 0 else 1.0
    return scale


# Draws placements as an ink canvas (0 = paper, 1 = ink) at px_per_xh pixels per x-height with pad x-heights of margin.
# Returns (canvas, left, top): the canvas' top-left corner in output x-heights (y up).
def rasterize(corpus, placements, px_per_xh=40, pad=0.3):
    # Each glyph's ink (density inside its own components, so no neighbour ink) scaled by placement and resolution,
    # with its top-left corner in output x-heights
    pieces = []
    for p in (p for p in placements if p.glyph_id):
        glyph = corpus.glyphs[p.glyph_id]
        line = corpus.lines[glyph.line_id]
        alpha, (r0, c0) = glyph_alpha(line, glyph)
        s = p.scale * px_per_xh / line.xheight_px
        size = (max(1, round(alpha.shape[1] * s)), max(1, round(alpha.shape[0] * s)))
        ink = np.asarray(Image.fromarray(alpha.astype(np.float32), "F").resize(size, Image.BILINEAR))
        pieces.append((ink, p.scale * c0 / line.xheight_px + p.x, p.scale * (line.base_row - r0) / line.xheight_px + p.y))

    # Canvas spans all ink, blanks for missing chars, and at least the x-height band
    blanks = [p for p in placements if not p.glyph_id]
    left = min([x for _, x, _ in pieces] + [p.x for p in blanks] + [0.0]) - pad
    right = max([x + a.shape[1] / px_per_xh for a, x, _ in pieces] + [p.x + char_width(p.char) for p in blanks] + [0.0]) + pad
    top = max([y for _, _, y in pieces] + [1.0]) + pad
    bottom = min([y - a.shape[0] / px_per_xh for a, _, y in pieces] + [0.0]) - pad
    canvas = np.zeros((int(np.ceil((top - bottom) * px_per_xh)), int(np.ceil((right - left) * px_per_xh))))

    # Max, not sum, so joins drawn by both neighbours don't darken
    for ink, x, y in pieces:
        r, c = round((top - y) * px_per_xh), round((x - left) * px_per_xh)
        region = canvas[r:r + ink.shape[0], c:c + ink.shape[1]]
        np.maximum(region, ink[:region.shape[0], :region.shape[1]], out=region)
    return canvas, left, top


# rasterize as a PIL image, dark ink on white.
def render(corpus, placements, px_per_xh=40, pad=0.3):
    canvas, _, _ = rasterize(corpus, placements, px_per_xh, pad)
    return Image.fromarray((255 * (1 - np.clip(canvas, 0, 1))).astype(np.uint8))


# Writes text in the corpus' handwriting. choose, place and scale are swappable strategies with the signatures of
# choose, place and size_scaler(...). Returns (PIL image, placements); glyph_id None marks a missing char.
def compose(corpus, text, px_per_xh=40, seed=None, choose=choose, place=place, scale=None):
    placements = place(corpus, choose(corpus, text, seed), scale=scale)
    return render(corpus, placements, px_per_xh), placements
