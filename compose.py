from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from glyphs import char_width, glyph_alpha, glyph_mask, graphemes, window

# Gaps in x-heights, used only when the corpus has no neighbouring letters or words to measure
LETTER_GAP = 0.1
WORD_GAP = 0.8


# 1 glyph in the output: its ink lands at its own frame coordinates shifted by (x, y) x-heights; output baseline is y = 0.
# glyph_id is None when the corpus has no sample of char, which leaves a blank.
@dataclass
class Placement:
    char: str
    glyph_id: Optional[str]
    x: float
    y: float


# Picks 1 random sample per character. Takes the corpus, text and optional seed; returns per word [(char, glyph id or None)].
def choose(corpus, text, seed=None):
    rng = np.random.default_rng(seed)
    samples = {}
    # Sorted so a seed always picks the same glyphs from the same corpus
    for g in sorted(corpus.glyphs.values(), key=lambda g: g.id):
        samples.setdefault(g.char, []).append(g.id)
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


# Lays chosen glyphs out left to right, core band to core band at the corpus' typical gaps.
# Takes the corpus, choose() output and optionally spacing() output; returns [Placement].
def place(corpus, chosen, gaps=None):
    letter_gap, word_gap = gaps or spacing(corpus)
    out, pen = [], 0.0
    for word in chosen:
        for ch, gid in word:
            if gid is None:
                out.append(Placement(ch, None, pen, 0.0))
                pen += char_width(ch) + letter_gap
                continue
            glyph = corpus.glyphs[gid]
            lo, hi = glyph_extent(corpus, gid)
            # dy is measured from a letter's anchor (baseline, body top or centre); punctuation has none, so it keeps
            # its height from the source line
            y = -glyph.dy if ch[0].isalnum() else 0.0
            # The y shift slides the slant-undone edge by y * shear, so it is added back to keep the core edge at pen
            out.append(Placement(ch, gid, pen - lo + y * corpus.lines[glyph.line_id].shear, y))
            pen += hi - lo + letter_gap
        pen += word_gap - letter_gap
    return out


# Draws placements dark on white at px_per_xh pixels per x-height with pad x-heights of margin. Returns a PIL image.
def render(corpus, placements, px_per_xh=40, pad=0.3):
    # Each glyph's ink (density inside its own components, so no neighbour ink) scaled to px_per_xh,
    # with its top-left corner in output x-heights (y up)
    pieces = []
    for p in (p for p in placements if p.glyph_id):
        glyph = corpus.glyphs[p.glyph_id]
        line = corpus.lines[glyph.line_id]
        alpha, (r0, c0) = glyph_alpha(line, glyph)
        s = px_per_xh / line.xheight_px
        size = (max(1, round(alpha.shape[1] * s)), max(1, round(alpha.shape[0] * s)))
        ink = np.asarray(Image.fromarray(alpha.astype(np.float32), "F").resize(size, Image.BILINEAR))
        pieces.append((ink, c0 / line.xheight_px + p.x, (line.base_row - r0) / line.xheight_px + p.y))

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
    return Image.fromarray((255 * (1 - np.clip(canvas, 0, 1))).astype(np.uint8))


# Writes text in the corpus' handwriting. Returns (PIL image, placements); placements with glyph_id None are missing chars.
def compose(corpus, text, px_per_xh=40, seed=None):
    placements = place(corpus, choose(corpus, text, seed))
    return render(corpus, placements, px_per_xh), placements
