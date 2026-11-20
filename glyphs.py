import unicodedata

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks

from corpus import Glyph
from lines import best_shear

# Body hangs below the baseline; anchored by its x-height top instead of its bottom
DESCENDERS = set("gjpqyγημρςχ")
# Both ascend and descend; anchored by their centre
ASC_DESC = set("βζξφψ")
NARROW = set("il1.,:;'!|ιı")
SEMI = set("fjrtIJ")
WIDE = set("mwMWωψΩΨΦφ")
# Merging across a blank column is allowed (pen lifts inside 1 letter) but penalized
GAP_SKIP = 2.0
MAX_SPLIT_COST = 1.5


# Splits text into characters as a reader sees them: NFC, accents kept with their base (ά, ῶ).
def graphemes(text):
    out = []
    for c in unicodedata.normalize("NFC", text):
        # Attaches to the previous char when it's a combining mark, else starts a new char
        if out and unicodedata.category(c).startswith("M"):
            out[-1] += c
        else:
            out.append(c)
    return out


# Expected width of 1 char in x-heights. Rough prior; only steers where cuts go.
def char_width(ch):
    c = ch[0]
    # Returns the width of the first class the char belongs to
    if c in NARROW:
        return 0.35
    if c in SEMI:
        return 0.55
    if c in WIDE:
        return 1.3
    # Falls back to 0.95 when uppercase or a digit, 0.8 otherwise
    return 0.95 if c.isupper() or c.isdigit() else 0.8


# Pixel window (r0, r1, c0, c1) around a frame-unit bbox, clipped to the line.
def window(line, bbox, pad=2):
    x0, y0, x1, y1 = bbox
    s, (h, w) = line.xheight_px, line.labels.shape
    return (max(0, int(np.floor(line.base_row - y1 * s)) - pad), min(h, int(np.ceil(line.base_row - y0 * s)) + pad + 1),
            max(0, int(np.floor(x0 * s)) - pad), min(w, int(np.ceil(x1 * s)) + pad + 1))


# Boolean mask of 1 glyph inside window win: owned components + shared ones clipped to its cut.
def glyph_mask(line, glyph, win):
    r0, r1, c0, c1 = win
    lab = line.labels[r0:r1, c0:c1]
    m = np.isin(lab, glyph.comp_ids)
    # Adds shared components, keeping only pixels between the cuts, when the glyph has any
    if glyph.cut_ids:
        rows, cols = np.ogrid[r0:r1, c0:c1]
        xs = (cols - (line.base_row - rows) * line.shear) / line.xheight_px
        m |= np.isin(lab, glyph.cut_ids) & (xs >= glyph.cut[0]) & (xs < glyph.cut[1])
    return m


# Character map of 1 glyph at full resolution.
# Takes line + glyph, returns (alpha = density x mask, (r0, c0) of the crop in the line).
def glyph_alpha(line, glyph):
    win = window(line, glyph.bbox)
    a = line.density[win[0]:win[1], win[2]:win[3]] * glyph_mask(line, glyph, win)
    return a, (win[0], win[2])


# Weighted quantile q of values v with weights w.
def _wquantile(v, w, q):
    i = np.argsort(v)
    c = np.cumsum(w[i])
    return float(v[i][min(np.searchsorted(c, q * c[-1]), len(v) - 1)])


# Frame-unit bbox of pixel coords (gy, gx).
def _bbox(line, gy, gx):
    s = line.xheight_px
    return (gx.min() / s, (line.base_row - gy.max()) / s, gx.max() / s, (line.base_row - gy.min()) / s)


# Measures 1 glyph's pixels. Returns slant (degrees, + leans right) and dy (anchor offset from baseline).
def _measure(line, ch, gy, gx, w):
    slant = float(np.degrees(np.arctan(best_shear(gx.astype(float), line.base_row - gy, w))))
    y = (line.base_row - gy) / line.xheight_px
    # Main body = largest component, so dots/accents don't shift the anchor
    lab = line.labels[gy, gx]
    body = lab == np.bincount(lab).argmax()
    # Anchors on the body top when the char descends, the body centre when it ascends
    # and descends, and the ink bottom otherwise
    if ch[0] in DESCENDERS:
        dy = _wquantile(y[body], w[body], 0.97) - 1
    elif ch[0] in ASC_DESC:
        dy = _wquantile(y[body], w[body], 0.5) - 0.5
    else:
        dy = _wquantile(y, w, 0.03)
    return slant, dy


# Cuts 1 OCR'd word into glyphs, 1 per character of its text, by fitting vertical cuts to its ink.
# Takes line + segment, returns the glyphs ([] if no good fit) and records split stats on the segment.
def split_glyphs(line, seg):
    # Gathers the word's ink pixels and the pixel window around them
    s, chars = line.xheight_px, graphemes(seg.text)
    ry, rx = np.nonzero(np.isin(line.labels, seg.comp_ids))
    win = (ry.min(), ry.max() + 1, rx.min(), rx.max() + 1)

    # Cost of cutting at each column = ink the cut would slice, measured with the slant undone (~1 = 1 stroke)
    xs = rx - (line.base_row - ry) * line.shear
    off = np.floor(xs.min())
    proj = ndimage.gaussian_filter1d(np.bincount((xs - off).astype(int), line.density[ry, rx]), max(1.0, 0.06 * s))
    cost = proj / np.percentile(proj[proj > 0], 75)

    # Candidate cuts = cheapest columns locally (gaps, thin joins), dropping thick ink and the word's edges
    cand, _ = find_peaks(-cost, distance=max(2, int(0.15 * s)))
    cand = cand[(cost[cand] < 0.6) & (cand > 0.15 * s) & (cand < len(cost) - 0.15 * s)]

    # OCR-free check: counts pieces the ink breaks into on its own and compares to the letter count
    seg.n_natural = 1 + int((cost[cand] < 0.15).sum())
    seg.count_match = seg.n_natural == len(chars)
    seg.split_ok, seg.glyph_ids = False, []
    n = len(chars)
    # Fails when OCR gave no text
    if n == 0:
        seg.split_score = 0.0
        return []

    # Lists cut positions P (word edges included), their cost C, and a running count of blank gaps
    P = np.r_[0, cand, len(cost)].astype(float)
    C = np.r_[0, cost[cand], 0]
    gaps = np.cumsum(np.r_[False, cost[cand] < 0.02, False])
    m = len(P)

    # DP over cuts: D[k, i] = cheapest way to fit the first k letters ending at cut i.
    # Each letter pays for a thick cut, a width far from its expected width, and each blank gap it spans
    D = np.full((n + 1, m), np.inf)
    D[0, 0] = 0
    back = np.zeros((n + 1, m), int)
    for k in range(1, n + 1):
        ew = char_width(chars[k - 1]) * s
        for i in range(1, m):
            j = np.arange(i)
            tot = D[k - 1, j] + C[i] + np.log((P[i] - P[j]) / ew) ** 2 + GAP_SKIP * (gaps[i - 1] - gaps[j])
            back[k, i] = int(np.argmin(tot))
            D[k, i] = tot[back[k, i]]

    # Turns the best fit's average cost per letter into a 0-1 score
    total = D[n, m - 1]
    seg.split_score = float(np.exp(-total / n)) if np.isfinite(total) else 0.0
    # Fails when there are fewer candidate cuts than letters need, or the best fit is too costly
    if not np.isfinite(total) or total / n > MAX_SPLIT_COST:
        return []

    # Traces the chosen cuts back from the right edge, giving each glyph's left/right bound in unslanted x
    ends = [m - 1]
    for k in range(n, 0, -1):
        ends.append(back[k, ends[-1]])
    ends = ends[::-1]
    bounds = P[ends] + off

    # Gives each ink blob to the glyph holding its centre; big blobs crossing a cut (joined letters) are shared instead
    lab = line.labels[ry, rx]
    comps = np.unique(lab)
    area = ndimage.sum(np.ones_like(xs), lab, comps)
    lo, hi, cx = ndimage.minimum(xs, lab, comps), ndimage.maximum(xs, lab, comps), ndimage.mean(xs, lab, comps)
    home = np.clip(np.searchsorted(bounds, cx, side="right") - 1, 0, n - 1)
    crosses = np.searchsorted(bounds, lo, side="right") != np.searchsorted(bounds, hi, side="right")
    shared = (area >= 0.3 * s * s) & crosses

    # Builds 1 glyph per letter from its owned blobs plus its slice of the shared ones
    glyphs = []
    for g, ch in enumerate(chars):
        a, b = bounds[g], bounds[g + 1]
        glyph = Glyph(f"{seg.id}_g{g}", seg.id, line.id, g, ch,
                      comp_ids=comps[~shared & (home == g)].tolist(),
                      cut_ids=comps[shared & (lo < b) & (hi >= a)].tolist(),
                      cut=(float(a / s), float(b / s)), bbox=(0, 0, 0, 0), slant=0.0, dy=0.0,
                      ocr_conf=seg.ocr_conf)
        gy, gx = np.nonzero(glyph_mask(line, glyph, win))
        # Fails when a glyph ends up with no ink
        if len(gy) == 0:
            return []

        # Measures bbox, slant and baseline offset from the glyph's own pixels
        gy, gx = gy + win[0], gx + win[2]
        glyph.bbox = _bbox(line, gy, gx)
        glyph.slant, glyph.dy = _measure(line, ch, gy, gx, line.density[gy, gx])

        # Scores this letter's share of the DP cost, splitting each bounding cut's cost with its neighbour
        j, i = ends[g], ends[g + 1]
        gcost = (C[j] + C[i]) / 2 + np.log((P[i] - P[j]) / (char_width(ch) * s)) ** 2 + GAP_SKIP * (gaps[i - 1] - gaps[j])
        glyph.split_score = float(np.exp(-gcost))
        glyphs.append(glyph)
    seg.split_ok = True
    seg.glyph_ids = [g.id for g in glyphs]
    return glyphs


# Splits every segment in the corpus and adds the glyphs to it.
def split_corpus(corpus):
    for seg in corpus.segments.values():
        corpus.glyphs.update((g.id, g) for g in split_glyphs(corpus.lines[seg.line_id], seg))
