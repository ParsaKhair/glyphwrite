import unicodedata

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks

from corpus import Glyph
from lines import best_shear, frame_bbox, relevel

# Body hangs below the baseline; anchored by its x-height top instead of its bottom
DESCENDERS = set("gjpqyγημρςχ")
# Both ascend and descend; anchored by their centre
ASC_DESC = set("βζξφψ")
# Nothing above the x-height or below the baseline, so their tops mark the x-height
CORE = set("acemnorsuvwxzαεικνοσυω")
# A line's frame is refit only from at least this many baseline points and x-height points
MIN_FIT_POINTS = 4
# Steeper fitted baselines (dy per dx) are taken as bad fits, e.g. 2 written lines merged into 1
MAX_SLOPE = 0.12
NARROW = set("il1.,:;'!|ιı")
SEMI = set("fjrtIJ")
WIDE = set("mwMWωψΩΨΦφ")
# Merging across a blank column is allowed (pen lifts inside 1 letter) but penalized
GAP_SKIP = 2.0
MAX_SPLIT_COST = 1.5
# A glyph whose anchor sits further than this (x-heights) from where its letter belongs is flagged: in a sample
# of these, most were wrong ink (a crossbar alone, a connector, a neighbour's stroke attached)
MAX_DY = 0.5


# Whether a glyph's measured baseline offset is plausible; flagged glyphs are skipped by compose when it can.
def offset_ok(glyph):
    return abs(glyph.dy) <= MAX_DY


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


# Relative width of 1 char (roughly x-heights in a typical hand). Only its proportion to other chars steers cuts.
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


# Splits m units in a row into n consecutive pieces at the lowest total cost, by DP over piece ends.
# cost(k, j, i) gives the cost of piece k (from 0) covering units j..i-1, for an array of starts j.
# Returns (total, bounds): piece k covers units bounds[k]..bounds[k+1]-1; total is inf when nothing fits.
def partition(n, m, cost):
    # D[k, i] = cheapest way to cover the first i units with k pieces; back[k, i] = where piece k starts
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    back = np.zeros((n + 1, m + 1), int)
    for k in range(1, n + 1):
        for i in range(1, m + 1):
            j = np.arange(i)
            tot = D[k - 1, j] + cost(k - 1, j, i)
            back[k, i] = int(np.argmin(tot))
            D[k, i] = tot[back[k, i]]

    # Traces the chosen piece starts back from the last unit
    bounds = [m]
    for k in range(n, 0, -1):
        bounds.append(back[k, bounds[-1]])
    return float(D[n, m]), bounds[::-1]


# Measures 1 glyph's pixels. Returns slant (degrees, + leans right) and dy (anchor offset from baseline).
def _measure(line, ch, gy, gx, w):
    slant = float(np.degrees(np.arctan(best_shear(gx.astype(float), line.base_row - gy, w))))
    y = (line.base_row - gy) / line.xheight_px
    # Main body = largest component, so dots/accents don't shift the anchor
    lab = line.labels[gy, gx]
    body = lab == np.bincount(lab).argmax()
    # Anchors on the body top when the char descends, the body centre when it ascends and descends, and the ink
    # bottom otherwise. Column medians and quantiles of column bottoms were tried and flagged as many glyphs or more:
    # large offsets mostly come from wrong ink (fragments, merged letters), which offset_ok flags instead
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

    # Expected letter widths from char_width, scaled so the letters span the word; only proportions matter,
    # since letter width relative to x-height varies a lot between writers
    unit = len(cost) / sum(char_width(ch) for ch in chars)
    expected = [char_width(ch) * unit for ch in chars]

    # Each letter pays for a thick cut at its right end, a width far from its expected width, and each blank gap it spans
    def letter_cost(k, j, i):
        return C[i] + np.log((P[i] - P[j]) / expected[k]) ** 2 + GAP_SKIP * (gaps[i - 1] - gaps[j])

    # Picks n-1 of the candidate cuts, then turns the average cost per letter into a 0-1 score
    total, ends = partition(n, m - 1, letter_cost)
    seg.split_score = float(np.exp(-total / n)) if np.isfinite(total) else 0.0
    # Fails when there are fewer candidate cuts than letters need, or the best fit is too costly
    if not np.isfinite(total) or total / n > MAX_SPLIT_COST:
        return []

    # Converts the chosen cuts to each glyph's left/right bound in unslanted x
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
                      label_source=seg.label_source, ocr_conf=seg.ocr_conf)
        gy, gx = np.nonzero(glyph_mask(line, glyph, win))
        # Fails when a glyph ends up with no ink
        if len(gy) == 0:
            return []

        # Measures bbox, slant and baseline offset from the glyph's own pixels
        gy, gx = gy + win[0], gx + win[2]
        glyph.bbox = frame_bbox(line, gy, gx)
        glyph.slant, glyph.dy = _measure(line, ch, gy, gx, line.density[gy, gx])

        # Scores this letter's share of the DP cost, splitting each bounding cut's cost with its neighbour
        j, i = ends[g], ends[g + 1]
        gcost = (C[j] + C[i]) / 2 + np.log((P[i] - P[j]) / expected[g]) ** 2 + GAP_SKIP * (gaps[i - 1] - gaps[j])
        glyph.split_score = float(np.exp(-gcost))
        glyphs.append(glyph)
    seg.split_ok = True
    seg.glyph_ids = [g.id for g in glyphs]
    return glyphs


# Splits every segment in the corpus and adds the glyphs to it.
def split_corpus(corpus):
    for seg in corpus.segments.values():
        corpus.glyphs.update((g.id, g) for g in split_glyphs(corpus.lines[seg.line_id], seg))


# Straight line y = base + slope * x through points, refit 3 times without outliers (descender tails,
# ascender tops). Returns (slope, base).
def _robust_line(x, y):
    keep = np.ones(len(x), bool)
    for _ in range(3):
        slope, base = np.polyfit(x[keep], y[keep], 1)
        r = y - (base + slope * x)
        keep = np.abs(r) <= 2.5 * np.median(np.abs(r[keep])) + 1
        if keep.sum() < 2:
            break
    return float(slope), float(base)


# Refits a line's baseline and x-height from its glyphs' letters, which say where each should touch: every
# letter and digit outside DESCENDERS/ASC_DESC sits on the baseline, every CORE letter and descender body
# reaches the x-height. Takes the line and its glyphs; returns (slope, base row at col 0, x-height px) or None.
def fit_frame(line, glyphs):
    bx, by, tx, ty = [], [], [], []
    for g in glyphs:
        win = window(line, g.bbox)
        rows, cols = np.nonzero(glyph_mask(line, g, win))
        rows, cols = rows + win[0], cols + win[2]
        w = line.density[rows, cols]
        x, c = float(np.average(cols, weights=w)), g.char[0]
        if c.isalnum() and c not in DESCENDERS and c not in ASC_DESC:
            bx.append(x)
            by.append(_wquantile(rows, w, 0.97))
        if c in CORE or c in DESCENDERS:
            # Body = largest component, so an i-dot or a joining stroke's neighbour can't set the top
            lab = line.labels[rows, cols]
            body = lab == np.bincount(lab).argmax()
            tx.append(x)
            ty.append(_wquantile(rows[body], w[body], 0.03))
    if len(bx) < MIN_FIT_POINTS or len(tx) < MIN_FIT_POINTS:
        return None
    slope, base = _robust_line(np.array(bx), np.array(by))
    # Short lines can't pin a slope; flat is safer than a wild tilt
    if np.ptp(bx) < 4 * line.xheight_px or abs(slope) > MAX_SLOPE:
        slope, base = 0.0, float(np.median(by))
    xheight = float(np.median(base + slope * np.array(tx) - np.array(ty)))
    return (slope, base, xheight) if xheight > 0 else None


# Refits every line's frame from a 1st glyph split, then clears the segments and glyphs, which were measured in
# the old frame; run words.split_corpus and split_corpus again afterwards. Returns the ids of lines refit.
def refit_frames(corpus):
    by_line = {}
    for g in corpus.glyphs.values():
        by_line.setdefault(g.line_id, []).append(g)
    refit = []
    for line_id, glyphs in by_line.items():
        fit = fit_frame(corpus.lines[line_id], glyphs)
        if fit:
            relevel(corpus.lines[line_id], *fit)
            refit.append(line_id)
    corpus.segments, corpus.glyphs = {}, {}
    return refit
