# Stage 3: line -> word Segments.
# Groups a labelled line's ink into clusters and picks the word breaks that best match the expected word widths.

import numpy as np
from scipy import ndimage

from corpus import Segment
from glyphs import char_width, graphemes, partition
from lines import frame_bbox

# Cost per x-height of blank space left inside a word; makes the widest gaps the preferred word breaks
GAP_WEIGHT = 2.0
# Largest average width misfit per word (squared log of actual / expected width) before the split is rejected
MAX_WIDTH_COST = 1.0
# Expected blank between words, in units of char_width, for scaling expected widths to the line
WORD_GAP = 1.0
# The core band reaches this many x-heights beyond the baseline and the x-height line
CORE_PAD = 0.25


# Groups a line's blobs into clusters that overlap side to side once the slant is undone. Overlap is
# judged on the core band only, so descender tails and ascender loops don't chain letters together.
# Blobs with no ink in the core (dots, accents) join the cluster they sit over. A word is always whole clusters.
# Takes a Line; returns [(left, right, blob ids)] in sheared x pixels, left to right.
def clusters(line):
    # Measures each blob's left and right edge with the slant undone, using only its pixels in the core band
    rows, cols = np.nonzero(line.labels)
    xs = cols - (line.base_row - rows) * line.shear
    lab = line.labels[rows, cols]
    y = (line.base_row - rows) / line.xheight_px
    core = (y > -CORE_PAD) & (y < 1 + CORE_PAD)
    ids = np.unique(lab[core])
    lo, hi = ndimage.minimum(xs[core], lab[core], ids), ndimage.maximum(xs[core], lab[core], ids)

    # Sweeps core blobs left to right, starting a new cluster when a blob begins past the current one's right edge
    out = []
    for b in np.argsort(lo):
        # Joins the current cluster when the blob overlaps it
        if out and lo[b] <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi[b])
            out[-1][2].append(int(ids[b]))
        else:
            out.append([lo[b], hi[b], [int(ids[b])]])

    # Attaches each blob outside the core to the nearest cluster, by its centre
    floating = np.setdiff1d(np.unique(lab), ids)
    for b, cx in zip(floating, ndimage.mean(xs, lab, floating)):
        # Skips floating blobs when the line has no core ink at all
        if not out:
            break
        nearest = min(out, key=lambda c: max(c[0] - cx, 0, cx - c[1]))
        nearest[2].append(int(b))
    return [tuple(c) for c in out]


# Splits 1 OCR'd line into word segments, 1 per word of its text, by choosing which blank gaps are word breaks.
# Takes a Line with text; returns its Segments ([] if no good fit) and sets the line's segment_ids.
def split_words(line):
    words = line.text.split()
    line.segment_ids = []
    groups = clusters(line) if line.labels.any() else []
    n, m = len(words), len(groups)
    # Fails when OCR gave no text, or the ink has fewer separate pieces than the text has words
    if n == 0 or m < n:
        return []

    # Lists cluster edges, and the blank space before each cluster as a running total in x-heights
    s = line.xheight_px
    lo = np.array([g[0] for g in groups])
    hi = np.array([g[1] for g in groups])
    blank = np.r_[0, np.cumsum(lo[1:] - hi[:-1])] / s

    # Expected word widths from their letters, scaled so words plus gaps span the line; only proportions matter
    units = [sum(char_width(ch) for ch in graphemes(word)) for word in words]
    px_per_unit = (hi[-1] - lo[0]) / (sum(units) + WORD_GAP * (n - 1))
    expected = [u * px_per_unit for u in units]

    # Width misfit of word k spanning clusters j..i-1: squared log of its ink width over its expected width
    def width_cost(k, j, i):
        return np.log((hi[i - 1] - lo[j] + 1) / expected[k]) ** 2

    # Each word pays for its width misfit and for the blank space it keeps inside
    def word_cost(k, j, i):
        return width_cost(k, j, i) + GAP_WEIGHT * (blank[i - 1] - blank[j])

    # Picks n-1 gaps as word breaks
    _, bounds = partition(n, m, word_cost)
    # Fails when the words' widths fit their text badly on average (likely wrong word count from OCR)
    misfit = np.mean([width_cost(k, np.array([bounds[k]]), bounds[k + 1])[0] for k in range(n)])
    if misfit > MAX_WIDTH_COST:
        return []

    # Builds 1 segment per word from its clusters' blobs, with the word's OCR confidence
    segs = []
    for k, word in enumerate(words):
        comps = sorted(b for g in groups[bounds[k]:bounds[k + 1]] for b in g[2])
        rows, cols = np.nonzero(np.isin(line.labels, comps))
        conf = line.word_confs[k] if k < len(line.word_confs) else line.ocr_conf
        segs.append(Segment(f"{line.id}_w{k}", line.id, k, comps, frame_bbox(line, rows, cols), text=word,
                            label_source=line.label_source, ocr_conf=conf))
    line.segment_ids = [seg.id for seg in segs]
    return segs


# Splits every line in the corpus into word segments and adds them to it.
def split_corpus(corpus):
    for line in corpus.lines.values():
        corpus.segments.update((seg.id, seg) for seg in split_words(line))
