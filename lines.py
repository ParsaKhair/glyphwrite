import math

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from corpus import Line

# Slant candidates as shear (dx per dy), -45 to 45 degrees in 1 degree steps
SLANTS = np.tan(np.radians(np.arange(-45, 45.1, 1.0)))
# Darkness below which nothing counts as ink, whatever Otsu says
MIN_INK = 0.14
# A blob's soft edge reaches down to this fraction of the ink threshold
EDGE = 0.5
# Smallest blob kept, in pixels; anything smaller is sensor noise
MIN_BLOB = 3
# Blank row runs shorter than this fraction of the median blob height are bridged (i-dots, accents)
ROW_GAP = 0.35
# Rows holding at least this fraction of the busiest row's ink count as the core
CORE = 0.5
# A blob smoother than this fraction of its region's median blob is a shadow, not ink
SHADOW_SMOOTH = 0.65
# A blob the lasso edge cuts is kept when at least this fraction of it lies in a text line of uncut blobs
CLIPPED_KEEP = 0.5


# Lighting-corrected darkness of a lasso crop: 0 = paper, 1 = black.
# Takes an RGB or RGBA PIL image (transparent = outside the lasso), returns a float array.
def darkness(image):
    # Flattens the crop onto white paper, so the area outside the lasso reads as paper
    rgba = image.convert("RGBA")
    gray = Image.alpha_composite(Image.new("RGBA", image.size, "white"), rgba).convert("L")

    # Estimates the paper's brightness at every pixel with a blur wide enough to wash out the ink
    paper = gray.filter(ImageFilter.GaussianBlur(radius=max(5, min(image.size) / 20)))

    # Divides out the lighting, so paper reads 0 under shadows and glare alike
    ratio = np.asarray(gray, float) / np.maximum(np.asarray(paper, float), 1)
    return np.clip(1 - ratio, 0, 1)


# Darkness that best separates ink from paper (Otsu's method), floored at MIN_INK.
# Takes a darkness array, returns a float threshold.
def ink_threshold(dark):
    # Histograms darkness into 256 bins, with running pixel counts and sums from the light end
    hist = np.bincount((dark * 255).astype(int).ravel(), minlength=256).astype(float)
    counts = np.cumsum(hist)
    sums = np.cumsum(hist * np.arange(256))

    # Picks the split that maximizes the variance between the light and dark classes
    denominator = counts * (counts[-1] - counts)
    variance = np.zeros(256)
    np.divide((sums[-1] * counts - sums * counts[-1]) ** 2, denominator, out=variance, where=denominator > 0)
    return max((int(np.argmax(variance)) + 0.5) / 255, MIN_INK)


# Rotates a 2D array counterclockwise by angle degrees about its centre, growing the canvas to fit.
# Returns the rotated array and (A, t), which map a rotated (x, y) back to the input as A @ xy + t.
def rotate(arr, angle, order=1):
    h, w = arr.shape
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    A = np.array([[c, -s], [s, c]])

    # Sizes the new canvas to hold the rotated corners, and centres the input in it
    nw, nh = math.ceil(abs(c) * w + abs(s) * h), math.ceil(abs(s) * w + abs(c) * h)
    t = np.array([w / 2, h / 2]) - A @ np.array([nw / 2, nh / 2])

    # Samples the input at each output pixel; ndimage works in (row, col), so x and y swap
    out = ndimage.affine_transform(arr, A[::-1, ::-1], offset=t[::-1], output_shape=(nh, nw), order=order)
    return out, (A, t)


# Rotation in degrees that makes text rows horizontal, searched within +-max_skew (0 disables).
# Takes a boolean ink mask, returns the angle to pass to rotate().
def skew_angle(mask, max_skew=15):
    # Returns 0 when deskewing is disabled
    if not max_skew:
        return 0.0

    # Searches a small copy for speed
    sample = Image.fromarray(mask.astype(np.uint8) * 255)
    sample.thumbnail((800, 800))
    sample = np.asarray(sample, float)

    # Picks the angle whose row sums are most peaked (level text piles ink into few rows); ties prefer less rotation
    def best(candidates):
        def score(angle):
            rows = rotate(sample, angle, order=0)[0].sum(axis=1)
            return np.sum(rows ** 2)
        return float(max(candidates, key=lambda a: (score(a), -abs(a))))

    # Searches in 1 degree steps, then refines in 0.1 degree steps around the best
    angle = best(np.linspace(-max_skew, max_skew, 2 * math.ceil(max_skew) + 1))
    return best(np.linspace(max(-max_skew, angle - 1), min(max_skew, angle + 1), 21))


# Which blobs are errant marks rather than the region's writing. Takes darkness, labels 1..n and the lasso's
# inside mask; returns a bool per blob id 1..n, True = errant.
def errant_blobs(dark, labels, inside):
    n = int(labels.max())
    ids = np.arange(1, n + 1)
    size = np.bincount(labels.ravel(), minlength=n + 1)[1:]

    # Shadows: pen strokes have sharp edges while shadows ramp over many pixels. Mean gradient per darkness
    # also falls as strokes thicken, so each blob is judged against its own region's median
    sharp = ndimage.mean(np.hypot(*np.gradient(dark)), labels, ids) / ndimage.mean(dark, labels, ids)
    ref = sharp[size > 30]
    errant = sharp < SHADOW_SMOOTH * np.median(ref) if len(ref) >= 3 else np.zeros(n, bool)

    # Clipped marks: pieces of the lines above and below that the lasso edge cuts through. Text lines are found
    # from uncut blobs only; a cut blob stays when it lies mostly inside one, as letters touching a tight lasso do
    cut = np.zeros(n + 1, bool)
    cut[labels[ndimage.binary_dilation(~inside, iterations=2) & (labels > 0)]] = True
    cut = cut[1:] & ~errant
    uncut = ~cut & ~errant
    if cut.any() and uncut.any():
        rows = np.zeros(labels.shape[0], bool)
        for top, bottom in line_bands(np.where(np.r_[False, uncut][labels], labels, 0)):
            rows[top:bottom] = True
        in_line = ndimage.mean(np.broadcast_to(rows[:, None], labels.shape), labels, ids)
        errant |= cut & (in_line < CLIPPED_KEEP)
    return errant


# Groups ink into blobs (connected components) with a soft edge: a blob is a patch darker than
# EDGE x threshold that somewhere passes the full threshold, so faint stroke edges stay attached.
# Takes darkness, threshold and optionally the lasso's inside mask (to drop errant marks);
# returns (density 0-1 with paper zeroed, labels 1..n with 0 = paper).
def ink_blobs(dark, threshold, inside=None):
    # Labels every patch darker than the edge level, counting diagonal neighbours as touching
    labels, n = ndimage.label(dark > EDGE * threshold, structure=np.ones((3, 3)))

    # Keeps patches that reach real ink darkness and aren't specks, renumbered 1..k
    ids = np.arange(1, n + 1)
    peak = ndimage.maximum(dark, labels, ids)
    size = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    keep = np.r_[False, (peak > threshold) & (size >= MIN_BLOB)]
    labels = (np.cumsum(keep) * keep)[labels]
    # Drops shadows and clipped marks before ink is normalized, so they can't skew it
    if inside is not None and labels.any():
        keep = np.r_[False, ~errant_blobs(dark, labels, inside)]
        labels = (np.cumsum(keep) * keep)[labels]

    # Returns empty when nothing qualifies as ink
    if not labels.any():
        return np.zeros_like(dark), labels

    # Scales darkness so typical solid ink reads 1, then zeroes everything outside a blob
    full = np.percentile(dark[labels > 0], 90)
    density = np.clip(dark / full, 0, 1) * (labels > 0)
    return density, labels


# Splits a region's ink into text lines: a line is a run of rows holding ink, with short blank runs
# bridged. Every blob lies inside 1 run, so no stroke is cut. Takes labels, returns [(top, bottom)].
def line_bands(labels):
    # Sets the largest bridgeable gap from the typical blob height
    heights = [o[0].stop - o[0].start for o in ndimage.find_objects(labels) if o is not None]
    gap = max(1, round(float(np.median(heights)) * ROW_GAP))

    # Marks rows holding any ink, then fills blank runs no taller than the gap
    occupied = (labels > 0).any(axis=1)
    rows = np.flatnonzero(occupied)
    for a, b in zip(rows[:-1], rows[1:]):
        # Bridges the run when it's short
        if b - a - 1 <= gap:
            occupied[a:b + 1] = True

    # Returns the start and end row of each run of marked rows
    edges = np.diff(np.pad(occupied.astype(int), 1))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


# Finds a line's core (the x-height zone) from its row profile: the rows where ink is dense.
# Takes a line's density, returns (base_row, xheight_px).
def core_rows(density):
    # Sums ink per row, smoothed so a single horizontal stroke doesn't stand out
    profile = ndimage.gaussian_filter1d(density.sum(axis=1), max(1.0, density.shape[0] / 30))

    # Takes the run of dense rows around the busiest row; its bottom is the baseline
    runs, _ = ndimage.label(profile >= CORE * profile.max())
    core = np.flatnonzero(runs == runs[np.argmax(profile)])
    return float(core[-1]), float(core[-1] - core[0] + 1)


# Shear s that makes the histogram of u - v*s most peaked, i.e. lines up the strokes.
# Takes pixel x (u), height above baseline (v) and ink weights (w); returns a shear from SLANTS.
def best_shear(u, v, w):
    def peakiness(s):
        p = u - v * s
        return np.square(np.bincount((p - p.min()).astype(int), w)).sum()
    return float(max(SLANTS, key=peakiness))


# Bbox of some of a line's pixels in frame units (x-heights, y up from the baseline).
# Takes the line and pixel rows + cols, returns (x0, y0, x1, y1).
def frame_bbox(line, rows, cols):
    s = line.xheight_px
    return (cols.min() / s, (line.base_row - rows.max()) / s, cols.max() / s, (line.base_row - rows.min()) / s)


# Builds 1 Line from a band of rows in the straightened region.
# Takes the region's density + labels, the band, the rotation map, the crop's (left, top) on the page,
# and the line's page, id and order; returns the Line.
def build_line(density, labels, band, back, offset, page, line_id, order):
    # Crops to the band's rows, then to the columns its ink spans
    top, bottom = band
    cols = np.flatnonzero(labels[top:bottom].any(axis=0))
    left, right = int(cols[0]), int(cols[-1]) + 1
    lab = labels[top:bottom, left:right]
    dens = density[top:bottom, left:right]

    # Renumbers this line's blobs 1..n, keeping 0 for paper
    lab = np.searchsorted(np.union1d([0], lab), lab)

    # Measures the baseline and x-height, then the slant of the ink standing on that baseline
    base_row, xheight_px = core_rows(dens)
    rows, cols = np.nonzero(lab)
    shear = best_shear(cols.astype(float), base_row - rows, dens[rows, cols])

    # Maps the baseline's ends back to source pixels: undo the line crop, the rotation, then the lasso crop
    A, t = back
    ends = np.array([[left, top + base_row], [right, top + base_row]], float)
    poly = ends @ A.T + t + np.asarray(offset, float)
    return Line(line_id, page, order, [], baseline_poly=poly,
                x_span=(float(poly[:, 0].min()), float(poly[:, 0].max())),
                xheight_px=xheight_px, base_row=base_row, shear=shear, density=dens, labels=lab)


# Moves a line onto a refit baseline row = base + slope * col with the given x-height: shears its pixels
# vertically so that baseline becomes horizontal, then remeasures slant and the baseline's page position.
def relevel(line, slope, base, xheight_px):
    # Pads above (rising baseline) or below (falling) so no ink leaves the canvas
    h, w = line.density.shape
    off = max(0.0, slope * (w - 1))
    size = (h + int(np.ceil(abs(slope) * (w - 1))), w)
    m, o = [[1, slope], [0, 1]], [-off, 0]
    line.density = ndimage.affine_transform(line.density, m, offset=o, output_shape=size, order=1)
    line.labels = ndimage.affine_transform(line.labels, m, offset=o, output_shape=size, order=0)

    # Slides the page baseline's ends along the line's downward normal by how far each end moved
    p = line.baseline_poly
    u = (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])
    shift = np.array([base - line.base_row, base + slope * w - line.base_row])
    line.baseline_poly = p + shift[:, None] * np.array([-u[1], u[0]])

    line.base_row, line.xheight_px = base + off, xheight_px
    rows, cols = np.nonzero(line.labels)
    line.shear = best_shear(cols.astype(float), line.base_row - rows, line.density[rows, cols])


# Turns 1 lasso crop into straightened text lines at full source resolution, top to bottom.
# Takes the crop, its page name, a region index (for IDs) and the crop's (left, top) on the page;
# returns Lines with no segments yet, or [] for a blank crop.
def extract_lines(crop, page, region, offset, max_skew=15):
    # Measures darkness and the ink threshold before rotating, so the added canvas can't sway Otsu
    dark = darkness(crop)
    threshold = ink_threshold(dark)

    # Straightens the region, then groups its ink into blobs
    angle = skew_angle(dark > threshold, max_skew)
    rotated, back = rotate(dark, angle)
    # Rotated canvas corners are outside the lasso too, so the inside mask is rotated rather than the outside
    inside = rotate((np.asarray(crop.convert("RGBA"))[..., 3] > 0).astype(float), angle, order=0)[0] > 0.5
    density, labels = ink_blobs(rotated, threshold, inside)
    # Returns nothing when the crop holds no ink
    if not labels.any():
        return []

    # Builds 1 Line per band of rows
    return [build_line(density, labels, band, back, offset, page, f"{page}_r{region}_l{i}", i)
            for i, band in enumerate(line_bands(labels))]
