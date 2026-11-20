import math
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk
from scipy import ndimage


def load_images(paths):
    imgs = [Image.open(path) for path in paths]
    return imgs


# Lasso around an image's text region
def select_text_region(image):
    """
    Click polygon vertices on a loaded PIL image, press Enter to crop
    Backspace removes last vertex.
    Returns an RGBA crop with transparency outside the polygon, at the original image resolution.
    Press Escape or close the window to cancel and return None.
    """
    window = tk.Tk()
    window.title(
        "Select text: click vertices, Enter to crop, Backspace to undo, Esc to cancel"
    )
    preview = image.copy()
    preview.thumbnail(
        (
            max(1, window.winfo_screenwidth() - 100),
            max(1, window.winfo_screenheight() - 150),
        )
    )
    photo = ImageTk.PhotoImage(preview, master=window)
    canvas = tk.Canvas(
        window, width=preview.width, height=preview.height, highlightthickness=0
    )
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=photo)
    points = []
    result = None

    def position(event):
        return (
            max(0, min(event.x, preview.width)),
            max(0, min(event.y, preview.height)),
        )

    def redraw():
        canvas.delete("selection")
        if len(points) >= 3:
            canvas.create_polygon(
                points, outline="red", fill="", width=2, tags="selection"
            )
        elif len(points) == 2:
            canvas.create_line(points, fill="red", width=2, tags="selection")
        for x, y in points:
            canvas.create_oval(
                x - 3, y - 3, x + 3, y + 3, fill="red", outline="red", tags="selection"
            )

    def add_vertex(event):
        point = position(event)
        if point not in points:
            points.append(point)
            redraw()

    def undo(event):
        if points:
            points.pop()
            redraw()

    def confirm(event):
        nonlocal result
        if len(points) < 3:
            return
        area = sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
        if area == 0:
            return
        scale_x = image.width / preview.width
        scale_y = image.height / preview.height
        vertices = [(x * scale_x, y * scale_y) for x, y in points]
        left = math.floor(min(x for x, y in vertices))
        top = math.floor(min(y for x, y in vertices))
        right = math.ceil(max(x for x, y in vertices))
        bottom = math.ceil(max(y for x, y in vertices))
        result = image.crop((left, top, right, bottom)).convert("RGBA")
        mask = Image.new("L", result.size, 0)
        ImageDraw.Draw(mask).polygon(
            [(x - left, y - top) for x, y in vertices], fill=255
        )
        # Clear the outside pixels as well as their alpha, so converting to RGB
        # cannot bring back text outside the selected polygon.
        result = Image.composite(
            result, Image.new("RGBA", result.size, (255, 255, 255, 0)), mask
        )
        window.destroy()

    canvas.bind("<Button-1>", add_vertex)
    window.bind("<BackSpace>", undo)
    window.bind("<Return>", confirm)
    window.bind("<Escape>", lambda event: window.destroy())
    try:
        window.mainloop()
    finally:
        preview.close()
    return result


def preprocess_text_region(image, line_height=64, padding=8, max_skew=15):
    """Return top-to-bottom handwriting line snippets as grayscale PIL images.

    Assumes dark ink on light paper and roughly horizontal, separated lines.
    Corrects rotations up to max_skew degrees and uneven lighting, then produces
    black ink on white. Each line's ink is scaled to line_height pixels with
    padding on all sides; characters and words within a line stay together.
    Blank images return []. This isolates ink, not handwriting versus print.

    Args:
        image: PIL image containing the text region to preprocess.
        line_height: Target ink height in pixels per line, before padding (>= 1).
        padding: White border width in pixels on each side of a line (>= 0).
        max_skew: Maximum rotation correction in degrees (0–45); 0 disables it.
    """
    if line_height < 1 or padding < 0 or not 0 <= max_skew <= 45:
        raise ValueError("Use line_height >= 1, padding >= 0, and max_skew in [0, 45].")

    rgba = image.convert("RGBA")
    gray = Image.alpha_composite(Image.new("RGBA", image.size, "white"), rgba).convert("L")
    pixels = np.asarray(gray, dtype=float)
    if pixels.max() - pixels.min() < 10:
        return []
    background = np.asarray(gray.filter(ImageFilter.GaussianBlur(
        radius=max(5, min(image.size) / 20))), dtype=float)
    normalized = np.clip(pixels * 255 / np.maximum(background, 1), 0, 255).astype(np.uint8)

    # Otsu's threshold separates the ink and paper intensity distributions.
    histogram = np.bincount(normalized.ravel(), minlength=256)
    counts = np.cumsum(histogram)
    sums = np.cumsum(histogram * np.arange(256))
    denominator = counts * (counts[-1] - counts)
    variance = np.zeros(256)
    np.divide((sums[-1] * counts - sums * counts[-1]) ** 2,
              denominator, out=variance, where=denominator > 0)
    ink = normalized <= min(int(np.argmax(variance)), 220)

    # Remove isolated pixel noise while retaining small dots and accents.
    labels, _ = ndimage.label(ink, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())
    keep = sizes >= 3
    keep[0] = False
    ink = keep[labels]
    if not ink.any():
        return []

    mask = Image.fromarray(ink.astype(np.uint8) * 255)
    sample = mask.copy()
    sample.thumbnail((800, 800))

    def alignment_score(angle):
        rotated = sample.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)
        rows = np.asarray(rotated, dtype=float).sum(axis=1) / 255
        return np.sum(rows ** 2)

    angle = 0.0
    if max_skew:
        candidates = np.linspace(-max_skew, max_skew, int(math.ceil(max_skew * 2)) + 1)
        # Prefer no rotation when scores tie.
        angle = max(candidates, key=lambda a: (alignment_score(a), -abs(a)))
        candidates = np.linspace(max(-max_skew, angle - 1), min(max_skew, angle + 1), 21)
        angle = max(candidates, key=lambda a: (alignment_score(a), -abs(a)))
    mask = mask.rotate(float(angle), resample=Image.Resampling.NEAREST, expand=True)
    ink = np.asarray(mask) > 0

    # Bridge small vertical gaps (e.g. dots above letters), but never split
    # horizontally into words or individual glyphs.
    labels, _ = ndimage.label(ink, structure=np.ones((3, 3)))
    heights = [region[0].stop - region[0].start
               for region in ndimage.find_objects(labels) if region is not None]
    gap = max(1, round(float(np.median(heights)) * 0.35))
    occupied = ink.any(axis=1)
    rows = np.flatnonzero(occupied)
    for previous, current in zip(rows[:-1], rows[1:]):
        if current - previous - 1 <= gap:
            occupied[previous:current + 1] = True
    edges = np.diff(np.pad(occupied.astype(int), (1, 1)))
    snippets = []
    for top, bottom in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        columns = np.flatnonzero(ink[top:bottom].any(axis=0))
        line = ink[top:bottom, columns[0]:columns[-1] + 1]
        snippet = Image.fromarray(np.where(line, 0, 255).astype(np.uint8))
        width = max(1, round(snippet.width * line_height / snippet.height))
        snippet = snippet.resize((width, line_height), Image.Resampling.LANCZOS)
        padded = Image.new("L", (width + 2 * padding, line_height + 2 * padding), 255)
        padded.paste(snippet, (padding, padding))
        snippets.append(padded)
    return snippets


def main():
    images = load_images(["x.png", "y.jpg"])
