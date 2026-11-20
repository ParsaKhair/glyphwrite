import math
import tkinter as tk

from PIL import Image, ImageDraw, ImageOps, ImageTk


# Opens images upright. Takes file paths, returns PIL images.
def load_images(paths):
    # Applies the camera's rotation flag, since phone photos are stored sideways
    imgs = [ImageOps.exif_transpose(Image.open(path)) for path in paths]
    return imgs


# Lasso around an image's text region
def select_text_region(image):
    """
    Click polygon vertices on a loaded PIL image, press Enter to crop
    Backspace removes last vertex.
    Returns (crop, (left, top)): an RGBA crop with transparency outside the polygon, at the
    original image resolution, and the crop's top-left corner in the image.
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
        result = (result, (left, top))
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


def main():
    images = load_images(["x.png", "y.jpg"])
