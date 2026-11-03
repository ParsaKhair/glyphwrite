import math
import tkinter as tk

from PIL import Image, ImageTk


def load_images(paths):
    imgs = [Image.open(path) for path in paths]
    return imgs


# Lasso around an image's text region
def select_text_region(image):
    """
    Pass loaded PIL image
    Press Escape or close the window to cancel and return None.
    """
    window = tk.Tk()
    window.title("Select text: drag a rectangle, Enter to confirm, Esc to cancel")
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
    rectangle = canvas.create_rectangle(0, 0, 0, 0, outline="red", width=2)
    start = None
    bounds = None
    result = None

    def position(event):
        return (
            max(0, min(event.x, preview.width)),
            max(0, min(event.y, preview.height)),
        )

    def begin(event):
        nonlocal start, bounds
        start = position(event)
        bounds = None
        canvas.coords(rectangle, *start, *start)

    def drag(event):
        nonlocal bounds
        if start is None:
            return
        x, y = position(event)
        bounds = (
            min(start[0], x),
            min(start[1], y),
            max(start[0], x),
            max(start[1], y),
        )
        canvas.coords(rectangle, *bounds)

    def confirm(event):
        nonlocal result
        if bounds is None or bounds[0] == bounds[2] or bounds[1] == bounds[3]:
            return
        scale_x = image.width / preview.width
        scale_y = image.height / preview.height
        left, top, right, bottom = bounds
        result = image.crop(
            (
                math.floor(left * scale_x),
                math.floor(top * scale_y),
                math.ceil(right * scale_x),
                math.ceil(bottom * scale_y),
            )
        )
        window.destroy()

    canvas.bind("<Button-1>", begin)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<ButtonRelease-1>", drag)
    window.bind("<Return>", confirm)
    window.bind("<Escape>", lambda event: window.destroy())
    try:
        window.mainloop()
    finally:
        preview.close()
    return result


def main():
    images = load_images(["x.png", "y.jpg"])
