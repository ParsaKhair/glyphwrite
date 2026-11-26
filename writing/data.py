# Turns corpus glyphs into fixed-size tensors for the scoring and layout models.

import numpy as np
import torch
from PIL import Image
from torch.utils.data import TensorDataset
from torchvision.datasets import EMNIST

from preprocess.glyphs import glyph_alpha, offset_ok

# Side of the square glyph image the models take, in pixels (EMNIST's 28 x 28 is resized up to it)
SIZE = 32
# Pixels per x-height in the frame-aligned edge crops pair.py takes
EDGE_PX = 16
# EMNIST "byclass" classes in label order: digits, then upper case, then lower case
EMNIST_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


# Crops ink to its bounding box, scales the longer side to SIZE minus a 2 px margin per side (keeping the aspect ratio)
# and centres it, so corpus glyphs and EMNIST images are framed the same way. Takes a 2D array, 0 = paper, 1 = ink;
# returns a [1, SIZE, SIZE] tensor.
def fit_square(ink):
    out = np.zeros((SIZE, SIZE), np.float32)

    # Returns a blank image when there is no ink
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    if len(rows) == 0:
        return torch.from_numpy(out)[None]

    # Crops to the ink's bounding box
    ink = ink[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

    # Scales the longer side to SIZE - 4
    h, w = ink.shape
    fit = (SIZE - 4) / max(h, w)
    new_h, new_w = max(1, round(h * fit)), max(1, round(w * fit))
    img = Image.fromarray(ink.astype(np.float32), mode="F").resize(
        (new_w, new_h), Image.BILINEAR
    )

    # Pastes it centred
    top, left = (SIZE - new_h) // 2, (SIZE - new_w) // 2
    out[top : top + new_h, left : left + new_w] = np.clip(np.asarray(img), 0, 1)
    return torch.from_numpy(out)[None]


# 1 glyph's own ink (glyphs.glyph_alpha), framed by fit_square. Takes the corpus and a glyph id; returns a
# [1, SIZE, SIZE] tensor, 0 = paper, 1 = ink.
def glyph_tensor(corpus, glyph_id):
    glyph = corpus.glyphs[glyph_id]
    alpha, _ = glyph_alpha(corpus.lines[glyph.line_id], glyph)
    return fit_square(alpha)


# The left or right edge of 1 glyph in its line frame (baseline and x-height fixed, slant undone) at EDGE_PX per
# x-height, so where a stroke leaves 1 letter and enters the next keeps its height. Takes the corpus, a glyph id and
# side "left" or "right"; returns a [1, H, W] tensor.
def edge_tensor(corpus, glyph_id, side): ...


# EMNIST "byclass" images and labels for pretraining, reframed by fit_square to match glyph_tensor. Its 62 class
# indices are positions in EMNIST_CHARS, so a charset that starts with EMNIST_CHARS needs no relabelling. Takes a
# download folder and whether to load the train or test split; returns a dataset of ([1, SIZE, SIZE] tensor, class
# index).
def emnist(root, train=True):
    return EMNIST(
        root, split="byclass", train=train, download=True, transform=emnist_image
    )


# 1 EMNIST image as fit_square framed ink. EMNIST stores images transposed, so it is flipped back first. Module level so
# DataLoader workers can pickle it. Takes a 28 x 28 PIL image; returns a [1, SIZE, SIZE] tensor.
def emnist_image(img):
    ink = np.asarray(img, np.float32).T / 255
    return fit_square(ink)


# Labelled corpus glyphs for fine-tuning, truth-labelled ones only unless ocr is True. Without a keep set, glyphs
# flagged for an implausible baseline offset (glyphs.offset_ok) are left out, as compose leaves them out; a keep set
# replaces that check. Images are built once up front, so each
# epoch only reads tensors. Takes the corpus, charset (list of chars, index = class) and optional set of glyph ids to
# keep (e.g. hand-curated); returns a TensorDataset of ([1, SIZE, SIZE] tensor, class index).
def glyph_dataset(corpus, charset, ocr=False, keep=None):
    index = {c: i for i, c in enumerate(charset)}
    images, labels = [], []

    # Sorted by id so the dataset's order doesn't depend on how the corpus was built
    for glyph in sorted(corpus.glyphs.values(), key=lambda g: g.id):
        # Skips when the char has no class, the label came from OCR and ocr is False, or the glyph is flagged
        if glyph.char not in index or (glyph.label_source != "truth" and not ocr):
            continue
        # Skips flagged glyphs, or with a keep set, glyphs not in it
        if (glyph.id not in keep) if keep is not None else not offset_ok(glyph):
            continue
        images.append(glyph_tensor(corpus, glyph.id))
        labels.append(index[glyph.char])

    # Empty tensors of the right shape when nothing qualifies
    if not images:
        return TensorDataset(
            torch.zeros(0, 1, SIZE, SIZE), torch.zeros(0, dtype=torch.long)
        )
    return TensorDataset(torch.stack(images), torch.tensor(labels))


# Neighbouring glyphs within each segment, the positives for pair.py. Takes the corpus; returns [(a id, b id)].
def neighbour_pairs(corpus): ...


# Measured layout of each neighbour pair, the targets for layout.py: core-band gap (as compose.core_extent) and b's
# baseline offset minus a's, both in x-heights. Takes the corpus; returns [(a id, b id, gap, dy)].
def layout_targets(corpus): ...
