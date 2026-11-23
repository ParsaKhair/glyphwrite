# Turns corpus glyphs into fixed-size tensors for the scoring and layout models.

import torch

# Side of the square glyph image the models take, in pixels (EMNIST's 28 x 28 is resized up to it)
SIZE = 32
# Pixels per x-height in the frame-aligned edge crops pair.py takes
EDGE_PX = 16


# 1 glyph's own ink (glyphs.glyph_alpha), cropped, scaled to fit and centred in SIZE x SIZE, as in EMNIST.
# Takes the corpus and a glyph id; returns a [1, SIZE, SIZE] tensor, 0 = paper, 1 = ink.
def glyph_tensor(corpus, glyph_id):
    ...


# The left or right edge of 1 glyph in its line frame (baseline and x-height fixed, slant undone) at EDGE_PX per
# x-height, so where a stroke leaves 1 letter and enters the next keeps its height. Takes the corpus, a glyph id and
# side "left" or "right"; returns a [1, H, W] tensor.
def edge_tensor(corpus, glyph_id, side):
    ...


# EMNIST images and labels for pretraining. Takes a download folder and EMNIST split; returns a dataset of
# ([1, SIZE, SIZE] tensor, class index) with classes in charset order.
def emnist(root, split="byclass"):
    ...


# Labelled corpus glyphs for fine-tuning, truth-labelled ones only unless ocr is True. Takes the corpus and charset
# (list of chars, index = class); returns a dataset of ([1, SIZE, SIZE] tensor, class index).
def glyph_dataset(corpus, charset, ocr=False):
    ...


# Neighbouring glyphs within each segment, the positives for pair.py. Takes the corpus; returns [(a id, b id)].
def neighbour_pairs(corpus):
    ...


# Measured layout of each neighbour pair, the targets for layout.py: core-band gap (as compose.core_extent) and b's
# baseline offset minus a's, both in x-heights. Takes the corpus; returns [(a id, b id, gap, dy)].
def layout_targets(corpus):
    ...
