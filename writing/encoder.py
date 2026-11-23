# E(g): CNN embedding of a glyph image, shared by the scoring and layout models.

import torch
from torch import nn


# E(g): a small CNN from a glyph image to an embedding. Takes [B, 1, SIZE, SIZE]; returns [B, dim].
class GlyphEncoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        ...

    def forward(self, x):
        ...


# Letter classifier W E(g). Takes an encoder and the number of classes; forward returns [B, n_classes] logits.
class Classifier(nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        ...

    def forward(self, x):
        ...


# Cross-entropy: L = -log softmax(W E(g))_y. Takes [B, n_classes] logits and [B] class indices; returns a scalar.
def classification_loss(logits, labels):
    ...


# Widens the head for chars the corpus has and EMNIST lacks (Greek, punctuation), keeping the trained rows.
# Takes a Classifier and the new class count; returns nothing.
def grow_head(model, n_classes):
    ...


# Trains a Classifier with classification_loss: on EMNIST to pretrain, then on glyph_dataset to fine-tune at a lower lr.
# Takes the model, a DataLoader, epochs and learning rate; returns per-epoch mean losses.
def train_classifier(model, loader, epochs, lr):
    ...
