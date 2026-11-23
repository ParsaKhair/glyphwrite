# p(a, b): contrastive model of how naturally glyph b follows glyph a, from their facing edges.

import torch
from torch import nn

# InfoNCE temperature
TAU = 0.1
# Independent per-side jitter ranges for augment: blur sigma in px, darkness gain, stroke width change in px
BLUR = (0.0, 1.0)
GAIN = (0.7, 1.3)
WIDTH = (-1, 1)


# p(a, b): high when piece b naturally follows piece a. f_R embeds a's right edge (how its stroke exits) as r, f_L
# embeds b's left edge (how its stroke enters) as l, and p = r^T l / tau, so every glyph's 2 embeddings are computed
# once and all pairs are scored at once. Takes [B, 1, H, W] right edges of a and left edges of b (data.edge_tensor);
# returns [B, B] scores, entry (i, j) = p(a_i, b_j).
class PairScorer(nn.Module):
    def __init__(self, dim=64, tau=TAU):
        super().__init__()
        ...

    def forward(self, a, b):
        ...


# InfoNCE, i.e. cross-entropy where the class is which candidate b_j is a_i's real neighbour:
# L_i = -log exp p(a_i, b_i) / sum_j exp p(a_i, b_j). Takes [B, B] scores with real neighbours on the diagonal;
# returns a scalar.
def infonce_loss(scores):
    ...


# Groups neighbour pairs into batches; the batch's other b's are each a_i's negatives, so grouping decides what p learns:
#   "random": any pairs; p partly learns which letters follow which ("q" -> "u"), not only whether they fit.
#   "char": b's share a letter; p learns only stylistic fit, matching compose, which picks among samples of a known letter.
#   "line": pairs share a line; p can't score "same photo" instead of "fits well".
# Takes the corpus, data.neighbour_pairs, mode, batch size and seed; returns [[(a id, b id)]]. Groups smaller than 2
# are dropped since they have no negatives.
def pair_batches(corpus, pairs, mode="char", batch_size=64, seed=None):
    ...


# Jitters blur, darkness and stroke width within BLUR, GAIN and WIDTH, drawn separately per image, so p can't match
# the 2 sides by photo quality. Apply to a's and b's edges independently. Takes [B, 1, H, W]; returns the same shape.
def augment(x):
    ...


# Trains a PairScorer with infonce_loss on pair_batches, augmenting each side independently. Takes the model, corpus,
# neighbour pairs, negatives mode, epochs and learning rate; returns per-epoch mean losses.
def train_pairs(model, corpus, pairs, mode, epochs, lr):
    ...
