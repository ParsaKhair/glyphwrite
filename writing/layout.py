# Predicts the gap and vertical offset between neighbouring glyphs as a Gaussian for compose to sample.

import torch
from torch import nn


# Predicts the gap and vertical offset of b relative to a as a Gaussian, so compose can sample natural variation.
# Takes [B, dim] features of a and of b (e.g. quality.GlyphEncoder embeddings); returns (mu, log sigma), each [B, 2]
# for (gap, dy) in x-heights. Log sigma so sigma stays positive.
class LayoutModel(nn.Module):
    def __init__(self, dim=128, hidden=128):
        super().__init__()
        ...

    def forward(self, ea, eb):
        ...


# Gaussian NLL: L = (y - mu)^2 / (2 sigma^2) + log sigma, averaged. Takes [B, 2] targets, mu and log sigma;
# returns a scalar.
def gaussian_nll(y, mu, log_sigma):
    ...


# Trains a LayoutModel on data.layout_targets with gaussian_nll. Takes the model, a DataLoader of (a features,
# b features, [gap, dy]), epochs and learning rate; returns per-epoch mean losses.
def train_layout(model, loader, epochs, lr):
    ...


# Draws (gap, dy) from the predicted Gaussian, or its mean when temperature is 0. Takes mu, log sigma and a
# temperature scaling sigma; returns a [B, 2] tensor.
def sample(mu, log_sigma, temperature=1.0):
    ...
