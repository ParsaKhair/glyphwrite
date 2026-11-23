# u(g): per-glyph quality score, from how typical a glyph's embedding is for its char.

import torch

# Weight of the typicality term in u(g)
LAMBDA = 0.1


# mu_c: the mean embedding E(g) of each char's glyphs. Takes a trained Classifier, the corpus and charset;
# returns a [n_classes, dim] tensor (NaN rows for chars with no glyphs).
def char_means(model, corpus, charset):
    ...


# u(g) = log P(c | g) - lambda ||E(g) - mu_c||^2: whether the classifier reads g as its label, and how typical g is of
# it. Takes a trained Classifier, [B, 1, SIZE, SIZE] glyphs, [B] label indices and char_means; returns [B] scores.
def quality(model, x, labels, means, lam=LAMBDA):
    ...
