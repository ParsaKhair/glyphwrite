# Scores glyph readability with a letter classifier and an optional embedding-distance penalty.

import torch
from torch import nn
from torch.nn import functional as F

from writing.data import SIZE, glyph_tensor

# A weight of 0.1 lowered ranking AUC on all 7 held-out corpus pages.
LAMBDA = 0.0
# Output channels of each conv block; each block halves the image side (32 -> 16 -> 8 -> 4)
CHANNELS = (32, 64, 128)
# Dropout on E(g) before the classifier layer, against overfitting the small corpus when fine-tuning
DROPOUT = 0.3


# 1 conv block: 3 x 3 conv, batch norm, ReLU, then 2 x 2 max pool. Takes input and output channel counts; returns an
# nn.Sequential mapping [B, in_ch, H, W] to [B, out_ch, H / 2, W / 2].
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        # Padding 1 keeps H and W, so only the pool shrinks the image
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


# E(g): layers 1 to d-1 of the classifier. 3 conv blocks turn the glyph into a [128, 4, 4] feature map, flattened so
# where a stroke sits is kept (a "b" and a "d" differ mostly in that); a linear layer then maps it to dim.
# Takes [B, 1, SIZE, SIZE]; returns [B, dim].
class GlyphEncoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

        # Conv blocks, from 1 input channel through CHANNELS
        chans = (1,) + CHANNELS
        self.convs = nn.Sequential(*[conv_block(a, b) for a, b in zip(chans, chans[1:])])

        # Flattened feature map to the embedding
        side = SIZE // 2 ** len(CHANNELS)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(CHANNELS[-1] * side * side, dim), nn.ReLU())

    def forward(self, x):
        return self.fc(self.convs(x))


# Letter classifier W E(g): the encoder's d-1 layers, then layer d, 1 linear layer from E(g) to class logits. Its
# rows are what grow_head widens. Takes an encoder and the number of classes; forward returns [B, n_classes] logits.
class Classifier(nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Linear(encoder.dim, n_classes)

    def forward(self, x):
        return self.head(self.dropout(self.encoder(x)))

    # E(g) alone, for char_means and quality. Takes [B, 1, SIZE, SIZE]; returns [B, dim].
    def embed(self, x):
        return self.encoder(x)


# Widens the head for chars the corpus has and EMNIST lacks (Greek, punctuation), keeping the trained rows.
# Takes a Classifier and the new class count with existing chars first. Call before creating the optimizer.
@torch.no_grad()
def grow_head(model, n_classes):
    old = model.head

    # Fails when expanding would remove existing classes.
    if n_classes < old.out_features:
        raise ValueError("the new head must include all existing classes")

    # Keeps the current head when the class count is equal.
    if n_classes == old.out_features:
        return

    # Initializes the new rows and copies the pretrained logits.
    head = nn.Linear(
        old.in_features, n_classes, device=old.weight.device, dtype=old.weight.dtype
    )
    head.weight[:old.out_features].copy_(old.weight)
    head.bias[:old.out_features].copy_(old.bias)
    head.weight.requires_grad_(old.weight.requires_grad)
    head.bias.requires_grad_(old.bias.requires_grad)
    model.head = head.train(old.training)


# Trains a Classifier with F.cross_entropy and Adam: on EMNIST to pretrain, then on glyph_dataset to fine-tune at a
# lower lr. Prints each epoch's loss. Takes the model, a DataLoader, epochs and learning rate; returns per-epoch mean
# losses.
def train_classifier(model, loader, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        # Training mode so dropout and batch norm update (char_means leaves the model in eval mode)
        model.train()
        total, n = 0.0, 0

        # 1 gradient step per batch
        for x, y in loader:
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)

        losses.append(total / n)
        print(f"epoch {epoch + 1}/{epochs}: loss {losses[-1]:.4f}", flush=True)
    return losses


# mu_c: the mean embedding E(g) of each char's glyphs, truth-labelled ones only unless ocr is True (as in
# data.glyph_dataset). Takes a trained Classifier, the corpus and charset (list of chars, index = class); returns a
# [n_classes, dim] tensor (NaN rows for chars with no glyphs).
def char_means(model, corpus, charset, ocr=False):
    index = {c: i for i, c in enumerate(charset)}
    sums = torch.zeros(len(charset), model.encoder.dim)
    counts = torch.zeros(len(charset))

    # Eval mode so batch norm uses its trained statistics
    model.eval()
    with torch.no_grad():
        for glyph in corpus.glyphs.values():
            # Skips when the char has no class, or the label came from OCR and ocr is False
            if glyph.char not in index or (glyph.label_source != "truth" and not ocr):
                continue

            # Adds the glyph's embedding to its char's sum
            c = index[glyph.char]
            sums[c] += model.embed(glyph_tensor(corpus, glyph.id)[None])[0]
            counts[c] += 1

    # Chars with no glyphs divide 0 by 0, giving their NaN rows
    return sums / counts[:, None]


# u(g) = log P(c | g) - lambda ||E(g) - mu_c||^2: whether the classifier reads g as its label, and how typical g is of
# it. Takes a trained Classifier, [B, 1, SIZE, SIZE] glyphs, [B] label indices, char_means and the charset.
# Returns [B] scores. A missing mean gives NaN when the distance penalty is enabled.
def quality(model, x, labels, means, charset, lam=LAMBDA):
    # Embeddings and class log probabilities, in eval mode so batch norm uses its trained statistics
    model.eval()
    with torch.no_grad():
        emb = model.embed(x)
        log_p = F.log_softmax(model.head(emb), 1)
    own = log_p.gather(1, labels[:, None])[:, 0]

    # Adds the other case's probability when it has a class.
    index = {c: i for i, c in enumerate(charset)}
    swap = labels.new_tensor([index.get(c.swapcase(), i) for i, c in enumerate(charset)])
    other = swap[labels]
    read = torch.where(other == labels, own, torch.logaddexp(own, log_p.gather(1, other[:, None])[:, 0]))

    # Returns readability alone when the distance penalty is disabled.
    if lam == 0:
        return read

    # Penalizes squared distance from the char's mean embedding.
    spread = ((emb - means[labels]) ** 2).sum(1)
    return read - lam * spread
