# Pretrains the letter classifier (writing/quality.py) on EMNIST and reports test accuracy.
# Run on biglab from the repo root: python demos/pretrain_demo.py demos/out/pretrain [epochs]
# Reads EMNIST from demos/data/emnist (downloads it there if missing). Writes to the output folder: classifier.pt
# (weights and charset, for fine-tuning and scoring) and results.json (per-epoch losses, test accuracy).

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from writing.data import EMNIST_CHARS, emnist
from writing.quality import Classifier, GlyphEncoder, train_classifier

EMNIST_ROOT = "demos/data/emnist"
EPOCHS = 3
LR = 1e-3
BATCH = 256
# DataLoader worker processes; biglab hosts have 8 cores
WORKERS = 6


# Share of test images the model labels correctly. Takes a Classifier and a DataLoader; returns a float in [0, 1].
def accuracy(model, loader):
    model.eval()
    right, n = 0, 0
    with torch.no_grad():
        for x, y in loader:
            right += (model(x).argmax(1) == y).sum().item()
            n += len(y)
    return right / n


# Trains for the given epochs, measures test accuracy and saves the model and results. Takes the output folder and
# epoch count; returns nothing.
def main(out, epochs=EPOCHS):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    # EMNIST train and test splits, framed like corpus glyphs
    train = DataLoader(emnist(EMNIST_ROOT, train=True), batch_size=BATCH, shuffle=True, num_workers=WORKERS)
    test = DataLoader(emnist(EMNIST_ROOT, train=False), batch_size=BATCH, num_workers=WORKERS)

    # Trains a fresh classifier with 1 class per EMNIST char
    model = Classifier(GlyphEncoder(), len(EMNIST_CHARS))
    start = time.time()
    losses = train_classifier(model, train, epochs, LR)
    minutes = (time.time() - start) / 60

    # Scores it on the test split
    acc = accuracy(model, test)
    print(f"test accuracy {acc:.4f}, {minutes:.1f} min training")

    # Saves weights with the charset so later stages know which row is which char
    torch.save({"state_dict": model.state_dict(), "charset": list(EMNIST_CHARS)}, out / "classifier.pt")
    results = {"epochs": epochs, "lr": LR, "batch": BATCH, "losses": losses, "test_accuracy": acc,
               "train_minutes": minutes}
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:3]))
