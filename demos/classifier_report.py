# Writeup of the pretrained letter classifier and the quality score u(g) (writing/quality.py): training data, a PCA of
# E(g), accuracy, and how well u(g) separates hand-curated good glyphs from bad. Not part of the automatic demo.
# 2 steps, like run_demo.sh:
#   python demos/classifier_report.py compute demos/out/classifier   (biglab: needs EMNIST, the model, TrOCR)
#   python3 demos/classifier_report.py report demos/out/classifier   (locally: needs matplotlib and LaTeX)
# compute writes data.npz (and caches corpus.pkl); report writes figures/ and classifier.tex, then compiles it.

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PRETRAIN = Path("demos/out/pretrain")
# Hand labels, glyph id -> {"char", "verdict": "good" | "bad" | "unsure"}, judged by eye from the 32 x 32 images
LABELS = Path("demos/data/glyph_labels.json")
EMNIST_ROOT = "demos/data/emnist"
# Chars shown in the data grid and the PCA: the corpus's most frequent lowercase letters
N_CHARS = 8
# Samples per char in the data grid, and EMNIST test points per char in the PCA
N_GRID = 8
N_PCA = 300
# Typicality weights tried for u(g)
LAMBDAS = (0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1)
SEED = 0


# ----- compute (biglab) -----

# Class probabilities and E(g) for a stack of images. Takes a Classifier and [N, 1, SIZE, SIZE]; returns numpy
# ([N, n_classes], [N, dim]).
def run_model(model, x):
    import torch
    probs, embs = [], []
    with torch.no_grad():
        for batch in torch.split(x, 512):
            emb = model.embed(batch)
            probs.append(torch.softmax(model.head(emb), 1))
            embs.append(emb)
    return torch.cat(probs).numpy(), torch.cat(embs).numpy()


# Share of EMNIST test images read right, exactly and ignoring case. Takes a Classifier, the test set and charset;
# returns (exact, folded).
def emnist_accuracy(model, test, charset):
    import torch
    from torch.utils.data import DataLoader
    pred, true = [], []
    with torch.no_grad():
        for x, y in DataLoader(test, batch_size=512, num_workers=6):
            pred.append(model(x).argmax(1))
            true.append(y)
    pred, true = torch.cat(pred).numpy(), torch.cat(true).numpy()
    lower = np.array([c.lower() for c in charset])
    return float((pred == true).mean()), float((lower[pred] == lower[true]).mean())


# Runs the classifier and u(g) on every corpus glyph and on EMNIST samples, and saves what the report needs to
# <out>/data.npz.
def compute(out):
    import torch
    from compose_demo import load_corpus
    from corpus import Corpus
    from writing.data import emnist, glyph_tensor
    from writing.quality import Classifier, GlyphEncoder, char_means, quality

    rng = np.random.default_rng(SEED)

    # Pretrained classifier and its charset
    saved = torch.load(PRETRAIN / "classifier.pt")
    charset = saved["charset"]
    model = Classifier(GlyphEncoder(), len(charset))
    model.load_state_dict(saved["state_dict"])
    model.eval()

    # Corpus glyphs whose char the classifier knows, truth-labelled only, with their hand verdicts
    corpus = load_corpus(out)
    labels = json.loads(LABELS.read_text())
    glyphs = [g for g in sorted(corpus.glyphs.values(), key=lambda g: g.id)
              if g.char in charset and g.label_source == "truth"]
    verdicts = np.array([labels.get(g.id, {}).get("verdict", "unsure") for g in glyphs])
    x = torch.stack([glyph_tensor(corpus, g.id) for g in glyphs])
    y = torch.tensor([charset.index(g.char) for g in glyphs])
    probs, embs = run_model(model, x)

    # Random half of the glyphs to measure u(g) on; the other half's good glyphs give the "curated" means
    test_half = rng.random(len(glyphs)) < 0.5
    curated = Corpus(lines=corpus.lines, glyphs={g.id: g for g, v, t in zip(glyphs, verdicts, test_half)
                                                 if v == "good" and not t})

    # Means as char_means gives them (all truth glyphs), and from curated good glyphs, falling back to the former
    # for chars with no curated glyph
    means_all = char_means(model, corpus, charset)
    means_cur = char_means(model, curated, charset)
    means_cur = torch.where(torch.isnan(means_cur), means_all, means_cur)

    # u(g) for every glyph at each lambda, with each set of means
    u_all = np.stack([quality(model, x, y, means_all, charset, lam).numpy() for lam in LAMBDAS])
    u_cur = np.stack([quality(model, x, y, means_cur, charset, lam).numpy() for lam in LAMBDAS])

    # Chars to show: the most frequent lowercase letters among good glyphs
    good = [g.char for g, v in zip(glyphs, verdicts) if v == "good" and g.char.islower()]
    chars = [c for c, _ in sorted(((c, good.count(c)) for c in set(good)), key=lambda t: (-t[1], t[0]))][:N_CHARS]

    # EMNIST accuracy, train samples of the shown chars for the data grid, and test embeddings for the PCA
    train, test = emnist(EMNIST_ROOT, True), emnist(EMNIST_ROOT, False)
    emnist_acc = emnist_accuracy(model, test, charset)
    grid, pca_x, pca_y = [], [], []
    for c in chars:
        k = charset.index(c)
        train_idx = np.flatnonzero(train.targets.numpy() == k)
        grid.append(torch.stack([train[i][0] for i in rng.choice(train_idx, N_GRID, replace=False)]))
        test_idx = np.flatnonzero(test.targets.numpy() == k)
        pick = rng.choice(test_idx, min(N_PCA, len(test_idx)), replace=False)
        pca_x.append(torch.stack([test[i][0] for i in pick]))
        pca_y += [k] * len(pick)
    _, test_embs = run_model(model, torch.cat(pca_x))

    np.savez_compressed(
        out / "data.npz", charset=np.array(charset), chars=np.array(chars), lambdas=np.array(LAMBDAS),
        emnist_acc=np.array(emnist_acc), emnist_grid=torch.stack(grid).numpy()[:, :, 0],
        test_embs=test_embs, test_labels=np.array(pca_y),
        glyph_ids=np.array([g.id for g in glyphs]), glyph_imgs=x.numpy()[:, 0],
        glyph_chars=np.array([g.char for g in glyphs]), glyph_verdicts=verdicts, glyph_probs=probs,
        glyph_embs=embs, test_half=test_half, u_all=u_all, u_cur=u_cur)
    print(f"{len(glyphs)} corpus glyphs, chars {''.join(chars)}, EMNIST acc {emnist_acc} -> {out / 'data.npz'}")


# ----- report (local) -----

# AUC: chance a random good glyph scores above a random bad one (ties count half). Takes 2 score arrays; returns a
# float.
def auc(good, bad):
    return float(((good[:, None] > bad[None]) + 0.5 * (good[:, None] == bad[None])).mean())


# Grid of glyph images with an optional caption and caption colour under each. Takes a matplotlib axes, [N, H, W]
# images, columns, and lists of captions and colours; draws in place.
def image_grid(ax, imgs, cols, captions=None, colours=None):
    rows = int(np.ceil(len(imgs) / cols))
    h = imgs.shape[1]
    # Space between rows, larger when a caption goes under each image
    gap = 12 if captions is not None else 2
    sheet = np.ones((rows * (h + gap), cols * (h + 2)))
    for n, im in enumerate(imgs):
        r, c = divmod(n, cols)
        sheet[r * (h + gap):r * (h + gap) + h, c * (h + 2):c * (h + 2) + h] = 1 - im
        # Caption under the image
        if captions is not None:
            ax.text(c * (h + 2) + h / 2, r * (h + gap) + h + 1, captions[n], ha="center", va="top", fontsize=7,
                    color=colours[n] if colours else "black")
    ax.imshow(sheet, cmap="gray", vmin=0, vmax=1)
    ax.set_axis_off()


# Saves a figure as a PDF in figs and closes it. Takes the figure, the folder and a name; returns nothing.
def save(fig, figs, name):
    fig.tight_layout()
    fig.savefig(figs / f"{name}.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)


# Builds the figures from <out>/data.npz, writes classifier.tex and compiles it.
def report(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False})
    GOOD, BAD = "#1a7f37", "#c62828"

    d = np.load(out / "data.npz")
    charset, chars, lams = list(d["charset"]), list(d["chars"]), d["lambdas"]
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    colours = plt.get_cmap("tab10").colors
    pre = json.loads((PRETRAIN / "results.json").read_text())
    labels, verdicts, imgs = d["glyph_chars"], d["glyph_verdicts"], d["glyph_imgs"]
    good, bad = verdicts == "good", verdicts == "bad"
    from writing.quality import LAMBDA

    # Fig 1a: EMNIST next to curated corpus glyphs of the same chars, framed identically
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
    image_grid(axes[0], d["emnist_grid"].reshape(-1, 32, 32), N_GRID)
    corpus_grid = np.zeros_like(d["emnist_grid"])
    for k, c in enumerate(chars):
        mine = imgs[good & (labels == c)]
        corpus_grid[k, :min(N_GRID, len(mine))] = mine[rng.permutation(len(mine))[:N_GRID]]
    image_grid(axes[1], corpus_grid.reshape(-1, 32, 32), N_GRID)
    for ax, title in zip(axes, ["EMNIST (pretraining)", "Our glyphs (curated good)"]):
        ax.set_title(title)
        # Row labels: the char each row shows
        for k, c in enumerate(chars):
            ax.text(-3, k * 34 + 16, c, ha="right", va="center", fontsize=10)
    save(fig, figs, "data")

    # Fig 1b: glyphs per char in the curated set, lowercase letters only
    lower = sorted({c for c in labels if c.islower()})
    counts = [((labels == c) & good).sum() for c in lower]
    bad_counts = [((labels == c) & bad).sum() for c in lower]
    fig, ax = plt.subplots(figsize=(6.5, 1.8))
    ax.bar(lower, counts, color=GOOD, label="good")
    ax.bar(lower, bad_counts, bottom=counts, color=BAD, alpha=0.35, label="bad")
    ax.set_ylabel("glyphs")
    ax.legend(frameon=False, ncol=2)
    save(fig, figs, "counts")

    # Fig 2: PCA of E(g), fitted on EMNIST test embeddings, with curated good corpus glyphs projected into it
    test, test_y = d["test_embs"], d["test_labels"]
    mean = test.mean(0)
    _, sv, vt = np.linalg.svd(test - mean, full_matrices=False)
    explained = (sv[:2] ** 2).sum() / (sv ** 2).sum()
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for k, c in enumerate(chars):
        p = (test[test_y == charset.index(c)] - mean) @ vt[:2].T
        ax.scatter(p[:, 0], p[:, 1], s=3, alpha=0.25, color=colours[k], linewidths=0)
        q = (d["glyph_embs"][good & (labels == c)] - mean) @ vt[:2].T
        ax.scatter(q[:, 0], q[:, 1], s=22, color=colours[k], edgecolors="black", linewidths=0.6, label=c)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(ncol=2, fontsize=7, frameon=False, markerscale=0.8, handletextpad=0.1)
    save(fig, figs, "pca")

    # Fig 3: accuracy, exactly and ignoring case, on EMNIST test and on our glyphs
    pred = np.array([charset[i] for i in d["glyph_probs"].argmax(1)])
    sets = {"EMNIST test": None, "ours: all": np.ones(len(labels), bool), "ours: good": good, "ours: bad": bad}
    exact, folded = [], []
    for name, m in sets.items():
        if m is None:
            exact.append(float(d["emnist_acc"][0]))
            folded.append(float(d["emnist_acc"][1]))
        else:
            exact.append((pred[m] == labels[m]).mean())
            folded.append((np.char.lower(pred[m]) == np.char.lower(labels[m])).mean())
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    pos = np.arange(len(sets))
    ax.bar(pos - 0.2, exact, 0.4, color="0.65", label="exact")
    ax.bar(pos + 0.2, folded, 0.4, color="0.25", label="ignoring case")
    for p, e, f in zip(pos, exact, folded):
        ax.text(p - 0.2, e + 0.01, f"{100 * e:.0f}", ha="center", va="bottom", fontsize=6)
        ax.text(p + 0.2, f + 0.01, f"{100 * f:.0f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(pos, [f"{n}\n(n={m.sum() if m is not None else 116323})" for n, m in sets.items()])
    ax.set_ylabel("top-1 accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.legend(frameon=False, loc="upper right", ncol=2)
    save(fig, figs, "accuracy")

    # Fig 4a: u(g) at lambda = 0 for good and bad glyphs
    u0 = d["u_all"][list(lams).index(0)]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))
    bins = np.linspace(-12, 0, 25)
    axes[0].hist(np.clip(u0[good], -12, 0), bins, color=GOOD, alpha=0.6, label=f"good (n={good.sum()})")
    axes[0].hist(np.clip(u0[bad], -12, 0), bins, color=BAD, alpha=0.6, label=f"bad (n={bad.sum()})")
    axes[0].set_xlabel(r"$u(g)$ at $\lambda=0$, i.e. $\log P(c \mid g)$")
    axes[0].set_ylabel("glyphs")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_title(f"AUC {auc(u0[good], u0[bad]):.2f}", fontsize=8)

    # Fig 4b: AUC on the held-out half against lambda, for both sets of means
    t = d["test_half"]
    for key, name, style in (("u_all", "means of all glyphs", "-o"), ("u_cur", "means of curated good", "--s")):
        aucs = [auc(u[t & good], u[t & bad]) for u in d[key]]
        axes[1].plot(range(len(lams)), aucs, style, color="0.2", ms=3.5, label=name)
    axes[1].axvline(list(lams).index(LAMBDA), color="0.6", lw=0.8, ls=":")
    axes[1].text(list(lams).index(LAMBDA), 0.52, rf" current $\lambda$", fontsize=7, color="0.4")
    axes[1].set_xticks(range(len(lams)), [f"{l:g}" for l in lams], rotation=40)
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel("AUC (good vs bad)")
    axes[1].set_ylim(0.5, 1)
    axes[1].legend(frameon=False, loc="lower left")
    save(fig, figs, "score")

    # Fig 5: the 12 lowest and 12 highest u(g) glyphs, captioned by label and coloured by hand verdict
    order = np.argsort(u0)
    rows = np.concatenate([order[:12], order[-12:][::-1]])
    tint = {"good": GOOD, "bad": BAD, "unsure": "0.5"}
    fig, ax = plt.subplots(figsize=(6.5, 1.75))
    image_grid(ax, imgs[rows], 12, [f"{labels[i]}  {u0[i]:.1f}" for i in rows], [tint[verdicts[i]] for i in rows])
    ax.text(-4, 16, "lowest", ha="right", va="center", rotation=90, fontsize=7)
    ax.text(-4, 16 + 44, "highest", ha="right", va="center", rotation=90, fontsize=7)
    save(fig, figs, "ranked")

    # Tex and PDF
    stats = {"n": len(labels), "n_good": int(good.sum()), "n_bad": int(bad.sum()), "explained": explained,
             "lam": LAMBDA}
    (out / "classifier.tex").write_text(tex(stats, pre))
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "classifier.tex"], cwd=out,
                   check=True, stdout=subprocess.DEVNULL)
    print("report:", out / "classifier.pdf")


# The writeup's LaTeX source. Takes the stats dict and the pretraining results; returns a string.
def tex(s, pre):
    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.9in]{{geometry}}
\usepackage{{graphicx,amsmath,amssymb}}
\usepackage[font=small,labelfont=bf]{{caption}}
\pagestyle{{empty}}
\begin{{document}}
\begin{{center}}{{\Large Letter classifier and glyph quality $u(g)$}}\end{{center}}

\noindent\textbf{{Setup.}} A CNN (3 conv blocks, then a linear layer) maps a 32$\times$32 glyph to $E(g) \in
\mathbb{{R}}^{{128}}$, and a linear layer on $E(g)$ gives 62 class logits (digits, A--Z, a--z). It is trained on EMNIST
only ({pre['epochs']} epochs, {pre['train_minutes']:.0f} min on CPU); none of our glyphs were used in training. Quality
score: $u(g) = \log P(c \mid g) - \lambda \lVert E(g) - \mu_c \rVert^2$, where $c$ is the glyph's label, $P(c \mid g)$
counts both cases of $c$, and $\mu_c$ is the mean $E(g)$ of $c$'s glyphs. Our {s['n']} glyphs were sorted by eye into
{s['n_good']} \emph{{good}} (clearly the labelled letter, whole and alone) and {s['n_bad']} \emph{{bad}} (bad cuts,
fragments, merged letters, smudges).

\begin{{figure}}[h]\centering
\includegraphics[width=\linewidth]{{figures/data.pdf}}\\[2pt]
\includegraphics[width=\linewidth]{{figures/counts.pdf}}
\caption{{Top: training images and our glyphs, cropped and scaled the same way. Bottom: our glyphs per lowercase letter;
the green part is the curated fine-tuning set.}}
\end{{figure}}

\begin{{figure}}[h]\centering
\begin{{minipage}}[t]{{0.49\linewidth}}\centering
\includegraphics[width=\linewidth]{{figures/pca.pdf}}
\caption{{$E(g)$ projected on its first 2 principal components ({100 * s['explained']:.0f}\% of variance), fitted on
EMNIST test images (faint). Outlined: our good glyphs.}}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.49\linewidth}}\centering
\includegraphics[width=\linewidth]{{figures/accuracy.pdf}}
\caption{{Share of glyphs whose most likely class is the label, exactly or ignoring case.}}
\end{{minipage}}
\end{{figure}}

\begin{{figure}}[h]\centering
\includegraphics[width=\linewidth]{{figures/score.pdf}}
\caption{{Left: $u(g)$ at $\lambda=0$ (scores below $-12$ are in the leftmost bin). Right: AUC, the chance a random good
glyph scores above a random bad one (0.5 = chance, 1 = perfect), measured on a random half of the glyphs. $\mu_c$ comes
from all glyphs (as \texttt{{char\_means}} does now) or from curated good glyphs of the other half. Current
$\lambda = {s['lam']}$.}}
\end{{figure}}

\begin{{figure}}[h]\centering
\includegraphics[width=\linewidth]{{figures/ranked.pdf}}
\caption{{Our 12 lowest and 12 highest glyphs by $u(g)$ at $\lambda=0$, captioned with label and score; colour is the
hand verdict (green good, red bad, grey unsure).}}
\end{{figure}}

\end{{document}}
"""


if __name__ == "__main__":
    step, folder = sys.argv[1], Path(sys.argv[2])
    folder.mkdir(parents=True, exist_ok=True)
    compute(folder) if step == "compute" else report(folder)
