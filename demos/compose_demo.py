import pickle
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from writing.compose import choose, place, render, spacing
from preprocess.ocr import segment_image

# Words in the corpus (compared with the real segment), then new text, then 1 with chars the corpus lacks
WORDS = ["entropy", "special", "change", "convolution", "back", "planets", "spin chains are simple", "quiz"]
SEEDS = (0, 1, 2)
PX = 48


# The 1st glyph-split segment that reads as word, rendered at PX per x-height, or None.
def real_segment(corpus, word):
    for seg in corpus.segments.values():
        if seg.split_ok and seg.text.strip(".,;:!?()'\"") == word:
            line = corpus.lines[seg.line_id]
            im = segment_image(line, seg)
            s = PX / line.xheight_px
            return im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return None


# Composes each of WORDS once per seed. Takes a corpus with segments; returns (gaps, rows) with
# rows = [(text, missing chars, real segment image or None, [composed image per seed])].
def compose_rows(corpus):
    gaps = spacing(corpus)
    rows = []
    for text in WORDS:
        placements = [place(corpus, choose(corpus, text, seed), gaps) for seed in SEEDS]
        missing = sorted({p.char for p in placements[0] if not p.glyph_id})
        rows.append((text, missing, real_segment(corpus, text), [render(corpus, p, PX) for p in placements]))
    return gaps, rows


# Compose stage of the demo run: saves the images to <out>/compose and returns the "compose" entry for results.json.
def run(corpus, out):
    folder = out / "compose"
    folder.mkdir(parents=True, exist_ok=True)
    gaps, rows = compose_rows(corpus)

    def save(im, name):
        im.save(folder / name)
        return {"image": f"compose/{name}", "image_px": list(im.size)}
    entries = [{"text": text, "missing": missing, "real": save(real, f"t{i}_real.png") if real else None,
                "composed": [dict(seed=seed, **save(im, f"t{i}_s{seed}.png")) for seed, im in zip(SEEDS, ims)]}
               for i, (text, missing, real, ims) in enumerate(rows)]
    return {"px_per_xh": PX, "glyphs": len(corpus.glyphs), "letter_gap": round(gaps[0], 3), "word_gap": round(gaps[1], 3), "texts": entries}


# 1 sheet image of compose_rows: label, real segment, then 1 column per seed.
def sheet(rows):
    label_w, gap = 330, 30
    cols = [max((r[2].width for r in rows if r[2]), default=100)] + [max(r[3][i].width for r in rows) for i in range(len(SEEDS))]
    heights = [max([r[2].height if r[2] else 0] + [im.height for im in r[3]]) for r in rows]
    im = Image.new("L", (label_w + sum(cols) + gap * len(cols), 50 + sum(heights) + gap * len(rows)), 255)
    d = ImageDraw.Draw(im)
    x = label_w
    for title, w in zip(["real segment"] + [f"compose seed {s}" for s in SEEDS], cols):
        d.text((x, 10), title, fill=120, font_size=22)
        x += w + gap
    y = 50
    for (text, missing, real, ims), h in zip(rows, heights):
        d.text((10, y + h // 2 - 12), text + (f"  (no {' '.join(missing)})" if missing else ""), fill=0, font_size=22)
        x = label_w
        for part, w in zip([real] + ims, cols):
            if part:
                im.paste(part, (x, y + (h - part.height) // 2))
            else:
                d.text((x, y + h // 2 - 12), "not in corpus", fill=160, font_size=20)
            x += w + gap
        d.line([(0, y + h + gap // 2), (im.width, y + h + gap // 2)], fill=220)
        y += h + gap
    return im


# The demo's corpus from all pages, cached in <out>/corpus.pkl so compose can be iterated on without rerunning OCR.
def load_corpus(out):
    path = out / "corpus.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    import segment_demo
    from preprocess.ocr import load_trocr
    recognize = load_trocr()
    pages = [(n, r, None) for n, r in segment_demo.REGIONS.items()] + [(n, r, {t: t for t in r}) for n, r in segment_demo.SAMPLES.items()]
    corpus = segment_demo.pool(segment_demo.build_corpus(n, r, recognize, t)[0] for n, r, t in pages)
    path.write_bytes(pickle.dumps(corpus))
    return corpus


# Standalone, for iterating on compose without the full demo: writes <out>/compose.png. Run on biglab:
#   python demos/compose_demo.py demos/out/compose
def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    gaps, rows = compose_rows(load_corpus(out))
    print(f"letter gap {gaps[0]:.2f}, word gap {gaps[1]:.2f} x-heights")
    sheet(rows).save(out / "compose.png")
    print("sheet:", out / "compose.png")


if __name__ == "__main__":
    main()
