import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Stubs tkinter when missing (some biglab hosts lack it); the demo lassos in code, never in a window
try:
    import tkinter  # noqa: F401
except ImportError:
    import types
    sys.modules["tkinter"] = types.ModuleType("tkinter")
import compose_demo
from samples import SAMPLES
from preprocess import glyphs
from preprocess import words
from corpus import Corpus
from preprocess.lines import extract_lines
from main import load_images
from preprocess.ocr import apply_truth, line_image, read_lines, segment_image

DATA = Path(__file__).parent / "data"
# Pipeline stages in order, with whether this demo runs them yet; update as stages get built
STAGES = {"Lasso": True, "Lines": True, "OCR": True, "Words": True, "Glyphs": True, "Compose": True}
# Labels lines with TRUTH instead of OCR; OCR still runs, for alignment and the CER
USE_TRUTH = True
# Longest side of any image placed in the figure, to keep the PDF small
FIG_PX = 1600

# Lasso polygons per sample in full-resolution page pixels, drawn around prose only (math left out)
REGIONS = {
    "IMG_7517": {
        "Critical Points": [(1080, 1950), (1760, 1950), (1760, 2120), (1080, 2120)],
        "trivial, equal mixing": [(1900, 2150), (3260, 2150), (3260, 2330), (1900, 2330)],
        "look at special case": [(2120, 2440), (3080, 2440), (3080, 2700), (2120, 2700)],
        "Phase separation": [(1100, 2760), (2030, 2760), (2030, 2930), (1100, 2930)],
        "For sufficient": [(1100, 3000), (1900, 3000), (1900, 3180), (1100, 3180)],
        "interaction": [(2270, 3020), (2900, 3020), (2900, 3170), (2270, 3170)],
        "entropy changes extrema": [(1400, 3180), (2900, 3180), (2900, 3360), (1400, 3360)],
        "away from": [(1950, 3370), (2620, 3370), (2620, 3560), (1950, 3560)],
        "entropy can make / them point in / same direction": [
            (1650, 3600), (2780, 3600), (2780, 4010), (1650, 4010)],
        "Suppose": [(780, 5020), (1260, 5020), (1260, 5150), (780, 5150)],
        "is big": [(2550, 5420), (2950, 5420), (2950, 5570), (2550, 5570)],
    },
    "IMG_7518": {
        "note lambdas cancel": [(3000, 820), (4270, 820), (4270, 960), (3000, 960)],
        "1 particle irreducible": [(2770, 2700), (3600, 2700), (3600, 2900), (2770, 2900)],
        "the change of variables cancels the": [
            (2680, 3250), (3800, 3250), (3800, 3540), (3080, 3540), (3080, 3680), (2680, 3680)],
        "Recall hard spheres": [(30, 4100), (1300, 4100), (1300, 4260), (30, 4260)],
        "No 3way intersection": [(880, 4440), (1980, 4440), (1980, 4580), (880, 4580)],
        "Gets us back to a convolution": [(600, 5310), (3900, 5310), (3900, 5570), (600, 5570)],
    },
}


# What each region actually says, transcribed by hand; lines of 1 region joined by spaces
TRUTH = {
    "IMG_7517": {
        "Critical Points": "Critical Points:",
        "trivial, equal mixing": "(trivial, equal mixing)",
        "look at special case": "look at special case",
        "Phase separation": "Phase separation!",
        "For sufficient": "For sufficient",
        "interaction": "interaction",
        "entropy changes extrema": "entropy changes extrema",
        "away from": "away from",
        "entropy can make / them point in / same direction": "entropy can make them point in same direction!",
        "Suppose": "Suppose",
        "is big": "is big",
    },
    "IMG_7518": {
        "note lambdas cancel": "note lambdas cancel,",
        "1 particle irreducible": "(1 particle irreducible)",
        "the change of variables cancels the": "the change of variables cancels the",
        "Recall hard spheres": "Recall hardspheres!",
        "No 3way intersection": "No 3way intersection",
        "Gets us back to a convolution": "Gets us back to a convolution",
    },
}

# Cuts a polygon out of a page the same way select_text_region does, without the window.
# Takes a page image and vertices in page pixels; returns (RGBA crop, (left, top)).
def crop_polygon(image, vertices):
    # Crops to the polygon's bounding box
    left, top = min(x for x, _ in vertices), min(y for _, y in vertices)
    right, bottom = max(x for x, _ in vertices), max(y for _, y in vertices)
    crop = image.crop((left, top, right, bottom)).convert("RGBA")

    # Clears everything outside the polygon to transparent white
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).polygon([(x - left, y - top) for x, y in vertices], fill=255)
    crop = Image.composite(crop, Image.new("RGBA", crop.size, (255, 255, 255, 0)), mask)
    return crop, (left, top)


# Renders 1 glyph's ink dark on white with a little padding. Takes its line + glyph, returns a grayscale PIL image.
def glyph_image(line, glyph, pad=0.15):
    alpha, _ = glyphs.glyph_alpha(line, glyph)
    ink = np.pad(alpha, int(pad * line.xheight_px))
    return Image.fromarray((255 * (1 - ink)).astype(np.uint8))


# Line image with its measured baseline (red) and x-height line (blue) drawn on.
# Takes a Line, returns an RGB PIL image.
def line_guides(line, pad=0.3):
    im = line_image(line, pad).convert("RGB")
    p = int(pad * line.xheight_px)
    d = ImageDraw.Draw(im)
    width = max(2, round(line.xheight_px / 15))
    d.line([(0, p + line.base_row), (im.width, p + line.base_row)], fill="red", width=width)
    d.line([(0, p + line.base_row - line.xheight_px), (im.width, p + line.base_row - line.xheight_px)],
           fill="blue", width=width)
    return im


# Page thumbnail with every lasso (green) and every line's baseline mapped back to the page (red).
# Takes the page, its lasso polygons and lines per region; returns an RGB PIL image.
def page_overlay(page, polygons, region_lines):
    im = page.convert("RGB")
    d = ImageDraw.Draw(im)
    for k, (vertices, lines) in enumerate(zip(polygons, region_lines)):
        d.polygon(vertices, outline=(0, 170, 0), width=10)
        d.text((vertices[0][0], vertices[0][1] - 90), str(k), fill=(0, 140, 0), font_size=80)
        # Draws each line's baseline where it sits on the page
        for line in lines:
            d.line([tuple(pt) for pt in line.baseline_poly], fill="red", width=8)
    im.thumbnail((FIG_PX, FIG_PX))
    return im


# Saves an image shrunk to fit the figure. Takes an image and a path, returns the path's name.
def save_small(im, path):
    im = im.copy()
    im.thumbnail((FIG_PX, FIG_PX))
    im.save(path)
    return path.name


# Describes 1 word for the figure: its image, text, OCR confidence and glyph split, with glyph images.
# Takes the corpus, the segment and the page's output folder; returns a dict for results.json.
def word_entry(corpus, seg, folder):
    line = corpus.lines[seg.line_id]
    im = segment_image(line, seg)
    entry = {"text": seg.text, "source": seg.label_source, "conf": round(seg.ocr_conf, 3), "image": f"{folder.name}/" + save_small(im, folder / f"{seg.id}.png"),
             "image_px": list(im.size), "split_ok": seg.split_ok, "n_natural": seg.n_natural,
             "split_score": round(seg.split_score, 3), "glyphs": []}
    # Adds 1 tile per glyph
    for glyph in (corpus.glyphs[g] for g in seg.glyph_ids):
        gim = glyph_image(line, glyph)
        entry["glyphs"].append({"char": glyph.char, "score": round(glyph.split_score, 3), "dy": round(glyph.dy, 3), "image_px": list(gim.size),
                                "image": f"{folder.name}/" + save_small(gim, folder / f"{glyph.id}.png")})
    return entry


# Runs 1 sample page through lasso -> lines -> line OCR -> words -> glyphs.
# Takes the page name, its lasso polygons, the OCR function and the true text per region (TRUTH[name] by default);
# returns (corpus, page image, [(crop, lines)] per region).
def build_corpus(name, regions, recognize, truth=None):
    page = load_images([DATA / f"{name}.jpeg"])[0]

    # Lassos each region and extracts its lines into a corpus
    corpus, region_lines = Corpus(), []
    for k, vertices in enumerate(regions.values()):
        crop, offset = crop_polygon(page, vertices)
        lines = extract_lines(crop, name, k, offset)
        corpus.lines.update((line.id, line) for line in lines)
        region_lines.append((crop, lines))

    # Runs the rest of the pipeline on the whole page
    read_lines(corpus, recognize)
    if USE_TRUTH:
        for label, (_, lines) in zip(regions, region_lines):
            apply_truth(lines, (truth or TRUTH[name])[label])
    words.split_corpus(corpus)
    glyphs.split_corpus(corpus)
    # The 1st split's letters show where each line's baseline and x-height really are; split again in that frame
    glyphs.refit_frames(corpus)
    words.split_corpus(corpus)
    glyphs.split_corpus(corpus)
    return corpus, page, region_lines


# Merges corpora into 1 (IDs carry the page name, so they don't clash).
def pool(corpora):
    out = Corpus()
    for corpus in corpora:
        out.lines.update(corpus.lines)
        out.segments.update(corpus.segments)
        out.glyphs.update(corpus.glyphs)
    return out


# Runs the pipeline on every sample, then composes test texts from all pages' glyphs.
# Takes the OCR function and a run folder; writes images and results.json there, returns the results.
def run(recognize, out):
    out.mkdir(parents=True, exist_ok=True)
    results, corpora = [], []
    for name, regions in REGIONS.items():
        corpus, page, region_lines = build_corpus(name, regions, recognize)
        corpora.append(corpus)
        folder = out / name
        folder.mkdir(exist_ok=True)

        # Saves the figure images and records what each stage produced
        entry = {"page": name, "overlay": f"{name}/" + save_small(
            page_overlay(page, list(regions.values()), [l for _, l in region_lines]), folder / "overlay.png"),
            "regions": []}
        for k, (label, (crop, lines)) in enumerate(zip(regions, region_lines)):
            flat = Image.alpha_composite(Image.new("RGBA", crop.size, "white"), crop).convert("RGB")
            region = {"label": label, "truth": TRUTH[name][label],
                      "crop": f"{name}/" + save_small(flat, folder / f"r{k}.png"), "lines": []}
            for line in lines:
                guides = line_guides(line)
                region["lines"].append({
                    "id": line.id, "image": f"{name}/" + save_small(guides, folder / f"{line.id}.png"),
                    "image_px": list(guides.size), "text": line.text, "ocr_text": line.ocr_text,
                    "source": line.label_source, "conf": round(line.ocr_conf, 3),
                    "xheight_px": line.xheight_px, "slant_deg": round(float(np.degrees(np.arctan(line.shear))), 1),
                    "blobs": int(line.labels.max()), "size": list(line.density.shape),
                    "clusters": len(words.clusters(line)),
                    "words": [word_entry(corpus, corpus.segments[i], folder) for i in line.segment_ids]})
            entry["regions"].append(region)
        results.append(entry)

    # Text-sample pages (keyed by their true text) feed compose only; the report doesn't show them
    corpora += [build_corpus(name, regions, recognize, {t: t for t in regions})[0] for name, regions in SAMPLES.items()]
    compose = compose_demo.run(pool(corpora), out)
    results = {"stages": STAGES, "pages": results, "compose": compose}
    (out / "results.json").write_text(json.dumps(results, indent=1))
    return results


# Runs the demo with TrOCR. Takes the run folder as the 1st command-line argument.
def main():
    from preprocess.ocr import load_trocr
    results = run(load_trocr(), Path(sys.argv[1]))
    # Prints 1 summary row per line, then the compose test's missing chars
    for page in results["pages"]:
        for region in page["regions"]:
            for line in region["lines"]:
                split = " ".join(f'{w["text"]}:{"ok" if w["split_ok"] else "FAIL"}' for w in line["words"]) or "no words"
                print(f'{line["id"]:18} conf {line["conf"]:.2f}  "{line["text"]}"  ->  {split}')
    for entry in results["compose"]["texts"]:
        print(f'compose "{entry["text"]}"  missing: {" ".join(entry["missing"]) or "none"}')


if __name__ == "__main__":
    main()
