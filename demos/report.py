import json
import sys
from pathlib import Path

RUNS = Path(__file__).parent / "out" / "runs"
HISTORY = Path(__file__).parent / "history.md"
# Line images share 1 scale (cm per source pixel) so their sizes compare; widths are capped to fit the page
LINE_CM_PER_PX = 0.0065
LINE_MAX_CM = 10.5
# Words and glyphs are drawn larger than lines, shrunk per line only when the row would overflow
WORD_CM_PER_PX = 0.011
ROW_MAX_CM = 16.0
# Blank between word cells, and between glyph tiles, in cm
WORD_SEP_CM = 0.35
GLYPH_SEP_CM = 0.08
# Compose images share 1 scale; at the demo's 48 px per x-height this draws an x-height 0.6 cm tall
COMPOSE_CM_PER_PX = 0.0125

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[margin=1.5cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{graphicx,booktabs,xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,chains}
\tikzset{
  stage/.style={draw, rounded corners, fill=green!15, minimum height=9mm, minimum width=18mm, align=center, font=\small},
  todo/.style={stage, fill=gray!8, dashed, text=gray},
  flow/.style={-{Stealth}, thick},
  img/.style={draw=gray!50, inner sep=1pt},
}
\setlength{\parindent}{0pt}
\setlength{\fboxsep}{1pt}
\begin{document}
"""


# Makes text safe for LaTeX: escapes special characters and replaces non-ASCII with '?'.
def tex(s):
    special = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
               "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(special.get(c, c if c.isascii() else "?") for c in s)


# Edit distance between 2 strings: fewest insertions, deletions and substitutions turning a into b.
def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


# What the pipeline read for a region (all its lines) and its edit distance to the truth, spaces ignored.
# Takes a region from results.json; returns (read text, errors, truth length).
def region_errors(region):
    read = " ".join(line.get("ocr_text", line["text"]) for line in region["lines"])
    squash = lambda t: "".join(t.split())
    return read, edit_distance(squash(read), squash(region["truth"])), len(squash(region["truth"]))


# Flow chart of the pipeline stages: green = ran in this demo, dashed gray = not built yet.
# Takes the stages dict from results.json, returns TikZ code.
def flowchart(stages):
    nodes = " ".join(
        rf"\node[{'stage' if ran else 'todo'}, on chain, join] {{{name}\\[-1pt]\tiny {'ran' if ran else 'not built yet'}}};"
        for name, ran in stages.items())
    return (r"\begin{center}\begin{tikzpicture}[start chain=going right, node distance=6mm, every join/.style=flow]"
            + nodes + r"\end{tikzpicture}\end{center}" + "\n")


# 1 line's words as a row of cells: word image (green frame = glyph split ok, red = failed), its text and
# confidence, then its glyph tiles labelled with their letters. Takes a line from results.json; returns LaTeX.
def words_row(line):
    # Shrinks the shared scale when the words and their glyph rows would overflow the page
    widths = [max(w["image_px"][0], sum(g["image_px"][0] for g in w["glyphs"])) for w in line["words"]]
    gaps = WORD_SEP_CM * (len(widths) - 1) + GLYPH_SEP_CM * sum(len(w["glyphs"]) for w in line["words"])
    scale = min(WORD_CM_PER_PX, (ROW_MAX_CM - gaps) / max(sum(widths), 1))

    cells = []
    for w in line["words"]:
        frame = "green!60!black" if w["split_ok"] else "red"
        image = rf"\fcolorbox{{{frame}}}{{white}}{{\includegraphics[width={w['image_px'][0] * scale:.2f}cm]{{{w['image']}}}}}"
        label = rf"\scriptsize ``{tex(w['text'])}'' " + ("truth" if w.get("source") == "truth" else f"{w['conf']:.2f}")
        # Shows the glyph tiles when the split worked, else why it didn't
        if w["glyphs"]:
            tiles = rf"\hspace{{{GLYPH_SEP_CM}cm}}".join(
                rf"\shortstack{{\includegraphics[width={g['image_px'][0] * scale:.2f}cm]{{{g['image']}}}\\\tiny {tex(g['char'])}}}"
                for g in w["glyphs"])
        else:
            tiles = rf"\tiny\color{{red}} glyph split failed (score {w['split_score']:.2f}, ink suggests {w['n_natural']})"
        cells.append(rf"\begin{{tabular}}[t]{{@{{}}c@{{}}}}{image}\\{label}\\[1pt]{tiles}\end{{tabular}}")
    return rf"\hspace{{{WORD_SEP_CM}cm}}".join(cells)


# 1 lasso region as a tree: the crop, then each extracted line with its OCR text and measurements,
# then each line's words and their glyphs. Takes a region from results.json and the stages dict; returns TikZ code.
def region_block(k, region, stages):
    read, errors, size = region_errors(region)
    out = [r"\begin{tikzpicture}[node distance=3mm and 5mm]",
           rf"\node[img] (crop) {{\includegraphics[height=1.4cm,width=0.5\textwidth,keepaspectratio]{{{region['crop']}}}}};",
           rf"\node[right=of crop, align=left, font=\small, text width=8cm] {{\textbf{{Region {k}}}"
           rf" \textcolor{{gray}}{{\scriptsize CER {100 * errors / max(size, 1):.0f}\% ({errors}/{size} chars)}}\\"
           rf" truth: ``{tex(region['truth'])}''\\ read: ``{tex(read)}''}};"]
    prev = "crop"

    # Hangs each line under the crop as 1 row: its image, then its OCR text and measurements, centred together
    for i, line in enumerate(region["lines"]):
        width = min(line["image_px"][0] * LINE_CM_PER_PX, LINE_MAX_CM)
        image = rf"\fcolorbox{{gray!50}}{{white}}{{\includegraphics[width={width:.2f}cm]{{{line['image']}}}}}"
        # Truth-labelled lines also show what OCR read, since the label no longer does
        read = (rf"\\ \scriptsize\color{{gray}} OCR read ``{tex(line['ocr_text'])}''"
                if line.get("source") == "truth" else "")
        label = rf"``{tex(line['text'])}''" if line["text"] else r"\color{gray}(no words)"
        text = (rf"\parbox{{6.2cm}}{{\small {label}{read}\\ \scriptsize\color{{gray}} OCR conf {line['conf']:.2f}"
                rf" \textbar{{}} x-height {line['xheight_px']:.0f}\,px \textbar{{}} slant {line['slant_deg']:.0f}$^\circ$"
                rf" \textbar{{}} {line['blobs']} blobs}}")
        out.append(rf"\node[below=of {prev}.south west, anchor=north west, inner sep=0, xshift={'6mm' if i == 0 else '0mm'}]"
                   rf" (l{i}) {{$\vcenter{{\hbox{{{image}}}}}$\hspace{{5mm}}$\vcenter{{\hbox{{{text}}}}}$}};")
        out.append(rf"\draw[flow] ([xshift=3mm]crop.south west) |- (l{i}.west);")
        prev = f"l{i}"

        # Hangs the line's words under it, or says why there are none
        if stages.get("Words"):
            if line.get("words"):
                content = words_row(line)
            elif not line["text"]:
                content = r"\scriptsize\color{gray} no words: the truth gave this line none (speck or math)"
            else:
                content = (rf"\scriptsize\color{{red}} word split failed: {len(line['text'].split())} words in the text,"
                           rf" {line.get('clusters', 0)} ink clusters")
            out.append(rf"\node[below=2mm of l{i}.south west, anchor=north west, inner sep=0, xshift=6mm] (w{i}) {{{content}}};")
            out.append(rf"\draw[flow] ([xshift=3mm]l{i}.south west) |- (w{i}.west);")
            prev = f"w{i}"

    # Shows a dashed placeholder when no lines came out, then the stages still to come
    if not region["lines"]:
        out.append(r"\node[todo, below=of crop.south west, anchor=north west, xshift=6mm] (l0) {no lines found};")
        out.append(r"\draw[flow] ([xshift=3mm]crop.south west) |- (l0.west);")
        prev = "l0"
    pending = [name for name in ("Words", "Glyphs") if not stages.get(name)]
    if pending:
        out.append(rf"\node[todo, below=of {prev}.south west, anchor=north west] {{{r' $\to$ '.join(pending)}: not built yet}};")
    out.append(r"\end{tikzpicture}\par\bigskip")
    return "\n".join(out) + "\n"


# Compose page: each test text's real segment (when the corpus has the word) and 1 composition per seed,
# wrapping onto more rows as needed. Takes the "compose" entry of results.json; returns LaTeX.
def compose_page(compose):
    def cell(part, label):
        width = min(part["image_px"][0] * COMPOSE_CM_PER_PX, 17.5)
        return (rf"\shortstack{{\fcolorbox{{gray!40}}{{white}}{{\includegraphics[width={width:.2f}cm]{{{part['image']}}}}}"
                rf"\\\tiny\color{{gray}} {label}}}")

    doc = [r"\clearpage\section*{Compose}",
           rf"\small Test texts written with random glyphs from all pages, text samples included ({compose.get('glyphs', '?')} glyphs), at {compose['px_per_xh']} px per x-height."
           rf" Measured median gaps: letters {compose['letter_gap']:.2f}, words {compose['word_gap']:.2f} x-heights."
           r" Each row: the real segment when the corpus has the word, then 1 composition per seed.\par\medskip\normalsize"]
    for entry in compose["texts"]:
        note = rf" \textcolor{{red}}{{\small no glyph for {tex(' '.join(entry['missing']))}}}" if entry["missing"] else ""
        cells = ([cell(entry["real"], "real segment")] if entry["real"] else []) + [
            cell(c, f"seed {c['seed']}") for c in entry["composed"]]
        doc.append(rf"\textbf{{``{tex(entry['text'])}''}}{note}\par\nopagebreak\raggedright "
                   + r"\hspace{5mm plus 3mm}".join(cells) + r"\par\medskip")
    return "\n".join(doc) + "\n"


# Headline numbers for a run. Takes results.json contents, returns a dict.
def summary(results):
    lines = [line for page in results["pages"] for region in page["regions"] for line in region["lines"]]
    confs = [line["conf"] for line in lines]
    words = [w for line in lines for w in line.get("words", [])]
    return {"regions": sum(len(page["regions"]) for page in results["pages"]), "lines": len(lines),
            "conf": sum(confs) / len(confs) if confs else 0.0,
            "lines_split": sum(bool(line.get("words")) for line in lines), "words": len(words),
            "words_split": sum(w["split_ok"] for w in words), "glyphs": sum(len(w["glyphs"]) for w in words),
            # Glyphs whose baseline offset is over 0.5 x-heights (glyphs.MAX_DY), which compose skips
            "flagged": sum(abs(g.get("dy", 0)) > 0.5 for w in words for g in w["glyphs"]),
            "cer": cer(results)}


# Character error rate over all regions with a truth, spaces ignored. Takes results.json contents, returns a fraction.
def cer(results):
    counts = [region_errors(r)[1:] for page in results["pages"] for r in page["regions"] if r.get("truth")]
    return sum(e for e, _ in counts) / max(sum(n for _, n in counts), 1)


# 1 sentence of headline numbers for the report and history. Takes summary() output, returns plain text.
def headline(stats):
    return (f"text CER {stats['cer']:.1%}; {stats['regions']} regions -> {stats['lines']} lines (mean OCR conf {stats['conf']:.2f}) -> "
            f"{stats['words']} words from {stats['lines_split']} lines -> {stats['glyphs']} glyphs from "
            f"{stats['words_split']} words ({stats['flagged']} flagged for baseline offset)")


# Regions whose lines or OCR text differ from the previous run.
# Takes 2 results.json contents, returns [(page, region label, before, after)] with "n lines: text" strings.
def changes(results, previous):
    # Describes a region by its line count, each line's OCR text, and how many of its words split into glyphs
    def describe(region):
        errors = f", CER {region_errors(region)[1] / max(region_errors(region)[2], 1):.0%}" if region.get("truth") else ""
        return f"{len(region['lines'])} lines{errors}: " + " / ".join(
            f"{line['text']} [{sum(w['split_ok'] for w in line.get('words', []))}/{len(line.get('words', []))} split]"
            for line in region["lines"])
    before = {(p["page"], r["label"]): describe(r) for p in previous["pages"] for r in p["regions"]}
    diffs = []
    for page in results["pages"]:
        for region in page["regions"]:
            key = (page["page"], region["label"])
            # Counts the region as changed when it's new or its description differs
            if before.get(key) != describe(region):
                diffs.append((*key, before.get(key, "(new region)"), describe(region)))
    return diffs


# Most recent run folder older than this one, or None.
def previous_run(run_dir):
    # Returns None when no run has been saved yet
    if not RUNS.exists():
        return None
    older = sorted(d for d in RUNS.iterdir() if d.is_dir() and d.name < run_dir.name and (d / "results.json").exists())
    return older[-1] if older else None


# Reads the commit the run was made from, saved by run_demo.sh as commit.txt
# (line 1 short hash, line 2 "dirty" or "clean", then the full message).
# Takes the run folder, returns (hash, dirty, subject, body lines); hash is "" for runs saved before commit.txt existed.
def read_commit(run_dir):
    path = run_dir / "commit.txt"
    # Falls back to the run name when the run has no commit.txt
    if not path.exists():
        return "", False, run_dir.name, []
    sha, state, *message = path.read_text().splitlines()
    # Drops trailers like Co-Authored-By and the blank line after the subject
    body = [line for line in message[1:] if not line.startswith("Co-Authored-By:")]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    return sha, state == "dirty", message[0] if message else "", body


# Commit message body as LaTeX: "- " lines become list items, indented lines continue them, other lines are paragraphs.
# Takes body lines from read_commit, returns LaTeX text.
def commit_body(body):
    out, in_list = [], False
    for line in body:
        # Starts a list item
        if line.startswith("- "):
            out.append((r"" if in_list else r"\begin{itemize}\itemsep0pt") + rf"\item {tex(line[2:])}")
            in_list = True
        # Continues the current item
        elif in_list and line.startswith(" ") and line.strip():
            out.append(tex(line.strip()))
        # Ends the list on any other line
        else:
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append(tex(line) if line.strip() else r"\par")
    if in_list:
        out.append(r"\end{itemize}")
    return "\n".join(out)


# Report header: the commit's hash, subject and message, flagged when the tree had uncommitted changes.
# Takes read_commit output, returns LaTeX text.
def commit_header(sha, dirty, subject, body):
    doc = [rf"{{\large\bfseries {tex(subject)}}}\\[2pt]"]
    if sha:
        note = r" \textcolor{red}{+ uncommitted changes}" if dirty else ""
        doc.append(rf"{{\small commit \texttt{{{tex(sha)}}}{note}}}\par")
    if body:
        doc.append(r"{\small " + commit_body(body) + r"\par}")
    return "\n".join(doc) + r"\medskip\hrule\medskip"


# Writes figure.tex in the run folder and appends an entry to history.md. Takes the run folder; returns nothing.
def write_report(run_dir):
    sha, dirty, subject, body = read_commit(run_dir)
    results = json.loads((run_dir / "results.json").read_text())
    prev_dir = previous_run(run_dir)
    previous = json.loads((prev_dir / "results.json").read_text()) if prev_dir else None
    stats = summary(results)
    diffs = changes(results, previous) if previous else []

    # Commit at the top, then pipeline status and headline numbers
    doc = [PREAMBLE, rf"{{\small\color{{gray}} Pipeline demo {tex(run_dir.name)}}}\\[2pt]",
           commit_header(sha, dirty, subject, body), flowchart(results["stages"]),
           tex(headline(stats)).replace("->", r"$\to$") + "."
           r" Line images show the measured baseline (red) and x-height (blue)."
           r" Word frames are green when the glyph split worked and red when it failed.\par\medskip"]

    # Changes since the previous run, as a table of regions that differ
    if previous:
        doc.append(rf"\subsection*{{Changes since {tex(prev_dir.name)}}}")
        rows = " \\\\\n".join(rf"{tex(p)} & {tex(label)} & {tex(b)} & {tex(a)}" for p, label, b, a in diffs)
        doc.append(r"\begin{tabular}{@{}l p{3.2cm} p{5.6cm} p{5.6cm}@{}}\toprule Page & Region & Before & After \\ \midrule "
                   + rows + r" \\ \bottomrule\end{tabular}" if diffs else "No region changed.")
        doc.append(r"\par\medskip")

    # 1 section per page: the page with lassos and baselines, then each region's tree
    for i, page in enumerate(results["pages"]):
        # Starts each page after the 1st on a fresh sheet
        if i > 0:
            doc.append(r"\clearpage")
        doc.append(rf"\section*{{{tex(page['page'])}}}")
        doc.append(rf"\begin{{center}}\includegraphics[height=0.4\textheight]{{{page['overlay']}}}\\"
                   r"\small lassos (green, numbered) and line baselines mapped back to the page (red)\end{center}")
        doc += [region_block(k, region, results["stages"]) for k, region in enumerate(page["regions"])]
    # Compose page, for runs made since compose joined the demo
    if "compose" in results:
        doc.append(compose_page(results["compose"]))
    doc.append(r"\end{document}")
    (run_dir / "figure.tex").write_text("\n".join(doc))

    # Logs the run, newest last; skips when the run is already logged (report rebuilt)
    log = HISTORY.read_text() if HISTORY.exists() else "# Demo history\n"
    if f"## {run_dir.name}\n" in log:
        return
    since = f"{len(diffs)} regions changed since {prev_dir.name}" if previous else "first run"
    ran = ", ".join(name for name, on in results["stages"].items() if on)
    title = subject + (" (+ uncommitted changes)" if dirty else "")
    HISTORY.write_text(log +
                       f"\n## {run_dir.name}\n{title}\n\n- stages run: {ran}\n"
                       f"- {headline(stats)}\n"
                       f"- {since}\n- report: out/runs/{run_dir.name}/figure.pdf\n")


if __name__ == "__main__":
    write_report(Path(sys.argv[1]))
