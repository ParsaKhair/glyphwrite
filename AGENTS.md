# Agent guidelines

## Pace
- Add a little at a time: 1 module or section per turn, then stop for review.
- Propose layout or design before writing code.
- Don't build ahead of what was asked.
- Keep end-of-turn summaries short.

## Layout
- 1 folder per system, flat inside: no subfolders unless a system clearly outgrows it. No package named after the repo.
  - Root: `main.py` (entry point) and `corpus.py` (data model shared by every system).
  - `preprocess/`: photo -> corpus (lines, OCR, words, glyphs).
  - `writing/`: picks, scores and places corpus glyphs to write text (compose, scoring and layout models).
  - `generation/`: future system that generates new glyphs; create it when its 1st module is written.
- Import across folders by package path from the repo root (`from preprocess.lines import ...`, `from writing.compose import ...`); run from the root.
- Demos and tests go in `demos/` and are tracked. Their data (`demos/data/`), outputs (`demos/out/`), notes (`demos/ideas/`), run log (`demos/history.md`) and host config (`demos/demo.env`) stay untracked.
- Ignore `demos/ideas/` (don't read, cite or build on it) unless the prompt explicitly asks for it.

## Comments
- Start each function, class, or code block with a short high-level comment: what it does, input, output.
- Put field-specific notes on the line above the field, not in the header.
- Inside functions, give each block a 1-line comment saying what it does. Conditionals read "Xs when Y" (e.g. "Fails when OCR gave no text").
- No end-of-line comments.
- No docstrings or comments that restate the code.
- Be terse. Skip boilerplate and filler.
- Write numbers as digits ("1", not "one").
- Write high level comments at top of each .py file describing purpose
