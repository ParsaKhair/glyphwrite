# Agent guidelines

## Pace
- Add a little at a time: 1 module or section per turn, then stop for review.
- Propose layout or design before writing code.
- Don't build ahead of what was asked.
- Keep end-of-turn summaries short.

## Layout
- Flat modules at the repo root next to `main.py`. No nested package named after the repo.
- Demos and tests go in `demos/`. Their data (`demos/data/`) and outputs (`demos/out/`) stay untracked.

## Comments
- Start each function, class, or code block with a short high-level comment: what it does, what it takes, what it returns.
- Put field-specific notes on the line above the field, not in the header.
- Inside functions, give each block a 1-line comment saying what it does. Conditionals read "Xs when Y" (e.g. "Fails when OCR gave no text").
- No end-of-line comments.
- No docstrings or comments that restate the code.
- Be terse. Skip boilerplate and filler.
- Write numbers as digits ("1", not "one").
