# Demos

A fixed smoke test of the pipeline as it currently exists. Run it after each commit or change,
then compare reports to see what the change did.

```bash
bash demos/run_demo.sh
```

It needs `demos/demo.env` (untracked) with the remote machine's login: `DEMO_HOST=user@host`.

This syncs the code and `data/` to biglab, runs the pipeline there (TrOCR lives in
`~/glyphwrite/.venv` on biglab), fetches the results, and writes:

- `out/runs/<date>-<time>_<commit>[-dirty]/figure.pdf`: pipeline status chart, a "changes since
  the previous run" table, then each page with its lassos and every region's tree of lines + OCR.
- an entry in `history.md` (untracked): stages run, line count, mean OCR confidence, regions changed.

`-dirty` means tracked `.py` files had uncommitted changes when the demo ran.
The report opens with the commit's hash, subject and full message.

## After every commit

`.git/hooks/post-commit` (local, untracked) starts `run_demo.sh` in the background, so `git commit`
returns at once. Its output goes to `out/logs/<commit>.log`. Runs queue 1 at a time; a queued run
is skipped when a newer commit has already replaced it. Rebases don't trigger runs.

## Files

- `segment_demo.py`: the lasso polygons (`REGIONS`, prose only, math left out) and the run itself.
  Update `STAGES` and the run when a pipeline stage gets built.
- `compose_demo.py`: the compose test texts (`WORDS`, `SEEDS`, `PX`). The demo run pools both pages' glyphs and
  renders each text next to its real segment; the report shows them on a final Compose page. Standalone, it
  writes 1 sheet from a corpus cached in its output folder, to iterate on compose without rerunning OCR:
  `python demos/compose_demo.py demos/out/compose` on biglab.
- `report.py`: builds `figure.tex` from a run's `results.json`, `commit.txt` and the previous run's
  results, and logs to `history.md`. Rebuilding a run's report doesn't log it twice:
  `python3 demos/report.py demos/out/runs/<run>`, then `latexmk -pdf figure.tex` in that folder.
- `samples.py`: extra lassos (keyed by true text) used only as glyph sources for compose.
- `ideas/` (untracked): experiments and ideas for later, 1 folder each; see `ideas/README.md`.
