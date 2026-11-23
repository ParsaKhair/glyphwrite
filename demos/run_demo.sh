#!/usr/bin/env bash
# Runs the demo for the current working tree on biglab, then files a dated report locally.
#   bash demos/run_demo.sh
# Also started in the background after every commit by .git/hooks/post-commit.
# Output: demos/out/runs/<date>-<time>_<commit>[-dirty]/figure.pdf, plus an entry in demos/history.md.
# "-dirty" means the .py files had uncommitted changes when the demo ran.
set -euo pipefail
cd "$(dirname "$0")/.."
# Loads DEMO_HOST (user@host of the remote machine) from the untracked demos/demo.env
[ -f demos/demo.env ] && . demos/demo.env
H=${DEMO_HOST:?set DEMO_HOST=user@host in demos/demo.env}
SSH="ssh -F /dev/null -o BatchMode=yes"
# Records the commit now, before the run, in case HEAD moves while it runs
sha=$(git rev-parse --short HEAD)
state=$(git diff --quiet HEAD -- '*.py' && echo clean || echo dirty)
message=$(git log -1 --format=%B)
tag=$(date +%Y%m%d-%H%M)_$sha$([ "$state" = dirty ] && echo -dirty || true)
run=demos/out/runs/$tag

# Copies code and samples to biglab, where the venv and TrOCR live
rsync -az -e "$SSH" --exclude .git --exclude .venv --exclude __pycache__ --exclude demos/out ./ "$H:glyphwrite/"

# Runs the pipeline there, retrying when the host biglab picked can't load the venv (big02/big03 can't; exit 99)
for attempt in 1 2 3 4 5; do
  status=0
  $SSH "$H" "cd ~/glyphwrite && { .venv/bin/python -c 'import PIL, torch' 2>/dev/null || exit 99; } && .venv/bin/python demos/segment_demo.py $run" || status=$?
  [ "$status" -ne 99 ] && break
done
[ "$status" -eq 0 ] || { echo "demo failed on biglab (exit $status)"; exit 1; }

# Fetches the run folder
mkdir -p "$run"
rsync -az -e "$SSH" "$H:glyphwrite/$run/" "$run/"

# Saves the commit for the report header: hash, clean/dirty, then the full message
printf '%s\n%s\n%s\n' "$sha" "$state" "$message" > "$run/commit.txt"

# Writes the report and history entry, then compiles the PDF
python3 demos/report.py "$run"
(cd "$run" && latexmk -pdf -interaction=nonstopmode -halt-on-error figure.tex > latexmk.log 2>&1) \
  || { echo "LaTeX failed, see $run/latexmk.log"; exit 1; }
echo "report: $run/figure.pdf"
