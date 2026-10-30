#!/usr/bin/env bash
# Run the notebook headless for the run named in .env (KASP_RUN_TAG), keep a
# viewable copy of every output.
#
#   scripts/run_notebook.sh
#
# Writes into runs/<tag>/:
#   stats.csv, manifest.json            committed -- the run's numbers
#   reproduce_figures.executed.ipynb    committed -- the notebook with every output; GitHub shows it
#   reproduce_figures.executed.html     committed -- the same, opens in any browser
#   figures/, tables/                   local only
#
# Local paths are removed from both executed copies (scripts/redact_html.py).
# The source notebook, notebooks/reproduce_figures.ipynb, stays without outputs.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "note: uncommitted changes to tracked files -- this run's manifest will say clean: false"
    git status --short --untracked-files=no
    echo
fi

TAG=$(python -c "from kasp import config; print(config.RUN_TAG)")
OUT="runs/$TAG"
mkdir -p "$OUT"
echo "run tag: $TAG  ->  $OUT/"

jupyter nbconvert --to notebook --execute notebooks/reproduce_figures.ipynb \
    --ExecutePreprocessor.timeout=-1 \
    --output-dir "$OUT" --output reproduce_figures.executed
jupyter nbconvert --to html "$OUT/reproduce_figures.executed.ipynb" --output-dir "$OUT"
# Both copies are committed: take local paths out of what the notebook printed.
python scripts/redact_html.py "$OUT/reproduce_figures.executed.html" "$OUT/reproduce_figures.executed.ipynb"

echo
echo "viewable copy: $OUT/reproduce_figures.executed.html"
