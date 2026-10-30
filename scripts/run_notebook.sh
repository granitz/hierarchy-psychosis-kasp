#!/usr/bin/env bash
# Run the notebook headless for the run named in .env (KASP_RUN_TAG), keep a
# viewable copy.
#
#   scripts/run_notebook.sh
#
# Writes into runs/<tag>/:
#   stats.csv, manifest.json            committed -- the run's numbers
#   reproduce_figures.executed.html     committed -- every figure and number, opens in a browser;
#                                       home paths redacted (scripts/redact_html.py)
#   reproduce_figures.executed.ipynb    local only -- the same as a notebook
#   figures/, tables/                   local only
#
# The executed .ipynb stays out of git (outputs make diffs unreadable); the
# HTML is committed as the run's viewable record. The source notebook stays
# stripped; the committed numbers are in stats.csv.
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
# The HTML is committed: take the username out of the paths it prints.
python scripts/redact_html.py "$OUT/reproduce_figures.executed.html"

echo
echo "viewable copy: $OUT/reproduce_figures.executed.html"
