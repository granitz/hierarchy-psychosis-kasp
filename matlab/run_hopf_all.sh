#!/bin/bash
# Launch the group fit and the subject refinement for one group, in the
# background, logging to a file.
#
#   ./run_hopf_all.sh -TAG main -ATLASNAME 416 -group sz -MED 0 -GRIDON 0
#
# Required: -TAG (names the run; results go to results/<TAG>/) and -ATLASNAME
# (416 or 116). There is deliberately no default atlas: a default is how one
# atlas's settings end up applied to the other's data.
#
# Environment (see kasp_paths.m): KASP_DATA is required; KASP_TIMESERIES,
# KASP_RESULTS_ROOT and KASP_LNA_DIR are optional.
#
# Refuses to start when the same run (tag, group, medication) is already
# running: two processes writing the same results files, and the same log,
# is how the antipsychotic-naive 116-parcel run of 2026-09-23 got mixed up.

MATLABCMD="${MATLABCMD:-matlab}"

# -------- defaults (can be overridden by flags) --------
GROUP='ctrl'
MED=0          # 0 = unmedicated, 1 = medicated (only used when GROUP='sz')
Tau=1
GRIDON=0       # 1 = grid over g; 0 = single run at g = 0.2
ATLASNAME=''
TAG=''

usage() {
  echo "Usage: $0 -TAG NAME -ATLASNAME {416|116} [-group ctrl|sz] [-MED 0|1] [-Tau N] [-GRIDON 0|1]"
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    -TAG)       TAG="$2"; shift 2;;
    -group)     GROUP="$2"; shift 2;;
    -Tau)       Tau="$2"; shift 2;;
    -ATLASNAME) ATLASNAME="$2"; shift 2;;
    -MED)       MED="$2"; shift 2;;
    -GRIDON)    GRIDON="$2"; shift 2;;
    -h|--help)  usage;;
    *) echo "Unknown option: $1"; usage;;
  esac
done

[ -z "$TAG" ]       && { echo "error: -TAG is required"; usage; }
[ -z "$ATLASNAME" ] && { echo "error: -ATLASNAME is required"; usage; }
[ -z "$KASP_DATA" ] && { echo "error: set KASP_DATA to the project root"; exit 1; }

cd "$(dirname "$0")" || exit 1          # run() below finds the scripts here

RUN_ID="${TAG}_${GROUP}_med${MED}"
if pgrep -f "RUN_ID='${RUN_ID}'" > /dev/null; then
  echo "error: run ${RUN_ID} is already running:"
  pgrep -af "RUN_ID='${RUN_ID}'"
  exit 1
fi

STAMP=$(date +%Y%m%d-%H%M%S)
LOGDIR="${KASP_LOG_DIR:-logs}"
LOGFILE="${LOGDIR}/hopf_${RUN_ID}_atlas-${ATLASNAME}_grid-${GRIDON}_${STAMP}.log"
mkdir -p "$LOGDIR"

export KASP_RUN_TAG="$TAG"

echo "=========================================================="
echo "  TAG       = $TAG"
echo "  GROUP     = $GROUP   MED = $MED"
echo "  ATLAS     = $ATLASNAME"
echo "  Tau       = $Tau     GRIDON = $GRIDON"
echo "  code      = $(git rev-parse --short HEAD 2>/dev/null)$( [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] && echo ' (modified)')"
echo "  log       = $LOGFILE"
echo "=========================================================="

nohup "$MATLABCMD" -nodisplay -nosplash -r "\
RUN_ID='${RUN_ID}'; \
try; \
    GROUP='$GROUP'; \
    ATLASNAME='$ATLASNAME'; \
    Tau=$Tau; \
    MED=$MED; \
    GRIDON=$GRIDON; \
    fprintf('Running: GROUP=%s MED=%d Tau=%d ATLAS=%s GRIDON=%d\n', GROUP, MED, Tau, ATLASNAME, GRIDON); \
    run('run_hopf_grid_group.m'); \
    run('run_hopf_subject_refinement.m'); \
catch ME; \
    disp('Error in MATLAB script:'); \
    disp(getReport(ME)); \
    exit(1); \
end; \
exit;" \
> "$LOGFILE" 2>&1 &

echo "MATLAB started in the background (PID $!)."
echo "Follow it with:  tail -f $LOGFILE"
