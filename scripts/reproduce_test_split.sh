#!/usr/bin/env bash
# The benchmark's test split with one registered configuration: 10,170 episodes on one machine.
#   source configs/paths.env && source "$ROWANT_DATA/env.sh"
#   bash scripts/reproduce_test_split.sh v9|v11 --test-split
# This touches the test split and runs only with the flag. Please develop on the training split.
# One RTX 4090: we ran ten workers a machine, which would be about 13 to 20 hours for v9 and 19 to 27 for v11 on one
# machine; the default here is eight, which is safer for a 24 GB card and about a quarter slower. It can be stopped
# and started again; finished episodes are kept. It needs the benchmark's archive evaluation.zip (ROWANT_BUNDLE): the test
# split's task files are read from it and its checksum is compared with the one registered before our runs.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
: "${ROWANT_DATA:?source configs/paths.env first}"
C=${1:?v9 or v11}
[ "${2:-}" = "--test-split" ] || { echo "this runs the benchmark's test split: add --test-split if you mean it"; exit 3; }
case "$C" in v9) ATT=1;; v11) ATT=2;; *) echo "v9 or v11"; exit 2;; esac
SPY=${ROWANT_SERVER_PYTHON:-$ROWANT_DATA/venvs/server/bin/python}
cd "$REPO"
"$SPY" scripts/machine_check.py
PYTHONPATH="$REPO" "$SPY" -P -m rowant_lab.pro2.scored verify --attempt "$ATT"
PYTHONPATH="$REPO" "$SPY" -P -m rowant_lab.pro2.scored unpack --attempt "$ATT" --test-split
PYTHONPATH="$REPO" "$SPY" -P -m rowant_lab.pro2.scored plan --attempt "$ATT" --test-split
PLAN=$ROWANT_DATA/runs/pro2/scored/plan/episodes.json
OUT=$ROWANT_DATA/runs/reproduce/test-split-$C
python3 scripts/run.py --config "$C" --episodes "$PLAN" --out "$OUT" --workers "${WORKERS:-8}" --test-split
python3 scripts/score.py --plan "$PLAN" --rows "$OUT" --out "$OUT/score.json"
