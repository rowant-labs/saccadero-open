#!/usr/bin/env bash
# A sample of the practice list (training-split episodes) with both registered configurations, compared with the
# outcomes we recorded.
#   source configs/paths.env && source "$ROWANT_DATA/env.sh"
#   bash scripts/reproduce_practice.sh [OUT_DIR] [N]             (default N: 60; "all" runs the 476)
# 60 episodes take about half an hour per configuration on one RTX 4090 with 8 workers.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
: "${ROWANT_DATA:?source configs/paths.env first}"
OUT=${1:-$ROWANT_DATA/runs/reproduce/practice}
N=${2:-60}
SPY=${ROWANT_SERVER_PYTHON:-$ROWANT_DATA/venvs/server/bin/python}
cd "$REPO"
"$SPY" scripts/machine_check.py
if [ "$N" = all ]; then python3 scripts/episodes.py practice --out "$OUT/episodes.json"
else python3 scripts/episodes.py practice --out "$OUT/episodes.json" --sample "$N"; fi
for C in v9 v11; do
    python3 scripts/run.py --config "$C" --episodes "$OUT/episodes.json" --out "$OUT/$C" --workers "${WORKERS:-8}"
    python3 scripts/compare.py "$OUT/$C" --reference "$C"
done
