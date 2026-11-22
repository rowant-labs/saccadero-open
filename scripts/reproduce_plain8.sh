#!/usr/bin/env bash
# The eight tasks of the benchmark's test split in their plain form, with both registered configurations.
#   source configs/paths.env && source "$ROWANT_DATA/env.sh"
#   bash scripts/reproduce_plain8.sh [OUT_DIR] [SEEDS...]        (default seeds: 142 143 144)
# 24 episodes per configuration by default; about 15 minutes each on one RTX 4090 with 8 workers.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
: "${ROWANT_DATA:?source configs/paths.env first}"
WPY=${ROWANT_WORKER_PYTHON:-$ROWANT_DATA/venvs/worker/bin/python}
OUT=${1:-$ROWANT_DATA/runs/reproduce/plain8}
shift || true
SEEDS=("$@"); [ ${#SEEDS[@]} -gt 0 ] || SEEDS=(142 143 144)
SPY=${ROWANT_SERVER_PYTHON:-$ROWANT_DATA/venvs/server/bin/python}
cd "$REPO"
"$SPY" scripts/machine_check.py
"$WPY" -I -B scripts/episodes.py plain8 --out "$OUT/episodes.json" --seeds "${SEEDS[@]}"
for C in v9 v11; do
    python3 scripts/run.py --config "$C" --episodes "$OUT/episodes.json" --out "$OUT/$C" --workers "${WORKERS:-8}"
    python3 scripts/compare.py "$OUT/$C"
done
python3 scripts/compare.py "$OUT/v11" --reference v11
python3 scripts/compare.py "$OUT/v9" "$OUT/v11"
