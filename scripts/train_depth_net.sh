#!/usr/bin/env bash
# Train a depth net of the kind the two registered configurations use. One GPU, the worker's environment.
#   source configs/paths.env && source "$ROWANT_DATA/env.sh"
#   bash scripts/train_depth_net.sh PLAN_DIR OUT_DIR [HOURS] [FIRST_NET]
#
# This is the training code as it ran, with the long net's own options. It does NOT rebuild our two files:
# training has no fixed seed and stops by the clock, and our training pictures were not kept (docs/DEPTH-NETS.md
# says what they were made of and how to make a frame plan). The nets we used are release files of this
# repository; their checksums are in docs/records.
#
#   PLAN_DIR      the folder holding a frame plan, frames-plan.json.gz (rowant_lab/pro2/colour_long.py plan,
#                 or colour_frames.py plan)
#   FIRST_NET     start from this net's weights (our main net is our first net trained on for three more hours)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
: "${ROWANT_DATA:?source configs/paths.env first}"
WPY=${ROWANT_WORKER_PYTHON:-$ROWANT_DATA/venvs/worker/bin/python}
PLAN=${1:?a frame plan}; OUT=${2:?an output folder}; H=${3:-3}; INIT=${4:-}
cd "$REPO"
"$WPY" -I -B rowant_lab/pro2/colour_frames.py run --plan "$PLAN" --out "$OUT" --workers "${WORKERS:-8}"
START=(); [ -z "$INIT" ] || START=(--init "$INIT")
"$WPY" -I -B rowant_lab/pro2/colour_train.py --frames "$OUT/frames" --out "$OUT/net" --hours "$H" \
    --workers "${WORKERS:-8}" --balance --by-key --val-frames 24 --lr 2e-4 --warmup-steps 500 ${START[@]+"${START[@]}"}
"$WPY" -I -B rowant_lab/pro2/colour_score.py --frames "$OUT/frames" --net "$OUT/net/last.pt" \
    --out "$OUT/net/score.json" --by-key --val-frames 24
