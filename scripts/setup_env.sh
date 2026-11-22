#!/usr/bin/env bash
# Build the two Python environments this code runs in. Linux with an NVIDIA GPU; needs `uv`.
#
#   source configs/paths.env        # your copy of configs/paths.example.env
#   bash scripts/setup_env.sh
#
# The steps are the ones the machines of both scored runs were set up with:
#   * Python 3.11.11 in both environments;
#   * the worker's environment (simulator side): requirements/worker.txt, then torch 2.5.1 built for CUDA 12.4;
#   * the perception server's environment: requirements/server.txt;
#   * the benchmark's own two source trees, made importable in the worker's environment only (they are used from
#     your download, never installed or copied);
#   * a LIBERO configuration file that points at your download;
#   * a probe for a renderer that works without a display (EGL, else OSMesa).
# System packages the renderer needs (Debian or Ubuntu):
#   apt-get install -y libegl1 libgl1 libgles2 libopengl0 libegl-mesa0 libgl1-mesa-dri libosmesa6 libglib2.0-0
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
: "${ROWANT_DATA:?set ROWANT_DATA first (see configs/paths.example.env)}"
BENCH=${ROWANT_BENCHMARK:-$ROWANT_DATA/benchmark/LIBERO-Pro-evaluation}
WPY=${ROWANT_WORKER_PYTHON:-$ROWANT_DATA/venvs/worker/bin/python}
SPY=${ROWANT_SERVER_PYTHON:-$ROWANT_DATA/venvs/server/bin/python}
W=$(dirname "$(dirname "$WPY")")
S=$(dirname "$(dirname "$SPY")")
CFG=$ROWANT_DATA/libero-config
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

[ -d "$BENCH/simulator-source/libero/libero/init_files" ] || {
    echo "the benchmark's download was not found under $BENCH (set ROWANT_BENCHMARK)"; exit 4; }
command -v uv >/dev/null || { echo "uv is needed: https://docs.astral.sh/uv/"; exit 4; }

log "Python 3.11.11 and the two environments"
uv python install 3.11.11
[ -x "$WPY" ] || uv venv -q --python 3.11.11 "$W"
[ -x "$SPY" ] || uv venv -q --python 3.11.11 "$S"
uv pip install -q --python "$WPY" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu -r "$REPO/requirements/worker.txt"
uv pip install -q --python "$WPY" --reinstall-package torch "torch==2.5.1+cu124" \
    --index-url https://download.pytorch.org/whl/cu124
uv pip install -q --python "$SPY" -r "$REPO/requirements/server.txt"

log "the benchmark's source trees and LIBERO's configuration, for the worker's environment"
SITE=$("$WPY" -c 'import site; print(site.getsitepackages()[0])')
printf '%s\n%s\n' "$BENCH/simulator-source" "$BENCH/service-source" > "$SITE/saccadero_benchmark.pth"
mkdir -p "$CFG" "$ROWANT_DATA/runs/libero-m2/numba-cache"
cat > "$CFG/config.yaml" <<EOF
assets: "$BENCH/simulator-source/libero/libero/assets"
bddl_files: "$BENCH/service-source/assets/bddl/libero-pro2.0-1680-v1"
benchmark_root: "$BENCH/simulator-source/libero/libero"
datasets: "$BENCH/simulator-source/libero/datasets"
init_states: "$BENCH/simulator-source/libero/libero/init_files"
EOF
echo "import os; os.environ.setdefault('LIBERO_CONFIG_PATH', '$CFG')" > "$SITE/saccadero_libero_config.pth"

log "a renderer without a display"
GL=""
for b in egl osmesa; do
    if MUJOCO_GL=$b PYOPENGL_PLATFORM=$b "$WPY" -I -c "
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom size=\".1\"/></worldbody></mujoco>')
r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); a = r.render()
assert a.shape == (64, 64, 3)
" 2>/dev/null; then GL=$b; break; fi
done
[ -n "$GL" ] || { log "no renderer works (install the system packages named at the top of this script)"; exit 5; }
echo "export MUJOCO_GL=$GL PYOPENGL_PLATFORM=$GL LP_NUM_THREADS=1" > "$ROWANT_DATA/env.sh"

log "checks"
"$WPY" -I -c "
import sys
sys.path.insert(0, '$REPO')
from rowant_lab.libero_env import m2_rpaths
m2_rpaths.pin_numba_cache()  # as the runner does, before robosuite is imported
import torch, numpy, mujoco, robosuite, libero
from rowant_lab.pro2 import dev_episode
dev_episode.adapter_class()  # the benchmark's environment class, loaded from your download as the runner loads it
assert torch.cuda.is_available(), 'no CUDA device'
print('worker: imports ok, numpy', numpy.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0),
      'mujoco', mujoco.__version__, 'robosuite', robosuite.__version__)"
PYTHONPATH="$REPO" "$SPY" -B -P -c "
import importlib, numpy
import rowant_lab.libero_env.m2_perceive as P
for m in P.PREIMPORTS + ('rowant_lab.fastlayer.scorers.render_compare', 'skimage.color', 'cv2', 'PIL.Image'):
    importlib.import_module(m)
print('server: imports ok, numpy', numpy.__version__)"
log "done. Before a run:  source $ROWANT_DATA/env.sh   (renderer: $GL, $(nproc) processors)"
