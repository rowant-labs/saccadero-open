"""Runner- and truth-side locations for LIBERO M2 (the M2 plan). Standard library only.

M2 (perceived poses, grounded instruction): perception in the loop. This module re-exports everything of the
policy-safe ``m2_paths`` and adds what only the runner, the evaluator, the gate reader and the freeze may know: the
evidence files (plan, amendments, frozen config, registry, peeks), the run folders and the GATE read log in the data
folder, the numba cache, the scratch folder, the interpreters, the DEV and GATE state ranges and the episode
protocol's constants. The policy-side files and the perception server never import it
(``tests/test_m2_truth_guard.py``). The evidence files are not part of this release.

``pin_numba_cache()`` sets ``NUMBA_CACHE_DIR`` to M2's own cache folder for this process (verifier minor M6: an
M2-phase venv-libero process once ran with M1's cache folder); every venv-libero entry point calls it before LIBERO
or robosuite is imported, and refuses when numba is already loaded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from rowant_lab.libero_env.m2_paths import *  # noqa: F401,F403  (the policy-safe names)
from rowant_lab.libero_env.m2_paths import LIBRARY, PACKAGE_DIR
from rowant_lab.settings import DATA

EVIDENCE = PACKAGE_DIR / "evidence-m2"
PLAN = EVIDENCE / "plan.json"
PLAN_SHA256_LOG = EVIDENCE / "plan-sha256.txt"
PLAN_SHA256 = "ef3af706d1e840d63cb4bb1b4cc85ac2b11a9a8672bfc39fc6eaa420e251f470"
AMENDMENTS = EVIDENCE / "plan-amendments.json"
FROZEN_CONFIG = EVIDENCE / "frozen-config.json"
FROZEN_CONFIG_SHA256_LOG = EVIDENCE / "frozen-config-sha256.txt"
GATE_REGISTRY = EVIDENCE / "gate-registry.json"
PEEKS = EVIDENCE / "peeks.json"
DEV_SUMMARY = EVIDENCE / "dev-summary.json"
ONBOARDING = EVIDENCE / "onboarding.json"
SEED_RESERVATIONS = EVIDENCE / "seed-reservations.json"

COLD = DATA
RUNS = COLD / "runs" / "libero-m2"
assert LIBRARY == RUNS / "library"
DEV_RUNS = RUNS / "dev"
D0_RUNS = RUNS / "d0"
D1_RUNS = RUNS / "d1"
FM7_RUNS = RUNS / "frame-tests" / "fm7"
GATE_RUNS = RUNS / "gate"
NUMBA_CACHE = RUNS / "numba-cache"
GATE_READ_LOG = RUNS / "GATE-READ-LOG.tsv"

M1_FROZEN_CONFIG = PACKAGE_DIR / "evidence-m1" / "frozen-config.json"
M1_FROZEN_CONFIG_SHA256 = "1aefb1038b69c28a4053e7f2c7ec0b14ccd530d1bb92329a49956b50e8e72777"
M1_GATE_SUMMARY_SHA256 = "75c4a179d18997426bfc26352cd6349734b3aa514923e9680a8789305145f663"
M1_INIT_MANIFEST_SHA256 = "b1bb658b3a2d5d07c6266ac005fa23e19290c4ddf080dc95e2a7a4f5c648e827"

SCRATCH = Path(os.environ.get("ROWANT_M2_SCRATCH") or DATA / "scratch" / "libero-m2")

# the two interpreters (the same settings as rowant_lab/pro2/machine.py)
LIBERO_VENV_PY = os.environ.get("ROWANT_WORKER_PYTHON") or str(DATA / "venvs" / "worker" / "bin" / "python")
MONTY_VENV_PY = os.environ.get("ROWANT_SERVER_PYTHON") or str(DATA / "venvs" / "server" / "bin" / "python")

DEV_STATES = tuple(range(0, 10))
GATE_STATES = tuple(range(10, 50))
POLICY_CAP = {"libero_object": 280, "libero_spatial": 220}
WAIT_STEPS = 10
HOLD_AFTER = 20


def pin_numba_cache() -> str:
    """Point numba's cache at M2's folder for this process (before robosuite imports numba)."""
    if "numba" in sys.modules and os.environ.get("NUMBA_CACHE_DIR") != str(NUMBA_CACHE):
        raise RuntimeError(f"numba was imported with NUMBA_CACHE_DIR={os.environ.get('NUMBA_CACHE_DIR')!r}, "
                           f"not M2's {NUMBA_CACHE}")
    os.environ["NUMBA_CACHE_DIR"] = str(NUMBA_CACHE)
    return str(NUMBA_CACHE)
