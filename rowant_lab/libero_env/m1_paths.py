"""Fixed locations for LIBERO M1 (the oracle gate; the M1 plan). Standard library only; reads nothing.

Large outputs (init-state arrays, the object library, run logs, numba cache) live in the data folder under
``RUNS``. The plan, its amendments, the frozen config and the small summaries that ``EVIDENCE`` names are not part
of this release.
"""

from __future__ import annotations

from pathlib import Path
from rowant_lab.settings import DATA

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
EVIDENCE = PACKAGE_DIR / "evidence-m1"
PLAN = EVIDENCE / "plan.json"
PLAN_SHA256_LOG = EVIDENCE / "plan-sha256.txt"
PLAN_SHA256 = "500efa853cadd96c551a9d96dee6c12593fa76924e29643746276bfc5a14c1a5"
AMENDMENTS = EVIDENCE / "plan-amendments.json"
FROZEN_CONFIG = EVIDENCE / "frozen-config.json"
FROZEN_CONFIG_SHA256_LOG = EVIDENCE / "frozen-config-sha256.txt"
GATE_REGISTRY = EVIDENCE / "gate-registry.json"
PEEKS = EVIDENCE / "peeks.json"
INIT_MANIFEST = EVIDENCE / "init-states-manifest.json"

COLD = DATA
RUNS = COLD / "runs" / "libero-m1"
INIT_DIR = RUNS / "init-states"
DEV_DIR = INIT_DIR / "dev"
GATE_DIR = INIT_DIR / "gate"
LIBRARY_DIR = RUNS / "library"
DEV_RUNS = RUNS / "dev"
NUMBA_CACHE = RUNS / "numba-cache"
GATE_READ_LOG = RUNS / "GATE-READ-LOG.tsv"

SUITES = ("libero_object", "libero_spatial")
TASK_ORDER = 0
N_TASKS = 10
N_STATES = 50
DEV_STATES = tuple(range(0, 10))
GATE_STATES = tuple(range(10, 50))
POLICY_CAP = {"libero_object": 280, "libero_spatial": 220}
ARM_B_CAP = 600
WAIT_STEPS = 10
HOLD_AFTER = 20  # no-op hold steps after the policy finishes (strict reading only)

LABEL = "ORACLE (true poses): diagnostic of the hands"
