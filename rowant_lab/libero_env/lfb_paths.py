"""LF-b (registration ``lfc_b``): fixed locations and constants. Standard library only (both venvs).

Layout under ``ROOT`` (in the data folder; frames are never viewed and never leave it)::

    frames/<cond>/<suite>-t<task>-L<L>/agentview.npz, wrist-<i>.npz, looks.json   (capture side)
    truth/<cond>/<suite>-t<task>-L<L>.json                                         (evaluator only)
    learn/<cond>/<suite>-t<task>-L<L>/leftover.json, masks.npz                     (learn side)
    models/<builder>/<cond>/<suite>-t<task>-L<L>/...                               (builders)
    lib/<variant>/m2/onboard-v3, m2/policy-v3, m1/                                 (variant libraries)
"""

from __future__ import annotations

import os
from pathlib import Path
from rowant_lab.settings import DATA

ROOT = DATA / "runs" / "libero-m3" / "lfb"
FRAMES, TRUTH, LEARN, MODELS, LIB = (ROOT / x for x in ("frames", "truth", "learn", "models", "lib"))

SUITE = "libero_object"
TASKS = tuple(range(10))
LEARN_STATES = (0, 3, 6)
DEV_STATES = tuple(range(10))
CONDITIONS = ("clean", "depth_noise")
BUILDERS = ("monty", "tsdf")  # A1 = monty + M+; A2 = tsdf + M-; A3 = tsdf + M+
ARMS = {"A1": ("monty", "plus"), "A2": ("tsdf", "minus"), "A3": ("tsdf", "plus")}
NOISE_BUILDER = "montyfused"  # lfc_b_noise A4: Monty learning on TSDF-fused depth
NOISE_ARMS = {"A4": (NOISE_BUILDER, "plus")}
TUNE_SUITE = "libero_spatial"
TUNE_TYPES = ("cookies", "glazed_rim_porcelain_ramekin")

# learning looks (registered): ring at 45 deg elevation, 90 deg apart, 0.30 m standoff; a failed pose moves 15 deg
N_LOOKS = 4
ELEVATION_DEG = 45.0
STANDOFF_M = 0.30
AZ_STEP_DEG = 15.0
AZ_TRIES = (0, 1, -1, 2, -2, 3, -3)  # multiples of AZ_STEP_DEG around the nominal azimuth
ELEVATION_FALLBACK_DEG = 70.0  # amendment LFB-RING: when every azimuth fails at 45 deg (the far side is out of reach)
WRIST = "robot0_eye_in_hand"
WRIST_W = 512
AGENT = "agentview"
AGENT_W = 1024
SEED_SALT = "rowant-lfb-v1"


def scene_id(suite: str, task: int, L: int, held_out: str | None = None) -> str:
    """``held_out`` names a tuning scene's held-out type (a test scene holds out its task's target)."""
    w2 = os.environ.get("ROWANT_W2_SCENE")  # the second test world (rowant_lab/pro2/world2.py): one of its scenes; off by default
    if w2:
        return w2
    batch = os.environ.get("ROWANT_ST_BATCH")  # ST-1 new-object test: each batch has its own scenes; off by default
    return f"{suite}-t{int(task)}-L{int(L)}" + (f"-{held_out}" if held_out else "") + (f"-{batch}" if batch else "")


def variant_id(builder: str, cond: str, suite: str, task: int, L: int) -> str:
    """A learned variant library: T replaced by the builder's model."""
    return f"{builder}-{cond}-{scene_id(suite, task, L)}"


def minus_id(suite: str, task: int) -> str:
    """The library with the task's target type removed (the learning phase's perception)."""
    return f"minus-{suite}-t{int(task)}"
