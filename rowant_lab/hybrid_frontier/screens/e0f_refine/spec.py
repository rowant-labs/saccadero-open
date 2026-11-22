"""TRUTH-FREE: declared constants of screen E-0f (fixed 2026-10-18, before any dev case was scored).

Nothing here was tuned on dev truth. The only dev data seen before this file was frozen were (a) E-0c's and E-0b's
reports and (b) an exploratory look at 12 dev frames: the plane fit reproduces E-0c's stored plane exactly, and at
the TRUE pose the blob depth points lie a median 0.03-0.34 mm from the U77 graph's tangent planes (a model-fidelity
check, not a tuning run).

One change before scoring, from a TRUTH-FREE smoke on 12 frames (latency and convergence status only; no pose
error was computed): the first draft took 25-70 ms per frame, 80% of it in the k-NN query, and most frames ran to
the iteration cap with sub-micrometre steps. So N_DATA 2000 -> 1000, K_NEIGHBOURS 8 -> 6, STAGES (5, 5, 20) -> (4, 4,
12) iterations and the stop step 5 um / 0.005 deg -> 20 um / 0.02 deg (both far below the 1 mm / 2 deg target).

The constants below follow from the camera (0.5 m, fx 515 px: one pixel is about 1 mm, depth
quantised to 0.1 mm), the U77 graph's node spacing (median about 2 mm) and E-0c's measured start error (median ADD-S
2.6 mm, p90 about 9 mm).

Frames. World B (the E6 renders): metres, z up, the table at z = 0. A pose ``(R, t)`` places a model-frame point m
(object-local x = m - p_train) at ``R (m - p_train) + t``, as E-0c's guesses do.
"""

from __future__ import annotations

import math

import numpy as np

TRUTH_FREE = True
P_TRAIN = np.array([0.0, 1.5, 0.0])

# ---------------------------------------------------------------- data (one frame)
DATA_SEED = 20261018  # the second seed word is the blinded token (as E-0c's centres)
N_DATA = 1000  # blob pixels sampled per frame (all of them when the blob is smaller)
DATA_NORMAL_STEP_PX = (2, 1)  # depth-image normal: cross product of the +-s px neighbours, s tried in this order
MIN_DATA = 30  # fewer valid blob points: no refinement (the start pose is returned)

# ---------------------------------------------------------------- models
FUSED_GRID_M = 0.0025  # voxel centroids of the 1 mm fused view-finder prior used as model points
FUSED_NORMAL_RADIUS_M = 0.003  # PCA normal of each model point from the 1 mm prior points within this radius
FUSED_NORMAL_MIN_POINTS = 6
FUSED_MAX_PLANARITY = 0.2  # smallest / middle eigenvalue above this: no reliable normal, the point is dropped

# ---------------------------------------------------------------- visibility filter (recomputed every iteration)
VIS_FRONT_COS = 0.0  # model normal . (camera - point) / |...| must exceed this (faces the camera)
VIS_BIN_PX = 3  # z-buffer bins of 3 x 3 px over the projected model points (self-occlusion)
VIS_TOL_M = 0.003  # a point is not occluded if it is within 3 mm of its bin's nearest model point
VIS_MIN_HEIGHT_M = 0.002  # model points under 2 mm above the fitted table are dropped (the blob mask starts at 3 mm)

# ---------------------------------------------------------------- ICP
K_NEIGHBOURS = 6  # the nearest VISIBLE model point among the k nearest is the correspondence
STAGES = ((4, 0.010, 0.002), (4, 0.005, 0.001), (12, 0.003, 0.0005))  # (iterations, distance gate m, Huber delta m)
NORMAL_COMPAT_DEG = 60.0  # data normal vs model normal; pairs beyond are rejected (when the data normal exists)
DAMPING = 1e-6  # Levenberg damping, times the weight sum, on every unknown (degenerate directions stay put)
MAX_STEP_M = 0.005  # per-iteration step clamps
MAX_STEP_DEG = 5.0
STOP_STEP_M = 2e-5  # final stage: stop when the step is below both
STOP_STEP_DEG = 0.02
MIN_CORR = 12  # fewer accepted correspondences: stop, keep the current pose

# ---------------------------------------------------------------- truth-free fit quality
INLIER_GATE_M = 0.003
INLIER_RESID_M = 0.001
FS_DILATE_PX = 3  # free space: visible model points that project off the blob dilated by 3 px
FS_MIN_HEIGHT_M = 0.006

# ---------------------------------------------------------------- arms (declared before scoring)
ARMS = {
    "pick": "E-0c's picked (object, pose), not refined (the 'before' reading)",
    "u77": "PRIMARY: point-to-plane ICP, Huber + gate, normal test, visibility filter, U77 graph (nodes + normals)",
    "fused": "the same refinement against the F1-77 fused view-finder cloud (1 mm prior, 2.5 mm grid, PCA normals)",
    "u77_best3": "SECONDARY: refine each of E-0c's 3 poses of the picked object (U77), keep the best truth-free fit "
                 "(inlier share minus free-space share)",
    "u77_p2p": "ABLATION: point-to-point (weighted Kabsch) instead of point-to-plane",
    "u77_novis": "ABLATION: no visibility filter (every model point is a candidate)",
    "u77_l2": "ABLATION: no Huber weights (plain least squares inside the distance gate)",
}
READINESS_ARMS = ("oracle", "pick", "u77", "fused", "u77_best3")  # E-0b's checkers (oracle = true mesh, true pose)
TIMED_ARMS = ("u77", "fused", "u77_best3")

# ---------------------------------------------------------------- readings (EVALUATOR-ONLY use, declared here)
GRASP_GRADE = (1.0, 2.0)  # mm (centroid translation), deg (symmetry-credited rotation): E-0b's grasp tau at 90%
BANDS = ((1.0, 2.0), (2.0, 2.0), (3.0, 5.0), (5.0, 10.0))  # E-0b's pose bands
SUCTION_GRADE = (3.0, 5.0)  # the loose end of E-0b's suction tolerance (1-3 mm, 2-5 deg)
ADDS_THRESHOLDS_MM = (1.0, 2.0, 5.0, 10.0)
SYMMETRY_LEVEL = "nominal"  # the v2 rescore's decision level
TIMING_SEED_NOTE = "tokens ordered by e9/latency.sample_tokens (seed 20262305), 10 warm-ups, every frame timed"


def deg2rad(d: float) -> float:
    return math.radians(d)


def declared() -> dict:
    """Every constant above as a JSON-ready dict (written into each output)."""
    out = {}
    for k, v in globals().items():
        if k.isupper() and not k.startswith("_"):
            out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    return out
