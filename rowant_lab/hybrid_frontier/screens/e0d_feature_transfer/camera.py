"""TRUTH-FREE: Monty-style patch features from one stored E6 render (RGB + 16-bit depth + camera.json).

Inputs are sensor data only: the stored PNGs and the fixed camera description that leaves the evaluator
(``e6/render_core.camera_dict``: 640x480, fovy 50 deg, fx = fy = 514.7 px, principal point (319.5, 239.5), OpenCV
axes, depth = z along the optical axis in 0.1 mm units, 0 invalid). Nothing here knows the object or its pose.

Steps:

1. ``e6/plane_crop.analyse`` (imported, unchanged) fits the table plane and returns the blob mask: every valid
   pixel more than 3 mm above the plane. That mask plays the part of Monty's semantic channel (``semantic_3d[:, 3]``
   > 0 on the object).
2. Every pixel is back-projected (``plane_crop.backproject``) and moved to the world with ``T_world_from_camera``.
3. Patch centres: ``n_centres`` distinct blob pixels drawn with ``numpy.random.default_rng([seed, episode])``,
   restricted to centres whose largest patch fits inside the image.
4. For each centre and each patch width W, a W x W window around it becomes Monty's flattened ``(W*W, 4)`` patch
   (x, y, z, on-object) in row-major order, with the centre at row/column ``W // 2`` exactly as
   ``ObservationProcessor.process`` indexes it. Monty's own functions then compute the features
   (``src/tbp/monty/frameworks/utils/sensor_processing.py``, imported read-only):

   * ``surface_normal_total_least_squares(patch, centre_id, view_dir)`` (TLS, the default
     ``SurfaceNormalMethod``), with ``view_dir`` = the world direction of the camera's back axis (Monty passes
     ``cam_to_world[:3, 2]``, its camera's +z, which points back at the viewer in Monty's convention);
   * ``principal_curvatures(patch, centre_id, normal, weighted=True)``;
   * ``log_sign([k1, k2])`` for ``principal_curvatures_log``;
   * ``pose_fully_defined = |k1 - k2| >= 10`` (``pc1_is_pc2_threshold``);
   * ``hsv = skimage.color.rgb2hsv(rgb[centre])`` (Monty: ``rgb2hsv(rgba[:3])`` on the centre pixel).

   A patch is used only if both the normal and the curvature are valid (Monty's ``process_features_in_lm``).

Differences from Monty's own sensing (declared, not corrected): the camera is 0.4-0.55 m away (about 0.8-1.1 mm
per pixel) instead of 2.5 cm (about 0.08 mm per pixel), so a W-pixel patch covers about 12x more surface; the
camera looks from one 45 deg viewpoint, not along the surface normal; lighting is the E6 render rule (key, fill,
headlight, shadows, RGB noise sd 3/255), not Monty's simulator lighting; no feature noise is added (Monty's E6
runs added hsv 0.1, curvature-log 0.1, 2 deg pose and 2 mm location noise to its sensed features).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rowant_lab.hybrid_frontier.e6 import plane_crop as pc
from rowant_lab.hybrid_frontier.e6 import png_clean

TRUTH_FREE = True
PC1_IS_PC2_THRESHOLD = 10.0  # ObservationProcessor default
DEFAULT_WIDTHS = (7, 13, 25, 49)
DEFAULT_N_CENTRES = 256
CENTRE_SEED = 20261017


def load_camera(blinded_dir) -> dict:
    return json.loads((Path(blinded_dir) / "camera.json").read_text())


def load_render(blinded_dir, files: dict) -> tuple[np.ndarray, np.ndarray]:
    b = Path(blinded_dir)
    return png_clean.read_png(b / files["rgb"]), png_clean.read_png(b / files["depth"])


def world_points(depth_u16: np.ndarray, camera: dict) -> np.ndarray:
    """(H, W, 3) world coordinates of every pixel (invalid depth gives z = 0 rows; they are never on the blob)."""
    pts = pc.backproject(pc.depth_metres(depth_u16, camera), camera)
    T = np.asarray(camera["T_world_from_camera"], dtype=float)
    return pts @ T[:3, :3].T + T[:3, 3]


def view_direction(camera: dict) -> np.ndarray:
    """World direction of the camera's back axis (minus the optical axis): Monty's ``cam_to_world[:3, 2]``."""
    T = np.asarray(camera["T_world_from_camera"], dtype=float)
    return -T[:3, 2] / np.linalg.norm(T[:3, 2])


def camera_centre(camera: dict) -> np.ndarray:
    return np.asarray(camera["T_world_from_camera"], dtype=float)[:3, 3]


def sample_centres(mask: np.ndarray, n: int, rng: np.random.Generator, margin: int) -> np.ndarray:
    """``(m, 2)`` (row, col) of up to ``n`` distinct blob pixels at least ``margin`` px from the image border."""
    h, w = mask.shape
    ok = mask.copy()
    ok[:margin, :] = False
    ok[h - margin:, :] = False
    ok[:, :margin] = False
    ok[:, w - margin:] = False
    rc = np.argwhere(ok)
    if len(rc) == 0:
        return np.zeros((0, 2), dtype=int)
    pick = rng.choice(len(rc), size=min(n, len(rc)), replace=False)
    return rc[np.sort(pick)]


def mm_per_px(depth_m: float, camera: dict) -> float:
    return 1e3 * float(depth_m) / float(camera["fx"])
