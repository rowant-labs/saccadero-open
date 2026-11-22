"""TRUTH-FREE: the vectorised P0 front end for the CHECK: about 256 patch features of one camera frame (location,
surface normal, hsv) in a few numpy calls, reproducing Monty's per-patch maths.

Inputs: the RGB and 16-bit depth of one frame, the fixed ``camera.json`` and the blob mask from
``e6/plane_crop.analyse`` (computed once per frame and shared with HPG, ``pipeline.py``). Nothing here knows the
object or its pose.

Per frame:

1. Patch centres: ``n`` distinct blob pixels at least ``margin`` px from the border, drawn by E-0d's
   ``camera.sample_centres`` (imported) with ``numpy.random.default_rng([seed, int(token, 16)])``. E-0d seeded with the
   case's episode number; a blinded token is used here so that the front end needs nothing from the render key.
2. For each centre, Monty's ``surface_normal_total_least_squares`` on a W x W patch (W = 7 by default, E-0d's
   recommended camera patch) is reproduced in batch:
   ``center_neighbors`` keeps the on-object pixels whose pixel distance to the centre is at most ``W / 3.2``, as
   coordinates relative to the centre point; then ``M = X^T X / n - p p^T`` with ``p = mean(X) / n`` (Monty's own
   formula, kept as is), the eigenvector of the smallest eigenvalue (``numpy.linalg.eigh`` on the stacked 3 x 3
   matrices here, ``numpy.linalg.eig`` per patch in Monty), turned to face the camera (``view_dir`` = the camera's
   back axis, Monty's ``cam_to_world[:3, 2]``). Only the pixels of the neighbourhood disc are back-projected.
3. ``hsv = skimage.color.rgb2hsv`` of the centre pixels (Monty: ``rgb2hsv(rgba[:3])`` of the centre pixel).
4. Location: the centre pixel's world point (``plane_crop.backproject`` maths, then ``T_world_from_camera``).

Curvature is not computed: E-0d found it does not transfer at camera scale and recommended location + normal + hsv.
A patch is valid when its normal is (the centre is on the blob by construction). E-0d also required a valid
curvature; at 7 px that dropped 1 of 29,952 patches. ``equivalence.compare_frontend`` checks the normals and hsv
against Monty's own per-patch functions (through E-0d's ``camera.patch_features``) on the same centres.
"""

from __future__ import annotations

import numpy as np

from rowant_lab.hybrid_frontier.screens.e0d_feature_transfer import camera as e0d_camera

TRUTH_FREE = True
DEFAULT_WIDTH = 7
DEFAULT_N_CENTRES = 256
CENTRE_SEED = 20261017  # E-0d's centre seed; the second seed word is the blinded token
NEIGHBOR_PATCH_FRAC = 3.2  # Monty's surface_normal_total_least_squares default


def token_seed(token: str, seed: int = CENTRE_SEED) -> list[int]:
    return [int(seed), int(token, 16)]


def disc_offsets(width: int, frac: float = NEIGHBOR_PATCH_FRAC) -> np.ndarray:
    """``(m, 2)`` (drow, dcol) of the patch pixels within ``width / frac`` px of the centre, in Monty's row-major
    patch order (``center_neighbors`` + ``pixel_dist_to_center``)."""
    half = width // 2
    rr, cc = np.mgrid[0:width, 0:width]
    dist = np.sqrt((cc - half) ** 2 + (rr - half) ** 2)
    keep = (dist <= width / frac).reshape(-1)
    return np.stack([rr.reshape(-1)[keep] - half, cc.reshape(-1)[keep] - half], axis=1)


def world_points_at(depth_u16: np.ndarray, rows: np.ndarray, cols: np.ndarray, camera: dict) -> np.ndarray:
    """World points of the given pixels (NaN where depth is invalid), with ``plane_crop``'s back-projection maths."""
    fx, fy, cx, cy = (float(camera[k]) for k in ("fx", "fy", "cx", "cy"))
    d = np.asarray(depth_u16)[rows, cols]
    z = d.astype(np.float64) * float(camera["depth"]["unit_m"])
    z[(d == 0) | (d == np.iinfo(np.uint16).max)] = np.nan
    u = cols.astype(np.float64)
    v = rows.astype(np.float64)
    pts = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1)
    T = np.asarray(camera["T_world_from_camera"], dtype=float)
    return pts @ T[:3, :3].T + T[:3, 3]


def sample_centres(mask: np.ndarray, token: str, *, n: int = DEFAULT_N_CENTRES, width: int = DEFAULT_WIDTH,
                   seed: int = CENTRE_SEED) -> np.ndarray:
    rng = np.random.default_rng(token_seed(token, seed))
    return e0d_camera.sample_centres(mask, n, rng, margin=width // 2 + 1)


def tls_normals(rel: np.ndarray, on: np.ndarray, view_dir: np.ndarray) -> np.ndarray:
    """Monty's TLS normal for P stacked neighbourhoods. ``rel``: (P, m, 3) points relative to each centre; ``on``:
    (P, m) on-object flags."""
    w = on.astype(np.float64)
    X = np.where(on[..., None], rel, 0.0)
    n_pts = w.sum(axis=1)  # >= 1: the centre itself is on the blob
    xtx = np.einsum("pmi,pmj->pij", X, X)
    mean = X.sum(axis=1) / n_pts[:, None]
    p_mean = mean / n_pts[:, None]  # Monty: 1 / n_points * np.mean(x_mat, axis=0)
    M = xtx / n_pts[:, None, None] - np.einsum("pi,pj->pij", p_mean, p_mean)
    _, vec = np.linalg.eigh(M)
    nrm = vec[:, :, 0].copy()
    flip = nrm @ np.asarray(view_dir, dtype=float) < 0
    nrm[flip] *= -1
    return nrm


def patch_features(rgb: np.ndarray, depth_u16: np.ndarray, mask: np.ndarray, camera: dict, centres: np.ndarray, *,
                   width: int = DEFAULT_WIDTH, view_dir: np.ndarray | None = None) -> dict:
    """``{"location", "normal", "hsv", "valid", "n_on", "centres"}`` for the given (row, col) centres."""
    from skimage.color import rgb2hsv  # noqa: PLC0415

    vd = e0d_camera.view_direction(camera) if view_dir is None else view_dir
    centres = np.asarray(centres, dtype=np.int64).reshape(-1, 2)
    P = len(centres)
    if P == 0:
        z3 = np.zeros((0, 3))
        return {"location": z3, "normal": z3, "hsv": z3, "valid": np.zeros(0, bool), "n_on": np.zeros(0, int),
                "centres": centres}
    off = disc_offsets(width)
    rows = centres[:, :1] + off[None, :, 0]
    cols = centres[:, 1:] + off[None, :, 1]
    pts = world_points_at(depth_u16, rows, cols, camera)  # (P, m, 3)
    on = np.asarray(mask, dtype=bool)[rows, cols]
    ci = int(np.flatnonzero((off[:, 0] == 0) & (off[:, 1] == 0))[0])
    loc = pts[:, ci, :].copy()
    rel = pts - loc[:, None, :]
    nrm = tls_normals(rel, on, vd)
    hsv = np.asarray(rgb2hsv(np.asarray(rgb)[centres[:, 0], centres[:, 1], :3]), dtype=np.float64)
    valid = on[:, ci] & np.isfinite(loc).all(axis=1)
    return {"location": loc, "normal": nrm, "hsv": hsv, "valid": valid, "n_on": on.sum(axis=1), "centres": centres}
