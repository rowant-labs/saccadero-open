"""TRUTH-FREE: table-plane fit from depth, the object blob, and the fixed physical crop window (E6 plan section 8).

Inputs are sensor data only: an RGB array, a 16-bit depth array and the fixed camera description (``camera.json``,
the intrinsics and extrinsics that plan section 8 lets leave the evaluator). Nothing here knows which object is on
the table, where it was placed or how it was posed.

Steps (all deterministic; the RANSAC draws come from a fixed seed in the spec, not from the data):

1. Back-project every valid depth pixel to the camera frame (OpenCV axes: x right, y down, z forward; pixel
   (u, v) has its centre at the integer index, principal point ``(cx, cy)`` from the camera file).
2. Fit the table plane by RANSAC on a strided subsample (3-point planes, inliers within ``inlier_m``), then refit
   by least squares (smallest-eigenvalue normal of the inlier covariance) on every pixel within ``inlier_m`` of
   the RANSAC plane. The normal is oriented towards the camera, so ``height = n . p + d`` is positive above the
   table.
3. The blob is every valid pixel more than ``min_height_m`` (3 mm, plan section 8) above the plane.
4. The crop is a square window of fixed physical side ``crop_side_m`` at the depth of the blob's 3D centroid,
   centred on the centroid's projection, resampled bilinearly to ``crop_out_px``; pixels outside the image take
   the constant ``fill`` colour. Scale is therefore constant up to the small depth spread of the placements, so
   apparent size stays a real cue.
5. Blob geometry for the classical matcher: in-plane principal extents (major, minor) and the height above the
   plane, from the 0.5-99.5 percentiles of the blob points, in millimetres. These describe the visible surface
   only (the far side and the lowest 3 mm are not seen).
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

TRUTH_FREE = True
DEFAULT_SPEC = {
    "min_height_m": 0.003,
    "ransac_iters": 128,
    "ransac_stride": 8,
    "ransac_seed": 0,
    "inlier_m": 0.001,
    "min_inlier_fraction": 0.25,
    "crop_side_m": 0.30,
    "crop_out_px": 384,
    "fill": [128, 128, 128],
    "extent_percentiles": [0.5, 99.5],
}


class PlaneError(RuntimeError):
    """No usable table plane in the depth image."""


def spec_sha256(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def intrinsics(camera: dict) -> tuple[float, float, float, float, int, int]:
    """(fx, fy, cx, cy, width, height) from a camera description."""
    return (float(camera["fx"]), float(camera["fy"]), float(camera["cx"]), float(camera["cy"]),
            int(camera["width"]), int(camera["height"]))


def depth_metres(depth_u16: np.ndarray, camera: dict) -> np.ndarray:
    """Float depth in metres; 0 (invalid) and 65535 (saturated) become NaN."""
    d = np.asarray(depth_u16)
    if d.dtype != np.uint16 or d.ndim != 2:
        raise ValueError(f"depth must be uint16 HxW, got {d.dtype} {d.shape}")
    unit = float(camera["depth"]["unit_m"])
    out = d.astype(np.float64) * unit
    out[(d == 0) | (d == np.iinfo(np.uint16).max)] = np.nan
    return out


def backproject(depth_m: np.ndarray, camera: dict) -> np.ndarray:
    """(H, W, 3) camera-frame points (NaN where depth is invalid)."""
    fx, fy, cx, cy, w, h = intrinsics(camera)
    if depth_m.shape != (h, w):
        raise ValueError(f"depth shape {depth_m.shape} != camera ({h}, {w})")
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.stack([(u - cx) / fx * depth_m, (v - cy) / fy * depth_m, depth_m], axis=-1)


def _plane_from(p0, p1, p2):
    n = np.cross(p1 - p0, p2 - p0)
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        return None
    n = n / norm
    return n, -float(n @ p0)


def fit_plane(points: np.ndarray, spec: dict = DEFAULT_SPEC) -> dict:
    """Table plane ``n . p + d = 0`` with ``n`` towards the camera (d > 0); PlaneError if too few inliers."""
    valid = np.isfinite(points[..., 2])
    stride = int(spec["ransac_stride"])
    sub = points[::stride, ::stride][valid[::stride, ::stride]]
    if len(sub) < 3:
        raise PlaneError("fewer than 3 valid depth samples")
    rng = np.random.default_rng(int(spec["ransac_seed"]))
    tol = float(spec["inlier_m"])
    best, best_count = None, -1
    for _ in range(int(spec["ransac_iters"])):
        idx = rng.choice(len(sub), size=3, replace=False)
        plane = _plane_from(*sub[idx])
        if plane is None:
            continue
        count = int(np.count_nonzero(np.abs(sub @ plane[0] + plane[1]) < tol))
        if count > best_count:
            best, best_count = plane, count
    if best is None:
        raise PlaneError("no non-degenerate RANSAC sample")
    pts = points[valid]
    inl = pts[np.abs(pts @ best[0] + best[1]) < tol]
    frac = len(inl) / max(1, len(pts))
    if frac < float(spec["min_inlier_fraction"]):
        raise PlaneError(f"inlier fraction {frac:.3f} < {spec['min_inlier_fraction']}")
    centre = inl.mean(axis=0)
    cov = np.cov((inl - centre).T)
    evals, evecs = np.linalg.eigh(cov)
    n = evecs[:, 0]
    d = -float(n @ centre)
    if d < 0:  # the camera (origin) must be on the positive side
        n, d = -n, -d
    resid = inl @ n + d
    return {"normal": [float(x) for x in n], "d": d, "inlier_fraction": float(frac),
            "rms_m": float(np.sqrt(np.mean(resid ** 2))), "n_inliers": int(len(inl))}


def heights(points: np.ndarray, plane: dict) -> np.ndarray:
    """Signed height above the plane (m), NaN where depth is invalid."""
    return points @ np.asarray(plane["normal"]) + float(plane["d"])


def _in_plane_basis(normal) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=float)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - (seed @ n) * n
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def project(point_cam, camera: dict) -> tuple[float, float]:
    fx, fy, cx, cy, _, _ = intrinsics(camera)
    x, y, z = (float(c) for c in point_cam)
    return cx + fx * x / z, cy + fy * y / z


def blob_geometry(points: np.ndarray, mask: np.ndarray, plane: dict, camera: dict, spec: dict = DEFAULT_SPEC) -> dict:
    """Pixel count, 3D centroid, its projection, pixel box and the in-plane / height extents (mm) of the blob."""
    n_px = int(np.count_nonzero(mask))
    _, _, _, _, w, h = intrinsics(camera)
    if n_px == 0:
        return {"n_px": 0}
    pts = points[mask]
    centroid = pts.mean(axis=0)
    rows, cols = np.nonzero(mask)
    lo, hi = (float(x) for x in spec["extent_percentiles"])
    e1, e2 = _in_plane_basis(plane["normal"])
    uv = np.stack([pts @ e1, pts @ e2], axis=1)
    uv -= uv.mean(axis=0)
    if n_px >= 3:
        _, evecs = np.linalg.eigh(np.cov(uv.T))
        axes = evecs[:, ::-1]  # major first
    else:
        axes = np.eye(2)
    proj = uv @ axes
    ext = [float(np.percentile(proj[:, i], hi) - np.percentile(proj[:, i], lo)) for i in range(2)]
    hts = pts @ np.asarray(plane["normal"]) + float(plane["d"])
    u_c, v_c = project(centroid, camera)
    return {
        "n_px": n_px,
        "centroid_cam_m": [float(x) for x in centroid],
        "centroid_px": [u_c, v_c],
        "bbox_px": [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())],
        "touches_border": bool(cols.min() == 0 or rows.min() == 0 or cols.max() == w - 1 or rows.max() == h - 1),
        "major_mm": 1000.0 * max(ext),
        "minor_mm": 1000.0 * min(ext),
        "height_mm": 1000.0 * float(np.percentile(hts, hi)),
    }


def crop_window(blob: dict, camera: dict, spec: dict = DEFAULT_SPEC) -> dict:
    """The fixed physical window: side ``crop_side_m`` at the blob centroid's depth, centred on its projection."""
    fx, fy, _, _, _, _ = intrinsics(camera)
    z = float(blob["centroid_cam_m"][2])
    side_px = float(spec["crop_side_m"]) * 0.5 * (fx + fy) / z
    u_c, v_c = blob["centroid_px"]
    return {"centre_px": [float(u_c), float(v_c)], "side_px": side_px, "depth_m": z,
            "px_per_m_out": float(spec["crop_out_px"]) / float(spec["crop_side_m"])}


def resample(rgb: np.ndarray, window: dict, out_px: int, fill) -> np.ndarray:
    """Bilinear resample of the square ``window`` to ``out_px`` x ``out_px`` (outside the image: ``fill``)."""
    img = np.asarray(rgb, dtype=np.float64)
    h, w = img.shape[:2]
    side = float(window["side_px"])
    u_c, v_c = window["centre_px"]
    t = (np.arange(out_px, dtype=np.float64) + 0.5) / out_px - 0.5  # in (-0.5, 0.5)
    us = u_c + t * side
    vs = v_c + t * side
    uu, vv = np.meshgrid(us, vs)
    u0, v0 = np.floor(uu), np.floor(vv)
    fu, fv = uu - u0, vv - v0
    u0i, v0i = u0.astype(np.int64), v0.astype(np.int64)
    fill_arr = np.asarray(fill, dtype=np.float64)
    out = np.zeros((out_px, out_px, img.shape[2]), dtype=np.float64)
    for du, dv, wgt in ((0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)), (0, 1, (1 - fu) * fv), (1, 1, fu * fv)):
        uu_i, vv_i = u0i + du, v0i + dv
        inside = (uu_i >= 0) & (uu_i < w) & (vv_i >= 0) & (vv_i < h)
        src = np.empty((out_px, out_px, img.shape[2]), dtype=np.float64)
        src[:] = fill_arr
        src[inside] = img[vv_i[inside], uu_i[inside]]
        out += wgt[..., None] * src
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)


def analyse(rgb: np.ndarray, depth_u16: np.ndarray, camera: dict, spec: dict = DEFAULT_SPEC) -> dict:
    """Plane, blob mask, blob geometry and (when there is a blob) the crop; never raises on a missing blob.

    Returns ``{"plane", "blob", "mask", "window", "crop", "status"}`` with ``status`` in {"ok", "no_plane",
    "no_blob"}; ``crop`` is None unless status is "ok".
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be uint8 HxWx3, got {rgb.dtype} {rgb.shape}")
    pts = backproject(depth_metres(depth_u16, camera), camera)
    if rgb.shape[:2] != pts.shape[:2]:
        raise ValueError("rgb and depth sizes differ")
    try:
        plane = fit_plane(pts, spec)
    except PlaneError as exc:
        return {"status": "no_plane", "reason": str(exc), "plane": None, "blob": {"n_px": 0}, "mask": None,
                "window": None, "crop": None}
    with np.errstate(invalid="ignore"):
        mask = heights(pts, plane) > float(spec["min_height_m"])
    mask &= np.isfinite(pts[..., 2])
    blob = blob_geometry(pts, mask, plane, camera, spec)
    if blob["n_px"] == 0:
        return {"status": "no_blob", "reason": "no pixel above the plane threshold", "plane": plane, "blob": blob,
                "mask": mask, "window": None, "crop": None}
    window = crop_window(blob, camera, spec)
    crop = resample(rgb, window, int(spec["crop_out_px"]), spec["fill"])
    return {"status": "ok", "reason": None, "plane": plane, "blob": blob, "mask": mask, "window": window,
            "crop": crop}


def summary(result: dict) -> dict:
    """JSON-safe part of ``analyse``'s result (no arrays)."""
    return {k: result[k] for k in ("status", "reason", "plane", "blob", "window") if k in result}


def angle_deg(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))
