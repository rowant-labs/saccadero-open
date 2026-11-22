"""The camera from the robot's own arm (item 2), server side: only the perception server holds the arm model.

M2 (perceived poses, grounded instruction): perception in the loop. TRUTH-FREE: perception-server side; it reads the
depth frames and proprioception the robot sends, and the server's Panda model, nothing else.

The colour line reads its camera from constants stored per suite (``colour_cameras.json``). When the camera has moved,
everything the robot measures is misplaced. One request, ``arm_cam``, on when the server's params carry ``arm_cam`` (a
dict of overrides of ``ARM_CAM``). The robot's mode is ``gate_fit`` (``rowant_lab/pro2/arm_calib.py``):

* **Frames:** three depth frames of the robot's own first motion, each with the assumed camera and the proprioception.
  In each frame the arm's moving pixels are those nearer than the frames' per-pixel median by ``mask_m`` (and with at
  least ``mask_min_neighbours`` such neighbours); the links used are those a point of which moved over ``moved_m``.
* **The gate** (``gate``): the moving pixels, lifted at the assumed camera, against the moving links. The camera is kept
  when their trimmed cost is at most ``motion_cost_m`` and at least ``motion_share_min`` of them lie within 1 cm; with
  fewer than ``motion_min_px`` pixels in a frame the gate does not decide (kept).
* **The fit** (``fit``, only when the gate fires):
  1. **Plane lock** (``plane_lock``): the largest plane in the first frame (RANSAC, the arm's pixels left out) is the
     support. The camera is tilted so that plane is horizontal and raised to the plane's distance above the workcell's
     support height that best matches the assumed camera (``support_heights``). The colour line's net carries the stored
     camera's geometry in its depth, so this keeps the camera consistent with the net; only the turn about the vertical
     and the horizontal position remain.
  2. **ICP:** turns every ``yaw_step_deg`` within ``yaw_span_deg``, each at the assumed horizontal position and at the one
     that maps the pixels' centroid onto the model's; point-to-plane ICP (``_Frames.icp_level``) from the best starts.
     A pixel only matches a model point whose face looks at the camera being tried (otherwise the links' tops match
     their own undersides when the camera is too high).
  3. **Render and compare** (``refine_rc``): the whole arm drawn into every frame's depth (``rc_cost``: silhouette and
     depth at once), Nelder-Mead over the same three numbers from the ICP's camera.
  4. **Accepted** when the render cost is at most ``rc_accept_ratio`` of the assumed camera's, the fitted camera explains
     the arm's moving pixels (their trimmed cost at it at most ``accept_cost_m``), and it is at least ``min_shift_m`` or
     ``min_turn_deg`` from the assumed one (nearer is within the fit's precision on net depth) and at most
     ``max_shift_m`` and ``max_turn_deg``. The refinement may not draw fewer than half the assumed camera's cells
     (``rc_cost``'s ``min_cells``), so it cannot lower its cost by moving the arm out of view.
  Without a level plane (``plane_lock`` off, or none found) the fit is the free six-degree one: rotations of the assumed
  camera about its own axes (``coarse_*``), then ICP; accepted on the ICP cost (``accept_cost_m``, ``accept_ratio``).
* ``check`` and ``fit`` are also served on their own (development: one-frame agreement of the drawn arm with the depth,
  and the fit without the gate).

Development record (2026-11-12, world-2 development scenes, Mac, the colour net's depth): the gate's trimmed cost was
1.7-2.5 mm with the right camera and 7-26 mm with a moved one; the fitted turned cameras were 1.4-2.9 cm and 0.7-1.6
degrees from the truth (stored: 27 cm, 15 degrees). The second world's registration, amendment W2-CAM.

Arm model: area-weighted samples of the Panda's visual meshes, each with its face's outward normal (``model_samples``;
the exported meshes are wound outward: positive volume on every large mesh, checked 2026-11-12).
"""

from __future__ import annotations

import time

import numpy as np

ARM_CAM = {
    "spacing_m": 0.004,  # model samples
    "stride": 4,  # the check and the fit read every 4th row and column of the frame
    "check_tol_m": 0.02,
    "check_min_agree": 0.6,
    "check_min_cells": 150,
    "mask_m": 0.02,
    "mask_min_neighbours": 3,  # a mask pixel needs this many mask pixels among its 8 neighbours (net speckle)
    "moved_m": 0.02,  # a link is used when a point of it moved this far between frames
    "trim": 0.8,
    "cap_m": 0.05,  # a pixel with no facing model point near it costs this
    "knn": 8,
    "coarse_deg": 20.0,
    "coarse_step_deg": 5.0,
    "coarse_roll_deg": 5.0,
    "sub_n": 600,  # pixels per frame in the coarse stage and the short ICP
    "short_starts": 25,
    "short_iters": 10,
    "full_starts": 6,
    "full_iters": 40,
    "accept_cost_m": 0.008,
    "accept_ratio": 0.8,
    "max_shift_m": 0.5,  # sanity: an estimate farther than this from the assumed camera is refused
    "max_turn_deg": 35.0,
    "min_shift_m": 0.03,  # and one nearer than the fit's precision on net depth (both below) changes nothing
    "min_turn_deg": 1.0,
    "plane_lock": True,  # the support plane in view fixes the camera's tilt and height (see ``fit``)
    "plane_tol_m": 0.01,
    "plane_iters": 200,
    "plane_max_tilt_deg": 30.0,  # the plane's normal at the assumed camera must be this near vertical
    "yaw_span_deg": 40.0,
    "yaw_step_deg": 2.0,
    "check_plane_height_m": 0.015,
    "check_plane_tilt_deg": 1.5,
    "rc_spacing_m": 0.008,  # render-and-compare: the whole arm drawn into each frame (``rc_cost``)
    "rc_tau_m": 0.03,
    "rc_yaw_starts": (-2.0, 0.0, 2.0),
    "rc_step": (1.5, 0.03, 0.03),  # the first simplex's steps: yaw deg, x m, y m
    "rc_maxiter": 150,
    "rc_accept_ratio": 0.78,  # the estimate is used when its render cost is at most this share of the assumed camera's
    "gate_drop": 0.25,  # check's render gate (diagnostic): moved when the one-frame refinement lowers the render cost
    "gate_shift_m": 0.03,  # by this share and moves the camera this far or turns it this much
    "gate_turn_deg": 1.5,
    "motion_cost_m": 0.006,  # the motion gate (``gate``): the assumed camera explains the arm's moving pixels when their
    "motion_share_min": 0.75,  # trimmed cost is at most this and this share of them is within 1 cm of the model
    "motion_min_px": 150,  # pixels per frame; with fewer the gate does not decide (the camera is kept)
    # item 2b (2026-11-13), off by default: the focal length too. The plane-locked search runs at each factor of the
    # assumed focal length in ``focal_scales``, then a golden-section search between the best one's neighbours
    # (``focal_refine`` steps), each judged by the arm's cost at its own plane lock; the render refinement then runs
    # at the factor found. A fit that changes the focal length by at least ``min_focal`` counts as a change even at
    # the assumed pose.
    "focal_scales": None,
    "focal_refine": 6,
    "min_focal": 0.03,
    # item 2b, off by default: overrides of these params for the second try (request mode ``gate_fit2``: the gate and
    # the fit again, on depth from the colour line's second net, sent only after the first fit was refused)
    "second": None,
    # item 4 (2026-11-14): the joint readings' offsets from the arm, the camera taken as right (``fit_joints``; request
    # modes ``joints`` and ``gate_joints``, sent only when the robot asks: ``arm_calib``'s ``arm_joints`` option)
    "joint_iters": 25,
    "joint_step_rad": 1e-4,  # the numerical Jacobian's step
    "joint_max_step_deg": 2.0,  # one iteration moves each offset at most this much
    "joint_accept_cost_m": 0.008,  # accepted when the arm's trimmed cost at the fitted readings is at most this,
    "joint_accept_ratio": 0.6,  # at most this share of the reported readings' cost,
    "joint_min_deg": 0.5,  # and the largest offset is at least this (nearer changes nothing) and at most the next
    "joint_max_deg": 15.0,
    # 2026-11-16: the readings' own offsets from the readings alone, no image (``fit_readings``; request mode
    # ``readings_fit``, sent only when the robot asks: ``arm_calib``'s ``arm_joints["readings_fit"]``)
    "read_max_n": 30,  # steps used, spread evenly over those sent
    "read_iters": 12,
    "read_step_rad": 1e-6,  # the numerical Jacobian's step
    "read_rms_m": 5e-5,  # accepted when the two offsets leave at most this at every step used (root mean square),
    "read_cond": 5e3,  # no other offsets would explain the steps as well (the fit's condition number, every column
    # at unit length, at most this: 380 to 1,300 on bent or diagonal motions of 17 to 40 steps, 14,000 and far more
    # on 12 cm along one axis; a training episode's own readings pass from its sixth step, 2,700, and read 625 at 11),
    "read_min_gap_m": 0.002,  # the readings disagreed by at least this before it,
    "read_max_rad": 0.5,  # and no joint offset is beyond this, nor the hand's beyond the next
    "read_max_m": 0.4,
    "read_max_tau_s": 0.05,  # the hand reading's lag behind the joint readings (found by the fit; one control step at most)
    # 2026-11-15: the image itself shifted or zoomed, the camera where it was (``fit_image``; request mode
    # ``gate_image``, sent only when the robot asks: ``arm_calib``'s ``arm_image`` option). Pixels, at the frame's size.
    "img_max_shift": 0.25,  # the search's shift, as a share of the image width, each way
    # measured 2026-11-15 (probes/s16_image.py): S16 shrinks the picture to 0.88 (Object, Spatial) or 0.67 (Goal) of its
    # size and shifts it by 5 and 10% of its width
    "img_zooms": (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4),
    "img_step_px": 16.0,  # the coarse grid's shift step
    "img_dilate": 1,  # grid cells of tolerance added around the changed pixels and around the drawn change
    "img_min_iou": 0.35,  # accepted when the changed pixels' overlap with the drawn change (intersection over union)
    "img_iou_gain": 1.2,  # is at least this, and at least this many times the overlap with the image as it came,
    "img_min_shift_px": 12.0,  # and the image moved at least this far or was scaled by at least the next (on plain
    "img_min_zoom": 0.03,  # frames the fit read up to 8 px and 0.02 off: the stored camera's own error)
    "img_min_px": 150,  # changed pixels (on the stride grid) needed to try
}
SEED = 20261112


def params(p: dict | None = None) -> dict:
    out = dict(ARM_CAM)
    out.update(p or {})
    return out


# ------------------------------------------------------------------------------------------------ the arm model
def model_samples(panda, spacing_m: float, seed: int = SEED) -> dict:
    """{body: (P, N)}: area-weighted samples of every visual mesh at about ``spacing_m``, body frame, each with its face's
    unit normal. Unlike ``Panda.surface_samples`` no mesh vertex is added (the meshes have 60-110 thousand per link)."""
    rng = np.random.default_rng(seed)
    out = {}
    for n in panda.order:
        Ps, Ns = [], []
        for key in panda.bodies[n]["geoms"]:
            V, F = panda.arrays[f"{key}__V"], panda.arrays[f"{key}__F"]
            c = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
            a = 0.5 * np.linalg.norm(c, axis=1)
            if a.sum() <= 0:
                continue
            k = int(np.ceil(a.sum() / spacing_m ** 2))
            fi = rng.choice(len(F), size=k, p=a / a.sum())
            r1, r2 = rng.random(k), rng.random(k)
            s1 = np.sqrt(r1)
            P = (V[F[fi, 0]] * (1 - s1)[:, None] + V[F[fi, 1]] * (s1 * (1 - r2))[:, None]
                 + V[F[fi, 2]] * (s1 * r2)[:, None])
            Ps.append(P)
            Ns.append(c[fi] / np.maximum(np.linalg.norm(c[fi], axis=1, keepdims=True), 1e-12))
        if Ps:
            out[n] = (np.concatenate(Ps), np.concatenate(Ns))
    return out


def arm_points(panda, samples: dict, q, g, bodies=None) -> tuple[np.ndarray, np.ndarray]:
    """World points and normals of the arm model at joints ``q`` and gripper ``g``."""
    poses = panda.fk(q, g)
    P, N = [], []
    for b, (Pb, Nb) in samples.items():
        if bodies is not None and b not in bodies:
            continue
        R, t = poses[b]
        P.append(Pb @ R.T + t)
        N.append(Nb @ R.T)
    return np.concatenate(P), np.concatenate(N)


def moved_bodies(panda, samples: dict, qs: list, gs: list, moved_m: float) -> list:
    """The links a point of which moved more than ``moved_m`` between any frame and the first."""
    poses = [panda.fk(q, g) for q, g in zip(qs, gs)]
    out = []
    for b, (Pb, _Nb) in samples.items():
        pts = Pb[np.linspace(0, len(Pb) - 1, min(len(Pb), 64)).astype(int)]
        R0, t0 = poses[0][b]
        ref = pts @ R0.T + t0
        d = max(float(np.linalg.norm(pts @ R.T + t - ref, axis=1).max()) for R, t in (P[b] for P in poses[1:]))
        if d > moved_m:
            out.append(b)
    return out


# ------------------------------------------------------------------------------------------------ geometry
def depth_m(fr: dict) -> np.ndarray:
    d = np.asarray(fr["depth_u16"])
    z = d.astype(np.float64) * float(fr["depth_scale_mm"]) * 1e-3
    z[d == 0] = np.nan
    return z


def grid(z: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every ``stride``-th row and column (from ``stride // 2``) and their pixel coordinates (u = column index, as the
    server's ``_frame_arrays``)."""
    o = stride // 2
    v, u = np.mgrid[o:z.shape[0]:stride, o:z.shape[1]:stride]
    return z[o::stride, o::stride], u.astype(np.float64), v.astype(np.float64)


def lift(z, u, v, K) -> np.ndarray:
    return np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)


def rot(axis, deg: float) -> np.ndarray:
    a = np.radians(deg)
    x = np.asarray(axis, float) / np.linalg.norm(axis)
    S = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    return np.eye(3) + np.sin(a) * S + (1 - np.cos(a)) * S @ S


def kabsch(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t minimising |R A + t - B| (rows are points)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cb - R @ ca


def pose_diff(Ta, Tb) -> tuple[float, float]:
    """(centre distance m, rotation angle deg) between two camera poses."""
    dR = Ta[:3, :3].T @ Tb[:3, :3]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), ang


def _T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


# ------------------------------------------------------------------------------------------------ the support plane
def support_heights(workcell: dict) -> list:
    """World heights of the horizontal support surfaces the robot's workcell model names (floor planes, box tops)."""
    out = set()
    for a in workcell.get("arena", []):
        if a.get("type") == "plane" and abs(float(np.asarray(a["normal"], float)[2])) > 0.99:
            out.add(round(float(a["point"][2]), 6))
        elif a.get("type") == "box":
            out.add(round(float(a["centre"][2]) + float(a["half"][2]), 6))
    return sorted(out)


def fit_plane(X: np.ndarray, tol: float, iters: int, seed: int = SEED):
    """The plane with the most points within ``tol`` (RANSAC, then a least-squares refit on its points), camera frame:
    ``n . x + d = 0`` with the camera on the positive side (``d > 0``). Returns (n, d, share of points on it)."""
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(int(iters)):
        i = rng.choice(len(X), 3, replace=False)
        n = np.cross(X[i[1]] - X[i[0]], X[i[2]] - X[i[0]])
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            continue
        n = n / nn
        inl = np.abs(X @ n - n @ X[i[0]]) <= tol
        k = int(inl.sum())
        if k > best_n:
            best, best_n = inl, k
    P = X[best]
    c = P.mean(0)
    n = np.linalg.svd(P - c, full_matrices=False)[2][2]
    d = -float(n @ c)
    if d < 0:
        n, d = -n, -d
    return n, d, float((np.abs(X @ n + d) <= tol).mean())


def plane_of_frame(fr: dict, p: dict, exclude=None, K=None):
    z, u, v = grid(depth_m(fr), int(p["stride"]))
    ok = np.isfinite(z)
    if exclude is not None:
        ok &= ~exclude
    X = lift(z[ok], u[ok], v[ok], np.asarray(fr["K"] if K is None else K, float))
    return fit_plane(X, float(p["plane_tol_m"]), int(p["plane_iters"]))


def level(T0: np.ndarray, n_c: np.ndarray, d_c: float, supports: list, p: dict):
    """The assumed camera tilted so the fitted plane is horizontal, at the height that puts the plane on the support it
    best matches. Returns (R0, t_z, z_support, tilt deg, height error m) or None when the plane is far from level."""
    m = T0[:3, :3] @ n_c
    tilt = float(np.degrees(np.arccos(np.clip(m[2], -1, 1))))
    if tilt > float(p["plane_max_tilt_deg"]) or not supports:
        return None
    ax = np.cross(m, [0.0, 0.0, 1.0])
    s_ = float(np.linalg.norm(ax))
    G = rot(ax / s_, tilt) if s_ > 1e-12 else np.eye(3)
    h0 = float(T0[2, 3]) - d_c  # the plane's world height at the assumed camera (ignoring its tilt)
    z_sup = min(supports, key=lambda z_: abs(h0 - z_))
    return G @ T0[:3, :3], z_sup + d_c, z_sup, tilt, h0 - z_sup


# ------------------------------------------------------------------------------------------------ render and compare
def rc_cost(T: np.ndarray, Zg: list, arms: list, K: np.ndarray, stride: int, tau: float, min_cells: int = 0) -> float:
    """The whole arm (every link, faces that look at the camera, nearest per cell) drawn into each frame's depth grid
    ``Zg`` at camera ``T``; the mean over drawn cells of min(r^2, tau^2), r = measured - drawn depth (a missing
    measurement counts tau). Silhouette and depth at once: a link drawn over the background costs tau^2. A camera that
    draws fewer than ``min_cells`` cells in all (``refine_rc``: half the assumed camera's) costs tau^2 outright, so moving
    the arm out of view never lowers the mean; a view that sees only part of the arm is not penalised for the rest."""
    tot, n = 0.0, 0
    c = T[:3, 3]
    for z, (Pw, Nw) in zip(Zg, arms):
        h, w = z.shape
        face = ((c - Pw) * Nw).sum(1) > 0
        X = (Pw[face] - c) @ T[:3, :3]
        X = X[X[:, 2] > 0.05]
        j = np.rint((K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2] - stride // 2) / stride).astype(int)
        i = np.rint((K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2] - stride // 2) / stride).astype(int)
        ins = (i >= 0) & (i < h) & (j >= 0) & (j < w)
        zb = np.full(h * w, np.inf)
        np.minimum.at(zb, i[ins] * w + j[ins], X[ins, 2])
        dr = np.isfinite(zb)
        r = z.ravel()[dr] - zb[dr]
        r = np.where(np.isfinite(r), r, tau)
        tot += float(np.minimum(r * r, tau * tau).sum())
        n += int(dr.sum())
    if n == 0 or n < min_cells:
        return tau * tau
    return tot / n


def rc_cells(T: np.ndarray, arms: list, K: np.ndarray, shape: tuple, stride: int) -> int:
    """The number of cells the arm draws at camera ``T`` over all frames (``rc_cost``'s denominator)."""
    n = 0
    c = T[:3, 3]
    h, w = shape
    for Pw, Nw in arms:
        face = ((c - Pw) * Nw).sum(1) > 0
        X = (Pw[face] - c) @ T[:3, :3]
        X = X[X[:, 2] > 0.05]
        j = np.rint((K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2] - stride // 2) / stride).astype(int)
        i = np.rint((K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2] - stride // 2) / stride).astype(int)
        ins = (i >= 0) & (i < h) & (j >= 0) & (j < w)
        n += int(np.unique(i[ins] * w + j[ins]).size)
    return n


def refine_rc(T: np.ndarray, Zg: list, arms: list, K: np.ndarray, p: dict,
              min_cells: int | None = None) -> tuple[np.ndarray, float]:
    """Nelder-Mead on ``rc_cost`` over a turn about the world vertical and a horizontal shift (tilt and height held),
    from each of ``rc_yaw_starts`` about ``T``. Returns the best camera and its cost."""
    from scipy.optimize import minimize

    s, tau = int(p["stride"]), float(p["rc_tau_m"])
    dy, dx_, dy_ = (float(v) for v in p["rc_step"])
    m = int(0.5 * rc_cells(T, arms, K, Zg[0].shape, s)) if min_cells is None else int(min_cells)

    def T_of(x):
        return _T(rot([0, 0, 1], x[0]) @ T[:3, :3], T[:3, 3] + np.array([x[1], x[2], 0.0]))

    def f(x):
        return rc_cost(T_of(x), Zg, arms, K, s, tau, m)

    best = None
    for y0 in p["rc_yaw_starts"]:
        y0 = float(y0)
        r_ = minimize(f, [y0, 0.0, 0.0], method="Nelder-Mead",
                      options={"xatol": 1e-3, "fatol": 1e-10, "maxiter": int(p["rc_maxiter"]),
                               "initial_simplex": [[y0, 0, 0], [y0 + dy, 0, 0], [y0, dx_, 0], [y0, 0, dy_]]})
        if best is None or r_.fun < best.fun:
            best = r_
    return T_of(best.x), float(best.fun)


def scaled_K(K: np.ndarray, f: float) -> np.ndarray:
    """``K`` with its focal length times ``f`` (principal point kept)."""
    Ks = np.array(K, dtype=np.float64, copy=True)
    Ks[0, 0] *= float(f)
    Ks[1, 1] *= float(f)
    return Ks


def _rc_data(panda, rc_samples: dict, frames: list, qs: list, gs: list, stride: int):
    Zg = [grid(depth_m(fr), stride)[0] for fr in frames]
    arms = [arm_points(panda, rc_samples, q, g) for q, g in zip(qs, gs)]
    return Zg, arms


# ------------------------------------------------------------------------------------------------ check
def check(panda, samples: dict, fr: dict, q, g, p: dict, supports: list | None = None,
          rc_samples: dict | None = None) -> dict:
    """Does the frame's camera explain the arm? (one frame, no motion)

    With ``rc_samples`` the decision is the render gate: the assumed camera is refined on this one frame
    (``refine_rc``); it is "moved" (``ok`` false) when that lowers the render cost by at least ``gate_drop`` and moves
    the camera at least ``gate_shift_m`` or turns it ``gate_turn_deg``. The cell agreement and plane numbers are then
    reported only."""
    T, K = np.asarray(fr["T_world_from_camera"], float), np.asarray(fr["K"], float)
    s = int(p["stride"])
    z, _u, _v = grid(depth_m(fr), s)
    h, w = z.shape
    Pw, Nw = arm_points(panda, samples, q, g)
    c = T[:3, 3]
    face = ((c - Pw) * Nw).sum(1) > 0
    X = (Pw[face] - c) @ T[:3, :3]
    X = X[X[:, 2] > 0.05]
    uu = K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2]
    vv = K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]
    j = np.rint((uu - s // 2) / s).astype(int)
    i = np.rint((vv - s // 2) / s).astype(int)
    inside = (i >= 0) & (i < h) & (j >= 0) & (j < w)
    zb = np.full(h * w, np.inf)
    np.minimum.at(zb, i[inside] * w + j[inside], X[inside, 2])
    drawn = np.isfinite(zb)
    r = z.ravel()[drawn] - zb[drawn]
    ok_r = np.isfinite(r)
    tol = float(p["check_tol_m"])
    n_agree = int((ok_r & (np.abs(r) <= tol)).sum())
    n_behind = int((ok_r & (r > tol)).sum())
    n_front = int((ok_r & (r < -tol)).sum())
    share = n_agree / max(n_agree + n_behind, 1)
    n_cells = int(drawn.sum())
    rv = r[ok_r]
    out = {"ok": bool(share >= p["check_min_agree"] and n_cells >= p["check_min_cells"]), "agree": round(share, 4),
           "cells": n_cells, "n_agree": n_agree, "n_behind": n_behind, "n_front": n_front,
           "med_abs_m": round(float(np.median(np.abs(rv))), 5) if len(rv) else None,
           "outside": round(1.0 - float(inside.mean()) if len(inside) else 1.0, 4)}
    if supports is not None:
        n_c, d_c, share_p = plane_of_frame(fr, p)
        lv = level(T, n_c, d_c, supports, p)
        if lv is None:
            out.update(plane_ok=False, plane_share=round(share_p, 3))
        else:
            out.update(plane_ok=bool(lv[3] <= p["check_plane_tilt_deg"] and abs(lv[4]) <= p["check_plane_height_m"]),
                       plane_tilt_deg=round(lv[3], 3), plane_height_err_m=round(lv[4], 4), plane_share=round(share_p, 3))
        out["ok"] = bool(out["ok"] and out["plane_ok"])
    if rc_samples is not None:
        Zg, arms = _rc_data(panda, rc_samples, [fr], [q], [g], s)
        c0 = rc_cost(T, Zg, arms, K, s, float(p["rc_tau_m"]))
        T1, c1 = refine_rc(T, Zg, arms, K, p)
        shift, turn = pose_diff(T1, T)
        moved = (c1 <= (1.0 - float(p["gate_drop"])) * c0
                 and (shift >= float(p["gate_shift_m"]) or turn >= float(p["gate_turn_deg"])))
        out.update(ok=not moved, gate={"rc0_mm2": round(c0 * 1e6, 2), "rc1_mm2": round(c1 * 1e6, 2),
                                       "shift_m": round(shift, 4), "turn_deg": round(turn, 3)})
    return out


# ------------------------------------------------------------------------------------------------ fit
def arm_masks(Z: np.ndarray, p: dict) -> np.ndarray:
    """(n, h, w) bool: the pixels of each frame nearer than the frames' per-pixel median by ``mask_m``, kept when enough
    of their 8 neighbours are mask pixels too."""
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(Z, axis=0)
        M = np.isfinite(Z) & np.isfinite(med)[None] & (Z < med[None] - float(p["mask_m"]))
    k = int(p["mask_min_neighbours"])
    if k > 0:
        Mi = M.astype(np.int16)
        P = np.pad(Mi, ((0, 0), (1, 1), (1, 1)))
        nb = sum(P[:, 1 + dy:P.shape[1] - 1 + dy, 1 + dx:P.shape[2] - 1 + dx]
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
        M &= nb >= k
    return M


class _Frames:
    """The fit's data: per frame, the lifted arm pixels (camera frame) and the moving links' model (world, a KD-tree).

    A pixel's residual is its distance to the tangent plane of the nearest model point whose face looks at the camera
    being tried, or the distance to that point when it is farther than ``near_m`` (the pixel is off the model's extent);
    capped at ``cap_m``. The plane distance removes the floor that the samples' spacing puts under point distances."""

    def __init__(self, src: list, dst: list, p: dict):
        from scipy.spatial import cKDTree

        self.src, self.p = src, p
        self.P = [d[0] for d in dst]
        self.N = [d[1] for d in dst]
        self.trees = [cKDTree(P) for P in self.P]
        n = int(p["sub_n"])
        self.sub = [s[np.linspace(0, len(s) - 1, min(len(s), n)).astype(int)] for s in src]
        self.near = 2.0 * float(p["spacing_m"])

    def residuals(self, i: int, T: np.ndarray, sub: bool):
        s = (self.sub if sub else self.src)[i]
        Xw = s @ T[:3, :3].T + T[:3, 3]
        k = int(self.p["knn"])
        d, idx = self.trees[i].query(Xw, k=k)
        facing = ((T[:3, 3] - self.P[i][idx]) * self.N[i][idx]).sum(-1) > 0
        d = np.where(facing, d, np.inf)
        j = np.argmin(d, axis=1)
        r = np.arange(len(d))
        d, idx = d[r, j], idx[r, j]
        ok = np.isfinite(d)
        e = np.full(len(d), float(self.p["cap_m"]))
        if ok.any():
            dp = np.abs(((Xw[ok] - self.P[i][idx[ok]]) * self.N[i][idx[ok]]).sum(1))
            e[ok] = np.minimum(np.where(d[ok] <= self.near, dp, d[ok]), float(self.p["cap_m"]))
        return e, Xw, idx, ok

    def cost(self, T: np.ndarray, sub: bool = False) -> float:
        tot, n = 0.0, 0
        trim = float(self.p["trim"])
        for i in range(len(self.src)):
            if not len(self.src[i]):
                continue
            e = self.residuals(i, T, sub)[0]
            k = max(1, int(trim * len(e)))
            tot += float(np.partition(e, k - 1)[:k].sum())
            n += k
        return tot / max(n, 1)

    def icp(self, T: np.ndarray, iters: int, sub: bool) -> np.ndarray:
        """Point-to-plane ICP over the trimmed pairs of all frames (one damped Gauss-Newton step per iteration)."""
        trim = float(self.p["trim"])
        for _ in range(iters):
            A, B, Nn = [], [], []
            for i in range(len(self.src)):
                if not len(self.src[i]):
                    continue
                e, Xw, idx, ok = self.residuals(i, T, sub)
                keep = ok & (e <= np.quantile(e, trim)) & (e < float(self.p["cap_m"]))
                A.append(Xw[keep])
                B.append(self.P[i][idx[keep]])
                Nn.append(self.N[i][idx[keep]])
            A, B, Nn = np.concatenate(A), np.concatenate(B), np.concatenate(Nn)
            if len(A) < 10:
                break
            c = A.mean(0)  # about the points' centroid: better conditioned
            a = A - c
            res = ((A - B) * Nn).sum(1)
            J = np.concatenate([np.cross(a, Nn), Nn], 1)
            H = J.T @ J
            x = np.linalg.solve(H + 1e-9 * np.trace(H) * np.eye(6), -J.T @ res)
            w, dt = x[:3], x[3:]
            ang = float(np.linalg.norm(w))
            dR = rot(w / ang, np.degrees(ang)) if ang > 1e-12 else np.eye(3)
            # x' = dR (x - c) + c + dt
            Tn = _T(dR @ T[:3, :3], dR @ (T[:3, 3] - c) + c + dt)
            done = ang < 1e-7 and float(np.linalg.norm(dt)) < 1e-7
            T = Tn
            if done:
                break
        return T

    def icp_level(self, T: np.ndarray, iters: int, sub: bool) -> np.ndarray:
        """As ``icp``, with the camera's tilt and height held: a turn about the world's vertical and a horizontal shift."""
        trim = float(self.p["trim"])
        for _ in range(iters):
            A, B, Nn = [], [], []
            for i in range(len(self.src)):
                if not len(self.src[i]):
                    continue
                e, Xw, idx, ok = self.residuals(i, T, sub)
                keep = ok & (e <= np.quantile(e, trim)) & (e < float(self.p["cap_m"]))
                A.append(Xw[keep])
                B.append(self.P[i][idx[keep]])
                Nn.append(self.N[i][idx[keep]])
            A, B, Nn = np.concatenate(A), np.concatenate(B), np.concatenate(Nn)
            if len(A) < 10:
                break
            c = A.mean(0)
            a = A - c
            res = ((A - B) * Nn).sum(1)
            J = np.stack([np.cross(a, Nn)[:, 2], Nn[:, 0], Nn[:, 1]], 1)
            H = J.T @ J
            x = np.linalg.solve(H + 1e-9 * np.trace(H) * np.eye(3), -J.T @ res)
            dR = rot([0, 0, 1], np.degrees(x[0]))
            t = dR @ (T[:3, 3] - c) + c + np.array([x[1], x[2], 0.0])
            done = abs(x[0]) < 1e-7 and abs(x[1]) < 1e-7 and abs(x[2]) < 1e-7
            T = _T(dR @ T[:3, :3], t)
            if done:
                break
        return T


def fit(panda, samples: dict, frames: list, qs: list, gs: list, p: dict, supports: list | None = None,
        rc_samples: dict | None = None) -> dict:
    """``frames``: dicts with depth_u16, depth_scale_mm, K, T_world_from_camera (the assumed camera, the same in every
    frame); ``qs``, ``gs``: each frame's joints and gripper. Returns the estimate and how it was reached."""
    t0 = time.perf_counter()
    T0 = np.asarray(frames[0]["T_world_from_camera"], float)
    K = np.asarray(frames[0]["K"], float)
    s = int(p["stride"])
    G = [grid(depth_m(fr), s) for fr in frames]
    Z = np.stack([g_[0] for g_ in G])
    u, v = G[0][1], G[0][2]
    M = arm_masks(Z, p)
    src = [lift(Z[i][M[i]], u[M[i]], v[M[i]], K) for i in range(len(frames))]
    moved = moved_bodies(panda, samples, qs, gs, float(p["moved_m"]))
    out = {"n_px": [int(len(x)) for x in src], "moved": len(moved), "accepted": False}
    if not moved or sum(len(x) for x in src) < 100:
        out.update(reason="no arm motion seen", s=round(time.perf_counter() - t0, 3))
        return out
    dst = [arm_points(panda, samples, q, g, bodies=set(moved)) for q, g in zip(qs, gs)]
    if p.get("focal_scales") and p.get("plane_lock") and supports and rc_samples is not None:
        return _fit_focal(panda, frames, qs, gs, p, supports, rc_samples, Z, u, v, M, dst, out, T0, K, t0)
    F = _Frames(src, dst, p)
    cs = np.concatenate(src).mean(0)
    cd = np.concatenate([d[0] for d in dst]).mean(0)
    lv = None
    if p.get("plane_lock") and supports:
        n_c, d_c, share_p = plane_of_frame(frames[0], p, exclude=M[0])
        lv = level(T0, n_c, d_c, supports, p)
        out.update(plane={"share": round(share_p, 3), "d_m": round(d_c, 4)})
        if lv is not None:
            out["plane"].update(tilt_deg=round(lv[3], 3), height_err_m=round(lv[4], 4), z_support=lv[2])
    if lv is not None:
        # plane locked: the camera's tilt and height from the support plane; its turn about the vertical and its
        # horizontal position from the arm
        R0, tz = lv[0], lv[1]
        span, step = float(p["yaw_span_deg"]), float(p["yaw_step_deg"])
        cands = []
        for yaw in np.arange(-span, span + 1e-9, step):
            R = rot([0, 0, 1], yaw) @ R0
            for t in (np.array([T0[0, 3], T0[1, 3], tz]), cd - R @ cs):
                T = _T(R, np.array([t[0], t[1], tz]))
                cands.append((F.cost(T, sub=True), T))
        cands.sort(key=lambda c_: c_[0])
        short = sorted(((F.cost(T1, sub=True), T1) for T1 in (F.icp_level(T, int(p["short_iters"]), True)
                                                              for _c, T in cands[:int(p["short_starts"])])),
                       key=lambda c_: c_[0])
        full = [(F.cost(T2), T2) for T2 in (F.icp_level(T, int(p["full_iters"]), False)
                                            for _c, T in short[:int(p["full_starts"])])]
        c_end, T_est = min(full, key=lambda c_: c_[0])
        rc = None
        if rc_samples is not None:
            Zg, arms = _rc_data(panda, rc_samples, frames, qs, gs, s)
            rc = (Zg, arms, K)
        return _finish(out, F, T0, T_est, c_end, p, t0, "plane", rc=rc)
    # 1. coarse starts around the assumed camera
    span, step, roll = float(p["coarse_deg"]), float(p["coarse_step_deg"]), float(p["coarse_roll_deg"])
    angs = np.arange(-span, span + 1e-9, step)
    cands = []
    for yaw in angs:
        for pitch in angs:
            for rl in sorted({-roll, 0.0, roll}):
                R = T0[:3, :3] @ rot([0, 1, 0], yaw) @ rot([1, 0, 0], pitch) @ rot([0, 0, 1], rl)
                for t in (T0[:3, 3], cd - R @ cs):
                    T = _T(R, t)
                    cands.append((F.cost(T, sub=True), T))
    cands.sort(key=lambda c_: c_[0])
    # 2. short ICP from the best starts, full ICP from the best of those
    short = sorted(((F.cost(T1, sub=True), T1) for T1 in (F.icp(T, int(p["short_iters"]), True)
                                                          for _c, T in cands[:int(p["short_starts"])])),
                   key=lambda c_: c_[0])
    full = [(F.cost(T2), T2) for T2 in (F.icp(T, int(p["full_iters"]), False) for _c, T in short[:int(p["full_starts"])])]
    c_end, T_est = min(full, key=lambda c_: c_[0])
    return _finish(out, F, T0, T_est, c_end, p, t0, "free")


def _fit_focal(panda, frames, qs, gs, p, supports, rc_samples, Z, u, v, M, dst, out, T0, K, t0) -> dict:
    """The plane-locked fit with the focal length free (item 2b): the search at each factor in ``focal_scales``, then a
    golden-section search between the best one's neighbours, each judged by the arm's cost at the factor's own plane
    lock; then ``_finish`` at the factor found, the assumed camera scored at the assumed K."""
    def frames_at(Ks):
        return _Frames([lift(Z[i][M[i]], u[M[i]], v[M[i]], Ks) for i in range(len(frames))], dst, p)

    cd = np.concatenate([d[0] for d in dst]).mean(0)
    span, step = float(p["yaw_span_deg"]), float(p["yaw_step_deg"])
    seen: dict = {}

    def search(fs: float):
        fs = round(float(fs), 5)
        if fs in seen:
            return seen[fs]
        Ks = scaled_K(K, fs)
        F = frames_at(Ks)
        n_c, d_c, share_p = plane_of_frame(frames[0], p, exclude=M[0], K=Ks)
        lv = level(T0, n_c, d_c, supports, p)
        if lv is None:
            seen[fs] = None
            return None
        cs = np.concatenate(F.src).mean(0)
        R0, tz = lv[0], lv[1]
        cands = []
        for yaw in np.arange(-span, span + 1e-9, step):
            R = rot([0, 0, 1], yaw) @ R0
            for t_ in (np.array([T0[0, 3], T0[1, 3], tz]), cd - R @ cs):
                T = _T(R, np.array([t_[0], t_[1], tz]))
                cands.append((F.cost(T, sub=True), T))
        cands.sort(key=lambda c_: c_[0])
        short = sorted(((F.cost(T1, sub=True), T1) for T1 in (F.icp_level(T, int(p["short_iters"]), True)
                                                              for _c, T in cands[:int(p["short_starts"])])),
                       key=lambda c_: c_[0])
        full = [(F.cost(T2), T2) for T2 in (F.icp_level(T, int(p["full_iters"]), False)
                                            for _c, T in short[:int(p["full_starts"])])]
        c_end, T_est = min(full, key=lambda c_: c_[0])
        seen[fs] = (c_end, T_est, F, Ks, fs, {"share": round(share_p, 3), "d_m": round(d_c, 4),
                                              "tilt_deg": round(lv[3], 3), "height_err_m": round(lv[4], 4),
                                              "z_support": lv[2]})
        return seen[fs]

    grid_f = sorted(float(f) for f in p["focal_scales"])
    res = [search(f) for f in grid_f]
    ok = [i for i, r in enumerate(res) if r is not None]
    out["focal_sweep"] = [{"f": round(f, 4), "cost_m": None if r is None else round(r[0], 5)} for f, r in zip(grid_f, res)]
    if not ok:
        out.update(reason="no level plane at any focal factor", s=round(time.perf_counter() - t0, 3))
        return out
    i = min(ok, key=lambda j: res[j][0])
    lo, hi = grid_f[max(i - 1, 0)], grid_f[min(i + 1, len(grid_f) - 1)]
    g = (5 ** 0.5 - 1) / 2
    for _ in range(int(p["focal_refine"])):  # golden section on [lo, hi]
        a_, b_ = hi - g * (hi - lo), lo + g * (hi - lo)
        ra, rb = search(a_), search(b_)
        ca = np.inf if ra is None else ra[0]
        cb = np.inf if rb is None else rb[0]
        if ca <= cb:
            hi = b_
        else:
            lo = a_
    best = min((r for r in seen.values() if r is not None), key=lambda r: r[0])
    c_end, T_est, F, Ks, fs, plane = best
    out["plane"] = plane
    out["focal_searched"] = len(seen)
    Zg, arms = _rc_data(panda, rc_samples, frames, qs, gs, int(p["stride"]))
    return _finish(out, F, T0, T_est, c_end, p, t0, "plane", rc=(Zg, arms, Ks),
                   focal={"K0": K, "f": fs, "frames_at": frames_at})


def _finish(out: dict, F: _Frames, T0, T_est, c_end: float, p: dict, t0: float, mode: str, rc=None,
            focal: dict | None = None) -> dict:
    """``rc``: (Zg, arms, K) for the render refinement (plane-locked mode); the estimate is then the refined camera and
    it is accepted on the render cost against the assumed camera's. ``focal`` (item 2b): the assumed K0, the factor
    that ``rc``'s K carries, and ``frames_at`` (the moving pixels lifted at a K); the assumed camera is then scored at
    K0."""
    c0 = F.cost(T0) if focal is None else focal["frames_at"](focal["K0"]).cost(T0)
    out.update(mode=mode, cost_m=round(c_end, 5), cost_assumed_m=round(c0, 5))
    ok = c_end <= float(p["accept_cost_m"]) and c_end <= float(p["accept_ratio"]) * c0
    f_total = 1.0
    if rc is not None:
        Zg, arms, K = rc
        K0 = K if focal is None else focal["K0"]
        s, tau = int(p["stride"]), float(p["rc_tau_m"])
        m = int(0.5 * rc_cells(T0, arms, K0, Zg[0].shape, s))  # the assumed camera's drawn cells, halved
        r0 = rc_cost(T0, Zg, arms, K0, s, tau)
        r_icp = rc_cost(T_est, Zg, arms, K, s, tau, m)
        T_rc, r1 = refine_rc(T_est, Zg, arms, K, p, min_cells=m)
        if r1 < r_icp:
            T_est = T_rc
        r_end = min(r1, r_icp)
        c_fit = F.cost(T_est)  # does the fitted camera explain the arm's moving pixels?
        out.update(rc_assumed_mm2=round(r0 * 1e6, 2), rc_icp_mm2=round(r_icp * 1e6, 2), rc_mm2=round(r_end * 1e6, 2),
                   cost_fitted_m=round(c_fit, 5))
        ok = r_end <= float(p["rc_accept_ratio"]) * r0 and c_fit <= float(p["accept_cost_m"])
        if focal is not None:
            f_total = float(focal["f"])
            out["focal_scale"] = round(f_total, 5)
    shift, turn = pose_diff(T_est, T0)
    ok = ok and shift <= float(p["max_shift_m"]) and turn <= float(p["max_turn_deg"])
    if ok and shift < float(p["min_shift_m"]) and turn < float(p["min_turn_deg"]) \
            and abs(f_total - 1.0) < float(p["min_focal"]):
        ok = False
        out["why"] = "within the fit's precision of the assumed camera"
    out.update(accepted=bool(ok), T_world_from_camera=T_est, shift_m=round(shift, 4), turn_deg=round(turn, 3),
               s=round(time.perf_counter() - t0, 3))
    return out


# ------------------------------------------------------------------------------------------------ the motion gate
def gate(panda, samples: dict, frames: list, qs: list, gs: list, p: dict) -> dict:
    """Does the assumed camera explain the arm's moving pixels? The frames' masks (``arm_masks``), lifted at the assumed
    camera, against the moving links: the trimmed cost and the share within 1 cm. ``moved`` is None when the arm did not
    move enough to decide."""
    T0 = np.asarray(frames[0]["T_world_from_camera"], float)
    K = np.asarray(frames[0]["K"], float)
    s = int(p["stride"])
    G = [grid(depth_m(fr), s) for fr in frames]
    Z = np.stack([g_[0] for g_ in G])
    u, v = G[0][1], G[0][2]
    M = arm_masks(Z, p)
    src = [lift(Z[i][M[i]], u[M[i]], v[M[i]], K) for i in range(len(frames))]
    n_px = [int(len(x)) for x in src]
    moved = moved_bodies(panda, samples, qs, gs, float(p["moved_m"]))
    out = {"n_px": n_px, "moved_links": len(moved)}
    if not moved or min(n_px) < int(p["motion_min_px"]):
        out.update(moved=None, reason="too little motion")
        return out
    dst = [arm_points(panda, samples, q, g, bodies=set(moved)) for q, g in zip(qs, gs)]
    F = _Frames(src, dst, p)
    c = F.cost(T0)
    e = np.concatenate([F.residuals(i, T0, False)[0] for i in range(len(src))])
    share = float((e <= 0.01).mean())
    out.update(moved=bool(c > float(p["motion_cost_m"]) or share < float(p["motion_share_min"])),
               cost_m=round(c, 5), share_1cm=round(share, 4))
    return out


def fit_joints(panda, samples: dict, frames: list, qs: list, gs: list, p: dict) -> dict:
    """Item 4 (2026-11-14): constant offsets of the joint readings, from the arm, with the camera taken as right.

    The arm's moving pixels (``arm_masks``), lifted at the frames' camera, against the moving links of the model at the
    readings minus the offsets ``delta`` (the same in every frame): Levenberg-Marquardt over the seven offsets on the
    point-to-plane residuals (a pixel matches the nearest model point whose face looks at the camera; trimmed, capped),
    with a numerical Jacobian of the model points. Returns the offsets, the trimmed cost at the readings and at the fit,
    and each frame's hand point (``gripper0_eef``) at the fitted joints, from which the robot measures its hand reading's
    own offset. Accepted when the fitted cost is at most ``joint_accept_cost_m`` and ``joint_accept_ratio`` of the
    readings' cost, and the largest offset lies between ``joint_min_deg`` and ``joint_max_deg``."""
    from scipy.spatial import cKDTree

    t0 = time.perf_counter()
    T0 = np.asarray(frames[0]["T_world_from_camera"], float)
    K = np.asarray(frames[0]["K"], float)
    s = int(p["stride"])
    G = [grid(depth_m(fr), s) for fr in frames]
    Z = np.stack([g_[0] for g_ in G])
    u, v = G[0][1], G[0][2]
    M = arm_masks(Z, p)
    src = [lift(Z[i][M[i]], u[M[i]], v[M[i]], K) for i in range(len(frames))]
    n_px = [int(len(x)) for x in src]
    moved = moved_bodies(panda, samples, qs, gs, float(p["moved_m"]))
    out = {"n_px": n_px, "moved_links": len(moved), "accepted": False}
    if not moved or min(n_px) < int(p["motion_min_px"]):
        out.update(reason="too little motion", s=round(time.perf_counter() - t0, 3))
        return out
    n = int(p["sub_n"])
    X = []
    for x in src:
        xw = x @ T0[:3, :3].T + T0[:3, 3]
        X.append(xw[np.linspace(0, len(xw) - 1, min(len(xw), n)).astype(int)])
    cam, bodies, knn, cap = T0[:3, 3], set(moved), int(p["knn"]), float(p["cap_m"])
    near, trim, h = 2.0 * float(p["spacing_m"]), float(p["trim"]), float(p["joint_step_rad"])

    def match(P, N, Xi):
        d, idx = cKDTree(P).query(Xi, k=knn)
        facing = ((cam - P[idx]) * N[idx]).sum(-1) > 0
        d = np.where(facing, d, np.inf)
        j = np.argmin(d, axis=1)
        r_ = np.arange(len(d))
        d, idx = d[r_, j], idx[r_, j]
        ok = np.isfinite(d)
        e = np.full(len(d), cap)
        if ok.any():
            dp = np.abs(((Xi[ok] - P[idx[ok]]) * N[idx[ok]]).sum(1))
            e[ok] = np.minimum(np.where(d[ok] <= near, dp, d[ok]), cap)
        return idx, ok, e

    def cost(delta) -> float:
        tot, cnt = 0.0, 0
        for i in range(len(X)):
            P, N = arm_points(panda, samples, qs[i] - delta, gs[i], bodies=bodies)
            e = match(P, N, X[i])[2]
            k = max(1, int(trim * len(e)))
            tot += float(np.partition(e, k - 1)[:k].sum())
            cnt += k
        return tot / max(cnt, 1)

    delta = np.zeros(len(qs[0]))
    c0 = c = cost(delta)
    lam, it, max_step = 1e-3, 0, np.radians(float(p["joint_max_step_deg"]))
    for it in range(int(p["joint_iters"])):
        Js, rs = [], []
        for i in range(len(X)):
            P, N = arm_points(panda, samples, qs[i] - delta, gs[i], bodies=bodies)
            idx, ok, e = match(P, N, X[i])
            keep = ok & (e < cap) & (e <= np.quantile(e, trim))
            if keep.sum() < 10:
                continue
            Pk, Nk = P[idx[keep]], N[idx[keep]]
            rs.append(((X[i][keep] - Pk) * Nk).sum(1))
            cols = []
            for j in range(len(delta)):
                dq = np.zeros(len(delta))
                dq[j] = h
                Pj, _ = arm_points(panda, samples, qs[i] - delta - dq, gs[i], bodies=bodies)
                cols.append(-(((Pj[idx[keep]] - Pk) / h) * Nk).sum(1))  # d residual / d delta_j
            Js.append(np.stack(cols, 1))
        if not rs:
            break
        J, r = np.concatenate(Js), np.concatenate(rs)
        H, g = J.T @ J, J.T @ r
        improved = False
        for _ in range(6):
            step = np.linalg.solve(H + lam * (np.diag(np.diag(H)) + 1e-9 * np.eye(len(delta))), -g)
            step = np.clip(step, -max_step, max_step)
            c_new = cost(delta + step)
            if c_new < c:
                delta, c, lam, improved = delta + step, c_new, max(lam / 3.0, 1e-6), True
                break
            lam *= 5.0
        if not improved or float(np.abs(step).max()) < 1e-6:
            break
    deg = np.degrees(delta)
    big = float(np.abs(deg).max())
    eef = [panda.fk(qs[i] - delta, gs[i])["gripper0_eef"][1] for i in range(len(qs))]
    out.update(delta=[round(float(x), 6) for x in delta], delta_deg=[round(float(x), 3) for x in deg],
               cost_before_m=round(c0, 5), cost_m=round(c, 5), iters=it + 1,
               eef_fk=[[round(float(x), 5) for x in e_] for e_ in eef],
               accepted=bool(c <= float(p["joint_accept_cost_m"]) and c <= float(p["joint_accept_ratio"]) * c0
                             and float(p["joint_min_deg"]) <= big <= float(p["joint_max_deg"])),
               s=round(time.perf_counter() - t0, 3))
    return out


def project(P: np.ndarray, T: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points as pixels (u, v) of the camera ``T`` (world from camera, z forward), those in front of it."""
    X = (np.asarray(P, float) - T[:3, 3]) @ T[:3, :3]
    X = X[X[:, 2] > 0.05]
    return K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2], K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]


def changed_mask(frames: list, p: dict) -> tuple[np.ndarray, bool]:
    """The stride grid's pixels that changed between the frames, and whether they are the robot's own: the frames'
    ``mask`` arrays joined (from the colour images: a pixel that differs from the frames' median, so one the arm
    covers in some frames and not in all), else the depth frames' arm masks joined (``arm_masks``: nearer than the
    frames' median, so with three frames a pixel the arm covers in one frame only)."""
    s = int(p["stride"])
    if all(f.get("mask") is not None for f in frames):
        M = np.any(np.stack([np.asarray(f["mask"], bool) for f in frames]), axis=0)
        if M.shape == np.asarray(frames[0]["depth_u16"]).shape and s > 1:  # at the frame's size: the grid's pixels
            M = M[s // 2::s, s // 2::s]
        # ``mask_one``: the robot kept only the pixels the arm covers in that frame alone (as the depth masks do)
        return M, not all(bool(f.get("mask_one")) for f in frames)
    Z = np.stack([grid(depth_m(fr), s)[0] for fr in frames])
    return np.any(arm_masks(Z, p), axis=0), False


def fit_image(panda, samples: dict, frames: list, qs: list, gs: list, p: dict) -> dict:
    """The image shifted and zoomed, the camera where it was assumed (a crop, another lens, a rescaled picture): find
    zoom ``s`` and shift ``t`` with ``o = s (m - c) + c + t``, where ``m`` is a pixel of the picture the assumed camera
    would take, ``o`` the pixel that shows the same point in the picture that came, and ``c`` the principal point.

    Two-dimensional, so it needs no depth. The pixels that changed between the frames are compared with the moving
    links drawn at every frame's joint readings: every pixel a drawing covers (the robot's own masks, which mark any
    change: a link's own markings slide under a pixel it keeps covering), or the pixels fewer than half the drawings
    cover (the depth masks, and masks of one frame alone). The score is the two sets' overlap, intersection over
    union, searched over zoom and shift on a grid and refined around the best.

    Why overlap and not distance (2026-11-15, ``probes/s16_fit.py`` on real frames): the trimmed distance between the
    sets had its minimum at the wrong zoom on plain frames (0.6-0.8 for 1.0), because a shrunk drawing lies inside the
    change wherever it is put; and in the wider S16 picture a strip at the bottom edge changes with the arm (18-35% of
    the changed pixels), which pulled the distance fit 270 px down. Overlap pays for both: a shrunk drawing loses
    intersection, a grown one adds union, and the strip costs the same at every candidate. On three tasks' plain and
    S16 frames it found the zoom within 0.02 and the shift within 8 px.

    Accepted when the overlap is good (``img_min_iou``), clearly better than with the picture as it came
    (``img_iou_gain``), and the change is larger than the fit's precision (``img_min_shift_px``, ``img_min_zoom``)."""
    from scipy.ndimage import binary_closing, binary_dilation

    t0 = time.perf_counter()
    T0 = np.asarray(frames[0]["T_world_from_camera"], float)
    K = np.asarray(frames[0]["K"], float)
    st = int(p["stride"])
    M, own = changed_mask(frames, p)
    gh, gw = M.shape
    o_ = st // 2
    moved = moved_bodies(panda, samples, qs, gs, float(p["moved_m"]))
    out = {"n_px": int(M.sum()), "moved_links": len(moved)}
    if not moved or out["n_px"] < int(p["img_min_px"]):
        out.update(accepted=False, reason="too little motion", s=round(time.perf_counter() - t0, 3))
        return out
    pad = max(gh, gw)  # the arm may be drawn outside the picture, and a wider lens shows it
    drawn = []
    for q, g in zip(qs, gs):
        P, _N = arm_points(panda, samples, q, g, bodies=set(moved))
        um, vm = project(P, T0, K)
        iu = np.round((um - o_) / st).astype(int) + pad
        iv = np.round((vm - o_) / st).astype(int) + pad
        ok = (iu >= 0) & (iu < gw + 2 * pad) & (iv >= 0) & (iv < gh + 2 * pad)
        S = np.zeros((gh + 2 * pad, gw + 2 * pad), bool)
        S[iv[ok], iu[ok]] = True
        drawn.append(binary_closing(S, structure=np.ones((3, 3), bool)))
    n_cov = np.stack(drawn).sum(0)
    E = (n_cov > 0) if own else ((n_cov > 0) & (2 * n_cov < len(drawn)))
    if not E.any():
        out.update(accepted=False, reason="the arm is not in view", s=round(time.perf_counter() - t0, 3))
        return out
    k = int(p["img_dilate"])
    if k > 0:  # a cell of tolerance on both sides
        E, M = binary_dilation(E, iterations=k), binary_dilation(M, iterations=k)
    vv, uu = np.nonzero(M)
    obs = np.stack([uu * st + o_, vv * st + o_], 1).astype(np.float64)  # pixels, as ``grid``
    n_obs = len(obs)
    SAT = np.zeros((E.shape[0] + 1, E.shape[1] + 1), np.int64)  # E's summed-area table: its cells in any rectangle
    SAT[1:, 1:] = E.cumsum(0).cumsum(1)
    c = np.array([K[0, 2], K[1, 2]])
    w_px, h_px = gw * st, gh * st

    def cell(x, n):
        return np.clip(np.round((x - o_) / st).astype(int) + pad, 0, n - 1)

    def iou(x) -> float:
        z, t = float(x[0]), np.array([x[1], x[2]])
        b = (obs - c - t) / z + c  # the changed pixels, taken back to the assumed picture
        inside = (b[:, 0] > -pad * st) & (b[:, 0] < (gw + pad) * st) & (b[:, 1] > -pad * st) & (b[:, 1] < (gh + pad) * st)
        inter = int(E[cell(b[inside, 1], E.shape[0]), cell(b[inside, 0], E.shape[1])].sum())
        # the drawn change that falls inside the picture that came, in that picture's cells: E's cells inside the
        # picture's frame taken back, times the zoom's area
        lo = (np.array([0.0, 0.0]) - c - t) / z + c
        hi = (np.array([w_px, h_px], float) - c - t) / z + c
        u0, u1 = int(cell(np.array([lo[0]]), E.shape[1] + 1)[0]), int(cell(np.array([hi[0]]), E.shape[1] + 1)[0])
        v0, v1 = int(cell(np.array([lo[1]]), E.shape[0] + 1)[0]), int(cell(np.array([hi[1]]), E.shape[0] + 1)[0])
        n_w = float(SAT[v1, u1] - SAT[v0, u1] - SAT[v1, u0] + SAT[v0, u0]) * z * z
        return inter / max(1.0, n_w + n_obs - inter)

    i0 = iou((1.0, 0.0, 0.0))
    if p.get("img_probe") is not None:  # a probe's question: the overlap at a zoom and shift it names (px)
        out["probe_iou"] = round(iou(tuple(float(v) for v in p["img_probe"])), 4)
    span = float(p["img_max_shift"]) * w_px
    step = float(p["img_step_px"])
    shifts = np.arange(-span, span + 1e-9, step)
    zs = sorted(float(z) for z in p["img_zooms"])
    best, x = i0, (1.0, 0.0, 0.0)
    for z in zs:
        for tx in shifts:
            for ty in shifts:
                v = iou((z, tx, ty))
                if v > best:
                    best, x = v, (z, float(tx), float(ty))
    for dz, dt in ((0.025, step / 2.0), (0.01, step / 4.0), (0.004, max(1.0, step / 16.0))):
        z0, x0, y0 = x
        for z in z0 + dz * np.arange(-3, 4):
            if not zs[0] <= z <= zs[-1]:
                continue
            for tx in x0 + dt * np.arange(-4, 5):
                for ty in y0 + dt * np.arange(-4, 5):
                    if abs(tx) > span or abs(ty) > span:
                        continue
                    v = iou((z, tx, ty))
                    if v > best:
                        best, x = v, (float(z), float(tx), float(ty))
    shift = float(np.hypot(x[1], x[2]))
    changed = shift >= float(p["img_min_shift_px"]) or abs(x[0] - 1.0) >= float(p["img_min_zoom"])
    ok = bool(changed and best >= float(p["img_min_iou"]) and best >= float(p["img_iou_gain"]) * i0)
    out.update(accepted=ok, zoom=round(float(x[0]), 4), shift_px=[round(float(x[1]), 2), round(float(x[2]), 2)],
               centre_px=[float(c[0]), float(c[1])], iou=round(best, 4), iou0=round(i0, 4),
               s=round(time.perf_counter() - t0, 3))
    if not ok:
        out["reason"] = "no change" if not changed else "the overlap is not good enough"
    return out


def readings(panda, frames: list, qs: list, gs: list) -> dict:
    """Item 4, the colour line's guard (2026-11-14): do the robot's readings agree with each other? Each frame's hand
    reading (``eef_pos``) against the arm model's hand point (``gripper0_eef``) at that frame's joint readings. They agree
    to well under a millimetre when both readings are right, or wrong together (a frozen or delayed observation); S18
    offsets them apart, by about 13 cm. No image is used."""
    dev = [float(np.linalg.norm(np.asarray(f["eef_pos"], float) - np.asarray(panda.fk(q, g)["gripper0_eef"][1], float)))
           for f, q, g in zip(frames, qs, gs)]
    return {"dev_mm": [round(d * 1000.0, 2) for d in dev], "max_mm": round(max(dev) * 1000.0, 2) if dev else None}


def fit_readings(panda, frames: list, qs: list, gs: list, p: dict) -> dict:
    """2026-11-16: constant offsets of the joint readings (``delta``) and of the hand reading (``dp``), from the
    readings alone. No image is used.

    With both readings right, the hand reading is the arm model's hand point (``gripper0_eef``) at the joint readings
    (``readings``). With a constant offset on each, ``hand_i = fk(q_i - delta) + dp`` at every step ``i``: the same
    two offsets have to explain every step, and once the arm has moved along more than one line only the true ones
    do. Gauss-Newton with a numerical Jacobian over ``dp`` and the offsets of the joints that move the hand point
    (the last joint turns about it and stays 0).

    **The lag** (when every frame carries ``joint_vel``): while the arm moves, the hand reading shows the hand a
    moment before the joint readings do, ``hand_i = fk(q_i - delta - tau v_i) + dp``, with one more unknown ``tau``
    (seconds). Measured 2026-11-16 on a training episode's readings: 0.46 mm rms left at the true offsets without
    it (up to 0.70 mm at 0.35 m/s), 0.0001 mm with ``tau`` 0.002; without it the fit of the first 11 to 40 steps put
    the hand's offset 2 to 6 mm off, with it 0.001 mm.

    Accepted when the offsets leave at most ``read_rms_m`` at the steps used, the fit's condition number is at most
    ``read_cond``, the readings were at least ``read_min_gap_m`` apart before it, and no offset is beyond
    ``read_max_rad`` / ``read_max_m`` (the lag within ``read_max_tau_s``). ``reason``: ``"readings agree"`` (nothing
    to fix), ``"undetermined"`` (the offsets explain the steps and others would too: too little motion yet, or all
    of it along one line; the robot asks again after more), ``"not constant offsets"``, ``"beyond bounds"``."""
    t0 = time.perf_counter()
    n_sent = len(frames)
    out = {"n_sent": n_sent, "accepted": False}
    if n_sent < 4:
        out.update(reason="undetermined", s=round(time.perf_counter() - t0, 3))
        return out
    idx = sorted(set(np.linspace(0, n_sent - 1, min(n_sent, int(p["read_max_n"]))).round().astype(int).tolist()))
    Q = np.array([np.asarray(qs[i], float) for i in idx])
    G = [np.asarray(gs[i], float) for i in idx]
    H = np.array([np.asarray(frames[i]["eef_pos"], float) for i in idx])
    lag = all(frames[i].get("joint_vel") is not None and len(frames[i]["joint_vel"]) == Q.shape[1] for i in idx)
    V = np.array([np.asarray(frames[i]["joint_vel"], float) for i in idx]) if lag else np.zeros_like(Q)

    def hands(dq, tau=0.0) -> np.ndarray:
        return np.array([np.asarray(panda.fk(q - dq - tau * v, g)["gripper0_eef"][1], float) for q, v, g in zip(Q, V, G)])

    nq, h = Q.shape[1], float(p["read_step_rad"])
    gap = float(np.linalg.norm(H - hands(np.zeros(nq)), axis=1).max())
    out.update(n=len(idx), gap_mm=round(gap * 1000.0, 2))
    if gap < float(p["read_min_gap_m"]):
        out.update(reason="readings agree", s=round(time.perf_counter() - t0, 3))
        return out
    dq, dp, tau = np.zeros(nq), np.zeros(3), 0.0
    free, sv = list(range(nq)), None
    ht = 1e-5  # the numerical step of the lag (s)
    for it in range(int(p["read_iters"])):
        F = hands(dq, tau)
        r = (H - F - dp).ravel()
        J = np.zeros((len(r), nq + 4))
        for j in range(nq):
            d = np.zeros(nq)
            d[j] = h
            J[:, j] = -((hands(dq + d, tau) - F) / h).ravel()  # d r / d dq_j, with r = H - fk(q - dq - tau v) - dp
        J[:, nq:nq + 3] = -np.tile(np.eye(3), (len(idx), 1))
        if lag:
            J[:, nq + 3] = -((hands(dq, tau + ht) - F) / ht).ravel()
        if it == 0:  # the joints that move the hand point at all
            norms = np.linalg.norm(J[:, :nq], axis=0)
            free = [j for j in range(nq) if norms[j] > 1e-6 * max(float(norms.max()), 1e-12)]
        cols = free + [nq, nq + 1, nq + 2] + ([nq + 3] if lag else [])
        Jf = J[:, cols]
        # radians, metres and seconds: every column at unit length, so the condition number compares like with like
        scale = 1.0 / np.maximum(np.linalg.norm(Jf, axis=0), 1e-12)
        step, _, _, sv = np.linalg.lstsq(Jf * scale, -r, rcond=1e-10)
        step = step * scale
        cap = float(np.abs(step[:len(free) + 3]).max())
        if cap > 0.2:  # one iteration moves no offset by more than this (rad, m)
            step = step * (0.2 / cap)
        dq[free] += step[:len(free)]
        dp += step[len(free):len(free) + 3]
        if lag:
            tau += float(step[-1])
        if float(np.abs(step[:len(free) + 3]).max()) < 1e-9:
            break
    res = H - hands(dq, tau) - dp
    rms = float(np.sqrt((res ** 2).sum(axis=1).mean()))
    cond = float(sv[0] / max(float(sv[-1]), 1e-300)) if sv is not None and len(sv) else float("inf")
    out.update(delta=[round(float(x), 6) for x in dq], dp=[round(float(x), 6) for x in dp],
               delta_deg=[round(float(np.degrees(x)), 3) for x in dq], dp_mm=[round(float(x) * 1000.0, 2) for x in dp],
               rms_mm=round(rms * 1000.0, 4), cond=round(cond, 1), iters=it + 1, joints_free=len(free))
    if lag:
        out["tau_ms"] = round(tau * 1000.0, 4)
    if not np.isfinite(cond) or cond > float(p["read_cond"]):  # first: an undetermined fit's residual says nothing
        out["reason"] = "undetermined"
    elif rms > float(p["read_rms_m"]):
        out["reason"] = "not constant offsets"
    elif float(np.abs(dq).max()) > float(p["read_max_rad"]) or float(np.linalg.norm(dp)) > float(p["read_max_m"]) \
            or abs(tau) > float(p["read_max_tau_s"]):
        out["reason"] = "beyond bounds"
    else:
        out["accepted"] = True
    out["s"] = round(time.perf_counter() - t0, 3)
    return out


# ------------------------------------------------------------------------------------------------ the request
def handle(per, msg: dict) -> dict:
    """The server's ``arm_cam`` request (``m2_perceive.Perceiver.handle``)."""
    wc = per.cell(msg["suite"])
    p = params(per.p.get("arm_cam"))
    if getattr(wc, "arm_samples", None) is None:
        wc.arm_samples = model_samples(wc.panda, float(p["spacing_m"]))
        wc.arm_rc_samples = model_samples(wc.panda, float(p["rc_spacing_m"]))
    frames = msg["frames"]
    qs = [np.asarray(f["joint_pos"], float) for f in frames]
    gs = [np.asarray(f["gripper_qpos"], float) for f in frames]
    sup = support_heights(wc.workcell)
    if msg["mode"] == "check":
        rep = check(wc.panda, wc.arm_samples, frames[0], qs[0], gs[0], p, supports=sup, rc_samples=wc.arm_rc_samples)
    elif msg["mode"] in ("gate_fit", "gate_fit2"):  # the gate, then the fit only when the camera does not explain the arm
        if msg["mode"] == "gate_fit2":  # item 2b: the second try, on the second net's depth, with its own overrides
            p = {**p, **(p.get("second") or {})}
        t0 = time.perf_counter()
        g_ = gate(wc.panda, wc.arm_samples, frames, qs, gs, p)
        g_["s"] = round(time.perf_counter() - t0, 3)
        rep = {"gate": g_}
        if g_.get("moved"):
            rep["fit"] = fit(wc.panda, wc.arm_samples, frames, qs, gs, p, supports=sup, rc_samples=wc.arm_rc_samples)
    elif msg["mode"] == "fit":
        rep = fit(wc.panda, wc.arm_samples, frames, qs, gs, p, supports=sup, rc_samples=wc.arm_rc_samples)
    elif msg["mode"] == "joints":  # item 4: the joint readings' offsets (the colour line, after a refused camera fit)
        rep = {"joints": fit_joints(wc.panda, wc.arm_samples, frames, qs, gs, p)}
    elif msg["mode"] == "readings":  # item 4, the colour line's guard: the readings' own consistency (no depth)
        rep = {"readings": readings(wc.panda, frames, qs, gs)}
    elif msg["mode"] == "readings_fit":  # 2026-11-16: the readings' offsets from the readings alone (no depth)
        rep = {"readings_fit": fit_readings(wc.panda, frames, qs, gs, p)}
    elif msg["mode"] == "gate":  # the gate alone (the robot checks a correction it has just applied)
        rep = {"gate": gate(wc.panda, wc.arm_samples, frames, qs, gs, p)}
    elif msg["mode"] == "gate_image":  # 2026-11-15: the gate, then the image's shift and zoom when it fires
        t0 = time.perf_counter()
        g_ = gate(wc.panda, wc.arm_samples, frames, qs, gs, p)
        g_["s"] = round(time.perf_counter() - t0, 3)
        rep = {"gate": g_}
        if g_.get("moved"):
            rep["image"] = fit_image(wc.panda, wc.arm_samples, frames, qs, gs, p)
    elif msg["mode"] == "gate_joints":  # item 4: the gate, then the joint offsets (the depth line: its camera is right)
        t0 = time.perf_counter()
        g_ = gate(wc.panda, wc.arm_samples, frames, qs, gs, p)
        g_["s"] = round(time.perf_counter() - t0, 3)
        rep = {"gate": g_}
        if g_.get("moved"):
            rep["joints"] = fit_joints(wc.panda, wc.arm_samples, frames, qs, gs, p)
    else:
        raise ValueError(f"arm_cam mode {msg['mode']!r}")
    rep["ok_reply"] = True
    return rep
