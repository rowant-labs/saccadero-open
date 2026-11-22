"""TRUTH-FREE: one-frame pose refinement. The frame's own depth, inside E-0c's plane-and-blob mask, is registered to
the chosen object's model points (``models.py``), starting from E-0c's picked pose.

Inputs (all a robot has): the 16-bit depth image, ``camera.json``, ``plane_crop.analyse``'s table plane and blob mask
(the same call E-0c makes), the blinded token (the sampling seed), the picked object and its start pose. No mesh, no
render key, no case record.

Steps:

1. **Data** (``extract_data``). ``N_DATA`` blob pixels, sampled with ``default_rng([DATA_SEED, int(token, 16)])``,
   back-projected to world points (E-0c ``frontend.world_points_at``). A depth-image normal per point: the cross
   product of the world points ``s`` px to either side in u and v (s = 2, else 1), all on the blob, turned toward the
   camera; NaN where no such neighbours exist.
2. **Visibility filter** (``visible``), recomputed at every iteration from the current pose: a model point is a
   candidate only if its normal faces the camera, it is not self-occluded (z-buffer of all placed model points on
   ``VIS_BIN_PX`` bins: within ``VIS_TOL_M`` of its bin's nearest point), it projects inside the image and it lies at
   least ``VIS_MIN_HEIGHT_M`` above the fitted table.
3. **Correspondences.** Each data point, moved into the model frame by the current pose, takes the nearest VISIBLE
   model point among its ``K_NEIGHBOURS`` nearest. Pairs farther than the stage's distance gate are dropped, and so
   are pairs whose data normal and model normal differ by more than ``NORMAL_COMPAT_DEG``.
4. **Point-to-plane step** (``icp``). Residual ``r = n_m . (y - m)``. Huber weights (delta per stage) and one damped
   Gauss-Newton step on the 6 unknowns (rotation about the data centroid, translation), clamped to ``MAX_STEP_*``.
   Three stages of (iterations, gate, delta): ``STAGES``. The last stage stops early on a step below ``STOP_STEP_*``.
   Ablations: point-to-point (weighted Kabsch), no visibility filter, no Huber weights.
5. **Fit quality** (``fit_quality``), truth-free: the inlier share (data points with a visible model point within
   ``INLIER_GATE_M`` and ``|r| <= INLIER_RESID_M``), the median ``|r|`` of the inliers, and the free-space share
   (visible model points more than ``FS_MIN_HEIGHT_M`` above the table that project off the dilated blob). The
   secondary arm ``u77_best3`` keeps, among the refined poses of E-0c's 3 poses of the picked object, the one with
   the highest ``inlier share - free-space share`` (ties: E-0c's order, the picked pose first).

Pose convention (as E-0c): ``(R, t)`` places model point m at ``R (m - p_train) + t``. Internally the world-to-model
map ``m = Rm x + tm`` is updated (``Rm = R^T``, ``tm = p_train - R^T t``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np

from rowant_lab.hybrid_frontier.screens.e0c_check_timing import frontend as e0c_frontend
from rowant_lab.hybrid_frontier.screens.e0c_check_timing import proposal as e0c_proposal
from rowant_lab.hybrid_frontier.screens.e0f_refine import spec as S
from rowant_lab.hybrid_frontier.screens.e0f_refine.models import ModelCloud

TRUTH_FREE = True


# ------------------------------------------------------------------------------------------- small maths
def rodrigues(w: np.ndarray) -> np.ndarray:
    """Rotation matrix of the rotation vector ``w`` (radians)."""
    w = np.asarray(w, float)
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        return np.eye(3)
    a = w / th
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation vector (radians) of a rotation matrix."""
    R = np.asarray(R, float)
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-12:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(th)
    if s < 1e-9:  # th near pi
        A = (R + np.eye(3)) / 2.0
        a = A[np.argmax(np.diag(A))]
        a = a / np.linalg.norm(a)
        return a * th
    return v / (2 * s) * th


def to_internal(R, t) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(R, float)
    return R.T.copy(), S.P_TRAIN - R.T @ np.asarray(t, float)


def from_internal(Rm, tm) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(Rm, float).T
    return R.copy(), R @ (S.P_TRAIN - np.asarray(tm, float))


# ------------------------------------------------------------------------------------------- camera and frame
@dataclass
class CameraGeom:
    fx: float
    fy: float
    cx: float
    cy: float
    W: int
    H: int
    R_wc: np.ndarray  # camera -> world rotation
    c_w: np.ndarray  # camera centre (world)

    @property
    def R_cw(self) -> np.ndarray:
        return self.R_wc.T

    @classmethod
    def from_camera(cls, camera: dict) -> "CameraGeom":
        T = np.asarray(camera["T_world_from_camera"], float)
        return cls(float(camera["fx"]), float(camera["fy"]), float(camera["cx"]), float(camera["cy"]),
                   int(camera["width"]), int(camera["height"]), T[:3, :3].copy(), T[:3, 3].copy())

    def project(self, Xw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Xc = (np.asarray(Xw) - self.c_w) @ self.R_wc  # rows: R_cw (x - c)
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * Xc[:, 0] / z + self.cx
            v = self.fy * Xc[:, 1] / z + self.cy
        return u, v, z


@dataclass
class FrameData:
    pts: np.ndarray  # (n, 3) world
    nrm: np.ndarray  # (n, 3) world, NaN where no depth normal
    has_n: np.ndarray  # (n,) bool
    plane_n: np.ndarray  # world unit normal of the table (up)
    plane_h0: float  # height above the table = plane_n . x + plane_h0
    blob_dil: np.ndarray  # (H, W) blob mask dilated by FS_DILATE_PX
    cam: CameraGeom
    n_blob: int
    diag: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(len(self.pts))

    def height(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X) @ self.plane_n + self.plane_h0


def plane_world(plane: dict, cam: CameraGeom) -> tuple[np.ndarray, float]:
    """``plane_crop``'s camera-frame plane (height = n_c . x_c + d) as world ``(n_w, h0)``."""
    n_c = np.asarray(plane["normal"], float)
    n_w = cam.R_wc @ n_c
    return n_w, float(plane["d"]) - float(n_w @ cam.c_w)


def token_rng(token: str) -> np.random.Generator:
    return np.random.default_rng([S.DATA_SEED, int(token, 16)])


def extract_data(depth_u16: np.ndarray, mask: np.ndarray, camera: dict, plane: dict, token: str, *,
                 n: int = S.N_DATA, cam: CameraGeom | None = None) -> FrameData:
    cam = CameraGeom.from_camera(camera) if cam is None else cam
    mask = np.asarray(mask, bool)
    rows, cols = np.nonzero(mask)
    n_blob = int(len(rows))
    if n_blob > n:
        pick = np.sort(token_rng(token).choice(n_blob, size=n, replace=False))
        rows, cols = rows[pick], cols[pick]
    P = e0c_frontend.world_points_at(depth_u16, rows, cols, camera)
    ok = np.isfinite(P).all(axis=1)
    rows, cols, P = rows[ok], cols[ok], P[ok]
    N = np.full(P.shape, np.nan)
    todo = np.ones(len(P), bool)
    Hh, Ww = mask.shape
    for s in S.DATA_NORMAL_STEP_PX:
        if not todo.any():
            break
        r, c = rows[todo], cols[todo]
        inside = (r - s >= 0) & (r + s < Hh) & (c - s >= 0) & (c + s < Ww)
        rr = np.clip(r, s, Hh - 1 - s)
        cc = np.clip(c, s, Ww - 1 - s)
        on = inside & mask[rr, cc - s] & mask[rr, cc + s] & mask[rr - s, cc] & mask[rr + s, cc]
        pu0 = e0c_frontend.world_points_at(depth_u16, rr, cc - s, camera)
        pu1 = e0c_frontend.world_points_at(depth_u16, rr, cc + s, camera)
        pv0 = e0c_frontend.world_points_at(depth_u16, rr - s, cc, camera)
        pv1 = e0c_frontend.world_points_at(depth_u16, rr + s, cc, camera)
        nn = np.cross(pu1 - pu0, pv1 - pv0)
        ln = np.linalg.norm(nn, axis=1)
        good = on & np.isfinite(nn).all(axis=1) & (ln > 0)
        nn = nn / np.where(ln > 0, ln, 1.0)[:, None]
        flip = np.einsum("ij,ij->i", nn, cam.c_w - P[todo]) < 0
        nn[flip] *= -1
        idx = np.flatnonzero(todo)
        N[idx[good]] = nn[good]
        todo[idx[good]] = False
    n_w, h0 = plane_world(plane, cam)
    return FrameData(pts=P, nrm=N, has_n=np.isfinite(N).all(axis=1), plane_n=n_w, plane_h0=h0,
                     blob_dil=e0c_proposal.dilate_square(mask, S.FS_DILATE_PX), cam=cam, n_blob=n_blob,
                     diag={"n_sampled": int(len(P)), "n_with_normal": int(np.isfinite(N).all(axis=1).sum())})


# ------------------------------------------------------------------------------------------- visibility
def visible(Xw: np.ndarray, Nw: np.ndarray, fr: FrameData, *, bin_px: int = S.VIS_BIN_PX, tol: float = S.VIS_TOL_M,
            min_height: float = S.VIS_MIN_HEIGHT_M, front_cos: float = S.VIS_FRONT_COS) -> np.ndarray:
    """(M,) model points that face the camera, are not self-occluded, project into the image and lie above the
    table (module doc, step 2)."""
    cam = fr.cam
    to_cam = cam.c_w - Xw
    dist = np.linalg.norm(to_cam, axis=1)
    front = np.einsum("ij,ij->i", Nw, to_cam) > front_cos * dist
    u, v, z = cam.project(Xw)
    inside = (z > 0) & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
    nbu = cam.W // bin_px + 1
    b = np.where(inside, np.floor(v / bin_px) * nbu + np.floor(u / bin_px), 0).astype(np.int64)
    nb = (cam.H // bin_px + 1) * nbu
    zbuf = np.full(nb, np.inf)
    np.minimum.at(zbuf, b[inside], z[inside])
    unocc = z <= zbuf[b] + tol
    return front & inside & unocc & (fr.height(Xw) > min_height)


# ------------------------------------------------------------------------------------------- ICP
@dataclass(frozen=True)
class IcpConfig:
    point_to_plane: bool = True
    visibility: bool = True
    huber: bool = True
    stages: tuple = S.STAGES
    k: int = S.K_NEIGHBOURS
    normal_compat_deg: float = S.NORMAL_COMPAT_DEG
    damping: float = S.DAMPING
    max_step_m: float = S.MAX_STEP_M
    max_step_deg: float = S.MAX_STEP_DEG
    stop_step_m: float = S.STOP_STEP_M
    stop_step_deg: float = S.STOP_STEP_DEG
    min_corr: int = S.MIN_CORR


CONFIGS = {"u77": IcpConfig(), "fused": IcpConfig(), "u77_best3": IcpConfig(),
           "u77_p2p": IcpConfig(point_to_plane=False), "u77_novis": IcpConfig(visibility=False),
           "u77_l2": IcpConfig(huber=False)}
ARM_MODEL = {"u77": "u77", "fused": "fused", "u77_best3": "u77", "u77_p2p": "u77", "u77_novis": "u77",
             "u77_l2": "u77"}


def placed(model: ModelCloud, Rm: np.ndarray, tm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World positions and normals of the model under the world-to-model map ``m = Rm x + tm``."""
    return (model.pos - tm) @ Rm, model.nrm @ Rm


def correspond(model: ModelCloud, fr: FrameData, Rm, tm, gate: float, cfg: IcpConfig, vis: np.ndarray | None):
    """Data points in the model frame, their matched model indices, distances and the accepted-pair mask."""
    Y = fr.pts @ Rm.T + tm
    k = min(cfg.k, model.n) if vis is not None else 1
    d, idx = model.tree.query(Y, k=k, workers=1)
    if k == 1:
        d, idx = d[:, None], idx[:, None]
    if vis is not None:
        v = vis[idx]
        first = np.argmax(v, axis=1)
        rows = np.arange(len(Y))
        j, dd, ok = idx[rows, first], d[rows, first], v[rows, first]
    else:
        j, dd, ok = idx[:, 0], d[:, 0], np.ones(len(Y), bool)
    ok = ok & (dd <= gate)
    if cfg.normal_compat_deg < 180.0:
        Nd = fr.nrm @ Rm.T
        cosang = np.einsum("ij,ij->i", np.nan_to_num(Nd), model.nrm[j])
        ok &= (~fr.has_n) | (cosang >= math.cos(math.radians(cfg.normal_compat_deg)))
    return Y, j, dd, ok


def _clamp(w: np.ndarray, dt: np.ndarray, cfg: IcpConfig) -> tuple[np.ndarray, np.ndarray]:
    th = np.linalg.norm(w)
    mx = math.radians(cfg.max_step_deg)
    if th > mx:
        w = w * (mx / th)
    lt = np.linalg.norm(dt)
    if lt > cfg.max_step_m:
        dt = dt * (cfg.max_step_m / lt)
    return w, dt


def _step_p2pl(Y, M, Nm, wts, cfg: IcpConfig):
    c = (Y * wts[:, None]).sum(axis=0) / wts.sum()
    r = np.einsum("ij,ij->i", Y - M, Nm)
    J = np.concatenate([np.cross(Y - c, Nm), Nm], axis=1)
    Hm = J.T @ (J * wts[:, None]) + cfg.damping * wts.sum() * np.eye(6)
    g = J.T @ (wts * r)
    x = -np.linalg.solve(Hm, g)
    return c, x[:3], x[3:]


def _step_p2p(Y, M, wts):
    sw = wts.sum()
    cy = (Y * wts[:, None]).sum(axis=0) / sw
    cm = (M * wts[:, None]).sum(axis=0) / sw
    Hm = ((Y - cy) * wts[:, None]).T @ (M - cm)
    U, _, Vt = np.linalg.svd(Hm)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0])
    dR = Vt.T @ D @ U.T
    return cy, rotvec(dR), cm - cy


def icp(model: ModelCloud, fr: FrameData, R0, t0, cfg: IcpConfig = IcpConfig()) -> dict:
    """Refine ``(R0, t0)`` (module doc, steps 2-4). Returns the refined ``R``, ``t`` and the run's diagnostics."""
    Rm, tm = to_internal(R0, t0)
    if fr.n < S.MIN_DATA:
        R, t = from_internal(Rm, tm)
        return {"R": R, "t": t, "status": "too_few_points", "iters": 0, "converged": False, "n_corr": 0}
    it = 0
    status, converged, n_corr = "max_iters", False, 0
    total_rot, total_t = 0.0, 0.0
    last_step = (None, None)
    for si, (n_it, gate, delta) in enumerate(cfg.stages):
        last = si == len(cfg.stages) - 1
        for _ in range(int(n_it)):
            vis = None
            if cfg.visibility:
                Xw, Nw = placed(model, Rm, tm)
                vis = visible(Xw, Nw, fr)
            Y, j, dd, ok = correspond(model, fr, Rm, tm, gate, cfg, vis)
            n_corr = int(ok.sum())
            if n_corr < cfg.min_corr:
                status = "too_few_correspondences"
                break
            Yk, Mk, Nk = Y[ok], model.pos[j[ok]], model.nrm[j[ok]]
            if cfg.point_to_plane:
                res = np.abs(np.einsum("ij,ij->i", Yk - Mk, Nk))
            else:
                res = dd[ok]
            wts = np.minimum(1.0, delta / np.maximum(res, 1e-12)) if cfg.huber else np.ones(len(res))
            if cfg.point_to_plane:
                c, w, dt = _step_p2pl(Yk, Mk, Nk, wts, cfg)
            else:
                c, w, dt = _step_p2p(Yk, Mk, wts)
            w, dt = _clamp(w, dt, cfg)
            dR = rodrigues(w)
            Rm = dR @ Rm
            tm = dR @ (tm - c) + c + dt
            it += 1
            last_step = (float(np.linalg.norm(dt)) * 1e3, math.degrees(np.linalg.norm(w)))
            total_rot += math.degrees(np.linalg.norm(w))
            total_t += float(np.linalg.norm(dt))
            if last and np.linalg.norm(dt) < cfg.stop_step_m and math.degrees(np.linalg.norm(w)) < cfg.stop_step_deg:
                status, converged = "converged", True
                break
        if status in ("converged", "too_few_correspondences"):
            break
    R, t = from_internal(Rm, tm)
    return {"R": R, "t": t, "status": status, "iters": it, "converged": converged, "n_corr": n_corr,
            "path_rot_deg": total_rot, "path_t_mm": total_t * 1e3, "last_step_mm": last_step[0],
            "last_step_deg": last_step[1]}


def fit_quality(model: ModelCloud, fr: FrameData, R, t) -> dict:
    """Truth-free fit of a pose to the frame (module doc, step 5)."""
    Rm, tm = to_internal(R, t)
    Xw, Nw = placed(model, Rm, tm)
    vis = visible(Xw, Nw, fr)
    cfg = IcpConfig()
    Y, j, dd, ok = correspond(model, fr, Rm, tm, S.INLIER_GATE_M, cfg, vis)
    r = np.abs(np.einsum("ij,ij->i", Y - model.pos[j], model.nrm[j]))
    inl = ok & (r <= S.INLIER_RESID_M)
    high = vis & (fr.height(Xw) > S.FS_MIN_HEIGHT_M)
    u, v, z = fr.cam.project(Xw[high])
    ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
    ins = (ui >= 0) & (ui < fr.cam.W) & (vi >= 0) & (vi < fr.cam.H)
    off = ins & ~fr.blob_dil[np.where(ins, vi, 0), np.where(ins, ui, 0)]
    fs = float(off.sum() / max(ins.sum(), 1))
    inlier = float(inl.mean()) if fr.n else 0.0
    return {"inlier_share": inlier, "resid_med_mm": float(np.median(r[inl]) * 1e3) if inl.any() else None,
            "gated_share": float(ok.mean()) if fr.n else 0.0, "free_space_share": fs,
            "n_visible": int(vis.sum()), "score": inlier - fs}


def refine(model: ModelCloud, fr: FrameData, R0, t0, cfg: IcpConfig = IcpConfig(), *, quality: bool = True) -> dict:
    out = icp(model, fr, R0, t0, cfg)
    if quality:
        out["quality"] = fit_quality(model, fr, out["R"], out["t"])
    return out


def refine_best(model: ModelCloud, fr: FrameData, starts: list, cfg: IcpConfig = IcpConfig()) -> dict:
    """Refine every start pose; keep the best truth-free fit (``quality.score``; ties: the earlier start)."""
    runs = [refine(model, fr, R, t, cfg) for R, t in starts]
    best = max(range(len(runs)), key=lambda i: (runs[i]["quality"]["score"], -i))
    out = dict(runs[best])
    out["chosen_start"] = int(best)
    out["scores"] = [r["quality"]["score"] for r in runs]
    return out


@dataclass
class Timer:
    parts: dict = field(default_factory=dict)

    def add(self, name: str, ns: int) -> None:
        self.parts[name] = self.parts.get(name, 0.0) + ns / 1e6


def timed_refine(models: dict, arm: str, obj: str, depth, mask, camera, plane, token, starts,
                 cam=None) -> tuple[dict, dict]:
    """One arm's whole refinement from the frame inputs (data extraction + ICP + fit quality), timed by parts (ms).
    ``models``: ``{"u77": {obj: ModelCloud}, "fused": {...}}``. ``starts``: [(R, t), ...]; only ``u77_best3`` uses
    more than the first."""
    a = time.perf_counter_ns()
    fr = extract_data(depth, mask, camera, plane, token, cam=cam)
    b = time.perf_counter_ns()
    model = models[ARM_MODEL[arm]][obj]
    cfg = CONFIGS[arm]
    if arm == "u77_best3":
        out = refine_best(model, fr, starts, cfg)
    else:
        out = refine(model, fr, starts[0][0], starts[0][1], cfg)
    c = time.perf_counter_ns()
    return out, {"data": (b - a) / 1e6, "icp_and_quality": (c - b) / 1e6, "total": (c - a) / 1e6}


def with_stages(cfg: IcpConfig, stages) -> IcpConfig:
    return replace(cfg, stages=tuple(stages))
