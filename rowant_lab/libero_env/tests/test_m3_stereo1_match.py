"""K0-stereo stereo CLI (m3_stereo1_match.py): G0 synthetic gate and unit tests of the plane-sweep method (PS). The
SG add-on (m3_stereo1_sgbm.py) is not part of this release, so its cases are not here.

Server environment, synthetic numpy arrays only (no LIBERO frame, no MuJoCo, no Perceiver construction).

``ROWANT_G0_OUT=<file.json>`` writes the G0 numbers (per method, width and config) to that file.

G0 (k0_stereo.methods.G0_gate; design section 5): a numpy ray caster, rays through pixel centres (u + 0.5),
3x3 supersampled colour, band-limited textures (Gaussian-filtered noise, sigma 1.2 mm); a textured plane (tilted 20
degrees, 0.25 m on the axis), a textured box, a uniform box, a 5 mm rod at 0.22 m, camera-rigid fingers and hand
(the synthetic hand model's bodies, lit by a camera-fixed light so their image is view-invariant), wrist K (fovy 75),
b 4.5 cm partners along +x, -x, +y and -y of the camera (verging on the axis point at 0.25 m, so the warps and
rectifications carry rotation), and two descent pairs (partner moved 4.5 cm along the optical axis: purely, and with
a 6 mm lateral offset so the view survives selection and the parallax rule is exercised near the epipole).
"""

from __future__ import annotations

import ast
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from rowant_lab.libero_env import m3_stereo1_match as M

MATCH_SRC = Path(M.__file__).read_text()
REPO = Path(M.__file__).resolve().parents[2]
FOVY = 75.0
B_G0 = 0.045
cv2.setNumThreads(8)


# ============================================================================================ synthetic scene

def wrist_K(W: int) -> np.ndarray:
    f = 0.5 * W / math.tan(math.radians(FOVY) / 2.0)
    return np.array([[f, 0.0, W / 2.0], [0.0, f, W / 2.0], [0.0, 0.0, 1.0]])


def make_texture(seed: int, n: int = 1600, pitch: float = 0.0004, sigma_m: float = 0.0012, mean: float = 128.0,
                 std: float = 40.0) -> dict:
    rng = np.random.default_rng(seed)
    t = rng.normal(size=(n, n)).astype(np.float32)
    t = cv2.GaussianBlur(t, (0, 0), sigma_m / pitch, borderType=cv2.BORDER_REFLECT)
    t = (t - t.mean()) / t.std() * std + mean
    return {"img": np.clip(t, 5, 250).astype(np.float32), "pitch": pitch, "n": n}


def tex_lookup(tex: dict, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Bilinear texture read at (s, t) metres from the texture centre."""
    n = int(np.size(s))
    cols = 4096
    rows = max(1, -(-n // cols))
    mx = np.zeros(rows * cols, np.float32)
    my = np.zeros(rows * cols, np.float32)
    mx[:n] = s / tex["pitch"] + tex["n"] / 2.0 - 0.5
    my[:n] = t / tex["pitch"] + tex["n"] / 2.0 - 0.5
    out = cv2.remap(tex["img"], mx.reshape(rows, cols), my.reshape(rows, cols), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT)
    return out.reshape(-1)[:n].astype(np.float64)


def rot_axis(axis, ang) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * Kx + (1 - math.cos(ang)) * Kx @ Kx


def rigid(R=None, t=(0, 0, 0)) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = t
    return T


class Box:
    def __init__(self, pid, T_world_from_box, half, tex=None, albedo=150.0, tint=(1.0, 1.0, 1.0), cam_lit=False):
        self.pid, self.T, self.half = pid, np.asarray(T_world_from_box, float), np.asarray(half, float)
        self.tex, self.albedo, self.tint, self.cam_lit = tex, albedo, np.asarray(tint, float), cam_lit

    def hit(self, o, D):
        R, c = self.T[:3, :3], self.T[:3, 3]
        ol = R.T @ (o - c)
        dl = D @ R
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-self.half - ol) / dl
            t2 = (self.half - ol) / dl
        tlo, thi = np.minimum(t1, t2), np.maximum(t1, t2)
        tlo = np.where(np.isnan(tlo), -np.inf, tlo)
        thi = np.where(np.isnan(thi), np.inf, thi)
        tmin, tmax = tlo.max(axis=1), thi.min(axis=1)
        ax = tlo.argmax(axis=1)
        ok = (tmax >= tmin) & (tmin > 1e-6)
        t = np.where(ok, tmin, np.inf)
        Pl = ol + t[:, None] * dl
        sgn = -np.sign(dl[np.arange(len(dl)), ax])
        nl = np.zeros_like(dl)
        nl[np.arange(len(dl)), ax] = sgn
        n = nl @ R.T
        # face-local texture coordinates (different offsets per face)
        u_ax = (ax + 1) % 3
        v_ax = (ax + 2) % 3
        idx = np.arange(len(dl))
        s = Pl[idx, u_ax] + 0.03 * ax + 0.01 * (sgn > 0)
        tt = Pl[idx, v_ax] - 0.02 * ax
        return t, n, s, tt


class Plane:
    def __init__(self, pid, p0, normal, e1, tex, tint=(1.0, 0.92, 0.75)):
        self.pid, self.p0 = pid, np.asarray(p0, float)
        self.n = np.asarray(normal, float) / np.linalg.norm(normal)
        self.e1 = np.asarray(e1, float) - np.dot(e1, self.n) * self.n
        self.e1 /= np.linalg.norm(self.e1)
        self.e2 = np.cross(self.n, self.e1)
        self.tex, self.tint, self.cam_lit, self.albedo = tex, np.asarray(tint, float), False, None

    def hit(self, o, D):
        den = D @ self.n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((self.p0 - o) @ self.n) / den
        t = np.where(np.isfinite(t) & (t > 1e-6), t, np.inf)
        P = o + np.where(np.isfinite(t), t, 0.0)[:, None] * D
        rel = P - self.p0
        return t, np.broadcast_to(self.n, D.shape), rel @ self.e1, rel @ self.e2


class Rod:
    def __init__(self, pid, centre, axis, radius, half_len, tex, tint=(0.8, 0.9, 1.0)):
        self.pid, self.c = pid, np.asarray(centre, float)
        self.a = np.asarray(axis, float) / np.linalg.norm(axis)
        self.r, self.L, self.tex, self.tint, self.cam_lit, self.albedo = radius, half_len, tex, np.asarray(tint), False, None

    def hit(self, o, D):
        w = o - self.c
        dp = D - np.outer(D @ self.a, self.a)
        wp = w - (w @ self.a) * self.a
        A = (dp * dp).sum(1)
        Bq = 2 * (dp @ wp)
        C = wp @ wp - self.r ** 2
        disc = Bq * Bq - 4 * A * C
        ok = (disc >= 0) & (A > 1e-15)
        sq = np.sqrt(np.where(ok, disc, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (-Bq - sq) / (2 * A)
        t = np.where(ok & (t > 1e-6), t, np.inf)
        P = o + np.where(np.isfinite(t), t, 0.0)[:, None] * D
        s = (P - self.c) @ self.a
        t = np.where(np.abs(s) <= self.L, t, np.inf)
        radial = (P - self.c) - np.outer(s, self.a)
        n = radial / self.r
        b1 = np.cross(self.a, [0.0, 0.0, 1.0])
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(self.a, b1)
        ang = np.arctan2(radial @ b2, radial @ b1)
        return t, n, s, ang * self.r * 4.0


LIGHT_WORLD = np.array([0.3, 0.6, 0.75]) / np.linalg.norm([0.3, 0.6, 0.75])  # direction the light travels
LIGHT_CAM = np.array([0.0, 0.0, 1.0])


def render(prims: list, T_cam: np.ndarray, W: int, ss: int = 3) -> dict:
    """rgb uint8, depth (z along the optical axis of the pixel-centre rays; inf = none) and the primitive id."""
    K = wrist_K(W)
    f, c = K[0, 0], K[0, 2]
    Rc, o = T_cam[:3, :3], T_cam[:3, 3]
    acc = np.zeros((W * W, 3))
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(W, dtype=np.float64))
    depth = pid = None
    for i in range(ss):
        for j in range(ss):
            x = (u + (j + 0.5) / ss - c) / f
            y = (v + (i + 0.5) / ss - c) / f
            Dc = np.stack([x.ravel(), y.ravel(), np.ones(W * W)], 1)
            D = Dc @ Rc.T  # camera z = 1, so t is z along the optical axis
            best = np.full(W * W, np.inf)
            col = np.zeros((W * W, 3))
            ids = np.full(W * W, -1)
            for p in prims:
                t, n, s, tt = p.hit(o, D)
                closer = t < best
                if not closer.any():
                    continue
                best = np.where(closer, t, best)
                ids = np.where(closer, p.pid, ids)
                nn = np.asarray(n)[closer]
                Dw = D[closer]
                flip = (nn * Dw).sum(1) > 0
                nn = np.where(flip[:, None], -nn, nn)
                if p.cam_lit:
                    nc = nn @ Rc  # camera-frame normal: shading rigid with the camera
                    shade = 0.35 + 0.65 * np.maximum(0.0, -(nc @ LIGHT_CAM))
                else:
                    shade = 0.3 + 0.7 * np.maximum(0.0, -(nn @ LIGHT_WORLD))
                alb = tex_lookup(p.tex, s[closer], tt[closer]) if p.tex is not None else np.full(int(closer.sum()),
                                                                                                    p.albedo)
                col[closer] = (alb * shade)[:, None] * p.tint[None, :]
            acc += col
            if i == ss // 2 and j == ss // 2:
                depth, pid = best.reshape(W, W), ids.reshape(W, W)
    rgb = np.clip(np.rint(acc / (ss * ss)), 0, 255).astype(np.uint8).reshape(W, W, 3)
    return {"rgb": rgb, "depth": depth, "pid": pid}


# ---- the synthetic hand model (eef frame) and hand-eye X
PID_PLANE, PID_TBOX, PID_UBOX, PID_ROD, PID_FL, PID_FR, PID_HAND = 1, 2, 3, 4, 5, 6, 7
X_G0 = rigid(rot_axis([1, 0, 0], math.radians(4.0)), (0.004, -0.012, -0.10))  # hand-from-camera
GRIP_Q = np.array([0.012, -0.012])


def box_mesh(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    F = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
         [1, 5, 7], [1, 7, 3]]
    return V, np.array(F)


FINGER_GAP = 0.02
HAND_BODIES = {  # name: (lo, hi in the body frame, T_eef_from_body_at_zero translation, joint)
    "hand": ((-0.045, -0.02, -0.03), (0.045, 0.02, 0.03), (0.0, 0.072, -0.115), None),
    "left_finger": ((-0.009, -0.006, -0.022), (0.009, 0.006, 0.022), (FINGER_GAP, 0.022, -0.03),
                    {"axis_eef": [1.0, 0.0, 0.0], "qpos_index": 0, "sign": 1.0}),
    "right_finger": ((-0.009, -0.006, -0.022), (0.009, 0.006, 0.022), (-FINGER_GAP, 0.022, -0.03),
                     {"axis_eef": [1.0, 0.0, 0.0], "qpos_index": 1, "sign": 1.0}),
}


def hand_model_doc() -> dict:
    bodies = []
    for name, (lo, hi, off, jt) in HAND_BODIES.items():
        V, F = box_mesh(lo, hi)
        bodies.append({"name": name, "T_eef_from_body_at_zero": rigid(None, off).tolist(), "joint": jt,
                       "vertices": V.tolist(), "faces": F.tolist()})
    return {"format": M.HAND_MODEL_FORMAT, "source": {"synthetic": True}, "frame": "eef", "bodies": bodies}


def hand_prims(T_cam: np.ndarray, q=GRIP_Q) -> list:
    """The hand model's boxes as camera-rigid scene primitives for a camera at T_cam (E = T_cam X^-1)."""
    E = T_cam @ np.linalg.inv(X_G0)
    out = []
    pids = {"hand": PID_HAND, "left_finger": PID_FL, "right_finger": PID_FR}
    tex = G0_TEX["finger"]
    for name, (lo, hi, off, jt) in HAND_BODIES.items():
        off = np.asarray(off, float).copy()
        if jt is not None:
            off = off + jt["sign"] * q[jt["qpos_index"]] * np.asarray(jt["axis_eef"])
        lo, hi = np.asarray(lo), np.asarray(hi)
        ctr = off + 0.5 * (lo + hi)
        out.append(Box(pids[name], E @ rigid(None, ctr), 0.5 * (hi - lo), tex=tex if name != "hand" else None,
                       albedo=90.0, tint=(0.7, 0.7, 0.72), cam_lit=True))
    return out


G0_TEX = {"plane": make_texture(11), "tbox": make_texture(12, n=600), "rod": make_texture(13, n=1200, std=50.0),
          "finger": make_texture(14, n=400, std=25.0)}


def g0_world() -> list:
    k = math.tan(math.radians(20.0))
    n = np.array([0.0, k, 1.0]) / math.hypot(k, 1.0)  # into the table (away from the camera)
    plane = Plane(PID_PLANE, (0.0, 0.0, 0.25), -n, (1.0, 0.0, 0.0), G0_TEX["plane"])
    up = -n

    def on_plane(x, y, half, ang):
        z = 0.25 - k * y
        p = np.array([x, y, z]) + up * half[2]
        R = np.stack([np.array([1.0, 0, 0]), np.cross(n, [1.0, 0, 0]), n], 1)
        R = R @ rot_axis([0, 0, 1], math.radians(ang))
        return rigid(R, p)

    hb_t = (0.03, 0.025, 0.02)
    hb_u = (0.04, 0.035, 0.025)
    tbox = Box(PID_TBOX, on_plane(-0.065, 0.005, hb_t, 20.0), hb_t, tex=G0_TEX["tbox"], tint=(0.9, 1.0, 0.8))
    ubox = Box(PID_UBOX, on_plane(0.07, 0.0, hb_u, -10.0), hb_u, tex=None, albedo=170.0, tint=(1.0, 0.85, 0.6))
    rod = Rod(PID_ROD, (0.0, -0.075, 0.22), (1.0, 0.0, 0.0), 0.0025, 0.2, G0_TEX["rod"])
    return [plane, tbox, ubox, rod]


def look_at(centre, target, up_hint=(0.0, -1.0, 0.0)) -> np.ndarray:
    """T_world_from_camera (OpenCV axes) at centre looking at target, x roughly along the reference x."""
    c = np.asarray(centre, float)
    z = np.asarray(target, float) - c
    z /= np.linalg.norm(z)
    x = np.cross(-np.asarray(up_hint, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return rigid(np.stack([x, y, z], 1), c)


G0_TARGET = (0.0, 0.0, 0.25)
G0_POSES = {
    "ref": rigid(),
    # the G0 partners: b 4.5 cm along +x, -x, +y and -y of the camera, same orientation
    "px": rigid(None, (B_G0, 0, 0)), "nx": rigid(None, (-B_G0, 0, 0)),
    "py": rigid(None, (0, B_G0, 0)), "ny": rigid(None, (0, -B_G0, 0)),
    # extra: the same baselines verging on the axis point at 0.25 m (rotation about 10 degrees)
    "vpx": look_at((B_G0, 0, 0), G0_TARGET), "vnx": look_at((-B_G0, 0, 0), G0_TARGET),
    "vpy": look_at((0, B_G0, 0), G0_TARGET), "vny": look_at((0, -B_G0, 0), G0_TARGET),
    # descent pairs: 4.5 cm along the optical axis, purely and with a 6 mm lateral offset
    "dob": rigid(None, (0.006, 0.0, B_G0)), "dax": rigid(None, (0.0, 0.0, B_G0)),
}
G0_LOOKS = {"g0": ("px", "nx", "py", "ny"), "verging": ("vpx", "vnx", "vpy", "vny"), "descent_oblique": ("dob",),
            "descent_axial": ("dax",)}


def g0_views(W: int, names=("ref", "px", "nx", "py", "ny"), hand: bool = False, cache={}) -> dict:
    """Rendered views of the G0 scene (the registered list: plane, boxes, rod), or with ``hand`` the same scene
    with the camera-rigid hand parts in view."""
    out = {}
    for nm in names:
        key = (W, nm, hand)
        if key not in cache:
            T = G0_POSES[nm]
            cache[key] = render(g0_world() + (hand_prims(T) if hand else []), T, W)
        out[nm] = cache[key]
    return out


def look_from_views(views: dict, W: int, partners, select="all") -> dict:
    """A loaded look (the loader's return shape) built in memory from rendered synthetic views."""
    K = wrist_K(W)

    def v(nm, tok):
        return {"view_id": f"traj1-{tok}-robot0_eye_in_hand-{W}", "token": tok, "rgb": views[nm]["rgb"],
                "gripper_qpos": GRIP_Q.copy(), "T": G0_POSES[nm].copy(), "K": K.copy()}

    return {"ref": v("ref", "rs00"), "partners": [v(nm, f"rs{i + 1:02d}") for i, nm in enumerate(partners)],
            "X": X_G0.copy(), "select": select, "look": "traj1-rs", "width": W, "pose_meta": {}}


# ================================================================================================= G0 evaluator

HAND_MODEL = M.parse_hand_model(hand_model_doc())
RIGID_IDS = (PID_FL, PID_FR, PID_HAND)
TEXTURED_IDS = (PID_PLANE, PID_TBOX)
G0_LIMITS = {"PS_plane_median_disp_err_px_max": 0.15, "SG_plane_median_disp_err_px_max": 0.30,
             "textured_coverage_min": 0.95, "uniform_box_valid_wrong_share_max": 0.10,
             "descent_dead_zone_valid_wrong_share_max": 0.01, "backprojection_mm_max": 0.1,
             "rigid_removed_share_min": 0.99, "peak_rss_gib_max": 1.5}


def g0_run(method: str, config: str, W: int, look: str = "g0", hand: bool = False):
    """One G0 look. The registered scene has no hand in view, so it runs without the hand raster; the hand scene
    runs with the synthetic hand model's raster, as the CLI does."""
    parts = G0_LOOKS[look]
    views = g0_views(W, ("ref",) + parts, hand=hand)
    ld = look_from_views(views, W, parts)
    sel = M.select_views(ld)
    cfg = M.parse_config(method, config, W)
    t0 = time.perf_counter()
    try:
        depth, info = M.run_look(ld, sel, cfg, HAND_MODEL if hand else None, want_maps=True)
        status, reason = "ok", None
    except M.LookFailure as exc:
        depth, info, status, reason = np.full((W, W), np.nan, np.float32), {}, exc.status, exc.reason
    wall = time.perf_counter() - t0
    return {"views": views, "look": ld, "sel": sel, "depth": depth, "info": info, "status": status, "reason": reason,
            "wall_s": wall}


def _disp_err(depth, zt, W):
    f = wrist_K(W)[0, 0]
    return np.abs(depth - zt) * f * B_G0 / zt ** 2


def _dead_zone(zt: np.ndarray, T_p: np.ndarray, W: int) -> np.ndarray:
    """Pixels where moving the true depth by +5 % or -5 % shifts the projection in the partner by under 1 px."""
    K = wrist_K(W)
    g = M.PartnerGeom(K, np.eye(4), K, T_p, (W, W))
    good = np.isfinite(zt) & (zt > 0)
    rho = np.where(good, 1.0 / np.where(good, zt, 1.0), 1.0).astype(np.float32)
    mx0, my0, _ = g.maps(rho)
    sh = []
    for s in (1.05, 0.95):
        mx1, my1, _ = g.maps(rho / np.float32(s))
        sh.append(np.hypot(mx1 - mx0, my1 - my0))
    return good & (np.minimum(sh[0], sh[1]) < 1.0)


def g0_numbers(method: str, config: str, W: int, with_verging: bool = True) -> dict:
    out = {"method": method, "config": config, "width": W, "tau_tex": M.TAU_TEX}
    r = g0_run(method, config, W, "g0")
    ref = r["views"]["ref"]
    pid, zt = ref["pid"], ref["depth"]
    d = r["depth"]
    valid = np.isfinite(d)
    de = _disp_err(d, zt, W)
    plane = pid == PID_PLANE
    pv = plane & valid
    out["g0_status"] = r["status"]
    out["plane_median_disp_err_px"] = float(np.median(de[pv])) if pv.any() else None
    out["plane_p90_disp_err_px"] = float(np.percentile(de[pv], 90)) if pv.any() else None
    out["plane_median_abs_dz_mm"] = float(np.median(np.abs(d - zt)[pv]) * 1e3) if pv.any() else None
    tex = np.isin(pid, TEXTURED_IDS)
    out["textured_coverage"] = float((valid & tex).sum() / tex.sum())
    out["plane_coverage"] = float(pv.sum() / plane.sum())
    for nm, ids in (("textured_box", (PID_TBOX,)), ("rod", (PID_ROD,))):
        m = np.isin(pid, ids)
        v = m & valid
        out[f"{nm}_coverage"] = float(v.sum() / m.sum())
        out[f"{nm}_valid_wrong_10mm_share"] = float((v & (np.abs(d - zt) > 0.01)).sum() / m.sum())
        out[f"{nm}_median_disp_err_px"] = float(np.median(de[v])) if v.any() else None
    ub = pid == PID_UBOX
    out["uniform_box_px"] = int(ub.sum())
    out["uniform_box_valid_share"] = float((ub & valid).sum() / ub.sum())
    out["uniform_box_valid_wrong_share"] = float((ub & valid & (np.abs(d - zt) > 0.01)).sum() / ub.sum())
    u16 = M.encode_map(d)
    out["holes_encoded_zero"] = bool(np.all(u16[~valid] == 0)) and bool(np.all(u16[valid] > 0))
    out["any_65535"] = bool(np.any(u16 == 65535))
    out["wall_s"] = round(r["wall_s"], 3)
    out["info"] = {k: v for k, v in r["info"].items() if k not in ("maps", "masks")}
    # the hand scene: camera-rigid hand parts in view, the hand raster on (the rigid-mask line)
    rh = g0_run(method, config, W, "g0", hand=True)
    pid_h, zt_h, dh = rh["views"]["ref"]["pid"], rh["views"]["ref"]["depth"], rh["depth"]
    vh = np.isfinite(dh)
    rig = np.isin(pid_h, RIGID_IDS)
    rig_in = cv2.erode(rig.astype(np.uint8), np.ones((M.RIGID_WIN, M.RIGID_WIN), np.uint8)).astype(bool)
    masks = rh["info"]["masks"]
    tex_h = np.isin(pid_h, TEXTURED_IDS)
    hd = {"status": rh["status"], "rigid_px": int(rig.sum()), "rigid_interior_px": int(rig_in.sum()),
          "rigid_removed_share": float((masks["rigid"] & rig_in).sum() / rig_in.sum()),
          "rigid_removed_share_all_px": float((masks["rigid"] & rig).sum() / rig.sum()),
          "rigid_or_hand_removed_share_all_px": float(((masks["rigid"] | masks["hand"]) & rig).sum() / rig.sum()),
          "rigid_valid_px": int((rig & vh).sum()),
          "hand_raster_covers_hand_share": float((masks["hand"] & rig).sum() / rig.sum()),
          "hand_raster_on_scene_share": float((masks["hand"] & ~rig).sum() / (~rig).sum()),
          "rigid_mask_on_textured_share": float((masks["rigid"] & tex_h).sum() / tex_h.sum()),
          "textured_coverage": float((vh & tex_h).sum() / tex_h.sum()),
          "plane_median_disp_err_px": float(np.median(_disp_err(dh, zt_h, W)[vh & (pid_h == PID_PLANE)])),
          "wall_s": round(rh["wall_s"], 3)}
    out["hand_scene"] = hd
    out["rigid_removed_share"] = hd["rigid_removed_share"]
    for look in ("descent_oblique", "descent_axial"):
        rd = g0_run(method, config, W, look)
        dz = _dead_zone(zt, G0_POSES[G0_LOOKS[look][0]], W)
        vd = np.isfinite(rd["depth"])
        out[look] = {"status": rd["status"], "reason": rd["reason"],
                     "b_eff_m": round(M.b_eff(np.eye(4), G0_POSES[G0_LOOKS[look][0]]), 6),
                     "dead_zone_px": int(dz.sum()),
                     "dead_zone_valid_wrong_share": float((dz & vd & (np.abs(rd["depth"] - zt) > 0.01)).sum()
                                                          / max(int(dz.sum()), 1)),
                     "dead_zone_valid_share": float((dz & vd).sum() / max(int(dz.sum()), 1)),
                     "outside_dead_zone_valid_share": float((~dz & vd).sum() / max(int((~dz).sum()), 1)),
                     "valid_wrong_10mm_share_all": float((vd & (np.abs(rd["depth"] - zt) > 0.01)).sum()
                                                         / max(int(vd.sum()), 1))}
    if with_verging:
        rv = g0_run(method, config, W, "verging")
        dv = rv["depth"]
        vv = np.isfinite(dv)
        pvv = plane & vv
        out["verging"] = {"status": rv["status"],
                          "plane_median_disp_err_px": float(np.median(_disp_err(dv, zt, W)[pvv])) if pvv.any() else None,
                          "textured_coverage": float((vv & tex).sum() / tex.sum()),
                          "uniform_box_valid_wrong_share": float((ub & vv & (np.abs(dv - zt) > 0.01)).sum() / ub.sum()),
                          "wall_s": round(rv["wall_s"], 3)}
    return out


def g0_pass_lines(n: dict) -> dict:
    lim = G0_LIMITS
    pmax = lim[f"{n['method']}_plane_median_disp_err_px_max"]
    return {"plane_disp_err": n["plane_median_disp_err_px"] is not None and n["plane_median_disp_err_px"] <= pmax,
            "textured_coverage": n["textured_coverage"] >= lim["textured_coverage_min"],
            "uniform_box": n["uniform_box_valid_wrong_share"] <= lim["uniform_box_valid_wrong_share_max"],
            "descent_oblique": n["descent_oblique"]["dead_zone_valid_wrong_share"]
            <= lim["descent_dead_zone_valid_wrong_share_max"],
            "descent_axial": n["descent_axial"]["dead_zone_valid_wrong_share"]
            <= lim["descent_dead_zone_valid_wrong_share_max"],
            "holes": n["holes_encoded_zero"] and not n["any_65535"],
            "rigid": n["rigid_removed_share"] >= lim["rigid_removed_share_min"]}


def g0_out(key: str, value) -> None:
    p = os.environ.get("ROWANT_G0_OUT")
    if not p:
        return
    p = Path(p)
    doc = json.loads(p.read_text()) if p.exists() else {}
    doc[key] = value
    p.write_text(json.dumps(doc, indent=1, sort_keys=True, default=float))


# ======================================================================================================= G0 tests

@pytest.mark.parametrize("method,W", [("PS", 512), ("PS", 1024)])
def test_g0_reference_config_passes(method, W):
    cfg = M.REFERENCE_CONFIG[method][W]
    n = g0_numbers(method, cfg, W, with_verging=(method == "PS"))
    lines = g0_pass_lines(n)
    n["pass_lines"] = lines
    g0_out(f"{method}-{W}-{cfg}", n)
    assert n["g0_status"] == "ok"
    assert all(lines.values()), lines
    if method == "PS":  # the verging look (rotation in every warp) passes the PS lines too
        v = n["verging"]
        assert v["plane_median_disp_err_px"] <= G0_LIMITS["PS_plane_median_disp_err_px_max"]
        assert v["textured_coverage"] >= G0_LIMITS["textured_coverage_min"]
        assert v["uniform_box_valid_wrong_share"] <= G0_LIMITS["uniform_box_valid_wrong_share_max"]
    if method == "SG":  # every branch: plain, mirrored, transposed, transposed and mirrored
        br = {(p["vertical"], p["mirrored"]) for p in n["info"]["pairs"]}
        assert br == {(False, False), (False, True), (True, False), (True, True)}
    assert n["descent_axial"]["status"] == "failed"  # b_eff 0: the view is dropped
    if method == "PS":
        assert n["descent_oblique"]["status"] == "ok" and n["descent_oblique"]["dead_zone_px"] > 1000
    else:
        assert n["descent_oblique"]["status"] == "failed"  # epipole inside the image: the only pair is skipped


@pytest.mark.skipif(not os.environ.get("ROWANT_G0_OUT"), reason="the full grid is computed only for the G0 report")
@pytest.mark.parametrize("method,W", [("PS", 512), ("PS", 1024)])
def test_g0_grid_numbers(method, W):
    grid = ([f"PS-w{w}-{g}" for w in M.PS_W[W] for g in ("loose", "strict")] if method == "PS"
            else [f"SG-b{b}-u{u}" for b in M.SG_BS[W] for u in M.SG_UR])
    for cfg in grid:
        n = g0_numbers(method, cfg, W, with_verging=True)
        n["pass_lines"] = g0_pass_lines(n)
        g0_out(f"grid/{method}-{W}-{cfg}", n)


@pytest.mark.parametrize("W", [512, 1024])
def test_g0_encoding_and_backprojection_through_frame_arrays(W, tmp_path):
    """Plane depth written through the stereo's output path and read back through m2_perceive._frame_arrays lies
    within 0.1 mm of the same back-projection applied to the float depth (encoding and packaging only)."""
    from rowant_lab.libero_env import m2_perceive as MP

    ref = g0_views(W, ("ref",))["ref"]
    plane = ref["pid"] == PID_PLANE
    zf = np.where(plane, ref["depth"], np.nan)
    u16 = M.encode_map(zf)
    K = wrist_K(W)
    T = look_at((0.31, -0.12, 0.83), (0.0, 0.05, 0.0))
    p = tmp_path / "net" / "g0" / "x.npz"
    M._excl_write(p, M._npz_bytes(depth_u16=u16, valid=u16 > 0, K_used=K, meta=np.asarray("{}")))
    with np.load(p, allow_pickle=False) as z:
        fr = {"depth_u16": z["depth_u16"], "depth_scale_mm": 0.1, "K": z["K_used"], "T_world_from_camera": T}
    _, _, _, _, _, Pw = MP.Perceiver._frame_arrays(fr)
    H, Wd = zf.shape
    v, u = np.mgrid[0:H, 0:Wd].astype(np.float64)
    Pc = np.stack([(u - K[0, 2]) / K[0, 0] * zf, (v - K[1, 2]) / K[1, 1] * zf, zf], -1)
    Pw_f = Pc @ T[:3, :3].T + T[:3, 3]
    err = np.linalg.norm(Pw - Pw_f, axis=-1)[plane]
    assert np.all(np.isfinite(err))
    assert float(err.max()) * 1e3 <= G0_LIMITS["backprojection_mm_max"]
    assert np.all(u16[~plane] == 0) and not np.any(u16 == 65535)
    g0_out(f"backprojection-{W}", {"max_mm": float(err.max()) * 1e3, "n_px": int(plane.sum())})


_RSS_CHILD = r"""
import json, resource, sys, time
import numpy as np, cv2
from rowant_lab.libero_env import m3_stereo1_match as M
cv2.setNumThreads(8)
a = np.load(sys.argv[1], allow_pickle=False)
W = int(a["W"]); method, config = sys.argv[2], sys.argv[3]
names = [str(x) for x in a["names"]]
def v(i, nm):
    return {"view_id": f"traj1-{nm}-robot0_eye_in_hand-{W}", "token": nm, "rgb": a[f"rgb_{nm}"],
            "gripper_qpos": a["q"], "T": a[f"T_{nm}"], "K": a["K"]}
ld = {"ref": v(0, names[0]), "partners": [v(i, nm) for i, nm in enumerate(names[1:])], "X": a["X"],
      "select": "all", "look": "traj1-rs", "width": W, "pose_meta": {}}
model = M.load_hand_model(sys.argv[4])
rss0 = M.peak_rss_bytes()
sel = M.select_views(ld)
t0 = time.perf_counter()
depth, info = M.run_look(ld, sel, M.parse_config(method, config, W), model)
print(json.dumps({"peak_rss_bytes": M.peak_rss_bytes(), "rss_before_bytes": rss0,
                  "wall_s": time.perf_counter() - t0, "valid_px": int(np.isfinite(depth).sum())}))
"""


def _child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(REPO)
    return env


@pytest.mark.parametrize("method", ["PS"])
def test_g0_peak_rss_at_1024(method, tmp_path):
    W = 1024
    names = ("ref",) + G0_LOOKS["g0"]
    views = g0_views(W, names, hand=True)
    arrs = {"W": np.asarray(W), "names": np.asarray(names), "K": wrist_K(W), "X": X_G0, "q": GRIP_Q}
    for nm in names:
        arrs[f"rgb_{nm}"] = views[nm]["rgb"]
        arrs[f"T_{nm}"] = G0_POSES[nm]
    np.savez(tmp_path / "look.npz", **arrs)
    (tmp_path / "hand.json").write_text(json.dumps(hand_model_doc()))
    cfg = M.REFERENCE_CONFIG[method][W]
    res = subprocess.run([sys.executable, "-B", "-c", _RSS_CHILD, str(tmp_path / "look.npz"), method, cfg,
                          str(tmp_path / "hand.json")], capture_output=True, text=True, env=_child_env(),
                         cwd=str(tmp_path), timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    out["peak_rss_gib"] = out["peak_rss_bytes"] / 2 ** 30
    g0_out(f"rss-{method}-1024-{cfg}", out)
    assert out["peak_rss_gib"] <= G0_LIMITS["peak_rss_gib_max"]
    assert out["valid_px"] > 0


# ========================================================================= synthetic timings (opt-in, numbers only)

def _ring_pose(i: int, n: int, radius: float, tilt_deg: float) -> np.ndarray:
    a = 2 * math.pi * i / n
    c = (radius * math.cos(a), radius * math.sin(a), 0.0)
    T = look_at(c, G0_TARGET)
    return T @ rigid(rot_axis([math.cos(a), math.sin(a), 0.0], math.radians(tilt_deg)))


@pytest.mark.skipif(not os.environ.get("ROWANT_G0_TIMING"), reason="timing runs only for the G0 report")
@pytest.mark.parametrize("W", [512, 1024])
def test_g0_synthetic_look_timings(W):
    """Per-look wall times of the reference configs on an rs-like look (10 partners on a 4.5 cm ring, rotated) and
    a pg-like look (3 partners at 0.8, 2.0 and 3.5 cm along camera x, same orientation), hand scene."""
    extra = {f"ring{i}": _ring_pose(i, 10, B_G0, 4.0) for i in range(10)}
    extra.update({"pg1": rigid(None, (0.008, 0, 0)), "pg2": rigid(None, (0.020, 0, 0)), "pg3": rigid(None, (0.035, 0, 0))})
    G0_POSES.update(extra)
    G0_LOOKS["rs_like"] = tuple(f"ring{i}" for i in range(10))
    G0_LOOKS["pg_like"] = ("pg1", "pg2", "pg3")
    res = {}
    for look in ("pg_like", "rs_like", "g0"):
        for method in ("PS",):
            cfg = M.REFERENCE_CONFIG[method][W]
            walls = []
            for _rep in range(3):
                r = g0_run(method, cfg, W, look, hand=True)
                walls.append(r["wall_s"])
            res[f"{look}/{method}/{cfg}"] = {"wall_s": [round(x, 3) for x in walls], "status": r["status"],
                                             "valid_px": int(np.isfinite(r["depth"]).sum()),
                                             "n_partners": len(G0_LOOKS[look]),
                                             "n_planes": r["info"].get("n_planes"),
                                             "load_1min": round(os.getloadavg()[0], 2)}
    g0_out(f"timing-{W}", res)


# ================================================================================================== unit tests

def test_look_keys_view_ids_and_specs():
    assert M.parse_look_key("abc-t01-s02/traj1-pg/512") == ("abc-t01-s02", "traj1-pg", 512)
    for bad in ("a/b", "a/traj1-xx/512", "../traj1-pg/512", "a b/traj1-pg/512", "/traj1-pg/512"):
        with pytest.raises(ValueError):
            M.parse_look_key(bad)
    assert M.view_id("traj2-E2", "C", 1024) == "traj2-C-robot0_eye_in_hand-1024"
    assert M.look_spec("traj1-pg", "S3-b1")["partners"] == ("pgp1",)
    assert M.look_spec("traj1-pg", "S3-b2")["partners"] == ("pgp1", "pgp2")
    assert M.look_spec("traj1-pg", "S3")["partners"] == ("pgp1", "pgp2", "pgp3")
    assert M.look_spec("traj1-pgm", "S3-RM")["select"] == "rm"
    assert M.look_spec("traj1-rs", "S2")["partners"] == tuple(f"rs{i:02d}" for i in range(1, 11))
    with pytest.raises(ValueError):
        M.look_spec("traj1-rs", "S3-b1")


def test_config_grid_per_width():
    assert M.parse_config("PS", "PS-w5-loose", 512) == {"method": "PS", "w": 5, "gate": "loose", "zncc_min": 0.5,
                                                        "margin_min": 0.02}
    assert M.parse_config("PS", "PS-w13-strict", 1024)["zncc_min"] == 0.7
    assert M.parse_config("SG", "SG-b11-u5", 1024) == {"method": "SG", "bs": 11, "ur": 5}
    for m, c, w in (("PS", "PS-w9-loose", 512), ("PS", "PS-w5-loose", 1024), ("SG", "SG-b5-u12", 1024),
                    ("SG", "SG-b5-u7", 512), ("PS", "PS-w5-loose", 256), ("XX", "PS-w5-loose", 512)):
        with pytest.raises(ValueError):
            M.parse_config(m, c, w)
    n_ps = sum(1 for w in (512, 1024) for x in M.PS_W[w] for g in M.PS_GATES)
    n_sg = sum(1 for w in (512, 1024) for x in M.SG_BS[w] for u in M.SG_UR)
    assert n_ps == 8 and n_sg == 8  # 4 configs per method and width
    for meth in ("PS", "SG"):
        for w in (512, 1024):
            M.parse_config(meth, M.REFERENCE_CONFIG[meth][w], w)


def test_b_eff_is_perpendicular_to_the_principal_ray():
    T0 = look_at((0.1, 0.2, 0.3), (0.4, -0.1, 0.9))
    z = T0[:3, 2]
    x = T0[:3, 0]
    T1 = T0.copy()
    T1[:3, 3] += 0.07 * z + 0.02 * x
    assert M.b_eff(T0, T1) == pytest.approx(0.02, abs=1e-12)
    T2 = T0.copy()
    T2[:3, 3] += 0.05 * z
    assert M.b_eff(T0, T2) == pytest.approx(0.0, abs=1e-12)


def _view(tok, T, rgb=None, W=64):
    return {"view_id": f"traj1-{tok}-robot0_eye_in_hand-{W}", "token": tok,
            "rgb": rgb if rgb is not None else np.zeros((W, W, 3), np.uint8), "gripper_qpos": np.zeros(2), "T": T,
            "K": wrist_K(W)}


def test_view_selection_drops_small_baselines_and_fails_when_all_dropped():
    ref = _view("rs00", np.eye(4))
    parts = [_view("rs01", rigid(None, (0.004, 0, 0))), _view("rs02", rigid(None, (0.0, 0.006, 0.05))),
             _view("rs03", rigid(None, (0, 0, 0.09)))]
    sel = M.select_views({"ref": ref, "partners": parts, "select": "all"})
    assert [p["token"] for p in sel["used"]] == ["rs02"] and sel["status"] is None
    assert set(sel["dropped"]) == {parts[0]["view_id"], parts[2]["view_id"]}
    sel = M.select_views({"ref": ref, "partners": [parts[0], parts[2]], "select": "all"})
    assert sel["status"] == "failed" and sel["reason"] == "every view dropped" and not sel["used"]
    sel = M.select_views({"ref": ref, "partners": [], "select": "all"})  # a blocked look: no partner frames
    assert sel["status"] == "failed" and sel["reason"] == "every view dropped"


def test_rm_partners_greedy_by_b_eff_with_separation():
    T0 = np.eye(4)
    Ts = [rigid(None, (0.075, 0, 0)), rigid(None, (0.07, 0.004, 0)), rigid(None, (0.05, 0, 0)),
          rigid(None, (0.0, 0.03, 0)), rigid(None, (0.0, 0.0085, 0)), rigid(None, (0.09, 0, 0)),
          rigid(None, (0.0, 0.025, 0.0))]
    # 0.09 is above 8 cm; 0.075 first; 0.07 is 6.4 mm from it; 0.05; 0.03; then three are chosen
    assert M.rm_partners(T0, Ts) == [0, 2, 3]
    assert M.rm_partners(T0, [rigid(None, (0.005, 0, 0)), rigid(None, (0.09, 0, 0))]) == []
    parts = [_view(f"pgm{i + 1:02d}", T) for i, T in enumerate([rigid(None, (0.005, 0, 0)), rigid(None, (0.0, 0.009, 0))])]
    sel = M.select_views({"ref": _view("pgr0", T0), "partners": parts, "select": "rm"})
    assert sel["status"] == "degenerate" and not sel["used"]
    sel = M.select_views({"ref": _view("pgr0", T0), "partners": [], "select": "rm"})
    assert sel["status"] == "failed"


def test_rigid_or_flat_mask_and_the_empty_partner_set_rule():
    rng = np.random.default_rng(3)
    ref = rng.uniform(0, 255, (64, 64)).astype(np.float32)
    moved = np.roll(ref, 9, axis=1)
    same = ref.copy()
    # against the >= 1 cm partner only: the 0.6 cm partner's identical image does not mask
    m = M.rigid_or_flat_mask(ref, [same, moved], [0.006, 0.02])
    assert not m[10:-10, 10:-10].any()
    m = M.rigid_or_flat_mask(ref, [same, same], [0.006, 0.02])
    assert m.all()
    # no used partner with b_eff >= 1 cm: against every used partner
    assert M.rigid_or_flat_mask(ref, [same, same], [0.006, 0.007]).all()
    assert not M.rigid_or_flat_mask(ref, [same, moved], [0.006, 0.007])[10:-10, 10:-10].any()
    assert not M.rigid_or_flat_mask(ref, [], []).any()
    flat = np.full((64, 64), 120.0, np.float32)
    assert M.rigid_or_flat_mask(flat, [flat + 1.5], [0.03]).all()  # flat: under 2.0 grey levels
    assert not M.rigid_or_flat_mask(flat, [flat + 2.5], [0.03]).any()


def test_best_half_mean_and_parabola():
    inf = np.inf
    c = [np.array([0.1, inf, 0.3, inf], np.float32), np.array([0.2, 0.5, inf, inf], np.float32),
         np.array([0.6, 0.4, 0.2, inf], np.float32)]
    out = M.best_half_mean(c)  # n = 3, 2, 2, 0 -> best 2, 1, 1, none
    assert out[0] == pytest.approx(0.15) and out[1] == pytest.approx(0.4) and out[2] == pytest.approx(0.2)
    assert np.isinf(out[3])
    four = [np.array([v], np.float32) for v in (0.4, 0.1, 0.3, 0.2)]
    assert M.best_half_mean(four)[0] == pytest.approx(0.15)
    five = [np.array([v], np.float32) for v in (0.5, 0.1, 0.3, 0.2, 0.4)]
    assert M.best_half_mean(five)[0] == pytest.approx(0.2)  # ceil(5 / 2) = 3
    assert M.best_half_mean([np.array([0.3, inf], np.float32), np.array([0.1, 0.2], np.float32)]).tolist() == \
        pytest.approx([0.1, 0.2])
    off = M.parabola_offset(np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 3.0]))
    assert off[0] == pytest.approx(0.0) and off[1] == pytest.approx(-0.25)


def test_min_tracker_best3_margin_and_edges():
    xs = np.arange(40, dtype=np.float64)
    curves = np.stack([0.5 + 0.01 * (xs - 17.3) ** 2,  # one minimum at 17.3
                       np.minimum(0.2 + 0.02 * (xs - 10) ** 2, 0.25 + 0.02 * (xs - 30) ** 2),  # two minima
                       0.9 - 0.01 * xs,  # minimum at the last sample: no reading
                       0.1 + 0.01 * xs], 1).astype(np.float32)  # minimum at the first sample
    tr = M.MinTracker((4,))
    for c in curves:
        tr.push(c)
    r = tr.result()
    assert r["ok"].tolist() == [True, True, False, False]
    assert r["k"][0] == 17 and r["k"][0] + r["off"][0] == pytest.approx(17.3, abs=1e-3)
    assert np.isinf(r["margin"][0])
    assert r["k"][1] == 10 and r["margin"][1] == pytest.approx(0.05, abs=1e-6)


def test_flying_pixels_and_encoding():
    z = np.full((8, 8), 0.30, np.float32)
    z[:, 4:] = 0.40
    v = np.ones((8, 8), bool)
    fly = M.flying_pixels(z, v)
    assert fly[:, 3:5].all() and not fly[:, :3].any() and not fly[:, 5:].any()
    v2 = v.copy()
    v2[:, 4:] = False
    assert not M.flying_pixels(z, v2).any()
    d = np.array([[np.nan, 0.25, 7.0], [0.0, -1.0, 6.4]], np.float32)
    u = M.encode_map(d)
    assert u.dtype == np.uint16 and u.tolist() == [[0, 2500, 0], [0, 0, 64000]]


def test_the_65535_assert(monkeypatch):
    monkeypatch.setattr(M.m2_camera, "encode_depth", lambda d, z: np.full(np.shape(d), 65535, np.uint16))
    with pytest.raises(AssertionError):
        M.encode_map(np.ones((4, 4)))


def test_photometric_arms_perturb_partners_deterministically():
    rgb = np.random.default_rng(5).integers(0, 256, (32, 32, 3)).astype(np.uint8)
    a = M.photometric(rgb, 0.6)
    assert np.array_equal(a, M.photometric(rgb, 0.6))
    assert not np.array_equal(a, M.photometric(rgb, 1.4))
    rgb2 = rgb.copy()
    rgb2[0, 0, 0] ^= 1
    d = M.photometric(rgb2, 0.6).astype(int) - np.clip(np.rint(rgb2 * 0.6), 0, 255).astype(int)
    e = a.astype(int) - np.clip(np.rint(rgb * 0.6), 0, 255).astype(int)
    assert not np.array_equal(d, e)  # independent per view (seeded by the view's own bytes)
    assert 2.0 < float(np.std(e)) < 4.0


def test_warp_at_the_true_depth_matches_the_partner():
    W = 512
    views = g0_views(W, ("ref", "px", "ny"))
    ref = views["ref"]
    L = M.luma(ref["rgb"])
    rw = M.RefWindow(L, 5)
    plane = ref["pid"] == PID_PLANE
    for nm in ("px", "ny"):
        J, inside = M.warp_to_reference(M.luma(views[nm]["rgb"]), ref["depth"], wrist_K(W), np.eye(4), wrist_K(W),
                                        G0_POSES[nm])
        z = rw.zncc(J)
        m = M.window_all(inside, 5) & cv2.erode(plane.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        m &= np.isfinite(z)
        assert m.sum() > 50000
        assert float(np.median(z[m])) > 0.98
        # a wrong partner pose (the other partner's) breaks it: the check this warp serves
        Jw, iw = M.warp_to_reference(M.luma(views[nm]["rgb"]), ref["depth"], wrist_K(W), np.eye(4), wrist_K(W),
                                     G0_POSES["px" if nm == "ny" else "ny"])
        zw = rw.zncc(Jw)
        mw = M.window_all(iw, 5) & m & np.isfinite(zw)
        assert float(np.median(zw[mw])) < 0.5


def test_hand_raster_matches_the_rendered_hand_and_catches_faults():
    W = 512
    ref = g0_views(W, ("ref",), hand=True)["ref"]
    hand = np.isin(ref["pid"], RIGID_IDS)
    K = wrist_K(W)
    r0 = M.hand_raster(HAND_MODEL, GRIP_Q, X_G0, K, W, dilate=False)
    iou = (r0 & hand).sum() / (r0 | hand).sum()
    assert iou > 0.93
    r = M.hand_raster(HAND_MODEL, GRIP_Q, X_G0, K, W)
    assert (r & hand).sum() / hand.sum() == 1.0
    # the dilation is a disc of 3 px at 512 and 6 px at 1024
    for Wd, rad in ((512, 3), (1024, 6)):
        a = M.hand_raster(HAND_MODEL, GRIP_Q, X_G0, wrist_K(Wd), Wd, dilate=False)
        b = M.hand_raster(HAND_MODEL, GRIP_Q, X_G0, wrist_K(Wd), Wd)
        dist = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        assert np.all(b[a]) and float(dist[b].max()) <= rad + 0.5  # OpenCV's elliptical (discrete disc) kernel
        assert np.all(b[dist <= rad - 1.0])
    # a finger-qpos sign fault moves the fingers
    doc = hand_model_doc()
    for bd in doc["bodies"]:
        if bd["joint"] is not None:
            bd["joint"]["sign"] = -bd["joint"]["sign"]
    flipped = M.hand_raster(M.parse_hand_model(doc), np.array([0.03, -0.03]), X_G0, K, W, dilate=False)
    right = M.hand_raster(HAND_MODEL, np.array([0.03, -0.03]), X_G0, K, W, dilate=False)
    assert (flipped & right).sum() < 0.8 * right.sum()
    # X on the wrong side, and the hand drawn at the camera instead (a wrong X), each miss the hand
    for Xb in (np.linalg.inv(X_G0), np.eye(4)):
        rb = M.hand_raster(HAND_MODEL, GRIP_Q, Xb, K, W, dilate=False)
        assert (rb & hand).sum() < 0.5 * hand.sum()


def test_hand_raster_clips_triangles_behind_the_camera():
    doc = {"format": M.HAND_MODEL_FORMAT, "frame": "eef", "bodies": [
        {"name": "b", "T_eef_from_body_at_zero": np.eye(4).tolist(), "joint": None,
         "vertices": [[-0.05, -0.05, -0.02], [0.05, -0.05, -0.02], [0.0, 0.05, 0.10]], "faces": [[0, 1, 2]]}]}
    m = M.hand_raster(M.parse_hand_model(doc), np.zeros(2), np.eye(4), wrist_K(128), 128, dilate=False)
    assert m[:100].any() and not m[110:].any()  # the apex projects at row 105; the clipped part fills above it


def test_hand_model_parse_refuses_bad_files():
    doc = hand_model_doc()
    doc["frame"] = "world"
    with pytest.raises(ValueError):
        M.parse_hand_model(doc)
    doc = hand_model_doc()
    doc["bodies"][0]["faces"][0] = [0, 1, 99]
    with pytest.raises(ValueError):
        M.parse_hand_model(doc)


def test_registered_hand_model_file_parses():
    p = REPO / M.HAND_MODEL_REL
    if not p.exists():
        pytest.skip("the hand model file is written at C2")
    m = M.load_hand_model(p)
    assert len(m["bodies"]) >= 3 and sum(b["joint"] is not None for b in m["bodies"]) == 2


def test_registered_constants():
    assert (M.Z_MIN_M, M.Z_MAX_M) == (0.08, 1.20)
    assert (M.B_DROP_M, M.B_RIGID_M, M.RM_B_MIN_M, M.RM_B_MAX_M, M.RM_SEP_M, M.RM_MAX) == (0.005, 0.01, 0.01, 0.08,
                                                                                        0.01, 3)
    assert (M.RIGID_WIN, M.RIGID_MAD_MAX, M.FLY_REL, M.PARALLAX_REL, M.PARALLAX_PX) == (5, 2.0, 0.03, 0.05, 1.0)
    assert (M.REFINE_HALF_PX, M.REFINE_STEP_PX, M.HAND_DILATE_PX_AT_512, M.CV2_THREADS) == (2.0, 0.25, 3, 8)
    assert M.PS_W == {512: (5, 7), 1024: (9, 13)} and M.PS_GATES == {"loose": (0.5, 0.02), "strict": (0.7, 0.05)}
    assert M.SG_BS == {512: (5, 7), 1024: (7, 11)} and M.SG_UR == (5, 12)
    assert M.PHOTOMETRIC == {"P1": 0.6, "P2": 1.4} and M.PHOTO_SIGMA == 3.0 and M.PHOTO_SEED == 7503
    assert M.TAU_TEX == 0.5


# ================================================================================================== loader + CLI

POLICY_KEYS_V5 = ("rgb", "camera", "width", "step", "robot0_joint_pos", "robot0_joint_vel", "robot0_eef_pos",
                  "robot0_eef_quat", "robot0_gripper_qpos", "robot0_gripper_qvel", "prev_eef_pos", "prev_eef_quat",
                  "prev_step", "obs_sha256", "look_id", "look_k", "steps_since_look_start", "code_tree", "clean")


def write_policy_frame(root: Path, scene: str, vid: str, rgb: np.ndarray, T: np.ndarray, extra: dict | None = None):
    E = T @ np.linalg.inv(X_G0)
    q = M_quat(E[:3, :3])
    arr = {"rgb": rgb, "camera": np.asarray("robot0_eye_in_hand"), "width": np.asarray(rgb.shape[1]),
           "step": np.asarray(3), "robot0_joint_pos": np.zeros(7), "robot0_joint_vel": np.zeros(7),
           "robot0_eef_pos": E[:3, 3], "robot0_eef_quat": q, "robot0_gripper_qpos": GRIP_Q,
           "robot0_gripper_qvel": np.zeros(2), "prev_eef_pos": E[:3, 3], "prev_eef_quat": q, "prev_step": np.asarray(2),
           "obs_sha256": np.asarray("0" * 64), "look_id": np.asarray(vid.split("-")[0]), "look_k": np.asarray(0),
           "steps_since_look_start": np.asarray(0), "code_tree": np.asarray("synthetic"), "clean": np.asarray(False)}
    assert set(arr) == set(POLICY_KEYS_V5)
    arr.update(extra or {})
    p = root / "frames" / scene / f"{vid}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **arr)


def M_quat(R) -> np.ndarray:
    """xyzw quaternion of a rotation matrix (the obs convention)."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([x, y, z, w])


def write_pose_file(root: Path, arm: str, scene: str, look: str, entries: dict, X=X_G0):
    p = root / "poses" / arm / scene / f"{look}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"arm": arm, "look": look, "X": np.asarray(X).tolist(),
           "width_views": {v: {"T": np.asarray(T).tolist(), "K": np.asarray(K).tolist()} for v, (T, K) in entries.items()},
           "code_tree": "synthetic", "clean": False}
    p.write_text(json.dumps(doc))


def make_root(tmp_path: Path, W: int = 512) -> Path:
    """A synthetic root with the G0 hand-scene look as traj1-rs (4 partners) under scene 'sceneA', plus failure
    cases: 'blocked' (reference only), 'nopose' (no pose file), 'partpose' (a partner frame without its pose),
    'corrupt' (an unreadable partner frame), 'pgmdeg' (ride-on-motion partners all under 1 cm), and 'kfile' (a
    policy file that carries K)."""
    root = tmp_path / "root"
    root.mkdir()
    (root / M.SYNTHETIC_MARKER).write_text("synthetic test root\n")
    views = g0_views(W, ("ref",) + G0_LOOKS["g0"], hand=True)
    K = wrist_K(W)

    def put(scene, arm="S3", parts=G0_LOOKS["g0"], pose_parts=None, look="traj1-rs", tokens=None, extra_ref=None):
        toks = tokens or ["rs00"] + [f"rs{i + 1:02d}" for i in range(len(parts))]
        names = ("ref",) + tuple(parts)
        ent = {}
        for tok, nm in zip(toks, names):
            vid = M.view_id(look, tok, W)
            write_policy_frame(root, scene, vid, views[nm]["rgb"], G0_POSES[nm], extra=extra_ref if nm == "ref" else None)
            if pose_parts is None or nm == "ref" or nm in pose_parts:
                ent[vid] = (G0_POSES[nm], K)
        if pose_parts != "none":
            write_pose_file(root, arm, scene, look, ent)

    put("sceneA")
    put("blocked", parts=())
    put("nopose", pose_parts="none")
    put("partpose", pose_parts=("px", "nx", "py"))
    put("corrupt")
    (root / "frames" / "corrupt" / f"{M.view_id('traj1-rs', 'rs02', W)}.npz").write_bytes(b"not an npz")
    G0_POSES.setdefault("near1", rigid(None, (0.004, 0, 0)))
    G0_POSES.setdefault("near2", rigid(None, (0.0, 0.0085, 0)))
    nv = g0_views(W, ("near1", "near2"), hand=True)
    views.update(nv)
    put("pgmdeg", arm="S3-RM", parts=("near1", "near2"), look="traj1-pgm", tokens=["pgr0", "pgm01", "pgm02"])
    put("kfile", extra_ref={"K": K})
    put("s6fail", arm="S6")
    (root / "poses" / "S6" / "s6fail" / "calib.json").write_text(json.dumps({"status": "failed",
                                                                            "reason": "rms 0.9 px > 0.7 px"}))
    put("s6ok", arm="S6")
    (root / "poses" / "S6" / "s6ok" / "calib.json").write_text(json.dumps({"status": "ok", "reason": None}))
    hand = tmp_path / "hand.json"
    hand.write_text(json.dumps(hand_model_doc()))
    return root


def _audit_importable() -> bool:
    try:
        __import__("rowant_lab.libero_env.m3_stereo1_audit")
        return True
    except ImportError:
        return False


def run_cli(root: Path, looks, arm="S3", method="PS", config="PS-w5-loose", W=512, gen="g0", tag="test",
            hand_sha=None, extra=()):
    hand = root.parent / "hand.json"
    sha = hand_sha or M.sha256_bytes(hand.read_bytes())
    args = ["--root", str(root), "--arm", arm, "--method", method, "--config", config, "--width", str(W), "--looks",
            *looks, "--pass-tag", tag, "--gen", gen, "--hand-sha", sha, "--code-tree", "synthetic-tree", "--clean", "0",
            "--hand-model", str(hand), *extra]
    res = subprocess.run([sys.executable, "-B", "-m", "rowant_lab.libero_env.m3_stereo1_match", *args],
                         capture_output=True, text=True, env=_child_env(), cwd=str(root.parent), timeout=900)
    return res


def read_net(root: Path, gen, method, config, arm, scene, vid):
    p = root / "net" / gen / method / config / arm / scene / f"{vid}.npz"
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, json.loads(str(z["meta"]))


needs_audit = pytest.mark.skipif(not _audit_importable(), reason="m3_stereo1_audit has not landed")


@needs_audit
def test_cli_end_to_end_failure_rule_writer_and_audit(tmp_path):
    W = 512
    root = make_root(tmp_path, W)
    looks = [f"{s}/traj1-rs/{W}" for s in ("sceneA", "blocked", "nopose", "partpose", "corrupt", "kfile")]
    res = run_cli(root, looks)
    assert res.returncode == 0, res.stderr[-3000:]
    vid = M.view_id("traj1-rs", "rs00", W)
    arrs, meta = read_net(root, "g0", "PS", "PS-w5-loose", "S3", "sceneA", vid)
    assert meta["status"] == "ok" and meta["n_valid"] > 100000
    assert set(arrs) == {"depth_u16", "valid", "K_used", "meta"}
    assert np.array_equal(arrs["valid"], arrs["depth_u16"] > 0) and not np.any(arrs["depth_u16"] == 65535)
    for k in ("frame_rgb_sha256", "views_used", "b_eff_m", "status", "reason", "timings", "utc", "load_1min",
              "peak_rss_bytes", "cv2_threads", "code_tree", "clean", "pass_tag", "look_index", "wall_s"):
        assert k in meta, k
    assert meta["cv2_threads"] == 8 and meta["code_tree"] == "synthetic-tree" and meta["clean"] is False
    assert meta["pass_tag"] == "test" and meta["gen"] == "g0" and meta["look_index"] == 0
    assert len(meta["views_used"]) == 4
    # the depth agrees with the in-process run of the same look
    d = arrs["depth_u16"].astype(np.float64) * 1e-4
    ref = g0_views(W, ("ref",), hand=True)["ref"]
    v = arrs["valid"] & (ref["pid"] == PID_PLANE)
    assert float(np.median(np.abs(d[v] - ref["depth"][v]))) < 0.001
    want = {"blocked": ("failed", "every view dropped"), "nopose": ("failed", "pose missing"),
            "partpose": ("failed", "pose missing"), "corrupt": ("failed", "exception"),
            "kfile": ("failed", "exception")}
    for scene, (st, why) in want.items():
        a2, m2 = read_net(root, "g0", "PS", "PS-w5-loose", "S3", scene, vid)
        assert m2["status"] == st and m2["reason"].startswith(why), (scene, m2["reason"])
        assert not a2["depth_u16"].any() and not a2["valid"].any()  # the failure rule: an all-zero map
    # ride-on-motion look with no partner in [1, 8] cm: degenerate, all zeros
    res = run_cli(root, [f"pgmdeg/traj1-pgm/{W}"], arm="S3-RM")
    assert res.returncode == 0, res.stderr[-3000:]
    a3, m3 = read_net(root, "g0", "PS", "PS-w5-loose", "S3-RM", "pgmdeg", M.view_id("traj1-pgm", "pgr0", W))
    assert m3["status"] == "degenerate" and not a3["depth_u16"].any()
    # the writer never overwrites: the same generation again is refused before any work
    res = run_cli(root, [f"sceneA/traj1-rs/{W}"])
    assert res.returncode == 4 and "output exists" in res.stdout
    # a re-run computes in memory and writes only a numbers JSON under net/rerun/<pass tag>/
    res = run_cli(root, [f"sceneA/traj1-rs/{W}"], gen="rerun", tag="retime-1")
    assert res.returncode == 0, res.stderr[-3000:]
    rj = root / "net" / "rerun" / "retime-1" / "PS" / "PS-w5-loose" / "S3" / "sceneA" / f"{vid}.json"
    body = json.loads(rj.read_text())
    assert body["depth_u16_sha256"] == M.array_sha256(arrs["depth_u16"])  # deterministic
    assert not list((root / "net" / "rerun").rglob("*.npz"))
    # the hand model's sha is checked by the process itself
    res = run_cli(root, [f"sceneA/traj1-rs/{W}"], gen="g1", hand_sha="0" * 64)
    assert res.returncode == 3 and not (root / "net" / "g1").exists()
    # every audited event of the stereo processes was allowed; module rows were written at install
    rows = [json.loads(x) for f in (root / "audit").glob("stereo-*.jsonl") for x in f.read_text().splitlines()]
    assert rows and not [r for r in rows if r.get("allowed") is False]
    assert any(r.get("event") == "module" for r in rows)
    opened = [r.get("path") or "" for r in rows if r.get("event") == "open"]
    assert opened and not [p for p in opened if "/cam_" + "truth/" in p or "/tr" + "uth/" in p]
    # S6: a failed episode calibration fails the look; an ok one runs
    res = run_cli(root, [f"s6fail/traj1-rs/{W}", f"s6ok/traj1-rs/{W}"], arm="S6")
    assert res.returncode == 0, res.stderr[-3000:]
    a4, m4 = read_net(root, "g0", "PS", "PS-w5-loose", "S6", "s6fail", vid)
    assert m4["status"] == "failed" and m4["reason"].startswith("calibration failed") and not a4["depth_u16"].any()
    a5, m5 = read_net(root, "g0", "PS", "PS-w5-loose", "S6", "s6ok", vid)
    assert m5["status"] == "ok" and np.array_equal(a5["depth_u16"], arrs["depth_u16"])


@needs_audit
def test_cli_outputs_do_not_depend_on_scene_ids(tmp_path):
    W = 512
    root = make_root(tmp_path, W)
    for sub in ("frames", "poses/S3"):
        shutil.copytree(root / sub / "sceneA", root / sub / "zz.renamed-9")
    res = run_cli(root, [f"sceneA/traj1-rs/{W}", f"zz.renamed-9/traj1-rs/{W}"])
    assert res.returncode == 0, res.stderr[-3000:]
    vid = M.view_id("traj1-rs", "rs00", W)
    a1, m1 = read_net(root, "g0", "PS", "PS-w5-loose", "S3", "sceneA", vid)
    a2, m2 = read_net(root, "g0", "PS", "PS-w5-loose", "S3", "zz.renamed-9", vid)
    for k in ("depth_u16", "valid", "K_used"):
        assert np.array_equal(a1[k], a2[k])
    vary = {"look_key", "utc", "timings", "wall_s", "load_1min", "peak_rss_bytes", "look_index", "info"}
    assert {k: v for k, v in m1.items() if k not in vary} == {k: v for k, v in m2.items() if k not in vary}
    i1 = {k: v for k, v in m1["info"].items() if not k.endswith("_s")}
    i2 = {k: v for k, v in m2["info"].items() if not k.endswith("_s")}
    assert i1 == i2


@needs_audit
def test_cli_refuses_unregistered_roots_and_bad_args(tmp_path):
    W = 512
    root = make_root(tmp_path, W)
    (root / M.SYNTHETIC_MARKER).unlink()
    res = run_cli(root, [f"sceneA/traj1-rs/{W}"])
    assert res.returncode == 2 and "refused" in res.stdout
    (root / M.SYNTHETIC_MARKER).write_text("synthetic\n")
    for kw in ({"config": "PS-w9-loose"}, {"W": 1024}, {"arm": "S3/../x"}, {"gen": "a/b"}, {"arm": "S3-b1"}):
        res = run_cli(root, [f"sceneA/traj1-rs/{W}"], **kw)
        assert res.returncode == 2, kw
    assert not (root / "net").exists()


def test_loader_refuses_policy_files_with_K_and_reports_missing_poses(tmp_path):
    W = 512
    root = tmp_path / "r"
    views = g0_views(W, ("ref", "px"), hand=True)
    K = wrist_K(W)
    v0, v1 = M.view_id("traj1-rs", "rs00", W), M.view_id("traj1-rs", "rs01", W)
    write_policy_frame(root, "s", v0, views["ref"]["rgb"], G0_POSES["ref"], extra={"T_world_from_camera": np.eye(4)})
    write_policy_frame(root, "s", v1, views["px"]["rgb"], G0_POSES["px"])
    write_pose_file(root, "S2", "s", "traj1-rs", {v0: (G0_POSES["ref"], K), v1: (G0_POSES["px"], K)})
    with pytest.raises(ValueError):
        M.load_look(root, "S2", "s", "traj1-rs", W)
    write_policy_frame(root, "t", v0, views["ref"]["rgb"], G0_POSES["ref"])
    write_policy_frame(root, "t", v1, views["px"]["rgb"], G0_POSES["px"])
    write_pose_file(root, "S2", "t", "traj1-rs", {v0: (G0_POSES["ref"], K)})
    with pytest.raises(M.LookFailure) as e:
        M.load_look(root, "S2", "t", "traj1-rs", W)
    assert e.value.status == "failed" and "pose missing" in e.value.reason
    write_pose_file(root, "S2", "t", "traj1-rs", {v0: (G0_POSES["ref"], K), v1: (G0_POSES["px"], K)})
    lo = M.load_look(root, "S2", "t", "traj1-rs", W)
    assert [p["token"] for p in lo["partners"]] == ["rs01"] and np.allclose(lo["X"], X_G0)
    assert np.array_equal(lo["ref"]["gripper_qpos"], GRIP_Q)
    # the photometric arms perturb partners only
    write_pose_file(root, "P1", "t", "traj1-rs", {v0: (G0_POSES["ref"], K), v1: (G0_POSES["px"], K)})
    lp = M.load_look(root, "P1", "t", "traj1-rs", W)
    assert np.array_equal(lp["ref"]["rgb"], lo["ref"]["rgb"])
    assert not np.array_equal(lp["partners"][0]["rgb"], lo["partners"][0]["rgb"])


# ======================================================================================================= AST rules

STDLIB = set(sys.stdlib_module_names)
ALLOWED_THIRD = {"numpy", "cv2", "scipy"}
ALLOWED_REPO = {"rowant_lab.libero_env.m2_camera", "rowant_lab.libero_env.m3_stereo1_audit",
                "rowant_lab.libero_env.m3_stereo1_sgbm"}
BANNED_MODULES = {"subprocess", "socket", "multiprocessing", "urllib", "http", "ftplib", "smtplib", "asyncio",
                  "concurrent", "pty", "ctypes"}
BANNED_ATTRS = {("os", "system"), ("os", "popen"), ("os", "fork"), ("os", "forkpty"), ("os", "posix_spawn"),
                ("os", "posix_spawnp"), ("cv2", "imread"), ("cv2", "imwrite"), ("cv2", "imreadmulti"),
                ("cv2", "FileStorage"), ("cv2", "VideoCapture"), ("cv2", "VideoWriter"), ("np", "memmap"),
                ("numpy", "memmap")}
BANNED_STRINGS = ("truth", "workcell", "frozen-config")


def _imports(src: str) -> list:
    out = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            assert n.level == 0, "relative import"
            out += [f"{n.module}.{a.name}" if n.module == "rowant_lab.libero_env" else n.module for a in n.names]
    return out


@pytest.mark.parametrize("name,src", [("match", MATCH_SRC)])
def test_ast_import_allowlist(name, src):
    allowed_repo = ALLOWED_REPO - ({"rowant_lab.libero_env.m3_stereo1_sgbm"} if name == "sgbm" else set())
    for mod in _imports(src):
        top = mod.split(".")[0]
        if mod.startswith("rowant_lab"):
            assert mod in allowed_repo, mod
        else:
            assert top in STDLIB or top in ALLOWED_THIRD, mod
            assert top not in BANNED_MODULES, mod
    if name == "sgbm":
        assert not [m for m in _imports(src) if m.startswith("rowant_lab")]


@pytest.mark.parametrize("name,src", [("match", MATCH_SRC)])
def test_ast_no_c_level_io_no_spawn_no_banned_constants(name, src):
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            assert (n.value.id, n.attr) not in BANNED_ATTRS, (n.value.id, n.attr)
            assert not (n.value.id == "os" and (n.attr.startswith("exec") or n.attr.startswith("spawn"))), n.attr
        if isinstance(n, ast.Name):
            assert n.id not in ("memmap", "imread", "imwrite", "FileStorage", "system"), n.id
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            low = n.value.lower()
            assert not any(b in low for b in BANNED_STRINGS), n.value[:80]
        if isinstance(n, ast.ImportFrom) and n.module in ("os", "cv2", "numpy"):
            assert not {a.name for a in n.names} & {"system", "imread", "imwrite", "FileStorage", "memmap", "popen"}


def test_ast_hook_installed_in_main_only_and_threads_set_there():
    tree = ast.parse(MATCH_SRC)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    audit_users = []
    for fname, fn in fns.items():
        for n in ast.walk(fn):
            if isinstance(n, ast.ImportFrom) and any(a.name == "m3_stereo1_audit" for a in n.names):
                audit_users.append(fname)
    assert audit_users == ["main"]
    top = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    for n in top:
        for c in ast.walk(n):
            if isinstance(c, ast.Call):
                f = c.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                assert name not in ("addaudithook", "install_hook", "install", "_install_hook", "setNumThreads",
                                    "main") or isinstance(n, ast.If), name
    calls = [c for c in ast.walk(fns["main"]) if isinstance(c, ast.Call)]
    names = [c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", "") for c in calls]
    assert "_install_hook" in names and "setNumThreads" in names
    assert names.index("_install_hook") < names.index("setNumThreads")
    # the SG module is imported only inside the SG branch, never at the top
    assert "m3_stereo1_sgbm" not in {a.name for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names}


def test_importing_match_installs_no_hook_and_loads_no_audit_module():
    code = ("import sys; import rowant_lab.libero_env.m3_stereo1_match as M; "
            "print(int('rowant_lab.libero_env.m3_stereo1_audit' in sys.modules), "
            "int('rowant_lab.libero_env.m3_stereo1_sgbm' in sys.modules), int('subprocess' in sys.modules))")
    res = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=_child_env(),
                         timeout=120)
    assert res.returncode == 0, res.stderr
    a, s, _sp = res.stdout.split()
    assert a == "0" and s == "0"


# ================================================================================================ integrator fixes
def test_rm_look_belongs_to_the_rm_arm_only():
    """traj1-pg and traj1-pgm share the pgr0 reference view, and the net layout keys a map by its reference view."""
    assert M.rm_look_refusal("S3-RM", "traj1-pgm") is None and M.rm_look_refusal("S3", "traj1-pg") is None
    assert M.rm_look_refusal("S3", "traj1-pgm") and M.rm_look_refusal("S2", "traj1-pgm")
    assert M.rm_look_refusal("S3-RM", "traj1-pg") and M.rm_look_refusal("S3-RM", "traj1-rs")
    assert M.rm_look_refusal("S3", "traj1-pp") is None and M.rm_look_refusal("S6", "traj2-E2") is None


@needs_audit
def test_cli_refuses_colliding_outputs_before_any_work(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / M.SYNTHETIC_MARKER).write_text("synthetic test root\n")
    (tmp_path / "hand.json").write_text(json.dumps(hand_model_doc()))
    res = run_cli(root, ["sceneA/traj1-pg/512", "sceneA/traj1-pgm/512"])
    assert res.returncode == 2 and "traj1-pgm" in res.stdout
    res = run_cli(root, ["sceneA/traj1-pg/512", "sceneA/traj1-pg/512"])
    assert res.returncode == 2 and "same output" in res.stdout
    assert not (root / "audit").exists() and not (root / "net").exists()  # refused before the hook and any write


@needs_audit
def test_a_hook_violation_stops_the_process_and_is_never_a_failed_look(tmp_path):
    """An audit-hook violation while a look loads (here a pose file that is a symlink to a file outside the
    allowlist) ends the stereo process with exit 70: the look is never written as a failed map, and no later look of
    the process is computed. The meta of every written map carries the process id (the launcher's pid)."""
    W = 512
    root = make_root(tmp_path, W)
    res = run_cli(root, [f"sceneA/traj1-rs/{W}"], gen="g1")
    assert res.returncode == 0, res.stderr[-3000:]
    _arrs, meta = read_net(root, "g1", "PS", "PS-w5-loose", "S3", "sceneA", M.view_id("traj1-rs", "rs00", W))
    assert isinstance(meta["pid"], int) and meta["pid"] > 1
    outside = tmp_path / "outside.json"
    outside.write_text((root / "poses" / "S3" / "sceneA" / "traj1-rs.json").read_text())
    (root / "poses" / "S3" / "leak").mkdir()
    (root / "poses" / "S3" / "leak" / "traj1-rs.json").symlink_to(outside)
    shutil.copytree(root / "frames" / "sceneA", root / "frames" / "leak")
    res = run_cli(root, [f"leak/traj1-rs/{W}", f"sceneA/traj1-rs/{W}"], gen="g2")
    assert res.returncode == 70, (res.returncode, res.stdout[-1500:], res.stderr[-1500:])
    assert "AuditViolation" in res.stderr
    assert not (root / "net" / "g2").exists() or not list((root / "net" / "g2").rglob("*.npz"))
    rows = [json.loads(x) for f in (root / "audit").glob("stereo-*.jsonl") for x in f.read_text().splitlines()]
    assert [r for r in rows if r.get("allowed") is False and os.path.realpath(outside) == r.get("path")]
