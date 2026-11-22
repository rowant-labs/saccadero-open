"""M3 K0-stereo: classical wrist motion stereo, the stereo CLI and its pure functions (Monty venv, CPU, own process).

Registered design: the M3 dev targets' key ``k0_stereo`` (``methods.*``) and the K0-stereo design note,
section 5; where they differ the block binds.

What this module is
-------------------
* ``main()``: the stereo process. Started only by ``m3_stereo1_eval.py launch`` (not in this release) with
  ``--root R --arm A --method {PS,SG} --config C --width W --looks <scene>/<look>/<width> ... --pass-tag P --gen G
  --hand-sha H --code-tree T --clean {0,1}``. It installs the strict audit hook (``m3_stereo1_audit``) first, sets
  ``cv2.setNumThreads(8)``, checks the hand model's sha256 itself, then for each look reads only
  ``R/frames/<scene>/<view>.npz`` (policy files), ``R/poses/<arm>/<scene>/<look>.json`` (and, for S6,
  ``R/poses/S6/<scene>/calib.json``) and the registered hand model file, and writes only
  ``R/net/<gen>/<method>/<config>/<arm>/<scene>/<view>.npz`` (O_EXCL, never over an existing file) or, for
  ``--gen rerun``, a numbers JSON under ``R/net/rerun/<pass tag>/``. Scene ids are opaque: they are path parts only.
* Pure functions the eval imports without ``main`` or the hook (the evaluator-side warp and hand-raster checks):
  ``load_look`` (the loader), ``select_views`` / ``b_eff`` / ``rm_partners`` (view selection), ``PartnerGeom`` and
  ``warp_to_reference`` (the warp at a given per-pixel depth), ``RefWindow.zncc`` and ``texture_map`` (the window
  statistics, ``TAU_TEX``), ``hand_raster`` (the hand-and-finger raster) and ``rigid_or_flat_mask``.

Geometry conventions
--------------------
Poses are ``T_world_from_camera`` in OpenCV axes (x right, y down, z forward) and ``K`` with the principal point in
the pixel-centre convention of the pose files (MuJoCo's: pixel ``u`` spans ``[u, u + 1)``, its centre ray is
``x = (u + 0.5 - K[0, 2]) / K[0, 0]``). OpenCV routines take index coordinates, so they get ``K`` with the principal
point moved by -0.5 (``K_index``). Output depth is z along the reference optical axis on the image grid.

Plane sweep PS (methods.plane_sweep_PS)
---------------------------------------
Fronto-parallel planes of the reference camera, uniform in inverse depth over z in [0.08, 1.20] m. Coarse pass at
half resolution (2x2 area mean; K halved) in 1 px disparity steps of the widest used view, window w x w
half-resolution pixels; full-resolution refine over +-2 px of the widest view in 0.25 px steps around the coarse
inverse depth (17 samples, per-pixel depth, the same warp as the coarse pass). "Disparity" is the image
displacement per unit inverse depth, maximised over the reference pixels a view sees (``PartnerGeom.rate``): f b
for a sideways move, larger for a move along the optical axis (a descent), so no used view moves more than 1 px
between coarse planes. INTER_CUBIC warps; cost ``1 - ZNCC`` over w x w; at each pixel the mean of the best ``ceil(n / 2)`` usable
views (n counted per pixel); parabola sub-sample fit on the best sample and its two neighbours. Streaming: per pixel
only the best cost, its index and its two neighbours (best-3), and the two best interior local minima of the coarse
curve (for the margin) are kept, so memory is O(HW). A minimum on the first or last sample is no reading.
Parallax rule: at refine a partner is used at a pixel only if moving the pixel's coarse depth by +5 % and by -5 %
each shifts its projection in that partner by at least 1 px; a pixel with no such partner has no reading.
Confidence gate (grid): best ZNCC (1 - the refined best cost) and margin (coarse second-best interior local minimum
minus coarse best, both in cost units) at least (0.5, 0.02) loose or (0.7, 0.05) strict.

Common parts (methods.common)
-----------------------------
Luma (BT.601 weights, float32). Texture gate: the w x w mean of the 3x3 Sobel gradient magnitude divided by 8
(grey levels per pixel) at least ``TAU_TEX`` (fixed in G0). Masks: the hand-and-finger raster from the hand model
at the reference view's obs gripper qpos through the pose file's X (hand-from-camera) and K, dilated 3 px at 512 and
6 px at 1024; the rigid-or-flat mask (zero-displacement 5x5 mean absolute luma difference <= 2.0 grey levels against
every used partner with b_eff >= 1 cm, or against every used partner when none has b_eff >= 1 cm). View selection:
b_eff = camera-centre displacement perpendicular to the reference principal ray, from the arm's own poses; views with
b_eff < 0.5 cm are dropped; ride-on-motion looks take up to 3 partners greedily by largest b_eff in [1, 8] cm, each
at least 1 cm (camera centres) from those already chosen, else the look is degenerate. Flying pixels: invalid where
the 3x3 range of valid depth exceeds 0.03 z. No smoothing, no hole filling, no fallback depth. Depth is encoded by
``m2_camera.encode_depth`` with a zero zbuf and invalid pixels as NaN (0.1 mm units, 0 = no reading; 65535 is
asserted absent).

Failure rule (methods.failure_rule): a look fails (all-zero map, status ``failed`` with its reason) when a pose is
missing, every view is dropped, every SG pair is skipped, an exception occurs, or (S6) the episode's calibration
failed; a ride-on-motion look with no qualifying partner is ``degenerate`` (all-zero map). Nothing is replaced by
other depth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from rowant_lab.libero_env import m2_camera

MODULE = "rowant_lab.libero_env.m3_stereo1_match"
NET_FORMAT = "rowant m3 stereo1 net v1"
REPO = Path(__file__).resolve().parents[2]
HAND_MODEL_REL = "rowant_lab/libero_env/evidence-m3/stereo1/panda_hand_model.json"
HAND_MODEL_FORMAT = "rowant panda hand model v1"
SYNTHETIC_MARKER = "SYNTHETIC"
CAMERA = "robot0_eye_in_hand"
CV2_THREADS = 8

# ---------------------------------------------------------------------------------------- registered constants
Z_MIN_M, Z_MAX_M = 0.08, 1.20
B_DROP_M = 0.005  # views with b_eff < 0.5 cm are dropped
B_RIGID_M = 0.01  # rigid-or-flat partners: b_eff >= 1 cm (else every used partner)
RM_B_MIN_M, RM_B_MAX_M, RM_SEP_M, RM_MAX = 0.01, 0.08, 0.01, 3
RIGID_WIN = 5
RIGID_MAD_MAX = 2.0
FLY_REL = 0.03
PARALLAX_REL = 0.05
PARALLAX_PX = 1.0
REFINE_HALF_PX = 2.0
REFINE_STEP_PX = 0.25
HAND_DILATE_PX_AT_512 = 3
# Texture gate threshold: the window mean of |Sobel 3x3| / 8, grey levels per pixel. Fixed in G0 (synthetic only,
# tests/test_m3_stereo1_match.py) before the stereo runs on any LIBERO frame; recorded in K0S-C2.
TAU_TEX = 0.5
PS_W = {512: (5, 7), 1024: (9, 13)}
PS_GATES = {"loose": (0.5, 0.02), "strict": (0.7, 0.05)}
SG_BS = {512: (5, 7), 1024: (7, 11)}
SG_UR = (5, 12)
# G0 reference configs (declared): their windows are the warp check's window.
REFERENCE_CONFIG = {"PS": {512: "PS-w5-loose", 1024: "PS-w9-loose"},
                    "SG": {512: "SG-b5-u12", 1024: "SG-b7-u12"}}
PHOTOMETRIC = {"P1": 0.6, "P2": 1.4}  # partner-view RGB gain of the descriptive photometric arms
PHOTO_SIGMA = 3.0
PHOTO_SEED = 7503
ZNCC_STD_MIN = 0.05  # grey levels: a window with less spread has no ZNCC
MAX_PLANES = 1024  # coarse planes per look at most (flagged in the meta when reached)

# Looks (the capture contract): reference token, partner tokens, partner selection.
LOOKS = {
    "traj1-rs": {"ref": "rs00", "partners": tuple(f"rs{i:02d}" for i in range(1, 11)), "select": "all"},
    "traj1-pg": {"ref": "pgr0", "partners": ("pgp1", "pgp2", "pgp3"), "select": "all"},
    "traj1-pgm": {"ref": "pgr0", "partners": tuple(f"pgm{i:02d}" for i in range(1, 13)), "select": "rm"},
    "traj1-pp": {"ref": "ppr0", "partners": tuple(f"ppm{i:02d}" for i in range(1, 13)), "select": "rm"},
    "traj2-E2": {"ref": "C", "partners": ("R", "L"), "select": "all"},
}
LOOK_LENGTH = {"b1": ("pgp1",), "b2": ("pgp1", "pgp2")}  # arm suffix -> partners (traj1-pg only)
TOKEN_RE = re.compile(r"^[A-Za-z0-9_.]+(-[A-Za-z0-9_.]+)*$")


class LookFailure(Exception):
    """A look that fails by the failure rule (status, reason)."""

    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


# ------------------------------------------------------------------------------------------------ small helpers

def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)  # bytes on macOS, KiB on Linux


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def array_sha256(a: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(a).tobytes())


def parse_look_key(key: str) -> tuple:
    """``<scene>/<look>/<width>`` -> (scene, look, width); the scene id is kept opaque (never parsed)."""
    parts = str(key).split("/")
    if len(parts) != 3 or not parts[0] or parts[0] in (".", "..") or not TOKEN_RE.match(parts[0]):
        raise ValueError(f"bad look key {key!r}")
    if parts[1] not in LOOKS:
        raise ValueError(f"unknown look {parts[1]!r}")
    return parts[0], parts[1], int(parts[2])


def view_id(look: str, token: str, width: int, camera: str = CAMERA) -> str:
    return f"{look.split('-')[0]}-{token}-{camera}-{int(width)}"


def look_spec(look: str, arm: str) -> dict:
    """The look's reference token, partner tokens and selection, with the b1 / b2 look-length arms applied."""
    spec = dict(LOOKS[look])
    suffix = str(arm).rsplit("-", 1)[-1]
    if suffix in LOOK_LENGTH:
        if look != "traj1-pg":
            raise ValueError(f"arm {arm!r} is a look-length arm of traj1-pg, not {look!r}")
        spec["partners"] = LOOK_LENGTH[suffix]
    return spec


def rm_look_refusal(arm: str, look: str) -> str | None:
    """The net layout keys a map by its reference view: traj1-pg and traj1-pgm share pgr0, so the ride-on-motion
    pre-grasp look belongs to the RM arm only (arms.S3_RM; the eval reads an RM arm's pgr0 map as traj1-pgm) and an RM
    arm runs that look only. Returns the refusal reason, or None."""
    rm = str(arm).endswith("-RM")
    if look == "traj1-pgm" and not rm:
        return f"traj1-pgm is the ride-on-motion arm's look (S3-RM), not {arm!r}'s (its pgr0 map is traj1-pg)"
    if rm and look != "traj1-pgm":
        return f"arm {arm!r} runs traj1-pgm only"
    return None


def parse_config(method: str, config: str, width: int) -> dict:
    """A grid config id (methods.grid_per_width) -> its parameters; refuses ids off the width's grid."""
    W = int(width)
    if method == "PS":
        m = re.fullmatch(r"PS-w(\d+)-(loose|strict)", config)
        if not m or W not in PS_W or int(m.group(1)) not in PS_W[W]:
            raise ValueError(f"PS config {config!r} is not on the {W} px grid")
        zmin, mmin = PS_GATES[m.group(2)]
        return {"method": "PS", "w": int(m.group(1)), "gate": m.group(2), "zncc_min": zmin, "margin_min": mmin}
    if method == "SG":
        m = re.fullmatch(r"SG-b(\d+)-u(\d+)", config)
        if not m or W not in SG_BS or int(m.group(1)) not in SG_BS[W] or int(m.group(2)) not in SG_UR:
            raise ValueError(f"SG config {config!r} is not on the {W} px grid")
        return {"method": "SG", "bs": int(m.group(1)), "ur": int(m.group(2))}
    raise ValueError(f"method {method!r}")


def K_index(K) -> np.ndarray:
    """K in index coordinates (OpenCV's: integer pixel = pixel centre) from the pixel-centre-convention K."""
    Ki = np.array(K, dtype=np.float64).reshape(3, 3).copy()
    Ki[0, 2] -= 0.5
    Ki[1, 2] -= 0.5
    return Ki


def luma(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma as float32 grey levels (no rounding)."""
    a = np.asarray(rgb, dtype=np.float32)
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)


def luma_u8(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)


def photometric(rgb: np.ndarray, gain: float) -> np.ndarray:
    """The photometric arms' partner perturbation: RGB x gain plus Gaussian noise sigma 3 grey levels per channel,
    seeded (7503) by the gain and the view's own RGB bytes (independent per view, opaque to scene ids)."""
    digest = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).digest()
    rng = np.random.default_rng([PHOTO_SEED, int(round(gain * 1000)), int.from_bytes(digest[:8], "little")])
    out = np.asarray(rgb, dtype=np.float64) * float(gain) + rng.normal(0.0, PHOTO_SIGMA, size=np.shape(rgb))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------------ loader

def _read_policy_view(path: Path) -> dict:
    """The fields of one policy file the stereo uses (rgb, gripper qpos, width, camera); a file that carries K or T
    is not a v5 policy file and is refused."""
    with np.load(path, allow_pickle=False) as z:
        names = set(z.files)
        if "K" in names or "T_world_from_camera" in names:
            raise ValueError(f"{path.name}: a policy file carries no K and no T")
        rgb = np.ascontiguousarray(z["rgb"], dtype=np.uint8)
        q = np.asarray(z["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)
        width = int(z["width"]) if "width" in names else int(rgb.shape[1])
        camera = str(z["camera"]) if "camera" in names else None
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] != rgb.shape[1] or rgb.shape[1] != width:
        raise ValueError(f"{path.name}: rgb {rgb.shape} at width {width}")
    return {"rgb": rgb, "gripper_qpos": q, "width": width, "camera": camera}


def _pose_entry(pose: dict, vid: str):
    e = (pose.get("width_views") or {}).get(vid)
    if e is None:
        return None
    T = np.asarray(e["T"], dtype=np.float64).reshape(4, 4)
    K = np.asarray(e["K"], dtype=np.float64).reshape(3, 3)
    if not (np.all(np.isfinite(T)) and np.all(np.isfinite(K))):
        raise ValueError(f"pose of {vid} is not finite")
    return T, K


def load_look(root, arm: str, scene: str, look: str, width: int, camera: str = CAMERA) -> dict:
    """Policy frames and the arm's poses of one look (the stereo's own loader).

    Reads ``R/poses/<arm>/<scene>/<look>.json`` and ``R/frames/<scene>/<view>.npz`` for the reference and every
    registered partner token whose frame exists (a blocked look has none). Raises ``LookFailure('failed', ...)``
    when the pose file or a pose of an existing frame is missing. The photometric arms' partners are perturbed here.
    Returns {ref, partners, X, pose_meta}; each view is {view_id, token, rgb, gripper_qpos, T, K}."""
    root = Path(root)
    spec = look_spec(look, arm)
    ppath = root / "poses" / arm / scene / f"{look}.json"
    if not ppath.is_file():
        raise LookFailure("failed", "pose missing: no pose file")
    with open(ppath, "r", encoding="utf-8") as f:
        pose = json.load(f)
    if pose.get("arm") not in (None, arm) or pose.get("look") not in (None, look):
        raise ValueError(f"pose file arm/look {pose.get('arm')!r}/{pose.get('look')!r} != {arm!r}/{look!r}")
    X = np.asarray(pose["X"], dtype=np.float64).reshape(4, 4)
    fdir = root / "frames" / scene

    def one(token: str):
        vid = view_id(look, token, width, camera)
        fp = fdir / f"{vid}.npz"
        if not fp.is_file():
            return None
        ent = _pose_entry(pose, vid)
        if ent is None:
            raise LookFailure("failed", f"pose missing: {token}")
        fr = _read_policy_view(fp)
        if fr["width"] != int(width):
            raise ValueError(f"{vid}: width {fr['width']}")
        return {"view_id": vid, "token": token, "rgb": fr["rgb"], "gripper_qpos": fr["gripper_qpos"],
                "T": ent[0], "K": ent[1]}

    ref = one(spec["ref"])
    if ref is None:
        raise LookFailure("failed", f"reference frame missing: {spec['ref']}")
    partners = [p for p in (one(t) for t in spec["partners"]) if p is not None]
    if arm in PHOTOMETRIC:
        for p in partners:
            p["rgb"] = photometric(p["rgb"], PHOTOMETRIC[arm])
    return {"ref": ref, "partners": partners, "X": X, "select": spec["select"], "look": look, "width": int(width),
            "pose_meta": {"code_tree": pose.get("code_tree"), "clean": pose.get("clean")}}


# ---------------------------------------------------------------------------------------------- view selection

def camera_centre(T) -> np.ndarray:
    return np.asarray(T, dtype=np.float64)[:3, 3]


def b_eff(T_ref, T_p) -> float:
    """Camera-centre displacement perpendicular to the reference principal ray (metres)."""
    T_ref = np.asarray(T_ref, dtype=np.float64)
    d = camera_centre(T_p) - camera_centre(T_ref)
    z = T_ref[:3, 2] / np.linalg.norm(T_ref[:3, 2])
    return float(np.linalg.norm(d - float(d @ z) * z))


def rm_partners(T_ref, Ts: list) -> list:
    """Ride-on-motion partners: indices of up to 3 views, greedily by largest b_eff in [1, 8] cm, each at least 1 cm
    (camera centres) from those already chosen; [] = degenerate."""
    cand = sorted(((b_eff(T_ref, T), i) for i, T in enumerate(Ts)), key=lambda x: (-x[0], x[1]))
    out = []
    for b, i in cand:
        if not (RM_B_MIN_M <= b <= RM_B_MAX_M):
            continue
        c = camera_centre(Ts[i])
        if all(float(np.linalg.norm(c - camera_centre(Ts[j]))) >= RM_SEP_M for j in out):
            out.append(i)
        if len(out) == RM_MAX:
            break
    return sorted(out)


def select_views(look_data: dict) -> dict:
    """{used: [partner dicts], b_eff: {view_id: m}, dropped: [view_id], status, reason} (status None = ok)."""
    ref = look_data["ref"]
    parts = look_data["partners"]
    be = {p["view_id"]: b_eff(ref["T"], p["T"]) for p in parts}
    if look_data.get("select") == "rm":
        idx = rm_partners(ref["T"], [p["T"] for p in parts])
        used = [parts[i] for i in idx]
        dropped = [p["view_id"] for i, p in enumerate(parts) if i not in idx]
        if not parts:
            return {"used": [], "b_eff": be, "dropped": dropped, "status": "failed", "reason": "every view dropped"}
        if not used:
            return {"used": [], "b_eff": be, "dropped": dropped, "status": "degenerate",
                    "reason": "no ride-on-motion partner with b_eff in [1, 8] cm"}
        return {"used": used, "b_eff": be, "dropped": dropped, "status": None, "reason": None}
    used = [p for p in parts if be[p["view_id"]] >= B_DROP_M]
    dropped = [p["view_id"] for p in parts if be[p["view_id"]] < B_DROP_M]
    if not used:
        return {"used": [], "b_eff": be, "dropped": dropped, "status": "failed", "reason": "every view dropped"}
    return {"used": used, "b_eff": be, "dropped": dropped, "status": None, "reason": None}


# ---------------------------------------------------------------------------------------------- window maths

def box(x: np.ndarray, w: int) -> np.ndarray:
    return cv2.boxFilter(np.asarray(x, dtype=np.float32), cv2.CV_32F, (int(w), int(w)), normalize=True,
                         borderType=cv2.BORDER_REFLECT_101)


def texture_map(lum: np.ndarray, w: int) -> np.ndarray:
    """Window mean (w x w) of the 3x3 Sobel gradient magnitude / 8 (grey levels per pixel)."""
    L = np.asarray(lum, dtype=np.float32)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return box(np.sqrt(gx * gx + gy * gy), w)


class RefWindow:
    """The reference's w x w window statistics, for ZNCC against warped partners."""

    def __init__(self, lum: np.ndarray, w: int):
        self.w = int(w)
        self.I = np.asarray(lum, dtype=np.float32) - np.float32(128.0)
        self.mI = box(self.I, self.w)
        v = box(self.I * self.I, self.w) - self.mI * self.mI
        self.sI = np.sqrt(np.maximum(v, 0.0)).astype(np.float32)

    def zncc(self, J: np.ndarray) -> np.ndarray:
        """ZNCC over the window (NaN where either window has under ``ZNCC_STD_MIN`` spread)."""
        J = np.asarray(J, dtype=np.float32) - np.float32(128.0)
        w = self.w
        mJ = box(J, w)
        vJ = box(J * J, w) - mJ * mJ
        cov = box(self.I * J, w) - self.mI * mJ
        sJ = np.sqrt(np.maximum(vJ, 0.0))
        den = self.sI * sJ
        ok = (self.sI > ZNCC_STD_MIN) & (sJ > ZNCC_STD_MIN)
        out = np.full(J.shape, np.nan, dtype=np.float32)
        np.divide(cov, den, out=out, where=ok)
        return np.clip(out, -1.0, 1.0)


def best_half_mean(costs: list) -> np.ndarray:
    """Per pixel: the mean of the best ceil(n / 2) finite costs, n = the number of finite costs (inf if none)."""
    if len(costs) == 1:
        return costs[0]
    C = np.stack(costs, axis=0)
    fin = np.isfinite(C)
    n = fin.sum(axis=0)
    if len(costs) == 2:
        both = np.minimum(C[0], C[1])
        return both.astype(np.float32)
    S = np.sort(C, axis=0)
    k = np.maximum((n + 1) // 2, 1)
    cs = np.cumsum(np.where(np.isfinite(S), S, 0.0), axis=0, dtype=np.float32)
    agg = np.take_along_axis(cs, (k - 1)[None].astype(np.intp), axis=0)[0] / k.astype(np.float32)
    agg[n == 0] = np.inf
    return agg.astype(np.float32)


def parabola_offset(cl, c0, cr) -> np.ndarray:
    """Sub-sample offset in [-0.5, 0.5] of the parabola through (-1, cl), (0, c0), (1, cr)."""
    cl = np.asarray(cl, dtype=np.float64)
    c0 = np.asarray(c0, dtype=np.float64)
    cr = np.asarray(cr, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        den = cl - 2.0 * c0 + cr
        num = 0.5 * (cl - cr)
    off = np.zeros(np.broadcast(cl, c0, cr).shape)
    ok = np.isfinite(den) & np.isfinite(num) & (den > 1e-12)
    np.divide(num, den, out=off, where=ok)
    return np.clip(off, -0.5, 0.5)


class MinTracker:
    """Streaming over samples 0..n-1 (O(HW)): the best cost with its index and two neighbours (best-3) and the two
    best interior local minima (cost, index), for the margin."""

    def __init__(self, shape):
        inf = np.float32(np.inf)
        self.k = 0
        self.prev2 = np.full(shape, inf, np.float32)
        self.prev = np.full(shape, inf, np.float32)
        self.cb = np.full(shape, inf, np.float32)
        self.kb = np.full(shape, -1, np.int32)
        self.lb = np.full(shape, inf, np.float32)
        self.rb = np.full(shape, inf, np.float32)
        self.c1 = np.full(shape, inf, np.float32)
        self.k1 = np.full(shape, -10, np.int32)
        self.c2 = np.full(shape, inf, np.float32)
        self.k2 = np.full(shape, -10, np.int32)

    def push(self, c: np.ndarray) -> None:
        c = np.asarray(c, dtype=np.float32)
        k = self.k
        # the right neighbour of a best found at the previous sample
        np.copyto(self.rb, c, where=(self.kb == k - 1))
        if k >= 2:  # is sample k-1 an interior local minimum?
            p = self.prev
            m = np.isfinite(self.prev2) & np.isfinite(c) & np.isfinite(p) & (p < self.prev2) & (p <= c)
            better = m & (p < self.c1)
            second = m & ~better & (p < self.c2)
            np.copyto(self.c2, self.c1, where=better)
            np.copyto(self.k2, self.k1, where=better)
            np.copyto(self.c2, p, where=second)
            np.copyto(self.k2, np.int32(k - 1), where=second)
            np.copyto(self.c1, p, where=better)
            np.copyto(self.k1, np.int32(k - 1), where=better)
        better = c < self.cb
        np.copyto(self.cb, c, where=better)
        np.copyto(self.kb, np.int32(k), where=better)
        np.copyto(self.lb, self.prev, where=better)
        np.copyto(self.rb, np.float32(np.inf), where=better)
        self.prev2, self.prev = self.prev, c
        self.k = k + 1

    def result(self) -> dict:
        n = self.k
        interior = (self.kb > 0) & (self.kb < n - 1) & np.isfinite(self.cb) & np.isfinite(self.lb) & np.isfinite(
            self.rb)
        c_other = np.where(np.abs(self.k1 - self.kb) >= 2, self.c1, self.c2)
        with np.errstate(invalid="ignore"):
            margin = np.where(np.isfinite(c_other) & np.isfinite(self.cb), c_other - self.cb,
                              np.float32(np.inf)).astype(np.float32)
        return {"ok": interior, "k": self.kb, "c": self.cb, "off": parabola_offset(self.lb, self.cb, self.rb),
                "margin": margin}


# ----------------------------------------------------------------------------------------------------- warping

class PartnerGeom:
    """Per-pixel projection of the reference pixel-centre rays into one partner at inverse depth rho.

    With M = T_p^-1 T_ref = [R | t] and the reference ray r (z = 1), a point at depth z is z r and lands in the
    partner at z (R r + t rho): so ``u = fx (a_x + t_x rho) / (a_z + t_z rho) + cx - 0.5`` (index coordinates)
    with a = R r, and it lies in front of the partner where a_z + t_z rho > 0."""

    def __init__(self, K_ref, T_ref, K_p, T_p, shape):
        H, W = shape
        K_ref = np.asarray(K_ref, dtype=np.float64)
        K_p = np.asarray(K_p, dtype=np.float64)
        M = np.linalg.inv(np.asarray(T_p, dtype=np.float64)) @ np.asarray(T_ref, dtype=np.float64)
        R, t = M[:3, :3], M[:3, 3]
        rx = (np.arange(W, dtype=np.float64) + 0.5 - K_ref[0, 2]) / K_ref[0, 0]
        ry = (np.arange(H, dtype=np.float64) + 0.5 - K_ref[1, 2]) / K_ref[1, 1]
        RX, RY = np.meshgrid(rx, ry)
        self.ax = (R[0, 0] * RX + R[0, 1] * RY + R[0, 2]).astype(np.float32)
        self.ay = (R[1, 0] * RX + R[1, 1] * RY + R[1, 2]).astype(np.float32)
        self.az = (R[2, 0] * RX + R[2, 1] * RY + R[2, 2]).astype(np.float32)
        self.t = [np.float32(v) for v in t]
        self.fx, self.fy = np.float32(K_p[0, 0]), np.float32(K_p[1, 1])
        self.cx, self.cy = np.float32(K_p[0, 2] - 0.5), np.float32(K_p[1, 2] - 0.5)
        self.shape = (H, W)

    def rate(self, rho: float, W_p: int, H_p: int, step: int = 1) -> float:
        """The largest image displacement per unit inverse depth (px m) over reference pixels whose projection at
        ``rho`` lies inside the partner: |d(u, v)/d rho| = f |t_xy a_z - a_xy t_z| / (a_z + t_z rho)^2."""
        ax, ay, az = self.ax[::step, ::step], self.ay[::step, ::step], self.az[::step, ::step]
        den = az + self.t[2] * np.float32(rho)
        front = den > np.float32(1e-6)
        if not front.any():
            return 0.0
        inv = np.where(front, 1.0 / np.where(front, den, 1.0), 0.0)
        mx = (ax + self.t[0] * np.float32(rho)) * inv * self.fx + self.cx
        my = (ay + self.t[1] * np.float32(rho)) * inv * self.fy + self.cy
        # in the partner's image and at least 0.08 m in front of it (z_p = den / rho)
        inside = front & (den >= np.float32(Z_MIN_M * rho)) & (mx >= 0) & (mx <= W_p - 1) & (my >= 0) & (my <= H_p - 1)
        if not inside.any():
            return 0.0
        du = self.fx * (self.t[0] * az - ax * self.t[2]) * inv * inv
        dv = self.fy * (self.t[1] * az - ay * self.t[2]) * inv * inv
        return float(np.hypot(du, dv)[inside].max())

    def maps(self, rho):
        """(map_x, map_y, in_front) float32 index coordinates in the partner for inverse depth ``rho`` (scalar or
        (H, W))."""
        rho = np.float32(rho) if np.ndim(rho) == 0 else np.asarray(rho, dtype=np.float32)
        den = self.az + self.t[2] * rho
        front = den > np.float32(1e-6)
        inv = np.divide(np.float32(1.0), den, out=np.zeros(self.shape, np.float32), where=front)
        mx = (self.ax + self.t[0] * rho) * inv * self.fx + self.cx
        my = (self.ay + self.t[1] * rho) * inv * self.fy + self.cy
        return mx.astype(np.float32, copy=False), my.astype(np.float32, copy=False), front


def remap_cubic(src: np.ndarray, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    return cv2.remap(np.asarray(src, dtype=np.float32), mx, my, interpolation=cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REPLICATE)


def in_bounds(mx, my, front, W_p: int, H_p: int) -> np.ndarray:
    """Projection inside the partner with the cubic kernel's support (index x in [1, W - 3])."""
    return front & (mx >= 1.0) & (mx <= W_p - 3.0) & (my >= 1.0) & (my <= H_p - 3.0)


def window_all(mask: np.ndarray, w: int) -> np.ndarray:
    """True where every pixel of the w x w window is True."""
    k = np.ones((int(w), int(w)), np.uint8)
    return cv2.erode(mask.astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)


def warp_to_reference(partner_lum, depth_ref, K_ref, T_ref, K_p, T_p, geom: PartnerGeom | None = None):
    """Warp one partner's luma into the reference at the given reference depth (z along the optical axis, NaN or
    <= 0 = none). Returns (warped float32, inside bool): inside = in front of the partner and within the image
    with the cubic kernel's support. The same warp the plane sweep uses."""
    z = np.asarray(depth_ref, dtype=np.float64)
    H, W = z.shape
    g = geom or PartnerGeom(K_ref, T_ref, K_p, T_p, (H, W))
    good = np.isfinite(z) & (z > 0)
    rho = np.where(good, 1.0 / np.where(good, z, 1.0), 0.0).astype(np.float32)
    mx, my, front = g.maps(rho)
    Hp, Wp = np.asarray(partner_lum).shape[:2]
    inside = in_bounds(mx, my, front, Wp, Hp) & good
    return remap_cubic(partner_lum, mx, my), inside


# ---------------------------------------------------------------------------------------------------- hand mask

def load_hand_model(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return parse_hand_model(json.load(f))


def parse_hand_model(doc: dict) -> dict:
    if doc.get("format") != HAND_MODEL_FORMAT or doc.get("frame") != "eef":
        raise ValueError("not a hand model v1 in the eef frame")
    bodies = []
    for b in doc["bodies"]:
        V = np.asarray(b["vertices"], dtype=np.float64).reshape(-1, 3)
        F = np.asarray(b["faces"], dtype=np.int64).reshape(-1, 3)
        if F.size and (F.min() < 0 or F.max() >= len(V)):
            raise ValueError(f"body {b.get('name')}: face index out of range")
        jt = b.get("joint")
        if jt is not None:
            jt = {"axis_eef": np.asarray(jt["axis_eef"], dtype=np.float64).reshape(3),
                  "qpos_index": int(jt["qpos_index"]), "sign": float(jt["sign"])}
        bodies.append({"name": str(b["name"]), "T0": np.asarray(b["T_eef_from_body_at_zero"], np.float64).reshape(4, 4),
                       "joint": jt, "V": V, "F": F})
    return {"bodies": bodies}


def hand_vertices_eef(model: dict, gripper_qpos) -> list:
    """[(V_eef (n, 3), F)] per body at the gripper qpos: T0 then sign * q[i] along axis_eef (eef frame)."""
    q = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
    out = []
    for b in model["bodies"]:
        V = b["V"] @ b["T0"][:3, :3].T + b["T0"][:3, 3]
        if b["joint"] is not None:
            j = b["joint"]
            V = V + j["sign"] * q[j["qpos_index"]] * j["axis_eef"]
        out.append((V, b["F"]))
    return out


def _clip_near(P: np.ndarray, zn: float) -> np.ndarray:
    """Sutherland-Hodgman clip of one polygon (k, 3) against z >= zn."""
    out = []
    k = len(P)
    for i in range(k):
        a, b = P[i], P[(i + 1) % k]
        ina, inb = a[2] >= zn, b[2] >= zn
        if ina:
            out.append(a)
        if ina != inb:
            s = (zn - a[2]) / (b[2] - a[2])
            out.append(a + s * (b - a))
    return np.asarray(out) if out else np.zeros((0, 3))


def hand_raster(model: dict, gripper_qpos, X, K, width: int, dilate: bool = True, z_near: float = 1e-3) -> np.ndarray:
    """The hand-and-finger raster: every hand-model triangle at the gripper qpos, put in the camera frame by
    X^-1 (X = hand-from-camera), projected with K (pixel-centre convention) and filled; dilated by
    round(3 W / 512) px (3 at 512, 6 at 1024) with a disc."""
    W = int(width)
    K = np.asarray(K, dtype=np.float64)
    Xinv = np.linalg.inv(np.asarray(X, dtype=np.float64))
    img = np.zeros((W, W), np.uint8)
    SH = 4
    lim = float(2 ** 26 // (1 << SH))

    def fill(Pc: np.ndarray) -> None:
        u = K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2] - 0.5
        v = K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2] - 0.5
        pts = np.stack([np.clip(u, -lim, lim), np.clip(v, -lim, lim)], -1)
        cv2.fillConvexPoly(img, np.rint(pts * (1 << SH)).astype(np.int32), 1, lineType=cv2.LINE_8, shift=SH)

    for V, F in hand_vertices_eef(model, gripper_qpos):
        if not len(F):
            continue
        Pc = V @ Xinv[:3, :3].T + Xinv[:3, 3]
        tri = Pc[F]  # (m, 3, 3)
        zin = tri[..., 2] >= z_near
        allin = zin.all(axis=1)
        some = zin.any(axis=1) & ~allin
        if allin.any():
            T3 = tri[allin]
            u = K[0, 0] * T3[..., 0] / T3[..., 2] + K[0, 2] - 0.5
            v = K[1, 1] * T3[..., 1] / T3[..., 2] + K[1, 2] - 0.5
            pts = np.rint(np.stack([np.clip(u, -lim, lim), np.clip(v, -lim, lim)], -1) * (1 << SH)).astype(np.int32)
            # skip triangles wholly outside the image
            lo, hi = pts.min(axis=1), pts.max(axis=1)
            vis = (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < W << SH) & (lo[:, 1] < W << SH)
            for p in pts[vis]:
                cv2.fillConvexPoly(img, p, 1, lineType=cv2.LINE_8, shift=SH)
        for t in tri[some]:
            P = _clip_near(t, z_near)
            if len(P) >= 3:
                fill(P)
    m = img.astype(bool)
    if dilate:
        r = max(1, int(round(HAND_DILATE_PX_AT_512 * W / 512.0)))
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        m = cv2.dilate(img, se).astype(bool)
    return m


def rigid_or_flat_mask(ref_lum: np.ndarray, partner_lums: list, b_effs: list) -> np.ndarray:
    """True where the zero-displacement 5x5 mean absolute luma difference is <= 2.0 grey levels against every used
    partner with b_eff >= 1 cm, or against every used partner when none has b_eff >= 1 cm (none used: all False)."""
    sel = [i for i, b in enumerate(b_effs) if b >= B_RIGID_M] or list(range(len(partner_lums)))
    ref = np.asarray(ref_lum, dtype=np.float32)
    if not sel:
        return np.zeros(ref.shape, bool)
    m = np.ones(ref.shape, bool)
    for i in sel:
        mad = box(np.abs(ref - np.asarray(partner_lums[i], dtype=np.float32)), RIGID_WIN)
        m &= mad <= RIGID_MAD_MAX
    return m


def flying_pixels(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """True where the 3x3 range of valid depth exceeds 0.03 z."""
    zz = np.asarray(z, dtype=np.float32)
    k = np.ones((3, 3), np.uint8)
    hi = cv2.dilate(np.where(valid, zz, 0.0).astype(np.float32), k, borderType=cv2.BORDER_REPLICATE)
    lo = cv2.erode(np.where(valid, zz, np.float32(1e9)).astype(np.float32), k, borderType=cv2.BORDER_REPLICATE)
    return valid & ((hi - lo) > FLY_REL * zz)


def common_masks(look_data: dict, sel: dict, model: dict | None, w_tex: int):
    """(excluded, info): hand raster | rigid-or-flat | texture gate failure, on the reference grid."""
    ref = look_data["ref"]
    W = look_data["width"]
    L = luma(ref["rgb"])
    used = sel["used"]
    t0 = time.perf_counter()
    hand = (hand_raster(model, ref["gripper_qpos"], look_data["X"], ref["K"], W) if model is not None
            else np.zeros(L.shape, bool))
    t1 = time.perf_counter()
    rig = rigid_or_flat_mask(L, [luma(p["rgb"]) for p in used], [sel["b_eff"][p["view_id"]] for p in used])
    tex = texture_map(L, w_tex) >= TAU_TEX
    t2 = time.perf_counter()
    info = {"hand_px": int(hand.sum()), "rigid_px": int(rig.sum()), "untextured_px": int((~tex).sum()),
            "hand_s": round(t1 - t0, 4), "masks_s": round(t2 - t1, 4)}
    return hand | rig | ~tex, {"hand": hand, "rigid": rig, "textured": tex}, info


# ------------------------------------------------------------------------------------------------- plane sweep

def _half(lum: np.ndarray) -> np.ndarray:
    H, W = lum.shape
    return cv2.resize(np.asarray(lum, dtype=np.float32), (W // 2, H // 2), interpolation=cv2.INTER_AREA)


def plane_sweep(ref_lum: np.ndarray, K_ref, T_ref, partners: list, w: int, zncc_min: float, margin_min: float,
                excluded: np.ndarray | None = None, want_maps: bool = False):
    """The PS depth map (float32 metres, NaN = no reading) of the reference from ``partners`` = [(luma, K, T)].

    Returns (depth, info)."""
    t0 = time.perf_counter()
    L = np.asarray(ref_lum, dtype=np.float32)
    H, W = L.shape
    if H % 2 or W % 2:
        raise ValueError(f"image {W} x {H}: the coarse pass needs even sizes")
    K_ref = np.asarray(K_ref, dtype=np.float64)
    b_list = [b_eff(T_ref, T) for (_, _, T) in partners]
    # ---- coarse: half resolution, 1 px of the widest view per plane
    w_c = int(w)  # 1 - ZNCC over w x w at each pass's own resolution
    Kh = np.diag([0.5, 0.5, 1.0]) @ K_ref
    Lh = _half(L)
    hh, wh = Lh.shape
    rw_h = RefWindow(Lh, w_c)
    ph = []
    for (lum, K_p, T_p) in partners:
        Kph = np.diag([0.5, 0.5, 1.0]) @ np.asarray(K_p, dtype=np.float64)
        ph.append((_half(lum), PartnerGeom(Kh, T_ref, Kph, T_p, (hh, wh))))
    rho_lo, rho_hi = 1.0 / Z_MAX_M, 1.0 / Z_MIN_M
    # "1 px disparity of the widest view": the largest image displacement per unit inverse depth over the used views
    # (f b for a sideways move; larger for a move along the optical axis), at full resolution
    rates = [max(g.rate(rho_lo, Jh.shape[1], Jh.shape[0], 4), g.rate(rho_hi, Jh.shape[1], Jh.shape[0], 4)) * 2.0
             for (Jh, g) in ph]
    rate_max = max(rates)
    if not rate_max > 0:
        raise LookFailure("failed", "no used view sees the reference")
    d_rho_c = 2.0 / rate_max
    n_planes = int(math.ceil((rho_hi - rho_lo) / d_rho_c)) + 1
    capped = n_planes > MAX_PLANES
    if capped:  # a partner moving toward close geometry: flagged, steps widened to the cap
        n_planes = MAX_PLANES
        d_rho_c = (rho_hi - rho_lo) / (n_planes - 1)
        rate_max = 2.0 / d_rho_c
    tr = MinTracker((hh, wh))
    for k in range(n_planes):
        rho = rho_lo + k * d_rho_c
        costs = []
        for (Jh, g) in ph:
            mx, my, front = g.maps(rho)
            ok = window_all(in_bounds(mx, my, front, Jh.shape[1], Jh.shape[0]), w_c)
            z = rw_h.zncc(remap_cubic(Jh, mx, my))
            c = np.where(ok & np.isfinite(z), np.float32(1.0) - z, np.float32(np.inf)).astype(np.float32)
            costs.append(c)
        tr.push(best_half_mean(costs))
    co = tr.result()
    rho_c = (rho_lo + (co["k"].astype(np.float64) + co["off"]) * d_rho_c).astype(np.float32)
    ok_c = co["ok"] & (rho_c > 0)
    t1 = time.perf_counter()

    def up(a):
        return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)

    rho_c_f = np.where(up(ok_c), up(rho_c), np.float32(rho_lo)).astype(np.float32)
    ok_cf = up(ok_c)
    margin_f = up(co["margin"])
    # ---- refine: full resolution, +-2 px of the widest view in 0.25 px steps, parallax rule per partner
    rw = RefWindow(L, w)
    pf = []
    for (lum, K_p, T_p) in partners:
        g = PartnerGeom(K_ref, T_ref, K_p, T_p, (H, W))
        mx0, my0, f0 = g.maps(rho_c_f)
        usable = f0 & ok_cf
        for s in (1.0 + PARALLAX_REL, 1.0 - PARALLAX_REL):
            mx1, my1, f1 = g.maps(rho_c_f / np.float32(s))
            usable &= f1 & (np.hypot(mx1 - mx0, my1 - my0) >= PARALLAX_PX)
        pf.append((np.asarray(lum, dtype=np.float32), g, usable))
        del mx0, my0, f0
    deltas = np.arange(-REFINE_HALF_PX, REFINE_HALF_PX + 1e-9, REFINE_STEP_PX)
    d_rho_f = 1.0 / rate_max
    trf = MinTracker((H, W))
    n_used_px = np.zeros((H, W), np.int16)
    for j, dl in enumerate(deltas):
        rho = (rho_c_f + np.float32(dl * d_rho_f)).astype(np.float32)
        rho = np.maximum(rho, np.float32(1e-3))
        costs = []
        cnt = np.zeros((H, W), np.int16)
        for (J, g, usable) in pf:
            mx, my, front = g.maps(rho)
            ok = window_all(in_bounds(mx, my, front, J.shape[1], J.shape[0]), w) & usable
            z = rw.zncc(remap_cubic(J, mx, my))
            ok &= np.isfinite(z)
            c = np.where(ok, np.float32(1.0) - z, np.float32(np.inf)).astype(np.float32)
            cnt += ok
            costs.append(c)
        if j == len(deltas) // 2:
            n_used_px = cnt
        trf.push(best_half_mean(costs))
    rf = trf.result()
    t2 = time.perf_counter()
    rho_f = rho_c_f + ((rf["k"].astype(np.float64) - len(deltas) // 2 + rf["off"]) * REFINE_STEP_PX
                       * d_rho_f).astype(np.float32)
    best_zncc = np.float32(1.0) - rf["c"]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(rho_f > 0, 1.0 / rho_f, np.nan).astype(np.float32)
    gate_conf = (best_zncc >= zncc_min) & (margin_f >= margin_min)
    valid = ok_cf & rf["ok"] & gate_conf & np.isfinite(z) & (z >= Z_MIN_M) & (z <= Z_MAX_M)
    if excluded is not None:
        valid &= ~excluded
    fly = flying_pixels(z, valid)
    valid &= ~fly
    depth = np.where(valid, z, np.nan).astype(np.float32)
    t3 = time.perf_counter()
    info = {"n_planes": n_planes, "planes_capped": bool(capped), "w_coarse": w_c, "rate_max_px_m": round(rate_max, 3),
            "rates_px_m": [round(r, 3) for r in rates], "b_eff_m": [round(b, 6) for b in b_list],
            "coarse_ok_px": int(ok_c.sum()) * 4, "refine_ok_px": int((rf["ok"] & ok_cf).sum()),
            "conf_ok_px": int((gate_conf & ok_cf & rf["ok"]).sum()), "flying_px": int(fly.sum()),
            "valid_px": int(valid.sum()), "no_usable_partner_px": int((ok_cf & (n_used_px == 0)).sum()),
            "coarse_s": round(t1 - t0, 4), "refine_s": round(t2 - t1, 4), "gate_s": round(t3 - t2, 4)}
    if want_maps:
        info["maps"] = {"best_zncc": best_zncc, "margin": margin_f, "coarse_ok": ok_cf, "refine_ok": rf["ok"],
                        "usable": [u for (_, _, u) in pf]}
    return depth, info


# ----------------------------------------------------------------------------------------------- one look

def encode_map(depth: np.ndarray) -> np.ndarray:
    """depth_u16 through m2_camera.encode_depth with a zero zbuf and NaN holes; 65535 is never written."""
    z = np.asarray(depth, dtype=np.float64)
    u16 = m2_camera.encode_depth(z, np.zeros_like(z))
    assert not np.any(u16 == 65535), "65535 in a stereo depth map"
    return u16


def run_look(look_data: dict, sel: dict, cfg: dict, model: dict | None, want_maps: bool = False):
    """Depth (NaN holes) and info for one loaded look with its view selection; raises LookFailure."""
    if sel["status"] is not None:
        raise LookFailure(sel["status"], sel["reason"])
    ref = look_data["ref"]
    used = sel["used"]
    t0 = time.perf_counter()
    if cfg["method"] == "PS":
        excl, masks, minfo = common_masks(look_data, sel, model, cfg["w"])
        depth, info = plane_sweep(luma(ref["rgb"]), ref["K"], ref["T"], [(luma(p["rgb"]), p["K"], p["T"]) for p in used],
                                  cfg["w"], cfg["zncc_min"], cfg["margin_min"], excluded=excl, want_maps=want_maps)
    else:
        from rowant_lab.libero_env import m3_stereo1_sgbm as SGM

        excl, masks, minfo = common_masks(look_data, sel, model, cfg["bs"])
        z, info = SGM.sgbm_depth(luma_u8(ref["rgb"]), ref["K"], ref["T"],
                                 [(luma_u8(p["rgb"]), p["K"], p["T"], p["view_id"]) for p in used], cfg["bs"], cfg["ur"])
        if not info["pairs_used"]:
            raise LookFailure("failed", "every SG pair skipped")
        valid = np.isfinite(z) & (z >= Z_MIN_M) & (z <= Z_MAX_M) & ~excl
        fly = flying_pixels(z, valid)
        valid &= ~fly
        depth = np.where(valid, z, np.nan).astype(np.float32)
        info.update({"flying_px": int(fly.sum()), "valid_px": int(valid.sum())})
    info.update(minfo)
    info["wall_s"] = round(time.perf_counter() - t0, 4)
    if want_maps:
        info["masks"] = masks
    return depth, info


# ---------------------------------------------------------------------------------------------------- the CLI

def _install_hook(AUD, root: Path, arm: str, hand_path: Path):
    """Install the strict audit hook of m3_stereo1_audit (stereo process; reads R/frames/**, R/poses/<arm>/**, the
    hand model; writes R/net/**, R/audit/**)."""
    fn = getattr(AUD, "install_hook", None)
    if fn is None:
        raise RuntimeError("m3_stereo1_audit has no install_hook")
    return fn(root=root, process="stereo", arm=arm, mode="strict", extra_read=(str(hand_path),))


def _excl_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def _npz_bytes(**arrays) -> bytes:
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def out_path(root: Path, gen: str, method: str, config: str, arm: str, scene: str, vid: str, pass_tag: str) -> Path:
    if gen == "rerun":
        return Path(root) / "net" / "rerun" / pass_tag / method / config / arm / scene / f"{vid}.json"
    return Path(root) / "net" / gen / method / config / arm / scene / f"{vid}.npz"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog=MODULE, description="K0-stereo stereo process (started by the launcher)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--method", required=True, choices=("PS", "SG"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--width", required=True, type=int)
    ap.add_argument("--looks", required=True, nargs="+")
    ap.add_argument("--pass-tag", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--hand-sha", required=True)
    ap.add_argument("--code-tree", required=True)
    ap.add_argument("--clean", required=True, choices=("0", "1"))
    ap.add_argument("--hand-model", default=None, help="synthetic test roots only")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    from rowant_lab.libero_env import m3_stereo1_audit as AUD

    root = Path(a.root)
    for name, v in (("arm", a.arm), ("gen", a.gen), ("pass tag", a.pass_tag)):
        if not TOKEN_RE.match(v):
            print(json.dumps({"refused": f"bad {name} {v!r}"}), flush=True)
            return 2
    if not AUD.root_ok(root, "stereo"):
        print(json.dumps({"refused": f"root {root} is not a registered root"}), flush=True)
        return 2
    synthetic = (root / SYNTHETIC_MARKER).is_file()
    if a.hand_model is not None and not synthetic:
        print(json.dumps({"refused": "--hand-model is for synthetic test roots only"}), flush=True)
        return 2
    hand_path = Path(a.hand_model) if a.hand_model is not None else REPO / HAND_MODEL_REL
    try:
        cfg = parse_config(a.method, a.config, a.width)
        keys = [parse_look_key(k) for k in a.looks]
    except ValueError as exc:
        print(json.dumps({"refused": str(exc)}), flush=True)
        return 2
    if any(w != a.width for (_, _, w) in keys):
        print(json.dumps({"refused": "a look key's width differs from --width"}), flush=True)
        return 2
    outs = []
    try:
        for scene, look, width in keys:
            why = rm_look_refusal(a.arm, look)
            if why:
                raise ValueError(why)
            spec = look_spec(look, a.arm)
            outs.append(out_path(root, a.gen, a.method, a.config, a.arm, scene, view_id(look, spec["ref"], width),
                                 a.pass_tag))
    except ValueError as exc:
        print(json.dumps({"refused": str(exc)}), flush=True)
        return 2
    if len(set(outs)) != len(outs):
        print(json.dumps({"refused": "two looks of this process would write the same output"}), flush=True)
        return 2
    for p in outs:
        if p.exists():
            print(json.dumps({"refused": f"output exists: {p}"}), flush=True)
            return 4
    _install_hook(AUD, root, a.arm, hand_path)
    cv2.setNumThreads(CV2_THREADS)
    with open(hand_path, "rb") as f:
        hand_bytes = f.read()
    hand_sha = sha256_bytes(hand_bytes)
    if hand_sha != a.hand_sha:
        print(json.dumps({"refused": f"hand model sha256 {hand_sha} != {a.hand_sha}"}), flush=True)
        return 3
    model = parse_hand_model(json.loads(hand_bytes.decode("utf-8")))
    n_fail = 0
    for i, ((scene, look, width), p) in enumerate(zip(keys, outs)):
        t_load = time.perf_counter()
        status, reason, info, depth, K_used, sha, views, beff = "ok", None, {}, None, np.zeros((3, 3)), None, [], {}
        try:
            if a.arm == "S6":
                cp = root / "poses" / "S6" / scene / "calib.json"
                if not cp.is_file():
                    raise LookFailure("failed", "calibration failed: no calibration record")
                with open(cp, "r", encoding="utf-8") as f:
                    cal = json.load(f)
                if cal.get("status") != "ok":
                    raise LookFailure("failed", f"calibration failed: {cal.get('reason')}")
            ld = load_look(root, a.arm, scene, look, width)
            K_used = ld["ref"]["K"]
            sha = array_sha256(ld["ref"]["rgb"])
            sel = select_views(ld)
            beff = {k: round(v, 6) for k, v in sel["b_eff"].items()}
            views = [q["view_id"] for q in sel["used"]]
            load_s = time.perf_counter() - t_load
            depth, info = run_look(ld, sel, cfg, model)
        except LookFailure as exc:
            status, reason = exc.status, exc.reason
            load_s = time.perf_counter() - t_load
        except AUD.AuditViolation:
            # a blocked read or write is a harness stop, never a failed look: the process ends here (the hook's exit
            # handler exits 70), so the launcher records rc 70 and the eval refuses this process's outputs
            raise
        except Exception as exc:  # noqa: BLE001 - the failure rule: an exception fails the look
            status, reason = "failed", f"exception: {type(exc).__name__}: {exc}"
            load_s = time.perf_counter() - t_load
        if status != "ok":
            n_fail += 1
            depth = np.full((width, width), np.nan, np.float32)
        u16 = encode_map(depth)
        valid = u16 > 0
        if status != "ok":
            assert not valid.any()
        meta = {"format": NET_FORMAT, "look_key": f"{scene}/{look}/{width}", "look": look, "arm": a.arm,
                "method": a.method, "config": a.config, "width": width, "gen": a.gen, "pass_tag": a.pass_tag,
                "status": status, "reason": reason, "frame_rgb_sha256": sha, "views_used": views, "b_eff_m": beff,
                "n_valid": int(valid.sum()), "valid_share": round(float(valid.mean()), 6),
                "timings": {"load_s": round(load_s, 4), "wall_s": info.get("wall_s"),
                            **{k: v for k, v in info.items() if k.endswith("_s") and k != "wall_s"}},
                "wall_s": info.get("wall_s"), "look_index": i, "pid": os.getpid(), "utc": utc(),
                "load_1min": round(os.getloadavg()[0], 3),
                "peak_rss_bytes": peak_rss_bytes(), "cv2_threads": cv2.getNumThreads(), "tau_tex": TAU_TEX,
                "hand_sha256": hand_sha, "code_tree": a.code_tree, "clean": a.clean == "1",
                "info": {k: v for k, v in info.items() if not k.endswith("_s")}}
        t_w = time.perf_counter()
        if a.gen == "rerun":
            body = {"meta": meta, "depth_u16_sha256": array_sha256(u16), "valid_sha256": array_sha256(valid)}
            _excl_write(p, (json.dumps(body, sort_keys=True) + "\n").encode("utf-8"))
        else:
            _excl_write(p, _npz_bytes(depth_u16=u16, valid=valid, K_used=np.asarray(K_used, np.float64),
                                      meta=np.asarray(json.dumps(meta, sort_keys=True))))
        print(json.dumps({"look": meta["look_key"], "status": status, "reason": reason, "wall_s": meta["wall_s"],
                          "write_s": round(time.perf_counter() - t_w, 4)}), flush=True)
    print(json.dumps({"done": len(keys), "failed": n_fail, "peak_rss_bytes": peak_rss_bytes()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
