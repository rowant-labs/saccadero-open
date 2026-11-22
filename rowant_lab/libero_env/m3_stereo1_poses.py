"""K0-stereo poses (Monty venv; numpy and the standard library): P-T, P-O, poses_pick.json and the pose checks.

Registered: k0_stereo.poses_and_calibration (P_T, P_O, dt_lag_pick,
poses_pick_schema, X_true, K_true, pipeline_checks (1), S3_D1, sweep), capture.lag_present_check; design section 4;
the contract's "Files under R" (pose files). Monty-venv launch form (``PYTHONPATH=$PWD`` at the repository root,
``python -B``, never ``-I``; no MuJoCo, robosuite or Perceiver)::

    python -B -m rowant_lab.libero_env.m3_stereo1_poses pt --root R [--scenes ID ...]
    python -B -m rowant_lab.libero_env.m3_stereo1_poses poses-pick --root R --pass-tag P
    python -B -m rowant_lab.libero_env.m3_stereo1_poses settled-pose --root R --pass-tag P [--dt-lag-ms D]
    python -B -m rowant_lab.libero_env.m3_stereo1_poses po --root R --arm A --method M --config C --width W \\
        --looks <scene>/<look>/<width> ... --pass-tag P --gen G --hand-sha H --code-tree T --clean {0,1}

* ``pt`` (P-T; logging-mode hook): arms A0 and S2. Every view's simulator T and K from ``cam_truth``;
  ``X = E_rendered^-1 T`` of the look's reference view. A0's ``traj1-rs`` file also holds the agentview rs00 view
  (A0's L0 seed); S2's does not. Opens cam_truth only for tune and smoke scenes until C4 is an ancestor of origin
  (``m3_stereo1_audit.c4_landed``); scoring and held-out scenes are refused before C4.
* ``po`` (P-O; strict hook; started only by the launcher): ``T_i = Ecorr_i X_true`` with ``K_true``,
  ``Ecorr_i = E_i exp((dt_lag / 0.05 s) log(E_prev_i^-1 E_i))``, E from the policy file's robot0_eef_pos /
  robot0_eef_quat (xyzw), E_prev from its prev_eef_pos / prev_eef_quat; reads policy files and
  ``R/eval/poses_pick.json`` only (and the hand model, for its sha check). Arms: S3 (S3-L14 is S3 on traj2-E2),
  S3-RM (traj1-pgm), S3-b1 / S3-b2 (traj1-pg with pgp1 / pgp1-pgp2 only), S3-D1 (``T = E_prev X_true``), P1 and P2
  (S3's poses; the photometric change is the stereo's), and ``S3-sweep-<factor>-<level>`` (``T Delta`` with Delta
  from seed 7501, K scaled at focal levels). Scene ids are opaque here: a look key is split on "/" and its scene
  part is only a path component.
* ``poses-pick`` (tune and smoke scenes only; logging hook): the lag-present check, the X_true check (median over
  every tune capture; stop beyond 1e-6 m or 1e-6 rad), f_true per width (cam_truth K, checked against fovy 75), the
  dt_lag pick (lowest median camera-centre error over every tune capture; within 0.02 mm the smaller dt wins) ->
  ``R/eval/poses_pick.json`` ``{dt_lag_ms, X_true, f_true, code_tree}`` (fixed schema, ``check_poses_pick``) and
  rows in ``R/eval/pipeline_checks.jsonl`` (numbers only). A failed check writes no poses_pick.json (exit 1).
* ``settled-pose`` (pipeline check 1): on tune rs00 views and tune E2 C captures that met the 0.01 rad/s rule (the
  policy file's robot0_joint_vel), at both widths, P-O (in memory, this module's own P-O code) against P-T within
  0.1 mm and 0.01 deg; plus the count of moving views whose P-O and P-T camera centres are identical.

Pose file ``R/poses/<arm>/<scene>/<look>.json``: ``{"arm", "look", "X", "width_views": {view_id: {"T", "K"}},
"code_tree", "clean"}`` (4x4 and 3x3 nested lists of float64, exact through json). Files are created with O_EXCL; an
existing file is left as it is (equal: skipped; different: an error, never overwritten). This module never names
subprocess, os.system, os.exec*, os.posix_spawn or socket (the git calls of P-T and the checks live in
``m3_stereo1_audit``; a P-O process never calls them).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from rowant_lab.libero_env import m3_stereo1_audit as AU

FORMAT = "rowant m3 stereo1 poses v1"
WRIST = "robot0_eye_in_hand"
AGENT = "agentview"
WIDTHS = (512, 1024)
CONTROL_DT_MS = 50.0  # 20 Hz control: one env step
DT_LAG_CANDIDATES_MS = (0, 2, 4, 6)
DT_TIE_MM = 0.02
X_TRUE_TOL_M = 1e-6
X_TRUE_TOL_RAD = 1e-6
FOVY_TRUE_DEG = 75.0  # robosuite panda robot.xml:238
F_TRUE_TOL_PX = 1e-9
SETTLED_CENTRE_MM_MAX = 0.1  # decision_lines_judgment.sanity.settled_pose_centre_mm_max
SETTLED_ROT_DEG_MAX = 0.01  # decision_lines_judgment.sanity.settled_pose_rot_deg_max
SETTLE_JOINT_VEL_MAX = 0.01  # capture.E2: max |robot0_joint_vel| < 0.01 rad/s
LAG_PRESENT_MM_MIN = 0.1
LAG_PRESENT_RAD_MIN = 1e-4
# "P-O and P-T camera centres are not identical on moving views" (verifier): identical = within float noise of the
# policy file's quaternion round trip (1 nm), far below any obs-timing difference (micrometres and up)
IDENTICAL_CENTRE_M = 1e-9
EEF_QUAT_ORDER = "xyzw"  # robosuite single_arm.py:304-309 (T.convert_quat(..., to="xyzw"))

RS_TOKENS = tuple(f"rs{k:02d}" for k in range(0, 11))
PGM_TOKENS = tuple(f"pgm{k:02d}" for k in range(1, 13))
PPM_TOKENS = tuple(f"ppm{k:02d}" for k in range(1, 13))
# look -> (traj, reference token, partner tokens): the contract's "Ids"
LOOKS = {
    "traj1-rs": ("traj1", "rs00", RS_TOKENS[1:]),
    "traj1-pg": ("traj1", "pgr0", ("pgp1", "pgp2", "pgp3")),
    "traj1-pgm": ("traj1", "pgr0", PGM_TOKENS),
    "traj1-pp": ("traj1", "ppr0", PPM_TOKENS),
    "traj2-E2": ("traj2", "C", ("R", "L")),
}
MOVING_TOKENS = tuple(f"rs{k:02d}" for k in range(1, 9)) + ("pgp1", "pgp2", "pgp3") + PGM_TOKENS + PPM_TOKENS
KNOWN_TOKENS = {"traj1": set(RS_TOKENS) | {"pgr0", "pgp1", "pgp2", "pgp3", "ppr0"} | set(PGM_TOKENS) | set(PPM_TOKENS),
                "traj2": {"C", "R", "L"}}

PT_ARMS = ("A0", "S2")
# P-O arms -> (formula, looks the arm may be written for (None: any), partner tokens kept (None: the look's))
PO_ARMS = {
    "S3": ("ecorr", None, None),
    "S3-RM": ("ecorr", ("traj1-pgm",), None),
    "S3-b1": ("ecorr", ("traj1-pg",), ("pgp1",)),
    "S3-b2": ("ecorr", ("traj1-pg",), ("pgp1", "pgp2")),
    "S3-D1": ("prev", ("traj1-pg",), None),
    "P1": ("ecorr", ("traj1-rs", "traj1-pg"), None),
    "P2": ("ecorr", ("traj1-rs", "traj1-pg"), None),
}
SWEEP_SEED = 7501
SWEEP_FACTORS = ("xrot", "xtrans", "focal")  # deg, mm, percent
SWEEP_LEVELS = {"xrot": ("0.1", "0.2", "0.4", "0.7", "1.0", "1.5"), "xtrans": ("1", "2", "3", "5", "8"),
                "focal": ("0.5", "1", "2", "3", "5")}
# axis (unit, camera frame) and sign per level, drawn once from default_rng(7501) in the order factor x level:
# axis = normal(3) / norm, sign = +1 if integers(0, 2) == 1 else -1 (a test regenerates them). Rotation levels turn
# about sign * axis, translation levels move along sign * axis, focal levels scale f by 1 + sign * level / 100.
SWEEP_DRAWS = {
    ("xrot", "0.1"): ((-0.039287, 0.505899, -0.861698), +1),
    ("xrot", "0.2"): ((-0.559098, -0.436542, -0.704869), +1),
    ("xrot", "0.4"): ((-0.566863, -0.808116, -0.160045), -1),
    ("xrot", "0.7"): ((-0.693932, -0.237129, 0.679874), -1),
    ("xrot", "1.0"): ((-0.936240, 0.349720, -0.033913), -1),
    ("xrot", "1.5"): ((0.579772, -0.554301, -0.597172), -1),
    ("xtrans", "1"): ((-0.662907, -0.187923, 0.724734), +1),
    ("xtrans", "2"): ((0.003643, 0.931035, -0.364912), +1),
    ("xtrans", "3"): ((-0.045719, 0.639739, 0.767231), +1),
    ("xtrans", "5"): ((0.171558, 0.720185, -0.672236), -1),
    ("xtrans", "8"): ((0.951058, 0.300939, -0.070182), +1),
    ("focal", "0.5"): ((-0.456729, -0.025346, -0.889245), +1),
    ("focal", "1"): ((-0.408256, 0.482118, -0.775171), +1),
    ("focal", "2"): ((-0.192977, -0.950720, 0.242677), -1),
    ("focal", "3"): ((-0.471907, -0.759720, 0.447358), -1),
    ("focal", "5"): ((0.879563, -0.055582, 0.472525), +1),
}
POSES_PICK_KEYS = ("dt_lag_ms", "X_true", "f_true", "code_tree")
PIPELINE = ("eval", "pipeline_checks.jsonl")


# ===================================================================================================== SE(3)
def _vee(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0], S[1, 0] - S[0, 1]]) * 0.5


def _hat(w) -> np.ndarray:
    x, y, z = [float(v) for v in w]
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_to_mat(q, order: str = EEF_QUAT_ORDER) -> np.ndarray:
    q = np.asarray(q, float).reshape(4)
    if order == "xyzw":
        x, y, z, w = q
    elif order == "wxyz":
        w, x, y, z = q
    else:
        raise ValueError(order)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if not (n > 0 and math.isfinite(n)):
        raise ValueError("quaternion has no direction")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat_xyzw(R) -> np.ndarray:
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def pose_mat(R, t) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = np.asarray(R, float)
    M[:3, 3] = np.asarray(t, float).reshape(3)
    return M


def inv_pose(M) -> np.ndarray:
    M = np.asarray(M, float)
    R, t = M[:3, :3], M[:3, 3]
    return pose_mat(R.T, -R.T @ t)


def rot_angle(R) -> float:
    """The rotation angle of R (radians), stable near 0 and pi."""
    R = np.asarray(R, float)
    s = float(np.linalg.norm(_vee(R)))
    c = (float(np.trace(R)) - 1.0) * 0.5
    return math.atan2(s, c)


def so3_log(R) -> np.ndarray:
    R = np.asarray(R, float)
    th = rot_angle(R)
    if th < 1e-7:
        return _vee(R) * (1.0 + th * th / 6.0)
    if math.pi - th < 1e-6:
        A = (R + np.eye(3)) * 0.5
        i = int(np.argmax(np.diag(A)))
        a = A[:, i] / math.sqrt(max(A[i, i], 1e-300))
        if float(np.dot(_vee(R), a)) < 0:
            a = -a
        return a / np.linalg.norm(a) * th
    return _vee(R) * (th / math.sin(th))


def so3_exp(w) -> np.ndarray:
    w = np.asarray(w, float).reshape(3)
    th = float(np.linalg.norm(w))
    W = _hat(w)
    if th < 1e-8:
        return np.eye(3) + W + 0.5 * (W @ W)
    return np.eye(3) + (math.sin(th) / th) * W + ((1 - math.cos(th)) / (th * th)) * (W @ W)


def _V(w) -> np.ndarray:
    th = float(np.linalg.norm(w))
    W = _hat(w)
    if th < 1e-6:
        return np.eye(3) + 0.5 * W + (1.0 / 6.0) * (W @ W)
    return np.eye(3) + ((1 - math.cos(th)) / th ** 2) * W + ((th - math.sin(th)) / th ** 3) * (W @ W)


def se3_log(M) -> np.ndarray:
    """(w, v) with M = se3_exp(w, v): w the rotation vector, v = V(w)^-1 t."""
    M = np.asarray(M, float)
    w = so3_log(M[:3, :3])
    v = np.linalg.solve(_V(w), M[:3, 3])
    return np.concatenate([w, v])


def se3_exp(xi) -> np.ndarray:
    xi = np.asarray(xi, float).reshape(6)
    w, v = xi[:3], xi[3:]
    return pose_mat(so3_exp(w), _V(w) @ v)


def ecorr(E, E_prev, dt_lag_ms: float) -> np.ndarray:
    """Ecorr = E exp((dt_lag / 0.05 s) log(E_prev^-1 E)); dt_lag 0 returns E itself (no arithmetic)."""
    E = np.asarray(E, float)
    s = float(dt_lag_ms) / CONTROL_DT_MS
    if s == 0.0:
        return E.copy()
    return E @ se3_exp(s * se3_log(inv_pose(E_prev) @ E))


def pose_err(A, B) -> tuple:
    """(camera-centre distance in metres, rotation angle of A_R^T B_R in radians)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3])), rot_angle(A[:3, :3].T @ B[:3, :3])


def project_so3(M) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(M, float))
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    return U @ D @ Vt


def median_pose(Ms) -> np.ndarray:
    """Element-wise median rotation (projected onto SO(3)) and element-wise median translation."""
    A = np.stack([np.asarray(M, float) for M in Ms])
    return pose_mat(project_so3(np.median(A[:, :3, :3], axis=0)), np.median(A[:, :3, 3], axis=0))


def K_true(f: float, W: int) -> np.ndarray:
    """Square pixels, principal point (W / 2, W / 2) (as m2_camera.intrinsics)."""
    return np.array([[float(f), 0.0, W / 2.0], [0.0, float(f), W / 2.0], [0.0, 0.0, 1.0]])


def f_of_fovy(fovy_deg: float, W: int) -> float:
    return 0.5 * W / math.tan(math.radians(fovy_deg) / 2.0)


def hand_pose(pos, quat, order: str = EEF_QUAT_ORDER) -> np.ndarray:
    """The obs hand pose E: world grip-site position and hand rotation."""
    return pose_mat(quat_to_mat(quat, order), np.asarray(pos, float).reshape(3))


# ===================================================================================================== sweep
def sweep_arm(factor: str, level: str) -> str:
    if factor not in SWEEP_LEVELS or level not in SWEEP_LEVELS[factor]:
        raise ValueError(f"no sweep level {factor} {level}")
    return f"S3-sweep-{factor}-{level}"


def parse_sweep_arm(arm: str):
    """(factor, level) of an ``S3-sweep-<factor>-<level>`` arm, else None."""
    if not arm.startswith("S3-sweep-"):
        return None
    rest = arm[len("S3-sweep-"):]
    factor, _, level = rest.partition("-")
    if factor in SWEEP_LEVELS and level in SWEEP_LEVELS[factor]:
        return factor, level
    raise ValueError(f"unknown sweep arm {arm!r}")


SWEEP_ARMS = tuple(sweep_arm(f, lv) for f in SWEEP_FACTORS for lv in SWEEP_LEVELS[f])


def regenerate_sweep_draws() -> dict:
    rng = np.random.default_rng(SWEEP_SEED)
    out = {}
    for f in SWEEP_FACTORS:
        for lv in SWEEP_LEVELS[f]:
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v)
            out[(f, lv)] = (tuple(float(x) for x in v), 1 if int(rng.integers(0, 2)) == 1 else -1)
    return out


def sweep_delta(factor: str, level: str) -> tuple:
    """(Delta 4x4 right-multiplied on T_world_from_camera, focal scale) for one sweep level (camera frame)."""
    axis, sign = SWEEP_DRAWS[(factor, level)]
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a) * sign
    x = float(level)
    if factor == "xrot":
        return pose_mat(so3_exp(a * math.radians(x)), np.zeros(3)), 1.0
    if factor == "xtrans":
        return pose_mat(np.eye(3), a * (x / 1000.0)), 1.0
    if factor == "focal":
        return np.eye(4), 1.0 + sign * x / 100.0
    raise ValueError(factor)


def apply_sweep(T, K, factor: str, level: str) -> tuple:
    """The sweep level's (T Delta, K scaled at focal levels; principal point unchanged)."""
    D, s = sweep_delta(factor, level)
    K2 = np.array(K, float)
    K2[0, 0] *= s
    K2[1, 1] *= s
    return np.asarray(T, float) @ D, K2


def look_max_rel_rot_deg(doc: dict, width=None) -> float:
    """The look's largest reference-to-partner relative rotation (degrees) from a pose file's T (sweep reuse rule:
    translation levels reuse level 0 when this is under 0.1 deg)."""
    look = doc["look"]
    traj, ref, partners = LOOKS[look]
    best = 0.0
    by_w = {}
    for v, e in doc["width_views"].items():
        pv = parse_view(v)
        if pv["camera"] != WRIST or (width is not None and pv["width"] != int(width)):
            continue
        by_w.setdefault(pv["width"], {})[pv["token"]] = np.asarray(e["T"], float)
    for W, toks in by_w.items():
        if ref not in toks:
            continue
        for t in partners:
            if t in toks:
                best = max(best, math.degrees(rot_angle(toks[ref][:3, :3].T @ toks[t][:3, :3])))
    return best


# ===================================================================================================== ids, files
def parse_view(view_id: str) -> dict:
    parts = str(view_id).split("-")
    if len(parts) != 4 or not parts[3].isdigit():
        raise ValueError(f"view id {view_id!r} is not traj-token-camera-width")
    return {"traj": parts[0], "token": parts[1], "camera": parts[2], "width": int(parts[3])}


def view_id(traj: str, token: str, camera: str, width: int) -> str:
    if "-" in token or "-" in camera or "-" in traj:
        raise ValueError("no dash inside a view-id part")
    return f"{traj}-{token}-{camera}-{int(width)}"


def look_view_ids(look: str, view_ids, partners=None, camera: str = WRIST) -> tuple:
    """(reference view ids, partner view ids) of a look among ``view_ids`` (every width), wrist camera."""
    traj, ref, parts = LOOKS[look]
    keep = tuple(partners) if partners is not None else parts
    refs, prt = [], []
    for v in sorted(view_ids):
        try:
            pv = parse_view(v)
        except ValueError:
            continue
        if pv["traj"] != traj or pv["camera"] != camera:
            continue
        if pv["token"] == ref:
            refs.append(v)
        elif pv["token"] in keep:
            prt.append(v)
    return refs, prt


def list_view_ids(folder) -> list:
    try:
        names = os.listdir(folder)
    except FileNotFoundError:
        return []
    return sorted(n[:-4] for n in names if n.endswith(".npz"))


def pose_path(R, arm: str, scene: str, look: str) -> Path:
    return Path(R) / "poses" / arm / scene / f"{look}.json"


def _scalar(a):
    a = np.asarray(a)
    if a.ndim == 0:
        return a.item()
    return a


def read_policy(path) -> dict:
    """The pose-relevant fields of a v5 policy file (never rgb)."""
    keys = ("robot0_eef_pos", "robot0_eef_quat", "prev_eef_pos", "prev_eef_quat", "prev_step", "step",
            "robot0_joint_vel", "robot0_gripper_qpos", "camera", "width")
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for k in keys:
            if k in z.files:
                out[k] = _scalar(z[k])
    return out


def read_cam(path) -> dict:
    """K, T_world_from_camera, E_rendered and eef_lag of a cam_truth file (evaluator side)."""
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for k in ("K", "T_world_from_camera"):
            out[k] = np.asarray(z[k], float)
        out["E_rendered"] = np.asarray(z["E_rendered"], float) if "E_rendered" in z.files else None
        if "eef_lag" in z.files:
            try:
                out["eef_lag"] = json.loads(str(_scalar(z["eef_lag"])))
            except ValueError:
                out["eef_lag"] = None
    return out


def obs_E(fr: dict, order: str = EEF_QUAT_ORDER) -> np.ndarray:
    return hand_pose(fr["robot0_eef_pos"], fr["robot0_eef_quat"], order)


def obs_E_prev(fr: dict, order: str = EEF_QUAT_ORDER):
    p, q = fr.get("prev_eef_pos"), fr.get("prev_eef_quat")
    if p is None or q is None:
        return None
    p, q = np.asarray(p, float).reshape(-1), np.asarray(q, float).reshape(-1)
    if p.shape != (3,) or q.shape != (4,) or not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
        return None
    if float(np.linalg.norm(q)) == 0.0:
        return None
    return hand_pose(p, q, order)


def _jdump(doc) -> bytes:
    return (json.dumps(doc, sort_keys=True) + "\n").encode("utf-8")


def write_or_compare(path, doc: dict) -> str:
    """'written', 'same' (an equal file exists) or 'differs' (an existing file is left as it is)."""
    data = _jdump(doc)
    p = Path(path)
    if p.exists():
        with open(p, "rb") as f:
            return "same" if f.read() == data else "differs"
    AU.write_new(p, data)
    return "written"


def _m(M) -> list:
    return np.asarray(M, float).tolist()


# ===================================================================================================== P-T
def pt_look_doc(arm: str, look: str, cams: dict, prov: dict) -> dict | None:
    """P-T pose file for one look: ``cams`` {view_id: read_cam(...)} of the scene (views with a policy file)."""
    refs, prt = look_view_ids(look, cams)
    if not refs:
        return None
    ref = sorted(refs, key=lambda v: parse_view(v)["width"])[0]
    X = inv_pose(cams[ref]["E_rendered"]) @ cams[ref]["T_world_from_camera"]
    wv = {v: {"T": _m(cams[v]["T_world_from_camera"]), "K": _m(cams[v]["K"])} for v in refs + prt}
    if arm == "A0" and look == "traj1-rs":
        for v in sorted(cams):
            pv = parse_view(v)
            if pv["traj"] == "traj1" and pv["token"] == "rs00" and pv["camera"] == AGENT:
                wv[v] = {"T": _m(cams[v]["T_world_from_camera"]), "K": _m(cams[v]["K"])}
    return {"arm": arm, "look": look, "X": _m(X), "width_views": wv, "code_tree": prov["code_tree"],
            "clean": bool(prov["clean"])}


# ===================================================================================================== P-O
def po_view_T(arm: str, fr: dict, pick: dict, order: str = EEF_QUAT_ORDER, x_side: str = "right"):
    """(T, K) of one view for a P-O arm, or None when a pose is missing. ``x_side`` and ``order`` exist only for the
    fault-injection tests (X on the wrong side; a quaternion read in the wrong order)."""
    sweep = parse_sweep_arm(arm)
    formula = "ecorr" if sweep else PO_ARMS[arm][0]
    W = int(fr["width"])
    f = pick["f_true"].get(str(W))
    if f is None:
        return None
    X = np.asarray(pick["X_true"], float)
    E = obs_E(fr, order)
    Ep = obs_E_prev(fr, order)
    if formula == "prev":
        if Ep is None:
            return None
        Eb = Ep
    else:
        dt = float(pick["dt_lag_ms"])
        if dt != 0.0 and Ep is None:
            return None
        Eb = ecorr(E, Ep, dt) if dt != 0.0 else E
    T = Eb @ X if x_side == "right" else X @ Eb
    K = K_true(float(f), W)
    if sweep:
        T, K = apply_sweep(T, K, *sweep)
    return T, K


def po_look_doc(arm: str, look: str, frames: dict, pick: dict, code_tree: str, clean: bool,
                order: str = EEF_QUAT_ORDER) -> tuple:
    """(doc or None, missing view ids): P-O pose file for one look; ``frames`` {view_id: read_policy(...)}."""
    sweep = parse_sweep_arm(arm)
    partners = None if sweep else PO_ARMS[arm][2]
    refs, prt = look_view_ids(look, frames, partners=partners)
    if not refs:
        return None, []
    wv, missing = {}, []
    for v in refs + prt:
        r = po_view_T(arm, frames[v], pick, order)
        if r is None:
            missing.append(v)
            continue
        wv[v] = {"T": _m(r[0]), "K": _m(r[1])}
    doc = {"arm": arm, "look": look, "X": _m(pick["X_true"]), "width_views": wv, "code_tree": code_tree,
           "clean": bool(clean)}
    return doc, missing


def po_arm_ok(arm: str, look: str) -> str:
    """'' when a P-O arm may be written for the look, else the reason."""
    if look not in LOOKS:
        return f"unknown look {look!r}"
    if parse_sweep_arm(arm):
        return "" if look == "traj1-pg" else "sweep levels re-run the pgr0 look only"
    if arm not in PO_ARMS:
        return f"unknown P-O arm {arm!r}"
    allowed = PO_ARMS[arm][1]
    return "" if allowed is None or look in allowed else f"{arm} is written for {allowed} only"


def split_look_key(key: str) -> tuple:
    """(scene, look, width) of an opaque look key ``<scene>/<look>/<width>``; the scene part is never parsed."""
    parts = str(key).split("/")
    if len(parts) != 3 or not parts[0] or parts[0] in (".", "..") or not parts[2].isdigit():
        raise ValueError(f"look key {key!r} is not <scene>/<look>/<width>")
    return parts[0], parts[1], int(parts[2])


# ===================================================================================================== poses_pick
def check_poses_pick(doc) -> dict:
    """The fixed poses_pick.json schema; raises ValueError on any other key (a per-scene entry included)."""
    if not isinstance(doc, dict):
        raise ValueError("poses_pick.json is not an object")
    extra = sorted(set(doc) - set(POSES_PICK_KEYS))
    missing = sorted(set(POSES_PICK_KEYS) - set(doc))
    if extra or missing:
        raise ValueError(f"poses_pick.json keys: extra {extra}, missing {missing}")
    if doc["dt_lag_ms"] not in DT_LAG_CANDIDATES_MS or isinstance(doc["dt_lag_ms"], bool):
        raise ValueError(f"dt_lag_ms {doc['dt_lag_ms']!r} is not one of {DT_LAG_CANDIDATES_MS}")
    X = np.asarray(doc["X_true"], float)
    if X.shape != (4, 4) or not np.all(np.isfinite(X)) or not np.allclose(X[3], [0, 0, 0, 1]) \
            or not np.allclose(X[:3, :3].T @ X[:3, :3], np.eye(3), atol=1e-9):
        raise ValueError("X_true is not a 4x4 rigid transform")
    ft = doc["f_true"]
    if not isinstance(ft, dict) or set(ft) != {str(w) for w in WIDTHS}:
        raise ValueError(f"f_true must hold exactly the widths {WIDTHS}")
    for k, v in ft.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"f_true[{k}] is not a positive number")
    if not (isinstance(doc["code_tree"], str) and len(doc["code_tree"]) == 64):
        raise ValueError("code_tree is not a sha256")
    return doc


def read_poses_pick(R) -> dict:
    with open(Path(R) / "eval" / "poses_pick.json", "rb") as f:
        return check_poses_pick(json.loads(f.read().decode("utf-8")))


def captures_of(R, scenes) -> tuple:
    """Every wrist capture of the given scenes: [{scene, traj, token, views: {W: view}, fr, cam}] keyed on the
    smallest width present; plus the largest difference across the widths of one capture (T, E_rendered and the
    policy file's eef fields: one capture is one state, rendered at both widths)."""
    caps, worst, worst_obs = [], 0.0, 0.0
    for sid in scenes:
        fdir, cdir = Path(R) / "frames" / sid, Path(R) / "cam_truth" / sid
        views = sorted(set(list_view_ids(fdir)) & set(list_view_ids(cdir)))
        groups = {}
        for v in views:
            pv = parse_view(v)
            if pv["camera"] != WRIST:
                continue
            groups.setdefault((pv["traj"], pv["token"]), {})[pv["width"]] = v
        for (traj, tok), byw in sorted(groups.items()):
            ws = sorted(byw)
            cams = {w: read_cam(cdir / f"{byw[w]}.npz") for w in ws}
            frs = {w: read_policy(fdir / f"{byw[w]}.npz") for w in ws}
            for w in ws[1:]:
                for k in ("T_world_from_camera", "E_rendered"):
                    worst = max(worst, float(np.abs(cams[w][k] - cams[ws[0]][k]).max()))
                for k in ("robot0_eef_pos", "robot0_eef_quat", "prev_eef_pos", "prev_eef_quat"):
                    if k in frs[w] and k in frs[ws[0]]:
                        d = np.asarray(frs[w][k], float) - np.asarray(frs[ws[0]][k], float)
                        worst_obs = max(worst_obs, float(np.abs(d).max()) if d.size else 0.0)
            caps.append({"scene": sid, "traj": traj, "token": tok, "views": dict(byw), "fr": frs[ws[0]],
                         "cam": cams[ws[0]]})
    return caps, max(worst, worst_obs)


def x_true_numbers(caps) -> tuple:
    """(X_true, numbers): the median of E_rendered^-1 T over the captures and the largest deviation."""
    Xs = [inv_pose(c["cam"]["E_rendered"]) @ c["cam"]["T_world_from_camera"] for c in caps]
    X = median_pose(Xs)
    dm = dr = 0.0
    for Xc in Xs:
        m, r = pose_err(X, Xc)
        dm, dr = max(dm, m), max(dr, r)
    ok = bool(Xs) and dm <= X_TRUE_TOL_M and dr <= X_TRUE_TOL_RAD
    return X, {"check": "x_true", "pass": ok, "n_captures": len(Xs), "max_dev_m": dm, "max_dev_rad": dr,
               "tol_m": X_TRUE_TOL_M, "tol_rad": X_TRUE_TOL_RAD,
               "X_true_t_m": [float(v) for v in X[:3, 3]],
               "X_true_rotvec_rad": [float(v) for v in so3_log(X[:3, :3])]}


def lag_present_numbers(caps, order: str = EEF_QUAT_ORDER) -> dict:
    """capture.lag_present_check: on moving captures, the median obs-minus-rendered hand pose difference must be at
    least 0.1 mm or at least 1e-4 rad; also the largest gap to the render's own eef_lag numbers."""
    pm, pr, gap_mm, gap_rad, n_lag = [], [], 0.0, 0.0, 0
    for c in caps:
        if c["token"] not in MOVING_TOKENS:
            continue
        m, r = pose_err(obs_E(c["fr"], order), c["cam"]["E_rendered"])
        pm.append(m * 1000.0)
        pr.append(r)
        lag = c["cam"].get("eef_lag")
        if isinstance(lag, dict) and "pos_mm" in lag and "rot_rad" in lag:
            try:
                lm, lr = (float(np.linalg.norm(np.asarray(lag[k], float))) for k in ("pos_mm", "rot_rad"))
                gap_mm, gap_rad = max(gap_mm, abs(lm - m * 1000.0)), max(gap_rad, abs(lr - r))
                n_lag += 1
            except (TypeError, ValueError):
                gap_mm = gap_rad = float("inf")
    med_mm = float(np.median(pm)) if pm else None
    med_rad = float(np.median(pr)) if pr else None
    ok = bool(pm) and (med_mm >= LAG_PRESENT_MM_MIN or med_rad >= LAG_PRESENT_RAD_MIN)
    return {"check": "lag_present", "pass": ok, "n_moving": len(pm), "median_mm": med_mm, "median_rad": med_rad,
            "p90_mm": float(np.percentile(pm, 90)) if pm else None,
            "p90_rad": float(np.percentile(pr, 90)) if pr else None, "max_mm": max(pm) if pm else None,
            "min_mm_rule": LAG_PRESENT_MM_MIN, "min_rad_rule": LAG_PRESENT_RAD_MIN,
            "n_with_render_eef_lag": n_lag, "max_gap_to_render_eef_lag_mm": gap_mm,
            "max_gap_to_render_eef_lag_rad": gap_rad}


def dt_lag_pick(caps, X, order: str = EEF_QUAT_ORDER) -> tuple:
    """(dt_lag_ms, numbers): the lowest median camera-centre error of Ecorr X_true against the simulator T over every
    capture with a previous obs; within 0.02 mm the smaller dt wins."""
    use = [c for c in caps if obs_E_prev(c["fr"], order) is not None]
    med = {}
    for dt in DT_LAG_CANDIDATES_MS:
        errs = []
        for c in use:
            E, Ep = obs_E(c["fr"], order), obs_E_prev(c["fr"], order)
            T = ecorr(E, Ep, dt) @ X
            errs.append(pose_err(T, c["cam"]["T_world_from_camera"])[0] * 1000.0)
        med[dt] = float(np.median(errs)) if errs else None
    good = [dt for dt in DT_LAG_CANDIDATES_MS if med[dt] is not None]
    if not good:
        return None, {"check": "dt_lag_pick", "pass": False, "n_captures": 0, "median_mm": med}
    best = min(med[dt] for dt in good)
    pick = min(dt for dt in good if med[dt] <= best + DT_TIE_MM)
    return pick, {"check": "dt_lag_pick", "pass": True, "dt_lag_ms": pick, "n_captures": len(use),
                  "n_without_prev": len(caps) - len(use), "median_mm": {str(k): v for k, v in med.items()},
                  "tie_mm": DT_TIE_MM}


def f_true_numbers(R, scenes) -> tuple:
    """(f_true {"512", "1024"}, numbers): the cam_truth K of every tune wrist view, one value per width, principal
    point W/2, against fovy 75."""
    seen = {}
    bad = 0
    for sid in scenes:
        cdir = Path(R) / "cam_truth" / sid
        for v in list_view_ids(cdir):
            pv = parse_view(v)
            if pv["camera"] != WRIST:
                continue
            K = read_cam(cdir / f"{v}.npz")["K"]
            W = pv["width"]
            if abs(K[0, 2] - W / 2) > 1e-12 or abs(K[1, 2] - W / 2) > 1e-12 or K[0, 0] != K[1, 1]:
                bad += 1
            seen.setdefault(W, set()).add(float(K[0, 0]))
    f = {}
    ok = bad == 0
    for W in WIDTHS:
        vals = sorted(seen.get(W, ()))
        if len(vals) != 1 or abs(vals[0] - f_of_fovy(FOVY_TRUE_DEG, W)) > F_TRUE_TOL_PX:
            ok = False
        if vals:
            f[str(W)] = vals[0]
    return f, {"check": "f_true", "pass": ok, "f_true": f, "n_values": {str(k): len(v) for k, v in seen.items()},
               "formula_fovy75": {str(W): f_of_fovy(FOVY_TRUE_DEG, W) for W in WIDTHS}, "n_bad_K": bad}


def settled_pose_numbers(pairs, pick: dict, order: str = EEF_QUAT_ORDER, x_side: str = "right") -> dict:
    """Pipeline check 1 on (..., policy, cam) tuples: P-O (S3 formula) against P-T; pass when every view is within
    0.1 mm and 0.01 deg."""
    cm, rd = [], []
    for *_, fr, cam in pairs:
        r = po_view_T("S3", fr, pick, order, x_side)
        if r is None:
            cm.append(float("inf"))
            rd.append(float("inf"))
            continue
        m, a = pose_err(r[0], cam["T_world_from_camera"])
        cm.append(m * 1000.0)
        rd.append(math.degrees(a))
    ok = bool(cm) and max(cm) <= SETTLED_CENTRE_MM_MAX and max(rd) <= SETTLED_ROT_DEG_MAX
    return {"check": "settled_pose", "pass": ok, "n_views": len(cm), "centre_mm_max": max(cm) if cm else None,
            "centre_mm_median": float(np.median(cm)) if cm else None, "rot_deg_max": max(rd) if rd else None,
            "rot_deg_median": float(np.median(rd)) if rd else None, "centre_mm_rule": SETTLED_CENTRE_MM_MAX,
            "rot_deg_rule": SETTLED_ROT_DEG_MAX}


def settled(fr: dict) -> bool:
    jv = fr.get("robot0_joint_vel")
    if jv is None:
        return False
    jv = np.asarray(jv, float).reshape(-1)
    return bool(jv.size) and float(np.abs(jv).max()) < SETTLE_JOINT_VEL_MAX


# ===================================================================================================== CLI helpers
def _scene_dirs(R) -> list:
    d = Path(R) / "frames"
    try:
        return sorted(n for n in os.listdir(d) if (d / n).is_dir())
    except FileNotFoundError:
        return []


def _tune_scenes(R, only=None) -> tuple:
    keep, skipped = [], {}
    for sid in only or _scene_dirs(R):
        try:
            sp = AU.split_of(sid)
        except ValueError:
            skipped[sid] = "not a registered scene"
            continue
        if sp in ("tune", "smoke"):
            keep.append(sid)
        else:
            skipped[sid] = sp
    return keep, skipped


def _row(base: dict, prov: dict, pass_tag: str, process: str, R) -> dict:
    out = dict(base)
    out.update({"pass_tag": pass_tag, "process": process, "utc": AU.utc(), "code_tree": prov["code_tree"],
                "clean": bool(prov["clean"]), "root_kind": AU.root_kind(R, "poses")[0], "format": FORMAT})
    return out


def _print(d: dict) -> None:
    print(json.dumps(d, sort_keys=True, default=str), flush=True)


def cmd_pt(a) -> int:
    R = Path(a.root)
    AU.install(R, "pt", "P-T")
    prov = AU.provenance()
    c4ok, c4why = AU.c4_landed()
    scenes = a.scenes or _scene_dirs(R)
    out = {"cmd": "pt", "written": 0, "same": 0, "differs": [], "refused": {}, "scenes": 0, "looks": 0,
           "views_without_cam_truth": 0, "unknown_tokens": [], **prov}
    for sid in scenes:
        try:
            sp = AU.split_of(sid)
        except ValueError as e:
            out["refused"][sid] = str(e)
            continue
        if sp in ("score", "held_out") and not c4ok:
            out["refused"][sid] = f"{sp} scene before C4 ({c4why})"
            continue
        fviews = set(list_view_ids(R / "frames" / sid))
        cviews = set(list_view_ids(R / "cam_truth" / sid))
        out["views_without_cam_truth"] += len(fviews - cviews)
        both = sorted(fviews & cviews)
        for v in both:
            pv = parse_view(v)
            if pv["token"] not in KNOWN_TOKENS.get(pv["traj"], ()):
                out["unknown_tokens"].append(v)
        cams = {v: read_cam(R / "cam_truth" / sid / f"{v}.npz") for v in both}
        out["scenes"] += 1
        for look in LOOKS:
            for arm in PT_ARMS:
                doc = pt_look_doc(arm, look, cams, prov)
                if doc is None:
                    continue
                st = write_or_compare(pose_path(R, arm, sid, look), doc)
                if st == "differs":
                    out["differs"].append(f"{arm}/{sid}/{look}")
                else:
                    out[st] += 1
            out["looks"] += 1 if look_view_ids(look, cams)[0] else 0
    _print(out)
    if out["differs"]:
        return 1
    return 2 if (a.scenes and out["refused"]) else 0


def cmd_po(a) -> int:
    R = Path(a.root)
    arm = a.arm
    try:
        keys = [split_look_key(k) for k in a.looks]
        bad = [f"{k}: {po_arm_ok(arm, k[1])}" for k in keys if po_arm_ok(arm, k[1])]
    except ValueError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2
    if bad:
        print(f"REFUSE: {bad[:3]}", file=sys.stderr)
        return 2
    AU.install(R, "po", "P-O", arm=arm)
    got = AU.sha256_file(AU.HAND_MODEL)  # every P-O process checks the hand model's sha itself (methods.launcher)
    if got != a.hand_sha:
        print(f"REFUSE: hand model sha256 {got} != {a.hand_sha}", file=sys.stderr)
        return 2
    pick = read_poses_pick(R)
    out = {"cmd": "po", "arm": arm, "written": 0, "same": 0, "differs": [], "no_reference": [], "missing_views": 0,
           "pass_tag": a.pass_tag, "gen": a.gen, "method": a.method, "config": a.config, "width": a.width}
    for scene, look, _w in keys:
        fdir = R / "frames" / scene
        frames = {v: read_policy(fdir / f"{v}.npz") for v in list_view_ids(fdir)
                  if parse_view(v)["traj"] == LOOKS[look][0] and parse_view(v)["camera"] == WRIST}
        doc, missing = po_look_doc(arm, look, frames, pick, a.code_tree, bool(int(a.clean)))
        out["missing_views"] += len(missing)
        if doc is None:
            out["no_reference"].append(f"{scene}/{look}")
            continue
        st = write_or_compare(pose_path(R, arm, scene, look), doc)
        if st == "differs":
            out["differs"].append(f"{scene}/{look}")
        else:
            out[st] += 1
    _print(out)
    return 1 if out["differs"] else 0


def cmd_poses_pick(a) -> int:
    R = Path(a.root)
    AU.install(R, "poses-pick", "poses-pick")
    prov = AU.provenance()
    scenes, skipped = _tune_scenes(R)
    caps, wdiff = captures_of(R, scenes)
    rows = []
    lag = lag_present_numbers(caps)
    lag["n_scenes"] = len(scenes)
    rows.append(lag)
    X, xr = x_true_numbers(caps)
    xr["max_width_diff"] = wdiff
    xr["pass"] = xr["pass"] and wdiff <= 1e-12
    rows.append(xr)
    f, fr = f_true_numbers(R, scenes)
    rows.append(fr)
    dt, dr = dt_lag_pick(caps, X)
    rows.append(dr)
    for r in rows:
        AU.append_jsonl(R.joinpath(*PIPELINE), _row(r, prov, a.pass_tag, "poses-pick", R))
    ok = all(r["pass"] for r in rows)
    status = None
    if ok:
        doc = check_poses_pick({"dt_lag_ms": int(dt), "X_true": _m(X), "f_true": f, "code_tree": prov["code_tree"]})
        status = write_or_compare(R / "eval" / "poses_pick.json", doc)
    _print({"cmd": "poses-pick", "ok": ok, "poses_pick": status, "n_scenes": len(scenes),
            "skipped_scenes": len(skipped), "rows": rows, **prov})
    return 0 if ok and status != "differs" else 1


def settled_pairs(R, scenes) -> tuple:
    """(pairs, moving, counts): (scene, view, policy, cam) of tune rs00 views and settled E2 C views (both widths),
    and of the moving views (MOVING_TOKENS)."""
    pairs, n_rs, n_c, n_c_unsettled, moving = [], 0, 0, 0, []
    for sid in scenes:
        fdir, cdir = Path(R) / "frames" / sid, Path(R) / "cam_truth" / sid
        for v in sorted(set(list_view_ids(fdir)) & set(list_view_ids(cdir))):
            pv = parse_view(v)
            if pv["camera"] != WRIST:
                continue
            is_rs = pv["traj"] == "traj1" and pv["token"] == "rs00"
            is_c = pv["traj"] == "traj2" and pv["token"] == "C"
            if not (is_rs or is_c or pv["token"] in MOVING_TOKENS):
                continue
            fr, cam = read_policy(fdir / f"{v}.npz"), read_cam(cdir / f"{v}.npz")
            if is_rs:
                pairs.append((sid, v, fr, cam))
                n_rs += 1
            elif is_c:
                if settled(fr):
                    pairs.append((sid, v, fr, cam))
                    n_c += 1
                else:
                    n_c_unsettled += 1
            else:
                moving.append((sid, v, fr, cam))
    return pairs, moving, {"n_rs00": n_rs, "n_e2c_settled": n_c, "n_e2c_unsettled_skipped": n_c_unsettled}


def cmd_settled(a) -> int:
    R = Path(a.root)
    AU.install(R, "settled-pose", "settled-pose")
    prov = AU.provenance()
    pick = dict(read_poses_pick(R))
    src = "poses_pick.json"
    if a.dt_lag_ms is not None:
        pick["dt_lag_ms"] = int(a.dt_lag_ms)
        src = "override (diagnostic)"
    scenes, _ = _tune_scenes(R)
    pairs, moving, counts = settled_pairs(R, scenes)
    res = settled_pose_numbers(pairs, pick)
    res.update(counts)
    n_ident = 0
    mv_mm = []
    for _sid, v, fr, cam in moving:
        r = po_view_T("S3", fr, pick)
        if r is None:
            continue
        m = pose_err(r[0], cam["T_world_from_camera"])[0]
        mv_mm.append(m * 1000.0)
        n_ident += int(m <= IDENTICAL_CENTRE_M)
    res.update({"dt_lag_ms": pick["dt_lag_ms"], "dt_source": src, "n_scenes": len(scenes),
                "n_moving": len(mv_mm), "n_moving_identical_centres": n_ident,
                "moving_centre_mm_median": float(np.median(mv_mm)) if mv_mm else None,
                "moving_centre_mm_max": max(mv_mm) if mv_mm else None})
    # the written S3 pose files, when they exist, must hold exactly these rs00 poses
    n_file = n_file_bad = 0
    for sid, v, fr, cam in pairs:
        look = "traj1-rs" if parse_view(v)["traj"] == "traj1" else "traj2-E2"
        p = pose_path(R, "S3", sid, look)
        if src != "poses_pick.json" or not p.exists():
            continue
        with open(p, "rb") as f:
            doc = json.loads(f.read().decode("utf-8"))
        if v in doc["width_views"]:
            n_file += 1
            r = po_view_T("S3", fr, pick)
            n_file_bad += int(r is None or not np.array_equal(np.asarray(doc["width_views"][v]["T"]), r[0]))
    res.update({"n_pose_file_views": n_file, "n_pose_file_mismatch": n_file_bad})
    res["pass"] = bool(res["pass"]) and n_file_bad == 0
    AU.append_jsonl(R.joinpath(*PIPELINE), _row(res, prov, a.pass_tag, "settled-pose", R))
    _print({"cmd": "settled-pose", **res, **prov})
    return 0 if res["pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="K0-stereo poses: P-T, P-O, poses_pick.json, settled-pose check")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pt")
    p.add_argument("--root", required=True)
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("po")
    p.add_argument("--root", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--method", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--looks", nargs="+", required=True)
    p.add_argument("--pass-tag", required=True)
    p.add_argument("--gen", default=None)
    p.add_argument("--hand-sha", required=True)
    p.add_argument("--code-tree", required=True)
    p.add_argument("--clean", choices=["0", "1"], required=True)
    p.add_argument("--force", action="store_true")
    for name in ("poses-pick", "settled-pose"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True)
        p.add_argument("--pass-tag", required=True)
        p.add_argument("--force", action="store_true")
        if name == "settled-pose":
            p.add_argument("--dt-lag-ms", type=int, default=None, choices=list(DT_LAG_CANDIDATES_MS))
    a = ap.parse_args(argv)
    try:
        AU.check_root(a.root, "poses", force=a.force)
    except AU.RootRefused as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2
    if a.cmd == "pt":
        return cmd_pt(a)
    if a.cmd == "po":
        return cmd_po(a)
    if a.cmd == "poses-pick":
        return cmd_poses_pick(a)
    return cmd_settled(a)


if __name__ == "__main__":
    sys.exit(main())
