"""The wrist camera's frames for the in-episode learner when the wrist gives colour and no depth (2026-11-17).

Runner side, policy side in what it holds: the wrist pictures the arm passes anyway and the hand's readings of the
same steps. Off unless ``params["learn_wrist_stereo"]`` is given (a dict of options; ``{}`` = ``DEFAULTS``), with
``params["learn_in_episode"]``.

``WristFrames`` keeps the last ``buffer`` wrist pictures, each with the camera pose that follows from the hand's
reading and the stored wrist mount (``robot/wrist2d.WRIST``; K0-stereo's P-O rule: the reading advanced by 2 ms of
the last step's motion). ``frame()`` is the newest picture as a camera frame WITHOUT depth (``depth_u16`` all zero =
no reading) and with its ``partners``: up to 3 earlier pictures, the widest within 1 to 4 cm to the side of its line
of sight, 1 cm apart (K0's ride-on-motion rule with its upper limit at 4 cm: with
K0's 8 cm the matcher refuses half of an object's pixels). The perception server turns such a frame into depth
(``libero_env/lfb_episode.wrist_depth``: K0's plane sweep, the fingers left out, unread pixels left as no reading).

On the DEPTH line this is a development option: the runner renders the wrist picture itself every step while the
robot is looking and the simulator's wrist depth is withheld, so the learner works from the stereo's depth. The
colour line will feed the same buffer from the episode's own wrist images.
"""

from __future__ import annotations

import collections

import numpy as np

WRIST = "robot0_eye_in_hand"
DEPTH_SCALE_MM = 0.1
DEFAULTS = {"width": 1024, "buffer": 12, "b_min_m": 0.01, "b_max_m": 0.04, "sep_m": 0.01, "partners": 3,
            "dt_lag_ms": 2.0}


def mount_X(wrist: dict | None = None) -> np.ndarray:
    """Hand-from-camera, OpenCV axes (x right, y down, z forward), from the stored wrist mount (``wrist2d.WRIST``,
    which is written in OpenGL axes): ``T_world_from_camera = E @ X`` with E the hand's reading."""
    if wrist is None:
        from rowant_lab.pro2.robot import wrist2d as W2D

        wrist = W2D.WRIST
    X = np.eye(4)
    X[:3, :3] = np.asarray(wrist["R_hand_cam"], float) @ np.diag([1.0, -1.0, -1.0])
    X[:3, 3] = np.asarray(wrist["t_hand_cam"], float)
    return X


def reading_pose(E, E_prev, X, dt_lag_ms: float) -> np.ndarray:
    """The wrist camera's pose from the hand's reading ``E`` (4x4), K0's P-O rule: advanced by ``dt_lag_ms`` of the
    motion since the previous reading ``E_prev`` (None, or no lag: the reading itself)."""
    from rowant_lab.libero_env import m3_stereo1_poses as PO

    E = np.asarray(E, float)
    if E_prev is None or not float(dt_lag_ms):
        return E @ X
    return PO.ecorr(E, np.asarray(E_prev, float), float(dt_lag_ms)) @ X


def b_eff(T_ref, T_p) -> float:
    """A partner's distance to the side of the reference's line of sight (``m3_stereo1_match.b_eff``)."""
    T_ref = np.asarray(T_ref, float)
    d = np.asarray(T_p, float)[:3, 3] - T_ref[:3, 3]
    z = T_ref[:3, 2] / np.linalg.norm(T_ref[:3, 2])
    return float(np.linalg.norm(d - float(d @ z) * z))


def pick_partners(T_ref, Ts: list, b_min: float = 0.01, b_max: float = 0.08, sep: float = 0.01, n: int = 3) -> list:
    """K0's ride-on-motion rule (``m3_stereo1_match.rm_partners``) with its numbers as arguments: up to ``n`` views,
    greedily by the largest sideways distance within [b_min, b_max], each at least ``sep`` from those chosen."""
    cand = sorted(((b_eff(T_ref, T), i) for i, T in enumerate(Ts)), key=lambda x: (-x[0], x[1]))
    out: list = []
    for b_, i in cand:
        if not (b_min <= b_ <= b_max):
            continue
        c = np.asarray(Ts[i], float)[:3, 3]
        if all(float(np.linalg.norm(c - np.asarray(Ts[j], float)[:3, 3])) >= sep for j in out):
            out.append(i)
        if len(out) == n:
            break
    return sorted(out)


class WristFrames:
    def __init__(self, opts: dict | None = None):
        from rowant_lab.libero_env import m3_stereo1_poses as PO
        from rowant_lab.pro2.robot import wrist2d as W2D

        self.o = {**DEFAULTS, **(opts or {})}
        self.W = int(self.o["width"])
        self.PO = PO
        self.X = mount_X()
        self.K = PO.K_true(PO.f_of_fovy(float(W2D.WRIST["fovy_deg"]), self.W), self.W)
        self.buf: collections.deque = collections.deque(maxlen=int(self.o["buffer"]))
        self.E_prev = None
        self.n_pushed = 0

    @staticmethod
    def wanted(agent) -> bool:
        """Pictures are kept only while the robot is looking: the learner, before its own agent takes over."""
        return type(agent).__name__ == "LearnThenAct" and getattr(agent, "inner", None) is None

    def push(self, t: int, rgb, obs: dict) -> None:
        """The wrist picture after step ``t`` (uint8, W x W x 3, row 0 at the top) and that step's readings."""
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if rgb.shape != (self.W, self.W, 3):
            raise ValueError(f"the wrist picture is {rgb.shape}, the stereo works at {self.W} px")
        E = self.PO.hand_pose(obs["robot0_eef_pos"], obs["robot0_eef_quat"])
        self.buf.append({"t": int(t), "rgb": rgb, "T": reading_pose(E, self.E_prev, self.X, float(self.o["dt_lag_ms"])),
                         "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], float).copy()})
        self.E_prev = E
        self.n_pushed += 1

    def clear(self) -> None:
        self.buf.clear()

    def frame(self) -> dict:
        """The newest picture as a frame with no depth, and its partners among the pictures before it."""
        if not self.buf:
            raise RuntimeError("no wrist picture yet")
        o, ref = self.o, self.buf[-1]
        cands = list(self.buf)[:-1]
        idx = pick_partners(ref["T"], [c["T"] for c in cands], float(o["b_min_m"]), float(o["b_max_m"]),
                            float(o["sep_m"]), int(o["partners"]))
        parts = [{"rgb": cands[i]["rgb"], "K": self.K.copy(), "T_world_from_camera": cands[i]["T"].copy(),
                  "gripper_qpos": cands[i]["gripper_qpos"].copy(), "steps_back": int(ref["t"] - cands[i]["t"]),
                  "b_eff_m": round(b_eff(ref["T"], cands[i]["T"]), 4)} for i in idx]
        return {"camera": WRIST, "rgb": ref["rgb"], "depth_u16": np.zeros((self.W, self.W), np.uint16),
                "depth_scale_mm": DEPTH_SCALE_MM, "K": self.K.copy(), "T_world_from_camera": ref["T"].copy(),
                "width": self.W, "height": self.W, "step": int(ref["t"]), "X": self.X.copy(),
                "gripper_qpos": ref["gripper_qpos"].copy(), "partners": parts}
