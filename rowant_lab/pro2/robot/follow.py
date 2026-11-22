"""M1 waypoint follower on LIBERO's own OSC_POSE controller (plan agent_design.motion; numpy only).

robosuite 1.4.0 OSC_POSE with LIBERO's defaults: each policy step the 7-D action in [-1, 1] sets the hand's goal to
its current pose plus a delta of up to 0.05 m and 0.5 rad (axis-angle, world frame: R_goal = R_delta R), and the
controller tracks it for 25 x 2 ms with kp 150 and critical damping. The gripper action is a sign: -1 opens, +1
closes (the finger command ramps 0.01 per physics substep). No controller parameter is changed.

The follower commands ``a_pos = clip(gain_pos (p* - p) / 0.05)`` and ``a_rot = clip(gain_rot rotvec(R* R^T) / 0.5)``,
each then scaled so its norm is at most the phase's cap (a fraction of the full command). Measured on DEV state 0
(2026-10-22, scratch): a saturated command moves the hand about 17 mm per step and a constant 1 cm command about
2.6 mm per step, so caps are fractions of the command, not of the motion (amendment A2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grasp import rotvec_from_mat

OUT_POS = 0.05
OUT_ROT = 0.5


@dataclass
class FollowParams:
    gain_pos: float = 2.0
    gain_rot: float = 2.0


def _cap(v: np.ndarray, cap: float) -> np.ndarray:
    v = np.clip(v, -1.0, 1.0)
    n = float(np.linalg.norm(v))
    return v * (cap / n) if n > cap > 0 else v


def _cap_box(v: np.ndarray, cap: float) -> np.ndarray:
    """``v`` scaled, direction kept, until its largest component is at most ``cap`` (the controller takes each axis
    up to 1 on its own, so a slanted move may ask for more than a move along one axis: up to 1.7 times)."""
    m = float(np.max(np.abs(v))) if len(v) else 0.0
    return v * (cap / m) if m > cap > 0 else v


def follow(p, R, p_tgt, R_tgt, grip: float, cap_pos: float = 1.0, cap_rot: float = 1.0,
           fp: FollowParams = FollowParams(), box: bool = False, box_rot: bool = False,
           each: bool = False) -> list[float]:
    """One OSC_POSE action toward the target pose (positions in m, rotations as 3x3 world matrices). ``box``: the
    position command is capped by its largest component instead of its length (see ``_cap_box``). ``box_rot``: the
    same for the rotation command (a turn about a slanted axis then asks each axis for up to its own full step).
    ``each`` (before ``box``): every axis of the position command is capped on its own, so a small error across
    a long move is still corrected in full (``box`` keeps the direction: 2.6 cm aside on a 15 cm move down asks
    for a sixth of a step aside)."""
    e = np.asarray(p_tgt, dtype=float) - np.asarray(p, dtype=float)
    if each:
        a_pos = np.clip(fp.gain_pos * e / OUT_POS, -abs(cap_pos), abs(cap_pos))
    else:
        a_pos = _cap_box(fp.gain_pos * e / OUT_POS, cap_pos) if box else _cap(fp.gain_pos * e / OUT_POS, cap_pos)
    r = rotvec_from_mat(np.asarray(R_tgt, dtype=float) @ np.asarray(R, dtype=float).T)
    if R_tgt is None:
        a_rot = np.zeros(3)
    else:
        a_rot = _cap_box(fp.gain_rot * r / OUT_ROT, cap_rot) if box_rot else _cap(fp.gain_rot * r / OUT_ROT, cap_rot)
    return [float(x) for x in a_pos] + [float(x) for x in a_rot] + [float(np.sign(grip) or -1.0)]


def pose_error(p, R, p_tgt, R_tgt) -> tuple[float, float]:
    """(position error m, rotation error deg)."""
    e = float(np.linalg.norm(np.asarray(p_tgt, dtype=float) - np.asarray(p, dtype=float)))
    r = float(np.degrees(np.linalg.norm(rotvec_from_mat(np.asarray(R_tgt) @ np.asarray(R).T))))
    return e, r
