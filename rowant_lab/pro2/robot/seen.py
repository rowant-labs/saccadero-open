"""An object the robot has no stored model of, taken as it is seen (2026-11-16).

A full look reports the regions standing on the support that no stored model explains (the reply's
``extra["blobs"]``: the middle, half sizes and turn of the region's box across the table, its top and its height
above the support). This module makes such a region usable by the skills the robot already has:

* ``box_of``: the region as a box (half sizes, turn, middle);
* ``seen_type``: a model made on the spot from that box: the collision box, and grasp records built for it the way
  the exporter builds them for the stored types, entered in the episode's library under a name of its own;
* ``seen_percept``: a percept of that model where the region stands.

The pick-and-place agent then grasps it, carries it and sets it down like any other object: its grasps come from what
is seen, not from a stored model. A percept made here has no ``gid`` (the perception server cannot track it), so the
look tool predicts it: where it stood, or rigid with the hand while held.

A named object that no stored model explains (another make of the same thing, a reshaped or resized one) can be
taken the same way: ``seen_percept(..., named=typ)`` marks the percept as that type for the grounding
(``ground.is_type``), while every size the agent uses is the seen box's.

Policy side: numpy only; it reads the library object the runner made for this episode and adds to it.
"""

from __future__ import annotations

import math

import numpy as np

from . import grasp as GR
from rowant_lab.libero_env.m2_kin import mat_to_quat

PREFIX = "seen_"
MIN_HALF_M = 0.004  # a box thinner than this in any direction is not a thing to grasp
RECORD_PARAMS = GR.RecordParams(theta_step_deg=15.0, b_step_m=0.006, z_step_m=0.006)  # as the agent's replanning


def box_of(blob: dict, bottom_z: float | None = None, shrink_m: float = 0.0):
    """(half sizes (3,), R (3, 3), middle (3,)) of an unexplained region's box in the world. Across the table the box
    is the region's own (``half_xy`` along its long axis and across it, turned by ``yaw``), each side taken in by
    ``shrink_m`` (a look's region is a little larger than the thing: mixed pixels at its edge). Upward it runs from
    ``bottom_z`` (given when the thing stands on something else) or the support (``z_top - height``) to ``z_top``."""
    hx, hy = (max(float(v) - float(shrink_m), MIN_HALF_M) for v in blob["half_xy"])
    top = float(blob["z_top"])
    bottom = float(bottom_z) if bottom_z is not None else top - float(blob["height"])
    hz = max(0.5 * (top - bottom), MIN_HALF_M)
    yaw = float(blob.get("yaw") or 0.0)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    mid = np.array([float(blob["xy"][0]), float(blob["xy"][1]), top - hz])
    return np.array([hx, hy, hz]), R, mid


def type_name(half) -> str:
    """The made model's name: its half sizes in millimetres (the same box is the same model)."""
    return PREFIX + "x".join(str(int(round(1000.0 * float(v)))) for v in half)


def seen_type(lib, half, mass: float = 0.05) -> str:
    """Enter a box of ``half`` sizes (m) in the episode's library as a type of its own and return its name: M1's
    collision model, the export's kinematics (one free body, no site, no part that opens) and its grasp records.
    Entering the same box again changes nothing."""
    half = [round(max(float(v), MIN_HALF_M), 3) for v in half]
    name = type_name(half)
    if name in lib.types:
        return name
    prim = {"R": np.eye(3).tolist(), "centre": [0.0, 0.0, 0.0], "exact": False, "geom": "seen", "half": half,
            "type": "box"}
    lib.types[name] = {"bodies": {"main": {"com": [0.0, 0.0, 0.0], "is_root": True, "mass": float(mass),
                                           "primitives": [prim]}},
                       "kind": "object", "rest_up_in_body": [0.0, 0.0, 1.0], "root_body": "main",
                       "subtree_mass": float(mass), "type": name, "seen": True}
    lib.export_types[name] = {
        "bodies": {"main": {"body_pos": [0.0, 0.0, 0.0], "body_quat": [1.0, 0.0, 0.0, 0.0], "geoms": [],
                            "is_root": True, "joints": [], "parent": None}},
        "fixed_joints": {}, "kind": "object", "part_of": {"main": "main"},
        "parts": {"main": {"bodies": ["main"], "joint": None, "parent_body": None}},
        "pose_in_part": {"main": {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]}},
        "root_body": "main", "sites": {}, "type": name, "seen": True}
    lib.records[name] = GR.build_records([prim], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], lib.jaw, RECORD_PARAMS, name=name)
    return name


def n_grasps(lib, name: str) -> int:
    """How many grasp records the made model has (0: nothing of it fits between the open fingers)."""
    return int((lib.records.get(name) or {}).get("n_records") or 0)


def seen_percept(lib, blob: dict, pid: str, bottom_z: float | None = None, shrink_m: float = 0.0,
                 named: str | None = None) -> dict:
    """A percept of the region as a made model (``seen_type``), in the reply's percept format. ``named``: the stored
    type this thing is taken to be (the grounding reads it; the agent's sizes stay the seen box's)."""
    half, R, mid = box_of(blob, bottom_z, shrink_m)
    name = seen_type(lib, half)
    p = {"id": pid, "type": name, "gid": None, "score": 0.0, "n_px": int(blob.get("n_px") or 0),
         "fit": {"n_agree": 0, "inlier": 0.0}, "source": "seen", "joints": {},
         "bodies": {"main": {"pos": [float(x) for x in mid], "quat_wxyz": mat_to_quat(R)}}}
    if named:
        p["named"] = str(named)
    return p


def blob_near(blobs, xy, max_m: float, min_px: int = 300, min_height_m: float = 0.01):
    """(index, blob) of the unexplained region whose middle is nearest ``xy`` within ``max_m``, or (None, None)."""
    best, key = None, None
    for k, b in enumerate(blobs or []):
        if int(b.get("n_px") or 0) < min_px or float(b.get("height") or 0.0) < min_height_m:
            continue
        d = float(np.linalg.norm(np.asarray(b["xy"], float) - np.asarray(xy, float)[:2]))
        if d <= max_m and (key is None or d < key):
            best, key = k, d
    return (best, blobs[best]) if best is not None else (None, None)
