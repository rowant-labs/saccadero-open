"""Tests of the perception server's facing prior for a jointed fixture (2026-11-16; ``m2_segment`` ``twin_prior``).

Synthetic type and workcell documents only: no library, no simulator, no image. The search-side tests (the twin's
own fit and the step that takes it) are in ``test_m2_segment.py``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from rowant_lab.libero_env import m2_perceive as MP
from rowant_lab.libero_env import m2_segment as SEG

TDOC = {"root_body": "main", "kind": "fixture",
        "bodies": {"main": {"parent": None},
                   "top": {"parent": "main", "body_quat": [1.0, 0.0, 0.0, 0.0], "body_pos": [0.0, -0.05, 0.1]},
                   "low": {"parent": "main", "body_quat": [1.0, 0.0, 0.0, 0.0], "body_pos": [0.0, -0.05, 0.02]}},
        "part_of": {"main": "main", "top": "top", "low": "low"}, "pose_in_part": {},
        # as LIBERO's cabinets: the drawers open along the root's -y as their joints go to -0.16
        "parts": {"main": {"joint": None},
                  "top": {"joint": {"type": "slide", "axis": [0.0, 1.0, 0.0], "range": [-0.16, 0.01], "qpos0": 0.0}},
                  "low": {"joint": {"type": "slide", "axis": [0.0, 1.0, 0.0], "range": [-0.16, 0.01], "qpos0": 0.0}}}}
TABLE = [{"type": "plane"}, {"type": "box", "centre": [0.0, 0.0, 0.875], "half": [0.5, 0.6, 0.025]}]
T = np.eye(4)  # a camera at the world's origin and axes: the camera frame is the world frame here


def _rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _server(arena=TABLE, tdoc=TDOC):
    per = object.__new__(MP.Perceiver)
    per.p = dict(MP.PARAMS)
    nodes = np.array([[x, y, 0.0] for x in (-0.1, 0.1) for y in (-0.08, 0.12)])  # footprint middle 2 cm off the origin
    wc = SimpleNamespace(part_of_gid={7: ("cupboard", "main"), 8: ("cupboard", "top"), 9: ("cupboard", "low")},
                         types={"cupboard": tdoc}, rest_up={"7": np.array([0.0, 0.0, 1.0])},
                         nodes={"7": (nodes, np.tile([0.0, 0.0, 1.0], (4, 1)))},
                         base=np.array([-0.66, 0.0, 0.912]), workcell={"arena": arena})
    return per, wc


def test_the_option_is_off_by_default():
    assert SEG.PARAMS["twin_prior"] is None
    assert {**SEG.PARAMS, **(MP.PARAMS.get("support") or {})}["twin_prior"] is None


def test_open_dir_is_the_way_the_slide_parts_open():
    per, wc = _server()
    assert np.allclose(per.open_dir(wc, "7"), [0.0, -1.0, 0.0])
    assert per.open_dir(wc, "8") is None  # not the root part
    one = {**TDOC, "parts": {**TDOC["parts"], "low": {"joint": {
        "type": "slide", "axis": [0.0, -1.0, 0.0], "range": [-0.16, 0.01], "qpos0": 0.0}}}}
    per, wc = _server(tdoc=one)
    assert per.open_dir(wc, "7") is None  # parts that open in opposed directions give no direction
    hinge = {**TDOC, "parts": {"main": {"joint": None}, "top": {"joint": {
        "type": "hinge", "axis": [0.0, 0.0, 1.0], "pos": [0.0, 0.0, 0.0], "range": [-2.0, 0.0], "qpos0": 0.0}},
        "low": {"joint": None}}}
    per, wc = _server(tdoc=hinge)
    assert per.open_dir(wc, "7") is None  # a door on a hinge: no slide part


def test_inward_says_which_way_round_and_abstains_when_unsure():
    per, wc = _server()
    segp = {**SEG.PARAMS, "twin_prior": {"inward": {}}}
    here = np.array([0.02, -0.23, 0.905])  # at the robot's right, as the Goal scene's cabinet
    # the point ahead of the base is (-0.21, 0), the table's middle (0, 0): a cabinet turned 180 deg opens to +y,
    # toward both
    good = per.twin_prior(wc, "libero_goal", "7", _rz(180.0), here, T, segp)
    bad = per.twin_prior(wc, "libero_goal", "7", _rz(0.0), here, T, segp)
    assert good > 0.9 and bad < -0.9
    # the Spatial scene's cabinet (turned 154 deg, a little further out): the table's middle answers, clearly
    there = np.array([0.03, -0.27, 0.905])
    assert per.twin_prior(wc, "libero_spatial", "7", _rz(154.3), there, T, segp) > 0.8
    assert per.twin_prior(wc, "libero_spatial", "7", _rz(154.3 - 180.0), there, T, segp) < -0.8
    # a fixture standing between the two points: they disagree, no verdict
    assert per.twin_prior(wc, "libero_goal", "7", _rz(90.0), np.array([-0.10, 0.0, 0.905]), T, segp) is None
    # opening across both directions: no verdict
    weak = {**SEG.PARAMS, "twin_prior": {"inward": {"min_cos": 0.99}}}
    assert per.twin_prior(wc, "libero_goal", "7", _rz(135.0), here, T, weak) is None
    # a workcell without a table box: the base itself is the point
    per2, wc2 = _server(arena=[{"type": "plane"}])
    assert per2.twin_prior(wc2, "libero_goal", "7", _rz(-90.0), np.array([0.3, 0.0, 0.9]), T, segp) > 0.9
    assert per2.twin_prior(wc2, "libero_goal", "7", _rz(90.0), np.array([0.3, 0.0, 0.9]), T, segp) < -0.9
    # listed workcells only, when a list is given
    only = {**SEG.PARAMS, "twin_prior": {"inward": {"suites": ["libero_goal"]}}}
    assert per.twin_prior(wc, "libero_goal", "7", _rz(0.0), here, T, only) < -0.6
    assert per.twin_prior(wc, "libero_long_kitchen", "7", _rz(0.0), here, T, only) is None
    # off: nothing is said about any pose
    assert per.twin_prior(wc, "libero_goal", "7", _rz(0.0), here, T, {**SEG.PARAMS, "twin_prior": {}}) is None


def test_memory_speaks_of_the_turn_only_and_a_split_is_no_verdict():
    per, wc = _server()
    mem = {"memory": {"libero_goal": {"cupboard": 180.0}}}
    segp = {**SEG.PARAMS, "twin_prior": mem}
    far = np.array([0.3, 0.35, 0.905])  # the fixture moved: the remembered turn still counts
    assert per.twin_prior(wc, "libero_goal", "7", _rz(176.0), far, T, segp) == 1.0
    assert per.twin_prior(wc, "libero_goal", "7", _rz(-3.0), far, T, segp) == -1.0
    assert per.twin_prior(wc, "libero_goal", "7", _rz(90.0), far, T, segp) is None  # a quarter turn: unknown
    assert per.twin_prior(wc, "libero_spatial", "7", _rz(0.0), far, T, segp) is None  # nothing remembered there
    both = {**SEG.PARAMS, "twin_prior": {**mem, "inward": {}}}
    here = np.array([0.02, -0.23, 0.905])
    assert per.twin_prior(wc, "libero_goal", "7", _rz(0.0), here, T, both) < 0  # both call it back to front
    assert per.twin_prior(wc, "libero_goal", "7", _rz(180.0), here, T, both) > 0
    # moved across the robot's middle line and not turned: memory says right, inward says wrong -> no verdict
    across = np.array([0.02, 0.30, 0.905])
    assert per.twin_prior(wc, "libero_goal", "7", _rz(180.0), across, T, both) is None
    assert per.twin_prior(wc, "libero_goal", "7", _rz(0.0), across, T, both) is None


def test_memory_seeds_are_the_remembered_poses_in_the_looks_plane_frame():
    per, wc = _server()
    # an oblique camera over a table at z = 0.9
    cpos = np.array([0.75, 0.0, 1.55])
    fwd = np.array([0.0, 0.0, 0.9]) - cpos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    Tc = np.eye(4)
    Tc[:3, :3], Tc[:3, 3] = np.stack([right, np.cross(fwd, right), fwd], 1), cpos
    n_c = Tc[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    p_c = Tc[:3, :3].T @ (np.array([0.0, 0.0, 0.9]) - cpos)
    pf = SEG.plane_frame({"normal": n_c, "d": float(-n_c @ p_c)}, Tc)
    mem = {"libero_goal": {"cupboard": [{"pos": [0.02, -0.23, 0.9], "yaw_deg": 180.0},
                                        {"pos": [0.30, 0.10, 0.9], "yaw_deg": 35.0}]}}
    segp = {**SEG.PARAMS, "memory": mem}
    seeds = per._memory_seeds(wc, "libero_goal", "7", Tc, pf, segp)
    assert len(seeds) == 2
    for (yaw, a, b), m in zip(seeds, mem["libero_goal"]["cupboard"]):
        R_c, t_c = SEG.pose_at(pf, a, b, yaw, 0.0, 0.0)  # an origin on the support
        Rw, tw = Tc[:3, :3] @ R_c, Tc[:3, :3] @ t_c + Tc[:3, 3]
        assert np.allclose(tw, m["pos"], atol=1e-9)
        assert abs((math.degrees(math.atan2(Rw[1, 0], Rw[0, 0])) - m["yaw_deg"] + 180.0) % 360.0 - 180.0) < 1e-6
    assert per._memory_seeds(wc, "libero_spatial", "7", Tc, pf, segp) == []  # nothing remembered in that workcell
    assert per._memory_seeds(wc, "libero_goal", "8", Tc, pf, segp) == []  # not the root part
    assert {**SEG.PARAMS, **(MP.PARAMS.get("support") or {})}["memory"] is None  # off by default
