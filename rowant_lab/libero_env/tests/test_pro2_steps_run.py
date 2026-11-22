"""Steps per attempt, run whole (2026-11-16): the agent on a kinematic stand-in for the simulator, with the new options
off and on. The stand-in moves the hand 12.5 mm per unit of command and axis, the fingers 1 cm of gap a step; a bowl
between closing fingers stops them and rides with the hand; let go, it drops straight down."""

from dataclasses import replace

import numpy as np
import pytest

from rowant_lab.libero_env import m2_paths as MP
from rowant_lab.libero_env.m2_kin import mat_to_quat
from rowant_lab.pro2.robot import agent as AG
from rowant_lab.pro2.robot import grasp as GR

LIB_ROOT = MP.POLICY_LIB  # the built object library (its policy folder), under the data folder of this machine
pytestmark = pytest.mark.skipif(not LIB_ROOT.is_dir(), reason="the object library is not built on this machine")

BOWL0, PLATE = np.array([0.05, 0.05, 0.898]), np.array([-0.10, 0.22, 0.903])
FAST = dict(transit_cut=True, lift_fast=True, close_early_m=0.025, close_window=2, brake_k=1.5, cap_box=True,
            lower_cap=1.0, lower_tol_m=0.006)


def _rot(rv):
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(rv, float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class World:
    def __init__(self, lib, knock_at=None):
        self.lib, self.knock_at = lib, knock_at
        self.p, self.R = np.array([-0.10, 0.0, 1.15]), GR.hand_matrix([0.0, 1.0, 0.0])
        self.cmd, self.bowl, self.held, self.t = -1.0, BOWL0.copy(), None, 0
        self.path, self.dropped_at = [], None  # path: (hand position, hand matrix, the held bowl's origin or None)

    def gap(self):
        q = 0.02 * (1.0 - self.cmd)
        return max(q, 0.004 if self.held is not None else 0.0)  # (per finger; a held rim keeps them 8 mm apart)

    def pobs(self):
        w, x, y, z = mat_to_quat(self.R)
        q = self.gap()
        return {"robot0_eef_pos": self.p.copy(), "robot0_eef_quat": np.array([x, y, z, w]),
                "robot0_gripper_qpos": np.array([q, -q])}

    def step(self, a):
        a = np.asarray(a, float)
        assert np.all(np.abs(a) <= 1.0 + 1e-9)
        self.p = self.p + 0.0125 * a[:3]
        self.R = _rot(0.09 * a[3:6]) @ self.R
        self.cmd = float(np.clip(self.cmd + 0.25 * np.sign(a[6]), -1.0, 1.0))
        self.t += 1
        rel = self.p - self.bowl
        if self.held is None and self.cmd > 0.55 and float(np.linalg.norm(rel[:2])) < 0.07 and 0.0 < rel[2] < 0.07 \
                and self.dropped_at != "gone":
            self.held = -rel  # the bowl's origin in the hand's frame (no turn: the stand-in keeps it upright)
        if self.held is not None and (self.cmd < 0.3 or self.t == self.knock_at):
            self.held = None  # let go (or knocked out of the hand): straight down
            on_plate = float(np.linalg.norm((self.bowl - PLATE)[:2])) < 0.07
            self.bowl = np.array([self.bowl[0], self.bowl[1], PLATE[2] + 0.014 if on_plate else BOWL0[2]])
            self.dropped_at = self.t
        if self.held is not None:
            self.bowl = self.p + self.held
        self.path.append((self.p.copy(), self.R.copy(), self.bowl.copy() if self.held is not None else None))


class Tool:
    """The truth as poses (M1's oracle interface), with the look tool's ``hand_opened`` recorded."""

    def __init__(self, lib, world, extra=()):
        self.lib, self.world, self.calls, self.t_now, self.told = lib, world, 0, 0, []
        self.target_missing, self.extra = False, list(extra)

    def percepts(self):
        q = [1.0, 0.0, 0.0, 0.0]
        return [{"id": "p0", "type": "akita_black_bowl", "bodies": {"main": {"pos": list(self.world.bowl), "quat_wxyz": q}}},
                {"id": "p1", "type": "plate", "bodies": {"main": {"pos": list(PLATE), "quat_wxyz": q}}}] + self.extra

    def hand_opened(self, lost=False):
        self.told.append(bool(lost))

    def __call__(self):
        self.calls += 1
        return self.lib.poses_from_percepts(self.percepts())


def _run(cap=220, knock_at=None, extra=(), **opts):
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.pro2.robot import make_params

    lib = LB.M2Library("libero_spatial")
    world = World(lib, knock_at)
    tool = Tool(lib, world, extra)
    ap, pp, fp = make_params(None)
    ap = replace(ap, **opts)
    spec = {"predicate": "on", "target": "p0", "place": "p1", "place_kind": "object", "place_instance": "p1",
            "instruction": "pick the akita black bowl and place it on the plate"}
    agent = AG.OracleAgent(lib, lib.scene_from_percepts(tool.percepts()), spec, tool, cap, ap, pp, fp)
    done_at = None
    for t in range(cap):
        world.step(agent.act(world.pobs()))
        placed = world.held is None and float(np.linalg.norm((world.bowl - PLATE)[:2])) < 0.025 and world.bowl[2] > 0.91
        if placed and done_at is None:
            done_at = t + 1  # (the benchmark ends an episode here)
        if agent.finished:
            break
    return agent, world, tool, done_at


def test_a_pick_and_place_runs_whole_with_the_options_off_and_on_and_is_shorter_with_them():
    slow, w0, _t, done0 = _run()
    fast, w1, _t, done1 = _run(**FAST)
    for ag, done in ((slow, done0), (fast, done1)):
        assert ag.finish_reason == "placed" and len(ag.attempts) == 1 and done is not None, ag.summary()["events"][-6:]
        a = ag.attempts[0]
        assert a["c1"]["pass"] and a["c2"]["pass"] and a["c3"]["pass"]
    assert done1 < done0, (done0, done1)  # (this stand-in has no inertia; the simulator's gain is larger)
    # the same grasp and the same release point either way
    assert slow.attempts[0]["frame"]["record"] == fast.attempts[0]["frame"]["record"]
    assert abs(slow.attempts[0]["place"]["z_release"] - fast.attempts[0]["place"]["z_release"]) < 0.003
    # every option alone runs whole too
    for name, value in FAST.items():
        ag, _w, _t, done = _run(**{name: value})
        assert ag.finish_reason == "placed" and done is not None, name


def test_a_bowl_knocked_out_of_the_hand_is_taken_again_and_the_tool_is_told():
    ag, world, tool, done = _run(cap=300, knock_at=55, carry_watch=True, tell_open=True, lost_looks=2)
    outs = [a.get("outcome") for a in ag.attempts]
    assert outs[0] == "lost_in_carry" and ag.attempts[0]["lost_look"] == {"looks": 1, "found": True}
    assert tool.told == [True] and done is not None and outs[-1] == "placed", (outs, ag.summary()["events"][-8:])
    # the second grasp is where the bowl fell, on the table, not where the hand was
    assert abs(ag.attempts[1]["frame"]["centre"][2] - ag.attempts[0]["frame"]["centre"][2]) < 0.01


# a box standing on its long edge between the bowl and the plate, 8 cm tall: the carried bowl has to go over it
WALL = {"id": "p2", "type": "cookies", "bodies": {"main": {"pos": [-0.025, 0.135, 0.941],
                                                           "quat_wxyz": [0.7071068, 0.7071068, 0.0, 0.0]}}}


def _least_clearance(agent, world, tool):
    """How far the carried bowl and the hand stayed from the box in the way, over the whole carry (m; under 0: hit)."""
    poses = tool.lib.poses_from_percepts(tool.percepts())
    box = agent._instance_boxes(poses, "p2")
    worst = None
    for p, Rh, bowl in world.path:
        if bowl is None:
            continue
        held = (Rh.T @ np.eye(3), Rh.T @ (bowl - p))
        swept = agent._swept_boxes(p, Rh, 0.004, held)
        lo, step = 0.0, 0.002
        while lo < 0.06 and not GR.any_overlap(swept, box.inflated(lo + step)):
            lo += step
        worst = lo if worst is None else min(worst, lo)
    return worst


def test_over_something_in_the_way_the_cut_transit_is_shorter_and_keeps_its_distance():
    old, w0, t0, done0 = _run(extra=[WALL])
    new, w1, t1, done1 = _run(extra=[WALL], transit_cut=True)
    for ag, done in ((old, done0), (new, done1)):
        assert ag.finish_reason == "placed" and done is not None and len(ag.attempts) == 1
    cross = [e for e in old.events if e["ev"] == "transit" and "z_cross" in e and e["t"] > old.attempts[0]["t_lift"]]
    assert cross, "the box does not block the straight carry: the test's scene is wrong"
    cut = [e for e in new.events if e["ev"] == "transit" and "cut" in e and e["t"] > new.attempts[0]["t_lift"]]
    assert cut and done1 < done0, (done0, done1)
    # the planner asks for 2 cm (``transit_margin_m``) and a corner may be passed 1 cm short of it (the transit
    # tolerance): the hand on the cut way keeps what the old way keeps
    c_old, c_new = _least_clearance(old, w0, t0), _least_clearance(new, w1, t1)
    assert c_old >= 0.008 and c_new >= c_old - 1e-9, (c_old, c_new)
    # and so with everything on
    fast, w2, t2, done2 = _run(extra=[WALL], **FAST)
    assert fast.finish_reason == "placed" and done2 < done1 and _least_clearance(fast, w2, t2) >= c_old - 1e-9


def test_a_staging_release_can_end_its_transport_just_above_the_release_point():
    old, w0, _t, _d = _run(trust_place=True)
    new, w1, _t, _d = _run(trust_place=True, stage_above_m=0.015)
    for ag, w in ((old, w0), (new, w1)):
        a = ag.attempts[0]
        assert ag.finish_reason == "placed" and a["c3"].get("trusted") and w.dropped_at is not None
        assert float(np.linalg.norm((w.bowl - PLATE)[:2])) < 0.03  # set down on the plate either way
    # the hand no longer goes 5 cm above the release and down again: the release comes sooner, from the same height
    assert new.attempts[0]["t_release"] < old.attempts[0]["t_release"]
    assert abs(new.released["p"][2] - old.released["p"][2]) < 0.012
    # a placement that is checked keeps its stop 5 cm up (where the receiver is looked at again), option or not
    a, _w, _t, da = _run()
    b, _w, _t, db = _run(stage_above_m=0.015)
    assert da == db and a.attempts[0]["t_release"] == b.attempts[0]["t_release"]


def _side_plan(yaw_deg, side, k):
    """The grasp frames planned for a bowl fitted at ``yaw_deg`` when a step asks for the hand on ``side``."""
    import math

    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.pro2.robot import make_params

    lib = LB.M2Library("libero_spatial")
    world = World(lib)
    tool = Tool(lib, world)
    a = math.radians(yaw_deg)
    q = [float(x) for x in mat_to_quat(np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0],
                                                 [0.0, 0.0, 1.0]]))]
    tool.percepts = lambda: [
        {"id": "p0", "type": "akita_black_bowl", "bodies": {"main": {"pos": list(world.bowl), "quat_wxyz": q}}},
        {"id": "p1", "type": "plate", "bodies": {"main": {"pos": list(PLATE), "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}}]
    ap, pp, fp = make_params(None)
    spec = {"predicate": "on", "target": "p0", "place": "p1", "place_kind": "object", "place_instance": "p1",
            "instruction": "pick the akita black bowl and place it on the plate", "grasp_side": list(side)}
    agent = AG.OracleAgent(lib, lib.scene_from_percepts(tool.percepts()), spec, tool, 220,
                           replace(ap, side_wide_k=k), pp, fp)
    agent.act(world.pobs())
    agent.tried = []  # (the first step planned once already)
    out, diag = agent._plan_grasp(tool())
    return out, diag, world.bowl[:2].copy()


def test_a_step_that_asks_for_one_side_looks_further_down_the_planners_list_before_it_takes_the_other():
    side = np.array([0.0, -1.0])

    def on_side(f, c):
        v = np.asarray(f["centre"], float)[:2] - c
        return float(v @ side) / float(np.linalg.norm(v))

    # the bowl as one look fitted it: the planner's own first frames (the ones that turn the hand least) are all on
    # the other side or the flanks, with nothing in the way (the stored rows' ``[0, 6, 2]``)
    old, d0, c = _side_plan(40.0, side, 0)
    assert d0["grasp_side"][0] == 0 and on_side(old[0], c) < -0.8 and "grasp_side_wide" not in d0
    new, d1, c = _side_plan(40.0, side, 200)
    n = d1["grasp_side_wide"][0]
    assert n >= 5 and all(on_side(f, c) >= 0.8 for f in new[:n])  # the asked side's frames first
    assert on_side(new[0], c) >= 0.95 and abs(new[0]["yaw_change_deg"]) <= 20.0  # squarely on it, a small turn
    assert [f["record"] for f in new[n:]] == [f["record"] for f in old]  # the planner's own behind them, as they were
    # where the planner's first frames already hold one on the asked side: the same list, option or not
    a, da, _c = _side_plan(40.0, -side, 0)
    b, db, _c = _side_plan(40.0, -side, 200)
    assert da["grasp_side"][0] > 0 and [f["record"] for f in a] == [f["record"] for f in b] and "grasp_side_wide" not in db
