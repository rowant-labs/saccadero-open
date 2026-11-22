"""2026-11-16 (GRASP stream), three options of the robot's grasp checks, all off by default:

* ``c2_trust_gap``: the lift check goes by the fingers when the look reads a held object elsewhere in the hand;
* ``release_here``: the look tool takes a target's pose where the hand let go of it, not at its next look;
* ``c1_soft``: a hold of another width than the record's goes on to the lift;
* ``empty_view_m``: after a close on nothing the next look is taken from higher;
* ``empty_next``: a close on nothing at a fallback pose refutes it, and the target takes its next candidate.

The first tests are the look tool's own (``release_here``, ``rehold``) and the checks as rules. The last ones run the
whole attempt of ``robot/agent.py`` in a kinematic stand-in for the simulator (the hand goes where it is sent, the
fingers stop on what is between them, a held object moves with the hand), on the policy library when it is built on
this machine.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from rowant_lab.libero_env import m2_library as LB
from rowant_lab.libero_env import m2_rpaths as MP
from rowant_lab.libero_env.m2_kin import mat_to_quat
from rowant_lab.libero_env.tests.test_close_look import _cam, _lib
from rowant_lab.pro2.robot import agent as AG
from rowant_lab.pro2.robot import grasp as GR
from rowant_lab.pro2.robot import look as LK

need = pytest.mark.skipif(not (MP.POLICY_LIB / "manifest.json").exists(),
                          reason="the policy library is not built on this machine")
OPEN, SHUT = (0.04, -0.04), (0.0, 0.0)


# ------------------------------------------------------------------------------------------------ the look tool
class Looks:
    """Tracks answer ``status`` (a tracked pose is the prior moved by ``move``); full looks return ``fresh``."""

    def __init__(self, status="predicted", fresh=(), move=(0.0, 0.0, 0.0)):
        self.calls, self.status, self.fresh, self.move = [], status, list(fresh), np.asarray(move, float)

    def call(self, msg):
        self.calls.append(msg)
        if msg.get("mode") == "full":
            return {"percepts": [dict(q) for q in self.fresh], "times": {}}
        out = {pid: {"status": self.status, "R_w": pr["R_w"], "t_w": (np.asarray(pr["t_w"], float) + self.move).tolist(),
                     "n_px": 10} for pid, pr in msg["priors"].items()}
        return {"tracks": out, "times": {}}


def _percept(pid, pos, typ="can"):
    return {"id": pid, "type": typ, "gid": "g1", "joints": {},
            "bodies": {"main": {"pos": list(pos), "quat_wxyz": mat_to_quat(np.eye(3))}}}


def _tool(client, hand=(0.1, 0.2, 0.93)):
    t = LK.PerceptionTool(_lib(), client, _cam, "libero_object", "ep")
    t.obs = {"robot0_joint_pos": np.zeros(7), "robot0_gripper_qpos": np.asarray(SHUT, float),
             "robot0_eef_pos": np.asarray(hand, float), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0])}
    t.percepts = [_percept("p0", (0.1, 0.2, 0.9))]
    t.target = "p0"
    t.calls = 1  # the first full look has been taken
    return t


def _z(t):
    return t.percepts[0]["bodies"]["main"]["pos"][2]


def _close_open_rise(t, tell: bool, rise=0.08):
    """The hand closes at the target (the look at the close), opens and rises; ``tell``: the agent's ``release_here``."""
    t()  # the look at the close: the target is near the closed hand, so it counts as held from here
    assert t._held is not None
    told = t.release_here() if tell else None
    t.obs["robot0_gripper_qpos"] = np.asarray(OPEN, float)
    t.obs["robot0_eef_pos"] = t.obs["robot0_eef_pos"] + np.array([0.0, 0.0, rise])
    t()  # the next attempt's look, with the open hand at its pre-grasp point
    return told


def test_without_release_here_a_target_never_carried_rises_with_the_hand():
    cl = Looks()  # the merged look sees no percept of the target's type (the colour line with the hand above it)
    t = _tool(cl)
    _close_open_rise(t, tell=False)
    assert [m["mode"] for m in cl.calls] == ["track", "full", "track"]
    assert abs(_z(t) - 0.98) < 1e-9  # stored 8 cm above where it stands: the next grasp closes on air


def test_release_here_keeps_the_target_where_the_hand_let_go():
    cl = Looks()
    t = _tool(cl)
    assert _close_open_rise(t, tell=True) is True and t.n_release_here == 1
    assert [m["mode"] for m in cl.calls] == ["track", "full", "track"]  # the merged look still runs, around that pose
    assert abs(_z(t) - 0.90) < 1e-9 and t._held is None and t._release_pending is False
    assert np.allclose(cl.calls[-1]["priors"]["p0"]["t_w"], [0.1, 0.2, 0.9])  # and the track starts from there


def test_release_here_with_a_fresh_percept_takes_it_and_does_nothing_when_nothing_is_held():
    cl = Looks(fresh=[_percept("q0", (0.13, 0.2, 0.9))])  # the target as the merged look sees it, 3 cm from the grasp
    t = _tool(cl)
    _close_open_rise(t, tell=True)
    assert np.allclose(t.percepts[0]["bodies"]["main"]["pos"], [0.13, 0.2, 0.9]) and t.percepts[0]["id"] == "p0"
    t2 = _tool(Looks())
    assert t2.release_here() is False and t2._release_pending is False and t2.n_release_here == 0


def test_a_hand_that_stays_shut_after_release_here_does_not_take_the_target_up_again():
    cl = Looks()
    t = _tool(cl)
    t()
    t.release_here()
    t.obs["robot0_eef_pos"] = t.obs["robot0_eef_pos"] + np.array([0.0, 0.0, 0.08])  # risen, the fingers still shut
    t()
    assert t._held is None and abs(_z(t) - 0.90) < 1e-9 and t._release_pending is True
    t.obs["robot0_gripper_qpos"] = np.asarray(OPEN, float)
    t()
    assert cl.calls[-2]["mode"] == "full" and t._release_pending is False and abs(_z(t) - 0.90) < 1e-9


def test_rehold_sets_the_target_back_where_the_close_measured_it_in_the_hand():
    cl = Looks(status="tracked")
    t = _tool(cl)
    t()  # the look at the close: tracked where it was predicted
    ph, Rh = t._hand()
    R_rel, t_rel = t._held  # the target in the hand frame, as the close measured it
    R_rel, t_rel = np.array(R_rel), np.array(t_rel)
    t.obs["robot0_eef_pos"] = t.obs["robot0_eef_pos"] + np.array([0.0, 0.0, 0.10])  # lifted
    cl.move = np.array([0.02, 0.015, 0.0])  # the look at the lifted target reads it 2.5 cm off in the hand
    t()
    assert np.allclose(t.percepts[0]["bodies"]["main"]["pos"], [0.12, 0.215, 1.0])
    assert t.rehold(R_rel, t_rel) is True and t.n_rehold == 1
    assert np.allclose(t.percepts[0]["bodies"]["main"]["pos"], [0.1, 0.2, 1.0]) and t.percepts[0]["source"] == "held"
    t.obs["robot0_eef_pos"] = t.obs["robot0_eef_pos"] + np.array([0.3, 0.0, 0.0])  # carried on: rigid with the hand
    assert np.allclose(t._predict(t.percepts[0])["bodies"]["main"]["pos"], [0.4, 0.2, 1.0])
    t._held = None
    assert t.rehold(R_rel, t_rel) is False  # nothing held: nothing to set


def _fallback_tool(monkeypatch, table_bound=False):
    """No percept of the named type: two rejected fits and one unexplained region stand in for it."""
    import rowant_lab.pro2.robot.ground as GD

    def fake_aabb(lib, p):
        c = np.asarray(p["bodies"]["main"]["pos"], float)
        return c - 0.03, c + 0.03

    monkeypatch.setattr(GD, "aabb", fake_aabb)

    def rej(pid, pos, n_agree):
        return {"id": pid, "type": "can", "gid": "g" + pid, "pos": list(pos), "source": "texture",
                "fit": {"inlier": 0.9, "n_agree": n_agree}, "why": {"accepted": False},
                "bodies": {"main": {"pos": list(pos), "quat_wxyz": mat_to_quat(np.eye(3))}}}

    t = _tool(Looks())
    t.table_bound = table_bound
    t.percepts = [_percept("p9", (0.0, 0.5, 0.9), typ="tray")]
    t.lib.export_types = {"can": t.lib.export_types["can"], "tray": t.lib.export_types["can"]}
    t.extra = {"rejected": [rej("r1", (0.30, 0.00, 0.9), 900), rej("r2", (0.50, 0.10, 0.9), 500)],
               "blobs": [{"xy": [0.10, -0.20], "half_xy": [0.03, 0.03], "yaw": 0.0, "z_top": 0.96, "height": 0.06,
                          "n_px": 2000}]}
    t.on_parsed({"target_type": "can", "place_type": None})
    t.target = t.percepts[-1]["id"]
    return t


def _xy(t):
    return [round(x, 3) for x in next(p for p in t.percepts if p["id"] == t.target)["bodies"]["main"]["pos"][:2]]


def test_a_refuted_fallback_pose_gives_way_to_the_next_candidate(monkeypatch):
    t = _fallback_tool(monkeypatch)
    pid = t.target
    assert _xy(t) == [0.30, 0.00] and t.fallbacks[-1]["source"].startswith("rejected:")  # the best rejected fit
    t.obs["robot0_eef_pos"] = np.array([0.30, 0.0, 0.93])
    t()  # the look at the close: near the shut hand, so it counts as held
    assert t._held is not None
    out = t.refute_target()  # the hand closed on nothing there
    assert out["n"] == 1 and out["source"].startswith("rejected:") and abs(out["moved_m"] - 0.224) < 2e-3
    assert _xy(t) == [0.50, 0.10] and t.target == pid and t._held is None and t._release_pending  # the next best fit
    out = t.refute_target()
    assert out["source"] == "blob" and out["n"] == 2 and _xy(t) == [0.10, -0.20]  # then the unexplained region
    assert t.refute_target() is None and _xy(t) == [0.10, -0.20] and t.n_refuted == 2  # nothing left: it stays
    assert [f.get("refuted") for f in t.fallbacks if f.get("refuted")] == [1, 2]


def test_a_candidate_next_to_a_refuted_place_is_that_place(monkeypatch):
    t = _fallback_tool(monkeypatch)
    t.extra["rejected"][1]["pos"] = [0.33, 0.02, 0.9]  # the second fit stands 3.6 cm from the first: the same place
    t.extra["rejected"][1]["bodies"]["main"]["pos"] = [0.33, 0.02, 0.9]
    assert t.refute_target()["source"] == "blob" and _xy(t) == [0.10, -0.20]


def test_a_reported_percept_is_never_refuted(monkeypatch):
    t = _fallback_tool(monkeypatch)
    t.percepts = [_percept("p0", (0.1, 0.2, 0.9))] + t.percepts[:1]  # the server reported the named type itself
    t.target = "p0"
    assert t.refute_target() is None and t.n_refuted == 0
    t = _fallback_tool(monkeypatch)
    tgt = next(p for p in t.percepts if p["id"] == t.target)
    tgt["source"] = "relook:texture"  # a later merged look reported it: no longer a guess
    assert t.refute_target() is None


# ------------------------------------------------------------------------------------------------ the rules
def _rules(**kw):
    return SimpleNamespace(ap=AG.AgentParams(**kw), jaw=SimpleNamespace(L=0.0382))


def test_the_lift_check_by_the_fingers():
    a = _rules(c2_trust_gap=True)
    f = lambda gap, gap2, moved, rise, tol=0.01: AG.OracleAgent._c2_fingers(a, gap, gap2, moved, rise, tol)  # noqa: E731
    assert f(0.0379, 0.0379, 0.047, 0.10)        # the largest false reading in the rows: 47 mm on a 10 cm rise
    assert not f(0.0379, 0.0379, 0.098, 0.10)    # left on the table: the look reads it the whole rise away
    assert not f(0.0379, 0.0340, 0.020, 0.10)    # the fingers moved 3.9 mm on what they hold: not the same hold
    assert not f(0.0379, 0.0013, 0.000, 0.10)    # the fingers have met
    assert not f(0.0379, 0.0379, 0.020, 0.02)    # a hand that barely rose: only the look's own tolerance
    assert f(0.0379, 0.0379, 0.025, 0.02, 0.03)  # (through a delay that tolerance is 3 cm)


def test_a_hold_of_another_width():
    a = _rules(c1_soft=True)
    f = lambda gap, width, stable=True: AG.OracleAgent._c1_soft(a, gap, width, stable)  # noqa: E731
    assert f(0.0274, 0.0384) and f(0.0521, 0.0626) and f(0.0489, 0.0447)  # narrower and wider holds from the rows
    assert not f(0.0013, 0.0384)             # closed on nothing
    assert not f(0.0500, 0.0233)             # more than twice the record's width: something else
    assert not f(0.0769, 0.0624)             # the hand did not close (the gap of an open hand)
    assert not f(0.0167, 0.0099)             # rim grasps keep their own band
    assert not f(0.0274, 0.0384, stable=False)  # the fingers were still moving
    assert not AG.OracleAgent._c1_soft(_rules(), 0.0274, 0.0384, True)  # off by default


def test_the_agent_tells_the_tool_only_when_the_option_is_on():
    for on in (False, True):
        tool = SimpleNamespace(n=0)
        tool.release_here = lambda tool=tool: setattr(tool, "n", tool.n + 1)
        AG.OracleAgent._let_go(SimpleNamespace(ap=AG.AgentParams(release_here=on), oracle=tool))
        assert tool.n == int(on)
    AG.OracleAgent._let_go(SimpleNamespace(ap=AG.AgentParams(release_here=True), oracle=SimpleNamespace()))  # no such tool


def test_the_options_are_off_by_default():
    ap = AG.AgentParams()
    assert not (ap.c2_trust_gap or ap.release_here or ap.c1_soft)


# ------------------------------------------------------------------------------------------------ the whole attempt
def _expm(rv):
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(rv, float) / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


class World:
    """The simulator and the look tool in one. The hand moves 0.35 of each commanded step; the fingers travel 12 mm
    a step and stop on the object when the hand is at its grasp point (``width`` times the frame's width), else on
    nothing; a held object moves with the hand. The tool reports the target ``bias`` from where it is, and a held
    one ``lift_err`` further once the hand is 2 cm above where it closed. ``comes_up`` False: the fingers hold the
    width but the target stays on the table (the hand closed on something else of that width)."""

    def __init__(self, lib, percepts, width=1.0, bias=(0, 0, 0), lift_err=(0, 0, 0), comes_up=True):
        self.lib, self.percepts = lib, copy.deepcopy(percepts)
        self.true = np.asarray(percepts[0]["bodies"]["main"]["pos"], float).copy()
        self.width, self.bias, self.lift_err, self.comes_up = width, np.asarray(bias, float), np.asarray(lift_err, float), comes_up
        self.p, self.R = np.array([-0.1, 0.0, 0.25]), GR.hand_matrix([0.0, 1.0, 0.0])
        self.gap, self.held, self.z_close, self.speed = 2 * lib.jaw.L, False, None, 0.012  # the fingers' travel a step
        self.calls, self.t_now, self.agent = 0, 0, None
        self.released, self.reheld = [], []

    def pobs(self):
        q = 0.5 * self.gap - self.lib.jaw.face_offset
        x, y, z, w = (lambda m: (m[1], m[2], m[3], m[0]))(mat_to_quat(self.R))
        return {"robot0_eef_pos": self.p.copy(), "robot0_eef_quat": np.array([x, y, z, w]),
                "robot0_gripper_qpos": np.array([q, -q])}

    def _at_grasp(self) -> bool:
        fr = self.agent.attempts[-1].get("frame") if self.agent.attempts else None
        if fr is None:
            return False
        point = np.asarray(fr["centre"], float) - self.bias  # the frame was planned on the reported pose
        return float(np.linalg.norm(self.p - point)) < 0.006

    def step(self, a):
        a = np.asarray(a, float)
        dp = a[:3] * 0.05 * 0.35
        self.p = self.p + dp
        self.R = _expm(a[3:6] * 0.5 * 0.35) @ self.R
        if self.held and self.comes_up:
            self.true = self.true + dp
        if a[6] > 0:
            fr = self.agent.attempts[-1].get("frame") if self.agent.attempts else None
            on = self.held or self._at_grasp()
            stop = self.width * fr["width"] if (on and fr) else 0.0013
            self.gap = max(stop, self.gap - self.speed)
            if on and self.gap <= stop + 1e-9 and not self.held:
                self.held, self.z_close = True, float(self.p[2])
        else:
            self.gap = min(2 * self.lib.jaw.L, self.gap + self.speed)
            self.held = False

    # -- the look tool
    def __call__(self):
        self.calls += 1
        pos = self.true + self.bias
        if self.held and self.z_close is not None and self.p[2] > self.z_close + 0.02:
            pos = pos + self.lift_err
        self.percepts[0]["bodies"]["main"]["pos"] = [float(x) for x in pos]
        return self.lib.poses_from_percepts(self.percepts)

    def release_here(self):
        self.released.append((self.agent.attempts[-1]["outcome"], round(float(self.p[2]), 3)))

    def rehold(self, R_rel, t_rel):
        self.reheld.append(np.asarray(t_rel, float).copy())

    def refute_target(self):
        """The reported pose was a wrong guess; the next candidate is where the object stands."""
        if not np.any(self.bias):
            return None
        self.bias = np.zeros(3)
        return {"source": "blob", "n": 1}


def _run(max_steps=400, until=None, **kw):
    opts = {k: kw.pop(k) for k in list(kw) if k in AG.AgentParams.__dataclass_fields__}
    lib = LB.M2Library("libero_object")
    rec = lib.records["alphabet_soup"]
    percepts = [{"id": "p0", "type": "alphabet_soup", "gid": "g0", "joints": {},
                 "bodies": {"main": {"pos": [0.0, 0.10, -float(rec["bottom"])], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}},
                {"id": "p1", "type": "basket", "gid": "g1", "joints": {},
                 "bodies": {"main": {"pos": [0.05, -0.20, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}}]
    w = World(lib, percepts, **kw)
    scene = lib.scene_from_percepts(percepts)
    spec = {"predicate": "in", "target": "p0", "place": "p1_contain_region", "place_kind": "region",
            "place_instance": "p1", "instruction": "pick the alphabet soup and place it in the basket"}
    agent = AG.OracleAgent(lib, scene, spec, w, 400, AG.AgentParams(**opts))
    w.agent = agent
    for _ in range(max_steps):
        w.step(agent.act(w.pobs()))
        if agent.finished or (until and until(agent)):
            break
    return agent, w


def _first_done(agent):
    return len(agent.attempts) > 1 or "t_release" in agent.attempts[0]


@need
def test_kinematic_world_a_plain_grasp_goes_through_to_the_release():
    agent, w = _run(until=_first_done)
    a = agent.attempts[0]
    assert a["c1"]["pass"] and a["c2"]["pass"] and "by" not in a["c2"] and "t_release" in a
    assert np.linalg.norm((w.true - np.array([0.05, -0.20, 0.0]))[:2]) < 0.005  # the can is over the basket's middle


@need
def test_a_look_that_reads_the_held_object_elsewhere_fails_the_lift_check_today():
    agent, w = _run(until=_first_done, lift_err=(0.02, 0.015, 0.0))
    a = agent.attempts[0]
    assert a["c1"]["pass"] and a["outcome"] == "c2_fail" and abs(a["c2"]["rel_move_m"] - 0.025) < 1e-3
    assert abs(a["c2"]["gap"] - a["c1"]["gap"]) < 1e-9 and not w.released and not w.reheld


@need
def test_with_c2_trust_gap_the_fingers_decide_and_the_pose_from_the_close_stands():
    agent, w = _run(until=_first_done, lift_err=(0.02, 0.015, 0.0), c2_trust_gap=True)
    a = agent.attempts[0]
    assert a["c2"]["pass"] and a["c2"]["by"] == "fingers" and a["c2"]["rise_m"] > 0.03 and "t_release" in a
    assert len(w.reheld) == 1
    # the release was planned from the pose measured at the close: the can is over the basket's middle, not 2.5 cm off
    assert np.linalg.norm((w.true - np.array([0.05, -0.20, 0.0]))[:2]) < 0.005


@need
def test_with_c2_trust_gap_an_object_left_on_the_table_still_fails():
    agent, w = _run(until=_first_done, comes_up=False, c2_trust_gap=True)
    a = agent.attempts[0]
    assert a["outcome"] == "c2_fail" and a["c2"]["rel_move_m"] > 0.03 and "by" not in a["c2"]


@need
def test_release_here_is_told_at_the_grasp_height_after_an_empty_close_and_after_a_failed_lift():
    agent, w = _run(until=lambda ag: len(ag.attempts) > 1, bias=(0.03, 0.0, 0.0), release_here=True)  # 3 cm off: on air
    a = agent.attempts[0]
    assert a["outcome"] == "c1_fail" and a["c1"]["gap"] < 0.003
    assert w.released == [("c1_fail", round(a["frame"]["centre"][2], 3))]  # where it closed, before it rose
    agent, w = _run(until=lambda ag: len(ag.attempts) > 1, lift_err=(0.02, 0.015, 0.0), release_here=True)
    a = agent.attempts[0]
    # set down again near the grasp height (the move back down ends within its 1 cm tolerance), not up at the lift
    assert a["outcome"] == "c2_fail" and len(w.released) == 1 and w.released[0][0] == "c2_fail"
    assert abs(w.released[0][1] - (a["frame"]["centre"][2] + 0.01)) <= 0.011 and a["lift_m"] >= 0.04
    agent, w = _run(until=lambda ag: len(ag.attempts) > 1, bias=(0.03, 0.0, 0.0))  # off: the tool is not told
    assert agent.attempts[0]["outcome"] == "c1_fail" and not w.released


@need
def test_a_hold_of_another_width_goes_to_the_lift_with_c1_soft_and_fails_without():
    agent, w = _run(until=_first_done, width=0.72)
    assert agent.attempts[0]["outcome"] == "c1_fail" and agent.attempts[0]["c1"]["gap"] > 0.03
    agent, w = _run(until=_first_done, width=0.72, c1_soft=True)
    a = agent.attempts[0]
    assert a["c1"]["pass"] is False and a["c1"]["soft"] is True
    assert a["c2"]["pass"] and a["c2"]["by"] == "fingers" and "t_release" in a
    agent, w = _run(until=_first_done, width=0.72, c1_soft=True, comes_up=False)  # that width, but the target stays
    assert agent.attempts[0]["outcome"] == "c2_fail"


@need
def test_after_a_close_on_nothing_a_look_that_finds_nothing_new_is_followed_by_one_from_higher():
    def gotos(agent, a):  # the heights the hand went to between this attempt's start and its first plan
        t0 = a["t_start"]
        return [e["at"][2] for e in agent.events if e["ev"] == "goto" and t0 <= e["t0"] <= t0 + 2]

    agent, w = _run(until=lambda ag: len(ag.attempts) > 2, bias=(0.03, 0.0, 0.0))
    a0, a1 = agent.attempts[:2]
    assert a0["outcome"] == "c1_fail" and "view_m" not in a0 and "view_m" not in a1
    # the stand-in tool reports the same wrong place again: the hand rises 18 cm above the grasp and looks once more
    agent, w = _run(until=lambda ag: len(ag.attempts) > 2, bias=(0.03, 0.0, 0.0), empty_view_m=0.18)
    a0, a1 = agent.attempts[:2]
    assert a0["outcome"] == "c1_fail" and "view_m" not in a0 and a1["view_m"] == 0.18
    assert max(gotos(agent, a1)) > a0["frame"]["centre"][2] + 0.16 and w.calls >= 4
    # a failed lift is not a close on nothing: nothing rises
    agent, w = _run(until=lambda ag: len(ag.attempts) > 1, lift_err=(0.02, 0.015, 0.0), empty_view_m=0.18)
    assert agent.attempts[0]["outcome"] == "c2_fail" and "view_m" not in agent.attempts[0]


# ------------------------------------------------------------- the whole attempt with the look tool itself in the loop
class Server:
    """The perception server over the kinematic world's truth. A full look reports each object where it is, except
    the target while the hand hides it (the hand under 13 cm high and within 12 cm of it across). A track finds the
    object when it is within 5 cm of its prior (and reads a lifted, held one ``lift_err`` off), else refuses."""

    def __init__(self, world):
        self.w, self.modes = world, []

    def _target(self, with_err=True):
        w = self.w
        pos = w.true.copy()
        if with_err and w.held and w.z_close is not None and w.p[2] > w.z_close + 0.02:
            pos = pos + w.lift_err
        return pos

    def call(self, msg):
        w = self.w
        self.modes.append(msg.get("mode"))
        if msg.get("mode") == "full":
            out = []
            hidden = w.p[2] < 0.13 and float(np.linalg.norm((w.p - w.true)[:2])) < 0.12
            for p in w.percepts:
                q = {**p, "fit": {"n_agree": 500, "inlier": 0.9}, "source": "texture"}
                if p["id"] == "p0":
                    if hidden:
                        continue
                    q["bodies"] = {"main": {"pos": [float(x) for x in self._target()], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}
                out.append(q)
            return {"percepts": out, "extra": {"kept": [], "rejected": [], "blobs": []}, "times": {}}
        tracks = {}
        for pid, pr in msg["priors"].items():
            pos = self._target()
            if float(np.linalg.norm(pos - np.asarray(pr["t_w"], float))) < 0.05:
                tracks[pid] = {"status": "tracked", "R_w": pr["R_w"], "t_w": [float(x) for x in pos], "n_px": 900}
            else:
                tracks[pid] = {"status": "predicted", "R_w": pr["R_w"], "t_w": pr["t_w"], "n_px": 0}
        return {"tracks": tracks, "times": {}}


class Sim(World):
    """The kinematic world as the simulator alone (the look tool is the real one). ``jump``: at step ``jump_at`` the
    target moves by it, as the scene does in the cases that move the target before the hand arrives; with
    ``trigger_m`` it moves instead when the hand first comes that near (what the replayed episodes showed)."""

    def __init__(self, lib, percepts, jump=(0, 0, 0), jump_at=10, trigger_m=None, **kw):
        super().__init__(lib, percepts, **kw)
        self.jump, self.jump_at, self.n, self.trigger_m = np.asarray(jump, float), jump_at, 0, trigger_m

    def _at_grasp(self) -> bool:  # a can is grasped across its middle: the hand is on it when it is over it
        d = self.p - self.true
        return float(np.linalg.norm(d[:2])) < 0.008 and abs(float(d[2])) < 0.04

    def step(self, a):
        self.n += 1
        if self.trigger_m is not None:
            if float(np.linalg.norm(self.p - self.true)) < self.trigger_m:
                self.true, self.trigger_m = self.true + self.jump, None
        elif self.n == self.jump_at:
            self.true = self.true + self.jump
        super().step(a)


def _run_with_tool(until, max_steps=400, sim=None, speed=None, **kw):
    opts = {k: kw.pop(k) for k in list(kw) if k in AG.AgentParams.__dataclass_fields__}
    lib = LB.M2Library("libero_object")
    rec = lib.records["alphabet_soup"]
    percepts = [{"id": "p0", "type": "alphabet_soup", "gid": "g0", "joints": {},
                 "bodies": {"main": {"pos": [0.0, 0.10, -float(rec["bottom"])], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}},
                {"id": "p1", "type": "basket", "gid": "g1", "joints": {},
                 "bodies": {"main": {"pos": [0.05, -0.20, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}}]
    w = (sim or Sim)(lib, percepts, **kw)
    if speed is not None:
        w.speed = speed
    srv = Server(w)
    tool = LK.PerceptionTool(lib, srv, _cam, "libero_object", "ep")
    tool.obs = {**w.pobs(), "robot0_joint_pos": np.zeros(7)}
    tool.full_look()  # the episode's first look, from the start pose
    tool.target = "p0"
    spec = {"predicate": "in", "target": "p0", "place": "p1_contain_region", "place_kind": "region",
            "place_instance": "p1", "instruction": "pick the alphabet soup and place it in the basket"}
    agent = AG.OracleAgent(lib, lib.scene_from_percepts(tool.percepts), spec, tool, 400, AG.AgentParams(**opts))
    w.agent = agent
    for _ in range(max_steps):
        tool.obs = {**w.pobs(), "robot0_joint_pos": np.zeros(7)}
        w.step(agent.act(tool.obs))
        if agent.finished or until(agent):
            break
    return agent, w, tool


def _base(a):
    """The height of the object's bottom this attempt was planned on (the grip height less the record's own)."""
    return a["frame"]["centre"][2] - a["frame"]["z_rel"]


@need
def test_with_the_look_tool_in_the_loop_a_plain_grasp_reaches_the_release():
    agent, w, tool = _run_with_tool(_first_done)
    a = agent.attempts[0]
    assert a["c1"]["pass"] and a["c2"]["pass"] and "t_release" in a and tool.n_release_here == 0 and tool.n_rehold == 0


@need
def test_today_a_target_the_scene_moved_costs_a_grasp_at_the_air_above_where_it_was():
    # the target jumps 7.5 cm before the hand arrives: the close is on nothing. The look after it, from the pre-grasp
    # point, does not see the target (the hand hides it) and its stored pose has risen with the hand: the second
    # grasp closes on air some 5 cm up. Only from there is the hand high enough for the third look to find it
    # (the depth line's rows: c1_fail at 924 mm, c1_fail at 983 mm, then the grasp 7 cm away)
    agent, w, tool = _run_with_tool(lambda ag: len(ag.attempts) > 2, jump=(0.075, 0.0, 0.0))
    a0, a1, a2 = agent.attempts[:3]
    assert a0["outcome"] == a1["outcome"] == "c1_fail" and a0["c1"]["gap"] < 0.003 and a1["c1"]["gap"] < 0.003
    assert _base(a1) > _base(a0) + 0.03 and abs(a1["frame"]["centre"][0] - a0["frame"]["centre"][0]) < 0.02
    assert abs(_base(a2) - _base(a0)) < 0.005 and abs(a2["frame"]["centre"][0] - a0["frame"]["centre"][0] - 0.075) < 0.01


@need
def test_release_here_and_the_higher_look_find_it_on_the_second_grasp():
    agent, w, tool = _run_with_tool(lambda ag: len(ag.attempts) > 2 or "c2" in ag.attempts[-1], jump=(0.075, 0.0, 0.0),
                                    release_here=True, empty_view_m=0.18)
    a0, a1 = agent.attempts[:2]
    # the look from the pre-grasp point is hidden by the hand; the one from 18 cm up finds the target 7.5 cm away
    assert a0["outcome"] == "c1_fail" and a1["view_m"] == 0.18 and tool.n_release_here == 1 and tool.n_redetect == 1
    assert abs(_base(a1) - _base(a0)) < 0.005 and abs(a1["frame"]["centre"][0] - a0["frame"]["centre"][0] - 0.075) < 0.01
    assert a1["c1"]["pass"]


@need
def test_release_here_alone_stays_at_the_place_the_object_left():
    # without the higher look the hand still hides the place: the pose no longer climbs, but nothing new is seen
    agent, w, tool = _run_with_tool(lambda ag: len(ag.attempts) > 2, jump=(0.075, 0.0, 0.0), release_here=True)
    a0, a1 = agent.attempts[:2]
    assert a0["outcome"] == a1["outcome"] == "c1_fail" and abs(_base(a1) - _base(a0)) < 0.005
    assert abs(a1["frame"]["centre"][0] - a0["frame"]["centre"][0]) < 0.02


@need
def test_with_the_look_tool_the_fingers_decide_a_lift_the_look_misreads():
    agent, w, tool = _run_with_tool(_first_done, lift_err=(0.02, 0.015, 0.0))
    assert agent.attempts[0]["outcome"] == "c2_fail" and abs(agent.attempts[0]["c2"]["rel_move_m"] - 0.025) < 2e-3
    agent, w, tool = _run_with_tool(_first_done, lift_err=(0.02, 0.015, 0.0), c2_trust_gap=True)
    a = agent.attempts[0]
    assert a["c2"]["pass"] and a["c2"]["by"] == "fingers" and tool.n_rehold == 1 and "t_release" in a
    assert np.linalg.norm((w.true - np.array([0.05, -0.20, 0.0]))[:2]) < 0.005  # set down over the basket's middle


@need
def test_refute_target_on_the_policy_library_itself():
    """The fallback builders with the library's own models: a rejected fit, then two regions, the nearer one within
    5 cm of the refuted place."""
    lib = LB.M2Library("libero_object")
    rec = lib.records["alphabet_soup"]
    z = -float(rec["bottom"])
    w = Sim(lib, [{"id": "p1", "type": "basket", "gid": "g1", "joints": {},
                   "bodies": {"main": {"pos": [0.05, -0.20, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}}])
    tool = LK.PerceptionTool(lib, Server(w), _cam, "libero_object", "ep")
    tool.obs = {**w.pobs(), "robot0_joint_pos": np.zeros(7)}
    tool.full_look()  # the basket alone: no percept of the named type
    tool.extra = {"kept": [], "rejected": [
        {"id": "r1", "type": "alphabet_soup", "gid": "gr1", "pos": [0.10, 0.10, z], "source": "texture",
         "fit": {"inlier": 0.8, "n_agree": 700}, "why": {"accepted": False},
         "bodies": {"main": {"pos": [0.10, 0.10, z], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}}],
        "blobs": [{"xy": [0.12, 0.13], "half_xy": [0.033, 0.033], "yaw": 0.0, "z_top": 0.076, "height": 0.076, "n_px": 3000},
                  {"xy": [-0.10, 0.25], "half_xy": [0.030, 0.030], "yaw": 0.3, "z_top": 0.070, "height": 0.070, "n_px": 2500}]}
    tool.on_parsed({"target_type": "alphabet_soup", "place_type": "basket"})
    tgt = tool.percepts[-1]
    tool.target = tgt["id"]
    assert tgt["type"] == "alphabet_soup" and np.allclose(tgt["bodies"]["main"]["pos"][:2], [0.10, 0.10])
    out = tool.refute_target()  # the first region is 3.6 cm from the refuted fit: the same place; the second is taken
    assert out["source"] == "blob" and out["n"] == 1
    new = next(p for p in tool.percepts if p["id"] == tool.target)
    assert np.linalg.norm(np.asarray(new["bodies"]["main"]["pos"][:2]) - [-0.10, 0.25]) < 0.03 and new["gid"] is None
    poses = lib.poses_from_percepts(tool.percepts)  # and the agent's scene still reads it under the same body name
    assert f"{tool.target}_main" in poses and tool.refute_target() is None


@need
def test_pre_look_sees_a_target_the_scene_moved_and_does_not_close_on_where_it_was():
    # the target jumps 13 cm when the hand comes within 11 cm, as the hand arrives over the pre-grasp point
    done = lambda ag: len(ag.attempts) > 2 or "c2" in ag.attempts[-1]  # noqa: E731
    agent, w, tool = _run_with_tool(done, jump=(0.13, 0.0, 0.0), trigger_m=0.11)
    assert agent.attempts[0]["outcome"] == "c1_fail" and agent.attempts[0]["c1"]["gap"] < 0.003  # today: on nothing
    agent, w, tool = _run_with_tool(done, jump=(0.13, 0.0, 0.0), trigger_m=0.11, pre_look=True)
    a0, a1 = agent.attempts[:2]
    assert a0["outcome"] == "moved_before_grasp" and abs(a0["pre_look_m"] - 0.13) < 0.005 and "c1" not in a0
    assert tool.n_redetect == 1 and tool.redetect is False  # the tool's own option is as it was
    assert a1["c1"]["pass"] and abs(a1["frame"]["centre"][0] - a0["frame"]["centre"][0] - 0.13) < 0.01
    assert a1["frame"]["record"] == a0["frame"]["record"]  # the frame never tried is still the first choice
    assert a1["t_start"] < 30  # and no descent, close and way back were spent on the old place


@need
def test_pre_look_changes_nothing_when_the_target_is_where_it_was_or_is_not_seen():
    agent, w, tool = _run_with_tool(_first_done, pre_look=True)  # nothing moved: the plain grasp, one more look
    a = agent.attempts[0]
    assert a["pre_look_m"] < 0.002 and a["c1"]["pass"] and a["c2"]["pass"] and "t_release" in a and tool.n_redetect == 0
    # moved 7.5 cm, where the hand over the old place hides it: the look finds nothing new and the grasp goes on
    agent, w, tool = _run_with_tool(lambda ag: len(ag.attempts) > 1, jump=(0.075, 0.0, 0.0), trigger_m=0.11,
                                    pre_look=True)
    a = agent.attempts[0]
    assert a["pre_look_m"] == 0.0 and a["outcome"] == "c1_fail" and tool.n_redetect == 1


@need
def test_pre_look_restarts_at_most_pre_look_max_times():
    class Flees(Sim):  # a target that moves on every time the hand comes near
        def step(self, a):
            if float(np.linalg.norm(self.p - self.true)) < 0.11:
                self.true = self.true + np.array([0.13, 0.0, 0.0]) * (1 if self.true[0] < 0.1 else -1)
            World.step(self, a)

    lib = LB.M2Library("libero_object")
    rec = lib.records["alphabet_soup"]
    percepts = [{"id": "p0", "type": "alphabet_soup", "gid": "g0", "joints": {},
                 "bodies": {"main": {"pos": [0.0, 0.10, -float(rec["bottom"])], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}},
                {"id": "p1", "type": "basket", "gid": "g1", "joints": {},
                 "bodies": {"main": {"pos": [0.05, -0.20, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}}]
    w = Flees(lib, percepts)
    tool = LK.PerceptionTool(lib, Server(w), _cam, "libero_object", "ep")
    tool.obs = {**w.pobs(), "robot0_joint_pos": np.zeros(7)}
    tool.full_look()
    tool.target = "p0"
    spec = {"predicate": "in", "target": "p0", "place": "p1_contain_region", "place_kind": "region",
            "place_instance": "p1", "instruction": "pick the alphabet soup and place it in the basket"}
    agent = AG.OracleAgent(lib, lib.scene_from_percepts(tool.percepts), spec, tool, 400, AG.AgentParams(pre_look=True))
    w.agent = agent
    for _ in range(200):
        tool.obs = {**w.pobs(), "robot0_joint_pos": np.zeros(7)}
        w.step(agent.act(tool.obs))
        if agent.finished or len(agent.attempts) > 3:
            break
    outs = [a.get("outcome") for a in agent.attempts[:3]]
    assert outs[:2] == ["moved_before_grasp", "moved_before_grasp"] and outs[2] != "moved_before_grasp"
    assert agent.n_pre_look == 2 and "pre_look_m" not in agent.attempts[2]


@need
def test_open_wait_the_look_after_a_failed_grasp_waits_for_the_open_hand():
    # fingers that need 19 steps to open (the robot's need 10 to 14, and the look came 6 to 12 after the command).
    # The target has jumped 13 cm; the close is on nothing; release_here keeps its pose where it was
    done = lambda ag: len(ag.attempts) > 2 or "c2" in ag.attempts[-1]  # noqa: E731
    agent, w, tool = _run_with_tool(done, jump=(0.13, 0.0, 0.0), trigger_m=0.11, speed=0.004, release_here=True)
    a0, a1 = agent.attempts[:2]
    # today: the next look finds the hand still closed, so it is a track from the old place and not the merged look
    assert a0["outcome"] == a1["outcome"] == "c1_fail" and abs(a1["frame"]["centre"][0] - a0["frame"]["centre"][0]) < 0.02
    agent, w, tool = _run_with_tool(done, jump=(0.13, 0.0, 0.0), trigger_m=0.11, speed=0.004, release_here=True,
                                    open_wait=15)
    a0, a1 = agent.attempts[:2]
    assert a0["outcome"] == "c1_fail" and 0 < a1["open_wait"] <= 15 and "open_wait" not in a0
    assert a1["c1"]["pass"] and abs(a1["frame"]["centre"][0] - a0["frame"]["centre"][0] - 0.13) < 0.01
    agent, w, tool = _run_with_tool(_first_done, open_wait=15)  # nothing failed: nothing waited for
    assert "open_wait" not in agent.attempts[0] and "t_release" in agent.attempts[0]


@need
def test_open_wait_after_a_lost_carry_the_merged_look_finds_the_object_before_any_grasp_at_the_air():
    class Drops(Sim):  # the object leaves the fingers 12 cm into the carry, once, and lies where it fell
        def step(self, a):
            World.step(self, a)
            if self.held and not self.dropped and float(np.linalg.norm((self.true - self.start)[:2])) > 0.12:
                self.held, self.gap, self.dropped = False, 0.0013, True
                self.true = np.array([self.true[0], self.true[1], self.start[2]])

        def __init__(self, lib, percepts, **kw):
            super().__init__(lib, percepts, **kw)
            self.start, self.dropped = self.true.copy(), False

    done = lambda ag: len(ag.attempts) > 2 or (len(ag.attempts) > 1 and "c1" in ag.attempts[1])  # noqa: E731
    agent, w, tool = _run_with_tool(done, sim=Drops, speed=0.004, carry_watch=True)
    a0, a1 = agent.attempts[:2]
    assert a0["outcome"] == "lost_in_carry"
    assert a1["outcome"] == "c1_fail" and a1["frame"]["centre"][2] > a0["frame"]["centre"][2] + 0.05  # at the air
    agent, w, tool = _run_with_tool(done, sim=Drops, speed=0.004, carry_watch=True, open_wait=15)
    a0, a1 = agent.attempts[:2]
    assert a0["outcome"] == "lost_in_carry" and a1["open_wait"] > 0
    assert a1["c1"]["pass"] and abs(a1["frame"]["centre"][2] - a0["frame"]["centre"][2]) < 0.02  # where it lies
    assert np.linalg.norm(np.asarray(a1["frame"]["centre"][:2]) - w.true[:2]) < 0.02 or w.held


def test_a_refused_track_of_a_held_target_made_the_tool_count_the_hands_move_twice():
    def lifted_then_lowered(**kw):
        cl = Looks(status="tracked")
        t = _tool(cl)                 # the target at z 0.90, the shut hand at 0.93
        for k, v in kw.items():
            setattr(t, k, v)
        t()                           # the look at the close: tracked
        cl.status = "predicted"       # the look at the lift is refused (the usual answer on the colour line)
        t.obs["robot0_eef_pos"] = t.obs["robot0_eef_pos"] + np.array([0.0, 0.0, 0.10])
        t()
        assert abs(t.percepts[0]["bodies"]["main"]["pos"][2] - 1.00) < 1e-9
        t.obs["robot0_eef_pos"] = t.obs["robot0_eef_pos"] + np.array([0.0, 0.0, -0.09])  # down to 1 cm above the grasp
        return t

    t = lifted_then_lowered()
    assert abs(t._predict(t.percepts[0])["bodies"]["main"]["pos"][2] - 1.01) < 1e-9  # as it is: 10 cm too high
    t = lifted_then_lowered(rigid_predict=True)
    assert abs(t._predict(t.percepts[0])["bodies"]["main"]["pos"][2] - 0.91) < 1e-9  # the agent's ``held_pose_fix``
    t = lifted_then_lowered()
    assert t.release_here() and abs(t.percepts[0]["bodies"]["main"]["pos"][2] - 0.91) < 1e-9  # release_here always
    lib = SimpleNamespace(jaw=SimpleNamespace(L=0.08, finger_z=(0.0, 0.0)))
    scene = {"instances": {"t": {"type": "can", "root_body_name": "main"}}, "arena": []}
    for on in (False, True):
        tool = SimpleNamespace(rigid_predict=False)
        AG.OracleAgent(lib, scene, {"target": "t"}, tool, 300, ap=AG.AgentParams(held_pose_fix=on))
        assert tool.rigid_predict is on


@need
def test_with_the_look_tool_a_lift_the_object_did_not_survive_is_followed_by_a_grasp_where_it_stands():
    class Slips(Sim):  # the object leaves the fingers 3 cm into the lift and stays where it was picked
        def step(self, a):
            World.step(self, a)
            if self.held and self.z_close is not None and self.p[2] > self.z_close + 0.03:
                self.held, self.gap = False, 0.0013
                self.true = self.start.copy()

        def __init__(self, lib, percepts, **kw):
            super().__init__(lib, percepts, **kw)
            self.start = self.true.copy()

    done = lambda ag: len(ag.attempts) > 2 or (len(ag.attempts) > 1 and "c1" in ag.attempts[1])  # noqa: E731
    agent, w, tool = _run_with_tool(done, sim=Slips)
    a0, a1 = agent.attempts[:2]
    assert a0["outcome"] == "c2_fail" and a0["c2"]["gap"] < 0.003
    # today: the look at the lift is refused, the hand's rise is then counted twice, and the next grasp is at the air
    assert a1["outcome"] == "c1_fail" and _base(a1) > _base(a0) + 0.08
    agent, w, tool = _run_with_tool(done, sim=Slips, release_here=True)
    a0, a1 = agent.attempts[:2]
    assert a0["outcome"] == "c2_fail" and abs(_base(a1) - _base(a0)) < 0.02 and a1["c1"]["pass"]


@need
def test_a_look_from_the_pre_grasp_point_that_finds_the_target_spares_the_higher_one():
    # the target has jumped 13 cm, clear of the hand: the first look after the close on nothing finds it, nothing rises
    agent, w, tool = _run_with_tool(lambda ag: len(ag.attempts) > 2 or "c2" in ag.attempts[-1], jump=(0.13, 0.0, 0.0),
                                    trigger_m=0.11, release_here=True, empty_view_m=0.28)
    a0, a1 = agent.attempts[:2]
    assert a0["outcome"] == "c1_fail" and "view_m" not in a1 and a1["c1"]["pass"] and tool.n_redetect == 0


@need
def test_a_close_on_nothing_at_a_fallback_pose_moves_on_to_the_next_candidate_with_empty_next():
    # restored 2026-11-16 after the merge with the integration branch (removed in 1350bb16; ``empty_next`` is on in
    # the combined arm and had no whole-attempt test left)
    agent, w = _run(until=lambda ag: len(ag.attempts) > 2 or "c2" in ag.attempts[-1], bias=(-0.15, 0.10, 0.0))
    assert [a["outcome"] for a in agent.attempts[:2]] == ["c1_fail", "c1_fail"]  # today: the same wrong place again
    assert all("refuted" not in a for a in agent.attempts)
    agent, w = _run(until=lambda ag: len(ag.attempts) > 2 or "c2" in ag.attempts[-1], bias=(-0.15, 0.10, 0.0),
                    empty_next=True)
    a0, a1 = agent.attempts[:2]
    assert a0["outcome"] == "c1_fail" and a0["refuted"] == {"source": "blob", "n": 1}
    assert a1["c1"]["pass"] and np.linalg.norm(np.asarray(a1["frame"]["centre"][:2]) - [0.0, 0.10]) < 0.01
