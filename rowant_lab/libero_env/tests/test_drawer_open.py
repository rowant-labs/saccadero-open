"""2026-11-16 night: the opening skill's options (drawer.OpenAgent: open_gap_check, open_settle, open_relook), the
half-turn twin about a fixture's carcass, and look.face_in's "carcass" mode. All off by default."""

import math
from pathlib import Path

import numpy as np
import pytest

from rowant_lab.pro2.robot import agent as AG
from rowant_lab.pro2.robot import drawer as DR


def _goal():
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.libero_env.m2_kin import mat_to_quat, part_body_poses
    from rowant_lab.pro2 import long_paths as LP
    from rowant_lab.pro2 import machine as MC

    if not (Path(MC.LONG_LIB) / "m2" / "policy-v3" / "manifest.json").is_file():
        pytest.skip("the library tree is not on this machine")
    _ob, policy, m1 = LB.lfb_dirs(MC.LONG_LIB)
    lib = LB.M2Library("libero_goal", lib_dir=policy, m1_library=m1, extra_suites=LP.extra_suites(["libero_goal"]))
    tdoc = lib.export_types["wooden_cabinet"]

    def percept(pos, yaw_deg, pid="p6", joints=None):
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        bp = part_body_poses(tdoc, {tdoc["root_body"]: (R, np.asarray(pos, float))}, joints or {})
        return {"id": pid, "type": "wooden_cabinet", "gid": None, "joints": dict(joints or {}),
                "fit": {"n_agree": 10, "inlier": 1.0}, "source": "support",
                "bodies": {b: {"pos": [float(x) for x in t], "quat_wxyz": mat_to_quat(RR)} for b, (RR, t) in bp.items()}}

    return lib, percept


HAND0 = np.array([-0.2168, 0.0094])  # where the hand starts on the benchmark's table (a replay's truth)
# a training episode's truth (S01, start state 0, seed 160): the cabinet moved and turned, and the first look's
# accepted fit of it back to front (the row's own numbers)
TRUE_ROOT, TRUE_YAW = [0.1085, -0.1854, 0.905], -161.67
FIT_ROOT, FIT_YAW = [0.1094, -0.2133, 0.9043], 18.5


def test_the_carcass_keeps_its_middle_under_a_half_turn_and_the_whole_box_does_not():
    lib, percept = _goal()
    plain = percept([0.0335, -0.2304, 0.905], 180.0)
    c = DR.carcass_centre(lib, plain)
    assert np.allclose(c, [0.0295, -0.2467], atol=0.001)  # the true base box of a replay: x -0.098..0.157, y -0.342..-0.151
    tw = DR.twin_percept(lib, plain)
    assert np.allclose(DR.carcass_centre(lib, tw), c, atol=1e-6) and tw["source"] == "support+twin"
    assert np.allclose(DR.twin_percept(lib, tw)["bodies"]["main"]["pos"], plain["bodies"]["main"]["pos"], atol=1e-6)
    h0, h1 = DR.handle_plan(lib, plain, "cabinet_middle"), DR.handle_plan(lib, tw, "cabinet_middle")
    assert np.allclose(h0["d"], [0, 1, 0], atol=1e-6) and np.allclose(h1["d"], [0, -1, 0], atol=1e-6)
    # the handle stands out in front: its twin lies behind the carcass, 3.3 cm nearer the middle than a turn about
    # the fixture's own origin would put it
    assert abs((2.0 * c[1] - h0["c"][1]) - h1["c"][1]) < 1e-6 and abs(h1["c"][2] - h0["c"][2]) < 1e-9


def test_a_back_to_front_fit_turned_about_its_carcass_lands_on_the_true_handle():
    lib, percept = _goal()
    truth = DR.handle_plan(lib, percept(TRUE_ROOT, TRUE_YAW), "cabinet_middle")
    fit = percept(FIT_ROOT, FIT_YAW)
    seen = DR.handle_plan(lib, fit, "cabinet_middle")
    assert float(seen["d"] @ truth["d"]) < -0.99 and float(np.linalg.norm(seen["c"] - truth["c"])) > 0.2
    assert DR.opens_away(lib, fit, HAND0, "cabinet_middle") is True
    tw = DR.twin_percept(lib, fit, "faced")
    got = DR.handle_plan(lib, tw, "cabinet_middle")
    assert DR.opens_away(lib, tw, HAND0, "cabinet_middle") is False
    assert float(got["d"] @ truth["d"]) > 0.999 and float(np.linalg.norm(got["c"] - truth["c"])) < 0.006
    # the plain cabinet, and one turned 17 degrees either way, face the hand as they stand
    for yaw in (180.0, 163.0, 197.0):
        assert DR.opens_away(lib, percept([0.03, -0.23, 0.905], yaw), HAND0, "cabinet_top") is False
    # the Spatial cabinet (turned 154 degrees) faces the hand; turned 17 degrees further and moved 9 cm it opens
    # across the hand's side: nothing is said, and nothing is turned
    assert DR.opens_away(lib, percept([0.0286, -0.2677, 0.9065], 154.3), HAND0, "cabinet_top") is False
    across = percept([0.12, -0.28, 0.9065], 137.0)
    assert DR.opens_away(lib, across, HAND0, "cabinet_top") is None
    assert DR.opens_away(lib, DR.twin_percept(lib, across), HAND0, "cabinet_top") is None


def test_face_in_about_the_carcass_turns_only_what_stands_back_to_front():
    from rowant_lab.pro2.robot.look import PerceptionTool

    lib, percept = _goal()
    truth = DR.handle_plan(lib, percept(TRUE_ROOT, TRUE_YAW), "cabinet_middle")

    def tool(p, mode):
        t = object.__new__(PerceptionTool)
        t.lib, t.percepts, t.obs = lib, [p], {"robot0_eef_pos": [HAND0[0], HAND0[1], 1.17]}
        t.face_in_on, t.face_in_mode, t._hand0, t.n_faced, t.opened = True, mode, None, 0, []
        t.n_full, t.n_track, t.budget = 0, 0, 0  # no look left: the old mode's tracker is not asked
        return t

    t = tool(percept(FIT_ROOT, FIT_YAW), "carcass")
    assert t.face_in() == 1 and t.percepts[0]["source"] == "support+faced" and t.percepts[0]["id"] == "p6"
    got = DR.handle_plan(lib, t.percepts[0], "cabinet_middle")
    assert float(np.linalg.norm(got["c"] - truth["c"])) < 0.006
    assert t.face_in() == 0  # it faces in now
    t = tool(percept([0.0335, -0.2304, 0.905], 180.0), "carcass")
    assert t.face_in() == 0 and t.percepts[0]["source"] == "support"
    for across in (percept([0.12, -0.28, 0.9065], 137.0), DR.twin_percept(lib, percept([0.12, -0.28, 0.9065], 137.0))):
        t = tool(across, "carcass")  # a fixture that opens across the hand's side: left as it is, either way round
        assert t.face_in() == 0
    # a plain start's back-to-front fit (the fit shares the carcass with the truth): the old mode leaves the handle
    # 3 cm off along the fixture's depth, the carcass mode puts it on the true one
    true0 = percept([0.0221, -0.2328, 0.905], 180.0)
    flipped = DR.twin_percept(lib, true0)
    h_true = DR.handle_plan(lib, true0, "cabinet_top")
    t_old, t_new = tool(dict(flipped), None), tool(dict(flipped), "carcass")
    assert t_old.face_in() == 1 and t_new.face_in() == 1
    e_old = np.linalg.norm(DR.handle_plan(lib, t_old.percepts[0], "cabinet_top")["c"] - h_true["c"])
    e_new = np.linalg.norm(DR.handle_plan(lib, t_new.percepts[0], "cabinet_top")["c"] - h_true["c"])
    assert 0.025 < e_old < 0.04 and e_new < 1e-6


# ------------------------------------------------------------------------------------------ the agent, stood in
R0 = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])


class Jaw:
    L = 0.04

    def gap_from_qpos(self, a, b):
        return float(a) - float(b)


class Tool:
    """A look tool stand-in: one fixture percept, and the looks ``_look_again`` asks for."""

    def __init__(self, percepts, replies=(), budget=4):
        self.percepts, self.replies, self.budget = list(percepts), list(replies), budget
        self.n_full = self.n_track = 0
        self.asked = []
        self._hand0 = HAND0.copy()
        self.t_now = 0

    def full_look(self, merge=False, only=None, **kw):
        self.asked.append((merge, only))
        self.n_full += 1
        return self.replies.pop(0) if self.replies else {"percepts": [], "extra": {"rejected": []}}


class World:
    """The hand goes where it is sent (5 cm a step at most), the jaws move 1 cm a step and stop on a bar when the
    hand stands at the handle; a pulled handle comes with the hand."""

    def __init__(self, handle, d, lag_gap=0, step_m=0.05):
        self.p = np.array([HAND0[0], HAND0[1], 1.17])
        self.gap, self.handle, self.d = 0.08, np.asarray(handle, float), np.asarray(d, float)
        self.held, self.sent, self.step_m = False, [], step_m

    def obs(self):
        return {"robot0_eef_pos": self.p.tolist(), "robot0_eef_quat": [1.0, 0.0, 0.0, 0.0],
                "robot0_gripper_qpos": [self.gap / 2, -self.gap / 2], "robot0_joint_pos": [0.0] * 7}

    def step(self, a):
        self.sent.append(list(a))
        mv = np.asarray(a[:3], float) * self.step_m
        at = float(np.linalg.norm(self.p - self.handle)) < 0.012
        if self.held:
            mv = self.d * max(0.0, float(mv @ self.d))  # the drawer slides along its own way only
            self.handle = self.handle + mv
        self.p = self.p + mv
        if a[6] > 0:
            self.gap = max(0.0163 if at and self.gap >= 0.0163 else 0.0, self.gap - 0.01)
            self.held = at and abs(self.gap - 0.0163) < 1e-9
        elif a[6] < 0:
            self.gap, self.held = min(0.08, self.gap + 0.01), False


def _run(world, tool, hp, cap=300, more_follow=None, **opts):
    ag = object.__new__(DR.OpenAgent)
    if more_follow is not None:
        ag.more_follow = more_follow  # the sequencer's mark on a step that others follow
    ag.lib, ag.oracle, ag.cap, ag.t = tool.lib if hasattr(tool, "lib") else None, tool, cap, 0
    ag.ap, ag.fp, ag.jaw = AG.AgentParams(open_fast=True, pull_stall_stop=True, **opts), AG.FollowParams(), Jaw()
    ag.spec, ag.target = {"part": "cabinet_middle", "target": "p6"}, "p6"
    ag.attempts, ag.events, ag.finished, ag.finish_reason = [], [], False, None
    ag.hp = hp
    ag.pobs = world.obs()
    ag._hand_R = lambda R_, d_: R0  # the stand-in's hand does not turn: every pose is asked for as it stands
    gen = ag._episode()
    next(gen)
    while not ag.finished and ag.t < cap:
        ag.pobs = world.obs()
        tool.t_now = ag.t
        try:
            a = gen.send(ag.pobs)
        except StopIteration:
            break
        if a is None:
            break
        world.step(a)
        ag.t += 1
    return ag


def _hp(c, d):
    return {"c": np.asarray(c, float), "d": np.asarray(d, float), "bar_h": 0.0163, "bar_w": 0.089, "bar_depth": 0.0154,
            "pull": 0.12, "q_now": 0.0, "far": -0.16, "n_boxes": 3}


def test_with_every_option_off_a_missed_handle_is_reached_for_three_times_at_the_same_place():
    w = World([0.14, -0.21, 1.0], [0.0, 1.0, 0.0])  # the handle is not where the plan has it
    tool = Tool([])
    ag = _run(w, tool, _hp([0.04, -0.13, 1.0], [0.0, 1.0, 0.0]))
    assert [a["outcome"] for a in ag.attempts] == ["missed"] * 3 and ag.finish_reason == "max_attempts"
    assert [round(a["grip"][2] - 1.0, 3) for a in ag.attempts] == [0.0, -0.008, 0.008]
    assert tool.asked == [] and ag.fx_used is None and all("relook" not in a for a in ag.attempts)


def test_look_again_takes_the_fixture_where_the_new_look_has_it():
    lib, percept = _goal()
    old, moved = percept([0.0335, -0.2304, 0.905], 180.0), percept([0.1335, -0.3104, 0.8984], 180.0, pid="p9")
    hp_old, hp_new = DR.handle_plan(lib, old, "cabinet_middle"), DR.handle_plan(lib, moved, "cabinet_middle")
    w = World(hp_new["c"], hp_new["d"])  # the scene moved the cabinet 13 cm after the first look
    tool = Tool([old], [{"percepts": [moved], "extra": {"rejected": []}}])
    tool.lib = lib
    ag = _run(w, tool, {**hp_old, "pull": 0.12}, open_relook=True)
    assert [a.get("outcome") for a in ag.attempts] == ["missed", "opened"] and ag.finish_reason == "opened"
    rl = ag.attempts[0]["relook"]
    assert rl["took"] == "seen" and abs(rl["moved_m"] - 0.1282) < 0.001 and tool.asked == [(True, ())]
    assert np.allclose(ag.attempts[1]["grip"], hp_new["c"], atol=1e-3)
    assert tool.percepts[0]["id"] == "p6" and tool.percepts[0]["source"] == "reopen:seen"  # its id kept, its pose new
    assert np.allclose(tool.percepts[0]["bodies"]["main"]["pos"], [0.1335, -0.3104, 0.8984])
    assert ag.fx_used is tool.percepts[0]


def test_look_again_turns_a_back_to_front_fixture_and_skips_what_already_failed():
    lib, percept = _goal()
    truth = DR.handle_plan(lib, percept(TRUE_ROOT, TRUE_YAW), "cabinet_middle")
    fit = percept(FIT_ROOT, FIT_YAW)
    w = World(truth["c"], truth["d"])
    # the new look gives the same back-to-front fit again: turned about its carcass it is the handle
    tool = Tool([fit], [{"percepts": [percept(FIT_ROOT, FIT_YAW, pid="p7")], "extra": {"rejected": []}}])
    tool.lib = lib
    w.p = np.array([HAND0[0], HAND0[1], 1.17])
    ag = _run(w, tool, DR.handle_plan(lib, fit, "cabinet_middle"), open_relook=True)
    assert ag.attempts[0]["relook"]["took"] == "seen+faced" and ag.finish_reason == "opened"
    assert float(np.linalg.norm(np.asarray(ag.attempts[1]["grip"]) - truth["c"])) < 0.006
    # no look left, and the hand could not reach: the plan's own twin
    lib2, percept2 = lib, percept
    tool = Tool([fit], [], budget=0)
    tool.lib = lib2
    rec, failed = {}, [(np.asarray([9.0, 9.0, 9.0]), np.asarray([1.0, 0.0, 0.0]))]
    ag = object.__new__(DR.OpenAgent)
    ag.lib, ag.oracle, ag.target, ag.spec, ag.t = lib2, tool, "p6", {"part": "cabinet_middle"}, 7
    ag._hand0 = HAND0.copy()
    gen = ag._look_again(rec, None, R0, np.array([0.0, 1.0, 0.0]), failed, unreachable=True)
    try:
        next(gen)
        raise AssertionError("no step is needed without a look")
    except StopIteration as e:
        hp = e.value
    assert rec["relook"]["took"] == "twin" and float(np.linalg.norm(hp["c"] - truth["c"])) < 0.006 and tool.asked == []
    # and a plan that already failed is not taken again (the tool's percept is the twin now; its own twin is the
    # first plan, which faces away from the hand)
    gen = ag._look_again({}, None, R0, np.array([0.0, 1.0, 0.0]), [(hp["c"], hp["d"])], unreachable=True)
    try:
        next(gen)
    except StopIteration as e:
        assert e.value is None
    # a plan that faced the hand and could not be reached is not turned away from it
    tool = Tool([percept([0.0335, -0.2304, 0.905], 180.0)], [], budget=0)
    tool.lib = lib
    ag.oracle = tool
    gen = ag._look_again({}, None, R0, np.array([0.0, 1.0, 0.0]), [], unreachable=True)
    try:
        next(gen)
    except StopIteration as e:
        assert e.value is None and tool.percepts[0]["source"] == "support"


def test_a_handle_the_hand_cannot_get_near_is_given_up_at_once_for_the_fixtures_twin():
    lib, percept = _goal()
    truth = DR.handle_plan(lib, percept(TRUE_ROOT, TRUE_YAW), "cabinet_middle")
    fit = percept(FIT_ROOT, FIT_YAW)

    class Wall(World):  # the arm does not reach behind the cabinet
        def step(self, a):
            before = self.p.copy()
            super().step(a)
            if self.p[1] < -0.15:
                self.p = before

    tool = Tool([fit], [{"percepts": [], "extra": {"rejected": []}}])  # the look itself has no cabinet to offer
    tool.lib = lib
    ag = _run(Wall(truth["c"], truth["d"]), tool, DR.handle_plan(lib, fit, "cabinet_middle"), open_relook=True)
    a0 = ag.attempts[0]
    assert a0["hi"] == "unreachable" and a0["outcome"] == "unreachable" and "pre" not in a0 and "gap" not in a0
    assert a0["relook"]["took"] == "twin" and a0["relook"]["unreachable"] is True
    assert ag.attempts[1]["outcome"] == "opened" and ag.finish_reason == "opened"
    assert float(np.linalg.norm(np.asarray(ag.attempts[1]["grip"]) - truth["c"])) < 0.006
    off = _run(Wall(truth["c"], truth["d"]), Tool([fit]), DR.handle_plan(lib, fit, "cabinet_middle"))
    assert [a["outcome"] for a in off.attempts if "outcome" in a][:1] == ["missed"] and off.finish_reason != "opened"
    assert ag.attempts[1]["t_start"] < off.attempts[1]["t_start"]  # and sooner than the old second attempt


def test_a_look_that_shows_the_fixture_turned_is_not_believed_and_not_asked_for_again():
    lib, percept = _goal()
    old = percept([0.0335, -0.2304, 0.905], 180.0)
    hp_old = DR.handle_plan(lib, old, "cabinet_middle")
    # what the Mac's depth line gave after the scene moved the cabinet: one accepted fit 20 cm off, 111 degrees round
    wrong = percept([0.166, -0.378, 0.9], 291.0, pid="p8")
    far = percept([0.40, -0.30, 0.905], 180.0, pid="p9")
    tool = Tool([old], [{"percepts": [wrong, far], "extra": {"rejected": []}}] * 3)
    tool.lib = lib
    ag = _run(World([0.5, 0.5, 1.0], [0.0, 1.0, 0.0]), tool, {**hp_old, "pull": 0.12}, open_relook=True)
    rl = ag.attempts[0]["relook"]
    assert rl["took"] is None and {c["no"] for c in rl["cands"]} == {"turned", "too far"}
    assert len(tool.asked) == 1 and tool.percepts[0]["source"] == "support" and ag.fx_used is None
    assert [a["outcome"] for a in ag.attempts] == ["missed"] * 4 and ag.finish_reason == "max_attempts"
    assert "seen" not in ag.attempts[1]["relook"]  # no second look: the attempts go on as before, the height varied
    assert [round(a["grip"][2] - hp_old["c"][2], 3) for a in ag.attempts] == [0.0, -0.008, 0.008, 0.008]


def test_jaws_too_far_shut_before_the_handle_are_opened_and_go_in_open():
    hp = _hp([0.04, -0.13, 1.0], [0.0, 1.0, 0.0])

    class Late(World):  # the closing commands of the shaping run on: the jaws meet before the approach
        def step(self, a):
            if 3 <= len(self.sent) < 12 and a[6] >= 0:
                a = list(a[:6]) + [1.0]
            super().step(a)

    off = _run(Late(hp["c"], hp["d"]), Tool([]), hp)
    assert off.attempts[0]["pre_gap"] < 0.02 and off.attempts[0]["outcome"] == "missed" and "reopened" not in off.attempts[0]
    on = _run(Late(hp["c"], hp["d"]), Tool([]), hp, open_gap_check=True)
    assert on.attempts[0]["pre_gap"] < 0.02 and on.attempts[0]["reopened"] > 0.06
    assert on.attempts[0]["outcome"] == "opened" and on.finish_reason == "opened"
    ok = _run(World(hp["c"], hp["d"]), Tool([]), hp, open_gap_check=True)  # jaws shaped as meant: nothing added
    assert "reopened" not in ok.attempts[0] and ok.finish_reason == "opened"
    plain = _run(World(hp["c"], hp["d"]), Tool([]), hp)
    assert [x[6] for x in ok.oracle.asked] == [] and len(ok.attempts) == len(plain.attempts) == 1
    assert ok.t == plain.t  # the same steps with and without the option when the jaws are right


def test_settle_asks_nothing_without_a_delay_and_fixes_the_approach_through_one():
    hp = _hp([0.04, -0.13, 1.0], [0.0, 1.0, 0.0])
    plain = _run(World(hp["c"], hp["d"]), Tool([]), hp)
    on = _run(World(hp["c"], hp["d"]), Tool([]), hp, open_settle=True)
    assert on.t == plain.t and "settle_fix" not in on.attempts[0]
    ag = object.__new__(DR.OpenAgent)
    ag.lag_steps = 6
    seen = {}

    def fix(p, R, grip, cap, tol_m, tol_deg):
        seen["args"] = (tuple(np.round(p, 3)), grip, tol_m)
        return {"left_m": 0.02, "status": "reached"}
        yield  # pragma: no cover

    ag._settle_fix = fix
    gen = fix(hp["c"], R0, 0.0, 0.5, 0.004, 4.0)
    try:
        next(gen)
    except StopIteration as e:
        assert e.value["left_m"] == 0.02 and seen["args"] == ((0.04, -0.13, 1.0), 0.0, 0.004)


# ------------------------------------------------------------------ the remembered place (fixture_memory)
MEM = {"wooden_cabinet": {"pos": [0.03, -0.234, 0.9076], "yaw_deg": 180.0}}
# the first looks' unexplained regions in three training episodes of the rehearsal (the rows' own numbers)
BLOB_S03 = {"xy": [-0.1091, -0.289], "z_top": 1.2478, "top": {"xy": [0.0181, -0.2599], "half_xy": [0.1234, 0.042], "z": 1.2287}}
BLOB_S07 = {"xy": [-0.1829, -0.3247], "z_top": 1.205, "top": {"xy": [0.0336, -0.2526], "half_xy": [0.1322, 0.044], "z": 1.1793}}
BLOBS_BASE = [{"xy": [0.0139, -0.1341], "z_top": 1.0241, "top": {"xy": [0.0369, -0.1251], "z": 1.0225}},
              {"xy": [-0.2226, -0.3915], "z_top": 1.1344, "top": {"xy": [-0.057, -0.3283], "z": 1.1286}}]
BLOBS_MOVED = [{"xy": [-0.003, -0.3388], "z_top": 1.1439, "top": {"xy": [0.1571, -0.275], "z": 1.135}}]


def _mem_tool(lib, blobs, memory=MEM, percepts=()):
    from rowant_lab.pro2.robot.look import PerceptionTool

    t = object.__new__(PerceptionTool)
    t.lib, t.percepts, t.extra, t.fixture_memory = lib, list(percepts), {"blobs": blobs, "rejected": []}, memory
    t.table_bound, t.behind, t.seen_target, t.fallback_top, t.blob_clear_first = False, None, False, False, False
    t.fallbacks = []
    t.stand = None  # (the appearance stream's fallback_stand, off: a real tool's __init__ sets it)
    return t


def test_a_fixture_the_look_did_not_find_is_taken_at_its_remembered_place_when_something_stands_there():
    lib, percept = _goal()
    # a cabinet on a 10 cm block at its plain place: the true root stood at (0.0335, -0.2304, 1.005), turn 180
    p = _mem_tool(lib, [BLOB_S03])._from_memory("wooden_cabinet")
    assert p["source"] == "memory+lifted" and abs(p["memory"]["dz_m"] - 0.098) < 0.003
    assert np.allclose(p["bodies"]["main"]["pos"], [0.03, -0.234, 1.0057], atol=0.002)
    truth = DR.handle_plan(lib, percept([0.0335, -0.2304, 1.005], 180.0), "cabinet_middle")
    got = DR.handle_plan(lib, p, "cabinet_middle")
    assert float(np.linalg.norm(got["c"] - truth["c"])) < 0.006 and float(got["d"] @ truth["d"]) > 0.9999
    # a region as tall as the model, or lower, lifts nothing; one that tops it is read as a lift (a taller cabinet
    # would be too: that fixture is not the model's, and nothing here makes it one)
    low = {**BLOB_S03, "top": {**BLOB_S03["top"], "z": 1.12}}
    q = _mem_tool(lib, [low])._from_memory("wooden_cabinet")
    assert q["source"] == "memory" and np.allclose(q["bodies"]["main"]["pos"], MEM["wooden_cabinet"]["pos"])
    assert _mem_tool(lib, [BLOB_S07])._from_memory("wooden_cabinet")["source"] == "memory+lifted"
    # nothing standing there (the plain look's own leftovers), a region elsewhere (a cabinet the scene moved), no
    # memory of the type, or no memory at all: the older fallbacks
    assert _mem_tool(lib, BLOBS_BASE)._from_memory("wooden_cabinet") is None
    assert _mem_tool(lib, BLOBS_MOVED)._from_memory("wooden_cabinet") is None
    assert _mem_tool(lib, [BLOB_S03], memory={"plate": MEM["wooden_cabinet"]})._from_memory("wooden_cabinet") is None
    assert _mem_tool(lib, [BLOB_S03], memory=None)._from_memory("wooden_cabinet") is None
    assert _mem_tool(lib, [BLOB_S03])._from_memory("plate") is None


def test_the_remembered_place_comes_before_a_rejected_fit_and_only_for_an_unseen_type():
    lib, percept = _goal()
    t = _mem_tool(lib, [BLOB_S03])
    rej = {**percept([-0.016, -0.31, 0.915], 244.0, pid=None), "pos": [-0.016, -0.31, 0.915],
           "fit": {"n_agree": 30, "inlier": 0.58}, "source": "support"}
    t.extra["rejected"] = [rej]
    t.on_parsed({"target_type": "wooden_cabinet", "place_type": None})
    assert [(f["role"], f["source"]) for f in t.fallbacks] == [("target", "memory+lifted")] and len(t.percepts) == 1
    assert t.percepts[0]["id"] == "f1" and t.percepts[0]["type"] == "wooden_cabinet"
    off = _mem_tool(lib, [BLOB_S03], memory=None)
    off.extra["rejected"] = [dict(rej)]
    off.on_parsed({"target_type": "wooden_cabinet", "place_type": None})
    assert [f["source"] for f in off.fallbacks] == ["rejected:support"]  # as before without the option
    seen = _mem_tool(lib, [BLOB_S03], percepts=[percept([0.0335, -0.2304, 0.905], 180.0)])
    seen.on_parsed({"target_type": "wooden_cabinet", "place_type": None})
    assert seen.fallbacks == [] and len(seen.percepts) == 1 and seen.percepts[0]["source"] == "support"


def test_a_rejected_fit_that_agrees_with_the_remembered_place_comes_before_it():
    lib, percept = _goal()
    low = {**BLOB_S03, "top": {**BLOB_S03["top"], "z": 1.12}}  # the fixture at the model's own height: plain memory
    near = {**percept([0.045, -0.225, 0.906], 176.0, pid=None), "pos": [0.045, -0.225, 0.906],
            "fit": {"n_agree": 400, "inlier": 0.72}, "source": "support"}
    far = {**percept([-0.016, -0.31, 0.915], 244.0, pid=None), "pos": [-0.016, -0.31, 0.915],
           "fit": {"n_agree": 400, "inlier": 0.72}, "source": "support"}

    def run(rej, mem):
        t = _mem_tool(lib, [low], memory={"wooden_cabinet": mem})
        t.extra["rejected"] = [dict(rej)]
        t.on_parsed({"target_type": "wooden_cabinet", "place_type": None})
        return [f["source"] for f in t.fallbacks]

    plain, first = MEM["wooden_cabinet"], {**MEM["wooden_cabinet"], "fit_first_m": 0.04}
    assert run(near, plain) == ["memory"]  # without the key: the remembered place, as built
    assert run(near, first) == ["rejected:support"]  # 1.7 cm and 4 degrees from the memory: the scene's own fit
    assert run(far, first) == ["memory"]  # a fit elsewhere, a quarter turn round: not the fixture
    turned = {**near, **percept([0.045, -0.225, 0.906], 150.0, pid=None), "pos": [0.045, -0.225, 0.906],
              "fit": {"n_agree": 400, "inlier": 0.72}, "source": "support"}
    assert run(turned, first) == ["memory"]  # at the place but 30 degrees round
    lifted = _mem_tool(lib, [BLOB_S03], memory={"wooden_cabinet": first})  # a lifted fixture: the memory, lifted
    lifted.extra["rejected"] = [dict(near)]
    lifted.on_parsed({"target_type": "wooden_cabinet", "place_type": None})
    assert [f["source"] for f in lifted.fallbacks] == ["memory+lifted"]


# ------------------------------------------------------------------ the pull (2026-11-17)
def _pull_first(d, lag=0, **opts):
    ag = object.__new__(DR.OpenAgent)
    ag.ap, ag.fp = AG.AgentParams(**opts), AG.FollowParams()
    ag.lag_steps = lag
    p0 = np.array([0.0771, -0.0939, 1.0217])
    ag._pose = lambda: (p0, R0)
    gen = ag._pull(p0 + np.asarray(d, float) * 0.1746, R0, np.asarray(d, float), 0.1746, cap=1.0)
    return np.asarray(next(gen)[:3], float)


def test_a_pull_is_sent_straight_along_the_part_when_asked_and_along_the_axes_diagonal_as_before():
    d = np.array([-0.329, 0.944, 0.0])  # a drawer that opens along 109 degrees (the training episode's own plan)
    old = _pull_first(d)
    assert np.allclose(old[:2], [-0.7071, 0.7071], atol=1e-3)  # 135 degrees: 26 off the part's way
    new = _pull_first(d, open_pull_straight=True)
    assert abs(math.degrees(math.atan2(new[1], new[0])) - math.degrees(math.atan2(d[1], d[0]))) < 0.01
    assert abs(float(np.abs(new).max()) - 1.0) < 1e-9 and abs(new[2]) < 1e-9
    assert np.allclose(_pull_first(d, lag=6, open_pull_straight=True), old)  # through a delay: as before
    axis = np.array([0.0, 1.0, 0.0])  # a part that opens along a table axis: the same command either way
    assert np.allclose(_pull_first(axis), _pull_first(axis, open_pull_straight=True))


class Slippy(World):
    """The bar leaves the fingers once the drawer has come ``hold_m`` out: the jaws go on to ``left`` apart and the
    drawer stays where it is."""

    def __init__(self, handle, d, hold_m=0.05, left=0.0072, once=True):
        super().__init__(handle, d)
        self.h0, self.hold_m, self.left, self.once, self.lost = np.asarray(handle, float).copy(), hold_m, left, once, False

    def step(self, a):
        if self.held and float((self.handle - self.h0) @ self.d) >= self.hold_m and not (self.once and self.lost):
            self.held, self.lost, self.gap = False, True, self.left
            self.sent.append(list(a))
            self.p = self.p + np.asarray(a[:3], float) * 0.05
            return
        if self.lost and not self.held and a[6] > 0 and self.gap <= self.left + 1e-9 \
                and float(np.linalg.norm(self.p - self.handle)) >= 0.012:
            self.sent.append(list(a))  # the jaws stay where the slip left them while the hand travels on
            self.p = self.p + np.asarray(a[:3], float) * 0.05
            return
        super().step(a)


def test_fingers_that_end_the_pull_under_the_bars_height_have_not_opened_the_part():
    lib, percept = _goal()
    shut = percept([0.0335, -0.2304, 0.905], 180.0)
    hp = DR.handle_plan(lib, shut, "cabinet_middle")
    # as before: the hand travelled its pull, the jaws are still apart: "opened" (the drawer stands 5 cm out)
    off = _run(Slippy(hp["c"], hp["d"]), Tool([shut]), dict(hp), pull_end_early=True)
    assert off.attempts[0]["outcome"] == "opened" and off.attempts[0]["gap_end"] < 0.01 and len(off.attempts) == 1
    # the check alone: "slipped", and the next attempt goes where the hand's travel puts the handle (the old guess)
    tool = Tool([shut])
    tool.lib = lib
    chk = _run(Slippy(hp["c"], hp["d"]), tool, dict(hp), pull_end_early=True, open_slip_check=True)
    assert chk.attempts[0]["outcome"] == "slipped" and tool.asked == []
    assert float((np.asarray(chk.attempts[1]["grip"]) - hp["c"]) @ hp["d"]) > 0.10  # far past where the drawer stands
    # with the look: the part's joint as the look reads it; the handle is taken again there for the rest of the way
    out5 = percept([0.0335, -0.2304, 0.905], 180.0, pid="p7", joints={"cabinet_middle": -0.05})
    tool = Tool([shut], [{"percepts": [out5], "extra": {"rejected": []}}])
    tool.lib = lib
    w = Slippy(hp["c"], hp["d"])
    ag = _run(w, tool, dict(hp), pull_end_early=True, open_slip_check=True, open_relook=True)
    a0, a1 = ag.attempts[0], ag.attempts[1]
    assert a0["outcome"] == "slipped" and a0["relook"]["took"] == "seen" and len(tool.asked) == 1
    assert abs(float((np.asarray(a1["grip"]) - hp["c"]) @ hp["d"]) - 0.05) < 0.002 and abs(a1["pull"] - 0.125) < 0.002
    assert a1["outcome"] == "opened" and ag.finish_reason == "opened" and a1["gap_end"] > 0.012
    assert float((w.handle - w.h0) @ w.d) > 0.14  # the drawer came the rest of its way


def test_the_whole_travel_is_pulled_whatever_the_look_read_of_the_joint():
    lib, percept = _goal()
    shut = percept([0.0335, -0.2304, 0.905], 180.0)
    hp = DR.handle_plan(lib, shut, "cabinet_middle")
    misread = {**hp, "pull": 0.1422}  # the look read the drawer 3.3 cm out; it is shut
    tool = Tool([shut])
    tool.lib = lib

    def run(**opts):
        w = World(hp["c"], hp["d"], step_m=0.01)
        ag = _run(w, tool, dict(misread), pull_end_early=True, **opts)
        ag_travel = float((w.handle - np.asarray(hp["c"], float)) @ np.asarray(hp["d"], float))
        return ag, ag_travel

    off, t_off = run()
    assert off.attempts[0]["outcome"] == "opened" and t_off < 0.14 and "reach" not in off.attempts[0]
    ag = object.__new__(DR.OpenAgent)
    ag.lib, ag.ttype, ag.spec = lib, "wooden_cabinet", {"part": "cabinet_middle"}
    assert abs(ag._full_travel(misread) - 0.175) < 1e-9
    ag.spec = {"part": "no_such_part"}
    assert ag._full_travel(misread) == 0.1422  # a joint that cannot be read: the plan's own pull
    w = World(hp["c"], hp["d"], step_m=0.01)
    full = _run_typed(w, tool, dict(misread), lib, pull_end_early=True, open_pull_full=True)
    assert full.attempts[0]["reach"] == 0.175 and full.attempts[0]["outcome"] == "opened"
    assert float((w.handle - np.asarray(hp["c"], float)) @ np.asarray(hp["d"], float)) > 0.14


def test_an_episode_that_goes_on_after_the_opening_has_a_part_that_is_not_open_yet():
    lib, percept = _goal()
    shut = percept([0.0335, -0.2304, 0.905], 180.0)
    hp = DR.handle_plan(lib, shut, "cabinet_middle")
    misread = {**hp, "pull": 0.1422}  # the hand will stop 12.2 cm out and call it opened; the task wants 14

    def run(more_follow=None, **opts):
        out12 = percept([0.0335, -0.2304, 0.905], 180.0, pid="p7", joints={"cabinet_middle": -0.122})
        tool = Tool([shut], [{"percepts": [out12], "extra": {"rejected": []}}])
        tool.lib = lib
        w = World(hp["c"], hp["d"], step_m=0.01)
        ag = _run(w, tool, dict(misread), pull_end_early=True, more_follow=more_follow, **opts)
        return ag, tool, float((w.handle - np.asarray(hp["c"], float)) @ np.asarray(hp["d"], float))

    off, tool, travel = run()
    assert [a["outcome"] for a in off.attempts] == ["opened"] and travel < 0.14 and tool.asked == []
    on, tool, travel = run(open_verify=True, open_relook=True)
    assert [a["outcome"] for a in on.attempts] == ["not open", "opened"] and on.finish_reason == "opened"
    assert on.attempts[0]["relook"]["took"] == "seen" and len(tool.asked) == 1 and travel > 0.14
    assert abs(float((np.asarray(on.attempts[1]["grip"]) - hp["c"]) @ hp["d"]) - 0.122) < 0.002
    # a step that others follow (the drawer test task: the object is put in afterwards) is not held up
    seq, tool, travel = run(more_follow=True, open_verify=True, open_relook=True)
    assert [a["outcome"] for a in seq.attempts] == ["opened"] and tool.asked == [] and seq.t == off.t
    # without a look to ask: the next attempt goes where the hand's travel puts the handle (the old guess, 2 cm short)
    nolook, tool, travel = run(open_verify=True)
    assert [a["outcome"] for a in nolook.attempts][:1] == ["not open"] and tool.asked == []
    back = float((np.asarray(nolook.attempts[1]["grip"]) - hp["c"]) @ hp["d"])
    assert abs(back - (nolook.attempts[0]["moved"] - 0.02)) < 0.002


def _run_typed(world, tool, hp, lib, **opts):
    """``_run`` with the agent's type set (``open_pull_full`` reads the type's joint)."""
    orig = DR.OpenAgent._full_travel

    def full(self, hp_):
        self.lib, self.ttype = lib, "wooden_cabinet"
        return orig(self, hp_)

    DR.OpenAgent._full_travel = full
    try:
        return _run(world, tool, hp, **opts)
    finally:
        DR.OpenAgent._full_travel = orig
