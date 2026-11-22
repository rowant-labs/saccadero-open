"""2026-11-17: the opening step's own steps and its hand-overs (drawer.OpenAgent: open_pre_pass, open_close_early_m,
open_release_early, open_retreat_pass_m, open_box, open_out_first, open_pre_brake_kz; SequenceAgent: chain; the staged
plan's overhang_open_need; follow: box_rot and each). All off by default."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from rowant_lab.libero_env.tests import test_drawer_open as TD
from rowant_lab.libero_env.tests import test_sequence as TS
from rowant_lab.pro2.robot import agent as AG
from rowant_lab.pro2.robot import drawer as DR
from rowant_lab.pro2.robot import sequence as SQ
from rowant_lab.pro2.robot.follow import follow

HANDLE, D = np.array([0.03, -0.13, 1.09]), np.array([0.0, 1.0, 0.0])
NEW = dict(open_pre_pass=True, open_close_early_m=0.012, open_release_early=2, open_retreat_pass_m=0.03)


class Slow(TD.World):
    """The stand-in's world with jaws that begin to part three steps after the first open command (the
    simulator's: the finger command ramps), so a bar is held for two steps of open commands."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.n_open = 0

    def step(self, a):
        if a[6] < 0 and self.held:
            self.n_open += 1
            if self.n_open < 3:  # the command is on its way: the hold stands, the hand's move is the drawer's
                self.sent.append(list(a))
                mv = self.d * max(0.0, float((np.asarray(a[:3], float) * self.step_m) @ self.d))
                self.handle = self.handle + mv
                self.p = self.p + mv
                return
        if a[6] > 0:
            self.n_open = 0
        super().step(a)


def _open(world_cls=TD.World, step_m=0.01, start=None, **opts):
    w = world_cls(HANDLE, D, step_m=step_m)
    if start is not None:
        w.p = np.asarray(start, float)
    ag = TD._run(w, TD.Tool([]), TD._hp(HANDLE, D), pull_end_early=True, **opts)
    return ag, w


def test_with_the_new_options_at_their_defaults_the_opening_is_sent_as_before():
    a, wa = _open()
    b, wb = _open(open_pre_pass=False, open_close_early_m=0.0, open_release_early=0, open_retreat_pass_m=0.0,
                  open_box=False, open_out_first=False, open_pre_brake_kz=0.0)
    assert wa.sent == wb.sent and a.attempts[0]["outcome"] == "opened"
    assert a.attempts[0]["pre"] == "reached" and "close_early" not in a.attempts[0]
    assert "released_early" not in a.attempts[0] and not [e for e in a.events if e["ev"] in ("pass", "retreat")]


def test_the_point_before_the_handle_is_passed_once_the_hand_stands_on_the_line_in():
    old, w0 = _open()
    new, w1 = _open(open_pre_pass=True)
    att = new.attempts[0]
    assert att["pre"] == "passed" and att["outcome"] == "opened" and len(w1.sent) < len(w0.sent)
    ev = [e for e in new.events if e["ev"] == "pass"]
    assert len(ev) == 1 and ev[0]["s"] >= 0.025 and ev[0]["dz"] <= 0.010 and ev[0]["dx"] <= 0.020
    assert AG.AgentParams().open_pre_pass is False and AG.AgentParams().open_pre_lat_m == 0.010


def test_the_fingers_close_on_the_way_in_and_the_hold_is_taken_when_the_gap_stops_at_the_bars_height():
    old, w0 = _open()
    new, w1 = _open(open_close_early_m=0.012)
    a0, a1 = old.attempts[0], new.attempts[0]
    assert a1["outcome"] == "opened" and 0.0 < a1["close_early"] <= 0.012 and abs(a1["gap"] - 0.0163) < 1e-6
    assert (a1["t_pulled"] - a1["t_pre"]) < (a0["t_pulled"] - a0["t_pre"])  # in, close and pull: fewer steps
    # no bar there: the fingers meet, and the attempt is a miss as before (never "holding" on nothing)
    w = TD.World(HANDLE + np.array([0.0, -0.2, 0.0]), D, step_m=0.01)  # the true handle is 20 cm away
    ag = TD._run(w, TD.Tool([]), TD._hp(HANDLE, D), pull_end_early=True, open_close_early_m=0.012)
    assert [a["outcome"] for a in ag.attempts] == ["missed"] * 3
    ag2 = object.__new__(DR.OpenAgent)
    assert ag2._holding([0.05, 0.03, 0.0163, 0.0163], 0.0163) and not ag2._holding([0.03, 0.0163], 0.0163)
    assert not ag2._holding([0.0, 0.0], 0.0163) and not ag2._holding([0.03, 0.03], 0.0163)  # shut, or not there yet


def test_the_open_command_is_sent_two_steps_before_the_pulls_end_and_the_bar_is_held_to_the_end():
    old, w0 = _open(Slow)
    new, w1 = _open(Slow, open_release_early=2)
    a0, a1 = old.attempts[0], new.attempts[0]
    assert a0["outcome"] == a1["outcome"] == "opened" and a1["released_early"] == 2 and "released_early" not in a0
    travel = lambda w: float((w.handle - HANDLE) @ D)  # noqa: E731
    assert abs(travel(w0) - travel(w1)) < 0.011 and travel(w1) > 0.09  # the drawer came as far (within a step)
    assert len(w1.sent) == len(w0.sent) - 2  # two steps of standing still with the hand open are gone
    # the pull's own last two commands carry "open", and four open commands in all come before the backing off
    k = max(i for i, a in enumerate(w1.sent) if a[6] > 0)
    assert [a[6] for a in w1.sent[k + 1:k + 5]] == [-1.0] * 4 and w1.sent[k + 1][1] > 0.5 and w1.sent[k + 2][1] > 0.5
    # through a delay, or with no early end of the pull, nothing is let go early
    ag = object.__new__(DR.OpenAgent)
    ag.ap, ag.fp = AG.AgentParams(), AG.FollowParams()
    p = [np.array([0.0, 0.0, 1.0])]
    ag._pose = lambda: (p[0], TD.R0)
    gen = ag._pull(p[0] + D * 0.17, TD.R0, D, 0.17, cap=1.0, done_at=None, release_early=2)
    for _ in range(12):
        a = next(gen)
        p[0] = p[0] + D * 0.011
        assert a[6] == 1.0
    assert ag._n_released == 0


def test_the_hand_backs_off_only_until_the_jaws_are_clear_of_the_bar():
    old, w0 = _open()
    new, w1 = _open(open_retreat_pass_m=0.03)
    ev = [e for e in new.events if e["ev"] == "retreat"]
    assert new.attempts[0]["outcome"] == "opened" and len(ev) == 1 and 0.03 <= ev[0]["out"] < 0.045
    assert len(w1.sent) < len(w0.sent)
    assert not [e for e in old.events if e["ev"] == "retreat"]


def test_a_hand_that_begins_over_the_fixture_goes_out_at_its_height_before_it_comes_down():
    over = HANDLE + np.array([-0.01, -0.037, 0.158])  # where the staging step leaves it: above the top, behind the bar

    def track(**opts):
        w = TD.World(HANDLE, D, step_m=0.01)
        w.p = over.copy()
        seen = []
        step = w.step

        def spy(a):
            step(a)
            seen.append(w.p.copy())
        w.step = spy
        ag = TD._run(w, TD.Tool([]), TD._hp(HANDLE, D), pull_end_early=True, **opts)
        return ag, np.array(seen)

    old, p0 = track()
    new, p1 = track(open_out_first=True)
    assert old.attempts[0]["outcome"] == new.attempts[0]["outcome"] == "opened" and new.attempts[0]["hi"] == "out"
    # the stand-in's hand goes where it is sent; the simulator's lags sideways and came down 1.5 cm before the grip
    # point, 0.6 mm from the bar by the library's own boxes (Mac, the plain drawer task): what is asked is kept here
    s0, s1 = (p0 - HANDLE) @ D, (p1 - HANDLE) @ D
    k = int(np.argmax(s1 >= 0.03))
    assert np.all(p1[:k, 2] > over[2] - 0.01) and p0[k, 2] < over[2] - 0.03  # level until 3 cm out; as built: down at once
    kp = new.attempts[0]["t_pre"]  # until the move in to the handle begins: never low while nearer than 3 cm
    assert np.all(s1[:kp][p1[:kp, 2] < HANDLE[2] + 0.08] >= 0.03)
    ev = [e for e in new.events if e["ev"] == "out"]
    assert len(ev) == 1 and ev[0]["s"] >= 0.03
    # a hand that begins far off (the practice task: from its start pose) goes by the high point as before
    far, _w = _open(open_out_first=True)
    assert far.attempts[0]["hi"] == "reached" and not [e for e in far.events if e["ev"] == "out"]


def test_all_four_together_open_the_part_in_fewer_steps():
    old, w0 = _open(Slow)
    new, w1 = _open(Slow, **NEW)
    assert old.attempts[0]["outcome"] == new.attempts[0]["outcome"] == "opened"
    assert len(w1.sent) <= len(w0.sent) - 8
    assert abs(float((w1.handle - w0.handle) @ D)) < 0.011


def test_a_slanted_turn_asks_each_axis_for_its_own_full_step_when_boxed():
    u = np.ones(3) / math.sqrt(3.0)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    R_tgt = np.eye(3) + math.sin(2.0) * K + (1 - math.cos(2.0)) * (K @ K)  # 115 degrees about the diagonal
    p = np.zeros(3)
    old = np.asarray(follow(p, np.eye(3), p, R_tgt, -1.0, 1.0, 1.0)[3:6])
    new = np.asarray(follow(p, np.eye(3), p, R_tgt, -1.0, 1.0, 1.0, box_rot=True)[3:6])
    assert abs(np.linalg.norm(old) - 1.0) < 1e-9 and np.allclose(old, u, atol=1e-6)
    assert np.allclose(new, [1.0, 1.0, 1.0], atol=1e-6)  # the same axis, 1.73 times the step
    Rx = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # a quarter turn about one axis: the same either way
    assert np.allclose(follow(p, np.eye(3), p, Rx, -1.0)[3:6], follow(p, np.eye(3), p, Rx, -1.0, box_rot=True)[3:6])
    small = np.eye(3) + math.sin(0.05) * K + (1 - math.cos(0.05)) * (K @ K)  # near its goal: no cap acts
    assert np.allclose(follow(p, np.eye(3), p, small, -1.0)[3:6], follow(p, np.eye(3), p, small, -1.0, box_rot=True)[3:6])


class Step:
    """An agent that moves for ``n`` steps and then ends as the robot's agents do: its last action is the idle one."""

    def __init__(self, n, tag):
        self.n, self.tag, self.i, self.finished = n, tag, 0, False

    def act(self, obs):
        self.i += 1
        if self.i > self.n:
            self.finished = True
            return [0.0] * 6 + [-1.0]
        return [self.tag] * 6 + [1.0]

    def summary(self):
        return {"steps": self.i, "finish_reason": "placed"}


def test_the_next_step_begins_in_the_control_step_the_one_before_it_ended_in():
    def run(**kw):
        tags = iter([0.1, 0.2, 0.3])
        ag = SQ.SequenceAgent(["a", "b", "c"], lambda c, cap, ex: {"agent": Step(3, next(tags)), "grounding": {}},
                              cap=300, **kw)
        acts = []
        while not ag.finished and len(acts) < 40:
            acts.append(ag.act({}))
        return ag, [a[0] for a in acts]

    old, a0 = run()
    assert a0 == [0.1] * 3 + [0.0] + [0.2] * 3 + [0.0] + [0.3] * 3 + [0.0]  # an idle step at every hand-over
    new, a1 = run(chain=True)
    assert a1 == [0.1] * 3 + [0.2] * 3 + [0.3] * 3 + [0.0] and new.t == old.t - 2  # only the last step's idle is left
    assert [r.get("chained") for r in new.records] == [None, True, True]
    assert [r["summary"]["finish_reason"] for r in new.records] == ["placed"] * 3
    assert not any("chained" in r for r in old.records)


def test_a_step_that_cannot_be_planned_is_not_chained_into():
    def planner(c, cap, ex):
        return {"agent": Step(2, 0.5), "grounding": {}} if c == "a" else {"agent": None, "grounding": {}}

    ag = SQ.SequenceAgent(["a", "b"], planner, cap=300, chain=True)
    acts = [ag.act({})[0] for _ in range(3)]
    assert acts == [0.5, 0.5, 0.0] and ag.finished and ag.records[1]["skipped"] == "no plan"


def test_the_staged_plan_keeps_only_what_the_opening_needs_when_asked():
    from rowant_lab.pro2.robot import episode as EPI

    scene = TS._goal_scene()
    if scene is None:
        pytest.skip("the long-lib tree is not on this machine")
    lib, percept, cab = scene
    steps = SQ.split("open the top layer of the drawer and put the bowl inside")
    bowl = percept("p0", "akita_black_bowl", np.eye(3), [-0.085, 0.004, 0.898])
    tool = SimpleNamespace(percepts=[bowl, cab], target=None)
    old = EPI._overhang_rewrite(steps, tool, lib, clear=False)
    assert old[1] == steps[0] and old == EPI._overhang_rewrite(steps, tool, lib, clear=False, open_need=0)
    new = EPI._overhang_rewrite(steps, tool, lib, clear=False, open_need=95)
    assert new[0] == old[0] and new[2] == old[2]
    assert new[1] == {"kind": "clause", "as_written": True, "clause": steps[0], "need": 95}
    caps = {}

    def planner(clause, cap, exclude):
        caps[clause["clause"] if isinstance(clause, dict) else clause] = cap
        return {"agent": Step(5, 0.1), "grounding": {}}

    SQ.SequenceAgent(old, planner, cap=300, reserve=130)
    stage = old[0]["clause"]
    assert caps[stage] == 300 - 130 - EPI.SWEEP_NEED  # as built: 98 steps for a step that takes about 90
    caps.clear()
    ag = SQ.SequenceAgent(new, planner, cap=300, reserve=130)
    assert caps[stage] == 300 - 95 - EPI.SWEEP_NEED
    assert ag.records[0]["clause"] == stage and ag._retriable(new[1], {"summary": {"finish_reason": "missed"}})


def test_the_way_down_to_the_point_before_the_handle_brakes_in_height_when_asked():
    ag = object.__new__(DR.OpenAgent)
    ag.ap = AG.AgentParams(brake_k=1.5)
    ag._v_step = np.array([0.002, -0.004, -0.0125])  # coming down 12.5 mm a step
    tgt = np.array([0.03, -0.07, 1.09])
    assert np.allclose(ag._braked(tgt), tgt - np.array([0.003, -0.006, 0.0]))  # the agent's own: sideways only
    ag._brake_kz = 1.5
    assert np.allclose(ag._braked(tgt), tgt - np.array([0.003, -0.006, -0.01875]))  # 1.9 cm short of it in height
    ag.lag_steps = 6
    assert np.allclose(ag._braked(tgt), tgt)  # through a delay: no brake of any kind, as before
    # the option switches it on for that one move and off again, also when the move is passed
    new, w1 = _open(open_pre_brake_kz=1.5, open_pre_pass=True, brake_k=1.5)
    assert new.attempts[0]["outcome"] == "opened" and new._brake_kz is None
    old, w0 = _open(brake_k=1.5)
    assert old._brake_kz is None and old.attempts[0]["outcome"] == "opened"


def test_each_axis_capped_on_its_own_still_corrects_a_small_error_across_a_long_move():
    p, tgt = np.array([0.0, 0.0, 1.25]), np.array([0.0, 0.026, 1.10])  # 15 cm down, 2.6 cm aside
    old = np.asarray(follow(p, np.eye(3), tgt, np.eye(3), 0.0)[:3])
    box = np.asarray(follow(p, np.eye(3), tgt, np.eye(3), 0.0, box=True)[:3])
    each = np.asarray(follow(p, np.eye(3), tgt, np.eye(3), 0.0, each=True)[:3])
    assert np.allclose(box, [0.0, 0.026 / 0.15, -1.0], atol=1e-6)  # the direction kept: a sixth of a step aside
    assert np.allclose(each, [0.0, 1.0, -1.0])  # a full step aside and a full step down
    assert np.allclose(old, np.array([0.0, 1.0, -1.0]) / math.sqrt(2.0))  # as built: both, sharing one step
    near = np.array([0.0, 0.004, 1.105])  # near the goal no cap acts and the three agree
    assert np.allclose(follow(near, np.eye(3), tgt, np.eye(3), 0.0)[:3], follow(near, np.eye(3), tgt, np.eye(3), 0.0, each=True)[:3])
    assert np.allclose(follow(p, np.eye(3), tgt, np.eye(3), 0.0, 0.5, each=True)[:3], [0.0, 0.5, -0.5])


def test_where_these_options_meet_the_touch_and_the_gentle_pull_they_give_way():
    """With SCALE's ``open_feel`` on as well: an attempt that finds the part's front by touch goes in with its jaws
    open and closes where the touch puts them (no early close, the built hold rule), and a gentle pull lets go as
    built (no early release). The pass, the short backing off and everything on the untouched attempts stay."""
    from rowant_lab.libero_env.tests import test_sized as TZ

    d = [0.0, 1.0, 0.0]
    true = np.array([0.04, -0.152, 1.0])  # the bar 2 cm deeper than the plan has it
    plan = TD._hp([0.04, -0.132, 1.0], d)
    alone = TD._run(TZ.Fronted(true, d), TD.Tool([]), dict(plan), open_feel=True)
    both = TD._run(TZ.Fronted(true, d), TD.Tool([]), dict(plan), open_feel=True, **NEW)
    assert [a.get("outcome") for a in both.attempts] == [a.get("outcome") for a in alone.attempts] == ["missed", "opened"]
    a1 = both.attempts[1]
    assert "feel" in a1 and "close_early" not in a1 and a1["approach"] != "held"  # the touch's attempt: mine stand back
    assert abs(a1["feel"]["grip"][1] - alone.attempts[1]["feel"]["grip"][1]) < 1e-6  # the touch finds what it found alone
    assert a1.get("gentle") is True and "released_early" not in a1  # its pull is the gentle one, let go as built
    assert both.attempts[0]["pre"] == "passed" and [e for e in both.events if e["ev"] == "retreat"]  # these stay


# ------------------------------------------------------------------ second request: the staging's release, the take-again
def _stub(cls, **opts):
    ag = object.__new__(cls)
    ag.ap, ag.fp = AG.AgentParams(**opts), AG.FollowParams()
    ag.attempts, ag.events, ag.t, ag.finished, ag.finish_reason = [], [], 0, False, None
    ag._ev = lambda name, **k: ag.events.append({"ev": name, **k})
    return ag


def test_a_staging_release_opens_on_its_way_down_when_asked():
    ag = _stub(AG.OracleAgent, stage_release_early=3)
    z = [1.1995]
    ag._pose = lambda: (np.array([0.0, 0.0, z[0]]), np.eye(3))
    ag._v_step = np.array([0.0, 0.0, -0.006])

    def down():
        for _ in range(6):
            yield [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]
        return "reached"

    gen, sent = ag._opening_early(down(), np.array([0.0, 0.0, 1.170]), 3), []
    try:
        while True:
            a = next(gen)
            sent.append(a[6])
            z[0] -= 0.006
    except StopIteration as stop:
        assert stop.value == "reached"
    # 3 cm above at 6 mm a step: held until within three steps (1.8 cm) of the release height, then open to the end
    # (the third command goes out 1.75 cm above it)
    assert sent == [1.0, 1.0, -1.0, -1.0, -1.0, -1.0] and ag._n_open_early == 4
    assert AG.AgentParams().stage_release_early == 0 and AG.OracleAgent._n_open_early == 0


def _take2(start, gaps=None, **kw):
    ag = _stub(DR.RetakeAgent, **kw)
    ag.hp = {"p": [0.02, -0.17, 1.19], "R": np.eye(3).tolist(), "drop": 0.03, "move": [0.0, 0.13]}
    ag.jaw = type("J", (), {"L": 0.04})()
    asked, g = [], iter(gaps or [])

    def goto(p, R_, grip, cap, tol_m, tol_deg, max_steps=None, **k):
        asked.append((np.asarray(p, float).copy(), tol_m))
        yield [0.0] * 6 + [grip]
        return "reached"

    ag._goto = goto
    ag._pose = lambda: (np.asarray(start, float), np.eye(3))
    ag._gap = lambda: next(g, 0.008)
    ag._v_step = np.zeros(3)
    ag._act_to = lambda p, R_, gr, cap=None: [0.0] * 6 + [float(gr)]
    ag._finish = lambda reason: setattr(ag, "finish_reason", reason)
    gen, acts = ag._episode(), []
    next(gen)
    try:
        while True:
            acts.append(next(gen))
    except StopIteration:
        pass
    return asked, acts, ag


def test_the_take_agains_lift_is_the_first_of_its_carry_and_its_hold_is_taken_at_once_when_asked():
    old, a0, ag0 = _take2([0.02, -0.17, 1.21])
    new, a1, ag1 = _take2([0.02, -0.17, 1.21], take_lift_carry=True)
    assert ag0.finish_reason == ag1.finish_reason == "placed"
    assert len(new) == len(old) - 1 and np.allclose(new[2][0], old[3][0])  # over, down, (lift,) carry: the same goal
    assert abs(old[2][0][2] - (1.19 - 0.03 + 0.012)) < 1e-9 and abs(new[2][0][2] - old[2][0][2]) < 1e-9
    # the hold: the gap stops on the rim (9 mm) at its fourth reading
    g = [0.078, 0.078, 0.05, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009]
    n_close = lambda acts: sum(1 for a in acts if a[6] == 1.0 and a[:6] == [0.0] * 6)  # noqa: E731
    slow = n_close(_take2([0.02, -0.17, 1.21], gaps=list(g))[1])
    quick = n_close(_take2([0.02, -0.17, 1.21], gaps=list(g), take_hold_quick=True)[1])
    assert quick == slow - 1
    # fingers that stand still wide open (their command still on its way) are not a hold
    wide = _take2([0.02, -0.17, 1.21], gaps=[0.078, 0.078, 0.078, 0.05, 0.009, 0.009, 0.009, 0.009], take_hold_quick=True)
    assert wide[2].finish_reason == "placed" and n_close(wide[1]) >= 4


def test_the_take_agains_way_down_ends_on_height_and_its_carry_lets_go_on_the_way_when_asked():
    ag = _stub(DR.RetakeAgent, take_down_lat_m=0.012)
    pose = [np.array([0.030, -0.170, 1.165]), np.eye(3)]
    ag._pose = lambda: (pose[0], pose[1])
    p_g = np.array([0.020, -0.170, 1.160])

    def forever():
        while True:
            yield [0.0] * 6 + [-1.0]

    gen = ag._down_lat(forever(), p_g, np.eye(3), 0.012)
    with pytest.raises(StopIteration) as stop:  # 1 cm aside and 5 mm up: there
        next(gen)
    assert stop.value.value == "reached" and ag.events[-1]["ev"] == "down"
    pose[0] = np.array([0.036, -0.170, 1.165])  # 1.6 cm aside: not yet
    gen = ag._down_lat(forever(), p_g, np.eye(3), 0.012)
    assert next(gen)[6] == -1.0
    # the carry: 13 cm at 12 mm a step to within 2 cm; the open command from two steps before that
    ag2 = _stub(DR.RetakeAgent)
    y = [0.0]
    ag2._pose = lambda: (np.array([0.0, y[0], 1.17]), np.eye(3))

    def carry():
        while 0.13 - y[0] > 0.02:
            yield [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        return "reached"

    count, sent = [0], []
    gen = ag2._release_on_the_way(carry(), np.array([0.0, 0.13, 1.17]), 0.02, 2, count)
    try:
        while True:
            sent.append(next(gen)[6])
            y[0] += 0.012
    except StopIteration as stop:
        assert stop.value == "reached"
    assert sent[-2:] == [-1.0, -1.0] and sent[:-2] == [1.0] * (len(sent) - 2) and count[0] == 2


def test_an_opening_that_a_take_again_follows_backs_off_already_turning_when_asked():
    turn = math.radians(90.0)
    R_next = TD.R0 @ np.array([[1.0, 0, 0], [0, math.cos(turn), -math.sin(turn)], [0, math.sin(turn), math.cos(turn)]])

    def run(world=TD.World, **opts):
        w = world(HANDLE, D, step_m=0.01)
        ag = object.__new__(DR.OpenAgent)
        ag.more_follow, ag.next_R = True, R_next.tolist()
        ag.lib, ag.oracle, ag.cap, ag.t = None, TD.Tool([]), 300, 0
        ag.ap, ag.fp, ag.jaw = AG.AgentParams(open_fast=True, pull_stall_stop=True, pull_end_early=True, **opts), AG.FollowParams(), TD.Jaw()
        ag.spec, ag.target = {"part": "cabinet_middle", "target": "p6"}, "p6"
        ag.attempts, ag.events, ag.finished, ag.finish_reason = [], [], False, None
        ag.hp, ag.pobs = TD._hp(HANDLE, D), w.obs()
        ag._hand_R = lambda R_, d_: TD.R0
        gen = ag._episode()
        next(gen)
        while not ag.finished and ag.t < 300:
            ag.pobs = w.obs()
            try:
                a = gen.send(ag.pobs)
            except StopIteration:
                break
            w.step(a)
            ag.t += 1
        return ag, w

    off, w0 = run(open_retreat_pass_m=0.03)
    on, w1 = run(open_retreat_pass_m=0.03, take_turn_early=True)
    assert off.attempts[0]["outcome"] == on.attempts[0]["outcome"] == "opened" and "turn_early" not in off.attempts[0]
    assert on.attempts[0]["turn_early"] is True and len(w1.sent) == len(w0.sent)
    turning = lambda w: [a for a in w.sent if float(np.linalg.norm(a[3:6])) > 0.5]  # noqa: E731
    assert not turning(w0) and 1 <= len(turning(w1)) <= 6 and all(a[6] == -1.0 and a[1] > 0 for a in turning(w1))
    # without the short backing off it does nothing (the built retreat waits for its pose)
    alone, w2 = run(take_turn_early=True)
    assert "turn_early" not in alone.attempts[0] and not turning(w2)
    # when the pull itself sent the open commands, the hand does not stand still for the rest of them: it backs off
    # along the pull's way while the jaws part (jaws that part three steps after the command, as the simulator's)
    held, w3 = run(Slow, open_retreat_pass_m=0.03, open_release_early=2)
    gone, w4 = run(Slow, open_retreat_pass_m=0.03, open_release_early=2, take_turn_early=True)
    assert held.attempts[0]["outcome"] == gone.attempts[0]["outcome"] == "opened" and gone.attempts[0]["turn_early"] is True
    assert len(w4.sent) == len(w3.sent) - 2 and abs(float((w4.handle - w3.handle) @ D)) < 0.025
    still = lambda w: sum(1 for a in w.sent if a[6] < 0 and float(np.linalg.norm(a[:3])) < 0.05)  # noqa: E731
    assert still(w3) >= 2 and still(w4) == 0
