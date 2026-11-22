"""The robot's own attempt first, the learner after it has failed (2026-11-18): off unless given
(``learn_skill.AFTER_FAIL``; the runner's fourth reason to plan again, ``own_failed``)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from rowant_lab.pro2 import learn_skill as LS
from rowant_lab.pro2 import machine as MC

ON = {"learn_in_episode": True, "learn_after_fail": {}}


class _R:
    def __init__(self, agent="own-agent"):
        self.calls, self.agent = [], agent

    @staticmethod
    def parse(instruction, types):
        return {"status": "ok", "target_type": "milk"}

    def plan_episode(self, instruction, tool, lib, left, params):
        self.calls.append((left, copy.deepcopy(params)))
        tool.percepts = list(tool.percepts) + [{"type": "stand-in", "id": "fb0"}]  # (the robot's plan adds its own)
        tool.fallbacks = list(tool.fallbacks) + [{"role": "target"}]
        if self.agent is None:
            return {"agent": None, "grounding": {"status": "fail", "reason": "target_missing"}}
        return {"agent": self.agent, "grounding": {"status": "ok"}, "parsed": {"target_type": "milk"}}


class _Lib:
    types = {"milk": 1, "basket": 1}
    export_types = {"milk": {"parts": {"a": 1}}}


class _Tool:
    def __init__(self, percepts=()):
        self.percepts, self.fallbacks = list(percepts), []


class _Own:
    """The robot's own agent as the runner sees it: its attempts so far, and whether it has ended."""

    def __init__(self, outcomes=(), begun=0, finished=False, reason=None):
        self.attempts = [{"index": i, "outcome": o} for i, o in enumerate(outcomes)] \
            + [{"index": len(outcomes) + i} for i in range(begun)]
        self.finished, self.finish_reason = finished, reason


def _waiting():
    return {"trigger": False, "own_first": {"pending": True, "target_type": "milk"}}


def test_the_option_is_off_unless_given_and_its_numbers_are_the_recorded_ones():
    assert LS.after_fail({}) is None and LS.after_fail({"learn_in_episode": True}) is None
    assert LS.after_fail({"learn_after_fail": {}}) is None  # (the learner itself is off)
    o = LS.after_fail(ON)
    assert o == {"attempts": 1, "min_left": 209,
                 "after": ("c1_fail", "c3_fail", "approach_stalled", "unreachable", "no_plan"),
                 "ended": ("no_plan", "max_attempts", "no_budget"), "retreat": 0, "retreat_tol_m": 0.04}
    assert o["min_left"] == LS.LATE["min_left"]
    assert LS.after_fail({**ON, "learn_after_fail": {"attempts": 2}})["attempts"] == 2


def test_with_the_option_the_robots_own_plan_is_made_first_where_it_has_one():
    R, tool = _R(), _Tool()
    out = LS.plan(R, "pick the milk", tool, _Lib, 280, ON, None)
    assert out["agent"] == "own-agent" and R.calls[0][0] == 280
    assert out["learn"] == {"trigger": False, "own_first": {"pending": True, "target_type": "milk"}}
    # without the option the learner starts at once, and the robot's plan is not asked
    R2 = _R()
    got = LS.plan(R2, "pick the milk", _Tool(), _Lib, 280, {"learn_in_episode": True}, None)
    assert type(got["agent"]).__name__ == "LearnThenAct" and "own_first" not in got["learn"] and not R2.calls
    # a scene whose target the look accepted never was the learner's
    seen = LS.plan(_R(), "pick the milk", _Tool([{"type": "milk"}]), _Lib, 280, ON, None)
    assert seen["learn"] == {"trigger": False}


def test_where_the_robot_has_no_plan_of_its_own_the_learner_is_due_as_soon_as_the_runner_asks():
    R, tool = _R(agent=None), _Tool([{"type": "basket", "id": "p0"}])
    out = LS.plan(R, "pick the milk", tool, _Lib, 280, ON, None)
    assert out["agent"] is None and len(R.calls) == 1 and out["grounding"]["status"] == "fail"
    rec = out["learn"]
    assert rec == {"trigger": False, "own_first": {"pending": True, "target_type": "milk", "no_own_plan": True,
                                                   "grounding": {"status": "fail", "reason": "target_missing"}}}
    doc = LS.fail_due(None, rec, 1, 280, LS.after_fail(ON))
    assert doc == {"t": 1, "why": "no_own_plan", "left": 279, "min_left": 209, "attempts": [], "target_type": "milk"}
    assert rec["own_first"]["pending"] is False and LS.fail_due(None, rec, 2, 280, LS.after_fail(ON)) is None
    # after the arm's own motion for the camera check (some 20 steps) there is still room; late in an episode not
    late = {"trigger": False, "own_first": {"pending": True, "target_type": "milk", "no_own_plan": True}}
    assert LS.fail_due(None, late, 100, 280, LS.after_fail(ON)) is None and late["own_first"]["short"]["left"] == 180
    # no agent without that record is not the learner's
    assert LS.fail_due(None, _waiting(), 1, 280, LS.after_fail(ON)) is None


def test_the_learner_is_due_when_the_first_attempt_failed_and_the_next_has_begun():
    o = LS.after_fail(ON)
    rec = _waiting()
    assert LS.fail_due(_Own(), rec, 3, 280, o) is None and rec["own_first"]["pending"] is True
    assert LS.fail_due(_Own(begun=1), rec, 20, 280, o) is None  # the first attempt is under way
    assert LS.fail_due(_Own(("c1_fail",)), rec, 45, 280, o) is None  # failed; the hand has not backed off yet
    assert rec["own_first"] == {"pending": True, "target_type": "milk"}
    doc = LS.fail_due(_Own(("c1_fail",), begun=1), rec, 57, 280, o)
    assert doc == {"t": 57, "why": "attempt", "left": 223, "min_left": 209, "attempts": ["c1_fail", None],
                   "target_type": "milk"}
    assert rec["own_first"]["pending"] is False and rec["own_first"]["due"] == doc
    assert LS.fail_due(_Own(("c1_fail",), begun=1), rec, 58, 280, o) is None  # once an episode


@pytest.mark.parametrize("own, why", [(_Own(("no_plan",), finished=True, reason="no_plan"), "ended"),
                                      (_Own((), finished=True, reason="no_plan"), "ended"),
                                      (_Own(("approach_stalled",), finished=True, reason="no_budget"), "ended"),
                                      (_Own(("unreachable",), begun=1), "attempt"),
                                      (_Own(("c3_fail", "c1_fail"), begun=1), "attempt")])
def test_the_endings_the_learner_follows(own, why):
    rec = _waiting()
    doc = LS.fail_due(own, rec, 30, 280, LS.after_fail(ON))
    assert doc is not None and doc["why"] == why and doc["left"] == 250
    assert (doc.get("finish_reason") is not None) == (why == "ended")


class _Steps:
    """A plan in steps (``SequenceAgent``): its ended steps' attempts, and the step that runs."""

    def __init__(self, done=(), cur=None, finished=False, reason=None):
        self.attempts = [{"index": i, "outcome": o, "step": 0} for i, o in enumerate(done)]
        self.cur, self.finished, self.finish_reason = cur, finished, reason


def test_a_plan_in_steps_is_read_through_the_step_that_runs():
    o = LS.after_fail(ON)
    rec = _waiting()
    assert LS.fail_due(_Steps(cur=_Own(begun=1)), rec, 20, 280, o) is None and rec["own_first"]["pending"] is True
    doc = LS.fail_due(_Steps(cur=_Own(("c1_fail",), begun=1)), rec, 60, 280, o)
    assert doc["why"] == "attempt" and doc["attempts"] == ["c1_fail", None]
    # its take found no plan and the steps ended: "steps done" is its ending whatever the steps did
    rec = _waiting()
    doc = LS.fail_due(_Steps(done=("no_plan",), finished=True, reason="steps done"), rec, 9, 280, o)
    assert doc["why"] == "ended" and doc["finish_reason"] == "steps done" and doc["attempts"] == ["no_plan"]
    # ended with no attempt at all and no named ending: not the learner's
    rec = _waiting()
    assert LS.fail_due(_Steps(finished=True, reason="steps done"), rec, 9, 280, o) is None
    assert rec["own_first"]["why"] == "other"
    # a step that placed its thing is a success of the robot's own
    rec = _waiting()
    assert LS.fail_due(_Steps(done=("placed",), finished=True, reason="steps done"), rec, 90, 280, o) is None
    assert rec["own_first"]["why"] == "other"


@pytest.mark.parametrize("own", [_Own(("placed",), finished=True, reason="placed"),
                                 _Own(("c2_fail",), begun=1),
                                 _Own(("moved_before_grasp",), begun=1),
                                 _Own(("placed",), begun=1),
                                 _Own((), finished=True, reason="generator_end")])
def test_other_endings_are_the_robots_own_business_and_the_learner_stops_waiting(own):
    rec = _waiting()
    assert LS.fail_due(own, rec, 30, 280, LS.after_fail(ON)) is None
    assert rec["own_first"]["pending"] is False and rec["own_first"]["started"] is False
    assert rec["own_first"]["why"] == "other"


def test_with_too_few_steps_left_the_robots_own_plan_goes_on_and_the_record_says_so():
    rec = _waiting()
    assert LS.fail_due(_Own(("c1_fail",), begun=1), rec, 72, 280, LS.after_fail(ON)) is None
    own = rec["own_first"]
    assert own["pending"] is False and own["started"] is False and own["short"]["left"] == 208 and "due" not in own
    # 209 left is enough
    assert LS.fail_due(_Own(("c1_fail",), begun=1), _waiting(), 71, 280, LS.after_fail(ON))["left"] == 209


def test_two_attempts_can_be_asked_for():
    o = LS.after_fail({**ON, "learn_after_fail": {"attempts": 2, "min_left": 100}})
    rec = _waiting()
    assert LS.fail_due(_Own(("c1_fail",), begun=1), rec, 57, 280, o) is None and rec["own_first"]["pending"] is True
    doc = LS.fail_due(_Own(("c1_fail", "c1_fail"), begun=1), rec, 99, 280, o)
    assert doc["attempts"] == ["c1_fail", "c1_fail", None] and doc["left"] == 181


def test_nothing_is_due_without_a_waiting_record_or_for_the_learners_own_agent():
    o = LS.after_fail(ON)
    failed = _Own(("c1_fail",), begun=1)
    for rec in (None, {}, {"trigger": False}, {"trigger": False, "own_first": {"pending": False}}):
        assert LS.fail_due(failed, rec, 57, 280, o) is None
    learner = LS.LearnThenAct.__new__(LS.LearnThenAct)
    learner.attempts, learner.finished_ = failed.attempts, False
    rec = _waiting()
    assert LS.fail_due(learner, rec, 57, 280, o) is None and rec["own_first"]["pending"] is True
    assert LS.fail_due(None, rec, 57, 280, o) is None and rec["own_first"]["pending"] is True


def test_after_the_failed_start_the_learner_starts_from_the_new_look_if_it_still_accepts_no_target():
    doc = {"t": 57, "why": "attempt", "left": 223, "min_left": 209, "attempts": ["c1_fail", None], "target_type": "milk"}
    R = _R()
    out = LS.plan_after_fail(R, "pick the milk", _Tool(), _Lib, 223, {**ON, "learn_late": {}}, None, doc)
    assert type(out["agent"]).__name__ == "LearnThenAct" and not R.calls  # (no second own attempt first)
    assert out["learn"]["trigger"] is True
    assert out["learn"]["own_first"] == {**doc, "pending": False, "started": True}
    # the new look accepted the target: the robot's own plan again, and the record says why the learner stayed out
    seen = LS.plan_after_fail(R, "pick the milk", _Tool([{"type": "milk"}]), _Lib, 223, ON, None, doc)
    assert seen["agent"] == "own-agent"
    assert seen["learn"] == {"trigger": False, "own_first": {**doc, "pending": False, "started": False, "why": "accepted"}}
    # too late by the late-start rule: the robot's own plan
    late = LS.plan_after_fail(R, "pick the milk", _Tool(), _Lib, 150, {**ON, "learn_late": {}}, None, doc)
    assert late["agent"] == "own-agent" and late["learn"]["own_first"]["why"] == "late"
    assert late["learn"]["late"] == {"left": 150, "min_left": 209, "target_type": "milk"}


def test_a_plan_made_again_for_another_reason_starts_the_learner_again_only_where_it_had_started():
    doc = {"t": 57, "why": "attempt", "left": 223, "min_left": 209, "attempts": ["c1_fail", None], "target_type": "milk"}
    started = {"trigger": True, "own_first": {**doc, "pending": False, "started": True}}
    assert LS.fail_before(started) == doc
    assert LS.fail_before({"trigger": True, "first": started}) == doc  # (kept under ``first`` by ``keep_first``)
    at_once = {"trigger": True, "own_first": {"pending": False, "started": True, "why": "no_own_plan", "t": 21}}
    assert LS.fail_before(at_once) == {"why": "no_own_plan", "t": 21}
    for rec in (None, {}, _waiting(), {"trigger": False, "own_first": {"pending": False, "started": False, "short": doc}}):
        assert LS.fail_before(rec) is None
        assert LS.again_after_fail(_R(), "pick the milk", _Tool(), _Lib, 200, ON, None, rec) is None
    R = _R()
    out = LS.again_after_fail(R, "pick the milk", _Tool(), _Lib, 215, ON, None, started)
    assert type(out["agent"]).__name__ == "LearnThenAct" and not R.calls and out["learn"]["own_first"]["t"] == 57
    # and where the learner is still waiting, the robot's own plan is made first again
    again = LS.plan(R, "pick the milk", _Tool(), _Lib, 215, ON, None)
    assert again["agent"] == "own-agent" and again["learn"]["own_first"]["pending"] is True


def test_the_runner_looks_again_only_with_the_option_and_never_before_another_reason():
    src = (Path(LS.__file__).resolve().parent / "dev_episode.py").read_text()
    assert src.count('if params.get("learn_after_fail") is not None and params.get("learn_in_episode"):') == 1
    assert src.count("    own1 = None\n") == 1 and src.count("own1 = LSF.after_fail(params)") == 1
    assert src.count("if own1 is not None and again is None and not done and agent_err is None "
                     "and perception_error is None:") == 1
    assert src.count("fail_doc = LSF.fail_due(agent, learn_rec, t, cap, own1)") == 1
    # the arm's camera check comes first: while it moves the arm, or is due to, the learner is not asked
    assert src.count("waits = calib is not None and (calib.moving or calib.can_move(") == 1
    assert src.index("waits = calib is not None") < src.index("LSF.fail_due(") and src.count("if not waits:") == 1
    assert src.count('again = "own_failed"') == 1 and src.count('elif again == "own_failed":') == 1
    assert src.count("LS2.plan_after_fail(R, instruction, tool, lib, cap - t, params, rep2, fail_doc)") == 1
    assert src.count("LS2.again_after_fail(R, instruction, tool, lib, cap - t, params, rep2, learn_rec)") == 1
    # the reasons that were there before come first: the delay, the camera fit, the way back
    assert src.index('again = "delay"') < src.index('again = "calib"') < src.index("LSB.back_due(") \
        < src.index("LSF.fail_due(")
    # candidate v9 does not name the key, and neither does the registered learner's arm
    from rowant_lab.pro2 import long_dev as LD

    b = json.dumps(LD.bundle(LD.colour_long_params(MC.REGISTERED["net"])))
    assert "learn_after_fail" not in b and "learn_in_episode" not in b
    skill = Path(LS.__file__).read_text()
    assert skill.count('params.get("learn_after_fail")') == 1


def test_the_camera_check_asks_to_move_exactly_when_the_robot_gave_up_before_it_was_done():
    """What the runner's ``waits`` rests on (``arm_calib.ArmCalib.can_move``): a robot that ended with no plan
    before its camera was checked is the check's cue to move; once the check is done, or with the agent still at
    work, it is not."""
    from rowant_lab.pro2 import arm_calib as AC

    c = AC.ArmCalib.__new__(AC.ArmCalib)
    c.moving, c.motion_done, c.done = False, False, False
    c.rec = {"status": "watching"}
    c.o = {"min_left": 60, "after": ("no_plan", "max_attempts", "no_budget", "grounding_failed")}
    gave_up = _Own(("no_plan",), finished=True, reason="no_plan")
    assert c.can_move(gave_up, True, 1, 280) is True
    assert c.can_move(_Own(begun=1), False, 20, 280) is False  # at work on its first attempt
    c.done, c.rec["status"] = True, "kept"
    assert c.can_move(gave_up, True, 30, 280) is False  # checked: the learner may be asked now
    c.done, c.rec["status"], c.motion_done = False, "watching", True
    assert c.can_move(gave_up, True, 30, 280) is False  # its own motion ran once


# -- the retreat before the new look (2026-11-18 evening; ``learn_after_fail.retreat``, 0 = off)
def _obs(p, quat=(0.0, 0.0, 0.0, 1.0)):
    return {"robot0_eef_pos": list(p), "robot0_eef_quat_xyzw": list(quat)}


def test_the_retreat_is_off_unless_asked_for_and_its_steps_count_against_the_floor():
    import numpy as np

    o = LS.after_fail(ON)
    assert o["retreat"] == 0 and o["retreat_tol_m"] == 0.04
    doc = {"t": 57, "why": "attempt", "left": 223, "min_left": 209, "attempts": ["c1_fail", None], "target_type": "milk"}
    assert LS.retreat_begin(doc, (np.zeros(3), np.eye(3)), 57, o) is None
    o2 = LS.after_fail({**ON, "learn_after_fail": {"retreat": 20, "min_left": 190}})
    assert LS.retreat_begin(doc, None, 57, o2) is None  # the start pose is not known: no retreat
    # 223 left with a retreat of 20 and a floor of 190: due; 205 left: short, though 205 is above the floor itself
    rec = _waiting()
    due = LS.fail_due(_Own(("c1_fail",), begun=1), rec, 57, 280, o2)
    assert due["left"] == 223 and due["min_left"] == 190 and due["retreat_max"] == 20
    rec = _waiting()
    assert LS.fail_due(_Own(("c1_fail",), begun=1), rec, 75, 280, o2) is None
    assert rec["own_first"]["short"]["left"] == 205 and rec["own_first"]["short"]["retreat_max"] == 20
    # without a retreat the same floor lets it start
    rec = _waiting()
    assert LS.fail_due(_Own(("c1_fail",), begun=1), rec, 75, 280,
                       LS.after_fail({**ON, "learn_after_fail": {"min_left": 190}}))["left"] == 205


def test_the_hand_goes_back_toward_its_start_pose_open_and_the_record_says_what_that_took():
    import numpy as np

    o = LS.after_fail({**ON, "learn_after_fail": {"retreat": 20, "min_left": 190}})
    doc = {"t": 57, "why": "attempt", "left": 223, "min_left": 190, "attempts": ["c1_fail", None],
           "target_type": "milk", "retreat_max": 20}
    home = (np.array([0.0, 0.0, 1.2]), np.eye(3))
    st = LS.retreat_begin(doc, home, 57, o)
    assert st["t0"] == 57 and st["max"] == 20 and st["tol"] == 0.04 and st["doc"] == doc and st["doc"] is not doc
    a = LS.retreat_action(_obs((0.2, 0.1, 0.95)), st)
    assert len(a) == 7 and a[6] == -1.0 and a[0] < 0 and a[1] < 0 and a[2] > 0  # toward the start pose, hand open
    assert max(abs(x) for x in a[:3]) <= 1.0 + 1e-9
    # on its way: nothing yet
    assert LS.retreat_done(_obs((0.2, 0.1, 0.95)), st, 60, 280) is None
    # there (within 4 cm): the record for the plan, with the step the learner starts at
    got = LS.retreat_done(_obs((0.02, 0.0, 1.19)), st, 70, 280)
    assert got["t"] == 70 and got["due_t"] == 57 and got["left"] == 210
    assert got["retreat"] == {"steps": 13, "err_m": 0.0224} and got["attempts"] == ["c1_fail", None]
    # not there after its steps: it ends all the same, and the record keeps how far off it was
    far = LS.retreat_done(_obs((0.2, 0.1, 0.95)), st, 77, 280)
    assert far["t"] == 77 and far["retreat"]["steps"] == 20 and far["retreat"]["err_m"] > 0.3
    assert st["doc"] == doc  # (the state's record is not written on)


def test_after_a_failed_start_the_options_own_floor_decides_whether_it_is_too_late():
    doc = {"t": 70, "why": "attempt", "left": 195, "min_left": 190, "attempts": ["c1_fail", None], "target_type": "milk"}
    R = _R()
    low = {**ON, "learn_late": {}, "learn_after_fail": {"retreat": 20, "min_left": 190}}
    out = LS.plan_after_fail(R, "pick the milk", _Tool(), _Lib, 195, low, None, doc)
    assert type(out["agent"]).__name__ == "LearnThenAct" and not R.calls  # 195 is under the late-start 209
    late = LS.plan_after_fail(R, "pick the milk", _Tool(), _Lib, 189, low, None, doc)
    assert late["agent"] == "own-agent" and late["learn"]["own_first"]["why"] == "late"
    assert late["learn"]["late"]["min_left"] == 190
    # a first plan (no failed start before it) keeps the late-start rule's own number
    R2 = _R()
    first = LS.plan(R2, "pick the milk", _Tool(), _Lib, 195, {"learn_in_episode": True, "learn_late": {}}, None)
    assert first["agent"] == "own-agent" and first["learn"]["late"]["min_left"] == 209


def test_the_runner_takes_the_hand_back_before_the_look_only_with_the_option():
    src = (Path(LS.__file__).resolve().parent / "dev_episode.py").read_text()
    assert src.count("    retreat, home = None, None") == 1
    assert src.count("elif retreat is not None:") == 1 and src.count("LSF.retreat_action(obs, retreat)") == 1
    assert src.count("fail_doc = LSF.retreat_done(obs, retreat, t, cap)") == 1
    assert src.count("retreat = LSF.retreat_begin(fail_doc, home, t, own1)") == 1
    # the camera check's own motion comes before a retreat, and the robot's agent is not asked during one
    a, b, c = (src.index("if calib is not None and calib.moving:"), src.index("elif retreat is not None:"),
               src.index("elif agent is not None and finished_at is None and agent_err is None and perception_error is None:"))
    assert a < b < c
    # another reason to plan again ends a retreat
    assert src.count('if again is not None and again != "own_failed":\n            retreat = None') == 1
    # the defaults ask for no retreat, so Step H's first form is unchanged by this
    assert LS.AFTER_FAIL["retreat"] == 0 and LS.AFTER_FAIL["min_left"] == 209
