"""The way back from a learned thing to the robot's own handling, and the late-start rule (2026-11-18): both off
unless given (``learn_skill.WAY_BACK``, ``LATE``; the runner's third reason to plan again)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from rowant_lab.pro2 import learn_skill as LS
from rowant_lab.pro2 import machine as MC


class _Inner:
    def __init__(self, finished, reason, outcomes=()):
        self.finished, self._s = finished, {"finish_reason": reason, "steps": 3,
                                            "attempts": [{"index": i, "outcome": o} for i, o in enumerate(outcomes)]}

    def summary(self):
        return self._s


def _learner(inner, acting="learned"):
    a = LS.LearnThenAct.__new__(LS.LearnThenAct)
    a.inner, a._done, a.rec = inner, False, {"trigger": True, "acting": acting}
    return a


def test_the_way_back_is_off_unless_given_and_its_numbers_are_the_recorded_ones():
    assert LS.way_back({"learn_in_episode": True}) is None
    assert LS.way_back({"learn_way_back": {}}) is None  # the learner itself is off
    o = LS.way_back({"learn_in_episode": True, "learn_way_back": {}})
    assert o == {"attempts": 1, "after": ("no_plan", "max_attempts"), "min_left": 71} == LS.WAY_BACK
    assert LS.way_back({"learn_in_episode": True, "learn_way_back": {"min_left": 98}})["min_left"] == 98
    assert LS.LATE == {"min_left": 209} and LS.ACT_RESERVE + 59 == 209
    assert LS.WAY_BACK["min_left"] < 95  # under a retry's need (the agent's T_attempt): no attempt is taken away
    from rowant_lab.pro2.robot.agent import AgentParams

    assert AgentParams().T_attempt == 95 and AgentParams().max_attempts == 5


def test_the_way_back_is_due_when_a_learned_thing_ended_with_enough_steps_left():
    o = dict(LS.WAY_BACK)
    a = _learner(_Inner(True, "no_plan", ["no_plan"]))
    assert LS.back_due(a, 82, 280, o) == {"t": 82, "why": "no_plan", "left": 198, "min_left": 71, "attempts": ["no_plan"]}
    b = _learner(_Inner(True, "max_attempts", ["c3_fail"]))
    assert LS.back_due(b, 209, 280, o)["why"] == "max_attempts" and LS.back_due(b, 209, 280, o)["left"] == 71
    # not due: still acting; another ending; the robot's own model; not the learner's agent; gone back before
    assert LS.back_due(_learner(_Inner(False, None)), 82, 280, o) is None
    assert LS.back_due(_learner(_Inner(True, "no_budget")), 82, 280, o) is None
    assert LS.back_due(_learner(_Inner(True, None)), 82, 280, o) is None
    assert LS.back_due(_learner(_Inner(True, "no_plan"), acting="own"), 82, 280, o) is None
    assert LS.back_due(_learner(_Inner(True, "no_plan"), acting="own_close"), 82, 280, o) is None
    assert LS.back_due(_Inner(True, "no_plan"), 82, 280, o) is None
    assert LS.back_due(_learner(None), 82, 280, o) is None  # still looking
    a.rec["way_back"] = {"t": 82}
    assert LS.back_due(a, 83, 280, o) is None


def test_with_too_few_steps_left_the_record_says_so_once():
    o = dict(LS.WAY_BACK)
    a = _learner(_Inner(True, "max_attempts", ["c3_fail"]))
    assert LS.back_due(a, 210, 280, o) is None
    assert a.rec["way_back_short"] == {"t": 210, "why": "max_attempts", "left": 70, "min_left": 71, "attempts": ["c3_fail"]}
    assert LS.back_due(a, 240, 280, o) is None and a.rec["way_back_short"]["t"] == 210 and "way_back" not in a.rec


class _R:
    def __init__(self):
        self.calls = []

    @staticmethod
    def parse(instruction, types):
        return {"status": "ok", "target_type": "milk"}

    def plan_episode(self, instruction, tool, lib, left, params):
        self.calls.append((left, copy.deepcopy(params)))
        return {"agent": "own-agent", "grounding": {"status": "ok"}, "parsed": {"target_type": "milk"}}


class _Lib:
    types = {"milk": 1, "basket": 1}
    export_types = {"milk": {"parts": {"a": 1}}}


class _Tool:
    def __init__(self, percepts=()):
        self.percepts = list(percepts)


def test_a_learner_that_would_start_too_late_does_not_start():
    R = _R()
    out = LS.plan(R, "pick the milk", _Tool(), _Lib, 140, {"learn_late": {}}, None)
    assert out["agent"] == "own-agent" and R.calls[0][0] == 140
    assert out["learn"] == {"trigger": False, "late": {"left": 140, "min_left": 209, "target_type": "milk"}}
    assert LS.plan(R, "pick the milk", _Tool(), _Lib, 208, {"learn_late": {}}, None)["learn"]["late"]["left"] == 208
    # with enough steps, or without the key, the learner starts as before
    for cap, params in ((209, {"learn_late": {}}), (280, {"learn_late": {}}), (140, {})):
        got = LS.plan(R, "pick the milk", _Tool(), _Lib, cap, params, None)
        assert type(got["agent"]).__name__ == "LearnThenAct" and got["learn"]["trigger"] is True
    assert len(R.calls) == 2
    # a scene whose target was accepted never was the learner's: no late record
    seen = LS.plan(R, "pick the milk", _Tool([{"type": "milk"}]), _Lib, 140, {"learn_late": {}}, None)
    assert seen["learn"] == {"trigger": False} and LS.plan(R, "x", _Tool(), _Lib, 140, {"learn_late": {"min_left": 100}},
                                                         None)["learn"]["trigger"] is True


@pytest.mark.parametrize("params, want", [({"learn_in_episode": True}, None),
                                          ({"learn_in_episode": True, "learn_way_back": {}}, 1),
                                          ({"learn_in_episode": True, "learn_way_back": {"attempts": 2}}, 2)])
def test_the_learned_thing_gets_its_attempts_cut_only_with_the_way_back(monkeypatch, params, want):
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.pro2 import goal_paths as GP

    monkeypatch.setattr(LB, "M2Library", lambda *a, **k: "variant-lib")
    monkeypatch.setattr(GP, "extra_suites", lambda suites: [])

    class Tool(_Tool):
        suite, lib, held_near_m = "libero_object", "own-lib", 0.1

        @staticmethod
        def _build(T, R_w, t_w):
            return []

    R = _R()
    a = LS.LearnThenAct.__new__(LS.LearnThenAct)
    a.R, a.instruction, a.tool, a.lib, a.cap, a.t, a.T = R, "pick the milk", Tool(), "own-lib", 280, 60, "milk"
    a.params = {**params, "m1": {"agent": {"c2_lift_m": 0.1}}}
    before = json.dumps(a.params, sort_keys=True)
    a.rec, a.finish_reason = {"learn": {"model": {"origin_w": [0.0, 0.0, 0.0]}}}, None
    assert a._act_agent({"policy_lib": "p", "m1_library": "m", "gid": "7"}) == "own-agent"
    left, used = R.calls[0]
    assert left == 220 and a.rec["acting"] == "learned" and json.dumps(a.params, sort_keys=True) == before
    assert a._tool0 == {"percepts": [], "fallbacks": [], "held_near_m": 0.1}  # the tool as it was, kept for a way back
    assert used["m1"]["agent"].get("max_attempts") == want and used["m1"]["agent"]["c1_band_lo_m"] == 0.012
    assert used["m1"]["agent"]["c2_lift_m"] == 0.1


def test_going_back_takes_the_learned_thing_off_the_tool_and_plans_the_robots_own_way():
    class Tool(_Tool):
        suite, lib, held_near_m = "libero_object", "variant-lib", LS.F7_HELD_NEAR_M

    R = _R()
    # the plan made on the learned thing added a place of its own (f1) to the percepts and to the fallbacks' record
    tool = Tool([{"id": "p1", "type": "plate"}, {"id": "learned0", "type": "milk"}, {"id": "f1", "type": "basket"}])
    tool.fallbacks = [{"role": "landmark", "type": "stove", "source": "rejected:x"},
                      {"role": "place", "type": "basket", "id": "f1", "source": "blob_top", "size": 3}]
    a = _learner(_Inner(True, "max_attempts", ["c3_fail"]))
    a.inner._s.update(steps=112, attempts=[{"index": 0, "t_start": 0, "t_end": 112, "outcome": "c3_fail", "c1": {"x": 1}}])
    a.R, a.instruction, a.tool, a.lib, a.cap, a.t, a.T = R, "pick the milk", tool, "own-lib", 280, 58, "milk"
    a.params, a._steps_learned, a.finish_reason = {"learn_in_episode": True, "learn_way_back": {}}, 0, None
    a._tool0 = {"percepts": [{"id": "p1", "type": "plate"}], "fallbacks": [tool.fallbacks[0]], "held_near_m": 0.1}
    doc = LS.back_due(a, 171, 280, dict(LS.WAY_BACK))
    assert doc["left"] == 109 and doc["why"] == "max_attempts"
    a.go_back(doc, 109)
    assert R.calls[0][0] == 109 and a.inner == "own-agent"
    assert tool.lib == "own-lib" and [p["id"] for p in tool.percepts] == ["p1"] and tool.held_near_m == 0.1
    assert tool.fallbacks == [{"role": "landmark", "type": "stove", "source": "rejected:x"}]
    assert a.rec["way_back"] is doc and a.rec["acting"] == "learned" and doc["grounding"] == "ok"
    assert doc["learned"] == {"finish_reason": "max_attempts", "steps": 112,
                              "attempts": [{"index": 0, "t_start": 0, "t_end": 112, "outcome": "c3_fail"}],
                              "fallbacks": [{"role": "place", "type": "basket", "source": "blob_top"}]}
    assert LS.back_due(a, 172, 280, dict(LS.WAY_BACK)) is None  # once only
    a.inner = _Inner(True, None)
    a.inner._s.update(steps=80)
    assert a.summary()["steps"] == 58 + 112 + 80 and a.summary()["learn"] is a.rec
    # no plan of the robot's own either: the episode's agent is finished, and the row says why

    class NoPlan(_R):
        def plan_episode(self, instruction, tool, lib, left, params):
            return {"agent": None, "grounding": {"status": "fail"}}

    b = _learner(_Inner(True, "no_plan", ["no_plan"]))
    b.R, b.instruction, b.tool, b.lib, b.cap, b.t, b.T = NoPlan(), "pick the milk", Tool(), "own-lib", 280, 60, "milk"
    b.params, b._tool0, b._steps_learned, b.finish_reason = {}, None, 0, None
    b.go_back({"t": 63, "why": "no_plan", "left": 217}, 217)
    assert b.inner is None and b.finished and b.finish_reason == "grounding_failed" and b.rec["way_back"]["grounding"] == "fail"
    assert b.summary()["finish_reason"] == "grounding_failed"


def test_the_runner_goes_back_only_with_the_option_and_never_before_another_reason():
    src = (Path(LS.__file__).resolve().parent / "dev_episode.py").read_text()
    assert src.count('if params.get("learn_way_back") is not None and params.get("learn_in_episode"):') == 1
    assert src.count("    back = None\n") == 1 and src.count("back = LSB.way_back(params)") == 1
    assert src.count("if back is not None and again is None and not done and agent_err is None "
                     "and perception_error is None:") == 1
    assert src.count("LSB.") == 2 and src.count("agent.go_back(back_doc, cap - t)") == 1
    # a plan made again for another reason is made as before (the learner again, or the robot's plan with it off)
    assert src.count('if params.get("learn_in_episode"):  # 2026-11-17: the learner again') == 1
    assert src.count('"back"') == 0
    # candidate v9 names neither key
    from rowant_lab.pro2 import long_dev as LD

    b = json.dumps(LD.bundle(LD.colour_long_params(MC.REGISTERED["net"])))
    assert "learn_way_back" not in b and "learn_late" not in b and "learn_in_episode" not in b
    skill = Path(LS.__file__).read_text()
    assert skill.count('params.get("learn_late") is not None') == 1 and skill.count("way_back(self.params)") == 1


def test_the_first_starts_record_is_kept_when_a_plan_is_made_again():
    old = {"trigger": True, "acting": "learned", "steps": 86}
    new = {"trigger": False, "late": {"left": 136}}
    got = LS.keep_first(old, new, 144)
    assert got is new and got["first"] == {"trigger": True, "acting": "learned", "steps": 86, "until": 144}
    # a third plan keeps the earliest start, with the step it ended at
    third = LS.keep_first(got, {"trigger": False}, 200)
    assert third["first"] == {"trigger": True, "acting": "learned", "steps": 86, "until": 144} and "first" not in third["first"]
    # nothing to keep: no learner before, or the same record goes on
    assert LS.keep_first({"trigger": False}, {"trigger": True}, 5) == {"trigger": True}
    assert LS.keep_first(None, {"trigger": True}, 5) == {"trigger": True} and LS.keep_first(old, None, 5) is None
    same = {"trigger": True}
    assert LS.keep_first(same, same, 9) is same and "first" not in same


def _learned_agent(monkeypatch, params, outcomes=("unreached",), steps=56):
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.pro2 import goal_paths as GP

    monkeypatch.setattr(LB, "M2Library", lambda *a, **k: "variant-lib")
    monkeypatch.setattr(GP, "extra_suites", lambda suites: [])

    class Tool(_Tool):
        suite, lib, held_near_m, fallbacks = "libero_object", "own-lib", 0.1, []

        @staticmethod
        def _build(T, R_w, t_w):
            return {"milk_main": {"pos": [float(x) for x in t_w]}}

    R = _R()
    a = _learner(_Inner(False, None, outcomes))
    a.inner._s.update(steps=steps)
    a.R, a.instruction, a.tool, a.lib, a.cap, a.t, a.T = R, "pick the milk", Tool(), "own-lib", 280, 86, "milk"
    a.params, a._steps_learned, a.finish_reason, a._tool0 = params, 0, None, None
    a._learned = {"policy_lib": "p", "m1_library": "m", "gid": "7"}
    a.rec.update(learn={"model": {"origin_w": [0.2, 0.0, 0.0]}}, grounding={"status": "ok"})
    return a, R, Tool


def test_a_plan_made_again_after_a_caught_arm_goes_on_with_the_learned_thing(monkeypatch):
    params = {"learn_in_episode": True, "learn_keep": {}, "learn_way_back": {}}
    a, R, Tool = _learned_agent(monkeypatch, params)
    new_tool = Tool([{"id": "p1", "type": "basket"}])
    got = LS.keep_learned(a, new_tool, 136, params, "delay", 144)
    assert got is not None and got["agent"] is a and got["learn"] is a.rec and a.tool is new_tool
    assert R.calls[0][0] == 136 and a.inner == "own-agent"  # (the canned plan's agent)
    assert [p["id"] for p in new_tool.percepts] == ["p1", "learned0"] and new_tool.lib == "variant-lib"
    assert new_tool.percepts[1]["bodies"]["milk_main"]["pos"] == [0.2, 0.0, 0.0] and new_tool.held_near_m == LS.F7_HELD_NEAR_M
    assert R.calls[0][1]["m1"]["agent"] == {"c1_band_lo_m": 0.012, "max_attempts": 1}  # the way back still counts attempts
    assert a.rec["kept"] == [{"t": 144, "why": "delay", "left": 136, "attempts": ["unreached"]}] and a.rec["acting"] == "learned"
    assert a._tool0["percepts"] == [{"id": "p1", "type": "basket"}] and a._steps_learned == 56
    assert LS.keep_first(a.rec, got["learn"], 144) is a.rec and "first" not in a.rec  # the same record goes on
    # once an episode: a second caught arm is handled as before
    a.inner = _Inner(False, None)
    assert LS.keep_learned(a, Tool(), 60, params, "delay", 220) is None and len(a.rec["kept"]) == 1
    assert LS.KEEP == {"times": 1, "after": ("delay",)}


@pytest.mark.parametrize("change", ["no_key", "calib", "own", "back", "learner_off", "not_learner", "looking"])
def test_the_learned_thing_is_kept_only_with_the_option_after_a_caught_arm(monkeypatch, change):
    params = {"learn_in_episode": True, "learn_keep": {}}
    a, R, Tool = _learned_agent(monkeypatch, params)
    why, agent = "delay", a
    if change == "no_key":
        params = {"learn_in_episode": True}
    elif change == "learner_off":
        params = {"learn_keep": {}}
    elif change == "calib":
        why = "calib"  # the camera was fitted again: the learned pose is not to be trusted
    elif change == "own":
        a.rec["acting"] = "own"
    elif change == "back":
        a.rec["way_back"] = {"t": 90}
    elif change == "not_learner":
        agent = _Inner(False, None)
    elif change == "looking":
        a.inner = None
    tool = Tool()
    assert LS.keep_learned(agent, tool, 136, params, why, 144) is None
    assert not R.calls and tool.percepts == [] and "kept" not in a.rec and a.tool is not tool


def test_the_runner_asks_to_keep_the_learned_thing_where_it_plans_again():
    src = (Path(LS.__file__).resolve().parent / "dev_episode.py").read_text()
    assert src.count("LS2.keep_learned(agent, tool, cap - t, params, again, t)") == 1
    assert src.count("or LS2.plan(R, instruction, tool, lib, cap - t, params, rep2)") == 1
    assert src.count("learn_rec = LS2.keep_first(learn_rec, planned.get(\"learn\"), t)") == 1
    from rowant_lab.pro2 import long_dev as LD

    b = json.dumps(LD.bundle(LD.colour_long_params(MC.REGISTERED["net"])))
    assert "learn_keep" not in b and "learn_in_episode" not in b
