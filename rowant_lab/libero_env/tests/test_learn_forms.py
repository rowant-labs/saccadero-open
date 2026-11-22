"""``learn_forms`` (2026-11-19, off unless given): the learner on the other forms of instruction. The robot's own
reading of the whole instruction (``episode.named_targets``), the start rule (``learn_skill.form_target``), and that
without the option the registered start rule is all there is."""

from __future__ import annotations

import types

import pytest

from rowant_lab.pro2 import learn_skill as LS
from rowant_lab.pro2.robot import episode as EP
from rowant_lab.pro2.robot import ground as GD

TYPES = ("akita_black_bowl", "plate", "wine_bottle", "cream_cheese", "wooden_cabinet", "flat_stove", "alphabet_soup",
         "tomato_sauce", "basket", "moka_pot", "glazed_rim_porcelain_ramekin", "black_book", "desk_caddy")


def _lib(two_part=("wooden_cabinet",)):
    parts = lambda t: {"main": {"joint": None}, **({"drawer": {"joint": {"type": "slide"}}} if t in two_part else {})}  # noqa: E731
    return types.SimpleNamespace(types=set(TYPES), export_types={t: {"parts": parts(t)} for t in TYPES}, workcell={})


def _tool(*have):
    return types.SimpleNamespace(percepts=[{"id": f"p{i}", "type": t} for i, t in enumerate(have)])


def _named(instruction, tool=None, sequence=True):
    out = EP.named_targets(instruction, tool or _tool(), _lib(), {"sequence": sequence})
    return [(x["step"], x["steps"], x["form"], x["type"], x["relation"], x["have"], x["single_fails"]) for x in out]


def test_the_robots_own_reading_of_each_form_of_instruction():
    assert _named("Pick the akita black bowl next to the ramekin and place it on the plate") == \
        [(0, 1, "pick", "akita_black_bowl", "next_to", 0, None)]
    assert _named("put the bowl on the plate") == [(0, 1, "put", "akita_black_bowl", None, 0, None)]
    assert _named("put the wine bottle on top of the cabinet") == [(0, 1, "put", "wine_bottle", None, 0, None)]
    assert _named("put the cream cheese in the bowl") == [(0, 1, "put", "cream_cheese", None, 0, None)]
    # several steps: one entry for each step that picks something up, in order; "both" names two
    assert _named("put both the alphabet soup and the tomato sauce in the basket") == \
        [(0, 2, "put", "alphabet_soup", None, 0, None), (1, 2, "put", "tomato_sauce", None, 0, None)]
    assert _named("put both moka pots on the stove") == \
        [(0, 2, "put", "moka_pot", None, 0, None), (1, 2, "put", "moka_pot", None, 0, None)]
    assert _named("turn on the stove and put the moka pot on it") == [(1, 2, "put", "moka_pot", None, 0, None)]
    assert _named("open the top drawer and put the bowl inside") == [(1, 2, "put", "akita_black_bowl", None, 0, None)]
    # steps that pick nothing up have no entry; without the sequencer a compound instruction is one clause
    assert _named("open the middle drawer of the cabinet") == [] and _named("turn on the stove") == []
    assert _named("put both the alphabet soup and the tomato sauce in the basket", sequence=False) == []
    # the percepts of the type are counted, and nothing of the tool is changed
    t = _tool("akita_black_bowl", "plate", "akita_black_bowl")
    before = [dict(q) for q in t.percepts]
    assert _named("put the bowl on the plate", t) == [(0, 1, "put", "akita_black_bowl", None, 2, None)]
    assert t.percepts == before


def test_a_single_percept_of_the_type_is_asked_whether_it_clearly_fails_the_relation(monkeypatch):
    seen = []

    def fails(parsed, cand, percepts, lib):
        seen.append((parsed["target_type"], cand["id"], len(percepts)))
        return True

    monkeypatch.setattr(GD, "single_fails", fails)
    ins = "Pick the akita black bowl next to the ramekin and place it on the plate"
    assert _named(ins, _tool("plate", "akita_black_bowl")) == [(0, 1, "pick", "akita_black_bowl", "next_to", 1, True)]
    assert seen == [("akita_black_bowl", "p1", 2)]
    # two percepts of the type, or none: not asked
    assert _named(ins, _tool("akita_black_bowl", "akita_black_bowl"))[0][5:] == (2, None) and len(seen) == 1
    monkeypatch.setattr(GD, "single_fails", lambda *a: (_ for _ in ()).throw(RuntimeError("geometry")))
    assert _named(ins, _tool("akita_black_bowl"))[0][5:] == (1, None)  # an error is "cannot be told"


def _R(named=None):
    def named_targets(instruction, tool, lib, params):
        return EP.named_targets(instruction, tool, lib, params) if named is None else named

    return types.SimpleNamespace(parse=GD.parse, named_targets=named_targets)


ON = {"learn_forms": {}, "sequence": True}


def test_without_the_option_there_is_no_other_start_and_with_it_each_form_has_its_own():
    lib, tool = _lib(), _tool("plate")
    put = "put the bowl on the plate"
    assert LS.form_target(_R(), put, tool, lib, {"sequence": True}) == (None, None)  # off unless given
    assert LS.form_target(_R(), put, tool, lib, {"learn_forms": False, "sequence": True}) == (None, None)
    assert LS.form_target(types.SimpleNamespace(parse=GD.parse), put, tool, lib, ON) == (None, None)  # (M2's robot)
    assert LS.FORMS == {"put": True, "steps": True, "relation": True}
    # the goal form, one step, no percept of the object
    assert LS.form_target(_R(), put, tool, lib, ON) == \
        ("akita_black_bowl", {"why": "put", "step": 0, "steps": 1, "form": "put", "relation": None})
    assert LS.form_target(_R(), put, _tool("akita_black_bowl"), lib, ON) == (None, None)  # it has a percept
    assert LS.form_target(_R(), put, tool, lib, {**ON, "learn_forms": {"put": False}}) == (None, None)
    # several steps: the first step whose object has no percept
    both = "put both the alphabet soup and the tomato sauce in the basket"
    assert LS.form_target(_R(), both, _tool("basket", "alphabet_soup"), lib, ON) == \
        ("tomato_sauce", {"why": "steps", "step": 1, "steps": 2, "form": "put", "relation": None})
    assert LS.form_target(_R(), both, _tool("basket"), lib, ON)[0] == "alphabet_soup"
    assert LS.form_target(_R(), both, _tool("basket"), lib, {**ON, "learn_forms": {"steps": False}}) == (None, None)
    assert LS.form_target(_R(), "open the top drawer and put the bowl inside", _tool("wooden_cabinet"), lib, ON)[1]["why"] == "steps"
    # the registered form with no percept is the registered trigger's, not this option's
    pick = "Pick the akita black bowl next to the ramekin and place it on the plate"
    assert LS.form_target(_R(), pick, tool, lib, ON) == (None, None)
    assert LS.triggers(_R(), pick, tool, lib) == "akita_black_bowl"
    # a type of several parts is never learned
    assert LS.form_target(_R(), "put the cabinet on the plate", tool, _lib(), ON) == (None, None)


def test_the_relations_case_needs_one_percept_that_clearly_fails_and_an_object_asked_for_once():
    lib, tool = _lib(), _tool("akita_black_bowl", "plate")
    e = {"step": 0, "steps": 1, "clause": "c", "form": "pick", "type": "akita_black_bowl", "relation": "next_to",
         "have": 1, "single_fails": True}
    want = ("akita_black_bowl", {"why": "relation", "step": 0, "steps": 1, "form": "pick", "relation": "next_to"})
    assert LS.form_target(_R([e]), "x", tool, lib, ON) == want
    assert LS.form_target(_R([e]), "x", tool, lib, {**ON, "learn_forms": {"relation": False}}) == (None, None)
    for change in ({"single_fails": False}, {"single_fails": None}, {"have": 2}):
        assert LS.form_target(_R([{**e, **change}]), "x", tool, lib, ON) == (None, None), change
    # "both pots" with one of the two accepted: the type is asked for twice, and the learner stays out
    pots = [{**e, "type": "moka_pot", "steps": 2, "step": k, "form": "put", "relation": None} for k in (0, 1)]
    assert LS.form_target(_R(pots), "x", tool, lib, ON) == (None, None)
    # a reading that raises is no start
    R = types.SimpleNamespace(parse=GD.parse, named_targets=lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    assert LS.form_target(R, "x", tool, lib, ON) == (None, None)


class _Agent:
    finished = False


def _plan_R(calls):
    def plan_episode(instruction, tool, lib, cap, params):
        calls.append(instruction)
        return {"agent": _Agent(), "parsed": {"status": "ok"}, "grounding": {"status": "ok"}}

    return types.SimpleNamespace(parse=GD.parse, named_targets=EP.named_targets, plan_episode=plan_episode)


def test_plan_starts_the_learner_on_the_goal_form_only_with_the_option():
    lib, calls = _lib(), []
    R = _plan_R(calls)
    put = "put the bowl on the plate"
    out = LS.plan(R, put, _tool("plate"), lib, 300, {"learn_in_episode": True, "sequence": True}, None)
    assert out["learn"] == {"trigger": False} and calls == [put]  # as registered: the robot's own plan
    out = LS.plan(R, put, _tool("plate"), lib, 300, {"learn_in_episode": True, **ON}, None)
    assert type(out["agent"]).__name__ == "LearnThenAct" and calls == [put]  # no plan made yet: the looks come first
    assert out["agent"].T == "akita_black_bowl" and out["agent"].form["why"] == "put"
    assert out["learn"]["trigger"] is True and out["learn"]["form"] == \
        {"why": "put", "step": 0, "steps": 1, "form": "put", "relation": None}
    # the registered trigger carries no such record, with or without the option
    pick = "Pick the akita black bowl next to the ramekin and place it on the plate"
    for params in ({"learn_in_episode": True}, {"learn_in_episode": True, **ON}):
        out = LS.plan(R, pick, _tool("plate"), lib, 300, params, None)
        assert out["agent"].T == "akita_black_bowl" and out["agent"].form is None and "form" not in out["learn"]
    # too late to finish: the robot's own plan, with the option as without
    out = LS.plan(R, put, _tool("plate"), lib, 100, {"learn_in_episode": True, "learn_late": {}, **ON}, None)
    assert out["learn"]["trigger"] is False and out["learn"]["late"]["target_type"] == "akita_black_bowl"


def test_the_look_from_above_asks_the_own_type_in_the_relations_case_and_the_cap_on_attempts_is_one_steps():
    lib = _lib()
    ask = {**LS.ASK, "own": True}

    def learner(form, have):
        a = LS.LearnThenAct.__new__(LS.LearnThenAct)  # (no generator: only the helper is used)
        a.R, a.instruction, a.tool, a.lib, a.T, a.form = _R(), "x", _tool(*have), lib, "akita_black_bowl", form
        return a

    # as registered: the own type is asked only when the first look accepted none of it
    assert learner(None, ("plate",))._ask_types(ask, {}) == ["akita_black_bowl"]
    assert learner(None, ("plate", "akita_black_bowl"))._ask_types(ask, {}) == []
    # the relation's case: the accepted one is another object, so the own type is asked
    assert learner({"why": "relation"}, ("plate", "akita_black_bowl"))._ask_types(ask, {}) == ["akita_black_bowl"]
    assert learner({"why": "put"}, ("plate", "akita_black_bowl"))._ask_types(ask, {}) == []
    assert LS.caps_attempts(None) and LS.caps_attempts({"why": "put", "steps": 1})
    assert not LS.caps_attempts({"why": "steps", "steps": 2})


# ---------------------------------------------------------------- ``learn_hint`` (2026-11-19, off unless given)

def test_the_thing_at_the_hinted_place_goes_first_and_without_one_near_the_order_stands():
    import numpy as np

    from rowant_lab.libero_env import lfb_episode as LE

    # three things: a large one at x 0.3, a small one at the origin, a tiny one at x -0.3 (voxel centres, metres)
    big = [[0.3 + 0.01 * i, 0.01 * j, 0.9] for i in range(8) for j in range(8)]
    small = [[0.01 * i, 0.0, 0.9] for i in range(6)]
    tiny = [[-0.3, 0.0, 0.9]]
    V = np.array(big + small + tiny)
    comps = [np.arange(0, 64), np.arange(64, 70), np.arange(70, 71)]
    order, d = LE.hint_order(V, comps, [0.02, 0.01], 0.06, 8, 3)
    assert order == [1, 0, 2] and d[1] < 0.02 and d[0] > 0.2
    # a place no thing is near: as it was (largest first)
    assert LE.hint_order(V, comps, [0.0, 0.5], 0.06, 8, 3)[0] == [0, 1, 2]
    # a thing too small to learn is never put first, however near
    assert LE.hint_order(V, comps, [-0.3, 0.0], 0.06, 8, 3)[0] == [0, 1, 2]
    # only the first ``top`` things are looked at
    assert LE.hint_order(V, comps, [0.02, 0.01], 0.06, 1, 3)[0] == [0]
    assert LE.HINT == {"within_m": 0.06, "top": 8}


def _hint_R(source="blob_stand", typ="akita_black_bowl", fallback=True, boom=False):
    seen = {}

    def plan_episode(instruction, tool, lib, cap, params):
        if boom:
            raise RuntimeError("no plan")
        seen.update(instruction=instruction, sequence=params.get("sequence"), tool=tool)
        tool.percepts.append({"id": "f1", "type": typ, "source": source,
                              "bodies": {"main": {"pos": [0.12, -0.34, 0.91]}}})
        if fallback:
            tool.fallbacks.append({"role": "target", "type": typ, "id": "f1", "source": source})
        tool.table_bound = True
        return {"agent": object(), "grounding": {"status": "ok", "target": "f1"}}

    return types.SimpleNamespace(parse=GD.parse, named_targets=EP.named_targets, plan_episode=plan_episode), seen


def _hint_lib():
    lib = _lib()
    for d in lib.export_types.values():
        d["root_body"] = "main"
    return lib


def test_the_robots_own_plan_is_made_on_a_copy_of_the_tool_and_only_its_fallbacks_place_is_taken():
    lib = _hint_lib()
    tool = types.SimpleNamespace(percepts=[{"id": "p0", "type": "plate"}], fallbacks=[], table_bound=False, extra={"blobs": []})
    R, seen = _hint_R()
    got = LS.fallback_place(R, "put the bowl on the plate", tool, lib, 300, {"sequence": True, "m1": {}}, "akita_black_bowl")
    assert got == {"xy": [0.12, -0.34], "source": "blob_stand"}
    # the plan was made for the one clause, without the sequencer, on another tool: the real one is as it was
    assert seen["instruction"] == "put the bowl on the plate" and seen["sequence"] is None and seen["tool"] is not tool
    assert tool.percepts == [{"id": "p0", "type": "plate"}] and tool.fallbacks == [] and tool.table_bound is False
    # an accepted percept, another type, no plan: no place
    assert LS.fallback_place(_hint_R(fallback=False)[0], "c", tool, lib, 300, {}, "akita_black_bowl") is None
    assert LS.fallback_place(_hint_R(typ="plate")[0], "c", tool, lib, 300, {}, "akita_black_bowl") is None
    assert LS.fallback_place(_hint_R(boom=True)[0], "c", tool, lib, 300, {}, "akita_black_bowl") is None
    assert tool.percepts == [{"id": "p0", "type": "plate"}] and tool.fallbacks == []


def test_the_place_is_sent_only_with_the_option():
    lib = _hint_lib()
    put = "put the bowl on the plate"

    def start(params):
        tool = types.SimpleNamespace(percepts=[{"id": "p0", "type": "plate"}], fallbacks=[], table_bound=False)
        return LS.plan(_hint_R()[0], put, tool, lib, 300, {"learn_in_episode": True, **ON, **params}, None)

    out = start({})
    assert out["agent"].hint is None and "hint" not in out["learn"]  # off unless given
    out = start({"learn_hint": {}})
    assert out["agent"].hint == {"xy": [0.12, -0.34], "source": "blob_stand"} and out["learn"]["hint"] == out["agent"].hint
    # with the registered trigger too (the clause is the instruction)
    pick = "Pick the akita black bowl next to the ramekin and place it on the plate"
    tool = types.SimpleNamespace(percepts=[{"id": "p0", "type": "plate"}], fallbacks=[], table_bound=False)
    R, seen = _hint_R()
    out = LS.plan(R, pick, tool, lib, 300, {"learn_in_episode": True, "learn_hint": {}}, None)
    assert out["agent"].form is None and out["agent"].hint["xy"] == [0.12, -0.34] and seen["instruction"] == pick


# ---------------------------------------------------------------- ``learn_forms["own_first"]`` (2026-11-19)

OWN = {"learn_in_episode": True, "sequence": True, "learn_after_fail": {"retreat": 20, "min_left": 190},
       "learn_forms": {"steps": False, "own_first": ["put"]}}


def test_the_own_attempt_first_is_asked_for_the_named_starts_only():
    put = {"why": "put", "step": 0, "steps": 1, "form": "put", "relation": None}
    rel = {**put, "why": "relation", "form": "pick", "relation": "next_to"}
    want = {**LS.AFTER_FAIL, "retreat": 20, "min_left": 190}
    assert LS.after_fail(OWN) == want  # (the runner arms its check)
    assert LS.own_first(OWN, put) == want
    assert LS.own_first(OWN, rel) is None and LS.own_first(OWN, None) is None  # the look first, as built
    # without ``learn_after_fail`` there is no own attempt first, whatever the list says; nor with the learner off
    assert LS.after_fail({"learn_in_episode": True, **ON}) is None and LS.own_first({"learn_in_episode": True, **ON}, put) is None
    only_list = {"learn_in_episode": True, "learn_forms": {"own_first": ["put"]}}
    assert LS.after_fail(only_list) is None and LS.own_first(only_list, put) is None
    assert LS.after_fail({**OWN, "learn_in_episode": False}) is None and LS.own_first({**OWN, "learn_in_episode": False}, put) is None
    # the option of 11-18 without the list is for every start, as it was
    old = {"learn_in_episode": True, "learn_after_fail": {"retreat": 5}}
    assert LS.after_fail(old) == {**LS.AFTER_FAIL, "retreat": 5}
    assert LS.own_first(old, None) == LS.own_first(old, rel) == LS.own_first(old, put) == {**LS.AFTER_FAIL, "retreat": 5}
    # an empty list: no start waits
    none = {**OWN, "learn_forms": {"own_first": []}}
    assert LS.own_first(none, put) is None and LS.own_first(none, None) is None


def test_a_put_start_waits_for_the_robots_own_attempt_and_starts_after_it_failed():
    lib, calls = _lib(), []
    R = _plan_R(calls)
    put = "put the bowl on the plate"
    out = LS.plan(R, put, _tool("plate"), lib, 300, OWN, None)
    assert type(out["agent"]).__name__ == "_Agent" and calls == [put]  # the robot's own plan, the learner waiting
    assert out["learn"] == {"trigger": False, "own_first": {"pending": True, "target_type": "akita_black_bowl", "form": "put"}}
    # the runner's record of the failed start: the learner now, from the new look
    failed = {"t": 61, "why": "attempt", "left": 239, "min_left": 190, "attempts": ["c1_fail", None]}
    out = LS.plan_after_fail(R, put, _tool("plate"), lib, 219, OWN, None, failed)
    assert type(out["agent"]).__name__ == "LearnThenAct" and out["agent"].form["why"] == "put" and calls == [put]
    assert out["learn"]["own_first"] == {**failed, "pending": False, "started": True}
    # the new look accepts the object after all: the robot's own plan, and the record says why the learner stayed out
    out = LS.plan_after_fail(R, put, _tool("plate", "akita_black_bowl"), lib, 219, OWN, None, failed)
    assert type(out["agent"]).__name__ == "_Agent" and out["learn"]["own_first"]["why"] == "accepted"
    # the steps' start is off in this form
    both = "put both the alphabet soup and the tomato sauce in the basket"
    out = LS.plan(R, both, _tool("basket"), lib, 520, OWN, None)
    assert out["learn"] == {"trigger": False}


def test_the_relations_start_and_the_registered_one_do_not_wait():
    lib, calls = _lib(), []
    e = {"step": 0, "steps": 1, "clause": "c", "form": "pick", "type": "akita_black_bowl", "relation": "next_to",
         "have": 1, "single_fails": True}
    R = types.SimpleNamespace(parse=GD.parse, named_targets=lambda *a: [e],
                              plan_episode=lambda *a: calls.append(a[0]) or {"agent": _Agent(), "grounding": {}})
    out = LS.plan(R, "x", _tool("akita_black_bowl", "plate"), lib, 220, OWN, None)
    assert type(out["agent"]).__name__ == "LearnThenAct" and out["agent"].form["why"] == "relation" and calls == []
    pick = "Pick the akita black bowl next to the ramekin and place it on the plate"
    out = LS.plan(_plan_R(calls), pick, _tool("plate"), lib, 220, OWN, None)
    assert type(out["agent"]).__name__ == "LearnThenAct" and out["agent"].form is None and calls == []
