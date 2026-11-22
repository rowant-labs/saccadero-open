"""Asking the known types about the thing after the look from above, the next thing, and the floor for jumping the
queue (2026-11-18): all off unless given (``lfb_episode.ASK_KNOWN``, ``RANK_FLOOR``; ``learn_skill.ASK``)."""

from __future__ import annotations

import json
import math
import types
from pathlib import Path

import numpy as np
import pytest

from rowant_lab.libero_env import lfb_episode as LE
from rowant_lab.libero_env import m2_audit as AU
from rowant_lab.libero_env.tests import test_lfb_episode as TE
from rowant_lab.pro2 import learn_skill as LS
from rowant_lab.pro2 import machine as MC

TABLE_Z = 0.9
FLOORS = {"inlier_min": 0.9, "seen_min": 0.5, "cover_min": 0.85}


def _top_scene(true_box, models: dict, part=None):
    """One look from nearly above a box on a table; ``models``: type -> box dims; ``part``: keep only the thing's
    points with x on that side of the box centre (a thing seen in part)."""
    K = np.array([[160.0, 0, 80.0], [0, 160.0, 80.0], [0, 0, 1.0]])
    (cx, cy), _yaw, _half, h = true_box
    tgt = np.array([cx, cy, TABLE_Z + h / 2])
    el = math.radians(80.0)
    T = TE._look_at(tgt + 0.30 * np.array([math.cos(el), 0.0, math.sin(el)]), tgt)
    z, m = TE._render(T, K, (160, 160), true_box, TABLE_Z)
    obj = LE._lift({"z": z, "K": K, "T": T, "mask": m & (z > 0)})
    if part is not None:
        obj = obj[(obj[:, 0] - cx) * part > 0]
    wc = types.SimpleNamespace(types={t: {"root_body": f"{t}_main"} for t in models},
                               gid={(t, f"{t}_main"): i for i, t in enumerate(models)},
                               nodes={str(i): TE._box_nodes(*d) for i, d in enumerate(models.values())},
                               rest_up={str(i): np.array([0.0, 0.0, 1.0]) for i in range(len(models))})
    per = types.SimpleNamespace(p={"agree_m": 0.01, "see_through_m": 0.015, "facing_cos": 0.15})
    obs = [(np.where(z > 0, z, np.nan), K, T)]
    plane = {"normal_w": [0.0, 0.0, 1.0], "height_w": TABLE_Z}
    return per, wc, obj, obs, plane


def test_the_true_type_explains_the_thing_and_its_pose_is_found():
    box = ((0.10, -0.05), math.radians(30.0), (0.04, 0.03), 0.10)
    per, wc, obj, obs, plane = _top_scene(box, {"butter": (0.04, 0.03, 0.10)})
    out = LE.fit_known(per, wc, "butter", obj, obs, plane, {**LE.ASK_KNOWN, **FLOORS})
    assert out["explains"] and out["status"] == "fit", out
    assert out["inlier"] >= 0.9 and out["seen"] >= 0.5 and out["cover"] >= 0.85
    t = np.asarray(out["t_w"])
    assert float(np.linalg.norm(t[:2] - [0.10, -0.05])) < 0.008 and abs(t[2] - TABLE_Z) < 0.006
    assert out["search_m"] == round(float(np.clip(0.05, 0.03, 0.12)), 4)  # the model's own half width


def test_a_large_thing_seen_in_part_is_still_found_within_its_own_half_width():
    box = ((0.0, 0.0), 0.0, (0.08, 0.07), 0.09)
    per, wc, obj, obs, plane = _top_scene(box, {"basket": (0.08, 0.07, 0.09)}, part=1.0)
    assert abs(float(obj[:, 0].mean())) > 0.03  # the part's centre is not the thing's
    out = LE.fit_known(per, wc, "basket", obj, obs, plane, {**LE.ASK_KNOWN, **FLOORS})
    assert out["explains"], out
    assert float(np.linalg.norm(np.asarray(out["t_w"])[:2])) < 0.01 and out["search_m"] > 0.1


def test_another_type_does_not_explain_it_and_no_floor_means_no():
    box = ((0.10, -0.05), 0.3, (0.04, 0.03), 0.10)
    per, wc, obj, obs, plane = _top_scene(box, {"butter": (0.04, 0.03, 0.10), "small": (0.02, 0.02, 0.05),
                                                "big": (0.08, 0.07, 0.14)})
    cfg = {**LE.ASK_KNOWN, **FLOORS}
    small, big = (LE.fit_known(per, wc, t, obj, obs, plane, cfg) for t in ("small", "big"))
    assert not small["explains"] and small["cover"] < 0.85, small  # the thing reaches beyond the model
    assert not big["explains"] and (big["cover"] < 0.85 or big["inlier"] < 0.9 or big["seen"] < 0.5), big
    # the acceptance numbers are not set by default: nothing is explained
    assert all(LE.ASK_KNOWN[k] is None for k in FLOORS) and LE.ASK_KNOWN["act"] is True
    assert not LE.fit_known(per, wc, "butter", obj, obs, plane, dict(LE.ASK_KNOWN))["explains"]
    assert LE.fit_known(per, wc, "none", obj, obs, plane, cfg) == {"type": "none", "explains": False, "status": "no_type"}
    assert LE.fit_known(per, wc, "butter", obj[:5], obs, plane, cfg)["status"] == "no_points"


def test_the_things_tried_before_are_left_out_of_the_next_choice():
    a = np.array([[0.0, 0.0, 0.05], [0.003, 0.0, 0.05], [0.006, 0.0, 0.05]])
    b = np.array([[0.2, 0.1, 0.05], [0.203, 0.1, 0.05]])
    U = np.concatenate([a, b, [[0.0, 0.007, 0.05]], [[0.0, 0.02, 0.05]]])
    left = LE.drop_tried(U, [a], 0.0075)
    assert len(left) == 3 and np.allclose(left[:2], b) and np.allclose(left[2], [0.0, 0.02, 0.05])
    assert len(LE.drop_tried(U, [a, b], 0.0075)) == 1 and LE.drop_tried(U, [], 0.0075) is U


def test_a_small_thing_does_not_jump_the_queue_under_the_floor():
    # round B, task 2: a 75-voxel thing went first because the four larger ones were covered
    sizes, covers = [9077, 1217, 155, 94, 75], [0.907, 0.859, 1.0, 0.904, 0.0]
    assert LE.rank_components(sizes, covers, 0.6, 60, 0.1) == [4, 0, 1, 2, 3]
    assert LE.rank_components(sizes, covers, 0.6, 60, 0.1, floor=500) == [0, 1, 2, 3, 4]
    # task 17: 342 voxels before 9,296 and 1,110
    sizes, covers = [9296, 1110, 342, 73, 28], [0.894, 0.724, 0.0, 0.0, 0.0]
    assert LE.rank_components(sizes, covers, 0.6, 60, 0.1) == [2, 3, 0, 1, 4]
    # (with the floor the margin rule still orders the two large ones: the less covered goes first)
    assert LE.rank_components(sizes, covers, 0.6, 60, 0.1, floor=500) == [1, 0, 2, 3, 4]
    assert LE.rank_components(sizes, covers, 0.6, 60, None, floor=500) == [0, 1, 2, 3, 4]
    # a thing of its own size still goes before a covered larger one, and with nothing covered nothing changes
    assert LE.rank_components([3000, 1500, 90], [0.9, 0.0, 0.0], 0.6, 60, 0.1, floor=500) == [1, 0, 2]
    assert LE.rank_components([3000, 1500, 90], [0.0, 0.0, 0.0], 0.6, 60, None, floor=500) == [0, 1, 2]
    assert LE.RANK_FLOOR == {"min_vox": 500}


# ---------------------------------------------------------------------------------------------- the robot's side
POSE = {"eef_pos": [0.1, 0.0, 0.4], "eef_quat_wxyz": [0.0, 1.0, 0.0, 0.0]}
OBS = {"robot0_joint_pos": [0.0] * 7, "robot0_gripper_qpos": [0.02, -0.02], "robot0_eef_pos": [0.1, 0.0, 0.4],
       "robot0_eef_quat": [1.0, 0.0, 0.0, 0.0]}


class _Server:
    """The first ``n_known`` things are explained by ``known_type`` (None: by the first type asked); the next one is
    found again and learned."""

    def __init__(self, n_known=1, known_type="basket", t_w=(0.1, 0.2, 0.0)):
        self.msgs, self.n_known, self.n_flank, self.known_type, self.t_w = [], n_known, 0, known_type, list(t_w)

    def call(self, msg):
        self.msgs.append(msg)
        look = lambda i, kind: {"look": i, "kind": kind, "az_used_deg": 0.0, "el_used_deg": 90.0, "tries": [],  # noqa: E731
                                "waypoints": [], **POSE}
        if msg["op"] == "learn_plan" and msg.get("stage") == "top":
            k = sum(1 for m in self.msgs if m.get("stage") == "top")
            return {"ok": True, "status": "ok", "stage": "top", "n_planned": 1, "looks": [look(0, "top")],
                    "leftover": {"centre_w": [0.1 * k, 0.0, 0.05], "aim_w": [0.1 * k, 0.0, 0.05]}}
        if msg["op"] == "learn_plan" and msg.get("stage") == "flanks":
            self.n_flank += 1
            lo = {"centre_w": [0.1 * self.n_flank, 0.01, 0.04], "n_vox_object": 700, "plane": {"height_w": 0.0}}
            ask = [{"type": t, "cover": 0.99, "t_w": self.t_w, "R_w": np.eye(3).tolist()}
                   for t in (msg.get("ask") or {}).get("types") or []]
            kt = self.known_type or (ask[0]["type"] if ask else None)
            if self.n_flank <= self.n_known and any(a["type"] == kt for a in ask):
                return {"ok": True, "status": "known", "stage": "flanks", "leftover": lo, "ask": ask,
                        "known": {"type": kt, "gid": "3", "R_w": np.eye(3).tolist(), "t_w": self.t_w,
                                  "inlier": 0.97, "seen": 0.7, "cover": 0.99, "n_agree": 150}}
            return {"ok": True, "status": "ok", "stage": "flanks", "n_planned": 1, "looks": [look(0, "flank")],
                    "leftover": lo, **({"ask": ask} if ask else {})}
        return {"ok": True, "status": "no_masks"}


class _R:
    def __init__(self):
        self.calls = []

    @staticmethod
    def parse(instruction, types_):
        return {"status": "ok", "target_type": "milk", "place_type": "basket"}

    def plan_episode(self, instruction, tool, lib, left, params):
        self.calls.append(([p["id"] for p in tool.percepts if p.get("id")], left))
        return {"agent": None, "grounding": {"status": "fail"}, "parsed": {}}


class _Lib:
    types = {"milk": 1, "basket": 1, "butter": 1, "ketchup": 1, "soup": 1}
    export_types = {t: {"parts": {"a": 1}, "root_body": f"{t}_main"} for t in types}


def _tool(server, ketchup_at=None):
    n = {"w": 0}

    class Cam:
        @staticmethod
        def wrist_frame():
            n["w"] += 1
            return {"camera": LS.WRIST, "n": n["w"]}

    class Tool:
        client, cam, suite, episode, bridge_s, looks = server, Cam(), "libero_object", "0123456789abcdef", 0.0, []

        def __init__(self):
            self.obs = OBS
            self.percepts = [{"id": "p1", "type": "ketchup", "gid": "1", "bodies": {} if ketchup_at is None else {
                "ketchup_main": {"pos": list(ketchup_at), "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}}]
            self.extra = {"rejected": [{"type": "butter", "gid": "2", "bodies": {"b": 1}},
                                       {"type": "ketchup", "gid": "1", "bodies": {"b": 1}},
                                       {"type": "milk", "gid": "0", "bodies": {"b": 1}},
                                       {"type": "butter", "gid": "2", "bodies": {"b": 2}}]}

        def _frame(self, camera):
            return {"camera": camera}

        @staticmethod
        def _build(typ, R_w, t_w):
            return {f"{typ}_main": {"pos": [float(x) for x in t_w], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}

        def wrist_find(self, typ, support_z, hint_xy=None, opts=None):
            return None

    return Tool()


def _run(params, server, cap=280, ketchup_at=None, prepare=None):
    R, tool = _R(), _tool(server, ketchup_at)
    if prepare is not None:
        prepare(tool)
    agent = LS.LearnThenAct(R, "pick the milk and place it in the basket", tool, _Lib, cap,
                            {"learn_top_first": {}, "learn_looks": "fast", "learn_rank_known": {}, **params}, "milk",
                            {"plane": {"normal_w": [0, 0, 1], "height_w": 0.0}})
    agent.act(OBS)  # every look is where the hand is: the whole learning phase runs in this call
    return agent, tool, R


def test_a_thing_a_known_type_explains_becomes_a_percept_and_the_next_thing_is_looked_at():
    server = _Server(n_known=1)
    agent, tool, R = _run({"learn_ask_known": {"rejected": True}}, server)  # (the measurement's wider list of types)
    ops = [(m["op"], m.get("stage"), bool(m.get("next"))) for m in server.msgs]
    assert ops == [("learn_plan", "top", False), ("learn_plan", "flanks", False), ("learn_plan", "top", True),
                   ("learn_plan", "flanks", False), ("learn", None, False)]
    # asked: the place's type first, then the rejected poses' types: never the instruction's own, never an accepted one
    assert server.msgs[1]["ask"] == {"types": ["basket", "butter"]}
    assert server.msgs[3]["ask"] == {"types": ["butter"]}  # the basket has a percept now, 22 cm from this thing
    nxt = server.msgs[2]
    assert set(nxt) == set(server.msgs[0]) | {"next"} and nxt["cameras"] is server.msgs[0]["cameras"]
    assert [p["type"] for p in nxt["percepts"]] == ["ketchup", "basket"] and nxt["wrist"]["cameras"][0]["n"] == 3
    assert [q["type"] for q in nxt["rejected"]] == ["butter", "butter"] == [q["type"] for q in server.msgs[0]["rejected"]]
    rec = agent.rec
    assert [k["type"] for k in rec["known"]] == ["basket"] and rec["known"][0]["id"] == "known0"
    assert rec["known"][0]["cover"] == 0.99 and rec["known"][0]["centre_w"] == [0.1, 0.01, 0.04]
    assert rec["known"][0]["first_centre_w"] == [0.1, 0.0, 0.05] and "known_end" not in rec
    kept = [p for p in tool.percepts if p.get("id") == "known0"][0]
    assert kept["type"] == "basket" and kept["source"] == "top_look" and kept["bodies"]["basket_main"]["pos"] == [0.1, 0.2, 0.0]
    assert rec["plan"]["leftover"]["centre_w"] == [0.2, 0.0, 0.05]  # the record is the thing gone on with
    assert rec["plan2"]["status"] == "ok" and [lk.get("kind") for lk in rec["looks"]] == ["top", "top", None]
    assert R.calls[0][0] == ["p1", "known0"]  # the robot's own plan (nothing was learned here) holds the basket
    assert rec["plan2"]["ask"] == [{"type": "butter", "cover": 0.99}]  # numbers only: no pose in the record
    # as it is switched on, only the place's type is asked here (no accepted percept lies near the thing), and once
    # the basket has a percept nothing is left to ask about the next thing
    plain = _Server(n_known=1)
    a2, _t2, _r2 = _run({"learn_ask_known": {}}, plain)
    assert plain.msgs[1]["ask"] == {"types": ["basket"]} and "ask" not in plain.msgs[3]
    assert [k["type"] for k in a2.rec["known"]] == ["basket"] and "ask" not in a2.rec["plan2"]
    assert [(m["op"], m.get("stage")) for m in plain.msgs] == [(m["op"], m.get("stage")) for m in server.msgs]


def test_a_part_of_an_object_the_first_look_placed_a_little_off_moves_that_percept():
    # the thing (first seen at x 0.1) lies 5 cm from the accepted ketchup: its type is asked, after the place's
    server = _Server(n_known=1, known_type="ketchup", t_w=(0.13, 0.0, 0.0))
    agent, tool, _R_ = _run({"learn_ask_known": {}}, server, ketchup_at=(0.15, 0.0, 0.0))
    assert server.msgs[1]["ask"] == {"types": ["basket", "ketchup"]}
    k = agent.rec["known"][0]
    assert k["type"] == "ketchup" and k["id"] == "p1" and k["had"] == "p1" and k["moved_m"] == 0.02
    assert [p["id"] for p in tool.percepts] == ["p1"]  # the same object, not a second one
    assert tool.percepts[0]["bodies"]["ketchup_main"]["pos"] == [0.13, 0.0, 0.0] and tool.percepts[0]["source"] == "top_look"
    # the next thing lies 7 cm from the moved ketchup: it may be another part of it, so the type is asked again
    assert server.msgs[3]["ask"] == {"types": ["basket", "ketchup"]}
    # an object beyond 15 cm of the thing is not asked
    far = _Server(n_known=1, known_type="ketchup")
    _a, _t, _r = _run({"learn_ask_known": {}}, far, ketchup_at=(0.30, 0.0, 0.0))
    assert far.msgs[1]["ask"] == {"types": ["basket"]}


def test_a_fit_that_puts_an_accepted_object_far_from_its_place_is_not_believed():
    server = _Server(n_known=1, known_type="ketchup", t_w=(0.02, 0.0, 0.0))  # 14 cm from where the look had put it
    agent, tool, _R_ = _run({"learn_ask_known": {}}, server, ketchup_at=(0.16, 0.0, 0.0))
    ops = [(m["op"], m.get("stage"), "ask" in m) for m in server.msgs]
    assert ops == [("learn_plan", "top", False), ("learn_plan", "flanks", True), ("learn_plan", "flanks", False),
                   ("learn", None, False)]  # asked again without the types: the thing is gone on with as unknown
    assert "known" not in agent.rec and tool.percepts[0]["bodies"]["ketchup_main"]["pos"] == [0.16, 0.0, 0.0]
    p2 = agent.rec["plan2"]
    assert p2["known_refused"] == {"type": "ketchup", "had": "p1", "moved_m": 0.14} and p2["status"] == "ok"
    assert [a.get("moved_m") for a in p2["ask"]] == [None, 0.14] and len(p2["looks"]) == 1
    assert LS.ASK["move_max_m"] == 0.03  # the six basket parts measured lay 8 to 9 mm from the accepted basket


def test_after_a_thing_was_explained_a_thing_not_found_again_is_passed_over():
    class Gone(_Server):  # the second thing is not there in the look from above
        def call(self, msg):
            if msg["op"] == "learn_plan" and msg.get("stage") == "flanks" and self.n_flank == 1:
                self.msgs.append(msg)
                self.n_flank += 1
                return {"ok": True, "status": "no_refind", "stage": "flanks", "leftover": {"plane": {"height_w": 0.0}}}
            return super().call(msg)

    server = Gone(n_known=1)
    agent, tool, _R_ = _run({"learn_ask_known": {}}, server)
    assert [(m["op"], m.get("stage"), bool(m.get("next"))) for m in server.msgs] == [
        ("learn_plan", "top", False), ("learn_plan", "flanks", False), ("learn_plan", "top", True),
        ("learn_plan", "flanks", False), ("learn_plan", "top", True), ("learn_plan", "flanks", False), ("learn", None, False)]
    rec = agent.rec
    assert [k["type"] for k in rec["known"]] == ["basket"] and len(rec["gone"]) == 1
    assert rec["gone"][0]["first_centre_w"] == [0.2, 0.0, 0.05]
    assert rec["plan"]["leftover"]["centre_w"] == pytest.approx([0.3, 0.0, 0.05])  # the third thing is gone on with
    assert rec["plan2"]["status"] == "ok" and "known_end" not in rec
    # a first thing that is not found again ends the learning as before: nothing was explained yet

    class GoneFirst(_Server):
        def call(self, msg):
            if msg["op"] == "learn_plan" and msg.get("stage") == "flanks":
                self.msgs.append(msg)
                return {"ok": True, "status": "no_refind", "stage": "flanks", "leftover": {}}
            return super().call(msg)

    first = GoneFirst()
    a1, _t, _r = _run({"learn_ask_known": {}}, first)
    assert [(m["op"], m.get("stage")) for m in first.msgs] == [("learn_plan", "top"), ("learn_plan", "flanks")]
    assert "gone" not in a1.rec and a1.rec["fallback"] is True


def test_an_episode_tries_three_things_at_most_and_only_while_the_steps_allow():
    def more(tool):  # a third type to ask about
        tool.extra["rejected"].append({"type": "soup", "gid": "4", "bodies": {"b": 1}})

    wide = {"learn_ask_known": {"rejected": True}}
    agent, _tool_, R = _run(wide, _Server(n_known=9, known_type=None), prepare=more)
    assert len(agent.rec["known"]) == 3 and agent.rec["known_end"] == "count" and agent.rec["fallback"] is True
    assert [(k["id"], k["type"]) for k in agent.rec["known"]] == [("known0", "basket"), ("known1", "butter"),
                                                                   ("known2", "soup")]
    two, _t, _r = _run({"learn_ask_known": {"rejected": True, "max_things": 2}}, _Server(n_known=9, known_type=None),
                       prepare=more)
    assert len(two.rec["known"]) == 2 and two.rec["known_end"] == "count"
    # every type asked is found: nothing is left to ask about the third thing, which is gone on with
    none_left, _t, _r = _run(wide, _Server(n_known=9, known_type=None))
    assert len(none_left.rec["known"]) == 2 and "known_end" not in none_left.rec
    # the looks' budget ends at cap - 150: with 91 steps asked for a further thing, a cap of 240 allows none
    short, _t, _r = _run({"learn_ask_known": {}}, _Server(n_known=9), cap=240)
    assert len(short.rec["known"]) == 1 and short.rec["known_end"] == "steps"
    assert LS.ASK == {"max_things": 3, "next_need": 91, "max_types": 8, "near_m": 0.15, "move_max_m": 0.03,
                      "rejected": False}
    assert LS.ACT_RESERVE == 150


def test_without_the_key_nothing_is_asked_and_the_flow_is_the_old_one():
    server = _Server(n_known=9)
    agent, tool, _R_ = _run({}, server)
    assert [(m["op"], m.get("stage")) for m in server.msgs] == [("learn_plan", "top"), ("learn_plan", "flanks"),
                                                                 ("learn", None)]
    assert all("ask" not in m and "next" not in m for m in server.msgs)
    assert "known" not in agent.rec and "ask" not in agent.rec["plan2"] and [p["id"] for p in tool.percepts] == ["p1"]


def test_the_audit_declares_the_two_request_keys_and_the_options_are_off_by_default():
    class Echo:
        def call(self, msg):
            return {"ok": True}

        def close(self):
            pass

    c = AU.AuditedClient(Echo())
    c.call({"op": "learn_plan", "stage": "flanks", "suite": "s", "episode": "0123456789abcdef", "top": {},
            "ask": {"types": ["basket"]}})
    c.call({"op": "learn_plan", "stage": "top", "suite": "s", "episode": "0123456789abcdef", "next": True})
    assert not c.violations
    c.call({"op": "learn_plan", "suite": "s", "episode": "0123456789abcdef", "truth": 1})
    assert c.violations == ["learn_plan: undeclared keys ['truth']"]
    srv = Path(LE.__file__).read_text()
    assert srv.count('per.p.get("learn_ask_known")') == 1 and srv.count('per.p.get("learn_rank_floor")') == 1
    assert srv.count('if top_first and msg.get("next"):') == 1
    skill = Path(LS.__file__).read_text()
    assert skill.count('self.params.get("learn_ask_known") if top_first else None') == 1
    from rowant_lab.pro2 import long_dev as LD

    b = json.dumps(LD.bundle(LD.colour_long_params(MC.REGISTERED["net"])))
    assert all(k not in b for k in ("learn_ask_known", "learn_rank_floor", "learn_in_episode"))


# ---------------------------------------------------------------------------- ``own``: the instruction's own type too
def test_with_own_the_instructions_type_is_asked_first_and_a_thing_it_explains_is_acted_on_at_once():
    server = _Server(n_known=1, known_type="milk")
    agent, tool, R = _run({"learn_ask_known": {"own": True}}, server)
    # the look from above, the question, and nothing more: no next thing, no ring of looks, no model built
    assert [(m["op"], m.get("stage"), bool(m.get("next"))) for m in server.msgs] == [
        ("learn_plan", "top", False), ("learn_plan", "flanks", False)]
    assert server.msgs[1]["ask"] == {"types": ["milk", "basket"]}  # its own type first, then the place's
    rec = agent.rec
    assert [k["type"] for k in rec["known"]] == ["milk"] and rec["known_end"] == "own" and rec["acting"] == "own_top"
    assert "learn" not in rec and [lk.get("kind") for lk in rec["looks"]] == ["top"]
    kept = [p for p in tool.percepts if p.get("type") == "milk"]
    assert len(kept) == 1 and kept[0]["source"] == "top_look" and kept[0]["bodies"]["milk_main"]["pos"] == [0.1, 0.2, 0.0]
    assert R.calls == [(["p1", "known0"], 280)]  # the robot's own plan, with the percept just made and every step left


def test_with_own_a_thing_its_own_type_does_not_explain_is_learned_as_before():
    server = _Server(n_known=0, known_type="milk")
    agent, _tool_, _R_ = _run({"learn_ask_known": {"own": True}}, server)
    assert [(m["op"], m.get("stage")) for m in server.msgs] == [("learn_plan", "top"), ("learn_plan", "flanks"),
                                                               ("learn", None)]
    assert server.msgs[1]["ask"] == {"types": ["milk", "basket"]}
    assert "known" not in agent.rec and "known_end" not in agent.rec and agent.rec["acting"] != "own_top"
    assert [a["type"] for a in agent.rec["plan2"]["ask"]] == ["milk", "basket"]  # the fits are in the row


def test_without_own_the_instructions_type_is_not_asked_and_own_is_off_by_default():
    server = _Server(n_known=0, known_type="milk")
    _run({"learn_ask_known": {}}, server)
    assert server.msgs[1]["ask"] == {"types": ["basket"]} and "own" not in LS.ASK


def test_with_own_another_type_that_explains_the_thing_is_still_kept_and_the_next_thing_taken():
    server = _Server(n_known=1, known_type="basket")
    agent, _tool_, _R_ = _run({"learn_ask_known": {"own": True}}, server)
    assert [k["type"] for k in agent.rec["known"]] == ["basket"] and agent.rec.get("known_end") != "own"
    assert ("learn_plan", "top", True) in [(m["op"], m.get("stage"), bool(m.get("next"))) for m in server.msgs]
    assert server.msgs[3]["ask"] == {"types": ["milk"]}  # the next thing: its own type again; the basket has a percept


# -- ``own_near`` (2026-11-18 evening): a known object seen under a change goes back to the robot's own plan
class _FitServer(_Server):
    """``_Server`` whose answers carry a fit for the instruction's own type (the server's floors refuse it)."""

    def __init__(self, fit, **kw):
        super().__init__(n_known=0, known_type="milk", **kw)
        self.fit = fit

    def call(self, msg):
        rep = super().call(msg)
        for a in rep.get("ask") or []:
            a.update({"status": "fit", "explains": False, **(self.fit if a["type"] == "milk" else
                                                             {"inlier": 0.2, "seen": 0.1, "cover": 0.0})})
        return rep


def test_the_looser_floors_are_the_recorded_ones_and_not_among_the_pinned_defaults():
    assert LS.OWN_NEAR == {"inlier": 0.93, "seen": 0.45, "cover": 0.85}
    assert "own_near" not in LS.ASK and "own" not in LS.ASK


def test_a_known_object_seen_under_a_change_ends_the_looks_and_the_robots_own_plan_is_made_as_it_was():
    server = _FitServer({"inlier": 0.94, "seen": 0.49, "cover": 0.92})
    agent, tool, R = _run({"learn_ask_known": {"own_near": {}}}, server)
    assert [(m["op"], m.get("stage")) for m in server.msgs] == [("learn_plan", "top"), ("learn_plan", "flanks")]
    assert server.msgs[1]["ask"] == {"types": ["milk", "basket"]}  # its own type is asked, first
    rec = agent.rec
    assert rec["known_end"] == "own_near" and rec["acting"] == "own_near" and "known" not in rec and "learn" not in rec
    assert rec["own_near"] == {"inlier": 0.94, "seen": 0.49, "cover": 0.92,
                               "floors": {"inlier": 0.93, "seen": 0.45, "cover": 0.85}}
    assert rec["fallback"] is True
    # nothing of the look is kept: the percepts are the first look's, and the robot's plan gets every step left
    assert [p["id"] for p in tool.percepts] == ["p1"] and R.calls == [(["p1"], 280)]


@pytest.mark.parametrize("fit", [{"inlier": 0.90, "seen": 0.63, "cover": 0.99},   # a new object that resembles it
                                 {"inlier": 0.99, "seen": 0.65, "cover": 0.37},
                                 {"inlier": 1.0, "seen": 0.17, "cover": 0.0},
                                 {"inlier": 0.79, "seen": 0.32, "cover": 0.77}])
def test_a_thing_under_any_of_the_looser_floors_is_learned_as_before(fit):
    server = _FitServer(fit)
    agent, tool, R = _run({"learn_ask_known": {"own_near": {}}}, server)
    assert [(m["op"], m.get("stage")) for m in server.msgs] == [("learn_plan", "top"), ("learn_plan", "flanks"),
                                                               ("learn", None)]
    assert "own_near" not in agent.rec and agent.rec.get("known_end") is None and agent.rec["acting"] != "own_near"


def test_the_looser_check_is_off_unless_given_and_its_numbers_can_be_set():
    near = {"inlier": 0.94, "seen": 0.49, "cover": 0.92}
    # with ``own`` alone the server's floors decide, as in Step G: the thing is learned
    a1, _t, _r = _run({"learn_ask_known": {"own": True}}, _FitServer(near))
    assert "own_near" not in a1.rec and a1.rec.get("known_end") is None
    # without either key the own type is not asked at all
    s2 = _FitServer(near)
    _run({"learn_ask_known": {}}, s2)
    assert s2.msgs[1]["ask"] == {"types": ["basket"]}
    # stricter numbers of its own
    a3, _t, _r = _run({"learn_ask_known": {"own_near": {"seen": 0.6}}}, _FitServer(near))
    assert "own_near" not in a3.rec
    a4, _t, _r = _run({"learn_ask_known": {"own": True, "own_near": {"seen": 0.4}}}, _FitServer(near))
    assert a4.rec["known_end"] == "own_near" and a4.rec["own_near"]["floors"]["seen"] == 0.4


def test_a_thing_the_servers_own_floors_accept_is_still_acted_on_from_above():
    # (``own`` and ``own_near`` together) the strict answer comes first: a percept at the fitted pose, ``own_top``
    server = _Server(n_known=1, known_type="milk")
    agent, tool, R = _run({"learn_ask_known": {"own": True, "own_near": {}}}, server)
    assert agent.rec["known_end"] == "own" and agent.rec["acting"] == "own_top" and "own_near" not in agent.rec


# -- 2026-11-19, after W2-LEARN-D: floors read from the pod's own rows, and a yes that is not acted on from above
def test_the_pods_floors_are_the_recorded_ones_and_not_among_the_pinned_defaults():
    assert LS.OWN_NEAR_SEEN == {"inlier": 0.93, "seen": 0.65, "cover": 0.6}
    assert LS.OWN_TOP_MIN == {"inlier": 0.93, "seen": 0.78, "cover": 0.95}
    assert "own_top_min" not in LS.ASK and "own_near" not in LS.ASK


ANY = {"own_near": {"any": [LS.OWN_NEAR, LS.OWN_NEAR_SEEN]}}


@pytest.mark.parametrize("fit, floors", [
    ({"inlier": 0.97, "seen": 0.76, "cover": 0.86}, LS.OWN_NEAR),        # passes the first set: that one is recorded
    ({"inlier": 0.96, "seen": 0.84, "cover": 0.67}, LS.OWN_NEAR_SEEN),   # a known type in another size: the second
    ({"inlier": 1.0, "seen": 0.52, "cover": 0.98}, LS.OWN_NEAR)])
def test_a_fit_that_passes_any_one_set_of_floors_goes_to_the_robots_own_plan(fit, floors):
    agent, tool, R = _run({"learn_ask_known": ANY}, _FitServer(fit))
    rec = agent.rec
    assert rec["known_end"] == "own_near" and rec["acting"] == "own_near" and rec["own_near"]["floors"] == floors
    assert [p_["id"] for p_ in tool.percepts] == ["p1"] and R.calls == [(["p1"], 280)]


@pytest.mark.parametrize("fit", [{"inlier": 1.0, "seen": 0.95, "cover": 0.5},     # a zoomed picture (S16): not near
                                 {"inlier": 1.0, "seen": 0.81, "cover": 0.5},
                                 {"inlier": 0.96, "seen": 0.62, "cover": 0.5},    # the pod's nearest new objects
                                 {"inlier": 0.82, "seen": 0.62, "cover": 0.4},
                                 {"inlier": 0.77, "seen": 0.63, "cover": 0.99},
                                 {"inlier": 0.99, "seen": 0.34, "cover": 0.39}])
def test_a_fit_under_every_set_of_floors_is_learned_as_before(fit):
    server = _FitServer(fit)
    agent, _tool_, _R_ = _run({"learn_ask_known": ANY}, server)
    assert ("learn", None) in [(m["op"], m.get("stage")) for m in server.msgs]
    assert "own_near" not in agent.rec and agent.rec.get("known_end") is None


def test_a_yes_by_the_servers_floors_is_not_acted_on_from_above_when_too_little_of_the_model_is_seen():
    # the fake server's yes carries inlier 0.97, seen 0.70, cover 0.99: under the floors for acting from above
    server = _Server(n_known=1, known_type="milk")
    agent, tool, R = _run({"learn_ask_known": {"own": True, "own_near": {}, "own_top_min": {}}}, server)
    rec = agent.rec
    assert rec["known_end"] == "own_near" and rec["acting"] == "own_near" and "known" not in rec and "learn" not in rec
    assert rec["own_near"] == {"inlier": 0.97, "seen": 0.7, "cover": 0.99, "strict": True, "top_min": LS.OWN_TOP_MIN}
    # nothing of the look is kept: no percept at the fitted pose, the robot's own plan on the first look's percepts
    assert [p_["id"] for p_ in tool.percepts] == ["p1"] and R.calls == [(["p1"], 280)]
    assert [(m["op"], m.get("stage")) for m in server.msgs] == [("learn_plan", "top"), ("learn_plan", "flanks")]


def test_a_yes_whose_fit_passes_the_top_floors_is_acted_on_from_above_as_before():
    server = _Server(n_known=1, known_type="milk")
    agent, tool, R = _run({"learn_ask_known": {"own": True, "own_top_min": {"seen": 0.6}}}, server)
    assert agent.rec["known_end"] == "own" and agent.rec["acting"] == "own_top" and "own_near" not in agent.rec
    assert R.calls == [(["p1", "known0"], 280)]


def test_the_top_floors_touch_no_other_type_and_nothing_without_the_key():
    server = _Server(n_known=1, known_type="basket")  # another type explains the thing: kept, the next thing taken
    agent, _tool_, _R_ = _run({"learn_ask_known": {"own": True, "own_top_min": {}}}, server)
    assert [k["type"] for k in agent.rec["known"]] == ["basket"] and agent.rec.get("known_end") != "own_near"
    plain, _t, _r = _run({"learn_ask_known": {"own": True}}, _Server(n_known=1, known_type="milk"))
    assert plain.rec["acting"] == "own_top"


class _Ended:
    finished, finish_reason = True, "no_plan"

    @staticmethod
    def act(obs):
        return [0.0] * 6 + [-1.0]


def test_the_camera_checks_cue_shows_the_robots_own_ending_only_with_the_key():
    for params, want in (({}, None), ({"learn_calib_cue": True}, "no_plan")):
        a = LS.LearnThenAct.__new__(LS.LearnThenAct)
        a.inner, a.params, a.finish_reason = _Ended(), params, None
        assert a.act(OBS) == [0.0] * 6 + [-1.0] and a.finish_reason == want
    # the runner never asks an ended agent for a command: reading ``finished`` is enough, and is what it does
    for params, want in (({}, None), ({"learn_calib_cue": True}, "no_plan")):
        a = LS.LearnThenAct.__new__(LS.LearnThenAct)
        a.inner, a.params, a.finish_reason = _Ended(), params, None
        assert a.finished is True and a.finish_reason == want
    a = LS.LearnThenAct.__new__(LS.LearnThenAct)  # (an agent made without its parameters, as some tests make it)
    a.inner, a.finish_reason = _Ended(), None
    assert a.finished is True and a.finish_reason is None
    # an agent still at work shows nothing
    a = LS.LearnThenAct.__new__(LS.LearnThenAct)
    a.inner, a.params, a.finish_reason = types.SimpleNamespace(finished=False, act=lambda obs: [0.0] * 7), \
        {"learn_calib_cue": True}, None
    a.act(OBS)
    assert a.finish_reason is None
    # what the check reads (``arm_calib.can_move``): the agent's own ``finish_reason``
    from rowant_lab.pro2 import arm_calib as AC

    src = Path(AC.__file__).read_text()
    assert src.count('getattr(agent, "finish_reason", None) in tuple(self.o["after"])') == 1
    # candidate v9 and the registered learner's arm name neither key
    from rowant_lab.pro2 import long_dev as LD

    b = json.dumps(LD.bundle(LD.colour_long_params(MC.REGISTERED["net"])))
    assert "learn_calib_cue" not in b and "own_near" not in b
