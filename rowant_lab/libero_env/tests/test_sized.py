"""2026-11-17: a stored model at another size (robot/sized.py) and the remembered fixture's ``size`` key
(look._from_memory). Off unless the memory entry carries the key."""

import copy
import json

import numpy as np
import pytest

from rowant_lab.libero_env.tests.test_drawer_open import BLOB_S03, BLOB_S07, MEM, _goal, _mem_tool
from rowant_lab.pro2.robot import drawer as DR
from rowant_lab.pro2.robot import sized as SZ

# the first look's region over the cabinet under the case that draws it smaller (a training episode's row: the opening
# task, start state 0, seed 161, colour line)
BLOB_S09 = {"xy": [-0.048, -0.297], "z_top": 1.111, "top": {"xy": [0.062, -0.255], "half_xy": [0.085, 0.079], "z": 1.0866}}
SIZED = {"wooden_cabinet": {**MEM["wooden_cabinet"], "size": {}}}


def _grip(lib, percept, part="cabinet_middle"):
    return DR.handle_plan(lib, percept([0.03, -0.234, 0.9076], 180.0), part)


def test_a_cabinet_drawn_taller_has_its_handles_higher_and_no_nearer():
    lib, percept = _goal()
    stored = copy.deepcopy((lib.types["wooden_cabinet"], lib.export_types["wooden_cabinet"]))
    h0 = _grip(lib, percept)
    assert SZ.sized(lib, "wooden_cabinet") is None
    assert SZ.resize_type(lib, "wooden_cabinet", (1.0, 1.0, 1.23)) == [1.0, 1.0, 1.23]
    # (the percept is built again from the scaled kinematics: ``_goal``'s helper reads the type at its call)
    lib2, percept2 = lib, _percept_of(lib)
    h1 = DR.handle_plan(lib2, percept2([0.03, -0.234, 0.9076], 180.0), "cabinet_middle")
    assert abs(float(h1["c"][2] - h0["c"][2]) - 0.23 * (float(h0["c"][2]) - 0.9076)) < 1e-3  # 2.5 cm up
    assert np.allclose(h1["c"][:2], h0["c"][:2], atol=1e-4) and abs(h1["bar_h"] - 1.23 * h0["bar_h"]) < 1e-4
    assert abs(h1["bar_w"] - h0["bar_w"]) < 1e-5 and abs(h1["pull"] - h0["pull"]) < 1e-6
    assert SZ.sized(lib, "wooden_cabinet") == [1.0, 1.0, 1.23]
    # the stored documents were not touched, and a second resize starts from them (0.8 of the stored, not of 1.23)
    assert json.dumps(stored[0], sort_keys=True) == json.dumps(lib._stored_docs["wooden_cabinet"][0], sort_keys=True)
    SZ.resize_type(lib, "wooden_cabinet", 0.8)
    h2 = DR.handle_plan(lib, _percept_of(lib)([0.03, -0.234, 0.9076], 180.0), "cabinet_middle")
    assert np.allclose(h2["c"] - np.array([0.03, -0.234, 0.9076]), 0.8 * (h0["c"] - np.array([0.03, -0.234, 0.9076])), atol=1e-3)
    assert abs(h2["bar_h"] - 0.8 * h0["bar_h"]) < 1e-4 and abs(h2["bar_w"] - 0.8 * h0["bar_w"]) < 1e-4
    assert abs(h2["pull"] - h0["pull"]) < 1e-6  # a smaller drawer's pull is planned whole: the pull stops with the part
    # drawn wider and deeper, the part travels further and its handle stands further out
    SZ.resize_type(lib, "wooden_cabinet", (1.29, 1.29, 1.0))
    h3 = DR.handle_plan(lib, _percept_of(lib)([0.03, -0.234, 0.9076], 180.0), "cabinet_middle")
    assert abs(h3["pull"] - (1.29 * 0.16 + 0.015)) < 1e-3 and h3["c"][1] > h0["c"][1] + 0.025 and abs(h3["c"][2] - h0["c"][2]) < 1e-4
    # back at the stored size the plan is the stored one, and the boxes the agent keeps clear of follow each resize
    SZ.resize_type(lib, "wooden_cabinet", 1.0)
    h4 = DR.handle_plan(lib, _percept_of(lib)([0.03, -0.234, 0.9076], 180.0), "cabinet_middle")
    assert np.allclose(h4["c"], h0["c"], atol=1e-6) and abs(h4["bar_h"] - h0["bar_h"]) < 1e-6
    top0 = float(np.asarray(lib.body_boxes("wooden_cabinet", "base").aabb()[1], float).max(0)[2])
    SZ.resize_type(lib, "wooden_cabinet", (1.0, 1.0, 1.5))
    top1 = float(np.asarray(lib.body_boxes("wooden_cabinet", "base").aabb()[1], float).max(0)[2])
    assert abs(top1 - 1.5 * top0) < 1e-3


def _percept_of(lib):
    import math

    from rowant_lab.libero_env.m2_kin import mat_to_quat, part_body_poses

    def percept(pos, yaw_deg, pid="p6"):
        tdoc = lib.export_types["wooden_cabinet"]
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        bp = part_body_poses(tdoc, {tdoc["root_body"]: (R, np.asarray(pos, float))}, {})
        return {"id": pid, "type": "wooden_cabinet", "gid": None, "joints": {}, "fit": {"n_agree": 10, "inlier": 1.0},
                "source": "support",
                "bodies": {b: {"pos": [float(x) for x in t], "quat_wxyz": mat_to_quat(RR)} for b, (RR, t) in bp.items()}}

    return percept


def test_an_objects_sites_and_grasp_records_follow_its_size():
    lib, _percept = _goal()
    w0 = sorted(float(r["width"]) for r in _records(lib, "wine_bottle"))
    site0 = copy.deepcopy(lib.export_types["wooden_cabinet"]["sites"]["top_side"])
    SZ.resize_type(lib, "wine_bottle", 0.7)
    w1 = sorted(float(r["width"]) for r in _records(lib, "wine_bottle"))
    assert w1 and abs(max(w1) - 0.7 * max(w0)) < 0.004  # the widest hold is the body's, seven tenths as wide
    SZ.resize_type(lib, "wooden_cabinet", (1.0, 1.0, 0.8))
    site1 = lib.export_types["wooden_cabinet"]["sites"]["top_side"]
    assert abs(site1["pos"][2] - 0.8 * site0["pos"][2]) < 1e-6 and np.allclose(site1["pos"][:2], site0["pos"][:2])
    assert np.allclose(site1["size"][:2], site0["size"][:2], atol=1e-6)
    with pytest.raises(ValueError):
        SZ.resize_type(lib, "plate", (1.0, -1.0, 1.0))
    stored = lib.records["cream_cheese"]
    with pytest.raises(ValueError):  # a flat box drawn flatter leaves the hand nothing to take: refused, as stored
        SZ.resize_type(lib, "cream_cheese", (1.0, 1.0, 0.72))
    assert lib.records["cream_cheese"] is stored and SZ.sized(lib, "cream_cheese") is None
    with pytest.raises(ValueError):
        SZ.scaled_docs(lib.types["plate"], {**lib.export_types["plate"], "bodies": {"main": {"body_quat": [0.7071, 0.7071, 0, 0]}}},
                       (1.0, 1.2, 1.0))  # a body turned against its root takes one factor only


def _records(lib, typ):
    rec = lib.records[typ]
    rows = rec if isinstance(rec, list) else getattr(rec, "records", None) or getattr(rec, "rows", None)
    if rows is None:  # the planner's own container: its widths by attribute or by key
        w = getattr(rec, "width", None)
        if w is None and isinstance(rec, dict):
            w = rec.get("width")
        return [{"width": float(x)} for x in np.asarray(w, float).ravel()]
    return [{"width": float(r["width"] if isinstance(r, dict) else r.width)} for r in rows]


def test_the_remembered_fixture_takes_the_size_its_region_shows_when_asked():
    # the case that draws the cabinet taller: with the key it is the cabinet a fifth taller at its remembered place
    lib, _p = _goal()
    p = _mem_tool(lib, [BLOB_S07], memory=SIZED)._from_memory("wooden_cabinet")
    assert p["source"] == "memory+sized" and np.allclose(p["bodies"]["main"]["pos"], MEM["wooden_cabinet"]["pos"])
    s = p["memory"]["size"]["s"]
    assert s[0] == s[1] == 1.0 and abs(s[2] - 1.219) < 0.005 and SZ.sized(lib, "wooden_cabinet") == s
    hp = DR.handle_plan(lib, p, "cabinet_middle")
    assert abs(float(hp["c"][2]) - (0.9076 + 1.219 * 0.1104)) < 0.002 and abs(float(hp["c"][1]) + 0.1324) < 1e-3
    assert abs(float(hp["bar_h"]) - 1.219 * 0.0163) < 2e-4
    # asked again in the same episode (a later step's plan): the cabinet stays as sized, at the same place
    again = _mem_tool(lib, [BLOB_S07], memory=SIZED)._from_memory("wooden_cabinet")
    assert again["source"] == "memory" and np.allclose(again["bodies"]["main"]["pos"], MEM["wooden_cabinet"]["pos"])
    assert SZ.sized(lib, "wooden_cabinet") == s
    # the case that draws it smaller: its top 4.4 cm low, part of it out of view: four fifths in every direction
    lib, _p = _goal()
    q = _mem_tool(lib, [BLOB_S09], memory=SIZED)._from_memory("wooden_cabinet")
    s = q["memory"]["size"]["s"]
    assert q["source"] == "memory+sized" and abs(s[2] - 0.803) < 0.005 and s[0] == s[1] == s[2]
    hp = DR.handle_plan(lib, q, "cabinet_middle")
    assert abs(float(hp["c"][2]) - (0.9076 + 0.803 * 0.1104)) < 0.002 and abs(float(hp["c"][1]) - (-0.234 + 0.803 * 0.1016)) < 0.002
    # on its block the cabinet is lifted as before; at the model's own height it is the model
    lib, _p = _goal()
    assert _mem_tool(lib, [BLOB_S03], memory=SIZED)._from_memory("wooden_cabinet")["source"] == "memory+lifted"
    low = {**BLOB_S03, "top": {**BLOB_S03["top"], "z": 1.12}}
    assert _mem_tool(lib, [low], memory=SIZED)._from_memory("wooden_cabinet")["source"] == "memory"
    assert SZ.sized(lib, "wooden_cabinet") is None
    # a factor outside the entry's bounds is not this fixture resized: nothing is changed (the look's own lower
    # limit for a region to count, 8 cm under the model's top, is 0.64 of this cabinet)
    lib, _p = _goal()
    flat = {**BLOB_S09, "top": {**BLOB_S09["top"], "z": 1.055}}
    tight = {"wooden_cabinet": {**MEM["wooden_cabinet"], "size": {"lo": 0.7}}}
    r = _mem_tool(lib, [flat], memory=tight)._from_memory("wooden_cabinet")
    assert r is not None and r["source"] == "memory" and SZ.sized(lib, "wooden_cabinet") is None
    assert _mem_tool(lib, [flat], memory=SIZED)._from_memory("wooden_cabinet")["memory"]["size"]["s"][2] < 0.67
    # without the key: as before (taller reads as lifted, smaller as the model)
    lib, _p = _goal()
    assert _mem_tool(lib, [BLOB_S07])._from_memory("wooden_cabinet")["source"] == "memory+lifted"
    assert _mem_tool(lib, [BLOB_S09])._from_memory("wooden_cabinet")["source"] == "memory"
    assert SZ.sized(lib, "wooden_cabinet") is None


def test_the_fallback_record_says_what_size_was_taken():
    lib, _p = _goal()
    t = _mem_tool(lib, [BLOB_S07], memory=SIZED)
    t.on_parsed({"target_type": "wooden_cabinet", "place_type": None})
    assert [(f["role"], f["source"]) for f in t.fallbacks] == [("target", "memory+sized")]
    assert t.fallbacks[0]["size"]["s"][2] > 1.2 and t.percepts[0]["type"] == "wooden_cabinet"
    off = _mem_tool(lib2 := _goal()[0], [BLOB_S07])
    off.on_parsed({"target_type": "wooden_cabinet", "place_type": None})
    assert "size" not in off.fallbacks[0] and off.fallbacks[0]["source"] == "memory+lifted" and SZ.sized(lib2, "wooden_cabinet") is None


# ------------------------------------------------------------------ the part's front found by touch (open_feel)
from rowant_lab.libero_env.tests.test_drawer_open import Tool, World, _hp, _run  # noqa: E402


class Fronted(World):
    """``World`` with the part's front ``stand`` behind its bar: the grip point cannot pass the front less the
    fingertip's length (0.93 cm) along the way in, wherever the plan has the handle."""

    def __init__(self, handle, d, stand=0.024, tip=0.0093, **kw):
        super().__init__(handle, d, **kw)
        self.stand, self.tip = float(stand), float(tip)

    def step(self, a):
        super().step(a)
        lim = float(self.handle @ self.d) - self.stand + self.tip
        s = float(self.p @ self.d)
        if s < lim and abs(float(self.p[2] - self.handle[2])) < 0.06:
            self.p = self.p + self.d * (lim - s)


def _run_lag(world, tool, hp, lag, cap=200, **opts):
    """``_run`` of the drawer tests with the agent told of a delay of ``lag`` steps (the stand-in itself has none)."""
    from rowant_lab.pro2.robot import agent as AG

    ag = object.__new__(DR.OpenAgent)
    ag.lag_steps = lag
    ag.lib, ag.oracle, ag.cap, ag.t = None, tool, cap, 0
    ag.ap, ag.fp, ag.jaw = AG.AgentParams(open_fast=True, pull_stall_stop=True, **opts), AG.FollowParams(), type("J", (), {"L": 0.04, "gap_from_qpos": staticmethod(lambda a, b: float(a) - float(b))})()
    ag.spec, ag.target = {"part": "cabinet_middle", "target": "p6"}, "p6"
    ag.attempts, ag.events, ag.finished, ag.finish_reason = [], [], False, None
    ag.hp, ag.pobs = hp, world.obs()
    ag._hand_R = lambda R_, d_: np.eye(3)
    gen = ag._episode()
    next(gen)
    while not ag.finished and ag.t < cap:
        ag.pobs = world.obs()
        try:
            a = gen.send(ag.pobs)
        except StopIteration:
            break
        if a is None:
            break
        world.step(a)
        ag.t += 1
    return ag


def test_after_a_miss_the_parts_front_is_found_by_touch_and_the_bar_taken_before_it():
    d = [0.0, 1.0, 0.0]
    true = np.array([0.04, -0.152, 1.0])  # the bar stands 2 cm deeper than the plan has it (a cabinet drawn smaller)
    plan = _hp([0.04, -0.132, 1.0], d)
    off = _run(Fronted(true, d), Tool([]), plan)
    assert [a["outcome"] for a in off.attempts] == ["missed"] * 3 and all("feel" not in a for a in off.attempts)
    on = _run(Fronted(true, d), Tool([]), plan, open_feel=True)
    assert [a.get("outcome") for a in on.attempts] == ["missed", "opened"] and on.finish_reason == "opened"
    fl = on.attempts[1]["feel"]
    assert "feel" not in on.attempts[0] and fl["stalled"] and abs(fl["in_m"] - (0.02 + 0.024 - 0.0093)) < 0.003
    assert abs(fl["back_m"] - (0.024 - 0.0093)) < 1e-4 and abs(fl["grip"][1] - true[1]) < 0.003
    # a bar 3 cm NEARER than planned: the first approach stops on the front early (the fingers close behind the
    # bar), the second starts from there, does not come further, and backs off onto the bar
    near = np.array([0.04, -0.102, 1.0])
    on2 = _run(Fronted(near, d), Tool([]), plan, open_feel=True)
    assert [a.get("outcome") for a in on2.attempts] == ["missed", "opened"]
    assert on2.attempts[1]["feel"]["in_m"] < -0.01 and abs(on2.attempts[1]["feel"]["grip"][1] - near[1]) < 0.003
    # nothing there at all (no front within reach): the hand goes its 6 cm, comes back, and the miss is a miss
    gone = _run(World([0.3, 0.3, 1.0], d), Tool([]), plan, open_feel=True)
    assert [a["outcome"] for a in gone.attempts] == ["missed"] * 3
    assert all(not a["feel"]["stalled"] and abs(a["feel"]["grip"][1] + 0.132) < 0.004 for a in gone.attempts[1:])
    # through a delay the touch is not used (the readings are late)
    lag = _run_lag(Fronted(true, d), Tool([]), plan, 3, open_feel=True, open_feel_first=True)
    assert lag.attempts and all("feel" not in a for a in lag.attempts)


def test_a_fixture_no_look_measured_is_felt_for_at_the_first_attempt_when_asked():
    lib, _p = _goal()
    d = [0.0, 1.0, 0.0]
    p = _mem_tool(lib, [BLOB_S07], memory=SIZED)._from_memory("wooden_cabinet")
    p["id"] = "p6"
    hp = DR.handle_plan(lib, p, "cabinet_middle")
    assert abs(hp["stand_m"] - 0.0237) < 0.002  # the drawer's front stands 2.4 cm behind the grip point (its posts do not count)
    # the room above and below the bar is to the next drawer's handle (7.4 cm apart at the stored size, a fifth more
    # on the cabinet drawn taller), and the touch's jaws are shaped by it: wider than usual here, no wider than the
    # fingers' own thickness lets them pass on a cabinet drawn smaller
    assert abs(hp["room_m"] - 1.2216 * 0.0611) < 0.003
    lib_s, percept_s = _goal()
    SZ.resize_type(lib_s, "wooden_cabinet", 0.75)
    small = DR.handle_plan(lib_s, _percept_of(lib_s)([0.03, -0.234, 0.9076], 180.0), "cabinet_middle")
    ag = object.__new__(DR.OpenAgent)
    ag.jaw = type("J", (), {"L": 0.0387, "finger_thick": 0.0269})()
    assert ag._feel_k(hp) == 1 and ag._feel_k(small) == 5 and abs(small["room_m"] - 0.0458) < 0.002
    true = np.asarray(hp["c"], float) + np.array([0.0, 0.022, 0.0])  # drawn deeper too: the bar 2.2 cm nearer
    tool = Tool([p])
    tool.lib = lib
    first = _run(Fronted(true, d, stand=1.22 * 0.0237), tool, {**hp, "pull": 0.12}, open_feel=True, open_feel_first=True)
    assert [a.get("outcome") for a in first.attempts] == ["opened"] and first.attempts[0]["feel"]["stalled"]
    assert abs(first.attempts[0]["feel"]["grip"][1] - true[1]) < 0.008  # within the pads' reach of the bar's middle
    # the touch also says how deep the fixture is: a cabinet drawn three quarters as deep has its front felt 2 cm
    # behind the model's and its bar 1.8 cm before that front, not the model's 2.4: the grip lands on the bar
    orig = DR.OpenAgent._feel

    def typed(self, *a, **k):
        self.ttype = "wooden_cabinet"
        return orig(self, *a, **k)

    DR.OpenAgent._feel = typed
    try:
        root_s, k = -0.234, 0.74
        bar = np.array([float(hp["c"][0]), root_s + k * (float(hp["c"][1]) - root_s), float(hp["c"][2])])
        tool2 = Tool([p])
        tool2.lib = lib
        deep = _run(Fronted(bar, d, stand=k * float(hp["stand_m"])), tool2, {**hp, "pull": 0.12}, open_feel=True,
                    open_feel_first=True)
    finally:
        DR.OpenAgent._feel = orig
    fl = deep.attempts[0]["feel"]
    assert deep.attempts[0].get("outcome") == "opened" and abs(fl["depth_ratio"] - k) < 0.04
    assert abs(fl["grip"][1] - bar[1]) < 0.004 and fl["back_m"] < 0.011
    # a fixture the look measured is taken as planned, the touch only after a miss
    seen = Tool([{**p, "source": "support"}])
    seen.lib = lib
    ag = _run(Fronted(true, d, stand=1.22 * 0.0237), seen, {**hp, "pull": 0.12}, open_feel=True, open_feel_first=True)
    assert "feel" not in ag.attempts[0] and ag.attempts[0]["outcome"] == "missed"


# ------------------------------------------------------------------ a reported object's own leftover (fallback_own)
def test_a_reported_objects_own_leftover_is_not_taken_for_the_receiver():
    from rowant_lab.libero_env.tests.test_place_evidence import _blob, _p, _tool

    def run(own, blobs, kept):
        t = _tool(list(kept), [], blobs)
        t.fallback_top, t.fallback_own = True, own
        t.on_parsed({"target_type": "bowl", "place_type": "basket"})
        got = [p for p in t.percepts if p["type"] == "basket"]
        return (got[0]["source"], [round(float(x), 3) for x in got[0]["bodies"]["main"]["pos"][:2]]) if got else None

    bowl = _p("p0", "bowl", (-0.11, -0.24, 0.9))  # the target as the look fitted it: its top at 0.95
    # what the fit left over of the target itself (read 4 cm taller than its model): a region drawn out along the
    # camera's rays whose top layer lies on the target; across and in height it fits the basket as a whole
    left = _blob((-0.185, -0.256), (0.086, 0.094), 0.995, 0.095, top=((-0.125, -0.236), (0.041, 0.023), 0.99), n_px=700)
    # the basket drawn smaller: its region runs on along the rays (too long to be the model), its top layer is the rim
    basket = _blob((-0.07, 0.28), (0.17, 0.094), 1.012, 0.112, top=((0.02, 0.251), (0.08, 0.066), 1.012), n_px=29000)
    assert run(False, [basket, left], [bowl]) == ("blob_whole", [-0.185, -0.256])  # as before: the basket on the target
    assert run(True, [basket, left], [bowl]) == ("blob_top", [0.02, 0.251])
    # a region on a reported object but well above its top (something standing on it) is not its leftover
    high = _blob((-0.185, -0.256), (0.086, 0.094), 1.07, 0.17, top=((-0.125, -0.236), (0.09, 0.07), 1.065), n_px=9000)
    assert run(True, [high], [bowl])[0] in ("blob_whole", "blob_top")
    # with nothing reported there the same region is a candidate as before
    t = _tool([], [], [left])
    t.fallback_top, t.fallback_own = True, True
    t.on_parsed({"target_type": None, "place_type": "basket"})
    assert [p["source"] for p in t.percepts if p["type"] == "basket"] == ["blob_whole"]


# ------------------------------------------------------------------ a named target at another height (fallback_stand's resize)
def _goal_tool(blobs, stand, percepts=()):
    from rowant_lab.libero_env.tests.test_close_look import _cam
    from rowant_lab.pro2.robot import look as LK

    lib, _p = _goal()
    t = LK.PerceptionTool(lib, None, _cam, "libero_goal", "ep")
    t.percepts, t.extra = list(percepts), {"rejected": [], "blobs": list(blobs)}
    t.table_bound = t.blob_on_table = True
    t.stand = {**LK.STAND, **stand}
    return t


# the first look's region where the bottle stood under the case that swaps it for a shorter one (a training episode's
# row, Mac, depth line: seven tenths of the model's height, its cap 1.8 cm across)
SHORT = {"xy": [-0.188, -0.038], "half_xy": [0.027, 0.019], "yaw": 0.0, "z_top": 1.0137, "height": 0.113, "n_px": 3818,
         "hang_share": 0.0, "top": {"xy": [-0.195, -0.039], "half_xy": [0.009, 0.009], "yaw": 0.0, "z": 1.0133, "n_px": 560}}


def test_a_named_target_standing_lower_than_its_model_is_stood_with_the_model_drawn_to_that_height():
    t = _goal_tool([SHORT], {"resize": {}})
    h0 = float(np.diff(np.asarray(t._model_box("wine_bottle"), float)[:, 2])[0])
    t.on_parsed({"target_type": "wine_bottle", "place_type": None})
    fb = [f for f in t.fallbacks if f.get("role") == "target"]
    assert [f["source"] for f in fb] == ["blob_stand"] and fb[0]["stand"]["size"][:2] == [1.0, 1.0]
    s_z = fb[0]["stand"]["size"][2]
    assert 0.68 < s_z < 0.76 and SZ.sized(t.lib, "wine_bottle") == [1.0, 1.0, s_z]
    lo, hi = t._box_of(t.percepts[-1])
    assert abs(float(hi[2] - lo[2]) - s_z * h0) < 1e-3 and abs(float(hi[2]) - 1.0133) < 0.012  # its top at the region's
    assert float(np.linalg.norm(0.5 * (lo[:2] + hi[:2]) - np.array([-0.195, -0.039]))) < 0.03  # under the top layer
    assert abs(float(hi[0] - lo[0]) - 0.043) < 0.004  # as wide as stored: the fingers find its width
    # the grasp records are the drawn model's: none above its top
    rec = t.lib.records["wine_bottle"]
    z = np.asarray(rec["z_rel"] if "z_rel" in rec else rec["z"], float) if isinstance(rec, dict) else None
    assert z is None or float(z.max()) <= s_z * h0 + 0.01
    # without the key: no model is stood at that height and the stored one is untouched (the older fallbacks follow)
    off = _goal_tool([SHORT], {})
    off.on_parsed({"target_type": "wine_bottle", "place_type": None})
    assert [f["source"] for f in off.fallbacks if f.get("role") == "target"] == ["blob"] and SZ.sized(off.lib, "wine_bottle") is None
    # a region the drawn model does not fit either (a slab four times the cap across): the stored model is put back
    slab = {**SHORT, "top": {**SHORT["top"], "half_xy": [0.08, 0.06]}}
    no = _goal_tool([slab], {"resize": {}})
    stored = no.lib.types["wine_bottle"]
    no.on_parsed({"target_type": "wine_bottle", "place_type": None})
    assert SZ.sized(no.lib, "wine_bottle") is None and no.lib.types["wine_bottle"] is stored
    assert all(f.get("source") != "blob_stand" for f in no.fallbacks)


# ------------------------------------------------------------------ the named target read as another type (misread_target)
def _goal_scene(option, blobs, second_plate=True, bowl=False):
    """A Goal-suite look as a training episode's row had it (Mac, depth line, the case that swaps the bowl): the real
    plate, a second "plate" the look agreed with half of, and the sliver of the bowl's rim that stood out above it."""
    from rowant_lab.pro2.robot import look as LK

    t = _goal_tool(blobs, {})
    t.stand = None
    t.misread = {**LK.MISREAD, **option} if option is not None else None

    def p(pid, typ, pos, n_vis, n_agree):
        return {"id": pid, "type": typ, "gid": "g", "joints": {}, "source": "support",
                "fit": {"n_vis": n_vis, "n_agree": n_agree, "inlier": 0.98}, "bodies": t._build(typ, np.eye(3), np.asarray(pos, float))}

    t.percepts = [p("p4", "plate", (0.051, -0.029, 0.903), 4254, 4152)]
    if second_plate:
        t.percepts.append(p("p5", "plate", (-0.107, -0.002, 0.901), 4266, 2230))
    if bowl:
        t.percepts.append(p("p0", "akita_black_bowl", (0.2, 0.2, 0.899), 4600, 3900))
    return t


RIM = {"xy": [-0.14, 0.031], "half_xy": [0.054, 0.015], "yaw": 0.0, "z_top": 0.9365, "height": 0.0361, "n_px": 985,
       "hang_share": 0.0, "top": {"xy": [-0.142, 0.031], "half_xy": [0.051, 0.012], "yaw": 0.0, "z": 0.9364, "n_px": 985}}
ON_STOVE = {"target_type": "akita_black_bowl", "place_type": "flat_stove", "landmarks": []}
ON_PLATE = {"target_type": "akita_black_bowl", "place_type": "plate", "landmarks": []}


def test_a_named_target_read_as_another_type_is_stood_where_that_percept_stands():
    t = _goal_scene({}, [RIM])
    t.on_parsed(ON_STOVE)
    fb = t.fallbacks
    assert [(f["role"], f.get("source")) for f in fb] == [("dropped", None), ("target", "misread")] and fb[0]["ids"] == ["p5"]
    note = fb[1]["misread"]
    assert note["as"] == "p5" and abs(note["share"] - 0.523) < 0.002 and 1.2 < note["across_ratio"] < 1.4
    assert 0.6 < note["height_ratio"] < 0.9 and note["size"][0] == note["size"][1]
    assert abs(note["size"][0] - note["across_ratio"]) < 1e-3 and abs(note["size"][2] - note["height_ratio"]) < 1e-3
    assert [q["id"] for q in t.percepts] == ["p4", "f1"] and SZ.sized(t.lib, "akita_black_bowl") == note["size"]
    lo, hi = t._box_of(t.percepts[-1])
    assert float(np.linalg.norm(0.5 * (lo[:2] + hi[:2]) - np.array([-0.107, -0.002]))) < 0.004  # over the false plate
    assert abs(float(hi[2]) - 0.9364) < 0.004 and abs(float(hi[0] - lo[0]) - 0.138) < 0.006  # its rim where it was seen
    # the instruction names the plate too: the weaker of two plates is the bowl, the only plate there is never
    t = _goal_scene({}, [RIM])
    t.on_parsed(ON_PLATE)
    assert [q["id"] for q in t.percepts] == ["p4", "f1"] and t.fallbacks[-1]["source"] == "misread"
    one = _goal_scene({}, [RIM], second_plate=False)
    one.percepts[0] = {**one.percepts[0], "fit": {"n_vis": 4254, "n_agree": 2200, "inlier": 0.98}}
    one.on_parsed(ON_PLATE)
    assert all(f.get("source") != "misread" for f in one.fallbacks) and SZ.sized(one.lib, "akita_black_bowl") is None
    # nothing stands out above the weak plate, or the plate is one the look agrees with: no bowl there
    for scene in (_goal_scene({}, []), _goal_scene({"weak": 0.5}, [RIM])):
        scene.on_parsed(ON_STOVE)
        assert all(f.get("source") != "misread" for f in scene.fallbacks) and "p5" in [q["id"] for q in scene.percepts]
    # without the option: as before (the sliver itself is taken for the bowl)
    off = _goal_scene(None, [RIM])
    off.on_parsed(ON_STOVE)
    assert [f.get("source") for f in off.fallbacks if f["role"] == "target"] == ["blob"] and "p5" in [q["id"] for q in off.percepts]
    # the bowl as the RECEIVER (the box goes into it): stood the same way, and the false plate leaves the scene
    rcv = _goal_scene({}, [RIM])
    rcv.on_parsed({"target_type": "cream_cheese", "place_type": "akita_black_bowl", "landmarks": []})
    place = [f for f in rcv.fallbacks if f.get("role") == "place"]
    assert [f["source"] for f in place] == ["misread"] and "p5" not in [q["id"] for q in rcv.percepts]
    # another bowl is reported (two bowls in the scene): the model stays as stored, the place is still offered
    two = _goal_scene({}, [RIM], bowl=True)
    got = two._misread("akita_black_bowl", ("plate",))
    assert [pid for pid, _q in got] == ["p5"] and "size" not in got[0][1]["misread"] and SZ.sized(two.lib, "akita_black_bowl") is None


def test_after_a_lost_bar_the_front_is_looked_for_back_along_the_pull_and_the_next_pull_is_gentle():
    from rowant_lab.libero_env.tests.test_drawer_open import Slippy

    class SlipFront(Slippy):
        """``Slippy`` with the part's front behind its bar (as ``Fronted``): the bar is lost once, 5 cm out."""

        def step(self, a):
            super().step(a)
            lim = float(self.handle @ self.d) - 0.024 + 0.0093
            s = float(self.p @ self.d)
            if s < lim and abs(float(self.p[2] - self.handle[2])) < 0.06:
                self.p = self.p + self.d * (lim - s)

    d = np.array([0.0, 1.0, 0.0])
    plan = _hp([0.04, -0.132, 1.0], d)
    # the check alone (as built): "slipped", then the handle is reached for where the hand's travel puts it: nothing there
    w = SlipFront(plan["c"], d)
    chk = _run(w, Tool([]), dict(plan), pull_end_early=True, open_slip_check=True)
    assert [a["outcome"] for a in chk.attempts][:2] == ["slipped", "missed"]
    # with the touch: from there the hand goes back in until the front stops it (the drawer stands 5 cm out), takes
    # the bar before it and pulls gently
    w = SlipFront(plan["c"], d)
    ag = _run(w, Tool([]), dict(plan), pull_end_early=True, open_slip_check=True, open_feel=True)
    a0, a1 = ag.attempts[0], ag.attempts[1]
    assert a0["outcome"] == "slipped" and "feel" not in a0 and "gentle" not in a0
    assert a1["feel"]["stalled"] and a1["feel"]["in_m"] > 0.04 and a1.get("gentle") is True
    assert abs(float((np.asarray(a1["feel"]["grip"]) - plan["c"]) @ d) - 0.05) < 0.004  # on the bar where it stood
    assert a1["outcome"] == "opened" and a1["gap_end"] > 0.012 and ag.finish_reason == "opened"
