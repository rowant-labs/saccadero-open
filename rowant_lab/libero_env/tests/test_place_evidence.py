"""2026-11-16, three options, all off by default (stream PLACE):

* ``params["relation_single"]``: the instruction's relation asked of a single candidate, and stand-ins for landmarks;
* ``params["fallback_top"]`` (with the server's ``extra_blob_top_m``): a named type with no percept follows the evidence;
* ``params["relook_strict"]``: a receiver's believed move stays level and off other known objects.
"""

import numpy as np

from rowant_lab.libero_env.m2_kin import mat_to_quat
from rowant_lab.libero_env.tests.test_close_look import _cam, _lib
from rowant_lab.pro2.robot import ground as G
from rowant_lab.pro2.robot import look as LK

# type -> (half extents of its one box, the box centre's height above the root)
SHAPES = {"bowl": ((0.05, 0.05, 0.025), 0.025), "plate": ((0.07, 0.07, 0.01), 0.01), "box": ((0.04, 0.03, 0.01), 0.01),
          "basket": ((0.10, 0.08, 0.07), 0.07), "stove": ((0.12, 0.12, 0.01), 0.01)}
TABLE = {"arena": [{"type": "box", "centre": [0.0, 0.0, 0.85], "R": np.eye(3).tolist(), "half": [0.5, 0.6, 0.05]}]}


def _p(pid, typ, pos, n_agree=1000, **kw):
    return {"id": pid, "type": typ, "gid": "g", "joints": {}, "fit": {"n_agree": n_agree, "inlier": 0.9},
            "source": "texture", "bodies": {"main": {"pos": list(pos), "quat_wxyz": mat_to_quat(np.eye(3))}}, **kw}


def _rej(pid, typ, pos, n_agree=800, inlier=0.8, accepted=False, tilt=0.0):
    return {**_p(pid, typ, pos, n_agree), "fit": {"n_agree": n_agree, "inlier": inlier}, "pos": list(pos),
            "source": "support", "why": {"accepted": accepted, "upright_err_deg": tilt}}


def _blob(xy, half, z_top, height, top=None, n_px=5000):
    b = {"xy": list(xy), "half_xy": list(half), "yaw": 0.0, "z_top": z_top, "height": height, "n_px": n_px}
    if top is not None:
        b["top"] = {"xy": list(top[0]), "half_xy": list(top[1]), "yaw": 0.0, "z": top[2], "n_px": 900}
    return b


def _tool(percepts, rejected=(), blobs=()):
    lib = _lib()
    base = lib.export_types["can"]
    lib.export_types = {t: {**base, "kind": "fixture" if t == "stove" else "object"} for t in SHAPES}
    lib.m1_type = lambda typ: {"bodies": {"main": {"primitives": [
        {"centre": [0.0, 0.0, SHAPES[typ][1]], "R": np.eye(3).tolist(), "half": list(SHAPES[typ][0])}]}}}
    lib.workcell = TABLE
    t = LK.PerceptionTool(lib, None, _cam, "libero_spatial", "ep")
    t.percepts = list(percepts)
    t.extra = {"rejected": list(rejected), "blobs": list(blobs)}
    t.table_bound = True
    return t


def _pos(t, pid):
    return [round(float(x), 4) for x in next(p for p in t.percepts if p["id"] == pid)["bodies"]["main"]["pos"]]


NEXT = {"status": "ok", "target_type": "bowl", "relation": "next_to", "landmarks": ["box"], "part": None,
        "predicate": "on", "place_type": "plate"}


# ---- the relation asked of one candidate
def test_a_single_candidate_is_asked_the_relation_with_room_to_spare():
    t = _tool([_p("p0", "bowl", (0.13, 0.0, 0.9)), _p("p1", "box", (0.0, 0.0, 0.9)), _p("p2", "plate", (0.3, 0.3, 0.9)),
               _p("p3", "stove", (-0.3, -0.3, 0.9))])
    lib, ps = t.lib, t.percepts
    assert G.relation_holds(NEXT, ps[0], ps, lib, TABLE) is True  # 13 cm: next to it
    far = _p("q", "bowl", (0.42, 0.0, 0.9))
    assert G.relation_holds(NEXT, far, ps, lib, TABLE) is False  # 42 cm: the other one
    assert G.relation_holds({**NEXT, "relation": None}, far, ps, lib, TABLE) is None
    assert G.relation_holds({**NEXT, "landmarks": ["basket"]}, far, ps, lib, TABLE) is None  # no such landmark seen
    stand_in = [_p("x", "basket", (0.45, 0.0, 0.9))]
    assert G.relation_holds({**NEXT, "landmarks": ["basket"]}, far, ps, lib, TABLE, stand_in) is True
    btw = {**NEXT, "relation": "between", "landmarks": ["box", "plate"]}
    assert G.relation_holds(btw, _p("q", "bowl", (0.15, 0.16, 0.9)), ps, lib, TABLE) is True  # midway, 1 cm off the line
    assert G.relation_holds(btw, _p("q", "bowl", (0.28, 0.40, 0.9)), ps, lib, TABLE) is False  # beyond the plate
    assert G.relation_holds(btw, _p("q", "bowl", (0.25, 0.05, 0.9)), ps, lib, TABLE) is False  # 14 cm off the line
    ctr = {**NEXT, "relation": "table_centre", "landmarks": ["table"]}
    assert G.relation_holds(ctr, _p("q", "bowl", (0.07, 0.03, 0.9)), ps, lib, TABLE) is True
    assert G.relation_holds(ctr, _p("q", "bowl", (0.0, 0.305, 0.9)), ps, lib, TABLE) is False
    on = {**NEXT, "relation": "on", "landmarks": ["stove"]}
    assert G.relation_holds(on, _p("q", "bowl", (-0.28, -0.31, 0.925)), ps, lib, TABLE) is True  # 5 mm above its top
    assert G.relation_holds(on, _p("q", "bowl", (-0.28, -0.31, 1.10)), ps, lib, TABLE) is False  # far above it
    assert G.relation_holds(on, _p("q", "bowl", (0.13, 0.0, 0.9)), ps, lib, TABLE) is False  # not over it


def test_a_landmark_stand_in_lets_the_relation_choose_instead_of_the_stronger_fit():
    t = _tool([_p("p0", "bowl", (0.42, 0.0, 0.9), n_agree=3000), _p("p1", "bowl", (0.13, 0.0, 0.9), n_agree=2000),
               _p("p2", "plate", (0.3, 0.3, 0.9))])
    g = G.ground(NEXT, t.percepts, t.lib, TABLE)
    assert g["target"] == "p0" and "landmark_missing" in g["flags"]  # as before: the stronger fit, a coin toss
    g = G.ground(NEXT, t.percepts, t.lib, TABLE, extra_landmarks=[_p("x", "box", (0.0, 0.0, 0.9))])
    assert g["target"] == "p1" and "landmark_fallback:box" in g["flags"] and "landmark_missing" not in g["flags"]
    assert g["place"] == "p2"


def test_the_named_object_the_look_did_not_report_joins_as_a_second_candidate():
    kept = [_p("p0", "bowl", (0.42, 0.0, 0.9)), _p("p1", "box", (0.0, 0.0, 0.9)), _p("p2", "plate", (0.3, 0.3, 0.9))]
    # a copy (accepted, then dropped as a duplicate), a weak fit and a weak tilted fit next to the box are never taken
    junk = [_rej("p100", "bowl", (0.10, 0.02, 0.9), accepted=True),
            _rej("p101", "bowl", (0.11, 0.0, 0.9), tilt=57.0, inlier=0.4), _rej("p102", "bowl", (0.12, 0.0, 0.9), inlier=0.4)]
    t = _tool(kept, junk)
    assert t.relation_candidates(NEXT) == [] and [p["id"] for p in t.percepts] == ["p0", "p1", "p2"]
    # a rejected pose next to the box: it joins, and the relation chooses it
    t = _tool(kept, junk + [_rej("p103", "bowl", (0.125, 0.01, 0.9))])
    assert t.relation_candidates(NEXT) == []
    assert [p["id"] for p in t.percepts] == ["p0", "p1", "p2", "r1"] and _pos(t, "r1") == [0.125, 0.01, 0.9]
    assert t.fallbacks[-1] == {"role": "target", "type": "bowl", "id": "r1", "source": "rejected:support",
                               "relation": "next_to"}
    assert t.relation_target == "r1"
    t.relation_candidates(NEXT)  # planned again: nothing more joins, and the same candidate is the target
    assert [p["id"] for p in t.percepts] == ["p0", "p1", "p2", "r1"] and t.relation_target == "r1"
    assert G.ground(NEXT, t.percepts, t.lib, TABLE)["target"] == "r1"
    # no pose, but an unexplained region whose top layer has a bowl's size, next to the box
    t = _tool(kept, [], [_blob((0.16, 0.0), (0.10, 0.07), 0.955, 0.055, top=((0.13, 0.0), (0.06, 0.06), 0.955))])
    t.relation_candidates(NEXT)
    assert t.percepts[-1]["source"] == "blob_top" and _pos(t, "r1") == [0.13, 0.0, 0.905]  # its top at the layer
    # the named object was also reported as a second, weaker plate where it stands: that reading leaves the scene
    t = _tool(kept + [_p("p5", "plate", (0.13, 0.015, 0.9), n_agree=400)], [_rej("p103", "bowl", (0.125, 0.01, 0.9))])
    t.relation_candidates(NEXT)
    assert [p["id"] for p in t.percepts] == ["p0", "p1", "p2", "r1"]
    assert {"role": "dropped", "ids": ["p5"], "why": "read twice"} in t.fallbacks
    t = _tool([kept[0], kept[1], _p("p5", "plate", (0.13, 0.015, 0.9))], [_rej("p103", "bowl", (0.125, 0.01, 0.9))])
    t.relation_candidates(NEXT)
    assert [p["id"] for p in t.percepts] == ["p0", "p1", "p5", "r1"]  # the only plate there is stays
    # only a tilted fit next to the box: it is stood upright where it is, and comes after poses and regions
    a = np.radians(40.0)
    tilt = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])
    lying = {**_rej("p105", "bowl", (0.12, -0.02, 0.91), tilt=40.0)}
    lying["bodies"] = {"main": {"pos": [0.12, -0.02, 0.91], "quat_wxyz": mat_to_quat(tilt)}}
    t = _tool(kept, [lying])
    t.relation_candidates(NEXT)
    assert t.percepts[-1]["source"] == "rejected+upright:support" and _pos(t, "r1") == [0.12, -0.02, 0.91]
    assert np.allclose(t.lib.quat_mat(t.percepts[-1]["bodies"]["main"]["quat_wxyz"]), np.eye(3), atol=1e-6)
    t = _tool(kept, [lying, _rej("p103", "bowl", (0.125, 0.01, 0.9))])
    t.relation_candidates(NEXT)
    assert t.percepts[-1]["source"] == "rejected:support"
    # the one candidate does stand in the relation: nothing is added
    t = _tool([_p("p0", "bowl", (0.13, 0.0, 0.9)), *kept[1:]], [_rej("p103", "bowl", (0.125, 0.3, 0.9))])
    t.relation_candidates(NEXT)
    assert [p["id"] for p in t.percepts] == ["p0", "p1", "p2"] and t.relation_target is None
    # a landmark with no percept: its best rejected pose that is no copy stands in, outside the percepts
    t = _tool([kept[0], kept[2]], [_rej("p100", "box", (-0.3, -0.3, 0.9), n_agree=1500, accepted=True),
                                   _rej("p101", "box", (0.0, 0.0, 0.9), n_agree=900),
                                   _rej("p103", "bowl", (0.125, 0.01, 0.9))])
    fb = t.relation_candidates(NEXT)
    assert [q["type"] for q in fb] == ["box"] and fb[0]["bodies"]["main"]["pos"] == [0.0, 0.0, 0.9]
    assert [p["id"] for p in t.percepts] == ["p0", "p2", "r1"] and all(p["type"] != "box" for p in t.percepts)
    assert G.ground(NEXT, t.percepts, t.lib, TABLE, extra_landmarks=fb)["target"] == "r1"
    # a rejected pose of the landmark on another reported object is no stand-in (a cookie box read onto the stove's
    # burner); under a candidate of the target's type it is (the bowl stands on the box and hides it)
    on_stove = _rej("p104", "box", (-0.3, -0.3, 0.92), n_agree=5000)
    t = _tool([kept[0], kept[2], _p("p3", "stove", (-0.3, -0.3, 0.9))], [on_stove])
    assert t.relation_candidates(NEXT) == []
    t = _tool([kept[0], kept[2], _p("p3", "stove", (-0.3, -0.3, 0.9))], [on_stove, _rej("p105", "box", (0.41, 0.01, 0.88))])
    assert [q["bodies"]["main"]["pos"] for q in t.relation_candidates(NEXT)] == [[0.41, 0.01, 0.88]]


# ---- the fallback follows the evidence
PUT = {"target_type": None, "place_type": "plate"}


def _run(rejected, blobs, typ="plate", on=True, kept=(), role="place"):
    t = _tool(list(kept), rejected, blobs)
    t.fallback_top = on
    t.on_parsed({"target_type": typ if role == "target" else None, "place_type": typ if role == "place" else None})
    got = [p for p in t.percepts if p["type"] == typ]
    return (got[0]["source"], [round(float(x), 4) for x in got[0]["bodies"]["main"]["pos"]]) if got else None


def test_a_rejected_pose_that_explains_its_own_place_comes_first_and_a_copy_never():
    clear = _rej("p100", "plate", (0.2, 0.1, 0.9), n_agree=500)
    copy = _rej("p101", "plate", (-0.3, -0.3, 0.92), n_agree=3000, accepted=True)
    assert _run([copy, clear], []) == ("rejected:support", [0.2, 0.1, 0.9])
    assert _run([copy, clear], [], on=False) == ("rejected:support", [-0.3, -0.3, 0.92])  # off: the strongest, as before
    assert _run([copy], [], on=True) == ("rejected:support", [-0.3, -0.3, 0.92])  # nothing else: the older fallbacks


def test_an_error_in_the_new_fallback_leaves_the_older_ones_standing():
    clear = _rej("p100", "plate", (0.2, 0.1, 0.9), n_agree=500)
    t = _tool([], [clear], [])
    t.fallback_top = True
    t._from_evidence = lambda typ, role, used: 1 / 0
    t.on_parsed({"target_type": None, "place_type": "plate"})
    assert [p["source"] for p in t.percepts] == ["rejected:support"]


def test_a_region_that_fits_as_a_whole_beats_a_pose_under_a_taller_region():
    # a plate drawn at 0.8 of its size: no pose where it is, a low region there; and a false pose under a taller
    # region (the enlarged bowl) elsewhere
    false_pose = _rej("p100", "plate", (-0.09, 0.02, 0.9))
    tall = _blob((-0.10, 0.02), (0.096, 0.077), 0.973, 0.064, top=((-0.10, 0.02), (0.06, 0.06), 0.97))
    low = _blob((0.06, 0.19), (0.056, 0.058), 0.921, 0.018, top=((0.06, 0.19), (0.05, 0.05), 0.92))
    assert _run([false_pose], [tall, low]) == ("blob_whole", [0.06, 0.19, 0.903])
    assert _run([false_pose], [tall, low], on=False) == ("rejected:support", [-0.09, 0.02, 0.9])  # as before


def test_a_pose_under_a_taller_region_is_raised_to_its_top_and_centred_on_the_top_layer():
    # a basket on a block: the rejected pose is at table level under it, 12 cm low and 2 cm off across
    low = _rej("p100", "basket", (0.02, 0.27, 0.9))
    region = _blob((-0.05, 0.30), (0.30, 0.16), 1.165, 0.265, top=((0.0, 0.272), (0.09, 0.075), 1.16))
    src, pos = _run([low], [region], typ="basket")
    assert src == "rejected+raised:support" and pos == [0.0, 0.272, 1.02]  # its top (0.14 above the root) at 1.16
    # no top layer reported (the server's option off): nothing says what tops the region, the pose stays as it is
    assert _run([low], [{k: v for k, v in region.items() if k != "top"}], typ="basket") == ("rejected:support", [0.02, 0.27, 0.9])
    assert _run([low], [region], typ="basket", on=False) == ("rejected:support", [0.02, 0.27, 0.9])
    # a target is never raised: the older fallbacks stand for it
    assert _run([low], [region], typ="basket", role="target") == ("rejected:support", [0.02, 0.27, 0.9])
    # a region only 4 cm above the pose's top is the object itself read a little high, and one topped by something
    # of another size (a panel in front of it) does not stand on it: the pose is taken as it is
    near = _blob((0.0, 0.27), (0.12, 0.10), 1.08, 0.18, top=((0.0, 0.272), (0.09, 0.075), 1.08))
    assert _run([low], [near], typ="basket") == ("rejected:support", [0.02, 0.27, 0.9])
    # ... also when that region would itself fit the model as a whole: the pose has the turn, the region has not
    whole = _blob((0.02, 0.27), (0.10, 0.08), 1.08, 0.18, top=((0.02, 0.27), (0.09, 0.075), 1.08))
    assert _run([low], [whole], typ="basket") == ("rejected:support", [0.02, 0.27, 0.9])
    panel = _blob((0.0, 0.27), (0.22, 0.15), 1.30, 0.39, top=((0.0, 0.27), (0.22, 0.01), 1.29))
    assert _run([low], [panel], typ="basket") == ("rejected:support", [0.02, 0.27, 0.9])


def test_a_region_whose_top_layer_has_the_models_size_takes_the_model_on_that_layer():
    # a plate on a block: no pose at all; the region is smeared 10 cm to one side, its top layer is the plate
    region = _blob((-0.03, 0.21), (0.20, 0.11), 1.088, 0.177, top=((0.061, 0.192), (0.063, 0.063), 1.081))
    thin = _blob((0.065, -0.18), (0.057, 0.014), 0.955, 0.054, top=((0.065, -0.18), (0.057, 0.012), 0.955))
    assert _run([], [thin, region]) == ("blob_top", [0.061, 0.192, 1.061])
    src, pos = _run([], [thin, region], on=False)
    assert src == "blob" and pos != [0.061, 0.192, 1.061]  # as before: the whole region's middle
    small = _blob((0.3, 0.0), (0.03, 0.02), 0.92, 0.02, top=((0.3, 0.0), (0.03, 0.02), 0.92))
    assert _run([], [thin]) is None  # a sliver: no fallback takes it, old or new
    assert _run([], [small])[0] == "blob"  # under half the model's size: only the older fallbacks take it


# ---- a receiver's believed move
class Scene:
    def __init__(self, fresh, rejected=()):
        self.fresh, self.rejected = fresh, rejected

    def call(self, msg):
        return {"percepts": [dict(q) for q in self.fresh], "extra": {"rejected": list(self.rejected)}, "times": {}}


def _relook(fresh, rejected=(), strict=True):
    t = _tool([_p("p0", "bowl", (0.06, 0.19, 0.97)), _p("p1", "plate", (0.06, 0.19, 0.9)),
               _p("p2", "bowl", (0.01, 0.305, 0.9)), _p("p3", "box", (0.3, -0.2, 0.9))])
    t.client = Scene(fresh, rejected)
    t.obs = {"robot0_joint_pos": np.zeros(7), "robot0_gripper_qpos": np.zeros(2),
             "robot0_eef_pos": np.array([0.06, 0.19, 1.0]), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0])}
    t.target, t.calls, t.relook_strict = "p0", 1, strict
    return t.relook_place("p1"), _pos(t, "p1")


def test_a_receiver_is_not_believed_to_have_moved_onto_another_object_or_to_another_height():
    others = [_p("q2", "bowl", (0.01, 0.305, 0.9)), _p("q3", "box", (0.3, -0.2, 0.9))]
    on_the_bowl = _rej("p100", "plate", (-0.02, 0.29, 0.9))  # 12 cm away, on the second bowl
    r, pos = _relook(others, [on_the_bowl])
    assert r["moved_m"] == 0.0 and r["refused"] == "on:p2" and abs(r["unconfirmed_m"] - 0.128) < 1e-3
    assert pos == [0.06, 0.19, 0.9]
    r, pos = _relook(others, [on_the_bowl], strict=False)  # off: followed, as before
    assert abs(r["moved_m"] - 0.128) < 1e-3 and "refused" not in r and pos == [-0.02, 0.29, 0.9]
    r, pos = _relook(others, [_rej("p100", "plate", (0.16, 0.14, 1.01))])  # 11 cm up: another surface
    assert r["moved_m"] == 0.0 and r["refused"] == "height" and pos == [0.06, 0.19, 0.9]
    r, pos = _relook(others + [_p("q1", "plate", (0.20, 0.14, 0.9))])  # a level jump of 15 cm to free table: believed
    assert abs(r["moved_m"] - 0.149) < 1e-3 and "refused" not in r and pos == [0.2, 0.14, 0.9]
    assert r["dz_m"] == 0.0


# ---- the server's report of a region's top layer
def test_the_top_layer_of_a_region_is_the_thing_standing_on_it():
    from rowant_lab.libero_env import m2_perceive as MP

    assert MP.PARAMS["extra_blob_top_m"] is None  # off: the reply is as before
    rng = np.random.default_rng(0)
    a, r = rng.uniform(0, 2 * np.pi, 4000), 0.07 * np.sqrt(rng.uniform(0, 1, 4000))
    plate = np.c_[0.07 + r * np.cos(a), 0.2 + r * np.sin(a), 1.075 + rng.uniform(0, 0.006, 4000)]
    skirt = np.c_[rng.uniform(-0.15, 0.0, 3000), rng.uniform(0.1, 0.4, 3000), rng.uniform(0.93, 1.05, 3000)]
    top = MP.blob_top(np.r_[plate, skirt], 0.015)
    assert np.allclose(top["xy"], [0.07, 0.2], atol=0.003) and np.allclose(top["half_xy"], 0.063, atol=0.004)
    assert abs(top["z"] - 1.08) < 0.002 and top["n_px"] == 4000
    assert MP.blob_top(plate[:10], 0.015) is None


# ---- 2026-11-16, ``params["relation_look"]``: the place the relation gives, for a look from above
def test_the_relations_place_for_a_look_from_above():
    t = _tool([_p("p0", "bowl", (0.42, 0.0, 0.9)), _p("p1", "box", (0.0, 0.0, 0.9)), _p("p2", "plate", (0.3, 0.3, 0.9)),
               _p("p3", "stove", (-0.3, -0.3, 0.9))])
    lib, ps = t.lib, t.percepts
    pt = G.relation_point(NEXT, ps, lib, TABLE)
    assert pt == {"xy": [0.0, 0.0], "support_z": 0.9, "relation": "next_to", "within_m": 0.22}  # above the box
    pt = G.relation_point({**NEXT, "relation": "between", "landmarks": ["box", "plate"]}, ps, lib, TABLE)
    assert np.allclose(pt["xy"], [0.15, 0.15]) and pt["support_z"] == 0.9  # halfway
    assert 0.14 < pt["within_m"] < 0.17  # 35% of the 42 cm between the two, and 6 cm to the side
    pt = G.relation_point({**NEXT, "relation": "on", "landmarks": ["stove"]}, ps, lib, TABLE)
    assert np.allclose(pt["xy"], [-0.3, -0.3]) and abs(pt["support_z"] - 0.92) < 1e-9  # on the stove's top
    pt = G.relation_point({**NEXT, "relation": "table_centre", "landmarks": ["table"]}, ps, lib, TABLE)
    assert pt["xy"] == [0.0, 0.0] and pt["support_z"] == 0.9
    assert G.relation_point({**NEXT, "relation": None}, ps, lib, TABLE) is None
    assert G.relation_point({**NEXT, "landmarks": ["basket"]}, ps, lib, TABLE) is None  # no such landmark in view
    stand_in = [_p("x", "basket", (0.2, -0.2, 0.9))]
    assert G.relation_point({**NEXT, "landmarks": ["basket"]}, ps, lib, TABLE, stand_in)["xy"] == [0.2, -0.2]


def test_the_wrist_picture_can_prefer_the_thing_of_the_named_size_near_the_expected_point():
    from rowant_lab.pro2.robot import wrist2d as W2

    n = 256
    rgb = np.full((n, n, 3), 200, np.uint8)  # a pale table seen from straight above
    K, T = W2.camera(np.array([0.0, 0.0, 1.27]), np.diag([1.0, -1.0, -1.0]), n)

    def paint(xy, half, colour):
        uv, _z = W2.project(K, T, [[xy[0] - half[0], xy[1] - half[1], 0.95], [xy[0] + half[0], xy[1] + half[1], 0.95]])
        u0, u1 = sorted((int(uv[0][0]), int(uv[1][0])))
        v0, v1 = sorted((int(uv[0][1]), int(uv[1][1])))
        rgb[max(v0, 0):v1, max(u0, 0):u1] = colour

    paint((0.06, 0.00), (0.02, 0.045), (60, 60, 60))    # a slab, 6 cm from the expected point
    paint((-0.03, 0.08), (0.05, 0.05), (30, 30, 90))    # a bowl-sized thing, 8.5 cm from it
    area = 0.107 * 0.107
    near = W2.find(rgb, K, T, 0.95, hint_xy=(0.0, 0.0), expect_area_m2=area)
    assert near is not None and abs(near["xy"][0] - 0.06) < 0.02  # as before: the nearest thing, the slab
    size = W2.find(rgb, K, T, 0.95, hint_xy=(0.0, 0.0), expect_area_m2=area, opts={"prefer": "size", "within_m": 0.25})
    assert size is not None and abs(size["xy"][0] + 0.03) < 0.02 and abs(size["xy"][1] - 0.08) < 0.02
    none = W2.find(rgb, K, T, 0.95, hint_xy=(0.0, 0.0), expect_area_m2=area, opts={"prefer": "size", "within_m": 0.04})
    assert none is None  # nothing stands that near the point
    # the things in the order of that preference, for a caller that asks a second viewpoint
    assert [abs(c["xy"][0] + 0.03) < 0.02 for c in size["ranked"]] == [True, False] and "ranked" not in near
    assert size["ranked"][0]["area_m2"] == size["area_m2"] > size["ranked"][1]["area_m2"]


def test_the_outline_seen_from_above_is_the_union_of_the_boxes_footprints():
    from rowant_lab.pro2.robot import wrist2d as W2

    E = np.eye(3)
    one = [((0.0, 0.0, 0.0), E, (0.05, 0.02, 0.01))]
    assert abs(W2.footprint_area(one) - 0.1 * 0.04) < 2e-4
    # two boxes that overlap count their common ground once; a box above another adds nothing
    cross = one + [((0.0, 0.0, 0.0), E, (0.02, 0.05, 0.01)), ((0.0, 0.0, 0.05), E, (0.01, 0.01, 0.01))]
    assert abs(W2.footprint_area(cross) - (2 * 0.1 * 0.04 - 0.04 * 0.04)) < 3e-4
    # a box turned 45 degrees about the vertical keeps its footprint; one tipped on its edge shows more
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    assert abs(W2.footprint_area([((0.0, 0.0, 0.0), np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]), (0.05, 0.02, 0.01))])
               - 0.004) < 3e-4
    tipped = W2.footprint_area([((0.0, 0.0, 0.0), np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]]), (0.05, 0.02, 0.01))])
    assert abs(tipped - 0.1 * 2 * (0.02 + 0.01) * c) < 3e-4
    assert W2.footprint_area([]) == 0.0
