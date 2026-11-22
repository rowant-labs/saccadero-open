"""CLEAR stream (2026-11-16): the wait for a scene at rest, the push of a thing off the receiver, and an object taken
as it is seen. All off by default; these tests run no simulator and no perception server."""
from types import SimpleNamespace

import numpy as np

from rowant_lab.pro2 import dev_episode as DE
from rowant_lab.pro2.robot import agent as AG
from rowant_lab.pro2.robot import grasp as GR
from rowant_lab.pro2.robot import seen as SN


# ------------------------------------------------------------------ the first look only when the scene is at rest
def test_a_moving_thing_is_seen_in_depth_and_in_colour_and_a_change_of_the_whole_picture_is_not_motion():
    d0 = np.full((128, 128), 1.0)
    d1 = d0.copy()
    d1[40:44, 40:44] -= 0.03                                   # a 4 x 4 px thing 3 cm nearer
    assert DE.scene_moving(d0, d1) and not DE.scene_moving(d0, d0)
    assert not DE.scene_moving(d0, d0 + 0.02)                  # every pixel changed: the camera, not the scene
    assert not DE.scene_moving(d0, d0 + 0.001)                 # under the depth limit
    c0 = np.full((1024, 1024, 3), 120, np.uint8)
    c1 = c0.copy()
    c1[300:380, 500:560] = 30                                  # an object-sized patch changed
    assert DE.scene_moving(c0, c1) and not DE.scene_moving(c0, c0)
    rng = np.random.default_rng(0)
    noisy = np.clip(c0.astype(int) + rng.integers(-12, 13, c0.shape), 0, 255).astype(np.uint8)
    assert not DE.scene_moving(c0, noisy)                      # pixel noise averages out over a block
    assert not DE.scene_moving(c0, (c0 * 0.7).astype(np.uint8))  # the exposure changed: the whole picture
    assert not DE.scene_moving(c0, c0[:512])                   # frames of different sizes are not compared


def test_the_wait_ends_after_two_still_pairs_and_a_still_scene_costs_two_steps():
    def run(frames, max_steps=12):
        sent = []

        def step():
            sent.append(1)
            return {"n": len(sent), "terminated": False, "truncated": False}

        def frame(r):
            return frames[min(0 if r is None else r["n"], len(frames) - 1)]

        replies, rec = DE.settle_first(step, frame, {"max_steps": max_steps})
        return len(replies), rec

    still = np.full((128, 128), 1.0)

    def moved(k):
        f = still.copy()
        f[10:20, 10:20] -= 0.01 * k
        return f

    assert run([still, still, still]) == (2, {"steps": 2, "moving": [False, False]})
    n, rec = run([moved(5), moved(3), moved(1), still, still, still])   # falling for three frames, then at rest
    assert n == 5 and rec["moving"] == [True, True, True, False, False]
    slow = still.copy()
    slow[10:20, 10:20] -= 0.002                                      # the last 2 mm of a settling thing still count
    assert DE.scene_moving(still, slow)
    n, rec = run([moved(k) for k in range(40, 0, -1)], max_steps=6)   # never at rest: the wait is bounded
    assert n == 6 and all(rec["moving"])


# ------------------------------------------------------------------ pushing a thing off the receiver
def _agent(hand, held_lo, others=(), predicate="in", cap=280, t=150, region=None, shoulder=None, kind="object",
           **opts):
    """A stand-in with what ``_cover_plan`` reads: a basket-sized receiver at the origin with its rim at z 0.10, a
    held box whose lowest point is ``held_lo`` below the hand, and the hand's closing axis along world y."""
    R_hand = GR.hand_matrix([0.0, 1.0, 0.0])
    a = SimpleNamespace(ap=AG.AgentParams(clear_cover=True, **opts), cap=cap, t=t, target="tgt",
                        spec={"predicate": predicate}, jaw=GR.JawModel(), supports=GR.Boxes.empty(), n_cleared=0,
                        shoulder=shoulder, scene={"instances": {"basket": {"kind": kind}}})
    a._pose = lambda: (np.asarray(hand, float), R_hand)
    basket = GR.Boxes.aabb_box((-0.09, -0.07, 0.0), (0.09, 0.07, 0.10))
    basket.tags = ["basket"]
    a._place_goal = lambda poses: (np.zeros(2), "basket", region)
    a._instance_boxes = lambda poses, inst: basket
    a._held_boxes = lambda R_rel, t_rel, p, R: GR.Boxes.aabb_box((-0.04, -0.02, -held_lo), (0.04, 0.02, 0.0), origin=p)
    extra = GR.Boxes.empty()
    for lo, hi in others:
        extra = extra + GR.Boxes.aabb_box(lo, hi)
    a._world = lambda poses, exclude=(): basket + extra
    return a, R_hand


def test_a_descent_stopped_on_a_cover_plans_a_push_along_the_closing_axis_just_above_the_rim():
    # the held box's lowest point rests on a 2 cm cover on the rim (0.10): the hand stopped 8 cm short of its goal
    a, R = _agent(hand=(0.0, 0.0, 0.12 + 0.03), held_lo=0.03)
    plan, rec = AG.OracleAgent._cover_plan(a, {}, np.array([0.0, 0.0, 0.07]), R, None, None, 0.04)
    assert plan is not None and rec["top"] == 0.12 and abs(rec["dz"] - 0.08) < 1e-9
    assert abs(abs(rec["dir"][1]) - 1.0) < 1e-6 and abs(rec["dir"][0]) < 1e-6      # along the closing axis (y)
    assert abs(rec["drop"] - 0.014) < 1e-6           # the box's lowest point: 6 mm above the rim, 14 mm into the cover
    d = np.array(rec["dir"])
    # the pushing face starts outside the cover (taken 3 cm wider than the receiver) and ends 2 cm past where the held
    # box's own far side will be when it is set down in the middle (the box is 4 cm across the closing axis)
    face0 = float(plan["start"][:2] @ d) + rec["lead"]
    face1 = float(plan["end"][:2] @ d) + rec["lead"]
    assert face0 < -(0.07 + 0.03) and abs(face1 - (0.02 + 0.02)) < 1e-6
    # the held box (4 cm across the closing axis) is lower than the fingertips here, so its side leads, not a finger
    assert abs(rec["lead"] - 0.02) < 1e-6
    assert abs(plan["start"][2] - (0.15 - 0.014)) < 1e-6 and plan["up"][2] > 0.15 and plan["back"][2] == plan["up"][2]


def test_a_receiver_read_too_high_does_not_stop_the_push_but_another_thing_in_the_way_does():
    # the held box rests on a 2.3 cm cover whose top is at 0.167; the look has the basket's rim 1 cm high, at 0.152
    a, R = _agent(hand=(0.0, 0.0, 0.167 + 0.03), held_lo=0.03)
    high = GR.Boxes.aabb_box((-0.09, -0.07, 0.0), (0.09, 0.07, 0.152))
    high.tags = ["basket"]
    a._instance_boxes = lambda poses, inst: high
    a._world = lambda poses, exclude=(): high
    plan, rec = AG.OracleAgent._cover_plan(a, {}, np.array([0.0, 0.0, 0.10]), R, None, None, 0.04)
    assert plan is not None and abs(rec["drop"] - 0.010) < 1e-9       # at least 1 cm of the thing is taken
    other = GR.Boxes.aabb_box((-0.03, -0.2, 0.0), (0.03, 0.2, 0.165))   # something else across the whole path
    other.tags = ["bottle"]
    a._world = lambda poses, exclude=(): high + other
    plan, rec = AG.OracleAgent._cover_plan(a, {}, np.array([0.0, 0.0, 0.10]), R, None, None, 0.04)
    assert plan is None and rec["skipped"] == "nothing to push against"


def test_an_ordinary_release_a_far_stop_and_too_few_steps_plan_nothing():
    p_rel = np.array([0.0, 0.0, 0.07])
    a, R = _agent(hand=(0.0, 0.0, 0.078), held_lo=0.03)             # 8 mm short: an ordinary release
    assert AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04) == (None, None)
    a, R = _agent(hand=(0.05, 0.0, 0.15), held_lo=0.03)             # stopped 5 cm to the side: not above the goal
    assert AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04) == (None, None)
    a, R = _agent(hand=(0.0, 0.0, 0.25), held_lo=0.03)              # 18 cm short: not a cover
    assert AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04) == (None, None)
    a, R = _agent(hand=(0.0, 0.0, 0.15), held_lo=0.03, t=250)       # 30 steps left
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04)
    assert plan is None and rec["skipped"] == "no steps"


def test_a_touch_down_well_above_a_surface_plans_a_push_and_one_on_the_surface_does_not():
    # a release ON a region whose top is the rim height here (0.10): the object touched 1.5 cm above it
    region = {"centre": np.array([0.0, 0.0, 0.095]), "half": np.array([0.06, 0.06, 0.005])}
    a, R = _agent(hand=(0.0, 0.0, 0.115 + 0.03), held_lo=0.03, predicate="on", region=region)
    p_rel = np.array([0.0, 0.0, 0.10 + 0.02 + 0.03])              # planned 2 cm above the surface
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04, touched_m=0.015)
    assert plan is not None and rec["touched_m"] == 0.015 and abs(rec["drop"] - 0.010) < 1e-6 and rec["r_recv"] == 0.06
    assert AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04, touched_m=0.004) == (None, None)
    # a region of a fixture is the receiver's extent whatever the predicate (a drawer of a cabinet, not the cabinet)
    a, R = _agent(hand=(0.0, 0.0, 0.15), held_lo=0.03, predicate="in", region=region, kind="fixture")
    assert AG.OracleAgent._cover_plan(a, {}, np.array([0.0, 0.0, 0.07]), R, None, None, 0.04)[1]["r_recv"] == 0.06
    a, R = _agent(hand=(0.0, 0.0, 0.15), held_lo=0.03, predicate="in", region=region)       # a basket: its own boxes
    assert AG.OracleAgent._cover_plan(a, {}, np.array([0.0, 0.0, 0.07]), R, None, None, 0.04)[1]["r_recv"] == 0.07


def test_the_push_goes_to_the_side_where_nothing_stands_in_its_way():
    p_rel = np.array([0.0, 0.0, 0.07])
    tall = ((-0.05, 0.12, 0.0), (0.05, 0.20, 0.25))                # a tall thing 5 cm past the receiver on +y
    a, R = _agent(hand=(0.0, 0.0, 0.15), held_lo=0.03, others=[tall])
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04)
    # a push toward -y would start where the tall thing stands, so the push goes toward it (it ends well short of it)
    assert plan is not None and rec["dir"][1] > 0.99 and rec["room"] == 0.05
    assert float(plan["end"][1]) + rec["lead"] <= 0.11 + 1e-9
    tall = ((-0.05, 0.25, 0.0), (0.05, 0.33, 0.25))                # further off: both ways are free
    a, R = _agent(hand=(0.0, 0.0, 0.15), held_lo=0.03, others=[tall])
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04)
    assert plan is not None and rec["dir"][1] < -0.99 and rec["room"] == 0.3   # away from it: the thing falls clear
    # both ways free: the push starts on the arm's own side and goes away from it (its far point is then the
    # nearer one: the start lies further out than the end), so the thing comes down beyond the receiver
    a, R = _agent(hand=(0.0, 0.0, 0.15), held_lo=0.03, shoulder=(-0.3, -0.6, 0.3))
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04)
    assert rec["dir"][1] > 0.99
    a, R = _agent(hand=(0.0, 0.0, 0.15), held_lo=0.03, shoulder=(-0.3, 0.6, 0.3))
    assert AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04)[1]["dir"][1] < -0.99
    walls = [((-0.05, 0.12, 0.0), (0.05, 0.20, 0.25)), ((-0.05, -0.20, 0.0), (0.05, -0.12, 0.25))]
    a, R = _agent(hand=(0.0, 0.0, 0.15), held_lo=0.03, others=walls)          # tall things on both sides
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04)
    assert plan is None and rec["skipped"] == "nothing to push against"


def test_the_sweep_runs_its_moves_in_order_and_reports_a_lost_object():
    def run(lost_at=None):
        a, R = _agent(hand=(0.0, 0.0, 0.15), held_lo=0.03)
        plan, rec = AG.OracleAgent._cover_plan(a, {}, np.array([0.0, 0.0, 0.07]), R, None, None, 0.04)
        seen = []

        def goto(p, R_, grip, cap, tol, tol_deg, max_steps=None):
            seen.append(np.asarray(p, float).copy())
            a._pose = lambda p=np.asarray(p, float): (p, R)          # the move arrives
            a.t += 3
            yield [0.0] * 6 + [1.0]
            return "reached"

        def watch(gen, gap):
            st = yield from gen
            return "lost" if lost_at is not None and len(seen) > lost_at else st

        a._goto, a._watch_carry = goto, watch
        g = AG.OracleAgent._clear_cover(a, plan, rec, R, 0.04)
        try:
            while True:
                next(g)
        except StopIteration as e:
            return e.value, seen, plan, rec

    how, seen, plan, rec = run()
    assert how == "swept" and len(seen) == 4                        # to the side, down beside it, across, back
    assert np.allclose(seen[0], plan["start_up"]) and np.allclose(seen[1], plan["start"])
    assert np.allclose(seen[2], plan["end"]) and np.allclose(seen[3], plan["back"])
    assert rec["push"] == "reached" and rec["pushed_m"] > 0.12
    assert run(lost_at=2)[0] == "lost"                              # the fingers met during the push


# ------------------------------------------------------------------ an object taken as it is seen
def _lib():
    return SimpleNamespace(types={}, export_types={}, records={}, jaw=GR.JawModel())


def test_a_seen_region_becomes_a_box_model_with_grasps_across_its_narrow_side():
    lib = _lib()
    blob = {"xy": [0.10, -0.20], "half_xy": [0.045, 0.022], "yaw": 0.5, "z_top": 0.95, "height": 0.05, "n_px": 900}
    half, R, mid = SN.box_of(blob)
    assert np.allclose(half, [0.045, 0.022, 0.025]) and np.allclose(mid, [0.10, -0.20, 0.925])
    assert np.allclose(R[:, 0], [np.cos(0.5), np.sin(0.5), 0.0])       # the box's long side along the region's
    p = SN.seen_percept(lib, blob, "s1", named="butter")
    name = p["type"]
    assert name == "seen_45x22x25" and p["named"] == "butter" and p["gid"] is None and p["source"] == "seen"
    assert SN.seen_type(lib, half) == name and len(lib.types) == 1      # the same box is the same model
    rec = lib.records[name]
    assert SN.n_grasps(lib, name) > 0
    # every record closes across the 4.4 cm side (the 9 cm side does not fit between the open fingers)
    assert np.allclose(np.abs(rec["axis"][:, 1]), 1.0, atol=0.3) and float(rec["width"].max()) < 0.06
    assert lib.export_types[name]["root_body"] == "main" and lib.export_types[name]["sites"] == {}
    assert lib.types[name]["bodies"]["main"]["primitives"][0]["half"] == [0.045, 0.022, 0.025]


def test_a_thing_wider_than_the_open_hand_has_no_grasp_and_one_standing_on_another_keeps_its_own_height():
    lib = _lib()
    cover = {"xy": [0.0, 0.0], "half_xy": [0.10, 0.09], "yaw": 0.0, "z_top": 0.14, "height": 0.14, "n_px": 5000}
    half, _R, mid = SN.box_of(cover, bottom_z=0.12)                    # it lies on the receiver's rim at 0.12
    assert abs(half[2] - 0.01) < 1e-9 and abs(mid[2] - 0.13) < 1e-9
    name = SN.seen_type(lib, half)
    assert SN.n_grasps(lib, name) == 0
    k, b = SN.blob_near([cover, {"xy": [0.3, 0.3], "half_xy": [0.02, 0.02], "z_top": 0.03, "height": 0.03, "n_px": 400}],
                        (0.28, 0.31), 0.05)
    assert k == 1 and b["n_px"] == 400
    assert SN.blob_near([cover], (0.5, 0.5), 0.05) == (None, None)


# ------------------------------------------------------------------ a target the main camera does not see
from rowant_lab.libero_env.m2_kin import mat_to_quat  # noqa: E402
from rowant_lab.pro2.robot import behind as BH  # noqa: E402
from rowant_lab.pro2.robot import ground as GD  # noqa: E402
from rowant_lab.pro2.robot import look as LK  # noqa: E402


def _tdoc(kind="object"):
    return {"bodies": {"main": {"body_pos": [0, 0, 0], "body_quat": [1, 0, 0, 0], "geoms": [], "is_root": True,
                                "joints": [], "parent": None}},
            "kind": kind, "part_of": {"main": "main"}, "parts": {"main": {"bodies": ["main"], "joint": None,
                                                                              "parent_body": None}},
            "pose_in_part": {"main": {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]}}, "root_body": "main", "sites": {}}


def _glib():
    """Bowls 14 cm across and 6 cm high (origin at the base), a plate and a ramekin, on a table whose top is z 0.9."""
    def m1(half, cz):
        return {"bodies": {"main": {"primitives": [{"centre": [0.0, 0.0, cz], "R": np.eye(3).tolist(), "half": half}]}}}

    types = {"bowl": m1([0.07, 0.07, 0.03], 0.03), "plate": m1([0.10, 0.10, 0.01], 0.01),
             "ramekin": m1([0.05, 0.05, 0.025], 0.025)}
    return SimpleNamespace(types=types, m1_type=lambda t: types[t], export_types={t: _tdoc() for t in types},
                           quat_mat=GR.quat_wxyz_to_mat, site_for=lambda p, part: None,
                           workcell={"arena": [{"type": "box", "centre": [0.0, 0.0, 0.45], "R": np.eye(3).tolist(),
                                                "half": [0.6, 0.6, 0.45]}]})


def _gp(pid, typ, xy, z=0.9):
    return {"id": pid, "type": typ, "gid": "g", "fit": {"n_agree": 100}, "joints": {},
            "bodies": {"main": {"pos": [xy[0], xy[1], z], "quat_wxyz": mat_to_quat(np.eye(3))}}}


def test_the_only_candidate_is_not_taken_when_it_clearly_fails_the_relation():
    lib = _glib()
    scene = [_gp("pl", "plate", (0.0, 0.0)), _gp("rm", "ramekin", (0.0, 0.30))]
    between = {"target_type": "bowl", "place_type": "plate", "relation": "between", "landmarks": ["plate", "ramekin"],
               "part": None}
    next_to = {**between, "relation": "next_to", "landmarks": ["plate"]}
    far, mid, beside = _gp("b1", "bowl", (0.30, 0.10)), _gp("b2", "bowl", (0.01, 0.15)), _gp("b3", "bowl", (0.17, 0.0))
    assert GD.single_fails(between, far, scene, lib) and not GD.single_fails(between, mid, scene, lib)
    assert GD.single_fails(next_to, far, scene, lib) and not GD.single_fails(next_to, beside, scene, lib)
    centre = {**between, "relation": "table_centre", "landmarks": ["table"]}             # the table's middle is (0, 0)
    assert GD.single_fails(centre, far, scene, lib) and not GD.single_fails(centre, mid, scene, lib)
    assert not GD.single_fails({**between, "relation": None}, far, scene, lib)            # no relation: cannot be told
    assert not GD.single_fails({**between, "landmarks": ["plate", "stove"]}, far, scene, lib)  # a landmark not in view
    g = GD.ground(between, scene + [far], lib, lib.workcell)                              # off: taken as before
    assert g["status"] == "ok" and g["target"] == "b1"
    g = GD.ground(between, scene + [far], lib, lib.workcell, strict_single=True)
    assert g["status"] == "fail" and g["hidden"] and g["only"] == "b1" and "single_fails_relation" in g["flags"]
    g = GD.ground(between, scene + [mid], lib, lib.workcell, strict_single=True)          # the right one: taken
    assert g["status"] == "ok" and g["target"] == "b2"
    g = GD.ground(between, scene + [far, mid], lib, lib.workcell, strict_single=True)     # two: chosen by the relation
    assert g["status"] == "ok" and g["target"] == "b2"


def test_a_thing_taken_as_seen_counts_as_the_type_it_is_named():
    lib = _glib()
    lib.types["seen_30x20x20"] = {"bodies": {"main": {"primitives": [{"centre": [0, 0, 0], "R": np.eye(3).tolist(),
                                                                         "half": [0.03, 0.02, 0.02]}]}}}
    lib.export_types["seen_30x20x20"] = _tdoc()
    thing = {**_gp("s1", "seen_30x20x20", (0.2, 0.2), 0.92), "named": "bowl"}
    parsed = {"target_type": "bowl", "place_type": "plate", "relation": None, "landmarks": [], "part": None}
    g = GD.ground(parsed, [_gp("pl", "plate", (0.0, 0.0)), thing], lib, lib.workcell)
    assert g["status"] == "ok" and g["target"] == "s1"
    assert GD.is_type(thing, "bowl") and GD.is_type(thing, "seen_30x20x20") and not GD.is_type(thing, "plate")


def _camera(eye, look_at):
    """K and T of a 1024 px camera at ``eye`` looking at ``look_at`` (OpenCV: z forward, y down)."""
    eye, look_at = np.asarray(eye, float), np.asarray(look_at, float)
    z = (look_at - eye) / np.linalg.norm(look_at - eye)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.column_stack([x, y, z]), eye
    f = 0.5 * 1024 / np.tan(np.radians(45.0) / 2)
    return {"K": np.array([[f, 0, 512.0], [0, f, 512.0], [0, 0, 1.0]]), "T": T}


def _pixel(cam, X):
    Xc = cam["T"][:3, :3].T @ (np.asarray(X, float) - cam["T"][:3, 3])
    uv = cam["K"] @ (Xc / Xc[2])
    return [float(uv[0] - 0.5), float(uv[1] - 0.5)]


def test_a_hidden_target_is_placed_on_the_cameras_ray_through_the_middle_of_what_hangs_in_front():
    lib = _glib()
    cam = _camera((1.0, 0.0, 1.6), (0.0, 0.0, 0.9))
    target = np.array([0.12, -0.20, 0.9])                                  # a bowl's origin, on the table
    panel = cam["T"][:3, 3] + 0.62 * (target - cam["T"][:3, 3])            # 0.62 of the way from the camera
    occ = {"uv": _pixel(cam, panel), "n_px": 40000, "h_min": 0.17, "xy": [0.4, -0.1]}
    assert BH.occluders([occ, {"uv": [1, 1], "n_px": 40000, "h_min": 0.02}, {"uv": [1, 1], "n_px": 500, "h_min": 0.3},
                         {"n_px": 90000, "h_min": 0.3}]) == [occ]       # standing, small and unplaced regions are not
    joined = {"uv": [300.0, 520.0], "n_px": 50510, "h_min": 0.02, "hang_share": 0.8, "hang_px": 42000,
              "hang_uv": [258.0, 500.0]}                                # a standing thing joined to it in the picture
    assert BH.occluders([joined])[0]["uv"] == [258.0, 500.0]            # taken, by the middle of what hangs
    assert BH.occluders([{**joined, "hang_share": 0.2}]) == [] and BH.occluders([{**joined, "hang_px": 900}]) == []
    p, rec = BH.behind_percept(lib, "bowl", occ, cam, "b1")
    assert p["source"] == "behind" and p["gid"] is None and p["type"] == "bowl"
    assert np.allclose(p["bodies"]["main"]["pos"], target, atol=1e-6) and rec["support_z"] == 0.9
    # standing on a plate: the plate's top is the support, and the point moves along the slanted ray
    plate = _gp("pl", "plate", (0.12, -0.20))
    on_plate = np.array([0.12, -0.20, 0.92])
    occ2 = {**occ, "uv": _pixel(cam, cam["T"][:3, 3] + 0.62 * (on_plate - cam["T"][:3, 3]))}
    p2, rec2 = BH.behind_percept(lib, "bowl", occ2, cam, "b2", percepts=[plate])
    assert np.allclose(p2["bodies"]["main"]["pos"], on_plate, atol=1e-6) and rec2["support_z"] == 0.92
    # aimed at the middle of the object's box instead of its origin
    mid = target + np.array([0.0, 0.0, 0.03])
    occ3 = {**occ, "uv": _pixel(cam, cam["T"][:3, 3] + 0.62 * (mid - cam["T"][:3, 3]))}
    p3, _ = BH.behind_percept(lib, "bowl", occ3, cam, "b3", aim="box")
    assert np.allclose(p3["bodies"]["main"]["pos"], target, atol=1e-6)


def test_the_look_tool_adds_the_hidden_target_only_when_asked_and_something_hangs_there():
    lib = _glib()
    lib.robot = {"finger_joints": {"a": {"qpos": 0.04}, "b": {"qpos": -0.04}}}
    lib.jaw = SimpleNamespace(gap_from_qpos=lambda a, b: abs(a) + abs(b))
    cam = _camera((1.0, 0.0, 1.6), (0.0, 0.0, 0.9))
    target = np.array([0.12, -0.20, 0.9])
    occ = {"uv": _pixel(cam, cam["T"][:3, 3] + 0.62 * (target - cam["T"][:3, 3])), "n_px": 40000, "h_min": 0.17,
           "xy": [0.4, -0.1], "half_xy": [0.06, 0.08], "yaw": 0.0, "z_top": 1.3, "height": 0.4}

    def tool(behind, blobs):
        t = LK.PerceptionTool(lib, None, None, "s", "ep")
        t.percepts = [_gp("pl", "plate", (0.0, 0.0))]
        t.extra, t.cams, t.behind = {"rejected": [], "blobs": blobs}, {"agentview": cam}, behind
        return t

    parsed = {"target_type": "bowl", "place_type": "plate"}
    t = tool({"mode": "ray"}, [occ])                                       # unseen, behind the middle of it
    t.on_parsed(parsed)
    assert [p["source"] for p in t.percepts if p["type"] == "bowl"] == ["behind"] and t.n_behind == 1
    assert np.allclose(t.percepts[-1]["bodies"]["main"]["pos"], target, atol=1e-6)
    assert t.fallbacks[-1]["source"] == "behind"
    t = tool({}, [occ])                                                    # the default: no guess; a look comes first
    t.on_parsed(parsed)
    assert not [p for p in t.percepts if p["type"] == "bowl"]
    pt = t.behind_point("bowl")                                            # 5 cm above the table, on the same ray
    cam_c = cam["T"][:3, 3]
    ray = (target - cam_c) / np.linalg.norm(target - cam_c)
    want = cam_c + ray * (0.95 - cam_c[2]) / ray[2]
    assert np.allclose(pt["xy"], want[:2], atol=1e-6) and pt["support_z"] == 0.9
    low = {**occ, "h_min": 0.01}                                           # a depth that does not show it hanging:
    assert tool({}, [low]).behind_point("bowl") is None                    # not taken, unless tall regions are asked for
    assert tool({"tall_m": 0.25}, [low]).behind_point("bowl") is not None
    assert tool({"tall_m": 0.5}, [low]).behind_point("bowl") is None
    t = tool({"mode": "done"}, [occ])                                      # after the look: the fallback as before
    t.on_parsed(parsed)
    assert [p["source"] for p in t.percepts if p["type"] == "bowl"] == ["blob"]
    t = tool(None, [occ])                                                  # off: the fallback as before (the blob)
    t.on_parsed(parsed)
    assert [p["source"] for p in t.percepts if p["type"] == "bowl"] == ["blob"] and t.n_behind == 0
    t = tool({}, [{**occ, "h_min": 0.01}])                                 # on, nothing hangs: the fallback as before
    t.on_parsed(parsed)
    assert [p["source"] for p in t.percepts if p["type"] == "bowl"] == ["blob"] and t.n_behind == 0


# ------------------------------------------------------------------ the wrist camera's colour picture, from above
from rowant_lab.pro2.robot import wrist2d as W2  # noqa: E402


def _wrist_picture(K, T, things, table=(150, 120, 90), n=512):
    """A synthetic picture: each thing is a flat rectangle (centre xy, half sizes, yaw, top z, colour) on a plain table
    at z 0.9, drawn from the far plane to the near one."""
    img = np.empty((n, n, 3), np.uint8)
    img[:] = table
    v, u = np.mgrid[0:n, 0:n]
    uv = np.column_stack([u.ravel(), v.ravel()])
    for (cx, cy), (hx, hy), yaw, top, col in sorted(things, key=lambda t: t[3]):
        P = W2.to_plane(K, T, uv, top)[:, :2] - np.array([cx, cy])
        c, s = np.cos(yaw), np.sin(yaw)
        a, b = P @ np.array([c, s]), P @ np.array([-s, c])
        img.reshape(-1, 3)[(np.abs(a) <= hx) & (np.abs(b) <= hy)] = col
    return img


def test_the_wrist_camera_looks_down_the_gripper_and_a_thing_is_found_where_it_stands():
    R = GR.hand_matrix([0.0, 1.0, 0.0])                         # a top-down hand, closing along world y
    hand = np.array([0.10, -0.20, 1.20])
    K, T = W2.camera(hand, R, 512)
    assert np.allclose(T[:3, 2], [0.0, 0.0, -1.0], atol=1e-9)   # the camera's viewing axis: straight down
    assert abs(T[2, 3] - (1.20 + 0.097)) < 1e-9                 # 9.7 cm behind the grip site, 5 cm to its side
    assert abs(np.linalg.norm(T[:2, 3] - hand[:2]) - 0.05) < 1e-9
    uv, z = W2.project(K, T, [[0.12, -0.18, 0.95]])
    assert np.allclose(W2.to_plane(K, T, uv, 0.95)[0], [0.12, -0.18, 0.95], atol=1e-9) and z[0] > 0
    box = ((0.13, -0.17), (0.04, 0.02), 0.6, 0.92, (200, 40, 40))       # an 8 x 4 cm box, 2 cm high, turned 0.6 rad
    got = W2.find(_wrist_picture(K, T, [box]), K, T, 0.92)
    assert got is not None and got["n_cands"] == 1
    assert np.allclose(got["xy"], [0.13, -0.17], atol=0.004) and abs(got["yaw"] - 0.6) < 0.06
    assert abs(got["area_m2"] - 0.0032) < 0.0008 and abs(got["half"][0] - 0.04) < 0.006
    assert W2.find(_wrist_picture(K, T, []), K, T, 0.92) is None        # nothing but table


def test_the_fingers_and_known_objects_are_not_taken_for_the_thing_and_the_hint_chooses_among_the_rest():
    R = GR.hand_matrix([0.0, 1.0, 0.0])
    hand = np.array([0.10, -0.20, 1.20])
    K, T = W2.camera(hand, R, 512)
    a = ((0.02, -0.25), (0.03, 0.03), 0.0, 0.95, (30, 30, 30))          # the thing looked for
    b = ((0.22, -0.12), (0.04, 0.03), 0.0, 0.95, (40, 160, 60))         # an object the main camera placed
    img = _wrist_picture(K, T, [a, b])
    known = GR.Boxes.aabb_box((0.18, -0.15, 0.90), (0.26, -0.09, 0.95)).corners()
    got = W2.find(img, K, T, 0.95, known_boxes=list(known))
    assert got["n_cands"] == 1 and np.allclose(got["xy"], [0.02, -0.25], atol=0.005)
    got = W2.find(img, K, T, 0.95)                                       # both in view: the hint decides
    assert got["n_cands"] == 2
    assert np.allclose(W2.find(img, K, T, 0.95, hint_xy=(0.0, -0.24))["xy"], [0.02, -0.25], atol=0.005)
    assert np.allclose(W2.find(img, K, T, 0.95, hint_xy=(0.2, -0.1))["xy"], [0.22, -0.12], atol=0.005)
    # the robot's own fingers, drawn into the picture from its gripper model, are never a thing
    jaw = GR.JawModel()
    fingers = jaw.gripper_boxes_G(jaw.L).placed(GR.G_matrix(R[:, 1]), hand)
    P = fingers.corners()
    img2 = _wrist_picture(K, T, [a])
    m = W2.draw_boxes(list(P), K, T, 512, 1.0)
    assert 0.002 < m.mean() < 0.5                                        # they are in view, and not all of it
    img2[m] = (20, 20, 20)
    got = W2.find(img2, K, T, 0.95, self_boxes=list(P))
    assert got["n_cands"] == 1 and np.allclose(got["xy"], [0.02, -0.25], atol=0.006)


def test_the_look_step_goes_above_the_point_looks_twice_and_leaves_a_percept_for_the_next_step():
    from rowant_lab.pro2.robot.search import LookAgent

    R = GR.hand_matrix([0.0, 1.0, 0.0])
    q = mat_to_quat(R)                                         # wxyz
    quat_xyzw = [q[1], q[2], q[3], q[0]]
    off = (R @ np.asarray(W2.WRIST["t_hand_cam"]))[:2]

    class Tool:
        def __init__(self, finds):
            self.finds, self.asked, self.added, self.t_now = list(finds), [], None, 0

        def wrist_find(self, typ, support_z, hint_xy=None):
            self.asked.append((typ, support_z, None if hint_xy is None else [float(x) for x in hint_xy], list(self.hand)))
            return self.finds.pop(0) if self.finds else None

        def add_found(self, typ, got):
            self.added = (typ, got)
            return {"id": "w1"}

    def run(finds, cap=110):
        tool = Tool(finds)
        ag = LookAgent(tool, "bowl", {"xy": [0.10, -0.20], "support_z": 0.9}, cap)
        hand = np.array([-0.2, 0.0, 1.15])
        for _ in range(cap):
            tool.hand = hand
            a = ag.act({"robot0_eef_pos": hand, "robot0_eef_quat": quat_xyzw})
            hand = hand + 0.017 * np.asarray(a[:3]) / max(1.0, np.linalg.norm(a[:3]))   # a saturated command: 17 mm
            if ag.finished:
                break
        return ag, tool

    first = {"xy": [0.14, -0.17], "n_cands": 1, "area_m2": 0.009, "half": [0.05, 0.05], "yaw": 0.0}
    second = {"xy": [0.145, -0.172], "n_cands": 1, "area_m2": 0.009, "half": [0.05, 0.05], "yaw": 0.0}
    ag, tool = run([first, second])
    assert ag.finish_reason == "found" and tool.added == ("bowl", second) and ag.t < 70
    # the first picture is taken with the CAMERA above the point at 0.27 m, the second above what the first found
    assert np.allclose(np.asarray(tool.asked[0][3])[:2] + off, [0.10, -0.20], atol=0.01)
    assert abs(tool.asked[0][3][2] - 1.17) < 0.01 and abs(tool.asked[1][3][2] - 1.10) < 0.01
    assert np.allclose(np.asarray(tool.asked[1][3])[:2] + off, [0.14, -0.17], atol=0.01) and tool.asked[1][2] == [0.14, -0.17]
    ag, tool = run([first, {**second, "xy": [0.30, -0.30]}])    # the second picture shows something else: the first stands
    assert ag.finish_reason == "found" and tool.added == ("bowl", first)
    ag, tool = run([None])
    assert ag.finish_reason == "not_found" and tool.added is None and len(tool.asked) == 1
    ag, tool = run([first, second], cap=30)                   # no steps for a second picture: the first is used
    assert tool.added == ("bowl", first) or ag.finish_reason in ("found", "not_found")


def test_an_unexplained_target_is_taken_as_seen_only_when_asked_and_only_if_it_can_be_grasped():
    lib = _glib()
    lib.robot = {"finger_joints": {"a": {"qpos": 0.04}, "b": {"qpos": -0.04}}}
    lib.jaw = GR.JawModel()
    lib.records = {}
    box = {"xy": [0.2, 0.1], "half_xy": [0.047, 0.024], "yaw": 0.3, "z_top": 0.94, "height": 0.04, "n_px": 2000}
    parsed = {"target_type": "ramekin", "place_type": "plate"}

    def tool(seen, blobs, rejected=()):
        t = LK.PerceptionTool(lib, None, None, "s", "ep")
        t.percepts = [_gp("pl", "plate", (0.0, 0.0))]
        t.extra, t.seen_target = {"rejected": list(rejected), "blobs": blobs}, seen
        return t

    t = tool(False, [box])                                      # off: the stored model set on the region, as before
    t.on_parsed(parsed)
    assert [(p["type"], p["source"]) for p in t.percepts[1:]] == [("ramekin", "blob")]
    t = tool(True, [box])                                       # on: a box of the region's own size, named the type
    t.on_parsed(parsed)
    p = t.percepts[-1]
    assert p["source"] == "seen" and p["named"] == "ramekin" and p["type"] == "seen_45x22x20" and p["id"] == "f1"
    assert np.allclose(p["bodies"]["main"]["pos"], [0.2, 0.1, 0.92])
    g = GD.ground({**parsed, "relation": None, "landmarks": [], "part": None}, t.percepts, lib, lib.workcell)
    assert g["status"] == "ok" and g["target"] == "f1"
    wide = {**box, "half_xy": [0.09, 0.08]}                     # wider than the open hand: the stored model, as before
    t = tool(True, [wide])
    t.on_parsed(parsed)
    assert [(p["type"], p["source"]) for p in t.percepts[1:]] == [("ramekin", "blob")]
    rej = {"type": "ramekin", "pos": [0.3, 0.3, 0.9], "fit": {"inlier": 0.9, "n_agree": 500}, "source": "mask",
           "bodies": {"main": {"pos": [0.3, 0.3, 0.9], "quat_wxyz": mat_to_quat(np.eye(3))}}}
    t = tool(True, [box], [rej])                                # a refused pose of the stored model comes first...
    t.on_parsed(parsed)
    assert t.percepts[-1]["source"].startswith("rejected")
    t = tool("first", [box], [rej])                             # ...unless what is seen is asked for first
    t.on_parsed(parsed)
    assert t.percepts[-1]["source"] == "seen"


def test_the_stored_planner_grasps_a_seen_box_across_its_narrow_side_inside_its_own_height():
    """The piece a thing with no fitting stored model needs: grasp frames planned from the seen region alone."""
    lib = _lib()
    blob = {"xy": [0.10, -0.20], "half_xy": [0.045, 0.022], "yaw": 0.5, "z_top": 0.05, "height": 0.05, "n_px": 900}
    p = SN.seen_percept(lib, blob, "s1")
    R_obj = GR.quat_wxyz_to_mat(p["bodies"]["main"]["quat_wxyz"])
    t_obj = np.asarray(p["bodies"]["main"]["pos"], float)
    target = GR.Boxes.from_prims(lib.types[p["type"]]["bodies"]["main"]["primitives"]).placed(R_obj, t_obj)
    # the hand starts turned as a square grasp needs (the planner prefers the frame nearest the hand's own turn)
    frames, diag = GR.plan_frames(lib.records[p["type"]], R_obj, t_obj, target, GR.Boxes.empty(), GR.Boxes.empty(), 0.0,
                                  lib.jaw, 0.5 + 0.5 * np.pi, GR.PlanParams())
    assert frames and diag.get("up_ok", True)
    long_axis = np.array([np.cos(0.5), np.sin(0.5), 0.0])
    for f in frames[:10]:
        assert abs(float(np.asarray(f["axis"]) @ long_axis)) < 0.5          # the fingers close across the 4.4 cm side
        assert 0.043 < f["width"] < 0.052                                    # on the seen width (records every 15 deg)
        assert 0.0 < f["centre"][2] < 0.05 + lib.jaw.face_z[1]               # at the box's own height, above the floor
        assert np.linalg.norm(np.asarray(f["centre"])[:2] - [0.10, -0.20]) < 0.05
    assert abs(frames[0]["width"] - 0.044) < 0.001                           # squarely across it, on its seen width


def test_the_server_tells_a_hanging_region_from_a_standing_one_by_its_columns():
    from rowant_lab.libero_env import m2_perceive as PV

    hpx = np.zeros((40, 40))
    hpx[5:15, 10:30] = 0.2                                     # a panel: rows 5-14, hanging 20 cm up
    v, u = np.nonzero(hpx > 0.01)
    got = PV.Perceiver._hanging(hpx, v, u)
    assert got["hang_share"] == 1.0 and got["hang_px"] == 200 and got["hang_uv"] == [19.5, 9.5]
    hpx[15:25, 10:16] = np.linspace(0.07, 0.02, 10)[:, None]   # a standing thing below it, joined in the picture
    v, u = np.nonzero(hpx > 0.01)
    got = PV.Perceiver._hanging(hpx, v, u)
    assert got["hang_share"] == 0.7 and got["hang_px"] == 200 and got["hang_uv"] == [19.5, 9.5]
    stand = np.zeros((40, 40))
    stand[5:30, 10:20] = np.linspace(0.3, 0.02, 25)[:, None]   # a tall thing standing on the support
    v, u = np.nonzero(stand > 0.01)
    assert PV.Perceiver._hanging(stand, v, u)["hang_share"] == 0.0


def test_a_model_is_stood_the_way_it_rests_before_it_is_set_where_it_was_found():
    lib = _glib()
    # a can that rests with its body's y axis up: 6 cm across, 8 cm high, the box given in the body's frame
    lib.types["can"] = {"rest_up_in_body": [0.0, 1.0, 0.0],
                        "bodies": {"main": {"primitives": [{"centre": [0.0, 0.0, 0.0], "R": np.eye(3).tolist(),
                                                            "half": [0.03, 0.04, 0.03]}]}}}
    lib.export_types["can"] = _tdoc()
    R = BH.rest_R(lib, "can")
    assert np.allclose(R @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) and np.allclose(R @ R.T, np.eye(3))
    assert np.allclose(BH.rest_R(lib, "bowl"), np.eye(3))                 # a type that rests with its z up: as it is
    lib.robot = {"finger_joints": {"a": {"qpos": 0.04}, "b": {"qpos": -0.04}}}
    lib.jaw = SimpleNamespace(gap_from_qpos=lambda a, b: abs(a) + abs(b))
    t = LK.PerceptionTool(lib, None, None, "s", "ep")
    t.percepts = []
    p = t.add_found("can", {"xy": [0.2, 0.1], "half": [0.03, 0.03], "yaw": 0.0, "support_z": 0.9, "n_cells": 50})
    lo, hi = GD.aabb(lib, p)
    assert np.allclose(lo, [0.17, 0.07, 0.90], atol=1e-9) and np.allclose(hi, [0.23, 0.13, 0.98], atol=1e-9)  # standing
    blob = {"xy": [0.2, 0.1], "half_xy": [0.03, 0.03], "yaw": 0.0, "z_top": 0.98, "height": 0.08, "n_px": 900}
    t.extra = {"rejected": [], "blobs": [blob]}
    lying = t._from_blob("can", "target", set())                           # the fallback as it was: body z up
    lo, hi = GD.aabb(lib, lying)
    assert abs(float(hi[2] - lo[2]) - 0.06) < 1e-9
    t.blob_rest_up = True
    lo, hi = GD.aabb(lib, t._from_blob("can", "target", set()))            # asked for: stood as it rests
    assert abs(float(hi[2] - lo[2]) - 0.08) < 1e-9 and abs(float(lo[2]) - 0.90) < 1e-9


# ------------------------------------------------------------------ 2026-11-16 night: what counts as a panel, and a
# lone candidate whose relation cannot be told (new keys of ``behind_occluder``; without them nothing changes)


def test_a_view_that_does_not_match_leaves_a_large_standing_region_and_then_nothing_in_it_is_a_panel():
    panel = {"uv": [400.0, 300.0], "n_px": 45000, "h_min": 0.17, "hang_share": 0.95, "hang_px": 44700,
             "hang_uv": [400.0, 300.0], "xy": [0.4, -0.1]}
    speck = {"uv": [1.0, 1.0], "n_px": 3800, "h_min": 0.01, "hang_share": 0.0, "hang_px": 0}
    table = {"uv": [500.0, 700.0], "n_px": 130000, "h_min": 0.012, "hang_share": 0.015, "hang_px": 100000}
    old = {"uv": [500.0, 700.0], "n_px": 130000, "h_min": 0.012}            # a reply without the field: not counted
    assert BH.standing_px([panel, speck]) == 3800 and BH.standing_px([panel, table]) == 130000
    assert BH.standing_px([panel, old]) == 0 and BH.standing_px([]) == 0 and BH.standing_px(None) == 0
    lib = _glib()
    lib.robot = {"finger_joints": {"a": {"qpos": 0.04}, "b": {"qpos": -0.04}}}
    lib.jaw = SimpleNamespace(gap_from_qpos=lambda a, b: abs(a) + abs(b))
    cam = _camera((1.0, 0.0, 1.6), (0.0, 0.0, 0.9))
    target = np.array([0.12, -0.20, 0.9])
    panel = {**panel, "uv": _pixel(cam, cam["T"][:3, 3] + 0.62 * (target - cam["T"][:3, 3]))}

    def tool(behind, blobs):
        t = LK.PerceptionTool(lib, None, None, "s", "ep")
        t.percepts = [_gp("pl", "plate", (0.0, 0.0))]
        t.extra, t.cams, t.behind = {"rejected": [], "blobs": blobs}, {"agentview": cam}, behind
        return t

    t = tool({}, [panel, table])                                            # the key not given: as before
    assert t.behind_point("bowl") is not None and t.behind_refused is None
    t = tool({"standing_max_px": 20000}, [panel, speck])                   # one thing hidden: a panel
    assert t.behind_point("bowl") is not None and t.behind_refused is None
    t = tool({"standing_max_px": 20000}, [panel, table])                   # the whole view is off: no panel
    assert t.behind_point("bowl") is None and t.behind_refused == {"standing_px": 130000}
    t = tool({"standing_max_px": 20000, "mode": "ray"}, [panel, table])    # and no guess behind it either
    assert t.add_behind("bowl") is None and t.n_behind == 0
    # the least size is a key too: the things that always hang in a picture (a few thousand pixels) are no panels
    small = {**panel, "n_px": 6300, "hang_px": 6300}
    assert tool({}, [small]).behind_point("bowl") is not None
    assert tool({"min_px": 25000}, [small]).behind_point("bowl") is None
    assert tool({"min_px": 25000}, [panel]).behind_point("bowl") is not None


def test_a_region_joined_to_something_standing_counts_by_the_share_asked_for():
    joined = {"uv": [300.0, 520.0], "n_px": 57000, "h_min": 0.02, "hang_share": 0.3, "hang_px": 52000,
              "hang_uv": [258.0, 500.0]}
    assert BH.occluders([joined], min_px=25000) == []                       # the built-in share is one half
    got = BH.occluders([joined], min_px=25000, share=0.25)
    assert len(got) == 1 and got[0]["uv"] == [258.0, 500.0]                 # by the middle of what hangs
    assert BH.occluders([{**joined, "hang_px": 9000}], min_px=25000, share=0.25) == []   # too little of it hangs
    assert BH.occluders([{**joined, "hang_share": 0.015}], min_px=25000, share=0.25) == []  # a standing thing


def test_a_lone_candidate_whose_landmark_is_not_in_view_cannot_be_told():
    scene = [_gp("pl", "plate", (0.0, 0.0)), _gp("b1", "bowl", (0.30, 0.10))]
    on = {"target_type": "bowl", "place_type": "plate", "relation": "on", "landmarks": ["ramekin"], "part": None}
    assert GD.single_untold(on, scene)                                      # the ramekin is not in view
    assert not GD.single_untold(on, scene + [_gp("rm", "ramekin", (0.0, 0.3))])            # it is: the relation decides
    assert not GD.single_untold(on, scene + [_gp("b2", "bowl", (0.0, 0.3))])               # two candidates
    assert not GD.single_untold(on, [scene[0]])                                            # none
    assert not GD.single_untold({**on, "relation": None, "landmarks": []}, scene)          # no relation asked
    assert not GD.single_untold({**on, "relation": "table_centre", "landmarks": ["table"]}, scene)
    between = {**on, "relation": "between", "landmarks": ["plate", "ramekin"]}
    assert GD.single_untold(between, scene)                                 # one of the two landmarks is missing
    named = {**_gp("s1", "seen_30x20x20", (0.2, 0.2)), "named": "bowl"}    # a thing taken as seen counts as its name
    assert GD.single_untold(on, [scene[0], named])


def test_with_the_landmark_hidden_too_the_robot_looks_behind_and_takes_what_it_finds_there(monkeypatch):
    from rowant_lab.pro2.robot import episode as EPI

    eye = mat_to_quat(np.eye(3))

    def thing(pid, typ, pos, n_agree=500, source="texture"):
        return {"id": pid, "type": typ, "fit": {"n_agree": n_agree}, "source": source,
                "bodies": {"main": {"pos": list(pos), "quat_wxyz": eye}}}

    tdocs = {t: {"kind": "object", "root_body": "main", "sites": {}} for t in ("bowl", "plate", "cookies")}

    def scene_from(percepts):
        return {"instances": {p["id"]: {"type": p["type"], "kind": "object", "bodies": {"main": f"{p['id']}_main"},
                                        "root_body_name": f"{p['id']}_main"} for p in percepts},
                "sites": {}, "arena": [], "robot_mount": None}

    lib = SimpleNamespace(types={t: 1 for t in tdocs}, export_types=tdocs, workcell={"arena": []},
                          scene_from_percepts=scene_from, quat_mat=lambda q: np.eye(3))
    parsed = {"status": "ok", "target_type": "bowl", "place_type": "plate", "relation": "next_to",
              "landmarks": ["cookies"], "part": None, "predicate": "on", "place_part": None}
    monkeypatch.setattr(EPI, "parse", lambda inst, types: dict(parsed))
    monkeypatch.setattr(EPI, "make_params", lambda m1: (None, None, None))
    monkeypatch.setattr(EPI, "OracleAgent", lambda lib, scene, spec, tool, cap, ap, pp, fp: SimpleNamespace(spec=spec))
    looked = []
    monkeypatch.setattr(EPI, "_look_then", lambda inst, tool, lib, cap, params, exclude, share, parsed, point, **kw:
                        looked.append((point, kw)) or {"agent": "LOOK", "parsed": parsed, "grounding": {"status": "ok"}})

    class Tool:
        def __init__(self, percepts, point):
            self.percepts, self.target, self.behind, self._point = percepts, None, None, point

        add_behind = None

        def behind_point(self, typ):
            return self._point

    other, plate = thing("p0", "bowl", (0.3, 0.1)), thing("pl", "plate", (0.0, 0.2))
    point = {"xy": [0.1, -0.2], "support_z": 0.9}

    def plan(percepts, behind, point=point):
        looked.clear()
        tool = Tool(percepts, point)
        out = EPI._plan_one("pick the bowl next to the cookies and place it on the plate", tool, lib, 220,
                            {"behind_occluder": behind} if behind is not None else {})
        return out, tool

    out, _ = plan([other, plate], {})                                       # the key not given: the one in view, as before
    assert out["agent"] != "LOOK" and out["agent"].spec["target"] == "p0" and not looked
    out, _ = plan([other, plate], {"untold": True})                         # asked for: a look behind comes first
    assert out["agent"] == "LOOK" and looked[0][0] == point and looked[0][1] == {"why": "untold"}
    out, _ = plan([other, plate], {"untold": True}, point=None)             # nothing hangs there: as before
    assert out["agent"].spec["target"] == "p0" and not looked
    out, _ = plan([other, plate, thing("ck", "cookies", (0.5, 0.0))], {"untold": True})   # the landmark is in view:
    assert not looked and out["agent"].spec["target"] == "p0"               # the relation's own business
    # after the look (mode "done"): what the wrist camera found behind it is the one meant
    found = thing("w1", "bowl", (0.12, -0.21), n_agree=0, source="wrist")
    out, _ = plan([other, plate, found], {"untold": True, "mode": "done", "why": "untold"})
    assert out["agent"].spec["target"] == "w1" and "behind_untold" in out["grounding"]["flags"]
    out, _ = plan([other, plate], {"untold": True, "mode": "done", "why": "untold"})       # it found nothing
    assert out["agent"].spec["target"] == "p0" and "behind_untold" not in (out["grounding"].get("flags") or [])
    out, _ = plan([other, plate, found], {"mode": "done"})                  # a look taken for another reason: by the
    assert out["agent"].spec["target"] == "p0"                              # fit, as before


# ------------------------------------------------------------------ 2026-11-16 night: the cover push of a rim-held
# object (``cover_skim_rim_m``, ``cover_touch_descent``; off by default)


def test_a_rim_held_object_passes_higher_over_a_receiver_that_can_be_moved_and_not_over_a_fixture():
    # the held box rests on a 4 cm thing whose top is at 0.14; the receiver's rim is at 0.10; the table's top is z 0
    p_rel = np.array([0.0, 0.0, 0.07])
    table = [((-1.0, -1.0, -0.05), (1.0, 1.0, 0.0))]
    a, R = _agent(hand=(0.0, 0.0, 0.14 + 0.03), held_lo=0.03, others=table)
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04)
    assert abs(rec["drop"] - 0.034) < 1e-9 and "skim" not in rec            # as before: 6 mm above the rim
    a, R = _agent(hand=(0.0, 0.0, 0.14 + 0.03), held_lo=0.03, others=table, cover_skim_rim_m=0.02)
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04)
    assert abs(rec["drop"] - 0.034) < 1e-9                                  # the option alone: not a rim grasp
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04, rim=True)
    assert plan is not None and abs(rec["drop"] - 0.020) < 1e-9 and rec["skim"] == 0.02   # 2 cm above the rim
    assert abs(plan["start"][2] - (0.17 - 0.020)) < 1e-9
    a, R = _agent(hand=(0.0, 0.0, 0.14 + 0.03), held_lo=0.03, others=table, cover_skim_rim_m=0.02, kind="fixture")
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04, rim=True)
    assert abs(rec["drop"] - 0.034) < 1e-9 and rec["skim"] == 0.006         # a fixture: as before
    # a thin thing on a movable receiver: at least 1 cm of it is still taken (the touch is trusted over the look)
    a, R = _agent(hand=(0.0, 0.0, 0.12 + 0.03), held_lo=0.03, others=table, cover_skim_rim_m=0.02)
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04, rim=True)
    assert plan is not None and abs(rec["drop"] - 0.010) < 1e-9


def test_the_descent_notes_where_the_fingers_gap_opened():
    def run(zs, gaps, gap_held=0.005):
        a = SimpleNamespace(ap=AG.AgentParams(clear_cover=True, cover_touch_descent=True))
        i = {"k": 0}
        a._gap = lambda: gaps[i["k"]]
        a._pose = lambda: (np.array([0.0, 0.0, zs[i["k"]]]), np.eye(3))

        def gen():
            for _ in range(len(zs) - 1):
                yield "act"
            return "reached"

        out = {}
        w = AG.OracleAgent._watch_open(a, gen(), gap_held, out)
        try:
            while True:
                next(w)
                i["k"] += 1          # the step is taken: the next reading
        except StopIteration as e:
            return e.value, out

    # the gap opens at the fourth reading: between 0.97 and 0.96
    st, out = run([1.00, 0.99, 0.97, 0.96, 0.95], [0.005, 0.005, 0.0055, 0.008, 0.010])
    assert st == "reached" and out == {"z_open": 0.96, "z_prev": 0.97}
    st, out = run([1.00, 0.99, 0.98], [0.005, 0.0052, 0.0061])                 # never by 2 mm: nothing noted
    assert st == "reached" and out == {}
    st, out = run([1.00, 0.99, 0.98], [0.009, 0.010, 0.010])                   # open from the start: no height before
    assert out == {"z_open": 0.99, "z_prev": None}


def test_a_target_whose_height_was_never_seen_is_raised_to_where_the_hand_stopped_on_it():
    lib = _glib()
    lib.robot = {"finger_joints": {"a": {"qpos": 0.04}, "b": {"qpos": -0.04}}}
    lib.jaw = SimpleNamespace(gap_from_qpos=lambda a, b: abs(a) + abs(b))

    def tool(source):
        t = LK.PerceptionTool(lib, None, None, "s", "ep")
        t.percepts = [_gp("pl", "plate", (0.0, 0.0)), {**_gp("w1", "bowl", (0.2, 0.1)), "source": source, "gid": None}]
        t.target = "w1"
        return t

    t = tool("wrist")
    assert t.raise_target(0.043)
    w = next(p for p in t.percepts if p["id"] == "w1")
    assert abs(w["bodies"]["main"]["pos"][2] - 0.943) < 1e-9 and w["bodies"]["main"]["pos"][:2] == [0.2, 0.1]
    assert w["raised_m"] == 0.043 and w["source"] == "wrist" and t.fallbacks[-1]["raised_m"] == 0.043
    assert next(p for p in t.percepts if p["id"] == "pl")["bodies"]["main"]["pos"][2] == 0.9      # nothing else moves
    assert t.raise_target(0.01) and next(p for p in t.percepts if p["id"] == "w1")["raised_m"] == 0.053
    assert tool("behind").raise_target(0.03)
    assert not tool("texture").raise_target(0.03)        # a measured pose is not raised by a stopped hand
    assert not tool("blob").raise_target(0.03)
    assert not tool("wrist").raise_target(0.0) and not tool("wrist").raise_target(0.3)
    assert AG.AgentParams().stall_unseen_m == 0.0        # off unless asked for


def test_a_thing_that_cannot_slide_is_pushed_slowly_across_a_receiver_that_can_be_moved():
    # the held box rests on a 4 cm thing whose top is at 0.14; the receiver's rim is at 0.10; the table's top is z 0
    p_rel = np.array([0.0, 0.0, 0.07])
    table = [((-1.0, -1.0, -0.05), (1.0, 1.0, 0.0))]
    hand = (0.0, 0.0, 0.14 + 0.03)
    a, R = _agent(hand=hand, held_lo=0.03, others=table)
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04, rim=True)
    assert "slow" not in plan and "slow_m" not in rec                       # as before without the option
    a, R = _agent(hand=hand, held_lo=0.03, others=table, cover_slow_cap=0.5)
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04)
    assert "slow" not in plan                                               # the option alone: not a rim grasp
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04, rim=True)
    d = np.array(rec["dir"])
    face_s = float(plan["slow"][:2] @ d) + rec["lead"]
    face1 = float(plan["end"][:2] @ d) + rec["lead"]
    # slowly from 1 cm before the receiver's near edge (its half depth along the push is 7 cm), at the push's height
    assert abs(face_s + 0.08) < 1e-6 and plan["slow"][2] == plan["start"][2]
    assert abs(rec["slow_m"] - (face1 + 0.08)) < 1e-4
    seen = []

    def goto(p, R_, grip, cap, tol, tol_deg, max_steps=None):
        seen.append((np.asarray(p, float).copy(), cap, max_steps))
        a._pose = lambda p=np.asarray(p, float): (p, R)                     # the move arrives
        a.t += 3
        yield [0.0] * 6 + [1.0]
        return "reached"

    def watch(gen, gap):
        return (yield from gen)

    a._goto, a._watch_carry = goto, watch
    g = AG.OracleAgent._clear_cover(a, plan, rec, R, 0.04)
    try:
        while True:
            next(g)
    except StopIteration as e:
        how = e.value
    assert how == "swept" and len(seen) == 5                                # to the side, down, up to it, across, back
    assert np.allclose(seen[2][0], plan["slow"]) and seen[2][1] == a.ap.cover_cap
    assert np.allclose(seen[3][0], plan["end"]) and seen[3][1] == 0.5
    # the slow part is given the steps its length needs at half speed (1.1 cm a step at cap 1), and 10 more
    assert seen[3][2] == int(np.ceil(rec["slow_m"] / (0.011 * 0.5))) + 10
    a, R = _agent(hand=hand, held_lo=0.03, others=table, cover_slow_cap=0.5, kind="fixture")
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04, rim=True)
    assert "slow" not in plan                                               # a fixture does not come along
    # ``cover_under_rim_m``: how deep under the thing's top a rim-held object goes over such a receiver
    a, R = _agent(hand=hand, held_lo=0.03, others=table, cover_under_rim_m=0.012)
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04)
    assert abs(rec["drop"] - 0.034) < 1e-9
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04, rim=True)
    assert plan is not None and abs(rec["drop"] - 0.012) < 1e-9
    a, R = _agent(hand=hand, held_lo=0.03, others=table, cover_under_rim_m=0.012, kind="fixture")
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04, rim=True)
    assert abs(rec["drop"] - 0.034) < 1e-9


def test_the_server_reports_what_stands_on_a_reported_object_and_not_what_stands_beside_it():
    from rowant_lab.libero_env import m2_perceive as PV

    # a camera 1 m above the table looking straight down; a plate (a 7 cm disc, 2 cm high) at the origin, a bowl
    # (5 cm disc, 6 cm high) beside it, and a 3 x 6 cm block turned 30 degrees lying on the plate, its top 6.5 cm up
    n, f = 300, 600.0
    K = np.array([[f, 0.0, n / 2], [0.0, f, n / 2], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [0.0, 0.0, 1.0]
    yy, xx = np.mgrid[0:n, 0:n]
    z = np.full((n, n), 1.0)

    def world_xy(depth):                                        # the world point across of each pixel at that depth
        return (xx + 0.5 - n / 2) / f * depth, -(yy + 0.5 - n / 2) / f * depth

    def disc(c, r, top):
        x, y = world_xy(1.0 - top)
        z[(x - c[0]) ** 2 + (y - c[1]) ** 2 <= r * r] = 1.0 - top
        a = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        pts = np.array([[c[0] + q * r * np.cos(b), c[1] + q * r * np.sin(b), top]
                        for q in np.sqrt(np.linspace(0.0, 1.0, 25)) for b in a])
        return (pts - T[:3, 3]) @ T[:3, :3]                     # the model's top as camera-frame points

    nodes = {"plate": disc((0.0, 0.0), 0.07, 0.02), "bowl": disc((0.13, 0.0), 0.05, 0.06)}
    c_blk, yaw = np.array([0.01, -0.005]), np.radians(30.0)
    x, y = world_xy(1.0 - 0.065)
    ax, ay = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    s, t_ = (x - c_blk[0]) * ax[0] + (y - c_blk[1]) * ax[1], (x - c_blk[0]) * ay[0] + (y - c_blk[1]) * ay[1]
    blk = (np.abs(s) <= 0.03) & (np.abs(t_) <= 0.015)
    me = SimpleNamespace(p={"extra_blob_min_px": 200, "extra_blob_top_min_px": 30, "facing_cos": 0.0},
                         _visible_nodes=lambda wc, g, R, t: nodes[g])
    me._silhouette = lambda *a, **k: PV.Perceiver._silhouette(me, *a, **k)
    kept = [{"gid": "plate", "R": np.eye(3), "t": np.zeros(3)}, {"gid": "bowl", "R": np.eye(3), "t": np.zeros(3)}]
    percepts = [{"id": "p0", "type": "plate"}, {"id": "p1", "type": "bowl"}]
    md = {"K": K, "T": T}
    assert PV.Perceiver._on_top(me, None, kept, percepts, z, md, 0.012) == {}      # nothing lies on either
    z[blk] = 1.0 - 0.065
    got = PV.Perceiver._on_top(me, None, kept, percepts, z, md, 0.012)
    assert list(got) == ["p0"]                                  # on the plate; the bowl beside it is not "on" it
    on = got["p0"]
    assert np.allclose(on["xy"], c_blk, atol=0.003) and abs(on["z"] - 0.065) < 1e-3 and on["z_obj"] == 0.02
    assert abs(on["half_xy"][0] - 0.03) < 0.003 and abs(on["half_xy"][1] - 0.015) < 0.003
    assert abs(((on["yaw"] - yaw + np.pi / 2) % np.pi) - np.pi / 2) < 0.06 and on["inside"] == 1.0
    assert on["type"] == "plate" and on["n_in"] >= 200


def _looker(on, q7=0.0, shoulder=(-0.6, -0.1, 0.3), reach="reached"):
    """``_agent`` with what ``_cover_look`` reads: a camera on +x, a look that reports ``on``, the wrist's last joint
    at ``q7``, and moves that arrive (or end ``reach``)."""
    hand = (0.0, 0.0, 0.14 + 0.03)
    a, R = _agent(hand=hand, held_lo=0.03, shoulder=np.array(shoulder), cover_look=True)
    T = np.eye(4)
    T[:3, 3] = [0.9, 0.0, 0.7]
    a.spec["place_instance"] = "basket"
    a.oracle = SimpleNamespace(cams={"agentview": {"T": T}}, on_receiver=lambda inst: on)
    a.pobs = {"robot0_joint_pos": [0.0] * 6 + [q7]}
    seen = []

    def goto(p, R_, grip, cap, tol, tol_deg, max_steps=None):
        seen.append((np.asarray(p, float).copy(), np.asarray(R_, float).copy(), tol_deg))
        a._pose = lambda p=np.asarray(p, float), R_=np.asarray(R_, float): (p, R_)
        a.t += 3
        yield [0.0] * 6 + [1.0]
        return reach

    def watch(gen, gap):
        return (yield from gen)

    a._goto, a._watch_carry = goto, watch
    return a, R, np.asarray(hand, float), seen


def _run(gen):
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value


def test_the_robot_looks_at_what_lies_on_the_receiver_and_turns_to_push_it_across_its_long_side():
    long_y = {"xy": [0.0, 0.0], "half_xy": [0.033, 0.0235], "yaw": np.pi / 2, "z": 0.14, "n_px": 900, "inside": 1.0}
    a, R, hand, seen = _looker(long_y)
    rec = {}
    R_push = _run(AG.OracleAgent._cover_look(a, {}, hand, R, None, None, 0.04, rec))
    # the hand went to the side across the camera's line (the camera is on +x: along y), the side nearer the robot
    # (-y), 3 cm clear of the receiver (7 cm that way) with the held box (4 cm across), and 3 cm up
    assert np.allclose(seen[0][0], [0.0, -(0.07 + 0.04 + 0.03), hand[2] + 0.03])
    # the thing is long along y: the push goes across it, along x; the closing axis was along y, so a quarter turn
    assert rec["look"]["on"]["half_xy"] == [0.033, 0.0235] and abs(abs(rec["look"]["turn_deg"]) - 90.0) < 1e-6
    assert abs(abs(R_push[0, 1]) - 1.0) < 1e-9 and abs(R_push[1, 1]) < 1e-9 and np.allclose(R_push[:, 2], R[:, 2])
    assert len(seen) == 2 and np.allclose(seen[1][1], R_push) and seen[1][2] == 6.0     # turned where it looked
    # long along x: the closing axis (y) is across it already; as long as wide, seen off the receiver, nothing seen,
    # or the hand did not get out of the camera's way: no turn
    for on in ({**long_y, "yaw": 0.0}, {**long_y, "half_xy": [0.025, 0.0235]}, {**long_y, "inside": 0.3}, None):
        a, R, hand, seen = _looker(on)
        assert _run(AG.OracleAgent._cover_look(a, {}, hand, R, None, None, 0.04, {})) is None and len(seen) == 1
    a, R, hand, seen = _looker(long_y, reach="unreachable")
    rec = {}
    assert _run(AG.OracleAgent._cover_look(a, {}, hand, R, None, None, 0.04, rec)) is None and "on" not in rec["look"]
    # 20 degrees off across: within ``cover_turn_min_deg``, no turn; 40 degrees off: a turn of 40 degrees
    a, R, hand, seen = _looker({**long_y, "yaw": np.radians(20.0)})
    rec = {}
    assert _run(AG.OracleAgent._cover_look(a, {}, hand, R, None, None, 0.04, rec)) is None
    assert abs(abs(rec["look"]["off_deg"]) - 20.0) < 0.1
    a, R, hand, seen = _looker({**long_y, "yaw": np.radians(40.0)})
    rec = {}
    R_push = _run(AG.OracleAgent._cover_look(a, {}, hand, R, None, None, 0.04, rec))
    assert abs(abs(rec["look"]["turn_deg"]) - 40.0) < 0.1
    ax = R_push[:2, 1] / np.linalg.norm(R_push[:2, 1])
    assert abs(abs(ax @ np.array([-np.sin(np.radians(40.0)), np.cos(np.radians(40.0))])) - 1.0) < 1e-6
    # the wrist's last joint near its end: the quarter turn the other way round, or none when neither way has room
    a, R, hand, seen = _looker(long_y, q7=2.0)
    rec = {}
    R_push = _run(AG.OracleAgent._cover_look(a, {}, hand, R, None, None, 0.04, rec))
    assert R_push is not None and abs(2.0 - np.radians(rec["look"]["turn_deg"])) <= 2.6
    a, R, hand, seen = _looker({**long_y, "yaw": np.radians(40.0)}, q7=2.4)
    rec = {}
    R_a = _run(AG.OracleAgent._cover_look(a, {}, hand, R, None, None, 0.04, rec))
    assert (R_a is None and rec["look"].get("turn") == "no room") or abs(2.4 - np.radians(rec["look"]["turn_deg"])) <= 2.6


def test_a_push_planned_after_the_look_is_planned_from_where_the_hand_stopped():
    # the hand has gone aside and turned a quarter turn: the plan is made for the stop above the release point
    p_rel = np.array([0.0, 0.0, 0.07])
    table = [((-1.0, -1.0, -0.05), (1.0, 1.0, 0.0))]
    a, R = _agent(hand=(0.0, -0.3, 0.2), held_lo=0.03, others=table)          # where the hand is now: not above
    assert AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04, rim=True) == (None, None)
    R_x = GR.hand_matrix([1.0, 0.0, 0.0])
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R_x, None, None, 0.04, rim=True, at=(0.0, 0.0, 0.17))
    assert plan is not None and abs(abs(rec["dir"][0]) - 1.0) < 1e-6 and rec["top"] == 0.14
    assert abs(plan["up"][2] - 0.20) < 1e-9 and np.allclose(plan["back"][:2], p_rel[:2])


def test_a_push_planned_for_a_seen_thing_runs_through_its_middle():
    p_rel = np.array([0.0, 0.0, 0.07])
    table = [((-1.0, -1.0, -0.05), (1.0, 1.0, 0.0))]
    a, R = _agent(hand=(0.0, 0.0, 0.17), held_lo=0.03, others=table)
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04, rim=True)
    assert abs(plan["start"][0]) < 1e-9 and abs(plan["end"][0]) < 1e-9          # through where the hand stopped
    # the thing's middle was seen 2.5 cm to one side (across the push, which runs along y): the held box's own
    # middle (under the hand here) passes over it
    plan, rec = AG.OracleAgent._cover_plan(a, {}, p_rel, R, None, None, 0.04, rim=True, centre=(0.025, 0.01))
    assert abs(plan["start"][0] - 0.025) < 1e-9 and abs(plan["end"][0] - 0.025) < 1e-9
    assert np.allclose(plan["back"][:2], p_rel[:2])                             # and back over the release point
