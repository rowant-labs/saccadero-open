"""COVER stream (2026-11-17): a thing lying on a receiver that can be moved is taken off by hand (``cover_pick``). Off
by default; these tests run no simulator and no perception server."""
from types import SimpleNamespace

import numpy as np

from rowant_lab.pro2.robot import agent as AG
from rowant_lab.pro2.robot import grasp as GR

BLOCK = {"xy": [0.005, 0.0], "half_xy": [0.0335, 0.0235], "yaw": np.pi / 2, "z": 0.054, "n_px": 6500, "inside": 1.0}


def _picker(on=BLOCK, others=(), kind="object", plate_half=0.07, cap=220, t=70, gaps=(0.08, 0.06, 0.047), q7=0.0,
            shoulder=(-0.55, 0.0, 0.3), reach="reached", **opts):
    """A stand-in with what ``_cover_pick`` reads: a plate at the origin on a table whose top is z 0 (the plate's rim
    at z 0.019), a held bowl 10.8 cm across whose lowest point is 3 cm below the hand, the hand's closing axis along
    world y, a camera on +x, a look that reports ``on``, and moves that arrive (or end ``reach``)."""
    R_hand = GR.hand_matrix([0.0, 1.0, 0.0])
    hand = np.array([0.0, 0.0, 0.054 + 0.03])  # the bowl rests on the thing's top (5.4 cm)
    a = SimpleNamespace(ap=AG.AgentParams(clear_cover=True, cover_pick=True, tell_open=True, release_here=True, **opts),
                        pp=GR.PlanParams(), fp=AG.FollowParams(), cap=cap, t=t, target="tgt", jaw=GR.JawModel(),
                        spec={"predicate": "on", "place_instance": "plate"}, n_cleared=0,
                        shoulder=np.array(shoulder), scene={"instances": {"plate": {"kind": kind}}},
                        supports=GR.Boxes.aabb_box((-0.5, -0.6, -0.05), (0.6, 0.6, 0.0)), tried=[("c", "a")],
                        pobs={"robot0_joint_pos": [0.0] * 6 + [q7]})
    pose = {"p": hand.copy(), "R": R_hand}
    a._pose = lambda: (pose["p"], pose["R"])
    plate = GR.Boxes.aabb_box((-plate_half, -plate_half, 0.0), (plate_half, plate_half, 0.019))
    plate.tags = ["plate"]
    a._place_goal = lambda poses: (np.zeros(2), "plate", None)
    a._instance_boxes = lambda poses, inst: plate
    a._held_boxes = lambda R_rel, t_rel, p, R: GR.Boxes.aabb_box((-0.054, -0.054, -0.03), (0.054, 0.054, 0.02),
                                                               origin=p)
    extra = GR.Boxes.empty()
    for lo, hi in others:
        extra = extra + GR.Boxes.aabb_box(lo, hi)
    a._world = lambda poses, exclude=(): (GR.Boxes.empty() if "plate" in exclude else plate) + extra
    T = np.eye(4)
    T[:3, 3] = [0.9, 0.0, 0.7]
    log = []
    a.oracle = SimpleNamespace(cams={"agentview": {"T": T}},
                               on_receiver=lambda inst: (log.append(("look", inst)), on)[1])
    a._let_go = lambda: log.append(("let_go",))
    a._tell_open = lambda lost=False: log.append(("tell_open",) + (("lost",) if lost else ()))
    g = list(gaps)

    def gap():
        return g.pop(0) if len(g) > 1 else g[0]

    def goto(p, R_, grip, cap_, tol, tol_deg, max_steps=None, stall_down=False, stall_after_m=0.0):
        log.append(("goto", np.asarray(p, float).copy(), np.asarray(R_, float).copy(), float(grip)))
        pose["p"], pose["R"] = np.asarray(p, float).copy(), np.asarray(R_, float).copy()
        a.t += 3
        yield [0.0] * 6 + [float(grip)]
        return reach

    def hold(p, R_, grip, n):
        log.append(("hold", float(grip), int(n)))
        a.t += n
        yield [0.0] * 6 + [float(grip)]

    def watch(gen, gap_held):
        return (yield from gen)

    a._gap, a._goto, a._hold, a._watch_carry = gap, goto, hold, watch
    a._with_gap = lambda gen, gap_m: gen  # (the fingers' gap on the way is tested on its own)
    for name in ("_spot_ok", "_support_top", "_reach_ok", "_cover_recv", "_cover_takeable", "_cover_spots",
                 "_round_held", "_cover_retake", "_cover_turn", "_c1"):
        setattr(a, name, (lambda f: lambda *x, **k: f(a, *x, **k))(getattr(AG.OracleAgent, name)))
    return a, R_hand, hand, log


def _run(gen):
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value


def _pick(a, R, rec):
    return _run(AG.OracleAgent._cover_pick(a, {}, np.array([0.0, 0.0, 0.05]), R, None, None, 0.005, rec))


def test_the_option_is_off_by_default():
    assert AG.AgentParams().cover_pick is False


def test_a_thing_the_fingers_can_hold_is_taken_off_and_the_object_waits_beside_the_receiver():
    a, R, hand, log = _picker()
    rec = {}
    assert _pick(a, R, rec) == "parked"
    pick = rec["pick"]
    gotos = [x for x in log if x[0] == "goto"]
    # the object goes aside held, to the free place nearest the shoulder (the robot is on -x, the camera on +x),
    # 3 cm up, its near side 2 cm clear of the plate; the look comes after that move and before the hand opens
    park = np.asarray(pick["park"])
    assert abs(park[1]) < 1e-6 and abs(park[0] + (0.07 + 0.054 + 0.02)) < 1e-6
    assert np.allclose(gotos[0][1], [park[0], park[1], hand[2] + 0.03]) and gotos[0][3] == 1.0
    kinds = [x[0] for x in log]
    assert kinds.index("look") == 1 and kinds.index("let_go") > kinds.index("look")
    # set down: at speed to 1.2 cm above, then on until the bowl's lowest point is 6 mm above the table (the hand 3 cm
    # above that); then the hand opens, for three steps
    assert np.allclose(gotos[1][1], [park[0], park[1], 0.006 + 0.03 + 0.012]) and gotos[1][3] == 1.0
    assert np.allclose(gotos[2][1], [park[0], park[1], 0.006 + 0.03]) and gotos[2][3] == 1.0
    k = kinds.index("let_go")
    assert kinds[k:k + 3] == ["let_go", "tell_open", "hold"] and log[k + 2][1:] == (-1.0, 3)
    assert a.tried == [] and a.n_cleared == 1 and a.n_picked == 1
    # straight up off the bowl's wall, open; then over the thing, turned so the fingers close across its long side
    # (it is long along y: the closing axis goes from y to x, a quarter turn), 3 cm of it between the fingertips
    assert np.allclose(gotos[3][1][:2], park) and gotos[3][3] == -1.0 and gotos[3][1][2] > gotos[2][1][2] + 0.04
    R_g = gotos[4][2]
    assert abs(abs(pick["turn_deg"]) - 90.0) < 1e-6 and abs(abs(R_g[0, 1]) - 1.0) < 1e-9
    assert np.allclose(gotos[4][1][:2], BLOCK["xy"]) and gotos[4][3] == -1.0
    assert abs(gotos[4][1][2] + a.jaw.finger_z[0] - (0.054 + 0.012)) < 1e-9  # the fingertips 1.2 cm above its top
    tip = gotos[6][1][2] + a.jaw.finger_z[0]
    assert abs(gotos[5][1][2] - gotos[6][1][2] - 0.012) < 1e-9  # (at speed to 1.2 cm above the hold, then slowly)
    assert np.allclose(gotos[6][1][:2], BLOCK["xy"]) and abs(tip - (0.054 - 0.03)) < 1e-9 and tip > 0.019
    assert gotos[5][3] == -1.0 and gotos[6][3] == 1.0  # (open at speed, the last 1.2 cm with the fingers closing)
    assert pick["gap"] == 0.047 and pick["took"] is True
    # lifted, carried off the plate to a free place clear of the waiting bowl and not toward the camera, let go
    # low (the thing's underside about 2 cm above the table), the hand opens and rises off it
    drop = np.asarray(pick["drop"])
    assert gotos[7][1][2] > gotos[6][1][2] + 0.025 and gotos[7][3] == 1.0  # (3 cm: off the plate's rim)
    assert np.allclose(gotos[8][1][:2], drop, atol=1e-4) and gotos[8][3] == 1.0 and len(gotos) == 10
    off_plate = float(np.linalg.norm(np.maximum(np.abs(drop) - 0.07, 0.0)))  # from the plate's box
    assert off_plate >= 0.05 - 1e-3 and np.linalg.norm(drop - park) >= 0.054 + 0.0335 + 0.02 - 1e-3
    assert drop[0] < 0.07 + 0.05 * 0.5 and gotos[8][1][2] < gotos[6][1][2] + 0.03
    assert log[-2] == ("hold", -1.0, 3) and gotos[9][3] == -1.0 and gotos[9][1][2] > gotos[8][1][2] + 0.02
    # of the free places, the one with the shortest way from the thing and on to the waiting bowl
    way = np.linalg.norm(drop - np.asarray(BLOCK["xy"])) + np.linalg.norm(drop - park)
    assert way < 0.37


def test_a_thing_too_wide_too_flat_or_not_on_the_receiver_is_left_to_the_push():
    for on, why in (({**BLOCK, "half_xy": [0.073, 0.073]}, "too wide"), ({**BLOCK, "z": 0.019 + 0.015}, "too flat"),
                    ({**BLOCK, "inside": 0.2}, "not on the receiver"), ({**BLOCK, "n_px": 50}, "not on the receiver"),
                    (None, None)):
        a, R, hand, log = _picker(on=on)
        rec = {}
        assert _pick(a, R, rec) == "back"
        assert rec["pick"].get("refused") == why and a.n_cleared == 0 and a.tried == [("c", "a")]
        gotos = [x for x in log if x[0] == "goto"]
        assert len(gotos) == 2 and all(x[3] == 1.0 for x in gotos) and not any(x[0] == "let_go" for x in log)
        assert np.allclose(gotos[1][1], [0.0, 0.0, hand[2] + 0.03])  # back over the release point, as high as it left
    # what the look took for the thing is the held object itself, beside the receiver
    a, R, hand, log = _picker()
    rec = {}
    _pick(a, R, rec)
    a, R, hand, log = _picker(on={**BLOCK, "xy": [rec["pick"]["park"][0], rec["pick"]["park"][1] * 0.9]})
    rec = {}
    assert _pick(a, R, rec) == "back" and rec["pick"]["refused"] == "the held object"
    # or it lies over an object the robot knows (set on the receiver in an earlier step, not reported by this look)
    a, R, hand, log = _picker(others=[((-0.02, -0.03, 0.0), (0.03, 0.03, 0.06))])
    rec = {}
    assert _pick(a, R, rec) == "back" and rec["pick"]["refused"] == "a known object"


def test_nothing_is_done_over_a_fixture_a_large_receiver_or_with_too_few_steps():
    for kw, why in (({"kind": "fixture"}, "receiver"), ({"plate_half": 0.12}, "receiver size"),
                    ({"t": 220 - 100}, "no steps")):
        a, R, hand, log = _picker(**kw)
        rec = {}
        assert _pick(a, R, rec) is None and rec["pick"]["skipped"] == why and log == [] and a.n_cleared == 0


def test_the_waiting_place_is_free_reachable_and_not_between_the_camera_and_the_receiver():
    # the free place nearest the shoulder: toward the robot here; with something standing there, the next nearest
    a, R, hand, log = _picker(shoulder=(-0.55, -0.1, 0.3))
    rec = {}
    _pick(a, R, rec)
    assert rec["pick"]["park"][0] < -0.1 and abs(rec["pick"]["park"][1]) < 0.09
    a, R, hand, log = _picker(shoulder=(-0.55, -0.1, 0.3), others=[((-0.25, -0.12, 0.0), (-0.09, 0.03, 0.05))])
    rec = {}
    assert _pick(a, R, rec) == "parked"
    p1 = np.asarray(rec["pick"]["park"])
    assert not (-0.25 - 0.069 < p1[0] < -0.09 + 0.069 and -0.12 - 0.069 < p1[1] < 0.03 + 0.069)   # clear of it
    assert p1[0] < 0.07 * 0.5 + 0.08                                                               # not toward the camera
    # nowhere: nothing is done
    ring = [((-0.3, -0.3, 0.0), (0.3, -0.09, 0.05)), ((-0.3, 0.09, 0.0), (0.3, 0.3, 0.05)),
            ((-0.3, -0.3, 0.0), (-0.09, 0.3, 0.05))]
    a, R, hand, log = _picker(others=ring)
    rec = {}
    assert _pick(a, R, rec) is None and rec["pick"]["skipped"] == "no place to wait" and log == []
    # the place itself: on the support, clear of what is known, within reach
    a, R, hand, log = _picker()
    assert a._spot_ok({}, [0.0, 0.2, ], 0.05, 0.05) and not a._spot_ok({}, [0.0, 0.58], 0.05, 0.05)   # off the table
    assert not a._spot_ok({}, [0.0, 0.10], 0.05, 0.05)                                                # on the plate
    assert not a._spot_ok({}, [0.40, 0.0], 0.05, 0.05)                                                # out of reach


def test_a_move_aside_that_does_not_arrive_brings_the_object_back_and_a_close_on_nothing_ends_without_a_carry():
    a, R, hand, log = _picker(reach="unreachable")
    rec = {}
    assert _pick(a, R, rec) == "back" and "on" not in rec["pick"] and not any(x[0] == "look" for x in log)
    a, R, hand, log = _picker(gaps=(0.08, 0.03, 0.004))
    rec = {}
    assert _pick(a, R, rec) == "parked" and rec["pick"]["took"] is False and "drop" not in rec["pick"]
    # (the thing is still there: no second take, but the push stays open to the attempt that follows)
    assert a.n_cleared == 0 and a.n_picked == 1 and log[-1][0] == "goto" and log[-1][3] == -1.0


def test_a_thing_seen_from_afar_ends_the_carry_and_the_object_is_set_down_at_the_nearest_free_place():
    # the carry: the hand comes in along -y toward the point above the release, 2 cm a step, 12 cm above the table
    def carry(a, pose_at):
        def gen():
            for y in np.arange(-0.30, 0.001, 0.02):
                pose_at(np.array([0.0, y, 0.12]))
                a.t += 1
                yield [0.0] * 6 + [1.0]
            return "reached"
        return gen()

    def run(on, **kw):
        a, R, hand, log = _picker(on=on, cover_see=True, t=50, **kw)
        pose = {}
        a._pose = lambda: (pose["p"], R)
        pose["p"] = np.array([0.0, -0.32, 0.12])
        att = {}
        st = _run(AG.OracleAgent._watch_see(a, carry(a, lambda p: pose.__setitem__("p", p)),
                                            np.array([0.0, 0.0, 0.10]), {}, None, None, att))
        return a, R, log, att, st, pose

    a, R, log, att, st, pose = run(BLOCK)
    # the first look comes when the hand is within 19 cm of the middle of the plate's top (10 cm above it: 16 cm
    # across); the thing can be taken: the carry ends there
    assert st == "cover_seen" and [x[0] for x in log] == ["look"] and abs(pose["p"][1] + 0.16) < 1e-9
    assert att["cover_seen"][0]["on"]["half_xy"] == BLOCK["half_xy"] and a._cover_take[1:3] == (0.0335, 0.0235)
    assert abs(att["cover_seen"][0]["d"] - np.hypot(0.16, 0.12 - 0.019)) < 1e-3
    # the object then goes straight to the free place nearest to it (it comes from -y: that side), not aside and
    # back, and no second look is taken
    rec = {}
    goto0 = len(log)
    assert _run(AG.OracleAgent._cover_pick(a, {}, np.array([0.0, 0.0, 0.05]), R, None, None, 0.005, rec,
                                           seen=a._cover_take)) == "parked"
    gotos = [x for x in log[goto0:] if x[0] == "goto"]
    assert rec["pick"]["seen"] is True and rec["pick"]["park"][1] < -0.14 and "on" not in rec["pick"]
    assert np.allclose(gotos[0][1], [rec["pick"]["park"][0], rec["pick"]["park"][1], 0.006 + 0.03 + 0.03])
    assert sum(1 for x in log if x[0] == "look") == 1 and rec["pick"]["took"] is True
    # a thing too wide for the fingers, nothing on the receiver, or the held object taken for it: a second look at
    # 15 cm (10 cm across), and the carry runs on to its end
    for on in ({**BLOCK, "half_xy": [0.073, 0.073]}, None, {**BLOCK, "xy": [0.0, -0.11]}):
        a, R, log, att, st, pose = run(on)
        assert st == "reached" and [x[0] for x in log] == ["look", "look"] and abs(pose["p"][1]) < 1e-9
        assert [round(x["d"], 2) for x in att["cover_seen"]] == [0.19, 0.14]
    assert att["cover_seen"][0]["refused"] == "the held object"
    # a carry that starts beside the receiver: one look for both distances, and none once the held object is over
    # the receiver's middle
    a, R, hand, log = _picker(on=None, cover_see=True, t=50)
    pose = {"p": np.array([0.0, -0.10, 0.03 + 0.06])}
    a._pose = lambda: (pose["p"], R)

    def short(a=a, pose=pose):
        for y in (-0.08, -0.06, -0.04, -0.02, 0.0):
            pose["p"] = np.array([0.0, y, 0.09])
            yield [0.0] * 6 + [1.0]
        return "reached"
    att = {}
    assert _run(AG.OracleAgent._watch_see(a, short(), np.zeros(3), {}, None, None, att)) == "reached"
    assert [x[0] for x in log] == ["look"] and len(att["cover_seen"]) == 1
    # off, too few steps left, a fixture or a step of a longer plan that trusts its place: no look at all
    for kw in ({"cover_see": False}, {"t": 220 - 100}, {"kind": "fixture"}, {"trust_place": True}):
        a, R, hand, log = _picker(**{"cover_see": True, "t": 50, **kw})
        pose = {"p": np.array([0.0, -0.32, 0.12])}
        a._pose = lambda: (pose["p"], R)
        st = _run(AG.OracleAgent._watch_see(a, carry(a, lambda p: pose.__setitem__("p", p)),
                                            np.array([0.0, 0.0, 0.10]), {}, None, None, {}))
        assert st == "reached" and log == []


def test_the_hand_follows_the_things_long_side_and_the_wrist_keeps_inside_its_range():
    # read nearly as wide as long, the long side is followed all the same (a quarter turn here)
    a, R, hand, log = _picker(on={**BLOCK, "half_xy": [0.025, 0.0235]})
    rec = {}
    assert _pick(a, R, rec) == "parked" and abs(abs(rec["pick"]["turn_deg"]) - 90.0) < 1e-6
    # long along x: the closing axis (y) is across it already; 5 degrees off is near enough; 20 degrees is turned
    for yaw, turn in ((0.0, 0.0), (np.radians(5.0), 0.0), (np.radians(20.0), 20.0)):
        a, R, hand, log = _picker(on={**BLOCK, "yaw": yaw})
        rec = {}
        assert _pick(a, R, rec) == "parked" and abs(abs(rec["pick"]["turn_deg"]) - turn) < 0.1
    # the wrist's last joint has room for it: the short way
    a, R, hand, log = _picker(q7=1.0)
    rec = {}
    assert _pick(a, R, rec) == "parked" and abs(abs(rec["pick"]["turn_deg"]) - 90.0) < 1e-6
    # no room for the whole turn: as far as the joint goes when that leaves the fingers within 15 degrees of the
    # short side; else the other way round
    for q7 in (2.0, -2.0):
        a, R, hand, log = _picker(q7=q7)
        rec = {}
        assert _pick(a, R, rec) == "parked" and abs(q7 - np.radians(rec["pick"]["turn_deg"])) <= 2.8 + 1e-9
    short = []
    for sgn in (1.0, -1.0):                                                   # the short way asks 30 degrees
        a, R, hand, log = _picker(on={**BLOCK, "yaw": np.radians(30.0)}, q7=sgn * (2.8 - np.radians(20.0)))
        rec = {}
        assert _pick(a, R, rec) == "parked"
        short.append((rec["pick"].get("turn_short_deg"), abs(rec["pick"]["turn_deg"])))
    # on the side where the joint stands 20 degrees from its end, it turns those 20 and stays 10 short
    assert sorted(short, key=lambda x: x[0] is None) == [(10.0, 20.0), (None, 30.0)]


def test_the_fingers_are_kept_a_gap_apart_on_a_move_of_the_empty_hand():
    gaps = iter([0.08, 0.07, 0.05, 0.03, 0.04, 0.03])
    a = SimpleNamespace(_gap=lambda: next(gaps))

    def gen():
        for _ in range(6):
            yield [0.0] * 6 + [-1.0]
        return "reached"

    out = []
    g = AG.OracleAgent._with_gap(a, gen(), 0.035)
    try:
        while True:
            out.append(next(g)[6])
    except StopIteration as e:
        assert e.value == "reached"
    assert out == [1.0, 1.0, 1.0, -1.0, 1.0, -1.0]      # closing while wider than 3.5 cm, opening while narrower
    assert _run(AG.OracleAgent._with_gap(a, gen(), None)) == "reached"


def test_the_descent_ends_at_the_step_a_rim_held_object_touches():
    # the gap opens by 2 mm at the fourth pose: the move ends there, with that height and the one before
    def step(zs, gaps):
        a = SimpleNamespace(ap=AG.AgentParams(clear_cover=True, cover_pick=True))
        i = {"k": 0}
        a._pose = lambda: (np.array([0.0, 0.0, zs[i["k"]]]), np.eye(3))
        a._gap = lambda: gaps[i["k"]]

        def gen():
            while i["k"] < len(zs) - 1:
                yield [0.0] * 7
            return "reached"

        out, n = {}, 0
        g = AG.OracleAgent._watch_touch(a, gen(), 0.005, out)
        try:
            next(g)
            while True:
                i["k"] += 1  # the step is taken: the next pose and gap
                n += 1
                next(g)
        except StopIteration as e:
            return e.value, out, n

    st, out, n = step([0.10, 0.09, 0.08, 0.07, 0.06, 0.05], [0.005, 0.005, 0.0055, 0.0075, 0.009, 0.010])
    assert st == "touched" and n == 3 and out == {"z_open": 0.07, "z_prev": 0.08}
    # it never opens: the move's own end, nothing noted
    st, out, n = step([0.10, 0.09, 0.08, 0.07], [0.005, 0.0052, 0.0049, 0.0051])
    assert st == "reached" and out == {}
    # open before the first step (a hold that has already slipped): the move runs as it is
    st, out, n = step([0.10, 0.09, 0.08], [0.009, 0.009, 0.009])
    assert st == "reached" and out == {}


def test_the_hand_goes_back_for_the_object_by_the_same_hold_and_sets_it_on_the_receiver():
    def run(gaps, moved=0.0, width=None, **kw):
        a, R, hand, log = _picker(gaps=gaps, cover_retake=True, **kw)
        a.scene["instances"]["plate"]["root_body_name"] = "plate_main"
        where = {"xy": np.zeros(2)}
        a.lib = SimpleNamespace(poses_from_percepts=lambda pcs: {"plate_main": {"xy": where["xy"].copy()}})
        a.oracle.percepts = []
        AG_pose_of = AG.pose_of
        AG.pose_of = lambda poses, body: (np.eye(3), np.r_[poses[body]["xy"], 0.0])

        def relook(inst):
            log.append(("relook", inst))
            where["xy"] = where["xy"] + np.array([moved, 0.0])
            return {"moved_m": moved, "assoc": moved}

        a.oracle.relook_place = relook
        a._idle = lambda: [0.0] * 6 + [-1.0]
        rec = {}
        p_set, p_rel = np.array([0.0, -0.149, 0.036]), np.array([0.0, 0.0, 0.05])
        try:
            got = _run(AG.OracleAgent._cover_retake(a, p_set, R, 0.005, p_rel, "plate", rec, width))
        finally:
            AG.pose_of = AG_pose_of
        return got, rec["retake"], [x for x in log if x[0] == "goto"], log, p_set, p_rel

    # the fingers read as they did at the lift check: taken, lifted over the rim, carried over the release point,
    # the receiver looked at once (it has not moved), down to the planned release, the hand opens
    got, r, gotos, log, p_set, p_rel = run(gaps=(0.08, 0.02, 0.0055))
    assert got == "retaken" and r["held"] is True and r["gap"] == 0.0055
    assert np.allclose(gotos[0][1], p_set + [0, 0, 0.045]) and gotos[0][3] == -1.0     # open, above the object
    # down at speed to 1.5 cm above the hold, the rest with the fingers closing
    assert np.allclose(gotos[1][1], p_set + [0, 0, 0.015]) and gotos[1][3] == -1.0
    assert np.allclose(gotos[2][1], p_set) and gotos[2][3] == 1.0
    assert np.allclose(gotos[3][1], [p_set[0], p_set[1], 0.08]) and gotos[3][3] == 1.0  # 3 cm over the release height
    assert np.allclose(gotos[4][1], [0.0, 0.0, 0.08])
    # down at speed to 1.2 cm above the release, then to it
    assert np.allclose(gotos[5][1], p_rel + [0, 0, 0.012]) and np.allclose(gotos[6][1], p_rel) and gotos[6][3] == 1.0
    kinds = [x[0] for x in log]
    assert kinds.count("relook") == 1
    assert ("hold", -1.0, 8) in log and gotos[-1][3] == -1.0 and len(gotos) == 8
    # the receiver was moved 4 cm when the thing came off: the release moves across with it
    got, r, gotos, log, p_set, p_rel = run(gaps=(0.08, 0.02, 0.0055), moved=0.04)
    assert got == "retaken" and np.allclose(gotos[5][1], [0.04, 0.0, 0.08]) and np.allclose(gotos[7][1], [0.04, 0.0, 0.05])
    # another gap than at the lift check (closed on nothing, or on something else): the hand opens and rises, and
    # the next attempt takes the object
    for gaps in ((0.08, 0.02, 0.0005), (0.08, 0.03, 0.012)):
        got, r, gotos, log, p_set, p_rel = run(gaps=gaps)
        assert got is None and r["held"] is False and len(gotos) == 4 and gotos[-1][3] == -1.0
        assert ("hold", -1.0, 6) in log and not any(x[0] == "relook" for x in log)
    # with the width of the grasp that took it, the close check itself decides: a rim hold reads up to 4 mm over
    # its width at the close (the wall still slanted between the fingers; 3.7 mm after the lift), an empty close
    # under 2.2 mm
    got, r, gotos, log, p_set, p_rel = run(gaps=(0.08, 0.03, 0.0134), width=0.0095)
    assert got == "retaken" and r["held"] is True and r["width"] == 0.0095
    got, r, gotos, log, p_set, p_rel = run(gaps=(0.08, 0.03, 0.0012), width=0.0095)
    assert got is None and r["held"] is False
    # and the look tool is told the hand lets go where it is: its next look searches for the object from there
    got, r, gotos, log, p_set, p_rel = run(gaps=(0.08, 0.02, 0.0055))
    assert ("tell_open", "lost") in log


def test_a_round_object_held_by_its_rim_is_set_down_turned_for_the_thing_and_taken_again_as_it_stands():
    def run(round_model, gaps=(0.08, 0.06) + (0.047,) * 7 + (0.08, 0.02, 0.005)):
        a, R, hand, log = _picker(gaps=gaps, cover_retake=True)
        # the bowl is held by its rim 5 cm from its middle (the hand is at x 0, the bowl's middle at x -0.05)
        a._held_boxes = lambda R_rel, t_rel, p, R_: GR.Boxes.aabb_box((-0.104, -0.054, -0.03), (0.004, 0.054, 0.02),
                                                                    origin=p)
        half = (0.054, 0.054, 0.025) if round_model else (0.054, 0.035, 0.025)
        a.ttype = "bowl"
        a.lib = SimpleNamespace(types={"bowl": {"root_body": "main"}},
                                body_boxes=lambda typ, rel: GR.Boxes.aabb_box([-x for x in half], half),
                                poses_from_percepts=lambda pcs: {"plate_main": 0})
        a.scene["instances"]["plate"]["root_body_name"] = "plate_main"
        a.oracle.percepts = []
        a.oracle.relook_place = lambda inst: None
        a._idle = lambda: [0.0] * 6 + [-1.0]
        rec = {}
        got = _run(AG.OracleAgent._cover_pick(a, {}, np.array([0.0, 0.0, 0.05]), R, None, None, 0.005, rec))
        return got, rec["pick"], [x for x in log if x[0] == "goto"], R

    got, pick, gotos, R = run(True)
    assert got == "retaken" and pick["turned_hold"] is True and pick["retake"]["held"] is True
    park = np.asarray(pick["park"])
    sgn = np.sign(pick["turn_deg"])
    off, off_t = np.array([0.05, 0.0]), np.array([0.0, sgn * 0.05])           # (0.05, 0) turned a quarter turn
    # aside as it is held, for the look; then the hand turns the quarter turn the thing asks while it still holds
    # the bowl, which stays over its waiting place (the hold goes a quarter turn round the bowl's middle)
    assert np.allclose(gotos[0][1][:2], park + off) and np.allclose(gotos[0][2], R) and gotos[0][3] == 1.0
    R_g = gotos[1][2]
    assert abs(abs(pick["turn_deg"]) - 90.0) < 1e-6 and abs(abs(R_g[0, 1]) - 1.0) < 1e-9
    assert np.allclose(gotos[1][1][:2], park + off_t) and gotos[1][3] == 1.0
    assert np.allclose(gotos[2][1][:2], park + off_t) and np.allclose(gotos[2][2], R_g)          # set down so
    p_set = gotos[2][1]
    # the hand never turns again: up, over the thing, down, off with it, and back to the bowl as it let it go
    assert all(np.allclose(x[2], R_g) for x in gotos[3:])
    k = next(i for i, x in enumerate(gotos) if x[3] == -1.0 and i > 9)       # the first open move of the way back
    assert np.allclose(gotos[k][1][:2], p_set[:2]) and gotos[k][1][2] > p_set[2] + 0.04
    lower = [x for x in gotos if x[3] == 1.0][-1]                            # the last move with the object held
    # and the release point goes the same quarter turn round the plate's middle
    assert np.allclose(lower[1], [-0.05 + off_t[0], off_t[1], 0.05], atol=1e-6)
    # not round (a mug with its handle, a box): set down as it is held, the hand turns on its way to the thing
    # and back again on its way to the object
    got, pick, gotos, R = run(False)
    assert got == "retaken" and "turned_hold" not in pick
    assert np.allclose(gotos[1][1][:2], park + off) and np.allclose(gotos[1][2], R)
    k = next(i for i, x in enumerate(gotos) if x[3] == -1.0 and i > 9)
    assert np.allclose(gotos[k][2], R) and np.allclose(gotos[k][1][:2], park + off, atol=1e-6)
    lower = [x for x in gotos if x[3] == 1.0][-1]
    assert np.allclose(lower[1], [0.0, 0.0, 0.05], atol=1e-6) and np.allclose(lower[2], R)


