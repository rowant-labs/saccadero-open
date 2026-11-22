"""The long-task suite's compound instructions as steps (robot/sequence.py; off unless params["sequence"])."""

from pathlib import Path
from types import SimpleNamespace

from rowant_lab.pro2.robot import sequence as SQ

LONG = {
    "put both the alphabet soup and the tomato sauce in the basket":
        ["put the alphabet soup in the basket", "put the tomato sauce in the basket"],
    "put both the cream cheese box and the butter in the basket":
        ["put the cream cheese box in the basket", "put the butter in the basket"],
    "turn on the stove and put the moka pot on it": ["turn on the stove", "put the moka pot on the stove"],
    "put the black bowl in the bottom drawer of the cabinet and close it":
        ["put the black bowl in the bottom drawer of the cabinet", "close the bottom drawer of the cabinet"],
    "put the white mug on the left plate and put the yellow and white mug on the right plate":
        ["put the white mug on the left plate", "put the yellow and white mug on the right plate"],
    "pick up the book and place it in the back compartment of the caddy":
        ["pick up the book and place it in the back compartment of the caddy"],
    "put the white mug on the plate and put the chocolate pudding to the right of the plate":
        ["put the white mug on the plate", "put the chocolate pudding to the right of the plate"],
    "put both the alphabet soup and the cream cheese box in the basket":
        ["put the alphabet soup in the basket", "put the cream cheese box in the basket"],
    "put both moka pots on the stove": ["put the moka pot on the stove", "put the moka pot on the stove"],
    "put the yellow and white mug in the microwave and close it":
        ["put the yellow and white mug in the microwave", "close the microwave"],
}


def test_the_long_suites_instructions_split_into_steps():
    for inst, want in LONG.items():
        assert SQ.split(inst) == want, inst


def test_single_step_forms_stay_whole():
    for inst in ("pick up the black bowl between the plate and the ramekin and place it on the plate",
                 "put the bowl on the plate", "Pick up the alphabet soup and place it in the basket.",
                 "please pick the akita black bowl next to the ramekin and place it on the plate",
                 "grab the bowl next to the ramekin and then put it on the plate",
                 "take the milk then drop it in the basket", "open the middle layer of the drawer",
                 "put the wine bottle on the top of the drawer", "push the plate to the front of the stove"):
        assert len(SQ.split(inst)) == 1, (inst, SQ.split(inst))


def test_open_then_put_inside_names_the_opened_drawer():
    assert SQ.split("open the top drawer and put the bowl inside") == ["open the top drawer",
                                                                       "put the bowl in the top drawer"]
    assert SQ.split("Please open the top layer of the drawer and put the bowl inside.") == [
        "open the top layer of the drawer", "put the bowl in the top layer of the drawer"]
    assert SQ.split("open the top drawer of the brown cabinet, then put the bowl in it") == [
        "open the top drawer of the brown cabinet", "put the bowl in the top drawer of the brown cabinet"]
    assert SQ.split("please put both the alphabet soup and the tomato sauce in the basket") == [
        "put the alphabet soup in the basket", "put the tomato sauce in the basket"]


def test_put_in_a_named_drawer_parses_the_part():
    from rowant_lab.pro2.robot import episode as EPI
    from rowant_lab.pro2.robot import goal as GO

    def part(name):
        return {"joint": {"type": "slide", "range": [-0.16, 0.01], "qpos0": 0.0}, "bodies": [name]}

    cab = {"root_body": "main", "kind": "fixture",
           "parts": {"main": {"joint": None, "bodies": ["main", "base"]}, "cabinet_top": part("cabinet_top"),
                     "cabinet_middle": part("cabinet_middle"), "cabinet_bottom": part("cabinet_bottom")},
           "sites": {"top_region": {"body": "cabinet_top"}, "middle_region": {"body": "cabinet_middle"},
                     "bottom_region": {"body": "cabinet_bottom"}, "top_side": {"body": "base"}}}
    lib = SimpleNamespace(types={"akita_black_bowl": 1, "wooden_cabinet": 1},
                          export_types={"akita_black_bowl": {"parts": {}}, "wooden_cabinet": cab})
    for inst in ("put the bowl in the top drawer", "put the bowl in the top layer of the drawer",
                 "put the bowl in the top drawer of the brown cabinet"):
        p = GO.parse_put(inst, lib)
        assert p["status"] == "ok" and p["target_type"] == "akita_black_bowl", inst
        assert p["place_type"] == "wooden_cabinet" and p["place_part"] == "top" and p["predicate"] == "in", inst
    on = GO.parse_put("put the bowl on the top of the drawer", lib)  # the fixture's top side, as before
    assert on["predicate"] == "on" and on["place_part"] == "top" and on["place_type"] == "wooden_cabinet"

    def site_name_for(percept, phrase):  # the library's rule: a unique name match, else a unique region
        sites = cab["sites"]
        hit = [s for s in sites if phrase and set(str(phrase).split()) & set(s.split("_"))]
        if len(hit) == 1:
            return hit[0]
        region = [s for s in sites if "region" in s]
        return region[0] if len(region) == 1 else None

    lib.site_name_for = site_name_for
    cabp = {"id": "p1", "type": "wooden_cabinet"}
    assert site_name_for(cabp, "top") is None  # two sites carry "top"
    assert EPI._in_site(lib, cabp, "top") == "top_region" and EPI._in_site(lib, cabp, "bottom") == "bottom_region"
    assert EPI._in_site(lib, cabp, None) is None


class Step:
    def __init__(self, n, reason):
        self.n, self.reason, self.i, self.finished = n, reason, 0, False

    def act(self, obs):
        self.i += 1
        self.finished = self.i >= self.n
        return [0.1] * 6 + [1.0]

    def summary(self):
        return {"steps": self.i, "finish_reason": self.reason}


def test_sequence_agent_runs_steps_in_order_and_excludes_placed_targets():
    calls = []

    def planner(clause, cap, exclude):
        calls.append((clause, cap, exclude))
        if clause.startswith("close"):
            return {"agent": None, "grounding": {"status": "fail", "reason": "no skill"}}
        return {"agent": Step(5, "placed"), "grounding": {"target": f"p{len(calls)}"}}

    ag = SQ.SequenceAgent(["put the a in the b", "put the c in the b", "close the b"], planner, cap=520)
    acts = [ag.act({}) for _ in range(12)]
    assert ag.finished and ag.finish_reason == "steps done" and ag.t == 10
    assert [c[0] for c in calls] == ["put the a in the b", "put the c in the b", "close the b"]
    assert calls[0][1] == 520 - 2 * SQ.RESERVE_STEPS and calls[0][2] == ()
    assert calls[1][2] == ("p1",) and calls[2][2] == ("p1", "p2")  # a later step never re-picks a placed object
    assert acts[-1] == SQ.NOOP and ag.records[2]["skipped"] == "no plan"
    assert ag.summary()["sequence"][0]["summary"] == {"steps": 5, "finish_reason": "placed"}


def test_a_failed_step_is_not_marked_placed_and_budget_is_respected():
    seen = []

    def planner(clause, cap, exclude):
        seen.append((cap, exclude))
        return {"agent": Step(30, "max_attempts"), "grounding": {"target": "p0"}}

    ag = SQ.SequenceAgent(["s1", "s2"], planner, cap=200)
    for _ in range(80):
        ag.act({})
    assert seen[0] == (200 - SQ.RESERVE_STEPS, ()) and seen[1] == (170, ())  # step 2 gets what is left
    ag2 = SQ.SequenceAgent(["s1", "s2"], planner, cap=100)  # 100 - 110 < MIN_STEP: step 1 skipped
    assert ag2.records[0]["skipped"] == "no budget" and ag2.cur is not None


def test_a_side_word_picks_one_of_several_places():
    import numpy as np

    from rowant_lab.pro2.robot import ground as GR
    from rowant_lab.pro2.robot import goal as GO

    doc = {"root_body": "main", "parts": {"main": {"joint": None, "bodies": ["main"]}}, "sites": {}}
    lib = SimpleNamespace(types={"plate": 1, "mug": 1}, export_types={"plate": doc, "mug": doc})
    parsed = GO.parse_put("put the mug on the left plate", lib)
    assert parsed["status"] == "ok" and parsed["place_type"] == "plate" and parsed["place_side"] == "left"
    assert "place_side" not in GO.parse_put("put the mug on the plate", lib)

    def pc(i, typ, xy):
        return {"id": i, "type": typ, "bodies": {"main": {"pos": [xy[0], xy[1], 0.9]}}, "fit": {"n_agree": 5}}

    percepts = [pc("m", "mug", (0.0, 0.0)), pc("a", "plate", (0.1, 0.2)), pc("b", "plate", (0.1, -0.2))]
    g = GR.ground(parsed, percepts, lib, {})
    assert g["place"] == "a" and "place_side_left" in g["flags"]  # +y is the robot's left
    g = GR.ground({**parsed, "place_side": "right"}, percepts, lib, {})
    assert g["place"] == "b"
    assert np.allclose(GR.SIDE_DIR["left"], [0.0, 1.0])


def test_a_part_word_in_the_place_phrase_reaches_the_site():
    from rowant_lab.pro2.robot import ground as GR
    from rowant_lab.pro2.robot import goal as GO

    lib = SimpleNamespace(types={"black_bowl": 1, "wooden_cabinet": 1, "book": 1, "desk_caddy": 1},
                          export_types={})
    p = GO.parse_put("put the black bowl in the bottom drawer of the cabinet", lib)
    assert p["status"] == "ok" and p["place_type"] == "wooden_cabinet" and p["place_part"] == "bottom"
    assert p["predicate"] == "in"
    q = GR.parse("pick up the book and place it in the back compartment of the caddy", sorted(lib.types))
    assert q["status"] == "ok" and q["place_type"] == "desk_caddy" and q["place_part"] == "back"
    r = GR.parse("pick up the book and place it in the caddy", sorted(lib.types))
    assert r["status"] == "ok" and "place_part" not in r
    assert GO.parse_put("put the black bowl on top of the cabinet", lib)["place_part"] == "top"


def test_a_second_object_is_released_where_the_region_is_free():
    import numpy as np

    from rowant_lab.pro2.robot import agent as AG
    from rowant_lab.pro2.robot import grasp as GRS

    region = {"centre": np.array([0.0, 0.0, 0.9]), "half": np.array([0.10, 0.06, 0.02])}

    def make(free):
        ag = object.__new__(AG.OracleAgent)
        ag.ap = AG.AgentParams(place_free_xy=free)
        ag.spec = {"place_instance": "basket"}
        ag.target = "t"
        ag._place_goal = lambda poses: (region["centre"][:2].copy(), "basket", region)
        ag._pose = lambda: (np.array([0.3, 0.0, 1.1]), np.eye(3))
        ag._held_boxes = lambda R_rel, t_rel, p, R: GRS.Boxes([p - np.array([0, 0, 0.1])], [np.eye(3)], [[0.02, 0.02, 0.03]])
        # an object already sits at the region's centre: a release there stays high; elsewhere it goes lower
        ag._world = lambda poses, exclude=(): GRS.Boxes([[0.0, 0.0, 0.92]], [np.eye(3)], [[0.03, 0.03, 0.03]])
        ag._instance_boxes = lambda poses, inst: GRS.Boxes([[0.0, 0.0, 0.9]], [np.eye(3)], [[0.10, 0.06, 0.02]])
        calls = []

        def place_at(poses, R_rel, t_rel, gap, goal_xy, cont, region_):
            calls.append(np.asarray(goal_xy, float))
            d = float(np.linalg.norm(goal_xy))
            return {"z_release": 1.0 if d < 0.04 else 0.95 + 0.1 * abs(goal_xy[1]), "hand_xy": goal_xy, "z_lo": 0.95,
                    "blocked_by": "other" if d < 0.04 else None}

        ag._place_at = place_at
        return ag, calls

    ag, calls = make(False)
    out = ag._plan_place({}, np.eye(3), np.zeros(3), 0.08)
    assert len(calls) == 1 and np.allclose(out["hand_xy"], [0.0, 0.0])  # off: the region's centre, as before
    ag, calls = make(True)
    assert ag._region_occupied({}, region)
    out = ag._plan_place({}, np.eye(3), np.zeros(3), 0.08)
    assert out["free_xy"] and out["z_release"] < 1.0 and abs(out["hand_xy"][1]) < 1e-9 and abs(out["hand_xy"][0]) > 0.04
    assert len(calls) == 1 + 25


def test_close_clauses_find_the_drawer_and_only_the_sequencer_reaches_them():
    from rowant_lab.pro2.robot import drawer as DR

    def part(name):
        return {"joint": {"type": "slide", "range": [-0.16, 0.0], "qpos0": 0.0}, "bodies": [name]}

    cab = {"root_body": "base", "parts": {"base": {"joint": None, "bodies": ["base"]}, "top_drawer": part("top_drawer"),
                                          "middle_drawer": part("middle_drawer"),
                                          "bottom_drawer": part("bottom_drawer")}, "sites": {}}
    lib = SimpleNamespace(types={"wooden_cabinet": 1, "black_bowl": 1},
                          export_types={"wooden_cabinet": cab, "black_bowl": {"parts": {}}})
    p = DR.parse_close("close the bottom drawer of the cabinet", lib, ["wooden_cabinet"])
    assert p["status"] == "ok" and p["form"] == "close" and p["part"] == "bottom_drawer"
    assert p["target_type"] == "wooden_cabinet" and p["predicate"] == "close"
    assert DR.parse_close("put the bowl in the drawer", lib)["status"] == "fail"
    assert DR.parse_open("open the top drawer of the cabinet", lib, ["wooden_cabinet"])["part"] == "top_drawer"
    tool = SimpleNamespace(percepts=[])
    assert DR.plan_close("close the bottom drawer of the cabinet", tool, lib, 300, {})["grounding"]["reason"] \
        == "no percept of the fixture"


def test_beside_a_landmark_is_a_table_region_past_its_edge():
    import numpy as np

    from rowant_lab.pro2.robot import episode as EPI
    from rowant_lab.pro2.robot import goal as GO

    lib = SimpleNamespace(types={"plate": 1, "chocolate_pudding": 1}, export_types={})
    p = GO.parse_put("put the chocolate pudding to the right of the plate", lib)
    assert p["status"] == "ok" and p["place_type"] == "plate" and p["place_beside"] == "right"
    assert p["target_type"] == "chocolate_pudding" and p["predicate"] == "on"
    land = {"id": "p1", "type": "plate"}
    lib2 = SimpleNamespace(m1_type=lambda t: None)
    import rowant_lab.pro2.robot.ground as GRD
    old = GRD.aabb
    GRD.aabb = lambda lib_, p_: (np.array([0.0, -0.1, 0.90]), np.array([0.2, 0.1, 0.92]))
    try:
        scene = {"arena": [{"type": "box", "centre": [0.0, 0.0, 0.85], "R": np.eye(3).tolist(), "half": [0.5, 0.6, 0.05]}]}
        bs = EPI._beside_site(lib2, scene, land, "right")
        assert np.allclose(bs["pos"][:2], [0.1, -(0.1 + EPI.BESIDE_CLEAR_M)]) and abs(bs["pos"][2] - 0.91) < 1e-9
        assert bs["frame_body"] == "world" and EPI._beside_site(lib2, {"arena": []}, land, "left") is None
    finally:
        GRD.aabb = old


def test_the_robot_chooses_the_table_whose_types_name_the_instruction():
    from rowant_lab.pro2.robot import episode as EPI

    cells = {"k": ["flat_stove", "moka_pot", "akita_black_bowl", "wooden_cabinet", "microwave", "white_yellow_mug"],
             "l": ["basket", "alphabet_soup", "tomato_sauce", "cream_cheese", "butter", "plate", "porcelain_mug",
                   "white_yellow_mug", "chocolate_pudding"],
             "s": ["black_book", "desk_caddy"]}
    assert EPI.choose_cell("put both moka pots on the stove", cells) == ("k", "ok")
    assert EPI.choose_cell("put both the alphabet soup and the tomato sauce in the basket", cells) == ("l", "ok")
    assert EPI.choose_cell("pick up the book and place it in the back compartment of the caddy", cells) == ("s", "ok")
    assert EPI.choose_cell("put the black bowl in the bottom drawer of the cabinet and close it", cells)[0] == "k"
    assert EPI.choose_cell("wave hello", cells) == (None, "no cell names the instruction's objects")


def test_a_two_colour_name_matches_its_type():
    from rowant_lab.pro2.robot import goal as GO

    lib = SimpleNamespace(types={"white_yellow_mug": 1, "porcelain_mug": 1, "plate": 1}, export_types={})
    p = GO.parse_put("put the yellow and white mug on the right plate", lib)
    assert p["status"] == "ok" and p["target_type"] == "white_yellow_mug" and p["place_side"] == "right"


def _goal_scene():
    """The Goal suite's real library with the cabinet posed as on the benchmark's table (the top drawer's handle at the
    measured point, opening along +y), or None when the library tree is not on this machine."""
    import math

    import numpy as np

    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.libero_env.m2_kin import mat_to_quat, part_body_poses
    from rowant_lab.pro2 import long_paths as LP
    from rowant_lab.pro2 import machine as MC
    from rowant_lab.pro2.robot import drawer as DR

    if not (Path(MC.LONG_LIB) / "m2" / "policy-v3" / "manifest.json").is_file():
        return None
    _ob, policy, m1 = LB.lfb_dirs(MC.LONG_LIB)
    lib = LB.M2Library("libero_goal", lib_dir=policy, m1_library=m1, extra_suites=LP.extra_suites(["libero_goal"]))

    def percept(pid, typ, R, t):
        tdoc = lib.export_types[typ]
        bp = part_body_poses(tdoc, {tdoc["root_body"]: (np.asarray(R, float), np.asarray(t, float))}, {})
        return {"id": pid, "type": typ, "gid": None, "joints": {}, "fit": {"n_agree": 10, "inlier": 1.0},
                "bodies": {b: {"pos": [float(x) for x in tt], "quat_wxyz": mat_to_quat(RR)} for b, (RR, tt) in bp.items()}}

    want = np.array([0.0307, -0.1301, 1.0932])
    for k in range(4):
        c, s = math.cos(k * math.pi / 2), math.sin(k * math.pi / 2)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        hp = DR.handle_plan(lib, percept("p9", "wooden_cabinet", R, np.zeros(3)), "cabinet_top")
        if hp is not None and hp["d"][1] > 0.9:
            return lib, percept, percept("p9", "wooden_cabinet", R, want - hp["c"])
    return None


def test_an_object_under_the_opening_drawer_is_staged_on_top_and_taken_in():
    import numpy as np
    import pytest

    from rowant_lab.pro2.robot import drawer as DR
    from rowant_lab.pro2.robot import episode as EPI

    scene = _goal_scene()
    if scene is None:
        pytest.skip("the long-lib tree is not on this machine")
    lib, percept, cab = scene
    lo, hi = DR.open_footprint(lib, cab, "cabinet_top")  # the simulator measured x -0.091..0.128, y -0.167..0.037, z 1.054..1.122
    assert np.allclose(lo, [-0.091, -0.167, 1.054], atol=0.01) and np.allclose(hi, [0.128, 0.037, 1.122], atol=0.01)
    steps = SQ.split("open the top layer of the drawer and put the bowl inside")
    bowl = percept("p0", "akita_black_bowl", np.eye(3), [-0.085, 0.004, 0.898])  # where the benchmark puts it
    tool = SimpleNamespace(percepts=[bowl, cab], target=None)
    out = EPI._overhang_rewrite(steps, tool, lib, clear=False)
    assert [x.get("kind") if isinstance(x, dict) else x for x in out] == ["stage", steps[0], "take"]
    away = SimpleNamespace(percepts=[percept("p0", "akita_black_bowl", np.eye(3), [-0.30, 0.20, 0.898]), cab], target=None)
    assert EPI._overhang_rewrite(steps, away, lib) == steps  # nothing in the way: the plain two steps
    assert EPI._overhang_rewrite(["put the bowl on the plate", "put the bowl on the stove"], tool, lib) == [
        "put the bowl on the plate", "put the bowl on the stove"]
    memo = {}
    st = EPI._plan_special(out[0], tool, lib, 150, {}, memo)
    assert st["agent"].spec["grasp_side"] == [0.0, 1.0] or np.allclose(st["agent"].spec["grasp_side"], [0, 1], atol=0.02)
    c_top, e_top = EPI._site_world(lib, cab, "top_side")
    front = c_top[1] + e_top[1]
    sx, sy = st["grounding"]["stage"]
    assert st["agent"] is not None and abs(sx - 0.0307) < 0.005  # on the drawer's middle line
    assert abs((sy + 0.054 + EPI.STAGE_BACK_M) - front) < 0.004  # the bowl's front edge just behind the top's
    on_top = percept("p0", "akita_black_bowl", np.eye(3), [sx, sy, float(c_top[2])])
    # the staging agent let go with its hand at the bowl's front rim, 4.5 cm above the top; it had gripped the bowl
    # 2.5 cm above the table: the bowl fell 2 cm, and the hand goes that far down to take it again
    lo_z = memo["stage"]["lo_z"]
    memo["stage"]["agent"].released = {"p": [sx, sy + 0.045, float(c_top[2]) + 0.045], "R": np.eye(3).tolist(),
                                       "frame_z": lo_z + 0.025, "t": 120}
    memo["stage"]["agent"].attempts.append({"place": {"surface": float(c_top[2])}})
    tk = EPI._plan_special(out[2], SimpleNamespace(percepts=[cab], target="p0"), lib, 80, {}, memo)  # the bowl unseen
    assert type(tk["agent"]).__name__ == "RetakeAgent" and tk["grounding"]["target"] == "p0"
    assert abs(tk["agent"].hp["drop"] - 0.02) < 1e-6 and tk["agent"].hp["p"][:2] == [sx, sy + 0.045]
    c_reg, _ = EPI._site_world(lib, cab, out[2]["site"], poses=DR.open_poses(lib, cab, "cabinet_top"))
    mv = tk["agent"].hp["move"]  # the bowl's middle goes just past the opened drawer's middle
    assert abs(mv[0]) < 0.01 and abs((sy + mv[1]) - (c_reg[1] + EPI.TAKE_PAST_M)) < 1e-4 and 0.10 < mv[1] < 0.18
    # no release pose left by the staging (a staging step that never ran): the push from behind
    sw = EPI._plan_special(out[2], SimpleNamespace(percepts=[on_top, cab], target="p0"), lib, 60, {}, {})
    hp = sw["agent"].hp
    assert type(sw["agent"]).__name__ == "SweepAgent" and np.allclose(hp["d"], [0, 1, 0], atol=0.02)
    assert hp["start"][1] < sy - 0.054 and hp["start"][2] > c_top[2] + 0.02  # from behind the bowl
    assert abs(hp["start"][1] + hp["dist"] - (front + EPI.SWEEP_PAST_M)) < 1e-6  # the hand ends past the top's edge
    assert np.allclose(hp["obj"], [sx, sy]) and 0.03 < hp["width"] and hp["safe_z"] > c_top[2] + 0.05
    assert out[2]["requires"] == 1  # only after the opening step did its work
    assert st["agent"].ap.trust_place and not EPI.make_params({})[0].trust_place  # only the staging step trusts


def test_clauses_that_share_a_place_take_turns_in_it():
    import numpy as np

    from rowant_lab.pro2.robot import agent as AG
    from rowant_lab.pro2.robot import episode as EPI
    from rowant_lab.pro2.robot import grasp as GRS

    lib = SimpleNamespace(types={"moka_pot": 1, "flat_stove": 1, "basket": 1, "alphabet_soup": 1, "tomato_sauce": 1,
                                 "plate": 1, "porcelain_mug": 1, "white_yellow_mug": 1, "akita_black_bowl": 1},
                          export_types={})
    out = EPI._shared_places(SQ.split("put both moka pots on the stove"), lib)
    assert [x["share"] for x in out] == [[0, 2], [1, 2]] and all(x["kind"] == "clause" for x in out)
    out = EPI._shared_places(SQ.split("put both the alphabet soup and the tomato sauce in the basket"), lib)
    assert [x["share"] for x in out] == [[0, 2], [1, 2]] and out[1]["clause"] == "put the tomato sauce in the basket"
    two = SQ.split("put the white mug on the left plate and put the yellow and white mug on the right plate")
    assert EPI._shared_places(two, lib) == two  # each names its own plate
    one = ["put the alphabet soup in the basket", "close the basket"]
    assert EPI._shared_places(one, lib) == one

    ag = object.__new__(AG.OracleAgent)
    ag.ap = AG.AgentParams()
    ag.target = "t"
    ag._instance_boxes = lambda poses, inst: GRS.Boxes([[0.0, 0.017, 1.0]], [np.eye(3)], [[0.0405, 0.074, 0.076]])
    half = np.array([0.075, 0.075, 0.003])  # the region; the object is 8.1 cm by 14.8 cm
    ag.spec = {"predicate": "on"}
    a0, a1 = ag._share_offset({}, half, 0, 2), ag._share_offset({}, half, 1, 2)
    assert np.allclose(a0, [0.05, 0.05]) and np.allclose(a1, [-0.05, -0.05])  # opposite corners, 2.5 cm inside the edge
    assert float(np.linalg.norm(a0 - a1)) > 0.0405 + 0.091  # one's handle (9.1 cm out) cannot reach the other's body
    assert np.allclose(ag._share_offset({}, half, 1, 3), [0.0, 0.0])  # three: the middle one at the centre
    ag.spec = {"predicate": "in"}  # in a container the whole object stays inside the walls
    assert np.allclose(ag._share_offset({}, np.array([0.12, 0.10, 0.05]), 0, 2), [0.12 - 0.0405 - 0.005, 0.10 - 0.074 - 0.005])


def test_a_step_that_needs_another_is_skipped_when_that_one_did_not_finish():
    def run(open_reason):
        log = []

        class A:
            def __init__(self, name, reason):
                self.name, self.reason, self.finished = name, reason, False

            def act(self, _obs):
                self.finished = True
                return list(SQ.NOOP)

            def summary(self):
                return {"steps": 1, "finish_reason": self.reason, "attempts": [], "events": []}

        def planner(step, _cap, _placed):
            kind = step["kind"] if isinstance(step, dict) else "open"
            log.append(kind)
            return {"agent": A(kind, {"stage": "placed", "open": open_reason, "sweep": "swept"}[kind]),
                    "grounding": {"status": "ok", "target": "p0"}}

        steps = [{"kind": "stage", "need": 5, "clause": "stage"}, "open the drawer",
                 {"kind": "sweep", "need": 5, "min": 1, "requires": 1, "clause": "sweep"}]
        ag = SQ.SequenceAgent(steps, planner, cap=300, reserve=5, min_step=1)
        for _ in range(6):
            ag.act(None)
        return log, ag.records

    log, recs = run("opened")
    assert log == ["stage", "open", "sweep"] and recs[2]["summary"]["finish_reason"] == "swept"
    log, recs = run("no_budget")  # the drawer did not open: nothing is pushed off the top
    assert log == ["stage", "open"] and recs[2]["skipped"] == "the step it needs did not finish"


def test_a_blocked_come_down_does_not_send_the_crossing_to_its_maximum():
    import numpy as np

    from rowant_lab.pro2.robot import agent as AG
    from rowant_lab.pro2.robot import grasp as GRS

    def agent(**kw):
        ag = object.__new__(AG.OracleAgent)
        ag.ap = AG.AgentParams(**kw)
        wall = GRS.Boxes([[0.30, 0.06, 0.5]], [np.eye(3)], [[0.05, 0.02, 0.5]])  # beside the goal, up to z 1.0
        ag._world = lambda poses, exclude=(): wall
        ag.supports = GRS.Boxes.empty()
        ag._swept_boxes = lambda pt, R, half_gap, held: GRS.Boxes([pt], [np.eye(3)], [[0.02, 0.05, 0.02]])
        return ag

    R = GRS.hand_matrix([0.0, 1.0, 0.0])
    p0, p1 = np.array([0.0, 0.0, 1.10]), np.array([0.30, 0.0, 0.80])  # the hand at the goal touches the wall
    old = agent()._cross_z({}, p0, R, p1, R, 0.04, None, (), 0.0)
    new = agent(cross_descent_free=True)._cross_z({}, p0, R, p1, R, 0.04, None, (), 0.0)
    assert abs(old - (1.10 + AG.AgentParams().cross_max_m)) < 1e-9  # as registered: all the way up
    assert abs(new - 1.10) < 1e-9  # the crossing is clear where the hand is: no rise for a come-down that stays blocked
    far = np.array([0.30, -0.20, 0.80])  # a goal clear of the wall: both agree
    assert agent()._cross_z({}, p0, R, far, R, 0.04, None, (), 0.0) == agent(cross_descent_free=True)._cross_z(
        {}, p0, R, far, R, 0.04, None, (), 0.0) == 1.10


def test_a_move_carries_its_gripper_commands():
    from rowant_lab.pro2.robot import drawer as DR

    ag = object.__new__(DR.OpenAgent)

    def goto(p, R, grip, cap, tol_m, tol_deg, max_steps=None):
        for _ in range(2):
            yield [0.0] * 6 + [grip]
        return "reached"

    ag._goto = goto
    ag._act_to = lambda p, R, g, cap=None: [0.0] * 6 + [float(g)]

    def drive(gs):
        gen = ag._goto_seq(None, None, gs, 0.0, 1.0, 0.01, 5.0, 10)
        out = []
        try:
            while True:
                out.append(next(gen)[6])
        except StopIteration as e:
            return out, e.value

    assert drive([-1.0, 1.0, 1.0, 1.0]) == ([-1.0, 1.0, 1.0, 1.0], "reached")  # all sent, though the move took two steps
    assert drive([1.0]) == ([1.0, 0.0], "reached")  # then the resting command
    assert drive([]) == ([0.0, 0.0], "reached")


def test_frames_on_the_asked_side_come_first_in_the_planners_order():
    import numpy as np

    from rowant_lab.pro2.robot import agent as AG

    t = np.array([-0.09, 0.0, 0.9])
    fr = [{"n": 0, "centre": t + [0.045, 0.015, 0.03]},   # the +x rim, a little to +y (2026-11-15: taken as "+y side")
          {"n": 1, "centre": t + [0.0, -0.047, 0.03]},    # the -y rim
          {"n": 2, "centre": t + [0.01, 0.046, 0.03]},    # the +y rim
          {"n": 3, "centre": t + [-0.02, 0.043, 0.03]}]   # the +y rim, a little to -x
    out, sizes = AG.side_order(fr, t, [0.0, 1.0])
    assert [f["n"] for f in out] == [2, 3, 1, 0] and sizes == [2, 1, 1]  # asked side, opposite side, flank
    skew = [{"n": 0, "centre": t + [0.030, -0.036, 0.03]},  # the opposite side, 40 degrees off the line
            {"n": 1, "centre": t + [0.004, -0.047, 0.03]}]  # the opposite side, on the line: first, whatever the order
    assert [f["n"] for f in AG.side_order(skew, t, [0.0, 1.0])[0]] == [1, 0]
    out, sizes = AG.side_order(fr, t, [0.0, -2.0])  # any length; the other side
    assert [f["n"] for f in out] == [1, 2, 3, 0] and sizes == [1, 2, 1]


def test_subset_runs_take_options_and_pair_by_episode():
    from rowant_lab.pro2 import subset as SB

    assert SB._pairs("relook_place=true,relook_min_m=0.02,name=x") == {"relook_place": True, "relook_min_m": 0.02, "name": "x"}
    eps = [{"suite": "libero_goal", "task": 0, "case": "D02"}, {"suite": "libero_goal", "task": 1, "case": "BASE"},
           {"suite": "libero_object", "task": 0, "case": "D02"}]
    assert len(SB.select(eps, {"D02"})) == 2 and len(SB.select(eps, {"D02"}, {"libero_goal"})) == 1 and len(SB.select(eps)) == 3

    def row(case, task, ok):
        return (("s", task, case, 8, 1), {"success": ok})

    ref = dict([row("D02", 0, False), row("D02", 1, True), row("BASE", 0, True), row("S01", 0, True)])
    new = dict([row("D02", 0, True), row("D02", 1, True), row("BASE", 0, False)])
    by = SB.compare(ref, new)
    assert by["D02"] == {"n": 2, "ref": 1, "new": 2, "won": 1, "lost": 0}
    assert by["ALL"] == {"n": 3, "ref": 2, "new": 2, "won": 1, "lost": 1} and "S01" not in by


def test_an_open_topped_object_under_the_opening_drawer_is_dragged_clear_first():
    import numpy as np
    import pytest

    from rowant_lab.pro2.robot import drawer as DR
    from rowant_lab.pro2.robot import episode as EPI

    scene = _goal_scene()
    if scene is None:
        pytest.skip("the long-lib tree is not on this machine")
    lib, percept, cab = scene
    steps = SQ.split("open the top layer of the drawer and put the bowl inside")
    bowl = percept("p0", "akita_black_bowl", np.eye(3), [-0.085, 0.004, 0.898])
    floor, rim = DR.cavity(lib, bowl)
    assert 0.005 < floor - 0.898 < 0.03 and 0.03 < rim - floor < 0.08  # a bowl: open at the top
    assert DR.cavity(lib, percept("p5", "cream_cheese", np.eye(3), [0.2, 0.2, 0.9])) is None  # a box is not
    tool = SimpleNamespace(percepts=[bowl, cab], target=None)
    out = EPI._overhang_rewrite(steps, tool, lib)
    assert [x.get("kind") if isinstance(x, dict) else x for x in out] == ["clear", steps[0], steps[1]]
    memo = {}
    cl = EPI._plan_special(out[0], tool, lib, 80, {}, memo)
    hp = cl["agent"].hp
    fp = DR.open_footprint(lib, cab, "cabinet_top")
    from rowant_lab.pro2.robot.ground import aabb

    lo_b, hi_b = aabb(lib, bowl)
    assert float(hi_b[0] - lo_b[0]) > 0.08  # the bowl's width
    end = np.array([-0.085, 0.004]) + np.asarray(hp["move"])
    gap = max(fp[0][0] - end[0], end[0] - fp[1][0], fp[0][1] - end[1], end[1] - fp[1][1])  # from the footprint
    assert type(cl["agent"]).__name__ == "DragAgent" and memo["moved"] == ["p0"]
    assert abs(gap - EPI.CLEAR_M) < 1e-6 and 0.03 < float(np.linalg.norm(hp["move"])) < 0.12
    assert abs(hp["p_in"][2] - (floor + 0.02)) < 1e-9 and hp["safe_z"] > rim + 0.03
    # a neighbour standing in the shortest way out: another way is taken, clear of it
    free = DR.clear_move(lib, tool.percepts, bowl, fp, EPI.CLEAR_M)
    block = percept("p7", "wine_bottle", np.eye(3), list(np.array([-0.085, 0.004]) + 2.3 * np.asarray(free["move"])) + [0.9])
    other = DR.clear_move(lib, [bowl, cab, block], bowl, fp, EPI.CLEAR_M)
    assert other is not None and other["dist"] >= free["dist"] - 1e-9
    assert float(np.asarray(other["move"]) @ np.asarray(free["move"])) < 0.99 * free["dist"] * other["dist"]
    # walled in on every side: no way out, and the plan falls back to staging
    ring = [percept(f"w{k}", "wine_bottle", np.eye(3), [-0.085 + 0.13 * np.cos(a), 0.004 + 0.13 * np.sin(a), 0.9])
            for k, a in enumerate(np.linspace(0, 2 * np.pi, 16, endpoint=False))]
    walled = SimpleNamespace(percepts=[bowl, cab] + ring, target=None)
    assert DR.clear_move(lib, walled.percepts, bowl, fp, EPI.CLEAR_M) is None
    assert [x.get("kind") if isinstance(x, dict) else x for x in EPI._overhang_rewrite(steps, walled, lib)] == [
        "stage", steps[0], "take"]


def test_a_part_the_robot_opened_is_held_open_in_the_scene():
    import numpy as np
    import pytest

    from rowant_lab.pro2.robot import drawer as DR
    from rowant_lab.pro2.robot import look as LK

    scene = _goal_scene()
    if scene is None:
        pytest.skip("the long-lib tree is not on this machine")
    lib, percept, cab = scene
    t = object.__new__(LK.PerceptionTool)
    t.lib, t.percepts, t.opened = lib, [dict(cab)], []
    body = lib.export_types["wooden_cabinet"]["parts"]["cabinet_top"]["bodies"][0]
    shut = np.asarray(t.percepts[0]["bodies"][body]["pos"], float)
    t.believe_open("p9", "cabinet_top")
    now = np.asarray(t.percepts[0]["bodies"][body]["pos"], float)
    far = DR.open_far(lib.export_types["wooden_cabinet"], "cabinet_top")
    assert abs(float(np.linalg.norm(now - shut)) - abs(far)) < 1e-6 and now[1] > shut[1] + 0.15  # out along +y
    assert t.percepts[0]["joints"]["cabinet_top"] == far and cab["bodies"][body]["pos"] == list(shut)  # a copy
    t.percepts = [dict(cab)]  # a later look read it shut again
    t._hold_open()
    assert np.allclose(t.percepts[0]["bodies"][body]["pos"], now)
