"""Item 5 (2026-11-14): a fresh full look when the target's track is refused with the hand open (off by default)."""

from types import SimpleNamespace

import numpy as np

from rowant_lab.libero_env.m2_kin import mat_to_quat
from rowant_lab.libero_env.tests.test_close_look import _cam, _lib
from rowant_lab.pro2.robot import agent as AG
from rowant_lab.pro2.robot import look as LK


def _percept(pid, pos, gid="g1"):
    return {"id": pid, "type": "can", "gid": gid, "joints": {},
            "bodies": {"main": {"pos": list(pos), "quat_wxyz": mat_to_quat(np.eye(3))}}}


class Client:
    """Track: ``status``; full look: one fresh percept of the target's type at ``fresh``."""

    def __init__(self, status="predicted", fresh=(0.25, 0.2, 0.9)):
        self.calls, self.status, self.fresh = [], status, fresh

    def call(self, msg):
        self.calls.append(msg)
        if msg.get("mode") == "full":
            return {"percepts": [_percept("q0", self.fresh)], "times": {}}
        out = {pid: {"status": self.status, "R_w": pr["R_w"], "t_w": pr["t_w"], "n_px": 10}
               for pid, pr in msg["priors"].items()}
        return {"tracks": out, "times": {"total_s": 0.1}}


def _tool(client, redetect=True, gripper=(0.04, -0.04)):
    t = LK.PerceptionTool(_lib(), client, _cam, "libero_object", "ep")
    t.obs = {"robot0_joint_pos": np.zeros(7), "robot0_gripper_qpos": np.asarray(gripper, float),
             "robot0_eef_pos": np.array([0.6, 0.6, 1.2]), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0])}
    t.percepts = [_percept("p0", (0.1, 0.2, 0.9))]
    t.target = "p0"
    t.calls = 1  # the first full look has been taken
    t.redetect = redetect
    return t


def _pos(t):
    return t.percepts[0]["bodies"]["main"]["pos"]


def test_off_a_refused_track_keeps_the_prediction():
    cl = Client()
    t = _tool(cl, redetect=False)
    t()
    assert [m["mode"] for m in cl.calls] == ["track"] and np.allclose(_pos(t), [0.1, 0.2, 0.9]) and t.n_redetect == 0


def test_on_a_refused_track_with_the_hand_open_relooks_and_finds_the_moved_target():
    cl = Client()
    t = _tool(cl)
    t()
    assert [m["mode"] for m in cl.calls] == ["track", "full"] and t.n_redetect == 1 and t.n_full == 1
    assert np.allclose(_pos(t), [0.25, 0.2, 0.9]) and t.percepts[0]["id"] == "p0"  # 15 cm away: kept its id
    assert t.looks[-1]["mode"] == "full" and t.looks[-1]["redetect"] and t.looks[-1]["merge"]


def test_on_nothing_changes_when_tracked_holding_or_over_budget():
    cl = Client(status="tracked")
    t = _tool(cl)
    t()
    assert [m["mode"] for m in cl.calls] == ["track"] and t.n_redetect == 0
    cl = Client()
    t = _tool(cl, gripper=(0.0, 0.0))  # the hand is closed (on the target, or on nothing)
    t()
    assert [m["mode"] for m in cl.calls] == ["track"] and t.n_redetect == 0
    cl = Client()
    t = _tool(cl)
    t.n_track = t.budget - 1  # the track takes the last look
    t()
    assert [m["mode"] for m in cl.calls] == ["track"] and t.n_redetect == 0


def test_agent_option_turns_it_on():
    lib = SimpleNamespace(jaw=SimpleNamespace(L=0.08, finger_z=(0.0, 0.0)))
    scene = {"instances": {"t": {"type": "can", "root_body_name": "main"}}, "arena": []}
    for on in (False, True):
        tool = SimpleNamespace(redetect=False)
        AG.OracleAgent(lib, scene, {"target": "t"}, tool, 300, ap=AG.AgentParams(redetect=on))
        assert tool.redetect is on


def test_a_named_types_percept_far_off_the_table_is_dropped_before_the_fallback():
    t = _tool(Client())
    t.lib.workcell = {"arena": [{"type": "box", "centre": [0.0, 0.0, 0.85], "R": np.eye(3).tolist(), "half": [0.5, 0.6, 0.05]}]}
    t.percepts = [_percept("p0", (-1.62, -0.48, 0.87)), _percept("p1", (0.1, 0.2, 0.9))]
    t.extra = {"rejected": [], "blobs": []}
    t.on_parsed({"target_type": "can", "place_type": None})
    assert [p["id"] for p in t.percepts] == ["p0", "p1"]  # off: as before
    t.table_bound = True
    t.on_parsed({"target_type": "can", "place_type": None})
    assert [p["id"] for p in t.percepts] == ["p1"] and t.fallbacks[-1]["dropped_off_table"] == 1


# ---- 2026-11-15: the receiver looked at again before letting go; a fixture never turned around by a later look
def _rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _p(pid, typ, pos, yaw=0.0):
    return {"id": pid, "type": typ, "gid": "g", "joints": {},
            "bodies": {"main": {"pos": list(pos), "quat_wxyz": mat_to_quat(_rz(yaw))}}}


class Scene:
    """Full looks return ``fresh`` (a list of percepts)."""

    def __init__(self, fresh):
        self.calls, self.fresh = [], fresh

    def call(self, msg):
        self.calls.append(msg)
        return {"percepts": [dict(q) for q in self.fresh], "times": {}}


def _scene_tool(fresh, percepts):
    lib = _lib()
    lib.export_types = {"can": {**lib.export_types["can"], "kind": "object"},
                        "tray": {**lib.export_types["can"], "kind": "object"},
                        "cupboard": {**lib.export_types["can"], "kind": "fixture"}}
    t = LK.PerceptionTool(lib, Scene(fresh), _cam, "libero_object", "ep")
    t.obs = {"robot0_joint_pos": np.zeros(7), "robot0_gripper_qpos": np.asarray((0.0, 0.0), float),
             "robot0_eef_pos": np.array([0.6, 0.6, 1.2]), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0])}
    t.percepts = percepts
    t.target = "p0"
    t.calls = 1
    return t


def test_the_receiver_is_measured_again_and_nothing_else_moves():
    held = _p("p0", "can", (0.30, 0.00, 1.05))            # in the hand, above the tray
    fresh = [_p("q0", "can", (0.32, 0.01, 1.04)),          # the held can as this look sees it
             _p("q1", "tray", (0.18, 0.09, 0.90)),         # the tray, 15 cm from where it was
             _p("q2", "can", (0.60, 0.30, 0.90))]          # another can, where it was
    t = _scene_tool(fresh, [held, _p("p1", "tray", (0.30, 0.00, 0.90)), _p("p2", "can", (0.60, 0.305, 0.90))])
    r = t.relook_place("p1")
    assert abs(r["moved_m"] - 0.15) < 1e-3 and r["assoc"] == 0.15 and t.n_full == 1
    assert r["fresh_m"] == [0.15] and r["rejected_m"] == []  # what this look saw of the receiver's type
    by = {p["id"]: p["bodies"]["main"]["pos"] for p in t.percepts}
    assert np.allclose(by["p1"], [0.18, 0.09, 0.90])                       # the receiver where it is now
    assert by["p0"] == [0.30, 0.00, 1.05] and by["p2"] == [0.60, 0.305, 0.90]  # everything else as it was
    assert t.looks[-1]["relook_place"] and set(t.looks[-1]["assoc"]) == {"p1"}


def test_a_receiver_is_not_sent_to_a_far_copy_or_past_the_look_budget():
    fresh = [_p("q1", "tray", (0.60, 0.30, 0.90))]  # 0.42 m off: not the receiver moved, something else
    t = _scene_tool(fresh, [_p("p0", "can", (0.3, 0.0, 1.05)), _p("p1", "tray", (0.30, 0.00, 0.90))])
    r = t.relook_place("p1")
    assert r == {"moved_m": 0.0, "assoc": None, "fresh_m": [0.424], "rejected_m": [], "others": {"n": 0, "seen_m": []}}
    assert t.percepts[1]["bodies"]["main"]["pos"] == [0.30, 0.00, 0.90]
    t.n_full = t.budget
    assert t.relook_place("p1") is None and t.relook_place("nope") is None and t.relook_place(None) is None


def test_a_later_look_does_not_turn_a_fixture_around():
    cup = _p("p1", "cupboard", (0.0, -0.23, 0.9), yaw=0.0)
    fresh = [_p("q1", "cupboard", (0.01, -0.25, 0.9), yaw=180.0), _p("q2", "tray", (0.2, 0.2, 0.9), yaw=170.0)]
    for no_flip, want_yaw_kept in ((False, False), (True, True)):
        t = _scene_tool(fresh, [_p("p0", "can", (0.3, 0.0, 1.05)), dict(cup), _p("p2", "tray", (0.2, 0.21, 0.9))])
        t.no_flip = no_flip
        t.full_look(merge=True)
        by = {p["id"]: p["bodies"]["main"] for p in t.percepts}
        kept = by["p1"]["pos"] == [0.0, -0.23, 0.9]
        assert kept == want_yaw_kept and t.n_flip_kept == int(no_flip)
        assert np.allclose(by["p2"]["pos"], [0.2, 0.2, 0.9])  # an object may turn: it takes the new pose
    t = _scene_tool([_p("q1", "cupboard", (0.05, -0.23, 0.9), yaw=20.0)], [_p("p0", "can", (0.3, 0, 1.05)), dict(cup)])
    t.no_flip = True
    t.full_look(merge=True)  # a fixture that moved a little without turning: the new pose
    assert np.allclose(t.percepts[1]["bodies"]["main"]["pos"], [0.05, -0.23, 0.9]) and t.n_flip_kept == 0


def test_a_later_look_does_not_turn_a_fixture_a_quarter_turn_when_the_limit_is_set():
    """``fixture_turn_max_deg`` (off unless set): with the half-turn rule on, a fit turned beyond the limit is refused
    too (the look taken with the hand over the cabinet read it a quarter turn round); as built only beyond 120."""
    cup = _p("p1", "cupboard", (0.0, -0.23, 0.9), yaw=0.0)
    for limit, yaw, want_kept in ((None, 90.0, False), (45.0, 90.0, True), (45.0, 20.0, False), (45.0, 180.0, True),
                                  (None, 180.0, True)):
        t = _scene_tool([_p("q1", "cupboard", (0.01, -0.25, 0.9), yaw=yaw)], [_p("p0", "can", (0.3, 0.0, 1.05)), dict(cup)])
        t.no_flip = True
        t.turn_max_deg = limit
        t.full_look(merge=True)
        kept = t.percepts[1]["bodies"]["main"]["pos"] == [0.0, -0.23, 0.9]
        assert kept == want_kept, (limit, yaw)
        assert t.n_flip_kept == int(want_kept) and t.turn_kept == ([yaw] if want_kept else [])
        assert t.looks[-1].get("turn_kept") == ([yaw] if want_kept and limit is not None else None)
    t = _scene_tool([_p("q1", "cupboard", (0.01, -0.25, 0.9), yaw=90.0)], [_p("p0", "can", (0.3, 0.0, 1.05)), dict(cup)])
    t.turn_max_deg = 45.0  # the limit alone does nothing: it belongs to the half-turn rule
    t.full_look(merge=True)
    assert t.percepts[1]["bodies"]["main"]["pos"] != [0.0, -0.23, 0.9] and t.n_flip_kept == 0


def test_a_fallback_pose_clear_of_the_other_objects_is_taken_before_one_inside_them():
    def rej(pos, n_agree):
        return {**_p("r", "tray", pos), "pos": list(pos), "fit": {"n_agree": n_agree, "inlier": 0.9}, "source": "support"}

    def run(rejected, table_bound=True):
        t = _scene_tool([], [_p("p0", "can", (0.3, 0.0, 0.9)), _p("p1", "cupboard", (-0.4, -0.15, 0.9))])
        t.table_bound = table_bound
        t.lib.workcell = {}
        t._from_blob = lambda typ, role, used: None  # no unexplained blob in these scenes
        t._inside_other = lambda typ, pos: abs(pos[0] + 0.4) < 0.1 and abs(pos[1] + 0.15) < 0.2  # the cupboard's footprint
        t.extra = {"rejected": rejected, "blobs": []}
        t.on_parsed({"target_type": "tray", "place_type": None})
        got = [p for p in t.percepts if p["type"] == "tray"]
        return got[0]["bodies"]["main"]["pos"] if got else None

    on_it, clear = rej((-0.46, -0.01, 0.91), 900), rej((-0.24, 0.15, 0.88), 300)
    assert run([on_it, clear]) == [-0.24, 0.15, 0.88]  # the weaker fit, but not another object's surface
    assert run([on_it, clear], table_bound=False) == [-0.46, -0.01, 0.91]  # off: the strongest fit, as before
    assert run([on_it]) == [-0.46, -0.01, 0.91]  # nothing else and no blob: still better than no percept
    assert run([]) is None


def test_a_moved_fixture_seen_only_as_a_rejected_fit_is_still_followed():
    def rej(pos, n_agree, yaw=0.0):
        return {**_p("r", "cupboard", pos, yaw=yaw), "pos": list(pos), "fit": {"n_agree": n_agree, "inlier": 0.9},
                "source": "support"}

    class Extra(Scene):
        def __init__(self, rejected):
            super().__init__([])
            self.rejected = rejected

        def call(self, msg):
            self.calls.append(msg)
            return {"percepts": [], "extra": {"rejected": self.rejected}, "times": {}}

    def run(rejected, no_flip=True):
        t = _scene_tool([], [_p("p0", "can", (0.3, 0.0, 1.05)), _p("p1", "cupboard", (0.0, -0.23, 0.9))])
        t.client = Extra(rejected)
        t.no_flip = no_flip
        r = t.relook_place("p1")
        return r, t.percepts[1]["bodies"]["main"]["pos"]

    r, pos = run([rej((0.15, -0.23, 0.9), 400), rej((0.40, -0.10, 0.9), 900)])  # 15 cm off; and a far false copy
    assert r["assoc"] == -1.0 and abs(r["moved_m"] - 0.15) < 1e-6 and pos == [0.15, -0.23, 0.9]
    r, pos = run([rej((0.005, -0.23, 0.9), 400)])  # the same place: nothing to follow
    assert r["assoc"] is None and r["moved_m"] == 0.0 and pos == [0.0, -0.23, 0.9]
    r, pos = run([rej((0.15, -0.23, 0.9), 400, yaw=180.0)])  # its mirror fit: not taken
    assert r["assoc"] is None and pos == [0.0, -0.23, 0.9]


def test_a_move_is_believed_only_with_the_rest_of_the_scene_seen_in_place():
    def run(fresh, extra=()):
        held = _p("p0", "can", (0.30, 0.00, 1.05))
        t = _scene_tool(fresh, [held, _p("p1", "tray", (0.30, 0.00, 0.90)), _p("p2", "can", (0.60, 0.30, 0.90)),
                                *extra])
        r = t.relook_place("p1")
        return r, {p["id"]: p["bodies"]["main"]["pos"] for p in t.percepts}

    tray_moved = _p("q1", "tray", (0.18, 0.09, 0.90))  # 15 cm from where it was
    # the other can where it was: the receiver did move
    r, by = run([tray_moved, _p("q2", "can", (0.60, 0.305, 0.90))])
    assert abs(r["moved_m"] - 0.15) < 1e-3 and "unconfirmed_m" not in r and np.allclose(by["p1"], [0.18, 0.09, 0.90])
    assert r["others"] == {"n": 1, "seen_m": [0.005]}
    # the other can 6 cm from where it was too: the look is off (the camera moved), nothing is followed
    r, by = run([tray_moved, _p("q2", "can", (0.56, 0.345, 0.90))])
    assert r["moved_m"] == 0.0 and abs(r["unconfirmed_m"] - 0.15) < 1e-3 and by["p1"] == [0.30, 0.00, 0.90]
    assert by["p2"] == [0.60, 0.30, 0.90]
    # the other can not seen at all: nothing confirms the look
    r, by = run([tray_moved])
    assert r["moved_m"] == 0.0 and r["others"] == {"n": 1, "seen_m": []} and by["p1"] == [0.30, 0.00, 0.90]
    # one of two others knocked aside, the other in place: the middle one decides (here: both count, median 3 cm)
    r, by = run([tray_moved, _p("q2", "can", (0.60, 0.36, 0.90)), _p("q3", "cupboard", (0.0, 0.5, 0.9))],
                extra=(_p("p3", "cupboard", (0.0, 0.5, 0.9)),))
    assert r["others"] == {"n": 2, "seen_m": [0.0, 0.06]} and r["moved_m"] == 0.0
    # the switch off: believed as before
    t = _scene_tool([tray_moved], [_p("p0", "can", (0.30, 0.00, 1.05)), _p("p1", "tray", (0.30, 0.00, 0.90)),
                                   _p("p2", "can", (0.60, 0.30, 0.90))])
    t.static_check = False
    assert abs(t.relook_place("p1")["moved_m"] - 0.15) < 1e-3


def test_a_fit_of_its_type_where_it_was_means_it_has_not_moved():
    def rej(pos, n_agree):
        return {**_p("r", "cupboard", pos), "pos": list(pos), "fit": {"n_agree": n_agree, "inlier": 0.9},
                "source": "support"}

    class Extra(Scene):
        def call(self, msg):
            self.calls.append(msg)
            return {"percepts": [], "extra": {"rejected": self.fresh}, "times": {}}

    t = _scene_tool([], [_p("p0", "can", (0.3, 0.0, 1.05)), _p("p1", "cupboard", (0.0, -0.23, 0.9))])
    t.client = Extra([rej((0.019, -0.23, 0.9), 300), rej((0.17, -0.23, 0.9), 900)])  # where it was; and a false copy
    r = t.relook_place("p1")
    assert r["moved_m"] == 0.0 and r["assoc"] is None and r["still"] and r["rejected_m"] == [0.019, 0.17]
    assert t.percepts[1]["bodies"]["main"]["pos"] == [0.0, -0.23, 0.9]


def test_a_fixture_whose_drawer_opens_away_from_the_workspace_is_turned_about_its_middle(monkeypatch):
    import rowant_lab.pro2.robot.ground as GD

    def fake_poses(tdoc, roots, joints):  # a cupboard whose drawer slides along its own -y as its joint goes negative
        (root, (R, t)), = roots.items()
        R, t = np.asarray(R, float), np.asarray(t, float)
        return {root: (R, t), "drawer": (R, t + R @ np.array([0.0, float(joints.get("top", 0.0)), 0.0]))}

    def fake_aabb(lib, p):  # its boxes' middle is 2 cm behind its origin (so the mirror pose has another origin)
        R = np.asarray(lib.quat_mat(p["bodies"]["main"]["quat_wxyz"]), float)
        c = np.asarray(p["bodies"]["main"]["pos"], float) + R @ np.array([0.0, 0.02, 0.0])
        return c - 0.1, c + 0.1

    monkeypatch.setattr(LK, "part_body_poses", fake_poses)
    monkeypatch.setattr(GD, "aabb", fake_aabb)

    def tool(yaw):
        t = _scene_tool([], [_p("p0", "can", (0.3, 0.0, 1.05)), _p("p1", "cupboard", (0.0, -0.25, 0.9), yaw=yaw)])
        t.lib.export_types["cupboard"] = {**t.lib.export_types["cupboard"], "parts": {
            "top": {"bodies": ["drawer"], "joint": {"range": [-0.16, 0.0], "qpos0": 0.0}}}}
        t.obs["robot0_eef_pos"] = np.array([-0.15, 0.0, 1.2])  # the hand starts over the table, at +y of the cupboard
        return t

    t = tool(0.0)  # as fitted its drawer opens toward -y, away from the hand: the mirror pose was taken
    assert t.face_in() == 0  # off until the planner turns it on
    t.face_in_on = True
    assert t.face_in() == 1 and t.n_faced == 1
    b = t.percepts[1]["bodies"]
    assert np.allclose(b["main"]["pos"], [0.0, -0.21, 0.9])  # turned about the boxes' middle (0, -0.23)
    R = np.asarray(t.lib.quat_mat(b["main"]["quat_wxyz"]), float)
    assert np.allclose(R @ np.array([0.0, -1.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6)  # the drawer now opens toward +y
    assert "drawer" in b and t.percepts[1]["source"].endswith("+faced") and t.percepts[0]["bodies"]["main"]["pos"] == [0.3, 0.0, 1.05]
    assert t.face_in() == 0  # once turned it faces in: nothing more to do
    t = tool(180.0)  # fitted the right way round: left alone
    t.face_in_on = True
    assert t.face_in() == 0 and t.percepts[1]["bodies"]["main"]["pos"] == [0.0, -0.25, 0.9]
    t = tool(0.0)  # the hand's first place is what counts: a hand now behind the cupboard changes nothing
    t.face_in_on = True
    t._hand0 = np.array([0.0, -0.6])
    assert t.face_in() == 0


def test_a_turned_fixture_takes_the_trackers_pose_only_when_it_stays_near_and_faces_in(monkeypatch):
    import rowant_lab.pro2.robot.ground as GD

    def fake_poses(tdoc, roots, joints):
        (root, (R, t)), = roots.items()
        R, t = np.asarray(R, float), np.asarray(t, float)
        return {root: (R, t), "drawer": (R, t + R @ np.array([0.0, float(joints.get("top", 0.0)), 0.0]))}

    def fake_aabb(lib, p):
        c = np.asarray(p["bodies"]["main"]["pos"], float)
        return c - 0.1, c + 0.1

    monkeypatch.setattr(LK, "part_body_poses", fake_poses)
    monkeypatch.setattr(GD, "aabb", fake_aabb)

    class Tracker(Scene):
        def __init__(self, shift, yaw=180.0, status="tracked"):
            super().__init__([])
            self.shift, self.yaw, self.status = shift, yaw, status

        def call(self, msg):
            self.calls.append(msg)
            (pid, pr), = msg["priors"].items()
            return {"tracks": {pid: {"status": self.status, "R_w": _rz(self.yaw).tolist(), "n_px": 500,
                                     "t_w": (np.asarray(pr["t_w"]) + np.asarray(self.shift)).tolist()}}}

    def run(client):
        t = _scene_tool([], [_p("p0", "can", (0.3, 0.0, 1.05)), _p("p1", "cupboard", (0.0, -0.25, 0.9))])
        t.lib.export_types["cupboard"] = {**t.lib.export_types["cupboard"], "parts": {
            "main": {"bodies": ["main"], "joint": None},
            "top": {"bodies": ["drawer"], "joint": {"range": [-0.16, 0.0], "qpos0": 0.0}}}}
        t.obs["robot0_eef_pos"] = np.array([-0.15, 0.0, 1.2])
        t.client, t.face_in_on = client, True
        assert t.face_in() == 1
        return t, t.percepts[1]

    t, q = run(Tracker((0.0, 0.034, 0.0)))  # the tracker slides it 3.4 cm forward: taken
    assert np.allclose(q["bodies"]["main"]["pos"], [0.0, -0.216, 0.9]) and q["source"].endswith("+faced+tracked")
    assert t.n_track == 1 and t.looks[-1]["face_in"] and t.client.calls[0]["mode"] == "track"
    t, q = run(Tracker((0.0, 0.2, 0.0)))  # a jump of 20 cm: the turned pose stands
    assert np.allclose(q["bodies"]["main"]["pos"], [0.0, -0.25, 0.9]) and q["source"].endswith("+faced")
    t, q = run(Tracker((0.0, 0.01, 0.0), yaw=0.0))  # the tracker's pose faces away again: not taken
    assert q["source"].endswith("+faced")
    t, q = run(Tracker((0.0, 0.01, 0.0), status="lost"))
    assert q["source"].endswith("+faced")


def test_a_carry_stops_as_soon_as_the_fingers_have_met():
    def run(gaps, watch=True):
        a = SimpleNamespace(ap=AG.AgentParams(carry_watch=watch))
        it = iter(gaps)
        a._gap = lambda: next(it)

        def move():
            for k in range(8):
                yield [k]
            return "reached"

        g, sent = AG.OracleAgent._watch_carry(a, move(), held), []
        try:
            while True:
                sent.append(next(g))
        except StopIteration as e:
            return e.value, len(sent)

    held, empty = 0.0141, 0.0011
    assert run([held] * 8) == ("reached", 8)                                  # the object stays in the hand
    assert run([held, held, empty, empty, empty, empty, empty, empty]) == ("lost", 5)  # three steps with the fingers met
    assert run([held, empty, empty, held, held, held, held, held]) == ("reached", 8)  # two odd readings are not a loss
    assert run([0.0024] * 8) == ("reached", 8)                                # a thin hold is not a loss
    assert run([empty] * 8, watch=False) == ("reached", 8)                    # off: the move runs as it is
    held = 0.0022                                                             # the thinnest hold the lift check passes:
    assert run([0.0019] * 8) == ("reached", 8) and run([0.0012] * 8) == ("lost", 3)   # the limit is 0.75 of it

