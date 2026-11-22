"""Wrist frames without depth for the in-episode learner: the runner's source (``pro2/wrist_stereo.py``), the server's
depth for them (``lfb_episode.wrist_depth``), the request audit's declared form, and the off path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rowant_lab.libero_env import lfb_episode as LE
from rowant_lab.libero_env import m2_audit as AU
from rowant_lab.libero_env import m3_stereo1_match as SM
from rowant_lab.libero_env import m3_stereo1_poses as PO
from rowant_lab.pro2 import machine as MC
from rowant_lab.pro2 import wrist_stereo as WR


def _obs(T_cam, X, grip=(0.02, -0.02)):
    """Readings whose hand pose puts the wrist camera at ``T_cam``."""
    E = np.asarray(T_cam, float) @ np.linalg.inv(X)
    return {"robot0_eef_pos": list(E[:3, 3]), "robot0_eef_quat": list(PO.mat_to_quat_xyzw(E[:3, :3])),
            "robot0_gripper_qpos": list(grip)}


def _cam(x: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = x
    return T


class LearnThenAct:
    inner = None


def test_b_eff_and_the_partner_rule_are_k0s():
    rng = np.random.default_rng(3)
    ref = PO.pose_mat(PO.quat_to_mat([0.1, 0.2, -0.3, 0.9]), [0.0, 0.1, 0.3])
    for _ in range(20):
        Ts = []
        for _k in range(11):
            T = ref.copy()
            T[:3, 3] += rng.uniform(-0.1, 0.1, 3)
            Ts.append(T)
        assert all(abs(WR.b_eff(ref, T) - SM.b_eff(ref, T)) < 1e-12 for T in Ts)
        assert WR.pick_partners(ref, Ts) == SM.rm_partners(ref, Ts)  # K0's numbers: K0's rule
        near = WR.pick_partners(ref, Ts, b_max=0.04)
        assert all(0.01 <= SM.b_eff(ref, Ts[i]) <= 0.04 for i in near) and len(near) <= 3
    assert WR.DEFAULTS["b_max_m"] == 0.04 and WR.DEFAULTS["dt_lag_ms"] == 2.0 and WR.DEFAULTS["width"] == 1024


def test_wrist_frames_give_a_frame_without_depth_and_its_partners_within_four_cm():
    wf = WR.WristFrames({"width": 32, "dt_lag_ms": 0.0})
    X = WR.mount_X()
    xs = [0.0, 0.03, 0.05, 0.065, 0.075, 0.08]  # the hand moves along camera x; the last picture is the reference
    pics = [np.full((32, 32, 3), i, np.uint8) for i in range(len(xs))]
    with pytest.raises(RuntimeError):
        wf.frame()
    for t, (x, pic) in enumerate(zip(xs, pics)):
        wf.push(t, pic, _obs(_cam(x), X))
    fr = wf.frame()
    assert tuple(fr) == AU.STEREO_FRAME_KEYS  # the form the request audit declares, in its order
    assert fr["camera"] == WR.WRIST and fr["width"] == fr["height"] == 32 and fr["step"] == 5
    assert fr["depth_u16"].dtype == np.uint16 and not fr["depth_u16"].any()  # no reading anywhere
    assert np.array_equal(fr["rgb"], pics[5]) and np.allclose(fr["T_world_from_camera"], _cam(0.08))
    assert np.allclose(fr["X"], X) and np.allclose(fr["gripper_qpos"], [0.02, -0.02])
    assert np.allclose(fr["K"], PO.K_true(PO.f_of_fovy(75.0, 32), 32))
    # sideways distances from the reference: 8, 5, 3, 1.5, 0.5 cm -> within 1 to 4 cm: 3 cm (t 2) and 1.5 cm (t 3)
    assert [p["steps_back"] for p in fr["partners"]] == [3, 2]
    assert [p["b_eff_m"] for p in fr["partners"]] == [0.03, 0.015]
    assert all(set(p) == AU.PARTNER_KEYS for p in fr["partners"])
    assert np.array_equal(fr["partners"][0]["rgb"], pics[2])
    with pytest.raises(ValueError, match="32 px"):
        wf.push(6, np.zeros((16, 16, 3), np.uint8), _obs(_cam(0.1), X))
    wf.clear()
    with pytest.raises(RuntimeError):
        wf.frame()


def test_the_reading_is_advanced_by_the_lag_and_pictures_are_kept_only_while_looking():
    wf = WR.WristFrames({"width": 8})
    X = WR.mount_X()
    wf.push(0, np.zeros((8, 8, 3), np.uint8), _obs(_cam(0.0), X))
    wf.push(1, np.zeros((8, 8, 3), np.uint8), _obs(_cam(0.02), X))
    T = wf.frame()["T_world_from_camera"]
    assert abs(T[0, 3] - (0.02 + 0.02 * 2.0 / 50.0)) < 1e-9  # 2 ms of the last step's 2 cm
    assert len(wf.buf) == 2 and wf.buf.maxlen == 12
    agent = LearnThenAct()
    assert WR.WristFrames.wanted(agent) is True
    agent.inner = object()
    assert WR.WristFrames.wanted(agent) is False and WR.WristFrames.wanted(object()) is False


# ------------------------------------------------------------------------------------- the server's depth
@pytest.fixture(scope="module")
def g0():
    from rowant_lab.libero_env.tests import test_m3_stereo1_match as G

    return G, G.g0_views(512, ("ref", "px", "nx"))


def _frame(G, views, partners):
    K = G.wrist_K(512)
    return {"camera": WR.WRIST, "rgb": views["ref"]["rgb"], "depth_u16": np.zeros((512, 512), np.uint16),
            "depth_scale_mm": 0.1, "K": K, "T_world_from_camera": G.G0_POSES["ref"], "width": 512, "height": 512,
            "step": 3, "X": G.X_G0, "gripper_qpos": G.GRIP_Q,
            "partners": [{"rgb": views[nm]["rgb"], "K": K, "T_world_from_camera": G.G0_POSES[nm],
                          "gripper_qpos": G.GRIP_Q, "steps_back": i + 1, "b_eff_m": 0.045}
                         for i, nm in enumerate(partners)]}


def test_the_registered_hand_model_is_loaded_once():
    LE._HAND.clear()
    LE.preload_stereo()
    m = LE._HAND["model"]
    assert len(m["bodies"]) >= 3  # the hand and two fingers (K0's registered file)
    LE.preload_stereo()
    assert LE._HAND["model"] is m


def test_wrist_depth_fills_the_frame_and_leaves_unread_pixels_without_a_reading(g0, monkeypatch):
    G, views = g0
    # the synthetic scene has its own mount and no hand in view: no raster here (the real mount's raster is K0's)
    monkeypatch.setitem(LE._HAND, "model", {"bodies": []})
    fr = _frame(G, views, ("px", "nx"))
    row = LE.wrist_depth(fr, {})
    assert row["status"] == "ok" and row["n_partners"] == 2 and row["b_eff_mm"] == [45.0, 45.0] and row["s"] > 0
    d = fr["depth_u16"]
    assert d.dtype == np.uint16 and d.shape == (512, 512) and not (d == 65535).any()
    z, zt = d.astype(np.float64) * 1e-4, views["ref"]["depth"].astype(np.float64)
    plane = views["ref"]["pid"] == G.PID_PLANE
    read = plane & (d > 0)
    assert read.sum() > 0.9 * plane.sum() and np.median(np.abs(z - zt)[read]) < 0.001
    assert abs(row["read_share"] - float((d > 0).mean())) < 1e-4
    assert (d == 0).any()  # the plain box and the edges: nothing is filled in
    assert LE.STEREO_CONFIG == {1024: "PS-w13-loose", 512: "PS-w7-strict"}  # K0's picks


def test_wrist_depth_leaves_other_frames_alone(g0, monkeypatch):
    G, views = g0
    monkeypatch.setitem(LE._HAND, "model", {"bodies": []})
    plain = {"depth_u16": np.full((4, 4), 7, np.uint16), "width": 4}
    assert LE.wrist_depth(plain) == {"status": "has_depth"} and (plain["depth_u16"] == 7).all()
    fr = _frame(G, views, ())
    fr["depth_u16"] = np.full((512, 512), 9, np.uint16)  # whatever came is not depth: a colour-only frame
    row = LE.wrist_depth(fr)
    assert row["status"] == "no_partner" and not fr["depth_u16"].any()
    same = _frame(G, views, ("px",))
    same["partners"][0]["T_world_from_camera"] = G.G0_POSES["ref"]  # a partner that did not move: K0 drops it
    row = LE.wrist_depth(same)
    assert row["status"] == "failed" and not same["depth_u16"].any()


# ------------------------------------------------------------------------------------- what to learn
def test_a_thing_a_rejected_known_pose_covers_is_considered_last():
    # sizes largest first, as the components come
    assert LE.rank_components([1300, 1100, 2], [0.75, 0.1, 0.0], 0.6, 60) == [1, 0, 2]  # the covered one goes behind
    assert LE.rank_components([1300, 1100], [0.2, 0.1], 0.6, 60) == [0, 1]  # nothing covered: the old order
    assert LE.rank_components([1300, 1100], [0.9, 0.8], 0.6, 60) == [0, 1]  # all covered: the old order
    assert LE.rank_components([1300, 40], [0.9, 0.0], 0.6, 60) == [0, 1]  # the other is too small to learn
    assert LE.rank_components([900, 800, 700], [0.7, 0.65, 0.1], 0.6, 60) == [2, 0, 1]
    assert LE.RANK_KNOWN == {"cover": 0.6, "margin": None, "size_share": 0.25, "size_vox": 300}
    # a basket-sized thing first (covered 0.97): an ordinary thing behind it still counts as comparable
    assert LE.rank_components([8971, 1500, 40], [0.965, 0.651, 0.0], 0.6, 60, margin=0.1) == [1, 0, 2]
    assert LE.rank_components([8971, 250, 40], [0.965, 0.651, 0.0], 0.6, 60, margin=0.1) == [0, 1, 2]  # a fragment
    # the margin (off unless given): everything covered, the largest clearly more than a thing of comparable size
    assert LE.rank_components([1319, 1128, 2], [0.847, 0.627, 1.0], 0.6, 60) == [0, 1, 2]
    assert LE.rank_components([1319, 1128, 2], [0.847, 0.627, 1.0], 0.6, 60, margin=0.1) == [1, 0, 2]
    assert LE.rank_components([801, 659, 79, 18, 14], [0.855, 0.687, 0.658, 0.0, 0.786], 0.6, 60, margin=0.1) == [1, 0, 2, 3, 4]
    assert LE.rank_components([801, 659], [0.75, 0.70], 0.6, 60, margin=0.1) == [0, 1]  # not clearly more
    assert LE.rank_components([1362, 679], [0.756, 0.0], 0.6, 60, margin=0.1) == [1, 0]  # the first rule still decides
    src = Path(LE.__file__).read_text()
    assert src.count('per.p.get("learn_rank_known")') == 1  # off unless the server is given it
    assert "rejected" in AU.REQUEST_KEYS["learn_plan"] and "rejected" not in AU.REQUEST_KEYS["learn"]
    assert {"stage", "top"} <= AU.REQUEST_KEYS["learn_plan"]


# ------------------------------------------------------------------------------------- the top-down look first
def test_the_top_look_aims_at_the_part_nearest_the_main_camera():
    # a thing smeared 20 cm along x, away from a camera at x = +0.9
    C = np.stack([np.linspace(0.0, -0.2, 41), np.zeros(41), np.full(41, 0.05)], 1)
    xy = LE.near_point(C, [0.9, 0.0, 0.65], 0.25)
    assert -0.03 < xy[0] <= 0.0 and abs(xy[1]) < 1e-12  # the near quarter: x from 0 to -4.5 cm
    assert np.allclose(LE.near_point(C[:1], [0.9, 0.0, 0.65], 0.25), C[0, :2])
    assert LE.TOP_EL == (90.0, 80.0, 70.0) and LE.TOP_FIRST["refind_m"] == 0.06 and LE.TOP_FIRST["plane"] == "wrist"
    assert LE.TOP_FIRST["mid"] is True and LE.TOP_FIRST["agent"] == "none" and LE.TOP_FIRST["carve_grow_px"] == 2
    assert Path(LE.__file__).read_text().count('per.p["learn_top_first"].get("self_px") is not None') == 1
    import inspect

    assert inspect.signature(LE.plan_looks).parameters["mid"].default is True  # the old plan is the default


def test_the_support_height_is_read_from_the_wrist_frame_when_it_shows_the_support():
    rng = np.random.default_rng(1)
    floor = np.stack([rng.uniform(-0.2, 0.2, 4000), rng.uniform(-0.2, 0.2, 4000), 0.004 + rng.normal(0, 0.0004, 4000)], 1)
    thing = np.stack([rng.uniform(-0.02, 0.02, 800), rng.uniform(-0.02, 0.02, 800), rng.uniform(0.01, 0.09, 800)], 1)
    Pw = np.concatenate([floor, thing]).reshape(-1, 1, 3)
    ok = np.ones(Pw.shape[:2], bool)
    first = {"normal_w": [0.0, 0.0, 1.0], "height_w": 0.0}
    got = LE.wrist_plane(Pw, ok, first, 0.03)
    assert abs(got["height_w"] - 0.004) < 0.0005 and got["first_look_height_w"] == 0.0 and got["normal_w"] == [0.0, 0.0, 1.0]
    tilted = {"normal_w": [0.0, 0.2, 0.98], "height_w": 0.0}
    assert LE.wrist_plane(Pw, ok, tilted, 0.03) is tilted  # not level: the first look's plane stays
    assert LE.wrist_plane(Pw[:100], ok[:100], first, 0.03) is first  # too little of the support in view


def test_the_robots_own_pixels_are_found_in_the_picture_and_grown():
    class Panda:
        @staticmethod
        def fk(joint_pos, gripper_qpos):
            return {"hand": (np.eye(3), np.zeros(3))}

    class WC:
        panda = Panda()
        robot_pts = {"hand": np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.0, -1.0], [5.0, 0.0, 1.0]])}

    fr = {"K": [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]], "T_world_from_camera": np.eye(4),
          "depth_u16": np.zeros((64, 64), np.uint16)}
    m = LE.self_pixels(WC, fr, None, None, 0)
    assert m.sum() == 2 and m[32, 32] and m[32, 42]  # the point behind the camera and the one off the picture: none
    g = LE.self_pixels(WC, fr, None, None, 3)
    assert g[32, 35] and g[29, 32] and g[32, 45] and not g[32, 37] and g.sum() > m.sum()
    assert LE.TOP_FIRST["self_px"] is None  # off unless given


def test_a_refused_look_says_why_in_one_word():
    class Arm:
        def __init__(self, r):
            self.r = r

        def ik(self, c, d, q):
            return self.r

        @staticmethod
        def robot_points(q):
            return np.zeros((2, 3))

    class Clear:
        def __init__(self, c):
            self.c = c

        def check(self, pts):
            return self.c

    q = np.zeros(7)
    conv = {"ok": True, "q": q, "iters": 3, "pos_err_m": 0.001, "ang_err_deg": 0.2, "limit": False}
    why = lambda r, c: LE._solve(Arm(r), Clear(c), None, None, [q])["why"]  # noqa: E731
    assert why({**conv, "ok": False, "iters": 150}, None) == "reach"
    assert why({**conv, "ok": False, "limit": True}, None) == "limit"
    assert why(conv, {"ok": False, "above_plane_m": 0.2, "near_scene_m": 0.003, "near_at": [0.0, 0.0, 0.3], "n_near": 9}) == "scene"
    assert why(conv, {"ok": False, "above_plane_m": 0.002, "near_scene_m": None}) == "plane"
    assert why(conv, {"ok": True, "above_plane_m": 0.2, "near_scene_m": None}) == "ok"
    # the scene check itself says where the point is
    cl = LE.Clearance(np.array([[0.0, 0.0, 0.30], [1.0, 1.0, 0.05]]), {"normal_w": [0, 0, 1], "height_w": 0.0})
    got = cl.check(np.array([[0.0, 0.003, 0.30], [0.5, 0.5, 0.5]]))
    assert got["ok"] is False and got["near_scene_m"] == 0.003 and got["near_at"] == [0.0, 0.0, 0.3] and got["n_near"] == 1
    assert "near_at" not in cl.check(np.array([[0.5, 0.5, 0.5]]))
    from rowant_lab.pro2 import learn_skill as LS

    rows = LS.why_tries([{"el_deg": 90.0, "az_deg": 0.0, "why": "scene", "clear": got},
                         {"el_deg": 80.0, "az_deg": 0.0, "why": "reach", "pos_err_mm": 41.0, "ang_err_deg": 3.0}, 1])
    assert rows == [{"el": 90.0, "az": 0.0, "why": "scene", "near_mm": 3.0, "near_at": [0.0, 0.0, 0.3], "n_near": 1},
                    {"el": 80.0, "az": 0.0, "why": "reach", "pos_err_mm": 41.0, "ang_err_deg": 3.0}]


def test_a_carve_only_look_gives_the_model_no_surface():
    from rowant_lab.libero_env import lfb_build as LB_

    W, h, half, top = 96, 0.30, 0.04, 0.05
    K = PO.K_true(PO.f_of_fovy(75.0, W), W)
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [0.0, 0.0, h]
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(W) + 0.5)
    x, y = (u - K[0, 2]) / K[0, 0], -(v - K[1, 2]) / K[1, 1]  # the ray's world slope (the camera looks down)
    on = (np.abs(x * (h - top)) <= half) & (np.abs(y * (h - top)) <= half)
    z = np.where(on, h - top, h)
    rgb = np.full((W, W, 3), 128, np.uint8)
    true = {"rgb": rgb, "z": z, "K": K, "T": T, "mask": on}
    smear = {"rgb": rgb, "z": np.where(on, h - top + 0.03, h), "K": K, "T": T, "mask": on}  # the thing 3 cm too far
    o = np.zeros(3)
    cfg = {"vox_mm": 4.0, "trunc_k": 3.0}
    one = LB_.tsdf_build([true], o, cfg)
    both = LB_.tsdf_build([true, {**smear, "carve_only": True}], o, cfg)
    assert both["n_points"] == one["n_points"] and both["grid"] == one["grid"]  # no point, no room for its surface
    assert np.allclose(both["V"][:, 2].max(), one["V"][:, 2].max())
    assert abs(one["V"][:, 2].max() - top) < 0.004
    mixed = LB_.tsdf_build([true, smear], o, cfg)  # taken as a look, the smear would put a second surface below
    assert mixed["n_points"] == 2 * one["n_points"]


class _Server:
    """Canned replies of the two-stage plan; it records what it was asked."""

    def __init__(self, pose):
        self.msgs, self.pose = [], pose

    def call(self, msg):
        self.msgs.append(msg)
        look = lambda i, kind: {"look": i, "kind": kind, "az_used_deg": 0.0, "el_used_deg": 90.0, "tries": [1],
                                "waypoints": [], **self.pose}
        if msg["op"] == "learn_plan" and msg.get("stage") == "top":
            return {"ok": True, "status": "ok", "stage": "top", "n_planned": 1, "looks": [look(0, "top")],
                    "leftover": {"centre_w": [0.0, 0.0, 0.05], "aim_w": [0.02, 0.0, 0.05]}}
        if msg["op"] == "learn_plan" and msg.get("stage") == "flanks":
            return {"ok": True, "status": "ok", "stage": "flanks", "n_planned": 2,
                    "looks": [look(0, "flank"), look(1, "flank")], "stereo": {"status": "ok", "read_share": 0.5},
                    "leftover": {"centre_w": [0.06, 0.01, 0.04], "shift_m": 0.0608, "n_vox_object": 700,
                                 "plane": {"height_w": 0.003, "first_look_height_w": 0.0}}}
        return {"ok": True, "status": "no_masks"}


def test_the_learner_looks_from_above_first_and_asks_for_the_flanks_with_that_frame():
    from rowant_lab.pro2 import learn_skill as LS

    pos, quat_xyzw = [0.1, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0]
    obs = {"robot0_joint_pos": [0.0] * 7, "robot0_gripper_qpos": [0.02, -0.02], "robot0_eef_pos": pos,
           "robot0_eef_quat": quat_xyzw}
    server = _Server({"eef_pos": pos, "eef_quat_wxyz": [0.0, 1.0, 0.0, 0.0]})  # every look is where the hand is
    n = {"wrist": 0}

    class Cam:
        @staticmethod
        def wrist_frame():
            n["wrist"] += 1
            return {"camera": LS.WRIST, "n": n["wrist"], "partners": [{"b_eff_m": 0.021}, {"b_eff_m": 0.012}]}

    class Tool:
        client, cam, suite, episode, percepts, bridge_s = server, Cam(), "libero_object", "0123456789abcdef", [], 0.0
        looks = []

        def __init__(self):
            self.obs = obs

        def _frame(self, camera):
            return {"camera": camera}

        def wrist_find(self, typ, support_z, hint_xy=None, opts=None):
            assert typ == "milk" and abs(support_z - 0.003) < 1e-12 and hint_xy == [0.06, 0.01]
            return {"xy": [0.07, 0.01], "top_z": 0.05}

    class R:
        @staticmethod
        def plan_episode(instruction, tool, lib, left, params):
            return {"agent": None, "grounding": {"status": "fail"}, "parsed": {}}

    agent = LS.LearnThenAct(R, "pick the milk", Tool(), None, 280, {"learn_top_first": {}, "learn_looks": "fast"},
                            "milk", {"plane": {"normal_w": [0, 0, 1], "height_w": 0.0}})
    act = agent.act(obs)  # every look is reached at once: the whole learning phase runs in this call
    assert act == [0.0] * 6 + [-1.0] and agent.finished
    ops = [(m["op"], m.get("stage")) for m in server.msgs]
    assert ops == [("learn_plan", "top"), ("learn_plan", "flanks"), ("learn", None)]
    assert set(server.msgs[1]) <= {"op", "stage", "suite", "episode", "top"}
    assert server.msgs[1]["top"]["cameras"][0]["n"] == 2  # the frame taken at the top look (1 was before any motion)
    assert [f["n"] for f in server.msgs[2]["cameras"]] == [3, 4] and len(server.msgs[2]["joints"]) == 2
    rec = agent.rec
    assert [lk.get("kind") for lk in rec["looks"]] == ["top", None, None]
    assert rec["looks"][0]["partners_mm"] == [21.0, 12.0]
    assert rec["plan2"]["leftover"]["shift_m"] == 0.0608 and rec["plan2"]["plane"]["height_w"] == 0.003
    assert rec["plan2"]["finder"] == {"found": True, "xy": [0.07, 0.01], "from_depth_m": 0.01}
    assert rec["plan"]["leftover"]["aim_w"] == [0.02, 0.0, 0.05] and rec["fallback"] is True  # no model: its own plan
    # a server without the option answers the first request as before: the robot then takes the old path
    class Old(_Server):
        def call(self, msg):
            self.msgs.append(msg)
            return {"ok": True, "status": "no_leftover"}

    old = Old({})
    t2 = Tool()
    t2.client = old
    a2 = LS.LearnThenAct(R, "pick the milk", t2, None, 280, {"learn_top_first": {}}, "milk",
                         {"plane": {"normal_w": [0, 0, 1], "height_w": 0.0}})
    a2.act(obs)
    assert [m["op"] for m in old.msgs] == ["learn_plan"] and "plan2" not in a2.rec


# ------------------------------------------------------------------------------------- the audit and the off path
class _Echo:
    def call(self, msg):
        return {"ok": True}

    def close(self):
        pass


def test_the_request_audit_declares_the_stereo_frame_for_the_learner_only(g0):
    G, views = g0
    fr = _frame(G, views, ("px", "nx"))
    c = AU.AuditedClient(_Echo())
    c.call({"op": "learn", "suite": "libero_object", "episode": "0123456789abcdef", "label": "milk", "cameras": [fr],
            "joints": [{"joint_pos": [0.0] * 7, "gripper_qpos": [0.0, 0.0]}]})
    assert c.violations == []
    bad = _frame(G, views, ("px",))
    bad["partners"][0]["depth_u16"] = np.zeros((2, 2), np.uint16)  # a partner carries no depth
    c.call({"op": "learn", "suite": "libero_object", "episode": "0123456789abcdef", "label": "milk", "cameras": [bad],
            "joints": []})
    assert len(c.violations) == 1 and "camera keys" in c.violations[0]
    c2 = AU.AuditedClient(_Echo())
    c2.call({"op": "look", "mode": "full", "suite": "libero_object", "episode": "0123456789abcdef", "look": 0,
             "cameras": [fr], "joint_pos": [0.0] * 7, "gripper_qpos": [0.0, 0.0]})
    assert len(c2.violations) == 1  # an ordinary look takes ordinary frames only


def test_the_learner_takes_its_wrist_frame_from_the_runners_source_when_there_is_one():
    from rowant_lab.pro2 import learn_skill as LS

    class Cam:
        pass

    class Tool:
        cam = Cam()

        def _frame(self, camera):
            return {"from": "the camera tool", "camera": camera}

    agent = LS.LearnThenAct.__new__(LS.LearnThenAct)
    agent.tool = Tool()
    assert agent._wrist() == {"from": "the camera tool", "camera": LS.WRIST}
    Tool.cam.wrist_frame = lambda: {"from": "the runner"}
    assert agent._wrist() == {"from": "the runner"}


def test_the_options_are_off_by_default_and_v9_does_not_name_them():
    src = (Path(WR.__file__).resolve().parent / "dev_episode.py").read_text()
    assert src.count('params.get("learn_wrist_stereo") is not None and params.get("learn_in_episode")') == 1
    assert src.count("if wframes is not None:") == 3 and src.count("wframes = None") == 1
    # the picture is pushed after the images are this step's (pushed before, picture and reading were a step apart)
    assert src.index("images.update(CL.episode_images(r[\"observation\"]))\n        if wframes is not None:") > 0
    assert src.count('if params.get("learn_in_episode"):  # 2026-11-17: the learner again') == 1  # the replan keeps it
    srv_src = Path(LE.__file__).read_text()
    assert srv_src.count('per.p.get("learn_top_first") is not None and msg.get("stage") is not None') == 1
    srv = (Path(LE.__file__).resolve().parent / "m2_perceive.py").read_text()
    assert srv.count('.get("learn_wrist_stereo") is not None') == 1
    assert Path(LE.__file__).read_text().count('per.p.get("learn_wrist_stereo") is not None') == 2  # learn, refind
    from rowant_lab.pro2 import long_dev as LD

    b = json.dumps(LD.bundle(LD.colour_long_params(MC.REGISTERED["net"])))
    assert "learn_wrist_stereo" not in b and "learn_in_episode" not in b
