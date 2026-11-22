"""In-episode learning (W2-LEARN-EP): the look geometry, the arm's IK on the exported Panda model, the server's audit
rule for learning, and the robot side's trigger and fallback."""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from rowant_lab.libero_env import lfb_episode as LE


def test_unit_and_goal_point_at_the_centre():
    c = np.array([0.1, -0.2, 0.05])
    base = np.array([-0.6, 0.0])
    u0 = LE.unit(base, c, 0.0, 45.0)
    v = (base - c[:2]) / np.linalg.norm(base - c[:2])
    assert np.allclose(u0[:2] / np.linalg.norm(u0[:2]), v) and math.isclose(u0[2], math.sin(math.radians(45)))
    cg, d = LE.goal(c, u0)
    assert math.isclose(float(np.linalg.norm(cg - c)), 0.30) and np.allclose(d, -u0)
    top = LE.unit(base, c, 123.0, 90.0)
    assert np.allclose(top, [0, 0, 1], atol=1e-12)


def test_slerp_stays_on_the_sphere():
    a, b = LE.unit([-0.6, 0], [0, 0, 0], -90, 45), LE.unit([-0.6, 0], [0, 0, 0], 90, 45)
    for f in np.linspace(0, 1, 7):
        u = LE.slerp(a, b, f)
        assert math.isclose(float(np.linalg.norm(u)), 1.0, rel_tol=1e-9)
    assert np.allclose(LE.slerp(a, b, 0.5), [0, 0, 1], atol=1e-9)  # opposite flanks at 45 deg meet over the top


def test_rotvec_round_trip():
    from rowant_lab.libero_env import m2_kin as KIN

    rng = np.random.default_rng(0)
    for _ in range(20):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = float(rng.uniform(0.01, math.pi - 0.01))
        v = LE._rotvec(KIN.axis_angle(axis, ang))
        assert np.allclose(v, axis * ang, atol=1e-6)


@pytest.fixture(scope="module")
def arm():
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.libero_env import m2_paths as MP

    try:
        panda = LB.LibraryReader(MP.POLICY_LIB, MP.M1_LIBRARY, allow_panda=True).panda("libero_object")
    except Exception as exc:  # the object library is not built on this machine
        pytest.skip(f"no Panda export here: {exc}")
    wc = types.SimpleNamespace(panda=panda, robot_pts=panda.surface_samples(0.004))
    q0 = np.array([0.0, -0.16, 0.0, -2.44, 0.0, 2.23, 0.79])  # LIBERO's rest pose (measured)
    gq = np.array([0.04, -0.04])
    P = panda.fk(q0, gq)
    Rh, th = P["robot0_right_hand"]
    T_cam = LE._T(Rh @ np.diag([1.0, -1.0, -1.0]), th + Rh @ np.array([0.05, 0.0, 0.0]))  # a mount on the hand
    from rowant_lab.libero_env import m2_kin as KIN

    q_xyzw = KIN.mat_to_quat(Rh)  # wxyz
    return LE.Arm(wc, q0, gq, th, [q_xyzw[1], q_xyzw[2], q_xyzw[3], q_xyzw[0]], T_cam)


def test_goal_rot_tilts_without_twist(arm):
    d = np.array([0.3, -0.2, -0.9])
    d /= np.linalg.norm(d)
    R = arm.goal_rot(d)
    assert np.allclose(R[:, 2], d, atol=1e-9)
    # the smallest rotation: R R0^T has its axis perpendicular to both optical axes
    v = LE._rotvec(R @ arm.R_cam0.T)
    assert abs(float(v @ arm.R_cam0[:, 2])) < 1e-6 and abs(float(v @ d)) < 1e-6
    ang = math.degrees(float(np.linalg.norm(v)))
    assert math.isclose(ang, math.degrees(math.acos(float(arm.R_cam0[:, 2] @ d))), abs_tol=1e-3)


def test_ik_reaches_a_pose_made_by_fk(arm):
    q_true = arm.q0 + np.array([0.2, 0.15, -0.1, 0.2, 0.1, 0.25, -0.2])
    c, ax = arm.camera(q_true)
    r = arm.ik(c, ax, arm.q0)
    assert r["ok"], r
    c2, ax2 = arm.camera(r["q"])
    assert float(np.linalg.norm(c2 - c)) <= LE.IK_POS_TOL
    assert np.all(r["q"] >= arm.range[:, 0] + LE.LIMIT_MARGIN) and np.all(r["q"] <= arm.range[:, 1] - LE.LIMIT_MARGIN)
    e_pos, e_quat = arm.eef(r["q"])
    assert len(e_pos) == 3 and math.isclose(float(np.linalg.norm(e_quat)), 1.0, rel_tol=1e-9)


def test_clearance_refuses_points_near_the_scene_and_the_plane():
    plane = {"normal_w": [0, 0, 1], "height_w": 0.0}
    cl = LE.Clearance(np.array([[0.0, 0.0, 0.05]]), plane)
    assert cl.check(np.array([[0.5, 0.5, 0.3]]))["ok"]
    assert not cl.check(np.array([[0.0, 0.0, 0.06]]))["ok"]  # 1 cm from a scene point
    assert not cl.check(np.array([[0.5, 0.5, 0.005]]))["ok"]  # 5 mm above the plane


def test_plan_looks_orders_flanks_and_middle(arm):
    c_now, _ = arm.camera(arm.q0)
    centre = np.array([c_now[0] + 0.05, c_now[1] - 0.25, 0.05])
    clear = LE.Clearance(np.zeros((0, 3)), {"normal_w": [0, 0, 1], "height_w": -1.0})
    looks = LE.plan_looks(arm, clear, centre, np.array([c_now[0] - 0.5, c_now[1]]))
    planned = [r for r in looks if "q" in r]
    assert len(planned) >= 2
    kinds = [r["kind"] for r in planned]
    if len(planned) == 3:
        assert kinds == ["flank", "mid", "flank"]  # the middle look (azimuth 0, from the robot's side)
        assert planned[0]["dist"] <= planned[2]["dist"]
    for r in planned[1:]:  # consecutive looks are joined by waypoints no more than WAYPOINT_DEG apart
        assert all("q" in w for w in r["waypoints"])


# ------------------------------------------------------------------------------------------------ the server
def test_audit_roots_add_the_learn_folder_only_when_learning(tmp_path):
    from rowant_lab.libero_env import m2_paths as MP
    from rowant_lab.libero_env import m2_perceive as PR

    plain = PR.audit_roots(tmp_path / "log", tmp_path / "s")
    learn = PR.audit_roots(tmp_path / "log", tmp_path / "s", learn_dir=tmp_path / "s" / "learn")
    ld = PR._real(tmp_path / "s" / "learn")
    assert ld not in plain["read"] and ld in learn["read"] and ld in learn["write"]
    assert PR._real(MP.M1_LIBRARY) in learn["read"] and PR._real(MP.M1_LIBRARY) not in plain["read"]


def test_handle_routes_learning_ops_only_when_on():
    from rowant_lab.libero_env import m2_perceive as PR

    per = PR.Perceiver.__new__(PR.Perceiver)
    per.p, per.cells, per.ep_scope, per.learn_state = {}, {}, {}, {}
    with pytest.raises(ValueError):
        per.handle({"op": "learn_plan", "episode": "0" * 16})


# ------------------------------------------------------------------------------------------------ the robot side
class _Tool:
    def __init__(self, percepts):
        self.percepts = percepts
        self.suite, self.episode = "libero_object", "0" * 16
        self.bridge_s = self.render_s = 0.0


def _R(target="milk", status="ok"):
    calls = []

    def parse(instruction, types_):
        return {"status": status, "target_type": target, "place_type": "basket", "predicate": "in"}

    def plan_episode(instruction, tool, lib, cap, params):
        calls.append(cap)
        return {"agent": None, "parsed": {"status": "ok", "target_type": target}, "grounding": {"status": "fail"}}

    return types.SimpleNamespace(parse=parse, plan_episode=plan_episode, calls=calls)


def _lib(parts=1):
    return types.SimpleNamespace(types={"milk": {}, "basket": {}},
                                 export_types={"milk": {"parts": {f"p{i}": {} for i in range(parts)}},
                                               "basket": {"parts": {"p0": {}}}})


def test_trigger_needs_a_named_single_part_type_with_no_accepted_percept():
    from rowant_lab.pro2 import learn_skill as LS

    assert LS.triggers(_R(), "pick the milk", _Tool([{"type": "basket"}]), _lib()) == "milk"
    assert LS.triggers(_R(), "pick the milk", _Tool([{"type": "milk"}]), _lib()) is None
    assert LS.triggers(_R(status="fail"), "?", _Tool([]), _lib()) is None
    assert LS.triggers(_R(), "pick the milk", _Tool([]), _lib(parts=2)) is None


def test_no_trigger_is_the_robots_own_plan():
    from rowant_lab.pro2 import learn_skill as LS

    R = _R()
    out = LS.plan(R, "pick the milk", _Tool([{"type": "milk"}]), _lib(), 280, {}, {"plane": None})
    assert out["learn"] == {"trigger": False} and R.calls == [280]


def test_no_plane_falls_back_with_the_full_cap():
    from rowant_lab.pro2 import learn_skill as LS

    R = _R()
    out = LS.plan(R, "pick the milk", _Tool([]), _lib(), 280, {}, {"plane": None})
    ag = out["agent"]
    obs = {"robot0_joint_pos": [0.0] * 7, "robot0_gripper_qpos": [0.04, -0.04], "robot0_eef_pos": [0, 0, 1],
           "robot0_eef_quat": [0, 0, 0, 1]}
    a = ag.act(obs)
    assert a == [0.0] * 6 + [-1.0] and ag.finished and R.calls == [280]
    assert ag.rec["fallback"] is True and ag.rec["steps"] == 0 and ag.summary()["finish_reason"] == "grounding_failed"


# ------------------------------------------------------------------------------------------------ check before replacing
def _box_nodes(hx, hy, h, step=0.0025):
    """Surface samples of a box (body frame: footprint centred on the origin, bottom at z = 0) with outward normals."""
    pts, nrm = [], []
    for ax, half in ((0, hx), (1, hy), (2, None)):
        for sgn in (-1.0, 1.0):
            if ax == 2:
                xs, ys = np.arange(-hx, hx + 1e-9, step), np.arange(-hy, hy + 1e-9, step)
                X, Y = np.meshgrid(xs, ys)
                P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, h if sgn > 0 else 0.0)], 1)
                N = np.tile([0.0, 0.0, sgn], (len(P), 1))
            else:
                other = hy if ax == 0 else hx
                a, z = np.meshgrid(np.arange(-other, other + 1e-9, step), np.arange(0.0, h + 1e-9, step))
                P = np.zeros((a.size, 3))
                P[:, ax], P[:, 1 - ax], P[:, 2] = sgn * half, a.ravel(), z.ravel()
                N = np.zeros((len(P), 3))
                N[:, ax] = sgn
            pts.append(P)
            nrm.append(N)
    return np.concatenate(pts), np.concatenate(nrm)


def _look_at(c, target):
    z = np.asarray(target, float) - c
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.stack([x, np.cross(z, x), z], 1), c
    return T


def _render(T, K, shape, box, table_z):
    """Depth (camera z) of a scene with one box (centre, yaw, half extents, height) on a table plane; and the box mask."""
    (cx, cy), yaw, (hx, hy), h = box
    H, W = shape
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    dw = dc @ T[:3, :3].T
    o = T[:3, 3]
    t_pl = (table_z - o[2]) / np.where(np.abs(dw[:, 2]) > 1e-9, dw[:, 2], 1e-9)
    t_pl = np.where(t_pl > 0, t_pl, np.inf)
    c, s = math.cos(yaw), math.sin(yaw)
    Rb = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    ob = (o - np.array([cx, cy, table_z + h / 2])) @ Rb
    db = dw @ Rb
    half = np.array([hx, hy, h / 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-half - ob) / db, (half - ob) / db
    tmin, tmax = np.nanmax(np.minimum(t1, t2), 1), np.nanmin(np.maximum(t1, t2), 1)
    hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 0)
    t_box = np.where(hit, tmin, np.inf)
    z = np.minimum(t_pl, t_box)
    z = np.where(np.isfinite(z), z, 0.0).reshape(H, W)
    return z, (hit & (t_box < t_pl)).reshape(H, W)


def _check_scene(true_box, model_dims):
    table_z = 0.9
    K = np.array([[160.0, 0, 80.0], [0, 160.0, 80.0], [0, 0, 1.0]])
    (cx, cy), _yaw, _half, h = true_box
    tgt = np.array([cx, cy, table_z + h / 2])
    lks = []
    for az, el in ((-90.0, 45.0), (0.0, 70.0), (90.0, 45.0), (180.0, 30.0)):
        u = np.array([math.cos(math.radians(el)) * math.cos(math.radians(az)),
                      math.cos(math.radians(el)) * math.sin(math.radians(az)), math.sin(math.radians(el))])
        T = _look_at(tgt + 0.30 * u, tgt)
        z, m = _render(T, K, (160, 160), true_box, table_z)
        lks.append({"z": z, "K": K, "T": T, "mask": m & (z > 0)})
    nodes, nrm = _box_nodes(*model_dims)
    wc = types.SimpleNamespace(types={"milk": {"root_body": "milk_main"}}, gid={("milk", "milk_main"): 7},
                               nodes={"7": (nodes, nrm)}, rest_up={"7": np.array([0.0, 0.0, 1.0])})
    per = types.SimpleNamespace(p={"agree_m": 0.01, "see_through_m": 0.015, "facing_cos": 0.15,
                                   "accept_inlier_region": 0.85, "accept_min_agree": 30})
    return LE.check_own(per, wc, "milk", lks, {"normal_w": [0.0, 0.0, 1.0], "height_w": table_z})


def test_check_own_accepts_its_own_shape_and_finds_its_pose():
    out = _check_scene(((0.10, -0.05), math.radians(30.0), (0.03, 0.02), 0.10), (0.03, 0.02, 0.10))
    assert out["own"], out
    assert out["inlier"] >= 0.85 and out["cover"] >= 0.85
    t = np.asarray(out["t_w"])
    assert float(np.linalg.norm(t[:2] - [0.10, -0.05])) < 0.006 and abs(t[2] - 0.9) < 0.006
    R = np.asarray(out["R_w"])
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0])) % 180.0  # a box is symmetric under a half turn
    assert min(abs(yaw - 30.0), 180.0 - abs(yaw - 30.0)) < 4.0


def test_check_own_refuses_a_smaller_object():
    out = _check_scene(((0.10, -0.05), 0.3, (0.02, 0.02), 0.06), (0.03, 0.02, 0.10))
    assert not out["own"] and out["inlier"] < 0.85, out


def test_check_own_refuses_a_bigger_object():
    out = _check_scene(((0.10, -0.05), 0.3, (0.05, 0.04), 0.12), (0.03, 0.02, 0.10))
    assert not out["own"] and out["cover"] < 0.85, out


def test_learn_builds_nothing_when_the_check_says_own(monkeypatch):
    calls = []
    monkeypatch.setattr(LE, "masks", lambda per, wc, st, frames: ([{"z": np.ones((2, 2)), "mask": np.ones((2, 2), bool)}],
                                                                [{"look": 0, "n_px": 4}]))
    monkeypatch.setattr(LE, "check_own", lambda per, wc, label, lks, plane: {
        "own": True, "status": "own", "R_w": np.eye(3).tolist(), "t_w": [0.1, 0.2, 0.9], "gid": "7", "n_agree": 900,
        "inlier": 0.95, "cover": 0.97})
    monkeypatch.setattr(LE, "build_model", lambda *a, **k: calls.append(a))
    per = types.SimpleNamespace(p={"learn_check_own": True}, cell=lambda suite: None, learn_state={"e": {
        "suite": "libero_object", "wrist0": None, "agent_look": {"mask": np.zeros((2, 2), bool)},
        "leftover": {"plane": {"normal_w": [0, 0, 1], "height_w": 0.9}}}})
    out = LE.learn(per, {"suite": "libero_object", "episode": "e", "label": "milk", "cameras": [], "joints": []})
    assert out["status"] == "own" and out["own"]["t_w"] == [0.1, 0.2, 0.9] and not calls


def test_own_model_acts_with_a_close_look_percept():
    from rowant_lab.pro2 import learn_skill as LS

    R = _R()
    tool = _Tool([{"type": "basket", "id": "p0"}])
    tool._build = lambda typ, Rm, t: {"milk_main": {"pos": [float(x) for x in t], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}
    ag = LS.LearnThenAct(R, "pick the milk", tool, _lib(), 280, {}, "milk", {"plane": None})
    ag.t = 70
    ag._act_agent(None, {"R_w": np.eye(3).tolist(), "t_w": [0.1, 0.2, 0.9], "gid": "7", "n_agree": 900, "inlier": 0.93})
    assert ag.rec["acting"] == "own_close" and R.calls == [210]
    close = [p for p in tool.percepts if p.get("id") == "close0"]
    assert len(close) == 1 and close[0]["type"] == "milk" and close[0]["source"] == "close_look"
    assert close[0]["bodies"]["milk_main"]["pos"] == [0.1, 0.2, 0.9]


def test_dev_force_drops_the_accepted_target_and_learns():
    from rowant_lab.pro2 import learn_skill as LS

    R = _R()
    tool = _Tool([{"type": "milk", "id": "p1"}, {"type": "basket", "id": "p0"}])
    out = LS.plan(R, "pick the milk", tool, _lib(), 280, {"learn_dev_force": True}, {"plane": None})
    assert out["learn"]["trigger"] and [p["type"] for p in tool.percepts] == ["basket"] and R.calls == []
    tool2 = _Tool([{"type": "milk", "id": "p1"}])
    out2 = LS.plan(_R(), "pick the milk", tool2, _lib(), 280, {}, {"plane": None})
    assert out2["learn"] == {"trigger": False} and len(tool2.percepts) == 1
