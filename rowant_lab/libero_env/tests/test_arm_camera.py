"""arm_camera (item 2): the check and the fit on synthetic depth renders of a three-box arm over a floor."""

import numpy as np
import pytest

from rowant_lab.libero_env import arm_camera as AC

PX = 256
FOVY = 45.0


def _box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    # vertex index = 4 ix + 2 iy + iz; two triangles per face, wound outward
    F = np.array([[0, 1, 3], [0, 3, 2],  # x = lo
                  [4, 6, 7], [4, 7, 5],  # x = hi
                  [0, 4, 5], [0, 5, 1],  # y = lo
                  [2, 3, 7], [2, 7, 6],  # y = hi
                  [0, 2, 6], [0, 6, 4],  # z = lo
                  [1, 5, 7], [1, 7, 3]])  # z = hi
    return V, F


def _axis_angle(axis, ang):
    return AC.rot(axis, np.degrees(ang))


class FakeArm:
    """base (static), upper (turns about z by q0, at the base's top), fore (turns about y by q1, at the upper's end)."""

    def __init__(self):
        self.order = ["base", "upper", "fore"]
        self.bodies = {"base": {"geoms": ["gb"], "parent": "world"}, "upper": {"geoms": ["gu"], "parent": "base"},
                       "fore": {"geoms": ["gf"], "parent": "upper"}}
        self.arrays = {}
        for k, (lo, hi) in {"gb": ((-0.08, -0.08, 0.0), (0.08, 0.08, 0.30)),
                            "gu": ((-0.04, -0.04, 0.0), (0.30, 0.04, 0.08)),
                            "gf": ((-0.03, -0.03, -0.25), (0.03, 0.03, 0.0))}.items():
            V, F = _box(lo, hi)
            self.arrays[f"{k}__V"], self.arrays[f"{k}__F"] = V, F

    def fk(self, q, g):
        Rb, tb = np.eye(3), np.zeros(3)
        Ru = _axis_angle([0, 0, 1], q[0])
        tu = tb + np.array([0.0, 0.0, 0.30])
        Rf = Ru @ _axis_angle([0, 1, 0], q[1])
        tf = tu + Ru @ np.array([0.28, 0.0, 0.04])
        return {"base": (Rb, tb), "upper": (Ru, tu), "fore": (Rf, tf)}


def _look_at(eye, target):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    zc = target - eye
    zc /= np.linalg.norm(zc)
    xc = np.cross(zc, [0, 0, 1.0])
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    T = np.eye(4)
    T[:3, :3] = np.stack([xc, yc, zc], 1)
    T[:3, 3] = eye
    return T


def _K():
    f = 0.5 * PX / np.tan(np.radians(FOVY) / 2)
    return np.array([[f, 0, PX / 2], [0, f, PX / 2], [0, 0, 1.0]])


def _render(arm, q, T, K):
    """Depth (m) at every pixel: the arm's dense samples (nearest per pixel), else the floor z = 0."""
    dense = AC.model_samples(arm, 0.002, seed=1)
    P, _N = AC.arm_points(arm, dense, q, [0.0, 0.0])
    X = (P - T[:3, 3]) @ T[:3, :3]
    u = np.floor(K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2]).astype(int)
    v = np.floor(K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]).astype(int)
    ok = (X[:, 2] > 0.05) & (u >= 0) & (u < PX) & (v >= 0) & (v < PX)
    zb = np.full(PX * PX, np.inf)
    np.minimum.at(zb, v[ok] * PX + u[ok], X[ok, 2])
    vv, uu = np.mgrid[0:PX, 0:PX].astype(float)
    ray = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
    dw = ray @ T[:3, :3].T
    s = -T[2, 3] / dw[..., 2]  # the floor: camera z along the ray where world z = 0
    floor = np.where(s > 0, s, np.inf)
    z = np.minimum(zb.reshape(PX, PX), floor)
    return z


def _frame(z, T, K):
    return {"depth_u16": np.clip(np.rint(np.nan_to_num(z, posinf=0) / 1e-4), 0, 65535).astype(np.uint16),
            "depth_scale_mm": 0.1, "K": K, "T_world_from_camera": T}


def _P(**kw):
    return AC.params({"stride": 2, "check_min_cells": 30, **kw})


@pytest.fixture(scope="module")
def arm():
    return FakeArm()


@pytest.fixture(scope="module")
def samples(arm):
    return AC.model_samples(arm, 0.006)


def test_boxes_are_wound_outward(arm):
    for k in ("gb", "gu", "gf"):
        V, F = arm.arrays[f"{k}__V"], arm.arrays[f"{k}__F"]
        vol = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6
        assert vol > 0


def test_model_samples_have_unit_outward_normals(arm, samples):
    P, N = samples["upper"]
    assert np.allclose(np.linalg.norm(N, axis=1), 1.0)
    c = P.mean(0)
    assert ((P - c) * N).sum(1).mean() > 0
    area = 2 * (0.34 * 0.08 + 0.34 * 0.08 + 0.08 * 0.08)
    assert abs(len(P) - area / 0.006 ** 2) <= 2  # face samples only, no mesh vertex


def test_check_accepts_the_true_camera_and_refuses_a_turned_one(arm, samples):
    K = _K()
    T = _look_at([1.0, 0.6, 0.9], [0.1, 0.0, 0.25])
    q = [0.3, 0.4]
    z = _render(arm, q, T, K)
    good = AC.check(arm, samples, _frame(z, T, K), q, [0, 0], _P())
    assert good["ok"] and good["agree"] > 0.9
    Tb = T.copy()
    Tb[:3, :3] = T[:3, :3] @ AC.rot([0, 1, 0], 12.0)
    bad = AC.check(arm, samples, _frame(z, Tb, K), q, [0, 0], _P())
    assert not bad["ok"]


def test_check_does_not_count_an_occluded_arm_against_the_camera(arm, samples):
    K = _K()
    T = _look_at([1.0, 0.6, 0.9], [0.1, 0.0, 0.25])
    q = [0.3, 0.4]
    z = _render(arm, q, T, K)
    z[:, : PX // 2] = np.minimum(z[:, : PX // 2], 0.3)  # something near the camera hides the left half
    rep = AC.check(arm, samples, _frame(z, T, K), q, [0, 0], _P())
    assert rep["n_front"] > 0 and rep["agree"] > 0.9


def test_fit_recovers_a_moved_camera(arm, samples):
    K = _K()
    T_true = _look_at([1.0, 0.6, 0.9], [0.1, 0.0, 0.25])
    T_st = T_true.copy()
    T_st[:3, :3] = T_true[:3, :3] @ AC.rot([0, 1, 0], 10.0) @ AC.rot([1, 0, 0], -4.0)
    T_st[:3, 3] = T_true[:3, 3] + np.array([0.05, -0.08, 0.04])
    qs = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.6], [-0.4, 0.5]]
    frames = [_frame(_render(arm, q, T_true, K), T_st, K) for q in qs]
    rep = AC.fit(arm, samples, frames, qs, [[0, 0]] * 4, _P())
    shift, turn = AC.pose_diff(rep["T_world_from_camera"], T_true)
    assert rep["accepted"]
    assert shift < 0.01 and turn < 0.5
    assert rep["cost_m"] < rep["cost_assumed_m"]


def test_fit_refuses_when_nothing_moved(arm, samples):
    K = _K()
    T = _look_at([1.0, 0.6, 0.9], [0.1, 0.0, 0.25])
    z = _render(arm, [0.2, 0.2], T, K)
    rep = AC.fit(arm, samples, [_frame(z, T, K)] * 3, [[0.2, 0.2]] * 3, [[0, 0]] * 3, _P())
    assert rep["accepted"] is False and rep["reason"] == "no arm motion seen"


def test_masks_drop_lone_pixels():
    Z = np.full((3, 10, 10), 1.0)
    Z[0, 5, 5] = 0.5  # one lone nearer pixel
    Z[1, 2:5, 2:5] = 0.5  # a 3x3 patch
    M = AC.arm_masks(Z, AC.params())
    assert not M[0].any()
    assert M[1, 3, 3] and M[1].sum() >= 4


def _scene(arm, qs, T_true, T_assumed, K):
    return [_frame(_render(arm, q, T_true, K), T_assumed, K) for q in qs]


QS = [[0.0, 0.0], [0.5, 0.3], [-0.5, 0.6]]


def test_gate_keeps_the_true_camera(arm, samples):
    K = _K()
    T = _look_at([1.0, 0.6, 0.9], [0.1, 0.0, 0.25])
    rep = AC.gate(arm, samples, _scene(arm, QS, T, T, K), QS, [[0, 0]] * 3, _P(motion_min_px=50))
    assert rep["moved"] is False and rep["cost_m"] < 0.003 and rep["share_1cm"] > 0.9


def test_gate_flags_a_moved_camera(arm, samples):
    K = _K()
    T = _look_at([1.0, 0.6, 0.9], [0.1, 0.0, 0.25])
    Tm = T.copy()
    Tm[:3, :3] = AC.rot([0, 0, 1], 12.0) @ T[:3, :3]
    rep = AC.gate(arm, samples, _scene(arm, QS, T, Tm, K), QS, [[0, 0]] * 3, _P(motion_min_px=50))
    assert rep["moved"] is True


def test_gate_does_not_decide_without_motion(arm, samples):
    K = _K()
    T = _look_at([1.0, 0.6, 0.9], [0.1, 0.0, 0.25])
    q = [[0.2, 0.2]] * 3
    rep = AC.gate(arm, samples, _scene(arm, q, T, T, K), q, [[0, 0]] * 3, _P())
    assert rep["moved"] is None


def test_plane_locked_fit_recovers_a_turned_camera(arm, samples):
    """The floor (z = 0) fixes the tilt and height; the arm gives the turn and the horizontal position."""
    K = _K()
    T_true = _look_at([1.0, 0.6, 0.9], [0.1, 0.0, 0.25])
    T_st = T_true.copy()
    T_st[:3, :3] = AC.rot([0, 0, 1], -10.0) @ T_true[:3, :3]
    T_st[:3, 3] = AC.rot([0, 0, 1], -10.0) @ T_true[:3, 3]  # an orbit about the vertical through the origin
    qs = QS + [[-0.4, 0.5]]
    rep = AC.fit(arm, samples, _scene(arm, qs, T_true, T_st, K), qs, [[0, 0]] * 4, _P(),
                 supports=[0.0], rc_samples=AC.model_samples(arm, 0.008))
    assert rep["mode"] == "plane" and rep["accepted"]
    shift, turn = AC.pose_diff(rep["T_world_from_camera"], T_true)
    assert shift < 0.01 and turn < 0.5


def test_support_heights_from_the_workcell():
    wc = {"arena": [{"type": "plane", "normal": [0, 0, 1], "point": [0, 0, 0.0]},
                    {"type": "box", "centre": [0, 0, 0.875], "half": [0.5, 0.6, 0.025]}]}
    assert AC.support_heights(wc) == [0.0, 0.9]


def test_fit_plane_finds_the_dominant_plane():
    rng = np.random.default_rng(0)
    P = np.c_[rng.uniform(-1, 1, (500, 2)), np.zeros(500)] + [0, 0, -1.0]  # z = -1 in the camera frame
    Q = rng.uniform(-1, 1, (100, 3))
    n, d, share = AC.fit_plane(np.r_[P, Q], 0.01, 100)
    assert abs(abs(n[2]) - 1) < 1e-3 and abs(d - 1.0) < 2e-3 and share > 0.8


def test_render_cost_does_not_reward_drawing_the_arm_out_of_view(arm):
    K = _K()
    T = _look_at([1.0, 0.6, 0.9], [0.1, 0.0, 0.25])
    q = [0.3, 0.4]
    Zg = [AC.grid(AC.depth_m(_frame(_render(arm, q, T, K), T, K)), 2)[0]]
    rcs = AC.model_samples(arm, 0.008)
    arms = [AC.arm_points(arm, rcs, q, [0, 0])]
    tau = 0.03
    m = int(0.5 * AC.rc_cells(T, arms, K, Zg[0].shape, 2))
    good = AC.rc_cost(T, Zg, arms, K, 2, tau, m)
    away = T.copy()
    away[:3, :3] = AC.rot([0, 0, 1], 60.0) @ T[:3, :3]  # the arm leaves the image
    assert m > 0 and good < 0.3 * tau * tau
    assert AC.rc_cells(away, arms, K, Zg[0].shape, 2) < m
    assert AC.rc_cost(away, Zg, arms, K, 2, tau, m) == tau * tau


def test_focal_sweep_recovers_a_zoomed_image_and_is_off_by_default(arm, samples):
    """Item 2b: an image taken with a longer focal length than assumed (a zoom), from a turned camera. With
    ``focal_scales`` the plane-locked fit finds the turn and the focal factor; without it the result is as before."""
    K = _K()
    K_true = AC.scaled_K(K, 1.15)
    T_true = _look_at([1.0, 0.6, 0.9], [0.1, 0.0, 0.25])
    T_st = T_true.copy()
    T_st[:3, :3] = AC.rot([0, 0, 1], -8.0) @ T_true[:3, :3]
    T_st[:3, 3] = AC.rot([0, 0, 1], -8.0) @ T_true[:3, 3]
    qs = QS + [[-0.4, 0.5]]
    frames = [_frame(_render(arm, q, T_true, K_true), T_st, K) for q in qs]
    rcs = AC.model_samples(arm, 0.008)
    rep = AC.fit(arm, samples, frames, qs, [[0, 0]] * 4, _P(focal_scales=(0.9, 1.0, 1.1, 1.2, 1.3)),
                 supports=[0.0], rc_samples=rcs)
    assert rep["mode"] == "plane" and rep["accepted"]
    assert abs(rep["focal_scale"] - 1.15) < 0.015
    shift, turn = AC.pose_diff(rep["T_world_from_camera"], T_true)
    assert shift < 0.02 and turn < 1.0
    assert [r["f"] for r in rep["focal_sweep"]] == [0.9, 1.0, 1.1, 1.2, 1.3]
    off = AC.fit(arm, samples, frames, qs, [[0, 0]] * 4, _P(), supports=[0.0], rc_samples=rcs)
    assert "focal_scale" not in off and "focal_sweep" not in off


def test_focal_sweep_keeps_a_right_camera_within_precision(arm, samples):
    K = _K()
    T = _look_at([1.0, 0.6, 0.9], [0.1, 0.0, 0.25])
    qs = QS + [[-0.4, 0.5]]
    frames = [_frame(_render(arm, q, T, K), T, K) for q in qs]
    rep = AC.fit(arm, samples, frames, qs, [[0, 0]] * 4, _P(focal_scales=(0.9, 1.0, 1.1)), supports=[0.0],
                 rc_samples=AC.model_samples(arm, 0.008))
    assert not rep["accepted"]
    assert abs(rep["focal_scale"] - 1.0) < 0.03


def test_gate_fit2_uses_the_second_overrides(monkeypatch):
    """item 2b: mode gate_fit2 runs the gate and fit with the params' ``second`` overrides; gate_fit without them."""
    from types import SimpleNamespace

    from rowant_lab.libero_env import arm_camera as AC

    seen = []
    monkeypatch.setattr(AC, "gate", lambda panda, s, frames, qs, gs, p: seen.append(("gate", p.get("focal_scales"), p.get("motion_cost_m"))) or {"moved": True})
    monkeypatch.setattr(AC, "fit", lambda panda, s, frames, qs, gs, p, supports=None, rc_samples=None: seen.append(("fit", p.get("focal_scales"), p.get("motion_cost_m"))) or {"accepted": False})
    wc = SimpleNamespace(panda=None, arm_samples={}, arm_rc_samples={}, workcell={"arena": []})
    per = SimpleNamespace(p={"arm_cam": {"second": {"focal_scales": [0.9, 1.0, 1.1]}}}, cell=lambda suite: wc)
    fr = {"joint_pos": [0.0] * 7, "gripper_qpos": [0.04, -0.04]}
    msg = {"op": "arm_cam", "suite": "libero_object", "episode": "e", "frames": [fr, fr, fr]}
    AC.handle(per, {**msg, "mode": "gate_fit"})
    AC.handle(per, {**msg, "mode": "gate_fit2"})
    base = AC.ARM_CAM["motion_cost_m"]
    assert seen == [("gate", None, base), ("fit", None, base), ("gate", [0.9, 1.0, 1.1], base), ("fit", [0.9, 1.0, 1.1], base)]


# ---- 2026-11-15: the image shifted and zoomed, the camera where it was (fit_image)
def _shifted_K(K, zoom, shift):
    """The lens of a picture zoomed by ``zoom`` about the principal point and then shifted by ``shift`` pixels."""
    Ko = K.copy()
    Ko[0, 0] *= zoom
    Ko[1, 1] *= zoom
    Ko[0, 2] += shift[0]
    Ko[1, 2] += shift[1]
    return Ko


def _image_p():
    return _P(img_min_px=50, img_step_px=4.0, img_min_shift_px=3.0, img_min_zoom=0.015)


def _image_frames(arm, zoom, shift):
    K = _K()
    T = _look_at([1.0, 0.6, 0.9], [0.1, 0.0, 0.25])
    return [_frame(_render(arm, q, T, _shifted_K(K, zoom, shift)), T, K) for q in QS]  # told the stored lens


def test_image_fit_recovers_a_shift_and_a_zoom_from_depth_masks(arm, samples):
    g0 = [[0, 0]] * 3
    frames = _image_frames(arm, 1.0, (-30.0, 30.0))
    assert AC.gate(arm, samples, frames, QS, g0, _P(motion_min_px=50))["moved"] is True
    rep = AC.fit_image(arm, samples, frames, QS, g0, _image_p())
    assert rep["accepted"] and abs(rep["zoom"] - 1.0) < 0.05
    assert abs(rep["shift_px"][0] + 30.0) < 4.0 and abs(rep["shift_px"][1] - 30.0) < 4.0
    assert rep["iou"] > 1.5 * rep["iou0"] and rep["iou"] > 0.6
    rep = AC.fit_image(arm, samples, _image_frames(arm, 0.86, (10.0, 6.0)), QS, g0, _image_p())
    # (on this small arm, two grid cells across its length are 5% of zoom; the benchmark's arm spans four times the cells)
    assert rep["accepted"] and abs(rep["zoom"] - 0.86) < 0.05
    assert abs(rep["shift_px"][0] - 10.0) < 6.0 and abs(rep["shift_px"][1] - 6.0) < 6.0
    rep = AC.fit_image(arm, samples, _image_frames(arm, 0.67, (-26.6, -26.6)), QS, g0, _image_p())  # as S16 on Goal
    assert rep["accepted"] and abs(rep["zoom"] - 0.67) < 0.05 and abs(rep["shift_px"][0] + 26.6) < 6.0
    rep = AC.fit_image(arm, samples, _image_frames(arm, 0.95, (22.0, -14.0)), QS, g0, _image_p())
    assert rep["accepted"] and abs(rep["zoom"] - 0.95) < 0.05
    assert abs(rep["shift_px"][0] - 22.0) < 6.0 and abs(rep["shift_px"][1] + 14.0) < 6.0


def test_image_fit_leaves_a_right_picture_alone_and_reads_the_robots_masks(arm, samples):
    g0 = [[0, 0]] * 3
    right = _image_frames(arm, 1.0, (0.0, 0.0))
    rep = AC.fit_image(arm, samples, right, QS, g0, _image_p())
    assert not rep["accepted"] and rep["reason"] == "no change" and rep["iou0"] > 0.7
    # masks the robot made itself (from the colour images): any change, so every pixel a moving link covers in some frame
    K, T = np.asarray(right[0]["K"]), np.asarray(right[0]["T_world_from_camera"])
    Z = np.stack([AC.grid(AC.depth_m(f), 2)[0] for f in right])
    floor = np.stack([AC.grid(AC.depth_m(_frame(_render_floor(T, K), T, K)), 2)[0]] * 3)
    cov = Z < floor - 0.02  # the arm's pixels in each frame
    for changed in (cov.any(0), cov.any(0) & ~cov.all(0)):  # a marked arm changes all over; a plain one at its edges
        moved = np.roll(np.roll(changed, 6, axis=1), -4, axis=0)  # 12 px right, 8 px up at stride 2
        with_masks = [{**f, "mask": moved} for f in right]
        rep = AC.fit_image(arm, samples, with_masks, QS, g0, _image_p())
        assert rep["accepted"] and abs(rep["shift_px"][0] - 12.0) < 4.0 and abs(rep["shift_px"][1] + 8.0) < 4.0
        assert abs(rep["zoom"] - 1.0) < 0.05
    # a strip of the picture that changes with the arm and is not the arm (the wider S16 picture's bottom edge)
    strip = np.roll(np.roll(cov.any(0), 6, axis=1), -4, axis=0)
    strip[-3:, 8:40] = True
    rep = AC.fit_image(arm, samples, [{**f, "mask": strip} for f in right], QS, g0, _image_p())
    assert rep["accepted"] and abs(rep["shift_px"][0] - 12.0) < 4.0 and abs(rep["shift_px"][1] + 8.0) < 4.0
    still = [right[0]] * 3
    assert AC.fit_image(arm, samples, still, [QS[0]] * 3, g0, _image_p())["reason"] == "too little motion"
    # the probe's question: the overlap at a named zoom and shift
    rep = AC.fit_image(arm, samples, right, QS, g0, {**_image_p(), "img_probe": [1.0, 0.0, 0.0]})
    assert rep["probe_iou"] == rep["iou0"]


def _render_floor(T, K):
    vv, uu = np.mgrid[0:PX, 0:PX].astype(float)
    ray = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
    dw = ray @ T[:3, :3].T
    s = -T[2, 3] / dw[..., 2]
    return np.where(s > 0, s, np.inf)


def test_project_matches_lift(arm):
    K = _K()
    T = _look_at([1.0, 0.6, 0.9], [0.1, 0.0, 0.25])
    X = AC.lift(np.array([1.2, 0.8]), np.array([40.0, 200.0]), np.array([100.0, 30.0]), K)
    u, v = AC.project(X @ T[:3, :3].T + T[:3, 3], T, K)
    assert np.allclose(u, [40.0, 200.0]) and np.allclose(v, [100.0, 30.0])
