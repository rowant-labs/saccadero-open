"""2026-11-16: the readings' offsets from the readings alone (arm_camera.fit_readings; arm_calib's
arm_joints["readings_fit"])."""

from types import SimpleNamespace

import numpy as np
import pytest

from rowant_lab.libero_env import arm_camera as AC
from rowant_lab.pro2 import arm_calib as ACAL


def _rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    return {"z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]), "y": np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]]),
            "x": np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])}[axis]


class Chain:
    """A seven-joint arm stand-in: six joints move the hand point, the seventh turns about it."""
    AXES = ("z", "y", "z", "y", "x", "y", "z")
    LINKS = ((0, 0, 0.33), (0, 0, 0.0), (0.08, 0, 0.32), (-0.08, 0, 0.0), (0, 0, 0.38), (0.09, 0, 0.0), (0, 0, 0.0))

    def fk(self, q, g):
        R, t = np.eye(3), np.zeros(3)
        for ax, link, a in zip(self.AXES, self.LINKS, q):
            t = t + R @ np.asarray(link, float)
            R = R @ _rot(ax, float(a))
        return {"gripper0_eef": (R, t + R @ np.array([0.0, 0.0, 0.1]))}


def _path(n=24, lines=3):
    """Joint readings of a motion with ``lines`` different directions (in joint space; the hand follows a bent path)."""
    q = np.array([0.1, 0.4, -0.2, -1.9, 0.1, 1.2, 0.5])
    dirs = [np.array([0.02, 0.01, 0.0, 0.015, 0.0, -0.01, 0.0]), np.array([-0.005, 0.02, 0.01, -0.01, 0.015, 0.0, 0.0]),
            np.array([0.01, -0.015, 0.02, 0.0, -0.01, 0.02, 0.0])][:lines]
    out = [q.copy()]
    for d in dirs:
        for _ in range(n // lines):
            q = q + d
            out.append(q.copy())
    return out


DQ = np.array([0.075, -0.06, 0.045, 0.0, 0.0, 0.0, 0.0])
DP = np.array([0.075, -0.06, 0.045])


def _frames(Q, dq, dp, arm=None, drift=0.0, tau=None, vel=True):
    """``tau`` (s): the hand reading shows the hand that long before the joint readings (the joints move at the
    path's own speed, 20 steps a second); ``vel``: the frames carry the joint velocities."""
    arm = arm or Chain()
    g = np.array([0.04, -0.04])
    out = []
    for i, q in enumerate(Q):
        fr = {"joint_pos": (q + dq).astype(np.float64), "gripper_qpos": g}
        v = (Q[i] - Q[i - 1]) * 20.0 if i else np.zeros(7)
        fr["eef_pos"] = (arm.fk(q - (tau or 0.0) * v, g)["gripper0_eef"][1] + dp + drift * i).astype(np.float64)
        if tau is not None and vel:
            fr["joint_vel"] = v
        out.append(fr)
    return out


def _fit(frames, arm=None, **over):
    p = AC.params(over or None)
    return AC.fit_readings(arm or Chain(), frames, [np.asarray(f["joint_pos"], float) for f in frames],
                           [np.asarray(f["gripper_qpos"], float) for f in frames], p)


def test_both_offsets_are_found_from_a_bent_motion():
    out = _fit(_frames(_path(), DQ, DP))
    assert out["accepted"] and out["joints_free"] == 6 and out["rms_mm"] < 0.01
    assert np.allclose(out["delta"], DQ, atol=2e-4) and np.allclose(out["dp"], DP, atol=2e-4)
    assert out["gap_mm"] > 20.0 and out["delta"][6] == 0.0


def test_either_offset_alone_is_found_and_not_mixed_with_the_other():
    out = _fit(_frames(_path(), np.zeros(7), DP))
    assert out["accepted"] and np.allclose(out["delta"], 0.0, atol=2e-4) and np.allclose(out["dp"], DP, atol=2e-4)
    out = _fit(_frames(_path(), DQ, np.zeros(3)))
    assert out["accepted"] and np.allclose(out["delta"], DQ, atol=2e-4) and np.allclose(out["dp"], 0.0, atol=2e-4)


def test_the_hand_readings_lag_is_found_with_the_offsets_when_velocities_are_read():
    out = _fit(_frames(_path(), DQ, DP, tau=0.002))
    assert out["accepted"] and abs(out["tau_ms"] - 2.0) < 0.01 and out["rms_mm"] < 0.01
    assert np.allclose(out["delta"], DQ, atol=2e-4) and np.allclose(out["dp"], DP, atol=2e-4)
    out = _fit(_frames(_path(), DQ, DP, tau=0.0))  # velocities read, no lag: the lag comes out zero
    assert out["accepted"] and abs(out["tau_ms"]) < 0.01 and np.allclose(out["dp"], DP, atol=2e-4)


def test_a_lag_without_velocities_is_not_taken_for_constant_offsets():
    out = _fit(_frames(_path(), DQ, DP, tau=0.004, vel=False))
    assert not out["accepted"] and out["reason"] == "not constant offsets" and "tau_ms" not in out


def test_agreeing_readings_are_left_alone():
    out = _fit(_frames(_path(), np.zeros(7), np.zeros(3)))
    assert not out["accepted"] and out["reason"] == "readings agree" and "delta" not in out


def test_an_arm_that_has_not_moved_is_undetermined_not_refused():
    Q = [_path()[0]] * 12
    out = _fit(_frames(Q, DQ, DP))
    assert not out["accepted"] and out["reason"] == "undetermined"
    assert _fit(_frames(Q[:3], DQ, DP))["reason"] == "undetermined"  # too few steps to try at all


def test_a_drifting_reading_is_not_a_constant_offset():
    out = _fit(_frames(_path(), DQ, DP, drift=np.array([0.002, 0.0, -0.001])))
    assert not out["accepted"] and out["reason"] == "not constant offsets" and out["rms_mm"] > 0.05


def test_offsets_beyond_the_bounds_are_refused():
    out = _fit(_frames(_path(), DQ, np.array([0.5, 0.0, 0.0])))
    assert not out["accepted"] and out["reason"] == "beyond bounds"


def test_the_steps_used_are_spread_over_those_sent():
    fr = _frames(_path(n=90), DQ, DP)
    out = _fit(fr)
    assert out["n_sent"] == len(fr) and out["n"] == 30 and out["accepted"]


def test_server_routes_the_mode_and_the_audit_takes_reading_frames():
    wc = SimpleNamespace(panda=Chain(), arm_samples={}, arm_rc_samples={}, workcell={"arena": []})
    per = SimpleNamespace(cell=lambda s: wc, p={"arm_cam": {}})
    rep = AC.handle(per, {"mode": "readings_fit", "suite": "s", "frames": _frames(_path(), DQ, DP)})
    assert rep["readings_fit"]["accepted"] and rep["ok_reply"]
    from rowant_lab.libero_env import m2_audit as AU

    class Inner:
        def call(self, msg):
            return {}

    au = AU.AuditedClient(Inner())
    fr = {"joint_pos": [0.0] * 7, "gripper_qpos": [0.0, 0.0], "eef_pos": [0.0, 0.0, 1.0]}
    au.call({"op": "arm_cam", "mode": "readings_fit", "suite": "s", "episode": "0123456789abcdef", "frames": [fr]})
    au.call({"op": "arm_cam", "mode": "readings_fit", "suite": "s", "episode": "0123456789abcdef",
             "frames": [{**fr, "joint_vel": [0.0] * 7}]})
    assert au.record()["ok"]
    au.call({"op": "arm_cam", "mode": "readings", "suite": "s", "episode": "0123456789abcdef",
             "frames": [{**fr, "joint_vel": [0.0] * 7}]})  # the velocities only with the fit
    assert not au.record()["ok"]
    au = AU.AuditedClient(Inner())
    au.call({"op": "arm_cam", "mode": "readings_fit", "suite": "s", "episode": "0123456789abcdef",
             "frames": [{**fr, "depth_u16": 0}]})  # never an image
    assert not au.record()["ok"]


def test_the_real_arm_model_from_its_first_motion():
    try:
        from rowant_lab.libero_env import m2_library as LB

        panda = LB.LibraryReader(allow_panda=True).panda("libero_object")
    except Exception as exc:  # the export is on the cold volume
        pytest.skip(f"no arm model here: {type(exc).__name__}")
    g = np.array([0.04, -0.04])
    q = np.array([0.0, -0.16, 0.0, -2.44, 0.0, 2.23, 0.79])
    Q = [q.copy()]
    for d in (np.array([0.01, 0.012, 0.0, 0.008, 0.0, -0.004, 0.0]), np.array([-0.004, 0.01, 0.006, -0.012, 0.004, 0.0, 0.0]),
              np.array([0.006, -0.008, 0.01, 0.004, -0.006, 0.008, 0.0])):
        for _ in range(6):
            q = q + d
            Q.append(q.copy())
    out = _fit(_frames(Q, DQ, DP, arm=panda), arm=panda)
    assert out["accepted"], out
    assert np.allclose(out["dp"], DP, atol=3e-4) and np.allclose(out["delta"], DQ, atol=1e-3)
    out = _fit(_frames(Q, DQ, DP, arm=panda, tau=0.002), arm=panda)  # as the simulator's readings come
    assert out["accepted"] and abs(out["tau_ms"] - 2.0) < 0.02, out
    assert np.allclose(out["dp"], DP, atol=3e-4) and np.allclose(out["delta"], DQ, atol=1e-3)


# ---------------------------------------------------------------------------------------------- the robot's side
class ColourCam:
    def __init__(self):
        self.override = {"T": None}

    def __call__(self, camera, width, step):
        return {"depth_u16": np.zeros((4, 4), np.uint16), "depth_scale_mm": 0.1, "K": np.eye(3),
                "T_world_from_camera": np.eye(4), "rgb": np.zeros((4, 4, 3), np.uint8)}


class DepthCam:
    __call__ = ColourCam.__call__


class SeqClient:
    def __init__(self, replies):
        self.replies, self.msgs = list(replies), []

    def call(self, msg):
        self.msgs.append(msg)
        return self.replies.pop(0)


def _obs(x):
    return {"robot0_eef_pos": [x, 0.0, 1.0], "robot0_joint_pos": [0.1 + x] + [0.1] * 6,
            "robot0_gripper_qpos": [0.04, -0.04], "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0]}


def _run(cam, replies, params, xs=(0.07, 0.14, 0.3)):
    cl = SeqClient(replies)
    c = ACAL.ArmCalib(cam, cl, "libero_object", "ep", params)
    c.start(_obs(0.0))
    return c, cl, [c.after_step(_obs(x), t + 1) for t, x in enumerate(xs)]


ON = {"arm_joints": {"guard_mm": 20.0, "readings_fit": {}}}
FOUND = {"readings_fit": {"accepted": True, "delta": list(DQ), "dp": list(DP), "rms_mm": 0.0, "cond": 500.0}}
WAIT = {"readings_fit": {"accepted": False, "reason": "undetermined", "cond": 1e7}}
NOT = {"readings_fit": {"accepted": False, "reason": "not constant offsets", "rms_mm": 3.0}}
APART = {"readings": {"max_mm": 135.0}}


def test_contradicting_readings_are_fixed_from_the_readings_before_any_picture_fit():
    cam = ColourCam()
    c, cl, outs = _run(cam, [APART, FOUND], ON)
    assert [m["mode"] for m in cl.msgs] == ["readings", "readings_fit"] and outs == [False, True, False]
    assert c.rec["status"] == "joints fixed" and c.rec["by"] == "readings" and cam.override["T"] is None
    assert np.allclose(c.fix["dq"], DQ) and np.allclose(c.fix["dp"], DP)
    sent = cl.msgs[1]["frames"]  # every step's readings so far, proprioception only
    assert len(sent) == 3 and all(set(f) == {"joint_pos", "gripper_qpos", "eef_pos"} for f in sent)
    fixed = c.correct(_obs(0.5))
    assert np.allclose(fixed["robot0_eef_pos"], np.array([0.5, 0.0, 1.0]) - DP)
    assert np.allclose(fixed["robot0_joint_pos"], np.array([0.6] + [0.1] * 6) - DQ)


def test_undetermined_offsets_are_asked_again_as_the_robot_moves_and_the_camera_is_left_alone():
    cam = ColourCam()
    xs = (0.07, 0.14, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55)
    c, cl, outs = _run(cam, [APART, WAIT, WAIT, FOUND], ON, xs=xs)
    assert [m["mode"] for m in cl.msgs] == ["readings", "readings_fit", "readings_fit", "readings_fit"]
    assert outs == [False, False, False, False, False, False, False, False, False, True]  # steps 2, 6, 10
    assert [len(m["frames"]) for m in cl.msgs[1:]] == [3, 7, 11]
    assert c.rec["status"] == "joints fixed" and c.rec["readings_fit"]["tries"] == 3 and c.rec["readings_fit"]["at"] == 10
    assert cam.override["T"] is None and not c.can_move(None, False, 11, 300)


def test_waiting_ends_at_its_last_step_and_the_robot_may_then_make_its_own_motion_once():
    cam = ColourCam()
    on = {"arm_joints": {"guard_mm": 20.0, "readings_fit": {"until": 6}}}
    c, cl, outs = _run(cam, [APART, WAIT], on, xs=(0.07, 0.14))
    assert c.rec["status"] == "readings fit waiting" and c.can_move(None, False, 3, 300)  # no agent: its own motion
    cl.replies += [WAIT]
    assert not any(c.after_step(_obs(0.14), t) for t in (3, 4, 5, 6))
    assert c.rec["status"] == "readings fit undetermined" and c.fix is None and not c._rfit_wait


def test_readings_that_no_constant_offset_explains_take_the_old_route():
    cam = ColourCam()
    joints = {"gate": {"moved": True}, "joints": {"accepted": True, "delta": list(DQ), "eef_fk": [[0.0, 0.0, 1.0]] * 3}}
    c, cl, outs = _run(cam, [APART, NOT, joints], ON)
    assert [m["mode"] for m in cl.msgs] == ["readings", "readings_fit", "gate_joints"] and outs == [False, True, False]
    assert c.rec["status"] == "joints fixed" and "by" not in c.rec and c.rec["readings_fit"]["reason"] == "not constant offsets"


def test_the_joint_velocities_go_with_the_readings_when_the_robot_has_them():
    cl = SeqClient([APART, FOUND])
    c = ACAL.ArmCalib(ColourCam(), cl, "libero_object", "ep", ON)
    c.start({**_obs(0.0), "robot0_joint_vel": [0.0] * 7})
    outs = [c.after_step({**_obs(x), "robot0_joint_vel": [0.5] * 7}, t + 1) for t, x in enumerate((0.07, 0.14))]
    sent = cl.msgs[1]["frames"]
    assert outs == [False, True] and all(set(f) == {"joint_pos", "gripper_qpos", "eef_pos", "joint_vel"} for f in sent)
    assert np.allclose(sent[1]["joint_vel"], 0.5) and np.allclose(sent[0]["joint_vel"], 0.0)


def test_agreeing_readings_never_ask_for_the_fit():
    cam = ColourCam()
    refused = {"gate": {"moved": True}, "fit": {"accepted": False, "T_world_from_camera": np.eye(4)}}
    c, cl, outs = _run(cam, [{"readings": {"max_mm": 0.0}}, refused], ON)
    assert [m["mode"] for m in cl.msgs] == ["readings", "gate_fit"] and c.fix is None and "readings_fit" not in c.rec


def test_the_depth_line_asks_when_its_readings_are_apart():
    c, cl, outs = _run(DepthCam(), [APART, FOUND], {"arm_joints": {"readings_fit": {}}})
    assert [m["mode"] for m in cl.msgs] == ["readings", "readings_fit"] and outs == [False, True, False]
    kept = {"gate": {"moved": False}}
    c, cl, outs = _run(DepthCam(), [{"readings": {"max_mm": 0.2}}, kept], {"arm_joints": {"readings_fit": {}}})
    assert [m["mode"] for m in cl.msgs] == ["readings", "gate_joints"] and c.rec["status"] == "kept"


def test_without_the_option_no_readings_are_kept_and_the_requests_are_the_old_ones():
    cam = ColourCam()
    rep = {"gate": {"moved": True}, "joints": {"accepted": True, "delta": list(DQ), "eef_fk": [[0.0, 0.0, 1.0]] * 3}}
    c, cl, outs = _run(cam, [APART, rep], {"arm_joints": {"guard_mm": 20.0}})
    assert [m["mode"] for m in cl.msgs] == ["readings", "gate_joints"] and c.samples == [] and c.rfit is None
