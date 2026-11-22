"""Item 4 (2026-11-14): the joint readings' offsets from the arm (arm_calib's arm_joints; arm_camera's joint modes)."""

from types import SimpleNamespace

import numpy as np

from rowant_lab.libero_env import arm_camera as AC
from rowant_lab.pro2 import arm_calib as ACAL


class DepthCam:  # the depth line's camera tool: no ``override`` (its camera is right)
    def __call__(self, camera, width, step):
        return {"depth_u16": np.zeros((4, 4), np.uint16), "depth_scale_mm": 0.1, "K": np.eye(3),
                "T_world_from_camera": np.eye(4), "rgb": np.zeros((4, 4, 3), np.uint8)}


class ColourCam(DepthCam):
    def __init__(self):
        self.override = {"T": None}


class SeqClient:
    def __init__(self, replies):
        self.replies, self.msgs = list(replies), []

    def call(self, msg):
        self.msgs.append(msg)
        return self.replies.pop(0)


def _obs(x):
    return {"robot0_eef_pos": [x, 0.0, 1.0], "robot0_joint_pos": [0.1] * 7, "robot0_gripper_qpos": [0.04, -0.04],
            "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0]}


def _run(cam, replies, params, xs=(0.07, 0.14, 0.3)):
    cl = SeqClient(replies)
    c = ACAL.ArmCalib(cam, cl, "libero_object", "ep", params)
    c.start(_obs(0.0))
    return c, cl, [c.after_step(_obs(x), t + 1) for t, x in enumerate(xs)]


DELTA = [0.075, -0.06, 0.045, 0.0, 0.0, 0.0, 0.0]
# frames taken with the hand reported at x = 0.0, 0.07 and 0.14; the fitted joints put it 3 cm lower in z, 1 cm across
FK = [[0.0 - 0.01, 0.0, 0.97], [0.07 - 0.01, 0.0, 0.97], [0.14 - 0.01, 0.0, 0.97]]


def test_depth_line_gate_then_joints_accepted_sets_the_fix():
    rep = {"gate": {"moved": True, "cost_m": 0.03}, "joints": {"accepted": True, "delta": DELTA, "eef_fk": FK,
                                                              "cost_m": 0.003, "cost_before_m": 0.03}}
    c, cl, outs = _run(DepthCam(), [rep], {"arm_joints": {}})
    assert [m["mode"] for m in cl.msgs] == ["gate_joints"] and outs == [False, True, False]
    assert c.rec["status"] == "joints fixed" and np.allclose(c.fix["dq"], DELTA)
    assert np.allclose(c.fix["dp"], [0.01, 0.0, 0.03]) and c.rec["joints"]["hand_offset_mm"] == [10.0, 0.0, 30.0]
    fixed = c.correct(_obs(0.5))
    assert np.allclose(fixed["robot0_joint_pos"], np.array([0.1] * 7) - DELTA)
    assert np.allclose(fixed["robot0_eef_pos"], [0.49, 0.0, 0.97])
    assert fixed["robot0_eef_quat"] == [0.0, 0.0, 0.0, 1.0] and _obs(0.5)["robot0_eef_pos"] == [0.5, 0.0, 1.0]


def test_depth_line_kept_or_refused_changes_nothing():
    c, cl, outs = _run(DepthCam(), [{"gate": {"moved": False}}], {"arm_joints": {}})
    assert not any(outs) and c.fix is None and c.rec["status"] == "kept" and c.correct(_obs(0.2)) == _obs(0.2)
    rep = {"gate": {"moved": True}, "joints": {"accepted": False, "delta": DELTA, "eef_fk": FK}}
    c, cl, outs = _run(DepthCam(), [rep], {"arm_joints": {}})
    assert not any(outs) and c.fix is None and c.rec["status"] == "joints refused"


def test_colour_line_joints_follow_a_refused_camera_fit():
    refused = {"gate": {"moved": True}, "fit": {"accepted": False, "T_world_from_camera": np.eye(4)}}
    joints = {"joints": {"accepted": True, "delta": DELTA, "eef_fk": FK}}
    cam = ColourCam()
    c, cl, outs = _run(cam, [refused, joints], {"arm_joints": {}})
    assert [m["mode"] for m in cl.msgs] == ["gate_fit", "joints"] and outs == [False, True, False]
    assert c.rec["status"] == "joints fixed" and cam.override["T"] is None
    c2, cl2, outs2 = _run(ColourCam(), [refused], {})  # without arm_joints: as item 2
    assert [m["mode"] for m in cl2.msgs] == ["gate_fit"] and not any(outs2) and c2.fix is None


def test_server_routes_the_joint_modes(monkeypatch):
    seen = []
    monkeypatch.setattr(AC, "gate", lambda *a, **k: {"moved": True})
    monkeypatch.setattr(AC, "fit_joints", lambda *a, **k: seen.append(a[5]) or {"accepted": False})
    wc = SimpleNamespace(panda=None, arm_samples={}, arm_rc_samples={}, workcell={"arena": []})
    per = SimpleNamespace(cell=lambda s: wc, p={"arm_cam": {}})
    fr = {"joint_pos": [0.0] * 7, "gripper_qpos": [0.0, 0.0]}
    monkeypatch.setattr(AC, "support_heights", lambda w: [])
    rep = AC.handle(per, {"mode": "gate_joints", "suite": "s", "frames": [fr]})
    assert rep["gate"]["moved"] and rep["joints"] == {"accepted": False} and rep["ok_reply"]
    rep = AC.handle(per, {"mode": "joints", "suite": "s", "frames": [fr]})
    assert rep["joints"] == {"accepted": False} and "gate" not in rep and len(seen) == 2


# -- the colour line's guard (arm_joints["guard_mm"]): the joint fit only when the readings contradict each other
GUARD = {"arm_joints": {"guard_mm": 20.0}}
REFUSED = {"gate": {"moved": True}, "fit": {"accepted": False, "T_world_from_camera": np.eye(4)}}


def test_guard_consistent_readings_take_the_camera_route_only():
    cam = ColourCam()
    c, cl, outs = _run(cam, [{"readings": {"dev_mm": [0.0] * 3, "max_mm": 0.0}}, REFUSED], GUARD)
    assert [m["mode"] for m in cl.msgs] == ["readings", "gate_fit"] and not any(outs)
    assert c.fix is None and c.rec["status"] == "fit refused" and c.rec["readings"]["max_mm"] == 0.0
    sent = cl.msgs[0]["frames"]
    assert len(sent) == 3 and all(set(f) == {"joint_pos", "gripper_qpos", "eef_pos"} for f in sent)
    assert np.allclose([f["eef_pos"][0] for f in sent], [0.0, 0.07, 0.14])


def test_guard_contradicting_readings_fix_the_joints_with_the_camera_kept():
    cam = ColourCam()
    rep = {"gate": {"moved": True}, "joints": {"accepted": True, "delta": DELTA, "eef_fk": FK}}
    c, cl, outs = _run(cam, [{"readings": {"max_mm": 135.0}}, rep], GUARD)
    assert [m["mode"] for m in cl.msgs] == ["readings", "gate_joints"] and outs == [False, True, False]
    assert c.rec["status"] == "joints fixed" and cam.override["T"] is None and np.allclose(c.fix["dq"], DELTA)


def test_guard_refused_joint_fit_falls_back_to_the_camera_route():
    cam = ColourCam()
    rep = {"gate": {"moved": True}, "joints": {"accepted": False, "delta": DELTA, "eef_fk": FK}}
    c, cl, outs = _run(cam, [{"readings": {"max_mm": 135.0}}, rep, REFUSED], GUARD)
    assert [m["mode"] for m in cl.msgs] == ["readings", "gate_joints", "gate_fit"] and not any(outs)
    assert c.fix is None and c.rec["joints_status"] == "joints refused" and c.rec["status"] == "fit refused"
    accepted = {"gate": {"moved": True}, "fit": {"accepted": True, "T_world_from_camera": 2 * np.eye(4)}}
    cam = ColourCam()
    c, cl, outs = _run(cam, [{"readings": {"max_mm": 135.0}}, {"gate": {"moved": False}}, accepted], GUARD)
    assert [m["mode"] for m in cl.msgs] == ["readings", "gate_joints", "gate_fit"] and outs == [False, True, False]
    assert c.rec["joints_status"] == "kept" and c.rec["status"] == "replaced" and np.allclose(cam.override["T"], 2 * np.eye(4))


def test_server_readings_compare_the_hand_reading_with_the_model():
    class Panda:
        def fk(self, q, g):
            return {"gripper0_eef": (np.eye(3), np.array([q[0], 0.0, 1.0]))}

    wc = SimpleNamespace(panda=Panda(), arm_samples={}, arm_rc_samples={}, workcell={"arena": []})
    per = SimpleNamespace(cell=lambda s: wc, p={"arm_cam": {}})
    frs = [{"joint_pos": [0.1] + [0.0] * 6, "gripper_qpos": [0.0, 0.0], "eef_pos": [0.1, 0.0, 1.0]},
           {"joint_pos": [0.2] + [0.0] * 6, "gripper_qpos": [0.0, 0.0], "eef_pos": [0.2, 0.0, 1.135]}]
    import rowant_lab.libero_env.arm_camera as ACm
    rep = ACm.handle(per, {"mode": "readings", "suite": "s", "frames": frs})
    assert rep["readings"]["dev_mm"] == [0.0, 135.0] and rep["readings"]["max_mm"] == 135.0 and rep["ok_reply"]


def test_audit_allows_reading_frames_only_in_the_readings_mode():
    from rowant_lab.libero_env import m2_audit as AU

    class Inner:
        def call(self, msg):
            return {}

    au = AU.AuditedClient(Inner())
    fr = {"joint_pos": [0.0] * 7, "gripper_qpos": [0.0, 0.0], "eef_pos": [0.0, 0.0, 1.0]}
    au.call({"op": "arm_cam", "mode": "readings", "suite": "s", "episode": "0123456789abcdef", "frames": [fr]})
    assert au.record()["ok"]
    au.call({"op": "arm_cam", "mode": "gate_fit", "suite": "s", "episode": "0123456789abcdef", "frames": [fr]})
    assert not au.record()["ok"]
