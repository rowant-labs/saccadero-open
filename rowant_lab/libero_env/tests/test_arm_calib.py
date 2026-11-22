"""arm_calib (item 2, runner side) and the colour line's camera override, with a fake camera tool and server."""

import numpy as np

from rowant_lab.pro2 import arm_calib as ACAL
from rowant_lab.pro2 import colour_line as CL


class FakeCam:
    def __init__(self):
        self.override = {"T": None}
        self.calls = []

    def __call__(self, camera, width, step):
        self.calls.append((camera, width, step))
        return {"depth_u16": np.zeros((4, 4), np.uint16), "depth_scale_mm": 0.1, "K": np.eye(3),
                "T_world_from_camera": np.eye(4), "rgb": np.zeros((4, 4, 3), np.uint8)}


class FakeClient:
    def __init__(self, reply):
        self.reply, self.msgs = reply, []

    def call(self, msg):
        self.msgs.append(msg)
        return self.reply


def _obs(x):
    return {"robot0_eef_pos": [x, 0.0, 1.0], "robot0_joint_pos": [0.0] * 7, "robot0_gripper_qpos": [0.04, -0.04]}


def _run(reply, xs):
    cam, cl = FakeCam(), FakeClient(reply)
    c = ACAL.ArmCalib(cam, cl, "libero_object", "ep_token", {})
    c.start(_obs(0.0))
    outs = [c.after_step(_obs(x), t + 1) for t, x in enumerate(xs)]
    return c, cam, cl, outs


def test_frames_wait_for_the_hand_to_move_and_the_gate_keeps_the_camera():
    c, cam, cl, outs = _run({"gate": {"moved": False, "cost_m": 0.002}}, [0.02, 0.04, 0.07, 0.10, 0.14, 0.2])
    assert c.rec["frames_at"] == [0, 3, 5]  # 0.00, then 0.07 (>= 6 cm from 0), then 0.14 (>= 6 cm from both)
    assert len(cl.msgs) == 1 and cl.msgs[0]["mode"] == "gate_fit" and len(cl.msgs[0]["frames"]) == 3
    assert set(cl.msgs[0]["frames"][0]) == {"depth_u16", "depth_scale_mm", "K", "T_world_from_camera", "joint_pos",
                                             "gripper_qpos"}
    assert not any(outs) and cam.override["T"] is None and c.rec["status"] == "kept"


def test_an_accepted_fit_replaces_the_camera_once():
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.2, 0.3]
    c, cam, cl, outs = _run({"gate": {"moved": True}, "fit": {"accepted": True, "T_world_from_camera": T, "cost_m": 0.002}},
                            [0.07, 0.14, 0.3, 0.5])
    assert outs == [False, True, False, False]
    assert np.allclose(cam.override["T"], T) and c.rec["status"] == "replaced"
    assert "T_world_from_camera" not in c.rec["fit"]


def test_a_refused_fit_keeps_the_camera():
    c, cam, cl, outs = _run({"gate": {"moved": True}, "fit": {"accepted": False, "T_world_from_camera": np.eye(4)}},
                            [0.07, 0.14])
    assert not any(outs) and cam.override["T"] is None and c.rec["status"] == "fit refused"


def test_no_request_without_enough_motion():
    c, cam, cl, outs = _run({"gate": {"moved": True}}, [0.01] * 45)
    assert not cl.msgs and c.rec["status"] == "too little motion"


def test_colour_tool_uses_the_override_only_once_set():
    img = {"agentview": np.zeros((8, 8, 3), np.uint8)}
    look = CL.make_tool(img, "libero_object", {"colour": {"depth": "sim", "calib": "arm"}},
                        sim_look=lambda c, w, s: {"depth_u16": np.ones((8, 8), np.uint16)})
    st = np.asarray(CL.stored_cameras()["libero_object"]["T_world_from_camera"], float)
    assert np.allclose(look("agentview", 8, 0)["T_world_from_camera"], st)
    T = np.eye(4)
    look.override["T"] = T
    assert np.allclose(look("agentview", 8, 1)["T_world_from_camera"], T)


def _obs_q(x, y=0.0, z=1.0):
    return {"robot0_eef_pos": [x, y, z], "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0], "robot0_joint_pos": [0.0] * 7,
            "robot0_gripper_qpos": [0.04, -0.04]}


class Agent:
    def __init__(self, reason):
        self.finish_reason = reason


def test_the_robot_moves_itself_when_it_did_not_move():
    cam, cl = FakeCam(), FakeClient({"gate": {"moved": False}})
    c = ACAL.ArmCalib(cam, cl, "libero_object", "ep_token", {})
    c.start(_obs_q(0.0))
    assert c.can_move(None, False, 1, 280)  # no agent at all
    c.begin_motion(_obs_q(0.0), 1)  # a fresh first frame where the hand is
    a = c.motion_action(_obs_q(0.0))
    assert len(a) == 7 and a[1] > 0.9 and a[6] == -1.0  # toward +y at full speed, gripper open
    # the hand reaches the first leg's end, then the second's: a frame at each, then the request
    assert not c.after_step(_obs_q(0.0, 0.05), 2) and c.rec["frames_at"] == [0, 1]
    assert not c.after_step(_obs_q(0.0, 0.09), 3) and c.rec["frames_at"] == [0, 1, 3]
    assert not c.after_step(_obs_q(0.0, 0.10, 1.09), 4)
    assert c.rec["frames_at"] == [0, 1, 3, 4] and len(cl.msgs) == 1 and c.done and not c.moving
    assert c.rec["status"] == "kept" and c.rec["motion_at"] == 1
    assert len(cl.msgs[0]["frames"]) == 3
    assert not c.can_move(None, False, 5, 280)  # once only


def test_when_the_robot_may_move_itself():
    c = ACAL.ArmCalib(FakeCam(), FakeClient({}), "libero_object", "ep_token", {})
    c.start(_obs_q(0.0))
    assert not c.can_move(Agent(None), False, 5, 280)  # the agent is still working
    assert c.can_move(Agent("no_plan"), True, 5, 280) and c.can_move(Agent("max_attempts"), True, 90, 280)
    assert not c.can_move(Agent("placed"), True, 90, 280)  # never after the task's own end
    assert not c.can_move(Agent("no_budget"), True, 90, 280)
    assert not c.can_move(Agent("no_plan"), True, 230, 280)  # too few steps left
    c.done, c.rec["status"] = True, "too little motion"  # watching gave up: still allowed
    assert c.can_move(Agent("no_plan"), True, 100, 280)
    c.rec["status"] = "kept"  # the camera was checked: never
    assert not c.can_move(Agent("no_plan"), True, 100, 280)


def test_a_leg_ends_after_its_step_limit():
    cam, cl = FakeCam(), FakeClient({"gate": {"moved": False}})
    c = ACAL.ArmCalib(cam, cl, "libero_object", "ep_token", {"arm_cam_robot": {"leg_max_steps": 3}})
    c.start(_obs_q(0.0))
    c.begin_motion(_obs_q(0.0), 0)
    outs = [c.after_step(_obs_q(0.0), t) for t in range(1, 7)]  # the hand is stuck: each leg gives up after 3 steps
    assert c.rec["frames_at"] == [0, 0, 3, 6] and len(cl.msgs) == 1 and not any(outs)


# -- item 2b: the second try on the second net's depth, only after the first fit was refused
class SeqClient:
    def __init__(self, replies):
        self.replies, self.msgs = list(replies), []

    def call(self, msg):
        self.msgs.append(msg)
        return self.replies.pop(0)


class FakeCam2(FakeCam):
    def __init__(self):
        super().__init__()
        self.seen = []

    def net2_depth(self, rgb, K):
        self.seen.append((rgb.shape, np.asarray(K).shape))
        return np.full((4, 4), 7, np.uint16)


def _run2(replies, xs=(0.07, 0.14)):
    cam, cl = FakeCam2(), SeqClient(replies)
    c = ACAL.ArmCalib(cam, cl, "libero_object", "ep_token", {})
    c.start(_obs(0.0))
    outs = [c.after_step(_obs(x), t + 1) for t, x in enumerate(xs)]
    return c, cam, cl, outs


REFUSED = {"gate": {"moved": True}, "fit": {"accepted": False, "T_world_from_camera": np.eye(4)}}


def test_second_try_accepted_switches_to_the_second_net():
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.1, 0.2]
    c, cam, cl, outs = _run2([REFUSED, {"gate": {"moved": True}, "fit": {"accepted": True, "T_world_from_camera": T,
                                                                          "focal_scale": 1.05}}])
    assert outs == [False, True]
    assert [m["mode"] for m in cl.msgs] == ["gate_fit", "gate_fit2"]
    f2 = cl.msgs[1]["frames"]
    assert len(f2) == 3 and all(set(fr) == set(cl.msgs[0]["frames"][0]) for fr in f2)
    assert all(int(fr["depth_u16"][0, 0]) == 7 for fr in f2) and all(int(fr["depth_u16"][0, 0]) == 0 for fr in cl.msgs[0]["frames"])
    assert cam.override["net"] == 2 and np.allclose(cam.override["T"], T) and cam.override["focal_scale"] == 1.05
    assert c.rec["status"] == "replaced" and c.rec["by"] == "second net" and c.rec["fit"]["accepted"] is False
    assert "T_world_from_camera" not in c.rec["second"]["fit"] and not c.rgbs and not c.frames


def test_second_try_refused_or_kept_changes_nothing():
    for second in ({"gate": {"moved": False}}, {"gate": {"moved": True}, "fit": {"accepted": False}}):
        c, cam, cl, outs = _run2([REFUSED, second])
        assert not any(outs) and len(cl.msgs) == 2 and c.rec["status"] == "fit refused"
        assert cam.override == {"T": None} and "by" not in c.rec and "second" in c.rec


def test_no_second_try_when_the_first_fit_is_accepted_or_the_gate_keeps():
    T = np.eye(4)
    c, cam, cl, outs = _run2([{"gate": {"moved": True}, "fit": {"accepted": True, "T_world_from_camera": T}}])
    assert len(cl.msgs) == 1 and c.rec["status"] == "replaced" and "net" not in cam.override and "by" not in c.rec
    c, cam, cl, outs = _run2([{"gate": {"moved": False}}])
    assert len(cl.msgs) == 1 and c.rec["status"] == "kept" and not cam.seen


class FakeNet:
    def __init__(self, v):
        self.v, self.sha256, self.calls = v, f"sha{v}", 0

    def depth_u16(self, rgb, K=None):
        self.calls += 1
        return np.full(rgb.shape[:2], self.v, np.uint16)


def test_colour_tool_second_net_only_after_the_switch(monkeypatch):
    n1, n2 = FakeNet(1), FakeNet(2)
    monkeypatch.setattr(CL, "_NETS", {("n1", 2): n1, ("n2", 2): n2})
    img = {"agentview": np.zeros((8, 8, 3), np.uint8)}
    p = {"colour": {"depth": "net", "calib": "arm", "net": "n1", "net2": "n2"}}
    look = CL.make_tool(img, "libero_object", p)
    assert int(look("agentview", 8, 0)["depth_u16"][0, 0]) == 1 and look.net2_depth is not None
    look.override["net"] = 2
    assert int(look("agentview", 8, 1)["depth_u16"][0, 0]) == 2
    assert CL.describe(p) == {"depth": "net", "calib": "arm", "net_sha256": "sha1", "net2_sha256": "sha2"}
    for o in ({"depth": "net", "calib": "arm", "net": "n1"}, {"depth": "net", "calib": "stored", "net": "n1", "net2": "n2"}):
        look = CL.make_tool(img, "libero_object", {"colour": o})
        look.override["net"] = 2
        assert look.net2_depth is None and int(look("agentview", 8, 0)["depth_u16"][0, 0]) == 1


# ---- 2026-11-15: the picture itself shifted or zoomed (arm_image)
class ImageCam(FakeCam):
    """A camera tool that can check an image fix: ``image_depth`` records the fixes it was asked to undo."""

    def __init__(self):
        super().__init__()
        self.undone = []
        self.image_depth = lambda rgb, K, fix: self.undone.append(fix) or np.ones((4, 4), np.uint16)

    def __call__(self, camera, width, step):
        fr = super().__call__(camera, width, step)
        fr["rgb"] = np.full((8, 8, 3), 10 * step, np.uint8)
        return fr


class Script:
    """Replies by request mode."""

    def __init__(self, by_mode):
        self.by, self.msgs = by_mode, []

    def call(self, msg):
        self.msgs.append(msg)
        return self.by[msg["mode"]]


def _run_image(by_mode, params=None):
    cam, cl = ImageCam(), Script(by_mode)
    c = ACAL.ArmCalib(cam, cl, "libero_object", "ep_token", {"arm_image": {}} if params is None else params)
    c.start(_obs(0.0))
    outs = [c.after_step(_obs(x), t + 1) for t, x in enumerate([0.07, 0.14, 0.3])]
    return c, cam, cl, outs


IMG = {"accepted": True, "zoom": 0.95, "shift_px": [100.0, -20.0], "centre_px": [512.0, 512.0], "cost_px": 2.0}


def test_an_image_fix_is_checked_then_undone_in_later_looks():
    c, cam, cl, outs = _run_image({"gate_image": {"gate": {"moved": True}, "image": IMG},
                                   "gate": {"gate": {"moved": False, "cost_m": 0.002}}})
    assert [m["mode"] for m in cl.msgs] == ["gate_image", "gate"] and outs == [False, True, False]
    # the robot's masks mark any change (not one frame's alone): the server draws every pixel a moving link covers
    assert all(fr["mask"].shape == (8, 8) and fr["mask"].dtype == bool and not fr["mask_one"] for fr in cl.msgs[0]["frames"])
    fix = cam.override["image"]
    assert fix == {"zoom": 0.95, "shift_px": [100.0, -20.0], "centre_px": [512.0, 512.0], "width": 8}
    assert len(cam.undone) == 3 and cam.override["T"] is None  # the three stored frames, checked with the fix undone
    assert c.rec["status"] == "image fixed" and c.rec["image"]["gate_after"]["moved"] is False


def test_an_image_fix_that_fails_its_check_goes_on_to_the_camera_fit():
    T = np.eye(4)
    c, cam, cl, outs = _run_image({"gate_image": {"gate": {"moved": True}, "image": IMG},
                                   "gate": {"gate": {"moved": True}},
                                   "gate_fit": {"gate": {"moved": True}, "fit": {"accepted": True, "T_world_from_camera": T}}})
    assert [m["mode"] for m in cl.msgs] == ["gate_image", "gate", "gate_fit"]
    assert "image" not in cam.override and c.rec["status"] == "replaced"
    c, cam, cl, outs = _run_image({"gate_image": {"gate": {"moved": True}, "image": {"accepted": False, "reason": "no change"}},
                                   "gate_fit": {"gate": {"moved": True}, "fit": {"accepted": False}}})
    assert [m["mode"] for m in cl.msgs] == ["gate_image", "gate_fit"] and c.rec["status"] == "fit refused"


def test_a_nearly_right_image_fix_is_the_last_resort_when_no_camera_fit_is_accepted():
    before = {"moved": True, "cost_m": 0.040}

    def run(after_cost, fit_ok=False):
        return _run_image({"gate_image": {"gate": before, "image": IMG},
                           "gate": {"gate": {"moved": True, "cost_m": after_cost}},
                           "gate_fit": {"gate": before, "fit": {"accepted": fit_ok, "T_world_from_camera": np.eye(4)}}})

    c, cam, cl, outs = run(0.008)  # the arm 8 mm off after the undo, from 40: not good enough for the check, far nearer
    assert [m["mode"] for m in cl.msgs] == ["gate_image", "gate", "gate_fit"] and outs == [False, True, False]
    assert cam.override["image"]["zoom"] == 0.95 and cam.override["T"] is None
    assert c.rec["status"] == "image fixed" and c.rec["by"] == "approximately"
    c, cam, cl, outs = run(0.008, fit_ok=True)  # an accepted camera fit comes first
    assert "image" not in cam.override and c.rec["status"] == "replaced"
    c, cam, cl, outs = run(0.020)  # still far off after the undo: nothing is changed
    assert "image" not in cam.override and c.rec["status"] == "fit refused" and not any(outs)
    c, cam, cl, outs = _run_image({"gate_image": {"gate": {"moved": True, "cost_m": 0.015}, "image": IMG},
                                   "gate": {"gate": {"moved": True, "cost_m": 0.008}},
                                   "gate_fit": {"gate": before, "fit": {"accepted": False}}})
    assert "image" not in cam.override  # 8 mm from 15: not the large drop the rule asks for


def test_the_image_route_asks_nothing_more_when_the_gate_keeps_and_is_off_by_default():
    c, cam, cl, outs = _run_image({"gate_image": {"gate": {"moved": False}}})
    assert [m["mode"] for m in cl.msgs] == ["gate_image"] and not any(outs) and c.rec["status"] == "kept"
    c, cam, cl, outs = _run_image({"gate_fit": {"gate": {"moved": False}}}, params={})
    assert [m["mode"] for m in cl.msgs] == ["gate_fit"] and "mask" not in cl.msgs[0]["frames"][0]
    plain = FakeCam()  # a camera tool that cannot check a fix (the depth line's): the option does nothing
    assert ACAL.ArmCalib(plain, Script({}), "libero_object", "ep", {"arm_image": {}}).image is None


def test_change_masks_mark_where_the_frames_differ_from_their_median():
    a = np.zeros((6, 6, 3), np.uint8)
    b, c_ = a.copy(), a.copy()
    b[1:4, 1:4] = 200      # the arm here in one frame
    c_[3:6, 3:6] = 200     # and there in another
    c_[0, 5] = 255         # a lone pixel: dropped
    m = ACAL.change_masks([a, b, c_], 120, 3)
    assert m[0].sum() == 0 and m[1][1:4, 1:4].sum() >= 8 and not m[1][4:, 4:].any()
    assert m[2][4:6, 4:6].all() and not m[2][0, 5]
    d = np.full((6, 6, 3), 200, np.uint8)
    e, f = d.copy(), d.copy()
    e[1:4, 1:4] = 255   # a pale arm here in one frame
    e[3:6, 0:3] = 100   # and its shadow beside it
    mb = ACAL.change_masks([d, e, f], 120, 3, brighter=True)
    assert mb[1][1:3, 1:4].all() and not mb[1][4:6, 0:3].any()  # the arm, not the shadow
    assert ACAL.change_masks([d, e, f], 120, 3)[1][4:6, 0:2].all()  # without it both are marked


def test_undo_image_takes_each_pixel_from_where_the_shift_put_it():
    rgb = np.zeros((16, 16, 3), np.uint8)
    rgb[5, 9] = 200  # the point the assumed camera has at (x 6, y 7), moved 3 right and 2 up
    out = CL.undo_image(rgb, {"zoom": 1.0, "shift_px": [3.0, -2.0], "centre_px": [8.0, 8.0], "width": 16})
    assert out[7, 6, 0] == 200 and out.sum() == 600 and out.dtype == np.uint8
    same = CL.undo_image(rgb, {"zoom": 1.0, "shift_px": [0.0, 0.0], "centre_px": [8.0, 8.0], "width": 16})
    assert np.array_equal(same, rgb)
    big = np.zeros((32, 32, 3), np.uint8)
    big[10, 18] = 200  # the same change on a picture twice the size: the fix is scaled by the width
    out = CL.undo_image(big, {"zoom": 1.0, "shift_px": [3.0, -2.0], "centre_px": [8.0, 8.0], "width": 16})
    assert out[14, 12, 0] == 200
    z = CL.undo_image(np.full((16, 16, 3), 50, np.uint8), {"zoom": 0.5, "shift_px": [40.0, 0.0], "centre_px": [8.0, 8.0], "width": 16})
    assert (z == 50).all()  # from outside the picture: the nearest edge


def test_last_resort_uses_an_accepted_zoom_unless_the_undo_made_the_arm_worse():
    before = {"moved": True, "cost_m": 0.040}
    on = {"arm_image": {"last_resort": True}}

    small = {**IMG, "shift_px": [1.0, -1.0]}  # the stand-in picture is 8 px wide

    def run(after, params=on, img=small):
        return _run_image({"gate_image": {"gate": before, "image": img}, "gate": {"gate": after},
                           "gate_fit": {"gate": before, "fit": {"accepted": False}}}, params)

    c, cam, cl, outs = run({"moved": True, "cost_m": 0.020})  # 20 mm from 40: refused by the approximate rule
    assert cam.override["image"]["zoom"] == 0.95 and c.rec["status"] == "image fixed" and c.rec["by"] == "last resort"
    assert outs == [False, True, False]
    c, cam, cl, outs = run({"moved": None, "reason": "too little motion"})  # the check could not be made
    assert cam.override["image"]["zoom"] == 0.95 and c.rec["by"] == "last resort"
    c, cam, cl, outs = run({"moved": True, "cost_m": 0.050})  # worse after the undo: not this
    assert "image" not in cam.override and c.rec["status"] == "fit refused" and not any(outs)
    c, cam, cl, outs = run({"moved": True, "cost_m": 0.008})  # the approximate rule still comes first, by its name
    assert c.rec["by"] == "approximately"
    c, cam, cl, outs = run({"moved": True, "cost_m": 0.020}, img={**small, "zoom": 1.4})  # at the search's end
    assert "image" not in cam.override and c.rec["status"] == "fit refused"
    c, cam, cl, outs = run({"moved": True, "cost_m": 0.020}, img={**small, "shift_px": [2.0, -1.0]})
    assert "image" not in cam.override  # a shift beyond 0.24 of the picture's width
    c, cam, cl, outs = run({"moved": True, "cost_m": 0.020}, params={"arm_image": {}})  # off: as before
    assert "image" not in cam.override and c.rec["status"] == "fit refused"
