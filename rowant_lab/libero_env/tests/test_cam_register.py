"""cam_register (stream PATH, round 3): a camera that moved in the middle of an episode, from the scene itself.
Pictures of a patterned floor drawn exactly from a known camera, with their exact depth."""

import cv2
import numpy as np
import pytest

from rowant_lab.pro2 import arm_calib as ACAL
from rowant_lab.pro2 import cam_register as CR
from rowant_lab.pro2 import colour_line as CL

PX, MM, HALF = 512, 0.002, 1.5  # the picture's width; the floor's pattern: metres per cell and its half width
F = 0.5 * PX / np.tan(np.radians(22.5))
K = np.array([[F, 0.0, PX / 2.0], [0.0, F, PX / 2.0], [0.0, 0.0, 1.0]])


def _pattern(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(2 * HALF / MM)
    img = np.full((n, n, 3), 120, np.uint8)
    for _ in range(1400):
        x, y = rng.integers(0, n, 2)
        w, h = rng.integers(10, 90, 2)
        img[y:y + h, x:x + w] = rng.integers(0, 256, 3)
    for _ in range(500):
        c = tuple(int(v) for v in rng.integers(0, n, 2))
        cv2.circle(img, c, int(rng.integers(4, 30)), tuple(int(v) for v in rng.integers(0, 256, 3)), -1)
    return img


FLOOR, OTHER = _pattern(1), _pattern(2)


def _camera(pos, target) -> np.ndarray:
    """T_world_from_camera of a camera at ``pos`` that looks at ``target`` (x right, y down, z forward)."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    z = (target - pos) / np.linalg.norm(target - pos)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(z, x), z, pos
    return T


def _frame(T_true, T_assumed=None, step=0, floor=FLOOR, patch=None) -> dict:
    """The floor (z = 0) as a camera at ``T_true`` sees it, with its exact depth; the frame names ``T_assumed``."""
    v, u = np.mgrid[0:PX, 0:PX].astype(np.float64)
    d = np.stack([(u - K[0, 2]) / F, (v - K[1, 2]) / F, np.ones_like(u)], -1) @ T_true[:3, :3].T
    s = -T_true[2, 3] / d[..., 2]
    P = T_true[:3, 3] + s[..., None] * d
    rgb = cv2.remap(floor, ((P[..., 0] + HALF) / MM).astype(np.float32), ((P[..., 1] + HALF) / MM).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if patch is not None:  # something of the robot's in front of the scene
        x, y, w, h = patch
        rgb[y:y + h, x:x + w] = (200, 30, 30)
        rgb[y + 10:y + h - 10:14, x + 10:x + w - 10] = (20, 20, 20)
    return {"camera": "agentview", "rgb": rgb, "depth_u16": np.clip(np.rint(s * 1e4), 0, 65535).astype(np.uint16),
            "depth_scale_mm": 0.1, "K": K.copy(), "T_world_from_camera": (T_true if T_assumed is None else T_assumed).copy(),
            "step": step}


T0 = _camera([0.0, -1.0, 1.0], [0.0, 0.0, 0.0])
T_BIG = _camera([0.12, -0.92, 1.06], [0.03, 0.02, 0.0])  # 15.6 cm away and turned
T_SMALL = T0.copy()
T_SMALL[:3, 3] += [0.025, -0.018, 0.012]  # 3.3 cm, no turn


def _armed(o=None) -> CR.Register:
    r = CR.Register(o)
    r.arm()
    assert r.look(_frame(T0)) is None and r.rec["reference"]["features"] > 500
    return r


def test_nothing_happens_before_it_is_armed_and_the_first_frame_after_is_the_reference():
    r = CR.Register()
    assert r.look(_frame(T0)) is None and r.ref is None and r.rec["looks"] == []
    r.arm()
    assert r.look(_frame(T0, step=7)) is None
    assert r.rec["reference"]["at"] == 7 and r.rec["reference"]["with_depth"] == r.rec["reference"]["features"]
    assert r.rec["looks"] == []


def test_a_camera_that_stands_still_changes_nothing_even_with_the_arm_in_the_picture():
    r = _armed()
    assert r.look(_frame(T0, step=3)) is None
    assert r.look(_frame(T0, step=4, patch=(150, 200, 120, 200))) is None
    assert [d["status"] for d in r.rec["looks"]] == ["same", "same"]
    assert r.rec["looks"][0]["px"] == 0.0 and r.rec["looks"][1]["px"] == 0.0 and r.last is None


@pytest.mark.parametrize("T1", [T_BIG, T_SMALL])
def test_a_moved_camera_is_found_from_the_scene(T1):
    r = _armed()
    got = r.look(_frame(T1, T_assumed=T0, step=30))
    T, d = got["T"], r.rec["looks"][-1]
    assert d["status"] == "moved" and d["px"] > 2.0 and "depth_u16" not in got
    shift, turn = CR.pose_diff(T, T1)
    assert shift < 0.005 and turn < 0.2  # exact depth: the fit is right to a few millimetres
    shift, turn = CR.pose_diff(r.last["T_pixels"], T1)
    assert shift < 0.01 and turn < 0.3 and d["pixels"]["from_depth"][0] < 0.01
    assert d["depth"]["inliers"] >= 40 and d["depth"]["rms_mm"] < 5.0


def test_later_frames_keep_the_registered_camera_and_get_the_reference_back_when_the_camera_returns():
    r = _armed()
    T = r.look(_frame(T_BIG, T_assumed=T0, step=30))["T"]
    assert r.look(_frame(T_BIG, T_assumed=T, step=40, patch=(100, 100, 80, 160))) is None
    assert r.rec["looks"][-1]["status"] == "as registered"
    back = r.look(_frame(T0, T_assumed=T, step=50))
    assert r.rec["looks"][-1]["status"] == "back" and np.allclose(back["T"], T0) and r.cur is None
    assert r.look(_frame(T0, step=60)) is None and r.rec["looks"][-1]["status"] == "same"


def test_a_picture_of_something_else_or_without_depth_is_used_as_it_came():
    r = _armed()
    assert r.look(_frame(T0, step=5, floor=OTHER)) is None
    assert r.rec["looks"][-1]["status"] in ("too few matches", "refused")
    fr = _frame(T_BIG, T_assumed=T0, step=6)
    fr["depth_u16"][:] = 0
    assert r.look(fr) is None and r.rec["looks"][-1]["status"] == "refused" and r.cur is None
    assert r.rec["s"] > 0


def test_a_change_within_the_floor_is_no_change():
    T1 = T0.copy()
    T1[:3, 3] += [0.008, 0.0, 0.0]  # 8 mm: under the 1.5 cm and half-degree floor
    r = _armed()
    assert r.look(_frame(T1, T_assumed=T0, step=9)) is None
    d = r.rec["looks"][-1]
    assert (d["status"], d.get("why")) in (("refused", "no change in the camera"), ("same", None))


def test_the_classical_fit_is_used_when_asked_for():
    r = _armed({"use": "pixels"})
    T = r.look(_frame(T_BIG, T_assumed=T0, step=30))["T"]
    assert r.rec["looks"][-1]["status"] == "moved" and np.allclose(T, r.last["T_pixels"])
    shift, turn = CR.pose_diff(T, T_BIG)
    assert shift < 0.01 and turn < 0.3
    with pytest.raises(ValueError):
        CR.Register({"use": "other"})


def test_the_same_frames_give_the_same_answer_twice():
    for use in ("depth", "scene"):
        a = _armed({"use": use}).look(_frame(T_BIG, T_assumed=T0, step=30))
        b = _armed({"use": use}).look(_frame(T_BIG, T_assumed=T0, step=30))
        assert np.array_equal(a["T"], b["T"]) and np.array_equal(a.get("depth_u16"), b.get("depth_u16"))


# ---- the form "scene": the true camera, and the frame's depth corrected to fit it
def _biased(T1, T_assumed, step=30, patch=None) -> dict:
    """The picture a camera at ``T1`` takes, with the depth a network stuck to the reference's geometry would give:
    at every pixel the floor's depth as the camera at T0 saw it there."""
    fr = _frame(T1, T_assumed=T_assumed, step=step, patch=patch)
    fr["depth_u16"] = _frame(T0)["depth_u16"].copy()
    return fr


def _mm_off(depth_u16, T_true) -> float:
    true = _frame(T_true)["depth_u16"].astype(np.float64)
    ok = (depth_u16 > 0) & (true < 30000)
    return float(np.median(np.abs(depth_u16.astype(np.float64)[ok] - true[ok]))) * 0.1


@pytest.mark.parametrize("T1", [T_BIG, T_SMALL])
def test_scene_finds_the_true_camera_and_corrects_a_depth_that_is_stuck_to_the_old_view(T1):
    r = _armed({"use": "depth"})
    Td = r.look(_biased(T1, T0))
    off_depth = 1.0 if Td is None else CR.pose_diff(Td["T"], T1)[0]
    r = _armed({"use": "scene"})
    fr = _biased(T1, T0)
    before = _mm_off(fr["depth_u16"], T1)
    got = r.look(fr)
    d = r.rec["looks"][-1]
    shift, turn = CR.pose_diff(got["T"], T1)
    assert d["status"] == "moved" and shift < 0.01 and turn < 0.3
    assert off_depth > 2 * shift  # the fit through the stuck depth lands elsewhere
    after = _mm_off(got["depth_u16"], T1)
    assert before > 8.0 and after < 3.0 and d["scene"]["rms_mm"] < 5.0 < d["scene"]["rms_mm_before"]
    assert d["scene"]["n"] >= 40
    # a later frame from the same place: the same camera, its own depth corrected by a plane fitted on it
    later = r.look(_biased(T1, got["T"], step=40, patch=(100, 100, 80, 160)))
    d2 = r.rec["looks"][-1]
    assert d2["status"] == "as registered" and "T" not in later and "scene" in d2
    assert _mm_off(later["depth_u16"], T1) < 3.0


def test_scene_leaves_a_right_depth_as_it_is_and_gives_the_reference_back():
    r = _armed({"use": "scene"})
    fr = _frame(T_BIG, T_assumed=T0, step=30)
    got = r.look(fr)
    # (2 mm: with the corners taken evenly, far ones count as much as near ones in the fit of the camera)
    assert _mm_off(got["depth_u16"], T_BIG) < 2.0 and abs(r.rec["looks"][-1]["scene"]["plane"][0]) < 2e-3
    back = r.look(_frame(T0, T_assumed=got["T"], step=50))
    assert np.allclose(back["T"], T0) and "depth_u16" not in back
    assert r.look(_frame(T0, step=60)) is None


def test_scene_refuses_when_the_depth_cannot_be_made_to_fit():
    r = _armed({"use": "scene"})
    fr = _frame(T_BIG, T_assumed=T0, step=30)
    rng = np.random.default_rng(3)
    fr["depth_u16"] = rng.integers(3000, 30000, fr["depth_u16"].shape).astype(np.uint16)  # no plane explains it
    assert r.look(fr) is None
    d = r.rec["looks"][-1]
    assert d["status"] == "refused" and d["why"] == "the depth does not fit" and r.cur is None


def test_depth_plane_is_exact_for_a_plane_and_leaves_stray_rows_out():
    rng = np.random.default_rng(5)
    un, vn = rng.uniform(-0.4, 0.4, 300), rng.uniform(-0.4, 0.4, 300)
    z_net = 1.0 / (0.8 + 0.3 * un - 0.2 * vn)
    z_true = 1.0 / (1.0 / z_net + 0.05 - 0.02 * un + 0.07 * vn)
    z_true[:30] *= rng.uniform(1.2, 1.6, 30)  # things that moved
    x, use = CR.depth_plane(z_true, z_net, un, vn)
    assert np.allclose(x, [0.05, -0.02, 0.07], atol=1e-6) and not use[:30].any() and use[30:].all()
    assert CR.depth_plane(z_true[:4], z_net[:4], un[:4], vn[:4]) is None


def test_corrected_keeps_pixels_without_depth_and_drops_those_beyond_the_frame():
    fr = {"depth_u16": np.array([[0, 10000], [10000, 60000]], np.uint16), "depth_scale_mm": 0.1,
          "K": np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])}
    out = CR.corrected(fr, (0.1, 0.0, 0.0))  # 1 m -> 1/(1 + 0.1) m
    assert out[0, 0] == 0 and out[0, 1] == out[1, 0] == round(10000 / 1.1) and out[1, 1] > 0
    assert CR.corrected(fr, (-0.9, 0.0, 0.0))[0, 1] == 0  # 1/(1 - 0.9) = 10 m: beyond what the frame holds


# ---- a thing that moves is not a camera that moves (2026-11-19, after the first pod run)
def _busy(seed: int, n: int) -> np.ndarray:
    """A small, densely patterned picture: a thing with more corners than anything else in view."""
    rng = np.random.default_rng(seed)
    return np.repeat(np.repeat(rng.integers(0, 256, (n // 4, n // 4, 3), dtype=np.uint8), 4, 0), 4, 1)


def _with_thing(fr: dict, at: tuple, thing: np.ndarray) -> dict:
    x, y = at
    fr = {**fr, "rgb": fr["rgb"].copy()}
    fr["rgb"][y:y + thing.shape[0], x:x + thing.shape[1]] = thing
    return fr


def test_corners_are_taken_evenly_so_no_one_thing_holds_most_of_them():
    fr = _with_thing(_frame(T0), (180, 220), _busy(7, 120))
    crowd, _d = CR.features(fr["rgb"], 3000, 0)
    even, _d = CR.features(fr["rgb"], 3000, 8)
    inside = lambda xy: float(np.mean((xy[:, 0] >= 180) & (xy[:, 0] < 300) & (xy[:, 1] >= 220) & (xy[:, 1] < 340)))  # noqa: E731
    assert inside(even) < inside(crowd) and inside(even) < 0.3
    cells = {(int(y * 8 / PX), int(x * 8 / PX)) for x, y in even}
    assert len(cells) >= 48  # the corners reach nearly every cell that has anything in it


def test_a_thing_that_moves_before_a_still_camera_changes_nothing():
    thing = _busy(7, 120)
    r = CR.Register({"use": "scene"})
    r.arm()
    r.look(_with_thing(_frame(T0), (180, 220), thing))
    for step, at in ((20, (192, 226)), (30, (260, 150))):  # pushed a little, then carried off
        assert r.look(_with_thing(_frame(T0, step=step), at, thing)) is None
        d = r.rec["looks"][-1]
        assert d["status"] in ("part moved", "same") and d["cells"][1] < 0.8 * d["cells"][0] and r.cur is None
    # and a camera that does move is still found, with the thing in the picture
    got = r.look(_with_thing(_frame(T_BIG, T_assumed=T0, step=40), (260, 150), thing))
    d = r.rec["looks"][-1]
    assert d["status"] == "moved" and d["cells"][1] >= 0.8 * d["cells"][0]
    shift, turn = CR.pose_diff(got["T"], T_BIG)
    assert shift < 0.01 and turn < 0.3


def test_without_the_guards_a_moving_thing_passes_for_a_moving_camera_where_nothing_else_has_corners():
    # the first build (grid 0, no vote), and a scene whose only strongly patterned thing moves: its motion is
    # every match's motion. With the guards too few cells of the picture have anything to say, and nothing changes
    thing = _busy(7, 120)

    def frame(at, step=0):
        fr = _frame(T0, step=step)
        fr["rgb"] = (fr["rgb"] // 16 + 100).astype(np.uint8)  # a floor too faint to give a corner
        return _with_thing(fr, at, thing)

    old = CR.Register({"grid": 0, "cells_moved": 0.0, "cells_min": 0, "use": "pixels"})
    old.arm()
    old.look(frame((150, 180)))
    old.look(frame((165, 190), 20))
    assert old.rec["looks"][-1]["px"] > 2.0 and old.rec["looks"][-1]["status"] in ("moved", "refused")
    new = CR.Register({"use": "pixels"})
    new.arm()
    new.look(frame((150, 180)))
    assert new.look(frame((165, 190), 20)) is None
    d = new.rec["looks"][-1]
    assert d["status"] == "part moved" and d["cells"][0] < 8 and new.cur is None


def test_the_cells_vote_counts_only_cells_with_enough_matches():
    xy = np.array([[10.0, 10.0]] * 3 + [[300.0, 300.0]] * 3 + [[500.0, 20.0]] * 2)
    shift = np.array([5.0, 6.0, 7.0, 0.0, 0.0, 9.0, 9.0, 9.0])
    assert CR.cell_vote(xy, shift, (512, 512), 8, 3, 2.0) == (2, 1)  # the third cell holds two matches: no vote


def test_a_registration_that_fails_costs_no_look(monkeypatch):
    r = _armed()

    def broken(*a, **k):
        raise RuntimeError("no corners today")

    monkeypatch.setattr(CR, "features", broken)
    assert r.look(_frame(T_BIG, T_assumed=T0, step=30)) is None  # the look goes on with the camera as it was
    assert r.rec["errors"] == [{"at": 30, "error": "RuntimeError: no corners today"}] and r.cur is None
    monkeypatch.undo()
    assert r.look(_frame(T_BIG, T_assumed=T0, step=31))["T"] is not None  # and the next look is registered as usual


# ---- the camera tool and the check of the first motion
class _Reg:
    def __init__(self, T):
        self.T, self.seen = T, []

    def look(self, fr):
        self.seen.append((fr["camera"], fr["step"]))
        if len(self.seen) == 2:
            return {"T": self.T}
        return {"depth_u16": np.full((8, 8), 7, np.uint16)} if len(self.seen) == 3 else None


class _Net:
    sha256 = "sha"

    def depth_u16(self, rgb, K=None):
        return np.ones(rgb.shape[:2], np.uint16)


def test_the_camera_tool_registers_front_camera_frames_only_and_uses_the_answer_from_then_on(monkeypatch):
    monkeypatch.setattr(CL, "_NETS", {("n1", 2): _Net()})
    img = {"agentview": np.zeros((8, 8, 3), np.uint8), "robot0_eye_in_hand": np.zeros((8, 8, 3), np.uint8)}
    look = CL.make_tool(img, "libero_object", {"colour": {"depth": "net", "calib": "arm", "net": "n1"}})
    st = np.asarray(CL.stored_cameras()["libero_object"]["T_world_from_camera"], float)
    assert np.allclose(look("agentview", 8, 0)["T_world_from_camera"], st)  # no register: as it always was
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.2, 0.3]
    look.override["register"] = reg = _Reg(T)
    assert np.allclose(look("agentview", 8, 1)["T_world_from_camera"], st) and look.override["T"] is None
    look("robot0_eye_in_hand", 8, 2)
    assert np.allclose(look("agentview", 8, 3)["T_world_from_camera"], T) and np.allclose(look.override["T"], T)
    fr = look("agentview", 8, 4)  # a corrected depth alone: the camera stays, this frame's depth is replaced
    assert np.allclose(fr["T_world_from_camera"], T) and int(fr["depth_u16"][0, 0]) == 7
    assert int(look("agentview", 8, 5)["depth_u16"][0, 0]) == 1
    assert reg.seen == [("agentview", 1), ("agentview", 3), ("agentview", 4), ("agentview", 5)]


class _Cam:
    def __init__(self):
        self.override = {"T": None}
        self.calls = []

    def __call__(self, camera, width, step):
        self.calls.append((camera, width, step))
        fr = {"camera": camera, "depth_u16": np.zeros((4, 4), np.uint16), "depth_scale_mm": 0.1, "K": np.eye(3),
              "T_world_from_camera": np.eye(4), "rgb": np.zeros((4, 4, 3), np.uint8), "step": step}
        if self.override.get("register") is not None:
            self.override["register"].look(fr)
        return fr


class _Client:
    def __init__(self):
        self.msgs = []

    def call(self, msg):
        self.msgs.append(msg)
        return {"gate": {"moved": False, "cost_m": 0.002}}


def _obs(x):
    return {"robot0_eef_pos": [x, 0.0, 1.0], "robot0_joint_pos": [0.0] * 7, "robot0_gripper_qpos": [0.04, -0.04]}


def test_the_reference_is_taken_once_one_step_after_the_first_check_ended():
    cam = _Cam()
    c = ACAL.ArmCalib(cam, _Client(), "libero_object", "ep_token", {"arm_cam_robot": {"register": {"max_px": 3.0}}})
    assert cam.override["register"] is c.register and c.register.o["max_px"] == 3.0 and not c.register.armed
    c.start(_obs(0.0))
    for t, x in enumerate([0.07, 0.14], start=1):
        c.after_step(_obs(x), t)
    assert c.done and c.rec["status"] == "kept" and not c.register.armed and len(cam.calls) == 3
    c.after_step(_obs(0.2), 3)
    assert c.register.armed and cam.calls[-1] == ("agentview", 1024, 3) and len(cam.calls) == 4
    assert c.rec["register"]["reference"]["at"] == 3
    c.after_step(_obs(0.4), 4)
    assert len(cam.calls) == 4  # once


def test_the_option_is_off_unless_given_and_needs_the_colour_line():
    cam = _Cam()
    c = ACAL.ArmCalib(cam, _Client(), "libero_object", "ep_token", {})
    assert c.register is None and "register" not in cam.override and "register" not in c.rec

    class Plain:  # the depth line's camera tool: no override, the camera is right
        def __call__(self, camera, width, step):
            return {"depth_u16": np.zeros((4, 4), np.uint16), "depth_scale_mm": 0.1, "K": np.eye(3),
                    "T_world_from_camera": np.eye(4), "rgb": np.zeros((4, 4, 3), np.uint8)}

    c = ACAL.ArmCalib(Plain(), _Client(), "libero_object", "ep_token", {"arm_cam_robot": {"register": {}}})
    assert c.register is None
