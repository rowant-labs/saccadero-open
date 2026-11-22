"""Item 3 (2026-11-14): the close look with the wrist camera before closing and before letting go (off by default)."""

from types import SimpleNamespace

import numpy as np

from rowant_lab.libero_env.m2_kin import mat_to_quat
from rowant_lab.pro2 import colour_line as CL
from rowant_lab.pro2.robot import agent as AG
from rowant_lab.pro2.robot import look as LK

TDOC = {"root_body": "main", "bodies": {"main": {"parent": None}}, "part_of": {"main": "main"}, "pose_in_part": {},
        "parts": {"main": {"joint": None}}}


def _quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _lib():
    return SimpleNamespace(robot={"finger_joints": {"a": {"qpos": 0.04}, "b": {"qpos": -0.04}}},
                           jaw=SimpleNamespace(gap_from_qpos=lambda a, b: abs(a) + abs(b)),
                           export_types={"can": TDOC}, quat_mat=_quat_mat,
                           poses_from_percepts=lambda ps: {f"{p['id']}_{r}": b for p in ps for r, b in p["bodies"].items()})


class Client:
    def __init__(self, move=(0.012, -0.02, 0.0), status="tracked"):
        self.calls, self.move, self.status = [], np.asarray(move, float), status

    def call(self, msg):
        self.calls.append(msg)
        out = {}
        for pid, pr in msg["priors"].items():
            t = np.asarray(pr["t_w"], float) + self.move
            out[pid] = {"status": self.status, "R_w": pr["R_w"], "t_w": t.tolist(), "n_px": 400,
                        "jump_m": float(np.linalg.norm(self.move))}
        return {"tracks": out, "times": {"total_s": 0.1}}


def _cam(camera, width, step):
    return {"camera": camera, "rgb": np.zeros((width, width, 3), np.uint8), "depth_u16": np.ones((width, width), np.uint16),
            "depth_scale_mm": 0.1, "K": np.eye(3), "T_world_from_camera": np.eye(4), "width": width, "height": width,
            "step": step}


def _tool(client, cam=_cam, gid="g1"):
    t = LK.PerceptionTool(_lib(), client, cam, "libero_object", "ep")
    t.obs = {"robot0_joint_pos": np.zeros(7), "robot0_gripper_qpos": np.array([0.04, -0.04]),
             "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0])}
    t.percepts = [{"id": "p0", "type": "can", "gid": gid, "joints": {},
                   "bodies": {"main": {"pos": [0.1, 0.2, 0.9], "quat_wxyz": mat_to_quat(np.eye(3))}}}]
    t.target = "p0"
    return t


def test_close_look_tracked_moves_the_percept():
    cl = Client()
    t = _tool(cl)
    rec = t.close_look("p0")
    assert rec["status"] == "tracked"
    assert np.allclose(rec["move"], [0.012, -0.02, 0.0])
    assert rec["turn_deg"] == 0.0
    assert np.allclose(t.percepts[0]["bodies"]["main"]["pos"], [0.112, 0.18, 0.9])
    assert t.percepts[0]["source"] == "close"
    assert rec["updated"] is True
    msg = cl.calls[0]
    assert msg["mode"] == "track" and msg["cameras"][0]["camera"] == LK.WRIST and msg["look"] == LK.CLOSE_LOOK_BASE
    assert t.n_close == 1 and t.n_full == 0 and t.n_track == 0
    t.close_look("p0")
    assert cl.calls[1]["look"] == LK.CLOSE_LOOK_BASE + 1  # close looks take look indices of their own
    assert t._msg("agentview")["look"] == 0  # the other looks keep theirs


def test_close_look_below_the_threshold_leaves_the_percept():
    t = _tool(Client(move=(0.001, 0.0, 0.0)))
    rec = t.close_look("p0", min_move_m=0.002)
    assert rec["status"] == "tracked" and rec["updated"] is False and np.allclose(rec["move"], [0.001, 0.0, 0.0])
    assert t.percepts[0]["bodies"]["main"]["pos"] == [0.1, 0.2, 0.9] and "source" not in t.percepts[0]


def test_close_look_skips_without_a_wrist_or_a_trackable_percept():
    cl = Client()
    cam = lambda *a: _cam(*a)  # noqa: E731
    cam.wrist_ok = False
    assert _tool(cl, cam).close_look("p0")["status"] == "no_wrist"
    assert _tool(cl).close_look("nope")["status"] == "no_percept"
    assert _tool(cl, gid=None).close_look("p0")["status"] == "untrackable"
    assert cl.calls == []


def test_close_look_not_tracked_keeps_the_percept():
    t = _tool(Client(status="predicted"))
    rec = t.close_look("p0")
    assert rec["status"] == "predicted" and "move" not in rec
    assert t.percepts[0]["bodies"]["main"]["pos"] == [0.1, 0.2, 0.9]


def _agent_ns(oracle, **ap):
    return SimpleNamespace(ap=AG.AgentParams(**ap), n_close=0, oracle=oracle)


def test_agent_close_look_applies_measured_moves_only():
    tracked = SimpleNamespace(close_look=lambda pid, **kw: {"status": "tracked", "move": [0.01, 0.0, 0.0], "n_px": 300,
                                                            "updated": True})
    ns = _agent_ns(tracked, close_look_grasp=True)
    mv, rec = AG.OracleAgent._close_look(ns, "p0")
    assert np.allclose(mv, [0.01, 0.0, 0.0]) and rec["applied"] and ns.n_close == 1
    tiny = SimpleNamespace(close_look=lambda pid, **kw: {"status": "tracked", "move": [0.001, 0.0, 0.0], "updated": False})
    mv, rec = AG.OracleAgent._close_look(_agent_ns(tiny), "p0")
    assert mv is None and "applied" not in rec
    lost = SimpleNamespace(close_look=lambda pid, **kw: {"status": "predicted"})
    assert AG.OracleAgent._close_look(_agent_ns(lost), "p0")[0] is None
    capped = _agent_ns(tracked, close_looks_max=1)
    capped.n_close = 1
    assert AG.OracleAgent._close_look(capped, "p0") == (None, None)
    assert AG.OracleAgent._close_look(_agent_ns(object()), "p0") == (None, None)  # a tool without close looks


def test_off_by_default_and_colour_tool_has_no_wrist():
    ap = AG.AgentParams()
    assert ap.close_look_grasp is False and ap.close_look_place is False
    look = CL.make_tool({"agentview": np.zeros((8, 8, 3), np.uint8)}, "libero_object",
                        {"colour": {"depth": "sim", "calib": "sim"}}, sim_look=lambda *a: None)
    assert look.wrist_ok is False
