"""M1 unit tests: grasp geometry, records and frames, and the tools. Synthetic inputs only. Never imports LIBERO,
MuJoCo or torch.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from rowant_lab.libero_env import grasp as GR
from rowant_lab.libero_env import tools as T


def rotz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def box_prim(centre, half, R=None):
    return {"centre": list(centre), "R": (np.eye(3) if R is None else np.asarray(R)).tolist(), "half": list(half)}


# ----------------------------------------------------------------------------------------------- rotations
def test_quat_and_rotvec_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v) * rng.uniform(0.01, 3.1)
        th = np.linalg.norm(v)
        k = v / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
        assert np.allclose(GR.rotvec_from_mat(R), v, atol=1e-6)
        q = [math.cos(th / 2), *(math.sin(th / 2) * k)]
        assert np.allclose(GR.quat_wxyz_to_mat(q), R, atol=1e-9)
        assert np.allclose(GR.quat_xyzw_to_mat([q[1], q[2], q[3], q[0]]), R, atol=1e-9)
    assert np.allclose(GR.rotvec_from_mat(np.diag([1.0, -1.0, -1.0])), [math.pi, 0, 0], atol=1e-6) or \
        np.allclose(GR.rotvec_from_mat(np.diag([1.0, -1.0, -1.0])), [-math.pi, 0, 0], atol=1e-6)


def test_hand_matrix_is_top_down_rotation():
    for yaw in np.linspace(-3, 3, 7):
        R = GR.hand_matrix([math.cos(yaw), math.sin(yaw), 0.3])
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and abs(np.linalg.det(R) - 1) < 1e-12
        assert np.allclose(R[:, 2], [0, 0, -1]) and abs(R[2, 1]) < 1e-12
        assert abs(GR.wrap_pi(GR.yaw_of_hand(R) - yaw)) < 1e-9


# ----------------------------------------------------------------------------------------------- boxes
def test_line_hits_unit_box():
    B = GR.Boxes.from_prims([box_prim([0, 0, 0], [0.1, 0.2, 0.3])])
    tin, tout, hit, n_in, n_out = GR.line_hits([[-1.0, 0, 0], [-1.0, 0.5, 0]], [[1.0, 0, 0], [1.0, 0, 0]], B)
    assert hit[0, 0] and not hit[1, 0]
    assert math.isclose(tin[0, 0], 0.9) and math.isclose(tout[0, 0], 1.1)
    assert np.allclose(n_in[0, 0], [-1, 0, 0]) and np.allclose(n_out[0, 0], [1, 0, 0])


def test_line_hits_rotated_box_and_parallel_lines():
    R = rotz(math.pi / 4)
    B = GR.Boxes.from_prims([box_prim([1, 2, 0], [0.1, 0.1, 0.1], R)])
    tin, tout, hit, n_in, _ = GR.line_hits([[1 - 5, 2, 0]], [[1, 0, 0]], B)
    assert hit[0, 0] and math.isclose(tout[0, 0] - tin[0, 0], 0.2 * math.sqrt(2), rel_tol=1e-9)
    assert abs(np.linalg.norm(n_in[0, 0]) - 1) < 1e-12 and n_in[0, 0][0] < 0
    # a line parallel to a face, inside and outside the slab
    B2 = GR.Boxes.from_prims([box_prim([0, 0, 0], [1, 1, 1])])
    _, _, hit2, _, _ = GR.line_hits([[0.5, -3, 0.5], [1.5, -3, 0.5]], [[0, 1, 0], [0, 1, 0]], B2)
    assert hit2[0, 0] and not hit2[1, 0]


def test_overlap_sat():
    A = GR.Boxes.from_prims([box_prim([0, 0, 0], [0.1, 0.1, 0.1])])
    near = GR.Boxes.from_prims([box_prim([0.19, 0, 0], [0.1, 0.1, 0.1]), box_prim([0.21, 0, 0], [0.1, 0.1, 0.1])])
    assert GR.overlap(A, near).tolist() == [[True, False]]
    # rotated 45 deg: the corner reaches 0.1 * sqrt(2)
    rot = GR.Boxes.from_prims([box_prim([0.1 + 0.1 * math.sqrt(2) - 0.005, 0, 0], [0.1, 0.1, 0.1], rotz(math.pi / 4)),
                               box_prim([0.1 + 0.1 * math.sqrt(2) + 0.005, 0, 0], [0.1, 0.1, 0.1], rotz(math.pi / 4))])
    assert GR.overlap(A, rot).tolist() == [[True, False]]
    # edge-edge separation that only a cross-product axis finds
    a = GR.Boxes.from_prims([box_prim([0, 0, 0], [1, 0.05, 0.05], rotz(0.0))])
    Rb = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]) @ rotz(math.pi / 4)
    b = GR.Boxes.from_prims([box_prim([0, 0.3, 0.3], [0.05, 0.05, 1], Rb)])
    assert not GR.overlap(a, b)[0, 0]


def test_boxes_placed_and_inflated():
    B = GR.Boxes.from_prims([box_prim([0.1, 0, 0], [0.01, 0.02, 0.03])])
    P = B.placed(rotz(math.pi / 2), [1, 2, 3])
    assert np.allclose(P.C[0], [1, 2.1, 3])
    lo, hi = P.aabb()
    assert np.allclose(hi[0] - lo[0], [0.04, 0.02, 0.06])
    assert np.allclose(B.inflated(0.005).H[0], [0.015, 0.025, 0.035])


# ----------------------------------------------------------------------------------------------- jaw and records
JAW = GR.JawModel()


def test_jaw_gap_and_boxes():
    assert math.isclose(JAW.gap_from_qpos(0.0387, -0.0387), 0.0387 * 2 - 0.001)
    b = JAW.gripper_boxes_G(JAW.L, sweep_up=0.1)
    lo, hi = b.aabb()
    f1 = b.tags.index("finger1")
    assert math.isclose(lo[f1, 1], JAW.L) and math.isclose(hi[f1, 2], JAW.finger_z[1] + 0.1)
    assert JAW.jaw_sweep_boxes_G(0.03).n == 2 and JAW.jaw_sweep_boxes_G(JAW.L + 0.01).n == 0


def test_jaw_from_robot_dict():
    robot = {"finger_joints": {"gripper0_finger_joint1": {"qpos": 0.0387}},
             "geoms": {"gripper0_finger1_pad_collision": {"aabb_eef_lo": [-0.008, 0.0382, -0.0116],
                                                          "aabb_eef_hi": [0.008, 0.0462, 0.0044]},
                       "gripper0_finger1_collision": {"aabb_eef_lo": [-0.0105, 0.0386, -0.0445],
                                                      "aabb_eef_hi": [0.0105, 0.0651, 0.0093]},
                       "gripper0_hand_collision": {"aabb_eef_lo": [-0.0316, -0.104, -0.1229],
                                                   "aabb_eef_hi": [0.0316, 0.1004, -0.031]}}}
    j = GR.JawModel.from_robot(robot)
    assert math.isclose(j.face_offset, -0.0005, abs_tol=1e-9)
    assert j.face_z == pytest.approx((-0.0093, 0.0116))
    assert j.hand_lo[2] == pytest.approx(0.031) and j.hand_hi[2] == pytest.approx(0.1229)


def test_records_on_a_box():
    """A 4 x 6 x 10 cm box standing on z = 0: grasps across 4 and 6 cm, horizontal axes, centres inside."""
    prims = [box_prim([0, 0, 0.05], [0.02, 0.03, 0.05])]
    rec = GR.build_records(prims, [0, 0, 1], [0, 0, 0.05], JAW, GR.RecordParams(theta_step_deg=15, b_step_m=0.005,
                                                                                    z_step_m=0.01,
                                                                                    require_centre_contact=True))
    assert rec["n_records"] > 0
    w = np.round(rec["width"], 4)
    assert {0.04, 0.06} <= set(w.tolist())
    assert np.all(np.abs(rec["axis"][:, 2]) < 1e-9)
    assert np.all(np.abs(rec["centre"][:, :2]).max(axis=0) <= [0.02 + 1e-6, 0.03 + 1e-6])
    assert np.all(rec["z_rel"] + JAW.face_z[0] >= 0.002 - 1e-9)
    assert math.isclose(rec["height"], 0.1)
    # serialisation round trip
    back = GR.records_from_json(json.loads(json.dumps(GR.records_to_json(rec))))
    assert back["centre"].shape == rec["centre"].shape


def test_records_reject_too_wide_and_find_rim():
    wide = [box_prim([0, 0, 0.05], [0.05, 0.05, 0.05])]  # 10 cm cube: no centred grasp fits the 7.6 cm opening
    rec = GR.build_records(wide, [0, 0, 1], [0, 0, 0.05], JAW, GR.RecordParams(theta_step_deg=15))
    assert rec["n_records"] == 0
    # a thin-walled square cup (walls 2 mm): rim grasps across one wall
    t, h, r = 0.001, 0.04, 0.05
    walls = [box_prim([r, 0, h / 2], [t, r, h / 2]), box_prim([-r, 0, h / 2], [t, r, h / 2]),
             box_prim([0, r, h / 2], [r, t, h / 2]), box_prim([0, -r, h / 2], [r, t, h / 2]),
             box_prim([0, 0, t], [r, r, t])]
    rec2 = GR.build_records(walls, [0, 0, 1], [0, 0, 0.01], JAW, GR.RecordParams(theta_step_deg=15))
    assert rec2["n_records"] > 0 and np.all(rec2["width"] <= 0.012) and rec2["width"].min() <= 0.002 + 1e-9


def test_plan_frames_isolated_and_blocked():
    prims = [box_prim([0, 0, 0.05], [0.02, 0.03, 0.05])]
    rec = GR.build_records(prims, [0, 0, 1], [0, 0, 0.05], JAW, GR.RecordParams(theta_step_deg=15, b_step_m=0.005,
                                                                                    z_step_m=0.01))
    target = GR.Boxes.from_prims(prims)
    frames, diag = GR.plan_frames(rec, np.eye(3), np.zeros(3), target, GR.Boxes.empty(), GR.Boxes.empty(), 0.0, JAW,
                                  0.0)
    assert frames and diag["n_free"] > 0
    f = frames[0]
    assert abs(f["axis"][2]) < 1e-12 and f["centre"][2] + JAW.finger_z[0] >= 0.002 - 1e-9
    assert np.allclose(f["R_eef"][:, 2], [0, 0, -1])
    # walls on both x sides 1 cm away block every grasp whose fingers go along x
    walls = GR.Boxes.from_prims([box_prim([0.04, 0, 0.1], [0.005, 0.2, 0.1]), box_prim([-0.04, 0, 0.1],
                                                                                        [0.005, 0.2, 0.1])])
    frames2, _ = GR.plan_frames(rec, np.eye(3), np.zeros(3), target, walls, GR.Boxes.empty(), 0.0, JAW, 0.0)
    for f in frames2:
        assert abs(f["axis"][0]) < 0.3  # the closing axis runs along y, between the walls
    # a reach cap below every candidate leaves no frame
    frames3, d3 = GR.plan_frames(rec, np.eye(3), np.zeros(3), target, GR.Boxes.empty(), GR.Boxes.empty(), 0.0, JAW,
                                 0.0, shoulder=[-0.6, 0, 0.3], reach_cap=0.1)
    assert not frames3 and d3["n_reach_drop"] > 0


# ----------------------------------------------------------------------------------------------- tools
BDDL = """(define (problem LIBERO_Floor_Manipulation)
  (:domain robosuite)
  (:language Pick the alphabet soup and place it in the basket)
  (:goal
    (And (In alphabet_soup_1 basket_1_contain_region))
  )
)"""


def test_parse_goal_and_task_spec():
    assert T.parse_goal(BDDL) == [("in", "alphabet_soup_1", "basket_1_contain_region")]
    scene = {"instances": {"alphabet_soup_1": {}, "basket_1": {}},
             "sites": {"basket_1_contain_region": {"parent_instance": "basket_1"}}}
    spec = T.task_spec(BDDL, scene)
    assert spec["place_kind"] == "region" and spec["place_instance"] == "basket_1"
    assert spec["instruction"].startswith("Pick the alphabet soup")
    two = BDDL.replace("(And (In alphabet_soup_1 basket_1_contain_region))",
                       "(And (On akita_black_bowl_1 plate_1))")
    scene2 = {"instances": {"akita_black_bowl_1": {}, "plate_1": {}}, "sites": {}}
    assert T.task_spec(two, scene2)["place_kind"] == "object"


def test_filter_obs_drops_truth():
    obs = {"robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0, 0, 0, 1.0]), "basket_1_pos": np.ones(3),
           "basket_1_to_robot0_eef_pos": np.ones(3), "object-state": np.ones(5), "agentview_image": np.zeros((2, 2))}
    out = T.filter_obs(obs)
    assert set(out) == {"robot0_eef_pos", "robot0_eef_quat"} and isinstance(out["robot0_eef_pos"], list)


def test_oracle_tool_counts_calls():
    tool = T.OracleTool(lambda: {"b": {"pos": [0, 0, 0], "quat_wxyz": [1, 0, 0, 0]}})
    tool.t_now = 7
    tool()
    tool()
    assert tool.calls == 2 and tool.call_steps == [7, 7]
    assert not any(hasattr(tool, n) for n in ("env", "sim"))
