"""Tests of the K0-stereo poses module and the hand model file (synthetic numpy roots; no LIBERO frame).

The registered text these numbers were fixed in (``evidence-m3/dev-targets-m3v2.json``) is not part of this release,
so the lines that compared the module's constants with that file's wording are not here. The CLI tests need the code
in a git checkout (each process records the code tree from git) and skip without one.

Synthetic scenes are built deterministically (no random draw): each capture has a previous-step hand pose, a one-step
body twist, the obs hand pose one step later, and the drawn hand pose ``DT_TRUE`` ms after the obs sample (constant
body velocity), so ``Ecorr(dt_lag = DT_TRUE) X`` is the simulator T exactly. The CLI tests run child interpreters
(the strict hook is never installed in the pytest process).
"""

from __future__ import annotations

import ast
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rowant_lab.libero_env import m2_camera as CAM
from rowant_lab.libero_env import m3_stereo1_audit as AU
from rowant_lab.libero_env import m3_stereo1_poses as PO

SRC = Path(PO.__file__).read_text()
REPO = AU.REPO_ROOT
DT_TRUE = 2.0
X_TRUE = PO.pose_mat([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [0.05, 0.0, -0.097])
E_BASE = PO.pose_mat(PO.so3_exp([3.05, 0.12, -0.2]), [0.08, -0.04, 1.15])
TOKENS1 = PO.RS_TOKENS + ("pgr0", "pgp1", "pgp2", "pgp3", "pgm01", "pgm02", "ppr0", "ppm01")
TOKENS2 = ("C", "R", "L")
STATIC = ("rs00", "C", "pgr0", "ppr0")


def _twists(i: int, moving: bool) -> tuple:
    a = np.array([0.02 * math.sin(i), 0.03 * math.cos(i), 0.01 * i / 10, 0.01 * i, -0.004 * i, 0.002 * (i % 3)])
    if not moving:
        return a, np.zeros(6)
    xi = np.array([0.01 * math.cos(i), 0.012 * math.sin(i + 1), 0.008, 0.006 * math.cos(i), 0.005,
                   -0.004 * math.sin(i)])
    return a, xi


def capture(i: int, moving: bool, dt_true: float = DT_TRUE, X=X_TRUE) -> dict:
    a, xi = _twists(i, moving)
    Ep = E_BASE @ PO.se3_exp(a)
    E = Ep @ PO.se3_exp(xi)
    Er = Ep @ PO.se3_exp((1.0 + dt_true / PO.CONTROL_DT_MS) * xi)
    return {"E_prev": Ep, "E": E, "E_rendered": Er, "T": Er @ X, "jv": 0.0 if not moving else 0.1}


def _policy(c: dict, camera: str, W: int, step: int) -> dict:
    return {"rgb": np.zeros((2, 2, 3), np.uint8), "camera": np.array(camera), "width": np.array(W),
            "step": np.array(step), "robot0_joint_pos": np.zeros(7), "robot0_joint_vel": np.full(7, c["jv"]),
            "robot0_eef_pos": c["E"][:3, 3].copy(), "robot0_eef_quat": PO.mat_to_quat_xyzw(c["E"][:3, :3]),
            "robot0_gripper_qpos": np.array([0.02, -0.02]), "robot0_gripper_qvel": np.zeros(2),
            "prev_eef_pos": c["E_prev"][:3, 3].copy(), "prev_eef_quat": PO.mat_to_quat_xyzw(c["E_prev"][:3, :3]),
            "prev_step": np.array(step - 1), "obs_sha256": np.array("0" * 64), "look_id": np.array("x"),
            "look_k": np.array(0), "steps_since_look_start": np.array(0), "code_tree": np.array("0" * 64),
            "clean": np.array(False)}


def _cam(c: dict, T, K) -> dict:
    m, r = PO.pose_err(c["E"], c["E_rendered"])
    return {"K": K, "T_world_from_camera": T, "E_rendered": c["E_rendered"], "qpos_after": np.zeros(9),
            "eef_lag": np.array(json.dumps({"pos_mm": m * 1000.0, "rot_rad": r})),
            "truth_snapshot": np.array("{}")}


def make_scene(R: Path, sid: str, dt_true: float = DT_TRUE, X=X_TRUE, salt: int = 0) -> None:
    (R / "frames" / sid).mkdir(parents=True, exist_ok=True)
    (R / "cam_truth" / sid).mkdir(parents=True, exist_ok=True)
    for traj, toks in (("traj1", TOKENS1), ("traj2", TOKENS2)):
        for i, tok in enumerate(toks):
            c = capture(i + 17 * (traj == "traj2") + salt, tok not in STATIC, dt_true, X)
            for W in PO.WIDTHS:
                v = PO.view_id(traj, tok, PO.WRIST, W)
                K = CAM.intrinsics(75.0, W, W)
                np.savez(R / "frames" / sid / f"{v}.npz", **_policy(c, PO.WRIST, W, 10 + i))
                np.savez(R / "cam_truth" / sid / f"{v}.npz", **_cam(c, c["T"], K))
            if traj == "traj1" and tok == "rs00":
                v = PO.view_id("traj1", "rs00", PO.AGENT, 1024)
                Ta = PO.pose_mat(PO.so3_exp([2.2, 0.0, 0.0]), [0.9, 0.0, 1.6])
                np.savez(R / "frames" / sid / f"{v}.npz", **_policy(c, PO.AGENT, 1024, 10))
                np.savez(R / "cam_truth" / sid / f"{v}.npz", **_cam(c, Ta, CAM.intrinsics(45.0, 1024, 1024)))


def make_root(tmp_path: Path, scenes=("libero_object-t00-s00", "libero_spatial-t03-s01"), name="R", **kw) -> Path:
    R = tmp_path / name
    R.mkdir(parents=True)
    (R / AU.SYNTHETIC_MARKER).write_text("synthetic\n")
    for k, sid in enumerate(scenes):
        make_scene(R, sid, salt=3 * k, **kw)
    return R


def pick_doc(dt: int = 2) -> dict:
    return {"dt_lag_ms": dt, "X_true": X_TRUE.tolist(), "f_true": {"512": PO.f_of_fovy(75.0, 512),
                                                                     "1024": PO.f_of_fovy(75.0, 1024)},
            "code_tree": "a" * 64}


def run_cli(*args, cwd: Path, env_extra=None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ROWANT_", "GIT_"))}
    env.update(PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    env.update(env_extra or {})
    cwd.mkdir(parents=True, exist_ok=True)
    return subprocess.run([sys.executable, "-B", "-m", "rowant_lab.libero_env.m3_stereo1_poses", *map(str, args)],
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=300)


def _git_checkout() -> bool:
    try:
        AU.code_tree(REPO, "HEAD")
        return True
    except (AU.GitError, OSError):
        return False


needs_git = pytest.mark.skipif(not _git_checkout(), reason="the code is not in a git checkout with a commit: the pose "
                                                           "CLIs record the code tree from git")


def audit_rows(R: Path, process: str) -> list:
    out = []
    for p in sorted((R / "audit").glob(f"{process}-*.jsonl")):
        out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return out


# ----------------------------------------------------------------------------------------------------- math
def test_quaternion_and_se3_round_trips():
    for w in ([0.0, 0.0, 0.0], [1e-9, -2e-9, 0.0], [0.3, -0.2, 0.1], [0.0, 0.0, math.pi - 1e-9], [2.0, 1.0, -0.5]):
        R = PO.so3_exp(w)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.allclose(PO.so3_exp(PO.so3_log(R)), R, atol=1e-9)
        q = PO.mat_to_quat_xyzw(R)
        assert np.allclose(PO.quat_to_mat(q, "xyzw"), R, atol=1e-12)
        assert np.allclose(PO.quat_to_mat(np.roll(q, 1), "wxyz"), R, atol=1e-12)
    for xi in (np.zeros(6), np.array([0.1, -0.2, 0.05, 0.01, 0.02, -0.03]), np.array([1e-9, 0, 0, 0.5, 0, 0])):
        M = PO.se3_exp(xi)
        assert np.allclose(PO.se3_log(M), xi, atol=1e-10)
    M = PO.se3_exp([0.2, 0.1, -0.3, 0.4, 0.5, 0.6])
    assert np.allclose(PO.inv_pose(M) @ M, np.eye(4), atol=1e-12)
    # robosuite's quaternion is xyzw: (x, y, z, w) = (0, 0, sin(a/2), cos(a/2)) turns about z
    a = 0.7
    Rz = PO.quat_to_mat([0.0, 0.0, math.sin(a / 2), math.cos(a / 2)])
    assert np.allclose(Rz, [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def test_ecorr_matches_constant_body_velocity():
    E0 = PO.pose_mat(PO.so3_exp([0.4, -1.1, 0.3]), [0.2, 0.1, 1.0])
    xi = np.array([0.05, -0.03, 0.08, 0.01, 0.02, -0.015])
    E1 = E0 @ PO.se3_exp(xi)
    for dt in (0, 2, 4, 6, -3.5):
        want = E0 @ PO.se3_exp((1 + dt / 50.0) * xi)
        assert np.allclose(PO.ecorr(E1, E0, dt), want, atol=1e-12)
    assert PO.ecorr(E1, None, 0) is not E1 and np.array_equal(PO.ecorr(E1, None, 0), E1)


def test_k_true_is_the_wrist_cameras_intrinsics():
    assert round(PO.f_of_fovy(75.0, 512), 2) == 333.63 and round(PO.f_of_fovy(75.0, 1024), 2) == 667.25
    for W in PO.WIDTHS:
        assert np.array_equal(PO.K_true(PO.f_of_fovy(75.0, W), W), CAM.intrinsics(75.0, W, W))


# ----------------------------------------------------------------------------------------------------- P-O
def _pairs(n: int = 6, moving: bool = True, dt_true: float = DT_TRUE) -> list:
    out = []
    for i in range(n):
        c = capture(i, moving, dt_true)
        for W in PO.WIDTHS:
            fr = {k: (v.item() if np.ndim(v) == 0 else v) for k, v in _policy(c, PO.WRIST, W, i).items()}
            out.append(("v", fr, {"T_world_from_camera": c["T"], "K": CAM.intrinsics(75.0, W, W),
                                  "E_rendered": c["E_rendered"]}))
    return out


def test_po_formula_and_settled_check_with_injected_faults():
    pick = pick_doc(2)
    moving = _pairs(moving=True)
    for _, fr, cam in moving:
        T, K = PO.po_view_T("S3", fr, pick)
        assert np.allclose(T, cam["T_world_from_camera"], atol=1e-12)
        assert np.array_equal(K, cam["K"])
    assert PO.settled_pose_numbers(moving, pick)["pass"]
    # dt_lag 0 on moving views: the uncorrected one-substep lag is visible (> 0.1 mm here)
    assert not PO.settled_pose_numbers(moving, pick_doc(0))["pass"]
    still = _pairs(moving=False)
    for dt in PO.DT_LAG_CANDIDATES_MS:
        r = PO.settled_pose_numbers(still, pick_doc(dt))
        assert r["pass"] and r["centre_mm_max"] < 1e-9 and r["n_views"] == 12
    # injected faults: the quaternion read in the wrong order, X on the wrong side
    r = PO.settled_pose_numbers(still, pick, order="wxyz")
    assert not r["pass"] and r["rot_deg_max"] > 1.0
    r = PO.settled_pose_numbers(still, pick, x_side="left")
    assert not r["pass"] and r["centre_mm_max"] > 1.0
    # a missing previous obs makes the pose missing (counted above every value) when dt_lag is not 0
    _, fr, cam = still[0]
    fr = dict(fr, prev_eef_quat=np.zeros(4))
    assert PO.po_view_T("S3", fr, pick) is None and PO.po_view_T("S3", fr, pick_doc(0)) is not None
    assert not PO.settled_pose_numbers([("v", fr, cam)], pick)["pass"]


def test_po_variants_d1_b1_b2_sweep():
    pick = pick_doc(2)
    c = capture(3, True)
    fr = {k: (v.item() if np.ndim(v) == 0 else v) for k, v in _policy(c, PO.WRIST, 512, 3).items()}
    T3, K3 = PO.po_view_T("S3", fr, pick)
    for arm in ("S3-RM", "S3-b1", "S3-b2", "P1", "P2"):
        T, K = PO.po_view_T(arm, fr, pick)
        assert np.array_equal(T, T3) and np.array_equal(K, K3)
    T, _ = PO.po_view_T("S3-D1", fr, pick)
    assert np.allclose(T, c["E_prev"] @ X_TRUE, atol=1e-12)
    for f in PO.SWEEP_FACTORS:
        for lv in PO.SWEEP_LEVELS[f]:
            T, K = PO.po_view_T(PO.sweep_arm(f, lv), fr, pick)
            D, s = PO.sweep_delta(f, lv)
            assert np.allclose(T, T3 @ D, atol=1e-12)
            assert np.allclose(np.linalg.inv(T3) @ T, D, atol=1e-9)
            if f == "xrot":
                assert math.isclose(math.degrees(PO.rot_angle(D[:3, :3])), float(lv), rel_tol=1e-9)
                assert np.array_equal(K, K3)
            elif f == "xtrans":
                assert math.isclose(np.linalg.norm(D[:3, 3]) * 1000, float(lv), rel_tol=1e-9)
            else:
                assert math.isclose(K[0, 0] / K3[0, 0], 1 + PO.SWEEP_DRAWS[(f, lv)][1] * float(lv) / 100)
                assert K[0, 2] == K3[0, 2] == 256.0 and np.array_equal(T, T3)
    frames = {PO.view_id("traj1", t, PO.WRIST, 512): fr for t in ("pgr0", "pgp1", "pgp2", "pgp3", "pgm01")}
    for arm, want in (("S3", 4), ("S3-b1", 2), ("S3-b2", 3)):
        doc, missing = PO.po_look_doc(arm, "traj1-pg", frames, pick, "c" * 64, False)
        assert len(doc["width_views"]) == want and not missing and doc["X"] == X_TRUE.tolist()
    doc, _ = PO.po_look_doc("S3-sweep-xtrans-8", "traj1-pg", frames, pick, "c" * 64, True)
    assert doc["X"] == X_TRUE.tolist() and doc["clean"] is True  # Delta is applied to T only
    assert PO.po_arm_ok("S3-RM", "traj1-pg") and not PO.po_arm_ok("S3-RM", "traj1-pgm")
    assert not PO.po_arm_ok("S3", "traj2-E2") and not PO.po_arm_ok("S3-sweep-xrot-0.1", "traj1-pg")
    assert PO.po_arm_ok("S3-sweep-xrot-0.1", "traj1-rs") and PO.po_arm_ok("S2", "traj1-rs")
    with pytest.raises(ValueError):
        PO.parse_sweep_arm("S3-sweep-xrot-9")


def test_sweep_draws_regenerate_from_seed_7501():
    regen = PO.regenerate_sweep_draws()
    assert list(regen) == list(PO.SWEEP_DRAWS)
    for k, (axis, sign) in PO.SWEEP_DRAWS.items():
        assert regen[k][1] == sign
        assert np.allclose(regen[k][0], axis, atol=1e-6)
    assert len(PO.SWEEP_ARMS) == 16 and PO.SWEEP_ARMS[0] == "S3-sweep-xrot-0.1"


def test_look_max_rel_rot_deg():
    pick = pick_doc(2)
    frames = {}
    for i, t in enumerate(("pgr0", "pgp1", "pgp2", "pgp3")):
        c = capture(i, False)
        frames[PO.view_id("traj1", t, PO.WRIST, 512)] = {
            k: (v.item() if np.ndim(v) == 0 else v) for k, v in _policy(c, PO.WRIST, 512, i).items()}
    doc, _ = PO.po_look_doc("S3", "traj1-pg", frames, pick, "c" * 64, False)
    Ts = {PO.parse_view(v)["token"]: np.array(e["T"]) for v, e in doc["width_views"].items()}
    want = max(math.degrees(PO.rot_angle(Ts["pgr0"][:3, :3].T @ Ts[t][:3, :3])) for t in ("pgp1", "pgp2", "pgp3"))
    assert math.isclose(PO.look_max_rel_rot_deg(doc), want, rel_tol=1e-12) and want > 0.1


# ----------------------------------------------------------------------------------------------------- tune checks
def _caps(R: Path, scenes) -> list:
    return PO.captures_of(R, scenes)[0]


def test_x_true_lag_present_dt_pick_on_synthetic_root(tmp_path):
    R = make_root(tmp_path)
    scenes = sorted(os.listdir(R / "frames"))
    caps, wdiff = PO.captures_of(R, scenes)
    assert wdiff == 0.0 and len(caps) == 2 * (len(TOKENS1) + len(TOKENS2))
    X, xr = PO.x_true_numbers(caps)
    assert xr["pass"] and np.allclose(X, X_TRUE, atol=1e-12)
    lag = PO.lag_present_numbers(caps)
    assert lag["pass"] and lag["n_moving"] == 2 * 14 and lag["max_gap_to_render_eef_lag_mm"] < 1e-9
    dt, dr = PO.dt_lag_pick(caps, X)
    assert dt == 2 and dr["median_mm"]["2"] < 1e-9 < dr["median_mm"]["0"]
    f, fr = PO.f_true_numbers(R, scenes)
    assert fr["pass"] and f == {"512": PO.f_of_fovy(75, 512), "1024": PO.f_of_fovy(75, 1024)}


def test_x_true_check_stops_on_a_1e6_deviation(tmp_path):
    R = make_root(tmp_path, scenes=("libero_object-t00-s00",))
    caps = _caps(R, ["libero_object-t00-s00"])
    bad = dict(caps[5])
    cam = dict(bad["cam"])
    T = cam["T_world_from_camera"].copy()
    T[:3, 3] += [2e-6, 0.0, 0.0]
    cam["T_world_from_camera"] = T
    bad["cam"] = cam
    X, xr = PO.x_true_numbers(caps[:5] + [bad] + caps[6:])
    assert not xr["pass"] and 1.9e-6 < xr["max_dev_m"] < 2.1e-6
    rot = dict(caps[6]["cam"])
    rot["T_world_from_camera"] = rot["T_world_from_camera"] @ PO.pose_mat(PO.so3_exp([0, 0, 3e-6]), np.zeros(3))
    _, xr = PO.x_true_numbers(caps[:6] + [dict(caps[6], cam=rot)] + caps[7:])
    assert not xr["pass"] and xr["max_dev_rad"] > 2.9e-6


def test_lag_present_fails_without_lag_and_dt_tie_prefers_smaller(tmp_path):
    R = make_root(tmp_path, scenes=("libero_object-t00-s00",), dt_true=0.0)
    caps = _caps(R, ["libero_object-t00-s00"])
    lag = PO.lag_present_numbers(caps)
    assert not lag["pass"] and lag["median_mm"] < 1e-9
    X, _ = PO.x_true_numbers(caps)
    dt, dr = PO.dt_lag_pick(caps, X)
    assert dt == 0
    still = [c for c in caps if c["token"] in STATIC]
    dt, dr = PO.dt_lag_pick(still, X)  # every candidate ties (static captures): the smallest wins
    assert dt == 0 and all(v < 1e-9 for v in dr["median_mm"].values())


def test_poses_pick_schema_rejects_per_scene_entries():
    good = pick_doc(2)
    assert PO.check_poses_pick(good) is good
    for bad in (dict(good, **{"libero_object-t00-s00": {"dt_lag_ms": 2}}), dict(good, scenes=["a"]),
                dict(good, f_true={**good["f_true"], "libero_object-t00-s00": 333.6}),
                dict(good, f_true={"512": {"libero_object-t00-s00": 333.6}, "1024": 667.25}),
                dict(good, f_true={"512": 333.6}), dict(good, dt_lag_ms=3), dict(good, dt_lag_ms=True),
                dict(good, X_true=np.eye(3).tolist()), dict(good, code_tree="short"),
                {k: v for k, v in good.items() if k != "code_tree"}, dict(good, n_captures=40), [good]):
        with pytest.raises(ValueError):
            PO.check_poses_pick(bad)
    assert PO.POSES_PICK_KEYS == ("dt_lag_ms", "X_true", "f_true", "code_tree")


# ----------------------------------------------------------------------------------------------------- the CLIs
@needs_git
def test_cli_pt_poses_pick_settled_po_end_to_end(tmp_path):
    tune = ("libero_object-t00-s00", "libero_spatial-t03-s01")
    score = "libero_object-t00-s05"
    R = make_root(tmp_path, scenes=tune + (score,))
    cwd = tmp_path / "cwd"
    # P-T: tune scenes only before C4; the scoring scene's cam_truth is never opened
    r = run_cli("pt", "--root", R, cwd=cwd)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["scenes"] == 2 and score in out["refused"] and out["written"] == 2 * 2 * 5 and not out["differs"]
    for arm in PO.PT_ARMS:
        for look in PO.LOOKS:
            doc = json.loads(PO.pose_path(R, arm, tune[0], look).read_text())
            assert set(doc) == {"arm", "look", "X", "width_views", "code_tree", "clean"} and doc["arm"] == arm
            assert np.allclose(doc["X"], X_TRUE, atol=1e-12) and doc["clean"] is False
            for v, e in doc["width_views"].items():
                cam = PO.read_cam(R / "cam_truth" / tune[0] / f"{v}.npz")
                assert np.array_equal(np.array(e["T"]), cam["T_world_from_camera"])
                assert np.array_equal(np.array(e["K"]), cam["K"])
        assert not (R / "poses" / arm / score).exists()
    a0 = json.loads(PO.pose_path(R, "A0", tune[0], "traj1-rs").read_text())
    s2 = json.loads(PO.pose_path(R, "S2", tune[0], "traj1-rs").read_text())
    assert set(a0["width_views"]) - set(s2["width_views"]) == {"traj1-rs00-agentview-1024"}
    assert len(s2["width_views"]) == 22
    assert not [x for x in audit_rows(R, "pt") if score in str(x.get("path"))]
    r = run_cli("pt", "--root", R, cwd=cwd)
    assert r.returncode == 0 and json.loads(r.stdout)["same"] == 20
    r = run_cli("pt", "--root", R, "--scenes", score, cwd=cwd)
    assert r.returncode == 2
    # poses-pick (tune scenes only) and the settled-pose check
    r = run_cli("poses-pick", "--root", R, "--pass-tag", "test", cwd=cwd)
    assert r.returncode == 0, r.stderr
    pk = PO.read_poses_pick(R)
    assert pk["dt_lag_ms"] == 2 and np.allclose(pk["X_true"], X_TRUE, atol=1e-12)
    rows = [json.loads(x) for x in (R / "eval" / "pipeline_checks.jsonl").read_text().splitlines()]
    assert [x["check"] for x in rows] == ["lag_present", "x_true", "f_true", "dt_lag_pick"]
    assert all(x["pass"] and x["pass_tag"] == "test" and x["root_kind"] == "synthetic" for x in rows)
    assert not [x for x in audit_rows(R, "poses-pick") if score in str(x.get("path"))]
    # P-O (strict hook): S3 on three looks, then S3-b1, then a renamed scene folder
    sha = AU.sha256_file(AU.HAND_MODEL)
    keys = [f"{tune[0]}/traj1-rs/512", f"{tune[0]}/traj1-pg/512", f"{tune[0]}/traj2-E2/1024"]
    base = ["--method", "PS", "--config", "PS-w5-loose", "--width", "512", "--pass-tag", "test", "--gen", "g0",
            "--hand-sha", sha, "--code-tree", "c" * 64, "--clean", "0"]
    r = run_cli("po", "--root", R, "--arm", "S3", "--looks", *keys, *base, cwd=cwd)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["written"] == 3
    rows = audit_rows(R, "po")
    assert rows[0]["strict"] is True and rows[-1]["n_violations"] == 0
    assert not [x for x in rows if "cam_truth" in str(x.get("path")) or "truth" in str(x.get("path", "")).split("/")]
    for look in ("traj1-rs", "traj1-pg", "traj2-E2"):
        doc = json.loads(PO.pose_path(R, "S3", tune[0], look).read_text())
        assert doc["code_tree"] == "c" * 64 and doc["clean"] is False
        for v, e in doc["width_views"].items():
            cam = PO.read_cam(R / "cam_truth" / tune[0] / f"{v}.npz")
            assert np.allclose(np.array(e["T"]), cam["T_world_from_camera"], atol=1e-12)
    r = run_cli("po", "--root", R, "--arm", "S3", "--looks", *keys, *base, cwd=cwd)
    assert r.returncode == 0 and json.loads(r.stdout)["same"] == 3
    r = run_cli("po", "--root", R, "--arm", "S3-b1", "--looks", f"{tune[0]}/traj1-pg/512", *base, cwd=cwd)
    assert r.returncode == 0, r.stderr
    b1 = json.loads(PO.pose_path(R, "S3-b1", tune[0], "traj1-pg").read_text())
    assert sorted({PO.parse_view(v)["token"] for v in b1["width_views"]}) == ["pgp1", "pgr0"]
    r = run_cli("po", "--root", R, "--arm", "S3-RM", "--looks", f"{tune[0]}/traj1-pg/512", *base, cwd=cwd)
    assert r.returncode == 2
    r = run_cli("po", "--root", R, "--arm", "S3", "--looks", f"{tune[0]}/traj1-pg/512", *base[:-6],
                "--hand-sha", "0" * 64, "--code-tree", "c" * 64, "--clean", "0", cwd=cwd)
    assert r.returncode == 2 and "hand model sha256" in r.stderr
    # the hand-model sha is required: a P-O process started without it refuses before its hook and any read
    n_audit = len(list((R / "audit").glob("po-*.jsonl")))
    r = run_cli("po", "--root", R, "--arm", "S3", "--looks", f"{tune[0]}/traj1-pg/512", *base[:-6],
                "--code-tree", "c" * 64, "--clean", "0", cwd=cwd)
    assert r.returncode == 2 and "--hand-sha" in r.stderr
    assert len(list((R / "audit").glob("po-*.jsonl"))) == n_audit
    # scene ids are opaque: the same frames under another folder name give the same pose file
    shutil.copytree(R / "frames" / tune[0], R / "frames" / "opaque01")
    r = run_cli("po", "--root", R, "--arm", "S3", "--looks", "opaque01/traj1-pg/512", *base, cwd=cwd)
    assert r.returncode == 0, r.stderr
    assert PO.pose_path(R, "S3", "opaque01", "traj1-pg").read_bytes() == \
        PO.pose_path(R, "S3", tune[0], "traj1-pg").read_bytes()
    # settled-pose, with the written S3 files compared too
    r = run_cli("settled-pose", "--root", R, "--pass-tag", "test", cwd=cwd)
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout)
    assert res["pass"] and res["n_rs00"] == 4 and res["n_e2c_settled"] == 4 and res["n_pose_file_views"] == 4
    assert res["n_pose_file_mismatch"] == 0 and res["n_moving"] == 56
    # these synthetic captures model the 2 ms lag exactly (constant body velocity), so dt_lag 2 reproduces P-T to
    # float noise on every moving view; with dt_lag 0 the lag shows on every one
    assert res["n_moving_identical_centres"] == 56
    r = run_cli("settled-pose", "--root", R, "--pass-tag", "test", "--dt-lag-ms", "0", cwd=cwd)
    res0 = json.loads(r.stdout)
    assert r.returncode == 0 and res0["dt_source"].startswith("override")
    assert res0["n_moving_identical_centres"] == 0 and res0["moving_centre_mm_median"] > 0.1


@needs_git
def test_moving_identical_centres_flag_a_policy_file_carrying_rendered_state(tmp_path):
    """No obs lag (the policy file carries the drawn state) with dt_lag 0: every moving P-O centre is P-T's."""
    R = make_root(tmp_path, scenes=("libero_object-t00-s00",), dt_true=0.0)
    (R / "eval").mkdir()
    (R / "eval" / "poses_pick.json").write_text(json.dumps(pick_doc(0)))
    r = run_cli("settled-pose", "--root", R, "--pass-tag", "test", cwd=tmp_path / "cwd")
    res = json.loads(r.stdout)
    assert r.returncode == 0 and res["n_moving"] == 28 and res["n_moving_identical_centres"] == 28


def test_cli_refuses_unregistered_roots_and_force(tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    r = run_cli("pt", "--root", bare, cwd=tmp_path / "cwd")
    assert r.returncode == 2 and "REFUSE" in r.stderr
    r = run_cli("pt", "--root", AU.DEPTH1D, cwd=tmp_path / "cwd")
    assert r.returncode == 2
    R = make_root(tmp_path, scenes=("libero_object-t00-s00",))
    r = run_cli("pt", "--root", R, "--force", cwd=tmp_path / "cwd")
    assert r.returncode == 2 and "--force" in r.stderr


@needs_git
def test_poses_pick_refuses_without_lag(tmp_path):
    R = make_root(tmp_path, scenes=("libero_object-t00-s00",), dt_true=0.0)
    r = run_cli("poses-pick", "--root", R, "--pass-tag", "test", cwd=tmp_path / "cwd")
    assert r.returncode == 1
    assert not (R / "eval" / "poses_pick.json").exists()
    rows = [json.loads(x) for x in (R / "eval" / "pipeline_checks.jsonl").read_text().splitlines()]
    assert rows[0]["check"] == "lag_present" and rows[0]["pass"] is False


def test_poses_module_imports_and_forbidden_calls():
    tree = ast.parse(SRC)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods |= {f"{n.module}.{a.name}" if n.module == "rowant_lab.libero_env" else n.module for a in n.names}
    assert mods <= {"__future__", "argparse", "json", "math", "os", "sys", "pathlib", "numpy",
                    "rowant_lab.libero_env.m3_stereo1_audit"}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not ({"subprocess", "socket"} & (names | mods))
    assert not ({"system", "posix_spawn", "posix_spawnp", "popen", "fork", "spawnv"} & attrs)
    assert not [a for a in attrs if a.startswith("exec") and a != "exec_prefix"]
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in ("memmap", "imread")]


def _po_scope_banned(src: str) -> tuple:
    """(names reached from cmd_po, banned string constants in them): module functions and module-level names."""
    tree = ast.parse(src)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assigns = {}
    for n in tree.body:
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                if isinstance(t, ast.Name):
                    assigns[t.id] = n
    seen, todo = set(), ["cmd_po"]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        node = fns.get(name) or assigns.get(name)
        todo += [x.id for x in ast.walk(node) if isinstance(x, ast.Name) and (x.id in fns or x.id in assigns)]
    bad = [(name, c.lineno, c.value) for name in sorted(seen) for c in ast.walk(fns.get(name) or assigns[name])
           if isinstance(c, ast.Constant) and isinstance(c.value, str)
           and any(w in c.value.lower() for w in ("truth", "workcell", "frozen-config"))]
    return seen, bad


def test_po_code_holds_no_truth_workcell_or_frozen_config_constant():
    """methods.process_and_imports for P-O: no string constant containing 'truth', 'workcell' or 'frozen-config' in
    P-O's code (cmd_po and every module function it reaches, plus the module-level names they read). P-T, poses-pick
    and settled-pose share the module and read cam_truth, so the ban is scoped to P-O's call graph."""
    seen, bad = _po_scope_banned(SRC)
    assert {"cmd_po", "po_look_doc", "po_view_T", "read_poses_pick", "read_policy", "write_or_compare"} <= seen
    assert not bad, bad
    # the scan is live: a constant planted in a function P-O reaches is found
    planted = SRC.replace("def po_view_T(", "def _planted():\n    return 'cam_truth'\n\n\ndef po_view_T(", 1)
    planted = planted.replace("def po_look_doc(", "def po_look_doc(*_a, _p=_planted, **_k):\n    _p()\n\n\n"
                              "def _po_look_doc_orig(", 1)
    _seen2, bad2 = _po_scope_banned(planted)
    assert [b for b in bad2 if b[2] == "cam_truth"]


# ----------------------------------------------------------------------------------------------------- hand model
def _export_ok() -> bool:
    from rowant_lab.libero_env import m2_paths as MP

    return (MP.EXPORT_DIR / "panda-libero_spatial.json").exists() and (MP.POLICY_LIB / "manifest.json").exists()


@pytest.mark.skipif(not _export_ok(),
                    reason="the object library (its export and policy folders) is not built on this machine")
def test_hand_model_reproduces_m2_kin_fk():
    from rowant_lab.libero_env import m3_stereo1_handmodel as HM

    model = HM.load()
    assert set(model) == {"format", "source", "frame", "bodies"} and model["format"] == HM.FORMAT
    assert model["frame"] == "eef" and [b["name"] for b in model["bodies"]] == list(HM.HAND_BODIES)
    for b in model["bodies"]:
        assert set(b) == {"name", "T_eef_from_body_at_zero", "joint", "vertices", "faces"}
    words = [s.lower() for s in HM.keys_and_strings(model) if not s.startswith("rowant_lab/libero_env/m3_")]
    assert not [w for w in HM.FORBIDDEN_WORDS if any(w in s for s in words)]
    pandas, _, _ = HM._export_pandas()
    for suite, panda in pandas.items():
        res = HM.fk_check(model, panda, HM.CHECK_JOINTS, HM.CHECK_GRIPPERS)
        assert res["n_poses"] == 45 and res["max_abs_R"] <= 1e-9 and res["max_abs_t_m"] <= 1e-9, (suite, res)
        for b in model["bodies"]:
            (key,) = panda.bodies[b["name"]]["geoms"]
            # The stored model was made from an export built on an Apple-silicon Mac. An export built on another
            # kind of processor has the same meshes with other last digits in the vertices
            # (docs/CHANGES-FROM-AS-RUN.md), so the vertices are compared within a micrometre and the faces exactly.
            stored, built = np.array(b["vertices"]), panda.arrays[f"{key}__V"]
            assert stored.shape == built.shape and float(np.abs(stored - built).max()) <= 1e-6, (suite, b["name"])
            assert np.array_equal(np.array(b["faces"]), panda.arrays[f"{key}__F"])
    # the fingers open along the hand y axis, in opposite directions, with the grip site 0.097 m down hand z
    assert model["source"]["grip_site_in_hand_m"] == [0.0, 0.0, 0.097]
    left = HM.body_poses(model, [0.04, -0.04])
    zero = HM.body_poses(model, [0.0, 0.0])
    assert np.allclose(left["gripper0_leftfinger"][:3, 3] - zero["gripper0_leftfinger"][:3, 3], [0, 0.04, 0])
    assert np.allclose(left["gripper0_rightfinger"][:3, 3] - zero["gripper0_rightfinger"][:3, 3], [0, -0.04, 0])
    V, F = HM.posed_vertices(model, [0.02, -0.02])
    assert V.shape == (3613 + 318 + 318, 3) and F.max() == len(V) - 1
    # the file is the tool's own output, byte for byte, and its build is deterministic
    assert HM.dumps(HM.make_doc()) == HM.OUT.read_bytes()
