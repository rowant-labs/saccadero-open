"""M2 unit tests (plan precondition P4): the grounding parser and relations, the bridge protocol, the kinematics,
the camera tool's depth and intrinsics maths, and the M2 library's contents.

M2 (perceived poses, grounded instruction): perception in the loop. Monty venv; nothing here imports LIBERO,
MuJoCo or torch::

    PYTHONPATH=$PWD <server python> -P -m pytest \\
        rowant_lab/libero_env/tests/test_m2_units.py -q
"""

from __future__ import annotations

import json
import math
import socket
import threading
from pathlib import Path

import numpy as np
import pytest

from rowant_lab.libero_env import m2_bridge as BR
from rowant_lab.libero_env import m2_camera as CAM
from rowant_lab.libero_env import m2_ground as GD
from rowant_lab.libero_env import m2_kin as KIN
from rowant_lab.libero_env import m2_paths as MP

TYPES = sorted(set(MP.WORKCELL_TYPES["libero_object"]) | set(MP.WORKCELL_TYPES["libero_spatial"]))
INSTRUCTIONS = {
    "libero_object": [(f"pick the {n} and place it in the basket", t) for n, t in
                      (("alphabet soup", "alphabet_soup"), ("bbq sauce", "bbq_sauce"), ("butter", "butter"),
                       ("chocolate pudding", "chocolate_pudding"), ("cream cheese", "cream_cheese"),
                       ("ketchup", "ketchup"), ("milk", "milk"), ("orange juice", "orange_juice"),
                       ("salad dressing", "salad_dressing"), ("tomato sauce", "tomato_sauce"))],
    "libero_spatial": [
        ("pick the akita black bowl between the plate and the ramekin and place it on the plate", "between"),
        ("pick the akita black bowl from table center and place it on the plate", "table_centre"),
        ("pick the akita black bowl in the top layer of the wooden cabinet and place it on the plate", "in"),
        ("pick the akita black bowl next to the cookies box and place it on the plate", "next_to"),
        ("pick the akita black bowl next to the plate and place it on the plate", "next_to"),
        ("pick the akita black bowl next to the ramekin and place it on the plate", "next_to"),
        ("pick the akita black bowl on the cookies box and place it on the plate", "on"),
        ("pick the akita black bowl on the ramekin and place it on the plate", "on"),
        ("pick the akita black bowl on the stove and place it on the plate", "on"),
        ("pick the akita black bowl on the wooden cabinet and place it on the plate", "on"),
    ],
}


# ------------------------------------------------------------------------------------------- grounding
def test_parses_every_object_instruction():
    for text, typ in INSTRUCTIONS["libero_object"]:
        r = GD.parse(text, MP.WORKCELL_TYPES["libero_object"])
        assert r["status"] == "ok", (text, r)
        assert r["target_type"] == typ and r["place_type"] == "basket" and r["predicate"] == "in"
        assert r["relation"] is None


def test_parses_every_spatial_instruction():
    for text, rel in INSTRUCTIONS["libero_spatial"]:
        r = GD.parse(text, MP.WORKCELL_TYPES["libero_spatial"])
        assert r["status"] == "ok", (text, r)
        assert r["target_type"] == "akita_black_bowl" and r["place_type"] == "plate" and r["predicate"] == "on"
        assert r["relation"] == rel, (text, r)
    r = GD.parse(INSTRUCTIONS["libero_spatial"][2][0], MP.WORKCELL_TYPES["libero_spatial"])
    assert r["part"] == "top" and r["landmarks"] == ["wooden_cabinet"]


@pytest.mark.parametrize("phrase,expect", [("ketchup", "ketchup"), ("bbq sauce", "bbq_sauce"),
                                            ("tomato sauce", "tomato_sauce"), ("salad dressing", "salad_dressing"),
                                            ("cookies box", "cookies"), ("ramekin",
                                                                         "glazed_rim_porcelain_ramekin"),
                                            ("stove", "flat_stove"), ("wooden cabinet", "wooden_cabinet"),
                                            ("black bowl", "akita_black_bowl")])
def test_name_rule(phrase, expect):
    assert GD.match_type(phrase, TYPES)[0] == expect


@pytest.mark.parametrize("phrase", ["sauce", "new salad dressing", "top layer", "banana"])
def test_name_rule_refuses(phrase):
    assert GD.match_type(phrase, TYPES)[0] is None


def test_unparsable_instruction_fails():
    for text in ("pick the banana and place it in the basket", "put the milk somewhere",
                 "pick the milk and place it in the fridge"):
        assert GD.parse(text, TYPES)["status"] == "fail"


class FakeLib:
    """A minimal library stand-in: one box primitive per type, and one region site per container."""

    def __init__(self, halves: dict, sites: dict | None = None):
        self.halves = halves
        self.sites = sites or {}
        self.types = {t: {"root_body": "main", "kind": "object", "sites": {}} for t in halves}

    def m1_type(self, typ):
        return {"bodies": {"main": {"primitives": [{"centre": [0, 0, 0], "R": np.eye(3).tolist(),
                                                     "half": list(self.halves[typ])}]}}}

    @staticmethod
    def quat_mat(q):
        return KIN.quat_to_mat(q)

    def site_for(self, percept, part_phrase):
        if percept["type"] not in self.sites:
            return None
        c, half = self.sites[percept["type"]]
        return np.asarray(percept["bodies"]["main"]["pos"], float) + np.asarray(c), np.eye(3), np.asarray(half)

    def site_name_for(self, percept, part_phrase):
        return "region" if percept["type"] in self.sites else None


def _p(pid, typ, pos, n_agree=100):
    return {"id": pid, "type": typ, "bodies": {"main": {"pos": list(pos), "quat_wxyz": [1, 0, 0, 0]}},
            "fit": {"n_agree": n_agree}}


WC = {"arena": [{"type": "box", "centre": [0.0, 0.0, 0.9], "half": [0.5, 0.6, 0.025]}]}


def test_next_to_picks_the_nearer_bowl():
    lib = FakeLib({"akita_black_bowl": (0.05, 0.05, 0.03), "plate": (0.1, 0.1, 0.01)})
    ps = [_p("p0", "akita_black_bowl", (0.5, 0.0, 1.0)), _p("p1", "akita_black_bowl", (0.0, 0.0, 1.0)),
          _p("p2", "plate", (0.05, 0.0, 0.95))]
    g = GD.ground(GD.parse("pick the akita black bowl next to the plate and place it on the plate",
                           list(lib.halves)), ps, lib, WC)
    assert g["status"] == "ok" and g["target"] == "p1" and g["place"] == "p2"


def test_between_picks_the_bowl_on_the_segment():
    lib = FakeLib({"akita_black_bowl": (0.05, 0.05, 0.03), "plate": (0.1, 0.1, 0.01),
                   "glazed_rim_porcelain_ramekin": (0.05, 0.05, 0.03)})
    ps = [_p("p0", "akita_black_bowl", (0.0, 0.5, 1.0)), _p("p1", "akita_black_bowl", (0.0, 0.0, 1.0)),
          _p("p2", "plate", (-0.3, 0.0, 0.95)), _p("p3", "glazed_rim_porcelain_ramekin", (0.3, 0.0, 0.95))]
    g = GD.ground(GD.parse("pick the akita black bowl between the plate and the ramekin and place it on the plate",
                           list(lib.halves)), ps, lib, WC)
    assert g["target"] == "p1"


def test_on_picks_the_stacked_bowl():
    lib = FakeLib({"akita_black_bowl": (0.05, 0.05, 0.03), "plate": (0.1, 0.1, 0.01), "cookies": (0.08, 0.05, 0.06)})
    # the box top is at 0.96; a bowl resting on it has its lowest point there (centre 0.99)
    ps = [_p("p0", "akita_black_bowl", (0.0, 0.0, 0.99)), _p("p1", "akita_black_bowl", (0.4, 0.0, 0.93)),
          _p("p2", "plate", (-0.3, 0.0, 0.91)), _p("p3", "cookies", (0.0, 0.0, 0.9))]
    g = GD.ground(GD.parse("pick the akita black bowl on the cookies box and place it on the plate",
                           list(lib.halves)), ps, lib, WC)
    assert g["target"] == "p0" and "relation_unsatisfied" not in g["flags"]


def test_in_part_uses_the_region_box():
    lib = FakeLib({"akita_black_bowl": (0.05, 0.05, 0.03), "plate": (0.1, 0.1, 0.01),
                   "wooden_cabinet": (0.15, 0.2, 0.3)},
                  sites={"wooden_cabinet": ((0.0, 0.0, 0.1), (0.1, 0.15, 0.06))})
    ps = [_p("p0", "akita_black_bowl", (0.6, 0.0, 1.0)), _p("p1", "akita_black_bowl", (0.0, 0.0, 1.0)),
          _p("p2", "plate", (-0.3, 0.0, 0.91)), _p("p3", "wooden_cabinet", (0.0, 0.0, 0.9))]
    g = GD.ground(GD.parse("pick the akita black bowl in the top layer of the wooden cabinet and place it on the "
                           "plate", list(lib.halves)), ps, lib, WC)
    assert g["target"] == "p1"


def test_table_centre():
    lib = FakeLib({"akita_black_bowl": (0.05, 0.05, 0.03), "plate": (0.1, 0.1, 0.01)})
    ps = [_p("p0", "akita_black_bowl", (0.4, 0.3, 1.0)), _p("p1", "akita_black_bowl", (0.01, 0.0, 1.0)),
          _p("p2", "plate", (-0.3, 0.0, 0.91))]
    g = GD.ground(GD.parse("pick the akita black bowl from table center and place it on the plate",
                           list(lib.halves)), ps, lib, WC)
    assert g["target"] == "p1"


def test_missing_target_type_is_a_grounding_fail():
    lib = FakeLib({"akita_black_bowl": (0.05, 0.05, 0.03), "plate": (0.1, 0.1, 0.01)})
    g = GD.ground(GD.parse("pick the akita black bowl next to the plate and place it on the plate",
                           list(lib.halves)), [_p("p2", "plate", (0.0, 0.0, 0.95))], lib, WC)
    assert g["status"] == "fail"
    g2 = GD.ground(GD.parse("pick the akita black bowl next to the plate and place it on the plate",
                            list(lib.halves)), [_p("p0", "akita_black_bowl", (0.0, 0.0, 1.0))], lib, WC)
    assert g2["status"] == "fail" and "place" in g2["reason"]


def test_missing_landmark_is_flagged_not_fatal():
    lib = FakeLib({"akita_black_bowl": (0.05, 0.05, 0.03), "plate": (0.1, 0.1, 0.01),
                   "glazed_rim_porcelain_ramekin": (0.05, 0.05, 0.03)})
    ps = [_p("p0", "akita_black_bowl", (0.0, 0.0, 1.0), 200), _p("p1", "akita_black_bowl", (0.4, 0.0, 1.0), 100),
          _p("p2", "plate", (-0.3, 0.0, 0.95))]
    g = GD.ground(GD.parse("pick the akita black bowl next to the ramekin and place it on the plate",
                           list(lib.halves)), ps, lib, WC)
    assert g["status"] == "ok" and "landmark_missing" in g["flags"] and g["target"] == "p0"


# ------------------------------------------------------------------------------------------- bridge
def _serve_in(tmp_path, handler):
    """macOS limits sun_path to 104 bytes, so the server binds a short RELATIVE name from its own folder (as the
    run does): the thread changes nothing globally because the test chdirs for both ends."""
    import os
    import time as _t

    os.chdir(tmp_path)
    th = threading.Thread(target=BR.serve, args=("p.sock", handler), daemon=True)
    th.start()
    for _ in range(500):
        if Path(tmp_path / "p.sock").exists():
            break
        _t.sleep(0.01)
    return th


def _connect(path: str, tries: int = 200):
    """bind() creates the socket file before listen(), so a connect may be refused for a moment."""
    import time as _t

    for _ in range(tries):
        try:
            return BR.Client(path, timeout_s=10)
        except (ConnectionRefusedError, FileNotFoundError):
            _t.sleep(0.01)
    raise AssertionError("the server never accepted a connection")


def test_bridge_round_trips_arrays_bit_exactly(tmp_path, monkeypatch):
    sock_path = "p.sock"
    rgb = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    depth = (np.arange(12, dtype=np.uint16) * 1000).reshape(3, 4)
    got = {}

    def handler(msg):
        got["msg"] = msg
        return {"ok": True, "echo_rgb": msg["cameras"][0]["rgb"], "n": int(msg["cameras"][0]["depth_u16"].sum())}

    th = _serve_in(tmp_path, handler)
    c = _connect(sock_path)
    rep = c.call({"op": "look", "cameras": [{"rgb": rgb, "depth_u16": depth, "K": np.eye(3)}], "x": 1.5})
    assert np.array_equal(rep["echo_rgb"], rgb)
    assert rep["n"] == int(depth.sum())
    assert np.array_equal(got["msg"]["cameras"][0]["rgb"], rgb)
    assert got["msg"]["cameras"][0]["rgb"].dtype == np.uint8
    assert got["msg"]["x"] == 1.5
    BR.send(c.sock, {"op": "shutdown"})
    c.sock.close()
    th.join(timeout=10)


def test_bridge_refuses_object_dtypes():
    with pytest.raises(TypeError):
        BR.encode({"a": np.array(["x", "y"])})


def test_bridge_server_returns_an_error_instead_of_dying(tmp_path):
    sock_path = "p.sock"

    def handler(msg):
        raise ValueError("boom")

    th = _serve_in(tmp_path, handler)
    c = _connect(sock_path)
    with pytest.raises(BR.RemoteError):
        c.call({"op": "look"})
    BR.send(c.sock, {"op": "shutdown"})
    c.sock.close()
    th.join(timeout=10)


def test_bridge_only_uses_af_unix_and_never_pickles():
    import ast

    src = Path(BR.__file__).read_text()
    assert "AF_INET" not in src and "AF_UNIX" in src
    tree = ast.parse(src)
    mods = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    mods += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
    assert not any("pickle" in m for m in mods)
    assert not any(isinstance(n, ast.Attribute) and "pickle" in n.attr for n in ast.walk(tree))
    assert socket.AF_UNIX is not None


# ------------------------------------------------------------------------------------------- kinematics
def test_quat_mat_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        R = KIN.quat_to_mat(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.allclose(KIN.quat_to_mat(KIN.mat_to_quat(R)), R, atol=1e-9)


def test_hinge_and_slide_joint_transforms():
    jt = {"type": "hinge", "axis": [0, 0, 1], "pos": [0.1, 0, 0], "qpos0": 0.0}
    R, t = KIN.apply_joint(np.eye(3), np.zeros(3), jt, math.pi / 2)
    assert np.allclose(R @ np.array([0.1, 0, 0]) + t, [0.1, 0, 0], atol=1e-12)  # the anchor does not move
    js = {"type": "slide", "axis": [0, 1, 0], "pos": [0, 0, 0], "qpos0": 0.0}
    R2, t2 = KIN.apply_joint(np.eye(3), np.zeros(3), js, -0.16)
    assert np.allclose(t2, [0, -0.16, 0]) and np.allclose(R2, np.eye(3))


# ------------------------------------------------------------------------------------------- camera maths
def test_intrinsics_match_robosuite_rule():
    K = CAM.intrinsics(45.0, 1024, 1024)
    assert math.isclose(K[0, 0], 512.0 / math.tan(math.radians(45.0) / 2.0), rel_tol=1e-12)
    assert K[0, 2] == 512.0 and K[1, 2] == 512.0 and K[1, 1] == K[0, 0]
    K2 = CAM.intrinsics(75.0, 512, 512)
    assert math.isclose(K2[0, 0], 256.0 / math.tan(math.radians(75.0) / 2.0), rel_tol=1e-12)


def test_depth_formula_and_encoding():
    extent, znear, zfar = 10.60981561453384, 0.0010000000474974513, 50.0
    near, far = znear * extent, zfar * extent
    for z in (0.1, 0.5, 0.9, 0.99):
        d = CAM.real_depth(np.array([[z]]), extent, znear, zfar)[0, 0]
        assert math.isclose(d, near / (1 - z * (1 - near / far)), rel_tol=1e-12)
    zb = np.array([[0.5, 1.0 - 1e-9]])
    dm = CAM.real_depth(zb, extent, znear, zfar)
    enc = CAM.encode_depth(dm, zb)
    assert enc.dtype == np.uint16
    assert enc[0, 1] == 0  # the far plane has no reading
    assert abs(enc[0, 0] * 1e-4 - dm[0, 0]) <= 5e-5  # 0.1 mm units


def test_cv_from_gl_is_the_robosuite_extrinsic_rule():
    assert np.allclose(CAM.CV_FROM_GL, np.diag([1.0, -1.0, -1.0, 1.0]))
    assert CAM.FRAME_KEYS == ("camera", "rgb", "depth_u16", "depth_scale_mm", "K", "T_world_from_camera", "width",
                              "height", "step")


# ------------------------------------------------------------------------------------------- the library
@pytest.mark.skipif(not (MP.EXPORT_DIR / "manifest.json").exists(), reason="the M2 library is not exported here")
def test_library_holds_no_instance_name_or_world_site():
    for typ in sorted(set(MP.WORKCELL_TYPES["libero_object"]) | set(MP.WORKCELL_TYPES["libero_spatial"])):
        doc = json.loads((MP.EXPORT_DIR / "types" / f"{typ}.json").read_text())
        text = json.dumps(doc)
        assert "_1_" not in text and not any(k.endswith("_1") for k in doc["bodies"]), typ
        assert "_2_" not in text
        for s in doc["sites"].values():
            assert s["body"] in doc["bodies"], typ
        for key in (":goal", ":init", "obj_of_interest", "robot_start"):
            assert key not in text, (typ, key)


@pytest.mark.skipif(not (MP.EXPORT_DIR / "manifest.json").exists(), reason="the M2 library is not exported here")
def test_workcell_is_suite_level():
    for suite in MP.SUITES:
        wc = json.loads((MP.EXPORT_DIR / "workcells" / f"{suite}.json").read_text())
        assert wc["identical_across_tasks"] is True
        assert sorted(wc["types"]) == sorted(MP.WORKCELL_TYPES[suite])
        assert set(wc["cameras"]) == set(MP.CAMERAS)
        assert "instances" not in wc and "goal" not in wc
