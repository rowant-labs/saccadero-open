"""Tests of m2_segment (amendment A14, m2f:perceive-r1): synthetic scenes only, no simulator, no image saved.

M2 (perceived poses, grounded instruction): perception in the loop. A synthetic support plane at the LIBERO-Spatial
table height, seen by an oblique pinhole camera (z-buffered dense surface samples), with a disc (a plate-like part)
and a box (a fixture-like part) standing on it:

* the TLS refit recovers the plane (normal and height);
* the support (relief) check passes the disc at its true pose and rejects the same model on bare table;
* the support search finds the disc and the box, and fits a welded box's height offset;
* the module reads no file and imports nothing beyond numpy, scipy and the standard library (the truth guard's
  perception rules also scan it).
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from rowant_lab.libero_env import m2_segment as SEG

HERE = Path(__file__).resolve().parents[1]
TABLE_Z = 0.90
F = 420.0
W = H = 320


def _cam():
    """A camera at (0.75, 0, 1.55) looking at (0, 0, 0.9) (agentview-like): T_world_from_camera, K."""
    c = np.array([0.75, 0.0, 1.55])
    fwd = np.array([0.0, 0.0, TABLE_Z]) - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd], 1)  # camera axes (x right, y down, z forward) in the world
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, c
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    return T, K


def _disc(r=0.1, h=0.015, step=0.002):
    """Surface samples and normals of a closed disc (body origin at the bottom centre)."""
    pts, nrm = [], []
    g = np.arange(-r, r + 1e-9, step)
    X, Y = np.meshgrid(g, g)
    m = X ** 2 + Y ** 2 <= r * r
    for z, nz in ((h, 1.0), (0.0, -1.0)):
        pts.append(np.stack([X[m], Y[m], np.full(m.sum(), z)], 1))
        nrm.append(np.tile([0, 0, nz], (m.sum(), 1)))
    th = np.arange(0, 2 * math.pi, step / r)
    for z in np.arange(0, h + 1e-9, step):
        pts.append(np.stack([r * np.cos(th), r * np.sin(th), np.full(len(th), z)], 1))
        nrm.append(np.stack([np.cos(th), np.sin(th), np.zeros(len(th))], 1))
    return np.concatenate(pts), np.concatenate(nrm).astype(float)


def _box(sx=0.12, sy=0.07, sz=0.05, step=0.002):
    pts, nrm = [], []
    hx, hy = sx / 2, sy / 2
    gx, gy, gz = np.arange(-hx, hx + 1e-9, step), np.arange(-hy, hy + 1e-9, step), np.arange(0, sz + 1e-9, step)
    for z, nz in ((sz, 1.0), (0.0, -1.0)):
        X, Y = np.meshgrid(gx, gy)
        pts.append(np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1))
        nrm.append(np.tile([0, 0, nz], (X.size, 1)))
    for x, nx in ((hx, 1.0), (-hx, -1.0)):
        Y, Z = np.meshgrid(gy, gz)
        pts.append(np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], 1))
        nrm.append(np.tile([nx, 0, 0], (Y.size, 1)))
    for y, ny in ((hy, 1.0), (-hy, -1.0)):
        X, Z = np.meshgrid(gx, gz)
        pts.append(np.stack([X.ravel(), np.full(X.size, y), Z.ravel()], 1))
        nrm.append(np.tile([0, ny, 0], (X.size, 1)))
    return np.concatenate(pts), np.concatenate(nrm).astype(float)


def _rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def _rays(T, K):
    """World origin and unit directions of every pixel centre (u + 0.5, v + 0.5)."""
    v, u = np.mgrid[0:H, 0:W].astype(float)
    dc = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    return T[:3, 3], dc @ T[:3, :3].T, dc


def _hit_box(o, d, lo, hi):
    """Ray-box entry parameter (inf when missed), slab method; o (3,), d (N, 3) in the box frame."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - o) / d
        t2 = (hi - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    return np.where((tmax >= tmin) & (tmin > 0), tmin, np.inf)


def _hit_disc(o, d, r, h):
    """Ray entry into a closed cylinder (axis z, 0 <= z <= h, radius r) in its frame."""
    out = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (h - o[2]) / d[:, 2]  # the top face
        p = o + t[:, None] * d
        ok = (t > 0) & (p[:, 0] ** 2 + p[:, 1] ** 2 <= r * r)
        out[ok] = t[ok]
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        b = 2 * (o[0] * d[:, 0] + o[1] * d[:, 1])
        c = o[0] ** 2 + o[1] ** 2 - r * r
        disc = b * b - 4 * a * c
        ts = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
        z = o[2] + ts * d[:, 2]
        ok = (disc >= 0) & (ts > 0) & (z >= 0) & (z <= h)
        out[ok] = np.minimum(out[ok], ts[ok])
    return out


def _render(T, K, parts):
    """Exact depth (camera z at each pixel centre) of the plane plus parts [(kind, dims, R_world, t_world)]."""
    o, d, dc = _rays(T, K)
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = (TABLE_Z - o[2]) / d[:, 2]
    tb = np.where(tp > 0, tp, np.inf)
    for kind, dims, R, t in parts:
        ob, db = R.T @ (o - t), d @ R
        hit = _hit_disc(ob, db, *dims) if kind == "disc" else _hit_box(ob, db, *dims)
        tb = np.minimum(tb, hit)
    z = (tb * dc[:, 2]).reshape(H, W)  # dc has unit z: camera depth = ray parameter
    z[~np.isfinite(z)] = np.nan
    return z


def _plane_guess(T):
    """A slightly-off plane in the camera frame, as the fast layer's fit would give."""
    Rwc, twc = T[:3, :3], T[:3, 3]
    n_w = np.array([0.003, -0.002, 1.0])
    n_w /= np.linalg.norm(n_w)
    p_w = np.array([0.0, 0.0, TABLE_Z + 0.001])
    n_c = Rwc.T @ n_w
    p_c = Rwc.T @ (p_w - twc)
    return {"normal": n_c.tolist(), "d": float(-n_c @ p_c)}


@pytest.fixture(scope="module")
def scene():
    T, K = _cam()
    disc, disc_n = _disc()
    box, box_n = _box()
    disc_t = np.array([-0.10, 0.12, TABLE_Z])
    box_R, box_t = _rz(25.0), np.array([0.05, -0.15, TABLE_Z + 0.006])  # a welded box 6 mm above the support
    z = _render(T, K, [("disc", (0.1, 0.015), np.eye(3), disc_t),
                       ("box", (np.array([-0.06, -0.035, 0.0]), np.array([0.06, 0.035, 0.05])), box_R, box_t)])
    return {"T": T, "K": K, "z": z, "disc": (disc, disc_n, disc_t), "box": (box, box_n, box_R, box_t)}


def _to_cam(T, R_w, t_w):
    Rwc, twc = T[:3, :3], T[:3, 3]
    return Rwc.T @ R_w, Rwc.T @ (t_w - twc)


def test_refit_support_recovers_plane(scene):
    T, K = scene["T"], scene["K"]
    sup = SEG.refit_support(scene["z"], K, _plane_guess(T))
    n_w = T[:3, :3] @ sup["normal"]
    assert np.degrees(np.arccos(min(1.0, n_w[2]))) < 0.05
    foot = T[:3, :3] @ (-sup["d"] * sup["normal"]) + T[:3, 3]
    assert abs(foot[2] - TABLE_Z) < 2e-4
    assert sup["rms_m"] < 1e-3 and sup["d"] > 0


def test_support_check_rejects_bare_table(scene):
    T, K, z = scene["T"], scene["K"], scene["z"]
    sup = SEG.refit_support(z, K, _plane_guess(T))
    hpx = SEG.heights(z, K, sup)
    disc, disc_n, disc_t = scene["disc"]
    R, t = _to_cam(T, np.eye(3), disc_t)
    good = SEG.support_check(disc, disc_n, R, t, hpx, K, sup, 0.15)
    assert good["pass"] and good["bare_share"] < 0.05
    R, t = _to_cam(T, np.eye(3), np.array([0.25, 0.25, TABLE_Z]))  # bare table
    bad = SEG.support_check(disc, disc_n, R, t, hpx, K, sup, 0.15)
    assert not bad["pass"] and bad["bare_share"] > 0.9


def test_support_search_finds_disc_and_welded_box(scene):
    T, K, z = scene["T"], scene["K"], scene["z"]
    disc, disc_n, disc_t = scene["disc"]
    box, box_n, box_R, box_t = scene["box"]
    nodes = {"1": (disc, disc_n), "2": (box, box_n)}
    fam = {"1": [np.eye(3)] + [_rz(a) for a in range(5, 360, 5)], "2": [np.eye(3), _rz(180.0)]}
    kind = {"1": "object", "2": "fixture"}
    res = SEG.support_search([("1", "disc"), ("2", "box")], lambda g: nodes[g][0], lambda g: nodes[g][1],
                             lambda g: fam[g], lambda g: 0.2 if g == "1" else 0.14, lambda g: kind[g], z, K, T,
                             _plane_guess(T), (-0.5, 0.5, -0.5, 0.5), 0.15, 0.01, 0.015)
    Rwc, twc = T[:3, :3], T[:3, 3]
    best = {}
    for c in res["candidates"]:
        tw = Rwc @ c["t"] + twc
        best.setdefault(c["gid"], []).append((c["refine"]["score"], tw, Rwc @ c["R"], c))
    s, tw, Rw, _c = max(best["1"], key=lambda x: x[0])
    assert np.linalg.norm(tw - disc_t) < 0.003
    s, tw, Rw, c = max(best["2"], key=lambda x: x[0])
    assert np.linalg.norm(tw[:2] - box_t[:2]) < 0.004
    assert abs(tw[2] - box_t[2]) < 0.0015  # the fitted height offset of a welded fixture
    yaw = math.degrees(math.atan2(Rw[1, 0], Rw[0, 0])) % 180.0
    assert abs(yaw - 25.0) < 2.5
    assert res["diag"]["total_s"] < 10.0


def test_yaw_period():
    fam = [np.eye(3)] + [_rz(a) for a in range(5, 360, 5)]
    assert SEG.type_yaws(fam) == [0.0]
    assert len(SEG.type_yaws([np.eye(3), _rz(180.0)])) == 18
    flip_x = np.diag([1.0, -1.0, -1.0])
    assert len(SEG.type_yaws([np.eye(3), flip_x])) == 36


def test_module_reads_no_file_and_imports_little():
    tree = ast.parse((HERE / "m2_segment.py").read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("open", "eval", "exec", "__import__", "getattr"), node.func.id
        elif isinstance(node, ast.Attribute):
            assert node.attr not in ("load", "loadtxt", "fromfile", "genfromtxt", "save", "savez", "imwrite",
                                     "read_text", "read_bytes", "write_text", "write_bytes", "__closure__"), node.attr
    assert mods <= {"__future__", "math", "time", "numpy", "scipy"}, mods


# ---- 2026-11-16: the half-turn twin of a jointed fixture (``twin_prior``, ``twin_fit``)
OFF = np.array([0.0, 0.02, 0.0])  # the body origin 2 cm from the footprint's middle, as a cabinet's


def _plane_pose(pf, R_c, t_c, nodes):
    """(a, b, yaw, dz) of a camera-frame pose: the inverse of ``SEG.pose_at``."""
    q = np.asarray(t_c, float) - pf["p0"]
    Rz = pf["G"].T @ np.asarray(R_c, float)
    return (float(q @ pf["ex"]), float(q @ pf["ey"]), math.atan2(float(Rz[1, 0]), float(Rz[0, 0])),
            float(q @ pf["n"]) + float(nodes[:, 2].min()))


@pytest.fixture(scope="module")
def cupboard():
    """A box-like carcass (the body origin off its middle) standing 6 mm into its fitted height window."""
    T, K = _cam()
    box, box_n = _box()
    nodes = box + OFF
    R, t = _rz(25.0), np.array([0.05, -0.15, TABLE_Z + 0.006])
    z = _render(T, K, [("box", (np.array([-0.06, -0.035, 0.0]) + OFF, np.array([0.06, 0.035, 0.05]) + OFF), R, t)])
    sup = SEG.refit_support(z, K, _plane_guess(T))
    return {"T": T, "K": K, "z": z, "nodes": nodes, "nrm": box_n, "R": R, "t": t, "pf": SEG.plane_frame(sup, T)}


def _world(T, R_c, t_c):
    return T[:3, :3] @ np.asarray(R_c), T[:3, :3] @ np.asarray(t_c) + T[:3, 3]


def _yaw_off(Rw, deg):
    return abs((math.degrees(math.atan2(Rw[1, 0], Rw[0, 0])) - deg + 180.0) % 360.0 - 180.0)


def test_twin_fit_turns_a_mirror_fit_about_its_footprint_and_finds_the_true_place(cupboard):
    c = cupboard
    T, pf, nodes = c["T"], c["pf"], c["nodes"]
    # the mirror fit: half a turn about the footprint's middle, and 3 cm off along the cupboard's depth
    R_m = _rz(205.0)
    t_m = c["R"] @ OFF + c["t"] - R_m @ OFF + c["R"] @ np.array([0.0, 0.03, 0.0])
    R_c, t_c = _to_cam(T, R_m, t_m)
    a, b, yaw, dz = _plane_pose(pf, R_c, t_c, nodes)
    R0, t0 = SEG.pose_at(pf, a, b, yaw, -float(nodes[:, 2].min()), dz)
    assert np.allclose(R0, R_c, atol=1e-6) and np.allclose(t0, t_c, atol=1e-6)  # the helper inverts pose_at
    rf = {"a": a, "b": b, "yaw": yaw, "dz": dz, "score": 0.0, "R": R_c, "t": t_c}
    tw, rec = SEG.twin_fit(rf, nodes, c["nrm"], c["z"], c["K"], pf, None, 0.01, 0.015, 0.15)
    Rw, tw_w = _world(T, tw["R"], tw["t"])
    assert _yaw_off(Rw, 25.0) < 2.0
    assert np.linalg.norm(tw_w - c["t"]) < 0.004
    assert rec["score"] > 0 and abs(rec["yaw_deg"] - math.degrees(tw["yaw"]) % 360.0) < 0.01
    assert "closed_polish" not in rec  # no closed parts were given


def _search(c, p, want_deg, calls):
    nodes, nrm, T = c["nodes"], c["nrm"], c["T"]

    def face_of(g, R_c, t_c):
        calls.append(g)
        Rw, _t = _world(T, R_c, t_c)
        off = _yaw_off(Rw, want_deg)
        return 1.0 if off < 45.0 else (-1.0 if off > 135.0 else None)

    return SEG.support_search(
        [("2", "cupboard")], lambda g: nodes, lambda g: nrm, lambda g: [np.eye(3), _rz(180.0)], lambda g: 0.14,
        lambda g: "fixture", c["z"], c["K"], T, _plane_guess(T), (-0.3, 0.4, -0.5, 0.2), 0.15, 0.01, 0.015, p,
        variants_of=lambda g: [("closed", nodes)], variants_n_of=lambda g: [("closed", nodes, nrm)],
        assembly_of=lambda g, R_, t_: (nodes, nrm, {}), slide_axes_of=lambda g: [[0.0, 1.0, 0.0]],
        closed_of=lambda g, R_, t_: (nodes, nrm, {}, [], []), face_of=face_of, accept_of=lambda *a_: True)


def test_support_search_takes_the_twin_on_a_back_to_front_verdict_only(cupboard):
    c = cupboard
    T = c["T"]
    calls: list = []
    plain = _search(c, SEG.PARAMS, 25.0, calls)["candidates"]
    assert calls == [] and all("face" not in x["refine"] for x in plain)  # off: the prior is never asked
    taken = []
    for want in (25.0, 205.0):  # a box fits both ways round: whichever the search chose, the prior's way comes out
        got = _search(c, {**SEG.PARAMS, "twin_prior": {}}, want, calls)["candidates"]
        best = max(got, key=lambda x: x["refine"]["score"])
        Rw, tw = _world(T, best["R"], best["t"])
        assert _yaw_off(Rw, want) < 3.0
        assert np.linalg.norm((Rw @ OFF + tw) - (c["R"] @ OFF + c["t"])) < 0.005  # the same footprint either way
        taken.append(bool(best["refine"]["face"].get("taken")))
        assert best["refine"]["face"]["prior"] in (1.0, -1.0)
    assert sorted(taken) == [False, True] and calls
    # a twin the server's floors refuse is not taken: the search's own choice stands
    first = max(plain, key=lambda x: x["refine"]["score"])
    Rw0, _t = _world(T, first["R"], first["t"])
    other = 205.0 if _yaw_off(Rw0, 25.0) < 45.0 else 25.0
    nodes, nrm = c["nodes"], c["nrm"]

    def face_of(g, R_c, t_c):
        off = _yaw_off(_world(T, R_c, t_c)[0], other)
        return 1.0 if off < 45.0 else -1.0

    got = SEG.support_search(
        [("2", "cupboard")], lambda g: nodes, lambda g: nrm, lambda g: [np.eye(3), _rz(180.0)], lambda g: 0.14,
        lambda g: "fixture", c["z"], c["K"], T, _plane_guess(T), (-0.3, 0.4, -0.5, 0.2), 0.15, 0.01, 0.015,
        {**SEG.PARAMS, "twin_prior": {}}, variants_of=lambda g: [("closed", nodes)],
        variants_n_of=lambda g: [("closed", nodes, nrm)], assembly_of=lambda g, R_, t_: (nodes, nrm, {}),
        slide_axes_of=lambda g: [[0.0, 1.0, 0.0]], closed_of=lambda g, R_, t_: (nodes, nrm, {}, [], []),
        face_of=face_of, accept_of=lambda *a_: False)["candidates"]
    best = max(got, key=lambda x: x["refine"]["score"])
    assert best["refine"]["face"]["taken"] is False and best["refine"]["face"]["prior"] == -1.0
    assert best["refine"]["face"]["twin_passed"] is False
    assert _yaw_off(_world(T, best["R"], best["t"])[0], other) > 135.0
    # ... unless the prior is told to take the twin always: a back-to-front pose then never leaves the search
    got = SEG.support_search(
        [("2", "cupboard")], lambda g: nodes, lambda g: nrm, lambda g: [np.eye(3), _rz(180.0)], lambda g: 0.14,
        lambda g: "fixture", c["z"], c["K"], T, _plane_guess(T), (-0.3, 0.4, -0.5, 0.2), 0.15, 0.01, 0.015,
        {**SEG.PARAMS, "twin_prior": {"take": "always"}}, variants_of=lambda g: [("closed", nodes)],
        variants_n_of=lambda g: [("closed", nodes, nrm)], assembly_of=lambda g, R_, t_: (nodes, nrm, {}),
        slide_axes_of=lambda g: [[0.0, 1.0, 0.0]], closed_of=lambda g, R_, t_: (nodes, nrm, {}, [], []),
        face_of=face_of, accept_of=lambda *a_: False)["candidates"]
    best = max(got, key=lambda x: x["refine"]["score"])
    assert best["refine"]["face"]["taken"] is True and best["refine"]["face"]["twin_passed"] is False
    assert _yaw_off(_world(T, best["R"], best["t"])[0], other) < 3.0


def _search_mem(c, seed_w, yaw_deg, accept, p=None):
    """The cupboard searched with one remembered place (world xyz of its origin, its turn)."""
    nodes, nrm, T, pf = c["nodes"], c["nrm"], c["T"], c["pf"]

    def seeds_of(g, pf_):
        R_c, t_c = _to_cam(T, _rz(yaw_deg), np.asarray(seed_w, float))
        a, b, yaw, _dz = _plane_pose(pf_, R_c, t_c, nodes)
        return [(yaw, a, b)]

    return SEG.support_search(
        [("2", "cupboard")], lambda g: nodes, lambda g: nrm, lambda g: [np.eye(3), _rz(180.0)], lambda g: 0.14,
        lambda g: "fixture", c["z"], c["K"], T, _plane_guess(T), (-0.3, 0.4, -0.5, 0.2), 0.15, 0.01, 0.015,
        p or {**SEG.PARAMS, "memory": {}}, variants_of=lambda g: [("closed", nodes)],
        variants_n_of=lambda g: [("closed", nodes, nrm)], assembly_of=lambda g, R_, t_: (nodes, nrm, {}),
        slide_axes_of=lambda g: [[0.0, 1.0, 0.0]], closed_of=lambda g, R_, t_: (nodes, nrm, {}, [], []),
        accept_of=accept, seeds_of=seeds_of)


def test_a_remembered_place_is_taken_when_the_fixture_is_seen_to_stand_there(cupboard):
    c = cupboard
    T = c["T"]
    near = lambda g, R_c, t_c, hm, pf: bool(np.linalg.norm(_world(T, R_c, t_c)[1] - c["t"]) < 0.006)  # noqa: E731
    # remembered 5 mm and 2 degrees off: refined onto the cupboard, accepted, and the search is skipped
    res = _search_mem(c, c["t"] + [0.004, -0.003, 0.0], 27.0, near)
    (only,) = res["candidates"]
    assert only["coarse"] == {"memory": True} and only["refine"]["memory"]["taken"] is True
    assert only["refine"]["memory"]["how"] == "refined"
    Rw, tw = _world(T, only["R"], only["t"])
    assert _yaw_off(Rw, 25.0) < 2.0 and np.linalg.norm(tw - c["t"]) < 0.004
    assert only["refine"]["memory"]["moved_mm"] < 8.0 and res["diag"]["types"]["cupboard"]["memory"]["taken"]
    # remembered 2.2 cm off (the benchmark sets a fixture down anew every time): found by the search around it
    res = _search_mem(c, c["t"] + [0.018, -0.012, 0.0], 25.0, near)
    (only,) = res["candidates"]
    assert only["refine"]["memory"]["taken"] is True and 15.0 < only["refine"]["memory"]["moved_mm"] < 28.0
    assert np.linalg.norm(_world(T, only["R"], only["t"])[1] - c["t"]) < 0.004
    # the remembered turn decides which way round a box-like carcass is read
    turned = c["R"] @ OFF + c["t"] - _rz(205.0) @ OFF
    (only,) = _search_mem(c, turned, 205.0, lambda *a_: True)["candidates"]
    assert _yaw_off(_world(T, only["R"], only["t"])[0], 205.0) < 2.0 and only["refine"]["memory"]["taken"]


def test_the_remembered_pose_itself_is_put_to_the_floors_when_its_refine_is_refused(cupboard):
    c = cupboard
    T, pf, nodes = c["T"], c["pf"], c["nodes"]
    seed_w = c["t"] + [0.003, 0.002, 0.0]  # remembered 3.6 mm off: the refine moves it onto the cupboard
    R_s, t_s = _to_cam(T, _rz(25.0), seed_w)

    def only_as_given(g, R_c, t_c, hm, pf_):  # floors that pass the remembered pose alone (whatever its height)
        tw = _world(T, R_c, t_c)[1]
        return bool(np.linalg.norm(tw[:2] - seed_w[:2]) < 2e-4)

    (only,) = _search_mem(c, seed_w, 25.0, only_as_given)["candidates"]
    rec = only["refine"]["memory"]
    assert rec["taken"] is True and rec["how"] == "as remembered" and "refined_score" in rec
    Rw, tw = _world(T, only["R"], only["t"])
    assert np.linalg.norm(tw[:2] - seed_w[:2]) < 2e-4 and _yaw_off(Rw, 25.0) < 1e-6
    assert abs(tw[2] - c["t"][2]) < 0.002  # its height offset was fitted
    r0 = SEG.pose_as_given(nodes, c["nrm"], c["z"], c["K"], pf, *_plane_pose(pf, R_s, t_s, nodes)[:2],
                           _plane_pose(pf, R_s, t_s, nodes)[2], 0.01, 0.015, 0.15)
    assert np.allclose(r0["t"], only["t"]) and r0["score"] > 0


def test_a_fixture_that_is_not_where_it_was_remembered_is_searched_for_as_before(cupboard):
    c = cupboard
    T = c["T"]
    near = lambda g, R_c, t_c, hm, pf: bool(np.linalg.norm(_world(T, R_c, t_c)[1] - c["t"]) < 0.006)  # noqa: E731
    plain = _search(c, SEG.PARAMS, 25.0, [])["candidates"]
    # remembered 15 cm away (the fixture was moved): the floors refuse the pose there
    res = _search_mem(c, c["t"] + [0.0, 0.15, 0.0], 25.0, near)
    assert all("memory" not in x["coarse"] for x in res["candidates"]) and len(res["candidates"]) == len(plain)
    assert res["diag"]["types"]["cupboard"]["memory"]["taken"] is False
    for x, y in zip(res["candidates"], plain):
        assert np.allclose(x["R"], y["R"]) and np.allclose(x["t"], y["t"])  # the search's own answer, unchanged
    # remembered 3 cm off: the refine may reach the cupboard, but that is not where it was remembered, and the
    # remembered pose itself does not pass the floors
    res = _search_mem(c, c["t"] + c["R"] @ np.array([0.03, 0.0, 0.0]), 25.0, near,
                      p={**SEG.PARAMS, "memory": {}, "memory_xy_m": 0.01})
    rec = res["diag"]["types"]["cupboard"]["memory"]
    assert rec["taken"] is False and all("memory" not in x["coarse"] for x in res["candidates"])
    # without the option the remembered places are never asked for
    asked = []
    res = SEG.support_search(
        [("2", "cupboard")], lambda g: c["nodes"], lambda g: c["nrm"], lambda g: [np.eye(3), _rz(180.0)],
        lambda g: 0.14, lambda g: "fixture", c["z"], c["K"], T, _plane_guess(T), (-0.3, 0.4, -0.5, 0.2), 0.15, 0.01,
        0.015, SEG.PARAMS, seeds_of=lambda g, pf_: asked.append(g) or [])
    assert asked == [] and "memory" not in res["diag"]["types"]["cupboard"]
