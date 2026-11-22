"""LF-b learning-phase perception (Monty venv; registration ``lfc_b``): the new object's region and its look masks.

Launch from the repository root (``PYTHONPATH=$PWD``, ``python -B``, never ``-I``)::

    python -B -m rowant_lab.libero_env.lfb_learn leftover --task T --L L --cond C [--suite S --held-out TYPE]
    python -B -m rowant_lab.libero_env.lfb_learn masks    --task T --L L --cond C [--suite S --held-out TYPE]

TRUTH-FREE: reads only the capture side's frames (``frames/``), the ``minus`` variant library and M2's frozen run
params; never ``truth/``. Perception is M2's own ``Perceiver`` in process, with the minus library, so the held-out
type T has no model and can only show up as unexplained points.

* ``leftover``, on the agentview frame:
  1. ``Perceiver.full`` gives M2's percepts and support plane, and its self-filter gives the robot-free depth.
  2. The world points more than ``BAND_M`` above the plane and within ``TOP_M`` of it are kept.
  3. Points within ``EXPLAIN_M`` of any percept's model nodes at its pose are removed.
  4. The rest is voxelled at ``VOX_M``, and its ``LINK_M`` connected components are taken. The largest with at least
     ``MIN_VOX`` voxels is the new object, giving its centre, bbox and voxels. The runner-up's size is kept as an
     ambiguity record.
* ``masks``, on each wrist look. The same steps 2-3 run with the agentview percepts, since the scene is static.
  Then the points within ``R_PAD_M`` (horizontally) of the leftover's footprint and below its top plus ``Z_PAD_M``
  are kept, and so is the connected component holding most of them. The mask is that component's pixels. It is
  shared by every builder (amendment LFB-MASK: the hidden back of the object lies outside the agentview bbox).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from rowant_lab.libero_env import lfb_paths as LP
from rowant_lab.settings import DATA

PARAMS_FILE = DATA / "runs" / "libero-m3" / "mminus" / "params-plus.json"
BAND_M, TOP_M, EXPLAIN_M = 0.008, 0.30, 0.010
VOX_M, LINK_M, MIN_VOX = 0.003, 0.0075, 60
R_PAD_M, Z_PAD_M = 0.04, 0.01
MASK_BAND_M = 0.004  # masks keep the object's sides down to 4 mm above the support (the leftover uses BAND_M)
MASKS = "masks-m2"  # the mask file's name (a new name for each mask rule, so no file is ever replaced)
FRAME_KEYS = ("camera", "rgb", "depth_u16", "depth_scale_mm", "K", "T_world_from_camera")


def load_frame(path) -> tuple[dict, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        fr = {k: np.asarray(z[k]) for k in FRAME_KEYS}
        jp, gq = np.asarray(z["joint_pos"], float), np.asarray(z["gripper_qpos"], float)
    fr["camera"] = str(fr["camera"])
    fr["depth_scale_mm"] = float(fr["depth_scale_mm"])
    w = int(fr["rgb"].shape[1])
    fr.update(width=w, height=w, step=0)
    return fr, jp, gq


def perceiver(minus_root: Path):
    from rowant_lab.libero_env import m2_perceive as PER

    sp = dict(json.loads(PARAMS_FILE.read_text())["server"])
    sp["lfb_library"] = str(minus_root)
    return PER.Perceiver(sp)


def world_points(per, wc, fr, jp, gq) -> tuple[np.ndarray, np.ndarray]:
    """(robot-free world points (H, W, 3) with NaN where no reading, valid mask)."""
    d2, md = per._mask_depth(wc, fr, jp, gq)
    z = d2.astype(np.float64) * float(fr["depth_scale_mm"]) * 1e-3
    H, W = z.shape
    K, T = md["K"], md["T"]
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    Pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
    Pw = Pc @ T[:3, :3].T + T[:3, 3]
    ok = d2 > 0
    Pw[~ok] = np.nan
    return Pw, ok


def above_plane(Pw, ok, plane, band: float = BAND_M) -> np.ndarray:
    n = np.asarray(plane["normal_w"], float)
    n = n / np.linalg.norm(n) * (1.0 if n[2] >= 0 else -1.0)
    h = (Pw - np.array([0.0, 0.0, float(plane["height_w"])])) @ n
    return ok & (h > band) & (h < TOP_M)


def explained(wc, percepts, P: np.ndarray) -> np.ndarray:
    """Points (N, 3) within EXPLAIN_M of any percept's model nodes at its perceived root pose."""
    from scipy.spatial import cKDTree

    from rowant_lab.libero_env import m2_kin as KIN

    pts = []
    for p in percepts:
        tdoc = wc.types[p["type"]]
        for r, b in p["bodies"].items():  # every part whose root body is perceived (a fixture's drawers too)
            pa = tdoc["part_of"].get(r)
            g = wc.gid.get((p["type"], pa))
            if g is None or tdoc["parts"][pa]["bodies"][0] != r or str(g) not in wc.nodes:
                continue
            Pp = tdoc["pose_in_part"][r]  # the body in its part's frame: part pose = body pose . inverse
            Rb, tb = KIN.quat_to_mat(b["quat_wxyz"]), np.asarray(b["pos"], float)
            Rp = Rb @ np.asarray(Pp["R"], float).T
            tp = tb - Rp @ np.asarray(Pp["t"], float)
            pts.append(wc.nodes[str(g)][0] @ Rp.T + tp)
    if not pts or not len(P):
        return np.zeros(len(P), bool)
    d, _ = cKDTree(np.concatenate(pts)).query(P, k=1, distance_upper_bound=EXPLAIN_M)
    return np.isfinite(d)


def components(P: np.ndarray) -> tuple[np.ndarray, list]:
    """Voxel centres of P and their connected components (largest first), linked within LINK_M."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    if not len(P):
        return np.zeros((0, 3)), []
    keys, inv = np.unique(np.floor(P / VOX_M).astype(np.int64), axis=0, return_inverse=True)
    V = (keys + 0.5) * VOX_M
    pairs = cKDTree(V).query_pairs(LINK_M, output_type="ndarray")
    n = len(V)
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else \
        coo_matrix((n, n))
    _, lab = connected_components(A, directed=False)
    comps = sorted((np.flatnonzero(lab == c) for c in np.unique(lab)), key=len, reverse=True)
    return V, comps


def cmd_leftover(a) -> int:
    from rowant_lab.libero_env import lfb_library as LL

    sid = LP.scene_id(a.suite, a.task, a.L, a.held_out)
    out = LP.LEARN / a.cond / sid / "leftover.json"
    if out.exists():
        print(f"exists: {out}")
        return 0
    typ = a.held_out or a.target_type
    var = LL.build_minus(a.suite, typ)
    t0 = time.perf_counter()
    per = perceiver(LP.LIB / f"minus-{a.suite}-{typ}")
    wc = per.cell(a.suite)
    fr, jp, gq = load_frame(LP.FRAMES / a.cond / sid / "agentview.npz")
    rep = per.full({"op": "look", "mode": "full", "suite": a.suite, "episode": f"lfb-{sid}", "look": 0,
                    "cameras": [fr], "joint_pos": jp, "gripper_qpos": gq})
    doc = {"scene": sid, "cond": a.cond, "held_out": typ, "minus_digest": var["digest"],
           "percepts": [{"type": p["type"], "gid": p["gid"], "bodies": p["bodies"]} for p in rep["percepts"]],
           "plane": rep["plane"], "perceive_s": rep["times"]["total_s"]}
    if rep["plane"] is None:
        doc["status"] = "no_plane"
    else:
        Pw, ok = world_points(per, wc, fr, jp, gq)
        keep = above_plane(Pw, ok, rep["plane"])
        P = Pw[keep]
        ex = explained(wc, rep["percepts"], P)
        V, comps = components(P[~ex])
        doc.update(n_above=int(keep.sum()), n_explained=int(ex.sum()), n_vox=int(len(V)),
                   comp_sizes=[int(len(c)) for c in comps[:5]])
        if comps and len(comps[0]) >= MIN_VOX:
            C = V[comps[0]]
            doc.update(status="ok", centre_w=[float(x) for x in C.mean(0)], bbox_min=[float(x) for x in C.min(0)],
                       bbox_max=[float(x) for x in C.max(0)], n_vox_object=int(len(C)),
                       runner_up_vox=int(len(comps[1])) if len(comps) > 1 else 0)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out.parent / "leftover-voxels.npy", C)
        else:
            doc["status"] = "no_leftover"
    doc["s"] = round(time.perf_counter() - t0, 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: doc.get(k) for k in ("scene", "status", "comp_sizes", "n_vox_object", "s")}))
    return 0


def cmd_masks(a) -> int:
    from scipy.spatial import cKDTree

    sid = LP.scene_id(a.suite, a.task, a.L, a.held_out)
    ld = LP.LEARN / a.cond / sid
    out = ld / f"{MASKS}.npz"
    if out.exists():
        print(f"exists: {out}")
        return 0
    lo = json.loads((ld / "leftover.json").read_text())
    looks = json.loads((LP.FRAMES / a.cond / sid / "looks.json").read_text())
    if lo.get("status") != "ok" or looks.get("status") != "ok":
        print(json.dumps({"scene": sid, "status": "skipped"}))
        return 0
    per = perceiver(LP.LIB / f"minus-{a.suite}-{lo['held_out']}")
    wc = per.cell(a.suite)
    foot = np.load(ld / "leftover-voxels.npy")[:, :2]
    ftree = cKDTree(foot)
    top = float(lo["bbox_max"][2]) + Z_PAD_M
    masks, rows = {}, []
    for r in looks["looks"]:
        if "sha256" not in r:
            continue
        i = int(r["look"])
        fr, jp, gq = load_frame(LP.FRAMES / a.cond / sid / f"wrist-{i}.npz")
        Pw, ok = world_points(per, wc, fr, jp, gq)
        keep = above_plane(Pw, ok, lo["plane"], MASK_BAND_M) & (np.nan_to_num(Pw[..., 2], nan=1e9) < top)
        idx = np.flatnonzero(keep.ravel())
        P = Pw.reshape(-1, 3)[idx]
        near, _ = ftree.query(P[:, :2], k=1, distance_upper_bound=R_PAD_M)
        sel = np.isfinite(near) & ~explained(wc, lo["percepts"], P)
        idx, P = idx[sel], P[sel]
        V, comps = components(P)
        m = np.zeros(keep.size, bool)
        if comps:
            vk = np.floor(P / VOX_M).astype(np.int64)
            ck = {tuple(x) for x in np.floor(V[comps[0]] / VOX_M).astype(np.int64)}
            m[idx[np.array([tuple(x) in ck for x in vk], bool)]] = True
        masks[f"mask_{i}"] = m.reshape(keep.shape)
        rows.append({"look": i, "n_px": int(m.sum()), "comp_sizes": [int(len(c)) for c in comps[:3]]})
    np.savez_compressed(out, **masks)
    (ld / f"{MASKS}.json").write_text(json.dumps({"scene": sid, "looks": rows}, indent=1))
    print(json.dumps({"scene": sid, "looks": rows}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("leftover", "masks"))
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--cond", choices=LP.CONDITIONS, default="clean")
    ap.add_argument("--suite", default=LP.SUITE)
    ap.add_argument("--held-out", default=None)
    ap.add_argument("--target-type", default=None, help="test scenes: the task's target type (the runner supplies "
                    "it from the instruction's grounding; it is the supervised label)")
    a = ap.parse_args(argv)
    if not (a.held_out or a.target_type):
        print("REFUSE: --held-out or --target-type is required", file=sys.stderr)
        return 2
    return cmd_leftover(a) if a.stage == "leftover" else cmd_masks(a)


if __name__ == "__main__":
    raise SystemExit(main())
