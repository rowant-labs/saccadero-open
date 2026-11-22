"""LF-b memory builders and the shared derivation (Monty venv; registration ``lfc_b``).

Launch from the repository root (``PYTHONPATH=$PWD``, ``python -B``, never ``-I``)::

    python -B -m rowant_lab.libero_env.lfb_build build --builder monty|tsdf --task T --L L --cond C \\
        [--suite S --held-out TYPE --target-type TYPE] [--config JSON] [--tag TAG]

TRUTH-FREE: reads the wrist looks (``frames/``), their masks and the leftover record (``learn/``), and M1's
robot.json (the jaw). Never ``truth/``, never a mesh of the held-out type.

**Model frame.** The world axes, with the origin at the leftover's centre on the support plane, so z = 0 is the
plane. No object pose is given: the object is static while it is learned. ``rest_up_in_body`` is +z.

**Builders** (all of them read the same masked looks):
* ``tsdf`` (A2/A3, the standard reconstruct-then-pose way):
  - projective TSDF over the masked looks: voxel ``vox_mm``, truncation ``trunc_k`` × voxel;
  - a voxel in front of an unmasked surface is free, and below the plane is outside;
  - voxels seen only behind the masked surface are inside;
  - marching cubes gives the largest mesh component, coloured by the mean of the looks that see each vertex;
  - written as ``render.ply``, then M2's own onboarding of that mesh (``m2_onboard.build_object_grid``, 2 mm)
    gives the nodes.
* ``monty`` (A1, Monty's learning):
  - patches of ``patch_px`` (odd) at a ``stride_px`` grid over each mask go through Monty's own
    ``ObservationProcessor.process`` (TLS normals, weighted principal curvatures);
  - the percepts it would pass to a learning module (``process_features_in_lm``) go to Monty's
    ``GridObjectModel.build_model`` (EvidenceGraphMemory's model: ``num_voxels_per_dim``, ``max_nodes``,
    ``max_size``);
  - its graph is exported through e0d's ``channel_arrays``.
  Monty's camera convention: ``cam_to_world``'s z column points back at the viewer (OpenCV T times diag(1, -1, -1)).

**Shared derivation** (``derive``, identical for every builder, from the nodes only):
1. 2 mm occupancy of the nodes, dilated once, the footprint extended down to the plane, closing, hole filling,
   then eroded once: a solid no larger than the nodes' hull plus a voxel.
2. Marching cubes gives ``surface.ply``, coloured by the nearest node's HSV.
3. Collision boxes: 1 cm z-slabs, each a box in the footprint's minimum-area rectangle axes, merged while within
   3 mm.
4. COM is the solid's centroid.
5. Grasp records: M1's ``grasp.build_records`` on those boxes.
Writes ``nodes.npz``, ``surface.ply``, ``m1type.json``, ``records.json`` and ``build.json`` (times, counts,
config) into ``models/<builder>[-tag]/<cond>/<scene>/``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from rowant_lab.libero_env import lfb_paths as LP

DEFAULTS = {"tsdf": {"vox_mm": 3.0, "trunc_k": 3.0},
            "monty": {"num_voxels_per_dim": 100, "max_nodes": 2000, "max_size": 0.3, "patch_px": 17, "stride_px": 3},
            # lfc_b_noise A4: LF-b's tuned winners (LFB-CODE), Monty's and the TSDF's, fixed (no retuning)
            "montyfused": {"num_voxels_per_dim": 150, "max_nodes": 2000, "max_size": 0.3, "patch_px": 17, "stride_px": 3,
                           "vox_mm": 2.0, "trunc_k": 3.0}}
OCC_M, SLAB_M, MERGE_M = 0.002, 0.010, 0.003
MONTY_FEATURES = ["pose_vectors", "pose_fully_defined", "hsv", "principal_curvatures_log"]


# ------------------------------------------------------------------------------------------------ inputs
def looks(cond: str, sid: str) -> tuple[list, dict]:
    from rowant_lab.libero_env import lfb_learn as LL

    lo = json.loads((LP.LEARN / cond / sid / "leftover.json").read_text())
    with np.load(LP.LEARN / cond / sid / f"{LL.MASKS}.npz") as z:
        masks = {k: np.asarray(z[k]) for k in z.files}
    out = []
    for k in sorted(masks, key=lambda s: int(s.split("_")[1])):
        fr, _jp, _gq = LL.load_frame(LP.FRAMES / cond / sid / f"wrist-{k.split('_')[1]}.npz")
        z_ = fr["depth_u16"].astype(np.float64) * float(fr["depth_scale_mm"]) * 1e-3
        out.append({"rgb": fr["rgb"], "z": z_, "K": np.asarray(fr["K"], float),
                    "T": np.asarray(fr["T_world_from_camera"], float), "mask": masks[k] & (z_ > 0)})
    return out, lo


def origin(lo: dict) -> np.ndarray:
    c = np.asarray(lo["centre_w"], float)
    return np.array([c[0], c[1], float(lo["plane"]["height_w"])])


def cam_points(lk: dict) -> np.ndarray:
    H, W = lk["z"].shape
    K = lk["K"]
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    z = lk["z"]
    return np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)


def masked_world(lks, o) -> np.ndarray:
    pts = []
    for lk in lks:
        P = cam_points(lk)[lk["mask"]]
        pts.append(P @ lk["T"][:3, :3].T + lk["T"][:3, 3] - o)
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def write_ply(path: Path, V_m, F, N, RGB01) -> None:
    from rowant_lab.libero_env import m2_onboard as OBM

    OBM.write_ply(path, {"V": np.asarray(V_m, float), "F": np.asarray(F, np.int64), "N": np.asarray(N, float),
                         "textured": False, "RGB": np.clip(np.asarray(RGB01, float), 0, 1)})


def orient_out(V, N) -> np.ndarray:
    """Unit normals, flipped as a whole when they point into the mesh on average (marching cubes' sign depends on
    which side holds the higher values)."""
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return -N if float(np.mean(np.sum((V - V.mean(0)) * N, 1))) < 0 else N


def largest_component(V, F):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(V)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
    keep_lab = np.bincount(lab).argmax()
    keepv = lab == keep_lab
    remap = -np.ones(n, np.int64)
    remap[keepv] = np.arange(keepv.sum())
    Fk = F[keepv[F].all(1)]
    return V[keepv], remap[Fk], keepv


# --------------------------------------------------------------------------------------------- tsdf builder
def project(lk, X_w):
    R, t, K = lk["T"][:3, :3], lk["T"][:3, 3], lk["K"]
    Xc = (X_w - t) @ R
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
        v = K[1, 1] * Xc[:, 1] / z + K[1, 2]
    H, W = lk["z"].shape
    ui, vi = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    inside = (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    return inside, np.clip(vi, 0, H - 1), np.clip(ui, 0, W - 1), z


def tsdf_build(lks, o, cfg) -> dict:
    from skimage.measure import marching_cubes

    vox = float(cfg["vox_mm"]) * 1e-3
    tr = float(cfg["trunc_k"]) * vox
    # a look marked ``carve_only`` (lfb_episode, ``learn_top_first``: the main camera's frame on the colour line, whose
    # depth of the thing itself is a guess) gives no surface: only the space its other pixels show to be empty
    P = masked_world([lk for lk in lks if not lk.get("carve_only")], o)
    lo_b = P.min(0) - 0.02
    hi_b = P.max(0) + 0.02
    lo_b[2] = -2 * vox
    shape = np.ceil((hi_b - lo_b) / vox).astype(int) + 1
    g = np.stack(np.meshgrid(*[lo_b[i] + vox * np.arange(shape[i]) for i in range(3)], indexing="ij"), -1)
    X = g.reshape(-1, 3)
    Xw = X + o
    S = np.zeros(len(X))
    Wt = np.zeros(len(X))
    behind = np.zeros(len(X), np.int64)
    seen_m = np.zeros(len(X), np.int64)
    for lk in lks:
        ins, vi, ui, z = project(lk, Xw)
        d = np.where(ins, lk["z"][vi, ui], 0.0)
        m = ins & lk["mask"][vi, ui]
        valid = ins & (d > 0)
        sdf = d - z
        carve = bool(lk.get("carve_only"))
        if not carve:
            on = valid & m & (sdf > -tr)
            S[on] += np.clip(sdf[on] / tr, -1, 1)
            Wt[on] += 1
        free = valid & ~m & (sdf > 0)  # in front of a non-object surface: empty (standard free space)
        S[free] += 1.0
        Wt[free] += 1
        if not carve:
            behind += valid & m & (sdf <= -tr)
            seen_m += valid & m
    F = np.where(Wt > 0, S / np.maximum(Wt, 1), 1.0)
    from rowant_lab.libero_env import lfb_learn as LL

    # inside: unobserved, behind the masked surface in every look that sees it (2 or more), and above the masks'
    # support band (below it the object's sides are unmasked, so "behind" carries no information)
    F[(Wt == 0) & (behind >= 2) & (behind == seen_m) & (X[:, 2] >= LL.MASK_BAND_M)] = -1.0
    F[X[:, 2] < 0] = 1.0
    vol = F.reshape(shape)
    V, Fc, N, _ = marching_cubes(np.pad(vol, 1, constant_values=1.0), level=0.0, spacing=(vox,) * 3)
    V = V - vox + lo_b
    V, Fc, keep = largest_component(V, Fc)
    N = orient_out(V, N[keep])
    rgb_sum, cnt = np.zeros((len(V), 3)), np.zeros(len(V))
    for lk in lks:
        ins, vi, ui, z = project(lk, V + o)
        d = np.where(ins, lk["z"][vi, ui], 0.0)
        ok = ins & lk["mask"][vi, ui] & (np.abs(d - z) < 2 * vox)
        rgb_sum[ok] += lk["rgb"][vi[ok], ui[ok]] / 255.0
        cnt[ok] += 1
    mean = rgb_sum[cnt > 0].mean(0) if (cnt > 0).any() else np.full(3, 0.5)
    RGB = np.where(cnt[:, None] > 0, rgb_sum / np.maximum(cnt, 1)[:, None], mean)
    return {"V": V, "F": Fc, "N": N, "RGB": RGB, "n_points": int(len(P)), "grid": [int(x) for x in shape],
            "vol": vol, "lo": lo_b, "vox": vox, "trunc": tr}


def observed(lks, o, P_m, N_m, vox: float) -> np.ndarray:
    """Nodes some look saw: facing the camera, projecting into its mask, and within 2 voxels of its depth."""
    seen = np.zeros(len(P_m), bool)
    for lk in lks:
        ins, vi, ui, z = project(lk, P_m + o)
        d = np.where(ins, lk["z"][vi, ui], 0.0)
        facing = np.sum(N_m * (lk["T"][:3, 3] - (P_m + o)), 1) > 0
        seen |= ins & lk["mask"][vi, ui] & (np.abs(d - z) < 2 * vox) & facing
    return seen


def fuse_depth(lks, o, cfg) -> tuple[list, dict]:
    """lfc_b_noise A4: each look's masked depth replaced by a raycast of the scene's TSDF volume (KinectFusion's
    fused depth). Along each masked pixel's ray, the first + to - zero crossing within 3 truncations of the observed
    depth (trilinear samples at half-voxel steps, linear root); pixels with no crossing leave the mask."""
    from scipy.ndimage import map_coordinates

    m = tsdf_build(lks, o, cfg)
    vol, lo_b, vox, tr = m["vol"], m["lo"], m["vox"], m["trunc"]
    zs = np.arange(-3 * tr, 3 * tr + 1e-12, vox / 2)
    out, kept, n_in, dz = [], 0, 0, []
    for lk in lks:
        K, R, t = lk["K"], lk["T"][:3, :3], lk["T"][:3, 3]
        rr, cc = np.nonzero(lk["mask"])
        dc = np.stack([(cc + 0.5 - K[0, 2]) / K[0, 0], (rr + 0.5 - K[1, 2]) / K[1, 1], np.ones(len(rr))], -1)
        z = lk["z"][rr, cc][:, None] + zs[None, :]
        X = (dc[:, None, :] * z[..., None]) @ R.T + t - o
        f = map_coordinates(vol, ((X - lo_b) / vox).reshape(-1, 3).T, order=1, mode="constant", cval=1.0)
        f = f.reshape(z.shape)
        cross = (f[:, :-1] > 0) & (f[:, 1:] <= 0)
        has = cross.any(1)
        i = np.argmax(cross, 1)
        a, b = f[np.arange(len(i)), i], f[np.arange(len(i)), i + 1]
        zf = z[np.arange(len(i)), i] + np.where(has, a / np.maximum(a - b, 1e-12), 0.0) * (vox / 2)
        z_new = np.zeros_like(lk["z"])
        z_new[rr[has], cc[has]] = zf[has]
        mask = np.zeros_like(lk["mask"])
        mask[rr[has], cc[has]] = True
        dz.append(np.abs(zf[has] - lk["z"][rr[has], cc[has]]))
        kept += int(has.sum())
        n_in += int(len(rr))
        out.append({**lk, "z": z_new, "mask": mask})
    dz = np.concatenate(dz) if dz else np.zeros(0)
    return out, {"fused_kept": kept, "fused_masked": n_in,
                 "fused_minus_raw_abs_mm_median": round(float(np.median(dz)) * 1e3, 3) if len(dz) else None}


# -------------------------------------------------------------------------------------------- monty builder
def monty_build(lks, o, cfg, typ: str) -> dict:
    from tbp.monty.frameworks.models.object_model import GridObjectModel
    from tbp.monty.frameworks.models.sensor_modules import ObservationProcessor

    from rowant_lab.hybrid_frontier.screens.e0d_feature_transfer import graphs as e0d

    op = ObservationProcessor(features=list(MONTY_FEATURES), sensor_module_id="patch")
    h, s = int(cfg["patch_px"]) // 2, int(cfg["stride_px"])
    locs, feats = [], {f: [] for f in MONTY_FEATURES}
    n_patch = 0
    gl = np.diag([1.0, -1.0, -1.0, 1.0])
    for lk in lks:
        Pc = cam_points(lk)
        Pw = Pc @ lk["T"][:3, :3].T + lk["T"][:3, 3] - o
        c2w = lk["T"] @ gl
        c2w[:3, 3] -= o
        H, W = lk["z"].shape
        rr, cc = np.nonzero(lk["mask"])
        sel = (rr % s == 0) & (cc % s == 0) & (rr >= h) & (rr < H - h) & (cc >= h) & (cc < W - h)
        for r, c in zip(rr[sel], cc[sel]):
            sl = (slice(r - h, r + h + 1), slice(c - h, c + h + 1))
            sem = lk["mask"][sl].astype(np.float64)
            pw = np.nan_to_num(Pw[sl].reshape(-1, 3))
            pc = np.nan_to_num(Pc[sl].reshape(-1, 3))
            rgba = np.concatenate([lk["rgb"][sl], np.full(sem.shape + (1,), 255, np.uint8)], -1)
            obs = {"semantic_3d": np.c_[pw, sem.ravel()], "sensor_frame_data": np.c_[pc, sem.ravel()],
                   "cam_to_world": c2w, "rgba": rgba, "depth": lk["z"][sl]}
            n_patch += 1
            msg = op.process(obs)
            nm = msg.non_morphological_features
            if not msg.process_features_in_lm or "principal_curvatures_log" not in nm:
                continue
            mf = msg.morphological_features
            locs.append(np.asarray(msg.location, float))
            feats["pose_vectors"].append(np.asarray(mf["pose_vectors"], float).reshape(9))
            feats["pose_fully_defined"].append(float(mf["pose_fully_defined"]))
            feats["hsv"].append(np.asarray(nm["hsv"], float))
            feats["principal_curvatures_log"].append(np.asarray(nm["principal_curvatures_log"], float))
    L = np.asarray(locs)
    fd = {k: np.asarray(v, float) for k, v in feats.items()}
    model = GridObjectModel(object_id=typ, max_nodes=int(cfg["max_nodes"]), max_size=float(cfg["max_size"]),
                            num_voxels_per_dim=int(cfg["num_voxels_per_dim"]))
    model.build_model(locations=L, features=fd)
    pos, xk, mapping = e0d.channel_arrays(model)
    return {"pos": pos, "x": xk, "mapping": mapping, "n_patches": n_patch, "n_obs": int(len(L))}


# ----------------------------------------------------------------------------------------- shared derivation
def min_area_axes(xy: np.ndarray) -> np.ndarray:
    """2x2 columns: the axes of the footprint's minimum-area bounding rectangle (rotating calipers on the hull)."""
    from scipy.spatial import ConvexHull

    H = xy[ConvexHull(xy).vertices]
    best = None
    for i in range(len(H)):
        e = H[(i + 1) % len(H)] - H[i]
        a = np.array([e, [-e[1], e[0]]]).T / max(np.linalg.norm(e), 1e-12)
        q = xy @ a
        area = float(np.prod(q.max(0) - q.min(0)))
        if best is None or area < best[0]:
            best = (area, a)
    return best[1]


def _slab_box(xy: np.ndarray, ax: np.ndarray, z0: float, z1: float) -> dict:
    loc = xy @ ax
    lo, hi = loc.min(0) - OCC_M / 2, loc.max(0) + OCC_M / 2
    cen = ax @ ((lo + hi) / 2)
    R = np.eye(3)
    R[:2, 0], R[:2, 1] = ax[:, 0], ax[:, 1]
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1
    return {"R": R, "centre": np.array([cen[0], cen[1], (z0 + z1) / 2]),
            "half": np.array([(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, (z1 - z0) / 2])}


def derive(pos_m: np.ndarray, hsv: np.ndarray, typ: str) -> dict:
    """Nodes (model frame, metres; z = 0 is the support) -> solid, surface mesh, boxes, COM, grasp records."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from skimage.color import hsv2rgb
    from skimage.measure import marching_cubes

    from rowant_lab.libero_env import grasp as GR
    from rowant_lab.libero_env import m2_paths as MP

    lo = pos_m.min(0) - 3 * OCC_M
    lo[2] = -OCC_M
    hi = pos_m.max(0) + 3 * OCC_M
    shape = np.ceil((hi - lo) / OCC_M).astype(int) + 1
    occ = np.zeros(shape, bool)
    ix = np.clip(np.floor((pos_m - lo) / OCC_M).astype(int), 0, shape - 1)
    occ[tuple(ix.T)] = True
    occ = ndimage.binary_dilation(occ, iterations=1)
    foot = ndimage.binary_fill_holes(occ.any(2))
    zmin = int(ix[:, 2].min())
    k0 = int(np.floor((0.0 - lo[2]) / OCC_M)) + 1
    for k in range(0, max(zmin, k0) + 1):  # the unseen bottom: the footprint down to (and under) the support
        occ[:, :, k] |= foot
    occ = ndimage.binary_closing(occ, iterations=2)
    solid = ndimage.binary_erosion(ndimage.binary_fill_holes(occ), iterations=1)  # undo the dilation's growth
    for k in range(min(k0, shape[2])):
        solid[:, :, k] = False
    V, F, N, _ = marching_cubes(np.pad(solid.astype(np.float32), 1), level=0.5, spacing=(OCC_M,) * 3)
    V = V - OCC_M + lo + OCC_M / 2
    V, F, keep = largest_component(V, F)
    N = orient_out(V, N[keep])
    _, nn = cKDTree(pos_m).query(V, k=1)
    RGB = hsv2rgb(np.clip(hsv[nn], 0, 1)[None])[0]
    # boxes from 1 cm slabs
    cells = np.argwhere(solid)
    P = lo + (cells + 0.5) * OCC_M
    boxes = []
    ax = min_area_axes(P[:, :2])  # one footprint orientation for every slab
    for z0 in np.arange(0.0, P[:, 2].max() + SLAB_M, SLAB_M):
        sel = (P[:, 2] >= z0) & (P[:, 2] < z0 + SLAB_M)
        if sel.sum() < 4:
            continue
        b = _slab_box(P[sel, :2], ax, max(z0, float(P[sel, 2].min()) - OCC_M / 2), float(P[sel, 2].max()) + OCC_M / 2)
        prev = boxes[-1] if boxes else None
        if prev is not None and np.all(np.abs(prev["half"][:2] - b["half"][:2]) <= MERGE_M) \
                and np.linalg.norm(prev["centre"][:2] - b["centre"][:2]) <= MERGE_M:
            z_lo = prev["centre"][2] - prev["half"][2]
            z_hi = b["centre"][2] + b["half"][2]
            prev["centre"][2], prev["half"][2] = (z_lo + z_hi) / 2, (z_hi - z_lo) / 2
            continue
        boxes.append(b)
    prims = [{"R": b["R"].tolist(), "centre": b["centre"].tolist(), "half": b["half"].tolist(), "type": "box",
              "exact": False, "geom": f"lfb{i}"} for i, b in enumerate(boxes)]
    com = P.mean(0)
    robot = json.loads((MP.M1_LIBRARY / "robot.json").read_text())
    jaw = GR.JawModel.from_robot(robot)
    t0 = time.perf_counter()
    rec = GR.build_records(prims, [0.0, 0.0, 1.0], com, jaw, GR.RecordParams(), name=typ)
    records = GR.records_to_json(rec)
    records["source"] = {"lfb": "derived from learned nodes"}
    m1type = {"bodies": {"main": {"com": [float(x) for x in com], "is_root": True, "mass": 0.1, "primitives": prims}},
              "kind": "object", "rest_up_in_body": [0.0, 0.0, 1.0], "root_body": "main", "subtree_mass": 0.1,
              "type": typ}
    return {"V": V, "F": F, "N": N, "RGB": RGB, "m1type": m1type, "records": records,
            "n_boxes": len(prims), "n_records_kept": int(len(rec["width"])), "records_s": round(time.perf_counter() - t0, 2),
            "solid_voxels": int(solid.sum()), "extent_mm": [round(float(x) * 1e3, 1) for x in P.max(0) - P.min(0)]}


# ---------------------------------------------------------------------------------------------------- run
def build(builder: str, cond: str, sid: str, typ: str, cfg: dict, out: Path) -> dict:
    from rowant_lab.fastlayer import onboard as OB
    from rowant_lab.libero_env import m2_onboard as OBM

    t0 = time.perf_counter()
    lks, lo = looks(cond, sid)
    o = origin(lo)
    out.mkdir(parents=True, exist_ok=False)
    info = {"builder": builder, "cond": cond, "scene": sid, "type": typ, "config": cfg, "origin_w": o.tolist(),
            "n_looks": len(lks)}
    a = time.perf_counter()
    if builder == "tsdf":
        m = tsdf_build(lks, o, cfg)
        write_ply(out / "render.ply", m["V"], m["F"], m["N"], m["RGB"])
        pos, x, diag = OBM.build_object_grid("1", out / "render.ply", None, OB.GRID_M)
        seen = observed(lks, o, pos - OB.P_TRAIN, x[:, 0:3], float(cfg["vox_mm"]) * 1e-3)
        info.update(n_points=m["n_points"], grid=m["grid"], render_vertices=int(len(m["V"])),
                    n_nodes_unobserved=int((~seen).sum()))
        pos, x = pos[seen], x[seen]  # the memory keeps observed surface only (never the filled-in bottom)
    else:
        if builder == "montyfused":  # lfc_b_noise A4: Monty learning on the TSDF-fused depth of the same looks
            lks, fi = fuse_depth(lks, o, {k: cfg[k] for k in ("vox_mm", "trunc_k")})
            info.update(fi)
        m = monty_build(lks, o, cfg, typ)
        pos, x = m["pos"] + OB.P_TRAIN, m["x"]
        if m["mapping"] != OB.FEATURE_MAPPING:
            raise ValueError(f"Monty graph layout {m['mapping']} differs from CHECK's {OB.FEATURE_MAPPING}")
        info.update(n_patches=m["n_patches"], n_obs=m["n_obs"])
    info["learn_s"] = round(time.perf_counter() - a, 2)
    np.savez(out / "nodes.npz", pos=np.ascontiguousarray(pos), x=np.ascontiguousarray(x))
    hs = OB.FEATURE_MAPPING["hsv"][0]
    d = derive(pos - OB.P_TRAIN, x[:, hs:hs + 3], typ)
    write_ply(out / "surface.ply", d["V"], d["F"], d["N"], d["RGB"])
    (out / "m1type.json").write_text(json.dumps(d["m1type"], indent=1, sort_keys=True))
    (out / "records.json").write_text(json.dumps(d["records"], sort_keys=True) + "\n")
    info.update(n_nodes=int(len(pos)), n_boxes=d["n_boxes"], n_records_kept=d["n_records_kept"],
                solid_voxels=d["solid_voxels"], extent_mm=d["extent_mm"], records_s=d["records_s"],
                total_s=round(time.perf_counter() - t0, 2))
    (out / "build.json").write_text(json.dumps(info, indent=1))
    return info


def model_dir(builder: str, cond: str, sid: str, tag: str | None = None) -> Path:
    return LP.MODELS / (builder + (f"-{tag}" if tag else "")) / cond / sid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("build",))
    ap.add_argument("--builder", choices=LP.BUILDERS + (LP.NOISE_BUILDER,), required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--cond", choices=LP.CONDITIONS, default="clean")
    ap.add_argument("--suite", default=LP.SUITE)
    ap.add_argument("--held-out", default=None)
    ap.add_argument("--target-type", default=None)
    ap.add_argument("--config", default=None, help="JSON overrides of the builder's DEFAULTS (tuning)")
    ap.add_argument("--tag", default=None, help="a tuning config's tag (its own model folder)")
    a = ap.parse_args(argv)
    typ = a.held_out or a.target_type
    if not typ:
        print("REFUSE: --held-out or --target-type is required", file=sys.stderr)
        return 2
    sid = LP.scene_id(a.suite, a.task, a.L, a.held_out)
    out = model_dir(a.builder, a.cond, sid, a.tag)
    if (out / "build.json").exists():
        print(f"exists: {out}")
        return 0
    cfg = {**DEFAULTS[a.builder], **(json.loads(a.config) if a.config else {})}
    info = build(a.builder, a.cond, sid, typ, cfg, out)
    print(json.dumps({k: info[k] for k in ("builder", "scene", "n_nodes", "n_boxes", "n_records_kept", "extent_mm",
                                           "learn_s", "total_s")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
