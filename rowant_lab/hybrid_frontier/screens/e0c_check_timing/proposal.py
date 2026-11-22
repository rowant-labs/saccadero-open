"""TRUTH-FREE pose proposal for the CHECK: a few (object, pose) guesses per HPG candidate, from the frame and the
object's OWN learned graph only. No S0c table, no mesh, no render key.

Why this exists. Screen E-0d searched poses on a grid built from the S0c rest table, an evaluator-only file from
which the dev cases were drawn, so every truth grid held the true rest class (E-0d VERIFIER, leakage-class
disclosure). This module replaces that with sources a robot would have.

**Rest classes from the learned graph** (``rest_classes``; computed once per object at load, from Monty's U77 graph
nodes):

* The graph's convex hull stands in for the support polygon of the object, and the hull's volume centroid for its
  centre of mass (uniform density assumed).
* Quasi-static drop model. For a gravity direction d (object frame, pointing down), the height of the centre of mass
  over the support plane is the support function ``h(d) = max_v (v - c) . d`` over the hull vertices. Rest poses are
  the local minima of h over the sphere of directions, and a pose's share is the share of directions whose steepest
  descent ends there. Here that is 2,000 Fibonacci directions, each moved to its lowest of 8 nearest neighbours until
  it stops. Each minimum is refined on a local polar grid (8 deg, then 1 deg).
* Minima within ``MERGE_DEG`` (15 deg) are merged, and their shares added. Classes with a share of at least 0.02 are
  kept, at most ``MAX_CLASSES`` (8), by share.
* **The learning pose.** The graph's own frame records how the object sat while it was learned: world A is y-up, so
  local -y is down. That direction is always offered. It is merged into a basin within 15 deg, or else added as its
  own class with share 0. This is the robot's own experience, not case truth. It is close to E-0d's "identity-drop
  class", so the same-face stratum (objects resting on their identity-drop face) is the easy stratum here too. Report
  the strata separately.

**Gravity and yaw from the frame** (``gravity_frame``, ``propose``):

* The table plane from ``plane_crop`` (RANSAC + least squares, already computed for HPG) gives the up axis. It also
  gives the gravity frame, with the table at z = 0.
* For each rest class k and yaw psi, the rotation is ``R = R_g Rz(psi) R_rest,k``, where R_rest,k turns the class's
  down direction to -z. The object's lowest hull vertex sits on the plane. The start xy aligns the centroid of the
  graph nodes that face the camera with the centroid of the patch centres.
* An xy-only trimmed point-to-point ICP follows (z stays fixed by the contact), from patch centres to graph nodes.
* **Coarse pass.** Yaw on a 30 deg grid, 64 patch centres, 3 ICP iterations (scored on the last iteration's
  distances, as E-0d's ICP).
* **Refinement.** The best 4 coarse poses per object, after 20 deg suppression of near duplicates, each with yaw
  offsets of +-5 and +-10 deg, 128 centres, 2 ICP iterations and a final scoring query.
* **Geometric score, lower is better:**
  - ``fit_mm``: the mean over the centres of min(nearest-node distance, 10 mm), in mm;
  - plus ``LAMBDA_MM`` (10) x ``free_space``: the share of the placed graph's nodes more than 6 mm above the table
    that project into the image OUTSIDE the blob (dilated by 3 px). For one object on a table, every object point
    that high must project onto the blob. This is the free-space test of direction note section 3.2, used here as a
    proposal score.
* The best ``POSES_PER_OBJECT`` (3) poses per object are then picked, skipping any pose within 10 deg and 10 mm of
  one already picked. With HPG's top 3 objects that gives k = 9 <= 10 guesses for the CHECK.

All constants are declared here and were fixed before any dev case was scored. Nothing was tuned on dev.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.spatial import ConvexHull, cKDTree

TRUTH_FREE = True
P_TRAIN = np.array([0.0, 1.5, 0.0])
LEARNING_DOWN = np.array([0.0, -1.0, 0.0])  # world A is y-up while Monty learns: local -y points down
N_DIRS = 2000
KNN_DIRS = 8
MERGE_DEG = 15.0
MIN_SHARE = 0.02
MAX_CLASSES = 8
COARSE_YAW_DEG = 30.0
REFINE_OFFSETS_DEG = (-10.0, -5.0, 5.0, 10.0)
N_ICP_COARSE = 64
N_ICP_FINE = 128
ICP_ITERS_COARSE = 3
ICP_ITERS_FINE = 2
TRIMS_M = (0.03, 0.02, 0.01, 0.01)
FIT_CAP_M = 0.01
LAMBDA_MM = 10.0
FS_MIN_HEIGHT_M = 0.006
FS_DILATE_PX = 3
N_FS_NODES = 256
N_VIS_NODES = 256
VIS_MIN_HEIGHT_M = 0.003
N_SEEDS = 4
NMS_SEED_DEG = 20.0
POSES_PER_OBJECT = 3
NMS_FINAL_DEG = 10.0
NMS_FINAL_M = 0.01
PARAMS = {k: (list(v) if isinstance(v, tuple) else (v.tolist() if isinstance(v, np.ndarray) else v))
          for k, v in dict(LEARNING_DOWN=LEARNING_DOWN, N_DIRS=N_DIRS, KNN_DIRS=KNN_DIRS, MERGE_DEG=MERGE_DEG,
                           MIN_SHARE=MIN_SHARE, MAX_CLASSES=MAX_CLASSES, COARSE_YAW_DEG=COARSE_YAW_DEG,
                           REFINE_OFFSETS_DEG=REFINE_OFFSETS_DEG, N_ICP_COARSE=N_ICP_COARSE, N_ICP_FINE=N_ICP_FINE,
                           ICP_ITERS_COARSE=ICP_ITERS_COARSE, ICP_ITERS_FINE=ICP_ITERS_FINE, TRIMS_M=TRIMS_M,
                           FIT_CAP_M=FIT_CAP_M, LAMBDA_MM=LAMBDA_MM, FS_MIN_HEIGHT_M=FS_MIN_HEIGHT_M,
                           FS_DILATE_PX=FS_DILATE_PX, N_FS_NODES=N_FS_NODES, N_VIS_NODES=N_VIS_NODES,
                           VIS_MIN_HEIGHT_M=VIS_MIN_HEIGHT_M, N_SEEDS=N_SEEDS, NMS_SEED_DEG=NMS_SEED_DEG,
                           POSES_PER_OBJECT=POSES_PER_OBJECT, NMS_FINAL_DEG=NMS_FINAL_DEG,
                           NMS_FINAL_M=NMS_FINAL_M).items()}


# ------------------------------------------------------------------------------------------- rotations
def rz(psi_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(psi_deg)), math.sin(math.radians(psi_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rz_stack(psis_deg) -> np.ndarray:
    a = np.radians(np.asarray(psis_deg, dtype=float))
    c, s = np.cos(a), np.sin(a)
    out = np.zeros((len(a), 3, 3))
    out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1], out[:, 2, 2] = c, -s, s, c, 1.0
    return out


def rotation_to(a, b) -> np.ndarray:
    """The minimal rotation R with R a = b (unit vectors)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -1 + 1e-12:  # antiparallel: 180 deg about any axis perpendicular to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def rotation_angle_deg(R1, R2) -> np.ndarray:
    """Angle(s) of R1^T R2 in degrees (broadcast over leading axes)."""
    c = (np.einsum("...ij,...ij->...", np.asarray(R1), np.asarray(R2)) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def angle_deg(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


# ------------------------------------------------------------------------------------------- rest classes
@dataclass
class RestClass:
    down: np.ndarray  # object-local unit vector pointing down at rest
    share: float
    source: str  # "support-basin", "support-basin+learning-pose" or "learning-pose"
    height_m: float  # support height h(down) of the centre of mass
    R_rest: np.ndarray  # object-local -> gravity frame at yaw 0 (turns ``down`` to -z)


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def hull_centroid(points: np.ndarray, hull: ConvexHull) -> np.ndarray:
    """Volume centroid of the convex hull (uniform density)."""
    c0 = points[hull.vertices].mean(axis=0)
    tri = points[hull.simplices]
    a, b, c = tri[:, 0] - c0, tri[:, 1] - c0, tri[:, 2] - c0
    vol = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
    if vol.sum() <= 0:
        return c0
    return c0 + ((a + b + c) / 4.0 * vol[:, None]).sum(axis=0) / vol.sum()


def _local_grid(d0: np.ndarray, radius_deg: float, n_r: int, n_a: int) -> np.ndarray:
    d0 = d0 / np.linalg.norm(d0)
    seed = np.array([1.0, 0.0, 0.0]) if abs(d0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d0, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d0, e1)
    out = [d0]
    for r in np.linspace(radius_deg / n_r, radius_deg, n_r):
        rr = math.radians(r)
        for a in np.linspace(0, 2 * math.pi, n_a, endpoint=False):
            v = math.cos(rr) * d0 + math.sin(rr) * (math.cos(a) * e1 + math.sin(a) * e2)
            out.append(v)
    return np.array(out)


def rest_classes(local: np.ndarray, *, n_dirs: int = N_DIRS, knn: int = KNN_DIRS, merge_deg: float = MERGE_DEG,
                 min_share: float = MIN_SHARE, max_classes: int = MAX_CLASSES,
                 learning_down=LEARNING_DOWN) -> tuple[list[RestClass], dict]:
    """Rest classes of one object from its learned graph's nodes (object-local, metres). See the module doc."""
    local = np.asarray(local, dtype=float)
    hull = ConvexHull(local)
    V = local[hull.vertices]
    com = hull_centroid(local, hull)
    Vr = V - com
    D = fibonacci_sphere(n_dirs)
    h = (Vr @ D.T).max(axis=0)
    _, nb = cKDTree(D).query(D, k=knn + 1)
    nxt = nb[np.arange(n_dirs), np.argmin(h[nb], axis=1)]
    root = nxt.copy()
    for _ in range(n_dirs):
        r2 = root[root]
        if np.array_equal(r2, root):
            break
        root = r2
    roots, counts = np.unique(root, return_counts=True)
    minima = []
    for r, cnt in zip(roots, counts):
        d = D[r]
        for rad, nr, na in ((8.0, 8, 16), (1.0, 4, 12)):
            G = _local_grid(d, rad, nr, na)
            hg = (Vr @ G.T).max(axis=0)
            d = G[int(np.argmin(hg))]
        minima.append((float(cnt) / n_dirs, d / np.linalg.norm(d), float((Vr @ d).max())))
    minima.sort(key=lambda m: -m[0])
    merged: list[list] = []  # [share, down, height]
    for share, d, hh in minima:
        for m in merged:
            if angle_deg(m[1], d) < merge_deg:
                m[0] += share
                break
        else:
            merged.append([share, d, hh])
    kept = [m for m in merged if m[0] >= min_share][:max_classes]
    classes = [RestClass(down=m[1], share=float(m[0]), source="support-basin", height_m=m[2],
                         R_rest=rotation_to(m[1], [0.0, 0.0, -1.0])) for m in kept]
    ld = np.asarray(learning_down, float) / np.linalg.norm(learning_down)
    near = [c for c in classes if angle_deg(c.down, ld) < merge_deg]
    if near:
        near[0].source = "support-basin+learning-pose"
    else:
        classes.append(RestClass(down=ld, share=0.0, source="learning-pose", height_m=float((Vr @ ld).max()),
                                 R_rest=rotation_to(ld, [0.0, 0.0, -1.0])))
    diag = {"n_hull_vertices": int(len(V)), "com_local": [float(x) for x in com], "n_minima": len(minima),
            "n_merged": len(merged), "share_kept": float(sum(c.share for c in classes)),
            "n_classes": len(classes)}
    return classes, diag


# ------------------------------------------------------------------------------------------- per-object shape
@dataclass
class ObjectShape:
    """What the proposal needs from one learned graph (built once at load)."""

    obj: str
    pos: np.ndarray  # (N, 3) model frame (the kernel's tree indexes these)
    tree: cKDTree
    hull_local: np.ndarray  # (V, 3) hull vertices, object-local
    vis_local: np.ndarray  # (<= N_VIS_NODES, 3)
    vis_normals: np.ndarray
    fs_local: np.ndarray  # (<= N_FS_NODES, 3)
    classes: list[RestClass]
    diag: dict = field(default_factory=dict)


def _stride(n: int, m: int) -> np.ndarray:
    return np.arange(n) if n <= m else np.unique(np.linspace(0, n - 1, m).round().astype(int))


def object_shape(obj: str, pos: np.ndarray, normals: np.ndarray, tree: cKDTree) -> ObjectShape:
    local = np.asarray(pos, float) - P_TRAIN
    classes, diag = rest_classes(local)
    hull = ConvexHull(local)
    iv = _stride(len(local), N_VIS_NODES)
    ifs = _stride(len(local), N_FS_NODES)
    return ObjectShape(obj=obj, pos=np.asarray(pos, float), tree=tree, hull_local=local[hull.vertices],
                       vis_local=local[iv], vis_normals=np.asarray(normals, float)[iv], fs_local=local[ifs],
                       classes=classes, diag=diag)


# ------------------------------------------------------------------------------------------- the frame
@dataclass
class Frame:
    """Per-frame inputs of the proposal, all truth-free (camera points in the gravity frame, camera, blob)."""

    R_g: np.ndarray  # gravity frame -> world rotation (z = table normal, up)
    o_g: np.ndarray  # world position of the gravity-frame origin (on the table plane)
    y: np.ndarray  # (P, 3) patch centres in the gravity frame
    cam_g: np.ndarray  # camera centre in the gravity frame
    R_cw: np.ndarray  # world -> camera rotation
    c_w: np.ndarray  # camera centre (world)
    intr: tuple  # fx, fy, cx, cy, width, height
    blob_dil: np.ndarray  # (H, W) dilated blob mask


def gravity_frame(plane: dict, camera: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(R_g, o_g)`` from ``plane_crop``'s plane (camera frame, normal towards the camera)."""
    T = np.asarray(camera["T_world_from_camera"], dtype=float)
    n_c = np.asarray(plane["normal"], float)
    n_w = T[:3, :3] @ n_c
    n_w /= np.linalg.norm(n_w)
    p0_w = T[:3, :3] @ (-float(plane["d"]) * n_c) + T[:3, 3]
    x = np.array([1.0, 0.0, 0.0]) - n_w[0] * n_w
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 1.0, 0.0]) - n_w[1] * n_w
    x /= np.linalg.norm(x)
    R_g = np.stack([x, np.cross(n_w, x), n_w], axis=1)
    return R_g, float(p0_w @ n_w) * n_w


def make_frame(plane: dict, camera: dict, mask: np.ndarray, centres_world: np.ndarray) -> Frame:
    R_g, o_g = gravity_frame(plane, camera)
    T = np.asarray(camera["T_world_from_camera"], dtype=float)
    c_w = T[:3, 3].copy()
    return Frame(R_g=R_g, o_g=o_g, y=(np.asarray(centres_world) - o_g) @ R_g, cam_g=(c_w - o_g) @ R_g,
                 R_cw=T[:3, :3].T, c_w=c_w,
                 intr=(float(camera["fx"]), float(camera["fy"]), float(camera["cx"]), float(camera["cy"]),
                       int(camera["width"]), int(camera["height"])),
                 blob_dil=dilate_square(mask, FS_DILATE_PX))


def dilate_square(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation by a (2r + 1)^2 square (separable max filter over the mask's bounding box plus r)."""
    m = np.asarray(mask, dtype=bool)
    out = np.zeros_like(m)
    rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
    if len(rows) == 0:
        return out
    r0, r1 = max(rows[0] - r, 0), min(rows[-1] + r + 1, m.shape[0])
    c0, c1 = max(cols[0] - r, 0), min(cols[-1] + r + 1, m.shape[1])
    sub = m[r0:r1, c0:c1].astype(np.uint8)
    sub = maximum_filter1d(maximum_filter1d(sub, 2 * r + 1, axis=0, mode="constant"), 2 * r + 1, axis=1,
                           mode="constant")
    out[r0:r1, c0:c1] = sub > 0
    return out


# ------------------------------------------------------------------------------------------- hypotheses
def _initial_t(shape: ObjectShape, Rp: np.ndarray, fr: Frame) -> np.ndarray:
    """(H, 3) gravity-frame start: lowest hull vertex on the table, camera-facing nodes' xy centroid on the
    centres' xy centroid."""
    RpT = np.transpose(Rp, (0, 2, 1))
    z = -(shape.hull_local @ Rp[:, 2, :].T).min(axis=0)
    X = shape.vis_local[None] @ RpT
    X[..., 2] += z[:, None]
    N = shape.vis_normals[None] @ RpT
    vis = (np.einsum("hni,hni->hn", N, fr.cam_g[None, None, :] - X) > 0) & (X[..., 2] > VIS_MIN_HEIGHT_M)
    few = vis.sum(axis=1) < 3
    vis[few] = True
    w = vis.astype(float)
    cxy = (X[..., :2] * w[..., None]).sum(axis=1) / w.sum(axis=1)[:, None]
    target = fr.y[:, :2].mean(axis=0)
    return np.concatenate([target[None, :] - cxy, z[:, None]], axis=1)


def _nn(shape: ObjectShape, Rp: np.ndarray, t: np.ndarray, y: np.ndarray):
    """Nearest-node distances (H, n) and the matched nodes in the gravity frame (H, n, 3)."""
    H, n = len(Rp), len(y)
    local = (y[None] - t[:, None, :]) @ Rp  # rows of Rp^T (y - t)
    d, idx = shape.tree.query((local + P_TRAIN).reshape(-1, 3), k=1, workers=1)
    g = shape.pos[idx.reshape(H, n)] - P_TRAIN
    return d.reshape(H, n), g @ np.transpose(Rp, (0, 2, 1)) + t[:, None, :]


def icp_xy(shape: ObjectShape, Rp: np.ndarray, t0: np.ndarray, y: np.ndarray, iters: int, *,
           final_query: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """xy-only trimmed point-to-point ICP for H fixed rotations. Returns ``(t, distances (H, n))``: the distances at
    the returned ``t`` (``final_query``), or those of the last iteration's query, before its step (E-0d's
    ``icp_translation`` convention; one tree query fewer)."""
    t = np.array(t0, dtype=float)
    d = None
    for it in range(iters):
        d, gw = _nn(shape, Rp, t, y)
        keep = d <= TRIMS_M[min(it, len(TRIMS_M) - 1)]
        cnt = keep.sum(axis=1)
        resid = (gw - y[None])[..., :2]
        step = np.where(cnt[:, None] > 0, (resid * keep[..., None]).sum(axis=1) / np.maximum(cnt, 1)[:, None], 0.0)
        t[:, :2] -= step
    if final_query or d is None:
        d, _ = _nn(shape, Rp, t, y)
    return t, d


def free_space(shape: ObjectShape, Rp: np.ndarray, t: np.ndarray, fr: Frame) -> np.ndarray:
    """(H,) share of the placed nodes above ``FS_MIN_HEIGHT_M`` that project inside the image but off the blob."""
    Xg = shape.fs_local[None] @ np.transpose(Rp, (0, 2, 1)) + t[:, None, :]
    high = Xg[..., 2] > FS_MIN_HEIGHT_M
    Xw = Xg @ fr.R_g.T + fr.o_g
    Xc = (Xw - fr.c_w) @ fr.R_cw.T
    fx, fy, cx, cy, W, Hh = fr.intr
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.rint(fx * Xc[..., 0] / Xc[..., 2] + cx)
        v = np.rint(fy * Xc[..., 1] / Xc[..., 2] + cy)
    inside = high & (Xc[..., 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < Hh)
    ui = np.where(inside, u, 0).astype(np.int64)
    vi = np.where(inside, v, 0).astype(np.int64)
    off = inside & ~fr.blob_dil[vi, ui]
    return off.sum(axis=1) / np.maximum(inside.sum(axis=1), 1)


def _score(d: np.ndarray, fs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fit_mm = np.minimum(d, FIT_CAP_M).mean(axis=1) * 1e3
    return fit_mm, fit_mm + LAMBDA_MM * fs


def _nms(order, Rs, ts, n_keep: int, deg: float, metres: float | None) -> list[int]:
    picked: list[int] = []
    for i in order:
        ok = True
        for j in picked:
            if rotation_angle_deg(Rs[i], Rs[j]) < deg and (metres is None or np.linalg.norm(ts[i] - ts[j]) < metres):
                ok = False
                break
        if ok:
            picked.append(int(i))
            if len(picked) >= n_keep:
                break
    return picked


def propose(shape: ObjectShape, fr: Frame, *, poses: int = POSES_PER_OBJECT, keep_all: bool = False) -> dict:
    """The proposal for one object (module doc). Returns ``{"guesses": [...], "n_hyp": ...}``; each guess has the
    WORLD-frame ``R`` (object-local -> world) and ``t``, its class, yaw and scores. ``keep_all`` also returns every
    refined hypothesis (for diagnostics)."""
    ycoarse = fr.y[_stride(len(fr.y), N_ICP_COARSE)]
    yfine = fr.y[_stride(len(fr.y), N_ICP_FINE)]
    yaws = np.arange(0.0, 360.0, COARSE_YAW_DEG)
    Rz = rz_stack(yaws)
    Rp = np.concatenate([Rz @ c.R_rest for c in shape.classes], axis=0)  # gravity frame rotations
    labels = [(k, float(p)) for k in range(len(shape.classes)) for p in yaws]
    t0 = _initial_t(shape, Rp, fr)
    t, d = icp_xy(shape, Rp, t0, ycoarse, ICP_ITERS_COARSE, final_query=False)
    fs = free_space(shape, Rp, t, fr)
    _, sc = _score(d, fs)
    seeds = _nms(np.argsort(sc, kind="stable"), Rp, t, N_SEEDS, NMS_SEED_DEG, None)
    # refinement around the seeds
    R2, t2, lab2 = [], [], []
    for s in seeds:
        k, psi = labels[s]
        for off in (0.0,) + tuple(REFINE_OFFSETS_DEG):
            R2.append(rz(psi + off) @ shape.classes[k].R_rest)
            t2.append(t[s])
            lab2.append((k, (psi + off) % 360.0))
    R2 = np.stack(R2)
    t2 = np.stack(t2)
    t2, d2 = icp_xy(shape, R2, t2, yfine, ICP_ITERS_FINE)
    fs2 = free_space(shape, R2, t2, fr)
    fit2, sc2 = _score(d2, fs2)
    order = np.argsort(sc2, kind="stable")
    Rw = fr.R_g[None] @ R2
    tw = t2 @ fr.R_g.T + fr.o_g
    pick = _nms(order, Rw, tw, poses, NMS_FINAL_DEG, NMS_FINAL_M)

    def rec(i):
        k, psi = lab2[i]
        return {"obj": shape.obj, "R": Rw[i], "t": tw[i], "class": int(k), "psi_deg": float(psi),
                "class_share": float(shape.classes[k].share), "class_source": shape.classes[k].source,
                "fit_mm": float(fit2[i]), "free_space": float(fs2[i]), "geo_score": float(sc2[i])}

    out = {"guesses": [rec(i) for i in pick], "n_hyp_coarse": int(len(Rp)), "n_hyp_fine": int(len(R2))}
    if keep_all:
        out["all"] = [rec(i) for i in order]
    return out
