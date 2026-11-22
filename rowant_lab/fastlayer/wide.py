"""TRUTH-FREE: the widened, clutter-aware pose proposal (``hope`` preset). New code, modelled on
``screens/e0c_check_timing/proposal.propose``; its helpers are imported from there unchanged.

Why it is needed. E-0c's proposal assumes one object resting on a visible table: the rotation is a rest class (from
the graph's hull) turned about the table normal, the object's lowest point touches the table (z fixed by contact),
and every object point more than 6 mm high must project onto the blob. HOPE scenes break all three: objects lean on
each other, lie on top of others, and occlude one another. So:

* **Rotations** ``R = R_g Rz(psi) R_down``: the rest-class downs from the hull (``proposal.rest_classes`` through
  ``memory.mesh_shape``) plus ``N_SPHERE_DOWNS`` Fibonacci directions over the whole sphere, each turned to -z, times
  a ``COARSE_YAW_DEG`` yaw grid about gravity. With downs over the sphere and every yaw, this covers all of SO(3)
  (at about 30 deg), so leaning and lying objects are in the set; the rest classes keep the likely ones exact.
  ``R_g`` comes from the frame's table plane (``proposal.gravity_frame``); if no plane was found, the camera's up
  (-y) stands in (any frame would do for the sphere set).
* **Translation**: start with the centroid of the hypothesis's camera-facing nodes on the centroid of the target's
  patch centres (3D), then a trimmed point-to-point ICP on the translation only, in 3D (no table contact).
* **Score** (lower is better): mean of min(nearest-node distance, ``FIT_CAP_M``) over the centres, in mm, plus
  ``LAMBDA_MM`` x the clutter free-space share: of the camera-facing nodes of the 256-node subsample that project to a
  pixel with a depth reading, the share that would lie more than ``FS_TOL_M`` IN FRONT of the observed surface (the
  camera would have seen them there). Nodes behind the observed surface are allowed (occluders, self-occlusion).
* **Coarse to fine**: coarse pass on ``N_ICP_COARSE`` centres; ``N_SEEDS`` seeds after ``NMS_SEED_DEG`` rotation
  suppression; each seed with yaw offsets ``REFINE_OFFSETS_DEG`` on ``N_ICP_FINE`` centres; the best ``POSES`` after
  ``NMS_FINAL_DEG`` / ``NMS_FINAL_M`` suppression go to the CHECK.

All constants are fixed here, before any HOPE val number was seen.

**Step 4 speed options** (``WideOpts``; the defaults are the step-3 proposal, bit for bit, tested):

* ``cache`` (E2, exact): the coarse rotation set and its labels are built once per object and process;
* ``extent_prune`` (X1a): before the coarse ICP, a rotation is dropped when the object's extent under it (its hull
  vertices) along one of the declared directions is smaller than the target's observed extent along it (the
  ``extent_q`` .. ``1 - extent_q`` quantile range of the patch centres) minus the slack (``extent_slack_m`` +
  ``extent_slack_rel`` x the object's extent): the visible part of an object cannot exceed the object. Directions:
  the gravity axes and the patch centres' principal axes;
* ``normal_prune`` (X1a): a rotation is dropped when more than ``normal_max_unmatched`` of the observed patch normals
  (gravity frame, turned into the object frame by the rotation) fall in a direction cell that no node normal of the
  object comes within ``normal_tol_deg`` of (a per-object lookup table on a ``normal_cell_deg`` latitude-longitude
  grid, conservative by the cell radius). A rotation is never dropped by both tests to below ``min_keep`` rotations
  (the best ones by the normal test are kept);
* ``keep_after_first`` (X1b): after the first coarse ICP iteration only this share of the hypotheses (the best by
  the step's own score: capped fit of that iteration's query + ``LAMBDA_MM`` x the free space after its step) runs
  the remaining coarse iterations (0: all of them);
* ``n_sphere_downs``, ``coarse_yaw_deg``, ``n_seeds``, ``refine_offsets_deg`` (X2): a cheaper coarse set.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from rowant_lab.hybrid_frontier.screens.e0c_check_timing import proposal as P
from rowant_lab.settings import DATA

TRUTH_FREE = True
N_SPHERE_DOWNS = 48
COARSE_YAW_DEG = 30.0
REFINE_OFFSETS_DEG = (-10.0, -5.0, 5.0, 10.0)
N_ICP_COARSE = 64
N_ICP_FINE = 128
ICP_ITERS_COARSE = 3
ICP_ITERS_FINE = 2
TRIMS_M = (0.04, 0.025, 0.015, 0.015)
FIT_CAP_M = 0.01
LAMBDA_MM = 10.0
FS_TOL_M = 0.010
N_SEEDS = 6
NMS_SEED_DEG = 20.0
POSES = 5
NMS_FINAL_DEG = 10.0
NMS_FINAL_M = 0.01
PARAMS = {k: (list(v) if isinstance(v, tuple) else v) for k, v in dict(
    N_SPHERE_DOWNS=N_SPHERE_DOWNS, COARSE_YAW_DEG=COARSE_YAW_DEG, REFINE_OFFSETS_DEG=REFINE_OFFSETS_DEG,
    N_ICP_COARSE=N_ICP_COARSE, N_ICP_FINE=N_ICP_FINE, ICP_ITERS_COARSE=ICP_ITERS_COARSE,
    ICP_ITERS_FINE=ICP_ITERS_FINE, TRIMS_M=TRIMS_M, FIT_CAP_M=FIT_CAP_M, LAMBDA_MM=LAMBDA_MM, FS_TOL_M=FS_TOL_M,
    N_SEEDS=N_SEEDS, NMS_SEED_DEG=NMS_SEED_DEG, POSES=POSES, NMS_FINAL_DEG=NMS_FINAL_DEG,
    NMS_FINAL_M=NMS_FINAL_M).items()}
_SPHERE = P.fibonacci_sphere(N_SPHERE_DOWNS)
_SPHERE_R = np.stack([P.rotation_to(d, [0.0, 0.0, -1.0]) for d in _SPHERE])


@dataclass
class WideFrame:
    """Per-target inputs of the wide proposal (gravity frame: z up, from the table plane or the camera)."""

    R_g: np.ndarray  # gravity frame -> world rotation
    o_g: np.ndarray  # world position of the gravity-frame origin
    y: np.ndarray  # (P, 3) target patch centres, gravity frame
    cam_g: np.ndarray  # camera centre, gravity frame
    R_cw: np.ndarray  # world -> camera rotation
    c_w: np.ndarray  # camera centre, world
    intr: tuple  # fx, fy, cx, cy, width, height
    z_obs: np.ndarray  # (H, W) observed camera-z (m), NaN where there is no reading
    gravity: str  # "plane" or "camera_up"


def camera_up_frame(camera: dict) -> tuple[np.ndarray, np.ndarray]:
    """Fallback gravity frame: z = the camera's up (-y), origin at the world origin."""
    T = np.asarray(camera["T_world_from_camera"], float)
    z = T[:3, :3] @ np.array([0.0, -1.0, 0.0])
    x = T[:3, :3] @ np.array([1.0, 0.0, 0.0])
    return np.stack([x, np.cross(z, x), z], axis=1), np.zeros(3)


def make_frame(plane: dict | None, camera: dict, centres_world: np.ndarray, z_obs: np.ndarray) -> WideFrame:
    if plane is not None:
        R_g, o_g = P.gravity_frame(plane, camera)
        grav = "plane"
    else:
        R_g, o_g = camera_up_frame(camera)
        grav = "camera_up"
    T = np.asarray(camera["T_world_from_camera"], dtype=float)
    c_w = T[:3, 3].copy()
    return WideFrame(R_g=R_g, o_g=o_g, y=(np.asarray(centres_world) - o_g) @ R_g, cam_g=(c_w - o_g) @ R_g,
                     R_cw=T[:3, :3].T, c_w=c_w,
                     intr=(float(camera["fx"]), float(camera["fy"]), float(camera["cx"]), float(camera["cy"]),
                           int(camera["width"]), int(camera["height"])), z_obs=z_obs, gravity=grav)


@dataclass(frozen=True)
class WideOpts:
    """Step-4 speed options of the wide proposal (module doc). The defaults reproduce the step-3 proposal."""

    cache: bool = False  # E2
    extent_prune: bool = False  # X1a
    extent_q: float = 0.02
    extent_slack_m: float = 0.005
    extent_slack_rel: float = 0.10
    normal_prune: bool = False  # X1a
    normal_tol_deg: float = 35.0
    normal_cell_deg: float = 4.0
    normal_max_unmatched: float = 0.25
    min_keep: int = 24
    keep_after_first: float = 0.0  # X1b (0: every hypothesis runs every coarse iteration)
    grid_dir: str | None = None  # X6: the node grids (GRIDS_OF_MEMORY or a folder; checked by GridSet); None: k-d tree
    grid_h_mm: float = 2.5
    grid_k: int = 1
    n_sphere_downs: int = N_SPHERE_DOWNS  # X2
    coarse_yaw_deg: float = COARSE_YAW_DEG
    n_seeds: int = N_SEEDS
    refine_offsets_deg: tuple = REFINE_OFFSETS_DEG

    def record(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


DEFAULT_OPTS = WideOpts()
_SPHERE_CACHE: dict = {N_SPHERE_DOWNS: _SPHERE_R}


def sphere_rotations(n: int) -> np.ndarray:
    if n not in _SPHERE_CACHE:
        _SPHERE_CACHE[n] = np.stack([P.rotation_to(d, [0.0, 0.0, -1.0]) for d in P.fibonacci_sphere(n)])
    return _SPHERE_CACHE[n]


def rotations(shape: P.ObjectShape, opts: WideOpts = DEFAULT_OPTS) -> tuple[np.ndarray, list]:
    """Gravity-frame rotations (object-local -> gravity) of the coarse set and their labels."""
    rest = [c.R_rest for c in shape.classes]
    sph = sphere_rotations(int(opts.n_sphere_downs))
    downs = np.concatenate([np.stack(rest), sph]) if rest else sph
    src = [("rest", k) for k in range(len(rest))] + [("sphere", k) for k in range(len(sph))]
    yaws = np.arange(0.0, 360.0, float(opts.coarse_yaw_deg))
    Rz = P.rz_stack(yaws)
    Rp = np.concatenate([Rz @ Rd for Rd in downs], axis=0)
    labels = [(s, k, float(p)) for (s, k) in src for p in yaws]
    return Rp, labels


def initial_t(shape: P.ObjectShape, Rp: np.ndarray, fr: WideFrame) -> np.ndarray:
    """(H, 3): the camera-facing nodes' centroid on the centres' centroid (3D)."""
    RpT = np.transpose(Rp, (0, 2, 1))
    X = shape.vis_local[None] @ RpT
    N = shape.vis_normals[None] @ RpT
    ybar = fr.y.mean(axis=0)
    vdir = fr.cam_g - ybar
    vdir = vdir / np.linalg.norm(vdir)
    vis = (N @ vdir) > 0
    few = vis.sum(axis=1) < 3
    vis[few] = True
    w = vis.astype(float)
    c = (X * w[..., None]).sum(axis=1) / w.sum(axis=1)[:, None]
    return ybar[None, :] - c


def icp_t3(shape: P.ObjectShape, Rp: np.ndarray, t0: np.ndarray, y: np.ndarray, iters: int, *,
           final_query: bool = True, first_iter: int = 0,
           grid: "NodeGrid | None" = None) -> tuple[np.ndarray, np.ndarray]:
    """Trimmed point-to-point ICP on the translation only, in 3D (``proposal.icp_xy`` without the contact).
    ``first_iter``: the index of the first iteration run here (its trim), so the passes can be split (X1b);
    ``grid``: nearest nodes from the object's voxel grid (X6) instead of the k-d tree."""
    t = np.array(t0, dtype=float)
    d = None
    nn = P._nn if grid is None else grid.nn  # noqa: SLF001 (the screens' own helper)
    for it in range(first_iter, iters):
        d, gw = nn(shape, Rp, t, y)
        keep = d <= TRIMS_M[min(it, len(TRIMS_M) - 1)]
        cnt = keep.sum(axis=1)
        resid = gw - y[None]
        step = np.where(cnt[:, None] > 0, (resid * keep[..., None]).sum(axis=1) / np.maximum(cnt, 1)[:, None], 0.0)
        t -= step
    if final_query or d is None:
        d, _ = nn(shape, Rp, t, y)
    return t, d


# ------------------------------------------------------------------------------------------- step 4: node grid (X6)
GRID_FORMAT = "rowant fastlayer node grid v1"  # one object's grid: <name>.npy (indices) + <name>.json (meta)
GRID_INDEX_FORMAT = "rowant fastlayer node grids v1"  # a grid folder's index: grids_h{h}_k{k}.json
GRID_MARGIN_M = max(TRIMS_M)  # a point outside the grid is farther than every trim from every node
# fix:bop (B1). A preset names no grid folder: ``WideOpts.grid_dir = GRIDS_OF_MEMORY`` takes the folder registered
# here for the layer's memory (by the memory's sha256), or the folder in ``$ROWANT_FASTLAYER_GRID_DIR`` when that is
# set; any other value is a folder. Whatever the folder, ``GridSet`` checks its files against the memory before use.
GRIDS_OF_MEMORY = "of-memory"
GRID_DIR_ENV = "ROWANT_FASTLAYER_GRID_DIR"
_RUNS = str(DATA / "runs" / "bop-dev-scratch")
GRID_SETS = {  # memory sha256 -> its node grids (h 2.5 mm, k 1)
    "85652f215e87c82fd66309e763eef050a814b6feff8be5446478ae22f99d41ca":  # HOPE, 28 objects (step 4)
        f"{_RUNS}/fastlayer-step4/speed/grids",
    "49242d1339cc1e70158fb83ee4a1838108bd1e9eb129866c3e50667d5750c518":  # LM-O, 7 objects, no driller (step 5)
        f"{_RUNS}/fastlayer-step5/vertex-colour/lmo/grids",
    "357faa05c118e876978af60c366761c14b6b20604b57ce63b892e91dae103f2d":  # IC-BIN, 2 objects (step 5)
        f"{_RUNS}/fastlayer-step5/vertex-colour/icbin/grids",
}


class GridMismatch(ValueError):
    """A node grid (or grid folder) that does not belong to the memory it would serve."""


def _obj_key(g: str):
    return (0, int(g), g) if str(g).isdigit() else (1, 0, str(g))


def _sha256(path) -> str:
    import hashlib  # noqa: PLC0415

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def grid_files(base) -> tuple:
    """(.npy, .json) of the grid saved at ``base`` (e.g. ``DIR/grid_1_h2.5_k1``). The names keep h and k
    (``base.name + suffix``). Grids saved before fix:bop went through ``Path.with_suffix``, which cut a fractional
    h and the k (``grid_1_h2.npy``); those legacy names are used when the full ones are absent (their meta is checked
    for h and k all the same)."""
    from pathlib import Path  # noqa: PLC0415

    base = Path(base)
    full = (base.parent / f"{base.name}.npy", base.parent / f"{base.name}.json")
    if full[0].exists() or full[1].exists():
        return full
    legacy = (base.with_suffix(".npy"), base.with_suffix(".json"))
    return legacy if (legacy[0].exists() or legacy[1].exists()) else full


class NodeGrid:
    """X6: per object, a voxel grid (model frame, the k-d tree's coordinates) over the nodes' bounding box plus
    ``GRID_MARGIN_M``; each voxel holds the ``k`` nodes nearest its centre. ``nn`` takes the nearest of them to the
    query point (its exact distance, so never below the true nearest-node distance, and at most the voxel's
    diagonal above it); a point outside the grid gets an infinite distance (farther than every trim from every
    node, so the coarse pass trims it and caps its fit, as with the exact distance)."""

    def __init__(self, origin, h: float, dims, idx: np.ndarray, meta: dict | None = None):
        self.origin = np.asarray(origin, float)
        self.h = float(h)
        self.dims = tuple(int(x) for x in dims)
        self.idx = idx  # (nx * ny * nz, k) int16 / int32, possibly a read-only memory map
        self.meta = meta or {}

    @classmethod
    def build(cls, pos: np.ndarray, tree, h: float, k: int = 1) -> "NodeGrid":
        pos = np.asarray(pos, float)
        lo = pos.min(axis=0) - GRID_MARGIN_M - h
        hi = pos.max(axis=0) + GRID_MARGIN_M + h
        dims = np.ceil((hi - lo) / h).astype(int)
        ax = [lo[i] + (np.arange(dims[i]) + 0.5) * h for i in range(3)]
        X, Y, Z = np.meshgrid(*ax, indexing="ij")
        c = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        dt = np.int16 if len(pos) < 32767 else np.int32
        out = np.empty((len(c), int(k)), dtype=dt)
        for a in range(0, len(c), 1 << 18):
            _, ii = tree.query(c[a:a + (1 << 18)], k=int(k), workers=1)
            out[a:a + (1 << 18)] = ii.reshape(-1, int(k))
        return cls(lo, h, dims, out)

    def nn(self, shape: P.ObjectShape, Rp: np.ndarray, t: np.ndarray, y: np.ndarray):
        """``proposal._nn``'s contract: distances (H, n) and the matched nodes in the gravity frame (H, n, 3)."""
        H, n = len(Rp), len(y)
        local = ((y[None] - t[:, None, :]) @ Rp + P.P_TRAIN).reshape(-1, 3)
        ijk = np.floor((local - self.origin) / self.h).astype(np.int64)
        nx, ny, nz = self.dims
        inside = ((ijk >= 0) & (ijk < np.array(self.dims))).all(axis=1)
        flat = np.where(inside, (ijk[:, 0] * ny + ijk[:, 1]) * nz + ijk[:, 2], 0)
        cand = np.asarray(self.idx[flat], dtype=np.int64)  # (m, k)
        diff = shape.pos[cand] - local[:, None, :]
        dd = np.sqrt((diff * diff).sum(axis=2))
        best = np.argmin(dd, axis=1)
        rows = np.arange(len(local))
        j = cand[rows, best]
        d = np.where(inside, dd[rows, best], np.inf)
        g = shape.pos[j.reshape(H, n)] - P.P_TRAIN
        return d.reshape(H, n), g @ np.transpose(Rp, (0, 2, 1)) + t[:, None, :]

    def save(self, path) -> dict:
        """Write ``<path>.npy`` and ``<path>.json`` (the full name: ``grid_files``); never overwrites."""
        npy, js = grid_files_new(path)
        if npy.exists() or js.exists():
            raise FileExistsError(f"{npy} or {js} exists (grids are never overwritten)")
        np.save(npy, self.idx)
        meta = {"format": GRID_FORMAT, "origin": self.origin.tolist(), "h": self.h, "dims": list(self.dims),
                "k": int(self.idx.shape[1]), "dtype": str(self.idx.dtype), **self.meta}
        js.write_text(json.dumps(meta, sort_keys=True) + "\n")
        return meta

    @classmethod
    def load(cls, path, mmap: bool = True) -> "NodeGrid":
        """The grid saved at ``path`` (full or legacy names, ``grid_files``). Checks the format only: a layer loads
        its grids through ``GridSet``, which checks them against the memory."""
        npy, js = grid_files(path)
        meta = json.loads(js.read_text())
        if meta.get("format") != GRID_FORMAT:
            raise ValueError(f"{js}: not a {GRID_FORMAT!r} file")
        idx = np.load(npy, mmap_mode="r" if mmap else None)
        return cls(meta["origin"], meta["h"], meta["dims"], idx, meta)


def grid_files_new(base) -> tuple:
    """(.npy, .json) under the full name (h and k kept), for a new grid."""
    from pathlib import Path  # noqa: PLC0415

    base = Path(base)
    return base.parent / f"{base.name}.npy", base.parent / f"{base.name}.json"


def grid_name(obj: str, h_mm: float, k: int) -> str:
    return f"grid_{obj}_h{h_mm:g}_k{int(k)}"


def grid_index_name(h_mm: float, k: int) -> str:
    return f"grids_h{h_mm:g}_k{int(k)}.json"


def build_grid_set(shapes: dict, memory_sha256: str, out_dir, h_mm: float, k: int, ids=None, extra: dict | None = None,
                   log=None) -> dict:
    """Build and save the node grids of a memory's objects (``shapes``: the memory's ``ObjectShape`` per object id;
    ``ids``: the objects to grid, default all) into ``out_dir``, with the folder index. Each grid file and the index
    carry the memory's sha256 and its object set (every object id of ``shapes``), so ``GridSet`` can check them
    against the memory that loads them. Never overwrites. ``extra``: more fields for the index. Returns the index."""
    import time  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / grid_index_name(h_mm, k)
    if index_path.exists():
        raise FileExistsError(f"{index_path} exists (grids are never overwritten)")
    object_set = sorted((str(g) for g in shapes), key=_obj_key)
    ids = object_set if ids is None else sorted((str(g) for g in ids), key=_obj_key)
    missing = sorted(set(ids) - set(object_set), key=_obj_key)
    if missing:
        raise ValueError(f"no shapes for {missing}")
    rows = {}
    t_all = time.perf_counter()
    for g in ids:
        base = out / grid_name(g, h_mm, k)
        sh = shapes[g]
        t0 = time.perf_counter()
        grid = NodeGrid.build(sh.pos, sh.tree, h_mm * 1e-3, k)
        grid.meta = {"obj": g, "memory_sha256": memory_sha256, "object_set": object_set, "n_nodes": int(len(sh.pos)),
                     "seconds": round(time.perf_counter() - t0, 3)}
        meta = grid.save(base)
        rows[g] = {"dims": meta["dims"], "seconds": meta["seconds"], "mb": round(grid.idx.nbytes / 1e6, 3),
                   "npy_sha256": _sha256(grid_files_new(base)[0])}
        if log is not None:
            log(g, rows[g])
    rec = {"format": GRID_INDEX_FORMAT, "truth_free": True, "h_mm": h_mm, "k": int(k), "margin_m": GRID_MARGIN_M,
           "memory_sha256": memory_sha256, "object_set": object_set, "objects": rows,
           "total_mb": round(sum(r["mb"] for r in rows.values()), 3), "wall_s": round(time.perf_counter() - t_all, 2),
           **(extra or {})}
    with open(index_path, "x", encoding="utf-8") as f:
        f.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


class GridSet:
    """The node grids of one memory, checked against it (fix:bop, B1).

    ``folder`` holds the folder index (``grids_h{h}_k{k}.json``) and one grid per object. The index must have the
    grid index format, the memory's sha256, the asked h and k, and an object set (its ``object_set``, or for grids
    saved before fix:bop the ids of its ``objects`` table) that holds every object of the memory. When an object's
    grid is first used (``get``), its meta must name that object, the memory's sha256, the same object set (when it
    carries one), the shape's node count, h and k; its index array must have the meta's shape; and its .npy must have
    the sha256 the index records. Anything else raises ``GridMismatch``: a grid of another memory (another
    dataset's objects with the same ids) is never used. Grids are memory-mapped and kept per GridSet."""

    def __init__(self, folder, memory_sha256: str, object_ids, h_mm: float, k: int):
        from pathlib import Path  # noqa: PLC0415

        self.folder = Path(folder)
        self.memory_sha256 = str(memory_sha256)
        self.h_mm, self.k = float(h_mm), int(k)
        self.index_path = self.folder / grid_index_name(h_mm, k)
        try:
            idx = json.loads(self.index_path.read_text())
        except (OSError, ValueError) as e:
            raise GridMismatch(f"{self.index_path}: no readable grid index ({type(e).__name__})") from None
        self.index = idx
        bad = []
        if idx.get("format") != GRID_INDEX_FORMAT:
            bad.append(f"format {idx.get('format')!r}")
        if idx.get("memory_sha256") != self.memory_sha256:
            bad.append(f"memory sha256 {str(idx.get('memory_sha256'))[:12]}... (the memory's is "
                       f"{self.memory_sha256[:12]}...)")
        if not (isinstance(idx.get("h_mm"), (int, float)) and abs(float(idx["h_mm"]) - self.h_mm) < 1e-9) \
                or idx.get("k") != self.k:
            bad.append(f"h {idx.get('h_mm')} mm, k {idx.get('k')} (asked {self.h_mm:g} mm, {self.k})")
        rows = idx.get("objects") if isinstance(idx.get("objects"), dict) else {}
        oset = idx.get("object_set", sorted(rows, key=_obj_key))
        if "object_set" in idx and not set(rows) <= set(oset):
            bad.append("objects outside its object set")
        self.object_set = frozenset(str(g) for g in oset)
        missing = sorted({str(g) for g in object_ids} - self.object_set, key=_obj_key)
        if missing:
            bad.append(f"object set {sorted(self.object_set, key=_obj_key)} lacks the memory's {missing}")
        if bad:
            raise GridMismatch(f"{self.index_path} is not the grid set of this memory: " + "; ".join(bad))
        self._rows = rows
        self._grids: dict = {}

    @classmethod
    def resolve(cls, grid_dir: str, memory_sha256: str, object_ids, h_mm: float, k: int) -> "GridSet":
        """``grid_dir`` as ``WideOpts`` holds it: ``GRIDS_OF_MEMORY`` (the folder of ``$ROWANT_FASTLAYER_GRID_DIR``,
        else the one ``GRID_SETS`` registers for the memory) or a folder."""
        folder = grid_dir
        if grid_dir == GRIDS_OF_MEMORY:
            folder = os.environ.get(GRID_DIR_ENV) or GRID_SETS.get(str(memory_sha256))
            if not folder:
                raise GridMismatch(f"no node grids are registered for memory {str(memory_sha256)[:12]}... "
                                   "(wide.GRID_SETS): build them (wide.build_grid_set) and register the folder, or "
                                   "give speed.wide.grid_dir")
        return cls(folder, memory_sha256, object_ids, h_mm, k)

    def record(self) -> dict:
        return {"folder": str(self.folder), "index_sha256": _sha256(self.index_path), "h_mm": self.h_mm,
                "k": self.k, "memory_sha256": self.memory_sha256}

    def get(self, shape: P.ObjectShape) -> NodeGrid:
        g = str(shape.obj)
        if g in self._grids:
            return self._grids[g]
        where = f"{self.folder}: grid of object {g}"
        row = self._rows.get(g)
        if g not in self.object_set or not isinstance(row, dict):
            raise GridMismatch(f"{where}: the grid index has no grid for it")
        base = self.folder / grid_name(g, self.h_mm, self.k)
        npy, js = grid_files(base)
        try:
            grid = NodeGrid.load(base)
        except (OSError, ValueError) as e:
            raise GridMismatch(f"{where}: unreadable ({type(e).__name__}: {e})") from None
        m = grid.meta
        bad = []
        if str(m.get("obj")) != g:
            bad.append(f"names object {m.get('obj')!r}")
        if m.get("memory_sha256") != self.memory_sha256:
            bad.append(f"memory sha256 {str(m.get('memory_sha256'))[:12]}...")
        if "object_set" in m and frozenset(str(x) for x in m["object_set"]) != self.object_set:
            bad.append("another object set")
        if m.get("n_nodes") != len(shape.pos):
            bad.append(f"{m.get('n_nodes')} nodes (the memory's object has {len(shape.pos)})")
        if abs(float(m.get("h", -1.0)) - self.h_mm * 1e-3) > 1e-12 or m.get("k") != self.k:
            bad.append(f"h {m.get('h')} m, k {m.get('k')}")
        if tuple(grid.idx.shape) != (int(np.prod(grid.dims)), self.k):
            bad.append(f"index array {tuple(grid.idx.shape)} for dims {grid.dims}")
        if row.get("npy_sha256") != _sha256(npy):
            bad.append(f"{npy.name} sha256 differs from the index's")
        if bad:
            raise GridMismatch(f"{where} does not belong to this memory: " + "; ".join(bad))
        self._grids[g] = grid
        return grid


def free_space(shape: P.ObjectShape, Rp: np.ndarray, t: np.ndarray, fr: WideFrame) -> np.ndarray:
    """(H,) clutter free-space share (module doc)."""
    RpT = np.transpose(Rp, (0, 2, 1))
    Xg = shape.vis_local[None] @ RpT + t[:, None, :]
    Ng = shape.vis_normals[None] @ RpT
    facing = np.einsum("hni,hni->hn", Ng, fr.cam_g[None, None, :] - Xg) > 0
    Xw = Xg @ fr.R_g.T + fr.o_g
    Xc = (Xw - fr.c_w) @ fr.R_cw.T
    fx, fy, cx, cy, W, Hh = fr.intr
    z = Xc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.rint(fx * Xc[..., 0] / z + cx)
        v = np.rint(fy * Xc[..., 1] / z + cy)
    inside = facing & (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < Hh)
    ui = np.where(inside, u, 0).astype(np.int64)
    vi = np.where(inside, v, 0).astype(np.int64)
    zo = fr.z_obs[vi, ui]
    known = inside & np.isfinite(zo)
    viol = known & (z < zo - FS_TOL_M)
    return viol.sum(axis=1) / np.maximum(known.sum(axis=1), 1)


def _score(d: np.ndarray, fs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fit_mm = np.minimum(d, FIT_CAP_M).mean(axis=1) * 1e3
    return fit_mm, fit_mm + LAMBDA_MM * fs


# ------------------------------------------------------------------------------------------- step 4: pruning (X1a)
@dataclass
class ObjectTables:
    """Per-object tables of the step-4 options, built once per object and process."""

    rot: dict = field(default_factory=dict)  # (n_downs, yaw) -> (Rp, labels)
    normal_cover: dict = field(default_factory=dict)  # (tol, cell) -> (n_theta, n_phi) bool


def tables(shape: P.ObjectShape, cache: dict | None) -> ObjectTables:
    if cache is None:
        return ObjectTables()
    tab = cache.get(shape.obj)
    if tab is None:
        tab = cache[shape.obj] = ObjectTables()
    return tab


def _cells(d: np.ndarray, cell_deg: float) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Latitude-longitude cell of unit directions ``d`` (..., 3)."""
    n_t = int(round(180.0 / cell_deg))
    n_p = int(round(360.0 / cell_deg))
    th = np.arccos(np.clip(d[..., 2], -1.0, 1.0))
    ph = np.arctan2(d[..., 1], d[..., 0]) % (2 * np.pi)
    it = np.minimum((th / np.pi * n_t).astype(np.int64), n_t - 1)
    ip = np.minimum((ph / (2 * np.pi) * n_p).astype(np.int64), n_p - 1)
    return it, ip, n_t, n_p


def normal_cover(normals_local: np.ndarray, tol_deg: float, cell_deg: float) -> np.ndarray:
    """(n_theta, n_phi) bool: a cell is covered when some node normal lies within ``tol_deg`` + the cell's radius of
    its centre (so every direction within ``tol_deg`` of a node normal lies in a covered cell)."""
    n_t = int(round(180.0 / cell_deg))
    n_p = int(round(360.0 / cell_deg))
    th = (np.arange(n_t) + 0.5) * np.pi / n_t
    ph = (np.arange(n_p) + 0.5) * 2 * np.pi / n_p
    T, Ph = np.meshgrid(th, ph, indexing="ij")
    c = np.stack([np.sin(T) * np.cos(Ph), np.sin(T) * np.sin(Ph), np.cos(T)], axis=-1).reshape(-1, 3)
    # cell radius: half the cell's diagonal on the sphere (at most), in degrees
    rad = np.degrees(np.sqrt((np.pi / n_t / 2) ** 2 + (np.sin(T.reshape(-1)) * np.pi / n_p) ** 2))
    lim = np.cos(np.radians(np.minimum(tol_deg + rad, 180.0)))
    nn = np.asarray(normals_local, float)
    nn = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
    best = np.full(len(c), -2.0)
    for a in range(0, len(nn), 4096):
        best = np.maximum(best, (c @ nn[a:a + 4096].T).max(axis=1))
    return (best >= lim).reshape(n_t, n_p)


def prune_rotations(shape: P.ObjectShape, Rp: np.ndarray, fr: WideFrame, normals_g: np.ndarray | None,
                    opts: WideOpts, tab: ObjectTables,
                    node_normals: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """X1a: the indices of the coarse rotations kept (module doc) and the counts. ``node_normals``: every node normal
    of the object (default: the 256-node subsample's)."""
    H = len(Rp)
    keep = np.ones(H, bool)
    info = {"n_rot": int(H)}
    if opts.extent_prune:
        y = fr.y
        yc = y - y.mean(axis=0)
        _, vec = np.linalg.eigh(yc.T @ yc)
        dirs = np.concatenate([np.eye(3), vec.T], axis=0)  # gravity axes + principal axes of the centres
        proj = y @ dirs.T
        q = float(opts.extent_q)
        obs = np.quantile(proj, 1.0 - q, axis=0) - np.quantile(proj, q, axis=0)  # (D,)
        # object extent under each rotation along each direction: hull vertices placed by Rp
        loc_dirs = np.einsum("hji,dj->hdi", Rp, dirs)  # Rp^T e, (H, D, 3)
        s = np.einsum("vi,hdi->hdv", shape.hull_local, loc_dirs)
        ext = s.max(axis=2) - s.min(axis=2)
        ok = (obs[None, :] <= ext * (1.0 + opts.extent_slack_rel) + opts.extent_slack_m).all(axis=1)
        info["n_extent_dropped"] = int((~ok).sum())
        keep &= ok
    if opts.normal_prune and normals_g is not None and len(normals_g):
        key = (float(opts.normal_tol_deg), float(opts.normal_cell_deg))
        cov = tab.normal_cover.get(key)
        if cov is None:
            cov = tab.normal_cover[key] = normal_cover(shape.vis_normals if node_normals is None else node_normals,
                                                        *key)
        nl = np.einsum("hji,pj->hpi", Rp, normals_g)  # Rp^T n, (H, P, 3)
        it, ip, _, _ = _cells(nl, float(opts.normal_cell_deg))
        unmatched = (~cov[it, ip]).mean(axis=1)
        ok = unmatched <= opts.normal_max_unmatched
        info["n_normal_dropped"] = int((~ok).sum())
        info["unmatched_p50"] = float(np.median(unmatched))
        keep &= ok
        if keep.sum() < opts.min_keep:
            order = np.lexsort((np.arange(H), unmatched))
            keep[order[:min(opts.min_keep, H)]] = True
            info["min_keep_applied"] = True
    if keep.sum() == 0:
        keep[:] = True
        info["all_dropped_kept_all"] = True
    info["n_kept"] = int(keep.sum())
    return np.flatnonzero(keep), info


def propose(shape: P.ObjectShape, fr: WideFrame, *, poses: int = POSES, keep_all: bool = False,
            opts: WideOpts = DEFAULT_OPTS, cache: dict | None = None, normals_g: np.ndarray | None = None,
            node_normals: np.ndarray | None = None, grids: "GridSet | None" = None) -> dict:
    """The wide proposal for one object (module doc). Guesses carry WORLD-frame ``R`` (object-local -> world), ``t``.
    Step 4: ``opts`` (``WideOpts``), ``cache`` (a per-process dict for the per-object tables) and ``normals_g`` (the
    target's patch normals, gravity frame, for ``normal_prune``; ``node_normals``: every node normal of the object).
    ``grids`` (fix:bop): with ``opts.grid_dir`` set, the memory's checked node grids (``GridSet``; ``FastLayer``
    binds them to its memory); a grid is never loaded here unchecked."""
    if len(shape.vis_local) != len(shape.fs_local) or not np.array_equal(shape.vis_local, shape.fs_local):
        raise ValueError("expected the screens' shared 256-node subsample for visibility and free space")
    if opts.grid_dir and grids is None:
        raise ValueError("WideOpts.grid_dir is set: pass the memory's checked node grids (grids=GridSet ...)")
    ycoarse = fr.y[P._stride(len(fr.y), N_ICP_COARSE)]  # noqa: SLF001
    yfine = fr.y[P._stride(len(fr.y), N_ICP_FINE)]  # noqa: SLF001
    tab = tables(shape, cache if opts.cache or opts.normal_prune or opts.grid_dir else None)
    rkey = (int(opts.n_sphere_downs), float(opts.coarse_yaw_deg))
    if opts.cache and rkey in tab.rot:
        Rp, labels = tab.rot[rkey]
    else:
        Rp, labels = rotations(shape, opts)
        if opts.cache:
            tab.rot[rkey] = (Rp, labels)
    prune = {}
    if opts.extent_prune or opts.normal_prune:
        kept, prune = prune_rotations(shape, Rp, fr, normals_g, opts, tab, node_normals)
        if len(kept) < len(Rp):
            Rp = Rp[kept]
            labels = [labels[i] for i in kept]
    t0 = initial_t(shape, Rp, fr)
    n_coarse = len(Rp)
    grid = grids.get(shape) if opts.grid_dir else None
    if opts.keep_after_first and 0 < opts.keep_after_first < 1 and ICP_ITERS_COARSE > 1:
        t, d = icp_t3(shape, Rp, t0, ycoarse, 1, final_query=False, grid=grid)
        fs = free_space(shape, Rp, t, fr)
        _, sc1 = _score(d, fs)
        m = max(int(np.ceil(opts.keep_after_first * len(Rp))), min(len(Rp), int(opts.n_seeds) * 4))
        sel = np.sort(np.argsort(sc1, kind="stable")[:m])
        Rp, t = Rp[sel], t[sel]
        labels = [labels[i] for i in sel]
        prune["n_after_first"] = int(len(sel))
        t, d = icp_t3(shape, Rp, t, ycoarse, ICP_ITERS_COARSE, final_query=False, first_iter=1, grid=grid)
    else:
        t, d = icp_t3(shape, Rp, t0, ycoarse, ICP_ITERS_COARSE, final_query=False, grid=grid)
    fs = free_space(shape, Rp, t, fr)
    _, sc = _score(d, fs)
    seeds = P._nms(np.argsort(sc, kind="stable"), Rp, t, int(opts.n_seeds), NMS_SEED_DEG, None)  # noqa: SLF001
    R2, t2, lab2 = [], [], []
    for s in seeds:
        src, k, psi = labels[s]
        for off in (0.0,) + tuple(opts.refine_offsets_deg):
            R2.append(P.rz(off) @ Rp[s])
            t2.append(t[s])
            lab2.append((src, k, (psi + off) % 360.0, float(sc[s])))
    R2 = np.stack(R2)
    t2 = np.stack(t2)
    t2, d2 = icp_t3(shape, R2, t2, yfine, ICP_ITERS_FINE)
    fs2 = free_space(shape, R2, t2, fr)
    fit2, sc2 = _score(d2, fs2)
    order = np.argsort(sc2, kind="stable")
    Rw = fr.R_g[None] @ R2
    tw = t2 @ fr.R_g.T + fr.o_g
    pick = P._nms(order, Rw, tw, poses, NMS_FINAL_DEG, NMS_FINAL_M)  # noqa: SLF001

    def rec(i):
        src, k, psi, coarse = lab2[i]
        out = {"obj": shape.obj, "R": Rw[i], "t": tw[i], "source": src, "down": int(k), "psi_deg": float(psi),
               "fit_mm": float(fit2[i]), "free_space": float(fs2[i]), "geo_score": float(sc2[i]),
               "coarse_score": coarse}
        if src == "rest":
            out["class_share"] = float(shape.classes[k].share)
        return out

    out = {"guesses": [rec(i) for i in pick], "n_hyp_coarse": int(n_coarse), "n_hyp_fine": int(len(R2)),
           "gravity": fr.gravity}
    if prune:
        out["prune"] = prune
    if keep_all:
        out["all"] = [rec(i) for i in order]
    return out
