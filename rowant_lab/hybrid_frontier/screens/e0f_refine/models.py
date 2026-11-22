"""TRUTH-FREE: the model point sets the refinement registers the frame's depth to. Never a mesh.

Two sources, both from the robot's own sensing during learning (no YCB mesh, no render key, no case record):

* **U77 learned graph** (primary): Monty's U77-s42 graph nodes and their stored surface normals
  (``pose_vectors[0:3]``), from screen E-0d's sha-pinned export (``U77-s42.npz``, the same file E-0c's CHECK reads).
  Model frame = object-local + p_train.
* **Fused view-finder cloud** (variant, direction note section 3.4): the truth-free 1 mm sensor prior of F1-77-s42
  (``priors/kpdev/sensor-F1-77-s42/<obj>.npz``, PriorShape v1, ``truth_used: false``; written by
  ``prior/sensor_cloud.py`` from ``bench-v3/F1-77fused-s42-merged/pretrained/fused_vf.npz``; model frame, checked
  there). Its stored ``normals`` are view hints (mean direction toward the camera that saw the voxel), not surface
  normals, so here the 1 mm points are re-gridded to ``FUSED_GRID_M`` voxel centroids and each centroid gets a PCA
  normal from the 1 mm points within ``FUSED_NORMAL_RADIUS_M``, turned to agree with the view hint (outward). Points
  with too few neighbours or no planar neighbourhood (``FUSED_MAX_PLANARITY``) are dropped. Every prior file's sha256
  is checked against the pinned manifest.

Both loaders call ``e9/isolation.require`` before reading a memory and refuse a sha mismatch.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from rowant_lab.hybrid_frontier.contracts import prior_shape_v1 as ps
from rowant_lab.hybrid_frontier.e9 import isolation
from rowant_lab.hybrid_frontier.screens.e0d_feature_transfer import graphs as e0d_graphs
from rowant_lab.hybrid_frontier.screens.e0f_refine import spec as S
from rowant_lab.settings import DATA

TRUTH_FREE = True
H = DATA / "runs" / "hybrid-frontier"
U77_NPZ = H / "screens-2026-10-17" / "e0d-feature-transfer" / "graphs" / "U77-s42.npz"
U77_NPZ_SHA256 = "3d138ee28b7a5286a59cf05b79f71c0070b47e89a820f3e01b54583365ecbf65"
FUSED_DIR = H / "priors" / "kpdev" / "sensor-F1-77-s42"
FUSED_MANIFEST_SHA256 = "6bc26567a307acaac5f4a6d659f6e1cdf418624980b2516a2a456d870c86fcd3"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class ModelCloud:
    """One object's model points (model frame), unit outward normals and KD tree."""

    obj: str
    pos: np.ndarray  # (M, 3)
    nrm: np.ndarray  # (M, 3)
    source: str
    tree: cKDTree = field(repr=False, default=None)
    diag: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pos = np.ascontiguousarray(self.pos, dtype=np.float64)
        n = np.asarray(self.nrm, dtype=np.float64)
        self.nrm = np.ascontiguousarray(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300))
        if self.tree is None:
            self.tree = cKDTree(self.pos)

    @property
    def n(self) -> int:
        return int(len(self.pos))


def outward_share(pos: np.ndarray, nrm: np.ndarray) -> float:
    """Share of points whose normal points away from the cloud's centroid (a coarse orientation diagnostic)."""
    return float(np.mean(np.einsum("ij,ij->i", pos - pos.mean(axis=0), nrm) > 0))


# ------------------------------------------------------------------------------------------- U77 graph
def u77_from_arrays(pos: dict, x: dict, feature_mapping: dict) -> dict[str, ModelCloud]:
    pv = int(feature_mapping["pose_vectors"][0])
    out = {}
    for g in sorted(pos):
        p = np.asarray(pos[g], float)
        n = np.asarray(x[g], float)[:, pv:pv + 3]
        ok = np.isfinite(n).all(axis=1) & (np.linalg.norm(n, axis=1) > 0.5)
        out[g] = ModelCloud(g, p[ok], n[ok], "u77_graph",
                            diag={"n_nodes": int(len(p)), "n_kept": int(ok.sum()),
                                  "outward_share": outward_share(p[ok], n[ok])})
    return out


def load_u77(npz=U77_NPZ, expect_sha: str | None = U77_NPZ_SHA256,
             guard: Callable[[], dict] = isolation.require) -> tuple[dict[str, ModelCloud], dict]:
    iso = guard()
    t0 = time.perf_counter()
    sha = sha256_file(npz)
    if expect_sha is not None and sha != expect_sha:
        raise ValueError(f"{npz}: sha256 {sha} != pinned {expect_sha}")
    pos, x, meta = e0d_graphs.load(npz)
    models = u77_from_arrays(pos, x, meta["feature_mapping"])
    return models, {"npz": str(npz), "sha256": sha, "isolation": iso, "load_s": round(time.perf_counter() - t0, 3),
                    "n_objects": len(models), "n_points": int(sum(m.n for m in models.values()))}


# ------------------------------------------------------------------------------------------- fused cloud
def grid_centroids(points: np.ndarray, grid: float) -> tuple[np.ndarray, np.ndarray]:
    """Voxel centroids of ``points`` on a ``grid`` lattice, and each input point's voxel index."""
    key = np.floor(np.asarray(points) / grid).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    acc = np.zeros((len(cnt), 3))
    np.add.at(acc, inv, points)
    return acc / cnt[:, None], inv


def pca_normals(query: np.ndarray, cloud: np.ndarray, tree: cKDTree, hint: np.ndarray, *, radius: float,
                min_points: int, max_planarity: float, k: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """PCA normals at ``query`` from the ``cloud`` points within ``radius`` (at most ``k``), oriented to agree with
    ``hint``; returns ``(normals, ok)``."""
    d, idx = tree.query(query, k=min(k, len(cloud)))
    d = np.atleast_2d(d)
    idx = np.atleast_2d(idx)
    w = (d <= radius) & np.isfinite(d)
    cnt = w.sum(axis=1)
    P = cloud[np.minimum(idx, len(cloud) - 1)]
    wf = w.astype(float)[..., None]
    mean = (P * wf).sum(axis=1) / np.maximum(cnt, 1)[:, None]
    Q = (P - mean[:, None, :]) * wf
    C = np.einsum("nki,nkj->nij", Q, Q) / np.maximum(cnt, 1)[:, None, None]
    evals, evecs = np.linalg.eigh(C)
    n = evecs[:, :, 0].copy()
    flip = np.einsum("ij,ij->i", n, hint) < 0
    n[flip] *= -1
    with np.errstate(divide="ignore", invalid="ignore"):
        planarity = evals[:, 0] / np.maximum(evals[:, 1], 1e-30)
    ok = (cnt >= min_points) & (planarity <= max_planarity) & np.isfinite(n).all(axis=1)
    return n, ok


def fused_from_prior(obj: str, points: np.ndarray, view_hint: np.ndarray) -> ModelCloud:
    pts = np.asarray(points, float)
    hint = np.asarray(view_hint, float)
    cen, inv = grid_centroids(pts, S.FUSED_GRID_M)
    hsum = np.zeros((len(cen), 3))
    np.add.at(hsum, inv, hint)
    tree = cKDTree(pts)
    n, ok = pca_normals(cen, pts, tree, hsum, radius=S.FUSED_NORMAL_RADIUS_M, min_points=S.FUSED_NORMAL_MIN_POINTS,
                        max_planarity=S.FUSED_MAX_PLANARITY)
    return ModelCloud(obj, cen[ok], n[ok], "fused_vf_sensor_prior",
                      diag={"n_prior_points": int(len(pts)), "n_grid": int(len(cen)), "n_kept": int(ok.sum()),
                            "outward_share": outward_share(cen[ok], n[ok])})


def load_fused(objects=None, fused_dir=FUSED_DIR, expect_manifest_sha: str | None = FUSED_MANIFEST_SHA256,
               guard: Callable[[], dict] = isolation.require) -> tuple[dict[str, ModelCloud], dict]:
    iso = guard()
    t0 = time.perf_counter()
    fused_dir = Path(fused_dir)
    msha = sha256_file(fused_dir / "manifest.json")
    if expect_manifest_sha is not None and msha != expect_manifest_sha:
        raise ValueError(f"{fused_dir}/manifest.json: sha256 {msha} != pinned {expect_manifest_sha}")
    manifest = json.loads((fused_dir / "manifest.json").read_text())
    names = sorted(manifest["objects"]) if objects is None else sorted(objects)
    out = {}
    for g in names:
        rec = manifest["objects"][g]
        path = fused_dir / rec["prior_npz"]
        if sha256_file(path) != rec["prior_sha256"]:
            raise ValueError(f"{path}: sha256 does not match the manifest")
        shape = ps.read(path, allow_dense=bool(rec.get("allow_dense")))
        if shape.meta["normals_kind"] != "view_hint" or shape.meta["frame"]["kind"] != "model":
            raise ValueError(f"{path}: expected model-frame points with view-hint normals")
        out[g] = fused_from_prior(g, shape.points, shape.normals)
    return out, {"dir": str(fused_dir), "manifest_sha256": msha, "isolation": iso,
                 "load_s": round(time.perf_counter() - t0, 3), "n_objects": len(out),
                 "n_points": int(sum(m.n for m in out.values()))}
