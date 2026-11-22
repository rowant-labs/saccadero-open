"""TRUTH-FREE: a graph memory as the fast layer uses it (CHECK kernel, proposal shapes, refinement clouds).

A memory is an E-0d graph npz (``screens/e0d_feature_transfer/graphs``): U77's export (the screens' memory) or a
mesh memory built by ``onboard.py``. ``Memory.load`` builds, once per memory (setup, timed, excluded from per-frame
times):

* the CHECK kernel: ``e0c_check_timing.kernel.CheckKernel`` with feature set ``lnh`` (location, normal, hsv) and one
  cKDTree per object, exactly as ``e0c_check_timing.pipeline.load_resources`` builds it;
* the proposal shapes: ``e0c_check_timing.proposal.object_shape`` (rest classes from the graph's convex hull). For a
  mesh memory, the extra "learning-pose" class that ``rest_classes`` adds (local -y down: the way Monty's world A
  held an object while learning) is dropped: a BOP mesh was never learned in that pose. Classes that merely
  coincide with a support basin keep it;
* the refinement clouds: ``e0f_refine.models.u77_from_arrays`` (graph nodes and stored normals).

``e9/isolation.require`` runs before a memory is read (the screens' rule for model loads).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from rowant_lab.hybrid_frontier.e9 import isolation
from rowant_lab.hybrid_frontier.screens.e0c_check_timing import kernel, proposal
from rowant_lab.hybrid_frontier.screens.e0d_feature_transfer import graphs as e0d_graphs
from rowant_lab.hybrid_frontier.screens.e0f_refine import models as e0f_models

TRUTH_FREE = True
FEATURE_SET = "lnh"


@dataclass
class Memory:
    pos: dict
    x: dict
    feature_mapping: dict
    meta: dict
    kernel: kernel.CheckKernel
    shapes: dict
    clouds: dict
    npz_sha256: str
    kind: str
    setup_s: dict = field(default_factory=dict)
    isolation: dict = field(default_factory=dict)

    @property
    def ids(self) -> list[str]:
        return sorted(self.pos)


def mesh_shape(obj: str, pos: np.ndarray, normals: np.ndarray, tree) -> proposal.ObjectShape:
    """``proposal.object_shape`` without the learning-pose-only class (module doc)."""
    s = proposal.object_shape(obj, pos, normals, tree)
    kept = [c for c in s.classes if c.source != "learning-pose"]
    diag = dict(s.diag, n_classes=len(kept), dropped_learning_pose=len(s.classes) - len(kept),
                share_kept=float(sum(c.share for c in kept)))
    return replace(s, classes=kept, diag=diag)


def load(npz, *, kind: str, expect_sha256: str | None = None, ids=None, workers: int = 1,
         guard: Callable[[], dict] = isolation.require) -> Memory:
    """``kind``: ``"u77"`` (screens memory: ``object_shape`` unchanged) or ``"mesh"`` (``mesh_shape``)."""
    if kind not in ("u77", "mesh"):
        raise ValueError(f"unknown memory kind {kind!r}")
    iso = guard()
    t0 = time.perf_counter()
    sha = e0d_graphs.sha256_file(npz)
    if expect_sha256 is not None and sha != expect_sha256:
        raise ValueError(f"{npz}: sha256 {sha} != pinned {expect_sha256}")
    pos, x, meta = e0d_graphs.load(npz)
    if ids is not None:
        keep = [str(i) for i in ids]
        missing = sorted(set(keep) - set(pos))
        if missing:
            raise ValueError(f"{npz}: no graphs for {missing}")
        pos = {g: pos[g] for g in keep}
        x = {g: x[g] for g in keep}
    t1 = time.perf_counter()
    ker = kernel.CheckKernel(pos, x, meta["feature_mapping"], FEATURE_SET, workers=workers)
    t2 = time.perf_counter()
    make = proposal.object_shape if kind == "u77" else mesh_shape
    shapes = {g: make(g, pos[g], ker.graphs[g].normal, ker.graphs[g].tree) for g in sorted(pos)}
    t3 = time.perf_counter()
    clouds = e0f_models.u77_from_arrays(pos, x, meta["feature_mapping"])
    t4 = time.perf_counter()
    per_obj = len(pos) or 1
    return Memory(pos=pos, x=x, feature_mapping=meta["feature_mapping"], meta=meta, kernel=ker, shapes=shapes,
                  clouds=clouds, npz_sha256=sha, kind=kind,
                  setup_s={"load_npz_s": round(t1 - t0, 4), "kd_trees_s": round(t2 - t1, 4),
                           "rest_classes_s": round(t3 - t2, 4), "refine_clouds_s": round(t4 - t3, 4),
                           "per_object_s": round((t4 - t0) / per_obj, 4)},
                  isolation=iso)


def shape_record(shape: proposal.ObjectShape) -> dict:
    return {"n_classes": len(shape.classes), "diag": shape.diag,
            "classes": [{"down": [round(float(v), 6) for v in c.down], "share": round(float(c.share), 6),
                         "source": c.source} for c in shape.classes]}


def path_of(p) -> Path:
    return Path(p).expanduser()
