"""``PriorShape`` v1: one interface for every prior surface source, stored as an npz.

Keys of the npz (exactly these four):

* ``points`` (N, 3) float64, metres;
* ``normals`` (N, 3) float64, unit length within ``NORMAL_TOL``; their meaning is given by
  ``meta.normals_kind``: ``"outward"`` (surface normals pointing out of the object, e.g. from a
  mesh or a rendered program) or ``"view_hint"`` (the direction toward the camera that saw the
  point, as the view-finder cloud records it; a same-side hint, not a fitted normal);
* ``part_id`` (N,) int16, a part label per point (0 when the source has no parts);
* ``meta``: a JSON string (0-d array) with exactly the keys ``schema``
  ("rowant.hybrid_frontier.prior_shape"), ``version`` (1), ``object``, ``frame`` {``kind``
  "model" | "prior", ``T_prior_to_model`` 4x4 or null}, ``provenance`` {``source``,
  ``generator_sha256``, ``scale``, ``note``}, ``normals_kind``.

Frames: ``"model"`` means the points are already in Monty's model frame (object-local + p_train,
``rowant_lab/metrics_v1/pose_metrics.py:38-49``) and ``T_prior_to_model`` must be null; ``"prior"``
means the source's own frame, with ``T_prior_to_model`` the 4x4 similarity (scale * rotation,
translation) a registration found, or null when not registered yet.

Spacing: the frozen densifier de-duplicates its cloud on a 0.5 mm voxel grid (``DEDUP = 0.0005``,
``rowant_lab/pose_v5/densify_checkpoint.py:62``, applied at ``:91``). ``read``/``write`` refuse a shape
whose closest pair of points is nearer than ``DEDUP_M`` unless ``allow_dense=True``. The check here is
the exact pairwise minimum (``scipy.spatial.cKDTree`` nearest neighbour), which is stricter than the
densifier's voxel rule (two points in adjacent voxels may be closer than 0.5 mm); that difference is
declared, not hidden.

Fail-closed: wrong keys, dtypes, shapes, non-finite values, non-unit normals, N < 1, unknown or
missing meta keys and duplicate JSON keys all raise ``ValueError``. No file here is ever read as
truth; the oracle source declares itself through ``provenance.source``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from rowant_lab.hybrid_frontier.contracts.sha import (
    is_hex64,
    sha256_arrays,
    sha256_file,
)

SCHEMA = "rowant.hybrid_frontier.prior_shape"
VERSION = 1
DEDUP_M = 0.0005  # densify_checkpoint.py:62
NORMAL_TOL = 1e-6
SIMILARITY_TOL = 1e-6
NPZ_KEYS = {"points", "normals", "part_id", "meta"}
META_KEYS = {"schema", "version", "object", "frame", "provenance", "normals_kind"}
FRAME_KEYS = {"kind", "T_prior_to_model"}
PROVENANCE_KEYS = {"source", "generator_sha256", "scale", "note"}
NORMALS_KINDS = ("outward", "view_hint")
FRAME_KINDS = ("model", "prior")
TEXT_MAX = 256


def _keys(data: Any, required: set[str], path: str) -> None:
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: not an object")
    missing = sorted(required - set(data))
    unknown = sorted(set(data) - required)
    if missing:
        raise ValueError(f"{path}: missing field(s) {missing}")
    if unknown:
        raise ValueError(f"{path}: unknown field(s) {unknown}")


def _text(x: Any, path: str, allow_empty: bool = False) -> str:
    if type(x) is not str or len(x) > TEXT_MAX or (not allow_empty and not x.strip()):
        raise ValueError(f"{path}: nonempty string of at most {TEXT_MAX} characters required")
    return x


def _no_duplicates(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"meta: duplicate JSON key {k!r}")
        out[k] = v
    return out


def _similarity_4x4(x: Any, path: str) -> list:
    try:
        m = np.array(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: 4x4 numeric matrix required") from exc
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError(f"{path}: finite 4x4 matrix required")
    if np.abs(m[3] - np.array([0.0, 0.0, 0.0, 1.0])).max() > SIMILARITY_TOL:
        raise ValueError(f"{path}: last row must be [0, 0, 0, 1]")
    a = m[:3, :3]
    det = np.linalg.det(a)
    if det <= 0:
        raise ValueError(f"{path}: 3x3 block must have positive determinant")
    scale = det ** (1.0 / 3.0)
    if np.abs(a @ a.T / (scale * scale) - np.eye(3)).max() > SIMILARITY_TOL:
        raise ValueError(f"{path}: 3x3 block is not scale * rotation within {SIMILARITY_TOL}")
    return [[float(v) for v in row] for row in m]


def make_meta(*, object: str, normals_kind: str, source: str, generator_sha256: str, scale: float = 1.0,
              note: str = "", frame_kind: str = "model", T_prior_to_model=None) -> dict:
    """A meta dict in canonical key order; validated by ``validate_meta``."""
    t = None if T_prior_to_model is None else [[float(v) for v in row] for row in np.asarray(T_prior_to_model, dtype=float)]
    return validate_meta({
        "schema": SCHEMA, "version": VERSION, "object": object,
        "frame": {"kind": frame_kind, "T_prior_to_model": t},
        "provenance": {"source": source, "generator_sha256": generator_sha256, "scale": float(scale), "note": note},
        "normals_kind": normals_kind,
    })


def validate_meta(meta: Mapping[str, Any]) -> dict:
    """Fail-closed check of the meta block; returns a canonical copy."""
    _keys(meta, META_KEYS, "meta")
    if meta["schema"] != SCHEMA:
        raise ValueError(f"meta.schema: {meta['schema']!r} is not {SCHEMA!r}")
    if type(meta["version"]) is not int or meta["version"] != VERSION:
        raise ValueError(f"meta.version: {meta['version']!r} is not {VERSION}")
    if meta["normals_kind"] not in NORMALS_KINDS:
        raise ValueError(f"meta.normals_kind: {meta['normals_kind']!r} not in {NORMALS_KINDS}")
    frame = meta["frame"]
    _keys(frame, FRAME_KEYS, "meta.frame")
    if frame["kind"] not in FRAME_KINDS:
        raise ValueError(f"meta.frame.kind: {frame['kind']!r} not in {FRAME_KINDS}")
    t = frame["T_prior_to_model"]
    if frame["kind"] == "model":
        if t is not None:
            raise ValueError("meta.frame.T_prior_to_model: must be null when kind is 'model'")
    elif t is not None:
        t = _similarity_4x4(t, "meta.frame.T_prior_to_model")
    prov = meta["provenance"]
    _keys(prov, PROVENANCE_KEYS, "meta.provenance")
    if not is_hex64(prov["generator_sha256"]):
        raise ValueError("meta.provenance.generator_sha256: 64-character lowercase hex sha256 required")
    scale = prov["scale"]
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("meta.provenance.scale: finite positive number required")
    return {
        "schema": SCHEMA, "version": VERSION, "object": _text(meta["object"], "meta.object"),
        "frame": {"kind": frame["kind"], "T_prior_to_model": t},
        "provenance": {"source": _text(prov["source"], "meta.provenance.source"),
                       "generator_sha256": prov["generator_sha256"], "scale": float(scale),
                       "note": _text(prov["note"], "meta.provenance.note", allow_empty=True)},
        "normals_kind": meta["normals_kind"],
    }


def min_pairwise_spacing(points) -> float:
    """Smallest distance between two distinct points (``cKDTree.query`` with k=2); inf when N < 2."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points: (N, 3) array required")
    if len(p) < 2:
        return math.inf
    d, _ = cKDTree(p).query(p, k=2)
    return float(d[:, 1].min())


def dedup_check(points, allow_dense: bool = False) -> float:
    """Refuse (``ValueError``) when two points are closer than ``DEDUP_M`` unless ``allow_dense``; returns the spacing."""
    spacing = min_pairwise_spacing(points)
    if spacing < DEDUP_M and not allow_dense:
        raise ValueError(f"points: closest pair {spacing * 1e3:.4f} mm is under the {DEDUP_M * 1e3:.1f} mm DEDUP spacing "
                         "(pass allow_dense=True to keep a denser cloud on purpose)")
    return spacing


def validate_arrays(points, normals, part_id, *, allow_dense: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Type/shape/finiteness/unit-normal/spacing checks; returns float64/int16 copies."""
    if not isinstance(points, np.ndarray) or points.dtype != np.float64:
        raise ValueError("points: float64 ndarray required")
    if not isinstance(normals, np.ndarray) or normals.dtype != np.float64:
        raise ValueError("normals: float64 ndarray required")
    if not isinstance(part_id, np.ndarray) or part_id.dtype != np.int16:
        raise ValueError("part_id: int16 ndarray required")
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
        raise ValueError("points: shape (N, 3) with N >= 1 required")
    n = points.shape[0]
    if normals.shape != (n, 3):
        raise ValueError(f"normals: shape ({n}, 3) required, got {normals.shape}")
    if part_id.shape != (n,):
        raise ValueError(f"part_id: shape ({n},) required, got {part_id.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points: non-finite value")
    if not np.isfinite(normals).all():
        raise ValueError("normals: non-finite value")
    norms = np.linalg.norm(normals, axis=1)
    if np.abs(norms - 1.0).max() > NORMAL_TOL:
        raise ValueError(f"normals: not unit length within {NORMAL_TOL} (max deviation {np.abs(norms - 1.0).max():.3e})")
    dedup_check(points, allow_dense)
    return np.array(points, dtype=np.float64), np.array(normals, dtype=np.float64), np.array(part_id, dtype=np.int16)


@dataclass(frozen=True)
class PriorShape:
    points: np.ndarray
    normals: np.ndarray
    part_id: np.ndarray
    meta: dict

    @property
    def n(self) -> int:
        return int(self.points.shape[0])

    @property
    def normals_kind(self) -> str:
        return self.meta["normals_kind"]

    @property
    def object(self) -> str:
        return self.meta["object"]

    def validate(self, allow_dense: bool = False) -> "PriorShape":
        p, nrm, pid = validate_arrays(self.points, self.normals, self.part_id, allow_dense=allow_dense)
        return PriorShape(p, nrm, pid, validate_meta(self.meta))

    def digest(self) -> str:
        """sha256 over the three arrays plus the canonical meta text (``sha.sha256_arrays``)."""
        meta_bytes = np.frombuffer(json.dumps(self.meta, sort_keys=True, separators=(",", ":")).encode(), dtype=np.uint8)
        return sha256_arrays({"points": self.points, "normals": self.normals, "part_id": self.part_id, "meta": meta_bytes})


def write(path, points, normals, part_id, meta: Mapping[str, Any], *, allow_dense: bool = False) -> str:
    """Validate then ``np.savez`` the four keys; returns the file's sha256."""
    shape = PriorShape(points, normals, part_id, dict(meta)).validate(allow_dense=allow_dense)
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"{path}: PriorShape files end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_text = json.dumps(shape.meta, indent=1, allow_nan=False)
    np.savez(path, points=shape.points, normals=shape.normals, part_id=shape.part_id, meta=np.array(meta_text))
    return sha256_file(path)


def read(path, *, allow_dense: bool = False) -> PriorShape:
    """Load and fail-closed validate a PriorShape npz."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        if keys != NPZ_KEYS:
            raise ValueError(f"{path}: npz keys {sorted(keys)} are not {sorted(NPZ_KEYS)}")
        meta_arr = z["meta"]
        if meta_arr.ndim != 0 or meta_arr.dtype.kind != "U":
            raise ValueError(f"{path}: meta must be a 0-d unicode array holding JSON")
        meta = json.loads(str(meta_arr), object_pairs_hook=_no_duplicates)
        points, normals, part_id = z["points"], z["normals"], z["part_id"]
    return PriorShape(points, normals, part_id, meta).validate(allow_dense=allow_dense)
