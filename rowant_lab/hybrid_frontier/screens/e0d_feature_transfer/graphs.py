"""TRUTH-FREE: export one Monty memory's graphs to a compact npz, and load it back.

Reads only a Monty ``model.pt`` (Monty's own learned graphs: ``state["lm_dict"][0]["graph_memory"][obj]["patch"]``,
as ``e6/truth_graph_ref.py`` and ``regime/graph_export.py`` read it). For each graph id it keeps ``pos`` (model
frame, metres; the model frame is world A at the training pose, offset by p_train = (0, 1.5, 0)) and the
``graph.x`` columns that Monty's evidence step reads: ``pose_vectors`` (9: surface normal, curvature direction 1,
curvature direction 2), ``pose_fully_defined`` (1), ``hsv`` (3) and ``principal_curvatures_log`` (2). Values are
the checkpoint's own, widened to float64 (exact). The saved ``x`` has its own ``feature_mapping`` (start, end) in
the meta.

Loading ``model.pt`` imports torch and, through unpickling, tbp.monty. ``export`` runs the isolation check first
(``isolation.require``: a stand-in in this release, it inspects nothing), refuses a sha256 mismatch before importing
torch, and is meant to run as its own process (``python -m ...graphs export``) so the caller never holds a memory.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import resource
import sys
from pathlib import Path

import numpy as np

from rowant_lab.hybrid_frontier.screens.e0d_feature_transfer import isolation
from rowant_lab.settings import DATA

FORMAT = "rowant hybrid_frontier screen e0d graph features v1"
CHANNEL = "patch"
KEEP = ("pose_vectors", "pose_fully_defined", "hsv", "principal_curvatures_log")
MAX_MODEL_PT_BYTES = 256 * 2**20
MEMORIES = {
    "F1-77-s42": {"path": DATA / "runs" / "hybrid-frontier" / "memories" / "A0-s42" / "model.pt",
                  "sha256": "b6b3e874fd3ca46eec4c4a8b8964a650fdf28062e414b1453b1543cc4e9ec6d4"},
    "U77-s42": {"path": DATA / "runs" / "bench-v3-u77pod" / "U77fused-s42-merged" / "pretrained" / "model.pt",
                "sha256": "45d3365ad9c5f7a8314bf46e968f890110ce408de39a2a7dd2a3187c9daa7094"},
}


class GraphError(ValueError):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _np(v) -> np.ndarray:
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v)


def channel_arrays(model) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(pos, x_kept, mapping)`` for one channel model; ``mapping`` gives each kept feature's columns in x_kept."""
    pos = np.asarray(_np(model.pos), dtype=np.float64)
    x = np.asarray(_np(model.x), dtype=np.float64)
    fm = dict(model.feature_mapping)
    cols, mapping, start = [], {}, 0
    for f in KEEP:
        if f not in fm:
            raise GraphError(f"graph lacks feature {f!r} (has {sorted(fm)})")
        a, b = int(fm[f][0]), int(fm[f][1])
        cols.append(x[:, a:b])
        mapping[f] = [start, start + (b - a)]
        start += b - a
    xk = np.ascontiguousarray(np.concatenate(cols, axis=1))
    if pos.ndim != 2 or pos.shape[1] != 3 or len(xk) != len(pos):
        raise GraphError(f"shapes pos {pos.shape} x {xk.shape}")
    return np.ascontiguousarray(pos), xk, mapping


def build(graph_memory, *, source: dict) -> tuple[dict, dict]:
    arrays, counts, mapping = {}, {}, None
    for g, channels in graph_memory.items():
        g = str(g)
        if CHANNEL not in channels:
            raise GraphError(f"graph {g} has no {CHANNEL!r} channel")
        pos, xk, m = channel_arrays(channels[CHANNEL])
        if mapping is None:
            mapping = m
        elif m != mapping:
            raise GraphError(f"graph {g}: feature layout {m} differs from {mapping}")
        arrays[f"pos__{g}"] = pos
        arrays[f"x__{g}"] = xk
        counts[g] = int(len(pos))
    meta = {"format": FORMAT, "truth_free": True, "channel": CHANNEL, "source": dict(source),
            "node_counts": counts, "feature_mapping": mapping, "kept": list(KEEP)}
    return arrays, meta


def write(path, arrays: dict, meta: dict) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, meta_json=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    data = buf.getvalue()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(Path(path), "xb") as fh:
        fh.write(data)
    return hashlib.sha256(data).hexdigest()


def export(memory: str, out_npz, *, model_pt=None, expect_sha256=None) -> dict:
    """Isolation check, sha check, then torch.load one memory and write its npz (exclusive create)."""
    spec = MEMORIES.get(memory, {})
    model_pt = Path(model_pt or spec["path"]).expanduser()
    expect_sha256 = expect_sha256 or spec["sha256"]
    iso = isolation.require()
    size = model_pt.stat().st_size
    if size > MAX_MODEL_PT_BYTES:
        raise GraphError(f"{model_pt}: {size} bytes exceeds {MAX_MODEL_PT_BYTES}")
    digest = sha256_file(model_pt)
    if digest != expect_sha256:
        raise GraphError(f"{model_pt}: sha256 {digest} != expected {expect_sha256}")
    import torch  # noqa: PLC0415 (only after the isolation and sha checks)

    state = torch.load(str(model_pt), weights_only=False)
    gm = state["lm_dict"][0]["graph_memory"]
    arrays, meta = build(gm, source={"memory": memory, "checkpoint": str(model_pt), "bytes": int(size),
                                     "sha256": digest})
    del state, gm
    meta["isolation"] = iso
    meta["max_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    meta["npz_sha256"] = write(out_npz, arrays, meta)
    return meta


def load(path) -> tuple[dict, dict, dict]:
    """``(pos, x, meta)``: dicts keyed by graph id (float64 arrays)."""
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        if meta.get("format") != FORMAT:
            raise GraphError(f"{path}: not a {FORMAT!r} file")
        pos = {k[5:]: np.asarray(z[k], dtype=np.float64) for k in z.files if k.startswith("pos__")}
        x = {k[3:]: np.asarray(z[k], dtype=np.float64) for k in z.files if k.startswith("x__")}
    if set(pos) != set(x) or set(pos) != set(meta["node_counts"]):
        raise GraphError(f"{path}: pos, x and node_counts name different graphs")
    return pos, x, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E-0d: export one memory's graph features (own process)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--memory", required=True, choices=sorted(MEMORIES))
    e.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    meta = export(args.memory, args.out)
    print(json.dumps({k: meta[k] for k in ("source", "feature_mapping", "max_rss_bytes", "npz_sha256")}, indent=1))
    print(json.dumps({"graphs": len(meta["node_counts"]), "nodes": int(sum(meta["node_counts"].values()))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
