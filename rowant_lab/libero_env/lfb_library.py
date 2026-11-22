"""LF-b variant libraries (Monty venv; registration ``lfc_b``, amendment LFB-LIB).

A variant tree ``lib/<id>/`` holds what ``m2_library.lfb_dirs`` maps:
- ``m2/onboard-v3/<suite>``: memory, colour memory, symmetry, templates, grids, models and onboarding.json;
- ``m2/policy-v3``: the type docs, m1types, workcells and manifest;
- ``m1``: ``records/`` and ``robot.json``.

Unchanged files are hard links to M2's own (the same drive, so no copy). Two kinds of tree:

* ``minus`` (``build_minus``): the held-out type T's graph ids are dropped from memory, colour memory, symmetry,
  templates and grids. The learning phase's perception runs on it, so T can only show up as unexplained points.
  Policy and M1 files are links.
* ``learned`` (``build_learned``): T comes back under its own graph id from a builder's model folder
  (``lfb_build``), which holds:
  - ``nodes.npz``: ``pos`` (model frame plus ``P_TRAIN``) and ``x`` (CHECK's columns). These are the builder's own
    nodes;
  - ``surface.ply``: the shared derivation's vertex-coloured surface, which becomes T's onboarded PLY. Colour
    memory, symmetry and templates come from it, for every arm alike;
  - ``render.ply`` (tsdf builder only): the reconstructed mesh, written as ``obj_<id>-render.ply`` beside the PLY.
    Only M−'s render-and-compare reads it (``m2_mminus.RenderCompareVC.object_model``);
  - ``m1type.json`` and ``records.json``: the shared derivation's collision boxes, COM, rest-up and grasp records.
  T's policy type doc keeps its kinematics (one free root body) and loses every export geometry reference.

Nothing here reads a scene, a frame or a truth file. T's true files are only ever the ones *removed*.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from rowant_lab.libero_env import lfb_paths as LP
from rowant_lab.libero_env import m2_paths as MP


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def link_tree(src: Path, dst: Path, skip=()) -> None:
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src).as_posix()
        if p.is_dir() or rel in skip or any(rel.startswith(s.rstrip("/") + "/") for s in skip if s.endswith("/")):
            continue
        q = dst / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        os.link(p, q)


def gids_of(ob: dict, typ: str) -> list:
    return sorted((g for g, (t, _pa) in ob["graph_ids"].items() if t == typ), key=int)


def _npz_arrays(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def write_onboard(suite: str, out: Path, drop: list, add: dict | None = None, base=None) -> dict:
    """M2's onboarded workcell with the graph ids ``drop`` removed and, when ``add`` is given (``{gid: model folder}``),
    those ids rebuilt from learned models. ``base``: the library tree to start from (``m2_library.lfb_dirs``; None is
    M2's own, as before)."""
    from rowant_lab.fastlayer import memory as ML
    from rowant_lab.fastlayer import onboard as OB
    from rowant_lab.fastlayer import segment as SG
    from rowant_lab.fastlayer import symmetry as SY
    from rowant_lab.fastlayer import wide
    from rowant_lab.fastlayer.scorers import monty_colour as MC
    from rowant_lab.hybrid_frontier.screens.e0d_feature_transfer import graphs as e0d
    from rowant_lab.libero_env import m2_onboard as OBM

    from rowant_lab.libero_env import m2_library as LB

    src = LB.lfb_dirs(base)[0] / suite
    ob = json.loads((src / "onboarding.json").read_text())
    add = add or {}
    drop = [str(g) for g in drop]
    out.mkdir(parents=True, exist_ok=False)
    models = out / "models"
    link_tree(src / "models", models, skip=[f"obj_{int(g):06d}.ply" for g in drop])
    t0 = time.perf_counter()
    # memory
    arr = _npz_arrays(src / "memory.npz")
    meta = json.loads(str(arr.pop("meta_json")))
    for g in drop:
        arr.pop(f"pos__{g}"), arr.pop(f"x__{g}")
        meta["node_counts"].pop(g)
        meta["objects"].pop(g)
    for g, mdir in add.items():
        mdir = Path(mdir)
        n = _npz_arrays(mdir / "nodes.npz")
        ply = models / f"obj_{int(g):06d}.ply"
        shutil.copyfile(mdir / "surface.ply", ply)
        if (mdir / "render.ply").exists():
            shutil.copyfile(mdir / "render.ply", models / f"obj_{int(g):06d}-render.ply")
        arr[f"pos__{g}"], arr[f"x__{g}"] = n["pos"], n["x"]
        meta["node_counts"][g] = int(len(n["pos"]))
        meta["objects"][g] = {"ply": str(ply), "ply_sha256": sha256_file(ply), "texture": None, "texture_sha256": None,
                              "colour_source": "vertex", "n_nodes": int(len(n["pos"])), "learned": True}
    meta["source"] = {**meta["source"], "models_dir": str(models), "lfb": True}
    meta.pop("npz_sha256", None)
    mem_npz = out / "memory.npz"
    meta["npz_sha256"] = e0d.write(mem_npz, arr, meta)
    # colour memory
    carr = _npz_arrays(src / "colour-memory.npz")
    cmeta = json.loads(bytes(carr.pop("__meta__")).decode())
    for g in drop:
        for k in [k for k in carr if k.startswith(f"{g}__")]:
            carr.pop(k)
        cmeta["objects"].pop(g, None)
    pv = OB.FEATURE_MAPPING["pose_vectors"][0]
    for g in add:
        a, d = MC.onboard_object(g, meta["objects"][g]["ply"], None, arr[f"pos__{g}"] - OB.P_TRAIN,
                                 arr[f"x__{g}"][:, pv:pv + 3], {"density_per_mm2": MC.ONBOARD["density_per_mm2"]})
        for k, v in a.items():
            carr[f"{g}__{k}"] = v
        cmeta["objects"][g] = d
    cmeta.update(memory_npz=str(mem_npz), memory_sha256=meta["npz_sha256"], lfb=True)
    carr["__meta__"] = np.frombuffer(json.dumps(cmeta, sort_keys=True, default=str).encode(), dtype=np.uint8)
    with open(out / "colour-memory.npz", "xb") as fh:
        np.savez(fh, **carr)
    # symmetry
    sdoc = json.loads((src / "symmetry.json").read_text())
    for g in drop:
        sdoc["objects"].pop(g, None)
    for g in add:
        sdoc["objects"][g] = SY.detect(g, meta["objects"][g]["ply"])
    sdoc["models_dir"] = str(models)
    (out / "symmetry.json").write_text(json.dumps(sdoc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    # templates: the cache minus the dropped ids, plus the learned ids' rendered views
    p = dict(SG.PARAMS, **OBM.TEMPLATE_PARAMS)
    tsrc = SG.template_cache_path(src / "templates", p)
    tarr = _npz_arrays(tsrc)
    shas = json.loads(bytes(tarr.pop("__ply_sha256__")).decode()) if "__ply_sha256__" in tarr else {}
    for g in drop:
        tarr.pop(f"{g}__des", None), tarr.pop(f"{g}__xyz", None)
        shas.pop(g, None)
    for g in add:
        des, xyz, _ = SG.build_template_views(g, meta["objects"][g]["ply"], None, p)
        tarr[f"{g}__des"], tarr[f"{g}__xyz"] = des.astype(np.float32), xyz.astype(np.float32)
        shas[g] = meta["objects"][g]["ply_sha256"]
    tarr["__ply_sha256__"] = np.frombuffer(json.dumps(shas, sort_keys=True).encode(), dtype=np.uint8)
    tdst = SG.template_cache_path(out / "templates", p)
    tdst.parent.mkdir(parents=True)
    with open(tdst, "xb") as fh:
        np.savez(fh, **tarr)
    # grids
    mem = ML.load(mem_npz, kind="mesh", expect_sha256=meta["npz_sha256"])
    wide.build_grid_set(mem.shapes, meta["npz_sha256"], out / "grids", OBM.GRID_H_MM, OBM.GRID_K)
    # onboarding.json
    doc = dict(ob)
    doc["graph_ids"] = {g: v for g, v in ob["graph_ids"].items() if g not in drop or g in add}
    doc["per_part"] = {g: v for g, v in ob["per_part"].items() if g not in drop}
    for g in add:
        doc["per_part"][g] = {"type": ob["graph_ids"][g][0], "part": ob["graph_ids"][g][1], "learned": True,
                              "n_nodes": meta["node_counts"][g]}
    doc.update(memory_sha256=meta["npz_sha256"], lfb={"drop": drop, "add": {g: str(m) for g, m in add.items()},
                                                      "seconds": round(time.perf_counter() - t0, 2)})
    files = sorted(q for q in out.rglob("*") if q.is_file())
    doc["files"] = {str(q.relative_to(out)): sha256_file(q) for q in files}
    (out / "onboarding.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    return doc


def _jbytes(doc) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()


def write_policy(out_root: Path, typ: str | None = None, mdir: Path | None = None, base=None) -> dict:
    """``m2/policy-v3`` and ``m1``: links to M2's own (or ``base``'s); with ``typ``/``mdir``, T's type doc, m1type and
    grasp records from the learned model, and the manifest updated to their sha256."""
    from rowant_lab.libero_env import m2_library as LB

    _ob, pol_src, m1_src = LB.lfb_dirs(base)
    pol, m1 = out_root / "m2" / "policy-v3", out_root / "m1"
    repl = {} if typ is None else {f"types/{typ}.json", f"m1types/{typ}.json", "manifest.json"}
    link_tree(pol_src, pol, skip=repl)
    link_tree(m1_src / "records", m1 / "records", skip={f"{typ}.json"} if typ else ())
    os.link(m1_src / "robot.json", m1 / "robot.json")
    if typ is None:
        return {"manifest_sha256": sha256_file(pol / "manifest.json")}
    man = json.loads((pol_src / "manifest.json").read_text())
    tdoc = json.loads((pol_src / "types" / f"{typ}.json").read_text())
    for b in tdoc["bodies"].values():
        b.pop("geoms", None)
    tdoc["npz"], tdoc["npz_sha256"] = None, sha256_file(mdir / "surface.ply")
    tdoc["learned"] = "lfb"
    m1t = json.loads((mdir / "m1type.json").read_text())
    rec = (mdir / "records.json").read_bytes()
    for rel, data in ((f"types/{typ}.json", _jbytes(tdoc)), (f"m1types/{typ}.json", _jbytes(m1t))):
        (pol / rel).write_bytes(data)
        man["files"][rel] = hashlib.sha256(data).hexdigest()
    (m1 / "records" / f"{typ}.json").write_bytes(rec)
    man["in_place"][f"m1/records/{typ}.json"] = hashlib.sha256(rec).hexdigest()
    man["lfb"] = "learned"
    (pol / "manifest.json").write_bytes(_jbytes(man))
    return {"manifest_sha256": sha256_file(pol / "manifest.json")}


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(q for q in root.rglob("*") if q.is_file()):
        h.update(p.relative_to(root).as_posix().encode() + b"\0" + sha256_file(p).encode() + b"\n")
    return h.hexdigest()


def finish(root: Path, kind: str, info: dict) -> dict:
    doc = {"kind": kind, **info, "digest": tree_digest(root / "m2") + ":" + tree_digest(root / "m1"),
           "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (root / "variant.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    return doc


def build_minus(suite: str, typ: str, vid: str | None = None) -> dict:
    root = LP.LIB / (vid or f"minus-{suite}-{typ}")
    if (root / "variant.json").exists():
        return json.loads((root / "variant.json").read_text())
    ob = json.loads((MP.ONBOARD_DIR / suite / "onboarding.json").read_text())
    drop = gids_of(ob, typ)
    write_onboard(suite, root / "m2" / "onboard-v3" / suite, drop)
    pol = write_policy(root)
    return finish(root, "minus", {"suite": suite, "held_out": typ, "dropped_gids": drop, **pol})


def build_learned(suite: str, typ: str, mdir: Path, vid: str, root: Path | None = None, base=None) -> dict:
    """``root``: where the tree goes (default ``LP.LIB / vid``); ``base``: the library it starts from (default M2's
    own). In-episode learning (``lfb_episode``) passes both: its own folder and the run's library."""
    from rowant_lab.libero_env import m2_library as LB

    root = Path(root) if root is not None else LP.LIB / vid
    if (root / "variant.json").exists():
        return json.loads((root / "variant.json").read_text())
    ob = json.loads((LB.lfb_dirs(base)[0] / suite / "onboarding.json").read_text())
    gids = gids_of(ob, typ)
    if len(gids) != 1:
        raise ValueError(f"{typ}: {len(gids)} parts; LF-b learns single-part types only")
    write_onboard(suite, root / "m2" / "onboard-v3" / suite, gids, {gids[0]: mdir}, base=base)
    pol = write_policy(root, typ, Path(mdir), base=base)
    return finish(root, "learned", {"suite": suite, "held_out": typ, "gid": gids[0], "model_dir": str(mdir),
                                    "model_sha256": {f.name: sha256_file(f) for f in sorted(Path(mdir).iterdir())
                                                     if f.is_file()}, **pol})


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("minus", "learned"))
    ap.add_argument("--suite", required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--vid", default=None)
    a = ap.parse_args(argv)
    doc = build_minus(a.suite, a.type, a.vid) if a.kind == "minus" else \
        build_learned(a.suite, a.type, Path(a.model_dir), a.vid)
    print(json.dumps({k: doc[k] for k in ("kind", "held_out", "digest")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
