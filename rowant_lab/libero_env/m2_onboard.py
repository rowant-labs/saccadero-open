"""M2 onboarding (plan object_library_and_onboarding.step_2): LIBERO's exported meshes -> the fast layer's artifacts.

M2 (perceived poses, grounded instruction): perception in the loop. Monty venv, worktree root::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONPATH=$PWD <server python> -P -m rowant_lab.libero_env.m2_onboard \\
        --suite libero_object

Per workcell (suite), per type and part (graph ids 1, 2, ... by sorted (type, part)):

1. **Mesh** (``part_mesh``): the part's visual geoms in the part frame (``pose_in_part``), written as an ASCII PLY
   in BOP's form (mm, per-vertex normals) under ``onboard/<suite>/models/obj_XXXXXX.ply``. A part whose geoms share
   one texture and all have UVs is TEXTURED (``texture_u``, ``texture_v`` with ``v' = 1 - v``, because FM2 found
   MuJoCo's texcoords in image-row order (row = v H) while the fast layer reads the toolkit's row = (1 - v) H); the
   texture is referenced in place by path. Any other part (several materials, rgba-only or UV-less geoms: the
   cabinet's drawers with their metal handles, the stove) is VERTEX-COLOURED: each vertex gets its material's texel
   at its UV (row = v H) times ``mat_rgba``, or the texture's mean colour times ``mat_rgba`` without UVs, or the
   geom's rgba.
2. **Graph memory** (E-0d npz, ``onboard.build_object``'s steps, copied here as ``build_object_grid`` only to take
   the lattice as an argument): the lattice is ``onboard.GRID_M`` (2 mm) unless the part would get more than
   ``MAX_NODES`` nodes; then it is scaled up by ``sqrt(n / MAX_NODES)`` (rounded up to 0.5 mm) until it fits; the
   scale is recorded.
3. **Colour memory** (scorer A, ``monty_colour.onboard_object`` unchanged; its dense density is capped so a part
   has at most ``MAX_COLOUR_DENSE`` dense samples, recorded).
4. **Symmetry** (``symmetry.detect`` unchanged, per part PLY).
5. **Templates** (``segment.load_templates`` unchanged, into the onboarding's own cache folder) at the template
   parameters ``TEMPLATE_PARAMS`` (agentview's focal length and working distance).
6. **Node grids** (``wide.build_grid_set``, h 2.5 mm, k 1), from ``memory.load(kind="mesh")``'s shapes.

Timed per object and part, per stage, in one process and one thread, with the load and free memory. No image is
written (textures stay where LIBERO keeps them).

m2r3:fix-r1 (amendment A65): the export names each texture by its path on the machine that exported it.
``asset_path`` reads the same file from this machine's LIBERO assets folder, given as ``--assets-root`` (this
perception-side module may not import the setup module ``paths``), when that path does not exist here, sha256-checked
against the export's record; on the exporting machine nothing changes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from rowant_lab.libero_env import m2_paths as MP

ASSET_KEY = "/libero/libero/assets/"
ASSETS_ROOT: list = []  # A65: --assets-root, this machine's LIBERO assets folder (set by main; empty: none given)


def asset_path(path, sha256: str | None = None):
    """A61/A65: a LIBERO asset named by its path on the exporting machine: that path when it exists here, else the same
    file under the ``--assets-root`` given (this machine's LIBERO assets folder); sha256-checked then."""
    import hashlib

    if not path:
        return path
    p = Path(path)
    if not p.exists():
        s_ = str(path)
        if ASSET_KEY not in s_ or not ASSETS_ROOT:
            raise FileNotFoundError(f"{s_} (no --assets-root given)")
        p = Path(ASSETS_ROOT[0]) / s_.split(ASSET_KEY, 1)[1]
        if sha256 and hashlib.sha256(p.read_bytes()).hexdigest() != sha256:
            raise RuntimeError(f"asset {p}: sha256 differs from the export's {sha256}")
    return str(p)

FORMAT = "rowant libero_env M2 onboarding v1"
MAX_NODES = 20000
MAX_COLOUR_DENSE = 1_500_000
TEMPLATE_PARAMS = {"view_f": 1236.0, "view_dist_m": 1.1}
GRID_H_MM = 2.5
GRID_K = 1


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parts_of_suite(suite: str, export_dir=MP.EXPORT_DIR) -> list[tuple[str, str]]:
    out = []
    for typ in sorted(MP.WORKCELL_TYPES[suite]):
        tdoc = json.loads((Path(export_dir) / "types" / f"{typ}.json").read_text())
        for pa in sorted(tdoc["parts"]):
            out.append((typ, pa))
    return out


def graph_ids(suite: str, export_dir=MP.EXPORT_DIR) -> dict:
    """{(type, part): integer graph id} by sorted (type, part), from 1."""
    return {tp: i + 1 for i, tp in enumerate(parts_of_suite(suite, export_dir))}


def _texture(path, cache):
    from rowant_lab.fastlayer import onboard as OB

    if path not in cache:
        cache[path] = OB.load_texture(path)
    return cache[path]


def texel_rows_v(tex, uv):
    """Bilinear texel at UV with MuJoCo's row = v H (FM2), wrapping."""
    from rowant_lab.fastlayer import onboard as OB

    uv2 = np.stack([np.mod(uv[:, 0], 1.0), 1.0 - np.mod(uv[:, 1], 1.0)], 1)
    return OB.texel(tex, uv2, flip_v=True)


def part_mesh(tdoc: dict, arrays: dict, part: str, texcache: dict) -> dict:
    """V (m), F, N, and either UV (toolkit convention) + texture path, or per-vertex RGB."""
    Vs, Fs, Ns, UVs, mats = [], [], [], [], []
    off = 0
    for r in tdoc["parts"][part]["bodies"]:
        P = tdoc["pose_in_part"][r]
        R, t = np.asarray(P["R"], float), np.asarray(P["t"], float)
        for g in tdoc["bodies"][r]["geoms"]:
            V = arrays[g["key"] + "__V"] @ R.T + t
            N = arrays[g["key"] + "__N"] @ R.T
            F = arrays[g["key"] + "__F"] + off
            UV = arrays[g["key"] + "__UV"] if g["has_uv"] else None
            Vs.append(V)
            Ns.append(N)
            Fs.append(F)
            UVs.append(UV)
            mats.append((g["material"], len(V)))
            off += len(V)
    V, F, N = np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Ns)
    texs = {m.get("texture") for m, _ in mats}
    textured = len(texs) == 1 and None not in texs and all(u is not None for u in UVs) \
        and all(abs(x - 1.0) < 1e-9 for m, _ in mats for x in m.get("mat_rgba", [1, 1, 1])[:3])
    out = {"V": V, "F": F, "N": N, "textured": textured, "materials": [m for m, _ in mats]}
    if textured:
        UV = np.concatenate(UVs)
        out["UV"] = np.stack([UV[:, 0], 1.0 - UV[:, 1]], 1)  # MuJoCo row = v H -> toolkit row = (1 - v) H
        out["texture"] = next(iter(texs))
        return out
    rgb = []
    for (m, n), UV in zip(mats, UVs):
        if m.get("texture"):
            tex = _texture(m["texture"], texcache)
            mul = np.asarray(m.get("mat_rgba", [1, 1, 1, 1])[:3], float)
            if UV is not None:
                c = texel_rows_v(tex, UV) * mul
            else:
                c = np.tile(tex.reshape(-1, 3).mean(0) / 255.0 * mul, (n, 1))
        else:
            c = np.tile(np.asarray(m.get("mat_rgba", m.get("rgba"))[:3], float), (n, 1))
        rgb.append(np.clip(c, 0, 1))
    out["RGB"] = np.concatenate(rgb)
    return out


def write_ply(path: Path, mesh: dict) -> None:
    V = mesh["V"] * 1000.0
    lines = ["ply", "format ascii 1.0", "comment rowant M2 LIBERO part mesh (mm)"]
    lines += [f"element vertex {len(V)}", "property float x", "property float y", "property float z",
              "property float nx", "property float ny", "property float nz"]
    if mesh["textured"]:
        lines += ["property float texture_u", "property float texture_v"]
        cols = np.concatenate([V, mesh["N"], mesh["UV"]], 1)
        fmt = " ".join(["%.6f"] * 8)
    else:
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
        cols = np.concatenate([V, mesh["N"], np.rint(mesh["RGB"] * 255)], 1)
        fmt = " ".join(["%.6f"] * 6 + ["%d"] * 3)
    lines += [f"element face {len(mesh['F'])}", "property list uchar int vertex_indices", "end_header"]
    with open(path, "x", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\n")
        np.savetxt(fh, cols, fmt=fmt)
        np.savetxt(fh, np.concatenate([np.full((len(mesh["F"]), 1), 3), mesh["F"]], 1), fmt="%d")


def build_object_grid(obj_id: str, ply_path, texture_path, grid_m: float):
    """``onboard.build_object`` (unchanged steps and helpers) with the node lattice ``grid_m`` as an argument."""
    from skimage.color import rgb2hsv
    from scipy.spatial import cKDTree

    from rowant_lab.fastlayer import onboard as OB

    t0 = time.perf_counter()
    mesh = OB.read_ply(ply_path)
    V = mesh["vertices_mm"] * 1e-3
    F = mesh["faces"]
    src = OB.colour_source(mesh, ply_path, texture_path)
    textured = src["kind"] == "texture"
    tex_path, tex = src["path"], src.get("texture")
    fn, area = OB.face_normals(V, F)
    area_mm2 = float(area.sum() * 1e6)
    n_dense = int(math.ceil(area_mm2 * OB.DENSE_PER_MM2))
    rng = OB.object_rng(obj_id)
    fi, bary = OB.sample_surface(V, F, n_dense, rng)
    pts = OB.interp(V, F, fi, bary)
    nv = OB.interp(mesh["normals"], F, fi, bary)
    ln = np.linalg.norm(nv, axis=1)
    bad_n = ~(ln > 1e-9)
    nv[bad_n] = fn[fi[bad_n]]
    nv /= np.linalg.norm(nv, axis=1, keepdims=True)
    if textured:
        uv = OB.interp(mesh["uv"], F, fi, bary)
    pick = OB.voxel_pick(pts, grid_m)
    nodes, nn = pts[pick], nv[pick]
    tree = cKDTree(pts)
    cv = OB.node_curvatures(nodes, nn, pts, tree)
    if textured:
        rgb = OB.texel(tex, uv[pick], flip_v=True)
    else:
        rgb = OB.vertex_rgb(src["rgb"], F, fi[pick], bary[pick])
    hsv = np.asarray(rgb2hsv(rgb[None])[0], dtype=np.float64)
    pfd = (np.abs(cv["k1"] - cv["k2"]) >= OB.PC1_IS_PC2_THRESHOLD) & cv["valid"]
    x = np.concatenate([nn, cv["dir1"], cv["dir2"], pfd[:, None].astype(np.float64), hsv,
                        OB.log_sign(np.stack([cv["k1"], cv["k2"]], axis=1))], axis=1)
    diag = {"ply": str(ply_path), "ply_sha256": OB.sha256_file(ply_path), "texture": str(tex_path) if textured else None,
            "texture_sha256": OB.sha256_file(tex_path) if textured else None, "n_vertices": int(len(V)),
            "n_faces": int(len(F)), "area_mm2": round(area_mm2, 2),
            "extent_mm": [round(float(e) * 1e3, 2) for e in V.max(0) - V.min(0)], "n_dense": n_dense,
            "n_nodes": int(len(nodes)), "grid_m": grid_m, "seconds": {"build_total": round(time.perf_counter() - t0, 4)}}
    if not textured:
        diag["colour_source"] = "vertex"
    return np.ascontiguousarray(nodes + OB.P_TRAIN), np.ascontiguousarray(x), diag


def lattice_for(area_mm2: float) -> float:
    """2 mm unless the part would exceed MAX_NODES nodes (a surface lattice holds about area / g^2 nodes)."""
    g = 0.002
    while area_mm2 * 1e-6 / (g * g) > MAX_NODES:
        g = math.ceil(g * math.sqrt(area_mm2 * 1e-6 / (g * g) / MAX_NODES) * 2000) / 2000.0
    return g


def machine() -> dict:
    out = {"load1": round(os.getloadavg()[0], 2)}
    try:
        mp = subprocess.run(["/usr/bin/memory_pressure"], capture_output=True, text=True, timeout=20).stdout
        ln = [x for x in mp.splitlines() if "free percentage" in x]
        out["mem_free_pct"] = int(ln[-1].split(":")[-1].strip().rstrip("%")) if ln else None
    except Exception:
        out["mem_free_pct"] = None
    return out


def onboard_suite(suite: str, out_root: Path) -> dict:
    from rowant_lab.fastlayer import memory as ML
    from rowant_lab.fastlayer import onboard as OB
    from rowant_lab.fastlayer import segment as SG
    from rowant_lab.fastlayer import symmetry as SY
    from rowant_lab.fastlayer import wide
    from rowant_lab.fastlayer.scorers import monty_colour as MC
    from rowant_lab.hybrid_frontier.e9 import isolation
    from rowant_lab.hybrid_frontier.screens.e0d_feature_transfer import graphs as e0d_graphs

    iso = isolation.require()
    out = out_root / suite
    models = out / "models"
    models.mkdir(parents=True, exist_ok=False)
    ids = graph_ids(suite)
    export_man = json.loads((MP.EXPORT_DIR / "manifest.json").read_text())
    texcache: dict = {}
    per: dict = {}
    arrays, counts, objects = {}, {}, {}
    m0 = machine()
    for (typ, pa), gid in ids.items():
        g = str(gid)
        rec = per.setdefault(g, {"type": typ, "part": pa, "seconds": {}, "export_s_type": export_man[
            "seconds_per_type"].get(typ)})
        tdoc = json.loads((MP.EXPORT_DIR / "types" / f"{typ}.json").read_text())
        for b_ in tdoc["bodies"].values():  # A65: the texture files on THIS machine (sha256-checked)
            for g_ in b_["geoms"]:
                m_ = g_.get("material") or {}
                if m_.get("texture"):
                    m_["texture"] = asset_path(m_["texture"], m_.get("texture_sha256"))
        t0 = time.perf_counter()
        with np.load(MP.EXPORT_DIR / "types" / tdoc["npz"], allow_pickle=False) as z:
            arr = {k: np.asarray(z[k]) for k in z.files}
        mesh = part_mesh(tdoc, arr, pa, texcache)
        ply = models / f"obj_{gid:06d}.ply"
        write_ply(ply, mesh)
        rec["seconds"]["mesh_read_and_ply"] = round(time.perf_counter() - t0, 3)
        rec["textured"] = mesh["textured"]
        rec["texture"] = mesh.get("texture")
        _fn, area = OB.face_normals(mesh["V"], mesh["F"])
        area_mm2 = float(area.sum() * 1e6)
        grid = lattice_for(area_mm2)
        t1 = time.perf_counter()
        pos, x, diag = build_object_grid(g, ply, mesh.get("texture"), grid)
        rec["seconds"]["memory"] = round(time.perf_counter() - t1, 3)
        rec.update(n_nodes=int(len(pos)), grid_m=grid, area_mm2=round(area_mm2, 1), extent_mm=diag["extent_mm"])
        arrays[f"pos__{g}"], arrays[f"x__{g}"] = pos, x
        counts[g] = int(len(pos))
        objects[g] = diag
        print(f"{suite} {g} {typ}:{pa} nodes {len(pos)} grid {grid} textured {mesh['textured']}", flush=True)
    meta = {"format": e0d_graphs.FORMAT, "truth_free": True, "channel": e0d_graphs.CHANNEL,
            "source": {"kind": OB.MESH_FORMAT, "m2": FORMAT, "models_dir": str(models), "suite": suite},
            "node_counts": counts, "feature_mapping": OB.FEATURE_MAPPING, "kept": list(e0d_graphs.KEEP),
            "params": {**OB.PARAMS, "GRID_M": "per part (m2_onboard.lattice_for)"}, "objects": objects,
            "graph_ids": {str(v): list(k) for k, v in ids.items()}}
    mem_npz = out / "memory.npz"
    meta["npz_sha256"] = e0d_graphs.write(mem_npz, arrays, meta)
    # colour memory (scorer A): monty_colour.onboard_object per part, the dense density capped
    pv = OB.FEATURE_MAPPING["pose_vectors"][0]
    carr, cper = {}, {}
    for g in sorted(counts, key=int):
        t2 = time.perf_counter()
        area_mm2 = per[g]["area_mm2"]
        dens = min(MC.ONBOARD["density_per_mm2"], MAX_COLOUR_DENSE / max(area_mm2, 1.0))
        a, d = MC.onboard_object(g, objects[g]["ply"], objects[g].get("texture"), arrays[f"pos__{g}"] - OB.P_TRAIN,
                                 arrays[f"x__{g}"][:, pv:pv + 3], {"density_per_mm2": dens})
        for k_, v in a.items():
            carr[f"{g}__{k_}"] = v
        cper[g] = d
        per[g]["seconds"]["colour_memory"] = round(time.perf_counter() - t2, 3)
        per[g]["colour_density_per_mm2"] = round(dens, 4)
    cmeta = {"format": MC.FORMAT, "truth_free": True, "params": dict(MC.ONBOARD, density_per_mm2="per part"),
             "memory_npz": str(mem_npz), "memory_sha256": meta["npz_sha256"], "objects": cper, "created_utc": now()}
    carr["__meta__"] = np.frombuffer(json.dumps(cmeta, sort_keys=True, default=str).encode(), dtype=np.uint8)
    col_npz = out / "colour-memory.npz"
    with open(col_npz, "xb") as fh:
        np.savez(fh, **carr)
    # symmetry
    sym_objs = {}
    for g in sorted(counts, key=int):
        t3 = time.perf_counter()
        sym_objs[g] = SY.detect(g, objects[g]["ply"])
        per[g]["seconds"]["symmetry"] = round(time.perf_counter() - t3, 3)
        per[g]["symmetry"] = {"rotational_axis": sym_objs[g]["rotational_axis"], "n_family": sym_objs[g]["n_family"]}
    sdoc = {"format": SY.FORMAT, "truth_free": True, "params": SY.PARAMS, "models_dir": str(models), "objects": sym_objs}
    sym_json = out / "symmetry.json"
    with open(sym_json, "x", encoding="utf-8") as f:
        f.write(json.dumps(sdoc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    # templates (segment's cache, our folder)
    p = dict(SG.PARAMS, **TEMPLATE_PARAMS)
    tcache = out / "templates"
    tim: dict = {}
    SG.load_templates(meta["objects"], sorted(counts, key=int), p, str(tcache), timings=tim)
    for g, s in tim.items():
        per[g]["seconds"]["templates"] = s
    # node grids
    t4 = time.perf_counter()
    mem = ML.load(mem_npz, kind="mesh", expect_sha256=meta["npz_sha256"])
    wide.build_grid_set(mem.shapes, meta["npz_sha256"], out / "grids", GRID_H_MM, GRID_K)
    grid_s = time.perf_counter() - t4
    for g in counts:
        per[g]["seconds"]["grids_share"] = round(grid_s / len(counts), 3)
        per[g]["seconds"]["total"] = round(sum(v for k, v in per[g]["seconds"].items() if k != "total"), 3)
    files = sorted(q for q in out.rglob("*") if q.is_file())
    doc = {"format": FORMAT, "label": MP.LABEL, "suite": suite, "time_utc": now(), "isolation_utc": iso.get("time_utc"),
           "graph_ids": {str(v): list(k) for k, v in ids.items()}, "per_part": per, "machine_start": m0,
           "machine_end": machine(), "template_params": TEMPLATE_PARAMS, "max_nodes": MAX_NODES,
           "max_colour_dense": MAX_COLOUR_DENSE, "memory_sha256": meta["npz_sha256"],
           "files": {str(q.relative_to(out)): OB.sha256_file(q) for q in files},
           "onboard_sha256": OB.sha256_file(Path(__file__))}
    (out / "onboarding.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=MP.SUITES)
    ap.add_argument("--out", default=str(MP.ONBOARD_DIR))
    ap.add_argument("--assets-root", default=None,
                    help="A65: this machine's LIBERO assets folder, for textures the export names elsewhere")
    a = ap.parse_args(argv)
    if a.assets_root:
        ASSETS_ROOT.append(str(a.assets_root))
    doc = onboard_suite(a.suite, Path(a.out))
    for g, r in doc["per_part"].items():
        print(g, r["type"], r["part"], r["n_nodes"], r["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
