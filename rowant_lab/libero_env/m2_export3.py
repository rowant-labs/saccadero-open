"""M2 library export-v3 (amendment A58, the orchestrator's D-STOVE): export v2 plus every RENDERED geom it left out.

M2 (perceived poses, grounded instruction): perception in the loop. Build side, run once, never at episode time.

Why. ``m2_export`` (export v1/v2) takes only the visual MESH geoms with ``contype = conaffinity = 0``. A geom that both
renders and collides is left out: on these libraries that is the flat stove's base box ``base_vis`` (LIBERO's
``articulated_objects/flat_stove.xml``: a box of half-sizes 0.095 x 0.095 x 0.02 m, group 1, material
``stove_knob_metal``, on the ``burner`` body), which FM1's probe found as the stove's remaining residual (A41's
``frame-tests/m2f2-fm1-cause.json``). The rule here is general, never per object: every geom of a type's bodies that
MuJoCo's offscreen render draws (its group enabled in the render context's ``vopt.geomgroup`` and its effective alpha
above 0, A41's rule) and that export v2 does not hold.

Two commands:

* ``probe`` (venv-libero, ``-I -B``, the simulator; A60)::

      cd <empty folder> && <venv-libero>/bin/python -I -B <repo>/rowant_lab/libero_env/m2_export3.py probe --out JSON

  For each suite and task at DEV state 0 (``seed(0)``, ``reset``, ``set_init_state``, as ``m2_export``), each instance
  of a type not probed yet: every rendered geom of its subtree that v2 does not hold, read from the COMPILED model
  (LIBERO's MJCF): its body (relative name), its own name (instance prefix stripped), MuJoCo type, size, position and
  quaternion in its body frame, rgba, group, contype / conaffinity, and its material (``mat_rgba``, ``texrepeat``,
  ``texuniform``, the texture's path RELATIVE to LIBERO's assets root and its sha256). It also checks that v2's geom
  list of every body equals the compiled model's visual meshes of that body (count and mesh names). Names and
  numbers only.

* ``build`` (any interpreter with numpy; on the MacBook)::

      python -I -B <repo>/rowant_lab/libero_env/m2_export3.py build --probe JSON [--src EXPORT_V2] [--out EXPORT_V3]

  ``export-v3`` = ``export`` (v2) with the probe's geoms ADDED: each primitive is tessellated in its body frame
  (``primitive_arrays``: a box as 6 faces, each face its own vertex grid with the face's normal, cells at most
  ``CELL_M``; other primitive types are refused until needed), appended to the type's npz under the next ``g<k>``
  keys and to its body's ``geoms`` list (``has_uv`` False, its material with the texture named as the v2 export names
  textures: by its path under the exporting machine's LIBERO assets root, sha256-checked there), and the type gains
  ``rendered_geoms_added`` (what was added and why). Every other file (types without added geoms, the workcells, the
  Panda files) is copied BYTE FOR BYTE; a new ``manifest.json`` lists every file's sha256 and the v2 manifest's.
  A version folder is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.settings import DATA  # noqa: E402

FORMAT = "rowant libero_env M2 library export v3"
PROBE_FORMAT = "rowant libero_env M2 export-v3 probe v1"
CELL_M = 0.01  # largest edge of a tessellated face cell
ASSET_KEY = "/libero/libero/assets/"
GEOM_TYPES = {2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}
V2_SRC = DATA / "runs" / "libero-m2" / "library" / "export"
V3_OUT = DATA / "runs" / "libero-m2" / "library" / "export-v3"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rendered(m, i: int, geomgroup) -> bool:
    """A41's rule: the geom's group is enabled in the render context's geomgroup and its effective alpha is above 0
    (the material's alpha when the geom keeps MuJoCo's default rgba)."""
    import numpy as np

    grp = int(m.geom_group[i])
    if geomgroup is not None and not (0 <= grp < len(geomgroup) and geomgroup[grp]):
        return False
    rgba = np.asarray(m.geom_rgba[i], np.float32)
    mat = int(m.geom_matid[i])
    default_rgba = np.array([0.5, 0.5, 0.5, 1.0], np.float32)
    alpha = float(m.mat_rgba[mat][3]) if mat >= 0 and np.allclose(rgba, default_rgba) else float(rgba[3])
    return alpha > 0.0


def in_v2(m, i: int) -> bool:
    """export v1/v2 holds exactly the mesh geoms with contype = conaffinity = 0 (m2_export.export_type)."""
    return int(m.geom_type[i]) == 7 and not m.geom_contype[i] and not m.geom_conaffinity[i]


def rel_asset(path: str | None) -> str | None:
    if not path:
        return None
    if ASSET_KEY not in path:
        raise RuntimeError(f"texture {path} is not under a LIBERO assets root")
    return path.split(ASSET_KEY, 1)[1]


def probe_type(env, inner, name: str, typ: str, v2_types: Path, geomgroup) -> dict:
    from rowant_lab.libero_env import m2_export as EX

    m = env.sim.model
    prefix = name + "_"
    root = int(inner.obj_body_id[name])
    bodies = EX.subtree(m, root)
    v2 = json.loads((v2_types / f"{typ}.json").read_text())
    added, checks = [], {}
    for b in bodies:
        bn = m.body_id2name(b)
        rel = bn[len(prefix):] if bn.startswith(prefix) else bn
        have = [g["mesh"] for g in v2["bodies"][rel]["geoms"]]
        want = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != b:
                continue
            if in_v2(m, g):
                want.append((m.mesh_id2name(int(m.geom_dataid[g])) or "")[len(prefix):])
                continue
            if not rendered(m, g, geomgroup):
                continue
            gt = int(m.geom_type[g])
            gname = m.geom_id2name(g) or f"geom{g}"
            gname = gname[len(prefix):] if gname.startswith(prefix) else gname
            mat = int(m.geom_matid[g])
            mrec = {"rgba": EX._l(m.geom_rgba[g], 6)}
            if mat >= 0:
                tex = int(m.mat_texid[mat][1])  # mjTEXROLE_RGB, as m2_export
                mrec = {"mat_rgba": EX._l(m.mat_rgba[mat], 6), "texrepeat": EX._l(m.mat_texrepeat[mat], 6),
                        "texuniform": bool(m.mat_texuniform[mat])}
                if tex >= 0:
                    tp = EX.tex_path(m, tex)
                    mrec.update(texture_rel=rel_asset(tp), texture_sha256=EX.sha256_file(tp) if tp else None,
                                tex_width=int(m.tex_width[tex]), tex_height=int(m.tex_height[tex]),
                                tex_type=int(m.tex_type[tex]))
            added.append({"body": rel, "name": gname, "geom_type": GEOM_TYPES.get(gt, str(gt)),
                          "size": EX._l(m.geom_size[g]), "pos": EX._l(m.geom_pos[g]), "quat": EX._l(m.geom_quat[g]),
                          "group": int(m.geom_group[g]), "contype": int(m.geom_contype[g]),
                          "conaffinity": int(m.geom_conaffinity[g]), "material": mrec,
                          "mesh": (m.mesh_id2name(int(m.geom_dataid[g])) or "")[len(prefix):] if gt == 7 else None})
        checks[rel] = {"v2": have, "compiled_visual_meshes": want, "equal": have == want}
    return {"type": typ, "added": added, "v2_body_check": checks,
            "v2_body_check_ok": all(c["equal"] for c in checks.values())}


def cmd_probe(a) -> int:
    if not sys.flags.isolated:
        print("REFUSE: run with python -I", file=sys.stderr)
        return 2
    from rowant_lab.libero_env import m2_rpaths as MP

    MP.pin_numba_cache()
    from rowant_lab.hybrid_frontier.e9 import isolation

    g = isolation.scan()
    if not g["clear"]:
        print("REFUSE: isolation guard not clear", file=sys.stderr)
        return 3
    from rowant_lab.libero_env.export_geom import instance_types, start_state

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    out = Path(a.out)
    if out.exists():
        print(f"REFUSE: {out} exists", file=sys.stderr)
        return 2
    src = Path(a.src) if a.src else MP.LIBRARY / "export"
    t0 = time.perf_counter()
    done: dict = {}
    for suite in MP.SUITES:
        bench = benchmark.get_benchmark_dict()[suite](task_order_index=0)
        for ti in range(MP.N_TASKS):
            env = OffScreenRenderEnv(bddl_file_name=bench.get_task_bddl_file_path(ti), use_camera_obs=False)
            env.seed(0)
            env.reset()
            env.set_init_state(start_state(bench, ti, 0))
            ctx = getattr(env.sim, "_render_context_offscreen", None)
            gg = [int(x) for x in ctx.vopt.geomgroup] if ctx is not None else None
            inner = env.env
            it = instance_types(inner)
            for name in list(inner.objects_dict) + list(inner.fixtures_dict):
                typ = it.get(name, name)
                if typ in done:
                    continue
                rec = probe_type(env, inner, name, typ, src / "types", gg)
                rec["geomgroup"] = gg
                done[typ] = rec
            env.close()
    doc = {"format": PROBE_FORMAT, "label": MP.LABEL, "time_utc": now(), "isolation_utc": g["time_utc"],
           "src_manifest_sha256": sha256_file(src / "manifest.json"), "probe_sha256": sha256_file(Path(__file__)),
           "states_used": "DEV state 0 only", "types": dict(sorted(done.items())),
           "n_added": sum(len(r["added"]) for r in done.values()),
           "v2_body_check_ok": all(r["v2_body_check_ok"] for r in done.values()),
           "elapsed_s": round(time.perf_counter() - t0, 2)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(json.dumps({"n_added": doc["n_added"], "v2_body_check_ok": doc["v2_body_check_ok"],
                      "added": {t: [x["name"] for x in r["added"]] for t, r in done.items() if r["added"]}}))
    return 0


# ------------------------------------------------------------------------------------------------ build
def quat_to_mat(q):
    import numpy as np

    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _grid_face(c, u, v, nu: int, nv: int):
    """A planar face centred at c spanned by +-u and +-v as an (nu x nv)-cell grid: vertices and triangles, wound so
    that (u x v) is the outward normal."""
    import numpy as np

    s = np.linspace(-1.0, 1.0, nu + 1)
    t = np.linspace(-1.0, 1.0, nv + 1)
    S, T = np.meshgrid(s, t, indexing="ij")
    V = c[None, :] + S.reshape(-1, 1) * u[None, :] + T.reshape(-1, 1) * v[None, :]
    F = []
    for i in range(nu):
        for j in range(nv):
            a, b = i * (nv + 1) + j, (i + 1) * (nv + 1) + j
            F += [[a, b, b + 1], [a, b + 1, a + 1]]
    return V, np.asarray(F, np.int64)


def primitive_arrays(geom_type: str, size, cell_m: float = CELL_M):
    """(V, F, N) of a primitive in its GEOM frame. A box: 6 faces, each its own vertex grid (no vertex shared across
    faces) with that face's outward normal, cells at most ``cell_m``. Other types are refused until a library needs
    them."""
    import math

    import numpy as np

    if geom_type != "box":
        raise NotImplementedError(f"primitive {geom_type!r}: not needed by these libraries yet")
    hx, hy, hz = (float(x) for x in size[:3])
    ex = np.eye(3)
    Vs, Fs, Ns, off = [], [], [], 0
    for ax, h, (ua, ha), (va, hb) in ((0, hx, (1, hy), (2, hz)), (1, hy, (2, hz), (0, hx)),
                                      (2, hz, (0, hx), (1, hy))):
        for sgn in (1.0, -1.0):
            n = sgn * ex[ax]
            u = ex[ua] * ha
            v = ex[va] * hb * sgn  # flip v on the negative face so that u x v stays outward
            V, F = _grid_face(n * h, u, v, max(1, math.ceil(2 * ha / cell_m)), max(1, math.ceil(2 * hb / cell_m)))
            if float(np.dot(np.cross(u, v), n)) < 0:
                F = F[:, ::-1]
            Vs.append(V)
            Fs.append(F + off)
            Ns.append(np.repeat(n[None, :], len(V), axis=0))
            off += len(V)
    return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Ns)


def body_frame_arrays(rec: dict):
    """The added geom's (V, F, N) in its BODY frame (geom_pos / geom_quat applied, as m2_export.mesh_arrays)."""
    import numpy as np

    V, F, N = primitive_arrays(rec["geom_type"], rec["size"])
    Rg = quat_to_mat(rec["quat"])
    pg = np.asarray(rec["pos"], float)
    return V @ Rg.T + pg, F, N @ Rg.T


def canonical_texture(src_types: Path, rel: str | None, sha: str | None) -> str | None:
    """The texture's path as the v2 export names textures: the exporting machine's LIBERO assets root (read from a
    v2 type document's own texture paths), joined with ``rel``; sha256-checked there."""
    if not rel:
        return None
    root = None
    for p in sorted(src_types.glob("*.json")):
        doc = json.loads(p.read_text())
        for b in doc["bodies"].values():
            for g in b["geoms"]:
                tp = (g.get("material") or {}).get("texture")
                if tp and ASSET_KEY in tp:
                    root = tp.split(ASSET_KEY, 1)[0] + ASSET_KEY
                    break
            if root:
                break
        if root:
            break
    if root is None:
        raise RuntimeError("no v2 texture path names the exporting machine's LIBERO assets root")
    path = root + rel
    if sha and sha256_file(path) != sha:
        raise RuntimeError(f"{path}: sha256 differs from the probe's {sha}")
    return path


def cmd_build(a) -> int:
    import numpy as np

    probe = json.loads(Path(a.probe).read_text())
    if probe.get("format") != PROBE_FORMAT or not probe.get("v2_body_check_ok"):
        print("REFUSE: not an export-v3 probe, or its v2 body check failed", file=sys.stderr)
        return 2
    src = Path(a.src) if a.src else V2_SRC
    out = Path(a.out) if a.out else V3_OUT
    if out.exists():
        print(f"REFUSE: {out} exists (a library version is never overwritten)", file=sys.stderr)
        return 2
    if sha256_file(src / "manifest.json") != probe["src_manifest_sha256"]:
        print("REFUSE: the probe read another export manifest", file=sys.stderr)
        return 2
    src_man = json.loads((src / "manifest.json").read_text())
    for rel, h in src_man["files"].items():  # every v2 file as its manifest says
        if sha256_file(src / rel) != h:
            print(f"REFUSE: export/{rel} differs from its manifest", file=sys.stderr)
            return 2
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        print(f"REFUSE: {tmp} exists", file=sys.stderr)
        return 2
    added_by_type = {t: r["added"] for t, r in probe["types"].items() if r["added"]}
    changed = {}
    for rel in sorted(src_man["files"]):
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        typ = Path(rel).stem if rel.startswith("types/") else None
        if typ in added_by_type and rel.endswith(".json"):
            continue  # written below with its npz
        if typ in added_by_type and rel.endswith(".npz"):
            continue
        shutil.copyfile(src / rel, tmp / rel)
    for typ, adds in sorted(added_by_type.items()):
        doc = json.loads((src / "types" / f"{typ}.json").read_text())
        with np.load(src / "types" / doc["npz"], allow_pickle=False) as z:
            arrays = {k: np.asarray(z[k]) for k in z.files}
        n_keys = len({k.split("__")[0] for k in arrays})
        recs = []
        for k, rec in enumerate(adds):
            V, F, N = body_frame_arrays(rec)
            key = f"g{n_keys + k}"
            arrays[f"{key}__V"], arrays[f"{key}__F"], arrays[f"{key}__N"] = V, F, N
            arrays[f"{key}__UV"] = np.zeros((0, 2))
            mat = dict(rec["material"])
            mrec = {kk: mat[kk] for kk in ("rgba", "mat_rgba", "texrepeat", "texuniform") if kk in mat}
            if mat.get("texture_rel"):
                mrec.update(texture=canonical_texture(src / "types", mat["texture_rel"], mat.get("texture_sha256")),
                            texture_sha256=mat.get("texture_sha256"), tex_width=mat.get("tex_width"),
                            tex_height=mat.get("tex_height"), tex_type=mat.get("tex_type"))
            doc["bodies"][rec["body"]]["geoms"].append(
                {"key": key, "n_vertices": int(len(V)), "n_faces": int(len(F)), "has_uv": False, "material": mrec,
                 "mesh": rec["name"], "primitive": {"type": rec["geom_type"], "size": rec["size"],
                                                    "pos": rec["pos"], "quat": rec["quat"]}})
            recs.append({"body": rec["body"], "geom": rec["name"], "key": key, "primitive": rec["geom_type"],
                         "size": rec["size"], "group": rec["group"], "contype": rec["contype"],
                         "conaffinity": rec["conaffinity"],
                         "why": "rendered by MuJoCo's offscreen render (group enabled, alpha > 0) and left out of "
                                "export v2, which held only visual meshes with contype = conaffinity = 0 (A58)"})
        np.savez(tmp / "types" / doc["npz"], **arrays)
        doc["format"] = FORMAT
        doc["npz_sha256"] = sha256_file(tmp / "types" / doc["npz"])
        doc["rendered_geoms_added"] = recs
        (tmp / "types" / f"{typ}.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
        changed[typ] = recs
    files = sorted(p for p in tmp.rglob("*") if p.is_file())
    man = {"format": FORMAT, "label": src_man.get("label"), "time_utc": now(), "amendment": "A58 (D-STOVE)",
           "builder_sha256": sha256_file(Path(__file__)), "src": "export (v2)",
           "src_manifest_sha256": probe["src_manifest_sha256"], "probe_sha256": sha256_file(a.probe),
           "probe_time_utc": probe["time_utc"], "types_changed": changed,
           # v2's own export records, carried as they are (m2_onboard reads seconds_per_type)
           "seconds_per_type": src_man.get("seconds_per_type"), "src_exporter_sha256": src_man.get("exporter_sha256"),
           "src_time_utc": src_man.get("time_utc"),
           "rule": "v2 plus every geom of a type's bodies that the offscreen render draws and v2 does not hold; "
                   "every other file copied byte for byte",
           "files": {str(p.relative_to(tmp)): sha256_file(p) for p in files}}
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    same = sorted(rel for rel in src_man["files"] if man["files"].get(rel) == src_man["files"][rel])
    tmp.rename(out)
    print(json.dumps({"out": str(out), "types_changed": {t: [r["geom"] for r in v] for t, v in changed.items()},
                      "n_files": len(man["files"]), "n_identical_to_v2": len(same),
                      "manifest_sha256": sha256_file(out / "manifest.json")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--out", required=True)
    p.add_argument("--src", default=None, help="the v2 export (default: the M2 library's export/)")
    b = sub.add_parser("build")
    b.add_argument("--probe", required=True)
    b.add_argument("--src", default=None)
    b.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    return cmd_probe(a) if a.cmd == "probe" else cmd_build(a)


if __name__ == "__main__":
    sys.exit(main())
