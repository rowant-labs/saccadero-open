"""The library the long-task suite needs (LIBERO-10, ``libero_10``). Build side; ``scripts/build_libraries.py`` runs
its steps in order.

``goal_lib.py`` built the Goal suite's library as a VARIANT TREE with one workcell per suite, as ``m2_library`` holds
them. The long suite's ten base tasks stand at three different tables (a kitchen table, a living-room table, a study
table), so this generalises it: the tasks whose compiled workcell (arena, robot mount and base, camera fields of view)
is the same form one CELL, registered as its own suite key ``libero_long_<table>`` with its combined type list. The
keys pass ``m2_library``'s rules (``name_problems``: ``libero_10`` and any ``_<digits>`` key read as an instance name).
Which task stands in which cell is a build record (``build/cells.json``), never a library field (``m2_library``
refuses per-task keys)::

    long-lib/m1/             goal-lib's m1 (copies: M1's robot.json and records, the bottle's) + the new objects'
                             grasp records
    long-lib/m2/policy-v3/   goal-lib's policy-v3 (copies: M2's files and Goal's) + the new types + one workcell per
                             cell + ``pandas/`` (the Panda of a cell whose robot mount is no M2 suite's)
    long-lib/m2/onboard-v3/  M2's two workcells and Goal's (copies) + one per cell (m2_onboard, unchanged, on a merged
                             export)

Object, Spatial and Goal files are byte for byte goal-lib's (whose Object and Spatial files are M2's). Steps, from the
repository's root (build files in ``long-lib/build``)::

    <worker python> -I -B rowant_lab/pro2/long_lib.py export
    PYTHONPATH=$PWD <server python> -P -m rowant_lab.pro2.long_lib policy
    PYTHONPATH=$PWD <server python> -P -m rowant_lab.pro2.long_lib onboard

``export`` creates each BASE task (ids 0-9) at start state 0 through the service's own adapter (``dev_episode``) with
Pro2.0's simulator, LIBERO's API and its own config (``libero_config_text``: the simulator's own task files and the
payload's start states). No image is saved; no perturbation file, manifest or evaluation archive is read. Per type:

* a type goal-lib's library holds must be the same object (``m1_primitives_same``, ``export_same``: collision
  primitives within 1e-6, meshes, regions, kinematics and export-v3's rendered additions);
* a new type is exported as M2's export-v3 was built: ``m2_export.export_type`` (the visual meshes; the non-mesh visual
  geoms it refuses are kept aside by ``_meshes_only``), then ``m2_export3.probe_type`` and its rule (``add_rendered``:
  every rendered geom the meshes leave out, tessellated, boxes as ``m2_export3`` and capsules by ``capsule_arrays``);
* a type that cannot be exported is a recorded gap (``failed``), not a dead build.

Each cell's Panda is compared with M2's two exported Pandas (``panda_diff``); a cell with neither gets its own export
(``build/export/panda-<cell>``). ``policy`` leaves out a new type whose documents ``m2_library.problems`` refuses (an
``excluded`` record: the frying pan's own name reads as an instance name).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import types as _types
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.pro2 import goal_lib as GL  # noqa: E402  (scene_of, m1_type, close; the goal-lib tree)
from rowant_lab.pro2 import goal_paths as GP  # noqa: E402
from rowant_lab.settings import DATA  # noqa: E402

LL = DATA / "runs" / "pro2" / "long-lib"
BUILD = LL / "build"
PAYLOAD = GL.PAYLOAD
SIM = PAYLOAD / "simulator-source" / "libero" / "libero"  # Pro2.0's simulator package (source and assets)
GOAL = GL.G  # goal-lib's variant tree: its Object, Spatial and Goal files are copied byte for byte
M2LIB, M1LIB = GL.M2LIB, GL.M1LIB
ORIG_BDDL = DATA / "datasets" / "libero" / "LIBERO-8f1084e" / "libero" / "libero" / "bddl_files"
SUITE = "libero_10"
N_TASKS = 10
STATE = 0
SEED = 142  # as goal_lib's export (a base scene's start state fixes every pose)
MAX_STEPS = 520  # the suite's step cap
CAMERA_PX = 64  # the adapter's observation images (never saved or read)
CELL_PREFIX = "libero_long_"
CELL_FORMAT = "rowant pro2 long-suite workcell v1"
TOL = GL.TOL
PANDA_TOL = 1e-5  # goal_lib's Panda comparison
NAME_RE = re.compile(r"^[a-z]+(?:_[a-z]+)*$")


def sha256_file(path) -> str:
    return GL.sha256_file(path)


# ------------------------------------------------------------------------------------------------ pure parts
def libero_config_text() -> str:
    """LIBERO's config for Pro2.0's simulator: its own assets and task files and the payload's start states 0-9.
    ``dev_setup.sh``'s pod config names the service's task folder instead, which holds no file at the API's base path
    (``<root>/libero_10/<task>.bddl``); the simulator's own base files are byte for byte LIBERO's (``export``
    records it)."""
    return (f'assets: "{SIM}/assets"\n'
            f'bddl_files: "{SIM}/bddl_files"\n'
            f'benchmark_root: "{SIM}"\n'
            f'datasets: "{PAYLOAD}/simulator-source/libero/datasets"\n'
            f'init_states: "{SIM}/init_files"\n')


def is_table(typ: str) -> bool:
    """The workspace's table (``table``, ``kitchen_table``, ...): part of the arena, never a library type (LIBERO
    keeps it out of ``fixtures_dict``; this is the guard)."""
    return typ == "table" or typ.endswith("_table")


def name_problems(name: str) -> list:
    """Why ``name`` cannot key a cell. A cell is a suite key of the variant library: ``m2_library`` refuses a document
    whose strings read as an instance name (``_<digits>``) or a task reference, and an extra suite that is M2's."""
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.libero_env import m2_paths as MP

    out = []
    if not NAME_RE.match(name):
        out.append("not lower-case words joined by underscores")
    if LB.INSTANCE_NAME.search(name):
        out.append("reads as an instance name")
    if LB.TASK_REF.search(name):
        out.append("reads as a task reference")
    if name in MP.WORKCELL_TYPES or name in (GP.SUITE, SUITE):
        out.append("another suite's key")
    if not name.startswith(CELL_PREFIX):
        out.append(f"not under {CELL_PREFIX!r}")
    return out


def group_by_workcell(rows: list, tol: float = TOL) -> list:
    """``[(workcell, [task ids])]`` in order of each group's first task: the tasks whose workcells are equal within
    ``tol`` (``goal_lib.close``). ``rows``: ``{"task", "workcell"}`` in task order."""
    groups: list = []
    for r in rows:
        for wc, tasks in groups:
            if GL.close(wc, r["workcell"], tol):
                tasks.append(r["task"])
                break
        else:
            groups.append((r["workcell"], [r["task"]]))
    return groups


def cell_names(workspaces: list) -> list:
    """One name per group from the tables its tasks stand at (``workspace_name``, ``_table`` dropped):
    ``libero_long_<table>``; ``_a``, ``_b``, ... in group order when several groups stand at the same table(s)."""
    bases = []
    for wss in workspaces:
        words = sorted({w[:-len("_table")] if w.endswith("_table") else w for w in wss})
        bases.append(CELL_PREFIX + "_".join(words))
    out = []
    for i, b in enumerate(bases):
        out.append(b if bases.count(b) == 1 else f"{b}_{'abcdefghijklmnopqrstuvwxyz'[bases[:i].count(b)]}")
    return out


def _geom_cut(body: dict) -> list:
    keep = ("key", "n_vertices", "n_faces", "has_uv", "mesh")
    return [{k: v for k, v in g.items() if k in keep} for g in body["geoms"]]


def export_same(old: dict, new: dict, added: list, tol: float = TOL) -> dict:
    """Is a type's fresh export (``new``: v2's visual meshes; ``added``: the probe's rendered geoms) the library's
    (``old``: export-v3)? goal_lib's check (the same bodies and regions, each body's meshes as a prefix of the
    library's, the regions' body-frame pose and size), made exact for v3: each body holds exactly its v2 meshes plus
    the library's additions, the additions are the probe's (body, geom, primitive, size, pose), and the kinematics
    (joints, non-root body offsets) agree."""
    out = {"bodies": set(old["bodies"]) == set(new["bodies"]), "sites": set(old["sites"]) == set(new["sites"])}
    old_added = old.get("rendered_geoms_added") or []
    n_old_added = {}
    for x in old_added:
        n_old_added[x["body"]] = n_old_added.get(x["body"], 0) + 1
    if out["bodies"]:
        out["meshes"] = all(
            len(old["bodies"][r]["geoms"]) == len(new["bodies"][r]["geoms"]) + n_old_added.get(r, 0)
            and _geom_cut(old["bodies"][r])[:len(new["bodies"][r]["geoms"])] == _geom_cut(new["bodies"][r])
            for r in new["bodies"])
        out["kinematics"] = all(
            GL.close(old["bodies"][r]["joints"], new["bodies"][r]["joints"], tol)
            and old["bodies"][r]["parent"] == new["bodies"][r]["parent"]
            and (new["bodies"][r]["is_root"]
                 or GL.close([old["bodies"][r]["body_pos"], old["bodies"][r]["body_quat"]],
                             [new["bodies"][r]["body_pos"], new["bodies"][r]["body_quat"]], tol))
            for r in new["bodies"])
    if out["sites"]:
        def f(s):
            return {k: {x: v[x] for x in ("pos", "R", "size", "body")} for k, v in s.items()}

        out["site_poses"] = GL.close(f(old["sites"]), f(new["sites"]), tol)
    prim = {}
    for g in (g for b in old["bodies"].values() for g in b["geoms"] if "primitive" in g):
        prim[g["mesh"]] = g["primitive"]
    a = [[x["body"], x["geom"], x["primitive"], x["size"], (prim.get(x["geom"]) or {}).get("pos"),
          (prim.get(x["geom"]) or {}).get("quat")] for x in old_added]
    b = [[x["body"], x["name"], x["geom_type"], x["size"], x["pos"], x["quat"]] for x in added]
    out["rendered_added"] = GL.close(a, b, tol)
    out["same"] = all(out.values())
    return out


def panda_diff(a: dict, b: dict, tol: float = PANDA_TOL) -> list:
    """The bodies of two exported Pandas (``m2_export.panda_model``) whose parent, body_pos or body_quat differ
    (goal_lib's comparison), and the bodies only one of them has."""
    out = sorted(set(a) ^ set(b))
    for k in sorted(set(a) & set(b)):
        if not GL.close({x: a[k][x] for x in ("parent", "body_pos", "body_quat")},
                        {x: b[k][x] for x in ("parent", "body_pos", "body_quat")}, tol):
            out.append(k)
    return out


def capsule_arrays(size, cell_m: float):
    """(V, F, N) of a MuJoCo capsule (radius ``size[0]``, cylinder half-length ``size[1]`` along z) in its GEOM frame:
    a surface of revolution (bottom pole, rings, top pole) with ring spacing and arc steps of at most ``cell_m`` (as
    ``m2_export3.primitive_arrays``' box cells), every face wound so that its normal points out, the vertex normals
    exact (the sphere's or the cylinder's)."""
    import numpy as np

    r, h = float(size[0]), float(size[1])
    na = max(8, math.ceil(2.0 * math.pi * r / cell_m))
    nc = max(2, math.ceil(0.5 * math.pi * r / cell_m))
    nl = max(1, math.ceil(2.0 * h / cell_m)) if h > 0 else 0
    prof = []  # (rho, z, n_rho, n_z) per ring, bottom to top, the poles apart
    for i in range(1, nc + 1):
        a = -0.5 * math.pi + 0.5 * math.pi * i / nc
        prof.append((r * math.cos(a), -h + r * math.sin(a), math.cos(a), math.sin(a)))
    for j in range(1, nl):
        prof.append((r, -h + 2.0 * h * j / nl, 1.0, 0.0))
    for i in range(0 if h > 0 else 1, nc):
        a = 0.5 * math.pi * i / nc
        prof.append((r * math.cos(a), h + r * math.sin(a), math.cos(a), math.sin(a)))
    th = 2.0 * math.pi * np.arange(na) / na
    c, s = np.cos(th), np.sin(th)
    V, N = [np.array([[0.0, 0.0, -h - r]])], [np.array([[0.0, 0.0, -1.0]])]
    for rho, z, nr, nz in prof:
        V.append(np.stack([rho * c, rho * s, np.full(na, z)], 1))
        N.append(np.stack([nr * c, nr * s, np.full(na, nz)], 1))
    V.append(np.array([[0.0, 0.0, h + r]]))
    N.append(np.array([[0.0, 0.0, 1.0]]))
    nring, top = len(prof), 1 + len(prof) * na

    def ring(i, k):
        return 1 + i * na + (k % na)

    F = []
    for k in range(na):
        F.append([0, ring(0, k + 1), ring(0, k)])
        for i in range(nring - 1):
            a0, b0, c0, d0 = ring(i, k), ring(i, k + 1), ring(i + 1, k), ring(i + 1, k + 1)
            F += [[a0, b0, c0], [b0, d0, c0]]
        F.append([ring(nring - 1, k), ring(nring - 1, k + 1), top])
    return np.concatenate(V), np.asarray(F, np.int64), np.concatenate(N)


def primitive_arrays(geom_type: str, size, cell_m: float | None = None):
    """(V, F, N) of a primitive in its GEOM frame: a box as ``m2_export3.primitive_arrays``, a capsule as
    ``capsule_arrays``; other types are refused until a library needs them (as ``m2_export3``)."""
    from rowant_lab.libero_env import m2_export3 as E3

    cell = E3.CELL_M if cell_m is None else float(cell_m)
    if geom_type == "box":
        return E3.primitive_arrays("box", size, cell)
    if geom_type == "capsule":
        return capsule_arrays(size, cell)
    raise NotImplementedError(f"primitive {geom_type!r}: not needed by these libraries yet")


def body_frame_arrays(rec: dict):
    """A probed geom's (V, F, N) in its BODY frame (``geom_pos`` / ``geom_quat`` applied, as ``m2_export3``)."""
    import numpy as np

    from rowant_lab.libero_env import m2_export3 as E3

    V, F, N = primitive_arrays(rec["geom_type"], rec["size"])
    Rg = E3.quat_to_mat(rec["quat"])
    return V @ Rg.T + np.asarray(rec["pos"], float), F, N @ Rg.T


class _MeshesOnly:
    """The compiled model as ``m2_export.export_type`` reads it, with every NON-mesh visual geom (contype = conaffinity
    = 0) marked as colliding: export v2 holds the visual meshes only and refuses any other visual geom, so those are
    left to export-v3's rule (``probe_type`` on the true model, then ``add_rendered``). Everything else is the
    model's."""

    def __init__(self, m):
        import numpy as np

        ct = np.asarray(m.geom_contype)
        hide = (np.asarray(m.geom_type) != 7) & (ct == 0) & (np.asarray(m.geom_conaffinity) == 0)
        self._m = m
        self.geom_contype = np.where(hide, 1, ct)
        self.hidden = [int(g) for g in np.flatnonzero(hide)]

    def __getattr__(self, k):
        return getattr(self._m, k)


def _meshes_only(env):
    return _types.SimpleNamespace(sim=_types.SimpleNamespace(model=_MeshesOnly(env.sim.model), data=env.sim.data))


def base_types() -> dict:
    """``{type: origin}`` of the types goal-lib's library holds: M2's (``m2_paths``' frozen tables) and Goal's two."""
    from rowant_lab.libero_env import m2_paths as MP

    out = {t: "m2" for s in MP.SUITES for t in MP.WORKCELL_TYPES[s]}
    out.update({t: "goal" for t in GL.NEW_TYPES})
    return out


def base_paths(typ: str, origin: str) -> tuple:
    """(export json, export npz, M1-format type) the library's copy of a shared type was built from."""
    if origin == "m2":
        d = M2LIB / "export-v3" / "types"
        return d / f"{typ}.json", d / f"{typ}.npz", M1LIB / "types" / f"{typ}.json"
    if origin == "goal":
        d = GOAL / "build" / "export" / "types"
        return d / f"{typ}.json", d / f"{typ}.npz", GOAL / "build" / "m1types" / f"{typ}.json"
    raise ValueError(origin)


# ------------------------------------------------------------------------------------------------ export
def add_rendered(doc: dict, added: list, types_dir: Path) -> dict:
    """export-v3's rule (``m2_export3.cmd_build``) on one fresh type: each probed geom tessellated in its body frame,
    appended under the next ``g<k>`` keys to the type's npz and to its body's geoms (its material as the probe read it,
    the texture under this machine's simulator assets, sha256-checked), and the type gains ``rendered_geoms_added``."""
    import numpy as np

    from rowant_lab.libero_env import m2_export3 as E3

    npz = types_dir / doc["npz"]
    with np.load(npz, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    n_keys = len({k.split("__")[0] for k in arrays})
    doc = json.loads(json.dumps(doc))
    recs = []
    for k, rec in enumerate(added):
        V, F, N = body_frame_arrays(rec)
        key = f"g{n_keys + k}"
        arrays[f"{key}__V"], arrays[f"{key}__F"], arrays[f"{key}__N"] = V, F, N
        arrays[f"{key}__UV"] = np.zeros((0, 2))
        mat = dict(rec["material"])
        mrec = {kk: mat[kk] for kk in ("rgba", "mat_rgba", "texrepeat", "texuniform") if kk in mat}
        if mat.get("texture_rel"):
            path = f"{SIM}/assets/{mat['texture_rel']}"
            if mat.get("texture_sha256") and sha256_file(path) != mat["texture_sha256"]:
                raise RuntimeError(f"{path}: sha256 differs from the probe's")
            mrec.update(texture=path, texture_sha256=mat.get("texture_sha256"), tex_width=mat.get("tex_width"),
                        tex_height=mat.get("tex_height"), tex_type=mat.get("tex_type"))
        doc["bodies"][rec["body"]]["geoms"].append(
            {"key": key, "n_vertices": int(len(V)), "n_faces": int(len(F)), "has_uv": False, "material": mrec,
             "mesh": rec["name"], "primitive": {"type": rec["geom_type"], "size": rec["size"], "pos": rec["pos"],
                                                "quat": rec["quat"]}})
        recs.append({"body": rec["body"], "geom": rec["name"], "key": key, "primitive": rec["geom_type"],
                     "size": rec["size"], "group": rec["group"], "contype": rec["contype"],
                     "conaffinity": rec["conaffinity"],
                     "why": "rendered by MuJoCo's offscreen render (group enabled, alpha > 0) and left out of export "
                            "v2, which holds only visual meshes with contype = conaffinity = 0 (A58's rule)"})
    np.savez(npz, **arrays)
    doc.update(format=E3.FORMAT, npz_sha256=sha256_file(npz), rendered_geoms_added=recs)
    return doc


def export_new(env, inner, name: str, typ: str, types_dir: Path, scene: dict, gg, m1doc: dict) -> dict:
    """A type the library lacks, from instance ``name``: export v2 + v3's rendered additions into ``types_dir``."""
    from rowant_lab.libero_env import m2_export as EX
    from rowant_lab.libero_env import m2_export3 as E3

    ex = EX.export_type(_meshes_only(env), inner, name, typ, types_dir, scene, name + "_")
    (types_dir / f"{typ}.json").write_text(json.dumps(ex, indent=1, sort_keys=True))
    probe = E3.probe_type(env, inner, name, typ, types_dir, gg)
    if not probe["v2_body_check_ok"]:
        raise RuntimeError(f"{typ}: export v2's meshes are not the compiled visual meshes")
    if probe["added"]:
        ex = add_rendered(ex, probe["added"], types_dir)
        (types_dir / f"{typ}.json").write_text(json.dumps(ex, indent=1, sort_keys=True))
    return {"kind": ex["kind"], "bodies": sorted(ex["bodies"]), "parts": sorted(ex["parts"]),
            "joints": {p: (v["joint"] or {}).get("type") for p, v in ex["parts"].items() if v["joint"]},
            "fixed_joints": sorted(ex["fixed_joints"]), "sites": sorted(ex["sites"]),
            "n_meshes": sum(len(b["geoms"]) for b in ex["bodies"].values()),
            "rendered_added": [[x["body"], x["name"], x["geom_type"]] for x in probe["added"]],
            "mesh_check": probe["v2_body_check_ok"], "rest_up": m1doc["rest_up_in_body"],
            "n_primitives": sum(len(b["primitives"]) for b in m1doc["bodies"].values())}


def check_shared(env, inner, name: str, typ: str, origin: str, scene: dict, gg, m1doc: dict) -> dict:
    """A type goal-lib's library holds: is this scene's the same object (``export_same``, M1's primitives)?"""
    from rowant_lab.libero_env import m2_export as EX
    from rowant_lab.libero_env import m2_export3 as E3

    ex_path, _npz, m1_path = base_paths(typ, origin)
    scratch = BUILD / "scratch-shared"
    ex = EX.export_type(_meshes_only(env), inner, name, typ, scratch, scene, name + "_")
    (scratch / f"{typ}.json").write_text(json.dumps(ex))
    probe = E3.probe_type(env, inner, name, typ, scratch, gg)
    chk = export_same(json.loads(ex_path.read_text()), ex, probe["added"])
    same_m1 = GL.close(json.loads(m1_path.read_text())["bodies"], m1doc["bodies"])
    return {"origin": origin, "kind": ex["kind"], "m1_primitives_same": same_m1, "export": chk,
            "same": bool(same_m1 and chk["same"])}


def cmd_export(a) -> int:
    if not sys.flags.isolated:
        print("REFUSE: run with python -I", file=sys.stderr)
        return 2
    out = BUILD / "export"
    if out.exists():
        print(f"REFUSE: {out} exists", file=sys.stderr)
        return 2
    cfg = BUILD / "libero-config"
    cfg.mkdir(parents=True, exist_ok=True)
    if not (cfg / "config.yaml").exists():
        (cfg / "config.yaml").write_text(libero_config_text())
    if (cfg / "config.yaml").read_text() != libero_config_text():
        print(f"REFUSE: {cfg / 'config.yaml'} is not this build's config", file=sys.stderr)
        return 2
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg)  # venv-libero's .pth points LIBERO at its own config by default
    if "numba" in sys.modules:
        print("REFUSE: numba is already imported", file=sys.stderr)
        return 2
    (BUILD / "numba-cache").mkdir(exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(BUILD / "numba-cache")  # this build's own; M2's cache folder is not touched
    if str(PAYLOAD / "simulator-source") not in sys.path:  # Pro2.0's simulator shadows our LIBERO
        sys.path[0:0] = [str(PAYLOAD / "simulator-source"), str(PAYLOAD / "service-source")]
    from rowant_lab.hybrid_frontier.e9 import isolation

    iso = isolation.scan()
    if not iso["clear"]:
        print("REFUSE: isolation guard not clear", file=sys.stderr)
        return 3
    import libero.libero as LIB

    if not Path(LIB.__file__).resolve().is_relative_to(SIM.resolve()) or Path(LIB.config_file) != cfg / "config.yaml":
        print(f"REFUSE: LIBERO is {LIB.__file__} with {LIB.config_file}", file=sys.stderr)
        return 2
    from libero.libero import benchmark

    from rowant_lab.libero_env import m2_export as EX
    from rowant_lab.libero_env import m2_paths as MP
    from rowant_lab.libero_env.export_geom import instance_types
    from rowant_lab.pro2 import dev_episode as DE

    import numpy as np

    t0 = time.perf_counter()
    bench = benchmark.get_benchmark_dict()[SUITE]()  # task order 0, as the service's adapter builds it
    base = base_types()
    types_dir, m1out = out / "types", BUILD / "m1types"
    types_dir.mkdir(parents=True)
    m1out.mkdir(parents=True, exist_ok=True)
    report: dict = {"new": {}, "shared": {}, "failed": {}, "repeat_mismatch": {}}
    rows, pandas, seconds, first = [], {}, {}, {}
    for ti in range(N_TASKS):
        task = bench.get_task(ti)
        bddl = Path(bench.get_task_bddl_file_path(ti))
        init = SIM / "init_files" / task.problem_folder / task.init_states_file
        if not (bddl.is_file() and bddl.resolve().is_relative_to((SIM / "bddl_files" / SUITE).resolve())):
            print(f"REFUSE: task {ti}: the API's base task file {bddl} is missing", file=sys.stderr)
            return 2
        ad = DE.adapter_class()()
        ep = {"suite": SUITE, "task": ti, "case": "BASE", "state": STATE, "seed": SEED, "bddl": str(bddl),
              "max_steps": MAX_STEPS}
        ad.create(DE.env_spec(ep, CAMERA_PX))
        try:
            env, inner = ad._env, ad._env.env
            m, d = env.sim.model, env.sim.data
            scene = GL.scene_of(env, inner)
            it = instance_types(inner)
            ctx = getattr(env.sim, "_render_context_offscreen", None)
            gg = [int(x) for x in ctx.vopt.geomgroup] if ctx is not None else None
            rb = m.body_name2id("robot0_base")
            wc = {"arena": scene["arena"], "robot_mount": scene["robot_mount"],
                  "robot_base": {"pos": EX._l(d.xpos[rb]), "quat_wxyz": EX._l(d.xquat[rb])},
                  "cameras": {c: {"fovy_deg": float(m.cam_fovy[m.camera_name2id(c)])} for c in MP.CAMERAS}}
            pandas[ti] = EX.panda_model(env)
            typs = []
            for name in list(inner.objects_dict) + list(inner.fixtures_dict):
                typ = it.get(name, name)
                if is_table(typ):
                    continue
                typs.append(typ)
                doc = GL.m1_type(env, inner, name, typ)
                if typ in first:  # seen in an earlier task (or this one): the same compiled model?
                    if not GL.close(first[typ]["bodies"], doc["bodies"]):
                        report["repeat_mismatch"].setdefault(typ, []).append(ti)
                    continue
                first[typ] = {"bodies": doc["bodies"], "task": ti}
                t1 = time.perf_counter()
                try:
                    if typ in base:
                        report["shared"][typ] = check_shared(env, inner, name, typ, base[typ], scene, gg, doc)
                    else:
                        report["new"][typ] = export_new(env, inner, name, typ, types_dir, scene, gg, doc)
                        (m1out / f"{typ}.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
                        seconds[typ] = round(time.perf_counter() - t1, 3)
                except Exception as e:  # a type that cannot be exported is a recorded gap, not a dead build
                    report["failed"][typ] = f"{type(e).__name__}: {e}"
                    for ext in ("json", "npz"):
                        (types_dir / f"{typ}.{ext}").unlink(missing_ok=True)
            rows.append({"task": ti, "name": task.name, "language": str(env.language_instruction),
                         "workspace": str(getattr(inner, "workspace_name", "unknown")),
                         "robot_model": type(inner.robots[0].robot_model).__name__, "types": sorted(set(typs)),
                         "bddl": str(bddl.relative_to(SIM)), "bddl_sha256": sha256_file(bddl),
                         # None: no copy of the original LIBERO release on this machine to compare with (optional)
                         "bddl_same_as_libero": (sha256_file(ORIG_BDDL / SUITE / bddl.name) == sha256_file(bddl)
                                                 if (ORIG_BDDL / SUITE / bddl.name).is_file() else None),
                         "init_states_sha256": sha256_file(init), "workcell": wc})
            print(ti, task.name, rows[-1]["workspace"], rows[-1]["robot_model"], flush=True)
        finally:
            ad.close()
    shutil.rmtree(BUILD / "scratch-shared", ignore_errors=True)
    groups = group_by_workcell(rows)
    by_task = {r["task"]: r for r in rows}
    names = cell_names([[by_task[t]["workspace"] for t in tasks] for _wc, tasks in groups])
    bad = {n: name_problems(n) for n in names if name_problems(n)}
    if bad:
        print(f"REFUSE: cell names {bad}", file=sys.stderr)
        return 2
    refs = {s: json.loads((M2LIB / "policy-v3" / "workcells" / f"{s}.json").read_text()) for s in MP.SUITES}
    refs[GP.SUITE] = json.loads((GOAL / "m2" / "policy-v3" / "workcells" / f"{GP.SUITE}.json").read_text())
    m2pandas = {s: json.loads((M2LIB / "export-v3" / f"panda-{s}.json").read_text())["bodies"] for s in MP.SUITES}
    cells = {}
    for (wc, tasks), cell in zip(groups, names, strict=True):
        pb, parr = pandas[tasks[0]]
        diffs = {s: panda_diff(m2pandas[s], pb) for s in MP.SUITES}
        panda_of = next((s for s in MP.SUITES if not diffs[s]), None)
        c = {"tasks": tasks, "workspaces": sorted({by_task[t]["workspace"] for t in tasks}),
             "robot_models": sorted({by_task[t]["robot_model"] for t in tasks}),
             "types": sorted({x for t in tasks for x in by_task[t]["types"]}), "workcell": wc,
             "panda_same_across_tasks": all(not panda_diff(pb, pandas[t][0]) for t in tasks),
             "panda_of": panda_of, "panda_bodies_differing": diffs,
             "workcell_same_as": {s: {k: GL.close(ref[k], wc[k]) for k in wc} for s, ref in refs.items()}}
        if panda_of is None:  # the server's self-filter needs this cell's own Panda (see cmd_policy)
            np.savez(out / f"panda-{cell}.npz", **parr)
            (out / f"panda-{cell}.json").write_text(json.dumps(
                {"format": EX.FORMAT, "suite": cell, "bodies": pb, "npz": f"panda-{cell}.npz",
                 "npz_sha256": sha256_file(out / f"panda-{cell}.npz")}, indent=1, sort_keys=True))
            c["panda_file"] = f"export/panda-{cell}.json"
        cells[cell] = c
    shared_types = {t for c in cells.values() for t in c["types"]} & set(base)
    cdoc = {"suite": SUITE, "state": STATE, "seed": SEED,
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "isolation_utc": iso["time_utc"],
            "builder_sha256": sha256_file(Path(__file__)),
            "libero": {"package": str(Path(LIB.__file__).resolve()), "config": libero_config_text()},
            "tasks": rows, "cells": cells, "new_types": sorted(report["new"]), "shared_types": sorted(shared_types),
            "failed_types": sorted(report["failed"]), "seconds_per_type": seconds}
    (BUILD / "cells.json").write_text(json.dumps(cdoc, indent=1, sort_keys=True))
    (BUILD / "seconds.json").write_text(json.dumps(seconds))
    report["elapsed_s"] = round(time.perf_counter() - t0, 2)
    report["cells"] = {k: {x: v[x] for x in ("tasks", "types", "panda_of", "panda_same_across_tasks", "workspaces",
                                             "robot_models", "workcell_same_as", "panda_bodies_differing")}
                       for k, v in cells.items()}
    (BUILD / "export-report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: report[k] for k in ("cells", "failed", "repeat_mismatch", "elapsed_s")}, indent=1,
                     sort_keys=True))
    print(json.dumps({"shared_same": {t: r["same"] for t, r in report["shared"].items()},
                      "new": {t: {k: r[k] for k in ("kind", "parts", "sites", "rendered_added")}
                              for t, r in report["new"].items()}}, indent=1, sort_keys=True))
    return 0


# ------------------------------------------------------------------------------------------------ policy
def check_goal_tree() -> list:
    """What in goal-lib's tree is not as its manifest says, or not M2's where it holds M2's files."""
    gpol, m2pol = GOAL / "m2" / "policy-v3", M2LIB / "policy-v3"
    gman = json.loads((gpol / "manifest.json").read_text())
    m2man = json.loads((m2pol / "manifest.json").read_text())
    bad = [f"goal {rel}" for rel, h in gman["files"].items() if sha256_file(gpol / rel) != h]
    bad += [f"m2 {rel}" for rel, h in m2man["files"].items() if gman["files"].get(rel) != h]
    bad += [f"m2 in_place {rel}" for rel, h in m2man["in_place"].items() if gman["in_place"].get(rel) != h]
    bad += [f"goal {rel}" for rel, h in gman["in_place"].items() if rel.startswith("m1/")
            and sha256_file(GOAL / rel) != h]
    return bad


def cmd_policy(a) -> int:
    """The variant's m1/ and m2/policy-v3/: goal-lib's bytes plus the long suite's additions under M2's rules.

    A new type whose sanitized documents ``m2_library.problems`` refuses is left out (``excluded``), with every cell
    that holds it. A shared type that is not the library's object, or a document of the variant that ``problems``
    refuses, stops the build. A cell whose robot mount is no M2 suite's gets its Panda under ``pandas/``, listed in
    the manifest: ``m2_library.LibraryReader.panda`` reads M2's export folder only, so the server needs a change to
    serve it (the build report says which)."""
    import numpy as np  # noqa: F401  (numpy is what the grasp records need)

    from rowant_lab.libero_env import grasp as GR
    from rowant_lab.libero_env import m2_libv2 as LV2
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.libero_env import m2_paths as MP

    m1, pol = LL / "m1", LL / "m2" / "policy-v3"
    m1tmp, tmp = m1.with_name("m1.tmp"), pol.with_name("policy-v3.tmp")
    if any(p.exists() for p in (m1, pol, m1tmp, tmp)):
        print("REFUSE: the variant's m1 or policy-v3 (or a .tmp) exists", file=sys.stderr)
        return 2
    cdoc = json.loads((BUILD / "cells.json").read_text())
    rep = json.loads((BUILD / "export-report.json").read_text())
    bad = check_goal_tree()
    if bad:
        print("REFUSE: goal-lib's tree:", bad[:10], file=sys.stderr)
        return 3
    unlike = {t: r for t, r in rep["shared"].items() if not r["same"]}
    unchecked = sorted(set(rep["failed"]) & set(base_types()))
    if unlike or unchecked or rep["repeat_mismatch"]:
        print("REFUSE: shared types unlike the library's or unchecked, or a type compiled differently between tasks:",
              json.dumps({"unlike": unlike, "unchecked": unchecked, "repeat": rep["repeat_mismatch"]})[:2000],
              file=sys.stderr)
        return 3
    m1tmp.mkdir(parents=True)
    shutil.copy2(GOAL / "m1" / "robot.json", m1tmp / "robot.json")
    shutil.copytree(GOAL / "m1" / "records", m1tmp / "records")
    robot = json.loads((m1tmp / "robot.json").read_text())
    jaw = GR.JawModel.from_robot(robot)
    params = GR.RecordParams()
    staged, excluded, built = {}, {}, {}
    for typ in sorted(rep["new"]):
        ex = LV2.sanitize_export_type(json.loads((BUILD / "export" / "types" / f"{typ}.json").read_text()))
        model = json.loads((BUILD / "m1types" / f"{typ}.json").read_text())
        m1d = LV2.sanitize_m1_type(model)
        pr = LB.problems(ex, "type") + LB.problems(m1d, "m1type")
        if pr:
            excluded[typ] = pr[:5]
            continue
        staged[f"types/{typ}.json"], staged[f"m1types/{typ}.json"] = ex, m1d
        if model["kind"] != "object":
            continue
        root = model["bodies"][model["root_body"]]
        rec = GR.build_records(root["primitives"], model["rest_up_in_body"], root["com"], jaw, params, name=typ)
        (m1tmp / "records" / f"{typ}.json").write_text(json.dumps(GR.records_to_json(rec), sort_keys=True) + "\n")
        built[typ] = {"n_records": rec["n_records"], "n_kept": int(len(rec["width"])),
                      "height": round(rec["height"], 4)}
    gman = json.loads((GOAL / "m2" / "policy-v3" / "manifest.json").read_text())
    have = {Path(rel).stem for rel in gman["files"] if rel.startswith("types/")}
    final, pandas = {}, {}
    for cell, c in sorted(cdoc["cells"].items()):
        tps = sorted(t for t in c["types"] if t not in excluded and t not in rep["failed"])
        missing = [t for t in tps if t not in have and f"types/{t}.json" not in staged]
        if missing:
            print(f"REFUSE: {cell}: no library documents for {missing}", file=sys.stderr)
            shutil.rmtree(m1tmp)
            return 3
        final[cell] = tps
        staged[f"workcells/{cell}.json"] = {**c["workcell"], "suite": cell, "types": tps,
                                            "identical_across_tasks": True, "format": CELL_FORMAT, "label": MP.LABEL}
        if c["panda_of"] is None:
            pandas[cell] = BUILD / c["panda_file"]
    kinds = {"types": "type", "m1types": "m1type", "workcells": "workcell"}
    bad = {rel: LB.problems(doc, kinds[rel.split("/")[0]])[:5] for rel, doc in staged.items()
           if LB.problems(doc, kinds[rel.split("/")[0]])}
    if bad:
        shutil.rmtree(m1tmp)
        print("REFUSE:", json.dumps(bad)[:2000], file=sys.stderr)
        return 3
    shutil.copytree(GOAL / "m2" / "policy-v3", tmp)
    man = json.loads((tmp / "manifest.json").read_text())
    for rel, doc in staged.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        b = LV2.dumps(doc)
        p.write_bytes(b)
        man["files"][rel] = hashlib.sha256(b).hexdigest()
    for _cell, src in sorted(pandas.items()):
        pdoc = json.loads(src.read_text())
        for f in (src.name, pdoc["npz"]):
            (tmp / "pandas").mkdir(exist_ok=True)
            shutil.copy2(src.parent / f, tmp / "pandas" / f)
            man["files"][f"pandas/{f}"] = sha256_file(tmp / "pandas" / f)
    for typ in built:
        man["in_place"][f"m1/records/{typ}.json"] = sha256_file(m1tmp / "records" / f"{typ}.json")
    man["variant_of"] = "policy-v3"
    man["variant_base"] = "runs/pro2/goal-lib policy-v3, its files unchanged (M2's and Goal's)"
    man["variant_note"] = ("goal-lib's files unchanged, plus the long-task suite's new types, one workcell per table "
                           "(cells) and the Panda of each cell whose robot mount is no M2 suite's "
                           "(rowant_lab/pro2/long_lib.py)")
    mbad = LB.problems({k: v for k, v in man.items() if k not in ("sources", "in_place")}, "manifest")
    if mbad:
        shutil.rmtree(m1tmp)
        shutil.rmtree(tmp)
        print("REFUSE: manifest", mbad[:5], file=sys.stderr)
        return 3
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    m1tmp.rename(m1)
    tmp.rename(pol)
    out = {"records": built, "excluded": excluded, "failed": rep["failed"], "cells": final,
           "panda_of": {cell: c["panda_of"] for cell, c in cdoc["cells"].items()},
           "own_panda": sorted(pandas), "n_files": len(man["files"]), "n_in_place": len(man["in_place"]),
           "manifest_sha256": sha256_file(pol / "manifest.json")}
    (BUILD / "policy-report.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


# ------------------------------------------------------------------------------------------------ onboard
def library_cells(pol: Path) -> dict:
    """``{cell: workcell}`` of the variant's cells, each sha-checked against its manifest."""
    man = json.loads((pol / "manifest.json").read_text())
    out = {}
    for rel, h in sorted(man["files"].items()):
        if rel.startswith("workcells/") and Path(rel).stem.startswith(CELL_PREFIX):
            if sha256_file(pol / rel) != h:
                raise RuntimeError(f"{rel}: sha256 differs from the manifest")
            out[Path(rel).stem] = json.loads((pol / rel).read_text())
    return out


def _copy_onboarded(src: Path, dst: Path) -> None:
    """A workcell's onboarding, byte for byte, checked against its own ``onboarding.json`` file list."""
    if not dst.exists():
        shutil.copytree(src, dst, symlinks=False)
    doc = json.loads((dst / "onboarding.json").read_text())
    if (dst / "onboarding.json").read_bytes() != (src / "onboarding.json").read_bytes():
        raise RuntimeError(f"{dst}: onboarding.json differs from {src}")
    bad = [rel for rel, h in doc["files"].items() if sha256_file(dst / rel) != h]
    if bad:
        raise RuntimeError(f"{dst}: {bad[:5]} differ from its onboarding.json")


def cmd_onboard(a) -> int:
    """m2_onboard, unchanged, per cell, on an export folder holding every cell type (new ones from this build,
    shared ones as the library built them); M2's two workcells and Goal's copied byte for byte."""
    from rowant_lab.libero_env import m2_onboard as OBM
    from rowant_lab.libero_env import m2_paths as MP

    pol, onb = LL / "m2" / "policy-v3", LL / "m2" / "onboard-v3"
    cells = library_cells(pol)
    todo = [a.cell] if a.cell else sorted(cells)
    if not set(todo) <= set(cells):
        print(f"REFUSE: {sorted(set(todo) - set(cells))} not cells of the library", file=sys.stderr)
        return 2
    base = base_types()
    merged = BUILD / "export-merged"
    (merged / "types").mkdir(parents=True, exist_ok=True)
    for typ in sorted({t for c in cells.values() for t in c["types"]}):
        if typ in base:
            js, npz, _m1 = base_paths(typ, base[typ])
        else:
            js, npz = BUILD / "export" / "types" / f"{typ}.json", BUILD / "export" / "types" / f"{typ}.npz"
        for src in (js, npz):
            dst = merged / "types" / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            if sha256_file(dst) != sha256_file(src):
                raise RuntimeError(f"{dst} differs from {src}")
    secs = json.loads((M2LIB / "export-v3" / "manifest.json").read_text()).get("seconds_per_type", {})
    secs.update(json.loads((GOAL / "build" / "seconds.json").read_text()))
    secs.update(json.loads((BUILD / "seconds.json").read_text()))
    (merged / "manifest.json").write_text(json.dumps({"seconds_per_type": secs}))
    onb.mkdir(parents=True, exist_ok=True)
    for s in MP.SUITES:
        _copy_onboarded(M2LIB / "onboard-v3" / s, onb / s)
    _copy_onboarded(GOAL / "m2" / "onboard-v3" / GP.SUITE, onb / GP.SUITE)
    OBM.MP.EXPORT_DIR = merged  # build side only, in this process: the module reads this name
    # build side only, in this process: m2_onboard reads a workcell's types from m2_paths' frozen table
    OBM.MP.WORKCELL_TYPES = {**OBM.MP.WORKCELL_TYPES, **{c: tuple(w["types"]) for c, w in cells.items()}}
    OBM.graph_ids.__defaults__ = (merged,)
    OBM.parts_of_suite.__defaults__ = (merged,)
    for cell in todo:
        if (onb / cell / "onboarding.json").exists():
            print(f"{cell}: onboarded already", flush=True)
            continue
        if (onb / cell).exists():
            print(f"REFUSE: {onb / cell} holds a partial onboarding", file=sys.stderr)
            return 2
        t0 = time.perf_counter()
        doc = OBM.onboard_suite(cell, onb)
        for g, r in doc["per_part"].items():
            print(cell, g, r["type"], r["part"], r["n_nodes"], r["seconds"].get("total"), flush=True)
        print(cell, "seconds", round(time.perf_counter() - t0, 1), flush=True)
    return 0


def cmd_config(a) -> int:
    """Write LIBERO's config for the benchmark's simulator (``libero_config_text``) as ``<--out>/config.yaml``: the
    folder ``LIBERO_CONFIG_PATH`` names for every simulator-side build step."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(libero_config_text())
    print(out / "config.yaml")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("export", "policy", "onboard", "config"))
    ap.add_argument("--cell", default=None, help="onboard: one cell only (default: every cell not onboarded yet)")
    ap.add_argument("--out", default=None, help="config: the folder to write config.yaml into")
    a = ap.parse_args(argv)
    if a.cmd == "config" and not a.out:
        ap.error("config needs --out")
    return {"export": cmd_export, "policy": cmd_policy, "onboard": cmd_onboard, "config": cmd_config}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
