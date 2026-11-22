"""The library the Goal suite needs (2026-11-08). Build side; ``scripts/build_libraries.py`` runs its steps in order.

The robot's models come from the simulator's own object files, as for Object and Spatial (known-object models, M2's
deal). The Goal suite adds a workcell and two types the robot has never had, the bottle and the rack. M2's folders are
frozen and never overwritten, so this builds a VARIANT TREE, the mechanism the policy and the server already have
(``lfb_library``, ``m2_library.lfb_dirs``)::

    goal-lib/m1/             robot.json and records/ (copies; the bottle's grasp records added)
    goal-lib/m2/policy-v3/   M2's policy-v3 (copies) + the two types + the Goal workcell, with its own manifest
    goal-lib/m2/onboard-v3/  M2's two workcells (copies) + libero_goal (m2_onboard, unchanged, on the merged export)

Object and Spatial's files are byte-for-byte M2's. Steps (paths inside ``goal-lib/build``):

    <venv-libero> -I -B goal_lib.py export     # the simulator: Pro2.0's Goal scene, state 0 (no image is saved)
    <monty venv>  -P -m rowant_lab.pro2.goal_lib policy
    <monty venv>  -P -m rowant_lab.pro2.goal_lib onboard

``export`` also checks that the types Goal shares with M2's library are the same objects (primitives within 1e-6, same
meshes) and that the workcell equals Spatial's, so the Panda model and the arena carry over.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.pro2 import goal_paths as GP  # noqa: E402  (the Goal workcell's types; M2's m2_paths has none)
from rowant_lab.settings import DATA  # noqa: E402
from rowant_lab.pro2.machine import BENCHMARK  # noqa: E402

G = DATA / "runs" / "pro2" / "goal-lib"
BUILD = G / "build"
PAYLOAD = BENCHMARK
M2LIB = DATA / "runs" / "libero-m2" / "library"
M1LIB = DATA / "runs" / "libero-m1" / "library"
SUITE = GP.SUITE
NEW_TYPES = ("wine_bottle", "wine_rack")
SHARED_TYPES = ("akita_black_bowl", "cream_cheese", "plate", "wooden_cabinet", "flat_stove")
TOL = 1e-6


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a, b, tol: float = TOL) -> bool:
    """Equal JSON documents, numbers within ``tol`` (two MuJoCo versions compile the same file a few bits apart)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) <= tol
    return a == b


def base_task_file() -> str:
    """Goal's base task 0 as the simulator's own task file (``simulator-source/libero/libero/bddl_files``), named
    through LIBERO's API. Called by ``export`` after the simulator is on the path; the service's task catalogue
    (which also lists the held-out task files) is never opened."""
    from libero.libero import benchmark

    sim_bddl = (PAYLOAD / "simulator-source" / "libero" / "libero" / "bddl_files" / SUITE).resolve()
    path = Path(benchmark.get_benchmark_dict()[SUITE]().get_task_bddl_file_path(0)).resolve()
    if not (path.is_file() and path.is_relative_to(sim_bddl)):
        raise SystemExit(f"no base task file: {path} is not a file of the simulator's own {sim_bddl}")
    return str(path)


def scene_of(env, inner) -> dict:
    """Sites, arena and robot mount of the compiled scene, as ``export_geom.export_task`` reads them."""
    import numpy as np

    from rowant_lab.libero_env import export_geom as EG

    m, d = env.sim.model, env.sim.data
    names = list(inner.objects_dict) + list(inner.fixtures_dict)
    claimed = set()
    for name in names:
        claimed |= set(EG.subtree(m, int(inner.obj_body_id[name])))
    sites, arena = {}, []
    for sname, so in inner.object_sites_dict.items():
        sid = m.site_name2id(sname)
        b = int(m.site_bodyid[sid])
        parent = next((n for n in names if b in EG.subtree(m, int(inner.obj_body_id[n]))), None)
        Rs = EG.quat_to_mat(m.site_quat[sid])
        if parent is None:
            Rb_w = np.asarray(d.xmat[b]).reshape(3, 3)
            pos, frame, R = np.asarray(d.xpos[b]) + Rb_w @ np.asarray(m.site_pos[sid]), "world", Rb_w @ Rs
        else:
            frame, pos, R = m.body_id2name(b), np.asarray(m.site_pos[sid], dtype=float), Rs
        sites[sname] = {"parent_instance": parent, "frame_body": frame, "pos": EG._l(pos), "R": [EG._l(r) for r in R],
                        "size": EG._l(np.asarray(so.size, dtype=float))}
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b in claimed or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        bname = m.body_id2name(b) or ""
        if bname.startswith(EG.ROBOT_PREFIXES):
            continue
        t = int(m.geom_type[g])
        rec = {"geom": m.geom_id2name(g), "body": bname, "type": EG.GEOM_TYPES.get(t, str(t))}
        Rg = np.asarray(d.geom_xmat[g]).reshape(3, 3)
        if t == 0:
            rec.update(point=EG._l(d.geom_xpos[g]), normal=EG._l(Rg[:, 2]))
        else:
            half, off = EG.geom_box(m, g)
            rec.update(centre=EG._l(np.asarray(d.geom_xpos[g]) + Rg @ off), R=[EG._l(r) for r in Rg], half=EG._l(half),
                       exact=t == 6)
        arena.append(rec)
    return {"sites": sites, "arena": arena,
            "robot_mount": {"base_pos": EG._l(d.xpos[m.body_name2id("robot0_base")], 6),
                            "shoulder_pos": EG._l(d.xpos[m.body_name2id("robot0_link2")], 6),
                            "note": "the robot's own mounting (fixed per scene type), for the agent's reach model"}}


def m1_type(env, inner, name: str, typ: str) -> dict:
    """One type in M1's format (collision primitives per body), as ``export_geom.export_task`` builds it."""
    import numpy as np

    from rowant_lab.libero_env import export_geom as EG

    m, d = env.sim.model, env.sim.data
    root = int(inner.obj_body_id[name])
    prefix = name + "_"
    bodies = {}
    for b in EG.subtree(m, root):
        bn = m.body_id2name(b)
        rel = bn[len(prefix):] if bn.startswith(prefix) else bn
        prims = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != b or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            rec = EG.geom_record(m, g)
            gn = rec["geom"] or ""
            rec["geom"] = gn[len(prefix):] if gn.startswith(prefix) else gn
            prims.append(rec)
        bodies[rel] = {"mass": round(float(m.body_mass[b]), 9), "com": EG._l(m.body_ipos[b]), "is_root": b == root,
                       "primitives": prims}
    up = np.asarray(d.xmat[root]).reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
    return {"type": typ, "kind": "fixture" if name in inner.fixtures_dict else "object",
            "root_body": next(k for k, v in bodies.items() if v["is_root"]), "bodies": bodies,
            "subtree_mass": round(float(m.body_subtreemass[root]), 9), "rest_up_in_body": EG._l(up, 6)}


def cmd_export(a) -> int:
    if not sys.flags.isolated:
        print("REFUSE: run with python -I", file=sys.stderr)
        return 2
    if str(PAYLOAD / "simulator-source") not in sys.path:  # Pro2.0's simulator shadows our LIBERO
        sys.path[0:0] = [str(PAYLOAD / "simulator-source"), str(PAYLOAD / "service-source")]
    os.environ.setdefault("LIBERO_CONFIG_PATH", a.config)
    from rowant_lab.libero_env import m2_rpaths as RP

    RP.pin_numba_cache()
    from rowant_lab.hybrid_frontier.e9 import isolation

    if not isolation.scan()["clear"]:
        print("REFUSE: isolation guard not clear", file=sys.stderr)
        return 3
    from rowant_lab.libero_env import m2_export as EX
    from rowant_lab.libero_env import m2_export3 as E3
    from rowant_lab.libero_env.export_geom import instance_types
    from rowant_lab.pro2 import dev_episode as DE

    out = BUILD / "export"
    if out.exists():
        print(f"REFUSE: {out} exists", file=sys.stderr)
        return 2
    ad = DE.adapter_class()()
    ep = {"suite": SUITE, "task": 0, "case": "BASE", "state": 0, "seed": 142, "bddl": base_task_file(), "max_steps": 300}
    ad.create(DE.env_spec(ep, 64))
    env, inner = ad._env, ad._env.env
    scene = scene_of(env, inner)
    it = instance_types(inner)
    t0 = time.perf_counter()
    report: dict = {"new": {}, "shared": {}}
    m1out = BUILD / "m1types"
    m1out.mkdir(parents=True)
    ctx = getattr(env.sim, "_render_context_offscreen", None)
    gg = [int(x) for x in ctx.vopt.geomgroup] if ctx is not None else None
    seconds: dict = {}
    for name in list(inner.objects_dict) + list(inner.fixtures_dict):
        typ = it.get(name, name)
        if typ == "table":
            continue
        t1 = time.perf_counter()
        doc = m1_type(env, inner, name, typ)
        if typ in NEW_TYPES:
            ex = EX.export_type(env, inner, name, typ, out / "types", scene, name + "_")
            (out / "types" / f"{typ}.json").write_text(json.dumps(ex, indent=1, sort_keys=True))
            (m1out / f"{typ}.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
            probe = E3.probe_type(env, inner, name, typ, out / "types", gg)
            seconds[typ] = round(time.perf_counter() - t1, 3)
            report["new"][typ] = {"kind": ex["kind"], "parts": sorted(ex["parts"]), "sites": sorted(ex["sites"]),
                                  "bodies": sorted(ex["bodies"]), "rest_up": doc["rest_up_in_body"],
                                  "rendered_not_exported": [x["name"] for x in probe["added"]],
                                  "mesh_check": probe["v2_body_check_ok"],
                                  "n_primitives": sum(len(b["primitives"]) for b in doc["bodies"].values())}
        elif typ in SHARED_TYPES:
            old = json.loads((M1LIB / "types" / f"{typ}.json").read_text())
            same_m1 = close(old["bodies"], doc["bodies"])
            ex = EX.export_type(env, inner, name, typ, BUILD / "scratch-shared", scene, name + "_")
            oldex = json.loads((M2LIB / "export-v3" / "types" / f"{typ}.json").read_text())
            cut = lambda b: [{k: v for k, v in g.items() if k in ("key", "n_vertices", "n_faces", "has_uv", "mesh")}  # noqa: E731
                             for g in b["geoms"]]
            same_ex = (set(oldex["bodies"]) == set(ex["bodies"]) and set(oldex["sites"]) == set(ex["sites"])
                       and all(cut(oldex["bodies"][r])[:len(ex["bodies"][r]["geoms"])] == cut(ex["bodies"][r])
                               for r in ex["bodies"])
                       and close({k: {x: v[x] for x in ("pos", "R", "size", "body")} for k, v in oldex["sites"].items()},
                                 {k: {x: v[x] for x in ("pos", "R", "size", "body")} for k, v in ex["sites"].items()}))
            report["shared"][typ] = {"m1_primitives_same": same_m1, "export_same": same_ex, "kind": ex["kind"]}
    spatial = json.loads((M2LIB / "policy-v3" / "workcells" / "libero_spatial.json").read_text())
    m = env.sim.model
    rb = m.body_name2id("robot0_base")
    wc = {"arena": scene["arena"], "robot_mount": scene["robot_mount"],
          "robot_base": {"pos": EX._l(env.sim.data.xpos[rb]), "quat_wxyz": EX._l(env.sim.data.xquat[rb])},
          "cameras": {c: {"fovy_deg": float(m.cam_fovy[m.camera_name2id(c)])} for c in RP.CAMERAS}}
    report["workcell_same_as_spatial"] = {k: close(spatial[k], wc[k]) for k in wc}
    (BUILD / "workcell.json").write_text(json.dumps(wc, indent=1, sort_keys=True))
    bodies, _arrays = EX.panda_model(env)
    old = json.loads((M2LIB / "export-v3" / "panda-libero_spatial.json").read_text())["bodies"]
    report["panda_same_as_spatial"] = close(
        {k: {x: v[x] for x in ("parent", "body_pos", "body_quat")} for k, v in old.items()},
        {k: {x: v[x] for x in ("parent", "body_pos", "body_quat")} for k, v in bodies.items()}, 1e-5)
    (BUILD / "seconds.json").write_text(json.dumps(seconds))
    ad.close()
    shutil.rmtree(BUILD / "scratch-shared", ignore_errors=True)
    report["elapsed_s"] = round(time.perf_counter() - t0, 2)
    (BUILD / "export-report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


def cmd_policy(a) -> int:
    """The variant's m1/ and m2/policy-v3/: M2's bytes plus the Goal additions, every new document under M2's rules."""
    import numpy as np  # noqa: F401  (numpy is what the grasp records need)

    from rowant_lab.libero_env import grasp as GR
    from rowant_lab.libero_env import m2_libv2 as LV2
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.libero_env import m2_paths as MP

    m1, pol = G / "m1", G / "m2" / "policy-v3"
    if m1.exists() or pol.exists():
        print("REFUSE: the variant's m1 or policy-v3 exists", file=sys.stderr)
        return 2
    m1.mkdir(parents=True)
    shutil.copy2(M1LIB / "robot.json", m1 / "robot.json")
    shutil.copytree(M1LIB / "records", m1 / "records")
    robot = json.loads((m1 / "robot.json").read_text())
    jaw = GR.JawModel.from_robot(robot)
    params = GR.RecordParams()
    built = {}
    for typ in NEW_TYPES:
        model = json.loads((BUILD / "m1types" / f"{typ}.json").read_text())
        if model["kind"] != "object":
            continue
        root = model["bodies"][model["root_body"]]
        rec = GR.build_records(root["primitives"], model["rest_up_in_body"], root["com"], jaw, params, name=typ)
        doc = GR.records_to_json(rec)
        (m1 / "records" / f"{typ}.json").write_text(json.dumps(doc, sort_keys=True) + "\n")
        built[typ] = {"n_records": rec["n_records"], "n_kept": int(len(rec["width"])), "height": round(rec["height"], 4)}
    tmp = pol.with_name("policy-v3.tmp")
    shutil.copytree(M2LIB / "policy-v3", tmp)
    man = json.loads((tmp / "manifest.json").read_text())
    staged = {}
    for typ in NEW_TYPES:
        staged[f"types/{typ}.json"] = LV2.sanitize_export_type(json.loads((BUILD / "export" / "types" / f"{typ}.json").read_text()))
        staged[f"m1types/{typ}.json"] = LV2.sanitize_m1_type(json.loads((BUILD / "m1types" / f"{typ}.json").read_text()))
    wc = json.loads((BUILD / "workcell.json").read_text())
    staged[f"workcells/{SUITE}.json"] = {**wc, "suite": SUITE, "types": list(GP.TYPES),
                                         "identical_across_tasks": True, "format": "rowant pro2 Goal workcell v1",
                                         "label": MP.LABEL}
    kinds = {"types": "type", "m1types": "m1type", "workcells": "workcell"}
    bad = {rel: LB.problems(doc, kinds[rel.split("/")[0]])[:5] for rel, doc in staged.items()
           if LB.problems(doc, kinds[rel.split("/")[0]])}
    if bad:
        shutil.rmtree(tmp)
        print("REFUSE:", json.dumps(bad)[:2000], file=sys.stderr)
        return 3
    for rel, doc in staged.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        b = LV2.dumps(doc)
        p.write_bytes(b)
        man["files"][rel] = hashlib.sha256(b).hexdigest()
    for typ in built:
        man["in_place"][f"m1/records/{typ}.json"] = sha256_file(m1 / "records" / f"{typ}.json")
    man["variant_of"] = "policy-v3"
    man["variant_note"] = "M2's files unchanged, plus a third workcell and two types (rowant_lab/pro2/goal_lib.py)"
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    tmp.rename(pol)
    print(json.dumps({"records": built, "n_files": len(man["files"]), "n_in_place": len(man["in_place"])}))
    return 0


def cmd_onboard(a) -> int:
    """m2_onboard, unchanged, on an export folder holding the Goal workcell's seven types; M2's two workcells copied."""
    from rowant_lab.libero_env import m2_onboard as OBM

    onb = G / "m2" / "onboard-v3"
    merged = BUILD / "export-merged"
    if (onb / SUITE).exists():
        print(f"REFUSE: {onb / SUITE} exists", file=sys.stderr)
        return 2
    (merged / "types").mkdir(parents=True, exist_ok=True)
    src = M2LIB / "export-v3" / "types"
    for typ in SHARED_TYPES:
        for ext in ("json", "npz"):
            shutil.copy2(src / f"{typ}.{ext}", merged / "types" / f"{typ}.{ext}")
    for typ in NEW_TYPES:
        for ext in ("json", "npz"):
            shutil.copy2(BUILD / "export" / "types" / f"{typ}.{ext}", merged / "types" / f"{typ}.{ext}")
    secs = json.loads((M2LIB / "export-v3" / "manifest.json").read_text()).get("seconds_per_type", {})
    secs.update(json.loads((BUILD / "seconds.json").read_text()))
    (merged / "manifest.json").write_text(json.dumps({"seconds_per_type": secs}))
    onb.mkdir(parents=True, exist_ok=True)
    for s in ("libero_object", "libero_spatial"):
        if not (onb / s).exists():
            shutil.copytree(M2LIB / "onboard-v3" / s, onb / s, symlinks=False)
    OBM.MP.EXPORT_DIR = merged  # build side only: the module reads this name, and its default arguments hold M2's
    # build side only, in this process: m2_onboard reads a suite's types from m2_paths, whose frozen table has no Goal
    OBM.MP.WORKCELL_TYPES = {**OBM.MP.WORKCELL_TYPES, SUITE: GP.TYPES}
    OBM.graph_ids.__defaults__ = (merged,)
    OBM.parts_of_suite.__defaults__ = (merged,)
    doc = OBM.onboard_suite(SUITE, onb)
    for g, r in doc["per_part"].items():
        print(g, r["type"], r["part"], r["n_nodes"], r["seconds"].get("total"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("export", "policy", "onboard"))
    ap.add_argument("--config", default="", help="export: the folder of a LIBERO config pointing at the payload")
    a = ap.parse_args(argv)
    return {"export": cmd_export, "policy": cmd_policy, "onboard": cmd_onboard}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
