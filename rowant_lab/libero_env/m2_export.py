"""M2 library exporter (plan object_library_and_onboarding.step_1). Run by ``scripts/build_libraries.py`` (step
``m2-export``) with the simulator environment's interpreter, isolated (``-I -B``), the benchmark's simulator on the path.

M2 (perceived poses, grounded instruction): perception in the loop. Once per object TYPE, from LIBERO's COMPILED
model at start state 0 of the first task that holds the type (as M1's exporter), it writes:

* ``types/<type>.npz`` and ``types/<type>.json``: every visual geom (contype 0 and conaffinity 0) of every body of
  the type's subtree as its compiled mesh (vertices, faces, per-vertex normals and texcoords, MuJoCo 3.3.2 stores
  them per vertex with the face index arrays shared) placed in its body's frame (``geom_pos``/``geom_quat``), plus
  its material (texture file by path and sha256, ``mat_rgba``, ``texrepeat``, ``texuniform``) or ``geom_rgba``; the
  kinematic tree of the type (parent, body offset, joint type, axis, position, range, qpos0); the PARTS (the rigid
  root part = the root and every body joined to it without a joint, plus jointed bodies whose joint moves their
  geometry's centre by under 1 cm over its range, fixed at qpos0; each other jointed body starts a part); each
  body's pose in its part frame; and the per-type body-frame sites (re-keyed from M1's scene sites whose frame body
  belongs to an instance of the type; world-frame sites are excluded: they are BDDL init-sampling regions);
* ``workcells/<suite>.json``: the arena primitives and the robot mount (from M1's scene files, checked identical
  across the suite's 10 tasks), the robot base pose and the camera names, and the workcell type list;
* ``panda-<suite>.json`` / ``.npz``: per suite (the mount differs), the Panda's kinematic tree (bodies ``robot0_*``/``gripper0_*``/``mount0_*``) with
  joint-to-proprioception indices and each body's visual mesh vertices and faces in its body frame (self-filter);
* ``manifest.json``: the sha256 of every file written.

Texture images are referenced in place in LIBERO's asset tree (path, sha256); no image is written or viewed. The
instance names never appear in the outputs (only type, body-relative and part names).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.libero_env import m1_paths as MP1  # noqa: E402
from rowant_lab.libero_env import m2_rpaths as MP  # noqa: E402

MP.pin_numba_cache()  # M2's own numba cache, before robosuite is imported (verifier M6)

FORMAT = "rowant libero_env M2 library export v1"
FIXED_PART_MOVE_M = 0.01
ROBOT_PREFIXES = ("robot0_", "gripper0_", "mount0_")
JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}


def _l(a, nd: int = 9) -> list:
    return [round(float(x), nd) for x in a]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quat_to_mat(q):
    import numpy as np

    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def axis_angle_mat(axis, ang: float):
    import numpy as np

    a = np.asarray(axis, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def tex_path(m, tex: int) -> str | None:
    P = bytes(m.paths)
    a = int(m.tex_pathadr[tex])
    if a < 0:
        return None
    return P[a:P.index(b"\0", a)].decode()


def mesh_arrays(m, g):
    """(V, F, N, UV or None) of a mesh geom, vertices and normals in the geom's BODY frame."""
    import numpy as np

    mid = int(m.geom_dataid[g])
    va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
    fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
    V = np.asarray(m.mesh_vert[va:va + vn], dtype=np.float64)
    F = np.asarray(m.mesh_face[fa:fa + fn], dtype=np.int64)
    na, nn = int(m.mesh_normaladr[mid]), int(m.mesh_normalnum[mid])
    Nn = np.asarray(m.mesh_normal[na:na + nn], dtype=np.float64)
    FN = np.asarray(m.mesh_facenormal[fa:fa + fn], dtype=np.int64)
    ta, tn = int(m.mesh_texcoordadr[mid]), int(m.mesh_texcoordnum[mid])
    # per-vertex normals: MuJoCo indexes normals per face corner; scatter them to the corner's vertex (a vertex
    # shared by faces with different normal indices gets the area-free mean)
    N = np.zeros_like(V)
    np.add.at(N, F.ravel(), Nn[FN.ravel()])
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    UV = None
    if tn > 0 and ta >= 0:
        T = np.asarray(m.mesh_texcoord[ta:ta + tn], dtype=np.float64)
        FT = np.asarray(m.mesh_facetexcoord[fa:fa + fn], dtype=np.int64)
        UV = np.full((vn, 2), np.nan)
        UV[F.ravel()] = T[FT.ravel()]
        if not np.isfinite(UV).all() or not np.array_equal(FT, F):
            # texcoords not 1:1 with vertices: unshare every face corner
            V = V[F.ravel()]
            N = N[F.ravel()]
            UV = T[FT.ravel()]
            F = np.arange(3 * fn).reshape(fn, 3)
    Rg = quat_to_mat(m.geom_quat[g])
    pg = np.asarray(m.geom_pos[g], dtype=np.float64)
    return V @ Rg.T + pg, F, N @ Rg.T, UV


def subtree(m, root: int) -> list[int]:
    out = []
    for b in range(m.nbody):
        p = b
        while p not in (0, root):
            p = int(m.body_parentid[p])
        if p == root:
            out.append(b)
    return out


def body_joints(m, b, prefix_strip: str = "") -> list[dict]:
    """The body's own joints, their names stripped of the instance prefix (the library holds no instance name)."""
    out = []
    for j in range(m.njnt):
        if int(m.jnt_bodyid[j]) == b:
            jn = m.joint_id2name(j) or ""
            out.append({"name": jn[len(prefix_strip):] if prefix_strip and jn.startswith(prefix_strip) else jn,
                        "type": JOINT_TYPES[int(m.jnt_type[j])], "axis": _l(m.jnt_axis[j]),
                        "pos": _l(m.jnt_pos[j]), "range": _l(m.jnt_range[j]),
                        "qpos0": round(float(m.qpos0[int(m.jnt_qposadr[j])]), 9),
                        "limited": bool(m.jnt_limited[j])})
    return out


def joint_transform(jt: dict, q: float):
    """(R, t) of one hinge or slide joint at value q, in its body's frame (MuJoCo's joint rule)."""
    import numpy as np

    d = float(q) - float(jt["qpos0"])
    if jt["type"] == "hinge":
        R = axis_angle_mat(jt["axis"], d)
        p = np.asarray(jt["pos"], float)
        return R, p - R @ p
    if jt["type"] == "slide":
        return np.eye(3), np.asarray(jt["axis"], float) * d
    raise ValueError(f"joint type {jt['type']} not handled")


def export_type(env, inner, name: str, typ: str, out_types: Path, m1_scene: dict, prefix_strip: str) -> dict:
    """One type from instance ``name`` of the current env."""
    import numpy as np

    m, d = env.sim.model, env.sim.data
    root = int(inner.obj_body_id[name])
    bodies = subtree(m, root)
    rel = {}
    for b in bodies:
        bn = m.body_id2name(b)
        rel[b] = bn[len(prefix_strip):] if bn.startswith(prefix_strip) else bn
    arrays = {}
    body_recs = {}
    for b in bodies:
        geoms = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != b or m.geom_contype[g] or m.geom_conaffinity[g]:
                continue
            if int(m.geom_type[g]) != 7:
                raise RuntimeError(f"{typ}: visual geom {g} is not a mesh")
            V, F, N, UV = mesh_arrays(m, g)
            k = len(arrays) // 4
            key = f"g{k}"
            arrays[f"{key}__V"], arrays[f"{key}__F"], arrays[f"{key}__N"] = V, F, N
            arrays[f"{key}__UV"] = UV if UV is not None else np.zeros((0, 2))
            mat = int(m.geom_matid[g])
            mrec = {"rgba": _l(m.geom_rgba[g], 6)}
            if mat >= 0:
                tex = int(m.mat_texid[mat][1])  # mjTEXROLE_RGB
                mrec = {"mat_rgba": _l(m.mat_rgba[mat], 6), "texrepeat": _l(m.mat_texrepeat[mat], 6),
                        "texuniform": bool(m.mat_texuniform[mat])}
                if tex >= 0:
                    tp = tex_path(m, tex)
                    mrec.update(texture=tp, texture_sha256=sha256_file(tp) if tp else None,
                                tex_width=int(m.tex_width[tex]), tex_height=int(m.tex_height[tex]),
                                tex_type=int(m.tex_type[tex]))
            geoms.append({"key": key, "n_vertices": int(len(V)), "n_faces": int(len(F)),
                          "has_uv": UV is not None, "material": mrec,
                          "mesh": (m.mesh_id2name(int(m.geom_dataid[g])) or "")[len(prefix_strip):]})
        par = int(m.body_parentid[b])
        body_recs[rel[b]] = {"parent": rel.get(par) if b != root else None, "body_pos": _l(m.body_pos[b]),
                             "body_quat": _l(m.body_quat[b]), "joints": body_joints(m, b, prefix_strip),
                             "geoms": geoms, "is_root": b == root}
    # parts: walk from the root; a body with a joint that moves its geometry's centre >= 1 cm starts a part
    def centre_of(rb):
        pts = [arrays[f"{g['key']}__V"] for g in body_recs[rb]["geoms"]]
        return np.concatenate(pts).mean(0) if pts else np.zeros(3)

    part_of, pose_in_part, fixed_joints = {}, {}, {}
    order = list(bodies)  # MuJoCo body ids are in tree order (parents first)
    for b in order:
        r = rel[b]
        rec = body_recs[r]
        Rb, tb = quat_to_mat(rec["body_quat"]), np.asarray(rec["body_pos"], float)
        if rec["is_root"]:
            part_of[r], pose_in_part[r] = r, (np.eye(3), np.zeros(3))
            continue
        pr = rec["parent"]
        Rp, tp = pose_in_part[pr]
        movable = [j for j in rec["joints"] if j["type"] in ("hinge", "slide")]
        if len(rec["joints"]) > len(movable):
            raise RuntimeError(f"{typ}: body {r} has a free or ball joint")
        new_part = False
        if movable:
            if len(movable) > 1:
                raise RuntimeError(f"{typ}: body {r} has several joints")
            jt = movable[0]
            c = centre_of(r)
            moves = []
            for q in np.linspace(jt["range"][0], jt["range"][1], 21):
                Rj, tj = joint_transform(jt, q)
                R0, t0 = joint_transform(jt, jt["qpos0"])
                moves.append(float(np.linalg.norm((Rj @ c + tj) - (R0 @ c + t0))))
            if max(moves) >= FIXED_PART_MOVE_M:
                new_part = True
            else:
                fixed_joints[r] = {"joint": jt["name"], "fixed_at": jt["qpos0"],
                                   "max_centre_move_m": round(max(moves), 5)}
        if new_part:
            part_of[r], pose_in_part[r] = r, (np.eye(3), np.zeros(3))
        else:
            part_of[r] = part_of[pr]
            pose_in_part[r] = (Rp @ Rb, Rp @ tb + tp)
    parts = {}
    for b in order:
        r = rel[b]
        pa = part_of[r]
        P = parts.setdefault(pa, {"bodies": [], "joint": None, "parent_body": None})
        P["bodies"].append(r)
        if pa == r and not body_recs[r]["is_root"]:
            jt = [j for j in body_recs[r]["joints"] if j["type"] in ("hinge", "slide")][0]
            P["joint"] = dict(jt)
            P["parent_body"] = body_recs[r]["parent"]
    # sites in body frames (from M1's scene of this task: frame_body inside the instance's subtree)
    sites = {}
    names_in = {m.body_id2name(b) for b in bodies}
    for sname, s in m1_scene["sites"].items():
        if s["frame_body"] == "world" or s["frame_body"] not in names_in:
            continue
        if not sname.startswith(prefix_strip):
            raise RuntimeError(f"site {sname} does not start with {prefix_strip}")
        sites[sname[len(prefix_strip):]] = {"body": s["frame_body"][len(prefix_strip):], "pos": s["pos"], "R": s["R"],
                                           "size": s["size"]}
    out_types.mkdir(parents=True, exist_ok=True)
    npz = out_types / f"{typ}.npz"
    np.savez(npz, **arrays)
    doc = {"format": FORMAT, "label": MP.LABEL, "type": typ, "kind": "fixture" if name in inner.fixtures_dict else
           "object", "root_body": rel[root], "bodies": body_recs, "parts": parts, "part_of": part_of,
           "pose_in_part": {r: {"R": [_l(x) for x in v[0]], "t": _l(v[1])} for r, v in pose_in_part.items()},
           "fixed_joints": fixed_joints, "sites": sites, "npz": npz.name, "npz_sha256": sha256_file(npz)}
    return doc


def panda_model(env) -> tuple[dict, dict]:
    import numpy as np

    m = env.sim.model
    arrays, bodies = {}, {}
    arm = [f"robot0_joint{i}" for i in range(1, 8)]
    fingers = ["gripper0_finger_joint1", "gripper0_finger_joint2"]
    for b in range(1, m.nbody):
        bn = m.body_id2name(b) or ""
        if not bn.startswith(ROBOT_PREFIXES):
            continue
        js = body_joints(m, b)
        for jt in js:
            jt["proprio"] = (["robot0_joint_pos", arm.index(jt["name"])] if jt["name"] in arm else
                             ["robot0_gripper_qpos", fingers.index(jt["name"])] if jt["name"] in fingers else None)
        geoms = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != b or int(m.geom_type[g]) != 7 or m.geom_contype[g] or m.geom_conaffinity[g]:
                continue
            V, F, _N, _UV = mesh_arrays(m, g)
            key = f"g{len(arrays) // 2}"
            arrays[f"{key}__V"], arrays[f"{key}__F"] = V, F
            geoms.append(key)
        par = m.body_id2name(int(m.body_parentid[b])) or "world"
        bodies[bn] = {"parent": par, "body_pos": _l(m.body_pos[b]), "body_quat": _l(m.body_quat[b]), "joints": js,
                      "geoms": geoms}
    return bodies, arrays


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(MP.EXPORT_DIR))
    a = ap.parse_args(argv)
    if not sys.flags.isolated:
        print("REFUSE: run with python -I", file=sys.stderr)
        return 2
    from rowant_lab.hybrid_frontier.e9 import isolation

    g = isolation.scan()
    if not g["clear"]:
        print("REFUSE: isolation guard not clear", file=sys.stderr)
        return 3
    import numpy as np

    from rowant_lab.libero_env.export_geom import instance_types, start_state

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    out = Path(a.out)
    if out.exists():
        print(f"REFUSE: {out} exists", file=sys.stderr)
        return 2
    t_start = time.perf_counter()
    done_types: dict = {}
    workcells: dict = {}
    panda: dict = {}
    timing = {}
    for suite in MP.SUITES:
        bench = benchmark.get_benchmark_dict()[suite](task_order_index=0)
        wc = None
        for ti in range(MP.N_TASKS):
            task = bench.get_task(ti)
            m1_scene = json.loads((MP1.LIBRARY_DIR / "scenes" / suite / f"{task.name}.json").read_text())
            env = OffScreenRenderEnv(bddl_file_name=bench.get_task_bddl_file_path(ti), use_camera_obs=False)
            env.seed(0)
            env.reset()
            env.set_init_state(start_state(bench, ti, 0))
            inner = env.env
            it = instance_types(inner)
            for name in list(inner.objects_dict) + list(inner.fixtures_dict):
                typ = it.get(name, name)
                if typ in done_types:
                    done_types[typ]["seen_in"].append(f"{suite}:{ti}")
                    continue
                t0 = time.perf_counter()
                doc = export_type(env, inner, name, typ, out / "types", m1_scene, name + "_")
                doc["exported_from"] = f"{suite} task {ti} DEV state 0"
                doc["seen_in"] = [f"{suite}:{ti}"]
                done_types[typ] = doc
                timing[typ] = round(time.perf_counter() - t0, 3)
            m = env.sim.model
            rb = m.body_name2id("robot0_base")
            this = {"arena": m1_scene["arena"], "robot_mount": {k: v for k, v in m1_scene["robot_mount"].items()},
                    "robot_base": {"pos": _l(env.sim.data.xpos[rb]), "quat_wxyz": _l(env.sim.data.xquat[rb])},
                    "cameras": {c: {"fovy_deg": float(m.cam_fovy[m.camera_name2id(c)])} for c in MP.CAMERAS}}
            sig = json.dumps(this, sort_keys=True)
            if wc is None:
                wc = (this, sig)
            elif sig != wc[1]:
                raise RuntimeError(f"{suite}: workcell differs at task {ti}")
            if suite not in panda:
                panda[suite] = panda_model(env)
            env.close()
        workcells[suite] = {**wc[0], "suite": suite, "types": list(MP.WORKCELL_TYPES[suite]),
                            "identical_across_tasks": True, "format": FORMAT, "label": MP.LABEL}
    missing = {t for s in MP.SUITES for t in MP.WORKCELL_TYPES[s]} - set(done_types)
    if missing:
        raise RuntimeError(f"types not exported: {missing}")
    for typ, doc in done_types.items():
        (out / "types" / f"{typ}.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    (out / "workcells").mkdir(parents=True, exist_ok=True)
    for suite, wc in workcells.items():
        (out / "workcells" / f"{suite}.json").write_text(json.dumps(wc, indent=1, sort_keys=True))
    for suite, (bodies, arrays) in panda.items():  # per suite: the robot's mount differs (floor or table)
        np.savez(out / f"panda-{suite}.npz", **arrays)
        (out / f"panda-{suite}.json").write_text(json.dumps(
            {"format": FORMAT, "suite": suite, "bodies": bodies, "npz": f"panda-{suite}.npz",
             "npz_sha256": sha256_file(out / f"panda-{suite}.npz")}, indent=1, sort_keys=True))
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {"format": FORMAT, "label": MP.LABEL, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "exporter_sha256": sha256_file(Path(__file__)), "isolation_utc": g["time_utc"],
                "seconds_per_type": timing, "elapsed_s": round(time.perf_counter() - t_start, 2),
                "files": {str(p.relative_to(out)): sha256_file(p) for p in files}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps({"types": sorted(done_types), "elapsed_s": manifest["elapsed_s"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
