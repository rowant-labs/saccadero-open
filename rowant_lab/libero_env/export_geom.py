"""M1 object library exporter (plan: agent_design.pipeline step 1). Run by ``scripts/build_libraries.py`` (step
``m1-export``) with the simulator environment's interpreter, isolated (``-I -B``), the benchmark's simulator on the path.

For each of the 20 tasks it builds LIBERO's own OffScreenRenderEnv (images off), resets, loads the task's start state 0
(``start_state``: the simulator's own start-state file, read through LIBERO's API) and reads LIBERO's COMPILED model:

* every collision geom (contype or conaffinity non-zero) of every object and fixture body subtree, as its type, size
  and pose in its own body's frame (``model.geom_pos``/``geom_quat``: MuJoCo's mesh re-centring is already applied).
  Mesh geoms are kept as the box bounding their compiled vertices in the geom frame (conservative for collisions);
* each body's mass and centre of mass (``body_ipos``) and, per object type, the body-frame axis that is world up at
  DEV state 0 (the rest pose's up, used to keep grasp closing axes horizontal);
* LIBERO's site regions (``object_sites_dict``): size and pose in the parent body frame (world for arena sites);
* the arena's static collision geoms (floor, table, walls) in the world frame;
* the Panda hand, finger and pad collision geoms in the end-effector frame (``robot0_eef_pos``, the grip site, and
  ``robot0_eef_quat``, the hand body's orientation), with the finger joints' axes and the open finger qpos.

The F1 frame check runs here: every exported primitive placed at its body's true pose must match
``data.geom_xpos``/``geom_xmat`` within 1e-6, at start state 0 right after ``set_init_state`` and after the 10 wait
steps. Outputs (JSON, with sha256) go to ``<data>/runs/libero-m1/library/`` (``m1_paths.LIBRARY_DIR``):
``types/<type>.json`` (one model per object or fixture type), ``scenes/<suite>/<task>.json`` (instances, types,
sites, arena, BDDL goal) and ``robot.json``, plus ``library-manifest.json``. Numbers only, never an image.

These are known-object models, as a model-based robot would hold; the policy never reads the simulator at run time.
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

from rowant_lab.libero_env import m1_paths as MP  # noqa: E402

FORMAT = "rowant libero_env M1 object library v1"
F1_TOL = 1e-6
ROBOT_PREFIXES = ("robot0_", "gripper0_", "mount0_")
GEOM_TYPES = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}


def _l(a, nd: int = 9) -> list:
    return [round(float(x), nd) for x in a]


def subtree(m, root: int) -> list[int]:
    out = []
    for b in range(m.nbody):
        p = b
        while p not in (0, root):
            p = int(m.body_parentid[p])
        if p == root:
            out.append(b)
    return out


def geom_box(m, g):
    """(half-size, centre offset in the geom frame) of the box this geom collides as (exact for boxes; for meshes
    the box bounding the compiled vertices; for spheres, capsules, cylinders and ellipsoids their bounding box)."""
    import numpy as np

    t = int(m.geom_type[g])
    s = np.asarray(m.geom_size[g], dtype=float)
    if t == 6:
        return s, np.zeros(3)
    if t == 7:
        mid = int(m.geom_dataid[g])
        a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        v = np.asarray(m.mesh_vert[a:a + n], dtype=float)
        lo, hi = v.min(0), v.max(0)
        return 0.5 * (hi - lo), 0.5 * (hi + lo)
    if t == 2:
        return np.array([s[0]] * 3), np.zeros(3)
    if t in (3, 5):
        return np.array([s[0], s[0], s[1] + (s[0] if t == 3 else 0.0)]), np.zeros(3)
    if t == 4:
        return s.copy(), np.zeros(3)
    raise ValueError(f"geom type {t} not handled")


def quat_to_mat(q):
    import numpy as np

    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def geom_record(m, g) -> dict:
    """A collision primitive in its body's frame: an oriented box (centre, rotation, half-size)."""
    import numpy as np

    half, off = geom_box(m, g)
    Rg = quat_to_mat(m.geom_quat[g])
    centre = np.asarray(m.geom_pos[g], dtype=float) + Rg @ off
    return {"geom": m.geom_id2name(g), "type": GEOM_TYPES.get(int(m.geom_type[g]), str(int(m.geom_type[g]))),
            "centre": _l(centre), "R": [_l(r) for r in Rg], "half": _l(half),
            "exact": int(m.geom_type[g]) == 6}


def f1_error(m, d, g, rec, body_pos, body_R) -> float:
    """|placed primitive - sim geom pose| (centre and rotation, max abs), using the geom's own frame."""
    import numpy as np

    Rg = quat_to_mat(m.geom_quat[g])
    half, off = geom_box(m, g)
    c_sim = np.asarray(d.geom_xpos[g]) + np.asarray(d.geom_xmat[g]).reshape(3, 3) @ off
    c_ex = body_pos + body_R @ np.asarray(rec["centre"])
    R_ex = body_R @ Rg
    return float(max(np.max(np.abs(c_ex - c_sim)), np.max(np.abs(R_ex - np.asarray(d.geom_xmat[g]).reshape(3, 3)))))


def instance_types(inner) -> dict[str, str]:
    out = {}
    for key in ("objects", "fixtures"):
        for cat, names in inner.parsed_problem.get(key, {}).items():
            for n in names:
                out[n] = cat
    return out


def start_state(bench, ti: int, k: int = 0):
    """Start state ``k`` of task ``ti`` from the simulator's own start-state file, read through LIBERO's API (as the
    benchmark's service reads it). The libraries are built at state 0 only."""
    import numpy as np

    states = bench.get_task_init_states(ti)
    if not 0 <= int(k) < len(states):
        raise IndexError(f"task {ti}: start state {k} is not in its file ({len(states)} states)")
    return np.asarray(states[int(k)], dtype=np.float64)


def export_task(suite: str, ti: int, out: Path, types_acc: dict, f1_acc: list) -> dict:
    import numpy as np

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()[suite](task_order_index=MP.TASK_ORDER)
    task = bench.get_task(ti)
    env = OffScreenRenderEnv(bddl_file_name=bench.get_task_bddl_file_path(ti), use_camera_obs=False)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(start_state(bench, ti, 0))
    inner, sim = env.env, env.sim
    m, d = sim.model, sim.data
    itypes = instance_types(inner)
    names = list(inner.objects_dict) + list(inner.fixtures_dict)
    claimed = set()
    scene = {"suite": suite, "task_index": ti, "task_name": task.name, "language": env.language_instruction,
             "goal": [list(s) for s in inner.parsed_problem["goal_state"]],
             "obj_of_interest": list(inner.obj_of_interest), "instances": {}, "sites": {}, "arena": []}

    def body_pose(b):
        return np.asarray(d.xpos[b], dtype=float), np.asarray(d.xmat[b], dtype=float).reshape(3, 3)

    pending_f1 = []
    for name in names:
        root = int(inner.obj_body_id[name])
        bodies = subtree(m, root)
        claimed |= set(bodies)
        prefix = name + "_"
        model_bodies = {}
        for b in bodies:
            bname = m.body_id2name(b)
            rel = bname[len(prefix):] if bname.startswith(prefix) else bname
            gs = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b
                  and (m.geom_contype[g] or m.geom_conaffinity[g])]
            prims = []
            for g in gs:
                rec = geom_record(m, g)
                gname = rec["geom"] or ""
                rec["geom"] = gname[len(prefix):] if gname.startswith(prefix) else gname
                prims.append(rec)
                pending_f1.append((g, rec, b))
            model_bodies[rel] = {"mass": round(float(m.body_mass[b]), 9), "com": _l(m.body_ipos[b]),
                                 "is_root": b == root, "primitives": prims}
        typ = itypes.get(name, name)
        Rroot = body_pose(root)[1]
        up_body = Rroot.T @ np.array([0.0, 0.0, 1.0])
        model = {"type": typ, "kind": "fixture" if name in inner.fixtures_dict else "object",
                 "root_body": next(k for k, v in model_bodies.items() if v["is_root"]), "bodies": model_bodies,
                 "subtree_mass": round(float(m.body_subtreemass[root]), 9)}
        sig = json.dumps(model["bodies"], sort_keys=True)
        prev = types_acc.get(typ)
        if prev is None:
            model["rest_up_in_body"] = _l(up_body, 6)
            model["rest_up_from"] = f"{suite} task {ti} DEV state 0"
            types_acc[typ] = {"model": model, "sig": sig, "seen_in": [f"{suite}:{ti}"]}
        else:
            if prev["sig"] != sig:
                raise RuntimeError(f"type {typ}: compiled model differs between tasks ({prev['seen_in']} vs {suite}:{ti})")
            prev["seen_in"].append(f"{suite}:{ti}")
        scene["instances"][name] = {"type": typ, "kind": model["kind"], "root_body_name": m.body_id2name(root),
                                    "bodies": {rel: m.body_id2name(b) for rel, b in zip(model_bodies, bodies)}}
    # sites
    for sname, so in inner.object_sites_dict.items():
        sid = m.site_name2id(sname)
        b = int(m.site_bodyid[sid])
        parent = None
        for name in names:
            if b in subtree(m, int(inner.obj_body_id[name])):
                parent = name
        Rs = quat_to_mat(m.site_quat[sid])
        if parent is None:
            Rb_w = np.asarray(d.xmat[b]).reshape(3, 3)
            pos = np.asarray(d.xpos[b]) + Rb_w @ np.asarray(m.site_pos[sid])
            frame, R = "world", Rb_w @ Rs
        else:
            frame = m.body_id2name(b)
            pos, R = np.asarray(m.site_pos[sid], dtype=float), Rs
        scene["sites"][sname] = {"parent_instance": parent, "frame_body": frame, "pos": _l(pos),
                                 "R": [_l(r) for r in R], "size": _l(np.asarray(so.size, dtype=float)),
                                 "libero_parent_name": so.parent_name}
    # arena (static) collision geoms
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b in claimed or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        bname = m.body_id2name(b) or ""
        if bname.startswith(ROBOT_PREFIXES):
            continue
        t = int(m.geom_type[g])
        rec = {"geom": m.geom_id2name(g), "body": bname, "type": GEOM_TYPES.get(t, str(t))}
        if t == 0:  # plane: a half-space below its surface (normal = geom z)
            Rg = np.asarray(d.geom_xmat[g]).reshape(3, 3)
            rec.update(point=_l(d.geom_xpos[g]), normal=_l(Rg[:, 2]))
        else:
            half, off = geom_box(m, g)
            Rg = np.asarray(d.geom_xmat[g]).reshape(3, 3)
            rec.update(centre=_l(np.asarray(d.geom_xpos[g]) + Rg @ off), R=[_l(r) for r in Rg], half=_l(half),
                       exact=t == 6)
        scene["arena"].append(rec)

    def f1_check(stage: str):
        err = 0.0
        for g, rec, b in pending_f1:
            bp, bR = body_pose(b)
            err = max(err, f1_error(m, d, g, rec, bp, bR))
        f1_acc.append({"suite": suite, "task": ti, "stage": stage, "n_geoms": len(pending_f1), "max_err": err,
                       "pass": err <= F1_TOL})

    f1_check("after set_init_state")
    for _ in range(MP.WAIT_STEPS):
        obs, _r, _d, _i = env.step([0.0] * 6 + [-1.0])
    f1_check("after the 10 wait steps")
    robot = robot_record(env, obs)
    scene["robot_start"] = {"eef_pos": _l(obs["robot0_eef_pos"], 6), "eef_quat_xyzw": _l(obs["robot0_eef_quat"], 6),
                            "joint_pos": _l(obs["robot0_joint_pos"], 6),
                            "gripper_qpos": _l(obs["robot0_gripper_qpos"], 6)}
    scene["obs_keys"] = sorted(obs.keys())
    scene["robot_mount"] = {"base_pos": _l(d.xpos[m.body_name2id("robot0_base")], 6),
                            "shoulder_pos": _l(d.xpos[m.body_name2id("robot0_link2")], 6),
                            "note": "the robot's own mounting (fixed per scene type), for the agent's reach model"}
    env.close()
    return {"scene": scene, "robot": robot}


def robot_record(env, obs) -> dict:
    """Hand, finger and pad collision geoms in the eef frame (eef_pos = grip site, eef_quat = hand body)."""
    import numpy as np

    m, d = env.sim.model, env.sim.data
    p = np.asarray(obs["robot0_eef_pos"], dtype=float)
    qx, qy, qz, qw = [float(v) for v in obs["robot0_eef_quat"]]
    R = quat_to_mat([qw, qx, qy, qz])
    out = {"frame": "eef: origin robot0_eef_pos (grip site), axes of robot0_eef_quat (hand body); recorded after "
                    "the 10 wait steps (gripper open)", "geoms": {}, "finger_joints": {}}
    for g in range(m.ngeom):
        bname = m.body_id2name(int(m.geom_bodyid[g])) or ""
        if not bname.startswith("gripper0_") or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        half, off = geom_box(m, g)
        Rg = np.asarray(d.geom_xmat[g]).reshape(3, 3)
        c = np.asarray(d.geom_xpos[g]) + Rg @ off
        if int(m.geom_type[g]) == 7:
            mid = int(m.geom_dataid[g])
            a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            v_local = np.asarray(m.mesh_vert[a:a + n], dtype=float)
        else:
            v_local = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + off
        v_eef = (v_local @ Rg.T + np.asarray(d.geom_xpos[g]) - p) @ R
        out["geoms"][m.geom_id2name(g)] = {"body": bname, "type": GEOM_TYPES.get(int(m.geom_type[g])),
                                          "centre_eef": _l(R.T @ (c - p)), "R_eef": [_l(r) for r in R.T @ Rg],
                                          "half": _l(half), "aabb_eef_lo": _l(v_eef.min(0)),
                                          "aabb_eef_hi": _l(v_eef.max(0)),
                                          "vertices_eef": [_l(v, 6) for v in v_eef]}
    for jn in ("gripper0_finger_joint1", "gripper0_finger_joint2"):
        j = m.joint_name2id(jn)
        b = int(m.jnt_bodyid[j])
        axis_w = np.asarray(d.xmat[b]).reshape(3, 3) @ np.asarray(m.jnt_axis[j])
        out["finger_joints"][jn] = {"axis_eef": _l(R.T @ axis_w), "qpos": round(float(d.qpos[m.jnt_qposadr[j]]), 9),
                                    "range": _l(m.jnt_range[j])}
    out["gripper_qpos_obs"] = _l(obs["robot0_gripper_qpos"])
    return out


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_json(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return sha256_bytes(data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(MP.LIBRARY_DIR))
    a = ap.parse_args(argv)
    from rowant_lab.hybrid_frontier.e9 import isolation

    guard = isolation.scan()
    if not guard["clear"]:
        print("REFUSE: isolation guard not clear", file=sys.stderr)
        return 3
    out = Path(a.out)
    types_acc: dict = {}
    f1: list = []
    files = {}
    robot = None
    t0 = time.time()
    for suite in MP.SUITES:
        for ti in range(MP.N_TASKS):
            r = export_task(suite, ti, out, types_acc, f1)
            files[f"scenes/{suite}/{r['scene']['task_name']}.json"] = write_json(
                out / "scenes" / suite / f"{r['scene']['task_name']}.json", r["scene"])
            if robot is None:
                robot = r["robot"]
            print(f"{suite} {ti}: {len(r['scene']['instances'])} instances, F1 max "
                  f"{max(x['max_err'] for x in f1[-2:]):.2e}", flush=True)
    for typ, v in sorted(types_acc.items()):
        v["model"]["seen_in"] = v["seen_in"]
        files[f"types/{typ}.json"] = write_json(out / "types" / f"{typ}.json", v["model"])
    files["robot.json"] = write_json(out / "robot.json", robot)
    manifest = {"format": FORMAT, "plan_sha256": MP.PLAN_SHA256, "label": MP.LABEL,
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "elapsed_s": round(time.time() - t0, 1),
                "exporter_sha256": sha256_bytes(Path(__file__).read_bytes()), "files": files,
                "f1": {"tol": F1_TOL, "rows": f1, "pass": all(x["pass"] for x in f1)},
                "isolation_utc": guard["time_utc"]}
    write_json(out / "library-manifest.json", manifest)
    print(f"F1 {'PASS' if manifest['f1']['pass'] else 'FAIL'}; {len(files)} files; {manifest['elapsed_s']} s")
    return 0 if manifest["f1"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
