"""Training frames for the colour-only depth net (Pro2.0 training split only). Truth is allowed here: this is training.

``plan`` (Mac) writes ``frames-plan.json.gz`` from the manifest and stored action chains. ``run`` (pod, venv-pro2dev)
makes the frames: agentview at 1024 px, colour and the simulator's depth, in OpenCV row order, with the camera matrices
and the proprioception.

* A frame at the start of every (task, case) for start states 0, 1, 4, 5, 6, 7.
* A frame every ``EVERY`` steps while replaying stored action chains of earlier runs.
* With ``plan --random-camera P`` (item 2b, 2026-11-13): each frame is taken, with probability P, from a camera drawn
  near the scene's own (``random_camera``: turned about a vertical axis through a point ahead of it, raised or lowered,
  shifted, re-aimed, with its own field of view), so a net trained on them cannot lean on one camera's geometry. The draw
  is seeded by the frame's name, the scene's camera is put back after the frame, and the frame records the camera it
  was taken from (``K``, ``T``) and ``rc`` (1 for a drawn camera). Its file name ends in ``x``.

Start states 2, 3 and 9 are never rendered here: state 3 is the colour line's fresh check (the depth line's result on it
exists), and 2 and 9 stay reserved. Base tasks 7 and 9 of each suite are the judge tasks: their frames are the validation
split only, so the net is never trained on them. Task files on the pod are training tasks only (no test task, no S06).

For the long suite and the plain test-task scenes (2026-11-15; ``colour_long.py plan`` writes such plans, this file's own
``plan`` is unchanged), a plan item may also carry:

* ``key``: the name its frames are filed under instead of the suite (the long suite's table, ``libero_long_...``; the
  simulator still gets ``suite``);
* ``plain``: the item is the plain version of a task (the simulator's own base file); its frames go under ``PLAIN``;
* ``sweep``: ``{"n", "steps"}``, a scripted arm motion (``colour_sweep.py``) where no stored chain exists, with a
  frame every ``every`` steps (none at step 0: the reset item has that frame).

A perturbed episode of a benchmark test task is refused at planning and again by the worker (``split_guard``).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.pro2.machine import BENCHMARK  # noqa: E402
from rowant_lab.settings import DATA  # noqa: E402

COLD = DATA
PAYLOAD = BENCHMARK
MANIFEST = PAYLOAD / "service-source/assets/task_manifest.json"
WRITER = COLD / "runs/pro2/writer"
SUITES = ("libero_object", "libero_spatial", "libero_goal")
TRAIN_STATES = (0, 1, 4, 5, 6, 7)
VAL_RESET_STATES = (8,)
JUDGE_TASKS = (7, 9)
EVERY = 12
VAL_EVERY = 30
SIZE = 1024
# the drawn cameras (item 2b): a general spread around the scene's own camera, not fitted to any benchmark case
RANDOM_CAMERA = {"pivot_m": 1.15,  # the turn's axis passes this far ahead of the camera (about the workspace)
                 "yaw_deg": 35.0, "height_m": 0.15, "shift_m": 0.15, "aim_m": 0.08, "fovy_deg": (35.0, 60.0)}


def split_of(task: int) -> str:
    return "val" if task in JUDGE_TASKS else "train"


def random_camera(T_world_from_cv, rng, o: dict | None = None) -> dict:
    """A camera drawn near the one at ``T_world_from_cv`` (OpenCV frame: z along the optical axis), in MuJoCo's form
    (``pos``, ``quat_wxyz``, ``fovy_deg``). Its position is turned about the vertical axis through the point ``pivot_m``
    ahead on the optical axis, moved up or down and sideways, and it looks at that point moved by up to ``aim_m``. Up
    stays the world's z (no roll). ``rng``: a ``random.Random``."""
    import math

    import numpy as np

    from rowant_lab.pro2.camera_math import _quat_wxyz, look_at_mujoco

    o = {**RANDOM_CAMERA, **(o or {})}
    T = np.asarray(T_world_from_cv, dtype=np.float64)
    p0, fwd = T[:3, 3], T[:3, 2]
    pivot = p0 + float(o["pivot_m"]) * fwd
    a = math.radians(rng.uniform(-o["yaw_deg"], o["yaw_deg"]))
    Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    s, h, m = float(o["shift_m"]), float(o["height_m"]), float(o["aim_m"])
    pos = pivot + Rz @ (p0 - pivot) + np.array([rng.uniform(-s, s), rng.uniform(-s, s), rng.uniform(-h, h)])
    aim = pivot + np.array([rng.uniform(-m, m), rng.uniform(-m, m), rng.uniform(-m, m)])
    R = look_at_mujoco(pos, aim)
    return {"pos": [float(v) for v in pos], "quat_wxyz": _quat_wxyz(R), "fovy_deg": float(rng.uniform(*o["fovy_deg"]))}


def frame_kind(it: dict) -> str:
    """``c`` (a replayed chain), ``r`` (a reset) or ``s<n>`` (the n-th sweep of this start)."""
    if it.get("sweep"):
        return f"s{int(it['sweep']['n']):02d}"
    return "c" if it["chain"] else "r"


def frame_seed(it: dict, step: int) -> str:
    return f"{it['suite']}|{it['task']}|{it['case']}|{it['state']}|{frame_kind(it)}|{step}"


def stored_chains() -> dict:
    """(suite, task, case, state, seed) -> actions: every r50 episode, and the Object and Spatial episodes of state 1
    from robot r31 (no later robot ran those)."""
    out: dict = {}

    def add(d: Path, keep) -> None:
        for f in sorted(d.glob("w*/rows.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if "success" in r and r.get("actions") and keep(r):
                    out[(r["suite"], r["task"], r["case"], r["state"], r["seed"])] = r["actions"]

    add(WRITER / "rows/r50", lambda r: True)
    add(WRITER / "rows/r31", lambda r: r["state"] == 1 and r["suite"] in ("libero_object", "libero_spatial"))
    return out


def cmd_plan(a) -> int:
    from rowant_lab.pro2 import split_guard

    m = json.loads(MANIFEST.read_text())
    items, seen = [], set()
    chains = stored_chains()
    bddl_of = {}
    for e in m["tasks"]:
        bddl = (MANIFEST.parent / e["bddl_path"]).resolve()
        if e["suite"] not in SUITES or str(e["case_id"]).startswith("S06") or not bddl.is_file():
            continue
        t = int(e["base_task_id"])
        bddl_of[(e["suite"], t, e["case_id"])] = (str(bddl), int(e["max_steps"]))
        if (e["suite"], t, e["case_id"]) in seen:
            continue
        seen.add((e["suite"], t, e["case_id"]))
        split = split_of(t)
        states = VAL_RESET_STATES + TRAIN_STATES[:2] if split == "val" else TRAIN_STATES
        for s in states:
            items.append({"suite": e["suite"], "task": t, "case": e["case_id"], "state": s, "seed": 142 + s,
                          "bddl": str(bddl), "max_steps": int(e["max_steps"]), "split": split, "every": 0, "chain": None,
                          "rcam": float(a.random_camera)})
    n_reset = len(items)
    for (suite, task, case, state, seed), actions in sorted(chains.items()):
        if (suite, task, case) not in bddl_of:
            continue
        bddl, max_steps = bddl_of[(suite, task, case)]
        split = split_of(task)
        items.append({"suite": suite, "task": task, "case": case, "state": state, "seed": seed, "bddl": bddl,
                      "max_steps": max_steps, "split": split, "every": VAL_EVERY if split == "val" else EVERY,
                      "chain": actions, "rcam": float(a.random_camera)})
    split_guard.check(items, "frame plan")  # never a perturbed episode of a benchmark test task
    a.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out / "frames-plan.json.gz", "wt", compresslevel=5) as f:
        json.dump(items, f)
    est = sum(1 + (len(i["chain"]) // i["every"] if i["every"] else 0) for i in items)
    print(json.dumps({"items": len(items), "resets": n_reset, "replays": len(items) - n_reset, "frames_est": est,
                      "random_camera": float(a.random_camera)}))
    return 0


def load_plan(d: Path) -> list:
    with gzip.open(d / "frames-plan.json.gz", "rt") as f:
        return json.load(f)


def frame_key(it: dict) -> str:
    """The name an item's frames are filed under: its ``key`` (a long-suite table), else its suite."""
    return str(it.get("key") or it["suite"])


def frame_name(it: dict, step: int, drawn: bool = False) -> str:
    if it.get("sweep"):
        kind = f"s{it['state']}-{int(it['sweep']['n']):02d}"
    else:
        kind = f"{'c' if it['chain'] else 'r'}{it['state']}"
    case = "PLAIN" if it.get("plain") else it["case"]
    return f"{it['split']}/{frame_key(it)}/{it['task']:02d}/{case}/{kind}_{step:04d}{'x' if drawn else ''}.npz"


def drawn_frame(path) -> bool:
    """A frame taken from a drawn camera (``random_camera``)."""
    return str(path).endswith("x.npz")


class _Camera:
    """Put a drawn camera into the simulator for one frame, then the scene's own back (``agentview``, fixed to the
    world)."""

    def __init__(self, env, name: str = "agentview"):
        import numpy as np

        self.env, self.name, self.np = env, name, np

    def __enter__(self):
        return self

    def set(self, cam: dict) -> None:
        np = self.np
        sim = self.env.sim
        cid = sim.model.camera_name2id(self.name)
        if int(sim.model.cam_bodyid[cid]) != 0:
            raise RuntimeError(f"{self.name} is attached to body {int(sim.model.cam_bodyid[cid])}, not the world")
        self.saved = (cid, np.array(sim.model.cam_pos[cid]), np.array(sim.model.cam_quat[cid]), float(sim.model.cam_fovy[cid]))
        sim.model.cam_pos[cid] = np.asarray(cam["pos"], dtype=np.float64)
        sim.model.cam_quat[cid] = np.asarray(cam["quat_wxyz"], dtype=np.float64)
        sim.model.cam_fovy[cid] = float(cam["fovy_deg"])
        sim.forward()

    def __exit__(self, *exc):
        if getattr(self, "saved", None) is not None:
            sim = self.env.sim
            cid, pos, quat, fovy = self.saved
            sim.model.cam_pos[cid] = pos
            sim.model.cam_quat[cid] = quat
            sim.model.cam_fovy[cid] = fovy
            sim.forward()
            self.saved = None
        return False


def cmd_worker(a) -> int:
    from rowant_lab.libero_env import m2_rpaths as MP

    MP.pin_numba_cache()
    import numpy as np

    from rowant_lab.libero_env import m2_camera as CAM
    from rowant_lab.pro2 import dev_episode as DE
    from rowant_lab.pro2 import split_guard

    items = load_plan(a.plan)[a.index::a.of]
    split_guard.check(items, "frame worker")  # before any simulator is built
    done_dir = a.out / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out / f"meta-{a.index:02d}.jsonl", "a") as fo:
        for j, it in enumerate(items):
            mark = done_dir / f"{a.index:02d}-{j:06d}"
            if mark.exists():
                continue
            if a.max_items and n >= a.max_items:
                return 0
            n += 1
            ep = {k: it[k] for k in ("suite", "task", "case", "state", "seed", "bddl", "max_steps")}
            t0 = time.perf_counter()
            ad = DE.adapter_class()()
            status = "ok"
            swept = None  # a sweep item's record: steps run, legs started, legs that reached their point, retreats
            try:
                first = ad.create(DE.env_spec(ep, 16))
                look = CAM.make_camera_tool(ad._env)
                obs = DE.proprio(first["observation"])
                frames = []

                def grab(step: int) -> None:
                    rng = random.Random(frame_seed(it, step))
                    drawn = float(it.get("rcam") or 0.0) > 0.0 and rng.random() < float(it["rcam"])
                    with _Camera(ad._env) as cam:
                        if drawn:
                            sim = ad._env.sim
                            cid = sim.model.camera_name2id("agentview")
                            Tgl = np.eye(4)
                            Tgl[:3, :3] = np.asarray(sim.data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
                            Tgl[:3, 3] = np.asarray(sim.data.cam_xpos[cid], dtype=np.float64)
                            cam.set(random_camera(Tgl @ CAM.CV_FROM_GL, rng))
                        fr = look("agentview", SIZE, step)
                    p = a.out / "frames" / frame_name(it, step, drawn)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(p, rgb=fr["rgb"], depth=fr["depth_u16"], K=fr["K"], T=fr["T_world_from_camera"],
                                        q=np.asarray(obs["robot0_joint_pos"], np.float32),
                                        g=np.asarray(obs["robot0_gripper_qpos"], np.float32), rc=np.uint8(drawn))
                    frames.append({"step": step, "t": [round(float(x), 6) for x in fr["T_world_from_camera"][:3, 3]],
                                   "rc": int(drawn)})

                if not it.get("sweep"):
                    grab(0)
                if it["chain"]:
                    for k, act in enumerate(it["chain"]):
                        r = ad.step(np.asarray(act, dtype=np.float32))
                        obs = DE.proprio(r["observation"])
                        if (k + 1) % it["every"] == 0:
                            grab(k + 1)
                        if r["terminated"] or r["truncated"]:
                            break
                elif it.get("sweep"):  # no stored chain for this scene: a scripted arm motion (colour_sweep.py)
                    from rowant_lab.pro2 import colour_sweep as SW

                    env = ad._env
                    sw = SW.Sweep(frame_seed(it, 0), SW.hand(env), SW.shoulder(env), R0=SW.hand_rot(env))
                    swept = {"steps": 0}
                    for k in range(int(it["sweep"]["steps"])):
                        act = sw.act(SW.hand(env), lambda: SW.anchors(env), SW.joints(env), SW.hand_rot(env))
                        if sw.stuck:  # the hand has not moved for a while: the frames so far are the sweep's
                            swept["stuck"] = True
                            break
                        r = ad.step(np.asarray(act, dtype=np.float32))
                        obs = DE.proprio(r["observation"])
                        swept.update(steps=k + 1, legs=sw.n_legs, reached=sw.n_reached, retreats=sw.n_retreats)
                        if (k + 1) % it["every"] == 0:
                            grab(k + 1)
                        if r["terminated"] or r["truncated"]:
                            break
            except Exception as exc:  # a case that refuses a start or a chain: no frames from it
                status = f"{type(exc).__name__}: {exc}"[:200]
                frames = []
            finally:
                ad.close()
            row = {"item": [it["suite"], it["task"], it["case"], it["state"]], "split": it["split"],
                   "status": status, "n": len(frames), "s": round(time.perf_counter() - t0, 2),
                   "t0": frames[0]["t"] if frames else None}
            if it.get("key") or it.get("plain") or it.get("sweep"):  # the long suite's and the plain scenes' items
                row.update(key=frame_key(it), plain=bool(it.get("plain")), kind=frame_kind(it))
                if swept is not None:
                    row["sweep"] = swept
            fo.write(json.dumps(row) + "\n")
            fo.flush()
            mark.touch()
    return 0


def cmd_run(a) -> int:
    import threading

    a.out.mkdir(parents=True, exist_ok=True)
    rc = [None] * a.workers
    plan_n = len(load_plan(a.plan))

    def left(i: int) -> int:
        n = len(range(i, plan_n, a.workers))
        return n - len(list((a.out / "done").glob(f"{i:02d}-*"))) if (a.out / "done").is_dir() else n

    def keep(i: int) -> None:
        while left(i) > 0:
            with open(a.out / f"worker-{i:02d}.log", "a") as lf:
                rc[i] = subprocess.call(
                    [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "worker", "--plan", str(a.plan),
                     "--out", str(a.out), "--index", str(i), "--of", str(a.workers), "--max-items", str(a.max_items)],
                    cwd=Path(__file__).resolve().parents[2], stdout=lf, stderr=subprocess.STDOUT, env=dict(os.environ))
            if rc[i] != 0:
                return

    th = [threading.Thread(target=keep, args=(i,)) for i in range(a.workers)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    print(json.dumps({"worker_rc": rc}))
    return 0 if all(r == 0 for r in rc) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "run", "worker"))
    ap.add_argument("--plan", type=Path, help="the folder holding frames-plan.json.gz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--max-items", type=int, default=40, help="episodes per worker process before it restarts")
    ap.add_argument("--random-camera", type=float, default=0.0,
                    help="plan: the share of frames taken from a drawn camera (item 2b); 0 keeps the scene's own")
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run, "worker": cmd_worker}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
