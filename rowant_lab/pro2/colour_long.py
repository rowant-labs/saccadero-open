"""The colour-only line on the long-task suite's tables: frames, stored cameras and the acceptance check (2026-11-15).

The first depth net (``colour_net``; ``runs/pro2/colour/net/last.pt``) was trained on renders of three suites, and the
colour line reads one stored camera per suite. The long suite (LIBERO-10) stands at three tables (the "cells" of
``long_paths.CELLS``), which had neither. Everything here is keyed by the cell, the ``suite`` the runner hands to the
colour line's camera tool (``dev_episode._act``)::

    plain          (Mac, venv-libero -I) the PLAIN versions of the benchmark's test tasks: the simulator's own base task
                   files through LIBERO's API, with ``long_lib``'s config -> ``plain-tasks.json``. No simulator starts.
    cameras        (Mac, venv-libero -I; or a pod) each cell's agentview camera, read from the
                   simulator on BASE tasks as the three suites' were -> ``colour_cameras_long.json``
    plan           (Mac, any Python) ONE frame plan: the first net's plan as it is, the long suite's training tasks
                   from an episode list of ``dev_run.py plan``, and the plain test-task scenes -> ``frames-plan.json.gz``
    frames-report  (pod, after the frames) what the frame run made of the plan: failed items, frames per key
    check-cameras  (pod, after the frames) the stored cameras against the cameras the frames were taken from
    accept         (pod, after training) two ``colour_score.py --by-key`` files against the acceptance rules

The pod driver (frames, the current net's scores, training, the new net's scores, accept) is not part of this
release; ``scripts/train_depth_net.sh`` runs the frames, the training and the scores.

**What the plan holds** (``colour_frames`` renders it; the old suites' items are the first net's, unchanged):

* long suite, seen tasks (0, 1, 2, 3, 4, 6), every case of the episode list: a frame at start states 0, 1, 4, 5, 6, 7
  (training); tasks 7 and 9 (ours, held back): start states 8, 0, 1, validation only. As the three suites.
* a moving arm at start states 0 and 1 of each of those: a stored action chain where ``--rows`` has one, else a sweep
  (``colour_sweep.py``), a frame every 12 steps (30 for validation), as the stored chains were replayed.
* the plain version of a test task (``split_guard``: BASE only, the simulator's own file): a frame at each training
  start state, and sweeps from each (``SWEEPS``). The plain long-suite scenes (tasks 5 and 8) are also taken from
  start state 8, which is never trained on, as validation: the scenes a scored run stands in, from a start the net has
  not seen. For the study table (task 5 is its only task) that is all the validation there is: a weaker check than
  another task. Each long-suite task is its own scene, so the kitchen's held-back task 9 checks a scene, with fixtures,
  that no kitchen training frame shows; ``accept`` marks a table task by task for that reason.
* never: a perturbed episode of a test task, start states 2, 3 and 9, or a training frame from start state 8.

Truth is used only where the frame factory always used it (training frames and their labels) and to read the cameras.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.pro2 import colour_frames as CF  # noqa: E402
from rowant_lab.pro2 import long_paths as LP  # noqa: E402
from rowant_lab.pro2 import split_guard as SG  # noqa: E402
from rowant_lab.settings import DATA  # noqa: E402

COLD = DATA
OUT = COLD / "runs/pro2/colour-long"
OLD_PLAN = COLD / "runs/pro2/colour"  # the first net's frame plan (the three suites)
CELLS_FILE = COLD / "runs/pro2/long-lib/build/cells.json"  # which task stands at which table (long_lib's record)
PLAIN_CONFIG = COLD / "runs/pro2/long-lib/build/libero-config"  # LIBERO's config naming the simulator's own task files
PAYLOAD = CF.PAYLOAD  # the benchmark's download (rowant_lab/pro2/machine.py)
SIM = PAYLOAD / "simulator-source" / "libero" / "libero"
LONG_CAMERAS = Path(__file__).resolve().parent / "colour_cameras_long.json"
SUITE = LP.SUITE
MAX_STEPS = {"libero_object": 280, "libero_spatial": 220, "libero_goal": 300, "libero_10": 520}  # the suites' step caps
MOTION_STATES = CF.TRAIN_STATES[:2]  # the start states that get a moving arm (the stored chains' were 0 and 1)
SWEEPS = {
    "seen": 1,         # per (task, case) and motion state of a long-suite training task without a stored chain
    "plain_long": 8,   # per training start state of a plain long-suite scene (task 5: the study table's only data)
    "plain_other": 2,  # per training start state of a plain Object, Spatial or Goal scene (sibling tasks cover these)
    "plain_val": 4,    # a plain long-suite scene's validation: the scene from start state 8
    "steps": 240,
}
CAMERA_STATE, CAMERA_SEED = 0, 142  # the start the cameras are read at (a BASE scene's camera does not depend on it)
CAMERA_TOL = {"pos_m": 1e-6, "rot_deg": 1e-4, "fovy_deg": 1e-9}  # a cell's tasks must show one camera
FRAME_TOL = {"pos_m": 1e-4, "rot_deg": 0.01, "fovy_deg": 1e-6}  # stored camera against the frames' own (check-cameras)
ACCEPT = {
    # a key the first net was trained on (the three suites): the new net against the current one, on the same frames
    "old": {"blob_offset_p90_add_mm": 1.0, "blob_offset_p90_rel": 0.15, "within_10mm_drop": 0.02,
            "table_median_add_mm": 0.5},
    # a new key (a long-suite table): absolute, at the first net's level on its own suites (6.4 mm, 0.90, 0.6 mm)
    "new": {"blob_offset_p90_mm": 10.0, "within_10mm": 0.85, "table_median_mm": 1.5},
    "gate_mm": 40.0,  # the pod driver's gate: above this a net's depth is garbage in the perception server
}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ cells (pure)
def task_cells(doc: dict) -> dict:
    """``{task: cell}`` from ``long_lib``'s build record (``build/cells.json``). Every cell must be one the runner
    knows (``long_paths.CELLS``) and every task stand in exactly one."""
    out: dict = {}
    for cell, c in doc["cells"].items():
        if cell not in LP.CELLS:
            raise ValueError(f"the build record names a cell the runner does not know: {cell}")
        for t in c["tasks"]:
            if int(t) in out:
                raise ValueError(f"task {t} stands in two cells: {out[int(t)]} and {cell}")
            out[int(t)] = cell
    return out


def load_cells(path=CELLS_FILE) -> dict:
    return task_cells(json.loads(Path(path).read_text()))


def key_for(suite: str, task: int, cells: dict) -> str:
    """The key an episode's frames and camera go by: the long suite's table, else the suite."""
    if suite != SUITE:
        return suite
    if int(task) not in cells:
        raise ValueError(f"{SUITE} task {task} stands in no known cell")
    return cells[int(task)]


# ------------------------------------------------------------------------------------------------ plan (pure)
def _item(suite, task, case, state, seed, bddl, max_steps, split, key, every=0, chain=None, **more) -> dict:
    it = {"suite": suite, "task": int(task), "case": case, "state": int(state), "seed": int(seed), "bddl": str(bddl),
          "max_steps": int(max_steps), "split": split, "every": int(every), "chain": chain, "rcam": 0.0}
    if key != suite:
        it["key"] = key
    it.update(more)
    return it


def allowed_states(split: str) -> tuple:
    """The start states a split's frames may come from (``colour_frames``' rule)."""
    return CF.VAL_RESET_STATES + CF.TRAIN_STATES[:2] if split == "val" else CF.TRAIN_STATES


def chains_from(dirs) -> dict:
    """``(suite, task, case, state, seed) -> actions`` from finished rows of earlier runs (``w*/rows.jsonl``)."""
    out: dict = {}
    for d in dirs or ():
        for f in sorted(Path(d).glob("w*/rows.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if "success" in r and r.get("actions"):
                    out[(r["suite"], int(r["task"]), r["case"], int(r["state"]), int(r["seed"]))] = r["actions"]
    return out


def long_items(episodes: list, cells: dict, chains: dict | None = None, sweeps: int = SWEEPS["seen"],
               steps: int = SWEEPS["steps"]) -> list:
    """The long suite's training-split items from an episode list of ``dev_run.py plan --suites libero_10`` (any
    states: only each (task, case)'s file and step cap are taken). Seen tasks: training frames; tasks 7 and 9:
    validation only."""
    SG.check(episodes, "the long suite's episode list")
    pairs: dict = {}
    for e in episodes:
        if e["suite"] != SUITE:
            raise ValueError(f"the episode list holds another suite: {e['suite']}")
        if SG.is_test_task(SUITE, e["task"]):  # a test task has no Pro2.0 file here; its plain version is plain_items'
            raise SG.Refused(f"the episode list holds test task {e['task']}")
        if str(e["case"]).startswith("S06"):
            raise ValueError("the episode list holds an S06 case (held)")
        k = (int(e["task"]), str(e["case"]))
        v = (str(e["bddl"]), int(e["max_steps"]))
        if pairs.setdefault(k, v) != v:
            raise ValueError(f"task {k[0]} case {k[1]} has two files or step caps in the episode list")
    chains = chains or {}
    by_tcs: dict = {}
    for (suite, task, case, state, seed), actions in chains.items():
        if suite == SUITE and seed == 142 + state:
            by_tcs[(task, case, state)] = actions
    items = []
    for (task, case), (bddl, cap) in sorted(pairs.items()):
        split, key = CF.split_of(task), key_for(SUITE, task, cells)
        every = CF.VAL_EVERY if split == "val" else CF.EVERY
        for s in allowed_states(split):
            items.append(_item(SUITE, task, case, s, 142 + s, bddl, cap, split, key))
        for s in allowed_states(split):
            if (task, case, s) in by_tcs:  # a stored chain of an earlier run, replayed as the three suites' were
                items.append(_item(SUITE, task, case, s, 142 + s, bddl, cap, split, key, every, by_tcs[(task, case, s)]))
            elif s in MOTION_STATES:
                for n in range(int(sweeps)):
                    items.append(_item(SUITE, task, case, s, 142 + s, bddl, cap, split, key, every,
                                       sweep={"n": n, "steps": min(int(steps), cap)}))
    return items


def plain_items(plain: list, cells: dict, sweeps_long: int = SWEEPS["plain_long"],
                sweeps_other: int = SWEEPS["plain_other"], sweeps_val: int = SWEEPS["plain_val"],
                steps: int = SWEEPS["steps"]) -> list:
    """The plain versions of test tasks (``plain``: rows of ``plain-tasks.json``): BASE only, training frames at the
    training start states and sweeps from each. A plain long-suite scene is also taken from the validation start
    state, as validation only (for the study table, which has no held-back task of ours, the only validation). The
    three suites' plain scenes add no validation frame: their validation stays what the first net's was."""
    items = []
    for p in sorted(plain, key=lambda r: (r["suite"], int(r["task"]))):
        suite, task, bddl = str(p["suite"]), int(p["task"]), str(p["bddl"])
        if not SG.is_test_task(suite, task):
            raise ValueError(f"{suite} task {task} is not a test task: its frames come from the training plan")
        if not SG.is_plain_file(suite, bddl):
            raise SG.Refused(f"{suite} task {task}: {bddl} is not the simulator's own base task file")
        cap = int(p.get("max_steps") or MAX_STEPS[suite])
        key = key_for(suite, task, cells)
        parts = [("train", CF.TRAIN_STATES, int(sweeps_long if suite == SUITE else sweeps_other))]
        if suite == SUITE:  # the scene a scored run stands in, from a start the net never trains on
            parts.append(("val", CF.VAL_RESET_STATES, int(sweeps_val)))
        for split, states, count in parts:
            for s in states:
                items.append(_item(suite, task, "BASE", s, 142 + s, bddl, cap, split, key, plain=True))
                for k in range(count):
                    items.append(_item(suite, task, "BASE", s, 142 + s, bddl, cap, split, key, CF.EVERY, plain=True,
                                       sweep={"n": k, "steps": min(int(steps), cap)}))
    return items


def frames_of(it: dict) -> int:
    """How many frames an item gives when it runs to its end."""
    if it.get("sweep"):
        return int(it["sweep"]["steps"]) // int(it["every"])
    return 1 + (len(it["chain"]) // int(it["every"]) if it["chain"] and it["every"] else 0)


def check_plan(items: list) -> None:
    """Refuse a plan that holds a perturbed test-task episode, a start state that is never rendered, a training frame
    from a validation-only task or start state, or two items writing the same frames."""
    SG.check(items, "frame plan")
    seen: set = set()
    for it in items:
        if it["state"] not in allowed_states(it["split"]):
            raise ValueError(f"start state {it['state']} is not a {it['split']} state: {CF.frame_name(it, 0)}")
        if it["split"] == "train" and it["task"] in CF.JUDGE_TASKS:
            raise ValueError(f"a training frame of held-back task {it['task']}: {CF.frame_name(it, 0)}")
        if it["split"] not in ("train", "val"):
            raise ValueError(f"unknown split {it['split']!r}")
        if it["split"] == "val" and it["task"] not in CF.JUDGE_TASKS \
                and not (it.get("plain") and it["state"] in CF.VAL_RESET_STATES):
            raise ValueError(f"a validation frame that is neither a held-back task's nor a plain scene's from the "
                             f"validation start: {CF.frame_name(it, 0)}")
        name = CF.frame_name(it, 0)
        if name in seen:
            raise ValueError(f"two items write {name}")
        seen.add(name)


def summary(items: list) -> dict:
    """Items and frames (an estimate: every item to its end) by key, split and kind; the plain scenes covered."""
    n, f = Counter(), Counter()
    plain: dict = {}
    for it in items:
        kind = "sweep" if it.get("sweep") else ("chain" if it["chain"] else "reset")
        k = f"{CF.frame_key(it)}/{it['split']}/{kind}"
        n[k] += 1
        f[k] += frames_of(it)
        f[f"{CF.frame_key(it)}/{it['split']}"] += frames_of(it)
        if it.get("plain"):
            row = plain.setdefault(f"{it['suite']}/{it['task']}", {
                "suite": it["suite"], "task": it["task"], "key": CF.frame_key(it), "scene": Path(it["bddl"]).stem,
                "states": {"train": set(), "val": set()}, "frames": 0})
            row["states"][it["split"]].add(it["state"])
            row["frames"] += frames_of(it)
    for row in plain.values():
        row["states"] = {k: sorted(v) for k, v in row["states"].items() if v}
    return {"items": len(items), "frames_est": sum(frames_of(it) for it in items),
            "items_by": dict(sorted(n.items())), "frames_by": dict(sorted(f.items())),
            "plain_scenes": [plain[k] for k in sorted(plain)]}


def frames_report(items: list, rows: list, on_disk: dict | None = None) -> dict:
    """What a frame run made of its plan. ``rows``: the workers' meta rows; ``on_disk``: frame files found, by
    ``key/split``. Gives the items finished, the items that ended on an error (by its kind; their earlier frames
    stay), and the frames per key and split against the plan's estimate (a replay that ends early gives fewer)."""
    plan, got, bad = Counter(), Counter(), Counter()
    for it in items:
        plan[f"{CF.frame_key(it)}/{it['split']}"] += frames_of(it)
    for r in rows:
        got[f"{r.get('key') or r['item'][0]}/{r['split']}"] += int(r["n"])
        if r["status"] != "ok":
            bad[str(r["status"]).split(":")[0]] += 1
    have = got if on_disk is None else on_disk
    return {"items": len(items), "finished": len(rows), "failed": sum(bad.values()), "failed_by": dict(bad.most_common(8)),
            "frames": {k: [int(have.get(k, 0)), plan[k]] for k in sorted(plan)},
            "frames_total": [int(sum(have.get(k, 0) for k in plan)), sum(plan.values())]}


# ------------------------------------------------------------------------------------------------ cameras (pure)
def pose_gap(Ta, Tb) -> tuple:
    """(metres, degrees) between two camera poses (4x4, world from camera)."""
    import numpy as np

    Ta, Tb = np.asarray(Ta, dtype=float), np.asarray(Tb, dtype=float)
    c = (np.trace(Ta[:3, :3].T @ Tb[:3, :3]) - 1.0) / 2.0
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), float(math.degrees(math.acos(min(1.0, max(-1.0, float(c))))))


def fovy_of(K, height: float) -> float:
    """The vertical field of view (degrees) of intrinsics ``K`` for an image ``height`` pixels high."""
    return float(2.0 * math.degrees(math.atan(0.5 * float(height) / float(K[1][1]))))


def one_camera(readings: list, tol: dict = CAMERA_TOL) -> dict:
    """One camera from several tasks' readings (``{"fovy_deg", "T"}``): they must agree within ``tol``; the stored
    values are the first reading's (as measured, no averaging). Returns the entry and the spread."""
    if not readings:
        raise ValueError("no camera reading")
    first = readings[0]
    gaps = [pose_gap(first["T"], r["T"]) for r in readings]
    spread = {"pos_m": max(g[0] for g in gaps), "rot_deg": max(g[1] for g in gaps),
              "fovy_deg": max(abs(float(r["fovy_deg"]) - float(first["fovy_deg"])) for r in readings)}
    bad = [k for k in tol if spread[k] > tol[k]]
    if bad:
        raise ValueError(f"the tasks show different cameras: {spread}")
    return {"camera": {"fovy_deg": float(first["fovy_deg"]),
                       "T_world_from_camera": [[float(x) for x in row] for row in first["T"]]},
            "spread": spread}


def cameras_doc(by_key: dict, measured: dict, date: str) -> dict:
    """``colour_cameras_long.json``'s content (``colour_cameras.json``'s shape, plus how each was measured)."""
    return {"note": (f"Agentview camera of each long-suite table (the cell keys of long_paths.CELLS), measured on {date} "
                     "from the simulator on BASE tasks, as colour_cameras.json's were: the cell's seen training tasks "
                     "(Pro2.0's BASE files) and the plain version of its benchmark test task (the simulator's own "
                     "base file); 'measured' lists them. Camera constants fixed at development; cases that change the "
                     "camera (S15, S16, D07) are not covered by them."),
            "cameras": {k: by_key[k] for k in sorted(by_key)}, "measured": {k: measured[k] for k in sorted(measured)}}


# ------------------------------------------------------------------------------------------------ accept (pure)
STATS = ("blob_offset_p90_mm", "within_10mm", "table_median_mm")


def _stats(v: dict | None) -> dict:
    v = v or {}
    return {"blob_offset_p90_mm": (v.get("blob_offset") or {}).get("p90_mm"),
            "within_10mm": (v.get("obj") or {}).get("within_10mm"),
            "table_median_mm": (v.get("table") or {}).get("median_mm"),
            "frames": v.get("frames")}


def _missing(s: dict) -> bool:
    return any(s[x] is None for x in STATS)


def _over(n: dict, limits: dict) -> list:
    """Which of a score's three numbers break their limits, as text."""
    why = []
    if n["blob_offset_p90_mm"] > limits["blob_offset_p90_mm"]:
        why.append(f"blob offset p90 {n['blob_offset_p90_mm']} mm over {limits['blob_offset_p90_mm']}")
    if n["within_10mm"] < limits["within_10mm"]:
        why.append(f"object pixels within 10 mm {n['within_10mm']} under {limits['within_10mm']}")
    if n["table_median_mm"] > limits["table_median_mm"]:
        why.append(f"table median {n['table_median_mm']} mm over {limits['table_median_mm']}")
    return why


def accept(new: dict, old: dict | None, rules: dict = ACCEPT, old_keys=CF.SUITES, expect=None) -> dict:
    """The acceptance check on two ``by_key`` score blocks (``colour_score.py --by-key``): ``new`` is the candidate
    net's, ``old`` the current net's on the same frames.

    * A key in ``old_keys`` (the first net's suites) must not have got worse than ``rules["old"]`` allows.
    * Any other key (a long-suite table) must meet ``rules["new"]``, and where its score has ``by_task`` every one of
      its validation tasks must: each long-suite task is its own scene, and a table's verdict names the task that
      fails (the kitchen's held-back task 9 stands in a scene no kitchen training frame shows).
    * Every key in ``expect`` (default: the three suites and the three tables) and every key either file scores must
      be scored by the candidate, with object blobs in its frames.
    """
    old = old or {}
    expect = tuple(old_keys) + tuple(sorted(LP.CELLS)) if expect is None else tuple(expect)
    gate_mm = float(rules["gate_mm"])
    none = "no object blob or no table pixel in the validation frames"
    out: dict = {}
    for k in sorted(set(new) | set(old) | set(expect)):
        n, o = _stats(new.get(k)), _stats(old.get(k))
        why, limits, tasks = [], {}, {}
        kind = "old" if k in old_keys else "new"
        gate = n["blob_offset_p90_mm"] is not None and n["blob_offset_p90_mm"] <= gate_mm
        if k not in new:
            why.append("the candidate has no score for this key")
        elif kind == "old":
            r = rules["old"]
            if _missing(n):
                why.append(none)
            elif _missing(o):
                why.append("no score of the current net to compare with")
            else:
                limits = {"blob_offset_p90_mm": round(o["blob_offset_p90_mm"] + max(
                              r["blob_offset_p90_add_mm"], r["blob_offset_p90_rel"] * o["blob_offset_p90_mm"]), 3),
                          "within_10mm": round(o["within_10mm"] - r["within_10mm_drop"], 4),
                          "table_median_mm": round(o["table_median_mm"] + r["table_median_add_mm"], 3)}
                why += _over(n, limits)
        else:
            limits = dict(rules["new"])
            groups = new[k].get("by_task") or {}
            was = (old.get(k) or {}).get("by_task") or {}
            for t in sorted(groups, key=int):  # a table with several validation tasks: each one is marked
                tn = _stats(groups[t])
                tw = [none] if _missing(tn) else _over(tn, limits)
                tg = tn["blob_offset_p90_mm"] is not None and tn["blob_offset_p90_mm"] <= gate_mm
                tasks[t] = {"pass": not tw, "new": tn, "old": _stats(was[t]) if t in was else None, "gate": bool(tg),
                            "why": tw}
                why += [f"task {t}: {x}" for x in tw]
                gate = gate and tg
            if not groups:
                why += [none] if _missing(n) else _over(n, limits)
        out[k] = {"pass": not why, "kind": kind, "new": n, "old": o if k in old else None, "limits": limits,
                  "gate": bool(gate), "why": why}
        if tasks:
            out[k]["tasks"] = tasks
    return {"pass": all(v["pass"] for v in out.values()), "keys": out, "rules": rules}


def _line(ok: bool, name: str, kind: str, n: dict, o: dict | None, why: list) -> str:
    o = o or {}
    return (f"{'PASS' if ok else 'FAIL'} {name} ({kind}): blob offset p90 {n['blob_offset_p90_mm']} mm"
            f" (was {o.get('blob_offset_p90_mm')}), within 10 mm {n['within_10mm']} (was {o.get('within_10mm')}),"
            f" table median {n['table_median_mm']} mm (was {o.get('table_median_mm')})"
            + ("" if ok else ": " + "; ".join(why)))


def accept_lines(res: dict) -> list:
    """The check as text: a line per key (per task for a table marked task by task), then the verdict, which names
    what failed."""
    lines, failed = [], []
    for k, v in res["keys"].items():
        if v.get("tasks"):
            for t, tv in v["tasks"].items():
                lines.append(_line(tv["pass"], f"{k} task {t}", v["kind"], tv["new"], tv["old"], tv["why"]))
                if not tv["pass"]:
                    failed.append(f"{k} task {t}")
        else:
            lines.append(_line(v["pass"], k, v["kind"], v["new"], v["old"], v["why"]))
            if not v["pass"]:
                failed.append(k)
    lines.append("ACCEPT PASS" if res["pass"] else "ACCEPT FAIL: " + ", ".join(failed))
    return lines


# ------------------------------------------------------------------------------------------------ commands
def _sim_setup(config) -> None:
    """Pro2.0's simulator and service source first on the path (a pod's venv-pro2dev has them from its .pth files, the
    Mac's venv-libero has its own LIBERO there), and LIBERO's config folder when one is named. Before LIBERO loads."""
    if "libero" in sys.modules:
        raise RuntimeError("LIBERO is already imported")
    for p in (str(PAYLOAD / "service-source"), str(PAYLOAD / "simulator-source")):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    if config:
        os.environ["LIBERO_CONFIG_PATH"] = str(config)


def cmd_plain(a) -> int:
    if not sys.flags.isolated:
        print("REFUSE: run with python -I", file=sys.stderr)
        return 2
    from rowant_lab.pro2 import long_lib as LL

    cfg = Path(a.config)
    if not (cfg / "config.yaml").is_file() or (cfg / "config.yaml").read_text() != LL.libero_config_text():
        print(f"REFUSE: {cfg / 'config.yaml'} is not long_lib's config (the simulator's own task files)", file=sys.stderr)
        return 2
    _sim_setup(cfg)
    import libero.libero as LIB

    if not Path(LIB.__file__).resolve().is_relative_to(SIM.resolve()) or Path(LIB.config_file) != cfg / "config.yaml":
        print(f"REFUSE: LIBERO is {LIB.__file__} with {LIB.config_file}", file=sys.stderr)
        return 2
    from libero.libero import benchmark

    rows = []
    for suite in a.suites:
        bench = benchmark.get_benchmark_dict()[suite]()  # task order 0, as the service's adapter builds it
        for t in SG.TEST_TASKS[suite]:
            task = bench.get_task(t)
            bddl = Path(bench.get_task_bddl_file_path(t))
            init = SIM / "init_files" / task.problem_folder / task.init_states_file
            if not bddl.is_file() or not SG.is_plain_file(suite, bddl) or not init.is_file():
                print(f"REFUSE: {suite} task {t}: the API's base task file or its start states are missing", file=sys.stderr)
                return 2
            rows.append({"suite": suite, "task": int(t), "bddl": str(bddl), "name": str(task.name),
                         "max_steps": MAX_STEPS[suite]})
    a.out.mkdir(parents=True, exist_ok=True)
    doc = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "what": "the plain versions of the benchmark's test tasks: the simulator's own base task files, through "
                   "LIBERO's API (get_task_bddl_file_path); BASE only, never a perturbed case (split_guard)",
           "libero": {"package": str(Path(LIB.__file__).resolve()), "config_dir": str(cfg),
                      "config": LL.libero_config_text()},
           "tasks": rows}
    (a.out / "plain-tasks.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"plain_tasks": [[r["suite"], r["task"], r["name"]] for r in rows]}, indent=1))
    return 0


def load_plain(path, suites=None) -> list:
    rows = json.loads(Path(path).read_text())["tasks"]
    return [r for r in rows if suites is None or r["suite"] in suites]


def cmd_plan(a) -> int:
    cells = load_cells(a.cells)
    eps = json.loads(Path(a.episodes).read_text())
    chains = chains_from(a.rows)
    items, info = [], {}
    if str(a.old_plan) != "none":
        old = CF.load_plan(Path(a.old_plan))
        if any(it["suite"] not in CF.SUITES for it in old):
            raise SystemExit("the old plan holds a suite that is not one of the first net's")
        items += old
        info["old_plan"] = {"path": str(Path(a.old_plan) / "frames-plan.json.gz"), "items": len(old),
                            "sha256": sha256_file(Path(a.old_plan) / "frames-plan.json.gz")}
    new = long_items(eps, cells, chains, a.sweeps, a.sweep_steps)
    info["episodes"] = {"path": str(a.episodes), "episodes": len(eps), "sha256": sha256_file(a.episodes)}
    info["rows"] = [str(d) for d in a.rows or ()]
    info["chains_used"] = sum(1 for it in new if it["chain"])
    plain = []
    if str(a.plain) != "none":
        plain = load_plain(a.plain, set(a.plain_suites))
        new += plain_items(plain, cells, a.plain_sweeps, a.plain_sweeps_other, a.plain_sweeps_val, a.sweep_steps)
        info["plain"] = {"path": str(a.plain), "sha256": sha256_file(a.plain),
                         "tasks": [[r["suite"], r["task"], r["name"]] for r in plain]}
    items += new
    check_plan(items)
    a.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out / "frames-plan.json.gz", "wt", compresslevel=5) as f:
        json.dump(items, f)
    info.update(summary(items), time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                plan_sha256=sha256_file(a.out / "frames-plan.json.gz"), cells=str(a.cells),
                planner_sha256=sha256_file(Path(__file__)))
    (a.out / "plan-info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: info[k] for k in ("items", "frames_est", "frames_by", "chains_used", "plan_sha256")}, indent=1))
    for r in info["plain_scenes"]:
        print("plain scene:", r["suite"], r["task"], "->", r["key"], r["scene"], r["states"], r["frames"], "frames")
    return 0


def camera_episodes(eps: list, plain: list, cells: dict) -> dict:
    """``{cell: [episode, ...]}`` to read each cell's camera on: the BASE case of its seen training tasks (from the
    episode list) and the plain version of its test task, at ``CAMERA_STATE``."""
    out: dict = {}
    seen = set()
    for e in sorted(eps, key=lambda e: int(e["task"])):
        if e["suite"] != SUITE or e["case"] != "BASE" or CF.split_of(int(e["task"])) != "train" or e["task"] in seen:
            continue
        seen.add(e["task"])
        out.setdefault(key_for(SUITE, e["task"], cells), []).append(
            {"suite": SUITE, "task": int(e["task"]), "case": "BASE", "state": CAMERA_STATE, "seed": CAMERA_SEED,
             "bddl": e["bddl"], "max_steps": int(e["max_steps"])})
    for p in sorted(plain, key=lambda r: int(r["task"])):
        if p["suite"] != SUITE:
            continue
        out.setdefault(key_for(SUITE, p["task"], cells), []).append(
            {"suite": SUITE, "task": int(p["task"]), "case": "BASE", "state": CAMERA_STATE, "seed": CAMERA_SEED,
             "bddl": p["bddl"], "max_steps": int(p.get("max_steps") or MAX_STEPS[SUITE]), "plain": True})
    return out


def old_camera_episodes(old_items: list, per_suite: int) -> dict:
    """``{suite: [episode, ...]}``: BASE resets of the first net's plan (seen tasks), to re-measure a stored camera."""
    out: dict = {}
    for it in sorted(old_items, key=lambda i: (i["suite"], i["task"], i["state"])):
        if it["case"] != "BASE" or it["chain"] or it["split"] != "train" or it["state"] != CAMERA_STATE:
            continue
        if len(out.setdefault(it["suite"], [])) < per_suite:
            out[it["suite"]].append({k: it[k] for k in ("suite", "task", "case", "state", "seed", "bddl", "max_steps")})
    return out


def cmd_cameras(a) -> int:
    if not sys.flags.isolated:
        print("REFUSE: run with python -I", file=sys.stderr)
        return 2
    cells = load_cells(a.cells)
    todo = camera_episodes(json.loads(Path(a.episodes).read_text()),
                           load_plain(a.plain) if str(a.plain) != "none" else [], cells)
    check = old_camera_episodes(CF.load_plan(Path(a.check_old)), a.check_tasks) if a.check_old else {}
    SG.check([e for v in list(todo.values()) + list(check.values()) for e in v], "camera measurement")
    _sim_setup(a.config)
    from rowant_lab.libero_env import m2_rpaths as MP

    MP.pin_numba_cache()
    import mujoco
    import numpy as np

    from rowant_lab.libero_env import m2_camera as CAM
    from rowant_lab.pro2 import colour_line as CL
    from rowant_lab.pro2 import dev_episode as DE

    def read(ep: dict) -> dict:
        ad = DE.adapter_class()()
        try:
            ad.create(DE.env_spec(ep, 16))
            s = ad._env.sim
            cid = s.model.camera_name2id("agentview")
            T = np.eye(4)
            T[:3, :3] = np.asarray(s.data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
            T[:3, 3] = np.asarray(s.data.cam_xpos[cid], dtype=np.float64)
            return {"fovy_deg": float(s.model.cam_fovy[cid]), "T": T @ CAM.CV_FROM_GL}
        finally:
            ad.close()

    stored = json.loads(CL.CAMERAS_FILE.read_text())["cameras"]
    rc = 0
    for suite, eps in sorted(check.items()):  # the method against the stored constants of the three suites
        one = one_camera([read(e) for e in eps])
        gap = pose_gap(one["camera"]["T_world_from_camera"], stored[suite]["T_world_from_camera"])
        same = one["camera"] == stored[suite]
        print(json.dumps({"check": suite, "tasks": [e["task"] for e in eps], "gap_m_deg": gap, "identical": same,
                          "fovy_deg": one["camera"]["fovy_deg"]}), flush=True)
        if gap[0] > CAMERA_TOL["pos_m"] or gap[1] > CAMERA_TOL["rot_deg"] or one["camera"]["fovy_deg"] != stored[suite]["fovy_deg"]:
            rc = 4  # this reading is not how the stored cameras were measured
    cams, measured = {}, {}
    for cell, eps in sorted(todo.items()):
        one = one_camera([read(e) for e in eps])
        cams[cell] = one["camera"]
        measured[cell] = {"training_tasks": [e["task"] for e in eps if not e.get("plain")],
                          "plain_test_tasks": [e["task"] for e in eps if e.get("plain")],
                          "state": CAMERA_STATE, "seed": CAMERA_SEED, "spread": one["spread"], "mujoco": mujoco.__version__}
        near = {s: [round(x, 9) for x in pose_gap(cams[cell]["T_world_from_camera"], c["T_world_from_camera"])]
                for s, c in stored.items()}
        print(json.dumps({"cell": cell, **measured[cell], "fovy_deg": cams[cell]["fovy_deg"],
                          "pos": cams[cell]["T_world_from_camera"][0][3:] + cams[cell]["T_world_from_camera"][1][3:]
                          + cams[cell]["T_world_from_camera"][2][3:], "gap_to_stored_m_deg": near}), flush=True)
    doc = cameras_doc(cams, measured, time.strftime("%Y-%m-%d", time.gmtime()))
    if rc == 0 and a.write:
        tmp = Path(a.write).with_name(Path(a.write).name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        os.replace(tmp, a.write)
        print("wrote", a.write)
    return rc


def cmd_check_cameras(a) -> int:
    import numpy as np

    from rowant_lab.pro2 import colour_line as CL

    stored = CL.stored_cameras()
    rows: dict = {}
    for split in ("train", "val"):
        root = a.frames / split
        for f in sorted(root.glob("*/*/*/r*_0000.npz")):
            key, _task, case = f.relative_to(root).parts[:3]
            if case not in ("BASE", "PLAIN"):
                continue
            z = np.load(f)
            T, K = z["T"], z["K"]
            r = rows.setdefault(key, {"frames": 0, "pos_m": 0.0, "rot_deg": 0.0, "fovy_deg": 0.0})
            r["frames"] += 1
            if key not in stored:
                r["missing"] = True
                continue
            gap = pose_gap(T, stored[key]["T_world_from_camera"])
            r["pos_m"], r["rot_deg"] = max(r["pos_m"], gap[0]), max(r["rot_deg"], gap[1])
            # the image's height from the principal point (the camera tool puts it at the centre)
            r["fovy_deg"] = max(r["fovy_deg"], abs(fovy_of(K, 2.0 * float(K[1][2])) - float(stored[key]["fovy_deg"])))
    bad = sorted(k for k, r in rows.items() if r.get("missing") or any(r[x] > FRAME_TOL[x] for x in FRAME_TOL))
    res = {"pass": bool(rows) and not bad, "keys": rows, "bad": bad, "tol": FRAME_TOL}
    if a.out:
        a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 0 if res["pass"] else 1


def cmd_frames_report(a) -> int:
    items = CF.load_plan(Path(a.plan))
    rows = []
    for f in sorted(Path(a.run).glob("meta-*.jsonl")):
        rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    on_disk: Counter = Counter()
    for split in ("train", "val"):
        root = Path(a.run) / "frames" / split
        if root.is_dir():
            for d in sorted(p for p in root.iterdir() if p.is_dir()):
                on_disk[f"{d.name}/{split}"] = sum(1 for _ in d.rglob("*.npz"))
    res = frames_report(items, rows, on_disk)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("items", "finished", "failed", "failed_by", "frames_total")}))
    return 0


def cmd_accept(a) -> int:
    new = json.loads(Path(a.new).read_text())
    old = json.loads(Path(a.old).read_text()) if a.old else {}
    res = accept(new["by_key"], old.get("by_key"), expect=a.expect or None)
    res.update(new_net=new.get("net_sha256"), old_net=old.get("net_sha256"), val_frames=new.get("val_frames"))
    if a.out:
        tmp = Path(a.out).with_name(Path(a.out).name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1) + "\n")
        os.replace(tmp, a.out)
    print("\n".join(accept_lines(res)))
    return 0 if res["pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plain")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--config", default=str(PLAIN_CONFIG), help="LIBERO's config folder naming the simulator's own task files")
    p.add_argument("--suites", nargs="+", default=sorted(SG.TEST_TASKS))
    p = sub.add_parser("cameras")
    p.add_argument("--episodes", required=True, help="an episode list of dev_run.py plan --suites libero_10")
    p.add_argument("--plain", default=str(OUT / "plain-tasks.json"), help="plain-tasks.json, or none")
    p.add_argument("--cells", default=str(CELLS_FILE))
    p.add_argument("--config", default="", help="LIBERO's config folder (the Mac: runs/pro2/mac-cfg; a pod has its own)")
    p.add_argument("--check-old", default="", help="a first-net plan folder: re-measure the three suites' stored cameras")
    p.add_argument("--check-tasks", type=int, default=2, help="tasks per suite for --check-old")
    p.add_argument("--write", default="", help="write the cameras to this file (rowant_lab/pro2/colour_cameras_long.json)")
    p = sub.add_parser("plan")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--episodes", required=True, help="an episode list of dev_run.py plan --suites libero_10")
    p.add_argument("--old-plan", default=str(OLD_PLAN), help="the first net's plan folder (its items are kept as they are), or none")
    p.add_argument("--plain", default=str(OUT / "plain-tasks.json"), help="plain-tasks.json (the plain test-task scenes), or none")
    p.add_argument("--plain-suites", nargs="+", default=sorted(SG.TEST_TASKS), help="whose plain test tasks to include")
    p.add_argument("--cells", default=str(CELLS_FILE))
    p.add_argument("--rows", nargs="*", default=[], help="run folders whose stored chains replace sweeps (w*/rows.jsonl)")
    p.add_argument("--sweeps", type=int, default=SWEEPS["seen"])
    p.add_argument("--plain-sweeps", type=int, default=SWEEPS["plain_long"])
    p.add_argument("--plain-sweeps-other", type=int, default=SWEEPS["plain_other"])
    p.add_argument("--plain-sweeps-val", type=int, default=SWEEPS["plain_val"])
    p.add_argument("--sweep-steps", type=int, default=SWEEPS["steps"])
    p = sub.add_parser("frames-report")
    p.add_argument("--plan", required=True, help="the folder holding frames-plan.json.gz")
    p.add_argument("--run", required=True, help="the frame run's folder (meta-*.jsonl, frames/)")
    p.add_argument("--out", default="")
    p = sub.add_parser("check-cameras")
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p = sub.add_parser("accept")
    p.add_argument("--new", required=True, help="the candidate net's colour_score.py --by-key file")
    p.add_argument("--old", default="", help="the current net's, on the same frames")
    p.add_argument("--out", default="")
    p.add_argument("--expect", nargs="*", default=[], help="the keys that must be scored (default: three suites, three tables)")
    a = ap.parse_args(argv)
    return {"plain": cmd_plain, "cameras": cmd_cameras, "plan": cmd_plan, "frames-report": cmd_frames_report,
            "check-cameras": cmd_check_cameras, "accept": cmd_accept}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
