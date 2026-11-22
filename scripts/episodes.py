#!/usr/bin/env python3
"""Make an episode list for ``scripts/run.py`` from your own copy of the benchmark.

    python scripts/episodes.py practice --out practice.json [--sample 60]
    <worker python> -I -B scripts/episodes.py plain8 --out plain8.json [--seeds 142 143 144]

``practice``: the 476 training-split episodes our configurations were compared on during development
(``configs/practice-476.json``: every training task that has the case, 41 perturbation types and the plain case, start
state 0, seed 170; the blind type S06 is not among them). ``--sample N`` takes a fixed, mixed N of them. The list
names episodes by suite, task and case; the task files are looked up in your download's manifest.

``plain8``: the eight tasks of the benchmark's test split in their plain, unperturbed form (LIBERO's own base task
files, which ship with the simulator), start state 0, one episode per task and seed. This command asks LIBERO for
the tasks' file names, so it runs in the worker's environment.

No perturbed episode of a test task can be made here (``rowant_lab/pro2/split_guard.py`` refuses it): the test split
is planned only by ``python -m rowant_lab.pro2.scored plan --test-split``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rowant_lab.pro2 import machine, split_guard  # noqa: E402

MAX_STEPS = {"libero_object": 280, "libero_spatial": 220, "libero_goal": 300, "libero_10": 520}  # the suites' step caps
ASSETS = machine.BENCHMARK / "service-source" / "assets"


def practice(keys_file: Path, sample: int | None) -> list:
    keys = json.loads(keys_file.read_text())["episodes"]
    manifest = json.loads((ASSETS / "task_manifest.json").read_text())["tasks"]
    path = {(e["suite"], int(e["base_task_id"]), str(e["case_id"]).strip().upper()): e["bddl_path"] for e in manifest}
    if sample:  # a fixed, mixed sample: the episodes whose ids hash lowest
        ident = lambda k: f"{k['suite']}|{k['task']}|{k['case']}|{k['state']}|{k['seed']}"  # noqa: E731
        keys = sorted(keys, key=lambda k: hashlib.sha256(ident(k).encode()).hexdigest())[:int(sample)]
    eps = []
    for k in keys:
        bddl = ASSETS / path[(k["suite"], int(k["task"]), k["case"])]
        if not bddl.is_file():
            raise SystemExit(f"no task file in your download for {k['suite']} task {k['task']} case {k['case']}: {bddl}")
        eps.append({**k, "bddl": str(bddl), "split": "train"})
    split_guard.check(eps, "practice list")  # never a perturbed episode of a test task
    return eps


def plain8(seeds) -> list:
    from libero.libero import benchmark  # the simulator's own package, from your download

    eps = []
    for suite, tasks in sorted(split_guard.TEST_TASKS.items()):
        bench = benchmark.get_benchmark_dict()[suite]()
        for t in tasks:
            task = bench.get_task(t)
            bddl = Path(split_guard.PLAIN_ROOT) / task.problem_folder / task.bddl_file
            if not bddl.is_file():
                raise SystemExit(f"no base task file at {bddl}")
            for s in seeds:
                eps.append({"suite": suite, "task": int(t), "case": "BASE", "case_kind": "base", "state": 0,
                            "seed": int(s), "bddl": str(bddl), "max_steps": MAX_STEPS[suite], "split": "plain"})
    return eps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("practice", "plain8"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--list", type=Path, default=REPO / "configs" / "practice-476.json")
    ap.add_argument("--sample", type=int, default=None, help="practice: this many episodes (a fixed, mixed sample)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[142, 143, 144], help="plain8 (the scored runs used 142 to 171)")
    a = ap.parse_args(argv)
    eps = practice(a.list, a.sample) if a.cmd == "practice" else plain8(a.seeds)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(eps, indent=0))
    print(json.dumps({"episodes": len(eps), "suites": sorted({e["suite"] for e in eps}),
                      "cases": len({e["case"] for e in eps}), "file": str(a.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
