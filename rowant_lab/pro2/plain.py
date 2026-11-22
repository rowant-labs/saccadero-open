"""Plain (unperturbed) episodes of any base task, test tasks included (decided 2026-11-15: the plain versions of the
eight test tasks may be developed on; every perturbed test-split episode stays sealed).

EVALUATOR side. The task files are the simulator's own base files (byte for byte the original LIBERO release's), reached
through LIBERO's API; the payload's test-split files and its manifest are not opened. Run with venv-libero::

    <worker python> -I -B rowant_lab/pro2/plain.py plan --suite libero_goal --tasks 3,6 --states 8 --out DIR

``DIR/episodes.json`` then holds one BASE episode per task and start state, in the runner's format (``case`` "BASE",
``split`` "plain"). Run folders need the LIBERO config ``long-lib/build/libero-config`` (``mac_worker.py --cfg``).
"""
import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.pro2.machine import BENCHMARK  # noqa: E402
from rowant_lab.settings import DATA  # noqa: E402

PAYLOAD = BENCHMARK
CFG = str(DATA / "runs" / "pro2" / "long-lib" / "build" / "libero-config")
MAX_STEPS = {"libero_object": 280, "libero_spatial": 220, "libero_goal": 300, "libero_10": 520}  # the suites' step caps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan",))
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--states", default="8")
    ap.add_argument("--seed", type=int, default=150)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path[0:0] = [str(PAYLOAD / "simulator-source"), str(PAYLOAD / "service-source")]
    os.environ["LIBERO_CONFIG_PATH"] = CFG
    from libero.libero import benchmark

    bench = benchmark.get_benchmark_dict()[a.suite]()
    eps = []
    for t in (int(x) for x in a.tasks.split(",")):
        bddl = Path(bench.get_task_bddl_file_path(t))
        if not bddl.is_file():
            raise SystemExit(f"task {t}: no base task file at {bddl}")
        for s in (int(x) for x in a.states.split(",")):
            eps.append({"suite": a.suite, "task": t, "case": "BASE", "case_kind": "base", "state": s, "seed": a.seed,
                        "bddl": str(bddl), "max_steps": MAX_STEPS[a.suite], "split": "plain"})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes.json").write_text(json.dumps(eps, indent=0))
    print(json.dumps({"episodes": len(eps), "suite": a.suite, "tasks": a.tasks, "states": a.states}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
