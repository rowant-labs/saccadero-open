"""The eight test tasks, plain (unperturbed): one run folder for all four suites, with one robot configuration.

EVALUATOR side (Mac). The plans come from ``plain.py plan`` (venv-libero) under ``runs/pro2/eight/plans/<suite>``; this
writes ``runs/pro2/eight/<name>`` with every plain episode and the candidate configuration for a scored run: the depth
line as decided, on the long-suite library tree (which holds the other suites' files byte for byte), with the step
sequencer (``long_dev.long_params``). The colour line's folder takes the adapter's colour block on top.

    python3 -m rowant_lab.pro2.eight dev --name e8-v1 [--tasks libero_goal:3,6] [--states 8] [--colour NET]
        [--set fixture_no_flip=false] [--agent rest_m=0.008] [--server-json '{"support": {...}}']    (as ``subset dev``)
        [--seeds 142-171]    each chosen plain episode once per seed (a scored run's seeds), not the plan's own
    python3 -m rowant_lab.pro2.eight report --name e8-v1 [--rows DIR]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.pro2 import long_dev as LD  # noqa: E402
from rowant_lab.settings import DATA  # noqa: E402

ROOT = DATA / "runs" / "pro2" / "eight"
EVAL = DATA / "runs" / "pro2" / "colour" / "eval"  # where the rows of finished runs are kept
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
TEST_TASKS = {"libero_object": (1, 8), "libero_spatial": (0, 8), "libero_goal": (3, 6), "libero_10": (5, 8)}


def cmd_dev(a) -> int:
    want = {}
    for item in (a.tasks.split(";") if a.tasks else []):
        s, ts = item.split(":")
        want[s] = {int(t) for t in ts.split(",")}
    states = {int(s) for s in a.states.split(",")} if a.states else None
    eps = []
    for s in SUITES:
        if want and s not in want:
            continue
        for e in json.loads((ROOT / "plans" / s / "episodes.json").read_text()):
            if e["case"] != "BASE":
                raise SystemExit("plain episodes only")
            if (not want or e["task"] in want[s]) and (states is None or e["state"] in states):
                eps.append(e)
    if a.seeds:  # a plain start under other seeds (still BASE, still the simulator's own base file)
        seeds = []
        for item in a.seeds.split(","):
            lo, _, hi = item.partition("-")
            seeds += list(range(int(lo), int(hi or lo) + 1))
        eps = [{**e, "seed": sd} for e in eps for sd in seeds]
    from rowant_lab.pro2 import split_guard

    split_guard.check(eps, "plain run")  # BASE from the simulator's own base files only
    d = ROOT / a.name
    d.mkdir(parents=True, exist_ok=True)
    b = json.dumps(eps).encode()
    (d / "episodes.json").write_bytes(b)
    # the depth line as decided, or (--colour NET) the colour adapter with the depth net at NET on the pod; then the
    # options under test, as ``subset dev`` takes them (none given: exactly the parameters above)
    from rowant_lab.pro2 import subset as SB

    text = json.dumps(SB.build_params(SB._pairs(a.agent), SB._pairs(a.set), a.colour, a.server_json))
    (d / "params.json").write_text(text)
    print(json.dumps({"run": d.name, "episodes": len(eps), "episodes_sha256": hashlib.sha256(b).hexdigest(),
                      "params_sha256": hashlib.sha256(text.encode()).hexdigest()}))
    return 0


def cmd_report(a) -> int:
    d = Path(a.rows) if a.rows else EVAL / a.name
    rows = []
    for f in sorted(d.glob("w*/rows.jsonl")):
        rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if "success" in r]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["suite"], r["task"])].append(r)
    tot = 0
    for (s, t), rs in sorted(by.items()):
        ok = sum(bool(r["success"]) for r in rs)
        tot += ok
        why = collections.Counter()
        for r in rs:
            if not r["success"]:
                seq = r.get("sequence")
                why[" > ".join(str(x.get("skipped") or (x.get("summary") or {}).get("finish_reason")) for x in seq)
                    if seq else str(r.get("finish_reason"))] += 1
        steps = sorted(r["steps"] for r in rs if r["success"])
        print(f"{s[7:]:8} task {t}: {ok:2}/{len(rs):2}" + (f"  steps median {steps[len(steps) // 2]}" if steps else "")
              + (f"  failures {dict(why)}" if why else ""))
    print(f"all: {tot}/{len(rows)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("dev", "report"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--states", default="")
    ap.add_argument("--rows", default="")
    ap.add_argument("--colour", default="", help="the colour line with this depth net (its path on the pod)")
    ap.add_argument("--set", default="", help="dev: top-level parameters, k=json,...")
    ap.add_argument("--agent", default="", help="dev: the agent's parameters, k=json,...")
    ap.add_argument("--server-json", default="", help="dev: a JSON object merged into the server's parameters")
    ap.add_argument("--seeds", default="", help="dev: each episode once per seed, e.g. 142-171")
    a = ap.parse_args(argv)
    return {"dev": cmd_dev, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
