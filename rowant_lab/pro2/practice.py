"""Practice variants (development only; EVALUATOR side): our own instruction on a TRAINING task's scene.

The board scores eight test tasks. Their public names say which skills they need, and
some of those skills are asked by no training task of the same suite (Goal: opening a drawer and then putting an object
inside it; putting a small box into a container object). A practice episode builds a training task's scene as the
benchmark does, gives the robot OUR instruction instead of the task's, and judges it with the simulator's own
predicates over its object states (``dev_episode._practice_ok``). No test-split file is opened and no test task is
built. The row's ``success`` stays the benchmark's (for the scene's own task); ``practice.success`` is ours.

    python3 -m rowant_lab.pro2.practice dev [--suite libero_goal] [--state 8] [--tasks 0,1,2,4,5,8] [--cases BASE]
    python3 -m rowant_lab.pro2.practice report [--runs NAME,NAME]

``dev`` writes one run folder per practice (``runs/pro2/practice/<prefix><state>-<name>``); the Mac runs them with
``mac_worker.py``, a pod with the usual queue.

The long suite (2026-11-16, night). Its two test tasks never run perturbed before the scored run, and one of them
(both moka pots on the stove) has its objects in exactly one TRAINING scene: long task 2, "turn on the stove and put
the moka pot on it", which the robot cannot do (no skill for turning the stove on). ``pot-stove`` runs that training
task's own episodes, every perturbation case, with OUR instruction "put the moka pot on the stove" and the check "the
moka pot on the stove's cook region": our own variant of a training task, one pot, not the test task and not its
scene. ``bowl-drawer`` does the same in long task 3's scene (its closing step is what fails there). Never the study
table (the book task's only scene is a test scene), never a test task's scene.

    python3 -m rowant_lab.pro2.practice long --names pot-stove --state 8 --seed 150 --cases all \
        [--colour NET] [--bundle] [--root devruns] [--prefix lp] [--tag v6]
    python3 -m rowant_lab.pro2.practice report --runs lp8c-pot-stove,lp8c-pot-stove-v6   # the second against the first

``long`` writes ``<root>/<prefix><state>[c]-<name>[-<tag>]`` with the unified configuration (``long_dev``; ``--bundle``:
the candidate's options on). ``report`` reads a run's rows under the practice root, else under ``colour/eval``.
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

from rowant_lab.settings import DATA  # noqa: E402

ROOT = DATA / "runs" / "pro2" / "practice"
EVAL = DATA / "runs" / "pro2" / "colour" / "eval"  # where the rows of finished runs are kept
ADAPTER = Path(__file__).with_name("colour_adapter_params.json")  # the colour line as decided
TEST_TASKS = {"libero_object": (1, 8), "libero_spatial": (0, 8), "libero_goal": (3, 6), "libero_10": (5, 8)}
JUDGE = (7, 9)  # our own held-back training tasks: not practised on either
# name -> (suite, our instruction, the goal over the simulator's object states, the instruction's other wordings)
PRACTICES = {
    "open-put": ("libero_goal", "open the top drawer and put the bowl inside",
                 [["open", "wooden_cabinet_1_top_region"], ["in", "akita_black_bowl_1", "wooden_cabinet_1_top_region"]]),
    "box-bowl": ("libero_goal", "put the cream cheese in the bowl", [["on", "cream_cheese_1", "akita_black_bowl_1"]]),
}
SOURCE = {"libero_goal": EVAL / "fair2-depth" / "episodes.json"}  # the suite's planned training episodes
# the long suite: name -> (the training task whose scene it runs in, our instruction, our check). The planned list is
# the runner's own (all eight training tasks, every case but the held one, start state 8)
LONG = {
    "pot-stove": (2, "put the moka pot on the stove", [["on", "moka_pot_1", "flat_stove_1_cook_region"]]),
    "bowl-drawer": (3, "put the black bowl in the bottom drawer of the cabinet",
                    [["in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"]]),
}
LONG_SOURCE = EVAL / "lc8-uni" / "episodes.json"
DEVRUNS = DATA / "runs" / "pro2" / "devruns"
HELD_CASES = ("S06",)  # never planned, never run


def run_params(line: str = "depth") -> dict:
    """The line as decided, with the step sequencer. Only the colour line is part of this release."""
    if line == "depth":
        raise ValueError("the depth line (it reads the simulator's depth) is not part of this release: use --line colour")
    p = json.loads(ADAPTER.read_text())
    p["sequence"] = True
    return p


def cmd_dev(a) -> int:
    tasks = {int(t) for t in a.tasks.split(",")}
    cases = set(a.cases.split(","))
    for name in a.names.split(","):
        suite, inst, goal = PRACTICES[name]
        if tasks & set(TEST_TASKS[suite]) or tasks & set(JUDGE):
            raise SystemExit("practice runs on seen training tasks only")
        eps = [dict(e, state=int(a.state), seed=int(a.seed), practice={"name": name, "instruction": inst, "goal": goal})
               for e in json.loads(SOURCE[suite].read_text())
               if e["suite"] == suite and e["task"] in tasks and e["case"] in cases]
        d = ROOT / f"{a.prefix}{a.state}-{name}"
        d.mkdir(parents=True, exist_ok=True)
        b = json.dumps(eps).encode()
        (d / "episodes.json").write_bytes(b)
        text = json.dumps(run_params(a.line))
        (d / "params.json").write_text(text)
        print(json.dumps({"run": d.name, "episodes": len(eps), "episodes_sha256": hashlib.sha256(b).hexdigest(),
                          "params_sha256": hashlib.sha256(text.encode()).hexdigest()}))
    return 0


def long_episodes(name: str, state: int, seed: int, cases, source=None) -> list:
    """The practice ``name``'s episodes: its training task's planned episodes (``cases``: a set, or None for every
    case), at ``state`` and ``seed``, each carrying our instruction and check. Checked by ``split_guard``; a test
    task, a held-back training task or a held case is refused."""
    from rowant_lab.pro2 import split_guard

    task, inst, goal = LONG[name]
    if task in TEST_TASKS["libero_10"] or task in JUDGE:
        raise SystemExit("practice runs on seen training tasks only")
    eps = [dict(e, state=int(state), seed=int(seed), practice={"name": name, "instruction": inst, "goal": goal})
           for e in json.loads(Path(source or LONG_SOURCE).read_text())
           if e["suite"] == "libero_10" and int(e["task"]) == task and (cases is None or e["case"] in cases)]
    if any(str(e["case"]).startswith(HELD_CASES) for e in eps):
        raise SystemExit("a held case is never run")
    split_guard.check(eps, "practice run")  # never a perturbed episode of a benchmark test task
    return eps


def cmd_long(a) -> int:
    from rowant_lab.pro2 import long_dev as LD

    cases = None if a.cases in ("all", "") else set(a.cases.split(","))
    root = DEVRUNS if a.root == "devruns" else (ROOT if a.root in ("", "practice") else Path(a.root))
    for name in a.names.split(","):
        eps = long_episodes(name, int(a.state), int(a.seed), cases)
        if not eps:
            raise SystemExit(f"{name}: no episode selected")
        run = f"{a.prefix}{a.state}{'c' if a.colour else ''}-{name}" + (f"-{a.tag}" if a.tag else "")
        if (EVAL / run).exists():
            raise SystemExit(f"{run}: rows of that name exist under colour/eval; take another --prefix or --tag")
        p = LD.colour_long_params(a.colour) if a.colour else LD.long_params()
        if a.bundle:
            p = LD.bundle(p)
        if a.agent or a.set:  # options on top (``subset``'s forms)
            from rowant_lab.pro2 import subset as SB

            p.update(SB._pairs(a.set))
            if a.agent:
                m1 = dict(p.get("m1") or {})
                m1["agent"] = {**(m1.get("agent") or {}), **SB._pairs(a.agent)}
                p["m1"] = m1
        d = root / run
        d.mkdir(parents=True, exist_ok=True)
        b = json.dumps(eps).encode()
        (d / "episodes.json").write_bytes(b)
        text = json.dumps(p)
        (d / "params.json").write_text(text)
        print(json.dumps({"run": run, "dir": str(d), "episodes": len(eps), "cases": len({e["case"] for e in eps}),
                          "episodes_sha256": hashlib.sha256(b).hexdigest(),
                          "params_sha256": hashlib.sha256(text.encode()).hexdigest()}))
    return 0


def practice_rows(run: str) -> dict:
    """A run's rows that carry a practice verdict, by (task, case, state, seed): under the practice root, the
    development runs, else ``colour/eval`` (a pod's rows)."""
    out = {}
    for root in (ROOT, DEVRUNS, EVAL):
        for f in sorted((root / run).glob("w*/rows.jsonl")):
            for x in f.read_text().splitlines():
                r = json.loads(x) if x.strip() else {}
                if "practice" in r and "success" in r:
                    out[(r["task"], r["case"], r["state"], r["seed"])] = r
        if out:
            break
    return out


def cmd_report(a) -> int:
    runs = a.runs.split(",")
    tabs = {run: practice_rows(run) for run in runs}
    for run in runs:
        rows = list(tabs[run].values())
        ok = sum(bool(r["practice"]["success"]) for r in rows)
        print(f"{run}: practice goal met in {ok} of {len(rows)}")
        if a.brief:
            continue
        for r in sorted(rows, key=lambda r: (r["task"], r["case"])):
            seq = [(s.get("clause"), s.get("skipped") or (s.get("summary") or {}).get("finish_reason"))
                   for s in r.get("sequence") or []]
            att = collections.Counter((x.get("step"), x.get("outcome")) for x in r.get("attempts") or [])
            print(f"  task {r['task']} {r['case']}: {'OK' if r['practice']['success'] else '--'} at {r['practice']['at']}"
                  f" | steps {r.get('steps')} | {seq or r.get('finish_reason')} | attempts {dict(att)}"
                  f" | grounding {r.get('grounding')}" + (f" | error {r['agent_error']}" if r.get("agent_error") else ""))
    ref = tabs[runs[0]]
    for run in runs[1:]:  # paired with the first run, by episode
        new = tabs[run]
        both = sorted(set(ref) & set(new))
        won = [k for k in both if new[k]["practice"]["success"] and not ref[k]["practice"]["success"]]
        lost = [k for k in both if ref[k]["practice"]["success"] and not new[k]["practice"]["success"]]
        print(f"{run} against {runs[0]}: {len(both)} episodes in both, "
              f"{sum(bool(ref[k]['practice']['success']) for k in both)} -> "
              f"{sum(bool(new[k]['practice']['success']) for k in both)}; won {len(won)} {[k[1] for k in won]}, "
              f"lost {len(lost)} {[k[1] for k in lost]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("dev", "long", "report"))
    ap.add_argument("--names", default="open-put,box-bowl")
    ap.add_argument("--state", default="8")
    ap.add_argument("--seed", default="150")
    ap.add_argument("--tasks", default="0,1,2,4,5,8")
    ap.add_argument("--cases", default="BASE")
    ap.add_argument("--line", default="depth")
    ap.add_argument("--prefix", default="pg")
    ap.add_argument("--runs", default="pg8-open-put,pg8-box-bowl")
    ap.add_argument("--colour", default="", help="long: the colour line with this depth net (its path on the pod)")
    ap.add_argument("--bundle", action="store_true", help="long: the candidate's options on (long_dev.bundle)")
    ap.add_argument("--root", default="devruns", help="long: where the run folder goes (devruns, practice, or a path)")
    ap.add_argument("--tag", default="", help="long: a suffix of the run's name (e.g. v6)")
    ap.add_argument("--agent", default="", help="long: agent options on top (subset's form: a=1,b=true)")
    ap.add_argument("--set", default="", help="long: top-level options on top")
    ap.add_argument("--brief", action="store_true", help="report: the counts only")
    a = ap.parse_args(argv)
    if a.cmd == "long" and a.names == "open-put,box-bowl":
        a.names = "pot-stove"
    if a.cmd == "long" and a.prefix == "pg":
        a.prefix = "lp"
    return {"dev": cmd_dev, "long": cmd_long, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
