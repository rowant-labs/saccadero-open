#!/usr/bin/env python3
"""Compare a run with another run, or with the outcomes we recorded.

    python scripts/compare.py RUN_DIR OTHER_RUN_DIR      # two runs of the same episodes, episode by episode
    python scripts/compare.py RUN_DIR --reference v11    # against docs/records/reference-outcomes.json

For two runs it counts the episodes with the same outcome, the same number of steps and the same commands (every
action sent to the robot, as the rows record them), and names the episodes whose outcome differs. Two runs of one
configuration on one machine differ in commands in about one episode in a hundred; two machines with different
processors or GPU types differ in many more, usually without a different outcome.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def key(r: dict) -> str:
    return f"{r['suite']}|{r['task']}|{r['case']}|{r['state']}|{r['seed']}"


def rows(run: Path) -> dict:
    """The last outcome row of each episode of a run folder."""
    last = {}
    for f in sorted(run.glob("w*/rows.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if "success" in r:
                last[key(r)] = r
    return last


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("other", type=Path, nargs="?")
    ap.add_argument("--reference", choices=("v9", "v11"), help="compare with our recorded outcomes for this configuration")
    a = ap.parse_args(argv)
    A = rows(a.run)
    if not A:
        raise SystemExit(f"no outcome rows under {a.run}")
    out = {"run": str(a.run), "episodes": len(A), "solved": sum(bool(r["success"]) for r in A.values())}
    if a.other:
        B = rows(a.other)
        both = sorted(set(A) & set(B))
        diff = [k for k in both if bool(A[k]["success"]) != bool(B[k]["success"])]
        out.update(other=str(a.other), other_episodes=len(B), other_solved=sum(bool(r["success"]) for r in B.values()),
                   in_both=len(both), same_outcome=len(both) - len(diff),
                   same_steps=sum(A[k].get("steps") == B[k].get("steps") for k in both),
                   same_commands=sum(A[k].get("actions") == B[k].get("actions") for k in both),
                   outcome_differs=[{"episode": k, "run": bool(A[k]["success"]), "other": bool(B[k]["success"])} for k in diff])
    if a.reference:
        ref = json.loads((REPO / "docs" / "records" / "reference-outcomes.json").read_text())
        table = {}
        for part in ("practice", "plain8"):
            table.update({k: v[a.reference] for k, v in ref[part]["episodes"].items() if a.reference in v})
        both = sorted(set(A) & set(table))
        diff = [k for k in both if bool(A[k]["success"]) != bool(table[k][0])]
        out.update(reference=a.reference, in_reference=len(both),
                   reference_solved=sum(bool(table[k][0]) for k in both),
                   run_solved_of_those=sum(bool(A[k]["success"]) for k in both), same_outcome=len(both) - len(diff),
                   same_steps=sum(A[k].get("steps") == table[k][1] for k in both),
                   outcome_differs=[{"episode": k, "run": bool(A[k]["success"]), "recorded": bool(table[k][0])} for k in diff])
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
