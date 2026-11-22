"""A development run's success by the formula registered before the scored runs (EVALUATOR side; not a placement).

That formula is the plain mean over the 42 perturbation types (22 static S01-S22, 20 dynamic D01-D21), each type the
mean over its suites, base tasks left out. The public board's own averages follow a slightly different weighting
(docs/RESULTS.md sets it beside this one). The development runs this tool was written for play three of the four
suites and never S06, so three views are printed: as run; S06 as zero; S06 and the fourth suite as zero (the
pessimistic four-suite view).

    python3 -m rowant_lab.pro2.board_average RUN [RUN ...]     # folders under runs/pro2/colour/eval
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.settings import DATA  # noqa: E402

EVAL = DATA / "runs" / "pro2" / "colour" / "eval"  # where the rows of finished runs are kept

SUITES = ("libero_object", "libero_spatial", "libero_goal")
N_STATIC, N_DYNAMIC = 22, 20


def rows(run: str) -> dict:
    last = {}
    for f in sorted((EVAL / run).glob("w*/rows.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if "success" in r:
                last[(r["suite"], r["task"], r["case"], r["state"], r["seed"])] = r
    return last


def rates(rs: dict) -> dict:
    per = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for k, r in rs.items():
        if k[0] in SUITES:
            per[k[2]][k[0]][0] += bool(r["success"])
            per[k[2]][k[0]][1] += 1
    out = {}
    for c, by in per.items():
        v = [by[s][0] / by[s][1] for s in SUITES if by[s][1]]
        if len(v) == len(SUITES):
            out[c] = 100.0 * sum(v) / len(v)
    return out


def views(rate: dict) -> dict:
    st = {c: v for c, v in rate.items() if c.startswith("S")}
    dy = {c: v for c, v in rate.items() if c.startswith("D")}
    s_sum, d_sum = sum(st.values()), sum(dy.values())
    four = len(SUITES) / 4.0
    return {"as_run": (s_sum / max(1, len(st)), d_sum / max(1, len(dy)), (s_sum + d_sum) / max(1, len(st) + len(dy))),
            "s06_zero": (s_sum / N_STATIC, d_sum / max(1, len(dy)), (s_sum + d_sum) / (N_STATIC + N_DYNAMIC)),
            "four_suite": (s_sum / N_STATIC * four, d_sum / N_DYNAMIC * four,
                           (s_sum + d_sum) / (N_STATIC + N_DYNAMIC) * four)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--types", default="", help="comma-separated types to print")
    a = ap.parse_args(argv)
    for run in a.runs:
        rate = rates(rows(run))
        v = views(rate)
        print(f"{run}: " + "; ".join(f"{k} static {s:.1f} dynamic {d:.1f} overall {o:.1f}" for k, (s, d, o) in v.items()))
        if a.types:
            print("   " + " ".join(f"{t}:{rate.get(t, float('nan')):.0f}" for t in a.types.split(",")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
