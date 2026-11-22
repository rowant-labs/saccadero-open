#!/usr/bin/env python3
"""Score rows against a plan: Overall four ways, by type, by suite and by task.

    python scripts/score.py --plan PLAN.json --rows RUN_DIR [RUN_DIR ...] [--out SCORE.json]

``PLAN.json`` is a list of planned episodes (suite, task, case, case_kind, state, seed); ``RUN_DIR`` folders hold
``w*/rows.jsonl`` as the runner writes them. With the rows and plan we publish for the two scored runs this prints
the numbers of ``docs/RESULTS.md``. The four readings of Overall are explained there and in
``rowant_lab/pro2/scored.py``; the first printed line leads with the public board's own weighting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rowant_lab.pro2 import scored  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--rows", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    plan = json.loads(a.plan.read_text())
    plan = plan["episodes"] if isinstance(plan, dict) else plan
    last, unplayable, events = {}, set(), 0
    for run in a.rows:
        r, u, e = scored.load_rows([run.name], eval_root=run.parent)
        last.update(r)
        unplayable |= u
        events += e
    res = scored.score(plan, last, unplayable - set(last))
    res["event_rows"] = events
    rd, f2 = res["readings"], (lambda x: "   n/a" if x is None else f"{x:6.2f}")
    print(f"Overall by the board's own weighting {f2(rd['board_weighting'])}")
    print(f"        by the formula registered before the runs {f2(rd['registered_formula'])}")
    print(f"        per episode {f2(rd['per_episode'])}  ({rd['per_episode_counts'][0]} of {rd['per_episode_counts'][1]})")
    print(f"        registered formula, refused episodes counted as failures {f2(rd['refused_counted_as_failures'])}")
    print(f"static {f2(res['static'])}  dynamic {f2(res['dynamic'])}  (registered formula; {res['static_types']} static and "
          f"{res['dynamic_types']} dynamic types)")
    if res["base"]:
        print(f"plain tasks (not in Overall): {f2(res['base']['pooled'])} of {res['base']['episodes']} episodes")
    print("episodes: " + json.dumps(res["episodes"]) + f"  event rows {events}")
    for c, t in res["by_type"].items():
        print(f"  {c:5} {t['kind']:7} {t['score']:5.1f}   " + "  ".join(f"{s[7:]} {v[0]}/{v[1]}" for s, v in t["by_suite"].items()))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1) + "\n")
        print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
