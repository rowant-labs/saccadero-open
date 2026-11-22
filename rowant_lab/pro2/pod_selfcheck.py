"""Does this machine compute the same on every processor? Run it on a pod before any evaluation::

    <server python> -P rowant_lab/pro2/pod_selfcheck.py [--repeat 2] [--jobs 8]

Any Python 3.10 or later with numpy will do (a new pod's own ``python3`` after ``pip install numpy``, before the long
upload): the digests are compared within the machine, never with another machine's.

2026-11-18: two pods were one rented machine whose physical core 23 (processors 23 and 87 of
128) returned a wrong quotient in the third of every four results of a vector division. Nothing raised an error
except, rarely, an index that could not exist ("index 174879 is out of bounds") and a server that would not start;
every run on those two pods carried wrong numbers. This check pins a fixed battery of numpy operations to each
processor the process may use, in a fresh process each, and compares the digests. Exit 0: every processor agrees.
Exit 1: some differ (they are named; do not run on this machine). Exit 2: the check itself could not run.

Where a process cannot be pinned (the Mac) the battery runs unpinned ``--repeat`` times and only those are compared.
Standard library and numpy only; it reads no file of the project.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def battery() -> dict:
    """A digest for each of a fixed set of operations on fixed inputs (64 and 32 bit floats, integers)."""
    import numpy as np

    def h(x) -> str:
        return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:8]

    rng = np.random.default_rng(7)
    out = {}
    for name, dt in (("f64", np.float64), ("f32", np.float32)):
        a = (rng.random(8192) + 0.5).astype(dt)
        b = (rng.random(8192) + 0.5).astype(dt)
        m = rng.random((96, 96)).astype(dt)
        rows = np.linspace(0.0, 1023.99, 8192).astype(dt)
        ops = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b, "div_scalar": a / dt(3), "sqrt": np.sqrt(a),
               "exp": np.exp(a), "log": np.log(a), "sin": np.sin(a), "floor": np.floor(a * 100),
               "sum": np.array([a.sum()]), "cumsum": np.cumsum(a), "matmul": m @ m, "sort": np.sort(a),
               "min": np.minimum(a, b), "less": a < b, "hypot": np.hypot(a, b), "arctan2": np.arctan2(a, b),
               "bins": np.floor(rows / 3)}  # the pose fitter's bin index, where the fault first showed
        out.update({f"{name}.{k}": h(v) for k, v in ops.items()})
    i = rng.integers(1, 1 << 40, 8192)
    j = rng.integers(1, 1 << 20, 8192)
    ops = {"add": i + j, "mul": i * j, "div": i // j, "mod": i % j, "shift": i >> 3, "and": i & j,
           "cast": i.astype(np.float64), "sort": np.sort(i), "less": i < j * 1000}
    out.update({f"int.{k}": h(v) for k, v in ops.items()})
    return out


def one(cpu) -> dict:
    """The battery in a fresh process, pinned to ``cpu`` (None: unpinned)."""
    pin = None
    if cpu is not None:
        def pin():  # noqa: E306  (runs in the child before exec)
            os.sched_setaffinity(0, {cpu})
    safe = ["-P"] if sys.version_info >= (3, 11) else []  # the script's folder off the module path where Python can
    p = subprocess.run([sys.executable, *safe, os.path.abspath(__file__), "--one"], capture_output=True, text=True,
                       preexec_fn=pin, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f"processor {cpu}: battery exit {p.returncode}: {p.stderr.strip()[-200:]}")
    return json.loads(p.stdout)


def off_majority(runs: list) -> dict:
    """{processor: [operations whose digest differs from the digest most runs gave]} for ``[(processor, digests)]``."""
    major = {k: collections.Counter(d[k] for _, d in runs).most_common(1)[0][0] for k in runs[0][1]}
    bad = collections.defaultdict(set)
    for cpu, d in runs:
        for k, v in major.items():
            if d[k] != v:
                bad[cpu].add(k)
    return {c: sorted(v) for c, v in sorted(bad.items(), key=lambda x: str(x[0]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true", help="print the battery's digests and exit (the child)")
    ap.add_argument("--repeat", type=int, default=2, help="fresh processes per processor")
    ap.add_argument("--jobs", type=int, default=8, help="processes at a time")
    a = ap.parse_args(argv)
    if a.one:
        print(json.dumps(battery(), sort_keys=True))
        return 0
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [None]
    todo = [c for c in cpus for _ in range(max(1, a.repeat))]
    try:
        with ThreadPoolExecutor(max(1, a.jobs)) as ex:
            runs = list(zip(todo, ex.map(one, todo)))
    except Exception as exc:  # the check could not run: say so, never pass
        print(json.dumps({"selfcheck": "could not run", "why": f"{type(exc).__name__}: {exc}"[:300]}))
        return 2
    bad = off_majority(runs)
    print(json.dumps({"selfcheck": "ok" if not bad else "PROCESSORS DIFFER", "processors": len(cpus), "pinned": cpus != [None],
                      "runs": len(runs), "operations": len(runs[0][1]), "differ": {str(c): v for c, v in bad.items()}}))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
