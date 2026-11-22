"""Our robot on LIBERO-Pro2.0 training episodes, on a pod (the depth line; ``dev_episode``). Run with venv-pro2dev::

    PY=<worker python>
    $PY -I -B rowant_lab/pro2/dev_run.py plan --out OUT [--suites ...] [--states 0] [--seeds 142]
    $PY -I -B rowant_lab/pro2/dev_run.py run --out OUT --workers N [--params JSON]
    $PY -I -B rowant_lab/pro2/dev_run.py summary --out OUT

* ``plan`` lists every training episode of the chosen suites whose task file is on the pod (no test task and no S06
  case was ever unpacked), for each start state and seed. ``JUDGE`` tasks are marked: they are held back from the
  self-writing loop's diagnosis and used only to check its fixes. They were fixed here before any run.
* ``run`` starts N workers. Each has its own perception server (M2's, Monty venv) and runs every N-th episode,
  skipping episodes it already finished, so a re-run resumes. A watchdog or bridge failure is an EVENT row, never an
  outcome; the episode is retried by the next run. The simulator's own refusal of an episode (``SIM_EVENTS``) is an
  event row too; after ``SIM_TRIES`` attempts the episode is marked ``unplayable``: finished, with no outcome.
* ``summary`` gives success by case, split and suite (bypassed cases apart) and the time per episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.pro2.machine import BENCHMARK  # noqa: E402

PAYLOAD = BENCHMARK
MANIFEST = PAYLOAD / "service-source" / "assets" / "task_manifest.json"
SUITES = ("libero_object", "libero_spatial")
JUDGE = {"libero_object": (7, 9), "libero_spatial": (7, 9), "libero_goal": (7, 9), "libero_10": (7, 9)}
HELD_CASES = ("S06",)
WATCHDOG_S = 900
SIM_EVENTS = ("PerturbationValidationError", "RandomizationError")  # Pro2.0's refusals; RandomizationError: LIBERO's placement at build (world 2, 11-10)
SIM_TRIES = 2
WORKER_TRIES = 3  # a worker slot's process may die this many times in a row before the slot gives up for this run
WORKER_PAUSE_S = 20


def key(ep: dict) -> str:
    return f"{ep['suite']}|{ep['task']}|{ep['case']}|{ep['state']}|{ep['seed']}"


def cmd_plan(a) -> int:
    from rowant_lab.pro2 import split_guard

    m = json.loads(MANIFEST.read_text())
    eps = []
    for e in m["tasks"]:
        bddl = (MANIFEST.parent / e["bddl_path"]).resolve()
        if e["suite"] not in a.suites or str(e["case_id"]).startswith(HELD_CASES) or not bddl.is_file():
            continue
        t = int(e["base_task_id"])
        if split_guard.is_test_task(e["suite"], t):  # by name too, not only by its file being absent (2026-11-16)
            continue
        if (a.tasks and t not in a.tasks) or (a.cases and e["case_id"] not in a.cases):
            continue
        for k in a.states:
            for s in a.seeds:
                eps.append({"suite": e["suite"], "task": t, "case": e["case_id"], "case_kind": e["case_kind"],
                            "state": k, "seed": s, "bddl": str(bddl), "max_steps": int(e["max_steps"]),
                            "split": "judge" if t in JUDGE[e["suite"]] else "seen"})
    eps.sort(key=lambda ep: hashlib.sha256(key(ep).encode()).hexdigest())  # mixed loads per worker
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "episodes.json").write_text(json.dumps(eps, indent=0))
    n = defaultdict(int)
    for ep in eps:
        n[f"{ep['suite']}/{ep['split']}"] += 1
    print(json.dumps({"episodes": len(eps), "by": dict(n), "cases": len({ep["case"] for ep in eps})}))
    return 0


def finished(r: dict) -> bool:
    return "success" in r or bool(r.get("unplayable"))


def rows(out: Path) -> list:
    from rowant_lab.pro2.cases import bypassed

    rs = []
    for f in sorted(out.glob("w*/rows.jsonl")):
        rs += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    for r in rs:
        r["bypassed"] = bypassed(r["case"]) and r.get("line", "depth") == "depth"  # from the case (corrected list)
    return rs


def cmd_worker(a) -> int:
    from rowant_lab.libero_env import m2_rpaths as MP

    MP.pin_numba_cache()
    from rowant_lab.libero_env import m2_bridge as BR
    from rowant_lab.libero_env import m2_episode as EP
    from rowant_lab.libero_env import m2_integrity as INT
    from rowant_lab.libero_env import m2_run as R
    from rowant_lab.pro2 import dev_episode as DE
    from rowant_lab.pro2 import long_paths as LP  # Goal's extra suite and the long-task suite's tables

    params = json.loads(a.params)
    eps = json.loads((a.out / "episodes.json").read_text())[a.index::a.of]
    wd = a.out / f"w{a.index:02d}"
    scratch = wd / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    rp = wd / "rows.jsonl"
    seen_rows = rows(a.out)
    done = {key(r) for r in seen_rows if finished(r)}  # every worker's rows: the worker count may change
    sim_tries = Counter(key(r) for r in seen_rows if str(r.get("event", "")).split(":")[0] in SIM_EVENTS)
    suites = LP.server_suites(sorted({ep["suite"] for ep in eps}))  # the long-task suite as its tables

    def connect():
        (scratch / "p.sock").unlink(missing_ok=True)  # a crashed server's socket would read as up
        # the Goal suite's types and Panda reach the server as extra_suites (goal_paths; not in M2's m2_paths)
        srv = R.start_server(scratch, wd / "server", suites, LP.server_params(params.get("server"), suites))
        try:
            c = DE.TracedClient(str(scratch / "p.sock"), timeout_s=R.SOCKET_TIMEOUT_S)  # (the server's traceback kept)
            rq = R.hello_request({"episodes": [[s] for s in suites], "params": params},
                                 INT.check(f"pro2-worker-{a.index}", None))
            bad = R.check_hello_reply(rq, c.call(rq))
            if bad:
                raise RuntimeError(f"hello refused: {bad}")
        except BaseException:
            R.stop_server(srv, scratch)
            raise
        return srv, c

    def on_alarm(_sig, _frm):
        raise EP.WatchdogTimeout(f"episode wall over {WATCHDOG_S} s")

    signal.signal(signal.SIGALRM, on_alarm)
    srv, client = connect()
    n_run = 0
    try:
        with open(rp, "a", encoding="utf-8") as fo:
            for ep in eps:
                if key(ep) in done:
                    continue
                if a.max_episodes and n_run >= a.max_episodes:
                    break  # the run restarts this worker (memory grows across episodes)
                n_run += 1
                try:
                    signal.alarm(WATCHDOG_S)
                    rec = DE.run_episode(ep, client, params, camera_px=a.camera_px)
                    signal.alarm(0)
                except EP.RUNNER_EVENTS as exc:
                    signal.alarm(0)
                    rec = {k: ep[k] for k in ("suite", "task", "case", "state", "seed")}
                    rec["event"] = f"{type(exc).__name__}: {exc}"[:300]
                    client.close()
                    R.stop_server(srv, scratch)
                    srv, client = connect()
                except Exception as exc:
                    signal.alarm(0)
                    if type(exc).__name__ not in SIM_EVENTS:
                        raise
                    sim_tries[key(ep)] += 1  # the next pass over this slot's share retries it, up to SIM_TRIES
                    rec = {k: ep[k] for k in ("suite", "task", "case", "state", "seed")}
                    rec["event"] = f"{type(exc).__name__}: {exc}"[:300]
                    if sim_tries[key(ep)] >= SIM_TRIES:
                        rec["unplayable"] = True
                rec.update(split=ep["split"], case_kind=ep["case_kind"], worker=a.index)
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
    finally:  # our own server child, always
        client.close()
        R.stop_server(srv, scratch)
    return 0


def stale_servers(slot: Path, proc: Path = Path("/proc")) -> list:
    """The process numbers of perception servers still running for one worker slot (its ``--log-dir`` is the slot's
    ``server`` folder). Read from ``/proc``; an empty list where there is none (the Mac)."""
    want, out = os.path.realpath(slot / "server"), []
    if not proc.is_dir():
        return out
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            args = [c.decode("utf-8", "replace") for c in (d / "cmdline").read_bytes().split(b"\0")]
        except OSError:
            continue  # the process ended while we looked
        if "rowant_lab.libero_env.m2_perceive" not in args or "--log-dir" not in args[:-1]:
            continue
        if os.path.realpath(args[args.index("--log-dir") + 1]) == want:
            out.append(int(d.name))
    return sorted(out)


def cmd_run(a) -> int:
    import threading

    eps = json.loads((a.out / "episodes.json").read_text())
    rc = [None] * a.workers

    def left(i: int) -> int:
        done = {key(r) for r in rows(a.out) if finished(r)}
        return sum(key(ep) not in done for ep in eps[i::a.workers])

    def keep(i: int) -> None:
        # one worker slot: restart its process until its share is done. A process that dies (2026-11-16: the GPU ran
        # out of memory under too many workers) is started again after a pause, WORKER_TRIES times in a row at most;
        # an episode it was in has no row and is simply played by the next process.
        fails = 0
        while left(i):
            # A worker the kernel kills for memory cannot stop its own server. 2026-11-18: 48 such servers held 37 GB
            # of a 41 GB pod and every new worker was killed in turn. No worker of this slot is alive here, so any
            # server of the slot is a leftover.
            gone = stale_servers(a.out / f"w{i:02d}")
            for pid in gone:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            if gone:
                print(json.dumps({"slot": i, "leftover_servers_ended": gone}), flush=True)
            with open(a.out / f"worker-{i:02d}.log", "a") as lf:
                rc[i] = subprocess.call(
                    [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "worker", "--out", str(a.out),
                     "--index", str(i), "--of", str(a.workers), "--params", a.params, "--camera-px",
                     str(a.camera_px), "--max-episodes", str(a.max_episodes)],
                    cwd=Path(__file__).resolve().parents[2], stdout=lf, stderr=subprocess.STDOUT)
            if rc[i] != 0:
                fails += 1
                if fails >= WORKER_TRIES:
                    return
                time.sleep(WORKER_PAUSE_S)
            else:
                fails = 0

    th = [threading.Thread(target=keep, args=(i,)) for i in range(a.workers)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    todo = sum(left(i) for i in range(a.workers))
    print(json.dumps({"worker_rc": rc, "unfinished": todo}))
    cmd_summary(a)
    # never "done" with episodes left: the queue then leaves the run unmarked and its supervisor starts it again
    return 4 if todo else 0


def cmd_summary(a) -> int:
    last = {}
    events = 0
    unplayable = set()
    for r in rows(a.out):
        if "success" in r:
            last[key(r)] = r
        else:
            events += 1
            if r.get("unplayable"):
                unplayable.add(key(r))
    by = defaultdict(lambda: [0, 0])
    for r in last.values():
        for g in (f"case/{r['case']}", f"split/{r['split']}", f"suite/{r['suite']}",
                  f"kind/{r.get('case_kind')}{'/bypassed' if r.get('bypassed') else ''}"):
            by[g][0] += bool(r["success"])
            by[g][1] += 1
    tested = [r for r in last.values() if not r.get("bypassed")]
    t = sorted(r["timing"]["total_s"] for r in last.values())
    res = {"episodes": len(last), "events": events, "unplayable": len(unplayable),
           "success_tested": [sum(bool(r["success"]) for r in tested), len(tested)],
           "by": {g: v for g, v in sorted(by.items())},
           "time_s": {"median": t[len(t) // 2] if t else None, "p90": t[int(0.9 * (len(t) - 1))] if t else None}}
    fr = defaultdict(int)
    cases: dict = {}
    for r in last.values():
        fr[str(r.get("finish_reason"))] += 1
        c = cases.setdefault(r["case"], {"kind": r.get("case_kind"), "bypassed": bool(r.get("bypassed"))})
        v = c.setdefault(r["suite"], [0, 0])
        v[0] += bool(r["success"])
        v[1] += 1
    res["finish_reasons"] = dict(sorted(fr.items(), key=lambda x: -x[1]))
    res["cases"] = {k: cases[k] for k in sorted(cases, key=lambda c: (c != "BASE", c))}
    (a.out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("episodes", "events", "unplayable", "success_tested", "time_s")}))
    for k, c in res["cases"].items():
        print(k, c["kind"], "BYPASSED" if c["bypassed"] else "", *(f"{s}={c[s][0]}/{c[s][1]}" for s in SUITES if s in c))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "run", "worker", "summary"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suites", nargs="+", default=list(SUITES))
    ap.add_argument("--states", nargs="+", type=int, default=[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[142])
    ap.add_argument("--tasks", nargs="+", type=int, help="plan: only these base tasks (a smoke test)")
    ap.add_argument("--cases", nargs="+", help="plan: only these cases (a smoke test)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2 - 1))
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--camera-px", type=int, default=256)
    ap.add_argument("--max-episodes", type=int, default=10, help="episodes per worker process before it restarts")
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run, "worker": cmd_worker, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
