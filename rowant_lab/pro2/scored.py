"""The scored runs on LIBERO-Pro2.0's test split, their rehearsal on the training split, and the scorer.

The benchmark's documented protocol (``docs/evaluation.md`` in its download)::

    run_grouped_eval.py plan --split test --group all --seed 142 --num-seeds 30 --init-state-ids 0 --control-mode relative

that is: every held-out instance (339: 8 BASE and 331 shifted), seeds 142 to 171, start state 0, the manifest's step
caps, relative control: 10,170 episodes. ``plan`` writes exactly that list in our runner's format and checks it
against the benchmark's own planner when its package can be imported.

Two runs were made on that list, each registered before it was started: the first with configuration v9 (2026-11-17),
the second attempt with v11 (2026-11-20). Each has a lock file in ``docs/records``: the checksums of the run
parameters, of the two depth nets and of the benchmark's archive, written when the registration was frozen.
``verify`` compares them with what is on this machine. (A lock also names the registration's checksum and a code
commit. Those belong to the private repository this code was carved out of and cannot be checked here; see
``docs/CHANGES-FROM-AS-RUN.md``.)

    python -m rowant_lab.pro2.scored verify [--attempt 2]              # parameters, nets, archive against the lock
    python -m rowant_lab.pro2.scored rehearse --name NAME --states 0 --seeds 160 161 [--suites ...] [--every N]
    python -m rowant_lab.pro2.scored check-plan                       # our planner against the benchmark's (training split)
    python -m rowant_lab.pro2.scored unpack --test-split [--attempt 2]
    python -m rowant_lab.pro2.scored plan --test-split [--attempt 2]
    python -m rowant_lab.pro2.scored shards --src plan --name scored --pods 3 [--skip-done scored-p0,scored-p1]
    python -m rowant_lab.pro2.scored score --src plan --runs scored-p0,scored-p1,scored-p2

``rehearse`` is the same list builder on the training split (training files only, the blind type S06 left out), so the
whole path from plan to score can be exercised without the test split. ``unpack`` and ``plan`` touch the test split
and refuse without ``--test-split``: please develop on the training split.

``score`` gives four readings of the same rows (``docs/RESULTS.md``). The formula registered before the runs: each
perturbation type is the mean over its suites of that suite's success rate, and Overall is the plain mean over the 42
types (22 static, 20 dynamic; BASE is reported apart). The public board's own averages follow a slightly different
weighting (static types are averaged over their type-and-suite cells), which is reported beside it, together with the
per-episode rate and the registered formula with the episodes the simulator refused to build counted as failures.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.pro2 import split_guard  # noqa: E402
from rowant_lab.pro2 import machine  # noqa: E402
from rowant_lab.settings import DATA  # noqa: E402

COLD = DATA
BUNDLE = machine.BUNDLE  # the folder that holds the benchmark's archive ``evaluation.zip`` as downloaded
PAYLOAD = machine.BENCHMARK
ASSETS = PAYLOAD / "service-source" / "assets"
ROOT = COLD / "runs" / "pro2" / "scored"
TEST_ROOT = ROOT / "test-files"  # the test split's task files, from `unpack` on
EVAL = COLD / "runs" / "pro2" / "colour" / "eval"  # where the rows of finished runs are kept
HERE = Path(__file__).resolve().parent
RECORDS = machine.REPO_ROOT / "docs" / "records"
LOCK = RECORDS / "SCORED.lock.json"
# The second attempt (candidate v11 on the same test plan) is labelled as a second attempt wherever it is reported.
# It has its own lock and registration; without ``--attempt 2`` every command is the first run's.
LOCKS = {1: LOCK, 2: RECORDS / "SCORED2.lock.json"}
PLAN_KEYS = RECORDS / "plan-keys.json"  # the test plan's episode ids as a checksum (no path, no file name)
ARCHIVE_ASSETS = "LIBERO-Pro-evaluation/service-source/assets/"
NET_POD = "/workspace/out/colour-long/net/last.pt"  # the retrained depth net on a pod; the second net is long_dev.NET2

SEEDS = tuple(range(142, 172))  # --seed 142 --num-seeds 30
STATE = 0  # --init-state-ids 0
N_STATIC, N_DYNAMIC = 22, 20
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")

class Refused(RuntimeError):
    """The scored run's own guard: no frozen registration, or something differs from what was frozen."""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def key(ep: dict) -> str:
    return f"{ep['suite']}|{ep['task']}|{ep['case']}|{ep['state']}|{ep['seed']}"


def run_params(attempt: int = 1) -> str:
    """The run parameters as one string: the candidate (``long_dev.bundle`` on the unified colour configuration,
    ``long_dev.colour_long_params``). The lock holds their sha256 from the freeze on. The second attempt's are
    candidate v11's (``candidate_v11.run_params``: the text its confirming pod ran)."""
    from rowant_lab.pro2 import long_dev as LD

    if int(attempt) == 2:
        from rowant_lab.pro2 import candidate_v11 as V11

        return V11.run_params(NET_POD)
    if int(attempt) != 1:
        raise Refused(f"no attempt {attempt}")
    return json.dumps(LD.bundle(LD.colour_long_params(NET_POD)))


# ---------------------------------------------------------------------------------------------- the manifest's units


def manifest_entries(assets: Path = ASSETS) -> tuple:
    """(entries, train units, test units). Only ``suite``, ``base_task_id``, ``case_id``, ``case_kind``,
    ``max_steps``, ``bddl_path`` and ``bddl_sha256`` of an entry are ever used."""
    m = json.loads((assets / "task_manifest.json").read_text())
    s = json.loads((assets / "benchmark_split_v1.json").read_text())
    unit = lambda name: {(su, int(t)) for su, ts in s["splits"][name].items() for t in ts}  # noqa: E731
    return m["tasks"], unit("train"), unit("test")


def episodes_for(entries, units, files_root: Path, states, seeds, split: str, suites=None, skip_cases=()) -> list:
    """One episode per entry of ``units``, start state and seed, in the runner's format, in a fixed mixed order."""
    eps = []
    for e in entries:
        u = (e["suite"], int(e["base_task_id"]))
        case = str(e["case_id"]).strip().upper()
        if u not in units or (suites and e["suite"] not in suites) or case.startswith(tuple(skip_cases) or ("\0",)):
            continue
        for k in states:
            for s in seeds:
                eps.append({"suite": e["suite"], "task": u[1], "case": case, "case_kind": e["case_kind"],
                            "state": int(k), "seed": int(s), "bddl": str(files_root / e["bddl_path"]),
                            "max_steps": int(e["max_steps"]), "split": split})
    eps.sort(key=lambda ep: hashlib.sha256(key(ep).encode()).hexdigest())  # mixed loads per worker and per pod
    return eps


def benchmark_plan_keys(entries, split_name: str, seeds, states, assets: Path = ASSETS):
    """The benchmark's own planner (``libero_remote.benchmark.build_benchmark_plan`` on the split-filtered catalogue)
    as a set of (suite, task, case, state, seed, max_steps), or None when its package cannot be imported here."""
    import importlib
    import types

    sys.path.insert(0, str(assets.parents[0]))
    try:
        # the planner is plain Python, but its module also imports the service's HTTP client; a third-party package
        # missing here (httpx on the Mac) is stood in for by an empty module, which the planner never touches
        for _ in range(6):
            try:
                build_benchmark_plan = importlib.import_module("libero_remote.benchmark").build_benchmark_plan
                BenchmarkSplit = importlib.import_module("libero_remote.benchmark_split").BenchmarkSplit
                break
            except ModuleNotFoundError as exc:
                if not exc.name or exc.name.startswith(("libero_remote", "rowant_lab")):
                    return None
                sys.modules[exc.name] = types.ModuleType(exc.name)
                for m in [m for m in sys.modules if m.startswith("libero_remote")]:
                    del sys.modules[m]
        else:
            return None
    except Exception:
        return None
    finally:
        sys.path.pop(0)
    catalog = {"tasks": [{"suite": e["suite"], "task_id": e["base_task_id"], "case_id": e["case_id"],
                          "case_kind": e["case_kind"], "max_steps": e["max_steps"]} for e in entries]}
    catalog = BenchmarkSplit.load(assets / "benchmark_split_v1.json").filter_catalog(catalog, split_name)
    plan = build_benchmark_plan(catalog, group="all", seeds=list(seeds), init_state_ids=list(states),
                                matrix=len(states) > 1, control_mode="relative")
    return {(e["suite"], int(e["task_id"]), "BASE" if e["variant"] == "base" else str(e["variant"]),
             int(e["init_state_id"]), int(e["seed"]), int(e["max_steps"])) for e in plan.episodes}


def our_keys(eps) -> set:
    return {(e["suite"], int(e["task"]), e["case"], int(e["state"]), int(e["seed"]), int(e["max_steps"])) for e in eps}


# ------------------------------------------------------------------------------------------------------- the lock


def _params(attempt: int) -> str:
    """``run_params`` for an attempt (the first attempt's call is the one it always was)."""
    return run_params() if int(attempt) == 1 else run_params(int(attempt))


def plan_keys_sha256(eps) -> str:
    """A checksum of a plan's episode ids (suite, task, case, start state, seed, step cap), free of any path."""
    keys = sorted([e["suite"], int(e["task"]), e["case"], int(e["state"]), int(e["seed"]), int(e["max_steps"])] for e in eps)
    return hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()


def need_lock(lock_file: Path | None = None, attempt: int = 1, nets: dict | None = None, bundle: Path | None = None) -> dict:
    """The lock of an attempt, verified against what is on this machine; raises ``Refused`` on any difference.

    Always checked: the run parameters built by this code against the lock's checksum. Checked when the file is
    there: both depth nets (``nets``: ``{"net": path, "net2": path}``, default this machine's) and the benchmark's
    archive. The returned lock gains ``checked`` and ``absent`` (the names of what was and was not compared)."""
    attempt = int(attempt)
    lock_file = LOCKS[attempt] if lock_file is None else lock_file
    if not lock_file.is_file():
        raise Refused(f"no {LOCKS[attempt].name}")
    lock = json.loads(lock_file.read_text())
    if int(lock.get("attempt", 1)) != attempt:
        raise Refused(f"{lock_file.name} is the lock of attempt {lock.get('attempt', 1)}, not of attempt {attempt}")
    if hashlib.sha256(_params(attempt).encode()).hexdigest() != lock["params_sha256"]:
        raise Refused("the run parameters differ from the registered ones")
    checked, absent = ["params"], []
    here = machine.local()
    nets = nets or {"net": here["net"], "net2": here["net2"]}
    bundle = BUNDLE / "evaluation.zip" if bundle is None else bundle
    for name, path, want in (("net", Path(nets["net"]), lock["net_sha256"]), ("net2", Path(nets["net2"]), lock["net2_sha256"]),
                             ("bundle", Path(bundle), lock["bundle_sha256"])):
        if not path.is_file():
            absent.append(name)
        elif sha256_file(path) != want:
            raise Refused(f"{name} ({path.name}) differs from the registered one")
        else:
            checked.append(name)
    return {**lock, "checked": checked, "absent": absent}


# ----------------------------------------------------------------------------------------------------- the commands


def cmd_rehearse(a) -> int:
    entries, train, _test = manifest_entries()
    eps = episodes_for(entries, train, ASSETS, a.states, a.seeds, "train", suites=a.suites, skip_cases=("S06",))
    eps = [e for e in eps if Path(e["bddl"]).is_file()]
    if int(a.every) > 1:
        eps = eps[int(a.offset)::int(a.every)]
    split_guard.check(eps, "rehearsal plan")  # never a perturbed episode of a test task
    if any(split_guard.is_test_task(e["suite"], e["task"]) for e in eps) or not eps:
        raise SystemExit("the rehearsal plan is empty or holds a test task")
    d = ROOT / a.name
    d.mkdir(parents=True, exist_ok=True)
    b = json.dumps(eps).encode()
    (d / "episodes.json").write_bytes(b)
    by = collections.Counter(e["suite"] for e in eps)
    print(json.dumps({"plan": a.name, "episodes": len(eps), "by_suite": by, "cases": len({e["case"] for e in eps}),
                      "episodes_sha256": hashlib.sha256(b).hexdigest()}))
    return 0


def cmd_check_plan(a) -> int:
    """Our list builder against the benchmark's own planner, on the training split with the protocol's seeds and start
    state (S06 on both sides; no file is needed for this)."""
    entries, train, _test = manifest_entries()
    ours = our_keys(episodes_for(entries, train, ASSETS, [STATE], SEEDS, "train"))
    theirs = benchmark_plan_keys(entries, "train", SEEDS, [STATE])
    if theirs is None:
        print(json.dumps({"ours": len(ours), "benchmark_planner": "not importable here"}))
        return 2
    print(json.dumps({"ours": len(ours), "benchmark": len(theirs), "only_ours": len(ours - theirs),
                      "only_benchmark": len(theirs - ours), "equal": ours == theirs}))
    return 0 if ours == theirs else 1


def cmd_verify(a) -> int:
    """Compare this machine's parameters, depth nets and benchmark archive with an attempt's lock."""
    lock = need_lock(attempt=a.attempt, nets={"net": a.net, "net2": a.net2})
    print(json.dumps({"attempt": int(a.attempt), "lock": LOCKS[int(a.attempt)].name, "equal_to_the_lock": lock["checked"],
                      "not_on_this_machine": lock["absent"], "params_sha256": lock["params_sha256"]}, indent=1))
    return 0


def _test_split_asked(a) -> None:
    if not a.test_split:
        raise Refused("this command touches the benchmark's test split; it runs only with --test-split. "
                      "Please develop on the training split (see `rehearse`).")


def cmd_unpack(a) -> int:
    _test_split_asked(a)
    lock = need_lock(attempt=a.attempt)
    if "bundle" not in lock["checked"]:
        raise Refused(f"no evaluation.zip under {BUNDLE} (set ROWANT_BUNDLE): the test files are read from it")
    entries, _train, test = manifest_entries()
    n = collections.Counter()
    with zipfile.ZipFile(BUNDLE / "evaluation.zip") as z:
        for e in entries:
            if (e["suite"], int(e["base_task_id"])) not in test:
                continue
            data = z.read(ARCHIVE_ASSETS + e["bddl_path"])
            canon = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            if e["bddl_sha256"] not in (hashlib.sha256(data).hexdigest(), hashlib.sha256(canon).hexdigest()):
                raise Refused(f"a test file's hash differs from the manifest ({e['suite']}, {e['case_id']})")
            t = TEST_ROOT / e["bddl_path"]
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(data)
            n[e["suite"]] += 1
            n["S06"] += str(e["case_id"]).strip().upper().startswith("S06")
    started = ROOT / ("STARTED.json" if int(a.attempt) == 1 else f"STARTED{int(a.attempt)}.json")
    if not started.is_file():
        started.write_text(json.dumps({"unpacked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                       "params_sha256": lock["params_sha256"]}, indent=1) + "\n")
    print(json.dumps({"test_files": sum(n[s] for s in SUITES), "by_suite": {s: n[s] for s in SUITES}, "S06": n["S06"]}))
    return 0


def cmd_plan(a) -> int:
    _test_split_asked(a)
    need_lock(attempt=a.attempt)
    entries, _train, test = manifest_entries()
    eps = episodes_for(entries, test, TEST_ROOT, [STATE], SEEDS, "test")
    missing = sum(not Path(e["bddl"]).is_file() for e in eps)
    if missing:
        raise Refused(f"{missing} planned episodes have no task file under {TEST_ROOT}: run unpack first")
    theirs = benchmark_plan_keys(entries, "test", SEEDS, [STATE])
    if theirs is not None and theirs != our_keys(eps):
        raise Refused("our plan differs from the benchmark's own planner")
    if PLAN_KEYS.is_file() and plan_keys_sha256(eps) != json.loads(PLAN_KEYS.read_text())["plan_keys_sha256"]:
        raise Refused("this plan's episodes differ from the plan of the two scored runs")
    d = ROOT / "plan"
    d.mkdir(parents=True, exist_ok=True)
    b = json.dumps(eps).encode()
    if (d / "episodes.json").is_file() and (d / "episodes.json").read_bytes() != b:
        raise Refused("a different scored plan already exists")
    (d / "episodes.json").write_bytes(b)
    info = {"episodes": len(eps), "instances": len({(e["suite"], e["task"], e["case"]) for e in eps}),
            "by_suite": collections.Counter(e["suite"] for e in eps),
            "by_kind": collections.Counter(e["case_kind"] for e in eps),
            "checked_against_benchmark_planner": theirs is not None,
            "plan_keys_sha256": plan_keys_sha256(eps), "episodes_sha256": hashlib.sha256(b).hexdigest()}
    (d / "plan-info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))
    return 0


def shard(eps: list, pods: int) -> list:
    return [eps[i::pods] for i in range(pods)]


def cmd_shards(a) -> int:
    src = ROOT / a.src
    eps = json.loads((src / "episodes.json").read_text())
    if any(e["split"] == "test" for e in eps):
        lock = need_lock(attempt=a.attempt)
    else:
        lock = None
        split_guard.check(eps, "rehearsal shards")
    text = _params(a.attempt)
    if lock and hashlib.sha256(text.encode()).hexdigest() != lock["params_sha256"]:
        raise Refused("the run parameters differ from the frozen ones")
    if a.skip_done:  # after a lost pod: only what has no outcome at home yet (finished episodes are never run again)
        rows, unplayable, _events = load_rows(a.skip_done.split(","))
        n0 = len(eps)
        eps = [e for e in eps if key(e) not in rows and key(e) not in unplayable]
        print(json.dumps({"planned": n0, "finished_at_home": n0 - len(eps), "left": len(eps)}))
        if not eps:
            return 0
    out = []
    for i, part in enumerate(shard(eps, int(a.pods))):
        d = Path(a.rundir) / f"{a.name}-p{i}"  # pod PREFIX-a runs part p0, PREFIX-b part p1, ... (scored_pods.sh)
        d.mkdir(parents=True, exist_ok=True)
        b = json.dumps(part).encode()
        (d / "episodes.json").write_bytes(b)
        (d / "params.json").write_text(text)
        out.append({"run": d.name, "episodes": len(part), "episodes_sha256": hashlib.sha256(b).hexdigest()})
    print(json.dumps({"params_sha256": hashlib.sha256(text.encode()).hexdigest(), "shards": out}))
    return 0


def load_rows(runs, eval_root: Path | None = None) -> tuple:
    """(last outcome row per episode key, keys the simulator refused twice, event rows)."""
    eval_root = EVAL if eval_root is None else eval_root
    last, unplayable, events = {}, set(), 0
    for run in runs:
        for f in sorted((eval_root / run).glob("w*/rows.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line) if line.strip() else {}
                if "success" in r:
                    last[key(r)] = r
                elif r:
                    events += 1
                    if r.get("unplayable"):
                        unplayable.add(key(r))
    return last, unplayable - set(last), events


def score(plan: list, rows: dict, unplayable=frozenset()) -> dict:
    """The numbers from the plan's episodes. The registered formula: a type's value is the mean over its suites of
    that suite's success rate; Overall is the mean over the types (BASE apart). ``pooled`` is the same with all of a
    type's episodes in one pot. Episodes the simulator refused are left out and counted; missing ones are counted.
    ``readings`` holds Overall four ways (``docs/RESULTS.md``): the registered formula, the board's own weighting,
    per episode, and the registered formula with the refused episodes counted as failures."""
    cell = collections.defaultdict(lambda: [0, 0])  # (case, suite) -> [successes, episodes]
    refused_cell = collections.Counter()  # (case, suite) -> episodes the simulator refused to build
    task = collections.defaultdict(lambda: [0, 0])
    kind = {}
    missing = refused = 0
    for ep in plan:
        k = key(ep)
        kind[ep["case"]] = ep["case_kind"]
        if k in unplayable:
            refused += 1
            refused_cell[(ep["case"], ep["suite"])] += 1
            continue
        if k not in rows:
            missing += 1
            continue
        ok = bool(rows[k]["success"])
        counts = [cell[(ep["case"], ep["suite"])]]
        if ep["case"] != "BASE":  # a task's line is its shifted episodes, as Overall is
            counts.append(task[(ep["suite"], ep["task"])])
        for c in counts:
            c[0] += ok
            c[1] += 1
    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else None

    types = {}
    for case in sorted(kind):
        suites = {s: cell[(case, s)] for s in SUITES if cell[(case, s)][1]}
        if not suites:
            continue
        rates = [v[0] / v[1] for v in suites.values()]
        ok, n = sum(v[0] for v in suites.values()), sum(v[1] for v in suites.values())
        types[case] = {"kind": kind[case], "score": 100 * sum(rates) / len(rates), "pooled": 100 * ok / n,
                       "episodes": n, "by_suite": {s: [v[0], v[1]] for s, v in suites.items()}}
    shifted = {c: t for c, t in types.items() if c != "BASE"}
    static = [t["score"] for c, t in shifted.items() if t["kind"] == "static"]
    dynamic = [t["score"] for c, t in shifted.items() if t["kind"] == "dynamic"]
    by_suite = {}
    for s in SUITES:
        by_suite[s] = mean(100 * t["by_suite"][s][0] / t["by_suite"][s][1] for t in shifted.values() if s in t["by_suite"])
    # Overall three more ways. The board's weighting: its printed static averages are means over type-and-suite cells
    # (one static type exists in three suites, not four), and its Overall weights static and dynamic by their numbers
    # of types. Per episode: all shifted episodes in one pot. Refused as failures: the registered formula with every
    # episode the simulator refused to build counted as a failed episode of its type and suite.
    static_cells = mean(100 * v[0] / v[1] for t in shifted.values() if t["kind"] == "static" for v in t["by_suite"].values())
    board = None
    if static and dynamic:
        board = (len(static) * static_cells + len(dynamic) * mean(dynamic)) / (len(static) + len(dynamic))
    ok_all = sum(v[0] for t in shifted.values() for v in t["by_suite"].values())
    n_all = sum(v[1] for t in shifted.values() for v in t["by_suite"].values())
    as_failed = mean(mean(100 * v[0] / (v[1] + refused_cell[(c, s)]) for s, v in t["by_suite"].items())
                     for c, t in shifted.items())
    readings = {"registered_formula": mean(t["score"] for t in shifted.values()), "board_weighting": board,
                "board_weighting_static": static_cells, "per_episode": 100 * ok_all / n_all if n_all else None,
                "per_episode_counts": [ok_all, n_all], "refused_counted_as_failures": as_failed}
    return {"overall": mean(t["score"] for t in shifted.values()), "static": mean(static), "dynamic": mean(dynamic),
            "readings": readings,
            "overall_pooled": mean(t["pooled"] for t in shifted.values()),
            "types": len(shifted), "static_types": len(static), "dynamic_types": len(dynamic),
            "expected_types": [N_STATIC, N_DYNAMIC], "by_suite": by_suite,
            "by_task": {f"{s}|{t}": [v[0], v[1]] for (s, t), v in sorted(task.items())},
            "base": types.get("BASE"), "by_type": shifted,
            "episodes": {"planned": len(plan), "scored": sum(v[1] for v in cell.values()), "missing": missing,
                         "refused_by_simulator": refused}}


def cmd_score(a) -> int:
    plan = json.loads((ROOT / a.src / "episodes.json").read_text())
    rows, unplayable, events = load_rows(a.runs.split(","))
    res = score(plan, rows, unplayable)
    res["event_rows"] = events
    out = ROOT / a.src / f"score-{a.runs.split(',')[0]}.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    f1 = lambda x: "  n/a" if x is None else f"{x:5.1f}"  # noqa: E731
    f2 = lambda x: "   n/a" if x is None else f"{x:6.2f}"  # noqa: E731
    rd = res["readings"]
    print(f"Overall by the board's own weighting {f2(rd['board_weighting'])}   by the registered formula "
          f"{f2(rd['registered_formula'])}   per episode {f2(rd['per_episode'])}   refused episodes as failures "
          f"{f2(rd['refused_counted_as_failures'])}")
    print(f"registered formula: Overall {f1(res['overall'])}  static {f1(res['static'])}  dynamic {f1(res['dynamic'])}  "
          f"(pooled {f1(res['overall_pooled'])}); types {res['static_types']} static + {res['dynamic_types']} dynamic")
    print("by suite: " + "  ".join(f"{s[7:]} {f1(v)}" for s, v in res["by_suite"].items()))
    print("by task:  " + "  ".join(f"{k[7:]} {v[0]}/{v[1]}" for k, v in res["by_task"].items()))
    if res["base"]:
        print(f"BASE (not in Overall): {f1(res['base']['pooled'])} of {res['base']['episodes']} episodes")
    print("episodes: " + json.dumps(res["episodes"]) + f"  event rows {events}")
    for c, t in res["by_type"].items():
        print(f"  {c:5} {t['kind']:7} {f1(t['score'])}   " + "  ".join(f"{s[7:]} {v[0]}/{v[1]}" for s, v in t["by_suite"].items()))
    print(f"written: {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("verify", "rehearse", "check-plan", "unpack", "plan", "shards", "score"))
    ap.add_argument("--test-split", action="store_true", help="unpack, plan: yes, touch the benchmark's test split")
    ap.add_argument("--name", default="")
    ap.add_argument("--src", default="plan", help="shards, score: the plan's folder under the scored root")
    ap.add_argument("--states", nargs="+", type=int, default=[STATE])
    ap.add_argument("--seeds", nargs="+", type=int, default=[160])
    ap.add_argument("--suites", nargs="+")
    ap.add_argument("--every", default="1")
    ap.add_argument("--offset", default="0")
    ap.add_argument("--pods", default="3")
    ap.add_argument("--rundir", default=str(COLD / "runs" / "pro2" / "devruns"))
    ap.add_argument("--runs", default="")
    ap.add_argument("--skip-done", default="", help="shards: leave out episodes that already have an outcome in these runs")
    ap.add_argument("--attempt", type=int, default=1, choices=sorted(LOCKS),
                    help="2: the labelled second attempt (candidate v11; its own lock and registration)")
    ap.add_argument("--net", default=machine.local()["net"], help="verify: the main depth net's file")
    ap.add_argument("--net2", default=machine.local()["net2"], help="verify: the second depth net's file")
    a = ap.parse_args(argv)
    try:
        return {"verify": cmd_verify, "rehearse": cmd_rehearse, "check-plan": cmd_check_plan, "unpack": cmd_unpack,
                "plan": cmd_plan, "shards": cmd_shards, "score": cmd_score}[a.cmd](a)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
