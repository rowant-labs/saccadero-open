"""A development run on part of a planned episode list, with the unified configuration and chosen options.

EVALUATOR side (Mac). The list is a run folder's ``episodes.json`` under ``runs/pro2/colour/eval`` (the runner's own
planning: training tasks only). ``split_guard`` checks it again before anything is written.

    python3 -m rowant_lab.pro2.subset dev --src RUN --name d02-on [--cases D02,BASE] [--suites libero_goal]
        [--agent relook_place=true,relook_min_m=0.02] [--set fixture_no_flip=false] [--colour NET]
        [--server-json '{"support": {"twin_prior": {"inward": {}}}}'] [--server extra_blob_top_m=0.015]
        [--delay start_lag=true,cmd_watch=true]
        [--params-from RUN] [--every N [--offset K]]
        [--tasks libero_10:0,1,7] [--states 0 [--seed 142]]
    python3 -m rowant_lab.pro2.subset report --runs d02-off,d02-on        # by case; the first run is the reference
    python3 -m rowant_lab.pro2.subset shard --runs d02-off,d02-on --pods 3   # each run as d02-off-p0 .. -p2: an episode
                                                                           # goes to the same pod in every run
    python3 -m rowant_lab.pro2.subset unshard --runs d02-off,d02-on        # the parts' rows gathered under the run

``dev`` writes ``runs/pro2/devruns/<name>`` (``episodes.json``, ``params.json``): ``long_dev.long_params`` (or the
colour adapter with ``--colour``), then ``--set`` on the top level and ``--agent`` on the agent's parameters (values
read as JSON), then ``--server-json`` (a JSON object) merged into the perception server's parameters
(``merge_server``). ``report`` reads the rows under ``runs/pro2/colour/eval/<name>`` and pairs the runs by episode.
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
from rowant_lab.pro2 import split_guard  # noqa: E402
from rowant_lab.settings import DATA  # noqa: E402

ROOT = DATA / "runs" / "pro2" / "devruns"
EVAL = DATA / "runs" / "pro2" / "colour" / "eval"  # where the rows of finished runs are kept


def _items(text: str) -> list:
    """``text`` cut at its commas, except those inside brackets, braces or double quotes (a JSON value may hold
    commas: ``behind_occluder={"min_px":25000,"untold":true}``)."""
    out, depth, quoted, cur = [], 0, False, ""
    for ch in text or "":
        if ch == '"':
            quoted = not quoted
        elif not quoted and ch in "[{":
            depth += 1
        elif not quoted and ch in "]}":
            depth -= 1
        if ch == "," and depth == 0 and not quoted:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur] if cur else out


def _pairs(text: str) -> dict:
    out = {}
    for item in _items(text):
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def merge_server(p: dict, text: str) -> dict:
    """``--server-json``: a JSON object merged into the perception server's parameters (``p["server"]``). A value
    that is a dict is merged into the dict already there, so ``{"support": {"twin_prior": {...}}}`` adds one
    ``m2_segment`` option and keeps the others; any other value replaces. ``p`` is changed and returned."""
    if not text:
        return p
    add = json.loads(text)
    if not isinstance(add, dict):
        raise SystemExit("--server-json takes a JSON object")
    srv = dict(p.get("server") or {})
    for k, v in add.items():
        srv[k] = {**srv[k], **v} if isinstance(v, dict) and isinstance(srv.get(k), dict) else v
    p["server"] = srv
    return p


def merge_top(p: dict, text: str) -> dict:
    """``--top-json``: a JSON object merged into the run's parameters themselves; a dict value is merged into the dict
    already there (``{"arm_joints": {"readings_fit": {}}}`` keeps ``guard_mm``), any other value replaces."""
    if not text:
        return p
    add = json.loads(text)
    if not isinstance(add, dict):
        raise SystemExit("--top-json takes a JSON object")
    for k, v in add.items():
        p[k] = {**p[k], **v} if isinstance(v, dict) and isinstance(p.get(k), dict) else v
    return p


def build_params(agent: dict, top: dict, colour: str = "", server: str = "", server_keys: dict | None = None,
                 delay: dict | None = None, bundle: bool = False) -> dict:
    p = LD.colour_long_params(colour) if colour else LD.long_params()
    if bundle:  # ``--bundle``: the candidate's options first (``long_dev.bundle``), the options under test on top
        p = LD.bundle(p)
    p.update(top)
    if delay:  # ``--delay``: keys of the delay compensator's own parameters (``robot/delay.py``)
        p["delay_comp"] = {**(p.get("delay_comp") or {}), **delay}
    if server_keys:  # ``--server``: keys of the perception server's parameters
        p["server"] = {**(p.get("server") or {}), **server_keys}
    if agent:
        m1 = dict(p.get("m1") or {})
        m1["agent"] = {**(m1.get("agent") or {}), **agent}
        p["m1"] = m1
    return merge_server(p, server)


def select(eps: list, cases=None, suites=None, tasks=None) -> list:
    return [e for e in eps if (not cases or e["case"] in cases) and (not suites or e["suite"] in suites)
            and (not tasks or (e["suite"], e["task"]) in tasks)]


def cmd_dev(a) -> int:
    eps = json.loads((EVAL / a.src / "episodes.json").read_text())
    tasks = None
    if a.tasks:  # "libero_10:0,1,7;libero_goal:2": only these base tasks of those suites
        tasks = {(x.split(":")[0], int(t)) for x in a.tasks.split(";") for t in x.split(":")[1].split(",")}
    eps = select(eps, set(a.cases.split(",")) if a.cases else None, set(a.suites.split(",")) if a.suites else None,
                 tasks)
    if a.states:  # the same episodes from other start states (the list's own states replaced, each episode once per state)
        eps = [{**e, "state": int(k)} for e in eps for k in a.states.split(",")]
    if a.seed:  # and with this seed (the scored protocol: start state 0, seeds 142 to 171)
        eps = [{**e, "seed": int(a.seed)} for e in eps]
    if int(a.every) > 1:  # every N-th episode of the list (its order: suite, task, case): a part with every case in it
        eps = eps[int(a.offset)::int(a.every)]
    split_guard.check(eps, "development run")
    if not eps:
        raise SystemExit("no episode selected")
    d = ROOT / a.name
    d.mkdir(parents=True, exist_ok=True)
    b = json.dumps(eps).encode()
    (d / "episodes.json").write_bytes(b)
    if a.params_from:  # another run's parameters as they are (the registered configuration: the reference arm)
        text = (EVAL / a.params_from / "params.json").read_text()
    else:
        text = json.dumps(merge_top(build_params(_pairs(a.agent), _pairs(a.set), a.colour, a.server_json,
                                                 _pairs(a.server), _pairs(a.delay), a.bundle), a.top_json))
    (d / "params.json").write_text(text)
    print(json.dumps({"run": d.name, "episodes": len(eps), "episodes_sha256": hashlib.sha256(b).hexdigest(),
                      "params_sha256": hashlib.sha256(text.encode()).hexdigest()}))
    return 0


def load_rows(run: str) -> dict:
    out = {}
    for f in sorted((EVAL / run).glob("w*/rows.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if "success" in r:
                out[(r["suite"], r["task"], r["case"], r["state"], r["seed"])] = r
    return out


def compare(ref: dict, new: dict) -> dict:
    """Per case and in all: episodes in both, successes in each, won (new only) and lost (ref only)."""
    by = collections.defaultdict(lambda: {"n": 0, "ref": 0, "new": 0, "won": 0, "lost": 0})
    for k in sorted(set(ref) & set(new)):
        a, b = bool(ref[k]["success"]), bool(new[k]["success"])
        for name in (k[2], "ALL"):
            c = by[name]
            c["n"] += 1
            c["ref"] += a
            c["new"] += b
            c["won"] += (b and not a)
            c["lost"] += (a and not b)
    return dict(by)


def pod_of(ep: dict, pods: int) -> int:
    """Which of ``pods`` pods an episode goes to: by its own key, so the same episode lands on the same pod in every
    run and a pair stays on one machine."""
    k = f"{ep['suite']}|{ep['task']}|{ep['case']}|{ep['state']}|{ep['seed']}"
    return int(hashlib.sha256(k.encode()).hexdigest(), 16) % int(pods)


def cmd_shard(a) -> int:
    """``shard --runs a,b --pods K``: each run folder cut into ``<run>-p0`` .. ``<run>-p<K-1>`` (the same parameters,
    the episodes that ``pod_of`` gives that pod; a part with no episode is not written)."""
    out = []
    for run in a.runs.split(","):
        eps = json.loads((ROOT / run / "episodes.json").read_text())
        split_guard.check(eps, "development run")
        text = (ROOT / run / "params.json").read_text()
        for i in range(int(a.pods)):
            part = [e for e in eps if pod_of(e, a.pods) == i]
            if not part:
                continue
            d = ROOT / f"{run}-p{i}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "episodes.json").write_text(json.dumps(part))
            (d / "params.json").write_text(text)
            out.append((d.name, len(part)))
    print(" ".join(f"{n}:{k}" for n, k in out))
    return 0


def cmd_unshard(a) -> int:
    """``unshard --runs a,b``: the rows the pods sent home under ``<run>-p*`` gathered under ``<run>`` (copies), so
    ``report`` and every other reader see one run."""
    for run in a.runs.split(","):
        d = EVAL / run
        n = 0
        for part in sorted(EVAL.glob(f"{run}-p[0-9]*")):
            for f in sorted(part.glob("w*/rows.jsonl")):
                t = d / f"{part.name[len(run) + 1:]}{f.parent.name}"
                t.mkdir(parents=True, exist_ok=True)
                (t / "rows.jsonl").write_bytes(f.read_bytes())
                n += sum(1 for line in f.read_text().splitlines() if line.strip())
        if (ROOT / run / "episodes.json").is_file():
            d.mkdir(parents=True, exist_ok=True)
            for name in ("episodes.json", "params.json"):
                (d / name).write_bytes((ROOT / run / name).read_bytes())
        print(f"{run}: {n} rows from {len(list(EVAL.glob(f'{run}-p[0-9]*')))} parts")
    return 0


def p_more_lost(won: int, lost: int) -> float:
    """The keep rule's test (decided 2026-11-15): one-sided sign test on the episodes that differ, the chance of at
    least ``lost`` losses among ``won + lost`` if a loss and a win were equally likely. Rejected when below 0.05."""
    from math import comb

    n = int(won) + int(lost)
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(int(lost), n + 1)) / 2 ** n


def cmd_report(a) -> int:
    runs = a.runs.split(",")
    ref = load_rows(runs[0])
    print(f"{runs[0]}: {sum(bool(r['success']) for r in ref.values())}/{len(ref)}")
    for run in runs[1:]:
        new = load_rows(run)
        by = compare(ref, new)
        c = by.get("ALL") or {"won": 0, "lost": 0}
        p = p_more_lost(c["won"], c["lost"])
        print(f"{run}: {sum(bool(r['success']) for r in new.values())}/{len(new)}; paired with {runs[0]}: "
              f"{json.dumps(by.get('ALL'))}; p(more lost than won) {p:.3f} -> {'REJECTED' if p < 0.05 else 'kept'}")
        for case in sorted(k for k in by if k != "ALL"):
            c = by[case]
            if c["won"] or c["lost"] or a.all:
                print(f"  {case:5} n {c['n']:3}  {c['ref']:3} -> {c['new']:3}  won {c['won']} lost {c['lost']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("dev", "report", "shard", "unshard"))
    ap.add_argument("--pods", default="2", help="shard: the number of pods")
    ap.add_argument("--src", default="jd8-joints")
    ap.add_argument("--name", default="")
    ap.add_argument("--cases", default="")
    ap.add_argument("--suites", default="")
    ap.add_argument("--tasks", default="", help="dev: only these base tasks, e.g. libero_10:0,1,7")
    ap.add_argument("--states", default="", help="dev: these start states instead of the list's (e.g. 0, the scored one)")
    ap.add_argument("--seed", default="", help="dev: this seed instead of the list's")
    ap.add_argument("--agent", default="")
    ap.add_argument("--set", default="")
    ap.add_argument("--server", default="", help="dev: keys of the perception server's parameters (k=v,...)")
    ap.add_argument("--colour", default="")
    ap.add_argument("--server-json", default="", help="dev: a JSON object merged into the server's parameters")
    ap.add_argument("--top-json", default="", help="dev: a JSON object merged into the run's parameters themselves")
    ap.add_argument("--delay", default="", help="dev: keys of delay_comp, as --agent (start_lag=true,cmd_watch=true)")
    ap.add_argument("--bundle", action="store_true", help="dev: the candidate's options on (long_dev.bundle) under the others")
    ap.add_argument("--params-from", default="", help="dev: copy this run's params.json instead of building them")
    ap.add_argument("--every", default="1", help="dev: every N-th episode of the selected list")
    ap.add_argument("--offset", default="0")
    ap.add_argument("--runs", default="")
    ap.add_argument("--all", action="store_true", help="report: every case, not only those that changed")
    a = ap.parse_args(argv)
    return {"dev": cmd_dev, "report": cmd_report, "shard": cmd_shard, "unshard": cmd_unshard}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
