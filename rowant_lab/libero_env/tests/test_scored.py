"""The scored runs' module: the lock check, the flag in front of the test split, the list builder, the shards and the
score's four readings. No file of the benchmark is opened here: the manifest entries are made up."""

import hashlib
import json

import pytest

from rowant_lab.pro2 import dev_run as DR
from rowant_lab.pro2 import scored as SC


def _entry(suite, task, case, kind=None, steps=280):
    kind = kind or ("base" if case == "BASE" else "static" if case.startswith("S") else "dynamic")
    return {"suite": suite, "base_task_id": task, "case_id": case, "case_kind": kind, "max_steps": steps,
            "bddl_path": f"bddl/{suite}/{task}/{case}.bddl", "bddl_sha256": "0" * 64}


def test_the_protocol_is_the_documented_one():
    assert SC.SEEDS == tuple(range(142, 172)) and len(SC.SEEDS) == 30 and SC.STATE == 0
    assert (SC.N_STATIC, SC.N_DYNAMIC) == (22, 20)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_the_test_split_needs_its_flag(capsys):
    assert SC.main(["unpack"]) == 3 and SC.main(["plan"]) == 3  # refused without --test-split
    assert capsys.readouterr().err.count("--test-split") == 2


def test_a_missing_lock_changed_parameters_or_another_net_are_refused(tmp_path, monkeypatch):
    with pytest.raises(SC.Refused, match="SCORED.lock.json"):
        SC.need_lock(tmp_path / "missing.json")
    monkeypatch.setattr(SC, "run_params", lambda: '{"a": 1}')
    net, other, absent = tmp_path / "net.pt", tmp_path / "other.pt", tmp_path / "absent.zip"
    net.write_bytes(b"n")
    other.write_bytes(b"o")
    lf = tmp_path / "SCORED.lock.json"
    lf.write_text(json.dumps({"params_sha256": _sha('{"a": 1}'), "net_sha256": SC.sha256_file(net),
                              "net2_sha256": SC.sha256_file(net), "bundle_sha256": "0" * 64}))
    got = SC.need_lock(lf, nets={"net": net, "net2": net}, bundle=absent)
    assert got["checked"] == ["params", "net", "net2"] and got["absent"] == ["bundle"]  # what is not here is named
    with pytest.raises(SC.Refused, match="net2"):
        SC.need_lock(lf, nets={"net": net, "net2": other}, bundle=absent)
    with pytest.raises(SC.Refused, match="bundle"):
        SC.need_lock(lf, nets={"net": net, "net2": net}, bundle=other)
    monkeypatch.setattr(SC, "run_params", lambda: '{"a": 2}')
    with pytest.raises(SC.Refused, match="parameters"):
        SC.need_lock(lf, nets={"net": net, "net2": net}, bundle=absent)


def test_episodes_follow_the_split_units(tmp_path):
    entries = [_entry("libero_object", 1, "BASE"), _entry("libero_object", 1, "S06"), _entry("libero_object", 1, "D02"),
               _entry("libero_object", 0, "S02"), _entry("libero_goal", 3, "S15", steps=300)]
    test = {("libero_object", 1), ("libero_goal", 3)}
    eps = SC.episodes_for(entries, test, tmp_path, [0], SC.SEEDS, "test")
    assert len(eps) == 4 * 30 and {e["task"] for e in eps} == {1, 3} and {e["state"] for e in eps} == {0}
    assert {e["case"] for e in eps} == {"BASE", "S06", "D02", "S15"}  # S06 is played in the scored run
    assert {e["max_steps"] for e in eps if e["suite"] == "libero_goal"} == {300}
    assert all(e["bddl"].startswith(str(tmp_path)) and e["split"] == "test" for e in eps)
    assert eps == SC.episodes_for(list(reversed(entries)), test, tmp_path, [0], SC.SEEDS, "test")  # a fixed order
    train = SC.episodes_for(entries, {("libero_object", 0)}, tmp_path, [0], [160], "train", skip_cases=("S06",))
    assert [(e["task"], e["case"]) for e in train] == [(0, "S02")]


def test_shards_cover_the_plan_once():
    eps = [{"i": i} for i in range(10)]
    parts = SC.shard(eps, 3)
    assert sorted(e["i"] for p in parts for e in p) == list(range(10)) and [len(p) for p in parts] == [4, 3, 3]


def _plan_and_rows():
    plan, rows = [], {}

    def add(suite, task, case, kind, outcomes):
        for seed, ok in enumerate(outcomes):
            ep = {"suite": suite, "task": task, "case": case, "case_kind": kind, "state": 0, "seed": 142 + seed}
            plan.append(ep)
            if ok is not None:
                rows[SC.key(ep)] = {**ep, "success": bool(ok)}

    add("libero_object", 1, "S02", "static", [1, 1, 1, 1])        # 100%
    add("libero_goal", 3, "S02", "static", [0, 0, 0, 0, 0, 0, 0, 0])  # 0%  -> S02 = 50, pooled 33.3
    add("libero_object", 1, "D07", "dynamic", [1, 0])               # 50%
    add("libero_object", 1, "BASE", "base", [1, 1])
    add("libero_10", 5, "D07", "dynamic", [1, None])                # one missing -> 100% of 1
    return plan, rows


def test_the_board_formula():
    plan, rows = _plan_and_rows()
    res = SC.score(plan, rows)
    assert res["by_type"]["S02"]["score"] == pytest.approx(50.0)
    assert res["by_type"]["S02"]["pooled"] == pytest.approx(100 * 4 / 12)
    assert res["by_type"]["D07"]["score"] == pytest.approx(75.0)
    assert res["static"] == pytest.approx(50.0) and res["dynamic"] == pytest.approx(75.0)
    assert res["overall"] == pytest.approx(62.5) and "BASE" not in res["by_type"]
    assert res["base"]["pooled"] == pytest.approx(100.0)
    assert res["episodes"] == {"planned": 18, "scored": 17, "missing": 1, "refused_by_simulator": 0}
    assert res["by_task"]["libero_object|1"] == [5, 6]  # shifted episodes only
    assert res["by_suite"]["libero_object"] == pytest.approx(75.0) and res["by_suite"]["libero_spatial"] is None


def test_episodes_the_simulator_refused_are_left_out_and_counted():
    plan, rows = _plan_and_rows()
    refused = SC.key(plan[0])
    del rows[refused]
    res = SC.score(plan, rows, {refused})
    assert res["episodes"]["refused_by_simulator"] == 1 and res["by_type"]["S02"]["by_suite"]["libero_object"] == [3, 3]


def test_rows_are_read_once_per_episode(tmp_path):
    d = tmp_path / "run-a" / "w00"
    d.mkdir(parents=True)
    ep = {"suite": "libero_object", "task": 1, "case": "S02", "state": 0, "seed": 142}
    other = {**ep, "seed": 143}
    lines = [{**ep, "event": "WatchdogTimeout: x"}, {**ep, "success": False}, {**ep, "success": True},
             {**other, "event": "PerturbationValidationError: y", "unplayable": True}]
    (d / "rows.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    rows, unplayable, events = SC.load_rows(["run-a"], tmp_path)
    assert rows[SC.key(ep)]["success"] is True and unplayable == {SC.key(other)} and events == 2


def test_the_development_planner_refuses_test_tasks_by_name(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    entries = [_entry("libero_object", 1, "D02"), _entry("libero_object", 0, "D02"), _entry("libero_object", 0, "S06")]
    for e in entries:  # every file is there, as after the scored run's unpack
        f = assets / e["bddl_path"]
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    (assets / "task_manifest.json").write_text(json.dumps({"tasks": entries}))
    monkeypatch.setattr(DR, "MANIFEST", assets / "task_manifest.json")
    out = tmp_path / "plan"
    assert DR.main(["plan", "--out", str(out), "--suites", "libero_object", "--states", "0", "--seeds", "142"]) == 0
    eps = json.loads((out / "episodes.json").read_text())
    assert [(e["task"], e["case"]) for e in eps] == [(0, "D02")]


def test_a_run_with_dead_workers_is_not_done(tmp_path, monkeypatch):
    """``dev_run run``: a worker process that keeps dying is started again a few times, and a run that still has
    episodes without an outcome does not exit 0 (the queue then leaves it unmarked and it is started again)."""
    eps = [{"suite": "libero_object", "task": 0, "case": "D02", "state": 0, "seed": 142 + i} for i in range(4)]
    (tmp_path / "episodes.json").write_text(json.dumps(eps))
    calls = []

    def dead(cmd, **kw):
        calls.append(cmd[cmd.index("--index") + 1])
        return 1

    monkeypatch.setattr(DR.subprocess, "call", dead)
    monkeypatch.setattr(DR, "WORKER_PAUSE_S", 0)
    monkeypatch.setattr(DR, "cmd_summary", lambda a: 0)
    assert DR.main(["run", "--out", str(tmp_path), "--workers", "2"]) == 4
    assert sorted(calls) == ["0"] * DR.WORKER_TRIES + ["1"] * DR.WORKER_TRIES

    def works(cmd, **kw):  # the process writes its share's rows and exits 0
        i, of = int(cmd[cmd.index("--index") + 1]), int(cmd[cmd.index("--of") + 1])
        d = tmp_path / f"w{i:02d}"
        d.mkdir(exist_ok=True)
        with open(d / "rows.jsonl", "a") as f:
            for e in eps[i::of]:
                f.write(json.dumps({**e, "success": True}) + "\n")
        return 0

    monkeypatch.setattr(DR.subprocess, "call", works)
    assert DR.main(["run", "--out", str(tmp_path), "--workers", "2"]) == 0


def test_both_locks_hold_the_parameters_this_code_builds(tmp_path):
    from rowant_lab.pro2 import candidate_v10 as V10
    from rowant_lab.pro2 import candidate_v11 as V11

    nowhere = {"nets": {"net": tmp_path / "a", "net2": tmp_path / "b"}, "bundle": tmp_path / "c"}
    # the first run: its parameters, built here, have the checksum registered before the run
    first = SC.run_params()
    lock1, lock2 = json.loads(SC.LOCK.read_text()), json.loads(SC.LOCKS[2].read_text())
    assert SC.run_params(1) == first and SC._params(1) == first and _sha(first) == lock1["params_sha256"]
    assert SC.LOCKS[1] == SC.LOCK and SC.LOCKS[2].name == "SCORED2.lock.json" and "attempt" not in lock1
    assert SC.need_lock(attempt=1, **nowhere)["checked"] == ["params"]
    # the second attempt: candidate v11, byte for byte (not v9's parameters, and not v10's)
    second = SC.run_params(2)
    assert second == V11.run_params() and second != first and second != V10.run_params()
    assert _sha(second) == V11.PARAMS_SHA256 == lock2["params_sha256"] != V10.PARAMS_SHA256
    assert lock2["attempt"] == 2 and SC.need_lock(attempt=2, **nowhere)["attempt"] == 2
    with pytest.raises(SC.Refused, match="no attempt 3"):
        SC.run_params(3)
    # the second lock names the first one, and the published file is that file byte for byte
    assert lock2["first_lock_sha256"] == SC.sha256_file(SC.LOCK)
    # each lock opens only its own attempt
    with pytest.raises(SC.Refused, match="lock of attempt 2, not of attempt 1"):
        SC.need_lock(SC.LOCKS[2], attempt=1, **nowhere)
    with pytest.raises(SC.Refused, match="lock of attempt 1, not of attempt 2"):
        SC.need_lock(SC.LOCK, attempt=2, **nowhere)


def test_the_four_readings_of_overall():
    plan, rows = _plan_and_rows()
    rd = SC.score(plan, rows)["readings"]
    assert rd["registered_formula"] == pytest.approx(62.5) == pytest.approx(rd["board_weighting"])
    assert rd["per_episode"] == pytest.approx(40.0) and rd["per_episode_counts"] == [6, 15]
    assert rd["refused_counted_as_failures"] == pytest.approx(62.5)
    # an episode the simulator refused: left out by the registered formula, a failure in the fourth reading
    refused = SC.key(plan[0])
    del rows[refused]
    rd = SC.score(plan, rows, {refused})["readings"]
    assert rd["registered_formula"] == pytest.approx(62.5) and rd["refused_counted_as_failures"] == pytest.approx(56.25)


def test_the_boards_weighting_averages_static_types_over_their_cells():
    plan, rows = [], {}

    def add(suite, case, kind, outcomes):
        for seed, ok in enumerate(outcomes):
            ep = {"suite": suite, "task": 1, "case": case, "case_kind": kind, "state": 0, "seed": 142 + seed}
            plan.append(ep)
            rows[SC.key(ep)] = {**ep, "success": bool(ok)}

    add("libero_object", "S01", "static", [1, 1])   # S01 in two suites: 100 and 0 -> the type is 50
    add("libero_goal", "S01", "static", [0, 0])
    add("libero_object", "S02", "static", [1, 1])   # S02 in one suite: 100
    add("libero_object", "D01", "dynamic", [1, 0])  # 50
    res = SC.score(plan, rows)
    assert res["static"] == pytest.approx(75.0) and res["overall"] == pytest.approx(200 / 3)  # types weigh the same
    rd = res["readings"]
    assert rd["board_weighting_static"] == pytest.approx(200 / 3)  # the three static cells weigh the same
    assert rd["board_weighting"] == pytest.approx((2 * 200 / 3 + 50) / 3)


def test_the_published_parameter_files_are_the_registered_texts():
    from rowant_lab.pro2 import machine

    for attempt, name in ((1, "v9"), (2, "v11")):
        text = (machine.REPO_ROOT / "configs" / f"{name}.registered.json").read_text()
        assert _sha(text) == json.loads(SC.LOCKS[attempt].read_text())["params_sha256"]  # byte for byte
        assert text == SC._params(attempt)  # and the code here builds the same text
