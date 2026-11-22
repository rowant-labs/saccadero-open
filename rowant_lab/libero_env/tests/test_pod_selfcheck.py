"""The pod's machine check (``pro2/pod_selfcheck.py``, 2026-11-18): one core of a rented machine divided wrongly and
nothing said so. The check pins a fixed battery to each processor and compares."""
import json

import pytest

from rowant_lab.pro2 import pod_selfcheck as SC


def test_the_battery_is_the_same_twice_and_covers_division_and_the_bin_index():
    a, b = SC.battery(), SC.battery()
    assert a == b and len(a) == 47
    assert {"f64.div", "f64.div_scalar", "f32.div", "f64.bins", "f64.matmul", "int.div", "f32.exp"} <= set(a)
    assert all(isinstance(v, str) and len(v) == 8 for v in a.values())


def test_a_processor_whose_division_differs_is_named_with_its_operations():
    good = {"f64.div": "aaaa", "f64.add": "bbbb", "f64.bins": "cccc"}
    bad = dict(good, **{"f64.div": "zzzz", "f64.bins": "yyyy"})
    runs = [(c, dict(good)) for c in range(8) for _ in range(2)] + [(23, bad), (23, bad), (87, bad), (87, bad)]
    assert SC.off_majority(runs) == {23: ["f64.bins", "f64.div"], 87: ["f64.bins", "f64.div"]}
    assert SC.off_majority([(c, dict(good)) for c in range(8)]) == {}


def test_a_fault_that_shows_in_one_run_of_two_still_names_the_processor():
    good = {"f64.div": "aaaa"}
    runs = [(c, dict(good)) for c in range(6) for _ in range(2)] + [(5, {"f64.div": "qqqq"})]
    assert SC.off_majority(runs) == {5: ["f64.div"]}


def test_this_machine_passes_and_the_exit_code_says_so(capsys):
    assert SC.main(["--repeat", "2", "--jobs", "2"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["selfcheck"] == "ok" and out["differ"] == {} and out["operations"] == 47 and out["runs"] >= 2


def test_a_differing_processor_fails_the_check(monkeypatch, capsys):
    good = SC.battery()
    monkeypatch.setattr(SC.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3}, raising=False)
    monkeypatch.setattr(SC, "one", lambda cpu: dict(good, **{"f64.div": "00000000"}) if cpu == 2 else dict(good))
    assert SC.main(["--repeat", "2", "--jobs", "2"]) == 1
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["selfcheck"] == "PROCESSORS DIFFER" and out["differ"] == {"2": ["f64.div"]} and out["processors"] == 4


def test_a_check_that_cannot_run_never_passes(monkeypatch, capsys):
    def broken(_cpu):
        raise RuntimeError("processor 0: battery exit 1: No module named 'numpy'")

    monkeypatch.setattr(SC, "one", broken)
    assert SC.main(["--repeat", "1"]) == 2
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["selfcheck"] == "could not run"
