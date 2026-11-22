"""The colour-only line on the long suite's tables: the test-task guard, frame plans keyed by table, the scripted
sweeps, the stored cameras, one net for every key (training options, per-key scores) and the acceptance check."""

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from rowant_lab.pro2 import colour_frames as CF
from rowant_lab.pro2 import colour_line as CL
from rowant_lab.pro2 import colour_long as CG
from rowant_lab.pro2 import colour_sweep as SW
from rowant_lab.pro2 import long_paths as LP
from rowant_lab.pro2 import machine as MC
from rowant_lab.pro2 import split_guard as SG

# where the benchmark's own task files are on this machine (used as text only: no file under it is opened here)
PRO2 = str(MC.BENCHMARK / "service-source" / "assets" / "bddl" / "libero-pro2.0-1680-v1")
CELLS = {0: "libero_long_living_room", 1: "libero_long_living_room", 4: "libero_long_living_room",
         6: "libero_long_living_room", 7: "libero_long_living_room", 2: "libero_long_kitchen",
         3: "libero_long_kitchen", 8: "libero_long_kitchen", 9: "libero_long_kitchen", 5: "libero_long_study"}


def _plain(suite, name="SCENE_task"):
    return f"{SG.PLAIN_ROOT}/{suite}/{name}.bddl"


def _ep(task, case, suite="libero_10", bddl=None, state=8):
    sub = "00_base" if case == "BASE" else "01_static"
    return {"suite": suite, "task": task, "case": case, "state": state, "seed": 142 + state, "max_steps": 520,
            "bddl": bddl or f"{PRO2}/{suite}/{sub}/{case}__t{task}.bddl", "split": "seen", "case_kind": "static"}


# ---------------------------------------------------------------------------------------------- the test-task guard
def test_guard_refuses_perturbed_test_tasks_only():
    assert SG.TEST_TASKS == {"libero_10": (5, 8), "libero_goal": (3, 6), "libero_object": (1, 8),
                             "libero_spatial": (0, 8)}
    for suite, tasks in SG.TEST_TASKS.items():
        for t in tasks:
            assert "perturbed" in SG.refusal({"suite": suite, "task": t, "case": "S01", "bddl": _plain(suite)})
            assert "perturbed" in SG.refusal({"suite": suite, "task": t, "case": "D07", "bddl": f"{PRO2}/x.bddl"})
            # the plain version: BASE from the simulator's own folder of base task files
            assert SG.refusal({"suite": suite, "task": t, "case": "BASE", "bddl": _plain(suite)}) is None
            # BASE from anywhere else (Pro2.0's own file of a test task, a missing path) is not the plain version
            assert "plain" in SG.refusal({"suite": suite, "task": t, "case": "BASE", "bddl": f"{PRO2}/{suite}/00_base/BASE__x.bddl"})
            assert "plain" in SG.refusal({"suite": suite, "task": t, "case": "BASE"})
            assert "plain" in SG.refusal({"suite": suite, "task": t, "case": "BASE",
                                          "bddl": f"{SG.PLAIN_ROOT}/{suite}/../other/x.bddl"})
            assert "plain" in SG.refusal({"suite": suite, "task": t, "case": "BASE", "bddl": _plain("libero_90")})
    # training tasks are never refused, whatever the case or the file
    for t in (0, 1, 2, 3, 4, 6, 7, 9):
        assert SG.refusal(_ep(t, "S15")) is None and SG.refusal(_ep(t, "BASE")) is None
    assert SG.refusal({"suite": "libero_long_kitchen", "task": 8, "case": "S01"}) is None  # a key is not a suite
    SG.check([_ep(0, "S01"), {"suite": "libero_10", "task": 5, "case": "BASE", "bddl": _plain("libero_10")}])
    with pytest.raises(SG.Refused, match="libero_10 task 8 is a benchmark test task"):
        SG.check([_ep(0, "S01"), _ep(8, "S01")], "frame plan")


def test_the_first_planner_and_the_worker_call_the_guard():
    import inspect

    assert "split_guard.check(items" in inspect.getsource(CF.cmd_plan)
    src = inspect.getsource(CF.cmd_worker)
    assert src.index("split_guard.check(items") < src.index("adapter_class()")  # before any simulator is built


# ---------------------------------------------------------------------------------------------- frame names
def test_frame_names_of_old_items_are_unchanged_and_new_items_go_by_key():
    old_r = {"split": "val", "suite": "libero_object", "task": 7, "case": "D03", "state": 8, "chain": None}
    old_c = {"split": "train", "suite": "libero_goal", "task": 2, "case": "BASE", "state": 1, "chain": [[0] * 7]}
    assert CF.frame_name(old_r, 0) == "val/libero_object/07/D03/r8_0000.npz"
    assert CF.frame_name(old_c, 24, True) == "train/libero_goal/02/BASE/c1_0024x.npz"
    assert CF.frame_seed(old_r, 0) == "libero_object|7|D03|8|r|0" and CF.frame_seed(old_c, 24) == "libero_goal|2|BASE|1|c|24"
    assert CF.frame_key(old_r) == "libero_object"
    new = {**old_r, "suite": "libero_10", "task": 9, "key": "libero_long_kitchen"}
    assert CF.frame_name(new, 0) == "val/libero_long_kitchen/09/D03/r8_0000.npz"
    sweep = {"split": "train", "suite": "libero_10", "task": 5, "case": "BASE", "state": 4, "chain": None,
             "key": "libero_long_study", "plain": True, "sweep": {"n": 3, "steps": 240}}
    assert CF.frame_name(sweep, 36) == "train/libero_long_study/05/PLAIN/s4-03_0036.npz"
    assert CF.frame_kind(sweep) == "s03" and CF.frame_seed(sweep, 0) == "libero_10|5|BASE|4|s03|0"
    assert CF.frame_seed({**sweep, "sweep": {"n": 4, "steps": 240}}, 0) != CF.frame_seed(sweep, 0)
    assert not CF.drawn_frame(CF.frame_name(sweep, 36))


# ---------------------------------------------------------------------------------------------- cells
def test_task_cells_from_the_build_record():
    doc = {"cells": {"libero_long_kitchen": {"tasks": [2, 3, 8, 9]}, "libero_long_living_room": {"tasks": [0, 1, 4, 6, 7]},
                     "libero_long_study": {"tasks": [5]}}}
    cells = CG.task_cells(doc)
    assert cells == CELLS
    assert CG.key_for("libero_10", 5, cells) == "libero_long_study" and CG.key_for("libero_goal", 5, cells) == "libero_goal"
    with pytest.raises(ValueError):
        CG.key_for("libero_10", 11, cells)
    with pytest.raises(ValueError, match="does not know"):
        CG.task_cells({"cells": {"libero_long_garage": {"tasks": [1]}}})
    with pytest.raises(ValueError, match="two cells"):
        CG.task_cells({"cells": {"libero_long_kitchen": {"tasks": [2]}, "libero_long_study": {"tasks": [2]}}})


@pytest.mark.skipif(not CG.CELLS_FILE.is_file(), reason="the long suite's library tree (its build record) is not on this machine")
def test_the_build_record_gives_the_cells_the_runner_knows():
    cells = CG.load_cells()
    assert cells == CELLS and set(cells.values()) == set(LP.CELLS)


# ---------------------------------------------------------------------------------------------- the plan
def _episodes():
    return [_ep(t, c) for t in (0, 1, 2, 3, 4, 6, 7, 9) for c in ("BASE", "S01", "D07")]


def test_long_items_mirror_the_three_suites_split():
    items = CG.long_items(_episodes(), CELLS)
    CG.check_plan(items)
    assert {it["suite"] for it in items} == {"libero_10"}
    assert {it["key"] for it in items} == {"libero_long_kitchen", "libero_long_living_room"}  # no study task here
    for it in items:
        assert it["key"] == CELLS[it["task"]] and it["seed"] == 142 + it["state"] and it["max_steps"] == 520
        assert it["state"] not in (2, 3, 9)
        if it["task"] in (7, 9):
            assert it["split"] == "val" and it["state"] in (8, 0, 1)
        else:
            assert it["split"] == "train" and it["state"] in CF.TRAIN_STATES
    resets = [it for it in items if not it.get("sweep") and not it["chain"]]
    sweeps = [it for it in items if it.get("sweep")]
    assert len(resets) == 6 * 3 * 6 + 2 * 3 * 3  # seen: six start states; held back: three
    assert len(sweeps) == 8 * 3 * 2 and {it["state"] for it in sweeps} == {0, 1}
    assert all(it["every"] == (CF.VAL_EVERY if it["split"] == "val" else CF.EVERY) for it in sweeps)
    assert all(it["sweep"] == {"n": 0, "steps": 240} and it["chain"] is None for it in sweeps)
    assert all(it["every"] == 0 for it in resets)
    # the episode list's own start state (8 here) is not what is rendered: only files and step caps are taken
    assert CG.long_items([_ep(0, "S01", state=3)], CELLS) == CG.long_items([_ep(0, "S01", state=8)], CELLS)


def test_stored_chains_replace_sweeps_only_at_allowed_starts():
    acts = [[0.1] * 7] * 50
    chains = {("libero_10", 0, "S01", 0, 142): acts,       # seen task, training start: replayed
              ("libero_10", 0, "S01", 5, 147): acts,       # another training start: replayed too
              ("libero_10", 0, "S01", 8, 150): acts,       # start 8 of a seen task is never a training frame
              ("libero_10", 7, "S01", 8, 150): acts,       # held-back task at the validation start: replayed
              ("libero_10", 0, "S01", 1, 999): acts,       # not the start's own seed
              ("libero_goal", 0, "S01", 1, 143): acts}     # another suite
    items = [it for it in CG.long_items(_episodes(), CELLS, chains) if it["case"] == "S01" and it["task"] in (0, 7)]
    CG.check_plan(items)
    got = {(it["task"], it["state"], "chain" if it["chain"] else "sweep") for it in items if it["chain"] or it.get("sweep")}
    assert got == {(0, 0, "chain"), (0, 5, "chain"), (0, 1, "sweep"), (7, 8, "chain"), (7, 0, "sweep"), (7, 1, "sweep")}
    assert CG.frames_of(next(it for it in items if it["chain"] and it["task"] == 0)) == 1 + 50 // CF.EVERY
    assert CG.frames_of(next(it for it in items if it["chain"] and it["task"] == 7)) == 1 + 50 // CF.VAL_EVERY


def test_long_items_refuse_what_is_not_a_training_episode_of_the_long_suite():
    with pytest.raises(SG.Refused):
        CG.long_items(_episodes() + [_ep(5, "S01")], CELLS)
    with pytest.raises(SG.Refused):  # a test task's BASE does not come from an episode list, even with the plain file
        CG.long_items(_episodes() + [_ep(8, "BASE", bddl=_plain("libero_10"))], CELLS)
    with pytest.raises(ValueError, match="another suite"):
        CG.long_items(_episodes() + [_ep(0, "S01", suite="libero_goal")], CELLS)
    with pytest.raises(ValueError, match="S06"):
        CG.long_items(_episodes() + [_ep(0, "S06")], CELLS)
    with pytest.raises(ValueError, match="two files"):
        CG.long_items(_episodes() + [_ep(0, "S01", bddl=f"{PRO2}/libero_10/01_static/S01__other.bddl")], CELLS)


def _plain_rows():
    return [{"suite": "libero_10", "task": 5, "bddl": _plain("libero_10", "STUDY"), "name": "STUDY", "max_steps": 520},
            {"suite": "libero_10", "task": 8, "bddl": _plain("libero_10", "KITCHEN"), "name": "KITCHEN", "max_steps": 520},
            {"suite": "libero_spatial", "task": 0, "bddl": _plain("libero_spatial", "bowl"), "name": "bowl", "max_steps": 220}]


def test_plain_items_are_base_only_with_sweeps_and_long_scenes_validate_on_start_8():
    items = CG.plain_items(_plain_rows(), CELLS)
    CG.check_plan(items)
    assert all(it["case"] == "BASE" and it["plain"] is True for it in items)
    by = {}
    for it in items:
        by.setdefault((it["suite"], it["task"], it["split"]), []).append(it)
    study, kitchen, spatial = by[("libero_10", 5, "train")], by[("libero_10", 8, "train")], by[("libero_spatial", 0, "train")]
    assert {it["key"] for it in study} == {"libero_long_study"} and {it["key"] for it in kitchen} == {"libero_long_kitchen"}
    assert all("key" not in it for it in spatial)  # filed under its suite, as the suite's own frames
    for group, n in ((study, CG.SWEEPS["plain_long"]), (kitchen, CG.SWEEPS["plain_long"]), (spatial, CG.SWEEPS["plain_other"])):
        assert {it["state"] for it in group} == set(CF.TRAIN_STATES)
        assert sum(1 for it in group if not it.get("sweep")) == 6 and sum(1 for it in group if it.get("sweep")) == 6 * n
    assert all(it["sweep"]["steps"] == 220 for it in spatial if it.get("sweep"))  # no longer than the suite's step cap
    # a plain long-suite scene is also taken from start 8, for validation only (the study table's only validation)
    for task in (5, 8):
        val = by[("libero_10", task, "val")]
        assert {it["state"] for it in val} == {8} and sum(1 for it in val if it.get("sweep")) == CG.SWEEPS["plain_val"]
        assert all(it["every"] in (0, CF.EVERY) for it in val)
    assert CF.frame_name(by[("libero_10", 5, "val")][0], 0) == "val/libero_long_study/05/PLAIN/r8_0000.npz"
    assert CF.frame_name(by[("libero_10", 8, "val")][0], 0) == "val/libero_long_kitchen/08/PLAIN/r8_0000.npz"
    assert ("libero_spatial", 0, "val") not in by  # the suites' validation stays the first net's
    with pytest.raises(ValueError, match="not a test task"):
        CG.plain_items([{"suite": "libero_10", "task": 0, "bddl": _plain("libero_10")}], CELLS)
    with pytest.raises(SG.Refused):
        CG.plain_items([{"suite": "libero_10", "task": 5, "bddl": f"{PRO2}/libero_10/00_base/BASE__x.bddl"}], CELLS)


def test_check_plan_refuses_bad_items_and_the_summary_counts_frames():
    good = CG.long_items(_episodes(), CELLS) + CG.plain_items(_plain_rows(), CELLS)
    CG.check_plan(good)
    with pytest.raises(ValueError, match="two items"):
        CG.check_plan(good + [good[0]])
    with pytest.raises(ValueError, match="not a train state"):
        CG.check_plan([{**good[0], "state": 8}])
    with pytest.raises(ValueError, match="not a val state"):
        CG.check_plan([{**next(it for it in good if it["split"] == "val"), "state": 4}])
    with pytest.raises(ValueError, match="held-back task"):
        CG.check_plan([{**good[0], "task": 7}])
    with pytest.raises(ValueError, match="neither a held-back task"):  # start 8 of a seen training task, as validation
        CG.check_plan([{**good[0], "split": "val", "state": 8}])
    with pytest.raises(SG.Refused):
        CG.check_plan([{**good[0], "task": 8, "case": "S01"}])
    for bad_state in (2, 3, 9):
        with pytest.raises(ValueError):
            CG.check_plan([{**good[0], "state": bad_state}])
    s = CG.summary(good)
    assert s["items"] == len(good) and s["frames_est"] == sum(CG.frames_of(it) for it in good)
    assert s["frames_by"]["libero_long_study/train"] == 6 + 6 * 8 * 20 and s["frames_by"]["libero_long_study/val"] == 1 + 4 * 20
    # task 9: three cases, two starts, a frame every 30 steps; plain task 8 from start 8: four sweeps
    assert s["frames_by"]["libero_long_kitchen/val/sweep"] == 3 * 2 * (240 // 30) + 4 * 20
    scenes = {(r["suite"], r["task"]): r for r in s["plain_scenes"]}
    assert scenes[("libero_10", 5)]["states"] == {"train": [0, 1, 4, 5, 6, 7], "val": [8]}
    assert scenes[("libero_10", 8)]["key"] == "libero_long_kitchen" and scenes[("libero_10", 8)]["scene"] == "KITCHEN"
    assert scenes[("libero_10", 8)]["states"] == {"train": [0, 1, 4, 5, 6, 7], "val": [8]}
    assert scenes[("libero_spatial", 0)]["states"] == {"train": [0, 1, 4, 5, 6, 7]}


def test_frames_report_compares_a_run_with_its_plan(tmp_path, capsys):
    import gzip

    items = CG.long_items([_ep(0, "S01"), _ep(9, "BASE")], CELLS) + CG.plain_items(_plain_rows()[:1], CELLS)
    rows = [{"item": [it["suite"], it["task"], it["case"], it["state"]], "split": it["split"], "status": "ok",
             "n": CG.frames_of(it), "key": CF.frame_key(it)} for it in items]
    rows[0].update(status="PerturbationValidationError: no start", n=0)
    rows[1].update(status="RuntimeError: x", n=0)
    rows += [{"item": ["libero_goal", 0, "BASE", 0], "split": "train", "status": "ok", "n": 1}]  # a first-net row: no key
    rep = CG.frames_report(items, rows[:-1])
    assert rep["items"] == len(items) and rep["finished"] == len(items) and rep["failed"] == 2
    assert rep["failed_by"] == {"PerturbationValidationError": 1, "RuntimeError": 1}
    assert rep["frames"]["libero_long_study/val"] == [81, 81] and rep["frames"]["libero_long_living_room/train"] == [44, 46]
    assert rep["frames_total"] == [sum(CG.frames_of(it) for it in items) - 2, sum(CG.frames_of(it) for it in items)]
    assert CG.frames_report(items, rows[:3], {"libero_long_study/train": 5})["frames"]["libero_long_study/train"][0] == 5
    # the command: the plan, the workers' rows and the files on disk
    with gzip.open(tmp_path / "frames-plan.json.gz", "wt") as f:
        json.dump(items, f)
    (tmp_path / "meta-00.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:-1]) + "\n")
    d = tmp_path / "frames" / "val" / "libero_long_study" / "05" / "PLAIN"
    d.mkdir(parents=True)
    (d / "r8_0000.npz").write_bytes(b"")
    assert CG.main(["frames-report", "--plan", str(tmp_path), "--run", str(tmp_path), "--out", str(tmp_path / "r.json")]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["failed"] == 2
    assert json.loads((tmp_path / "r.json").read_text())["frames"]["libero_long_study/val"] == [1, 81]


def test_plan_command_keeps_the_old_plan_and_writes_the_union(tmp_path):
    import gzip

    old = [{"suite": "libero_goal", "task": 0, "case": "BASE", "state": 0, "seed": 142, "bddl": f"{PRO2}/g.bddl",
            "max_steps": 300, "split": "train", "every": 0, "chain": None},
           {"suite": "libero_goal", "task": 7, "case": "S01", "state": 1, "seed": 143, "bddl": f"{PRO2}/h.bddl",
            "max_steps": 300, "split": "val", "every": 30, "chain": [[0.0] * 7] * 61}]
    (tmp_path / "old").mkdir()
    with gzip.open(tmp_path / "old" / "frames-plan.json.gz", "wt") as f:
        json.dump(old, f)
    (tmp_path / "eps.json").write_text(json.dumps(_episodes()))
    (tmp_path / "plain.json").write_text(json.dumps({"tasks": _plain_rows()}))
    (tmp_path / "cells.json").write_text(json.dumps({"cells": {c: {"tasks": [t for t, k in CELLS.items() if k == c]}
                                                              for c in set(CELLS.values())}}))
    args = ["plan", "--out", str(tmp_path / "out"), "--episodes", str(tmp_path / "eps.json"), "--old-plan",
            str(tmp_path / "old"), "--plain", str(tmp_path / "plain.json"), "--cells", str(tmp_path / "cells.json")]
    assert CG.main(args) == 0
    items = CF.load_plan(tmp_path / "out")
    assert items[:2] == old  # the first net's items, as they are, first
    info = json.loads((tmp_path / "out" / "plan-info.json").read_text())
    assert info["items"] == len(items) and info["old_plan"]["items"] == 2 and info["chains_used"] == 0
    assert [r[:2] for r in info["plain"]["tasks"]] == [["libero_10", 5], ["libero_10", 8], ["libero_spatial", 0]]
    assert info["frames_by"]["libero_goal/val/chain"] == 1 + 61 // 30
    # only the long suite's plain scenes, and no old plan
    assert CG.main(args[:5] + ["--old-plan", "none", "--plain", str(tmp_path / "plain.json"), "--cells",
                               str(tmp_path / "cells.json"), "--plain-suites", "libero_10"]) == 0
    items = CF.load_plan(tmp_path / "out")
    assert {it["suite"] for it in items} == {"libero_10"} and {it["task"] for it in items if it.get("plain")} == {5, 8}
    # a perturbed test-task episode in the list stops the plan
    (tmp_path / "eps.json").write_text(json.dumps(_episodes() + [_ep(8, "D01")]))
    with pytest.raises(SG.Refused):
        CG.main(args)


# ---------------------------------------------------------------------------------------------- sweeps
def _scene():
    p0 = np.array([-0.20, 0.0, 1.15])
    shoulder = np.array([-0.66, 0.0, 0.912 + 0.333])
    anchors = np.array([[0.0, 0.1, 0.92], [0.05, -0.2, 0.95], [-0.1, 0.25, 0.90], [0.45, 0.0, 0.91]])
    return p0, shoulder, anchors


def test_next_leg_is_seeded_and_stays_in_the_workspace():
    p0, shoulder, anchors = _scene()
    a = SW.next_leg(random.Random("s|1"), p0, anchors, shoulder)
    assert a == SW.next_leg(random.Random("s|1"), p0, anchors, shoulder)
    kinds = set()
    lo, hi = SW.SWEEP["wrist_m"]
    for k in range(400):
        rng = random.Random(f"s|{k}")
        leg = SW.next_leg(rng, p0, anchors, shoulder)
        before = rng.random()
        rng2 = random.Random(f"s|{k}")
        [rng2.random() for _ in range(SW.N_DRAWS)]
        assert before == rng2.random()  # a leg takes the same number of draws whatever it turns out to be
        kinds.add(leg["kind"])
        pos = np.asarray(leg["pos"])
        # where the arm reaches it bent: the wrist of a hand pointing down neither too far from the shoulder nor too near
        assert lo - 1e-9 <= np.linalg.norm(SW.wrist_of(pos, shoulder) - shoulder) <= hi + 1e-9
        # between the lowest object and a little over the start (the reach clamp may move a point a few cm past them)
        assert anchors[:, 2].min() - 0.12 <= pos[2] <= p0[2] + SW.SWEEP["top_m"] + 0.12
        assert leg["grip"] in (-1.0, 1.0) and abs(leg["turn"]) <= SW.SWEEP["turn"]
        assert SW.SWEEP["leg_steps"][0] <= leg["steps"] <= SW.SWEEP["leg_steps"][1]
        assert SW.SWEEP["dwell_steps"][0] <= leg["dwell"] <= SW.SWEEP["dwell_steps"][1]
        assert SW.SWEEP["turn_steps"][0] <= leg["turn_steps"] <= SW.SWEEP["turn_steps"][1]
    assert kinds == {"anchor", "free"}
    free = SW.next_leg(random.Random(0), p0, np.zeros((0, 3)), shoulder)  # an empty scene still gives a point
    assert free["kind"] == "free" and np.isfinite(free["pos"]).all()


def test_clamp_reach_and_move_action():
    s = np.array([0.0, 0.0, 1.0])
    assert np.allclose(SW.clamp_reach([2.0, 0.0, 1.0], s, 0.3, 0.78), [0.78, 0.0, 1.0])
    assert np.allclose(SW.clamp_reach([0.1, 0.0, 1.0], s, 0.3, 0.78), [0.3, 0.0, 1.0])
    assert np.allclose(SW.clamp_reach([0.5, 0.0, 1.0], s, 0.3, 0.78), [0.5, 0.0, 1.0])
    assert np.isfinite(SW.clamp_reach(s, s, 0.3, 0.78)).all()
    far = SW.move_action([0.0, 0.0, 1.0], [0.3, -0.2, 0.9], 1.0, -0.4)
    assert len(far) == 7 and np.linalg.norm(far[:3]) == pytest.approx(1.0) and far[3:] == [0.0, 0.0, -0.4, 1.0]
    near = SW.move_action([0.299, -0.2, 0.9], [0.3, -0.2, 0.9], -1.0)
    assert near[0] == pytest.approx(2.0 * 0.001 / 0.05) and near[5] == 0.0 and near[6] == -1.0
    assert all(-1.0 <= x <= 1.0 for x in far + near)
    assert SW.move_action([0, 0, 0], [0, 0, 0], 1.0, 0.3, rot=[0.1, -0.2, 0.05])[3:] == [0.1, -0.2, 0.05, 1.0]


def test_points_are_kept_where_the_wrist_is_in_reach():
    s = np.array([-0.66, 0.0, 1.245])
    w = SW.wrist_of([0.062, -0.02, 0.95], s)  # a hand pointing down: the wrist above the point, a little back
    assert w[2] == pytest.approx(0.95 + SW.SWEEP["wrist_up_m"])
    assert np.linalg.norm((w - [0.062, -0.02, 0.95])[:2]) == pytest.approx(SW.SWEEP["wrist_back_m"])
    assert np.linalg.norm(w[:2] - s[:2]) < np.linalg.norm(np.array([0.062, -0.02]) - s[:2])
    lo, hi = SW.SWEEP["wrist_m"]
    near_ok = np.array([0.0, 0.05, 0.95])
    assert np.allclose(SW.clamp_point(near_ok, s), near_ok)  # a point over the table's middle is left alone
    far = SW.clamp_point([0.104, -0.042, 1.092], s)  # the far, high point that stretched the arm out in the Mac check
    assert np.linalg.norm(SW.wrist_of(far, s) - s) == pytest.approx(hi) and far[0] < 0.104
    close = SW.clamp_point([-0.56, 0.02, 1.0], s)  # nearly under the shoulder: moved out
    assert np.linalg.norm(SW.wrist_of(close, s) - s) == pytest.approx(lo)
    assert np.isfinite(SW.clamp_point(s, s)).all()


def test_the_hand_orientation_is_held_and_turned_only_about_the_vertical_axis():
    assert np.allclose(SW.rotvec_of(np.eye(3)), 0.0)
    assert np.allclose(SW.rotvec_of(SW.rotz(0.7)), [0.0, 0.0, 0.7])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(-0.4), -math.sin(-0.4)], [0.0, math.sin(-0.4), math.cos(-0.4)]])
    assert np.allclose(SW.rotvec_of(Rx), [-0.4, 0.0, 0.0])
    p0, shoulder, anchors = _scene()
    R0 = np.diag([1.0, -1.0, -1.0])  # a hand pointing down
    sw = SW.Sweep("item|3", p0, shoulder, R0=R0)
    act = sw.act(p0, anchors, None, R0 @ Rx)  # the hand was tipped by a push: the command tips it back
    want = SW.rotz(sw.yaw) @ R0
    assert np.allclose(act[3:6], SW._capped(2.0 * SW.rotvec_of(want @ (R0 @ Rx).T) / 0.5))
    assert abs(act[3]) + abs(act[4]) > 0.1
    held = SW.Sweep("item|3", p0, shoulder, R0=R0)
    assert np.allclose(held.act(p0, anchors, None, SW.rotz(held.yaw) @ R0)[3:6], 0.0) or held.yaw != 0.0
    yaws, eef = [], p0.copy()
    for _ in range(600):
        a = sw.act(eef, anchors, None, SW.rotz(sw.yaw) @ R0)
        eef = eef + 0.017 * np.asarray(a[:3])
        yaws.append(sw.yaw)
    assert max(abs(y) for y in yaws) <= SW.SWEEP["yaw_max"] and len(set(yaws)) >= 3  # some legs turn the hand
    assert SW.Sweep("item|3", p0, shoulder).act(p0, anchors)[3:5] == [0.0, 0.0]  # without the pose: no tilt command


def test_a_hand_that_stays_put_ends_the_sweep():
    p0, shoulder, anchors = _scene()
    sw, n = SW.Sweep("item|4", p0, shoulder), 0
    while not sw.stuck and n < 200:
        sw.act(p0, anchors)
        n += 1
    assert sw.stuck and n == SW.SWEEP["stuck_steps"] + 1
    moving, eef = SW.Sweep("item|4", p0, shoulder), p0.copy()
    for _ in range(240):
        eef = eef + 0.017 * np.asarray(moving.act(eef, anchors)[:3])
    assert not moving.stuck


def _run_sweep(speed: float, steps: int = 240, blocked_below: float | None = None):
    """A sweep with a hand that follows the command at ``speed`` a step (and cannot go below ``blocked_below``)."""
    p0, shoulder, anchors = _scene()
    sw, eef, legs, parts, path = SW.Sweep("item|0", p0, shoulder), p0.copy(), [], [], []
    for _ in range(steps):
        act = sw.act(eef, lambda: anchors)
        assert len(act) == 7 and all(-1.0 <= x <= 1.0 for x in act)
        if not legs or legs[-1] is not sw.leg:
            legs.append(sw.leg)
            parts.append([])
        if not parts[-1] or parts[-1][-1] != sw.part:
            parts[-1].append(sw.part)
        eef = eef + speed * np.asarray(act[:3])
        if blocked_below is not None:
            eef[2] = max(eef[2], blocked_below)
        path.append(eef.copy())
    return sw, legs, parts, np.asarray(path)


def test_sweep_legs_do_not_depend_on_how_the_hand_moves():
    fast, legs_fast, parts, path = _run_sweep(0.017)
    slow, legs_slow, _p, _q = _run_sweep(0.004)
    n = min(len(legs_fast), len(legs_slow))
    assert n >= 4 and legs_fast[:n] == legs_slow[:n]  # the same series of legs, however far the hand gets
    assert fast.n_reached > slow.n_reached and fast.n_reached >= 4 and fast.n_legs == len(legs_fast)
    # a leg goes up, across and down (a short hop straight down), in that order
    assert all(p in (["up", "across", "down"], ["down"], ["up", "across"], ["up"]) for p in parts)
    assert sum(p == ["up", "across", "down"] for p in parts) >= 3
    p0, _shoulder, anchors = _scene()
    assert path[:, 2].max() <= p0[2] + SW.SWEEP["top_m"] + 0.12 and path[:, 2].min() >= anchors[:, 2].min() - 0.12


def test_a_sweep_gives_up_what_it_cannot_reach_and_moves_on():
    still, legs_still, _p, _q = _run_sweep(0.0)  # a hand that never moves: every part stalls, every leg ends early
    assert still.n_reached == 0 and len(legs_still) >= 6
    # a hand that cannot go below a level (it rests on something): legs to lower points are given up, the sweep goes on
    _p0, _shoulder, anchors = _scene()
    level = float(anchors[:, 2].max()) + 0.12
    held, legs, _parts, path = _run_sweep(0.017, blocked_below=level)
    free, legs_free, _p2, _q2 = _run_sweep(0.017)
    assert held.n_reached < free.n_reached and len(legs) >= len(legs_free) - 1  # no leg waits out its whole budget
    assert np.ptp(path[-120:, 0]) > 0.1 or np.ptp(path[-120:, 1]) > 0.1  # still moving in the second half


def test_a_joint_near_its_limit_sends_the_hand_back_towards_its_start():
    home = [0.0, -0.16, 0.0, -2.44, 0.0, 2.23, 0.79]  # the arm's start: far from every limit
    assert not SW.near_limit(home, SW.SWEEP["limit_margin"])
    assert SW.near_limit([0.0, 1.2, 0.0, -0.10, 0.0, 1.0, 0.8], 0.25)   # the elbow straight: stretched out
    assert SW.near_limit([0.0, 0.3, 0.0, -2.90, 0.0, 2.0, 0.8], 0.25)   # the elbow folded up
    assert SW.near_limit([0.0, 0.3, 0.0, -2.00, 0.0, 3.60, 0.8], 0.25)  # the wrist bent to its end
    assert not SW.near_limit([0.0, 0.3, 0.0, -2.00, 0.0, 3.40, 0.8], 0.25)
    p0, shoulder, anchors = _scene()
    sw, eef = SW.Sweep("item|1", p0, shoulder), p0.copy()
    for _ in range(12):  # an ordinary start: legs as usual
        eef = eef + 0.017 * np.asarray(sw.act(eef, anchors, home)[:3])
    assert sw.n_retreats == 0 and sw.n_legs == 1 and np.linalg.norm(eef - p0) > 0.08
    stretched = [0.0, 1.2, 0.0, -0.10, 0.0, 1.0, 0.8]
    gaps, legs_before = [], sw.n_legs
    for _ in range(40):
        act = sw.act(eef, anchors, stretched if sw.n_retreats == 0 else home)
        eef = eef + 0.017 * np.asarray(act[:3])
        gaps.append(float(np.linalg.norm(eef - p0)))
        if sw.retreat > 0:
            assert act[6] == -1.0 and act[5] == 0.0  # going back: gripper open, no turn
    assert sw.n_retreats == 1 and min(gaps) <= SW.SWEEP["retreat_m"] + 0.02  # back at the start, then on to a new leg
    assert sw.n_legs > legs_before and sw.retreat == 0
    # the same seed without joint readings takes the same legs (the retreat draws nothing)
    a, b = SW.Sweep("item|2", p0, shoulder), SW.Sweep("item|2", p0, shoulder)
    ea, eb, la, lb = p0.copy(), p0.copy(), [], []
    for k in range(200):
        ea = ea + 0.017 * np.asarray(a.act(ea, anchors)[:3])
        eb = eb + 0.017 * np.asarray(b.act(eb, anchors, stretched if 30 <= k < 34 else home)[:3])
        if not la or la[-1] is not a.leg:
            la.append(a.leg)
        if not lb or lb[-1] is not b.leg:
            lb.append(b.leg)
    n = min(len(la), len(lb))
    assert b.n_retreats == 1 and n >= 3 and la[:n] == lb[:n]


# ---------------------------------------------------------------------------------------------- cameras
def _T(pos, yaw_deg=0.0):
    a = math.radians(yaw_deg)
    T = np.eye(4)
    T[:3, :3] = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = pos
    return T


def test_pose_gap_fovy_and_one_camera():
    assert CG.pose_gap(_T([0, 0, 1]), _T([0, 0, 1])) == (0.0, 0.0)
    m, deg = CG.pose_gap(_T([0, 0, 1]), _T([0.03, 0.04, 1], 10.0))
    assert m == pytest.approx(0.05) and deg == pytest.approx(10.0)
    K = CL.intrinsics(45.0, 1024)
    assert CG.fovy_of(K, 1024) == pytest.approx(45.0) and CG.fovy_of(K, 2.0 * K[1][2]) == pytest.approx(45.0)
    one = CG.one_camera([{"fovy_deg": 45.0, "T": _T([0.6, 0, 1.6])}, {"fovy_deg": 45.0, "T": _T([0.6, 0, 1.6 + 1e-9])}])
    assert one["camera"]["fovy_deg"] == 45.0 and one["camera"]["T_world_from_camera"][2][3] == 1.6
    assert one["spread"]["pos_m"] == pytest.approx(1e-9) and one["spread"]["rot_deg"] == 0.0
    with pytest.raises(ValueError, match="different cameras"):
        CG.one_camera([{"fovy_deg": 45.0, "T": _T([0.6, 0, 1.6])}, {"fovy_deg": 45.0, "T": _T([0.6, 0, 1.61])}])
    with pytest.raises(ValueError, match="different cameras"):
        CG.one_camera([{"fovy_deg": 45.0, "T": _T([0, 0, 1])}, {"fovy_deg": 60.0, "T": _T([0, 0, 1])}])
    with pytest.raises(ValueError):
        CG.one_camera([])
    doc = CG.cameras_doc({"libero_long_study": one["camera"]}, {"libero_long_study": {"plain_test_tasks": [5]}}, "2026-11-15")
    assert set(doc) == {"note", "cameras", "measured"} and set(doc["cameras"]["libero_long_study"]) == {"fovy_deg", "T_world_from_camera"}


def test_camera_episodes_are_seen_base_tasks_and_the_plain_test_task():
    todo = CG.camera_episodes(_episodes(), _plain_rows(), CELLS)
    got = {k: [(e["task"], bool(e.get("plain"))) for e in v] for k, v in todo.items()}
    assert got == {"libero_long_living_room": [(0, False), (1, False), (4, False), (6, False)],
                   "libero_long_kitchen": [(2, False), (3, False), (8, True)], "libero_long_study": [(5, True)]}
    eps = [e for v in todo.values() for e in v]
    assert all(e["case"] == "BASE" and e["state"] == 0 and e["seed"] == 142 for e in eps)
    SG.check(eps)  # held-back tasks 7 and 9 are not measured on; the test tasks only in their plain version
    old = CG.old_camera_episodes([{"suite": "libero_goal", "task": t, "case": c, "state": s, "seed": 142 + s, "bddl": "b",
                                   "max_steps": 300, "split": "val" if t in (7, 9) else "train", "chain": None}
                                  for t in (0, 1, 2, 7) for c in ("BASE", "S01") for s in (0, 1)], 2)
    assert [(e["task"], e["case"], e["state"]) for e in old["libero_goal"]] == [(0, "BASE", 0), (1, "BASE", 0)]


def test_stored_cameras_gain_the_tables_and_keep_the_suites(tmp_path, monkeypatch):
    suites = json.loads(CL.CAMERAS_FILE.read_text())["cameras"]
    monkeypatch.setattr(CL, "LONG_CAMERAS_FILE", tmp_path / "none.json")
    assert CL.stored_cameras() == suites  # without the tables' file: exactly as before
    cam = {"fovy_deg": 60.0, "T_world_from_camera": _T([0.7, 0.0, 1.2]).tolist()}
    (tmp_path / "long.json").write_text(json.dumps({"cameras": {"libero_long_study": cam}}))
    monkeypatch.setattr(CL, "LONG_CAMERAS_FILE", tmp_path / "long.json")
    both = CL.stored_cameras()
    assert both["libero_long_study"] == cam and {k: both[k] for k in suites} == suites
    # the camera tool finds a table's camera when the runner hands it the table's key as the suite
    rgb = np.zeros((64, 64, 3), np.uint8)
    look = CL.make_tool({"agentview": rgb}, "libero_long_study", {"colour": {"depth": "sim", "calib": "stored"}},
                        lambda cam_, w, s: {"depth_u16": np.ones((w, w), np.uint16), "K": None, "T_world_from_camera": None})
    fr = look("agentview", 64, 0)
    assert np.allclose(fr["T_world_from_camera"], cam["T_world_from_camera"]) and np.allclose(fr["K"], CL.intrinsics(60.0, 64))
    (tmp_path / "long.json").write_text(json.dumps({"cameras": {"libero_goal": cam}}))
    with pytest.raises(ValueError, match="repeats keys"):
        CL.stored_cameras()


def test_the_committed_tables_cameras_cover_every_cell_the_runner_knows():
    if not CL.LONG_CAMERAS_FILE.is_file():
        pytest.skip("colour_cameras_long.json is not measured yet")
    doc = json.loads(CL.LONG_CAMERAS_FILE.read_text())
    assert set(doc["cameras"]) == set(LP.CELLS)
    cams = CL.stored_cameras()
    for cell in LP.CELLS:
        T = np.asarray(cams[cell]["T_world_from_camera"])
        assert T.shape == (4, 4) and np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9) and np.allclose(T[3], [0, 0, 0, 1])
        assert 20.0 < cams[cell]["fovy_deg"] < 90.0
        m = doc["measured"][cell]
        assert not set(m["training_tasks"]) & {5, 7, 8, 9} and set(m["plain_test_tasks"]) <= {5, 8}
    # the suites' file is the frozen one
    import hashlib

    assert hashlib.sha256(CL.CAMERAS_FILE.read_bytes()).hexdigest().startswith("850ef9b6")


def test_colour_line_parameters_for_the_long_suite(tmp_path, monkeypatch, capsys):
    from rowant_lab.pro2 import long_dev as LD

    adapter = json.loads((Path(LD.__file__).parent / "colour_adapter_params.json").read_text())
    p = LD.colour_long_params(MC.REGISTERED["net"])
    assert p["line"] == "colour" and p["colour"] == {**adapter["colour"], "net": MC.REGISTERED["net"],
                                                     "net2": LD.NET2}
    assert p["colour"]["calib"] == "arm" and p["arm_joints"] == adapter["arm_joints"]  # the adapter as decided
    assert p["lfb_library"] == LP.LIB and p["server"]["lfb_library"] == LP.LIB and p["sequence"] is True
    assert set(LP.CELLS) <= set(p["server"]["routes"]) and set(adapter["server"]["routes"]) <= set(p["server"]["routes"])
    # the camera's own changes (2026-11-15): the picture's zoom and shift undone, then the second net's camera fit
    assert p["server"]["arm_cam"] == {**(adapter["server"].get("arm_cam") or {}), "second": {}} and p["arm_image"] == {}
    assert p["robot_dir"] == adapter["robot_dir"]
    assert adapter["colour"]["net"] != p["colour"]["net"]  # the adapter's own file is not changed
    assert LD.main(["params", "--colour", "/x/net.pt"]) == 0
    assert json.loads(capsys.readouterr().out)["colour"]["net"] == "/x/net.pt"
    # a run folder for the colour line; a perturbed test-task episode is refused before anything is written
    monkeypatch.setattr(LD, "ROOT", tmp_path)
    (tmp_path / "eps.json").write_text(json.dumps(_episodes()))
    assert LD.main(["dev", "--episodes", str(tmp_path / "eps.json"), "--name", "lc8", "--colour", "/x/net.pt"]) == 0
    assert json.loads((tmp_path / "lc8" / "params.json").read_text())["colour"]["net"] == "/x/net.pt"
    (tmp_path / "eps.json").write_text(json.dumps(_episodes() + [_ep(5, "S01")]))
    with pytest.raises(SG.Refused):
        LD.main(["dev", "--episodes", str(tmp_path / "eps.json"), "--name", "bad", "--colour", "/x/net.pt"])
    assert not (tmp_path / "bad").exists()


# ---------------------------------------------------------------------------------------------- the acceptance check
def _score(p90, within, table, frames=80):
    return {"blob_offset": {"median_mm": 1.0, "p90_mm": p90, "n": 100}, "obj": {"median_mm": 1.0, "within_10mm": within},
            "table": {"median_mm": table}, "frames": frames}


def _both():
    old = {"libero_goal": _score(7.0, 0.90, 0.6), "libero_object": _score(5.0, 0.92, 0.5), "libero_spatial": _score(6.0, 0.91, 0.7),
           "libero_long_kitchen": _score(80.0, 0.1, 30.0), "libero_long_living_room": _score(300.0, 0.0, 200.0),
           "libero_long_study": _score(250.0, 0.0, 150.0)}
    new = {"libero_goal": _score(7.5, 0.89, 0.8), "libero_object": _score(5.9, 0.91, 0.6), "libero_spatial": _score(6.0, 0.91, 0.7),
           "libero_long_kitchen": _score(9.0, 0.86, 1.2), "libero_long_living_room": _score(6.0, 0.9, 0.7),
           "libero_long_study": _score(4.0, 0.95, 0.5)}
    return new, old


def test_accept_passes_a_net_that_keeps_the_suites_and_learns_the_tables():
    new, old = _both()
    res = CG.accept(new, old)
    assert res["pass"] and all(v["pass"] and v["gate"] for v in res["keys"].values())
    assert res["keys"]["libero_goal"]["kind"] == "old" and res["keys"]["libero_long_study"]["kind"] == "new"
    assert res["keys"]["libero_goal"]["limits"] == {"blob_offset_p90_mm": 8.05, "within_10mm": 0.88, "table_median_mm": 1.1}
    assert res["keys"]["libero_object"]["limits"]["blob_offset_p90_mm"] == 6.0  # 1 mm, the larger of 1 mm and 15%
    assert res["keys"]["libero_long_kitchen"]["limits"] == CG.ACCEPT["new"]
    lines = CG.accept_lines(res)
    assert lines[-1] == "ACCEPT PASS" and len(lines) == 7 and lines[0].startswith("PASS libero_goal (old)")


def test_accept_fails_on_a_regressed_suite_a_poor_table_or_a_missing_key():
    new, old = _both()
    r = CG.accept({**new, "libero_object": _score(6.1, 0.92, 0.5)}, old)
    assert not r["pass"] and not r["keys"]["libero_object"]["pass"] and "blob offset" in r["keys"]["libero_object"]["why"][0]
    assert all(v["pass"] for k, v in r["keys"].items() if k != "libero_object")
    r = CG.accept({**new, "libero_spatial": _score(6.0, 0.88, 0.7)}, old)
    assert not r["pass"] and "within 10 mm" in r["keys"]["libero_spatial"]["why"][0]
    r = CG.accept({**new, "libero_goal": _score(7.0, 0.90, 1.2)}, old)
    assert not r["pass"] and "table median" in r["keys"]["libero_goal"]["why"][0]
    r = CG.accept({**new, "libero_long_kitchen": _score(45.0, 0.5, 5.0)}, old)
    k = r["keys"]["libero_long_kitchen"]
    assert not r["pass"] and len(k["why"]) == 3 and k["gate"] is False  # over the server's gate as well
    r = CG.accept({k: v for k, v in new.items() if k != "libero_long_study"}, old)
    assert not r["pass"] and "no score" in r["keys"]["libero_long_study"]["why"][0]
    r = CG.accept({**new, "libero_long_study": {"table": {"median_mm": 0.5}, "frames": 3}}, old)  # no object blob found
    assert not r["pass"] and "no object blob" in r["keys"]["libero_long_study"]["why"][0]
    r = CG.accept(new, {k: v for k, v in old.items() if k != "libero_goal"})  # nothing to compare a suite with
    assert not r["pass"] and "current net" in r["keys"]["libero_goal"]["why"][0]
    r = CG.accept({k: v for k, v in new.items() if k.startswith("libero_long")}, None)  # the suites must be scored too
    assert not r["pass"] and not r["keys"]["libero_object"]["pass"] and r["keys"]["libero_long_study"]["pass"]
    assert CG.accept_lines(r)[-1] == "ACCEPT FAIL: libero_goal, libero_object, libero_spatial"
    only = CG.accept({"libero_long_study": new["libero_long_study"]}, None, expect=["libero_long_study"])
    assert only["pass"] and list(only["keys"]) == ["libero_long_study"]


def test_accept_marks_a_table_task_by_task():
    new, old = _both()
    good, poor = _score(5.0, 0.93, 0.6), _score(31.0, 0.55, 0.9)
    # the kitchen's validation: plain task 8 from start 8 (a scene it trained on) and held-back task 9 (a scene it never saw)
    new["libero_long_kitchen"] = {**_score(28.0, 0.6, 0.9), "tasks": [8, 9], "by_task": {"8": good, "9": poor}}
    old["libero_long_kitchen"] = {**old["libero_long_kitchen"], "by_task": {"8": _score(90.0, 0.1, 30.0)}}
    r = CG.accept(new, old)
    k = r["keys"]["libero_long_kitchen"]
    assert not r["pass"] and not k["pass"] and k["tasks"]["8"]["pass"] and not k["tasks"]["9"]["pass"]
    assert all(w.startswith("task 9: ") for w in k["why"]) and len(k["why"]) == 2 and k["gate"] is True
    assert k["tasks"]["8"]["old"]["blob_offset_p90_mm"] == 90.0 and k["tasks"]["9"]["old"] is None
    lines = CG.accept_lines(r)
    assert lines[-1] == "ACCEPT FAIL: libero_long_kitchen task 9" and len(lines) == 8  # a line per kitchen task
    assert any(x.startswith("PASS libero_long_kitchen task 8 (new)") for x in lines)
    assert any(x.startswith("FAIL libero_long_kitchen task 9 (new)") for x in lines)
    # every task good: the table passes, whatever its pooled numbers
    new["libero_long_kitchen"]["by_task"]["9"] = good
    assert CG.accept(new, old)["pass"]
    # a task over the server's gate, or without an object blob
    new["libero_long_kitchen"]["by_task"]["9"] = _score(55.0, 0.2, 4.0)
    assert CG.accept(new, old)["keys"]["libero_long_kitchen"]["gate"] is False
    new["libero_long_kitchen"]["by_task"]["9"] = {"table": {"median_mm": 0.5}}
    assert "no object blob" in CG.accept(new, old)["keys"]["libero_long_kitchen"]["why"][0]
    # a suite is marked on its pooled score against the current net's, with or without per-task scores
    new["libero_goal"] = {**new["libero_goal"], "by_task": {"7": poor, "9": poor}}
    assert CG.accept(new, old)["keys"]["libero_goal"]["pass"] and "tasks" not in CG.accept(new, old)["keys"]["libero_goal"]


def test_accept_command_reads_two_score_files(tmp_path, capsys):
    new, old = _both()
    (tmp_path / "new.json").write_text(json.dumps({"net_sha256": "n", "val_frames": 80, "by_key": new}))
    (tmp_path / "old.json").write_text(json.dumps({"net_sha256": "o", "val_frames": 80, "by_key": old}))
    args = ["accept", "--new", str(tmp_path / "new.json"), "--old", str(tmp_path / "old.json"), "--out", str(tmp_path / "a.json")]
    assert CG.main(args) == 0
    res = json.loads((tmp_path / "a.json").read_text())
    assert res["pass"] and res["new_net"] == "n" and res["old_net"] == "o"
    assert "ACCEPT PASS" in capsys.readouterr().out
    new["libero_goal"] = _score(20.0, 0.5, 3.0)
    (tmp_path / "new.json").write_text(json.dumps({"net_sha256": "n", "val_frames": 80, "by_key": new}))
    assert CG.main(args) == 1 and not json.loads((tmp_path / "a.json").read_text())["pass"]


# ---------------------------------------------------------------------------------------------- one net, every key
torch = pytest.importorskip("torch")


def _write_frame(path, rng, shift=0):
    """A 128 px frame looking straight down from 1 m at a floor with a 10 cm block on it."""
    H = W = 128
    f = 0.5 * W / math.tan(math.radians(45.0) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[2, 3] = 1.0
    depth = np.full((H, W), 1.0)
    depth[40 + shift:70 + shift, 40:70] = 0.9
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rgb=(rng.random((H, W, 3)) * 255).astype(np.uint8),
                        depth=np.rint(depth / 1e-4).astype(np.uint16), K=K, T=T, rc=np.uint8(0))


def _frames(tmp_path, counts):
    """``counts``: {(split, key, task): n} -> a frames tree as the frame factory lays it out."""
    rng = np.random.default_rng(7)
    for (split, key, task), n in counts.items():
        for i in range(n):
            _write_frame(tmp_path / "frames" / split / key / f"{task:02d}" / "BASE" / f"r{i}_0000.npz", rng, i % 5)
    return tmp_path / "frames"


def test_keys_balanced_samples_and_the_worst_key(tmp_path):
    from rowant_lab.pro2 import colour_train as CT

    root = tmp_path / "frames" / "train"
    files = [str(root / k / "00" / "BASE" / f"r{i}_0000.npz") for k, n in (("libero_goal", 40), ("libero_long_study", 3))
             for i in range(n)]
    assert CT.key_of(files[0], root) == "libero_goal" and CT.key_of(files[-1], root) == "libero_long_study"
    groups = CT.group_by_key(files, root)
    assert {k: len(v) for k, v in groups.items()} == {"libero_goal": 40, "libero_long_study": 3}
    s = CT.balanced_sample(groups, 21, random.Random(5))
    assert s == CT.balanced_sample(groups, 21, random.Random(5)) and len(s) == 21
    n = {k: sum(1 for f in s if CT.key_of(f, root) == k) for k in groups}
    assert n == {"libero_goal": 11, "libero_long_study": 10}  # equal shares; the first key takes the odd one
    study = [f for f in s if CT.key_of(f, root) == "libero_long_study"]
    assert all(study.count(f) >= 3 for f in groups["libero_long_study"])  # three whole passes of its 3 frames, plus one
    assert len(set(f for f in s if CT.key_of(f, root) == "libero_goal")) == 11  # no repeat where there are enough
    assert CT.balanced_sample({}, 10, random.Random(0)) == [] and CT.balanced_sample({"a": []}, 10, random.Random(0)) == []
    by = {"a": {"blob_offset": {"p90_mm": 4.0}}, "b": {"blob_offset": {"p90_mm": 9.5}}, "c": {"table": {}}}
    assert CT.worst_key(by) == ("b", 9.5) and CT.worst_key({"c": {}}) == (None, None)
    assert [CT.warmup_factor(i, 4) for i in (0, 1, 3, 4, 100)] == [0.25, 0.5, 1.0, 1.0, 1.0]
    assert CT.warmup_factor(0, 0) == 1.0


def test_score_by_key_and_skip_if_current(tmp_path, monkeypatch, capsys):
    from rowant_lab.pro2 import colour_net as CN
    from rowant_lab.pro2 import colour_score as CS
    from rowant_lab.pro2 import colour_train as CT

    frames = _frames(tmp_path, {("val", "libero_goal", 7): 3, ("val", "libero_goal", 9): 2, ("val", "libero_long_study", 5): 2})
    torch.manual_seed(0)
    m = CN.DepthUNet(base=4, stages=3, blocks=1, cap=4)
    net = tmp_path / "net.pt"
    torch.save({"config": m.config, "state": m.state_dict(), "crop": 64}, net)
    val = sorted(str(p) for p in (frames / "val").rglob("*.npz"))
    groups = CT.group_by_key(val, frames / "val")
    assert CS.tasks_by_key(groups, frames / "val") == {"libero_goal": [7, 9], "libero_long_study": [5]}
    out = tmp_path / "score.json"
    monkeypatch.setattr(sys, "argv", ["colour_score.py", "--frames", str(frames), "--net", str(net), "--out", str(out),
                                      "--val-frames", "4", "--by-key", "--skip-if-current"])
    assert CS.main() == 0
    row = json.loads(out.read_text())
    assert "val" not in row and set(row["by_key"]) == {"libero_goal", "libero_long_study"}
    g = row["by_key"]["libero_goal"]
    assert g["tasks"] == [7, 9] and g["files"] == 5 and g["frames"] == 4 and "blob_offset" in g and "table" in g
    # two tasks in a key's validation: each also scored on its own frames
    assert set(g["by_task"]) == {"7", "9"} and g["by_task"]["7"]["frames"] == 3 and g["by_task"]["9"]["frames"] == 2
    assert row["by_key"]["libero_long_study"]["frames"] == 2 and row["net_sha256"] == CN.sha256_file(net)
    assert "by_task" not in row["by_key"]["libero_long_study"] and CS.task_of(val[0], frames / "val") == 7
    capsys.readouterr()
    assert CS.main() == 0 and json.loads(capsys.readouterr().out) == {"current": str(out)}  # already this net's: left alone
    assert CS.is_current(out, row["net_sha256"], 4, True)
    assert not CS.is_current(out, row["net_sha256"], 80, True) and not CS.is_current(out, "other", 4, True)
    assert not CS.is_current(out, row["net_sha256"], 4, False) and not CS.is_current(tmp_path / "missing.json", "x", 4, True)
    # without --by-key the file is the pooled score, as before
    monkeypatch.setattr(sys, "argv", ["colour_score.py", "--frames", str(frames), "--net", str(net), "--out", str(out),
                                      "--val-frames", "4"])
    assert CS.main() == 0
    row = json.loads(out.read_text())
    assert list(row) == ["net", "net_sha256", "crop", "val_frames", "val"] and row["val"]["frames"] == 4


def test_training_from_another_net_balanced_and_scored_by_key(tmp_path, monkeypatch, capsys):
    from rowant_lab.pro2 import colour_net as CN
    from rowant_lab.pro2 import colour_train as CT

    frames = _frames(tmp_path, {("train", "libero_goal", 0): 6, ("train", "libero_long_study", 5): 2,
                                ("val", "libero_goal", 7): 2, ("val", "libero_long_study", 5): 2})
    torch.manual_seed(1)
    first = CN.DepthUNet(base=4)
    init = tmp_path / "first.pt"
    torch.save({"config": first.config, "state": first.state_dict(), "epoch": 19, "crop": 64}, init)
    out = tmp_path / "net"
    base = ["colour_train.py", "--frames", str(frames), "--out", str(out), "--crop", "64", "--batch", "2",
            "--steps-per-epoch", "2", "--workers", "0", "--base", "4", "--val-frames", "2", "--save-every-s", "1e9"]
    monkeypatch.setattr(sys, "argv", base + ["--hours", "1e-9", "--init", str(init)])  # no time: nothing trained
    assert CT.main() == 0 and not (out / "last.pt").exists()
    monkeypatch.setattr(sys, "argv", base + ["--hours", "0.0003", "--init", str(init), "--balance", "--by-key",
                                             "--warmup-steps", "3", "--lr", "1e-4"])
    assert CT.main() == 0
    text = capsys.readouterr().out
    assert '"train_by_key": {"libero_goal": 6, "libero_long_study": 2}' in text and CN.sha256_file(init) in text
    rows = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    assert rows and rows[0]["steps"] == 2  # 4 crops an epoch: two from each key, whatever their frame counts
    assert set(rows[0]["val"]["by_key"]) == {"libero_goal", "libero_long_study"} and "worst" in rows[0]["val"]
    assert rows[0]["lr"] < 1e-4 * 0.9  # the first epoch ends inside the warm-up
    ck = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
    assert ck["init"] == {"path": str(init), "sha256": CN.sha256_file(init), "epoch": 19} and ck["config"] == first.config
    assert CN.DepthNet(out / "last.pt").crop == 64  # the robot's loader reads it
    # after a restart the driver passes --resume as well: it wins, and the checkpoint still names where the run began
    n = len(rows)
    monkeypatch.setattr(sys, "argv", base + ["--hours", "0.0006", "--init", str(tmp_path / "gone.pt"), "--balance",
                                             "--by-key", "--resume", str(out / "last.pt")])
    assert CT.main() == 0
    rows = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    assert len(rows) > n and rows[n]["epoch"] == rows[n - 1]["epoch"] + 1
    assert torch.load(out / "last.pt", map_location="cpu", weights_only=True)["init"]["epoch"] == 19
    # a net of another form cannot be the start
    other = CN.DepthUNet(base=8)
    torch.save({"config": other.config, "state": other.state_dict()}, tmp_path / "other.pt")
    monkeypatch.setattr(sys, "argv", base + ["--out", str(tmp_path / "net2"), "--hours", "0.0003", "--init", str(tmp_path / "other.pt")])
    assert CT.main() == 2


def test_check_cameras_compares_the_stored_camera_with_the_frames(tmp_path, monkeypatch, capsys):
    frames = _frames(tmp_path, {("train", "libero_long_study", 5): 2, ("val", "libero_long_study", 5): 1})
    z = np.load(next((frames / "train").rglob("*.npz")))
    cam = {"fovy_deg": 45.0, "T_world_from_camera": z["T"].tolist()}
    (tmp_path / "long.json").write_text(json.dumps({"cameras": {"libero_long_study": cam}}))
    monkeypatch.setattr(CL, "LONG_CAMERAS_FILE", tmp_path / "long.json")
    assert CG.main(["check-cameras", "--frames", str(frames), "--out", str(tmp_path / "c.json")]) == 0
    res = json.loads((tmp_path / "c.json").read_text())
    assert res["pass"] and res["keys"]["libero_long_study"]["frames"] == 3 and res["keys"]["libero_long_study"]["pos_m"] == 0.0
    cam["T_world_from_camera"][0][3] += 0.01  # the stored camera 1 cm off the frames' own
    (tmp_path / "long.json").write_text(json.dumps({"cameras": {"libero_long_study": cam}}))
    assert CG.main(["check-cameras", "--frames", str(frames)]) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["bad"] == ["libero_long_study"]
    (tmp_path / "long.json").write_text(json.dumps({"cameras": {}}))  # frames of a key with no stored camera
    assert CG.main(["check-cameras", "--frames", str(frames)]) == 1
    assert CG.main(["check-cameras", "--frames", str(tmp_path / "nothing")]) == 1


def test_the_bundle_is_the_candidate_that_ran_and_leaves_the_default_alone():
    """``long_dev.bundle``: the options that ran together on the untouched half (less the command watch), on top of
    the configuration, which itself stays as it was."""
    from rowant_lab.pro2 import long_dev as LD

    net = MC.REGISTERED["net"]
    base = LD.colour_long_params(net)
    before = json.dumps(base, sort_keys=True)
    b = LD.bundle(base)
    assert json.dumps(base, sort_keys=True) == before  # the default is not touched
    assert "cmd_watch" not in b["delay_comp"] and b["delay_comp"]["unlock"] is True
    assert b["m1"]["agent"]["carry_watch"] is True and b["m1"]["agent"]["rest_m"] == 0.008  # the default's own stay
    assert b["m1"]["agent"]["tell_open"] is True and b["m1"]["agent"]["release_here"] is True
    assert b["relation_single"] is True and b["sequence_reserve"] == 130
    assert b["server"]["support"]["twin_prior"] == {"inward": {"suites": ["libero_goal", "libero_spatial"]}}
    assert b["server"]["arm_cam"]["second"] == {} and b["colour"]["net2"] == LD.NET2
    assert b["m1"]["agent"]["clear_cover"] is True and b["settle_first"] == {}
    assert b["take_by_open"] is True and b["take_near_rim"] is True
    assert b["m1"]["agent"]["side_wide_k"] == 200
    assert b["m1"]["agent"]["c3_measured"] is True and b["m1"]["agent"]["detour_from_cross"] is True
    # v7 (the night of 11-16): each addition as its own paired run had it
    assert b["arm_joints"] == {"guard_mm": 20.0, "readings_fit": {}} and b["arm_image"] == {"last_resort": True}
    assert b["delay_comp"]["caught_elbow"] is True and b["delay_comp"]["unlock_back"] is True
    assert b["m1"]["agent"]["share_keep_up_m"] == 0.04 and b["m1"]["agent"]["held_tilt_max_deg"] == 10
    assert b["m1"]["agent"]["drop_rigid"] is True and b["m1"]["agent"]["c1_stand"] is True
    assert all(b["m1"]["agent"][k] is True for k in ("open_relook", "open_gap_check", "open_settle"))
    assert b["behind_occluder"] == {"min_px": 25000, "standing_max_px": 20000, "share": 0.25, "untold": True}
    assert b["relation_look"] is True and b["relation_standin_strict"] is True and "keep_target" not in b
    assert b["fixture_face_in"] == "carcass" and b["fallback_stand"] == {}
    assert b["fixture_memory"]["libero_goal"]["wooden_cabinet"]["fit_first_m"] == 0.04
    # v8 (the morning of 11-17): the drawer's pull
    assert all(b["m1"]["agent"][k] is True for k in ("open_pull_straight", "open_pull_full", "open_slip_check",
                                                      "open_verify"))
    # v9 (11-17): each addition after its own paired run on top of v8
    ag = b["m1"]["agent"]
    assert ag["transit_unexplained"] is True and ag["c1_soft"] is True
    assert ag["open_pre_pass"] is True and ag["open_box"] is True and ag["open_close_early_m"] == 0.012
    assert ag["open_release_early"] == 2 and ag["open_retreat_pass_m"] == 0.03
    assert ag["stage_release_early"] == 3 and ag["take_turn_early"] is True and ag["take_release_early"] == 2
    assert ag["take_down_lat_m"] == 0.012 and ag["take_close_dz_m"] == 0.025
    assert ag["take_hold_quick"] is True and ag["take_lift_carry"] is True
    assert all(ag[k] is True for k in ("lag_lift_no_settle", "lag_settle_pre", "open_reopen_cmd"))
    assert all(b["delay_comp"][k] is True for k in ("close_hold_cmd", "grip_as_sent", "grip_echo"))
    assert b["fixture_turn_max_deg"] == 45 and b["sequence_chain"] is True and b["overhang_open_need"] == 85
    assert b["misread_target"] == {} and b["fallback_own"] is True
    for left_out in ("carry_relook", "carry_block_steps", "held_pose_fix", "cover_touch_descent", "open_feel",
                     "open_feel_first", "cover_pick", "cover_see", "open_out_first"):
        assert left_out not in ag  # run and not kept, or not run
    assert "probe_grip" not in b["delay_comp"] and "probe_cmd" not in b["delay_comp"]
    assert "resize" not in b["fallback_stand"] and "stage_nudge" not in b and "extra_on_top_m" not in b["server"]
    assert "size" not in b["fixture_memory"]["libero_goal"]["wooden_cabinet"]
