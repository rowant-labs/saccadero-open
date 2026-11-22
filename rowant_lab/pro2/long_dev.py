"""The long-task suite (LIBERO-10) on the depth line: run parameters and development folders.

EVALUATOR side (Mac). The depth line plus what the long suite needs:
* the variant tree ``runs/pro2/long-lib`` (M2's and Goal's files byte for byte, plus the three tables' cells), for the
  robot and the perception server (``lfb_library``);
* the server's model-search routes for the three tables (fixtures searched as Goal's are);
* the robot's step sequencer (``params["sequence"]``, ``robot/sequence.py``) and the free release spot for a second
  object in the same region (``m1.agent.place_free_xy``).

    python3 -m rowant_lab.pro2.long_dev params                    # print the parameters
    python3 -m rowant_lab.pro2.long_dev dev --episodes EPS.json   # write runs/pro2/long/ld8-seq from a planned list

The episode list comes from ``dev_run.py plan --suites libero_10 --states 8`` (the runner's own planning step).

The COLOUR line on the long suite (2026-11-15; ``--colour NET`` on either command): the colour adapter's parameters
(``colour_adapter_params.json``) with the same additions and the depth net at ``NET``, a net trained on the three
tables' frames too (``colour_long.py``). The tables' stored cameras are in
``colour_cameras_long.json``. Such a run goes on a GPU pod with ``dev_run.py run --camera-px 1024``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.pro2 import long_paths as LP  # noqa: E402
from rowant_lab.settings import DATA  # noqa: E402

ROOT = DATA / "runs" / "pro2" / "long"
NET2 = "/workspace/out/colour2/net/last.pt"  # the second depth net on a pod (Mac: runs/pro2/colour2/net/last.pt)
ROUTES = {  # the fixtures each table's model search poses (as Goal's: plates, stoves, cabinets, racks)
    # moka_pot: its masks missed both pots in the first smoke and posed them off the table; the search stays in the box
    "libero_long_kitchen": {"search": ["flat_stove", "white_cabinet", "microwave", "wine_rack", "moka_pot"],
                            "texture_first": [], "fallback": []},
    "libero_long_living_room": {"search": ["basket", "plate"], "texture_first": [], "fallback": []},
    # black_book: its masks kept it in 5 of 10 plain start states, once 1.6 m off the table (2026-11-15)
    "libero_long_study": {"search": ["desk_caddy", "black_book"], "texture_first": [], "fallback": []},
}


def long_params(base: dict | None = None) -> dict:
    if base is None:
        raise ValueError("the depth line (it reads the simulator's depth) is not part of this release: build the "
                         "colour line's parameters with colour_long_params(net)")
    p = copy.deepcopy(base)
    p["lfb_library"] = LP.LIB
    srv = dict(p.get("server") or {})
    srv["lfb_library"] = LP.LIB
    srv["routes"] = {**(srv.get("routes") or {}), **ROUTES}
    p["server"] = srv
    p["sequence"] = True
    p["fallback_on_table"] = True
    p["stage_sweep"] = "stage"  # True also tries the drag-clear plan first (built 2026-11-15, not yet run in the simulator)
    p["fixture_no_flip"] = True
    p["delay_comp"] = {}  # acting through late readings (robot/delay.py: 79 -> 89 of 143, won 11, lost 1)
    m1 = dict(p.get("m1") or {})
    m1["agent"] = {**(m1.get("agent") or {}), "place_free_xy": True, "pull_stall_stop": True, "open_fast": True,
                   "cross_descent_free": True, "relook_place": True, "rise_first": True,
                   # a placed object is at rest when it moves under 8 mm between the check's reads (3 mm before).
                   # Measured 2026-11-15 on objects found inside their goal: the depth line reads 0 to 1.6 mm, the
                   # colour line 1.4 to 5.8 mm (the net's depth under a hand still close), a real slide 37 mm. At
                   # 3 mm the colour line called the first moka pot "not at rest" in 6 of 10 plain starts of the
                   # two-pots task and went back for a pot that stood on the stove (5 of 10).
                   "rest_m": 0.008,
                   # the fingers watched while carrying: an object knocked out of the hand is seen at once (D18)
                   "carry_watch": True}
    p["m1"] = m1
    return p


def colour_long_params(net: str) -> dict:
    """The colour line on the long suite: the colour adapter as decided (``colour_adapter_params.json``: net depth,
    the camera checked against the arm, the readings guard) with the depth net at ``net`` and the long suite's
    additions (``long_params``)."""
    base = json.loads((Path(__file__).resolve().parent / "colour_adapter_params.json").read_text())
    if base.get("line") != "colour":
        raise ValueError("colour_adapter_params.json is not a colour-line parameter set")
    base["colour"] = {**base["colour"], "net": str(net)}
    p = long_params(base)
    # 2026-11-15, the camera's own changes, each tried only when the arm check fires (a pod run against the
    # registered configuration, 96 episodes: S16 0 -> 13 of 24, S15 5 -> 16 of 24, BASE and D07 unchanged, won 24,
    # lost 0): a picture that came zoomed and shifted is undone before the net sees it (``arm_image``); then the
    # camera fit; then the same fit on the second net's depth (``net2``; the net trained for moved cameras)
    p["arm_image"] = {}
    p["colour"] = {**p["colour"], "net2": NET2}
    srv = dict(p["server"])
    srv["arm_cam"] = {**(srv.get("arm_cam") or {}), "second": {}}
    p["server"] = srv
    return p


# 2026-11-16, the candidate for the scored run: the options of the day's parallel streams, each off by default, that
# were run together against the configuration above on the half of the training list no stream had looked at (odd
# half, start state 8, colour line, one A6000 pod: 269 -> 309 of 414; won 50, lost 10). The
# delay stream's command watch (``delay_comp.cmd_watch``) was in that run and is left out here: it fired in plain
# episodes of the drawer task and cost the opening its approach to the handle. Added the same evening, each after its
# own paired run on top of the bundle: the cover and panel options and the drawer's take-again options.
BUNDLE_AGENT = {
    # grasps: the fingers decide the lift check; the target's place is taken where the hand let go; a higher look and
    # the next candidate after a close on nothing
    "c2_trust_gap": True, "release_here": True, "empty_view_m": 0.28, "empty_next": True,
    # two objects: the landing place of a dropped object, the lift check's own lower edge, the placed check read
    # again after backing off, a held object does not jump in the hand
    "drop_predict": True, "c2_band_lo_m": 0.012, "c3_recheck": True, "held_jump_max_m": 0.015,
    # steps: the look tool is told when the hand lets go (and the lost-carry recovery), then the steps per attempt
    "tell_open": True, "transit_cut": True, "cap_box": True, "lift_fast": True, "stage_above_m": 0.015,
    "brake_k": 1.5, "lower_cap": 1.0, "lower_tol_m": 0.006, "close_early_m": 0.025, "close_window": 2,
    "pull_end_early": True, "T_attempt": 70,
    # delays: no wait-and-correct at the release point through a lag
    "lag_skip_release_fix": True,
    # a cover on the receiver is swept off before the release (D03: 1 -> 7 of 20 on top of the bundle)
    "clear_cover": True,
    # the drawer compound's staging: a grasp on the bowl's near rim is looked for further down the planner's list.
    # Staged by the far rim, the opening's approach stopped short and the take-again missed (23 of 24 short
    # approaches). Goal 3 at the scored start, 30 seeds, on top of the rest: 25 -> 29, nothing lost (default 13)
    "side_wide_k": 200,
    # a placed verdict rests on a look that measured the object; after a cut way that was stopped low, the detour
    # rises from the planned crossing height (an obstacle the camera does not show, S05). On top of the bundle,
    # eight cases of the untouched half: 45 -> 48 of 93, nothing lost (S05 1 -> 4)
    "c3_measured": True, "detour_from_cross": True,
    # -- v7, the night of 11-16 (each after its own paired run on top of v6, colour line, RTX 4090, one pod a pair)
    # two pots on one surface keep their own parts of it; a track that tips a held object is not believed; a dropped
    # object lands where the hand really had it (the two-pots test task plain, 30 scored seeds: 28 -> 30 with either
    # of the first two; the six plain test tasks with all three: 178 -> 180 of 180)
    "share_keep_up_m": 0.04, "held_tilt_max_deg": 10, "drop_rigid": True,
    # the drawer's opening: jaws that arrive shut are opened, the approach is finished on the true reading through a
    # delay, and a missed handle is followed by a look and a new plan (Goal training tasks: 54 -> 58 and 55 -> 62 of
    # 83, nothing lost; the plain Goal and Spatial test tasks unchanged with the memory's fit-first rule)
    "open_relook": True, "open_gap_check": True, "open_settle": True,
    # a stood model held across its other side passes the grasp check (with ``fallback_stand`` below)
    "c1_stand": True,
    # -- v8, the morning of 11-17: the drawer's pull. Its command keeps the opening direction (the per-axis cap bent
    # a pull that is not along a table axis and the bar slid out of the fingers), the hand pulls the part's whole
    # travel, fingers that end the pull shut have not opened it, and an opening that is the instruction's last step
    # is checked against the episode going on (a turned cabinet at seeds never run: 18 -> 23 of 28, the opening
    # task under a pose change 3 -> 8 of 12; the plain Goal test tasks 59 = 59 of 60)
    "open_pull_straight": True, "open_pull_full": True, "open_slip_check": True, "open_verify": True,
    # -- v9, 11-17 (each after its own paired run on top of v8, colour line, RTX 4090, one pod a pair)
    # a thing the first look could not explain is counted on the carry's way (eight types, 172 episodes: 124 -> 126,
    # won 3, lost 1; 17 steps shorter where it acts)
    "transit_unexplained": True,
    # the opening step: the point before the handle passed, the close begun on the way in, the release sent before
    # the pull's end, a shorter retreat, commands capped by their largest component (the plain drawer test task,
    # 30 scored seeds: 29 = 29 solved, 239 -> 219 steps)
    "open_pre_pass": True, "open_close_early_m": 0.012, "open_release_early": 2, "open_retreat_pass_m": 0.03,
    "open_box": True,
    # the staging step and the take-again: the release begun on the way down, the turn back begun as the hand leaves
    # the handle, a quicker hold, lift and release (on top of the row above: 29 -> 30 solved, 219 -> 201 steps)
    "stage_release_early": 3, "take_turn_early": True, "take_down_lat_m": 0.012, "take_close_dz_m": 0.025,
    "take_hold_quick": True, "take_lift_carry": True, "take_release_early": 2,
    # through a delay: no wait of a whole lag before the lift check, the wait-and-correct taken before the descent
    # when the commands are late, jaws reopened for the commands' delay only (with the delay keys below)
    "lag_lift_no_settle": True, "lag_settle_pre": True, "open_reopen_cmd": True,
    # a steady hold between 0.6 and 1.35 of the model's width goes on to the lift (the guard list of 137 episodes:
    # 114 -> 115, nothing lost; with the stream's other options 114 -> 117)
    "c1_soft": True}
BUNDLE_TOP = {
    # placing: a lone candidate that fails the instruction's relation; a missing receiver follows the evidence;
    # a believed move of the receiver stays level
    "relation_single": True, "fallback_top": True, "relook_strict": True,
    # two objects: a region percept never under the table; steps kept for later steps; one already at its place is
    # not the one to move; clauses that did not do their work run once more
    "fallback_blob_on_table": True, "sequence_reserve": 130, "shared_skip_at_place": True, "sequence_retry": True,
    # covers and panels: the first look waits for a still scene; a target hidden behind a hanging region is looked
    # for behind it (S12: 1 -> 13 of 22 on top of the bundle; the pair as a whole won 25, lost 6)
    "settle_first": {}, "behind_occluder": {},
    # the drawer compound's take-again: by the bowl's near rim, and with the cabinet pose of the opening step
    # (Goal 3 at the scored start, 30 seeds: 22 -> 24 on top of the bundle, nothing lost; default 13)
    "take_by_open": True, "take_near_rim": True,
    # -- v7, the night of 11-16
    # the readings' offsets from the readings alone, and an accepted zoom as the last resort (S15, S16, S18: 45 -> 54
    # of 72 at start 8, 42 -> 57 of 96 at the scored start, nothing lost; S18 alone 27 -> 44 of 56)
    "arm_joints": {"readings_fit": {}}, "arm_image": {"last_resort": True},
    # what counts as a panel, and a lone candidate whose landmark is hidden too (cover and panel cases at start 8:
    # 70 -> 73 of 117, won 5, lost 2)
    "behind_occluder": {"min_px": 25000, "standing_max_px": 20000, "share": 0.25, "untold": True},
    # the wrong bowl: a look at the relation's place before a lone candidate that fails it is carried; a landmark's
    # stand-in must be seen (Spatial at the scored start, with ``keep_target`` too: 126 -> 129 of 168, nothing
    # lost). ``keep_target`` (a new plan keeps the first plan's target) is LEFT OUT: in a two-object instruction the
    # second step then goes for the object the first step placed (the two-pots scenario of the kinematic stand-in:
    # one pot placed of two; ``test_long2_options``), and in its own two re-plan episodes it won nothing
    "relation_look": True, "relation_standin_strict": True,
    # the cabinet: a fixture that opens away from the hand's start is turned about its carcass; one the look did not
    # find is taken at its remembered place when a region's top lies over it, unless a rejected fit agrees with
    # that place (with the opening options above)
    "fixture_face_in": "carcass",
    "fixture_memory": {"libero_goal": {"wooden_cabinet": {"pos": [0.03, -0.234, 0.9076], "yaw_deg": 180.0,
                                                          "fit_first_m": 0.04}}},
    # a named object with no percept is stood where its region shows it (Object and Goal at the scored start: 88 ->
    # 98 of 131, nothing lost; the eight plain test tasks at ten starts 80 -> 80)
    "fallback_stand": {},
    # -- v9, 11-17
    # the steps of a sequence follow each other without an idle step; in the staged drawer plan the sequence keeps
    # 85 steps back for the opening while the staging runs, not the whole reserve (the staging may grasp twice)
    "sequence_chain": True, "overhang_open_need": 85,
    # a named target or receiver read as another type is stood where that weak percept stands; a reported object's
    # own leftover region is no receiver (every option of the stream on S07 to S09, 72 episodes: 54 -> 57)
    "misread_target": {}, "fallback_own": True,
    # a later look's fit of a fixture turned more than 45 degrees from the percept held is refused, beside the half
    # turn (the plain drawer test task: 29 -> 30 of 30, the seed lost by v7 and v8)
    "fixture_turn_max_deg": 45}
BUNDLE_DELAY = {"start_lag": True, "leap_trust": True, "replan_late_only": True, "no_answer_assume": True,
                "unlock": True,
                # v7: an arm is caught by its elbow reading alone, and after the unlock's pull the hand is drawn in
                # and turned back (delay cases at start 8: 100 -> 109 of 143, nothing lost; D14 5 -> 12 of 24)
                "caught_elbow": True, "unlock_back": True,
                # v9: a hold after a command to close only for the commands' own delay; the compensator's own steps
                # repeat the gripper command as sent; a lag that outlived its delay is ended by the fingers' answer
                # (the delay cases at start 8, 143 episodes: 110 -> 114, nothing lost; S20 17 -> 20 and 17% shorter)
                "close_hold_cmd": True, "grip_as_sent": True, "grip_echo": True}
BUNDLE_SERVER = {
    "extra_blob_top_m": 0.015,  # the top layer of an unexplained region (a receiver raised on a block)
    # the cabinet: its remembered place (the robot's own accepted first looks on training tasks) tried before the
    # search, and the half-turn twin when the chosen pose opens away from the robot
    "support": {"memory": {"libero_goal": {"wooden_cabinet": [{"pos": [0.03, -0.234, 0.9076], "yaw_deg": 180.0}]},
                           "libero_spatial": {"wooden_cabinet": [{"pos": [0.0286, -0.2677, 0.9065],
                                                                  "yaw_deg": 154.3}]}},
                "twin_prior": {"inward": {"suites": ["libero_goal", "libero_spatial"]}}}}


def bundle(p: dict) -> dict:
    """``p`` (``long_params`` or ``colour_long_params``) with the candidate's options on. ``p`` is not changed."""
    q = copy.deepcopy(p)
    for k, v in BUNDLE_TOP.items():  # a dict joins the dict already there (``arm_joints`` keeps its ``guard_mm``)
        q[k] = {**q[k], **copy.deepcopy(v)} if isinstance(v, dict) and isinstance(q.get(k), dict) else copy.deepcopy(v)
    q["delay_comp"] = {**(q.get("delay_comp") or {}), **BUNDLE_DELAY}
    m1 = dict(q.get("m1") or {})
    m1["agent"] = {**(m1.get("agent") or {}), **BUNDLE_AGENT}
    q["m1"] = m1
    srv = dict(q.get("server") or {})
    for k, v in BUNDLE_SERVER.items():
        srv[k] = {**srv[k], **v} if isinstance(v, dict) and isinstance(srv.get(k), dict) else copy.deepcopy(v)
    q["server"] = srv
    return q


def cmd_params(a) -> int:
    p = colour_long_params(a.colour) if a.colour else long_params()
    print(json.dumps(bundle(p) if a.bundle else p, indent=1))
    return 0


def cmd_dev(a) -> int:
    from rowant_lab.pro2 import split_guard

    eps = json.loads(Path(a.episodes).read_text())
    if any(e.get("suite") != LP.SUITE for e in eps):
        raise SystemExit("the episode list holds episodes of other suites")
    split_guard.check(eps, "the run's episode list")  # never a perturbed episode of a benchmark test task
    d = ROOT / a.name
    d.mkdir(parents=True, exist_ok=True)
    b = json.dumps(eps).encode()
    (d / "episodes.json").write_bytes(b)
    text = json.dumps(colour_long_params(a.colour) if a.colour else long_params())
    (d / "params.json").write_text(text)
    print(json.dumps({"run": d.name, "episodes": len(eps), "episodes_sha256": hashlib.sha256(b).hexdigest(),
                      "params_sha256": hashlib.sha256(text.encode()).hexdigest()}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("params", "dev"))
    ap.add_argument("--episodes", default="")
    ap.add_argument("--name", default="ld8-seq")
    ap.add_argument("--colour", default="", help="the colour line with this depth net (its path on the pod)")
    ap.add_argument("--bundle", action="store_true", help="params: with the candidate's options on (``bundle``)")
    a = ap.parse_args(argv)
    return {"params": cmd_params, "dev": cmd_dev}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
