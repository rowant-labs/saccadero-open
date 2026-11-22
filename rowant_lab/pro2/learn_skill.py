"""In-episode learning, robot side (the second world's registration, amendment W2-LEARN-EP; off unless
``params["learn_in_episode"]``). The server side is ``rowant_lab.libero_env.lfb_episode``.

**When.** The instruction names a type, and the episode's first full look accepted no percept of it: the robot's memory
does not explain the object it is asked to move. Instead of guessing (the robot's fallbacks: a rejected pose of that
type, or the blob nearest its stored size), it learns the object it cannot explain, then acts.

**How**, all inside the episode:
1. ``learn_plan`` (no step): the robot sends back its first agentview frame, its accepted percepts and the support
   plane, and a wrist frame at the current pose. The server returns the leftover (the largest unexplained object) and
   LF-b's ring of looks as hand poses, with waypoints between them.
2. **Looks, with real arm motion** (every step counts): the hand follows the waypoints and stops at each look under
   LIBERO's own OSC controller (``robot/follow.py``, unchanged), and the wrist camera is read there (the depth line's
   camera tool, as for every look). The looks stop early when fewer than ``ACT_RESERVE`` steps would be left for acting.
3. ``learn`` (no step): masks, TSDF model, ST-1's F7 entry, a variant library, and this episode's server memory.
4. **Acting**: the robot's library becomes the variant, the target is placed in the model's own frame (the model was
   built in this scene's world frame: world axes, origin at the leftover's centre on the plane), and the robot plans
   as usual with the steps left, with F7's acting settings (C1 band lower edge 12 mm, held radius 0.20 m).
If a step fails (no leftover, no reachable look, fewer than ``MIN_LOOKS`` looks, no model), the robot plans as it
would have, with the steps left.

**Check before replacing** (amendment W2-LEARN-EP2; the server option ``learn_check_own``): before building a model, the
server tests the robot's own model of the type against the looks. When it explains them (``lfb_episode.check_own``),
nothing is learned: the robot adds a percept of its own type at the pose the looks gave (``close0``) and plans as usual,
with its own library and settings and the steps left. ``params["learn_looks"] == "fast"`` reads the looks sooner
(``FAST``). ``params["learn_dev_force"]`` is for development only, never a registered run: it starts learning even when
the first look accepted the target, as if it had missed it, to test the check on objects the robot knows.

**The top-down look first** (2026-11-17; ``params["learn_top_first"]`` with the server's option of the same name, off
by default; for the colour line, whose first look smears a thing it has no model of along the main camera's line of
sight): the first look goes above the part of the leftover nearest the main camera; the server finds the thing again
in that look's own depth and plans the two flank looks around it (``lfb_episode.refind``); the robot's top-down finder
(``wrist_find``) is asked for the same thing and its distance from the stereo's place is recorded. The model is built
from the top look and the flanks. ``params["learn_wrist_stereo"]``: the wrist frames come from the runner's own
source, colour with partner pictures and no depth (``wrist_stereo.py``); the server makes their depth.

Policy side: no simulator handle, no file read (the variant is read by ``m2_library``, as at the episode's start).
"""

from __future__ import annotations

import copy
import time

import numpy as np

ACT_RESERVE = 150  # steps kept for acting: form 1's successful acting took a median 115 and a 90th percentile 170
MIN_LOOKS = 2
WAYPOINT_TOL = (0.04, 15.0)  # m, deg: pass a waypoint at this distance
LOOK_TOL = (0.015, 6.0)  # m, deg: read the wrist camera within this of the planned look
WAYPOINT_MAX_STEPS, LOOK_MAX_STEPS = 25, 40
PLATEAU_WINDOW, PLATEAU_GAIN_M, PLATEAU_GAIN_DEG = 8, 0.002, 1.0
F7_AGENT = {"c1_band_lo_m": 0.012}
F7_HELD_NEAR_M = 0.20
WRIST = "robot0_eye_in_hand"
AGENT = "agentview"
# W2-LEARN-EP2 (``params["learn_looks"] == "fast"``): the camera's pose is measured where it stops, so a look need not
# hit its planned pose closely, only see the object. Read a look within 3 cm and 12 degrees (the follower still
# commands its full speed beyond 2.5 cm, so no slow final approach), pass waypoints within 6 cm and 20 degrees, and give
# up on a waypoint after 12 steps (in W2-LEARN-EP's development rows 12 of 118 waypoints used all 25 steps and ended
# farther away than they began)
FAST = {"look_tol": (0.03, 12.0), "waypoint_tol": (0.06, 20.0), "waypoint_max_steps": 12, "look_max_steps": 40}


# The way back (2026-11-18; ``params["learn_way_back"]``, off unless given). Without it an episode the learner started
# in ends on the learned thing: when that thing gives no grasp, or its attempts fail, nothing returns the episode to
# the robot's own handling. With it the learned thing gets ``attempts`` attempts, and when it ends in one of ``after``
# with at least ``min_left`` steps left, the learner's agent takes the robot's own plan from there (``go_back``), made on
# what the first look saw: the same plan the robot acts on when learning fails. No new look is taken: the arm stands
# in the picture then, and a first try with a new look grounded on another thing (Mac, task 65: the grasp closed on
# nothing twice). The row: ``learn.way_back``; with too few steps left, ``learn.way_back_short``. The numbers
# come from round B's pod rows. One attempt: all 21 scenes solved on a learned model were
# solved at the first, and none of 11 later attempts succeeded. 71 steps: the fewest v9 alone needed in a scene it
# solved on a fallback target (22 scenes, median 98).
WAY_BACK = {"attempts": 1, "after": ("no_plan", "max_attempts"), "min_left": 71}
# A late start (``params["learn_late"]``, off unless given): with fewer than ``min_left`` steps left the learner does
# not start, and the robot's own plan is made as ``plan_episode`` alone makes it. The row: ``learn.late``. 209 steps:
# the 150 kept for acting (``ACT_RESERVE``) and the fewest steps the looks took in a scene the learner solved (59).
LATE = {"min_left": 209}
# The robot's own attempt first (2026-11-18; ``params["learn_after_fail"]``, off unless given). The registered run
# W2-LEARN-C failed its benchmark keep rule: on the practice list the learner started in 26 episodes whose thing is a
# known object the first look did not accept, and v9 solves 21 of those at its first attempt. On the second world's
# development part v9's first attempt at a new object has failed by step 29 at the median (50 of 60 episodes; its
# second attempt begins at step 57, with 223 steps left), and none of its 46 second attempts succeeded. With the
# option the robot's own plan is made first wherever it has one (``learn.own_first.pending``). When its first
# ``attempts`` attempts have each ended in one of ``after`` and the next has begun (the hand has let go and backed off
# by then), or the plan has ended in one of ``ended`` without more than ``attempts`` attempts, and at least
# ``min_left`` steps are left, the runner looks again and the learner starts from that look (``fail_due``,
# ``plan_after_fail``; the row: ``learn.own_first`` with the step, the steps left and the attempts' outcomes). With
# too few steps left the robot's own plan goes on (``own_first.short``). Where the robot has no plan of its own the
# learner is due as soon as the runner asks (``why``: ``no_own_plan``). The runner asks only when the arm's camera
# check is not moving the arm and is not due to: a robot that gave up at once, or has no plan, is that check's cue,
# and a camera found moved is looked through again first (Mac, practice Object 6 S16: v9 has no plan at step 0,
# checks its camera, finds it moved and solves the episode; the learner started at step 0 instead and lost it).
# 209 steps: ``LATE``'s.
# ``retreat`` (2026-11-18 evening, 0 = off): before the new look the hand goes back toward where it stood at the
# episode's start, for at most that many steps (done within ``retreat_tol_m``). Without it the look is taken with the
# hand where the failed attempt left it, over the thing: on the Mac a new object was then hidden, the largest
# unexplained thing was another one 55 cm away, and the learner went for that (second world, development scene 12).
# ``min_left`` is asked for when the learner starts, so with a retreat the failed start must leave that many more.
AFTER_FAIL = {"attempts": 1, "min_left": 209,
              "after": ("c1_fail", "c3_fail", "approach_stalled", "unreachable", "no_plan"),
              "ended": ("no_plan", "max_attempts", "no_budget"), "retreat": 0, "retreat_tol_m": 0.04}
# Ask the known types (2026-11-18; ``params["learn_ask_known"]``, off unless given; with ``learn_top_first``): after the
# look from above the server fits the known types the robot names to the thing (``lfb_episode.ASK_KNOWN``). A type
# that explains it becomes a percept of that type (a basket found this way can be placed into), and the next thing in
# the order gets its own look from above. An episode tries at most ``max_things`` things, and another one only while
# ``next_need`` steps of the looks' budget are left: a look from above and the ring after it at their 90th
# percentiles in round B's pod rows (24 and 67 steps; medians 18 and 54). The types asked, never the instruction's own:
# the place's type when no percept of it was accepted, and the types of accepted percepts within ``near_m`` of the
# thing (a part of an object the first look placed a little off stays unexplained: the percept then takes the fitted
# pose, if that lies within ``move_max_m`` of the old one; the six basket parts measured lay 8 to 9 mm off).
# ``rejected`` (off; the measurement's setting): also the types of the first look's rejected poses. Measured on the
# Mac (28 scenes, 2026-11-18) these are not safe to act on: shape alone cannot tell a new object from a known type it
# resembles (one new object was explained by two types that were not in its scene, with numbers as good as a true
# fit's), and a type has a twin of the same shape. The row: ``learn.known`` (each thing explained),
# ``learn.plan2.ask`` (the fits).
# ``own`` (2026-11-18, off): the instruction's own type is asked too, first. The registered run W2-LEARN-C lost three
# of 476 practice episodes: there the thing is a known object the first look did not accept, the learner spent its
# ring of looks (a median 76 steps) to find that out or to learn the object again, and v9's own handling solves 23 of
# those 26 episodes without it. With ``own``, a thing the robot's own model explains from the look from above alone
# becomes a percept of the instruction's type at once and the robot's own plan acts on it (``acting``: ``own_top``;
# the row: ``learn.known`` with the type, ``learn.known_end`` ``"own"``); no ring of looks, no further thing.
# (own is read from the option itself and is not among the defaults below, which round E's arms pin.)
# ``own_near`` (2026-11-18 evening, off; read from the option itself too): the instruction's own type is asked, and
# when the server's floors refuse it but its fit passes the looser ``OWN_NEAR`` floors, the thing is taken for the
# known object seen under a change: the looks end and the robot's own plan is made, on what the first look saw, as
# the robot alone would have made it (``acting``: ``own_near``; the row: ``learn.own_near`` with the fit). Why: the
# registered learner lost practice episodes on known objects that the robot's own handling solves, and "the robot's
# own attempt first" (``AFTER_FAIL``) keeps those but takes about 70 steps and the second thing's looks from the
# learner where the thing is new (a pod run: new objects 45 -> 36 of 60). Where the floors come from (the Mac's
# Step G rows, the own type's fit from above): five known objects seen under a change gave inlier, seen, cover of
# (0.98, 0.83, 0.88), (0.97, 0.83, 0.89), (0.94, 0.49, 0.92), (1.00, 0.70, 1.00) and (0.79, 0.32, 0.77); of 18 new
# objects none has all three above (0.93, 0.45, 0.85): the two with cover 0.99 have inlier 0.90 and 0.66.
# 2026-11-19, after the second registered run (W2-LEARN-D: practice 411 -> 409, the same two episodes
# W2-LEARN-C lost). Both are known objects seen under a change that the robot's own handling solves:
#   - one passed the SERVER's floors on the pod (1.00, 0.69, 0.98; the Mac had read 0.94, 0.49, 0.92), and a yes by
#     those floors acts on the model as fitted from above (``own_top``), which failed there in every run;
#   - one read (0.96, 0.84, 0.67) on the pod (the Mac: 0.97, 0.83, 0.89): a known type in another size, refused on
#     its cover, learned, and out of steps.
# The pod's own rows (the own type's fit from above, 55 new objects and 13 known ones): with inlier at least 0.93, no
# new object has more than 0.62 of the model seen; six of the seven known objects under a change that the robot's
# own handling solves have at least 0.69 (the seventh reads 0.87, 0.34, 0.82 and is learned and solved); cover does
# not tell the groups apart (0.50 to 1.00 on the known ones). Two keys of the option, both off unless given:
# ``own_near["any"]``: a list of floors; the fit passes when it passes all three numbers of any one of them (each
# over ``OWN_NEAR``). ``OWN_NEAR_SEEN`` is the second set read from the pod's rows (its cover: see below).
# ``own_top_min`` (floors on inlier, seen, cover): a thing the server's floors accept as the instruction's own type
# is acted on from above only when its fit also passes these; else it is taken as ``own_near`` is (nothing of the
# look is kept, the robot's own plan is made on what the first look saw; the row: ``learn.own_near`` with
# ``strict`` true). The four fits acted on from above and solved on the pod had 0.87 to 1.00 of the model seen;
# the one that failed, 0.69. The floor on ``seen`` stands midway (0.78); the other two are the server's own.
# The second set's floor on cover (2026-11-19, 04:30): its first check on a pod (with no
# floor on cover) solved both lost episodes, and took two S16 episodes for known objects (1.00, 0.95, 0.50 and 1.00,
# 0.81, 0.50). There the picture comes zoomed until the arm's camera check has undone it; the robot's own plan,
# made on that picture, then needed 184 and 263 steps (v9: 97 and 114; W2-LEARN-D, which went on with its looks
# until the check had ended: 117 and 140). A zoomed picture reads a cover of 0.50; the known type in another size,
# 0.67. The floor stands between them (0.60), so the S16 episodes go on as they did in W2-LEARN-D.
ASK = {"max_things": 3, "next_need": 91, "max_types": 8, "near_m": 0.15, "move_max_m": 0.03, "rejected": False}
OWN_NEAR = {"inlier": 0.93, "seen": 0.45, "cover": 0.85}
OWN_NEAR_SEEN = {"inlier": 0.93, "seen": 0.65, "cover": 0.6}
OWN_TOP_MIN = {"inlier": 0.93, "seen": 0.78, "cover": 0.95}
# The camera check's cue (``params["learn_calib_cue"]``, 2026-11-18 evening, off unless given): when the learner has
# gone on to the robot's own plan and that plan has ended, the learner's agent shows that ending as its own
# (``finish_reason``). The arm's camera check reads it there: a robot that gave up before its camera was checked is
# that check's cue to move the arm by itself (``arm_calib.can_move``). Without it the check never ran once the
# learner had started (Mac, practice Object 6 S16: the look from above was too short a motion for the check, the
# thing was not found again through the shifted camera, the robot's own plan found nothing, and the episode stood).


# A plan made again keeps what was learned (2026-11-18; ``params["learn_keep"]``, off unless given). Round D's five
# "late starts" were learners that started at step 0, built their model and acted on it; the learned thing's first way
# to the grasp ran the arm to the end of its reach (things 0.8 to 0.9 m from the base), v9's caught-arm routine
# fired, the plan was made again, and the new look tool knew nothing of the learned thing. With the option the
# learner's agent goes on with it on the new tool (``keep_learned``), at most ``times`` an episode, and only after a
# caught arm (``delay``): after a camera fit the learned pose is no longer to be trusted. The row: ``learn.kept``.
KEEP = {"times": 1, "after": ("delay",)}


def keep_learned(agent, tool, left: int, params: dict, why: str, t: int) -> dict | None:
    """For the runner, when a plan is made again: the plan on the thing the ending agent had learned, or None (the
    option is absent, the agent is not the learner's or was not acting on a learned thing, it has gone back to
    the robot's own handling, the reason is not one of ``after``, or it was kept ``times`` before)."""
    o = params.get("learn_keep")
    if o is None or not params.get("learn_in_episode") or type(agent).__name__ != "LearnThenAct":
        return None
    o = {**KEEP, **o}
    rec = agent.rec
    if getattr(agent, "inner", None) is None or rec.get("acting") != "learned" or rec.get("way_back") is not None \
            or getattr(agent, "_learned", None) is None or why not in tuple(o["after"]) \
            or len(rec.get("kept") or []) >= int(o["times"]):
        return None
    s = agent.inner.summary() if hasattr(agent.inner, "summary") else {}
    agent._steps_learned = int(getattr(agent, "_steps_learned", 0) or 0) + int(s.get("steps") or 0)
    rec.setdefault("kept", []).append({"t": int(t), "why": why, "left": int(left),
                                       "attempts": [a.get("outcome") for a in s.get("attempts") or []]})
    agent.tool = tool
    agent.inner = agent._act_agent(agent._learned, None, left=int(left))
    if agent.inner is None:
        agent._done, agent.finish_reason = True, agent.finish_reason or "grounding_failed"
    return {"agent": agent, "parsed": {"status": "ok", "target_type": agent.T, "learning": True},
            "grounding": rec.get("grounding") or {"status": "learning"}, "learn": rec}


def keep_first(old, new, t: int):
    """Record only (2026-11-18): when a plan is made again, the learner's record of the plan that ends is kept
    under ``first`` of the new one (the earliest, when plans were made again more than once), with the step it
    ended at. Before, a learner that started at step 0 and was planned again read as one that started late."""
    if not isinstance(new, dict) or not isinstance(old, dict):
        return new
    first = old.get("first") if isinstance(old.get("first"), dict) else (old if old.get("trigger") else None)
    if first is not None and first is not new:
        new["first"] = {**{k: v for k, v in first.items() if k != "first"}, "until": first.get("until", int(t))}
    return new


def way_back(params: dict) -> dict | None:
    """The way back's settings, or None when the option is absent (or the learner is off)."""
    o = params.get("learn_way_back")
    if o is None or not params.get("learn_in_episode"):
        return None
    return {**WAY_BACK, **o}


def back_due(agent, t: int, cap: int, o: dict) -> dict | None:
    """For the runner, after a step: the record of a way back that is due now, or None. Due: the episode's agent is
    the learner's, it acts on a learned thing, that agent has ended in one of ``o["after"]``, at least
    ``o["min_left"]`` of the episode's ``cap`` steps are left at step ``t``, and the episode has not gone back
    before. With too few steps left the learner's record says so once (``way_back_short``)."""
    if type(agent).__name__ != "LearnThenAct" or getattr(agent, "inner", None) is None:
        return None
    rec = agent.rec
    if rec.get("acting") != "learned" or rec.get("way_back") is not None or not agent.finished:
        return None
    s = agent.inner.summary() if hasattr(agent.inner, "summary") else {}
    why = s.get("finish_reason")
    if why not in tuple(o["after"]):
        return None
    doc = {"t": int(t), "why": why, "left": int(cap) - int(t), "min_left": int(o["min_left"]),
           "attempts": [a.get("outcome") for a in s.get("attempts") or []]}
    if doc["left"] < doc["min_left"]:
        rec.setdefault("way_back_short", doc)
        return None
    return doc


def after_fail(params: dict) -> dict | None:
    """The settings of "the robot's own attempt first", or None when the option is absent (or the learner is off)."""
    o = params.get("learn_after_fail")
    if o is None or not params.get("learn_in_episode"):
        return None
    return {**AFTER_FAIL, **o}


def own_first(params: dict, form: dict | None) -> dict | None:
    """Whether THIS start waits for the robot's own attempt (``learn_after_fail``): the settings, or None. As built on
    11-18 every start does. ``learn_forms["own_first"]`` (2026-11-19) narrows that to the starts it names, which are
    starts the option itself made: ``["put"]`` leaves the registered start and the relation's start with the look
    first. (On the practice list the robot solves known objects of the ``put`` tasks at its first attempt in 21 of
    23 episodes, and the learner in their place lost two.)"""
    o = after_fail(params)
    if o is None:
        return None
    f = params.get("learn_forms")
    if isinstance(f, dict) and f.get("own_first") is not None:
        return o if (form is not None and form.get("why") in tuple(f["own_first"])) else None
    return o


def fail_due(agent, rec, t: int, cap: int, o: dict) -> dict | None:
    """For the runner, after a step: the record of the robot's own failed start when the learner is due now, or
    None. Due: the plan was made with the learner waiting (``rec["own_first"]["pending"]``), the agent is the robot's
    own, its first ``o["attempts"]`` attempts each ended in one of ``o["after"]`` and it has begun the next (or it
    has ended in one of ``o["ended"]`` with no attempt that ended otherwise, or the robot had no plan at all:
    ``agent`` None with ``own_first.no_own_plan``), and at least ``o["min_left"]`` of the
    episode's ``cap`` steps are left at step ``t``. With too few steps left the record says so once
    (``own_first.short``) and the robot's own plan goes on."""
    own = rec.get("own_first") if isinstance(rec, dict) else None
    if not isinstance(own, dict) or not own.get("pending") or type(agent).__name__ == "LearnThenAct":
        return None
    if agent is None and not own.get("no_own_plan"):
        return None
    att = list(getattr(agent, "attempts", None) or [])
    cur = getattr(agent, "cur", None)  # a plan in steps (a look, then the take): the running step's attempts too
    if cur is not None and cur is not agent:
        att += list(getattr(cur, "attempts", None) or [])
    outs = [a.get("outcome") for a in att if isinstance(a, dict)]
    n, after = int(o["attempts"]), tuple(o["after"])
    ended = bool(getattr(agent, "finished", False))
    if agent is None:
        why = "no_own_plan"  # the robot had no plan of its own at the start
    elif len(att) > n and all(x in after for x in outs[:n]):
        why = "attempt"
    elif ended and len(att) <= n and all(x in after for x in outs) \
            and (outs or getattr(agent, "finish_reason", None) in tuple(o["ended"])):
        # it has ended, and every attempt it made failed in a way the learner follows (a plan in steps ends as
        # "steps done" whatever its steps did); with no attempt at all, only on one of the named endings
        why = "ended"
    else:
        if ended or (len(att) > n):  # it ended, or went on, in a way the learner does not follow: no longer waiting
            own.update(pending=False, started=False, why="other",
                       attempts=outs[:n + 1], finish_reason=getattr(agent, "finish_reason", None))
        return None
    doc = {"t": int(t), "why": why, "left": int(cap) - int(t), "min_left": int(o["min_left"]),
           "attempts": outs[:n + 1], "target_type": own.get("target_type")}
    if why == "ended":
        doc["finish_reason"] = getattr(agent, "finish_reason", None)
    back = max(0, int(o.get("retreat") or 0))
    if back:
        doc["retreat_max"] = back
    if doc["left"] - back < doc["min_left"]:
        own.update(pending=False, started=False, short=doc)
        return None
    own.update(pending=False, due=doc)
    return doc


def hand_pose(obs: dict) -> tuple:
    """(position, 3x3 rotation) of the hand from the runner's readings."""
    from rowant_lab.pro2.robot.grasp import quat_xyzw_to_mat

    q = obs["robot0_eef_quat_xyzw"] if "robot0_eef_quat_xyzw" in obs else obs["robot0_eef_quat"]
    return np.asarray(obs["robot0_eef_pos"], float), quat_xyzw_to_mat(q)


def retreat_begin(doc: dict, home: tuple, t: int, o: dict) -> dict | None:
    """The retreat's state for the runner (``AFTER_FAIL``), or None when the option asks for none or the start pose
    is not known: the failed start's record, where to go, and the step it began at."""
    n = max(0, int(o.get("retreat") or 0))
    if n == 0 or home is None:
        return None
    return {"doc": dict(doc), "p": np.asarray(home[0], float), "R": np.asarray(home[1], float), "t0": int(t),
            "max": n, "tol": float(o.get("retreat_tol_m", AFTER_FAIL["retreat_tol_m"]))}


def retreat_action(obs: dict, st: dict) -> list:
    """One command toward the start pose at full speed, the hand open."""
    from rowant_lab.pro2.robot.follow import follow

    p, R = hand_pose(obs)
    return follow(p, R, st["p"], st["R"], -1.0, 1.0, 1.0)


def retreat_done(obs: dict, st: dict, t: int, cap: int) -> dict | None:
    """None while the hand is still on its way; else the failed start's record for ``plan_after_fail``, with the
    step the learner starts at, the steps left then, and what the retreat took (``retreat``: steps, how far from
    the start pose it ended)."""
    p, _ = hand_pose(obs)
    err = float(np.linalg.norm(p - st["p"]))
    if err > st["tol"] and int(t) - st["t0"] < st["max"]:
        return None
    doc = dict(st["doc"])
    doc.update(due_t=doc.get("t"), t=int(t), left=int(cap) - int(t),
               retreat={"steps": int(t) - st["t0"], "err_m": round(err, 4)})
    return doc


def fail_before(rec) -> dict | None:
    """The record of the robot's own failed start if the learner had started after it in this episode (or at once,
    the robot having no plan of its own), or None."""
    for r in (rec, (rec or {}).get("first") if isinstance(rec, dict) else None):
        own = r.get("own_first") if isinstance(r, dict) else None
        if isinstance(own, dict) and own.get("started"):
            return {k: v for k, v in own.items() if k not in ("pending", "started")}
    return None


def again_after_fail(R, instruction: str, tool, lib, cap: int, params: dict, first_rep: dict | None, rec) -> dict | None:
    """For the runner, when a plan is made again for another reason: ``plan_after_fail`` where the learner had
    started after the robot's own failed start (or at once, the robot having no plan), else None."""
    doc = fail_before(rec)
    return None if doc is None else plan_after_fail(R, instruction, tool, lib, cap, params, first_rep, doc)


def plan_after_fail(R, instruction: str, tool, lib, cap: int, params: dict, first_rep: dict | None,
                    failed: dict) -> dict:
    """The plan from the look taken after the robot's own start failed (``fail_due``'s record): the learner, when
    this look still accepts no percept of the target's type and enough steps are left; else the robot's own plan."""
    return plan(R, instruction, tool, lib, cap, params, first_rep, failed=dict(failed))


def why_tries(tries) -> list:
    """A planned look's tries for the record (2026-11-18), numbers and status words only: where the camera was
    asked to stand, and why the pose was refused (``lfb_episode._solve``: reach, limit, scene, plane, ok)."""
    out = []
    for t in tries or []:
        if not isinstance(t, dict):
            continue
        row = {"el": t.get("el_deg"), "az": t.get("az_deg"), "why": t.get("why")}
        if t.get("up_m") is not None:
            row["up_m"] = t["up_m"]
        if t.get("why") == "reach":
            row.update(pos_err_mm=t.get("pos_err_mm"), ang_err_deg=t.get("ang_err_deg"))
        cl = t.get("clear") or {}
        if t.get("why") == "scene":
            row.update(near_mm=None if cl.get("near_scene_m") is None else round(cl["near_scene_m"] * 1e3, 1),
                       near_at=cl.get("near_at"), n_near=cl.get("n_near"))
        if t.get("why") == "plane":
            row["above_plane_mm"] = None if cl.get("above_plane_m") is None else round(cl["above_plane_m"] * 1e3, 1)
        out.append(row)
    return out


def target_type(R, instruction: str, lib) -> str | None:
    try:
        parsed = R.parse(instruction, sorted(lib.types))
    except Exception:
        return None
    if not isinstance(parsed, dict) or parsed.get("status") != "ok":
        return None
    return parsed.get("target_type")


def triggers(R, instruction: str, tool, lib) -> str | None:
    """The instruction's target type, when the first look accepted no percept of it and the type is single-part."""
    T = target_type(R, instruction, lib)
    if not T or T not in lib.export_types or len(lib.export_types[T].get("parts") or {}) != 1:
        return None
    if any(p.get("type") == T for p in tool.percepts):
        return None
    return T


# The other forms of instruction (2026-11-19; ``params["learn_forms"]``, off unless given). As registered, the learner
# starts only where ``ground.parse`` reads the instruction, the form "pick the X and place it ...", and the first look
# accepted no percept of X (``triggers``). Every scene of the second world has that form, so its runs never showed the
# limit; the pre-flight of 11-19 did (new objects in the table scenes of the Spatial, Goal and Long suites: no start
# in any). With the option the robot's own reading of the whole instruction is asked (``episode.named_targets``: the
# goal form "put the X on the Y" and the sequencer's steps as well), and the learner also starts for:
# * ``put``: a one-step instruction of the goal form whose object has no accepted percept;
# * ``steps``: a step of an instruction of several steps whose object has no accepted percept (the first such step;
#   the thing is learned before the first step begins, and the robot's whole plan is then made with it);
# * ``relation``: an object asked for once, of whose type exactly one percept was accepted, and that one clearly
#   fails the instruction's relation (``ground.single_fails``: it is another object of the type; Spatial's second
#   bowl). The look from above then asks the robot's own type first, as it does when no percept was accepted.
# What stays out: a type asked for twice with one of the two accepted ("both pots"): the learned model would take
# the type's place in the library for both.
FORMS = {"put": True, "steps": True, "relation": True}


def form_target(R, instruction: str, tool, lib, params: dict) -> tuple:
    """``learn_forms``: (the type to learn, the record of why), or (None, None)."""
    return _form_target(R, instruction, tool, lib, params)[:2]


def _form_target(R, instruction: str, tool, lib, params: dict) -> tuple:
    """As ``form_target``, with the step's clause as the robot reads it (for ``learn_hint``)."""
    o = params.get("learn_forms")
    named_targets = getattr(R, "named_targets", None)
    if o is None or o is False or named_targets is None:
        return None, None, None
    o = {**FORMS, **(o if isinstance(o, dict) else {})}
    try:
        named = named_targets(instruction, tool, lib, params)
    except Exception:
        return None, None, None
    asked: dict = {}
    for x in named:
        asked[x["type"]] = asked.get(x["type"], 0) + 1
    for x in named:
        T = x["type"]
        if T not in lib.export_types or len(lib.export_types[T].get("parts") or {}) != 1:
            continue
        if x["have"] == 0:
            why = "steps" if x["steps"] > 1 else ("put" if x["form"] == "put" else None)
        elif x["have"] == 1 and x["single_fails"] is True and asked[T] == 1:
            why = "relation"
        else:
            why = None
        if why is not None and o.get(why):
            return T, {"why": why, "step": int(x["step"]), "steps": int(x["steps"]), "form": x["form"],
                       "relation": x["relation"]}, x["clause"]
    return None, None, None


# The place the robot would have guessed (2026-11-19; ``params["learn_hint"]`` with the server's option of the same
# name, off unless given). The server takes the largest unexplained thing for the object to learn. Where every other
# thing has a model that is the right one; in the table scenes a pan with no model and a strip of table edge were
# taken instead (Mac). With the option the robot first makes its own plan for the clause
# on a COPY of the look tool, as it would without the learner, and where that plan acts on a fallback for the object
# (a rejected fit, a region its model was stood on), the fallback's place is sent along: the server takes the
# unexplained thing there. So the learner learns the thing the robot would otherwise have guessed at. Nothing of
# that plan is kept: not its percepts, not its settings, not its agent.
def fallback_place(R, clause: str, tool, lib, cap: int, params: dict, T: str) -> dict | None:
    """``learn_hint``: ``{"xy", "source"}`` of the fallback percept of type ``T`` the robot's own plan for ``clause``
    would act on, or None (no plan, a target that is no fallback, another type, an error)."""
    try:
        t2 = copy.copy(tool)
        for k, v in list(vars(tool).items()):
            if isinstance(v, (list, dict, set)):
                setattr(t2, k, copy.deepcopy(v) if k in ("percepts", "fallbacks") else copy.copy(v))
        out = R.plan_episode(clause, t2, lib, int(cap), {k: v for k, v in params.items() if k != "sequence"})
        tid = (out.get("grounding") or {}).get("target")
        p = next((q for q in getattr(t2, "percepts", None) or [] if q.get("id") == tid), None)
        if p is None or (p.get("type") != T and p.get("named") != T):
            return None
        if not any(f.get("id") == tid and f.get("role") == "target" for f in getattr(t2, "fallbacks", None) or []):
            return None  # (an accepted percept: nothing to point at)
        pos = ((p.get("bodies") or {}).get(lib.export_types[p["type"]]["root_body"]) or {}).get("pos")
        if pos is None:
            return None
        return {"xy": [round(float(pos[0]), 4), round(float(pos[1]), 4)], "source": str(p.get("source"))[:40]}
    except Exception:
        return None


def caps_attempts(form: dict | None) -> bool:
    """Whether the way back's cap on attempts applies to the plan made on a learned thing: an instruction of one
    step (as registered). With several steps (``learn_forms``) the cap would bind every step's agent."""
    return int((form or {}).get("steps") or 1) <= 1


def _forced(R, instruction: str, tool, lib) -> str | None:
    """Development only: the target type with its accepted percepts dropped, as if the first look had missed it."""
    T = target_type(R, instruction, lib)
    if not T or T not in lib.export_types or len(lib.export_types[T].get("parts") or {}) != 1:
        return None
    tool.percepts = [p for p in tool.percepts if p.get("type") != T]
    return T


def plan(R, instruction: str, tool, lib, cap: int, params: dict, first_rep: dict | None,
         failed: dict | None = None) -> dict:
    """``plan_episode``'s contract (``robot/episode.py``): the robot's own plan, or the learn-then-act agent.
    ``failed`` (``plan_after_fail``): the record of the robot's own start that failed before this look."""
    def not_started(why: str) -> dict:  # (the record of a learner that was due after a failed start and stays out)
        return {} if failed is None else {"own_first": {**failed, "pending": False, "started": False, "why": why}}

    T = triggers(R, instruction, tool, lib)
    form, clause = None, instruction
    if T is None:  # (``learn_forms``, off unless given: the other forms of instruction)
        T, form, c = _form_target(R, instruction, tool, lib, params)
        clause = c or instruction
    if T is not None and params.get("learn_late") is not None:
        # 2026-11-18, off unless given: too late to finish, so the learner does not start (see ``LATE``)
        need = int({**LATE, **params["learn_late"]}["min_left"])
        if failed is not None and after_fail(params) is not None:
            need = min(need, int(after_fail(params)["min_left"]))  # after a failed start: that option's own floor
        if int(cap) < need:
            out = R.plan_episode(instruction, tool, lib, cap, params)
            return {**out, "learn": {"trigger": False, "late": {"left": int(cap), "min_left": need, "target_type": T},
                                     **not_started("late")}}
    if T is None and params.get("learn_dev_force"):  # development only (see the module's docstring)
        T = _forced(R, instruction, tool, lib)
    if T is None:
        out = R.plan_episode(instruction, tool, lib, cap, params)
        return {**out, "learn": {"trigger": False, **not_started("accepted")}}
    if failed is None and own_first(params, form) is not None:
        # 2026-11-18, off unless given (``AFTER_FAIL``): the robot's own plan first. Where it has none, the episode
        # goes on as the robot's own does (the arm's camera check moves the arm when there is no agent), and the
        # runner asks ``fail_due`` once that check has had its turn
        out = R.plan_episode(instruction, tool, lib, cap, params)
        own = {"pending": True, "target_type": T, **({"form": form["why"]} if form is not None else {})}
        if out.get("agent") is None:
            g = out.get("grounding") or {}
            own.update(no_own_plan=True, grounding={k: g.get(k) for k in ("status", "reason")})
        return {**out, "learn": {"trigger": False, "own_first": own}}
    agent = LearnThenAct(R, instruction, tool, lib, cap, params, T, first_rep, form=form)
    hn = params.get("learn_hint")
    if hn is not None and hn is not False:  # (off unless given: the place the robot's own plan would have guessed)
        agent.hint = fallback_place(R, clause, tool, lib, cap, params, T)
        agent.rec["hint"] = agent.hint
    if failed is not None:
        agent.rec["own_first"] = {**failed, "pending": False, "started": True}
    return {"agent": agent, "parsed": {"status": "ok", "target_type": T, "learning": True},
            "grounding": {"status": "learning"}, "learn": agent.rec}


class LearnThenAct:
    def __init__(self, R, instruction, tool, lib, cap, params, T, first_rep, form=None):
        self.R, self.instruction, self.tool, self.lib, self.cap = R, instruction, tool, lib, int(cap)
        self.params, self.T = params, T
        self.form = form  # (``learn_forms``: why it started, when not by the registered trigger; else None)
        self.hint = None  # (``learn_hint``: the place the robot's own plan would have guessed, sent to the server)
        self.plane = (first_rep or {}).get("plane")
        self.t = 0
        self.obs = None
        self.inner = None
        self._done = False
        self.finish_reason = None
        self.rec: dict = {"trigger": True, "target_type": T, "looks": [], "steps": 0, "fallback": None}
        if form is not None:
            self.rec["form"] = dict(form)
        self._steps_learned = 0  # the way back: the steps the learned thing's agent used before the robot's own took over
        self._tool0 = None  # the look tool's percepts, fallbacks and setting as they were before a learned thing joined
        self._learned = None  # the server's reply for the thing learned (``learn_keep``: a plan made again goes on with it)
        self._gen = self._run()
        self._started = False
        self._prev_sig = None

    # -- the agent interface (dev_episode)
    @property
    def finished(self) -> bool:
        if self.inner is not None:
            done = bool(getattr(self.inner, "finished", False))
            if done and (getattr(self, "params", None) or {}).get("learn_calib_cue"):
                # (off unless given) the robot's own plan has ended: the camera check reads its ending here. The
                # runner reads ``finished`` every step and never asks an ended agent for a command, so this is the
                # place (the first build copied it in ``act`` alone, and the check never saw it: Mac, 20:55)
                self.finish_reason = getattr(self.inner, "finish_reason", None)
            return done
        return self._done

    def act(self, obs: dict) -> list:
        if self.inner is not None:
            a = self.inner.act(obs)
            if self.params.get("learn_calib_cue") and getattr(self.inner, "finished", False):
                # (off unless given) the robot's own plan has ended: the camera check reads its ending here
                self.finish_reason = getattr(self.inner, "finish_reason", None)
            return a
        self.obs = obs
        try:
            a = next(self._gen) if not self._started else self._gen.send(obs)
            self._started = True
        except StopIteration:
            a = None
        if a is None:  # the learning phase is over: the episode's agent takes over from this same step
            if self.inner is not None:
                return self.inner.act(obs)
            self._done = True
            self.finish_reason = self.finish_reason or "learn_no_agent"
            return [0.0] * 6 + [-1.0]
        self.t += 1
        return [float(x) for x in a]

    def summary(self) -> dict:
        s = self.inner.summary() if self.inner is not None and hasattr(self.inner, "summary") else {
            "finish_reason": self.finish_reason, "steps": 0, "attempts": [], "events": []}
        s = dict(s)
        s["steps"] = int(s.get("steps") or 0) + self.t + int(getattr(self, "_steps_learned", 0) or 0)
        s["learn"] = self.rec
        return s

    def go_back(self, doc: dict, left: int) -> None:
        """The way back (``WAY_BACK``; the runner calls it when ``back_due`` says so): the learned thing's library,
        percept and settings come off the look tool, and the robot's own plan is made with the ``left`` steps, as
        ``_act_agent`` makes it when learning fails. ``doc`` becomes ``learn.way_back``; it keeps what the learned
        thing's agent did (its ending, steps and each attempt's outcome)."""
        tool = self.tool
        s = self.inner.summary() if self.inner is not None and hasattr(self.inner, "summary") else {}
        doc["learned"] = {"finish_reason": s.get("finish_reason"), "steps": int(s.get("steps") or 0),
                          "attempts": [{k: a.get(k) for k in ("index", "t_start", "t_end", "outcome")}
                                       for a in s.get("attempts") or []]}
        self._steps_learned = int(s.get("steps") or 0)
        self.rec["way_back"] = doc
        tool.lib = self.lib
        t0 = self._tool0 or {}
        # the percepts as they were before the learned thing joined them: the plan made on it added its own (with the
        # basket learned as the object, the place fell to an unexplained patch 1.3 m behind the robot; Mac, task 65)
        tool.percepts = list(t0["percepts"]) if "percepts" in t0 else [p for p in tool.percepts if p.get("id") != "learned0"]
        if "fallbacks" in t0 and hasattr(tool, "fallbacks"):
            doc["learned"]["fallbacks"] = [{k: x.get(k) for k in ("role", "type", "source")}
                                           for x in list(tool.fallbacks)[len(t0["fallbacks"]):]]
            tool.fallbacks = list(t0["fallbacks"])
        if t0.get("held_near_m") is not None:
            tool.held_near_m = t0["held_near_m"]
        out = self.R.plan_episode(self.instruction, tool, self.lib, max(0, int(left)), self.params)
        doc["grounding"] = (out.get("grounding") or {}).get("status", "ok")
        if self.params.get("learn_calib_cue"):
            self.finish_reason = None  # (the learned thing's ending is not the new plan's)
        self.inner = out.get("agent")
        if self.inner is None:
            self._done, self.finish_reason = True, "grounding_failed"

    # -- helpers
    def _call(self, msg: dict) -> dict:
        a = time.perf_counter()
        rep = self.tool.client.call(msg)
        self.tool.bridge_s += time.perf_counter() - a
        return rep

    @staticmethod
    def _plan_doc(rep: dict) -> dict:
        """A ``learn_plan`` reply for the record (numbers and status words only)."""
        lo = rep.get("leftover") or {}
        return {"status": rep.get("status"), "n_planned": rep.get("n_planned"), "times": rep.get("times"),
                "leftover": {k: lo.get(k) for k in ("centre_w", "bbox_min", "bbox_max", "n_vox_object",
                                                    "runner_up_vox", "comp_sizes", "rank_known", "aim_w",
                                                    "scene", "self_px_points")},
                "looks": [{k: r.get(k) for k in ("look", "kind", "az_used_deg", "el_used_deg")} |
                          {"n_tries": len(r.get("tries") or []), "n_waypoints": len(r.get("waypoints") or []),
                           "tries": why_tries(r.get("tries"))}
                          for r in rep.get("looks") or []]}

    def _near_types(self, o: dict, centre) -> dict:
        """``learn_ask_known``: {type: (percept id, its root position)} of the accepted percepts within ``near_m``
        (horizontally) of the thing's centre: the thing may be a part of one of them. One percept a type, the
        nearest; never the instruction's own type."""
        out = {}
        if centre is None:
            return out
        for p in self.tool.percepts:
            t = p.get("type")
            tdoc = self.lib.export_types.get(t) if t and t != self.T else None
            pos = ((p.get("bodies") or {}).get((tdoc or {}).get("root_body")) or {}).get("pos") if tdoc else None
            if pos is None or p.get("id") is None:
                continue
            d = float(np.hypot(pos[0] - centre[0], pos[1] - centre[1]))
            if d <= float(o["near_m"]) and (t not in out or d < out[t][2]):
                out[t] = (p["id"], [float(x) for x in pos], d)
        return out

    @staticmethod
    def _moved(had, t_w) -> dict:
        """For a type that has a percept: its id and how far the fitted root position lies from it."""
        if not had or t_w is None:
            return {}
        return {"had": had[0], "moved_m": round(float(np.linalg.norm(np.asarray(t_w, float) - np.asarray(had[1], float))), 4)}

    def _ask_types(self, o: dict, near: dict) -> list:
        """``learn_ask_known``: the known types the thing may be, in the order they are asked: the place's type when
        the first look accepted no percept of it, the types of accepted percepts near the thing (``near``), and with
        ``rejected`` the types of the first look's rejected poses that have no percept. Never the instruction's own
        type, unless ``own`` is set: then it is asked first. At most ``max_types``."""
        tool, lib = self.tool, self.lib
        have = {p.get("type") for p in tool.percepts}
        out = []
        try:
            parsed = self.R.parse(self.instruction, sorted(lib.types))
        except Exception:
            parsed = {}
        place = (parsed.get("place_types") or [parsed.get("place_type")]) if isinstance(parsed, dict) else []
        rejected = [q.get("type") for q in (getattr(tool, "extra", None) or {}).get("rejected") or [] if q.get("bodies")]
        # (``learn_forms``, the relation's case: the accepted percept of the type is another object, so the own type
        # is asked all the same)
        other = (getattr(self, "form", None) or {}).get("why") == "relation"
        own = [self.T] if (o.get("own") or o.get("own_near") is not None) and self.T in lib.export_types \
            and (self.T not in have or other) else []
        for t in own + [t for t in place if t not in have] + sorted(near, key=lambda t: near[t][2]) \
                + ([t for t in rejected if t not in have] if o.get("rejected") else []):
            if t and (t != self.T or own) and t not in out and t in lib.export_types:
                out.append(t)
        return out[:int(o["max_types"])]

    def _own_near(self, o: dict, asked) -> dict | None:
        """``own_near``: the fit of the instruction's own type from the look from above, when it passes the looser
        floors (``OWN_NEAR``, or the option's own numbers); else None."""
        f = o.get("own_near")
        if f is None or not asked:
            return None
        f = f if isinstance(f, dict) else {}
        sets = [{**OWN_NEAR, **dict(x)} for x in f["any"]] if f.get("any") else [{**OWN_NEAR, **f}]
        for a in asked:
            if a.get("type") != self.T or a.get("status") != "fit":
                continue
            fit = {k: a.get(k) for k in ("inlier", "seen", "cover")}
            for fl in sets:  # (``any``: the first set of floors the fit passes)
                if all(isinstance(fit[k], (int, float)) and float(fit[k]) >= float(fl[k]) for k in fit):
                    return {**{k: round(float(v), 4) for k, v in fit.items()}, "floors": fl}
        return None

    def _under_top_min(self, o: dict, kn: dict) -> dict | None:
        """``own_top_min``: the fit of a thing the server's floors accept as the instruction's own type, when it
        does not pass the option's floors for acting on it from above; else None (also without the option)."""
        f = o.get("own_top_min")
        if f is None or kn.get("type") != self.T:
            return None
        f = {**OWN_TOP_MIN, **(f if isinstance(f, dict) else {})}
        fit = {k: kn.get(k) for k in ("inlier", "seen", "cover")}
        if all(isinstance(fit[k], (int, float)) and float(fit[k]) >= float(f[k]) for k in fit):
            return None
        return {**{k: (round(float(v), 4) if isinstance(v, (int, float)) else None) for k, v in fit.items()},
                "strict": True, "top_min": f}

    def _proprio(self):
        # the readings as they came (``tool.obs``): under the delay compensator ``self.obs`` is where the next command
        # will find the hand, which steers the motion; a frame's pose is the pose the camera showed
        o = getattr(self.tool, "obs", None) or self.obs
        return (np.asarray(o["robot0_joint_pos"], float), np.asarray(o["robot0_gripper_qpos"], float),
                np.asarray(o["robot0_eef_pos"], float), np.asarray(o["robot0_eef_quat"], float))

    def _wrist(self) -> dict:
        """The wrist camera's frame now: the camera tool's, or, when the runner has its own source
        (``wrist_stereo.WristFrames``, ``params["learn_wrist_stereo"]``), a frame with colour, the pose from the
        hand's readings and partner pictures, and no depth (the server makes the depth)."""
        src = getattr(self.tool.cam, "wrist_frame", None)
        return self.tool._frame(WRIST) if src is None else src()

    def _finder_check(self, lo: dict) -> dict:
        """The robot's top-down finder (one wrist colour picture, no depth) asked for the same thing: how far its
        place is from the one the server found in the look's depth. Numbers for the record; nothing acts on it."""
        c = lo.get("centre_w")
        try:
            got = self.tool.wrist_find(self.T, float((lo.get("plane") or self.plane or {}).get("height_w", 0.0)),
                                       hint_xy=None if c is None else [float(c[0]), float(c[1])])
        except Exception as exc:
            return {"error": type(exc).__name__}
        if not got or got.get("xy") is None:
            return {"found": False}
        out = {"found": True, "xy": [round(float(x), 4) for x in got["xy"]]}
        if c is not None:
            out["from_depth_m"] = round(float(np.linalg.norm(np.asarray(got["xy"], float) - np.asarray(c[:2], float))), 4)
        return out

    def _pose(self):
        from rowant_lab.pro2.robot.grasp import quat_xyzw_to_mat

        return np.asarray(self.obs["robot0_eef_pos"], float), quat_xyzw_to_mat(self.obs["robot0_eef_quat"])

    def _goto(self, doc: dict, tol: tuple, max_steps: int, cap_pos: float):
        """Follow one planned hand pose; returns (status, steps). A plateau over ``PLATEAU_WINDOW`` steps ends it."""
        from rowant_lab.pro2.robot.follow import follow, pose_error
        from rowant_lab.pro2.robot.grasp import quat_wxyz_to_mat

        p_tgt = np.asarray(doc["eef_pos"], float)
        R_tgt = quat_wxyz_to_mat(doc["eef_quat_wxyz"])
        hist = []
        t0 = self.t
        out = None
        for k in range(max_steps):
            p, R = self._pose()
            e, r = pose_error(p, R, p_tgt, R_tgt)
            if e <= tol[0] and r <= tol[1]:
                out = ("reached", k, e, r)
                break
            hist.append((e, r))
            if len(hist) > PLATEAU_WINDOW:
                e0, r0 = hist[-1 - PLATEAU_WINDOW]
                if e0 - e < PLATEAU_GAIN_M and r0 - r < PLATEAU_GAIN_DEG:
                    out = ("plateau", k, e, r)
                    break
            yield follow(p, R, p_tgt, R_tgt, -1.0, cap_pos, 1.0)
        if out is None:
            p, R = self._pose()
            e, r = pose_error(p, R, p_tgt, R_tgt)
            out = ("max_steps", max_steps, e, r)
        self.rec.setdefault("gotos", []).append({
            "kind": doc.get("kind"), "t0": t0, "steps": self.t - t0, "status": out[0], "err_m": round(float(out[2]), 4),
            "err_deg": round(float(out[3]), 1), "q": [round(float(x), 2) for x in self.obs["robot0_joint_pos"]],
            "start_err_m": round(float(hist[0][0]), 3) if hist else None})
        return out

    # -- the episode
    def _run(self):
        t0 = time.perf_counter()
        tool, rec = self.tool, self.rec
        ok = self.plane is not None
        rep = None
        top_first = self.params.get("learn_top_first") is not None
        if ok:
            jp, gq, ep_, eq = self._proprio()
            agent_fr = tool._frame(AGENT)
            wrist_fr = self._wrist()
            msg = {"op": "learn_plan", "suite": tool.suite, "episode": tool.episode, "cameras": [agent_fr],
                   "joint_pos": jp, "gripper_qpos": gq,
                   "percepts": [{"type": p["type"], "gid": p.get("gid"), "bodies": p["bodies"]} for p in tool.percepts],
                   "plane": self.plane,
                   "wrist": {"cameras": [wrist_fr], "joint_pos": jp, "gripper_qpos": gq, "eef_pos": ep_,
                             "eef_quat_xyzw": eq}}
            if getattr(self, "hint", None) is not None:
                msg["hint"] = {"xy": list(self.hint["xy"])}  # (``learn_hint``)
            if self.params.get("learn_rank_known") is not None:
                # 2026-11-17, off by default: the rejected poses of known types the first look accepted nowhere
                # (never the instruction's own type). The server considers an unexplained thing such a pose covers
                # last: it is most likely that known object, not the new one (``lfb_episode.rank_components``).
                have = {p["type"] for p in tool.percepts}
                msg["rejected"] = [{"type": q["type"], "gid": q.get("gid"), "bodies": q["bodies"]}
                                   for q in (getattr(tool, "extra", None) or {}).get("rejected") or []
                                   if q.get("bodies") and q.get("type") != self.T and q.get("type") not in have]
            if top_first:
                msg["stage"] = "top"
            rep = self._call(msg)
            top_first = top_first and rep.get("stage") == "top"  # (a server without the option plans as before)
            rec["plan"] = self._plan_doc(rep)
            ok = rep.get("status") == "ok" and int(rep.get("n_planned") or 0) >= (1 if top_first else MIN_LOOKS)
        frames, joints = [], []
        fast = self.params.get("learn_looks") == "fast"
        look_tol, wp_tol = (FAST["look_tol"], FAST["waypoint_tol"]) if fast else (LOOK_TOL, WAYPOINT_TOL)
        wp_max, look_max = ((FAST["waypoint_max_steps"], FAST["look_max_steps"]) if fast
                            else (WAYPOINT_MAX_STEPS, LOOK_MAX_STEPS))
        end = self.cap - ACT_RESERVE  # the looks' step budget: every look and waypoint stops at it

        def visit(look):
            """Through a look's waypoints to its pose; the wrist frame and the joints there, or None when the budget
            cut it. A generator: it yields the motion's actions."""
            if self.t >= end:
                rec["looks"].append({"look": look["look"], "status": "skipped_budget"})
                return None
            t_look = self.t
            for wp in look.get("waypoints") or []:
                if end - self.t > 0:
                    yield from self._goto({**wp, "kind": f"wp{look['look']}"}, wp_tol, min(wp_max, end - self.t), 1.0)
            status, _k, e, r = yield from self._goto({**look, "kind": f"look{look['look']}"}, look_tol,
                                                     max(0, min(look_max, end - self.t)), 1.0)
            if status == "max_steps" and self.t >= end and e > 3 * look_tol[0]:
                rec["looks"].append({"look": look["look"], "status": "cut_budget", "steps": self.t - t_look,
                                     "err_m": round(float(e), 4), "err_deg": round(float(r), 2)})
                return None
            jp, gq, _ep, _eq = self._proprio()
            fr = self._wrist()
            rec["looks"].append({"look": look["look"], "status": status, "steps": self.t - t_look,
                                 "err_m": round(float(e), 4), "err_deg": round(float(r), 2)})
            if look.get("kind") == "top":
                rec["looks"][-1]["kind"] = "top"
            if "partners" in fr:  # numbers only: how far to the side each partner picture was taken
                rec["looks"][-1]["partners_mm"] = [round(float(p_["b_eff_m"]) * 1e3, 1) for p_ in fr["partners"]]
            return fr, {"joint_pos": jp, "gripper_qpos": gq}

        ask = self.params.get("learn_ask_known") if top_first else None
        n_things = 1
        while ok and top_first:  # the look from above, then the thing found again in it and the flank looks around it
            got = None
            if "eef_pos" in rep["looks"][0]:
                got = yield from visit(rep["looks"][0])
            ok = got is not None
            if not ok:
                break
            jp, gq, ep_, eq = self._proprio()
            m2 = {"op": "learn_plan", "stage": "flanks", "suite": tool.suite, "episode": tool.episode,
                  "top": {"cameras": [got[0]], "joint_pos": jp, "gripper_qpos": gq, "eef_pos": ep_,
                          "eef_quat_xyzw": eq}}
            near, refused, asked = {}, None, None
            if ask is not None:
                o = {**ASK, **ask}
                near = self._near_types(o, (rec["plan"].get("leftover") or {}).get("centre_w"))
                # (a type explained before has a percept now: it is asked again only when that lies near this thing,
                # which may be another part of the same object)
                ask_types = self._ask_types(o, near)
                if ask_types:
                    m2["ask"] = {"types": ask_types}
            rep = self._call(m2)
            kn, mv = None, {}
            if rep.get("ask") is not None:
                # numbers only; for a type that has a percept: which, and how far the fitted pose lies from it
                asked = [{**{k: v for k, v in a.items() if k not in ("R_w", "t_w")},
                          **self._moved(near.get(a.get("type")), a.get("t_w"))} for a in rep["ask"]]
            if ask is not None and rep.get("status") == "known":
                kn = rep["known"]
                mv = self._moved(near.get(kn.get("type")), kn.get("t_w"))
                if mv and mv["moved_m"] > float(o["move_max_m"]):
                    # the fit puts an accepted object far from where the first look placed it: not believed. The
                    # thing is gone on with as an unknown one (asked again without the types: the ring is planned)
                    refused, kn = {"type": kn.get("type"), **mv}, None
                    rep = self._call({k: v for k, v in m2.items() if k != "ask"})
            lo = rep.get("leftover") or {}
            rec["plan2"] = {"status": rep.get("status"), "n_planned": rep.get("n_planned"), "times": rep.get("times"),
                            "stereo": rep.get("stereo"),
                            "leftover": {k: lo.get(k) for k in ("centre_w", "bbox_min", "bbox_max", "n_vox_object",
                                                                "runner_up_vox", "comp_sizes", "shift_m")},
                            "plane": {k: (lo.get("plane") or {}).get(k) for k in ("height_w", "first_look_height_w")},
                            "looks": [{k: r.get(k) for k in ("look", "kind", "az_used_deg", "el_used_deg")} |
                                      {"n_tries": len(r.get("tries") or []),
                                       "n_waypoints": len(r.get("waypoints") or []),
                                       "tries": why_tries(r.get("tries"))}
                                      for r in rep.get("looks") or []],
                            "finder": self._finder_check(lo)}
            if asked is not None:
                rec["plan2"]["ask"] = asked
            if refused is not None:
                rec["plan2"]["known_refused"] = refused
            # after a thing was explained, a thing the look from above does not find again was a leftover of it (the
            # first look's smear of an object that now stands where the fit put it): the next thing is taken too
            gone = kn is None and ask is not None and rep.get("status") == "no_refind" and bool(rec.get("known"))
            if kn is None and not gone:
                near_own = self._own_near(o, asked) if ask is not None else None
                if near_own is not None:
                    # (``own_near``) a known object seen under a change: nothing to learn, and nothing of this look
                    # is kept. The robot's own plan is made on what the first look saw
                    rec["own_near"] = near_own
                    ok, rec["known_end"] = False, "own_near"
                    break
                ok = rep.get("status") == "ok" and int(rep.get("n_planned") or 0) >= 1
                break
            if gone:
                rec.setdefault("gone", []).append({"t": self.t,
                                                   "first_centre_w": (rec["plan"].get("leftover") or {}).get("centre_w")})
            elif self._under_top_min(o, kn) is not None:
                # (``own_top_min``) the server takes the thing for the instruction's own type, and the fit is not
                # good enough to act on from above: a known object seen under a change. Nothing of this look is
                # kept; the robot's own plan is made on what the first look saw
                rec["own_near"] = self._under_top_min(o, kn)
                ok, rec["known_end"] = False, "own_near"
                break
            else:
                # a known type explains the thing: it is that object, and the next thing in the order is taken
                doc = {"type": kn.get("type"), "t": self.t, "inlier": kn.get("inlier"), "seen": kn.get("seen"),
                       "cover": kn.get("cover"), "n_agree": kn.get("n_agree"),
                       "first_centre_w": (rec["plan"].get("leftover") or {}).get("centre_w"),
                       "centre_w": lo.get("centre_w"), "n_vox_object": lo.get("n_vox_object"), **mv}
                rec.setdefault("known", []).append(doc)
                try:
                    bodies = tool._build(kn["type"], np.asarray(kn["R_w"], float), np.asarray(kn["t_w"], float))
                    doc["id"] = mv.get("had") or f"known{len(rec['known']) - 1}"
                    new = {"id": doc["id"], "type": kn["type"], "gid": kn.get("gid"), "bodies": bodies, "joints": {},
                           "fit": {"n_agree": kn.get("n_agree"), "inlier": kn.get("inlier")}, "score": None,
                           "source": "top_look"}
                    # an object the first look had placed keeps its name and takes the fitted pose
                    tool.percepts = [p for p in tool.percepts if p.get("id") != doc["id"]] + [new]
                except Exception as exc:  # the percept could not be built: no further thing; the robot's own plan
                    doc["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
                    ok, rec["known_end"] = False, "error"
                    break
                if kn.get("type") == self.T:
                    # (``own``) the instruction's own thing, and the robot's own model explains it: nothing to learn.
                    # The robot's own plan acts on the percept just made; no ring of looks, no further thing
                    ok, rec["known_end"] = False, "own"
                    break
            if n_things >= int(o["max_things"]):
                ok, rec["known_end"] = False, "count"
                break
            if self.t + int(o["next_need"]) > end:
                ok, rec["known_end"] = False, "steps"
                break
            n_things += 1
            jp, gq, ep_, eq = self._proprio()
            msg = {**msg, "next": True,
                   "percepts": [{"type": p["type"], "gid": p.get("gid"), "bodies": p["bodies"]} for p in tool.percepts],
                   "wrist": {"cameras": [self._wrist()], "joint_pos": jp, "gripper_qpos": gq, "eef_pos": ep_,
                             "eef_quat_xyzw": eq}}
            if "rejected" in msg and kn is not None:
                msg["rejected"] = [q for q in msg["rejected"] if q.get("type") != kn["type"]]
            rep = self._call(msg)
            rec["plan"] = self._plan_doc(rep)
            ok = rep.get("status") == "ok" and int(rep.get("n_planned") or 0) >= 1
            if not ok:
                rec["known_end"] = "no_next"
        if ok:
            for look in rep["looks"]:
                if "eef_pos" not in look:
                    continue
                got = yield from visit(look)
                if got is not None:
                    frames.append(got[0])
                    joints.append(got[1])
            ok = len(frames) >= (1 if top_first else MIN_LOOKS)
        rec["steps"] = self.t
        learned, own = None, None
        if ok:
            rep = self._call({"op": "learn", "suite": tool.suite, "episode": tool.episode, "label": self.T,
                              "cameras": frames, "joints": joints})
            rec["learn"] = {k: rep.get(k) for k in ("status", "masks", "model", "times", "gid", "n_looks", "check")}
            if rep.get("stereo") is not None:
                rec["learn"]["stereo"] = rep["stereo"]
            if rep.get("status") == "ok":
                learned = rep
                self._learned = rep
            elif rep.get("status") == "own":
                own = rep.get("own")
        rec["fallback"] = learned is None and own is None
        rec["s"] = round(time.perf_counter() - t0, 3)
        self.inner = self._act_agent(learned, own)
        return None

    def _act_agent(self, learned: dict | None, own: dict | None = None, left: int | None = None):
        """The episode's agent with the steps left: on the variant library and F7's settings when learned; on the robot's
        own library with a percept at the looks' pose when its own model explains the looks; else the robot's own plan
        (its fallbacks)."""
        from rowant_lab.libero_env import m2_library as LB
        from rowant_lab.pro2 import long_paths as LPX

        tool, R = self.tool, self.R
        left = max(0, self.cap - self.t) if left is None else max(0, int(left))
        params, lib = self.params, self.lib
        if learned is None and own is not None:
            try:
                bodies = tool._build(self.T, np.asarray(own["R_w"], float), np.asarray(own["t_w"], float))
                tool.percepts = list(tool.percepts) + [{
                    "id": "close0", "type": self.T, "gid": own.get("gid"), "bodies": bodies, "joints": {},
                    "fit": {"n_agree": own.get("n_agree"), "inlier": own.get("inlier")}, "score": None,
                    "source": "close_look"}]
                self.rec["acting"] = "own_close"
            except Exception as exc:  # the percept could not be built: the robot's own plan
                tool.percepts = [p for p in tool.percepts if p.get("id") != "close0"]
                self.rec["acting"] = f"own (close-look error: {type(exc).__name__}: {str(exc)[:120]})"
        elif learned is not None:
            try:
                self._tool0 = {"percepts": list(tool.percepts), "fallbacks": list(getattr(tool, "fallbacks", None) or []),
                               "held_near_m": getattr(tool, "held_near_m", None)}
                # (2026-11-19: the long suite's cells need their own entry, ``long_paths.extra_suites``; for the Object,
                # Spatial and Goal suites it gives what ``goal_paths.extra_suites`` gives. Before, a thing learned in
                # a scene of the long suite was refused here, "neither M2's nor an extra suite".)
                lib = LB.M2Library(tool.suite, lib_dir=learned["policy_lib"], m1_library=learned["m1_library"],
                                   extra_suites=LPX.extra_suites([tool.suite]))
                o = np.asarray(self.rec["learn"]["model"]["origin_w"], float)
                tool.lib = lib
                bodies = tool._build(self.T, np.eye(3), o)
                tool.percepts = list(tool.percepts) + [{
                    "id": "learned0", "type": self.T, "gid": learned.get("gid"), "bodies": bodies, "joints": {},
                    "fit": {"n_agree": 0, "inlier": 1.0}, "score": None, "source": "learned_in_episode"}]
                tool.held_near_m = F7_HELD_NEAR_M
                params = copy.deepcopy(self.params)
                m1 = params.setdefault("m1", {}) or {}
                params["m1"] = {**m1, "agent": {**(m1.get("agent") or {}), **F7_AGENT}}
                wb = way_back(self.params)
                if wb is not None and caps_attempts(getattr(self, "form", None)):
                    # the learned thing's attempts end here; the runner then asks ``back_due``. (``learn_forms``, an
                    # instruction of several steps: the cap would bind every step's agent, also those on objects the
                    # robot knows, and the sequence's ending is none the way back acts on; no cap there.)
                    params["m1"]["agent"]["max_attempts"] = int(wb["attempts"])
                self.rec["acting"] = "learned"
            except Exception as exc:  # the learned library could not be used: the robot's own plan
                tool.lib, lib, params = self.lib, self.lib, self.params
                tool.percepts = [p for p in tool.percepts if p.get("id") != "learned0"]
                self.rec["acting"] = f"own (library error: {type(exc).__name__}: {str(exc)[:120]})"
        else:
            self.rec["acting"] = {"own": "own_top", "own_near": "own_near"}.get(self.rec.get("known_end"), "own")
        out = R.plan_episode(self.instruction, tool, lib, left, params)
        self.rec["grounding"] = out.get("grounding")
        self.rec["parsed_target"] = (out.get("parsed") or {}).get("target_type")
        if out.get("agent") is None:
            self.finish_reason = "grounding_failed"
        return out.get("agent")
