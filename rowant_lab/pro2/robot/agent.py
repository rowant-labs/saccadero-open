"""M1 oracle agent: grasp-and-place on true poses with LIBERO's OSC_POSE (plan agent_design; numpy only).

In this package ``oracle`` is the perception tool (``look.PerceptionTool``): the poses are perceived, not true. The
text below describes the agent as first written, for true poses (``tools``: ``rowant_lab/libero_env/tools.py``).

ORACLE (true poses): diagnostic of the hands. The policy sees only:

* the proprioception whitelist (``tools.filter_obs``), passed to ``act`` each step;
* the declared truth channel ``oracle()`` (``tools.OracleTool``): world poses of every task object and fixture body;
* the task spec parsed from the BDDL text (``tools.task_spec``) and its own object library (collision primitives in
  body frames, site regions in body frames, the arena's static geoms, grasp records and the Panda jaw model), read
  from files the exporter wrote once.

It never sees the simulator, ``done``, rewards, contacts, the evaluator or any image. It returns one 7-D action per
call; only the runner steps the environment.

One attempt (at most ``max_attempts``; a retry starts only if at least ``T_attempt`` steps are left):

1. read: one oracle call; plan top-down frames (``grasp.plan_frames``), skipping records already tried;
2. pre-grasp: rise to the corridor's safe height if needed, move above the grasp and turn to its yaw (open);
3. approach: straight down to the grasp pose; a stall (z not falling for ``stall_steps``) aborts the attempt;
4. close: hold and close until the jaw gap changes less than ``close_stop_m`` over ``close_window`` steps;
   C1: the gap is within the record's width band (per family: wide grasps, rim grasps);
5. lift: ``lift_slow_m`` slowly, then to ``lift_m``; C2 (oracle): the gap still in band and the target moved with
   the hand (relative position change under ``c2_rel_m``);
6. transport: rise to the transit height (the highest primitive top in the xy corridor + ``transit_margin_m`` +
   the held extent below the hand), move above the place point, lower to the release height (the held object's
   lowest point ``place_clear`` above the place surface, raised until the held object and fingers clear the
   container by ``place_gripper_clear_m`` and other primitives by ``margin_m``);
7. release: open for ``release_steps`` holding position, then ``settle_steps`` no-op steps; C3 (oracle): the target
   is at rest and passes the agent's own place test (the region box shrunk by ``c3_shrink_m``, or the bowl within
   ``c3_plate_xy_m`` of the plate's origin and resting above it);
8. retreat: up ``retreat_m`` with the gripper open; the agent then finishes (the runner idles).

C1 failure: open, back to the pre-grasp, next frame. C2 failure: lower back near the grasp height, open, rise, replan.
C3 failure: re-grasp from where the target lies. Stall or unreachable: rise, next frame.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import grasp as GR
from .follow import OUT_POS, FollowParams, follow, pose_error

UPV = np.array([0.0, 0.0, 1.0])


@dataclass
class AgentParams:
    pre_height_m: float = 0.10
    pre_tol_m: float = 0.006
    pre_tol_deg: float = 4.0
    transit_tol_m: float = 0.010
    transit_tol_deg: float = 6.0
    reach_tol_m: float = 0.003
    reach_tol_deg: float = 3.0
    approach_cap: float = 0.6
    transit_cap: float = 1.0
    slow_cap: float = 0.3
    rot_cap: float = 1.0
    empty_clear_m: float = 0.015
    transit_margin_m: float = 0.02
    cross_step_m: float = 0.01
    cross_max_m: float = 0.35
    path_step_m: float = 0.02
    lift_min_m: float = 0.04
    close_stop_m: float = 0.0005
    close_window: int = 3
    close_max: int = 15
    c1_band_lo_m: float = 0.006
    c1_band_hi_m: float = 0.004
    c1_min_gap_m: float = 0.003
    c1_seen_lo_m: float = 0.012  # the band's lower edge for a model made from what was seen (``seen.py``)
    rim_width_m: float = 0.012
    c1_rim_min_gap_m: float = 0.0022  # an empty close reads 0.0004-0.0015; held rim walls read >= 0.0033
    c2_rim_rel_m: float = 0.08
    pre_height_min_m: float = 0.05
    pre_top_clear_m: float = 0.02
    lift_m: float = 0.10
    lift_slow_m: float = 0.02
    c2_rel_m: float = 0.01
    c2_lag_rel_m: float = 0.03  # the same limit while the loop has a measured lag (see the lift check)
    # 2026-11-15, off by default: watch the fingers while carrying. When they meet (the gap under
    # ``carry_lost_gap_m`` for ``carry_lost_steps`` steps: an empty close reads 0.4-1.5 mm, the thinnest hold 2.2 mm)
    # the object is no longer in the hand: stop there, open, look and start again. Without it a lost object was
    # found out only by the check after the release, some 60 steps later, with no steps left to try again.
    carry_watch: bool = False
    carry_lost_gap_m: float = 0.0018
    carry_lost_share: float = 0.75  # and under this share of the gap read at the lift check (a thin hold stays a hold)
    carry_lost_steps: int = 3
    # 2026-11-16 (stream DELAY; off by default): through a delay, let go without the wait-and-correct at the release
    # point (a whole lag of steps for a few millimetres; the grasp side keeps its own)
    lag_skip_release_fix: bool = False
    # 2026-11-17 (stream LAG; off by default): through a delay, no wait of a whole lag before the lift check, when the
    # fingers are watched while carrying (``carry_watch``). The wait lets the picture catch up with the hand, so that
    # the look after the lift shows the object lifted. Without it that look shows the hand some steps back, the check
    # passes on what the fingers hold, and an object left behind shows in the fingers during the carry (late by the
    # readings' delay). Stored rows (S20, 24 episodes at start 8): the wait is 15 steps in every grasp
    lag_lift_no_settle: bool = False
    # ``lag_settle_pre`` (2026-11-17, stream LAG; off by default): through a delay the wait-and-correct of the grasp
    # (``_settle_fix``: the lag waited out, the hand read where it truly is and moved onto its point) is done at the
    # pre-grasp point, BEFORE the hand goes down, and not again after. The hand arrives there on its predicted
    # pose, and after a fast move that is centimetres off (stored undelayed episodes: the compensator's prediction
    # 15 steps ahead is a median 4.5 cm from where the hand is then read); going straight down from there it came
    # down on the object: under S20 the descent ended stalled or unreachable in 13 of 32 first attempts (none of
    # 70 plain or D11 ones), and 12 of the 32 closed on an object that had been pushed away (the close's look
    # found it 1 to 4 cm off, or not at all). Six stored S20 episodes played again (the true hand is the reading 7
    # steps on): when the descent began the hand was 3 to 102 mm across from its column, and when it first came
    # under 3 cm above the grasp point 9 to 81 mm (plain: 2 to 5 mm). The wait is taken again (``lag_pre_rounds``
    # times at most) while what it finds is over ``lag_pre_ok_m`` (the correcting move is made on the predicted
    # pose too), and after the last correcting move the commands' own delay is waited (its last commands are still
    # on their way). The closing that follows the descent corrects what the descent leaves. Only when the COMMANDS
    # are late (the lag is longer than the readings' own delay and a step): with readings alone late the hand goes
    # where it is sent (six stored D12 episodes: 0.3 to 10 mm across when it came low), and the wait AFTER the descent
    # is what puts a grasp right when such a lag has ended unnoticed (the hand then stops short of where it was
    # sent: its last commands are counted twice)
    lag_settle_pre: bool = False
    lag_pre_rounds: int = 2
    lag_pre_ok_m: float = 0.01
    # ``lag_close_blind`` (set by the runner when the delay compensator's ``close_hold_cmd`` is on; no option of its
    # own): through a delay the hand closes without reading its fingers, for as long as they take to shut on the
    # planned width (the gap to close at ``close_blind_rate_m`` a step, ``close_blind_margin`` steps more), lifts, and
    # reads the grasp check when the reading of the shut fingers is in. The fingers as the stored episodes show them
    # after a command to close: 0.7, 4.0 and 6.6 mm in the first three steps, then 9.4 mm a step. Steps from the
    # command to fingers at rest, over 95 stored grasps: 10 or 11 on a bowl's rim (8 to 13 mm), 7 at 35 to 37 mm,
    # 6 to 8 at 45 to 48 mm, 4 or 5 at 52 to 62 mm; with these two numbers the hand is given 12, 9, 8 and 6 to 7
    close_blind_rate_m: float = 0.0094
    close_blind_margin: int = 4
    place_clear_region_m: float = 0.02
    place_clear_object_m: float = 0.01
    place_gripper_clear_m: float = 0.005
    margin_m: float = 0.003
    place_step_m: float = 0.002
    lower_slow_m: float = 0.05
    release_steps: int = 8
    settle_steps: int = 5
    rest_m: float = 0.003
    c3_shrink_m: float = 0.01
    c3_plate_xy_m: float = 0.025
    c3_plate_dz_m: tuple = (0.0, 0.04)
    retreat_m: float = 0.10
    max_attempts: int = 5  # the step budget (T_attempt) is the real limit; 3 left 90-190 steps unused
    T_attempt: int = 95  # steps a retry needs to reach its release (success is checked by the env at any step)
    stall_steps: int = 3
    stall_eps_m: float = 0.0005
    stall_retry: bool = True
    stall_retry_max_xy_m: float = 0.02
    stall_retry_rise_m: float = 0.015
    stall_retry_tol_m: float = 0.002
    stall_retry_steps: int = 20
    stall_retry_max: int = 1
    unreach_window: int = 10
    unreach_gain_m: float = 0.001
    unreach_gain_deg: float = 1.0
    max_goto_steps: int = 80
    reach_backoff_m: float = 0.015
    unreach_excl_m: float = 0.03
    near_pre_m: float = 0.015
    near_grasp_max_m: float = 0.01
    near_grasp_margin_m: float = 0.003
    near_z_m: float = 0.01
    open_steps: int = 6
    max_relax: int = 2
    relax_gain_scale: float = 0.6
    freeze_max: int = 40
    respond_m: float = 0.005
    respond_grace: int = 25
    moving_path_m: float = 0.01
    max_damp: int = 3
    min_gain_scale: float = 0.25
    detour_rise_m: float = 0.18  # a transport blocked far from its goal rises this much and crosses again
    detour_min_m: float = 0.06
    # 2026-11-16, off by default: the detour rises above the transport's planned crossing height, not above where the
    # hand happened to stop. With ``transit_cut`` the hand meets an unseen wall on its way UP (truth replay of one
    # S05 episode: at 0.17 m where the old way crossed at 0.23), so the same 18 cm ended 5 cm lower than before,
    # the carried can did not clear the wall, and the hand let go above it. Only after a cut way: the old way
    # (straight up, then across) is unchanged, also where it is stopped while still rising.
    detour_from_cross: bool = False
    # 2026-11-16, off by default: the hand's transits also keep clear of what the last look saw standing tall and
    # could not name (``PerceptionTool.tall_unexplained``): the crossing height and the straight-way test count those
    # boxes as they count the known objects. The Mac depth line, a training task with a slab before the bowl (start
    # state 0): the bowl behind the slab was found, grasped and lifted, the straight carry met the slab and pushed
    # against it for 80 steps before the detour, and the episode ended with the bowl in the hand above the plate.
    # Only a hand that holds something, and only where it helps: when no crossing height clears those things within
    # the arm's reach (a panel hanging over the receiver), the way is planned as before. The grasp plan and the
    # empty hand's moves are as before (a thing beside the target is not in their way by this option).
    transit_unexplained: bool = False
    # (colour line, one Object task: a thing 12 cm tall beside the way was counted in every look of that task, the
    # crossing height stayed what the known scene asked, and only the legs changed; a slab is 13 cm and more)
    transit_unexplained_min_h_m: float = 0.13
    # the reach of a top-down hand above its shoulder as the rows have it (1,273 rehearsal episodes and the Mac's
    # carries: every move that arrived, and the four that were asked higher and did not): the grip site gets as
    # far from the shoulder as 0.78 m level with it and 0.68 m at 27 degrees up (0.31 m up at 0.61 m across;
    # 0.20 at 0.69; 0.10 at 0.77). A crossing asked beyond that, less 1 cm, is not planned.
    transit_reach_m: float = 0.77
    transit_reach_drop_per_deg: float = 0.0037
    transit_reach_near_m: float = 0.35  # (across, from the shoulder: the nearest a move arrived was 0.38)
    # 2026-11-16, off by default (with ``transit_unexplained``): when no crossing height clears such a thing within
    # reach, the carry goes around it: across at the height the known scene asks, by the nearer end of the thing
    # (a point beside that end on the near side, one on the far side), each leg tested as the crossing is. (Truth:
    # a slab 34 cm wide and 43 cm tall dropped between a bowl and its plate; 1.40 m is out of reach.)
    transit_around: bool = False
    transit_around_clear_m: float = 0.005  # (over the transit margin, which the legs' own test already asks)
    transit_around_front_m: float = 0.25  # (a point of the way lies at least this far in front of the shoulder, +x)
    # 2026-11-16, off by default (0): a carry that has all but stopped far from its goal is blocked. The hand's path
    # over the last ``carry_block_steps`` steps is under ``carry_block_path_m``, it has not turned, and its goal is
    # more than 5 cm away across: the move ends "blocked". (A hand pushing along a slab was told "unreachable"
    # after 80 and 100 steps of 220: it kept sliding, which the damping rule takes for an oscillation.) Not
    # through a delay of the commands.
    carry_block_steps: int = 0
    carry_block_path_m: float = 0.03
    # 2026-11-16, off by default: a carry that ended blocked or unreachable far from its goal backs off 3 cm, looks
    # again (the receiver's look: the scene as it is now, with what was put in the way since the first look), and
    # is planned again from where the hand is; once a transport. Then the detour, as before.
    carry_relook: bool = False
    carry_relook_back_m: float = 0.03
    # 2026-11-17 (stream PATH; all off by default; they act only with ``transit_unexplained`` and never through a
    # delay of the commands). Practice rows of the colour line, the two types that put a slab in the way (73
    # episodes): a slab set down as the carry starts is in no look the robot takes before it meets it (eight truth
    # replays of training tasks: it appears 5 to 8 steps after the lift begins), and 15 of 18 solved such episodes
    # bump into it first; the empty hand never counts a slab that every first look shows (12 of 29 failures).
    # ``look_cross``: one look for the way alone, ``look_cross_after`` steps into a carry
    # (``PerceptionTool.look_route``: no percept changes, only the latest look's unnamed regions; no step is
    # spent). A thing it shows on the way that the plan did not count: the carry is planned again from where the
    # hand is. Row mark: ``attempts[].route`` (``why`` "cross", ``s`` seconds, ``new``, ``plan``).
    look_cross: bool = False
    look_cross_after: int = 4
    # ``look_stall``: so many looks a move may take at the first sign of a stall: for ``look_stall_steps`` steps in
    # a row the hand comes less than ``look_stall_ratio`` of its usual way toward where it is sent (its usual way:
    # the most it has done in this move for each metre asked, at least ``look_stall_ref``), more than 5 cm
    # across from its goal. A new thing on the way: the hand backs off ``carry_relook_back_m`` and the move is
    # planned again. Nothing new: the move goes on as it would have. Row mark: ``route`` with ``why`` "stall".
    look_stall: int = 0
    look_stall_steps: int = 2
    look_stall_ratio: float = 0.3
    look_stall_ref: float = 0.15
    # 2026-11-18 (the broad practice pair, 476 episodes: the sign fired 30 times in 16 episodes of the type whose
    # commands arrive 8 steps late, and the looks counted the robot's own arm as a thing in 15% of the looks of
    # the types that add nothing). The sign now asks for a hand that hardly moves at all (under ``look_stall_ratio``
    # of its usual way toward where it is sent, not going back from it, and under ``look_stall_slide`` of its
    # usual way in any direction): a hand that runs on past its goal or against its commands is not stopped by a
    # thing, its commands are late, and from the first such step (``attempts[].route_late``) the attempt takes no
    # look at a stall. And a look for the way leaves the robot out in the PICTURE (``look_self_px``: its model at
    # the read joints drawn into the frame, each sample grown so many pixels; the server's own filter is 1 cm in
    # space, and the colour line's depth of the arm is off by more); the way then counts what the latest such look
    # shows and no older look's things. A look in which the things that stand still read as moved (the camera
    # has moved, or the look is off) is not used (``route[].shifted``).
    look_stall_slide: float = 0.8
    look_self_px: int = 10
    look_route_max: int = 12
    # (2026-11-18, after the second pod pair's slab episodes; both True is that pair's arm as it ran.)
    # ``look_route_only``: once a look for the way was had, the way counts that look's things alone. A look taken in
    # the carry reads a slab that stood there from the start 5 to 7 cm lower than the first look did in 4 of 8 such
    # episodes, and not at all in 4 of 8 others. False: the first look's things and the latest look's, as before that
    # pair (for a server that leaves the robot out of every full look in the picture, ``full_self_px``).
    # ``look_start``: the empty hand's way starts with a look for the way of its own. False: it is planned on the
    # looks there are.
    look_route_only: bool = True
    look_start: bool = True
    # ``route_empty``: the empty hand's way to the point above its target counts the tall unnamed things as the
    # loaded hand's way does. On the way only: a thing the hand would touch at that point itself is the grasp's
    # business and is left out. Row mark: a ``transit`` event with ``unexplained`` and ``empty``.
    route_empty: bool = False
    # ``detour_seen``: a carry stopped far from its goal looks, and its detour crosses at the lowest height that
    # clears what the looks show, within the arm's reach (not ``detour_rise_m`` above where it stopped); around
    # the thing's nearer end with ``transit_around`` when no such height exists; and when nothing seen explains the
    # stop, the fixed detour, no higher than the arm reaches (six of the 29 failures ended with the arm caught at
    # the edge of its reach, elbow at its limit, after a push or a detour far out). Row mark: the ``detour``
    # event's ``seen`` ("over", "around", "none"), and ``capped`` when the reach lowered it.
    detour_seen: bool = False
    # item 3 (2026-11-14), off by default: one look with the wrist camera at the pre-grasp pose
    # (the target) and one above the place (the receiving object), each a track look from the predicted pose; the grasp
    # or the release point moves by what it measured (the server accepts moves up to its track_max_jump_m, 5 cm)
    close_look_grasp: bool = False
    close_look_place: bool = False
    close_min_m: float = 0.002  # a measured move below this changes nothing
    close_looks_max: int = 10  # close looks per episode
    # item 5 (2026-11-14), off by default: a fresh full look when the target's track is refused with the hand open and
    # nothing held (the look tool's ``redetect``): the robot no longer grasps the place where the target was
    redetect: bool = False
    # long-task suite (2026-11-14), off by default: when another object already lies in the target region (a second
    # object into the same basket), try a grid of release points across the region and take the one where the held
    # object goes lowest (nearest the centre on a tie), instead of the region's centre
    place_free_xy: bool = False
    place_free_n: int = 5  # grid points per axis
    # opening a part (``drawer.OpenAgent``): end the pull when the part stops coming instead of waiting out the
    # no-progress window (about 25 steps saved; matters when another step follows in the same episode)
    pull_stall_stop: bool = False
    # a placement that is one step of a longer plan (staging): once released and settled it counts as placed; the
    # agent's own place check is recorded but does not send it back for another try
    trust_place: bool = False
    # when the come-down above the goal is itself not clear of the inflated obstacles at any height (a grasp next to a
    # fixture), rising further gains nothing: cross at the lowest height at which the crossing is clear, instead of the
    # maximum (2026-11-15: 35 cm up and down again, about 70 steps, in 3 of 10 plain starts of one task)
    cross_descent_free: bool = False
    # opening a part, when another step follows: straight to the point in front of the handle with the gripper shaped
    # on the way, a shorter and faster approach, the pull at the full cap (about 50 steps saved)
    open_fast: bool = False
    # the receiving object is looked at again with the hand above the release point, and once more after the hand has
    # backed off (2026-11-15: it was measured once, in the first look; a receiver the scene moved
    # since got the object set down where it had been). Moved more than ``relook_min_m``: before letting go, the release
    # is planned again and the hand goes there; after, the attempt counts as failed and the next one takes the object
    # to where the receiver is
    relook_place: bool = False
    relook_min_m: float = 0.02
    # an empty hand that starts more than 2 cm lower than its goal goes straight up to the goal's height before it
    # crosses (see ``_move_via``)
    rise_first: bool = False
    # 2026-11-16, off by default (D03: while the target is carried, a cover is put on the
    # receiver; the release then ended on the cover in every episode). A last descent that is stopped from below,
    # straight above its goal and between ``cover_min_m`` and ``cover_max_m`` short of it, with the object still in
    # the hand, means something lies on the receiver. The hand goes back to one side, comes down beside the thing
    # and pushes it across and off, along the gripper's closing axis (so a finger backs the object that pushes); then
    # it returns and lets go as planned. Once per episode (``_clear_cover``).
    clear_cover: bool = False
    # ordinary releases that stop short stop under 1.3 cm short (stored rows, both lines); a small thing on a plate
    # stops the bowl 2 cm short, a cover on the basket 4 to 12 cm short
    cover_min_m: float = 0.015
    cover_max_m: float = 0.13
    # a release ON a surface region is planned 2 cm above it, so a flat thing up to 2 cm thick lying there stops
    # nothing (2026-11-16: covers of 1.5 and 2 cm on a stove and on a cabinet; the object was set down on them). With
    # ``clear_cover`` the hand then goes on down slowly until the object touches: touching ``cover_touch_m`` or more
    # above the surface means something lies there. Touching the surface itself, the object is let go where it rests.
    cover_touch_m: float = 0.012
    # an object held by its rim hangs from the fingers like a hinge: set down on something it tilts and the hand
    # goes on down, so nothing is felt in the arm (2026-11-16, a bowl on a block on the plate). But its wall turns
    # between the fingers, and their gap opens: measured on the Mac's depth line, two bowls set on the block read
    # 10.5 and 10.8 mm against 3.9 and 6.7 mm at the lift check; a bowl let go in the air above the plate read 3.1
    # against 3.2 mm. A gap ``cover_gap_m`` wider at the end of the descent than at the lift check means the object
    # rests on something; the hand rises until it hangs free again (the height of the touch), and the push is planned
    # from there. Only with ``clear_cover``; 0 turns it off.
    cover_gap_m: float = 0.002
    cover_xy_m: float = 0.02
    cover_over_m: float = 0.03  # the thing is taken to reach this far past the receiver's edge
    cover_past_m: float = 0.05  # the pushing face never goes further than this past the receiver's far edge
    cover_clear_m: float = 0.02  # and ends this far past where the held object's own far side will be when set down
    cover_under_m: float = 0.035  # the pushing object's lowest point: this far below the thing's top, never lower than
    cover_skim_m: float = 0.006   # this far above what the push passes over
    cover_grip_m: float = 0.010   # and at least this much of the thing is taken, whatever height the receiver is read at
    cover_rise_m: float = 0.03
    cover_cap: float = 1.0
    cover_need_steps: int = 60
    # 2026-11-16 night, all off by default (the pod's eight Spatial episodes of the cover case, a 4.6 cm block on the
    # plate; the push ran in four and one succeeded):
    # ``cover_touch_descent``: where a rim-held object touched is read from the descent itself, the hand's height
    # between the last step with the fingers' gap as it was and the first with it opened. Rising until the gap
    # closes again reads it only when the wall turns back between the fingers: in four of the eight it did not, the
    # rise ran its whole 6 cm, and the block was taken 2 to 3.6 cm higher than it is (the push then ran at its top).
    cover_touch_descent: bool = False
    # ``cover_skim_rim_m`` (0: ``cover_skim_m``): how far above the receiver a rim-held object's lowest point passes
    # in the push. Held by its rim it hangs lower than its model on the side away from the fingers, and a receiver
    # that can be moved is dragged along by a touch: with the touch read right (two of the eight) the bowl ran 1 cm
    # above the plate's floor, and the placed check then found the bowl outside the plate.
    cover_skim_rim_m: float = 0.0
    # ``cover_relook``: after the push the receiver is looked at again before the object is set down (it may have been
    # moved by the push), and a gap that opens again on the way down is recorded (the thing is still there).
    cover_relook: bool = False
    # 2026-11-16 night, second round, off by default (truth replays of those eight episodes: in none did the block
    # leave the plate. It is longer than the plate's floor and lies on the sloped rim all round, so it does not slide:
    # a push takes the plate along from its first step, 7 and 10 cm with the block still on it; the plate weighs 12 g,
    # the block 28 g):
    # ``cover_slow_cap`` (0: ``cover_cap`` all the way): the speed cap of the push from the receiver's near edge on,
    # for a rim-held object over a receiver that can be moved. A thing that cannot slide is tipped over its far edge
    # by a push at its top, and that takes less force than moving the receiver under it; a fast hand gives whatever
    # force stops it, and both give way at once.
    cover_slow_cap: float = 0.0
    # ``cover_under_rim_m`` (0: ``cover_under_m``): how far below the thing's top a rim-held object's lowest point goes
    # in the push over a receiver that can be moved.
    cover_under_rim_m: float = 0.0
    # ``cover_look``: before the push the hand goes to one side of the receiver, out of the main camera's line, and
    # the robot looks once at what lies on the receiver (``look.on_receiver``; the server's ``extra_on_top_m``). A thing
    # longer one way than the other is pushed across its long side: the hand turns about the vertical when its
    # closing axis is more than ``cover_turn_min_deg`` off that, and turns back before the object is set down. (Of
    # the six pushes in those episodes the one that ran across the block's long side turned it over, a quarter turn
    # toward the rim, and the plate stayed; the three that ran along it took the plate with them.) For a rim-held
    # object over a receiver that can be moved; nothing seen there, or a thing as long as wide: the push as before.
    cover_look: bool = False
    cover_turn_min_deg: float = 30.0
    wrist_turn_max_rad: float = 2.6  # (the Panda's last joint turns 2.90 rad either way)
    # ``cover_pick`` (2026-11-17, off by default): a thing felt on a receiver that can be moved is taken off by hand.
    # The held object is carried to a free place beside the receiver, out of the main camera's line to it; from
    # there the robot looks at what lies on the receiver (``look.on_receiver``, the server's ``extra_on_top_m``).
    # A thing narrow enough for the fingers (its short side at most ``cover_pick_max_m`` as the look reads it) that
    # stands ``cover_pick_min_h_m`` or more above the receiver and is no object the robot knows: the held object is
    # set down where it is, the thing is taken top-down across its short side, carried off the receiver and let go
    # low over free table, and the attempt ends "parked" (the next attempt takes the object again from where it
    # waits, and may start with ``cover_retake_need`` steps left). Anything else seen, or nothing: the hand goes
    # back over the release point and the push runs as before. With the option on, the descent of a rim-held
    # object ends at the step its fingers' gap opens (``_watch_touch``). Why: the Spatial suite's cover is a block
    # of 4.7 x 4.7 x 6.7 cm wedged across the plate's rim; pushed, it takes the 12 g plate with it (truth replays
    # of 2026-11-16 night: the block left the plate in none of eight episodes, whatever the push).
    cover_pick: bool = False
    cover_pick_need: int = 125  # steps that must be left when the thing is felt (aside and set down, the thing,
    # and the attempt that takes the object again)
    cover_see_need: int = 125  # and when it is seen from afar (the object is set down on its way)
    cover_open_steps: int = 3  # the hand opens this long before it moves off what it let go
    cover_retake_need: int = 50  # steps the attempt that takes the waiting object again needs (it stands beside
    # the receiver: ``T_attempt`` counts a way across the table)
    # ``cover_retake`` (with ``cover_pick``; off by default): once the thing is off, the hand goes straight back to
    # where it let the object go, takes it by the same hold (the fingers' gap must pass the close check of the grasp
    # that took it), carries it over the receiver, looks at the receiver once more (taking the thing off may
    # have moved it) and sets it down as the first plan had it. No new plan and no new attempt: the attempt that
    # takes a waiting object again took 69 steps on the Mac (33 of them to its close), and 220 do not hold it
    # after a cover seen later than step 75. When the hold does not read right the hand opens and the next
    # attempt takes the object as before.
    cover_retake: bool = False
    cover_wrist_max_rad: float = 2.8  # the turn for the thing keeps the wrist's last joint inside this
    cover_close_early: bool = True  # the fingers begin to close in the last centimetre of a descent onto a hold
    # a rim-held object that comes down on something turns in the fingers before their gap opens by ``cover_gap_m``:
    # on the Mac a bowl set on the 3.4 cm block showed it with the hand 1.8 cm lower than where it touched (the
    # reading was 1.55 cm above the surface, 3.5 mm over ``cover_touch_m``). The touch read at that step is taken
    # this much higher.
    cover_gap_lag_m: float = 0.015
    cover_pregap_m: float = 0.035  # the fingers' gap kept on the way back to the waiting object (its wall and 3 cm)
    cover_turn_slack_deg: float = 15.0  # the fingers may stand this far off the thing's short side
    cover_pick_turn_min_deg: float = 8.0  # no turn for less
    cover_pick_max_m: float = 0.068  # the thing's short side as the look reads it (the fingers open 8 cm)
    cover_pick_long_m: float = 0.16  # and its long side
    cover_pick_min_h_m: float = 0.022  # (the fingers need 1 to 2 cm of it; a depth error is not a thing)
    cover_pick_min_px: int = 1000  # (the 4.7 by 6.7 cm block reads 6,200 to 6,600 pixels on both lines)
    cover_pick_depth_m: float = 0.03  # how far below the thing's top the fingertips go
    cover_pick_recv_max_m: float = 0.20  # receivers up to this size across (a plate, a bowl)
    cover_park_gap_m: float = 0.02  # between the waiting object and the receiver
    cover_park_clear_m: float = 0.006  # the waiting object's lowest point above the support when the hand opens
    cover_park_rise_m: float = 0.045  # the open hand rises this far off the waiting object before it moves across
    cover_spot_clear_m: float = 0.015  # a waiting or a dropping place is this far clear of every known thing
    cover_drop_out_m: tuple = (0.05, 0.08, 0.12)  # the thing's middle is let go this far outside the receiver's edge
    # ``cover_see`` (with ``cover_pick``; off by default): the receiver is looked at during the carry, while the held
    # object is not over it yet: when the hand first comes within each of ``cover_see_m`` of the receiver's top
    # (in a straight line), as long as the held object's near edge is still ``cover_see_clear_m`` short of the
    # receiver's middle. A thing the fingers can take seen on it: the object goes straight to the nearest free
    # place beside the receiver and the thing is taken off, without first coming down on it (felt only at the end
    # of the descent, the touch and the way aside cost 20 to 50 steps of a 220-step episode). Why two distances:
    # the Spatial suite's cover is put on the plate during the carry, in eleven truth replays when the hand was
    # 0.17 to 0.23 m from it (and the object had left its place by some 6 cm: a bowl that starts beside the plate
    # has the cover arrive under it, and only the touch tells).
    cover_see: bool = False
    cover_see_m: tuple = (0.19, 0.15)
    cover_see_clear_m: float = 0.03
    # ``stall_unseen_m`` (0: off): a target whose HEIGHT was never seen (found by the wrist camera's colour picture,
    # or set behind what hides it: its place across the support is a look's, its height an assumption, the support
    # it was found over) is taken where the open hand stops on it. An approach stopped straight above its goal by
    # up to this much is the grasp at that height, and the target's stored pose is raised by as much (the hidden
    # bowl of the panel case stands on a box, a ramekin or a stove in three Spatial tasks).
    stall_unseen_m: float = 0.0
    # 2026-11-16, all off by default (what the rows of one run showed; the depth line on the same episodes had
    # two such lift checks and eleven such grasps):
    # ``c2_trust_gap``: the lift check goes by the fingers. On the colour line the look at a lifted object read it
    # 1.5 to 4.7 cm from where it sat in the hand (the net's depth of a thing in the air) while the fingers held the
    # same width to within 2 mm: 60 lift checks failed that way in 48 of 449 episodes, 17 of them lost, each one
    # an object set down again and a grasp wasted. With the fingers still on the width they closed on (within
    # ``c2_gap_keep_m``), the look fails the check only when it shows the object left behind: moved against the
    # hand by ``c2_left_share`` of the rise or more (the false readings reached 0.48 of a 10 cm rise; an object
    # left on the table reads 1.0). The in-hand pose measured at the close then stands for the release.
    c2_trust_gap: bool = False
    c2_gap_keep_m: float = 0.003
    c2_left_share: float = 0.6
    # ``release_here``: the look tool is told where the hand lets go of a target it did not carry away (an empty
    # close, a failed lift, no place). The tool took the target's pose from the hand at the NEXT look, by which time
    # the open hand had risen to the pre-grasp point: the target was stored 5 to 10 cm above where it stood, and
    # when the merged look found no percept that near, the next grasp closed on air where the hand had been (41 of
    # 449 episodes, 26 of them lost; grasp heights like 922, 983, 1023, 1145 mm in one episode).
    release_here: bool = False
    # ``empty_view_m`` (0 = off): after a close on nothing the target is not where the look put it. The next
    # attempt's first look is taken from the pre-grasp point as before; when that look does not find the target
    # anywhere else (it stands within ``empty_same_m`` of where the hand just closed on nothing), the open hand
    # rises this far above the grasp point and looks once more from there (the look tool's re-detection on for it),
    # and the attempt plans on that. In the rows of the case that moves the target 7.5 cm before the hand arrives,
    # the look from the pre-grasp point found it in 4 of 6 episodes on the depth line and 3 of 6 on the colour line;
    # the colour line also missed it from 11 to 20 cm up (3 looks) and found it from about 30 cm (1 look). On the
    # Mac (depth line, 2026-11-16) the look from 28 cm found it where the look from the pre-grasp point had not,
    # and rising at once cost 36 steps in an episode where the first look would have done. 0.28 is about the
    # height the episode's first look is taken from; about 20 steps when it is used.
    empty_view_m: float = 0.0
    empty_same_m: float = 0.02
    # ``held_pose_fix``: the look tool's in-hand prediction no longer counts the hand's move twice after a refused
    # track (``look.PerceptionTool._carried``). It changes where a target is believed to be after a lift whose
    # look was refused: when it is set down again, when it is released and the merged look does not find it, and
    # after a lost carry.
    held_pose_fix: bool = False
    # ``open_wait`` (0 = off): an attempt after the first takes its first look only once the hand is open (the gap
    # within ``open_wait_gap_m`` of fully open), waiting up to this many steps. The fingers need 10 to 14 steps to
    # open; the look came 6 to 12 steps after the command, and while the hand still reads closed the look tool
    # takes the target for held: no merged look is taken and the target's pose goes with the hand (2026-11-16, the
    # looks of four Mac episodes: after a lost carry, a failed lift and a close on nothing the next look was a
    # refused track from the hand's place, the grasp after it closed on air there, and only the look after THAT was
    # the merged one that found the target 51 to 59 cm away).
    open_wait: int = 0
    open_wait_gap_m: float = 0.004
    # ``pre_look``: one look at the target from the pre-grasp point, before the descent. A target the scene has
    # moved since the attempt's first look (the tracker refuses it, and the merged look that follows finds it
    # ``pre_look_min_m`` or more away) is not grasped at where it was: the attempt ends there and the next one plans
    # on where it is. Replayed 2026-11-16: the case that moves the target does it when the hand comes within 11 cm,
    # 8 to 20 steps before the close, and every such episode closed on nothing first (30 to 40 steps). At most
    # ``pre_look_max`` such restarts an episode; a smaller measured move changes nothing.
    pre_look: bool = False
    pre_look_min_m: float = 0.04
    pre_look_max: int = 2
    # ``empty_next``: a close on nothing at a pose that was only a fallback (no percept of the named type; a rejected
    # fit or an unexplained region stood in for it) refutes that place: the look tool gives the target its next
    # candidate (``refute_target``) and every grasp frame counts as untried there.
    empty_next: bool = False
    # ``c1_soft``: a hold of another width than the record's. The fingers stopped on something and stayed there,
    # between ``c1_soft_lo`` and ``c1_soft_hi`` of the predicted width (an object drawn or shaped otherwise than its
    # stored model, or held on another chord): the grasp check lets it through and the lift decides, by the fingers
    # (the width band no longer applies). Not for rim grasps; never a hand that did not close (a gap within
    # ``c1_soft_open_m`` of fully open) nor one that closed on nothing (under ``c1_soft_min_m``).
    c1_soft: bool = False
    c1_soft_lo: float = 0.6
    c1_soft_hi: float = 1.35
    c1_soft_min_m: float = 0.006
    c1_soft_open_m: float = 0.006
    # ``c1_stand`` (2026-11-16 night, off by default; with the look tool's ``fallback_stand``): a target that is a
    # model stood on an unexplained region has no measured turn about the vertical (a bottle under its round cap).
    # Until a look measures it, the grasp check also takes a steady hold as wide as the model's OTHER side across:
    # the fingers then hold the thing a quarter turn from where the model was stood. The lift decides as always.
    c1_stand: bool = False
    # 2026-11-16 (two objects into one container, the long suite), off by default: an object let go above its place
    # falls. When its lowest point is more than ``drop_min_m`` above the surface under it at the release (a basket
    # whose walls keep the hand at the rim: 12 cm), the look tool is told how far it will fall, so the object's
    # stored pose is where it lands and not the point in mid-air where the hand let go. Without this the track
    # after the release was refused (a jump over its limit), the placed check read "at rest, not inside", and the
    # next attempts closed on air at the release point (one run: 39 of the 63 two-object episodes).
    # ``drop_settle_steps``: more steps before the check's reads after such a release (the fall and the rocking).
    drop_predict: bool = False
    drop_min_m: float = 0.03
    drop_settle_steps: int = 6
    # 2026-11-16 (night), off by default: the fall is taken from where the hand really has the object. The look
    # tool's in-hand prediction counts the hand's move twice once a track of the held object was refused or not
    # believed (its ``_predict``; the usual case at the lift on the colour line), so the pose the fall was taken
    # from stood a lift height too high and the landing pose came out near the release point again. The candidate on
    # the long suite's two-object tasks (colour): the can that is task 7's first object was let go 34
    # times; the check read it within 1.5 cm of the basket's rim height in 32, failed 26 times, and 23 of those were
    # followed by a close on air. The flat box that is task 1's first object falls further (12 cm against 8), read
    # 3 cm under the rim and passed 25 of 26 times: the same error, on the lucky side.
    drop_rigid: bool = False
    # 2026-11-16, off by default: a placed check that fails only on "at rest" (the object reads inside its place) is
    # read again after the hand has backed off, ``c3_recheck_steps`` reads one step apart. The colour line's track
    # jitters while the hand is close: the first moka pot standing on the stove read 3.7 to 5.8 mm between reads
    # in 7 of 10 plain starts (5.8 mm in start state 0, the scored one) against a limit of 8, and at a limit of
    # 3 the robot went back for a pot that stood where it belonged (5 of 10). A sliding object still reads as
    # moving, or ends outside.
    c3_recheck: bool = False
    c3_recheck_steps: int = 3
    # 2026-11-16, off by default: a "placed" verdict needs a measurement. After a release the look tool searches for
    # the object (one full look) and tracks it. When that look does not find it and every track is refused, its
    # stored pose is still the hand's own prediction of where it let go, and the placed check passes on it wherever
    # the hand opened (the bundle on the untouched half: 3 of 15 "placed" endings the benchmark did not count rested
    # on nothing else; truth: a bottle lying 17 cm from the cabinet, a bowl 19 cm from it, a can on a cover). With
    # this on such a pass is not taken: with the hand backed off, one full look searches for the object at any
    # distance and the check is read again (``c3_measured_reads`` reads, one step apart). No look measures it, or
    # the measured pose fails the check: the attempt has failed, and the next one starts from where the object
    # was found, while steps remain. (A pass on a measured pose is taken as before.)
    c3_measured: bool = False
    c3_measured_reads: int = 2
    # 2026-11-16, off by default (0): a held object does not jump inside the hand. A track taken while the hand holds
    # the target that puts it further than this from where the hand's own motion puts it is not believed (the look
    # tool's ``held_jump_max_m``). The colour line's net reads depth badly around a closed hand: with the same
    # grasp, the book of the long suite's book task hung 3.5 cm aside and 1.5 cm lower in the hand at start state 0
    # than on the depth line, the release went 3.5 cm off, and the task ended 15 steps after the release instead
    # of 7; and 61 of 446 lift checks read "moved in the hand" by 1 cm or more with the gap unchanged, against 2
    # of 416 on the depth line (the same episodes). Never for a rim grasp: a bowl held by its rim swings between
    # the jaws as it is lifted, by centimetres (``c2_rim_rel_m``), and that move is real
    held_jump_max_m: float = 0.0
    # 2026-11-16 (night), off by default (0): objects asked ONTO one surface keep their own parts of it. The second
    # of two takes its planned part (``_share_offset``: the corner opposite the first) whenever the hand can let go
    # there within this height of the lowest release, instead of the point of the whole fixture top where it could
    # let go lowest (``place_free_xy``'s search, made for a second object INTO a basket). Plain two-pots task at the
    # scored start, 30 scored seeds, colour line: in 27 the search ended within a centimetre of the planned corner;
    # in seeds 158 and 167 (both lost, and the only two lost) and 156 the second pot's lowest point read 2 cm lower
    # in the hand (``held_tilt_max_deg``), and in the two lost ones the lowest release was not at the planned corner:
    # at the region's far edge beside the first pot (158), and off the burner on the stove's lower top, 11.6 cm
    # from the region's middle (167). The pot was let go there and the task never ended. When the planned part is
    # blocked higher than this, the search stays on the region and takes the point furthest from the other objects
    # among those within 2 cm of the lowest.
    share_keep_up_m: float = 0.0
    # 2026-11-16 (night), off by default (0): a held object does not tip inside the hand either. A track taken
    # while the hand holds the target that tips its upright axis more than this many degrees from where the hand's
    # own motion puts it is not believed (the look tool's ``held_tilt_max_deg``; never for a rim grasp, as
    # ``held_jump_max_m``). A turn about the vertical is left alone: a box taken across a diagonal does square up
    # between the jaws. In three of the 30 scored seeds of the plain two-pots task (156, 158, 167; the same grasp
    # and the same physical start as the other 27) the second pot's lowest point read 2.0 cm lower in the hand.
    # That is what a pot turned upside down at the right place gives (8.6 cm long under its origin turned over,
    # against 6.6; a moka pot is nearly the same top and bottom) and no smaller tip can (1.2 cm at most): the
    # kinematic stand-in gives the same 2.0 cm with such a track (test_long2_options), and none with this limit.
    # The rows did not keep the rotation; the look log does from now on (``held_tilt_deg``).
    held_tilt_max_deg: float = 0.0
    # the lift check's lower gap edge, when set (0: the close check's, ``c1_band_lo_m``). A box taken across a
    # diagonal turns flat between the jaws as it is lifted and the gap ends at its true width, up to 7 mm under
    # the record's (one run: 17 second objects failed the lift check by about 1 mm while they came up with the
    # hand, 0.4 to 1.1 mm of relative move)
    c2_band_lo_m: float = 0.0
    # 2026-11-16, steps per attempt (all off by default; one-attempt successes took a median 107-118 steps of a
    # 220-300 cap, so a second attempt rarely fitted, and the drawer compound used 265-295 of 300):
    # ``transit_cut``: the rise-cross-descend way with its corners cut. The hand climbs on a slant to the farthest
    # point of the crossing line it reaches in a clear straight move, comes down on a slant where that is clear, and
    # passes the points in between within ``transit_blend_m`` without stopping (each cut is checked like the crossing).
    transit_cut: bool = False
    transit_blend_m: float = 0.03
    # ``lift_fast``: the lift in one move (its first ``lift_slow_m`` at the approach cap, without a stop at its end),
    # and no stop at the lift height when the transport rises further anyway (the look for the lift check is taken
    # on the way up)
    lift_fast: bool = False
    # the last move down to the release point, onto a surface or an object (not into a container): its cap and
    # tolerance (0: the approach cap and the reach tolerance)
    lower_cap: float = 0.0
    lower_tol_m: float = 0.0
    # ``close_early_m`` (0: off): the fingers start to close while the hand is still on its way down to the grasp:
    # within this distance above it, no more than three steps from it at the speed it is coming down, and over it
    # to within 4 mm. They need 10 steps from fully open to a thin wall, so they still meet the object after the
    # hand has arrived. Only for a grasp at least 5 cm narrower than the open hand (a rim; never a wide box, which
    # the fingers would reach first)
    close_early_m: float = 0.0
    # opening a part: the pull ends once the hand has come the part's own travel less 5 mm, instead of pulling
    # 1.5 cm past its end and waiting four steps for it to stop coming
    pull_end_early: bool = False
    # ``brake_k`` (0: off): a move that must end within 1.2 cm aims short of its target by this many steps of the
    # hand's own sideways (horizontal) movement, and so does the hold while the fingers close. The command follows
    # the error, but sideways the hand follows the command slowly (measured 2026-11-16: about a third of the
    # difference a step; up and down it follows within two steps), so a hand that came in sideways at 10-12 mm a
    # step passed its target by 2 cm and swung back and forth over some twenty steps: the 5 cm down to a grasp took
    # 15 steps instead of 7, the hand was still swinging 7 mm either way while the fingers closed, the way to a
    # handle took 15 to 24 steps. ``brake_kz`` is the same for its vertical movement (0: none; it needs none).
    # Moves that only pass a point on the way are left as they are. Keep it at 2 or under: a hand that followed its
    # command at once would swing ever wider above 2.5 (1.5 damps a slow hand well and still settles a quick one)
    brake_k: float = 0.0
    brake_kz: float = 0.0
    brake_tol_m: float = 0.012
    # ``cap_box``: on a transit move (to the point above a grasp, to the point above a release) the position command
    # keeps its direction and is capped by its largest component, not by its length. The old cap clips every axis
    # first, so far from its target the hand moves along the diagonal of the signs until one axis arrives and then
    # bends: not the straight line the planner tested for clearance, and slow (measured 2026-11-16: 12.5-13.3 mm a
    # step in all, so the axis with farthest to go crawls at 7 while three axes are far). With it the hand stays on
    # the tested line and its longest axis runs at the full 12.5 mm a step. Moves down to a grasp or a release are
    # left as they are (there each axis closing its own small error at once is what is wanted)
    cap_box: bool = False
    # a staging release (``trust_place``): the transport ends this far above the release point instead of
    # ``lower_slow_m`` (0: as before). That stop 5 cm up is where the receiver is looked at again, and a staging
    # release takes no such look
    stage_above_m: float = 0.0
    # 2026-11-16, off by default: the look tool is told when the hand lets go after a grasp, a lift or a carry that
    # failed (``look.hand_opened``). Without it a look taken before the fingers read fully open (about 13 steps; a
    # failed attempt opens for 6 and goes on) still counts the target as held, and the next grasp is planned at the
    # hand, in the air
    tell_open: bool = False
    # a planned step that asks for the hand on one side of the object (``grasp_side``: the staging before a drawer
    # is opened) chooses among the planner's own first 8 frames, and takes the opposite side when none of them is on
    # the asked one. With this set (the number of frames the planner is then asked for; 0: as before) the asked
    # side's frames are looked for further down the planner's list first. (2026-11-16, colour line, the plain
    # drawer task at the scored start, ten runs: the staging held the bowl by its far rim in a third of the starts,
    # with ``[0, 6, 2]`` frames on the asked, opposite and flank sides and nothing in the way; after it the approach
    # to the handle stopped 1.6 to 4 cm short in 23 of 90, against 1 of 178 after a staging by the near rim, and the
    # take-again was solved in 27 of 52 against 133 of 133)
    side_wide_k: int = 0
    # taking a staged object again (``drawer.RetakeAgent``): the carry into the opened part ends within this of its
    # goal (as built: 2 cm, "the opened part is wider than the object by more than that on every side"; along the
    # drawer it is not: a bowl has 2 cm either way, the hand stopped 1.7 cm short, and when it held the bowl by
    # its far rim the bowl's back edge came down on the cabinet's front edge: 2 of 4 such scored starts lost)
    take_carry_tol_m: float = 0.02
    # 2026-11-16 night, the opening skill (``drawer.OpenAgent``); all three off by default:
    # ``open_gap_check``: jaws that reach the point before the handle too far shut to take the bar are opened and
    # go in open (late commands or readings spoil the shaping: the first attempt closed on nothing in every S20
    # and D12 episode of the opening task).
    open_gap_check: bool = False
    # ``open_settle``: through a delay, the approach is finished on the hand's true reading before the fingers
    # close (``_settle_fix``; S20: the true hand stood 1.6 to 2.1 cm short of the handle and 2.5 cm aside).
    open_settle: bool = False
    # ``open_relook``: an attempt that found no handle (the fingers met, or the hand could not get near) is
    # followed by one look at the fixture and a new plan from it, or by the fixture's half-turn twin, never by the
    # same grip again; one attempt more is allowed (D01: the scene moved the cabinet 13 cm at step 32 and it was
    # reached for three times at its old place; S01: a back-to-front fit was reached for three times).
    open_relook: bool = False
    # 2026-11-17, the pull (``drawer.OpenAgent``); all three off by default:
    # ``open_pull_straight``: the pull's command keeps its direction (see ``OpenAgent._pull``).
    open_pull_straight: bool = False
    # ``open_pull_full``: the hand pulls the part's whole travel from where it took the bar, whatever the look read
    # of the joint.
    open_pull_full: bool = False
    # ``open_slip_check``: fingers that end the pull more than 4 mm under the bar's height apart have not opened the
    # part; with ``open_relook`` the part is then looked at again (its joint) instead of guessed from the hand's travel.
    open_slip_check: bool = False
    # ``open_verify``: when the opening is the instruction's last step and the episode goes on three steps after an
    # "opened", the part is not open as far as the task wants: it is looked at again (``open_relook``) and taken for
    # the rest of its way, once.
    open_verify: bool = False
    # ``open_feel`` (2026-11-17, off by default; never through a delay; with ``open_fast``): after an attempt whose
    # fingers met on nothing, the next one finds the part's front by touch. The jaws come wider than usual (4.5 cm
    # more than the bar is high, less where the handles above and below leave no room), the hand goes in
    # along the part's own way until it stops coming (the fingertips on the part's front, at most
    # ``open_feel_in_m`` past the planned grip point), backs off by the model's distance from that front to the bar
    # less the fingertip's length, and closes there. How deep the handle stands is then the hand's to find, not the
    # look's: a fixture drawn at another size, or standing a centimetre or two from where it is remembered, has its
    # bar that same distance before its own front (2.4 cm on the cabinet; a fifth more or less is 5 mm).
    # ``open_feel_first``: the same on the FIRST attempt when the fixture is not one the look measured (a remembered
    # place, a rejected fit, a region): a miss costs 45 steps, the touch about 15.
    # A pull that follows a touch, or a pull that lost the bar, goes at half the speed and starts at a third of
    # that; after a lost bar the touch looks for the front from where the hand let go back to where it took it.
    open_feel: bool = False
    open_feel_first: bool = False
    open_feel_in_m: float = 0.06
    # ``open_reopen_cmd`` (2026-11-17, stream LAG; off by default): jaws that came out too far shut are opened for
    # as long as they take plus the COMMANDS' own delay, not the whole lag (the readings' delay is not waited for:
    # nothing is read before the hand goes in). With the delay compensator's ``open_hold_cmd`` the hold before it
    # is the commands' delay too: under S20 22 steps where it was 38
    open_reopen_cmd: bool = False
    # 2026-11-17, the opening step's own steps (``drawer.OpenAgent``). The plain drawer task (stage the bowl, open,
    # take it again) used 240 of its 300 steps, 81 of them here: 35 to the point before the handle (a quarter turn
    # of the hand, 21 steps, then 13 more creeping the last 8 mm up to that point), 10 in to the handle, 9 closing,
    # 15 pulling, 4 letting go, 6 backing off, 1 idle. All off by default; none acts through a delay.
    # ``open_pre_pass``: the point before the handle is passed, not stopped at: the move in begins once the hand is
    # turned to within ``open_pre_deg`` and stands within ``open_pre_lat_m`` of the line in to the grip point (across
    # it and in height: the jaws go in 1.7 cm clear of the bar either side), 3 cm or more before the grip point.
    open_pre_pass: bool = False
    open_pre_lat_m: float = 0.010
    open_pre_deg: float = 6.0
    # ``open_pre_out_m`` (0: 6 cm as built): how far before the grip point that point lies. Only with
    # ``open_out_first``'s way down, which answers the turn's swing toward the bar in full. (With the caps by
    # length or by the largest component a point 4.5 cm out brought the fingers within a millimetre of the bar:
    # Mac, seed 142.) A hand that comes down 4.5 cm out need not turn back from 6 to go in.
    open_pre_out_m: float = 0.0
    # ``open_close_early_m`` (0: off): the fingers begin to close when the hand is within this distance of the grip
    # point on its way in (they take five steps to reach the bar; the hand's last centimetre takes four), and the
    # hold is taken as made once the gap has stopped within 4 mm of the bar's height (as built: after three more
    # steps of an unchanged gap).
    open_close_early_m: float = 0.0
    # ``open_release_early`` (0: off): the open command is sent this many steps before the pull's end (the fingers
    # begin to part three steps after the command; with 2 the bar is held to the end).
    open_release_early: int = 0
    # ``open_retreat_pass_m`` (0: off): after the pull the hand backs off only until it is this far out from where
    # the pull ended (as built: toward a point 5 cm out until within 1 cm of it, six steps of which the last three
    # creep; the jaws are clear of the bar 2.5 cm out).
    open_retreat_pass_m: float = 0.0
    # ``open_box``: on the way to the handle the position and rotation commands are each capped by their largest
    # component, not their length (``follow``'s ``box`` and ``box_rot``): the quarter turn from a hand that points
    # down begins about a slanted axis, and the way out and down is slanted too.
    open_box: bool = False
    # ``open_out_first``: a hand that begins the step over the fixture (within 12 cm of the point before the handle,
    # and above it: the step before set an object on the fixture's top) keeps its height until it is 3 cm out
    # before the grip point, and comes down there with every axis of its command capped on its own. (As built it
    # comes down at once, 1.5 cm before the grip point: the jaws, still turning, pass the bar's front edge within
    # a millimetre. Sent down with the command's direction kept, the turn swung it 2.6 cm back in toward the bar
    # and nothing asked it out again: the 15 cm down took the whole command.)
    open_out_first: bool = False
    # ``open_pre_brake_kz`` (0: off): on the way down to the point before the handle the move aims short of it in
    # height by this many steps of the hand's own downward movement (``brake_k``'s rule, which the bundle applies
    # sideways only). The hand comes down 16 cm at 12 mm a step and passes the point by 8 to 13 mm; the shoulder
    # and elbow that carried it down are then turned back by a 1 cm error, slowly: every command of the move in
    # to the handle is followed at a quarter of its usual speed for some ten steps.
    open_pre_brake_kz: float = 0.0
    # 2026-11-17, second request: the staging's release and the take-again (``drawer.RetakeAgent``) of the staged
    # plan. With the opening's options on, the plain drawer task is about 219 steps: stage 89 to 92, open 62, take
    # again 65. All off by default; none acts through a delay.
    # ``stage_release_early`` (0: off): a staging release sends its open command from this many steps before the
    # hand reaches the release height (the fingers begin to part three steps after the command; as built the hand
    # arrives, then opens for six steps), and the hold that follows is that much shorter.
    stage_release_early: int = 0
    # ``take_turn_early``: with ``open_retreat_pass_m``, an opening that another step follows backs off from the
    # handle already turning toward the pose the staging let go in (the sequencer hands it over), and, when the
    # pull itself sent the open commands (``open_release_early``), without standing still first: it backs off
    # along the pull's own way while the jaws part. The quarter turn back takes some 17 steps before the hand
    # travels freely (while it turns, the hand moves sideways at a few millimetres a step); as built the turn
    # begins with the take-again's first move, after two steps of standing still and the backing off.
    take_turn_early: bool = False
    # ``take_close_dz_m`` (0: 1.2 cm as built): the take-again's fingers may begin to close this far above the
    # grasp height (with the hand within 1.2 cm of the point sideways, as built). They are fully open then and
    # take ten steps to reach the rim; from 2.5 cm up they are still 5 cm apart when the hand arrives.
    take_close_dz_m: float = 0.0
    # ``take_down_lat_m`` (0: off): the take-again's way down ends once the hand is within 6 mm of the grasp height
    # and this far from the grasp point sideways (as built: within 6 mm of the point in all; a hand that comes
    # 1 to 2 cm off sideways creeps to that for five steps, and the jaws, fully open, take the wall 3 cm either way).
    take_down_lat_m: float = 0.0
    # ``take_hold_quick``: the take-again's hold is made once the gap has stopped (one step) on something thin
    # (above the rim's least gap, under 3 cm); as built after two more steps of an unchanged gap.
    take_hold_quick: bool = False
    # ``take_lift_carry``: the lift (1.2 cm) and the carry into the opened part as one move (the hand follows an
    # upward command at once and a sideways one slowly, so the object is up before it is across).
    take_lift_carry: bool = False
    # ``take_release_early`` (0: off): the take-again's open command is sent from this many steps before the carry
    # is within its tolerance of the goal (the fingers part three steps after it).
    take_release_early: int = 0
    # with ``tell_open``, after a carry lost its object: the hand opens while it rises this far (0: it opens where it
    # is, as before), then the scene is looked at, up to ``lost_looks`` times six steps apart until the object is
    # found. (2026-11-16, depth line: the look six steps after the loss, with the hand still over the place, did
    # not find the bowl; the next one, 45 steps later and after a grasp at nothing, did.)
    lost_rise_m: float = 0.10
    lost_looks: int = 3
    # debugging, off by default: every step's hand position, quaternion, jaw gap and gripper command as a "p" event
    trace_steps: bool = False


SHARE_EDGE_M = 0.025  # objects sharing a surface region: each origin this far inside its edge (place error 1-2 cm)


def side_order(frames: list, t_obj, side) -> tuple[list, list]:
    """Grasp frames with those on the asked side of the object first: the hand's point within about 37 degrees of
    ``side`` seen from the object's origin, then those as near the opposite side, then the ones to either flank
    (2026-11-15: with the asked side blocked, a flank grasp set the hand 5 cm off the fixture's middle line, where it
    reached badly; the opposite side stays on the line). Within a group the frames more squarely on the line come
    first (in steps of 0.1 of the cosine; a frame 40 degrees off it reached badly too), then the planner's own order.
    Returns (frames, the groups' sizes)."""
    sd = np.asarray(side, dtype=float)[:2]
    sd = sd / max(float(np.linalg.norm(sd)), 1e-9)

    def group(f) -> int:
        v = np.asarray(f["centre"], dtype=float)[:2] - np.asarray(t_obj, dtype=float)[:2]
        c = float(v @ sd) / max(float(np.linalg.norm(v)), 1e-9)
        return 0 if c >= 0.8 else (1 if c <= -0.8 else 2)

    def along(f) -> float:  # how squarely the hand's point lies on the asked line (1 on it, 0 across), in steps
        v = np.asarray(f["centre"], dtype=float)[:2] - np.asarray(t_obj, dtype=float)[:2]
        return round(abs(float(v @ sd)) / max(float(np.linalg.norm(v)), 1e-9), 1)

    g = [group(f) for f in frames]
    order = sorted(range(len(frames)), key=lambda i: (g[i], -along(frames[i])))  # stable
    return [frames[i] for i in order], [g.count(k) for k in (0, 1, 2)]


def pose_of(poses: dict, body: str) -> tuple[np.ndarray, np.ndarray]:
    p = poses[body]
    return GR.quat_wxyz_to_mat(p["quat_wxyz"]), np.asarray(p["pos"], dtype=float)


# =============================================================================== the agent
class OracleAgent:
    def __init__(self, lib, scene: dict, spec: dict, oracle, cap: int, ap: AgentParams = AgentParams(),
                 pp: GR.PlanParams = GR.PlanParams(), fp: FollowParams = FollowParams()):
        self.lib, self.scene, self.spec, self.oracle, self.cap = lib, scene, spec, oracle, int(cap)
        self.ap, self.pp, self.fp = ap, pp, fp
        self.jaw = lib.jaw
        self.target = spec["target"]
        self.ttype = scene["instances"][self.target]["type"]
        self.troot = scene["instances"][self.target]["root_body_name"]
        self.t = 0
        self.finished = False
        self.finish_reason = None
        self.attempts: list[dict] = []
        self.events: list[dict] = []
        self.tried: list[tuple[np.ndarray, np.ndarray]] = []
        self.pobs = None
        planes = [a for a in scene["arena"] if a["type"] == "plane" and a["normal"][2] > 0.99]
        self.floor_z = max((a["point"][2] for a in planes), default=None)
        sup = [a for a in scene["arena"] if a["type"] != "plane"]
        self.supports = GR.Boxes([a["centre"] for a in sup], [a["R"] for a in sup], [a["half"] for a in sup],
                                 ["arena"] * len(sup)) if sup else GR.Boxes.empty()
        mount = scene.get("robot_mount") or {}
        self.shoulder = np.asarray(mount["shoulder_pos"], dtype=float) if "shoulder_pos" in mount else None
        self.reach_cap = None  # lowered after an unreachable attempt (proprioceptive: the hand did not get there)
        self.failed_xy: list = []
        self.n_relax = 0
        self._last_poses = None
        self._prev_sig = None
        self._last_a = None
        self._frozen = False
        self._n_frozen = 0
        self.n_freeze_ev = 0
        self.n_stall_retry = 0
        self.n_close = 0
        self.n_cleared = 0  # ``clear_cover``: sweeps done this episode
        self.n_pre_look = 0
        self._empty = None  # ``empty_view_m``: (where the target was believed, the point to look from, the hand's R)
        if ap.redetect and hasattr(oracle, "redetect"):  # item 5: the look tool re-detects a lost target
            oracle.redetect = True
        if ap.held_pose_fix and hasattr(oracle, "rigid_predict"):
            oracle.rigid_predict = True
        if ap.relook_place and hasattr(oracle, "place_free"):  # the look after the release may move the receiver too
            oracle.place_free = (spec["place_instance"],) if spec.get("place_instance") is not None else ()
        self._gen = self._episode()
        next(self._gen)

    # ------------------------------------------------------------------ interface
    def _sig(self, pobs: dict):
        """A bit-exact signature of the proprioception (a moving or even a held robot never repeats it exactly)."""
        try:
            out = []
            for k in ("robot0_eef_pos", "robot0_joint_pos", "robot0_joint_vel"):
                v = pobs.get(k)
                if v is not None:
                    out.extend(float(x) for x in np.asarray(v, dtype=float).ravel())
            return tuple(out) if out else None
        except Exception:
            return None

    def act(self, pobs: dict) -> list[float]:
        # stale proprioception: when the observation repeats bit-exactly after a motion command, the robot's state
        # is not being refreshed (observation timing). Acting on it overshoots by many cm and the goto windows read
        # it as "no progress". Hold still (keep the gripper command) without advancing the step generator until the
        # observation changes, at most freeze_max steps in a row.
        sig = self._sig(pobs)
        prev, self._prev_sig = self._prev_sig, sig
        if sig is not None and prev is not None and sig == prev and not self.finished:
            moving = self._last_a is not None and float(np.linalg.norm(np.asarray(self._last_a[:6]))) > 1e-4
            if (self._frozen or moving) and self._n_frozen < self.ap.freeze_max:
                if not self._frozen:
                    self.n_freeze_ev += 1
                    if self.n_freeze_ev <= 20:
                        self._ev("stale_obs")
                self._frozen = True
                self._n_frozen += 1
                self.t += 1
                g = float(self._last_a[6]) if self._last_a is not None else -1.0
                return [0.0] * 6 + [g]
        self._frozen = False
        self._n_frozen = 0
        self.pobs = pobs
        p_now = np.asarray(pobs["robot0_eef_pos"], dtype=float)
        p_last = getattr(self, "_p_last", None)
        self._v_step = (p_now - p_last) if p_last is not None else np.zeros(3)  # the hand's movement in the last step
        self._p_last = p_now
        self.oracle.t_now = self.t
        try:
            a = self._gen.send(pobs)
        except StopIteration:
            a = None
        if a is None:
            self._finish(self.finish_reason or "generator_end")
            a = self._idle()
        if self.ap.trace_steps:
            try:
                self._ev("p", p=[round(float(x), 4) for x in pobs["robot0_eef_pos"]],
                         q=[round(float(x), 3) for x in pobs["robot0_eef_quat"]], g=round(float(self._gap()), 4),
                         a=round(float(a[6]), 1),
                         **({"j": [round(float(x), 3) for x in np.asarray(pobs["robot0_joint_pos"], float).ravel()]}
                            if pobs.get("robot0_joint_pos") is not None else {}))
            except Exception:
                pass
        self.t += 1
        a = [float(x) for x in a]
        self._last_a = a
        return a

    def summary(self) -> dict:
        return {"finished": self.finished, "finish_reason": self.finish_reason, "steps": self.t,
                "attempts": self.attempts, "events": self.events, "oracle_calls": self.oracle.calls}

    # ------------------------------------------------------------------ helpers
    def _ev(self, name: str, **kw):
        self.events.append({"t": self.t, "ev": name, **kw})

    def _finish(self, reason: str):
        if not self.finished:
            self.finished = True
            self.finish_reason = reason
            self._ev("finish", reason=reason)

    def _idle(self) -> list[float]:
        return [0.0] * 6 + [-1.0]

    def _after_loss(self, R_now):
        """The carried object is gone (``carry_watch``): let go and find it again (see ``lost_rise_m``). Without
        ``tell_open``: open where the hand is, as before. Returns what the looks found (a record), or None."""
        ap = self.ap
        self._tell_open(lost=True)
        p_l, _ = self._pose()
        if not (ap.tell_open and ap.lost_rise_m > 0 and hasattr(self.oracle, "hand_opened")):
            yield from self._hold(p_l, R_now, -1.0, ap.open_steps)
            return None
        yield from self._goto(p_l + np.array([0.0, 0.0, ap.lost_rise_m]), R_now, -1.0, ap.transit_cap, 0.02, 180.0,
                              max_steps=12)
        rec = {"looks": 0, "found": False}
        for k in range(max(1, int(ap.lost_looks))):
            if k:
                yield from self._hold(self._pose()[0], R_now, -1.0, 6)
                self.oracle.hand_opened(lost=True)
            try:
                self._last_poses = self.oracle()
            except Exception as ex:
                self._ev("look_error", err=str(ex)[:120])
                break
            rec["looks"] += 1
            if not getattr(self.oracle, "target_missing", False):
                rec["found"] = True
                break
        return rec

    def _half_turn_same(self, tol_m: float = 0.01) -> bool:
        """The target's model is the same turned half a turn about its root body's upright axis: every primitive has
        one of about the same size at the opposite place, to within 1 cm (a bowl, a box, a bottle; not a mug with
        its handle). The bowl's forty boxes answer each other to 7 mm in place and 5 mm in size."""
        model = self.lib.types[self.ttype]
        b = self.lib.body_boxes(self.ttype, model["root_body"])
        if not b.n:
            return False
        C, H = np.asarray(b.C, dtype=float), np.sort(np.asarray(b.H, dtype=float), axis=1)
        M = np.array([-1.0, -1.0, 1.0])
        for i in range(b.n):
            near = np.linalg.norm(C - C[i] * M, axis=1) <= tol_m
            if not np.any(near & np.all(np.abs(H - H[i]) <= tol_m, axis=1)):
                return False
        return True

    def _tell_open(self, lost: bool = False) -> None:
        """``tell_open``: the hand is about to let go of nothing, or of an object it will not place."""
        if self.ap.tell_open and hasattr(self.oracle, "hand_opened"):
            self.oracle.hand_opened(lost=lost)

    def _settle(self, grip: float):
        """Hold still for the loop's lag (``lag_steps``, set by the runner's delay compensator; 0 without one): the
        robot steers by its hand's predicted pose, but a look shows the scene of some steps ago. Before a check that
        compares the two (did the object come up with the hand), both are let catch up."""
        n = int(getattr(self, "lag_steps", 0) or 0)
        if n > 0:
            p, R = self._pose()
            self._ev("settle", n=n)
            yield from self._hold(p, R, grip, n)

    def _cmds_late(self) -> bool:
        """A lag is in use and the commands themselves are late: the lag is longer than the readings' own delay and
        a step (``lag_obs_steps``, from the runner), or no late readings were found at all."""
        lag, obs = int(getattr(self, "lag_steps", 0) or 0), int(getattr(self, "lag_obs_steps", 0) or 0)
        return lag > 0 and (obs <= 0 or lag - obs - 1 >= 1)

    def _close_wait(self) -> int:
        """Steps a closing that watches the fingers waits before it believes them still, when the delay compensator
        held the hand for the commands' delay alone (``close_hold_cmd``; the runner's ``lag_close_blind``): the
        readings' own delay and a step, which is what the whole-lag hold had waited. 0 otherwise."""
        if not getattr(self, "lag_close_blind", False) or int(getattr(self, "lag_steps", 0) or 0) <= 0:
            return 0
        return int(getattr(self, "lag_obs_steps", 0) or 0) + 1

    def _settle_fix(self, p, R, grip: float, cap: float, tol_m: float, tol_deg: float):
        """Through a delay only (nothing, and None, without one): a short move is over before any reading of it comes
        back, so it ends where the predicted hand says, and the arm's true answer to a few quick commands is not the
        predicted one (2026-11-15, S20: the grasp check failed after most approaches). So: wait out the lag, read
        where the hand truly is, and close what is left with one more small move. Returns what was left (m)."""
        if int(getattr(self, "lag_steps", 0) or 0) <= 0:
            return None
        yield from self._settle(grip)
        p_now, _ = self._pose()
        left = float(np.linalg.norm(p_now - np.asarray(p, dtype=float)))
        st = yield from self._goto(p, R, grip, cap, tol_m, tol_deg, max_steps=25)
        return {"left_m": round(left, 4), "status": st}

    def _pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self.pobs["robot0_eef_pos"], dtype=float),
                GR.quat_xyzw_to_mat(self.pobs["robot0_eef_quat"]))

    def _gap(self) -> float:
        q = self.pobs["robot0_gripper_qpos"]
        return self.jaw.gap_from_qpos(q[0], q[1])

    def _world(self, poses: dict, exclude=()) -> GR.Boxes:
        out = GR.Boxes.empty()
        for inst, info in self.scene["instances"].items():
            if inst in exclude:
                continue
            for rel, bname in info["bodies"].items():
                R, t = pose_of(poses, bname)
                b = self.lib.body_boxes(info["type"], rel)
                if b.n:
                    bb = b.placed(R, t)
                    bb.tags = [inst] * bb.n
                    out = out + bb
        return out

    def _instance_boxes(self, poses: dict, inst: str) -> GR.Boxes:
        info = self.scene["instances"][inst]
        out = GR.Boxes.empty()
        for rel, bname in info["bodies"].items():
            R, t = pose_of(poses, bname)
            b = self.lib.body_boxes(info["type"], rel)
            if b.n:
                out = out + b.placed(R, t)
        return out

    # ------------------------------------------------------------------ motion primitives (generators)
    def _goto(self, p_tgt, R_tgt, grip, cap_pos, tol_m, tol_deg, max_steps=None, stall_down=False,
              cap_rot=None, stall_after_m: float = 0.0, straight=False, rot_box=False, each=False):
        """``straight``: the position command keeps its direction (``follow``'s ``box``): for a leg whose straight
        line was tested for clearance. (The old cap clips every axis first: far from its target the hand moves
        along the diagonal of the signs until one axis arrives, then bends.) ``rot_box``: the rotation command is
        capped by its largest component too (``follow``'s ``box_rot``; never through a delay). ``each``: every
        axis of the position command capped on its own (``follow``'s ``each``; never through a delay)."""
        ap = self.ap
        max_steps = ap.max_goto_steps if max_steps is None else max_steps
        cap_rot = ap.rot_cap if cap_rot is None else cap_rot
        errs, zs, ps = [], [], []
        t0 = self.t
        status = "unreachable"
        fp_use = self.fp
        scale = 1.0
        n_damp = 0
        k_ref = 0  # the progress window counts from here (restarted after a damping)
        p_start = None
        no_lag = int(getattr(self, "lag_steps", 0) or 0) <= 0
        brake = float(ap.brake_k) if (ap.brake_k > 0 and tol_m <= ap.brake_tol_m and no_lag) else 0.0
        box = bool(straight) and no_lag
        for k in range(max_steps):
            p, R = self._pose()
            e, r = pose_error(p, R, p_tgt, R_tgt)
            if e <= tol_m and r <= tol_deg:
                status = "reached"
                break
            if p_start is None:
                p_start = np.array(p, dtype=float)
            errs.append((e, r))
            zs.append(float(p[2]))
            ps.append(np.array(p, dtype=float))
            if stall_down and k >= ap.stall_steps and p[2] - p_tgt[2] > tol_m:
                # ``stall_after_m``: only a hand that has already moved this far counts as stopped (a hand that has
                # not started yet, as under a delay of the commands, is not stopped by anything)
                if zs[-1 - ap.stall_steps] - zs[-1] < ap.stall_eps_m \
                        and (stall_after_m <= 0.0 or float(np.linalg.norm(ps[-1] - p_start)) >= stall_after_m):
                    status = "stalled"
                    break
            if k - k_ref >= ap.unreach_window:
                e0, r0 = errs[-1 - ap.unreach_window]
                if e0 - e < ap.unreach_gain_m and r0 - r < ap.unreach_gain_deg:
                    # timing tolerance: a hand that has not responded yet, or one that is moving (overshoot or
                    # oscillation from delayed actions), is not blocked; only a flat plateau is reported
                    moved0 = float(np.linalg.norm(ps[-1] - p_start))
                    win = ps[-1 - ap.unreach_window:]
                    path = float(sum(np.linalg.norm(win[i + 1] - win[i]) for i in range(len(win) - 1)))
                    if moved0 < ap.respond_m and k < ap.respond_grace:
                        pass
                    elif path > ap.moving_path_m and n_damp < ap.max_damp:
                        n_damp += 1
                        scale = max(ap.min_gain_scale, 0.5 * scale)
                        fp_use = FollowParams(gain_pos=self.fp.gain_pos * scale, gain_rot=self.fp.gain_rot * scale)
                        k_ref = k
                    else:
                        status = "unreachable"
                        break
            yield follow(p, R, self._braked(p_tgt) if brake > 0.0 else p_tgt, R_tgt, grip, cap_pos, cap_rot, fp_use,
                         box=box, box_rot=bool(rot_box) and no_lag, each=bool(each) and no_lag)
        p, R = self._pose()
        e, r = pose_error(p, R, p_tgt, R_tgt)
        self._ev("goto", t0=t0, status=status, damp=n_damp, err_m=round(e, 4), err_deg=round(r, 1),
                 tgt=[round(float(x), 4) for x in p_tgt], at=[round(float(x), 4) for x in p])
        return status

    def _braked(self, p_tgt):
        """``brake_k``: the point to aim at so that the hand stops at ``p_tgt`` (see the parameter's note)."""
        v = getattr(self, "_v_step", None)
        if v is None or int(getattr(self, "lag_steps", 0) or 0) > 0:
            return p_tgt
        k, kz = float(self.ap.brake_k), float(self.ap.brake_kz)
        return np.asarray(p_tgt, dtype=float) - np.array([k * float(v[0]), k * float(v[1]), kz * float(v[2])])

    def _hold(self, p_tgt, R_tgt, grip, n):
        for _ in range(n):
            p, R = self._pose()
            yield follow(p, R, p_tgt, R_tgt, grip, self.ap.slow_cap, self.ap.rot_cap, self.fp)

    def _swept_boxes(self, p, R, half_gap, held) -> GR.Boxes:
        b = self._gripper_world(p, R, half_gap)
        if held is not None:
            b = b + self._held_boxes(held[0], held[1], p, R)
        return b

    def _extra_obst(self) -> GR.Boxes:
        """The boxes the move being planned keeps clear of beyond the known scene (``transit_unexplained``; none
        otherwise, and none outside ``_move_via``)."""
        ex = getattr(self, "_transit_extra", None)
        return ex if ex is not None else GR.Boxes.empty()

    def _unexplained(self, held=None) -> GR.Boxes:
        """``transit_unexplained``: the look tool's tall unnamed things as boxes (none when the option is off).
        ``held`` (read with the looks for the way alone): what the hand holds, as ``(R, t)`` in the hand's frame; a
        look taken on the way shows it as a thing of its own."""
        ap = self.ap
        if not ap.transit_unexplained or not hasattr(self.oracle, "tall_unexplained"):
            return GR.Boxes.empty()
        on_way = bool(ap.look_cross or ap.look_stall > 0 or ap.detour_seen)
        try:
            p_hand = np.asarray(self._pose()[0], dtype=float)
            hand = p_hand[:2]
            at = getattr(self, "_pick_xy", None)  # where the carried object stood when its grasp was planned
            seg = None
            if self.shoulder is not None:
                # the robot's own arm in a look taken with the hand in the scene: its highest part is the elbow, so
                # from the shoulder to 20 cm short of the hand (what the load rests against stands within 10 cm of
                # the hand, and is not the arm)
                sh = np.asarray(self.shoulder[:2], dtype=float)
                d = hand - sh
                n = float(np.linalg.norm(d))
                if n > 0.25:
                    seg = ([float(x) for x in sh], [float(x) for x in hand - d / n * 0.20], 0.12)
            # (once a look for the way was had, its things alone: the robot is left out of it in the picture)
            kw = {"route": True} if on_way and ap.look_route_only \
                and getattr(self.oracle, "extra_route", None) is not None else {}
            bx = self.oracle.tall_unexplained(float(ap.transit_unexplained_min_h_m),
                                              clear_xy=() if at is None else (at,), clear_seg=seg,
                                              held_ids=(self.target,), **kw)
            if bx and on_way:
                # a look taken on the way shows the hand and what it holds as a thing of its own (colour rows, the
                # ten such regions in twenty looks taken at a stopped carry: 1.8 to 5.9 cm from the grip site,
                # their tops 2.5 to 4.0 cm above it, up to 14 cm long). By that band of height alone: a wall the
                # hand rests against is as near, and its top is where it is
                bx = [(c, R_, h) for c, R_, h in bx
                      if not (float(np.linalg.norm((c - p_hand)[:2])) <= 0.06
                              and 0.015 <= float(c[2] + h[2]) - float(p_hand[2]) <= 0.05)]
                if bx and held is not None:
                    # (2026-11-18) and what the hand holds: a thing whose middle lies in the load's own place
                    # (3 cm around it) and whose top lies within the load's own height. What the load rests
                    # against has its middle beside the load or stands taller; what lies under it ends lower
                    lo, hi = self._held_boxes(held[0], held[1], p_hand, self._pose()[1]).aabb()
                    lo, hi = lo.min(axis=0) - 0.03, hi.max(axis=0) + 0.03
                    bx = [(c, R_, h) for c, R_, h in bx
                          if not (bool(np.all(c[:2] >= lo[:2]) and np.all(c[:2] <= hi[:2]))
                                  and float(lo[2]) <= float(c[2] + h[2]) <= float(hi[2]) - 0.01)]
            if not bx:
                return GR.Boxes.empty()
            return GR.Boxes([c for c, _R, _h in bx], [R for _c, R, _h in bx], [h for _c, _R, h in bx],
                            ["unexplained"] * len(bx))
        except Exception:
            return GR.Boxes.empty()

    def _reach_ok(self, pts) -> bool:
        """Whether a top-down grip site at each of ``pts`` is within the arm's reach: no farther across from the
        shoulder than the grasp planner's own limit, and, above the shoulder, no farther from it than
        ``transit_reach_m`` less ``transit_reach_drop_per_deg`` for each degree of elevation. True when the scene
        names no shoulder."""
        if self.shoulder is None:
            return True
        sh = np.asarray(self.shoulder, dtype=float)
        for q in pts:
            d = np.asarray(q, dtype=float) - sh
            across, up = float(np.linalg.norm(d[:2])), float(d[2])
            if across > self.pp.reach_hard_m:
                return False
            if up > 0.0 and float(np.linalg.norm(d)) > self.ap.transit_reach_m \
                    - self.ap.transit_reach_drop_per_deg * math.degrees(math.atan2(up, max(across, 1e-9))):
                return False
        return True

    def _choose_unexplained(self, poses, p0, R0, p1, R1, half_gap, held, exclude, margin):
        """``transit_unexplained``: ``(boxes, way)`` for this carry. The tall unnamed things count (``boxes``) when the
        straight way is clear of them anyway, or when a crossing height clear of them exists within the arm's
        reach. Else, with ``transit_around``, a way around them (``way``: legs as ``_cut_way`` gives them). Else
        neither: the way is planned as before."""
        ap = self.ap
        empty = GR.Boxes.empty()
        self._ux_plan = None  # what the things did to this plan: "clear" (not in the way), "over", "around", "ignored"
        self._ux_seen = empty  # (the things this plan knew of, counted or not: read by the looks on the way)
        if not ap.transit_unexplained or (held is None and not ap.route_empty):
            return empty, None
        ux = self._unexplained(held)
        if ux.n and held is None:
            # (``route_empty``) on the way only: a thing the empty hand touches at its goal is not the way's
            at_goal = self._swept_boxes(p1, R1, half_gap, None)
            keep = np.array([not GR.any_overlap(at_goal, ux.subset(np.arange(ux.n) == j).inflated(margin))
                             for j in range(ux.n)], dtype=bool)
            ux = ux.subset(keep) if not keep.all() else ux
        self._ux_seen = ux
        if not ux.n:
            return empty, None
        self._transit_extra = ux
        mark = {"empty": True} if held is None else {}
        try:
            seen = [[round(float(c[0]), 3), round(float(c[1]), 3), round(float(c[2] + h[2]), 3),
                     round(float(max(h[0], h[1])), 3)] for c, h in zip(ux.C, ux.H)]
            if self._segment_clear(poses, p0, R0, p1, R1, half_gap, held, exclude, margin):
                self._ux_plan = "clear"
                return ux, None
            z0 = max(float(p0[2]), float(p1[2]))
            z = self._cross_z(poses, p0, R0, p1, R1, half_gap, held, exclude, margin)
            if z < z0 + ap.cross_max_m - 1e-9 and self._reach_ok([[p0[0], p0[1], z], [p1[0], p1[1], z]]):
                self._ev("transit", unexplained=int(ux.n), seen=seen, **mark)
                self._ux_plan = "over"
                return ux, None
            if ap.transit_around:
                self._transit_extra = None
                z_c = max(self._cross_z(poses, p0, R0, p1, R1, half_gap, held, exclude, margin), z0)
                way = self._way_around(poses, p0, R0, p1, R1, half_gap, held, exclude, margin, ux, z_c)
                if way is not None:
                    self._ev("transit", unexplained=int(ux.n), seen=seen, over=round(float(z), 4),
                             around=[[round(float(x), 3) for x in pt] for pt, _R, _t, _d in way], **mark)
                    self._ux_plan = "around"
                    return empty, way
            self._ev("transit", unexplained_ignored=round(float(z), 4), seen=seen, **mark)
            self._ux_plan = "ignored"
            return empty, None
        except Exception as ex:
            self._ev("transit", unexplained_error=str(ex)[:120])
            return empty, None
        finally:
            self._transit_extra = None

    def _way_around(self, poses, p0, R0, p_goal, R_goal, half_gap, held, exclude, margin, ux: GR.Boxes, z_c: float):
        """``transit_around``: legs ``(point, R, tol_m, tol_deg)`` from ``p0`` to ``p_goal`` across at height ``z_c``
        around one of the boxes ``ux``, or None. For each box and each of its two ends: a point beside the end on
        the hand's side of the box and one on the other side, both clear of the box by the carried load's half
        width and ``transit_around_clear_m``; the way is up to ``z_c``, to the first point, past the end to the
        second, to above the goal and down. Every leg is tested as ``_cross_z`` tests a crossing (the gripper and
        the held object at both end orientations and their mid-yaw, every ``path_step_m``, against the known scene,
        the supports and all of ``ux``); every point must be within reach. The shortest such way."""
        ap = self.ap
        p0, g = np.asarray(p0, dtype=float), np.asarray(p_goal, dtype=float)
        obst = (self._world(poses, exclude=exclude) + self.supports + ux).inflated(margin)
        yaw0, yaw1 = GR.yaw_of_hand(R0), GR.yaw_of_hand(R_goal)
        ym = yaw0 + 0.5 * GR.wrap_pi(yaw1 - yaw0)
        Rs = [R0, GR.hand_matrix([math.cos(ym), math.sin(ym), 0.0]), R_goal]

        def line(a, b):
            n = max(2, int(math.ceil(float(np.linalg.norm(b - a)) / ap.path_step_m)) + 1)
            return [a + (b - a) * f for f in np.linspace(0.0, 1.0, n)]

        def clear(pts):
            return not any(GR.any_overlap(self._swept_boxes(pt, R, half_gap, held), obst) for pt in pts for R in Rs)

        # how far the carried load reaches from the hand's middle, across, at any of the three orientations
        r_load = 0.0
        for R in Rs:
            lo, hi = self._swept_boxes(p0, R, half_gap, held).aabb()
            r_load = max(r_load, float(np.max(np.abs(np.concatenate([lo[:, :2] - p0[:2], hi[:, :2] - p0[:2]])))))
        d = r_load + margin + float(ap.transit_around_clear_m)
        A, B = np.array([p0[0], p0[1], z_c]), np.array([g[0], g[1], z_c])
        best = None
        for j, (c, Rb, h) in enumerate(zip(ux.C, ux.R, ux.H)):
            k = 0 if h[0] >= h[1] else 1
            u, n_ = np.asarray(Rb[:2, k], dtype=float), np.asarray(Rb[:2, 1 - k], dtype=float)
            u, n_ = u / max(float(np.linalg.norm(u)), 1e-9), n_ / max(float(np.linalg.norm(n_)), 1e-9)
            L, t = float(h[k]), float(h[1 - k])
            off = float((A[:2] - c[:2]) @ n_)
            near = 1.0 if off >= 0.0 else -1.0
            # a hand that rests against the thing first goes straight off it, to the distance it then keeps; that
            # leg is tested against everything else (it only moves away from the thing itself)
            back = None
            if abs(off) < t + d and abs(float((A[:2] - c[:2]) @ u)) <= L + d:
                back = np.array([*(A[:2] + near * (t + d - abs(off)) * n_), z_c])
                keep = np.ones(ux.n, dtype=bool)
                keep[j] = False
                rest = (self._world(poses, exclude=exclude) + self.supports + ux.subset(keep)).inflated(margin)
                if any(GR.any_overlap(self._swept_boxes(pt, R, half_gap, held), rest) for pt in line(A, back)[1:]
                       for R in Rs):
                    continue
            S0 = A if back is None else back
            for s in (-1.0, 1.0):
                E = c[:2] + s * (L + d) * u
                N = np.array([*(E + near * (t + d) * n_), z_c])
                F = np.array([*(E - near * (t + d) * n_), z_c])
                pts = [S0, N, F, B]
                if not self._reach_ok(pts[1:3]):
                    continue
                if self.shoulder is not None and any(
                        not ap.transit_reach_near_m <= GR.reach_dist(q, yaw1, self.shoulder, self.pp)
                        <= self.pp.reach_hard_m for q in pts[1:3]):
                    continue  # (too far across, or too near the robot's own base)
                if self.shoulder is not None and any(
                        float(q[0] - self.shoulder[0]) < ap.transit_around_front_m for q in pts[1:3]):
                    continue  # (beside or behind the robot: colour line, a way planned 30 cm behind its base)
                # (the hand's own place is not tested, as in ``_segment_clear``: it is there, and rises from there)
                if not all(clear(line(pts[i], pts[i + 1])[1 if (i == 0 and back is None) else 0:]) for i in range(3)):
                    continue
                length = sum(float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(3)) \
                    + (0.0 if back is None else float(np.linalg.norm(back - A)))
                if length > 3.0 * float(np.linalg.norm((B - A)[:2])) + 0.3:
                    continue  # (no way around that is three straight ways long)
                if best is None or length < best[0]:
                    best = (length, ([A] if back is not None else []) + pts)
        if best is None:
            return None
        way = best[1]
        legs = []
        if p0[2] < z_c - ap.transit_tol_m:
            legs.append((way[0], R0, ap.transit_tol_m, 180.0))
        legs += [(q, R_goal, ap.transit_tol_m, 180.0) for q in way[1:-1]]
        legs.append((way[-1], R_goal, ap.transit_tol_m, ap.transit_tol_deg))
        if float(np.linalg.norm(g - way[-1])) > 1e-6:
            legs.append((g, R_goal, None, None))
        else:
            legs[-1] = (g, R_goal, None, None)
        return legs

    def _cross_z(self, poses, p0, R0, p1, R1, half_gap, held, exclude, margin) -> float:
        """The lowest crossing height (site z) at which the gripper (and the held object) can move horizontally from
        above p0 to above p1, and descend at p1 to ``p1``'s height + 3 cm, clear of every non-excluded primitive and
        the arena's supports inflated by ``margin``: a swept-volume test of the boxes at both end orientations and
        their mid-yaw, sampled every ``path_step_m`` (amendment A3)."""
        ap = self.ap
        obst = (self._world(poses, exclude=exclude) + self.supports + self._extra_obst()).inflated(margin)
        yaw0, yaw1 = GR.yaw_of_hand(R0), GR.yaw_of_hand(R1)
        ym = yaw0 + 0.5 * GR.wrap_pi(yaw1 - yaw0)
        Rm = GR.hand_matrix([math.cos(ym), math.sin(ym), 0.0])
        Rs = [R0, Rm, R1]
        z0 = max(float(p0[2]), float(p1[2]))
        a, b = np.asarray(p0[:2], dtype=float), np.asarray(p1[:2], dtype=float)
        n = max(2, int(math.ceil(np.linalg.norm(b - a) / ap.path_step_m)) + 1)
        z_cross_only = None  # the lowest height at which the crossing alone is clear
        for dz in np.arange(0.0, ap.cross_max_m + 1e-9, ap.cross_step_m):
            z = z0 + dz
            ok = True
            pts = [np.array([*(a + (b - a) * f), z]) for f in np.linspace(0.0, 1.0, n)]
            zd = np.arange(z, float(p1[2]) + 0.03 - 1e-9, -ap.path_step_m)
            pts += [np.array([b[0], b[1], zz]) for zz in zd]
            for k, pt in enumerate(pts):
                for R in (Rs if k < n else [R1]):
                    if GR.any_overlap(self._swept_boxes(pt, R, half_gap, held), obst):
                        ok = False
                        break
                if not ok:
                    if k >= n and z_cross_only is None:
                        z_cross_only = z
                    break
            if ok:
                return z
        if ap.cross_descent_free and z_cross_only is not None:
            return z_cross_only
        return z0 + ap.cross_max_m

    def _segment_clear(self, poses, p0, R0, p1, R1, half_gap, held, exclude, margin) -> bool:
        """Is the straight move p0 -> p1 (yaw interpolated) clear (swept-volume test, as ``_cross_z``)?"""
        ap = self.ap
        obst = (self._world(poses, exclude=exclude) + self.supports + self._extra_obst()).inflated(margin)
        yaw0, yaw1 = GR.yaw_of_hand(R0), GR.yaw_of_hand(R1)
        n = max(2, int(math.ceil(np.linalg.norm(np.asarray(p1) - np.asarray(p0)) / ap.path_step_m)) + 1)
        for f in np.linspace(0.0, 1.0, n):
            pt = np.asarray(p0) + (np.asarray(p1) - np.asarray(p0)) * f
            y = yaw0 + f * GR.wrap_pi(yaw1 - yaw0)
            R = GR.hand_matrix([math.cos(y), math.sin(y), 0.0])
            if f > 0.0 and GR.any_overlap(self._swept_boxes(pt, R, half_gap, held), obst):
                return False
        return True

    def _cut_way(self, poses, p0, R0, p_goal, R_goal, z_cross, half_gap, held, exclude, margin):
        """``transit_cut``: the way from p0 over the crossing height to p_goal as a list of legs ``(point, R, tol_m,
        tol_deg)``, the last one the goal itself (its tolerances are the caller's: None here), or None when the
        crossing line is not clear at ``z_cross`` (the old way then stands).

        The old way is three legs with a stop at each corner: straight up to the crossing height, across, straight
        down. Here the hand climbs on a slant to the farthest of four points of the crossing line (its end, 3/4,
        1/2, 1/4) it reaches in a straight clear move, and leaves the line on a slant from the farthest of the same
        points of what is left, where that is clear. Every slanted leg and the crossing are tested as ``_cross_z``
        tests the crossing: the gripper (and the held object) at both end orientations and their mid-yaw, every
        ``path_step_m``, against every non-excluded primitive and the supports inflated by ``margin``. A point
        between two legs is passed within ``transit_blend_m`` when the straight line from there to the next point
        is clear too (else within the transit tolerance, as the old way's corners are)."""
        ap = self.ap
        p0, g = np.asarray(p0, dtype=float), np.asarray(p_goal, dtype=float)
        if float(np.linalg.norm((g - p0)[:2])) < 2.0 * ap.path_step_m:
            return None  # hardly any way across: nothing to cut
        obst = (self._world(poses, exclude=exclude) + self.supports + self._extra_obst()).inflated(margin)
        yaw0, yaw1 = GR.yaw_of_hand(R0), GR.yaw_of_hand(R_goal)
        ym = yaw0 + 0.5 * GR.wrap_pi(yaw1 - yaw0)
        Rs = [R0, GR.hand_matrix([math.cos(ym), math.sin(ym), 0.0]), R_goal]

        def line(a, b):
            n = max(2, int(math.ceil(float(np.linalg.norm(b - a)) / ap.path_step_m)) + 1)
            return [a + (b - a) * f for f in np.linspace(0.0, 1.0, n)]

        def clear(pts):
            return not any(GR.any_overlap(self._swept_boxes(pt, R, half_gap, held), obst) for pt in pts for R in Rs)

        A, B = np.array([p0[0], p0[1], z_cross]), np.array([g[0], g[1], z_cross])
        if not clear(line(A, B)):
            return None
        up, down = p0[2] < z_cross - ap.transit_tol_m, z_cross > g[2] + ap.transit_tol_m
        V, fv = A, 0.0
        if up:
            for f in (1.0, 0.75, 0.5, 0.25):
                cand = A + f * (B - A)
                if clear(line(p0, cand)[1:]):  # the hand's own place is not tested (as ``_segment_clear``)
                    V, fv = cand, f
                    break
        W, slant_down = B, False
        if down and fv < 1.0:
            for f in (1.0, 0.75, 0.5, 0.25):
                cand = B - f * (B - V)
                if not up and f == 1.0:
                    continue  # that is the straight move, already found blocked
                if clear(line(cand, g)):
                    W, slant_down = cand, True
                    break
        # straight down at the goal the planner tested the goal's own yaw alone: unless the way down is clear at
        # every yaw, the hand has finished its turn before it starts down (as the old way's crossing ends)
        wait_yaw = down and not slant_down and not clear(line(B, g))
        legs, at = [], p0  # (point, the orientation asked on the way to it)
        # (a slanted climb was tested at every yaw, so the hand may turn on it; straight up it keeps its yaw)
        for pt, Rt in ([(V, R_goal if fv > 0.0 else R0)] if up else []) + [(W, R_goal)]:
            if float(np.linalg.norm(pt - at)) > 1e-6:
                legs.append((pt, Rt))
                at = pt
        if legs and not down and float(np.linalg.norm(g - legs[-1][0])) <= ap.transit_tol_m + 1e-9:
            legs[-1] = (g, legs[-1][1])  # the crossing ends at the goal's own height, to within the transit tolerance
        elif float(np.linalg.norm(g - at)) > 1e-6 or not legs:
            legs.append((g, R_goal))
        out = []
        prev = p0
        for k, (pt, Rt) in enumerate(legs):
            if k == len(legs) - 1:
                out.append((pt, Rt, None, None))
                break
            nxt = legs[k + 1][0]
            tol = ap.transit_tol_m
            if wait_yaw and k == len(legs) - 2:
                out.append((pt, Rt, tol, ap.transit_tol_deg))
                prev = pt
                continue
            b = float(ap.transit_blend_m)
            l_in, l_out = float(np.linalg.norm(pt - prev)), float(np.linalg.norm(nxt - pt))
            if b > tol and l_in > 1e-6 and l_out > 1e-6:
                # the hand leaves this leg ``b`` before its end and heads straight for the next point: that line,
                # not the two legs, is its way from there, so that line is what is tested
                c_in = pt + (prev - pt) * min(1.0, b / l_in)
                if clear(line(c_in, nxt)):
                    tol = b
            out.append((pt, Rt, tol, 180.0))
            prev = pt
        return out

    def _closing_early(self, gen, c, width: float, lat_m: float = 0.004, max_dz_m: float | None = None):
        """``close_early_m``: run a move down to the grasp point ``c`` (a generator of actions) and turn its gripper
        command to "close" once the hand is within ``close_early_m`` above ``c`` (at most ``max_dz_m``) and ``lat_m``
        beside it. Returns the move's own result."""
        ap = self.ap
        if ap.close_early_m <= 0 or int(getattr(self, "lag_steps", 0) or 0) > 0:
            return (yield from gen)
        dz_m = float(ap.close_early_m) if max_dz_m is None else min(float(ap.close_early_m), float(max_dz_m))
        closing = False
        while True:
            try:
                a = next(gen)
            except StopIteration as e:
                return e.value
            if not closing:
                p, _R = self._pose()
                v = getattr(self, "_v_step", None)
                down = max(0.002, -float(v[2])) if v is not None else 0.002  # how fast it is coming down (m a step)
                if (float(p[2] - c[2]) <= min(dz_m, 3.0 * down) and float(np.linalg.norm((p - c)[:2])) <= lat_m
                        and self._gap() - float(width) >= 0.05):
                    closing = True
                    self._ev("close_early", dz=round(float(p[2] - c[2]), 4))
            if closing:
                a = list(a)
                a[6] = 1.0
            yield a

    _n_open_early = 0  # ``stage_release_early``: open commands the way down to the release itself sent

    def _opening_early(self, gen, p_rel, n: int):
        """``stage_release_early``: run the way down to a staging release (a generator of actions) and turn its
        gripper command to "open" once the hand is within ``n`` steps of the release height at the speed it is
        coming down (at least 6 mm a step is counted: the staging's way down is 1.5 cm and begins from a hand that
        was moving across). Returns the move's own result; ``_n_open_early`` counts."""
        self._n_open_early = 0
        while True:
            try:
                a = next(gen)
            except StopIteration as e:
                return e.value
            p, _R = self._pose()
            v = getattr(self, "_v_step", None)
            down = max(0.004, -float(v[2])) if v is not None else 0.004
            if self._n_open_early or float(p[2] - p_rel[2]) <= n * max(down, 0.006):
                a = list(a)
                a[6] = -1.0
                self._n_open_early += 1
            yield a

    def _watch_carry(self, gen, gap_held: float):
        """Run a carrying move (a generator of actions) and watch the fingers: "lost" as soon as they have met for
        ``carry_lost_steps`` steps, else the move's own result. ``gap_held``: the gap read at the lift check.
        Without ``carry_watch`` the move runs as it is."""
        ap = self.ap
        if not ap.carry_watch:
            return (yield from gen)
        limit = min(ap.carry_lost_gap_m, ap.carry_lost_share * float(gap_held))
        n = 0
        while True:
            try:
                a = next(gen)
            except StopIteration as e:
                return e.value
            yield a
            n = n + 1 if self._gap() < limit else 0
            if n >= ap.carry_lost_steps:
                gen.close()
                return "lost"

    def _watch_block(self, gen, p_goal):
        """``carry_block_steps``: run a carrying move (a generator of actions) and end it "blocked" when the hand has
        all but stopped far from ``p_goal``: over the last ``carry_block_steps`` steps its path is shorter than
        ``carry_block_path_m`` and it has not turned by 3 degrees, more than 5 cm (across) from the goal. Off (0),
        or through a delay of the commands, the move runs as it is."""
        ap = self.ap
        n = int(ap.carry_block_steps)
        if n <= 0 or int(getattr(self, "lag_steps", 0) or 0) > 0:
            return (yield from gen)
        g = np.asarray(p_goal, dtype=float)
        hist = []
        while True:
            try:
                a = next(gen)
            except StopIteration as e:
                return e.value
            yield a
            p, R = self._pose()
            hist.append((np.array(p, dtype=float), np.array(R, dtype=float)))
            if len(hist) > n:
                win = hist[-(n + 1):]
                path = float(sum(np.linalg.norm(win[i + 1][0] - win[i][0]) for i in range(n)))
                cos = 0.5 * (float(np.trace(win[0][1].T @ win[-1][1])) - 1.0)
                turn = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
                if path < ap.carry_block_path_m and turn < 3.0 and float(np.linalg.norm((win[-1][0] - g)[:2])) > 0.05:
                    gen.close()
                    self._ev("carry_blocked", at=[round(float(x), 4) for x in win[-1][0]], path_m=round(path, 4))
                    return "blocked"

    # -- 2026-11-17 (stream PATH): looks for the way, and the detour that uses them (all off by default)
    def _route_look(self, att: dict, why: str) -> dict:
        """One look for the way alone (``look_cross``, ``look_stall``, ``detour_seen``): the tool's ``look_route`` (no
        percept changes; the latest look's unnamed regions are replaced). No step is spent. The record, kept in
        ``att["route"]``: when, why, whether the look was had, and its seconds."""
        rec = {"t": self.t, "why": why, "at": [round(float(x), 3) for x in self._pose()[0]]}
        a = time.perf_counter()
        got = None
        try:
            f = getattr(self.oracle, "look_route", None)
            if f is not None:
                try:
                    got = f(cap=int(self.ap.look_route_max), self_px=int(self.ap.look_self_px))
                except TypeError:  # (a tool from before 2026-11-18: the look as it takes every look)
                    got = f()
        except Exception as ex:  # (a look that fails is no look: the move goes on as it would have)
            rec["error"] = str(ex)[:80]
        rec.update(looked=got is not None, s=round(time.perf_counter() - a, 3))
        if isinstance(got, dict) and got.get("shifted") is not None:
            rec.update(looked=False, shifted=got["shifted"])  # (taken, and not believed: nothing is read from it)
        att.setdefault("route", []).append(rec)
        self._t_route = self.t
        return rec

    def _route_new(self, rec: dict, poses, p_goal, R_goal, half_gap, held, exclude, margin) -> bool:
        """After a look on the way: does it show a thing on the hand's way to ``p_goal`` that the plan in hand did
        not count, and is there a plan that counts it (straight past it, over it, around it)? The record gets
        ``new`` (such things), ``counted`` (all the look's things as the way counts them) and ``plan``."""
        before = getattr(self, "_ux_seen", None)
        p_now, R_now = self._pose()
        p_now, g = np.array(p_now, dtype=float), np.asarray(p_goal, dtype=float)
        ux = self._unexplained(held)
        rec["counted"] = [[round(float(c[0]), 3), round(float(c[1]), 3), round(float(c[2] + h[2]), 3)]
                          for c, h in zip(ux.C, ux.H)]
        d = (g - p_now)[:2]
        L = float(np.linalg.norm(d))
        fresh = 0
        for c, h in zip(ux.C, ux.H):
            if before is not None and any(float(np.linalg.norm((c - c0)[:2])) <= 0.05 for c0 in before.C):
                continue  # (the plan in hand knew it)
            if L < 1e-6:
                continue
            u = d / L
            rel = (c - p_now)[:2]
            r = float(np.hypot(h[0], h[1])) + 0.10  # (its own half diagonal, and the hand with its load beside it)
            if -r <= float(rel @ u) <= L + r and abs(float(rel @ np.array([-u[1], u[0]]))) <= r:
                fresh += 1
        rec["new"] = fresh
        if not fresh:
            return False
        n_ev, keep = len(self.events), (getattr(self, "_ux_plan", None), before)
        self._choose_unexplained(poses, p_now, R_now, g, R_goal, half_gap, held, exclude, margin)
        del self.events[n_ev:]  # (the plan that follows says it once)
        rec["plan"] = getattr(self, "_ux_plan", None)
        if rec["plan"] in ("clear", "over", "around"):
            return True
        self._ux_plan, self._ux_seen = keep[0], ux  # (seen and of no use to a plan: not new a second time)
        return False

    def _route_watch(self, gen, att: dict, poses, p_goal, R_goal, half_gap, held, exclude, margin):
        """``look_cross`` and ``look_stall``: run a move (a generator of actions) and look for the way while it runs:
        once ``look_cross_after`` steps into a carry, and at the first sign of a stall (see the parameters). A look
        that shows a thing on the way which the plan did not count, and a plan that counts it: the move ends there,
        "route_seen" (``_route_why``: which look; ``_route_from``: where the hand was six steps before), and the
        caller plans it again. Otherwise, with both options off, without ``transit_unexplained``, for an empty hand
        without ``route_empty``, or through a delay of the commands, the move runs as it is."""
        ap = self.ap
        cross = bool(ap.look_cross) and held is not None and getattr(self, "_route_cross_att", None) is not att
        n_stall = int(ap.look_stall) - sum(1 for r in att.get("route") or [] if r.get("why") == "stall")
        if att.get("route_late") is not None:
            n_stall = 0  # (its commands are late: no thing is looked for at a stall in this attempt)
        if not ap.transit_unexplained or not (cross or n_stall > 0) or (held is None and not ap.route_empty) \
                or int(getattr(self, "lag_steps", 0) or 0) > 0:
            return (yield from gen)
        g = np.asarray(p_goal, dtype=float)
        ref, low, k, k_dir, u_prev, hist = float(ap.look_stall_ref), 0, 0, 0, None, []
        while True:
            p0 = np.array(self._pose()[0], dtype=float)
            try:
                a = next(gen)
            except StopIteration as e:
                return e.value
            yield a
            k += 1
            p1 = np.array(self._pose()[0], dtype=float)
            hist.append(p0)
            cmd = np.asarray(a[:3], dtype=float)
            m = float(np.linalg.norm(cmd))
            if m >= 0.3:
                u = cmd / m
                k_dir = k_dir + 1 if (u_prev is not None and float(u @ u_prev) > 0.87) else 0
                u_prev = u
                r = float((p1 - p0) @ u) / (m * OUT_POS)  # the way made toward where it was sent, per metre asked
                w = float(np.linalg.norm(p1 - p0)) / (m * OUT_POS)  # the way made in any direction
                if k_dir < 3:
                    low = 0  # (a new direction: the hand is still turning into it)
                elif r <= -ap.look_stall_ratio * ref:
                    # it goes back from where it is sent: late commands, not a thing (a thing lets nothing run on)
                    low = 0
                    if att.get("route_late") is None:
                        att["route_late"] = self.t
                    n_stall = 0
                elif r < ap.look_stall_ratio * ref and w < ap.look_stall_slide * ref:
                    low += 1
                else:
                    low, ref = 0, max(ref, r)
            else:
                low, k_dir, u_prev = 0, 0, None
            why = None
            if cross and k >= int(ap.look_cross_after):
                cross, why, self._route_cross_att = False, "cross", att
            elif n_stall > 0 and low >= int(ap.look_stall_steps) and float(np.linalg.norm((g - p1)[:2])) > 0.05:
                n_stall, low, why = n_stall - 1, 0, "stall"
            if why is None:
                continue
            rec = self._route_look(att, why)
            if why == "stall":
                rec["made"] = [round(r, 3), round(w, 3)]  # (the last step's way toward its goal and in all, per metre asked)
            pressed = why == "stall" or low > 0  # (the look on a free way may find the hand at the thing already)
            if rec["looked"] and self._route_new(rec, poses, g, R_goal, half_gap, held, exclude, margin):
                gen.close()
                self._route_why, self._route_from = ("stall" if pressed else why), hist[max(0, len(hist) - 6)]
                return "route_seen"

    def _route_back(self, R_now, grip: float):
        """After a look at a stall showed a new thing: off it by ``carry_relook_back_m`` the way the hand came (the
        plan that follows starts clear of it). Nothing after the look taken on a free way."""
        ap = self.ap
        if getattr(self, "_route_why", None) != "stall" or ap.carry_relook_back_m <= 0:
            return
        p_b = np.array(self._pose()[0], dtype=float)
        back = (np.asarray(self._route_from, dtype=float) - p_b)[:2]
        nb = float(np.linalg.norm(back))
        if nb > 1e-6:
            yield from self._goto(p_b + np.array([*(back / nb * ap.carry_relook_back_m), 0.0]), R_now, grip,
                                  ap.transit_cap, ap.transit_tol_m, 180.0, max_steps=8)

    def _detour_seen(self, att: dict, poses, p_b, R_now, p_goal, half_gap, held, exclude, margin, z_fixed: float):
        """``detour_seen``: ``(z, way, mark)`` for the detour of a carry stopped at ``p_b`` far from ``p_goal``. A look
        first, unless one was had in the last three steps. Then, when what the looks show raises the crossing over
        what the known scene asks: that crossing height if the arm reaches it (``way`` None, "over"); else, with
        ``transit_around``, legs around the thing's nearer end ("around"). Else (nothing seen explains the stop,
        or no way over or around it) the fixed height ``z_fixed``, lowered a centimetre at a time until the arm
        reaches it, never under ``detour_min_m`` above the higher of the two ends ("none")."""
        ap = self.ap
        mark = {"seen": "none"}
        if getattr(self, "_t_route", None) is None or self.t - self._t_route > 3:
            self._route_look(att, "detour")
        try:
            ux = self._unexplained(held)
            if ux.n:
                z0 = max(float(p_b[2]), float(p_goal[2]))
                z_k = self._cross_z(poses, p_b, R_now, p_goal, R_now, half_gap, held, exclude, margin)
                self._transit_extra = ux
                try:
                    z_s = self._cross_z(poses, p_b, R_now, p_goal, R_now, half_gap, held, exclude, margin)
                finally:
                    self._transit_extra = None
                if z_s > z_k + 1e-6:  # (what the looks show stands in this way)
                    if z_s < z0 + ap.cross_max_m - 1e-9 \
                            and self._reach_ok([[p_b[0], p_b[1], z_s], [p_goal[0], p_goal[1], z_s]]):
                        return float(z_s), None, {"seen": "over"}
                    if ap.transit_around:
                        z_c = max(float(z_k), float(p_b[2]))
                        way = self._way_around(poses, p_b, R_now, p_goal, R_now, half_gap, held, exclude, margin,
                                               ux, z_c)
                        if way is not None:
                            return z_c, way, {"seen": "around"}
        except Exception as ex:
            mark["error"] = str(ex)[:80]
        z, z_min = float(z_fixed), max(float(p_b[2]), float(p_goal[2])) + ap.detour_min_m
        while z > z_min and not self._reach_ok([[p_b[0], p_b[1], z], [p_goal[0], p_goal[1], z]]):
            z -= 0.01
        if z < float(z_fixed) - 1e-9:
            mark["capped"] = round(float(z_fixed) - z, 3)
        return z, None, mark

    def _watch_open(self, gen, gap_held: float, out: dict):
        """Run a lowering move (a generator of actions) and note where the fingers' gap first opened by
        ``cover_gap_m`` over ``gap_held``: ``out["z_open"]`` is the hand's height at that step and ``out["z_prev"]``
        its height one step before (None when the gap was open from the first step). The move itself is unchanged."""
        ap = self.ap
        prev = None if self._gap() - gap_held >= ap.cover_gap_m else float(self._pose()[0][2])
        first = prev is None
        while True:
            try:
                a = next(gen)
            except StopIteration as e:
                return e.value
            yield a
            z = float(self._pose()[0][2])
            if "z_open" not in out and self._gap() - gap_held >= ap.cover_gap_m:
                out.update(z_open=z, z_prev=None if first else prev)
            prev, first = z, False

    def _with_gap(self, gen, gap_m):
        """Run a move of the empty hand (a generator of actions) with the fingers kept ``gap_m`` apart: closing while
        they are wider, opening while they are narrower (the gripper takes a sign a step). None: the move as it is."""
        if gap_m is None:
            return (yield from gen)
        while True:
            try:
                a = next(gen)
            except StopIteration as e:
                return e.value
            a = list(a)
            a[6] = 1.0 if self._gap() > float(gap_m) else -1.0
            yield a

    def _watch_touch(self, gen, gap_held: float, out: dict):
        """``cover_pick``: run a lowering move (a generator of actions) with a rim-held object and end it "touched" at
        the first step where the fingers' gap has opened by ``cover_gap_m`` over ``gap_held``: the object rests on
        something. ``out`` gets the hand's height at that step and one step before (as ``_watch_open`` gives them).
        A gap that is open before the first step: the move runs as it is."""
        ap = self.ap
        if self._gap() - gap_held >= ap.cover_gap_m:
            return (yield from gen)
        prev = float(self._pose()[0][2])
        while True:
            try:
                a = next(gen)
            except StopIteration as e:
                return e.value
            yield a
            z = float(self._pose()[0][2])
            if self._gap() - gap_held >= ap.cover_gap_m:
                out.update(z_open=z, z_prev=prev)
                gen.close()
                return "touched"
            prev = z

    def _move_via(self, poses, p_goal, R_goal, grip, exclude, half_gap, held, margin, tol_m, tol_deg,
                  cap_last=None, stall_last=False):
        """Straight to the goal when that move is clear; otherwise rise to the swept-volume crossing height, cross at
        it, then descend to the goal. ``stall_last``: the last descent ends "stalled" as soon as the hand stops
        coming down (``_goto``'s ``stall_down``), instead of after the no-progress window."""
        ap = self.ap
        p0, R0 = self._pose()
        self._z_cross = None  # (this move's crossing height, when it went the cut way; kept for ``detour_from_cross``)
        if ap.rise_first and held is None and float(p0[2]) < float(p_goal[2]) - 0.02:
            # an empty hand that starts lower than its goal (never so from the start pose; so after something moved
            # it: 2026-11-15, a delay that replayed the first commands left it beside the target at grasp height, and
            # the straight way up to the point above the target, checked against everything but the target, took the
            # open fingers through it): straight up to the goal's height first, then across
            self._ev("transit", rise_first=round(float(p_goal[2] - p0[2]), 4))
            yield from self._goto(np.array([p0[0], p0[1], float(p_goal[2])]), R0, grip, ap.transit_cap,
                                  ap.transit_tol_m, 180.0, max_steps=40)
            p0, R0 = self._pose()
        sb = bool(ap.cap_box)  # the hand kept on each leg's own straight line (see ``cap_box``)
        if ap.transit_unexplained and (held is not None or ap.route_empty):
            # (off by default) the way is planned clear of what the look saw standing tall and could not name: the
            # three tests below read these boxes, and they are forgotten once the way is planned
            self._transit_extra, around = self._choose_unexplained(poses, p0, R0, p_goal, R_goal, half_gap, held,
                                                                   exclude, margin)
            if around is not None:  # (``transit_around``) no height within reach clears them: around
                self._transit_extra = None
                st = "reached"
                for k, (pt, Rt, tol, deg) in enumerate(around):
                    last = k == len(around) - 1
                    st = yield from self._goto(pt, Rt, grip, ap.transit_cap if (cap_last is None or not last)
                                               else cap_last, tol_m if last else tol, tol_deg if last else deg,
                                               straight=True)
                    if st != "reached":
                        return st
                return st
        try:
            if self._segment_clear(poses, p0, R0, p_goal, R_goal, half_gap, held, exclude, margin):
                self._transit_extra = None
                self._ev("transit", straight=True)
                st = yield from self._goto(p_goal, R_goal, grip, ap.transit_cap if cap_last is None else cap_last,
                                           tol_m, tol_deg, straight=sb)
                return st
            z_cross = self._cross_z(poses, p0, R0, p_goal, R_goal, half_gap, held, exclude, margin)
            z_cross = max(z_cross, float(p_goal[2]))
            self._ev("transit", z_cross=round(z_cross, 4))
            way = None
            if ap.transit_cut and int(getattr(self, "lag_steps", 0) or 0) <= 0:  # (through a delay: the old way)
                try:
                    way = self._cut_way(poses, p0, R0, p_goal, R_goal, z_cross, half_gap, held, exclude, margin)
                except Exception as ex:  # a degenerate geometry: the old way
                    self._ev("cut_error", err=str(ex)[:120])
                    way = None
        finally:
            self._transit_extra = None
        if way is not None:
            self._z_cross = float(z_cross)  # (the cut way rises as it goes: it can be stopped below this)
            self._ev("transit", cut=[[round(float(x), 3) for x in pt] + [tol] for pt, _R, tol, _d in way])
            st = "reached"
            for k, (pt, Rt, tol, deg) in enumerate(way):
                last = k == len(way) - 1
                # (each leg's own straight line was tested, so the hand is kept on it)
                st = yield from self._goto(pt, Rt, grip, ap.transit_cap if (cap_last is None or not last)
                                           else cap_last, tol_m if last else tol, tol_deg if last else deg,
                                           straight=True)
                if st != "reached":
                    return st
            return st
        if p0[2] < z_cross - ap.transit_tol_m:
            st = yield from self._goto(np.array([p0[0], p0[1], z_cross]), R0, grip, ap.transit_cap,
                                       ap.transit_tol_m, 180.0, straight=sb)
            if st != "reached":
                return st
        if z_cross > p_goal[2] + ap.transit_tol_m:
            st = yield from self._goto(np.array([p_goal[0], p_goal[1], z_cross]), R_goal, grip, ap.transit_cap,
                                       ap.transit_tol_m, ap.transit_tol_deg, straight=sb)
            if st != "reached":
                return st
        # (the stall keywords only when asked for: with ``stall_last`` off the call is the same as before)
        stall = {"stall_down": True, "stall_after_m": 0.004} if stall_last else {}
        st = yield from self._goto(p_goal, R_goal, grip, ap.transit_cap if cap_last is None else cap_last,
                                   tol_m, tol_deg, straight=sb, **stall)
        return st

    # ------------------------------------------------------------------ planning
    def _plan_grasp(self, poses, relaxed: bool = False):
        reach_cap = None if relaxed else self.reach_cap
        R_obj, t_obj = pose_of(poses, self.troot)
        self._pick_xy = [float(t_obj[0]), float(t_obj[1])]  # (kept for ``transit_unexplained``; nothing else reads it)
        target = self._instance_boxes(poses, self.target)
        obstacles = self._world(poses, exclude=(self.target,))
        rec = self.lib.records[self.ttype]
        _, R = self._pose()
        yaw = GR.yaw_of_hand(R)
        frames, diag = GR.plan_frames(rec, R_obj, t_obj, target, obstacles, self.supports, self.floor_z, self.jaw,
                                      yaw, self.pp, shoulder=self.shoulder, reach_cap=reach_cap)
        if not diag.get("up_ok", True):
            # the target is not in its rest orientation (or nothing fits): records for its current up axis
            up_now = R_obj.T @ UPV
            model = self.lib.types[self.ttype]
            root = model["bodies"][model["root_body"]]
            rec2 = GR.build_records(root["primitives"], up_now, root["com"], self.jaw,
                                    GR.RecordParams(theta_step_deg=15.0, b_step_m=0.006, z_step_m=0.006),
                                    name=self.ttype + "@now")
            pp2 = GR.PlanParams(**{**asdict(self.pp)})
            frames2, diag2 = GR.plan_frames(rec2, R_obj, t_obj, target, obstacles, self.supports, self.floor_z,
                                            self.jaw, yaw, pp2, shoulder=self.shoulder, reach_cap=reach_cap)
            diag["replanned_for_current_up"] = diag2
            if frames2:
                frames, rec = frames2, rec2
        if not frames and reach_cap is not None:
            # the reach cap left nothing: plan again without it (the cap is a preference, not a wall)
            frames, diag2 = GR.plan_frames(rec, R_obj, t_obj, target, obstacles, self.supports, self.floor_z,
                                           self.jaw, yaw, self.pp, shoulder=self.shoulder, reach_cap=None)
            diag["uncapped"] = {kk: v for kk, v in diag2.items()}
        # skip records already tried (in the object frame) and frames near one the hand could not reach
        out, again = [], []
        for f in frames:
            if not relaxed and any(np.linalg.norm(f["centre"][:2] - xy) < self.ap.unreach_excl_m
                                   for xy in self.failed_xy):
                continue
            c_obj = R_obj.T @ (f["centre"] - t_obj)
            a_obj = R_obj.T @ f["axis"]
            near = any(np.linalg.norm(c_obj - c) < self.pp.spacing_m and abs(float(a_obj @ a)) >
                       math.cos(math.radians(self.pp.spacing_deg)) for c, a in self.tried)
            f["c_obj"], f["a_obj"] = c_obj, a_obj
            if not near:
                out.append(f)
            elif relaxed:
                again.append(f)
        if relaxed:
            out = out + again  # a retry of a frame judged unreachable (likely timing), untried ones first
        diag["n_frames"] = len(frames)
        diag["n_untried"] = len(out)
        if not out and not relaxed and (self.failed_xy or self.reach_cap is not None) \
                and self.n_relax < self.ap.max_relax:
            # every frame was excluded only because a goto was judged unreachable. Under delayed actions or
            # observations that judgement is often wrong (the hand oscillates rather than being blocked), so retry
            # without the reach cap and the exclusion, with a damped follower gain (fewer overshoots)
            self.n_relax += 1
            out, d2 = self._plan_grasp(poses, relaxed=True)
            diag["relaxed"] = {"n_frames": d2.get("n_frames"), "n_untried": d2.get("n_untried")}
            if out:
                self.fp = FollowParams(gain_pos=self.fp.gain_pos * self.ap.relax_gain_scale,
                                       gain_rot=self.fp.gain_rot * self.ap.relax_gain_scale)
                self._ev("relax", n=self.n_relax, gain=round(self.fp.gain_pos, 3))
        side = self.spec.get("grasp_side")
        if side is not None and out:  # a planned step asks for the hand on one side of the object
            out, diag["grasp_side"] = side_order(out, t_obj, side)
            if int(self.ap.side_wide_k) > 0 and diag["grasp_side"][0] == 0:
                # ``side_wide_k``: none of the planner's own first frames is on the asked side (it keeps the 8 that
                # turn the hand least, and on a round object those are as often on the other side: nothing is in the
                # way). The planner is asked for more, and the asked side's frames go first; the rest stay behind
                # them as they were
                try:
                    pp_w = GR.PlanParams(**{**asdict(self.pp), "k_frames": int(self.ap.side_wide_k)})
                    more, _d = GR.plan_frames(rec, R_obj, t_obj, target, obstacles, self.supports, self.floor_z,
                                              self.jaw, yaw, pp_w, shoulder=self.shoulder, reach_cap=reach_cap)
                    wide = []
                    for f in more:
                        if not relaxed and any(np.linalg.norm(f["centre"][:2] - xy) < self.ap.unreach_excl_m
                                               for xy in self.failed_xy):
                            continue
                        c_obj = R_obj.T @ (f["centre"] - t_obj)
                        a_obj = R_obj.T @ f["axis"]
                        if any(np.linalg.norm(c_obj - c) < self.pp.spacing_m and abs(float(a_obj @ a)) >
                               math.cos(math.radians(self.pp.spacing_deg)) for c, a in self.tried):
                            continue
                        f["c_obj"], f["a_obj"] = c_obj, a_obj
                        wide.append(f)
                    wide, sizes = side_order(wide, t_obj, side)
                    diag["grasp_side_wide"] = sizes
                    if sizes[0] > 0:
                        out = wide[:sizes[0]] + out
                except Exception as ex:  # the planner's own list stands
                    diag["grasp_side_wide"] = str(ex)[:80]
        return out, diag

    def _lower_reach(self, fr):
        self.failed_xy.append(np.asarray(fr["centre"][:2], dtype=float))
        cap = fr["reach_m"] - self.ap.reach_backoff_m
        self.reach_cap = cap if self.reach_cap is None else min(self.reach_cap, cap)
        self._ev("reach_cap", cap=round(self.reach_cap, 4))

    def _c1(self, gap: float, width: float) -> bool:
        """The jaw gap matches the record's width. Rim grasps (width <= rim_width_m) pinch a slanted wall and the
        object pivots until the wall is flat between the jaws, so their band is [c1_rim_min_gap_m, w + hi]."""
        ap = self.ap
        if width <= ap.rim_width_m:
            return ap.c1_rim_min_gap_m < gap <= width + ap.c1_band_hi_m
        lo = ap.c1_band_lo_m
        if str(getattr(self, "ttype", "")).startswith("seen_"):
            # a model made from what was seen (``seen.py``): its width is a look's, not a stored one (the band's lower
            # edge as for the models the robot learnt itself, ST-1: 12 mm)
            lo = max(lo, ap.c1_seen_lo_m)
        return (width - lo <= gap <= width + ap.c1_band_hi_m) and gap > ap.c1_min_gap_m

    def _c1_soft(self, gap: float, width: float, stable: bool) -> bool:
        """``c1_soft``: the fingers stopped on something of another width than the record's and stayed there."""
        ap = self.ap
        if not ap.c1_soft or not stable or width <= ap.rim_width_m:
            return False
        lo = max(ap.c1_soft_min_m, ap.c1_soft_lo * width)
        hi = min(2.0 * self.jaw.L - ap.c1_soft_open_m, ap.c1_soft_hi * width)
        return lo <= gap <= hi

    def _stood_other_side(self, width: float):
        """``c1_stand``: the other side across of a stood, unmeasured target's model (None when the target is no
        such model, or its two sides are alike, or the other one is a rim's width)."""
        ap = self.ap
        try:
            if not ap.c1_stand or not getattr(self.oracle, "is_stood", lambda: False)():
                return None
            from .ground import aabb

            p = next(x for x in self.oracle.percepts if x.get("id") == self.target)
            lo, hi = aabb(self.lib, p)
            sx, sy = float(hi[0] - lo[0]), float(hi[1] - lo[1])
            other = sy if abs(sx - width) <= abs(sy - width) else sx
            return other if abs(other - width) > ap.c1_band_hi_m and other > ap.rim_width_m else None
        except Exception:
            return None

    def _c2_fingers(self, gap: float, gap2: float, moved: float, rise: float, rel_tol: float) -> bool:
        """The lift check by the fingers: they hold the width they closed on (``gap`` at the close, ``gap2`` after
        the lift), and the look has not shown the object left behind (``moved`` against the hand under
        ``c2_left_share`` of the hand's ``rise``, or under the look's own tolerance)."""
        ap = self.ap
        return bool(gap2 > ap.c1_rim_min_gap_m and abs(gap2 - gap) <= ap.c2_gap_keep_m
                    and moved < max(rel_tol, ap.c2_left_share * rise))

    def _let_go(self) -> None:
        """``release_here``: the hand opens here on a target it does not carry away; the look tool takes the
        target's pose from the hand now, not at its next look."""
        if self.ap.release_here and hasattr(self.oracle, "release_here"):
            self.oracle.release_here()

    def _look_redetect(self):
        """One more look from where the hand is, with the look tool's re-detection on for it: a refused track
        brings a merged full look. Returns the poses, or None when the look could not be taken."""
        had = getattr(self.oracle, "redetect", None)
        if had is not None:
            self.oracle.redetect = True
        try:
            new = self.oracle()
        except Exception as ex:  # a degenerate look: the attempt goes on with what it has
            self._ev("look_error", err=str(ex)[:120])
            return None
        finally:
            if had is not None:
                self.oracle.redetect = had
        self._last_poses = new
        return new

    def _pre_look(self, poses) -> float | None:
        """``pre_look``: one more look from the pre-grasp point (the hand open). Returns how far the target now
        stands from where ``poses`` had it, or None when the look could not be taken."""
        new = self._look_redetect()
        if new is None:
            return None
        return float(np.linalg.norm(pose_of(new, self.troot)[1] - pose_of(poses, self.troot)[1]))

    def _held_limit(self, width: float) -> None:
        """``held_jump_max_m``: tell the look tool how far a track may put the target from where the hand's motion
        puts it while this grasp holds it (0, no limit, for a rim grasp or when the option is off)."""
        ap = self.ap
        if ap.held_jump_max_m > 0 and hasattr(self.oracle, "held_jump_max_m"):
            self.oracle.held_jump_max_m = 0.0 if width <= ap.rim_width_m else float(ap.held_jump_max_m)
        if ap.held_tilt_max_deg > 0 and hasattr(self.oracle, "held_tilt_max_deg"):
            self.oracle.held_tilt_max_deg = 0.0 if width <= ap.rim_width_m else float(ap.held_tilt_max_deg)

    def _target_source(self):
        """The record only: how the look tool last got the target's pose (``track``, ``predicted``, ...), or None."""
        try:
            return next((str(p.get("source")) for p in (getattr(self.oracle, "percepts", None) or [])
                         if p.get("id") == self.target), None)
        except Exception:
            return None

    def _c2_gap(self, gap: float, width: float) -> bool:
        """The lift check's gap test: the close check's, with its own lower edge when ``c2_band_lo_m`` is set (an
        object that turned flat between the jaws while it was lifted is still held; rim grasps keep their band)."""
        ap = self.ap
        if ap.c2_band_lo_m <= 0.0 or width <= ap.rim_width_m:
            return self._c1(gap, width)
        return (width - max(ap.c2_band_lo_m, ap.c1_band_lo_m) <= gap <= width + ap.c1_band_hi_m) and gap > ap.c1_min_gap_m

    def _rel(self, poses):
        """Target pose in the hand frame (R_rel, t_rel)."""
        p, R = self._pose()
        Ro, to = pose_of(poses, self.troot)
        return R.T @ Ro, R.T @ (to - p)

    def _held_boxes(self, R_rel, t_rel, p_hand, R_hand) -> GR.Boxes:
        model = self.lib.types[self.ttype]
        b = self.lib.body_boxes(self.ttype, model["root_body"])
        return b.placed(R_hand @ R_rel, R_hand @ t_rel + p_hand)

    def _gripper_world(self, p_hand, R_hand, half_gap) -> GR.Boxes:
        G = GR.G_matrix(R_hand[:, 1])
        return self.jaw.gripper_boxes_G(max(half_gap, 0.0)).placed(G, p_hand)

    def _place_goal(self, poses):
        """(desired target-root xy, place-surface boxes, container instance, region box or None)."""
        sp = self.spec
        cont = sp["place_instance"]
        if sp["place_kind"] == "region":
            site = self.scene["sites"][sp["place"]]
            if site["frame_body"] == "world":
                Rb, tb = np.eye(3), np.zeros(3)
            else:
                Rb, tb = pose_of(poses, site["frame_body"])
            c = tb + Rb @ np.asarray(site["pos"])
            Rs = Rb @ np.asarray(site["R"])
            half = np.abs(Rs @ np.asarray(site["size"]))
            goal = c[:2].copy()
            sh = sp.get("place_share")
            if sh and int(sh[1]) > 1:  # several objects asked into this region: each takes its own part of it
                goal = goal + self._share_offset(poses, half, int(sh[0]), int(sh[1]))
            return goal, cont, {"centre": c, "half": half}
        Rp, tp = pose_of(poses, self.scene["instances"][cont]["root_body_name"])
        return tp[:2].copy(), cont, None

    def _plan_place(self, poses, R_rel, t_rel, gap):
        goal_xy, cont, region = self._place_goal(poses)
        plan = self._place_at(poses, R_rel, t_rel, gap, goal_xy, cont, region)
        blocked = (plan is not None and plan.get("blocked_by") == "other"  # another object (not the container's walls)
                   and plan["z_release"] > plan["z_lo"] + 0.02)
        if not (self.ap.place_free_xy and region is not None and (blocked or self._region_occupied(poses, region))):
            return plan
        keep = (self.ap.share_keep_up_m > 0 and self.spec.get("place_share") and self.spec.get("predicate") == "on")
        if keep and plan is not None and plan["z_release"] <= plan["z_lo"] + self.ap.share_keep_up_m:
            return plan  # ``share_keep_up_m``: its own part of the shared surface, as planned
        # another object already lies in the region: the lowest release over a grid of points across it
        p, R = self._pose()
        lo, hi = self._held_boxes(R_rel, t_rel, p, R).aabb()
        obj_root = R @ t_rel + p
        reach = np.maximum(obj_root[:2] - lo[:, :2].min(0), hi[:, :2].max(0) - obj_root[:2])  # root to footprint edge
        span = np.maximum(np.asarray(region["half"][:2], float) - reach, 0.0)
        n = max(2, int(self.ap.place_free_n))

        def search(span):
            best, best_key, cands = plan, (plan["z_release"] if plan else np.inf, 0.0), []
            for fx in np.linspace(-1.0, 1.0, n):
                for fy in np.linspace(-1.0, 1.0, n):
                    off = np.array([fx, fy]) * span
                    cand = self._place_at(poses, R_rel, t_rel, gap, np.asarray(region["centre"][:2], float) + off,
                                          cont, region)
                    if cand is None:
                        continue
                    cands.append((cand, off))
                    key = (round(cand["z_release"], 3), float(np.linalg.norm(off)))
                    if key < best_key:
                        best, best_key = cand, key
            return best, cands

        cands = []
        if keep:  # ``share_keep_up_m``: the object's origin stays on the region, as its planned part does
            best, cands = search(np.maximum(np.asarray(region["half"][:2], float) - SHARE_EDGE_M, 0.0))
        if not cands:  # (and with no point on the region at all, the search as it was)
            keep = False
            if cont is not None:  # a small region on a larger object (a stove's burner): its whole top may hold it
                clo, chi = self._instance_boxes(poses, cont).aabb()
                top = 0.5 * (chi[:, :2].max(0) - clo[:, :2].min(0))
                span = np.maximum(span, np.maximum(top - reach, 0.0))
            best, cands = search(span)
        if keep and cands:
            # among the points within 2 cm of the lowest release, the one furthest from the other objects
            others = [np.asarray(pose_of(poses, info["root_body_name"])[1], float)[:2]
                      for inst, info in self.scene["instances"].items() if inst not in (self.target, cont)
                      and info["root_body_name"] in poses]
            z_min = min(c["z_release"] for c, _ in cands)
            near = [(c, off) for c, off in cands if c["z_release"] <= z_min + 0.02]

            def clear(item):
                c, off = item
                xy = np.asarray(region["centre"][:2], float) + off
                return (min((float(np.linalg.norm(xy - o)) for o in others), default=0.0), -c["z_release"])

            best = max(near, key=clear)[0]
        if best is not None:
            best["free_xy"] = True
        return best

    def _share_offset(self, poses, half, i: int, n: int) -> np.ndarray:
        """Where the i-th of n objects goes in a region they share: as far apart as the region allows, along its
        diagonal (the first at the corner away from the robot, the last at the opposite one). A measured yaw is not
        trusted to keep a handle clear of the neighbour (2026-11-15: side by side and 10 cm apart, the second pot came
        down on the first and both fell in the replayed start; 9 of 10 plain starts failed). On a surface the
        object's origin stays ``SHARE_EDGE_M`` inside the region's edge; in a container the whole object stays inside
        its walls."""
        usable = np.maximum(np.asarray(half, float)[:2] - SHARE_EDGE_M, 0.0)
        if self.spec.get("predicate") == "in":
            lo, hi = self._instance_boxes(poses, self.target).aabb()
            ext = 0.5 * (hi.max(0) - lo.min(0))[:2]
            usable = np.maximum(np.asarray(half, float)[:2] - ext - 0.005, 0.0)
        return usable * (1.0 - 2.0 * i / max(n - 1, 1))

    def _near(self, poses, xy, radius: float = 0.2) -> list:
        """[instance, type, distance] of the other instances whose root lies within ``radius`` of ``xy`` (record)."""
        out = []
        for inst, info in self.scene["instances"].items():
            if inst == self.target:
                continue
            try:
                _R, t = pose_of(poses, info["root_body_name"])
            except Exception:
                continue
            dist = float(np.linalg.norm(np.asarray(t, float)[:2] - np.asarray(xy, float)[:2]))
            if dist <= radius:
                out.append([inst, info["type"], round(dist, 3), round(float(t[2]), 3)])
        return sorted(out, key=lambda x: x[2])

    def _region_occupied(self, poses, region) -> bool:
        """Another percept's box overlaps the region's box (in xy, and within its height)."""
        c, h = np.asarray(region["centre"], float), np.asarray(region["half"], float)
        boxes = self._world(poses, exclude=(self.target, self.spec["place_instance"]))
        if not boxes.n:
            return False
        lo, hi = boxes.aabb()
        hit = np.all((lo[:, :2] <= c[:2] + h[:2]) & (hi[:, :2] >= c[:2] - h[:2]), axis=1) \
            & (hi[:, 2] >= c[2] - h[2]) & (lo[:, 2] <= c[2] + h[2] + 0.05)
        return bool(hit.any())

    def _place_at(self, poses, R_rel, t_rel, gap, goal_xy, cont, region):
        ap = self.ap
        p, R = self._pose()
        held_now = self._held_boxes(R_rel, t_rel, p, R)
        obj_root = R @ t_rel + p
        hand_xy = goal_xy + (p[:2] - obj_root[:2])
        # no place object (a point on the table beside a landmark): the table is the container
        container = self._instance_boxes(poses, cont) if cont is not None else self.supports
        others = self._world(poses, exclude=(self.target, cont)) + (self.supports if cont is not None else GR.Boxes.empty())
        # place surface: highest container top under the held object's lowest boxes
        lo_h, hi_h = held_now.aabb()
        zmin = float(lo_h[:, 2].min())
        low = lo_h[:, 2] <= zmin + 0.005
        fp_lo = lo_h[low, :2].min(0) - (p[:2] - hand_xy)
        fp_hi = hi_h[low, :2].max(0) - (p[:2] - hand_xy)
        clo, chi = container.aabb()
        under = np.all((clo[:, :2] <= fp_hi) & (chi[:, :2] >= fp_lo), axis=1)
        surface = float(chi[under, 2].max()) if under.any() else float(chi[:, 2].min())
        clear = ap.place_clear_region_m if region is not None else ap.place_clear_object_m
        low_rel = zmin - p[2]  # held lowest point relative to the hand
        z_lo = surface + clear - low_rel
        # the swept-volume crossing height; the release height is searched downward from it
        held = (R_rel, t_rel)
        z_top = self._cross_z(poses, p, R, np.r_[hand_xy, max(p[2], z_lo)], R, 0.5 * gap, held, (self.target,),
                              ap.transit_margin_m)
        cont_i = container.inflated(ap.place_gripper_clear_m)
        others_i = others.inflated(ap.margin_m)
        z = max(z_top, z_lo)
        z_rel = None
        steps = 0
        blocked_by = None
        while z >= z_lo - 1e-9 and steps < 400:
            ph = np.array([hand_xy[0], hand_xy[1], z])
            hb = self._held_boxes(R_rel, t_rel, ph, R) + self._gripper_world(ph, R, 0.5 * gap)
            if GR.any_overlap(hb, cont_i) or (others_i.n and GR.any_overlap(hb, others_i)):
                blocked_by = "other" if (others_i.n and GR.any_overlap(hb, others_i)) else "container"
                break
            z_rel = z
            z -= ap.place_step_m
            steps += 1
        if z_rel is None:
            return None
        return {"hand_xy": hand_xy, "z_release": z_rel, "z_top": z_top, "z_lo": z_lo, "surface": surface,
                "low_rel": low_rel, "region": region, "container": cont, "blocked_by": blocked_by}

    def _cover_plan(self, poses, p_rel, R_hand, R_rel, t_rel, gap, touched_m: float | None = None, rim: bool = False,
                    at=None, centre=None):
        """Where to push a thing off the receiver from, given that the last descent stopped on it (the hand is where
        it stopped). Returns (plan, record); the plan is None when the stop is not such a case or no side is free.
        ``touched_m``: the stop was a touch-down below the planned release height, this far above the surface.

        The thing's top is the held object's lowest point now. It is taken to lie over the receiver (the place
        region when the object goes ON a region, else the receiving object's own boxes) and to reach
        ``cover_over_m`` past its edge. The push runs along the gripper's closing axis, in the sense whose path
        passes over the lower things: the pushing face (the held object or a finger, whichever leads) starts just
        outside the thing and ends ``cover_past_m`` past the receiver's far edge, with the held object's lowest point
        ``cover_under_m`` below the thing's top but at least ``cover_skim_m`` above everything it passes over.
        ``at``: where the hand stopped, when it has moved since (``cover_look``); ``R_hand`` is then the hand as it
        will push (turned about the vertical where it stopped). ``centre``: the middle of the thing as a look saw it;
        the held object's own middle then passes over it (else the push runs through where the hand stopped)."""
        ap = self.ap
        p0 = np.asarray(at, dtype=float) if at is not None else self._pose()[0]
        dz = float(p0[2] - p_rel[2])
        exy = float(np.linalg.norm((p0 - np.asarray(p_rel, dtype=float))[:2]))
        rec = {"t": self.t, "dz": round(dz, 4), "exy": round(exy, 4)}
        if touched_m is not None:
            rec["touched_m"] = round(float(touched_m), 4)
            if not (ap.cover_touch_m <= touched_m and exy <= ap.cover_xy_m):
                return None, None  # it touched the surface itself
        elif not (ap.cover_min_m <= dz <= ap.cover_max_m and exy <= ap.cover_xy_m):
            return None, None  # an ordinary release, or a hand stopped somewhere else
        if self.cap - self.t < ap.cover_need_steps:
            return None, {**rec, "skipped": "no steps"}
        goal_xy, cont, region = self._place_goal(poses)
        if cont is None and region is None:
            return None, {**rec, "skipped": "no receiver"}
        fixture = cont is not None and (self.scene["instances"].get(cont) or {}).get("kind") == "fixture"
        # ``rim`` (``cover_skim_rim_m``): the pushing object is held by its rim and the receiver is an object that can
        # be moved: the push passes higher over it. Over a fixture a touch moves nothing, and its covers are thin.
        skim = float(ap.cover_skim_rim_m) if (rim and ap.cover_skim_rim_m > 0 and cont is not None and not fixture) \
            else float(ap.cover_skim_m)
        under_m = float(ap.cover_under_rim_m) if (rim and ap.cover_under_rim_m > 0 and cont is not None
                                                  and not fixture) else float(ap.cover_under_m)
        if region is not None and (cont is None or fixture or self.spec.get("predicate") == "on"):
            # a region of something larger (a burner, a cabinet's top, a drawer): the thing lies over the region
            c = np.asarray(region["centre"], dtype=float)[:2]
            half = np.asarray(region["half"], dtype=float)[:2]
        else:
            clo, chi = self._instance_boxes(poses, cont).aabb()
            c = 0.5 * (clo[:, :2].min(0) + chi[:, :2].max(0))
            half = 0.5 * (chi[:, :2].max(0) - clo[:, :2].min(0))
        held = self._held_boxes(R_rel, t_rel, p0, R_hand)
        fingers = self.jaw.gripper_boxes_G(max(0.5 * gap, 0.0), with_hand=False).placed(GR.G_matrix(R_hand[:, 1]), p0)
        lo_h, _hi_h = held.aabb()
        top = float(lo_h[:, 2].min())  # the thing's top: what the held object rests on
        corners = np.concatenate([held.corners().reshape(-1, 3), fingers.corners().reshape(-1, 3)])
        axis = np.array([R_hand[0, 1], R_hand[1, 1]], dtype=float)
        n = float(np.linalg.norm(axis))
        if n < 1e-6:
            return None, {**rec, "skipped": "no closing axis"}
        axis = axis / n
        obst = self._world(poses, exclude=(self.target,)) + self.supports  # the receiver too: the push passes over it
        olo, ohi = obst.aabb() if obst.n else (np.zeros((0, 3)), np.zeros((0, 3)))
        best = None
        for sign in (1.0, -1.0):
            d = sign * axis
            e = np.array([-d[1], d[0]])
            lead = float(((corners - p0)[:, :2] @ d).max())
            wide = float(np.abs((corners - p0)[:, :2] @ e).max())
            r_recv = float(np.abs(d) @ half)
            a0 = float(c @ d) - (r_recv + ap.cover_over_m) - 0.015  # the pushing face along d: start and end
            far = float(c @ d) + r_recv
            # everything pushed ends beyond the face. The face stops just past where the held object's far side will
            # be once it is set down in the middle: the place is then free for it, and a cover as wide as the receiver
            # has its middle well past the far edge and tips off (2026-11-16: pushing on to 5 cm past the far edge
            # took 45 steps, and the episode's steps ran out)
            hc = held.corners().reshape(-1, 3)[:, :2] @ d
            a1 = min(far + ap.cover_past_m, float(c @ d) + 0.5 * float(hc.max() - hc.min()) + ap.cover_clear_m)
            lat = float(p0[:2] @ e)
            if centre is not None:
                he = held.corners().reshape(-1, 3)[:, :2] @ e
                lat += float(np.asarray(centre, dtype=float)[:2] @ e) - 0.5 * float(he.max() + he.min())
            under, under_recv, room = None, None, 0.30
            if obst.n:
                ctr, hh = 0.5 * (olo + ohi)[:, :2], 0.5 * (ohi - olo)[:, :2]  # each box's world-axis box, on d and e
                los, his = ctr @ d - hh @ np.abs(d), ctr @ d + hh @ np.abs(d)
                lo_e, hi_e = ctr @ e - hh @ np.abs(e), ctr @ e + hh @ np.abs(e)
                # room for the thing to come down past the far edge: the nearest other thing there that stands as
                # high (not the receiver, not the table). The push ends 1 cm short of it.
                other = np.array([tg != cont for tg in obst.tags]) if cont is not None else np.ones(obst.n, bool)
                side = float(np.abs(e) @ half) + ap.cover_over_m
                near = other & (ohi[:, 2] > top - under_m) & (los >= far - 0.01) \
                    & (hi_e >= lat - side) & (lo_e <= lat + side) & (hh.max(1) < 0.4)
                if near.any():
                    room = float(np.clip((los[near] - far).min(), 0.0, 0.30))
                    a1 = min(a1, far + room - 0.01)
                # the highest top of anything the pushing object passes over, from its start to the far edge
                hit = (his >= a0 - 2 * lead) & (los <= min(a1, far)) & (hi_e >= lat - wide - 0.01) \
                    & (lo_e <= lat + wide + 0.01)
                if (hit & other).any():
                    under = float(ohi[hit & other, 2].max())  # other things: never pushed into
                if (hit & ~other).any():
                    under_recv = float(ohi[hit & ~other, 2].max())  # the receiver itself, as the look has it
            floor = (under if under is not None else top - under_m) + skim
            drop = min(under_m, top - floor)  # how far below the thing's top the pushing object goes
            if under_recv is not None and drop >= 0.008:
                # over the receiver the touch is trusted before the look: the thing's top was felt, the receiver's
                # height is a look's (2026-11-16: a basket read 1 cm high left "nothing to push against" under a
                # 2.3 cm cover). At least ``cover_grip_m`` of the thing is taken; coming down beside it the hand
                # feels the receiver if it is higher than that (``_clear_cover``)
                drop = min(drop, max(top - (under_recv + skim), ap.cover_grip_m))
            # what leads at the thing's height: the parts of the held object and the fingers that are below its top
            # during the push (a finger that passes above the thing does not push it)
            low = (corners[:, 2] - drop) < top - 0.002
            if drop > 0 and low.any():
                lead = float(((corners[low] - p0)[:, :2] @ d).max())
            cand = {"d": d, "e": e, "lead": lead, "a0": a0, "a1": a1, "lat": lat, "drop": drop, "r_recv": r_recv,
                    "room": room}
            # of two free sides, the one the arm reaches more easily (20 cm of room is room enough)
            reach = 0.0
            if self.shoulder is not None:
                sh = np.asarray(self.shoulder, dtype=float)[:2]
                reach = max(float(np.linalg.norm(d * (a - lead) + e * lat - sh)) for a in (a0, a1))
            key = (drop >= 0.008 and a1 >= float(c @ d), min(round(room, 2), 0.20), -round(reach, 2), round(drop, 3))
            if best is None or key > best["key"]:
                best = {**cand, "key": key}
        rec.update(top=round(top, 4), drop=round(best["drop"], 4), dir=[round(float(x), 3) for x in best["d"]],
                   r_recv=round(best["r_recv"], 4), lead=round(best["lead"], 4), room=round(best["room"], 3))
        if rim:  # the record: the skim the push was planned with
            rec["skim"] = round(skim, 4)
        if not best["key"][0]:
            return None, {**rec, "skipped": "nothing to push against" if best["drop"] < 0.008 else "no room"}
        d, e = best["d"], best["e"]
        z_up = float(p0[2]) + ap.cover_rise_m
        z_push = float(p0[2]) - best["drop"]

        def at(along: float, z: float) -> np.ndarray:  # the hand, with the pushing face at ``along`` on d
            xy = d * (along - best["lead"]) + e * best["lat"]
            return np.array([xy[0], xy[1], z])

        plan = {"up": np.array([p0[0], p0[1], z_up]), "start_up": at(best["a0"], z_up), "start": at(best["a0"], z_push),
                "end": at(best["a1"], z_push), "end_up": at(best["a1"], z_up),
                "back": np.array([p_rel[0], p_rel[1], z_up]), "out": d * 0.04}
        if rim and ap.cover_slow_cap > 0 and cont is not None and not fixture:
            # ``cover_slow_cap``: slowly from where the pushing face comes to the receiver's near edge
            a_s = float(c @ d) - best["r_recv"] - 0.01
            if best["a0"] < a_s < best["a1"]:
                plan["slow"] = at(a_s, z_push)
                rec["slow_m"] = round(best["a1"] - a_s, 4)
        return plan, rec

    def _cover_look(self, poses, p_touch, R_hand, R_rel, t_rel, gap_held: float, rec: dict):
        """``cover_look`` (a generator of actions): the hand, with what it holds, goes to one side of the receiver,
        out of the main camera's line to it, and the robot looks at what lies on the receiver. Returns the hand's
        rotation for a push across the thing's long side when that needs a turn, else None ("lost": the object left
        the hand). The record gets ``look``.

        The side: across the camera's line, the one nearer the robot. (Not toward the robot: an arm stretched out
        over a far receiver does not come back in, 2026-11-16, two of the eight episodes; across, it moves.)"""
        ap = self.ap
        _goal, cont, _region = self._place_goal(poses)
        if cont is None or (self.scene["instances"].get(cont) or {}).get("kind") == "fixture":
            return None
        clo, chi = self._instance_boxes(poses, cont).aabb()
        c = 0.5 * (clo[:, :2].min(0) + chi[:, :2].max(0))
        half = 0.5 * (chi[:, :2].max(0) - clo[:, :2].min(0))
        hlo, hhi = self._held_boxes(R_rel, t_rel, p_touch, R_hand).aabb()
        c_h = 0.5 * (hlo[:, :2].min(0) + hhi[:, :2].max(0))
        r_h = 0.5 * float((hhi[:, :2].max(0) - hlo[:, :2].min(0)).max())
        cam = (getattr(self.oracle, "cams", None) or {}).get("agentview")
        sh = None if self.shoulder is None else np.asarray(self.shoulder, dtype=float)[:2]
        v = None
        if cam is not None:
            v = np.asarray(cam["T"], dtype=float)[:2, 3] - c
        elif sh is not None:
            v = c - sh  # no camera pose: taken to look at the robot from the other side of the scene
        if v is None or float(np.linalg.norm(v)) < 1e-6:
            return None
        v = v / float(np.linalg.norm(v))
        w = np.array([-v[1], v[0]])
        if sh is not None and float(np.linalg.norm(c - w - sh)) < float(np.linalg.norm(c + w - sh)):
            w = -w
        aside = c + w * (float(np.abs(w) @ half) + r_h + 0.03)  # where the held object's middle goes
        p_look = np.array([aside[0] + p_touch[0] - c_h[0], aside[1] + p_touch[1] - c_h[1],
                           float(p_touch[2]) + ap.cover_rise_m])
        st = yield from self._watch_carry(self._goto(p_look, R_hand, 1.0, ap.transit_cap, ap.transit_tol_m, 180.0,
                                                     max_steps=25), gap_held)
        look = rec.setdefault("look", {})
        look.update(t=self.t, moved=st)
        if st == "lost":
            return "lost"
        if st != "reached":
            return None  # (still over the receiver: what the hand holds would be taken for the thing)
        on = self.oracle.on_receiver(self.spec["place_instance"])
        look["on"] = None if on is None else {k: on.get(k) for k in ("xy", "half_xy", "yaw", "z", "n_px", "inside")}
        if on is None:
            return None
        try:
            a, b = (float(x) for x in on["half_xy"])
            psi = float(on["yaw"])
        except (KeyError, TypeError, ValueError):
            return None
        if float(on.get("inside") or 0.0) < 0.5 or a < 1.15 * b:
            return None  # not seen over the receiver, or as long as wide: any side does
        n = np.array([-math.sin(psi), math.cos(psi)])  # across its long side
        axis = np.array([R_hand[0, 1], R_hand[1, 1]], dtype=float)
        if float(np.linalg.norm(axis)) < 1e-6:
            return None
        axis = axis / float(np.linalg.norm(axis))
        ang = math.atan2(float(axis[0] * n[1] - axis[1] * n[0]), float(axis @ n))  # the closing axis onto n
        if ang > 0.5 * math.pi:
            ang -= math.pi  # (a line: the smaller of the two turns)
        elif ang < -0.5 * math.pi:
            ang += math.pi
        look["off_deg"] = round(math.degrees(ang), 1)
        if abs(ang) < math.radians(ap.cover_turn_min_deg):
            return None
        # the wrist's last joint does the turn: the smaller turn when that joint has the room for it, else the other
        # way round (a top-down hand turned by +a about the vertical turns that joint by -a)
        q = self.pobs.get("robot0_joint_pos")
        if q is not None and len(q) >= 7:
            q7 = float(q[6])
            if abs(q7 - ang) > ap.wrist_turn_max_rad:
                alt = ang - math.copysign(math.pi, ang)
                if abs(q7 - alt) > ap.wrist_turn_max_rad:
                    look["turn"] = "no room"
                    return None
                ang = alt
        cz, sz = math.cos(ang), math.sin(ang)
        R_push = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]) @ np.asarray(R_hand, dtype=float)
        st = yield from self._watch_carry(self._goto(p_look, R_push, 1.0, ap.transit_cap, 0.02, 6.0, max_steps=25),
                                          gap_held)
        look.update(turn_deg=round(math.degrees(ang), 1), turned=st)
        if st == "lost":
            return "lost"
        if st != "reached":
            return None
        return R_push

    def _spot_ok(self, poses, xy, r: float, z_grip: float, exclude=()) -> bool:
        """``cover_pick``: a disc of radius ``r`` at ``xy`` lies on a support, ``cover_spot_clear_m`` clear of every
        known thing but ``exclude``, and a top-down hand ``z_grip`` high reaches it."""
        ap = self.ap
        xy = np.asarray(xy, dtype=float)
        if self.supports.n:
            slo, shi = self.supports.aabb()
            if not bool(np.any(np.all(xy - r >= slo[:, :2] + 0.01, axis=1) & np.all(xy + r <= shi[:, :2] - 0.01, axis=1))):
                return False
        w = self._world(poses, exclude=exclude)
        if w.n:
            lo, hi = w.aabb()
            d = np.maximum(np.maximum(lo[:, :2] - xy, xy - hi[:, :2]), 0.0)
            if bool(np.any(np.linalg.norm(d, axis=1) < r + ap.cover_spot_clear_m)):
                return False
        if self.shoulder is not None:
            across = float(np.linalg.norm(xy - np.asarray(self.shoulder, dtype=float)[:2]))
            if across < ap.transit_reach_near_m or across > self.pp.reach_soft_m + 0.06:
                return False
        return self._reach_ok([np.array([xy[0], xy[1], z_grip])])

    def _support_top(self, xy, default: float) -> float:
        """The top of the support under ``xy`` (the highest arena box that holds the point), else ``default``."""
        if not self.supports.n:
            return default
        slo, shi = self.supports.aabb()
        xy = np.asarray(xy, dtype=float)
        on = np.all(xy >= slo[:, :2], axis=1) & np.all(xy <= shi[:, :2], axis=1)
        return float(shi[on, 2].max()) if bool(on.any()) else default

    def _cover_recv(self, poses):
        """``cover_pick``: ((instance, middle across, half sizes across, top), None) for a receiver the option acts
        on (an object that can be moved, up to ``cover_pick_recv_max_m`` across), else (None, why)."""
        _goal, cont, _region = self._place_goal(poses)
        if cont is None or (self.scene["instances"].get(cont) or {}).get("kind") == "fixture":
            return None, "receiver"
        clo, chi = self._instance_boxes(poses, cont).aabb()
        c = 0.5 * (clo[:, :2].min(0) + chi[:, :2].max(0))
        half = 0.5 * (chi[:, :2].max(0) - clo[:, :2].min(0))
        if 2.0 * float(half.max()) > self.ap.cover_pick_recv_max_m:
            return None, "receiver size"
        return (cont, c, half, float(chi[:, 2].max())), None

    def _cover_takeable(self, on, z_recv: float, held_xy, r_h: float, poses=None, cont=None):
        """``cover_pick``: ((middle across, long half, short half, yaw of the long side, top), None) when the look's
        ``on`` is a thing the fingers can take off the receiver, else (None, why); (None, None) for nothing seen.
        ``held_xy``, ``r_h``: where the held object is now and its half size across (never taken for the thing).
        With ``poses``: nor is a thing whose middle lies over an object the robot knows (one it set on the receiver
        itself in an earlier step, and this look did not report)."""
        ap = self.ap
        if on is None:
            return None, None
        try:
            a, b = sorted((float(x) for x in on["half_xy"]), reverse=True)
            psi, z_top = float(on["yaw"]), float(on["z"])
            xy_t = np.asarray(on["xy"], dtype=float)[:2]
        except (KeyError, TypeError, ValueError):
            return None, "no box"
        if float(on.get("inside") or 0.0) < 0.5 or int(on.get("n_px") or 0) < ap.cover_pick_min_px:
            return None, "not on the receiver"
        if 2.0 * b > ap.cover_pick_max_m or 2.0 * a > ap.cover_pick_long_m:
            return None, "too wide"
        if z_top - z_recv < ap.cover_pick_min_h_m:
            return None, "too flat"
        if float(np.linalg.norm(xy_t - np.asarray(held_xy, dtype=float)[:2])) < r_h + b + 0.01:
            return None, "the held object"
        if poses is not None:
            w = self._world(poses, exclude=tuple(x for x in (self.target, cont) if x is not None))
            if w.n:
                lo, hi = w.aabb()
                over = np.all(xy_t >= lo[:, :2] - 0.01, axis=1) & np.all(xy_t <= hi[:, :2] + 0.01, axis=1) \
                    & (hi[:, 2] > z_recv - 0.01)
                if bool(over.any()):
                    return None, "a known object"
        return (xy_t, a, b, psi, z_top), None

    def _cover_spots(self, poses, c, half, z_recv: float, r_h: float, low_rel: float) -> list:
        """``cover_pick``: the free places beside the receiver where the held object can wait, each as
        (how much along the arm's reach: 0 across, 1 slanted, 2 along; the reach; the place; the hand's height when
        the object stands there). ``low_rel``: the held object's lowest point from the hand (negative). Not between
        the receiver and the main camera: the looks that follow are at the receiver."""
        ap = self.ap
        cam = (getattr(self.oracle, "cams", None) or {}).get("agentview")
        v = None
        if cam is not None:
            v = np.asarray(cam["T"], dtype=float)[:2, 3] - c
            v = v / max(float(np.linalg.norm(v)), 1e-9)
        sh = None if self.shoulder is None else np.asarray(self.shoulder, dtype=float)[:2]
        rad = None if sh is None else (c - sh) / max(float(np.linalg.norm(c - sh)), 1e-9)
        spots = []
        for k in range(12):
            u = np.array([math.cos(k * math.pi / 6.0), math.sin(k * math.pi / 6.0)])
            if v is not None and float(u @ v) > 0.5:
                continue
            # (no farther out than the receiver's larger half size: a round one's corners are not there, and a
            # square one's are refused below)
            xy = c + u * (min(float(np.abs(u) @ half), float(half.max())) + r_h + ap.cover_park_gap_m)
            z_hand = self._support_top(xy, z_recv) - low_rel + ap.cover_park_clear_m
            if not self._spot_ok(poses, xy, r_h, z_hand, exclude=(self.target,)):
                continue
            along = 0.0 if rad is None else abs(float(u @ rad))
            reach = 0.0 if sh is None else float(np.linalg.norm(xy - sh))
            spots.append((0 if along <= 0.3 else 1 if along <= 0.75 else 2, round(reach, 2), xy, z_hand))
        return spots

    def _watch_see(self, gen, p_goal, poses, R_rel, t_rel, att: dict):
        """``cover_see``: run a carrying move (a generator of actions) and look at the receiver on the way, when the
        hand first comes within each of ``cover_see_m`` of the middle of its top, while the held object is not over
        it yet (see the parameter's note). A thing the fingers can take seen on it: the move ends there,
        "cover_seen" (the thing is kept for ``_cover_pick``). Anything else, nothing, or the option off: the move
        runs as it is. The record: ``att["cover_seen"]``, one entry a look."""
        ap = self.ap
        if not (ap.cover_see and ap.cover_pick) or self.n_cleared > 0 or getattr(self, "n_picked", 0) > 0 \
                or ap.trust_place \
                or self.spec.get("place_instance") is None or not hasattr(self.oracle, "on_receiver") \
                or int(getattr(self, "lag_steps", 0) or 0) > 0 or self.cap - self.t < ap.cover_see_need:
            return (yield from gen)
        recv, _why = self._cover_recv(poses)
        if recv is None:
            return (yield from gen)
        top = np.array([recv[1][0], recv[1][1], recv[3]])  # the middle of the receiver's top
        left = sorted((float(x) for x in ap.cover_see_m), reverse=True)
        seen_rec = []
        while left:
            p, R = self._pose()
            p = np.asarray(p, dtype=float)
            if float(np.linalg.norm(p - top)) <= left[0]:
                d_look = float(np.linalg.norm(p - top))
                while left and d_look <= left[0]:
                    left.pop(0)  # (one look for every distance already passed)
                hlo, hhi = self._held_boxes(R_rel, t_rel, p, R).aabb()
                c_h = 0.5 * (hlo[:, :2].min(0) + hhi[:, :2].max(0))
                r_h = 0.5 * float((hhi[:, :2].max(0) - hlo[:, :2].min(0)).max())
                if float(np.linalg.norm(c_h - top[:2])) < r_h + ap.cover_see_clear_m:
                    break  # the held object is over the receiver's middle: a look would show it, not the thing
                on = self.oracle.on_receiver(self.spec["place_instance"])
                take, why = self._cover_takeable(on, recv[3], c_h, r_h, poses, recv[0])
                seen_rec.append({"t": self.t, "d": round(d_look, 3), "on": None if on is None else {
                    k: on.get(k) for k in ("xy", "half_xy", "yaw", "z", "n_px", "inside")},
                    **({"refused": why} if why else {})})
                att["cover_seen"] = seen_rec
                if take is not None:
                    gen.close()
                    self._cover_take = take
                    return "cover_seen"
            try:
                a = next(gen)
            except StopIteration as e:
                return e.value
            yield a
        return (yield from gen)

    def _cover_turn(self, R_hand, psi: float, pick: dict) -> float:
        """``cover_pick``: the turn about the vertical (rad) that brings the hand's closing axis across a thing whose
        long side lies along ``psi``: the smaller of the two ways, none under ``cover_pick_turn_min_deg``, and
        inside the wrist's last joint's range (see the notes below). The record gets ``turn_deg``."""
        ap = self.ap
        n = np.array([-math.sin(psi), math.cos(psi)])  # across its long side
        axis = np.array([R_hand[0, 1], R_hand[1, 1]], dtype=float)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
        ang = math.atan2(float(axis[0] * n[1] - axis[1] * n[0]), float(axis @ n))  # the closing axis onto n
        if ang > 0.5 * math.pi:
            ang -= math.pi  # (a line: the smaller of the two turns)
        elif ang < -0.5 * math.pi:
            ang += math.pi
        if abs(ang) < math.radians(ap.cover_pick_turn_min_deg):
            ang = 0.0  # (near enough: the hand as it is. The look's long side is followed even when the thing reads
            # nearly as wide as long: the colour line reads the 4.7 by 6.7 cm block 5.8 by 6.3 cm, its long side
            # within 9 degrees in six looks, and across a corner it would not go between the fingers)
        q = self.pobs.get("robot0_joint_pos")
        if ang != 0.0 and q is not None and len(q) >= 7 and abs(float(q[6]) - ang) > ap.cover_wrist_max_rad:
            # the wrist's last joint has no room for the whole turn: as far as it goes when that leaves the fingers
            # within ``cover_turn_slack_deg`` of the short side (a turn the other way round is 140 degrees and more:
            # 50 steps on the Mac); else the other way round, else none
            q7 = float(q[6])
            part = q7 - math.copysign(ap.cover_wrist_max_rad, q7 - ang)
            alt = ang - math.copysign(math.pi, ang)
            if abs(part - ang) <= math.radians(ap.cover_turn_slack_deg):
                pick["turn_short_deg"] = round(math.degrees(abs(part - ang)), 1)
                ang = part
            else:
                ang = alt if abs(q7 - alt) <= ap.cover_wrist_max_rad else 0.0
        pick["turn_deg"] = round(math.degrees(ang), 1)
        return ang

    def _cover_pick(self, poses, p_rel, R_hand, R_rel, t_rel, gap_held: float, rec: dict, seen=None, width=None):
        """``cover_pick`` (a generator of actions; see the parameter's note). The hand has stopped on a thing over the
        receiver, or (``seen``, from ``_watch_see``) a thing the fingers can take was seen on it during the carry.
        Returns "parked" (the object waits beside the receiver, the hand is empty; whether the thing was taken off
        is in the record), "lost" (the object left the hand on the way), "back" (the hand went aside, saw nothing
        it can take, and is back over the release point with the object) or None (nothing was done, or the move to
        the waiting place did not arrive: the carry goes on). The record gets ``pick``."""
        ap = self.ap
        pick = rec.setdefault("pick", {})
        pick["t"] = self.t
        recv, why = self._cover_recv(poses)
        if recv is None:
            pick["skipped"] = why
            return None
        if self.cap - self.t < (ap.cover_see_need if seen is not None else ap.cover_pick_need):
            pick["skipped"] = "no steps"
            return None
        _cont, c, half, z_recv = recv
        p_touch = np.asarray(self._pose()[0], dtype=float).copy()
        hlo, hhi = self._held_boxes(R_rel, t_rel, p_touch, R_hand).aabb()
        c_h = 0.5 * (hlo[:, :2].min(0) + hhi[:, :2].max(0))
        r_h = 0.5 * float((hhi[:, :2].max(0) - hlo[:, :2].min(0)).max())
        # where the object can wait: across the arm's reach before along it (a stretched arm moves across, and may
        # not come back in: 2026-11-16, two of eight); seen from afar, the place nearest to where the object is
        spots = self._cover_spots(poses, c, half, z_recv, r_h, float(hlo[:, 2].min()) - float(p_touch[2]))
        if not spots:
            pick["skipped"] = "no place to wait"
            return None
        off = p_touch[:2] - c_h  # the hand from the held object's middle
        # a round thing held by its rim is the same from every side: the hand turns for the thing while it still
        # holds the object, sets it down turned, and takes it again as it is (a turn on the way to the thing
        # cost 10 to 17 steps on the Mac, and another on the way back)
        can_turn = bool(gap_held <= ap.rim_width_m and self._round_held())

        def turned(take_):  # (the turn for the thing, the hand so turned, where it then holds an object at ``park``)
            ang_ = self._cover_turn(R_hand, take_[3], pick)
            cz_, sz_ = math.cos(ang_), math.sin(ang_)
            R_g_ = np.array([[cz_, -sz_, 0.0], [sz_, cz_, 0.0], [0.0, 0.0, 1.0]]) @ np.asarray(R_hand, dtype=float)
            off_ = np.array([cz_ * off[0] - sz_ * off[1], sz_ * off[0] + cz_ * off[1]])
            return ang_, R_g_, off_

        if seen is not None:
            take = seen
            ang, R_g, off_t = turned(take)
            first = can_turn and ang != 0.0
            R_set, off_s = (R_g, off_t) if first else (R_hand, off)
            _a, _r, park, z_park = min(spots, key=lambda x: (round(float(np.linalg.norm(x[2] - c_h)), 2), x[0], x[1]))
            p_side = np.array([park[0] + off_s[0], park[1] + off_s[1], z_park + 0.03])
            pick.update(park=[round(float(x), 4) for x in park], seen=True)
            st = yield from self._watch_carry(self._goto(p_side, R_set, 1.0, ap.transit_cap, ap.transit_tol_m,
                                                         10.0 if first else 180.0, max_steps=30), gap_held)
            pick["aside"] = st
            if st == "lost":
                return "lost"
            if st != "reached":
                return None
        else:
            _a, _r, park, z_park = min(spots, key=lambda x: (x[1], x[0]))  # (the nearest to the shoulder)
            z_up = float(p_touch[2]) + ap.cover_rise_m
            p_side = np.array([park[0] + off[0], park[1] + off[1], z_up])
            pick["park"] = [round(float(x), 4) for x in park]
            st = yield from self._watch_carry(self._goto(p_side, R_hand, 1.0, ap.transit_cap, ap.transit_tol_m, 180.0,
                                                         max_steps=25), gap_held)
            pick["aside"] = st
            if st == "lost":
                return "lost"
            p_back = np.array([p_rel[0], p_rel[1], z_up])
            if st != "reached":
                yield from self._watch_carry(self._goto(p_back, R_hand, 1.0, ap.transit_cap, ap.transit_tol_m, 180.0,
                                                        max_steps=25), gap_held)
                return "back"
            on = self.oracle.on_receiver(self.spec["place_instance"])
            pick["on"] = None if on is None else {k: on.get(k) for k in ("xy", "half_xy", "yaw", "z", "n_px", "inside")}
            take, why = self._cover_takeable(on, z_recv, park, r_h, poses, _cont)
            if why:
                pick["refused"] = why
            if take is None:
                st_b = yield from self._watch_carry(self._goto(p_back, R_hand, 1.0, ap.transit_cap, ap.transit_tol_m,
                                                               180.0, max_steps=25), gap_held)
                pick["back"] = st_b
                return "lost" if st_b == "lost" else "back"
            ang, R_g, off_t = turned(take)
            first = can_turn and ang != 0.0
            R_set, off_s = (R_g, off_t) if first else (R_hand, off)
            p_side = np.array([park[0] + off_s[0], park[1] + off_s[1], z_up])
        # the object waits here (one take an episode; the push stays open to a later attempt until the thing is off)
        self.n_picked = 1
        if first:
            pick["turned_hold"] = True
        if first or float(self._pose()[0][2]) - z_park > 0.02:
            # (down at speed to 1.2 cm above, turning where the hand turns, then onto the support)
            st = yield from self._watch_carry(self._goto(np.array([p_side[0], p_side[1], z_park + 0.012]), R_set, 1.0,
                                                         ap.transit_cap, 0.008, 10.0 if first else 180.0,
                                                         max_steps=25, stall_down=True, stall_after_m=0.004),
                                              gap_held)
            if st == "lost":
                return "lost"
        st = yield from self._watch_carry(self._goto(np.array([p_side[0], p_side[1], z_park]), R_set, 1.0,
                                                     ap.approach_cap, 0.006, 180.0, max_steps=20, stall_down=True,
                                                     stall_after_m=0.004), gap_held)
        pick["set_down"] = st
        if st == "lost":
            return "lost"
        self._let_go()
        self._tell_open()
        p_n, _ = self._pose()
        p_set = np.asarray(p_n, dtype=float).copy()  # where the hand let the object go
        yield from self._hold(p_n, R_set, -1.0, ap.cover_open_steps)
        if self.tried:
            self.tried.pop()  # the grasp held: the same hold may be taken again
        pick["t_parked"] = self.t
        # the thing: top-down, the fingers across its short side
        xy_t, a, b, psi, z_top = take
        z_tip = max(z_top - ap.cover_pick_depth_m, z_recv + 0.004)
        p_g = np.array([xy_t[0], xy_t[1], z_tip - self.jaw.finger_z[0]])
        # straight up first: the open fingers stand either side of the waiting object's wall
        z_free = float(p_n[2]) + ap.cover_park_rise_m
        yield from self._goto(np.array([p_n[0], p_n[1], z_free]), R_set, -1.0, ap.transit_cap, 0.02, 180.0,
                              max_steps=8)
        # over the thing, the fingertips 1.2 cm above its top (above the waiting object's rim all the way: it is
        # lower than the thing or the hand could not have rested on the thing), turning on the way
        p_hi = np.array([xy_t[0], xy_t[1], z_top + 0.012 - self.jaw.finger_z[0]])
        pick.update(turn_deg=round(math.degrees(ang), 1), grip=[round(float(x), 4) for x in p_g])
        st = yield from self._goto(p_hi, R_g, -1.0, ap.transit_cap, 0.02, 10.0, max_steps=30)
        pick["over"] = st
        if float(self._pose()[0][2]) - float(p_g[2]) > 0.02:  # (down at speed to 1.2 cm above the hold)
            yield from self._goto(p_g + np.array([0.0, 0.0, 0.012]), R_g, -1.0, ap.transit_cap, 0.008,
                                  ap.reach_tol_deg, max_steps=10, stall_down=True)
        # (the last 1.2 cm with the fingers closing: they start 1.8 cm and more below the thing's top, and take
        # some ten steps to meet on it)
        st = yield from self._goto(p_g, R_g, 1.0 if ap.cover_close_early else -1.0, ap.approach_cap, 0.004,
                                   ap.reach_tol_deg, max_steps=12, stall_down=True)
        p_n, _ = self._pose()
        pick["down"] = st
        pick["down_dz"] = round(float(p_n[2] - p_g[2]), 4)
        gaps = [self._gap()]
        for i in range(ap.close_max):
            p, R = self._pose()
            yield follow(p, R, p_n, R_g, 1.0, ap.slow_cap, ap.rot_cap, self.fp)
            gaps.append(self._gap())
            if i + 1 >= ap.close_window and abs(gaps[-1 - ap.close_window] - gaps[-1]) < ap.close_stop_m:
                break
        g_t = gaps[-1]
        pick["gap"] = round(g_t, 5)
        if g_t <= ap.c1_min_gap_m + 0.005:
            # closed on nothing: the hand opens and the attempt ends (the object is taken again all the same)
            pick["took"] = False
            yield from self._hold(p_n, R_g, -1.0, ap.open_steps)
            yield from self._goto(p_n + np.array([0.0, 0.0, 0.05]), R_g, -1.0, ap.transit_cap, 0.02, 180.0,
                                  max_steps=10)
            pick["steps"] = self.t - pick["t"]
            return "parked"
        pick["t_took"] = self.t
        # off the receiver, to a free place on the hand's way back to the waiting object: the shortest way from
        # the thing to the place and on to the object
        cam = (getattr(self.oracle, "cams", None) or {}).get("agentview")
        v = None
        if cam is not None:
            v = np.asarray(cam["T"], dtype=float)[:2, 3] - c
            v = v / max(float(np.linalg.norm(v)), 1e-9)
        z_low = float(p_g[2]) - (z_recv - 0.02)  # the hand above the thing's underside (taken 2 cm below the
        # receiver's top at most)
        drops = []
        for out in ap.cover_drop_out_m:
            for k in range(18):
                u = np.array([math.cos(k * math.pi / 9.0), math.sin(k * math.pi / 9.0)])
                xy = c + u * (min(float(np.abs(u) @ half), float(half.max())) + out)
                if float(np.linalg.norm(xy - park)) < r_h + a + 0.02:
                    continue
                z_sup = self._support_top(xy, z_recv)
                if not self._spot_ok(poses, xy, a, z_sup + 0.02 + max(z_low, 0.02), exclude=(self.target,)):
                    continue
                # (not between the camera and the receiver when another place is free: the next looks are at them)
                way = float(np.linalg.norm(xy - xy_t)) + float(np.linalg.norm(xy - park))
                drops.append((bool(v is not None and float(u @ v) > 0.5), round(way, 2), xy, z_sup))
        if drops:
            _f, _w, drop, z_sup = min(drops, key=lambda x: x[:2])
        else:  # no free place found: beyond the waiting object, on the line from the receiver through it
            out = park - c
            out = out / max(float(np.linalg.norm(out)), 1e-9)
            drop = park + out * (r_h + a + 0.03)
            z_sup = self._support_top(drop, z_recv)
            pick["drop_free"] = False
        pick["drop"] = [round(float(x), 4) for x in drop]
        # up 3 cm (off the receiver's rim), or 6 cm when the way passes over the waiting object; then across and
        # down in one move, and let go low: the thing's underside 2 cm above the support
        seg = np.asarray(drop, dtype=float) - xy_t
        along = float(np.clip((park - xy_t) @ seg / max(float(seg @ seg), 1e-9), 0.0, 1.0))
        over_obj = float(np.linalg.norm(park - (xy_t + along * seg))) < r_h + a + 0.01
        z_lift = float(p_n[2]) + (0.06 if over_obj else 0.03)
        yield from self._goto(np.array([p_n[0], p_n[1], z_lift]), R_g, 1.0, ap.transit_cap, 0.02, 180.0, max_steps=12)
        pick["gap_up"] = round(self._gap(), 5)
        z_let = z_sup + 0.02 + max(z_low, 0.02)
        if over_obj:
            yield from self._goto(np.array([drop[0], drop[1], z_lift]), R_g, 1.0, ap.transit_cap, 0.03, 180.0,
                                  max_steps=25)
            pick["over_object"] = True
        st = yield from self._goto(np.array([drop[0], drop[1], z_let]), R_g, 1.0, ap.transit_cap, 0.03, 180.0,
                                   max_steps=25)
        pick["carried"] = st
        pick["gap_drop"] = round(self._gap(), 5)
        pick["took"] = bool(self._gap() > ap.c1_min_gap_m + 0.005)
        if pick["took"]:
            self.n_cleared += 1  # the receiver is free: the release runs as after a push
        p_n, _ = self._pose()
        yield from self._hold(p_n, R_g, -1.0, ap.cover_open_steps)
        # (up off the thing before the next move across: the open fingers are level with its top)
        yield from self._goto(p_n + np.array([0.0, 0.0, 0.025]), R_g, -1.0, ap.transit_cap, 0.02, 180.0, max_steps=4)
        pick["steps"] = self.t - pick["t"]
        if ap.cover_retake and pick["took"]:
            # back to where the hand let the object go, as it stood then (turned: the release point over the
            # receiver goes round its middle with the hold)
            p_take, R_take, p_goal = p_set, R_set, np.asarray(p_rel, dtype=float)
            if first:
                p_goal = np.array([p_rel[0] - off[0] + off_t[0], p_rel[1] - off[1] + off_t[1], p_rel[2]])
            got = yield from self._cover_retake(p_take, R_take, gap_held, p_goal, _cont, pick, width)
            if got is not None:
                return got
        return "parked"

    def _round_held(self) -> bool:
        """The target's model is as wide one way as the other and centred on its root (a bowl; not a mug with its
        handle, not a box): its boxes' extents across agree to 6 mm and their middle lies within 4 mm of the root."""
        try:
            b = self.lib.body_boxes(self.ttype, self.lib.types[self.ttype]["root_body"])
            if not b.n:
                return False
            lo, hi = b.aabb()
            lo, hi = lo[:, :2].min(0), hi[:, :2].max(0)
            return bool(abs(float((hi - lo)[0] - (hi - lo)[1])) <= 0.006 and float(np.abs(0.5 * (lo + hi)).max()) <= 0.004)
        except Exception:
            return False

    def _cover_retake(self, p_set, R_hand, gap_held: float, p_rel, cont, rec: dict, width=None):
        """``cover_retake`` (a generator of actions; see the parameter's note): back to where the hand let the object
        go, the same hold again, over the receiver, one look at the receiver, down and open. Returns "retaken"
        (the object was let go over the receiver as first planned), "lost" (it left the hand on the way) or None
        (the hold did not read right: the hand is open above the object, and the next attempt takes it)."""
        ap = self.ap
        r = rec.setdefault("retake", {})
        r["t"] = self.t
        up = np.array([0.0, 0.0, ap.cover_park_rise_m])
        # (on the way the fingers come in to ``cover_pregap_m``: from wide open they take 14 steps to meet on a rim)
        st = yield from self._with_gap(self._goto(p_set + up, R_hand, -1.0, ap.transit_cap, 0.012, 8.0, max_steps=35),
                                       ap.cover_pregap_m if ap.cover_close_early else None)
        r["over"] = st
        if ap.cover_close_early:  # (down at speed to 1.5 cm above the hold, the rest with the fingers closing)
            yield from self._with_gap(self._goto(p_set + np.array([0.0, 0.0, 0.015]), R_hand, -1.0, ap.transit_cap,
                                                 0.006, ap.reach_tol_deg, max_steps=8, stall_down=True),
                                      ap.cover_pregap_m)
        st = yield from self._goto(p_set, R_hand, 1.0 if ap.cover_close_early else -1.0, ap.approach_cap, 0.004,
                                   ap.reach_tol_deg, max_steps=12, stall_down=True)
        p_n, _ = self._pose()
        r.update(down=st, down_dz=round(float(p_n[2] - p_set[2]), 4))
        gaps = [self._gap()]
        for i in range(ap.close_max):
            p, R = self._pose()
            yield follow(p, R, p_n, R_hand, 1.0, ap.slow_cap, ap.rot_cap, self.fp)
            gaps.append(self._gap())
            if i + 1 >= ap.close_window and abs(gaps[-1 - ap.close_window] - gaps[-1]) < ap.close_stop_m:
                break
        g = gaps[-1]
        r.update(gap=round(g, 5), gap_was=round(float(gap_held), 5),
                 **({"width": round(float(width), 5)} if width is not None else {}))
        # the same hold: the close check of the grasp that took it (``width``: that grasp's; a rim hold reads wider
        # at the close than after the lift, where the wall has turned flat between the fingers), or, with no width
        # given, the gap of the lift check to 3 mm under and 4 mm over
        held = self._c1(g, float(width)) if width is not None else \
            (max(gap_held - 0.003, ap.c1_min_gap_m) < g <= gap_held + 0.004)
        if not held:
            yield from self._hold(p_n, R_hand, -1.0, ap.open_steps)
            yield from self._goto(p_n + up, R_hand, -1.0, ap.transit_cap, 0.02, 180.0, max_steps=8)
            r["held"] = False
            return None
        r["held"] = True
        z_c = max(float(p_rel[2]) + 0.03, float(p_n[2]) + 0.03)  # over the receiver's rim with the object's underside
        st = yield from self._watch_carry(self._goto(np.array([p_n[0], p_n[1], z_c]), R_hand, 1.0, ap.transit_cap, 0.02,
                                                     180.0, max_steps=10), g)
        if st != "lost":
            st = yield from self._watch_carry(self._goto(np.array([p_rel[0], p_rel[1], z_c]), R_hand, 1.0,
                                                         ap.transit_cap, ap.transit_tol_m, ap.transit_tol_deg,
                                                         max_steps=30), g)
        r["carried"] = st
        if st == "lost":
            return "lost"
        # the receiver once more (taking the thing off may have moved it): the release moves across with it
        goal = np.asarray(p_rel, dtype=float).copy()
        if self.spec.get("place_instance") is not None and hasattr(self.oracle, "relook_place") and cont is not None:
            root = self.scene["instances"][cont]["root_body_name"]
            try:
                was = pose_of(self.lib.poses_from_percepts(self.oracle.percepts), root)[1][:2].copy()
                rl = self.oracle.relook_place(self.spec["place_instance"])
                if rl is not None:
                    r["relook"] = {k: rl[k] for k in ("moved_m", "assoc") if k in rl}
                    if rl["moved_m"] > ap.relook_min_m:
                        now = pose_of(self.lib.poses_from_percepts(self.oracle.percepts), root)[1][:2]
                        goal[:2] += np.asarray(now, dtype=float) - np.asarray(was, dtype=float)
                        st_m = yield from self._watch_carry(self._goto(np.array([goal[0], goal[1], z_c]), R_hand, 1.0,
                                                                       ap.transit_cap, ap.transit_tol_m,
                                                                       ap.transit_tol_deg, max_steps=20), g)
                        r["relook"]["regoto"] = st_m
                        if st_m == "lost":
                            return "lost"
            except Exception as ex:  # a look that cannot be read: the release as first planned
                r["relook_error"] = str(ex)[:80]
        # (down at speed to 1.2 cm above the release, then as a release is lowered)
        st = yield from self._watch_carry(self._goto(goal + np.array([0.0, 0.0, 0.012]), R_hand, 1.0, ap.transit_cap,
                                                     0.008, ap.reach_tol_deg, max_steps=10, stall_down=True,
                                                     stall_after_m=0.004), g)
        if st != "lost":
            st = yield from self._watch_carry(self._goto(goal, R_hand, 1.0, ap.approach_cap,
                                                         max(ap.reach_tol_m, ap.lower_tol_m), ap.reach_tol_deg,
                                                         max_steps=15, stall_down=True, stall_after_m=0.004), g)
        r["lowered"] = st
        if st == "lost":
            return "lost"
        p_h, _ = self._pose()
        r["t_release"] = self.t
        # (the look tool was not told of this hold: should the episode go on, its next look searches for the object
        # from where the hand lets go)
        self._tell_open(lost=True)
        yield from self._hold(p_h, R_hand, -1.0, ap.release_steps)
        for _ in range(ap.settle_steps):
            yield self._idle()
        yield from self._goto(p_h + np.array([0.0, 0.0, 0.07]), R_hand, -1.0, ap.transit_cap, 0.02, 180.0,
                              max_steps=10)
        r["steps"] = self.t - r["t"]
        return "retaken"

    def _clear_cover(self, plan: dict, rec: dict, R_hand, gap_held: float):
        """The sweep itself (a generator of actions): up, to the side, down beside the thing, across, up, back over
        the release point. Returns "swept", "lost" (the fingers met: the object left the hand) or the step that
        failed. The record is filled in place."""
        ap = self.ap

        def go(p, cap, tol, n):
            return self._watch_carry(self._goto(p, R_hand, 1.0, cap, tol, 180.0, max_steps=n), gap_held)

        self.n_cleared += 1
        t0 = self.t
        # up and to the side in one move (the first centimetre drags the thing a little: no harm)
        st = yield from go(plan["start_up"], ap.transit_cap, ap.transit_tol_m, 30)
        if st == "lost":
            return "lost"
        st = yield from go(plan["start"], ap.approach_cap, 0.006, 15)
        if st == "lost":
            return "lost"
        z_top = plan["up"][2] - ap.cover_rise_m  # the hand's height when the held object rested on the thing
        p_now, _ = self._pose()
        if float(p_now[2] - plan["start"][2]) > 0.006 and float(p_now[2]) > z_top - 0.006:
            # stopped at the thing's own height: it reaches further out than taken. Once more from 4 cm further.
            rec["wider"] = round(float(p_now[2] - plan["start"][2]), 4)
            for k in ("start_up", "start"):
                plan[k] = plan[k] - np.r_[plan["out"], 0.0]
            st = yield from go(np.array([p_now[0], p_now[1], plan["up"][2]]), ap.transit_cap, ap.transit_tol_m, 10)
            if st != "lost":
                st = yield from go(plan["start_up"], ap.transit_cap, ap.transit_tol_m, 12)
            if st != "lost":
                st = yield from go(plan["start"], ap.approach_cap, 0.006, 15)
            if st == "lost":
                return "lost"
            p_now, _ = self._pose()
            if float(p_now[2] - plan["start"][2]) > 0.006 and float(p_now[2]) > z_top - 0.006:
                rec["beside"] = "blocked"
                yield from go(np.array([p_now[0], p_now[1], plan["up"][2]]), ap.transit_cap, ap.transit_tol_m, 10)
                yield from go(plan["back"], ap.transit_cap, ap.transit_tol_m, 30)
                return "blocked"
        if float(p_now[2] - plan["start"][2]) > 0.004:
            # stopped lower, on what the thing lies on (it stands a little higher than it was read): the push runs
            # just above where the hand stopped
            rec["floor_up"] = round(float(p_now[2] - plan["start"][2]), 4)
            plan["end"] = np.array([plan["end"][0], plan["end"][1], float(p_now[2]) + 0.002])
            if plan.get("slow") is not None:
                plan["slow"] = np.array([plan["slow"][0], plan["slow"][1], float(p_now[2]) + 0.002])
        if plan.get("slow") is not None:
            # ``cover_slow_cap``: up to the receiver at speed, then slowly across it (a hand at cap 1 covers 1.1 cm
            # a step: the slow part is given the steps its length needs, and 10 more)
            st = yield from go(plan["slow"], ap.cover_cap, 0.01, 20)
            if st == "lost":
                rec["push"] = st
                return "lost"
            n_slow = int(math.ceil(float(np.linalg.norm((plan["end"] - plan["slow"])[:2]))
                                   / (0.011 * max(float(ap.cover_slow_cap), 0.05)))) + 10
            st = yield from go(plan["end"], float(ap.cover_slow_cap), 0.01, min(n_slow, 70))
        else:
            st = yield from go(plan["end"], ap.cover_cap, 0.01, 45)
        rec["push"] = st
        if st == "lost":
            return "lost"
        p_now, _ = self._pose()
        rec["pushed_m"] = round(float(np.linalg.norm((p_now - plan["start"])[:2])), 4)
        st = yield from go(plan["back"], ap.transit_cap, ap.transit_tol_m, 30)  # up and back in one move
        rec["steps"] = self.t - t0
        return "lost" if st == "lost" else "swept"

    def _expect_drop(self, place: dict, p_hold, att: dict) -> float:
        """``drop_predict``: how far the held object falls when the hand opens here (its lowest point above the
        place surface under it), told to the look tool when it is at least ``drop_min_m``; else 0."""
        ap = self.ap
        if not ap.drop_predict or not hasattr(self.oracle, "drop_m"):
            return 0.0
        try:
            drop = float(p_hold[2]) + float(place["low_rel"]) - float(place["surface"])
        except (KeyError, TypeError, ValueError):
            return 0.0
        if not np.isfinite(drop) or drop < ap.drop_min_m:
            return 0.0
        self.oracle.drop_m = drop
        if ap.drop_rigid and hasattr(self.oracle, "drop_rigid"):
            self.oracle.drop_rigid = True
        att["drop_m"] = round(drop, 4)
        return drop

    def _close_look(self, inst: str):
        """Item 3: one wrist look at ``inst`` (a percept id). Returns (the move to apply, or None; the record, or None
        when no look was taken)."""
        ap = self.ap
        if self.n_close >= ap.close_looks_max or not hasattr(self.oracle, "close_look"):
            return None, None
        self.n_close += 1
        cl = self.oracle.close_look(inst, min_move_m=ap.close_min_m)  # the tool updates the percept only for a move
        rec = {k: cl[k] for k in ("status", "n_px", "jump_m", "move", "turn_deg") if cl.get(k) is not None}
        mv = cl.get("move")
        if cl.get("status") != "tracked" or mv is None or not cl.get("updated"):
            return None, rec
        rec["applied"] = True
        return np.asarray(mv, dtype=float), rec

    def _c3(self, reads, poses_last) -> tuple[bool, dict]:
        ap = self.ap
        pts = [np.asarray(r[self.troot]["pos"]) for r in reads]
        moves = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(pts) - 1)]
        rest = bool(moves) and max(moves) < ap.rest_m
        pos = pts[-1]
        goal_xy, cont, region = self._place_goal(poses_last)
        if region is not None:
            c, h = region["centre"], region["half"]
            inner = np.maximum(h - ap.c3_shrink_m, 0.0)
            if self.spec.get("predicate") == "on":  # resting on a surface region: above it, not inside it
                inside = bool(np.all(np.abs(pos[:2] - c[:2]) <= inner[:2])
                              and (c[2] - h[2] - 0.03) < pos[2] < (c[2] + h[2] + 0.25))
            else:
                inside = bool(np.all(np.abs(pos[:2] - c[:2]) <= inner[:2])
                              and (c[2] - h[2] - 0.01) < pos[2] < (c[2] + h[2]))
        else:
            _, tp = pose_of(poses_last, self.scene["instances"][cont]["root_body_name"])
            dz = float(pos[2] - tp[2])
            dxy = float(np.linalg.norm(pos[:2] - tp[:2]))
            inside = bool(dxy <= ap.c3_plate_xy_m and ap.c3_plate_dz_m[0] <= dz <= ap.c3_plate_dz_m[1])
            return rest and inside, {"rest": rest, "inside": inside, "dxy_m": round(dxy, 4), "dz_m": round(dz, 4),
                                     "max_move_m": round(max(moves), 5) if moves else None}
        return rest and inside, {"rest": rest, "inside": inside, "max_move_m": round(max(moves), 5) if moves else None,
                                 # the record only: the object's origin from the region's centre, and the region's size
                                 "off_m": [round(float(x), 4) for x in pos - c],
                                 "half_m": [round(float(x), 4) for x in h]}

    def _c3_measure(self, att: dict):
        """``c3_measured``: the placed check passed on a pose no look has measured since the hand let go. With the
        hand backed off, the look tool searches for the object once (a full look, at any distance from where the hand
        left it), then the check is read again. Returns (the check, its reads): passed only when a look measured the
        object and the measured pose passes. The attempt's record keeps the first verdict as ``unmeasured``."""
        first = {k: att["c3"].get(k) for k in ("rest", "inside", "max_move_m", "dxy_m", "dz_m") if k in att["c3"]}
        n0 = len(self.oracle.looks)
        found = bool(self.oracle.recheck_target())
        reads = []
        for _ in range(max(2, int(self.ap.c3_measured_reads))):
            yield self._idle()
            reads.append(self.oracle())
        measured = found or bool(self.oracle.target_measured(n0))
        c3, d2 = self._c3(reads, reads[-1])
        att["c3"].update({"unmeasured": first, "measure_t": self.t, **d2, "measured": measured,
                          "found": found, "pass": bool(c3 and measured)})
        att["t_end"] = self.t
        return bool(c3 and measured), reads

    def _c3_again(self, att: dict, c3d: dict):
        """``c3_recheck``: the placed check read again, the hand backed off (a generator: one idle step a read).
        Returns (the check, its reads); the attempt's record keeps the first check's readings as ``first``."""
        reads = []
        for _ in range(max(2, int(self.ap.c3_recheck_steps))):
            yield self._idle()
            reads.append(self.oracle())
        c3, d2 = self._c3(reads, reads[-1])
        att["c3"].update({"first": {k: c3d.get(k) for k in ("rest", "inside", "max_move_m")}, "recheck_t": self.t,
                          **d2, "pass": c3})
        att["t_end"] = self.t
        return c3, reads

    # ------------------------------------------------------------------ the episode
    def _episode(self):
        yield None  # primed by __init__
        ap = self.ap
        while True:
            k = len(self.attempts)
            if k >= ap.max_attempts:
                self._finish("max_attempts")
                return
            need = ap.T_attempt
            if getattr(self, "_retake_need", None) is not None:
                # ``cover_pick``: the object waits beside the receiver, and this attempt takes it again from there
                need, self._retake_need = min(need, int(self._retake_need)), None
            if k > 0 and self.cap - self.t < need:
                self._finish("no_budget")
                return
            att = {"index": k, "t_start": self.t}
            self.attempts.append(att)
            if k > 0 and ap.open_wait > 0:  # the look below is taken with the hand open (see ``open_wait``)
                n = 0
                while n < ap.open_wait and self._gap() < 2.0 * self.jaw.L - ap.open_wait_gap_m:
                    yield self._idle()
                    n += 1
                if n:
                    att["open_wait"] = n
            try:
                poses = self.oracle()
                self._last_poses = poses
            except Exception as ex:  # a degenerate look (e.g. an empty body list): keep the last measured scene
                if self._last_poses is None:
                    raise
                self._ev("look_error", err=str(ex)[:120])
                poses = self._last_poses
            if self._empty is not None:  # ``empty_view_m``: the last close was on nothing
                was, p_view, R_view = self._empty
                self._empty = None
                if float(np.linalg.norm(pose_of(poses, self.troot)[1] - was)) < ap.empty_same_m:
                    # this look did not find the target anywhere else: once more, from where the hand is out of the way
                    yield from self._goto(p_view, R_view, -1.0, ap.transit_cap, ap.transit_tol_m, 180.0, max_steps=30)
                    att["view_m"] = round(float(ap.empty_view_m), 3)
                    again = self._look_redetect()
                    if again is not None:
                        poses = again
            try:
                frames, diag = self._plan_grasp(poses)
            except Exception as ex:  # a degenerate geometry in planning ends the episode cleanly, not as a crash
                self._ev("plan_error", err=str(ex)[:120])
                frames, diag = [], {"error": str(ex)[:120]}
            att["plan"] = {kk: v for kk, v in diag.items() if kk != "replanned_for_current_up"}
            if "replanned_for_current_up" in diag:
                att["plan"]["replanned"] = {kk: v for kk, v in diag["replanned_for_current_up"].items()}
            if not frames:
                att["outcome"] = "no_plan"
                self._finish("no_plan")
                return
            fr = frames[0]
            c_w = fr["centre"]
            self.tried.append((fr["c_obj"], fr["a_obj"]))
            att["frame"] = {"record": fr["record"], "width": round(fr["width"], 5), "z_rel": round(fr["z_rel"], 4),
                            "com_dist": round(fr["com_dist"], 4), "yaw_change_deg": round(fr["yaw_change_deg"], 1),
                            "reach_m": round(fr["reach_m"], 4), "yaw_deg": round(math.degrees(fr["yaw"]), 1),
                            "centre": [round(float(x), 4) for x in c_w]}
            c, Rg = fr["centre"], fr["R_eef"]
            t_top = float(self._instance_boxes(poses, self.target).aabb()[1][:, 2].max())
            pre_z = max(c[2] + ap.pre_height_min_m, t_top + ap.pre_top_clear_m - self.jaw.finger_z[0])
            p_pre = np.array([c[0], c[1], min(pre_z, c[2] + ap.pre_height_m)])
            # 2. pre-grasp
            att["t_pre"] = self.t
            if ap.route_empty and ap.transit_unexplained and ap.look_start and (ap.look_cross or ap.look_stall > 0) \
                    and int(getattr(self, "lag_steps", 0) or 0) <= 0 and not att.get("route"):
                # (2026-11-18) the empty hand's way is planned on a look for the way too: the first look is taken as
                # every look is, with the robot's own arm left standing in it where the depth is off
                self._route_look(att, "start")
            st = yield from self._route_watch(
                self._move_via(poses, p_pre, Rg, -1.0, (self.target,), self.jaw.L, None,
                               ap.empty_clear_m, ap.pre_tol_m, ap.pre_tol_deg),
                att, poses, p_pre, Rg, self.jaw.L, None, (self.target,), ap.empty_clear_m)
            n_seen = 0
            while st == "route_seen" and n_seen < 2:  # (``look_stall`` with ``route_empty``: a new thing on the way)
                n_seen += 1
                yield from self._route_back(Rg, -1.0)
                st = yield from self._route_watch(
                    self._move_via(poses, p_pre, Rg, -1.0, (self.target,), self.jaw.L, None,
                                   ap.empty_clear_m, ap.pre_tol_m, ap.pre_tol_deg),
                    att, poses, p_pre, Rg, self.jaw.L, None, (self.target,), ap.empty_clear_m)
            if st == "route_seen":
                st = "unreachable"
            if st != "reached":
                p_now, _ = self._pose()
                if float(np.linalg.norm(p_now - p_pre)) <= ap.near_pre_m:
                    att["pre_near"] = round(float(np.linalg.norm(p_now - p_pre)), 4)  # a waypoint: close is enough
                else:
                    att["outcome"] = "unreachable"
                    att["stage"] = "pre_grasp:" + st
                    self._lower_reach(fr)
                    continue
            # 2b. item 3 (off by default): the close look before closing. The target as the wrist camera sees it from
            # here; the grasp moves with it (the move is measured from the pose this attempt planned on)
            if ap.close_look_grasp:
                mv, cl = self._close_look(self.target)
                if cl is not None:
                    att["close"] = cl
                if mv is not None:
                    c, p_pre = c + mv, p_pre + mv
                    att["close"]["regoto"] = yield from self._goto(p_pre, Rg, -1.0, ap.slow_cap, ap.pre_tol_m,
                                                                   ap.pre_tol_deg, max_steps=30)
            # 2c. ``pre_look`` (off by default): is the target still where this attempt was planned on?
            if ap.pre_look and self.n_pre_look < ap.pre_look_max:
                gone = self._pre_look(poses)
                if gone is not None:
                    att["pre_look_m"] = round(gone, 4)
                    if gone >= ap.pre_look_min_m:
                        self.n_pre_look += 1
                        self.tried.pop()  # this frame was never tried; it may be the best one where the target is now
                        att["outcome"] = "moved_before_grasp"
                        continue
            # 3. approach
            pre_fix = None
            if ap.lag_settle_pre and self._cmds_late():
                # (2026-11-17, off by default) the lag waited out and the hand put truly over the grasp point here,
                # before it goes down (see ``lag_settle_pre``)
                rounds = []
                for _ in range(max(1, int(ap.lag_pre_rounds))):
                    fx = yield from self._settle_fix(p_pre, Rg, -1.0, ap.slow_cap, ap.reach_tol_m, ap.pre_tol_deg)
                    if fx is None:  # (the delay ended meanwhile)
                        break
                    rounds.append(fx)
                    if fx["left_m"] <= ap.lag_pre_ok_m:
                        break
                if rounds:
                    lag_now, obs_now = int(getattr(self, "lag_steps", 0) or 0), int(getattr(self, "lag_obs_steps", 0) or 0)
                    if lag_now > 0:  # the last correcting move's commands are still on their way
                        yield from self._hold(p_pre, Rg, -1.0, (lag_now - obs_now - 1 if obs_now > 0 else lag_now) + 1)
                    pre_fix = {"left_m": rounds[0]["left_m"], "status": rounds[-1]["status"], "rounds": len(rounds)}
                    if len(rounds) > 1:
                        pre_fix["left_last_m"] = rounds[-1]["left_m"]
                    att["pre_fix"] = pre_fix
            att["t_approach"] = self.t
            st = yield from self._closing_early(self._goto(c, Rg, -1.0, ap.approach_cap, ap.reach_tol_m,
                                                           ap.reach_tol_deg, stall_down=True), c, fr["width"])
            if st == "stalled" and ap.stall_retry and self.n_stall_retry < ap.stall_retry_max:
                # a fast descent deflected on contact: back off a little, re-centre over the grasp xy precisely at
                # that height, and descend once more slowly (cheaper than a new attempt from the pre-grasp)
                p_now, _ = self._pose()
                exy0 = float(np.linalg.norm((p_now - c)[:2]))
                if p_now[2] - c[2] > ap.reach_tol_m and exy0 <= ap.stall_retry_max_xy_m:
                    self.n_stall_retry += 1
                    att["stall_retry"] = {"exy": round(exy0, 4), "ez": round(float(p_now[2] - c[2]), 4)}
                    z_mid = min(float(p_pre[2]), float(p_now[2]) + ap.stall_retry_rise_m)
                    p_mid = np.array([c[0], c[1], z_mid])
                    yield from self._goto(p_mid, Rg, -1.0, ap.slow_cap, ap.stall_retry_tol_m, ap.reach_tol_deg,
                                          max_steps=ap.stall_retry_steps)
                    st = yield from self._goto(c, Rg, -1.0, ap.slow_cap, ap.reach_tol_m, ap.reach_tol_deg,
                                               max_steps=ap.stall_retry_steps + 10, stall_down=True)
                    att["stall_retry"]["status"] = st
            if st == "stalled" and ap.stall_unseen_m > 0 and self._target_source() in ("wrist", "behind"):
                # ``stall_unseen_m``: the open hand stopped on a target whose height was never seen
                p_now, _ = self._pose()
                exy, ez = float(np.linalg.norm((p_now - c)[:2])), float(p_now[2] - c[2])
                if exy <= ap.stall_retry_max_xy_m and ap.near_z_m < ez <= ap.stall_unseen_m \
                        and hasattr(self.oracle, "raise_target") and self.oracle.raise_target(ez):
                    att["raised"] = {"ez": round(ez, 4), "exy": round(exy, 4)}
                    c = np.array([c[0], c[1], float(p_now[2])])
                    st = "reached"
            if st in ("unreachable", "stalled"):
                # accept a slow or stalled approach when the residual error is inside this record's jaw clearance
                # (a hand blocked on top of something stalls well above the 1 cm vertical tolerance)
                p_now, _ = self._pose()
                exy = float(np.linalg.norm((p_now - c)[:2]))
                ez = abs(float(p_now[2] - c[2]))
                clear = min(ap.near_grasp_max_m, 0.5 * (2 * self.jaw.L - fr["width"]) - ap.near_grasp_margin_m)
                if exy <= clear and ez <= ap.near_z_m:
                    att["approach_near"] = {"status": st, "exy": round(exy, 4), "ez": round(ez, 4),
                                            "allowed": round(clear, 4)}
                    st = "reached"
            if st != "reached":
                att["outcome"] = "approach_" + st
                if st == "unreachable":
                    self._lower_reach(fr)
                yield from self._goto(p_pre, Rg, -1.0, ap.transit_cap, ap.transit_tol_m, 180.0, max_steps=30)
                continue
            if pre_fix is not None and int(getattr(self, "lag_steps", 0) or 0) > 0:
                fix = None  # (``lag_settle_pre``: waited and corrected before the descent; the closing corrects on)
                att["approach_fix"] = {"skipped": "pre"}
            else:
                fix = yield from self._settle_fix(c, Rg, -1.0, ap.slow_cap, ap.reach_tol_m, ap.reach_tol_deg)
            if fix is not None:
                att["approach_fix"] = fix
            # 4. close, C1
            self._held_limit(fr["width"])
            att["t_close"] = self.t
            # ``lag_close_blind`` (2026-11-17, off unless the delay compensator's ``close_hold_cmd`` is on; only with
            # ``carry_watch``): through a delay the gap read now is of fingers that have not begun to move, and the
            # reading of the shut fingers comes the readings' delay after they shut. The hand closes for as long as the
            # fingers take, lifts, and reads its grasp check then (below, after the lift)
            blind = bool(getattr(self, "lag_close_blind", False)) and bool(ap.carry_watch) \
                and int(getattr(self, "lag_steps", 0) or 0) > 0
            t_shut, obs_blind = None, int(getattr(self, "lag_obs_steps", 0) or 0)
            if blind:
                gap0 = self._gap()
                n_blind = int(np.clip(math.ceil(max(0.0, gap0 - fr["width"]) / max(ap.close_blind_rate_m, 1e-4))
                                      + ap.close_blind_margin, 3, ap.close_max))
                for i in range(n_blind):
                    if int(getattr(self, "lag_steps", 0) or 0) <= 0:
                        # the lag ended while the hand closed (the fingers answered at once: ``grip_echo``): the
                        # hand was steered here by a lag that was no longer there. Open, onto the grasp point by
                        # the readings as they now are, and the ordinary close
                        blind = False
                        att["blind_ended"] = self.t
                        yield from self._goto(c, Rg, -1.0, ap.slow_cap, ap.reach_tol_m, ap.reach_tol_deg,
                                              max_steps=25)
                        break
                    p, R = self._pose()
                    yield follow(p, R, c, Rg, 1.0, ap.slow_cap, ap.rot_cap, self.fp)
                t_shut = self.t
                gap, c1, stable, soft, side = float(fr["width"]), True, True, False, None  # (read after the lift)
            if not blind:
                gaps = [self._gap()]
                wait = self._close_wait()  # (0 unless the closing hold was shortened and this close is not blind)
                for i in range(ap.close_max + wait):
                    p, R = self._pose()
                    yield follow(p, R, self._braked(c) if ap.brake_k > 0 else c, Rg, 1.0, ap.slow_cap, ap.rot_cap,
                                 self.fp)
                    gaps.append(self._gap())
                    if i + 1 >= ap.close_window + wait \
                            and abs(gaps[-1 - ap.close_window] - gaps[-1]) < ap.close_stop_m:
                        break
                gap = gaps[-1]
                c1 = self._c1(gap, fr["width"])
                stable = len(gaps) > ap.close_window and abs(gaps[-1 - ap.close_window] - gaps[-1]) < ap.close_stop_m
                soft = (not c1) and self._c1_soft(gap, fr["width"], stable)
                side = None
                if not (c1 or soft) and ap.c1_stand and stable:  # a stood model, held across its other side
                    side = self._stood_other_side(fr["width"])
                    if side is not None and self._c1(gap, side):
                        soft = True
                    else:
                        side = None
            try:  # the record only: where the object would sit in the hand had it not moved since the look
                att["in_hand"] = {"look": [round(float(x), 4) for x in self._rel(poses)[1]]}
            except Exception:
                att["in_hand"] = {}
            poses_c = self.oracle()
            Rr_c, tr_c = self._rel(poses_c)
            att["in_hand"].update(close=[round(float(x), 4) for x in tr_c], close_src=self._target_source())
            att["c1"] = {"t": self.t, "gap": round(gap, 5), "pred": round(fr["width"], 5), "pass": c1}
            if blind:
                att["c1"] = {"t": self.t, "pred": round(fr["width"], 5), "blind": n_blind}  # (filled after the lift)
            if soft:  # the band check failed (recorded as it is); the lift decides
                att["c1"]["soft"] = True
                if side is not None:
                    att["c1"]["stood_side"] = round(float(side), 5)
            if not (c1 or soft):
                att["outcome"] = "c1_fail"
                self._let_go()
                self._tell_open()
                if ap.empty_next and gap < ap.c1_min_gap_m and hasattr(self.oracle, "refute_target"):
                    nx = self.oracle.refute_target()
                    if nx is not None:
                        att["refuted"] = nx
                        self.tried = []  # another place: every frame is untried there
                yield from self._hold(c, Rg, -1.0, ap.open_steps)
                if ap.empty_view_m > 0 and gap < ap.c1_min_gap_m and c[2] + ap.empty_view_m > p_pre[2]:
                    # closed on nothing: the next attempt's looks settle where it is (see ``empty_view_m``)
                    self._empty = (pose_of(poses_c, self.troot)[1].copy(),
                                   np.array([c[0], c[1], c[2] + ap.empty_view_m]), Rg)
                yield from self._goto(p_pre, Rg, -1.0, ap.transit_cap, ap.transit_tol_m, 180.0, max_steps=30)
                continue
            # 5. lift (to the transport's crossing height, between lift_min_m and lift_m), C2
            att["t_lift"] = self.t
            p_c, R_c = self._pose()
            goal_xy, _cont, _reg = self._place_goal(poses_c)
            obj_c = R_c @ tr_c + p_c
            hand_goal = np.r_[goal_xy + (p_c[:2] - obj_c[:2]), p_c[2]]
            z_need = self._cross_z(poses_c, p_c, R_c, hand_goal, R_c, 0.5 * gap, (Rr_c, tr_c), (self.target,),
                                   ap.transit_margin_m)
            lift = float(np.clip(z_need - c[2], ap.lift_min_m, ap.lift_m))
            att["lift_m"] = round(lift, 4)
            if ap.lift_fast and int(getattr(self, "lag_steps", 0) or 0) <= 0:
                # one move up (see ``lift_fast``): no stop after the first centimetres, and none at the lift height
                # when the transport has to rise at least 3 cm more
                more = (z_need - c[2]) > lift + 0.03
                yield from self._goto(c + np.array([0, 0, min(ap.lift_slow_m, lift)]), Rg, 1.0, ap.approach_cap,
                                      0.008, 180.0, max_steps=30)
                yield from self._goto(c + np.array([0, 0, lift]), Rg, 1.0, ap.transit_cap,
                                      0.03 if more else ap.transit_tol_m, 180.0, max_steps=40)
            else:
                yield from self._goto(c + np.array([0, 0, min(ap.lift_slow_m, lift)]), Rg, 1.0, ap.slow_cap,
                                      ap.reach_tol_m, 180.0, max_steps=30)
                yield from self._goto(c + np.array([0, 0, lift]), Rg, 1.0, ap.transit_cap, ap.transit_tol_m, 180.0,
                                      max_steps=40)
            if blind:
                # the reading of the shut fingers is in the readings' delay (and a step) after they shut
                p_l, _ = self._pose()
                while self.t - t_shut < obs_blind + 1:
                    yield follow(*self._pose(), p_l, Rg, 1.0, ap.slow_cap, ap.rot_cap, self.fp)
                gaps = [self._gap()]
                # fingers not read at the planned width may still be on their way (to nothing, to something thinner,
                # or a step short of the object): read on until they stop
                while not self._c1(gaps[-1], fr["width"]) and gaps[-1] > ap.c1_min_gap_m \
                        and len(gaps) <= ap.close_max:
                    yield follow(*self._pose(), p_l, Rg, 1.0, ap.slow_cap, ap.rot_cap, self.fp)
                    gaps.append(self._gap())
                    if len(gaps) > ap.close_window and abs(gaps[-1 - ap.close_window] - gaps[-1]) < ap.close_stop_m:
                        break
                gap = gaps[-1]
                c1 = self._c1(gap, fr["width"])
                soft = (not c1) and self._c1_soft(gap, fr["width"], True)
                if not (c1 or soft) and ap.c1_stand:  # a stood model, held across its other side
                    side = self._stood_other_side(fr["width"])
                    if side is not None and self._c1(gap, side):
                        soft = True
                    else:
                        side = None
                att["c1"].update(t=self.t, gap=round(gap, 5), **{"pass": c1})
                if soft:
                    att["c1"]["soft"] = True
                    if side is not None:
                        att["c1"]["stood_side"] = round(float(side), 5)
                if not (c1 or soft):
                    att["outcome"] = "c1_fail"
                    if gap >= ap.c1_min_gap_m:
                        # something of another width is between the fingers: back down to where it was taken
                        yield from self._goto(c + np.array([0, 0, 0.01]), Rg, 1.0, ap.transit_cap, ap.transit_tol_m,
                                              180.0, max_steps=30)
                    self._let_go()
                    self._tell_open()
                    if ap.empty_next and gap < ap.c1_min_gap_m and hasattr(self.oracle, "refute_target"):
                        nx = self.oracle.refute_target()
                        if nx is not None:
                            att["refuted"] = nx
                            self.tried = []  # another place: every frame is untried there
                    yield from self._hold(self._pose()[0], Rg, -1.0, ap.open_steps)
                    if ap.empty_view_m > 0 and gap < ap.c1_min_gap_m and c[2] + ap.empty_view_m > p_pre[2]:
                        self._empty = (pose_of(poses_c, self.troot)[1].copy(),
                                       np.array([c[0], c[1], c[2] + ap.empty_view_m]), Rg)
                    yield from self._goto(p_pre, Rg, -1.0, ap.transit_cap, ap.transit_tol_m, 180.0, max_steps=30)
                    continue
            stale = False
            if (blind or ap.lag_lift_no_settle) and ap.carry_watch and int(getattr(self, "lag_steps", 0) or 0) > 0:
                # (2026-11-17, off by default) no wait of a whole lag here. The look below then shows the hand some
                # steps back, so it cannot say whether the object came up: the fingers say whether they still hold
                # what they closed on, where it sits in the hand is the close's own measure, and an object left
                # behind shows in the fingers during the carry (``carry_watch``)
                att["lift_settle"] = "skipped"
                stale = True
            else:
                yield from self._settle(1.0)  # through a delay the picture still shows the hand below (``_settle``)
            poses_l = self.oracle()
            Rr, tr = self._rel(poses_l)
            if stale:
                Rr, tr = Rr_c, tr_c
                if hasattr(self.oracle, "rehold"):
                    self.oracle.rehold(Rr_c, tr_c)
            att["in_hand"].update(lift=[round(float(x), 4) for x in tr], lift_src=self._target_source())
            gap2 = self._gap()
            moved = float(np.linalg.norm(tr - tr_c))
            rel_tol = ap.c2_rim_rel_m if fr["width"] <= ap.rim_width_m else ap.c2_rel_m
            if int(getattr(self, "lag_steps", 0) or 0) > 0:
                # through a delay the lift is one burst of commands and the object shifts in the fingers (2026-11-15,
                # S20: 1.0-1.6 cm with the gap unchanged, where an undelayed lift shows 1-3 mm): it is still held, and
                # the release is planned from where it now sits in the hand. A lost object still shows in the gap.
                rel_tol = max(rel_tol, ap.c2_lag_rel_m)
            held = soft or self._c2_gap(gap2, fr["width"])
            c2 = held and not soft and moved < rel_tol
            att["c2"] = {"t": self.t, "gap": round(gap2, 5), "rel_move_m": round(moved, 4), "pass": c2}
            if stale:
                att["c2"]["stale"] = True  # (the look was not waited for: the fingers alone)
            if not c2 and held and (soft or ap.c2_trust_gap):
                rise = float(self._pose()[0][2] - p_c[2])
                if self._c2_fingers(gap, gap2, moved, rise, rel_tol):
                    c2 = True
                    att["c2"].update({"pass": True, "by": "fingers", "rise_m": round(rise, 4)})
                    if moved >= rel_tol:
                        # the fingers have not moved on what they hold, so it has not moved in them: the look that
                        # read it elsewhere is not believed, and the pose measured at the close stands
                        Rr, tr = Rr_c, tr_c
                        if hasattr(self.oracle, "rehold"):
                            self.oracle.rehold(Rr_c, tr_c)
            if not c2:
                att["outcome"] = "c2_fail"
                yield from self._goto(c + np.array([0, 0, 0.01]), Rg, 1.0, ap.transit_cap, ap.transit_tol_m, 180.0,
                                      max_steps=30)
                self._let_go()
                self._tell_open()
                yield from self._hold(c + np.array([0, 0, 0.01]), Rg, -1.0, ap.open_steps)
                yield from self._goto(p_pre, Rg, -1.0, ap.transit_cap, ap.transit_tol_m, 180.0, max_steps=30)
                continue
            # 6. transport
            place = self._plan_place(poses_l, Rr, tr, gap2)
            if place is None:
                att["outcome"] = "no_place"
                self._let_go()
                self._tell_open(lost=True)  # let go where the hand is: it falls from here
                yield from self._hold(self._pose()[0], Rg, -1.0, ap.open_steps)
                continue
            att["place"] = {"z_release": round(place["z_release"], 4), "z_top": round(place["z_top"], 4),
                            "z_lo": round(place["z_lo"], 4), "surface": round(place["surface"], 4)}
            if ap.place_free_xy or self.spec.get("place_share"):  # the long-suite work: what stood near the release
                att["place"].update(xy=[round(float(x), 4) for x in place["hand_xy"]], free_xy=bool(place.get("free_xy")),
                                    blocked_by=place.get("blocked_by"), near=self._near(poses_l, place["hand_xy"]))
            att["t_transport"] = self.t
            _, R_now = self._pose()
            p_rel = np.array([place["hand_xy"][0], place["hand_xy"][1], place["z_release"]])
            p_above = p_rel + np.array([0, 0, ap.stage_above_m if (ap.trust_place and ap.stage_above_m > 0)
                                        else ap.lower_slow_m])
            # with ``clear_cover`` a hand stopped from below on its way down is known at once (3 steps without
            # coming down; the no-progress window took 21 steps in the first smoke)
            feel = bool(ap.clear_cover and not ap.trust_place and self.n_cleared == 0)
            p_t0 = np.array(self._pose()[0], dtype=float)
            st = yield from self._watch_see(self._watch_carry(self._watch_block(self._route_watch(
                self._move_via(poses_l, p_above, R_now, 1.0, (self.target,), 0.5 * gap2, (Rr, tr),
                               ap.transit_margin_m, ap.transit_tol_m, ap.transit_tol_deg,
                               **({"stall_last": True} if feel else {})),
                att, poses_l, p_above, R_now, 0.5 * gap2, (Rr, tr), (self.target,), ap.transit_margin_m),
                p_above), gap2), p_above, poses_l, Rr, tr, att)
            n_seen = 0
            while st == "route_seen" and n_seen < 3:
                # (``look_cross``, ``look_stall``) a look on the way showed a thing the plan did not count: off it
                # when the hand rests against it, and the way planned again from here with what the look shows
                n_seen += 1
                yield from self._route_back(R_now, 1.0)
                st = yield from self._watch_carry(self._watch_block(self._route_watch(
                    self._move_via(poses_l, p_above, R_now, 1.0, (self.target,), 0.5 * gap2, (Rr, tr),
                                   ap.transit_margin_m, ap.transit_tol_m, ap.transit_tol_deg,
                                   **({"stall_last": True} if feel else {})),
                    att, poses_l, p_above, R_now, 0.5 * gap2, (Rr, tr), (self.target,), ap.transit_margin_m),
                    p_above), gap2)
            if st == "route_seen":
                st = "blocked"
            if st == "cover_seen":
                # ``cover_see``: a thing the fingers can take lies on the receiver; the object waits beside it
                crec_s = {"t": self.t, "seen": True}
                att["cover"] = crec_s
                picked_s = yield from self._cover_pick(poses_l, p_rel, R_now, Rr, tr, gap2, crec_s,
                                                       seen=self._cover_take, width=fr["width"])
                if picked_s == "lost":
                    att.update(outcome="lost_in_carry", t_lost=self.t)
                    yield from self._hold(self._pose()[0], R_now, -1.0, ap.open_steps)
                    continue
                if picked_s in ("parked", "retaken"):
                    att["outcome"] = picked_s
                    self._retake_need = ap.cover_retake_need
                    continue
                st = yield from self._watch_carry(self._watch_block(  # (no place to wait: the carry as planned)
                    self._move_via(poses_l, p_above, R_now, 1.0, (self.target,), 0.5 * gap2, (Rr, tr),
                                   ap.transit_margin_m, ap.transit_tol_m, ap.transit_tol_deg,
                                   **({"stall_last": True} if feel else {})), p_above), gap2)
            if ap.carry_relook and st not in ("reached", "lost") and int(getattr(self, "lag_steps", 0) or 0) <= 0:
                p_b = np.array(self._pose()[0], dtype=float)
                if float(np.linalg.norm((p_above - p_b)[:2])) > ap.detour_min_m:
                    # blocked far from the goal by something the looks so far did not show: off it by a few
                    # centimetres the way the hand came, the scene looked at again; when that look shows a thing
                    # the carry did not count before, the way is planned again
                    back = (p_t0 - p_b)[:2]
                    nb = float(np.linalg.norm(back))
                    rec_rl = {"t": self.t, "was": st, "at": [round(float(x), 4) for x in p_b]}
                    att["carry_relook"] = rec_rl
                    z_keep = getattr(self, "_z_cross", None)  # (the detour rises above the first plan's crossing)
                    before = self._unexplained()
                    if nb > 1e-6 and ap.carry_relook_back_m > 0:
                        yield from self._goto(p_b + np.array([*(back / nb * ap.carry_relook_back_m), 0.0]), R_now, 1.0,
                                              ap.transit_cap, ap.transit_tol_m, 180.0, max_steps=8)
                    looked = None
                    if self.spec.get("place_instance") is not None and hasattr(self.oracle, "relook_place"):
                        try:
                            looked = self.oracle.relook_place(self.spec["place_instance"])
                        except Exception:
                            looked = None
                    rec_rl["looked"] = looked is not None
                    ux = self._unexplained()
                    p_now = np.array(self._pose()[0], dtype=float)
                    # new to the carry: not counted before the look, and not the hand's own top (colour rows: a
                    # region 2 to 5 cm from the grip site that tops out 2 to 4 cm above it, in six of twenty looks)
                    fresh = [c for c, h in zip(ux.C, ux.H)
                             if not any(float(np.linalg.norm((c - c0)[:2])) <= 0.05 for c0 in before.C)
                             and not (float(np.linalg.norm((c - p_now)[:2])) <= 0.06
                                      and float(c[2] + h[2]) < float(p_now[2]) + 0.06)]
                    try:  # (the record: what the second look shows standing tall, raw and as the carry counts it)
                        ex = getattr(self.oracle, "extra_last", None) or {}
                        rec_rl["regions"] = [[round(float(x), 3) for x in (b.get("xy") or [0, 0])[:2]]
                                             + [round(float(b.get("z_top") or 0), 3), round(float(b.get("height") or 0), 3)]
                                             + [round(float(x), 3) for x in ((b.get("top") or {}).get("xy") or [])[:2]]
                                             + [round(float(max((b.get("top") or {}).get("half_xy") or [0])), 3),
                                                int((b.get("top") or {}).get("n_px") or 0), round(float(b.get("h_min") or 0), 3)]
                                             for b in ex.get("blobs") or [] if float(b.get("height") or 0) >= 0.08][:6]
                        rec_rl["counted"] = [[round(float(c[0]), 3), round(float(c[1]), 3), round(float(c[2] + h[2]), 3)]
                                             for c, h in zip(ux.C, ux.H)]
                        rec_rl["new"] = len(fresh)
                    except Exception:
                        pass
                    usable = False
                    if fresh:
                        pcs = self.oracle.percepts  # (the percepts as they stand: no further look is asked for)
                        poses_l = self.lib.poses_from_percepts(pcs() if callable(pcs) else pcs)
                        # (a thing the plan cannot use, too tall to cross within reach and with no way around, is
                        # no reason to try the same straight way again)
                        self._choose_unexplained(poses_l, p_now, R_now, p_above, R_now, 0.5 * gap2, (Rr, tr),
                                                 (self.target,), ap.transit_margin_m)
                        usable = getattr(self, "_ux_plan", None) in ("over", "around")  # (the way changes)
                        rec_rl["plan"] = getattr(self, "_ux_plan", None)
                    if usable:
                        st = yield from self._watch_carry(self._watch_block(
                            self._move_via(poses_l, p_above, R_now, 1.0, (self.target,), 0.5 * gap2, (Rr, tr),
                                           ap.transit_margin_m, ap.transit_tol_m, ap.transit_tol_deg,
                                           **({"stall_last": True} if feel else {})), p_above), gap2)
                        rec_rl["status"] = st
                    # (a second plan from here crosses lower than the first one did: colour line, one dropped slab,
                    # the detour after it rose to 1.31 m where the reference's rose to 1.37 and cleared the slab)
                    z_now = getattr(self, "_z_cross", None)
                    if z_keep is not None:
                        self._z_cross = z_keep if z_now is None else max(float(z_keep), float(z_now))
            if feel:
                att["gap_above"] = round(self._gap(), 5)  # record: does a bowl held by its rim show a touch here?
            if st == "lost":  # the fingers met on the way: the object is no longer in the hand
                att.update(outcome="lost_in_carry", t_lost=self.t)
                found = yield from self._after_loss(R_now)
                if found is not None:
                    att["lost_look"] = found
                continue
            if st != "reached" and ap.detour_rise_m > 0:
                # blocked on the way by something the looks do not show (a barrier): the hand stopped far from
                # the goal. Rise straight up, cross high, and come down above the goal (once per transport).
                p_b, _ = self._pose()
                d_left = float(np.linalg.norm((p_above - p_b)[:2]))
                if d_left > ap.detour_min_m:
                    z_hi = max(float(p_b[2]), float(p_above[2])) + ap.detour_rise_m
                    if ap.detour_from_cross and getattr(self, "_z_cross", None) is not None:
                        z_hi = max(z_hi, float(self._z_cross) + ap.detour_rise_m)
                    way_d, mark_d = None, {}
                    if ap.detour_seen and ap.transit_unexplained and int(getattr(self, "lag_steps", 0) or 0) <= 0:
                        z_hi, way_d, mark_d = self._detour_seen(att, poses_l, np.array(p_b, dtype=float), R_now,
                                                               p_above, 0.5 * gap2, (Rr, tr), (self.target,),
                                                               ap.transit_margin_m, z_hi)
                    self._ev("detour", z=round(z_hi, 4), left_m=round(d_left, 4), **mark_d)
                    if way_d is not None:  # (``detour_seen``: around the thing's nearer end)
                        st_w = "reached"
                        for k_w, (pt_w, R_w, tol_w, deg_w) in enumerate(way_d):
                            last_w = k_w == len(way_d) - 1
                            st_w = yield from self._goto(pt_w, R_w, 1.0, ap.transit_cap,
                                                         ap.transit_tol_m if last_w else tol_w,
                                                         ap.transit_tol_deg if last_w else deg_w, straight=True)
                            if st_w != "reached":
                                break
                        st = "detour_" + st_w
                    else:
                        st_u = yield from self._goto(np.array([p_b[0], p_b[1], z_hi]), R_now, 1.0, ap.transit_cap,
                                                     ap.transit_tol_m, 180.0, max_steps=30)
                        if st_u == "reached":
                            st_c = yield from self._goto(np.array([p_above[0], p_above[1], z_hi]), R_now, 1.0,
                                                         ap.transit_cap, ap.transit_tol_m, ap.transit_tol_deg)
                            if st_c == "reached":
                                st = yield from self._goto(p_above, R_now, 1.0, ap.transit_cap, ap.transit_tol_m,
                                                           ap.transit_tol_deg, max_steps=40)
                                st = "detour_" + st
            relook_seen = False
            if ap.relook_place and not ap.trust_place and self.spec.get("place_instance") is not None \
                    and hasattr(self.oracle, "relook_place"):
                # the receiver looked at again with the hand above the release point (2026-11-15). Seen moved: the
                # release is planned again and the hand goes there. (With the hand down at the release point the
                # look saw less: a receiver that had jumped 15 cm, in view from here, was not found from there.)
                rl = self.oracle.relook_place(self.spec["place_instance"])
                if rl is not None:
                    att["relook_place"] = rl
                    if rl["moved_m"] > ap.relook_min_m:
                        relook_seen = True
                        poses_n = self.lib.poses_from_percepts(self.oracle.percepts)
                        place_n = self._plan_place(poses_n, Rr, tr, gap2)
                        if place_n is not None:
                            place = place_n
                            p_rel = np.array([place["hand_xy"][0], place["hand_xy"][1], place["z_release"]])
                            p_above = p_rel + np.array([0, 0, ap.lower_slow_m])
                            att["place"].update(z_release=round(place["z_release"], 4), relooked=True)
                            rl["regoto"] = yield from self._move_via(poses_n, p_above, R_now, 1.0, (self.target,),
                                                                     0.5 * gap2, (Rr, tr), ap.transit_margin_m,
                                                                     ap.transit_tol_m, ap.transit_tol_deg)
            # item 3 (off by default): the close look before letting go. The receiving object as the wrist camera
            # sees it from above; the release point moves across with it (its height stays as planned)
            if ap.close_look_place:
                mv, cl = self._close_look(self.spec["place_instance"])
                if cl is not None:
                    att["close_place"] = cl
                if mv is not None:
                    mv = np.array([mv[0], mv[1], 0.0])
                    p_rel, p_above = p_rel + mv, p_above + mv
                    att["close_place"]["regoto"] = yield from self._goto(p_above, R_now, 1.0, ap.transit_cap,
                                                                         ap.transit_tol_m, ap.transit_tol_deg,
                                                                         max_steps=30)
            opened: dict = {}  # ``cover_touch_descent``: the hand's heights around the step at which the gap opened
            self._n_open_early = 0
            if ap.trust_place:  # a staging release needs no millimetre: down at full speed, to within 1 cm
                gen_d = self._goto(p_rel, R_now, 1.0, ap.transit_cap, 0.01, 6.0, max_steps=25)
                if ap.stage_release_early > 0 and int(getattr(self, "lag_steps", 0) or 0) <= 0:
                    gen_d = self._opening_early(gen_d, p_rel, int(ap.stage_release_early))
                st2 = yield from self._watch_carry(gen_d, gap2)
            elif ap.clear_cover and self.n_cleared == 0 and not relook_seen and st in ("unreachable", "stalled") \
                    and float(np.linalg.norm((self._pose()[0] - p_above)[:2])) <= ap.cover_xy_m \
                    and float(self._pose()[0][2] - p_above[2]) > 0.01:
                # already stopped from below on the way down to the point above the release: pressing down a second
                # time would only show the same (about 10 steps)
                st2 = "blocked"
            else:
                # (``lower_cap``, ``lower_tol_m``: onto a surface or an object only; down into a container, between
                # its walls, the move stays as it was)
                onto = self.spec.get("predicate") != "in"
                down = self._goto(p_rel, R_now, 1.0, ap.lower_cap if (ap.lower_cap > 0 and onto) else ap.approach_cap,
                                  max(ap.reach_tol_m, ap.lower_tol_m) if onto else ap.reach_tol_m, ap.reach_tol_deg,
                                  max_steps=40,
                                  **({"stall_down": True, "stall_after_m": 0.004} if ap.clear_cover else {}))
                if ap.clear_cover and ap.cover_pick and self.n_cleared == 0 and fr["width"] <= ap.rim_width_m \
                        and int(getattr(self, "lag_steps", 0) or 0) <= 0:  # (not through a delay: late readings)
                    # ``cover_pick``: the descent ends at the step a rim-held object touches (and where it touched
                    # is read from that step: pressing on, the stall rule and the rise that reads the height
                    # otherwise take up to 30 steps, and the steps are what taking the thing off needs)
                    down = self._watch_touch(down, gap2, opened)
                elif ap.clear_cover and ap.cover_touch_descent and fr["width"] <= ap.rim_width_m:
                    down = self._watch_open(down, gap2, opened)  # where a rim-held object touched on the way down
                st2 = yield from self._watch_carry(down, gap2)
            if ap.clear_cover:
                att["gap_down"] = round(self._gap(), 5)  # record, as ``gap_above``
            if st2 == "lost":
                att.update(outcome="lost_in_carry", t_lost=self.t)
                found = yield from self._after_loss(R_now)
                if found is not None:
                    att["lost_look"] = found
                continue
            touched = None
            if ap.clear_cover and not ap.trust_place and st2 == "reached" and self.n_cleared == 0 \
                    and place.get("region") is not None and self.spec.get("predicate") == "on" \
                    and place["z_release"] <= place["z_lo"] + 0.003:
                # a release on a surface, planned 2 cm above it: on down slowly until the object touches
                clear = ap.place_clear_region_m
                st_t = yield from self._watch_carry(self._goto(p_rel - np.array([0.0, 0.0, clear + 0.004]), R_now, 1.0,
                                                               ap.slow_cap, 0.002, ap.reach_tol_deg, max_steps=25,
                                                               stall_down=True), gap2)
                if st_t == "lost":
                    att.update(outcome="lost_in_carry", t_lost=self.t)
                    yield from self._hold(self._pose()[0], R_now, -1.0, ap.open_steps)
                    continue
                touched = float(self._pose()[0][2] - (p_rel[2] - clear))
                att["touch_m"] = round(touched, 4)
            if ap.clear_cover and not ap.trust_place and ap.cover_gap_m > 0 and touched is None \
                    and self.n_cleared == 0 and fr["width"] <= ap.rim_width_m \
                    and self._gap() - gap2 >= ap.cover_gap_m:
                # held by its rim, and the fingers' gap has opened on the way down: the object rests on something.
                g_rest = self._gap()
                p_dn, _ = self._pose()
                if opened.get("z_prev") is not None and opened["z_prev"] > float(p_dn[2]) + 0.002:
                    # ``cover_touch_descent``: it touched between the last step with the gap as it was and the first
                    # with it open; the hand goes back up to the middle of the two
                    z_t = 0.5 * (float(opened["z_open"]) + float(opened["z_prev"]))
                    z_t = float(np.clip(z_t, p_dn[2], p_dn[2] + 0.10))
                    yield from self._goto(np.array([p_dn[0], p_dn[1], z_t]), R_now, 1.0, ap.transit_cap, 0.004,
                                          180.0, max_steps=12)
                    touched = float(self._pose()[0][2] - p_rel[2]) + ap.place_clear_object_m
                    if ap.cover_pick and ap.cover_gap_lag_m > 0:
                        # ``cover_pick``: a rim-held object turns in the fingers before their gap shows it; the
                        # touch is that much higher than where the gap opened (see ``cover_gap_lag_m``)
                        touched += ap.cover_gap_lag_m
                    att["gap_touch"] = {"gap_rest": round(g_rest, 5), "gap_free": round(self._gap(), 5),
                                        "rose_m": round(float(self._pose()[0][2] - p_dn[2]), 4),
                                        "touched_m": round(touched, 4), "by": "descent",
                                        "step_m": round(float(opened["z_prev"]) - float(opened["z_open"]), 4)}
                else:
                    # Up slowly until it hangs free again: that height is where it touched
                    for _ in range(20):
                        if self._gap() - gap2 < 0.5 * ap.cover_gap_m or self._pose()[0][2] - p_dn[2] > 0.06:
                            break
                        p_n, R_n = self._pose()
                        yield follow(p_n, R_n, p_n + np.array([0.0, 0.0, 0.01]), R_now, 1.0, ap.transit_cap,
                                     ap.rot_cap, self.fp)
                    touched = float(self._pose()[0][2] - p_rel[2]) + ap.place_clear_object_m
                    att["gap_touch"] = {"gap_rest": round(g_rest, 5), "gap_free": round(self._gap(), 5),
                                        "rose_m": round(float(self._pose()[0][2] - p_dn[2]), 4),
                                        "touched_m": round(touched, 4)}
            if ap.clear_cover and not ap.trust_place and (st2 != "reached" or touched is not None) \
                    and self.n_cleared == 0 and self._c1(self._gap(), fr["width"]):
                # the descent was stopped from below, straight above its goal: something lies on the receiver
                # (planned on the scene as the last look left it: the receiver may have been looked at again)
                poses_c = self.lib.poses_from_percepts(self.oracle.percepts) if hasattr(self.oracle, "percepts") \
                    else poses_l
                cplan, crec = self._cover_plan(poses_c, p_rel, R_now, Rr, tr, gap2, touched,
                                               **({"rim": True} if ((ap.cover_skim_rim_m > 0 or ap.cover_slow_cap > 0
                                                                     or ap.cover_under_rim_m > 0)
                                                                    and fr["width"] <= ap.rim_width_m) else {}))
                if crec is not None:
                    if att.get("cover") is not None:  # (``cover_see``: seen on the way and not taken; kept)
                        att["cover_first"] = att["cover"]
                    att["cover"] = crec
                R_c, looked = R_now, False
                picked = None
                if ap.cover_pick and crec is not None and self.spec.get("place_instance") is not None \
                        and getattr(self, "n_picked", 0) == 0 and hasattr(self.oracle, "on_receiver") \
                        and int(getattr(self, "lag_steps", 0) or 0) <= 0:
                    # ``cover_pick``: a thing the hand can hold is taken off the receiver; the object waits beside it
                    picked = yield from self._cover_pick(poses_c, p_rel, R_now, Rr, tr, gap2, crec, width=fr["width"])
                    if picked == "lost":
                        att.update(outcome="lost_in_carry", t_lost=self.t)
                        yield from self._hold(self._pose()[0], R_now, -1.0, ap.open_steps)
                        continue
                    if picked in ("parked", "retaken"):
                        att["outcome"] = picked
                        self._retake_need = ap.cover_retake_need
                        continue
                    looked = picked == "back"  # (the hand went aside and came back: down again below, push or not)
                if ap.cover_look and picked is None and crec is not None and crec.get("skipped") != "no steps" \
                        and fr["width"] <= ap.rim_width_m and self.spec.get("place_instance") is not None \
                        and hasattr(self.oracle, "on_receiver"):
                    # ``cover_look``: what lies there, and which way it is long; the push runs across that
                    p_touch = np.asarray(self._pose()[0], dtype=float).copy()
                    looked = True
                    R_t = yield from self._cover_look(poses_c, p_touch, R_now, Rr, tr, gap2, crec)
                    if isinstance(R_t, str):  # "lost"
                        att.update(outcome="lost_in_carry", t_lost=self.t)
                        yield from self._hold(self._pose()[0], R_now, -1.0, ap.open_steps)
                        continue
                    on_xy = ((crec.get("look") or {}).get("on") or {}).get("xy")
                    if R_t is not None or on_xy is not None:
                        # planned again for what was seen: the hand as it will push, the push through the thing's
                        # middle
                        R_u = R_t if R_t is not None else R_now
                        cplan_t, crec_t = self._cover_plan(poses_c, p_rel, R_u, Rr, tr, gap2, touched, rim=True,
                                                           at=p_touch, centre=on_xy)
                        if cplan_t is not None:
                            cplan, R_c = cplan_t, R_u
                            crec.update({k: v for k, v in crec_t.items() if k not in ("t", "dz", "exy", "touched_m")})
                            crec.pop("skipped", None)
                        else:
                            crec["look"]["plan"] = (crec_t or {}).get("skipped")
                            if R_t is not None:
                                # no push that way: the hand turns back, and the push is the one planned before
                                yield from self._watch_carry(self._goto(self._pose()[0], R_now, 1.0, ap.transit_cap,
                                                                        0.02, 6.0, max_steps=25), gap2)
                    if cplan is None:
                        # no push at all: back over the release point (the hand went aside for the look), then down
                        # again as after a push
                        st_b = yield from self._watch_carry(self._goto(
                            np.array([p_rel[0], p_rel[1], float(p_touch[2]) + ap.cover_rise_m]), R_now, 1.0,
                            ap.transit_cap, ap.transit_tol_m, 180.0, max_steps=30), gap2)
                        if st_b == "lost":
                            att.update(outcome="lost_in_carry", t_lost=self.t)
                            yield from self._hold(self._pose()[0], R_now, -1.0, ap.open_steps)
                            continue
                if cplan is not None:
                    how = yield from self._clear_cover(cplan, crec, R_c, gap2)
                    if R_c is not R_now and how != "lost":
                        # the hand as it was when it took the object: the release is planned for that
                        st_b = yield from self._watch_carry(self._goto(cplan["back"], R_now, 1.0, ap.transit_cap, 0.02,
                                                                       6.0, max_steps=25), gap2)
                        crec["look"]["back"] = st_b
                        if st_b == "lost":
                            how = "lost"
                    crec["status"] = how
                    if how == "lost":
                        att.update(outcome="lost_in_carry", t_lost=self.t)
                        yield from self._hold(self._pose()[0], R_now, -1.0, ap.open_steps)
                        continue
                    if ap.cover_relook and how == "swept" and self.spec.get("place_instance") is not None \
                            and hasattr(self.oracle, "relook_place"):
                        # ``cover_relook``: the push may have moved the receiver itself. Where is it now?
                        rl2 = self.oracle.relook_place(self.spec["place_instance"])
                        if rl2 is not None:
                            crec["relook"] = {k: rl2[k] for k in ("moved_m", "assoc", "still", "unconfirmed_m")
                                              if k in rl2}
                            if rl2["moved_m"] > ap.relook_min_m:
                                poses_n = self.lib.poses_from_percepts(self.oracle.percepts)
                                place_n = self._plan_place(poses_n, Rr, tr, gap2)
                                if place_n is not None:
                                    place = place_n
                                    p_rel = np.array([place["hand_xy"][0], place["hand_xy"][1], place["z_release"]])
                                    att["place"].update(z_release=round(place["z_release"], 4), relooked=True)
                                    st_m = yield from self._watch_carry(
                                        self._move_via(poses_n, p_rel + np.array([0, 0, ap.lower_slow_m]), R_now, 1.0,
                                                       (self.target,), 0.5 * gap2, (Rr, tr), ap.transit_margin_m,
                                                       ap.transit_tol_m, ap.transit_tol_deg), gap2)
                                    crec["relook"]["regoto"] = st_m
                                    if st_m == "lost":
                                        att.update(outcome="lost_in_carry", t_lost=self.t)
                                        yield from self._hold(self._pose()[0], R_now, -1.0, ap.open_steps)
                                        continue
                if cplan is not None or looked:
                    # down again where the release was planned (the receiver has not moved; the thing has)
                    st2 = yield from self._watch_carry(self._goto(p_rel, R_now, 1.0, ap.approach_cap, ap.reach_tol_m,
                                                                  ap.reach_tol_deg, max_steps=40, stall_down=True,
                                                                  stall_after_m=0.004), gap2)
                    if st2 == "lost":
                        att.update(outcome="lost_in_carry", t_lost=self.t)
                        yield from self._hold(self._pose()[0], R_now, -1.0, ap.open_steps)
                        continue
                    crec["after_dz"] = round(float(self._pose()[0][2] - p_rel[2]), 4)
                    if ap.cover_relook:  # the record: a gap that opened again means the object rests on something
                        crec["after_gap"] = round(float(self._gap() - gap2), 5)
            if ap.lag_skip_release_fix and int(getattr(self, "lag_steps", 0) or 0) > 0:
                # through a delay, no wait-and-correct before letting go: the hand is within some 5 mm of the release
                # point (what the wait found left in one run: 2.5 to 5.3 mm), and the wait is a whole lag of steps
                fix = None
                att["release_fix"] = {"skipped": True}
            else:
                fix = yield from self._settle_fix(p_rel, R_now, 1.0, ap.slow_cap, 0.006, ap.reach_tol_deg)
            if fix is not None:
                att["release_fix"] = fix
            att["transport_status"] = [st, st2]
            # 7. release, C3
            att["t_release"] = self.t
            p_hold, _ = self._pose()
            if ap.trust_place:  # a step of a longer plan: open, clear the object, done (no settle wait, no check)
                # where the hand let go, for a later step that takes the object again without looking
                self.released = {"p": [float(x) for x in p_hold], "R": np.asarray(R_now, dtype=float).tolist(),
                                 "frame_z": float(fr["centre"][2]), "t": self.t}
                try:
                    # (for ``take_by_open`` and ``take_near_rim``, read by the planner of the later step; not used
                    # here) where the hand's point lay from the object's own origin when it took it, as the look
                    # before the grasp and the grasp's own geometry gave it: the hand has not turned since, so the
                    # object's origin is that far from where the hand lets go. And whether the object is the same
                    # turned half a turn about its upright axis (the same grasp then exists on its other side)
                    t_obj0 = pose_of(poses, self.troot)[1]
                    self.released["hold_xy"] = [float(x) for x in (np.asarray(fr["centre"], float) - t_obj0)[:2]]
                    self.released["half_turn"] = self._half_turn_same()
                except Exception:
                    pass
                n_early = int(getattr(self, "_n_open_early", 0) or 0)  # ``stage_release_early``: sent on the way down
                if n_early:
                    att["released_early"] = n_early
                yield from self._hold(p_hold, R_now, -1.0, max(0, ap.open_steps - n_early))
                att["c3"] = {"t": self.t, "trusted": True, "pass": True}
                att["t_retreat"] = self.t
                yield from self._goto(p_hold + np.array([0, 0, 0.07]), R_now, -1.0, ap.transit_cap, 0.02, 180.0,
                                      max_steps=12)
                att["t_end"] = self.t
                att["outcome"] = "placed"
                self._finish("placed")
                return
            drop = self._expect_drop(place, p_hold, att)
            # ``c3_measured``: the looks taken from here on are the ones that can measure the object where it is put
            n_looks0 = len(self.oracle.looks) if (ap.c3_measured and hasattr(self.oracle, "target_measured")) else None
            yield from self._hold(p_hold, R_now, -1.0, ap.release_steps)
            reads = []
            n_settle = ap.settle_steps + (ap.drop_settle_steps if drop else 0)
            for i in range(n_settle):
                yield self._idle()
                if i >= n_settle - 3:
                    reads.append(self.oracle())
            c3, c3d = self._c3(reads, reads[-1])
            att["c3"] = {"t": self.t, **c3d, "pass": c3}
            # 8. retreat
            att["t_retreat"] = self.t
            p_now, R_now2 = self._pose()
            yield from self._goto(p_now + np.array([0, 0, ap.retreat_m]), R_now2, -1.0, ap.transit_cap,
                                  ap.transit_tol_m, 180.0, max_steps=30)
            att["t_end"] = self.t
            if ap.c3_recheck and not c3 and c3d.get("inside") and not c3d.get("rest"):
                c3, reads = yield from self._c3_again(att, c3d)
            if c3 and n_looks0 is not None and not getattr(self, "more_follow", False) \
                    and not self.oracle.target_measured(n_looks0):
                # "placed" on the hand's own prediction: no look has measured the object since the hand let go.
                # (Not when another step of a longer plan follows: the benchmark ends an episode only when all of
                # its task is done, so an episode that goes on says nothing against this step, and an object that
                # stands right, unmeasured, must not be taken up again.)
                c3, reads = yield from self._c3_measure(att)
            pc = getattr(self.oracle, "place_check", None)  # ST-1 H8: a measurement replaces the in-hand prediction
            region = self._place_goal(reads[-1])[2] if (pc and self.spec.get("predicate") != "on") else None
            if region is not None:
                pl = self.oracle.place_look(region)
                inner = np.maximum(region["half"][:2] - ap.c3_shrink_m, 0.0)
                c3 = bool(pl["n_px"] >= pc["min_px"] and np.all(np.abs(np.asarray(pl["xy"]) - region["centre"][:2]) <= inner))
                att["c3"].update({"stale_pass": att["c3"]["pass"], "pass": c3, "measured": pl})
            if ap.relook_place and not relook_seen and self.spec.get("place_instance") is not None \
                    and hasattr(self.oracle, "relook_place"):
                # and once more with the hand backed off: a receiver that moved during the last centimetres of the
                # descent (hidden under the hand until then) is in view now. Moved: the object was set down where
                # the receiver had been, and this attempt has failed, whatever its own check read.
                rl2 = self.oracle.relook_place(self.spec["place_instance"])
                if rl2 is not None:
                    att["relook_after"] = rl2
                    if rl2["moved_m"] > ap.relook_min_m:
                        att["c3"]["receiver_moved"] = True
                        c3 = False
            if c3:
                att["outcome"] = "placed"
                self._finish("placed")
                return
            att["outcome"] = "c3_fail"
