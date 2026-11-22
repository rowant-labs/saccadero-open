"""Acting through a delay (2026-11-15; off unless ``params["delay_comp"]``).

A robot's commands can take effect late and its readings can arrive late (a real arm's actuation and sensing lags; the
benchmark's S20 puts 7 steps on each from its 8th step). The follower then steers by a hand that is no longer where
the reading says, and its loop swings from side to side until the steps run out. ``DelayComp`` stands between the
robot and the world. It reads only what the robot sent and what it was told:

* **Late readings are seen at once**: a changing reading equal, to the last bit, to the reading of d steps before is
  d steps late. A reading that leaps further than any command in flight could carry it says a delay has ended.
* **Then the lag is measured, not guessed** (``busy``/``own_action``): the robot is paused, the hand is left to come
  to rest, one small upward pulse is sent and the step at which the readings answer is the loop's lag (commands' and
  readings' delays together), exactly; the pulse is taken back and the robot goes on. About twice the lag in steps.
  (2026-11-15: scaling the robot's commands down until a fit settled confused the robot's own stall checks and in
  places never ended.)
* A delay of the commands alone shows no repeat; the lag is then fitted from the robot's own motion (the reported
  step against the command of each lag, over the last steps), as before.
* **The hand where the next command will find it** (``predict``): the reading plus the effect of every command still
  on its way, by the fitted gain (the share of a commanded move the arm makes in a step). The robot steers by that.
  The pose the cameras show is still the reading's, so looks keep the readings as they came, and the robot waits out
  the lag before a check that compares a look with its hand (``OracleAgent._settle``).
* **The gripper** cannot be predicted (contact). After the gripper's command changes sign the hand is held still for
  the lag (``holding``), so the gap the robot then reads is of fingers that have begun to move.

2026-11-16 (each off unless its key is set):

* ``start_lag``: **the lag read again, for nothing, from the first move after a rest.** When the hand is at rest and
  nothing is on its way (the robot's waits are such rests), the step at which the readings answer the robot's next
  command is the loop's lag at that moment. An answer at once ends a lag in use: a delay that ended while the hand
  was still shows no leap (D12 in the suite with the shortest episodes: the lag stayed in use to the end in six of
  eight). A late answer where no lag was in use starts the pulse measurement.
* ``cmd_watch``: **commands that take effect late, or not at all** (D13: every command about 8 steps late for some
  40 steps; D14: one command held for 10 to 24 steps, during which nothing sent is carried out). The readings are
  current, so nothing repeats. What shows is a hand that moves against its commands for several steps in a row.
  Then the hand is pulled back toward where it began to go against its commands (by position alone) until the
  readings answer, and after that nothing is sent until it is read at rest; the robot then looks and plans again
  from where the hand is, and its first move times what lag is left (``start_lag``; without it, the pulse).
  Why a pull and not a wait: a held command carries the arm toward the edge of its reach, and an arm that arrives
  there with its elbow straight cannot be drawn back by any command tried (Mac runs: the hand stood
  at the same far line through 45 steps of commands home, the elbow at its limit throughout). The pull is ignored
  while the hold lasts and takes hold in the step it ends. Sending nothing instead let the arm coast the last
  centimetre into that trap in an episode the robot, left alone, had come back from (Spatial task 4, D14).
* ``unlock``: **an arm caught with its elbow straight.** A held command carries the hand out until the elbow is
  at its limit, and from there no command of position draws it back: the hand slides up and sideways along the
  edge of its reach, and the robot's own pull, with the hand held upright, stands still (last night: the hand at
  the same far line in 10 of 209 colour episodes, 8 of them D14, its targets 6 to 24 cm nearer the robot, none
  solved). What frees it, measured on the Mac from that state (``probes/delay_probe.py``, six ways tried): the
  pull toward the robot together with a turn of the hand that swings the fingers toward the robot (0.8 and 0.6
  of a full command). The elbow bent from its limit to 0.43 rad in 12 steps and the hand came 12 cm in at the
  same height. A rise of 20 cm bent it too, more slowly (0.24 rad after 24 steps) and it needs the room overhead.
  The pull alone, down, down and in, or the pull with the turn the other way left the elbow at its limit. So: an
  open hand whose elbow reads straight (``elbow_straight``) is unlocked that way until the elbow reads
  ``elbow_free``, then the robot looks and plans again. It is tried after a watch, and whenever the robot asks
  such a hand to come in for ``caught_n`` steps and it does not (``_caught``); at most ``caught_max`` times an
  episode.
* ``flush_rest``: the pause before a pulse lasts until the hand is read at rest (not ``flush_n`` steps only).
* ``no_answer_assume``: when the pulse after late readings gets no answer, the commands are taken to be as late as
  the readings (the lag is twice the readings' delay and one), not on time, and the wait for the answer ends once
  that lag has had its turn. Every pulse without an answer last night was S20's (6 of 48 episodes; D12's were all
  answered), where that is the truth: its first commands, played twice, had carried the hand low and near the
  robot until the wrist folded right back, and from there the arm drifts and does not answer a short pulse (Mac
  trace: it rose 2.7 mm for the pulse where 11 to 14 are usual). The robot went on with the readings'
  delay alone as its lag, half the truth, and lost all six. A lag assumed wrongly is put right at the next move
  after a rest.
* ``leap_trust``: a leap of the readings ends the lag in use at once; no pulse is sent to confirm it.
* ``replan_late_only``: the robot looks and plans again after a measurement only when the COMMANDS were late (the
  lag is two or more steps longer than the readings' own delay; S20). When only the readings are late (D12) the hand
  went nowhere it was not sent, the first plan still holds, and a new look with the hand standing over the target
  is a worse one (one run, Goal task 1: the first grasp of the second plan was 3.6 cm off and closed on
  nothing four times).
* ``open_hold_cmd``: after a command to OPEN the hand is held only for the commands' own delay (the lag less the
  readings' delay), not the whole lag: the fingers have then begun to open, which is all the release waits for, and
  the robot's own release steps follow. Closing keeps the whole lag: the robot reads the gap while it closes.

2026-11-17 (stream LAG; each off unless its key is set). Where the lag cases' extra steps go was counted from the
stored rows: most of them are waits of a whole lag, and two of those waits are for a
reading the robot does not need before it goes on.

* ``close_hold_cmd``: after a command to CLOSE the hand is held only for the commands' own delay, as
  ``open_hold_cmd`` does for opening. The robot must then not read its fingers until their reading is in: the
  runner tells the agents (``lag_close_blind``), and the pick-and-place agent closes for as long as the fingers take,
  lifts, and reads its grasp check when the reading of the shut fingers has arrived (the readings' delay later, by
  which time the lift is done). Before: the hold of a whole lag, then the closing watched in late readings (26 to 30
  steps for a bowl's rim under S20 where 9 or 10 are plain).
* ``probe_grip``: **late readings, and are the commands late too? Asked of the fingers, on the move.** When the
  readings are found late while the hand is open and has not yet closed on anything, the question the pulse answers
  (3 steps of pause, 6 of pulse, and the lag itself of waiting: 21 steps under D12, 25 under S20, the hand at rest)
  is put to the fingers instead: two steps of "close" in place of the robot's "open", which move nothing but the
  fingers, a few millimetres in and out again. The step at which the fingers' reading dips is the loop's lag,
  exactly, as the pulse's answer is. Meanwhile the robot goes on, steering by the lag that late readings alone
  would give, at ``probe_scale`` of its commands. Readings alone late (D12): the dip comes when that lag says, the
  lag stands, and nothing was waited for. Commands late too (S20): no dip by then, so the robot is stopped until it
  comes (some six steps), the lag is taken from it, and the robot looks and plans again as after a pulse. A robot
  that closes its hand meanwhile, fingers that are not open and still, or no dip at all: the pulse, as before.
* ``grip_echo``: **the fingers' answer to the robot's own gripper commands is timed too**, with nothing sent for
  it. Each time the robot's command to the fingers changes sign, with the fingers read still before it, the step
  at which their reading follows is the loop's lag at that moment. An answer at once ends a lag in use: late
  readings that end while the hand is at rest or moving slowly show no leap, and the lag then stays in use (the
  Mac, three D12 episodes run with the shortened closing hold: the delay ended before the close, the lag stood for
  60 to 110 steps more, the grasp and the placing went by a hand predicted 11 commands ahead, and all three lost
  their first attempt; with the whole-lag hold the first move after it was timed by ``start_lag``, which needs
  as many quiet commands as the lag before the move). An answer that comes two or more steps SOONER than the lag
  in use allows also ends it (the readings caught up between the command and its answer). It only ever ends a
  lag: a later answer to a closing is noted (``rec["echo"]``) and nothing more, and fingers letting go of
  something answer late by themselves.
* ``grip_as_sent``: the steps this object sends itself (the pause, the pulse, the wait for its answer) repeat the
  robot's last gripper command as it was sent, "keep as it is" (0) included. Before, they repeated its last command
  to open or to close: a measurement that fell in the drawer step's approach, where the jaws are shaped and then
  kept, shut them for its 21 steps, and the opening then had to open them again (a hold and 8 steps and the lag).
"""

from __future__ import annotations

import numpy as np

from .follow import OUT_POS, OUT_ROT, follow
from .grasp import quat_xyzw_to_mat

DELAY = {
    "max_lag": 24,       # steps
    "window": 16,        # steps of history a fit reads (short: a delay that ends must leave it soon)
    "min_n": 8,          # steps a fit needs
    "min_drive": 0.2,    # the commands' summed squares in the fit's steps below which nothing is decided (nine gentle steps)
    "better": 0.5,       # a lag is a candidate when its unexplained share is at most this share of the lag in use's
    "explains": 0.5,     # and at most this share of the motion's own energy
    "near": 0.05,        # among lags whose unexplained share is within this of the best, the shortest is taken
    "confirm": 3,        # steps in a row the same candidate must win
    "gain": 0.23,        # the share of a commanded move made in a step (fitted on undelayed runs, 2026-11-15)
    "gain_range": (0.08, 0.6),
    "min_lag": 2,        # a lag below this is the arm's own response time: not a delay
    "repeat_confirm": 2,  # steps in a row a reading must repeat the reading of the same number of steps before
    # the measurement after a sign of delay: flush_n steps of nothing, then pulse_n steps of +z and pulse_n of -z at
    # pulse_amp (up about 2 cm and back). The answer is looked for in the readings' z steps as that same up-and-down
    # shape, at every lag: commands still in flight from before move the hand one way and cancel in it. Taken when
    # it is at least answer_share of what the default gain would give
    "flush_n": 3, "pulse_n": 3, "pulse_amp": 0.5, "answer_share": 0.5,
    "leap": 2.5,         # a reading step this many times the largest step any command in flight could make: a delay ended
    "leap_min_m": 0.035,  # and at least this long (one run: D12's readings caught up by 4.2-7.6 cm; under S20 the arm,
    # fed fifteen full commands at once, made steps of 2.0-3.1 cm that were taken for it, and plain steps of 1.2 cm
    # were taken for it wherever a wrongly fitted lag was in use)
    # the lag fitted from the robot's own motion (a delay of the commands alone, which shows no repeat). Off: in
    # that run it found lags of 2-5 steps in undelayed episodes and lost one of them, and gained nothing on the
    # cases it was meant for (D13 14 -> 13, D14 9 -> 9 of 24)
    "fit": False,
    # --- 2026-11-16, each off unless set (see the module's text)
    "flush_rest": False,   # the pause before a pulse lasts until the hand is read at rest
    "rest_m": 0.0006,      # at rest: the readings' steps under this (m) ...
    "rest_n": 2,           # ... for this many steps in a row
    "flush_max": 30,       # and never longer than this many steps
    "start_lag": False,    # the lag read from the first move after a rest
    "start_quiet": 0.08,   # a command (of 1.0, its xyz) under this asks for no move
    "start_quiet_n": 6,    # so many such commands in a row before the move (and the lag in use + 2)
    "start_cmd": 0.25,     # the move's first command asks for at least this
    "start_move_m": 0.001,  # the answer: the hand this far along that command's direction, and further the step after
    "start_late_min": 4,   # with no lag in use, an answer this many steps late (or more) starts the pulse; one of
    # 2 or 3 steps is a slow start, not a delay (the cases' delays are 6 steps and more)
    "cmd_watch": False,    # a hand that moves against its commands: late or held commands
    "watch_n": 5,          # steps in a row
    "watch_min_m": 0.004,  # a hand step at least this long (a slow hand is the robot's own stall checks' business)
    "watch_cos": 0.3,      # "against": the cosine of the hand's step and the commands due is under this
    "watch_back": 2,       # the commands due: the mean of this many
    "watch_idle": 0.1,     # or the commands due ask for less than this (of 1.0) while the hand moves on
    "watch_cool": 4,       # steps after a measurement in which nothing is watched (its own pulse is still playing out)
    "watch_tries": 2,      # pulses in a row when one gets no answer (it was sent into a hold)
    "watch_lagged": False,  # also while a lag is in use (off: a false alarm then costs a pause of twice the lag)
    # 2026-11-16 evening: the rule made safe (the text above). "loose" is the morning's rule, kept to compare its runs
    "watch_rule": "strict",
    "watch_steady": 0.15,  # strict: the hand's step changes by no more than this share of itself from step to step
    "watch_turn_rad": 0.03,  # strict: and the hand itself turns less than this a step (as read)
    "watch_cmd_turn": 0.5,   # strict: and the commands of the last ``watch_n`` steps ask for less turn than this (of
    # a full command, their mean). The loose rule's false firings all came with 0.7 to 1.0 asked; in 15 of the 16
    # stored D13/D14 episodes where the strict rule fires, under 0.5 was asked in the five steps before
    "watch_trail": 4,      # strict: the command the hand is carrying out is looked for among the last so many due
    "watch_agree": 0.97,   # strict: ... and is one its step points along (cosine)
    "watch_old": 0,        # strict: when none of those explains the step, one of the last so many may (0: off)
    "watch_asked": 0.7,    # strict: the commands due ask for a change of step of this share of the anchored step ...
    "watch_made": 0.2,     # ... and the hand has made no more than this share of that change
    "watch_jump_m": 0.03,  # a reading that moves further than the arm can in a step: a correction of the pose landed
    "watch_jump_cool": 8,  # nothing is watched for so many steps after it
    "watch_dry": False,    # the watch only notes what it would do (the record's ``watch_dry``) and does nothing
    "watch_pull": True,    # during the wait the hand is pulled back toward where it turned against its commands
    "pull_answer_m": 0.002,  # the answer: the readings move back by this much a step, two steps in a row
    "pull_near_m": 0.02,   # or the hand is back within this of that point
    "unlock": False,        # an arm caught with its elbow straight is drawn in with a turn of the hand (the text above)
    "elbow_joint": 3,       # the elbow in ``robot0_joint_pos`` (the Panda's fourth joint: limit -0.07 rad, straight)
    "elbow_straight": -0.12,  # straighter than this: caught
    "elbow_free": -0.4,     # bent again to this: free (the robot's own pull works from -0.34)
    "unlock_in": 0.8,       # the pull toward the robot (of a full command)
    "unlock_turn": 0.6,     # the turn of the hand, fingers toward the robot (of a full command)
    "unlock_max": 20,       # steps at most
    # 2026-11-16 night: after the pull with the turn, the arm is folded and the hand turned back before the robot
    # goes on. From the candidate's caught episodes played again on the Mac (18 unlocks): the pull
    # brings the GRIP POINT 7 to 14 cm in by tilting the hand 20 to 40 degrees more, while the WRIST comes in 1 to 2 cm
    # and the elbow reads -0.42: the arm's reach is its wrist's. The 12 unlocks after which the robot asked outward
    # with a full turn upright were straight again in 6 to 8 steps (the second unlock of every caught episode); the 5
    # after which it asked inward bent on to -0.8 .. -2.1. Scripted (Object task 2, D14): a turn back straight after
    # the pull swings the grip point OUT about the wrist and straightens the elbow, even against a full pull (from
    # -0.43 to -0.07 in 16 steps); after ten steps of a pull alone (elbow -0.95, wrist 5 cm in) the same turn back,
    # with no pull, folds the arm further (-1.54) and the hand then goes out 16 cm upright with the elbow at -1.2;
    # from -0.70 it works with a full pull kept up. So: the whole hand is drawn in as it is (``unlock_deep_*``) until
    # the elbow is well bent, and only then turned back, to the way it stood as last read with the elbow free, before
    # any unlock (when the unlock begins the hand is already tilted 5 to 35 degrees: the grip point crept in by
    # tilting). On the Mac's depth line (three D14 episodes that ended caught in the candidate's run): both unlocks
    # ended folded (elbow -1.87 and -2.15, the hand within 7 degrees), no second one followed, one episode was solved.
    "unlock_back": False,
    "unlock_deep_in": 1.0,  # first the whole hand is drawn in as it is (of a full command, no turn) ...
    "unlock_deep_up": 0.0,  # ... (with this much rise)
    "unlock_deep": -1.0,    # ... until the elbow is bent this far (the wrist some 5 cm in; about 11 steps)
    "unlock_deep_max": 15,  # ... or for so many steps
    "unlock_back_in": 0.0,  # a pull toward the robot kept up during the turn back (of a full command)
    "unlock_back_deg": 10.0,  # turned back: within this of the way the hand stood
    "unlock_back_max": 20,  # steps at most
    # caught by the elbow alone: asked in with the elbow reading straight, whether or not the grip point still creeps
    # in (it creeps by tilting the hand: 9 cm in the 30 steps before the 1 mm rule fired at step 66; the elbow had
    # read straight since step 37)
    "caught_elbow": False,
    "base_xy": (-0.63, 0.0),  # the robot's own place (LIBERO: -0.6 on the floor, -0.66 at the tables), for "toward"
    "caught_in": 0.3,       # asked to come in: this much of a full command toward the robot
    "caught_m": 0.001,      # and coming in less than this a step
    "caught_n": 8,          # for so many steps in a row
    "caught_max": 2,        # times an episode
    "no_answer_assume": False,  # no answer after late readings: the commands are taken to be as late as the readings
    "leap_trust": False,        # a leap ends the lag at once, with no pulse (11 steps saved in D12)
    "replan_late_only": False,  # a new look and plan after a measurement only when the commands were late
    "open_hold_cmd": False,     # after a command to open, hold for the commands' delay only
    # --- 2026-11-17 (stream LAG), each off unless set (see the module's text)
    "close_hold_cmd": False,    # after a command to close, hold for the commands' delay only (the agents close blind)
    "grip_as_sent": False,      # this object's own steps repeat the robot's last gripper command, 0 included
    "probe_grip": False,        # late readings: whether the commands are late too is asked of the fingers, on the move
    "probe_n": 2,               # steps of "close" sent in place of the robot's "open"
    "probe_dip_m": 0.003,       # the answer: the fingers read this much nearer each other than before the probe
    "probe_scale": 0.5,         # the robot's commands are sent at this share until the answer is in
    "probe_open_m": 0.03,       # only with the fingers read at least this far apart ...
    "probe_still_m": 0.002,     # ... and still to within this over the last three readings (the fingers creep open
    # 0.1 mm a step for the first twenty steps of an episode, and late readings begin with a step back: 0.6 mm)
    "probe_slack": 1,           # steps past the lag of late readings alone before the commands are taken to be late
    "grip_echo": False,         # the fingers' answer to the robot's own gripper commands is timed
}


def _exp(r) -> np.ndarray:
    """Rotation matrix of a rotation vector."""
    r = np.asarray(r, dtype=float)
    a = float(np.linalg.norm(r))
    if a < 1e-12:
        return np.eye(3)
    k = r / a
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


def mat_to_quat_xyzw(R) -> list:
    R = np.asarray(R, dtype=float)
    t = float(np.trace(R))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(R[i, i] - R[j, j] - R[k, k] + 1.0, 1e-12)) * 2.0
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i], q[j], q[k] = 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    return [float(x) for x in q]


class DelayComp:
    takes_foreign = True  # ``after_step`` knows the ``foreign`` keyword (the runner asks before it passes it)

    def __init__(self, overrides: dict | None = None):
        self.o = {**DELAY, **(overrides or {})}
        self.acts: list = []   # the commands sent, one a step
        self.pos: list = []    # the hand as read: before the first step, then after each
        self.lag = 0
        self.gain = float(self.o["gain"])
        self._cand, self._n_cand = None, 0
        self._hold, self._grip = 0, None
        self._grip_raw = None   # ``grip_as_sent``: the robot's last gripper command as it was sent (0 included)
        self.gaps: list = []    # ``probe_grip``: the fingers' gap as read (None without the reading), one a reading
        self._pg: dict = {}     # ``probe_grip``: the probe under way {k0, g0, t0, late}
        self._pg_last = None    # the step the last probe began at (its dip is not taken for another command's answer)
        self._ge: dict = {}     # ``grip_echo``: the robot's own gripper command being timed {k, g0, dir}
        self.joints: list = []  # the joint readings, as bytes: a delayed reading repeats an earlier one exactly
        self.obs_lag = 0        # readings found to repeat those of this many steps before
        self.onset = 0          # the step at which they first did: commands sent before it took effect at once
        self._rep, self._n_rep = None, 0
        self.mode = "run"       # "flush", "pulse", "wait": the lag is being measured and the robot is not asked
        self._m: dict = {}
        self._closed = False    # the robot has sent a closing command (it is past its first approach)
        self._replan = False
        self.rec: dict = {"lags": [], "holds": 0}
        self._last = None       # the latest readings
        self._turn_p = None     # ``cmd_watch``: where the hand was when it began to move against its commands
        self._u: dict = {}      # ``unlock``: the manoeuvre under way
        self._st = None         # ``start_lag``: the move being timed {k, u, p0, at}
        self._n_against = 0     # ``cmd_watch``: steps in a row the hand moved against its commands
        self._steady = None     # strict rule: the hand's steady stretch {"n", "ref0"}
        self._jump_cool = 0     # steps left after a jump of the readings
        self.quat: list = []    # the hand's orientation as read (the last two)
        self._cool = 0          # steps left in which nothing is watched
        self._cooling = False   # this step is one of them
        self._tries = 0         # pulses sent in a row without an answer
        self._n_caught = 0      # ``unlock``: steps in a row a straight-elbowed hand was asked in and did not come
        self._R_free = None     # ``unlock_back``: the hand's orientation as last read with the elbow free
        self._adopt = False     # after a watch: the next timed start's answer is taken as the lag, with no pulse

    # -- what the robot sent and was told
    def _sig(self, obs: dict):
        q = obs.get("robot0_joint_pos")
        return None if q is None else np.asarray(q, dtype=np.float64).tobytes()

    @staticmethod
    def _gap_of(obs: dict):
        g = obs.get("robot0_gripper_qpos")
        return None if g is None or len(g) < 2 else float(abs(g[0]) + abs(g[1]))

    def start(self, obs: dict) -> None:
        self.pos = [np.asarray(obs["robot0_eef_pos"], dtype=float)]
        self.joints = [self._sig(obs)]
        self.gaps = [self._gap_of(obs)]
        self._last = obs

    def _repeats(self) -> None:
        """A changing reading equal, to the last bit, to the reading of d >= 2 steps before, which had itself just
        changed: the readings are d steps late. (A moving arm never comes back to an earlier reading exactly; a still
        arm repeats itself, and a frozen reading repeats the one before it.)"""
        J = self.joints
        t = len(J) - 1
        d_hit = None
        if J[t] is not None and J[t] != J[t - 1]:  # only a reading that is itself changing (a still arm repeats too)
            for d in range(2, min(int(self.o["max_lag"]), t - 1) + 1):
                if J[t - d] == J[t] and J[t - d - 1] is not None and J[t - d - 1] != J[t - d]:
                    d_hit = d
                    break
        self._n_rep = self._n_rep + 1 if (d_hit is not None and d_hit == self._rep) else (1 if d_hit is not None else 0)
        self._rep = d_hit
        if d_hit is not None and self._n_rep >= int(self.o["repeat_confirm"]) and d_hit + 1 > self.lag:
            # readings later than the lag in use allows: at least their delay and the arm's own step
            self.obs_lag = int(d_hit)
            self.onset = max(0, t - self._n_rep)
            self.rec.setdefault("repeats", []).append({"t": t, "obs_lag": self.obs_lag})
            self.lag = self.obs_lag + 1
            self.rec["lags"].append({"t": t, "lag": self.lag, "from": "repeat"})
            if self.o.get("probe_grip") and self._probe_ok():
                self._pg = {"k0": len(self.acts), "t0": len(self.acts), "g0": float(self.gaps[-1]), "late": False}
                self._pg_last = len(self.acts)
                self._st, self._n_against = None, 0
            else:
                self._measure("repeat")

    def after_step(self, act, obs: dict, foreign: bool = False) -> None:
        """``foreign``: this step's command was neither the robot's nor this object's (the arm check's own motion).
        Read only by the 2026-11-16 options: such steps are neither watched nor timed."""
        self.acts.append(np.asarray(act, dtype=float))
        self.pos.append(np.asarray(obs["robot0_eef_pos"], dtype=float))
        self.joints.append(self._sig(obs))
        self.gaps.append(self._gap_of(obs))
        self._last = obs
        if obs.get("robot0_eef_quat") is not None:
            self.quat = (self.quat + [np.asarray(obs["robot0_eef_quat"], dtype=float)])[-2:]
            if self.o.get("unlock_back") and self.mode == "run" and not self.rec.get("unlock"):
                e = self._elbow()
                if e is not None and e <= float(self.o["elbow_free"]):
                    self._R_free = quat_xyzw_to_mat(obs["robot0_eef_quat"])
        if self.mode == "unlock":
            self._unlock_step()
        elif self.mode == "probe":
            self._probe_step()
        elif self.mode == "run":
            self._cooling, self._cool = self._cool > 0, max(0, self._cool - 1)  # the steps right after a measurement
            self._repeats()
            if self.mode == "run":
                self._leap()
            if self.mode == "run" and self.o.get("fit"):
                self._estimate()
            if self.mode == "run" and self.o.get("start_lag"):
                self._start(foreign)
            if self.mode == "run" and self.o.get("cmd_watch"):
                self._watch(foreign)
            if self.mode == "run" and self.o.get("unlock"):
                self._caught(foreign)
            if self.mode == "run" and self._pg:
                self._probe_step()
        else:
            self._measure_step()
        if self._ge:
            self._echo_step()
        if self.o.get("trace_q") and obs.get("robot0_joint_pos") is not None:  # development: the joints as read
            self.rec.setdefault("trace_q", []).append([round(float(x), 3) for x in obs["robot0_joint_pos"]])
        if self.o.get("trace"):  # development: per step, the hand as read, as predicted for the next command, the lag
            p = self.predict(obs)["robot0_eef_pos"]
            g = obs.get("robot0_gripper_qpos")
            self.rec.setdefault("trace", []).append(
                [round(float(x), 4) for x in self.pos[-1]] + [round(float(x), 4) for x in p] + [int(self.lag)]
                + ([round(float(abs(g[0]) + abs(g[1])), 4)] if g is not None else []) + [self.mode[0]])

    def _leap(self) -> None:
        """A reading that moved further in one step than every command in flight could carry it: readings that were
        late have caught up (a delay ended). The lag is measured again."""
        if self.lag <= 0 or len(self.pos) < 2:
            return
        v = float(np.linalg.norm(self.pos[-1] - self.pos[-2]))
        fly = self.acts[-(self.lag + 1):]
        one = self.gain * OUT_POS * max(float(np.linalg.norm(a[:3])) for a in fly) if fly else 0.0
        if v > max(float(self.o["leap"]) * one, float(self.o["leap_min_m"])):
            self.rec.setdefault("leaps", []).append({"t": len(self.acts), "step_m": round(v, 4)})
            self.obs_lag = 0
            if self.o.get("leap_trust"):
                # the readings have caught up: the delay is over, and no pulse is spent on saying so (in last
                # night's runs a leap was seen in D12 only, 12 of 22 episodes a run, and each pulse after it
                # answered "no lag"). A lag that were still there would show at the next move after a rest.
                self.lag = 0
                self.rec["lags"].append({"t": len(self.acts), "lag": 0, "from": "leap"})
                return
            self._measure("leap")

    # -- 2026-11-17 (``probe_grip``): late readings; whether the commands are late too is asked of the fingers
    def _probe_ok(self) -> bool:
        """The fingers can be asked: the robot has them open and has not yet closed on anything, and they are read
        open and still."""
        if self._closed or self._grip_raw is None or self._grip_raw >= 0.0:
            return False
        return self._still(open_too=True)

    def _still(self, open_too: bool = False) -> bool:
        """The fingers are read still over the last three readings (and, asked, open)."""
        o = self.o
        if len(self.gaps) < 3 or any(x is None for x in self.gaps[-3:]):
            return False
        g = self.gaps[-3:]
        return max(g) - min(g) <= float(o["probe_still_m"]) and (not open_too or min(g) >= float(o["probe_open_m"]))

    def _echo_step(self) -> None:
        """``grip_echo``: one reading while the robot's own gripper command is timed."""
        o, ge, T = self.o, self._ge, len(self.acts)
        g = self.gaps[-1]
        n = T - ge["k"] - 1
        if g is None or n > int(o["max_lag"]) + 4:
            self._ge = {}                            # no answer to read (fingers that could not move): nothing said
            return
        moved = (ge["g0"] - g) if ge["dir"] > 0 else (g - ge["g0"])
        if moved < float(o["probe_dip_m"]):
            return
        self._ge = {}
        was = int(self.lag)
        why = "close" if ge["dir"] > 0 else "open"
        if was > 0 and n <= was - 2:
            # an answer sooner than the lag in use allows: at once (the delay was over before the command), or part
            # way (the readings caught up between the command and its answer). Either way the delay is over, and no
            # leap had shown it
            self.lag, self.obs_lag = 0, 0
            self.rec["lags"].append({"t": T, "lag": 0, "from": "grip", "why": why, "was": was, "after": int(n)})
        elif n >= max(was, 0) + 2 and ge["dir"] > 0:
            # a LATER answer is only noted, never taken for a longer lag. Fingers let go of something answer late by
            # themselves (their command was wound shut on it: 2 to 7 steps in the stored undelayed episodes, more
            # for a wider thing), and one stored S20 release would have turned its lag of 15 into 18
            self.rec.setdefault("echo", []).append({"t": T, "lag": int(n), "why": why, "was": was})

    def _probe_end(self, classic: bool = False) -> None:
        self._pg = {}
        if self.mode == "probe":
            self.mode = "run"
        if classic:
            self._measure("repeat")

    def _probe_step(self) -> None:
        """One reading during the probe. The fingers' first dip is the answer to the probe's first command: the lag
        (``pos[t] - pos[t - 1]`` answers ``acts[t - 1 - lag]``, and the fingers answer as the hand does)."""
        o, pg, T = self.o, self._pg, len(self.acts)
        if self.lag <= 0 or self.obs_lag <= 0:      # the delay ended meanwhile (a leap): nothing left to ask
            self.rec.setdefault("probe", []).append({"t": T, "ended": True})
            return self._probe_end()
        g = self.gaps[-1]
        if g is None:
            return self._probe_end(classic=True)
        if pg.get("again"):
            # the answer is in and the robot is to look and plan again: not before the fingers are read open again
            # (the look tool takes fingers read part shut for a closed hand)
            pg["stopped"] = int(pg.get("stopped", 0)) + 1
            if g >= pg["g0"] - 0.5 * float(o["probe_dip_m"]) or pg["stopped"] - pg["again"] >= 8:
                self._probe_end()
                self._replan = True
            return
        if g <= pg["g0"] - float(o["probe_dip_m"]):
            lag = T - pg["k0"] - 1
            self.lag = int(lag) if lag >= int(o["min_lag"]) else 0
            late = self.lag - self.obs_lag >= 2
            self.onset = pg["k0"]
            self.rec["lags"].append({"t": T, "lag": self.lag, "from": "grip", "why": "repeat", "steps": T - pg["t0"],
                                     **({"stopped": int(pg["stopped"])} if pg.get("stopped") else {})})
            self._cand, self._n_cand = None, 0
            self._tries, self._cool = 0, int(o["watch_cool"])
            if self.lag > 0 and not self._closed and (late or not o.get("replan_late_only")):
                # as after a pulse: commands that were late carried the hand off its plan. The robot stays stopped
                # until its fingers are read open again, then looks and plans
                pg["again"] = int(pg.get("stopped", 0)) + 1
                pg["stopped"] = pg["again"]
                self.mode = "probe"
                return
            self._probe_end()
            return
        n = T - pg["k0"] - 1                        # the lag this reading would be the answer for
        if not pg["late"] and n > self.obs_lag + 1 + int(o["probe_slack"]):
            # no answer where commands on time would have given it: they are late too. The robot is stopped (it has
            # been steering by too short a lag) until the answer comes
            pg["late"], pg["stopped"] = True, 0
            self.mode = "probe"
        elif pg["late"]:
            pg["stopped"] += 1
        if n > int(o["max_lag"]) + int(o["probe_n"]) + 4:
            self.rec.setdefault("probe", []).append({"t": T, "no_answer": True})
            self._probe_end(classic=True)           # no answer at all: the pulse

    # -- the lag measured: a pause, one up-and-down pulse, the answer
    def _measure(self, why: str) -> None:
        self._pg = {}  # (a probe under way is over: the pulse decides)
        self.mode, self._m = "flush", {"why": why, "t0": len(self.acts), "n": 0}
        self._st, self._n_against = None, 0

    def _at_rest(self) -> bool:
        """The readings' last ``rest_n`` steps are all under ``rest_m``."""
        n = int(self.o["rest_n"])
        if len(self.pos) <= n:
            return False
        return all(float(np.linalg.norm(self.pos[-i] - self.pos[-i - 1])) < float(self.o["rest_m"])
                   for i in range(1, n + 1))

    def busy(self) -> bool:
        return self.mode != "run"

    def own_action(self) -> list:
        """One step of the measurement (the robot is not asked): nothing before the pulse and while its answer is
        awaited, +z then -z during it. The gripper's command is kept."""
        g = float(self._grip if self._grip is not None else -1.0)
        if self.mode == "unlock":
            return self._unlock_action()
        if self.o.get("grip_as_sent") and self._grip_raw is not None:
            g = float(self._grip_raw)  # the fingers go on as the robot left them ("keep as it is" stays that)
        m = self._m
        if self.mode == "flush" and m.get("why") == "watch" and self.o.get("watch_pull") and m.get("turn") is not None \
                and not m.get("pulled") and self._last is not None and self._last.get("robot0_eef_quat") is not None:
            # the pull back: toward where the hand turned against its commands, the hand turned as it is
            seen = self.predict(self._last)
            R = quat_xyzw_to_mat(seen["robot0_eef_quat"])
            return follow(seen["robot0_eef_pos"], R, m["turn"], R, g)[:6] + [g]
        if self.mode == "pulse":
            up = self._m["n"] < int(self.o["pulse_n"])
            return [0.0, 0.0, float(self.o["pulse_amp"]) * (1.0 if up else -1.0), 0.0, 0.0, 0.0, g]
        return [0.0] * 6 + [g]

    def _answer(self):
        """(lag, score) of the pulse's shape in the readings' z steps: the lag whose 2 pulse_n steps are all in, whose
        score passes the threshold and whose successor (also all in) scores lower; None until there is one."""
        o, m, T = self.o, self._m, len(self.acts)
        n, k0 = int(o["pulse_n"]), m["k0"]
        sign = np.array([1.0] * n + [-1.0] * n)
        need = float(o["answer_share"]) * 2 * n * float(o["gain"]) * float(o["pulse_amp"]) * OUT_POS
        scores = {}
        for lag in range(0, int(o["max_lag"]) + 2):
            t_last = k0 + lag + 2 * n  # v(t) = pos[t] - pos[t - 1] answers acts[t - 1 - lag]
            if t_last > T:
                break
            vz = np.array([self.pos[k0 + 1 + lag + k][2] - self.pos[k0 + lag + k][2] for k in range(2 * n)])
            up, down = float(vz[:n].sum()), float(vz[n:].sum())
            # both halves must be there: the hand rose, then fell (a hand that was rising and stopped is not an answer)
            scores[lag] = float(sign @ vz) if (up >= 0.25 * need and down <= -0.25 * need) else 0.0
        for lag in sorted(scores):
            if scores[lag] >= need and (lag + 1) in scores and scores[lag + 1] < scores[lag] \
                    and (lag == 0 or scores[lag - 1] <= scores[lag]):
                return lag, scores[lag]
        return None

    def _answer_wait(self, m: dict) -> int:
        """Steps after the pulse's first command by which its answer must have come."""
        o, n = self.o, int(self.o["pulse_n"])
        if o.get("no_answer_assume") and m["why"] == "repeat" and self.obs_lag > 0:
            # long enough for commands as late as the readings (and two steps more), not for every lag there could be
            return min(int(o["max_lag"]), 2 * int(self.obs_lag) + 3) + 2 * n + 4
        return int(o["max_lag"]) + 2 * n + 4

    def _measure_step(self) -> None:
        o, m, T = self.o, self._m, len(self.acts)
        n = int(o["pulse_n"])
        m["n"] += 1
        if self.mode == "flush":
            if m["why"] == "watch" and o.get("watch_pull") and m.get("turn") is not None and not m.get("pulled"):
                # first the pull back (``own_action``), until the readings answer it or the hand is back
                far = float(np.linalg.norm(self.pos[-1] - m["turn"]))
                m["far"] = max(float(m.get("far", 0.0)), far)
                u = m["turn"] - self.pos[-2]
                nu = float(np.linalg.norm(u))
                back = float((self.pos[-1] - self.pos[-2]) @ u) / nu if nu > 1e-6 else 0.0
                m["back"] = m.get("back", 0) + 1 if back >= float(o["pull_answer_m"]) else 0
                if self._unlock_due() and self._at_rest():
                    # the hold is over (the hand stands) and it left the elbow straight: no pull will be answered
                    m["pulled"] = m["n"]
                    m["flew"] = m["far"]
                    self.rec.setdefault("watch", [{}])[-1].update(flew_m=round(m["far"], 4), pulled=int(m["n"]))
                    self._unlock_begin("watch")
                    self._tries = 0
                    return
                if m["back"] >= 2 or far <= float(o["pull_near_m"]) or m["n"] >= int(o["flush_max"]):
                    m["pulled"], m["n"] = m["n"], 0  # then nothing is sent until rest (counted from here)
                return
            if o.get("flush_rest") or m["why"] == "watch":
                # until the hand is read at rest: commands on their way, or a held one, are still moving it
                # (after a watch at least as long as a timed start needs behind it)
                least = max(int(o["flush_n"]), int(o["start_quiet_n"])) if m["why"] == "watch" else int(o["flush_n"])
                go = (m["n"] >= least and self._at_rest()) or m["n"] >= int(o["flush_max"])
            else:
                go = m["n"] >= int(o["flush_n"])
            if go:
                m["flush"] = m["n"]
                if m.get("turn") is not None and "flew" not in m:
                    # how far the hand got from where it turned against its commands
                    m["flew"] = max(float(m.get("far", 0.0)), float(np.linalg.norm(self.pos[-1] - m["turn"])))
                    self.rec.setdefault("watch", [{}])[-1].update(
                        flew_m=round(m["flew"], 4), **({"pulled": int(m["pulled"])} if "pulled" in m else {}))
                if m["why"] == "watch" and self._unlock_due():
                    # the hold left the elbow straight: first the hand is drawn in (``_unlock_action``), then the
                    # new look. No pulse from out there
                    self._unlock_begin("watch")
                    self._tries = 0
                    return
                if m["why"] == "watch" and o.get("start_lag"):
                    # no pulse either: the hand is at rest and nothing is on its way, so the robot's own next
                    # move times what lag is left (``_start``; its answer is taken as it comes). A pulse here would
                    # be sent into the last steps of the trouble it is to measure (D13's late commands drain away
                    # some 35 steps after they begin) and wait a whole ``max_lag`` for nothing.
                    self._adopt = True
                    self.mode = "run"
                    self._tries, self._cool = 0, int(o["watch_cool"])
                    if self._grip is None or self._grip < 0.0:
                        self._replan = True
                    return
                self.mode = "pulse"
                m.update(k0=T, n=0)  # the pulse's first command is acts[k0]
            return
        if self.mode == "pulse" and m["n"] >= 2 * n:
            self.mode = "wait"
        got = self._answer() if T >= m["k0"] + 2 * n else None
        if got is not None:
            lag, score = got
            gain = score / (2 * n * float(o["pulse_amp"]) * OUT_POS)
            lo_g, hi_g = o["gain_range"]
            self.lag = int(lag) if lag >= int(o["min_lag"]) else 0
            if lo_g <= gain <= hi_g:
                # a short pulse moves the arm less than a command that lasts (the arm is still gathering speed):
                # never below the gain a lasting command was measured to give
                self.gain = max(float(gain), float(o["gain"]))
            self.onset = m["k0"]
            self.rec["lags"].append({"t": T, "lag": self.lag, "gain": round(self.gain, 4), "from": "pulse",
                                     "why": m["why"], "steps": T - m["t0"]})
        elif T - m["k0"] < self._answer_wait(m):
            return
        else:
            self.rec["lags"].append({"t": T, "lag": self.lag, "from": "no answer", "why": m["why"]})
            if o.get("no_answer_assume") and m["why"] == "repeat" and self.obs_lag > 0:
                self.lag = 2 * int(self.obs_lag) + 1  # the commands as late as the readings (S20)
                self.rec["lags"].append({"t": T, "lag": self.lag, "from": "assumed", "why": "no answer"})
            if m["why"] == "watch" and self._tries + 1 < int(o["watch_tries"]):
                # the pulse went into a hold (nothing sent was carried out): once more, from rest
                self._tries += 1
                self.mode, self._m = "flush", {"why": "watch", "t0": T, "n": 0, "turn": m.get("turn"),
                                               **({"flew": m["flew"]} if "flew" in m else {})}
                return
        if self.mode == "pulse":  # the answer came before the pulse was all sent (no lag): send the rest of it
            m["early"] = True
            return
        self.mode = "run"
        self._cand, self._n_cand = None, 0
        self._tries, self._cool = 0, int(o["watch_cool"])
        # a delay that begins replays the commands of the steps before it (S20: the first seven, so the hand goes
        # twice as far as it was sent before anything can be noticed). The way planned at the start no longer starts
        # where the hand is: while the robot has not yet closed on anything, it should look and plan again.
        # (``cmd_watch``: the same after a hold or late commands, whatever lag is left: the hand flew on.)
        late = self.lag - self.obs_lag >= 2  # the commands are late too (not only the readings)
        if m["why"] == "watch":
            # the hand flew on: a new look and plan whenever the hand is open now (nothing in it to lose), even
            # after an earlier grasp that failed
            if self._grip is None or self._grip < 0.0:
                self._replan = True
        elif self.lag > 0 and not self._closed and (late or not o.get("replan_late_only")):
            self._replan = True

    # -- 2026-11-16: an arm caught with its elbow straight is drawn in (``unlock``)
    def _elbow(self):
        q = (self._last or {}).get("robot0_joint_pos")
        k = int(self.o["elbow_joint"])
        return None if q is None or len(q) <= k else float(q[k])

    def _out(self) -> np.ndarray:
        """The level direction from the robot to its hand (as read)."""
        d = np.asarray(self.pos[-1][:2], dtype=float) - np.asarray(self.o["base_xy"], dtype=float)
        n = float(np.linalg.norm(d))
        return np.array([d[0] / n, d[1] / n, 0.0]) if n > 1e-6 else np.array([1.0, 0.0, 0.0])

    def _unlock_due(self) -> bool:
        e = self._elbow()
        return bool(self.o.get("unlock")) and e is not None and e > float(self.o["elbow_straight"]) \
            and (self._grip is None or self._grip < 0.0) and len(self.rec.get("unlock", [])) < int(self.o["caught_max"])

    def _unlock_begin(self, why: str) -> None:
        q = (self._last or {}).get("robot0_eef_quat")
        self.mode, self._u = "unlock", {"n": 0, "t0": len(self.acts), "why": why, "elbow0": self._elbow(),
                                        "p0": self.pos[-1].copy(), "phase": "in", "n_deep": 0, "n_back": 0,
                                        "R0": self._R_free if self._R_free is not None
                                        else (None if q is None else quat_xyzw_to_mat(q))}
        self._n_caught, self._st, self._n_against = 0, None, 0

    def _unlock_left_deg(self) -> float:
        """How far the hand is turned from the way it stood with the elbow free (or, never read so, when the unlock
        began): degrees; 0 without readings."""
        q, R0 = (self._last or {}).get("robot0_eef_quat"), self._u.get("R0")
        if q is None or R0 is None:
            return 0.0
        c = (float(np.trace(R0 @ quat_xyzw_to_mat(q).T)) - 1.0) / 2.0
        return float(np.degrees(np.arccos(min(1.0, max(-1.0, c)))))

    def _unlock_action(self) -> list:
        """The pull toward the robot with the turn that swings the fingers toward it (about the level axis square
        to the pull; the controller's turns are about the world's axes). Then, with ``unlock_back``: the whole hand
        drawn in as it is until the elbow is well bent (the wrist has to come in, and the pull with the turn brought
        in the grip point only), and then the turn back to the way the hand stood."""
        o, out = self.o, self._out()
        if self._u.get("phase") == "deep":  # the whole hand drawn in as it is, until the elbow is well bent
            a = -float(o["unlock_deep_in"]) * out + np.array([0.0, 0.0, float(o["unlock_deep_up"])])
            return [float(x) for x in a] + [0.0, 0.0, 0.0, -1.0]
        if self._u.get("phase") == "back":
            q = self._last["robot0_eef_quat"]
            R = quat_xyzw_to_mat(q)
            turn = follow(self.pos[-1], R, self.pos[-1], self._u["R0"], -1.0)[3:6]
            return [float(x) for x in -float(o["unlock_back_in"]) * out] + [float(x) for x in turn] + [-1.0]
        turn = np.cross([0.0, 0.0, 1.0], out)  # for a hand straight ahead of the robot (+x): the world's +y
        a = np.r_[-float(o["unlock_in"]) * out, float(o["unlock_turn"]) * turn]
        return [float(x) for x in a] + [-1.0]

    def _unlock_step(self) -> None:
        o, u, T = self.o, self._u, len(self.acts)
        e = self._elbow()
        if u["phase"] == "in":
            u["n"] += 1
            if not ((e is not None and e <= float(o["elbow_free"])) or u["n"] >= int(o["unlock_max"])):
                return
            u["elbow1"], u["came"] = e, round(float((u["p0"] - self.pos[-1]) @ self._out()), 4)
            if o.get("unlock_back") and e is not None and e <= float(o["elbow_free"]) and u["R0"] is not None \
                    and (self._last or {}).get("robot0_eef_quat") is not None:
                u["phase"] = "deep"  # the hand is in, its fingers at the robot, the wrist still out where it was
                return
        elif u["phase"] == "deep":
            u["n_deep"] += 1
            straight = e is None or e > float(o["elbow_straight"])  # caught again: nothing more is tried
            if not straight and e > float(o["unlock_deep"]) and u["n_deep"] < int(o["unlock_deep_max"]):
                return
            u["elbow2"] = e
            if not straight:
                u["phase"] = "back"
                return
        else:
            u["n_back"] += 1
            straight = e is None or e > float(o["elbow_straight"])
            if not straight and self._unlock_left_deg() > float(o["unlock_back_deg"]) \
                    and u["n_back"] < int(o["unlock_back_max"]):
                return
        self.rec.setdefault("unlock", []).append(
            {"t": T, "steps": u["n"], "why": u["why"], "elbow": [u["elbow0"], u["elbow1"]], "came_in_m": u["came"],
             **({"deep": {"steps": u["n_deep"], "elbow": u.get("elbow2")}} if u["n_deep"] else {}),
             **({"back": {"steps": u["n_back"], "left_deg": round(self._unlock_left_deg(), 1), "elbow": e}}
                if u["phase"] == "back" else {})})
        self.mode = "run"
        self._cool = int(o["watch_cool"])
        self._replan = True  # the hand is somewhere else (and, without the turn back, turned): a new look and plan

    # -- 2026-11-16: the lag from the first move after a rest; a hand that moves against its commands
    def _start(self, foreign: bool = False) -> None:
        """``start_lag``. With the hand at rest and nothing on its way, the step at which the readings answer the
        next command is the loop's lag now (``pos[t] - pos[t - 1]`` answers ``acts[t - 1 - lag]``)."""
        o, T = self.o, len(self.acts)
        j = T - 1                                     # the newest command; pos[T] is the reading after it
        if foreign:
            self._st = None
            return
        st = self._st
        if st is None:
            # nothing on its way: the hold after a gripper command is exactly ``lag`` steps of nothing, so that many
            # quiet commands are asked for (and six with no lag in use). The command before the hold is the robot
            # keeping its pose, up to 0.12 with the hand 3 mm off; by the lag in use it has been carried out before
            # the two steps of rest checked below, and a hand it still moved would not be at rest. (Mac run
            # of D12: 0.07, eleven zeros, then a move of 0.30 that the readings answered at once. With
            # ``lag + 2`` asked for, that move went untimed, the lag stayed at 11 after the delay had ended, and the
            # lift that followed, steered by a hand predicted 11 commands ahead, shook the bowl out of the fingers.)
            q = max(int(o["start_quiet_n"]), self.lag)
            a = self.acts[j][:3]
            na = float(np.linalg.norm(a))
            if na < float(o["start_cmd"]) or j < q:
                return
            if any(float(np.linalg.norm(x[:3])) >= float(o["start_quiet"]) for x in self.acts[j - q:j]):
                return
            n = int(o["rest_n"])  # at rest before the command: the readings up to pos[j]
            if j < n or any(float(np.linalg.norm(self.pos[j - i] - self.pos[j - i - 1])) >= float(o["rest_m"])
                            for i in range(n)):
                return
            self._st = {"k": j, "u": a / na, "p0": self.pos[j].copy(), "at": None, "s": 0.0}
            st = self._st
        if st["k"] != j and float(self.acts[j][:3] @ st["u"]) < -float(o["start_quiet"]):
            self._st = None                           # the robot turned round before any answer: nothing to read
            return
        s_now = float((self.pos[T] - st["p0"]) @ st["u"])
        if st["at"] is None:
            if s_now >= float(o["start_move_m"]):
                st["at"], st["s"] = T, s_now
            elif T - st["k"] > int(o["max_lag"]) + 6:
                self._st = None                       # no answer at all (a blocked hand): nothing to read
            return
        if s_now <= st["s"]:                          # it did not go on: a twitch, not the answer
            st["at"] = None
            return
        lag = st["at"] - st["k"] - 1
        was = self.lag
        self._st = None
        closed = self._grip is not None and self._grip > 0.0
        self.rec.setdefault("starts", []).append({"t": T, "lag": int(lag), "was": int(was),
                                                  **({"closed": True} if closed else {})})
        adopt, self._adopt = self._adopt, False
        if adopt and was == 0 and not closed and lag >= int(o["start_late_min"]):
            self.lag = int(lag)                       # after a watch: commands are late by this much
            self.rec["lags"].append({"t": T, "lag": int(lag), "from": "start", "was": 0, "why": "watch"})
        elif lag < int(o["min_lag"]):
            if was > 0:                               # the delay has ended (no leap showed it)
                self.lag, self.obs_lag = 0, 0
                self.rec["lags"].append({"t": T, "lag": 0, "from": "start", "was": int(was)})
        elif closed:
            # a closed hand holds something (an object, a drawer's handle): what it holds can keep the hand back
            # for a step or two, and a pulse would shake it. Only an answer at once is believed with the hand closed
            pass
        elif was == 0:
            if lag >= int(o["start_late_min"]):
                self._measure("start")                # a delay nothing had shown: measured exactly
        elif abs(lag - was) >= 2:
            self.lag = int(lag)
            self.rec["lags"].append({"t": T, "lag": int(lag), "from": "start", "was": int(was)})

    def _turned(self) -> float:
        """The angle between the last two orientation readings (rad); 0 without them."""
        if len(self.quat) < 2:
            return 0.0
        a, b = self.quat[-1], self.quat[-2]
        d = abs(float(a @ b)) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)) + 1e-12)
        return 2.0 * float(np.arccos(min(1.0, d)))

    def _watch(self, foreign: bool = False) -> None:
        """``cmd_watch``. Commands that are held or late show in the readings' own timing: the commands move on and
        the hand's motion does not.

        The strict rule (the default). The hand moves (``watch_min_m`` a step or more), its step is the same as the
        step before (within ``watch_steady`` of itself) and it is not turning (``watch_turn_rad``, as read): a
        stretch of such steps is the hand carrying out one command, or a slowly changing one. The stretch is
        anchored at a step that points along one of the last ``watch_trail`` commands due (``watch_agree``): that
        command is the one being carried out, and the anchor moves up for as long as such a command is found. An
        arm that obeys trails a changing command by about three steps (its controller's own lag), so it always
        finds one. From the anchor the commands due ask for a different step (by the stretch's own gain: the
        anchored step's length for the anchored command's size). ``watch_n`` steps in a row in which no recent
        command explains the hand's step, the change asked for is at least ``watch_asked`` of the anchored step and
        the hand has made no more than ``watch_made`` of it, are commands held or late.
        What the loose rule (a moving hand ``watch_n`` steps against the commands due) took for that in episodes
        with no delay, 2026-11-16: the hand turning as it moves (the grip point then travels with no move asked,
        or not the way asked), the turn-about at full speed at a waypoint together with such a turn, and the first
        long move. None of them is a steady, unturning step."""
        o, T = self.o, len(self.acts)
        strict = o.get("watch_rule") != "loose"
        v = self.pos[T] - self.pos[T - 1]
        nv = float(np.linalg.norm(v))
        if strict and nv > float(o["watch_jump_m"]):
            self._jump_cool = int(o["watch_jump_cool"])
        jumped, self._jump_cool = self._jump_cool > 0, max(0, self._jump_cool - 1)
        if self._cooling or foreign or jumped or (self.lag > 0 and not o.get("watch_lagged")):
            self._n_against, self._steady = 0, None
            return
        hi = T - self.lag                             # commands [.., hi - 1] are due: acts[hi - 1] moved this step
        lo = max(0, hi - int(o["watch_back"]))
        against, ref = False, None
        if nv >= float(o["watch_min_m"]) and hi > lo:
            ref = np.mean([a[:3] for a in self.acts[lo:hi]], axis=0)
            nr = float(np.linalg.norm(ref))
            against = nr < float(o["watch_idle"]) or float(v @ ref) / (nv * nr) < float(o["watch_cos"])
        if strict:
            vp = self.pos[T - 1] - self.pos[T - 2] if T >= 2 else None
            last = self.acts[max(0, hi - int(o["watch_n"])):hi]
            asked_turn = float(np.mean([np.linalg.norm(a[3:6]) for a in last])) if last else 0.0
            same = ref is not None and vp is not None \
                and float(np.linalg.norm(v - vp)) <= float(o["watch_steady"]) * nv \
                and self._turned() < float(o["watch_turn_rad"]) and asked_turn < float(o["watch_cmd_turn"])
            against = False
            if not same:
                self._steady = None
            else:
                # the command the hand is carrying out: the one of the last few due that its step points along
                # (an arm that obeys trails its commands by about three steps, its controller's own lag)
                best, m = -2.0, None
                for a in self.acts[max(0, hi - int(o["watch_trail"])):hi]:
                    na_ = float(np.linalg.norm(a[:3]))
                    if na_ >= float(o["watch_idle"]) and float(v @ a[:3]) / (nv * na_) > best:
                        best, m = float(v @ a[:3]) / (nv * na_), np.asarray(a[:3], dtype=float)
                st = self._steady
                if st is None:
                    st = self._steady = {"n": 0, "m": None, "v0": None, "g": 0.0}
                st["n"] += 1
                if best >= float(o["watch_agree"]):
                    st.update(m=m, v0=v.copy(), g=nv / float(np.linalg.norm(m)))  # the anchor (it moves up with them)
                else:
                    # no command of the last few explains the step. ``watch_old``: one sent longer ago may (late
                    # commands; a hold already under way when the stretch began): the stretch is anchored on it
                    bo, mo = -2.0, None
                    for a in self.acts[max(0, hi - int(o["watch_old"])):max(0, hi - int(o["watch_trail"]))]:
                        na_ = float(np.linalg.norm(a[:3]))
                        if na_ >= float(o["watch_idle"]) and float(v @ a[:3]) / (nv * na_) > bo:
                            bo, mo = float(v @ a[:3]) / (nv * na_), np.asarray(a[:3], dtype=float)
                    if bo >= float(o["watch_agree"]):
                        st.update(m=mo, v0=v.copy(), g=nv / float(np.linalg.norm(mo)))
                    if st["m"] is not None:
                        asked = st["g"] * (ref - st["m"])  # the change of step the commands due ask for
                        made = v - st["v0"]
                        na, n0 = float(np.linalg.norm(asked)), float(np.linalg.norm(st["v0"]))
                        against = na >= float(o["watch_asked"]) * n0 \
                            and float(np.linalg.norm(made)) <= float(o["watch_made"]) * na
        self._n_against = self._n_against + 1 if against else 0
        if self._n_against == 1:
            self._turn_p = self.pos[T - 1].copy()  # about where the robot wanted its hand to turn or stop
        if self._n_against >= int(o["watch_n"]):
            note = {"t": T, "step_m": round(nv, 4), **({"steady": int(self._steady["n"])} if self._steady else {})}
            if o.get("watch_dry"):
                self.rec.setdefault("watch_dry", []).append(note)
                self._n_against, self._steady = 0, None
                return
            self.rec.setdefault("watch", []).append(note)
            turn = self._turn_p
            self._measure("watch")
            self._m["turn"] = turn

    def _caught(self, foreign: bool = False) -> None:
        """``unlock``. An open hand whose elbow reads straight, asked to come toward the robot and not coming, step
        after step, with no lag in use: the arm is caught at the edge of its reach."""
        o, T = self.o, len(self.acts)
        if foreign or self._cooling or self.lag > 0 or not self._unlock_due():
            self._n_caught = 0
            return
        out = self._out()
        asked = -float(self.acts[T - 1][:3] @ out) >= float(o["caught_in"])
        came = float((self.pos[T - 1] - self.pos[T]) @ out)
        self._n_caught = self._n_caught + 1 if (asked and (came < float(o["caught_m"]) or o.get("caught_elbow"))) else 0
        if self._n_caught >= int(o["caught_n"]):
            self._unlock_begin("caught")

    def replan_due(self) -> bool:
        """True once after a lag was measured during the robot's first approach: the runner looks and plans again."""
        due, self._replan = self._replan, False
        return due

    def fit(self, lag: int):
        """(gain, unexplained share, drive, steps) of the reported steps against the commands ``lag`` steps earlier,
        over the window's steps that have such a command. None with fewer than ``min_n`` steps or no motion."""
        T = len(self.acts)
        lo = max(lag + 1 + self.onset, T - int(self.o["window"]) + 1)  # only commands sent since the delay began
        n = T - lo + 1
        if n < int(self.o["min_n"]):
            return None
        V = np.stack([self.pos[t] - self.pos[t - 1] for t in range(lo, T + 1)])
        A = np.stack([self.acts[t - 1 - lag][:3] for t in range(lo, T + 1)]) * OUT_POS
        aa, vv = float((A * A).sum()), float((V * V).sum())
        if aa <= 1e-12 or vv <= 1e-12:
            return None
        g = float((V * A).sum() / aa)
        return g, float(((V - g * A) ** 2).sum()) / vv, aa / (OUT_POS * OUT_POS), n

    def _estimate(self) -> None:
        """The lag fitted from the robot's own motion (what a delay of the commands alone leaves to go by)."""
        o = self.o
        lo_g, hi_g = o["gain_range"]
        fits = {}
        for lag in range(max(0, self.obs_lag), int(o["max_lag"]) + 1):
            f = self.fit(lag)
            if f is not None and f[2] >= float(o["min_drive"]):
                fits[lag] = f
        cur = fits.get(self.lag)
        if not fits:
            self._cand, self._n_cand = None, 0
            return
        best_share = min(f[1] for f in fits.values())
        best = min(lag for lag, f in fits.items() if f[1] <= best_share + float(o["near"]))  # the shortest near the best
        if 0 < best < int(o["min_lag"]) and 0 in fits:
            best = 0
        bf = fits[best]
        ok = (best != self.lag and bf[1] <= float(o["explains"]) and lo_g <= bf[0] <= hi_g
              and (cur is None or bf[1] <= float(o["better"]) * cur[1]))
        if not ok:
            self._cand, self._n_cand = None, 0
            if cur is not None and self.lag and lo_g <= cur[0] <= hi_g and cur[1] <= float(o["explains"]):
                self.gain = float(cur[0])  # the lag in use explains the motion: its gain follows the arm
            return
        self._n_cand = self._n_cand + 1 if self._cand == best else 1
        self._cand = best
        if self._n_cand >= int(o["confirm"]):
            self.lag, self.gain = int(best), float(bf[0])
            self.rec["lags"].append({"t": len(self.acts), "lag": self.lag, "gain": round(self.gain, 4)})
            self._cand, self._n_cand = None, 0

    def shape(self, act):
        """The command to send: the robot's own, unchanged, except during a ``probe_grip`` probe (its first
        ``probe_n`` commands close the fingers, and all of them are sent at ``probe_scale`` until the answer)."""
        pg = self._pg
        if not pg or self.mode != "run":
            return act
        a = np.array(act, dtype=float)
        a[:6] *= float(self.o["probe_scale"])
        if len(self.acts) - pg["k0"] < int(self.o["probe_n"]):
            a[6] = 1.0
        return a

    # -- what the robot is given
    def predict(self, obs: dict) -> dict:
        """The readings with the hand where the next command will find it (the readings themselves when no lag)."""
        if self.lag <= 0 or not self.acts:
            return obs
        fly = self.acts[-self.lag:]
        p = np.asarray(obs["robot0_eef_pos"], dtype=float) + self.gain * OUT_POS * np.sum([a[:3] for a in fly], axis=0)
        R = quat_xyzw_to_mat(obs["robot0_eef_quat"])
        for a in fly:  # each command turns the hand about the world's axes, in the order sent
            R = _exp(self.gain * OUT_ROT * a[3:6]) @ R
        return {**obs, "robot0_eef_pos": p, "robot0_eef_quat": np.asarray(mat_to_quat_xyzw(R), dtype=float)}

    # -- the gripper
    def sent(self, act) -> None:
        """The robot's command for this step: a change of the gripper's direction starts a hold of ``lag`` steps."""
        g = float(np.sign(float(act[6])))
        self._grip_raw = float(act[6])
        if self._pg and g >= 0.0:
            # the robot wants its fingers itself: their reading is no longer the probe's alone (the pulse, as before)
            self.rec.setdefault("probe", []).append({"t": len(self.acts), "robot_closed": True})
            self._probe_end(classic=True)
        if self.o.get("grip_echo") and g != 0.0 and self._grip is not None and g != self._grip and not self._pg \
                and self.mode == "run" and self._still() \
                and (self._pg_last is None or len(self.acts) - self._pg_last > int(self.lag) + int(self.o["probe_n"]) + 6):
            # the robot turns its fingers the other way from rest: the step at which their reading follows is timed
            # (not while a probe's own dip may still be on its way through late readings)
            self._ge = {"k": len(self.acts), "g0": float(self.gaps[-1]), "dir": g}
        if g > 0.0:
            self._closed = True
        if g != 0.0:
            if self._grip is not None and g != self._grip and self.lag > 0:
                self._hold = int(self.lag)
                if (g < 0.0 and self.o.get("open_hold_cmd")) or (g > 0.0 and self.o.get("close_hold_cmd")):
                    # only until the fingers have the command (the readings' delay is not waited for). Opening: the
                    # release waits for no more. Closing (2026-11-17): the agents close blind (``lag_close_blind``)
                    self._hold = max(0, int(self.lag) - int(self.obs_lag) - 1) if self.obs_lag > 0 else int(self.lag)
                self.rec["holds"] += 1
            self._grip = g

    def holding(self) -> bool:
        return self._hold > 0

    def hold_action(self) -> list:
        """One step of the hold: no motion, the gripper's command kept."""
        self._hold -= 1
        return [0.0] * 6 + [float(self._grip if self._grip is not None else -1.0)]
