"""A scripted arm motion for the depth net's training frames (truth side; used only by ``colour_frames``' worker).

The three suites' moving-arm frames came from replays of stored action chains (``colour_frames.stored_chains``). A
scene with no stored chain gets them from a SWEEP instead: the long suite's tables, and above all the plain version of
a benchmark test task, whose only case is BASE with a handful of start states (so the moving arm, and the objects it
pushes, are most of what can vary). A sweep is an ordinary rollout in the unchanged scene: the hand visits a seeded
series of points over the workspace, most of them just above an object or a fixture, with the gripper open or closed
and, on some legs, a turn about the vertical axis. The objects' positions, the hand's pose and the joint angles are read
from the simulator (training data: truth is allowed here). Nothing in the scene is edited.

What keeps a sweep moving (each from a Mac check on 2026-11-15, numbers only):

* most legs go as a pick-and-place move does: up to a clear height, across, then down to their point. The others
  (``p_straight``), and short hops, go in a straight line, which is what pushes objects to new places. Straight lines
  alone left the fingers caught inside the study table's caddy for half a sweep;
* a part of a leg that stops getting closer (the hand rests on something) is given up for the next part;
* a point is kept where the arm reaches it bent: its wrist (behind and above a hand that points down) neither too far
  from the shoulder nor too near. A far, high point stretched the arm out straight, and it stayed there;
* the hand's orientation is held (the start's, turned about the vertical axis by the legs' turns): left free, pushes
  turned the hand over and the arm ended in poses it could not leave;
* when a joint still comes near its limit, the leg is dropped and the hand goes back towards where it started;
* a hand that has not moved for ``stuck_steps`` ends the sweep (``Sweep.stuck``): the frames so far are kept.

Each leg takes a fixed number of draws from ``random.Random(seed)``, so the series of points does not depend on the
physics: a plan item gives the same legs on every machine (how far the hand gets on a leg may differ in the last bits,
as in any replay). Everything down to ``Sweep`` is pure (numpy); ``hand``, ``hand_rot``, ``joints``, ``shoulder`` and
``anchors`` read the simulator.
"""

from __future__ import annotations

import math
import random

import numpy as np

SWEEP = {
    "steps": 240,            # one sweep's length (the plan's default); frames every ``every`` steps of it
    "leg_steps": (40, 64),   # a leg's step budget (drawn); the hand covers about 11 mm a step over a leg (Mac check)
    "dwell_steps": (0, 8),   # steps a leg stays at its point once reached (the gripper's command acts), then the next
    "tol_m": 0.02,           # reached: the hand within this of where it is going
    "gain": 2.0,             # as the robot's follower: a = clip(gain * error / out), its norm capped at 1
    "out_pos_m": 0.05,       # OSC_POSE: a full command moves the hand's goal 5 cm
    "out_rot": 0.5,          # and turns it 0.5 rad
    "p_anchor": 0.65,        # a leg aims near an object or a fixture; otherwise at a free point over the workspace
    "jitter_m": 0.05,        # sideways offset from the anchor
    "above_m": (0.0, 0.18),  # height over the anchor's origin
    "margin_m": 0.08,        # the free points' box: the anchors' extent in x and y, widened by this
    "p_close": 0.3,          # the gripper closes for the leg (else it opens)
    "p_turn": 0.4,           # the leg turns the hand about the vertical axis
    "turn": 0.5,             # by at most this (rad) times ``yaw_per_turn``; without the hand's pose, a rate for
    "turn_steps": (2, 8),    # this many steps (share of the full rotation command)
    "yaw_per_turn": 2.0,
    "yaw_max": 1.3,          # the hand stays within this (rad) of its start about the vertical axis
    "wrist_m": (0.38, 0.64),  # the wrist stays this far from the shoulder: the arm neither folds up nor stretches out
    "wrist_back_m": 0.088,   # the wrist's place for a hand that points down: this far behind the grip point
    "wrist_up_m": 0.204,     # and this far above it
    "shoulder_up_m": 0.333,  # the Panda's shoulder above its base
    "top_m": 0.05,           # no point higher than this above where the hand started
    "clear_m": 0.04,         # the crossing height: at least this above the higher of the leg's two ends
    "via_tol_m": 0.04,       # the crossing's corners are cut: within this of a corner the next part starts
    "p_straight": 0.3,       # the share of legs that go in a straight line (they push what is in the way)
    "hop_m": 0.06,           # a leg whose point is this near sideways goes straight
    "stall_steps": 8,        # a part of a leg that got no closer (by 5 mm) for this many steps is given up
    "limit_margin": 0.25,    # a joint this near (rad) its limit drops the leg: the hand goes back towards its start
    "retreat_steps": 20,     # for at most this many steps, or until within ``retreat_m`` of the start
    "retreat_m": 0.05,
    "grace_steps": 10,       # the next leg's first steps are not checked (the joints are still coming back)
    "stuck_steps": 36,       # a hand that stayed within ``stuck_m`` for this many steps: the sweep is over
    "stuck_m": 0.015,
}
N_DRAWS = 12  # random numbers per leg, whatever the leg turns out to be
PANDA_LIMITS = ((-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698), (-2.8973, 2.8973),
                (-0.0175, 3.7525), (-2.8973, 2.8973))  # the arm's seven joints (rad)


def clamp_reach(p, shoulder, lo: float, hi: float) -> np.ndarray:
    """``p`` moved along its ray from the shoulder to a distance within [lo, hi]."""
    p, s = np.asarray(p, dtype=float), np.asarray(shoulder, dtype=float)
    v = p - s
    d = float(np.linalg.norm(v))
    if d < 1e-9:
        v, d = np.array([1.0, 0.0, 0.0]), 1.0
    if d > hi:
        return s + v * (hi / d)
    if d < lo:
        return s + v * (lo / d)
    return p.copy()


def wrist_of(p, shoulder, o: dict | None = None) -> np.ndarray:
    """Where the arm's wrist is when a hand pointing down holds its grip point at ``p``: above it, and a little back
    towards the shoulder's axis."""
    o = {**SWEEP, **(o or {})}
    p, s = np.asarray(p, dtype=float), np.asarray(shoulder, dtype=float)
    h = p[:2] - s[:2]
    n = float(np.linalg.norm(h))
    u = h / n if n > 1e-9 else np.array([1.0, 0.0])
    return p + np.array([-float(o["wrist_back_m"]) * u[0], -float(o["wrist_back_m"]) * u[1], float(o["wrist_up_m"])])


def clamp_point(p, shoulder, o: dict | None = None) -> np.ndarray:
    """``p`` moved so that its wrist (``wrist_of``) is within ``wrist_m`` of the shoulder."""
    o = {**SWEEP, **(o or {})}
    p = np.asarray(p, dtype=float)
    w = wrist_of(p, shoulder, o)
    return p + (clamp_reach(w, shoulder, *o["wrist_m"]) - w)


def near_limit(q, margin: float, limits=PANDA_LIMITS) -> bool:
    """Any of the arm's joint angles ``q`` within ``margin`` of its limit (or past it)."""
    return any(float(a) < lo + margin or float(a) > hi - margin for a, (lo, hi) in zip(q, limits))


def rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotvec_of(R) -> np.ndarray:
    """The rotation vector (axis times angle) of a rotation matrix; zeros for no rotation (and for a half turn, whose
    axis this does not look for: a held orientation is never that far off)."""
    R = np.asarray(R, dtype=float)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.zeros(3)
    return v / s * math.atan2(s / 2.0, (float(np.trace(R)) - 1.0) / 2.0)


def _between(u: float, lo: int, hi: int) -> int:
    return int(lo) + min(int(u * (int(hi) - int(lo) + 1)), int(hi) - int(lo))


def _capped(a) -> np.ndarray:
    a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
    n = float(np.linalg.norm(a))
    return a / n if n > 1.0 else a


def next_leg(rng: random.Random, p0, anchors, shoulder, o: dict | None = None) -> dict:
    """The next leg: ``pos`` (the hand's point, world), ``grip`` (-1 open, +1 close), ``turn`` (about the vertical
    axis; 0 for none), ``turn_steps``, ``steps`` (budget), ``dwell`` and ``via`` (0 to 1: under ``p_straight`` the leg
    goes straight, above it how high it crosses). ``p0``: where the hand started; ``anchors``: (N, 3) origins of the
    scene's objects and fixtures (may be empty); ``shoulder``: (3,)."""
    o = {**SWEEP, **(o or {})}
    u = [rng.random() for _ in range(N_DRAWS)]
    p0 = np.asarray(p0, dtype=float)
    A = np.asarray(anchors, dtype=float).reshape(-1, 3)
    top = float(p0[2]) + float(o["top_m"])
    if len(A):
        lo, hi = A.min(0), A.max(0)
    else:  # nothing to aim at: a box ahead of and below the start
        lo, hi = p0 + np.array([-0.05, -0.25, -0.20]), p0 + np.array([0.30, 0.25, 0.0])
    floor = float(lo[2])
    if len(A) and u[0] < float(o["p_anchor"]):
        a = A[min(int(u[1] * len(A)), len(A) - 1)]
        j, (z0, z1) = float(o["jitter_m"]), o["above_m"]
        pos = a + np.array([(2 * u[2] - 1) * j, (2 * u[3] - 1) * j, z0 + u[4] * (z1 - z0)])
        kind = "anchor"
    else:
        m = float(o["margin_m"])
        pos = np.array([lo[0] - m + u[2] * (hi[0] - lo[0] + 2 * m), lo[1] - m + u[3] * (hi[1] - lo[1] + 2 * m),
                        floor + 0.02 + u[4] * max(top - floor - 0.02, 0.0)])
        kind = "free"
    pos[2] = min(max(float(pos[2]), floor), max(top, floor))
    pos = clamp_point(pos, shoulder, o)
    return {"pos": [float(x) for x in pos], "kind": kind,
            "grip": 1.0 if u[5] < float(o["p_close"]) else -1.0,
            "turn": (2 * u[7] - 1) * float(o["turn"]) if u[6] < float(o["p_turn"]) else 0.0,
            "turn_steps": _between(u[8], *o["turn_steps"]),
            "steps": _between(u[9], *o["leg_steps"]),
            "dwell": _between(u[10], *o["dwell_steps"]),
            "via": float(u[11])}


def move_action(eef, goal, grip: float, turn: float = 0.0, o: dict | None = None, rot=None) -> list:
    """One OSC_POSE action (7 numbers in [-1, 1]) toward ``goal``. Rotation: ``rot`` (a rotation-vector command) when
    given, else no tilt and ``turn`` about the vertical axis."""
    o = {**SWEEP, **(o or {})}
    a = _capped(float(o["gain"]) * (np.asarray(goal, dtype=float) - np.asarray(eef, dtype=float)) / float(o["out_pos_m"]))
    r = [0.0, 0.0, float(turn)] if rot is None else [float(x) for x in rot]
    return [float(a[0]), float(a[1]), float(a[2]), r[0], r[1], r[2], float(grip)]


class Sweep:
    """The legs of one sweep, one action per step. ``seed``: the plan item's name (``colour_frames.frame_seed``);
    ``R0``: the hand's rotation matrix at the start (held, when ``act`` is given the hand's rotation too)."""

    PARTS = ("up", "across", "down")

    def __init__(self, seed: str, p0, shoulder, o: dict | None = None, R0=None):
        self.o = {**SWEEP, **(o or {})}
        self.rng = random.Random(seed)
        self.p0 = np.array(p0, dtype=float)
        self.shoulder = np.array(shoulder, dtype=float)
        self.R0 = None if R0 is None else np.array(R0, dtype=float).reshape(3, 3)
        self.yaw = 0.0
        self.leg: dict | None = None
        self.part = "down"
        self.left = self.turn_left = self.stall = self.retreat = self.grace = 0
        self.best = float("inf")
        self.via_z = 0.0
        self.start = self.p0.copy()
        self.reached = self.stuck = False
        self.trail: list = []
        self.n_legs = self.n_reached = self.n_retreats = 0

    def _begin(self, eef, anchors) -> None:
        o = self.o
        self.leg = next_leg(self.rng, self.p0, anchors() if callable(anchors) else anchors, self.shoulder, o)
        pos = np.asarray(self.leg["pos"])
        self.left = int(self.leg["steps"])
        self.turn_left = int(self.leg["turn_steps"]) if self.leg["turn"] else 0
        self.yaw = min(max(self.yaw + float(self.leg["turn"]) * float(o["yaw_per_turn"]), -float(o["yaw_max"])),
                       float(o["yaw_max"]))
        self.reached = False
        self.n_legs += 1
        self.start = eef.copy()
        top = max(float(self.p0[2]) + float(o["top_m"]), float(eef[2]), float(pos[2]))
        clear = min(max(float(eef[2]), float(pos[2])) + float(o["clear_m"]), top)
        via, p = float(self.leg["via"]), float(o["p_straight"])
        straight = via < p or float(np.linalg.norm((pos - eef)[:2])) <= float(o["hop_m"])
        self.via_z = clear + (max(via - p, 0.0) / max(1.0 - p, 1e-9)) * (top - clear)
        self._part("down" if straight else "up")

    def _part(self, part: str) -> None:
        self.part, self.best, self.stall = part, float("inf"), 0

    def _goal(self) -> np.ndarray:
        pos = np.asarray(self.leg["pos"])
        if self.part == "up":
            return np.array([self.start[0], self.start[1], self.via_z])
        if self.part == "across":
            return np.array([pos[0], pos[1], self.via_z])
        return pos

    def _rot(self, R):
        """The rotation command that holds the start's orientation turned by ``yaw``; None without the hand's pose."""
        if self.R0 is None or R is None:
            return None
        want = rotz(self.yaw) @ self.R0
        return _capped(float(self.o["gain"]) * rotvec_of(want @ np.asarray(R, dtype=float).reshape(3, 3).T)
                       / float(self.o["out_rot"]))

    def act(self, eef, anchors, q=None, R=None) -> list:
        """The action for this step. ``eef``: the hand's position now; ``anchors``: the (N, 3) origins, or a function
        giving them (called once per leg); ``q``: the arm's seven joint angles and ``R``: the hand's rotation matrix,
        when the caller has them."""
        o = self.o
        eef = np.asarray(eef, dtype=float)
        self.trail.append(eef.copy())
        n = int(o["stuck_steps"])
        if len(self.trail) > n:
            self.trail.pop(0)
            if max(float(np.linalg.norm(p - eef)) for p in self.trail) < float(o["stuck_m"]):
                self.stuck = True  # the caller ends the sweep
        rot = self._rot(R)
        if self.retreat == 0 and self.grace == 0 and self.leg is not None and q is not None \
                and near_limit(q, float(o["limit_margin"])):
            self.retreat = int(o["retreat_steps"])  # stretched out or folded up: drop the leg, back towards the start
            self.n_retreats += 1
        if self.retreat > 0:
            self.retreat -= 1
            if float(np.linalg.norm(eef - self.p0)) <= float(o["retreat_m"]):
                self.retreat = 0
            if self.retreat == 0:
                self.left, self.grace = 0, int(o["grace_steps"])  # the next call starts the next leg
            return move_action(eef, self.p0, -1.0, 0.0, o, rot)
        self.grace = max(self.grace - 1, 0)
        if self.leg is None or self.left <= 0:
            self._begin(eef, anchors)
            rot = self._rot(R)
        while True:  # the part of the leg the hand is in: a part that is reached, or stalled, hands on to the next
            gap = float(np.linalg.norm(eef - self._goal()))
            if gap < self.best - 0.005:
                self.best, self.stall = gap, 0
            else:
                self.stall += 1
            done = gap <= float(o["tol_m"] if self.part == "down" else o["via_tol_m"])
            if self.part == "down":
                if done and not self.reached:
                    self.reached = True
                    self.n_reached += 1
                    self.left = min(self.left, int(self.leg["dwell"]) + 1)
                elif not self.reached and self.stall > int(o["stall_steps"]):
                    self.left = min(self.left, 1)  # resting on something: this step, then the next leg
                break
            if not done and self.stall <= int(o["stall_steps"]):
                break
            self._part(self.PARTS[self.PARTS.index(self.part) + 1])
        act = move_action(eef, self._goal(), self.leg["grip"], self.leg["turn"] if self.turn_left > 0 else 0.0, o, rot)
        self.left -= 1
        self.turn_left -= 1
        return act


# ---- the simulator's side (truth): the frame worker hands these to ``Sweep``
def hand(env) -> np.ndarray:
    """The hand's position (the gripper's grip site), from the simulator: readings may be offset or late in some cases."""
    s = env.sim
    return np.array(s.data.site_xpos[s.model.site_name2id("gripper0_grip_site")], dtype=float)


def hand_rot(env) -> np.ndarray:
    """The hand's rotation matrix (the grip site's, world frame), from the simulator."""
    s = env.sim
    return np.array(s.data.site_xmat[s.model.site_name2id("gripper0_grip_site")], dtype=float).reshape(3, 3)


def joints(env) -> np.ndarray:
    """The arm's seven joint angles, from the simulator."""
    s = env.sim
    return np.array([s.data.get_joint_qpos(f"robot0_joint{i}") for i in range(1, 8)], dtype=float)


def shoulder(env, o: dict | None = None) -> np.ndarray:
    s = env.sim
    base = np.array(s.data.body_xpos[s.model.body_name2id("robot0_base")], dtype=float)
    return base + np.array([0.0, 0.0, float({**SWEEP, **(o or {})}["shoulder_up_m"])])


def anchors(env) -> np.ndarray:
    """(N, 3): the origins of the scene's objects and fixtures (the table left out)."""
    inner, s = env.env, env.sim
    names = [n for n in list(inner.objects_dict) + list(inner.fixtures_dict)
             if "table" not in n and n in inner.obj_body_id]
    return np.array([s.data.body_xpos[int(inner.obj_body_id[n])] for n in names], dtype=float).reshape(-1, 3)
