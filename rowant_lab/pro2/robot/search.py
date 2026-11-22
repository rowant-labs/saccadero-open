"""Looking where the main camera does not see (2026-11-16: the case in which a panel hangs on
the camera's line of sight to the target).

``LookAgent`` is one step of a plan: the open hand goes above a point (the middle of the patch that something hides
from the main camera) and the wrist camera's colour picture is searched for a thing the main camera did not place
(``look.wrist_find``, ``wrist2d.py``). A second picture from straight above what the first found, lower down, takes the
slant out of the first. The thing becomes a percept of the instruction's target type (``look.add_found``), and the
next step plans its pick as for any other percept. The step ends ``found`` or ``not_found``; either way the next step
runs.

Policy side: numpy only.
"""

from __future__ import annotations

import numpy as np

from .follow import FollowParams, follow
from .grasp import quat_xyzw_to_mat
from .wrist2d import WRIST

NOOP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
# heights of the grip site above the support for the two pictures (the camera is 9.7 cm higher; at 0.27 m it sees
# 28 cm to each side). Higher is out of the arm's reach over the far part of the tables (a stored row: 1.25 m asked,
# 1.23 m reached, over a table at 0.90 m)
LOOK = {"h1_m": 0.27, "h2_m": 0.20, "move1": 45, "move2": 25, "tol_m": 0.008, "still": 2, "second": True}
# ``confirm`` (``params["relation_look"]``, 2026-11-16; off unless asked): a find counts only when a picture from
# straight above it shows a thing within ``same_m`` of the same place; the first picture's things are tried in its
# order, at most ``confirm_n`` of them. (The Mac depth line, a training task with a slab beside the hidden bowl: the
# slab's slanted image lay 14 cm from where the slab stands and was taken; from above it, the thing was 10 cm away.)
CONFIRM = {"confirm": False, "confirm_n": 2, "same_m": 0.06}


class LookAgent:
    """``point``: ``look.behind_point``'s record (``xy``, ``support_z``). ``typ``: the instruction's target type."""

    def __init__(self, tool, typ: str, point: dict, cap: int, fp: FollowParams = FollowParams(), opts: dict | None = None):
        self.tool, self.typ, self.cap, self.fp = tool, typ, int(cap), fp
        self.o = {**LOOK, **CONFIRM, **(opts or {})}
        self.xy = np.asarray(point["xy"], dtype=float)[:2]
        self.support_z = float(point["support_z"])
        self.t = 0
        self.finished = False
        self.finish_reason = None
        self.attempts: list = []
        self.events: list = []
        self.found = None
        self.pobs = None
        self._gen = self._run()
        next(self._gen)

    def _pose(self):
        return (np.asarray(self.pobs["robot0_eef_pos"], dtype=float), quat_xyzw_to_mat(self.pobs["robot0_eef_quat"]))

    def _goto(self, p, R, n: int):
        for _ in range(int(n)):
            ph, Rh = self._pose()
            if float(np.linalg.norm(ph - p)) <= float(self.o["tol_m"]) or self.cap - self.t <= 3:
                break
            yield follow(ph, Rh, p, R, -1.0, 1.0, 1.0, self.fp)
        for _ in range(int(self.o["still"])):  # the picture is taken with the hand at rest
            ph, Rh = self._pose()
            yield follow(ph, Rh, p, R, -1.0, 0.3, 1.0, self.fp)
        return float(np.linalg.norm(self._pose()[0] - p))

    def _run(self):
        yield None
        _p0, R0 = self._pose()  # the hand keeps the turn it has (top-down from the start pose)
        off = (R0 @ np.asarray(WRIST["t_hand_cam"], dtype=float))[:2]  # the camera, not the grip site, goes above the point
        rec = {"t": self.t, "ev": "look_behind", "point": [round(float(x), 4) for x in self.xy]}
        self.events.append(rec)
        err = yield from self._goto(np.array([*(self.xy - off), self.support_z + float(self.o["h1_m"])]), R0,
                                    self.o["move1"])
        self.tool.t_now = self.t
        got = self.tool.wrist_find(self.typ, self.support_z, hint_xy=self.xy)
        rec["first"] = None if got is None else {"xy": [round(float(x), 4) for x in got["xy"]], "err_m": round(err, 4),
                                                 "n_cands": got["n_cands"], "area_m2": round(got["area_m2"], 5)}
        if self.o["confirm"]:
            ranked = [] if got is None else (got.get("ranked") or [got])
            rec["tries"] = []
            got = None
            for c in ranked[: int(self.o["confirm_n"])]:
                if self.cap - self.t <= int(self.o["move2"]) + 5:
                    break
                xy1 = np.asarray(c["xy"], dtype=float)
                err = yield from self._goto(np.array([*(xy1 - off), self.support_z + float(self.o["h2_m"])]), R0,
                                            self.o["move2"])
                self.tool.t_now = self.t
                got2 = self.tool.wrist_find(self.typ, self.support_z, hint_xy=xy1, opts={"prefer": "near"})
                moved = None if got2 is None else float(np.linalg.norm(np.asarray(got2["xy"], dtype=float) - xy1))
                rec["tries"].append({"xy": [round(float(x), 4) for x in xy1], "err_m": round(err, 4),
                                     "moved_m": None if moved is None else round(moved, 4),
                                     "area_m2": None if got2 is None else round(float(got2["area_m2"]), 5)})
                if moved is not None and moved <= float(self.o["same_m"]):
                    got = got2  # the same thing, seen straight on
                    break
        elif got is not None and self.o["second"] and self.cap - self.t > int(self.o["move2"]) + 5:
            xy1 = np.asarray(got["xy"], dtype=float)
            err = yield from self._goto(np.array([*(xy1 - off), self.support_z + float(self.o["h2_m"])]), R0,
                                        self.o["move2"])
            self.tool.t_now = self.t
            got2 = self.tool.wrist_find(self.typ, self.support_z, hint_xy=xy1)
            rec["second"] = None if got2 is None else {"xy": [round(float(x), 4) for x in got2["xy"]],
                                                       "err_m": round(err, 4), "n_cands": got2["n_cands"],
                                                       "moved_m": round(float(np.linalg.norm(np.asarray(got2["xy"]) - xy1)), 4)}
            if got2 is not None and float(np.linalg.norm(np.asarray(got2["xy"]) - xy1)) <= 0.06:
                got = got2  # the same thing, seen straight on
        if got is None:
            self.finish_reason = "not_found"
            return
        self.found = self.tool.add_found(self.typ, got)
        rec["percept"] = self.found["id"]
        self.finish_reason = "found"

    def act(self, pobs: dict) -> list:
        self.pobs = pobs
        try:
            a = self._gen.send(pobs)
        except StopIteration:
            a = None
        if a is None:
            if not self.finished:
                self.finished = True
                self.finish_reason = self.finish_reason or "generator_end"
                self.events.append({"t": self.t, "ev": "finish", "reason": self.finish_reason})
            a = NOOP
        self.t += 1
        return [float(x) for x in a]

    def summary(self) -> dict:
        return {"finished": self.finished, "finish_reason": self.finish_reason, "steps": self.t, "attempts": self.attempts,
                "events": self.events}
