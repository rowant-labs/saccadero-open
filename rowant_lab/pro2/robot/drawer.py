"""Open a drawer: ``open the <part words> [layer|drawer] of the <fixture>``.

The fixture is a library type with parts on slide joints (named by the instruction's words when it names one,
else any such type, the perceived one first). The part is the slide part whose name words the instruction names
(``top``, ``middle``, ``bottom``; ``upper``/``lower``/``centre`` read as those). The opening direction comes from the
type's kinematics (the joint moved toward the far end of its range from rest); the handle is the part's collision
box that reaches farthest along it (the widest of those within a few mm).

The skill: point the hand horizontally at the handle (closing axis vertical), pre-close the jaws to the handle's
height plus a margin (so the fingers clear the neighbouring handles), move in straight along the opening axis,
close on the handle, pull it out the rest of the joint's range, open, back off and rise.
"""

from __future__ import annotations

import math
import re

import numpy as np

from rowant_lab.libero_env.m2_kin import mat_to_quat, part_body_poses

from . import grasp as GR
from .agent import OracleAgent
from .follow import follow
from .ground import _content, words

UPV = np.array([0.0, 0.0, 1.0])
RELOOK_FAR_M = 0.25  # ``open_relook``: a fixture found further than this from where the plan had it is not believed
RELOOK_TURN_DEG = 35.0  # nor one turned more than this from the plan's pose and from its half turn
OPEN = re.compile(r"\b(?:open|pull open|pull out|slide open|slide out)(?: up)? (?P<rest>.+)$")
SYN = {"upper": "top", "highest": "top", "topmost": "top", "lower": "bottom", "lowest": "bottom",
       "bottommost": "bottom", "center": "middle", "centre": "middle", "central": "middle", "mid": "middle"}


def _slide_parts(tdoc) -> list:
    out = []
    for pn, pd in (tdoc.get("parts") or {}).items():
        j = (pd or {}).get("joint")
        if j and j.get("type") == "slide" and (pd.get("bodies") or []):
            out.append(pn)
    return sorted(out)


def _norm(ws) -> set:
    return {SYN.get(w, w) for w in _content(ws)}


def parse_open(instruction: str, lib, seen_types=()) -> dict:
    text = " ".join(words(instruction))
    m = OPEN.search(text)
    if not m:
        return {"status": "fail", "reason": "no open verb"}
    rest = m.group("rest").split(" and ")[0]
    wset = _norm(rest.split())
    cands = []
    for t, d in (lib.export_types or {}).items():
        if _slide_parts(d or {}):
            cands.append(t)
    cands.sort()
    if not cands:
        return {"status": "fail", "reason": "no type with slide parts"}
    named = [t for t in cands if _content(t.split("_")) & wset]
    if named:
        cands = named
    seen = [t for t in cands if t in set(seen_types)]
    if seen:
        cands = seen
    typ = cands[0]
    parts = _slide_parts(lib.export_types[typ])
    toks = {p: _norm(p.split("_")) for p in parts}
    common = set.intersection(*toks.values()) if len(parts) > 1 else set()
    scored = sorted(((len((toks[p] - common) & wset), p) for p in parts), reverse=True)
    if scored[0][0] > 0 and (len(scored) == 1 or scored[1][0] < scored[0][0]):
        part = scored[0][1]
    elif len(parts) == 1:
        part = parts[0]
    else:
        return {"status": "fail", "reason": f"part of {rest!r}"}
    return {"status": "ok", "form": "open", "target_type": typ, "relation": None, "landmarks": [], "part": part,
            "predicate": "open", "place_type": None}


def _jv(percept) -> dict:
    out = {}
    for k, v in (percept.get("joints") or {}).items():
        q = v.get("q") if isinstance(v, dict) else v
        try:
            out[k] = float(q)
        except (TypeError, ValueError):
            continue
    return out


def handle_plan(lib, percept: dict, part: str):
    """{c: grip point, d: opening direction (horizontal unit), bar_h, pull, ...} from the percept and the model."""
    typ = percept["type"]
    tdoc = lib.export_types[typ]
    root = tdoc["root_body"]
    Rr = lib.quat_mat(percept["bodies"][root]["quat_wxyz"])
    tr = np.asarray(percept["bodies"][root]["pos"], float)
    pd = tdoc["parts"][part]
    j = pd["joint"]
    lo, hi = [float(x) for x in j["range"]]
    q0 = float(j.get("qpos0") or 0.0)
    far = lo if abs(lo - q0) >= abs(hi - q0) else hi
    s = 1.0 if far > q0 else -1.0
    jv = _jv(percept)
    q_now = float(np.clip(jv.get(part, q0), min(lo, hi), max(lo, hi)))
    bodies = list(pd["bodies"])
    A = part_body_poses(tdoc, {root: (Rr, tr)}, {**jv, part: q_now})
    B = part_body_poses(tdoc, {root: (Rr, tr)}, {**jv, part: q_now + s * 0.05})
    b0 = bodies[0]
    d = np.asarray(B[b0][1], float) - np.asarray(A[b0][1], float)
    d[2] = 0.0
    if float(np.linalg.norm(d)) < 0.02:
        return None
    d = d / float(np.linalg.norm(d))
    e = np.cross(UPV, d)
    m1 = lib.m1_type(typ)
    boxes = []
    for b in bodies:
        if b in (percept.get("bodies") or {}):
            Rb = lib.quat_mat(percept["bodies"][b]["quat_wxyz"])
            tb = np.asarray(percept["bodies"][b]["pos"], float)
        elif b in A:
            Rb, tb = np.asarray(A[b][0], float), np.asarray(A[b][1], float)
        else:
            continue
        for pr in ((m1.get("bodies") or {}).get(b) or {}).get("primitives") or []:
            cw = tb + Rb @ np.asarray(pr["centre"], float)
            Rw = Rb @ np.asarray(pr["R"], float)
            h = np.asarray(pr["half"], float)
            ext = lambda u: float(np.abs(Rw.T @ u) @ h)  # noqa: E731
            boxes.append({"c": cw, "s": float(cw @ d), "ed": ext(d), "ez": ext(UPV), "ee": ext(e)})
    if not boxes:
        return None
    fronts = [bx["s"] + bx["ed"] for bx in boxes]
    fmax = max(fronts)
    hand = [bx for bx, f in zip(boxes, fronts) if f >= fmax - 0.004]
    bar = max(hand, key=lambda bx: bx["ee"])
    front = bar["s"] + bar["ed"]
    grip_s = max(bar["s"], front - 0.012)
    c = bar["c"] + d * (grip_s - bar["s"])
    pull = float(np.clip(abs(far - q_now) + 0.015, 0.03, 0.25))
    # (``open_feel``) how far behind the grip point the part's own front stands: the nearest face, along the way in,
    # of a box of the part that lies wholly behind the bar and is at least as long as it (its posts are not)
    behind = [bx["s"] + bx["ed"] for bx in boxes
              if bx is not bar and bx["s"] + bx["ed"] <= bar["s"] - bar["ed"] + 0.001 and bx["ee"] >= bar["ee"] - 1e-6]
    stand = float(grip_s - max(behind)) if behind else None
    # (``open_feel``) the room above and below the bar: to the nearest box of the fixture's OTHER bodies that stands
    # out in front of this part's own front (another drawer's handle), from the bar's middle to that box's near face
    room = None
    if behind:
        try:
            for b, (Rb, tb) in A.items():
                if b in bodies:
                    continue
                Rb, tb = np.asarray(Rb, float), np.asarray(tb, float)
                for pr in ((m1.get("bodies") or {}).get(b) or {}).get("primitives") or []:
                    cw = tb + Rb @ np.asarray(pr["centre"], float)
                    Rw = Rb @ np.asarray(pr["R"], float)
                    h = np.asarray(pr["half"], float)
                    if float(cw @ d) + float(np.abs(Rw.T @ d) @ h) <= max(behind) + 0.005:
                        continue  # it does not stand out in front of the part's front
                    if abs(float((cw - bar["c"]) @ e)) > bar["ee"] + float(np.abs(Rw.T @ e) @ h):
                        continue  # not over or under the bar
                    gap_z = abs(float(cw[2] - bar["c"][2])) - float(np.abs(Rw.T @ UPV) @ h)
                    room = gap_z if room is None else min(room, gap_z)
        except Exception:
            room = None
    return {"c": c, "d": d, "bar_h": 2.0 * bar["ez"], "bar_w": 2.0 * bar["ee"], "bar_depth": 2.0 * bar["ed"],
            "pull": pull, "q_now": q_now, "far": far, "n_boxes": len(boxes), "stand_m": stand, "room_m": room}


# -- a fixture read back to front (2026-11-16 night; used only by options that are off by default)
def carcass_centre(lib, percept: dict):
    """The world xy of the middle of the fixture's carcass: the boxes of its bodies that belong to no sliding part,
    taken in the root body's frame. A box-like carcass explains the depth as well turned half a turn, and the mirror
    fit shares the carcass with the true pose, not the whole fixture's box (the handles stand out in front of it):
    seven back-to-front fits of the Goal cabinet on plain starts lay 3.4 cm from the truth, and their half turn
    about this point 0.6 mm from it. None when the type has no such boxes."""
    typ = percept["type"]
    tdoc = lib.export_types[typ]
    root = tdoc["root_body"]
    slide = {b for pn in _slide_parts(tdoc) for b in tdoc["parts"][pn]["bodies"]}
    Rr = np.asarray(lib.quat_mat(percept["bodies"][root]["quat_wxyz"]), float)
    tr = np.asarray(percept["bodies"][root]["pos"], float)
    los, his = [], []
    for b, pose in (percept.get("bodies") or {}).items():
        if b in slide:
            continue
        bx = lib.body_boxes(typ, b)
        if not bx.n:
            continue
        Rb = np.asarray(lib.quat_mat(pose["quat_wxyz"]), float)
        tb = np.asarray(pose["pos"], float)
        lo, hi = bx.placed(Rr.T @ Rb, Rr.T @ (tb - tr)).aabb()
        los.append(np.asarray(lo, float).min(0))
        his.append(np.asarray(hi, float).max(0))
    if not los:
        return None
    c = 0.5 * (np.min(los, 0) + np.max(his, 0))
    return (tr + Rr @ c)[:2]


def twin_percept(lib, percept: dict, how: str = "twin"):
    """The fixture's percept turned half a turn about the middle of its carcass (``carcass_centre``), its parts'
    joints as they are. None when it has no carcass."""
    c = carcass_centre(lib, percept)
    if c is None:
        return None
    tdoc = lib.export_types[percept["type"]]
    root = tdoc["root_body"]
    Rr = np.asarray(lib.quat_mat(percept["bodies"][root]["quat_wxyz"]), float)
    tr = np.asarray(percept["bodies"][root]["pos"], float)
    R2 = np.diag([-1.0, -1.0, 1.0]) @ Rr
    t2 = np.array([2.0 * float(c[0]) - tr[0], 2.0 * float(c[1]) - tr[1], tr[2]])
    bp = part_body_poses(tdoc, {root: (R2, t2)}, _jv(percept))
    return {**percept, "bodies": {b_: {"pos": [float(x) for x in t], "quat_wxyz": mat_to_quat(R)}
                                  for b_, (R, t) in bp.items()},
            "source": f"{percept.get('source')}+{how}"}


FACE_MIN_COS = 0.3  # a fixture is called back to front (or the right way round) only beyond this cosine


def opens_away(lib, percept: dict, hand_xy, part: str, min_cos: float = FACE_MIN_COS):
    """True when ``part`` opens away from ``hand_xy`` as seen from the carcass's middle (the fixture stands back to
    front for a hand that works from there), False when toward it, None when it cannot be told: no handle, no
    carcass, or the opening direction lies within ``min_cos`` of square to the line from the carcass to the hand (a
    fixture that opens across the hand's side says nothing; the two true mirror fits read -0.72 and -0.77, the
    Spatial cabinet as it stands +0.42, and turned 17 degrees further by a pose change +0.13)."""
    try:
        hp = handle_plan(lib, percept, part)
        c = carcass_centre(lib, percept)
    except Exception:
        return None
    if hp is None or c is None:
        return None
    v = np.asarray(hand_xy, float)[:2] - np.asarray(c, float)
    d = np.asarray(hp["d"], float)[:2]
    n = float(np.linalg.norm(v)) * float(np.linalg.norm(d))
    if n < 1e-9:
        return None
    cos = float(d @ v) / n
    return True if cos < -min_cos else (False if cos > min_cos else None)


class OpenAgent(OracleAgent):
    """Pull a drawer open by its handle (see the module doc)."""

    pre_m = 0.10
    pre_fast_m = 0.06
    high_m = 0.08
    high_fast_m = 0.05

    hp: dict = {}  # set by plan_open right after construction (the generator reads it only from the first step)

    def act(self, pobs):
        try:
            return super().act(pobs)
        except Exception as ex:  # never crash the episode from this skill
            if not self.finished:
                self._ev("agent_error", err=str(ex)[:120])
                self._finish("agent_error")
            self.t += 1
            return self._idle()

    def _hand_R(self, R0, d):
        z = -np.asarray(d, float)
        best, ang = None, None
        for y in (UPV, -UPV):
            x = np.cross(y, z)
            R = np.column_stack([x, y, z])
            a = GR.rot_angle_deg(R, R0)
            if ang is None or a < ang:
                best, ang = R, a
        return best

    def _act_to(self, p, R, g, cap=None):
        pc, Rc = self._pose()
        a = follow(pc, Rc, p, R, 1.0, self.ap.slow_cap if cap is None else cap, self.ap.rot_cap, self.fp)
        a[6] = float(g)
        return a

    def _goto_g(self, p, R, g, cap, tol_m, tol_deg, max_steps, **kw):
        gen = self._goto(p, R, 1.0, cap, tol_m, tol_deg, max_steps=max_steps, **kw)
        try:
            a = next(gen)
        except StopIteration as e:
            return e.value
        while True:
            a = list(a)
            a[6] = float(g)
            yield a
            try:
                a = next(gen)
            except StopIteration as e:
                return e.value

    def _goto_seq(self, p, R, gs, g_rest, cap, tol_m, tol_deg, max_steps, **kw):
        """``_goto`` with the gripper command taken from ``gs`` step by step and ``g_rest`` after it; every command of
        ``gs`` is sent, also when the pose is reached first."""
        gen = self._goto(p, R, 1.0, cap, tol_m, tol_deg, max_steps=max_steps, **kw)
        k, st = 0, None
        while True:
            try:
                a = next(gen)
            except StopIteration as e:
                st = e.value
                break
            a = list(a)
            a[6] = float(gs[k]) if k < len(gs) else float(g_rest)
            k += 1
            yield a
        while k < len(gs):
            yield self._act_to(p, R, gs[k])
            k += 1
        return st

    def _pull(self, p_tgt, Rh, d, pull: float, max_steps: int = 120, cap: float = 0.5, done_at: float | None = None,
              ramp: int = 0, release_early: int = 0):
        """Pull toward ``p_tgt`` and stop when the part stops coming (its joint's end): at least half the pull done and
        under 1 mm of progress along ``d`` in the last 4 steps. (``_goto`` waits out its whole no-progress window.)
        ``done_at``: the progress along ``d`` at which the pull has done its work and ends at once.

        ``open_pull_straight`` (2026-11-17, off by default; never through a delay): the command keeps its direction
        (``follow``'s ``box``). The old cap clips each axis first, so a pull that is far from its goal goes along
        the diagonal of the two axes' signs: for a part that opens along 109 degrees the hand was sent along 135 for
        six steps and then bent back (a truth replay of the training episode: the bar slid 2 cm sideways and 3 cm
        along in the fingers, the fingers ended 7 mm apart, and the drawer stopped at 13.1 cm, short of open). A
        part that opens along a table axis is pulled straight either way.

        ``release_early`` (``open_release_early``, with ``done_at``): the gripper command turns to "open" once the
        pull, at the speed it is coming, will reach ``done_at`` within that many steps (``self._n_released`` counts
        the open commands sent; the fingers begin to part three steps after the first)."""
        p0, _ = self._pose()
        prog = []
        box = bool(self.ap.open_pull_straight) and int(getattr(self, "lag_steps", 0) or 0) <= 0
        self._n_released = 0
        for _ in range(max_steps):
            p, R = self._pose()
            if float(np.linalg.norm(p - p_tgt)) <= 0.006:
                return "reached"
            prog.append(float((p - p0) @ d))
            if done_at is not None and prog[-1] >= done_at:
                return "reached"
            if release_early > 0 and done_at is not None and not self._n_released and len(prog) >= 3:
                v = prog[-1] - prog[-2]  # a steady pull only: the part is coming as fast as the step before
                if v > 0.004 and abs(v - (prog[-2] - prog[-3])) < 0.002 and done_at - prog[-1] <= release_early * v:
                    self._n_released = 1
            elif self._n_released:
                self._n_released += 1
            if len(prog) > 4 and prog[-1] >= 0.5 * pull and prog[-1] - prog[-5] < 0.001:
                return "stalled"
            if self.ap.open_slip_check and len(prog) > 15 and prog[-1] < 0.01:
                # (2026-11-17, with ``open_slip_check``) a pull that has not come a centimetre in 15 steps is not
                # going to: the hand holds something that does not slide (the Mac: 120 steps spent on one)
                return "stuck"
            cap_now = cap if len(prog) > ramp else 0.3 * cap  # (``ramp``: the first steps gently; 0 as before)
            a = follow(p, R, p_tgt, Rh, 1.0, cap_now, self.ap.rot_cap, self.fp, box=True) if box \
                else follow(p, R, p_tgt, Rh, 1.0, cap_now, self.ap.rot_cap, self.fp)
            a[6] = -1.0 if self._n_released else 1.0
            yield a
        return "unreachable"

    # -- 2026-11-17: the opening step's own steps (``open_pre_pass``, ``open_close_early_m``; off by default)
    _n_released = 0  # ``open_release_early``: open commands the pull itself sent
    _brake_kz = None  # ``open_pre_brake_kz``: the vertical brake of the move that is running (None: the agent's own)

    def _braked(self, p_tgt):
        kz = self._brake_kz
        if kz is None:
            return super()._braked(p_tgt)
        v = getattr(self, "_v_step", None)
        if v is None or int(getattr(self, "lag_steps", 0) or 0) > 0:
            return p_tgt
        k = float(self.ap.brake_k)
        return np.asarray(p_tgt, dtype=float) - np.array([k * float(v[0]), k * float(v[1]), float(kz) * float(v[2])])

    def _line_off(self, cg, d):
        """(how far before the grip point the hand stands along the opening direction, how far off the line in to
        it in height, and along the bar), m."""
        p, _R = self._pose()
        off = np.asarray(p, float) - np.asarray(cg, float)
        d = np.asarray(d, float)
        return float(off @ d), abs(float(off[2])), abs(float(off @ np.cross(UPV, d)))

    def _pass_pre(self, gen, n_shape: int, cg, d, Rh, s_max: float):
        """``open_pre_pass``: run the move to the point before the handle (a generator of actions) until the hand
        is turned to within ``open_pre_deg`` and stands on the line in to the grip point: within
        ``open_pre_lat_m`` of it in height (the jaws go in 1.7 cm clear of the bar above and below), within twice
        that along the bar (the bar is 9 cm long), 2.5 cm or more before the grip point (and not beyond
        ``s_max``), with the gripper's shaping (``n_shape`` commands) sent. "passed", or the move's own result
        when it ends first."""
        ap = self.ap
        t0, k = self.t, 0
        while True:
            _p, R = self._pose()
            s_, dz, dx = self._line_off(cg, d)
            r = float(GR.rot_angle_deg(R, Rh))
            if k >= n_shape and r <= ap.open_pre_deg and dz <= ap.open_pre_lat_m and dx <= 2.0 * ap.open_pre_lat_m \
                    and 0.025 <= s_ <= s_max:
                gen.close()
                self._ev("pass", t0=t0, s=round(s_, 4), dz=round(dz, 4), dx=round(dx, 4), deg=round(r, 1))
                return "passed"
            try:
                a = next(gen)
            except StopIteration as e:
                return e.value
            k += 1
            yield a

    def _out_first(self, gen, n_shape: int, cg, d, s_min: float):
        """``open_out_first``: run the move out (a generator of actions) until the hand is ``s_min`` before the
        grip point along the opening direction, with the gripper's shaping sent. "out", or the move's own result."""
        t0, k = self.t, 0
        while True:
            s_ = self._line_off(cg, d)[0]
            if k >= n_shape and s_ >= s_min:
                gen.close()
                self._ev("out", t0=t0, s=round(s_, 4))
                return "out"
            try:
                a = next(gen)
            except StopIteration as e:
                return e.value
            k += 1
            yield a

    def _holding(self, gaps: list, bar_h: float) -> bool:
        """The fingers have stopped on something of the bar's height: the gap unchanged over the last step and
        within 4 mm of it."""
        return len(gaps) >= 2 and abs(gaps[-1] - gaps[-2]) < 0.0005 and abs(gaps[-1] - float(bar_h)) <= 0.004

    def _close_in(self, gen, cg, d, Rh, bar_h: float, att: dict):
        """``open_close_early_m``: run the move in to the grip point (a generator of actions) and turn its gripper
        command to "close" once the hand is within that distance of the grip point along the opening direction,
        within 8 mm of the line and turned to within 5 degrees. The move ends at once when the fingers hold the
        bar. Returns (the move's result or "held", the gaps read since the closing began)."""
        ap = self.ap
        closing, gaps = False, []
        while True:
            _p, R = self._pose()
            if not closing:
                s_, dz, dx = self._line_off(cg, d)
                if s_ <= ap.open_close_early_m and dz <= 0.008 and dx <= 0.02 \
                        and float(GR.rot_angle_deg(R, Rh)) <= 5.0:
                    closing = True
                    att["close_early"] = round(s_, 4)
            if closing:
                gaps.append(self._gap())
                if self._holding(gaps, bar_h):
                    gen.close()
                    return "held", gaps
            try:
                a = next(gen)
            except StopIteration as e:
                return e.value, gaps
            if closing:
                a = list(a)
                a[6] = 1.0
            yield a

    # -- 2026-11-16 night: options that are off by default (``open_gap_check``, ``open_settle``, ``open_relook``)
    fx_used = None  # ``open_relook``: the fixture's percept as the opening ended up using it (None: as planned)
    _look_dry = False  # ``open_relook``: the last look gave no new plan
    _verified = False  # ``open_verify``: the episode was seen to go on after an "opened" (once)

    def _full_travel(self, hp: dict) -> float:
        """``open_pull_full``: the part's whole travel (from rest to its far end) plus the 1.5 cm ``handle_plan``
        adds; the plan's own pull when the type's joint cannot be read."""
        try:
            j = self.lib.export_types[self.ttype]["parts"][self.spec["part"]]["joint"]
            lo, hi = [float(x) for x in j["range"]]
            q0 = float(j.get("qpos0") or 0.0)
            far = lo if abs(lo - q0) >= abs(hi - q0) else hi
            return float(np.clip(abs(far - q0) + 0.015, 0.03, 0.25))
        except Exception:
            return float(hp["pull"])

    def _far(self, p_tgt, far_m: float = 0.10) -> bool:
        p, _ = self._pose()
        return float(np.linalg.norm(p - np.asarray(p_tgt, float))) > far_m

    def _feel(self, cg, Rh, d, hp, more_m: float = 0.0):
        """``open_feel``: from the planned grip point, in along the part's way until the hand stops coming (or
        ``open_feel_in_m`` past the point), then back by the model's front-to-bar distance less the fingertip's
        length (when it stopped), or back to the planned point (when nothing stopped it). Returns ``(record, the
        point the fingers close at)``. The jaws are not commanded here (they come shaped by ``_feel_k`` and stay so)."""
        cg, d = np.asarray(cg, float), np.asarray(d, float)
        # ``more_m``: after a pull that lost the bar the part stands somewhere between shut and where the hand let
        # go: the way in is that much longer, and faster (the stop is judged against the speed)
        max_in = float(self.ap.open_feel_in_m) + max(0.0, float(more_m))
        cap = 0.6 if more_m > 0.03 else 0.3
        tgt = cg - d * max_in
        p, _ = self._pose()
        s = [float((cg - p) @ d)]  # how far past the planned grip point the hand stands, along the way in
        stalled = False
        for _ in range(60):
            yield self._act_to(tgt, Rh, 0.0, cap=cap)
            p, _ = self._pose()
            s.append(float((cg - p) @ d))
            if len(s) > 4 and s[-1] - s[-4] < 0.005 * cap:
                stalled = True  # it has stopped coming: the fingertips stand on something
                break
            if s[-1] >= max_in - 0.002:
                break
        went = s[-1]
        tip = abs(float(getattr(self.jaw, "finger_z", (-0.0093, 0.0))[0]))
        stand = 0.024 if hp.get("stand_m") is None else float(hp["stand_m"])
        rec = {"in_m": round(went, 4), "stalled": bool(stalled)}
        k = 1.0
        if stalled:
            # how deep the fixture is, by the touch: its front as the fingertips found it over its front as the model
            # has it, both from the fixture's root along the part's way. The bar stands before the front by the
            # model's distance times that ratio (a cabinet drawn shallower has its bar nearer its front: 2026-11-17,
            # Mac, the case that draws it taller: the front felt 2.2 cm deeper than the model's, 0.74 of its depth;
            # with the model's own 1.4 cm the fingers took the bar by its front edge and lost it as the pull began)
            try:
                tdoc = self.lib.export_types[self.ttype]
                root = next(q for q in self.oracle.percepts if q.get("id") == self.target)["bodies"][tdoc["root_body"]]
                s_root = float(np.asarray(root["pos"], float) @ d)
                f0 = float(cg @ d) - stand - s_root
                f1 = float(p @ d) - tip - s_root
                if f0 > 0.03 and 0.6 <= f1 / f0 <= 1.4:
                    k = f1 / f0
                    rec["depth_ratio"] = round(k, 3)
            except Exception:
                k = 1.0
        back = float(np.clip(stand * k - tip, 0.003, 0.03)) if stalled else went
        c2 = cg - d * (went - back)
        for _ in range(12):
            p, _ = self._pose()
            if float(np.linalg.norm(p - c2)) < 0.003:
                break
            yield self._act_to(c2, Rh, 0.0, cap=0.5)
        return {**rec, "back_m": round(back, 4), "grip": [round(float(x), 4) for x in c2]}, c2

    def _feel_k(self, hp: dict) -> int:
        """``open_feel``: how many closing steps shape the jaws for the touch: 4.5 cm wider than the bar is high
        (2.2 cm of room above and below it for a height the look did not give to the centimetre), and no wider than
        the fingers' own thickness lets them pass the handles above and below (``handle_plan``'s ``room_m``)."""
        open_gap = 2.0 * self.jaw.L
        bar = float(hp["bar_h"])
        gap = bar + 0.045
        if hp.get("room_m") is not None:
            thick = float(getattr(self.jaw, "finger_thick", 0.027))
            gap = min(gap, 2.0 * (float(hp["room_m"]) - thick - 0.003))
        gap = float(np.clip(gap, bar + 0.02, open_gap))
        return int(np.clip(round(8.0 * (1.0 - gap / max(open_gap, 1e-3))), 0, 8))

    def _measured(self) -> bool:
        """The fixture's percept is one a look measured (an accepted fit), not a remembered place, a rejected fit or a
        region (``open_feel_first``)."""
        try:
            src = str(next(q for q in self.oracle.percepts if q.get("id") == self.target).get("source") or "")
        except Exception:
            return True
        return not src.startswith(("memory", "rejected", "blob", "reopen:rejected", "reopen:twin"))

    def _reopen(self, att: dict, p_hold, Rh):
        """``open_gap_check``: the jaws came out too far shut to take the bar (2026-11-16, late commands and late
        readings: the shaping's three closing steps ran on while its open ones were still under way, the jaws met,
        and the first attempt closed on nothing in every S20 and D12 episode of the opening task; the attempts that
        then went in open took the bar). Open, wait as long as the jaws take plus the loop's lag, and go in open."""
        lag = int(getattr(self, "lag_steps", 0) or 0)
        if self.ap.open_reopen_cmd and lag > 0:
            # (2026-11-17, off by default) only the commands' own delay: the jaws are open by then, read or not
            obs = int(getattr(self, "lag_obs_steps", 0) or 0)
            lag = max(0, lag - obs - 1) if obs > 0 else lag
        n = 8 + lag
        for _ in range(n):
            yield self._act_to(p_hold, Rh, -1.0)
        att["reopened"] = round(self._gap(), 4)

    def _look_again(self, att: dict, p_from, Rh, d, failed: list, unreachable: bool = False):
        """``open_relook``: after an attempt that found no handle, one look at the fixture, and the handle planned
        again from it. Returns the new plan, or None when the look has nothing new (the next attempt then varies
        the height as before).

        The hand first rises and backs off from the fixture's face (the picture's way), then one full look that
        changes no percept. The candidates, in order: the look's accepted percepts of the fixture's type, then its
        rejected fits (an inlier share of 0.5 at least), each by its agreeing points; and, after an attempt the hand
        could not reach at all, the half-turn twin of the percept the plan was made on (when that twin opens toward
        where the hand began the episode). A candidate of the look that opens away from there is turned half a turn
        about its carcass first (``twin_percept``). One
        is taken when it stands within ``RELOOK_FAR_M`` of the planned fixture, is not turned from it (within
        ``RELOOK_TURN_DEG`` of the plan's opening direction or of its opposite) and its grip point is not one that
        already failed (within 1.5 cm and 10 degrees). The tool's percept of the fixture takes its pose, so the
        steps after this one plan on it too. (Mac, depth line, the first run: the look after the scene had moved the
        cabinet 13 cm gave one accepted fit 20 cm off and a quarter turn round; it was taken and could not be
        reached. Hence the turn limit.)"""
        tool, lib = self.oracle, self.lib
        part = self.spec.get("part")
        old = next((q for q in getattr(tool, "percepts", None) or [] if q.get("id") == self.target), None)
        if old is None or part is None:
            return None
        rec = {"t": self.t, "unreachable": bool(unreachable)}
        att["relook"] = rec
        root = lib.export_types[old["type"]]["root_body"]
        t_old = np.asarray(old["bodies"][root]["pos"], float)
        h0 = getattr(tool, "_hand0", None)  # where the hand began the episode, when the tool kept it; else this step
        hand0 = np.asarray(h0 if h0 is not None else self._hand0, float)[:2]
        cands = []
        if hasattr(tool, "full_look") and tool.n_full + tool.n_track < tool.budget and not self._look_dry:
            if p_from is not None:
                p_look = np.asarray(p_from, float) + np.asarray(d, float) * 0.06 + UPV * 0.10
                rec["back"] = yield from self._goto(p_look, Rh, -1.0, self.ap.transit_cap, 0.03, 20.0, max_steps=25)
            yield from self._settle(-1.0)
            try:
                rep = tool.full_look(merge=True, only=())
            except Exception as ex:
                rep = None
                rec["look_error"] = str(ex)[:80]
            if rep is not None:
                def key(q):
                    f = q.get("fit") or {}
                    return float(f.get("n_agree") or 0.0) * float(f.get("inlier") or 0.0)

                fresh = [q for q in rep.get("percepts") or [] if q.get("type") == old["type"]
                         and root in (q.get("bodies") or {})]
                rej = [q for q in ((rep.get("extra") or {}).get("rejected") or []) if q.get("type") == old["type"]
                       and root in (q.get("bodies") or {}) and float((q.get("fit") or {}).get("inlier") or 0.0) >= 0.5]
                cands += [("seen", q) for q in sorted(fresh, key=key, reverse=True)]
                cands += [("rejected", q) for q in sorted(rej, key=key, reverse=True)]
                rec["seen"], rec["rejected"] = len(fresh), len(rej)
                try:  # the record only: the look's unexplained regions that stand a hand high or more (numbers)
                    rec["regions"] = [{"top_xy": [round(float(x), 3) for x in (b_.get("top") or {}).get("xy") or []],
                                       "top_half": [round(float(x), 3) for x in (b_.get("top") or {}).get("half_xy") or []],
                                       "top_z": (b_.get("top") or {}).get("z"), "n_px": b_.get("n_px"),
                                       "top_px": (b_.get("top") or {}).get("n_px")}
                                      for b_ in ((rep.get("extra") or {}).get("blobs") or [])
                                      if float(b_.get("height") or 0.0) >= 0.08][:8]
                    rec["rejected_all"] = [{"xy": [round(float(x), 3) for x in (q.get("pos") or q["bodies"][root]["pos"])[:2]],
                                            "inlier": round(float((q.get("fit") or {}).get("inlier") or 0.0), 2),
                                            "n_agree": (q.get("fit") or {}).get("n_agree")}
                                           for q in ((rep.get("extra") or {}).get("rejected") or [])
                                           if q.get("type") == old["type"]][:6]
                except Exception:
                    pass
        if unreachable:
            cands.append(("twin", twin_percept(lib, old, "twin")))
        try:
            d_old = np.asarray(handle_plan(lib, old, part)["d"], float)
        except Exception:
            d_old = np.asarray(d, float)
        rec["cands"] = []
        for how, q in cands:
            if q is None:
                continue
            note = {"how": how}
            rec["cands"].append(note)
            try:
                f_ = q.get("fit") or {}
                note.update(xy=[round(float(x), 3) for x in q["bodies"][root]["pos"][:2]],
                            n_agree=f_.get("n_agree"), inlier=round(float(f_.get("inlier") or 0.0), 2))
                away = opens_away(lib, q, hand0, part)
                if how == "twin" and away is not False:
                    note["no"] = "does not face the hand"
                    continue  # the plan faced the hand as it stood (or stood across): its twin would not
                if how != "twin" and away:
                    q, how = twin_percept(lib, q, "faced"), how + "+faced"
                hp = handle_plan(lib, q, part) if q is not None else None
            except Exception:
                hp = None
            if hp is None:
                note["no"] = "no handle"
                continue
            tq = np.asarray(q["bodies"][root]["pos"], float)
            cn, dn = np.asarray(hp["c"], float), np.asarray(hp["d"], float)
            note["dir_deg"] = round(math.degrees(math.atan2(float(dn[1]), float(dn[0]))), 1)
            if float(np.linalg.norm(tq[:2] - t_old[:2])) > RELOOK_FAR_M:
                note["no"] = "too far"
                continue
            # a fixture does not turn (a quarter-turn fit of a box-like carcass is the search's other mistake): the
            # opening direction stays within RELOOK_TURN_DEG of the plan's, or of its half turn
            if abs(float(dn @ d_old)) < math.cos(math.radians(RELOOK_TURN_DEG)):
                note["no"] = "turned"
                continue
            if any(float(np.linalg.norm(cn - cf)) < 0.015 and float(dn @ df) > math.cos(math.radians(10.0))
                   for cf, df in failed):
                note["no"] = "failed before"
                continue
            new = {**old, "bodies": q["bodies"], "joints": q.get("joints") or old.get("joints"),
                   "fit": q.get("fit") or old.get("fit"), "source": f"reopen:{how}"}
            for k_, x in enumerate(tool.percepts):
                if x.get("id") == self.target:
                    tool.percepts[k_] = new
            if hasattr(tool, "_hold_open"):
                tool._hold_open()
            self.fx_used = new
            rec.update(took=how, moved_m=round(float(np.linalg.norm(tq - t_old)), 4),
                       grip=[round(float(x), 4) for x in cn], dir=[round(float(x), 3) for x in dn])
            return hp
        rec["took"] = None
        if "seen" in rec:
            self._look_dry = True  # a look that gave nothing new is not asked for again (the scene has not changed)
        return None

    def _episode(self):
        yield None
        ap, hp = self.ap, self.hp
        d = np.asarray(hp["d"], float)
        c = np.asarray(hp["c"], float).copy()
        _p0, R0 = self._pose()
        Rh = self._hand_R(R0, d)
        open_gap = 2.0 * self.jaw.L
        tgt_gap = float(np.clip(hp["bar_h"] + 0.03, 0.03, 0.06))
        k_close = int(np.clip(round(8.0 * (1.0 - tgt_gap / max(open_gap, 1e-3))), 0, 8))
        fast = bool(ap.open_fast)
        dzs = (0.0, -0.008, 0.008)
        j = 0  # the next height offset (it starts again when ``open_relook`` takes a new plan)
        failed: list = []  # ``open_relook``: (grip point, direction) of the attempts that found no handle
        self._hand0 = np.asarray(_p0, float)[:2].copy()  # where the hand began this step
        no_lag = int(getattr(self, "lag_steps", 0) or 0) <= 0
        feel_next = bool(ap.open_feel_first) and fast and no_lag and not self._measured()  # ``open_feel``
        feel_more = 0.0  # ``open_feel``: how far the part may stand behind the next plan's grip point (after a slip)
        gentle = False  # ``open_feel``: the pull after a grip found by touch, or after a slip, starts gently
        for i in range(len(dzs) + (1 if ap.open_relook else 0)):
            dz = dzs[min(j, len(dzs) - 1)]
            j += 1
            if self.cap - self.t < (45 if fast else 60):
                self._finish("no_budget")
                return
            cg = c + np.array([0.0, 0.0, dz])
            no_lag = int(getattr(self, "lag_steps", 0) or 0) <= 0
            bkw = {"straight": True, "rot_box": True} if (ap.open_box and no_lag) else {}
            gaps_in: list = []
            pre_m = self.pre_fast_m if fast else self.pre_m
            p_pre = cg + d * pre_m
            p_hi = p_pre + np.array([0.0, 0.0, self.high_fast_m if fast else self.high_m])
            att = {"index": i, "t_start": self.t, "skill": "open", "grip": [round(float(x), 4) for x in cg],
                   "dir": [round(float(x), 3) for x in d], "bar_h": round(float(hp["bar_h"]), 4),
                   "pull": round(float(hp["pull"]), 4), "k_close": k_close}
            self.attempts.append(att)
            feel = feel_next and fast and no_lag
            if fast:
                # by the high point without stopping there (it keeps the hand above what stands in front of the
                # fixture, and above an object the step before set on it), the gripper shaped on the way
                shape = [-1.0] * 3 + [1.0] * k_close
                if feel:  # ``open_feel``: the jaws wider (the bar's height is not known to the centimetre)
                    shape = [-1.0] * 3 + [1.0] * self._feel_k(hp)
                p_n0, _ = self._pose()
                pkw = bkw  # the caps of the move to the point before the handle
                if i > 0:
                    att["hi"] = "skipped"
                elif ap.open_out_first and no_lag and p_n0[2] > p_hi[2] + 0.02 \
                        and float(np.linalg.norm((p_n0 - p_pre)[:2])) < 0.12:
                    # out at the height it has, then down (see the option's note)
                    att["hi"] = yield from self._out_first(
                        self._goto_seq(np.array([p_hi[0], p_hi[1], p_n0[2]]), Rh, shape, 0.0, ap.transit_cap, 0.02,
                                       35.0, 90, **bkw), len(shape), cg, d, 0.03)
                    shape = []
                    pkw = {"each": True, "rot_box": bool(ap.open_box)}
                    if ap.open_pre_out_m > 0 and ap.open_pre_pass:  # a nearer point, on this way down only
                        pre_m = float(ap.open_pre_out_m)
                        p_pre = cg + d * pre_m
                else:
                    att["hi"] = yield from self._goto_seq(p_hi, Rh, shape, 0.0, ap.transit_cap, 0.06, 35.0, 90, **bkw)
                    shape = []
                    if ap.open_relook and att["hi"] == "unreachable" and self._far(p_hi):
                        # the hand cannot get near the handle where the plan has it: no reach for the grip point,
                        # no close on air; look again (or turn the fixture round) at once
                        att["outcome"] = "unreachable"
                        failed.append((cg.copy(), d.copy()))
                        new = yield from self._look_again(att, None, Rh, d, failed, unreachable=True)
                        if new is not None:
                            hp, j = new, 0
                            d, c = np.asarray(hp["d"], float), np.asarray(hp["c"], float).copy()
                            Rh = self._hand_R(self._pose()[1], d)
                        continue
                if ap.open_pre_brake_kz > 0 and no_lag:
                    self._brake_kz = float(ap.open_pre_brake_kz)
                try:
                    if ap.open_pre_pass and no_lag:
                        st = yield from self._pass_pre(
                            self._goto_seq(p_pre, Rh, shape, 0.0, ap.transit_cap, 0.006, 4.0, 90, **pkw),
                            len(shape), cg, d, Rh, pre_m + 0.04)
                    else:
                        st = yield from self._goto_seq(p_pre, Rh, shape, 0.0, ap.transit_cap, 0.006, 4.0, 90, **pkw)
                finally:
                    self._brake_kz = None
                att["pre_gap"] = round(self._gap(), 4)
                att["pre"] = st
                att["t_pre"] = self.t
                if ap.open_gap_check and att["pre_gap"] < float(hp["bar_h"]) + 0.012:
                    yield from self._reopen(att, p_pre, Rh)
                # (``open_close_early_m`` gives way to ``open_feel`` on an attempt that finds the part's front by
                # touch: the jaws go in open, wider than usual, and close where the touch puts them)
                if ap.open_close_early_m > 0 and no_lag and not feel:
                    st, gaps_in = yield from self._close_in(self._goto_g(cg, Rh, 0.0, 0.7, 0.004, 4.0, 40), cg, d, Rh,
                                                            float(hp["bar_h"]), att)
                else:
                    st = yield from self._goto_g(cg, Rh, 0.0, 0.7, 0.004, 4.0, 40)
                if feel:
                    att["feel"], cg = yield from self._feel(cg, Rh, d, hp, feel_more)
                    feel_more, gentle = 0.0, True
            else:
                st = yield from self._goto(p_hi, Rh, -1.0, ap.transit_cap, 0.01, 6.0, max_steps=90)
                att["hi"] = st
                for _ in range(3):
                    yield self._act_to(p_hi, Rh, -1.0)
                for _ in range(k_close):
                    yield self._act_to(p_hi, Rh, 1.0)
                for _ in range(4):
                    yield self._act_to(p_hi, Rh, 0.0)
                att["pre_gap"] = round(self._gap(), 4)
                if ap.open_gap_check and att["pre_gap"] < float(hp["bar_h"]) + 0.012:
                    yield from self._reopen(att, p_hi, Rh)
                st = yield from self._goto_g(p_pre, Rh, 0.0, ap.transit_cap, 0.006, 4.0, 60)
                att["pre"] = st
                st = yield from self._goto_g(cg, Rh, 0.0, 0.5, 0.004, 4.0, 50)
            p_a, _ = self._pose()
            att["approach"] = st
            att["approach_err"] = round(float(np.linalg.norm(p_a - cg)), 4)
            att["t_in"] = self.t
            if st not in ("reached", "held"):
                # the record only (2026-11-16): where the hand stood from the grip point and the arm's joints as read.
                # (Colour line, the plain drawer task: the approach ended 1.6 to 4 cm short in 24 of 268, 23 of them
                # after a staging by the far rim, in the default arm too, the hand asked for its full cap for 18 steps
                # and not coming; what holds it there the rows did not say)
                try:
                    att["approach_off"] = [round(float(x), 4) for x in p_a - cg]
                    q = self.pobs.get("robot0_joint_pos") if isinstance(self.pobs, dict) else None
                    if q is not None:
                        att["approach_q"] = [round(float(x), 3) for x in np.asarray(q, float).ravel()]
                except Exception:
                    pass
            if ap.open_settle:
                # through a delay only: the approach ended where the predicted hand says; wait the lag out, read where
                # the hand truly is and close what is left (``_settle_fix``; nothing without a delay)
                fx = yield from self._settle_fix(cg, Rh, 0.0, 0.5, 0.004, 4.0)
                if fx is not None:
                    att["settle_fix"] = fx
            # ``open_close_early_m``: the hold is made once the gap has stopped at the bar's height. Not on an attempt
            # that found the part by touch (``open_feel``: the bar's height is then not the model's to say), and
            # never through a delay (where ``_close_wait`` is the delay stream's)
            quick = ap.open_close_early_m > 0 and no_lag and not feel
            # (the move in read the gap at this very step when it was closing: that reading is not taken twice)
            gaps = list(gaps_in) if (quick and gaps_in) else [self._gap()]
            wait = self._close_wait()  # (2026-11-17: 0 unless the delay compensator's closing hold was shortened)
            if not (quick and (st == "held" or self._holding(gaps, hp["bar_h"]))):
                for k in range(15 + wait):
                    yield self._act_to(cg, Rh, 1.0)
                    gaps.append(self._gap())
                    if quick and self._holding(gaps, hp["bar_h"]):
                        break
                    if k >= 3 + wait and abs(gaps[-4] - gaps[-1]) < 0.0005:
                        break
            gap = gaps[-1]
            att["gap"] = round(gap, 4)
            att["t_closed"] = self.t
            if gap < 0.003:
                att["outcome"] = "missed"
                feel_next = bool(ap.open_feel) and fast and no_lag
                for _ in range(5):
                    yield self._act_to(cg, Rh, -1.0)
                yield from self._goto(p_pre, Rh, -1.0, ap.transit_cap, 0.01, 8.0, max_steps=30)
                if ap.open_relook:
                    # the fingers met on nothing: the handle is not where the plan has it. Not the same grip again
                    # without a new look at the fixture (2026-11-16: a cabinet the scene moved 13 cm at step 32 was
                    # reached for three times at its old place, in every run)
                    failed.append((cg.copy(), d.copy()))
                    new = yield from self._look_again(att, p_pre, Rh, d, failed,
                                                      unreachable=st not in ("reached", "held") and self._far(cg))
                    if new is not None:
                        hp, j = new, 0
                        d, c = np.asarray(hp["d"], float), np.asarray(hp["c"], float).copy()
                        Rh = self._hand_R(self._pose()[1], d)
                continue
            p_s, _ = self._pose()
            if ap.pull_stall_stop:
                # ``pull_end_early``: the pull is the part's travel plus 1.5 cm (``handle_plan``); 2 cm short of it
                # the part is within 5 mm of its end
                reach = float(hp["pull"])
                if ap.open_pull_full:
                    # the hand pulls the part's WHOLE travel from where it took the bar, whatever the look read of
                    # the joint (a drawer read 3 cm open that was shut was pulled 12.7 cm and left short of open);
                    # a part that was out already stops at its end and the stall rule ends the pull there
                    reach = max(reach, self._full_travel(hp))
                    att["reach"] = round(reach, 4)
                early = reach - 0.02 if (ap.pull_end_early and reach > 0.06) else None
                if gentle:  # at half the speed, its first four steps at a third of that
                    att["gentle"] = True
                st = yield from self._pull(cg + d * reach, Rh, d, float(hp["pull"]),
                                           cap=0.5 if gentle else (1.0 if fast else 0.5), done_at=early,
                                           ramp=4 if gentle else 0,
                                           # (``open_release_early`` gives way on a gentle pull: its end is felt for)
                                           release_early=int(ap.open_release_early) if (no_lag and not gentle) else 0)
            else:
                st = yield from self._goto(cg + d * float(hp["pull"]), Rh, 1.0, 0.5, 0.006, 8.0, max_steps=120)
            p_e, _ = self._pose()
            moved = float((p_e - p_s) @ d)
            gap_e = self._gap()
            att["pull_status"] = st
            att["moved"] = round(moved, 4)
            att["gap_end"] = round(gap_e, 4)
            att["t_pulled"] = self.t
            n_sent = int(self._n_released) if ap.pull_stall_stop else 0  # open commands the pull itself sent
            if n_sent:
                att["released_early"] = n_sent
            # ``take_turn_early``: a pull that held the bar to its end, and a step to follow that starts from the
            # pose the staging let go in: the turn back begins with the backing off
            turn = bool(ap.take_turn_early and ap.open_retreat_pass_m > 0 and no_lag and getattr(self, "more_follow", False)
                        and getattr(self, "next_R", None) is not None and gap_e > 0.002
                        and moved >= 0.6 * float(hp["pull"]) and gap_e >= float(hp["bar_h"]) - 0.004)
            # (and when the pull sent two open commands or more, the hand does not stand still for the rest of
            # them: it backs off along the pull's own way, the part at its end, while the jaws part)
            for _ in range(0 if (turn and n_sent >= 2) else max(0, (4 if fast else 6) - n_sent)):
                yield self._act_to(p_e, Rh, -1.0)
            if ap.open_retreat_pass_m > 0 and no_lag:
                # out along the opening direction only until the jaws are clear of the bar
                t0r = self.t
                R_out = Rh
                if turn:
                    R_out = np.asarray(self.next_R, float)
                    att["turn_early"] = True
                for _ in range(12):
                    p_n, R_n = self._pose()
                    if float((p_n - p_e) @ d) >= float(ap.open_retreat_pass_m):
                        break
                    a = follow(p_n, R_n, p_e + d * (float(ap.open_retreat_pass_m) + 0.03), R_out, -1.0, ap.transit_cap,
                               ap.rot_cap, self.fp)
                    a[6] = -1.0
                    yield a
                self._ev("retreat", t0=t0r, out=round(float((self._pose()[0] - p_e) @ d), 4))
            else:
                yield from self._goto(p_e + d * 0.05, Rh, -1.0, ap.transit_cap, 0.01, 8.0, max_steps=25)
            if not ap.pull_stall_stop:  # with it another step follows, and that step makes its own way up
                p_b, _ = self._pose()
                yield from self._goto(p_b + np.array([0.0, 0.0, 0.08]), Rh, -1.0, ap.transit_cap, 0.01, 8.0,
                                      max_steps=25)
            att["t_end"] = self.t
            # ``open_slip_check`` (off by default): fingers that end the pull well under the bar's height apart are
            # losing it (7 mm for a 16 mm bar in the training episode whose drawer stopped at 13.1 cm): not opened
            bar_ref = float(hp["bar_h"])
            if ap.open_feel:
                # (2026-11-17, with ``open_feel``) the bar is as thick as the fingers found it at the close when that
                # is less than the model has it (a fixture drawn at another size: a cabinet a fifth taller by its
                # top had a bar of 1.4 cm where the model drawn to that height says 2.0)
                bar_ref = min(bar_ref, float(gap))
            slipping = bool(ap.open_slip_check) and gap_e < bar_ref - 0.004
            if gap_e > 0.002 and moved >= 0.6 * float(hp["pull"]) and not slipping:
                att["outcome"] = "opened"
                if not (ap.open_verify and not self._verified and not getattr(self, "more_follow", False)
                        and self.cap - self.t > 60):
                    self._finish("opened")
                    return
                # ``open_verify`` (off by default; only when nothing follows this step in the instruction): the
                # benchmark ends an episode the moment its task is done, so an episode that goes on three steps
                # after the pull has a part that is not as far out as the task wants (13.1 cm of 16 in the
                # training episode, "opened" by the hand's travel). Once: the part is looked at again and taken
                # for the rest of its way.
                self._verified = True
                p_w, _ = self._pose()
                for _ in range(3):
                    yield self._act_to(p_w, Rh, -1.0)
                att["outcome"] = "not open"
            else:
                att["outcome"] = "slipped"
                if ap.open_feel and fast and no_lag:
                    # the bar left the fingers somewhere on the way: the next attempt finds the part's front by touch
                    # from where the plan will have it (the hand's travel) back to where it was taken, and pulls gently
                    feel_next, gentle, feel_more = True, True, max(0.0, moved - 0.02)
            new = None
            if ap.open_relook and (ap.open_slip_check or att["outcome"] == "not open"):
                # where the part stands now is the look's to say (its joint as the fit reads it), not the hand's
                # travel: the bar left the fingers somewhere on the way (the hand went 15.7 cm, the drawer 5 and 13)
                p_n, _ = self._pose()
                new = yield from self._look_again(att, p_n, Rh, d, failed)
            if new is not None:
                hp, j = new, 0
                d, c = np.asarray(hp["d"], float), np.asarray(hp["c"], float).copy()
                Rh = self._hand_R(self._pose()[1], d)
                continue
            c = c + d * max(0.0, moved - 0.02)
            hp = {**hp, "pull": float(max(0.03, float(hp["pull"]) - max(0.0, moved - 0.02)))}
        self._finish("max_attempts")


def plan_open(instruction: str, tool, lib, cap: int, params: dict):
    """{agent, parsed, grounding} for an open-drawer instruction, or None when it is not one."""
    from . import make_params

    seen = [p.get("type") for p in tool.percepts]
    parsed = parse_open(instruction, lib, seen)
    if parsed.get("status") != "ok":
        return None
    typ = parsed["target_type"]
    if not any(p.get("type") == typ for p in tool.percepts) and hasattr(tool, "on_parsed"):
        tool.on_parsed({**parsed, "place_type": None})
    cands = [p for p in tool.percepts if p.get("type") == typ]
    if not cands:
        return {"agent": None, "parsed": parsed, "grounding": {"status": "fail", "reason": "no percept of the fixture"}}

    def key(p):
        f = p.get("fit") or {}
        return float(f.get("n_agree") or 0.0) * float(f.get("inlier") or 0.0)

    pc = max(cands, key=key)
    hp = handle_plan(lib, pc, parsed["part"])
    if hp is None:
        return {"agent": None, "parsed": parsed, "grounding": {"status": "fail", "reason": "no handle"}}
    scene = lib.scene_from_percepts(tool.percepts)
    if pc["id"] not in (scene.get("instances") or {}):
        return {"agent": None, "parsed": parsed, "grounding": {"status": "fail", "reason": "not in scene"}}
    spec = {"predicate": "open", "target": pc["id"], "place": None, "place_kind": None, "place_instance": None,
            "instruction": instruction, "part": parsed["part"]}
    ap, pp, fp = make_params(params.get("m1"))
    grounding = {"status": "ok", "target": pc["id"], "place": None, "predicate": "open", "part": parsed["part"],
                 "handle": [round(float(x), 4) for x in hp["c"]], "dir": [round(float(x), 3) for x in hp["d"]],
                 "bar": [round(hp["bar_h"], 4), round(hp["bar_w"], 4), round(hp["bar_depth"], 4)],
                 "q_now": round(hp["q_now"], 4)}
    agent = OpenAgent(lib, scene, spec, tool, cap, ap, pp, fp)
    agent.hp = hp
    return {"agent": agent, "parsed": parsed, "grounding": grounding}


# -- closing (the long-task suite's "... and close it"; reached only through the step sequencer's clauses)
CLOSE = re.compile(r"\b(?:close|shut|push closed|push in|slide closed|slide in) (?P<rest>.+)$")


def parse_close(instruction: str, lib, seen_types=()) -> dict:
    """``close the <part words> drawer of the <fixture>``: the drawer as ``parse_open`` finds it."""
    text = " ".join(words(instruction))
    m = CLOSE.search(text)
    if not m:
        return {"status": "fail", "reason": "no close verb"}
    p = parse_open("open " + m.group("rest"), lib, seen_types)
    if p.get("status") == "ok":
        p = {**p, "form": "close", "predicate": "close"}
    return p


class CloseAgent(OpenAgent):
    """Push a drawer closed: the jaws shut into a fist pointing into the fixture, in front of the handle, then straight
    in by the drawer's measured opening (from its rest, closed, position) and a margin, then back off and rise."""

    safe_m = 0.20  # the approach crosses this high above the handle
    front_m = 0.06  # and comes down this far in front of it

    def _episode(self):
        yield None
        ap, hp = self.ap, self.hp
        d = np.asarray(hp["d"], float)
        c = np.asarray(hp["c"], float)
        push = float(hp["push"])
        _p0, R0 = self._pose()
        Rh = self._hand_R(R0, d)
        att = {"index": 0, "t_start": self.t, "skill": "close", "grip": [round(float(x), 4) for x in c],
               "dir": [round(float(x), 3) for x in d], "push": round(push, 4)}
        self.attempts.append(att)
        if push < 0.015:
            att["outcome"] = "already closed"
            self._finish("closed")
            return
        # the hand keeps pointing down (as it ends a placement): a sideways fist needed a wrist turn and a reach the arm
        # could not hold (first smokes). Rise clear of the fixture, cross high to the front of the drawer, come down
        # in front of the handle, then push the drawer home horizontally with the closed jaws.
        Rd = R0
        p_pre = c + d * self.front_m
        z_safe = float(c[2]) + self.safe_m
        p_now, _ = self._pose()
        if float(p_now[2]) < z_safe:
            att["rise"] = yield from self._goto(np.array([p_now[0], p_now[1], z_safe]), Rd, -1.0, ap.transit_cap, 0.01,
                                                8.0, max_steps=60)
        p_hi = np.array([p_pre[0], p_pre[1], z_safe])
        att["hi"] = yield from self._goto(p_hi, Rd, -1.0, ap.transit_cap, 0.01, 6.0, max_steps=90)
        for _ in range(8):  # shut the jaws into a fist
            yield self._act_to(p_hi, Rd, 1.0)
        att["pre"] = yield from self._goto_g(p_pre, Rd, 1.0, ap.transit_cap, 0.006, 4.0, 80)
        p_s, _ = self._pose()
        att["push_status"] = yield from self._goto_g(c - d * (push + 0.03), Rd, 1.0, 0.5, 0.006, 8.0, 120)
        p_e, _ = self._pose()
        moved = float((p_s - p_e) @ d)
        att["moved"] = round(moved, 4)
        yield from self._goto_g(p_e + d * 0.08, Rd, 1.0, ap.transit_cap, 0.01, 8.0, 30)
        p_b, _ = self._pose()
        yield from self._goto_g(p_b + np.array([0.0, 0.0, 0.08]), Rd, 1.0, ap.transit_cap, 0.01, 8.0, 25)
        att["t_end"] = self.t
        att["outcome"] = "pushed" if moved >= self.front_m + 0.6 * push else "short"
        self._finish("closed" if att["outcome"] == "pushed" else "push_short")


def plan_close(instruction: str, tool, lib, cap: int, params: dict):
    """{agent, parsed, grounding} for a close-drawer clause, or None when it is not one."""
    from . import make_params

    parsed = parse_close(instruction, lib, [p.get("type") for p in tool.percepts])
    if parsed.get("status") != "ok":
        return None
    typ = parsed["target_type"]
    cands = [p for p in tool.percepts if p.get("type") == typ]
    if not cands:
        return {"agent": None, "parsed": parsed, "grounding": {"status": "fail", "reason": "no percept of the fixture"}}
    pc = max(cands, key=lambda p: float((p.get("fit") or {}).get("n_agree") or 0.0))
    hp = handle_plan(lib, pc, parsed["part"])
    if hp is None:
        return {"agent": None, "parsed": parsed, "grounding": {"status": "fail", "reason": "no handle"}}
    j = lib.export_types[typ]["parts"][parsed["part"]]["joint"]
    q_rest = float(j.get("qpos0") or 0.0)
    hp = {**hp, "push": float(np.clip(abs(float(hp["q_now"]) - q_rest), 0.0, 0.30))}
    scene = lib.scene_from_percepts(tool.percepts)
    if pc["id"] not in (scene.get("instances") or {}):
        return {"agent": None, "parsed": parsed, "grounding": {"status": "fail", "reason": "not in scene"}}
    spec = {"predicate": "close", "target": pc["id"], "place": None, "place_kind": None, "place_instance": None,
            "instruction": instruction, "part": parsed["part"]}
    ap, pp, fp = make_params(params.get("m1"))
    agent = CloseAgent(lib, scene, spec, tool, cap, ap, pp, fp)
    agent.hp = hp
    grounding = {"status": "ok", "target": pc["id"], "place": None, "predicate": "close", "part": parsed["part"],
                 "handle": [round(float(x), 4) for x in hp["c"]], "push": round(hp["push"], 4)}
    return {"agent": agent, "parsed": parsed, "grounding": grounding}


# -- an object that the opened part would cover (the long-suite work, 2026-11-15; reached only through the sequencer)
def open_far(tdoc: dict, part: str) -> float:
    j = tdoc["parts"][part]["joint"]
    lo, hi = [float(x) for x in j["range"]]
    q0 = float(j.get("qpos0") or 0.0)
    return lo if abs(lo - q0) >= abs(hi - q0) else hi


def open_poses(lib, percept: dict, part: str) -> dict:
    """{body: (R, t)} of the fixture with ``part`` fully open (the type's kinematics on the percept's root pose)."""
    tdoc = lib.export_types[percept["type"]]
    root = tdoc["root_body"]
    Rr = lib.quat_mat(percept["bodies"][root]["quat_wxyz"])
    tr = np.asarray(percept["bodies"][root]["pos"], float)
    return part_body_poses(tdoc, {root: (Rr, tr)}, {**_jv(percept), part: open_far(tdoc, part)})


def open_footprint(lib, percept: dict, part: str):
    """(lo, hi) of the part's collision boxes in the world when it is fully open, or None."""
    typ = percept["type"]
    P = open_poses(lib, percept, part)
    los, his = [], []
    for b in lib.export_types[typ]["parts"][part]["bodies"]:
        if b not in P:
            continue
        bx = lib.body_boxes(typ, b)
        if not bx.n:
            continue
        lo, hi = bx.placed(np.asarray(P[b][0], float), np.asarray(P[b][1], float)).aabb()
        los.append(lo.min(0))
        his.append(hi.max(0))
    if not los:
        return None
    return np.min(los, 0), np.max(his, 0)


def cavity(lib, percept: dict, min_depth_m: float = 0.025):
    """(floor z, rim z) of an object that is open at the top (a bowl): its primitives' top at its middle lies at least
    ``min_depth_m`` below its highest point. None for anything else."""
    from .ground import aabb, top_under

    root = lib.export_types[percept["type"]]["root_body"]
    c = np.asarray(percept["bodies"][root]["pos"], float)
    lo, hi = aabb(lib, percept)
    floor = top_under(lib, percept, c[:2])
    if floor is None or float(hi[2]) - float(floor) < min_depth_m:
        return None
    return float(floor), float(hi[2])


def clear_move(lib, percepts: list, target: dict, footprint, clear_m: float, table=None, n_dir: int = 12,
               max_m: float = 0.25, hand_z: float | None = None):
    """The shortest straight move (world xy) that takes ``target``'s middle ``clear_m`` outside ``footprint`` ((lo,
    hi), a part that is about to open) and ends over the table (``table``: (centre, half) of its box). That is
    enough for a hand to take it by the rim afterwards with its thin side to the part (2026-11-15: asking the whole
    object 7 cm clear left no free way on the benchmark's table). The object's own way (its width plus 1 cm, from
    1 cm ahead of its middle to where it ends) must be free of every other percept. The hand that drags it is long
    and rides above it: its way (8 cm wide, from 10 cm behind the object's middle to 12 cm past where it ends) must
    be free of what stands higher than ``hand_z`` (a plate beside the object is under it). None when no way is free."""
    from .ground import aabb

    root = lib.export_types[target["type"]]["root_body"]
    c = np.asarray(target["bodies"][root]["pos"], float)[:2]
    lo_t, hi_t = aabb(lib, target)
    r = 0.5 * float(max(hi_t[0] - lo_t[0], hi_t[1] - lo_t[1]))
    lo = np.asarray(footprint[0], float)[:2] - clear_m
    hi = np.asarray(footprint[1], float)[:2] + clear_m
    others = []
    for q in percepts:
        if q["id"] == target["id"] or not q.get("bodies"):
            continue
        try:
            a, b = aabb(lib, q)
        except Exception:
            continue
        others.append((np.asarray(a, float)[:2], np.asarray(b, float)[:2], float(b[2])))
    best = None
    for k in range(n_dir):
        ang = 2.0 * np.pi * k / n_dir
        u = np.array([np.cos(ang), np.sin(ang)])
        ts = [((hi[i] - c[i]) / u[i]) if u[i] > 1e-9 else ((c[i] - lo[i]) / -u[i]) for i in (0, 1) if abs(u[i]) > 1e-9]
        t = float(min(x for x in ts if x > 0.0)) if any(x > 0.0 for x in ts) else None
        if t is None or t > max_m:
            continue
        end = c + u * t
        if table is not None and not np.all(np.abs(end - np.asarray(table[0], float)[:2])
                                            <= np.asarray(table[1], float)[:2] - r):
            continue
        v = np.array([-u[1], u[0]])
        free = True
        for a, b, top in others:  # a box against each strip: its corners' extent along and across the move
            cs = np.array([[a[0], a[1]], [a[0], b[1]], [b[0], a[1]], [b[0], b[1]]]) - c
            al, ac = cs @ u, cs @ v
            in_way = al.max() > 0.01 and al.min() < t + r and ac.max() > -(r + 0.01) and ac.min() < (r + 0.01)
            tall = hand_z is None or top > hand_z
            under_hand = tall and al.max() > -0.10 and al.min() < t + 0.12 and ac.max() > -0.04 and ac.min() < 0.04
            if in_way or under_hand:
                free = False
                break
        if free and (best is None or t < best[0] - 1e-9):
            best = (t, u)
    if best is None:
        return None
    t, u = best
    return {"move": [float(u[0] * t), float(u[1] * t)], "dist": float(t), "r": float(r)}


class DragAgent(OpenAgent):
    """Move an open-topped object aside without grasping it: the closed fingers go down inside it and pull it along
    the support by its inner wall, which keeps it centred on them (a push from outside sends a round object off to a
    side). ``hp``: ``p_in`` (the point inside, a little above its floor), ``move`` (world xy), ``safe_z``."""

    lead_m = 0.02  # the fingers travel this far before they meet the wall

    def _episode(self):
        yield None
        ap, hp = self.ap, self.hp
        p_in = np.asarray(hp["p_in"], float)
        move = np.array([float(hp["move"][0]), float(hp["move"][1]), 0.0])
        u = move / max(float(np.linalg.norm(move)), 1e-9)
        _p, R_now = self._pose()
        R = min((GR.hand_matrix(u), GR.hand_matrix(-u)), key=lambda X: GR.rot_angle_deg(X, R_now))  # the hand along the move
        att = {"index": 0, "t_start": self.t, "skill": "drag", "p_in": [round(float(x), 4) for x in p_in],
               "move": [round(float(x), 4) for x in move[:2]]}
        self.attempts.append(att)
        over = np.array([p_in[0], p_in[1], float(hp["safe_z"])])
        att["over"] = yield from self._goto(over, R, 1.0, ap.transit_cap, 0.01, 6.0, max_steps=80)
        att["down"] = yield from self._goto_g(p_in, R, 1.0, ap.transit_cap, 0.006, 6.0, 30)
        p_s, _ = self._pose()
        att["t_drag"] = self.t
        att["drag"] = yield from self._goto_g(p_in + move + u * self.lead_m, R, 1.0, 0.8, 0.008, 8.0, 60)
        p_e, _ = self._pose()
        att["moved"] = round(float((p_e - p_s) @ u), 4)
        # up and out with the fingers opening (2026-11-15: left closed, the opening step that followed
        # shaped its gripper from shut and missed the handle)
        yield from self._goto_g(np.array([p_e[0], p_e[1], float(hp["safe_z"])]), R, -1.0, ap.transit_cap, 0.02, 180.0, 20)
        att["t_end"] = self.t
        ok = att["moved"] >= 0.8 * float(np.linalg.norm(move))
        att["outcome"] = "cleared" if ok else "short"
        self._finish("cleared" if ok else "drag_short")


class RetakeAgent(OpenAgent):
    """Take again an object this hand set down a moment ago, without looking: back to the pose it was released from
    (``hp``: ``p``, ``R``), down by ``drop`` (the object fell that far when it was let go), close, lift ``lift``, move
    the hand by ``move`` (world xy), let go. Staging put the object where the hand reaches it; this carries it the
    short way into the part opened since (2026-11-15: pushing it there needs the hand turned to a yaw the arm does
    not reach from the pull's pose)."""

    def _episode(self):
        yield None
        ap, hp = self.ap, self.hp
        p_rel, R = np.asarray(hp["p"], float), np.asarray(hp["R"], float)
        move = np.array([float(hp["move"][0]), float(hp["move"][1]), 0.0])
        p_g = p_rel - UPV * float(hp.get("drop", 0.02))
        att = {"index": 0, "t_start": self.t, "skill": "take", "grip": [round(float(x), 4) for x in p_g],
               "move": [round(float(x), 4) for x in move[:2]]}
        self.attempts.append(att)
        # the fingers stay fully open on the way down: the object shifts as it lands (2026-11-15: with a 5 cm gap a
        # finger came down on the bowl's rim in 4 of 8 starts; fully open, in none)
        att["over"] = yield from self._goto(p_g + UPV * float(hp.get("over", 0.05)), R, -1.0, ap.transit_cap, 0.02,
                                            10.0, max_steps=70)
        # (``close_early_m``: the object stands where this hand let go of it, and closing brings its wall between
        # the fingers from up to a centimetre either way; only over the last 1.2 cm, where the fingers are well past
        # the rim, because this grasp is made without a look)
        no_lag = int(getattr(self, "lag_steps", 0) or 0) <= 0
        gen_dn = self._goto(p_g, R, -1.0, ap.transit_cap, 0.006, 4.0, max_steps=30)
        if ap.take_down_lat_m > 0 and no_lag:
            gen_dn = self._down_lat(gen_dn, p_g, R, float(ap.take_down_lat_m))
        dz_close = float(ap.take_close_dz_m) if (ap.take_close_dz_m > 0 and no_lag) else 0.012
        att["down"] = yield from self._closing_early(gen_dn, p_g, 0.009, lat_m=0.012, max_dz_m=dz_close)
        att["t_close"] = self.t
        gaps = [self._gap()]
        wait = self._close_wait()  # (2026-11-17: 0 unless the delay compensator's closing hold was shortened)
        for k in range(ap.close_max + wait):
            yield self._act_to(p_g, R, 1.0)
            gaps.append(self._gap())
            if k >= 2 + wait and abs(gaps[-3] - gaps[-1]) < 0.0005:
                break
            if ap.take_hold_quick and no_lag and abs(gaps[-1] - gaps[-2]) < 0.0005 \
                    and ap.c1_rim_min_gap_m <= gaps[-1] < 0.03:
                break  # ``take_hold_quick``: stopped on something thin: the rim is held
        att["gap"] = round(gaps[-1], 4)
        if gaps[-1] < ap.c1_rim_min_gap_m:  # the fingers met: nothing between them
            for _ in range(ap.open_steps):
                yield self._act_to(p_g, R, -1.0)
            yield from self._goto(p_g + UPV * 0.08, R, -1.0, ap.transit_cap, 0.01, 180.0, max_steps=20)
            att["outcome"] = "missed"
            self._finish("take_missed")
            return
        p_l = p_g + UPV * float(hp.get("lift", 0.012))
        if not (ap.take_lift_carry and no_lag):  # (``take_lift_carry``: the lift is the carry's own first centimetre)
            yield from self._goto_g(p_l, R, 1.0, 0.6, 0.004, 6.0, 15)
        # let go once within 2 cm: the opened part is wider than the object by more than that on every side
        # (``take_carry_tol_m``: nearer, when set)
        gen_c = self._goto_g(p_l + move, R, 1.0, ap.transit_cap, float(ap.take_carry_tol_m), 8.0, 60)
        n_rel = [0]
        if ap.take_release_early > 0 and no_lag:
            gen_c = self._release_on_the_way(gen_c, p_l + move, float(ap.take_carry_tol_m), int(ap.take_release_early), n_rel)
        att["carry"] = yield from gen_c
        att["gap_carry"] = round(self._gap(), 4)
        p_e, _ = self._pose()
        att["t_release"] = self.t
        if n_rel[0]:
            att["released_early"] = n_rel[0]
        for _ in range(max(0, ap.open_steps - n_rel[0])):
            yield self._act_to(p_e, R, -1.0)
        yield from self._goto(p_e + UPV * 0.08, R, -1.0, ap.transit_cap, 0.01, 180.0, max_steps=20)
        att["t_end"] = self.t
        if att["gap_carry"] < ap.c1_rim_min_gap_m:
            att["outcome"] = "dropped"
            self._finish("take_dropped")
            return
        att["outcome"] = "placed"
        self._finish("placed")


    # -- 2026-11-17, second set (``take_down_lat_m``, ``take_release_early``; off by default)
    def _down_lat(self, gen, p_g, R, lat_m: float):
        """``take_down_lat_m``: run the way down to the grasp point (a generator of actions) until the hand is
        within 6 mm of its height, ``lat_m`` of it sideways and turned to within 6 degrees. "reached", or the
        move's own result when it ends first."""
        while True:
            p, R_n = self._pose()
            off = np.asarray(p, float) - np.asarray(p_g, float)
            if abs(float(off[2])) <= 0.006 and float(np.linalg.norm(off[:2])) <= lat_m \
                    and float(GR.rot_angle_deg(R_n, R)) <= 6.0:
                gen.close()
                self._ev("down", lat=round(float(np.linalg.norm(off[:2])), 4), dz=round(float(off[2]), 4))
                return "reached"
            try:
                a = next(gen)
            except StopIteration as e:
                return e.value
            yield a

    def _release_on_the_way(self, gen, p_goal, tol_m: float, n: int, count: list):
        """``take_release_early``: run the carry (a generator of actions) and turn its gripper command to "open"
        once the hand, at the speed it is coming, will be within ``tol_m`` of the goal in ``n`` steps. ``count[0]``:
        the open commands sent. Returns the carry's own result."""
        prev = None
        while True:
            try:
                a = next(gen)
            except StopIteration as e:
                return e.value
            p, _R = self._pose()
            left = float(np.linalg.norm((np.asarray(p_goal, float) - p)[:2])) - tol_m
            v = float(np.linalg.norm((p - prev)[:2])) if prev is not None else 0.0
            prev = np.array(p, dtype=float)
            if count[0] or (v > 0.004 and left <= n * v):
                a = list(a)
                a[6] = -1.0
                count[0] += 1
            yield a


class SweepAgent(OpenAgent):
    """Slide an object along the surface it stands on: come down behind it, push straight along ``d`` for ``dist``.
    ``hp``: ``start`` (the hand's point behind the object), ``d``, ``dist``, ``safe_z`` (the height to pass over the
    object at), and optionally ``obj`` (the object's middle), ``r`` (its half-width along ``d``) and ``width``.

    The hand is top-down with its closing axis across the push and the fingers part open, so the object rests against
    both: one closed finger block behind a round object pushes it off to a side (the offset grows as it travels).
    A hand that starts on the far side (it has just pulled the part open) first goes to a point above the surface in
    front of the object, turning on the way, and then passes over the object."""

    push_cap = 1.0

    def _episode(self):
        yield None
        ap, hp = self.ap, self.hp
        d = np.asarray(hp["d"], float)
        d = d / float(np.linalg.norm(d))
        start = np.asarray(hp["start"], float)
        dist = float(hp["dist"])
        p_now, R_now = self._pose()
        e = np.cross(UPV, d)
        Rv = min((GR.hand_matrix(e), GR.hand_matrix(-e)), key=lambda R: GR.rot_angle_deg(R, R_now))
        open_gap = 2.0 * self.jaw.L
        gap = float(np.clip(0.6 * float(hp.get("width") or 0.0), 0.03, 0.06)) if hp.get("width") else 0.0
        k_close = int(np.clip(round(8.0 * (1.0 - gap / max(open_gap, 1e-3))), 0, 8)) if gap else None
        shape = ([-1.0] * 3 + [1.0] * k_close) if k_close is not None else []
        g = 0.0 if k_close is not None else 1.0  # shaped and held, or (no width given) closed
        att = {"index": 0, "t_start": self.t, "skill": "sweep", "start": [round(float(x), 4) for x in start],
               "dir": [round(float(x), 3) for x in d], "dist": round(dist, 4), "k_close": k_close}
        self.attempts.append(att)
        z_over = float(hp["safe_z"])
        obj = np.asarray(hp["obj"], float)[:2] if hp.get("obj") is not None else None
        if obj is not None and float((p_now[:2] - obj) @ d[:2]) > 0.0:  # the hand is in front of the object
            front = obj + d[:2] * (float(hp.get("r") or 0.0) + 0.03)
            att["rise"] = yield from self._goto_seq(np.array([front[0], front[1], z_over]), Rv, shape, g,
                                                    ap.transit_cap, 0.03, 20.0, 60)
            shape = []
        elif float(p_now[2]) < z_over - 0.01:
            att["rise"] = yield from self._goto_seq(np.array([p_now[0], p_now[1], z_over]), Rv, shape, g,
                                                    ap.transit_cap, 0.015, 15.0, 40)
            shape = []
        att["over"] = yield from self._goto_seq(np.array([start[0], start[1], z_over]), Rv, shape, g, ap.transit_cap,
                                                0.01, 6.0, 60)
        att["gap"] = round(self._gap(), 4)
        att["down"] = yield from self._goto_g(start, Rv, g, 0.8, 0.006, 6.0, 40)
        p_s, _ = self._pose()
        att["t_push"] = self.t
        att["push"] = yield from self._goto_g(start + d * dist, Rv, g, self.push_cap, 0.006, 8.0, 80)
        p_e, _ = self._pose()
        att["moved"] = round(float((p_e - p_s) @ d), 4)
        att["t_push_end"] = self.t
        yield from self._goto_g(p_e + UPV * 0.08, Rv, g, ap.transit_cap, 0.01, 8.0, 25)
        att["t_end"] = self.t
        att["outcome"] = "swept" if att["moved"] >= 0.8 * dist else "short"
        self._finish("swept" if att["outcome"] == "swept" else "sweep_short")
