"""Stream PATH (2026-11-17): looks for the hand's way (``look_cross``, ``look_stall``), the empty hand's way past a
tall unnamed thing (``route_empty``), and the detour that uses what was seen (``detour_seen``). All off by default.
The kinematic stand-in; where the real look tool is in the loop its looks are shaped as the colour line's (a long
region with the thing as its top layer, a slice of the arm, the hand's own top)."""

from dataclasses import replace

import numpy as np
import pytest

from rowant_lab.libero_env.tests import test_place_measured as PM
from rowant_lab.libero_env.tests import test_pro2_steps_run as SR
from rowant_lab.pro2.robot import agent as AG
from rowant_lab.pro2.robot import look as LK

pytestmark = pytest.mark.skipif(not SR.LIB_ROOT.is_dir(), reason="the object library is not built on this machine")

V9 = dict(transit_unexplained=True, transit_cut=True, detour_from_cross=True)  # how candidate v9 carries


class Things(SR.Tool):
    """The look tool's side of the stand-in: the tall unnamed things a look shows (``(centre, R, half)`` boxes).
    ``late``: no look shows them until one is taken for the way (``look_route``), as a thing put there after the
    first look."""

    def __init__(self, lib, world, extra=(), things=(), late=False):
        super().__init__(lib, world, extra)
        self.things, self.shown, self.n_route, self.asked = list(things), not late, 0, 0
        self.reveal = 1  # (the look for the way that first shows a late thing)

    def relook_place(self, place_id):
        return {"moved_m": 0.0, "assoc": 0.0}

    def look_route(self):
        self.n_route += 1
        self.shown = self.shown or self.n_route >= self.reveal
        return {"n": self.n_route}

    def tall_unexplained(self, min_h_m=0.08, clear_xy=(), clear_seg=None, held_ids=()):
        self.asked += 1
        return list(self.things) if self.shown else []


def _box(x, y, z, yaw=0.0):
    """A box from its ranges ``(lo, hi)`` along x, y and z."""
    c = np.array([0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1]), 0.5 * (z[0] + z[1])])
    return c, np.eye(3), np.array([0.5 * (x[1] - x[0]), 0.5 * (y[1] - y[0]), 0.5 * (z[1] - z[0])])


def _wall_seen(world_cls, top=None):
    """What a look shows of ``PM.WallWorld``'s wall (it ends 5 cm short of where the hand comes down at the plate)."""
    y0, y1 = world_cls.Y
    return _box((-0.04, 0.40), (y0, y1), (0.97, (world_cls.TOP - 0.01) if top is None else top))


def _go(world, tool, cap=300, prep=None, **opts):
    from rowant_lab.pro2.robot import make_params

    lib = tool.lib
    ap, pp, fp = make_params(None)
    spec = {"predicate": "on", "target": "p0", "place": "p1", "place_kind": "object", "place_instance": "p1",
            "instruction": "pick the akita black bowl and place it on the plate"}
    agent = AG.OracleAgent(lib, lib.scene_from_percepts(tool.percepts()), spec, tool, cap, replace(ap, **opts), pp, fp)
    if prep is not None:
        prep(agent)
    done_at = None
    for t in range(cap):
        world.step(agent.act(world.pobs()))
        if done_at is None and world.held is None and world.bowl[2] > 0.91 \
                and float(np.linalg.norm((world.bowl - SR.PLATE)[:2])) < 0.025:
            done_at = t + 1
        if agent.finished:
            break
    return agent, done_at


def _lib():
    from rowant_lab.libero_env import m2_library as LB

    return LB.M2Library("libero_spatial")


def _wall(world_cls, late=True, top=None, known=True, **opts):
    lib = _lib()
    world = world_cls(lib)
    tool = Things(lib, world, [SR.WALL] if known else [], [_wall_seen(world_cls, top)], late)
    agent, done = _go(world, tool, **opts)
    return agent, done, tool


def _detours(agent):
    return [e for e in agent.events if e["ev"] == "detour"]


def _route(agent, k=0):
    return agent.attempts[k].get("route") or []


class GateWorld(SR.World):
    """A wall between where the hand starts and the bowl: an EMPTY hand does not pass it lower than its top."""
    X, TOP = (-0.04, -0.02), 1.20

    def step(self, a):
        p_old = self.p.copy()
        super().step(a)
        if self.held is None and self.X[0] - 0.01 < self.p[0] < self.X[1] + 0.01 and self.p[2] < self.TOP + 0.005 \
                and p_old[0] <= self.X[0] - 0.01:
            self.p = p_old + np.array([0.0, 0.0, 2e-5 * (1 if self.t % 2 else -1)])  # (a blocked hand still trembles)
            self.path[-1] = (self.p.copy(), self.R.copy(), None)


def _gate(things=None, late=False, reveal=1, **opts):
    lib = _lib()
    world = GateWorld(lib)
    seen = [_box(GateWorld.X, (-0.25, 0.25), (0.90, GateWorld.TOP))] if things is None else things
    tool = Things(lib, world, [], seen, late)
    tool.reveal = reveal
    agent, done = _go(world, tool, **opts)
    return agent, done, tool


def _transits(agent, key):
    return [e for e in agent.events if e["ev"] == "transit" and key in e]


# ---- ``look_cross``, ``look_stall``: the real look tool, its looks shaped as the colour line's
def test_a_slab_set_down_as_the_carry_starts_is_seen_by_the_look_on_the_way_and_crossed_untouched():
    free, tool0, _res, there0 = PM._slab_cell(10 ** 6, transit_unexplained=True)  # (no slab: when the carry starts)
    t_carry = free.attempts[0]["t_transport"]
    arrives = t_carry + 1  # (truth replays of training tasks: it stands there 0 to 3 steps after the carry starts)
    # as candidate v9 carries: the slab is in no look, the carry meets it, then the detour
    v9, tool, _res, there = PM._slab_cell(arrives, transit_unexplained=True)
    assert there0 and there and len(_detours(v9)) == 1 and "route" not in v9.attempts[0]
    assert getattr(tool, "n_route", 0) == 0
    # the look four steps into the carry shows it: planned over it from there, nothing met, and sooner in the basket
    on, tool, _res, there = PM._slab_cell(arrives, transit_unexplained=True, look_cross=True)
    (rec,) = _route(on)
    assert there and not _detours(on) and on.t < v9.t - 8, (on.t, v9.t)
    assert rec["why"] == "cross" and rec["t"] == t_carry + 4 and rec["looked"] and rec["s"] >= 0.0
    assert rec["new"] == 1 and rec["plan"] == "over" and rec["counted"] == [[0.02, -0.05, 0.3]]  # (not arm, not hand)
    assert [e["t"] for e in _transits(on, "unexplained")] == [rec["t"]]
    assert on.attempts[0]["transport_status"][0] == "reached"
    # the look is the way's alone: outside the budget, and no percept moved by it
    assert tool.n_route == 1 and tool.budget == LK.LOOK_BUDGET + 1 and tool.looks[-1].get("route") is None
    assert sum(1 for x in tool.looks if x.get("route")) == 1
    # nothing new in the look: the carry runs as it would have, to the step
    same, tool, _res, there = PM._slab_cell(10 ** 6, transit_unexplained=True, look_cross=True, look_stall=2)
    assert there and same.t == free.t and [r["new"] for r in _route(same)] == [0] and not _detours(same)
    assert [e for e in same.events if e["ev"] == "goto"] == [e for e in free.events if e["ev"] == "goto"]


def test_a_slab_that_comes_after_that_look_is_seen_at_the_first_sign_of_a_stall():
    free = PM._slab_cell(10 ** 6, transit_unexplained=True)[0]
    late = free.attempts[0]["t_transport"] + 6  # (two steps after the look on the way)
    v9, _tool, _res, there0 = PM._slab_cell(late, transit_unexplained=True)
    stop = _detours(v9)[0]["t"]  # (the step at which the pushing move was given up)
    assert there0
    # the look four steps in sees nothing; alone it changes nothing
    cross, _tool, _res, there1 = PM._slab_cell(late, transit_unexplained=True, look_cross=True)
    assert there1 and cross.t == v9.t and [r["new"] for r in _route(cross)] == [0] and len(_detours(cross)) == 1
    # with the look at a stall: two steps of hardly any way made, the look, the slab, over it; no detour
    for opts in (dict(look_stall=2), dict(look_cross=True, look_stall=2)):
        on, tool, _res, there = PM._slab_cell(late, transit_unexplained=True, **opts)
        rec = _route(on)[-1]
        assert there and not _detours(on) and on.t < v9.t - 5, (on.t, v9.t)
        assert rec["why"] == "stall" and rec["new"] == 1 and rec["plan"] == "over"
        assert late < rec["t"] <= late + 4 and rec["t"] <= stop - 6, (late, rec["t"], stop)
        assert len(_route(on)) == len(opts) and tool.n_route == len(opts)
    # the block watcher needs a dozen steps for the same
    block = PM._slab_cell(late, transit_unexplained=True, carry_block_steps=12)[0]
    assert next(e["t"] for e in block.events if e["ev"] in ("carry_blocked", "detour")) >= rec["t"] + 6


def test_the_look_for_the_way_changes_no_percept_and_has_a_cap_of_its_own():
    ag, tool, _res, _there = PM._slab_cell(0, transit_unexplained=True)
    import copy

    before, n_full, budget = copy.deepcopy(tool.percepts), tool.n_full, tool.budget
    tool.extra_last = {"blobs": []}
    out = tool.look_route(cap=2)
    assert out == {"n": 1} and tool.percepts == before and tool.n_full == n_full + 1 and tool.budget == budget + 1
    assert len(tool.extra_last["blobs"]) == 3 and tool.looks[-1]["route"] is True and tool.looks[-1]["merge"] is True
    assert tool.look_route(cap=2) == {"n": 2} and tool.look_route(cap=2) is None and tool.n_full == n_full + 2
    # the hand's own top in a look taken from where the hand is (the stand-in's server: 3.6 cm from the grip site,
    # 3 cm above it) is a thing to the carry as v9 plans it, and none once looks are taken on the way
    hand = np.asarray(ag._pose()[0], dtype=float)

    def own(bx):
        return [c for c, h in zip(bx.C, bx.H)
                if float(np.linalg.norm((c - hand)[:2])) <= 0.06 and 0.015 <= float(c[2] + h[2] - hand[2]) <= 0.05]

    off = ag._unexplained()
    ag.ap = replace(ag.ap, look_stall=1)
    on = ag._unexplained()
    assert len(own(off)) == 1 and off.n == 3  # (as v9 counts: the first look's things and the latest look's)
    # with looks for the way: the latest such look alone, and the hand's own top is no thing: the slab is left
    assert not own(on) and on.n == 1 and np.allclose(on.C[0][:2], [0.02, -0.05], atol=1e-3)


# ---- the same on the stand-in's own tool: walls of other heights, and the detour
def test_a_wall_met_on_the_way_is_looked_at_and_crossed_at_once_instead_of_pushed_against():
    v9, done0, _ = _wall(PM.LowWallWorld, **V9)  # in no look: met, pushed against, then the fixed detour
    assert done0 is not None and len(_detours(v9)) == 1 and not _route(v9)
    stall, done1, tool1 = _wall(PM.LowWallWorld, look_stall=2, **V9)
    (rec,) = _route(stall)
    assert done1 is not None and done1 < done0 - 20 and not _detours(stall) and tool1.n_route == 1
    assert rec["why"] == "stall" and rec["new"] == 1 and rec["plan"] == "over"
    assert rec["t"] <= stall.attempts[0]["t_transport"] + 6 and rec["t"] < _detours(v9)[0]["t"] - 15
    # a wall taller than the fixed detour rises: lost as v9 carries, placed with the look
    lost, done2, _ = _wall(TallWallWorld, **V9)
    seen, done3, _ = _wall(TallWallWorld, look_stall=2, **V9)
    assert done2 is None and lost.attempts[0]["outcome"] == "c3_fail" and done3 is not None and not _detours(seen)
    # seen from the first look on there is nothing to look for: no look is taken at all
    first, done4, tool4 = _wall(PM.LowWallWorld, late=False, look_cross=False, look_stall=2, **V9)
    plain, done5, _ = _wall(PM.LowWallWorld, late=False, **V9)
    assert done4 == done5 and first.t == plain.t and not _route(first) and tool4.n_route == 0
    # through a delay of the commands the move runs as it is
    lib = _lib()
    world = PM.LowWallWorld(lib)
    tool = Things(lib, world, [SR.WALL], [_wall_seen(PM.LowWallWorld)], True)
    from rowant_lab.pro2.robot import make_params

    ap, pp, fp = make_params(None)
    spec = {"predicate": "on", "target": "p0", "place": "p1", "place_kind": "object", "place_instance": "p1",
            "instruction": "pick the akita black bowl and place it on the plate"}
    ag = AG.OracleAgent(lib, lib.scene_from_percepts(tool.percepts()), spec, tool, 120,
                        replace(ap, look_cross=True, look_stall=2, detour_seen=True, **V9), pp, fp)
    ag.lag_steps = 2
    for _t in range(120):
        world.step(ag.act(world.pobs()))
        if ag.finished:
            break
    assert tool.n_route == 0 and all("route" not in a for a in ag.attempts)


class TallWallWorld(PM.WallWorld):
    TOP = 1.30


def test_the_detour_crosses_at_the_height_the_seen_top_asks_not_a_fixed_rise():
    v9, done0, _ = _wall(PM.LowWallWorld, **V9)
    z_fixed = _detours(v9)[0]["z"]
    # a low wall: the detour looks, and crosses 3 cm lower than the fixed rise, and sooner
    low, done1, tool1 = _wall(PM.LowWallWorld, detour_seen=True, **V9)
    (d1,) = _detours(low)
    assert done1 is not None and done1 < done0 and d1["seen"] == "over" and d1["z"] < z_fixed - 0.02
    assert [(r["why"], r["looked"]) for r in _route(low)] == [("detour", True)] and tool1.n_route == 1
    assert low.attempts[0]["transport_status"][0] == "detour_reached"
    # a wall the fixed rise does not clear: lost without, over it with
    lost, done2, _ = _wall(TallWallWorld, **V9)
    tall, done3, _ = _wall(TallWallWorld, detour_seen=True, **V9)
    assert done2 is None and done3 is not None and _detours(tall)[0]["seen"] == "over"
    assert _detours(tall)[0]["z"] > TallWallWorld.TOP + 0.03 > _detours(lost)[0]["z"]
    # a look that shows nothing: the fixed detour, to the step (the stand-in names no shoulder: no reach to cap it)
    lib = _lib()
    world = PM.LowWallWorld(lib)
    blind, done4 = _go(world, Things(lib, world, [SR.WALL], [], True), detour_seen=True, **V9)
    assert done4 == done0 and blind.t == v9.t and _detours(blind)[0] == {**_detours(v9)[0], "seen": "none"}
    # too tall to cross within reach: around its nearer end, when that is allowed
    E = PM.EndWallWorld

    def end(**opts):
        w = E(lib)
        return _go(w, Things(lib, w, [], [_box((E.X0, 0.35), E.Y, (0.90, E.TOP))], True), **opts)

    stuck, done5 = end(**V9)
    over, done6 = end(detour_seen=True, **V9)  # (no way over it and none around allowed: the fixed detour)
    around, done7 = end(detour_seen=True, transit_around=True, **V9)
    assert done5 is None and done6 is None and _detours(over)[0]["seen"] == "none"
    assert done7 is not None and _detours(around)[0]["seen"] == "around"
    assert around.attempts[0]["transport_status"][0] == "detour_reached"
    # and with the look at the stall it goes around without a detour
    quick, done8 = end(look_stall=2, transit_around=True, **V9)
    assert done8 is not None and done8 < done7 and not _detours(quick) and _route(quick)[0]["plan"] == "around"
    assert any("around" in e for e in _transits(quick, "unexplained"))


def test_the_arm_does_not_rise_out_of_its_reach_for_a_fixed_detour():
    lib = _lib()
    world = PM.LowWallWorld(lib)
    tool = Things(lib, world, [SR.WALL], [], True)
    from rowant_lab.pro2.robot import make_params

    ap, pp, fp = make_params(None)
    spec = {"predicate": "on", "target": "p0", "place": "p1", "place_kind": "object", "place_instance": "p1",
            "instruction": "pick the akita black bowl and place it on the plate"}
    ag = AG.OracleAgent(lib, lib.scene_from_percepts(tool.percepts()), spec, tool, 300,
                        replace(ap, detour_seen=True, **V9), pp, fp)
    world.step(ag.act(world.pobs()))  # (the hand has been read once)
    att, poses = {}, tool()
    p_b, p_goal, R = np.array([0.05, 0.05, 1.03]), np.array([-0.10, 0.22, 1.02]), np.eye(3)
    held = (np.eye(3), np.array([0.0, 0.0, -0.05]))
    z, way, mark = ag._detour_seen(att, poses, p_b, R, p_goal, 0.02, held, ("p0",), 0.02, 1.21)
    assert (z, way, mark) == (1.21, None, {"seen": "none"}) and att["route"][0]["why"] == "detour"
    # a shoulder 64 cm behind the stop: 1.21 m there is beyond the arm, and the detour stays within its reach
    ag.shoulder = np.array([-0.59, 0.05, 0.95])
    assert not ag._reach_ok([[0.05, 0.05, 1.21]])
    z2, way2, mark2 = ag._detour_seen(att, poses, p_b, R, p_goal, 0.02, held, ("p0",), 0.02, 1.21)
    assert way2 is None and 1.09 <= z2 < 1.21 and ag._reach_ok([[0.05, 0.05, z2], [-0.10, 0.22, z2]])
    assert not ag._reach_ok([[0.05, 0.05, z2 + 0.01]])  # (the highest centimetre it reaches)
    assert mark2 == {"seen": "none", "capped": round(1.21 - z2, 3)} and len(att["route"]) == 1  # (looked 0 steps ago)


# ---- ``route_empty``: the empty hand's way
def test_the_empty_hand_goes_over_a_wall_between_it_and_its_target_that_the_look_showed():
    v9, done0, _ = _gate(**V9)  # the loaded hand's rule alone: the empty hand pushes at the wall, attempt after attempt
    assert done0 is None and all(a.get("stage") == "pre_grasp:unreachable" for a in v9.attempts[:3])
    assert not _transits(v9, "empty")
    on, done1, _ = _gate(route_empty=True, **V9)
    (ev,) = _transits(on, "empty")
    assert done1 is not None and on.attempts[0]["outcome"] == "placed" and ev["unexplained"] == 1 and ev["t"] == 0
    first = next(e for e in on.events if e["ev"] == "goto")
    assert first["status"] == "reached" and first["tgt"][2] > GateWorld.TOP + 0.02  # (up and over, then down)
    # a wall that is in no look yet: met, looked at when the hand stalls, and crossed (the two options together)
    blind, done2, tool2 = _gate(late=True, route_empty=True, **V9)
    assert done2 is None and tool2.n_route == 0
    # with looks for the way the empty hand's way starts with one (step 0): the wall is in it, and is crossed at once
    start, done3, tool3 = _gate(late=True, route_empty=True, look_stall=2, **V9)
    (first,) = _route(start)
    assert first["why"] == "start" and first["t"] == 0 and first["looked"] and tool3.n_route == 1
    assert done3 is not None and [e["t"] for e in _transits(start, "empty")] == [0]
    # a wall that comes after that look: met, looked at when the hand stalls, and crossed
    stall, done5, tool5 = _gate(late=True, reveal=2, route_empty=True, look_stall=2, **V9)
    first, rec = _route(stall)
    assert done5 is not None and first["why"] == "start" and tool5.n_route == 2
    assert rec["why"] == "stall" and rec["new"] == 1 and rec["plan"] == "over" and rec["t"] <= 10
    assert rec["at"][0] < GateWorld.X[0] and len(rec["made"]) == 2 and abs(rec["made"][0]) < 0.05
    # without ``route_empty`` the empty hand's moves are not watched at all
    alone, done4, tool4 = _gate(late=True, look_stall=2, **V9)
    assert done4 is None and tool4.n_route == 0 and alone.t == blind.t


def test_a_thing_at_the_grasp_point_is_not_the_ways_business():
    lib = _lib()
    over_bowl = [_box((0.0, 0.10), (0.0, 0.10), (1.02, 1.10))]  # (hanging over the bowl: the hand's goal is in it)

    def run(things, **opts):
        world = SR.World(lib)
        return _go(world, Things(lib, world, [], things), **opts)

    off, done0 = run(over_bowl, **V9)
    on, done1 = run(over_bowl, route_empty=True, **V9)
    assert done0 is not None and done1 == done0 and on.t == off.t and not _transits(on, "empty")
    assert [e for e in on.events if e["ev"] == "goto"] == [e for e in off.events if e["ev"] == "goto"]
    # a thing beside the way, far from the goal and not in the way: counted, and the way is the same
    beside = [_box((-0.30, -0.25), (-0.30, -0.25), (0.90, 1.20))]
    off2, done2 = run(beside, **V9)
    on2, done3 = run(beside, route_empty=True, **V9)
    assert done3 == done2 and on2.t == off2.t and not _transits(on2, "empty")  # ("clear": no event, no change)


def test_the_way_around_stays_in_front_of_the_robots_base_and_within_its_reach():
    lib, E = _lib(), PM.EndWallWorld

    def end(shoulder):
        def prep(agent):  # the stand-in names no shoulder: one is given to the way around alone
            orig = agent._way_around

            def way_around(*a, **k):
                agent.shoulder = None if shoulder is None else np.array(shoulder, dtype=float)
                try:
                    return orig(*a, **k)
                finally:
                    agent.shoulder = None

            agent._way_around = way_around

        w = E(lib)
        tool = Things(lib, w, [], [_box((E.X0, 0.35), E.Y, (0.90, E.TOP))], True)
        agent, done = _go(w, tool, prep=prep, detour_seen=True, transit_around=True, **V9)
        return done, _detours(agent)[0]["seen"]

    assert end(None)[1] == "around" and end(None)[0] is not None
    # the wall's nearer end 60 cm in front of the shoulder, level with the way: around it
    assert end([-0.85, 0.0, 1.00]) == end(None)
    # the same end less than 25 cm in front of the shoulder (beside the robot): no way around is planned
    assert end([-0.40, 0.0, 1.00]) == (None, "none")
    # and one the arm does not reach (1.2 m across): none either
    assert end([-1.45, 0.0, 1.00]) == (None, "none")


# ---- 2026-11-18: late commands, and the robot's own arm, hand and load in a look for the way
class LateWorld(SR.World):
    """The arm's commands are carried out ``LAG`` steps late from step ``t0`` on, for ``N`` steps; the fingers' at
    once. ``pause``: nothing is carried out while the first late command is on its way (else the command of that
    moment goes on)."""
    LAG, N = 8, 40

    def __init__(self, lib, t0, pause=False):
        super().__init__(lib)
        self.t0, self.pause, self.q = t0, pause, None

    def step(self, a):
        a = np.asarray(a, float).copy()
        if self.t0 <= self.t < self.t0 + self.N:
            if self.q is None:
                self.q = [np.zeros(6) if self.pause else a[:6].copy() for _ in range(self.LAG)]
            self.q.append(a[:6].copy())
            a[:6] = self.q.pop(0)
        super().step(a)


def test_late_commands_are_not_a_stall_and_no_look_is_taken_for_them():
    lib = _lib()
    free, _done = _go(SR.World(lib), Things(lib, SR.World(lib)), **V9)
    t0 = free.attempts[0]["t_transport"] + 8  # (the carry is under way when the commands start to arrive late)

    def run(pause, **opts):
        world = LateWorld(lib, t0, pause)
        tool = Things(lib, world)
        agent, done = _go(world, tool, **opts, **V9)
        return agent, done, tool

    for pause in (False, True):
        ref, done0, _ = run(pause)
        on, done1, tool = run(pause, look_cross=True, look_stall=2)
        stalls = [r for a in on.attempts for r in a.get("route") or [] if r["why"] == "stall"]
        late = [a["route_late"] for a in on.attempts if a.get("route_late") is not None]
        assert not stalls, (pause, stalls)  # no look at a stall, whichever way the delay begins
        if not pause:
            # the hand runs on past where it is sent, and back against its commands: read as late commands
            assert late and late[0] >= t0, late
        # (``pause``: a reading that stands exactly still is a frozen reading to the robot as it is, and it waits)
        # and the looks change nothing: the same commands, to the step
        assert (done1, on.t) == (done0, ref.t)
        assert [e for e in on.events if e["ev"] == "goto"] == [e for e in ref.events if e["ev"] == "goto"]


def test_a_wall_still_shows_as_a_stall_though_the_hand_slides_a_little_along_it():
    class SlideWall(PM.LowWallWorld):
        """The wall of ``LowWallWorld``; a hand that pushes against it slides 2 mm a step along it."""

        def step(self, a):
            p_old = self.p.copy()
            super().step(a)
            if self.held is not None and float(np.linalg.norm(self.p - p_old)) < 1e-4 and abs(float(a[1])) > 0.3:
                self.p = self.p + np.array([0.002, 0.0, 0.0])
                self.bowl = self.p + self.held
                self.path[-1] = (self.p.copy(), self.R.copy(), self.bowl.copy())

    agent, done, tool = _wall(SlideWall, look_stall=2, **V9)
    (rec,) = _route(agent)
    assert done is not None and rec["why"] == "stall" and rec["new"] == 1 and not _detours(agent)
    assert agent.attempts[0].get("route_late") is None and 0.0 <= rec["made"][1] < 0.6 * 0.25


def test_what_the_hand_holds_is_not_a_thing_on_its_way_and_what_it_rests_against_is():
    lib = _lib()
    world = PM.LowWallWorld(lib)
    tool = Things(lib, world, [SR.WALL], [], True)
    from rowant_lab.pro2.robot import make_params

    ap, pp, fp = make_params(None)
    spec = {"predicate": "on", "target": "p0", "place": "p1", "place_kind": "object", "place_instance": "p1",
            "instruction": "pick the akita black bowl and place it on the plate"}
    ag = AG.OracleAgent(lib, lib.scene_from_percepts(tool.percepts()), spec, tool, 300,
                        replace(ap, look_stall=2, **V9), pp, fp)
    world.step(ag.act(world.pobs()))
    hand = np.asarray(ag._pose()[0], dtype=float)
    held = (np.eye(3), np.array([0.07, 0.0, -0.05]))  # (a bowl held by its rim: its middle 7 cm beside the grip site)
    lo, hi = ag._held_boxes(held[0], held[1], hand, ag._pose()[1]).aabb()
    lo, hi = lo.min(axis=0), hi.max(axis=0)  # where the load is, as the agent has it
    assert float(np.linalg.norm((0.5 * (lo + hi) - hand)[:2])) > 0.06  # (not at the grip site: the hand's band misses it)
    load = _box((lo[0], hi[0]), (lo[1], hi[1]), (lo[2] - 0.05, hi[2] - 0.02))  # as a look shows it: a little low
    wall = _box((lo[0] - 0.05, hi[0] + 0.05), (hi[1] + 0.05, hi[1] + 0.06), (lo[2] - 0.10, hi[2] + 0.15))
    low = _box((lo[0], hi[0]), (lo[1], hi[1]), (lo[2] - 0.30, lo[2] - 0.10))
    tool.things, tool.shown = [load, wall, low], True
    seen = ag._unexplained(held)
    tops = sorted(round(float(c[2] + h[2] - hi[2]), 2) for c, h in zip(seen.C, seen.H))
    assert tops == [round(float(lo[2] - hi[2]) - 0.10, 2), 0.15]  # the load is left out; the wall and what lies under it are not
    assert ag._unexplained().n == 3  # (an empty hand has no load to leave out)
    ag.ap = replace(ag.ap, look_stall=0)
    assert ag._unexplained(held).n == 3  # (and as v9 carries, every one counts, as before)


def test_a_look_for_the_way_leaves_the_robot_out_in_the_picture_and_is_the_ways_one_source():
    ag, tool, _res, _there = PM._slab_cell(0, transit_unexplained=True)
    sent, call = [], tool.client.call

    def recording(msg):
        sent.append(dict(msg))
        return call(msg)

    tool.client.call = recording
    assert tool.look_route(cap=5) == {"n": 1} and "self_px" not in sent[-1]  # (as every look is taken)
    assert tool.look_route(cap=5, self_px=10) == {"n": 2}
    assert sent[-1]["self_px"] == 10 and sent[-1]["mode"] == "full" and tool.extra_route is tool.extra_last
    tool.full_look(merge=True)  # (any other look: its key is not sent, and the way's own look stays)
    assert "self_px" not in sent[-1] and tool.extra_route is not tool.extra_last
    both = tool.tall_unexplained(0.13)
    tool.extra_route = {"blobs": []}
    assert len(both) >= 2 and tool.tall_unexplained(0.13, route=True) == [] and len(tool.tall_unexplained(0.13)) == len(both)
    # a look in which the things that stand still read as moved is not kept
    full = tool.full_look

    def moved(**kw):
        out = full(**kw)
        tool._assoc_all = {"p0": 0.30, "p1": 0.05, "p2": 0.04}
        return out

    tool.full_look, kept, last = moved, tool.extra_route, tool.extra_last
    tool.target = "p0"
    assert tool.look_route(cap=5, self_px=10) == {"n": 3, "shifted": 0.05}
    assert tool.extra_route is kept and tool.extra_last is last and tool.looks[-1]["shifted"] == 0.05
    tool.full_look = lambda **kw: (full(**kw), setattr(tool, "_assoc_all", {"p0": 0.3, "p1": 0.05, "p2": 0.004}))[0]
    assert tool.look_route(cap=5) == {"n": 4} and tool.extra_route is tool.extra_last  # (one of two moved: kept)


def test_the_agent_does_not_read_a_shifted_look():
    class Shifted(Things):
        def look_route(self, cap=8, self_px=0):
            self.asked_px = self_px
            return {**super().look_route(), "shifted": 0.06}

    lib = _lib()
    world = PM.LowWallWorld(lib)
    tool = Shifted(lib, world, [SR.WALL], [_wall_seen(PM.LowWallWorld)], True)
    agent, done = _go(world, tool, look_stall=2, **V9)
    recs = _route(agent)
    assert tool.asked_px == 10 and recs and all(r["looked"] is False and r["shifted"] == 0.06 for r in recs)
    assert all("new" not in r for r in recs) and len(_detours(agent)) == 1 and done is not None  # (as v9: the detour)


def test_the_servers_filter_takes_the_robots_pixels_out_of_the_depth_when_a_look_asks_for_it():
    from types import SimpleNamespace

    from rowant_lab.libero_env import m2_perceive as MP

    class Panda:
        @staticmethod
        def fk(joint_pos, gripper_qpos):
            return {"hand": (np.eye(3), np.zeros(3))}

    wc = SimpleNamespace(panda=Panda(), base=np.zeros(3),
                         robot_pts={"hand": np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])})
    depth = np.full((64, 64), 1050, np.uint16)  # everything 5 cm behind where the robot's model stands
    depth[32, 10] = 0
    fr = {"K": [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]], "T_world_from_camera": np.eye(4),
          "depth_u16": depth, "depth_scale_mm": 1.0}
    per = SimpleNamespace(p=dict(MP.PARAMS), _frame_arrays=MP.Perceiver._frame_arrays)
    plain, md0 = MP.Perceiver._mask_depth(per, wc, fr, None, None)
    assert md0["robot_px"] == 0 and plain[32, 32] == 1050 and plain[32, 42] == 1050  # 1 cm in space finds none of it
    out, md1 = MP.Perceiver._mask_depth(per, wc, fr, None, None, self_px=3)
    assert "picture_px" not in md0 and md1["picture_px"] == md1["robot_px"] == int((out == 0).sum()) - 1  # (the record)
    assert out[32, 32] == 0 and out[32, 42] == 0 and out[32, 35] == 0 and out[29, 32] == 0
    assert out[32, 37] == 1050 and out[10, 10] == 1050 and (out == 0).sum() < 80  # and nothing else
    assert np.array_equal(MP.Perceiver._mask_depth(per, wc, fr, None, None, self_px=None)[0], plain)


def test_a_full_look_leaves_the_robot_out_in_the_picture_only_when_the_look_or_the_server_asks():
    from types import SimpleNamespace

    from rowant_lab.libero_env import m2_perceive as MP
    from rowant_lab.pro2 import long_dev as LD

    f = MP.Perceiver._full_self_px
    plain = SimpleNamespace(p=dict(MP.PARAMS))
    assert "full_self_px" not in MP.PARAMS and f(plain, {}) is None and f(plain, {"self_px": 10}) == 10  # (as it was)
    asked = SimpleNamespace(p={**MP.PARAMS, "full_self_px": 10})
    assert f(asked, {}) == 10 and f(asked, {"self_px": 6}) == 6  # the server's key for every full look; the look's own first
    for p in (LD.colour_long_params("net.pt"), LD.bundle(LD.colour_long_params("net.pt"))):  # no candidate names it
        assert "full_self_px" not in (p.get("server") or {})


def test_the_two_switches_of_the_ways_source_give_back_the_first_looks_things_and_drop_the_start_look():
    # ``look_start`` off: the empty hand's way is planned on the looks there are; a wall no look has shown is met,
    # looked at when the hand stalls, and crossed (as the options did before the start look came)
    start, done0, tool0 = _gate(late=True, route_empty=True, look_stall=2, **V9)
    plain, done1, tool1 = _gate(late=True, route_empty=True, look_stall=2, look_start=False, **V9)
    assert _route(start)[0]["why"] == "start" and done0 is not None and done1 is not None
    assert [r["why"] for r in _route(plain)][0] == "stall" and "start" not in [r["why"] for r in _route(plain)]
    assert tool1.n_route == tool0.n_route  # (the stall look takes the start look's place here: one look each)

    # ``look_route_only`` off: the way reads the first look's things and the latest look's, not the way's look alone
    class Sourced(Things):
        extra_route = {"blobs": []}  # (a look for the way was had)

        def tall_unexplained(self, min_h_m=0.08, clear_xy=(), clear_seg=None, held_ids=(), route=False):
            self.routes = getattr(self, "routes", []) + [bool(route)]
            return super().tall_unexplained(min_h_m, clear_xy, clear_seg, held_ids)

    def run(**opts):
        lib = _lib()
        world = GateWorld(lib)
        tool = Sourced(lib, world, [], [_box(GateWorld.X, (-0.25, 0.25), (0.90, GateWorld.TOP))], False)
        agent, done = _go(world, tool, route_empty=True, look_stall=2, **V9, **opts)
        return tool, done

    only, done2 = run()
    both, done3 = run(look_route_only=False)
    assert done2 is not None and done3 is not None and any(only.routes) and both.routes and not any(both.routes)
    assert AG.AgentParams().look_route_only is True and AG.AgentParams().look_start is True  # (round 2 as it ran)


def test_a_pixel_the_robots_model_covers_stays_in_the_look_where_something_stands_in_front_of_it():
    # ``self_front_m`` (2026-11-18): the model is drawn whole, hidden parts included; a pixel it covers is the
    # robot's own only where the frame's depth is not clearly nearer than the model drawn there
    from types import SimpleNamespace

    from scipy.ndimage import binary_dilation

    from rowant_lab.libero_env import lfb_episode as LE
    from rowant_lab.libero_env import m2_perceive as MP
    from rowant_lab.pro2 import long_dev as LD

    class Panda:
        @staticmethod
        def fk(joint_pos, gripper_qpos):
            return {"hand": (np.eye(3), np.zeros(3))}

    # two samples 1 m from the camera (pixels (32, 32) and (32, 42)) and a part 2 m away that the scene hides (32, 34)
    wc = SimpleNamespace(panda=Panda(), base=np.zeros(3),
                         robot_pts={"hand": np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.04, 0.0, 2.0]])})
    depth = np.full((64, 64), 1060, np.uint16)  # everything 6 cm behind the near samples: the arm, read a little far
    depth[32, 42] = 700  # a thing 30 cm in front of the second sample
    depth[31, 42] = 960  # and a pixel 4 cm in front of it: still the arm
    fr = {"K": [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]], "T_world_from_camera": np.eye(4),
          "depth_u16": depth, "depth_scale_mm": 1.0}

    # the model's depth, pixel by pixel: the nearest sample wins, and it grows exactly as the mask grows
    zg, z0 = LE.self_depth(wc, fr, None, None, 3)
    assert z0[32, 32] == 1.0 and z0[32, 42] == 1.0 and z0[32, 34] == 2.0 and np.isfinite(z0).sum() == 3
    assert zg[32, 34] == 1.0 and zg[32, 35] == 1.0 and zg[29, 32] == 1.0  # (a grown pixel takes the nearest depth)
    assert zg[32, 37] == 2.0 and zg[30, 35] == 2.0  # (reached from the hidden part alone)
    old = LE.self_pixels(wc, fr, None, None, 3)
    assert np.array_equal(np.isfinite(zg), old)
    rng = np.random.default_rng(3)
    for grow in (0, 1, 4, 10):  # the same on scattered samples, the picture's border included
        P = np.stack([rng.uniform(-0.4, 0.4, 60), rng.uniform(-0.4, 0.4, 60), rng.uniform(0.8, 1.6, 60)], 1)
        w2 = SimpleNamespace(panda=Panda(), base=np.zeros(3), robot_pts={"hand": P})
        g, plain = LE.self_depth(w2, fr, None, None, grow)
        m = np.isfinite(plain)
        assert np.array_equal(np.isfinite(g), binary_dilation(m, iterations=grow) if grow else m)
        assert np.array_equal(LE.self_pixels(w2, fr, None, None, grow), np.isfinite(g))  # (no ``front_m``: as it was)

    # the mask with the check: what stands in front stays (the thing, and the scene before the hidden part); the arm
    # read 4 cm near or 6 cm far does not
    hidden = zg == 2.0
    front = hidden.copy()
    front[32, 42] = True
    own = LE.self_pixels(wc, fr, None, None, 3, front_m=0.10)
    assert hidden.sum() > 5 and np.array_equal(own, old & ~front)
    assert not own[32, 42] and own[31, 42] and own[32, 32] and own[32, 35] and not own[32, 37]
    assert not LE.self_pixels(wc, fr, None, None, 3, front_m=0.03)[31, 42]  # (a smaller number keeps that pixel too)
    gone = dict(fr, depth_u16=np.where(depth == 700, 0, depth).astype(np.uint16))
    assert LE.self_pixels(wc, gone, None, None, 3, front_m=0.10)[32, 42]  # no depth there: nothing is seen in front

    # the server: absent, the look is the look of ``self_px`` as it was; given, what stands in front keeps its depth
    per = SimpleNamespace(p=dict(MP.PARAMS), _frame_arrays=MP.Perceiver._frame_arrays)
    was, md0 = MP.Perceiver._mask_depth(per, wc, fr, None, None, self_px=3)
    assert was[32, 42] == 0 and was[32, 37] == 0 and "front_px" not in md0 and "picture_dz" not in md0
    asked = SimpleNamespace(p={**MP.PARAMS, "self_front_m": 0.10}, _frame_arrays=MP.Perceiver._frame_arrays)
    out, md1 = MP.Perceiver._mask_depth(asked, wc, fr, None, None, self_px=3)
    assert out[32, 42] == 700 and out[32, 37] == 1060 and out[31, 42] == 0 and out[32, 32] == 0
    assert md1["front_px"] == int(front.sum()) and md1["picture_px"] == md0["picture_px"] - md1["front_px"]
    assert md1["robot_px"] == md0["robot_px"] - md1["front_px"]
    assert np.array_equal(np.where(front, 0, out), was)  # nothing else differs
    plain, md2 = MP.Perceiver._mask_depth(asked, wc, fr, None, None)  # a look that does not ask for the picture
    assert plain[32, 32] == 1060 and "front_px" not in md2 and "picture_px" not in md2

    # the probe changes no pixel and counts, by how far in front of the model the frame reads
    probed = SimpleNamespace(p={**MP.PARAMS, "self_px_probe": True}, _frame_arrays=MP.Perceiver._frame_arrays)
    same, md3 = MP.Perceiver._mask_depth(probed, wc, fr, None, None, self_px=3)
    dz = md3["picture_dz"]
    assert np.array_equal(same, was) and "front_px" not in md3 and md3["picture_px"] == md0["picture_px"]
    assert dz["edges_m"] == list(MP.SELF_DZ_EDGES) and len(dz["model"]) == len(dz["rim"]) == len(MP.SELF_DZ_EDGES) + 1
    assert sum(dz["model"]) + sum(dz["rim"]) == md0["picture_px"] and sum(dz["model"]) == 3
    assert dz["model"][3] == 2 and dz["model"][9] == 1  # two samples' pixels 6 cm behind; the thing 30 cm in front
    assert dz["rim"][6] == 1 and dz["rim"][10] == int(hidden.sum())  # the pixel 4 cm in front; the hidden part's rim
    both = SimpleNamespace(p={**MP.PARAMS, "self_px_probe": True, "self_front_m": 0.10},
                           _frame_arrays=MP.Perceiver._frame_arrays)
    out2, md4 = MP.Perceiver._mask_depth(both, wc, fr, None, None, self_px=3)
    assert np.array_equal(out2, out) and md4["picture_dz"] == dz and md4["front_px"] == md1["front_px"]

    # no candidate names either key, and the server's defaults do not
    assert "self_front_m" not in MP.PARAMS and "self_px_probe" not in MP.PARAMS
    for p in (LD.colour_long_params("net.pt"), LD.bundle(LD.colour_long_params("net.pt"))):
        assert not {"self_front_m", "self_px_probe"} & set(p.get("server") or {})


def test_the_learners_first_look_passes_the_servers_front_check_to_the_picture_mask():
    import inspect

    from rowant_lab.libero_env import lfb_episode as LE

    src = inspect.getsource(LE)
    assert src.count('front_m=per.p.get("self_front_m")') == 1  # absent: None, and the mask is the mask as it was
    assert inspect.signature(LE.self_pixels).parameters["front_m"].default is None
