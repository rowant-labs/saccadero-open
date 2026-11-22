"""2026-11-16, ``AgentParams.c3_measured`` (off by default): a "placed" verdict needs a look that measured the object.

The look tool's two helpers on its own records, and the agent run whole on the kinematic stand-in of
``test_pro2_steps_run`` with a tool that goes blind when the hand lets go (every track refused: the stored pose stays
the hand's prediction)."""

import numpy as np
import pytest

from rowant_lab.libero_env.m2_kin import mat_to_quat
from rowant_lab.libero_env.tests import test_pro2_steps_run as SR
from rowant_lab.libero_env.tests.test_close_look import _cam, _lib
from rowant_lab.pro2.robot import agent as AG
from rowant_lab.pro2.robot import look as LK
from rowant_lab.pro2.robot import sequence as SQ


# ---- the look tool
def _p(pid, typ, pos):
    return {"id": pid, "type": typ, "gid": "g", "joints": {},
            "bodies": {"main": {"pos": list(pos), "quat_wxyz": mat_to_quat(np.eye(3))}}}


class Scene:
    def __init__(self, fresh):
        self.fresh, self.calls = fresh, 0

    def call(self, msg):
        self.calls += 1
        return {"percepts": [dict(q) for q in self.fresh], "times": {}}


def _tool(fresh=()):
    lib = _lib()
    lib.export_types = {t: {**lib.export_types["can"], "kind": "object"} for t in ("can", "tray")}
    t = LK.PerceptionTool(lib, Scene(list(fresh)), _cam, "libero_object", "ep")
    t.obs = {"robot0_joint_pos": np.zeros(7), "robot0_gripper_qpos": np.asarray((0.04, -0.04), float),
             "robot0_eef_pos": np.array([0.3, 0.0, 1.2]), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0])}
    t.percepts = [_p("p0", "can", (0.30, 0.00, 0.95)), _p("p1", "tray", (0.30, 0.00, 0.90)), _p("p2", "can", (0.6, 0.3, 0.9))]
    t.target, t.calls = "p0", 1
    return t


def test_only_a_found_pose_or_an_accepted_track_measures_the_target():
    t = _tool()
    t.looks = [{"t": 5, "mode": "track", "status": "tracked"},  # before the release: not asked about
               {"t": 90, "mode": "full", "merge": True, "assoc": {"p0": None, "p1": 0.001, "p2": 0.0}},
               {"t": 90, "mode": "track", "status": "predicted", "jump_m": 0.09},
               {"t": 91, "mode": "track", "status": "untrackable"},
               {"t": 99, "mode": "full", "merge": True, "assoc": {"p1": 0.0}, "relook_place": True}]
    assert t.target_measured(1) is False  # not found, refused, cannot be tracked, a look for the receiver alone
    assert t.target_measured(0) is True
    for one in ({"mode": "full", "merge": True, "assoc": {"p0": 0.12}}, {"mode": "full", "merge": True, "assoc": {"p0": -1.0}},
                {"mode": "track", "status": "tracked", "jump_m": 0.004}):
        t.looks = t.looks[:5] + [one]
        assert t.target_measured(1) is True
    t.held_jump_max_m = 0.015  # a track not believed while the hand held the object is no measurement
    t.looks = t.looks[:5] + [{"mode": "track", "status": "tracked", "held_off_m": 0.04}]
    assert t.target_measured(1) is False
    t.looks[-1]["held_off_m"] = 0.004
    assert t.target_measured(1) is True


def test_the_recheck_searches_for_the_target_alone_at_any_distance():
    fresh = [_p("q0", "can", (0.05, -0.20, 0.90)), _p("q1", "tray", (0.36, 0.05, 0.90)), _p("q2", "can", (0.6, 0.3, 0.9))]
    t = _tool(fresh)
    assert t.recheck_target() is True  # the can on the table 32 cm away: no other can claims it
    by = {p["id"]: p["bodies"]["main"]["pos"] for p in t.percepts}
    assert by["p0"] == [0.05, -0.20, 0.90] and by["p1"] == [0.30, 0.00, 0.90] and by["p2"] == [0.6, 0.3, 0.9]
    assert t.looks[-1]["recheck_target"] and t.target_measured(len(t.looks) - 1)
    t = _tool([fresh[1], fresh[2]])  # nothing of its type but the other can, which is where it was
    assert t.recheck_target() is False and t.percepts[0]["bodies"]["main"]["pos"] == [0.30, 0.00, 0.95]
    assert not t.target_measured(len(t.looks) - 1)
    t = _tool(fresh)
    t.n_full = t.budget  # the look budget is spent: no look
    assert t.recheck_target() is False and t.client.calls == 0


def test_a_blinded_look_leaves_out_one_object_and_nothing_else():
    fresh = [_p("q0", "can", (0.30, 0.00, 0.95)), _p("q1", "tray", (0.30, 0.00, 0.90)), _p("q2", "can", (0.6, 0.3, 0.9))]
    t = _tool(fresh)
    t.percepts, t.target = [], None
    assert [q["id"] for q in t.full_look()["percepts"]] == ["q0", "q1", "q2"]  # off: the look as the server gave it
    t = _tool(fresh)
    t.percepts, t.target = [], None
    t.blind = {"type": "can", "xy": [0.31, 0.01], "within_m": 0.06}
    assert [q["id"] for q in t.full_look()["percepts"]] == ["q1", "q2"]  # the can there, not the tray under it
    assert [p["type"] for p in t.percepts] == ["tray", "can"]
    rep = t._blinded({"percepts": [], "extra": {"rejected": [{"type": "can", "pos": [0.3, 0.0, 0.95]},
                                                              {"type": "can", "pos": [0.6, 0.3, 0.9]}], "blobs": [1]}})
    assert rep["extra"] == {"rejected": [{"type": "can", "pos": [0.6, 0.3, 0.9]}], "blobs": [1]}
    t.blind = {"type": "no_such_type", "xy": [0.0, 0.0]}  # a bad record blinds nothing
    assert len(t.full_look()["percepts"]) == 3


# ---- the agent, run whole
class BlindTool(SR.Tool):
    """The truth until the hand lets go; from then on every track is refused (the bowl's stored pose stays where the
    hand had it) until ``recheck_target`` finds it (``finds``). The receiver is believed ``plate_err`` from where it
    stands."""

    def __init__(self, lib, world, blind=True, finds=True, plate_err=(0.0, 0.0, 0.0)):
        super().__init__(lib, world)
        self.blind, self.finds, self.plate_err = blind, finds, np.asarray(plate_err, float)
        self.looks, self.target, self.held_jump_max_m = [], "p0", 0.0
        self.frozen, self.seen_drop, self.n_recheck = None, None, 0

    def percepts(self):
        ps = super().percepts()
        ps[1]["bodies"]["main"]["pos"] = list(SR.PLATE + self.plate_err)
        if self.frozen is not None:
            ps[0]["bodies"]["main"]["pos"] = list(self.frozen)
        return ps

    def __call__(self):
        w = self.world
        if self.blind and w.held is None and w.dropped_at is not None and w.dropped_at != self.seen_drop:
            # the hand has let go since the last look: the stored pose stays where the hand had it (the stand-in's
            # last step with the bowl in the hand)
            self.seen_drop = w.dropped_at
            self.frozen = next(b for _p, _R, b in reversed(w.path) if b is not None).copy()
        self.looks.append({"t": w.t, "mode": "track", "status": "predicted" if self.frozen is not None else "tracked"})
        return super().__call__()

    target_measured = LK.PerceptionTool.target_measured

    def recheck_target(self):
        self.n_recheck += 1
        if self.finds:
            self.frozen = None
        self.looks.append({"t": self.world.t, "mode": "full", "merge": True, "recheck_target": True,
                           "assoc": {"p0": 0.1 if self.finds else None}})
        return bool(self.finds)


def _run(cap=300, more_follow=False, **kw):
    from dataclasses import replace

    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.pro2.robot import make_params

    opts = {k: kw.pop(k) for k in list(kw) if k in ("c3_measured", "T_attempt")}
    lib = LB.M2Library("libero_spatial")
    world = SR.World(lib)
    tool = BlindTool(lib, world, **kw)
    ap, pp, fp = make_params(None)
    ap = replace(ap, **opts)
    spec = {"predicate": "on", "target": "p0", "place": "p1", "place_kind": "object", "place_instance": "p1",
            "instruction": "pick the akita black bowl and place it on the plate"}
    agent = AG.OracleAgent(lib, lib.scene_from_percepts(tool.percepts()), spec, tool, cap, ap, pp, fp)
    agent.more_follow = more_follow
    done_at = None
    for t in range(cap):
        world.step(agent.act(world.pobs()))
        if done_at is None and world.held is None and world.bowl[2] > 0.91 \
                and float(np.linalg.norm((world.bowl - SR.PLATE)[:2])) < 0.025:
            done_at = t + 1  # (the benchmark ends an episode here)
        if agent.finished:
            break
    return agent, world, tool, done_at


@pytest.mark.skipif(not SR.LIB_ROOT.is_dir(), reason="the object library is not built on this machine")
def test_a_placed_verdict_on_the_hands_own_prediction_is_not_taken():
    off_the_plate = (0.10, 0.0, 0.0)  # the receiver is believed 10 cm from where it stands: the bowl lands on the table
    old, w0, t0, done0 = _run(plate_err=off_the_plate)
    assert old.finish_reason == "placed" and len(old.attempts) == 1 and done0 is None  # "placed", and it is not
    assert old.attempts[0]["c3"]["pass"] and t0.n_recheck == 0
    new, w1, t1, done1 = _run(plate_err=off_the_plate, c3_measured=True)
    a = new.attempts[0]
    assert a["outcome"] == "c3_fail" and a["c3"]["unmeasured"]["inside"] and a["c3"]["measured"] and not a["c3"]["pass"]
    assert t1.n_recheck >= 1 and len(new.attempts) >= 2 and new.finish_reason != "placed"
    # the next attempt took the bowl where the look found it (on the table, 10 cm from the plate), not in mid-air
    assert abs(new.attempts[1]["frame"]["centre"][2] - new.attempts[0]["frame"]["centre"][2]) < 0.01
    assert float(np.linalg.norm(np.asarray(new.attempts[1]["frame"]["centre"][:2]) - (SR.PLATE + off_the_plate)[:2])) < 0.08
    # the object cannot be found at all: unmeasured stays unmeasured, and that is a failed attempt too
    lost, _w, t2, _d = _run(plate_err=off_the_plate, c3_measured=True, finds=False, T_attempt=400)
    assert [x["outcome"] for x in lost.attempts] == ["c3_fail"] and lost.attempts[0]["c3"]["measured"] is False
    assert lost.finish_reason == "no_budget" and t2.n_recheck == 1


@pytest.mark.skipif(not SR.LIB_ROOT.is_dir(), reason="the object library is not built on this machine")
def test_a_measured_pass_and_a_step_with_more_to_follow_are_taken_as_before():
    # the tracks measure the bowl after the release: nothing changes, to the step
    a, _w, ta, da = _run(blind=False)
    b, _w, tb, db = _run(blind=False, c3_measured=True)
    assert a.finish_reason == b.finish_reason == "placed" and da == db is not None and a.t == b.t and tb.n_recheck == 0
    assert "unmeasured" not in b.attempts[0]["c3"]
    # blind at the release, but the bowl stands on the plate and the look with the hand away finds it there
    c, wc, tc, dc = _run(c3_measured=True)
    assert c.finish_reason == "placed" and dc is not None and tc.n_recheck == 1
    assert c.attempts[0]["c3"]["measured"] and c.attempts[0]["c3"]["pass"] and c.attempts[0]["c3"]["unmeasured"]["inside"]
    # another step follows (the step sequencer says so): the unmeasured pass stands, the bowl is left where it is
    d, _w, td, _dd = _run(c3_measured=True, more_follow=True, plate_err=(0.10, 0.0, 0.0))
    assert d.finish_reason == "placed" and len(d.attempts) == 1 and td.n_recheck == 0


def test_the_sequencer_tells_each_step_whether_another_follows():
    class Step:
        finished, finish_reason = False, None

        def act(self, pobs):
            self.finished, self.finish_reason = True, "placed"
            return [0.0] * 7

        def summary(self):
            return {"finish_reason": "placed", "steps": 1, "attempts": [], "events": []}

    made = []

    def planner(clause, cap, exclude):
        made.append(Step())
        return {"agent": made[-1], "grounding": {"status": "ok", "target": f"p{len(made)}"}, "parsed": {}}

    seq = SQ.SequenceAgent(["first", "second"], planner, 400)
    assert made[0].more_follow is True
    while not seq.finished:
        seq.act({})
    assert len(made) == 2 and made[1].more_follow is False


# ---- ``detour_from_cross``: an unseen wall met on the way up
class WallWorld(SR.World):
    """An unseen wall across the way, right behind the bowl: nothing held passes under its top."""
    Y, TOP = (0.058, 0.075), 1.175

    def step(self, a):
        p_old = self.p.copy()
        super().step(a)
        if self.held is not None and self.Y[0] < self.bowl[1] < self.Y[1] and self.bowl[2] < self.TOP:
            self.p = p_old + np.array([0.0, 0.0, 2e-5 * (1 if self.t % 2 else -1)])  # (a blocked hand still trembles)
            self.bowl = self.p + self.held
            self.path[-1] = (self.p.copy(), self.R.copy(), self.bowl.copy())


class LowWallWorld(WallWorld):
    TOP = 1.14


class SeeingTool(SR.Tool):
    """The look saw the wall standing there and could not name it (``PerceptionTool.tall_unexplained``'s reply).
    ``late``: only a look taken after the carry met it shows it (it was put there after the first look)."""
    top, asked, late, relooked = LowWallWorld.TOP - 0.01, 0, False, 0
    box = None  # (centre x, half x, y range) of what the look saw; None: the wall of ``WallWorld``

    def relook_place(self, place_id):
        self.relooked += 1
        return {"moved_m": 0.0, "assoc": 0.0}

    def tall_unexplained(self, min_h_m=0.08, clear_xy=(), clear_seg=None, held_ids=()):
        self.asked += 1
        if self.late and not self.relooked:
            return []
        # (the part of it that the look saw: it ends 5 cm short of where the hand comes down at the plate)
        cx, hx, (y0, y1) = self.box or (0.18, 0.22, WallWorld.Y)
        return [(np.array([cx, 0.5 * (y0 + y1), 0.5 * (0.97 + self.top)]), np.eye(3),
                 np.array([hx, 0.5 * (y1 - y0), 0.5 * (self.top - 0.97)]))]


class EndWallWorld(SR.World):
    """The bowl 42 cm from the plate, and between them a wall too tall to cross, from x = -0.15 on towards the far
    side of the table: nothing held passes it but by its near end."""
    X0, Y, TOP = -0.15, (0.0, 0.01), 1.60

    def __init__(self, lib, knock_at=None):
        super().__init__(lib, knock_at)
        self.bowl = np.array([0.05, -0.20, 0.898])

    def step(self, a):
        p_old = self.p.copy()
        super().step(a)
        r = 0.06  # (the bowl's own half width)
        if self.held is not None and self.bowl[0] > self.X0 - r and self.Y[0] - r < self.bowl[1] < self.Y[1] + r \
                and self.bowl[2] < self.TOP:
            self.p = p_old + np.array([0.0, 0.0, 2e-5 * (1 if self.t % 2 else -1)])
            self.bowl = self.p + self.held
            self.path[-1] = (self.p.copy(), self.R.copy(), self.bowl.copy())


def _wall_run(cap=300, seen=None, world_cls=WallWorld, late=False, box=None, known=True, **opts):
    from dataclasses import replace

    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.pro2.robot import make_params

    lib = LB.M2Library("libero_spatial")
    world = world_cls(lib)
    # (a known low box: the carry has a crossing height of its own)
    tool = (SR.Tool if seen is None else SeeingTool)(lib, world, [SR.WALL] if known else [])
    if seen is not None:
        tool.top, tool.late, tool.box = seen, late, box
    ap, pp, fp = make_params(None)
    spec = {"predicate": "on", "target": "p0", "place": "p1", "place_kind": "object", "place_instance": "p1",
            "instruction": "pick the akita black bowl and place it on the plate"}
    agent = AG.OracleAgent(lib, lib.scene_from_percepts(tool.percepts()), spec, tool, cap, replace(ap, **opts), pp, fp)
    done_at = None
    for t in range(cap):
        world.step(agent.act(world.pobs()))
        if done_at is None and world.held is None and world.bowl[2] > 0.91 \
                and float(np.linalg.norm((world.bowl - SR.PLATE)[:2])) < 0.025:
            done_at = t + 1
        if agent.finished:
            break
    detour = [e["z"] for e in agent.events if e["ev"] == "detour"]
    agent.tool_asked = getattr(tool, "asked", None)
    return agent, done_at, detour


@pytest.mark.skipif(not SR.LIB_ROOT.is_dir(), reason="the object library is not built on this machine")
def test_a_detour_rises_above_the_crossing_height_also_when_the_cut_way_met_the_wall_lower():
    old, done0, d0 = _wall_run()  # the old way meets the wall at the crossing height: 18 cm more clears it
    cut, done1, d1 = _wall_run(transit_cut=True)  # the cut way meets it on the way up: the same 18 cm end lower
    new, done2, d2 = _wall_run(transit_cut=True, detour_from_cross=True)
    assert done0 is not None and old.attempts[0]["transport_status"][0] == "detour_reached"
    assert done1 is None and d1[0] < d0[0] - 0.01 and cut.attempts[0]["outcome"] == "c3_fail"  # let go short of the plate
    assert done2 is not None and d2[0] >= d0[0] and new.attempts[0]["transport_status"][0] == "detour_reached"
    same, done3, d3 = _wall_run(detour_from_cross=True)  # without the cut it is the old way, to the step
    assert done3 == done0 and d3 == d0 and same.t == old.t


@pytest.mark.skipif(not SR.LIB_ROOT.is_dir(), reason="the object library is not built on this machine")
def test_a_carry_crosses_above_what_the_look_saw_standing_tall_and_could_not_name():
    W = dict(world_cls=LowWallWorld, transit_cut=True, detour_from_cross=True)
    old, done0, d0 = _wall_run(**W)  # unseen: the carry meets it, then the detour
    assert done0 is not None and len(d0) == 1
    # the look saw it: with the option off the way is the same to the step, and the tool is not asked
    off, done1, d1 = _wall_run(seen=LowWallWorld.TOP - 0.01, **W)
    assert (done1, d1, off.t, off.tool_asked) == (done0, d0, old.t, 0)
    # on: the carry is planned over it, meets nothing, and the bowl lies on the plate sooner
    on, done2, d2 = _wall_run(seen=LowWallWorld.TOP - 0.01, transit_unexplained=True, **W)
    assert done2 is not None and d2 == [] and done2 < done0 - 10, (done0, done2)
    assert on.attempts[0]["transport_status"][0] == "reached" and on.attempts[0]["outcome"] != "c3_fail"
    assert any(e.get("unexplained") == 1 for e in on.events if e["ev"] == "transit")
    # the empty hand's way to the bowl is not changed by it
    first_goto = lambda ag: next(e for e in ag.events if e["ev"] == "goto")  # noqa: E731
    assert first_goto(on) == first_goto(old)
    # a thing that no crossing within the arm's reach clears (it stands 45 cm tall): the way as before, to the step
    tall, done3, d3 = _wall_run(seen=1.35, transit_unexplained=True, **W)
    assert (done3, d3, tall.t) == (done0, d0, old.t)
    assert any("unexplained_ignored" in e for e in tall.events if e["ev"] == "transit")
    # without the cut way it crosses above the thing too
    plain, done4, d4 = _wall_run(seen=LowWallWorld.TOP - 0.01, world_cls=LowWallWorld, transit_unexplained=True)
    assert done4 is not None and d4 == []


@pytest.mark.skipif(not SR.LIB_ROOT.is_dir(), reason="the object library is not built on this machine")
def test_a_carry_goes_around_the_near_end_of_a_thing_too_tall_to_cross():
    E = EndWallWorld
    W = dict(world_cls=E, known=False, seen=E.TOP, box=(0.10, 0.25, E.Y), transit_cut=True, detour_from_cross=True)
    # as before, and with the thing counted but no way around planned: met, a detour that does not clear it, lost
    old, done0, d0 = _wall_run(**W)
    over, done1, d1 = _wall_run(transit_unexplained=True, **W)
    assert done0 is None and done1 is None and len(d0) >= 1 and d1 == d0
    assert any("unexplained_ignored" in e for e in over.events if e["ev"] == "transit")
    # around: by the end at x = -0.15, the bowl on the plate, nothing met
    on, done2, d2 = _wall_run(transit_unexplained=True, transit_around=True, **W)
    ev = next(e for e in on.events if e["ev"] == "transit" and "around" in e)
    assert done2 is not None and d2 == [] and on.attempts[0]["transport_status"][0] == "reached"
    pts = ev["around"]  # (up, a step straight off the thing when the hand is near it, the two points, above the goal, down)
    near, far = sorted(sorted(pts, key=lambda q: q[0])[:2], key=lambda q: q[1])  # the two points beside the end
    assert near[0] < E.X0 - 0.08 and far[0] < E.X0 - 0.08 and near[1] < E.Y[0] - 0.08 and far[1] > E.Y[1] + 0.08
    assert near[2] == far[2] == pts[0][2]  # across at one height
    # the thing reaching to the near edge too (no end within reach to pass by): the way as before
    shut, done3, d3 = _wall_run(transit_unexplained=True, transit_around=True, **{**W, "box": (-0.2, 0.6, E.Y)})
    assert done3 is None and any("unexplained_ignored" in e for e in shut.events if e["ev"] == "transit")


@pytest.mark.skipif(not SR.LIB_ROOT.is_dir(), reason="the object library is not built on this machine")
def test_a_blocked_carry_is_noticed_in_a_dozen_steps_looks_again_and_goes_over_what_it_now_sees():
    W = dict(world_cls=LowWallWorld, transit_cut=True, detour_from_cross=True)
    old, done0, d0 = _wall_run(**W)

    def stop_step(ag):  # the step at which the carry's blocked move ended
        return next(e["t"] for e in ag.events if e["ev"] in ("carry_blocked", "detour"))

    # the block watcher alone: the same detour, begun sooner
    fast, done1, d1 = _wall_run(carry_block_steps=12, **W)
    assert done1 is not None and len(d1) == 1 and stop_step(fast) < stop_step(old) and done1 < done0
    assert any(e["ev"] == "carry_blocked" for e in fast.events)
    # a thing put in the way after the first look: the carry meets it, backs off, looks again, and crosses above it
    L = dict(seen=LowWallWorld.TOP - 0.01, late=True, transit_unexplained=True, carry_block_steps=12, **W)
    blind, done2, d2 = _wall_run(**L)  # no second look: met, then the detour
    assert done2 is not None and len(d2) == 1
    look, done3, d3 = _wall_run(carry_relook=True, **L)
    rl = look.attempts[0]["carry_relook"]
    assert done3 is not None and d3 == [] and rl["was"] == "blocked" and rl["looked"] and rl["status"] == "reached"
    assert rl["new"] == 1 and rl["plan"] == "over" and len(rl["counted"]) == 1
    assert any(e.get("unexplained") == 1 for e in look.events if e["ev"] == "transit")
    assert look.attempts[0]["transport_status"][0] == "reached"
    # with nothing new in the second look no second straight try is made: the detour at once, and no later than
    # without the look (the look costs the few steps of backing off)
    none, done4, d4 = _wall_run(carry_relook=True, carry_block_steps=12, **W)
    rl4 = none.attempts[0]["carry_relook"]
    assert done4 is not None and len(d4) == 1 and rl4["new"] == 0 and "status" not in rl4
    assert sum(e["ev"] == "carry_blocked" for e in none.events) == 1 and done4 <= done1 + 8
    # the detour after a second plan keeps the first plan's crossing height (a thing seen that the plan cannot use:
    # too tall to cross within reach)
    tall = _wall_run(seen=1.50, late=True, carry_relook=True, transit_unexplained=True, carry_block_steps=12, **W)
    rl5 = tall[0].attempts[0]["carry_relook"]
    assert tall[1] is not None and tall[2] == d1 and rl5["new"] == 1 and rl5["plan"] == "ignored" and "status" not in rl5
    assert sum(e["ev"] == "carry_blocked" for e in tall[0].events) == 1  # (no second straight try at it)


@pytest.mark.skipif(not SR.LIB_ROOT.is_dir(), reason="the object library is not built on this machine")
def test_what_stands_tall_without_a_name_is_boxed_by_its_top_layer_from_its_lowest_seen_point_up():
    from rowant_lab.libero_env import m2_library as LB

    lib = LB.M2Library("libero_spatial")
    t = _spatial_tool(lib, [(0.138, -0.085, 0.899)])
    t.percepts.append({"id": "p7", "type": "wooden_cabinet", "gid": "g", "joints": {}, "fit": {"n_agree": 900},
                       "source": "support", "bodies": t._build("wooden_cabinet", np.eye(3), np.array([0.032, -0.272, 0.907]))})
    # the regions of one stored look (numbers only): a slab 13 cm tall that floats 7 cm up, a low thing, a sliver on
    # the cabinet's top edge, and a region that tops out 4 cm above the plate (the receiver)
    slab = {"xy": [-0.1543, 0.2972], "half_xy": [0.0592, 0.0681], "yaw": 1.222, "z_top": 1.1002, "height": 0.1999,
            "n_px": 20719, "h_min": 0.071, "top": {"xy": [-0.102, 0.2709], "half_xy": [0.05, 0.0049], "yaw": 1.124,
                                                    "z": 1.1002, "n_px": 3614}}
    low = {"xy": [0.059, -0.1809], "half_xy": [0.0597, 0.0171], "yaw": -0.438, "z_top": 0.955, "height": 0.0547,
           "n_px": 2208, "h_min": 0.0424, "top": {"xy": [0.0785, -0.1773], "half_xy": [0.0434, 0.0053], "yaw": -0.44,
                                                   "z": 0.9549, "n_px": 2198}}
    edge = {"xy": [0.03, -0.30], "half_xy": [0.046, 0.0063], "yaw": 1.134, "z_top": 1.1269, "height": 0.2266,
            "n_px": 1540, "h_min": 0.21, "top": {"xy": [0.03, -0.30], "half_xy": [0.0437, 0.0044], "yaw": 1.134,
                                                  "z": 1.1269, "n_px": 540}}
    cover = {"xy": [0.07, 0.21], "half_xy": [0.07, 0.07], "yaw": 0.0, "z_top": 0.96, "height": 0.09, "n_px": 5000,
             "h_min": 0.0, "top": {"xy": [0.065, 0.205], "half_xy": [0.06, 0.06], "yaw": 0.0, "z": 0.96, "n_px": 4000}}
    t.extra = {"rejected": [], "blobs": [slab, low, edge, cover, {**slab, "top": None}, {**slab, "height": 0.07}]}
    got = t.tall_unexplained(0.08, clear_xy=([0.138, -0.085],))
    assert len(got) == 1
    c, R, h = got[0]
    assert np.allclose(c, [-0.102, 0.2709, 0.5 * (0.9713 + 1.1002)], atol=1e-4)
    assert np.allclose(h, [0.05, 0.01, 0.5 * (1.1002 - 0.9713)], atol=1e-4)  # the top layer's own size, 1 cm thick at the least
    # the region's own box is the patch the thing hides, on the colour line a metre long and into the arm: not used
    smear = {**slab, "xy": [-0.34, 0.15], "half_xy": [0.54, 0.14], "yaw": 0.15}
    t.extra = {"rejected": [], "blobs": [smear]}
    c2, _R2, h2 = t.tall_unexplained(0.08)[0]
    assert np.allclose(c2[:2], [-0.102, 0.2709], atol=1e-4) and np.allclose(h2[:2], [0.05, 0.01], atol=1e-4)
    # a top layer 70 cm long is no thing to cross at one place; a thin slice floating 37 cm up is a cut of the arm
    t.extra = {"rejected": [], "blobs": [{**slab, "top": {**slab["top"], "half_xy": [0.35, 0.02]}},
                                         {**slab, "z_top": 1.312, "height": 0.406, "h_min": 0.37}]}
    assert t.tall_unexplained(0.08) == []
    t.extra = {"rejected": [], "blobs": [slab, low, edge, cover, {**slab, "top": None}, {**slab, "height": 0.07}]}
    assert len(t.tall_unexplained(0.13)) == 1 and t.tall_unexplained(0.25) == []  # (the height from which a thing counts)
    assert np.allclose(R[:, 0], [np.cos(1.124), np.sin(1.124), 0.0]) and np.allclose(R[2], [0.0, 0.0, 1.0])
    assert len(t.tall_unexplained(0.08)) == 1  # (a region that tops out at a reported percept's top is that percept)
    # a thing standing 15 cm tall on the plate's rim is no part of the plate: it counts
    on_rim = {**cover, "z_top": 1.07, "height": 0.17}
    t.extra = {"rejected": [], "blobs": [on_rim]}
    assert len(t.tall_unexplained(0.08)) == 1
    t.extra = {"rejected": [], "blobs": [slab, low, edge, cover]}
    # the region the carried object itself was read from (a model set on it, lifted away since) is no obstacle
    t.percepts = [q for q in t.percepts if q["id"] != "p0"]
    own = {**slab, "top": {**slab["top"], "xy": [0.138, -0.085], "half_xy": [0.05, 0.05]}}
    t.extra = {"rejected": [], "blobs": [own]}
    assert len(t.tall_unexplained(0.08)) == 1 and t.tall_unexplained(0.08, clear_xy=([0.20, -0.14],)) == []
    assert len(t.tall_unexplained(0.08, clear_xy=([0.25, -0.085],))) == 1  # 11 cm from its middle: not this region
    # a thing put there after the first look is in the latest look's regions alone; one seen by both counts once
    t.extra, t.extra_last = {"rejected": [], "blobs": []}, {"blobs": [slab]}
    assert len(t.tall_unexplained(0.08)) == 1
    t.extra = {"rejected": [], "blobs": [{**slab, "top": {**slab["top"], "xy": [-0.11, 0.28]}}]}
    assert len(t.tall_unexplained(0.08)) == 1
    # the robot's own arm in a look taken with the hand in the scene: what lies along it is left out
    assert t.tall_unexplained(0.08, clear_seg=([-0.66, 0.0], [-0.05, 0.25], 0.12)) == []
    assert len(t.tall_unexplained(0.08, clear_seg=([-0.66, 0.0], [-0.30, 0.15], 0.12))) == 1
    # what the load in the hand rests against stands within the held object's own footprint: it counts
    t.percepts.append({"id": "p0", "type": "akita_black_bowl", "gid": "g", "joints": {}, "fit": {}, "source": "texture",
                       "bodies": t._build("akita_black_bowl", np.eye(3), np.array([-0.10, 0.21, 1.0]))})
    assert t.tall_unexplained(0.08) == [] and len(t.tall_unexplained(0.08, held_ids=("p0",))) == 1
    t.extra, t.extra_last = {}, None
    assert t.tall_unexplained() == []


# ---- ``params["relation_look"]``: a look from above at the relation's place before a wrong object is carried
def _spatial_tool(lib, bowls):
    t = LK.PerceptionTool(lib, None, None, "libero_spatial", "ep")
    t.table_bound = True
    ps = [("p9", "plate", (0.06, 0.20, 0.902)), ("p8", "glazed_rim_porcelain_ramekin", (-0.20, 0.19, 0.900))]
    ps += [(f"p{i}", "akita_black_bowl", b) for i, b in enumerate(bowls)]
    t.percepts = [{"id": i, "type": typ, "gid": "g", "joints": {}, "fit": {"n_agree": 3000, "inlier": 0.95},
                   "source": "texture", "bodies": t._build(typ, np.eye(3), np.asarray(pos, float))} for i, typ, pos in ps]
    t.extra = {"rejected": [], "blobs": []}
    return t


@pytest.mark.skipif(not SR.LIB_ROOT.is_dir(), reason="the object library is not built on this machine")
def test_the_one_bowl_in_view_fails_the_relation_so_the_robot_looks_where_the_named_one_should_be():
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.pro2.robot import episode as EP

    lib = LB.M2Library("libero_spatial")
    ins = "pick the akita black bowl next to the ramekin and place it on the plate"
    far, near = (0.127, -0.074, 0.898), (-0.187, 0.324, 0.898)  # the other bowl, and where the named one stands
    base = {"relation_single": True, "fallback_on_table": True}
    # off: the only bowl in view is taken, as before
    t = _spatial_tool(lib, [far])
    out = EP._plan_one(ins, t, lib, 220, base)
    assert type(out["agent"]).__name__ == "OracleAgent" and out["grounding"]["target"] == "p0"
    # on: a look first, above the ramekin (the relation's place), then the instruction
    t = _spatial_tool(lib, [far])
    out = EP._plan_one(ins, t, lib, 220, {**base, "relation_look": True})
    seq = out["agent"]
    assert type(seq).__name__ == "SequenceAgent" and out["grounding"]["flags"] == ["look_at"]
    assert np.allclose(out["grounding"]["point"], [-0.20, 0.19], atol=0.005) and type(seq.cur).__name__ == "LookAgent"
    assert seq.cur.more_follow is True
    seen = []

    def wrist_find(typ, support_z, hint_xy=None, opts=None):
        seen.append((typ, round(float(support_z), 3), [round(float(x), 3) for x in hint_xy], opts))
        return {"xy": [near[0], near[1]], "yaw": 0.0, "half": [0.053, 0.053], "area_m2": 0.009, "n_cands": 1,
                "n_cells": 400, "support_z": float(support_z)}

    t.wrist_find = wrist_find
    world = SR.World(lib)
    for _ in range(120):
        world.step(seq.act(world.pobs()))
        if type(seq.cur).__name__ != "LookAgent":
            break
    assert seen and seen[0][0] == "akita_black_bowl" and abs(seen[0][1] - 0.9) < 0.01
    # two pictures: the place by the thing's size, then from straight above the find, the thing nearest it
    assert [x[3] for x in seen] == [None, {"prefer": "near"}] and seen[1][2] == [round(near[0], 3), round(near[1], 3)]
    assert t.wrist_opts["prefer"] == "size"
    # the thing found is the second candidate, and the relation chooses it; the hand has not touched the other bowl
    assert type(seq.cur).__name__ == "OracleAgent" and seq.cur.spec["target"] == "w1" and seq.cur.more_follow is False
    w1 = next(p for p in t.percepts if p["id"] == "w1")
    assert np.allclose(w1["bodies"]["main"]["pos"][:2], near[:2], atol=1e-3) and w1["source"] == "wrist"
    assert world.t < 80
    # nothing found: the plan is what it was without the look
    t = _spatial_tool(lib, [far])
    t.wrist_find = lambda typ, support_z, hint_xy=None, opts=None: None
    seq = EP._plan_one(ins, t, lib, 220, {**base, "relation_look": True})["agent"]
    world = SR.World(lib)
    for _ in range(120):
        world.step(seq.act(world.pobs()))
        if type(seq.cur).__name__ != "LookAgent":
            break
    assert type(seq.cur).__name__ == "OracleAgent" and seq.cur.spec["target"] == "p0"
    # both bowls in view, or the one in view stands in the relation: no look
    for bowls in ([far, near], [near]):
        t = _spatial_tool(lib, bowls)
        out = EP._plan_one(ins, t, lib, 220, {**base, "relation_look": True})
        assert type(out["agent"]).__name__ == "OracleAgent"
        tgt = next(p for p in t.percepts if p["id"] == out["grounding"]["target"])
        assert np.allclose(tgt["bodies"]["main"]["pos"][:2], near[:2])


@pytest.mark.skipif(not SR.LIB_ROOT.is_dir(), reason="the object library is not built on this machine")
def test_a_find_counts_only_when_a_picture_from_above_it_shows_a_thing_at_the_same_place():
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.pro2.robot import episode as EP

    lib = LB.M2Library("libero_spatial")
    ins = "pick the akita black bowl next to the ramekin and place it on the plate"
    far, near = (0.127, -0.074, 0.898), (-0.195, 0.315, 0.898)
    slant, slab = (-0.038, 0.331), (-0.121, 0.274)  # a slab's slanted image in the first picture, and where it stands
    P = {"relation_single": True, "fallback_on_table": True, "relation_look": True}

    def cand(xy, area):
        return {"xy": list(xy), "yaw": 0.0, "half": [0.05, 0.05], "area_m2": area, "n_cells": 400, "edge": False,
                "rgb": [30, 30, 30]}

    def run(above):
        """``above``: what a picture from straight above a place shows nearest it ({place: find or None})."""
        t = _spatial_tool(lib, [far])
        asked = []

        def wrist_find(typ, support_z, hint_xy=None, opts=None):
            asked.append(([round(float(x), 3) for x in hint_xy], opts))
            if opts is None:  # the first picture: both things, the slab's image the nearer in size
                r = [cand(slant, 0.0095), cand(near[:2], 0.0080)]
                return {**r[0], "n_cands": 2, "ranked": r, "support_z": float(support_z)}
            got = above.get(tuple(round(float(x), 3) for x in hint_xy))
            return None if got is None else {**got, "n_cands": 1, "support_z": float(support_z)}

        t.wrist_find = wrist_find
        seq = EP._plan_one(ins, t, lib, 220, P)["agent"]
        look = seq.cur
        world = SR.World(lib)
        for _ in range(160):
            world.step(seq.act(world.pobs()))
            if type(seq.cur).__name__ != "LookAgent":
                break
        return t, seq, look, asked, world

    # from above the slab's image the nearest thing is the slab itself, 10 cm away: not the same thing. The next
    # thing of the first picture is tried, and from above it the bowl is where the first picture had it
    t, seq, look, asked, world = run({slant: cand(slab, 0.0018), near[:2]: cand((near[0] + 0.004, near[1] - 0.003), 0.0085)})
    assert [a[1] for a in asked] == [None, {"prefer": "near"}, {"prefer": "near"}]
    assert asked[1][0] == list(slant) and asked[2][0] == [round(near[0], 3), round(near[1], 3)]
    tries = look.events[0]["tries"]
    assert [x["moved_m"] > 0.06 for x in tries] == [True, False] and look.finish_reason == "found"
    assert type(seq.cur).__name__ == "OracleAgent" and seq.cur.spec["target"] == "w1" and world.t < 110
    w1 = next(q for q in t.percepts if q["id"] == "w1")
    assert np.allclose(w1["bodies"]["main"]["pos"][:2], [near[0] + 0.004, near[1] - 0.003], atol=1e-3)
    # nothing is the same from above: nothing was found, and the plan is what it was without the look
    t, seq, look, asked, _w = run({slant: cand(slab, 0.0018), near[:2]: None})
    assert look.finish_reason == "not_found" and len(asked) == 3
    assert type(seq.cur).__name__ == "OracleAgent" and seq.cur.spec["target"] == "p0"
    assert not any(q["id"] == "w1" for q in t.percepts)


@pytest.mark.skipif(not SR.LIB_ROOT.is_dir(), reason="the object library is not built on this machine")
def test_the_wrist_look_compares_things_with_the_bowls_outline_seen_from_above_not_with_its_box():
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.libero_env.m2_kin import mat_to_quat
    from rowant_lab.pro2.robot import grasp as GR
    from rowant_lab.pro2.robot import wrist2d as W2

    lib = LB.M2Library("libero_spatial")
    t = _spatial_tool(lib, [(0.127, -0.074, 0.898)])
    top = t._top_area("akita_black_bowl")
    assert 0.74 < top / (0.1072 * 0.107) < 0.82 and t._top_area("no_such_type") is None
    # the hand top-down with its camera above the ramekin, 27 cm over the table; a pale table
    R = GR.hand_matrix([0.0, 1.0, 0.0])
    q = mat_to_quat(R)
    hand = np.array([-0.20, 0.19, 1.17]) - R @ np.asarray(W2.WRIST["t_hand_cam"]) * np.array([1.0, 1.0, 0.0])
    n = 256
    rgb = np.full((n, n, 3), 200, np.uint8)
    K, T = W2.camera(hand, R, n)
    mid_z = 0.90 + 0.5 * 0.0505

    def paint(xy, half, colour):
        uv, _z = W2.project(K, T, [[xy[0] - half[0], xy[1] - half[1], mid_z], [xy[0] + half[0], xy[1] + half[1], mid_z]])
        u0, u1 = sorted((int(round(uv[0][0])), int(round(uv[1][0]))))
        v0, v1 = sorted((int(round(uv[0][1])), int(round(uv[1][1]))))
        rgb[max(v0, 0):v1, max(u0, 0):u1] = colour

    bowl, image = (-0.195, 0.315), (-0.34, 0.30)  # (the hand itself hides the other side of the picture)
    paint(bowl, (0.045, 0.045), (25, 25, 25))        # 0.0081 m2: what a bowl shows from above
    paint(image, (0.052, 0.052), (40, 50, 60))       # 0.0108 m2: a slab's slanted image, nearer the box's 0.0115
    t.cam = type("Cam", (), {"colour": lambda self, name: rgb})()
    t.obs = {"robot0_eef_pos": hand, "robot0_eef_quat": [q[1], q[2], q[3], q[0]], "robot0_gripper_qpos": [0.04, -0.04]}
    t.wrist_opts = {"prefer": "size", "within_m": 0.25}
    got = t.wrist_find("akita_black_bowl", 0.90, hint_xy=(-0.20, 0.19))
    assert got["n_cands"] == 2 and np.allclose(got["xy"], bowl, atol=0.01)
    assert [np.allclose(c["xy"], x, atol=0.012) for c, x in zip(got["ranked"], (bowl, image))] == [True, True]
    log = t.looks[-1]
    assert abs(log["expect_m2"] - top) < 1e-5 and len(log["ranked"]) == 2 and log["ranked"][0][2] < log["ranked"][1][2]
    # with the box's area the image would have been the first (as it was on the Mac)
    a_bowl, a_image = got["ranked"][0]["area_m2"], got["ranked"][1]["area_m2"]
    box = 0.1072 * 0.107
    assert abs(np.log(a_image / box)) < abs(np.log(a_bowl / box)) and abs(np.log(a_bowl / top)) < abs(np.log(a_image / top))
    # the confirming picture asks for the thing nearest a place, whatever its size, and logs no ranking
    got = t.wrist_find("akita_black_bowl", 0.90, hint_xy=image, opts={"prefer": "near"})
    assert np.allclose(got["xy"], image, atol=0.012) and "ranked" not in got and "ranked" not in t.looks[-1]


@pytest.mark.skipif(not SR.LIB_ROOT.is_dir(), reason="the object library is not built on this machine")
def test_no_look_at_a_place_that_stands_high_or_on_a_fixture_that_is_only_a_stand_in():
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.pro2.robot import episode as EP

    lib = LB.M2Library("libero_spatial")
    far = (0.127, -0.074, 0.898)
    P = {"relation_single": True, "fallback_on_table": True, "relation_look": True}

    def plan(ins, extra=(), rejected=()):
        t = _spatial_tool(lib, [far])
        for i, (typ, pos) in enumerate(extra):
            t.percepts.append({"id": f"p2{i}", "type": typ, "gid": "g", "joints": {}, "fit": {"n_agree": 3000, "inlier": 0.95},
                               "source": "texture", "bodies": t._build(typ, np.eye(3), np.asarray(pos, float))})
        t.extra = {"rejected": [{"id": f"p10{i}", "type": typ, "gid": "g", "joints": {}, "fit": {"n_agree": 4000, "inlier": 0.99},
                                 "source": "support", "pos": list(pos), "why": {"accepted": False, "upright_err_deg": 0.0, **why},
                                 "bodies": t._build(typ, np.eye(3), np.asarray(pos, float))}
                                for i, (typ, pos, why) in enumerate(rejected)], "blobs": []}
        return t, EP._plan_one(ins, t, lib, 220, P)

    on_box = "pick the akita black bowl on the cookies box and place it on the plate"
    # the box on the table: the look is from 27 cm above its top, within reach
    _t, out = plan(on_box, extra=[("cookies", (0.06, 0.04, 0.91))])
    assert type(out["agent"]).__name__ == "SequenceAgent" and out["grounding"]["flags"] == ["look_at"]
    # the box set a quarter of a metre up (on a cabinet): out of reach, no look, the plan as without the option
    _t, out = plan(on_box, extra=[("cookies", (0.04, -0.27, 1.13))])
    assert type(out["agent"]).__name__ == "OracleAgent" and out["grounding"]["target"] == "p0"
    # "on the stove", the stove not reported and a rejected pose standing in for it: a fixture's top, no look
    on_stove = "pick the akita black bowl on the stove and place it on the plate"
    t, out = plan(on_stove, rejected=[("flat_stove", (-0.415, -0.15, 0.906), {"vis_share": 0.6, "share": 0.99})])
    assert [q["type"] for q in t.landmark_fb] == ["flat_stove"]
    assert type(out["agent"]).__name__ == "OracleAgent" and out["grounding"]["target"] == "p0"


# ---- ``params["keep_target"]``: the object an instruction names does not change when it is planned again
@pytest.mark.skipif(not SR.LIB_ROOT.is_dir(), reason="the object library is not built on this machine")
def test_planned_again_the_target_is_the_object_the_first_plan_chose():
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.pro2.robot import episode as EP

    lib = LB.M2Library("libero_spatial")
    ins = "pick the akita black bowl next to the ramekin and place it on the plate"
    far, near = (0.127, -0.074, 0.898), (-0.187, 0.324, 0.898)
    P = {"relation_single": True, "fallback_on_table": True, "keep_target": True}

    def first():
        t = _spatial_tool(lib, [far, near])
        out = EP._plan_one(ins, t, lib, 220, P)
        assert out["grounding"]["target"] == "p1"  # the bowl next to the ramekin
        return t

    def again(t0, bowls, ramekin=True, n_agree=(3000, 3000)):
        """A new look tool as the runner makes it when it plans again: the first tool's memory handed over."""
        t = _spatial_tool(lib, bowls)
        if not ramekin:
            t.percepts = [p for p in t.percepts if p["type"] != "glazed_rim_porcelain_ramekin"]
        for p, n in zip([p for p in t.percepts if p["type"] == "akita_black_bowl"], n_agree):
            p["fit"] = {"n_agree": n, "inlier": 0.95}
        t.prior = dict(t0.prior)
        return t

    t0 = first()
    assert t0.prior[ins]["pos"][:2] == list(near[:2]) and t0.prior[ins]["others"] == [list(far)]
    # the hand hides the target and the landmark: only the other bowl is in the new look
    t = again(t0, [far], ramekin=False)
    out = EP._plan_one(ins, t, lib, 180, P)
    g = out["grounding"]
    assert g["target"] == "m1" and "kept_target" in g["flags"] and t.percepts[-1]["source"] == "remembered"
    assert np.allclose(t.percepts[-1]["bodies"]["main"]["pos"], near)
    assert {"role": "target", "type": "akita_black_bowl", "id": "m1", "source": "remembered", "held": False} in t.fallbacks
    off = EP._plan_one(ins, again(t0, [far], ramekin=False), lib, 180, {k: v for k, v in P.items() if k != "keep_target"})
    assert off["grounding"]["target"] == "p0"  # off: the other bowl, as before
    # both bowls in view, the landmark not, and the other bowl the stronger fit: still the first choice
    t = again(t0, [far, near], ramekin=False, n_agree=(4000, 2000))
    assert EP._plan_one(ins, t, lib, 180, P)["grounding"]["target"] == "p1"
    t = again(t0, [far, near], ramekin=False, n_agree=(4000, 2000))
    assert EP._plan_one(ins, t, lib, 180, {k: v for k, v in P.items() if k != "keep_target"})["grounding"]["target"] == "p0"
    # the target was knocked 20 cm away and lies there: the one bowl that is not the remembered other one
    t = again(t0, [far, (0.0, 0.25, 0.898)], ramekin=False, n_agree=(4000, 2000))
    g = EP._plan_one(ins, t, lib, 180, P)["grounding"]
    assert g["target"] == "p1" and "kept_target" in g["flags"]
    # another instruction, or no memory: the plan is its own
    t = again(t0, [far, near])
    t.prior = {}
    assert EP._plan_one(ins, t, lib, 180, P)["grounding"]["target"] == "p1" and ins in t.prior
    # the option off: nothing is remembered
    t = _spatial_tool(lib, [far, near])
    EP._plan_one(ins, t, lib, 220, {"relation_single": True, "fallback_on_table": True})
    assert t.prior == {}


# ---- the carry options with the real look tool in the loop (the stand-in of ``probes/kin_same.py``: a can, a basket)
def _slab_cell(appears: int, **opts):
    """A can carried to the basket across a slab 30 cm tall that stands in the way from step ``appears`` on. The
    stand-in server's full looks report it as the colour line does: an unexplained region a metre long with the
    slab as its top layer, a thin slice of the arm floating above the hand, and the hand's own top."""
    from types import SimpleNamespace

    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.pro2.probes import kin_same as KS
    from rowant_lab.pro2.robot import episode as EPI
    from rowant_lab.pro2.robot import grasp as GR

    lib = LB.M2Library("libero_object")
    R = SimpleNamespace(grasp=GR, look=LK, agent=AG, episode=EPI)
    Y, X, TOP = (-0.06, -0.04), (-0.10, 0.14), 0.30
    bottom = float(lib.records["alphabet_soup"]["bottom"])

    class Cell(KS.World):
        def step(self, a):
            p_old, held = self.p.copy(), self.held
            t_old = None if held is None else self.true[held].copy()
            super().step(a)
            if self.n >= appears and held is not None and self.held == held:
                c = self.true[held]
                if X[0] - 0.035 < c[0] < X[1] + 0.035 and Y[0] - 0.035 < c[1] < Y[1] + 0.035 and c[2] + bottom < TOP:
                    self.p = p_old + np.array([0.0, 0.0, 2e-5 * (1 if self.n % 2 else -1)])  # (it trembles)
                    self.true[held] = t_old

    class Srv(KS.Server):
        def call(self, msg):
            rep = super().call(msg)
            if msg.get("mode") == "full" and self.w.n >= appears:
                h = self.w.p
                slab = {"xy": [-0.30, -0.05], "half_xy": [0.45, 0.10], "yaw": 0.0, "z_top": TOP, "height": TOP,
                        "n_px": 5000, "h_min": 0.02, "top": {"xy": [0.02, -0.05], "half_xy": [0.12, 0.012], "yaw": 0.0,
                                                              "z": TOP, "n_px": 4000}}
                arm = {"xy": [h[0] - 0.06, h[1] + 0.04], "half_xy": [0.02, 0.02], "yaw": 0.0, "z_top": 0.41,
                       "height": 0.41, "n_px": 600, "h_min": 0.37,
                       "top": {"xy": [h[0] - 0.06, h[1] + 0.04], "half_xy": [0.02, 0.02], "yaw": 0.0, "z": 0.41, "n_px": 600}}
                own = {"xy": [h[0] - 0.1, h[1]], "half_xy": [0.2, 0.05], "yaw": 0.0, "z_top": h[2] + 0.03,
                       "height": h[2] + 0.03, "n_px": 3000, "h_min": max(0.0, h[2] - 0.1),
                       "top": {"xy": [h[0] + 0.03, h[1] + 0.02], "half_xy": [0.04, 0.02], "yaw": 0.0, "z": h[2] + 0.03,
                               "n_px": 800}}
                rep["extra"] = {**rep["extra"], "blobs": [slab, arm, own]}
            return rep

    pc = KS._percepts(lib)
    w = Cell(R, lib, pc)
    tool = LK.PerceptionTool(lib, Srv(w, pc), KS._cam, "libero_object", "ep")
    tool.obs = w.pobs()
    tool.full_look()
    tool.target = "p0"
    spec = {"predicate": "in", "target": "p0", "place": "p1_contain_region", "place_kind": "region",
            "place_instance": "p1", "instruction": "pick the alphabet soup and place it in the basket"}
    ag = AG.OracleAgent(lib, lib.scene_from_percepts(tool.percepts), spec, tool, 400,
                        AG.AgentParams(**{**KS.BASE_AGENT, "transit_cut": True, "detour_from_cross": True, **opts}))
    res = KS._drive(R, ag, w, obs_to=tool)
    in_basket = float(np.linalg.norm((w.true["p0"] - w.true["p1"])[:2])) < 0.05
    return ag, tool, res, in_basket


@pytest.mark.skipif(not SR.LIB_ROOT.is_dir(), reason="the object library is not built on this machine")
def test_the_real_look_tool_gives_the_carry_the_slab_and_not_its_long_region_the_arm_or_the_hand():
    ON = dict(transit_unexplained=True, carry_block_steps=12, carry_relook=True)
    # the slab stands there from the start: it is in the first look, and the carry crosses above it at once
    ag, tool, res, there = _slab_cell(0, **ON)
    ev = [e for e in ag.events if e["ev"] == "transit" and "unexplained" in e]
    assert there and len(ev) == 1, res
    # the slab by its top layer, not by its region; the slice of the arm is left out; the hand's own top at the
    # start pose is one more box, 15 cm from the way and of no consequence
    assert np.allclose(ev[0]["seen"][0], [0.02, -0.05, 0.30, 0.12], atol=1e-3) and ev[0]["unexplained"] <= 2
    assert all(b[3] <= 0.12 + 1e-9 and b[2] < 0.35 for b in ev[0]["seen"])
    assert not any(e["ev"] in ("carry_blocked", "detour") for e in ag.events) and "carry_relook" not in ag.attempts[0]
    # put there after the can was lifted: the carry meets it, the second look shows it, the carry crosses above it
    ag, tool, res, there = _slab_cell(36, **ON)
    rl = ag.attempts[0]["carry_relook"]
    assert there and rl["looked"] and rl["new"] == 1 and rl["plan"] == "over" and rl["status"] == "reached", (res, rl)
    # counted: the slab, and the hand's own top (which is not what is new); the arm's slice is not
    assert rl["counted"][0][:2] == [0.02, -0.05] and len(rl["counted"]) == 2 and len(rl["regions"]) == 3
    assert not any(e["ev"] == "detour" for e in ag.events)
    assert tool.extra["blobs"] == [] and len(tool.extra_last["blobs"]) == 3  # (the first look's regions stay)
    # the options off: met, then the detour (as before)
    ag, tool, res, there = _slab_cell(36)
    assert any(e["ev"] == "detour" for e in ag.events) and "carry_relook" not in ag.attempts[0]
    assert not any(e["ev"] == "carry_blocked" for e in ag.events)


def test_keep_target_does_nothing_in_a_sequence_of_two_objects(monkeypatch):
    """The two-pots scenario of ``test_long2_options`` (the candidate's bundle in the kinematic stand-in) with
    ``keep_target`` on: both pots placed, at the same places, as without it."""
    from rowant_lab.libero_env.tests import test_long2_options as TL
    from rowant_lab.pro2 import long_dev as LD

    lib = TL._kitchen()
    if lib is None:
        pytest.skip("the long suite's library is not on this machine")
    plain, _tool = TL._two_pots(lib, False)
    assert [r[:2] for r in plain] == [(0, "placed"), (1, "placed")]
    bundle = LD.bundle
    monkeypatch.setattr(LD, "bundle", lambda p: {**bundle(p), "keep_target": True})
    kept, tool = TL._two_pots(lib, False)
    assert kept == plain and tool.prior == {}  # (nothing is remembered in a sequence)


@pytest.mark.skipif(not SR.LIB_ROOT.is_dir(), reason="the object library is not built on this machine")
def test_a_stand_in_under_a_bowl_must_be_as_tall_as_the_bowl_stands_high():
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.pro2.robot import episode as EP

    lib = LB.M2Library("libero_spatial")
    ins = "pick the akita black bowl on the ramekin and place it on the plate"
    on_box, on_ramekin = (0.056, 0.043, 0.917), (-0.214, 0.202, 0.933)  # 2 cm up (a cookie box), 3.5 cm up (the ramekin)

    def tool(rejected):
        t = _spatial_tool(lib, [on_box, on_ramekin])
        t.percepts = [p for p in t.percepts if p["type"] != "glazed_rim_porcelain_ramekin"]  # hidden under its bowl
        t.extra = {"rejected": [{"id": f"p10{i}", "type": "glazed_rim_porcelain_ramekin", "gid": "g", "joints": {},
                                 "fit": {"n_agree": n, "inlier": 0.9}, "source": "support", "pos": list(pos),
                                 "why": {"accepted": False, "upright_err_deg": 0.0},
                                 "bodies": t._build("glazed_rim_porcelain_ramekin", np.eye(3), np.asarray(pos, float))}
                                for i, (pos, n) in enumerate(rejected)], "blobs": []}
        return t

    false_first = [((0.055, 0.044, 0.904), 3000), ((-0.207, 0.199, 0.900), 2000)]  # the box read as a ramekin, stronger
    P = {"relation_single": True, "fallback_on_table": True}
    g = EP._plan_one(ins, tool(false_first), lib, 220, P)["grounding"]
    assert g["target"] == "p0" and "landmark_fallback:glazed_rim_porcelain_ramekin" in g["flags"]  # as before: misled
    g = EP._plan_one(ins, tool(false_first), lib, 220, {**P, "relation_standin_strict": True})["grounding"]
    assert g["target"] == "p1" and "landmark_fallback:glazed_rim_porcelain_ramekin" in g["flags"]  # the true ramekin
    # only the false pose: no stand-in, and the choice is as with the landmark missing
    g = EP._plan_one(ins, tool(false_first[:1]), lib, 220, {**P, "relation_standin_strict": True})["grounding"]
    assert "landmark_missing" in g["flags"] and "landmark_fallback:glazed_rim_porcelain_ramekin" not in g["flags"]
    # a stand-in that singles out neither bowl (out on the table) is not believed
    stray = [((0.30, -0.20, 0.900), 3000)]
    g = EP._plan_one(ins, tool(stray), lib, 220, {**P, "relation_standin_strict": True})["grounding"]
    assert "landmark_missing" in g["flags"]
    g = EP._plan_one(ins, tool(stray), lib, 220, P)["grounding"]
    assert "landmark_fallback:glazed_rim_porcelain_ramekin" in g["flags"]  # off: believed, as before

    # a stand-in must be seen in the look: one that lies hidden and shows next to nothing, or whose seen part agrees
    # too little, is not believed (the true ramekin under its bowl reads 0.33 seen and 0.83 of what is not hidden)
    def with_why(why):
        t = tool([((-0.207, 0.199, 0.900), 2000)])
        t.extra["rejected"][0]["why"].update(why)
        return EP._plan_one(ins, t, lib, 220, {**P, "relation_standin_strict": True})["grounding"]

    assert with_why({"vis_share": 0.33, "share": 0.83, "covered_share": 0.0})["target"] == "p1"
    for why in ({"vis_share": 0.09, "share": 0.95, "covered_share": 0.98}, {"vis_share": 0.35, "share": 0.59}):
        g = with_why(why)
        assert "landmark_missing" in g["flags"] and "landmark_fallback:glazed_rim_porcelain_ramekin" not in g["flags"]
        t = tool([((-0.207, 0.199, 0.900), 2000)])
        t.extra["rejected"][0]["why"].update(why)
        assert "landmark_fallback:glazed_rim_porcelain_ramekin" in EP._plan_one(ins, t, lib, 220, P)["grounding"]["flags"]
