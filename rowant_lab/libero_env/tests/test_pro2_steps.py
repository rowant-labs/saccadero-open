"""Steps per attempt (2026-11-16; every option off by default): the cut transit way, the one-move lift, the faster
last move down."""

import numpy as np

from rowant_lab.pro2.robot import agent as AG
from rowant_lab.pro2.robot import grasp as GRS

R = GRS.hand_matrix([0.0, 1.0, 0.0])


def _agent(walls=(), **kw):
    """An agent whose gripper sweeps a 4 x 4 x 4 cm box and whose world is ``walls`` ((centre, half) pairs)."""
    ag = object.__new__(AG.OracleAgent)
    ag.ap = AG.AgentParams(**kw)
    world = GRS.Boxes([w[0] for w in walls], [np.eye(3)] * len(walls), [w[1] for w in walls]) if walls \
        else GRS.Boxes.empty()
    ag._world = lambda poses, exclude=(): world
    ag.supports = GRS.Boxes.empty()
    ag._swept_boxes = lambda pt, R_, half_gap, held: GRS.Boxes([pt], [np.eye(3)], [[0.02, 0.02, 0.02]])
    ag.events = []
    ag.t = 0
    return ag


def _way(ag, p0, goal, z_cross):
    return ag._cut_way({}, np.asarray(p0, float), R, np.asarray(goal, float), R, z_cross, 0.04, None, (), 0.0)


def test_with_nothing_in_the_way_the_hand_climbs_straight_to_the_end_of_the_crossing():
    # (the planner only asks for a crossing when the straight move is blocked; here the slanted climb to the far end
    # of the crossing line is free, so the way is that climb and the come-down)
    way = _way(_agent(), [0.0, 0.0, 0.90], [0.30, 0.0, 0.90], 1.10)
    assert [tuple(np.round(w[0], 3)) for w in way] == [(0.30, 0.0, 1.10), (0.30, 0.0, 0.90)]
    assert way[0][2] == AG.AgentParams().transit_blend_m and way[0][3] == 180.0  # passed without stopping
    assert way[-1][2] is None and way[-1][3] is None  # the goal keeps the caller's tolerances


def test_a_wall_next_to_the_goal_shortens_the_slant_and_one_next_to_the_start_removes_it():
    wall_far = [([0.22, 0.0, 0.53], [0.02, 0.2, 0.53])]  # up to z 1.06, 8 cm before the goal
    way = _way(_agent(wall_far), [0.0, 0.0, 0.90], [0.30, 0.0, 0.90], 1.10)
    pts = [tuple(np.round(w[0], 3)) for w in way]
    assert pts[0][2] == 1.10 and 0.0 < pts[0][0] < 0.30  # joins the crossing line part of the way along
    assert pts[-1] == (0.30, 0.0, 0.90) and pts[-2][2] == 1.10  # and still comes down from the line
    wall_near = [([0.06, 0.0, 0.5], [0.02, 0.2, 0.52])]  # up to z 1.02, right beside the start
    way = _way(_agent(wall_near), [0.0, 0.0, 0.90], [0.30, 0.0, 0.90], 1.10)
    pts = [tuple(np.round(w[0], 3)) for w in way]
    assert pts[0] == (0.0, 0.0, 1.10)  # straight up first, as before


def test_a_blocked_crossing_or_a_short_way_leaves_the_old_way():
    roof = [([0.15, 0.0, 1.10], [0.02, 0.2, 0.02])]  # on the crossing line itself
    assert _way(_agent(roof), [0.0, 0.0, 0.90], [0.30, 0.0, 0.90], 1.10) is None
    assert _way(_agent(), [0.0, 0.0, 0.90], [0.02, 0.0, 0.90], 1.10) is None


def test_a_hand_already_at_the_crossing_height_comes_down_on_a_slant_where_that_is_clear():
    box = [([0.15, 0.0, 0.5], [0.03, 0.2, 0.5])]  # up to z 1.0, in the middle of the way
    way = _way(_agent(box), [0.0, 0.0, 1.10], [0.30, 0.0, 0.80], 1.10)
    pts = [tuple(np.round(w[0], 3)) for w in way]
    assert pts[-1] == (0.30, 0.0, 0.80) and len(pts) == 2
    assert pts[0][2] == 1.10 and 0.15 < pts[0][0] < 0.30  # leaves the line past the box, before the goal's column


def test_a_corner_is_passed_without_stopping_only_when_the_line_from_there_on_is_clear():
    way = _way(_agent(), [0.0, 0.0, 0.90], [0.30, 0.0, 0.90], 1.10)
    assert way[0][2] == 0.03
    # a small thing inside the corner at the far end of the crossing line: the slanted climb and the column are clear
    # of it, the line from 3 cm before the corner to the goal is not
    knob = [([0.283, 0.0, 1.072], [0.003, 0.2, 0.003])]
    ag = _agent(knob)
    ag._swept_boxes = lambda pt, R_, half_gap, held: GRS.Boxes([pt], [np.eye(3)], [[0.005, 0.005, 0.005]])
    way = _way(ag, [0.0, 0.0, 0.90], [0.30, 0.0, 0.90], 1.10)
    last_corner = way[-2]
    assert tuple(np.round(last_corner[0], 3)) == (0.30, 0.0, 1.10) and last_corner[2] == AG.AgentParams().transit_tol_m


def _drive(ag, p0, goal, **kw):
    """Run ``_move_via`` with a ``_goto`` that jumps to its target; returns (legs asked, status)."""
    state = {"p": np.asarray(p0, float)}
    asked = []

    def goto(p_tgt, R_tgt, grip, cap, tol_m, tol_deg, max_steps=None, stall_down=False, cap_rot=None, straight=False):
        asked.append((tuple(np.round(np.asarray(p_tgt, float), 3)), tol_m, tol_deg, cap, straight))
        state["p"] = np.asarray(p_tgt, float)
        yield [0.0] * 6 + [grip]
        return "reached"

    ag._goto = goto
    ag._pose = lambda: (state["p"], R)
    ag._ev = lambda name, **k: ag.events.append({"ev": name, **k})
    gen = ag._move_via({}, np.asarray(goal, float), R, 1.0, (), 0.04, None, 0.0, 0.01, 6.0, **kw)
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return asked, e.value


WALL = [([0.15, 0.0, 0.5], [0.03, 0.2, 0.5])]  # between start and goal, up to z 1.0


def test_off_the_transit_is_rise_cross_descend_and_on_it_is_cut():
    old, st = _drive(_agent(WALL), [0.0, 0.0, 0.90], [0.30, 0.0, 0.90])
    assert st == "reached" and len(old) == 3 and old[0][0][:2] == (0.0, 0.0) and old[1][0][:2] == (0.30, 0.0)
    z = old[0][0][2]
    assert old[0][0] == (0.0, 0.0, z) and old[1][0] == (0.30, 0.0, z) and old[2][0] == (0.30, 0.0, 0.90)
    new, st = _drive(_agent(WALL, transit_cut=True), [0.0, 0.0, 0.90], [0.30, 0.0, 0.90])
    assert st == "reached" and new[-1][0] == (0.30, 0.0, 0.90) and new[-1][1:3] == (0.01, 6.0)  # the caller's tolerances
    assert new[0][0][0] > 0.0 and new[0][0][2] == z  # the first leg already goes across
    assert all(leg[2] == 180.0 for leg in new[:-1])  # no waiting for the turn at a point on the way
    assert all(leg[4] for leg in new) and not any(leg[4] for leg in old)  # every cut leg is held to its tested line

    def length(legs):
        pts = [np.array([0.0, 0.0, 0.90])] + [np.array(leg[0]) for leg in legs]
        return sum(float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:]))

    assert length(new) < length(old) - 0.02


def test_through_a_delay_the_old_way_stands_and_the_last_legs_cap_is_kept():
    ag = _agent(WALL, transit_cut=True)
    ag.lag_steps = 7
    legs, _st = _drive(ag, [0.0, 0.0, 0.90], [0.30, 0.0, 0.90])
    assert len(legs) == 3 and legs[0][0][:2] == (0.0, 0.0)
    legs, _st = _drive(_agent(WALL, transit_cut=True), [0.0, 0.0, 0.90], [0.30, 0.0, 0.90], cap_last=0.4)
    assert legs[-1][3] == 0.4 and all(leg[3] == AG.AgentParams().transit_cap for leg in legs[:-1])


# ------------------------------------------------------------------ the hand lets go: the look tool is told
def _held_tool(gap_each=0.02):
    """A look tool that took the target at (0.1, 0.2, 0.9) into the hand, with the hand now 30 cm up and aside and
    the fingers ``gap_each`` open each (0.04 is fully open; under 0.0375 this tool reads the hand as closed)."""
    from rowant_lab.libero_env.tests.test_redetect import Client, _tool

    cl = Client(status="predicted", fresh=(0.12, 0.2, 0.9))  # a full look sees it on the table, near where it stood
    t = _tool(cl, redetect=False, gripper=(0.004, -0.004))
    t.obs["robot0_eef_pos"] = np.array([0.1, 0.2, 0.95])
    t()  # the look after closing: the target is taken as held
    assert t._held is not None
    t.obs["robot0_eef_pos"] = np.array([0.3, 0.4, 1.2])
    t.obs["robot0_gripper_qpos"] = np.array([gap_each, -gap_each])
    cl.calls.clear()
    return t, cl


def _target_pos(t):
    return np.asarray(t.percepts[0]["bodies"]["main"]["pos"], float)


def test_a_half_open_hand_still_carries_the_target_unless_the_tool_is_told():
    t, cl = _held_tool()
    t()
    assert [m["mode"] for m in cl.calls] == ["track"]
    assert np.allclose(_target_pos(t), [0.3, 0.4, 1.15])  # riding in the empty hand, 25 cm above the table
    t, cl = _held_tool()
    t.hand_opened()
    t()
    assert cl.calls[0]["mode"] == "full" and t._held is None and t.looks[-1].get("told_open") or t.looks[-2].get("told_open")
    assert np.allclose(_target_pos(t), [0.12, 0.2, 0.9])  # where the full look sees it
    assert t._opening is False and t.n_told_open == 1


def test_told_once_the_tool_does_not_take_the_target_back_into_a_hand_that_is_still_opening():
    from rowant_lab.libero_env.tests.test_redetect import Client

    t, cl = _held_tool()
    cl.fresh = (0.3, 0.4, 1.19)  # the full look reports something of its type right at the hand (within 12 cm)
    t.hand_opened(lost=True)
    assert np.allclose(t._lost_xy, [0.3, 0.4])
    t()
    assert t._held is None and t._lost_xy is None  # still half open, and not counted as holding it
    # the next grasp: the hand closes on it again and the tool takes it as held, as before
    t.obs["robot0_gripper_qpos"] = np.array([0.004, -0.004])
    t()
    assert t._held is not None


def test_a_target_the_full_look_does_not_find_is_not_taken_back_into_the_opening_hand():
    t, cl = _held_tool()
    full = cl.call

    def call(msg):  # the full look reports nothing of the target's type
        return {"percepts": [], "times": {}} if msg.get("mode") == "full" else full(msg)

    cl.call = call
    t.hand_opened()
    t()
    assert [L["mode"] for L in t.looks[-2:]] == ["full", "track"]  # the merged look, then one track from where it was
    assert t._held is None and t._opening is False
    # after a grasp or a lift that failed it is where it stood when the hand closed on it, not riding in the hand
    assert np.allclose(_target_pos(t), [0.1, 0.2, 0.9]) and t.percepts[0]["source"] == "predicted"
    assert t.looks[-2].get("unheld") is True
    # lost in a carry it is not where it stood: the pose stays at the hand and the agent is told it is missing
    t, cl = _held_tool()
    cl.call = call
    t.hand_opened(lost=True)
    t()
    assert np.allclose(_target_pos(t), [0.3, 0.4, 1.15]) and t.target_missing


def test_a_target_lost_in_a_carry_is_looked_for_again_until_a_look_finds_it():
    t, cl = _held_tool()
    full = cl.call
    seen = {"n": 0}

    def call(msg):  # the first full look after the loss shows nothing of its type, the second shows it on the table
        if msg.get("mode") != "full":
            return full(msg)
        seen["n"] += 1
        return {"percepts": [], "times": {}} if seen["n"] == 1 else full(msg)

    cl.call = call
    t.hand_opened(lost=True)
    t()
    assert t.target_missing and t.looks[-1].get("lost") == "missing" and t._held is None
    t()  # not asked again: an ordinary track look, and the flag is down
    assert not t.target_missing and t.looks[-1]["mode"] == "track" and seen["n"] == 1
    t.hand_opened(lost=True)
    t()  # asked again: a full look though nothing is held
    assert seen["n"] == 2 and not t.target_missing and np.allclose(_target_pos(t), [0.12, 0.2, 0.9])
    assert any(L.get("lost") == "found" for L in t.looks)


def test_after_a_lost_carry_the_hand_rises_opens_and_looks_until_the_object_is_found():
    def run(missing, **kw):
        ag = object.__new__(AG.OracleAgent)
        ag.ap = AG.AgentParams(**kw)
        ag.events, ag.t = [], 0
        ag._ev = lambda name, **k: None
        ag._pose = lambda: (np.array([0.1, 0.2, 1.0]), R)
        log, it = [], iter(missing)

        class Tool:
            target_missing = False

            def hand_opened(self, lost=False):
                log.append("told" if lost else "told-open")

            def __call__(self):
                self.target_missing = next(it)
                log.append("look")
                return {"poses": len(log)}

        ag.oracle = Tool()

        def goto(p_tgt, R_tgt, grip, cap, tol_m, tol_deg, max_steps=None, **k):
            log.append(("rise", round(float(p_tgt[2]) - 1.0, 3), grip))
            yield [0.0] * 6 + [grip]
            return "reached"

        def hold(p, R_, grip, n):
            log.append(("hold", n, grip))
            for _ in range(n):
                yield [0.0] * 6 + [grip]

        ag._goto, ag._hold = goto, hold
        gen = ag._after_loss(R)
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return log, e.value

    log, rec = run([], carry_watch=True)  # without ``tell_open``: open where the hand is, no look
    assert log == [("hold", AG.AgentParams().open_steps, -1.0)] and rec is None
    log, rec = run([False], tell_open=True)
    assert log == ["told", ("rise", 0.10, -1.0), "look"] and rec == {"looks": 1, "found": True}
    log, rec = run([True, True, False], tell_open=True)
    assert log == ["told", ("rise", 0.10, -1.0), "look", ("hold", 6, -1.0), "told", "look", ("hold", 6, -1.0), "told",
                   "look"] and rec == {"looks": 3, "found": True}
    log, rec = run([True, True, True], tell_open=True)
    assert rec == {"looks": 3, "found": False} and log.count("look") == 3


def test_both_notices_together_release_here_sets_the_place_and_hand_opened_brings_the_look():
    """``release_here`` (the other stream's notice) and ``hand_opened`` given one after the other, as the agent gives
    them with both options on: the target's pose is the one ``release_here`` took from the hand as it let go, the
    merged look comes at the very next look though the fingers still read closed, and neither flag outlasts it."""
    t, cl = _held_tool()
    t.obs["robot0_eef_pos"] = np.array([0.1, 0.2, 0.96])  # the hand lets go 1 cm above where it closed
    assert t.release_here() is True and t._held is None and t._release_pending
    assert np.allclose(_target_pos(t), [0.1, 0.2, 0.91])  # taken from the hand now, not after it has risen
    t.hand_opened()
    t.obs["robot0_eef_pos"] = np.array([0.1, 0.2, 1.05])  # risen to the point above the grasp, fingers half open
    full = cl.call
    cl.call = lambda msg: {"percepts": [], "times": {}} if msg.get("mode") == "full" else full(msg)  # hidden by the hand
    t()
    assert [L["mode"] for L in t.looks[-2:]] == ["full", "track"] and t.looks[-2].get("told_open")
    assert not t.looks[-2].get("unheld")  # ``release_here``'s place stands; the other last resort is not used
    assert np.allclose(_target_pos(t), [0.1, 0.2, 0.91]) and t._held is None
    assert t._release_pending is False and t._opening is False
    # ``release_here`` alone, the fingers still reading closed: no merged look yet, and its flag stays up
    t, cl = _held_tool()
    t.release_here()
    t()
    assert t.looks[-1]["mode"] == "track" and t._release_pending is True


def test_the_agent_tells_only_with_the_option_on():
    told = []
    for on in (False, True):
        ag = object.__new__(AG.OracleAgent)
        ag.ap = AG.AgentParams(tell_open=on)
        ag.oracle = type("T", (), {"hand_opened": lambda self, lost=False: told.append(lost)})()
        ag._tell_open()
        ag._tell_open(lost=True)
    assert told == [False, True]


# ------------------------------------------------------------------ closing early, the pull's early end
def _down(ag, zs, gaps, c, width):
    """Run ``_closing_early`` over a move that takes one step per height in ``zs``; returns the gripper commands."""
    state = {"k": 0}
    ag._pose = lambda: (np.array([c[0], c[1], zs[min(state["k"], len(zs) - 1)]]), R)
    ag._gap = lambda: gaps[min(state["k"], len(gaps) - 1)]
    ag._ev = lambda name, **k: None

    def move():
        for k in range(len(zs)):
            ag._v_step = np.array([0.0, 0.0, zs[k] - zs[k - 1]]) if k else np.zeros(3)  # as ``act`` keeps it
            yield [0.0] * 6 + [-1.0]
        return "reached"

    gen, sent = ag._closing_early(move(), np.asarray(c, float), width), []
    try:
        while True:
            sent.append(next(gen)[6])
            state["k"] += 1
    except StopIteration as e:
        return sent, e.value


def test_the_fingers_start_closing_three_steps_before_the_hand_is_down_on_a_thin_grasp_only():
    c = [0.0, 0.0, 0.90]
    zs = [0.95, 0.942, 0.934, 0.926, 0.918, 0.910, 0.904, 0.901]  # 8 mm a step, then slowing
    open_gap = [0.078] * 8
    assert _down(_agent(), zs, open_gap, c, 0.009) == ([-1.0] * 8, "reached")  # off: the move as it is
    sent, st = _down(_agent(close_early_m=0.025), zs, open_gap, c, 0.009)
    assert st == "reached" and sent == [-1.0] * 4 + [1.0] * 4  # from 1.8 cm above: three steps of 8 mm away
    assert _down(_agent(close_early_m=0.012), zs, open_gap, c, 0.009)[0] == [-1.0] * 5 + [1.0] * 3  # capped at 1.2 cm
    slow = [0.95, 0.945, 0.94, 0.935, 0.93, 0.925, 0.92, 0.915, 0.91, 0.905, 0.9015]  # a hand coming down 5 mm a step
    assert _down(_agent(close_early_m=0.025), slow, [0.078] * 11, c, 0.009)[0] == [-1.0] * 7 + [1.0] * 4  # from 1.5 cm
    assert _down(_agent(close_early_m=0.025), zs, open_gap, c, 0.05)[0] == [-1.0] * 8  # a wide grasp: never
    ag = _agent(close_early_m=0.025)
    ag.lag_steps = 7
    assert _down(ag, zs, open_gap, c, 0.009)[0] == [-1.0] * 8  # through a delay: never


def test_a_pull_can_end_as_soon_as_the_part_is_nearly_out():
    from rowant_lab.pro2.robot import drawer as DR

    def run(done_at):
        ag = object.__new__(DR.OpenAgent)
        ag.ap = AG.AgentParams()
        ag.fp = AG.FollowParams()
        ys = [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]  # the drawer ends at 10 cm
        state = {"k": 0}
        ag._pose = lambda: (np.array([0.0, ys[min(state["k"], len(ys) - 1)], 1.0]), R)
        gen = ag._pull(np.array([0.0, 0.115, 1.0]), R, np.array([0.0, 1.0, 0.0]), 0.115, done_at=done_at)
        n = 0
        try:
            while True:
                next(gen)
                n += 1
                state["k"] += 1
        except StopIteration as e:
            return n, e.value

    assert run(None) == (10, "stalled")  # as before: four steps after it stopped coming
    assert run(0.095) == (6, "reached")  # with the early end: when it is 5 mm from its end


# ------------------------------------------------------------------ braking before a precise stop
def _goto_actions(ag, p_tgt, tol_m, v_step, n=1, **kw):
    p0 = np.array([0.0, 0.0, 1.0])
    ag._pose = lambda: (p0, R)
    ag._ev = lambda name, **k: None
    ag.fp = AG.FollowParams()
    ag._v_step = np.asarray(v_step, float)
    gen = ag._goto(np.asarray(p_tgt, float), R, -1.0, 1.0, tol_m, 180.0, max_steps=n, **kw)
    return [next(gen) for _ in range(n)]


def test_a_precise_move_brakes_the_hands_sideways_movement_and_a_passing_one_does_not():
    tgt = [0.0, 0.01, 1.0]  # 1 cm to go, along y
    v = [0.004, 0.002, -0.006]  # the hand's movement in the last step
    plain = _goto_actions(_agent(), tgt, 0.006, v)[0]
    assert plain[0] == 0.0 and plain[2] == 0.0 and abs(plain[1] - 0.4) < 1e-9  # as before: straight at the target
    braked = _goto_actions(_agent(brake_k=3.0), tgt, 0.006, v)[0]
    assert abs(braked[0] - 2.0 * (-3.0 * 0.004) / 0.05) < 1e-9  # against the sideways movement, both ways
    assert abs(braked[1] - 2.0 * (0.01 - 3.0 * 0.002) / 0.05) < 1e-9
    assert braked[2] == 0.0  # its movement up or down is left alone
    both = _goto_actions(_agent(brake_k=3.0, brake_kz=1.0), tgt, 0.006, v)[0]
    assert abs(both[2] - 2.0 * (1.0 * 0.006) / 0.05) < 1e-9
    assert _goto_actions(_agent(brake_k=3.0), [0.0, 0.05, 1.0], 0.03, v)[0][0] == 0.0  # a point passed on the way
    ag = _agent(brake_k=3.0)
    ag.lag_steps = 7
    assert _goto_actions(ag, tgt, 0.006, v)[0][0] == 0.0  # through a delay: as before


def test_a_slanted_move_may_ask_each_axis_for_its_full_command():
    from rowant_lab.pro2.robot.follow import follow

    p, tgt = np.zeros(3), np.array([0.2, 0.2, 0.1])
    plain = np.array(follow(p, R, tgt, R, -1.0, 1.0)[:3])
    boxed = np.array(follow(p, R, tgt, R, -1.0, 1.0, box=True)[:3])
    # as before: every axis far from its goal is clipped to 1 first, so the command points along the diagonal of
    # the signs, not at the target (the hand's way bends where an axis arrives), and its length is 1
    assert np.allclose(plain, np.ones(3) / np.sqrt(3.0))
    # boxed: at the target, with the axis that has farthest to go at its full command
    assert np.allclose(boxed, [1.0, 1.0, 0.5]) and np.allclose(boxed / np.linalg.norm(boxed), tgt / np.linalg.norm(tgt))
    axis = np.array([0.0, 0.0, 0.2])
    assert follow(p, R, axis, R, -1.0, 0.6)[:3] == follow(p, R, axis, R, -1.0, 0.6, box=True)[:3]  # along one axis: the same
    near = np.array([0.004, 0.003, 0.0])
    assert follow(p, R, near, R, -1.0, 1.0)[:3] == follow(p, R, near, R, -1.0, 1.0, box=True)[:3]  # near the goal: the same
    slant = _goto_actions(_agent(), [0.2, 0.2, 1.1], 0.006, [0.0, 0.0, 0.0], straight=True)[0]
    assert abs(max(abs(x) for x in slant[:3]) - 1.0) < 1e-9 and np.linalg.norm(slant[:3]) > 1.4
    ag = _agent()
    ag.lag_steps = 5  # through a delay: the old cap
    assert abs(np.linalg.norm(_goto_actions(ag, [0.2, 0.2, 1.1], 0.006, [0.0, 0.0, 0.0], straight=True)[0][:3]) - 1.0) < 1e-9
    # a move that is not a transit leg keeps the old cap, with the option on too
    assert abs(np.linalg.norm(_goto_actions(_agent(cap_box=True), [0.2, 0.2, 1.1], 0.006, [0.0, 0.0, 0.0])[0][:3]) - 1.0) < 1e-9
    # the transit's own legs take the option
    legs, _st = _drive(_agent(cap_box=True), [0.0, 0.0, 0.90], [0.30, 0.0, 0.90])
    assert legs and all(leg[4] for leg in legs)
    legs, _st = _drive(_agent(), [0.0, 0.0, 0.90], [0.30, 0.0, 0.90])
    assert legs and not any(leg[4] for leg in legs)


def test_the_cut_way_always_ends_at_the_goal_itself():
    # the crossing height within the transit tolerance of the goal's: the last point is the goal, not the line's end
    way = _way(_agent(), [0.0, 0.0, 0.90], [0.30, 0.0, 1.095], 1.10)
    assert tuple(np.round(way[-1][0], 3)) == (0.30, 0.0, 1.095) and len(way) == 1
    # a hand at the crossing height and a goal at it too: one leg, the goal
    way = _way(_agent(), [0.0, 0.0, 1.10], [0.30, 0.0, 1.10], 1.10)
    assert [tuple(np.round(w[0], 3)) for w in way] == [(0.30, 0.0, 1.10)]
    for p0, goal in (([0.0, 0.0, 0.9], [0.3, 0.1, 0.8]), ([0.0, 0.0, 1.1], [0.3, 0.1, 0.8]), ([0.0, 0.0, 0.8], [0.3, 0.1, 1.1])):
        way = _way(_agent(WALL), p0, goal, 1.10)
        assert np.allclose(way[-1][0], goal) and way[-1][2] is None


def test_a_way_straight_down_that_is_clear_only_at_the_goals_yaw_waits_for_the_turn():
    """A hand that turns a quarter on the way and comes straight down next to a wall: the wall is clear of the
    gripper at the goal's yaw and not across it, so the hand may not start down before its turn is done."""
    R0, R1 = GRS.hand_matrix([1.0, 0.0, 0.0]), GRS.hand_matrix([0.0, 1.0, 0.0])
    ag = _agent([([0.345, 0.0, 0.5], [0.01, 0.2, 0.5])])  # a wall just past the goal, up to z 1.0

    def swept(pt, R_, half_gap, held):  # a gripper 8 cm long across its closing axis... along it: 2 cm
        long_x = abs(float(R_[0, 1])) > 0.5  # the closing axis along x
        return GRS.Boxes([pt], [np.eye(3)], [[0.04, 0.01, 0.02] if long_x else [0.01, 0.04, 0.02]])

    ag._swept_boxes = swept
    way = ag._cut_way({}, np.array([0.0, 0.0, 1.10]), R0, np.array([0.30, 0.0, 0.80]), R1, 1.10, 0.04, None, (), 0.0)
    assert tuple(np.round(way[-1][0], 3)) == (0.30, 0.0, 0.80)
    before_down = way[-2]
    assert tuple(np.round(before_down[0], 3)) == (0.30, 0.0, 1.10)  # no slant is clear at every yaw: straight down
    assert before_down[3] == AG.AgentParams().transit_tol_deg and before_down[2] == AG.AgentParams().transit_tol_m
    # with nothing near, the same move does not wait
    way = _agent()._cut_way({}, np.array([0.0, 0.0, 1.10]), R0, np.array([0.30, 0.0, 0.80]), R1, 1.10, 0.04, None, (), 0.0)
    assert all(leg[3] in (180.0, None) for leg in way)


# ------------------------------------------------------------------ taking the staged object again
def _staged(memo_open=True):
    """The drawer task's planned steps up to the take-again: (lib, percept, cabinet, steps, memo, stage xy)."""
    import copy

    import pytest

    from rowant_lab.libero_env.tests.test_sequence import _goal_scene
    from rowant_lab.pro2.robot import episode as EPI
    from rowant_lab.pro2.robot import sequence as SQ

    scene = _goal_scene()
    if scene is None:
        pytest.skip("the long-lib tree is not on this machine")
    lib, percept, cab = scene
    steps = SQ.split("open the top layer of the drawer and put the bowl inside")
    bowl = percept("p0", "akita_black_bowl", np.eye(3), [-0.085, 0.004, 0.898])
    tool = type("T", (), {})()
    tool.percepts, tool.target = [bowl, cab], None
    out = EPI._overhang_rewrite(steps, tool, lib, clear=False)
    memo = {}
    st = EPI._plan_special(out[0], tool, lib, 150, {}, memo)
    sx, sy = st["grounding"]["stage"]
    c_top, _e = EPI._site_world(lib, cab, "top_side")
    memo["stage"]["agent"].released = {"p": [sx, sy + 0.045, float(c_top[2]) + 0.045], "R": np.eye(3).tolist(),
                                       "frame_z": memo["stage"]["lo_z"] + 0.025, "t": 90}
    memo["stage"]["agent"].attempts.append({"place": {"surface": float(c_top[2])}})
    if memo_open:  # what the planner keeps when the opening step is planned
        memo["open_fx"] = {(cab["id"], "cabinet_top"): copy.deepcopy(cab)}
    return lib, percept, cab, out, memo, (sx, sy)


def _turned(lib, percept, cab, deg, shift):
    """The cabinet as a later look fitted it: turned ``deg`` about the vertical where it stands, moved by ``shift``."""
    root = lib.export_types[cab["type"]]["root_body"]
    R0 = np.asarray(lib.quat_mat(cab["bodies"][root]["quat_wxyz"]), float)
    t0 = np.asarray(cab["bodies"][root]["pos"], float)
    a = np.radians(deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return percept(cab["id"], cab["type"], Rz @ R0, t0 + np.array([shift[0], shift[1], 0.0]))


def test_the_take_again_goes_by_the_fixture_as_it_stood_when_its_drawer_was_pulled_open():
    from types import SimpleNamespace

    from rowant_lab.pro2.robot import episode as EPI

    lib, percept, cab, out, memo, (sx, sy) = _staged()
    seen = lambda fx: SimpleNamespace(percepts=[fx], target="p0")  # noqa: E731
    good = EPI._plan_special(out[2], seen(cab), lib, 80, {}, memo)["grounding"]
    assert abs(good["move"][0]) < 0.01 and 0.10 < good["move"][1] < 0.18  # into the drawer
    assert good["fx_moved"] == {"shift_m": 0.0, "turn_deg": 0.0} and "by_open" not in good
    # the look since fitted the cabinet a quarter turn off and 4 cm aside (what the rows show: carries of about
    # (-0.16, -0.08) where (-0.01, 0.13) goes into the drawer)
    off = _turned(lib, percept, cab, 90.0, (0.0, -0.04))
    bad = EPI._plan_special(out[2], seen(off), lib, 80, {}, memo)["grounding"]
    assert bad["move"][0] < -0.12 and bad["move"][1] < 0.0  # as before: sideways, out of the drawer
    assert abs(abs(bad["fx_moved"]["turn_deg"]) - 90.0) < 0.5 and abs(bad["fx_moved"]["shift_m"] - 0.04) < 1e-3
    fixed = EPI._plan_special(out[2], seen(off), lib, 80, {"take_by_open": True}, memo)["grounding"]
    assert fixed["move"] == good["move"] and fixed["by_open"] is True
    # moved 4 cm but not turned: the look since is kept (in the rows it was the first look that had the cabinet too
    # far back, and the look since that was right), unless the option is asked to apply always
    near = _turned(lib, percept, cab, 0.0, (0.0, -0.04))
    short = EPI._plan_special(out[2], seen(near), lib, 80, {}, memo)["grounding"]
    assert abs(short["move"][1] - (good["move"][1] - 0.04)) < 1e-3
    kept = EPI._plan_special(out[2], seen(near), lib, 80, {"take_by_open": True}, memo)["grounding"]
    assert kept == short and "by_open" not in kept
    always = EPI._plan_special(out[2], seen(near), lib, 80, {"take_by_open": True, "take_open_turn_deg": 0}, memo)["grounding"]
    assert always["move"] == good["move"] and always["by_open"] is True
    # turned a little (a look's own scatter), then past the threshold
    for deg, by in ((8.0, False), (-8.0, False), (25.0, True), (-25.0, True), (-90.0, True)):
        g = EPI._plan_special(out[2], seen(_turned(lib, percept, cab, deg, (0.0, 0.0))), lib, 80, {"take_by_open": True}, memo)["grounding"]
        assert ("by_open" in g) is by and (g["move"] == good["move"]) is by
    # nothing kept from an opening step (the take-again planned without one): the look's fixture, option or not
    lib, percept, cab, out, memo2, _xy = _staged(memo_open=False)
    g = EPI._plan_special(out[2], seen(_turned(lib, percept, cab, 90.0, (0.0, -0.04))), lib, 80, {"take_by_open": True}, memo2)["grounding"]
    assert g["move"][0] < -0.12 and "by_open" not in g and "fx_moved" not in g


def test_a_drawer_left_short_of_its_end_moves_the_take_agains_goal_back_by_as_much():
    from types import SimpleNamespace

    from rowant_lab.pro2.robot import episode as EPI

    ON = {"take_by_open": True, "take_open_turn_deg": 0}  # applied whatever the look since made of the fixture
    lib, percept, cab, out, memo, _xy = _staged()
    full = EPI._plan_special(out[2], SimpleNamespace(percepts=[cab], target="p0"), lib, 80, ON, memo)
    opened = SimpleNamespace(finish_reason="opened", attempts=[{"skill": "open", "outcome": "opened", "pull": 0.175, "moved": 0.155}])
    memo["opens"] = [(cab["id"], "cabinet_top", opened)]
    assert abs(EPI._pull_short(memo, cab["id"], "cabinet_top") - 0.005) < 1e-9
    short = EPI._plan_special(out[2], SimpleNamespace(percepts=[cab], target="p0"), lib, 80, ON, memo)
    assert abs(short["grounding"]["move"][1] - (full["grounding"]["move"][1] - 0.005)) < 1e-4
    assert short["grounding"]["short_m"] == 0.005
    off = EPI._plan_special(out[2], SimpleNamespace(percepts=[cab], target="p0"), lib, 80, {}, memo)["grounding"]
    assert off["move"] == full["grounding"]["move"] and "short_m" not in off  # without the option: as before
    # the staging let go 8 mm off the point it aimed the hand at: the object stands that far off, and the carry is
    # counted from there
    lib2, percept2, cab2, out2, memo2, (sx, sy) = _staged()
    memo2["stage"]["agent"].attempts[-1]["place"]["xy"] = [sx + 0.006, sy + 0.045 + 0.005]
    g_on = EPI._plan_special(out2[2], SimpleNamespace(percepts=[cab2], target="p0"), lib2, 80, ON, memo2)["grounding"]
    g_off = EPI._plan_special(out2[2], SimpleNamespace(percepts=[cab2], target="p0"), lib2, 80, {}, memo2)["grounding"]
    assert g_on["staged_off"] == [-0.006, -0.005] and "staged_off" not in g_off
    assert abs(g_on["move"][0] - (g_off["move"][0] + 0.006)) < 1e-4 and abs(g_on["move"][1] - (g_off["move"][1] + 0.005)) < 1e-4
    opened.attempts[0]["moved"] = 0.171  # pulled past its end (the handle slid in the fingers): no correction
    assert EPI._pull_short(memo, cab["id"], "cabinet_top") == 0.0
    opened.finish_reason = "slipped"
    assert EPI._pull_short(memo, cab["id"], "cabinet_top") == 0.0


def test_the_take_agains_carry_ends_as_near_its_goal_as_asked():
    from rowant_lab.pro2.robot import drawer as DR

    def tol_asked(**kw):
        ag = object.__new__(DR.RetakeAgent)
        ag.ap = AG.AgentParams(**kw)
        ag.hp = {"p": [0.0, 0.0, 1.2], "R": np.eye(3).tolist(), "drop": 0.02, "move": [0.0, 0.13]}
        ag.attempts, ag.events, ag.t, ag.finished, ag.finish_reason = [], [], 0, False, None
        ag.jaw = type("J", (), {"L": 0.04})()
        asked = []

        def goto(p, R_, grip, cap, tol_m, tol_deg, max_steps=None, **k):
            asked.append(tol_m)
            yield [0.0] * 6 + [grip]
            return "reached"

        ag._goto = goto
        ag._pose = lambda: (np.array([0.0, 0.0, 1.2]), np.eye(3))
        ag._gap = lambda: 0.008
        ag._v_step = np.zeros(3)
        ag._ev = lambda name, **k: None
        ag._act_to = lambda p, R_, g, cap=None: [0.0] * 6 + [float(g)]
        ag._finish = lambda reason: setattr(ag, "finish_reason", reason)
        gen = ag._episode()
        next(gen)
        try:
            while True:
                next(gen)
        except StopIteration:
            pass
        return asked, ag.finish_reason

    asked, reason = tol_asked()
    assert reason == "placed" and asked[3] == 0.02  # over, down, lift, carry: as built
    asked, reason = tol_asked(take_carry_tol_m=0.008)
    assert reason == "placed" and asked[3] == 0.008 and asked[:3] == tol_asked()[0][:3]


def test_a_bowl_staged_by_its_far_rim_is_taken_again_by_the_near_one():
    from types import SimpleNamespace

    from rowant_lab.pro2.robot import episode as EPI

    lib, percept, cab, out, memo, (sx, sy) = _staged()
    rel = memo["stage"]["agent"].released
    seen = SimpleNamespace(percepts=[cab], target="p0")
    # as the staging left it: the hand at the bowl's FAR rim (4.6 cm behind its middle, seen along the way the
    # drawer opens), 3 mm short of where the plan aimed it
    rel["p"] = [sx + 0.002, sy - 0.046 - 0.003, rel["p"][2]]
    rel["hold_xy"], rel["half_turn"] = [0.002, -0.046], True
    plain = EPI._plan_special(out[2], seen, lib, 80, {}, memo)
    assert plain["agent"].hp["p"][:2] == rel["p"][:2] and "near_rim" not in plain["grounding"]  # back where it let go
    near = EPI._plan_special(out[2], seen, lib, 80, {"take_near_rim": True}, memo)
    px, py = near["agent"].hp["p"][:2]
    assert abs(px - (sx - 0.002)) < 1e-9 and abs(py - (sy + 0.046 - 0.003)) < 1e-9  # the point opposite, same height
    assert near["agent"].hp["p"][2] == rel["p"][2] and near["agent"].hp["R"] == rel["R"]
    assert near["grounding"]["near_rim"] == [round(px, 4), round(py, 4)]
    assert near["agent"].hp["move"] == plain["agent"].hp["move"]  # the bowl goes the same way, whichever rim holds it
    # with ``take_by_open`` and a fixture the look since has not turned: the same plan
    both = EPI._plan_special(out[2], seen, lib, 80, {"take_near_rim": True, "take_by_open": True}, memo)
    assert both["grounding"] == near["grounding"] and both["agent"].hp == near["agent"].hp
    # where ``take_by_open`` applies, the carry is counted from where the bowl stands: the hand's point less the
    # grasp's reach
    both = EPI._plan_special(out[2], seen, lib, 80, {"take_near_rim": True, "take_by_open": True, "take_open_turn_deg": 0}, memo)["grounding"]
    assert both["staged_off"] == [0.0, -0.003] and abs(both["move"][1] - (plain["grounding"]["move"][1] + 0.003)) < 1e-4
    # held by the near rim already, or an object that is not the same turned half a turn: as it was
    rel["p"], rel["hold_xy"] = [sx, sy + 0.046, rel["p"][2]], [0.0, 0.046]
    assert "near_rim" not in EPI._plan_special(out[2], seen, lib, 80, {"take_near_rim": True}, memo)["grounding"]
    rel["p"], rel["hold_xy"], rel["half_turn"] = [sx, sy - 0.046, rel["p"][2]], [0.0, -0.046], False
    assert "near_rim" not in EPI._plan_special(out[2], seen, lib, 80, {"take_near_rim": True}, memo)["grounding"]


def test_only_an_object_that_is_the_same_turned_half_a_turn_has_its_grasp_on_the_other_side():
    import pytest

    from rowant_lab.libero_env.tests.test_sequence import _goal_scene

    scene = _goal_scene()
    if scene is None:
        pytest.skip("the long-lib tree is not on this machine")
    lib = scene[0]

    def same(typ):
        ag = object.__new__(AG.OracleAgent)
        ag.lib, ag.ttype = lib, typ
        return ag._half_turn_same()

    assert same("akita_black_bowl") and same("cream_cheese") and same("wine_bottle")
    assert not same("wooden_cabinet")  # (its root body has no primitive of its own: never taken that way)
    # a bowl with a handle on one side is not
    ag = object.__new__(AG.OracleAgent)
    b = lib.body_boxes("akita_black_bowl", lib.types["akita_black_bowl"]["root_body"])
    handle = GRS.Boxes([[0.08, 0.0, 0.03]], [np.eye(3)], [[0.02, 0.006, 0.01]])
    ag.lib = type("L", (), {"types": lib.types, "body_boxes": staticmethod(lambda typ, body: b + handle)})()
    ag.ttype = "akita_black_bowl"
    assert not ag._half_turn_same()


def test_a_plan_with_no_step_says_why_each_step_was_not_planned():
    """A plan made again with 40 steps left (as after the delay compensator freed a caught arm at step 260): no step
    fits, and the record says so step by step instead of only "no step planned"."""
    from types import SimpleNamespace

    import pytest

    from rowant_lab.libero_env.tests.test_sequence import _goal_scene
    from rowant_lab.pro2.robot import episode as EPI

    scene = _goal_scene()
    if scene is None:
        pytest.skip("the long-lib tree is not on this machine")
    lib, percept, cab = scene
    bowl = percept("p0", "akita_black_bowl", np.eye(3), [-0.085, 0.004, 0.898])
    tool = SimpleNamespace(percepts=[bowl, cab], target=None)
    params = {"sequence": True, "stage_sweep": "stage", "sequence_reserve": 130}
    out = EPI.plan_episode("open the top layer of the drawer and put the bowl inside", tool, lib, 40, params)
    g = out["grounding"]
    assert out["agent"] is None and g["status"] == "fail" and g["reason"] == "no step planned"
    assert [s["skipped"] for s in g["steps"]] == ["no budget", "no budget", "no budget"]
    assert g["steps"][0]["clause"].startswith("stage") and all(s["cap"] < 60 for s in g["steps"])
    full = EPI.plan_episode("open the top layer of the drawer and put the bowl inside",
                            SimpleNamespace(percepts=[bowl, cab], target=None), lib, 300, params)
    assert full["agent"] is not None and "steps" not in full["grounding"]  # a plan with a step: the record as it was


def test_an_approach_to_a_handle_that_stops_short_records_where_the_hand_stood_and_the_arms_joints():
    from dataclasses import replace
    from types import SimpleNamespace

    from rowant_lab.pro2.robot import drawer as DR

    def run(status, short_m):
        ag = object.__new__(DR.OpenAgent)
        ag.ap = replace(AG.AgentParams(), open_fast=True)
        ag.fp, ag.jaw, ag.cap, ag.t, ag.attempts, ag.finished = AG.FollowParams(), SimpleNamespace(L=0.04), 300, 0, [], False
        ag.hp = {"c": [0.03, -0.135, 1.09], "d": [0.0, 1.0, 0.0], "bar_h": 0.0165, "pull": 0.175}
        ag.pobs = {"robot0_joint_pos": np.array([0.1, 0.9, -0.2, -0.071, 0.3, 1.7, 0.8])}
        ag._pose = lambda: (np.array([0.03, -0.135 + short_m, 1.092]), R)
        ag._gap = lambda: 0.001  # the fingers close on nothing

        def move(st):
            def gen(*a, **k):
                return st
                yield  # (a generator that ends at once)
            return gen

        ag._goto, ag._goto_seq, ag._goto_g = move("reached"), move("reached"), move(status)
        ag._act_to = lambda *a, **k: [0.0] * 7
        ag._finish = lambda why: setattr(ag, "finish_reason", why)
        list(ag._episode())
        return ag

    ag = run("unreachable", 0.035)
    assert ag.finish_reason == "max_attempts" and [a["outcome"] for a in ag.attempts] == ["missed"] * 3
    a = ag.attempts[0]
    assert a["approach"] == "unreachable" and a["approach_err"] == 0.0351
    assert a["approach_off"] == [0.0, 0.035, 0.002] and a["approach_q"] == [0.1, 0.9, -0.2, -0.071, 0.3, 1.7, 0.8]
    ok = run("reached", 0.002).attempts[0]  # an approach that arrives: the record as it was
    assert ok["approach"] == "reached" and "approach_off" not in ok and "approach_q" not in ok
