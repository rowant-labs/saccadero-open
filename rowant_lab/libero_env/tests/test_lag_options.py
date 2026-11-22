"""2026-11-17 (stream LAG; every option off by default): through a delay, the waits of a whole lag that the robot does
not need before it goes on. The closing hold for the commands' delay alone with the grasp check read after the lift
(``close_hold_cmd``), the lift check without its wait (``lag_lift_no_settle``), the compensator's own steps with the
gripper command as the robot left it (``grip_as_sent``), and the drawer's jaws opened for the commands' delay alone
(``open_reopen_cmd``). The stand-in is ``probes/kin_same.py``'s world with late commands and late readings."""

import numpy as np
import pytest

from rowant_lab.libero_env import m2_library as LB
from rowant_lab.libero_env import m2_paths as MP
from rowant_lab.pro2.probes import kin_same as KS
from rowant_lab.pro2.robot import agent as AG
from rowant_lab.pro2.robot import delay as DL
from rowant_lab.pro2.robot import drawer as DR

SPEC = {"predicate": "in", "target": "p0", "place": "p1_contain_region", "place_kind": "region",
        "place_instance": "p1", "instruction": "pick the alphabet soup and place it in the basket"}
BUNDLE_DELAY = {"start_lag": True, "leap_trust": True, "replan_late_only": True, "no_answer_assume": True}


class LagWorld(KS.World):
    """The stand-in from step ``at`` to step ``until``: commands carried out ``d_act`` steps late (nothing in the gap)
    and readings, with the looks, of ``d_obs`` steps before."""

    def __init__(self, *a, at=8, d_act=7, d_obs=7, until=10 ** 6, **k):
        super().__init__(*a, **k)
        self.at, self.d_act, self.d_obs, self.until = at, d_act, d_obs, until
        self.sent, self.h_obs, self.h_true, self._g = [], [], [], -1.0
        self._snap()

    def _snap(self):
        o = KS.World.pobs(self)
        # joints that change with the hand (the compensator knows late readings by their exact repeats); the fourth
        # is the elbow, well bent
        o["robot0_joint_pos"] = np.array([*self.p, -1.0, self.gap, float(self.R[0, 0]), float(self.R[1, 0])])
        self.h_obs.append(o)
        self.h_true.append({k: v.copy() for k, v in self.true.items()})

    def _late(self) -> int:
        t = len(self.sent)
        return t - self.d_obs if (self.at <= t < self.until and self.d_obs) else t

    def pobs(self):
        return self.h_obs[max(self._late(), 0)]

    def late_true(self):
        return self.h_true[max(self._late(), 0)]

    def step(self, a):
        t = len(self.sent)
        self.sent.append(np.asarray(a, float))
        if self.at <= t < self.until and self.d_act:
            k = t - self.d_act
            a_now = self.sent[k] if k >= self.at else np.r_[np.zeros(6), self._g]
        else:
            a_now = self.sent[t]
        self._g = 1.0 if a_now[6] > 0 else -1.0
        KS.World.step(self, a_now)
        self._snap()


class LateLooks(KS.Looks):
    """Looks of the scene as the late readings' moment had it (the picture and the hand reading belong together)."""

    def __call__(self):
        self.calls += 1
        true = self.w.late_true()
        for q in self.percepts:
            pos = true[q["id"]].copy()
            if q["id"] == self.target:
                pos = pos + self.bias
            q["bodies"]["main"]["pos"] = [float(x) for x in pos]
        return self.w.lib.poses_from_percepts(self.percepts)


def _episode(delay=None, agent=None, world=None, looks=None, cap=600):
    """One pick and place in the stand-in through the runner's loop (``dev_episode``'s, in small)."""
    if not (MP.POLICY_LIB / "manifest.json").exists():
        pytest.skip("the object library is not built on this machine")
    R = KS.load(str(AG.__file__).rsplit("/", 1)[0])
    lib = LB.M2Library("libero_object")
    pc = KS._percepts(lib)
    w = LagWorld(R, lib, pc, **(world or {}))
    lk = LateLooks(w, pc, **(looks or {}))
    ag = R.agent.OracleAgent(lib, lib.scene_from_percepts(pc), SPEC, lk, cap, R.agent.AgentParams(
        **{"carry_watch": True, "c2_trust_gap": True, **(agent or {})}))
    comp = R.pkg.delay.DelayComp({**BUNDLE_DELAY, **(delay or {})}) if hasattr(R.pkg, "delay") else None
    if comp is None:
        import importlib

        comp = importlib.import_module(R.pkg.__name__ + ".delay").DelayComp({**BUNDLE_DELAY, **(delay or {})})
    w.frame_width = lambda: ((ag.attempts[-1].get("frame") or {}).get("width") if ag.attempts else None)
    obs = w.pobs()
    comp.start(obs)
    acts, hand = [], [w.p.copy()]
    while len(acts) < cap and not ag.finished:
        if comp.busy() or comp.holding():
            a = comp.own_action() if comp.busy() else comp.hold_action()
            ag.t += 1
        else:
            ag.lag_steps, ag.lag_obs_steps = int(comp.lag), int(comp.obs_lag)
            ag.lag_close_blind = bool(comp.o.get("close_hold_cmd"))
            a = ag.act(comp.predict(obs))
            comp.sent(a)
            a = comp.shape(a)
        acts.append([float(x) for x in a])
        w.step(a)
        hand.append(w.p.copy())
        obs = w.pobs()
        comp.after_step(a, obs)
        comp.replan_due()  # (the stand-in keeps its plan)
    basket = w.true["p1"]
    return {"n": len(acts), "acts": acts, "finish": ag.finish_reason, "attempts": ag.attempts, "comp": comp, "hand": hand,
            "in_basket": bool(np.linalg.norm((w.true["p0"] - basket)[:2]) < 0.08), "events": ag.events}


def _outcomes(res):
    return [a.get("outcome") for a in res["attempts"]]


def test_every_option_is_off_by_default():
    ap = AG.AgentParams()
    assert ap.lag_lift_no_settle is False and ap.open_reopen_cmd is False and ap.lag_settle_pre is False
    assert DL.DELAY["close_hold_cmd"] is False and DL.DELAY["grip_as_sent"] is False
    from rowant_lab.pro2 import robot as R

    ap, _pp, _fp = R.make_params({"agent": {"lag_lift_no_settle": True, "open_reopen_cmd": True}})
    assert ap.lag_lift_no_settle is True and ap.open_reopen_cmd is True


def test_a_closing_hand_is_held_for_the_commands_delay_alone_when_asked():
    for opts, lag, obs_lag, held_close, held_open in (
            ({}, 15, 7, 15, 15), ({"close_hold_cmd": True}, 15, 7, 7, 15), ({"close_hold_cmd": True}, 11, 10, 0, 11),
            ({"close_hold_cmd": True}, 8, 0, 8, 8), ({"close_hold_cmd": True, "open_hold_cmd": True}, 15, 7, 7, 7)):
        comp = DL.DelayComp(opts)
        comp.lag, comp.obs_lag = lag, obs_lag
        comp.sent([0, 0, 0, 0, 0, 0, -1.0])
        comp.sent([0, 0, 0, 0, 0, 0, 1.0])
        n_close = 0
        while comp.holding():
            assert comp.hold_action() == [0.0] * 6 + [1.0]
            n_close += 1
        comp.sent([0, 0, 0, 0, 0, 0, -1.0])
        n_open = 0
        while comp.holding():
            comp.hold_action()
            n_open += 1
        assert (n_close, n_open) == (held_close, held_open)


def test_the_compensators_own_steps_keep_the_fingers_as_the_robot_left_them_when_asked():
    def own(opts, cmds):
        comp = DL.DelayComp(opts)
        comp.start({"robot0_eef_pos": np.zeros(3), "robot0_joint_pos": np.zeros(7)})
        for g in cmds:
            comp.sent([0.1, 0, 0, 0, 0, 0, g])
        comp._measure("repeat")
        return [comp.own_action()[6] for _ in range(3)]

    shaped = [-1.0, -1.0, 1.0, 1.0, 0.0, 0.0]  # the drawer step: opened, part closed, then kept as they are
    assert own({}, shaped) == [1.0] * 3  # as before: the last command to open or close, repeated (the jaws shut)
    assert own({"grip_as_sent": True}, shaped) == [0.0] * 3
    for cmds in ([-1.0, -1.0], [-1.0, 1.0, 1.0]):  # a robot that never says "keep": the same either way
        assert own({}, cmds) == own({"grip_as_sent": True}, cmds) == [cmds[-1]] * 3
    comp = DL.DelayComp({"grip_as_sent": True})  # before the robot has sent anything: open, as before
    assert comp.own_action()[6] == -1.0


def test_without_a_delay_the_options_change_nothing():
    off = _episode(world={"at": 10 ** 6})
    on = _episode(world={"at": 10 ** 6}, delay={"close_hold_cmd": True, "grip_as_sent": True},
                  agent={"lag_lift_no_settle": True})
    assert off["acts"] == on["acts"] and off["in_basket"] and off["finish"] == "placed"


def test_late_commands_and_readings_the_grasp_is_read_after_the_lift_and_the_waits_go():
    # commands and readings 7 steps late from step 8 on (S20's form; the stand-in's hand answers in the step, so its
    # lag is 14 where the simulator's is 15)
    off = _episode()
    assert off["comp"].lag == 14 and off["finish"] == "placed" and off["in_basket"] and _outcomes(off) == ["placed"]
    n_settle = sum(e["n"] for e in off["events"] if e["ev"] == "settle")
    assert n_settle >= 28  # before the close and before the lift check, a whole lag each (and before the release)
    a = off["attempts"][0]
    assert a["c1"]["t"] - a["t_close"] >= 14 + 4  # the hold of a whole lag, then the closing watched

    lift = _episode(agent={"lag_lift_no_settle": True})
    assert lift["finish"] == "placed" and lift["in_basket"] and _outcomes(lift) == ["placed"]
    assert lift["attempts"][0]["lift_settle"] == "skipped" and lift["attempts"][0]["c2"]["stale"] is True
    assert lift["attempts"][0]["c2"]["pass"] is True and lift["n"] <= off["n"] - 13

    blind = _episode(delay={"close_hold_cmd": True})
    b = blind["attempts"][0]
    assert blind["finish"] == "placed" and blind["in_basket"] and _outcomes(blind) == ["placed"]
    assert b["c1"]["blind"] >= 3 and b["c1"]["pass"] is True and abs(b["c1"]["gap"] - b["c1"]["pred"]) < 0.002
    assert b["c1"]["t"] > b["t_lift"]  # read after the lift
    assert b["lift_settle"] == "skipped" and blind["n"] <= off["n"] - 20
    assert blind["comp"].rec["holds"] == off["comp"].rec["holds"]  # the same holds, shorter


def test_late_readings_alone_the_same_and_a_delay_that_ends_is_no_trouble():
    # readings 10 steps late from step 8 to step 70, commands on time (D12's form): lag 11, then none
    world = {"at": 8, "d_act": 0, "d_obs": 10, "until": 70}
    off = _episode(world=world)
    on = _episode(world=world, delay={"close_hold_cmd": True})
    for r in (off, on):
        assert r["finish"] == "placed" and r["in_basket"] and _outcomes(r) == ["placed"]
        assert [x["lag"] for x in r["comp"].rec["lags"] if x.get("from") in ("repeat", "pulse")][:2] == [11, 10]
    assert on["n"] < off["n"]
    assert on["attempts"][0]["c1"].get("blind") and on["attempts"][0]["c1"]["pass"] is True


def test_a_blind_close_on_nothing_is_found_after_the_lift_and_tried_again():
    # the look puts the object 5 cm from where it is: the fingers meet on nothing
    res = _episode(delay={"close_hold_cmd": True}, looks={"bias": (0.05, 0.03, 0.0)}, agent={"max_attempts": 2})
    first = res["attempts"][0]
    assert first["outcome"] == "c1_fail" and first["c1"]["blind"] and first["c1"]["gap"] < 0.003
    assert first["c1"]["t"] > first["t_lift"] and "c2" not in first
    assert len(res["attempts"]) == 2 and not res["in_basket"]
    # and the same close with the wait: found before the lift
    ref = _episode(looks={"bias": (0.05, 0.03, 0.0)}, agent={"max_attempts": 2})
    assert ref["attempts"][0]["outcome"] == "c1_fail" and "t_lift" not in ref["attempts"][0]


def test_without_the_fingers_watched_while_carrying_the_close_and_the_lift_check_wait_as_before():
    ref = _episode(agent={"carry_watch": False})
    res = _episode(delay={"close_hold_cmd": True}, agent={"carry_watch": False})
    a, b = ref["attempts"][0], res["attempts"][0]
    assert "blind" not in b["c1"] and "lift_settle" not in b and b["c1"]["pass"] is True
    # the hold was the commands' delay alone, and the closing waited the readings' delay before it believed the
    # fingers still: the same steps as the whole-lag hold, and the same gap read
    assert b["c1"]["t"] - b["t_close"] == a["c1"]["t"] - a["t_close"] and b["c1"]["gap"] == a["c1"]["gap"]
    assert res["finish"] == "placed" and res["in_basket"] and res["n"] == ref["n"]
    res = _episode(agent={"lag_lift_no_settle": True, "carry_watch": False})
    assert "lift_settle" not in res["attempts"][0] and res["n"] == ref["n"]


def test_the_drawers_jaws_are_opened_for_the_commands_delay_alone_when_asked():
    R0 = np.eye(3)
    p0 = np.array([0.08, -0.09, 1.02])

    def steps(lag, obs, **opts):
        ag = object.__new__(DR.OpenAgent)
        ag.ap, ag.fp = AG.AgentParams(**opts), AG.FollowParams()
        ag.lag_steps, ag.lag_obs_steps = lag, obs
        ag._pose = lambda: (p0, R0)
        ag._gap = lambda: 0.08
        return sum(1 for _ in ag._reopen({}, p0, R0))

    assert steps(0, 0) == steps(0, 0, open_reopen_cmd=True) == 8
    assert steps(15, 7) == 23 and steps(11, 10) == 19  # as before: the whole lag
    assert steps(15, 7, open_reopen_cmd=True) == 15  # S20: the commands are 7 late
    assert steps(11, 10, open_reopen_cmd=True) == 8  # D12: the commands are on time
    assert steps(8, 0, open_reopen_cmd=True) == 16  # commands late and readings on time: the whole lag is theirs


def _off_at_descent(res):
    """How far across the true hand stood from the grasp point when the first attempt's descent began (m)."""
    a = res["attempts"][0]
    return float(np.linalg.norm((res["hand"][a["t_approach"]] - np.asarray(a["frame"]["centre"]))[:2]))


def test_the_wait_before_the_descent_puts_the_hand_over_the_grasp_before_it_goes_down():
    # an arm that makes more of each command than the compensator believes (0.35 against 0.23: the pulse's own
    # measure is refused here), S20's form of delay: the hand reaches the pre-grasp point on its predicted pose
    wrong = {"gain_range": (0.2, 0.25)}
    off = _episode(delay=wrong)
    on = _episode(delay=wrong, agent={"lag_settle_pre": True})
    assert off["comp"].gain == 0.23 and on["comp"].gain == 0.23
    a, b = off["attempts"][0], on["attempts"][0]
    assert "pre_fix" not in a and a["approach_fix"]["status"] == "reached"
    assert b["approach_fix"] == {"skipped": "pre"} and b["pre_fix"]["status"] == "reached"
    assert _off_at_descent(off) > 0.010  # went down over a centimetre off its point
    assert _off_at_descent(on) < 0.003  # the correcting move's last commands were waited for
    assert b["pre_fix"]["left_m"] > 0.006 and b["pre_fix"]["rounds"] == 1  # what the wait found, before any descent
    # an arm further off than ``lag_pre_ok_m`` after the first wait is waited for and corrected once more
    twice = _episode(delay=wrong, agent={"lag_settle_pre": True, "lag_pre_ok_m": 0.002})
    c = twice["attempts"][0]
    assert c["pre_fix"]["rounds"] == 2 and c["pre_fix"]["left_last_m"] < 0.5 * c["pre_fix"]["left_m"]
    assert twice["finish"] == "placed" and twice["in_basket"]
    for r in (off, on):
        assert r["finish"] == "placed" and r["in_basket"]
    assert on["n"] <= off["n"] + 12  # the same wait, taken earlier, and the commands' delay after the correction
    settles = [e["t"] for e in on["events"] if e["ev"] == "settle"]
    assert settles and settles[0] < b["t_approach"] and not any(b["t_approach"] <= s < b["t_close"] for s in settles)
    # with everything together: the wait before the descent, the blind close, no wait at the lift
    all_on = _episode(delay={**wrong, "close_hold_cmd": True}, agent={"lag_settle_pre": True})
    assert all_on["finish"] == "placed" and all_on["in_basket"] and all_on["n"] <= off["n"] - 14
    # and without a delay nothing is waited for
    plain = _episode(world={"at": 10 ** 6}, agent={"lag_settle_pre": True})
    assert "pre_fix" not in plain["attempts"][0] and plain["acts"] == _episode(world={"at": 10 ** 6})["acts"]


def test_a_closing_that_watches_the_fingers_waits_the_readings_delay_only_on_a_shortened_hold():
    ag = object.__new__(DR.OpenAgent)
    assert ag._close_wait() == 0  # nothing told by the runner
    ag.lag_steps, ag.lag_obs_steps, ag.lag_close_blind = 15, 7, False
    assert ag._close_wait() == 0  # the whole-lag hold: as before
    ag.lag_close_blind = True
    assert ag._close_wait() == 8  # the hold was 7: the fingers' reading is 7 and a step behind
    ag.lag_steps, ag.lag_obs_steps = 11, 10
    assert ag._close_wait() == 11  # readings alone late: no hold at all, the whole lag waited here
    ag.lag_steps = 0
    assert ag._close_wait() == 0  # no delay in use
    assert object.__new__(DR.RetakeAgent)._close_wait() == 0


# ------------------------------------------------------------------ ``probe_grip`` (2026-11-17): the fingers are asked
from rowant_lab.pro2.robot.follow import OUT_POS, follow  # noqa: E402
from rowant_lab.pro2.robot.grasp import quat_xyzw_to_mat  # noqa: E402

QUAT = np.array([0.0, 0.0, 0.0, 1.0])


class PlantG:
    """A hand that makes ``kappa`` of each commanded move in a step, with fingers that travel 12 mm a step. From step
    ``at`` to step ``until`` commands are carried out ``d_act`` steps late (nothing in the gap, the fingers kept) and
    readings show the hand and the fingers of ``d_obs`` steps earlier."""

    def __init__(self, kappa=0.26, at=7, d_act=7, d_obs=7, until=10 ** 9, gap=0.08):
        self.kappa, self.at, self.d_act, self.d_obs, self.until = kappa, at, d_act, d_obs, until
        self.p, self.gap = np.zeros(3), float(gap)
        self.hist, self.sent = [(self.p.copy(), self.gap)], []

    def obs(self):
        t = len(self.sent)
        k = t - self.d_obs if (self.at <= t < self.until and self.d_obs) else t
        p, g = self.hist[max(k, 0)]
        return {"robot0_eef_pos": p.copy(), "robot0_eef_quat": QUAT.copy(),
                "robot0_joint_pos": np.concatenate([p, [-1.0, g, 0.0, 0.0]]),
                "robot0_gripper_qpos": np.array([0.5 * g, -0.5 * g])}

    def step(self, a):
        t = len(self.sent)
        self.sent.append(np.asarray(a, float))
        if self.at <= t < self.until and self.d_act:
            k = t - self.d_act
            a_now = self.sent[k] if k >= self.at else np.zeros(7)
        else:
            a_now = self.sent[t]
        self.p = self.p + self.kappa * OUT_POS * a_now[:3]
        if a_now[6] > 0:
            self.gap = max(0.0013, self.gap - 0.012)
        elif a_now[6] < 0:
            self.gap = min(0.08, self.gap + 0.012)
        self.hist.append((self.p.copy(), self.gap))
        return self.obs()


def _reach(plant, opts, target=(0.30, -0.12, 0.06), steps=140, close_at=None):
    """The follower's hand to ``target`` through the compensator, as the runner does it. Returns the compensator, the
    steps it took itself, the plans asked again, and the step at which the hand was within 5 mm for good. (The
    compensator is given the plant's own gain: the pulse measures it and the probe does not; in the simulator the
    pulse's measure never rose above the built-in one in the stored rows.)"""
    comp = DL.DelayComp({**BUNDLE_DELAY, "gain": plant.kappa, **opts})
    target = np.asarray(target, float)
    obs = plant.obs()
    comp.start(obs)
    own, again, errs = 0, 0, []
    for i in range(steps):
        if comp.busy() or comp.holding():
            a = comp.own_action() if comp.busy() else comp.hold_action()
            own += 1
        else:
            seen = comp.predict(obs)
            a = follow(seen["robot0_eef_pos"], quat_xyzw_to_mat(seen["robot0_eef_quat"]), target, np.eye(3),
                       1.0 if (close_at is not None and i >= close_at) else -1.0)
            comp.sent(a)
            a = comp.shape(a)
        obs = plant.step(np.asarray(a, float))
        comp.after_step(a, obs)
        again += int(comp.replan_due())
        errs.append(float(np.linalg.norm(plant.p - target)))
    there = next((i for i in range(len(errs)) if max(errs[i:]) < 0.005), None)
    return comp, own, again, there


def test_the_probe_is_off_by_default_and_changes_nothing_without_a_delay():
    assert DL.DELAY["probe_grip"] is False
    a = _reach(PlantG(at=10 ** 9), {})
    b = _reach(PlantG(at=10 ** 9), {"probe_grip": True})
    assert a[1:] == b[1:] and b[1] == 0 and "probe" not in b[0].rec and not b[0]._pg


def test_late_readings_alone_are_confirmed_by_the_fingers_without_a_pause():
    # readings 10 steps late, commands on time (D12's form)
    old, own_old, again_old, there_old = _reach(PlantG(d_act=0, d_obs=10), {})
    new, own_new, again_new, there_new = _reach(PlantG(d_act=0, d_obs=10), {"probe_grip": True})
    assert own_old >= 19 and [x["from"] for x in old.rec["lags"]][:2] == ["repeat", "pulse"]
    assert own_new == 0 and [x["from"] for x in new.rec["lags"]] == ["repeat", "grip"]
    assert new.lag == old.lag == 10 and again_new == again_old == 0
    assert there_new is not None and there_new <= there_old - 12  # the hand is there a dozen steps sooner
    # at full commands during the probe it is there as soon as with no delay at all
    full = _reach(PlantG(d_act=0, d_obs=10), {"probe_grip": True, "probe_scale": 1.0})
    assert full[1] == 0 and full[3] <= _reach(PlantG(at=10 ** 9), {})[3] + 2
    assert not new._pg and new.mode == "run"


def test_commands_late_too_stop_the_robot_only_until_the_fingers_answer():
    # commands and readings 7 steps late each (S20's form)
    old, own_old, again_old, there_old = _reach(PlantG(d_act=7, d_obs=7), {})
    new, own_new, again_new, there_new = _reach(PlantG(d_act=7, d_obs=7), {"probe_grip": True})
    assert old.lag == new.lag == 14 and again_old == again_new == 1
    assert own_old >= 23
    grip = [x for x in new.rec["lags"] if x["from"] == "grip"][0]
    assert grip["lag"] == 14 and grip["stopped"] >= 1 and 4 <= own_new <= 14 and own_new < own_old - 8
    assert there_new is not None and there_new <= there_old - 8
    assert not new._pg and new.mode == "run"


def test_the_probes_commands_close_the_fingers_twice_and_halve_the_robots_until_the_answer():
    plant = PlantG(d_act=0, d_obs=10)
    comp = DL.DelayComp({**BUNDLE_DELAY, "probe_grip": True})
    obs = plant.obs()
    comp.start(obs)
    sent, shaped = [], []
    for _ in range(40):
        assert not comp.busy()
        a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        comp.sent(a)
        b = [float(x) for x in comp.shape(a)]
        sent.append(a)
        shaped.append(b)
        obs = plant.step(np.asarray(b))
        comp.after_step(b, obs)
    k0 = next(i for i, b in enumerate(shaped) if b != sent[i])
    grip = [x for x in comp.rec["lags"] if x["from"] == "grip"][0]
    assert [b[6] for b in shaped[k0:k0 + 4]] == [1.0, 1.0, -1.0, -1.0]
    assert all(b[0] == 0.5 for b in shaped[k0:grip["t"]]) and all(b == sent[i] for i, b in enumerate(shaped) if i >= grip["t"])
    assert grip["steps"] == grip["t"] - k0 and grip["lag"] == 10


def test_a_robot_that_closes_its_hand_or_fingers_not_open_and_still_leave_it_to_the_pulse():
    # the robot closes two steps into the probe (the readings are found late at step 12)
    comp, own, _again, _there = _reach(PlantG(d_act=0, d_obs=10), {"probe_grip": True}, close_at=14)
    assert {"robot_closed": True}.items() <= comp.rec["probe"][0].items()
    assert [x["from"] for x in comp.rec["lags"]][:2] == ["repeat", "pulse"] and own >= 19
    # fingers part shut when the readings are found late
    comp, own, _again, _there = _reach(PlantG(d_act=0, d_obs=10, gap=0.02), {"probe_grip": True})
    assert "probe" not in comp.rec and [x["from"] for x in comp.rec["lags"]][:2] == ["repeat", "pulse"]
    # a hand that has closed before (it holds something)
    comp = DL.DelayComp({"probe_grip": True})
    comp.gaps, comp._grip_raw = [0.08, 0.08, 0.08], -1.0
    assert comp._probe_ok()
    comp._closed = True
    assert not comp._probe_ok()


def test_a_delay_that_ends_during_the_probe_ends_the_probe():
    comp, own, _again, there = _reach(PlantG(d_act=0, d_obs=10, until=14), {"probe_grip": True})
    assert comp.lag == 0 and not comp._pg and comp.mode == "run" and there is not None


def test_the_probe_in_the_stand_ins_pick_and_place():
    # late readings alone, ending at step 70 (D12's form), and late commands and readings (S20's form)
    # (the compensator is given the stand-in's own gain, which the pulse would have measured)
    world = {"at": 8, "d_act": 0, "d_obs": 10, "until": 70}
    off, on = _episode(world=world, delay={"gain": 0.35}), _episode(world=world, delay={"gain": 0.35, "probe_grip": True})
    assert on["finish"] == "placed" and on["in_basket"] and _outcomes(on) == ["placed"]
    assert [x["from"] for x in on["comp"].rec["lags"]][:2] == ["repeat", "grip"]
    assert on["attempts"][0]["t_approach"] <= off["attempts"][0]["t_approach"] - 12 and on["n"] <= off["n"] - 12
    off, on = _episode(delay={"gain": 0.35}), _episode(delay={"gain": 0.35, "probe_grip": True})
    assert on["finish"] == "placed" and on["in_basket"] and on["comp"].lag == 14
    assert [x["from"] for x in on["comp"].rec["lags"]][:2] == ["repeat", "grip"]
    assert on["attempts"][0]["t_approach"] <= off["attempts"][0]["t_approach"] - 6 and on["n"] <= off["n"] - 6
    # and with everything of 2026-11-17 together
    every = _episode(delay={"gain": 0.35, "probe_grip": True, "close_hold_cmd": True, "grip_as_sent": True},
                     agent={"lag_settle_pre": True, "lag_lift_no_settle": True})
    assert every["finish"] == "placed" and every["in_basket"] and every["n"] <= off["n"] - 22


# ------------------------------------------------------------------ ``grip_echo``: the robot's own gripper commands
def test_the_fingers_answer_to_the_robots_own_close_ends_a_lag_that_outlived_its_delay():
    # readings 10 late from step 7 to step 60; the hand is at its target and at rest when they stop being late, so
    # no leap shows it. The robot closes its hand at step 80
    old, _own, _again, _there = _reach(PlantG(d_act=0, d_obs=10, until=60), {}, close_at=80)
    assert old.lag == 10 and not any(x.get("why") == "close" for x in old.rec["lags"])  # the lag stood to the end
    new, _own, _again, _there = _reach(PlantG(d_act=0, d_obs=10, until=60), {"grip_echo": True}, close_at=80)
    end = [x for x in new.rec["lags"] if x.get("from") == "grip"]
    assert new.lag == 0 and new.obs_lag == 0 and len(end) == 1
    assert end[0]["why"] == "close" and end[0]["was"] == 10 and end[0]["lag"] == 0 and 80 < end[0]["t"] <= 83
    # the delay ends between the command (step 80) and its answer (due at 91): the answer comes at 86, sooner than
    # the lag in use allows, and that too says the delay is over (it is not a lag of 5)
    mid, _own, _again, _there = _reach(PlantG(d_act=0, d_obs=10, until=86), {"grip_echo": True}, close_at=80)
    end = [x for x in mid.rec["lags"] if x.get("from") == "grip"]
    assert mid.lag == 0 and len(end) == 1 and end[0]["lag"] == 0 and 2 <= end[0]["after"] <= 8


def test_an_answer_at_the_lag_in_use_changes_nothing_and_a_late_one_with_no_lag_in_use_is_only_noted():
    # the delay still there when the robot closes: the fingers answer as late as the lag says
    comp, _own, _again, _there = _reach(PlantG(d_act=0, d_obs=10), {"grip_echo": True}, close_at=80)
    assert comp.lag == 10 and [x["from"] for x in comp.rec["lags"]] == ["repeat", "pulse"] and "echo" not in comp.rec
    comp, _own, _again, _there = _reach(PlantG(d_act=7, d_obs=7), {"grip_echo": True}, close_at=80)
    assert comp.lag == 14 and not any(x.get("from") == "grip" for x in comp.rec["lags"])
    # commands alone late from step 60 on (nothing shows it, the hand is at rest): the answer is noted and the
    # lag is left to what measures it
    comp, _own, _again, _there = _reach(PlantG(at=60, d_act=8, d_obs=0), {"grip_echo": True}, close_at=80)
    assert comp.lag == 0 and comp.rec["echo"] == [{"t": comp.rec["echo"][0]["t"], "lag": 8, "why": "close", "was": 0}]
    # a later answer never lengthens a lag in use, and fingers letting go are not noted at all (they answer late by
    # themselves when they held something)
    comp = DL.DelayComp({"grip_echo": True})
    comp.lag, comp.obs_lag, comp.gaps = 11, 10, [0.03, 0.03, 0.03]
    comp.acts = [np.zeros(7)] * 40
    comp._ge = {"k": 20, "g0": 0.03, "dir": -1.0}
    comp.gaps.append(0.04)
    comp._echo_step()                               # the fingers opened 19 steps after the command: 8 later than the lag
    assert comp.lag == 11 and "echo" not in comp.rec and not comp._ge
    comp._ge = {"k": 20, "g0": 0.08, "dir": 1.0}
    comp.gaps.append(0.07)
    comp._echo_step()
    assert comp.lag == 11 and comp.rec["echo"][0]["lag"] == 19 and comp.rec["echo"][0]["was"] == 11
    # and without a delay, or without the option, nothing is noted or changed
    comp, _own, _again, _there = _reach(PlantG(at=10 ** 9), {"grip_echo": True}, close_at=80)
    assert comp.lag == 0 and "echo" not in comp.rec and comp.rec["lags"] == []
    assert DL.DELAY["grip_echo"] is False


def test_fingers_that_were_moving_or_a_probe_just_sent_are_not_timed():
    comp = DL.DelayComp({"grip_echo": True})
    comp.lag, comp.obs_lag = 11, 10
    comp.gaps = [0.08, 0.07, 0.06]                 # still opening or closing: no rest to time from
    comp.sent([0, 0, 0, 0, 0, 0, -1.0])
    comp.sent([0, 0, 0, 0, 0, 0, 1.0])
    assert not comp._ge
    comp = DL.DelayComp({"grip_echo": True})
    comp.lag, comp.obs_lag = 11, 10
    comp.gaps = [0.08, 0.08, 0.08]
    comp.acts = [np.zeros(7)] * 30
    comp._pg_last = 20                              # a probe began ten steps ago: its dip is still on its way
    comp.sent([0, 0, 0, 0, 0, 0, -1.0])
    comp.sent([0, 0, 0, 0, 0, 0, 1.0])
    assert not comp._ge
    comp._pg_last, comp._grip = 5, -1.0
    comp.sent([0, 0, 0, 0, 0, 0, 1.0])
    assert comp._ge == {"k": 30, "g0": 0.08, "dir": 1.0}


def test_the_wait_before_the_descent_is_for_late_commands_only():
    ag = object.__new__(AG.OracleAgent)
    assert not ag._cmds_late()
    for lag, obs, late in ((15, 7, True), (11, 10, False), (8, 0, True), (0, 0, False), (9, 7, True), (8, 7, False)):
        ag.lag_steps, ag.lag_obs_steps = lag, obs
        assert ag._cmds_late() is late
    # readings alone late (D12's form): the old order, the wait after the descent
    res = _episode(world={"at": 8, "d_act": 0, "d_obs": 10}, agent={"lag_settle_pre": True})
    a = res["attempts"][0]
    assert "pre_fix" not in a and a["approach_fix"]["status"] == "reached" and res["finish"] == "placed"


def test_a_blind_close_gives_way_when_the_lag_ends_under_it():
    # readings 10 late from step 8; they stop being late at step 60, with the hand at rest over the grasp (no leap).
    # The robot closes blind; the fingers answer at once; the hand opens, is put on its point by the readings as they
    # now are, and closes the ordinary way
    world = {"at": 8, "d_act": 0, "d_obs": 10, "until": 60}
    ref = _episode(world=world, delay={"gain": 0.35, "close_hold_cmd": True})
    res = _episode(world=world, delay={"gain": 0.35, "close_hold_cmd": True, "grip_echo": True})
    a = res["attempts"][0]
    ended = [x for x in res["comp"].rec["lags"] if x.get("from") == "grip" and x.get("why") == "close"]
    if a["t_close"] > 60:                           # (the close came after the delay's end in this run of the stand-in)
        assert ended and ended[0]["lag"] == 0 and "blind_ended" in a and "blind" not in a["c1"]
        assert a["c1"]["pass"] is True and "lift_settle" not in a and "stale" not in a["c2"]
    assert res["finish"] == "placed" and res["in_basket"] and _outcomes(res) == ["placed"]
    assert res["comp"].lag == 0 and res["n"] <= ref["n"]
