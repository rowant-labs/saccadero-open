"""robot/delay.py (2026-11-15): the loop's lag from the robot's own motion, the hand predicted forward, gripper holds."""

import numpy as np

from rowant_lab.pro2.robot import delay as DL
from rowant_lab.pro2.robot.follow import OUT_POS, follow
from rowant_lab.pro2.robot.grasp import quat_xyzw_to_mat

QUAT = [0.0, 0.0, 0.0, 1.0]


class Plant:
    """A hand that makes ``kappa`` of each commanded move in a step. From step ``at`` on, commands are applied
    ``d_act`` steps late (no-ops fill the gap) and readings show the hand of ``d_obs`` steps earlier."""

    def __init__(self, kappa=0.26, at=7, d_act=7, d_obs=7):
        self.kappa, self.at, self.d_act, self.d_obs = kappa, at, d_act, d_obs
        self.p = np.zeros(3)
        self.hist = [self.p.copy()]
        self.sent = []

    def obs(self):
        t = len(self.sent)
        k = t - self.d_obs if (t >= self.at and self.d_obs) else t
        return {"robot0_eef_pos": self.hist[max(k, 0)].copy(), "robot0_eef_quat": np.asarray(QUAT, float),
                "robot0_joint_pos": np.concatenate([self.hist[max(k, 0)], np.zeros(4)])}

    def step(self, a):
        t = len(self.sent)
        self.sent.append(np.asarray(a, float))
        if t >= self.at and self.d_act:
            k = t - self.d_act
            a_now = self.sent[k] if k >= self.at else np.zeros(7)
        else:
            a_now = self.sent[t]
        self.p = self.p + self.kappa * OUT_POS * a_now[:3]
        self.hist.append(self.p.copy())
        return self.obs()


def _drive(plant, comp, target, steps):
    obs = plant.obs()
    if comp is not None:
        comp.start(obs)
    errs = []
    for _ in range(steps):
        if comp is not None and comp.busy():
            a = comp.own_action()  # the lag is being measured: the robot is not asked
        else:
            seen = comp.predict(obs) if comp is not None else obs
            a = follow(seen["robot0_eef_pos"], quat_xyzw_to_mat(seen["robot0_eef_quat"]), target, np.eye(3), -1.0)
            if comp is not None:
                comp.sent(a)
        obs = plant.step(a)
        if comp is not None:
            comp.after_step(a, obs)
        errs.append(float(np.linalg.norm(plant.p - target)))
    return errs


def test_without_a_delay_nothing_changes():
    plant, comp = Plant(at=10 ** 9), DL.DelayComp()
    errs = _drive(plant, comp, np.array([0.2, -0.1, 0.05]), 60)
    assert comp.lag == 0 and comp.rec["lags"] == [] and errs[-1] < 0.002 and not comp.busy()
    obs = plant.obs()
    assert comp.predict(obs) is obs and not comp.holding()


def test_a_delayed_loop_swings_and_the_measured_lag_settles_it():
    target = np.array([0.25, -0.10, 0.05])
    bare = _drive(Plant(), None, target, 150)
    assert max(bare[100:]) > 0.03  # as S20's episodes: still swinging at the end
    plant, comp = Plant(), DL.DelayComp()
    errs = _drive(plant, comp, target, 150)
    # the readings' own 7 steps show at once (they repeat those of 7 steps before); the robot is paused, the hand
    # comes to rest, one pulse is sent, and the step of the answer is the whole lag
    assert comp.rec["repeats"] == [{"t": 9, "obs_lag": 7}] and comp.rec["lags"][0] == {"t": 9, "lag": 8, "from": "repeat"}
    pulse = comp.rec["lags"][1]
    assert pulse["from"] == "pulse" and pulse["lag"] == 14 and abs(pulse["gain"] - 0.26) < 0.01 and pulse["t"] < 60
    off = DL.DelayComp()  # without the fit (the default) a delay of the commands alone is left alone
    _drive(Plant(d_act=6, d_obs=0), off, target, 120)
    assert off.lag == 0 and off.rec["lags"] == []
    assert comp.lag == 14 and not comp.busy()
    assert max(errs[110:]) < 0.004  # at rest on the target
    assert max(errs[9:]) <= errs[9] + 0.02  # and never further from it than when the delay was noticed (the pulse aside)


def test_the_lag_is_measured_whatever_its_length_and_a_delay_of_the_commands_alone_is_fitted():
    for d_act, d_obs in ((3, 2), (0, 6), (10, 10), (2, 9)):
        plant, comp = Plant(d_act=d_act, d_obs=d_obs), DL.DelayComp()
        errs = _drive(plant, comp, np.array([0.3, 0.1, 0.0]), 220)
        assert comp.lag == d_act + d_obs and max(errs[170:]) < 0.004
        assert [x["from"] for x in comp.rec["lags"][:2]] == ["repeat", "pulse"]
    plant, comp = Plant(d_act=6, d_obs=0), DL.DelayComp({"fit": True})  # no reading repeats: the fit of the robot's own motion
    errs = _drive(plant, comp, np.array([0.3, 0.1, 0.0]), 220)
    assert comp.lag == 6 and "repeats" not in comp.rec and max(errs[170:]) < 0.004
    plant, comp = Plant(d_act=1, d_obs=0), DL.DelayComp({"fit": True})  # one step late: below min_lag, taken as the arm's own
    _drive(plant, comp, np.array([0.3, 0.1, 0.0]), 80)
    assert comp.lag == 0


class Passing(Plant):
    """Readings ``d_obs`` steps late from step ``at`` until step ``until``, then on time again."""

    def __init__(self, until, **kw):
        super().__init__(**kw)
        self.until = until

    def obs(self):
        t = len(self.sent)
        k = t - self.d_obs if (self.at <= t < self.until) else t
        return {"robot0_eef_pos": self.hist[max(k, 0)].copy(), "robot0_eef_quat": np.asarray(QUAT, float),
                "robot0_joint_pos": np.concatenate([self.hist[max(k, 0)], np.zeros(4)])}


def test_a_delay_that_ends_is_noticed_and_measured_away():
    plant, comp = Passing(until=60, at=12, d_act=0, d_obs=10), DL.DelayComp()
    errs = _drive(plant, comp, np.array([0.6, -0.3, 0.2]), 200)  # a long way: still moving when the delay ends
    kinds = [(x.get("from", "fit"), x["lag"]) for x in comp.rec["lags"]]
    assert kinds[:2] == [("repeat", 11), ("pulse", 10)]  # the readings' 10 steps, and nothing more behind them
    assert comp.rec["leaps"] and comp.rec["leaps"][0]["t"] in (60, 61) and kinds[-1] == ("pulse", 0) and len(kinds) == 3
    assert comp.lag == 0 and max(errs[150:]) < 0.004


def test_the_predicted_hand_adds_the_commands_still_on_their_way():
    comp = DL.DelayComp()
    comp.lag, comp.gain = 3, 0.2
    comp.acts = [np.array([1.0, 0, 0, 0, 0, 0, -1]), np.array([0, 1.0, 0, 0, 0, 0.5, -1]), np.array([0, 0, -1.0, 0, 0, 0.5, -1]),
                 np.array([0.5, 0, 0, 0, 0, 0, -1])]
    out = comp.predict({"robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.asarray(QUAT, float), "x": 1})
    assert np.allclose(out["robot0_eef_pos"], 0.2 * OUT_POS * np.array([0.5, 1.0, -1.0])) and out["x"] == 1
    R = quat_xyzw_to_mat(out["robot0_eef_quat"])  # two turns of 0.2 * 0.5 * 0.5 rad about the world's z
    assert np.allclose(R, DL._exp([0, 0, 0.1]), atol=1e-9)
    for r in ([0.3, -0.2, 0.9], [3.0, 0.1, 0.0], [0.0, 0.0, 0.0]):
        assert np.allclose(quat_xyzw_to_mat(DL.mat_to_quat_xyzw(DL._exp(r))), DL._exp(r), atol=1e-9)


def test_a_change_of_the_grippers_direction_holds_the_hand_for_the_lag():
    comp = DL.DelayComp()
    comp.sent([0, 0, 0, 0, 0, 0, -1.0])
    comp.sent([0, 0, 0, 0, 0, 0, 1.0])
    assert not comp.holding()  # no lag: no hold
    comp.lag = 4
    comp.sent([0.3, 0, 0, 0, 0, 0, 1.0])
    assert not comp.holding()  # the same direction
    comp.sent([0, 0, 0, 0, 0, 0, 0.0])
    assert not comp.holding()  # "keep as it is" is not a change
    comp.sent([0.3, 0, 0, 0, 0, 0, -1.0])
    held = []
    while comp.holding():
        held.append(comp.hold_action())
    assert held == [[0.0] * 6 + [-1.0]] * 4 and comp.rec["holds"] == 1


def test_a_hand_that_only_stops_is_not_taken_for_the_pulses_answer():
    comp = DL.DelayComp()
    comp._m = {"k0": 5}
    z = [0.0] * 6 + [0.006 * k for k in range(1, 7)] + [0.036] * 12  # rising, then at rest: half the shape
    comp.pos = [np.array([0.0, 0.0, v]) for v in z]
    comp.acts = [np.zeros(7)] * (len(z) - 1)
    assert comp._answer() is None
    up_down = [0.0] * 9 + [0.0065, 0.013, 0.0195, 0.013, 0.0065, 0.0] + [0.0] * 8  # the pulse itself, 3 steps late
    comp.pos = [np.array([0.0, 0.0, v]) for v in up_down]
    comp.acts = [np.zeros(7)] * (len(up_down) - 1)
    lag, score = comp._answer()
    assert lag == 3 and abs(score - 6 * 0.0065) < 1e-9


# ---- 2026-11-16: the lag from the first move after a rest, late or held commands, the pause until rest


class Heavy(Plant):
    """A hand that gathers speed: each step its velocity closes ``beta`` of the gap to the commanded one (the real
    arm turns round in three or four steps). No delay unless asked."""

    def __init__(self, beta=0.5, **kw):
        super().__init__(**{"at": 10 ** 9, **kw})
        self.beta, self.v = beta, np.zeros(3)

    def carried_out(self, t):
        return self.sent[t]

    def step(self, a):
        t = len(self.sent)
        self.sent.append(np.asarray(a, float))
        a_now = np.asarray(self.carried_out(t), float)
        self.v = self.v + self.beta * (self.kappa * OUT_POS * a_now[:3] - self.v)
        self.p = self.p + self.v
        self.hist.append(self.p.copy())
        return self.obs()


class Held(Heavy):
    """D14: from step ``at`` the command of the step before is carried out for ``n`` steps, whatever is sent."""

    def __init__(self, hold_at, n, **kw):
        super().__init__(**kw)
        self.hold_at, self.n = hold_at, n

    def carried_out(self, t):
        return self.sent[self.hold_at - 1] if self.hold_at <= t < self.hold_at + self.n else self.sent[t]


class Late(Heavy):
    """D13: from step ``late_at`` to ``until`` every command is carried out ``d`` steps late; then at once."""

    def __init__(self, late_at, until, d, **kw):
        super().__init__(**kw)
        self.late_at, self.until, self.d = late_at, until, d

    def carried_out(self, t):
        return self.sent[t - self.d] if self.late_at <= t < self.until else self.sent[t]


def _legs(plant, comp, legs):
    """Drive to each target in turn for its number of steps; the errors of the last leg, and whether a new look was
    asked for on the way."""
    obs = plant.obs()
    comp.start(obs)
    again, errs = 0, []
    for target, steps in legs:
        errs = []
        for _ in range(steps):
            if comp.busy():
                a = comp.own_action()
            else:
                seen = comp.predict(obs)
                a = follow(seen["robot0_eef_pos"], quat_xyzw_to_mat(seen["robot0_eef_quat"]), target, np.eye(3), -1.0)
                comp.sent(a)
            obs = plant.step(a)
            comp.after_step(a, obs)
            again += bool(comp.replan_due())
            errs.append(float(np.linalg.norm(plant.p - target)))
    return errs, again


NEW = {"start_lag": True, "cmd_watch": True}
LEGS = [(np.array([0.20, -0.10, 0.05]), 60), (np.array([-0.05, 0.10, 0.0]), 60), (np.array([0.10, 0.0, 0.12]), 60)]


def test_without_a_delay_the_new_watchers_leave_the_robot_alone():
    plant, comp = Heavy(), DL.DelayComp(NEW)
    errs, again = _legs(plant, comp, LEGS)  # three legs with full-speed turns between them
    assert comp.lag == 0 and comp.rec["lags"] == [] and "watch" not in comp.rec and again == 0 and errs[-1] < 0.002
    # every move after a rest was timed, and each was answered at once
    assert len(comp.rec["starts"]) == 2 and all(s["lag"] <= 1 and s["was"] == 0 for s in comp.rec["starts"])
    # the same commands as with the options off, step for step
    plant0, comp0 = Heavy(), DL.DelayComp()
    _legs(plant0, comp0, LEGS)
    assert np.allclose(np.array(plant.sent), np.array(plant0.sent))


def test_a_delay_that_ends_with_the_hand_at_rest_is_read_from_the_next_move():
    def run(opts):
        plant, comp = Passing(until=95, at=12, d_act=0, d_obs=10), DL.DelayComp(opts)
        errs, _ = _legs(plant, comp, [(np.array([0.12, 0.0, 0.0]), 120), (np.array([0.12, 0.2, 0.0]), 60)])
        return comp, errs

    comp, errs = run({})
    kinds = [(x["from"], x["lag"]) for x in comp.rec["lags"]]
    assert kinds == [("repeat", 11), ("pulse", 10)] and "leaps" not in comp.rec and comp.lag == 10  # never noticed
    comp, errs = run({"start_lag": True})
    kinds = [(x["from"], x["lag"]) for x in comp.rec["lags"]]
    assert kinds == [("repeat", 11), ("pulse", 10), ("start", 0)] and comp.lag == 0
    assert 120 < comp.rec["lags"][-1]["t"] <= 124 and errs[-1] < 0.004  # at the second leg's first steps
    # while the delay lasts, a move after a rest reads the lag in use and changes nothing
    plant, comp = Plant(at=7, d_act=0, d_obs=10), DL.DelayComp({"start_lag": True})
    _legs(plant, comp, [(np.array([0.12, 0.0, 0.0]), 120), (np.array([0.12, 0.2, 0.0]), 80)])
    assert comp.lag == 10 and [s for s in comp.rec["starts"] if abs(s["lag"] - 10) <= 1]
    assert [(x["from"], x["lag"]) for x in comp.rec["lags"]] == [("repeat", 11), ("pulse", 10)]


def test_a_held_command_is_seen_waited_out_and_planned_again():
    target = np.array([0.15, 0.0, 0.0])
    plant, comp = Held(hold_at=8, n=20), DL.DelayComp(NEW)
    errs, again = _legs(plant, comp, [(target, 120)])
    assert max(float(p[0]) for p in plant.hist) > 0.30  # the hand flew twice as far as it was sent: nothing stops that
    w = comp.rec["watch"]
    assert len(w) == 1 and 14 <= w[0]["t"] <= 22  # five steps after the hand passed its target
    # from the watch on the hand was pulled back toward where it had turned (ignored while the hold lasted, taking
    # hold in the step it ended); once the readings answered nothing more was sent until rest; then the robot looked
    # again and its first move, answered at once, said no lag was left: no pulse was spent
    sent = np.array(plant.sent)
    assert np.all(sent[w[0]["t"]:28, 0] < -0.5) and w[0]["pulled"] <= 28 - w[0]["t"] + 3
    far = max(float(p[0]) for p in plant.hist)
    assert float(plant.hist[28 + 4][0]) < far - 0.01  # coming back within four steps of the hold's end
    assert comp.rec["lags"] == [] and comp.rec["starts"][0]["lag"] <= 1 and 28 < comp.rec["starts"][0]["t"] < 46
    assert comp.lag == 0 and again == 1 and errs[-1] < 0.002
    # without the timed start (``cmd_watch`` alone) one pulse says the same
    plant, comp = Held(hold_at=8, n=20), DL.DelayComp({"cmd_watch": True})
    errs, again = _legs(plant, comp, [(target, 120)])
    assert comp.rec["lags"] == [{"t": comp.rec["lags"][0]["t"], "lag": 0, "gain": comp.rec["lags"][0]["gain"],
                                 "from": "pulse", "why": "watch", "steps": comp.rec["lags"][0]["steps"]}]
    assert comp.lag == 0 and again == 1 and errs[-1] < 0.002
    # (``cmd_watch`` alone) a pulse sent into the hold gets no answer and is sent once more
    comp = DL.DelayComp({"cmd_watch": True})
    comp.pos, comp.joints = [np.zeros(3)], [None]
    comp._measure("watch")
    for _ in range(120):
        a = comp.own_action() if comp.busy() else [0.0] * 6 + [-1.0]
        comp.after_step(a, {"robot0_eef_pos": np.zeros(3)})  # a hand that answers nothing
    assert [x["from"] for x in comp.rec["lags"]] == ["no answer", "no answer"] and not comp.busy()


def test_late_commands_are_measured_and_their_end_is_read_from_the_next_move():
    # a long reach, a turn-about at step 25 while the commands are 8 steps late, then (the delay over) a third place
    legs = [(np.array([0.60, 0.0, 0.0]), 25), (np.array([0.05, 0.10, 0.05]), 95), (np.array([0.20, -0.05, 0.0]), 70)]
    bare, _ = _legs(Late(late_at=10, until=70, d=8), DL.DelayComp(), legs[:2])
    plant, comp = Late(late_at=10, until=70, d=8), DL.DelayComp(NEW)
    errs, again = _legs(plant, comp, legs)
    w = comp.rec["watch"]
    # the commands turned about at step 25; four steps later none of the last four explains the hand's step, and
    # for five steps more the hand's step stays what it was
    assert len(w) == 1 and 32 <= w[0]["t"] <= 34 and w[0]["steady"] >= 9
    kinds = [(x["from"], x["lag"], x.get("why")) for x in comp.rec["lags"]]
    # the robot's first move after the wait was answered 8 steps late and that was taken as the lag (no pulse)
    assert kinds[0] == ("start", 8, "watch") and kinds[-1] == ("start", 0, None) and again == 1
    assert comp.rec["lags"][0]["t"] - w[0]["t"] < 40  # the pull, the rest, and the timed move
    assert comp.lag == 0 and errs[-1] < 0.003
    # ``cmd_watch`` alone: the pulse measures it
    plant, comp = Late(late_at=10, until=120, d=8), DL.DelayComp({"cmd_watch": True})
    _legs(plant, comp, legs[:2])
    assert [(x["from"], x["lag"], x.get("why")) for x in comp.rec["lags"]][0] == ("pulse", 8, "watch")


def test_the_pause_before_a_pulse_can_last_until_the_hand_is_at_rest():
    target = np.array([0.25, -0.10, -0.20])
    for opts, flush in (({}, 3), ({"flush_rest": True}, None)):
        plant, comp = Plant(), DL.DelayComp(opts)
        _drive(plant, comp, target, 150)
        pulse = comp.rec["lags"][1]
        assert pulse["from"] == "pulse" and pulse["lag"] == 14 and comp.lag == 14
        if flush is None:
            assert pulse["steps"] > 24  # the hand was still being read moving after the three steps
        else:
            assert pulse["steps"] == 24  # 3 steps of pause, 6 of pulse, the lag, and the step that shows the answer


def test_the_arm_checks_own_steps_are_not_watched():
    comp = DL.DelayComp(NEW)
    comp.start({"robot0_eef_pos": np.zeros(3)})
    p = np.zeros(3)

    def steps(n, cmd_x, foreign=False):
        nonlocal p
        for _ in range(n):
            p = p + np.array([0.01, 0.0, 0.0])  # the hand goes on one way whatever is sent
            a = [cmd_x, 0, 0, 0, 0, 0, -1.0]
            if foreign:
                comp.after_step(a, {"robot0_eef_pos": p.copy()}, foreign=True)
            else:
                comp.after_step(a, {"robot0_eef_pos": p.copy()})

    steps(6, 1.0)                  # as asked
    steps(20, -1.0, foreign=True)  # 20 steps against the commands, the arm check's own: not watched
    assert "watch" not in comp.rec and not comp.busy()
    steps(6, 1.0)                  # as asked again (the robot's own commands)
    steps(7, -1.0)                 # three steps until no recent command explains the hand's step, and four more
    assert "watch" not in comp.rec
    steps(1, -1.0)                 # the fifth such step in a row
    assert [w["t"] for w in comp.rec["watch"]] == [40] and comp.rec["watch"][0]["step_m"] == 0.01 and comp.busy()


def test_only_late_commands_ask_for_a_new_plan_and_opening_waits_for_the_commands_alone():
    target = np.array([0.25, -0.10, 0.05])
    for d_act, d_obs, again in ((7, 7, 1), (0, 10, 0)):  # S20: commands late too; D12: only the readings
        plant, comp = Plant(at=7, d_act=d_act, d_obs=d_obs), DL.DelayComp({"replan_late_only": True})
        _, n = _legs(plant, comp, [(target, 120)])
        assert comp.lag == d_act + d_obs and n == again
        plant, comp = Plant(at=7, d_act=d_act, d_obs=d_obs), DL.DelayComp()
        _, n = _legs(plant, comp, [(target, 120)])
        assert n == 1  # as before without the option
    for opts, lag, obs_lag, held_open in (({}, 15, 7, 15), ({"open_hold_cmd": True}, 15, 7, 7),
                                          ({"open_hold_cmd": True}, 11, 10, 0), ({"open_hold_cmd": True}, 8, 0, 8)):
        comp = DL.DelayComp(opts)
        comp.lag, comp.obs_lag = lag, obs_lag
        comp.sent([0, 0, 0, 0, 0, 0, -1.0])
        comp.sent([0, 0, 0, 0, 0, 0, 1.0])
        n_close = 0
        while comp.holding():
            comp.hold_action()
            n_close += 1
        comp.sent([0, 0, 0, 0, 0, 0, -1.0])
        n_open = 0
        while comp.holding():
            comp.hold_action()
            n_open += 1
        assert (n_close, n_open) == (lag, held_open)


def test_with_the_hand_closed_only_an_answer_at_once_is_believed():
    def run(grip, lag_in_use, answer_after):
        comp = DL.DelayComp({"start_lag": True})
        comp.start({"robot0_eef_pos": np.zeros(3)})
        comp.lag = lag_in_use
        comp._grip = grip
        p = np.zeros(3)
        quiet = max(8, lag_in_use + 3)
        for k in range(quiet):  # at rest, nothing asked for
            comp.after_step([0.0] * 6 + [grip], {"robot0_eef_pos": p.copy()})
        for k in range(answer_after + 4):  # then a steady command up; the hand follows ``answer_after`` steps late
            if k >= answer_after:
                p = p + np.array([0.0, 0.0, 0.004])
            comp.after_step([0.0, 0.0, 0.6, 0, 0, 0, grip], {"robot0_eef_pos": p.copy()})
        return comp

    comp = run(1.0, 0, 3)  # closed, no lag in use, the hand held back three steps (a drawer's handle): left alone
    assert comp.rec["starts"] == [{"t": comp.rec["starts"][0]["t"], "lag": 3, "was": 0, "closed": True}]
    assert comp.lag == 0 and not comp.busy()
    comp = run(-1.0, 0, 3)  # open, three steps late: a slow start, not a delay
    assert not comp.busy() and comp.lag == 0
    comp = run(-1.0, 0, 6)  # open, six steps late: a delay nothing had shown, so the pulse measures it
    assert comp.busy() and comp._m["why"] == "start"
    comp = run(1.0, 11, 0)  # closed, a lag in use, the hand answers at once: the delay is over
    assert comp.lag == 0 and comp.rec["lags"][-1]["from"] == "start"
    comp = run(1.0, 11, 14)  # closed, answering later than the lag in use: left alone
    assert comp.lag == 11 and comp.rec["lags"] == []
    comp = run(-1.0, 11, 14)  # open: the longer lag is taken
    assert comp.lag == 14


def test_the_release_option_is_off_by_default_and_reaches_the_agent():
    from rowant_lab.pro2 import robot as R

    ap, _pp, _fp = R.make_params(None)
    assert ap.lag_skip_release_fix is False
    ap, _pp, _fp = R.make_params({"agent": {"lag_skip_release_fix": True}})
    assert ap.lag_skip_release_fix is True
    for key in ("start_lag", "cmd_watch", "unlock", "flush_rest", "replan_late_only", "open_hold_cmd", "watch_lagged",
                "leap_trust", "no_answer_assume"):
        assert DL.DELAY[key] is False  # every option of 2026-11-16 is off unless asked for


def test_a_leap_can_end_the_lag_without_a_pulse():
    target = np.array([0.6, -0.3, 0.2])  # a long way: still moving when the delay ends
    plant, comp = Passing(until=60, at=12, d_act=0, d_obs=10), DL.DelayComp({"leap_trust": True})
    errs = _drive(plant, comp, target, 200)
    kinds = [(x["from"], x["lag"]) for x in comp.rec["lags"]]
    assert kinds == [("repeat", 11), ("pulse", 10), ("leap", 0)] and comp.lag == 0 and max(errs[150:]) < 0.004
    plant0, comp0 = Passing(until=60, at=12, d_act=0, d_obs=10), DL.DelayComp()
    errs0 = _drive(plant0, comp0, target, 200)
    first = lambda e: next(i for i, x in enumerate(e) if x < 0.004)  # noqa: E731
    assert first(errs) <= first(errs0) - 8  # there sooner by about the pulse it did not send


class Deaf(Plant):
    """S20's delays, and an arm that does not answer the pulse (its hand drifts where the delay's start left it)."""

    def step(self, a):
        a = np.asarray(a, float)
        if abs(a[2]) == 0.5 and not a[:2].any():  # the pulse's own commands go unanswered
            a = np.zeros(7)
        return super().step(a)


def test_a_pulse_without_an_answer_can_take_the_commands_to_be_as_late_as_the_readings():
    target = np.array([0.25, -0.10, 0.05])
    plant, comp = Deaf(), DL.DelayComp()
    errs = _drive(plant, comp, target, 200)
    assert [(x["from"], x["lag"]) for x in comp.rec["lags"]] == [("repeat", 8), ("no answer", 8)]
    assert comp.rec["lags"][1]["t"] == 46 and max(errs[150:]) > 0.02  # as last night: half the lag, still swinging
    plant, comp = Deaf(), DL.DelayComp({"no_answer_assume": True})
    obs = plant.obs()
    comp.start(obs)
    again, errs = 0, []
    for _ in range(200):
        if comp.busy():
            a = comp.own_action()
        else:
            seen = comp.predict(obs)
            a = follow(seen["robot0_eef_pos"], quat_xyzw_to_mat(seen["robot0_eef_quat"]), target, np.eye(3), -1.0)
            comp.sent(a)
        obs = plant.step(a)
        comp.after_step(a, obs)
        again += bool(comp.replan_due())
        errs.append(float(np.linalg.norm(plant.p - target)))
    kinds = [(x["from"], x["lag"]) for x in comp.rec["lags"]]
    assert kinds == [("repeat", 8), ("no answer", 8), ("assumed", 15)] and comp.rec["lags"][1]["t"] == 12 + 17 + 6 + 4
    assert comp.lag == 15 and again == 1 and max(errs[150:]) < 0.006  # one step more than the plant's 14: it still settles


class Trapped(Heavy):
    """An arm with an elbow. Out at ``x = wall`` the elbow is straight (it reads -0.07) and the hand cannot be drawn
    back by any command of position; a pull toward the robot together with a turn of the hand about +y bends the
    elbow a little each step and brings the hand in, and from 0.3 rad of bend the arm answers as usual. A held
    command (``hold_at``, ``n``) can carry the hand out there. The robot stands at (-0.63, 0)."""

    def __init__(self, wall=0.25, hold_at=10 ** 9, n=0, **kw):
        super().__init__(**kw)
        self.wall, self.hold_at, self.n, self.bend = wall, hold_at, n, 1.0

    def elbow(self):
        return -0.07 - self.bend

    def obs(self):
        o = super().obs()
        o["robot0_joint_pos"] = np.r_[o["robot0_joint_pos"][:3], self.elbow(), np.zeros(3)]
        return o

    def step(self, a):
        t = len(self.sent)
        self.sent.append(np.asarray(a, float))
        a_now = np.array(self.sent[self.hold_at - 1] if self.hold_at <= t < self.hold_at + self.n else self.sent[t], float)
        straight = self.bend < 0.3
        if straight and a_now[0] < 0.0:
            if a_now[0] <= -0.5 and a_now[4] >= 0.4:      # the pull with the turn: the elbow gives
                self.bend += 0.03
                self.v = np.array([self.kappa * OUT_POS * a_now[0], 0.0, 0.0])
            else:                                          # any other way in: the hand stands (it may slide sideways)
                a_now[0] = 0.0
                self.v = self.v * np.array([0.0, 1.0, 1.0])
                self.v = self.v + self.beta * (self.kappa * OUT_POS * a_now[:3] - self.v)
        else:
            self.v = self.v + self.beta * (self.kappa * OUT_POS * a_now[:3] - self.v)
        self.p = self.p + self.v
        if self.p[0] >= self.wall:                         # carried out to the edge: the elbow straightens
            self.p[0], self.v[0], self.bend = self.wall, min(self.v[0], 0.0), 0.0
        elif not straight:
            self.bend = max(0.3, 3.0 * (self.wall - self.p[0]))
        self.hist.append(self.p.copy())
        return self.obs()


def _run(plant, comp, target, steps, turn_back=0.0):
    obs = plant.obs()
    comp.start(obs)
    again = 0
    for _ in range(steps):
        if comp.busy():
            a = comp.own_action()
        else:
            seen = comp.predict(obs)
            a = follow(seen["robot0_eef_pos"], quat_xyzw_to_mat(seen["robot0_eef_quat"]), target, np.eye(3), -1.0)
            a[4] = turn_back  # the robot holding its hand upright out there asks for a turn the other way
            comp.sent(a)
        obs = plant.step(a)
        comp.after_step(a, obs)
        again += bool(comp.replan_due())
    return again


def test_an_arm_caught_with_its_elbow_straight_is_drawn_in_by_the_pull_with_the_turn():
    target = np.array([0.05, 0.10, 0.0])

    def caught():
        plant = Trapped(wall=0.25)
        plant.p, plant.bend = np.array([0.25, 0.0, 0.0]), 0.0  # out at the edge, the elbow at its limit
        plant.hist = [plant.p.copy()]
        return plant

    plant, comp = caught(), DL.DelayComp({})
    assert _run(plant, comp, target, 120, turn_back=-0.3) == 0 and plant.p[0] >= 0.25 - 1e-9  # left alone: caught to the end
    plant, comp = caught(), DL.DelayComp({"unlock": True})
    again = _run(plant, comp, target, 120, turn_back=-0.3)
    u = comp.rec["unlock"]
    assert len(u) == 1 and u[0]["why"] == "caught" and u[0]["t"] - u[0]["steps"] == 8  # eight steps asked in, not coming
    assert u[0]["elbow"][0] == -0.07 and u[0]["elbow"][1] <= -0.4 and u[0]["came_in_m"] > 0.08 and u[0]["steps"] <= 20
    assert again == 1 and float(np.linalg.norm(plant.p - target)) < 0.003
    # caught a second time later in the episode (no watch in use): unlocked again, and not a third time
    for k in range(2):
        plant.p, plant.bend, plant.v = np.array([0.25, 0.0, 0.0]), 0.0, np.zeros(3)
        plant.hist.append(plant.p.copy())
        obs = plant.obs()
        for _ in range(60):
            if comp.busy():
                a = comp.own_action()
            else:
                a = follow(obs["robot0_eef_pos"], np.eye(3), target, np.eye(3), -1.0)
                a[4] = -0.3
                comp.sent(a)
            obs = plant.step(a)
            comp.after_step(a, obs)
            comp.replan_due()
        assert len(comp.rec["unlock"]) == 2 and (plant.p[0] < 0.2) == (k == 0)
    # the manoeuvre itself: toward the robot, and the turn about the level axis square to that
    comp = DL.DelayComp({"unlock": True})
    comp.start({"robot0_eef_pos": np.array([0.07, 0.70, 1.0]), "robot0_joint_pos": [0, 0, 0, -0.07, 0, 0, 0]})
    a = np.array(comp._unlock_action())
    out = np.array([0.7, 0.7, 0.0]) / np.linalg.norm([0.7, 0.7])
    assert np.allclose(a[:3], -0.8 * out, atol=1e-6) and np.allclose(a[3:6], 0.6 * np.cross([0, 0, 1.0], out), atol=1e-6)
    # a hand that stands with its elbow bent (commands dropped for a while, D11) is not unlocked
    plant, comp = Trapped(wall=0.60), DL.DelayComp({"unlock": True})
    obs = plant.obs()
    comp.start(obs)
    for _ in range(40):
        a = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        comp.sent(a)
        comp.after_step(a, plant.obs())  # asked in, the readings do not change, the elbow reads -1.07
    assert "unlock" not in comp.rec and not comp.busy()


def test_after_a_held_command_that_left_the_elbow_straight_the_arm_is_unlocked_before_the_new_look():
    target = np.array([0.15, 0.0, 0.0])
    plant, comp = Trapped(wall=0.33, hold_at=8, n=20), DL.DelayComp({**NEW, "unlock": True})
    again = _run(plant, comp, target, 140)
    w, u = comp.rec["watch"][0], comp.rec["unlock"][0]
    assert u["why"] == "watch" and u["t"] - u["steps"] <= 28 + 6  # as soon as the hold was over and the hand stood
    assert again == 1 and float(np.linalg.norm(plant.p - target)) < 0.003 and comp.lag == 0
    assert w["flew_m"] > 0.1 and comp.rec["lags"] == []
    # with the unlock off the same episode ends with the hand out at the edge
    plant, comp = Trapped(wall=0.33, hold_at=8, n=20), DL.DelayComp(NEW)
    _run(plant, comp, target, 140)
    assert plant.p[0] >= 0.33 - 1e-9


class Wristed(Heavy):
    """An arm whose reach is its WRIST's: the wrist stands ``0.2 sin(pitch)`` further out than the grip point when
    the fingers are swung toward the robot by ``pitch``, and the elbow is straight when the wrist is out at ``W``
    (its bend, read as the fourth joint, grows as the square root of the room left). With the elbow straight no pull
    brings the hand in; the pull with the turn (fingers toward the robot) does, and swings the hand as it goes. A
    wrist that would pass ``W`` stops the hand. ``creep``: asked in with the elbow straight, the grip point creeps in
    as the hand tilts (rad a step at first, dying out toward 0.4 rad), the wrist where it is (as measured on the
    episodes played again: 9 cm in 30 steps, to 22 degrees)."""
    L, K, TURN = 0.2, 0.175, 0.18  # the hand's length; reach lost per rad^2 of bend; the share of a turn made in a step

    def __init__(self, W=0.31, pitch=0.31, creep=0.0, **kw):
        super().__init__(**kw)
        self.W, self.pitch = W, pitch  # (0.31 rad: how far the fingers pointed at the robot in the caught arm measured)
        self.creep = creep

    def wrist(self, p=None, pitch=None):
        return float((self.p if p is None else p)[0]) + self.L * np.sin(self.pitch if pitch is None else pitch)

    def elbow(self):
        return -0.07 - np.sqrt(max(0.0, self.W - self.wrist()) / self.K)

    def obs(self):
        o = super().obs()
        o["robot0_joint_pos"] = np.r_[o["robot0_joint_pos"][:3], self.elbow(), np.zeros(3)]
        o["robot0_eef_quat"] = np.asarray(DL.mat_to_quat_xyzw(DL._exp([0.0, self.pitch, 0.0])), float)
        return o

    def step(self, a):
        a = np.asarray(a, float)
        self.sent.append(a)
        straight = self.elbow() > -0.12
        dpitch = self.TURN * 0.5 * a[4]
        if straight and a[0] < 0.0 and not (a[0] <= -0.5 and a[4] >= 0.4):
            move = self.kappa * OUT_POS * a[:3] * np.array([0.0, 1.0, 1.0])  # the hand stands (it may slide sideways)
            dpitch = self.creep * max(0.0, 1.0 - self.pitch / 0.4)
            move[0] = self.L * (np.sin(self.pitch) - np.sin(self.pitch + dpitch))  # the wrist stays where it is
        else:
            move = self.kappa * OUT_POS * a[:3]
        p, pitch = self.p + move, self.pitch + dpitch
        if self.wrist(p, pitch) > self.W:  # the wrist cannot pass its reach: what would carry it there is not made
            if self.wrist(p, self.pitch) <= self.W:
                pitch = self.pitch
            elif dpitch != 0.0 and self.wrist(self.p, pitch) <= self.W:
                p = self.p + move * np.array([0.0, 1.0, 1.0])
            else:
                p, pitch = self.p + move, self.pitch  # (the hand goes as far out as its wrist's reach lets it)
            p[0] = min(p[0], self.W - self.L * np.sin(pitch))
        self.v, self.p, self.pitch = p - self.p, p, pitch
        self.hist.append(self.p.copy())
        return self.obs()


def test_after_the_unlock_the_hand_is_turned_back_before_the_robot_goes_on():
    def run(opts, steps=160):
        plant = Wristed(W=0.31)
        plant.p = np.array([0.31 - 0.2 * np.sin(0.31), 0.0, 0.0])  # caught: the wrist out where its reach ends
        plant.hist = [plant.p.copy()]
        target = plant.p + np.array([-0.09, 0.05, 0.0])  # 9 cm nearer the robot than where the hand is caught
        comp = DL.DelayComp(opts)
        obs = plant.obs()
        comp.start(obs)
        for _ in range(steps):
            if comp.busy():
                a = comp.own_action()
            else:  # the robot: to its target, the hand upright, as fast as it goes
                a = follow(obs["robot0_eef_pos"], quat_xyzw_to_mat(obs["robot0_eef_quat"]), target, np.eye(3), -1.0)
                comp.sent(a)
            obs = plant.step(a)
            comp.after_step(a, obs)
            comp.replan_due()
        return plant, comp, target

    plant, comp, target = run({})
    assert plant.p[0] > target[0] + 0.05 and "unlock" not in comp.rec  # left alone: caught to the end
    plant, comp, target = run({"unlock": True, "unlock_back": True})
    u = comp.rec["unlock"]
    assert len(u) == 1 and u[0]["elbow"][1] <= -0.4 and 8 <= u[0]["steps"] <= 20 and 0.08 < u[0]["came_in_m"] < 0.2
    # the pull left the fingers swung at the robot and the wrist nearly where it was. First the whole hand is drawn
    # in as it is (a full command, no turn) until the elbow is well bent; only then is the hand turned back to the
    # way it stood, with nothing else asked
    k = next(i for i, a in enumerate(plant.sent) if a[0] <= -0.5 and a[4] >= 0.4) + u[0]["steps"]
    d, b = u[0]["deep"], u[0]["back"]
    assert 3 <= d["steps"] <= 15 and d["elbow"] < u[0]["elbow"][1] - 0.5
    deep = np.array(plant.sent[k:k + d["steps"]])
    assert np.all(deep[:, 0] < -0.95) and np.allclose(deep[:, 3:6], 0.0) and np.all(deep[:, 6] == -1.0)
    assert 3 <= b["steps"] <= 20 and b["left_deg"] <= 10.0 and b["elbow"] <= d["elbow"]
    back = np.array(plant.sent[k + d["steps"]:k + d["steps"] + b["steps"]])
    assert np.allclose(back[:, :3], 0.0) and np.all(back[:, 4] < 0.0) and np.all(back[:, 6] == -1.0)
    assert float(np.linalg.norm(plant.p - target)) < 0.003 and abs(plant.pitch) < 0.05
    # without the option the hand is left as the pull leaves it (the record says so)
    plant, comp, target = run({"unlock": True})
    assert "back" not in comp.rec["unlock"][0] and "deep" not in comp.rec["unlock"][0]


def _stretched(opts, steps=200, creep=0.03):
    """A free, upright hand sent beyond its reach for 30 steps (the wrist stops where its reach ends, the elbow
    straight), then asked 12 cm back in: its grip point only creeps in, by tilting."""
    plant = Wristed(W=0.31, pitch=0.0, creep=creep)
    plant.p = np.array([0.22, 0.0, 0.0])
    plant.hist = [plant.p.copy()]
    comp = DL.DelayComp(opts)
    obs = plant.obs()
    comp.start(obs)
    pitches = []
    for t in range(steps):
        target = np.array([0.40, 0.0, 0.0]) if t < 30 else np.array([0.19, 0.04, 0.0])
        if comp.busy():
            a = comp.own_action()
        else:
            a = follow(obs["robot0_eef_pos"], quat_xyzw_to_mat(obs["robot0_eef_quat"]), target, np.eye(3), -1.0)
            comp.sent(a)
        obs = plant.step(a)
        comp.after_step(a, obs)
        comp.replan_due()
        pitches.append(float(plant.pitch))
    return plant, comp, np.array([0.19, 0.04, 0.0]), pitches


def test_an_arm_whose_grip_point_creeps_in_by_tilting_is_caught_by_its_elbow():
    plant, comp, target, _ = _stretched({})
    assert "unlock" not in comp.rec and plant.p[0] > target[0] + 0.03  # left alone: stuck short, the hand tilted
    # the morning's rule waits until the creep is under 1 mm a step; by the elbow reading the arm is caught at once
    slow = _stretched({"unlock": True})[1].rec["unlock"][0]
    fast = _stretched({"unlock": True, "caught_elbow": True})[1].rec["unlock"][0]
    assert fast["t"] - fast["steps"] == 30 + 8 and slow["t"] - slow["steps"] >= fast["t"] - fast["steps"] + 12


def test_the_turn_back_goes_to_the_way_the_hand_stood_with_its_elbow_free():
    plant, comp, target, pitches = _stretched({"unlock": True, "caught_elbow": True, "unlock_back": True})
    u = comp.rec["unlock"]
    assert len(u) == 1 and u[0]["elbow"][1] <= -0.4
    t0 = u[0]["t"] - u[0]["back"]["steps"] - u[0]["deep"]["steps"] - u[0]["steps"]
    # when the unlock began the hand had tilted already; the pull tilted it more; the turn back went past where the
    # unlock began, to the way the hand stood before the arm was stretched (upright)
    assert pitches[t0 - 1] > 0.12 and max(pitches) > pitches[t0 - 1] + 0.1
    assert pitches[u[0]["t"] - 1] < 0.18 and u[0]["back"]["left_deg"] <= 10.0 and u[0]["back"]["elbow"] < -0.8
    assert float(np.linalg.norm(plant.p - target)) < 0.003 and abs(plant.pitch) < 0.05


# ---- 2026-11-16 (night): the strict watch: what is not a hand flying against its commands


def _readings(opts, steps):
    """Feed a compensator the given readings: ``steps`` yields (the robot's command, the hand's place, its turn
    about y in rad or None for no orientation reading). Its own commands are sent while it is busy."""
    comp, first = DL.DelayComp(opts), True
    for a, p, th in steps:
        obs = {"robot0_eef_pos": np.asarray(p, float).copy()}
        if th is not None:
            obs["robot0_eef_quat"] = np.array([0.0, np.sin(th / 2), 0.0, np.cos(th / 2)])
        if first:
            comp.start(obs)
            first = False
            continue
        comp.after_step(comp.own_action() if comp.busy() else a, obs)
    return comp


def test_a_grip_point_carried_along_by_the_hands_own_turn_is_not_watched():
    # as on the plain drawer task: after a rise the robot asks for a full move down and across WITH a full turn of
    # the hand, and while the hand turns (0.06 rad a step) its grip point swings 5 mm a step another way
    def steps(turn):
        p, th = np.zeros(3), 0.0
        yield None, p, th
        for t in range(45):
            if 10 <= t < 30:
                th += turn
                p = p + np.array([0.005, 0.0, 0.0])
                yield [-0.58, 0.58, -0.58, 0.0, 1.0, 0.0, 1.0], p, th
            else:
                p = p + np.array([0.0, 0.0, 0.008])
                yield [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], p, th

    loose = _readings({"cmd_watch": True, "watch_rule": "loose"}, steps(0.06))
    assert loose.rec["watch"][0]["t"] == 15  # the rule of the morning: five steps of a moving hand off its commands
    strict = _readings({"cmd_watch": True}, steps(0.06))
    assert "watch" not in strict.rec and not strict.busy() and strict.rec["lags"] == []
    # the same readings from a hand that does not turn are a hand against its commands (a command is being held)
    held = _readings({"cmd_watch": True}, steps(0.0))
    assert "watch" not in held.rec  # ... but nothing it was sent before explains that step: no anchor, no firing


def test_a_held_command_is_seen_while_the_robot_asks_for_some_turn_but_not_for_a_full_one():
    # the hold itself: the hand keeps the step of the command before (straight, not turning) while the robot, seeing
    # the hand pass its target, asks for the way back and for a turn of the hand as well
    def steps(turn):
        p = np.zeros(3)
        yield None, p, 0.0
        for t in range(30):
            p = p + np.array([0.008, 0.0, 0.0])
            yield ([1.0, 0, 0, 0, 0, 0, -1.0] if t < 8 else [-0.7, 0.3, 0.0, 0.0, turn, 0.0, -1.0]), p, 0.0

    comp = _readings({"cmd_watch": True}, steps(0.3))  # (under 0.5 was asked before 15 of 16 firings in stored holds)
    assert [w["t"] for w in comp.rec["watch"]] == [16] and comp.busy()
    # while a full turn is asked nothing is watched: a turning hand's grip point goes its own way
    comp = _readings({"cmd_watch": True}, steps(1.0))
    assert "watch" not in comp.rec and not comp.busy()


def test_the_dry_watch_only_notes_where_it_would_have_fired():
    target = np.array([0.15, 0.0, 0.0])
    plant0, comp0 = Held(hold_at=8, n=20), DL.DelayComp()
    _legs(plant0, comp0, [(target, 120)])
    plant, comp = Held(hold_at=8, n=20), DL.DelayComp({"cmd_watch": True, "watch_dry": True})
    _, again = _legs(plant, comp, [(target, 120)])
    wet = DL.DelayComp({"cmd_watch": True})
    _legs(Held(hold_at=8, n=20), wet, [(target, 120)])
    assert [d["t"] for d in comp.rec["watch_dry"]][:1] == [wet.rec["watch"][0]["t"]]  # where the watch fires
    assert "watch" not in comp.rec and comp.rec["lags"] == [] and again == 0
    assert np.allclose(np.array(plant.sent), np.array(plant0.sent))  # and the robot's commands are untouched


def test_a_leap_of_the_pose_reading_stops_the_watch_for_some_steps():
    def fired(leap_at):
        def steps():
            p = np.zeros(3)
            yield None, p, None
            for t in range(40):
                p = p + np.array([0.01, 0.0, 0.0])  # the hand goes on one way whatever is sent (a held command)
                if t == leap_at:
                    p = p + np.array([0.0, 0.05, 0.0])  # a pose correction lands: the reading leaps 5 cm sideways
                yield [1.0 if t < 6 else -1.0, 0, 0, 0, 0, 0, -1.0], p, None
        return [w["t"] for w in _readings({"cmd_watch": True, "watch_dry": True}, steps()).rec.get("watch_dry", [])]

    assert fired(None) == [14]
    # a leap in the hold: the watch is off for eight steps, and then no command it was sent explains the hand's step
    assert fired(8) == []
    # a leap well before: the watch is back in time
    assert fired(0) == [14]


def _elbowed(comp, n, cmd, elbow, p, th=0.0):
    """``n`` steps of the robot's command ``cmd`` (the compensator's own while it is busy) with the elbow read as given."""
    sent = []
    for _ in range(n):
        a = comp.own_action() if comp.busy() else cmd
        sent.append([float(x) for x in a])
        comp.after_step(a, {"robot0_eef_pos": np.asarray(p, float), "robot0_joint_pos": np.r_[np.zeros(3), elbow, np.zeros(3)],
                            "robot0_eef_quat": np.array([0.0, np.sin(th / 2), 0.0, np.cos(th / 2)])})
    return np.array(sent)


def test_the_fold_and_the_turn_back_are_only_tried_on_an_arm_that_came_free():
    opts = {"unlock": True, "caught_elbow": True, "unlock_back": True}
    p, IN = [0.2, 0.0, 0.0], [-1.0, 0, 0, 0, 0, 0, -1.0]
    # the pull with the turn never frees the elbow: after its steps the robot looks again, nothing more is tried
    comp = DL.DelayComp(opts)
    comp.start({"robot0_eef_pos": np.asarray(p, float)})
    _elbowed(comp, 8, IN, -0.07, p)
    assert comp.busy()
    _elbowed(comp, 20, IN, -0.07, p)
    assert not comp.busy() and comp.replan_due()
    u = comp.rec["unlock"][0]
    assert u["steps"] == 20 and "deep" not in u and "back" not in u
    # it comes free, is folded, and reads straight again in the turn back: the turn stops there
    comp = DL.DelayComp(opts)
    comp.start({"robot0_eef_pos": np.asarray(p, float)})
    _elbowed(comp, 8, IN, -0.07, p)
    _elbowed(comp, 3, IN, -0.2, p, th=0.3)
    _elbowed(comp, 1, IN, -0.45, p, th=0.6)                 # free: the fold begins
    fold = _elbowed(comp, 4, IN, -0.9, p, th=0.6)
    assert np.all(fold[:, 0] < -0.95) and np.allclose(fold[:, 3:6], 0.0)
    _elbowed(comp, 1, IN, -1.25, p, th=0.6)                 # well bent: the turn back begins
    back = _elbowed(comp, 2, IN, -1.0, p, th=0.5)
    assert np.allclose(back[:, :3], 0.0) and np.all(back[:, 4] < 0.0) and comp.busy()
    _elbowed(comp, 1, IN, -0.07, p, th=0.4)                 # straight again
    assert not comp.busy() and comp.replan_due()
    u = comp.rec["unlock"][0]
    assert u["deep"] == {"steps": 5, "elbow": -1.25} and u["back"]["steps"] == 3 and u["back"]["left_deg"] > 10.0
