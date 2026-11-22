"""2026-11-16 (two objects into one container): an object let go above its place is where it lands, and a box that
turned flat between the jaws while it was lifted is still held. Both off by default."""

from types import SimpleNamespace

import numpy as np

from rowant_lab.libero_env.m2_kin import mat_to_quat
from rowant_lab.libero_env.tests.test_close_look import _cam, _lib
from rowant_lab.pro2.robot import agent as AG
from rowant_lab.pro2.robot import look as LK


def _percept(pid, pos, gid="g1"):
    return {"id": pid, "type": "can", "gid": gid, "joints": {},
            "bodies": {"main": {"pos": list(pos), "quat_wxyz": mat_to_quat(np.eye(3))}}}


class Client:
    """Full look: the fresh percepts given (none: the object lies hidden in the container). Track: ``status``, at the
    prior moved by ``move``."""

    def __init__(self, fresh=(), status="predicted", move=(0.0, 0.0, 0.0)):
        self.calls, self.fresh, self.status, self.move = [], list(fresh), status, np.asarray(move, float)

    def call(self, msg):
        self.calls.append(msg)
        if msg.get("mode") == "full":
            return {"percepts": [dict(p) for p in self.fresh], "times": {}}
        out = {pid: {"status": self.status, "R_w": pr["R_w"], "t_w": (np.asarray(pr["t_w"], float) + self.move).tolist(),
                     "n_px": 400} for pid, pr in msg["priors"].items()}
        return {"tracks": out, "times": {"total_s": 0.1}}


def _released(client, drop_m):
    """A tool whose hand has just opened 6 cm above the held target, told (or not) how far the target will fall."""
    t = LK.PerceptionTool(_lib(), client, _cam, "libero_object", "ep")
    t.obs = {"robot0_joint_pos": np.zeros(7), "robot0_gripper_qpos": np.array([0.0, 0.0]),  # closed on the target
             "robot0_eef_pos": np.array([0.1, 0.2, 0.66]), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0])}
    t.percepts = [_percept("p0", (0.1, 0.2, 0.6)), _percept("p1", (0.4, 0.0, 0.5), gid="g2")]
    t.target = "p0"
    t.calls = 1  # the first full look has been taken
    t._set_held()
    t.obs = {**t.obs, "robot0_gripper_qpos": np.array([0.04, -0.04])}  # and the hand has opened
    t.drop_m = drop_m
    return t


def _z(t):
    return float(t.percepts[0]["bodies"]["main"]["pos"][2])


def test_off_the_object_stays_where_the_hand_let_go():
    cl = Client()
    t = _released(cl, 0.0)
    t()
    assert [m["mode"] for m in cl.calls] == ["full", "track"] and np.isclose(_z(t), 0.6) and t.n_drop == 0
    assert np.isclose(cl.calls[1]["priors"]["p0"]["t_w"][2], 0.6)


def test_let_go_above_its_place_the_object_is_where_it_lands():
    cl = Client()  # hidden behind the container's walls: no fresh percept, and the track is refused
    t = _released(cl, 0.12)
    poses = t()
    assert [m["mode"] for m in cl.calls] == ["full", "track"] and t.n_drop == 1 and t.drop_m == 0.0
    assert np.isclose(cl.calls[1]["priors"]["p0"]["t_w"][2], 0.48)  # the track is asked around the landing pose
    assert np.allclose(t.percepts[0]["bodies"]["main"]["pos"], [0.1, 0.2, 0.48])  # unseen, it keeps the landing pose
    assert np.allclose(poses["p0_main"]["pos"], [0.1, 0.2, 0.48])
    assert np.allclose(t.percepts[1]["bodies"]["main"]["pos"], [0.4, 0.0, 0.5])  # nothing else moves
    t()  # the next read starts from the landing pose too, and the fall is not applied twice
    assert np.isclose(_z(t), 0.48) and t.n_drop == 1


def test_a_landing_that_is_seen_is_measured():
    cl = Client(status="tracked", move=(0.01, 0.0, 0.015))  # it came to rest 1.8 cm from straight below
    t = _released(cl, 0.12)
    t()
    assert np.allclose(t.percepts[0]["bodies"]["main"]["pos"], [0.11, 0.2, 0.495])
    cl = Client(fresh=[_percept("q0", (0.3, 0.35, 0.47))])  # the full look finds it outside, where it bounced to
    t = _released(cl, 0.12)
    t()
    assert [m["mode"] for m in cl.calls] == ["full"] and np.allclose(t.percepts[0]["bodies"]["main"]["pos"], [0.3, 0.35, 0.47])


def test_the_agent_tells_the_tool_how_far_the_object_falls():
    place = {"low_rel": -0.09, "surface": 0.455}  # the held object's lowest point is 9 cm under the hand

    def run(z_hand, **ap):
        a = SimpleNamespace(ap=AG.AgentParams(**ap), oracle=SimpleNamespace(drop_m=0.0))
        att = {}
        return AG.OracleAgent._expect_drop(a, place, np.array([0.0, 0.3, z_hand]), att), a.oracle.drop_m, att

    drop, told, att = run(0.669, drop_predict=True)  # let go at a basket's rim: 12.4 cm of fall
    assert np.isclose(drop, 0.124) and np.isclose(told, 0.124) and att == {"drop_m": 0.124}
    assert run(0.565, drop_predict=True) == (0.0, 0.0, {})  # set down 2 cm above a plate: no fall to speak of
    assert run(0.669) == (0.0, 0.0, {})  # off
    a = SimpleNamespace(ap=AG.AgentParams(drop_predict=True), oracle=SimpleNamespace())  # a tool that cannot be told
    assert AG.OracleAgent._expect_drop(a, place, np.array([0.0, 0.3, 0.669]), {}) == 0.0
    a = SimpleNamespace(ap=AG.AgentParams(drop_predict=True), oracle=SimpleNamespace(drop_m=0.0))
    assert AG.OracleAgent._expect_drop(a, {"surface": 0.455}, np.array([0.0, 0.3, 0.669]), {}) == 0.0  # no height known


def test_the_lift_check_takes_a_box_that_turned_flat_between_the_jaws():
    def gap_ok(gap, width, **ap):
        a = SimpleNamespace(ap=AG.AgentParams(**ap))
        a._c1 = lambda g, w: AG.OracleAgent._c1(a, g, w)
        return AG.OracleAgent._c2_gap(a, gap, width)

    # long8d-a: the record says 52.0 mm across a diagonal, the close check reads 48.3, the lift check 44.9
    assert not gap_ok(0.0449, 0.05199)                              # off: 7.1 mm under, outside the 6 mm band
    assert gap_ok(0.0449, 0.05199, c2_band_lo_m=0.012)              # on: still held
    assert gap_ok(0.0467, 0.05199) and gap_ok(0.0467, 0.05199, c2_band_lo_m=0.012)
    assert not gap_ok(0.0390, 0.05199, c2_band_lo_m=0.012)          # 13 mm under: not this grasp any more
    assert not gap_ok(0.0011, 0.010, c2_band_lo_m=0.012)            # an empty close on a rim grasp: the rim band stands
    assert not gap_ok(0.0025, 0.0140, c2_band_lo_m=0.012)           # within 12 mm of a thin record, but an empty hand
    assert not gap_ok(0.0570, 0.05199, c2_band_lo_m=0.012)          # wider than the record allows
    assert gap_ok(0.0412, 0.0427, c2_band_lo_m=0.003) and not gap_ok(0.0363, 0.0427, c2_band_lo_m=0.003)  # never tighter than the close check


# ---- the blob fallback's order (``params["fallback_blob_clear"]``)
HALF = {"book": (0.0143, 0.0552, 0.0672), "mug": (0.0699, 0.053, 0.0516), "caddy": (0.0755, 0.2169, 0.0834)}


def _typed(pid, typ, pos):
    return {"id": pid, "type": typ, "gid": "g", "joints": {},
            "bodies": {"main": {"pos": list(pos), "quat_wxyz": mat_to_quat(np.eye(3))}}}


def _fallback_tool(blob_clear_first):
    lib = _lib()
    lib.export_types = {k: {**lib.export_types["can"], "kind": "object"} for k in HALF}
    lib.m1_type = lambda typ: {"bodies": {"main": {"primitives": [
        {"centre": [0.0, 0.0, HALF[typ][2]], "R": np.eye(3).tolist(), "half": list(HALF[typ])}]}}}
    lib.workcell = {}
    t = LK.PerceptionTool(lib, Client(), _cam, "libero_long_study", "ep")
    t.percepts = [_typed("p0", "mug", (-0.09, 0.019, 0.885)), _typed("p1", "caddy", (-0.408, -0.145, 0.888))]
    t.table_bound, t.blob_clear_first = True, blob_clear_first
    return t


def _blob(xy, half_xy, height, n_px):
    return {"xy": list(xy), "half_xy": list(half_xy), "yaw": 0.0, "z_top": 0.88 + height, "height": height, "n_px": n_px}


# the first look of the lost start (run e8cl-v3, the book task, colour line): no fit of the book but two on the caddy
BOOK, LEFTOVER = _blob((-0.273, 0.169), (0.133, 0.028), 0.135, 11761), _blob((-0.105, -0.025), (0.032, 0.017), 0.093, 448)
ON_CADDY = {**_typed("r0", "book", (-0.478, -0.099, 0.91)), "pos": [-0.478, -0.099, 0.91],
            "fit": {"n_agree": 657, "inlier": 1.0}, "source": "support"}


def _book_xy(t):
    got = [p for p in t.percepts if p["type"] == "book"]
    return (np.round(got[0]["bodies"]["main"]["pos"][:2], 3).tolist(), got[0]["source"]) if got else None


def test_a_blob_clear_of_the_reported_objects_comes_before_another_objects_leftover():
    extra = {"rejected": [ON_CADDY], "blobs": [BOOK, _blob((-0.089, -0.028), (0.023, 0.012), 0.06, 989), LEFTOVER]}
    off = _fallback_tool(False)
    off.extra = extra
    off.on_parsed({"target_type": "book", "place_type": None})
    assert _book_xy(off) == ([-0.105, -0.025], "blob")  # as before: the leftover beside the mug fits the size best
    on = _fallback_tool(True)
    on.extra = extra
    on.on_parsed({"target_type": "book", "place_type": None})
    assert _book_xy(on) == ([-0.273, 0.169], "blob")  # the blob that stands clear of the mug and the caddy


def test_with_no_clear_blob_the_order_goes_on_to_a_fit_inside_an_object_then_to_a_leftover():
    on = _fallback_tool(True)
    on.extra = {"rejected": [ON_CADDY], "blobs": [LEFTOVER]}
    on.on_parsed({"target_type": "book", "place_type": None})
    assert _book_xy(on) == ([-0.478, -0.099], "rejected:support")
    on = _fallback_tool(True)
    on.extra = {"rejected": [], "blobs": [LEFTOVER]}
    on.on_parsed({"target_type": "book", "place_type": None})
    assert _book_xy(on) == ([-0.105, -0.025], "blob")  # still better than no percept
    on = _fallback_tool(True)
    on.extra = {"rejected": [], "blobs": []}
    on.on_parsed({"target_type": "book", "place_type": None})
    assert _book_xy(on) is None


# ---- the steps the sequencer keeps back for a later step (``params["sequence_reserve"]``)
def test_the_sequencer_keeps_back_the_steps_asked_for(monkeypatch):
    from rowant_lab.pro2.robot import episode as EPI
    from rowant_lab.pro2.robot import sequence as SQ

    caps = []

    class Done:
        finished = False

        def act(self, _obs):
            self.finished = True
            return list(SQ.NOOP)

        def summary(self):
            return {"steps": 1, "finish_reason": "placed"}

    def plan_one(clause, tool, lib, cap, params, exclude, share=None):
        caps.append(cap)
        return {"agent": Done(), "grounding": {"target": "p0"}}

    monkeypatch.setattr(EPI, "_plan_one", plan_one)
    lib = SimpleNamespace(types={"basket": 1, "alphabet_soup": 1, "tomato_sauce": 1}, export_types={})
    inst = "put both the alphabet soup and the tomato sauce in the basket"
    EPI.plan_episode(inst, SimpleNamespace(), lib, 520, {"sequence": True})
    EPI.plan_episode(inst, SimpleNamespace(), lib, 520, {"sequence": True, "sequence_reserve": 135})
    assert caps == [520 - SQ.RESERVE_STEPS, 520 - 135]  # the first step's own budget, as before and with 135 kept back


# ---- the subset tool: base tasks and start states of a planned list
def test_a_subset_run_can_take_named_tasks_and_another_start_state(tmp_path, monkeypatch):
    import json

    from rowant_lab.pro2 import subset as SB

    eps = [{"suite": "libero_10", "task": t, "case": c, "state": 8, "seed": 150, "bddl": "x.bddl", "max_steps": 520}
           for t in (0, 1, 2, 7) for c in ("BASE", "D04")]
    (tmp_path / "eval" / "src").mkdir(parents=True)
    (tmp_path / "eval" / "src" / "episodes.json").write_text(json.dumps(eps))
    monkeypatch.setattr(SB, "EVAL", tmp_path / "eval")
    monkeypatch.setattr(SB, "ROOT", tmp_path / "dev")
    monkeypatch.setattr(SB, "build_params", lambda agent, top, colour="", *server: {"agent": agent, "top": top})
    assert SB.main(["dev", "--src", "src", "--name", "a", "--tasks", "libero_10:1,7", "--cases", "D04", "--states", "0",
                    "--seed", "142", "--agent", "drop_predict=true", "--set", "sequence_reserve=130"]) == 0
    got = json.loads((tmp_path / "dev" / "a" / "episodes.json").read_text())
    assert [(e["task"], e["case"], e["state"], e["seed"]) for e in got] == [(1, "D04", 0, 142), (7, "D04", 0, 142)]
    assert json.loads((tmp_path / "dev" / "a" / "params.json").read_text()) == {
        "agent": {"drop_predict": True}, "top": {"sequence_reserve": 130}}
    assert SB.main(["dev", "--src", "src", "--name", "b"]) == 0  # nothing asked: the list as it is
    assert json.loads((tmp_path / "dev" / "b" / "episodes.json").read_text()) == eps


# ---- of several objects asked to one place, one that lies there already is not the one to move
def test_a_step_does_not_choose_an_object_that_already_lies_at_the_shared_place(monkeypatch):
    from rowant_lab.pro2.robot import episode as EPI

    eye = mat_to_quat(np.eye(3))

    def pot(pid, pos, n_agree):
        return {"id": pid, "type": "moka_pot", "fit": {"n_agree": n_agree},
                "bodies": {"main": {"pos": list(pos), "quat_wxyz": eye}}}

    stove = {"id": "s0", "type": "flat_stove", "fit": {"n_agree": 900},
             "bodies": {"main": {"pos": [0.0, 0.3, 0.9], "quat_wxyz": eye}}}
    tdocs = {"moka_pot": {"kind": "object", "root_body": "main", "sites": {}},
             "flat_stove": {"kind": "fixture", "root_body": "main",
                            "sites": {"cook_region": {"body": "main", "pos": [0.0, 0.0, 0.02], "R": np.eye(3).tolist(),
                                                      "size": [0.075, 0.075, 0.003]}}}}

    def scene_from(percepts):
        inst, sites = {}, {}
        for p in percepts:
            inst[p["id"]] = {"type": p["type"], "kind": tdocs[p["type"]]["kind"], "bodies": {"main": f"{p['id']}_main"},
                             "root_body_name": f"{p['id']}_main"}
            for name, st in tdocs[p["type"]]["sites"].items():
                sites[f"{p['id']}_{name}"] = {"parent_instance": p["id"], "frame_body": f"{p['id']}_main",
                                              "pos": st["pos"], "R": st["R"], "size": st["size"]}
        return {"instances": inst, "sites": sites, "arena": [], "robot_mount": None}

    lib = SimpleNamespace(types={"moka_pot": 1, "flat_stove": 1}, export_types=tdocs, workcell={"arena": []},
                          scene_from_percepts=scene_from, quat_mat=lambda q: np.eye(3),
                          poses_from_percepts=lambda ps: {f"{p['id']}_{r}": b for p in ps for r, b in p["bodies"].items()})
    monkeypatch.setattr(EPI, "parse", lambda inst, types: {
        "status": "ok", "target_type": "moka_pot", "place_type": "flat_stove", "relation": None, "landmarks": [],
        "part": None, "predicate": "on", "place_part": None, "form": "put"})
    monkeypatch.setattr("rowant_lab.pro2.robot.goal.on_site", lambda lib, typ, part: "cook_region")
    monkeypatch.setattr(EPI, "make_params", lambda m1: (None, None, None))
    monkeypatch.setattr(EPI, "OracleAgent", lambda lib, scene, spec, tool, cap, ap, pp, fp: SimpleNamespace(spec=spec))

    def plan(percepts, share=(1, 2), **params):
        tool = SimpleNamespace(percepts=percepts, target=None)
        out = EPI._plan_one("put the moka pot on the stove", tool, lib, 300, params, (), list(share) if share else None)
        return (out["agent"].spec["target"] if out["agent"] is not None else None), out["grounding"].get("reason")

    on_stove, on_table = pot("p0", (0.05, 0.35, 0.93), 800), pot("p1", (-0.3, 0.0, 0.9), 500)
    assert plan([on_stove, on_table, stove]) == ("p0", None)                                # off: the strongest fit, as before
    assert plan([on_stove, on_table, stove], shared_skip_at_place=True) == ("p1", None)     # on: the pot still on the table
    assert plan([on_table, pot("p2", (-0.2, 0.1, 0.9), 900), stove], shared_skip_at_place=True) == ("p2", None)  # none there
    both = [on_stove, pot("p3", (-0.05, 0.25, 0.93), 400), stove]
    assert plan(both, shared_skip_at_place=True) == (None, "already at its place")           # nothing left to move
    assert plan([on_stove, on_table, stove], share=None, shared_skip_at_place=True) == ("p0", None)  # a step of its own


# ---- the placed check read again after the hand has backed off (``c3_recheck``)
def test_a_check_that_failed_only_on_rest_is_read_again_with_the_hand_backed_off():
    def run(zs, steps=3):
        a = SimpleNamespace(ap=AG.AgentParams(c3_recheck=True, c3_recheck_steps=steps, rest_m=0.008), t=150, n_idle=0)
        it = iter(zs)

        def idle():
            a.n_idle += 1
            a.t += 1
            return [0.0] * 6 + [-1.0]

        a._idle = idle
        a.oracle = lambda: next(it)

        def c3(reads, last):  # at rest when no two reads in a row differ by 8 mm; these reads are all inside
            mv = max(abs(reads[i + 1] - reads[i]) for i in range(len(reads) - 1))
            return mv < a.ap.rest_m, {"rest": mv < a.ap.rest_m, "inside": True, "max_move_m": round(mv, 5)}

        a._c3 = c3
        att = {"c3": {"t": 137, "rest": False, "inside": True, "max_move_m": 0.0093, "pass": False}}
        g, sent = AG.OracleAgent._c3_again(a, att, {"rest": False, "inside": True, "max_move_m": 0.0093}), 0
        try:
            while True:
                next(g)
                sent += 1
        except StopIteration as e:
            return e.value[0], len(e.value[1]), sent, att

    ok, n_reads, sent, att = run([0.930, 0.931, 0.9305])  # a pot standing on the stove: the jitter is gone with the hand away
    assert ok and n_reads == 3 and sent == 3 and att["c3"]["pass"] and att["c3"]["rest"] and att["t_end"] == 153
    assert att["c3"]["first"] == {"rest": False, "inside": True, "max_move_m": 0.0093} and att["c3"]["recheck_t"] == 153
    ok, _n, _s, att = run([0.930, 0.950, 0.990])  # still moving: the attempt has failed, as before
    assert not ok and not att["c3"]["pass"] and att["c3"]["max_move_m"] == 0.04
    assert run([0.93, 0.93], steps=1)[1:3] == (2, 2)  # never fewer than two reads


# ---- clauses that did not do their work are run once more when steps are left (``params["sequence_retry"]``)
class _Step:
    def __init__(self, n, reason):
        self.n, self.reason, self.i, self.finished = n, reason, 0, False

    def act(self, _obs):
        self.i += 1
        self.finished = self.i >= self.n
        return [0.1] * 6 + [1.0]

    def summary(self):
        return {"steps": self.i, "finish_reason": self.reason}


def test_clauses_that_did_not_do_their_work_are_run_once_more():
    from rowant_lab.pro2.robot import sequence as SQ

    def run(reasons, retry=True, cap=520, steps=("put the a in the c", "put the b in the c"), n=20):
        calls = []

        def planner(step, step_cap, exclude):
            clause = step["clause"] if isinstance(step, dict) else step
            calls.append((clause, step_cap, exclude, isinstance(step, dict) and step.get("retry_of")))
            why = reasons[len(calls) - 1]
            if why is None:
                return {"agent": None, "grounding": {"status": "fail", "reason": "no percept"}}
            return {"agent": _Step(n, why), "grounding": {"target": f"p{len(calls)}"}}

        ag = SQ.SequenceAgent(list(steps), planner, cap=cap, retry=retry)
        for _ in range(cap):
            if ag.finished:
                break
            ag.act({})
        return calls, ag

    calls, ag = run(["max_attempts", "placed", "placed"])  # the first object failed, the second went in: the first again
    assert [c[0] for c in calls] == ["put the a in the c", "put the b in the c", "put the a in the c"]
    assert calls[2][2] == ("p2",) and calls[2][3] == 0 and calls[2][1] == 520 - 40  # all the steps that are left
    assert ag.records[2]["retry_of"] == 0 and ag.records[2]["summary"]["finish_reason"] == "placed"
    assert ag.finished and ag.finish_reason == "steps done" and ag.placed == ["p2", "p3"]
    calls, ag = run(["max_attempts", "placed"], retry=False)  # off: as before
    assert len(calls) == 2 and ag.finished
    calls, ag = run(["placed", "placed"])  # nothing failed: nothing more to do
    assert len(calls) == 2 and ag.finished
    calls, ag = run([None, "placed", "no_plan"])  # a clause that could not be planned is planned again, once
    assert [c[3] for c in calls] == [False, False, 0] and ag.finished and len(ag.records) == 3
    calls, ag = run(["max_attempts", "no_budget", "placed", "placed"])  # both failed: both again, in their order
    assert [c[0] for c in calls][2:] == ["put the a in the c", "put the b in the c"] and [c[3] for c in calls][2:] == [0, 1]
    assert calls[2][1] == 520 - 40 - SQ.RESERVE_STEPS  # the second retry's steps are kept back
    calls, ag = run(["max_attempts", "placed"], cap=250, n=100)  # under 60 steps left: no second pass
    assert len(calls) == 2 and ag.finished
    planned = [{"kind": "stage", "need": 5, "clause": "stage"}, "open the drawer"]  # a step planned around another
    calls, ag = run(["no_budget", "opened"], steps=planned)                           # one is never run again
    assert len(calls) == 2
    calls, ag = run(["placed", "max_attempts", "opened"], steps=planned)  # the instruction's own clause is
    assert [c[0] for c in calls] == ["stage", "open the drawer", "open the drawer"] and calls[2][3] == 1


# ---- a percept built from a blob never stands lower than the table top under it
def test_a_blob_fallback_stands_on_the_table_top_not_inside_the_table():
    def root_z(on, blob, arena):
        t = _fallback_tool(False)
        t.lib.workcell = {"arena": arena}
        t.blob_on_table = on
        t.percepts = []
        t.extra = {"rejected": [], "blobs": [blob]}
        t.on_parsed({"target_type": "mug", "place_type": None})
        return round(float(t.percepts[0]["bodies"]["main"]["pos"][2]), 4)

    # the living-room cell: the table top is at 0.4368 and the server's support plane 1.1 cm under it; a flat object's
    # blob (top 0.4556, 3.03 cm over the plane). The model's lowest point is its origin.
    table = [{"type": "box", "centre": [-0.25, 0.05, 0.425], "half": [0.444, 0.672, 0.0118]},
             {"type": "box", "centre": [-0.25, 0.25, 0.204], "half": [0.412, 0.435, 0.209]}]
    flat = {"xy": [0.099, -0.22], "half_xy": [0.07, 0.053], "yaw": 0.0, "z_top": 0.4556, "height": 0.0303, "n_px": 10274}
    assert root_z(False, flat, table) == 0.4253  # as before: 1.15 cm inside the table
    assert root_z(True, flat, table) == 0.4368   # on its top
    high = {**flat, "z_top": 0.60, "height": 0.10}  # a blob whose base is over the table top keeps its own base
    assert root_z(True, high, table) == root_z(False, high, table) == 0.5
    assert root_z(True, flat, []) == 0.4253  # no table box in the workcell: nothing to hold on to
    off = {**flat, "xy": [0.9, -0.22]}  # beside the table: no top under it
    assert root_z(True, off, table) == 0.4253


# ---- a held object does not jump inside the hand (the agent's ``held_jump_max_m``)
def _holding(client, max_m):
    t = LK.PerceptionTool(_lib(), client, _cam, "libero_long_study", "ep")
    t.obs = {"robot0_joint_pos": np.zeros(7), "robot0_gripper_qpos": np.array([0.0, 0.0]),  # closed on the target
             "robot0_eef_pos": np.array([0.1, 0.2, 1.0]), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0])}
    t.percepts = [_percept("p0", (0.1, 0.2, 0.9))]
    t.target = "p0"
    t.calls = 1
    t.held_jump_max_m = max_m
    return t


def test_a_track_that_moves_a_held_object_inside_the_hand_is_not_believed():
    jump = (0.025, -0.025, -0.015)  # the book task, colour line, start state 0: 3.5 cm aside and 1.5 cm lower
    t = _holding(Client(status="tracked", move=jump), 0.0)
    t()
    assert np.allclose(t.percepts[0]["bodies"]["main"]["pos"], [0.125, 0.175, 0.885]) and t.n_held_kept == 0  # off: taken
    t = _holding(Client(status="tracked", move=jump), 0.015)
    t()
    assert np.allclose(t.percepts[0]["bodies"]["main"]["pos"], [0.1, 0.2, 0.9]) and t.n_held_kept == 1
    assert t.percepts[0]["source"] == "held" and t.looks[-1]["held_off_m"] == 0.0384
    t.obs = {**t.obs, "robot0_eef_pos": np.array([0.1, 0.2, 1.1])}  # the lift: the object comes with the hand
    t()
    assert np.allclose(t.percepts[0]["bodies"]["main"]["pos"], [0.1, 0.2, 1.0]) and t.n_held_kept == 2
    t = _holding(Client(status="tracked", move=(0.008, 0.0, 0.0)), 0.015)  # a small shift as the fingers closed: taken
    t()
    assert np.allclose(t.percepts[0]["bodies"]["main"]["pos"], [0.108, 0.2, 0.9]) and t.n_held_kept == 0
    assert t.percepts[0]["source"] == "track" and t.looks[-1]["held_off_m"] == 0.008
    t = _holding(Client(status="tracked", move=jump), 0.015)
    t.obs = {**t.obs, "robot0_gripper_qpos": np.array([0.04, -0.04])}  # an open hand holds nothing: every track is taken
    t()
    assert np.allclose(t.percepts[0]["bodies"]["main"]["pos"], [0.125, 0.175, 0.885]) and "held_off_m" not in t.looks[-1]


def test_the_agent_hands_the_limit_to_the_look_tool_for_each_grasp_but_a_rim_grasp():
    def limit(width, **ap):
        a = SimpleNamespace(ap=AG.AgentParams(**ap), oracle=SimpleNamespace(held_jump_max_m=0.007))
        AG.OracleAgent._held_limit(a, width)
        return a.oracle.held_jump_max_m

    assert limit(0.0369, held_jump_max_m=0.015) == 0.015  # the book, taken across its thickness
    assert limit(0.0100, held_jump_max_m=0.015) == 0.0    # a bowl by its rim swings in the jaws: no limit
    assert limit(0.0369) == 0.007                          # off: the tool is left as it is
    a = SimpleNamespace(ap=AG.AgentParams(held_jump_max_m=0.015), oracle=SimpleNamespace())  # a tool without the option
    AG.OracleAgent._held_limit(a, 0.0369)
    assert not hasattr(a.oracle, "held_jump_max_m")


def test_the_keep_rule_sign_test():
    from rowant_lab.pro2 import subset as SB

    assert SB.p_more_lost(0, 0) == 1.0
    assert abs(SB.p_more_lost(17, 7) - 0.9887) < 1e-3  # 11-15's colour pair: kept
    assert SB.p_more_lost(2, 9) < 0.05 and SB.p_more_lost(3, 9) >= 0.05  # 9 of 11 lost rejects; 9 of 12 does not


# ---- 2026-11-16 (night): the fall is taken from where the hand really has the object (``drop_rigid``)
def _lifted_then_released(drop_rigid):
    """A held target whose track at the lift was refused (the stored pose is then the prediction, 10 cm up), carried
    30 cm aside and let go 8 cm above where it will land."""
    cl = Client()  # every track refused, no fresh percept
    t = LK.PerceptionTool(_lib(), cl, _cam, "libero_object", "ep")
    obs = {"robot0_joint_pos": np.zeros(7), "robot0_gripper_qpos": np.array([0.0, 0.0]),
           "robot0_eef_pos": np.array([0.1, 0.2, 0.66]), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0])}
    t.obs = dict(obs)
    t.percepts = [_percept("p0", (0.1, 0.2, 0.6))]
    t.target = "p0"
    t.calls = 1
    t()                                                        # the close: the hold is anchored (the track is refused)
    t.obs = {**obs, "robot0_eef_pos": np.array([0.1, 0.2, 0.76])}
    t()                                                        # the lift, 10 cm: refused again; the prediction is stored
    assert np.allclose(t.percepts[0]["bodies"]["main"]["pos"], [0.1, 0.2, 0.7])
    t.obs = {**obs, "robot0_eef_pos": np.array([0.4, 0.2, 0.76]), "robot0_gripper_qpos": np.array([0.04, -0.04])}
    t.drop_m, t.drop_rigid = 0.08, drop_rigid                  # let go there, 8 cm above the landing place
    t()
    return t.percepts[0]["bodies"]["main"]["pos"]


def test_the_fall_starts_where_the_hand_has_the_object_not_a_lift_height_above_it():
    # as before: the hand's rise is counted twice (0.8 for 0.7), and the fall brings the pose back near the release point
    assert np.allclose(_lifted_then_released(False), [0.4, 0.2, 0.72])
    assert np.allclose(_lifted_then_released(True), [0.4, 0.2, 0.62])  # where it lands


def test_the_agent_tells_the_tool_to_take_the_fall_from_the_hand():
    place = {"low_rel": -0.09, "surface": 0.455}
    for on in (False, True):
        a = SimpleNamespace(ap=AG.AgentParams(drop_predict=True, drop_rigid=on),
                            oracle=SimpleNamespace(drop_m=0.0, drop_rigid=False))
        AG.OracleAgent._expect_drop(a, place, np.array([0.0, 0.3, 0.669]), {})
        assert a.oracle.drop_rigid is on and np.isclose(a.oracle.drop_m, 0.124)


# ---- 2026-11-16 (night): objects asked onto one surface keep their own parts of it (``share_keep_up_m``)
def _place_agent(z_at, predicate="on", share=(1, 2), blocked_at_goal=0.0, **ap):
    """A stand-in with what ``_plan_place`` reads: a 15 cm square region centred at (0, -0.20) on a wider top, the
    first object standing at (+0.05, -0.15), this one's planned part at (-0.05, -0.25); ``z_at(xy)`` is the height
    the hand can let go at over ``xy`` (the lowest release is 1.10)."""
    eye = mat_to_quat(np.eye(3))
    region = {"centre": np.array([0.0, -0.20, 0.93]), "half": np.array([0.075, 0.075, 0.003])}
    goal = np.array([-0.05, -0.25])
    a = SimpleNamespace(ap=AG.AgentParams(place_free_xy=True, **ap), target="t",
                        spec={"predicate": predicate, **({"place_share": list(share)} if share else {})},
                        scene={"instances": {"t": {"root_body_name": "t_main"}, "o": {"root_body_name": "o_main"},
                                             "stove": {"root_body_name": "stove_main"}}})
    a._place_goal = lambda poses: (goal.copy(), "stove", region)
    a._region_occupied = lambda poses, reg: True
    a._pose = lambda: (np.array([0.0, 0.0, 1.2]), np.eye(3))
    box = SimpleNamespace(aabb=lambda: (np.array([[-0.04, -0.04, 1.0]]), np.array([[0.04, 0.04, 1.15]])))
    a._held_boxes = lambda R_rel, t_rel, p, R: box
    a._instance_boxes = lambda poses, inst: SimpleNamespace(
        aabb=lambda: (np.array([[-0.15, -0.30, 0.90]]), np.array([[0.15, -0.10, 0.93]])))
    calls = []

    def place_at(poses, R_rel, t_rel, gap, xy, cont, reg):
        xy = np.asarray(xy, float)
        calls.append(xy.round(3).tolist())
        z = float(z_at(xy))
        if np.allclose(xy, goal):
            z += blocked_at_goal
        return {"hand_xy": xy, "z_release": z, "z_lo": 1.10, "z_top": 1.25, "surface": 0.93, "low_rel": -0.15,
                "region": reg, "container": cont, "blocked_by": "other" if z > 1.10 else None}

    a._place_at = place_at
    poses = {"t_main": {"pos": [0.0, 0.0, 1.05], "quat_wxyz": eye}, "o_main": {"pos": [0.05, -0.15, 0.97], "quat_wxyz": eye},
             "stove_main": {"pos": [0.0, -0.20, 0.90], "quat_wxyz": eye}}
    out = AG.OracleAgent._plan_place(a, poses, np.eye(3), np.zeros(3), 0.02)
    return np.round(out["hand_xy"], 3).tolist(), bool(out.get("free_xy")), calls


def test_the_second_object_on_a_shared_surface_keeps_its_planned_part():
    def tilted(xy):  # two-pots task, seeds 158 and 167: every point reads 2 cm up but one at the top's far side
        return 1.10 if (xy[0] > 0.10 and xy[1] < -0.25) else 1.12

    xy, free, _ = _place_agent(tilted)
    assert (xy, free) == ([0.11, -0.26], True)  # as before: the lowest release on the whole top, off the region
    xy, free, calls = _place_agent(tilted, share_keep_up_m=0.04)
    assert (xy, free) == ([-0.05, -0.25], False) and len(calls) == 1  # on: its planned part; no search
    xy, free, _ = _place_agent(tilted, share_keep_up_m=0.04, predicate="in")
    assert (xy, free) == ([0.11, -0.26], True)  # into a container: the lowest point, as before
    xy, free, _ = _place_agent(tilted, share_keep_up_m=0.04, share=None)
    assert (xy, free) == ([0.11, -0.26], True)  # a step of its own on an occupied region: as before


def test_when_the_planned_part_is_blocked_the_search_stays_on_the_region_and_away_from_the_other_object():
    def flat(xy):
        return 1.10 if abs(xy[0]) <= 0.05 and abs(xy[1] + 0.20) <= 0.05 else 1.115

    xy, free, calls = _place_agent(flat, share_keep_up_m=0.04, blocked_at_goal=0.08)
    assert free and all(abs(c[0]) <= 0.0501 and abs(c[1] + 0.20) <= 0.0501 for c in calls[1:])  # on the region
    # of the points within 2 cm of the lowest (the planned corner itself is 8 cm up), the one furthest from the
    # first object, which stands at (+0.05, -0.15): a neighbour of the corner
    assert xy in ([-0.05, -0.225], [-0.025, -0.25])
    xy, free, _ = _place_agent(flat, blocked_at_goal=0.08)
    assert xy == [0.0, -0.2] and free  # off: the lowest, nearest the middle (5 cm from the first object's line)


# ---- 2026-11-16 (night): a held object does not tip inside the hand either (``held_tilt_max_deg``)
class Turner(Client):
    """A track that turns the object ``deg`` about a level axis (``axis`` "x": it tips) or the vertical ("z")."""

    def __init__(self, deg, move=(0.0, 0.0, 0.0), axis="x"):
        super().__init__(status="tracked", move=move)
        a = np.radians(deg)
        c, s_ = np.cos(a), np.sin(a)
        self.Rt = (np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]]) if axis == "x"
                   else np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]))

    def call(self, msg):
        rep = super().call(msg)
        for pid, tr in (rep.get("tracks") or {}).items():
            tr["R_w"] = (self.Rt @ np.asarray(tr["R_w"], float)).tolist()
        return rep


def test_a_track_that_tips_a_held_object_inside_the_hand_is_not_believed():
    def run(deg, limit, gripper=(0.0, 0.0), axis="x"):
        t = _holding(Turner(deg, move=(0.002, 0.0, 0.001), axis=axis), 0.0)
        t.held_tilt_max_deg = limit
        t.obs = {**t.obs, "robot0_gripper_qpos": np.asarray(gripper, float)}
        t()
        q = t.percepts[0]["bodies"]["main"]["quat_wxyz"]
        turned = not np.allclose(np.abs(q), [1.0, 0.0, 0.0, 0.0], atol=1e-6)
        return turned, t.percepts[0]["source"], t.n_held_kept, t.looks[-1]

    turned, src, kept, look = run(16.0, 0.0)  # off: taken, as before; the turn and the tip are in the record
    assert turned and src == "track" and kept == 0 and "held_kept" not in look
    assert look["held_turn_deg"] == 16.0 and look["held_tilt_deg"] == 16.0
    turned, src, kept, look = run(16.0, 10.0)  # the second pot of seeds 156, 158 and 167: not believed
    assert not turned and src == "held" and kept == 1 and look["held_kept"] is True
    turned, src, kept, look = run(5.0, 10.0)   # a small tip as the fingers closed: taken
    assert turned and src == "track" and kept == 0 and look["held_tilt_deg"] == 5.0
    turned, src, kept, look = run(25.0, 10.0, axis="z")  # a box squaring up between the jaws turns, it does not tip
    assert turned and src == "track" and kept == 0 and look["held_turn_deg"] == 25.0 and look["held_tilt_deg"] == 0.0
    turned, src, kept, look = run(16.0, 10.0, gripper=(0.04, -0.04))  # an open hand holds nothing: taken
    assert turned and src == "track" and "held_tilt_deg" not in look


def test_a_track_not_believed_for_its_tip_is_not_a_measurement():
    t = _holding(Turner(16.0), 0.015)
    t.held_tilt_max_deg = 10.0
    t()
    assert not t.target_measured(0)  # (``c3_measured``: the stored pose is still the prediction)
    t = _holding(Turner(5.0), 0.015)
    t.held_tilt_max_deg = 10.0
    t()
    assert t.target_measured(0)


def test_the_agent_hands_the_tilt_limit_to_the_look_tool_but_not_for_a_rim_grasp():
    def limit(width, **ap):
        a = SimpleNamespace(ap=AG.AgentParams(**ap), oracle=SimpleNamespace(held_jump_max_m=0.0, held_tilt_max_deg=0.0))
        AG.OracleAgent._held_limit(a, width)
        return a.oracle.held_tilt_max_deg

    assert limit(0.0128, held_tilt_max_deg=10.0) == 10.0 and limit(0.0100, held_tilt_max_deg=10.0) == 0.0
    assert limit(0.0128) == 0.0


# ---- 2026-11-16 (night): our own instruction in a long-suite TRAINING scene (``practice long``)
def test_a_long_practice_run_is_a_training_tasks_episodes_with_our_instruction(tmp_path, monkeypatch):
    import json

    import pytest

    from rowant_lab.pro2 import practice as PR

    eps = [{"suite": "libero_10", "task": t, "case": c, "case_kind": "x", "state": 8, "seed": 150,
            "bddl": f"/x/{c}__t{t}.bddl", "max_steps": 520, "split": "seen"}
           for t in (2, 3, 5, 8) for c in ("BASE", "D04", "S12")]
    src = tmp_path / "list.json"
    src.write_text(json.dumps(eps))
    monkeypatch.setattr(PR, "LONG_SOURCE", src)
    monkeypatch.setattr(PR, "DEVRUNS", tmp_path / "dev")
    monkeypatch.setattr(PR, "EVAL", tmp_path / "eval")
    got = PR.long_episodes("pot-stove", 0, 142, None)
    assert [(e["task"], e["case"], e["state"], e["seed"]) for e in got] == [(2, "BASE", 0, 142), (2, "D04", 0, 142),
                                                                           (2, "S12", 0, 142)]
    assert all(e["practice"] == {"name": "pot-stove", "instruction": "put the moka pot on the stove",
                                 "goal": [["on", "moka_pot_1", "flat_stove_1_cook_region"]]} for e in got)
    assert [e["case"] for e in PR.long_episodes("bowl-drawer", 8, 150, {"BASE"})] == ["BASE"]
    monkeypatch.setitem(PR.LONG, "two-pots", (8, "put both moka pots on the stove", []))  # a test task's scene: never
    with pytest.raises(SystemExit):
        PR.long_episodes("two-pots", 8, 150, None)
    monkeypatch.setitem(PR.LONG, "book", (5, "put the book in the caddy", []))
    with pytest.raises(SystemExit):
        PR.long_episodes("book", 8, 150, None)
    src.write_text(json.dumps(eps + [{**eps[0], "case": "S06"}]))  # a held case in the list: never
    with pytest.raises(SystemExit):
        PR.long_episodes("pot-stove", 8, 150, None)
    src.write_text(json.dumps(eps))
    monkeypatch.setattr("rowant_lab.pro2.long_dev.long_params", lambda: {"m1": {"agent": {"a": 1}}, "sequence": True})
    monkeypatch.setattr("rowant_lab.pro2.long_dev.bundle", lambda p: {**p, "bundled": True})
    assert PR.main(["long", "--names", "pot-stove", "--state", "0", "--seed", "142", "--cases", "all", "--bundle",
                    "--tag", "v6", "--agent", "share_keep_up_m=0.04", "--set", "look_log=true"]) == 0
    d = tmp_path / "dev" / "lp0-pot-stove-v6"
    assert len(json.loads((d / "episodes.json").read_text())) == 3
    assert json.loads((d / "params.json").read_text()) == {
        "m1": {"agent": {"a": 1, "share_keep_up_m": 0.04}}, "sequence": True, "bundled": True, "look_log": True}
    (tmp_path / "eval" / "lp0-pot-stove-v6").mkdir(parents=True)  # a name with rows under colour/eval is refused
    with pytest.raises(SystemExit):
        PR.main(["long", "--names", "pot-stove", "--state", "0", "--seed", "142", "--bundle", "--tag", "v6"])


def test_the_practice_report_pairs_two_runs_by_episode(tmp_path, monkeypatch, capsys):
    import json

    from rowant_lab.pro2 import practice as PR

    monkeypatch.setattr(PR, "ROOT", tmp_path / "practice")
    monkeypatch.setattr(PR, "DEVRUNS", tmp_path / "dev")
    monkeypatch.setattr(PR, "EVAL", tmp_path / "eval")

    def rows(run, oks):
        d = tmp_path / "eval" / run / "w00"
        d.mkdir(parents=True)
        d.joinpath("rows.jsonl").write_text("".join(json.dumps({
            "suite": "libero_10", "task": 2, "case": c, "state": 0, "seed": 142, "success": False, "steps": 100,
            "practice": {"success": ok, "at": 90 if ok else None}}) + "\n" for c, ok in oks.items()))

    rows("a", {"BASE": True, "D04": False, "S12": False})
    rows("b", {"BASE": True, "D04": True, "S12": False})
    assert PR.main(["report", "--runs", "a,b", "--brief"]) == 0
    out = capsys.readouterr().out
    assert "a: practice goal met in 1 of 3" in out and "b: practice goal met in 2 of 3" in out
    assert "b against a: 3 episodes in both, 1 -> 2; won 1 ['D04'], lost 0 []" in out


# ---- 2026-11-16 (night): the two-pots task in a kinematic stand-in, the second pot read upside down in the hand
def _kitchen():
    from pathlib import Path

    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.pro2 import long_paths as LP
    from rowant_lab.pro2 import machine as MC

    if not Path(MC.LONG_LIB, "build").exists():
        return None
    _ob, policy_lib, m1_lib = LB.lfb_dirs(MC.LONG_LIB)
    return LB.M2Library("libero_long_kitchen", lib_dir=policy_lib, m1_library=m1_lib,
                        extra_suites=LP.extra_suites(["libero_long_kitchen"]))


def _two_pots(lib, flipped: bool, **agent):
    """"put both moka pots on the stove" with the candidate's bundle in the stand-in of ``probes/kin_same.py`` (the
    stove and the pots where the scored start's first look has them). ``flipped``: every track of the second pot
    while the hand holds it reads it upside down, at the right place. Returns the two releases' records."""
    from rowant_lab.pro2 import long_dev as LD
    from rowant_lab.pro2 import machine as MC
    from rowant_lab.pro2.probes import kin_same as KS
    from rowant_lab.pro2.robot import episode as EPI
    from rowant_lab.pro2.robot import grasp as GR

    R = SimpleNamespace(grasp=GR, look=LK, agent=AG, episode=EPI)
    ns = SimpleNamespace(lib=lib)

    def bodies(typ, pos):
        return LK.PerceptionTool._build(ns, typ, np.eye(3), np.asarray(pos, float))

    pc = [{"id": "p0", "type": "flat_stove", "gid": "g0", "joints": {}, "bodies": bodies("flat_stove", (-0.194, -0.208, 0.906))},
          {"id": "p1", "type": "moka_pot", "gid": "g1", "joints": {}, "bodies": bodies("moka_pot", (-0.054, 0.258, 0.966))},
          {"id": "p2", "type": "moka_pot", "gid": "g2", "joints": {}, "bodies": bodies("moka_pot", (0.041, 0.043, 0.966))}]

    class Pots(KS.World):
        agent = None

        def _on(self):  # the hand is on the current step's target when it stands at that attempt's grasp point
            ats = KS._attempts_of(self.agent)
            fr = ats[-1].get("frame") if ats else None
            tgt = getattr(getattr(self.agent, "cur", None), "target", None)
            if fr is None or tgt not in self.true:
                return None
            return tgt if float(np.linalg.norm(self.p - np.asarray(fr["centre"], float))) < 0.006 else None

    class Srv(KS.Server):
        def call(self, msg):
            rep = super().call(msg)
            if msg.get("mode") == "full":
                for q in rep["percepts"]:
                    q["bodies"] = bodies(q["type"], self.w.true[q["id"]])
            elif flipped:
                for pid, tr in rep["tracks"].items():
                    if tr["status"] == "tracked" and pid == "p2" and self.w.held == "p2":
                        tr["R_w"] = np.diag([1.0, -1.0, -1.0]).tolist()  # half a turn about a level axis
            return rep

    w = Pots(R, lib, pc, start=(-0.2, 0.0, 1.17))
    tool = LK.PerceptionTool(lib, Srv(w, pc), KS._cam, "libero_long_kitchen", "ep")
    tool.obs = w.pobs()
    tool.full_look()
    p = LD.bundle(LD.colour_long_params(MC.REGISTERED["net"]))
    p["m1"]["agent"] = {**p["m1"]["agent"], **agent}
    ag = EPI.plan_episode("put both moka pots on the stove", tool, lib, 520, p)["agent"]
    w.agent = ag
    w.frame_width = lambda: ((KS._attempts_of(ag)[-1].get("frame") or {}).get("width") if KS._attempts_of(ag) else None)
    for _ in range(520):
        tool.obs = w.pobs()
        w.step(ag.act(tool.obs))
        if ag.finished:
            break
    rel = [a for a in ag.summary()["attempts"] if a.get("place")]
    return [(a["step"], a["outcome"], round(a["place"]["z_release"] - a["place"]["surface"], 3),
             [round(x, 2) for x in a["place"]["xy"]], bool(a["place"].get("free_xy"))) for a in rel], tool


def test_a_pot_read_upside_down_in_the_hand_is_let_go_two_centimetres_too_high_unless_the_tip_is_refused():
    import pytest

    lib = _kitchen()
    if lib is None:
        pytest.skip("the long suite's library is not on this machine")
    plain, _ = _two_pots(lib, False)
    assert [r[:2] for r in plain] == [(0, "placed"), (1, "placed")]
    up1, up2 = plain[0][2], plain[1][2]
    flipped, tool = _two_pots(lib, True, held_tilt_max_deg=0.0)
    # the candidate as it was before v7 (no tilt limit): the flip is taken (the pot is 2 cm longer under its origin
    # than over it), and the second pot is let go 2 cm higher; what the scored start's seeds 156, 158 and 167 showed
    # (19.1 cm for 17.1)
    assert flipped[0][2] == up1 and np.isclose(flipped[1][2] - up2, 0.02, atol=0.002)
    assert any(l.get("held_tilt_deg", 0) > 170 for l in tool.looks) and tool.n_held_kept == 0
    kept, tool = _two_pots(lib, True)  # the candidate (v7 holds ``held_tilt_max_deg`` 10)
    assert kept == plain and tool.n_held_kept >= 1  # the tip is not believed: as if the track had read it right
    # the shared-surface rule (in v7 too): the second pot at its planned corner, no search; the same places without
    assert plain[1][4] is False
    off, _ = _two_pots(lib, False, share_keep_up_m=0.0)
    assert [r[:4] for r in off] == [r[:4] for r in plain]
