"""Every option off: does a robot package act as another one does? A kinematic stand-in for the simulator (no physics,
no renderer, no perception server): the hand goes where it is sent, the fingers stop on what is between them, a held
object moves with the hand and falls to where it stood when it is let go. Numbers only.

    P=<server python>
    PYTHONPATH=$PWD $P -P rowant_lab/pro2/probes/kin_same.py dump ROBOT_DIR OUT.json [--bundle]
    PYTHONPATH=$PWD $P -P rowant_lab/pro2/probes/kin_same.py compare BASE.json HEAD.json

``--bundle`` adds the scenarios once more with the candidate's options on (``long_dev.bundle``, this checkout's):
for two packages that both know those options, it says whether a change left the candidate as it was.

``dump`` loads the robot package at ``ROBOT_DIR`` (a checkout's ``rowant_lab/pro2/robot``, or a copy of it at another
commit: ``git archive COMMIT rowant_lab/pro2/robot | tar -x -C DIR``) under a name of its own and runs every
scenario below with the package's default parameters and with the unified configuration's agent options as they
stood at the base (the options that were on before the parallel streams began). It writes each episode's action
chain (a digest and the chain's length), how it finished and each attempt's outcome. ``compare`` says which episodes
differ. The library and the stand-in are this checkout's in both runs; only the robot package changes.

Scenarios (the policy library's Object cell: a can, a box, a basket):
* ``agent``: one pick and place, the agent on the stand-in's own looks: the reported pose off by a bias, a hold of
  another width, an empty close, a lift the look misreads, an object left on the table;
* ``tool``: the same with the look tool and a stand-in server in the loop: a target the scene moves before or as
  the hand arrives, slow fingers;
* ``plan``: ``plan_episode`` on a one-step and a two-object instruction (the step sequencer), each object where
  the first look puts it or moved once on the way.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from rowant_lab.libero_env import m2_library as LB  # noqa: E402
from rowant_lab.libero_env.m2_kin import mat_to_quat  # noqa: E402

# the unified configuration's agent options at the base commit (long_dev.long_params at 0496a52e)
BASE_AGENT = {"place_free_xy": True, "pull_stall_stop": True, "open_fast": True, "cross_descent_free": True,
              "relook_place": True, "rise_first": True, "rest_m": 0.008, "carry_watch": True}
BASE_TOP = {"sequence": True, "fallback_on_table": True, "stage_sweep": "stage", "fixture_no_flip": True}


def load(robot_dir: str) -> SimpleNamespace:
    d = Path(robot_dir).resolve()
    h = hashlib.sha256()
    for f in sorted(d.glob("*.py")):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    name = f"kinrobot_{h.hexdigest()[:16]}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, d / "__init__.py", submodule_search_locations=[str(d)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    m = {k: importlib.import_module(f"{name}.{k}") for k in ("agent", "look", "grasp", "episode")}
    return SimpleNamespace(pkg=sys.modules[name], sha=h.hexdigest(), **m)


def _expm(rv):
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(rv, float) / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _cam(camera, width, step):
    return {"camera": camera, "rgb": np.zeros((4, 4, 3), np.uint8), "depth_u16": np.ones((4, 4), np.uint16),
            "depth_scale_mm": 0.1, "K": np.eye(3), "T_world_from_camera": np.eye(4), "width": 4, "height": 4,
            "step": step}


class World:
    """Objects by percept id, each at its true place. The hand moves 0.35 of each commanded step; the fingers travel
    ``speed`` a step and stop on the object the hand is on (``width`` times that grasp's width) or on nothing; a held
    object moves with the hand and, let go, comes down to the height it stood at. ``jump``: {step: (id, move)}."""

    def __init__(self, R, lib, percepts, width=1.0, comes_up=True, speed=0.012, jump=None, start=(-0.1, 0.0, 0.25)):
        self.R_, self.lib = R, lib
        self.true = {p["id"]: np.asarray(p["bodies"]["main"]["pos"], float).copy() for p in percepts}
        self.rest_z = {k: float(v[2]) for k, v in self.true.items()}
        self.types = {p["id"]: p["type"] for p in percepts}
        self.width, self.comes_up, self.speed, self.jump = width, comes_up, speed, dict(jump or {})
        self.p, self.R = np.asarray(start, float).copy(), R.grasp.hand_matrix([0.0, 1.0, 0.0])
        self.gap, self.held, self.n = 2 * lib.jaw.L, None, 0
        self.frame_width = lambda: None  # set by the runner: the current attempt's grasp width

    def pobs(self):
        q = 0.5 * self.gap - self.lib.jaw.face_offset
        x, y, z, w = (lambda m: (m[1], m[2], m[3], m[0]))(mat_to_quat(self.R))
        return {"robot0_eef_pos": self.p.copy(), "robot0_eef_quat": np.array([x, y, z, w]),
                "robot0_gripper_qpos": np.array([q, -q]), "robot0_joint_pos": np.zeros(7)}

    def _on(self):
        """The object the hand is on: within 8 mm across and 6 cm in height of its origin (not the basket)."""
        best = None
        for pid, pos in self.true.items():
            if self.types[pid] == "basket":
                continue
            d = self.p - pos
            if float(np.linalg.norm(d[:2])) < 0.008 and abs(float(d[2])) < 0.06:
                best = pid
        return best

    def step(self, a):
        self.n += 1
        if self.n in self.jump:
            pid, mv = self.jump[self.n]
            if self.held != pid:
                self.true[pid] = self.true[pid] + np.asarray(mv, float)
        a = np.asarray(a, float)
        dp = a[:3] * 0.05 * 0.35
        self.p = self.p + dp
        self.R = _expm(a[3:6] * 0.5 * 0.35) @ self.R
        if self.held is not None and self.comes_up:
            self.true[self.held] = self.true[self.held] + dp
        if a[6] > 0:
            on = self.held or self._on()
            w = self.frame_width()
            stop = self.width * w if (on and w) else 0.0013
            self.gap = max(stop, self.gap - self.speed)
            if on and self.gap <= stop + 1e-9 and self.held is None:
                self.held = on
        else:
            self.gap = min(2 * self.lib.jaw.L, self.gap + self.speed)
            if self.held is not None:
                self.true[self.held][2] = self.rest_z[self.held]  # it comes down to the height it stood at
            self.held = None


class Looks:
    """The agent's look tool as the stand-in itself: every object where it is, the target ``bias`` off, and
    ``lift_err`` further while it is held 2 cm or more above where the hand closed."""

    def __init__(self, world, percepts, target="p0", bias=(0, 0, 0), lift_err=(0, 0, 0)):
        self.w, self.percepts, self.target = world, copy.deepcopy(percepts), target
        self.bias, self.lift_err, self.z_close, self.calls = np.asarray(bias, float), np.asarray(lift_err, float), None, 0

    def __call__(self):
        self.calls += 1
        w = self.w
        if w.held == self.target and self.z_close is None:
            self.z_close = float(w.p[2])
        if w.held != self.target:
            self.z_close = None
        for p in self.percepts:
            pos = w.true[p["id"]].copy()
            if p["id"] == self.target:
                pos = pos + self.bias
                if self.z_close is not None and w.p[2] > self.z_close + 0.02:
                    pos = pos + self.lift_err
            p["bodies"]["main"]["pos"] = [float(x) for x in pos]
        return w.lib.poses_from_percepts(self.percepts)


class Server:
    """The perception server over the stand-in's truth. A full look reports each object where it is, except one the
    hand hides (the hand under 13 cm high and within 12 cm of it across). A track finds an object within 5 cm of its
    prior, else refuses."""

    def __init__(self, world, percepts):
        self.w, self.percepts = world, copy.deepcopy(percepts)

    def call(self, msg):
        w = self.w
        if msg.get("mode") == "full":
            out = []
            for p in self.percepts:
                pos = w.true[p["id"]]
                if p["type"] != "basket" and w.p[2] < 0.13 and float(np.linalg.norm((w.p - pos)[:2])) < 0.12:
                    continue
                out.append({**p, "fit": {"n_agree": 500, "inlier": 0.9}, "source": "texture",
                            "bodies": {"main": {"pos": [float(x) for x in pos], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}})
            return {"percepts": out, "extra": {"kept": [], "rejected": [], "blobs": []}, "times": {}}
        tracks = {}
        by_gid = {p["gid"]: p["id"] for p in self.percepts}
        for pid, pr in msg["priors"].items():
            pos = w.true.get(by_gid.get(pr.get("gid"), pid), None)
            if pos is not None and float(np.linalg.norm(pos - np.asarray(pr["t_w"], float))) < 0.05:
                tracks[pid] = {"status": "tracked", "R_w": pr["R_w"], "t_w": [float(x) for x in pos], "n_px": 900}
            else:
                tracks[pid] = {"status": "predicted", "R_w": pr["R_w"], "t_w": pr["t_w"], "n_px": 0}
        return {"tracks": tracks, "times": {}}


def _percepts(lib, can=(0.0, 0.10), box=(0.16, 0.12), basket=(0.05, -0.20), with_box=False):
    rc = lib.records["alphabet_soup"]
    out = [{"id": "p0", "type": "alphabet_soup", "gid": "g0", "joints": {},
            "bodies": {"main": {"pos": [can[0], can[1], -float(rc["bottom"])], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}},
           {"id": "p1", "type": "basket", "gid": "g1", "joints": {},
            "bodies": {"main": {"pos": [basket[0], basket[1], 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}}]
    if with_box:
        rb = lib.records["cream_cheese"]
        out.append({"id": "p2", "type": "cream_cheese", "gid": "g2", "joints": {},
                    "bodies": {"main": {"pos": [box[0], box[1], -float(rb["bottom"])], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}})
    return out


def _attempts_of(agent):
    cur = getattr(agent, "cur", None)
    return (cur.attempts if cur is not None and hasattr(cur, "attempts") else None) or getattr(agent, "attempts", [])


def _drive(R, agent, world, obs_to=None, cap=400):
    acts = []
    world.frame_width = lambda: ((_attempts_of(agent)[-1].get("frame") or {}).get("width") if _attempts_of(agent) else None)
    for _ in range(cap):
        obs = world.pobs()
        if obs_to is not None:
            obs_to.obs = obs
        a = agent.act(obs)
        acts.append([round(float(x), 9) for x in a])
        world.step(a)
        if getattr(agent, "finished", False):
            break
    summ = agent.summary() if hasattr(agent, "summary") else {}
    return {"n": len(acts), "sha": hashlib.sha256(json.dumps(acts).encode()).hexdigest()[:16],
            "finish": summ.get("finish_reason"),
            "attempts": [a.get("outcome") for a in (summ.get("attempts") or [])],
            "end": {k: [round(float(x), 4) for x in v] for k, v in sorted(world.true.items())}}


def scenarios(R, bundle: bool = False) -> dict:
    lib = LB.M2Library("libero_object")
    spec = {"predicate": "in", "target": "p0", "place": "p1_contain_region", "place_kind": "region",
            "place_instance": "p1", "instruction": "pick the alphabet soup and place it in the basket"}
    out = {}
    cfgs = [("default", {}, {"sequence": True}), ("base", BASE_AGENT, BASE_TOP)]
    if bundle:
        from rowant_lab.pro2 import long_dev as LD

        cfgs.append(("bundle", {**BASE_AGENT, **LD.BUNDLE_AGENT}, {**BASE_TOP, **LD.BUNDLE_TOP}))
    for cfg, agent_opts, top in cfgs:
        ap = lambda: R.agent.AgentParams(**agent_opts)  # noqa: E731
        # -- the agent on the stand-in's own looks
        cases = [("plain", {}, {}), ("bias_x", {}, {"bias": (0.02, 0.0, 0.0)}), ("bias_z", {}, {"bias": (0.0, 0.0, 0.04)}),
                 ("bias_far", {}, {"bias": (0.05, 0.03, 0.0)}), ("wide", {"width": 1.3}, {}), ("thin", {"width": 0.7}, {}),
                 ("lift_err", {}, {"lift_err": (0.03, 0.0, 0.0)}), ("lift_err_z", {}, {"lift_err": (0.0, 0.0, -0.03)}),
                 ("left", {"comes_up": False}, {}), ("slow", {"speed": 0.004}, {}),
                 ("jump_early", {"jump": {8: ("p0", (0.06, 0.0, 0.0))}}, {}),
                 ("jump_late", {"jump": {60: ("p0", (0.0, -0.05, 0.0))}}, {})]
        for can in ((0.0, 0.10), (0.12, 0.16), (-0.08, 0.06)):
            for name, wk, lk in cases:
                pc = _percepts(lib, can=can)
                w = World(R, lib, pc, **wk)
                looks = Looks(w, pc, **lk)
                ag = R.agent.OracleAgent(lib, lib.scene_from_percepts(pc), spec, looks, 400, ap())
                out[f"{cfg}/agent/{name}/{can[0]:+.2f},{can[1]:+.2f}"] = _drive(R, ag, w)
        # -- the look tool and a stand-in server in the loop
        for can in ((0.0, 0.10), (0.12, 0.16)):
            for name, wk in (("plain", {}), ("jump_early", {"jump": {8: ("p0", (0.075, 0.0, 0.0))}}),
                             ("jump_mid", {"jump": {30: ("p0", (0.0, 0.075, 0.0))}}),
                             ("jump_far", {"jump": {12: ("p0", (0.15, 0.0, 0.0))}}), ("slow", {"speed": 0.004}),
                             ("wide", {"width": 1.3}), ("left", {"comes_up": False}),
                             ("basket_jump", {"jump": {70: ("p1", (0.075, 0.0, 0.0))}})):
                pc = _percepts(lib, can=can)
                w = World(R, lib, pc, **wk)
                tool = R.look.PerceptionTool(lib, Server(w, pc), _cam, "libero_object", "ep")
                tool.obs = w.pobs()
                tool.full_look()
                tool.target = "p0"
                ag = R.agent.OracleAgent(lib, lib.scene_from_percepts(tool.percepts), spec, tool, 400, ap())
                out[f"{cfg}/tool/{name}/{can[0]:+.2f},{can[1]:+.2f}"] = _drive(R, ag, w, obs_to=tool)
        # -- plan_episode: one step, and two objects (the step sequencer)
        params = {**top, "m1": {"agent": dict(agent_opts)}}
        for inst, with_box in (("pick up the alphabet soup and place it in the basket", False),
                               ("put both the alphabet soup and the cream cheese in the basket", True),
                               ("put the cream cheese in the basket", True)):
            for name, wk in (("plain", {}), ("jump_can", {"jump": {8: ("p0", (0.06, 0.0, 0.0))}}),
                             ("jump_box", {"jump": {40: ("p2", (0.0, 0.06, 0.0))}}), ("slow", {"speed": 0.004})):
                if name == "jump_box" and not with_box:
                    continue
                pc = _percepts(lib, with_box=with_box)
                w = World(R, lib, pc, **wk)
                tool = R.look.PerceptionTool(lib, Server(w, pc), _cam, "libero_object", "ep")
                tool.obs = w.pobs()
                tool.full_look()
                try:
                    planned = R.episode.plan_episode(inst, tool, lib, 520, params)
                    ag = planned.get("agent")
                    res = _drive(R, ag, w, obs_to=tool, cap=520) if ag is not None else {
                        "n": 0, "sha": "", "finish": "no agent", "grounding": planned.get("grounding")}
                except Exception as exc:  # the same crash on both sides is the same behaviour
                    res = {"n": 0, "sha": "", "finish": f"{type(exc).__name__}: {exc}"[:160]}
                out[f"{cfg}/plan/{name}/{inst[:28]}"] = res
    return out


def main(argv) -> int:
    if argv[0] == "dump":
        R = load(argv[1])
        res = scenarios(R, bundle="--bundle" in argv)
        Path(argv[2]).write_text(json.dumps({"robot_sha256": R.sha, "episodes": res}, indent=1))
        fin = {}
        for v in res.values():
            fin[str(v.get("finish"))] = fin.get(str(v.get("finish")), 0) + 1
        print(json.dumps({"robot_sha256": R.sha[:16], "episodes": len(res), "finish": fin}))
        return 0
    a, b = (json.loads(Path(x).read_text()) for x in argv[1:3])
    ea, eb = a["episodes"], b["episodes"]
    same = [k for k in sorted(set(ea) & set(eb)) if (ea[k]["sha"], ea[k]["n"], ea[k].get("finish")) ==
            (eb[k]["sha"], eb[k]["n"], eb[k].get("finish"))]
    diff = [k for k in sorted(set(ea) & set(eb)) if k not in same]
    print(json.dumps({"a": a["robot_sha256"][:16], "b": b["robot_sha256"][:16], "episodes": len(set(ea) & set(eb)),
                      "identical": len(same), "differ": len(diff), "only_one_side": len(set(ea) ^ set(eb))}))
    for k in diff:
        print("  differs:", k, {x: ea[k].get(x) for x in ("n", "finish", "attempts")},
              {x: eb[k].get(x) for x in ("n", "finish", "attempts")})
    return 0 if not diff and not (set(ea) ^ set(eb)) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
