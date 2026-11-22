"""Our robot on one LIBERO-Pro2.0 training episode, in process: the DEPTH LINE (2026-11-04; a declared deviation).

Runs in venv-pro2dev on a pod. Its Python 3.11.11 and numpy 1.26.4 are venv-libero's (the bridge's
pins), MuJoCo 3.2.3 and robosuite 1.4.0 are Pro2.0's reference versions, ``libero`` is Pro2.0's simulator, and the
service source is on the path. The environment is the service's own ``LiberoEnvironmentAdapter``, so create, reset,
step, success and the step cap are exactly a scored run's, without the HTTP layer:
* reset = seed, perturbation seed, start state, 10 settling steps, rollout start, controller mode;
* success = ``check_success`` at every step; the episode ends at success or at the case's step cap.

What the robot gets:
* proprioception from the episode's observation (``robot0_eef_quat_xyzw`` renamed back to ``robot0_eef_quat``), so
  the robot-state, calibration and timing cases act on it as they would in a scored run;
* the instruction from the episode, so the language cases act on it;
* DEPTH LINE: its looks come from M2's camera tool reading the live simulator (agentview 1024 px RGB-D, K and the
  camera pose), never from the episode's 256 px images. So every case that acts on the images or the camera is
  bypassed by this line: ``cases.BYPASSED``. They are reported apart.

The robot is M2 as frozen (``params`` {} = M2's defaults, M2's policy library), as at the M2 gate, or with
``params["robot_dir"]`` the self-writing loop's robot package (``robot``). After the robot
finishes it sends no-ops to the cap, as a scored run would. The row keeps the action chain for the parity replay
(``replay.py``) and the perturbation plan's fingerprints. Nothing here is passed back to the robot or a diagnoser.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np

from rowant_lab.libero_env import m2_audit as AU
from rowant_lab.libero_env import m2_bridge as BR
from rowant_lab.libero_env import m2_camera as CAM
from rowant_lab.libero_env import m2_episode as EP
from rowant_lab.libero_env import m2_library as LB

TOKEN_SALT = "rowant-pro2-dev-token-v1"
CAMERAS = ["agentview", "robot0_eye_in_hand"]
PROPRIO = {"robot0_joint_pos": "robot0_joint_pos", "robot0_joint_vel": "robot0_joint_vel",
           "robot0_eef_pos": "robot0_eef_pos", "robot0_eef_quat_xyzw": "robot0_eef_quat",
           "robot0_gripper_qpos": "robot0_gripper_qpos"}
from rowant_lab.pro2.cases import BYPASSED  # noqa: E402  (cases acting only on images or the camera)
from rowant_lab.pro2 import goal_paths as GP  # noqa: E402  (the Goal suite's types and Panda, not in M2's m2_paths)
from rowant_lab.pro2 import long_paths as LP  # noqa: E402  (the long-task suite's tables; Goal's entry included)
NOOP = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


class TracedClient(BR.Client):
    """The bridge's client, keeping the server's own traceback when a request fails (2026-11-18, record only). The
    bridge's client raises the message alone; an index error in six rows of 11-18 could not be placed from it. The
    bridge module is M2's and stays as it is."""

    def call(self, msg: dict) -> dict:
        self.bytes_out += BR.send(self.sock, msg)
        rep, n = BR.recv(self.sock)
        self.bytes_in += n
        if rep.get("error"):
            err = BR.RemoteError(rep["error"])
            err.server_trace = rep.get("traceback")
            raise err
        return rep


def server_trace(exc) -> list | None:
    """Where in the server a failed request raised: the last frames of the traceback it sent back, as ``file.py:line
    function`` (file names only; ``TracedClient`` keeps the text on the error). None when there is none."""
    import re

    text = getattr(exc, "server_trace", None)
    if not text:
        return None
    return [f"{Path(f).name}:{ln} {name}" for f, ln, name in re.findall(r'File "([^"]+)", line (\d+), in (\S+)', text)][-8:]


def proprio(obs: dict) -> dict:
    return {ours: [float(x) for x in obs[k]] for k, ours in PROPRIO.items() if k in obs}


def token(suite: str, task: int, case: str, state: int, seed: int) -> str:
    """Opaque to the robot and the server, deterministic for a re-run (as M2's ``episode_token``)."""
    return hashlib.sha256(f"{TOKEN_SALT}|{suite}|{int(task)}|{case}|{int(state)}|{int(seed)}".encode()).hexdigest()[:16]


def env_spec(ep: dict, camera_px: int) -> dict:
    spec = {"suite": ep["suite"], "task_id": int(ep["task"]), "variant": "base" if ep["case"] == "BASE" else ep["case"],
            "_bddl_path": ep["bddl"], "camera_names": CAMERAS, "camera_height": camera_px, "camera_width": camera_px,
            "max_steps": int(ep["max_steps"]), "control_mode": "relative", "init_state_id": int(ep["state"]),
            "seed": int(ep["seed"])}
    if ep.get("world2"):  # the second test world (world2.py): our scene, start states and camera; evaluator side
        spec["_world2"] = ep["world2"]
    return spec


def adapter_class():
    """The service's ``LiberoEnvironmentAdapter`` without running ``libero_remote/__init__``, which imports the HTTP
    client (httpx), unused in process: the package is registered by its path only, then the module imports as is."""
    if "libero_remote" not in sys.modules:
        d = next((Path(p) / "libero_remote" for p in sys.path
                  if (Path(p) / "libero_remote" / "environment" / "libero.py").is_file()), None)
        if d is None:
            raise ImportError("Pro2.0's service source is not on the path")
        pkg = types.ModuleType("libero_remote")
        pkg.__path__ = [str(d)]
        sys.modules["libero_remote"] = pkg
    from libero_remote.environment.libero import LiberoEnvironmentAdapter

    return LiberoEnvironmentAdapter


def blank_step_images(env, px: int) -> None:
    """The episode's observation images, which the depth line never reads, as blank frames (``no_step_render``): on a
    CPU pod drawing them takes over half of an episode. The robot's looks (``depth=True``, their own size) still
    render, and physics never reads a render. A rebuilt simulator would simply draw again."""
    sim, real = env.sim, env.sim.render
    blank = np.zeros((px, px, 3), dtype=np.uint8)

    def render(*args, **kw):
        if not kw.get("depth") and kw.get("width", args[0] if args else None) == px:
            return blank.copy()
        return real(*args, **kw)

    sim.render = render


def _brief_fit(fit: dict | None) -> dict | None:
    return None if fit is None else {k: fit[k] for k in ("n_vis", "n_agree", "n_through", "inlier") if k in fit}


def look_brief(extra: dict) -> dict:
    """The record of a full look's ``extra`` block (``m2_perceive._report_extra``), without the bodies and joints."""
    rj = [{"id": q["id"], "type": q["type"], "pos": q.get("pos"), "source": q.get("source"), "fit": _brief_fit(q.get("fit")),
           "n_px": q.get("n_px"), "why": {k: v for k, v in (q.get("why") or {}).items() if v is not None}}
          for q in extra["rejected"]]
    kp = [{**q, "fit": _brief_fit(q.get("fit"))} for q in extra["kept"]]
    return {"kept": kp, "rejected": rj, "blobs": extra["blobs"]}


class FirstLook:
    """The audited client, keeping the brief of the episode's first full look (the loop's records; the robot's own
    calls and replies are untouched)."""

    def __init__(self, inner):
        self.inner, self.brief = inner, None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def call(self, msg: dict) -> dict:
        rep = self.inner.call(msg)
        if self.brief is None and msg.get("op") == "look" and msg.get("mode") == "full" and isinstance(rep, dict) \
                and rep.get("extra") is not None:
            self.brief = look_brief(rep["extra"])
        return rep


def robot_dir_sha256(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(d).glob("*.py")):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def robot(params: dict) -> types.SimpleNamespace:
    """The policy side: M2's frozen modules, or the loop's robot package (``rowant_lab/pro2/robot`` or a candidate
    copy of it) loaded from ``params["robot_dir"]`` under a name keyed by its content."""
    d = params.get("robot_dir")
    if not d:
        from rowant_lab.libero_env import m2_agent as AG
        from rowant_lab.libero_env import m2_ground as GD
        from rowant_lab.libero_env.oracle_agent import OracleAgent
        from rowant_lab.libero_env.run_suite import make_params

        return types.SimpleNamespace(PerceptionTool=AG.PerceptionTool, LOOK_BUDGET=AG.LOOK_BUDGET,
                                     HELD_NEAR_M=AG.HELD_NEAR_M, parse=GD.parse, ground=GD.ground,
                                     OracleAgent=OracleAgent, make_params=make_params, plan_episode=None, version="m2")
    sha = robot_dir_sha256(Path(d))
    name = f"pro2robot_{sha[:16]}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(d) / "__init__.py",
                                                      submodule_search_locations=[str(d)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    pkg = sys.modules[name]
    look, ground, agent = (importlib.import_module(f"{name}.{m}") for m in ("look", "ground", "agent"))
    epm = importlib.import_module(f"{name}.episode") if (Path(d) / "episode.py").is_file() else None
    plan = epm.plan_episode if epm is not None else None
    dly = importlib.import_module(f"{name}.delay") if (Path(d) / "delay.py").is_file() else None
    return types.SimpleNamespace(PerceptionTool=look.PerceptionTool, LOOK_BUDGET=look.LOOK_BUDGET,
                                 HELD_NEAR_M=look.HELD_NEAR_M, parse=ground.parse, ground=ground.ground,
                                 OracleAgent=agent.OracleAgent, make_params=pkg.make_params, plan_episode=plan,
                                 choose_cell=getattr(epm, "choose_cell", None),
                                 named_targets=getattr(epm, "named_targets", None),  # (2026-11-19: ``learn_forms``)
                                 DelayComp=getattr(dly, "DelayComp", None), version=sha)


def _practice_ok(ad, goal) -> bool:
    """EVALUATOR side, development only: a practice goal over the simulator's own object states, by its own predicates
    (each item ``[predicate, name]`` or ``[predicate, name, name]``, all must hold)."""
    inner = ad._env.env
    try:
        return all(bool(inner._eval_predicate([str(x) for x in g])) for g in goal)  # noqa: SLF001
    except KeyError:
        return False


def scene_moving(prev, cur, opt: dict | None = None) -> bool:
    """Whether two frames of a camera that did not move show something moving (2026-11-16, ``settle_first``).

    Depth frames (float, metres): at least ``px`` pixels changed by more than ``depth_m``. Colour frames (uint8,
    H x W x 3): the picture is cut into ``grid`` x ``grid`` blocks, and a block has changed when its mean colour moved
    by more than ``colour`` (of 255); something moves when at least ``blocks`` blocks changed. Either way a change of
    more than ``max_share`` of the picture is not motion in the scene (the exposure, noise, a dropped frame).
    The limits are fine on purpose: a thing settling its last centimetre moves 2 to 3 mm a step (2026-11-16: with
    4 mm and 6 of 255 the wait ended while a basket was still 1 cm up, and the first look read it there)."""
    o = {"depth_m": 0.0008, "px": 4, "grid": 64, "colour": 3.0, "blocks": 2, "max_share": 0.35, **(opt or {})}
    a, b = np.asarray(prev), np.asarray(cur)
    if a.shape != b.shape:
        return False
    if a.ndim == 2:  # depth
        ch = np.abs(b.astype(np.float64) - a.astype(np.float64)) > float(o["depth_m"])
        n = int(ch.sum())
        return int(o["px"]) <= n <= float(o["max_share"]) * ch.size
    g = int(o["grid"])
    h, w = (a.shape[0] // g) * g, (a.shape[1] // g) * g
    if h == 0 or w == 0:
        return False

    def blocks(x):
        x = x[:h, :w].astype(np.float32).reshape(g, h // g, g, w // g, -1)
        return x.mean(axis=(1, 3))

    ch = np.abs(blocks(b) - blocks(a)).mean(axis=-1) > float(o["colour"])
    n = int(ch.sum())
    return int(o["blocks"]) <= n <= float(o["max_share"]) * ch.size


def settle_first(step, frame, opt: dict | None = None) -> tuple:
    """Before the episode's first look: no-op steps until ``still`` pairs of frames in a row (2) show nothing moving,
    at most ``max_steps`` (2026-11-16: in one case the first frame shows objects 3 to 16 cm off
    in height and falling back, at rest by step 7, and the first look planned its grasp on them). ``step()`` sends
    one no-op and returns the adapter's reply; ``frame(reply)`` gives the frame to compare (``frame(None)``: the one
    before any step). A still scene costs two steps. Returns (the replies, the record)."""
    o = dict(opt or {})
    need = int(o.get("still", 2))
    prev = frame(None)
    replies, flags = [], []
    while len(replies) < int(o.get("max_steps", 12)):
        r = step()
        replies.append(r)
        cur = frame(r)
        flags.append(bool(scene_moving(prev, cur, o)))
        prev = cur
        if (len(flags) >= need and not any(flags[-need:])) or r.get("terminated") or r.get("truncated"):
            break
    return replies, {"steps": len(replies), "moving": flags}


def run_episode(ep: dict, client, params: dict, camera_px: int = 256) -> dict:
    """``ep``: suite, task, case, state, seed, bddl (the training task file), max_steps."""
    R = robot(params)
    pc = time.perf_counter
    t0 = pc()
    suite = ep["suite"]
    line = str(params.get("line", "depth"))
    rec = {k: ep[k] for k in ("suite", "task", "case", "state", "seed", "max_steps")}
    rec.update(line=line, bypassed=line == "depth" and ep["case"] in BYPASSED, robot=R.version)
    if ep.get("world2"):
        from rowant_lab.pro2 import world2 as W2

        ad = W2.adapter_class()()
    else:
        ad = adapter_class()()
    first = ad.create(env_spec(ep, camera_px))
    try:
        rep = first["environment_reproducibility"]
        rec.update(create_s=round(pc() - t0, 3), init_fp=rep["initial_state_fingerprint"],
                   plan_fp=rep["perturbation_plan_fingerprint"], instruction=first["language_instruction"])
        if params.get("no_step_render") and line == "depth":
            blank_step_images(ad._env, camera_px)
        return _act(ad, first, ep, rec, client, params, R, pc, t0)
    finally:
        ad.close()


def _act(ad, first, ep, rec, client, params, R, pc, t0) -> dict:
    suite, cap = ep["suite"], int(ep["max_steps"])
    cells = LP.cells_of(suite)
    if cells:  # the long-task suite: the robot picks its table from the instruction (the cells' types, as data)
        choose = getattr(R, "choose_cell", None)
        cell, why = choose(first["language_instruction"], cells) if choose else (None, "the robot has no choose_cell")
        rec["cell"] = {"key": cell, "reason": why}
        suite = cell if cell is not None else sorted(cells)[0]  # no cell named: the first, flagged in the record
    obs = proprio(first["observation"])
    line, images, CL = rec["line"], {}, None
    if line == "colour":  # COLOUR-ONLY LINE: the episode's own images (colour_line.py)
        from rowant_lab.pro2 import colour_line as CL

        images.update(CL.episode_images(first["observation"]))
        cam_tool = CL.make_tool(images, suite, params, CAM.make_camera_tool(ad._env) if CL.needs_sim(params) else None)
        rec["colour"] = CL.describe(params)
    else:
        cam_tool = CAM.make_camera_tool(ad._env)  # DEPTH LINE: the live simulator, not the episode's images
    wprobe = None
    if params.get("probe_wrist_stereo") is not None:
        # development only, EVALUATOR side, off unless given (probes/wrist_stereo_check.py, 2026-11-17): wrist motion
        # stereo against the simulator's depth at the in-episode learner's own looks. The robot's frames are unchanged.
        from rowant_lab.pro2.probes import wrist_stereo_check as WSC

        wprobe = WSC.Probe(ad._env, ep, line, params["probe_wrist_stereo"])
        cam_tool = wprobe.wrap(cam_tool)
    wframes = None
    if params.get("learn_wrist_stereo") is not None and params.get("learn_in_episode"):
        # 2026-11-17, off unless given (wrist_stereo.py): the learner's wrist frames carry colour, the pose from the
        # hand's readings and partner pictures, and NO depth; the perception server makes the depth (plane sweep).
        # On the depth line this is a development option: the simulator's wrist depth is withheld.
        from rowant_lab.pro2 import wrist_stereo as WSR

        wframes = WSR.WristFrames(params["learn_wrist_stereo"])
        cam_tool.wrist_frame = wframes.frame
    back = None
    if params.get("learn_way_back") is not None and params.get("learn_in_episode"):
        # 2026-11-18, off unless given (learn_skill.WAY_BACK): when a learned thing gives no plan or its attempts
        # fail, the episode goes back to the robot's own handling, planned on what the first look saw
        from rowant_lab.pro2 import learn_skill as LSB

        back = LSB.way_back(params)
    own1 = None
    if params.get("learn_after_fail") is not None and params.get("learn_in_episode"):
        # 2026-11-18, off unless given (learn_skill.AFTER_FAIL): the robot's own attempt first; when it has failed and
        # enough steps are left, the runner looks again and the learner starts from that look
        from rowant_lab.pro2 import learn_skill as LSF

        own1 = LSF.after_fail(params)
    retreat, home = None, None  # (``learn_after_fail.retreat``: the hand goes back before the new look)

    def wrist_push(obs_, t_, agent_):
        if not wframes.wanted(agent_):
            wframes.clear()
            return
        rgb = images[WSR.WRIST] if CL is not None else cam_tool(WSR.WRIST, wframes.W, int(t_))["rgb"]
        wframes.push(t_, rgb, obs_)
    _ob, policy_lib, m1_lib = LB.lfb_dirs(params.get("lfb_library"))
    lib = LB.M2Library(suite, lib_dir=policy_lib, m1_library=m1_lib, extra_suites=LP.extra_suites([suite]))
    aclient = FirstLook(AU.AuditedClient(client))
    episode = token(ep["suite"], ep["task"], ep["case"], ep["state"], ep["seed"])
    aclient.call({"op": "begin_episode", "episode": episode})
    tool = R.PerceptionTool(lib, aclient, cam_tool, suite, episode,
                             budget=int(params.get("look_budget", R.LOOK_BUDGET)),
                             held_near_m=float(params.get("held_near_m", R.HELD_NEAR_M)),
                             place_check=params.get("place_check"))
    if params.get("look_blind"):  # development only, off unless given: see ``PerceptionTool.blind``
        tool.blind = dict(params["look_blind"])
    pre = []  # the replies of the no-op steps taken before the first look (``settle_first``)
    if params.get("settle_first") is not None:
        # 2026-11-16, off unless ``params["settle_first"]`` (its options; {} = the defaults): the first look only
        # when the scene is at rest. The colour line compares the episode's own pictures; the depth line, whose looks
        # read the simulator, compares small depth frames from the same camera tool.
        def _frame(r):
            if CL is not None:
                if r is not None:
                    images.update(CL.episode_images(r["observation"]))
                return images["agentview"].copy()
            return np.asarray(cam_tool("agentview", 128, len(pre))["depth_u16"], dtype=np.float64) * 1e-4

        pre, rec["settle"] = settle_first(lambda: ad.step(NOOP), _frame, params["settle_first"])
        if pre:
            obs = proprio(pre[-1]["observation"])
    tool.obs = obs
    a = pc()
    first_rep = tool.full_look()
    if tool.place_check:
        tool.place_base()
    perception_s = pc() - a
    a = pc()
    instruction = first["language_instruction"]
    practice = ep.get("practice")  # development only: our own instruction on a training scene, with its own goal
    practice_at = None
    if practice:
        instruction = str(practice["instruction"])
    agent_err = perception_error = agent_trace = perception_trace = None
    parsed, grounding, agent = {}, {}, None
    learn_rec = None
    try:
        if R.plan_episode is not None and params.get("learn_in_episode"):  # W2-LEARN-EP: learn an unexplained target
            from rowant_lab.pro2 import learn_skill as LS

            planned = LS.plan(R, instruction, tool, lib, cap - len(pre), params, first_rep)
            agent, parsed, grounding = planned.get("agent"), planned.get("parsed") or {}, planned.get("grounding") or {}
            learn_rec = planned.get("learn")
        elif R.plan_episode is not None:  # the loop's robot decides what to do (episode.py): its parsing, grounding, skills
            planned = R.plan_episode(instruction, tool, lib, cap - len(pre), params)
            agent, parsed, grounding = planned.get("agent"), planned.get("parsed") or {}, planned.get("grounding") or {}
        else:
            parsed = R.parse(instruction, sorted(lib.types))
            if parsed["status"] != "ok":
                grounding = {"status": "fail", "reason": parsed.get("reason")}
            else:
                if hasattr(tool, "on_parsed"):  # the loop's robot may add percepts, knowing what the instruction names
                    tool.on_parsed(parsed)
                grounding = R.ground(parsed, tool.percepts, lib, lib.workcell)
    except Exception as exc:
        if isinstance(exc, EP.RUNNER_EVENTS):  # infrastructure, never an outcome
            raise
        agent_err = f"{type(exc).__name__}: {exc}"[:500]
        agent_trace = [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in traceback.extract_tb(exc.__traceback__)][-6:]
        agent, grounding = None, {"status": "fail", "reason": "crash while grounding"}
    if R.plan_episode is None and grounding.get("status") != "fail" and grounding.get("target") is not None:
        tool.target = grounding["target"]
        syn = lib.scene_from_percepts(tool.percepts)
        place_p = next(p for p in tool.percepts if p["id"] == grounding["place"])
        site = lib.site_name_for(place_p, None) if parsed["predicate"] == "in" else None
        pspec = {"predicate": parsed["predicate"], "target": grounding["target"],
                 "place": f"{place_p['id']}_{site}" if site else place_p["id"],
                 "place_kind": "region" if site else "object", "place_instance": place_p["id"],
                 "instruction": instruction}
        ap, pp, fp = R.make_params(params.get("m1"))
        agent = R.OracleAgent(lib, syn, pspec, tool, cap, ap, pp, fp)
    planning_s = pc() - a
    calib = None
    want = (CL is not None and CL.options(params)["calib"] == "arm") or (CL is None and params.get("arm_joints") is not None)
    if want and R.plan_episode is not None and agent_err is None:
        from rowant_lab.pro2.arm_calib import ArmCalib  # item 2: the camera from the robot's own arm, first motion
        # (item 4: and the joint readings' offsets, ``arm_joints``; on the depth line only those)

        calib = ArmCalib(cam_tool, aclient, suite, episode, params)
        calib.start(obs)
    raw = obs  # item 4: the readings as reported (the arm check's frames); ``obs`` is what the robot is given
    # 2026-11-15 (off unless ``params["delay_comp"]``; the robot's own ``delay.DelayComp``): acting through a delay. The
    # robot steers by the hand where its next command will find it; looks keep the readings as they came (the pose
    # the cameras show); after a change of the gripper's direction the hand is held while the readings catch up.
    lagc = None
    if params.get("delay_comp") is not None and getattr(R, "DelayComp", None) is not None and agent is not None:
        lagc = R.DelayComp(params["delay_comp"])
        lagc.start(obs)
    agent_s = step_s = 0.0
    actions, t, finished_at = [[float(x) for x in NOOP] for _ in pre], len(pre), None
    earlier: list = []  # the record only: what each plan had done when another was made in its place
    success = any(bool(x["success"]) for x in pre)
    done = bool(pre and (pre[-1]["terminated"] or pre[-1]["truncated"]))
    r: dict = pre[-1] if pre else {}
    if wprobe is not None:
        wprobe.after_step(raw, t, agent)
    if wframes is not None:
        wrist_push(raw, t, agent)
    while not done:
        act = NOOP
        foreign = False  # this step's command is the arm check's own motion (the delay compensator does not time it)
        if calib is not None and not success and agent_err is None and perception_error is None \
                and calib.can_move(agent, finished_at is not None, t, cap):
            calib.begin_motion(obs, t)  # the robot gave up (or never started) before its camera was checked
        if own1 is not None and home is None:
            try:
                home = LSF.hand_pose(obs)  # where the hand stands as the episode's commands begin
            except Exception:
                home = None
        if calib is not None and calib.moving:
            act = np.asarray(calib.motion_action(obs), dtype=np.float32)
            foreign = True
        elif retreat is not None:
            # the robot's own start failed and the learner is due: the hand goes back first, so that the new look
            # sees the thing (the agent is not asked; the delay compensator does not time these steps)
            act = np.asarray(LSF.retreat_action(obs, retreat), dtype=np.float32)
            foreign = True
        elif agent is not None and finished_at is None and agent_err is None and perception_error is None:
            if bool(getattr(agent, "finished", False)):
                finished_at = t
            elif lagc is not None and (lagc.busy() or lagc.holding()):
                # the lag is being measured, or the hand is held after a gripper command: the robot is not asked
                act = np.asarray(lagc.own_action() if lagc.busy() else lagc.hold_action(), dtype=np.float32)
                for x in (agent, getattr(agent, "cur", None)):  # these steps are steps of the robot's budget too
                    if x is not None and hasattr(x, "t"):
                        x.t += 1
            else:
                tool.obs = obs
                if lagc is not None:  # the robot waits this long before a check that compares a look with its hand
                    for x in (agent, getattr(agent, "cur", None)):
                        if x is not None:
                            x.lag_steps = int(lagc.lag)
                            # 2026-11-17 (stream LAG; read only by options that are off by default): the readings'
                            # own delay, and whether a closing hand is held for the commands' delay alone (the agent
                            # then closes without reading its fingers and reads them when their reading is in)
                            x.lag_obs_steps = int(getattr(lagc, "obs_lag", 0) or 0)
                            x.lag_close_blind = bool((getattr(lagc, "o", None) or {}).get("close_hold_cmd"))
                b0 = tool.bridge_s + tool.render_s
                a = pc()
                try:
                    act = np.asarray(agent.act(obs if lagc is None else lagc.predict(obs)), dtype=np.float32)
                    if act.shape != (7,) or not np.isfinite(act).all():  # the simulator's step would raise on these
                        raise ValueError(f"an action must be 7 finite numbers, got shape {act.shape}")
                except Exception as exc:
                    if isinstance(exc, EP.RUNNER_EVENTS):  # infrastructure, never an outcome
                        raise
                    if isinstance(exc, BR.RemoteError):
                        perception_error = str(exc)[:500]
                        perception_trace = server_trace(exc)
                    else:
                        agent_err = f"{type(exc).__name__}: {exc}"[:500]
                        agent_trace = [f"{Path(f.filename).name}:{f.lineno} {f.name}"  # where, file names only
                                       for f in traceback.extract_tb(exc.__traceback__)][-6:]
                    act = NOOP
                if params.get("learn_in_episode") and type(agent).__name__ == "LearnThenAct" \
                        and getattr(agent, "inner", None) is None:
                    # 2026-11-17: the learner's looks are not the delay compensator's to time or judge. On the colour
                    # line its check for a caught arm took the hand away in the middle of a look and planned again.
                    foreign = True
                if lagc is not None:
                    lagc.sent(act)
                    act = np.asarray(lagc.shape(act), dtype=np.float32)
                dt, db = pc() - a, (tool.bridge_s + tool.render_s) - b0
                perception_s += db
                agent_s += dt - db
        a = pc()
        r = ad.step(act)
        step_s += pc() - a
        actions.append([float(x) for x in act])
        t += 1
        obs = raw = proprio(r["observation"])
        if wprobe is not None:
            wprobe.after_step(raw, t, agent)
        if calib is not None:
            obs = calib.correct(raw)  # item 4: the fitted offsets off every reading once a fit was accepted
        if lagc is not None:
            if foreign and getattr(lagc, "takes_foreign", False):  # (a robot package from before 2026-11-16 does not)
                lagc.after_step(act, obs, foreign=True)
            else:
                lagc.after_step(act, obs)
        if CL is not None:
            images.update(CL.episode_images(r["observation"]))
        if wframes is not None:
            # after the images are this step's (the colour line's picture and reading belong to one moment: measured
            # 2026-11-17, both about 5 ms before the step's end); the readings with the arm check's fitted offsets off
            wrist_push(obs, t, agent)
        success = success or bool(r["success"])
        done = bool(r["terminated"] or r["truncated"])
        if practice and practice_at is None and _practice_ok(ad, practice["goal"]):  # EVALUATOR side
            practice_at, done = t, True  # as the benchmark ends an episode when its goal first holds
        again = None
        if lagc is not None and not done and agent_err is None and perception_error is None and lagc.replan_due():
            again = "delay"  # the lag was measured during the first approach: the hand is not where the plan began
        if calib is not None and not done and agent_err is None and perception_error is None \
                and calib.after_step(raw, t):
            again = "calib"
        if back is not None and again is None and not done and agent_err is None and perception_error is None:
            back_doc = LSB.back_due(agent, t, cap, back)
            if back_doc is not None:
                # the learned thing gave no plan or used up its attempts, and enough steps are left: the learner's
                # agent takes the robot's own plan from here (no new look: the arm stands in the picture now)
                a = pc()
                try:
                    agent.go_back(back_doc, cap - t)
                except Exception as exc:
                    if isinstance(exc, EP.RUNNER_EVENTS):  # infrastructure, never an outcome
                        raise
                    if isinstance(exc, BR.RemoteError):
                        perception_error = str(exc)[:500]
                        perception_trace = server_trace(exc)
                    else:
                        agent_err = f"{type(exc).__name__}: {exc}"[:500]
                        agent_trace = [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in traceback.extract_tb(exc.__traceback__)][-6:]
                planning_s += pc() - a
        fail_doc = None
        if own1 is not None and again is None and not done and agent_err is None and perception_error is None:
            # The arm's camera check comes first. A robot that gave up at once is the check's cue to move the arm
            # by itself, and a camera found moved is looked through again before the learner is asked (Mac,
            # practice Object 6 S16: v9 has no plan at step 0, checks its camera, finds it moved and solves the
            # episode; the learner started at step 0 instead and lost it).
            waits = calib is not None and (calib.moving or calib.can_move(
                agent, bool(getattr(agent, "finished", False)), t, cap))
            if retreat is not None:
                fail_doc = LSF.retreat_done(obs, retreat, t, cap)  # (None while the hand is on its way back)
                if fail_doc is not None:
                    retreat = None
            elif not waits:
                fail_doc = LSF.fail_due(agent, learn_rec, t, cap, own1)
                if fail_doc is not None:
                    retreat = LSF.retreat_begin(fail_doc, home, t, own1)
                    if retreat is not None:
                        fail_doc = None  # (the look and the learner follow when the hand is back)
            if fail_doc is not None:
                again = "own_failed"  # the robot's own start failed: look again, and the learner starts from that look
        if again is not None and again != "own_failed":
            retreat = None  # (a plan made again for another reason ends a retreat: the new plan decides anew)
        if again is not None:
            # the arm showed the camera had moved and the fit replaced it (item 4: or that the joint readings are off,
            # and they are corrected from here): look and plan again from here, with the steps left (the first plan's
            # look counts are kept in the record)
            if calib is not None:
                obs = calib.correct(raw)
            a = pc()
            try:  # the record only: what the plan that ends here had done (the row's own record is the new plan's)
                prev = agent.summary() if agent is not None else {}
                earlier.append({"t": t, "why": again, "finish_reason": prev.get("finish_reason"),
                                "steps": prev.get("steps"), "attempts": prev.get("attempts"),
                                **({"sequence": prev["sequence"]} if prev.get("sequence") else {}),
                                "looks": {"full": tool.n_full, "track": tool.n_track, "predicted": tool.n_predicted}})
            except Exception:
                pass
            if again == "calib":
                calib.rec["first_looks"] = {"full": tool.n_full, "track": tool.n_track, "predicted": tool.n_predicted}
            # ``keep_target`` (2026-11-16, off by default): what the first plan chose goes over to the new look tool.
            # Not after a camera fit: the remembered places were measured through the camera as it was believed then.
            prior = None
            if params.get("keep_target") and again != "calib" and isinstance(getattr(tool, "prior", None), dict):
                prior = dict(tool.prior)
                try:
                    memo = tool.remember()  # (where the target is believed now: in the hand, or where it was left)
                    if memo is not None and instruction in prior:
                        prior[instruction] = memo
                except Exception:
                    pass
            tool = R.PerceptionTool(lib, aclient, cam_tool, suite, episode,
                                     budget=int(params.get("look_budget", R.LOOK_BUDGET)),
                                     held_near_m=float(params.get("held_near_m", R.HELD_NEAR_M)),
                                     place_check=params.get("place_check"))
            tool.obs = obs
            if params.get("look_blind"):
                tool.blind = dict(params["look_blind"])
            if prior is not None:
                tool.prior = prior
            try:
                rep2 = tool.full_look()
                if tool.place_check:
                    tool.place_base()
                if params.get("learn_in_episode"):  # 2026-11-17: the learner again, from this look (before, a plan
                    from rowant_lab.pro2 import learn_skill as LS2  # made again on the way dropped it)

                    # 2026-11-18 (``learn_keep``, off unless given): after a caught arm the learned thing is gone on with
                    planned = None
                    if fail_doc is not None:  # (off unless given) the robot's own start failed: the learner, from this look
                        planned = LS2.plan_after_fail(R, instruction, tool, lib, cap - t, params, rep2, fail_doc)
                    # (``again_after_fail``: where the learner had started that way, it starts again from this look)
                    planned = planned or LS2.keep_learned(agent, tool, cap - t, params, again, t) \
                        or (own1 is not None
                            and LS2.again_after_fail(R, instruction, tool, lib, cap - t, params, rep2, learn_rec)) \
                        or LS2.plan(R, instruction, tool, lib, cap - t, params, rep2)
                    learn_rec = LS2.keep_first(learn_rec, planned.get("learn"), t)  # (record only)
                else:
                    planned = R.plan_episode(instruction, tool, lib, cap - t, params)
                agent, parsed, grounding = planned.get("agent"), planned.get("parsed") or {}, planned.get("grounding") or {}
                if wframes is not None:
                    wrist_push(obs, t, agent)  # a learner planned again needs the picture of where the hand is now
            except Exception as exc:
                if isinstance(exc, EP.RUNNER_EVENTS):  # infrastructure, never an outcome
                    raise
                if isinstance(exc, BR.RemoteError):
                    perception_error = str(exc)[:500]
                    perception_trace = server_trace(exc)
                else:
                    agent_err = f"{type(exc).__name__}: {exc}"[:500]
                    agent_trace = [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in traceback.extract_tb(exc.__traceback__)][-6:]
                agent, grounding = None, {"status": "fail", "reason": "crash while planning again"}
            finished_at = None
            if again == "calib":
                calib.rec.update(replanned_at=t, replan_s=round(pc() - a, 3))
            elif again == "own_failed":
                if isinstance(learn_rec, dict) and isinstance(learn_rec.get("own_first"), dict):
                    learn_rec["own_first"]["replan_s"] = round(pc() - a, 3)
            else:
                lagc.rec["replanned_at"] = t
            perception_s += pc() - a
    try:
        summ = agent.summary() if agent is not None else {}
    except Exception as exc:  # a skill's summary() is the loop's code: a crash is an outcome of the episode
        if isinstance(exc, EP.RUNNER_EVENTS):
            raise
        summ = {}
        agent_err = agent_err or f"{type(exc).__name__}: {exc}"[:500]
    rec.update(success=success, terminated=bool(r.get("terminated")), truncated=bool(r.get("truncated")), steps=t,
               policy_steps=summ.get("steps"), finish_reason=summ.get("finish_reason") or (
                   "grounding_failed" if agent is None else None),
               finished_at=finished_at, n_attempts=len(summ.get("attempts") or []), attempts=summ.get("attempts"),
               events=summ.get("events"), agent_error=agent_err, agent_trace=agent_trace,
               **({"sequence": summ["sequence"]} if summ.get("sequence") else {}),  # the long suite's steps
               perception_error=perception_error,
               # record only (2026-11-18): where in the server the failed request raised, file names and lines
               **({"perception_trace": perception_trace} if perception_trace else {}),
               grounding={"status": grounding.get("status", "ok"), "reason": grounding.get("reason"),
                          "predicate": parsed.get("predicate"),
                          # record only (2026-11-16): which percepts were chosen, the flags, and the fallbacks used
                          **({k: grounding[k] for k in ("target", "place", "flags", "costs", "steps") if grounding.get(k)}),
                          **({"fallbacks": list(tool.fallbacks)} if getattr(tool, "fallbacks", None) else {})},
               looks={"full": tool.n_full, "track": tool.n_track, "predicted": tool.n_predicted,
                      # record only (2026-11-18): the pixels each full look took as the robot's own
                      **({"own_px": list(tool.own_px)} if getattr(tool, "own_px", None) else {}),
                      **({"own_dz": list(tool.own_dz)} if getattr(tool, "own_dz", None) else {}),
                      **({"close": tool.n_close} if getattr(tool, "n_close", 0) else {}),  # item 3's close looks
                      **({"faced": tool.n_faced} if getattr(tool, "n_faced", 0) else {}),  # fixtures turned to face in
                      # a target taken as it is seen (``seen_target``): which fallbacks were made that way
                      **({"seen": [x for x in getattr(tool, "fallbacks", []) if x.get("source") == "seen"]}
                         if any(x.get("source") == "seen" for x in getattr(tool, "fallbacks", [])) else {}),
                      # a target looked for where the main camera does not see: the wrist pictures and what they gave
                      **({"wrist": [x for x in tool.looks if x.get("mode") == "wrist"],
                          "behind": [x for x in getattr(tool, "fallbacks", []) if x.get("source") in ("behind", "wrist")]}
                         if any(x.get("mode") == "wrist" for x in tool.looks) or getattr(tool, "n_behind", 0) else {}),
                      # the record: what hung in view was not taken for a panel, and why (``look._occluders``)
                      **({"behind_refused": tool.behind_refused} if getattr(tool, "behind_refused", None) else {}),
                      **({"dropped": tool.n_drop} if getattr(tool, "n_drop", 0) else {}),  # releases told to fall
                      **({"held_kept": tool.n_held_kept} if getattr(tool, "n_held_kept", 0) else {}),  # in-hand jumps refused
                      **({"redetect": tool.n_redetect} if getattr(tool, "n_redetect", 0) else {})},  # item 5
               look_budget_exhausted=bool(tool.budget_exhausted), look0=aclient.brief, actions=actions,
               timing={"perception_s": round(perception_s, 3), "planning_s": round(planning_s, 3),
                       "agent_s": round(agent_s, 3), "env_step_s": round(step_s, 3),
                       "render_s": round(tool.render_s, 3), "bridge_s": round(tool.bridge_s, 3),
                       "server_s": round(float(getattr(tool, "server_s", 0.0)), 3), "total_s": round(pc() - t0, 3)})
    if earlier:  # a plan was made again on the way (the arm's camera check, the delay compensator)
        rec["before_replan"] = earlier
    if params.get("record_looks"):  # debugging, off by default: every look's own record (numbers only)
        rec["look_log"] = list(tool.looks)
    if practice:
        rec["practice"] = {"instruction": instruction, "goal": practice["goal"], "success": practice_at is not None,
                           "at": practice_at}
    if params.get("learn_in_episode"):
        rec["learn"] = learn_rec
    if wprobe is not None:
        rec["wrist_stereo"] = wprobe.record()
    if params.get("look_log"):  # every look of the episode as the tool recorded it (development only; numbers)
        try:  # through JSON once, so that no odd value in a look's record can stop the row from being written
            import json

            rec["look_log"] = json.loads(json.dumps([{k: v for k, v in lk.items() if k != "times"}
                                                     for lk in getattr(tool, "looks", [])],
                                                    default=lambda o: float(o) if isinstance(o, np.generic) else str(o)))
            rec["look_calls"] = [int(x) for x in getattr(tool, "call_steps", [])]
        except Exception as exc:
            rec["look_log"] = f"{type(exc).__name__}: {exc}"[:200]
    if lagc is not None:
        rec["delay"] = {"lag": int(lagc.lag), "gain": round(float(lagc.gain), 4), **lagc.rec}
    if calib is not None:
        rec["arm_cam"] = dict(calib.rec)
        if CL is not None:
            rec["arm_cam"]["eval_camera"] = _camera_errors(ad, cam_tool, suite)  # EVALUATOR side
        if calib.joints is not None:
            rec["arm_cam"]["eval_joints"] = _joint_errors(ad, raw)  # EVALUATOR side
    return rec


def _joint_errors(ad, raw: dict) -> dict | None:
    """EVALUATOR side (item 4): the reported joint and hand readings against the simulator's own at the last step, as
    the offsets the robot would have to find (deg, mm)."""
    try:
        s = ad._env.sim
        q = np.array([s.data.get_joint_qpos(f"robot0_joint{i}") for i in range(1, 8)], float)
        eef = np.asarray(s.data.site_xpos[s.model.site_name2id("gripper0_grip_site")], float)
        return {"joint_deg": [round(float(x), 3) for x in np.degrees(np.asarray(raw["robot0_joint_pos"], float) - q)],
                "hand_mm": [round(float(x) * 1000, 2) for x in np.asarray(raw["robot0_eef_pos"], float) - eef]}
    except Exception as exc:  # scoring only; never an outcome
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def _camera_errors(ad, cam_tool, suite: str) -> dict | None:
    """EVALUATOR side (the runner's simulator handle; the robot never sees it): how far the stored camera and the camera
    the robot ended with are from the simulator's agentview camera, as (m, deg)."""
    try:
        from rowant_lab.libero_env.arm_camera import pose_diff
        from rowant_lab.pro2 import colour_line as CL

        s = ad._env.sim
        cid = s.model.camera_name2id("agentview")
        T = np.eye(4)
        T[:3, :3] = np.asarray(s.data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
        T[:3, 3] = np.asarray(s.data.cam_xpos[cid], dtype=np.float64)
        T = T @ CAM.CV_FROM_GL
        st = CL.stored_cameras()[suite]
        used = cam_tool.override.get("T")
        out = {"used": "fitted" if used is not None else "stored"}
        if used is not None:
            out["fitted_err"] = [round(x, 4) for x in pose_diff(np.asarray(used, float), T)]
        out["stored_err"] = [round(x, 4) for x in pose_diff(np.asarray(st["T_world_from_camera"], float), T)]
        last = getattr(cam_tool.override.get("register"), "last", None)  # stream PATH, round 3: both fits' cameras
        for k in ("T_depth", "T_pixels"):
            if last and last.get(k) is not None:
                out[k[2:] + "_err"] = [round(x, 4) for x in pose_diff(np.asarray(last[k], float), T)]
        return out
    except Exception as exc:  # scoring only; never an outcome
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}
