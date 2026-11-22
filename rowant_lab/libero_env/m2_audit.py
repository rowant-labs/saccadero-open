"""M2 runtime runner-side input audit (plan truth_guard.runner_side; verifier blocker B2). Runner side, numpy only.

M2 (perceived poses, grounded instruction): perception in the loop. In every episode the worker records what the
policy and the perception server actually received, and a test (``tests/test_m2_runtime.py``) checks a recorded DEV
episode:

* the policy-side tool and M1's agent receive only the proprioception whitelist (``tools.filter_obs`` output: every
  observation the runner hands either of them is recorded by its key set), the instruction string, the workcell
  model, the library (rooted at ``m2_paths.POLICY_LIB``) and the camera tool closure (``m2_camera.make_camera_tool``'s
  ``look``) with the bridge client;
* a reachability walk over the tool's, the agent's and the library's attributes (dicts, lists, objects; functions are
  not entered: the camera closure is the one declared callable) finds no simulator, LIBERO, robosuite, MuJoCo or
  evaluator object, no truth-side dict (``goal``, ``obj_of_interest``, ``robot_start``, ``obs_keys``, truth keys) and
  no other callable;
* ``AuditedClient`` records the key schema of every request the server receives: only frames (exactly the camera
  tool's keys), joint state, the episode token, the look index, the mode, the suite and track priors.
"""

from __future__ import annotations

import re
import types as _types

import numpy as np

from rowant_lab.libero_env import tools as T

TOKEN = re.compile(r"^[0-9a-f]{16}$")
TRUTH_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]*_\d+_(pos|quat)$")
TRUTH_DICT_KEYS = frozenset({"goal", "obj_of_interest", "robot_start", "obs_keys", "truth", "truth_log",
                             "object-state", "init_state", "init_states", "regions", "seen_in", "exported_from",
                             "rest_up_from"})
FORBIDDEN_MODULE_PREFIXES = ("robosuite", "libero", "mujoco", "gym", "torch", "rowant_lab.libero_env.m2_evaluator",
                             "rowant_lab.libero_env.evaluator", "rowant_lab.libero_env.tools",
                             "rowant_lab.libero_env.m2_episode", "rowant_lab.libero_env.m2_run",
                             "rowant_lab.libero_env.m2_evalx", "rowant_lab.libero_env.m2_gatereport")
FRAME_KEYS = ("camera", "rgb", "depth_u16", "depth_scale_mm", "K", "T_world_from_camera", "width", "height", "step")
REQUEST_KEYS = {
    "hello": {"op", "protocol", "suites", "client", "code_sha256", "library_sha256", "frozen_config_sha256"},
    "begin_episode": {"op", "episode"},
    "look": {"op", "mode", "suite", "episode", "look", "cameras", "joint_pos", "gripper_qpos", "priors", "base", "rise_m",
             "region"},
    "stats": {"op"},
    "bye": {"op"},
    # in-episode learning (lfb_episode; W2-LEARN-EP): the first look's frame and percepts back to the server, a wrist
    # frame for the camera's mount, then the wrist looks
    "learn_plan": {"op", "suite", "episode", "cameras", "joint_pos", "gripper_qpos", "percepts", "plane", "wrist",
                   "rejected",  # (``learn_rank_known``: the first look's rejected poses of types it accepted nowhere)
                   "stage", "top",  # (``learn_top_first``: the look from above, then its frame for the second plan)
                   "ask", "next"},  # (``learn_ask_known``: the known types to fit to the thing; the next thing)
    "learn": {"op", "suite", "episode", "label", "cameras", "joints"},
    # the camera from the robot's own arm (arm_camera; item 2): depth frames of the robot's first motion, each with the
    # assumed camera and the proprioception at that step
    "arm_cam": {"op", "mode", "suite", "episode", "frames"},
}
# a wrist frame without depth for the in-episode learner (lfb_episode.wrist_depth, 2026-11-17): the usual keys, then
# the wrist mount, the gripper reading and the partner pictures (each with its own camera and gripper reading)
STEREO_FRAME_KEYS = FRAME_KEYS + ("X", "gripper_qpos", "partners")
PARTNER_KEYS = {"rgb", "K", "T_world_from_camera", "gripper_qpos", "steps_back", "b_eff_m"}
ARM_FRAME_KEYS = {"depth_u16", "depth_scale_mm", "K", "T_world_from_camera", "joint_pos", "gripper_qpos"}
# item 4's readings check (``arm_cam`` mode ``readings``, the colour line's guard): proprioception only, no image
# (2026-11-16: and mode ``readings_fit``, the readings' offsets from the readings alone, the same keys)
ARM_READING_KEYS = {"joint_pos", "gripper_qpos", "eef_pos"}
ARM_READING_FIT_MORE = {"joint_vel"}  # ``readings_fit`` may carry the joint velocities too (the hand reading's lag)
PRIOR_KEYS = {"gid", "R_w", "t_w"}
EVAL_REPLY_KEY = "eval_masks"  # m2f2:readiness-r1: the server's evaluator-only reply block, never seen by the tool
TOOL_ATTRS = {"lib", "client", "cam", "suite", "episode", "budget", "calls", "call_steps", "t_now", "percepts",
              "target", "obs", "looks", "n_full", "n_track", "n_predicted", "budget_exhausted", "bridge_s",
              "render_s", "server_s", "_held", "_held_R0", "_held_t0", "_open_gap"}
SPEC_KEYS = {"predicate", "target", "place", "place_kind", "place_instance", "instruction"}
SCENE_KEYS = {"instances", "sites", "arena", "robot_mount"}
INSTANCE_KEYS = {"type", "kind", "bodies", "root_body_name"}
SITE_KEYS = {"parent_instance", "frame_body", "pos", "R", "size"}
PERCEPT_ID = re.compile(r"^p\d+$")


class AuditedClient:
    """Wraps the bridge client: records the key schema of every request (runner side)."""

    def __init__(self, inner, legacy_ok: bool = False, forbid_array=None):
        self.inner = inner
        self.legacy_ok = bool(legacy_ok)
        self.n = 0
        self.ops: dict = {}
        self.keys: dict = {}
        self.camera_keys: set = set()
        self.violations: list = []
        # m2f:perceive-r2 (verifier blocker B1): the GATE path's init-state row; no request may carry it. Only its
        # FINGERPRINT (size and sha256 of its float64 bytes) is kept: the policy-side tool holds this client, so the
        # row itself must not be reachable from it (the first dry run's walk found it here)
        self.forbid_fp = None if forbid_array is None else fingerprint(forbid_array)
        self.forbid_hits = 0
        # m2f2:readiness-r1: the server's evaluator-only reply block (its L0 segmentation masks, sent only when its
        # ``eval_masks`` parameter is on) is taken off the reply here, runner side, before the tool sees the reply;
        # the runner takes it right after the look (``take_eval``)
        self._eval: dict = {}

    def _check(self, msg: dict) -> None:
        op = msg.get("op")
        self.n += 1
        self.ops[op] = self.ops.get(op, 0) + 1
        ks = set(msg)
        self.keys.setdefault(op, set()).update(ks)
        allowed = REQUEST_KEYS.get(op)
        if allowed is None:
            self.violations.append(f"request op {op!r} is not declared")
            return
        if not ks <= allowed:
            self.violations.append(f"{op}: undeclared keys {sorted(ks - allowed)}")
        ep = msg.get("episode")
        if ep is not None and not TOKEN.match(str(ep)) and not self.legacy_ok:
            self.violations.append(f"{op}: the episode field is not an opaque token")
        for fr in msg.get("cameras") or []:
            self.camera_keys.update(fr)
            stereo = (op == "learn" and tuple(fr) == STEREO_FRAME_KEYS
                      and all(isinstance(p, dict) and set(p) == PARTNER_KEYS for p in fr["partners"]))
            if tuple(fr) != FRAME_KEYS and not stereo:
                self.violations.append(f"{op}: camera keys {sorted(fr)}")
        if op == "arm_cam":
            want = ARM_READING_KEYS if msg.get("mode") in ("readings", "readings_fit") else ARM_FRAME_KEYS
            more = ARM_READING_FIT_MORE if msg.get("mode") == "readings_fit" else set()
            for fr in msg.get("frames") or []:
                if set(fr) - more != want:
                    self.violations.append(f"arm_cam: frame keys {sorted(fr)}")
        for pid, pr in (msg.get("priors") or {}).items():
            if not PERCEPT_ID.match(str(pid)) or set(pr) != PRIOR_KEYS:
                self.violations.append(f"{op}: prior {pid!r} keys {sorted(pr)}")

    def call(self, msg: dict) -> dict:
        self._check(msg)
        if self.forbid_fp is not None and carries_array(msg, self.forbid_fp):
            self.forbid_hits += 1
            self.violations.append(f"{msg.get('op')}: a request carries the init-state array")
        rep = self.inner.call(msg)
        if isinstance(rep, dict) and EVAL_REPLY_KEY in rep:
            self._eval[int(msg.get("look", -1))] = rep.pop(EVAL_REPLY_KEY)
        return rep

    def take_eval(self) -> dict:
        """{look index: the evaluator-only reply block}, then cleared (runner side)."""
        out, self._eval = self._eval, {}
        return out

    def record(self) -> dict:
        return {"n": self.n, "ops": dict(self.ops), "keys": {k: sorted(v) for k, v in self.keys.items()},
                "camera_keys": sorted(self.camera_keys), "violations": list(self.violations),
                "ok": not self.violations}


def fingerprint(a) -> tuple:
    """(size, sha256 of the C-order float64 bytes) of an array: what the checks compare, never the values."""
    import hashlib

    b = np.ascontiguousarray(np.asarray(a, dtype=np.float64).ravel())
    return int(b.size), hashlib.sha256(b.tobytes()).hexdigest()


def _same_array(a, fp) -> bool:
    """Is array ``a`` (any shape) equal in values to the array with fingerprint ``fp``?"""
    try:
        a = np.asarray(a)
        if a.dtype == object or a.size != fp[0] or a.size == 0:
            return False
        return fingerprint(a) == fp
    except (TypeError, ValueError):
        return False


def carries_array(obj, ref, max_depth: int = 8) -> bool:
    """Does ``obj`` (a request: dicts, lists, arrays) hold an array equal to ``ref`` (a fingerprint, or an array)?"""
    if not (isinstance(ref, tuple) and len(ref) == 2 and isinstance(ref[1], str)):
        ref = fingerprint(ref)
    stack = [(obj, 0)]
    while stack:
        o, d = stack.pop()
        if d > max_depth:
            continue
        if isinstance(o, np.ndarray):
            if _same_array(o, ref):
                return True
        elif isinstance(o, dict):
            stack.extend((v, d + 1) for v in o.values())
        elif isinstance(o, (list, tuple)):
            if o and all(isinstance(x, (int, float)) for x in o) and len(o) == ref[0] and _same_array(o, ref):
                return True
            stack.extend((v, d + 1) for v in o)
    return False


def state_reach(init_state, k: int, client: "AuditedClient", tool, agent=None) -> dict:
    """The GATE path's runtime check (m2f:perceive-r2, verifier blocker B1 fix 4): the init-state row and the state
    index ``k`` never reached the perception server, the policy-side tool or M1's agent. (1) the audited client saw
    no request carrying the array and every request's episode field was the opaque token (never ``k``); (2) a walk of
    the tool's and the agent's attributes finds no array equal to the row and no attribute or key named like a state
    index."""
    ref = fingerprint(init_state)
    v = []
    if client.forbid_fp is None or client.forbid_fp != ref:
        v.append("the audited client was not armed with this init state")
    if client.forbid_hits:
        v.append(f"{client.forbid_hits} request(s) carried the init-state array")
    if any("opaque token" in x for x in client.violations):
        v.append("a request's episode field was not the opaque token")
    names = re.compile(r"^(state|init_state|state_index|k_state|gate_state)s?$")
    for root, label in ((tool, "tool"), (agent, "agent")):
        if root is None:
            continue
        seen: set = set()
        stack = [(root, label, 0)]
        while stack:
            o, path, d = stack.pop()
            if id(o) in seen or d > 8:
                continue
            seen.add(id(o))
            if isinstance(o, np.ndarray):
                if _same_array(o, ref):
                    v.append(f"{path}: an array equal to the init state")
                continue
            if o is None or isinstance(o, (bool, int, float, complex, str, bytes, np.generic)) or callable(o) \
                    and not hasattr(o, "__dict__"):
                continue
            if isinstance(o, (_types.ModuleType, _types.FunctionType, _types.MethodType, _types.BuiltinFunctionType,
                              type)):
                continue
            if isinstance(o, dict):
                for kk, vv in o.items():
                    if names.match(str(kk)):
                        v.append(f"{path}[{kk!r}]: a state-index key")
                    stack.append((vv, f"{path}[{kk!r}]", d + 1))
                continue
            if isinstance(o, (list, tuple, set, frozenset)):
                for i, vv in enumerate(o):
                    stack.append((vv, f"{path}[{i}]", d + 1))
                continue
            attrs = getattr(o, "__dict__", None)
            if isinstance(attrs, dict):
                for kk, vv in attrs.items():
                    if names.match(str(kk)):
                        v.append(f"{path}.{kk}: a state-index attribute")
                    stack.append((vv, f"{path}.{kk}", d + 1))
    return {"ok": not v, "violations": v, "n_requests": client.n, "state": int(k)}


def _type_name(o) -> str:
    t = type(o)
    return f"{t.__module__}.{t.__qualname__}"


def walk(root, label: str, allowed_callables: tuple, out: list, max_depth: int = 8) -> dict:
    """Walk ``root``'s attributes and containers; append violations to ``out``; return {type name: count}."""
    seen: set = set()
    counts: dict = {}
    stack = [(root, label, 0)]
    while stack:
        o, path, d = stack.pop()
        if id(o) in seen or d > max_depth:
            continue
        seen.add(id(o))
        tn = _type_name(o)
        counts[tn] = counts.get(tn, 0) + 1
        mod = type(o).__module__ or ""
        if mod.startswith(FORBIDDEN_MODULE_PREFIXES):
            out.append(f"{path}: forbidden object {tn}")
            continue
        if o is None or isinstance(o, (bool, int, float, complex, str, bytes, np.ndarray, np.generic)):
            continue
        if isinstance(o, (_types.FunctionType, _types.BuiltinFunctionType, _types.MethodType, type)):
            if not any(o is c for c in allowed_callables) and not isinstance(o, type):
                out.append(f"{path}: undeclared callable {getattr(o, '__qualname__', tn)}")
            continue
        if isinstance(o, (_types.GeneratorType, _types.ModuleType)):
            if isinstance(o, _types.ModuleType):
                out.append(f"{path}: a module object {o.__name__}")
            continue
        if isinstance(o, dict):
            for k, v in o.items():
                ks = str(k)
                if ks in TRUTH_DICT_KEYS or TRUTH_KEY.match(ks) or "_to_robot0" in ks:
                    out.append(f"{path}[{ks!r}]: truth-side key")
                stack.append((v, f"{path}[{ks!r}]", d + 1))
            continue
        if isinstance(o, (list, tuple, set, frozenset)):
            for i, v in enumerate(o):
                stack.append((v, f"{path}[{i}]", d + 1))
            continue
        attrs = getattr(o, "__dict__", None)
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                stack.append((v, f"{path}.{k}", d + 1))
        slots = getattr(type(o), "__slots__", ())
        for k in slots if isinstance(slots, (list, tuple)) else ():
            if hasattr(o, k):
                stack.append((getattr(o, k), f"{path}.{k}", d + 1))
    return counts


def audit(tool, lib, cam_tool, client, instruction: str, policy_root, agent=None, spec=None, scene=None,
          tool_obs_keys=(), agent_obs_keys=()) -> dict:
    """The runtime check of one episode's policy inputs (see the module doc)."""
    v: list = []
    wl = set(T.PROPRIO_WHITELIST)
    if not set(tool_obs_keys) <= wl:
        v.append(f"tool saw observation keys outside the whitelist: {sorted(set(tool_obs_keys) - wl)}")
    if not set(agent_obs_keys) <= wl:
        v.append(f"agent saw observation keys outside the whitelist: {sorted(set(agent_obs_keys) - wl)}")
    attrs = set(vars(tool))
    if not attrs <= TOOL_ATTRS:
        v.append(f"tool attributes outside the declared set: {sorted(attrs - TOOL_ATTRS)}")
    if tool.lib is not lib:
        v.append("tool.lib is not the episode's library")
    if tool.cam is not cam_tool or getattr(cam_tool, "__qualname__", "") != "make_camera_tool.<locals>.look" or \
            getattr(cam_tool, "__module__", "") != "rowant_lab.libero_env.m2_camera":
        v.append("tool.cam is not the camera tool closure")
    if tool.client is not client:
        v.append("tool.client is not the audited bridge client")
    if not TOKEN.match(str(tool.episode)) and not getattr(client, "legacy_ok", False):
        v.append("tool.episode is not an opaque token")
    if tool.obs is not None and not set(tool.obs) <= wl:
        v.append(f"tool.obs keys {sorted(set(tool.obs) - wl)}")
    from pathlib import Path

    if Path(lib.root) != Path(policy_root).resolve():
        v.append(f"library root {lib.root} is not {policy_root}")
    # the tool and the library: walked (the camera closure is the one declared callable and is not entered;
    # the client's socket is the declared transport)
    counts = {}
    counts["tool"] = walk(tool, "tool", (cam_tool,), v)
    agent_rec = None
    if agent is not None:
        if agent.lib is not lib:
            v.append("agent.lib is not the episode's library")
        if agent.oracle is not tool:
            v.append("agent.oracle is not the perception tool")
        if set(spec) != SPEC_KEYS:
            v.append(f"spec keys {sorted(spec)}")
        if spec.get("instruction") != instruction:
            v.append("spec instruction differs from the env's language instruction")
        if set(scene) != SCENE_KEYS:
            v.append(f"scene keys {sorted(scene)}")
        for iid, inst in scene.get("instances", {}).items():
            if not PERCEPT_ID.match(str(iid)) or set(inst) != INSTANCE_KEYS:
                v.append(f"scene instance {iid!r} keys {sorted(inst)}")
        for sid, s in scene.get("sites", {}).items():
            if set(s) != SITE_KEYS or not PERCEPT_ID.match(str(s.get("parent_instance"))):
                v.append(f"scene site {sid!r}")
        if scene.get("arena") != lib.workcell["arena"] or scene.get("robot_mount") != lib.workcell["robot_mount"]:
            v.append("scene arena or robot mount is not the workcell's")
        if agent.pobs is not None and not set(agent.pobs) <= wl:
            v.append(f"agent.pobs keys {sorted(set(agent.pobs) - wl)}")
        counts["agent"] = walk(agent, "agent", (cam_tool,), v)
        agent_rec = {"spec_keys": sorted(spec), "scene_keys": sorted(scene), "n_instances": len(scene["instances"]),
                     "cap": int(agent.cap), "params": [type(agent.ap).__name__, type(agent.pp).__name__,
                                                       type(agent.fp).__name__]}
    return {"ok": not v, "violations": v[:50], "n_violations": len(v),
            "tool_obs_keys": sorted(tool_obs_keys), "agent_obs_keys": sorted(agent_obs_keys),
            "instruction": instruction, "workcell_suite": lib.workcell.get("suite"),
            "library_root": str(lib.root), "library_manifest_sha256": lib.manifest_sha256,
            "library_opened": [str(x) for x in lib.opened],
            "tool_closure": f"{cam_tool.__module__}.{cam_tool.__qualname__}",
            "tool_attrs": {k: _type_name(val) for k, val in sorted(vars(tool).items())},
            "agent": agent_rec, "walk_types": {k: sorted(c) for k, c in counts.items()},
            "server_requests": client.record() if hasattr(client, "record") else None}
