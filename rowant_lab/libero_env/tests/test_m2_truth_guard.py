"""M2 truth guard (plan precondition P3): M1's AST test extended to the M2 policy, camera tool and server.

M2 (perceived poses, grounded instruction): perception in the loop. M1's own guard keeps running unchanged over
M1's files (``test_m1_truth_guard.py``). Hardened in the m2f:guard round (verifier blocker B2, minors M1 and M9):

* **policy side**: ``m2_ground.py``, ``m2_library.py``, ``m2_kin.py``, ``m2_paths.py`` and the CLIENT half of
  ``m2_bridge.py`` (the module without ``serve``); the earlier robot's agent, ``m2_agent.py``, is not part of this
  release. Imports only from an allowlist (per file: ``pathlib``/``hashlib`` for the library, ``pathlib`` and the
  one data folder of ``rowant_lab.settings`` for the paths, ``struct``/``socket`` for the bridge); no name or
  attribute that reaches the simulator, the success flag, the evaluator, a render or segmentation, and no reflection
  (``getattr``,
  ``setattr``, ``vars``, ``globals``, ``__import__``, ``__closure__``, ``cell_contents``, ``__dict__``,
  ``__class__``, ``__subclasses__``, ...); no string naming an observation truth key, the BDDL, the scenes folder,
  the init states or a run folder -- including strings BUILT by concatenation, f-strings, ``join``, ``format``,
  ``%`` or slicing of a literal, and suffix fragments of truth keys (``'_pos'``); no string-built subscript key in a
  read; an observation (``obs`` / ``pobs``) may only be read by a literal whitelisted key, never iterated, copied or
  passed on; only ``m2_library.py`` opens files (no ``open``, ``read_text``, ``np.load``, ``np.fromfile``,
  ``np.loadtxt``, ``np.genfromtxt``, ``np.memmap``, globbing or listing anywhere else).
* **camera tool** (``m2_camera.py``): alias-proof: the only name bound to a simulator object is ``s = env_.sim``,
  and every chain from ``s`` is one of the declared leaves in its declared use (a call, a subscript or a float); no
  simulator object is passed, returned, stored or re-bound; ``render`` is called with no segmentation argument.
* **perception side** (``m2_perceive.py``, ``m2_onboard.py``, the SERVER half of ``m2_bridge.py``): imports from its
  own allowlist (never libero, robosuite, mujoco, torch, nor the runner, evaluator or truth-side path modules); every
  ``MP.<name>`` exists in the policy-safe ``m2_paths`` (so no run folder or truth location can be named through
  it); no string naming a run folder, a truth log, the scenes or the init states; no image write.
* **planted leaks** are caught: the verifier's nine policy-side forms and its server-side truth-log read, plus
  further forms (the guard's own sanity test).

It never imports LIBERO, MuJoCo or torch.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent
POLICY_FILES = ("m2_ground.py", "m2_library.py", "m2_kin.py", "m2_paths.py", "m2_bridge.py")
PERCEPTION_FILES = ("m2_perceive.py", "m2_onboard.py", "m2_bridge.py", "m2_segment.py",
                    "lfb_episode.py",  # W2-LEARN-EP: in-episode learning, server side
                    "arm_camera.py")  # item 2: the camera from the robot's own arm, server side
SPLIT_FILES = {"m2_bridge.py": "serve"}  # policy side = the module without this function; perception side = it
POLICY_IMPORTS = {
    "__future__", "math", "json", "re", "dataclasses", "numpy", "time",
    "rowant_lab.libero_env.grasp",
    "rowant_lab.libero_env.m2_ground", "rowant_lab.libero_env.m2_library", "rowant_lab.libero_env.m2_bridge",
    "rowant_lab.libero_env.m2_kin", "rowant_lab.libero_env.m2_paths",
}
# m2_paths.py takes the one data folder from ``rowant_lab.settings`` (standard library only: it names that folder and
# nothing else)
POLICY_IMPORTS_PER_FILE = {"m2_library.py": {"pathlib", "hashlib"}, "m2_paths.py": {"pathlib", "rowant_lab.settings"},
                           "m2_bridge.py": {"struct", "socket"}}
POLICY_FORBIDDEN_MODULES = {"robosuite", "libero", "mujoco", "torch", "gym", "glfw", "cv2", "os", "sys", "io",
                            "importlib", "inspect", "gc", "ctypes", "subprocess", "pickle", "marshal", "shelve",
                            "builtins", "operator", "shutil", "glob", "runpy", "code", "codecs", "base64",
                            "rowant_lab.libero_env.tools", "rowant_lab.libero_env.run_suite",
                            "rowant_lab.libero_env.evaluator", "rowant_lab.libero_env.m2_evaluator",
                            "rowant_lab.libero_env.m2_run", "rowant_lab.libero_env.m2_camera",
                            "rowant_lab.libero_env.m2_perceive", "rowant_lab.libero_env.m2_gate",
                            "rowant_lab.libero_env.init_states", "rowant_lab.libero_env.gate",
                            "rowant_lab.libero_env.m2_episode", "rowant_lab.libero_env.m2_rpaths",
                            "rowant_lab.libero_env.m2_freeze", "rowant_lab.libero_env.m2_audit",
                            "rowant_lab.libero_env.m2_fm7", "rowant_lab.libero_env.m1_paths",
                            "rowant_lab.libero_env.m2_fm7p", "rowant_lab.libero_env.m2_frame_tests",
                            "rowant_lab.libero_env.m2_libv2", "rowant_lab.libero_env.m2_records",
                            "rowant_lab.libero_env.m2_integrity", "rowant_lab.libero_env.m2_evalx",
                            "rowant_lab.libero_env.m2_gatereport", "rowant_lab.libero_env.m2_gatesum"}
NUMPY_FORBIDDEN = {"load", "fromfile", "loadtxt", "genfromtxt", "memmap", "save", "savez", "savez_compressed",
                   "savetxt", "lib", "ctypeslib", "DataSource", "fromregex"}
M1_FORBIDDEN = {"sim", "env", "step", "get_sim_state", "check_success", "_check_success", "_eval_predicate",
                "objects_dict", "fixtures_dict", "obj_body_id", "body_xpos", "body_xquat", "done", "reward",
                "evaluator"}
PLAN_FORBIDDEN = {"segmentation", "camera_segmentations", "render", "read_pixels", "xpos", "xquat", "xmat", "qpos",
                  "qvel", "body_pos", "body_quat", "geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat", "contact",
                  "init_states", "gate_states", "truth_snapshot", "match_percepts", "env_", "truth_log"}
# builtins that reach any object or build any name: forbidden as names (``re.compile`` stays an allowed attribute)
REFLECTION_BUILTINS = {"getattr", "setattr", "delattr", "hasattr", "vars", "globals", "locals", "__import__", "eval",
                       "exec", "compile", "breakpoint", "chr", "open"}
REFLECTION = {"__builtins__", "__builtin__", "builtins", "__closure__",
              "cell_contents", "__globals__", "__code__", "__dict__", "__class__", "__subclasses__", "__bases__",
              "__base__", "__mro__", "__self__", "__func__", "__wrapped__", "__getattribute__", "__getattr__",
              "__reduce__", "__reduce_ex__", "__loader__", "__spec__", "f_back", "f_globals", "f_locals", "f_code",
              "tb_frame", "gi_frame", "gi_code", "cr_frame", "sys", "os", "importlib", "inspect", "gc", "ctypes",
              "attrgetter", "methodcaller", "modules"}
FORBIDDEN_NAMES = M1_FORBIDDEN | PLAN_FORBIDDEN | REFLECTION | REFLECTION_BUILTINS
FORBIDDEN_ATTRS = M1_FORBIDDEN | PLAN_FORBIDDEN | REFLECTION
TRUTH_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]*_\d+_(pos|quat)$")
TRUTH_FRAGMENT = re.compile(r"(^_\d*_?(pos|quat|qpos|qvel|xpos|xquat|xmat)$)|(_\d+_(pos|quat))|(-state$)|(^object-)")
FORBIDDEN_STRINGS = ("_to_robot0", "object-state", "segmentation", ":goal", ":init", ":regions", ":objects",
                     ":fixtures", ".bddl", "obj_of_interest", "robot_start", "obs_keys", "scenes", "GATE-READ-LOG",
                     "init-states", "init_states", "libero-m1/gate", "truth", "worker-", "runs/libero-m2/dev",
                     "runs/libero-m2/d1", "runs/libero-m2/gate", "evidence-m1", "evidence-m2", "peeks",
                     "gate-registry")
SUFFIXES = ("_pos", "_quat", "_qpos", "_qvel", "_vel", "_xpos", "_xquat", "_xmat")
ALLOWED_SUFFIX_CONSTANTS = {"robot0_joint_pos", "robot0_joint_vel", "robot0_eef_pos", "robot0_eef_quat",
                            "robot0_gripper_qpos", "robot0_gripper_qvel", "joint_pos", "gripper_qpos", "body_pos",
                            "body_quat"}
OBS_NAMES = {"obs", "pobs"}
OBS_KEYS = {"robot0_joint_pos", "robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel",
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_gripper_qvel"}
FILE_OPENERS = {"open", "read_text", "read_bytes", "write_text", "write_bytes", "load", "fromfile", "loadtxt",
                "genfromtxt", "memmap", "open_memmap", "save", "savez", "savez_compressed", "savetxt", "tofile",
                "imwrite", "imread", "iterdir", "glob", "rglob", "listdir", "scandir", "walk", "stat", "lstat",
                "exists", "is_file", "is_dir", "mkdir", "unlink", "rename", "touch", "readlink", "fdopen",
                "expanduser", "home", "cwd", "fromregex"}
CAMERA_LEAVES = {  # chain from the simulator alias -> the declared use of that leaf
    "s.render": "call", "s.model.camera_name2id": "call", "s.model.cam_fovy": "subscript",
    "s.data.cam_xpos": "subscript", "s.data.cam_xmat": "subscript", "s.model.stat.extent": "float",
    "s.model.vis.map.znear": "float", "s.model.vis.map.zfar": "float"}
CAMERA_WHITELIST = {("sim", "render"), ("model", "camera_name2id"), ("model", "cam_fovy"), ("data", "cam_xpos"),
                    ("data", "cam_xmat"), ("model", "stat"), ("model", "vis")}
PERCEPTION_IMPORTS_FORBIDDEN = {"libero", "robosuite", "mujoco", "torch", "gym", "glfw", "pickle",
                                "rowant_lab.libero_env.m2_rpaths", "rowant_lab.libero_env.m2_run",
                                "rowant_lab.libero_env.m2_episode", "rowant_lab.libero_env.m2_evaluator",
                                "rowant_lab.libero_env.m2_gate", "rowant_lab.libero_env.m2_freeze",
                                "rowant_lab.libero_env.init_states", "rowant_lab.libero_env.gate",
                                "rowant_lab.libero_env.evaluator", "rowant_lab.libero_env.tools",
                                "rowant_lab.libero_env.run_suite", "rowant_lab.libero_env.m1_paths",
                                "rowant_lab.libero_env.m2_camera", "rowant_lab.libero_env.m2_agent",
                                "rowant_lab.libero_env.oracle_agent", "rowant_lab.libero_env.m2_fm7",
                                "rowant_lab.libero_env.m2_fm7p",
                                "rowant_lab.libero_env.m2_audit", "rowant_lab.libero_env.m2_frame_tests",
                                "rowant_lab.libero_env.m2_export", "rowant_lab.libero_env.m2_libv2",
                                "rowant_lab.libero_env.export_geom", "rowant_lab.libero_env.build_records",
                                "rowant_lab.libero_env.smoke", "rowant_lab.libero_env.frame_tests",
                                "rowant_lab.libero_env.freeze", "rowant_lab.bop.inputs",
                                "rowant_lab.libero_env.m2_evalx", "rowant_lab.libero_env.m2_gatereport",
                                "rowant_lab.libero_env.m2_gatesum"}
PERCEPTION_ROWANT_ALLOWED = ("rowant_lab.fastlayer", "rowant_lab.hybrid_frontier.screens",
                             "rowant_lab.hybrid_frontier.e6", "rowant_lab.hybrid_frontier.e9.isolation",
                             "rowant_lab.libero_env.m2_bridge", "rowant_lab.libero_env.m2_library",
                             "rowant_lab.libero_env.m2_kin", "rowant_lab.libero_env.m2_paths",
                             "rowant_lab.libero_env.m2_onboard", "rowant_lab.libero_env.m2_integrity",
                             "rowant_lab.libero_env.m2_records", "rowant_lab.libero_env.m2_segment",
                             "rowant_lab.libero_env.m2_mminus",  # mminus_dev's M- arm (truth-free)
                             # W2-LEARN-EP: in-episode learning and the LF-b steps it reuses (truth-free: frames,
                             # masks and models only; the server's audit hook bounds what they can read at run time)
                             "rowant_lab.libero_env.lfb_episode", "rowant_lab.libero_env.lfb_learn",
                             "rowant_lab.libero_env.lfb_build", "rowant_lab.libero_env.lfb_library",
                             "rowant_lab.libero_env.lfb_paths",
                             # wrist frames without depth (lfb_episode.wrist_depth, 2026-11-17): K0-stereo's matcher,
                             # truth-free by its own import and read rules (test_m3_stereo1_match.py)
                             "rowant_lab.libero_env.m3_stereo1_match",
                             # item 2: the camera from the robot's own arm (truth-free: depth frames, proprioception
                             # and the server's Panda model)
                             "rowant_lab.libero_env.arm_camera")
PERCEPTION_FORBIDDEN_EXACT = {"truth", "truth_log", "goal", "obs_keys", "instances"}
PERCEPTION_FORBIDDEN_STRINGS = ("truth_log", "truth_snapshot", "worker-", "scenes", "GATE-READ-LOG", "init-states",
                                "init_states",
                                "libero-m1", ".bddl", "obj_of_interest", ":goal", ":init", "runs/libero-m2/dev",
                                "runs/libero-m2/d1", "runs/libero-m2/gate", "evidence-m1", "evidence-m2",
                                "_to_robot0", "object-state", "seen_in", "exported_from")
PERCEPTION_SAVES = {"imwrite", "imsave", "save", "savez", "savez_compressed", "savetxt", "tofile"}


# ------------------------------------------------------------------------------------------------ helpers
def _src(name: str) -> str:
    return (PKG / name).read_text()


def _tree(name: str):
    return ast.parse(_src(name), filename=name)


def _split(tree, fn: str, keep_fn: bool):
    """The module without top-level function ``fn`` (``keep_fn`` False) or only that function (True)."""
    body = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == fn) == keep_fn]
    if keep_fn:
        return ast.Module(body=body, type_ignores=[])
    return ast.Module(body=body, type_ignores=[])


def _policy_tree(name: str):
    t = _tree(name)
    return _split(t, SPLIT_FILES[name], False) if name in SPLIT_FILES else t


def _perception_tree(name: str):
    t = _tree(name)
    return _split(t, SPLIT_FILES[name], True) if name in SPLIT_FILES else t


def _parents(tree) -> dict:
    out = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            out[id(ch)] = node
    return out


def _docstrings(tree) -> set:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def _imports(tree) -> list:
    """Every imported module (``from X import y`` of a rowant_lab package gives ``X.y``)."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            out += [f"{mod}.{a.name}" if mod.startswith("rowant_lab") else mod for a in node.names]
    return out


def _is_str_const(n) -> bool:
    return isinstance(n, ast.Constant) and isinstance(n.value, str)


def fold(n):
    """The string a composite string expression builds from literals (None if it is not one): concatenation,
    f-strings (their literal parts), ``sep.join([...])``, ``fmt.format(...)``, ``fmt % (...)``, a sliced literal."""
    if _is_str_const(n):
        return n.value
    if isinstance(n, ast.JoinedStr):
        return "".join(v.value for v in n.values if _is_str_const(v))
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
        a, b = fold(n.left), fold(n.right)
        if a is None and b is None:
            return None
        return (a or "") + (b or "")
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod) and _is_str_const(n.left):
        parts = n.right.elts if isinstance(n.right, ast.Tuple) else [n.right]
        return n.left.value.replace("%s", "").replace("%r", "") + "".join(fold(p) or "" for p in parts)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and _is_str_const(n.func.value):
        if n.func.attr == "join" and n.args and isinstance(n.args[0], (ast.List, ast.Tuple)):
            return n.func.value.value.join(fold(e) or "" for e in n.args[0].elts)
        if n.func.attr == "format":
            return n.func.value.value.replace("{}", "") + "".join(fold(e) or "" for e in n.args)
    if isinstance(n, ast.Subscript) and _is_str_const(n.value):
        lit = all(isinstance(x, (ast.Constant, ast.Slice, ast.UnaryOp, ast.USub, ast.Load))
                  for x in ast.walk(n.slice))
        if lit:  # a literal sliced by literals only: evaluate it exactly
            try:
                return eval(compile(ast.Expression(body=n), "<fold>", "eval"), {"__builtins__": {}})  # noqa: S307
            except Exception:
                pass
        return n.value.value[::-1] + n.value.value
    return None


def _composite(n) -> bool:
    return (isinstance(n, ast.JoinedStr) or (isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Mod))
                                             and fold(n) is not None)
            or (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("join", "format"))
            or (isinstance(n, ast.Subscript) and _is_str_const(n.value)))


def _string_hits(s: str, forbidden=FORBIDDEN_STRINGS, allow=()) -> list:
    out = []
    if TRUTH_KEY.match(s):
        out.append("truth key")
    if TRUTH_FRAGMENT.search(s) and s not in ALLOWED_SUFFIX_CONSTANTS:
        out.append("truth-key fragment")
    if s.endswith(SUFFIXES) and s not in ALLOWED_SUFFIX_CONSTANTS:
        out.append("observation-key suffix")
    out += [f"contains {x!r}" for x in forbidden if x in s and x not in allow]
    return out


# ------------------------------------------------------------------------------------------------ policy checks
def policy_import_violations(name: str, tree=None) -> list:
    tree = tree or _policy_tree(name)
    allowed = POLICY_IMPORTS | POLICY_IMPORTS_PER_FILE.get(name, set())
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in allowed or a.name.split(".")[0] in POLICY_FORBIDDEN_MODULES:
                    hits.append((node.lineno, f"import {a.name}"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                hits.append((node.lineno, "relative import"))
                continue
            for a in node.names:
                if a.name == "*":
                    hits.append((node.lineno, f"from {mod} import *"))
                full = f"{mod}.{a.name}" if mod == "rowant_lab.libero_env" else mod
                if full in POLICY_FORBIDDEN_MODULES or full.split(".")[0] in POLICY_FORBIDDEN_MODULES:
                    hits.append((node.lineno, f"from {mod} import {a.name}"))
                elif full not in allowed:
                    hits.append((node.lineno, f"from {mod} import {a.name} (not on the allowlist)"))
                if mod == "numpy" and a.name in NUMPY_FORBIDDEN:
                    hits.append((node.lineno, f"from numpy import {a.name}"))
    return hits


def _libero_aliases(tree) -> dict:
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("rowant_lab.libero_env"):
            for a in node.names:
                full = f"{node.module}.{a.name}" if node.module == "rowant_lab.libero_env" else None
                if full:
                    out[a.asname or a.name] = full
    return out


def policy_name_violations(name: str, tree=None) -> list:
    tree = tree or _policy_tree(name)
    hits = []
    aliases = _libero_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            if node.attr not in _module_attrs(aliases[node.value.id]):
                hits.append((node.lineno, f"{node.value.id}.{node.attr} (not in {aliases[node.value.id]})"))
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            hits.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            hits.append((node.lineno, node.attr))
        elif isinstance(node, ast.arg) and node.arg in FORBIDDEN_NAMES:
            hits.append((node.lineno, node.arg))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in FORBIDDEN_NAMES:
            hits.append((node.lineno, node.name))
        elif isinstance(node, ast.keyword) and node.arg in FORBIDDEN_NAMES:
            hits.append((node.lineno, node.arg))
        elif isinstance(node, ast.alias) and (node.asname or node.name.split(".")[-1]) in FORBIDDEN_NAMES:
            hits.append((getattr(node, "lineno", 0), node.asname or node.name))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            hits.append((node.lineno, "global/nonlocal"))
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__") \
                and node.attr not in ("__name__", "__init__", "__future__", "__file__"):
            hits.append((node.lineno, node.attr))
    return hits


def policy_obs_violations(name: str, tree=None) -> list:
    """An observation may be read only by a literal whitelisted key: ``obs['robot0_eef_pos']``."""
    tree = tree or _policy_tree(name)
    par = _parents(tree)
    hits = []
    for node in ast.walk(tree):
        is_obs = (isinstance(node, ast.Name) and node.id in OBS_NAMES) or \
                 (isinstance(node, ast.Attribute) and node.attr in OBS_NAMES)
        if not is_obs or isinstance(getattr(node, "ctx", None), ast.Store):
            continue
        p = par.get(id(node))
        if isinstance(p, ast.Subscript) and p.value is node and _is_str_const(p.slice) and p.slice.value in OBS_KEYS:
            continue
        if isinstance(p, ast.Attribute) and p.value is node and isinstance(node, ast.Name):
            pass  # a Name 'obs' with an attribute taken: still a use of the whole observation
        hits.append((node.lineno, "observation used other than by a literal whitelisted key"))
    return hits


def policy_string_violations(name: str, tree=None) -> list:
    tree = tree or _policy_tree(name)
    docs = _docstrings(tree)
    par = _parents(tree)
    # m2_library.py holds "scenes" in its own refusal rule (it is the file that refuses that path), and the names
    # of the per-task fields it refuses (its PER_TASK_KEYS set and its TASK_REF / INSTANCE_NAME patterns)
    allow = ("scenes",) if name == "m2_library.py" else ()
    skip = set()
    if name == "m2_library.py":
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                                                    ("PER_TASK_KEYS", "TASK_REF", "INSTANCE_NAME")
                                                    for t in node.targets):
                skip |= {id(c) for c in ast.walk(node.value) if _is_str_const(c)}
    hits = []
    for node in ast.walk(tree):
        if _is_str_const(node) and id(node) not in docs and id(node) not in skip:
            p = par.get(id(node))
            in_fstring = isinstance(p, ast.JoinedStr)
            s = node.value
            for why in _string_hits(s, allow=allow):
                hits.append((node.lineno, s, why))
            del in_fstring
        if _composite(node):
            s = fold(node)
            if s is None:
                continue
            for why in _string_hits(s, allow=allow):
                hits.append((node.lineno, s, f"built string: {why}"))
            if s in FORBIDDEN_NAMES:
                hits.append((node.lineno, s, "built string names a forbidden attribute"))
            if isinstance(node, ast.Subscript):
                hits.append((node.lineno, s, "a sliced string literal"))
        if name in BUILT_KEYS_EXEMPT:
            continue
        if isinstance(node, ast.Subscript) and not isinstance(node.ctx, ast.Store) and _unsafe_built(node.slice):
            hits.append((node.lineno, fold(node.slice), "string-built subscript key in a read"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                node.func.attr in ("get", "pop", "setdefault", "__getitem__") and node.args and \
                _unsafe_built(node.args[0]):
            hits.append((node.lineno, fold(node.args[0]), "string-built key lookup"))
        if isinstance(node, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in node.ops) and \
                _unsafe_built(node.left):
            hits.append((node.lineno, fold(node.left), "string-built membership test"))
    return hits


# a key built in a read may carry only these literal fragments: the Panda mesh arrays (``f"{key}__V"``) and the
# synthetic body names (``f"{id}_{rel}"``). ``m2_library.py``, the file reader that never sees an observation, builds
# its manifest keys from paths and is exempt from the built-key rules (not from the string-content rules).
SAFE_KEY_FRAGMENTS = {"_", "__V", "__F", "__N", "__UV"}
BUILT_KEYS_EXEMPT = {"m2_library.py"}


def _fragments(n) -> list:
    if _is_str_const(n):
        return [n.value]
    if isinstance(n, ast.JoinedStr):
        return [v.value for v in n.values if _is_str_const(v)]
    if isinstance(n, ast.BinOp):
        return _fragments(n.left) + _fragments(n.right)
    return [c.value for c in ast.walk(n) if _is_str_const(c)]


def _unsafe_built(n) -> bool:
    return _composite(n) and not set(_fragments(n)) <= SAFE_KEY_FRAGMENTS


def policy_file_violations(name: str, tree=None) -> list:
    if name == "m2_library.py":
        return []
    tree = tree or _policy_tree(name)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FILE_OPENERS:
            hits.append((node.lineno, node.attr))
        elif isinstance(node, ast.Name) and node.id in FILE_OPENERS:
            hits.append((node.lineno, node.id))
    return hits


def policy_violations(name: str, tree=None) -> dict:
    tree = tree or _policy_tree(name)
    out = {"imports": policy_import_violations(name, tree), "names": policy_name_violations(name, tree),
           "obs": policy_obs_violations(name, tree), "strings": policy_string_violations(name, tree),
           "file_access": policy_file_violations(name, tree)}
    return {k: v for k, v in out.items() if v}


# ------------------------------------------------------------------------------------------------ policy tests
# (the four names below are the ones the verifier's planted-leak drivers call; each runs its own rule set and the
# reflection, observation and built-string rules are folded into them so a driver that calls only these four sees
# every rule)
@pytest.mark.parametrize("name", POLICY_FILES)
def test_policy_imports(name):
    hits = policy_import_violations(name)
    assert not hits, f"{name}: imports {hits}"


@pytest.mark.parametrize("name", POLICY_FILES)
def test_policy_no_forbidden_names(name):
    hits = policy_name_violations(name) + policy_obs_violations(name)
    assert not hits, f"{name}: forbidden names, reflection or observation use {hits}"


@pytest.mark.parametrize("name", POLICY_FILES)
def test_policy_no_truth_strings(name):
    hits = policy_string_violations(name)
    assert not hits, f"{name}: truth strings {hits}"


def test_only_the_library_opens_files():
    for name in POLICY_FILES:
        hits = policy_file_violations(name)
        assert not hits, f"{name}: file access {hits}"


def test_policy_files_are_every_module_the_policy_imports():
    """Every libero_env module a policy file may import is itself scanned (M1's under M1's own guard)."""
    m1 = {"grasp.py"}
    for m in POLICY_IMPORTS:
        if m.startswith("rowant_lab.libero_env."):
            f = m.rsplit(".", 1)[1] + ".py"
            assert f in POLICY_FILES or f in m1, f


def test_library_refuses_paths_outside_the_m2_library(tmp_path):
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.libero_env import m2_paths as MP2

    v = MP2.POLICY_LIB_VERSION  # m2r3:fix-r1 (A65): the library version in use (policy-v3)
    lib = tmp_path / v
    lib.mkdir()
    (lib / "manifest.json").write_text('{"version": "%s", "files": {}, "in_place": {}}' % v)
    r = LB.LibraryReader(lib, tmp_path / "m1")
    for bad in (Path("/etc/hosts"), lib / "scenes" / "x.json", tmp_path / f"{v}-evil" / "x.json",
                tmp_path / "m1" / "scenes" / "libero_object" / "x.json", tmp_path / "m1" / "types" / "plate.json"):
        with pytest.raises(LB.LibraryRefused):
            r.read(bad)
    (lib / "x.json").write_text("{}")
    with pytest.raises(LB.LibraryRefused):  # inside the root but not in the manifest
        r.read(lib / "x.json")


# ------------------------------------------------------------------------------------------------ camera tool
def camera_violations(src: str | None = None) -> list:
    """Alias-proof: taint from ``make_camera_tool``'s parameter; the only binding is ``s = <param>.sim``."""
    tree = ast.parse(src if src is not None else _src("m2_camera.py"))
    par = _parents(tree)
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "make_camera_tool"), None)
    if fn is None:
        return [(0, "no make_camera_tool")]
    params = {a.arg for a in fn.args.args}
    hits = []
    sim_alias = set()
    for node in ast.walk(fn):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            val = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(val, ast.Attribute) and isinstance(val.value, ast.Name) and val.value.id in params \
                    and val.attr == "sim" and len(targets) == 1 and isinstance(targets[0], ast.Name) \
                    and isinstance(node, ast.Assign):
                sim_alias.add(targets[0].id)
    if sim_alias != {"s"}:
        hits.append((fn.lineno, f"the simulator alias must be exactly 's = env_.sim' (found {sorted(sim_alias)})"))
    tainted = params | sim_alias
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Name) and node.id in tainted and isinstance(node.ctx, ast.Load)):
            if isinstance(node, ast.Name) and node.id in tainted and not isinstance(node.ctx, ast.Load):
                p = par.get(id(node))
                ok = node.id == "s" and isinstance(p, ast.Assign) and isinstance(p.value, ast.Attribute) \
                    and p.value.attr == "sim"
                if not ok:
                    hits.append((node.lineno, f"re-binding of {node.id}"))
            continue
        chain, top = [node.id], node
        while isinstance(par.get(id(top)), ast.Attribute) and par[id(top)].value is top:
            top = par[id(top)]
            chain.append(top.attr)
        dotted = ".".join(chain)
        use = par.get(id(top))
        if node.id in params:
            ok = dotted.endswith(".sim") and len(chain) == 2 and isinstance(use, ast.Assign) \
                and len(use.targets) == 1 and isinstance(use.targets[0], ast.Name) and use.targets[0].id == "s"
            if not ok:
                hits.append((node.lineno, f"{dotted}: the camera parameter may only give 's = {node.id}.sim'"))
            continue
        want = CAMERA_LEAVES.get(dotted)
        if want is None:
            hits.append((node.lineno, f"{dotted}: not a declared camera leaf"))
            continue
        if want == "call":
            ok = isinstance(use, ast.Call) and use.func is top
        elif want == "subscript":
            ok = isinstance(use, ast.Subscript) and use.value is top
        else:
            ok = isinstance(use, ast.Call) and isinstance(use.func, ast.Name) and use.func.id == "float" \
                and use.args and use.args[0] is top
        if not ok:
            hits.append((node.lineno, f"{dotted}: used other than as a {want}"))
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "render":
            kw = {k.arg for k in node.keywords}
            if kw & {"segmentation", "mode"} or None in kw:
                hits.append((node.lineno, f"render with {kw}"))
            if not ({"camera_name", "depth"} <= kw):
                hits.append((node.lineno, "render without camera_name and depth"))
            depth = next((k.value for k in node.keywords if k.arg == "depth"), None)
            if not (isinstance(depth, ast.Constant) and depth.value is True):
                hits.append((node.lineno, "render depth must be the literal True"))
    return hits


def test_camera_tool_attribute_whitelist():
    hits = camera_violations()
    assert not hits, f"m2_camera: {hits}"


def test_camera_tool_renders_no_segmentation():
    """No segmentation outside the module's own docstrings (which say it is never requested)."""
    tree = _tree("m2_camera.py")
    docs = _docstrings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            assert "segmentation" not in node.value.lower(), f"m2_camera: line {node.lineno}"
        if isinstance(node, (ast.Name, ast.Attribute)):
            label = node.id if isinstance(node, ast.Name) else node.attr
            assert "segmentation" not in label.lower(), f"m2_camera: {label}"


@pytest.mark.parametrize("form", [
    "    d = s.data\n    x = d.xpos\n",
    "    x = s.data.xpos[0]\n",
    "    m = s.model\n",
    "    y = s.data.cam_xpos\n",
    "    e = env_\n",
    "    t = getattr(s, 'data')\n",
    "    f = s.render\n",
    "    q = float(s.model.stat.meaninertia)\n",
    "    r = s.render(8, 8, camera_name='agentview', depth=True, segmentation=True)\n",
    "    out = [s]\n",
    "    s = env_.sim.model\n",
])
def test_camera_guard_catches_aliases(form):
    src = _src("m2_camera.py")
    planted = src.replace("        s = env_.sim\n", "        s = env_.sim\n" + "    " + form.replace("\n    ", "\n        "), 1)
    assert planted != src
    assert camera_violations(planted), form


# ------------------------------------------------------------------------------------------------ perception side
# m2_perceive.py still names this module behind a switched-off option, and the module is not part of this release.
# Its import stays under the allow-list above; its names cannot be listed, and nothing can be read through a module
# that is absent. The exception holds only while the file is absent: once it exists its names are resolved as before.
NOT_IN_RELEASE = {"rowant_lab.libero_env.m2_mminus"}


def _absent(modname: str) -> bool:
    return modname in NOT_IN_RELEASE and not (PKG / (modname.rsplit(".", 1)[1] + ".py")).exists()


def _module_attrs(modname: str) -> set:
    import importlib

    return set(dir(importlib.import_module(modname)))


def perception_violations(name: str, tree=None) -> dict:
    tree = tree or _perception_tree(name)
    docs = _docstrings(tree)
    out = {"imports": [], "refs": [], "strings": [], "saves": []}
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                m = a.name
                if m.split(".")[0] in PERCEPTION_IMPORTS_FORBIDDEN or m in PERCEPTION_IMPORTS_FORBIDDEN:
                    out["imports"].append((node.lineno, m))
                if m.startswith("rowant_lab") and not m.startswith(PERCEPTION_ROWANT_ALLOWED):
                    out["imports"].append((node.lineno, m))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for a in node.names:
                full = f"{mod}.{a.name}" if mod.startswith("rowant_lab") else mod
                if full.split(".")[0] in PERCEPTION_IMPORTS_FORBIDDEN or full in PERCEPTION_IMPORTS_FORBIDDEN \
                        or mod in PERCEPTION_IMPORTS_FORBIDDEN:
                    out["imports"].append((node.lineno, full))
                elif full.startswith("rowant_lab") and not full.startswith(PERCEPTION_ROWANT_ALLOWED):
                    out["imports"].append((node.lineno, full))
                if full.startswith("rowant_lab.libero_env."):
                    aliases[a.asname or a.name] = full
    # every attribute taken from an imported libero_env module must exist in it (MP.DEV_RUNS does not)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases \
                and not _absent(aliases[node.value.id]):
            if node.attr not in _module_attrs(aliases[node.value.id]):
                out["refs"].append((node.lineno, f"{node.value.id}.{node.attr}"))
        if _is_str_const(node) and id(node) not in docs:
            out["strings"] += [(node.lineno, node.value, f"contains {x!r}") for x in PERCEPTION_FORBIDDEN_STRINGS
                               if x in node.value]
            if node.value in PERCEPTION_FORBIDDEN_EXACT:
                out["strings"].append((node.lineno, node.value, "a truth-side key"))
        if _composite(node):
            s = fold(node) or ""
            out["strings"] += [(node.lineno, s, f"built: contains {x!r}") for x in PERCEPTION_FORBIDDEN_STRINGS
                               if x in s]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("imwrite", "imsave"):
                out["saves"].append((node.lineno, node.func.attr))
            if name not in ("m2_onboard.py", "lfb_episode.py") and node.func.attr in PERCEPTION_SAVES:  # models
                out["saves"].append((node.lineno, node.func.attr))
    return {k: v for k, v in out.items() if v}


@pytest.mark.parametrize("name", PERCEPTION_FILES)
def test_perception_imports(name):
    """Imports from the allowlist, and every module reference resolves in the policy-safe module it names."""
    v = perception_violations(name)
    hits = v.get("imports", []) + v.get("refs", [])
    assert not hits, f"{name}: imports or truth-side references {hits}"


@pytest.mark.parametrize("name", PERCEPTION_FILES)
def test_perception_writes_no_image(name):
    v = perception_violations(name)
    assert not v.get("saves"), f"{name}: image or array write {v['saves']}"


@pytest.mark.parametrize("name", PERCEPTION_FILES)
def test_perception_no_truth_strings(name):
    v = perception_violations(name)
    assert not v.get("strings"), f"{name}: truth-side strings {v['strings']}"


def test_server_audit_hook_is_installed():
    from rowant_lab.libero_env import m2_perceive as PR

    src = _src("m2_perceive.py")
    fn = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "install_audit_hook"]
    assert fn, "the server must install an audit hook"
    assert "socket.getaddrinfo" in src and "read refused" in src
    assert PR.TRUTH_FREE is True


# ------------------------------------------------------------------------------------------------ planted leaks
PLANT_IN = "m2_ground.py"  # the policy file the forms are appended to (as run: m2_agent.py, not in this release)
VERIFIER_FORMS = {  # the verifier's nine (scratchpad libero-m2/verify/driver.py)
    "control_plain_sim": "\ndef _leak(tool):\n    return tool.cam.env_.sim.data.xpos\n",
    "control_truth_key": "\ndef _leak(o):\n    return o['ketchup_1_pos']\n",
    "control_import_mujoco": "\nimport mujoco\n",
    "raw_obs_key_scan": "\ndef _leak(self):\n    return {k: v for k, v in self.obs.items() if k.endswith('_pos') and "
                        "not k.startswith('robot0')}\n",
    "closure_getattr": "\ndef _leak(self):\n    e = self.cam.__closure__[0].cell_contents\n    s = getattr(e, 'si' + 'm')"
                       "\n    return getattr(getattr(s, 'da' + 'ta'), 'body_x' + 'pos')\n",
    "dunder_import": "\ndef _leak():\n    return __import__('mu' + 'joco')\n",
    "np_file_read": "\ndef _leak():\n    return np.loadtxt(str(MP.DEV_RUNS / 'd04-frozen' / 'worker-0.jsonl'), dtype=str, "
                    "delimiter='\\x01')\n",
    "string_concat_truth_key": "\ndef _leak(o):\n    return o['ketchup' + '_1_pos']\n",
    "obs_object_state": "\ndef _leak(o):\n    return o['object' + '-state']\n",
}
MORE_FORMS = {
    "fstring_key": "\ndef _leak(o, n):\n    return o[f'{n}_1_pos']\n",
    "join_key": "\ndef _leak(o):\n    return o[''.join(['obj', 'ect-state'])]\n",
    "format_key": "\ndef _leak(o):\n    return o['{}{}'.format('object', '-state')]\n",
    "reversed_literal": "\ndef _leak(o):\n    return o['etats-tcejbo'[::-1]]\n",
    "obs_copy": "\ndef _leak(self):\n    return dict(self.obs)\n",
    "obs_alias": "\ndef _leak(self):\n    o = self.obs\n    return o\n",
    "obs_get": "\ndef _leak(self, k):\n    return self.obs.get(k)\n",
    "vars_walk": "\ndef _leak(self):\n    return vars(self.cam)\n",
    "globals_walk": "\ndef _leak():\n    return globals()\n",
    "subclasses": "\ndef _leak():\n    return ().__class__.__base__.__subclasses__()\n",
    "dict_attr": "\ndef _leak(self):\n    return self.cam.__dict__\n",
    "from_numpy_load": "\nfrom numpy import load as _l\n",
    "import_os": "\nimport os\n",
    "import_sys_modules": "\nimport sys\n",
    "path_read_text": "\ndef _leak(p):\n    return p.read_text()\n",
    "np_fromfile": "\ndef _leak(p):\n    return np.fromfile(p)\n",
    "np_genfromtxt": "\ndef _leak(p):\n    return np.genfromtxt(p)\n",
    "glob_runs": "\ndef _leak(p):\n    return list(p.glob('*'))\n",
    "chr_build": "\ndef _leak(o):\n    return o[chr(111) + 'bject-state']\n",
    "pct_key": "\ndef _leak(o, n):\n    return o['%s_1_pos' % n]\n",
    "suffix_fragment": "\ndef _leak(o):\n    return [k for k in sorted(o) if k[-4:] == '_pos']\n",
    "env_param": "\ndef _leak(env_):\n    return 0\n",
    "truth_log_string": "\nX = 'truth_log'\n",
}


def _planted(form: str, file: str = PLANT_IN):
    return ast.parse(_src(file) + form)


@pytest.mark.parametrize("label", sorted(VERIFIER_FORMS) + sorted(MORE_FORMS))
def test_guard_catches_planted_leaks(label):
    form = {**VERIFIER_FORMS, **MORE_FORMS}[label]
    assert not policy_violations(PLANT_IN), f"the clean {PLANT_IN} must pass"
    v = policy_violations(PLANT_IN, _planted(form))
    assert v, f"{label}: not caught"


def test_guard_catches_the_verifiers_server_truth_read():
    """The verifier's driver2 form: a truth-log read planted in the server copy."""
    leak = ("\ndef _truth_peek(run):\n    rows = [json.loads(l) for l in open(MP.DEV_RUNS / run / 'worker-0.jsonl')]\n"
            "    return [r['truth_log'][0]['truth'] for r in rows]\n")
    tree = ast.parse(_src("m2_perceive.py") + leak)
    v = perception_violations("m2_perceive.py", tree)
    assert v.get("refs") and v.get("strings"), v
    assert not perception_violations("m2_perceive.py"), "the clean server must pass"


def test_guard_catches_planted_perception_imports():
    for form in ("\nimport mujoco\n", "\nfrom rowant_lab.libero_env import m2_rpaths as R\n",
                 "\nfrom rowant_lab.libero_env import m2_evaluator\n", "\nimport robosuite\n"):
        v = perception_violations("m2_perceive.py", ast.parse(_src("m2_perceive.py") + form))
        assert v.get("imports"), form


def test_m2_modules_carry_the_label():
    from rowant_lab.libero_env import m2_paths as MP

    for name in POLICY_FILES + PERCEPTION_FILES + ("m2_camera.py", "m2_episode.py", "m2_run.py", "m2_audit.py",
                                                   "m2_libv2.py", "m2_rpaths.py"):
        assert MP.LABEL.split(":")[0] in (PKG / name).read_text(), name
