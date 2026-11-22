"""M1 truth guard (plan precondition P3), in the pattern of H0's tests/test_harness_truth_guard.py.

ORACLE (true poses): diagnostic of the hands. Of M1's policy modules this release carries grasp.py (the oracle agent
and its follower are not part of it). It may:

* import only from an allowlist (no robosuite, libero, mujoco, torch, gym, glfw; nothing of the runner or evaluator);
* use no name or attribute that reaches the simulator, the success flag or the evaluator (``sim``, ``env``, ``step``,
  ``get_sim_state``, ``check_success``, ``_check_success``, ``_eval_predicate``, ``objects_dict``, ``fixtures_dict``,
  ``obj_body_id``, ``body_xpos``, ``body_xquat``, ``done``, ``reward``, ``evaluator``);
* hold no string constant naming an observation truth key (``<instance>_pos``/``_quat`` of a task object, anything
  with ``_to_robot0_eef``, ``object-state``).

Docstrings and comments are excluded (comments are not in the AST; docstrings are skipped). A runtime check: the
runner's whitelist filter keeps only proprioception keys of a DEV observation's recorded key list.

Never imports LIBERO, MuJoCo or torch.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from rowant_lab.libero_env import m1_paths as MP
from rowant_lab.libero_env import tools as T

PKG = Path(__file__).resolve().parent.parent
POLICY_FILES = ("grasp.py",)
ALLOWED_IMPORTS = {"__future__", "math", "json", "dataclasses", "pathlib", "numpy", "rowant_lab.libero_env.grasp"}
FORBIDDEN_NAMES = {"sim", "env", "step", "get_sim_state", "check_success", "_check_success", "_eval_predicate",
                   "objects_dict", "fixtures_dict", "obj_body_id", "body_xpos", "body_xquat", "done", "reward",
                   "evaluator"}
TRUTH_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]*_\d+_(pos|quat)$")


def _docstring_nodes(tree) -> set[int]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def _tree(name: str):
    return ast.parse((PKG / name).read_text(), filename=name)


@pytest.mark.parametrize("name", POLICY_FILES)
def test_imports_allowlisted(name):
    tree = _tree(name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name in ALLOWED_IMPORTS, f"{name}: import {a.name}"
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{name}: relative import"
            if node.module == "rowant_lab.libero_env":
                for a in node.names:
                    assert f"rowant_lab.libero_env.{a.name}" in ALLOWED_IMPORTS, f"{name}: from ... import {a.name}"
            else:
                assert node.module in ALLOWED_IMPORTS, f"{name}: from {node.module} import ..."


@pytest.mark.parametrize("name", POLICY_FILES)
def test_no_forbidden_names(name):
    tree = _tree(name)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            hits.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            hits.append((node.lineno, node.attr))
        elif isinstance(node, ast.arg) and node.arg in FORBIDDEN_NAMES:
            hits.append((node.lineno, node.arg))
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in FORBIDDEN_NAMES:
            hits.append((node.lineno, node.name))
        elif isinstance(node, ast.keyword) and node.arg in FORBIDDEN_NAMES:
            hits.append((node.lineno, node.arg))
    assert not hits, f"{name}: forbidden names {hits}"


@pytest.mark.parametrize("name", POLICY_FILES)
def test_no_truth_key_strings(name):
    tree = _tree(name)
    docs = _docstring_nodes(tree)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            s = node.value
            if TRUTH_KEY.match(s) or "_to_robot0_eef" in s or s == "object-state":
                hits.append((node.lineno, s))
    assert not hits, f"{name}: truth-key strings {hits}"


def test_guard_catches_planted_leaks(tmp_path):
    """The guard's own sanity check: planted leaks are caught."""
    src = ('import mujoco\nfrom robosuite import x\n'
           'def f(env):\n    a = env.sim\n    b = obs["alphabet_soup_1_pos"]\n    return done\n')
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
        n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    assert names & FORBIDDEN_NAMES >= {"env", "sim", "done"}
    consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert any(TRUTH_KEY.match(c) for c in consts)
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "mujoco" not in ALLOWED_IMPORTS and mods == {"mujoco"}


def test_whitelist_on_recorded_dev_keys():
    """Runtime check on a DEV observation's key list (recorded by the exporter at DEV state 0 of each task)."""
    scenes = sorted((MP.LIBRARY_DIR / "scenes").glob("*/*.json")) if MP.LIBRARY_DIR.exists() else []
    if not scenes:
        pytest.skip("library not exported on this machine")
    for p in scenes:
        keys = json.loads(p.read_text())["obs_keys"]
        fake = {k: [0.0] for k in keys}
        out = T.filter_obs(fake)
        assert set(out) <= set(T.PROPRIO_WHITELIST), p.name
        assert {"robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"} <= set(out), p.name
        dropped = T.dropped_keys(keys)
        assert "object-state" in dropped and any(k.endswith("_pos") and not k.startswith("robot0_") for k in dropped)


def test_whitelist_is_proprioception_only():
    for k in T.PROPRIO_WHITELIST:
        assert k.startswith("robot0_") and "_to_" not in k and "image" not in k and "depth" not in k
