"""M1 policy tools: the observation whitelist, the ONE declared truth channel ``oracle_poses`` and the task spec.

ORACLE (true poses): diagnostic of the hands. This module is the only place where the runner's simulator handle
meets the policy's inputs. The policy (``oracle_agent``) receives:

* ``filter_obs(obs)``: the proprioception keys of ``PROPRIO_WHITELIST`` only (copies). Every observation truth key
  (``<obj>_pos``, ``<obj>_quat``, ``<obj>_to_robot0_eef_*``, ``object-state``) and every image is dropped.
* ``OracleTool``: a callable returning the world pose of every body of every task object and fixture subtree (an
  articulated fixture's drawer included), read from ``sim.data.xpos``/``xquat``. The simulator handle lives in a
  closure, never in an attribute the policy could reach; every call is counted and logged (step index, UTC-free).
  In M2 this one tool is swapped for perception; nothing else changes.
* ``task_spec(bddl_text, scene)``: the goal predicate, target and place object or region parsed from the BDDL text,
  and the instruction. The oracle takes the target's identity from the goal (no language or spatial grounding).

The policy never sees ``done``, ``reward``, ``info``, contacts, the evaluator's flags or any image.
"""

from __future__ import annotations

import re

PROPRIO_WHITELIST = ("robot0_joint_pos", "robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel",
                     "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_gripper_qvel")


def filter_obs(obs: dict) -> dict:
    """The proprioception whitelist only, as float lists copied out of the simulator's buffers."""
    out = {}
    for k in PROPRIO_WHITELIST:
        if k in obs:
            out[k] = [float(x) for x in obs[k]]
    return out


def dropped_keys(obs_keys) -> list[str]:
    return sorted(k for k in obs_keys if k not in PROPRIO_WHITELIST)


class OracleTool:
    """``tool()`` -> {body_name: {"pos": [x, y, z], "quat_wxyz": [w, x, y, z]}} for every task-object and fixture
    subtree body. ``tool.calls`` counts the reads."""

    def __init__(self, read_fn):
        self._read = read_fn  # a closure over the simulator; holds no named handle
        self.calls = 0
        self.call_steps: list[int] = []
        self.t_now = 0

    def __call__(self) -> dict:
        self.calls += 1
        self.call_steps.append(int(self.t_now))
        return self._read()


def make_oracle(env_, body_names) -> OracleTool:
    """Runner side: build the oracle tool over ``env_``'s simulator for the given body names."""
    s = env_.sim
    ids = [(n, s.model.body_name2id(n)) for n in body_names]

    def read():
        x, q = s.data.xpos, s.data.xquat
        return {n: {"pos": [float(v) for v in x[i]], "quat_wxyz": [float(v) for v in q[i]]} for n, i in ids}

    return OracleTool(read)


_GOAL = re.compile(r"\(:goal", re.I)
_ATOM = re.compile(r"\((\w+)\s+([\w]+)\s+([\w]+)\)")
_LANG = re.compile(r"\(:language\s+([^)]*)\)", re.I)


def _balanced(text: str, start: int) -> str:
    """The parenthesised block that opens at ``text[start] == '('``."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unbalanced parentheses in the BDDL text")


def parse_goal(bddl_text: str) -> list[tuple[str, str, str]]:
    m = _GOAL.search(bddl_text)
    if not m:
        raise ValueError("no (:goal ...) in the BDDL text")
    block = _balanced(bddl_text, m.start())
    return [(p.lower(), a, b) for p, a, b in _ATOM.findall(block)]


def task_spec(bddl_text: str, scene: dict) -> dict:
    """{predicate, target, place, place_kind: 'region' | 'object', place_instance, instruction}."""
    goals = parse_goal(bddl_text)
    if len(goals) != 1:
        raise ValueError(f"expected one goal atom, got {goals}")
    pred, target, place = goals[0]
    lang = _LANG.search(bddl_text)
    if place in scene["sites"]:
        kind, inst = "region", scene["sites"][place]["parent_instance"]
    elif place in scene["instances"]:
        kind, inst = "object", place
    else:
        raise ValueError(f"place {place!r} is neither a site nor an instance")
    if target not in scene["instances"]:
        raise ValueError(f"target {target!r} not an instance")
    return {"predicate": pred, "target": target, "place": place, "place_kind": kind, "place_instance": inst,
            "instruction": lang.group(1).strip() if lang else None}
