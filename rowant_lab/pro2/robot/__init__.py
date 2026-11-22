"""The robot the self-writing loop owns (from 2026-11-07). It started as M2's frozen policy side,
copied unchanged with its imports made relative:

* ``agent.py`` (M1's ``oracle_agent``: the step generator), ``grasp.py`` (grasp and place planning), ``follow.py``
  (``osc_follow``: the OSC pose follower), ``ground.py`` (``m2_ground``: the rule parser and grounding) and
  ``look.py`` (``m2_agent``: the perception tool, every call a look through the bridge);
* ``make_params`` below (``run_suite.make_params``).

Shared with the perception server and therefore NOT copied (the bridge's hello checks they match):
``rowant_lab.libero_env.m2_paths``, ``m2_kin``, ``m2_library`` and ``m2_bridge``. The loop edits files here, never M2's.
``dev_episode`` loads this package from any folder (``params["robot_dir"]``), so each candidate is a copy.
"""

from __future__ import annotations


def make_params(overrides: dict | None):
    from dataclasses import asdict

    from . import grasp as GR
    from .agent import AgentParams
    from .follow import FollowParams

    o = overrides or {}
    ap = AgentParams(**{**asdict(AgentParams()), **o.get("agent", {})})
    pp = GR.PlanParams(**{**asdict(GR.PlanParams()), **o.get("plan", {})})
    fp = FollowParams(**{**asdict(FollowParams()), **o.get("follow", {})})
    return ap, pp, fp
