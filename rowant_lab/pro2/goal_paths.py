"""The Goal suite's entries for the Pro2.0 line (decided 2026-11-08: the Goal suite next). Pro2.0 side only.

Commit 25ccee5c (2026-11-08) first put these in M2's ``rowant_lab/libero_env/m2_paths.py``. That module is frozen
(M2's frozen configuration records its bytes), so on 2026-11-14 the
entries moved here, unchanged, and m2_paths went back to its frozen bytes:

* ``TYPES``: the Goal workcell's types. ``goal_lib policy`` writes them into the variant tree's
  ``workcells/libero_goal.json`` (``runs/pro2/goal-lib``), which ``M2Library`` checks against them.
* ``PANDA_OF``: the M2 suite whose exported Panda model Goal uses: the same table and mount (``goal_lib export``
  checked it).

M2's shared modules never import this file (the M2 truth guard allows them no Pro2.0 import). The Pro2.0 line hands
the entries over as ``extra_suites`` (``{suite: {"types": [...], "panda_of": suite}}``): to ``m2_library.M2Library``
on the policy side (``dev_episode``, ``learn_skill``) and in the perception server's params (``dev_run``), from where
the server's ``LibraryReader`` takes them. M2's own runs never pass it, so they read m2_paths' frozen tables only.
"""

from __future__ import annotations

SUITE = "libero_goal"
TYPES = ("akita_black_bowl", "cream_cheese", "plate", "wine_bottle", "wooden_cabinet", "flat_stove", "wine_rack")
PANDA_OF = "libero_spatial"


def extra_suites(suites) -> dict | None:
    """The ``extra_suites`` entry these suites need: Goal's when Goal is among them, else None (M2's suites need
    none, and an M2-only run stays exactly as it was)."""
    if SUITE not in set(suites):
        return None
    return {SUITE: {"types": list(TYPES), "panda_of": PANDA_OF}}


def server_params(server: dict | None, suites) -> dict | None:
    """The perception server's params with ``extra_suites`` added when the suites need it, else ``server`` as given."""
    extra = extra_suites(suites)
    if extra is None:
        return server
    if (server or {}).get("extra_suites") not in (None, extra):
        raise ValueError(f"server params carry other extra_suites: {server['extra_suites']}")
    return {**(server or {}), "extra_suites": extra}
