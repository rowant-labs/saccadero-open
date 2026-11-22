"""The long-task suite's tables for the Pro2.0 line (runner side; the counterpart of ``goal_paths`` for Goal).

LIBERO-10's ten base tasks stand at three tables. ``long_lib.py`` built one workcell ("cell") per table in the variant
tree ``runs/pro2/long-lib`` (built 2026-11-14, cherry-picked as f7e2e76c), each registered as its own suite key because
``m2_library`` reads ``libero_10`` as an instance name:

* ``libero_long_kitchen``: the workcell equals Spatial's and Goal's, so it uses Spatial's Panda;
* ``libero_long_living_room`` and ``libero_long_study``: their own robot mounts, so their own Pandas, in the tree's
  ``pandas/`` folder (``panda_of`` None; ``m2_library.LibraryReader.panda``).

The robot is never told which cell an episode's task uses (that is evaluator truth). The runner hands it the cells'
type lists as data, and the robot picks the cell whose types name the instruction's objects (``robot.episode
.choose_cell``). The perception server serves every cell of the suite (``server_suites``), and each request carries the
cell key as its ``suite``.
"""

from __future__ import annotations

from rowant_lab.pro2 import goal_paths as GP

SUITE = "libero_10"
LIB = "/Volumes/RowantNVMe/rowant-cold/runs/pro2/long-lib"  # the variant tree (``lfb_library``)
CELLS = {
    "libero_long_kitchen": {
        "types": ["akita_black_bowl", "flat_stove", "microwave", "moka_pot", "porcelain_mug", "white_cabinet",
                  "white_yellow_mug", "wine_bottle", "wine_rack"],
        "panda_of": "libero_spatial"},
    "libero_long_living_room": {
        "types": ["alphabet_soup", "basket", "butter", "chocolate_pudding", "cream_cheese", "ketchup", "milk",
                  "orange_juice", "plate", "porcelain_mug", "red_coffee_mug", "tomato_sauce", "white_yellow_mug"],
        "panda_of": None},
    "libero_long_study": {
        "types": ["black_book", "desk_caddy", "white_yellow_mug"],
        "panda_of": None},
}


def cells_of(suite: str) -> dict:
    """``{cell: [type, ...]}`` for a suite whose tasks use several tables, else {}."""
    return {k: list(v["types"]) for k, v in CELLS.items()} if suite == SUITE else {}


def server_suites(suites) -> list:
    """The suites the perception server serves: the long suite as its cells."""
    out = []
    for s in suites:
        out.extend(sorted(CELLS) if s == SUITE else [s])
    return sorted(set(out))


def extra_suites(suites) -> dict | None:
    """Goal's entry (``goal_paths``) and the long suite's cells, as the suites need them; None for M2's own suites."""
    out = dict(GP.extra_suites(suites) or {})
    if SUITE in set(suites) or any(s in CELLS for s in suites):
        out.update({k: {"types": list(v["types"]), "panda_of": v["panda_of"]} for k, v in CELLS.items()})
    return out or None


def server_params(server: dict | None, suites) -> dict | None:
    """The perception server's params with ``extra_suites`` for these suites (``server`` as given when none is needed)."""
    extra = extra_suites(suites)
    if extra is None:
        return server
    if (server or {}).get("extra_suites") not in (None, extra):
        raise ValueError(f"server params carry other extra_suites: {server['extra_suites']}")
    return {**(server or {}), "extra_suites": extra}
