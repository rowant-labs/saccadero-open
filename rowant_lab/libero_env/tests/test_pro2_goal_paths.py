"""The Goal suite's entries live on the Pro2.0 side (``rowant_lab/pro2/goal_paths.py``), not in M2's
``m2_paths.py``; M2's shared library and server take them only as ``extra_suites``, which M2's own runs never pass.

The library checks skip when M2's library or the Goal variant tree is not on this machine. (As run, ``m2_paths.py``
was frozen byte for byte by ``evidence-m2/frozen-config.json``; that file is not part of this release, and here
``m2_paths.py`` takes its data folder from ``rowant_lab/settings.py``, so the byte check is not here.)
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rowant_lab.libero_env import m2_library as LB
from rowant_lab.libero_env import m2_paths as MP
from rowant_lab.pro2 import goal_lib as GL
from rowant_lab.pro2 import goal_paths as GP

GOAL = {"libero_goal": {"types": ["akita_black_bowl", "cream_cheese", "plate", "wine_bottle", "wooden_cabinet",
                                  "flat_stove", "wine_rack"], "panda_of": "libero_spatial"}}  # as 25ccee5c had them
HAVE_M2 = (MP.POLICY_LIB / "manifest.json").exists()
HAVE_GOAL = (GL.G / "m2" / "policy-v3" / "manifest.json").exists()
need_m2 = pytest.mark.skipif(not HAVE_M2, reason="M2's library is not on this machine")
need_goal = pytest.mark.skipif(not (HAVE_M2 and HAVE_GOAL), reason="the Goal variant tree is not on this machine")


def test_m2_paths_names_no_goal():
    assert set(MP.WORKCELL_TYPES) == set(MP.SUITES) and "libero_goal" not in MP.WORKCELL_TYPES
    assert not hasattr(MP, "PANDA_OF")


def test_goal_entries_moved_unchanged():
    assert GP.extra_suites([GP.SUITE]) == GOAL
    assert set(GL.NEW_TYPES) | set(GL.SHARED_TYPES) == set(GP.TYPES) and GL.SUITE == GP.SUITE
    assert set(GL.SHARED_TYPES) <= {t for s in MP.SUITES for t in MP.WORKCELL_TYPES[s]}  # types M2's library has
    assert set(GL.NEW_TYPES).isdisjoint(t for s in MP.SUITES for t in MP.WORKCELL_TYPES[s])
    assert GP.PANDA_OF in MP.SUITES


def test_only_goal_adds_extra_suites():
    assert GP.extra_suites(["libero_object", "libero_spatial"]) is None
    assert GP.extra_suites(["libero_spatial", "libero_goal"]) == GOAL
    assert GP.server_params(None, ["libero_object"]) is None
    srv = {"report_extra": True}
    assert GP.server_params(srv, ["libero_object", "libero_spatial"]) is srv  # an M2-suite run: params as given
    assert GP.server_params(srv, ["libero_goal"]) == {"report_extra": True, "extra_suites": GOAL}
    assert srv == {"report_extra": True}
    assert GP.server_params({"extra_suites": GOAL}, ["libero_goal"]) == {"extra_suites": GOAL}
    with pytest.raises(ValueError):
        GP.server_params({"extra_suites": {"libero_goal": {"types": [], "panda_of": "libero_spatial"}}}, ["libero_goal"])


def test_extra_suite_table_refuses_m2_suites_and_unknown_pandas():
    assert LB.extra_suite_table(None) == {}
    assert LB.extra_suite_table(GOAL) == {"libero_goal": (tuple(GP.TYPES), "libero_spatial")}
    with pytest.raises(LB.LibraryRefused):
        LB.extra_suite_table({"libero_spatial": {"types": ["plate"], "panda_of": "libero_spatial"}})
    with pytest.raises(LB.LibraryRefused):
        LB.extra_suite_table({"libero_goal": {"types": ["plate"], "panda_of": "libero_goal"}})


@need_m2
def test_m2_reader_without_extra_suites_is_m2s():
    r = LB.LibraryReader(allow_panda=True)
    assert r.suites() == set(MP.SUITES)
    for s in MP.SUITES:
        assert r.suite_types(s) is MP.WORKCELL_TYPES[s]
    with pytest.raises(LB.LibraryRefused):
        r.suite_types("libero_goal")


@need_goal
def test_goal_through_its_variant_tree():
    _ob, pol, m1 = LB.lfb_dirs(GL.G)
    assert json.loads((pol / "workcells" / "libero_goal.json").read_text())["types"] == list(GP.TYPES)  # as built
    with pytest.raises(LB.LibraryRefused):
        LB.M2Library("libero_goal", lib_dir=pol, m1_library=m1)  # no extra_suites: M2's tables only
    lib = LB.M2Library("libero_goal", lib_dir=pol, m1_library=m1, extra_suites=GP.extra_suites(["libero_goal"]))
    assert set(lib.export_types) == set(lib.types) == set(GP.TYPES)
    r = LB.LibraryReader(pol, m1, allow_panda=True, extra_suites=GP.extra_suites(["libero_goal"]))
    assert r.suites() == set(MP.SUITES) | {"libero_goal"}
    goal, spatial = r.panda("libero_goal"), r.panda("libero_spatial")
    assert goal.order == spatial.order and set(goal.arrays) == set(spatial.arrays)
    assert all(np.array_equal(goal.arrays[k], spatial.arrays[k]) for k in goal.arrays)


@need_goal
def test_the_server_takes_goal_from_its_params_only():
    from rowant_lab.libero_env import m2_perceive as P

    assert P.Perceiver({"lfb_library": str(GL.G)}).reader.suites() == set(MP.SUITES)
    srv = GP.server_params({"lfb_library": str(GL.G)}, ["libero_goal", "libero_spatial"])
    assert P.Perceiver(srv).reader.suites() == set(MP.SUITES) | {"libero_goal"}
