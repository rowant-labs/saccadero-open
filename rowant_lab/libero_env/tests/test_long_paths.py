"""The long-task suite's tables in the runner (long_paths) and the library (m2_library: a cell's own Panda)."""

from pathlib import Path

import pytest

from rowant_lab.libero_env import m2_library as LB
from rowant_lab.pro2 import goal_paths as GP
from rowant_lab.pro2 import long_paths as LP
from rowant_lab.pro2 import machine as MC

TREE = Path(MC.LONG_LIB)  # the built tree on this machine (``LP.LIB`` is the registered text, not a place to read)


def test_server_suites_and_extra_suites():
    assert LP.server_suites(["libero_10"]) == sorted(LP.CELLS)
    assert LP.server_suites(["libero_goal", "libero_object"]) == ["libero_goal", "libero_object"]
    assert LP.extra_suites(["libero_object", "libero_spatial"]) is None  # M2's own suites: nothing extra
    assert LP.extra_suites(["libero_goal"]) == GP.extra_suites(["libero_goal"])  # Goal exactly as before
    ex = LP.extra_suites(["libero_long_study"])
    assert set(ex) == set(LP.CELLS) and ex["libero_long_study"]["panda_of"] is None
    both = LP.extra_suites(["libero_goal", "libero_long_kitchen"])
    assert "libero_goal" in both and "libero_long_kitchen" in both
    assert LP.cells_of("libero_10")["libero_long_study"] == ["black_book", "desk_caddy", "white_yellow_mug"]
    assert LP.cells_of("libero_goal") == {}
    assert LP.server_params(None, ["libero_object"]) is None
    assert LP.server_params({"routes": {}}, ["libero_10"])["extra_suites"] == LP.extra_suites(["libero_10"])


def test_extra_suite_table_allows_an_own_panda_only_as_none():
    t = LB.extra_suite_table({"x_cell": {"types": ["a"], "panda_of": None}, "y_cell": {"types": ["b"],
                                                                                   "panda_of": "libero_spatial"}})
    assert t == {"x_cell": (("a",), None), "y_cell": (("b",), "libero_spatial")}
    with pytest.raises(LB.LibraryRefused):
        LB.extra_suite_table({"z_cell": {"types": ["a"], "panda_of": "libero_10"}})


@pytest.mark.skipif(not (TREE / "m2" / "policy-v3" / "manifest.json").is_file(),
                    reason="the long suite's library tree is not built on this machine")
def test_each_cell_loads_from_the_built_tree_with_its_panda():
    _ob, policy, m1 = LB.lfb_dirs(str(TREE))
    ex = LP.extra_suites(["libero_10"])
    for cell, spec in LP.CELLS.items():
        lib = LB.M2Library(cell, lib_dir=policy, m1_library=m1, extra_suites=ex)
        assert set(spec["types"]) <= set(lib.types)
        rd = LB.LibraryReader(lib_dir=policy, m1_library=m1, allow_panda=True, extra_suites=ex)
        pa = rd.panda(cell)
        assert pa is not None
        if spec["panda_of"] is None:  # its own model, from the tree, sha-checked through the manifest
            assert any(str(policy / "pandas") in o for o in rd.opened)
