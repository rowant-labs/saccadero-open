"""The long-task suite's library build (``rowant_lab/pro2/long_lib.py``): its pure parts (cell grouping and names,
the same-object checks, the capsule tessellation, the export-v2 model view), and the built variant tree when it is on
this machine (byte-for-byte copies of goal-lib's files, each cell loading through ``m2_library``).

The tree checks skip when the long-lib variant tree is not on this machine.
"""

from __future__ import annotations

import copy
import hashlib
import json
import types

import numpy as np
import pytest

from rowant_lab.libero_env import m2_export3 as E3
from rowant_lab.libero_env import m2_library as LB
from rowant_lab.libero_env import m2_paths as MP
from rowant_lab.pro2 import goal_lib as GL
from rowant_lab.pro2 import goal_paths as GP
from rowant_lab.pro2 import long_lib as LL

POL = LL.LL / "m2" / "policy-v3"
HAVE_TREE = (POL / "manifest.json").exists() and (LL.BUILD / "cells.json").exists()
need_tree = pytest.mark.skipif(not HAVE_TREE, reason="the long-lib variant tree is not on this machine")
WC = {"arena": [{"geom": "floor", "body": "world", "type": "plane", "point": [0.0, 0.0, 0.0],
                 "normal": [0.0, 0.0, 1.0]}],
      "robot_mount": {"base_pos": [-0.66, 0.0, 0.912], "shoulder_pos": [-0.66, 0.0, 1.245], "note": "n"},
      "robot_base": {"pos": [-0.66, 0.0, 0.912], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
      "cameras": {"agentview": {"fovy_deg": 45.0}, "robot0_eye_in_hand": {"fovy_deg": 75.0}}}


# ------------------------------------------------------------------------------------------------ cells
def test_cell_names_from_tables_pass_the_library_rules():
    names = LL.cell_names([["kitchen_table"], ["living_room_table", "living_room_table"], ["study_table"]])
    assert names == ["libero_long_kitchen", "libero_long_living_room", "libero_long_study"]
    for n in names:
        assert LL.name_problems(n) == []
        doc = {"suite": n, "types": ["plate"]}
        assert LB.problems(doc, "workcell") == []  # the workcell document names its cell
        assert LB.extra_suite_table({n: {"types": ["plate"], "panda_of": "libero_spatial"}})


def test_cell_names_take_letters_when_a_table_repeats():
    assert LL.cell_names([["kitchen_table"], ["study_table"], ["kitchen_table"]]) == [
        "libero_long_kitchen_a", "libero_long_study", "libero_long_kitchen_b"]
    assert LL.cell_names([["kitchen_table", "study_table"]]) == ["libero_long_kitchen_study"]


@pytest.mark.parametrize("name", ["libero_10", "libero_10_kitchen", "libero_long_kitchen_2", "libero_long_k9",
                                  "libero_goal", "libero_spatial", "libero_object", "kitchen", "Libero_long_x",
                                  "libero_long_kitchen_"])
def test_name_problems_refuses_what_the_library_would(name):
    assert LL.name_problems(name)


def test_the_suite_key_itself_reads_as_an_instance_name():
    assert LB.INSTANCE_NAME.search(LL.SUITE)  # why the cells need keys of their own
    assert LB.problems({"suite": LL.SUITE}, "workcell")


def test_the_frying_pans_own_type_name_reads_as_an_instance_name():
    """Why ``policy`` leaves the kitchen's frying pan out: the rule cannot tell its model name from an instance."""
    assert LB.problems({"type": "chefmate_8_frypan"}, "type")
    assert not LB.problems({"type": "moka_pot"}, "type")


def test_group_by_workcell_groups_equal_workcells_in_task_order():
    near = copy.deepcopy(WC)
    near["robot_base"]["pos"][0] += 1e-8  # two MuJoCo builds a few bits apart
    moved = copy.deepcopy(WC)
    moved["robot_mount"]["base_pos"] = [-0.75, 0.0, 0.912]
    rows = [{"task": 0, "workcell": moved}, {"task": 1, "workcell": WC}, {"task": 2, "workcell": near},
            {"task": 3, "workcell": moved}]
    groups = LL.group_by_workcell(rows)
    assert [t for _wc, t in groups] == [[0, 3], [1, 2]]
    assert groups[0][0] is moved


def test_is_table():
    assert all(LL.is_table(t) for t in ("table", "kitchen_table", "living_room_table", "study_table"))
    assert not any(LL.is_table(t) for t in ("plate", "desk_caddy", "white_cabinet", "tablet"))


def test_base_types_are_m2s_and_goals():
    base = LL.base_types()
    assert {t for t, o in base.items() if o == "goal"} == set(GL.NEW_TYPES)
    assert {t for t, o in base.items() if o == "m2"} == {t for s in MP.SUITES for t in MP.WORKCELL_TYPES[s]}
    assert set(GP.TYPES) <= set(base)
    js, npz, m1 = LL.base_paths("plate", "m2")
    assert js.parent == MP.EXPORT_DIR / "types" and npz.suffix == ".npz" and m1.parent == GL.M1LIB / "types"
    js, _npz, m1 = LL.base_paths("wine_rack", "goal")
    assert js.is_relative_to(GL.G / "build") and m1.is_relative_to(GL.G / "build")
    with pytest.raises(ValueError):
        LL.base_paths("plate", "elsewhere")


def test_libero_config_names_the_simulators_own_task_files_and_start_states():
    text = LL.libero_config_text()
    assert f'bddl_files: "{LL.SIM}/bddl_files"' in text and f'init_states: "{LL.SIM}/init_files"' in text
    assert "service-source" not in text and str(LL.PAYLOAD) in text  # no task folder of the service


# ------------------------------------------------------------------------------------------------ same object
def _type_doc():
    body = {"parent": None, "body_pos": [0.1, 0.2, 0.3], "body_quat": [1.0, 0.0, 0.0, 0.0], "is_root": True,
            "joints": [], "geoms": [{"key": "g0", "n_vertices": 10, "n_faces": 8, "has_uv": True, "mesh": "m_vis",
                                     "material": {}}]}
    door = {"parent": "main", "body_pos": [0.0, 0.1, 0.0], "body_quat": [1.0, 0.0, 0.0, 0.0], "is_root": False,
            "joints": [{"name": "hinge", "type": "hinge", "axis": [0.0, 0.0, 1.0], "pos": [0.0, 0.0, 0.0],
                        "range": [0.0, 1.5], "qpos0": 0.0, "limited": True}],
            "geoms": [{"key": "g1", "n_vertices": 6, "n_faces": 4, "has_uv": False, "mesh": "door_vis",
                       "material": {}}]}
    return {"bodies": {"main": body, "door": door},
            "sites": {"heating_region": {"body": "main", "pos": [0.0, 0.0, 0.1], "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                         "size": [0.1, 0.1, 0.05]}}}


def _v3(doc, added):
    """``doc`` as export-v3 holds it: the probe's geoms appended to their bodies, and its record."""
    out = copy.deepcopy(doc)
    recs = []
    for k, a in enumerate(added):
        out["bodies"][a["body"]]["geoms"].append(
            {"key": f"g{9 + k}", "n_vertices": 4, "n_faces": 2, "has_uv": False, "mesh": a["name"],
             "material": {}, "primitive": {"type": a["geom_type"], "size": a["size"], "pos": a["pos"],
                                           "quat": a["quat"]}})
        recs.append({"body": a["body"], "geom": a["name"], "primitive": a["geom_type"], "size": a["size"]})
    out["rendered_geoms_added"] = recs
    return out


ADDED = [{"body": "door", "name": "g11", "geom_type": "capsule", "size": [0.01, 0.065, 0.0], "pos": [0.0, 0.0, 0.0],
          "quat": [1.0, 0.0, 0.0, 0.0]}]


def test_export_same_accepts_the_same_object_and_a_new_scene_pose():
    new = _type_doc()
    old = copy.deepcopy(new)
    old["bodies"]["main"]["body_pos"] = [9.0, 9.0, 9.0]  # the root's pose is the scene's, not the type's
    old["sites"]["heating_region"]["pos"][0] += 1e-8
    assert LL.export_same(old, new, [])["same"]
    assert LL.export_same(_v3(old, ADDED), new, ADDED)["same"]  # v3's additions, as the probe finds them again


@pytest.mark.parametrize("change", ["site_moved", "site_missing", "mesh_renamed", "mesh_missing", "joint_range",
                                    "child_offset", "added_missing", "added_elsewhere", "body_missing"])
def test_export_same_refuses_another_object(change):
    new = _type_doc()
    old = _v3(new, ADDED)
    added = copy.deepcopy(ADDED)
    if change == "site_moved":
        old["sites"]["heating_region"]["pos"][2] += 1e-3
    elif change == "site_missing":
        del old["sites"]["heating_region"]
    elif change == "mesh_renamed":
        old["bodies"]["main"]["geoms"][0]["mesh"] = "other_vis"
    elif change == "mesh_missing":  # the library's body holds a v2 mesh the fresh export lacks
        old["bodies"]["main"]["geoms"].insert(0, dict(old["bodies"]["main"]["geoms"][0], key="g7"))
    elif change == "joint_range":
        old["bodies"]["door"]["joints"][0]["range"] = [0.0, 1.0]
    elif change == "child_offset":
        old["bodies"]["door"]["body_pos"] = [0.0, 0.2, 0.0]
    elif change == "added_missing":
        added = []
    elif change == "added_elsewhere":
        added[0]["pos"] = [0.0, 0.0, 0.01]
    elif change == "body_missing":
        del old["bodies"]["door"]
    assert not LL.export_same(old, new, added)["same"]


def test_panda_diff_names_the_bodies_that_differ():
    a = {"robot0_base": {"parent": "world", "body_pos": [-0.66, 0.0, 0.912], "body_quat": [1, 0, 0, 0]},
         "robot0_link1": {"parent": "robot0_base", "body_pos": [0.0, 0.0, 0.333], "body_quat": [1, 0, 0, 0]}}
    b = copy.deepcopy(a)
    assert LL.panda_diff(a, b) == []
    b["robot0_base"]["body_pos"] = [-0.75, 0.0, 0.912]
    b["mount0_base"] = {"parent": "robot0_base", "body_pos": [0, 0, 0], "body_quat": [1, 0, 0, 0]}
    assert LL.panda_diff(a, b) == ["mount0_base", "robot0_base"]
    b = copy.deepcopy(a)
    b["robot0_link1"]["body_pos"][2] += 5e-6  # within goal_lib's 1e-5
    assert LL.panda_diff(a, b) == []


# ------------------------------------------------------------------------------------------------ tessellation
@pytest.mark.parametrize("size", [(0.01, 0.065, 0.0), (0.01, 0.015, 0.0), (0.03, 0.0, 0.0), (0.002, 0.2, 0.0)])
def test_capsule_arrays_lie_on_the_capsule_and_face_out(size):
    r, h = size[0], size[1]
    V, F, N = LL.capsule_arrays(size, E3.CELL_M)
    assert F.min() == 0 and F.max() == len(V) - 1 and len(np.unique(F)) == len(V)
    seg = np.clip(V[:, 2], -h, h)
    d = np.linalg.norm(V - np.stack([np.zeros_like(seg), np.zeros_like(seg), seg], 1), axis=1)
    assert np.allclose(d, r, atol=1e-12)  # every vertex on the surface
    assert np.allclose(np.linalg.norm(N, axis=1), 1.0)
    P = V - np.stack([np.zeros_like(seg), np.zeros_like(seg), seg], 1)
    assert np.allclose(N, P / r, atol=1e-12)  # the exact normals
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(B - A, C - A)
    area = 0.5 * np.linalg.norm(fn, axis=1)
    assert (area > 0).all()  # no degenerate face
    cen = (A + B + C) / 3.0
    cseg = np.clip(cen[:, 2], -h, h)
    out = cen - np.stack([np.zeros_like(cseg), np.zeros_like(cseg), cseg], 1)
    assert ((fn * out).sum(1) > 0).all()  # every face wound outward
    true = 2 * np.pi * r * 2 * h + 4 * np.pi * r * r
    assert 0.9 * true < area.sum() <= true  # a polyhedron inscribed in the capsule
    edges = np.concatenate([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1),
                            np.linalg.norm(A - C, axis=1)])
    assert edges.max() <= np.sqrt(2) * E3.CELL_M + 1e-12  # cells of at most CELL_M (a quad's diagonal)


def test_primitive_arrays_boxes_are_m2s_and_other_types_are_refused():
    for a, b in zip(LL.primitive_arrays("box", [0.095, 0.095, 0.02]), E3.primitive_arrays("box", [0.095, 0.095, 0.02]),
                    strict=True):
        assert np.array_equal(a, b)
    with pytest.raises(NotImplementedError):
        LL.primitive_arrays("cylinder", [0.01, 0.02, 0.0])
    rec = {"geom_type": "capsule", "size": [0.01, 0.065, 0.0], "pos": [0.1, 0.0, 0.0],
           "quat": [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]}  # 90 degrees about x: the axis along -y
    V, _F, N = LL.body_frame_arrays(rec)
    assert np.allclose([V[:, 1].min(), V[:, 1].max()], [-0.075, 0.075]) and np.allclose(V[:, 0].mean(), 0.1)
    assert np.allclose(np.linalg.norm(N, axis=1), 1.0)


def test_meshes_only_view_hides_non_mesh_visual_geoms_only():
    m = types.SimpleNamespace(geom_type=np.array([7, 3, 6, 7, 3]), geom_contype=np.array([0, 0, 1, 1, 0]),
                              geom_conaffinity=np.array([0, 0, 1, 1, 1]), nbody=4, body_id2name=lambda b: f"b{b}")
    env = types.SimpleNamespace(sim=types.SimpleNamespace(model=m, data="data"))
    view = LL._meshes_only(env)
    assert view.sim.data == "data"
    vm = view.sim.model
    assert vm.hidden == [1]  # the visual capsule; a visual mesh and the colliding geoms stay as they are
    assert list(vm.geom_contype) == [0, 1, 1, 1, 0] and list(m.geom_contype) == [0, 0, 1, 1, 0]
    assert vm.nbody == 4 and vm.body_id2name(2) == "b2" and vm.geom_conaffinity is m.geom_conaffinity


# ------------------------------------------------------------------------------------------------ the built tree
def _cells():
    return json.loads((LL.BUILD / "cells.json").read_text())["cells"]


@need_tree
def test_goal_libs_files_are_copied_byte_for_byte():
    gman = json.loads((GL.G / "m2" / "policy-v3" / "manifest.json").read_text())
    man = json.loads((POL / "manifest.json").read_text())
    assert all(man["files"].get(rel) == h for rel, h in gman["files"].items())
    assert all(man["in_place"].get(rel) == h for rel, h in gman["in_place"].items())
    for rel, h in man["files"].items():
        assert hashlib.sha256((POL / rel).read_bytes()).hexdigest() == h, rel
    m1 = LL.LL / "m1"
    for rel, h in man["in_place"].items():
        if rel.startswith("m1/"):
            assert hashlib.sha256((m1 / rel[3:]).read_bytes()).hexdigest() == h, rel
    onb = LL.LL / "m2" / "onboard-v3"
    for s, src in [(s, MP.ONBOARD_DIR / s) for s in MP.SUITES] + [(GP.SUITE, GL.G / "m2" / "onboard-v3" / GP.SUITE)]:
        if (onb / s).exists():
            assert (onb / s / "onboarding.json").read_bytes() == (src / "onboarding.json").read_bytes()


@need_tree
def test_every_cell_loads_through_the_library():
    cells, libcells = _cells(), LL.library_cells(POL)
    assert set(libcells) == set(cells) and all(not LL.name_problems(c) for c in cells)
    for cell, wc in libcells.items():
        # policy side: the Panda is the server's, so any M2 suite passes extra_suite_table's check here
        extra = {cell: {"types": wc["types"], "panda_of": cells[cell]["panda_of"] or "libero_spatial"}}
        lib = LB.M2Library(cell, lib_dir=POL, m1_library=LL.LL / "m1", extra_suites=extra)
        assert set(lib.export_types) == set(lib.types) == set(wc["types"])
        assert {t for t in wc["types"] if lib.types[t]["kind"] == "object"} <= set(lib.records)
        with pytest.raises(LB.LibraryRefused):
            LB.M2Library(cell, lib_dir=POL, m1_library=LL.LL / "m1")  # never without extra_suites


@need_tree
def test_a_cell_on_spatials_mount_serves_spatials_panda():
    for cell, c in _cells().items():
        if c["panda_of"] is None:
            assert (POL / "pandas" / f"panda-{cell}.json").exists()
            continue
        wc = LL.library_cells(POL)[cell]
        r = LB.LibraryReader(POL, LL.LL / "m1", allow_panda=True,
                             extra_suites={cell: {"types": wc["types"], "panda_of": c["panda_of"]}})
        mine, theirs = r.panda(cell), r.panda(c["panda_of"])
        assert mine.order == theirs.order and all(np.array_equal(mine.arrays[k], theirs.arrays[k])
                                                  for k in mine.arrays)


@need_tree
def test_excluded_and_added_as_the_build_recorded():
    rep = json.loads((LL.BUILD / "policy-report.json").read_text())
    libcells = LL.library_cells(POL)
    for typ in rep["excluded"]:
        assert not (POL / "types" / f"{typ}.json").exists()
        assert all(typ not in wc["types"] for wc in libcells.values())
    man = json.loads((POL / "manifest.json").read_text())
    for rel in man["files"]:
        if rel.startswith("types/"):
            doc = json.loads((POL / rel).read_text())
            for x in doc.get("rendered_geoms_added") or []:  # v3's additions sit in their body's geoms
                assert any(g["key"] == x["key"] and g["primitive"]["type"] == x["primitive"]
                           for g in doc["bodies"][x["body"]]["geoms"] if "primitive" in g)


@need_tree
def test_onboarded_cells_hold_their_workcell_types():
    onb = LL.LL / "m2" / "onboard-v3"
    for cell, wc in LL.library_cells(POL).items():
        if not (onb / cell / "onboarding.json").exists():
            continue
        doc = json.loads((onb / cell / "onboarding.json").read_text())
        assert {t for t, _p in doc["graph_ids"].values()} == set(wc["types"])
