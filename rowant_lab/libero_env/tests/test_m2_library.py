"""M2 library contents (plan truth_guard.library_contents; verifier blocker B1 and minor M7).

M2 (perceived poses, grounded instruction): perception in the loop. The policy and the perception server load the
policy-v2 library (amendment A9). These tests FAIL on any non-identity root pose, any per-task field (``seen_in``,
``exported_from``, ``rest_up_from``, task references) and any instance name in what they load -- on disk and as
loaded by ``m2_library.M2Library`` (the policy) and ``m2_library.LibraryReader`` (the server) -- and check that the
policy-v2 files are exactly their sources minus those fields (the kinematics and primitives are bit-identical, so
the change cannot move a percept or a plan).

Nothing here imports LIBERO, MuJoCo or torch. The tests that read the built library skip when it is absent.
"""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from rowant_lab.libero_env import m2_kin as KIN
from rowant_lab.libero_env import m2_library as LB
from rowant_lab.libero_env import m2_rpaths as MP

V2 = MP.POLICY_LIB
HAVE = (V2 / "manifest.json").exists()
need = pytest.mark.skipif(not HAVE, reason="the object library is not built on this machine")
ALL_TYPES = sorted(set(MP.WORKCELL_TYPES["libero_object"]) | set(MP.WORKCELL_TYPES["libero_spatial"]))


def _json(p):
    return json.loads(p.read_text())


def _all_keys_and_strings(obj, out_keys, out_strs):
    if isinstance(obj, dict):
        for k, v in obj.items():
            out_keys.append(str(k))
            _all_keys_and_strings(v, out_keys, out_strs)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _all_keys_and_strings(v, out_keys, out_strs)
    elif isinstance(obj, str):
        out_strs.append(obj)


def assert_clean(doc: dict, kind: str, label: str):
    """The library rule itself: no non-identity root pose, no per-task field, no instance name."""
    bad = LB.problems(doc, kind)
    assert not bad, f"{label}: {bad[:5]}"
    keys, strs = [], []
    _all_keys_and_strings(doc, keys, strs)
    assert not set(keys) & LB.PER_TASK_KEYS, (label, sorted(set(keys) & LB.PER_TASK_KEYS))
    for s in keys + strs:
        assert not LB.INSTANCE_NAME.search(s), (label, s)
        assert "_1_" not in s and "_2_" not in s, (label, s)
        assert not LB.TASK_REF.search(s), (label, s)
    if kind in ("type", "m1type") and "body_pos" in doc["bodies"][doc["root_body"]]:
        root = doc["bodies"][doc["root_body"]]
        assert root["body_pos"] == [0.0, 0.0, 0.0] and root["body_quat"] == [1.0, 0.0, 0.0, 0.0], label


# ------------------------------------------------------------------------------------------------ on disk
@need
@pytest.mark.parametrize("typ", ALL_TYPES)
def test_v2_types_have_identity_root_and_no_scene_fields(typ):
    assert_clean(_json(V2 / "types" / f"{typ}.json"), "type", f"types/{typ}")
    assert_clean(_json(V2 / "m1types" / f"{typ}.json"), "m1type", f"m1types/{typ}")


@need
@pytest.mark.parametrize("suite", MP.SUITES)
def test_v2_workcells_are_suite_level(suite):
    wc = _json(V2 / "workcells" / f"{suite}.json")
    assert_clean(wc, "workcell", suite)
    assert wc["identical_across_tasks"] is True and sorted(wc["types"]) == sorted(MP.WORKCELL_TYPES[suite])
    assert "instances" not in wc and "goal" not in wc and "sites" not in wc


@need
def test_v2_manifest_matches_every_file_and_names_no_value():
    man = _json(V2 / "manifest.json")
    assert man["version"] == MP.POLICY_LIB_VERSION
    for rel, h in man["files"].items():
        assert hashlib.sha256((V2 / rel).read_bytes()).hexdigest() == h, rel
    exp = _json(MP.EXPORT_DIR / "manifest.json")["files"]
    for rel, h in man["in_place"].items():
        if rel.startswith("m1/"):
            assert hashlib.sha256((MP.M1_LIBRARY / rel[3:]).read_bytes()).hexdigest() == h, rel
        else:
            assert exp[rel[len("export/"):]] == h, rel
    # the manifest says which fields were removed or reset, never their values
    text = json.dumps(man)
    assert "0.035610584" not in text and "-0.417132934" not in text and "libero_spatial:" not in text


@need
@pytest.mark.parametrize("typ", ALL_TYPES)
def test_v2_is_its_source_minus_exactly_the_scene_fields(typ):
    src = _json(MP.EXPORT_DIR / "types" / f"{typ}.json")
    out = _json(V2 / "types" / f"{typ}.json")
    want = copy.deepcopy(src)
    for k in ("seen_in", "exported_from"):
        want.pop(k, None)
    want["bodies"][want["root_body"]]["body_pos"] = [0.0, 0.0, 0.0]
    want["bodies"][want["root_body"]]["body_quat"] = [1.0, 0.0, 0.0, 0.0]
    assert out == want, typ
    m1 = _json(MP.M1_LIBRARY / "types" / f"{typ}.json")
    m1o = _json(V2 / "m1types" / f"{typ}.json")
    want1 = {k: v for k, v in m1.items() if k not in ("seen_in", "rest_up_from")}
    assert m1o == want1, typ  # primitives, com, masses and rest_up_in_body unchanged


@need
@pytest.mark.parametrize("typ", ["wooden_cabinet", "flat_stove", "basket", "akita_black_bowl"])
def test_kinematics_do_not_depend_on_the_removed_root_pose(typ):
    """Every body pose a percept gets (``m2_kin.part_body_poses``) is bit-identical with the old and new files."""
    src = _json(MP.EXPORT_DIR / "types" / f"{typ}.json")
    out = _json(V2 / "types" / f"{typ}.json")
    rng = np.random.default_rng(7)
    for _ in range(20):
        q = rng.normal(size=4)
        R = KIN.quat_to_mat(q / np.linalg.norm(q))
        t = rng.normal(size=3)
        jv = {pa: float(rng.uniform(*P["joint"]["range"])) for pa, P in src["parts"].items() if P["joint"]}
        a = KIN.part_body_poses(src, {src["root_body"]: (R, t)}, jv)
        b = KIN.part_body_poses(out, {out["root_body"]: (R, t)}, jv)
        assert a.keys() == b.keys()
        for k in a:
            assert np.array_equal(a[k][0], b[k][0]) and np.array_equal(a[k][1], b[k][1]), (typ, k)


# ------------------------------------------------------------------------------------------------ as loaded
@need
@pytest.mark.parametrize("suite", MP.SUITES)
def test_what_the_policy_loads(suite):
    lib = LB.M2Library(suite)
    assert lib.root == V2.resolve()
    assert_clean(lib.workcell, "workcell", f"{suite}: workcell")
    for t in MP.WORKCELL_TYPES[suite]:
        assert_clean(lib.export_types[t], "type", f"{suite}: export_types[{t}]")
        assert_clean(lib.types[t], "m1type", f"{suite}: types[{t}]")
        root = lib.types[t]["bodies"][lib.types[t]["root_body"]]
        assert "primitives" in root and "com" in root, t  # what M1's agent reads (M7)
    for p in lib.opened:
        assert ("scenes" not in p) and (p.startswith(str(V2.resolve())) or p.startswith(str(MP.M1_LIBRARY))), p
    assert not any(p.startswith(str((MP.M1_LIBRARY / "types").resolve())) for p in lib.opened)


@need
@pytest.mark.parametrize("suite", MP.SUITES)
def test_what_the_server_loads(suite):
    r = LB.LibraryReader(allow_panda=True)
    for t in MP.WORKCELL_TYPES[suite]:
        assert_clean(r.export_type(t), "type", t)
        assert_clean(r.m1_type(t), "m1type", t)
    assert_clean(r.workcell(suite), "workcell", suite)
    pd = r.panda(suite)
    assert pd.order and pd.arrays
    for p in r.opened:
        assert "scenes" not in p and "/export/types/" not in p and "/libero-m1/library/types/" not in p, p


# ------------------------------------------------------------------------------------------------ the rule itself
PLANTED = [
    ("root pose", lambda d: d["bodies"][d["root_body"]].update(body_pos=[0.0356, -0.2776, 0.905])),
    ("root quat", lambda d: d["bodies"][d["root_body"]].update(body_quat=[0.218, 0.0, 0.0, 0.976])),
    ("seen_in", lambda d: d.update(seen_in=["libero_spatial:0"])),
    ("exported_from", lambda d: d.update(exported_from="libero_spatial task 0 DEV state 0")),
    ("rest_up_from", lambda d: d.update(rest_up_from="x")),
    ("instance name", lambda d: d["parts"].setdefault("x", {})
     .update(joint={"name": "wooden_cabinet_1_top_level"})),
    ("instance key", lambda d: d["bodies"].update({"akita_black_bowl_1_main": {}})),
    ("task reference", lambda d: d.update(note="from libero_object:3")),
    ("instances", lambda d: d.update(instances={})),
    ("goal", lambda d: d.update(goal=[])),
]


@need
@pytest.mark.parametrize("label,plant", PLANTED, ids=[x[0] for x in PLANTED])
def test_the_rule_catches_planted_scene_information(label, plant):
    doc = _json(V2 / "types" / "wooden_cabinet.json")
    assert not LB.problems(doc, "type")
    plant(doc)
    assert LB.problems(doc, "type"), label


def test_the_loader_refuses_a_file_with_scene_information(tmp_path):
    lib = tmp_path / MP.POLICY_LIB_VERSION  # m2r3:fix-r1 (A65): the version in use
    (lib / "types").mkdir(parents=True)
    doc = {"type": "plate", "root_body": "main", "kind": "object", "sites": {},
           "bodies": {"main": {"body_pos": [0.1, 0.0, 0.9], "body_quat": [1.0, 0.0, 0.0, 0.0], "parent": None}}}
    b = json.dumps(doc).encode()
    (lib / "types" / "plate.json").write_bytes(b)
    man = {"version": MP.POLICY_LIB_VERSION, "files": {"types/plate.json": hashlib.sha256(b).hexdigest()},
           "in_place": {}}
    (lib / "manifest.json").write_text(json.dumps(man))
    r = LB.LibraryReader(lib, tmp_path / "m1")
    with pytest.raises(LB.LibraryRefused):
        r.export_type("plate")
    (lib / "types" / "plate.json").write_bytes(b.replace(b"0.1", b"0.0"))  # now clean but the sha differs
    with pytest.raises(LB.LibraryRefused):
        r.export_type("plate")


