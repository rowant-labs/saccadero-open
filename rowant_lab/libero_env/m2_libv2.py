"""M2 library version policy-v2 (amendment A9; verifier blocker B1): what the policy and the server load.

M2 (perceived poses, grounded instruction): perception in the loop. Build side, run once (Monty venv or
venv-libero; standard library only), never at episode time::

    PYTHONPATH=<repository root> <server python> -P \\
        -m rowant_lab.libero_env.m2_libv2 build

It writes ``runs/libero-m2/library/policy-v2/`` from

* the M2 export (``export/types/<type>.json`` and ``export/workcells/<suite>.json``), each checked against the
  export manifest's sha256: the ROOT body's ``body_pos`` and ``body_quat`` are set to the identity (the export had
  stored each type's root pose from the compiled model of the first task holding it: for the cabinet and the stove
  that is LIBERO-Spatial task 0's true fixture pose) and ``seen_in`` / ``exported_from`` (per-task object lists and
  provenance) are removed;
* M1's frozen types (``types/<type>.json``), each checked against M1's frozen config (1aefb103...): ``seen_in`` and
  ``rest_up_from`` are removed (primitives, ``com``, masses and ``rest_up_in_body`` are copied unchanged).

Nothing else changes: every other field is copied as it is (``tests/test_m2_library.py`` checks that each output is
its source minus exactly these fields). M1's grasp records, M1's ``robot.json`` and the export's Panda files stay in
place; the manifest lists their sha256 (checked against M1's frozen config and the export manifest), so
``m2_library.LibraryReader`` sha-checks every file it opens. The manifest names which fields were removed or reset,
never their values. A version folder is never overwritten.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.libero_env import m2_library as LB  # noqa: E402
from rowant_lab.libero_env import m2_records as R  # noqa: E402
from rowant_lab.libero_env import m2_rpaths as MP  # noqa: E402

FORMAT = "rowant libero_env M2 library policy-v2 v1"
STRIP_EXPORT = ("seen_in", "exported_from")
STRIP_M1 = ("seen_in", "rest_up_from")


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dumps(doc) -> bytes:
    return json.dumps(doc, indent=1, sort_keys=True).encode("utf-8")


def sanitize_export_type(doc: dict) -> dict:
    out = copy.deepcopy(doc)
    for k in STRIP_EXPORT:
        out.pop(k, None)
    root = out["bodies"][out["root_body"]]
    root["body_pos"] = list(LB.IDENTITY_POS)
    root["body_quat"] = list(LB.IDENTITY_QUAT)
    return out


def sanitize_m1_type(doc: dict) -> dict:
    out = copy.deepcopy(doc)
    for k in STRIP_M1:
        out.pop(k, None)
    return out


def all_types() -> list:
    return sorted(set(MP.WORKCELL_TYPES["libero_object"]) | set(MP.WORKCELL_TYPES["libero_spatial"]))


def build(out_dir: Path = MP.POLICY_LIB) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise SystemExit(f"REFUSE: {out_dir} exists (a library version is never overwritten)")
    exp_man_path = MP.EXPORT_DIR / "manifest.json"
    exp_man = json.loads(exp_man_path.read_text())
    m1fz = json.loads(MP.M1_FROZEN_CONFIG.read_text())
    if sha256_file(MP.M1_FROZEN_CONFIG) != MP.M1_FROZEN_CONFIG_SHA256:
        raise SystemExit("REFUSE: M1's frozen config differs from its logged sha256")
    m1lib = m1fz["library_sha256"]

    def check_export(rel: str) -> bytes:
        b = (MP.EXPORT_DIR / rel).read_bytes()
        if hashlib.sha256(b).hexdigest() != exp_man["files"][rel]:
            raise SystemExit(f"REFUSE: export/{rel} differs from the export manifest")
        return b

    def check_m1(rel: str) -> bytes:
        b = (MP.M1_LIBRARY / rel).read_bytes()
        if hashlib.sha256(b).hexdigest() != m1lib[rel]:
            raise SystemExit(f"REFUSE: M1 {rel} differs from M1's frozen config")
        return b

    staged: dict = {}
    sources: dict = {}
    for typ in all_types():
        src = json.loads(check_export(f"types/{typ}.json"))
        staged[f"types/{typ}.json"] = sanitize_export_type(src)
        sources[f"types/{typ}.json"] = {"from": f"export/types/{typ}.json",
                                        "from_sha256": exp_man["files"][f"types/{typ}.json"],
                                        "removed": [k for k in STRIP_EXPORT if k in src],
                                        "reset_to_identity": [f"bodies.{src['root_body']}.body_pos",
                                                              f"bodies.{src['root_body']}.body_quat"]}
        m1 = json.loads(check_m1(f"types/{typ}.json"))
        staged[f"m1types/{typ}.json"] = sanitize_m1_type(m1)
        sources[f"m1types/{typ}.json"] = {"from": f"m1/types/{typ}.json", "from_sha256": m1lib[f"types/{typ}.json"],
                                          "removed": [k for k in STRIP_M1 if k in m1], "reset_to_identity": []}
    for suite in MP.SUITES:
        wc = json.loads(check_export(f"workcells/{suite}.json"))
        staged[f"workcells/{suite}.json"] = wc
        sources[f"workcells/{suite}.json"] = {"from": f"export/workcells/{suite}.json",
                                              "from_sha256": exp_man["files"][f"workcells/{suite}.json"],
                                              "removed": [], "reset_to_identity": []}
    kinds = {"types": "type", "m1types": "m1type", "workcells": "workcell"}
    bad = {}
    for rel, doc in staged.items():
        pr = LB.problems(doc, kinds[rel.split("/")[0]])
        if pr:
            bad[rel] = pr[:5]
    if bad:
        raise SystemExit(f"REFUSE: sanitized documents still carry scene information: {json.dumps(bad)[:2000]}")
    in_place = {}
    for typ in all_types():
        rel = f"records/{typ}.json"
        if rel in m1lib:
            check_m1(rel)
            in_place[f"m1/{rel}"] = m1lib[rel]
    check_m1("robot.json")
    in_place["m1/robot.json"] = m1lib["robot.json"]
    for suite in MP.SUITES:
        for ext in ("json", "npz"):
            rel = f"panda-{suite}.{ext}"
            check_export(rel)
            in_place[f"export/{rel}"] = exp_man["files"][rel]
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        raise SystemExit(f"REFUSE: {tmp} exists")
    files = {}
    for rel, doc in staged.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        b = dumps(doc)
        p.write_bytes(b)
        files[rel] = hashlib.sha256(b).hexdigest()
    man = {"format": FORMAT, "label": MP.LABEL, "version": MP.POLICY_LIB_VERSION, "item": "m2f:guard",
           "amendment": "A9", "builder_sha256": sha256_file(Path(__file__)),
           "rules": {"root_body_pose": "identity (body_pos [0,0,0], body_quat [1,0,0,0]) in every type",
                     "removed_from_export_types": list(STRIP_EXPORT), "removed_from_m1_types": list(STRIP_M1),
                     "everything_else": "copied unchanged"},
           "sources": {"export_manifest_sha256": sha256_file(exp_man_path),
                       "m1_frozen_config_sha256": MP.M1_FROZEN_CONFIG_SHA256, "per_file": sources},
           "files": files, "in_place": in_place}
    R.write_json(tmp / "manifest.json", man)
    tmp.rename(out_dir)
    return {"dir": str(out_dir), "manifest_sha256": sha256_file(out_dir / "manifest.json"), "n_files": len(files),
            "n_in_place": len(in_place)}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        print(json.dumps(build(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
