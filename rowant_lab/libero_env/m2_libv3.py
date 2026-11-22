"""M2 library version policy-v3 (amendment A58, the orchestrator's D-STOVE): policy-v2's rules over export-v3.

M2 (perceived poses, grounded instruction): perception in the loop. Build side, run once (standard library only),
never at episode time::

    python -I -B <worktree>/rowant_lab/libero_env/m2_libv3.py build [--export DIR] [--out DIR]

It writes ``runs/libero-m2/library/policy-v3/`` exactly as ``m2_libv2.build`` wrote policy-v2, with the SAME rules
(``m2_libv2.sanitize_export_type`` and ``sanitize_m1_type``: every type's ROOT body pose set to the identity,
``seen_in`` / ``exported_from`` / ``rest_up_from`` removed, everything else copied unchanged; every document checked
by ``m2_library.problems``: no per-task field, no instance name, no task reference; M1's grasp records and robot.json
and the export's Panda files listed in place with their sha256), from ``export-v3`` (``m2_export3``: export v2 plus
the rendered geoms v2 left out; on these libraries the flat stove's base box) instead of ``export``. The manifest's
``version`` is ``policy-v3`` and it names its export. A version folder is never overwritten.

M1's files are checked against M1's own two manifests (``m1_hashes``: ``library-manifest.json`` written by
``export_geom`` and ``records/records-manifest.json`` written by ``build_records``), so the build needs nothing but
the library folders it reads.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.libero_env import m2_libv2 as LV2  # noqa: E402
from rowant_lab.libero_env import m2_library as LB  # noqa: E402
from rowant_lab.libero_env import m2_records as R  # noqa: E402
from rowant_lab.libero_env import m2_rpaths as MP  # noqa: E402

FORMAT = "rowant libero_env M2 library policy-v3 v1"
VERSION = "policy-v3"
EXPORT_V3 = MP.LIBRARY / "export-v3"
OUT_V3 = MP.LIBRARY / "policy-v3"


def m1_hashes(m1_library: Path = MP.M1_LIBRARY) -> tuple:
    """(``{relative path: sha256}`` of M1's types, robot.json and grasp records as M1's own two manifests list them,
    ``{manifest name: its sha256}``)."""
    m1_library = Path(m1_library)
    lib_man, rec_man = m1_library / "library-manifest.json", m1_library / "records" / "records-manifest.json"
    out = {rel: h for rel, h in json.loads(lib_man.read_text())["files"].items()
           if rel == "robot.json" or rel.startswith("types/")}
    for rec in json.loads(rec_man.read_text())["types"].values():
        out[rec["file"]] = rec["sha256"]
    return out, {"library-manifest.json": sha256_file(lib_man), "records/records-manifest.json": sha256_file(rec_man)}


def build(export_dir: Path = EXPORT_V3, out_dir: Path = OUT_V3, version: str = VERSION) -> dict:
    export_dir, out_dir = Path(export_dir), Path(out_dir)
    if out_dir.exists():
        raise SystemExit(f"REFUSE: {out_dir} exists (a library version is never overwritten)")
    exp_man_path = export_dir / "manifest.json"
    exp_man = json.loads(exp_man_path.read_text())
    m1lib, m1_manifests = m1_hashes()

    def check_export(rel: str) -> bytes:
        b = (export_dir / rel).read_bytes()
        if hashlib.sha256(b).hexdigest() != exp_man["files"][rel]:
            raise SystemExit(f"REFUSE: {export_dir.name}/{rel} differs from its manifest")
        return b

    def check_m1(rel: str) -> bytes:
        b = (MP.M1_LIBRARY / rel).read_bytes()
        if hashlib.sha256(b).hexdigest() != m1lib[rel]:
            raise SystemExit(f"REFUSE: M1 {rel} differs from M1's manifest")
        return b

    staged: dict = {}
    sources: dict = {}
    for typ in LV2.all_types():
        src = json.loads(check_export(f"types/{typ}.json"))
        staged[f"types/{typ}.json"] = LV2.sanitize_export_type(src)
        sources[f"types/{typ}.json"] = {"from": f"{export_dir.name}/types/{typ}.json",
                                        "from_sha256": exp_man["files"][f"types/{typ}.json"],
                                        "removed": [k for k in LV2.STRIP_EXPORT if k in src],
                                        "reset_to_identity": [f"bodies.{src['root_body']}.body_pos",
                                                              f"bodies.{src['root_body']}.body_quat"]}
        m1 = json.loads(check_m1(f"types/{typ}.json"))
        staged[f"m1types/{typ}.json"] = LV2.sanitize_m1_type(m1)
        sources[f"m1types/{typ}.json"] = {"from": f"m1/types/{typ}.json", "from_sha256": m1lib[f"types/{typ}.json"],
                                          "removed": [k for k in LV2.STRIP_M1 if k in m1], "reset_to_identity": []}
    for suite in MP.SUITES:
        wc = json.loads(check_export(f"workcells/{suite}.json"))
        staged[f"workcells/{suite}.json"] = wc
        sources[f"workcells/{suite}.json"] = {"from": f"{export_dir.name}/workcells/{suite}.json",
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
    for typ in LV2.all_types():
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
        b = LV2.dumps(doc)
        p.write_bytes(b)
        files[rel] = hashlib.sha256(b).hexdigest()
    man = {"format": FORMAT, "label": MP.LABEL, "version": version, "item": "m2r3:fix-r1",
           "amendment": "A58 (D-STOVE), under A9's rules", "builder_sha256": sha256_file(Path(__file__)),
           "rules_from": "m2_libv2 (sanitize_export_type, sanitize_m1_type, m2_library.problems)",
           "rules_builder_sha256": sha256_file(Path(LV2.__file__)),
           "rules": {"root_body_pose": "identity (body_pos [0,0,0], body_quat [1,0,0,0]) in every type",
                     "removed_from_export_types": list(LV2.STRIP_EXPORT), "removed_from_m1_types": list(LV2.STRIP_M1),
                     "everything_else": "copied unchanged"},
           "export": export_dir.name,
           "sources": {"export_manifest_sha256": sha256_file(exp_man_path),
                       "m1_manifests_sha256": m1_manifests, "per_file": sources},
           "files": files, "in_place": in_place}
    R.write_json(tmp / "manifest.json", man)
    tmp.rename(out_dir)
    return {"dir": str(out_dir), "manifest_sha256": sha256_file(out_dir / "manifest.json"), "n_files": len(files),
            "n_in_place": len(in_place)}


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--export", default=str(EXPORT_V3))
    b.add_argument("--out", default=str(OUT_V3))
    a = ap.parse_args(argv)
    print(json.dumps(build(Path(a.export), Path(a.out)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
