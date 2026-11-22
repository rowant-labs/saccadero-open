"""Build the M1 grasp records for every object type of the library (plan pipeline step 2, as amended: A1).

Numpy only; no LIBERO, MuJoCo or torch. Run in the Monty venv from the worktree root::

    PYTHONPATH=$PWD <server python> -m rowant_lab.libero_env.build_records

Reads ``library/types/<type>.json`` (objects only, not fixtures) and ``library/robot.json``; writes
``library/records/<type>.json`` and ``library/records/records-manifest.json`` (sha256 of every input and output,
the parameters, the generator's sha256, counts and timings).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.libero_env import grasp as GR  # noqa: E402
from rowant_lab.libero_env import m1_paths as MP  # noqa: E402


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def root_prims(model: dict) -> tuple[list, list]:
    root = model["bodies"][model["root_body"]]
    return root["primitives"], root["com"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default=str(MP.LIBRARY_DIR))
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    lib = Path(a.library)
    robot_b = (lib / "robot.json").read_bytes()
    jaw = GR.JawModel.from_robot(json.loads(robot_b))
    params = GR.RecordParams()
    man = {"format": "rowant libero_env M1 grasp records v1", "plan_sha256": MP.PLAN_SHA256, "label": MP.LABEL,
           "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "generator": {"build_records.py": sha256_bytes(Path(__file__).read_bytes()),
                         "grasp.py": sha256_bytes(Path(GR.__file__).read_bytes())},
           "robot_json_sha256": sha256_bytes(robot_b), "jaw": dataclasses.asdict(jaw),
           "params": dataclasses.asdict(params), "types": {}}
    for f in sorted((lib / "types").glob("*.json")):
        raw = f.read_bytes()
        model = json.loads(raw)
        if model["kind"] != "object" or (a.only and model["type"] not in a.only):
            continue
        prims, com = root_prims(model)
        t0 = time.time()
        rec = GR.build_records(prims, model["rest_up_in_body"], com, jaw, params, name=model["type"])
        dt = time.time() - t0
        out = GR.records_to_json(rec)
        out["source"] = {"type_json": f.name, "type_json_sha256": sha256_bytes(raw)}
        data = (json.dumps(out, sort_keys=True) + "\n").encode()
        p = lib / "records" / f"{model['type']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        man["types"][model["type"]] = {"file": f"records/{p.name}", "sha256": sha256_bytes(data),
                                       "n_records": rec["n_records"], "n_kept": int(len(rec["width"])),
                                       "height_m": round(rec["height"], 5), "seconds": round(dt, 2)}
        w = rec["width"]
        print(f"{model['type']:30s} records {rec['n_records']:6d} kept {len(w):5d} height {rec['height']:.4f} "
              f"width [{w.min() if len(w) else 0:.4f}, {w.max() if len(w) else 0:.4f}] {dt:.1f}s", flush=True)
    (lib / "records").mkdir(parents=True, exist_ok=True)
    (lib / "records" / "records-manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
