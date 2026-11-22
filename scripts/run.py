#!/usr/bin/env python3
"""Run one of the two registered configurations on a list of episodes, on one machine.

    python scripts/run.py --config v9  --episodes EPISODES.json --out RUN_DIR [--workers 10]
    python scripts/run.py --config v11 --episodes EPISODES.json --out RUN_DIR

What it does, in order:

1. reads ``configs/<config>.registered.json`` and checks its checksum against the lock written before that scored
   run (``docs/records``): the parameters are the registered ones, byte for byte;
2. replaces the four places the registered text names (the policy's folder, the object libraries, the two depth
   nets) by this machine's (``rowant_lab/pro2/machine.py``) and checks that they exist;
3. writes ``RUN_DIR/episodes.json`` and ``RUN_DIR/params.json`` and starts the runner (``rowant_lab/pro2/dev_run.py``)
   in the worker's environment with N worker processes, each with its own perception server.

A run that is stopped can be started again with the same command: finished episodes are not played again.
Pictures are asked at 1024 pixels, as in both scored runs (the benchmark's default is 256; the depth nets were
trained at 1024). Episodes of the benchmark's test split are refused unless ``--test-split`` is given.

Any Python 3.9 or later runs this script; it needs no package. Episode lists come from ``scripts/episodes.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rowant_lab import settings  # noqa: E402
from rowant_lab.pro2 import machine  # noqa: E402

CONFIGS = {"v9": (1, "SCORED.lock.json"), "v11": (2, "SCORED2.lock.json")}


def registered(config: str) -> tuple:
    """(the registered parameter text, its sha256), refused when it differs from the lock."""
    attempt, lock_name = CONFIGS[config]
    text = (REPO / "configs" / f"{config}.registered.json").read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    lock = json.loads((REPO / "docs" / "records" / lock_name).read_text())
    if sha != lock["params_sha256"]:
        raise SystemExit(f"configs/{config}.registered.json differs from the parameters registered for attempt {attempt}")
    return text, sha


def missing(params: dict) -> list:
    lib = Path(params["lfb_library"])
    need = {"the policy package": Path(params["robot_dir"]) / "agent.py",
            "the object libraries (scripts/build_libraries.py)": lib / "m2" / "policy-v3" / "manifest.json",
            "the main depth net": Path(params["colour"]["net"]), "the second depth net": Path(params["colour"]["net2"]),
            "the worker's interpreter (scripts/setup_env.sh)": Path(machine.WORKER_PYTHON),
            "the perception server's interpreter (scripts/setup_env.sh)": Path(machine.SERVER_PYTHON)}
    return [f"{what}: {p}" for what, p in need.items() if not p.exists()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", choices=sorted(CONFIGS), required=True)
    ap.add_argument("--episodes", type=Path, required=True, help="a list made by scripts/episodes.py")
    ap.add_argument("--out", type=Path, required=True, help="the run folder (rows are appended; a re-run resumes)")
    ap.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 2) // 2 - 1)))
    ap.add_argument("--camera-px", type=int, default=1024, help="1024 in both scored runs")
    ap.add_argument("--test-split", action="store_true", help="yes, these are episodes of the benchmark's test split")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                    help="where the depth nets run. Both scored runs used cuda; cpu is slower and is not the "
                         "registered configuration (outcomes can differ)")
    a = ap.parse_args(argv)

    text, sha = registered(a.config)
    params = machine.localize(json.loads(text))
    if a.device != params["colour"].get("device", "cpu"):
        params["colour"]["device"] = a.device
        print(f"NOTE: depth nets on {a.device}: not the registered configuration", flush=True)
    gone = missing(params)
    if gone:
        raise SystemExit("not found on this machine:\n  " + "\n  ".join(gone))
    eps = json.loads(a.episodes.read_text())
    if any(e.get("split") == "test" for e in eps) and not a.test_split:
        raise SystemExit("this list holds episodes of the benchmark's test split; they run only with --test-split")
    a.out.mkdir(parents=True, exist_ok=True)
    (settings.DATA / "runs" / "libero-m2" / "numba-cache").mkdir(parents=True, exist_ok=True)  # numba's cache folder
    had = a.out / "episodes.json"
    if had.is_file() and json.loads(had.read_text()) != eps:
        raise SystemExit(f"{had} holds another list: use a new run folder")
    had.write_text(json.dumps(eps, indent=0))
    (a.out / "params.json").write_text(json.dumps(params))
    (a.out / "run-info.json").write_text(json.dumps({
        "config": a.config, "registered_params_sha256": sha, "camera_px": a.camera_px, "workers": a.workers,
        "device": params["colour"].get("device"),
        "episodes": len(eps), "local_places": machine.local()}, indent=1) + "\n")
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", env["MUJOCO_GL"])
    env.setdefault("LP_NUM_THREADS", "1")
    cmd = [machine.WORKER_PYTHON, "-I", "-B", str(REPO / "rowant_lab" / "pro2" / "dev_run.py"), "run", "--out",
           str(a.out), "--workers", str(a.workers), "--camera-px", str(a.camera_px), "--max-episodes", "10",
           "--params", json.dumps(params)]
    print(json.dumps({"config": a.config, "registered_params_sha256": sha, "episodes": len(eps), "workers": a.workers,
                      "run_folder": str(a.out)}), flush=True)
    with open(a.out / "run.log", "a") as log:
        rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"runner ended with {rc} (0: every episode has an outcome; 4: some are left, run the command again); "
          f"log: {a.out / 'run.log'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
