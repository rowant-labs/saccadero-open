"""The perception server's start, stop and hello check, as the worker uses them.

This is what the LIBERO-Pro2.0 runner (``rowant_lab/pro2/dev_run.py``) uses of the earlier robot's runner: one
constant and four functions, their code unchanged. The earlier runner's own launch, gate and freeze commands
are not part of this release (docs/CHANGES-FROM-AS-RUN.md).

M2 (perceived poses, grounded instruction): perception in the loop. That is the earlier robot this module was
written for; its modules keep this label.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.libero_env import m2_rpaths as MP  # noqa: E402

MP.pin_numba_cache()  # before anything imports robosuite (and numba): this code's own cache folder

WATCHDOG_S = 600.0
SOCKET_TIMEOUT_S = 660.0  # the bridge's own timeout sits above the episode watchdog, so the watchdog fires first


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def start_server(scratch: Path, log_dir: Path, suites, params: dict | None, frozen: dict | None = None,
                 policy_lib_sha256: str | None = None) -> subprocess.Popen:
    """The perception server, logging to its OWN folder ``log_dir`` (``<run>/server/``: its audit hook may write
    only there and read no run folder). ``frozen``: {sha256, expect_code path} of a frozen run."""
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", NUMBA_CACHE_DIR=str(MP.NUMBA_CACHE),
               PYTHONPATH=str(MP.REPO_ROOT))
    cmd = [MP.MONTY_VENV_PY, "-P", "-m", "rowant_lab.libero_env.m2_perceive", "--socket", "p.sock", "--log-dir",
           str(log_dir), "--suites", *suites]
    if params:
        cmd += ["--params", json.dumps(params)]
    if policy_lib_sha256:
        cmd += ["--policy-lib-sha256", policy_lib_sha256]
    if frozen:
        cmd += ["--frozen-config-sha256", frozen["sha256"], "--expect-code", str(frozen["expect_code"])]
    out = open(log_dir / "server.log", "a")
    p = subprocess.Popen(cmd, cwd=scratch, env=env, stdout=out, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 600:
        if (scratch / "p.sock").exists():
            return p
        if p.poll() is not None:
            raise RuntimeError(f"perception server exited with {p.returncode}; see {log_dir / 'server.log'}")
        time.sleep(0.5)
    raise RuntimeError("perception server did not come up in 600 s")


def stop_server(p: subprocess.Popen, scratch: Path) -> int:
    from rowant_lab.libero_env import m2_bridge as BR

    try:
        cwd = os.getcwd()
        os.chdir(scratch)
        c = BR.Client("p.sock", timeout_s=30)
        BR.send(c.sock, {"op": "shutdown"})
        c.sock.close()
        os.chdir(cwd)
    except Exception:
        pass
    try:
        return p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p.terminate()  # our own child, by its handle (PID)
        return p.wait(timeout=30)


def hello_request(spec: dict, integ: dict) -> dict:
    """The worker's hello (plan venv_bridge.requests): protocol, venv versions, shared code and library sha256."""
    import numpy as np

    from rowant_lab.libero_env import m2_bridge as BR
    from rowant_lab.libero_env import m2_library as LB

    return {"op": "hello", "protocol": BR.PROTOCOL, "suites": sorted({e[0] for e in spec["episodes"]}),
            "client": {"python": sys.version.split()[0], "numpy": np.__version__},
            "code_sha256": {f: integ["modules"].get(f) for f in BR.SHARED_CODE},
            "library_sha256": {f"{MP.POLICY_LIB_VERSION}/manifest.json": sha256_file(
                LB.lfb_dirs((spec.get("params") or {}).get("lfb_library"))[1] / "manifest.json")},  # LFB-LIB
            "frozen_config_sha256": spec.get("frozen_config_sha256")}


def check_hello_reply(req: dict, rep: dict) -> list:
    """The client's side of the hello check: the server must match too (refuse otherwise)."""
    from rowant_lab.libero_env import m2_bridge as BR

    srv = rep.get("server") or {}
    bad = []
    if srv.get("protocol") != BR.PROTOCOL:
        bad.append(f"server protocol {srv.get('protocol')!r}")
    if srv.get("versions") != BR.EXPECTED_VERSIONS["monty"]:
        bad.append(f"server versions {srv.get('versions')}")
    if req["client"] != BR.EXPECTED_VERSIONS["venv-libero"]:
        bad.append(f"own versions {req['client']} != {BR.EXPECTED_VERSIONS['venv-libero']}")
    if srv.get("code_sha256") != req["code_sha256"]:
        bad.append("server shared code differs")
    if srv.get("library_sha256") != req["library_sha256"]:
        bad.append("server library manifest differs")
    if srv.get("frozen_config_sha256") != req["frozen_config_sha256"]:
        bad.append("server frozen config differs")
    return bad
