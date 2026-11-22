"""M2 runtime checks: the perception server's audit hook, the hello gate and check, the camera's output check and
the audit's own rules.

M2 (perceived poses, grounded instruction): perception in the loop. Monty venv; nothing here imports LIBERO, MuJoCo
or torch:

* the server's audit hook (in a subprocess: a hook cannot be removed) refuses to open a run folder, M1's scenes, the
  init states, the evidence folders and the export's raw types, refuses a listing of a run folder and an AF_INET
  socket, and allows the library and its own log folder;
* the hello gate, the hello check, the camera output check and the audit's own rules.

The two tests that need the built object library skip when it is absent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

from rowant_lab.libero_env import m2_audit as AU
from rowant_lab.libero_env import m2_bridge as BR
from rowant_lab.libero_env import m2_camera as CAM
from rowant_lab.libero_env import m2_rpaths as MP

REPO = MP.REPO_ROOT


# ------------------------------------------------------------------------------------------------ the audit hook
HOOK_PROBE = r"""
import json, socket, sys, os
from pathlib import Path
from rowant_lab.libero_env import m2_perceive as PR
from rowant_lab.libero_env import m2_rpaths as R
from rowant_lab.libero_env import m1_paths as M1
log, scr, outside = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
PR.install_audit_hook(log, scr)
out = {}
def tryopen(label, p, mode="r"):
    try:
        with open(p, mode) as f:
            if "w" in mode or "a" in mode:
                f.write("x")
            else:
                f.read(1)
        out[label] = "opened"
    except PermissionError as e:
        out[label] = "refused"
    except OSError as e:
        out[label] = "oserror:" + type(e).__name__
tryopen("run_folder", R.DEV_RUNS / "d04-frozen" / "run.json")
tryopen("run_worker_rows", R.DEV_RUNS / "d04-frozen" / "worker-0.jsonl")
tryopen("m1_scenes", M1.LIBRARY_DIR / "scenes" / "libero_object" /
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket.json")
tryopen("init_states", M1.DEV_DIR / "libero_object" / "0.npy")
tryopen("init_manifest", M1.INIT_MANIFEST)
tryopen("evidence_m2", R.EVIDENCE / "plan.json")
tryopen("export_raw_type", R.EXPORT_DIR / "types" / "wooden_cabinet.json")
tryopen("m1_raw_type", R.M1_LIBRARY / "types" / "wooden_cabinet.json")
tryopen("write_outside", outside / "hook-probe.txt", "w")
tryopen("policy_lib", R.POLICY_LIB / "manifest.json")
tryopen("onboard", R.ONBOARD_DIR / "libero_object" / "onboarding.json")
tryopen("own_log_write", log / "probe.txt", "w")
try:
    os.listdir(R.DEV_RUNS); out["list_runs"] = "listed"
except PermissionError:
    out["list_runs"] = "refused"
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.connect(("127.0.0.1", 9)); out["inet"] = "connected"
except PermissionError:
    out["inet"] = "refused"
except OSError:
    out["inet"] = "oserror"
try:
    import subprocess; subprocess.run(["/bin/echo", "x"], capture_output=True); out["subprocess"] = "ran"
except PermissionError:
    out["subprocess"] = "refused"
print(json.dumps(out))
"""


@pytest.mark.skipif(not MP.POLICY_LIB.exists(), reason="the object library is not built on this machine")
def test_server_audit_hook_refuses_truth_reads(tmp_path):
    log, scr = tmp_path / "log", tmp_path / "scr"
    log.mkdir()
    scr.mkdir()
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1",
               NUMBA_CACHE_DIR=str(MP.NUMBA_CACHE))
    outside = MP.SCRATCH / "hook-probe-outside"  # not an allowed root: a write there must be refused
    outside.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([sys.executable, "-P", "-c", HOOK_PROBE, str(log), str(scr), str(outside)],
                       capture_output=True, text=True, env=env, cwd=scr, timeout=120)
    assert not (outside / "hook-probe.txt").exists()
    try:
        outside.rmdir()
    except OSError:
        pass
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    for k in ("run_folder", "run_worker_rows", "m1_scenes", "init_states", "init_manifest", "evidence_m2",
              "export_raw_type", "m1_raw_type", "write_outside", "list_runs", "inet", "subprocess"):
        assert out[k] == "refused", (k, out[k])
    for k in ("policy_lib", "onboard", "own_log_write"):
        assert out[k] == "opened", (k, out[k])


# ------------------------------------------------------------------------------------------------ hello
def _serve_in(tmp_path, handler, **kw):
    import time as _t

    os.chdir(tmp_path)
    th = threading.Thread(target=BR.serve, args=("p.sock", handler), kwargs=kw, daemon=True)
    th.start()
    for _ in range(500):
        if (tmp_path / "p.sock").exists():
            break
        _t.sleep(0.01)
    return th


def _connect(tries: int = 200):
    import time as _t

    for _ in range(tries):
        try:
            return BR.Client("p.sock", timeout_s=10)
        except (ConnectionRefusedError, FileNotFoundError):
            _t.sleep(0.01)
    raise AssertionError("no connection")


def test_hello_gate(tmp_path):
    def handler(msg):
        if msg["op"] == "hello" and msg.get("protocol") != BR.PROTOCOL:
            raise ValueError("refused")
        return {"ok": True}

    th = _serve_in(tmp_path, handler, require_hello=True)
    c = _connect()
    with pytest.raises(BR.RemoteError, match="HelloRequired"):
        c.call({"op": "look"})
    with pytest.raises(BR.RemoteError):
        c.call({"op": "hello", "protocol": "old"})
    with pytest.raises(BR.RemoteError, match="HelloRequired"):
        c.call({"op": "begin_episode", "episode": "0" * 16})
    assert c.call({"op": "hello", "protocol": BR.PROTOCOL})["ok"]
    assert c.call({"op": "look"})["ok"]
    c2 = _connect()  # a second connection is gated on its own
    with pytest.raises(BR.RemoteError, match="HelloRequired"):
        c2.call({"op": "look"})
    BR.send(c.sock, {"op": "shutdown"})
    c.sock.close()
    c2.sock.close()
    th.join(timeout=10)


@pytest.mark.skipif(not MP.POLICY_LIB.exists(), reason="the object library is not built on this machine")
def test_server_hello_check_refuses_mismatches():
    from rowant_lab.libero_env import m2_integrity as INT
    from rowant_lab.libero_env import m2_perceive as PR

    per = PR.Perceiver()
    mods = INT.loaded_repo_modules()
    good = {"op": "hello", "protocol": BR.PROTOCOL, "suites": ["libero_object"],
            "client": dict(BR.EXPECTED_VERSIONS["venv-libero"]),
            "code_sha256": {f: mods.get(f) for f in BR.SHARED_CODE}, "library_sha256": per.library_sha256(),
            "frozen_config_sha256": None}
    if PR.own_versions() == BR.EXPECTED_VERSIONS["monty"]:
        assert per.check_hello(good)["protocol"] == BR.PROTOCOL
    bad = [dict(good, protocol="rowant-m2-bridge-v1"), dict(good, client={"python": "3.11.11", "numpy": "2.4.6"}),
           dict(good, code_sha256={**good["code_sha256"], "rowant_lab/libero_env/m2_kin.py": "0" * 64}),
           dict(good, library_sha256={f"{MP.POLICY_LIB_VERSION}/manifest.json": "0" * 64}),
           dict(good, frozen_config_sha256="cd146eaa" + "0" * 56), dict(good, suites=["libero_10"])]  # Goal is served now
    for b in bad:
        with pytest.raises(PR.HelloRefused):
            per.check_hello(b)


# ------------------------------------------------------------------------------------------------ camera, token, audit
def _frame(W=16):
    return {"camera": "agentview", "rgb": np.zeros((W, W, 3), np.uint8), "depth_u16": np.zeros((W, W), np.uint16),
            "depth_scale_mm": 0.1, "K": CAM.intrinsics(45.0, W, W), "T_world_from_camera": np.eye(4),
            "width": W, "height": W, "step": 0}


def test_camera_output_check():
    assert CAM.check_frame(_frame(), 16)["keys"] == list(CAM.FRAME_KEYS)
    for mutate in (lambda f: f.update(segmentation=np.zeros((16, 16), np.int32)),
                   lambda f: f.pop("K"),
                   lambda f: f.update(rgb=np.zeros((16, 16, 3), np.float32)),
                   lambda f: f.update(depth_u16=np.zeros((16, 16), np.float32)),
                   lambda f: f.update(depth_u16=np.zeros((8, 16), np.uint16)),
                   lambda f: f.update(K=np.eye(3)),
                   lambda f: f.update(T_world_from_camera=np.full((4, 4), np.nan)),
                   lambda f: f.update(width=8),
                   lambda f: f.update(step=1.5),
                   lambda f: f.update(depth_scale_mm=1.0)):
        f = _frame()
        mutate(f)
        with pytest.raises(CAM.CameraOutputError):
            CAM.check_frame(f, 16)


def test_audit_rules():
    out: list = []
    AU.walk({"a": {"goal": 1}, "b": [{"ketchup_1_pos": [0.0]}], "c": lambda: 0}, "x", (), out)
    assert any("goal" in v for v in out) and any("ketchup_1_pos" in v for v in out) and any("callable" in v for v in out)

    class Inner:
        def call(self, msg):
            return {"ok": True}

    ac = AU.AuditedClient(Inner())
    ac.call({"op": "begin_episode", "episode": "libero_object:0:0"})
    ac.call({"op": "look", "mode": "full", "episode": "0" * 16, "look": 0, "suite": "libero_object",
             "cameras": [{"rgb": 0}], "joint_pos": [], "gripper_qpos": [], "truth": 1})
    rec = ac.record()
    assert not rec["ok"] and len(rec["violations"]) >= 3


def test_numba_cache_is_pinned():
    p = subprocess.run([sys.executable, "-P", "-c",
                        "import os; os.environ['NUMBA_CACHE_DIR']='/elsewhere';"
                        "from rowant_lab.libero_env import m2_rpaths as R; print(R.pin_numba_cache());"
                        "print(os.environ['NUMBA_CACHE_DIR'])"],
                       capture_output=True, text=True, timeout=60,
                       env=dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1"))
    assert p.returncode == 0, p.stderr
    assert p.stdout.split() == [str(MP.NUMBA_CACHE)] * 2
    for f in ("m2_run.py", "m2_export.py"):
        assert "MP.pin_numba_cache()" in (MP.PACKAGE_DIR / f).read_text(), f


