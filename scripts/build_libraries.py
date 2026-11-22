#!/usr/bin/env python3
"""Build the object libraries from your own copy of the LIBERO-Pro2.0 evaluation download.

The robot reads "object libraries": per object type its meshes, kinematics, collision boxes and grasp records, and
per table (workcell) the perception memory (surface nodes, colour memory, symmetry, templates, node grids), the
table's own geometry and the Panda's model. All of it is made from the benchmark's third-party assets, so none of
it is shipped with this repository: this command builds it on your machine from your download.

    python scripts/build_libraries.py --data DIR --benchmark DIR

``--benchmark`` is the unpacked download, the folder that holds ``simulator-source`` and ``service-source``, with
the assets archive unpacked into ``simulator-source/libero/libero/assets``. ``--data`` is the data folder
(``ROWANT_DATA``); the libraries are written under ``<data>/runs``. Both default to the settings
(``rowant_lab/settings.py``, ``rowant_lab/pro2/machine.py``), as do the two interpreters:

* the simulator environment (``--worker-python``; LIBERO's dependencies, robosuite, MuJoCo, torch) runs the export
  steps, which compile each base task's scene and read the compiled models;
* the perception environment (``--server-python``; numpy 2, scipy, scikit-image, OpenCV) runs everything else.

On a machine without a display, pass ``--gl egl`` (a GPU) or ``--gl osmesa`` (software), or set ``MUJOCO_GL``
yourself: the simulator steps open a render context, though no picture is used. The whole build takes about
45 minutes on one processor core; nearly all of it is the perception memory.

What is read from the download: the simulator's source and assets, its own base task files
(``simulator-source/libero/libero/bddl_files``, ten per suite) and start state 0 of each of those tasks. The
service's task catalogue and the perturbed task files are never opened.

The libraries were first built in four stages over a month, each stage adding to the tree before it. The steps
below replay those stages in order with the same builder modules (``--list`` prints them)::

    config                 LIBERO's config for the benchmark's simulator
    m1-export              Object and Spatial: collision models, table geometry, the jaw       (simulator)
    m1-records             grasp records of Object's and Spatial's object types
    m2-export              Object and Spatial: visual meshes, kinematics, workcells, Pandas    (simulator)
    m2-probe               the rendered geoms the mesh export leaves out                       (simulator)
    m2-export3             the export with those geoms added
    m2-policy              the documents the policy and the perception server load
    m2-onboard-*           the perception memory of the Spatial and Object workcells           (minutes each)
    goal-export/-policy/-onboard     the Goal suite's two new types and its workcell
    long-export/-policy/-onboard-*   the long-task suite's new types, its three tables, two more Pandas
    hand-model             the Panda hand model, rebuilt for comparison with the file in the repository
    check                  every workcell loaded through the library readers, every listed checksum checked
    manifest               <data>/runs/build-libraries/manifest.json: file, sha256, bytes

A finished step is detected by its last output and skipped, so the command can be re-run after an interruption;
the partial output of an unfinished step is removed first. ``--steps a,b`` runs a selection, ``--redo a,b`` removes
those steps' outputs and runs them again (later steps that copied from them are NOT redone automatically: redo
them too). Each step prints its duration and the files it wrote; its console output is kept under
``<data>/runs/build-libraries/logs``.

What the run reads afterwards: ``<data>/runs/pro2/long-lib/m1`` and ``.../m2``, the four Panda files
``<data>/runs/libero-m2/library/export-v3/panda-libero_{object,spatial}.{json,npz}`` and
``<data>/runs/libero-m1/library/robot.json``. The other folders are the stages' own trees and build records; the
later stages hold hard links to (or, where the file system has none, copies of) the earlier stages' files. The
libraries name textures and meshes by absolute path, so a built tree cannot be moved to another place: rebuild it
there. ``scripts/compare_libraries.py`` compares two built trees file by file.

Standard library only; any Python 3.9 or later can run the command itself.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
M2_SUITES = ("libero_spatial", "libero_object")  # in the order the first build onboarded them
GOAL_SUITE = "libero_goal"
LONG_CELLS = ("libero_long_kitchen", "libero_long_living_room", "libero_long_study")
PANDAS = tuple(f"panda-{s}.{ext}" for s in ("libero_object", "libero_spatial") for ext in ("json", "npz"))


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ child side
def child_setup(side: str) -> None:
    """Make this repository importable and, on the simulator side, put the benchmark's simulator and service source
    first on the path, so that they are the ``libero`` and ``libero_remote`` every builder imports (an installed
    LIBERO, if the environment has one, is shadowed)."""
    sys.path.insert(0, str(REPO))
    if side != "sim":
        return
    bench = Path(os.environ["ROWANT_BENCHMARK"])
    sys.path[0:0] = [str(bench / "simulator-source"), str(bench / "service-source")]
    spec = importlib.util.find_spec("libero")
    origin = Path(spec.origin).resolve() if spec and spec.origin else None
    if origin is None or not origin.is_relative_to((bench / "simulator-source").resolve()):
        raise SystemExit(f"REFUSE: 'libero' would be imported from {origin}, not from the benchmark's simulator")
    # LIBERO's config must name the simulator's OWN task files: the build never opens the service's task folder,
    # which also holds the perturbed and held-out task files
    cfg = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    own = f'bddl_files: "{bench / "simulator-source" / "libero" / "libero" / "bddl_files"}"'
    if cfg.exists() and own not in cfg.read_text().splitlines():
        raise SystemExit(f"REFUSE: {cfg} does not name the simulator's own task files ({own})")


def child_info() -> int:
    """The versions that decide what a build writes, as one JSON line."""
    out = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
           "MUJOCO_GL": os.environ.get("MUJOCO_GL"), "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM")}
    for name in ("numpy", "scipy", "skimage", "cv2", "mujoco", "robosuite", "torch", "numba"):
        if importlib.util.find_spec(name) is None:
            continue
        try:
            out[name] = getattr(importlib.import_module(name), "__version__", "?")
        except Exception as e:  # an import that fails here would fail in the step that needs it, with its own message
            out[name] = f"import failed: {type(e).__name__}"
    print(json.dumps(out, sort_keys=True))
    return 0


def child_check() -> int:
    """Load every workcell as the policy and the perception server do (every file they open is checked against the
    library's manifest), and check each onboarded workcell's files against its own file list."""
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.libero_env import m2_paths as MP
    from rowant_lab.pro2 import long_paths as LP
    from rowant_lab.pro2.machine import LONG_LIB

    onboard, policy, m1 = LB.lfb_dirs(LONG_LIB)
    extra = LP.extra_suites([GOAL_SUITE, LP.SUITE])
    reader = LB.LibraryReader(policy, m1, allow_panda=True, extra_suites=extra)
    bad = []
    for suite in list(MP.SUITES) + sorted(extra):
        lib = LB.M2Library(suite, lib_dir=policy, m1_library=m1, extra_suites=extra)
        panda = reader.panda(suite)
        objects = sorted(t for t in lib.types if lib.types[t]["kind"] == "object")
        missing = [t for t in objects if t not in lib.records]
        doc = json.loads((onboard / suite / "onboarding.json").read_text())
        wrong = [rel for rel, h in doc["files"].items() if sha256_file(onboard / suite / rel) != h]
        parts = [f"{r['type']}:{r['part']}" for _g, r in sorted(doc["per_part"].items(), key=lambda kv: int(kv[0]))]
        print(f"{suite}: {len(lib.types)} types, {len(lib.records)} with grasp records, {len(panda.bodies)} Panda "
              f"bodies, {len(parts)} onboarded parts, {len(doc['files'])} onboarded files checked", flush=True)
        if sorted({p.split(':')[0] for p in parts}) != sorted(lib.types):
            bad.append(f"{suite}: the onboarded types are not the workcell's")
        if missing:
            bad.append(f"{suite}: no grasp records for {missing}")
        if wrong:
            bad.append(f"{suite}: {wrong[:5]} differ from onboarding.json")
    if json.loads((MP.M1_LIBRARY / "robot.json").read_text()) != reader.robot():
        bad.append("libero-m1/library/robot.json is not the library's robot.json")
    for line in bad:
        print("FAIL:", line, file=sys.stderr)
    return 1 if bad else 0


def child_main(argv: list) -> int:
    side, module, args = argv[0], argv[1], argv[2:]
    child_setup(side)
    if module == "@info":
        return child_info()
    if module == "@check":
        return child_check()
    rc = importlib.import_module(module).main(args)
    return int(rc or 0)


# ------------------------------------------------------------------------------------------------ driver side
class Ctx:
    def __init__(self, a):
        self.data = Path(a.data).expanduser().resolve()
        self.bench = Path(a.benchmark).expanduser().resolve()
        self.worker, self.server = a.worker_python, a.server_python
        self.gl = a.gl
        self.runs = self.data / "runs"
        self.work = self.runs / "build-libraries"
        self.cfg = self.work / "libero-config"
        self.m1 = self.runs / "libero-m1" / "library"
        self.m2 = self.runs / "libero-m2" / "library"
        self.probe = self.runs / "libero-m2" / "build" / "export-v3-probe.json"
        self.goal = self.runs / "pro2" / "goal-lib"
        self.long = self.runs / "pro2" / "long-lib"
        self.hand = self.runs / "libero-m3" / "stereo1" / "panda_hand_model.json"

    def env(self, side: str) -> dict:
        env = dict(os.environ)
        env.update(ROWANT_DATA=str(self.data), ROWANT_BENCHMARK=str(self.bench), ROWANT_WORKER_PYTHON=self.worker,
                   ROWANT_SERVER_PYTHON=self.server)
        for k in THREAD_VARS:  # one thread: the perception memory was built single-threaded
            env[k] = "1"
        if side == "sim":
            env["LIBERO_CONFIG_PATH"] = str(self.cfg)
            env["NUMBA_CACHE_DIR"] = str(self.work / "numba-cache")
            if self.gl != "auto":  # headless rendering on Linux: egl (a GPU) or osmesa (software)
                env["MUJOCO_GL"] = env["PYOPENGL_PLATFORM"] = self.gl
        return env

    def python(self, side: str) -> str:
        return self.worker if side == "sim" else self.server


def link_tree(src: Path, dst: Path) -> None:
    """``dst`` as a copy of the folder ``src`` made of hard links (plain copies where the file system refuses a link),
    put in place by one rename so that an interrupted copy leaves nothing behind under ``dst``."""
    if dst.exists():
        return
    tmp = dst.with_name(dst.name + ".copying")
    shutil.rmtree(tmp, ignore_errors=True)

    def link(a, b):
        try:
            os.link(a, b)
        except OSError:
            shutil.copy2(a, b)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, tmp, copy_function=link)
    tmp.rename(dst)


def steps(c: Ctx) -> list:
    """The build, in order. Per step: name, side ('sim' or 'perc'), module, its arguments, the paths it creates
    (removed when the step is found unfinished) and its last output (present = the step is done)."""
    L, E = "rowant_lab.libero_env.", "rowant_lab.pro2."
    onb3, gb, lb = c.m2 / "onboard-v3", c.goal / "build", c.long / "build"
    out = [
        dict(name="config", side="sim", module=E + "long_lib", args=["config", "--out", str(c.cfg)],
             owns=[c.cfg], last=c.cfg / "config.yaml"),
        dict(name="m1-export", side="sim", module=L + "export_geom", args=[],
             owns=[c.m1 / "types", c.m1 / "scenes", c.m1 / "robot.json", c.m1 / "library-manifest.json"],
             last=c.m1 / "library-manifest.json"),
        dict(name="m1-records", side="perc", module=L + "build_records", args=[],
             owns=[c.m1 / "records"], last=c.m1 / "records" / "records-manifest.json"),
        dict(name="m2-export", side="sim", module=L + "m2_export", args=["--out", str(c.m2 / "export")],
             owns=[c.m2 / "export"], last=c.m2 / "export" / "manifest.json"),
        dict(name="m2-probe", side="sim", module=L + "m2_export3", args=["probe", "--out", str(c.probe)],
             owns=[c.probe], last=c.probe),
        # numpy only; run where the first build ran it (numpy 2 writes +0.0 where numpy 1 writes -0.0 in the added box)
        dict(name="m2-export3", side="perc", module=L + "m2_export3", args=["build", "--probe", str(c.probe)],
             owns=[c.m2 / "export-v3", c.m2 / "export-v3.tmp"], last=c.m2 / "export-v3" / "manifest.json"),
        dict(name="m2-policy", side="perc", module=L + "m2_libv3", args=["build"],
             owns=[c.m2 / "policy-v3", c.m2 / "policy-v3.tmp"], last=c.m2 / "policy-v3" / "manifest.json"),
    ]
    for s in M2_SUITES:
        out.append(dict(name="m2-onboard-" + s[len("libero_"):], side="perc", module=L + "m2_onboard",
                        args=["--suite", s], owns=[onb3 / s], last=onb3 / s / "onboarding.json"))
    gonb, lonb = c.goal / "m2" / "onboard-v3", c.long / "m2" / "onboard-v3"
    out += [
        dict(name="goal-export", side="sim", module=E + "goal_lib", args=["export", "--config", str(c.cfg)],
             owns=[gb / "export", gb / "m1types", gb / "scratch-shared", gb / "workcell.json", gb / "seconds.json",
                   gb / "export-report.json"], last=gb / "export-report.json"),
        dict(name="goal-policy", side="perc", module=E + "goal_lib", args=["policy"],
             owns=[c.goal / "m1", c.goal / "m2" / "policy-v3", c.goal / "m2" / "policy-v3.tmp"],
             last=c.goal / "m2" / "policy-v3" / "manifest.json"),
        # the earlier workcells are put in place first (links), so the builder's own copy step finds and checks them
        dict(name="goal-onboard", side="perc", module=E + "goal_lib", args=["onboard"],
             before=[(onb3 / s, gonb / s) for s in M2_SUITES],
             owns=[gonb / GOAL_SUITE, gb / "export-merged"], last=gonb / GOAL_SUITE / "onboarding.json"),
        dict(name="long-export", side="sim", module=E + "long_lib", args=["export"],
             owns=[lb / "export", lb / "m1types", lb / "scratch-shared", lb / "cells.json", lb / "seconds.json",
                   lb / "export-report.json", lb / "libero-config", lb / "numba-cache"],
             last=lb / "export-report.json"),
        dict(name="long-policy", side="perc", module=E + "long_lib", args=["policy"],
             owns=[c.long / "m1", c.long / "m1.tmp", c.long / "m2" / "policy-v3", c.long / "m2" / "policy-v3.tmp",
                   lb / "policy-report.json"], last=lb / "policy-report.json"),
    ]
    copies = [(onb3 / s, lonb / s) for s in M2_SUITES] + [(gonb / GOAL_SUITE, lonb / GOAL_SUITE)]
    for i, cell in enumerate(LONG_CELLS):  # the first one also gathers every cell type's export in one folder
        out.append(dict(name="long-onboard-" + cell[len("libero_long_"):].replace("_", "-"), side="perc",
                        module=E + "long_lib", args=["onboard", "--cell", cell], before=copies,
                        owns=[lonb / cell] + ([lb / "export-merged"] if i == 0 else []),
                        last=lonb / cell / "onboarding.json"))
    out += [
        dict(name="hand-model", side="perc", module=L + "m3_stereo1_handmodel", args=["write", "--out", str(c.hand)],
             owns=[c.hand], last=c.hand, after=hand_model_report),
        dict(name="check", side="perc", module="@check", args=[], owns=[], last=None),
        dict(name="manifest", side=None, module=None, args=[], owns=[], last=None),
    ]
    return out


def hand_model_report(c: Ctx) -> None:
    """Say whether the rebuilt hand model is the one shipped in the repository (the perception server reads the
    shipped file; the rebuilt one is for this comparison only)."""
    shipped = REPO / "rowant_lab" / "libero_env" / "evidence-m3" / "stereo1" / "panda_hand_model.json"
    if not shipped.is_file():
        print(f"  no shipped hand model at {shipped} to compare with")
        return
    a, b = json.loads(shipped.read_text()), json.loads(c.hand.read_text())
    same_file = shipped.read_bytes() == c.hand.read_bytes()
    same_geometry = a.get("bodies") == b.get("bodies") and a.get("frame") == b.get("frame")
    print(f"  shipped hand model: byte-identical {same_file}; same bodies, transforms, vertices and faces "
          f"{same_geometry}")


def path_problem(assets: Path) -> str | None:
    """Why a library built from this assets folder could not be loaded, or None. The library names each texture by
    its absolute path, and the library's readers refuse any document holding a string that reads as an instance
    name (a lower-case letter, ``_``, digits, then ``_`` or the end: ``.../libero_2_0/...``) or as a task reference.
    A download kept under such a folder name would build and then be refused, so it is refused here, early."""
    try:
        from rowant_lab.libero_env.m2_library import INSTANCE_NAME, TASK_REF
    except ImportError:  # this interpreter lacks the readers' dependencies: the m2-policy step makes the same check
        return None
    probe = f"{assets}/x/texture.png"
    if INSTANCE_NAME.search(probe) or TASK_REF.search(probe):
        return (f"the path {assets} holds a part that the library's readers take for an instance name or a task "
                f"reference (for example a folder named like 'name_2_x'); keep the download under another folder name")
    return None


def snapshot(root: Path) -> dict:
    out = {}
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("numba-cache", "logs", "steps")]
        for f in filenames:
            p = Path(dirpath) / f
            try:
                st = p.lstat()
            except OSError:
                continue
            out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns, st.st_ino)
    return out


def remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def is_done(c: Ctx, st: dict) -> bool:
    if st["last"] is not None:
        return st["last"].exists()
    return (c.work / "steps" / f"{st['name']}.json").exists()


def run_step(c: Ctx, st: dict) -> dict:
    partial = [p for p in st["owns"] if p.exists()]
    if partial:
        print(f"  removing the partial output of an unfinished run: {', '.join(str(p) for p in partial)}", flush=True)
        for p in partial:
            remove(p)
    for src, dst in st.get("before", []):
        link_tree(src, dst)
    before = snapshot(c.runs)
    t0, started = time.perf_counter(), now()
    if st["name"] == "manifest":
        rc, cmd = write_manifest(c), ["(this command)"]
    else:
        cmd = [c.python(st["side"]), "-I", "-B", str(Path(__file__).resolve()), "--child", st["side"], st["module"],
               *st["args"]]
        log = c.work / "logs" / f"{st['name']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w", encoding="utf-8") as lf:
            lf.write(" ".join(cmd) + "\n")
            proc = subprocess.Popen(cmd, env=c.env(st["side"]), cwd=str(c.work), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, errors="replace")
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                print("  | " + line.rstrip("\n"), flush=True)
            rc = proc.wait()
    seconds = round(time.perf_counter() - t0, 1)
    after = snapshot(c.runs)
    written = sorted(k for k, v in after.items() if before.get(k) != v)
    rec = {"step": st["name"], "returncode": rc, "seconds": seconds, "started_utc": started, "finished_utc": now(),
           "command": cmd, "files": [{"file": k, "bytes": after[k][0]} for k in written]}
    if rc == 0 and st.get("after"):
        st["after"](c)
    if rc == 0:
        (c.work / "steps").mkdir(parents=True, exist_ok=True)
        (c.work / "steps" / f"{st['name']}.json").write_text(json.dumps(rec, indent=1))
    return rec


def runtime_file(c: Ctx, rel: str) -> bool:
    """Is this file (relative to ``<data>/runs``) one a run reads?"""
    p = Path(rel)
    long_ = Path("pro2") / "long-lib"
    return (p.is_relative_to(long_ / "m1") or p.is_relative_to(long_ / "m2")
            or (p.parent == Path("libero-m2") / "library" / "export-v3" and p.name in PANDAS)
            or p == Path("libero-m1") / "library" / "robot.json")


def write_manifest(c: Ctx) -> int:
    """``manifest.json``: every file under the library folders with its sha256 and size, the files a run reads
    marked, the versions of both environments and the step records."""
    roots = [c.m1, c.runs / "libero-m2", c.runs / "libero-m3", c.goal, c.long]
    files = []
    for root in roots:
        for rel in sorted(snapshot(root)):
            p = root / rel
            r = str(p.relative_to(c.runs))
            files.append({"file": r, "sha256": sha256_file(p), "bytes": p.stat().st_size, "runtime": runtime_file(c, r)})
    versions = {}
    for side in ("sim", "perc"):
        cmd = [c.python(side), "-I", "-B", str(Path(__file__).resolve()), "--child", side, "@info"]
        p = subprocess.run(cmd, env=c.env(side), cwd=str(c.work), capture_output=True, text=True)
        try:
            versions[side] = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            versions[side] = {"error": (p.stderr or p.stdout)[-500:]}
    recs = []
    for f in sorted((c.work / "steps").glob("*.json")):
        d = json.loads(f.read_text())
        recs.append({k: d[k] for k in ("step", "seconds", "started_utc", "finished_utc")})
    rt = [f for f in files if f["runtime"]]
    doc = {"format": "object library build manifest v1", "written_utc": now(), "paths_relative_to": "<data>/runs",
           "n_files": len(files), "bytes": sum(f["bytes"] for f in files), "n_runtime_files": len(rt),
           "runtime_bytes": sum(f["bytes"] for f in rt), "versions": versions,
           "steps": sorted(recs, key=lambda r: r["started_utc"]), "files": files}
    c.work.mkdir(parents=True, exist_ok=True)
    (c.work / "manifest.json").write_text(json.dumps(doc, indent=1))
    print(f"  {c.work / 'manifest.json'}: {len(files)} files, {doc['bytes'] / 1e6:.1f} MB; "
          f"{len(rt)} of them read by a run, {doc['runtime_bytes'] / 1e6:.1f} MB", flush=True)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--child"]:
        return child_main(argv[1:])
    ap = argparse.ArgumentParser(description="Build the object libraries from the LIBERO-Pro2.0 evaluation download.",
                                 epilog="See the top of this file for what each step does.")
    ap.add_argument("--data", default=os.environ.get("ROWANT_DATA"), help="the data folder (ROWANT_DATA)")
    ap.add_argument("--benchmark", default=os.environ.get("ROWANT_BENCHMARK"),
                    help="the unpacked download: the folder holding simulator-source and service-source")
    ap.add_argument("--worker-python", default=None, help="the simulator environment's interpreter")
    ap.add_argument("--server-python", default=None, help="the perception environment's interpreter")
    ap.add_argument("--steps", default="all", help="comma-separated step names (default: all)")
    ap.add_argument("--redo", default="", help="comma-separated steps whose outputs are removed and built again")
    ap.add_argument("--gl", default="auto", choices=("auto", "egl", "osmesa"),
                    help="MuJoCo's rendering backend for the simulator steps (auto: leave the environment as it is; "
                         "on a headless Linux machine use egl with a GPU, osmesa without)")
    ap.add_argument("--list", action="store_true", help="print the steps and which are done, then stop")
    a = ap.parse_args(argv)
    if not a.data:
        ap.error("--data (or ROWANT_DATA) is required")
    os.environ["ROWANT_DATA"] = str(Path(a.data).expanduser().resolve())
    if a.benchmark:
        os.environ["ROWANT_BENCHMARK"] = str(Path(a.benchmark).expanduser().resolve())
    for flag, var in ((a.worker_python, "ROWANT_WORKER_PYTHON"), (a.server_python, "ROWANT_SERVER_PYTHON")):
        if flag:
            os.environ[var] = flag
    sys.path.insert(0, str(REPO))
    sys.dont_write_bytecode = True  # this command leaves no cache files in the repository
    from rowant_lab.pro2 import machine  # the settings: read after the flags are in the environment

    a.benchmark = str(machine.BENCHMARK)
    a.worker_python, a.server_python = machine.WORKER_PYTHON, machine.SERVER_PYTHON
    c = Ctx(a)
    todo = steps(c)
    names = [s["name"] for s in todo]
    chosen = names if a.steps == "all" else [s for s in a.steps.split(",") if s]
    redo = [s for s in a.redo.split(",") if s]
    unknown = sorted(set(chosen + redo) - set(names))
    if unknown:
        ap.error(f"unknown step(s) {unknown}; the steps are: {', '.join(names)}")
    if a.list:
        for s in todo:
            print(f"{s['name']:28s} {'done' if is_done(c, s) else '-':5s} {s['side'] or '':5s}")
        return 0
    sim = c.bench / "simulator-source" / "libero" / "libero"
    for need in (sim / "bddl_files", sim / "init_files", sim / "assets", c.bench / "service-source" / "libero_remote"):
        if not need.is_dir():
            print(f"REFUSE: {need} is missing: --benchmark must be the unpacked download with the assets archive "
                  f"unpacked into simulator-source/libero/libero/assets", file=sys.stderr)
            return 2
    for side in ("sim", "perc"):
        if not Path(c.python(side)).exists():
            print(f"REFUSE: no interpreter at {c.python(side)} (--worker-python / --server-python)", file=sys.stderr)
            return 2
    why = path_problem(sim / "assets")
    if why:
        print(f"REFUSE: {why}", file=sys.stderr)
        return 2
    c.work.mkdir(parents=True, exist_ok=True)
    print(f"data      {c.data}\nbenchmark {c.bench}\nsimulator environment  {c.worker}\nperception environment "
          f"{c.server}\n", flush=True)
    t_all, summary = time.perf_counter(), []
    for i, st in enumerate(todo):
        if st["name"] not in chosen and st["name"] not in redo:
            continue
        if st["name"] in redo:
            for p in st["owns"] + [c.work / "steps" / f"{st['name']}.json"]:
                remove(p)
        elif st["last"] is not None and is_done(c, st):
            print(f"[{st['name']}] done already ({st['last'].relative_to(c.data)} exists): skipped", flush=True)
            summary.append((st["name"], "skipped", 0.0, 0))
            continue
        undone = [p["name"] for p in todo[:i] if p["last"] is not None and not is_done(c, p)
                  and p["name"] != "hand-model"]
        if undone:
            print(f"REFUSE: [{st['name']}] needs the earlier step(s) {undone} (not done)", file=sys.stderr)
            return 2
        print(f"[{st['name']}] start {now()}", flush=True)
        rec = run_step(c, st)
        n_bytes = sum(f["bytes"] for f in rec["files"])
        for f in rec["files"]:
            print(f"  wrote {f['file']}  ({f['bytes']} B)")
        print(f"[{st['name']}] {'ok' if rec['returncode'] == 0 else 'FAILED (exit %d)' % rec['returncode']}: "
              f"{rec['seconds']} s, {len(rec['files'])} files, {n_bytes / 1e6:.1f} MB\n", flush=True)
        summary.append((st["name"], "ok" if rec["returncode"] == 0 else "FAILED", rec["seconds"], len(rec["files"])))
        if rec["returncode"] != 0:
            print(f"The step's output is in {c.work / 'logs' / (st['name'] + '.log')}. Re-run the same command to "
                  f"try it again; finished steps are skipped.", file=sys.stderr)
            break
    print("step                         result   seconds  files")
    for name, res, secs, n in summary:
        print(f"{name:28s} {res:8s} {secs:7.1f}  {n:5d}")
    print(f"total {time.perf_counter() - t_all:.1f} s")
    return 0 if all(r[1] != "FAILED" for r in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
