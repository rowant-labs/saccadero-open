"""M2 start-up integrity check (plan P6 "checked after start-up, as fl6 does"; verifier minor M2). Standard library.

M2 (perceived poses, grounded instruction): perception in the loop. Runs in both venvs (the perception server and
the episode workers). ``loaded_repo_modules()`` hashes the source file of every module in ``sys.modules`` that lives
in this worktree (package ``__init__`` files included: the fast layer's hybrid_frontier screens, e6 and e9 modules,
Monty's ``src`` modules, every libero_env module). ``check(role, expected)`` compares them with the frozen config's
``code_sha256``: a loaded repo module that is missing from the freeze, or whose sha256 differs, is a mismatch, and a
frozen run refuses to start (``IntegrityError``). Without a frozen config the record is kept for the run folder (and
for the freeze's coverage test). ``late(record)`` lists the repo modules loaded after the start-up check.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class IntegrityError(RuntimeError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def loaded_repo_modules(repo_root: Path = REPO_ROOT) -> dict:
    """{repo-relative source path: sha256} of every loaded module whose file lies in the worktree."""
    root = Path(repo_root).resolve()
    out = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        if p.suffix == ".pyc":  # a cached module: hash its source
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        if p.exists():
            out[rel.as_posix()] = _sha(p)
    return dict(sorted(out.items()))


def check(role: str, expected: dict | None = None, repo_root: Path = REPO_ROOT, refuse: bool = True) -> dict:
    mods = loaded_repo_modules(repo_root)
    rec = {"role": role, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_modules": len(mods),
           "modules": mods, "frozen": expected is not None}
    if expected is not None:
        missing = sorted(m for m in mods if m not in expected)
        differ = sorted(m for m in mods if m in expected and expected[m] != mods[m])
        rec.update(missing_from_freeze=missing, differ_from_freeze=differ, ok=not missing and not differ)
        if refuse and not rec["ok"]:
            raise IntegrityError(f"{role}: loaded repo modules differ from the frozen config: missing {missing[:8]}, "
                                 f"differ {differ[:8]}")
    else:
        rec["ok"] = None
    return rec


def late(rec: dict, repo_root: Path = REPO_ROOT) -> dict:
    """The repo modules loaded (or changed on disk) since ``rec`` was taken."""
    now = loaded_repo_modules(repo_root)
    return {m: h for m, h in now.items() if rec["modules"].get(m) != h}
