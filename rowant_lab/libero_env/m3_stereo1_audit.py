"""K0-stereo audit hook, registered roots, the split function and provenance (standard library only).

Registered in the M3 dev targets (``k0_stereo``: methods.audit_hook, capture.roots,
scenes.split_enforcement, integrity.provenance) and the K0-stereo design note (section 2 split
enforcement, 3.10 roots, 5 run-time allowlist, 10.2 FIX-COMMIT). Importable from venv-libero (``-I -B``) and from the
Monty venv; it imports only the standard library, and ``subprocess`` only inside the git helpers (never at import),
so a strict process that imports this module for its hook never imports ``subprocess`` through it.

What it holds:

* ``install(root, process, kind, arm=...)``: ``sys.addaudithook`` in strict mode (process kinds ``stereo``, ``P-O``,
  ``P-S``) or logging mode (``render``, ``eval``, ``launch``, ``P-T``, ``poses-pick``, ``settled-pose``,
  ``classify``). Strict mode raises ``AuditViolation`` on any ``open``, ``os.listdir``, ``os.scandir`` or
  ``glob.glob`` outside the kind's allowlist (``allowlist``), on every process-spawning event (``subprocess.Popen``,
  ``os.system``, ``os.exec``, ``os.posix_spawn``, ``os.spawn``, ``os.fork``, ``os.forkpty``) and on every network
  event (``socket.*`` and the stdlib clients' connect events), and on every deletion (the writer never deletes).
  Logging mode logs every path and mode and raises only on a write outside the process's root guard (its registered
  root, plus the caller's declared extra write folders; never under K0's ``depth1d``). Every audited event is one row
  ``{event, path, mode, allowed}`` in ``R/audit/<process>-<pid>.jsonl``; at install one ``{event: "module", name,
  file, sha256}`` row per module already in ``sys.modules``, then one ``{event: "import", name, file}`` row per later
  import, and at exit a ``module`` row (``phase: "exit"``) for every module imported after install. A violation that
  the caller swallowed still ends a strict process with exit code ``VIOLATION_EXIT`` (70). ``install_hook(root,
  process, arm, mode, extra_read(s))`` is the same call by process name (stereo, selfcal, po, ...); in strict mode an
  extra read is accepted only for the registered hand model, or on a synthetic root.
* ``root_ok(path, cli)``: the single root guard of every array writer (render, pose, stereo, S6, eval): R2, a fresh
  ``stereo1-smoke/<n>``, ``stereo1-drift`` (eval rows only), a re-render root ``stereo1-<letters>`` named in a K0S
  amendment, or a synthetic test root holding a ``SYNTHETIC`` marker (outside the M3 runs folder and the repo); it
  refuses K0's ``depth1d`` (and anything inside it) for every CLI and ``stereo1-drift`` for every CLI except the eval.
  The old temp-folder allowance (m3_depth1d_render.py:1170-1172) does not exist here.
* ``split_of(scene_id)``: tune (measure states 0-1), score (2-9), held_out (libero_goal / libero_10 states 0-1) and
  smoke (exactly the three smoke scenes); ``in_diagnostics`` (score states 2-5), ``tune_like`` (tune or smoke).
* ``code_tree(repo, commit)`` and ``clean(repo, commit)`` (FIX-COMMIT): the sha256 of the ``git ls-tree -r`` lines
  over ``rowant_lab/`` minus ``rowant_lab/libero_env/evidence-m3/**``, ``rowant_lab/hybrid_frontier/notes/**`` and
  every ``*.md``; clean = no ``git status --porcelain --untracked-files=all`` entry inside those paths and
  ``code_tree(HEAD) == code_tree(commit)``. Git runs with ``GIT_OPTIONAL_LOCKS=0`` (status never rewrites the index).
  ``read_head_stdlib`` is the test-only .git reader (the worktree's own ``<gitdir>/HEAD``).
* ``is_ancestor`` / ``c4_landed``: the C4 ancestry test (``git merge-base --is-ancestor <C4> origin/rowant/
  hybrid-frontier-20261012``); ``c4_landed`` also requires the C4 commit to carry K0S-PICK and the two pick copies.
* ``write_new`` (O_EXCL, never overwrites, never deletes) and ``append_jsonl``.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from rowant_lab.settings import DATA

FORMAT = "rowant m3 stereo1 audit v1"
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
DEV_TARGETS = PACKAGE_DIR / "evidence-m3" / "dev-targets-m3v2.json"
EVIDENCE_DIR = PACKAGE_DIR / "evidence-m3" / "stereo1"
HAND_MODEL = EVIDENCE_DIR / "panda_hand_model.json"

M3_RUNS = DATA / "runs" / "libero-m3"
R2 = M3_RUNS / "stereo1"
SMOKE_PARENT = M3_RUNS / "stereo1-smoke"
DRIFT_ROOT = M3_RUNS / "stereo1-drift"
DEPTH1D = M3_RUNS / "depth1d"
SYNTHETIC_MARKER = "SYNTHETIC"
ORIGIN_REF = "origin/HEAD"  # (as run: a branch of the private repository this code was carved from)
C4_ENV = "ROWANT_C4_COMMIT"
CODE_COMMIT_ENV = "ROWANT_CODE_COMMIT"
VIOLATION_EXIT = 70

CODE_TREE_PREFIX = "rowant_lab/"
CODE_TREE_EXCLUDE_PREFIXES = ("rowant_lab/libero_env/evidence-m3/", "rowant_lab/hybrid_frontier/notes/")
CODE_TREE_EXCLUDE_SUFFIXES = (".md",)
C4_REQUIRED_FILES = ("rowant_lab/libero_env/evidence-m3/stereo1/pick.json",
                     "rowant_lab/libero_env/evidence-m3/stereo1/poses_pick.json")
C4_AMENDMENT = "K0S-PICK"
DEV_TARGETS_REL = "rowant_lab/libero_env/evidence-m3/dev-targets-m3v2.json"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ===================================================================================================== splits
MEASURE_SUITES = ("libero_object", "libero_spatial")
HELD_OUT_SUITES = ("libero_goal", "libero_10")
TASKS = tuple(range(10))
TUNE_STATES = (0, 1)
SCORE_STATES = tuple(range(2, 10))
DIAGNOSTIC_STATES = (2, 3, 4, 5)
HELD_OUT_STATES = (0, 1)
# scenes.smoke: libero_object task 0 and libero_spatial task 0 at init state 1 (E0, E1, E2) and the held-out-suite
# survey on libero_goal task 0 init state 2 (outside the held-out set): split_of returns smoke for exactly these
SMOKE_SCENES = ("libero_object-t00-s01", "libero_spatial-t00-s01", "libero_goal-t00-s02")
SPLITS = ("tune", "score", "held_out", "smoke")
_SCENE_RE = re.compile(r"^(libero_object|libero_spatial|libero_goal|libero_10)-t(\d{2})-s(\d{2})$")


def parse_scene_id(scene_id: str) -> tuple:
    """``(suite, task, init_state)`` of ``f"{suite}-t{task:02d}-s{state:02d}"``; ValueError otherwise."""
    m = _SCENE_RE.match(str(scene_id))
    if not m:
        raise ValueError(f"not a K0-stereo scene id: {scene_id!r}")
    return m.group(1), int(m.group(2)), int(m.group(3))


def split_of(scene_id: str) -> str:
    """The one split function every CLI applies (scenes.split_enforcement). Raises ValueError for any scene that is
    not registered (a task outside 0-9, a measure state outside 0-9, a held-out state outside 0-1 except the smoke
    survey scene)."""
    if scene_id in SMOKE_SCENES:
        return "smoke"
    suite, task, state = parse_scene_id(scene_id)
    if task not in TASKS:
        raise ValueError(f"{scene_id}: task {task} is not registered")
    if suite in MEASURE_SUITES:
        if state in TUNE_STATES:
            return "tune"
        if state in SCORE_STATES:
            return "score"
    elif suite in HELD_OUT_SUITES and state in HELD_OUT_STATES:
        return "held_out"
    raise ValueError(f"{scene_id}: init state {state} is not registered for {suite}")


def in_diagnostics(scene_id: str) -> bool:
    """The diagnostics subset (score states 2-5): E2 settled look, F_A_diag, the photometric arm."""
    return split_of(scene_id) == "score" and parse_scene_id(scene_id)[2] in DIAGNOSTIC_STATES


def tune_like(scene_id: str) -> bool:
    """Tune, or a smoke scene (treated like tune by every CLI; never scored, never in a line)."""
    return split_of(scene_id) in ("tune", "smoke")


def has_e2(scene_id: str) -> bool:
    """Scenes with an E2 episode: the diagnostics subset, tune states 0-1 (pipeline checks only) and the two measure
    smoke scenes (not the held-out-suite survey scene)."""
    sp = split_of(scene_id)
    if sp == "smoke":
        return parse_scene_id(scene_id)[0] in MEASURE_SUITES
    return sp == "tune" or in_diagnostics(scene_id)


# ===================================================================================================== roots
CLIS = ("render", "poses", "stereo", "selfcal", "eval")
CLI_ALIASES = {"pt": "poses", "po": "poses", "poses-pick": "poses", "settled-pose": "poses", "P-T": "poses",
               "P-O": "poses", "P-S": "selfcal", "ps": "selfcal", "S6": "selfcal", "match": "stereo",
               "sgbm": "stereo", "launch": "eval", "classify": "eval"}
DRIFT_CLIS = ("eval",)  # stereo1-drift holds DRIFT-1's eval rows only


class RootRefused(ValueError):
    """A root that is not one of the registered roots for this CLI."""


def _real(p) -> str:
    return os.path.realpath(os.fsdecode(os.fspath(p)))


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _cli(cli: str) -> str:
    c = CLI_ALIASES.get(cli, cli)
    if c not in CLIS:
        raise ValueError(f"unknown CLI {cli!r} (one of {CLIS} or {sorted(CLI_ALIASES)})")
    return c


_AMENDED: dict = {}


def amended_roots(dev_targets=None) -> tuple:
    """Re-render roots ``stereo1-<letters>`` named in a K0S amendment (id, field or text) of the dev targets; read
    once per process and cached (a strict process calls ``install`` before its hook exists)."""
    p = _real(dev_targets or DEV_TARGETS)
    if p not in _AMENDED:
        names = set()
        try:
            with open(p, "rb") as f:
                doc = json.loads(f.read().decode("utf-8"))
        except (OSError, ValueError):
            doc = {}
        for a in doc.get("amendments", []) if isinstance(doc, dict) else []:
            if not (isinstance(a, dict) and str(a.get("id", "")).startswith("K0S")):
                continue
            blob = " ".join(str(a.get(k, "")) for k in ("id", "field", "text"))
            for m in re.finditer(r"(?<![A-Za-z0-9_-])stereo1-([a-z]+)(?![A-Za-z0-9_-])", blob):
                if m.group(1) not in ("smoke", "drift"):
                    names.add(m.group(0))
        _AMENDED[p] = tuple(sorted(_real(M3_RUNS / n) for n in names))
    return _AMENDED[p]


def root_kind(path, cli: str, dev_targets=None) -> tuple:
    """``(kind, reason)``: kind in {R2, smoke, drift, rerender, synthetic} when ``path`` is a registered root for
    ``cli``, else ``(None, reason)``."""
    c = _cli(cli)
    p = _real(path)
    m3 = _real(M3_RUNS)
    if _inside(p, _real(DEPTH1D)):
        return None, "depth1d is K0's root: read-only for every CLI"
    if p == _real(R2):
        return "R2", ""
    if os.path.dirname(p) == _real(SMOKE_PARENT):
        if re.fullmatch(r"0|[1-9][0-9]*", os.path.basename(p)):
            return "smoke", ""
        return None, "a smoke root is stereo1-smoke/<n> with n an integer"
    if p == _real(DRIFT_ROOT):
        if c in DRIFT_CLIS:
            return "drift", ""
        return None, f"stereo1-drift holds DRIFT-1 eval rows only (refused for {c})"
    if p in amended_roots(dev_targets):
        return "rerender", ""
    marker = os.path.join(p, SYNTHETIC_MARKER)
    if os.path.isfile(marker) and not os.path.islink(marker):
        if _inside(p, m3):
            return None, "a SYNTHETIC marker inside the M3 runs folder is not a synthetic root"
        if _inside(p, _real(REPO_ROOT)):
            return None, "a synthetic root is never inside the repo"
        return "synthetic", ""
    return None, "not a registered root (R2, stereo1-smoke/<n>, stereo1-drift, an amended re-render root, or a " \
                 "synthetic root with a SYNTHETIC marker)"


def root_ok(path, cli: str, dev_targets=None) -> bool:
    return root_kind(path, cli, dev_targets)[0] is not None


def check_root(path, cli: str, force: bool = False, dev_targets=None) -> str:
    """Raise ``RootRefused`` unless ``path`` is a registered root for ``cli``; ``--force`` is refused for every
    registered root. Returns the root's kind."""
    kind, why = root_kind(path, cli, dev_targets)
    if kind is None:
        raise RootRefused(f"{path}: {why}")
    if force:
        raise RootRefused(f"{path}: --force is refused for every registered root")
    return kind


# ===================================================================================================== allowlists
STRICT_KINDS = ("stereo", "P-O", "P-S")
LOGGING_KINDS = ("render", "eval", "launch", "P-T", "poses-pick", "settled-pose", "classify")
KIND_CLI = {"stereo": "stereo", "P-O": "poses", "P-S": "selfcal", "render": "render", "eval": "eval",
            "launch": "eval", "P-T": "poses", "poses-pick": "poses", "settled-pose": "poses", "classify": "eval"}
_ARM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
_PROC_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class AuditViolation(RuntimeError):
    """An audited event outside the process's allowlist (a RuntimeError, so ``except OSError`` cannot hide it)."""


def interpreter_dirs() -> tuple:
    """The interpreter's and the venv's install trees (realpaths): every read there is a code read."""
    out = []
    for p in (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix,
              os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))):
        if p:
            rp = _real(p)
            if rp not in out and rp != os.sep:
                out.append(rp)
    return tuple(out)


class Policy:
    """A process kind's allowlist: reads (directory prefixes and single files), writes, mkdir ancestors, and the code
    trees. Paths are realpaths; strict kinds raise outside them, logging kinds raise only on writes outside. A strict
    process never reads or lists its write folders (R/net/** holds other arms' maps); os.devnull is always writable."""

    def __init__(self, kind: str, root, arm=None, repo=None, hand_model=None, extra_read=(), extra_write=()):
        if kind not in STRICT_KINDS + LOGGING_KINDS:
            raise ValueError(f"unknown process kind {kind!r}")
        self.kind = kind
        self.strict = kind in STRICT_KINDS
        self.root = _real(root)
        self.arm = arm
        self.repo = _real(repo or REPO_ROOT)
        self.hand_model = _real(hand_model or HAND_MODEL)
        self.code_dirs = interpreter_dirs()
        self.deny_write = (_real(DEPTH1D),)
        R = self.root
        if self.strict:
            if not (isinstance(arm, str) and _ARM_RE.match(arm)) or arm in (".", ".."):
                raise ValueError(f"a {kind} process needs an arm name without '/': {arm!r}")
            own = os.path.join(R, "poses", arm)
            self.read_dirs = (os.path.join(R, "frames"), own)
            files = [self.hand_model]
            if kind == "P-O":
                files.append(os.path.join(R, "eval", "poses_pick.json"))
            self.read_files = tuple(files)
            self.write_dirs = (os.path.join(R, "net"), own, os.path.join(R, "audit"))
        else:
            self.read_dirs = ()
            self.read_files = ()
            self.write_dirs = (R,) + tuple(_real(p) for p in extra_write)
        self.read_dirs = self.read_dirs + tuple(_real(p) for p in extra_read)
        for w in self.write_dirs:
            if any(_inside(w, d) for d in self.deny_write):
                raise ValueError(f"{w}: never a write root (K0's depth1d)")
        anc = set()
        for w in self.write_dirs:
            if _inside(w, R):
                q = w
                while _inside(q, R) and q != R:
                    q = os.path.dirname(q)
                    anc.add(q)
        self.mkdir_ok = tuple(sorted(anc))

    def as_row(self) -> dict:
        return {"kind": self.kind, "strict": self.strict, "root": self.root, "arm": self.arm,
                "read_dirs": list(self.read_dirs), "read_files": list(self.read_files),
                "write_dirs": list(self.write_dirs), "code_dirs": list(self.code_dirs), "repo": self.repo}

    # ------------------------------------------------------------------------------------------- predicates
    def _code_read(self, p: str) -> bool:
        if any(_inside(p, d) for d in self.code_dirs):
            return True
        if not _inside(p, self.repo):
            return False
        if p.endswith(".py"):
            return True
        # Python reads an existing __pycache__/*.pyc of a repo module even under -B (it only never writes one)
        return p.endswith(".pyc") and os.path.basename(os.path.dirname(p)) == "__pycache__"

    def _code_dir(self, p: str) -> bool:
        if any(_inside(p, d) for d in self.code_dirs):
            return True
        if not _inside(p, self.repo):
            return False
        if os.path.isfile(os.path.join(p, "__init__.py")):
            return True
        return p in {_real(x) for x in sys.path if isinstance(x, str) and x}

    def can_read(self, p: str) -> bool:
        if not self.strict:
            return True
        if p in self.read_files or any(_inside(p, d) for d in self.read_dirs):
            return True
        return self._code_read(p)

    def can_list(self, p: str) -> bool:
        if not self.strict:
            return True
        if any(_inside(p, d) for d in self.read_dirs):
            return True
        return self._code_dir(p)

    def can_write(self, p: str) -> bool:
        if p == os.devnull:
            return True
        if any(_inside(p, d) for d in self.deny_write):
            return False
        return any(_inside(p, d) for d in self.write_dirs)

    def can_mkdir(self, p: str) -> bool:
        return self.can_write(p) or p in self.mkdir_ok


_O_WRITE = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC | getattr(os, "O_EXCL", 0)
_EXEC_EVENTS = frozenset(("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
                          "os.forkpty", "os.startfile", "pty.spawn"))
_NET_PREFIXES = ("socket.", "urllib.", "http.client.", "ftplib.", "smtplib.", "poplib.", "imaplib.", "nntplib.",
                 "telnetlib.", "webbrowser.")
_DELETE_EVENTS = frozenset(("os.remove", "os.rmdir", "shutil.rmtree"))
_WRITE_PATH_EVENTS = frozenset(("os.chmod", "os.chown", "os.truncate", "os.utime", "os.chflags", "os.setxattr",
                                "os.removexattr"))
_TWO_PATH_EVENTS = frozenset(("os.rename", "os.link", "os.symlink", "shutil.copyfile", "shutil.copymode",
                              "shutil.copystat", "shutil.copytree", "shutil.move", "shutil.copyxattr"))
_LIST_EVENTS = frozenset(("os.listdir", "os.scandir"))
_GLOB_EVENTS = frozenset(("glob.glob", "glob.glob/2"))
_MAGIC = re.compile(r"[*?[]")


def _abs(path, dir_fd=None):
    """Absolute realpath of an event's path, or None when it is a file descriptor or relative to a dir_fd."""
    if path is None:
        path = "."
    if isinstance(path, int):
        return None
    try:
        s = os.fsdecode(os.fspath(path))
    except TypeError:
        return None
    if not os.path.isabs(s):
        if dir_fd not in (None, -1):
            return None
        s = os.path.join(os.getcwd(), s)
    return os.path.realpath(s)


def _open_is_write(mode, flags) -> bool:
    if isinstance(mode, str) and mode:
        return any(c in mode for c in "wax+")
    try:
        return bool(int(flags or 0) & _O_WRITE)
    except (TypeError, ValueError):
        return True


def _glob_base(pattern, root_dir=None):
    s = os.fsdecode(os.fspath(pattern)) if pattern is not None else ""
    if root_dir is not None and not os.path.isabs(s):
        s = os.path.join(os.fsdecode(os.fspath(root_dir)), s)
    parts = []
    for part in s.split(os.sep):
        if _MAGIC.search(part):
            break
        parts.append(part)
    base = os.sep.join(parts) if parts else "."
    if s.startswith(os.sep) and not base:
        base = os.sep
    return _abs(base or ".")


def decide(policy: Policy, event: str, args: tuple):
    """``(allowed, row)`` for one audit event under ``policy``; ``row`` is None for an event that is not audited.
    Pure apart from ``os.getcwd``/``realpath``/``isfile``: the hook and the tests both call it."""
    strict = policy.strict
    if event in ("os.chdir", "os.fchdir"):
        # strict: the cwd never moves, so a relative path (and a dir_fd-relative open, whose dir_fd the open event
        # does not carry) cannot be resolved against a folder other than the one the hook resolves it against
        what = args[0] if args else None
        return (not strict), {"event": event, "path": repr(what)[:300], "mode": "chdir", "allowed": not strict}
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        write = _open_is_write(mode, flags)
        raw = mode if isinstance(mode, str) else f"flags={int(flags or 0):#x}"
        if strict and not isinstance(path, int):
            try:
                rel = not os.path.isabs(os.fsdecode(os.fspath(path)))
            except TypeError:
                rel = True
            if rel:  # strict: absolute paths only (the open event carries no dir_fd)
                return False, {"event": event, "path": repr(path)[:300], "mode": "w" if write else "r", "raw": raw,
                               "relative": True, "allowed": False}
        p = _abs(path)
        if p is None:  # an already-open file descriptor (os.fdopen): its own open was audited
            return True, {"event": event, "path": f"fd:{path}", "mode": "w" if write else "r", "raw": raw,
                          "allowed": True}
        ok = policy.can_write(p) if write else policy.can_read(p)
        return ok, {"event": event, "path": p, "mode": "w" if write else "r", "raw": raw, "allowed": ok}
    if event in _LIST_EVENTS:
        path = args[0] if args else None
        p = _abs(path)
        ok = (not strict) if p is None else policy.can_list(p)
        return ok, {"event": event, "path": p if p is not None else f"fd:{path}", "mode": "list", "allowed": ok}
    if event in _GLOB_EVENTS:
        pattern = args[0] if args else ""
        root_dir = args[2] if event == "glob.glob/2" and len(args) > 2 else None
        p = _glob_base(pattern, root_dir)
        ok = (not strict) if p is None else policy.can_list(p)
        return ok, {"event": event, "path": p, "mode": "glob", "pattern": os.fsdecode(os.fspath(pattern or "")),
                    "allowed": ok}
    if event == "os.mkdir":
        p = _abs(args[0], args[2] if len(args) > 2 else None)
        # an mkdir of a directory that already exists writes nothing (EEXIST; e.g. matplotlib's and numba's
        # makedirs(exist_ok=True) of their existing config and cache folders): logged, allowed
        exists = p is not None and os.path.isdir(p)
        ok = p is not None and (policy.can_mkdir(p) or exists)
        return ok, {"event": event, "path": p, "mode": "mkdir", "exists": exists, "allowed": ok}
    if event in _WRITE_PATH_EVENTS:
        p = _abs(args[0] if args else None)
        ok = p is not None and policy.can_write(p)
        return ok, {"event": event, "path": p if p is not None else f"fd:{args[0]}", "mode": "w", "allowed": ok}
    if event in _TWO_PATH_EVENTS:
        src, dst = (tuple(args) + (None, None))[:2]
        ps, pd = _abs(src), _abs(dst)
        ok = pd is not None and policy.can_write(pd)
        if event in ("os.rename", "shutil.move"):
            ok = ok and ps is not None and policy.can_write(ps)
        elif event != "os.symlink":
            ok = ok and ps is not None and policy.can_read(ps)
        return ok, {"event": event, "path": pd, "src": ps, "mode": "w", "allowed": ok}
    if event in _DELETE_EVENTS:
        p = _abs(args[0] if args else None)
        ok = (not strict) and p is not None and policy.can_write(p)
        return ok, {"event": event, "path": p, "mode": "delete", "allowed": ok}
    if event in _EXEC_EVENTS:
        what = args[0] if args else None
        if event == "os.spawn" and len(args) > 1:
            what = args[1]
        try:
            what = os.fsdecode(what) if isinstance(what, (bytes, str)) else repr(what)
        except Exception:  # noqa: BLE001
            what = repr(what)
        return (not strict), {"event": event, "path": what[:500], "mode": "exec", "allowed": not strict}
    if event.startswith(_NET_PREFIXES):
        return (not strict), {"event": event, "path": repr(args[:2])[:300], "mode": "net", "allowed": not strict}
    return True, None


class Auditor:
    """The installed hook: one per process (``install``)."""

    def __init__(self, policy: Policy, process: str, log_path: str, fh):
        self.policy = policy
        self.process = process
        self.log_path = log_path
        self.fh = fh
        self.n_events = 0
        self.violations = []
        self.recorded = set()
        self._lock = threading.Lock()
        self._local = threading.local()
        self._sha = {}

    def _write(self, row: dict) -> None:
        with self._lock:
            self.fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def _file_sha(self, f):
        if not (isinstance(f, str) and f):
            return None, None
        rp = os.path.realpath(f)
        if rp not in self._sha:
            try:
                self._sha[rp] = sha256_file(rp) if os.path.isfile(rp) else None
            except OSError:
                self._sha[rp] = None
        return rp, self._sha[rp]

    def module_rows(self, phase: str) -> None:
        for name in sorted(k for k in list(sys.modules) if k not in self.recorded):
            mod = sys.modules.get(name)
            if mod is None:
                continue
            f, sha = self._file_sha(getattr(mod, "__file__", None))
            self.recorded.add(name)
            self._write({"event": "module", "name": name, "file": f, "sha256": sha, "phase": phase})

    def hook(self, event: str, args: tuple) -> None:
        if getattr(self._local, "busy", False):
            return
        if event == "import":
            self._local.busy = True
            try:
                f = args[1] if len(args) > 1 and isinstance(args[1], str) else None
                self._write({"event": "import", "name": str(args[0]), "file": f})
            finally:
                self._local.busy = False
            return
        if not (event == "open" or event.startswith(("os.", "glob.", "shutil.", "subprocess.", "pty."))
                or event.startswith(_NET_PREFIXES)):
            return
        self._local.busy = True
        ok, row = True, None
        try:
            ok, row = decide(self.policy, event, args)
            if row is None:
                return
            self.n_events += 1
            self._write(row)
            if not ok:
                self.violations.append(row)
        finally:
            self._local.busy = False
        if not ok:
            raise AuditViolation(f"{self.policy.kind} process {self.process}: {event} {row.get('path')!r} "
                                 f"({row.get('mode')}) is outside its allowlist")

    def finish(self) -> None:
        self._local.busy = True
        try:
            self.module_rows("exit")
            self._write({"event": "summary", "n_events": self.n_events, "n_violations": len(self.violations),
                         "utc": utc()})
            self.fh.flush()
        except Exception:  # noqa: BLE001
            pass
        if self.policy.strict and self.violations:
            try:
                sys.stdout.flush()
                sys.stderr.write(f"AUDIT: {len(self.violations)} violation(s); exit {VIOLATION_EXIT}\n")
                sys.stderr.flush()
            finally:
                os._exit(VIOLATION_EXIT)


_INSTALLED: list = []


def installed():
    return _INSTALLED[0] if _INSTALLED else None


def install(root, process: str, kind: str, arm=None, extra_read=(), extra_write=(), repo=None, hand_model=None,
            dev_targets=None) -> Auditor:
    """Install this process's audit hook (once; in main(), never at import). The root guard runs first; strict
    kinds also drop sys.path entries outside the interpreter, the venv and the repo (the cwd included), so a later
    lazy import never lists or reads an unregistered folder."""
    if _INSTALLED:
        raise RuntimeError("the audit hook is already installed in this process")
    if not _PROC_RE.match(str(process)):
        raise ValueError(f"process name {process!r}")
    check_root(root, KIND_CLI[kind], dev_targets=dev_targets)
    policy = Policy(kind, root, arm=arm, repo=repo, hand_model=hand_model, extra_read=extra_read,
                    extra_write=extra_write)
    pruned = []
    if policy.strict:
        keep = []
        for p in sys.path:
            rp = _real(p) if isinstance(p, str) and p else None
            if rp is not None and (any(_inside(rp, d) for d in policy.code_dirs) or _inside(rp, policy.repo)):
                keep.append(p)
            else:
                pruned.append(p)
        sys.path[:] = keep
        sys.path_importer_cache.clear()
    adir = os.path.join(policy.root, "audit")
    os.makedirs(adir, exist_ok=True)
    log_path = os.path.join(adir, f"{process}-{os.getpid()}.jsonl")
    fd = os.open(log_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_APPEND, 0o644)
    fh = os.fdopen(fd, "w", buffering=1, encoding="utf-8")
    aud = Auditor(policy, process, log_path, fh)
    aud._write({"event": "install", "format": FORMAT, "process": process, "pid": os.getpid(), "utc": utc(),
                "argv": list(sys.argv), "cwd": os.getcwd(), "executable": sys.executable,
                "python": sys.version.split()[0], "sys_path_pruned": pruned, **policy.as_row()})
    aud.module_rows("install")
    sys.addaudithook(aud.hook)
    atexit.register(aud.finish)
    _INSTALLED.append(aud)
    return aud


PROCESS_KIND = {"stereo": "stereo", "match": "stereo", "sgbm": "stereo", "po": "P-O", "P-O": "P-O",
                "selfcal": "P-S", "ps": "P-S", "P-S": "P-S", "pt": "P-T", "P-T": "P-T", "poses-pick": "poses-pick",
                "settled-pose": "settled-pose", "render": "render", "eval": "eval", "launch": "launch",
                "classify": "classify"}


def strict_extra_reads(root, reads) -> tuple:
    """A strict process may add only the registered hand model to its reads, except on a synthetic root (a test
    passes its own hand model there). Raises ValueError otherwise."""
    reads = tuple(os.fspath(p) for p in reads or ())
    if not reads:
        return ()
    kind = root_kind(root, "stereo")[0]
    for p in reads:
        if _real(p) != _real(HAND_MODEL) and kind != "synthetic":
            raise ValueError(f"{p}: a strict process reads no extra file but the registered hand model")
    return reads


def install_hook(root, process: str, arm=None, mode: str = "strict", kind=None, extra_read=(), extra_reads=(),
                 extra_write=(), extra_writes=(), **_ignored) -> Auditor:
    """``install`` by process name (the form the stereo, P-S and render modules call): ``mode`` "strict" for
    stereo / P-O / P-S processes, "logging" for the rest; the kind comes from PROCESS_KIND unless given."""
    k = kind or PROCESS_KIND.get(process)
    if k is None:
        raise ValueError(f"no process kind for {process!r}; pass kind= (one of {STRICT_KINDS + LOGGING_KINDS})")
    if (mode == "strict") != (k in STRICT_KINDS):
        raise ValueError(f"process kind {k} runs the hook in {'strict' if k in STRICT_KINDS else 'logging'} mode")
    reads = tuple(extra_read) + tuple(extra_reads)
    if k in STRICT_KINDS:
        reads = strict_extra_reads(root, reads)
    return install(root, process, k, arm=arm, extra_read=reads, extra_write=tuple(extra_write) + tuple(extra_writes))


def install_logging_hook(root, process: str, kind: str = "eval", extra_write=()) -> Auditor:
    return install(root, process, kind, extra_write=extra_write)


# ===================================================================================================== writers
def write_new(path, data: bytes) -> None:
    """Create ``path`` with O_EXCL (never overwrites; no temporary, so nothing to unlink); parents are created."""
    p = os.fspath(path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def append_jsonl(path, row: dict) -> None:
    p = os.fspath(path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


# ===================================================================================================== git
class GitError(RuntimeError):
    pass


def _git(repo, *args) -> tuple:
    import subprocess  # here only: a strict process importing this module never imports subprocess through it

    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_OPTIONAL_LOCKS="0", LC_ALL="C", GIT_TERMINAL_PROMPT="0")
    r = subprocess.run(["git", "-C", os.fspath(repo), *args], capture_output=True, env=env, check=False)
    return r.returncode, r.stdout, r.stderr


def in_code_tree(path: str) -> bool:
    return (path.startswith(CODE_TREE_PREFIX) and not path.startswith(CODE_TREE_EXCLUDE_PREFIXES)
            and not path.endswith(CODE_TREE_EXCLUDE_SUFFIXES))


def code_tree_lines(repo=None, commit: str = "HEAD") -> list:
    rc, out, err = _git(repo or REPO_ROOT, "ls-tree", "-r", "-z", "--full-tree", str(commit), "--",
                        CODE_TREE_PREFIX)
    if rc != 0:
        raise GitError(f"git ls-tree {commit}: {err.decode('utf-8', 'replace').strip()}")
    lines = []
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, _, path = rec.partition(b"\t")
        p = path.decode("utf-8", "surrogateescape")
        if in_code_tree(p):
            lines.append(meta.decode("ascii") + "\t" + p + "\n")
    return lines


def code_tree(repo=None, commit: str = "HEAD") -> str:
    """sha256 of the ``git ls-tree -r <commit>`` lines (git's format, unquoted paths, newline-terminated, git's
    order) over rowant_lab/ minus evidence-m3/**, hybrid_frontier/notes/** and every *.md."""
    return sha256_bytes("".join(code_tree_lines(repo, commit)).encode("utf-8", "surrogateescape"))


def porcelain_code_entries(repo=None) -> list:
    """``git status --porcelain=v1 --untracked-files=all`` entries (untracked included) inside the code_tree paths."""
    rc, out, err = _git(repo or REPO_ROOT, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--",
                        CODE_TREE_PREFIX)
    if rc != 0:
        raise GitError(f"git status: {err.decode('utf-8', 'replace').strip()}")
    recs = out.split(b"\0")
    hits, i = [], 0
    while i < len(recs):
        rec = recs[i]
        i += 1
        if not rec:
            continue
        xy, path = rec[:2].decode("ascii", "replace"), rec[3:].decode("utf-8", "surrogateescape")
        paths = [path]
        if "R" in xy or "C" in xy:
            if i < len(recs):
                paths.append(recs[i].decode("utf-8", "surrogateescape"))
            i += 1
        for p in paths:
            if in_code_tree(p):
                hits.append(f"{xy} {p}")
    return hits


def clean_detail(repo=None, commit=None) -> dict:
    repo = repo or REPO_ROOT
    head = code_tree(repo, "HEAD")
    out = {"clean": False, "code_tree": head, "commit": commit, "entries": [], "commit_code_tree": None,
           "reason": ""}
    if not commit:
        out["reason"] = f"no driver-supplied commit ({CODE_COMMIT_ENV})"
        return out
    out["entries"] = porcelain_code_entries(repo)
    try:
        out["commit_code_tree"] = code_tree(repo, commit)
    except GitError as e:
        out["reason"] = str(e)
        return out
    if out["entries"]:
        out["reason"] = f"{len(out['entries'])} porcelain entries inside the code_tree paths"
    elif out["commit_code_tree"] != head:
        out["reason"] = "HEAD's code_tree differs from the commit's"
    else:
        out["clean"] = True
    return out


def clean(repo=None, commit=None) -> bool:
    """FIX-COMMIT: no porcelain entry (untracked included) inside the code_tree paths and code_tree(HEAD) ==
    code_tree(commit); False without a commit."""
    return clean_detail(repo, commit)["clean"]


def provenance(repo=None, commit=None) -> dict:
    """``{"code_tree", "clean"}`` for this process (the driver-supplied commit from ROWANT_CODE_COMMIT by default)."""
    d = clean_detail(repo, commit if commit is not None else os.environ.get(CODE_COMMIT_ENV))
    return {"code_tree": d["code_tree"], "clean": bool(d["clean"])}


_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def is_ancestor(repo, commit: str, ref: str = ORIGIN_REF) -> bool:
    """``git merge-base --is-ancestor <commit> <ref>`` (the local remote-tracking ref; nothing is fetched)."""
    rc, _, err = _git(repo or REPO_ROOT, "merge-base", "--is-ancestor", str(commit), str(ref))
    if rc == 0:
        return True
    if rc == 1:
        return False
    raise GitError(f"git merge-base --is-ancestor {commit} {ref}: {err.decode('utf-8', 'replace').strip()}")


def c4_landed(repo=None, c4=None, ref: str = ORIGIN_REF) -> tuple:
    """``(ok, reason)``: the C4 commit (``ROWANT_C4_COMMIT`` by default) carries a K0S-PICK amendment and both pick
    copies, and is an ancestor of ``ORIGIN_REF``. False before C4 exists."""
    repo = repo or REPO_ROOT
    c4 = c4 if c4 is not None else os.environ.get(C4_ENV)
    if not c4:
        return False, f"no C4 commit ({C4_ENV})"
    if not _SHA_RE.match(str(c4)):
        return False, "the C4 commit must be a hex sha"
    rc, out, _ = _git(repo, "show", f"{c4}:{DEV_TARGETS_REL}")
    if rc != 0:
        return False, "the C4 commit has no dev targets"
    try:
        ids = [a.get("id") for a in json.loads(out.decode("utf-8")).get("amendments", [])]
    except (ValueError, AttributeError):
        return False, "the C4 commit's dev targets do not parse"
    if C4_AMENDMENT not in ids:
        return False, f"the C4 commit carries no {C4_AMENDMENT} amendment"
    for f in C4_REQUIRED_FILES:
        if _git(repo, "cat-file", "-e", f"{c4}:{f}")[0] != 0:
            return False, f"the C4 commit has no {f}"
    try:
        ok = is_ancestor(repo, c4, ref)
    except GitError as e:
        return False, str(e)
    return ok, "" if ok else f"{c4} is not an ancestor of {ref}"


def read_head_stdlib(repo=None) -> str:
    """Test-only stdlib reader: HEAD's commit from the worktree's own ``<gitdir>/HEAD`` (refs from the worktree's
    gitdir, then its commondir's loose refs and packed-refs). Never the main clone's HEAD."""
    repo = Path(repo or REPO_ROOT)
    dotgit = repo / ".git"
    if dotgit.is_file():
        txt = dotgit.read_text().strip()
        if not txt.startswith("gitdir:"):
            raise GitError(f"{dotgit}: not a gitdir link")
        gitdir = Path(txt.split(":", 1)[1].strip())
        if not gitdir.is_absolute():
            gitdir = (repo / gitdir).resolve()
    else:
        gitdir = dotgit
    head = (gitdir / "HEAD").read_text().strip()
    if not head.startswith("ref:"):
        return head
    ref = head.split(":", 1)[1].strip()
    common = gitdir
    if (gitdir / "commondir").is_file():
        c = Path((gitdir / "commondir").read_text().strip())
        common = c if c.is_absolute() else (gitdir / c).resolve()
    for base in (gitdir, common):
        p = base / ref
        if p.is_file():
            return p.read_text().strip()
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if line and not line.startswith(("#", "^")):
                sha, _, name = line.partition(" ")
                if name.strip() == ref:
                    return sha
    raise GitError(f"{ref}: not found in {gitdir} or {common}")
