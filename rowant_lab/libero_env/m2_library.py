"""M2 policy-side library (plan modules.m2_library): the only file reader the policy uses, and the server's loader.

M2 (perceived poses, grounded instruction): perception in the loop. Since amendment A9 (verifier blocker B1) it
reads the library version ``policy-v2`` (``m2_paths.POLICY_LIB``, written by ``m2_libv2``) and gives, per object TYPE
(never per task or instance):

* ``types``: M1's collision primitives, ``com`` and ``rest_up_in_body`` per body (M1's frozen types, sanitized:
  ``seen_in`` and ``rest_up_from`` removed), which M1's frozen ``OracleAgent`` reads (``primitives`` and ``com`` in its
  ``up_ok=False`` replanning branch: verifier minor M7);
* ``export_types``: the M2 export's kinematics and body-frame sites (contain, cook and drawer regions; world-frame
  sites are not exported), sanitized: the ROOT body pose is identity and ``seen_in`` / ``exported_from`` are removed;
* M1's frozen grasp records and the Panda jaw model (M1's ``records/`` and ``robot.json``, read in place);
* the per-suite workcell model (arena primitives, robot mount and base, camera names, the suite's type list).

Every file is opened through ``LibraryReader.read``, which refuses any path outside the policy-v2 root, M1's
``records/`` and M1's ``robot.json`` (and, for the server, the Panda export), refuses anything under a ``scenes``
folder, and checks every file's sha256 against the policy-v2 manifest. Every loaded document is checked by
``problems``: a non-identity root pose, a per-task field or an instance name is refused (``LibraryRefused``).
``opened`` lists every path read (a runtime record).

Suites M2 never had (Pro2.0's Goal) are read only when a caller passes ``extra_suites`` (``{suite: {"types": [...],
"panda_of": an M2 suite}}``, from ``rowant_lab/pro2/goal_paths.py``) and the library is a variant tree that holds the
suite's workcell. M2's own runs never pass it, so they read ``m2_paths``' frozen tables only.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from rowant_lab.libero_env import m2_paths as MP

IDENTITY_POS = (0.0, 0.0, 0.0)
IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
# fields that carry per-task (scene) information: never in a type, workcell or record the policy or server loads
PER_TASK_KEYS = frozenset({"seen_in", "exported_from", "rest_up_from", "task", "tasks", "task_name", "scene",
                           "scenes", "instances", "instance", "goal", "init", "regions", "obj_of_interest",
                           "robot_start", "obs_keys", "init_state", "init_states", "fixture_pose", "world_pose"})
# an instance name carries LIBERO's instance suffix: <type>_<digit>[_<rel>] (e.g. a cabinet's joint as compiled)
INSTANCE_NAME = re.compile(r"(?:^|[a-z])_\d+(?:_|$)")
# a task reference: "libero_spatial:3", "libero_object task 2", "DEV state 0"
TASK_REF = re.compile(r"(libero_(object|spatial)\s*(:|task)\s*\d|\btask\s+\d|\bdev state\b|\bgate state\b)", re.I)


class LibraryRefused(Exception):
    pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _walk_problems(obj, path: str, out: list) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            if ks in PER_TASK_KEYS:
                out.append(f"{path}.{ks}: per-task field")
            if INSTANCE_NAME.search(ks):
                out.append(f"{path}.{ks}: instance name in a key")
            _walk_problems(v, f"{path}.{ks}", out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk_problems(v, f"{path}[{i}]", out)
    elif isinstance(obj, str):
        if INSTANCE_NAME.search(obj):
            out.append(f"{path}: instance name {obj!r}")
        if TASK_REF.search(obj):
            out.append(f"{path}: task reference {obj!r}")


def root_pose_problems(doc: dict, tol: float = 0.0) -> list:
    """A type's root body must sit at the identity: its pose in any scene is perceived, never stored."""
    out = []
    root = doc.get("root_body")
    body = (doc.get("bodies") or {}).get(root) if root else None
    if body is None:
        return out
    pos = body.get("body_pos")
    quat = body.get("body_quat")
    if pos is not None and any(abs(float(a) - b) > tol for a, b in zip(pos, IDENTITY_POS)):
        out.append(f"bodies.{root}.body_pos: non-identity root pose {pos}")
    if quat is not None and any(abs(float(a) - b) > tol for a, b in zip(quat, IDENTITY_QUAT)):
        out.append(f"bodies.{root}.body_quat: non-identity root pose {quat}")
    for key in ("pose", "pos", "quat", "quat_wxyz", "R_world", "t_world", "xpos", "xquat"):
        if key in body:
            out.append(f"bodies.{root}.{key}: a stored root pose")
    return out


def problems(doc: dict, kind: str) -> list:
    """Why ``doc`` (kind: 'type', 'm1type', 'workcell', 'manifest') may not be loaded; empty when it may."""
    out: list = []
    _walk_problems(doc, kind, out)
    if kind in ("type", "m1type"):
        out += root_pose_problems(doc)
    return out


def lfb_dirs(root=None) -> tuple:
    """LF-b (registration lfc_b, amendment LFB-LIB): ``(onboard_dir, policy_lib, m1_library)`` of the variant
    library tree ``root`` (``m2/onboard-v3``, ``m2/policy-v3``, ``m1``), or M2's own folders when ``root`` is None. The
    policy and the server take it from their params (``lfb_library``); the evaluator never does."""
    if not root:
        return MP.ONBOARD_DIR, MP.POLICY_LIB, MP.M1_LIBRARY
    r = Path(root)
    return r / "m2" / "onboard-v3", r / "m2" / "policy-v3", r / "m1"


def extra_suite_table(extra_suites: dict | None) -> dict:
    """``{suite: (types, panda_of)}`` of suites outside M2's ``m2_paths`` (the Pro2.0 line's Goal), from the
    ``extra_suites`` a caller passes; refused when one names an M2 suite or maps its Panda to a suite M2 has none for."""
    out = {}
    for suite, spec in sorted((extra_suites or {}).items()):
        if suite in MP.WORKCELL_TYPES:
            raise LibraryRefused(f"extra suite {suite!r} is one of M2's suites")
        po = spec.get("panda_of")
        if po is not None and po not in MP.SUITES:
            raise LibraryRefused(f"extra suite {suite!r}: Panda of {po!r}, which is not an M2 suite")
        # None: the suite's own Panda, in the variant tree (``pandas/``; a table and mount no M2 suite has: the
        # long-task suite's living room and study)
        out[str(suite)] = (tuple(str(t) for t in spec["types"]), None if po is None else str(po))
    return out


def _within(p: Path, root: Path) -> bool:
    try:
        return p.is_relative_to(root)
    except AttributeError:  # pragma: no cover (Python < 3.9)
        return p.parts[:len(root.parts)] == root.parts


class LibraryReader:
    """Reads and sha-checks the files of the policy-v2 library and the in-place files its manifest lists.

    ``extra_suites``: suites outside M2's ``m2_paths`` (see the module docstring); None for M2's own runs."""

    def __init__(self, lib_dir=None, m1_library=None, expect_manifest_sha256: str | None = None,
                 allow_panda: bool = False, extra_suites: dict | None = None):
        self.lib_dir = Path(lib_dir or MP.POLICY_LIB).resolve()
        self.m1 = Path(m1_library or MP.M1_LIBRARY).resolve()
        self.allow_panda = bool(allow_panda)
        self.extra = extra_suite_table(extra_suites)
        self.opened: list = []
        raw = self._raw(self.lib_dir / "manifest.json")
        self.manifest_sha256 = sha256_bytes(raw)
        if expect_manifest_sha256 is not None and self.manifest_sha256 != expect_manifest_sha256:
            raise LibraryRefused(f"policy-v2 manifest sha256 {self.manifest_sha256} != {expect_manifest_sha256}")
        self.manifest = json.loads(raw.decode("utf-8"))
        if self.manifest.get("version") != MP.POLICY_LIB_VERSION:
            raise LibraryRefused(f"library version {self.manifest.get('version')!r} is not {MP.POLICY_LIB_VERSION}")
        bad = problems({k: v for k, v in self.manifest.items() if k not in ("sources", "in_place")}, "manifest")
        if bad:
            raise LibraryRefused(f"manifest: {bad[:3]}")

    # ------------------------------------------------------------------ the one opener
    def _allowed(self, p: Path) -> bool:
        if "scenes" in p.parts:
            return False
        if _within(p, self.lib_dir):
            return True
        if _within(p, self.m1 / "records") or p == self.m1 / "robot.json":
            return True
        if self.allow_panda and p.parent == (MP.EXPORT_DIR.resolve()) and p.name.startswith("panda-"):
            return True
        return False

    def _raw(self, path) -> bytes:
        p = Path(path).resolve()
        if "scenes" in p.parts:
            raise LibraryRefused(f"{p}: a scenes folder is never a policy input")
        if not self._allowed(p):
            raise LibraryRefused(f"{p}: outside the M2 policy library")
        b = p.read_bytes()
        self.opened.append(str(p))
        return b

    def _expected_sha(self, p: Path) -> str | None:
        if _within(p, self.lib_dir):
            return self.manifest["files"].get(p.relative_to(self.lib_dir).as_posix())
        if _within(p, self.m1):
            return self.manifest["in_place"].get("m1/" + p.relative_to(self.m1).as_posix())
        if p.parent == MP.EXPORT_DIR.resolve():
            return self.manifest["in_place"].get("export/" + p.name)
        return None

    def read(self, path) -> bytes:
        p = Path(path).resolve()
        b = self._raw(p)
        want = self._expected_sha(p)
        if want is None:
            raise LibraryRefused(f"{p}: not listed in the policy-v2 manifest")
        if sha256_bytes(b) != want:
            raise LibraryRefused(f"{p}: sha256 differs from the policy-v2 manifest")
        return b

    def json(self, path, kind: str | None = None) -> dict:
        doc = json.loads(self.read(path).decode("utf-8"))
        if kind is not None:
            bad = problems(doc, kind)
            if bad:
                raise LibraryRefused(f"{path}: {bad[:3]}")
        return doc

    # ------------------------------------------------------------------ documents
    def export_type(self, typ: str) -> dict:
        return self.json(self.lib_dir / "types" / f"{typ}.json", "type")

    def m1_type(self, typ: str) -> dict:
        return self.json(self.lib_dir / "m1types" / f"{typ}.json", "m1type")

    def workcell(self, suite: str) -> dict:
        return self.json(self.lib_dir / "workcells" / f"{suite}.json", "workcell")

    def suite_types(self, suite: str) -> tuple:
        """The suite's workcell types: M2's from ``m2_paths`` (frozen), else an extra suite's."""
        if suite in MP.WORKCELL_TYPES:
            return MP.WORKCELL_TYPES[suite]
        if suite in self.extra:
            return self.extra[suite][0]
        raise LibraryRefused(f"suite {suite!r} is neither M2's nor an extra suite")

    def suites(self) -> set:
        """The suites this reader serves: M2's (``m2_paths.SUITES``) and the extra ones."""
        return set(MP.SUITES) | set(self.extra)

    def records(self, typ: str):
        p = self.m1 / "records" / f"{typ}.json"
        if "m1/records/" + p.name not in self.manifest["in_place"]:
            return None
        from rowant_lab.libero_env import grasp as GR

        return GR.records_from_json(json.loads(self.read(p).decode("utf-8")))

    def robot(self) -> dict:
        return json.loads(self.read(self.m1 / "robot.json").decode("utf-8"))

    def panda(self, suite: str):
        """The suite's exported Panda model (server side: the self-filter)."""
        from rowant_lab.libero_env import m2_kin as KIN

        if not self.allow_panda:
            raise LibraryRefused("the Panda model is the server's (allow_panda)")
        base = MP.EXPORT_DIR
        if suite in self.extra and self.extra[suite][1] is None:  # its own Panda, sha-checked through the manifest
            base = self.lib_dir / "pandas"
        elif suite in self.extra:  # an extra suite on the same table and mount as an M2 suite (Pro2.0's Goal)
            suite = self.extra[suite][1]
        doc = json.loads(self.read(base / f"panda-{suite}.json").decode("utf-8"))
        npz = base / doc["npz"]
        self.read(npz)  # the sha check (the bytes are then loaded by numpy from the same path)
        with np.load(npz, allow_pickle=False) as z:
            arrays = {k: np.asarray(z[k]) for k in z.files}
        return KIN.Panda(doc, arrays)


class M2Library:
    """The policy's library for one suite: M1's types as ``types`` (the agent's), the export as ``export_types``."""

    def __init__(self, suite: str, lib_dir=None, m1_library=None, expect_manifest_sha256: str | None = None,
                 extra_suites: dict | None = None):
        from rowant_lab.libero_env import grasp as GR

        self.suite = suite
        self.reader = LibraryReader(lib_dir, m1_library, expect_manifest_sha256, extra_suites=extra_suites)
        self.root = self.reader.lib_dir
        self.manifest_sha256 = self.reader.manifest_sha256
        self.workcell = self.reader.workcell(suite)
        names = self.reader.suite_types(suite)
        if sorted(self.workcell["types"]) != sorted(names):
            raise LibraryRefused(f"workcell types differ from the suite's: {self.workcell['types']}")
        self.export_types = {t: self.reader.export_type(t) for t in names}
        self.types = {t: self.reader.m1_type(t) for t in names}
        self.records = {}
        for t in names:
            r = self.reader.records(t)
            if r is not None:
                self.records[t] = r
        self.robot = self.reader.robot()
        self.jaw = GR.JawModel.from_robot(self.robot)
        self._body_boxes: dict = {}

    @property
    def opened(self) -> list:
        return list(self.reader.opened)

    # ------------------------------------------------------------------ models
    def m1_type(self, typ: str) -> dict:
        return self.types[typ]

    def body_boxes(self, typ: str, rel: str):
        """As M1's ``Library.body_boxes`` (M1's collision primitives of one body)."""
        from rowant_lab.libero_env import grasp as GR

        key = (typ, rel)
        if key not in self._body_boxes:
            self._body_boxes[key] = GR.Boxes.from_prims(self.types[typ]["bodies"][rel]["primitives"])
        return self._body_boxes[key]

    @staticmethod
    def quat_mat(q) -> np.ndarray:
        from rowant_lab.libero_env.grasp import quat_wxyz_to_mat

        return quat_wxyz_to_mat(q)

    def site_for(self, percept: dict, part_phrase: str | None):
        """(centre, R, half) in the world of the percept's region site that the part phrase names, or its only one."""
        tdoc = self.export_types[percept["type"]]
        sites = tdoc["sites"]
        if not sites:
            return None
        name = None
        if part_phrase:
            from rowant_lab.libero_env.m2_ground import match_type

            cand = [s for s in sites if match_type(part_phrase, [s])[0] == s]
            name = cand[0] if len(cand) == 1 else None
            if name is None:
                pw = set(str(part_phrase).split())
                hit = [s for s in sites if pw & set(s.split("_"))]
                name = hit[0] if len(hit) == 1 else None
        if name is None:
            region = [s for s in sites if "region" in s]
            name = region[0] if len(region) == 1 else (sorted(sites)[0] if len(sites) == 1 else None)
        if name is None:
            return None
        s = sites[name]
        if s["body"] not in percept["bodies"]:
            return None
        R = self.quat_mat(percept["bodies"][s["body"]]["quat_wxyz"])
        t = np.asarray(percept["bodies"][s["body"]]["pos"], float)
        Rs = R @ np.asarray(s["R"], float)
        return t + R @ np.asarray(s["pos"], float), Rs, np.abs(Rs @ np.asarray(s["size"], float))

    def site_name_for(self, percept: dict, part_phrase: str | None) -> str | None:
        tdoc = self.export_types[percept["type"]]
        if not tdoc["sites"]:
            return None
        if part_phrase:
            pw = set(str(part_phrase).split())
            hit = [s for s in tdoc["sites"] if pw & set(s.split("_"))]
            if len(hit) == 1:
                return hit[0]
        region = [s for s in tdoc["sites"] if "region" in s]
        return region[0] if len(region) == 1 else (sorted(tdoc["sites"])[0] if len(tdoc["sites"]) == 1 else None)

    # ------------------------------------------------------------------ the synthetic scene (M1's format)
    def scene_from_percepts(self, percepts: list) -> dict:
        """M1's scene dict built from percepts: instances keyed by percept id, their bodies named ``<id>_<rel>``,
        the per-type body-frame sites re-keyed to those bodies, and the workcell's arena and robot mount."""
        inst, sites = {}, {}
        for p in percepts:
            tdoc = self.export_types[p["type"]]
            bodies = {rel: f"{p['id']}_{rel}" for rel in p["bodies"]}
            inst[p["id"]] = {"type": p["type"], "kind": tdoc["kind"], "bodies": bodies,
                             "root_body_name": bodies[tdoc["root_body"]]}
            for sname, s in tdoc["sites"].items():
                if s["body"] not in bodies:
                    continue
                sites[f"{p['id']}_{sname}"] = {"parent_instance": p["id"], "frame_body": bodies[s["body"]],
                                               "pos": s["pos"], "R": s["R"], "size": s["size"]}
        return {"instances": inst, "sites": sites, "arena": self.workcell["arena"],
                "robot_mount": self.workcell["robot_mount"]}

    def poses_from_percepts(self, percepts: list) -> dict:
        out = {}
        for p in percepts:
            for rel, b in p["bodies"].items():
                out[f"{p['id']}_{rel}"] = {"pos": list(b["pos"]), "quat_wxyz": list(b["quat_wxyz"])}
        return out
