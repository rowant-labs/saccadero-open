"""K0-stereo hand model file: a one-off tool (Monty venv; numpy, the standard library and M2's library reader).

Writes ``rowant_lab/libero_env/evidence-m3/stereo1/panda_hand_model.json`` (k0_stereo.methods.hand_model_file;
CONTRACT.md "Hand model file") from the M2 export's Panda, read as K0's PandaMesh reads it
(``m2_library.LibraryReader(allow_panda=True).panda(...)``, sha-checked against the policy-v3 manifest; the
``m2_kin.Panda`` body tree and its ``<geom>__V`` / ``<geom>__F`` arrays, vertices in body frames)::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<repository root> <server python> -B \\
        -m rowant_lab.libero_env.m3_stereo1_handmodel write      (once; refuses to overwrite a different file)
    ... m3_stereo1_handmodel check                                (sha256 and the FK agreement numbers)

File layout: ``{"format": "rowant panda hand model v1", "source": {...}, "frame": "eef", "bodies": [{"name",
"T_eef_from_body_at_zero": 4x4, "joint": null | {"axis_eef": [x, y, z], "qpos_index": 0 | 1, "sign": +1},
"vertices": [[x, y, z], ...], "faces": [[i, j, k], ...]}]}``. Frame "eef" is the obs eef frame: origin at the grip
site (the ``gripper0_eef`` body origin, where robosuite's ``grip_site`` sits; panda_gripper.xml) with the axes of the
hand body ``robot0_right_hand`` (the frame of robot0_eef_pos / robot0_eef_quat, which E and X use). A finger at
gripper qpos q is ``T_eef_from_body_at_zero`` followed by a translation of ``sign * q[qpos_index]`` along ``axis_eef``.
Hand and finger bodies only (the three bodies that carry visual meshes: the hand shell and the two fingers); no
camera, base, mount, arena or workcell field, and no suite name (the hand and finger parts of both M2 exports are
checked identical and both are recorded by sha256 only). ``body_poses`` / ``posed_vertices`` are the pure readers
(the stereo re-implements them from this layout; it may not import this module).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from rowant_lab.libero_env import m3_stereo1_audit as AU

FORMAT = "rowant panda hand model v1"
OUT = AU.HAND_MODEL
AXES_BODY = "robot0_right_hand"
ORIGIN_BODY = "gripper0_eef"
HAND_BODIES = ("gripper0_right_gripper", "gripper0_leftfinger", "gripper0_rightfinger")
SOURCE_SUITE = "libero_spatial"  # the export K0's hand_camera_offset used; libero_object's parts are checked equal
CHECK_SUITES = ("libero_object", "libero_spatial")
FORBIDDEN_WORDS = ("camera", "base", "mount", "arena", "workcell", "libero_")


def _qmat(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _local(body: dict) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = _qmat(body["body_quat"])
    M[:3, 3] = np.asarray(body["body_pos"], float)
    return M


def _joint_at_zero(body: dict) -> np.ndarray:
    """The body's own joints at q = 0 (m2_kin.apply_joint with q = 0: a displacement of -qpos0)."""
    from rowant_lab.libero_env import m2_kin as KIN

    R, t = np.eye(3), np.zeros(3)
    for jt in body["joints"]:
        R, t = KIN.apply_joint(R, t, jt, 0.0)
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def chain(bodies: dict, ancestor: str, name: str) -> np.ndarray:
    """T_ancestor_from_name at every joint's zero, composed exactly as m2_kin.Panda.fk composes it."""
    path = []
    n = name
    while n != ancestor:
        if n not in bodies:
            raise KeyError(f"{name} does not descend from {ancestor}")
        path.append(n)
        n = bodies[n]["parent"]
    M = np.eye(4)
    for n in reversed(path):
        M = M @ _local(bodies[n]) @ _joint_at_zero(bodies[n])
    return M


def build(panda) -> dict:
    """The hand model document from an ``m2_kin.Panda`` (doc bodies and arrays)."""
    bodies = panda.bodies
    T_hand_eef = chain(bodies, AXES_BODY, ORIGIN_BODY)
    grip = T_hand_eef[:3, 3].copy()
    T_eef_hand = np.eye(4)
    T_eef_hand[:3, 3] = -grip  # the eef frame keeps the hand body's axes
    out = []
    for name in HAND_BODIES:
        b = bodies[name]
        T0 = T_eef_hand @ chain(bodies, AXES_BODY, name)
        joint = None
        live = [jt for jt in b["joints"] if jt.get("proprio") is not None]
        if len(b["joints"]) > 1 or (b["joints"] and not live):
            raise ValueError(f"{name}: expected at most one proprioceptive joint")
        if live:
            jt = live[0]
            if jt["type"] != "slide" or jt["proprio"][0] != "robot0_gripper_qpos":
                raise ValueError(f"{name}: a finger joint must be a gripper-qpos slide")
            a = np.asarray(jt["axis"], float)
            a = a / np.linalg.norm(a)
            joint = {"axis_eef": [float(v) for v in T0[:3, :3] @ a], "qpos_index": int(jt["proprio"][1]),
                     "sign": 1}
        Vs, Fs, off = [], [], 0
        for key in b["geoms"]:
            V = np.asarray(panda.arrays[f"{key}__V"], np.float64)
            F = np.asarray(panda.arrays[f"{key}__F"], np.int64)
            Vs.append(V)
            Fs.append(F + off)
            off += len(V)
        if not Vs:
            raise ValueError(f"{name}: no visual mesh")
        out.append({"name": name, "T_eef_from_body_at_zero": T0.tolist(), "joint": joint,
                    "vertices": np.concatenate(Vs).tolist(), "faces": np.concatenate(Fs).tolist()})
    return {"format": FORMAT, "frame": "eef", "bodies": out,
            "source": {"tool": "rowant_lab/libero_env/m3_stereo1_handmodel.py",
                       "model": "the M2 export's Panda (m2_library.LibraryReader(allow_panda=True).panda; "
                                "m2_kin.Panda arrays, as K0's PandaMesh reads them)",
                       "axes_body": AXES_BODY, "origin_body": ORIGIN_BODY,
                       "grip_site_in_hand_m": [float(v) for v in grip],
                       "geoms_by_body": {n: list(bodies[n]["geoms"]) for n in HAND_BODIES},
                       "finger_joint_sign_rule": "translation sign * q[qpos_index] along axis_eef (the model's "
                                                 "own slide axis; its range carries the direction)"}}


def body_poses(model: dict, gripper_qpos) -> dict:
    """{body: T_eef_from_body (4x4)} at the gripper qpos."""
    q = [float(v) for v in gripper_qpos]
    out = {}
    for b in model["bodies"]:
        T = np.array(b["T_eef_from_body_at_zero"], float)
        jt = b["joint"]
        if jt is not None:
            T = T.copy()
            T[:3, 3] += float(jt["sign"]) * q[int(jt["qpos_index"])] * np.asarray(jt["axis_eef"], float)
        out[b["name"]] = T
    return out


def posed_vertices(model: dict, gripper_qpos) -> tuple:
    """(V, F): every hand and finger triangle in the eef frame at the gripper qpos."""
    poses = body_poses(model, gripper_qpos)
    Vs, Fs, off = [], [], 0
    for b in model["bodies"]:
        T = poses[b["name"]]
        V = np.asarray(b["vertices"], float)
        Vs.append(V @ T[:3, :3].T + T[:3, 3])
        Fs.append(np.asarray(b["faces"], np.int64) + off)
        off += len(V)
    return np.concatenate(Vs), np.concatenate(Fs)


def dumps(doc: dict) -> bytes:
    return (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def load(path=OUT) -> dict:
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def _export_pandas() -> tuple:
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.libero_env import m2_paths as MP

    rd = LB.LibraryReader(allow_panda=True)
    pandas = {s: rd.panda(s) for s in CHECK_SUITES}
    shas = []
    for s in CHECK_SUITES:
        js = MP.EXPORT_DIR / f"panda-{s}.json"
        doc = json.loads(js.read_bytes().decode("utf-8"))
        shas.append({"json_sha256": hashlib.sha256(js.read_bytes()).hexdigest(), "npz_sha256": doc["npz_sha256"]})
    return pandas, shas, rd.manifest_sha256


def _same_parts(a, b) -> bool:
    names = set(HAND_BODIES) | {AXES_BODY, ORIGIN_BODY}
    for n in names:
        if a.bodies[n] != b.bodies[n]:
            return False
        for key in a.bodies[n]["geoms"]:
            for suf in ("__V", "__F"):
                if not np.array_equal(a.arrays[key + suf], b.arrays[key + suf]):
                    return False
    return True


def make_doc() -> dict:
    pandas, shas, manifest = _export_pandas()
    if not _same_parts(pandas["libero_object"], pandas["libero_spatial"]):
        raise ValueError("the hand and finger parts differ between the two M2 exports")
    doc = build(pandas[SOURCE_SUITE])
    doc["source"].update({"exports_sha256": sorted(shas, key=lambda d: d["json_sha256"]),
                          "identical_in_both_exports": True, "policy_manifest_sha256": manifest})
    return doc


def keys_and_strings(doc) -> list:
    """Every key and string value in a document (the forbidden-field test)."""
    out = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            out.append(str(k))
            out.extend(keys_and_strings(v))
    elif isinstance(doc, list):
        for v in doc:
            if isinstance(v, (dict, list, str)):
                out.extend(keys_and_strings(v))
    elif isinstance(doc, str):
        out.append(doc)
    return out


def fk_check(model: dict, panda, joint_sets, gripper_sets) -> dict:
    """Largest difference between the hand model and m2_kin FK (rotation entries and translation in metres)."""
    worst_R = worst_t = 0.0
    n = 0
    for jp in joint_sets:
        for gq in gripper_sets:
            fk = panda.fk(jp, gq)
            Rh, _ = fk[AXES_BODY]
            _, pe = fk[ORIGIN_BODY]
            E = np.eye(4)
            E[:3, :3], E[:3, 3] = Rh, pe
            mp = body_poses(model, gq)
            for name in HAND_BODIES:
                Rb, tb = fk[name]
                W = E @ mp[name]
                worst_R = max(worst_R, float(np.abs(W[:3, :3] - Rb).max()))
                worst_t = max(worst_t, float(np.abs(W[:3, 3] - tb).max()))
                n += 1
    return {"n_poses": n, "max_abs_R": worst_R, "max_abs_t_m": worst_t}


CHECK_JOINTS = ([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785], [0.4, 0.2, -0.3, -1.8, 0.2, 2.4, -0.6],
                [-0.9, 0.6, 0.7, -2.6, -0.5, 1.6, 1.9])
CHECK_GRIPPERS = ([0.0, 0.0], [0.04, -0.04], [0.02, -0.02], [0.0125, -0.031], [0.039, -0.001])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="K0-stereo Panda hand model file (one-off)")
    ap.add_argument("cmd", choices=["write", "check"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.cmd == "write":
        data = dumps(make_doc())
        if out.exists():
            same = out.read_bytes() == data
            print(json.dumps({"path": str(out), "exists": True, "identical": same,
                              "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}))
            return 0 if same else 1
        AU.write_new(out, data)
        print(json.dumps({"path": str(out), "written": True, "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()}))
        return 0
    model = load(out)
    pandas, _, _ = _export_pandas()
    res = {"path": str(out), "sha256": AU.sha256_file(out),
           "fk": {s: fk_check(model, pandas[s], CHECK_JOINTS, CHECK_GRIPPERS) for s in CHECK_SUITES},
           "forbidden_words": [w for w in FORBIDDEN_WORDS
                               if any(w in s.lower() for s in keys_and_strings(model)
                                      if not s.startswith("rowant_lab/libero_env/m3_stereo1_handmodel"))]}
    print(json.dumps(res))
    ok = all(v["max_abs_R"] <= 1e-9 and v["max_abs_t_m"] <= 1e-9 for v in res["fk"].values())
    return 0 if ok and not res["forbidden_words"] else 1


if __name__ == "__main__":
    sys.exit(main())
