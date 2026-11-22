"""M2 kinematics (numpy only; both venvs): MuJoCo's forward kinematics rule for the exported Panda model (the
self-filter) and for articulated types (the cabinet's drawers).

M2 (perceived poses, grounded instruction): perception in the loop. MuJoCo's rule per body: ``R = R_parent
R(body_quat)``, ``t = t_parent + R_parent body_pos``; then per joint, a hinge turns about its axis through ``jnt_pos``
by ``q - qpos0`` (``R <- R R(axis, q - qpos0)``, ``t <- anchor - R jnt_pos``) and a slide moves along
``R axis`` by ``q - qpos0``. The Panda's joint values come from proprioception only (``robot0_joint_pos`` and
``robot0_gripper_qpos``); FM3 checks this against the simulator's link poses.
"""

from __future__ import annotations

import math

import numpy as np


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R) -> list:
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return [float(v) for v in q]


def axis_angle(axis, ang: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def apply_joint(R: np.ndarray, t: np.ndarray, jt: dict, q: float) -> tuple[np.ndarray, np.ndarray]:
    d = float(q) - float(jt.get("qpos0", 0.0))
    if jt["type"] == "hinge":
        p = np.asarray(jt["pos"], float)
        anchor = t + R @ p
        R2 = R @ axis_angle(jt["axis"], d)
        return R2, anchor - R2 @ p
    if jt["type"] == "slide":
        return R, t + R @ (np.asarray(jt["axis"], float) * d)
    raise ValueError(f"joint type {jt['type']!r}")


class Panda:
    """The exported Panda model of a suite (``panda-<suite>.json`` / ``.npz``): FK from proprioception and surface
    samples. It takes the already-read document and arrays: this module opens no file (it is scanned under the
    policy rules); ``m2_library.LibraryReader.panda`` reads and sha-checks them."""

    def __init__(self, doc: dict, arrays: dict):
        self.bodies = doc["bodies"]
        self.arrays = {k: np.asarray(v) for k, v in arrays.items()}
        self.order = self._order()

    def _order(self) -> list:
        out, seen = [], set()
        names = list(self.bodies)
        while len(out) < len(names):
            for n in names:
                if n in seen:
                    continue
                par = self.bodies[n]["parent"]
                if par == "world" or par in seen:
                    out.append(n)
                    seen.add(n)
        return out

    def fk(self, joint_pos, gripper_qpos) -> dict:
        """{body: (R, t)} at the given proprioception (world frame)."""
        src = {"robot0_joint_pos": list(joint_pos), "robot0_gripper_qpos": list(gripper_qpos)}
        poses = {}
        for n in self.order:
            b = self.bodies[n]
            par = b["parent"]
            Rp, tp = poses[par] if par in poses else (np.eye(3), np.zeros(3))
            R = Rp @ quat_to_mat(b["body_quat"])
            t = tp + Rp @ np.asarray(b["body_pos"], float)
            for jt in b["joints"]:
                if jt.get("proprio") is None:
                    continue
                key, i = jt["proprio"]
                R, t = apply_joint(R, t, jt, src[key][i])
            poses[n] = (R, t)
        return poses

    def surface_samples(self, spacing_m: float = 0.004, seed: int = 20261022) -> dict:
        """{body: (n, 3) body-frame points}: area-weighted samples of every visual mesh at about ``spacing_m``."""
        rng = np.random.default_rng(seed)
        out = {}
        for n in self.order:
            pts = []
            for key in self.bodies[n]["geoms"]:
                V, F = self.arrays[f"{key}__V"], self.arrays[f"{key}__F"]
                a = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
                n_s = int(max(len(V), math.ceil(a.sum() / (spacing_m ** 2))))
                fi = rng.choice(len(F), size=n_s, p=a / a.sum())
                r1, r2 = rng.random(n_s), rng.random(n_s)
                s1 = np.sqrt(r1)
                b0, b1, b2 = 1 - s1, s1 * (1 - r2), s1 * r2
                P = V[F[fi, 0]] * b0[:, None] + V[F[fi, 1]] * b1[:, None] + V[F[fi, 2]] * b2[:, None]
                pts.append(np.concatenate([P, V]))
            if pts:
                out[n] = np.concatenate(pts)
        return out


def type_body_order(tdoc: dict) -> list:
    """The type's bodies with every parent before its children."""
    out, seen = [], set()
    names = sorted(tdoc["bodies"])
    while len(out) < len(names):
        for n in names:
            par = tdoc["bodies"][n]["parent"]
            if n not in seen and (par is None or par in seen):
                out.append(n)
                seen.add(n)
    return out


def part_body_poses(tdoc: dict, part_poses: dict, joint_values: dict) -> dict:
    """{body rel: (R, t)} of an articulated type from its parts' frames: ``part_poses`` {part: (R, t)} for the root
    part (and optionally others), ``joint_values`` {part: q} for the jointed parts (defaults to qpos0). A jointed
    part's frame is its parent body's frame composed with the body offset and the joint at q."""
    poses = {}
    order = type_body_order(tdoc)
    pip = tdoc["pose_in_part"]
    for r in order:
        pa = tdoc["part_of"][r]
        if pa == r:
            if pa in part_poses:
                Rpa, tpa = part_poses[pa]
            else:
                b = tdoc["bodies"][r]
                Rp, tp = poses[b["parent"]]
                Rpa = Rp @ quat_to_mat(b["body_quat"])
                tpa = tp + Rp @ np.asarray(b["body_pos"], float)
                jt = tdoc["parts"][pa]["joint"]
                if jt is not None:
                    Rpa, tpa = apply_joint(Rpa, tpa, jt, joint_values.get(pa, jt.get("qpos0", 0.0)))
            poses[pa] = (np.asarray(Rpa, float), np.asarray(tpa, float))
        if r != pa:
            Rpa, tpa = poses[pa]
            Ri, ti = np.asarray(pip[r]["R"], float), np.asarray(pip[r]["t"], float)
            poses[r] = (Rpa @ Ri, Rpa @ ti + tpa)
    return poses
