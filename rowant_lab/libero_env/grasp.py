"""M1 grasp layer: collision primitives, parallel-jaw grasp records and top-down frame selection (numpy only).

ORACLE (true poses): diagnostic of the hands. This module is truth-free given its inputs: it plans on the object
models of the library (collision primitives in each body's frame) placed at the poses it is given. In M1 those poses
come from the declared oracle tool; in M2 the same calls take perceived poses.

Design (plan agent_design, as amended by amendment A1):

* **Geometry.** Every collision primitive is an oriented box (LIBERO's objects and fixtures collide as unions of
  boxes). ``Boxes`` holds n boxes (centre, rotation whose columns are the box axes, half-size) in one frame.
  ``line_hits`` intersects lines with boxes (slab method, exact entry/exit and face normals) and ``overlap`` tests
  box pairs with the separating-axis theorem (15 axes, exact for boxes).
* **Grasp records** (``build_records``), in the object's root-body frame, from the collision model only: for closing
  directions every ``theta_step_deg`` in the plane normal to the rest-pose up axis, lateral offsets every
  ``b_step_m`` and grip-site heights every ``z_step_m``, the jaw face (``JawModel``: the Panda pad and the finger's
  lower inner face, a ``face_x`` x ``face_z`` rectangle) is swept along the closing line from both sides. The solid
  the face meets is the union, along the line, of every box interval on a grid of sub-rays covering the face. Each
  connected component [s0, s1] of that union is a candidate: centred on the component (both jaws touch at once),
  width ``w = s1 - s0`` within [w_min, 2 L - 2 jaw_clearance], no other solid between the open jaw and the
  component, nor within the finger's thickness beyond the open jaw, contact normals within ``normal_tol_deg`` of
  the closing axis on both sides, and the fingertip at least ``tip_above_bottom_m`` above the object's bottom.
  (Optionally both jaws must touch along the face's central column, ``require_centre_contact``; off, as DEV
  success fell with it on: amendment A1.)
  This replaces the plan's voxel/marching-cubes mesh plus E-0b's ``grasp_library`` (amendment A1): the ray tests run
  on the exact primitives, so there is no meshing error and the bowl's 2.6 mm walls need no 0.3 mm voxels.
* **Runtime frames** (``plan_frames``): records placed at the target's pose, kept when the closing axis is within
  ``axis_tilt_deg`` of horizontal and the record's up axis within ``up_tilt_deg`` of world up; the open gripper
  (hand, both fingers) swept from the pre-grasp height down to the grasp pose must miss every non-target primitive
  inflated by ``margin_m``, the arena's support boxes inflated by ``support_clear_m`` and the target itself; the jaw
  sweep regions (open jaw to contact) must miss every non-target primitive; the fingertip stays ``support_clear_m``
  above the floor. Ranking: horizontal distance from the grasp centre to the target's centre of mass (in
  ``rank_com_bin_m`` bins), then the contact height's distance from ``rank_height_frac`` of the object's height, then
  the yaw change. Accepted frames are at least ``spacing_m`` or ``spacing_deg`` apart, so each retry uses a
  different grasp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


# =============================================================================== rotations
def quat_wxyz_to_mat(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_xyzw_to_mat(q) -> np.ndarray:
    return quat_wxyz_to_mat([q[3], q[0], q[1], q[2]])


def rotvec_from_mat(R) -> np.ndarray:
    """Axis-angle vector (axis * angle, radians) of a rotation matrix; robust near 0 and pi."""
    R = np.asarray(R, dtype=float)
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    if ang > math.pi - 1e-4:
        M = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(M)))
        v = M[:, k] / math.sqrt(max(M[k, k], 1e-12))
        return unit(v) * ang
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(ang))
    return v * ang


def rot_angle_deg(R1, R2) -> float:
    return float(np.degrees(np.linalg.norm(rotvec_from_mat(np.asarray(R1) @ np.asarray(R2).T))))


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def perp_basis(u) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors e1, e2 with (e1, e2, u) right-handed and orthonormal."""
    u = unit(u)
    ref = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = unit(ref - (ref @ u) * u)
    return e1, np.cross(u, e1)


def hand_matrix(closing_axis) -> np.ndarray:
    """The eef (hand body) orientation for a top-down grasp: y along the closing axis (made horizontal), z down."""
    y = unit(np.array([closing_axis[0], closing_axis[1], 0.0]))
    z = -UP
    return np.column_stack([np.cross(y, z), y, z])


def yaw_of_hand(R_eef) -> float:
    """Yaw (radians) of the hand's closing axis (eef y) in the world xy plane."""
    y = np.asarray(R_eef)[:, 1]
    return math.atan2(y[1], y[0])


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# =============================================================================== boxes
@dataclass
class Boxes:
    """n oriented boxes in one frame: point = C + R @ local, |local_i| <= H_i."""

    C: np.ndarray
    R: np.ndarray
    H: np.ndarray
    tags: list = field(default_factory=list)

    def __post_init__(self):
        self.C = np.asarray(self.C, dtype=float).reshape(-1, 3)
        self.R = np.asarray(self.R, dtype=float).reshape(-1, 3, 3)
        self.H = np.asarray(self.H, dtype=float).reshape(-1, 3)
        if not self.tags:
            self.tags = [""] * len(self.C)

    @property
    def n(self) -> int:
        return len(self.C)

    @classmethod
    def empty(cls) -> "Boxes":
        return cls(np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3)), [])

    @classmethod
    def from_prims(cls, prims, tag: str = "") -> "Boxes":
        if not prims:
            return cls.empty()
        return cls([p["centre"] for p in prims], [p["R"] for p in prims], [p["half"] for p in prims],
                   [tag] * len(prims))

    @classmethod
    def aabb_box(cls, lo, hi, R=None, origin=None, tag: str = "") -> "Boxes":
        """One box given by its extent [lo, hi] in a frame (R, origin)."""
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        R = np.eye(3) if R is None else np.asarray(R, dtype=float)
        origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
        return cls([origin + R @ (0.5 * (lo + hi))], [R], [0.5 * (hi - lo)], [tag])

    def placed(self, Rb, tb) -> "Boxes":
        Rb, tb = np.asarray(Rb, dtype=float), np.asarray(tb, dtype=float)
        return Boxes(self.C @ Rb.T + tb, np.einsum("ij,njk->nik", Rb, self.R), self.H.copy(), list(self.tags))

    def inflated(self, m: float) -> "Boxes":
        return Boxes(self.C.copy(), self.R.copy(), self.H + float(m), list(self.tags))

    def corners(self) -> np.ndarray:
        s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
        return self.C[:, None, :] + np.einsum("nij,nkj->nki", self.R, s[None] * self.H[:, None, :])

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        ext = np.einsum("nij,nj->ni", np.abs(self.R), self.H)
        return self.C - ext, self.C + ext

    def __add__(self, other: "Boxes") -> "Boxes":
        if self.n == 0:
            return other
        if other.n == 0:
            return self
        return Boxes(np.vstack([self.C, other.C]), np.vstack([self.R, other.R]), np.vstack([self.H, other.H]),
                     list(self.tags) + list(other.tags))

    def subset(self, mask) -> "Boxes":
        mask = np.asarray(mask)
        idx = np.nonzero(mask)[0] if mask.dtype == bool else mask
        return Boxes(self.C[idx], self.R[idx], self.H[idx], [self.tags[i] for i in idx])

    def extent_along(self, d) -> tuple[float, float]:
        if self.n == 0:
            return 0.0, 0.0
        p = self.corners().reshape(-1, 3) @ unit(d)
        return float(p.min()), float(p.max())


def overlap(A: Boxes, B: Boxes) -> np.ndarray:
    """(nA, nB) bool: do the boxes intersect (separating-axis theorem, 15 axes; touching counts as overlap)."""
    out = np.zeros((A.n, B.n), dtype=bool)
    if A.n == 0 or B.n == 0:
        return out
    loA, hiA = A.aabb()
    loB, hiB = B.aabb()
    pre = np.all((loA[:, None, :] <= hiB[None, :, :]) & (loB[None, :, :] <= hiA[:, None, :]), axis=-1)
    ia, ib = np.nonzero(pre)
    if not len(ia):
        return out
    RA, RB, HA, HB = A.R[ia], B.R[ib], A.H[ia], B.H[ib]
    T = B.C[ib] - A.C[ia]
    axes = [RA[:, :, i] for i in range(3)] + [RB[:, :, j] for j in range(3)]
    axes += [np.cross(RA[:, :, i], RB[:, :, j]) for i in range(3) for j in range(3)]
    sep = np.zeros(len(ia), dtype=bool)
    for ax in axes:
        nrm = np.linalg.norm(ax, axis=1)
        ok = nrm > 1e-9
        axn = ax / np.where(ok, nrm, 1.0)[:, None]
        rA = np.sum(HA * np.abs(np.einsum("kij,ki->kj", RA, axn)), axis=1)
        rB = np.sum(HB * np.abs(np.einsum("kij,ki->kj", RB, axn)), axis=1)
        d = np.abs(np.sum(T * axn, axis=1))
        sep |= ok & (d > rA + rB)
    out[ia, ib] = ~sep
    return out


def any_overlap(A: Boxes, B: Boxes) -> bool:
    return bool(overlap(A, B).any())


def line_hits(O, D, B: Boxes):
    """Lines O + t D (m lines) against n boxes: (tin, tout, hit, n_in, n_out), each (m, n) [normals (m, n, 3)].

    ``tin``/``tout`` are the entry and exit parameters along the infinite line, ``hit`` where tout > tin; ``n_in`` is
    the outward face normal at entry and ``n_out`` at exit (world frame of the boxes)."""
    O = np.atleast_2d(np.asarray(O, dtype=float))
    D = np.atleast_2d(np.asarray(D, dtype=float))
    m, n = len(O), B.n
    if n == 0:
        z = np.zeros((m, 0))
        return z, z, z.astype(bool), np.zeros((m, 0, 3)), np.zeros((m, 0, 3))
    rel = O[:, None, :] - B.C[None, :, :]
    o = np.einsum("nij,mni->mnj", B.R, rel)
    d = np.einsum("nij,mi->mnj", B.R, D)
    H = B.H[None, :, :]
    par = np.abs(d) < 1e-12
    dd = np.where(par, 1.0, d)
    t1 = (-H - o) / dd
    t2 = (H - o) / dd
    lo = np.minimum(t1, t2)
    hi = np.maximum(t1, t2)
    inside = np.abs(o) <= H
    lo = np.where(par, np.where(inside, -np.inf, np.inf), lo)
    hi = np.where(par, np.where(inside, np.inf, -np.inf), hi)
    tin, tout = lo.max(-1), hi.min(-1)
    hit = tout > tin
    kin, kout = lo.argmax(-1), hi.argmin(-1)
    s_in = -np.sign(np.take_along_axis(d, kin[..., None], -1)[..., 0])
    s_out = np.sign(np.take_along_axis(d, kout[..., None], -1)[..., 0])
    RT = np.transpose(B.R, (0, 2, 1))  # [box, axis k, xyz]
    nb = np.arange(n)[None, :]
    n_in = RT[nb, kin] * s_in[..., None]
    n_out = RT[nb, kout] * s_out[..., None]
    return tin, tout, hit, n_in, n_out


# =============================================================================== the Panda jaw and gripper
@dataclass
class JawModel:
    """The Panda gripper in the grasp frame G (origin at the grip site, y = closing axis, z = world up).

    Measured by the exporter from robosuite's compiled collision geoms (library robot.json): the jaw face is the
    pad's inner face at ``q + face_offset`` from the centre (q = finger qpos) and the finger's lower inner face just
    behind it; it spans ``face_z`` (fingertip to pad top) and ``+-face_x``."""

    q_open: float = 0.0387
    face_offset: float = -0.0005
    face_x: float = 0.0086
    face_z: tuple = (-0.0093, 0.0116)
    finger_x: float = 0.0105
    finger_thick: float = 0.0265
    finger_z: tuple = (-0.0093, 0.0445)
    hand_lo: tuple = (-0.0316, -0.1040, 0.0310)
    hand_hi: tuple = (0.0316, 0.1040, 0.1229)

    @property
    def L(self) -> float:
        """Half the open inner gap (jaw face to centre)."""
        return self.q_open + self.face_offset

    def gap_from_qpos(self, q1: float, q2: float) -> float:
        """Inner gap between the jaw faces from the two finger joint positions (q2 is negative)."""
        return float(q1) - float(q2) + 2.0 * self.face_offset

    @classmethod
    def from_robot(cls, robot: dict) -> "JawModel":
        g = robot["geoms"]
        q = float(robot["finger_joints"]["gripper0_finger_joint1"]["qpos"])

        def up(lo, hi):  # eef frame (z down) -> G (z up): z -> -z
            return (-float(hi[2]), -float(lo[2]))

        pad_lo, pad_hi = g["gripper0_finger1_pad_collision"]["aabb_eef_lo"], g["gripper0_finger1_pad_collision"]["aabb_eef_hi"]
        fin_lo, fin_hi = g["gripper0_finger1_collision"]["aabb_eef_lo"], g["gripper0_finger1_collision"]["aabb_eef_hi"]
        hand_lo, hand_hi = g["gripper0_hand_collision"]["aabb_eef_lo"], g["gripper0_hand_collision"]["aabb_eef_hi"]
        fz = up(fin_lo, fin_hi)
        pz = up(pad_lo, pad_hi)
        hz = up(hand_lo, hand_hi)
        return cls(q_open=q, face_offset=float(pad_lo[1]) - q, face_x=float(max(abs(pad_lo[0]), abs(pad_hi[0]))),
                   face_z=(fz[0], pz[1]), finger_x=float(max(abs(fin_lo[0]), abs(fin_hi[0]))),
                   finger_thick=float(fin_hi[1]) - float(pad_lo[1]), finger_z=fz,
                   hand_lo=(float(hand_lo[0]), float(hand_lo[1]), hz[0]),
                   hand_hi=(float(hand_hi[0]), float(hand_hi[1]), hz[1]))

    def gripper_boxes_G(self, half_gap: float, sweep_up: float = 0.0, with_hand: bool = True) -> Boxes:
        """Finger (and hand) boxes in G with the jaw faces at +-half_gap, extended upward by ``sweep_up``."""
        z0, z1 = self.finger_z
        f1 = Boxes.aabb_box((-self.finger_x, half_gap, z0), (self.finger_x, half_gap + self.finger_thick,
                                                              z1 + sweep_up), tag="finger1")
        f2 = Boxes.aabb_box((-self.finger_x, -half_gap - self.finger_thick, z0), (self.finger_x, -half_gap,
                                                                                z1 + sweep_up), tag="finger2")
        out = f1 + f2
        if with_hand:
            hl, hh = np.array(self.hand_lo), np.array(self.hand_hi)
            out = out + Boxes.aabb_box(hl, hh + np.array([0, 0, sweep_up]), tag="hand")
        return out

    def jaw_sweep_boxes_G(self, half_w: float) -> Boxes:
        """The regions the two jaw faces sweep while closing from open to the contact at +-half_w."""
        L = self.L
        if half_w >= L:
            return Boxes.empty()
        z0, z1 = self.face_z
        return (Boxes.aabb_box((-self.face_x, half_w, z0), (self.face_x, L, z1), tag="sweep1")
                + Boxes.aabb_box((-self.face_x, -L, z0), (self.face_x, -half_w, z1), tag="sweep2"))


def G_matrix(closing_axis) -> np.ndarray:
    """Grasp frame G in the world: x = y x z, y = the horizontal closing axis, z = world up."""
    y = unit(np.array([closing_axis[0], closing_axis[1], 0.0]))
    return np.column_stack([np.cross(y, UP), y, UP])


# =============================================================================== grasp records
@dataclass
class RecordParams:
    theta_step_deg: float = 7.5
    b_step_m: float = 0.004
    z_step_m: float = 0.004
    sub_x: int = 5
    sub_z: int = 6
    jaw_clearance_m: float = 0.005
    w_min_m: float = 0.001
    normal_tol_deg: float = 25.0
    contact_tol_m: float = 0.001
    require_centre_contact: bool = False
    tip_above_bottom_m: float = 0.002
    top_margin_m: float = 0.002
    max_records: int = 3000


def object_axes(up) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = unit(up)
    e1, e2 = perp_basis(u)
    return e1, e2, u


def build_records(prims, up, com, jaw: JawModel, p: RecordParams = RecordParams(), name: str = "") -> dict:
    """Antipodal parallel-jaw grasp records in the object's root-body frame (see the module doc).

    Returns arrays: centre (k, 3) [grip-site target], axis (k, 3) [closing axis, unit, normal to ``up``], width,
    z_rel (grip-site height above the object's bottom along ``up``), n_pos / n_neg (contact sub-rays per jaw),
    nerr_pos / nerr_neg (normal error, deg), com_dist (horizontal distance of the centre from the COM), and the
    object's height and bottom along ``up``."""
    B = Boxes.from_prims(prims)
    e1, e2, u = object_axes(up)
    com = np.asarray(com, dtype=float)
    corners = B.corners().reshape(-1, 3)
    hu = corners @ u
    bottom, top = float(hu.min()), float(hu.max())
    height = top - bottom
    L = jaw.L
    reach = L + jaw.finger_thick
    w_max = 2 * L - 2 * p.jaw_clearance_m
    thetas = np.radians(np.arange(0.0, 180.0, p.theta_step_deg))
    fx = np.linspace(-jaw.face_x + 0.001, jaw.face_x - 0.001, p.sub_x)
    fz = np.linspace(jaw.face_z[0] + 0.0005, jaw.face_z[1] - 0.0005, p.sub_z)
    z_lo = bottom + p.tip_above_bottom_m - jaw.face_z[0]
    z_hi = top - p.top_margin_m
    zs = np.arange(z_lo, z_hi + 1e-9, p.z_step_m) if z_hi > z_lo else np.zeros(0)
    com_u = com - (com @ u) * u
    out = {k: [] for k in ("centre", "axis", "width", "z_rel", "n_pos", "n_neg", "nerr_pos", "nerr_neg",
                           "com_dist", "theta", "b")}
    for th in thetas:
        a = math.cos(th) * e1 + math.sin(th) * e2
        bv = np.cross(u, a)
        blo, bhi = (corners @ bv).min() - com @ bv, (corners @ bv).max() - com @ bv
        bs = np.arange(math.floor(blo / p.b_step_m) * p.b_step_m, bhi + 1e-9, p.b_step_m)
        if not len(bs) or not len(zs):
            continue
        BB, ZZ = np.meshgrid(bs, zs, indexing="ij")
        fam_b, fam_z = BB.ravel(), ZZ.ravel()
        F = len(fam_b)
        base = com_u[None, :] + fam_b[:, None] * bv[None, :] + fam_z[:, None] * u[None, :]  # (F, 3)
        sx, sz = np.meshgrid(fx, fz, indexing="ij")
        offs = sx.ravel()[:, None] * bv[None, :] + sz.ravel()[:, None] * u[None, :]  # (S, 3)
        S = len(offs)
        O = (base[:, None, :] + offs[None, :, :]).reshape(-1, 3)
        # every line origin has the COM's coordinate along a, so t is the coordinate along a measured from the COM
        tin, tout, hit, n_in, n_out = line_hits(O, np.repeat(a[None], len(O), 0), B)
        tin = tin.reshape(F, S * B.n)
        tout = tout.reshape(F, S * B.n)
        hit = hit.reshape(F, S * B.n)
        n_in = n_in.reshape(F, S * B.n, 3)
        n_out = n_out.reshape(F, S * B.n, 3)
        sub_idx = np.repeat(np.arange(S), B.n)
        centre_subs = np.nonzero(np.abs(sx.ravel()) < 1e-9)[0]  # the face's central column (pad centre line)
        s0 = np.where(hit, tin, np.inf)
        s1 = np.where(hit, tout, -np.inf)
        order = np.argsort(s0, axis=1)
        s0o = np.take_along_axis(s0, order, 1)
        s1o = np.take_along_axis(s1, order, 1)
        cm = np.maximum.accumulate(s1o, axis=1)
        new = np.zeros_like(s0o, dtype=bool)
        new[:, 0] = np.isfinite(s0o[:, 0])
        new[:, 1:] = np.isfinite(s0o[:, 1:]) & (s0o[:, 1:] > cm[:, :-1])
        for f in np.nonzero(new.any(axis=1))[0]:
            starts = np.nonzero(new[f])[0]
            valid_end = np.isfinite(s0o[f])
            last = int(np.nonzero(valid_end)[0][-1])
            ends_idx = list(starts[1:] - 1) + [last]
            comps = [(float(s0o[f, i]), float(cm[f, j])) for i, j in zip(starts, ends_idx)]
            for ci, (c0, c1) in enumerate(comps):
                w = c1 - c0
                if not (p.w_min_m <= w <= w_max):
                    continue
                c = 0.5 * (c0 + c1)
                lo_free, hi_free = c - reach, c + reach
                if any(o1 > lo_free and o0 < c0 for o0, o1 in comps[:ci]) or \
                        any(o0 < hi_free and o1 > c1 for o0, o1 in comps[ci + 1:]):
                    continue
                inside = hit[f] & (tin[f] >= c0 - 1e-9) & (tout[f] <= c1 + 1e-9)
                pos = inside & (tout[f] >= c1 - p.contact_tol_m)
                neg = inside & (tin[f] <= c0 + p.contact_tol_m)
                if not pos.any() or not neg.any():
                    continue
                if p.require_centre_contact and not (np.isin(sub_idx[pos], centre_subs).any()
                                                     and np.isin(sub_idx[neg], centre_subs).any()):
                    continue
                ang_p = np.degrees(np.arccos(np.clip(n_out[f][pos] @ a, -1, 1)))
                ang_n = np.degrees(np.arccos(np.clip(n_in[f][neg] @ (-a), -1, 1)))
                if ang_p.min() > p.normal_tol_deg or ang_n.min() > p.normal_tol_deg:
                    continue
                centre = base[f] + c * a  # base[f] has the COM's coordinate along a (u is normal to a)
                out["centre"].append(centre)
                out["axis"].append(a)
                out["width"].append(w)
                out["z_rel"].append(float(fam_z[f] - bottom))
                out["n_pos"].append(int(len(np.unique(sub_idx[pos]))))
                out["n_neg"].append(int(len(np.unique(sub_idx[neg]))))
                out["nerr_pos"].append(float(ang_p.min()))
                out["nerr_neg"].append(float(ang_n.min()))
                cc = centre - com
                out["com_dist"].append(float(np.linalg.norm(cc - (cc @ u) * u)))
                out["theta"].append(float(th))
                out["b"].append(float(fam_b[f]))
    rec = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    rec["n_records"] = int(len(rec["width"]))
    if rec["n_records"] > p.max_records:  # keep the best by COM distance, deterministically
        keep = np.argsort(rec["com_dist"], kind="stable")[:p.max_records]
        keep.sort()
        for k in list(out):
            rec[k] = rec[k][keep]
        rec["n_records_kept"] = p.max_records
    rec.update(height=height, bottom=bottom, top=top, up=np.asarray(u), com=com, name=name)
    return rec


def records_to_json(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if isinstance(v, np.ndarray):
            out[k] = np.round(v, 7).tolist()
        else:
            out[k] = v
    return out


def records_from_json(d: dict) -> dict:
    rec = {}
    for k, v in d.items():
        rec[k] = np.asarray(v, dtype=float) if isinstance(v, list) else v
    for k in ("centre", "axis"):
        rec[k] = rec[k].reshape(-1, 3)
    return rec


# =============================================================================== runtime frame selection
@dataclass
class PlanParams:
    axis_tilt_deg: float = 15.0
    up_tilt_deg: float = 15.0
    pre_height_m: float = 0.10
    margin_m: float = 0.003
    support_clear_m: float = 0.002
    target_margin_m: float = 0.0
    rank_com_bin_m: float = 0.005
    rank_height_frac: float = 0.65
    spacing_m: float = 0.010
    spacing_deg: float = 30.0
    k_frames: int = 8
    min_n_contact: int = 2
    reach_soft_m: float = 0.72
    reach_hard_m: float = 0.85
    reach_bin_m: float = 0.02
    flip_max_deg: float = 0.0


def reach_dist(site, yaw: float, shoulder, pp: "PlanParams") -> float:
    """The arm's horizontal reach to a grip-site position: its xy distance from the shoulder (Panda link 2). The
    yaw is not used (DEV showed a wrist-offset model ranking the wrong side of the bowl as nearer; amendment A4)."""
    d = np.asarray(site, dtype=float)[:2] - np.asarray(shoulder, dtype=float)[:2]
    return float(np.hypot(d[0], d[1]))


def place_records(rec: dict, R_obj, t_obj) -> dict:
    R_obj, t_obj = np.asarray(R_obj, dtype=float), np.asarray(t_obj, dtype=float)
    return {"centre": rec["centre"] @ R_obj.T + t_obj, "axis": rec["axis"] @ R_obj.T, "up": R_obj @ rec["up"],
            "com": R_obj @ rec["com"] + t_obj}


def plan_frames(rec: dict, R_obj, t_obj, target: Boxes, obstacles: Boxes, supports: Boxes, floor_z: float | None,
                jaw: JawModel, current_yaw: float, pp: PlanParams = PlanParams(),
                shoulder=None, reach_cap: float | None = None) -> tuple[list[dict], dict]:
    """World-frame top-down grasp frames for the target, best first (see the module doc), and a diagnostic."""
    W = place_records(rec, R_obj, t_obj)
    diag = {"n_records": int(len(rec["width"]))}
    if not len(rec["width"]):
        return [], diag
    up_ok = float(W["up"] @ UP) >= math.cos(math.radians(pp.up_tilt_deg))
    diag["up_ok"] = up_ok
    ax = W["axis"]
    ok = np.abs(ax @ UP) <= math.sin(math.radians(pp.axis_tilt_deg))
    ok &= (rec["n_pos"] >= pp.min_n_contact) & (rec["n_neg"] >= pp.min_n_contact)
    if not up_ok:
        ok &= False
    diag["n_axis_ok"] = int(ok.sum())
    idx = np.nonzero(ok)[0]
    cands = []
    z_tip = jaw.finger_z[0]
    n_reach_drop = 0
    for i in idx:
        c = W["centre"][i]
        a = ax[i]
        if floor_z is not None and c[2] + z_tip < floor_z + pp.support_clear_m:
            continue
        # yaw: the closing-axis sign nearest the current yaw (mod 180 deg); the flipped hand (the other sign) is a
        # second option, used when it is needed for reach (its wrist sits on the other side of the grip site)
        yaw = math.atan2(a[1], a[0])
        d1 = wrap_pi(yaw - current_yaw)
        if abs(d1) > math.pi / 2:
            yaw = wrap_pi(yaw + math.pi)
            d1 = wrap_pi(yaw - current_yaw)
        opts = [(yaw, abs(d1))]
        flip = wrap_pi(yaw + math.pi)
        if math.degrees(abs(wrap_pi(flip - current_yaw))) <= pp.flip_max_deg:
            opts.append((flip, abs(wrap_pi(flip - current_yaw))))
        for (y, dy) in opts:
            rd = reach_dist(c, y, shoulder, pp) if shoulder is not None else 0.0
            if rd > (pp.reach_hard_m if reach_cap is None else min(pp.reach_hard_m, reach_cap)):
                n_reach_drop += 1
                continue
            cands.append((i, c, np.array([math.cos(y), math.sin(y), 0.0]), y, dy, rd))
    diag["n_floor_ok"] = len(cands)
    diag["n_reach_drop"] = n_reach_drop
    if not cands:
        return [], diag
    # collision checks, vectorised per candidate
    obst_i = obstacles.inflated(pp.margin_m) if obstacles.n else obstacles
    sup_i = supports.inflated(pp.support_clear_m) if supports.n else supports
    tgt_i = target.inflated(pp.target_margin_m) if pp.target_margin_m and target.n else target
    keep = []
    n_coll = {"obstacle": 0, "support": 0, "target": 0, "sweep": 0}
    for (i, c, a_h, yaw, dy, rd) in cands:
        G = G_matrix(a_h)
        grip = jaw.gripper_boxes_G(jaw.L, sweep_up=pp.pre_height_m).placed(G, c)
        if obst_i.n and any_overlap(grip, obst_i):
            n_coll["obstacle"] += 1
            continue
        if sup_i.n and any_overlap(grip, sup_i):
            n_coll["support"] += 1
            continue
        if tgt_i.n and any_overlap(grip, tgt_i):
            n_coll["target"] += 1
            continue
        sw = jaw.jaw_sweep_boxes_G(0.5 * float(rec["width"][i])).placed(G, c)
        if sw.n and obst_i.n and any_overlap(sw, obst_i):
            n_coll["sweep"] += 1
            continue
        keep.append((i, c, a_h, yaw, dy, rd))
    diag["collisions"] = n_coll
    diag["n_free"] = len(keep)
    if not keep:
        return [], diag
    h = float(rec["height"])
    keys = []
    for (i, c, a_h, yaw, dy, rd) in keep:
        keys.append((math.ceil(max(0.0, rd - pp.reach_soft_m) / pp.reach_bin_m),
                     round(float(rec["com_dist"][i]) / pp.rank_com_bin_m),
                     abs(float(rec["z_rel"][i]) / max(h, 1e-6) - pp.rank_height_frac), dy))
    order = sorted(range(len(keep)), key=lambda j: keys[j])
    frames = []
    for j in order:
        i, c, a_h, yaw, dy, rd = keep[j]
        near = False
        for f in frames:
            if np.linalg.norm(f["centre"] - c) < pp.spacing_m:
                ang = abs(math.degrees(wrap_pi(2 * (f["yaw"] - yaw)))) / 2.0
                if ang < pp.spacing_deg:
                    near = True
                    break
        if near:
            continue
        frames.append({"record": int(i), "centre": c, "axis": a_h, "yaw": yaw, "yaw_change_deg": math.degrees(dy),
                       "width": float(rec["width"][i]), "z_rel": float(rec["z_rel"][i]),
                       "com_dist": float(rec["com_dist"][i]), "reach_m": rd, "R_eef": hand_matrix(a_h),
                       "rank": len(frames)})
        if len(frames) >= pp.k_frames:
            break
    return frames, diag
