"""TRUTH-FREE: geometric near-symmetries of an object, detected from its mesh at onboarding, and the flip family
used to expand the refined candidates (fastlayer step 3).

Why. Step 2's main failure was a near-symmetry flip: the posed surface fits the depth, but the rotation is wrong by
about 180 deg (an upside-down can, a box turned front to back), and for two thirds of the instances the right flip
never reached the candidate list. So every refined candidate is expanded by the object's flip family; the flips are
refined the same way, and the decision step (geometry, or geometry plus an appearance scorer) chooses among them.

Detection (``detect``; the mesh only: BOP's ``obj_XXXXXX.ply`` in mm, read with ``onboard.read_ply``):

* ``N_REF`` area-weighted surface samples (seeded by the object id) form the reference cloud (a k-d tree), and
  ``N_TEST`` other samples are the test points;
* centre ``c``: the area-weighted surface centroid; principal axes: the eigenvectors of the samples' covariance
  about ``c``, largest variance first, right-handed;
* diameter ``D``: the largest distance between two mesh vertices (on the convex hull);
* deviation of a rotation ``F`` about ``c``: the mean distance from each turned test point ``c + F (x - c)`` to the
  nearest reference point, as a fraction of ``D``. This is ADD-S between a pose and its flipped pose, the same
  measure as the step-2 breakdown's near-symmetry class (posed surfaces within 0.05 x ``D``). ``floor``: the
  deviation of the identity (sampling noise only);
* frame. Without a rotational axis: the principal axis whose variance stands most apart, then the perpendicular
  direction whose half turn deviates least (searched: two equal variances leave the principal axes in that plane
  arbitrary, e.g. a square cross-section), then their cross product;
* rotational axis. The principal axes are unreliable when two variances are close (a can as tall as it is wide),
  so the axis is searched: the principal axes and ``N_AXES`` Fibonacci directions (one hemisphere) are screened by
  the largest deviation of the turns by ``SCREEN_DEG``; the best one is refined on a local grid (``LOCAL_DEG``,
  ``LOCAL_N``) and then checked on every ``TURN_GRID_DEG`` turn. The mesh is close to rotationally symmetric when all
  of them are within ``ROT_FRAC``. Its frame is then (axis, the perpendicular direction whose half turn deviates
  least, their cross product);
* a rotation is a near-symmetry when its deviation is at most ``NEAR_FRAC``. Discrete turns are only tested at the
  n-fold angles ``DISCRETE_DEG`` (at least 60 deg from the identity): small turns have a small ADD-S but are not
  distinct poses (refinement handles them). A discrete turn (and a product of two family members that is not a
  half turn about a principal axis) joins the family only when its deviation is at most ``DISCRETE_FRAC``, the
  same bar as the rotational test: ADD-S is lenient for flat and long shapes (a flat box turned 60 deg about its
  face normal keeps most of its area on its own two faces), and every such member costs one refinement per base.

Flip family (``build_family``; each element is a rotation ``F = G Z`` of the object frame about ``c``: a
generator ``G`` and an optional turn ``Z`` about the rotational axis):

* rotational axis: turns ``Z_j`` by ``j x 360 / n`` deg, ``n`` the smallest of ``TURN_COUNTS`` whose half step
  moves the surface point farthest from the axis by at most ``TURN_DISP_FRAC x D`` (else the largest). With ``h``
  the perpendicular half turn: if ``h`` is a near-symmetry the family is ``{I, h} x {Z_j}`` (upright and upside
  down at every turn); otherwise it is the turns plus the two perpendicular half turns;
* no rotational axis: the three half turns about the principal axes (always, near or not), plus every n-fold turn
  about a principal axis within ``DISCRETE_FRAC``, plus the products of two near members that are within
  ``DISCRETE_FRAC`` (e.g. a square cross-section's diagonal half turns); at most ``MAX_FAMILY`` elements (the half
  turns, then the smallest deviations).

Every element records its deviation and its ``near`` flag. A flip of a pose ``(R, t)`` (object-local ``x`` placed
at ``R x + t``) is ``(R F, t + R (c - F c))``: the centre stays where it was.

Refinement modes (the layer's ``flip_refine``): ``each`` refines every flipped pose with the same ICP; ``generators``
refines ``base x G`` for each generator and then applies the turns ``Z`` to the refined pose (the turns are
geometric near-symmetries by construction, so refining them again would mostly redo the same fit).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from . import onboard as OB

TRUTH_FREE = True
FORMAT = "rowant fastlayer symmetry v1"
SYM_SEED = 20261022
N_REF = 40000
N_TEST = 4000
TURN_GRID_DEG = 15.0
NEAR_FRAC = 0.05
ROT_FRAC = 0.02
DISCRETE_FRAC = 0.02
TURN_DISP_FRAC = 0.05
TURN_COUNTS = (4, 6, 8, 12, 16, 18, 24, 36)
DISCRETE_DEG = (60.0, 90.0, 120.0, 240.0, 270.0, 300.0)
N_AXES = 256
SCREEN_DEG = (45.0, 90.0, 135.0, 180.0)
LOCAL_DEG = 8.0
LOCAL_N = 40
N_PERP = 24
MAX_FAMILY = 24
SAME_DEG = 1.0
PARAMS = {"SYM_SEED": SYM_SEED, "N_REF": N_REF, "N_TEST": N_TEST, "TURN_GRID_DEG": TURN_GRID_DEG,
          "NEAR_FRAC": NEAR_FRAC, "ROT_FRAC": ROT_FRAC, "DISCRETE_FRAC": DISCRETE_FRAC,
          "TURN_DISP_FRAC": TURN_DISP_FRAC,
          "TURN_COUNTS": list(TURN_COUNTS), "DISCRETE_DEG": list(DISCRETE_DEG), "N_AXES": N_AXES,
          "SCREEN_DEG": list(SCREEN_DEG), "LOCAL_DEG": LOCAL_DEG, "LOCAL_N": LOCAL_N, "N_PERP": N_PERP,
          "MAX_FAMILY": MAX_FAMILY, "SAME_DEG": SAME_DEG}


# ------------------------------------------------------------------------------------------- small maths
def axis_rotation(axis, deg: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    th = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def angle_deg(R1, R2) -> float:
    c = (np.trace(np.asarray(R1).T @ np.asarray(R2)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def flip_pose(R, t, F, centre) -> tuple[np.ndarray, np.ndarray]:
    """The pose with the object turned by ``F`` about its centre (module doc)."""
    R = np.asarray(R, float)
    F = np.asarray(F, float)
    c = np.asarray(centre, float)
    return R @ F, np.asarray(t, float) + R @ (c - F @ c)


def centre_world(R, t, centre) -> np.ndarray:
    return np.asarray(R, float) @ np.asarray(centre, float) + np.asarray(t, float)


# ------------------------------------------------------------------------------------------- detection
def symmetry_rng(obj_id: str) -> np.random.Generator:
    word = int.from_bytes(hashlib.sha256(f"symmetry:{obj_id}".encode()).digest()[:8], "big")
    return np.random.default_rng([SYM_SEED, word])


def principal_frame(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre, axes as columns e0 e1 e2 (largest variance first, right-handed), variances)."""
    c = pts.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((pts - c).T))
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    if np.linalg.det(v) < 0:
        v[:, 2] *= -1
    return c, v, w


def diameter(V: np.ndarray) -> float:
    try:
        P = V[ConvexHull(V).vertices]
    except Exception:  # noqa: BLE001 (degenerate hull: fall back to every vertex)
        P = V
    if len(P) > 4000:
        P = P[np.linspace(0, len(P) - 1, 4000).astype(int)]
    d2 = np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2)
    return float(np.sqrt(d2.max()))


class Deviation:
    """Deviation of rotations about the centre (module doc), as a fraction of the diameter."""

    def __init__(self, ref: np.ndarray, test: np.ndarray, centre: np.ndarray, diam: float):
        self.tree = cKDTree(ref)
        self.test = test - centre
        self.centre = centre
        self.diam = diam

    def __call__(self, F: np.ndarray) -> float:
        q = self.test @ np.asarray(F).T + self.centre
        d, _ = self.tree.query(q, k=1, workers=1)
        return float(d.mean() / self.diam)


def turn_count(rho_max: float, diam: float) -> int:
    for n in TURN_COUNTS:
        if 2.0 * rho_max * math.sin(math.pi / (2 * n)) <= TURN_DISP_FRAC * diam:
            return int(n)
    return int(TURN_COUNTS[-1])


def _el(label: str, G: np.ndarray, gen: str, turn_deg: float, axis, dev: Deviation) -> dict:
    Z = axis_rotation(axis, turn_deg) if axis is not None and turn_deg else np.eye(3)
    F = G @ Z
    d = dev(F)
    return {"label": label, "R": F, "G": G, "gen": gen, "turn_deg": float(turn_deg), "dev_frac": d,
            "near": bool(d <= NEAR_FRAC)}


def build_family(axes: np.ndarray, half: list, discrete: dict, rot_axis, n_turns: int, dev: Deviation) -> list[dict]:
    """The flip family (module doc). ``axes``: the frame's columns; ``half``: deviations of the half turns about
    them; ``discrete``: {axis: {angle: deviation}} of the n-fold turns (no rotational axis)."""
    H = [axis_rotation(axes[:, k], 180.0) for k in range(3)]
    I3 = np.eye(3)
    fam: list[dict] = []
    if rot_axis is not None:
        k = int(rot_axis)
        a = axes[:, k]
        perp = [m for m in range(3) if m != k]
        m = min(perp, key=lambda j: half[j])
        step = 360.0 / n_turns
        gens = [("I", I3)]
        if half[m] <= NEAR_FRAC:
            gens.append((f"h{m}", H[m]))
        for gname, G in gens:
            for j in range(n_turns):
                if gname == "I" and j == 0:
                    continue
                lab = gname if j == 0 else (f"z{k}:{j * step:g}" if gname == "I" else f"{gname}*z{k}:{j * step:g}")
                fam.append(_el(lab, G, gname, j * step, a, dev))
        if len(gens) == 1:
            for mm in perp:
                fam.append(_el(f"h{mm}", H[mm], f"h{mm}", 0.0, None, dev))
        return fam
    base = [_el(f"h{k}", H[k], f"h{k}", 0.0, None, dev) for k in range(3)]
    for k in range(3):
        for th, dv in sorted(discrete[k].items()):
            if dv <= DISCRETE_FRAC:
                base.append(_el(f"t{k}:{th:g}", axis_rotation(axes[:, k], th), f"t{k}:{th:g}", 0.0, None, dev))
    fam = list(base)

    def known(R):
        return angle_deg(R, I3) < SAME_DEG or any(angle_deg(R, e["R"]) < SAME_DEG for e in fam)

    near = [e for e in base if e["near"]]
    for e1 in near:
        for e2 in near:
            P = e1["R"] @ e2["R"]
            if known(P):
                continue
            d = dev(P)
            if d <= DISCRETE_FRAC:
                lab = f"{e1['label']}*{e2['label']}"
                fam.append({"label": lab, "R": P, "G": P, "gen": lab, "turn_deg": 0.0, "dev_frac": d, "near": True})
    if len(fam) > MAX_FAMILY:
        fam = fam[:3] + sorted(fam[3:], key=lambda e: e["dev_frac"])[:MAX_FAMILY - 3]
    return fam


def hemisphere(n: int) -> np.ndarray:
    """``n`` Fibonacci directions with z >= 0 (an axis and its opposite are the same axis)."""
    i = np.arange(n) + 0.5
    z = i / n
    r = np.sqrt(1 - z * z)
    phi = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def local_axes(a: np.ndarray, radius_deg: float, n: int) -> np.ndarray:
    """``a`` and ``n`` directions within ``radius_deg`` of it (a sunflower on the cap)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    out = [a]
    for i in range(n):
        r = math.radians(radius_deg) * math.sqrt((i + 0.5) / n)
        phi = math.pi * (1 + 5 ** 0.5) * i
        d = math.cos(r) * a + math.sin(r) * (math.cos(phi) * u + math.sin(phi) * v)
        out.append(d / np.linalg.norm(d))
    return np.stack(out)


def screen(dev: Deviation, a: np.ndarray) -> float:
    return max(dev(axis_rotation(a, th)) for th in SCREEN_DEG)


def find_rotational_axis(dev: Deviation, pca_axes: np.ndarray) -> dict:
    """The best rotational-axis candidate (module doc): screened over the principal axes and a hemisphere, refined
    locally, then checked on the full turn grid."""
    cands = np.concatenate([pca_axes.T, hemisphere(N_AXES)])
    sc = [screen(dev, a) for a in cands]
    i = int(np.argmin(sc))
    best, best_sc = cands[i], sc[i]
    for a in local_axes(best, LOCAL_DEG, LOCAL_N)[1:]:
        s_ = screen(dev, a)
        if s_ < best_sc:
            best, best_sc = a, s_
    grid = np.arange(TURN_GRID_DEG, 360.0, TURN_GRID_DEG)
    devs = [dev(axis_rotation(best, th)) for th in grid]
    pca_cos = np.abs(pca_axes.T @ best)
    return {"axis": best / np.linalg.norm(best), "screen": float(best_sc), "grid_dev_frac": devs,
            "worst": float(max(devs)), "from": "pca" if i < 3 else "sphere",
            "nearest_pca_deg": float(np.degrees(np.arccos(np.clip(pca_cos.max(), -1, 1)))),
            "n_screened": int(len(cands))}


def perpendicular_frame(dev: Deviation, a: np.ndarray, hints=()) -> tuple[np.ndarray, list]:
    """Columns (a, b, a x b): b the direction perpendicular to ``a`` whose half turn deviates least. Tried: ``N_PERP``
    directions over 180 deg and the ``hints`` projected on the plane, then a 0.5 deg grid around the best."""
    a = a / np.linalg.norm(a)
    u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    phis = [math.pi * j / N_PERP for j in range(N_PERP)]
    for h in hints:
        hp = np.asarray(h, float) - (np.asarray(h, float) @ a) * a
        if np.linalg.norm(hp) > 1e-6:
            phis.append(math.atan2(hp @ v, hp @ u) % math.pi)
    tries = [(dev(axis_rotation(math.cos(ph) * u + math.sin(ph) * v, 180.0)), ph) for ph in phis]
    d0, ph0 = min(tries)
    half = math.pi / N_PERP / 2
    for ph in ph0 + np.radians(np.arange(-math.degrees(half), math.degrees(half) + 1e-9, 0.5)):
        d = dev(axis_rotation(math.cos(ph) * u + math.sin(ph) * v, 180.0))
        if d < d0:
            d0, ph0 = d, float(ph)
    b = math.cos(ph0) * u + math.sin(ph0) * v
    return np.stack([a, b, np.cross(a, b)], axis=1), [float(x[0]) for x in tries[:N_PERP]]


def distinct_axis(pca: np.ndarray, var: np.ndarray) -> int:
    """The principal axis whose variance stands most apart from the other two (the largest or the smallest)."""
    w = np.maximum(np.asarray(var, float), 1e-30)
    return 0 if w[0] / w[1] >= w[1] / w[2] else 2


def detect(obj_id: str, ply_path) -> dict:
    """The symmetry record of one object (module doc). JSON-ready (rotations as nested lists)."""
    t0 = time.perf_counter()
    mesh = OB.read_ply(ply_path)
    V = mesh["vertices_mm"] * 1e-3
    F = mesh["faces"]
    rng = symmetry_rng(str(obj_id))
    fi, bary = OB.sample_surface(V, F, N_REF + N_TEST, rng)
    pts = OB.interp(V, F, fi, bary)
    ref, test = pts[:N_REF], pts[N_REF:]
    c, pca, var = principal_frame(ref)
    D = diameter(V)
    dev = Deviation(ref, test, c, D)
    floor = dev(np.eye(3))
    rot = find_rotational_axis(dev, pca)
    rot_axis = rho = n_turns = None
    perp_devs = None
    discrete = {k: {} for k in range(3)}
    if rot["worst"] <= ROT_FRAC:
        axes, perp_devs = perpendicular_frame(dev, rot["axis"], hints=pca.T)
        rot_axis = 0
        a = axes[:, 0]
        rel = V - c
        rho = float(np.max(np.linalg.norm(rel - np.outer(rel @ a, a), axis=1)))
        n_turns = turn_count(rho, D)
    else:
        k0 = distinct_axis(pca, var)
        axes, perp_devs = perpendicular_frame(dev, pca[:, k0], hints=[pca[:, k] for k in range(3) if k != k0])
        for k in range(3):
            discrete[k] = {float(th): dev(axis_rotation(axes[:, k], th)) for th in DISCRETE_DEG}
    half = [dev(axis_rotation(axes[:, k], 180.0)) for k in range(3)]
    fam = build_family(axes, half, discrete, rot_axis, n_turns or 0, dev)
    t1 = time.perf_counter()
    return {
        "obj": str(obj_id), "ply": str(ply_path), "ply_sha256": OB.sha256_file(ply_path),
        "centre_m": c.tolist(), "pca_axes_columns": pca.tolist(), "variances_m2": var.tolist(), "diameter_m": D,
        "axes_columns": axes.tolist(), "floor_dev_frac": floor, "half_turn_dev_frac": half,
        "rotational_search": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rot.items()},
        "rotational_axis": rot_axis, "rho_max_m": rho, "n_turns": n_turns,
        "turn_step_deg": (360.0 / n_turns) if n_turns else None, "perpendicular_half_turn_dev_frac": perp_devs,
        "discrete_turn_dev_frac": {str(k): {f"{th:g}": v for th, v in d.items()} for k, d in discrete.items()},
        "family": [{**{k: v for k, v in e.items() if k not in ("R", "G")}, "R": np.asarray(e["R"]).tolist(),
                    "G": np.asarray(e["G"]).tolist()} for e in fam],
        "n_family": len(fam), "n_near": int(sum(e["near"] for e in fam)), "seconds": round(t1 - t0, 4),
    }


def build_all(models_dir, obj_ids, out_json) -> dict:
    """Detect every object's symmetries; write one JSON (exclusive create)."""
    models_dir = Path(models_dir)
    t0 = time.perf_counter()
    objs = {str(int(o)): detect(str(int(o)), models_dir / f"obj_{int(o):06d}.ply") for o in obj_ids}
    doc = {"format": FORMAT, "truth_free": True, "params": PARAMS, "models_dir": str(models_dir), "objects": objs,
           "wall_s": round(time.perf_counter() - t0, 3)}
    text = json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n"
    with open(out_json, "x", encoding="utf-8") as f:
        f.write(text)
    return doc


# ------------------------------------------------------------------------------------------- run time
class SymmetryTable:
    """The flip families of a memory's objects, as the layer uses them."""

    def __init__(self, doc: dict, sha256: str | None = None):
        if doc.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT!r} document")
        self.doc = doc
        self.sha256 = sha256
        self.objects = {}
        for g, rec in doc["objects"].items():
            fam = [{**e, "R": np.asarray(e["R"], float), "G": np.asarray(e["G"], float)} for e in rec["family"]]
            ax = rec["rotational_axis"]
            self.objects[g] = {"centre": np.asarray(rec["centre_m"], float),
                               "axes": np.asarray(rec["axes_columns"], float), "family": fam,
                               "rot_axis": ax, "n_turns": rec["n_turns"], "diameter": float(rec["diameter_m"]),
                               "turn_step_deg": rec["turn_step_deg"]}

    @classmethod
    def load(cls, path, expect_sha256: str | None = None) -> "SymmetryTable":
        raw = Path(path).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if expect_sha256 is not None and sha != expect_sha256:
            raise ValueError(f"{path}: sha256 {sha} != pinned {expect_sha256}")
        return cls(json.loads(raw.decode("utf-8")), sha)

    def __contains__(self, g) -> bool:
        return str(g) in self.objects

    def get(self, g) -> dict:
        return self.objects[str(g)]

    def summary(self) -> dict:
        return {g: {"rotational_axis": o["rot_axis"], "n_turns": o["n_turns"], "n_family": len(o["family"]),
                    "n_near": int(sum(e["near"] for e in o["family"]))} for g, o in self.objects.items()}
