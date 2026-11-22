"""TRUTH-FREE: appearance scorer A of fastlayer step 3, Monty-only (item ``bop3:monty-colour``): colour and texture
evidence per graph node, the Monty way (a clipped, tolerance-scaled agreement of each node's features with what the
sensor sees at that node), made able to tell near-symmetric flips apart.

**Why step 2's per-point hsv did not settle the flips** (measured on the tune side).
CHECK (``lnh``) compares the hsv of 256 random patch centres (one camera pixel each) with the hsv of the 10 graph
nodes nearest each centre's 3D point (one bilinear texel each): ``max over the 10 of (pose + mean_w(clip(tol - |d|,
0) / tol))``, tolerances (0.1, 0.2, 0.2), weights (2, 0.5, 0.5).

* It is not blind: without it (``ln``, geometry only) the choice among run R1's candidates falls from MM 0.559 to
  0.325, and among the near-equal candidates its spread is about 4 x the pose term's. In step 2 the bigger cause was
  upstream: the right flip was not among the candidates for about two thirds of the instances (fixed by the flip
  expansion of ``bop3:base``). With the flips present it picks an acceptable candidate for 295 of 486 instances.
* Its errors come from comparing raw colours point to point. At the layer's own chosen pose, half the patches
  differ from their nearest texel by more than the 0.2 value tolerance (median |dV| 0.20) and 53 % by more than
  the saturation tolerance, so V and S give zero evidence there for every flip alike (lighting, exposure and the
  camera's response move V and S of the whole object); 38 % of the patches have no defined hue (S or V < 0.15 on a
  side: white, grey and black print), yet hue carries 2/3 of the weight; one pixel against one texel at about 3 mm
  pose error and fine print do not see the same surface patch; and the max over 10 neighbours lifts the colour
  evidence per patch from 0.34 (nearest node) to 0.51, so a wrong flip also finds matches.
* More of the same does not help: the same raw per-point comparison at every visible node, added to CHECK, gives
  MM 0.532-0.558 (CHECK alone: 0.559).

**What this scorer does instead** (per candidate; all constants are parameters, tuned on the tune side only):

* **onboarding** (``onboard_object``; the textured mesh, once per object, saved in a colour memory npz): for every
  graph node of the memory, the texture averaged over a Gaussian footprint on the surface (``scales_mm``; dense
  area-weighted texture samples, ``OB.texel`` with the toolkit's convention) and the local texture gradient: a
  weighted least-squares plane of the luma over the node's tangent plane, i.e. a 3D vector (luma per metre). A
  vertex-coloured mesh (``OB.colour_source``; LM-O, IC-BIN) gives each of the same dense samples its barycentric
  vertex colour (``OB.vertex_rgb``) instead of a texel; everything after that is the same;
* **visibility and occlusion** (``project``): the nodes are placed by the candidate pose and projected; a node counts
  only when it faces the camera (``cos_min``), lands on the target's mask (its given visible pixels with valid
  depth: other objects' pixels are not on it) and its depth agrees with the observed depth (``depth_tol_m``: this
  removes self-occluded nodes and nodes in front of or behind the seen surface);
* **the image at the same footprint**: the RGB crop blurred inside the mask (normalised convolution, so the
  background does not bleed in) with the same physical scale (``sigma_px = scale x f / median depth``);
* **per-node evidence**, Monty's form ``clip(tol - d, 0) / tol``, averaged over the visible nodes:
  - value and saturation after a per-candidate photometric normalisation (``norm = "z"``: both sides z-scored over
    the visible nodes, so a gain and an offset of the lighting drop out; ``"none"``: step 2's absolute differences);
  - hue (circular), weighted per node by the saturation both sides share (``hue_s_lo``..``hue_s_hi``) and a value
    floor, so unsaturated print votes only through V and S;
  - optionally the edge term: the model's texture gradient projected into the image (the 2 x 2 Jacobian of the
    projection on the node's tangent plane) against the image gradient at that pixel, as a normalised correlation
    of the two vector fields (polarity matters, so an upside-down label disagrees);
* correlation forms of the same features (``ncc_v``, ``ncc_s``: z-scored correlation; ``ncc_hs``: the hsv cone's
  chroma plane (S cos H, S sin H) as a 2D field, which needs no hue gate; ``ncc_vhp``: value minus a coarser scale);
* the score is the weighted mean of the chosen feature evidences (``weights``: ``"<scale>:<feature>"``; correlations
  mapped to [0, 1]), or ``None`` (no opinion) when fewer than ``min_visible`` nodes are visible.

The tuned choice (``TUNED``, ``TUNED_RULE``): saturation, hue (tolerance 0.3) and the chroma plane at a 2 mm footprint
plus the edge term, value weight 0, nodes thinned to 2.5 mm, and the ``weighted`` rule with weight 1.

``scale = 0`` reproduces step 2's inputs (the memory's point-sampled node hsv, the unblurred pixel, ``norm =
"none"``) for the ablation.

Interface: ``appearance.py`` (``score`` / ``score_many``; ``stats_many`` returns the feature evidences per candidate
for tuning). Build it by import path ``rowant_lab.fastlayer.scorers.monty_colour:MontyColour`` (any parameter) or
``...:MontyColourTuned`` (the tuned constants as defaults; for the ``weighted`` rule with weight 1). The
colour memory comes from ``colour_npz`` (default: ``$ROWANT_MONTY_COLOUR_NPZ`` or ``DEFAULT_COLOUR_NPZ``; sha256 pin
``colour_sha256`` or ``$ROWANT_MONTY_COLOUR_SHA256``); with ``"lazy"`` or none found, objects are onboarded from their
mesh and texture paths (the assets), once per process.

CLI (Monty venv, repository root, ``python -P``)::

    ... -m rowant_lab.fastlayer.scorers.monty_colour onboard --memory MEM.npz --out COLOUR.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from rowant_lab.fastlayer import onboard as OB
from rowant_lab.settings import DATA

TRUTH_FREE = True
FORMAT = "rowant fastlayer monty-colour node memory v1"
LUMA = np.array([0.299, 0.587, 0.114])
ONBOARD_SEED = 20261023
ONBOARD = {"density_per_mm2": 12.0, "k": 160, "scales_mm": [1.0, 2.0, 4.0], "seed": ONBOARD_SEED,
           "luma": LUMA.tolist(), "texture_convention": "row = (1 - v) * H (toolkit renderer_py flipud)"}
DEFAULT_COLOUR_NPZ = str(DATA / "runs" / "bop-dev-scratch" / "fastlayer-step3" / "monty-colour" / "colour-memory.npz")
CHUNK = 2048


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def nodes_sha(nodes_m: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(nodes_m, dtype=np.float64)).tobytes()).hexdigest()


def _rgb2hsv(rgb: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2hsv  # noqa: PLC0415

    shp = rgb.shape
    return np.asarray(rgb2hsv(np.clip(rgb, 0.0, 1.0).reshape(-1, 1, 3)), dtype=np.float64).reshape(shp)


# ------------------------------------------------------------------------------------------- onboarding
def _rng(obj: str) -> np.random.Generator:
    word = int.from_bytes(hashlib.sha256(f"monty-colour:{obj}".encode()).digest()[:8], "big")
    return np.random.default_rng([ONBOARD_SEED, word])


def onboard_object(obj: str, ply_path, texture_path, nodes_m: np.ndarray, normals: np.ndarray,
                   params: dict | None = None) -> tuple[dict, dict]:
    """Per node and footprint scale: the mean texture RGB and the luma gradient over the tangent plane (module doc).
    ``nodes_m``: (N, 3) object-local metres (the memory's nodes); ``normals``: (N, 3) their unit normals."""
    p = dict(ONBOARD, **(params or {}))
    t0 = time.perf_counter()
    mesh = OB.read_ply(ply_path)
    V = mesh["vertices_mm"] * 1e-3
    F = mesh["faces"]
    src = OB.colour_source(mesh, ply_path, texture_path)  # the texture (as before) or the vertex colours
    textured = src["kind"] == "texture"
    tex_path, tex = src["path"], src.get("texture")
    t1 = time.perf_counter()
    _, area = OB.face_normals(V, F)
    area_mm2 = float(area.sum() * 1e6)
    n_dense = int(math.ceil(area_mm2 * float(p["density_per_mm2"])))
    fi, bary = OB.sample_surface(V, F, n_dense, _rng(obj))
    pts = OB.interp(V, F, fi, bary)
    if textured:
        puv = OB.interp(mesh["uv"], F, fi, bary)
        rgb = OB.texel(tex, puv, flip_v=True)
    else:  # the same dense samples: barycentric vertex colours
        rgb = OB.vertex_rgb(src["rgb"], F, fi, bary)
    luma = rgb @ LUMA
    t2 = time.perf_counter()
    nodes = np.asarray(nodes_m, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    e1, e2 = OB.monty_tangent_basis(nrm)
    out, diag = {}, {"n_dense": n_dense, "area_mm2": round(area_mm2, 2), "scales": {}}
    k = int(p["k"])
    for s in p["scales_mm"]:
        sm = float(s) * 1e-3
        dens = min(float(p["density_per_mm2"]), k / (math.pi * (2.0 * float(s)) ** 2))
        m = min(n_dense, max(k, int(math.ceil(area_mm2 * dens))))
        tree = cKDTree(pts[:m])
        kk = min(k, m)
        rgb_s = np.zeros((len(nodes), 3))
        grad_s = np.zeros((len(nodes), 3))
        rad = []
        for a in range(0, len(nodes), CHUNK):
            q = nodes[a:a + CHUNK]
            d, idx = tree.query(q, k=kk, workers=1)
            w = np.exp(-(d ** 2) / (2.0 * sm * sm))
            w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
            rgb_s[a:a + CHUNK] = np.einsum("pk,pkc->pc", w, rgb[idx])
            rel = pts[idx] - q[:, None, :]
            ua = np.einsum("pki,pi->pk", rel, e1[a:a + CHUNK])
            vb = np.einsum("pki,pi->pk", rel, e2[a:a + CHUNK])
            X = np.stack([np.ones_like(ua), ua, vb], axis=-1)  # (P, k, 3)
            A = np.einsum("pki,pk,pkj->pij", X, w, X) + np.eye(3)[None] * 1e-12
            b = np.einsum("pki,pk,pk->pi", X, w, luma[idx])
            sol = np.linalg.solve(A, b[..., None])[..., 0]
            grad_s[a:a + CHUNK] = sol[:, 1:2] * e1[a:a + CHUNK] + sol[:, 2:3] * e2[a:a + CHUNK]
            rad.append(np.median(d[:, -1]))
        key = _skey(s)
        out[f"rgb_{key}"] = rgb_s.astype(np.float32)
        out[f"grad_{key}"] = grad_s.astype(np.float32)
        diag["scales"][key] = {"n_cloud": m, "k": kk, "knn_radius_mm_median": round(float(np.median(rad)) * 1e3, 3),
                               "grad_norm_p50": float(np.median(np.linalg.norm(grad_s, axis=1)))}
    t3 = time.perf_counter()
    diag.update({"ply": str(ply_path), "texture": str(tex_path) if textured else None,
                 "ply_sha256": sha256_file(ply_path), "texture_sha256": sha256_file(tex_path) if textured else None,
                 "n_nodes": int(len(nodes)), "nodes_sha256": nodes_sha(nodes),
                 "seconds": {"read": round(t1 - t0, 4), "sample": round(t2 - t1, 4), "nodes": round(t3 - t2, 4),
                             "total": round(t3 - t0, 4)}})
    if not textured:  # textured records keep their keys unchanged
        diag["colour_source"] = "vertex"
        diag["vertex_colour_rule"] = OB.VERTEX_COLOUR_RULE
    return out, diag


def _skey(s) -> str:
    return f"{float(s):g}"


def build_memory(memory_npz, out_npz, *, ids=None, params: dict | None = None, guard=None) -> dict:
    """Onboard every object of a mesh memory (``onboard.build_all``'s npz) into one colour memory npz (exclusive
    create). Returns the meta (also stored in the npz)."""
    from rowant_lab.hybrid_frontier.screens.e0d_feature_transfer import graphs as e0d_graphs  # noqa: PLC0415

    iso = guard() if guard is not None else None
    pos, x, meta = e0d_graphs.load(memory_npz)
    fm = meta["feature_mapping"]
    pv = fm["pose_vectors"][0]
    objs = meta.get("objects", {})
    arrays, per = {}, {}
    t0 = time.perf_counter()
    for g in sorted(pos, key=lambda s: int(s)) if ids is None else [str(i) for i in ids]:
        nodes = np.asarray(pos[g]) - OB.P_TRAIN
        a, d = onboard_object(g, objs[g]["ply"], objs[g].get("texture"), nodes, np.asarray(x[g])[:, pv:pv + 3],
                              params)
        for k_, v in a.items():
            arrays[f"{g}__{k_}"] = v
        per[g] = d
        print(f"monty-colour onboard {g}: {d['n_nodes']} nodes, {d['seconds']['total']:.2f} s", flush=True)
    out = {"format": FORMAT, "truth_free": True, "params": dict(ONBOARD, **(params or {})),
           "memory_npz": str(memory_npz), "memory_sha256": sha256_file(memory_npz), "objects": per,
           "wall_s": round(time.perf_counter() - t0, 3), "isolation": iso,
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    arrays["__meta__"] = np.frombuffer(json.dumps(out, sort_keys=True, default=str).encode(), dtype=np.uint8)
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    with open(out_npz, "xb") as fh:
        np.savez(fh, **arrays)
    out["npz_sha256"] = sha256_file(out_npz)
    return out


class ColourMemory:
    """The colour memory (``build_memory``'s npz): per object and scale, node RGB and luma gradient."""

    def __init__(self, arrays: dict, meta: dict, sha256: str | None = None):
        self.arrays = arrays
        self.meta = meta
        self.sha256 = sha256

    @classmethod
    def load(cls, path, expect_sha256: str | None = None) -> "ColourMemory":
        sha = sha256_file(path)
        if expect_sha256 is not None and sha != expect_sha256:
            raise ValueError(f"{path}: sha256 {sha} != pinned {expect_sha256}")
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: np.asarray(z[k]) for k in z.files if k != "__meta__"}
            meta = json.loads(bytes(z["__meta__"]).decode())
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT!r} file")
        return cls(arrays, meta, sha)

    def has(self, obj: str) -> bool:
        return str(obj) in self.meta["objects"]

    def get(self, obj: str) -> tuple[dict, dict]:
        g = str(obj)
        pref = f"{g}__"
        return {k[len(pref):]: v for k, v in self.arrays.items() if k.startswith(pref)}, self.meta["objects"][g]


# ------------------------------------------------------------------------------------------- per object (scorer side)
@dataclass
class ObjectColour:
    """Everything the scorer needs per object (cached per process)."""

    obj: str
    nodes: np.ndarray  # (n, 3) object-local metres (the kept subset)
    normals: np.ndarray  # (n, 3)
    hsv: dict  # scale key -> (n, 3)
    grad: dict  # scale key -> (n, 3) luma per metre (absent for the point scale)
    keep: np.ndarray  # indices of the kept nodes in the memory's order
    info: dict = field(default_factory=dict)


def subsample(nodes: np.ndarray, grid_mm: float) -> np.ndarray:
    """Indices of a voxel pick of the nodes on a ``grid_mm`` lattice (all nodes when ``grid_mm <= 0``)."""
    if not grid_mm or grid_mm <= 0:
        return np.arange(len(nodes))
    return OB.voxel_pick(np.asarray(nodes, float), float(grid_mm) * 1e-3)


# ------------------------------------------------------------------------------------------- per target (image side)
@dataclass
class Prep:
    r0: int
    c0: int
    H: int
    W: int
    on: np.ndarray  # (H*W,) bool: the target's pixels (mask with valid depth)
    z: np.ndarray  # (H*W,) observed depth, metres
    fx: float
    fy: float
    cx: float
    cy: float
    z_med: float
    hsv: dict  # scale key -> (H*W, 3)
    gx: dict  # scale key -> (H*W,) d luma / d column
    gy: dict  # scale key -> (H*W,) d luma / d row
    sigma_px: dict


def prepare(frame, mask: np.ndarray, scales, *, pad_sigmas: float = 3.0) -> Prep | None:
    """The target's crop at each footprint scale (module doc). ``scales``: mm; 0 = the unblurred pixel."""
    m = np.asarray(mask, dtype=bool)
    rr = np.flatnonzero(m.any(axis=1))
    if len(rr) == 0:
        return None
    cc = np.flatnonzero(m[rr[0]:rr[-1] + 1].any(axis=0))
    cam = frame.camera()
    fx, fy, cx, cy = (float(cam[k]) for k in ("fx", "fy", "cx", "cy"))
    unit = float(cam["depth"]["unit_m"])
    d_all = np.asarray(frame.depth)
    box = (slice(rr[0], rr[-1] + 1), slice(cc[0], cc[-1] + 1))
    z_med = float(np.median(d_all[box][m[box]].astype(np.float64) * unit))
    f = 0.5 * (fx + fy)
    sig = {_skey(s): (float(s) * 1e-3 * f / max(z_med, 1e-3)) for s in scales}
    pad = int(math.ceil(pad_sigmas * max(sig.values()))) + 2
    Hf, Wf = m.shape
    r0, r1 = max(0, int(rr[0]) - pad), min(Hf, int(rr[-1]) + pad + 1)
    c0, c1 = max(0, int(cc[0]) - pad), min(Wf, int(cc[-1]) + pad + 1)
    mc = m[r0:r1, c0:c1]
    H, W = mc.shape
    rgb = np.asarray(frame.rgb)[r0:r1, c0:c1].astype(np.float64) / 255.0
    z = d_all[r0:r1, c0:c1].astype(np.float64) * unit
    mf = mc.astype(np.float64)
    hsv, gx, gy = {}, {}, {}
    for key, s_px in sig.items():
        if float(key) <= 0 or s_px < 0.3:
            blur = rgb
        else:
            den = gaussian_filter(mf, s_px, mode="constant")
            num = gaussian_filter(rgb * mf[..., None], (s_px, s_px, 0), mode="constant")
            blur = np.where(den[..., None] > 1e-6, num / np.maximum(den, 1e-6)[..., None], rgb)
        on = mc.reshape(-1)
        hv = np.zeros((H * W, 3))
        hv[on] = _rgb2hsv(blur.reshape(-1, 3)[on])  # only the target's pixels are ever sampled
        hsv[key] = hv
        lum = blur @ LUMA
        g_r, g_c = np.gradient(lum)
        gx[key] = g_c.reshape(-1)
        gy[key] = g_r.reshape(-1)
    return Prep(r0=r0, c0=c0, H=H, W=W, on=mc.reshape(-1), z=z.reshape(-1), fx=fx, fy=fy, cx=cx, cy=cy, z_med=z_med,
                hsv=hsv, gx=gx, gy=gy, sigma_px=sig)


# ------------------------------------------------------------------------------------------- per candidate
def _rotate(R: np.ndarray, x: np.ndarray) -> np.ndarray:
    """(C, n, 3): every row of ``x`` (n, 3) rotated by each R (C, 3, 3)."""
    return np.matmul(x[None], np.transpose(R, (0, 2, 1)))


def project(R: np.ndarray, t: np.ndarray, nodes: np.ndarray, normals: np.ndarray, prep: Prep, *, cos_min: float,
            depth_tol_m: float) -> dict:
    """Visibility of the nodes under C camera-frame poses (R: (C, 3, 3), t: (C, 3) metres): the flat crop index of
    each node's pixel and the visible flags (facing, on the mask, depth agrees)."""
    X = _rotate(R, nodes) + t[:, None, :]
    Nc = _rotate(R, normals)
    Z = X[..., 2]
    rng_ = np.linalg.norm(X, axis=-1)
    facing = -(Nc * X).sum(-1) > cos_min * rng_
    Zs = np.where(Z > 1e-6, Z, 1.0)
    u = prep.fx * X[..., 0] / Zs + prep.cx
    v = prep.fy * X[..., 1] / Zs + prep.cy
    iu = np.rint(u).astype(np.int64) - prep.c0
    iv = np.rint(v).astype(np.int64) - prep.r0
    inb = (Z > 1e-6) & facing & (iu >= 0) & (iu < prep.W) & (iv >= 0) & (iv < prep.H)
    flat = np.where(inb, iv * prep.W + iu, 0)
    on = inb & prep.on[flat]
    vis = on & (np.abs(Z - prep.z[flat]) < depth_tol_m)
    return {"X": X, "N": Nc, "flat": flat, "vis": vis, "n_facing": facing.sum(1), "n_on": on.sum(1)}


class _Groups:
    """Per-candidate sums over the visible (candidate, node) pairs, kept as flat arrays (``ci``: candidate index)."""

    def __init__(self, ci: np.ndarray, C: int):
        self.ci = ci
        self.C = C
        self.n = np.bincount(ci, minlength=C).astype(np.float64)

    def sum(self, x) -> np.ndarray:
        return np.bincount(self.ci, weights=x, minlength=self.C)

    def mean(self, x) -> np.ndarray:
        return self.sum(x) / np.maximum(self.n, 1e-12)

    def centre(self, x) -> np.ndarray:
        return x - self.mean(x)[self.ci]


def _zpair(g: _Groups, a, b, mode: str, floor: float):
    """(a', b') for the difference |a' - b'|: ``none`` unchanged, ``z`` z-scored per candidate."""
    if mode == "none":
        return a, b
    if mode != "z":
        raise ValueError(f"unknown normalisation {mode!r}")
    a0, b0 = g.centre(a), g.centre(b)
    sa = np.sqrt(g.mean(a0 * a0))[g.ci]
    sb = np.sqrt(g.mean(b0 * b0))[g.ci]
    return a0 / np.maximum(sa, floor), b0 / np.maximum(sb, floor)


def _ncc2(g: _Groups, p, q) -> np.ndarray:
    """Normalised correlation per candidate of two 2D vector fields (rows of ``p``, ``q``)."""
    num = g.sum((p * q).sum(-1))
    den = np.sqrt(g.sum((p * p).sum(-1)) * g.sum((q * q).sum(-1)))
    return np.where(den > 1e-18, num / np.maximum(den, 1e-18), 0.0)


def edge_terms(G3: np.ndarray, X: np.ndarray, Nc: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Predicted image gradient (luma per pixel, (col, row) order) of a surface luma gradient ``G3`` (camera frame,
    per metre) at camera points ``X`` with normals ``Nc``: ``|g| / det(A) (d, -b)``, A = [J g_hat, J e2] the
    projection Jacobian J on the tangent basis (g_hat, e2 = n x g_hat)."""
    gn = np.linalg.norm(G3, axis=-1)
    gh = G3 / np.maximum(gn, 1e-12)[..., None]
    e2 = np.cross(Nc, gh)
    Z = np.where(X[..., 2] > 1e-6, X[..., 2], 1.0)

    def jac(d):
        return (fx * (d[..., 0] / Z - X[..., 0] * d[..., 2] / Z ** 2),
                fy * (d[..., 1] / Z - X[..., 1] * d[..., 2] / Z ** 2))

    a, c = jac(gh)
    b, d = jac(e2)
    det = a * d - b * c
    ok = np.abs(det) > 1e-9
    s = np.where(ok, gn / np.where(ok, det, 1.0), 0.0)
    return np.stack([s * d, -s * b], axis=-1)


def candidate_features(R, t, oc: ObjectColour, prep: Prep, cfg: dict, *, tol_grid: dict | None = None,
                       need: set | None = None) -> dict:
    """Feature evidences of C candidates (camera-frame R (C, 3, 3), t (C, 3) metres) for one object and target, each
    a mean over the candidate's visible nodes. Returns arrays (C,): ``n_vis`` (and ``n_facing``, ``n_on``) and, per
    scale key s: ``s:v``, ``s:s`` (Monty's clipped agreement of value and saturation), ``s:ncc_v``, ``s:ncc_s``
    (their correlation; ``norm = "z"``), ``s:h`` (gated hue), ``s:hue_cov``, ``s:ncc_hs`` (the chroma plane),
    ``s:ncc_vhp`` (with ``hp_scale``) and ``s:g`` (edges); with ``tol_grid`` {feature: [tols]} also ``s:v@tol`` etc.
    ``need``: only these feature names (``v``, ``s``, ``h``, ``hs``, ``vhp``, ``g``; None: all)."""

    def want(f):
        return need is None or f in need

    pr = project(R, t, oc.nodes, oc.normals, prep, cos_min=cfg["cos_min"], depth_tol_m=cfg["depth_tol_m"])
    C = len(R)
    ci, ni = np.nonzero(pr["vis"])
    g = _Groups(ci, C)
    n = g.n
    out = {"n_vis": n, "n_facing": pr["n_facing"].astype(np.float64), "n_on": pr["n_on"].astype(np.float64)}
    fl = pr["flat"][ci, ni]
    floor = cfg["z_floor"]
    for s in cfg["scales"]:
        key = _skey(s)
        hm = oc.hsv[key][ni]  # (k, 3) model, at the visible pairs
        ho = prep.hsv[key][fl]  # (k, 3) image
        # value and saturation
        for ch, name in ((2, "v"), (1, "s")):
            if not want(name):
                continue
            a, b = _zpair(g, hm[:, ch], ho[:, ch], cfg["norm"], floor)
            dd = np.abs(a - b)
            tol = float(cfg[f"tol_{name}"])
            out[f"{key}:{name}"] = g.mean(np.clip(tol - dd, 0.0, None) / tol)
            for tol in (tol_grid or {}).get(name, []):
                out[f"{key}:{name}@{tol:g}"] = g.mean(np.clip(tol - dd, 0.0, None) / tol)
            if cfg["norm"] == "z":
                out[f"{key}:ncc_{name}"] = g.mean(a * b)
        # hue, gated by the saturation both sides share
        if want("h"):
            _hue(out, key, g, hm, ho, n, cfg, tol_grid)
        if want("hs"):
            # hue and saturation jointly: the hsv cone's chroma plane (S cos H, S sin H), normalised correlation of
            # the mean-removed 2D fields (zero chroma votes nothing)
            am, ao = 2 * np.pi * hm[:, 0], 2 * np.pi * ho[:, 0]
            cm = np.stack([g.centre(hm[:, 1] * np.cos(am)), g.centre(hm[:, 1] * np.sin(am))], axis=-1)
            co = np.stack([g.centre(ho[:, 1] * np.cos(ao)), g.centre(ho[:, 1] * np.sin(ao))], axis=-1)
            out[f"{key}:ncc_hs"] = _ncc2(g, cm, co)
        # value with the slow shading removed: V at this scale minus V at the coarser ``hp_scale``
        hp = cfg.get("hp_scale")
        if want("vhp") and hp and _skey(hp) != key and _skey(hp) in oc.hsv:
            vm = hm[:, 2] - oc.hsv[_skey(hp)][ni, 2]
            vo = ho[:, 2] - prep.hsv[_skey(hp)][fl, 2]
            a, b = _zpair(g, vm, vo, "z", floor * 0.25)
            out[f"{key}:ncc_vhp"] = g.mean(a * b)
        # edges: the model's texture gradient projected into the image vs the image gradient
        if want("g") and key in oc.grad and cfg.get("edges", True):
            G3 = np.matmul(oc.grad[key][ni][:, None, :], np.transpose(R, (0, 2, 1))[ci])[:, 0, :]
            gp = edge_terms(G3, pr["X"][ci, ni], pr["N"][ci, ni], prep.fx, prep.fy)
            go = np.stack([prep.gx[key][fl], prep.gy[key][fl]], axis=-1)
            out[f"{key}:g"] = _ncc2(g, gp, go)
        elif want("g"):
            out[f"{key}:g"] = np.zeros(C)
    return out


def _hue(out: dict, key: str, g: _Groups, hm, ho, n, cfg: dict, tol_grid) -> None:
    """Hue agreement (circular, Monty's clipped form), weighted per pair by the saturation both sides share and a
    value floor, shrunk to 0.5 by ``hue_shrink`` pairs' worth of no opinion."""
    dh = np.abs(ho[:, 0] - hm[:, 0])
    dh = np.minimum(dh, 1.0 - dh)
    smin = np.minimum(ho[:, 1], hm[:, 1])
    vmin = np.minimum(ho[:, 2], hm[:, 2])
    lo, hi = cfg["hue_s_lo"], cfg["hue_s_hi"]
    wh = np.clip((smin - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * (vmin > cfg["hue_v_lo"])
    swh = g.sum(wh)
    kap = cfg["hue_shrink"]
    out[f"{key}:hue_cov"] = swh / np.maximum(n, 1e-12)

    def hue_ev(tol):
        return (g.sum(wh * np.clip(tol - dh, 0.0, None) / tol) + 0.5 * kap) / (swh + kap)

    out[f"{key}:h"] = hue_ev(float(cfg["tol_h"]))
    for tol in (tol_grid or {}).get("h", []):
        out[f"{key}:h@{tol:g}"] = hue_ev(float(tol))


# ------------------------------------------------------------------------------------------- the scorer
FEATURE_NEEDS = {"v": "v", "ncc_v": "v", "s": "s", "ncc_s": "s", "h": "h", "hue_cov": "h", "ncc_hs": "hs",
                 "ncc_vhp": "vhp", "g": "g"}


def needed(weights: dict) -> set:
    """The feature blocks of ``candidate_features`` that a weights dict reads."""
    return {FEATURE_NEEDS[k.split("@")[0].split(":", 1)[1]] for k, w in weights.items() if w}


def combine_features(st: dict, weights: dict, min_visible: int) -> np.ndarray:
    """Weighted mean of the named feature evidences per candidate (``"<scale>:<feature>[@tol]"`` keys of
    ``candidate_features``; the correlations ``g`` and ``ncc_*`` in [-1, 1] are mapped to [0, 1]); NaN (no opinion)
    below ``min_visible`` visible nodes."""
    n = np.asarray(st["n_vis"], dtype=np.float64)
    tot = np.zeros(len(n))
    ws = 0.0
    for k, wk in weights.items():
        if not wk:
            continue
        x = np.asarray(st[k], dtype=np.float64)
        base = k.split("@")[0]
        if base.endswith(":g") or ":ncc_" in base:
            x = 0.5 * (1.0 + x)
        tot += float(wk) * x
        ws += float(wk)
    tot = tot / ws if ws > 0 else np.zeros(len(n))
    return np.where(n < min_visible, np.nan, tot)


DEFAULTS = {
    "scales": [2.0],  # footprint sigma(s), mm (0: step 2's point samples)
    "norm": "z",  # photometric normalisation of V and S: "z" or "none"
    "z_floor": 0.02,  # smallest std used in the z-score (a flat patch)
    "tol_v": 1.0, "tol_s": 1.0, "tol_h": 0.1,  # Monty tolerances (z units for "z"; hsv units for "none")
    "weights": {"2:v": 1.0, "2:s": 0.5, "2:h": 1.0, "2:g": 1.0},  # feature -> weight ("<scale>:<feature>")
    "hue_s_lo": 0.15, "hue_s_hi": 0.35, "hue_v_lo": 0.15, "hue_shrink": 10.0,  # hue gate (saturation, value)
    "edges": True,
    "hp_scale": 0.0,  # > 0: also the shading-free value term (V minus V at this coarser scale, mm; must be loaded)
    "cos_min": 0.15,  # a node counts when its normal faces the camera by at least this cosine
    "depth_tol_m": 0.010,  # |node depth - observed depth| for a visible node
    "min_visible": 30,  # fewer visible nodes: no opinion (None)
    "sub_grid_mm": 0.0,  # voxel subsample of the nodes (0: every node)
    "chunk": 24,  # candidates per vectorised block
}


class MontyColour:
    """Scorer A (module doc). ``colour_npz``: the colour memory (default ``$ROWANT_MONTY_COLOUR_NPZ`` or
    ``DEFAULT_COLOUR_NPZ`` when it exists; ``"lazy"`` or none found: onboarding from the assets' mesh and texture,
    once per object and process); its sha256 is
    checked against ``colour_sha256`` (or ``$ROWANT_MONTY_COLOUR_SHA256``) when given."""

    name = "monty_colour"

    def __init__(self, colour_npz: str | None = None, colour_sha256: str | None = None, **params):
        unknown = sorted(set(params) - set(DEFAULTS))
        if unknown:
            raise ValueError(f"unknown monty_colour parameters {unknown}")
        self.cfg = {**DEFAULTS, **params}
        self.cfg["scales"] = [float(s) for s in self.cfg["scales"]]
        for k in self.cfg["weights"]:
            if k.split(":")[0] not in {_skey(s) for s in self.cfg["scales"]}:
                raise ValueError(f"weight {k!r} names a scale that is not computed ({self.cfg['scales']})")
        if self.cfg["norm"] not in ("z", "none"):
            raise ValueError(f"unknown normalisation {self.cfg['norm']!r}")
        if colour_npz == "lazy":  # no colour memory: onboard every object from its mesh, once per process
            path = None
        else:
            path = colour_npz or os.environ.get("ROWANT_MONTY_COLOUR_NPZ") or (
                DEFAULT_COLOUR_NPZ if Path(DEFAULT_COLOUR_NPZ).exists() else None)
        self.colour_npz = path
        self.colour_sha256 = colour_sha256 or os.environ.get("ROWANT_MONTY_COLOUR_SHA256") or None
        self._mem: ColourMemory | None = None
        self._objs: dict = {}
        self.timing = {"prepare_ms": [], "candidates_ms": [], "n_candidates": []}

    def record(self) -> dict:
        return {"name": self.name, "cfg": self.cfg, "colour_npz": self.colour_npz,
                "colour_sha256": self._mem.sha256 if self._mem is not None else self.colour_sha256}

    # -- per object
    def _memory(self) -> ColourMemory | None:
        if self._mem is None and self.colour_npz:
            self._mem = ColourMemory.load(self.colour_npz, self.colour_sha256)
        return self._mem

    def object_colour(self, assets) -> ObjectColour:
        g = str(assets.obj)
        if g in self._objs:
            return self._objs[g]
        nodes = np.asarray(assets.nodes_m, dtype=np.float64)
        normals = np.asarray(assets.features["pose_vectors"])[:, :3].astype(np.float64)
        need = [s for s in self.cfg["scales"] if s > 0]
        arrays, info = {}, {}
        if need:
            mem = self._memory()
            if mem is not None and mem.has(g):
                arrays, info = mem.get(g)
                if int(info["n_nodes"]) != len(nodes) or info["nodes_sha256"] != nodes_sha(nodes):
                    raise ValueError(f"colour memory {self.colour_npz}: object {g} was onboarded on other nodes")
            else:
                if not assets.mesh_path:
                    raise ValueError(f"object {g}: no colour memory and no mesh path to onboard from")
                arrays, info = onboard_object(g, assets.mesh_path, assets.texture_path, nodes, normals,
                                              {"scales_mm": need})
        keep = subsample(nodes, self.cfg["sub_grid_mm"])
        hsv, grad = {}, {}
        for s in self.cfg["scales"]:
            key = _skey(s)
            if s <= 0:
                hsv[key] = np.asarray(assets.features["hsv"], dtype=np.float64)[keep]
                continue
            if f"rgb_{key}" not in arrays:
                raise ValueError(f"object {g}: no colour at scale {key} mm in the colour memory")
            hsv[key] = _rgb2hsv(np.asarray(arrays[f"rgb_{key}"], dtype=np.float64))[keep]
            grad[key] = np.asarray(arrays[f"grad_{key}"], dtype=np.float64)[keep]
        oc = ObjectColour(obj=g, nodes=nodes[keep], normals=normals[keep], hsv=hsv, grad=grad, keep=keep, info=info)
        self._objs[g] = oc
        return oc

    # -- per target
    def stats_many(self, candidates, frame, mask, object_assets, *, tol_grid: dict | None = None,
                   need: set | None = None) -> dict:
        """Feature evidences per candidate (arrays in candidate order; ``candidate_features``)."""
        t0 = time.perf_counter()
        oc = self.object_colour(object_assets)
        prep = prepare(frame, mask, self.cfg["scales"])
        t1 = time.perf_counter()
        C = len(candidates)
        if prep is None or C == 0:
            return {"n_vis": np.zeros(C)}
        Rs, ts = [], []
        for c in candidates:
            R, tm = c.camera_pose_mm(frame)
            Rs.append(R)
            ts.append(tm / 1000.0)
        Rs, ts = np.asarray(Rs, float), np.asarray(ts, float)
        parts = []
        ch = max(1, int(self.cfg["chunk"]))
        for a in range(0, C, ch):
            parts.append(candidate_features(Rs[a:a + ch], ts[a:a + ch], oc, prep, self.cfg, tol_grid=tol_grid,
                                            need=need))
        out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        t2 = time.perf_counter()
        self.timing["prepare_ms"].append((t1 - t0) * 1e3)
        self.timing["candidates_ms"].append((t2 - t1) * 1e3)
        self.timing["n_candidates"].append(C)
        out["_ms"] = np.array([(t1 - t0) * 1e3, (t2 - t1) * 1e3])
        out["_sigma_px"] = np.array([prep.sigma_px[_skey(s)] for s in self.cfg["scales"]])
        return out

    def combine(self, st: dict) -> list:
        """The score of each candidate from its feature evidences (``combine_features``), None below
        ``min_visible``."""
        sc = combine_features(st, self.cfg["weights"], self.cfg["min_visible"])
        return [None if np.isnan(v) else float(v) for v in sc]

    def score_many(self, candidates, frame, mask, object_assets) -> list:
        if not candidates:
            return []
        return self.combine(self.stats_many(candidates, frame, mask, object_assets, need=needed(self.cfg["weights"])))

    def score(self, candidate_pose, frame, mask, object_assets):
        return self.score_many([candidate_pose], frame, mask, object_assets)[0]


# The tune-side choice (rounds r1-r7 of the offline replay of run R1, each scored
# by the official toolkit on the tune side; checked on two folds of the tune scenes). One footprint scale (2 mm);
# saturation (z-scored correlation), hue (tolerance 0.3, saturation-gated), the chroma plane (S cos H, S sin H) and
# the edge term; value dropped (its weight tuned to 0: shading, which the texture lacks, dominates it; the value
# pattern still enters through the luma edges). Decision rule: ``weighted`` with weight 1 (CHECK evidence per patch
# + appearance score). The nodes are thinned to a 2.5 mm voxel pick (about 60 % of them): the same tune-side score
# (MM 0.674 against 0.670 with every node) for about a third less time.
TUNED = {**DEFAULTS, "scales": [2.0], "tol_h": 0.3, "sub_grid_mm": 2.5,
         "weights": {"2:ncc_s": 0.5, "2:h": 1.0, "2:g": 1.0, "2:ncc_hs": 1.0}}
TUNED_ALL_NODES = {**TUNED, "sub_grid_mm": 0.0}  # the first verified choice (run RMC1)
TUNED_RULE = {"name": "weighted", "weight": 1.0}


class MontyColourTuned(MontyColour):
    """``MontyColour`` with the constants chosen on the tune side as defaults (``TUNED``)."""

    name = "monty_colour_tuned"

    def __init__(self, colour_npz: str | None = None, colour_sha256: str | None = None, **params):
        super().__init__(colour_npz, colour_sha256, **{**TUNED, **params})


# ------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="monty-colour scorer: onboarding of the colour memory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("onboard")
    o.add_argument("--memory", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--ids", default=None)
    a = ap.parse_args(argv)
    from rowant_lab.hybrid_frontier.e9 import isolation  # noqa: PLC0415

    ids = [s for s in a.ids.split(",")] if a.ids else None
    meta = build_memory(Path(a.memory).expanduser(), Path(a.out).expanduser(), ids=ids, guard=isolation.require)
    secs = [d["seconds"]["total"] for d in meta["objects"].values()]
    print(json.dumps({"npz_sha256": meta["npz_sha256"], "wall_s": meta["wall_s"],
                      "per_object_s": {"min": min(secs), "median": float(np.median(secs)), "max": max(secs)}},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
