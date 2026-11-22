"""TRUTH-FREE: appearance scorer B of fastlayer step 3, render and compare (item ``bop3:render-compare``), the strong
non-Monty rival of the appearance step.

For every candidate pose the object's textured mesh is rendered on the CPU and compared with the image inside the
target's mask. It plugs into ``appearance.py``'s interface (``score`` / ``score_many``); the layer's decision rule
(``appearance.DecideRule``: ``gate`` or ``weighted``) combines it with CHECK.

**Inputs** (nothing else): the candidate poses, the frame (RGB, raw depth, ``K``, depth scale), the target's mask
(its given pixels with valid depth, as the layer passes it) and the object's assets: the BOP model
(``obj_XXXXXX.ply``, mm, per-vertex texture UVs; read with ``onboard.read_ply``) and its texture PNG, the files
onboarding read. No scene answer, no other image.

**Per object, once per process** (``load_object``; cached): the mesh in metres; the texture reduced to at most
``tex_max`` texels a side (PIL box filter) and a pyramid of ``tex_levels`` halvings, each in CIE Lab
(``skimage.color.rgb2lab``, float16); the texel density (texels per metre at level 0, from the UV and 3D areas).

**Per target, once** (``Observation``): a crop around the mask's bounding box (padded by ``pad_frac`` of its longer
side, at least ``pad_px``), reduced by an integer block factor ``s`` so its longer side is at most ``target_px``
(block means); in it the observed Lab colour, the mask share and the valid-depth mean depth of each block; the
intrinsics of the reduced crop (pixel centres at ``k + 0.5``, as ``render_zbuf``). The texture level is chosen per
target: the coarsest level that still has at least one texel per crop pixel at the candidates' median depth.

**Per candidate** (``render``): a z-buffer rasteriser in numpy after ``rowant_lab/bop/render_zbuf.py`` (same
projection and pixel-centre rule; triangles with a vertex at ``Z <= 1e-6`` dropped; back faces culled by the
camera-frame winding, which onboarding checked against the normals; nearest surface wins), that also keeps each
pixel's face and perspective-correct barycentric weights, hence its depth and its texture UV; the colour is a
bilinear lookup in the Lab texture with the toolkit's convention (texel row = (1 - v) x height). Out: rendered Lab,
depth and silhouette in the reduced crop.

**Features** (``compare``; R = rendered silhouette, M = mask share >= 0.5, O = R and M; all on the reduced crop;
colour vectors are (L, ``chroma`` x a, ``chroma`` x b)):

* ``color``: normalised cross-correlation (means removed per channel, the three channels pooled) of the rendered
  and observed colour over O (lighting gain and offset drop out);
* ``color_hp``: the same after removing local means (a box of ``hp_frac`` x the crop's longer side; the observed
  means over M, the rendered over R): texture structure without slow shading;
* ``grad``: normalised correlation of the colour gradients (a [1 2 1] smoothing, central differences) over O
  eroded by 2 pixels (no silhouette edges): edges;
* ``depth``: share of O (with observed depth) whose rendered depth is within ``depth_tol_m`` of the observed one;
* ``sil``: share of M that R covers, minus the share of R outside M where the observed surface lies more than
  ``depth_tol_m`` behind the rendered one (the object would hide that surface).

The score is ``sum_k weights[k] x feature_k``; ``None`` (no opinion) when O has fewer than ``min_overlap`` pixels.
``prefilter_margin``: candidates whose CHECK evidence per patch is more than this below the best are not rendered
(``None``): with the ``gate`` rule and the same margin they cannot be chosen anyway, so the choice is unchanged and
the time is saved.

Make it by import path (worker processes build their own): ``rowant_lab.fastlayer.scorers.render_compare:
RenderCompare`` (declared defaults) or ``...:RenderCompareTuned`` (the tune-side choice, read from
``render_compare_tuned.json`` next to this file).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

TRUTH_FREE = True
NAME = "render_compare"
FEATURES = ("color", "color_hp", "grad", "depth", "sil")
Z_EPS = 1e-6
CHUNK_ELEMS = 1 << 18
PARAMS = {  # declared before any tune-side number
    "target_px": 96, "pad_frac": 0.10, "pad_px": 3, "tex_max": 512, "tex_levels": 4, "chroma": 2.0,
    "hp_frac": 0.12, "depth_tol_m": 0.006, "min_overlap": 30, "cull": True,
}
DEFAULT_WEIGHTS = {"color": 1.0, "color_hp": 0.0, "grad": 1.0, "depth": 0.0, "sil": 0.0}
TUNED_JSON = Path(__file__).with_name("render_compare_tuned.json")

_OBJECTS: dict = {}


# ------------------------------------------------------------------------------------------- per object
class ObjectModel:
    """The textured mesh of one object (metres, model frame) and its Lab texture pyramid."""

    def __init__(self, V, F, UV, levels, density_per_m, info):
        self.V = V  # (N, 3) float64, metres
        self.F = F  # (M, 3) int64
        self.UV = UV  # (N, 2) float64
        self.levels = levels  # [(H, W, 3) float16 Lab], finest first
        self.density_per_m = density_per_m  # texels per metre at level 0
        self.info = info


def _box2(a: np.ndarray) -> np.ndarray:
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    a = a[:h, :w]
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


def load_object(mesh_path, texture_path=None, *, tex_max: int = PARAMS["tex_max"],
                tex_levels: int = PARAMS["tex_levels"]) -> ObjectModel:
    """Read (once per process) the mesh and texture of one object (module doc)."""
    key = (str(mesh_path), str(texture_path), int(tex_max), int(tex_levels))
    if key in _OBJECTS:
        return _OBJECTS[key]
    from PIL import Image  # noqa: PLC0415
    from skimage.color import rgb2lab  # noqa: PLC0415

    from .. import onboard as OB  # noqa: PLC0415

    t0 = time.perf_counter()
    mesh = OB.read_ply(mesh_path)
    if "uv" not in mesh:
        raise ValueError(f"{mesh_path}: no texture UVs")
    V = np.asarray(mesh["vertices_mm"], np.float64) / 1000.0
    F = np.asarray(mesh["faces"], np.int64)
    UV = np.asarray(mesh["uv"], np.float64)
    tpath = Path(texture_path) if texture_path else Path(mesh_path).with_name(mesh["texture"])
    with Image.open(tpath) as im:
        im = im.convert("RGB")
        W0, H0 = im.size
        f = max(1, int(math.ceil(max(W0, H0) / float(tex_max))))
        if f > 1:
            im = im.resize((max(1, W0 // f), max(1, H0 // f)), Image.Resampling.BOX)
        rgb = np.asarray(im, dtype=np.float32) / 255.0
    levels = []
    a = rgb
    for _ in range(max(1, int(tex_levels))):
        levels.append(rgb2lab(a).astype(np.float16))
        if min(a.shape[:2]) < 8:
            break
        a = _box2(a)
    Hl, Wl = levels[0].shape[:2]
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area3 = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    q0, q1, q2 = UV[F[:, 0]], UV[F[:, 1]], UV[F[:, 2]]
    d1, d2 = q1 - q0, q2 - q0
    areauv = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) * Wl * Hl
    density = float(math.sqrt(areauv.sum() / max(area3.sum(), 1e-12)))
    info = {"mesh": str(mesh_path), "texture": str(tpath), "texture_px": [int(W0), int(H0)],
            "level0_px": [int(Wl), int(Hl)], "n_levels": len(levels), "n_vertices": int(len(V)),
            "n_faces": int(len(F)), "texels_per_mm_level0": round(density / 1000.0, 4),
            "load_s": round(time.perf_counter() - t0, 3)}
    obj = ObjectModel(V, F, UV, levels, density, info)
    _OBJECTS[key] = obj
    return obj


# ------------------------------------------------------------------------------------------- per target
class Observation:
    """The target's reduced crop (module doc)."""

    def __init__(self, frame, mask, p: dict):
        from skimage.color import rgb2lab  # noqa: PLC0415

        mask = np.asarray(mask, bool)
        H, W = mask.shape
        ys, xs = np.nonzero(mask)
        self.ok = len(ys) >= p["min_overlap"]
        if not self.ok:
            return
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        pad = max(int(p["pad_px"]), int(round(p["pad_frac"] * max(y1 - y0, x1 - x0))))
        y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
        y1, x1 = min(H, y1 + pad), min(W, x1 + pad)
        s = max(1, int(math.ceil(max(y1 - y0, x1 - x0) / float(p["target_px"]))))
        hh, ww = (y1 - y0 + s - 1) // s, (x1 - x0 + s - 1) // s
        y0, x0 = max(0, min(y0, H - hh * s)), max(0, min(x0, W - ww * s))
        hh, ww = min(hh, (H - y0) // s), min(ww, (W - x0) // s)
        y1, x1 = y0 + hh * s, x0 + ww * s
        self.s, self.h, self.w, self.y0, self.x0 = s, hh, ww, y0, x0

        def blocks(a):  # a: already cropped
            a = np.asarray(a, np.float64)
            sh = (hh, s, ww, s) + a.shape[2:]
            return a.reshape(sh).mean(axis=(1, 3))

        rgb = blocks(np.asarray(frame.rgb)[y0:y1, x0:x1]) / 255.0
        self.lab = rgb2lab(np.clip(rgb, 0.0, 1.0))
        self.mshare = blocks(mask[y0:y1, x0:x1])
        self.M = self.mshare >= 0.5
        dep = np.asarray(frame.depth)[y0:y1, x0:x1]
        valid = (dep != 0) & (dep != np.iinfo(np.uint16).max)
        zm = dep.astype(np.float64) * (float(frame.depth_scale) / 1000.0)
        vs = blocks(valid)
        zsum = blocks(np.where(valid, zm, 0.0))
        self.zvalid = vs >= 0.5
        self.z = np.where(vs > 0, zsum / np.maximum(vs, 1e-12), 0.0)
        K = np.asarray(frame.K, np.float64)
        self.K = np.array([[K[0, 0] / s, 0.0, (K[0, 2] - x0) / s], [0.0, K[1, 1] / s, (K[1, 2] - y0) / s],
                           [0.0, 0.0, 1.0]])
        self.n_M = int(self.M.sum())
        ch = np.array([1.0, p["chroma"], p["chroma"]])
        self.vec = self.lab * ch
        self.k_hp = max(3, int(round(p["hp_frac"] * max(hh, ww))) | 1)
        self.vec_hp = self.vec - local_mean(self.vec, self.M, self.k_hp)
        self.grad = grad(self.vec)


def box_sum(a: np.ndarray, k: int) -> np.ndarray:
    """Sum over a k x k box (k odd) centred on each pixel, zero outside the array (integral image)."""
    r = k // 2
    a = np.asarray(a, np.float64)
    pad = [(r + 1, r), (r + 1, r)] + [(0, 0)] * (a.ndim - 2)
    c = np.pad(a, pad).cumsum(0).cumsum(1)
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def local_mean(vec: np.ndarray, region: np.ndarray, k: int) -> np.ndarray:
    m = region.astype(np.float64)
    num = box_sum(vec * m[..., None], k)
    den = box_sum(m, k)[..., None]
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)


def grad(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[1 2 1] smoothing along both axes, then central differences (edge-replicated)."""
    a = np.pad(vec, [(1, 1), (1, 1), (0, 0)], mode="edge")
    a = 0.25 * (a[:-2] + 2 * a[1:-1] + a[2:])
    a = 0.25 * (a[:, :-2] + 2 * a[:, 1:-1] + a[:, 2:])
    b = np.pad(a, [(1, 1), (1, 1), (0, 0)], mode="edge")
    gx = 0.5 * (b[1:-1, 2:] - b[1:-1, :-2])
    gy = 0.5 * (b[2:, 1:-1] - b[:-2, 1:-1])
    return gx, gy


def erode(m: np.ndarray, r: int) -> np.ndarray:
    out = np.asarray(m, bool).copy()
    for _ in range(int(r)):
        e = out.copy()
        e[1:] &= out[:-1]
        e[:-1] &= out[1:]
        e[:, 1:] &= out[:, :-1]
        e[:, :-1] &= out[:, 1:]
        e[0], e[-1], e[:, 0], e[:, -1] = False, False, False, False
        out = e
    return out


def _ncc(a: np.ndarray, b: np.ndarray, centre: bool) -> float:
    if centre:
        a = a - a.mean(axis=0)
        b = b - b.mean(axis=0)
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-12 else 0.0


# ------------------------------------------------------------------------------------------- per candidate
def rasterize(Vc: np.ndarray, F: np.ndarray, K: np.ndarray, w: int, h: int, cull: bool = True):
    """Visible surface of a mesh (camera-frame vertices ``Vc``, metres) in a w x h image with intrinsics K: per
    covered pixel its flat index, face, perspective-correct barycentric weights and depth (render_zbuf's rules)."""
    z = Vc[:, 2]
    ok = z > Z_EPS
    zs = np.where(ok, z, 1.0)
    u = K[0, 0] * Vc[:, 0] / zs + K[0, 2]
    v = K[1, 1] * Vc[:, 1] / zs + K[1, 2]
    fidx = np.nonzero(ok[F].all(axis=1))[0]
    Fk = F[fidx]
    if cull and len(Fk):
        P0, P1, P2 = Vc[Fk[:, 0]], Vc[Fk[:, 1]], Vc[Fk[:, 2]]
        front = np.einsum("ij,ij->i", np.cross(P1 - P0, P2 - P0), P0) < 0
        fidx, Fk = fidx[front], Fk[front]
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(0))
    if not len(Fk):
        return empty
    u0, u1, u2 = u[Fk[:, 0]], u[Fk[:, 1]], u[Fk[:, 2]]
    v0, v1, v2 = v[Fk[:, 0]], v[Fk[:, 1]], v[Fk[:, 2]]
    xmin = np.clip(np.ceil(np.minimum(np.minimum(u0, u1), u2) - 0.5), 0, w - 1)
    xmax = np.clip(np.floor(np.maximum(np.maximum(u0, u1), u2) - 0.5), -1, w - 1)
    ymin = np.clip(np.ceil(np.minimum(np.minimum(v0, v1), v2) - 0.5), 0, h - 1)
    ymax = np.clip(np.floor(np.maximum(np.maximum(v0, v1), v2) - 0.5), -1, h - 1)
    area = (u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0)
    keep = (xmax >= xmin) & (ymax >= ymin) & (area != 0) & np.isfinite(area)
    if not keep.any():
        return empty
    sel = np.nonzero(keep)[0]
    bw = (xmax[sel] - xmin[sel] + 1).astype(np.int64)
    bh = (ymax[sel] - ymin[sel] + 1).astype(np.int64)
    key = bh * (w + 1) + bw
    order = np.argsort(key, kind="stable")
    sel, bw, bh, key = sel[order], bw[order], bh[order], key[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(sel)]
    zf = z[Fk]
    izf = 1.0 / np.where(zf > Z_EPS, zf, 1.0)
    out_pix, out_face, out_l0, out_l1 = [], [], [], []
    for s_, e_ in zip(starts, ends):
        gw, gh = int(bw[s_]), int(bh[s_])
        per = gw * gh
        step = max(1, CHUNK_ELEMS // per)
        dx = np.arange(gw, dtype=np.float64)
        dy = np.arange(gh, dtype=np.float64)
        for a_ in range(s_, e_, step):
            idx = sel[a_:min(e_, a_ + step)]
            x0 = xmin[idx][:, None, None]
            y0 = ymin[idx][:, None, None]
            xs = x0 + dx[None, None, :] + 0.5
            ys = y0 + dy[None, :, None] + 0.5
            U0, U1, U2 = (q[idx][:, None, None] for q in (u0, u1, u2))
            V0, V1, V2 = (q[idx][:, None, None] for q in (v0, v1, v2))
            A = area[idx][:, None, None]
            l0 = ((U2 - U1) * (ys - V1) - (V2 - V1) * (xs - U1)) / A
            l1 = ((U0 - U2) * (ys - V2) - (V0 - V2) * (xs - U2)) / A
            inside = (l0 >= 0) & (l1 >= 0) & (1.0 - l0 - l1 >= 0)
            if not inside.any():
                continue
            ii, ry, rx = np.nonzero(inside)
            cols = (xmin[idx][ii] + rx).astype(np.int64)
            rows = (ymin[idx][ii] + ry).astype(np.int64)
            out_pix.append(rows * w + cols)
            out_face.append(idx[ii])
            out_l0.append(l0[ii, ry, rx])
            out_l1.append(l1[ii, ry, rx])
    if not out_pix:
        return empty
    pix = np.concatenate(out_pix)
    face = np.concatenate(out_face)
    l0 = np.concatenate(out_l0)
    l1 = np.concatenate(out_l1)
    lw = np.stack([l0, l1, 1.0 - l0 - l1], axis=1) * izf[face]
    inv = lw.sum(axis=1)
    good = inv > 0
    pix, face, lw, inv = pix[good], face[good], lw[good], inv[good]
    depth = 1.0 / inv
    # nearest surface per pixel: one sort on (pixel, depth) packed in one float key
    dmin, dmax = float(depth.min()), float(depth.max())
    keyf = pix + (depth - dmin) / ((dmax - dmin) * 1.0001 + 1e-12)
    o = np.argsort(keyf, kind="stable")
    ps = pix[o]
    first = np.r_[True, ps[1:] != ps[:-1]]
    win = o[first]
    return pix[win], fidx[face[win]], lw[win] / inv[win, None], depth[win]


def texture_lookup(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    H, W = tex.shape[:2]
    x = uv[:, 0] * W - 0.5
    y = (1.0 - uv[:, 1]) * H - 0.5
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    xi0 = np.clip(x0, 0, W - 1).astype(np.int64)
    xi1 = np.clip(x0 + 1, 0, W - 1).astype(np.int64)
    yi0 = np.clip(y0, 0, H - 1).astype(np.int64)
    yi1 = np.clip(y0 + 1, 0, H - 1).astype(np.int64)
    t = tex
    return ((1 - fx) * (1 - fy) * t[yi0, xi0] + fx * (1 - fy) * t[yi0, xi1] + (1 - fx) * fy * t[yi1, xi0]
            + fx * fy * t[yi1, xi1]).astype(np.float64)


class Render:
    def __init__(self, h, w, pix, lab, depth):
        self.sil = np.zeros(h * w, bool)
        self.sil[pix] = True
        self.sil = self.sil.reshape(h, w)
        self.lab = np.zeros((h * w, 3))
        self.lab[pix] = lab
        self.lab = self.lab.reshape(h, w, 3)
        self.z = np.zeros(h * w)
        self.z[pix] = depth
        self.z = self.z.reshape(h, w)


def render(obj: ObjectModel, R_cam: np.ndarray, t_cam_m: np.ndarray, obs: Observation, level: int,
           cull: bool = True) -> Render:
    Vc = obj.V @ np.asarray(R_cam, np.float64).T + np.asarray(t_cam_m, np.float64).reshape(3)
    pix, face, bary, depth = rasterize(Vc, obj.F, obs.K, obs.w, obs.h, cull=cull)
    uv = np.einsum("ij,ijk->ik", bary, obj.UV[obj.F[face]]) if len(face) else np.zeros((0, 2))
    lab = texture_lookup(obj.levels[level], uv) if len(face) else np.zeros((0, 3))
    return Render(obs.h, obs.w, pix, lab, depth)


def compare(obs: Observation, rd: Render, p: dict) -> dict | None:
    O = rd.sil & obs.M
    n = int(O.sum())
    if n < p["min_overlap"]:
        return None
    ch = np.array([1.0, p["chroma"], p["chroma"]])
    rvec = rd.lab * ch
    f = {"n_overlap": n}
    f["color"] = _ncc(rvec[O], obs.vec[O], centre=True)
    rhp = rvec - local_mean(rvec, rd.sil, obs.k_hp)
    f["color_hp"] = _ncc(rhp[O], obs.vec_hp[O], centre=False)
    O2 = erode(O, 2)
    if int(O2.sum()) >= max(4, p["min_overlap"] // 3):
        rgx, rgy = grad(rvec)
        ogx, ogy = obs.grad
        a = np.concatenate([rgx[O2], rgy[O2]])
        b = np.concatenate([ogx[O2], ogy[O2]])
        f["grad"] = _ncc(a, b, centre=False)
    else:
        f["grad"] = 0.0
    Od = O & obs.zvalid
    tol = p["depth_tol_m"]
    f["depth"] = float((np.abs(rd.z[Od] - obs.z[Od]) <= tol).mean()) if Od.any() else 0.0
    cov = n / max(obs.n_M, 1)
    leak = int((rd.sil & ~obs.M & obs.zvalid & (obs.z > rd.z + tol)).sum()) / max(int(rd.sil.sum()), 1)
    f["sil"] = float(cov - leak)
    return f


# ------------------------------------------------------------------------------------------- the scorer
class RenderCompare:
    """Render-and-compare appearance scorer (module doc)."""

    name = NAME

    def __init__(self, weights: dict | None = None, prefilter_margin: float | None = None, **params):
        unknown = set(params) - set(PARAMS)
        if unknown:
            raise ValueError(f"unknown render_compare parameters {sorted(unknown)}")
        self.params = {**PARAMS, **params}
        w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        bad = set(w) - set(FEATURES)
        if bad:
            raise ValueError(f"unknown features {sorted(bad)}")
        self.weights = {k: float(w.get(k, 0.0)) for k in FEATURES}
        self.prefilter_margin = None if prefilter_margin is None else float(prefilter_margin)
        self.last: dict = {}

    def record(self) -> dict:
        return {"name": self.name, "params": self.params, "weights": self.weights,
                "prefilter_margin": self.prefilter_margin}

    def combine(self, f: dict | None) -> float | None:
        if f is None:
            return None
        return float(sum(self.weights[k] * f[k] for k in FEATURES if self.weights[k]))

    def object_model(self, assets) -> ObjectModel:
        return load_object(assets.mesh_path, assets.texture_path, tex_max=self.params["tex_max"],
                           tex_levels=self.params["tex_levels"])

    def choose_level(self, obj: ObjectModel, obs: Observation, depths_m) -> int:
        """The coarsest texture level with at least one texel per crop pixel at the median candidate depth."""
        z = float(np.median(depths_m)) if len(depths_m) else 1.0
        m_per_px = max(z, 1e-3) / obs.K[0, 0]
        tpp = obj.density_per_m * m_per_px
        lv = int(math.floor(math.log2(tpp))) if tpp > 1.0 else 0
        return int(min(max(lv, 0), len(obj.levels) - 1))

    def features_many(self, candidates, frame, mask, object_assets, *, active=None) -> list:
        """Per candidate its feature dict (``None``: not rendered, or too little overlap); ``active``: which to render
        (default: all). ``self.last`` gets the times."""
        t0 = time.perf_counter()
        obj = self.object_model(object_assets)
        t1 = time.perf_counter()
        obs = Observation(frame, mask, self.params)
        t2 = time.perf_counter()
        n = len(candidates)
        act = list(range(n)) if active is None else [i for i in range(n) if active[i]]
        out: list = [None] * n
        ms = [None] * n
        lv = 0
        if obs.ok and act:
            poses = [c.camera_pose_mm(frame) for c in candidates]
            lv = self.choose_level(obj, obs, [poses[i][1][2] / 1000.0 for i in act])
            for i in act:
                a = time.perf_counter()
                R, t_mm = poses[i]
                rd = render(obj, R, t_mm / 1000.0, obs, lv, cull=self.params["cull"])
                f = compare(obs, rd, self.params)
                ms[i] = (time.perf_counter() - a) * 1e3
                if f is not None:
                    f["ms"] = ms[i]
                out[i] = f
        t3 = time.perf_counter()
        done = [x for x in ms if x is not None]
        self.last = {"object_ms": (t1 - t0) * 1e3, "observation_ms": (t2 - t1) * 1e3,
                     "candidates_ms": (t3 - t2) * 1e3, "n_candidates": n, "n_rendered": len(done),
                     "ms_per_rendered": (float(np.mean(done)) if done else None), "level": lv,
                     "crop": ([obs.h, obs.w, obs.s] if obs.ok else None)}
        return out

    def active(self, candidates) -> list | None:
        if self.prefilter_margin is None:
            return None
        geo = [c.geo for c in candidates]
        if any(g is None for g in geo):
            return None
        top = max(geo)
        lim = top - self.prefilter_margin - 1e-9  # the gate's own test is on summed evidence: keep the borderline
        return [g >= lim for g in geo]

    def score_many(self, candidates, frame, mask, object_assets) -> list:
        feats = self.features_many(candidates, frame, mask, object_assets, active=self.active(candidates))
        return [self.combine(f) for f in feats]

    def score(self, candidate_pose, frame, mask, object_assets) -> float | None:
        return self.score_many([candidate_pose], frame, mask, object_assets)[0]


class RenderCompareTuned(RenderCompare):
    """The tune-side choice (``render_compare_tuned.json``: weights, prefilter margin, parameters)."""

    name = NAME + "_tuned"

    def __init__(self, **override):
        doc = json.loads(TUNED_JSON.read_text(encoding="utf-8"))
        kw = {"weights": doc["weights"], "prefilter_margin": doc.get("prefilter_margin"), **doc.get("params", {})}
        kw.update(override)
        super().__init__(**kw)
        self.tuned = doc
