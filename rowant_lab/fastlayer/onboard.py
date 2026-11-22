"""TRUTH-FREE object onboarding: a BOP object mesh -> a graph memory in Monty's format (the E-0d graph npz).

Input per object: the BOP model (``obj_XXXXXX.ply``: ASCII, millimetres, per-vertex normals and either texture UVs
plus its texture PNG, e.g. HOPE, or per-vertex colours, e.g. LM-O and IC-BIN), as BOP provides them for onboarding.
Nothing here reads an image of a scene, a pose or any answer.

Colour source (``colour_source``; BOP's renderer rule): the texture when the PLY names one (``comment TextureFile``)
or a texture path is given, else the PLY's per-vertex ``red green blue`` (uchar / 255; alpha ignored, as the toolkit
renderer ignores it). The surface colour at a surface point is then the bilinear texel at its interpolated UV
(texture) or the barycentric interpolation of its face's three vertex colours (vertex colours, ``vertex_rgb``): what
an OpenGL renderer shows at that point. Both are point samples at the same surface samples, in the same frames; the
textured path is computed exactly as before vertex colours were supported (bit-identical outputs).

Output: one ``.npz`` in ``screens/e0d_feature_transfer/graphs.FORMAT`` (written with that module's ``write``), so the
CHECK kernel (``e0c_check_timing.kernel``), the proposal (``e0c_check_timing.proposal``) and the refinement models
(``e0f_refine.models.u77_from_arrays``) read it unchanged. Per graph id (the BOP object id as a string):

* ``pos__<id>``: node locations in the model frame, metres, plus ``P_TRAIN`` (the screens' model-frame offset). So a
  pose ``(R, t)`` that places a node m at ``R (m - P_TRAIN) + t`` is exactly BOP's ``R``, ``t`` (t in metres here,
  mm in the results file): the BOP model frame is the object-local frame.
* ``x__<id>``: Monty's ``graph.x`` columns that CHECK reads, in Monty's layout (``FEATURE_MAPPING``):
  ``pose_vectors`` (surface normal, curvature direction 1, curvature direction 2), ``pose_fully_defined``, ``hsv``,
  ``principal_curvatures_log``.

Steps per object (all constants below, fixed before any dev number was seen):

1. Read the mesh (``read_ply``), convert mm -> m.
2. Dense area-weighted surface samples (``DENSE_PER_MM2`` per mm^2, seeded by the object id): position, the
   interpolated vertex normal (unit; outward, checked), the interpolated UV.
3. Nodes: a voxel pick on a ``GRID_M`` lattice (the dense sample nearest each occupied voxel's centre). Target: a
   median nearest-node spacing like U77's (median 1.7 mm, p25-p75 1.2-2.1 mm). U77 graphs leave gaps where the
   finger never went; mesh graphs cover the whole surface.
4. Curvature: Monty's ``principal_curvatures`` maths (``tbp.monty.frameworks.utils.sensor_processing``), vectorised:
   the dense samples within ``CURV_RADIUS_M`` of the node (at most ``CURV_K``), relative to the node; Monty's tangent
   basis (``u = [1, 0, -n0 / n2]`` unless ``n2 == 0``, ``v = n x u``); weighted quadric regression
   ``n = a u^2 + b v^2 + c u v + d u + e v + f`` with Gaussian weights of the 3D distance (``CURV_SIGMA_M``; Monty
   weights by pixel distance in its sensor patch, which a mesh does not have); the first and second fundamental
   forms; eigenvalues of ``g^-1 b`` sorted large to small; directions in the tangent basis; dir2 to the right-hand
   side of dir1. Singular systems (Monty's ``non_singular_mat`` rule) give k = 0, a tangent basis and
   ``pose_fully_defined`` false. ``pose_fully_defined = |k1 - k2| >= 10`` (``pc1_is_pc2_threshold``) and
   ``principal_curvatures_log = log_sign([k1, k2])``. Curvatures are in 1/m, as in Monty.
5. Colour: the texture at the node's UV, bilinear, with the toolkit renderer's convention (``renderer_py`` flips the
   texture up-down: texel row = (1 - v) x height), then ``skimage.color.rgb2hsv``. A truth-free check
   (``seam_check``) records, for the duplicated vertices at UV seams, how well the two texels agree under each v
   direction (sanity only; the convention is not chosen from it). A vertex-coloured mesh: the barycentric vertex
   colour at the node's surface sample (``vertex_rgb``), then ``rgb2hsv``; ``seam_check`` does not apply.

``build_object`` times each object; BOP's onboarding rule is at most 5 minutes per object.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from rowant_lab.hybrid_frontier.screens.e0d_feature_transfer import graphs as e0d_graphs

TRUTH_FREE = True
P_TRAIN = np.array([0.0, 1.5, 0.0])
GRID_M = 0.002
DENSE_PER_MM2 = 2.0
CURV_RADIUS_M = 0.006
CURV_SIGMA_M = 0.003
CURV_K = 160
CURV_MIN_POINTS = 8
PC1_IS_PC2_THRESHOLD = 10.0
SAMPLE_SEED = 20261021
CHUNK = 2048
FEATURE_MAPPING = {"pose_vectors": [0, 9], "pose_fully_defined": [9, 10], "hsv": [10, 13],
                   "principal_curvatures_log": [13, 15]}
MESH_FORMAT = "rowant fastlayer mesh graph memory v1"
PARAMS = {"P_TRAIN": P_TRAIN.tolist(), "GRID_M": GRID_M, "DENSE_PER_MM2": DENSE_PER_MM2,
          "CURV_RADIUS_M": CURV_RADIUS_M, "CURV_SIGMA_M": CURV_SIGMA_M, "CURV_K": CURV_K,
          "CURV_MIN_POINTS": CURV_MIN_POINTS, "PC1_IS_PC2_THRESHOLD": PC1_IS_PC2_THRESHOLD,
          "SAMPLE_SEED": SAMPLE_SEED, "texture_convention": "row = (1 - v) * H (toolkit renderer_py flipud)"}
# recorded per object for a vertex-coloured mesh (PARAMS stays as it was, so textured memories are unchanged)
VERTEX_COLOUR_RULE = ("barycentric interpolation of the face's per-vertex red/green/blue (uchar / 255, alpha ignored) "
                      "at the surface point; the texture wins when the PLY names one (toolkit renderer rule)")
COLOUR_PROPS = ("red", "green", "blue")
COLOUR_SCALE = {"uchar": 255.0, "uint8": 255.0, "float": 1.0, "float32": 1.0, "double": 1.0, "float64": 1.0}


class OnboardError(ValueError):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------- mesh input
def read_ply(path) -> dict:
    """An ASCII PLY with vertex x y z [nx ny nz] [texture_u texture_v] [red green blue [alpha]] and triangle faces
    (BOP's model files). Vertex colours come back as ``vertex_rgb`` ((n, 3) float64 in [0, 1]: uchar / 255) and,
    when present, ``vertex_alpha`` (same scale); a mesh without them has neither key."""
    path = Path(path)
    with open(path, "rb") as fh:
        header = []
        while True:
            line = fh.readline()
            if not line:
                raise OnboardError(f"{path}: no end_header")
            text = line.decode("ascii", errors="strict").strip()
            header.append(text)
            if text == "end_header":
                break
        body = fh.read()
    if header[0] != "ply" or "format ascii 1.0" not in header:
        raise OnboardError(f"{path}: not an ASCII PLY")
    n_v = n_f = None
    vprops: list[str] = []
    vtypes: dict[str, str] = {}
    texture = None
    cur = None
    for h in header:
        parts = h.split()
        if not parts:
            continue
        if parts[0] == "comment" and len(parts) >= 3 and parts[1] == "TextureFile":
            texture = parts[2]
        elif parts[0] == "element":
            cur = parts[1]
            if cur == "vertex":
                n_v = int(parts[2])
            elif cur == "face":
                n_f = int(parts[2])
            else:
                raise OnboardError(f"{path}: unexpected element {cur!r}")
        elif parts[0] == "property":
            if cur == "vertex":
                vprops.append(parts[-1])
                vtypes[parts[-1]] = parts[1]
            elif cur == "face" and parts[1] != "list":
                raise OnboardError(f"{path}: face property {h!r} is not supported")
    if n_v is None or n_f is None or vprops[:3] != ["x", "y", "z"]:
        raise OnboardError(f"{path}: needs vertex x y z and faces")
    tok = np.array(body.split(), dtype=np.float64)
    nv_tok = n_v * len(vprops)
    vert = tok[:nv_tok].reshape(n_v, len(vprops))
    ftok = tok[nv_tok:]
    if len(ftok) != 4 * n_f or not np.all(ftok.reshape(n_f, 4)[:, 0] == 3):
        raise OnboardError(f"{path}: faces must all be triangles")
    faces = ftok.reshape(n_f, 4)[:, 1:].astype(np.int64)
    if faces.min() < 0 or faces.max() >= n_v:
        raise OnboardError(f"{path}: face index out of range")
    col = {p: i for i, p in enumerate(vprops)}
    out = {"vertices_mm": vert[:, [col["x"], col["y"], col["z"]]].copy(), "faces": faces, "texture": texture}
    if all(k in col for k in ("nx", "ny", "nz")):
        out["normals"] = vert[:, [col["nx"], col["ny"], col["nz"]]].copy()
    if "texture_u" in col and "texture_v" in col:
        out["uv"] = vert[:, [col["texture_u"], col["texture_v"]]].copy()
    if all(k in col for k in COLOUR_PROPS):
        kinds = {vtypes[k] for k in COLOUR_PROPS}
        if len(kinds) != 1 or next(iter(kinds)) not in COLOUR_SCALE:
            raise OnboardError(f"{path}: vertex colour types {sorted(kinds)} are not supported")
        scale = COLOUR_SCALE[next(iter(kinds))]
        rgb = vert[:, [col[k] for k in COLOUR_PROPS]] / scale
        if not (np.isfinite(rgb).all() and rgb.min() >= 0.0 and rgb.max() <= 1.0):
            raise OnboardError(f"{path}: vertex colours outside [0, 1] after scaling by 1/{scale:g}")
        out["vertex_rgb"] = rgb
        if "alpha" in col:
            out["vertex_alpha"] = vert[:, col["alpha"]] / COLOUR_SCALE.get(vtypes["alpha"], 1.0)
    return out


# ------------------------------------------------------------------------------------------- colour source
def colour_source(mesh: dict, ply_path, texture_path=None) -> dict:
    """Where the mesh's surface colour comes from (module doc; BOP's renderer rule).

    ``{"kind": "texture", "texture": (H, W, 3) uint8, "path": Path}`` when a texture path is given or the PLY names
    one (it then needs UVs), else ``{"kind": "vertex", "rgb": (n_v, 3) float64 in [0, 1], "path": None}``."""
    if texture_path or mesh.get("texture"):
        if "uv" not in mesh:
            raise OnboardError(f"{ply_path}: names a texture but has no texture UVs")
        tex_path = Path(texture_path) if texture_path else Path(ply_path).with_name(mesh["texture"])
        return {"kind": "texture", "texture": load_texture(tex_path), "path": tex_path}
    if "vertex_rgb" in mesh:
        return {"kind": "vertex", "rgb": mesh["vertex_rgb"], "path": None}
    raise OnboardError(f"{ply_path}: no texture and no vertex colours")


def vertex_rgb(rgb_v: np.ndarray, F: np.ndarray, fi: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """(n, 3) RGB in [0, 1] at surface samples (face index, barycentric weights): the barycentric interpolation of
    the face's three vertex colours (what an OpenGL renderer shows there; perspective-correct weights in a render)."""
    return np.clip(interp(np.asarray(rgb_v, np.float64), F, fi, bary), 0.0, 1.0)


def vertex_colour_record(mesh: dict) -> dict:
    """Truth-free facts about a mesh's vertex colours (diagnostics)."""
    rgb = mesh["vertex_rgb"]
    hsv_s = (rgb.max(1) - rgb.min(1)) / np.maximum(rgb.max(1), 1e-12)
    rec = {"rule": VERTEX_COLOUR_RULE, "n_vertices": int(len(rgb)),
           "rgb_mean": [round(float(v), 6) for v in rgb.mean(0)],
           "rgb_p05": [round(float(v), 6) for v in np.percentile(rgb, 5, axis=0)],
           "rgb_p95": [round(float(v), 6) for v in np.percentile(rgb, 95, axis=0)],
           "share_low_saturation": round(float(np.mean(hsv_s < 0.15)), 6),
           "n_distinct_colours": int(len(np.unique(np.rint(rgb * 255).astype(np.int64) @ [65536, 256, 1])))}
    if "vertex_alpha" in mesh:
        rec["alpha_min"] = float(mesh["vertex_alpha"].min())
        rec["alpha_max"] = float(mesh["vertex_alpha"].max())
    return rec


def load_texture(path) -> np.ndarray:
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as im:
        return np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))


def texel(tex: np.ndarray, uv: np.ndarray, *, flip_v: bool = True) -> np.ndarray:
    """(N, 3) RGB in [0, 1] at the given UVs, bilinear, clamped at the borders. ``flip_v``: the toolkit's convention
    (row = (1 - v) * H); False: row = v * H."""
    H, W = tex.shape[:2]
    u = np.asarray(uv[:, 0], float)
    v = np.asarray(uv[:, 1], float)
    x = u * W - 0.5
    y = ((1.0 - v) if flip_v else v) * H - 0.5
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx, fy = x - x0, y - y0
    out = np.zeros((len(u), 3))
    t = tex.astype(np.float64) / 255.0
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = np.clip(x0 + dx, 0, W - 1).astype(np.int64)
        yi = np.clip(y0 + dy, 0, H - 1).astype(np.int64)
        out += w[:, None] * t[yi, xi]
    return out


def seam_check(mesh: dict, tex: np.ndarray, tol_mm: float = 1e-4) -> dict:
    """Truth-free texture-convention check. A mesh with per-vertex UVs duplicates a vertex at a UV seam: the same 3D
    point appears with two UVs, one per chart. Under the right v convention both texels show the same surface
    point, so their colours agree; under the wrong one they are unrelated. Returns the mean RGB difference (sum over
    channels, colours in [0, 1]) of those pairs under both conventions."""
    V, uv = mesh["vertices_mm"], mesh["uv"]
    pairs = cKDTree(V).query_pairs(tol_mm, output_type="ndarray")
    if len(pairs):
        pairs = pairs[np.abs(uv[pairs[:, 0]] - uv[pairs[:, 1]]).sum(axis=1) > 1e-3]
    if not len(pairs):
        return {"n_seam_pairs": 0, "toolkit_convention_agrees": None}
    diff = {}
    for name, flip in (("toolkit", True), ("other", False)):
        a = texel(tex, uv[pairs[:, 0]], flip_v=flip)
        b = texel(tex, uv[pairs[:, 1]], flip_v=flip)
        diff[name] = float(np.abs(a - b).sum(axis=1).mean())
    return {"n_seam_pairs": int(len(pairs)), "seam_rgb_diff_toolkit": diff["toolkit"],
            "seam_rgb_diff_other": diff["other"], "toolkit_convention_agrees": bool(diff["toolkit"] < diff["other"])}


# ------------------------------------------------------------------------------------------- sampling
def face_normals(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit face normals from the winding and face areas."""
    c = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a = np.linalg.norm(c, axis=1)
    return c / np.maximum(a, 1e-300)[:, None], a / 2.0


def sample_surface(V: np.ndarray, F: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """``n`` area-weighted samples: (face index, barycentric weights (n, 3))."""
    _, area = face_normals(V, F)
    p = area / area.sum()
    fi = rng.choice(len(F), size=int(n), p=p)
    r1 = np.sqrt(rng.random(int(n)))
    r2 = rng.random(int(n))
    bary = np.stack([1.0 - r1, r1 * (1.0 - r2), r1 * r2], axis=1)
    return fi, bary


def interp(attr: np.ndarray, F: np.ndarray, fi: np.ndarray, bary: np.ndarray) -> np.ndarray:
    return np.einsum("nk,nkd->nd", bary, attr[F[fi]])


def voxel_pick(points: np.ndarray, grid: float) -> np.ndarray:
    """Indices of the point nearest the centre of each occupied voxel (sorted)."""
    key = np.floor(points / grid).astype(np.int64)
    d2 = np.sum((points - (key + 0.5) * grid) ** 2, axis=1)
    order = np.lexsort((d2, key[:, 2], key[:, 1], key[:, 0]))
    ks = key[order]
    first = np.ones(len(order), bool)
    first[1:] = np.any(ks[1:] != ks[:-1], axis=1)
    return np.sort(order[first])


# ------------------------------------------------------------------------------------------- Monty's curvature
def monty_tangent_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monty's ``principal_curvatures`` tangent basis for each normal (rows)."""
    n = np.asarray(n, float)
    nz = n[:, 2] != 0
    u = np.zeros_like(n)
    u[nz, 0] = 1.0
    u[nz, 2] = -n[nz, 0] / n[nz, 2]
    u[~nz, 2] = 1.0
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(n, u)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return u, v


def quadric_curvatures(rel: np.ndarray, w: np.ndarray, n: np.ndarray) -> dict:
    """Monty's quadric fit for P neighbourhoods at once.

    ``rel``: (P, m, 3) neighbour positions relative to each centre; ``w``: (P, m) weights (0 = not a neighbour);
    ``n``: (P, 3) unit normals. Returns ``k1``, ``k2`` (k1 >= k2), ``dir1``, ``dir2`` (P, 3) and ``valid`` (P,).
    """
    u_dir, v_dir = monty_tangent_basis(n)
    u = np.einsum("pmi,pi->pm", rel, u_dir)
    v = np.einsum("pmi,pi->pm", rel, v_dir)
    h = np.einsum("pmi,pi->pm", rel, n)
    X = np.stack([u * u, v * v, u * v, u, v, np.ones_like(u)], axis=-1)  # (P, m, 6)
    A = np.einsum("pmi,pm,pmj->pij", X, w, X)
    b = np.einsum("pmi,pm,pm->pi", X, w, h)
    P = len(n)
    k1 = np.zeros(P)
    k2 = np.zeros(P)
    d1 = np.zeros((P, 3))
    d2 = np.zeros((P, 3))
    with np.errstate(all="ignore"):
        cond = np.linalg.cond(A)
    valid = np.isfinite(cond) & (cond < 1.0 / np.finfo(float).eps)
    if valid.any():
        idx = np.flatnonzero(valid)
        prm = np.linalg.solve(A[idx], b[idx][..., None])[..., 0]
        guv = np.zeros((len(idx), 2, 2))
        guv[:, 0, 0] = 1 + prm[:, 3] * prm[:, 3]
        guv[:, 0, 1] = guv[:, 1, 0] = prm[:, 3] * prm[:, 4]
        guv[:, 1, 1] = 1 + prm[:, 4] * prm[:, 4]
        buv = np.zeros((len(idx), 2, 2))
        buv[:, 0, 0] = 2 * prm[:, 0]
        buv[:, 0, 1] = buv[:, 1, 0] = prm[:, 2]
        buv[:, 1, 1] = 2 * prm[:, 1]
        m = np.linalg.inv(guv) @ buv
        ev, evec = np.linalg.eig(m)
        ev, evec = np.real(ev), np.real(evec)
        order = np.argsort(ev, axis=1)[:, ::-1]
        evs = np.take_along_axis(ev, order, axis=1)
        vecs = np.take_along_axis(evec, order[:, None, :], axis=2)
        k1[idx], k2[idx] = evs[:, 0], evs[:, 1]
        p1 = vecs[:, 0, 0][:, None] * u_dir[idx] + vecs[:, 1, 0][:, None] * v_dir[idx]
        p2 = vecs[:, 0, 1][:, None] * u_dir[idx] + vecs[:, 1, 1][:, None] * v_dir[idx]
        rh = np.arctan2(np.einsum("pi,pi->p", np.cross(p1, p2), n[idx]), np.einsum("pi,pi->p", p1, p2))
        p2[rh < 0] *= -1
        d1[idx], d2[idx] = p1, p2
    bad = ~valid
    d1[bad], d2[bad] = u_dir[bad], v_dir[bad]
    return {"k1": k1, "k2": k2, "dir1": d1, "dir2": d2, "valid": valid}


def log_sign(x: np.ndarray) -> np.ndarray:
    """Monty's ``log_sign``: sign(x) * log(|x| + 1)."""
    x = np.asarray(x, float)
    return np.sign(x) * np.log(np.abs(x) + 1)


def node_curvatures(nodes: np.ndarray, normals: np.ndarray, cloud: np.ndarray, tree: cKDTree, *,
                    radius: float = CURV_RADIUS_M, sigma: float = CURV_SIGMA_M, k: int = CURV_K,
                    min_points: int = CURV_MIN_POINTS) -> dict:
    out = {"k1": [], "k2": [], "dir1": [], "dir2": [], "valid": []}
    kk = min(int(k), len(cloud))
    for a in range(0, len(nodes), CHUNK):
        q = nodes[a:a + CHUNK]
        d, idx = tree.query(q, k=kk, workers=1)
        rel = cloud[idx] - q[:, None, :]
        near = d <= radius
        w = np.where(near, np.exp(-(d ** 2) / (2 * sigma ** 2)), 0.0)
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
        r = quadric_curvatures(rel, w, normals[a:a + CHUNK])
        r["valid"] &= near.sum(axis=1) >= min_points
        for key in out:
            out[key].append(r[key])
    return {key: np.concatenate(v) for key, v in out.items()}


# ------------------------------------------------------------------------------------------- one object
def object_rng(obj_id: str) -> np.random.Generator:
    word = int.from_bytes(hashlib.sha256(f"onboard:{obj_id}".encode()).digest()[:8], "big")
    return np.random.default_rng([SAMPLE_SEED, word])


def build_object(obj_id: str, ply_path, texture_path=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(pos, x, diag)`` for one object (module doc, steps 1-5)."""
    from skimage.color import rgb2hsv  # noqa: PLC0415

    t0 = time.perf_counter()
    mesh = read_ply(ply_path)
    V = mesh["vertices_mm"] * 1e-3
    F = mesh["faces"]
    if "normals" not in mesh:
        raise OnboardError(f"{ply_path}: needs per-vertex normals")
    src = colour_source(mesh, ply_path, texture_path)
    textured = src["kind"] == "texture"
    tex_path, tex = src["path"], src.get("texture")
    t1 = time.perf_counter()
    fn, area = face_normals(V, F)
    area_mm2 = float(area.sum() * 1e6)
    n_dense = int(math.ceil(area_mm2 * DENSE_PER_MM2))
    rng = object_rng(obj_id)
    fi, bary = sample_surface(V, F, n_dense, rng)
    pts = interp(V, F, fi, bary)
    nv = interp(mesh["normals"], F, fi, bary)
    ln = np.linalg.norm(nv, axis=1)
    bad_n = ~(ln > 1e-9)
    nv[bad_n] = fn[fi[bad_n]]
    nv /= np.linalg.norm(nv, axis=1, keepdims=True)
    if textured:
        uv = interp(mesh["uv"], F, fi, bary)
    t2 = time.perf_counter()
    pick = voxel_pick(pts, GRID_M)
    nodes, nn = pts[pick], nv[pick]
    if textured:
        nuv = uv[pick]
    t3 = time.perf_counter()
    tree = cKDTree(pts)
    cv = node_curvatures(nodes, nn, pts, tree)
    t4 = time.perf_counter()
    if textured:
        rgb = texel(tex, nuv, flip_v=True)
    else:  # the node's own surface sample (face, barycentric weights): its interpolated vertex colour
        rgb = vertex_rgb(src["rgb"], F, fi[pick], bary[pick])
    hsv = np.asarray(rgb2hsv(rgb[None])[0], dtype=np.float64)
    t5 = time.perf_counter()
    pfd = (np.abs(cv["k1"] - cv["k2"]) >= PC1_IS_PC2_THRESHOLD) & cv["valid"]
    x = np.concatenate([nn, cv["dir1"], cv["dir2"], pfd[:, None].astype(np.float64), hsv,
                        log_sign(np.stack([cv["k1"], cv["k2"]], axis=1))], axis=1)
    pos = nodes + P_TRAIN
    # truth-free diagnostics
    ntree = cKDTree(nodes)
    dnn, _ = ntree.query(nodes, k=2)
    spacing = dnn[:, 1]
    seam = seam_check(mesh, tex) if textured else {"n_seam_pairs": 0, "toolkit_convention_agrees": None,
                                                   "not_applicable": "vertex colours (no texture)"}
    centroid = nodes.mean(axis=0)
    outward = float(np.mean(np.einsum("ij,ij->i", nodes - centroid, nn) > 0))
    winding = float(np.mean(np.einsum("ij,ij->i", nv, fn[fi]) > 0))
    t6 = time.perf_counter()
    diag = {
        "ply": str(ply_path), "ply_sha256": sha256_file(ply_path), "texture": str(tex_path) if textured else None,
        "texture_sha256": sha256_file(tex_path) if textured else None, "n_vertices": int(len(V)),
        "n_faces": int(len(F)),
        "area_mm2": round(area_mm2, 2), "extent_mm": [round(float(e) * 1e3, 2) for e in V.max(0) - V.min(0)],
        "n_dense": n_dense, "n_nodes": int(len(nodes)),
        "spacing_mm": {"median": float(np.median(spacing) * 1e3), "p25": float(np.percentile(spacing, 25) * 1e3),
                       "p75": float(np.percentile(spacing, 75) * 1e3)},
        "outward_share": outward, "vertex_vs_winding_normal_agree": winding,
        "n_vertex_normals_replaced": int(bad_n.sum()),
        "curvature_valid_share": float(cv["valid"].mean()), "pose_fully_defined_share": float(pfd.mean()),
        "texture_v_check": seam,
        "seconds": {"read": round(t1 - t0, 4), "sample": round(t2 - t1, 4), "nodes": round(t3 - t2, 4),
                    "curvature": round(t4 - t3, 4), "colour": round(t5 - t4, 4), "diagnostics": round(t6 - t5, 4),
                    "build_total": round(t5 - t0, 4)},
    }
    if not textured:  # textured diagnostics keep their keys (and so their memories' meta) unchanged
        diag["colour_source"] = "vertex"
        diag["vertex_colour"] = vertex_colour_record(mesh)
    return np.ascontiguousarray(pos), np.ascontiguousarray(x), diag


def build_all(models_dir, obj_ids, out_npz, *, source: dict | None = None) -> dict:
    """Onboard every object into one npz (exclusive create). Returns the meta (also stored in the npz)."""
    models_dir = Path(models_dir)
    arrays, counts, objects = {}, {}, {}
    for oid in obj_ids:
        g = str(int(oid))
        ply = models_dir / f"obj_{int(oid):06d}.ply"
        pos, x, diag = build_object(g, ply)
        arrays[f"pos__{g}"] = pos
        arrays[f"x__{g}"] = x
        counts[g] = int(len(pos))
        objects[g] = diag
    meta = {"format": e0d_graphs.FORMAT, "truth_free": True, "channel": e0d_graphs.CHANNEL,
            "source": dict(source or {}, kind=MESH_FORMAT, models_dir=str(models_dir)),
            "node_counts": counts, "feature_mapping": FEATURE_MAPPING, "kept": list(e0d_graphs.KEEP),
            "params": PARAMS, "objects": objects}
    if list(FEATURE_MAPPING) != list(e0d_graphs.KEEP):
        raise OnboardError("feature layout differs from E-0d's")
    meta["npz_sha256"] = e0d_graphs.write(out_npz, arrays, meta)
    return meta


def meta_json(meta: dict) -> str:
    return json.dumps(meta, indent=1, sort_keys=True, allow_nan=False)
