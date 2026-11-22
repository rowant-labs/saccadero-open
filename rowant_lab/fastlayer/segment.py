"""TRUTH-FREE: a CPU object segmenter for the fast layer (fastlayer step 4, item ``bop4:segment``; method side).

Input: one RGB-D frame (``Frame``), the pose layer's per-image frame context (``FastLayer.frame_context``: the fitted
plane and the observed depth, computed once per image and shared, so its time is the pose layer's), the image's
targets (object id and instance count, as BOP gives them for the localization task) and the onboarded library
(``Library``: per object its graph nodes, its colour memory, and texture templates made from its textured mesh).
Output: at most ``inst_count`` masks per target object, each with a score (``segment``); ``detection`` gives one in
the CNOS / BOP default-detections format, so the pose layer can run on our masks.

Method (constants in ``PARAMS``; set on the HOPE val TUNE side only):

1. **Texture recognition** (``recognise``), most of the masks:
   - onboarding (``build_template_views``, once per object, cached): the textured mesh rendered from 26 directions
     (flat texture, HOPE's focal length, 0.7 m); SIFT keypoints of each view and the model point under each. A
     vertex-coloured mesh (LM-O, IC-BIN) is rendered with its barycentric vertex colours instead
     (``template_views``; ``onboard.colour_source``) and has no texture atlas (``build_template`` gives none);
   - per image: SIFT on the full-resolution grey image (RootSIFT), FLANN neighbours among all objects' template
     descriptors; a ratio test between the two best neighbours of the image's own objects, where the runner-up must
     be another surface point (the same point seen in another view does not count);
   - per object, up to its instance count (plus one retry): 3D-3D RANSAC (Kabsch on triples) between the template
     points and the back-projected keypoints (observed depth), PnP as a fallback; ICP of the model nodes to the
     observed points in the silhouette; the visible mask = the projected front surface where the observed depth
     agrees (or has no reading); a pose is kept when enough of its silhouette agrees with the depth.
2. **Regions for the remaining slots** (only when texture leaves some unfilled):
   - working resolution 1/``factor`` (480 x 270 for HOPE); patches = connected pixels that cross no depth cut and no
     strong colour edge, inside a ``cell`` grid; links between adjacent patches carry a dissimilarity (normal angle,
     convex or concave, depth step, colour difference);
   - background: large flat surfaces and patches on the fitted plane;
   - a merge tree (Kruskal on the dissimilarity, never past the size of the largest target);
   - colour: log-chromaticity histograms of the colour-memory node colours, mapped into the image by an affine chroma
     map fitted on the recognised instances' (texture colour, pixel colour) pairs (``calibration_pairs``);
   - each tree node is scored for each remaining object: texture votes inside it (log-odds), colour log-likelihood
     ratio, and a log-normal prior on its area (the mean silhouette of a convex body is area / 4); a greedy pick per
     slot, without overlap.
3. Invalid-depth pixels next to an instance are filled; masks are upsampled (nearest) to full resolution.

A detection's ``score`` is 0.5-1.0 for a texture instance (by inliers and depth agreement) and 0.25 for a region
instance; texture instances also carry their pose (``R_m2c``, ``t_m2c_mm``: model to camera, mm), a by-product.

What the segmenter reads: the frame's arrays, its K and depth scale, the shared frame context, the targets list and
the onboarded library (graph memory, colour memory, meshes and textures). It never opens a given-masks file or
anything on the scorer side.

The command-line runner (``run``) is not part of this release. A detection's ``time`` is the image's segmenter time
(s), from the decoded frame plus the shared frame context to the masks in memory, in one process (the RLE encoding
is excluded and timed apart).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rowant_lab.settings import DATA

TRUTH_FREE = True
FORMAT = "rowant fastlayer segmenter v1"
DEFAULT_MEMORY = str(DATA / "runs" / "bop-dev-scratch" / "fastlayer-step2" / "memory" / "hope-mesh-memory.npz")
DEFAULT_MEMORY_SHA = "85652f215e87c82fd66309e763eef050a814b6feff8be5446478ae22f99d41ca"
DEFAULT_COLOUR = str(DATA / "runs" / "bop-dev-scratch" / "fastlayer-step3" / "monty-colour" / "colour-memory.npz")
DEFAULT_COLOUR_SHA = "ee99a14d67f14f9f46da985d874f3cd6ff55084b8bb87c4a9add6c82ca38eed6"
DEFAULT_CACHE = str(DATA / "runs" / "bop-dev-scratch" / "fastlayer-step4" / "segment" / "cache")
MAX_WORKERS = 4

PARAMS = {
    "factor": 4,  # working resolution = full / factor
    "cut_abs_m": 0.004,  # depth cut: step > cut_abs_m + cut_rel x z
    "cut_rel": 0.012,
    "colour_edges": True,  # patches also stop at strong colour edges
    "canny_lo": 40,
    "canny_hi": 100,
    "canny_ab": True,
    "cell": 5,  # patch grid cell (working pixels)
    "join_passes": 3,
    "flat_deg": 12.0,  # links with normals closer than this are flat
    "coplanar_m": 0.004,  # flat-surface links also need the centroids within this of each other's tangent plane
    "convex_deg": 60.0,  # link dissimilarity: convex angle / convex_deg + concave angle / concave_deg ...
    "concave_deg": 20.0,
    "no_normal_d": 1.0,  # ... (a link where a patch has no normal) ...
    "step_unit": 0.5,  # ... + mean normalised depth step / step_unit ...
    "colour_unit": 25.0,  # ... + Lab distance of the patch means / colour_unit
    "proposal_mode": "tree",
    "tree_select": "greedy",
    "score_mode": "prob",  # "prob": vote log-odds + colour LLR / tau + log-normal size prior; "sum": LLR + votes
    "vote_logodds": 3.0,
    "vote_cap": 20.0,
    "vote_other_penalty": 0.0,  # log-odds taken off per vote of another object inside the region
    "colour_tau": 15.0,
    "size_sigma": 0.35,
    "vis_share": 0.8,  # expected visible share of the mean silhouette (area / 4)  # "cut": the most stable antichain of the tree, then a Hungarian assignment to the slots
    "cut_size_exp": 1.0,  # cut quality = n_px ** exp x lifetime (death - birth)  # "tree": every region of the size-limited Kruskal merge tree; "levels": components
    "levels": [0.4, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0],  # "levels": components below each dissimilarity level
    "bg_extent_factor": 1.3,  # a flat surface longer than this x the largest target diameter is background
    "plane_bg_m": 0.008,  # patches within this of the fitted plane (and parallel to it) are background
    "below_plane_m": None,  # patches more than this below the plane are background (None: off)
    "chroma_eps": 0.03,
    "chroma_lo": -2.0,
    "chroma_hi": 2.0,
    "chroma_bins": 40,
    "hist_sigma_bins": 1.0,
    "hist_floor": 1e-3,  # of the uniform density, mixed into every model
    "colour_map": "calib",  # per-image library-to-image colour map: "moments", "alpha_shift", "shift" or "none"
    "colour_feat": "uv",  # moments: features "uv", "uvl", "lab" or "ab", each standardised by robust moments
    "nd_bins": 24,  # moments: bins per feature axis ...
    "nd_range": 3.0,  # ... over +- this many robust scales
    "luma_eps": 0.02,
    "joint_alphas": [0.15, 0.25, 0.4, 0.6, 1.0],  # joint: coarse grid (au = av) ...
    "joint_shifts": [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6],  # ... and shifts (tu, tv)
    "joint_refine": [0.25, 0.1],  # joint: local steps (relative for au, av; absolute for tu, tv)
    "joint_refine_rounds": 6,
    "joint_min_px": 30,
    "alphas": [0.15, 0.2, 0.27, 0.35, 0.45, 0.6, 0.8, 1.0],
    "uv_samples": 3000,  # node colours kept per object for the colour-map fit
    "wb_shift_bins": 6,  # white-balance shift search range (bins per axis)
    "bg_uniform": 0.5,  # background model = this x uniform + rest x background-likely pixels' histogram
    "alt_bg_weight": 1.0,  # the competitor of an object: the other targets' models + this x the background model
    "min_px": 15,  # proposals smaller than this (working pixels) are dropped
    "size_factor": 1.6,  # a proposal's extent must be at most this x the object's diameter ...
    "area_factor": 1.3,  # ... and its area at most this x the object's largest silhouette at that depth
    "min_score": -1e9,
    "max_overlap": 0.2,  # a proposal overlapping chosen ones by more than this share is dropped
    "fill_iters": 2,  # invalid-depth pixels next to an instance are filled this many working pixels deep
    "prop_levels": [2, 3, 4, 5],
    "prop_min_px": 40,  # diagnostics: smallest tree region written
    "prop_tree": False,  # diagnostics: write every tree region (large files)
    "diag_label_score": "full",  # diagnostics: labels of the tree regions by "full", "colour", "votes", "colour_mean"  # diagnostics: which levels' regions are written as proposals
    "recognise": True,  # SIFT matches to the texture atlases -> PnP -> projected visible silhouettes
    "template_mode": "views",  # texture templates: "atlas" (the texture image), "views" (rendered views), "both"
    "view_dirs": 26,
    "view_f": 1390.0,
    "view_dist_m": 0.7,
    "view_dists": None,  # several render distances (overrides view_dist_m)
    "view_kp": 500,
    "atlas_px": 512,
    "sift_atlas_features": 1500,
    "lookup_per_mm2": 6.0,
    "sift_scale": 1.0,  # image scale for the SIFT features
    "sift_features": 6000,
    "sift_ratio": 0.85,
    "rootsift": True,
    "flann_checks": 32,
    "match_k": 12,  # > 2: ratio test between the two best neighbours of the image's own objects
    "ratio_skip_m": 0.01,  # > 0: the ratio test's runner-up skips neighbours of the same object within this distance
    "pose_solver": "both",  # "kabsch": 3D-3D RANSAC on the depth at the keypoints; "pnp"; "both": kabsch, then pnp
    "ransac3d_iters": 5000,
    "ransac3d_thr_m": 0.012,
    "ransac3d_min_inliers": 4,
    "ransac3d_min_side_m": 0.01,
    "ransac_seed": 20261021,
    "extra_tries": 1,  # further RANSAC rounds per object after a rejected pose
    "kp_depth_win": 0,  # > 0: keypoint depth from a (2r+1)^2 window (3 x 3 samples at stride r) ...
    "kp_depth_stat": "median",  # ... its median or its "min" (the nearest surface)
    "pnp_reproj_px": 8.0,  # full-resolution pixels
    "pnp_iters": 200,
    "pnp_min_inliers": 6,
    "icp_iters": 6,  # ICP refinement of each recognised pose on the observed depth
    "icp_nodes": 800,
    "icp_max_m": 0.02,
    "vis_tol_m": 0.02,  # a silhouette pixel is visible when the observed depth is within this of the model's
    "min_depth_agree": 0.3,  # a pose is kept when this share of its silhouette agrees with the observed depth
    "recog_only": False,  # diagnostic: only the recognised instances
    "vote_ratio": 0.8,  # looser ratio test for the texture votes inside region proposals
    "vote_weight": 20.0,  # nats per vote added to a region's score for that object
    "calib_min_pairs": 200,
    "calib_huber": 0.15,
    "debug": False,
    "diag_mode": None,  # "pixel_argmax": diagnostic masks = every foreground pixel to its best colour model
    "px_smooth": 0,  # pixel_argmax: box-filter radius (working px) of the log-likelihood ratios
}


# ------------------------------------------------------------------------------------------- library
@dataclass
class ObjectModel:
    obj_id: int
    diameter_m: float
    extents_m: np.ndarray  # (3,) principal extents of the graph nodes, descending
    chroma: np.ndarray  # (B, B) log-chromaticity density (sums to 1), unsmoothed
    uv: np.ndarray | None = None  # (m, 2) log-chromaticity of (a subsample of) the node colours
    rgb: np.ndarray | None = None  # (m, 3) the same subsample's RGB in [0, 1]
    nodes: np.ndarray | None = None  # (n, 3) graph nodes, model frame, metres (silhouettes)
    normals: np.ndarray | None = None  # (n, 3) their unit normals
    sift_des: np.ndarray | None = None  # (k, 128) SIFT descriptors of the texture atlas ...
    sift_xyz: np.ndarray | None = None  # ... and the model points (metres) under them
    node_rgb: np.ndarray | None = None  # (n, 3) colour-memory RGB of every node (2 mm footprint)
    area_m2: float = 0.0  # mesh surface area (mean silhouette of a convex body = area / 4)


@dataclass
class Library:
    objects: dict  # obj_id -> ObjectModel
    info: dict = field(default_factory=dict)

    @classmethod
    def build(cls, memory_npz: str = DEFAULT_MEMORY, colour_npz: str = DEFAULT_COLOUR, *,
              memory_sha256: str | None = DEFAULT_MEMORY_SHA, colour_sha256: str | None = DEFAULT_COLOUR_SHA,
              params: dict | None = None, ids=None, cache_dir: str | None = None) -> "Library":
        from rowant_lab.hybrid_frontier.screens.e0d_feature_transfer import graphs as e0d_graphs  # noqa: PLC0415

        from .scorers.monty_colour import ColourMemory  # noqa: PLC0415

        p = dict(PARAMS, **(params or {}))
        PARAMS_ROOTSIFT[0] = bool(p["rootsift"])
        PARAMS_ROOTSIFT[1] = int(p["flann_checks"])
        t0 = time.perf_counter()
        msha = _sha256(memory_npz)
        if memory_sha256 is not None and msha != memory_sha256:
            raise ValueError(f"{memory_npz}: sha256 {msha} != pinned {memory_sha256}")
        from . import onboard as OB  # noqa: PLC0415

        pos, xfeat, meta = e0d_graphs.load(memory_npz)
        pv = meta["feature_mapping"]["pose_vectors"][0]
        cm = ColourMemory.load(colour_npz, expect_sha256=colour_sha256)
        objs = {}
        keep = sorted(pos, key=int) if ids is None else [str(i) for i in ids]
        tmpl = load_templates(meta.get("objects", {}), keep, p, cache_dir) if p["recognise"] else {}
        for g in keep:
            nodes = np.asarray(pos[g], dtype=np.float64)
            rgb = cm.get(g)[0]["rgb_2"].astype(np.float64)
            sub = rgb
            if len(sub) > int(p["uv_samples"]):
                sub = sub[np.linspace(0, len(sub) - 1, int(p["uv_samples"])).astype(int)]
            u, v = chroma_coords(sub, p)
            uv = np.stack([u, v], 1).astype(np.float32)
            nrm = np.asarray(xfeat[g])[:, pv:pv + 3].astype(np.float64)
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            des, xyz = tmpl.get(g, (None, None))
            objs[int(g)] = ObjectModel(int(g), _diameter(nodes), _extents(nodes), chroma_hist(rgb, p), uv,
                                       sub.astype(np.float32), nodes - OB.P_TRAIN, nrm, des, xyz,
                                       rgb.astype(np.float32), float(cm.get(g)[1]["area_mm2"]) * 1e-6)
        info = {"memory_npz": str(memory_npz), "memory_sha256": msha, "colour_npz": str(colour_npz),
                "colour_sha256": cm.sha256, "n_objects": len(objs), "build_s": round(time.perf_counter() - t0, 3)}
        return cls(objs, info)


    def matcher(self):
        """(FLANN index over every object's atlas descriptors, object id per descriptor, model point per
        descriptor), built once per process."""
        if getattr(self, "_matcher", None) is None:
            import cv2  # noqa: PLC0415

            ids = [o for o in sorted(self.objects) if self.objects[o].sift_des is not None
                   and len(self.objects[o].sift_des)]
            des = np.concatenate([self.objects[o].sift_des for o in ids]).astype(np.float32)
            if PARAMS_ROOTSIFT[0]:
                des = rootsift(des)
            oid = np.concatenate([np.full(len(self.objects[o].sift_des), o) for o in ids])
            xyz = np.concatenate([self.objects[o].sift_xyz for o in ids]).astype(np.float64)
            fl = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=int(PARAMS_ROOTSIFT[1])))
            fl.add([des])
            fl.train()
            self._matcher = (fl, oid, xyz)
        return self._matcher


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _extents(nodes: np.ndarray) -> np.ndarray:
    c = nodes - nodes.mean(axis=0)
    _, vec = np.linalg.eigh(c.T @ c)
    proj = c @ vec
    return np.sort(proj.max(axis=0) - proj.min(axis=0))[::-1]


def _diameter(nodes: np.ndarray) -> float:
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    hull = nodes[ConvexHull(nodes).vertices]
    d2 = ((hull[:, None, :] - hull[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.max()))


PARAMS_ROOTSIFT = [True, 32]  # set by Library.build: RootSIFT descriptors, FLANN checks


def rootsift(des: np.ndarray) -> np.ndarray:
    d = np.asarray(des, dtype=np.float32)
    d = d / np.maximum(np.abs(d).sum(1, keepdims=True), 1e-12)
    return np.sqrt(d)


# ------------------------------------------------------------------------------------------- texture templates
TEMPLATE_KEYS = ("atlas_px", "sift_atlas_features", "lookup_per_mm2", "template_mode", "view_dirs", "view_f",
                 "view_dist_m", "view_kp", "view_dists")


def build_template(obj: str, ply_path, texture_path, p: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """SIFT keypoints of the object's texture atlas (grey, ``atlas_px`` square) and the model point under each
    keypoint (an atlas-pixel -> surface-point lookup made by splatting dense surface samples by their UV)."""
    import cv2  # noqa: PLC0415

    from . import onboard as OB  # noqa: PLC0415

    t0 = time.perf_counter()
    mesh = OB.read_ply(ply_path)
    if not texture_path and not mesh.get("texture") and "vertex_rgb" in mesh:
        # a vertex-coloured mesh has no texture atlas: no atlas keypoints (its rendered views still give templates)
        return (np.zeros((0, 128), np.float32), np.zeros((0, 3), np.float32),
                {"n_kp": 0, "skipped": "vertex colours: no texture atlas"})
    if "uv" not in mesh:
        raise ValueError(f"{ply_path}: no texture UVs")
    V = mesh["vertices_mm"] * 1e-3
    Fc = mesh["faces"]
    tex_path = Path(texture_path) if texture_path else Path(ply_path).with_name(mesh["texture"])
    tex = OB.load_texture(tex_path)
    S = int(p["atlas_px"])
    grey = cv2.resize(cv2.cvtColor(np.ascontiguousarray(tex[..., :3]), cv2.COLOR_RGB2GRAY), (S, S),
                      interpolation=cv2.INTER_AREA)
    _, area = OB.face_normals(V, Fc)
    n_s = int(min(3e6, max(1e5, area.sum() * 1e6 * float(p["lookup_per_mm2"]))))
    word = int.from_bytes(hashlib.sha256(f"segment-template:{obj}".encode()).digest()[:8], "big")
    fi, bary = OB.sample_surface(V, Fc, n_s, np.random.default_rng([20261021, word]))
    pts = OB.interp(V, Fc, fi, bary)
    puv = OB.interp(mesh["uv"], Fc, fi, bary)
    x = np.clip((puv[:, 0] * S).astype(np.int64), 0, S - 1)
    y = np.clip(((1.0 - puv[:, 1]) * S).astype(np.int64), 0, S - 1)
    flat = y * S + x
    cnt = np.bincount(flat, minlength=S * S)
    look = np.stack([np.bincount(flat, weights=pts[:, k], minlength=S * S) for k in range(3)], 1)
    look /= np.maximum(cnt, 1)[:, None]
    ok = cnt > 0
    sift = cv2.SIFT_create(nfeatures=int(p["sift_atlas_features"]))
    kps, des = sift.detectAndCompute(grey, (ok.reshape(S, S) * 255).astype(np.uint8))
    if des is None:
        return np.zeros((0, 128), np.float32), np.zeros((0, 3), np.float32), {"n_kp": 0}
    kx = np.array([k.pt for k in kps])
    ki = np.clip(np.rint(kx[:, 1]).astype(int), 0, S - 1) * S + np.clip(np.rint(kx[:, 0]).astype(int), 0, S - 1)
    good = ok[ki]
    return (des[good].astype(np.float32), look[ki[good]].astype(np.float32),
            {"n_kp": int(good.sum()), "n_samples": n_s, "atlas_cover": round(float(ok.mean()), 4),
             "seconds": round(time.perf_counter() - t0, 3)})


def view_directions(kind: int) -> np.ndarray:
    """Unit view directions (object -> camera): 14 = cube faces + corners, 26 = + edges."""
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    dirs += [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    if kind >= 26:
        dirs += [(x, y, 0) for x in (-1, 1) for y in (-1, 1)] + [(x, 0, z) for x in (-1, 1) for z in (-1, 1)] \
            + [(0, y, z) for y in (-1, 1) for z in (-1, 1)]
    d = np.array(dirs, dtype=np.float64)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def template_views(ply_path, texture_path, p: dict):
    """The rendered views behind ``build_template_views``, one dict per view that covers at least 100 pixels:
    ``dist`` (m), ``dir`` (object -> camera), ``size`` (px, square), ``K``, ``R`` and ``t_m`` (model -> camera:
    ``x_cam = R x + t_m``, metres), ``pix`` / ``face`` / ``bary`` / ``depth`` (``render_compare.rasterize``),
    ``rgb`` (the covered pixels' flat colour: the texel at the interpolated UV, or the barycentric vertex colour of
    a vertex-coloured mesh), ``grey`` (uint8 image), ``xyz`` (model point per pixel, NaN off the object) and ``msk``
    (the eroded silhouette). Flat colour, no shading; HOPE-like focal length and distance (``view_f``,
    ``view_dist_m``)."""
    import cv2  # noqa: PLC0415

    from . import onboard as OB  # noqa: PLC0415
    from .scorers.render_compare import rasterize  # noqa: PLC0415

    mesh = OB.read_ply(ply_path)
    V = mesh["vertices_mm"] * 1e-3
    Fc = mesh["faces"]
    src = OB.colour_source(mesh, ply_path, texture_path)
    textured = src["kind"] == "texture"
    tex = src.get("texture")
    c = V.mean(axis=0)
    diam = float(np.linalg.norm(V.max(0) - V.min(0)))
    f = float(p["view_f"])
    dists = p["view_dists"] if p.get("view_dists") else [p["view_dist_m"]]
    for dist, d in ((float(dd), d_) for dd in dists for d_ in view_directions(int(p["view_dirs"]))):
        size = int(np.ceil(1.2 * f * diam / dist)) | 1
        K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])
        zc = -d
        helper = np.array([0.0, 0.0, 1.0]) if abs(zc[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        xc = np.cross(helper, zc)
        xc /= np.linalg.norm(xc)
        yc = np.cross(zc, xc)
        R = np.stack([xc, yc, zc])
        Vc = (V - c) @ R.T + np.array([0.0, 0.0, dist])
        pix, face, bary, depth = rasterize(Vc, Fc, K, size, size)
        if len(pix) < 100:
            continue
        if textured:
            uv = (mesh["uv"][Fc[face]] * bary[:, :, None]).sum(1)
            rgb = OB.texel(tex, uv, flip_v=True)
        else:  # perspective-correct barycentric weights of the visible face: its interpolated vertex colours
            rgb = np.clip((src["rgb"][Fc[face]] * bary[:, :, None]).sum(1), 0.0, 1.0)
        grey = np.zeros(size * size, np.uint8)
        grey[pix] = np.clip(rgb @ np.array([0.299, 0.587, 0.114]) * 255, 0, 255).astype(np.uint8)
        xyz = np.full((size * size, 3), np.nan)
        xyz[pix] = (V[Fc[face]] * bary[:, :, None]).sum(1)
        msk = np.zeros(size * size, np.uint8)
        msk[pix] = 255
        msk = cv2.erode(msk.reshape(size, size), np.ones((3, 3), np.uint8))
        yield {"dist": dist, "dir": d, "size": size, "K": K, "R": R, "t_m": np.array([0.0, 0.0, dist]) - R @ c,
               "pix": pix, "face": face, "bary": bary, "depth": depth, "rgb": rgb, "grey": grey.reshape(size, size),
               "xyz": xyz, "msk": msk}


def build_template_views(obj: str, ply_path, texture_path, p: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """SIFT keypoints of rendered views of the mesh (``template_views``: flat texture or vertex colours, HOPE-like
    focal length and distance) and the model point under each keypoint."""
    import cv2  # noqa: PLC0415

    t0 = time.perf_counter()
    sift = cv2.SIFT_create(nfeatures=int(p["view_kp"]))
    des_all, xyz_all = [], []
    size = None
    for view in template_views(ply_path, texture_path, p):
        size, grey, xyz, msk = view["size"], view["grey"], view["xyz"], view["msk"]
        kps, des = sift.detectAndCompute(grey, msk)
        if des is None:
            continue
        kx = np.array([k.pt for k in kps])
        ki = np.clip(np.rint(kx[:, 1]).astype(int), 0, size - 1) * size + np.clip(np.rint(kx[:, 0]).astype(int), 0,
                                                                                    size - 1)
        good = np.isfinite(xyz[ki, 0])
        des_all.append(des[good])
        xyz_all.append(xyz[ki[good]])
    if not des_all:
        return np.zeros((0, 128), np.float32), np.zeros((0, 3), np.float32), {"n_kp": 0}
    des = np.concatenate(des_all).astype(np.float32)
    xyz = np.concatenate(xyz_all).astype(np.float32)
    return des, xyz, {"n_kp": int(len(des)), "size_px": size, "seconds": round(time.perf_counter() - t0, 3)}


def template_cache_path(cache_dir, p: dict) -> Path:
    """The template cache file for these parameters (keyed by the template parameters only: one cache folder per
    dataset, since object ids repeat across datasets)."""
    key = hashlib.sha256(json.dumps({k: p[k] for k in TEMPLATE_KEYS}, sort_keys=True).encode()).hexdigest()[:12]
    return Path(cache_dir) / f"templates-{key}.npz"


def load_templates(objects_meta: dict, keep: list, p: dict, cache_dir: str | None,
                   timings: dict | None = None) -> dict:
    """{obj: (descriptors, model points)}, from the cache npz when it exists for these parameters.

    A cache written since vertex colours were supported stores each object's ply sha256 (``__ply_sha256__``) and
    is refused when a kept object's mesh differs. A cache without that record (step 4's HOPE cache) is refused for
    a vertex-coloured object (it was made from textured HOPE meshes only). ``timings``: when given, filled with
    the seconds per built object (onboarding records)."""
    path = template_cache_path(cache_dir, p) if cache_dir else None
    if path is not None and path.exists():
        with np.load(path, allow_pickle=False) as z:
            if "__ply_sha256__" in z.files:
                have = json.loads(bytes(z["__ply_sha256__"]).decode())
                bad = [g for g in keep if g in have and objects_meta.get(g, {}).get("ply_sha256") not in (
                    None, have[g])]
                if bad:
                    raise ValueError(f"{path}: templates of objects {bad} were made from other meshes")
            else:
                vc = [g for g in keep if objects_meta.get(g, {}).get("colour_source") == "vertex"]
                if vc:
                    raise ValueError(f"{path}: a cache without mesh hashes (textured meshes only) cannot serve the "
                                     f"vertex-coloured objects {vc}; use this dataset's own cache folder")
            return {g: (np.asarray(z[f"{g}__des"]), np.asarray(z[f"{g}__xyz"])) for g in keep if f"{g}__des" in z}
    out, arrays, shas = {}, {}, {}
    for g in keep:
        meta = objects_meta[g]
        t0 = time.perf_counter()
        parts = []
        if p["template_mode"] in ("atlas", "both"):
            parts.append(build_template(g, meta["ply"], meta.get("texture"), p)[:2])
        if p["template_mode"] in ("views", "both"):
            parts.append(build_template_views(g, meta["ply"], meta.get("texture"), p)[:2])
        des = np.concatenate([q[0] for q in parts]).astype(np.float32)
        xyz = np.concatenate([q[1] for q in parts]).astype(np.float32)
        out[g] = (des, xyz)
        arrays[f"{g}__des"] = des
        arrays[f"{g}__xyz"] = xyz
        if meta.get("ply_sha256"):
            shas[g] = meta["ply_sha256"]
        if timings is not None:
            timings[g] = round(time.perf_counter() - t0, 4)
    if shas:
        arrays["__ply_sha256__"] = np.frombuffer(json.dumps(shas, sort_keys=True).encode(), dtype=np.uint8)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    return out


# ------------------------------------------------------------------------------------------- colour
def chroma_coords(rgb: np.ndarray, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Log-chromaticity (ln(R/G), ln(B/G)) of float RGB in [0, 1] (any leading shape)."""
    e = float(p["chroma_eps"])
    r, g, b = rgb[..., 0] + e, rgb[..., 1] + e, rgb[..., 2] + e
    return np.log(r / g), np.log(b / g)


def chroma_bin(rgb: np.ndarray, p: dict) -> np.ndarray:
    """Flat bin index (u * B + v) of each colour; out-of-range values go to the edge bins."""
    u, v = chroma_coords(rgb, p)
    B = int(p["chroma_bins"])
    lo, hi = float(p["chroma_lo"]), float(p["chroma_hi"])
    s = B / (hi - lo)
    iu = np.clip(((u - lo) * s).astype(np.int32), 0, B - 1)
    iv = np.clip(((v - lo) * s).astype(np.int32), 0, B - 1)
    return iu * B + iv


def chroma_hist(rgb: np.ndarray, p: dict) -> np.ndarray:
    B = int(p["chroma_bins"])
    h = np.bincount(chroma_bin(rgb, p).ravel(), minlength=B * B).astype(np.float64).reshape(B, B)
    return h / max(h.sum(), 1.0)


def smooth_model(h: np.ndarray, p: dict) -> np.ndarray:
    """A density with Gaussian smoothing (in bins) and a uniform floor; sums to 1."""
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    s = float(p["hist_sigma_bins"])
    out = gaussian_filter(h, s, mode="constant") if s > 0 else h.copy()
    out = out / max(out.sum(), 1e-12)
    f = float(p["hist_floor"])
    return (1.0 - f) * out + f / out.size


def shift_hist(h: np.ndarray, du: int, dv: int) -> np.ndarray:
    """``h`` moved by (du, dv) bins (zero fill)."""
    out = np.zeros_like(h)
    B = h.shape[0]
    su = slice(max(du, 0), B + min(du, 0))
    sv = slice(max(dv, 0), B + min(dv, 0))
    tu = slice(max(-du, 0), B + min(-du, 0))
    tv = slice(max(-dv, 0), B + min(-dv, 0))
    out[su, sv] = h[tu, tv]
    return out


def fit_white_balance(h_img: np.ndarray, models: list, weights: list, p: dict) -> tuple[int, int, float]:
    """The (du, dv) shift (bins) of the models' mixture that maximises the image histogram's log-likelihood."""
    mix = sum(w * m for w, m in zip(weights, models)) / max(sum(weights), 1e-12)
    logm = np.log(smooth_model(mix, p))
    R = int(p["wb_shift_bins"])
    B = logm.shape[0]
    floor = np.log(float(p["hist_floor"]) / logm.size)
    pad = np.full((B + 2 * R, B + 2 * R), floor)
    pad[R:R + B, R:R + B] = logm
    win = np.lib.stride_tricks.sliding_window_view(pad, (B, B))  # win[i, j] = logm(b + (i - R, j - R))
    ll = np.einsum("ijab,ab->ij", win, h_img)
    i, j = np.unravel_index(int(np.argmax(ll)), ll.shape)
    return R - int(i), R - int(j), float(ll[i, j])


def uv_hist(uv: np.ndarray, weights: np.ndarray | None, p: dict) -> np.ndarray:
    B = int(p["chroma_bins"])
    lo, hi = float(p["chroma_lo"]), float(p["chroma_hi"])
    s = B / (hi - lo)
    iu = np.clip(((uv[:, 0] - lo) * s).astype(np.int32), 0, B - 1)
    iv = np.clip(((uv[:, 1] - lo) * s).astype(np.int32), 0, B - 1)
    h = np.bincount(iu * B + iv, weights=weights, minlength=B * B).astype(np.float64).reshape(B, B)
    return h / max(h.sum(), 1e-12)


def fit_colour_map(h_img: np.ndarray, models: list, weights: list, p: dict) -> tuple[float, int, int, float]:
    """Per-image map of the library's log-chromaticity to the image's: ``uv_img = alpha x uv_lib + shift`` (a
    desaturation towards grey, then a white-balance shift). Grid over alpha, exhaustive over integer bin shifts;
    the choice maximises the image histogram's log-likelihood under the targets' mixture."""
    uv = np.concatenate([m.uv for m in models])
    wt = np.concatenate([np.full(len(m.uv), w / max(len(m.uv), 1)) for m, w in zip(models, weights)])
    best = (1.0, 0, 0, -np.inf)
    R = int(p["wb_shift_bins"])
    B = int(p["chroma_bins"])
    floor = np.log(float(p["hist_floor"]) / (B * B))
    for alpha in p["alphas"]:
        logm = np.log(smooth_model(uv_hist(uv * float(alpha), wt, p), p))
        pad = np.full((B + 2 * R, B + 2 * R), floor)
        pad[R:R + B, R:R + B] = logm
        win = np.lib.stride_tricks.sliding_window_view(pad, (B, B))
        ll = np.einsum("ijab,ab->ij", win, h_img)
        i, j = np.unravel_index(int(np.argmax(ll)), ll.shape)
        if ll[i, j] > best[3]:
            best = (float(alpha), R - int(i), R - int(j), float(ll[i, j]))
    return best


def colour_features(rgb: np.ndarray, mode: str, p: dict) -> np.ndarray:
    """(N, d) colour features of (N, 3) float RGB in [0, 1]: ``uv`` log-chromaticity, ``uvl`` plus log luma,
    ``lab`` CIE Lab, ``ab`` its a* b*."""
    import cv2  # noqa: PLC0415

    rgb = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
    if mode in ("uv", "uvl"):
        u, v = chroma_coords(rgb, p)
        cols = [u, v]
        if mode == "uvl":
            cols.append(np.log(rgb @ np.array([0.299, 0.587, 0.114], np.float32) + float(p["luma_eps"])))
        return np.stack(cols, 1).astype(np.float32)
    lab = cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)
    return lab[:, 1:] if mode == "ab" else lab


def _robust(x: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-column centre and scale (median and half the 16-84 % range; weighted when ``w`` is given)."""
    if w is None:
        q = np.percentile(x, [16, 50, 84], axis=0)
        return q[1], np.maximum((q[2] - q[0]) / 2, 1e-3)
    out_c, out_s = [], []
    for k in range(x.shape[1]):
        o = np.argsort(x[:, k])
        cw = np.cumsum(w[o])
        cw /= cw[-1]
        q16, q50, q84 = (x[o, k][np.searchsorted(cw, t)] for t in (0.16, 0.5, 0.84))
        out_c.append(q50)
        out_s.append(max((q84 - q16) / 2, 1e-3))
    return np.array(out_c), np.array(out_s)


def nd_bins(f: np.ndarray, p: dict) -> np.ndarray:
    """Flat bin index of standardised features in [-range, range] per axis, ``nd_bins`` bins per axis."""
    nbin = int(p["nd_bins"])
    r = float(p["nd_range"])
    idx = np.clip(((f + r) / (2 * r) * nbin).astype(np.int64), 0, nbin - 1)
    out = np.zeros(len(f), dtype=np.int64)
    for k in range(f.shape[1]):
        out = out * nbin + idx[:, k]
    return out


def nd_density(bins: np.ndarray, weights, d: int, p: dict) -> np.ndarray:
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    nbin = int(p["nd_bins"])
    h = np.bincount(bins, weights=weights, minlength=nbin ** d).astype(np.float64).reshape((nbin,) * d)
    if float(p["hist_sigma_bins"]) > 0:
        h = gaussian_filter(h, float(p["hist_sigma_bins"]), mode="constant")
    h = h.ravel()
    h /= max(h.sum(), 1e-12)
    f = float(p["hist_floor"])
    return (1 - f) * h + f / h.size


def uv_density(uv: np.ndarray, au: float, av: float, tu: float, tv: float, p: dict) -> np.ndarray:
    """Smoothed, floored density (flat, B*B) of the log-chromaticities mapped by u' = au u + tu, v' = av v + tv."""
    m = np.empty_like(uv)
    m[:, 0] = uv[:, 0] * au + tu
    m[:, 1] = uv[:, 1] * av + tv
    return smooth_model(uv_hist(m, None, p), p).ravel()


def fit_joint_map(RH, valid: np.ndarray, slot_k: np.ndarray, models: list, dens_bg: np.ndarray, p: dict) -> tuple:
    """The per-image chroma map (au, av, tu, tv) under which the best assignment of region proposals to the target
    slots explains the most: maximise the summed positive log-likelihood ratio of the Hungarian assignment
    (coarse grid with au = av, then a local refinement of each parameter)."""
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    K = len(models)
    lam = float(p["alt_bg_weight"])

    def J(par):
        D_ = np.stack([uv_density(m.uv, *par, p) for m in models], 1)
        alt = (D_.sum(1, keepdims=True) - D_ + lam * dens_bg[:, None]) / max(K - 1 + lam, 1e-9)
        S = np.asarray(RH @ (np.log(D_) - np.log(alt)))
        S = np.where(valid, S, -1e9)[:, slot_k]
        r, c = linear_sum_assignment(-S)
        return float(np.maximum(S[r, c], 0).sum())

    best, bj = None, -np.inf
    n_eval = 0
    for a_ in p["joint_alphas"]:
        for tu in p["joint_shifts"]:
            for tv in p["joint_shifts"]:
                par = (a_, a_, tu, tv)
                j = J(par)
                n_eval += 1
                if j > bj:
                    best, bj = par, j
    steps = list(p["joint_refine"])
    for _ in range(int(p["joint_refine_rounds"])):
        improved = False
        for k in range(4):
            for sgn in (-1, 1):
                par = list(best)
                par[k] = par[k] * (1 + sgn * steps[0]) if k < 2 else par[k] + sgn * steps[1]
                j = J(tuple(par))
                n_eval += 1
                if j > bj:
                    best, bj, improved = tuple(par), j, True
        if not improved:
            steps = [x / 2 for x in steps]
    return best, bj, n_eval


# ------------------------------------------------------------------------------------------- geometry
def working_arrays(frame, ctx: dict, p: dict) -> dict:
    import cv2  # noqa: PLC0415

    F = int(p["factor"])
    H, W = frame.rgb.shape[:2]
    h, w = H // F, W // F
    rgb = cv2.resize(np.ascontiguousarray(frame.rgb[:h * F, :w * F]), (w, h), interpolation=cv2.INTER_AREA)
    rgb = rgb.astype(np.float32) / 255.0
    z_full = ctx["z_obs"]
    z = np.ascontiguousarray(z_full[F // 2:h * F:F, F // 2:w * F:F]).astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    z = np.where(valid, z, 0.0)
    K = frame.K
    vv, uu = np.mgrid[0:h, 0:w]
    uf = uu * F + F // 2
    vf = vv * F + F // 2
    P = np.stack([(uf - K[0, 2]) / K[0, 0] * z, (vf - K[1, 2]) / K[1, 1] * z, z], axis=-1)
    return {"rgb": rgb, "z": z, "valid": valid, "P": P, "F": F, "shape": (h, w), "full": (H, W)}


def depth_cuts(z: np.ndarray, valid: np.ndarray, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Cut flags between horizontal (h, w-1) and vertical (h-1, w) neighbours."""
    a, b = float(p["cut_abs_m"]), float(p["cut_rel"])
    zh = np.maximum(z[:, 1:], z[:, :-1])
    zv = np.maximum(z[1:, :], z[:-1, :])
    ch = ~(valid[:, 1:] & valid[:, :-1]) | (np.abs(z[:, 1:] - z[:, :-1]) > a + b * zh)
    cv = ~(valid[1:, :] & valid[:-1, :]) | (np.abs(z[1:, :] - z[:-1, :]) > a + b * zv)
    return ch, cv


def normals(P: np.ndarray, valid: np.ndarray, ch: np.ndarray, cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals towards the camera from central differences (one working pixel each side, not across a cut)."""
    h, w = valid.shape
    du = np.zeros_like(P)
    dv = np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1, :] = P[2:, :] - P[:-2, :]
    okh = np.zeros((h, w), bool)
    okv = np.zeros((h, w), bool)
    okh[:, 1:-1] = ~ch[:, 1:] & ~ch[:, :-1]
    okv[1:-1, :] = ~cv[1:, :] & ~cv[:-1, :]
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1)
    ok = valid & okh & okv & (norm > 0)
    n = n / np.maximum(norm, 1e-12)[..., None]
    flip = (n * P).sum(-1) > 0
    n[flip] *= -1
    n[~ok] = 0
    return n, ok


def colour_edges(lab8: np.ndarray, p: dict) -> np.ndarray:
    """Strong colour edges at the working resolution (Canny on L, a and b of the 8-bit Lab image)."""
    import cv2  # noqa: PLC0415

    lo, hi = int(p["canny_lo"]), int(p["canny_hi"])
    e = cv2.Canny(np.ascontiguousarray(lab8[..., 0]), lo, hi) > 0
    if p["canny_ab"]:
        for k in (1, 2):
            ch = np.clip((lab8[..., k].astype(np.int16) - 128) * 2 + 128, 0, 255).astype(np.uint8)
            e |= cv2.Canny(ch, lo, hi) > 0
    return e


def _shifted(arr: np.ndarray, dr: int, dc: int, fill=0) -> np.ndarray:
    """out[r, c] = arr[r + dr, c + dc] (``fill`` outside)."""
    h, w = arr.shape[:2]
    out = np.full_like(arr, fill)
    rs = slice(max(dr, 0), h + min(dr, 0))
    cs = slice(max(dc, 0), w + min(dc, 0))
    rt = slice(max(-dr, 0), h + min(-dr, 0))
    ct = slice(max(-dc, 0), w + min(-dc, 0))
    out[rt, ct] = arr[rs, cs]
    return out


def patches(valid: np.ndarray, z: np.ndarray, ch: np.ndarray, cv: np.ndarray, edges: np.ndarray, lab_f: np.ndarray,
            p: dict) -> tuple[np.ndarray, int]:
    """Patch label image (0 = none) and the number of labels (patch ids 1..n-1). Patches never cross a depth cut or
    a strong colour edge and fit in a ``cell`` x ``cell`` grid cell; cut-boundary and edge pixels then join the
    depth-continuous neighbour patch of the closest colour."""
    import cv2  # noqa: PLC0415

    h, w = valid.shape
    boundary = np.zeros((h, w), bool)
    farther_h = z[:, 1:] >= z[:, :-1]
    boundary[:, 1:] |= ch & farther_h
    boundary[:, :-1] |= ch & ~farther_h
    farther_v = z[1:, :] >= z[:-1, :]
    boundary[1:, :] |= cv & farther_v
    boundary[:-1, :] |= cv & ~farther_v
    core = valid & ~boundary & ~edges
    _n_cc, cc = cv2.connectedComponents(core.astype(np.uint8), connectivity=4, ltype=cv2.CV_32S)
    C = int(p["cell"])
    gw = (w + C - 1) // C
    cell = (np.arange(h)[:, None] // C) * gw + (np.arange(w)[None, :] // C)
    n_cell = ((h + C - 1) // C) * gw
    key = np.where(cc > 0, cc.astype(np.int64) * n_cell + cell, 0)
    uniq, inv = np.unique(key.ravel(), return_inverse=True)
    lab = inv.reshape(h, w).astype(np.int32)
    if uniq[0] != 0:
        lab += 1
    n = int(lab.max()) + 1
    tol = float(p["cut_abs_m"]) + float(p["cut_rel"]) * z
    for _ in range(int(p["join_passes"])):
        todo = valid & (lab == 0)
        if not todo.any():
            break
        best = np.full((h, w), np.inf)
        new = lab.copy()
        for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nb = _shifted(lab, dr, dc)
            nz = _shifted(z, dr, dc)
            ncol = _shifted(lab_f, dr, dc)
            dcol = ((ncol - lab_f) ** 2).sum(-1)
            take = todo & (nb > 0) & (np.abs(nz - z) <= np.maximum(tol, _shifted(tol, dr, dc))) & (dcol < best)
            new[take] = nb[take]
            best[take] = dcol[take]
        lab = new
    return lab, n


def link_features(lab: np.ndarray, z: np.ndarray, ch: np.ndarray, cv: np.ndarray, n: int, p: dict) -> dict:
    """Adjacent patch pairs (a < b) not separated by a cut: boundary length and mean normalised depth step."""
    a1, b1 = lab[:, :-1], lab[:, 1:]
    m1 = (a1 != b1) & (a1 > 0) & (b1 > 0) & ~ch
    a2, b2 = lab[:-1, :], lab[1:, :]
    m2 = (a2 != b2) & (a2 > 0) & (b2 > 0) & ~cv
    A = np.concatenate([a1[m1], a2[m2]]).astype(np.int64)
    Bb = np.concatenate([b1[m1], b2[m2]]).astype(np.int64)
    zh = (z[:, :-1], z[:, 1:])
    zv = (z[:-1, :], z[1:, :])
    za = np.concatenate([zh[0][m1], zv[0][m2]])
    zb = np.concatenate([zh[1][m1], zv[1][m2]])
    step = np.abs(za - zb) / (float(p["cut_abs_m"]) + float(p["cut_rel"]) * np.maximum(za, zb))
    lo, hi = np.minimum(A, Bb), np.maximum(A, Bb)
    key, inv, cnt = np.unique(lo * n + hi, return_inverse=True, return_counts=True)
    ssum = np.bincount(inv, weights=step, minlength=len(key))
    return {"a": key // n, "b": key % n, "blen": cnt.astype(np.float64), "step": ssum / np.maximum(cnt, 1)}


def components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected-component id of every node of an undirected graph with edges (a, b)."""
    from scipy.sparse import coo_matrix  # noqa: PLC0415
    from scipy.sparse.csgraph import connected_components  # noqa: PLC0415

    g = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n))
    return connected_components(g, directed=False)[1]


# ------------------------------------------------------------------------------------------- merge tree
@dataclass
class MergeTree:
    """Regions made by merging patches in order of link dissimilarity (Kruskal), never past ``max_ext``: node ids
    0..n-1 are the patches, later ids the merges. ``left``/``right`` are -1 for a leaf."""

    npx: np.ndarray
    centre: np.ndarray  # (N, 3) pixel-weighted centroid
    radius: np.ndarray  # bounding-sphere radius (m)
    left: np.ndarray
    right: np.ndarray
    parent: np.ndarray
    n_leaf: int
    birth: np.ndarray | None = None  # link dissimilarity at which the node was made (0 for a patch)
    death: np.ndarray | None = None  # ... at which it was merged, or its first merge refused by the size limit

    def best_cut(self, quality: np.ndarray, valid: np.ndarray) -> list:
        """The antichain (disjoint nodes) maximising the summed quality (HDBSCAN-style selection; a node counts
        only where ``valid``)."""
        N = len(self.npx)
        best = np.where(valid, quality, 0.0).astype(np.float64)
        take = valid.copy()
        for x in range(self.n_leaf, N):
            kids = best[self.left[x]] + best[self.right[x]]
            if kids >= best[x] or not valid[x]:
                best[x] = kids
                take[x] = False
        out, stack = [], [x for x in range(N) if self.parent[x] < 0]
        while stack:
            x = stack.pop()
            if take[x]:
                out.append(x)
            elif x >= self.n_leaf:
                stack.append(int(self.left[x]))
                stack.append(int(self.right[x]))
        return out

    def leaves(self, node: int) -> list:
        out, stack = [], [node]
        while stack:
            x = stack.pop()
            if x < self.n_leaf:
                out.append(x)
            else:
                stack.append(int(self.left[x]))
                stack.append(int(self.right[x]))
        return out

    def subtree(self, node: int) -> list:
        out, stack = [], [node]
        while stack:
            x = stack.pop()
            out.append(x)
            if x >= self.n_leaf:
                stack.append(int(self.left[x]))
                stack.append(int(self.right[x]))
        return out

    def ancestors(self, node: int) -> list:
        out = []
        x = int(self.parent[node])
        while x >= 0:
            out.append(x)
            x = int(self.parent[x])
        return out

    def accumulate(self, leaf_vals: np.ndarray) -> np.ndarray:
        """Per-node sums of per-leaf values (rows), children before parents."""
        N = len(self.npx)
        out = np.zeros((N,) + leaf_vals.shape[1:], dtype=np.float64)
        out[:self.n_leaf] = leaf_vals
        for x in range(self.n_leaf, N):
            out[x] = out[self.left[x]] + out[self.right[x]]
        return out


def merge_tree(n: int, a: np.ndarray, b: np.ndarray, D: np.ndarray, use: np.ndarray, cnt: np.ndarray,
               cen: np.ndarray, rleaf: np.ndarray, max_ext: float) -> MergeTree:
    order = np.argsort(D, kind="stable")
    uf = list(range(n))
    node_of = list(range(n))
    npx = list(cnt.astype(float))
    cx, cy, cz = (list(cen[:, k].astype(float)) for k in range(3))
    rad = list(rleaf.astype(float))
    left, right = [-1] * n, [-1] * n
    parent = [-1] * n
    birth = [0.0] * n
    death = [-1.0] * n
    usel = use.tolist()
    Dl = D.tolist()
    al, bl = a.tolist(), b.tolist()
    lim = 0.5 * float(max_ext)

    def find(x):
        r = x
        while uf[r] != r:
            r = uf[r]
        while uf[x] != r:
            uf[x], x = r, uf[x]
        return r

    for e in order.tolist():
        x, y = al[e], bl[e]
        if not (usel[x] and usel[y]):
            continue
        rx, ry = find(x), find(y)
        if rx == ry:
            continue
        nx, ny = node_of[rx], node_of[ry]
        n1, n2 = npx[nx], npx[ny]
        tot = n1 + n2
        mx = (n1 * cx[nx] + n2 * cx[ny]) / tot
        my = (n1 * cy[nx] + n2 * cy[ny]) / tot
        mz = (n1 * cz[nx] + n2 * cz[ny]) / tot
        d1 = ((mx - cx[nx]) ** 2 + (my - cy[nx]) ** 2 + (mz - cz[nx]) ** 2) ** 0.5 + rad[nx]
        d2 = ((mx - cx[ny]) ** 2 + (my - cy[ny]) ** 2 + (mz - cz[ny]) ** 2) ** 0.5 + rad[ny]
        r = max(d1, d2)
        if r > lim:
            if death[nx] < 0:
                death[nx] = Dl[e]
            if death[ny] < 0:
                death[ny] = Dl[e]
            continue
        new = len(npx)
        if death[nx] < 0:
            death[nx] = Dl[e]
        if death[ny] < 0:
            death[ny] = Dl[e]
        birth.append(Dl[e])
        death.append(-1.0)
        npx.append(tot)
        cx.append(mx)
        cy.append(my)
        cz.append(mz)
        rad.append(r)
        left.append(nx)
        right.append(ny)
        parent.append(-1)
        parent[nx] = new
        parent[ny] = new
        uf[ry] = rx
        node_of[rx] = new
    dmax_ = float(np.max(D)) if len(D) else 1.0
    death = [x if x >= 0 else dmax_ for x in death]
    return MergeTree(np.array(npx), np.stack([cx, cy, cz], 1), np.array(rad), np.array(left), np.array(right),
                     np.array(parent), n, np.array(birth), np.array(death))


# ------------------------------------------------------------------------------------------- recognition
def render_visible(model: ObjectModel, R: np.ndarray, t: np.ndarray, K: np.ndarray, wa: dict, p: dict) -> dict | None:
    """The model's front surface at the working resolution under pose (R, t: model -> camera, metres): its z-buffer,
    silhouette, and the visible part (observed depth agrees, or no reading)."""
    import cv2  # noqa: PLC0415

    F = wa["F"]
    h, w = wa["shape"]
    X = model.nodes @ R.T + t
    Nc = model.normals @ R.T
    keep = ((Nc * X).sum(1) < 0) & (X[:, 2] > 0.05)
    X = X[keep]
    if len(X) < 10:
        return None
    u = K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2]
    v = K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]
    j = np.rint((u - F // 2) / F).astype(np.int64)
    i = np.rint((v - F // 2) / F).astype(np.int64)
    ok = (i >= 0) & (i < h) & (j >= 0) & (j < w)
    if ok.sum() < 10:
        return None
    zb = np.full(h * w, np.inf)
    np.minimum.at(zb, i[ok] * w + j[ok], X[ok, 2])
    zb = zb.reshape(h, w)
    sil0 = np.isfinite(zb)
    big = np.where(sil0, zb, 1e3).astype(np.float32)
    k3 = np.ones((3, 3), np.uint8)
    zf = cv2.erode(big, k3)  # min filter: fills one-pixel holes
    sil = cv2.morphologyEx(sil0.astype(np.uint8), cv2.MORPH_CLOSE, k3) > 0
    zb = np.where(sil0, zb, np.where(sil, zf, np.inf))
    z, valid = wa["z"], wa["valid"]
    tol = float(p["vis_tol_m"])
    agree = sil & valid & (np.abs(z - zb) < tol)
    seen = sil & valid
    vis = sil & (~valid | (np.abs(z - zb) < tol))
    return {"zb": zb, "sil": sil, "vis": vis, "agree": float(agree.sum() / max(seen.sum(), 1)),
            "n_sil": int(sil.sum())}


def kabsch(A: np.ndarray, B: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """R, t minimising sum w |R a + t - b|^2."""
    w = np.ones(len(A)) if w is None else w
    ca = (A * w[:, None]).sum(0) / w.sum()
    cb = (B * w[:, None]).sum(0) / w.sum()
    H = ((A - ca) * w[:, None]).T @ (B - cb)
    U, _S, Vt = np.linalg.svd(H)
    dd = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, dd]) @ U.T
    return R, cb - R @ ca


def refine_icp(model: ObjectModel, R: np.ndarray, t: np.ndarray, sil: np.ndarray, wa: dict, p: dict):
    """Point-to-point ICP of the model's camera-facing nodes to the observed points inside the (dilated) silhouette;
    trimmed by distance, so occluded nodes do not pull. Returns (R, t)."""
    import cv2  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    reg = cv2.dilate(sil.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    reg &= wa["valid"]
    Q = wa["P"][reg]
    if len(Q) < 30:
        return R, t
    tree = cKDTree(Q)
    nodes = model.nodes
    step = max(1, len(nodes) // int(p["icp_nodes"]))
    X0, N0 = nodes[::step], model.normals[::step]
    for it in range(int(p["icp_iters"])):
        X = X0 @ R.T + t
        front = ((N0 @ R.T) * X).sum(1) < 0
        if front.sum() < 30:
            break
        d, j = tree.query(X[front], k=1, workers=1)
        thr = float(p["icp_max_m"]) if it < 2 else float(p["icp_max_m"]) / 2
        ok = d < thr
        if ok.sum() < 30:
            break
        A = X0[front][ok]
        Bq = Q[j[ok]]
        R, t = kabsch(A, Bq)
    return R, t


def kabsch_batch(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched Kabsch: A, B (H, m, 3) -> R (H, 3, 3), t (H, 3) with R a + t ~ b."""
    ca = A.mean(1)
    cb = B.mean(1)
    Hm = np.einsum("hmi,hmj->hij", A - ca[:, None], B - cb[:, None])
    U, _S, Vt = np.linalg.svd(Hm)
    V = np.transpose(Vt, (0, 2, 1))
    d = np.sign(np.linalg.det(V @ np.transpose(U, (0, 2, 1))))
    D = np.zeros((len(A), 3, 3))
    D[:, 0, 0] = 1
    D[:, 1, 1] = 1
    D[:, 2, 2] = d
    R = V @ D @ np.transpose(U, (0, 2, 1))
    t = cb - np.einsum("hij,hj->hi", R, ca)
    return R, t


def ransac_3d(model_pts: np.ndarray, scene_pts: np.ndarray, p: dict, rng: np.random.Generator):
    """Rigid model -> scene transform from 3D-3D matches by RANSAC on triples (batched); (R, t, inlier indices) or
    None. Triples whose side lengths disagree by more than twice the threshold, or with a side under 1 cm, are
    skipped."""
    N = len(model_pts)
    if N < 3:
        return None
    it = int(p["ransac3d_iters"])
    thr = float(p["ransac3d_thr_m"])
    if N > 3:  # distinct triples: three draws, re-drawn where they collide
        tri = rng.integers(0, N, size=(it, 3))
        for _ in range(8):
            bad = (tri[:, 0] == tri[:, 1]) | (tri[:, 1] == tri[:, 2]) | (tri[:, 0] == tri[:, 2])
            if not bad.any():
                break
            tri[bad] = rng.integers(0, N, size=(int(bad.sum()), 3))
        tri = tri[~((tri[:, 0] == tri[:, 1]) | (tri[:, 1] == tri[:, 2]) | (tri[:, 0] == tri[:, 2]))]
    else:
        tri = np.array([[0, 1, 2]] * it)
    a, b_ = model_pts[tri], scene_pts[tri]  # (it, 3, 3)
    da = np.linalg.norm(a - a[:, [1, 2, 0]], axis=2)
    db = np.linalg.norm(b_ - b_[:, [1, 2, 0]], axis=2)
    ok = (da.min(1) >= float(p["ransac3d_min_side_m"])) & (np.abs(da - db).max(1) <= 2 * thr)
    if not ok.any():
        return None
    R, t = kabsch_batch(a[ok], b_[ok])
    res = np.linalg.norm(np.einsum("hij,nj->hni", R, model_pts) + t[:, None, :] - scene_pts[None], axis=2)
    n_in = (res < thr).sum(1)
    h = int(np.argmax(n_in))
    if n_in[h] < int(p["ransac3d_min_inliers"]):
        return None
    inl = np.flatnonzero(res[h] < thr)
    for _ in range(2):
        R1, t1 = kabsch(model_pts[inl], scene_pts[inl])
        r1 = np.linalg.norm(model_pts @ R1.T + t1 - scene_pts, axis=1)
        inl = np.flatnonzero(r1 < thr)
        if len(inl) < int(p["ransac3d_min_inliers"]):
            return None
    R1, t1 = kabsch(model_pts[inl], scene_pts[inl])
    return R1, t1, inl


def recognise(frame, wa: dict, targets: list, lib: Library, p: dict) -> tuple[list, dict]:
    """Instances found by texture: SIFT features of the image matched to the atlases of the image's objects (ratio
    test against all objects), then per object PnP RANSAC (sequentially, up to its instance count), a depth
    correction along the viewing ray, and the visible silhouette."""
    import cv2  # noqa: PLC0415

    t0 = time.perf_counter()
    fl, oid, xyz = lib.matcher()
    sc = float(p["sift_scale"])
    grey = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
    if sc != 1.0:
        grey = cv2.resize(grey, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=int(p["sift_features"]))
    kps, des = sift.detectAndCompute(grey, None)
    t1 = time.perf_counter()
    info = {"n_kp": 0 if des is None else len(kps)}
    out = []
    if des is None or len(kps) < 4:
        return out, info
    pts = np.array([k.pt for k in kps], dtype=np.float64) / sc
    qd = rootsift(des) if p["rootsift"] else des.astype(np.float32)
    kk = int(p["match_k"])
    m = fl.knnMatch(qd, k=kk)
    t2 = time.perf_counter()
    ratio = float(p["sift_ratio"])
    vr = float(p["vote_ratio"])
    present = {int(o): int(c) for o, c in targets}
    pres = np.zeros(int(oid.max()) + 1, bool)
    pres[[o for o in present if o <= oid.max()]] = True
    # (n, k) arrays of the neighbours (missing ones: distance inf)
    nq = len(m)
    TI = np.zeros((nq, kk), np.int64)
    DI = np.full((nq, kk), np.inf)
    for q_, lst in enumerate(m):
        for j_, x in enumerate(lst[:kk]):
            TI[q_, j_] = x.trainIdx
            DI[q_, j_] = x.distance
    ok_n = np.isfinite(DI)
    if kk > 2:  # the two best neighbours among the image's objects (present-only ratio test)
        ok_n &= pres[oid[TI]]
    first = np.argmax(ok_n, axis=1)
    has1 = ok_n[np.arange(nq), first]
    f0 = TI[np.arange(nq), first]
    cand = ok_n & (np.arange(kk)[None, :] > first[:, None])
    skip = float(p["ratio_skip_m"])
    if skip > 0:  # the runner-up must be another point: not the same surface point seen in another rendered view
        cand &= (oid[TI] != oid[f0][:, None]) | (np.linalg.norm(xyz[TI] - xyz[f0][:, None, :], axis=2) > skip)
    second = np.argmax(cand, axis=1)
    has2 = cand[np.arange(nq), second]
    d0 = DI[np.arange(nq), first]
    d1 = DI[np.arange(nq), second]
    both = has1 & has2
    gq = both & (d0 < ratio * d1)
    vg = both & (d0 < vr * d1)
    qi, ti = np.flatnonzero(gq), f0[gq]
    vq, vt = np.flatnonzero(vg), f0[vg]
    info["n_good"] = int(len(qi))
    info["votes"] = (pts[vq], oid[vt]) if len(vq) else (np.zeros((0, 2)), np.zeros(0, np.int64))
    K = frame.K
    n_pnp = 0
    zfull = frame.depth.astype(np.float64) * (float(frame.depth_scale) / 1000.0)
    Hf, Wf = zfull.shape
    rng = np.random.default_rng(int(p["ransac_seed"]))
    mode = p["pose_solver"]
    fails = {"no_pose": 0, "behind": 0, "render": 0, "depth_disagree": 0}
    for o, cnt_o in present.items():
        sel = np.flatnonzero(oid[ti] == o) if len(ti) else np.zeros(0, np.int64)
        img_pts = pts[qi[sel]]
        obj_pts = xyz[ti[sel]]
        xi = np.clip(np.rint(img_pts[:, 0]).astype(int), 0, Wf - 1)
        yi = np.clip(np.rint(img_pts[:, 1]).astype(int), 0, Hf - 1)
        zk = zfull[yi, xi]
        if int(p["kp_depth_win"]) > 0:  # the nearest valid reading in a small window (a keypoint on an edge)
            r_ = int(p["kp_depth_win"])
            win = np.stack([zfull[np.clip(yi + dy, 0, Hf - 1), np.clip(xi + dx, 0, Wf - 1)]
                            for dy in range(-r_, r_ + 1, max(1, r_)) for dx in range(-r_, r_ + 1, max(1, r_))], 1)
            win = np.where(win > 0.05, win, np.inf)
            zk = np.where(np.isfinite(win.min(1)), np.median(np.where(np.isfinite(win), win, np.nan), axis=1)
                          if p["kp_depth_stat"] == "median" else win.min(1), 0.0)
            zk = np.nan_to_num(zk, nan=0.0)
        scene = np.stack([(img_pts[:, 0] - K[0, 2]) / K[0, 0] * zk, (img_pts[:, 1] - K[1, 2]) / K[1, 1] * zk, zk], 1)
        n_found_o, tries = 0, 0
        while n_found_o < cnt_o and tries < cnt_o + int(p["extra_tries"]):
            tries += 1
            R = None
            n_in = 0
            if mode in ("kabsch", "both"):
                okz = np.flatnonzero(zk > 0.05)
                res3 = ransac_3d(obj_pts[okz], scene[okz], p, rng) if len(okz) >= 3 else None
                if res3 is not None:
                    R, t, inl3 = res3
                    inl = okz[inl3]
                    n_in = len(inl)
            if R is None and mode in ("pnp", "both") and len(img_pts) >= int(p["pnp_min_inliers"]):
                ok, rvec, tvec, inl = cv2.solvePnPRansac(obj_pts, img_pts, K, None,
                                                         iterationsCount=int(p["pnp_iters"]),
                                                         reprojectionError=float(p["pnp_reproj_px"]),
                                                         confidence=0.99, flags=cv2.SOLVEPNP_EPNP)
                if ok and inl is not None and len(inl) >= int(p["pnp_min_inliers"]):
                    inl = inl.ravel()
                    ok, rvec, tvec = cv2.solvePnP(obj_pts[inl], img_pts[inl], K, None, rvec, tvec,
                                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec.ravel()
                    n_in = len(inl)
            if R is None:
                fails["no_pose"] += 1
                break
            n_pnp += 1
            keep = np.ones(len(img_pts), bool)
            keep[inl] = False
            img_pts, obj_pts, scene, zk = img_pts[keep], obj_pts[keep], scene[keep], zk[keep]
            if t[2] <= 0.05:
                fails["behind"] += 1
                continue
            r = render_visible(lib.objects[o], R, t, K, wa, p)
            if r is None:
                fails["render"] += 1
                continue
            # depth correction along the viewing ray
            z, valid = wa["z"], wa["valid"]
            near = r["sil"] & valid & (np.abs(z - r["zb"]) < 0.08)
            if near.sum() >= 10:
                ratio_z = float(np.median(z[near]) / np.median(r["zb"][near]))
                if 0.8 < ratio_z < 1.25:
                    t = t * ratio_z
                    r2 = render_visible(lib.objects[o], R, t, K, wa, p)
                    if r2 is not None:
                        r = r2
            if p["icp_iters"]:
                R, t = refine_icp(lib.objects[o], R, t, r["sil"], wa, p)
                r2 = render_visible(lib.objects[o], R, t, K, wa, p)
                if r2 is not None:
                    r = r2
            if r["agree"] < float(p["min_depth_agree"]) or not r["vis"].any():
                fails["depth_disagree"] += 1
                continue
            out.append({"obj_id": o, "R": R, "t": t, "n_inliers": int(n_in), "vis": r["vis"],
                        "agree": r["agree"], "n_sil": r["n_sil"]})
            n_found_o += 1
    found = {}
    for r_ in out:
        found[r_["obj_id"]] = found.get(r_["obj_id"], 0) + 1
    vo_all = oid[vt] if len(vt) else np.zeros(0, np.int64)
    info["unfound_votes"] = [int((vo_all == o).sum()) for o, c in present.items() for _ in range(c - found.get(o, 0))]
    info.update({"n_pnp": n_pnp, "n_found": len(out), "fails": fails,
                 "ms": {"sift": round((t1 - t0) * 1e3, 2), "match": round((t2 - t1) * 1e3, 2),
                        "pnp": round((time.perf_counter() - t2) * 1e3, 2)}})
    return out, info


def calibration_pairs(recog: list, lib: Library, K: np.ndarray, wa: dict, lab_inst: np.ndarray, p: dict) -> tuple:
    """(texture RGB, image RGB) pairs at the visible nodes of the recognised instances (their own pixels only)."""
    F = wa["F"]
    h, w = wa["shape"]
    tex, img = [], []
    for k, r in enumerate(recog):
        if r.get("inst_id") is None:
            continue
        m = lib.objects[r["obj_id"]]
        X = m.nodes @ r["R"].T + r["t"]
        Nc = m.normals @ r["R"].T
        front = ((Nc * X).sum(1) < 0) & (X[:, 2] > 0.05)
        u = K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2]
        v = K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]
        j = np.rint((u - F // 2) / F).astype(np.int64)
        i = np.rint((v - F // 2) / F).astype(np.int64)
        ok = front & (i >= 0) & (i < h) & (j >= 0) & (j < w)
        i, j, idx = i[ok], j[ok], np.flatnonzero(ok)
        good = (lab_inst[i, j] == r["inst_id"]) & wa["valid"][i, j] & (np.abs(wa["z"][i, j] - X[idx, 2]) < p["vis_tol_m"])
        tex.append(m.node_rgb[idx[good]])
        img.append(wa["rgb"][i[good], j[good]])
    if not tex:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(tex), np.concatenate(img)


def fit_chroma_map(tex_rgb: np.ndarray, img_rgb: np.ndarray, p: dict) -> np.ndarray | None:
    """Affine map (2 x 3: uv_img = A uv_tex + b) by Huber IRLS on the pairs' log-chromaticities."""
    if len(tex_rgb) < int(p["calib_min_pairs"]):
        return None
    ut, vt = chroma_coords(tex_rgb, p)
    ui, vi = chroma_coords(img_rgb, p)
    X = np.stack([ut, vt, np.ones_like(ut)], 1)
    Y = np.stack([ui, vi], 1)
    wgt = np.ones(len(X))
    sol = None
    for _ in range(5):
        Xw = X * wgt[:, None]
        sol = np.linalg.lstsq(Xw.T @ X + 1e-6 * np.eye(3), Xw.T @ Y, rcond=None)[0]  # (3, 2)
        res = np.linalg.norm(Y - X @ sol, axis=1)
        c = float(p["calib_huber"])
        wgt = np.where(res <= c, 1.0, c / np.maximum(res, 1e-12))
    return sol.T  # (2, 3)


# ------------------------------------------------------------------------------------------- main
@dataclass
class SegMask:
    obj_id: int
    score: float
    crop: tuple  # (r0, r1, c0, c1) at full resolution
    mask: np.ndarray  # bool, the crop's pixels
    diag: dict = field(default_factory=dict)

    def full(self, shape) -> np.ndarray:
        out = np.zeros(shape, dtype=bool)
        r0, r1, c0, c1 = self.crop
        out[r0:r1, c0:c1] = self.mask
        return out


def max_projected_area(ext: np.ndarray) -> float:
    """Largest silhouette area (m^2) of a box with these extents (seen along its diagonal)."""
    e1, e2, e3 = (float(x) for x in ext)
    return float(np.sqrt((e1 * e2) ** 2 + (e1 * e3) ** 2 + (e2 * e3) ** 2))


def segment(frame, ctx: dict, targets: list, lib: Library, params: dict | None = None,
            want_proposals: bool = False) -> tuple[list, dict]:
    """``targets``: [(obj_id, inst_count), ...] of the image. Returns (``SegMask`` list, diag). With
    ``want_proposals``, ``diag["proposals"]`` holds working-resolution region label images (diagnostics only)."""
    import cv2  # noqa: PLC0415
    from scipy.sparse import coo_matrix, csr_matrix  # noqa: PLC0415

    p = dict(PARAMS, **(params or {}))
    tm = {}
    t0 = time.perf_counter()
    wa = working_arrays(frame, ctx, p)
    h, w = wa["shape"]
    z, valid, P, rgb = wa["z"], wa["valid"], wa["P"], wa["rgb"]
    objs = [int(o) for o, _ in targets]
    models = [lib.objects[o] for o in objs]
    dmax = max(m.diameter_m for m in models) if models else 0.3
    plane = ctx.get("plane")
    K = len(objs)
    tr = time.perf_counter()
    recog, rinfo = recognise(frame, wa, targets, lib, p) if p["recognise"] else ([], {})
    tm["recognise"] = time.perf_counter() - tr
    # recognised instances first (texture)
    lab_inst = np.zeros((h, w), np.int32)
    chosen = []  # (instance id, obj_id, score, n_px working, source)
    counts = [c for _, c in targets]
    left = {k: int(c) for k, c in enumerate(counts)}
    kidx = {o: k for k, o in enumerate(objs)}
    for rcg in sorted(recog, key=lambda r_: -r_["n_inliers"] * r_["agree"]):
        k = kidx[rcg["obj_id"]]
        if left[k] <= 0:
            continue
        mk = rcg["vis"] & (lab_inst == 0)
        if mk.sum() < p["min_px"]:
            continue
        chosen.append((len(chosen) + 1, rcg["obj_id"], 0.5 + 0.5 * min(1.0, rcg["n_inliers"] / 40.0) * rcg["agree"],
                       float(mk.sum()), "texture", rcg))
        lab_inst[mk] = len(chosen)
        rcg["inst_id"] = len(chosen)
        left[k] -= 1
    masks = []
    cinfo, n_prop, rows, cols_ = {}, 0, [], []
    live = np.zeros(1, bool)
    a = np.zeros(0, np.int64)
    bg_like = np.zeros(1, bool)
    need_regions = (not p["recog_only"] and any(v > 0 for v in left.values())) or bool(want_proposals and K)
    if need_regions:
        t0r = time.perf_counter()
        lab8 = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2Lab)
        labf = lab8.astype(np.float32)
        ch, cv = depth_cuts(z, valid, p)
        nrm, nok = normals(P, valid, ch, cv)
        edges = colour_edges(lab8, p) if p["colour_edges"] else np.zeros((h, w), bool)
        plab, n = patches(valid, z, ch, cv, edges, labf, p)
        tm["patches"] = time.perf_counter() - t0r

        # patch statistics and links
        t1 = time.perf_counter()
        L = plab.ravel()
        cnt = np.bincount(L, minlength=n).astype(np.float64)
        cnt[0] = 0
        live = cnt > 0
        inv_cnt = 1.0 / np.maximum(cnt, 1)
        cen = np.stack([np.bincount(L, weights=P[..., k].ravel(), minlength=n) for k in range(3)], 1) * inv_cnt[:, None]
        col = np.stack([np.bincount(L, weights=labf[..., k].ravel(), minlength=n) for k in range(3)], 1) * inv_cnt[:, None]
        nsum = np.stack([np.bincount(L, weights=(nrm[..., k] * nok).ravel(), minlength=n) for k in range(3)], 1)
        nn = np.linalg.norm(nsum, axis=1)
        pn = nsum / np.maximum(nn, 1e-12)[:, None]
        has_n = nn > 0
        lk = link_features(plab, z, ch, cv, n, p)
        a, b = lk["a"], lk["b"]
        d = cen[a] - cen[b]
        cosang = np.clip((pn[a] * pn[b]).sum(1), -1, 1)
        ang = np.degrees(np.arccos(cosang))
        both_n = has_n[a] & has_n[b]
        convex = ((pn[a] - pn[b]) * d).sum(1) > 0
        d_geo = np.where(convex, ang / p["convex_deg"], ang / p["concave_deg"])
        d_geo = np.where(both_n, d_geo, p["no_normal_d"])
        dE = np.linalg.norm(col[a] - col[b], axis=1)
        D = d_geo + lk["step"] / p["step_unit"] + dE / p["colour_unit"]
        flat = both_n & (ang < p["flat_deg"])
        coplanar = flat & (np.abs((pn[a] * d).sum(1)) < p["coplanar_m"]) & (np.abs((pn[b] * d).sum(1)) < p["coplanar_m"])
        tm["graph"] = time.perf_counter() - t1

        # background: large flat surfaces and the fitted plane
        t2 = time.perf_counter()
        surf = components(n, a[coplanar], b[coplanar])
        smin = np.full((n, 3), np.inf)
        smax = np.full((n, 3), -np.inf)
        np.minimum.at(smin, surf[live], cen[live])
        np.maximum.at(smax, surf[live], cen[live])
        sext = np.linalg.norm(np.where(np.isfinite(smax - smin), smax - smin, 0), axis=1)
        bg_like = live & (sext[surf] > p["bg_extent_factor"] * dmax)
        if plane is not None:
            pnrm = np.asarray(plane["normal"])
            hgt = cen @ pnrm + float(plane["d"])
            bg_like |= live & (np.abs(hgt) < p["plane_bg_m"]) & has_n & (np.abs(pn @ pnrm) > 0.9)
            if p["below_plane_m"] is not None:
                bg_like |= live & (hgt < -float(p["below_plane_m"]))
        fgp = live & ~bg_like
        tm["background"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        if p["proposal_mode"] == "levels":
            keep_link = fgp[a] & fgp[b]
            rows, cols_, seen = [], [], set()
            n_prop = 0
            fg_ids = np.flatnonzero(fgp)
            for lev in p["levels"]:
                m_ = keep_link & (D < lev)
                comp = components(n, a[m_], b[m_])
                cf = comp[fg_ids]
                order = np.argsort(cf, kind="stable")
                ids_sorted = fg_ids[order]
                cuts = np.flatnonzero(np.diff(cf[order])) + 1
                for members in np.split(ids_sorted, cuts):
                    if cnt[members].sum() < p["min_px"]:
                        continue
                    key_ = (int(members[0]), len(members))
                    if key_ in seen:
                        continue
                    seen.add(key_)
                    rows.append(np.full(len(members), n_prop))
                    cols_.append(members)
                    n_prop += 1
            tm["proposals"] = time.perf_counter() - t3

        tm["proposals"] = time.perf_counter() - t3
        calib = None
        n_pairs = 0
        if p["colour_map"] == "calib" and chosen:
            tex_rgb, img_rgb = calibration_pairs(recog, lib, frame.K, wa, lab_inst, p)
            n_pairs = len(tex_rgb)
            calib = fit_chroma_map(tex_rgb, img_rgb, p)

        # proposal statistics and scores
        t4 = time.perf_counter()
        cand_diag = {}
        sel_px = L > 0
        fg_px = sel_px & fgp[L]
        bg_px = sel_px & bg_like[L]
        counts = [c for _, c in targets]
        K = len(objs)
        alpha, du, dv = 1.0, 0, 0
        bu = float(p["bg_uniform"])
        if p["colour_map"] == "moments":
            d_ = {"uv": 2, "uvl": 3, "lab": 3, "ab": 2}[p["colour_feat"]]
            nb = int(p["nd_bins"]) ** d_
            fpx = colour_features(rgb.reshape(-1, 3), p["colour_feat"], p)
            ci, si = _robust(fpx[fg_px] if fg_px.any() else fpx)
            fl = [colour_features(m.rgb, p["colour_feat"], p) for m in models]
            wl = np.concatenate([np.full(len(f_), float(c) * m.diameter_m ** 2 / len(f_)) for f_, m, c in
                                 zip(fl, models, counts)]) if K else None
            cm_, sm_ = _robust(np.concatenate(fl), wl) if K else (ci, si)
            pix_bin = nd_bins((fpx - ci) / si, p)
            dens = [nd_density(nd_bins((f_ - cm_) / sm_, p), None, d_, p) for f_ in fl]
            h_bg = np.bincount(pix_bin[bg_px], minlength=nb).astype(np.float64)
            h_bg = h_bg / max(h_bg.sum(), 1.0)
            dens_bg = (1 - bu) * h_bg + bu / nb
            cinfo = {"img_centre": np.round(ci, 3).tolist(), "img_scale": np.round(si, 3).tolist(),
                     "lib_centre": np.round(cm_, 3).tolist(), "lib_scale": np.round(sm_, 3).tolist()}
        elif p["colour_map"] == "calib" and calib is not None:
            B = int(p["chroma_bins"])
            nb = B * B
            pix_bin = chroma_bin(rgb, p).ravel()
            h_bg = np.bincount(pix_bin[bg_px], minlength=nb).astype(np.float64)
            h_bg = h_bg / max(h_bg.sum(), 1.0)
            dens_bg = smooth_model(((1 - bu) * h_bg + bu / nb).reshape(B, B), p).ravel()
            dens = []
            for m in models:
                uvm = np.concatenate([m.uv, np.ones((len(m.uv), 1), np.float32)], 1) @ calib.T
                dens.append(smooth_model(uv_hist(uvm, None, p), p).ravel())
            cinfo = {"calib": np.round(calib, 4).tolist(), "n_pairs": int(n_pairs)}
        elif p["colour_map"] == "joint":
            B = int(p["chroma_bins"])
            nb = B * B
            pix_bin = chroma_bin(rgb, p).ravel()
            h_bg = np.bincount(pix_bin[bg_px], minlength=nb).astype(np.float64)
            h_bg = h_bg / max(h_bg.sum(), 1.0)
            dens_bg = smooth_model(((1 - bu) * h_bg + bu / nb).reshape(B, B), p).ravel()
            Hp0 = coo_matrix((np.ones(int(sel_px.sum())), (L[sel_px], pix_bin[sel_px])), shape=(n, nb)).tocsr()
            par = (1.0, 1.0, 0.0, 0.0)
            n_eval = 0
            if n_prop and K:
                rr0 = np.concatenate(rows)
                cc0 = np.concatenate(cols_)
                M0 = csr_matrix((np.ones(len(rr0)), (rr0, cc0)), shape=(n_prop, n))
                npx0 = np.asarray(M0 @ cnt).ravel()
                C0 = np.asarray(M0 @ (cen * cnt[:, None])) / np.maximum(npx0, 1)[:, None]
                rmax0 = np.zeros(n_prop)
                np.maximum.at(rmax0, rr0, np.linalg.norm(cen[cc0] - C0[rr0], axis=1))
                zc0 = np.maximum(C0[:, 2], 0.05)
                f20 = (frame.K[0, 0] / wa["F"]) * (frame.K[1, 1] / wa["F"])
                valid0 = np.stack([(2 * rmax0 <= p["size_factor"] * m.diameter_m)
                                   & (npx0 <= p["area_factor"] * max_projected_area(m.extents_m) * f20 / zc0 ** 2)
                                   & (npx0 >= p["joint_min_px"]) for m in models], 1)
                slot_k = np.array([k for k, c in enumerate(counts) for _ in range(int(c))])
                par, _jv, n_eval = fit_joint_map(M0 @ Hp0, valid0, slot_k, models, dens_bg, p)
            dens = [uv_density(m.uv, *par, p) for m in models]
            cinfo = {"map": [round(float(x), 4) for x in par], "n_eval": n_eval}
        else:
            B = int(p["chroma_bins"])
            nb = B * B
            pix_bin = chroma_bin(rgb, p).ravel()
            h_img = np.bincount(pix_bin[fg_px], minlength=nb).astype(np.float64).reshape(B, B)
            h_img /= max(h_img.sum(), 1.0)
            if p["colour_map"] == "alpha_shift" and models:
                alpha, du, dv, _ll = fit_colour_map(h_img, models, [float(c) * m.diameter_m ** 2
                                                                     for m, c in zip(models, counts)], p)
            elif p["colour_map"] in ("shift", "calib") and models:
                du, dv, _ll = fit_white_balance(h_img, [m.chroma for m in models],
                                                [float(c) * m.diameter_m ** 2 for m, c in zip(models, counts)], p)
            if alpha != 1.0:
                dens = [smooth_model(shift_hist(uv_hist(m.uv * alpha, None, p), du, dv), p).ravel() for m in models]
            else:
                dens = [smooth_model(shift_hist(m.chroma, du, dv), p).ravel() for m in models]
            h_bg = np.bincount(pix_bin[bg_px], minlength=nb).astype(np.float64)
            h_bg = h_bg / max(h_bg.sum(), 1.0)
            dens_bg = smooth_model(((1 - bu) * h_bg + bu / nb).reshape(B, B), p).ravel()
            cinfo = {"alpha": alpha, "shift_bins": [int(du), int(dv)], "n_pairs": int(n_pairs)}
        Hp = coo_matrix((np.ones(int(sel_px.sum())), (L[sel_px], pix_bin[sel_px])), shape=(n, nb)).tocsr()
        Dm = np.stack(dens, 1) if K else np.zeros((nb, 0))  # (nb, K)
        lam = float(p["alt_bg_weight"])
        tot = Dm.sum(1, keepdims=True)
        alt = (tot - Dm + lam * dens_bg[:, None]) / max(K - 1 + lam, 1e-9)
        LLR = np.log(Dm) - np.log(alt)  # (nb, K)
        if p["diag_mode"] == "pixel_argmax" and K:  # diagnostic: every foreground pixel to its best colour model
            F = wa["F"]
            llr_px = LLR[pix_bin].reshape(h, w, K) if not p["px_smooth"] else None
            if p["px_smooth"]:
                r = int(p["px_smooth"])
                llr_px = np.stack([cv2.boxFilter(LLR[pix_bin, k].reshape(h, w).astype(np.float32), -1, (2 * r + 1, 2 * r + 1))
                                   for k in range(K)], -1)
            best = llr_px.argmax(-1)
            fgmask = (plab > 0) & fgp[plab]
            for k in range(K):
                mk = fgmask & (best == k)
                rws = np.flatnonzero(mk.any(axis=1))
                if not len(rws):
                    continue
                cls = np.flatnonzero(mk[rws[0]:rws[-1] + 1].any(axis=0))
                r0, r1, c0, c1 = int(rws[0]), int(rws[-1]) + 1, int(cls[0]), int(cls[-1]) + 1
                up = np.repeat(np.repeat(mk[r0:r1, c0:c1], F, axis=0), F, axis=1)
                masks.append(SegMask(objs[k], 1.0, (r0 * F, r1 * F, c0 * F, c1 * F), up))
            n_prop = 0
        # proposals for the remaining slots (colour + geometry + texture votes)
        Vp = None
        if p["vote_weight"] and "votes" in rinfo and K:
            vp, vo = rinfo["votes"]
            F_ = wa["F"]
            vj = np.clip(np.rint((vp[:, 0] - F_ // 2) / F_).astype(np.int64), 0, w - 1)
            vi = np.clip(np.rint((vp[:, 1] - F_ // 2) / F_).astype(np.int64), 0, h - 1)
            vpatch = plab[vi, vj]
            okv = (vpatch > 0) & np.isin(vo, objs)
            if okv.any():
                Vp = coo_matrix((np.ones(int(okv.sum())), (vpatch[okv], np.array([kidx[int(o)] for o in vo[okv]]))),
                                shape=(n, K)).tocsr()
        if p["proposal_mode"] == "tree" and K and not p["recog_only"] and any(v > 0 for v in left.values()):
            tt = time.perf_counter()
            f_ = frame.K[0, 0] / wa["F"]
            rleaf = 0.5 * np.sqrt(np.maximum(cnt, 1)) * np.maximum(cen[:, 2], 0.05) / f_ * np.sqrt(2)
            tree = merge_tree(n, a, b, D, fgp, cnt, cen, rleaf, p["size_factor"] * dmax)
            SP = np.asarray(Hp @ LLR)
            if Vp is not None:
                SP = SP + float(p["vote_weight"]) * np.asarray(Vp.todense())
            taken = np.bincount(L, weights=(lab_inst > 0).ravel().astype(np.float64), minlength=n)
            leafv = np.concatenate([SP, taken[:, None]], 1)
            leafv[~fgp] = 0
            acc = tree.accumulate(leafv)
            Sc, usedpx = acc[:, :K], acc[:, K]
            npx = tree.npx
            zc = np.maximum(tree.centre[:, 2], 0.05)
            f2 = (frame.K[0, 0] / wa["F"]) * (frame.K[1, 1] / wa["F"])
            isfg = np.ones(len(npx), bool)
            isfg[:n] = fgp
            ok_all = isfg & (npx >= p["min_px"]) & (usedpx <= p["max_overlap"] * npx)
            if p["score_mode"] == "prob":
                colour_sum, votes = acc[:, :K] - 0.0, None
                rawv = tree.accumulate(np.asarray(Vp.todense()) if Vp is not None else np.zeros((n, K)))
                rawc = tree.accumulate(np.asarray(Hp @ LLR))
                Sc = float(p["vote_logodds"]) * np.minimum(rawv, float(p["vote_cap"])) + rawc / float(p["colour_tau"])
                if p["vote_other_penalty"]:
                    other = rawv.sum(1, keepdims=True) - rawv
                    Sc -= float(p["vote_other_penalty"]) * np.minimum(other, float(p["vote_cap"]))
                for k, m in enumerate(models):
                    a_vis = float(p["vis_share"]) * m.area_m2 / 4.0 * f2 / zc ** 2
                    Sc[:, k] -= np.log(np.maximum(npx, 1) / a_vis) ** 2 / (2 * float(p["size_sigma"]) ** 2)
            for k, m in enumerate(models):
                ok = ok_all & (2 * tree.radius <= p["size_factor"] * m.diameter_m)
                ok &= npx <= p["area_factor"] * max_projected_area(m.extents_m) * f2 / zc ** 2
                Sc[~ok, k] = -np.inf
            used = taken > 0.5 * np.maximum(cnt, 1)
            alive = np.ones(len(npx), bool)
            tm["tree"] = time.perf_counter() - tt
            if p["tree_select"] in ("cut", "sizecut"):
                from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

                vnode = ok_all & (2 * tree.radius <= p["size_factor"] * dmax)
                if p["tree_select"] == "cut":
                    life = np.maximum(tree.death - tree.birth, 0.0)
                    qual = np.power(npx, float(p["cut_size_exp"])) * life
                else:  # pixels explained by object-sized regions (log-normal size prior of the best remaining object)
                    g = np.zeros(len(npx))
                    for k, m in enumerate(models):
                        if left[k] <= 0:
                            continue
                        a_vis = float(p["vis_share"]) * m.area_m2 / 4.0 * f2 / zc ** 2
                        g = np.maximum(g, np.exp(-np.log(np.maximum(npx, 1) / a_vis) ** 2
                                                 / (2 * float(p["size_sigma"]) ** 2)))
                    qual = npx * g
                segs = tree.best_cut(qual, vnode)
                if want_proposals:
                    cut_props = []
                    for x in segs:
                        mem = np.zeros(n, bool)
                        mem[tree.leaves(x)] = True
                        sx = np.where(np.isfinite(Sc[x]), Sc[x], -1e18)
                        cut_props.append((mem[plab] & (plab > 0), [objs[int(q)] for q in np.argsort(-sx)[:3]]))
                slots = [k for k, v in left.items() for _ in range(v)]
                if segs and slots:
                    Sseg = Sc[segs][:, slots]
                    Sfin = np.where(np.isfinite(Sseg), Sseg, -1e12)
                    r_i, c_i = linear_sum_assignment(-Sfin)
                    for ri, ci in sorted(zip(r_i, c_i), key=lambda t_: -Sfin[t_[0], t_[1]]):
                        if Sfin[ri, ci] <= p["min_score"]:
                            continue
                        x, k = segs[ri], slots[ci]
                        members = np.array(tree.leaves(x), dtype=np.int64)
                        members = members[~used[members]]
                        used[members] = True
                        chosen.append((len(chosen) + 1, objs[k], 0.25, float(npx[x]), "region", None))
                        pm = np.zeros(n, bool)
                        pm[members] = True
                        lab_inst[pm[plab] & (plab > 0) & (lab_inst == 0)] = len(chosen)
                        left[k] -= 1
            while p["tree_select"] == "greedy" and any(v > 0 for v in left.values()):
                Sw = Sc.copy()
                for k, v in left.items():
                    if v <= 0:
                        Sw[:, k] = -np.inf
                Sw[~alive] = -np.inf
                j = int(np.argmax(Sw))
                r_, k = divmod(j, K)
                if not np.isfinite(Sw[r_, k]) or Sw[r_, k] <= p["min_score"]:
                    break
                members = np.array(tree.leaves(r_), dtype=np.int64)
                members = members[~used[members]]
                used[members] = True
                alive[tree.subtree(r_)] = False
                alive[tree.ancestors(r_)] = False
                chosen.append((len(chosen) + 1, objs[k], 0.25, float(npx[r_]), "region", None))
                if p["debug"]:
                    a_vis_k = float(p["vis_share"]) * models[k].area_m2 / 4.0 * f2 / zc[r_] ** 2
                    print(f"pick obj {objs[k]} node {r_} npx {npx[r_]:.0f} a_vis {a_vis_k:.0f} r {tree.radius[r_]:.3f} "
                          f"diam {models[k].diameter_m:.3f} score {Sw[r_, k]:.1f} votes {rawv[r_, k]:.0f} "
                          f"colour {rawc[r_, k] / float(p['colour_tau']):.1f} z {zc[r_]:.2f} members {len(members)}",
                          flush=True)
                pm = np.zeros(n, bool)
                pm[members] = True
                lab_inst[pm[plab] & (plab > 0) & (lab_inst == 0)] = len(chosen)
                left[k] -= 1
            n_prop = len(npx)
            if want_proposals and p["prop_tree"]:
                img_t = np.zeros((h, w), np.int32)
                labels_t = {}
                big = [x for x in range(len(npx)) if isfg[x] and npx[x] >= p["prop_min_px"]
                       and 2 * tree.radius[x] <= p["size_factor"] * dmax]
                mode_ = p["diag_label_score"]
                raw = tree.accumulate(np.concatenate([np.asarray(Hp @ LLR), (np.asarray(Vp.todense()) if Vp is not None
                                                                             else np.zeros((n, K)))], 1))
                if mode_ == "colour":
                    Sl = raw[:, :K]
                elif mode_ == "votes":
                    Sl = raw[:, K:]
                elif mode_ == "colour_mean":
                    Sl = raw[:, :K] / np.maximum(npx, 1)[:, None]
                else:
                    Sl = Sc
                tree_props = []
                for x in big:
                    mem = np.zeros(n, bool)
                    mem[tree.leaves(x)] = True
                    tree_props.append((mem[plab] & (plab > 0),
                                       [objs[int(q)] for q in np.argsort(-np.where(np.isfinite(Sl[x]), Sl[x], -1e18))[:3]]))
                diag_tree_props = tree_props
        if p["proposal_mode"] == "levels" and n_prop and K and not p["recog_only"] and any(v > 0 for v in left.values()):
            rr = np.concatenate(rows)
            cc_ = np.concatenate(cols_)
            M = csr_matrix((np.ones(len(rr)), (rr, cc_)), shape=(n_prop, n))
            npx = np.asarray(M @ cnt).ravel()
            Sc = np.asarray((M @ Hp) @ LLR)  # (n_prop, K)
            if Vp is not None:
                Sc = Sc + float(p["vote_weight"]) * np.asarray((M @ Vp).todense())
            C = np.asarray(M @ (cen * cnt[:, None])) / np.maximum(npx, 1)[:, None]
            dist = np.linalg.norm(cen[cc_] - C[rr], axis=1)
            rmax = np.zeros(n_prop)
            np.maximum.at(rmax, rr, dist)
            ext = 2.0 * rmax
            zc = np.maximum(C[:, 2], 0.05)
            f2 = (frame.K[0, 0] / wa["F"]) * (frame.K[1, 1] / wa["F"])
            for k, m in enumerate(models):
                ok = ext <= p["size_factor"] * m.diameter_m
                ok &= npx <= p["area_factor"] * max_projected_area(m.extents_m) * f2 / zc ** 2
                Sc[~ok, k] = -np.inf
            # patches already taken by recognised instances
            taken = np.bincount(L, weights=(lab_inst > 0).ravel().astype(np.float64), minlength=n)
            used = taken > 0.5 * np.maximum(cnt, 1)
            ov = np.asarray(M @ (used * cnt)).ravel()
            alive = ov <= p["max_overlap"] * npx
            Sw = Sc.copy()
            while any(v > 0 for v in left.values()):
                for k, v in left.items():
                    if v <= 0:
                        Sw[:, k] = -np.inf
                Sw[~alive] = -np.inf
                j = int(np.argmax(Sw))
                r_, k = divmod(j, K)
                if not np.isfinite(Sw[r_, k]) or Sw[r_, k] <= p["min_score"]:
                    break
                members = cc_[rr == r_]
                members = members[~used[members]]
                used[members] = True
                chosen.append((len(chosen) + 1, objs[k], 0.25, float(npx[r_]), "region", None))
                pm = np.zeros(n, bool)
                pm[members] = True
                lab_inst[pm[plab] & (plab > 0) & (lab_inst == 0)] = len(chosen)
                left[k] -= 1
                ov = np.asarray(M @ (used * cnt)).ravel()
                alive &= ov <= p["max_overlap"] * npx
    for _ in range(int(p["fill_iters"])):
        hole = (~valid) & (lab_inst == 0)
        if not hole.any():
            break
        grown = cv2.dilate(lab_inst.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        lab_inst = np.where(hole, grown, lab_inst)
    F = wa["F"]
    for kk, o, sc, npx_r, src, rcg in chosen:
        mk = lab_inst == kk
        rws = np.flatnonzero(mk.any(axis=1))
        if not len(rws):
            continue
        cls = np.flatnonzero(mk[rws[0]:rws[-1] + 1].any(axis=0))
        r0, r1, c0, c1 = int(rws[0]), int(rws[-1]) + 1, int(cls[0]), int(cls[-1]) + 1
        up = np.repeat(np.repeat(mk[r0:r1, c0:c1], F, axis=0), F, axis=1)
        info_m = {"npx_work": npx_r, "source": src}
        if rcg is not None:  # the texture pose (model -> camera; t in mm), a by-product the pose layer may use
            info_m.update({"R_m2c": np.asarray(rcg["R"]).round(6).tolist(),
                           "t_m2c_mm": (np.asarray(rcg["t"]) * 1000.0).round(3).tolist(),
                           "n_inliers": int(rcg["n_inliers"]), "depth_agree": round(float(rcg["agree"]), 4)})
        masks.append(SegMask(o, sc, (r0 * F, r1 * F, c0 * F, c1 * F), up, info_m))
    tm["select"] = (time.perf_counter() - t4) if need_regions else 0.0
    tm["total"] = time.perf_counter() - t0
    diag = {"n_patches": int(live.sum()), "n_links": int(len(a)), "n_bg_like": int(bg_like.sum()),
            "colour": cinfo, "n_proposals": int(n_prop), "n_slots": int(sum(counts)), "recognise": {k_: v_ for k_, v_ in rinfo.items() if k_ != "votes"},
            "n_texture": int(sum(1 for c in chosen if c[4] == "texture")),
            "n_masks": len(masks), "ms": {k: round(v * 1e3, 2) for k, v in tm.items()}, "plane": plane is not None}
    if want_proposals and K and p["proposal_mode"] == "tree":
        diag["proposals"] = {}
        if "diag_tree_props" in locals():
            diag["proposals"]["tree"] = diag_tree_props
        if "cut_props" in locals():
            diag["proposals"]["cut"] = cut_props
    elif want_proposals and K:
        prop_imgs = {}
        HL = (Hp @ LLR) if K else None  # (n, K) per patch summed LLR
        for li, lev in enumerate(p["levels"]):
            if li not in p["prop_levels"]:
                continue
            m_ = keep_link & (D < lev)
            comp = components(n, a[m_], b[m_]) + 1
            comp[~fgp] = 0
            Mc = csr_matrix((np.ones(int(fgp.sum())), (comp[fgp], np.flatnonzero(fgp))), shape=(n + 1, n))
            sc = np.asarray(Mc @ HL)
            top = np.argsort(-sc, axis=1)[:, :3]
            labels = {int(c): [objs[int(k)] for k in top[c]] for c in np.unique(comp[fgp])}
            prop_imgs[f"L{li}"] = (comp[plab] * (plab > 0), labels)
        diag["proposals"] = prop_imgs
    return masks, diag


def proposal_detections(scene_id: int, im_id: int, img, F: int, full_shape, min_px: int = 4) -> list:
    """Every region of a working-resolution label image as a detection with category 0 (diagnostics only); with
    ``img = (label image, {region: ranked object ids})`` each detection also carries ``labels``."""
    out = []
    labels = None
    if isinstance(img, list):  # [(working mask, ranked labels), ...]
        for k, (m, labs) in enumerate(img):
            rows = np.flatnonzero(m.any(axis=1))
            if not len(rows):
                continue
            cols = np.flatnonzero(m[rows[0]:rows[-1] + 1].any(axis=0))
            r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
            up = np.repeat(np.repeat(m[r0:r1, c0:c1], F, axis=0), F, axis=1)
            det = detection(scene_id, im_id, SegMask(0, float(m.sum()), (r0 * F, r1 * F, c0 * F, c1 * F), up),
                            full_shape, -1.0)
            det["labels"] = labs
            out.append(det)
        return out
    if isinstance(img, tuple):
        img, labels = img
    ids, cnt = np.unique(img[img > 0], return_counts=True)
    for k, c in zip(ids, cnt):
        if c < min_px:
            continue
        m = img == k
        rows = np.flatnonzero(m.any(axis=1))
        cols = np.flatnonzero(m[rows[0]:rows[-1] + 1].any(axis=0))
        r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
        up = np.repeat(np.repeat(m[r0:r1, c0:c1], F, axis=0), F, axis=1)
        det = detection(scene_id, im_id, SegMask(0, float(c), (r0 * F, r1 * F, c0 * F, c1 * F), up), full_shape, -1.0)
        if labels is not None:
            det["labels"] = labels.get(int(k), [])
        out.append(det)
    return out


# ------------------------------------------------------------------------------------------- output
def rle_counts(full_shape, crop: tuple, mask: np.ndarray) -> list:
    """COCO uncompressed RLE (column-major runs, background first) of a crop placed in a full-size mask."""
    H, W = full_shape
    r0, r1, c0, c1 = crop
    sub = np.zeros((H, c1 - c0), dtype=bool)
    sub[r0:r1] = mask
    flat = sub.T.reshape(-1)
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds)
    lead = c0 * H
    if flat[0]:
        runs = np.concatenate([[0], runs])
    runs = runs.tolist()
    runs[0] += lead
    tail = (W - c1) * H
    if tail:
        if len(runs) % 2 == 1:  # ends on background
            runs[-1] += tail
        else:
            runs.append(tail)
    return [int(x) for x in runs]


def detection(scene_id: int, im_id: int, m: SegMask, full_shape, time_s: float) -> dict:
    r0, r1, c0, c1 = m.crop
    rows = np.flatnonzero(m.mask.any(axis=1))
    cols = np.flatnonzero(m.mask.any(axis=0))
    bbox = [int(c0 + cols[0]), int(r0 + rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)]
    out = {"scene_id": int(scene_id), "image_id": int(im_id), "category_id": int(m.obj_id), "bbox": bbox,
           "score": round(float(m.score), 6), "time": round(float(time_s), 6),
           "segmentation": {"counts": rle_counts(full_shape, m.crop, m.mask),
                            "size": [int(full_shape[0]), int(full_shape[1])]}}
    for k in ("source", "R_m2c", "t_m2c_mm", "n_inliers", "depth_agree"):
        if k in m.diag:
            out[k] = m.diag[k]
    return out
