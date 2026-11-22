"""M2 perception server (plan perception_pipeline, venv_bridge.server). Monty venv, one process, single-threaded::

    cd <run scratch folder> && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
        PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<repository root> <server python> \\
        -P -m rowant_lab.libero_env.m2_perceive --socket p.sock --log-dir DIR [--params JSON]

M2 (perceived poses, grounded instruction): perception in the loop. TRUTH-FREE: it receives only camera frames
(RGB, depth, K, T), the robot's proprioception (joint and gripper positions, for the self-filter) and request fields;
it reads only the M2 library (exported meshes and onboarded artifacts). It never imports LIBERO, robosuite, MuJoCo
or torch, and it never writes an image.

**Full look** (``Perceiver.full``), on the agentview frame:

1. world points from the depth, K and T; the workspace = points within ``ws_radius_m`` (horizontal) of the robot
   base (given by the workcell model) and above the floor;
2. self-filter: the Panda's visual surface (exported) at forward kinematics of the proprioception, sampled at about
   4 mm; observed points within ``self_filter_m`` of it are the robot. Robot and out-of-workspace pixels get depth 0
   (no reading) for every later step;
3. the fast layer's plane fit (the support plane; its world height is logged, never replaced);
4. segmentation, our own: ``segment.segment`` (texture recognition with SIFT templates of the onboarded meshes, then
   region proposals for the remaining slots) over the workcell's rigid parts with ``k_max`` instances each (no
   inventory and no counts are given);
5. pose, the fast layer (FL6's preset: ``hope_fast`` + ``px_ref_f`` 1390, flips ``generators``, refine; scorer A
   ``MontyColourTuned``; the weighted rule, weight 1; variant B) on each mask;
6. acceptance floor (``fit_check``): the part's visible graph nodes at the pose against the observed depth: the share
   that agree within ``agree_m`` among those that agree or are seen through (observed farther than the surface by
   ``see_through_m``) must reach ``accept_inlier``, with at least ``accept_min_agree`` agreeing nodes; duplicates
   (centres within ``dup_frac`` of the smaller diameter) keep the better fit;
7. articulated fixtures: each jointed part's joint value is fitted alone (grid over its range in 2 mm steps, then a
   golden-section refine) by the same node agreement (agree minus 2 x seen-through) of the part placed at the value;
8. percepts: per percept its type, the world pose of every body of the type (``m2_kin.part_body_poses``), joints,
   score, fit, source and pixel count. Never an image or a mask.

Determinism: every seeded draw is keyed by the request's ``episode`` and ``look`` tokens, never by server-side
counters, so the two episode workers that share one server cannot change each other's seeds.

**History-free output** (amendment A39, m2f2:determinism-r1; verifier blockers B1/R1/X1): a reply is a function only
of (frozen config, episode token, look inputs), never of what the process did before. The fast layer builds each
suite's texture matcher (``segment.Library.matcher``: a FLANN randomized kd-tree index) from OpenCV's process-global
RNG, lazily at that suite's first look, so a suite's index used to depend on which suite a fresh server served first
(and on any earlier draw). ``build_matcher`` now builds it EAGERLY when the suite's workcell is loaded, from the fixed
state ``MATCHER_RNG_SEED`` (0: a fresh OpenCV thread's state, so each index is the one a fresh process builds for its
first suite), and ``reseed_request`` sets OpenCV's, numpy's and Python's global RNGs from the request's own episode
token and look index at the start of every look, so no draw of one request can reach another. The fast layer itself
is unchanged.

**Track look** (``Perceiver.track``): for the requested percepts, the fast layer on the pixels inside the predicted
silhouette (the visible nodes at the predicted pose, dilated) that agree with the prediction within ``track_gate_m``;
accepted when it passes the floor and lies within ``track_max_jump_m`` of the prediction, else the prediction is
returned with status ``predicted``.

**Guards** (m2f:guard; verifier B1, M1, M2, M9): the server loads its types, M1's rest-up axes, the workcell and the
Panda through ``m2_library.LibraryReader`` (the policy-v2 library: root poses identity, no per-task field, every file
sha-checked); its audit hook refuses every file READ outside the M2 library, Python's own installation (``sys.prefix``
and ``base_prefix``), the worktree's Python sources, the numba cache and the temp folder, every write outside its own
log folder (``<run>/server/``, never the run folder with the workers' records) and the scratch folder, any listing of
a folder outside those, any subprocess and any non-AF_UNIX socket; after start-up it hashes every repo module it has
loaded (``m2_integrity``) and, in a frozen run, refuses to serve when one is missing from the freeze or differs; and
it answers a connection only after a ``hello`` whose protocol, venv versions, shared code sha256, library manifest
sha256 and frozen config sha256 match its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from rowant_lab.libero_env import m2_bridge as BR
from rowant_lab.libero_env import m2_integrity as INT
from rowant_lab.libero_env import m2_kin as KIN
from rowant_lab.libero_env import m2_library as LB
from rowant_lab.libero_env import m2_paths as MP
from rowant_lab.libero_env import m2_segment as SEG

TRUTH_FREE = True
FORMAT = "rowant libero_env M2 perception server v1"
# ``self_px_probe``: the bins of the model depth minus the frame depth, over the pixels the picture alone takes
SELF_DZ_EDGES = (-0.40, -0.20, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.20, 0.40)
SELF_RECORD = ("robot_px", "outside_px", "picture_px", "front_px", "picture_dz")  # the filter's record in a reply

PARAMS = {
    "k_max": 2,
    "ws_radius_m": 1.1,
    "self_filter_m": 0.01,
    "robot_spacing_m": 0.004,
    "agree_m": 0.01,
    "see_through_m": 0.015,
    "facing_cos": 0.15,
    "accept_inlier": 0.8,
    "accept_min_agree": 30,
    "dup_frac": 0.5,
    "dup_overlap": 0.5,
    "accept_inlier_region": 0.85,
    "upright_tol_deg": 30.0,
    "upright_min_diam_m": 0.18,
    "upright_nodes": 1500,
    "upright_yaw_deg": 10.0,
    "upright_xy_m": 0.08,
    "upright_xy_step_m": 0.02,
    "joint_step_m": 0.002,
    "track_gate_m": 0.03,
    "track_dilate_px": 3,
    "track_max_jump_m": 0.05,
    "seg": {"view_f": 1236.0, "view_dist_m": 1.1},
    "layer_overrides": {"px_ref_f": 1390.0},
    # mminus_dev: "plus" = M2 as frozen; "minus" = CHECK and scorer A swapped (m2_mminus)
    "monty_arm": "plus",
    "mminus": {"tau_m": 0.010, "cos_min": 0.5, "weight": 0.5},
    # amendment A14 (m2f:perceive-r1): m2_segment's support refit, support check, fitted height offset and model
    # search; False reproduces the pre-A14 server exactly (the fast layer's switch)
    "m2_segment": True,
    "support": dict(SEG.PARAMS),
    # per-suite routing of the model search: ``search`` types are no longer posed from masks (``texture_first``
    # types keep their texture masks and are searched too); ``fallback`` types are searched only when no percept of
    # the type was accepted from its masks
    "routes": {"libero_spatial": {"search": ["plate", "glazed_rim_porcelain_ramekin", "flat_stove", "wooden_cabinet",
                                             "cookies"], "texture_first": ["cookies"], "fallback": []},
               "libero_object": {"search": ["basket"], "texture_first": [], "fallback": []}},
    # m2f2:readiness-r1 (evaluator only; changes no percept): with True, the reply to an episode's FIRST look (look
    # 0) also carries ``eval_masks``, the segmentation mask of every kept mask-proposed percept as C-order run
    # lengths; the runner's audited client takes it off the reply before the tool sees it, and the evaluator scores
    # it against true silhouettes after the episode (segmentation IoU). False: the reply is exactly as before
    "eval_masks": False,
    # m2r3:fix-r1 (A63, record only): with True every mask candidate also carries ``est_diag``, the fast layer's choice
    # among its refined bases and flips (labels, CHECK and appearance scores, each one's rotation from the chosen
    # pose); no truth, no change to any pose or decision
    "diag_est": False,
    # m2r3:fix-r1 (A64): the TURN refinement of a type whose onboarded flip family samples a turn about a rotational
    # axis (``turn_refine``); False reproduces the A50 server exactly
    "turn_refine": True,  # A73: on (the private set)
    "turn_margin_deg": 1.0,
    "turn_coarse_deg": 1.0,
    "turn_fine_deg": 0.25,
    # A75: only a type whose family's turn step EXCEEDS this is refined; a type quantised at or below it is
    # axisymmetric for the evaluator too (A43: the plate, the bowl, the ramekin), its turn is not observable
    "turn_min_step_deg": 10.0,
    # m2r3:fix-r1 (A66): the depth-COMPLETED mask of a texture proposal whose pose fits weakly (``mask_complete``);
    # False reproduces the A50 server exactly
    "mask_complete": True,  # A73: on
    "complete_below_inlier": 0.9,
    "complete_reach_frac": 0.5,
    "complete_min_h_m": 0.003,
    # m2r3:fix-r1 (A67): the 2D duplicate rule asks the explained-depth question too (``_dedup``); False reproduces
    # the A50 server exactly
    "dedup_depth": True,  # A73: on
    # Pro2.0 colour line (2026-11-10): two parts of one shape at one spot fit the depth almost equally, so "best fit first"
    # is a coin toss on which label survives. [tie_frac, px_ratio]: when the fits are within tie_frac, the candidate whose
    # colour mask has at least px_ratio times the pixels goes first (``_label_tie_order``). None: the frozen behaviour
    "dedup_label_tie": None,
    # the Pro2.0 self-writing loop (2026-11-08; reply fields only, no percept changes): with True the reply to a full
    # look also carries ``extra`` (``_report_extra``). False: the reply is exactly as before
    "report_extra": False,
    "extra_rejected_per_type": 2,
    "extra_max_rejected": 16,
    "extra_max_blobs": 8,
    "extra_blob_min_px": 200,
    "extra_blob_h_m": (0.01, 0.6),
    # 2026-11-16 (reply fields only; the dead-types design, fix 2): each unexplained region also carries ``top``, the
    # box of its pixels within this many metres of its top (the thing that stands on whatever is under it: a plate on
    # a block, a basket's rim). None: the reply is exactly as before
    "extra_blob_top_m": None,
    "extra_blob_top_min_px": 30,
    # 2026-11-16 night (reply field only): ``extra.on``, what stands ON a reported object (``_on_top``): inside the
    # object's outline, the pixels this many metres or more above the highest point of its model. The outline is cut
    # out of the unexplained regions, so a thing lying on a reported object was reported nowhere. None: the reply is
    # exactly as before
    "extra_on_top_m": None,
}
SCORER = "rowant_lab.fastlayer.scorers.monty_colour:MontyColourTuned"
RULE = {"name": "weighted", "margin": 0.0, "weight": 1.0}
# amendment A39: OpenCV's RNG state for every texture-matcher build (0 = a fresh OpenCV thread's state), and the salt
# of the per-request RNG states
MATCHER_RNG_SEED = 0
LOOK_RNG_SALT = "rowant-m2-look-rng-v1"


def blob_top(W, top_m: float, min_px: int = 30) -> dict | None:
    """The top layer of an unexplained region (``extra_blob_top_m``): its world points within ``top_m`` of the
    region's top (the 95th percentile of height, so a few stray pixels do not set it). Returns the layer's box across
    (centre ``xy``, ``half_xy`` and ``yaw`` of its principal axes, extents taken between the 2nd and 98th
    percentiles), its level ``z`` and ``n_px``; None when the layer has under ``min_px`` points."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or len(W) < max(3, int(min_px)):
        return None
    z95 = float(np.percentile(W[:, 2], 95))
    sel = W[:, 2] >= z95 - float(top_m)
    if int(sel.sum()) < max(3, int(min_px)):
        return None
    P = W[sel, :2]
    ctr = P.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((P - ctr).T))
    ax = evecs[:, int(np.argmax(evals))]
    ax = ax if ax[0] >= 0 else -ax  # the sign of a principal axis is arbitrary
    ay = np.array([-ax[1], ax[0]])
    s, t_ = (P - ctr) @ ax, (P - ctr) @ ay
    s0, s1 = np.percentile(s, [2, 98])
    t0, t1 = np.percentile(t_, [2, 98])
    mid = ctr + ax * 0.5 * float(s0 + s1) + ay * 0.5 * float(t0 + t1)
    return {"xy": [round(float(x), 4) for x in mid],
            "half_xy": [round(0.5 * float(s1 - s0), 4), round(0.5 * float(t1 - t0), 4)],
            "yaw": round(float(math.atan2(ax[1], ax[0])), 3), "z": round(z95, 4), "n_px": int(sel.sum())}


def build_matcher(seg_lib) -> dict:
    """Build ``seg_lib``'s texture matcher NOW from OpenCV's global RNG at ``MATCHER_RNG_SEED`` (amendment A39), and
    leave that RNG at the same fixed state. Refuses a library whose matcher was already built (its RNG state would be
    unknown). Returns the record the hello reports."""
    import cv2

    if getattr(seg_lib, "_matcher", None) is not None:
        raise RuntimeError("the texture matcher was built before build_matcher: its RNG state is unknown")
    t0 = time.perf_counter()
    cv2.setRNGSeed(MATCHER_RNG_SEED)
    fl, oid, _xyz = seg_lib.matcher()
    cv2.setRNGSeed(MATCHER_RNG_SEED)
    if seg_lib.matcher()[0] is not fl:
        raise RuntimeError("the texture matcher is not cached: a later look would rebuild it")
    return {"rng_seed": MATCHER_RNG_SEED, "n_descriptors": int(len(oid)),
            "build_s": round(time.perf_counter() - t0, 3)}


def reseed_request(episode, look) -> int:
    """Set OpenCV's, numpy's and Python's global RNGs from this request's episode token and look index (amendment
    A39), so a draw from any of them during the request depends on the request only. Returns OpenCV's seed."""
    import random

    import cv2

    word = int.from_bytes(hashlib.sha256(f"{LOOK_RNG_SALT}|{episode}|{int(look)}".encode()).digest()[:8], "big")
    seed = int(word % 2147483647)
    cv2.setRNGSeed(seed)
    np.random.seed(int(word % (2 ** 32)))
    random.seed(word)
    return seed


def est_summary(est, k: int = 12) -> dict:
    """m2r3:fix-r1 (A63, record only): the fast layer's choice for one mask: its refined bases and flip candidates
    (label, mode, CHECK per patch, appearance score, the rotation and the shift from the CHOSEN pose), the best ``k``
    by CHECK + appearance. Server-side numbers only (no truth)."""
    d = est.diag or {}
    ref = d.get("refined") or []
    fl = (d.get("_raw") or {}).get("flips") or {}
    cands = fl.get("cands") or []
    R0, t0 = np.asarray(est.R), np.asarray(est.t)
    rows = []
    for j, rr in enumerate(ref):
        e = {"j": j, "label": rr.get("label"), "mode": rr.get("mode"),
             "check": None if rr.get("check") is None else round(float(rr["check"]), 4),
             "app": None if rr.get("app") is None else round(float(rr["app"]), 4)}
        if j < len(cands):
            e["rot_from_chosen_deg"] = round(rot_err_deg(R0, cands[j]["R"]), 2)
            e["shift_from_chosen_mm"] = round(1e3 * float(np.linalg.norm(np.asarray(cands[j]["t"]) - t0)), 2)
        rows.append(e)
    rows.sort(key=lambda e: -((e["check"] or 0.0) + (e["app"] or 0.0)))
    return {"n_hyp": d.get("n_hyp"), "pick": d.get("pick"), "b_choice": d.get("b_choice"),
            "b0_choice": d.get("b0_choice"), "n_refined": len(ref), "top": rows[:k],
            "flips": {kk: (d.get("flips") or {}).get(kk) for kk in ("n_bases", "n_starts", "n_icp", "n_derived",
                                                                     "n_candidates")}}


def mask_rle(m) -> np.ndarray:
    """Run lengths (int32) of a boolean mask in C order, starting with a False run (m2f2:readiness-r1)."""
    f = np.ascontiguousarray(np.asarray(m, dtype=bool)).ravel()
    if f.size == 0:
        return np.zeros(1, np.int32)
    cut = np.flatnonzero(f[1:] != f[:-1]) + 1
    runs = np.diff(np.concatenate([[0], cut, [f.size]]))
    if f[0]:
        runs = np.concatenate([[0], runs])
    return runs.astype(np.int32)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rot_err_deg(R1, R2) -> float:
    c = (np.trace(np.asarray(R1).T @ np.asarray(R2)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


class Workcell:
    """Everything of one suite's library the server needs (loaded once)."""

    def __init__(self, suite: str, params: dict, reader: "LB.LibraryReader"):
        from rowant_lab.fastlayer import layer as LY
        from rowant_lab.fastlayer import memory as ML
        from rowant_lab.fastlayer import segment as SG
        from rowant_lab.fastlayer import symmetry as SY
        from rowant_lab.fastlayer.appearance import DecideRule, make
        from rowant_lab.libero_env import m2_onboard as OBM

        t0 = time.perf_counter()
        self.suite = suite
        onboard_dir = LB.lfb_dirs(params.get("lfb_library"))[0]  # LFB-LIB: a variant tree, else M2's own
        root = onboard_dir / suite
        self.ob = json.loads((root / "onboarding.json").read_text())
        files = self.ob["files"]
        self.mem = ML.load(root / "memory.npz", kind="mesh", expect_sha256=files["memory.npz"])
        self.sym = SY.SymmetryTable.load(root / "symmetry.json", files["symmetry.json"])
        over = dict(params["layer_overrides"])
        over["speed"] = {"wide": {"grid_dir": str(root / "grids")}}
        preset = LY.make_preset("hope_fast", over)
        self.arm = params.get("monty_arm", "plus")
        if self.arm == "plus":
            scorer = make(SCORER, colour_npz=str(root / "colour-memory.npz"), colour_sha256=files["colour-memory.npz"])
            self.layer = LY.FastLayer(self.mem, preset, symmetry=self.sym, scorer=scorer, rule=DecideRule(**RULE))
        elif self.arm == "minus":
            from rowant_lab.libero_env import m2_mminus as MM

            mm = {**MM.DEFAULTS, **(params.get("mminus") or {})}
            MM.ONBOARD_ROOT = onboard_dir  # LFB-LIB: render-and-compare reads this tree's PLYs
            self.layer = MM.MinusLayer(self.mem, preset, symmetry=self.sym, scorer=MM.RenderCompareVC(),
                                       rule=DecideRule(**{**RULE, "weight": float(mm["weight"])}),
                                       tau_m=mm["tau_m"], cos_min=mm["cos_min"])
        else:
            raise ValueError(f"unknown monty_arm {self.arm!r}")
        self.seg_params = dict(params["seg"])
        self.seg_lib = SG.Library.build(str(root / "memory.npz"), str(root / "colour-memory.npz"),
                                        memory_sha256=files["memory.npz"], colour_sha256=files["colour-memory.npz"],
                                        params=self.seg_params, cache_dir=str(root / "templates"))
        # amendment A39: the texture matcher is built HERE, eagerly, from OpenCV's RNG at a fixed state
        self.matcher = build_matcher(self.seg_lib)
        self.gid = {tuple(v): int(k) for k, v in self.ob["graph_ids"].items()}  # (type, part) -> id
        self.part_of_gid = {v: k for k, v in self.gid.items()}
        # the policy-v2 library (amendment A9): root poses identity, no per-task field, every file sha-checked
        self.types = {t: reader.export_type(t) for t in reader.suite_types(suite)}
        self.workcell = reader.workcell(suite)
        self.base = np.asarray(self.workcell["robot_base"]["pos"], float)
        self.nodes = {}
        for g in self.mem.ids:
            from rowant_lab.fastlayer import onboard as OB

            x = self.mem.x[g]
            pv = self.mem.feature_mapping["pose_vectors"][0]
            self.nodes[g] = (np.asarray(self.mem.pos[g]) - OB.P_TRAIN, np.asarray(x[:, pv:pv + 3]))
        self.diam = {g: float(np.linalg.norm(n[0].max(0) - n[0].min(0))) for g, n in self.nodes.items()}
        # the body-frame axis that is world up in the rest pose (M1's frozen types; the parts inherit the root's)
        m1t = {t: reader.m1_type(t) for t in reader.suite_types(suite)}
        self.rest_up = {str(g): np.asarray(m1t[t]["rest_up_in_body"], float) for (t, pa), g in self.gid.items()}
        self.panda = reader.panda(suite)
        self.robot_pts = self.panda.surface_samples(params["robot_spacing_m"])
        self.rigid_gids = sorted(g for (t, pa), g in self.gid.items()
                                 if self.types[t]["parts"][pa]["joint"] is None)
        # m2_segment: the onboarded flip family (yaw period) and the type's kind per part
        self.family = {str(g): [np.eye(3)] + ([np.asarray(e["R"], float) for e in self.sym.get(g)["family"]]
                                              if g in self.sym else []) for g in self.nodes}
        self.kind = {str(g): self.types[t]["kind"] for (t, pa), g in self.gid.items()}
        self._colour_npz = (root / "colour-memory.npz", files["colour-memory.npz"])
        self._colour = None
        self.setup_s = round(time.perf_counter() - t0, 3)
        _ = OBM  # the onboarding module's ids rule is the one recorded in onboarding.json

    def variants(self, g: str) -> list:
        """Whole-assembly node sets of a jointed type's root part ``g`` in the root's body frame (amendment A21):
        all jointed parts closed (qpos0), and each jointed part alone at the open end of its range (the end farther
        from qpos0). [] for a type without jointed parts."""
        if not hasattr(self, "_variants"):
            self._variants = {}
        if g in self._variants:
            return self._variants[g]
        typ, pa = self.part_of_gid[int(g)]
        tdoc = self.types[typ]
        out = []
        jparts = [jp for jp, P in tdoc["parts"].items() if P["joint"] is not None]
        if pa == tdoc["root_body"] and jparts:
            def assembly(q):
                bp = KIN.part_body_poses(tdoc, {tdoc["root_body"]: (np.eye(3), np.zeros(3))}, q)
                pts = [self.nodes[str(g)][0]]
                for jp in jparts:
                    R, t = bp[jp]
                    pts.append(self.nodes[str(self.gid[(typ, jp)])][0] @ R.T + t)
                return np.concatenate(pts)[::2]
            q0 = {jp: float(tdoc["parts"][jp]["joint"].get("qpos0", 0.0)) for jp in jparts}
            out.append(("closed", assembly(q0)))
            for jp in jparts:
                lo, hi = (float(x) for x in tdoc["parts"][jp]["joint"]["range"])
                q_open = lo if abs(lo - q0[jp]) >= abs(hi - q0[jp]) else hi
                out.append((f"{jp}_open", assembly({**q0, jp: q_open})))
        self._variants[g] = out
        return out

    def variants_n(self, g: str) -> list:
        """As ``variants``, with the nodes' normals: [(name, nodes, normals)] in the root's body frame (amendment A25:
        the wide refine stage of a jointed fixture scores these assemblies)."""
        if not hasattr(self, "_variants_n"):
            self._variants_n = {}
        if g in self._variants_n:
            return self._variants_n[g]
        typ, pa = self.part_of_gid[int(g)]
        tdoc = self.types[typ]
        out = []
        jparts = [jp for jp, P in tdoc["parts"].items() if P["joint"] is not None]
        if pa == tdoc["root_body"] and jparts:
            def assembly(q):
                bp = KIN.part_body_poses(tdoc, {tdoc["root_body"]: (np.eye(3), np.zeros(3))}, q)
                pts, nrs = [self.nodes[str(g)][0]], [self.nodes[str(g)][1]]
                for jp in jparts:
                    R, t = bp[jp]
                    n_, r_ = self.nodes[str(self.gid[(typ, jp)])]
                    pts.append(n_ @ R.T + t)
                    nrs.append(r_ @ R.T)
                return np.concatenate(pts)[::2], np.concatenate(nrs)[::2]
            q0 = {jp: float(tdoc["parts"][jp]["joint"].get("qpos0", 0.0)) for jp in jparts}
            out.append(("closed", *assembly(q0)))
            for jp in jparts:
                lo, hi = (float(x) for x in tdoc["parts"][jp]["joint"]["range"])
                q_open = lo if abs(lo - q0[jp]) >= abs(hi - q0[jp]) else hi
                out.append((f"{jp}_open", *assembly({**q0, jp: q_open})))
        self._variants_n[g] = out
        return out

    def colour(self, g: str):
        """The onboarded node colours (blur 2) of part ``g`` (the colour memory, sha-checked once)."""
        if self._colour is None:
            path, sha = self._colour_npz
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != sha:
                raise RuntimeError("colour memory sha256 differs from the onboarding record")
            import io

            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                self._colour = {k.split("__")[0]: np.asarray(z[k]) for k in z.files if k.endswith("__rgb_2")}
        return self._colour.get(str(g))


class HelloRefused(RuntimeError):
    """A client whose protocol, versions or sha256 differ from the server's (plan venv_bridge.requests.hello)."""


def own_versions() -> dict:
    return {"python": sys.version.split()[0], "numpy": np.__version__}


class Perceiver:
    def __init__(self, params: dict | None = None, policy_lib_sha256: str | None = None,
                 frozen_config_sha256: str | None = None):
        self.p = {**PARAMS, **(params or {})}
        self.cells: dict = {}
        self.n_looks = 0  # a counter for the log only: every seeded draw is keyed by the REQUEST's episode token and
        # look index (``m2_agent`` sends them), so two workers on one server cannot change each other's seeds
        _ob, pl_, m1_ = LB.lfb_dirs(self.p.get("lfb_library"))  # LFB-LIB
        # extra_suites: the Pro2.0 line's Goal (rowant_lab/pro2/goal_paths.py, through dev_run); absent in M2's runs
        self.reader = LB.LibraryReader(pl_, m1_, allow_panda=True, expect_manifest_sha256=policy_lib_sha256,
                                       extra_suites=self.p.get("extra_suites"))
        self.frozen_config_sha256 = frozen_config_sha256
        self.integrity: dict | None = None  # the start-up module record (main sets it)
        self.hellos: list = []
        # in-episode learning (lfb_episode; off unless params carry learn_in_episode): per episode token, the
        # leftover and look plan, then the episode's own memory and settings (dropped at the next begin_episode)
        self.learn_state: dict = {}
        self.ep_scope: dict = {}

    def cell(self, suite) -> Workcell:
        if suite not in self.cells:
            self.cells[suite] = Workcell(suite, self.p, self.reader)
        return self.cells[suite]

    def shared_code_sha256(self) -> dict:
        mods = (self.integrity or {}).get("modules") or INT.loaded_repo_modules()
        return {f: mods.get(f) for f in BR.SHARED_CODE}

    def library_sha256(self) -> dict:
        return {f"{MP.POLICY_LIB_VERSION}/manifest.json": self.reader.manifest_sha256}

    def check_hello(self, msg: dict) -> dict:
        """Refuse a client that does not match this server (``HelloRefused``); return the server's own record."""
        bad = []
        if msg.get("protocol") != BR.PROTOCOL:
            bad.append(f"protocol {msg.get('protocol')!r} != {BR.PROTOCOL!r}")
        if own_versions() != BR.EXPECTED_VERSIONS["monty"]:
            bad.append(f"server versions {own_versions()} != {BR.EXPECTED_VERSIONS['monty']}")
        if (msg.get("client") or {}) != BR.EXPECTED_VERSIONS["venv-libero"]:
            bad.append(f"client versions {msg.get('client')} != {BR.EXPECTED_VERSIONS['venv-libero']}")
        mine = self.shared_code_sha256()
        theirs = msg.get("code_sha256") or {}
        for f in BR.SHARED_CODE:
            if not mine.get(f) or theirs.get(f) != mine.get(f):
                bad.append(f"code {f} differs")
        if (msg.get("library_sha256") or {}) != self.library_sha256():
            bad.append("library manifest sha256 differs")
        if msg.get("frozen_config_sha256") != self.frozen_config_sha256:
            bad.append(f"frozen config {msg.get('frozen_config_sha256')} != {self.frozen_config_sha256}")
        if not set(msg.get("suites") or []) <= self.reader.suites():  # M2's SUITES, plus extra_suites if given
            bad.append(f"suites {msg.get('suites')}")
        self.hellos.append({"ok": not bad, "refusals": bad})
        if bad:
            raise HelloRefused("; ".join(bad))
        return {"protocol": BR.PROTOCOL, "versions": own_versions(), "code_sha256": mine,
                "library_sha256": self.library_sha256(), "frozen_config_sha256": self.frozen_config_sha256}

    # ------------------------------------------------------------------ geometry helpers
    @staticmethod
    def _frame_arrays(fr: dict):
        depth = np.asarray(fr["depth_u16"])
        z = depth.astype(np.float64) * float(fr["depth_scale_mm"]) * 1e-3
        z[depth == 0] = np.nan
        K = np.asarray(fr["K"], float)
        H, W = depth.shape
        v, u = np.mgrid[0:H, 0:W].astype(np.float64)
        Pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
        T = np.asarray(fr["T_world_from_camera"], float)
        Pw = Pc @ T[:3, :3].T + T[:3, 3]
        return depth, z, K, T, Pc, Pw

    def _full_self_px(self, msg: dict):
        """How far a full look grows the robot's own pixels before it leaves them out of the depth (``_mask_depth``):
        the look's own key ``self_px`` (the looks for the hand's way send it), else the server's ``full_self_px``
        (2026-11-18, absent unless given: every full look then leaves the robot out in the PICTURE, the first look
        and each look of a plan made again). Neither given: None, and the look is taken as it always was."""
        return msg.get("self_px") or self.p.get("full_self_px")

    def _mask_depth(self, wc: Workcell, fr: dict, joint_pos, gripper_qpos, self_px=None) -> tuple[np.ndarray, dict]:
        from scipy.spatial import cKDTree

        depth, z, K, T, Pc, Pw = self._frame_arrays(fr)
        valid = depth > 0
        d_xy = np.linalg.norm(Pw[..., :2] - wc.base[:2], axis=-1)
        ws = valid & (d_xy <= self.p["ws_radius_m"]) & (Pw[..., 2] > -0.05)
        poses = wc.panda.fk(joint_pos, gripper_qpos)
        pts = np.concatenate([wc.robot_pts[b] @ poses[b][0].T + poses[b][1] for b in wc.robot_pts])
        tree = cKDTree(pts)
        idx = np.flatnonzero(ws.ravel())
        dd, _ = tree.query(Pw.reshape(-1, 3)[idx], k=1, distance_upper_bound=self.p["self_filter_m"])
        robot = np.zeros(depth.size, bool)
        robot[idx[np.isfinite(dd)]] = True
        robot = robot.reshape(depth.shape)
        more = {}
        if self_px:
            # 2026-11-18 (stream PATH; a full look's own key ``self_px``, sent by the robot's looks for the hand's
            # way alone, absent from every other look): the robot found in the PICTURE too, as the learner's first
            # look finds it. A depth that is off by more than ``self_filter_m`` on the arm (the colour line's net)
            # leaves the arm standing in the scene as a thing without a name.
            from rowant_lab.libero_env.lfb_episode import self_depth, self_pixels

            front_m, probe = self.p.get("self_front_m"), self.p.get("self_px_probe")
            if front_m is None and not probe:
                own = self_pixels(wc, fr, joint_pos, gripper_qpos, int(self_px)) & ws
            else:
                # 2026-11-18, the server's ``self_front_m`` (absent unless given; it acts on every look that leaves
                # the robot out in the picture, the learner's first look included): a pixel the model covers stays
                # in the look where the frame's depth is nearer than the model by more than this. The model is
                # drawn whole, hidden parts included, and took the table and the things in front of the robot with
                # it (a paired pod run). ``self_px_probe`` (record only): how far in front of the model, or
                # behind it, the covered pixels read that the filter in space does not take.
                zg, z0 = self_depth(wc, fr, joint_pos, gripper_qpos, int(self_px))
                drawn = np.isfinite(zg) & ws
                with np.errstate(invalid="ignore"):
                    front = drawn & (z < zg - float(front_m)) if front_m is not None else np.zeros_like(drawn)
                    if probe:
                        dz, pic = zg - z, drawn & ~robot  # (positive: the frame sees something NEARER than the model)
                        more["picture_dz"] = {"edges_m": list(SELF_DZ_EDGES), **{k: np.histogram(
                            dz[pic & m_], bins=[-np.inf, *SELF_DZ_EDGES, np.inf])[0].tolist()
                            for k, m_ in (("model", np.isfinite(z0)), ("rim", ~np.isfinite(z0)))}}
                own = drawn & ~front
                if front_m is not None:
                    more["front_px"] = int((front & ~robot).sum())  # record only: covered, and left in the look
            more["picture_px"] = int((own & ~robot).sum())  # record only: found by the picture, not by the filter
            robot = robot | own
        keep = ws & ~robot
        d2 = np.where(keep, depth, 0).astype(np.uint16)
        return d2, {"robot_px": int(robot.sum()), "outside_px": int((valid & ~ws).sum()), "T": T, "K": K, **more}

    def _visible_nodes(self, wc: Workcell, g: str, R, t):
        nodes, nrm = wc.nodes[g]
        X = nodes @ R.T + t
        N = nrm @ R.T
        view = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        vis = (-(N * view).sum(1)) > self.p["facing_cos"]
        return X[vis]

    def self_occluded(self, wc: Workcell, g: str, R, t, z_obs: np.ndarray, K: np.ndarray) -> dict:
        """Amendment A44: of a JOINTED fixture's visible root nodes whose pixel sees something NEARER (occluded), how
        many are occluded by the fixture's own jointed parts (drawers) at their fitted values: the observed depth there
        agrees within ``agree_m`` with those parts' surface (their nodes splatted over ``self_occ_splat_px``). These
        nodes are hidden by the fixture itself, so A44's visibility share leaves them out of its denominator."""
        from scipy.ndimage import grey_erosion

        typ, pa = wc.part_of_gid[int(g)]
        tdoc = wc.types[typ]
        jparts = [jp for jp, P_ in tdoc["parts"].items() if P_["joint"] is not None]
        if pa != tdoc["root_body"] or not jparts:
            return {"n_selfocc": 0, "joints": {}}
        segp = {**SEG.PARAMS, **(self.p.get("support") or {})}
        R, t = np.asarray(R, float), np.asarray(t, float)
        q = {jp: self._fit_joint_fast(wc, tdoc, typ, jp, (R, t), z_obs, K) for jp in jparts}
        bp = KIN.part_body_poses(tdoc, {tdoc["root_body"]: (R, t)}, q)
        H, W = z_obs.shape
        Dp = np.full((H, W), np.inf)
        for jp in jparts:
            Rj, tj = bp[jp]
            Xp = wc.nodes[str(wc.gid[(typ, jp)])][0] @ np.asarray(Rj).T + np.asarray(tj)
            Xp = Xp[Xp[:, 2] > 0.05]
            u = np.rint(K[0, 0] * Xp[:, 0] / Xp[:, 2] + K[0, 2] - 0.5).astype(int)
            v = np.rint(K[1, 1] * Xp[:, 1] / Xp[:, 2] + K[1, 2] - 0.5).astype(int)
            ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            np.minimum.at(Dp, (v[ins], u[ins]), Xp[ins, 2])
        r = int(segp.get("self_occ_splat_px", 3))
        Dp = grey_erosion(Dp, size=2 * r + 1)  # spread each node over its neighbourhood (min over inf background)
        X = self._visible_nodes(wc, g, R, t)
        X = X[X[:, 2] > 0.05]
        u = np.rint(K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2] - 0.5).astype(int)
        v = np.rint(K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2] - 0.5).astype(int)
        ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, X = u[ins], v[ins], X[ins]
        zo = z_obs[v, u]
        has = np.isfinite(zo)
        ag = float(self.p["agree_m"])
        occ = has & ((zo - X[:, 2]) < -ag)
        own = occ & np.isfinite(Dp[v, u]) & (np.abs(zo - Dp[v, u]) <= ag)
        return {"n_selfocc": int(own.sum()), "n_occluded": int(occ.sum()),
                "joints": {k_: round(float(v_), 5) for k_, v_ in q.items()}}

    def turn_refine(self, wc: Workcell, g: str, est, frame) -> dict | None:
        """m2r3:fix-r1 (A64): refine the TURN of a type whose onboarded flip family samples a turn about a rotational
        axis. The symmetry table gives the axis, the centre and the step s (360 / n_turns); the fast layer's derived
        turns are s apart and are not refined, so its chosen turn is off by up to s / 2 (alphabet_soup and
        tomato_sauce 15 deg, salad_dressing 22.5 deg, the bowl, ramekin and plate 10 deg). The chosen pose is turned
        about that axis through the symmetry centre on a grid of ``turn_coarse_deg`` over +-(s / 2 +
        ``turn_margin_deg``), then of ``turn_fine_deg`` over +-``turn_coarse_deg`` around the best, and every turn is
        scored by the fast layer's OWN decision rule on the same mask (CHECK per patch + the weighted rule's weight x
        appearance scorer A); the best key wins, ties to the smaller turn. A type-level rule (no object, scene or pose
        constant). Returns {R, t, delta_deg, key, key0} or None for a type without such a family."""
        from rowant_lab.fastlayer import appearance as AP
        from rowant_lab.fastlayer import symmetry as SY

        if g not in wc.sym:
            return None
        sym = wc.sym.get(g)
        if sym.get("rot_axis") is None or not sym.get("turn_step_deg"):
            return None
        if float(sym["turn_step_deg"]) <= float(self.p.get("turn_min_step_deg", 0.0)) + 1e-6:
            return None  # A75: an axisymmetric type (A43's class): its turn is not perceived
        raw = (est.diag or {}).get("_raw") or {}
        ps, px = raw.get("ps"), raw.get("pixels")
        if ps is None or px is None or not ps.n:
            return None
        axis = np.asarray(sym["axes"], float)[:, int(sym["rot_axis"])]
        cen = sym["centre"]
        R0, t0 = np.asarray(est.R, float), np.asarray(est.t, float)
        w = float(wc.layer.rule.weight)
        n = float(ps.n)
        assets = wc.layer.assets.get(g)

        def keys(deltas):
            poses = [SY.flip_pose(R0, t0, SY.axis_rotation(axis, float(d_)), cen) for d_ in deltas]
            geo = wc.layer._check([(g, R_, t_) for R_, t_ in poses], ps)
            ac = [AP.Candidate(obj=g, R=np.asarray(R_), t=np.asarray(t_), label="turn", base=0,
                               geo=float(geo[i]) / n, info={}) for i, (R_, t_) in enumerate(poses)]
            app = AP.score_all(wc.layer.scorer, ac, frame, px, assets)
            return poses, [float(geo[i]) / n + w * (0.0 if app[i] is None else float(app[i])) for i in range(len(poses))]

        half = 0.5 * float(sym["turn_step_deg"]) + float(self.p["turn_margin_deg"])
        st = float(self.p["turn_coarse_deg"])
        grid = np.round(np.arange(-half, half + 1e-9, st), 6)
        poses, kk = keys(grid)
        i0 = int(np.argmin(np.abs(grid)))
        b = max(range(len(grid)), key=lambda i: (kk[i], -abs(grid[i])))
        fine = np.round(grid[b] + np.arange(-st, st + 1e-9, float(self.p["turn_fine_deg"])), 6)
        poses2, kk2 = keys(fine)
        b2 = max(range(len(fine)), key=lambda i: (kk2[i], -abs(fine[i])))
        R_, t_ = poses2[b2]
        return {"R": np.asarray(R_), "t": np.asarray(t_), "delta_deg": float(fine[b2]), "key": round(kk2[b2], 5),
                "key0": round(kk[i0], 5), "step_deg": float(sym["turn_step_deg"]), "n_eval": len(grid) + len(fine)}

    def completed_mask(self, wc: Workcell, g: str, px: np.ndarray, hpx: np.ndarray, K: np.ndarray,
                       z_obs: np.ndarray):
        """m2r3:fix-r1 (A66): the depth-completed mask of a texture proposal. A texture mask is the silhouette of the
        recognised TEMPLATE at its RANSAC pose, so when that pose is off the mask cuts the object and takes in
        background, and the pose found on it inherits the error (cream_cheese on the private set: a 21.6 deg miss
        accepted at inlier 0.807, and a shifted look rejected at 0.58, while region masks of the same object fit at
        0.96). The completion: the pixels standing above the support (height above ``complete_min_h_m``, the robot
        and the out-of-workspace pixels already have no depth) within ``complete_reach_frac`` x the part's diameter
        (in pixels at the mask's median depth) of the mask, in the connected components that the mask's own
        above-support pixels touch. Type-level geometry only. None when the mask has too few readings."""
        from scipy.ndimage import binary_dilation, label

        H, W = px.shape
        zs = z_obs[px & np.isfinite(z_obs)]
        if zs.size < int(self.p["accept_min_agree"]):
            return None
        zc = float(np.median(zs))
        r = max(1, int(math.ceil(float(self.p["complete_reach_frac"]) * wc.diam[g] * float(K[0, 0]) / max(zc, 0.05))))
        rows, cols = np.nonzero(px)
        r0, r1 = max(int(rows.min()) - r, 0), min(int(rows.max()) + r + 1, H)
        c0, c1 = max(int(cols.min()) - r, 0), min(int(cols.max()) + r + 1, W)
        sp = px[r0:r1, c0:c1]
        sh = hpx[r0:r1, c0:c1]
        fg = np.isfinite(sh) & (sh > float(self.p["complete_min_h_m"]))
        zone = binary_dilation(sp, structure=np.ones((3, 3), bool), iterations=r)
        lab, n = label(fg & zone)
        keep = np.unique(lab[sp & fg])
        keep = keep[keep > 0]
        if not len(keep):
            return None
        out = np.zeros((H, W), bool)
        out[r0:r1, c0:c1] = np.isin(lab, keep)
        return out

    def fit_check(self, wc: Workcell, g: str, R, t, z_obs: np.ndarray, K: np.ndarray) -> dict:
        """Node agreement of part ``g`` at camera-frame pose (R, t) with the observed depth (metres, NaN = none)."""
        X = self._visible_nodes(wc, g, R, t)
        H, W = z_obs.shape
        ok = X[:, 2] > 0.05
        X = X[ok]
        u = np.rint(K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2] - 0.5).astype(int)
        v = np.rint(K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2] - 0.5).astype(int)
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, X = u[inside], v[inside], X[inside]
        zo = z_obs[v, u]
        has = np.isfinite(zo)
        dz = zo - X[:, 2]
        agree = has & (np.abs(dz) <= self.p["agree_m"])
        through = has & (dz > self.p["see_through_m"])
        n_a, n_t = int(agree.sum()), int(through.sum())
        return {"n_vis": int(len(X)), "n_agree": n_a, "n_through": n_t, "n_nodepth": int((~has).sum()),
                "n_occluded": int((has & (dz < -self.p["agree_m"])).sum()), "inlier": n_a / max(n_a + n_t, 1)}

    # ------------------------------------------------------------------ looks
    def full(self, msg: dict) -> dict:
        from rowant_lab.fastlayer import layer as LY
        from rowant_lab.fastlayer import segment as SG
        from rowant_lab.fastlayer.frame import Frame

        wc = self.cell(msg["suite"])
        reseed_request(msg["episode"], msg["look"])  # amendment A39: after the cell (its matcher build reseeds too)
        tm = {}
        t0 = time.perf_counter()
        fr = msg["cameras"][0]
        d2, md = self._mask_depth(wc, fr, msg["joint_pos"], msg["gripper_qpos"], self_px=self._full_self_px(msg))
        tm["self_filter_s"] = time.perf_counter() - t0
        frame = Frame(rgb=np.asarray(fr["rgb"]), depth=d2, K=md["K"], depth_scale=float(fr["depth_scale_mm"]))
        a = time.perf_counter()
        ctx = wc.layer.frame_context(frame)
        tm["plane_s"] = time.perf_counter() - a
        z_obs = ctx["z_obs"]
        plane = ctx.get("plane")
        plane_w = None
        if plane is not None:
            T = md["T"]
            n_w = T[:3, :3] @ np.asarray(plane["normal"], float)
            p0 = -float(plane["d"]) * np.asarray(plane["normal"], float)
            plane_w = {"normal_w": n_w.tolist(), "height_w": float((T[:3, :3] @ p0 + T[:3, 3])[2])}
        a = time.perf_counter()
        targets = [(g, int(self.p["k_max"])) for g in wc.rigid_gids]
        masks, sdiag = SG.segment(frame, ctx, targets, wc.seg_lib, wc.seg_params)
        tm["segment_s"] = time.perf_counter() - a
        # amendment A14: m2_segment's support refit and per-pixel height map (stage 1), and the per-suite routing
        use_seg = bool(self.p.get("m2_segment")) and plane is not None
        segp = {**SEG.PARAMS, **(self.p.get("support") or {})}
        route = (self.p.get("routes") or {}).get(msg["suite"]) or {} if use_seg else {}
        search_t = set(route.get("search") or [])
        tex_first = set(route.get("texture_first") or [])
        sup = hpx = None
        if use_seg:
            a = time.perf_counter()
            sup = SEG.refit_support(z_obs, md["K"], plane, segp)
            hpx = SEG.heights(z_obs, md["K"], sup)
            tm["support_refit_s"] = time.perf_counter() - a
        a = time.perf_counter()
        up_cam = md["T"][:3, :3].T @ np.array([0.0, 0.0, 1.0])  # world up in the camera frame (declared extrinsics)
        cands = []
        mask_of: dict = {}  # m2f2:readiness-r1: id(candidate) -> its mask index (record only; eval_masks)
        n_routed = 0
        for i, mk in enumerate(masks):
            g = str(mk.obj_id)
            typ_g = wc.part_of_gid[int(g)][0]
            src0 = (mk.diag or {}).get("source")
            if typ_g in search_t and not (typ_g in tex_first and (src0 or "").startswith("texture")):
                n_routed += 1  # a routed type is searched as a whole model on the support, never posed from a mask
                continue
            px = mk.full(frame.depth.shape)
            tok = LY.token_for((msg["episode"], msg["look"], i), [g])
            est = wc.layer.estimate_target(frame, ctx, LY.Target(obj_ids=(g,), pixels=px, seed=tok, key=i))
            rec = {"gid": g, "source": src0, "seg_score": float(mk.score),
                   "n_px": int(px.sum()),
                   "status": est.status}
            mask_of[id(rec)] = i
            if self.p.get("mask_complete") and use_seg and (src0 or "").startswith("texture"):
                # A66: a texture proposal whose pose fits weakly gets a second pose on its depth-completed mask; the
                # better fit (agreeing minus twice the seen-through nodes, the joint fit's own measure) is kept
                fc1 = self.fit_check(wc, g, est.R, est.t, z_obs, md["K"]) if est.status == "ok" else None
                if fc1 is None or fc1["inlier"] < float(self.p["complete_below_inlier"]) \
                        or fc1["n_agree"] < int(self.p["accept_min_agree"]):
                    px2 = self.completed_mask(wc, g, px, hpx, md["K"], z_obs)
                    mc = {"fit_mask": None if fc1 is None else {k_: fc1[k_] for k_ in ("n_agree", "n_through",
                                                                                      "inlier")}}
                    if px2 is not None:
                        inter = int((px2 & px).sum())
                        mc.update(n_px=int(px2.sum()), iou=round(inter / max(int((px2 | px).sum()), 1), 4))
                        tok2 = LY.token_for((msg["episode"], msg["look"], i, "completed"), [g])
                        est2 = wc.layer.estimate_target(frame, ctx, LY.Target(obj_ids=(g,), pixels=px2, seed=tok2,
                                                                              key=i))
                        mc["status"] = est2.status
                        if est2.status == "ok":
                            fc2 = self.fit_check(wc, g, est2.R, est2.t, z_obs, md["K"])
                            mc["fit_completed"] = {k_: fc2[k_] for k_ in ("n_agree", "n_through", "inlier")}

                            def js(fc_):
                                return -1e9 if fc_ is None else fc_["n_agree"] - 2.0 * fc_["n_through"]
                            if js(fc2) > js(fc1):
                                est, px = est2, px2
                                rec.update(status=est.status, n_px=int(px.sum()), source=f"{src0}+completed")
                                mc["chosen"] = "completed"
                            else:
                                mc["chosen"] = "mask"
                    rec["mask_completion"] = mc
            if est.status == "ok":
                if self.p.get("diag_est"):  # A63: record only
                    rec["est_diag"] = est_summary(est)
                if self.p.get("turn_refine"):  # A64: the turn of a rotational type, finer than its flip family
                    tr = self.turn_refine(wc, g, est, frame)
                    if tr is not None:
                        est.R, est.t = tr.pop("R"), tr.pop("t")
                        rec["turn_refine"] = tr
                fc = self.fit_check(wc, g, est.R, est.t, z_obs, md["K"])
                T_ = md["T"]
                rec["t_w"] = [round(float(x), 4) for x in T_[:3, :3] @ np.asarray(est.t) + T_[:3, 3]]
                if self.p.get("diag_variants"):
                    rec["variants"] = {}
                    for vn, vv in est.variants.items():
                        if vv.get("R") is None:
                            continue
                        fcv = self.fit_check(wc, str(vv["obj"]), vv["R"], vv["t"], z_obs, md["K"])
                        rec["variants"][vn] = {"t_w": [round(float(x), 4) for x in T_[:3, :3] @ np.asarray(vv["t"])
                                                       + T_[:3, 3]], "inlier": round(fcv["inlier"], 3),
                                               "n_agree": fcv["n_agree"]}
                rec["upright_err_deg"] = round(self.upright_err_deg(wc, g, est.R, up_cam), 1)
                rec.update(R=est.R, t=est.t, score=float(est.score), fit=fc,
                           accepted=self._accept(fc, rec["source"])
                           and rec["upright_err_deg"] <= self.p["upright_tol_deg"])
            if not rec.get("accepted") and wc.diam[g] >= self.p["upright_min_diam_m"] and plane is not None:
                up = self.upright_search(wc, g, px, z_obs, md["K"], plane, sup=sup, segp=segp if use_seg else None)
                if up is not None:
                    R_, t_, fc_ = up[:3]
                    rec["upright"] = {"fit": fc_, "fast_layer_status": est.status}
                    if len(up) > 3:
                        rec["upright"]["dz_m"] = up[3]
                    ue = self.upright_err_deg(wc, g, R_, up_cam)
                    rec["upright"]["upright_err_deg"] = round(ue, 1)
                    if self._accept(fc_, "upright") and ue <= self.p["upright_tol_deg"] \
                            and fc_["n_agree"] > (rec.get("fit") or {}).get("n_agree", -1):
                        T_ = md["T"]
                        rec.update(R=R_, t=t_, fit=fc_, accepted=True, source=f"{rec['source']}+upright",
                                   score=rec.get("score", 0.0) or 0.0,
                                   t_w=[round(float(x), 4) for x in T_[:3, :3] @ t_ + T_[:3, 3]])
            if use_seg and rec.get("accepted") and not (rec.get("source") or "").startswith("texture"):
                # stage 1 (c): the support (relief) check on every non-texture pose
                nodes, nrm = wc.nodes[g]
                sc = SEG.support_check(nodes, nrm, rec["R"], rec["t"], hpx, md["K"], sup, self.p["facing_cos"], segp)
                rec["support_check"] = sc
                if not sc["pass"]:
                    rec.update(accepted=False, rejected="support_check")
            cands.append(rec)
        tm["pose_s"] = time.perf_counter() - a
        support_diag = None
        if use_seg:
            a = time.perf_counter()
            have = {wc.part_of_gid[int(c["gid"])][0] for c in cands if c.get("accepted")}
            # texture-first types are searched only when texture found none; fallback types only when no mask did
            want = sorted(t for t in search_t if t not in tex_first or t not in have) + \
                sorted(t for t in (route.get("fallback") or []) if t not in have)
            parts = [(str(g), wc.part_of_gid[g][0]) for g in wc.rigid_gids if wc.part_of_gid[g][0] in want]
            # the footprint of the parts already perceived: what covers a searched part is then explained
            explain = [wc.nodes[c["gid"]][0][::3] @ np.asarray(c["R"]).T + np.asarray(c["t"])
                       for c in cands if c.get("accepted")]
            if parts:
                res = SEG.support_search(parts, lambda g: wc.nodes[g][0], lambda g: wc.nodes[g][1],
                                         lambda g: wc.family[g], lambda g: wc.diam[g], lambda g: wc.kind[g], z_obs,
                                         md["K"], md["T"], plane, self._search_box(wc, msg["suite"]),
                                         self.p["facing_cos"], self.p["agree_m"], self.p["see_through_m"], segp,
                                         sup=sup, explain_pts=explain, variants_of=lambda g: wc.variants(g),
                                         variants_n_of=lambda g: wc.variants_n(g),
                                         assembly_of=lambda g, R_, t_: self._assembly_at(wc, g, R_, t_, z_obs,
                                                                                         md["K"]),
                                         slide_axes_of=lambda g: self.slide_axes(wc, g),
                                         closed_of=lambda g, R_, t_: self._assembly_closed_at(wc, g, R_, t_, z_obs,
                                                                                              md["K"]),  # A85
                                         # 2026-11-16 (``twin_prior``; None without it: the search as before)
                                         face_of=(lambda g, R_, t_: self.twin_prior(wc, msg["suite"], g, R_, t_,
                                                                                    md["T"], segp))
                                         if segp.get("twin_prior") is not None else None,
                                         accept_of=(lambda g, R_, t_, hm_, pf_: self._twin_accept(
                                             wc, g, R_, t_, z_obs, md["K"], hpx, sup, hm_, pf_, segp))
                                         if (segp.get("twin_prior") is not None or segp.get("memory") is not None)
                                         else None,
                                         # 2026-11-16 (``memory``; None without it): the remembered places first
                                         seeds_of=(lambda g, pf_: self._memory_seeds(wc, msg["suite"], g, md["T"],
                                                                                     pf_, segp))
                                         if segp.get("memory") is not None else None)
                support_diag = res["diag"]
                T_ = md["T"]
                for c in res["candidates"]:
                    g = c["gid"]
                    nodes, nrm = wc.nodes[g]
                    fc = self.fit_check(wc, g, c["R"], c["t"], z_obs, md["K"])
                    sc = SEG.support_check(nodes, nrm, c["R"], c["t"], hpx, md["K"], sup, self.p["facing_cos"], segp)
                    cc = SEG.cover_check(res["map"], res["frame"], nodes, c["R"], c["t"], segp)
                    vis_share = fc["n_agree"] / max(fc["n_vis"] - fc["n_nodepth"], 1)
                    share = fc["n_agree"] / max(fc["n_vis"] - fc["n_nodepth"] - fc["n_occluded"], 1)
                    selfocc = None
                    if segp.get("vis_self_occlusion") and wc.variants(g):
                        # A44: a jointed fixture's root nodes hidden by its own drawers leave the visibility share's
                        # denominator (the drawer fronts cover most of the carcass's facing surface)
                        selfocc = self.self_occluded(wc, g, c["R"], c["t"], z_obs, md["K"])
                        vis_share = fc["n_agree"] / max(fc["n_vis"] - fc["n_nodepth"] - selfocc["n_selfocc"], 1)
                    ok = (self._accept(fc, "support") and sc["pass"] and share >= float(segp["share_min"])
                          and vis_share >= float(segp["vis_min"]) and cc["covered_share"] <= float(segp["cover_max"]))
                    col = None
                    if wc.part_of_gid[int(g)][0] in tex_first:  # a textured type: its search pose must agree in colour
                        col = self.colour_check(wc, g, c["R"], c["t"], z_obs, md["K"], np.asarray(fr["rgb"]))
                        passed = col["r"] is not None and col["r"] >= float(segp["colour_min_r"])
                        step = int(segp.get("colour_sweep_deg") or 0)  # ST-1 F5: off (0) reproduces M2 exactly
                        if ok and not passed and step:
                            # depth alone cannot fix a near-round object's yaw: the best yaw about the model's own
                            # vertical axis by colour, at a stricter bar, and it must still fit the depth
                            best = None
                            ctr = np.array([*nodes[:, :2].mean(0), 0.0])
                            R0, t0_ = np.asarray(c["R"], float), np.asarray(c["t"], float)
                            for deg in range(step, 360, step):
                                a_ = np.deg2rad(deg)
                                Rz = np.array([[np.cos(a_), -np.sin(a_), 0.0], [np.sin(a_), np.cos(a_), 0.0],
                                               [0.0, 0.0, 1.0]])
                                R2 = R0 @ Rz
                                t2 = t0_ + R0 @ ctr - R2 @ ctr
                                x = self.colour_check(wc, g, R2, t2, z_obs, md["K"], np.asarray(fr["rgb"]))
                                if x["r"] is not None and (best is None or x["r"] > best[0]["r"]):
                                    best = (x, R2, t2, deg)
                            if best is not None and best[0]["r"] >= float(segp.get("colour_sweep_min_r", 0.6)):
                                fc2 = self.fit_check(wc, g, best[1], best[2], z_obs, md["K"])
                                if self._accept(fc2, "support"):
                                    c = {**c, "R": best[1], "t": best[2]}
                                    fc = fc2
                                    col = {**best[0], "yaw_sweep_deg": best[3]}
                                    passed = True
                        ok = ok and passed
                    cands.append({"gid": g, "source": "support", "status": "ok", "R": c["R"], "t": c["t"],
                                  "t_w": [round(float(x), 4) for x in T_[:3, :3] @ np.asarray(c["t"]) + T_[:3, 3]],
                                  "score": float(c["refine"]["score"]), "fit": fc, "support_check": sc,
                                  "share": round(float(share), 4), "vis_share": round(float(vis_share), 4),
                                  "cover_check": cc, "colour_check": col, "coarse": c["coarse"],
                                  **({"self_occlusion": selfocc} if selfocc is not None else {}),
                                  "refine": c["refine"],
                                  "n_px": 0, "upright_err_deg": 0.0, "accepted": bool(ok)})
            tm["support_s"] = time.perf_counter() - a
        kept = self._dedup(wc, [c for c in cands if c.get("accepted") and c.get("source") != "support"], md["K"],
                           z_obs.shape, z_obs=z_obs)
        if use_seg:
            sup_c = [c for c in cands if c.get("accepted") and c.get("source") == "support"]
            if segp.get("same_type_by_refine"):
                sup_c = self._same_type_by_refine(wc, sup_c, segp)  # amendment A25
            kept = self._dedup_3d(wc, kept, sup_c, segp, z_obs, md["K"])
        a = time.perf_counter()
        percepts = [self._percept(wc, c, z_obs, md, k) for k, c in enumerate(kept)]
        tm["fixture_s"] = time.perf_counter() - a
        tm["total_s"] = time.perf_counter() - t0
        self.n_looks += 1
        out = {"percepts": percepts, "plane": plane_w, "self_filter": {k: v for k, v in md.items() if k in SELF_RECORD},
               "candidates": [{k: v for k, v in c.items() if k not in ("R", "t")} for c in cands],
               "times": {k: round(v, 4) for k, v in tm.items()}, "n_masks": len(masks)}
        if use_seg:
            out["support"] = {"diag": support_diag, "n_routed_masks": n_routed,
                              "refit": {"rms_m": sup.get("rms_m"), "n_pts": sup.get("n_pts")}}
        if self.p.get("eval_masks") and int(msg.get("look", -1)) == 0:  # m2f2:readiness-r1 (evaluator only)
            shp = tuple(int(x) for x in frame.depth.shape)
            out["eval_masks"] = {"shape": list(shp), "runs": {
                p["id"]: mask_rle(masks[mask_of[id(c)]].full(shp)) for p, c in zip(percepts, kept)
                if id(c) in mask_of}}
        if self.p.get("report_extra"):
            out["extra"] = self._report_extra(wc, cands, kept, percepts, z_obs, md, hpx, np.asarray(fr["rgb"]))
        return out

    def _report_extra(self, wc: Workcell, cands: list, kept: list, percepts: list, z_obs, md, hpx, rgb) -> dict:
        """What perception did not report, for the Pro2.0 self-writing loop (``report_extra``; no percept changes):

        * ``kept``: a brief of each reported percept (id, type, world position, source, fit);
        * ``rejected``: the best poses that were not reported, as whole percepts in the reply's percept format (ids
          from ``p100``; types with no reported percept first, at most ``extra_rejected_per_type`` a type and
          ``extra_max_rejected`` in all), each with ``pos`` and ``why`` (the checks it met);
        * ``blobs``: connected regions standing on the support (``extra_blob_h_m`` above it) that no reported percept
          explains, largest first (at most ``extra_max_blobs``, each at least ``extra_blob_min_px`` pixels): world
          centre ``xy``, ``half_xy`` and ``yaw`` of the principal box, ``z_top``, ``height``, ``n_px`` and the median
          ``rgb``; its middle ``uv`` and box ``uv_box`` in the picture, ``h_min``, the height above the support
          of its lowest part (the 2nd percentile), and how much of it hangs (``_hanging``). A blob has no type.
        """
        from scipy import ndimage as ndi

        def quality(c):
            return (-float(c["fit"]["n_agree"]) * float(c["fit"]["inlier"]), int(c["gid"]), tuple(c.get("t_w") or ()))

        kept_at = {(c["gid"], tuple(c["t_w"])) for c in kept if c.get("t_w") is not None}
        have = {p["type"] for p in percepts}
        by_type: dict = {}
        for c in cands:
            if c.get("R") is None or c.get("fit") is None or (c["gid"], tuple(c.get("t_w") or ())) in kept_at:
                continue
            by_type.setdefault(wc.part_of_gid[int(c["gid"])][0], []).append(c)
        chosen = []
        for typ in sorted(by_type, key=lambda t: (t in have, t)):
            chosen += sorted(by_type[typ], key=quality)[:int(self.p["extra_rejected_per_type"])]
        rejected = []
        for j, c in enumerate(chosen[:int(self.p["extra_max_rejected"])]):
            q = self._percept(wc, c, z_obs, md, 100 + j)
            q["pos"] = c.get("t_w")
            q["why"] = {"accepted": bool(c.get("accepted")), "rejected": c.get("rejected"),
                        "upright_err_deg": c.get("upright_err_deg"), "share": c.get("share"),
                        "vis_share": c.get("vis_share"),
                        "support_pass": (c.get("support_check") or {}).get("pass"),
                        "covered_share": (c.get("cover_check") or {}).get("covered_share"),
                        "colour_r": (c.get("colour_check") or {}).get("r")}
            rejected.append(q)
        brief = [{"id": p["id"], "type": p["type"], "pos": c.get("t_w"), "source": p["source"], "fit": p["fit"],
                  "n_px": p["n_px"]} for p, c in zip(percepts, kept)]
        blobs = []
        if hpx is not None:
            K, T = md["K"], md["T"]
            lo, hi = self.p["extra_blob_h_m"]
            mask = np.isfinite(hpx) & (hpx > float(lo)) & (hpx < float(hi))
            for c in kept:
                mask &= ~self._silhouette(wc, c["gid"], c["R"], c["t"], K, z_obs.shape, dilate=3)
            lab, _n = ndi.label(ndi.binary_opening(mask, iterations=2))
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            pts = SEG.backproject(z_obs, K)
            for L in sorted(np.nonzero(sizes >= int(self.p["extra_blob_min_px"]))[0], key=lambda x: (-sizes[x], x))[
                    :int(self.p["extra_max_blobs"])]:
                v, u = np.nonzero(lab == L)
                W_ = pts[v, u] @ T[:3, :3].T + T[:3, 3]
                ctr = W_[:, :2].mean(0)
                evals, evecs = np.linalg.eigh(np.cov((W_[:, :2] - ctr).T))
                ax = evecs[:, int(np.argmax(evals))]
                ax = ax if ax[0] >= 0 else -ax  # the sign of a principal axis is arbitrary
                ay = np.array([-ax[1], ax[0]])
                s, t_ = (W_[:, :2] - ctr) @ ax, (W_[:, :2] - ctr) @ ay
                mid = ctr + ax * 0.5 * (float(s.max()) + float(s.min())) + ay * 0.5 * (float(t_.max()) + float(t_.min()))
                blobs.append({"xy": [round(float(x), 4) for x in mid],
                              "half_xy": [round(0.5 * float(s.max() - s.min()), 4),
                                          round(0.5 * float(t_.max() - t_.min()), 4)],
                              "yaw": round(float(math.atan2(ax[1], ax[0])), 3), "z_top": round(float(W_[:, 2].max()), 4),
                              "height": round(float(np.percentile(hpx[v, u], 95)), 4), "n_px": int(sizes[L]),
                              "rgb": [int(x) for x in np.median(rgb[v, u], axis=0)]})
                try:  # 2026-11-16 (reply fields only): where the region is in the picture (its middle and box, in
                    # pixels) and how far above the support its lowest part is (a thing that hangs in front of the
                    # scene does not reach down to the support); never the look's failure
                    blobs[-1].update({"uv": [round(float(u.mean()), 1), round(float(v.mean()), 1)],
                                      "uv_box": [int(u.min()), int(v.min()), int(u.max()), int(v.max())],
                                      "h_min": round(float(np.percentile(hpx[v, u], 2)), 4),
                                      **self._hanging(hpx, v, u)})
                except Exception:
                    pass
                if self.p.get("extra_blob_top_m"):
                    try:
                        top = blob_top(W_, float(self.p["extra_blob_top_m"]), int(self.p["extra_blob_top_min_px"]))
                    except Exception:  # a report field only: never the look's failure
                        top = None
                    if top is not None:
                        blobs[-1]["top"] = top
        on = {}
        if self.p.get("extra_on_top_m") and hpx is not None:
            try:  # a reply field only: never the look's failure
                on = self._on_top(wc, kept, percepts, z_obs, md, float(self.p["extra_on_top_m"]))
            except Exception:
                on = {}
        return {"kept": brief, "rejected": rejected, "blobs": blobs, **({"on": on} if on else {})}

    def _on_top(self, wc: Workcell, kept: list, percepts: list, z_obs, md, on_m: float) -> dict:
        """Reply field only (2026-11-16 night, ``extra_on_top_m``): what stands ON each reported object, by percept
        id. The pixels whose world point is ``on_m`` to 25 cm above the highest point of the object's model, outside
        every other reported object's outline, are cut into connected regions; the region with the most pixels
        inside the object's own outline (at least ``extra_blob_min_px`` of them) is the thing. Reported: the box
        across of its top layer (``blob_top`` over 1.5 cm: ``xy``, ``half_xy``, ``yaw``, ``z``), the region's
        ``n_px``, ``inside`` (the share of the top layer's points over the object's own extent across) and the
        object's own top ``z_obj``. Nothing is reported for an object with no such region."""
        from scipy import ndimage as ndi

        K, T = md["K"], md["T"]
        Wd = SEG.backproject(z_obs, K) @ T[:3, :3].T + T[:3, 3]
        zw = Wd[..., 2]
        sils = [self._silhouette(wc, c["gid"], c["R"], c["t"], K, z_obs.shape, dilate=0) for c in kept]
        out = {}
        for i, (p, c) in enumerate(zip(percepts, kept)):
            X = self._visible_nodes(wc, c["gid"], np.asarray(c["R"]), np.asarray(c["t"]))
            if len(X) < 10:
                continue
            Xw = X @ T[:3, :3].T + T[:3, 3]
            z_obj = float(np.percentile(Xw[:, 2], 99))
            lo_xy, hi_xy = Xw[:, :2].min(0), Xw[:, :2].max(0)
            m = np.isfinite(zw) & (zw > z_obj + on_m) & (zw < z_obj + 0.25)
            for j, sj in enumerate(sils):
                if j != i:
                    m &= ~sj
            lab, n = ndi.label(ndi.binary_opening(m, iterations=1))
            if n == 0:
                continue
            ins = np.bincount(lab[sils[i]].ravel(), minlength=n + 1)
            ins[0] = 0
            L = int(np.argmax(ins))
            if int(ins[L]) < int(self.p["extra_blob_min_px"]):
                continue
            v, u = np.nonzero(lab == L)
            Wr = Wd[v, u]
            top = blob_top(Wr, 0.015, int(self.p["extra_blob_top_min_px"]))
            if top is None:
                continue
            sel = Wr[Wr[:, 2] >= top["z"] - 0.015][:, :2]
            inside = float(np.mean(np.all((sel >= lo_xy - 0.01) & (sel <= hi_xy + 0.01), axis=1)))
            out[str(p["id"])] = {**top, "n_px": int(len(v)), "n_in": int(ins[L]), "inside": round(inside, 3),
                                 "z_obj": round(z_obj, 4), "type": p["type"]}
        return out

    @staticmethod
    def _hanging(hpx, v, u, h_m: float = 0.08) -> dict:
        """Reply fields only (2026-11-16): how much of an unexplained region HANGS. A thing standing on the support
        reaches down to it in every picture column; a thing hanging in front of the scene ends above it, and what is
        seen below its lower edge is something else, further off. ``hang_share``: the share of the region's columns
        whose lowest pixel is over ``h_m`` above the support; ``hang_px`` and ``hang_uv``: the count and the middle
        (in the picture) of its pixels over ``h_m`` (a standing object joined to it in the picture is left out)."""
        h = hpx[v, u]
        low = np.full(int(u.max()) + 1, -1, dtype=np.int64)
        np.maximum.at(low, u, v)
        cols = np.nonzero(low >= 0)[0]
        hi = h > float(h_m)
        out = {"hang_share": round(float(np.mean(hpx[low[cols], cols] > float(h_m))), 3), "hang_px": int(hi.sum())}
        if hi.any():
            out["hang_uv"] = [round(float(u[hi].mean()), 1), round(float(v[hi].mean()), 1)]
        return out

    def _search_box(self, wc: Workcell, suite: str) -> tuple:
        """The declared workcell box the model search maps (world x0, x1, y0, y1): the table top's extents when the
        workcell has a table box, else the workspace square around the robot base."""
        for a_ in wc.workcell.get("arena") or []:
            if a_.get("type") == "box":
                c, h = np.asarray(a_["centre"], float), np.asarray(a_["half"], float)
                # the box's world extents: the long-task suite's living-room and study tables are rotated boxes (for
                # an unrotated box, as every earlier suite's, |R| @ half is half exactly)
                h = np.abs(np.asarray(a_.get("R", np.eye(3)), float)) @ h
                return (float(c[0] - h[0]), float(c[0] + h[0]), float(c[1] - h[1]), float(c[1] + h[1]))
        r = float(self.p["ws_radius_m"])
        return (float(wc.base[0] - r), float(wc.base[0] + r), float(wc.base[1] - r), float(wc.base[1] + r))

    def colour_check(self, wc: Workcell, g: str, R, t, z_obs, K, rgb) -> dict:
        """The Pearson correlation of the observed luma and the onboarded node luma (colour memory, blur 2) at the
        visible nodes that agree in depth (m2_segment's texture-first types)."""
        nodes, nrm = wc.nodes[g]
        X, vis = SEG.visible(nodes, nrm, R, t, self.p["facing_cos"])
        idx = np.flatnonzero(vis)
        X = X[vis]
        u, v, ins = SEG.project(X, K, z_obs.shape)
        idx, X, u, v = idx[ins], X[ins], u[ins], v[ins]
        zo = z_obs[v, u]
        ag = np.isfinite(zo) & (np.abs(zo - X[:, 2]) <= self.p["agree_m"])
        n = int(ag.sum())
        col = wc.colour(g)
        if col is None or n < int((self.p.get("support") or SEG.PARAMS).get("colour_min_nodes", 50)):
            return {"n": n, "r": None}
        w = np.array([0.299, 0.587, 0.114])
        lo = rgb[v[ag], u[ag]].astype(float) @ w
        lm = col[idx[ag]].astype(float) @ w
        if lo.std() < 1e-9 or lm.std() < 1e-9:
            return {"n": n, "r": 0.0}
        return {"n": n, "r": round(float(np.corrcoef(lo, lm)[0, 1]), 4)}

    def _agree_nodes(self, wc: Workcell, g: str, R, t, z_obs, K) -> np.ndarray:
        """Camera-frame visible nodes of part ``g`` at (R, t) whose depth agrees with the observation."""
        nodes, nrm = wc.nodes[g]
        X, vis = SEG.visible(nodes, nrm, R, t, self.p["facing_cos"])
        X = X[vis]
        u, v, ins = SEG.project(X, K, z_obs.shape)
        X, u, v = X[ins], u[ins], v[ins]
        zo = z_obs[v, u]
        return X[np.isfinite(zo) & (np.abs(zo - X[:, 2]) <= self.p["agree_m"])]

    def _assembly_at(self, wc: Workcell, g: str, R, t, z_obs, K) -> tuple:
        """Amendment A26: a jointed fixture's whole assembly (nodes, normals; root body frame) with each jointed part's
        value fitted by ``_fit_joint`` at the camera-frame root pose (R, t), and the fitted values."""
        typ, pa = wc.part_of_gid[int(g)]
        tdoc = wc.types[typ]
        q = {}
        for jp, P in tdoc["parts"].items():
            if P["joint"] is not None:
                q[jp] = self._fit_joint_fast(wc, tdoc, typ, jp, (np.asarray(R), np.asarray(t)), z_obs, K)
        bp = KIN.part_body_poses(tdoc, {tdoc["root_body"]: (np.eye(3), np.zeros(3))}, q)
        pts, nrs = [wc.nodes[str(g)][0]], [wc.nodes[str(g)][1]]
        for jp in q:
            Rj, tj = bp[jp]
            n_, r_ = wc.nodes[str(wc.gid[(typ, jp)])]
            pts.append(n_ @ Rj.T + tj)
            nrs.append(r_ @ Rj.T)
        return np.concatenate(pts), np.concatenate(nrs), q

    def _assembly_closed_at(self, wc: Workcell, g: str, R, t, z_obs, K) -> tuple:
        """m2r3:fix-r2 (A85): a jointed fixture's root with its CLOSED parts held at their rest value (qpos0) and its
        open parts left out (root body frame): the joints are fitted once at the camera-frame root pose (R, t) as
        ``_assembly_at`` fits them; a part whose fitted value lies within ``closed_tol_m`` (a slide) or
        ``closed_tol_deg`` (a hinge) of qpos0 is closed. A closed drawer rests at its stop (A69), so its front moves
        with the carcass; an open one's value is unknown and would absorb a slide of the carcass along its axis.
        Returns (nodes, normals, {part: fitted value}, [closed parts], [open parts])."""
        segp = {**SEG.PARAMS, **(self.p.get("support") or {})}
        typ, pa = wc.part_of_gid[int(g)]
        tdoc = wc.types[typ]
        q, closed, opened = {}, [], []
        for jp, P in tdoc["parts"].items():
            jt = P["joint"]
            if jt is None:
                continue
            q[jp] = self._fit_joint_fast(wc, tdoc, typ, jp, (np.asarray(R), np.asarray(t)), z_obs, K)
            q0 = float(jt.get("qpos0", 0.0))
            tol = float(segp["closed_tol_m"]) if jt["type"] == "slide" else math.radians(float(segp["closed_tol_deg"]))
            (closed if abs(q[jp] - q0) <= tol else opened).append(jp)
        bp = KIN.part_body_poses(tdoc, {tdoc["root_body"]: (np.eye(3), np.zeros(3))},
                                 {jp: float(tdoc["parts"][jp]["joint"].get("qpos0", 0.0)) for jp in closed})
        pts, nrs = [wc.nodes[str(g)][0]], [wc.nodes[str(g)][1]]
        for jp in closed:
            Rj, tj = bp[jp]
            n_, r_ = wc.nodes[str(wc.gid[(typ, jp)])]
            pts.append(n_ @ Rj.T + tj)
            nrs.append(r_ @ Rj.T)
        return np.concatenate(pts), np.concatenate(nrs), q, closed, opened

    def open_dir(self, wc: Workcell, g: str):
        """2026-11-16 (``twin_prior``): the unit direction, in the ROOT body frame and across the root's vertical,
        in which the slide-jointed parts of the jointed type whose root part is ``g`` OPEN (each part from its rest
        value qpos0 to the end of its range farther from it; the mean over the parts). None for any other part, a
        type without slide parts, or parts that open in opposed directions."""
        typ, pa = wc.part_of_gid[int(g)]
        tdoc = wc.types[typ]
        if pa != tdoc["root_body"]:
            return None
        up = np.asarray(wc.rest_up[str(g)], float)
        ds = []
        for jp, P_ in tdoc["parts"].items():
            jt = P_["joint"]
            if jt is None or jt["type"] != "slide":
                continue
            q0 = float(jt.get("qpos0", 0.0))
            lo, hi = float(jt["range"][0]), float(jt["range"][1])
            q1 = lo if abs(lo - q0) >= abs(hi - q0) else hi
            b0 = KIN.part_body_poses(tdoc, {tdoc["root_body"]: (np.eye(3), np.zeros(3))}, {jp: q0})[jp][1]
            b1 = KIN.part_body_poses(tdoc, {tdoc["root_body"]: (np.eye(3), np.zeros(3))}, {jp: q1})[jp][1]
            d = np.asarray(b1, float) - np.asarray(b0, float)
            d = d - (d @ up) * up
            n_ = float(np.linalg.norm(d))
            if n_ > 1e-6:
                ds.append(d / n_)
        if not ds:
            return None
        m = np.mean(ds, axis=0)
        n_ = float(np.linalg.norm(m))
        return None if n_ < 0.5 else m / n_

    def twin_prior(self, wc: Workcell, suite: str, g: str, R, t, T, segp: dict):
        """2026-11-16 (``segp["twin_prior"]``): the facing prior's verdict on the camera-frame root pose (R, t) of the
        jointed fixture whose root part is ``g``: positive = the right way round, negative = back to front, None =
        no verdict. The prior is a dict of up to two priors; each abstains when it is unsure:

        * ``"inward"`` {"ahead_m", "min_cos", "suites"}: the parts open toward where the robot works. The cosine
          between the opening direction (``open_dir``) and the direction from the root's footprint centroid to (a) the
          point ``ahead_m`` in front of the robot base, toward the middle of the declared table box, and (b) that
          middle itself (the base alone in a workcell without a table box). The larger of the two in size is its
          answer when that is at least ``min_cos``; when both reach ``min_cos`` with opposite signs (a fixture
          standing between the two points) it says nothing. It speaks only in the workcells listed in ``suites``
          when that list is given. From the declared workcell and the type's geometry alone.
        * ``"memory"`` {suite: {type: yaw_deg}} (``"yaw_tol_deg"``): the turn about the vertical, in the world, at
          which the robot's own earlier looks found this type in this workcell (given in the parameters; the server
          reads no file for it). +1 within the tolerance of it, -1 within the tolerance of its half turn; it says
          nothing about where the fixture stands, so a fixture moved without being turned keeps its verdict.

        A negative verdict needs every prior that speaks to say so, and a positive one likewise; a split is None."""
        pr = segp.get("twin_prior") or {}
        od = self.open_dir(wc, g)
        if od is None:
            return None
        T = np.asarray(T, float)
        Rw = T[:3, :3] @ np.asarray(R, float)
        tw = T[:3, :3] @ np.asarray(t, float) + T[:3, 3]
        said = []
        inw = pr.get("inward")
        if inw is not None and (inw.get("suites") is None or suite in inw["suites"]):
            nodes = wc.nodes[str(g)][0]
            ctr = Rw @ np.array([float(nodes[:, 0].mean()), float(nodes[:, 1].mean()), 0.0]) + tw
            base = np.asarray(wc.base[:2], float)
            x0, x1, y0, y1 = self._search_box(wc, suite)
            mid = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
            refs = [base]
            if float(np.linalg.norm(mid - base)) > 1e-6:
                refs = [base + float(inw.get("ahead_m", 0.45)) * (mid - base) / float(np.linalg.norm(mid - base)), mid]
            d = (Rw @ od)[:2]
            cs = []
            for ref in refs:
                v = ref - ctr[:2]
                if float(np.linalg.norm(v)) > 1e-6 and float(np.linalg.norm(d)) > 1e-6:
                    cs.append(float(d @ v) / (float(np.linalg.norm(d)) * float(np.linalg.norm(v))))
            lim = float(inw.get("min_cos", 0.3))
            loud = [c for c in cs if abs(c) >= lim]
            if loud and not (min(loud) < 0.0 < max(loud)):
                said.append(max(loud, key=abs))
        mem = ((pr.get("memory") or {}).get(suite) or {}).get(wc.part_of_gid[int(g)][0])
        if mem is not None:
            tol = float(pr.get("yaw_tol_deg", 45.0))
            off = abs((math.degrees(math.atan2(float(Rw[1, 0]), float(Rw[0, 0]))) - float(mem) + 180.0) % 360.0
                      - 180.0)
            if off <= tol:
                said.append(1.0)
            elif off >= 180.0 - tol:
                said.append(-1.0)
        if said and all(x < 0 for x in said):
            return max(said)
        if said and all(x > 0 for x in said):
            return min(said)
        return None

    def _memory_seeds(self, wc: Workcell, suite: str, g: str, T, pf: dict, segp: dict) -> list:
        """2026-11-16 (``segp["memory"]`` = {suite: {type: [{"pos": [x, y, z], "yaw_deg": d}, ...]}}): the places
        where the robot's own earlier looks at this workcell found the type whose root part is ``g``, as the
        search's plane-frame starts [(yaw, a, b)] under this look's camera ``T`` (the inverse of ``SEG.pose_at``).
        The parameters carry them; the server reads no file for it. [] when nothing is remembered."""
        typ, pa = wc.part_of_gid[int(g)]
        if pa != wc.types[typ]["root_body"]:
            return []
        T = np.asarray(T, float)
        out = []
        for m in ((segp.get("memory") or {}).get(suite) or {}).get(typ) or []:
            yw = math.radians(float(m["yaw_deg"]))
            Rw = np.array([[math.cos(yw), -math.sin(yw), 0.0], [math.sin(yw), math.cos(yw), 0.0], [0.0, 0.0, 1.0]])
            R_c = T[:3, :3].T @ Rw
            q = T[:3, :3].T @ (np.asarray(m["pos"], float) - T[:3, 3]) - pf["p0"]
            Rz = pf["G"].T @ R_c
            out.append((math.atan2(float(Rz[1, 0]), float(Rz[0, 0])), float(q @ pf["ex"]), float(q @ pf["ey"])))
        return out

    def _twin_accept(self, wc: Workcell, g: str, R, t, z_obs, K, hpx, sup, hm, pf, segp: dict) -> bool:
        """2026-11-16 (``twin_prior``): whether ``full`` would accept this model-search pose of ``g`` (the same floors:
        the fit check, the support check, the occlusion-aware share, the visibility share with the fixture's own
        parts left out, the cover check). Used on a twin pose before it replaces the search's choice."""
        nodes, nrm = wc.nodes[g]
        fc = self.fit_check(wc, g, R, t, z_obs, K)
        sc = SEG.support_check(nodes, nrm, R, t, hpx, K, sup, self.p["facing_cos"], segp)
        cc = SEG.cover_check(hm, pf, nodes, R, t, segp)
        vis_share = fc["n_agree"] / max(fc["n_vis"] - fc["n_nodepth"], 1)
        share = fc["n_agree"] / max(fc["n_vis"] - fc["n_nodepth"] - fc["n_occluded"], 1)
        if segp.get("vis_self_occlusion") and wc.variants(g):
            selfocc = self.self_occluded(wc, g, R, t, z_obs, K)
            vis_share = fc["n_agree"] / max(fc["n_vis"] - fc["n_nodepth"] - selfocc["n_selfocc"], 1)
        return bool(self._accept(fc, "support") and sc["pass"] and share >= float(segp["share_min"])
                    and vis_share >= float(segp["vis_min"]) and cc["covered_share"] <= float(segp["cover_max"]))

    def slide_axes(self, wc: Workcell, g: str) -> list:
        """Amendment A68: the distinct unit axes, in the ROOT body frame, along which the slide-jointed parts of the
        jointed type whose root part is ``g`` move (each part placed at qpos0 and at qpos0 + 1 cm; parallel axes
        merged). [] for any other part."""
        typ, pa = wc.part_of_gid[int(g)]
        tdoc = wc.types[typ]
        if pa != tdoc["root_body"]:
            return []
        out = []
        for jp, P_ in tdoc["parts"].items():
            jt = P_["joint"]
            if jt is None or jt["type"] != "slide":
                continue
            q0 = float(jt.get("qpos0", 0.0))
            b0 = KIN.part_body_poses(tdoc, {tdoc["root_body"]: (np.eye(3), np.zeros(3))}, {jp: q0})[jp][1]
            b1 = KIN.part_body_poses(tdoc, {tdoc["root_body"]: (np.eye(3), np.zeros(3))}, {jp: q0 + 0.01})[jp][1]
            ax = (np.asarray(b1, float) - np.asarray(b0, float)) / 0.01
            n_ = float(np.linalg.norm(ax))
            if n_ < 1e-9:
                continue
            ax = ax / n_
            if not any(abs(float(np.dot(ax, o))) > math.cos(math.radians(1.0)) for o in out):
                out.append(ax)
        return [o.tolist() for o in out]

    @staticmethod
    def joint_range(jt: dict, segp: dict) -> tuple:
        """The fitted range of one joint: its full range, or (A69, ``joint_rest_stop``) the span between its range
        end farther from qpos0 and qpos0."""
        lo, hi = float(jt["range"][0]), float(jt["range"][1])
        if not segp.get("joint_rest_stop"):
            return lo, hi
        q0 = float(jt.get("qpos0", 0.0))
        q_open = lo if abs(lo - q0) >= abs(hi - q0) else hi
        return (min(q_open, q0), max(q_open, q0))

    def _fit_joint_fast(self, wc, tdoc, typ, jp, root_cam, z_obs, K) -> float:
        """``_fit_joint``'s criterion (agreeing minus 2 x seen-through visible nodes of the part at the value) on
        every ``asm_joint_node_stride``-th node, a ``asm_joint_step_m`` grid and 8 golden-section steps (amendment
        A29: the refine-time joint fit of ``_assembly_at``; the percept's joints are still fitted by
        ``_fit_joint``)."""
        segp = {**SEG.PARAMS, **(self.p.get("support") or {})}
        P = tdoc["parts"][jp]
        jt = P["joint"]
        g = str(wc.gid[(typ, jp)])
        n_, r_ = wc.nodes[g]
        stride = int(segp.get("asm_joint_node_stride", 3))
        n_, r_ = n_[::stride], r_[::stride]
        lo, hi = self.joint_range(jt, segp)  # A69
        H, W = z_obs.shape
        ag, see, fc = float(self.p["agree_m"]), float(self.p["see_through_m"]), float(self.p["facing_cos"])

        def score(q):
            bp = KIN.part_body_poses(tdoc, {tdoc["root_body"]: root_cam}, {jp: q})
            Rj, tj = bp[jp]
            X = n_ @ Rj.T + tj
            N = r_ @ Rj.T
            view = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
            X = X[((-(N * view).sum(1)) > fc) & (X[:, 2] > 0.05)]
            u = np.rint(K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2] - 0.5).astype(int)
            v = np.rint(K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2] - 0.5).astype(int)
            ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            zo = z_obs[v[ins], u[ins]]
            dz = zo - X[ins, 2]
            has = np.isfinite(zo)
            return float((has & (np.abs(dz) <= ag)).sum()) - 2.0 * float((has & (dz > see)).sum())

        step = float(segp.get("asm_joint_step_m", 0.004)) if jt["type"] == "slide" else math.radians(4.0)
        qs = np.arange(lo, hi + 1e-9, step)
        vals = [score(q) for q in qs]
        i = int(np.argmax(vals))
        a_, b_ = qs[max(i - 1, 0)], qs[min(i + 1, len(qs) - 1)]
        gr = (math.sqrt(5) - 1) / 2
        c_, d_ = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_)
        for _ in range(8):
            if score(c_) >= score(d_):
                b_ = d_
            else:
                a_ = c_
            c_, d_ = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_)
        return float(0.5 * (a_ + b_))

    def _same_type_by_refine(self, wc: Workcell, support: list, segp: dict) -> list:
        """Amendment A25: of two model-search candidates of ONE part that cannot both be true (centres within
        ``dup_frac`` of its diameter, or more than ``dup_inside`` of one's nodes inside the other's shrunk body box:
        the dedup's same-type rule), keep the one with the better refine score (the refine's own criterion: soft
        agreement minus 2 x seen-through); the existing dedup then runs on the rest. A probe: a false stove pose
        30 cm away with more 1 cm agreeing nodes displaced the true one (refine score 409 against 843)."""
        order = sorted(support, key=lambda c: (-float(c.get("score") or 0.0), int(c["gid"])))
        out = []
        for c in order:
            g = c["gid"]
            n_ = wc.nodes[g][0]
            ext = n_.max(0) - n_.min(0)
            bx = SEG.body_box(n_, np.minimum(float(segp["dup_shrink_m"]), 0.2 * ext))
            Xc = n_[::4] @ np.asarray(c["R"]).T + np.asarray(c["t"])
            clash = None
            for k in out:
                if k["gid"] != g:
                    continue
                d = float(np.linalg.norm(np.asarray(c["t"]) - np.asarray(k["t"])))
                if d < self.p["dup_frac"] * wc.diam[g] or SEG.inside_share(Xc, k["R"], k["t"], bx) > \
                        float(segp["dup_inside"]):
                    clash = k
                    break
            if clash is not None:
                c["dup_of"] = f"same_type_refine:{round(float(clash.get('score') or 0.0), 1)}"
                continue
            out.append(c)
        return out

    def _dedup_3d(self, wc: Workcell, kept: list, support: list, segp: dict, z_obs=None, K=None) -> list:
        """Model-search candidates, best explained first (agreeing nodes), join the kept percepts unless they
        duplicate one (amendments A16, A17):

        * the same type within ``dup_frac`` of its diameter, or with more than ``dup_inside`` of its nodes inside the
          kept one's body box (shrunk by min(``dup_shrink_m``, 20% of each extent)): two rigid parts of one type
          cannot interpenetrate;
        * another type: more than ``dup_inside`` of its nodes AND its centre inside the kept part's box, or more than
          ``dup_inside_strong`` of its nodes inside it (a part under another that rests on it, the cookies box under
          a bowl, keeps its centre outside the bowl's box);
        * EXPLAINED DEPTH: more than ``dup_explained`` of its depth-agreeing visible nodes lie within
          ``explained_m`` of the kept part's model surface (two percepts may not explain the same observed surface).

        Then a kept NON-texture mask percept whose agreeing nodes are explained in the same way by a kept search
        part with more agreeing nodes is dropped (region groceries fitted onto the basket's weave). A
        grocery placed in the basket explains its own surface, not the basket's."""
        from scipy.spatial import cKDTree

        out = list(kept)
        boxes, trees = {}, {}

        def box(g):
            if g not in boxes:
                n_ = wc.nodes[g][0]
                ext = n_.max(0) - n_.min(0)
                boxes[g] = SEG.body_box(n_, np.minimum(float(segp["dup_shrink_m"]), 0.2 * ext))
            return boxes[g]

        def tree(k):
            key = id(k)
            if key not in trees:
                trees[key] = cKDTree(wc.nodes[k["gid"]][0][::2] @ np.asarray(k["R"]).T + np.asarray(k["t"]))
            return trees[key]

        agree_cache = {}

        def explained(c, k):
            if z_obs is None:
                return 0.0
            if id(c) not in agree_cache:
                agree_cache[id(c)] = self._agree_nodes(wc, c["gid"], c["R"], c["t"], z_obs, K)
            A = agree_cache[id(c)]
            if not len(A):
                return 0.0
            d, _ = tree(k).query(A, k=1, distance_upper_bound=float(segp["explained_m"]))
            return float(np.isfinite(d).mean())

        for c in sorted(support, key=lambda c: (-c["fit"]["n_agree"], int(c["gid"]))):
            g = c["gid"]
            Xc = wc.nodes[g][0][::4] @ np.asarray(c["R"]).T + np.asarray(c["t"])
            ctr = np.asarray(c["t"]) + np.asarray(c["R"]) @ (0.5 * (wc.nodes[g][0].min(0) + wc.nodes[g][0].max(0)))
            clash = None
            for k in out:
                sh_in = SEG.inside_share(Xc, k["R"], k["t"], box(k["gid"]))
                if k["gid"] == g and (float(np.linalg.norm(np.asarray(c["t"]) - np.asarray(k["t"])))
                                      < self.p["dup_frac"] * wc.diam[g] or sh_in > float(segp["dup_inside"])):
                    clash = "same_type"
                    break
                if sh_in > float(segp["dup_inside_strong"]) or (
                        sh_in > float(segp["dup_inside"]) and SEG.point_inside(ctr, k["R"], k["t"], box(k["gid"]))):
                    clash = f"inside:{k['gid']}"
                    break
                ex = explained(c, k)
                if ex > float(segp["dup_explained"]):
                    clash = f"explained:{k['gid']}:{round(ex, 3)}"
                    break
            if clash:
                c["dup_of"] = clash
            else:
                out.append(c)
        final = []
        for k in out:
            if k.get("source") != "support" and not (k.get("source") or "").startswith("texture"):
                hit = next((s_ for s_ in out if s_.get("source") == "support"
                            and s_["fit"]["n_agree"] > k["fit"]["n_agree"]
                            and explained(k, s_) > float(segp["dup_explained"])), None)
                if hit is not None:
                    k["dup_of"] = f"explained_by_search:{hit['gid']}"
                    continue
            final.append(k)
        return final

    def _accept(self, fc: dict, source: str | None) -> bool:
        floor = self.p["accept_inlier"] if (source or "").startswith("texture") else self.p["accept_inlier_region"]
        return fc["inlier"] >= floor and fc["n_agree"] >= self.p["accept_min_agree"]

    def upright_err_deg(self, wc: Workcell, g: str, R, up_cam) -> float:
        """The angle between world up and the type's rest-up axis at this pose. Every object and fixture of these
        workcells rests upright on a support surface (the rest pose M1's exporter recorded), so a pose that tilts a
        part far from upright is rejected (a flat plate fitted onto a vertical cabinet front)."""
        a = np.asarray(R, float) @ wc.rest_up[str(g)]
        c = float(np.clip(a @ np.asarray(up_cam, float) / max(np.linalg.norm(a), 1e-9), -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    def _silhouette(self, wc: Workcell, g: str, R, t, K, shape, dilate: int = 2) -> np.ndarray:
        from scipy.ndimage import binary_dilation

        X = self._visible_nodes(wc, g, np.asarray(R), np.asarray(t))
        X = X[X[:, 2] > 0.05]
        H, W = shape
        u = np.rint(K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2] - 0.5).astype(int)
        v = np.rint(K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2] - 0.5).astype(int)
        ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        out = np.zeros(shape, bool)
        out[v[ins], u[ins]] = True
        return binary_dilation(out, iterations=dilate + 2)

    def _explained_share(self, wc: Workcell, c: dict, k: dict, z_obs, K, explained_m: float) -> float:
        """The share of candidate ``c``'s depth-agreeing visible nodes that lie within ``explained_m`` of kept part
        ``k``'s model surface (A16/A17's explained-depth measure)."""
        from scipy.spatial import cKDTree

        A = self._agree_nodes(wc, c["gid"], c["R"], c["t"], z_obs, K)
        if not len(A):
            return 0.0
        tr = cKDTree(wc.nodes[k["gid"]][0][::2] @ np.asarray(k["R"]).T + np.asarray(k["t"]))
        d, _ = tr.query(A, k=1, distance_upper_bound=float(explained_m))
        return float(np.isfinite(d).mean())

    def _label_tie_order(self, wc: Workcell, order: list, tie_frac: float, px_ratio: float) -> list:
        """``dedup_label_tie``: a best-fit-first list, with a candidate moved ahead of one it ties with at the same spot
        (another type, centres within ``dup_frac`` of the smaller diameter, fits within ``tie_frac``) when its colour mask
        has at least ``px_ratio`` times the pixels. The dedup's rules then keep the label the colour supports."""
        def fit(c):
            return float(c["fit"]["n_agree"]) * float(c["fit"]["inlier"])

        out = list(order)
        for i in range(len(out)):
            a = out[i]
            rivals = [b for b in out[i + 1:] if b["gid"] != a["gid"]
                      and float(np.linalg.norm(np.asarray(a["t"]) - np.asarray(b["t"])))
                      < self.p["dup_frac"] * min(wc.diam[a["gid"]], wc.diam[b["gid"]])
                      and abs(fit(a) - fit(b)) <= tie_frac * max(fit(a), fit(b))
                      and int(b.get("n_px") or 0) >= px_ratio * max(int(a.get("n_px") or 0), 1)]
            if rivals:
                b = max(rivals, key=lambda c: int(c.get("n_px") or 0))
                j = next(n for n, c in enumerate(out) if c is b)
                out.insert(i, out.pop(j))
                b["label_tie_over"] = str(a["gid"])
        return out

    def _dedup(self, wc: Workcell, cands: list, K, shape, z_obs=None) -> list:
        """Best fit first; a candidate whose centre is within ``dup_frac`` of the smaller diameter of a kept one, or
        whose node silhouette lies mostly (``dup_overlap``) inside the kept ones' silhouettes, is a duplicate.

        m2r3:fix-r1 (A67, ``dedup_depth``): a silhouette overlap alone does not tell a duplicate from an OCCLUDED
        neighbour (a grocery moved behind another: its visible surface is its own, at another depth). With
        ``dedup_depth`` the silhouette rule also needs the explained-depth test of A16/A17: more than
        ``dup_explained`` of the candidate's depth-agreeing visible nodes lie within ``explained_m`` of an
        overlapping kept part's model surface. The centre rule is unchanged."""
        segp = {**SEG.PARAMS, **(self.p.get("support") or {})}
        order = sorted(cands, key=lambda c: (-c["fit"]["n_agree"] * c["fit"]["inlier"], int(c["gid"])))
        if self.p.get("dedup_label_tie"):
            order = self._label_tie_order(wc, order, *self.p["dedup_label_tie"])
        kept = []
        sils = []
        cover = np.zeros(shape, bool)
        for c in order:
            cw = c["t"]
            clash = False
            for k in kept:
                lim = self.p["dup_frac"] * min(wc.diam[c["gid"]], wc.diam[k["gid"]])
                if float(np.linalg.norm(np.asarray(cw) - np.asarray(k["t"]))) < lim:
                    clash = True
                    c["dup_of"] = f"centre:{k['gid']}"
                    break
            sil = self._silhouette(wc, c["gid"], c["R"], c["t"], K, shape, dilate=0)
            if not clash and sil.any() and kept:
                ov = float((sil & cover).sum()) / float(sil.sum())
                if ov > self.p["dup_overlap"]:
                    if self.p.get("dedup_depth") and z_obs is not None:
                        ex = max((self._explained_share(wc, c, k, z_obs, K, float(segp["explained_m"]))
                                  for k, sk in zip(kept, sils) if (sil & sk).any()), default=0.0)
                        c["dedup_check"] = {"silhouette_overlap": round(ov, 3), "explained": round(ex, 3)}
                        if ex > float(segp["dup_explained"]):
                            clash = True
                            c["dup_of"] = "silhouette+explained"
                    else:
                        clash = True
                        c["dup_of"] = "silhouette"
            if not clash:
                kept.append(c)
                sk = self._silhouette(wc, c["gid"], c["R"], c["t"], K, shape)
                sils.append(sk)
                cover |= sk
        return kept

    def upright_search(self, wc: Workcell, g: str, px: np.ndarray, z_obs, K, plane, sup: dict | None = None,
                       segp: dict | None = None) -> tuple | None:
        """A support-constrained pose search (amendment A2): the part upright on the fitted plane (its model z along
        the plane normal), yaw and the in-plane position searched coarse to fine around the mask's centroid,
        scored by node agreement minus 2 x seen-through over the whole frame.

        With ``segp`` (amendment A14, m2_segment on): the plane is m2_segment's TLS refit ``sup``; after the coarse
        grid the agreement is soft (an Epanechnikov kernel over +-``agree_m``: the hard count is flat inside its
        window, so it can neither fit a height nor settle the yaw), and a height offset over ``dz_fixture_m`` is
        fitted (coarse ``dz_coarse_m``, then ``dz_fine_m`` after each refine stage). The visual mesh of a resting part
        need not touch the support (the basket's lowest visual node sits 5 mm below its resting pose); returns
        (R, t, fit, dz)."""
        if segp is not None and sup is not None:
            n = np.asarray(sup["normal"], float)
            d0 = float(sup["d"])
        else:
            n = np.asarray(plane["normal"], float)
            d0 = float(plane["d"])
        zz = z_obs[px]
        ok = np.isfinite(zz)
        if ok.sum() < 50:
            return None
        vv, uu = np.nonzero(px)
        vv, uu, zz = vv[ok], uu[ok], zz[ok]
        P = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0] * zz, (vv + 0.5 - K[1, 2]) / K[1, 1] * zz, zz], 1)
        c = P.mean(0)
        c_on = c - (c @ n + d0) * n  # the centroid dropped onto the plane
        a = np.cross(n, [1.0, 0.0, 0.0])
        if np.linalg.norm(a) < 0.1:
            a = np.cross(n, [0.0, 1.0, 0.0])
        ex = a / np.linalg.norm(a)
        ey = np.cross(n, ex)
        G = np.stack([ex, ey, n], 1)  # plane frame -> camera
        nodes, nrm = wc.nodes[g]
        sel = np.linspace(0, len(nodes) - 1, min(len(nodes), int(self.p["upright_nodes"]))).astype(int)
        Xn, Nn = nodes[sel], nrm[sel]
        h0 = -float(nodes[:, 2].min())  # the model's up is body z (LIBERO rest pose; checked by the exporter)
        H, W = z_obs.shape
        agree_m = float(self.p["agree_m"])

        def score_many(yaws, offs, dzs=(0.0,), soft=False):
            best = (-1e18, None)
            for yw in yaws:
                cz, sz = math.cos(yw), math.sin(yw)
                Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]])
                R = G @ Rz
                Xr, Nr = Xn @ R.T, Nn @ R.T
                for o in offs:
                    for dz_ in dzs:
                        t = c_on + G @ np.array([o[0], o[1], 0.0]) + (h0 + dz_) * n
                        X = Xr + t
                        view = X / np.linalg.norm(X, axis=1, keepdims=True)
                        vis = (-(Nr * view).sum(1)) > self.p["facing_cos"]
                        Xv = X[vis]
                        u = np.rint(K[0, 0] * Xv[:, 0] / Xv[:, 2] + K[0, 2] - 0.5).astype(int)
                        v = np.rint(K[1, 1] * Xv[:, 1] / Xv[:, 2] + K[1, 2] - 0.5).astype(int)
                        ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                        zo = z_obs[v[ins], u[ins]]
                        dz = zo - Xv[ins, 2]
                        has = np.isfinite(zo)
                        if soft:
                            ag = float(np.where(has, np.clip(1.0 - (dz / agree_m) ** 2, 0.0, 1.0), 0.0).sum())
                        else:
                            ag = float((has & (np.abs(dz) <= agree_m)).sum())
                        sc = ag - 2.0 * (has & (dz > self.p["see_through_m"])).sum()
                        if sc > best[0]:
                            best = (sc, (yw, o, dz_))
            return best

        r, st = float(self.p["upright_xy_m"]), float(self.p["upright_xy_step_m"])
        grid = [(x, y) for x in np.arange(-r, r + 1e-9, st) for y in np.arange(-r, r + 1e-9, st)]
        yaws = np.radians(np.arange(0.0, 360.0, float(self.p["upright_yaw_deg"])))
        sc, (yw, o, dzv) = score_many(yaws, grid)
        soft = segp is not None and sup is not None
        lo = hi = 0.0
        if soft:
            lo, hi = (float(x) for x in segp["dz_fixture_m"])
            st_c, st_f = float(segp["dz_coarse_m"]), float(segp["dz_fine_m"])
            sc, (yw, o, dzv) = score_many([yw], [o], list(np.arange(lo, hi + 1e-9, st_c)), soft=True)
        for dyaw, dxy in ((math.radians(4.0), 0.008), (math.radians(1.5), 0.003), (math.radians(0.5), 0.001)):
            ys = [yw + k * dyaw for k in (-2, -1, 0, 1, 2)]
            offs = [(o[0] + i * dxy, o[1] + j * dxy) for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2)]
            sc, (yw, o, dzv) = score_many(ys, offs, (dzv,), soft=soft)
            if soft:
                dzs = [q for q in np.arange(dzv - 4 * st_f, dzv + 4 * st_f + 1e-9, st_f) if lo - 1e-9 <= q <= hi + 1e-9]
                sc, (yw, o, dzv) = score_many([yw], [o], dzs, soft=True)
        cz, sz = math.cos(yw), math.sin(yw)
        R = G @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]])
        t = c_on + G @ np.array([o[0], o[1], 0.0]) + (h0 + dzv) * n
        if soft:
            return R, t, self.fit_check(wc, g, R, t, z_obs, K), round(float(dzv), 5)
        return R, t, self.fit_check(wc, g, R, t, z_obs, K)

    def _percept(self, wc: Workcell, c: dict, z_obs, md, k: int) -> dict:
        typ, pa = wc.part_of_gid[int(c["gid"])]
        tdoc = wc.types[typ]
        T = md["T"]
        Rw = T[:3, :3] @ np.asarray(c["R"])
        tw = T[:3, :3] @ np.asarray(c["t"]) + T[:3, 3]
        joints = {}
        for jp, P in tdoc["parts"].items():
            if P["joint"] is None:
                continue
            joints[jp] = self._fit_joint(wc, tdoc, typ, jp, (np.asarray(c["R"]), np.asarray(c["t"])), z_obs, md["K"])
        bp = KIN.part_body_poses(tdoc, {tdoc["root_body"]: (Rw, tw)}, {jp: v["q"] for jp, v in joints.items()})
        return {"id": f"p{k}", "type": typ, "gid": c["gid"], "score": c["score"], "fit": c["fit"],
                "source": c["source"], "n_px": c["n_px"],
                "bodies": {r: {"pos": [float(x) for x in t], "quat_wxyz": KIN.mat_to_quat(R)}
                           for r, (R, t) in bp.items()},
                "joints": {jp: v for jp, v in joints.items()}}

    def _fit_joint(self, wc, tdoc, typ, jp, root_cam, z_obs, K) -> dict:
        P = tdoc["parts"][jp]
        jt = P["joint"]
        g = str(wc.gid[(typ, jp)])
        lo, hi = self.joint_range(jt, {**SEG.PARAMS, **(self.p.get("support") or {})})  # A69

        def score(q):
            bp = KIN.part_body_poses(tdoc, {tdoc["root_body"]: root_cam}, {jp: q})
            R, t = bp[jp]
            fc = self.fit_check(wc, g, R, t, z_obs, K)
            return fc["n_agree"] - 2.0 * fc["n_through"], fc

        step = self.p["joint_step_m"] if jt["type"] == "slide" else math.radians(2.0)
        qs = np.arange(lo, hi + 1e-9, step)
        vals = [score(q)[0] for q in qs]
        i = int(np.argmax(vals))
        a_, b_ = qs[max(i - 1, 0)], qs[min(i + 1, len(qs) - 1)]
        gr = (math.sqrt(5) - 1) / 2
        c_, d_ = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_)
        for _ in range(12):
            if score(c_)[0] >= score(d_)[0]:
                b_ = d_
            else:
                a_ = c_
            c_, d_ = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_)
        q = 0.5 * (a_ + b_)
        s, fc = score(q)
        return {"q": float(q), "score": float(s), "fit": fc}

    def track(self, msg: dict) -> dict:
        """Refine the requested percepts from their predicted poses (``priors``: {id: {type, gid, R_w, t_w}})."""
        from rowant_lab.fastlayer import layer as LY
        from rowant_lab.fastlayer.frame import Frame

        wc = self.cell(msg["suite"])
        reseed_request(msg["episode"], msg["look"])  # amendment A39
        t0 = time.perf_counter()
        fr = msg["cameras"][0]
        d2, md = self._mask_depth(wc, fr, msg["joint_pos"], msg["gripper_qpos"])
        frame = Frame(rgb=np.asarray(fr["rgb"]), depth=d2, K=md["K"], depth_scale=float(fr["depth_scale_mm"]))
        ctx = wc.layer.frame_context(frame)
        z_obs = ctx["z_obs"]
        T = md["T"]
        Rcw, tcw = T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]
        out = {}
        H, W = z_obs.shape
        for pid, pr in msg["priors"].items():
            g = str(pr["gid"])
            Rc = Rcw @ np.asarray(pr["R_w"], float)
            tc = Rcw @ np.asarray(pr["t_w"], float) + tcw
            X = self._visible_nodes(wc, g, Rc, tc)
            X = X[X[:, 2] > 0.05]
            u = np.rint(md["K"][0, 0] * X[:, 0] / X[:, 2] + md["K"][0, 2] - 0.5).astype(int)
            v = np.rint(md["K"][1, 1] * X[:, 1] / X[:, 2] + md["K"][1, 2] - 0.5).astype(int)
            ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            sil = np.zeros((H, W), bool)
            sil[v[ins], u[ins]] = True
            from scipy.ndimage import binary_dilation

            sil = binary_dilation(sil, iterations=int(self.p["track_dilate_px"]) + 2)
            zpred = np.full((H, W), np.nan)
            zpred[v[ins], u[ins]] = X[ins, 2]
            from scipy.ndimage import grey_dilation

            zp = -grey_dilation(np.where(np.isfinite(zpred), -zpred, -np.inf), size=7)
            mask = sil & np.isfinite(z_obs) & (np.abs(z_obs - np.where(np.isfinite(zp), zp, np.inf))
                                               <= self.p["track_gate_m"])
            rec = {"status": "predicted", "n_px": int(mask.sum())}
            if mask.sum() >= 50:
                tok = LY.token_for((msg["episode"], msg["look"], pid), [g])
                est = wc.layer.estimate_target(frame, ctx, LY.Target(obj_ids=(g,), pixels=mask, seed=tok, key=pid))
                if est.status == "ok" and self.p.get("turn_refine"):  # A64: as in the full look
                    tr = self.turn_refine(wc, g, est, frame)
                    if tr is not None:
                        est.R, est.t = tr.pop("R"), tr.pop("t")
                        rec["turn_refine"] = tr
                if est.status == "ok":
                    fc = self.fit_check(wc, g, est.R, est.t, z_obs, md["K"])
                    jump = float(np.linalg.norm(np.asarray(est.t) - tc))
                    rec.update(fit=fc, jump_m=round(jump, 4), score=float(est.score))
                    if fc["inlier"] >= self.p["accept_inlier"] and fc["n_agree"] >= self.p["accept_min_agree"] \
                            and jump <= self.p["track_max_jump_m"]:
                        Rw = T[:3, :3] @ np.asarray(est.R)
                        tw = T[:3, :3] @ np.asarray(est.t) + T[:3, 3]
                        rec.update(status="tracked", R_w=Rw.tolist(), t_w=tw.tolist())
            out[pid] = rec
        self.n_looks += 1
        return {"tracks": out, "times": {"total_s": round(time.perf_counter() - t0, 4)}}

    def place(self, msg: dict) -> dict:
        """ST-1 H8, the measured place check. The camera is fixed, so depth from before anything was placed (``base``) can
        be compared pixel by pixel with this look. Pixels that rose by ``rise_m`` inside the container's box, in the largest
        connected patch after a one-pixel opening, are the placed object. Robot pixels have no depth in either look."""
        from scipy.ndimage import binary_opening, label

        wc = self.cell(msg["suite"])
        fr = msg["cameras"][0]
        d2, _ = self._mask_depth(wc, fr, msg["joint_pos"], msg["gripper_qpos"])
        _, z, _, _, _, Pw = self._frame_arrays({**fr, "depth_u16": d2})
        if msg.get("base"):
            self._place_base = (msg["episode"], z)
            return {"ok": True}
        episode, z0 = self._place_base
        if episode != msg["episode"]:
            raise ValueError("place look without this episode's base")
        c, h = (np.asarray(msg["region"][k], float) for k in ("centre", "half"))
        box = (np.abs(Pw[..., :2] - c[:2]) <= h[:2]).all(-1) & (Pw[..., 2] > c[2] - h[2] - 0.01)
        both = np.isfinite(z) & np.isfinite(z0)
        rec = {"n_box": int((box & both).sum()), "by_rise": {}}
        for rise in sorted({float(msg["rise_m"]), 0.008, 0.02}):
            rose = binary_opening(box & both & (z0 - z >= rise))
            lab, n = label(rose)
            sizes = np.bincount(lab.ravel())[1:] if n else np.zeros(0, int)
            rec["by_rise"][f"{rise:.3f}"] = int(sizes.max()) if n else 0
            if rise == float(msg["rise_m"]):
                rec["n_px"] = rec["by_rise"][f"{rise:.3f}"]
                if n:
                    P = Pw[lab == 1 + int(sizes.argmax())]
                    rec.update(xy=[round(float(x), 4) for x in np.median(P[:, :2], axis=0)], top_z=round(float(P[:, 2].max()), 4))
        return {"place": rec}

    # ------------------------------------------------------------------ requests
    def handle(self, msg: dict) -> dict:
        op = msg.get("op")
        c0 = time.process_time()
        if op == "hello":
            server = self.check_hello(msg)
            for s in msg.get("suites", []):
                self.cell(s)
            rep = {"ok": True, "format": FORMAT, "params": self.p, "server": server,
                   "setup_s": {s: c.setup_s for s, c in self.cells.items()},
                   "matchers": {s: c.matcher for s, c in self.cells.items()},  # amendment A39
                   "onboarding_sha256": {s: sha256_file(LB.lfb_dirs(self.p.get("lfb_library"))[0] / s / "onboarding.json")
                                         for s in self.cells}}
        elif op == "begin_episode":
            if self.p.get("learn_in_episode"):
                from rowant_lab.libero_env import lfb_episode as LE

                LE.end_episode(self)
            rep = {"ok": True, "episode": msg["episode"]}
        elif op == "look":
            mode = msg.get("mode", "full")
            scope = self.ep_scope.get(msg.get("episode")) if self.ep_scope else None
            saved = None
            if scope is not None:  # in-episode learning: this episode's memory and settings (lfb_episode.learn)
                saved = (self.p, self.cells)
                self.p, self.cells = scope["p"], {**self.cells, **scope["cells"]}
            try:
                rep = self.full(msg) if mode == "full" else self.place(msg) if mode == "place" else self.track(msg)
            finally:
                if saved is not None:
                    self.p, self.cells = saved
        elif op in ("learn_plan", "learn") and self.p.get("learn_in_episode"):
            from rowant_lab.libero_env import lfb_episode as LE

            rep = LE.learn_plan(self, msg) if op == "learn_plan" else LE.learn(self, msg)
        elif op == "arm_cam" and isinstance(self.p.get("arm_cam"), dict):  # item 2: the camera from the robot's own arm
            from rowant_lab.libero_env import arm_camera as AC

            rep = AC.handle(self, msg)
        elif op == "stats":
            rep = {"ok": True, "n_cells": len(self.cells)}
        elif op == "fm6_echo":  # FM6 (m2_fm6): the sha256 of the received frame bytes, and the arrays sent back
            fr = msg["cameras"][0]
            rgb, dep = np.asarray(fr["rgb"]), np.asarray(fr["depth_u16"])
            rep = {"ok": True, "sha256": {"rgb": hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
                                          "depth_u16": hashlib.sha256(np.ascontiguousarray(dep).tobytes()).hexdigest()},
                   "rgb": rgb, "depth_u16": dep}
        else:
            raise ValueError(f"unknown op {op!r}")
        rep["server_cpu_s"] = round(time.process_time() - c0, 4)
        return rep


def _real(p) -> str:
    return os.path.realpath(os.fspath(p))


def _under(p: str, roots) -> bool:
    return any(p == r or p.startswith(r.rstrip(os.sep) + os.sep) for r in roots)


def audit_roots(log_dir: Path, scratch: Path, extra_read=(), lfb_library=None, learn_dir=None) -> dict:
    """The server's allowlists (realpaths): reads, Python sources, writes and listings. ``learn_dir`` (in-episode
    learning only): that folder is read and written, and the library trees a variant is linked from are read."""
    onboard_dir, policy_lib, _m1 = LB.lfb_dirs(lfb_library)  # LFB-LIB: a variant tree replaces M2's own
    prefixes = sorted({_real(x) for x in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)})
    numba = os.environ.get("NUMBA_CACHE_DIR")
    # the temp folder: macOS's (the scored pods set TMPDIR inside it) and this machine's own
    tmp = sorted({_real("/private/var/folders"), _real(os.environ.get("TMPDIR") or "/tmp")})
    # the library: the onboarded artifacts and the policy-v2 version only (never the export's raw type files)
    read = [_real(onboard_dir), _real(policy_lib)] + prefixes + tmp + ([_real(numba)] if numba else [])
    panda = [_real(MP.EXPORT_DIR / f"panda-{s}.{e}") for s in MP.SUITES for e in ("json", "npz")]
    read_files = ["/dev/null", "/dev/urandom", "/dev/random"] + panda + [_real(x) for x in extra_read]
    code = [_real(MP.REPO_ROOT / "src"), _real(MP.REPO_ROOT / "rowant_lab")]
    write = [_real(log_dir), _real(scratch)] + tmp + ([_real(numba)] if numba else [])
    listing = read + code + [_real(MP.REPO_ROOT), _real(scratch), _real(log_dir)]
    if learn_dir is not None:
        src = [_real(x) for x in (_m1, MP.ONBOARD_DIR, MP.POLICY_LIB, MP.M1_LIBRARY)]
        read += [_real(learn_dir)] + src
        write += [_real(learn_dir)]
        listing += [_real(learn_dir)] + src
    return {"read": read, "read_files": read_files, "code": code, "write": write, "write_files": ["/dev/null"],
            "list": listing}


GUARD_COMMANDS = ()  # as run, a machine guard's four read-only commands were allowed here; the guard is not part of
# this release (hybrid_frontier/e9/isolation.py), so the server may start no subprocess at all


def guard_command(argv_) -> bool:
    """The subprocesses the server may start: none."""
    if not argv_:
        return False
    base = os.path.basename(os.fsdecode(argv_[0]) if isinstance(argv_[0], bytes) else str(argv_[0]))
    return base in GUARD_COMMANDS


def install_audit_hook(log_dir: Path, scratch: Path, extra_read=(), lfb_library=None, learn_dir=None) -> dict:
    """The server's audit hook (plan venv_bridge.server; verifier minor M1). It refuses:

    * a file READ outside the M2 library (onboarded artifacts, policy-v2, the Panda export), Python's own installation
      (``sys.prefix``, ``base_prefix``, ``exec_prefix``), the numba cache and the temp folder, except the worktree's
      own ``.py`` / ``.pyc`` sources (imports) and the files in ``extra_read`` -- so a run folder, M1's ``scenes/``,
      the init states, the evidence folders and the GATE files are all refused;
    * a WRITE outside the server's own log folder, the scratch folder, the numba cache and the temp folder;
    * a folder listing outside those roots; a rename, removal, chmod or mkdir outside the write roots;
    * any subprocess, ``os.system`` or exec; any non-AF_UNIX socket and any name lookup.
    """
    import socket as _socket

    roots = audit_roots(log_dir, scratch, extra_read, lfb_library, learn_dir)
    wflags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def path_of(x):
        if isinstance(x, int) or x is None:
            return None
        if isinstance(x, bytes):
            x = os.fsdecode(x)
        return _real(x)

    def read_ok(p: str) -> bool:
        return (p in roots["read_files"] or _under(p, roots["read"])
                or (_under(p, roots["code"]) and p.endswith((".py", ".pyc"))))

    def write_ok(p: str) -> bool:
        return p in roots["write_files"] or _under(p, roots["write"])

    def hook(event, args):
        if event == "open":
            p = path_of(args[0]) if args else None
            if p is None:
                return
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ""
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            writing = any(ch in mode for ch in "wax+") or bool(flags & wflags)
            if writing:
                if not write_ok(p):
                    raise PermissionError(f"m2 server: write refused: {p}")
            elif not read_ok(p):
                raise PermissionError(f"m2 server: read refused: {p}")
        elif event in ("os.listdir", "os.scandir", "glob.glob", "glob.glob/2"):
            p = path_of(args[0]) if args else None
            if p is None and event in ("os.listdir", "os.scandir"):
                p = _real(".")
            if p is not None and event.startswith("glob"):
                p = _real(os.path.dirname(p) or ".")
            if p is not None and not _under(p, roots["list"]):
                raise PermissionError(f"m2 server: listing refused: {p}")
        elif learn_dir is not None and event in ("os.link", "shutil.copyfile") and len(args) >= 2:
            # in-episode learning: a variant links or copies FROM a readable library file INTO a write root
            src, dst = path_of(args[0]), path_of(args[1])
            if src is None or dst is None or not read_ok(src) or not write_ok(dst):
                raise PermissionError(f"m2 server: {event} refused: {src} -> {dst}")
        elif event in ("os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.truncate", "os.utime",
                       "os.link", "os.symlink", "shutil.rmtree", "shutil.copyfile", "shutil.move"):
            p = path_of(args[0]) if args else None
            if p is not None and not write_ok(p):
                raise PermissionError(f"m2 server: {event} refused: {p}")
        elif event == "os.rename" and len(args) >= 2:
            for a in args[:2]:
                p = path_of(a)
                if p is not None and not write_ok(p):
                    raise PermissionError(f"m2 server: rename refused: {p}")
        elif event in ("subprocess.Popen", "os.posix_spawn"):
            argv_ = list(args[1]) if len(args) > 1 and isinstance(args[1], (list, tuple)) else []
            if not guard_command(argv_):
                raise PermissionError(f"m2 server: {event} refused: {argv_[:3]}")
        elif event in ("os.system", "os.exec", "os.spawn", "os.fork", "os.forkpty", "pty.spawn", "webbrowser.open",
                       "urllib.Request"):
            raise PermissionError(f"m2 server: {event} refused")
        elif event in ("socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg") and args:
            sock = args[0]
            if getattr(sock, "family", None) != _socket.AF_UNIX:
                raise PermissionError("m2 server: only AF_UNIX sockets")
        elif event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr", "socket.getnameinfo"):
            raise PermissionError("m2 server: no name lookups")

    sys.addaudithook(hook)
    return roots


PREIMPORTS = ("rowant_lab.libero_env.m2_segment", "rowant_lab.fastlayer.frame", "rowant_lab.fastlayer.layer", "rowant_lab.fastlayer.segment",
              "rowant_lab.fastlayer.memory", "rowant_lab.fastlayer.symmetry", "rowant_lab.fastlayer.appearance",
              "rowant_lab.fastlayer.onboard", "rowant_lab.fastlayer.scorers.monty_colour",
              "rowant_lab.libero_env.m2_onboard", "scipy.spatial", "scipy.ndimage", "scipy.optimize")


def _visible_numbers(inner, Xw, Nw, fr, kw: dict) -> dict:
    """What ``refine.visible`` works on, evaluated a second time, and whether a second call raises again."""
    from rowant_lab.hybrid_frontier.screens.e0f_refine import spec as S

    bin_px = kw.get("bin_px", S.VIS_BIN_PX)
    cam = fr.cam
    u, v, z = cam.project(Xw)
    inside = (z > 0) & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
    nbu = cam.W // bin_px + 1
    nb = (cam.H // bin_px + 1) * nbu
    b = np.where(inside, np.floor(v / bin_px) * nbu + np.floor(u / bin_px), 0).astype(np.int64)
    bad = np.flatnonzero(inside & ((b >= nb) | (b < 0)))
    try:
        inner(Xw, Nw, fr, **kw)
        again = False
    except IndexError:
        again = True
    return {"WH": [repr(cam.W), repr(cam.H)], "bin": repr(bin_px), "nb": int(nb), "n": int(len(Xw)),
            "dt": [str(np.asarray(Xw).dtype), str(u.dtype)], "in": int(inside.sum()), "bad": int(len(bad)),
            "again": again, "nan": int((~np.isfinite(u) | ~np.isfinite(v)).sum()),
            "f": [round(float(x), 2) for x in (cam.fx, cam.fy, cam.cx, cam.cy)],
            "uvzb": [[round(float(u[i]), 3), round(float(v[i]), 3), round(float(z[i]), 4), int(b[i])] for i in bad[:3]],
            "cpu": os.sched_getcpu() if hasattr(os, "sched_getcpu") else None, "np": np.__version__}


def watch_visible() -> None:
    """Record only (2026-11-18). The pose fitter's visibility buffer (``refine.visible``, a frozen M2 file that stays as
    it is) raised an IndexError in 12 rows on one pod host and nowhere else, and read by itself it cannot: it indexes
    only points it has just found inside the picture. When it raises, the numbers it worked on go into the error's own
    text (``| visible {...}``) and the same error is raised again. A call that does not raise is untouched."""
    from rowant_lab.hybrid_frontier.screens.e0f_refine import refine as RF

    inner = RF.visible
    if getattr(inner, "_watched", False):
        return

    def visible(Xw, Nw, fr, **kw):
        try:
            return inner(Xw, Nw, fr, **kw)
        except IndexError as exc:
            try:
                note = _visible_numbers(inner, Xw, Nw, fr, kw)
            except Exception as exc2:  # the record must never hide the error it records
                note = {"note_failed": f"{type(exc2).__name__}: {exc2}"[:120]}
            exc.args = (f"{exc} | visible {json.dumps(note, separators=(',', ':'))}",)
            raise

    visible._watched = True
    RF.visible = visible


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="p.sock")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--params", default=None)
    ap.add_argument("--suites", nargs="*", default=[])
    ap.add_argument("--policy-lib-sha256", default=None)
    ap.add_argument("--frozen-config-sha256", default=None)
    ap.add_argument("--expect-code", default=None, help="JSON file {path: sha256}: the frozen config's code_sha256")
    a = ap.parse_args(argv)
    import importlib

    from rowant_lab.hybrid_frontier.e9 import isolation

    iso = isolation.require()  # (a stand-in in this release: it inspects nothing)
    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    expect = None
    if a.expect_code:
        expect = json.loads(Path(a.expect_code).read_text())
    if (expect is None) != (a.frozen_config_sha256 is None):
        print(json.dumps({"refused": "a frozen run needs both --expect-code and --frozen-config-sha256"}), flush=True)
        return 5
    params = json.loads(a.params) if a.params else None
    extra = ()
    if (params or {}).get("monty_arm") == "minus":  # mminus_dev: scorer B reads the textures M2 never reads
        from rowant_lab.libero_env import m2_mminus as MM

        ob_dir = LB.lfb_dirs((params or {}).get("lfb_library"))[0]  # LFB-LIB
        metas = [json.loads(str(np.load(ob_dir / s / "memory.npz")["meta_json"])) for s in MP.SUITES
                 if (ob_dir / s / "memory.npz").exists()]
        extra = MM.texture_files(metas)
        for m in ("rowant_lab.libero_env.m2_mminus", "rowant_lab.fastlayer.scorers.render_compare", "skimage.color",
                  "PIL.Image"):
            importlib.import_module(m)
    learn_dir = None
    if (params or {}).get("learn_in_episode"):  # lfb_episode: the server's own learn folder, under its scratch
        from rowant_lab.libero_env import lfb_episode as LE

        learn_dir = LE.learn_dir()
        learn_dir.mkdir(parents=True, exist_ok=True)
        importlib.import_module("rowant_lab.libero_env.lfb_episode")
        if (params or {}).get("learn_wrist_stereo") is not None:  # wrist frames without depth: K0's matcher and its
            LE.preload_stereo()  # hand model, read before the hook is installed (an evidence file, outside its roots)
    roots = install_audit_hook(log_dir, Path.cwd(), extra, (params or {}).get("lfb_library"), learn_dir)
    watch_visible()  # record only: an index error of the pose fitter's buffer carries its numbers
    per = Perceiver(params, policy_lib_sha256=a.policy_lib_sha256,
                    frozen_config_sha256=a.frozen_config_sha256)
    for s in a.suites:
        per.cell(s)
    for m in PREIMPORTS:  # lazily imported modules, loaded now so the start-up check covers them
        importlib.import_module(m)
    try:
        per.integrity = INT.check("server", expect)
    except INT.IntegrityError as exc:
        (log_dir / f"server-modules-{os.getpid()}.json").write_text(json.dumps({"refused": str(exc)}))
        print(json.dumps({"refused": str(exc)}), flush=True)
        return 5
    if own_versions() != BR.EXPECTED_VERSIONS["monty"]:
        print(json.dumps({"refused": f"server versions {own_versions()}"}), flush=True)
        return 5
    logf = open(log_dir / f"server-{os.getpid()}.jsonl", "a", encoding="utf-8")

    def handler(msg):
        rep = per.handle(msg)
        if msg.get("op") == "look":
            logf.write(json.dumps({"episode": msg.get("episode"), "look": msg.get("look"), "mode": msg.get("mode"),
                                   "times": rep.get("times"), "cpu_s": rep.get("server_cpu_s")}) + "\n")
            logf.flush()
        return rep

    print(json.dumps({"ready": True, "pid": os.getpid(), "isolation_utc": iso.get("time_utc"),
                      "n_repo_modules": per.integrity["n_modules"], "frozen": expect is not None}), flush=True)
    BR.serve(a.socket, handler, max_clients=2, require_hello=True,
             log=lambda r: logf.write(json.dumps({"error": r["error"]}) + "\n"))
    logf.close()
    end = {"start": per.integrity, "late": INT.late(per.integrity), "hellos": per.hellos,
           "audit_roots": roots, "time_utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (log_dir / f"server-modules-{os.getpid()}.json").write_text(json.dumps(end, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
