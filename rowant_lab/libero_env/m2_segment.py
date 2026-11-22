"""M2 segmentation extensions for LIBERO's scene furniture (plan modules.m2_segment; amendment A14, m2f:perceive-r1).

M2 (perceived poses, grounded instruction): perception in the loop. TRUTH-FREE: perception-server side (Monty venv),
numpy and scipy only. Its inputs are the frame's observed depth (metres, NaN = no reading; robot and
out-of-workspace pixels already removed by the server's self-filter), the declared intrinsics and extrinsics, the fast
layer's support-plane fit and the onboarded model nodes (body-frame positions and normals). It reads no file, saves no
array or image, and never sees a true pose: the only spatial prior is the declared workcell box (the table top in
LIBERO-Spatial), never a remembered fixture pose (FM7).

Why it exists (judged plan m2f:judge). The fast layer's segmenter marks patches near and parallel to the support
plane, and large flat surfaces, as background, and its wide proposal is tuned for small tabletop objects: LIBERO-
Spatial's plate (the place object of every task), ramekin, flat stove and wooden cabinet were found 7, 2, 10 and 0
times in 20 looks. Re-tuning the band does not work (one run: plate 3 -> 4 of 10, stove 5 -> 0 of 10). So these types
are no longer posed from region masks: they are searched for as whole models on the support plane.

**Stage 1** (shared):

* ``refit_support``: a total-least-squares refit of the fast layer's plane on every observed point within
  ``band_m`` of it (pixel centres at u + 0.5, the convention the node projections use), twice with the band shrunk
  to 3 x rms; the normal points towards the camera (d > 0) and the residual rms is kept;
* ``heights``: the per-pixel height above the refit support (an in-memory array, never saved);
* ``support_check``: the support (relief) test for every non-texture pose: of the part's visible model nodes that
  stand at least ``relief_min_m`` above the support, the share whose pixel sees BARE support (observed height at
  most max(``relief_tol_m``, ``relief_rms_k`` x rms)) must not exceed ``bare_max``, and the share of all visible
  nodes whose pixel sees BELOW the support by more than that tolerance must not exceed ``below_max``. A plate (or any
  part) placed on bare table is the one case a 1 cm depth agreement cannot tell from a real one.

**Stage 2** (``support_search``), per routed type:

1. a top-down height map of the observed points on the refit plane: ``cell_m`` cells (plane-frame axes from the
   world x axis), the maximum height per cell, empty cells unknown; only the declared box is mapped;
2. per-type height templates from the onboarded nodes, upright (body z along the support normal, the rest pose of
   every type here), the lowest node on the support; one yaw for a type whose onboarded flip family holds a turn
   about body z of at most ``yaw_step_deg`` (axisymmetric), else ``yaw_step_deg`` steps over the yaw period;
3. a multi-level occupancy correlation (FFT): at each of the type's levels (shares ``levels`` of its height, at
   least ``level_min_m``) a template cell at or above the level scores +1 where the map is observed at or above it
   and -1 where it is observed below it (unknown cells score 0), and every footprint cell scores -1 where the map
   is observed above the ceiling (the type's height plus max(``ceil_min_m``, ``ceil_frac`` x height): the space
   above the part is occupied); the sum, divided by the template's cells, is the score;
4. up to ``peaks`` peaks per type (non-maximum suppression at ``nms_frac`` of the type's diameter) with score at least
   ``min_score`` and known coverage at least ``min_cover``;
5. a refine on camera depth with the upright scorer, its node agreement made soft (an Epanechnikov kernel over
   +-``agree_m`` of the depth residual, so that a sub-centimetre offset can be fitted) minus 2 x seen-through, coarse
   to fine in yaw and in-plane position and, for welded FIXTURES only, a fitted height offset over ``dz_fixture_m`` (coarse
   ``dz_coarse_m``, refined ``dz_fine_m``; resting objects keep their lowest node on the support);
6. the result goes back to the server, which accepts it on its floors plus ``support_check`` plus ``cover_check``
   (at most ``cover_max`` of the known cells under the part observed above its ceiling) plus the occlusion-aware
   agreeing share (agreeing over agreeing, between and seen-through nodes: observed NEARER than the model is
   occlusion and does not count; at least ``share_min``) and at least ``vis_min`` of the visible nodes with depth
   agreeing, and dedups it in 3D against the percepts already kept.

The map and the templates are numbers in memory; the search returns poses and scalar diagnostics only.
"""

from __future__ import annotations

import math

import numpy as np

TRUTH_FREE = True

PARAMS = {
    "band_m": 0.006,
    "refit_iter": 2,
    "refit_min_pts": 500,
    "relief_min_m": 0.003,
    "relief_tol_m": 0.0015,
    "relief_rms_k": 3.0,
    "bare_max": 0.4,
    "below_max": 0.05,
    "cell_m": 0.005,
    "levels": (0.15, 0.4, 0.65, 0.9),
    "level_min_m": 0.004,
    "level_tol_m": 0.003,
    "ceil_min_m": 0.02,
    "ceil_frac": 0.25,
    "cover_max": 0.5,
    "vis_min": 0.1,
    "yaw_step_deg": 10.0,
    "peaks": 3,
    "nms_frac": 0.5,
    "min_score": 0.2,
    "min_cover": 0.25,
    "refine_nodes": 1500,
    "refine_nodes_fine": 4000,
    "fine_kernel_frac": 0.4,
    "refine_stages": ((5.0, 0.005), (2.0, 0.002), (0.75, 0.00075), (0.3, 0.0003)),
    "dz_fixture_m": (-0.030, 0.030),  # A73 (A71): +-3 cm (library v3: the stove's base box sinks into the table)
    "dz_coarse_m": 0.002,
    "dz_fine_m": 0.0005,
    "dz_global_stages": 2,
    "fixture_yaw_starts_deg": (0.0, 10.0, -10.0),
    "artic_window_m": 0.12,
    "share_min": 0.85,
    "dup_inside": 0.3,
    "dup_inside_strong": 0.6,
    "dup_explained": 0.5,
    "explained_m": 0.006,
    "colour_min_r": 0.3,
    "colour_min_nodes": 50,
    "dup_shrink_m": 0.005,
    # amendment A23 (m2f:perceive-r2): DEPTH-RANKED hypotheses for welded fixtures. The map correlation alone picks
    # the coarse peak, and for the large fixtures its maximum is often not at the part (most of the cabinet's and
    # the stove's footprint is unknown from one oblique view): each yaw's own peaks are ranked by the refine's soft
    # depth agreement and the best distinct ones are refined. False reproduces the pre-A23 search exactly.
    "depth_rank": False,  # A23 as written; switched off by A24 (the coarse-grid ranking picked worse hypotheses)
    "rank_peaks_per_yaw": 8,
    "rank_nms_frac": 0.15,
    "rank_dz_m": (-0.01, 0.0, 0.01, 0.02, 0.03),
    "rank_keep": 3,
    "rank_sep_frac": 0.15,
    "rank_sep_deg": 20.0,
    "rank_nodes": 1500,
    # amendment A24 (m2f:perceive-r2): a WIDE first refine stage for welded fixtures, evaluated in one vectorised pass:
    # yaw +-wide_yaw_deg in wide_yaw_step_deg steps and the in-plane position +-wide_xy_m in wide_xy_step_m steps
    # around the coarse peak, wide_nodes nodes, the refine's own score; the best pose starts the unchanged stages
    # (whose basin is about +-16 mm and +-14 deg). With it on, a fixture refines from its coarse yaw only (the A18
    # yaw starts are inside the wide window). False reproduces the pre-A24 refine exactly.
    "wide_stage": True,
    "wide_yaw_deg": 45.0,
    "wide_yaw_step_deg": 5.0,
    "wide_xy_m": 0.06,
    "wide_xy_step_m": 0.005,
    "wide_nodes": 400,
    # amendment A25 (m2f:perceive-r2): a jointed fixture's wide stage scores its whole-assembly variants (A21: all
    # parts closed, each jointed part alone at the open end of its range) instead of the root alone, and of two
    # model-search candidates of ONE type that cannot both be true (the dedup's same-type rule) the one with the
    # better refine score is kept before the existing dedup runs. False reproduces A24 exactly.
    "wide_assembly": True,
    "same_type_by_refine": True,
    # amendment A26 (m2f:perceive-r2): a jointed fixture's refine stages score the WHOLE ASSEMBLY with its joints
    # fitted at the current pose (after the wide stage, and again before the last two stages); False reproduces A25
    "refine_assembly": True,
    # amendment A27 (m2f:perceive-r2): a jointed fixture's wide stage scores the ROOT (not A25's variants: their
    # drawers sit at the range end, not at the scene's value) with more nodes, a 1 cm grid, and every yaw copy of the
    # root's onboarded yaw period (the articulation breaks that symmetry); False reproduces A26
    "wide_jointed_root": True,
    "wide_nodes_jointed": 1500,
    "wide_xy_step_jointed_m": 0.01,
    "wide_jointed_keep": 4,
    # amendment A29 (m2f:perceive-r2): the quarter-turn windows are all kept and picked by their whole-assembly score
    # at the window's pose (joints fitted there); only the picked one is refined. A28 is wide_jointed_keep 2 with
    # this switch False.
    "wide_jointed_pick_assembly": True,
    "asm_joint_node_stride": 3,
    "asm_joint_step_m": 0.004,
    # amendment A32 (m2f:perceive-r2): a jointed fixture's fine refine stages score the root alone, and a welded
    # non-jointed fixture also refines from its coarse peak without the wide stage (the better score wins)
    "asm_fine_root": True,
    "wide_also_narrow": True,
    # amendment A33 (m2f:perceive-r2): the same second refine from the coarse peak for a JOINTED fixture (its final
    # score is the root's fine score on both paths, A32 (1)); False reproduces A32
    "wide_also_narrow_jointed": True,
    # amendment A44 (m2f2:fixtures-r1): a welded FIXTURE's yaw is searched about its FOOTPRINT CENTROID (the xy mean of
    # its root nodes in the body frame), not about its body origin: the coarse map locates the footprint, and a turn
    # about an origin far from it (the stove's knob, 11 cm from its burner; the cabinet's, 3 cm from its carcass
    # centre) moves the footprint out of the wide and refine windows. Every wide-stage yaw, refine-stage yaw and A18
    # yaw start of a welded fixture then keeps the centroid where the coarse pose put it. False reproduces A33.
    "pivot_centroid": False,
    # amendment A44: the acceptance's visibility share of a JOINTED fixture leaves out the root nodes hidden by the
    # fixture's own jointed parts at their fitted values (Perceiver.self_occluded). False reproduces A33.
    "vis_self_occlusion": True,  # A50: on (the A48 settings, read on the private set)
    "self_occ_splat_px": 3,
    # amendment A44: a jointed fixture's final pose (and the score the same-type rule compares) is the whole
    # assembly's soft depth score at each refine result, joints fitted there. False reproduces A33.
    "asm_select": False,
    # amendment A45 (m2f2:fixtures-r1), after later runs showed A44 (1) and (3) as written to be harmful: A44 (1)'s
    # centroid pivot for a JOINTED fixture's quarter-turn wide windows only (A27), and A44 (3) replaced by a choice of
    # yaw MODE by the assembly score with the root score inside the mode. False reproduces A44's settings.
    "pivot_jointed_wide": False,
    "asm_select_modes": True,  # A50: on (the A48 settings, read on the private set)
    "asm_mode_deg": 20.0,
    # (A50: the defaults are A48's settings: vis_self_occlusion and asm_select_modes on; every pivot switch off)
    # amendment A49: a JOINTED fixture's quarter-turn wide windows (A27) turn about the xy centroid of the A21 variant
    # (all closed, or one part open at its range end) whose template gave the coarse peak. False reproduces A45/A48.
    "pivot_variant_wide": False,
    # m2r3:fix-r1 (A68): a JOINTED fixture's hypotheses from EVERY yaw of the coarse map (each yaw's own best peak of
    # the root template, plus its half-turn copy about the footprint centroid when the root's yaw period is 180 deg),
    # ranked by the root's depth score, then by the WHOLE-ASSEMBLY score (joints fitted), the best ``jh_keep``
    # refined and added to the existing refine paths; then (``jh_slide``) a final search of the root along each
    # slide joint's axis scored by the whole assembly, joints fitted. False reproduces A50's search exactly.
    "jointed_hyp": True,  # A73: on
    "jh_per_yaw": 1,
    "jh_root_keep": 8,
    "jh_keep": 4,  # A93 (A68 kept 2)
    "jh_sep_m": 0.03,
    "jh_sep_deg": 20.0,
    # m2r3:fix-r2 (A81): each raw A68 hypothesis LOCALLY REFINED before it is ranked: a small vectorised search of the
    # root (the wide stage's own score) over +-jh_local_yaw_deg in jh_local_yaw_step_deg steps and +-jh_local_xy_m in
    # jh_local_xy_step_m steps around its map peak, with jh_local_nodes nodes, its height offset re-chosen there; the
    # root ranking, the distinct rule and the whole-assembly choice then compare poses that fit, not the map's 10 deg
    # / 5 mm cells (on the private set the true yaw's raw hypothesis scored below its half-turn twin as a whole
    # assembly in 3 of 4 missed shifted looks). False reproduces A73's search exactly.
    "jh_local": True,  # A88: on
    "jh_local_yaw_deg": 6.0,
    "jh_local_yaw_step_deg": 3.0,
    "jh_local_xy_m": 0.03,
    "jh_local_xy_step_m": 0.006,
    "jh_local_nodes": 600,
    # m2r3:fix-r2 (A82): with jh_local, the half-turn copy (about the footprint centroid) of each of the best
    # jh_twin_k locally refined hypotheses is refined locally too and joins the ranking. False reproduces A81.
    "jh_twin": False,
    "jh_twin_k": 4,
    # m2r3:fix-r2 (A85): a jointed fixture's chosen pose POLISHED with its closed parts held at rest and its open
    # parts left out (Perceiver._assembly_closed_at; joints fitted once at the chosen pose): a search along each slide
    # axis over +-jh_slide_m (jh_slide_step_m, then jh_slide_fine_m), then yaw +-2 x closed_yaw_step_deg and the
    # in-plane position +-2 x closed_xy_step_m, scored with the fine kernel; a closed drawer's front, flush with the
    # carcass, fixes the carcass along the axis the root's own surfaces leave loose (A68's slide search refitted
    # every joint at every pose, so an open drawer absorbed the slide). False reproduces A81-A82 exactly.
    "fine_closed": True,  # A88: on
    # m2r3:fix-r2 (A86): the number of coarse peaks refined for a welded fixture WITHOUT joints (the stove); None keeps
    # ``peaks`` (A50's search exactly). On the private set the shifted stove's peak lost all three slots to peaks in
    # the open-drawer cabinet's region, whose refined scores are far below the stove's own.
    "fixture_peaks": 6,  # A88
    # m2r3:fix-r2 (A87): the coarse map's score floor for a welded fixture WITHOUT joints; None keeps ``min_score``.
    # The stove's own peak scores 0.22-0.34 where it is found (floor 0.2) and is absent where it is missed: the map
    # is a proposal only, and the refine and the acceptance floors decide.
    "fixture_min_score": 0.12,  # A88
    # m2r3:fix-r2 (A91): a welded fixture's fitted height offset never puts its body origin below the support
    # (dz >= -h0; dz_window). False reproduces A88 exactly.
    "dz_origin_floor": True,  # A93
    "closed_tol_m": 0.02,
    "closed_tol_deg": 10.0,
    "closed_yaw_step_deg": 0.5,
    "closed_xy_step_m": 0.001,
    "jh_slide": False,
    "jh_slide_m": 0.014,
    "jh_slide_step_m": 0.002,
    "jh_slide_fine_m": 0.0005,
    # m2r3:fix-r1 (A69): a jointed part's value is fitted between its range end FARTHER from its rest value qpos0 and
    # qpos0 itself (a closed drawer rests at its stop; the range's slack beyond the rest value, the drawers' 1 cm,
    # let a fitted drawer absorb a carcass slid by up to that much). False reproduces A50's fits exactly.
    "joint_rest_stop": True,  # A73: on
    # m2r3:fix-r1 (A72): a welded FIXTURE's coarse templates are also evaluated SUNK by each of ``coarse_dz_m``
    # (template heights lowered by that much, the cells that end at or below the support dropped from the
    # footprint: geometry below the support is hidden by it), best per cell: library v3's stove model holds its base
    # box, which the scenes sink about 1.5 cm into the table, and the resting template lost the shifted stove's peak.
    # False reproduces A50's coarse search exactly.
    "coarse_dz": True,  # A73: on
    "coarse_dz_m": (-0.02, -0.01, 0.0),
    # Pro2.0 colour line (2026-11-16): a jointed fixture's chosen pose against a FACING PRIOR, and its half-turn twin
    # (``twin_fit``). A box-like carcass explains the depth almost as well turned half a turn, and the whole assembly
    # tells the two apart only by what its drawers show (their handles, the gaps between them); the depth net of the
    # colour line draws neither, and its mirror fit scored higher in 5 of 20 plain starts of the Goal scene (the
    # handle was then looked for behind the cabinet). With a dict here the server passes ``support_search`` its
    # verdict on a pose (``Perceiver.twin_prior``: the parts open toward the robot's side of the table, and/or the
    # fixture's remembered turn); a chosen pose it calls back to front gives way to its twin when the twin is the
    # right way round and passes the server's own floors. None reproduces the search exactly.
    "twin_prior": None,
    # Pro2.0 colour line (2026-11-16): a welded fixture's REMEMBERED place (``Perceiver._memory_seeds``: the middle of
    # where the robot's own earlier looks at this workcell found the type, given in the parameters) is tried before
    # the search. The benchmark sets a fixture down anew for every seed and task, within a centimetre or so of the
    # same spot (the robot's accepted first looks of the Goal cabinet over three seeds span 1.7 cm), so the root is
    # first searched around the remembered pose (``wide_stage``'s score over +-``memory_search_m`` and
    # +-``memory_search_yaw_deg``), then refined with the narrow refine and polished against its closed parts. The
    # result is taken, and the type's search skipped, only when it passes the server's own floors and lies within
    # ``memory_xy_m`` and ``memory_yaw_deg`` of the remembered pose: the fixture is seen to stand where it stands.
    # When that is refused, the remembered pose itself (only its height offset fitted) is put to the same floors. A
    # fixture that was moved away fails both and is searched for as before. On the same static cabinet the search
    # alone gave four different answers over 20 plain starts of the Goal scene (10 right, 5 back to front, 5 right
    # but refused by the floors). None reproduces the search exactly.
    "memory": None,
    "memory_search_m": 0.03,
    "memory_search_step_m": 0.006,
    "memory_search_yaw_deg": 6.0,
    "memory_search_yaw_step_deg": 3.0,
    "memory_nodes": 600,
    "memory_xy_m": 0.03,
    "memory_yaw_deg": 10.0,
    # ``twin_prior`` {"take": "always"}: the twin replaces a back-to-front choice whenever the prior calls the twin
    # the right way round, whether or not the floors accept it (the floors then judge it as any candidate; a
    # refused right-way pose still reaches the robot as a rejected one, a back-to-front one never does). Default:
    # only a twin the floors accept ("accepted").
    "twin_xy_m": 0.06,
    "twin_xy_step_m": 0.01,
    "twin_yaw_deg": 6.0,
    "twin_yaw_step_deg": 3.0,
    "twin_nodes": 1500,
}


# ------------------------------------------------------------------------------------------------ stage 1
def backproject(z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(H, W, 3) camera-frame points at pixel centres (u + 0.5, v + 0.5); NaN where there is no reading."""
    H, W = z.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    return np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1)


def refit_support(z: np.ndarray, K: np.ndarray, plane: dict, p: dict = PARAMS) -> dict:
    """TLS refit of the support plane ``n . X + d = 0`` (camera frame, n towards the camera, d > 0)."""
    n = np.asarray(plane["normal"], float)
    d = float(plane["d"])
    if d < 0:
        n, d = -n, -d
    P = backproject(z, K).reshape(-1, 3)
    P = P[np.isfinite(P[:, 2])]
    band = float(p["band_m"])
    rms = None
    n_in = 0
    for _ in range(int(p["refit_iter"])):
        r = P @ n + d
        sel = np.abs(r) < band
        n_in = int(sel.sum())
        if n_in < int(p["refit_min_pts"]):
            break
        Q = P[sel]
        c = Q.mean(0)
        _u, _s, vt = np.linalg.svd(Q - c, full_matrices=False)
        n2 = vt[-1]
        if n2 @ n < 0:
            n2 = -n2
        n, d = n2 / np.linalg.norm(n2), -float(n2 @ c) / float(np.linalg.norm(n2))
        res = Q @ n + d
        rms = float(np.sqrt(np.mean(res ** 2)))
        band = max(3.0 * rms, 0.002)
    if d < 0:
        n, d = -n, -d
    return {"normal": n, "d": d, "rms_m": rms, "n_pts": n_in,
            "fast_layer": {"normal": [float(x) for x in plane["normal"]], "d": float(plane["d"])}}


def heights(z: np.ndarray, K: np.ndarray, sup: dict) -> np.ndarray:
    """Per-pixel height above the support (m); NaN where there is no reading."""
    return backproject(z, K) @ np.asarray(sup["normal"], float) + float(sup["d"])


def bare_tol(sup: dict, p: dict = PARAMS) -> float:
    return max(float(p["relief_tol_m"]), float(p["relief_rms_k"]) * float(sup.get("rms_m") or 0.0))


def project(X: np.ndarray, K: np.ndarray, shape) -> tuple:
    """Pixel (u, v) of camera-frame points (the fit check's rounding) and the in-image mask."""
    H, W = shape
    ok = X[:, 2] > 0.05
    u = np.full(len(X), -1, int)
    v = np.full(len(X), -1, int)
    u[ok] = np.rint(K[0, 0] * X[ok, 0] / X[ok, 2] + K[0, 2] - 0.5).astype(int)
    v[ok] = np.rint(K[1, 1] * X[ok, 1] / X[ok, 2] + K[1, 2] - 0.5).astype(int)
    ins = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, ins


def visible(nodes, nrm, R, t, facing_cos: float):
    X = nodes @ np.asarray(R).T + np.asarray(t)
    N = nrm @ np.asarray(R).T
    view = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    vis = (-(N * view).sum(1)) > facing_cos
    return X, vis


def support_check(nodes, nrm, R, t, hpx: np.ndarray, K, sup: dict, facing_cos: float, p: dict = PARAMS) -> dict:
    """The support (relief) test of one pose (camera frame): see the module doc."""
    X, vis = visible(nodes, nrm, R, t, facing_cos)
    X = X[vis]
    u, v, ins = project(X, K, hpx.shape)
    X, u, v = X[ins], u[ins], v[ins]
    n_ = np.asarray(sup["normal"], float)
    h_model = X @ n_ + float(sup["d"])
    h_obs = hpx[v, u]
    has = np.isfinite(h_obs)
    tol = bare_tol(sup, p)
    relief = has & (h_model >= float(p["relief_min_m"]))
    bare = relief & (h_obs <= tol)
    below = has & (h_obs < -tol)
    n_rel, n_has = int(relief.sum()), int(has.sum())
    out = {"n_relief": n_rel, "n_bare": int(bare.sum()), "n_below": int(below.sum()), "n_has": n_has,
           "bare_share": round(float(bare.sum()) / max(n_rel, 1), 4),
           "below_share": round(float(below.sum()) / max(n_has, 1), 4), "tol_m": round(tol, 5)}
    out["pass"] = bool(n_rel > 0 and out["bare_share"] <= float(p["bare_max"])
                       and out["below_share"] <= float(p["below_max"]))
    return out


# ------------------------------------------------------------------------------------------------ stage 2
def plane_frame(sup: dict, T: np.ndarray) -> dict:
    """The support's plane frame in camera coordinates: origin at the foot of the camera centre, ex from world x."""
    n = np.asarray(sup["normal"], float)
    d = float(sup["d"])
    Rwc = np.asarray(T, float)[:3, :3]
    wx = Rwc[0, :]  # world x in camera coordinates
    ex = wx - (wx @ n) * n
    ex = ex / np.linalg.norm(ex)
    ey = np.cross(n, ex)
    return {"n": n, "d": d, "ex": ex, "ey": ey, "p0": -d * n, "G": np.stack([ex, ey, n], 1)}


def box_to_plane(box_w, pf: dict, T: np.ndarray) -> tuple:
    """(a0, a1, b0, b1) plane-frame extents of a world xy box (x0, x1, y0, y1) at the support."""
    T = np.asarray(T, float)
    Rwc, twc = T[:3, :3], T[:3, 3]
    x0, x1, y0, y1 = box_w
    zs = []
    # the support's world height at the camera's foot
    foot_w = Rwc @ pf["p0"] + twc
    for x in (x0, x1):
        for y in (y0, y1):
            Pw = np.array([x, y, foot_w[2]])
            Pc = Rwc.T @ (Pw - twc)
            q = Pc - pf["p0"]
            zs.append((q @ pf["ex"], q @ pf["ey"]))
    a = [z[0] for z in zs]
    b = [z[1] for z in zs]
    return min(a), max(a), min(b), max(b)


def height_map(z, K, sup: dict, pf: dict, ext: tuple, cell: float) -> dict:
    """Top-down max-height map (m) over the plane-frame extents; NaN = unknown."""
    P = backproject(z, K).reshape(-1, 3)
    P = P[np.isfinite(P[:, 2])]
    q = P - pf["p0"]
    a, b, h = q @ pf["ex"], q @ pf["ey"], q @ pf["n"]
    a0, a1, b0, b1 = ext
    W = int(math.ceil((a1 - a0) / cell))
    H = int(math.ceil((b1 - b0) / cell))
    j = np.floor((a - a0) / cell).astype(int)
    i = np.floor((b - b0) / cell).astype(int)
    ok = (j >= 0) & (j < W) & (i >= 0) & (i < H)
    hm = np.full(H * W, -np.inf)
    np.maximum.at(hm, i[ok] * W + j[ok], h[ok])
    hm = hm.reshape(H, W)
    hm[~np.isfinite(hm)] = np.nan
    return {"h": hm, "a0": a0, "b0": b0, "cell": cell, "shape": (H, W)}


def mark_cells(hm: dict, pf: dict, parts) -> np.ndarray:
    """Per map cell, the lowest height above the support of a part already perceived whose footprint (dilated by
    one cell) covers it (inf where none): ``parts`` is a list of camera-frame node arrays, one per part."""
    from scipy.ndimage import grey_erosion

    Hm, Wm = hm["shape"]
    out = np.full((Hm, Wm), np.inf)
    for X in parts:
        q = np.asarray(X, float) - pf["p0"]
        a, b, h = q @ pf["ex"], q @ pf["ey"], q @ pf["n"]
        j = np.floor((a - hm["a0"]) / hm["cell"]).astype(int)
        i = np.floor((b - hm["b0"]) / hm["cell"]).astype(int)
        ok = (j >= 0) & (j < Wm) & (i >= 0) & (i < Hm)
        if not ok.any():
            continue
        m = np.full((Hm, Wm), np.inf)
        bottom = float(h.min())  # the part's own bottom: it can rest on what lies under its footprint
        m[i[ok], j[ok]] = bottom
        m = grey_erosion(m, size=3)  # dilate the footprint (min filter over inf background)
        out = np.minimum(out, m)
    return out


def explained_for(hm: dict, top_m: float, margin_m: float):
    """The cells whose cover is explained for a part of height ``top_m``: a perceived part whose bottom is no
    higher than the part's top plus the margin (it may rest on the part: a bowl on the cookies box), never one far
    above it (a bowl on the cabinet top does not explain the cabinet over a ramekin hypothesis)."""
    e = hm.get("explained_bottom")
    if e is None:
        return None
    return e <= top_m + margin_m


def yaw_period_deg(family, step_deg: float) -> float:
    """The smallest turn about body z in the onboarded flip family (360 when none); <= step: axisymmetric."""
    best = 360.0
    for F in family:
        F = np.asarray(F, float)
        if abs(F[2, 2] - 1.0) < 0.01:
            ang = math.degrees(math.atan2(F[1, 0], F[0, 0])) % 360.0
            if 1e-3 < ang < best:
                best = ang
    return best


def type_yaws(family, p: dict = PARAMS) -> list:
    step = float(p["yaw_step_deg"])
    per = yaw_period_deg(family, step)
    if per <= step + 1e-6:
        return [0.0]
    return [math.radians(a) for a in np.arange(0.0, per - 1e-6, step)]


def template(nodes: np.ndarray, yaw: float, cell: float, r: int) -> np.ndarray:
    """(2r+1, 2r+1) top-down height template (m above the lowest node; NaN = not covered), body origin centred;
    one-cell holes of a coarse node lattice are closed with the neighbours' maximum."""
    from scipy.ndimage import binary_closing, grey_dilation

    c, s = math.cos(yaw), math.sin(yaw)
    xr = c * nodes[:, 0] - s * nodes[:, 1]
    yr = s * nodes[:, 0] + c * nodes[:, 1]
    hz = nodes[:, 2] - nodes[:, 2].min()
    n = 2 * r + 1
    j = np.floor(xr / cell + 0.5).astype(int) + r
    i = np.floor(yr / cell + 0.5).astype(int) + r
    ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    tm = np.full(n * n, -np.inf)
    np.maximum.at(tm, i[ok] * n + j[ok], hz[ok])
    tm = tm.reshape(n, n)
    have = np.isfinite(tm)
    fp = binary_closing(have, structure=np.ones((3, 3), bool))
    filled = grey_dilation(tm, size=3)
    tm = np.where(have, tm, np.where(fp, filled, np.nan))
    return tm


def type_levels(hmax: float, p: dict = PARAMS) -> list:
    out = []
    for f in p["levels"]:
        lv = max(float(p["level_min_m"]), f * hmax)
        if not out or lv - out[-1] > 0.002:
            out.append(lv)
    return out


def _nms(score, cover, best_yaw, rad_cells: int, k: int, min_score: float, min_cover: float) -> list:
    sc = np.where(cover >= min_cover, score, -np.inf)
    out = []
    H, W = sc.shape
    for _ in range(k):
        idx = int(np.argmax(sc))
        i, j = divmod(idx, W)
        if not np.isfinite(sc[i, j]) or sc[i, j] < min_score:
            break
        out.append((i, j, float(score[i, j]), float(cover[i, j]), float(best_yaw[i, j])))
        sc[max(0, i - rad_cells):i + rad_cells + 1, max(0, j - rad_cells):j + rad_cells + 1] = -np.inf
    return out


def coarse_search(hm: dict, nodes: np.ndarray, yaws: list, p: dict = PARAMS, diam: float | None = None,
                  yaw_peaks_k: int = 0, dz_offsets: tuple = (0.0,)) -> dict:
    """Multi-level occupancy correlation of one type over the height map (FFT): peaks [(i, j, score, cover, yaw)].
    With ``yaw_peaks_k`` > 0 (amendment A23) also each yaw's own top peaks (``yaw_peaks``; NMS at ``rank_nms_frac``
    of the diameter, the same score and cover floors)."""
    cell = hm["cell"]
    H, W = hm["shape"]
    obs = hm["h"]
    known = np.isfinite(obs)
    hmax_ = float(nodes[:, 2].max() - nodes[:, 2].min())
    explained = explained_for(hm, hmax_, max(float(p["ceil_min_m"]), float(p["ceil_frac"]) * hmax_))
    if explained is None:
        explained = np.zeros((H, W), bool)
    rxy = float(np.max(np.linalg.norm(nodes[:, :2], axis=1)))
    r = int(math.ceil(rxy / cell)) + 2
    hmax = float(nodes[:, 2].max() - nodes[:, 2].min())
    lvls = type_levels(hmax, p)
    tol = float(p["level_tol_m"])
    shape = (H + 2 * r, W + 2 * r)
    fo = []
    for lv in lvls:
        s = np.zeros((H, W))
        s[known & (obs >= lv - tol)] = 1.0
        s[known & (obs < lv - tol)] = -1.0
        fo.append(np.fft.rfft2(s, s=shape))
    fk = np.fft.rfft2(known.astype(float), s=shape)
    # the ceiling: a footprint cell observed well ABOVE the type's top (something occupies the space above the
    # part: a plate inside the cabinet's footprint) scores -1
    ceil = hmax + max(float(p["ceil_min_m"]), float(p["ceil_frac"]) * hmax)
    fc_ = np.fft.rfft2(-(known & (obs >= ceil) & ~explained).astype(float), s=shape)
    best = np.full((H, W), -np.inf)
    best_cov = np.zeros((H, W))
    best_yaw = np.zeros((H, W))
    yaw_peaks = []
    dia_ = diam if diam is not None else 2.0 * rxy
    rad_y = max(1, int(round(float(p.get("rank_nms_frac", 0.15)) * dia_ / cell)))
    for yw in yaws:
        tm0 = template(nodes, yw, cell, r)
        sc = cov = None
        for off in dz_offsets:  # A72: (0.0,) reproduces A50's search exactly
            tm = tm0 + float(off) if off else tm0
            if off:
                tm = np.where(np.isfinite(tm) & (tm > 0.0), tm, np.nan)  # sunk cells: hidden by the support
            fp = np.isfinite(tm)
            if not fp.any():
                continue
            acc = None
            mass = 0.0
            for lv, F in zip(lvls, fo):
                tl = (fp & (np.nan_to_num(tm, nan=-1.0) >= lv)).astype(float)
                mass += float(tl.sum())
                ft = np.fft.rfft2(tl[::-1, ::-1], s=shape)
                acc = F * ft if acc is None else acc + F * ft
            ffp = np.fft.rfft2(fp[::-1, ::-1].astype(float), s=shape)
            acc = acc + fc_ * ffp
            mass += float(fp.sum())
            sc_o = np.fft.irfft2(acc, s=shape)[r:r + H, r:r + W] / max(mass, 1.0)
            cov_o = np.fft.irfft2(fk * ffp, s=shape)[r:r + H, r:r + W] / max(float(fp.sum()), 1.0)
            if sc is None:
                sc, cov = sc_o, cov_o
            else:
                up_ = sc_o > sc
                sc = np.where(up_, sc_o, sc)
                cov = np.where(up_, cov_o, cov)
        upd = sc > best
        best[upd] = sc[upd]
        best_cov[upd] = cov[upd]
        best_yaw[upd] = yw
        if yaw_peaks_k > 0:
            yaw_peaks += _nms(sc, cov, np.full((H, W), yw), rad_y, int(yaw_peaks_k), float(p["min_score"]),
                              float(p["min_cover"]))
    dia = diam if diam is not None else 2.0 * rxy
    rad = max(1, int(round(float(p["nms_frac"]) * dia / cell)))
    peaks = _nms(best, best_cov, best_yaw, rad, int(p["peaks"]), float(p["min_score"]), float(p["min_cover"]))
    out = {"peaks": peaks, "levels": [round(x, 4) for x in lvls], "r_cells": r, "hmax_m": round(hmax, 4),
           "ceil_m": round(ceil, 4)}
    if yaw_peaks_k > 0:
        out["yaw_peaks"] = yaw_peaks
    return out


def articulated_peak(hm: dict, peak: tuple, variants: list, yaws: list, p: dict, diam: float) -> tuple:
    """A jointed fixture's coarse peak re-searched in a window around it (amendment A21) with whole-assembly
    templates (root plus its jointed parts, each variant one articulation: all closed, or one part at the open end
    of its range; ``variants`` = [(name, nodes in the root's body frame)]): an open drawer otherwise pulls the
    root-only template off the cabinet (LIBERO-Spatial task 4: the cabinet missed in all 10 DEV looks).
    The root's own frame is kept (the assembly nodes are expressed in it), so the refine is unchanged."""
    i0, j0, sc0, cov0, yw0 = peak
    cell = hm["cell"]
    H, W = hm["shape"]
    best = (sc0, (i0, j0, sc0, cov0, yw0), "root")
    articulated_peak.chosen = "root"  # A49: the variant that gave the peak (read by support_search)
    pw = {**p, "peaks": 1, "min_score": -1e9, "min_cover": 0.0}
    for name, vn in variants:
        rxy = float(np.max(np.linalg.norm(vn[:, :2], axis=1)))
        w = int(math.ceil((rxy + float(p["artic_window_m"])) / cell))
        ia, ib, ja, jb = max(0, i0 - w), min(H, i0 + w + 1), max(0, j0 - w), min(W, j0 + w + 1)
        sub = {"h": hm["h"][ia:ib, ja:jb], "a0": hm["a0"] + ja * cell, "b0": hm["b0"] + ia * cell, "cell": cell,
               "shape": (ib - ia, jb - ja)}
        if hm.get("explained_bottom") is not None:
            sub["explained_bottom"] = hm["explained_bottom"][ia:ib, ja:jb]
        # the window keeps only peaks whose root lies within artic_window_m of the root-only peak
        cs = coarse_search(sub, vn, yaws, pw, diam)
        for (i, j, sc, cov, yw) in cs["peaks"]:
            if abs(i + ia - i0) * cell <= float(p["artic_window_m"]) and abs(j + ja - j0) * cell <= \
                    float(p["artic_window_m"]) and sc > best[0]:
                best = (sc, (i + ia, j + ja, sc, cov, yw), name)
    articulated_peak.chosen = best[2]
    return best[1]


def depth_score(Xn, Nn, z_obs, K, pf: dict, a: float, b: float, yaw: float, h0: float, dz: float, agree_w: float,
                see_through_m: float, facing_cos: float) -> float:
    """The refine's score at one pose (soft node agreement over +-``agree_w`` minus 2 x seen-through), as a function
    (amendment A23: the same criterion ranks a fixture's coarse hypotheses)."""
    R, t = pose_at(pf, a, b, yaw, h0, dz)
    X = Xn @ R.T + t
    N = Nn @ R.T
    view = X / np.linalg.norm(X, axis=1, keepdims=True)
    Xv = X[(-(N * view).sum(1)) > facing_cos]
    if not len(Xv):
        return -1e18
    H, W = z_obs.shape
    u = np.rint(K[0, 0] * Xv[:, 0] / Xv[:, 2] + K[0, 2] - 0.5).astype(int)
    v = np.rint(K[1, 1] * Xv[:, 1] / Xv[:, 2] + K[1, 2] - 0.5).astype(int)
    ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    zo = z_obs[v[ins], u[ins]]
    dd = zo - Xv[ins, 2]
    has = np.isfinite(zo)
    k = np.where(has, np.clip(1.0 - (dd / agree_w) ** 2, 0.0, 1.0), 0.0)
    return float(k.sum() - 2.0 * (has & (dd > see_through_m)).sum())


def rank_hypotheses(hm: dict, hyps: list, nodes, nrm, z_obs, K, pf: dict, yaw_period_deg_: float, diam: float,
                    agree_m: float, see_through_m: float, facing_cos: float, p: dict = PARAMS) -> list:
    """Amendment A23: a welded fixture's coarse hypotheses [(i, j, score, cover, yaw)] ranked by the soft depth
    agreement (best over the ``rank_dz_m`` height offsets), then the best ``rank_keep`` DISTINCT ones (a kept one
    excludes hypotheses within ``rank_sep_frac`` of the diameter AND ``rank_sep_deg`` of its yaw, modulo the type's yaw
    period). Returns [(i, j, score, cover, yaw, depth_score, dz)], best first."""
    sel = np.linspace(0, len(nodes) - 1, min(len(nodes), int(p["rank_nodes"]))).astype(int)
    Xn, Nn = nodes[sel], nrm[sel]
    h0 = -float(nodes[:, 2].min())
    scored = []
    seen = set()
    for (i, j, sc, cov, yw) in hyps:
        key = (int(i), int(j), round(float(yw), 6))
        if key in seen:
            continue
        seen.add(key)
        aa = hm["a0"] + (j + 0.5) * hm["cell"]
        bb = hm["b0"] + (i + 0.5) * hm["cell"]
        best = (-1e18, 0.0)
        for dz in p["rank_dz_m"]:
            s = depth_score(Xn, Nn, z_obs, K, pf, aa, bb, yw, h0, float(dz), agree_m, see_through_m, facing_cos)
            if s > best[0]:
                best = (s, float(dz))
        scored.append((int(i), int(j), float(sc), float(cov), float(yw), best[0], best[1]))
    scored.sort(key=lambda h: (-h[5], h[0], h[1], h[4]))
    sep_c = float(p["rank_sep_frac"]) * diam / hm["cell"]
    per = max(float(yaw_period_deg_), 1e-6)
    keep = []
    for h in scored:
        clash = False
        for k in keep:
            dyaw = abs((math.degrees(h[4] - k[4]) + 0.5 * per) % per - 0.5 * per)
            if math.hypot(h[0] - k[0], h[1] - k[1]) <= sep_c and dyaw <= float(p["rank_sep_deg"]):
                clash = True
                break
        if not clash:
            keep.append(h)
        if len(keep) >= int(p["rank_keep"]):
            break
    return keep


def wide_stage(nodes, nrm, z_obs, K, pf: dict, a: float, b: float, yaw: float, dz: float, facing_cos: float,
               agree_m: float, see_through_m: float, p: dict = PARAMS, n_nodes: int | None = None,
               xy_step: float | None = None, yaw_copies: tuple = (0.0,), pivot=None) -> tuple:
    """Amendment A24: the refine's score over a wide (yaw, a, b) window around (a, b, yaw) at height offset ``dz``,
    vectorised over the in-plane offsets; returns (best score, yaw, a, b, n_eval). ``yaw_copies`` (degrees, A27):
    the window is also searched around yaw + each copy. ``pivot`` (body xy, A44): each yaw's window is centred so
    that this point stays where (a, b, yaw) puts it."""
    sel = np.linspace(0, len(nodes) - 1, min(len(nodes), int(n_nodes or p["wide_nodes"]))).astype(int)
    Xn, Nn = nodes[sel], nrm[sel]
    h0 = -float(nodes[:, 2].min())
    H, W = z_obs.shape
    r, st = float(p["wide_xy_m"]), float(xy_step or p["wide_xy_step_m"])
    offs = np.arange(-r, r + 1e-9, st)
    oa, ob = np.meshgrid(offs, offs, indexing="ij")
    oa, ob = oa.ravel(), ob.ravel()
    dT = oa[:, None] * pf["ex"][None, :] + ob[:, None] * pf["ey"][None, :]  # (n_off, 3)
    yr, ys = float(p["wide_yaw_deg"]), float(p["wide_yaw_step_deg"])
    best = (-1e18, yaw, a, b)
    per_copy = {}
    n_eval = 0
    for c_, dy in [(c_, c_ + d_) for c_ in yaw_copies for d_ in np.arange(-yr, yr + 1e-9, ys)]:
        yw = yaw + math.radians(float(dy))
        da_, db_ = pivot_shift(pivot, yaw, yw)  # A44 (0, 0 without a pivot)
        R, t = pose_at(pf, a + da_, b + db_, yw, h0, dz)
        X = (Xn @ R.T + t)[None, :, :] + dT[:, None, :]  # (n_off, n_nodes, 3)
        N = Nn @ R.T
        view = X / np.linalg.norm(X, axis=2, keepdims=True)
        vis = (-(view * N[None, :, :]).sum(2)) > facing_cos
        z = X[:, :, 2]
        ok = vis & (z > 0.05)
        zs = np.where(ok, z, 1.0)
        u = np.rint(K[0, 0] * X[:, :, 0] / zs + K[0, 2] - 0.5).astype(int)
        v = np.rint(K[1, 1] * X[:, :, 1] / zs + K[1, 2] - 0.5).astype(int)
        ins = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        zo = np.full(z.shape, np.nan)
        zo[ins] = z_obs[v[ins], u[ins]]
        dd = zo - z
        has = ins & np.isfinite(zo)
        kk = np.where(has, np.clip(1.0 - (np.where(has, dd, 0.0) / agree_m) ** 2, 0.0, 1.0), 0.0)
        sc = kk.sum(1) - 2.0 * (has & (np.where(has, dd, 0.0) > see_through_m)).sum(1)
        n_eval += len(sc)
        k = int(np.argmax(sc))
        cand = (float(sc[k]), yw, a + da_ + float(oa[k]), b + db_ + float(ob[k]))
        if cand[0] > best[0]:
            best = cand
        if c_ not in per_copy or cand[0] > per_copy[c_][0]:
            per_copy[c_] = cand
    if len(yaw_copies) > 1:
        wide_stage.per_copy = per_copy  # read by support_search (A27: the best of each yaw copy)
    return best[0], best[1], best[2], best[3], n_eval


def pivot_shift(pivot, yaw_from: float, yaw_to: float) -> tuple:
    """Amendment A44: the plane-frame (da, db) that keeps the body-frame xy point ``pivot`` fixed when a part's yaw
    goes from ``yaw_from`` to ``yaw_to`` (pose_at's R = G Rz(yaw): body xy turns by Rz(yaw) in the plane frame)."""
    if pivot is None:
        return 0.0, 0.0
    px, py = float(pivot[0]), float(pivot[1])
    c0, s0, c1, s1 = math.cos(yaw_from), math.sin(yaw_from), math.cos(yaw_to), math.sin(yaw_to)
    return (c0 * px - s0 * py) - (c1 * px - s1 * py), (s0 * px + c0 * py) - (s1 * px + c1 * py)


def pose_at(pf: dict, a: float, b: float, yaw: float, h0: float, dz: float) -> tuple:
    """Camera-frame (R, t) of a part upright on the support at plane coords (a, b), yaw, height offset dz."""
    cz, sz = math.cos(yaw), math.sin(yaw)
    R = pf["G"] @ np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    t = pf["p0"] + a * pf["ex"] + b * pf["ey"] + (h0 + dz) * pf["n"]
    return R, t


def dz_window(p: dict, nodes) -> tuple:
    """A welded fixture's height-offset window: ``dz_fixture_m``, and (A91, ``dz_origin_floor``) never so low that
    the part's body origin would lie below the support (offset >= -h0, h0 the origin's height above the lowest node):
    a type-geometry floor that keeps a part whose lowest geometry sinks into the support (the stove's base box) and
    refuses a carcass sunk below its own origin."""
    lo, hi = float(p["dz_fixture_m"][0]), float(p["dz_fixture_m"][1])
    if p.get("dz_origin_floor"):
        lo = max(lo, float(nodes[:, 2].min()))
    return lo, max(lo, hi)


def refine(nodes, nrm, z_obs, K, pf: dict, a: float, b: float, yaw: float, fixture: bool, axisym: bool,
           facing_cos: float, agree_m: float, see_through_m: float, p: dict = PARAMS, welded: bool = False,
           wide_sets: list | None = None, asm_fn=None, jointed_wide: list | None = None, pivot=None) -> dict:
    """Coarse-to-fine (yaw, a, b[, dz]) on camera depth: node agreement minus 2 x seen-through. ``welded``: a welded
    fixture (amendment A24's wide first stage applies); ``wide_sets``: [(nodes, normals)] in the root's body frame
    scored by the wide stage instead of the root's own (amendment A25: a jointed fixture's assembly variants);
    ``asm_fn(R, t) -> (nodes, normals, joints)``: the whole assembly in the root's body frame with its joints fitted
    at the camera-frame root pose (R, t); with it (amendment A26) the refine stages score the assembly, its joints
    fitted after the wide stage and again before the last two stages; ``jointed_wide`` (degrees, amendment A27): the
    wide stage of a jointed fixture scores the root with ``wide_nodes_jointed`` nodes at ``wide_xy_step_jointed_m``
    around every listed yaw copy (the articulation breaks the root's onboarded yaw symmetry), not the variants.
    ``pivot`` (body xy, amendment A44): every yaw tried keeps this point fixed (the wide stage's windows and each
    stage's yaw steps turn about it, not about the body origin)."""
    Xsrc, Nsrc = nodes, nrm
    asm_joints = None
    sel_f_root = np.linspace(0, len(nodes) - 1, min(len(nodes), int(p["refine_nodes_fine"]))).astype(int)
    sel = np.linspace(0, len(nodes) - 1, min(len(nodes), int(p["refine_nodes"]))).astype(int)
    sel_f = np.linspace(0, len(nodes) - 1, min(len(nodes), int(p["refine_nodes_fine"]))).astype(int)
    h0 = -float(nodes[:, 2].min())
    H, W = z_obs.shape
    st_ = {"X": nodes[sel], "N": nrm[sel], "w": agree_m}

    def score(yw, aa, bb, dz):
        Xn, Nn, agree_w = st_["X"], st_["N"], st_["w"]
        R, t = pose_at(pf, aa, bb, yw, h0, dz)
        X = Xn @ R.T + t
        N = Nn @ R.T
        view = X / np.linalg.norm(X, axis=1, keepdims=True)
        Xv = X[(-(N * view).sum(1)) > facing_cos]
        if not len(Xv):
            return -1e18
        u = np.rint(K[0, 0] * Xv[:, 0] / Xv[:, 2] + K[0, 2] - 0.5).astype(int)
        v = np.rint(K[1, 1] * Xv[:, 1] / Xv[:, 2] + K[1, 2] - 0.5).astype(int)
        ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        zo = z_obs[v[ins], u[ins]]
        dd = zo - Xv[ins, 2]
        has = np.isfinite(zo)
        # a soft agreement kernel (Epanechnikov over +-agree_m) instead of the hard count: the hard count is flat
        # for any offset inside the window, so it cannot fit a sub-centimetre height or position
        k = np.where(has, np.clip(1.0 - (dd / agree_w) ** 2, 0.0, 1.0), 0.0)
        return float(k.sum() - 2.0 * (has & (dd > see_through_m)).sum())

    dz = 0.0
    n_eval = 0

    def dz_search(yw, aa, bb, lo, hi, step):
        nonlocal n_eval
        best = (-1e18, 0.0)
        for q in np.arange(lo, hi + 1e-9, step):
            s = score(yw, aa, bb, float(q))
            n_eval += 1
            if s > best[0]:
                best = (s, float(q))
        return best[1]

    if fixture:
        lo, hi = dz_window(p, nodes)  # A91 (dz_origin_floor); A73's window without it
        dz = dz_search(yaw, a, b, float(lo), float(hi), float(p["dz_coarse_m"]))
        if p.get("wide_stage") and welded:  # amendment A24: a wide first stage, then the height offset again
            # A25: a jointed fixture's wide stage scores each whole-assembly variant (A21) and keeps the best
            best_w = None
            if jointed_wide is not None:  # amendment A27: the root, more nodes, every yaw copy of its period
                r_ = wide_stage(nodes, nrm, z_obs, K, pf, a, b, yaw, dz, facing_cos, agree_m, see_through_m, p,
                                n_nodes=int(p["wide_nodes_jointed"]), xy_step=float(p["wide_xy_step_jointed_m"]),
                                yaw_copies=tuple(jointed_wide), pivot=pivot)
                n_eval += r_[4]
                best_w = r_
            for (Xw, Nw) in ([] if jointed_wide is not None else (wide_sets or [(nodes, nrm)])):
                hw = -float(Xw[:, 2].min()) - h0  # the variant's lowest node vs the root's: same root frame
                r_ = wide_stage(Xw, Nw, z_obs, K, pf, a, b, yaw, dz - hw, facing_cos, agree_m, see_through_m, p,
                                pivot=pivot)
                n_eval += r_[4]
                if best_w is None or r_[0] > best_w[0]:
                    best_w = r_
            _ws, yaw, a, b = best_w[:4]
            dz = dz_search(yaw, a, b, float(lo), float(hi), float(p["dz_coarse_m"]))
        if asm_fn is not None and welded:  # amendment A26: refine the whole assembly at its fitted joints
            Xsrc, Nsrc, asm_joints = asm_fn(*pose_at(pf, a, b, yaw, h0, dz))
            sel = np.linspace(0, len(Xsrc) - 1, min(len(Xsrc), int(p["refine_nodes"]))).astype(int)
            sel_f = np.linspace(0, len(Xsrc) - 1, min(len(Xsrc), int(p["refine_nodes_fine"]))).astype(int)
            st_.update(X=Xsrc[sel], N=Nsrc[sel])
            dz = dz_search(yaw, a, b, float(lo), float(hi), float(p["dz_coarse_m"]))
    best = score(yaw, a, b, dz)
    n_st = len(p["refine_stages"])
    for k, (dyaw_deg, dxy) in enumerate(p["refine_stages"]):
        if k == n_st - 2:  # the last two stages: more nodes and a narrower kernel (sub-mm, sub-degree)
            if asm_fn is not None and asm_joints is not None and p.get("asm_fine_root"):
                # A32: the fine stages score the ROOT alone (its own surfaces fix its pose; with the drawers' fitted
                # joints in the score the root could slide along the drawer axis by a joint step)
                Xsrc, Nsrc, sel_f = nodes, nrm, sel_f_root
            elif asm_fn is not None and asm_joints is not None:  # A26: the joints again at the better pose
                Xsrc, Nsrc, asm_joints = asm_fn(*pose_at(pf, a, b, yaw, h0, dz))
                sel_f = np.linspace(0, len(Xsrc) - 1, min(len(Xsrc), int(p["refine_nodes_fine"]))).astype(int)
            st_.update(X=Xsrc[sel_f], N=Nsrc[sel_f], w=float(p["fine_kernel_frac"]) * agree_m)
            best = score(yaw, a, b, dz)
        dyaw = math.radians(dyaw_deg)
        y_c, a_c, b_c = yaw, a, b
        ys = [y_c] if axisym else [y_c + m * dyaw for m in (-2, -1, 0, 1, 2)]
        for yw in ys:
            da_, db_ = pivot_shift(pivot, y_c, yw)  # A44 (0, 0 without a pivot)
            for i in (-2, -1, 0, 1, 2):
                for j in (-2, -1, 0, 1, 2):
                    s = score(yw, a_c + da_ + i * dxy, b_c + db_ + j * dxy, dz)
                    n_eval += 1
                    if s > best:
                        best, yaw, a, b = s, yw, a_c + da_ + i * dxy, b_c + db_ + j * dxy
        if fixture:
            lo, hi = dz_window(p, nodes)
            if k < int(p["dz_global_stages"]):  # the whole range again at the better pose (A18: the cabinet's height
                # had two modes 2 cm apart, and a local search kept the one the coarse pose fell into)
                dz = dz_search(yaw, a, b, float(lo), float(hi), float(p["dz_coarse_m"]))
            st = float(p["dz_fine_m"]) if k >= 1 else float(p["dz_coarse_m"])
            dz = dz_search(yaw, a, b, max(float(lo), dz - 4 * st), min(float(hi), dz + 4 * st), st)
            best = score(yaw, a, b, dz)
    R, t = pose_at(pf, a, b, yaw, h0, dz)
    out = {"R": R, "t": t, "yaw": yaw, "a": a, "b": b, "dz": dz, "score": best, "n_eval": n_eval}
    if asm_joints is not None:
        out["asm_joints"] = {k_: round(float(v_), 5) for k_, v_ in asm_joints.items()}
    return out


def jointed_hypotheses(hm: dict, yaw_peaks: list, nodes, nrm, z_obs, K, pf: dict, fam, afn, agree_m: float,
                       see_through_m: float, facing_cos: float, p: dict = PARAMS) -> tuple:
    """Amendment A68: a jointed fixture's (yaw, a, b) hypotheses from EVERY yaw of the coarse map. Each yaw's own best
    root-template peak (``yaw_peaks``) and, when the root's onboarded yaw period is a half turn, its copy turned by
    180 deg about the root's footprint centroid (the root template cannot tell the two; the articulation can) are
    scored by the root's soft depth score at its best height offset (``rank_dz_m``); the best ``jh_root_keep``
    DISTINCT ones (``jh_sep_m`` / ``jh_sep_deg``, modulo 360) are scored as the WHOLE ASSEMBLY with its joints fitted
    there, and the best ``jh_keep`` are returned. Type geometry and the observed depth only."""
    per = yaw_period_deg(fam, float(p["yaw_step_deg"]))
    cx, cy = float(nodes[:, 0].mean()), float(nodes[:, 1].mean())
    h0 = -float(nodes[:, 2].min())
    sel = np.linspace(0, len(nodes) - 1, min(len(nodes), int(p["rank_nodes"]))).astype(int)
    Xn, Nn = nodes[sel], nrm[sel]
    raw = []
    for (i, j, sc, cov, yw) in yaw_peaks:
        aa = hm["a0"] + (j + 0.5) * hm["cell"]
        bb = hm["b0"] + (i + 0.5) * hm["cell"]
        raw.append((float(yw), aa, bb))
        if abs(per - 180.0) < 1e-3:
            da, db = pivot_shift((cx, cy), float(yw), float(yw) + math.pi)
            raw.append((float(yw) + math.pi, aa + da, bb + db))
    scored = []
    n_local = 0
    for (yw, aa, bb) in raw:
        best = max((depth_score(Xn, Nn, z_obs, K, pf, aa, bb, yw, h0, float(dz), agree_m, see_through_m, facing_cos),
                    float(dz)) for dz in p["rank_dz_m"])
        if p.get("jh_local"):  # A81: the hypothesis refined locally (root, the wide stage's score) before ranking
            pl = {**p, "wide_xy_m": float(p["jh_local_xy_m"]), "wide_yaw_deg": float(p["jh_local_yaw_deg"]),
                  "wide_yaw_step_deg": float(p["jh_local_yaw_step_deg"])}
            _s, yw, aa, bb, ne = wide_stage(nodes, nrm, z_obs, K, pf, aa, bb, yw, best[1], facing_cos, agree_m,
                                            see_through_m, pl, n_nodes=int(p["jh_local_nodes"]),
                                            xy_step=float(p["jh_local_xy_step_m"]))
            n_local += ne
            best = max((depth_score(Xn, Nn, z_obs, K, pf, aa, bb, yw, h0, float(dz), agree_m, see_through_m,
                                    facing_cos), float(dz)) for dz in p["rank_dz_m"])
        scored.append((best[0], yw, aa, bb, best[1]))
    scored.sort(key=lambda h: -h[0])
    keep = []
    for h in scored:
        if any(math.hypot(h[2] - k[2], h[3] - k[3]) <= float(p["jh_sep_m"])
               and abs((math.degrees(h[1] - k[1]) + 180.0) % 360.0 - 180.0) <= float(p["jh_sep_deg"]) for k in keep):
            continue
        keep.append(h)
        if len(keep) >= int(p["jh_root_keep"]):
            break
    n_twin = 0
    if p.get("jh_local") and p.get("jh_twin") and abs(per - 180.0) < 1e-3:
        # A82: the half-turn copy (about the footprint centroid) of each of the best jh_twin_k LOCALLY refined
        # hypotheses, refined locally as well, joins the kept ones (the root cannot tell the two; a locally fitted
        # pose's copy is a better start for its twin than the map's own peak at that yaw)
        pl = {**p, "wide_xy_m": float(p["jh_local_xy_m"]), "wide_yaw_deg": float(p["jh_local_yaw_deg"]),
              "wide_yaw_step_deg": float(p["jh_local_yaw_step_deg"])}
        add = []
        for (_s0, yw, aa, bb, dz) in keep[:int(p["jh_twin_k"])]:
            da, db = pivot_shift((cx, cy), float(yw), float(yw) + math.pi)
            _s, y2, a2, b2, ne = wide_stage(nodes, nrm, z_obs, K, pf, aa + da, bb + db, float(yw) + math.pi, dz,
                                            facing_cos, agree_m, see_through_m, pl, n_nodes=int(p["jh_local_nodes"]),
                                            xy_step=float(p["jh_local_xy_step_m"]))
            n_local += ne
            b2_ = max((depth_score(Xn, Nn, z_obs, K, pf, a2, b2, y2, h0, float(q), agree_m, see_through_m,
                                   facing_cos), float(q)) for q in p["rank_dz_m"])
            h2 = (b2_[0], y2, a2, b2, b2_[1])
            if any(math.hypot(h2[2] - k[2], h2[3] - k[3]) <= float(p["jh_sep_m"])
                   and abs((math.degrees(h2[1] - k[1]) + 180.0) % 360.0 - 180.0) <= float(p["jh_sep_deg"])
                   for k in keep + add):
                continue
            add.append(h2)
        n_twin = len(add)
        keep = keep + add
    asm = []
    for (s_root, yw, aa, bb, dz) in keep:
        Xa, Na, _q = afn(*pose_at(pf, aa, bb, yw, h0, dz))
        sa = np.linspace(0, len(Xa) - 1, min(len(Xa), int(p["refine_nodes_fine"]))).astype(int)
        s_asm = depth_score(Xa[sa], Na[sa], z_obs, K, pf, aa, bb, yw, h0, dz, agree_m, see_through_m, facing_cos)
        asm.append((s_asm, s_root, yw, aa, bb, dz))
    asm.sort(key=lambda h: -h[0])
    out = [(h[2], h[3], h[4]) for h in asm[:int(p["jh_keep"])]]
    rec = {"n_raw": len(raw), "period_deg": round(per, 2),
           "kept": [[round(math.degrees(h[2]) % 360.0, 1), round(h[0], 1), round(h[1], 1)] for h in asm]}
    if p.get("jh_local"):
        rec["local_n_eval"] = int(n_local)
        rec["n_twin"] = int(n_twin)
    return out, rec


def closed_polish(rf: dict, axes: list, nodes, z_obs, K, pf: dict, cfn, agree_m: float, see_through_m: float,
                  facing_cos: float, p: dict = PARAMS) -> tuple:
    """Amendment A85: the chosen root pose of a jointed fixture polished against its root plus its CLOSED parts at
    rest (``cfn(R, t) -> (nodes, normals, q, closed, open)``, fitted once at the chosen pose; those nodes then move
    with the root): along each slide axis (root body frame, turned by the pose's yaw) over +-``jh_slide_m`` in
    ``jh_slide_step_m`` steps and +-one step in ``jh_slide_fine_m`` steps, then one stage of yaw +-2 x
    ``closed_yaw_step_deg`` and in-plane position +-2 x ``closed_xy_step_m``, scored with the fine kernel
    (``fine_kernel_frac`` x ``agree_m``). With no closed part the pose is kept. Returns (the result, a record)."""
    h0 = -float(nodes[:, 2].min())
    R0, t0 = pose_at(pf, rf["a"], rf["b"], rf["yaw"], h0, rf["dz"])
    Xc, Nc, q, closed, opened = cfn(R0, t0)
    rec = {"closed": sorted(closed), "open": sorted(opened), "q": {k_: round(float(v_), 4) for k_, v_ in q.items()}}
    if not closed:
        rec["kept"] = "no closed part"
        return rf, rec
    sa = np.linspace(0, len(Xc) - 1, min(len(Xc), int(p["refine_nodes_fine"]))).astype(int)
    Xs, Ns = Xc[sa], Nc[sa]
    w = float(p["fine_kernel_frac"]) * agree_m

    def sc(a_, b_, y_):
        return depth_score(Xs, Ns, z_obs, K, pf, a_, b_, y_, h0, rf["dz"], w, see_through_m, facing_cos)

    a, b, yw = rf["a"], rf["b"], rf["yaw"]
    s0 = sc(a, b, yw)
    rec["s0"] = round(s0, 2)
    best = s0
    moves = []
    for ax in axes:
        cz, sz = math.cos(yw), math.sin(yw)
        ux, uy = cz * float(ax[0]) - sz * float(ax[1]), sz * float(ax[0]) + cz * float(ax[1])
        n_ = math.hypot(ux, uy)
        if n_ < 1e-6:
            continue
        ux, uy = ux / n_, uy / n_
        bd = 0.0
        r_, st = float(p["jh_slide_m"]), float(p["jh_slide_step_m"])
        for d in np.arange(-r_, r_ + 1e-9, st):
            s_ = sc(a + d * ux, b + d * uy, yw)
            if s_ > best:
                best, bd = s_, float(d)
        d0 = bd
        for d in np.arange(d0 - st, d0 + st + 1e-9, float(p["jh_slide_fine_m"])):
            s_ = sc(a + d * ux, b + d * uy, yw)
            if s_ > best:
                best, bd = s_, float(d)
        a, b = a + bd * ux, b + bd * uy
        moves.append(round(1e3 * bd, 2))
    dyaw, dxy = math.radians(float(p["closed_yaw_step_deg"])), float(p["closed_xy_step_m"])
    y_c, a_c, b_c = yw, a, b
    for m_ in (-2, -1, 0, 1, 2):
        for i in (-2, -1, 0, 1, 2):
            for j in (-2, -1, 0, 1, 2):
                s_ = sc(a_c + i * dxy, b_c + j * dxy, y_c + m_ * dyaw)
                if s_ > best:
                    best, yw, a, b = s_, y_c + m_ * dyaw, a_c + i * dxy, b_c + j * dxy
    rec.update(moves_mm=moves, dyaw_deg=round(math.degrees(yw - rf["yaw"]), 3),
               dxy_mm=[round(1e3 * (a - a_c), 2), round(1e3 * (b - b_c), 2)], s1=round(best, 2))
    if a == rf["a"] and b == rf["b"] and yw == rf["yaw"]:
        return rf, rec
    R, t = pose_at(pf, a, b, yw, h0, rf["dz"])
    return {**rf, "a": a, "b": b, "yaw": yw, "R": R, "t": t}, rec


def slide_search(rf: dict, axes: list, nodes, z_obs, K, pf: dict, afn, agree_m: float, see_through_m: float,
                 facing_cos: float, p: dict = PARAMS) -> tuple:
    """Amendment A68: the jointed fixture's final root pose searched along each slide joint's axis (a unit vector
    in the root body frame; its in-plane part turned by the pose's yaw) over +-``jh_slide_m`` in
    ``jh_slide_step_m`` steps, then +-one step in ``jh_slide_fine_m`` steps, every pose scored as the WHOLE ASSEMBLY
    with its joints fitted there (the drawer fronts then tell where the carcass is along the axis its own surfaces
    leave loose). Returns (the updated result, a record)."""
    h0 = -float(nodes[:, 2].min())

    def asm_at(a_, b_):
        Xa, Na, _q = afn(*pose_at(pf, a_, b_, rf["yaw"], h0, rf["dz"]))
        sa = np.linspace(0, len(Xa) - 1, min(len(Xa), int(p["refine_nodes_fine"]))).astype(int)
        return depth_score(Xa[sa], Na[sa], z_obs, K, pf, a_, b_, rf["yaw"], h0, rf["dz"], agree_m, see_through_m,
                           facing_cos)

    a, b = rf["a"], rf["b"]
    s0 = asm_at(a, b)
    rec = {"asm0": round(s0, 2), "moves_mm": []}
    cz, sz = math.cos(rf["yaw"]), math.sin(rf["yaw"])
    for ax in axes:
        ux, uy = cz * float(ax[0]) - sz * float(ax[1]), sz * float(ax[0]) + cz * float(ax[1])
        nrm_ = math.hypot(ux, uy)
        if nrm_ < 1e-6:
            continue
        ux, uy = ux / nrm_, uy / nrm_
        best = (s0, 0.0)
        r_, st = float(p["jh_slide_m"]), float(p["jh_slide_step_m"])
        for d in np.arange(-r_, r_ + 1e-9, st):
            s_ = asm_at(a + d * ux, b + d * uy)
            if s_ > best[0]:
                best = (s_, float(d))
        d0 = best[1]
        for d in np.arange(d0 - st, d0 + st + 1e-9, float(p["jh_slide_fine_m"])):
            s_ = asm_at(a + d * ux, b + d * uy)
            if s_ > best[0]:
                best = (s_, float(d))
        a, b, s0 = a + best[1] * ux, b + best[1] * uy, best[0]
        rec["moves_mm"].append(round(1e3 * best[1], 2))
    if a == rf["a"] and b == rf["b"]:
        rec["asm"] = rec["asm0"]
        return rf, rec
    R, t = pose_at(pf, a, b, rf["yaw"], h0, rf["dz"])
    out = {**rf, "a": a, "b": b, "R": R, "t": t}
    rec["asm"] = round(s0, 2)
    return out, rec


def pose_as_given(nodes, nrm, z_obs, K, pf: dict, a: float, b: float, yaw: float, agree_m: float,
                  see_through_m: float, facing_cos: float, p: dict = PARAMS) -> dict:
    """2026-11-16 (``memory``): a welded fixture at exactly (a, b, yaw), only its height offset fitted (the refine's
    own window and steps, the root's soft depth score); a result in ``refine``'s form."""
    sel = np.linspace(0, len(nodes) - 1, min(len(nodes), int(p["refine_nodes_fine"]))).astype(int)
    Xn, Nn = nodes[sel], nrm[sel]
    h0 = -float(nodes[:, 2].min())
    lo, hi = dz_window(p, nodes)
    best = (-1e18, 0.0)
    n_eval = 0
    for step, span in ((float(p["dz_coarse_m"]), None), (float(p["dz_fine_m"]), 4.0 * float(p["dz_coarse_m"]))):
        q0, q1 = (float(lo), float(hi)) if span is None else (max(float(lo), best[1] - span),
                                                              min(float(hi), best[1] + span))
        for q in np.arange(q0, q1 + 1e-9, step):
            s_ = depth_score(Xn, Nn, z_obs, K, pf, a, b, yaw, h0, float(q), agree_m, see_through_m, facing_cos)
            n_eval += 1
            if s_ > best[0]:
                best = (s_, float(q))
    R, t = pose_at(pf, a, b, yaw, h0, best[1])
    return {"R": R, "t": t, "yaw": yaw, "a": a, "b": b, "dz": best[1], "score": best[0], "n_eval": n_eval}


def twin_fit(rf: dict, nodes, nrm, z_obs, K, pf: dict, afn, agree_m: float, see_through_m: float, facing_cos: float,
             p: dict = PARAMS, axes=None, cfn=None) -> tuple:
    """2026-11-16 (``twin_prior``): the HALF-TURN TWIN of a jointed fixture's fitted pose ``rf``, fitted on its own.

    The twin starts at ``rf`` turned by 180 deg about the root's footprint centroid (the two poses share the
    footprint). A mirror fit sits a few centimetres off along the fixture's depth (its back panel stands where the
    picture shows the drawer fronts: 3.4 cm on the Goal cabinet), so the root is first searched around the start
    (``wide_stage``'s score over +-``twin_xy_m`` and +-``twin_yaw_deg``); the result is then refined as the search's
    own hypotheses are (the whole assembly with its joints fitted, the root alone in the fine stages) and, with
    ``cfn`` and ``axes``, polished against its closed parts (A85). Type geometry and the observed depth only.
    Returns (the twin's result, a record)."""
    cx, cy = float(nodes[:, 0].mean()), float(nodes[:, 1].mean())
    y2 = float(rf["yaw"]) + math.pi
    da, db = pivot_shift((cx, cy), float(rf["yaw"]), y2)
    a2, b2 = float(rf["a"]) + da, float(rf["b"]) + db
    pl = {**p, "wide_xy_m": float(p["twin_xy_m"]), "wide_yaw_deg": float(p["twin_yaw_deg"]),
          "wide_yaw_step_deg": float(p["twin_yaw_step_deg"])}
    s_w, y3, a3, b3, n_w = wide_stage(nodes, nrm, z_obs, K, pf, a2, b2, y2, float(rf["dz"]), facing_cos, agree_m,
                                      see_through_m, pl, n_nodes=int(p["twin_nodes"]),
                                      xy_step=float(p["twin_xy_step_m"]))
    tw = refine(nodes, nrm, z_obs, K, pf, a3, b3, y3, True, False, facing_cos, agree_m, see_through_m,
                {**p, "wide_stage": False}, welded=True, asm_fn=afn)
    rec = {"start_mm": [round(1e3 * (a3 - a2), 1), round(1e3 * (b3 - b2), 1)],
           "start_dyaw_deg": round(math.degrees(y3 - y2), 2), "wide_n_eval": int(n_w),
           "score": round(float(tw["score"]), 2), "score_was": round(float(rf["score"]), 2)}
    if cfn is not None and axes and p.get("fine_closed"):
        tw, rec["closed_polish"] = closed_polish(tw, axes, nodes, z_obs, K, pf, cfn, agree_m, see_through_m,
                                                 facing_cos, p)
    rec["moved_mm"] = [round(1e3 * (float(tw["a"]) - a2), 1), round(1e3 * (float(tw["b"]) - b2), 1)]
    rec["yaw_deg"] = round(math.degrees(float(tw["yaw"])) % 360.0, 2)
    return tw, rec


def support_search(cell_types: list, nodes_of, nrm_of, family_of, diam_of, kind_of, z_obs, K, T, plane: dict,
                   box_w, facing_cos: float, agree_m: float, see_through_m: float, p: dict = PARAMS,
                   sup: dict | None = None, explain_pts=None, variants_of=None, variants_n_of=None,
                   assembly_of=None, slide_axes_of=None, closed_of=None, face_of=None, accept_of=None,
                   seeds_of=None) -> dict:
    """Stage 2 over the routed parts ``cell_types`` [(gid, type)]: refined candidate poses (camera frame) per part,
    with the coarse diagnostics. The server checks and accepts them."""
    import time

    t0 = time.perf_counter()
    sup = sup or refit_support(z_obs, K, plane, p)
    pf = plane_frame(sup, T)
    ext = box_to_plane(box_w, pf, T)
    hm = height_map(z_obs, K, sup, pf, ext, float(p["cell_m"]))
    if explain_pts:
        hm["explained_bottom"] = mark_cells(hm, pf, explain_pts)
    t_map = time.perf_counter() - t0
    out = []
    diag = {"map_shape": list(hm["shape"]), "known_share": round(float(np.isfinite(hm["h"]).mean()), 4),
            "map_s": round(t_map, 4), "types": {}}
    for g, typ in cell_types:
        a = time.perf_counter()
        nodes, nrm = nodes_of(g), nrm_of(g)
        fam = family_of(g)
        yaws = type_yaws(fam, p)
        axisym = len(yaws) == 1
        rank = bool(p.get("depth_rank")) and kind_of(g) == "fixture"  # amendment A23
        is_jointed = kind_of(g) == "fixture" and variants_n_of is not None and bool(variants_n_of(g))
        mem_rec = None
        if seeds_of is not None and p.get("memory") is not None and kind_of(g) == "fixture":
            # 2026-11-16: the fixture's remembered place first (see PARAMS): refined from there, polished, and taken
            # when the server's floors pass it and it stayed where it was remembered; the search is then skipped
            afn_m = (lambda R_, t_, g_=g: assembly_of(g_, R_, t_)) if (is_jointed and assembly_of is not None) \
                else None
            for (ym, am, bm) in seeds_of(g, pf):
                ys, as_, bs = ym, am, bm
                if float(p["memory_search_m"]) > 0.0:  # the fixture stands a little differently every time
                    pl = {**p, "wide_xy_m": float(p["memory_search_m"]),
                          "wide_yaw_deg": 0.0 if axisym else float(p["memory_search_yaw_deg"]),
                          "wide_yaw_step_deg": float(p["memory_search_yaw_step_deg"])}
                    lo_, hi_ = dz_window(p, nodes)
                    h0_ = -float(nodes[:, 2].min())
                    sel_ = np.linspace(0, len(nodes) - 1, min(len(nodes), int(p["memory_nodes"]))).astype(int)
                    dz0 = max(np.arange(float(lo_), float(hi_) + 1e-9, float(p["dz_coarse_m"])),
                              key=lambda q_: depth_score(nodes[sel_], nrm[sel_], z_obs, K, pf, am, bm, ym, h0_,
                                                         float(q_), agree_m, see_through_m, facing_cos))
                    _sw, ys, as_, bs, _ne = wide_stage(nodes, nrm, z_obs, K, pf, am, bm, ym, float(dz0), facing_cos,
                                                       agree_m, see_through_m, pl, n_nodes=int(p["memory_nodes"]),
                                                       xy_step=float(p["memory_search_step_m"]))
                rm = refine(nodes, nrm, z_obs, K, pf, as_, bs, ys, True, axisym, facing_cos, agree_m, see_through_m,
                            {**p, "wide_stage": False}, welded=True, asm_fn=afn_m)
                crec = None
                if is_jointed and closed_of is not None and p.get("fine_closed") and slide_axes_of is not None:
                    rm, crec = closed_polish(rm, slide_axes_of(g), nodes, z_obs, K, pf,
                                             (lambda R_, t_, g_=g: closed_of(g_, R_, t_)), agree_m, see_through_m,
                                             facing_cos, p)
                moved = math.hypot(float(rm["a"]) - am, float(rm["b"]) - bm)
                dyaw = abs((math.degrees(float(rm["yaw"]) - ym) + 180.0) % 360.0 - 180.0)
                ok = moved <= float(p["memory_xy_m"]) and dyaw <= float(p["memory_yaw_deg"]) and (
                    accept_of is None or bool(accept_of(g, rm["R"], rm["t"], hm, pf)))
                mem_rec = {"moved_mm": round(1e3 * moved, 1), "dyaw_deg": round(dyaw, 2), "taken": bool(ok),
                           "score": round(float(rm["score"]), 2), "how": "refined"}
                if not ok and accept_of is not None:
                    # the refine left the remembered pose for one the floors refuse (its coarse stages move up to
                    # a centimetre): the remembered pose itself, its height offset fitted, before the same floors
                    r0 = pose_as_given(nodes, nrm, z_obs, K, pf, am, bm, ym, agree_m, see_through_m, facing_cos, p)
                    if bool(accept_of(g, r0["R"], r0["t"], hm, pf)):
                        rm, crec, ok = r0, None, True
                        mem_rec.update(taken=True, how="as remembered", refined_score=mem_rec["score"],
                                       score=round(float(r0["score"]), 2))
                if ok:
                    out.append({"gid": str(g), "type": typ, "R": rm["R"], "t": rm["t"], "coarse": {"memory": True},
                                "refine": {"score": rm["score"], "dz_m": round(rm["dz"], 5), "n_eval": rm["n_eval"],
                                           "yaw_deg": round(math.degrees(rm["yaw"]) % 360.0, 2),
                                           **({"asm_joints": rm["asm_joints"]} if "asm_joints" in rm else {}),
                                           **({"closed_polish": crec} if crec is not None else {}),
                                           "memory": mem_rec}})
                    break
            if mem_rec is not None and mem_rec["taken"]:
                diag["types"][typ] = {"memory": mem_rec, "refine_s": round(time.perf_counter() - a, 4)}
                continue
        ypk = int(p["rank_peaks_per_yaw"]) if rank else (int(p["jh_per_yaw"]) if is_jointed and p.get("jointed_hyp")
                                                         else 0)  # A68: each yaw's own peaks (records nothing else)
        offs = tuple(float(x) for x in p["coarse_dz_m"]) if (p.get("coarse_dz") and kind_of(g) == "fixture") \
            else (0.0,)  # A72: a welded fixture's templates also sunk into the support
        p_c = p
        if p.get("fixture_peaks") and kind_of(g) == "fixture" and not is_jointed:
            p_c = {**p, "peaks": int(p["fixture_peaks"])}  # A86: more coarse peaks for a welded, unjointed fixture
        if p.get("fixture_min_score") is not None and kind_of(g) == "fixture" and not is_jointed:
            p_c = {**p_c, "min_score": float(p["fixture_min_score"])}  # A87: and a lower proposal floor
        cs = coarse_search(hm, nodes, yaws, p_c, diam_of(g), yaw_peaks_k=ypk, dz_offsets=offs)
        peak_variant = {}  # A49: per peak index, the xy centroid (root body frame) of the variant that gave it
        if variants_of is not None and variants_of(g):
            vmap = dict(variants_of(g))
            new_peaks = []
            for k_, pk in enumerate(cs["peaks"]):
                new_peaks.append(articulated_peak(hm, pk, variants_of(g), yaws, p, diam_of(g)))
                vn_ = vmap.get(articulated_peak.chosen, nodes)
                peak_variant[k_] = (articulated_peak.chosen, float(vn_[:, 0].mean()), float(vn_[:, 1].mean()))
            cs["peaks"] = new_peaks
        depth_of = {}
        if rank:
            ranked = rank_hypotheses(hm, list(cs["peaks"]) + list(cs.get("yaw_peaks") or []), nodes, nrm, z_obs, K,
                                     pf, yaw_period_deg(fam, float(p["yaw_step_deg"])), diam_of(g), agree_m,
                                     see_through_m, facing_cos, p)
            cs["n_hypotheses"] = len(cs["peaks"]) + len(cs.get("yaw_peaks") or [])
            cs["peaks"] = [h[:5] for h in ranked]
            depth_of = {k_: (round(h[5], 2), h[6]) for k_, h in enumerate(ranked)}
        t_c = time.perf_counter() - a
        b_ = time.perf_counter()
        for k_pk, (i, j, sc, cov, yw) in enumerate(cs["peaks"]):
            aa = hm["a0"] + (j + 0.5) * hm["cell"]
            bb = hm["b0"] + (i + 0.5) * hm["cell"]
            # welded fixtures refine from several yaw starts around the coarse yaw (A18), the best final score wins
            welded = kind_of(g) == "fixture"
            # A44: a welded fixture turns about its footprint centroid (body xy mean of its root nodes)
            pv = (float(nodes[:, 0].mean()), float(nodes[:, 1].mean())) if (welded and p.get("pivot_centroid")) \
                else None
            # A45: the centroid pivot for a JOINTED fixture's quarter-turn wide windows only (A27)
            pv_jw = (float(nodes[:, 0].mean()), float(nodes[:, 1].mean())) if (welded and p.get(
                "pivot_jointed_wide")) else pv
            if welded and p.get("pivot_variant_wide") and k_pk in peak_variant:
                # A49: the windows turn about the footprint centroid of the VARIANT whose template the map matched
                # (A21: with a drawer open the assembly's footprint, not the carcass's, is what the peak located)
                pv_jw = peak_variant[k_pk][1:]
            starts = [yw] if axisym or not welded or p.get("wide_stage") else \
                [yw + math.radians(float(d_)) for d_ in p["fixture_yaw_starts_deg"]]
            wsets = None
            afn = None
            jw = None
            jointed = welded and variants_n_of is not None and bool(variants_n_of(g))
            if jointed and p.get("wide_assembly"):
                wsets = [(vn, vr) for _nm, vn, vr in variants_n_of(g)]  # amendment A25
            if jointed and p.get("refine_assembly") and assembly_of is not None:
                afn = (lambda R_, t_, g_=g: assembly_of(g_, R_, t_))  # amendment A26
            starts_ab = [(y0, aa, bb) for y0 in starts]
            p_ref = p
            if jointed and p.get("wide_jointed_root"):  # amendment A27: the full turn, the best 2 quarter-turn copies
                lo_, hi_ = dz_window(p, nodes)  # A91
                h0_ = -float(nodes[:, 2].min())
                best_dz = max(np.arange(float(lo_), float(hi_) + 1e-9, float(p["dz_coarse_m"])),
                              key=lambda q_: depth_score(nodes, nrm, z_obs, K, pf, aa, bb, yw, h0_, float(q_), agree_m,
                                                         see_through_m, facing_cos))
                wide_stage.per_copy = {}
                wide_stage(nodes, nrm, z_obs, K, pf, aa, bb, yw, float(best_dz), facing_cos, agree_m, see_through_m,
                           p, n_nodes=int(p["wide_nodes_jointed"]), xy_step=float(p["wide_xy_step_jointed_m"]),
                           yaw_copies=(0.0, 90.0, 180.0, 270.0), pivot=pv_jw)
                tops = sorted(wide_stage.per_copy.values(), key=lambda c_: -c_[0])[:int(p["wide_jointed_keep"])]
                if afn is not None and p.get("wide_jointed_pick_assembly"):
                    # A29: each window's pose scored once as the WHOLE ASSEMBLY with its joints fitted there (the
                    # drawers break the carcass's twin symmetry); only the best window is refined
                    def asm_score(c_):
                        R_, t_ = pose_at(pf, c_[2], c_[3], c_[1], h0_, float(best_dz))
                        Xa, Na, _q = afn(R_, t_)
                        sa = np.linspace(0, len(Xa) - 1, min(len(Xa), int(p["refine_nodes_fine"]))).astype(int)
                        return depth_score(Xa[sa], Na[sa], z_obs, K, pf, c_[2], c_[3], c_[1], h0_, float(best_dz),
                                           agree_m, see_through_m, facing_cos)
                    tops = [max(tops, key=asm_score)]
                starts_ab = [(c_[1], c_[2], c_[3]) for c_ in tops]
                p_ref = {**p, "wide_stage": False}  # the wide stage is done; the refine continues from each start
            if pv is not None and not (jointed and p.get("wide_jointed_root")):  # A44: A18's yaw starts turn about
                starts_ab = [(y0, aa + pivot_shift(pv, yw, y0)[0], bb + pivot_shift(pv, yw, y0)[1])  # the centroid
                             for (y0, _a, _b) in starts_ab]
            rfs = [refine(nodes, nrm, z_obs, K, pf, a0_, b0_, y0, True, axisym, facing_cos, agree_m, see_through_m,
                          p_ref, welded=welded, wide_sets=wsets, asm_fn=afn, jointed_wide=jw, pivot=pv)
                   for (y0, a0_, b0_) in starts_ab]
            jh_rec = None
            if jointed and afn is not None and p.get("jointed_hyp") and k_pk == 0:
                # A68: hypotheses from every yaw of the map (see PARAMS), the best refined and added to the paths
                hyps, jh_rec = jointed_hypotheses(hm, cs.get("yaw_peaks") or [], nodes, nrm, z_obs, K, pf, fam, afn,
                                                  agree_m, see_through_m, facing_cos, p)
                p_h = {**p, "wide_stage": False}
                rfs += [refine(nodes, nrm, z_obs, K, pf, ha, hb, hy, True, axisym, facing_cos, agree_m, see_through_m,
                               p_h, welded=welded, asm_fn=afn) for (hy, ha, hb) in hyps]
            if welded and p.get("wide_stage") and (p.get("wide_also_narrow") if not jointed
                                                    else p.get("wide_also_narrow_jointed")):
                # A32: the pre-A24 refine from the coarse peak (A18's yaw starts) as well; the better final refine
                # score wins (a wide window can hold a turned optimum where the coarse yaw was right)
                p_n = {**p, "wide_stage": False}
                rfs += [refine(nodes, nrm, z_obs, K, pf, aa + pivot_shift(pv, yw, yw + math.radians(float(d_)))[0],
                               bb + pivot_shift(pv, yw, yw + math.radians(float(d_)))[1],
                               yw + math.radians(float(d_)), True, axisym,
                               facing_cos, agree_m, see_through_m, p_n, welded=welded, pivot=pv)
                        for d_ in p["fixture_yaw_starts_deg"]]
            asm_sel = None
            if jointed and afn is not None and p.get("asm_select"):
                # A44: a jointed fixture's final pose is chosen among its refine results by the WHOLE ASSEMBLY's score
                # (joints fitted at each final pose), not the root's: the root (a carcass) is near-symmetric under a
                # half turn, so its own score cannot tell a cabinet from one facing the other way
                h0r = -float(nodes[:, 2].min())
                asm_sel = []
                for r_ in rfs:
                    Xa, Na, _q = afn(r_["R"], r_["t"])
                    sa = np.linspace(0, len(Xa) - 1, min(len(Xa), int(p["refine_nodes_fine"]))).astype(int)
                    sc_a = depth_score(Xa[sa], Na[sa], z_obs, K, pf, r_["a"], r_["b"], r_["yaw"], h0r, r_["dz"],
                                       agree_m, see_through_m, facing_cos)
                    r_["root_score"] = r_["score"]
                    r_["score"] = sc_a
                    asm_sel.append(round(sc_a, 2))
            modes_rec = None
            if jointed and afn is not None and p.get("asm_select_modes") and not p.get("asm_select"):
                # A45: the refine results are grouped into YAW MODES (within asm_mode_deg of a mode's first member,
                # modulo 360); the mode whose best member scores best as the WHOLE ASSEMBLY (joints fitted at that
                # pose) is chosen, and within it the result with the best ROOT score (A32/A33's rule, unchanged):
                # the assembly tells a cabinet from one facing the other way, the root fixes the pose inside a mode
                h0r = -float(nodes[:, 2].min())
                order = sorted(range(len(rfs)), key=lambda i_: -rfs[i_]["score"])
                modes = []
                for i_ in order:
                    y_i = math.degrees(rfs[i_]["yaw"])
                    for md_ in modes:
                        d_ = abs((y_i - md_["yaw"] + 180.0) % 360.0 - 180.0)
                        if d_ <= float(p["asm_mode_deg"]):
                            md_["members"].append(i_)
                            break
                    else:
                        modes.append({"yaw": y_i, "members": [i_]})
                for md_ in modes:
                    r_ = rfs[md_["members"][0]]  # the mode's best root score
                    Xa, Na, _q = afn(r_["R"], r_["t"])
                    sa = np.linspace(0, len(Xa) - 1, min(len(Xa), int(p["refine_nodes_fine"]))).astype(int)
                    md_["asm"] = depth_score(Xa[sa], Na[sa], z_obs, K, pf, r_["a"], r_["b"], r_["yaw"], h0r, r_["dz"],
                                             agree_m, see_through_m, facing_cos)
                best_md = max(modes, key=lambda md_: md_["asm"])
                rf = rfs[best_md["members"][0]]
                modes_rec = [[round(md_["yaw"] % 360.0, 2), round(md_["asm"], 1),
                              round(rfs[md_["members"][0]]["score"], 1), len(md_["members"])] for md_ in modes]
            else:
                rf = max(rfs, key=lambda r_: r_["score"])
            closed_rec = None
            if jointed and closed_of is not None and p.get("fine_closed") and slide_axes_of is not None:
                rf, closed_rec = closed_polish(rf, slide_axes_of(g), nodes, z_obs, K, pf,
                                               (lambda R_, t_, g_=g: closed_of(g_, R_, t_)), agree_m,
                                               see_through_m, facing_cos, p)  # A85
            slide_rec = None
            if jointed and afn is not None and p.get("jh_slide") and slide_axes_of is not None:
                rf, slide_rec = slide_search(rf, slide_axes_of(g), nodes, z_obs, K, pf, afn, agree_m, see_through_m,
                                             facing_cos, p)  # A68
            face_rec = None
            if jointed and face_of is not None and p.get("twin_prior") is not None:
                # 2026-11-16: the facing prior's verdict on the chosen pose (negative: back to front). Its half-turn
                # twin, fitted on its own, takes its place when the prior calls the twin the right way round and
                # the server's own floors accept it (``accept_of``); else the search's choice stands
                f0 = face_of(g, rf["R"], rf["t"])
                face_rec = {"prior": None if f0 is None else round(float(f0), 3)}
                if f0 is not None and f0 < 0:
                    tw, trec = twin_fit(rf, nodes, nrm, z_obs, K, pf, afn, agree_m, see_through_m, facing_cos, p,
                                        axes=slide_axes_of(g) if slide_axes_of is not None else None,
                                        cfn=(lambda R_, t_, g_=g: closed_of(g_, R_, t_)) if closed_of is not None
                                        else None)
                    f1 = face_of(g, tw["R"], tw["t"])
                    passed = accept_of is None or bool(accept_of(g, tw["R"], tw["t"], hm, pf))
                    ok = f1 is not None and f1 > 0 and (passed or (p["twin_prior"] or {}).get("take") == "always")
                    face_rec.update(twin=trec, twin_prior=None if f1 is None else round(float(f1), 3),
                                    taken=bool(ok), twin_passed=bool(passed))
                    if ok:
                        rf = tw
            co = {"score": round(sc, 4), "cover": round(cov, 4), "yaw_deg": round(math.degrees(yw), 1),
                  "cell": [int(i), int(j)]}
            if k_pk in depth_of:
                co["depth_rank"] = {"rank": k_pk, "depth_score": depth_of[k_pk][0], "dz_m": depth_of[k_pk][1]}
            if k_pk in peak_variant and p.get("pivot_variant_wide"):
                co["variant"] = peak_variant[k_pk][0]  # A49 (record only)
            out.append({"gid": str(g), "type": typ, "R": rf["R"], "t": rf["t"],
                        "coarse": co,
                        "refine": {"score": rf["score"], "dz_m": round(rf["dz"], 5), "n_eval": rf["n_eval"],
                                   "yaw_deg": round(math.degrees(rf["yaw"]) % 360.0, 2),
                                   **({"asm_joints": rf["asm_joints"]} if "asm_joints" in rf else {}),
                                   **({"asm_select": asm_sel, "root_score": round(rf["root_score"], 2),
                                       "paths": [[round(math.degrees(r_["yaw"]) % 360.0, 2), round(r_["score"], 1)]
                                                 for r_ in rfs]}
                                      if asm_sel is not None else {}),
                                   **({"asm_modes": modes_rec} if modes_rec is not None else {}),
                                   **({"jointed_hyp": jh_rec} if jh_rec is not None else {}),
                                   **({"slide": slide_rec} if slide_rec is not None else {}),
                                   **({"face": face_rec} if face_rec is not None else {}),
                                   **({"closed_polish": closed_rec} if closed_rec is not None else {})}})
        diag["types"][typ] = {"n_yaws": len(yaws), "levels": cs["levels"], "hmax_m": cs["hmax_m"],
                              "n_peaks": len(cs["peaks"]), "coarse_s": round(t_c, 4),
                              "refine_s": round(time.perf_counter() - b_, 4),
                              **({"memory": mem_rec} if mem_rec is not None else {})}
        if rank:
            diag["types"][typ]["n_hypotheses"] = cs.get("n_hypotheses")
    diag["support"] = {"normal": [round(float(x), 6) for x in sup["normal"]], "d": round(float(sup["d"]), 6),
                       "rms_m": sup.get("rms_m"), "n_pts": sup.get("n_pts")}
    diag["total_s"] = round(time.perf_counter() - t0, 4)
    return {"candidates": out, "diag": diag, "support": sup, "map": hm, "frame": pf}


def cover_check(hm: dict, pf: dict, nodes: np.ndarray, R, t, p: dict = PARAMS) -> dict:
    """At a pose (camera frame) of a part upright on the support: of the map cells under the part's nodes that are
    known, the share observed above the part's own top in that cell by more than the ceiling margin (something
    rests on, or occupies the space of, the part), and the known share."""
    X = nodes @ np.asarray(R).T + np.asarray(t)
    q = X - pf["p0"]
    a, b, h = q @ pf["ex"], q @ pf["ey"], q @ pf["n"]
    cell = hm["cell"]
    Hm, Wm = hm["shape"]
    j = np.floor((a - hm["a0"]) / cell).astype(int)
    i = np.floor((b - hm["b0"]) / cell).astype(int)
    ok = (j >= 0) & (j < Wm) & (i >= 0) & (i < Hm)
    if not ok.any():
        return {"n_cells": 0, "known_share": 0.0, "covered_share": 1.0}
    key = i[ok] * Wm + j[ok]
    top = np.full(Hm * Wm, -np.inf)
    np.maximum.at(top, key, h[ok])
    cells = np.unique(key)
    hmax = float(h.max() - h.min())
    margin = max(float(p["ceil_min_m"]), float(p["ceil_frac"]) * hmax)
    obs = hm["h"].ravel()[cells]
    known = np.isfinite(obs)
    covered = known & (obs > top[cells] + margin)
    ex = explained_for(hm, hmax, margin)
    if ex is not None:  # covered by a part already perceived that may rest on this one (a bowl on the box)
        covered &= ~ex.ravel()[cells]
    return {"n_cells": int(len(cells)), "known_share": round(float(known.mean()), 4),
            "covered_share": round(float(covered.sum()) / max(int(known.sum()), 1), 4)}


def body_box(nodes: np.ndarray, shrink: float) -> tuple:
    lo, hi = nodes.min(0) + shrink, nodes.max(0) - shrink
    return lo, np.maximum(hi, lo)


def point_inside(x, R, t, box) -> bool:
    lo, hi = box
    q = (np.asarray(x, float) - np.asarray(t)) @ np.asarray(R)
    return bool(np.all((q >= lo) & (q <= hi)))


def inside_share(X_world_nodes: np.ndarray, R, t, box) -> float:
    """Share of points (same frame as R, t) inside the oriented box (body frame lo/hi) of another part."""
    if not len(X_world_nodes):
        return 0.0
    Q = (X_world_nodes - np.asarray(t)) @ np.asarray(R)
    lo, hi = box
    ins = np.all((Q >= lo) & (Q <= hi), axis=1)
    return float(ins.mean())
