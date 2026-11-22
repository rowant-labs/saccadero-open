"""TRUTH-FREE: the fast layer as a library. In: one RGB-D frame (``frame.Frame``: RGB, raw depth, ``K``, depth
scale, optional camera-to-world) and its targets (object id or candidate ids, optional pixels per target). Out: per
target a pose, a score and a time (``Estimate``).

Steps (screens code imported unchanged unless ``provenance.COPIED`` says otherwise):

1. **Plane fit and crop** (once per frame). ``screens`` preset: ``e6/plane_crop.analyse`` (table plane, blob = every
   pixel more than 3 mm above it, crop), as E-0c. ``hope`` preset: ``plane_crop.backproject`` + ``fit_plane`` with a
   real-depth spec (no blob: the targets bring their pixels). A target's pixels are its given pixels AND valid depth,
   or the blob when none are given.
2. **Front end**: ``e0c_check_timing.frontend`` patch centres (seeded by the target's token) and their location,
   TLS normal and hsv, from the target's pixels only.
3. **Proposal** per candidate object: ``e0c_check_timing.proposal.propose`` (``screens``) or ``wide.propose``
   (``hope``: rotations over the sphere, 3D translation, clutter free space).
4. **CHECK**: ``e0c_check_timing.kernel.CheckKernel.score`` (Monty's one-observation evidence, ``lnh``) of every guess
   on the target's patches; ``kernel.decide`` picks (ties: the candidate order, then the guess order).
5. **Refine and decide**: ``e0f_refine.refine.icp`` (point-to-plane, Huber, visibility filter) on data from
   ``refine_data.extract_data``. Decision rules (all computed in one call, so the variants share every step):
   - ``A0``: CHECK's pick, not refined;
   - ``A`` (screens order, E-0c then E-0f): CHECK's pick, refined;
   - ``B`` (``hope`` primary): CHECK's best ``refine_topk`` guesses each refined, then the refined pose with the most
     CHECK evidence (ties: CHECK's order).
   The ``screens`` preset's primary is ``A`` with E-0f's ``fit_quality``; the ``hope`` preset's is ``B`` with
   ``refine_data.clutter_quality``.

6. **Flips** (step 3; presets ``hope_flips`` and ``hope_flips_gen``). Every refined base of ``B`` is expanded by
   its object's flip family (``symmetry``: half turns about the principal axes, n-fold turns, turns about a
   rotational axis, upside down); the flips are refined with the same ICP (``flip_refine = "each"``: every flip;
   ``"generators"``: the generators, and the turns about a rotational axis are applied to the refined generator),
   duplicates are dropped (``flip_dedup_*``), and every candidate is scored by CHECK and by the appearance scorer
   (``appearance``; default ``geometry``, a constant). The decision rule (``appearance.DecideRule``; default
   ``geometry``: the most CHECK evidence) picks ``B``. ``B0`` keeps step 2's B (the bases only), bit for bit.

7. **Speed** (step 4, ``Speed``; all off = the step-3 configuration, bit for bit, so ``hope_flips_gen`` stays the
   parity reference). Estimate-preserving: ``lean`` (E1: no clutter_quality, a diagnostic only), ``dup_vec``
   (amendment E6: the duplicate test vectorised, with the scalar test deciding every borderline pair),
   ``icp_bounded`` (amendment E7: the ICP's neighbour search bounded by its gate), ``check_bounded`` (amendment
   E8: CHECK's neighbour search bounded by Monty's match distance), ``wide.cache`` (E2).
   Approximating: the wide proposal's pruning (X1, X2: ``wide.WideOpts``), cheaper ICP (X3: ``refine_stages``,
   ``refine_n``, ``icp_k``, ``vis_every``, ``stage_stop``), flip pre-scoring (X4: ``flip_m``, ``flip_delta``), the
   plane fit (X5: ``plane_spec``) and ``refine_topk`` (X7). ``make_preset(name, overrides)`` applies a runner's
   ``--speed``; ``hope_fast`` is the accepted stack (``FAST_OVERRIDES``) and ``hope_fast_max`` every change.
   Their node grids (X6) are the layer's memory's own (fix:bop): the presets name no folder
   (``wide.GRIDS_OF_MEMORY``), ``FastLayer`` binds ``wide.GridSet`` for its memory, and every grid is checked against
   the memory (sha256, object set, node count, h, k) before it is used.

8. **Step 6: other cameras and CNOS masks** (``Preset.px_ref_f``, ``Preset.mask_trim``; the defaults keep every
   earlier preset bit for bit).
   - **Pixel-constant port (P).** Every image-space constant of the layer was set at HOPE's resolution (1920 x 1080,
     fx about 1390 px). With ``px_ref_f > 0`` each is scaled per frame by ``s = fx / px_ref_f``
     (``pixel_constants``): the front end's patch width (odd: ``2 rint((w s - 1) / 2) + 1``, at least 3), the
     refinement data's normal steps (``rint(step s)``, at least 1), the ICP visibility bin (``rint(3 s)``, at least
     1), the plane RANSAC stride (``rint(stride s)``, at least 1) and the minimum pixel count (an area:
     ``rint(min_pixels s^2)``, at least 1); ``rint(x) = floor(x + 0.5)``. At HOPE's fx (1390.53) with
     ``px_ref_f = 1390`` every value equals today's. Everything else is metric already (the wide proposal, CHECK,
     ICP gates, scorer A's footprints in mm). The dilated blob of ``RF.fit_quality`` (``FS_DILATE_PX``) is not on
     the ``hope`` presets' path and is not ported.
   - **Mask trim (S4).** ``mask_trim``: ``"none"``; ``"depth_cc"``: of the given pixels with valid depth keep the
     largest 4-connected component whose neighbouring pixels differ in depth by at most ``trim_dz_m``;
     ``"erode"``: the given pixels with valid depth eroded by a disc of radius ``rint(erode_m fx / z_med)`` px
     (``z_med``: their median depth; radius 0 = no change); if the trim leaves fewer than the minimum pixel count, the
     untrimmed pixels are used. The trimmed pixels are the target's pixels for every later step (front end,
     refinement data, scorer A's mask).

The score of a pose is its CHECK evidence per patch (Monty's summed evidence divided by the patch count).
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field, fields, replace

import numpy as np

from rowant_lab.hybrid_frontier.e6 import plane_crop as pc
from rowant_lab.hybrid_frontier.screens.e0c_check_timing import frontend, kernel, proposal
from rowant_lab.hybrid_frontier.screens.e0f_refine import refine as RF
from rowant_lab.hybrid_frontier.screens.e0f_refine import spec as S

from . import appearance as AP
from . import check as CHK
from . import icp as ICP
from . import refine_data, symmetry as SY, wide
from .frame import Frame, valid_depth
from .memory import Memory

TRUTH_FREE = True
VARIANTS = ("A0", "A", "B")
FLIP_VARIANTS = ("A0", "A", "B0", "B")  # with flips: B0 = step 2's B (bases only), B = with the flips
HOPE_PLANE_SPEC = {**pc.DEFAULT_SPEC, "inlier_m": 0.005, "ransac_iters": 256, "min_inlier_fraction": 0.10}


@dataclass(frozen=True)
class Speed:
    """Step-4 speed knobs (module doc, item 7). The defaults are the step-3 configuration."""

    lean: bool = False  # E1: skip clutter_quality (diagnostic only)
    dup_vec: bool = False  # E6 (amendment): vectorised duplicate test, exact
    icp_k: int = S.K_NEIGHBOURS  # X3: neighbours per data point in the ICP's correspondence search
    vis_every: int = 1  # X3: visibility z-buffer every n-th ICP iteration
    stage_stop: bool = False  # X3: E-0f's stop rule in every stage
    icp_bounded: bool = False  # E7 (amendment): the ICP's neighbour search bounded by the stage's gate, exact
    check_bounded: bool = False  # E8 (amendment): CHECK's neighbour search bounded by Monty's match distance, exact
    threaded_read: bool = False  # E5: the RGB decodes in a thread while the runner reads the depth and fits the plane
    flip_m: int = 0  # X4: refine only the best m flip starts per target (0: every start, as step 3)
    flip_delta: float = 0.05  # X4: ... plus every start within this of the best start's key
    wide: wide.WideOpts = wide.WideOpts()  # X1, X2, E2

    def record(self) -> dict:
        out = {k: v for k, v in self.__dict__.items() if k != "wide"}
        out["wide"] = self.wide.record()
        return out


@dataclass(frozen=True)
class Preset:
    name: str
    plane_mode: str  # "analyse" (plane + blob + crop) or "fit" (plane only)
    plane_spec: dict
    proposal: str  # "screens" or "wide"
    poses: int
    width: int
    n_centres: int
    refine_n: int
    refine_steps: tuple
    refine_stages: tuple
    refine_use_plane: bool
    refine_topk: int
    quality: str  # "fit_quality" or "clutter"
    primary: str  # "A" or "B"
    min_pixels: int = 30
    min_patches: int = 3
    inlier_gate_m: float = S.INLIER_GATE_M
    inlier_resid_m: float = S.INLIER_RESID_M
    fs_tol_m: float = wide.FS_TOL_M
    flips: bool = False  # step 3: expand the refined candidates by the object's flip family (``symmetry``)
    flip_refine: str = "each"  # "each": every flip refined with the ICP; "generators": turns inherit (``symmetry``)
    flip_dedup_deg: float = 5.0  # two candidates are one when their rotations differ by less than this ...
    flip_dedup_m: float = 0.010  # ... and their centres by less than this
    speed: Speed = Speed()  # step 4
    px_ref_f: float = 0.0  # step 6 (P): 0 = the pixel constants as set at HOPE's resolution; > 0: scaled by fx / px_ref_f
    mask_trim: str = "none"  # step 6 (S4): "none", "depth_cc" or "erode" (module doc, item 8)
    trim_dz_m: float = 0.010  # depth_cc: neighbouring pixels are connected when their depths differ by at most this
    erode_m: float = 0.002  # erode: the erosion radius on the surface (metres; pixels = rint(erode_m fx / z_med))

    def record(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            if k == "speed":
                out[k] = v.record()
                continue
            out[k] = [list(s) if isinstance(s, tuple) else s for s in v] if isinstance(v, tuple) else v
        return out


SCREENS = Preset(name="screens", plane_mode="analyse", plane_spec=dict(pc.DEFAULT_SPEC), proposal="screens",
                 poses=proposal.POSES_PER_OBJECT, width=frontend.DEFAULT_WIDTH, n_centres=frontend.DEFAULT_N_CENTRES,
                 refine_n=S.N_DATA, refine_steps=tuple(S.DATA_NORMAL_STEP_PX), refine_stages=tuple(S.STAGES),
                 refine_use_plane=True, refine_topk=3, quality="fit_quality", primary="A")
HOPE = Preset(name="hope", plane_mode="fit", plane_spec=HOPE_PLANE_SPEC, proposal="wide", poses=wide.POSES,
              width=15, n_centres=256, refine_n=1000, refine_steps=(10, 5),
              refine_stages=((4, 0.020, 0.004), (4, 0.010, 0.002), (12, 0.006, 0.001)), refine_use_plane=False,
              refine_topk=3, quality="clutter", primary="B", inlier_gate_m=0.006, inlier_resid_m=0.003)
HOPE_FLIPS = replace(HOPE, name="hope_flips", flips=True, flip_refine="each")
HOPE_FLIPS_GEN = replace(HOPE, name="hope_flips_gen", flips=True, flip_refine="generators")
PRESETS = {"screens": SCREENS, "hope": HOPE, "hope_flips": HOPE_FLIPS, "hope_flips_gen": HOPE_FLIPS_GEN}


FAST_GRID_DIR = wide.GRIDS_OF_MEMORY  # fix:bop: the grids registered for the layer's memory, never one dataset's folder
# Step 4's fast default: the estimate-preserving E1, E2, E5, E6, E7 and the
# approximating changes the tune side accepted in the declared order (X1b, X3b-d, X4 with m = 3, X5, X6, X7); left
# out (each would have taken the stack below the AR floor): X1a, X2, X3a, X4 with m = 2.
FAST_OVERRIDES = {
    "refine_n": 600, "refine_topk": 2, "plane_spec": {"ransac_stride": 16, "ransac_iters": 128},
    "speed": {"lean": True, "dup_vec": True, "icp_bounded": True, "threaded_read": True, "vis_every": 2,
              "icp_k": 4, "flip_m": 3, "flip_delta": 0.05,
              "wide": {"cache": True, "keep_after_first": 0.25, "grid_dir": FAST_GRID_DIR, "grid_h_mm": 2.5,
                       "grid_k": 1}}}
# ... and every speed change at once, the left-out ones too: labelled with its tune-side AR cost in the report.
FAST_MAX_OVERRIDES = {
    **FAST_OVERRIDES, "refine_stages": ((3, 0.020, 0.004), (3, 0.010, 0.002), (8, 0.006, 0.001)),
    "speed": {**FAST_OVERRIDES["speed"], "stage_stop": True, "flip_m": 2,
              "wide": {**FAST_OVERRIDES["speed"]["wide"], "n_sphere_downs": 32, "n_seeds": 4,
                       "refine_offsets_deg": (-7.5, 7.5)}}}


def _tupled(v):
    return tuple(_tupled(x) for x in v) if isinstance(v, (list, tuple)) else v


def make_preset(name: str | Preset, overrides: dict | None = None) -> Preset:
    """A preset with step-4 overrides: top-level fields (e.g. ``refine_n``, ``refine_stages``, ``plane_spec`` (merged
    into the preset's), ``refine_topk``), ``speed`` (a dict of ``Speed`` fields) and ``speed.wide`` (a dict of
    ``wide.WideOpts`` fields). Unknown names are refused."""
    p = PRESETS[name] if isinstance(name, str) else name
    if not overrides:
        return p
    ov = dict(overrides)
    known = {f.name for f in fields(Preset)}
    bad = sorted(set(ov) - known - {"name"})
    if bad:
        raise ValueError(f"unknown preset fields {bad}")
    sp = ov.pop("speed", None)
    if "plane_spec" in ov:
        ov["plane_spec"] = {**p.plane_spec, **ov["plane_spec"]}
    for k in ("refine_stages", "refine_steps"):
        if k in ov:
            ov[k] = _tupled(ov[k])
    if sp:
        sp = dict(sp)
        wd = sp.pop("wide", None)
        sk = {f.name for f in fields(Speed)}
        bad = sorted(set(sp) - sk)
        if bad:
            raise ValueError(f"unknown speed fields {bad}")
        speed = replace(p.speed, **sp)
        if wd:
            wk = {f.name for f in fields(wide.WideOpts)}
            bad = sorted(set(wd) - wk)
            if bad:
                raise ValueError(f"unknown wide options {bad}")
            wd = {k: (_tupled(v) if isinstance(v, list) else v) for k, v in wd.items()}
            speed = replace(speed, wide=replace(speed.wide, **wd))
        ov["speed"] = speed
    return replace(p, **ov)


HOPE_FAST = make_preset(HOPE_FLIPS_GEN, {**FAST_OVERRIDES, "name": "hope_fast"})
HOPE_FAST_MAX = make_preset(HOPE_FLIPS_GEN, {**FAST_MAX_OVERRIDES, "name": "hope_fast_max"})
PRESETS.update({"hope_fast": HOPE_FAST, "hope_fast_max": HOPE_FAST_MAX})


@dataclass
class Target:
    obj_ids: tuple  # one id (BOP) or candidate ids in prior order (screens: HPG's top 3)
    pixels: np.ndarray | None = None  # (H, W) bool, the target's pixels; None = the plane blob
    seed: str | None = None  # hex token for the seeded draws; default: from key and ids
    prior_rank: dict | None = None  # CHECK tie-break ranks; default: the order of obj_ids
    key: object = None  # caller's label, echoed back


@dataclass
class Estimate:
    key: object
    status: str  # "ok" or why there is no pose
    obj_id: str | None = None
    R: np.ndarray | None = None  # model -> world
    t: np.ndarray | None = None  # metres, world
    score: float | None = None
    time_s: float = 0.0
    variants: dict = field(default_factory=dict)  # name -> {"obj", "R", "t", "score"}
    diag: dict = field(default_factory=dict)

    def camera_pose_mm(self, frame: Frame, variant: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(R, t in mm), model -> camera, for BOP."""
        v = self.variants[variant] if variant else {"R": self.R, "t": self.t}
        R, t = frame.to_camera(v["R"], v["t"])
        return R, t * 1000.0


@dataclass
class FrameResult:
    estimates: list
    diag: dict
    parts_ms: dict
    time_s: float


def token_for(key, obj_ids) -> str:
    return hashlib.sha256(f"fastlayer:{key!r}:{list(obj_ids)!r}".encode()).hexdigest()[:16]


def _ns_ms(a: int, b: int) -> float:
    return (b - a) / 1e6


def _pose_rec(R, t) -> dict:
    return {"R": np.asarray(R).round(9).tolist(), "t": np.asarray(t).round(9).tolist()}


MASK_TRIMS = ("none", "depth_cc", "erode")


def _rint(x: float) -> int:
    return int(math.floor(float(x) + 0.5))


def pixel_constants(p: Preset, fx: float) -> dict:
    """The image-space constants of preset ``p`` for a camera of focal length ``fx`` (module doc, item 8)."""
    stride = int(p.plane_spec.get("ransac_stride", pc.DEFAULT_SPEC["ransac_stride"]))
    base = {"width": int(p.width), "refine_steps": tuple(int(x) for x in p.refine_steps),
            "min_pixels": int(p.min_pixels), "vis_bin_px": int(S.VIS_BIN_PX), "ransac_stride": stride, "scale": None}
    if not p.px_ref_f:
        return base
    s = float(fx) / float(p.px_ref_f)
    steps = []
    for st in p.refine_steps:
        v = max(1, _rint(st * s))
        if v not in steps:
            steps.append(v)
    return {"width": max(3, 2 * _rint((p.width * s - 1.0) / 2.0) + 1), "refine_steps": tuple(steps),
            "min_pixels": max(1, _rint(p.min_pixels * s * s)), "vis_bin_px": max(1, _rint(S.VIS_BIN_PX * s)),
            "ransac_stride": max(1, _rint(stride * s)), "scale": s}


def trim_depth_cc(px: np.ndarray, z_m: np.ndarray, dz_m: float) -> np.ndarray:
    """``mask_trim = "depth_cc"``: the largest 4-connected component of ``px`` (bool, (H, W)) whose neighbouring
    pixels differ in observed depth (``z_m``, metres, finite on ``px``) by at most ``dz_m``. Ties: the component whose
    first pixel comes first in row-major order."""
    from scipy.sparse import coo_matrix  # noqa: PLC0415
    from scipy.sparse.csgraph import connected_components  # noqa: PLC0415

    rr = np.flatnonzero(px.any(axis=1))
    if len(rr) == 0:
        return px.copy()
    cc = np.flatnonzero(px[rr[0]:rr[-1] + 1].any(axis=0))
    r0, r1, c0, c1 = int(rr[0]), int(rr[-1]) + 1, int(cc[0]), int(cc[-1]) + 1
    m = px[r0:r1, c0:c1]
    z = np.where(m, z_m[r0:r1, c0:c1], np.nan)
    idx = np.full(m.shape, -1, dtype=np.int64)
    n = int(m.sum())
    idx[m] = np.arange(n)
    rows, cols = [], []
    for a, b, za, zb in ((idx[:, :-1], idx[:, 1:], z[:, :-1], z[:, 1:]), (idx[:-1, :], idx[1:, :], z[:-1, :], z[1:, :])):
        ok = (a >= 0) & (b >= 0) & (np.abs(za - zb) <= dz_m)
        rows.append(a[ok])
        cols.append(b[ok])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    g = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    size = np.bincount(lab)
    keep = int(np.argmax(size))  # the first label of the largest size; labels follow row-major first pixels
    out = np.zeros_like(px)
    sub = np.zeros(m.shape, dtype=bool)
    sub[m] = lab == keep
    out[r0:r1, c0:c1] = sub
    return out


def trim_erode(px: np.ndarray, z_m: np.ndarray, erode_m: float, fx: float) -> tuple[np.ndarray, int]:
    """``mask_trim = "erode"``: ``px`` eroded by a disc of radius ``rint(erode_m fx / z_med)`` px (returns the mask and
    the radius)."""
    from scipy.ndimage import distance_transform_edt  # noqa: PLC0415

    zm = float(np.median(z_m[px]))
    rad = _rint(float(erode_m) * float(fx) / max(zm, 1e-3))
    if rad <= 0:
        return px.copy(), 0
    rr = np.flatnonzero(px.any(axis=1))
    cc = np.flatnonzero(px[rr[0]:rr[-1] + 1].any(axis=0))
    r0, r1, c0, c1 = int(rr[0]), int(rr[-1]) + 1, int(cc[0]), int(cc[-1]) + 1
    # erosion by the disc {o : |o|^2 <= rad^2} (outside the image counts as off) == the pixels whose exact Euclidean
    # distance to the nearest off pixel exceeds rad (the crop is padded by one off pixel)
    sub = np.pad(px[r0:r1, c0:c1], 1, constant_values=False)
    keep = distance_transform_edt(sub)[1:-1, 1:-1] > rad
    out = np.zeros_like(px)
    out[r0:r1, c0:c1] = keep
    return out, rad


def check_order(scores, guesses, prior_rank: dict) -> list[int]:
    """Guess indices from CHECK's best down, with ``kernel.decide``'s tie rule."""
    return sorted(range(len(guesses)), key=lambda i: (-float(scores[i]), prior_rank.get(guesses[i][0], 0), i))


class FastLayer:
    def __init__(self, memory: Memory | None, preset: Preset | str = "hope", *, symmetry: SY.SymmetryTable | None = None,
                 scorer=None, rule: AP.DecideRule | None = None):
        """``memory`` may be None for frame-level work only (``frame_context``). Step 3: ``symmetry`` (the flip
        families; needed by a preset with ``flips``), ``scorer`` (an appearance scorer, default geometry only) and
        ``rule`` (``appearance.DecideRule``, default ``geometry``)."""
        self.memory = memory
        self.preset = PRESETS[preset] if isinstance(preset, str) else preset
        if self.preset.flips and symmetry is None:
            raise ValueError(f"preset {self.preset.name!r} expands flips: it needs a symmetry table")
        if self.preset.flip_refine not in ("each", "generators"):
            raise ValueError(f"unknown flip_refine {self.preset.flip_refine!r}")
        if self.preset.mask_trim not in MASK_TRIMS:
            raise ValueError(f"unknown mask_trim {self.preset.mask_trim!r}; known: {MASK_TRIMS}")
        if self.preset.px_ref_f < 0:
            raise ValueError("px_ref_f must be >= 0")
        self.symmetry = symmetry
        self.scorer = scorer if scorer is not None else AP.GeometryOnly()
        self.rule = rule if rule is not None else AP.DecideRule("geometry")
        self._assets = None
        self._wide_cache: dict = {}  # step 4: per-object tables of the wide proposal (per process)
        wo = self.preset.speed.wide
        self.grids = None  # X6: the memory's node grids, checked against it (fix:bop)
        if wo.grid_dir and memory is not None:
            self.grids = wide.GridSet.resolve(wo.grid_dir, memory.npz_sha256, memory.ids, wo.grid_h_mm, wo.grid_k)

    def _check(self, guesses, ps):
        """CHECK's summed evidence of each guess: the screens' kernel, or its bounded copy (E8, the same numbers)."""
        if self.preset.speed.check_bounded:
            return CHK.score(self.memory.kernel, guesses, ps)
        return self.memory.kernel.score(guesses, ps)

    @property
    def assets(self) -> dict:
        if self._assets is None:
            self._assets = AP.assets_from_memory(self.memory, self.symmetry)
        return self._assets

    # ------------------------------------------------------------------------------------- per frame
    def frame_context(self, frame: Frame) -> dict:
        """Plane fit (and, for ``screens``, blob and crop), once per frame."""
        p = self.preset
        cam = frame.camera()
        ctx: dict = {"camera": cam}
        if p.plane_mode == "analyse":
            res = pc.analyse(frame.rgb, frame.depth, cam, p.plane_spec)
            ctx["analyse"] = res
            ctx["plane"] = res["plane"] if res["status"] != "no_plane" else None
            ctx["blob"] = res["mask"]
            ctx["plane_status"] = res["status"]
        else:
            return self.frame_context_depth(frame.depth, cam)
        if p.proposal == "wide" or p.quality == "clutter":
            ctx["z_obs"] = refine_data.observed_depth_m(frame.depth, cam)
        ctx["camgeom"] = RF.CameraGeom.from_camera(cam)
        ctx["valid"] = frame.valid_depth()
        return ctx

    def frame_context_depth(self, depth: np.ndarray, cam: dict) -> dict:
        """``frame_context`` of a ``fit`` preset from the raw depth and the camera dict alone (no RGB needed, so a
        runner can start it before the RGB is decoded: E5)."""
        p = self.preset
        if p.plane_mode != "fit":
            raise ValueError("frame_context_depth is for presets whose plane mode is 'fit'")
        ctx: dict = {"camera": cam}
        pts = pc.backproject(pc.depth_metres(depth, cam), cam)
        spec = p.plane_spec
        if p.px_ref_f:  # step 6 (P): the RANSAC stride scaled with the focal length
            stride = pixel_constants(p, cam["fx"])["ransac_stride"]
            if stride != int(spec["ransac_stride"]):
                spec = {**spec, "ransac_stride": stride}
        try:
            ctx["plane"] = pc.fit_plane(pts, spec)
            ctx["plane_status"] = "ok"
        except pc.PlaneError as exc:
            ctx["plane"] = None
            ctx["plane_status"] = f"no_plane: {exc}"
        ctx["blob"] = None
        if p.proposal == "wide" or p.quality == "clutter":
            ctx["z_obs"] = refine_data.observed_depth_m(depth, cam)
        ctx["camgeom"] = RF.CameraGeom.from_camera(cam)
        ctx["valid"] = valid_depth(depth)
        return ctx

    def estimate(self, frame: Frame, targets) -> FrameResult:
        t0 = time.perf_counter_ns()
        ctx = self.frame_context(frame)
        t1 = time.perf_counter_ns()
        ests = [self.estimate_target(frame, ctx, tg) for tg in targets]
        t2 = time.perf_counter_ns()
        diag = {"plane_status": ctx["plane_status"], "plane": ctx["plane"], "n_targets": len(ests)}
        return FrameResult(estimates=ests, diag=diag, parts_ms={"frame_context": _ns_ms(t0, t1),
                                                                "targets": _ns_ms(t1, t2)},
                           time_s=(t2 - t0) / 1e9)

    # ------------------------------------------------------------------------------------- per target
    def target_pixels(self, frame: Frame, ctx: dict, tg: Target) -> tuple[np.ndarray | None, dict]:
        if tg.pixels is None:
            px = ctx["blob"]
            return px, {"source": "plane_blob", "n_px": int(px.sum()) if px is not None else 0}
        given = np.asarray(tg.pixels, bool)
        if given.shape != frame.depth.shape:
            raise ValueError(f"target pixels {given.shape} do not match the frame {frame.depth.shape}")
        px = given & ctx["valid"]
        n_g, n_v = int(given.sum()), int(px.sum())
        rec = {"source": "given", "n_px_given": n_g, "n_px": n_v, "valid_share": (n_v / n_g) if n_g else 0.0}
        p = self.preset
        if p.mask_trim == "none" or n_v == 0:
            return px, rec
        z = ctx["z_obs"]
        if p.mask_trim == "depth_cc":
            tr = trim_depth_cc(px, z, p.trim_dz_m)
            rad = None
        else:
            tr, rad = trim_erode(px, z, p.erode_m, frame.camera()["fx"])
        n_t = int(tr.sum())
        min_px = pixel_constants(p, frame.camera()["fx"])["min_pixels"]
        used = n_t >= min_px
        rec.update({"trim": p.mask_trim, "n_px_untrimmed": n_v, "n_px_trimmed": n_t, "trim_used": bool(used),
                    "erode_px": rad})
        if used:
            rec["n_px"] = n_t
            return tr, rec
        return px, rec

    def estimate_target(self, frame: Frame, ctx: dict, tg: Target) -> Estimate:
        p = self.preset
        mem = self.memory
        cam = ctx["camera"]
        objs = [str(g) for g in tg.obj_ids]
        for g in objs:
            if g not in mem.kernel.graphs:
                raise KeyError(f"object {g!r} is not in the memory")
        token = tg.seed or token_for(tg.key, objs)
        prior_rank = tg.prior_rank or {g: i + 1 for i, g in enumerate(objs)}
        ms: dict = {}
        a = time.perf_counter_ns()
        px, pdiag = self.target_pixels(frame, ctx, tg)
        pxc = pixel_constants(p, cam["fx"])
        diag = {"token": token, "objs": objs, "pixels": pdiag, "plane_status": ctx["plane_status"]}
        if p.px_ref_f:
            diag["px_constants"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in pxc.items()}
        if px is None or pdiag["n_px"] < pxc["min_pixels"]:
            return Estimate(key=tg.key, status="too_few_pixels", time_s=(time.perf_counter_ns() - a) / 1e9,
                            diag=diag)
        centres = frontend.sample_centres(px, token, n=p.n_centres, width=pxc["width"])
        f = frontend.patch_features(frame.rgb, frame.depth, px, cam, centres, width=pxc["width"])
        v = f["valid"]
        ps = mem.kernel.patches(f["location"][v], f["normal"][v], f["hsv"][v])
        b = time.perf_counter_ns()
        ms["front"] = _ns_ms(a, b)
        diag["n_centres"], diag["n_patches"] = int(len(f["centres"])), ps.n
        if ps.n < p.min_patches:
            return Estimate(key=tg.key, status="too_few_patches", time_s=(b - a) / 1e9, diag=diag)
        props = {}
        if p.proposal == "screens":
            fr = proposal.make_frame(ctx["plane"], cam, px, ps.loc)
            for g in objs:
                props[g] = proposal.propose(mem.shapes[g], fr, poses=p.poses)
        else:
            wf = wide.make_frame(ctx["plane"], cam, ps.loc, ctx["z_obs"])
            wo = p.speed.wide
            if wo == wide.DEFAULT_OPTS:
                for g in objs:
                    props[g] = wide.propose(mem.shapes[g], wf, poses=p.poses)
            else:
                ng = None
                if wo.normal_prune:
                    ng = ps.normal[proposal._stride(ps.n, wide.N_ICP_COARSE)] @ wf.R_g  # noqa: SLF001
                for g in objs:
                    props[g] = wide.propose(mem.shapes[g], wf, poses=p.poses, opts=wo, cache=self._wide_cache,
                                            normals_g=ng, node_normals=mem.kernel.graphs[g].normal, grids=self.grids)
        guesses = [gq for g in objs for gq in props[g]["guesses"]]
        c = time.perf_counter_ns()
        ms["proposal"] = _ns_ms(b, c)
        tup = [(gq["obj"], gq["R"], gq["t"]) for gq in guesses]
        scores = self._check(tup, ps)
        pick = kernel.decide(scores, tup, prior_rank)
        order = check_order(scores, tup, prior_rank)
        d = time.perf_counter_ns()
        ms["check"] = _ns_ms(c, d)
        plane_ref = ctx["plane"] if p.refine_use_plane else None
        frd = refine_data.extract_data(frame.depth, px, cam, plane_ref, token, n=p.refine_n,
                                       steps=pxc["refine_steps"], cam=ctx["camgeom"])
        sp = p.speed
        cfg = RF.IcpConfig(stages=tuple(p.refine_stages), k=int(sp.icp_k))
        top = order[:max(1, p.refine_topk)] if p.primary == "B" else [pick]
        if pick not in top:
            top = [pick] + top[:-1]
        refined, refine_ms = {}, {}
        d2 = time.perf_counter_ns()
        ms["refine_data"] = _ns_ms(d, d2)
        for i in top:
            gq = guesses[i]
            r0 = time.perf_counter_ns()
            refined[i] = ICP.icp(mem.clouds[gq["obj"]], frd, gq["R"], gq["t"], cfg, vis_every=sp.vis_every,
                                 stage_stop=sp.stage_stop, bounded=sp.icp_bounded, vis_bin_px=pxc["vis_bin_px"])
            refine_ms[i] = _ns_ms(r0, time.perf_counter_ns())
        e = time.perf_counter_ns()
        ms["refine"] = _ns_ms(d2, e)
        rtup = [(guesses[i]["obj"], refined[i]["R"], refined[i]["t"]) for i in top]
        rscores = self._check(rtup, ps)
        best = max(range(len(top)), key=lambda j: (float(rscores[j]), -j))
        quality = {}
        for j, i in enumerate(top):
            model = mem.clouds[guesses[i]["obj"]]
            if sp.lean:
                quality[i] = None
            elif p.quality == "fit_quality":
                quality[i] = RF.fit_quality(model, frd, refined[i]["R"], refined[i]["t"])
            else:
                quality[i] = refine_data.clutter_quality(model, frd, refined[i]["R"], refined[i]["t"], ctx["z_obs"],
                                                         inlier_gate_m=p.inlier_gate_m,
                                                         inlier_resid_m=p.inlier_resid_m, fs_tol_m=p.fs_tol_m)
        g_ = time.perf_counter_ns()
        ms["decide"] = _ns_ms(e, g_)
        n = float(ps.n)
        ia = top.index(pick)
        variants = {
            "A0": {"obj": guesses[pick]["obj"], "R": guesses[pick]["R"], "t": guesses[pick]["t"],
                   "score": float(scores[pick]) / n},
            "A": {"obj": guesses[pick]["obj"], "R": refined[pick]["R"], "t": refined[pick]["t"],
                  "score": float(rscores[ia]) / n},
        }
        if p.primary == "B":
            ib = top[best]
            variants["B"] = {"obj": guesses[ib]["obj"], "R": refined[ib]["R"], "t": refined[ib]["t"],
                             "score": float(rscores[best]) / n}
        flip_rec = None
        if p.flips and p.primary == "B":
            variants["B0"] = variants["B"]
            g0_ = time.perf_counter_ns()
            flip_rec = self._expand(frame, ctx, px, ps, frd, cfg, guesses, top, refined, rscores, quality, ms,
                                    vis_bin_px=pxc["vis_bin_px"])
            g_ = time.perf_counter_ns()
            ms["flips_total"] = _ns_ms(g0_, g_)
            fb = flip_rec["cands"][flip_rec["best"]]
            variants["B"] = {"obj": fb["obj"], "R": fb["R"], "t": fb["t"], "score": float(flip_rec["geo"][flip_rec["best"]]) / n}
        prim = variants[p.primary]
        pr = {g: props[g]["prune"] for g in objs if "prune" in props[g]}
        if pr:
            diag["prune"] = pr
        diag.update({
            "n_hyp": {g: [props[g]["n_hyp_coarse"], props[g]["n_hyp_fine"]] for g in objs},
            "gravity": props[objs[0]].get("gravity", "plane"),
            "guesses": [{**{k: v_ for k, v_ in gq.items() if k not in ("R", "t")}, **_pose_rec(gq["R"], gq["t"]),
                         "check": float(scores[i]) / n} for i, gq in enumerate(guesses)],
            "pick": int(pick), "check_order": [int(i) for i in order],
            "refined": [{"guess": int(i), **_pose_rec(refined[i]["R"], refined[i]["t"]),
                         "status": refined[i]["status"], "iters": int(refined[i]["iters"]),
                         "n_corr": int(refined[i]["n_corr"]), "check": float(rscores[j]) / n,
                         "ms": round(refine_ms[i], 3), "quality": quality[i]} for j, i in enumerate(top)],
            "b_choice": int(top[best]) if p.primary == "B" else None,
            "data": frd.diag, "parts_ms": {k: round(v_, 3) for k, v_ in ms.items()},
        })
        total = (g_ - a) / 1e9
        a0_ms = ms["front"] + ms["proposal"] + ms["check"]
        diag["variant_ms"] = {"A0": round(a0_ms, 3), "A": round(a0_ms + ms["refine_data"] + refine_ms[pick], 3),
                              "B": round(total * 1e3, 3)}
        if flip_rec is not None:
            diag["variant_ms"]["B0"] = round(total * 1e3 - ms["flips_total"], 3)
            diag["b0_choice"] = diag["b_choice"]
            diag["b_choice"] = int(flip_rec["best"])  # index into diag["refined"] (bases first, then the flips)
            diag["refined"] = diag["refined"] + flip_rec["records"]
            for j, rr in enumerate(diag["refined"]):
                rr.setdefault("label", "I")
                rr.setdefault("base", j)
                rr.setdefault("mode", "base")
                rr["app"] = flip_rec["app"][j]
            diag["flips"] = flip_rec["summary"]
            diag["decision"] = flip_rec["decision"]
        out = Estimate(key=tg.key, status="ok", obj_id=prim["obj"], R=np.asarray(prim["R"]), t=np.asarray(prim["t"]),
                       score=prim["score"], time_s=total, variants=variants, diag=diag)
        out.diag["_raw"] = {"refined": refined, "scores": scores, "guesses": guesses, "rscores": rscores, "top": top,
                            "frd": frd, "ps": ps, "front": f, "pixels": px, "flips": flip_rec}
        return out

    # ------------------------------------------------------------------------------------- step 3: flips
    def _expand(self, frame: Frame, ctx: dict, px, ps, frd, cfg, guesses, top, refined, rscores, quality,
                ms: dict, vis_bin_px: int = S.VIS_BIN_PX) -> dict:
        """Expand every refined base by its object's flip family (``symmetry``), refine the flips the same way
        (``flip_refine``: ``each`` or ``generators``), drop duplicates (``flip_dedup_*``), score every candidate with
        CHECK and the appearance scorer, and choose with the decision rule. Bases keep CHECK's order; the flips
        follow, base by base, in family order."""
        p = self.preset
        mem = self.memory
        t0 = time.perf_counter_ns()
        cos_lim = np.cos(np.radians(p.flip_dedup_deg))
        cands = _Pool() if p.speed.dup_vec else []
        for j, i in enumerate(top):
            cands.append({"obj": guesses[i]["obj"], "R": np.asarray(refined[i]["R"]), "t": np.asarray(refined[i]["t"]),
                          "label": "I", "base": j, "guess": int(i), "mode": "base", "near": True, "dev": 0.0})
        n_base = len(cands)
        summ = {"mode": p.flip_refine, "n_bases": n_base, "n_starts": 0, "n_dup_start": 0, "n_dup_refined": 0,
                "n_icp": 0, "n_derived": 0, "objects_without_symmetry": []}

        def centre_of(c):
            return SY.centre_world(c["R"], c["t"], self.symmetry.get(c["obj"])["centre"])

        sp = p.speed

        def dup_scalar(R, cw, obj, pool) -> bool:
            for q in pool:
                if q["obj"] != obj:
                    continue
                if np.linalg.norm(cw - q["_c"]) >= p.flip_dedup_m:
                    continue
                if (np.trace(R.T @ q["R"]) - 1.0) / 2.0 > cos_lim:
                    return True
            return False

        def dup(R, cw, obj, pool) -> bool:
            if not sp.dup_vec or len(pool) < 8:
                return dup_scalar(R, cw, obj, pool)
            # E6: a vectorised screen with a margin; the scalar test decides on the pairs it keeps (exact)
            ix = _DupIndex.of(pool)
            near = ix.screen(R, cw, obj, p.flip_dedup_m, cos_lim)
            return dup_scalar(R, cw, obj, [pool[k] for k in near]) if len(near) else False

        for c in cands:
            if c["obj"] in self.symmetry:
                c["_c"] = centre_of(c)
            else:
                c["_c"] = np.asarray(c["t"])
                if c["obj"] not in summ["objects_without_symmetry"]:
                    summ["objects_without_symmetry"].append(c["obj"])
        icp_ms = []

        def refine_start(st):
            r0 = time.perf_counter_ns()
            res = ICP.icp(mem.clouds[st["obj"]], frd, st["R0"], st["t0"], cfg, vis_every=sp.vis_every,
                          stage_stop=sp.stage_stop, bounded=sp.icp_bounded, vis_bin_px=vis_bin_px)
            icp_ms.append(_ns_ms(r0, time.perf_counter_ns()))
            summ["n_icp"] += 1
            return res

        def add_refined(st, res, mode):
            R, t = np.asarray(res["R"]), np.asarray(res["t"])
            cw = SY.centre_world(R, t, self.symmetry.get(st["obj"])["centre"])
            if dup(R, cw, st["obj"], cands):
                summ["n_dup_refined"] += 1
                return None
            c = {k: v for k, v in st.items() if k not in ("R0", "t0")}
            c.update({"R": R, "t": t, "_c": cw, "mode": mode, "icp": {k: res.get(k) for k in ("status", "iters",
                                                                                              "n_corr")}})
            cands.append(c)
            return c

        bases = list(cands)
        prescore = sp.flip_m > 0
        all_starts = _Pool() if sp.dup_vec else []  # X4: the starts of every base (dup-checked across bases)

        def make_starts(b, sym):
            cen, fam = sym["centre"], sym["family"]
            if p.flip_refine == "each":
                elems = fam
            else:
                seen, elems = set(), []
                for e in fam:
                    if e["gen"] != "I" and e["gen"] not in seen:
                        seen.add(e["gen"])
                        elems.append({**e, "R": e["G"], "label": e["gen"], "turn_deg": 0.0})
            starts = _Pool() if sp.dup_vec else []
            for e in elems:
                R0, t0_ = SY.flip_pose(b["R"], b["t"], e["R"], cen)
                cw = SY.centre_world(R0, t0_, cen)
                summ["n_starts"] += 1
                if dup(R0, cw, b["obj"], cands) or dup(R0, cw, b["obj"], all_starts if prescore else starts):
                    summ["n_dup_start"] += 1
                    continue
                st = {"obj": b["obj"], "R": R0, "R0": R0, "t0": t0_, "_c": cw, "label": e["label"],
                      "base": b["base"], "guess": b["guess"], "near": bool(e["near"]),
                      "dev": float(e["dev_frac"]), "gen": e["gen"]}
                starts.append(st)
                if prescore:
                    all_starts.append(st)
            return starts

        def refine_and_turn(b, sym, starts):
            cen, fam = sym["centre"], sym["family"]
            made = [(st, add_refined(st, refine_start(st), "icp")) for st in starts]
            if p.flip_refine == "generators" and sym["rot_axis"] is not None:
                axis = sym["axes"][:, int(sym["rot_axis"])]
                parents = [("I", b)] + [(st["gen"], c) for st, c in made if c is not None]
                for gname, par in parents:
                    for e in fam:
                        if e["gen"] != gname or not e["turn_deg"]:
                            continue
                        Z = SY.axis_rotation(axis, e["turn_deg"])
                        R, t = SY.flip_pose(par["R"], par["t"], Z, cen)
                        cw = SY.centre_world(R, t, cen)
                        if dup(R, cw, b["obj"], cands):
                            summ["n_dup_refined"] += 1
                            continue
                        cands.append({"obj": b["obj"], "R": R, "t": t, "_c": cw, "label": e["label"], "base": b["base"],
                                      "guess": b["guess"], "near": bool(e["near"]), "dev": float(e["dev_frac"]),
                                      "gen": gname, "mode": "derived"})
                        summ["n_derived"] += 1

        if not prescore:  # step 3: base by base (each base's starts are checked against the earlier bases' flips)
            for b in bases:
                if b["obj"] not in self.symmetry:
                    continue
                sym = self.symmetry.get(b["obj"])
                refine_and_turn(b, sym, make_starts(b, sym))
        else:  # X4: every base's starts first, pre-scored together; only the kept ones are refined
            per_base = [(b, self.symmetry.get(b["obj"])) for b in bases if b["obj"] in self.symmetry]
            per_base = [(b, sym, make_starts(b, sym)) for b, sym in per_base]
            chosen = self._prescore_flips(frame, px, ps, per_base, summ)
            for b, sym, starts in per_base:
                refine_and_turn(b, sym, [st for st in starts if id(st) in chosen])
        t1 = time.perf_counter_ns()
        flips = cands[n_base:]
        fscores = self._check([(c["obj"], c["R"], c["t"]) for c in flips], ps) if flips else np.zeros(0)
        geo = [float(x) for x in rscores] + [float(x) for x in fscores]
        nps = float(ps.n)
        acands = [AP.Candidate(obj=c["obj"], R=c["R"], t=c["t"], label=c["label"], base=int(c["base"]),
                               geo=geo[k] / nps, info={"mode": c["mode"], "near": c["near"], "dev": c["dev"]})
                  for k, c in enumerate(cands)]
        t2 = time.perf_counter_ns()
        app = AP.score_all(self.scorer, acands, frame, px, self.assets.get(cands[0]["obj"]))
        t3 = time.perf_counter_ns()
        best, drec = AP.decide(geo, app, self.rule, n_patches=ps.n)
        if best >= n_base and "quality" not in cands[best] and not sp.lean:
            c = cands[best]
            cands[best]["quality"] = refine_data.clutter_quality(
                mem.clouds[c["obj"]], frd, c["R"], c["t"], ctx["z_obs"], inlier_gate_m=p.inlier_gate_m,
                inlier_resid_m=p.inlier_resid_m, fs_tol_m=p.fs_tol_m)
        t4 = time.perf_counter_ns()
        ms["flip_refine"] = _ns_ms(t0, t1)
        ms["flip_check"] = _ns_ms(t1, t2)
        ms["appearance"] = _ns_ms(t2, t3)
        ms["flip_decide"] = _ns_ms(t3, t4)
        summ.update({"n_candidates": len(cands), "n_flips": len(flips),
                     "icp_ms_mean": round(float(np.mean(icp_ms)), 3) if icp_ms else None})
        records = []
        for k, c in enumerate(flips, start=n_base):
            rec = {"guess": int(c["guess"]), **_pose_rec(c["R"], c["t"]), "check": geo[k] / nps, "label": c["label"],
                   "base": int(c["base"]), "mode": c["mode"], "near": bool(c["near"]), "dev": round(c["dev"], 6)}
            if "icp" in c:
                rec.update({"status": c["icp"]["status"], "iters": int(c["icp"]["iters"]),
                            "n_corr": int(c["icp"]["n_corr"])})
            if "quality" in c:
                rec["quality"] = c["quality"]
            records.append(rec)
        for c in cands:
            c.pop("_c", None)
        return {"cands": cands, "geo": geo, "app": app, "best": int(best), "records": records, "summary": summ,
                "decision": {**drec, "rule": self.rule.record(), "scorer": getattr(self.scorer, "name", None),
                             "gate_size": len(drec["gate"]) if drec.get("gate") is not None else None}}


    def _prescore_flips(self, frame: Frame, px, ps, per_base: list, summ: dict) -> set:
        """X4: every unrefined flip start of the target gets the decision rule's key (CHECK per patch + weight x
        appearance, as ``appearance.decide``'s weighted rule; under the ``geometry`` rule the CHECK alone) and the
        wide proposal's cheap fit (mean capped nearest-node distance of 64 patch centres, mm; a tie-break). Kept:
        the best ``flip_m`` starts, plus every start within ``flip_delta`` of the best key. Returns their ids."""
        sp = self.preset.speed
        mem = self.memory
        allst = [st for _, _, starts in per_base for st in starts]
        summ["n_prescored"] = len(allst)
        if len(allst) <= sp.flip_m:
            summ["n_prescore_kept"] = len(allst)
            return {id(st) for st in allst}
        geo = self._check([(st["obj"], st["R0"], st["t0"]) for st in allst], ps)
        nps = float(ps.n)
        w = float(self.rule.weight) if self.rule.name == "weighted" else 0.0
        if w:
            obj = allst[0]["obj"]
            acands = [AP.Candidate(obj=st["obj"], R=st["R0"], t=st["t0"], label=st["label"], base=int(st["base"]),
                                   geo=float(geo[k]) / nps, info={"mode": "start"}) for k, st in enumerate(allst)]
            app = AP.score_all(self.scorer, acands, frame, px, self.assets.get(obj))
        else:
            app = [None] * len(allst)
        key = [float(geo[k]) / nps + w * (0.0 if app[k] is None else app[k]) for k in range(len(allst))]
        yc = ps.loc[proposal._stride(ps.n, wide.N_ICP_COARSE)]  # noqa: SLF001
        fit = []
        for st in allst:
            shape = mem.shapes[st["obj"]]
            local = (yc - st["t0"]) @ st["R0"] + proposal.P_TRAIN
            d, _ = shape.tree.query(local, k=1, workers=1)
            fit.append(float(np.minimum(d, wide.FIT_CAP_M).mean() * 1e3))
        order = sorted(range(len(allst)), key=lambda k: (-key[k], fit[k], k))
        top = key[order[0]]
        keep = set(order[:sp.flip_m]) | {k for k in range(len(allst)) if key[k] >= top - sp.flip_delta}
        summ["n_prescore_kept"] = len(keep)
        return {id(allst[k]) for k in keep}


class _DupIndex:
    """E6: the candidates of a duplicate-test pool as arrays (built once per pool list, extended as it grows)."""

    _MARGIN_M = 1e-9
    _MARGIN_COS = 1e-9

    def __init__(self):
        self.n = 0
        self.objs: list = []
        self.C = np.zeros((0, 3))
        self.R = np.zeros((0, 3, 3))

    @classmethod
    def of(cls, pool: list) -> "_DupIndex":
        ix = getattr(pool, "_dup_index", None)
        if ix is None:
            ix = cls()
            try:
                pool._dup_index = ix  # a plain list cannot hold attributes; then rebuild each time
            except AttributeError:
                pass
        ix.extend(pool)
        return ix

    def extend(self, pool: list) -> None:
        if len(pool) < self.n:
            self.__init__()
        if len(pool) == self.n:
            return
        new = pool[self.n:]
        self.objs += [q["obj"] for q in new]
        self.C = np.concatenate([self.C, np.stack([np.asarray(q["_c"], float) for q in new])])
        self.R = np.concatenate([self.R, np.stack([np.asarray(q["R"], float) for q in new])])
        self.n = len(pool)

    def screen(self, R, cw, obj, dist_m: float, cos_lim: float) -> np.ndarray:
        """Indices of the pool members that may be duplicates (same object, centre within ``dist_m`` and rotation
        cosine above ``cos_lim``, each with a margin); the scalar test decides on them."""
        same = np.fromiter((o == obj for o in self.objs), bool, self.n)
        d = np.sqrt(((self.C - cw) ** 2).sum(axis=1))
        c = (np.einsum("ij,nij->n", np.asarray(R, float), self.R) - 1.0) / 2.0
        return np.flatnonzero(same & (d < dist_m + self._MARGIN_M) & (c > cos_lim - self._MARGIN_COS))


class _Pool(list):
    """A list that can carry its ``_DupIndex``."""


def strip_raw(est: Estimate) -> Estimate:
    """Drop the in-memory arrays kept for tests and diagnostics."""
    est.diag.pop("_raw", None)
    return est
