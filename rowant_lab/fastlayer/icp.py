"""TRUTH-FREE: the refinement ICP with the step-4 speed knobs (fastlayer step 4, item ``bop4:speed``).

``icp`` is ``screens/e0f_refine/refine.py::icp`` (sha256 and commit in ``provenance.COPIED``), line for line, with
two knobs added:

* ``vis_every`` (X3): the visibility filter (``refine.visible``, a z-buffer of every placed model point) is
  recomputed every ``vis_every``-th iteration instead of every iteration; in between, the last one is reused;
* ``stage_stop`` (X3): E-0f's stop rule (a step below ``stop_step_m`` and ``stop_step_deg``) ends every stage, not
  only the last one; ending an earlier stage moves on to the next stage, ending the last one is convergence.

and one estimate-preserving option (amendment E7):

* ``bounded``: the correspondence search (``correspond``, a copy of ``refine.correspond``) asks the k-d tree only for
  neighbours within the stage's distance gate (``distance_upper_bound``, the gate plus 1e-9 m). A pair is accepted
  only when the nearest VISIBLE of the k nearest model points lies within the gate; when it does, every closer
  neighbour lies within the gate too, so the bounded search returns the same first visible neighbour, and when it
  does not, both searches reject the pair. The accepted pairs, and so every step, are the same bit for bit (tested).

and one port knob (fastlayer step 6, the pixel-constant port P):

* ``vis_bin_px``: the visibility z-buffer's bin size in pixels (``refine.visible``'s ``bin_px``; default
  ``spec.VIS_BIN_PX`` = 3, the value every step before 6 used). The layer passes ``layer.pixel_constants``'s value,
  which is 3 at HOPE's focal length.

With the defaults (``vis_every = 1``, ``stage_stop = False``, ``bounded = False``, ``vis_bin_px = 3``) the screens' own
``refine.icp`` is called, so the result is the step-3 result bit for bit (``tests/test_step4_speed.py``). The other step-4 refinement
knobs need no copy: fewer data points (``Preset.refine_n``), fewer neighbours (``IcpConfig.k``) and fewer iterations
(``Preset.refine_stages``) are arguments of the screens' code already.
"""

from __future__ import annotations

import math

import numpy as np

from rowant_lab.hybrid_frontier.screens.e0f_refine import refine as RF
from rowant_lab.hybrid_frontier.screens.e0f_refine import spec as S
from rowant_lab.hybrid_frontier.screens.e0f_refine.models import ModelCloud

TRUTH_FREE = True


BOUND_EPS_M = 1e-9


def correspond(model: ModelCloud, fr: RF.FrameData, Rm, tm, gate: float, cfg: RF.IcpConfig, vis: np.ndarray | None):
    """``refine.correspond`` with the k-d tree search bounded by the gate (module doc, ``bounded``). Rejected pairs
    may carry another (clipped) model index than the unbounded search; only accepted pairs are used."""
    Y = fr.pts @ Rm.T + tm
    k = min(cfg.k, model.n) if vis is not None else 1
    d, idx = model.tree.query(Y, k=k, workers=1, distance_upper_bound=float(gate) + BOUND_EPS_M)
    if k == 1:
        d, idx = d[:, None], idx[:, None]
    found = idx < model.n
    idx = np.where(found, idx, 0)
    if vis is not None:
        v = vis[idx] & found
        first = np.argmax(v, axis=1)
        rows = np.arange(len(Y))
        j, dd, ok = idx[rows, first], d[rows, first], v[rows, first]
    else:
        j, dd, ok = idx[:, 0], d[:, 0], found[:, 0]
    ok = ok & (dd <= gate)
    if cfg.normal_compat_deg < 180.0:
        Nd = fr.nrm @ Rm.T
        cosang = np.einsum("ij,ij->i", np.nan_to_num(Nd), model.nrm[j])
        ok &= (~fr.has_n) | (cosang >= math.cos(math.radians(cfg.normal_compat_deg)))
    return Y, j, dd, ok


def icp(model: ModelCloud, fr: RF.FrameData, R0, t0, cfg: RF.IcpConfig = RF.IcpConfig(), *, vis_every: int = 1,
        stage_stop: bool = False, bounded: bool = False, vis_bin_px: int = S.VIS_BIN_PX) -> dict:
    """``refine.icp`` with ``vis_every``, ``stage_stop``, ``bounded`` and ``vis_bin_px`` (module doc)."""
    vis_bin_px = int(vis_bin_px)
    if vis_bin_px < 1:
        raise ValueError("vis_bin_px must be >= 1")
    if int(vis_every) == 1 and not stage_stop and not bounded and vis_bin_px == S.VIS_BIN_PX:
        return RF.icp(model, fr, R0, t0, cfg)
    corr = correspond if bounded else RF.correspond
    Rm, tm = RF.to_internal(R0, t0)
    if fr.n < S.MIN_DATA:
        R, t = RF.from_internal(Rm, tm)
        return {"R": R, "t": t, "status": "too_few_points", "iters": 0, "converged": False, "n_corr": 0}
    it = 0
    status, converged, n_corr = "max_iters", False, 0
    total_rot, total_t = 0.0, 0.0
    last_step = (None, None)
    vis = None
    n_vis = 0
    for si, (n_it, gate, delta) in enumerate(cfg.stages):
        last = si == len(cfg.stages) - 1
        for _ in range(int(n_it)):
            if cfg.visibility and (vis is None or it % int(vis_every) == 0):
                Xw, Nw = RF.placed(model, Rm, tm)
                vis = RF.visible(Xw, Nw, fr, bin_px=vis_bin_px)
                n_vis += 1
            Y, j, dd, ok = corr(model, fr, Rm, tm, gate, cfg, vis if cfg.visibility else None)
            n_corr = int(ok.sum())
            if n_corr < cfg.min_corr:
                status = "too_few_correspondences"
                break
            Yk, Mk, Nk = Y[ok], model.pos[j[ok]], model.nrm[j[ok]]
            if cfg.point_to_plane:
                res = np.abs(np.einsum("ij,ij->i", Yk - Mk, Nk))
            else:
                res = dd[ok]
            wts = np.minimum(1.0, delta / np.maximum(res, 1e-12)) if cfg.huber else np.ones(len(res))
            if cfg.point_to_plane:
                c, w, dt = RF._step_p2pl(Yk, Mk, Nk, wts, cfg)  # noqa: SLF001
            else:
                c, w, dt = RF._step_p2p(Yk, Mk, wts)  # noqa: SLF001
            w, dt = RF._clamp(w, dt, cfg)  # noqa: SLF001
            dR = RF.rodrigues(w)
            Rm = dR @ Rm
            tm = dR @ (tm - c) + c + dt
            it += 1
            last_step = (float(np.linalg.norm(dt)) * 1e3, math.degrees(np.linalg.norm(w)))
            total_rot += math.degrees(np.linalg.norm(w))
            total_t += float(np.linalg.norm(dt))
            small = np.linalg.norm(dt) < cfg.stop_step_m and math.degrees(np.linalg.norm(w)) < cfg.stop_step_deg
            if small and last:
                status, converged = "converged", True
                break
            if small and stage_stop:
                break
        if status in ("converged", "too_few_correspondences"):
            break
    R, t = RF.from_internal(Rm, tm)
    return {"R": R, "t": t, "status": status, "iters": it, "converged": converged, "n_corr": n_corr,
            "path_rot_deg": total_rot, "path_t_mm": total_t * 1e3, "last_step_mm": last_step[0],
            "last_step_deg": last_step[1], "n_vis": n_vis}
