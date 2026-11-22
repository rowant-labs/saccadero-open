"""TRUTH-FREE: the refinement's frame data (copied from E-0f with its constants as arguments) and a clutter-aware
fit quality.

``extract_data`` is ``screens/e0f_refine/refine.py::extract_data`` (sha256 and commit in ``provenance.COPIED``),
line for line, except that the sample count, the data-normal pixel steps and the table plane's use are arguments:

* ``steps``: the depth-image normal of a data point is the cross product of the world points ``s`` px to either side
  in u and v (all on the target's pixels), for the first ``s`` in ``steps`` that has them. E-0f's (2, 1) suits its
  0.1 mm synthetic depth; real depth needs a wider baseline (the ``hope`` preset uses (10, 5)).
* ``plane``: E-0f's visibility filter drops model points under 2 mm above the table. In clutter an object can sit
  on another one, so ``plane=None`` turns that filter off (every height counts as 1 m above the table).

With E-0f's arguments (``n = spec.N_DATA``, ``steps = spec.DATA_NORMAL_STEP_PX``, the frame's plane) the result is
equal to E-0f's (``tests/test_parity.py``).

``clutter_quality`` is new: E-0f's ``fit_quality`` counts visible model points that project off the dilated blob as
free-space violations, which is wrong when other objects occlude the target. Here a visible model point is a
violation only when it lies more than ``fs_tol_m`` in front of the observed depth at its pixel (the camera would
see it there), on or off the target's pixels; points behind the observed surface (occluded) are fine.
"""

from __future__ import annotations

import numpy as np

from rowant_lab.hybrid_frontier.screens.e0c_check_timing import frontend as e0c_frontend
from rowant_lab.hybrid_frontier.screens.e0c_check_timing import proposal as e0c_proposal
from rowant_lab.hybrid_frontier.screens.e0f_refine import refine as RF
from rowant_lab.hybrid_frontier.screens.e0f_refine import spec as S
from rowant_lab.hybrid_frontier.screens.e0f_refine.models import ModelCloud

TRUTH_FREE = True
NO_PLANE = {"normal": [0.0, 0.0, 0.0], "d": 1.0}  # height = 1 m everywhere: the table-height filter never fires


def extract_data(depth_u16: np.ndarray, pixels: np.ndarray, camera: dict, plane: dict | None, token: str, *,
                 n: int = S.N_DATA, steps=S.DATA_NORMAL_STEP_PX, cam: RF.CameraGeom | None = None) -> RF.FrameData:
    """E-0f's ``extract_data`` with the sample count, normal steps and plane use as arguments (module doc)."""
    cam = RF.CameraGeom.from_camera(camera) if cam is None else cam
    pixels = np.asarray(pixels, bool)
    rows, cols = np.nonzero(pixels)
    n_blob = int(len(rows))
    if n_blob > n:
        pick = np.sort(RF.token_rng(token).choice(n_blob, size=n, replace=False))
        rows, cols = rows[pick], cols[pick]
    P = e0c_frontend.world_points_at(depth_u16, rows, cols, camera)
    ok = np.isfinite(P).all(axis=1)
    rows, cols, P = rows[ok], cols[ok], P[ok]
    N = np.full(P.shape, np.nan)
    todo = np.ones(len(P), bool)
    Hh, Ww = pixels.shape
    for s in steps:
        if not todo.any():
            break
        r, c = rows[todo], cols[todo]
        inside = (r - s >= 0) & (r + s < Hh) & (c - s >= 0) & (c + s < Ww)
        rr = np.clip(r, s, Hh - 1 - s)
        cc = np.clip(c, s, Ww - 1 - s)
        on = inside & pixels[rr, cc - s] & pixels[rr, cc + s] & pixels[rr - s, cc] & pixels[rr + s, cc]
        pu0 = e0c_frontend.world_points_at(depth_u16, rr, cc - s, camera)
        pu1 = e0c_frontend.world_points_at(depth_u16, rr, cc + s, camera)
        pv0 = e0c_frontend.world_points_at(depth_u16, rr - s, cc, camera)
        pv1 = e0c_frontend.world_points_at(depth_u16, rr + s, cc, camera)
        nn = np.cross(pu1 - pu0, pv1 - pv0)
        ln = np.linalg.norm(nn, axis=1)
        good = on & np.isfinite(nn).all(axis=1) & (ln > 0)
        nn = nn / np.where(ln > 0, ln, 1.0)[:, None]
        flip = np.einsum("ij,ij->i", nn, cam.c_w - P[todo]) < 0
        nn[flip] *= -1
        idx = np.flatnonzero(todo)
        N[idx[good]] = nn[good]
        todo[idx[good]] = False
    n_w, h0 = RF.plane_world(plane if plane is not None else NO_PLANE, cam)
    return RF.FrameData(pts=P, nrm=N, has_n=np.isfinite(N).all(axis=1), plane_n=n_w, plane_h0=h0,
                        blob_dil=e0c_proposal.dilate_square(pixels, S.FS_DILATE_PX), cam=cam, n_blob=n_blob,
                        diag={"n_sampled": int(len(P)), "n_with_normal": int(np.isfinite(N).all(axis=1).sum())})


def observed_depth_m(depth_u16: np.ndarray, camera: dict) -> np.ndarray:
    """(H, W) observed camera-z in metres, NaN where there is no reading."""
    d = np.asarray(depth_u16)
    z = d.astype(np.float64) * float(camera["depth"]["unit_m"])
    z[(d == 0) | (d == np.iinfo(np.uint16).max)] = np.nan
    return z


def clutter_quality(model: ModelCloud, fr: RF.FrameData, R, t, z_obs: np.ndarray, *, inlier_gate_m: float,
                    inlier_resid_m: float, fs_tol_m: float) -> dict:
    """Truth-free fit of a pose in clutter (module doc). ``z_obs``: ``observed_depth_m`` of the frame."""
    Rm, tm = RF.to_internal(R, t)
    Xw, Nw = RF.placed(model, Rm, tm)
    vis = RF.visible(Xw, Nw, fr)
    cfg = RF.IcpConfig()
    Y, j, dd, ok = RF.correspond(model, fr, Rm, tm, inlier_gate_m, cfg, vis)
    r = np.abs(np.einsum("ij,ij->i", Y - model.pos[j], model.nrm[j]))
    inl = ok & (r <= inlier_resid_m)
    u, v, z = fr.cam.project(Xw[vis])
    ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
    ins = (ui >= 0) & (ui < fr.cam.W) & (vi >= 0) & (vi < fr.cam.H)
    zo = np.full(len(ui), np.nan)
    zo[ins] = z_obs[vi[ins], ui[ins]]
    known = ins & np.isfinite(zo)
    viol = known & (z < zo - fs_tol_m)
    fs = float(viol.sum() / max(known.sum(), 1))
    inlier = float(inl.mean()) if fr.n else 0.0
    return {"inlier_share": inlier, "resid_med_mm": float(np.median(r[inl]) * 1e3) if inl.any() else None,
            "gated_share": float(ok.mean()) if fr.n else 0.0, "free_space_share": fs,
            "n_visible": int(vis.sum()), "n_known": int(known.sum()), "score": inlier - fs}
