"""TRUTH-FREE: the CHECK (Monty's one-observation evidence) with its neighbour search bounded (fastlayer step 4,
amendment E8).

``evidence`` is ``screens/e0c_check_timing/kernel.py::CheckKernel.evidence`` (sha256 and commit in
``provenance.COPIED``), line for line, except that the k-d tree is asked only for the neighbours within
``max_match_distance`` (times 1 + 1e-9). Why that changes nothing: a neighbour farther than ``max_match_distance``
has a custom distance ``|d| + |d . n| / (|curv| + 0.5)`` of at least ``|d|``, so Monty masks it, and a masked
neighbour's evidence is exactly -1 (pose -1, feature 0). The bounded search returns the same nearest neighbours up to
that distance (a prefix of the k nearest), each computed with the same arithmetic; the missing ones get -1. The
maximum over the neighbours is then the same number (tested on real candidates by the step-4 parity runs and on
synthetic ones in ``tests/test_step4_speed.py``). ``score`` is ``CheckKernel.score`` on top of it.
"""

from __future__ import annotations

import numpy as np

from rowant_lab.hybrid_frontier.screens.e0c_check_timing import kernel as K

TRUTH_FREE = True
BOUND_REL = 1e-9


def evidence(ker: K.CheckKernel, obj: str, ps: K.Patches, R, t) -> np.ndarray:
    """``(H, P)`` per-patch evidence (``CheckKernel.evidence`` with the bounded search; module doc)."""
    g = ker.graphs[obj]
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if R.ndim == 2:
        R, t = R[None], t[None]
    H, P = len(R), ps.n
    if H == 0 or P == 0:
        return np.zeros((H, P))
    RT = np.ascontiguousarray(np.transpose(R, (0, 2, 1)))
    search = np.einsum("hij,hpj->hpi", RT, ps.loc[None, :, :] - t[:, None, :]) + K.P_TRAIN
    q = search.reshape(-1, 3)
    n_rot = np.einsum("hij,pj->hpi", RT, ps.normal).reshape(-1, 3)
    k = min(ker.k, len(g.pos))
    _, ids = g.tree.query(q, k=k, p=2, workers=ker.workers, distance_upper_bound=ker.mmd * (1.0 + BOUND_REL))
    if k == 1:
        ids = ids[:, None]
    found = ids < len(g.pos)
    ids = np.where(found, ids, 0)
    diff = g.pos[ids] - q[:, None, :]
    dots = np.einsum("ijk,ik->ij", diff, n_rot)
    eu = np.linalg.norm(diff, axis=2)
    inv_c = np.tile(ps.inv_curv, H)
    custom = eu + np.abs(dots) * inv_c[:, None]
    mask = (((ker.mmd - custom) / ker.mmd) <= 0) | ~found
    sn_err = np.arccos(np.clip(np.einsum("ijk,ik->ij", g.normal[ids], n_rot), -1, 1))
    pose = -(np.sin(sn_err / 2) - 0.5) * ker.w_normal
    if ker.curvature and ps.pfd is not None and ps.pfd.any():
        cd_rot = np.einsum("hij,pj->hpi", RT, ps.dir1).reshape(-1, 3)
        cd_angle = np.arccos(np.clip(np.einsum("ijk,ik->ij", g.cd1[ids], cd_rot), -1, 1))
        cd_err = np.pi / 2 - np.abs(cd_angle - np.pi / 2)
        cd_ev = -(np.sin(cd_err) - 0.5) * g.pfd[ids]
        use = np.tile(ps.pfd, H)
        pose = np.where(use[:, None], pose + cd_ev * ker.w_cd1, pose)
    pose[mask] = -1
    if ker.uses_features:
        nf = g.feats[ids]
        qf = np.tile(ps.feats, (H, 1))[:, None, :]
        d = np.abs(nf - qf)
        if ker.circular.any():
            c = ker.circular
            d[..., c] = np.minimum(np.minimum(np.abs(1 + nf[..., c] - qf[..., c]), np.abs(nf[..., c] - qf[..., c])),
                                   np.abs(nf[..., c] - (qf[..., c] + 1)))
        fe = np.clip(ker.tol - d, 0, np.inf) / ker.tol
        feat = np.multiply(fe, ker.wf).sum(axis=-1) / ker.wf.sum()
        feat[mask] = 0
        total = pose + feat
    else:
        total = pose
    return np.max(total, axis=1).reshape(H, P)


def score(ker: K.CheckKernel, guesses, ps: K.Patches) -> np.ndarray:
    """``CheckKernel.score`` with the bounded ``evidence``."""
    out = np.zeros(len(guesses))
    by_obj: dict[str, list[int]] = {}
    for i, (g, _, _) in enumerate(guesses):
        by_obj.setdefault(g, []).append(i)
    for g, idx in by_obj.items():
        R = np.stack([guesses[i][1] for i in idx])
        t = np.stack([guesses[i][2] for i in idx])
        out[idx] = evidence(ker, g, ps, R, t).sum(axis=1)
    return out
