"""TRUTH-FREE: the vectorised one-frame CHECK (direction note section 3.2 P2): Monty's one-observation evidence for
every (patch, guess) pair of one camera frame in a few numpy / scipy calls.

What it computes. A guess is (object, R, t): the object's learned graph placed at ``x = R (m - p_train) + t``
(world frame, metres). For each sensed patch p (world location x_p, world surface normal n_p, hsv) and each guess h
of object g, Monty's ``DefaultHypothesesDisplacer._calculate_evidence_for_new_locations``
(``src/tbp/monty/frameworks/models/evidence_matching/hypotheses_displacer.py:230-338``) does, per patch, one call:

1. search location ``s = R^T (x_p - t) + p_train`` and hypothesis-rotated normal ``R^T n_p`` (Monty's
   ``rotate_pose_dependent_features`` with ``channel_possible_poses = R^T``);
2. the ``max_nneighbors`` (10) nearest graph nodes to ``s`` (Monty's ``_location_tree``, a scipy cKDTree);
3. Monty's custom distance ``|d| + |d . n| / (|curv| + 0.5)`` (``get_custom_distances``; curv = 10 when no curvature
   feature is used, ``get_relevant_curvature``), nodes at or beyond ``max_match_distance`` (0.01 m) masked;
4. pose evidence ``-(sin(angle(n_node, R^T n_p) / 2) - 0.5) * w_normal`` (plus the curvature-direction term when the
   patch's ``pose_fully_defined`` is true and curvature is in the feature set), -1 for masked nodes;
5. feature evidence ``average(clip(tol - |f_node - f_patch|, 0) / tol, weights)`` (hue circular), 0 for masked nodes;
6. the maximum over the neighbours of (pose + feature) evidence.

Monty computes step 5 for EVERY node of the graph per patch (``DefaultFeatureEvidenceCalculator``) and then indexes
the neighbours; here it is computed only at the gathered neighbours, which gives the same numbers. All guesses of one
object are queried in one ``cKDTree.query`` call (``workers`` threads, default 1: the CPU reference path), so a frame
with k guesses over 3 objects costs 3 tree calls of (k_g x P) points each. The case score of a guess is the sum of
its per-patch evidence (Monty's accumulation with past and present weight 1, without the displacement update), as in
screen E-0d (``screens/e0d_feature_transfer/evidence.py``), whose reference scorer this kernel must reproduce
(``equivalence.py``).

Feature sets and settings are E-0d's (``evidence.FEATURE_SETS``, imported): ``lnh`` (location, normal, hsv: the
configuration E-0d recommends for camera data), ``ln`` and ``all`` (curvature too). Inputs are sensor-derived patch
features and the learned graphs only; nothing here reads a truth file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from rowant_lab.hybrid_frontier.screens.e0d_feature_transfer import evidence as e0d_evidence

TRUTH_FREE = True
P_TRAIN = e0d_evidence.P_TRAIN
FEATURE_SETS = e0d_evidence.FEATURE_SETS
MAX_MATCH_DISTANCE = e0d_evidence.MAX_MATCH_DISTANCE
MAX_NNEIGHBORS = e0d_evidence.MAX_NNEIGHBORS
NO_CURVATURE = 10.0  # Monty's get_relevant_curvature when no curvature feature is present


@dataclass
class ObjectGraph:
    """One object's learned graph, as the kernel reads it (built once at load)."""

    pos: np.ndarray  # (N, 3) model frame (object-local + p_train)
    normal: np.ndarray  # (N, 3) stored surface normal (pose_vectors[0:3])
    cd1: np.ndarray  # (N, 3) stored first curvature direction (pose_vectors[3:6])
    pfd: np.ndarray  # (N,) stored pose_fully_defined (bool)
    feats: np.ndarray  # (N, F) the feature columns of the feature set, in Monty's order
    tree: cKDTree


@dataclass
class Patches:
    """The sensed patches of one frame (world frame), as the kernel reads them."""

    loc: np.ndarray  # (P, 3)
    normal: np.ndarray  # (P, 3)
    feats: np.ndarray  # (P, F) query feature values in the feature set's column order
    dir1: np.ndarray | None = None  # (P, 3), only for curvature feature sets
    pfd: np.ndarray | None = None  # (P,) bool
    inv_curv: np.ndarray | None = None  # (P,) 1 / (|relevant curvature| + 0.5)

    @property
    def n(self) -> int:
        return int(len(self.loc))


class CheckKernel:
    """Monty's evidence for many (patch, guess) pairs at once, for one feature set over a dict of graphs."""

    def __init__(self, pos: dict, x: dict, feature_mapping: dict, feature_set: str = "lnh", *, ids=None,
                 max_match_distance: float = MAX_MATCH_DISTANCE, max_nneighbors: int = MAX_NNEIGHBORS,
                 workers: int = 1, leafsize: int = 16, trees: dict | None = None):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"unknown feature set {feature_set!r}")
        fs = FEATURE_SETS[feature_set]
        self.feature_set = feature_set
        self.curvature = bool(fs["curvature"])
        self.tol_keys = list(fs["tolerances"])  # Monty's column order = the tolerance dict's order
        self.w_normal = float(fs["weights"]["pose_vectors"][0])
        self.w_cd1 = float(fs["weights"]["pose_vectors"][1])
        self.uses_features = bool(self.tol_keys) and any(k not in ("pose_vectors", "pose_fully_defined")
                                                          for k in fs["weights"])
        self.tol = (np.concatenate([np.atleast_1d(np.asarray(fs["tolerances"][k], float)) for k in self.tol_keys])
                    if self.tol_keys else np.zeros(0))
        self.wf = (np.concatenate([np.atleast_1d(np.asarray(fs["weights"][k], float)) for k in self.tol_keys])
                   if self.tol_keys else np.zeros(0))
        self.circular = np.zeros(len(self.tol), dtype=bool)
        start = 0
        for k in self.tol_keys:
            n = len(np.atleast_1d(fs["tolerances"][k]))
            if k == "hsv":
                self.circular[start] = True
            start += n
        self.mmd = float(max_match_distance)
        self.k = int(max_nneighbors)
        self.workers = int(workers)
        pv = feature_mapping["pose_vectors"][0]
        pf = feature_mapping["pose_fully_defined"][0]
        self.graphs: dict[str, ObjectGraph] = {}
        for g in (ids if ids is not None else sorted(pos)):
            p = np.ascontiguousarray(pos[g], dtype=np.float64)
            xx = np.asarray(x[g], dtype=np.float64)
            cols = [xx[:, feature_mapping[k][0]:feature_mapping[k][1]] for k in self.tol_keys]
            tree = trees[g] if trees is not None and g in trees else cKDTree(p, leafsize=leafsize)
            self.graphs[g] = ObjectGraph(
                pos=p, normal=np.ascontiguousarray(xx[:, pv:pv + 3]), cd1=np.ascontiguousarray(xx[:, pv + 3:pv + 6]),
                pfd=xx[:, pf] > 0.5,
                feats=np.ascontiguousarray(np.concatenate(cols, axis=1)) if cols else np.zeros((len(p), 0)),
                tree=tree)

    # ------------------------------------------------------------------------------------- inputs
    def patches(self, loc, normal, hsv=None, *, dir1=None, pose_fully_defined=None, pc_log=None) -> Patches:
        """Patch inputs for this feature set (world frame). ``hsv`` is needed for ``lnh`` / ``all``; ``dir1``,
        ``pose_fully_defined`` and ``pc_log`` for ``all``."""
        loc = np.ascontiguousarray(loc, dtype=np.float64)
        normal = np.ascontiguousarray(normal, dtype=np.float64)
        cols = []
        for k in self.tol_keys:
            if k == "hsv":
                cols.append(np.asarray(hsv, dtype=np.float64))
            elif k == "principal_curvatures_log":
                cols.append(np.asarray(pc_log, dtype=np.float64))
            else:
                raise ValueError(f"feature {k!r} is not supported by the kernel")
        feats = np.ascontiguousarray(np.concatenate(cols, axis=1)) if cols else np.zeros((len(loc), 0))
        if self.curvature:
            pcl = np.asarray(pc_log, dtype=np.float64)
            curv = np.max(np.abs(pcl), axis=1)
            return Patches(loc, normal, feats, dir1=np.ascontiguousarray(dir1, dtype=np.float64),
                           pfd=np.asarray(pose_fully_defined, dtype=bool), inv_curv=1 / (np.abs(curv) + 0.5))
        return Patches(loc, normal, feats, inv_curv=np.full(len(loc), 1 / (abs(NO_CURVATURE) + 0.5)))

    # ------------------------------------------------------------------------------------- evidence
    def evidence(self, obj: str, ps: Patches, R, t) -> np.ndarray:
        """``(H, P)`` per-patch evidence of the patches under H guesses (R[h], t[h]) of ``obj``."""
        g = self.graphs[obj]
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        if R.ndim == 2:
            R, t = R[None], t[None]
        H, P = len(R), ps.n
        if H == 0 or P == 0:
            return np.zeros((H, P))
        RT = np.ascontiguousarray(np.transpose(R, (0, 2, 1)))
        # 1. search locations and rotated normals (E-0d's einsum order: R^T (x - t) + p_train)
        search = np.einsum("hij,hpj->hpi", RT, ps.loc[None, :, :] - t[:, None, :]) + P_TRAIN
        q = search.reshape(-1, 3)
        n_rot = np.einsum("hij,pj->hpi", RT, ps.normal).reshape(-1, 3)
        # 2. neighbours
        k = min(self.k, len(g.pos))
        _, ids = g.tree.query(q, k=k, p=2, workers=self.workers)
        if k == 1:
            ids = ids[:, None]
        # 3. custom distance and mask
        diff = g.pos[ids] - q[:, None, :]
        dots = np.einsum("ijk,ik->ij", diff, n_rot)
        eu = np.linalg.norm(diff, axis=2)
        inv_c = np.tile(ps.inv_curv, H)
        custom = eu + np.abs(dots) * inv_c[:, None]
        mask = ((self.mmd - custom) / self.mmd) <= 0
        # 4. pose evidence
        sn_err = np.arccos(np.clip(np.einsum("ijk,ik->ij", g.normal[ids], n_rot), -1, 1))
        pose = -(np.sin(sn_err / 2) - 0.5) * self.w_normal
        if self.curvature and ps.pfd is not None and ps.pfd.any():
            cd_rot = np.einsum("hij,pj->hpi", RT, ps.dir1).reshape(-1, 3)
            cd_angle = np.arccos(np.clip(np.einsum("ijk,ik->ij", g.cd1[ids], cd_rot), -1, 1))
            cd_err = np.pi / 2 - np.abs(cd_angle - np.pi / 2)
            cd_ev = -(np.sin(cd_err) - 0.5) * g.pfd[ids]
            use = np.tile(ps.pfd, H)
            pose = np.where(use[:, None], pose + cd_ev * self.w_cd1, pose)
        pose[mask] = -1
        # 5. feature evidence at the neighbours only
        if self.uses_features:
            nf = g.feats[ids]  # (HP, k, F)
            qf = np.tile(ps.feats, (H, 1))[:, None, :]  # (HP, 1, F)
            d = np.abs(nf - qf)
            if self.circular.any():
                c = self.circular
                d[..., c] = np.minimum(np.minimum(np.abs(1 + nf[..., c] - qf[..., c]), np.abs(nf[..., c] - qf[..., c])),
                                       np.abs(nf[..., c] - (qf[..., c] + 1)))
            fe = np.clip(self.tol - d, 0, np.inf) / self.tol
            feat = np.multiply(fe, self.wf).sum(axis=-1) / self.wf.sum()
            feat[mask] = 0
            total = pose + feat
        else:
            total = pose
        # 6. maximum over the neighbours
        return np.max(total, axis=1).reshape(H, P)

    def score(self, guesses, ps: Patches) -> np.ndarray:
        """Summed evidence of each guess ``(obj, R, t)`` (one tree call per object)."""
        out = np.zeros(len(guesses))
        by_obj: dict[str, list[int]] = {}
        for i, (g, _, _) in enumerate(guesses):
            by_obj.setdefault(g, []).append(i)
        for g, idx in by_obj.items():
            R = np.stack([guesses[i][1] for i in idx])
            t = np.stack([guesses[i][2] for i in idx])
            out[idx] = self.evidence(g, ps, R, t).sum(axis=1)
        return out


def decide(scores, guesses, prior_rank: dict | None = None) -> int:
    """Index of the CHECK's pick: the highest summed evidence; ties go to the better prior (HPG) rank, then the
    earlier guess."""
    prior_rank = prior_rank or {}
    best = max(range(len(guesses)), key=lambda i: (float(scores[i]), -prior_rank.get(guesses[i][0], 0), -i))
    return int(best)
