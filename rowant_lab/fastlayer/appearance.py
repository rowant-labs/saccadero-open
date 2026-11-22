"""TRUTH-FREE: the pluggable appearance scorer and the declared decision rules that combine it with geometry
(fastlayer step 3).

**Interface.** An appearance scorer is any object with a ``name`` and a method

    score(candidate_pose, frame, mask, object_assets) -> float     (higher = the image looks more like this pose)

* ``candidate_pose``: a ``Candidate`` (object id, world-frame ``R``, ``t`` in metres, the candidate's label, its
  base, its flip, and its geometric score; ``Candidate.camera_pose_mm(frame)`` gives BOP's model-to-camera pose);
* ``frame``: the ``frame.Frame`` (RGB, raw depth, ``K``, depth scale);
* ``mask``: (H, W) bool, the target's pixels (its given pixels with valid depth);
* ``object_assets``: an ``ObjectAssets`` (the object's graph nodes and features from the memory, its mesh and
  texture paths from onboarding, its symmetry record).

A scorer may also define ``score_many(candidates, frame, mask, object_assets) -> list[float]`` (one call per target,
e.g. to share one image crop's embedding); the default loops over ``score``. It may return ``None`` for "no
opinion" (treated as the lowest value in the gate and as 0 in a weighted sum). Scorers are made by name from
``SCORERS`` or by import path ``"package.module:Class"`` (``make``), so worker processes can build their own (the
runner's ``--scorer`` takes either).

**Decision rules** (``DecideRule``; ``decide`` returns the chosen candidate's index; ties: the candidate order, which
is CHECK's order of the bases, then the family order):

* ``geometry``: the most CHECK evidence per patch (step 2's B, extended to the flips);
* ``gate``: the candidates within ``margin`` of the best CHECK evidence per patch form the gate (geometrically
  equivalent: typically the members of one flip family); among them the highest appearance score wins; ties: CHECK;
* ``weighted``: CHECK evidence per patch + ``weight`` x appearance score.

With the ``geometry`` scorer (a constant score) all three rules choose the same candidate as ``geometry``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TRUTH_FREE = True
RULES = ("geometry", "gate", "weighted")


@dataclass
class Candidate:
    obj: str
    R: np.ndarray  # object-local -> world
    t: np.ndarray  # metres, world
    label: str = "I"  # the flip ("I": the refined base itself)
    base: int = 0  # index of the base (CHECK's order of the refined guesses)
    geo: float | None = None  # CHECK evidence per patch, once scored
    info: dict = field(default_factory=dict)

    def camera_pose_mm(self, frame) -> tuple[np.ndarray, np.ndarray]:
        R, t = frame.to_camera(self.R, self.t)
        return R, t * 1000.0


@dataclass
class ObjectAssets:
    obj: str
    nodes_m: np.ndarray  # (N, 3) graph nodes, object-local metres (BOP model frame)
    features: dict  # name -> (N, k) Monty graph features (hsv, pose_vectors, ...)
    mesh_path: str | None = None
    texture_path: str | None = None
    symmetry: dict | None = None
    extra: dict = field(default_factory=dict)


def assets_from_memory(memory, symmetry=None, *, p_train=None) -> dict:
    """{obj: ObjectAssets} from a ``memory.Memory`` (and a ``symmetry.SymmetryTable``)."""
    from . import onboard as OB  # noqa: PLC0415

    p = OB.P_TRAIN if p_train is None else np.asarray(p_train, float)
    objs = (memory.meta or {}).get("objects", {})
    out = {}
    for g in memory.ids:
        x = memory.x[g]
        feats = {k: x[:, a:b] for k, (a, b) in memory.feature_mapping.items()}
        o = objs.get(g, {})
        out[g] = ObjectAssets(obj=g, nodes_m=np.asarray(memory.pos[g]) - p, features=feats, mesh_path=o.get("ply"),
                              texture_path=o.get("texture"),
                              symmetry=symmetry.get(g) if symmetry is not None and g in symmetry else None)
    return out


class GeometryOnly:
    """The default scorer: no appearance information (a constant), so the decision is CHECK's (geometry)."""

    name = "geometry"

    def score(self, candidate_pose, frame, mask, object_assets) -> float:  # noqa: ARG002
        return 0.0

    def score_many(self, candidates, frame, mask, object_assets) -> list[float]:  # noqa: ARG002
        return [0.0] * len(candidates)


SCORERS = {"geometry": GeometryOnly}


def register(name: str, cls) -> None:
    """Add a scorer class (built with ``cls(**params)`` in every process that runs the layer)."""
    SCORERS[name] = cls


def make(name: str = "geometry", **params):
    """A scorer by registered name, or by import path ``"package.module:Class"`` (worker processes start fresh, so
    a scorer registered only in the main process is unknown there; an import path works in every process)."""
    if ":" in name:
        import importlib  # noqa: PLC0415

        mod, _, attr = name.partition(":")
        return getattr(importlib.import_module(mod), attr)(**params)
    try:
        cls = SCORERS[name]
    except KeyError:
        raise KeyError(f"unknown appearance scorer {name!r}; known: {sorted(SCORERS)}") from None
    return cls(**params)


def score_all(scorer, candidates, frame, mask, assets) -> list:
    if hasattr(scorer, "score_many"):
        out = list(scorer.score_many(candidates, frame, mask, assets))
    else:
        out = [scorer.score(c, frame, mask, assets) for c in candidates]
    if len(out) != len(candidates):
        raise ValueError(f"scorer {getattr(scorer, 'name', scorer)!r} returned {len(out)} scores for "
                         f"{len(candidates)} candidates")
    return [None if v is None else float(v) for v in out]


@dataclass(frozen=True)
class DecideRule:
    name: str = "geometry"
    margin: float = 0.0  # gate: CHECK evidence per patch below the best
    weight: float = 0.0  # weighted: appearance weight

    def __post_init__(self):
        if self.name not in RULES:
            raise ValueError(f"unknown decision rule {self.name!r}; known: {RULES}")
        if self.margin < 0:
            raise ValueError("the gate margin must be >= 0")

    def record(self) -> dict:
        return {"name": self.name, "margin": self.margin, "weight": self.weight}


def decide(geo, app, rule: DecideRule, n_patches: int = 1) -> tuple[int, dict]:
    """Index of the chosen candidate and the decision's record (module doc). ``geo``: CHECK's summed evidence of each
    candidate (in candidate order) over ``n_patches`` patches (so ``geo / n_patches`` is the evidence per patch, the
    unit of ``margin`` and of the weighted sum); ``app``: appearance scores (None = no opinion)."""
    geo = [float(g) for g in geo]
    n = len(geo)
    per = float(max(int(n_patches), 1))
    if n == 0:
        raise ValueError("no candidates")
    if len(app) != n:
        raise ValueError("geometry and appearance scores differ in length")
    a = [(-np.inf if v is None else float(v)) for v in app]
    if rule.name == "geometry":
        best = max(range(n), key=lambda i: (geo[i], -i))
        return best, {"rule": "geometry", "gate": None}
    if rule.name == "gate":
        top = max(geo)
        gate = [i for i in range(n) if geo[i] >= top - rule.margin * per]
        best = max(gate, key=lambda i: (a[i], geo[i], -i))
        return best, {"rule": "gate", "gate": gate}
    w = rule.weight
    key = [geo[i] / per + w * (0.0 if app[i] is None else float(app[i])) for i in range(n)]
    best = max(range(n), key=lambda i: (key[i], geo[i], -i))
    return best, {"rule": "weighted", "gate": None}
