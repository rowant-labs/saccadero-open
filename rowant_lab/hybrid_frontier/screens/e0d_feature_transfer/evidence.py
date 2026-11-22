"""TRUTH-FREE: Monty's own one-observation evidence, driven through a read-only graph-memory shim.

The evidence of one sensed patch under H pose hypotheses of one object is computed by Monty's
``DefaultHypothesesDisplacer._calculate_evidence_for_new_locations``
(``src/tbp/monty/frameworks/models/evidence_matching/hypotheses_displacer.py:230-338``, imported unchanged) with
Monty's ``DefaultFeatureEvidenceScorer`` and ``DefaultFeatureEvidenceCalculator``. For each hypothesis it finds the
``max_nneighbors`` nearest graph nodes to the search location, keeps those within ``max_match_distance`` under
Monty's curvature-flattened custom distance, scores the surface-normal angle (and the first curvature direction when
the sensed patch's curvature is fully defined), adds the hsv / curvature feature evidence (tolerance-scaled) and
takes the maximum over the neighbours; -1 if no node is in range. The shim supplies only what that method reads:
``get_locations_in_graph``, ``get_graph(...).find_nearest_neighbors`` (a scipy cKDTree, as Monty's
``_location_tree``), ``get_features_at_node``, ``get_feature_array`` and ``get_feature_order``.

Hypotheses are static here: a hypothesis is (object, R, t) with the object placed at ``x = R (m - p_train) + t``.
For a sensed world location x and world pose vectors V, Monty's quantities are the model-frame search location
``R^T (x - t) + p_train`` and the hypothesis rotation ``R^T`` (``rotate_pose_dependent_features`` applies
``poses @ v``). A case's score for a hypothesis is the sum over its patches, i.e. Monty's accumulation with
``past_weight = present_weight = 1`` (the defaults), without the displacement update.

Settings (E6's composed learning module: ``evidence_1lm_nn10_dod0025.yaml`` plus the protocol overrides):
``max_match_distance`` 0.01 m, ``max_nneighbors`` 10, ``feature_weights.patch`` hsv [2, 0.5, 0.5],
pose_vectors [1, 1, 1], principal_curvatures_log [1, 1]; ``tolerances.patch`` hsv [0.1, 0.2, 0.2],
principal_curvatures_log [1, 1].

Feature sets (``FEATURE_SETS``):

* ``all``: normal + curvature direction + hsv + curvature (log), exactly E6's settings;
* ``lnh``: location, normal and hsv only: no curvature feature, no curvature direction (``pose_fully_defined``
  forced false, which is Monty's own path when curvature is undefined) and no curvature in the custom distance
  (Monty's ``get_relevant_curvature`` then returns 10, a near-round search sphere);
* ``ln``: location and normal only (no hsv either): a geometry-only decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

TRUTH_FREE = True
CHANNEL = "patch"
MAX_MATCH_DISTANCE = 0.01
MAX_NNEIGHBORS = 10
P_TRAIN = np.array([0.0, 1.5, 0.0])
FEATURE_SETS = {
    "all": {"weights": {"pose_vectors": np.ones(3), "hsv": np.array([2.0, 0.5, 0.5]),
                        "principal_curvatures_log": np.ones(2)},
            "tolerances": {"hsv": np.array([0.1, 0.2, 0.2]), "principal_curvatures_log": np.ones(2)},
            "curvature": True},
    "lnh": {"weights": {"pose_vectors": np.array([1.0, 0.0, 0.0]), "hsv": np.array([2.0, 0.5, 0.5])},
            "tolerances": {"hsv": np.array([0.1, 0.2, 0.2])},
            "curvature": False},
    "ln": {"weights": {"pose_vectors": np.array([1.0, 0.0, 0.0])},
           "tolerances": {},
           "curvature": False},
}


def patches_from_arrays(arr: dict, keep: np.ndarray) -> list[dict]:
    """List of per-patch dicts (the fields ``channel_features`` reads) for the kept rows of ``camera.stack``."""
    idx = np.flatnonzero(keep)
    return [{"location": arr["location"][i], "normal": arr["normal"][i], "dir1": arr["dir1"][i], "dir2": arr["dir2"][i],
             "pose_fully_defined": bool(arr["pose_fully_defined"][i]), "hsv": arr["hsv"][i],
             "pc_log": arr["pc_log"][i]} for i in idx]
