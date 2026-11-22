"""Fixed locations for LIBERO M2 that the POLICY and the perception server may know (the M2 plan).

M2 (perceived poses, grounded instruction): perception in the loop. Standard library only (``pathlib``). Since the
m2f:guard fix round this module is scanned under the policy rules of ``tests/test_m2_truth_guard.py`` (verifier
blocker B2): it names only the M2 library (the export, the onboarded artifacts and the policy-v2 version the policy
and server load), M1's library root (its grasp records and robot.json are read through ``m2_library``, never its
scenes), the suites, the workcell types, the cameras and the label. Every truth-side or runner-side location (run
folders, init states, the GATE registry and read log, the peek ledger, the frozen config, the numba cache, the
scratch folder, the interpreters) lives in ``m2_rpaths`` (runner side), which re-exports these names.
"""

from __future__ import annotations

from pathlib import Path
from rowant_lab.settings import DATA

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent

LIBRARY = DATA / "runs" / "libero-m2" / "library"  # the M2 library (export + onboarding)
# library v3 (m2r3:fix-r1, amendments A58 (D-STOVE) and A65): export-v3 = export v2 plus every rendered geom v2 left
# out (m2_export3; the flat stove's base box), policy-v3 = policy-v2's rules over it (m2_libv3), onboard-v3 =
# m2_onboard over it. The v2 folders (export, onboard, policy-v2) stay as they are.
EXPORT_DIR = LIBRARY / "export-v3"  # m2_export (+ m2_export3): per-type meshes and kinematics, workcells, the Panda
ONBOARD_DIR = LIBRARY / "onboard-v3"  # m2_onboard: per-workcell memory, colour memory, symmetry, templates, grids
POLICY_LIB = LIBRARY / "policy-v3"  # m2_libv3 under m2_libv2's rules (amendment A9): what the policy and server load
POLICY_LIB_VERSION = "policy-v3"

# M1's frozen library root: m2_library reads only its records/ and robot.json (sha-checked), never scenes/
M1_LIBRARY = DATA / "runs" / "libero-m1" / "library"

SUITES = ("libero_object", "libero_spatial")
N_TASKS = 10

WORKCELL_TYPES = {
    "libero_object": ("alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce", "ketchup", "tomato_sauce",
                      "butter", "milk", "chocolate_pudding", "orange_juice", "basket"),
    "libero_spatial": ("akita_black_bowl", "plate", "glazed_rim_porcelain_ramekin", "cookies", "wooden_cabinet",
                       "flat_stove"),
}

CAMERAS = {"agentview": 1024, "robot0_eye_in_hand": 512}

LABEL = "M2 (perceived poses, grounded instruction): perception in the loop"
