"""LIBERO environment setup (item libero:setup, 2026-10-22).

First written for the original LIBERO (commit 8f1084e3132a) in a Python 3.11 environment called venv-libero
(robosuite 1.4.0, MuJoCo, torch 2.5.1, gym 0.25.2, bddl 1.0.1 and the small runtime dependencies; the exact versions
of this release are in ``requirements/worker.txt``). No demo dataset.

The setup modules (download, lock file, venv check, smoke test) are not part of this release; the two environments
are built by ``scripts/setup_env.sh``.

``tests/``: the tests, run in the Monty venv::

    PYTHONPATH=$PWD <server python> -m pytest rowant_lab/libero_env/tests -q

M1 oracle agent (item m1:build, 2026-10-22). ORACLE (true poses): diagnostic of the hands, never a LIBERO result.
What this release keeps of it:

* ``m1_paths``: M1 locations and constants.
* ``export_geom`` (venv-libero): the object library from LIBERO's compiled models (F1 runs here).
  ``build_records``: grasp records per object type. ``grasp``: boxes, ray casting, SAT, records, frame selection.
* ``tools``: observation whitelist, the oracle tool, the task spec.
* Tests: ``tests/test_m1_truth_guard.py``, ``tests/test_m1_units.py``.

The M1 policy, follower, evaluator, runners and gate are not part of this release.

Using LIBERO: run venv-libero's interpreter (``<worker python>``). Its
``saccadero_libero_config.pth`` (``scripts/setup_env.sh``) sets LIBERO_CONFIG_PATH, so ``import libero.libero`` never
prompts. Known side effect: every ``import robosuite`` (1.4.0) appends DEBUG lines to ``/tmp/robosuite.log``
(hard-coded upstream).
"""
