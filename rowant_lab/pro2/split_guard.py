"""The benchmark's test tasks, and which of their episodes may be planned, rendered or run (standard library only).

LIBERO-Pro2.0's board is scored on two base tasks per suite (``TEST_TASKS``).
The rule since 2026-11-15:

* a PERTURBED episode of a test task (any case other than ``BASE``) is never planned, rendered or run before the one
  scored run, and Pro2.0's own task files of a test task (the 339 never unpacked) are never used, whatever the case;
* the PLAIN version of a test task may be used for development, frames for the depth net included: the ``BASE`` case
  from the simulator's own base task file, reached through LIBERO's API (``get_task_bddl_file_path`` with the config
  of ``long_lib.libero_config_text``), which is a file directly under ``PLAIN_ROOT/<suite>/``.

``refusal`` decides from an episode's or a frame-plan item's own fields (``suite``, ``task``, ``case``, ``bddl``); no
file is opened. Training tasks are never refused here. The frame planners (``colour_frames plan``, ``colour_long
plan``), the frame worker and the camera measurement call ``check`` before they use an item.
"""

from __future__ import annotations

import os
from rowant_lab.pro2.machine import BENCHMARK

TEST_TASKS = {"libero_10": (5, 8), "libero_goal": (3, 6), "libero_object": (1, 8), "libero_spatial": (0, 8)}
PLAIN_ROOT = str(BENCHMARK / "simulator-source" / "libero" / "libero" / "bddl_files")  # the simulator's own base task files (LIBERO's, byte for byte)


class Refused(ValueError):
    """An episode of a benchmark test task that may not be planned, rendered or run."""


def is_test_task(suite: str, task: int) -> bool:
    return int(task) in TEST_TASKS.get(str(suite), ())


def is_plain_file(suite: str, bddl) -> bool:
    """``bddl`` names a file directly in the simulator's own folder of base task files for ``suite`` (by its path; the
    file is not opened)."""
    if not bddl:
        return False
    p = os.path.normpath(str(bddl))
    return p.endswith(".bddl") and os.path.dirname(p) == os.path.join(PLAIN_ROOT, str(suite))


def refusal(ep: dict) -> str | None:
    """Why this episode (or frame-plan item) may not be used, or None. Only test tasks are ever refused."""
    suite, task = str(ep.get("suite")), ep.get("task")
    if task is None or not is_test_task(suite, int(task)):
        return None
    name = f"{suite} task {int(task)} is a benchmark test task"
    if str(ep.get("case")) != "BASE":
        return f"{name}: its perturbed case {ep.get('case')} is never planned, rendered or run"
    if not is_plain_file(suite, ep.get("bddl")):
        return f"{name}: only its plain version may be used (the simulator's own base file under {PLAIN_ROOT}/{suite})"
    return None


def check(eps, what: str = "plan") -> None:
    """Raise ``Refused`` when any of ``eps`` may not be used (the message names the first few reasons)."""
    bad = sorted({r for r in (refusal(ep) for ep in eps) if r})
    if bad:
        more = f" (+{len(bad) - 3} more)" if len(bad) > 3 else ""
        raise Refused(f"{what} refused: " + "; ".join(bad[:3]) + more)
