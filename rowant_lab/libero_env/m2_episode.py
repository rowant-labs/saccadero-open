"""Two names the LIBERO-Pro2.0 runner uses from the earlier robot's episode module: the watchdog's exception and
the list of failures that count as runner events (never as an episode's outcome). Their code is unchanged. The
earlier robot's own episode loop is not part of this release (docs/CHANGES-FROM-AS-RUN.md).

M2 (perceived poses, grounded instruction): perception in the loop. That is the earlier robot this module was
written for; its modules keep this label.
"""

from __future__ import annotations

# These imports are kept from the earlier robot's episode module. Nothing below uses them, but importing this module
# loaded them in the worker, and the worker's hello sends the perception server the checksums of the shared modules it
# has loaded (``m2_bridge.SHARED_CODE``), so they must be loaded before it. ``m2_paths``, ``grasp`` and ``m2_kin`` came
# in through the earlier robot's look tool, which is not part of this release.
from rowant_lab.libero_env import m2_paths  # noqa: F401, I001
from rowant_lab.libero_env import grasp  # noqa: F401
from rowant_lab.libero_env import m2_kin  # noqa: F401
from rowant_lab.libero_env import m2_audit as AU  # noqa: F401
from rowant_lab.libero_env import m2_ground as GD  # noqa: F401
from rowant_lab.libero_env import m2_library as LB  # noqa: F401
from rowant_lab.libero_env import m2_rpaths as MP  # noqa: F401
from rowant_lab.libero_env import tools as T  # noqa: F401


class WatchdogTimeout(Exception):
    """The runner's per-episode wall watchdog fired (plan watchdog: 600 s wall per episode, runner side)."""


# m2f:perceive-r2 (verifier blocker B2/R1): infrastructure failures inside the episode are RUNNER EVENTS, never an
# outcome: a socket timeout or a dead perception server (the bridge raises TimeoutError / ConnectionError, not
# RemoteError) and the watchdog. The runner logs them; the crash and watchdog rules decide what is re-run.
RUNNER_EVENTS = (TimeoutError, ConnectionError, EOFError, WatchdogTimeout)
