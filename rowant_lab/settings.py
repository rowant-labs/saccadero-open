"""Where this code keeps its data on this machine. Standard library only.

One folder holds everything the code builds or reads apart from the code itself: the object libraries built from
the benchmark's download, caches and run folders. Set it with the environment variable ``ROWANT_DATA``; the
default is ``~/saccadero-data``.

This module names that one folder and nothing else, so that the policy side may import it. Locations only the
runner may know (the benchmark's download, the interpreters, the depth nets) are in ``rowant_lab/pro2/machine.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA = Path(os.environ.get("ROWANT_DATA") or Path.home() / "saccadero-data").expanduser()
