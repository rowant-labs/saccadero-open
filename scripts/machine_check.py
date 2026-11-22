#!/usr/bin/env python3
"""Does this machine compute the same on every processor? Run it before any evaluation.

    <server python> scripts/machine_check.py [--repeat 2] [--jobs 8]

It runs a fixed set of numpy operations pinned to each processor in turn and compares the results within this
machine. Exit 0: every processor agrees. Exit 1: some differ (they are named; do not run on this machine).
During development one rented machine had a processor core that returned wrong quotients without any error, and
every run on it carried wrong numbers. Needs Python 3.10 or later and numpy; reads no file of the project.
The check itself is ``rowant_lab/pro2/pod_selfcheck.py``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rowant_lab.pro2 import pod_selfcheck  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(pod_selfcheck.main())
