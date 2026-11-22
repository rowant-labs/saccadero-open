"""A stand-in for a guard that is not part of this release.

On the machines this code was developed on, every program that loads an object memory first called a guard. It
listed the running processes and refused to start while another experiment of ours was running on that machine.
It protected our own experiments and has no use on anyone else's machine, so it is not published: it inspected the
machine it ran on.

This file keeps the guard's three entry points so that the calling code is unchanged. They inspect nothing and
always report that the way is clear. See docs/CHANGES-FROM-AS-RUN.md.
"""

from __future__ import annotations

import time

ISOLATION_FORMAT = "rowant hybrid_frontier e9 isolation check v1 (public stand-in: nothing is inspected)"


class IsolationError(RuntimeError):
    """Kept for callers that name it. The stand-in never raises it."""


def scan(**_kw) -> dict:
    """The record the guard returned, with nothing inspected: no process list, no lock file, always clear."""
    return {"format": ISOLATION_FORMAT, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "locks_checked": [], "patterns": [], "hits": [], "exempt": [], "n_processes": None, "locks_present": [],
            "clear": True}


def public(rec) -> dict:
    """The record as it may be written to a file (the stand-in's record holds nothing private)."""
    return dict(rec)


def require(**kw) -> dict:
    """The record of ``scan``; the stand-in never refuses."""
    return public(scan(**kw))
