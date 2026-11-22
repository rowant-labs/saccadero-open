"""A stand-in for a second guard that is not part of this release (see ``hybrid_frontier/e9/isolation.py``).

The private version listed the running processes before an object memory was loaded. This one inspects nothing.
"""

from __future__ import annotations


class IsolationError(RuntimeError):
    """Kept for callers that name it. The stand-in never raises it."""


def hits(*_a, **_kw) -> list[str]:
    """Nothing is inspected, so nothing is found."""
    return []


def require(*_a, **_kw) -> dict:
    """The record the guard returned, with nothing inspected."""
    return {"lock_files_checked": [], "pattern": "", "hits": []}
