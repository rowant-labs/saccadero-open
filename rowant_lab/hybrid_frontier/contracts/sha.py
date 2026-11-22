"""sha256 helpers for the hybrid-frontier contracts.

Two digests are used across the package:

* ``sha256_file(path)``: the digest of a file's bytes, the same one-liner the lab's protocol
  freezers use (``rowant_lab/bench_v1/freeze_bench_protocol.py:172-173``).
* ``sha256_arrays(arrays)``: the digest of a dict of numpy arrays. Keys are sorted, and each
  array contributes ``key`` + NUL + ``dtype`` + ``shape`` + raw bytes, the dtype/shape/bytes
  recipe of ``_tensor_sha256`` in ``rowant_lab/bench_v2/fused_recorder.py:156-158`` (used
  there for the graph-digest binding), extended with the key so that renaming an array changes
  the digest.

Nothing here reads run logs or truth; the helpers hash whatever bytes they are given.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

HEX64 = 64


def sha256_bytes(data: bytes) -> str:
    """Hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    """Hex digest of a file's bytes (``freeze_bench_protocol.py:172-173``). Reads the whole file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    """Hex digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj) -> str:
    """Sorted-key, compact JSON (the one byte form a dict hashes to)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(obj) -> str:
    """Hex digest of ``canonical_json(obj)``. Refuses NaN/inf (``allow_nan=False``)."""
    return sha256_text(canonical_json(obj))


def _array_chunks(a: np.ndarray):
    a = np.ascontiguousarray(a)
    yield f"{a.dtype}{a.shape}".encode()
    yield a.tobytes()


def sha256_array(a) -> str:
    """Hex digest of one array: ``dtype`` + ``shape`` + bytes (``fused_recorder.py:156-158``)."""
    h = hashlib.sha256()
    for chunk in _array_chunks(np.asarray(a)):
        h.update(chunk)
    return h.hexdigest()


def array_digests(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    """``{key: sha256_array(value)}`` with keys in sorted order."""
    return {k: sha256_array(arrays[k]) for k in sorted(arrays)}


def sha256_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    """One hex digest over a dict of arrays: sorted keys, each as key + NUL + dtype + shape + bytes."""
    if not isinstance(arrays, Mapping):
        raise TypeError("sha256_arrays expects a mapping of name -> array")
    h = hashlib.sha256()
    for key in sorted(arrays):
        if not isinstance(key, str):
            raise TypeError(f"array key {key!r} is not a string")
        h.update(key.encode("utf-8") + b"\0")
        for chunk in _array_chunks(np.asarray(arrays[key])):
            h.update(chunk)
    return h.hexdigest()


def is_hex64(value) -> bool:
    """True for a lowercase 64-character hex string (a full sha256)."""
    return isinstance(value, str) and len(value) == HEX64 and all(c in "0123456789abcdef" for c in value)
