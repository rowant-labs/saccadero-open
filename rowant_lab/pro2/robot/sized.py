"""A stored model at another size (2026-11-17): the same thing, taller, wider or smaller than the library has it.

The library holds each type at one size. A scene may hold the thing drawn at another: a cabinet a fifth taller, a
basket four fifths the size. The server's search, which fits the stored model, then accepts nothing, and what is left
is an unexplained region whose top stands at another height than the model's. This module makes the stored model
usable at the size that was measured:

* ``scaled_docs``: M1's collision model and the export's kinematics of a type, scaled by ``(sx, sy, sz)`` along the
  root body's own axes about its origin (a thing drawn at another size keeps its root where it was put);
* ``resize_type``: the episode's library takes those in place of the type's own, so every reader of the type (the
  handle plan, the sites, the obstacle boxes, the grasp records) sees the thing at its measured size.

Only for a type with ONE instance in the scene (a fixture, a receiver the look did not report): the library holds a
type once. The episode's library is made anew for every episode (``dev_episode``), so nothing outlives it.

What is exact and what is not: boxes whose axes lie along the root's (a cabinet's panels) scale exactly; a box
turned against them (a bowl's slanted wall) keeps its turn and takes the stretch along each of its own axes, which is
exact only for one factor in all directions. Bodies are taken as standing square to the root at rest (the cabinet's
do); a type with a body turned against its root is scaled by one factor only (refused otherwise). A sliding part's
travel grows with a thing drawn larger along it and is left as it is for one drawn smaller: the pull stops when the
part does (``OpenAgent._pull``), and a pull planned short would leave the part short of open.

Policy side: numpy only; it reads the library object the runner made for this episode and changes its entries.
"""

from __future__ import annotations

import copy

import numpy as np

from . import grasp as GR

RECORD_PARAMS = GR.RecordParams(theta_step_deg=15.0, b_step_m=0.006, z_step_m=0.006)  # as the agent's replanning


def _s3(s) -> np.ndarray:
    a = np.asarray(s, float).ravel()
    a = np.repeat(a, 3) if a.size == 1 else a
    if a.size != 3 or not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError(f"scale {s!r}")
    return a


def _square(exdoc: dict) -> bool:
    """Every body stands square to its parent at rest (no turn in ``body_quat``)."""
    for b in (exdoc.get("bodies") or {}).values():
        q = np.asarray(b.get("body_quat") or [1.0, 0.0, 0.0, 0.0], float)
        if abs(abs(float(q[0])) - 1.0) > 1e-6:
            return False
    return True


def _stretch(R, half, s3):
    """A box's half sizes after the scaling, each along its own axis (the columns of ``R``)."""
    R = np.asarray(R, float)
    return [float(h) * float(np.linalg.norm(s3 * R[:, i])) for i, h in enumerate(half)]


def scaled_docs(m1doc: dict, exdoc: dict, s) -> tuple:
    """``(m1doc', exdoc')``: deep copies scaled by ``s`` (one factor, or three along the root's axes)."""
    s3 = _s3(s)
    if not np.allclose(s3, s3[0]) and not _square(exdoc):
        raise ValueError("a type with a body turned against its root takes one factor only")
    m1, ex = copy.deepcopy(m1doc), copy.deepcopy(exdoc)
    for b in (m1.get("bodies") or {}).values():
        for pr in b.get("primitives") or []:
            pr["half"] = _stretch(pr["R"], pr["half"], s3)
            pr["centre"] = [float(x) for x in s3 * np.asarray(pr["centre"], float)]
            pr["exact"] = False
        if b.get("com") is not None:
            b["com"] = [float(x) for x in s3 * np.asarray(b["com"], float)]

    def joint(j):
        if not isinstance(j, dict):
            return
        if j.get("pos") is not None:
            j["pos"] = [float(x) for x in s3 * np.asarray(j["pos"], float)]
        if j.get("type") == "slide" and j.get("range") is not None and j.get("axis") is not None:
            k = float(np.linalg.norm(s3 * np.asarray(j["axis"], float)) / max(float(np.linalg.norm(j["axis"])), 1e-9))
            if k > 1.0:  # a thing drawn larger along its slide travels further; one drawn smaller: see the module doc
                q0 = float(j.get("qpos0") or 0.0)
                j["range"] = [float(q0 + (float(x) - q0) * k) for x in j["range"]]

    for b in (ex.get("bodies") or {}).values():
        if b.get("body_pos") is not None:
            b["body_pos"] = [float(x) for x in s3 * np.asarray(b["body_pos"], float)]
        for j in b.get("joints") or []:
            joint(j)
    for pd in (ex.get("parts") or {}).values():
        joint((pd or {}).get("joint"))
    for st in (ex.get("sites") or {}).values():
        st["size"] = _stretch(st.get("R") or np.eye(3), st["size"], s3)
        st["pos"] = [float(x) for x in s3 * np.asarray(st["pos"], float)]
    for pp in (ex.get("pose_in_part") or {}).values():
        if pp.get("t") is not None:
            pp["t"] = [float(x) for x in s3 * np.asarray(pp["t"], float)]
    ex["sized"] = m1["sized"] = [round(float(x), 4) for x in s3]
    return m1, ex


def resize_type(lib, typ: str, s) -> list:
    """The episode's library holds ``typ`` at ``s`` times its stored size from here on (``scaled_docs`` of the
    STORED documents, kept aside on the first call: resizing twice does not compound). An object's grasp records are
    built again for the scaled boxes, the way the agent builds them when it plans for another rest pose. Returns the
    factors as taken."""
    s3 = _s3(s)
    kept = getattr(lib, "_stored_docs", None)
    if kept is None:
        kept = lib._stored_docs = {}
    if typ not in kept:
        kept[typ] = (lib.types[typ], lib.export_types[typ], (lib.records or {}).get(typ))
    m1, ex = scaled_docs(kept[typ][0], kept[typ][1], s3)
    lib.types[typ], lib.export_types[typ] = m1, ex
    cache = getattr(lib, "_body_boxes", None)
    if isinstance(cache, dict):
        for k in [k for k in cache if k[0] == typ]:
            del cache[k]
    if kept[typ][2] is not None:
        root = m1["bodies"][m1["root_body"]]
        rec = GR.build_records(root["primitives"], m1.get("rest_up_in_body") or [0.0, 0.0, 1.0],
                               root.get("com") or [0.0, 0.0, 0.0], lib.jaw, RECORD_PARAMS, name=typ)
        if not len(np.asarray(rec.get("width", []) if isinstance(rec, dict) else [], float).ravel()):
            restore_type(lib, typ)  # a model the hand cannot take at that size (a flat box drawn flatter) is refused
            raise ValueError(f"{typ} at {s3.tolist()}: no grasp record")
        lib.records[typ] = rec
    return [round(float(x), 4) for x in s3]


def restore_type(lib, typ: str) -> bool:
    """The stored documents and grasp records of ``typ`` back in the episode's library, as before any resize (the
    very objects, not copies). True when something was put back."""
    kept = getattr(lib, "_stored_docs", None) or {}
    if typ not in kept:
        return False
    m1, ex, rec = kept.pop(typ)
    lib.types[typ], lib.export_types[typ] = m1, ex
    if rec is not None:
        lib.records[typ] = rec
    cache = getattr(lib, "_body_boxes", None)
    if isinstance(cache, dict):
        for k in [k for k in cache if k[0] == typ]:
            del cache[k]
    return True


def sized(lib, typ: str):
    """The factors ``typ`` stands at in this episode's library, or None when it is as stored."""
    return (lib.export_types.get(typ) or {}).get("sized")
