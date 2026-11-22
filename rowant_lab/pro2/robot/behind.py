"""Where a named object stands that the main camera cannot see (2026-11-16; the case in which
a panel hangs on the camera's line of sight to the target).

A thing that hangs in front of the scene hides a patch of the support from the camera. The look reports it as an
unexplained region whose lowest part is well above the support (``h_min``) and gives its middle in the picture
(``uv``). When the instruction's target is in none of the look's percepts (or the only one of its type fails the
instruction's relation) and such a region is in view, the target is taken to stand behind the middle of it: on the
camera's ray through that pixel, where the ray comes down to the height at which the target's model rests on
whatever is there (the table, or a reported object's top). The model is set there upright, in the turn it was
onboarded in: nothing was seen of it, so its turn is not known, and the grasp check (the gap between the closed
fingers) is what confirms that the hand found it.

Policy side: numpy only. ``cam`` is the look's own camera (``K`` and ``T_world_from_camera`` of the frame it sent).
"""

from __future__ import annotations

import numpy as np

from rowant_lab.libero_env.m2_kin import mat_to_quat, part_body_poses

HANG_MIN_M = 0.08   # a region whose lowest part is this far above the support hangs (an object stands on it)
HANG_MIN_PX = 3000  # and is at least this large in a 1024 px picture (a 12 x 16 cm panel is 30,000 to 60,000)
HANG_SHARE = 0.5    # or: this share of the region's picture columns end above the support (``hang_share``)


def rest_R(lib, typ: str) -> np.ndarray:
    """The rotation that stands ``typ``'s model the way it rests: its ``rest_up_in_body`` axis onto the world's up,
    by the least turn (six of the Object suite's types rest with their body's y axis up, not its z)."""
    up = np.asarray((getattr(lib, "types", None) or {}).get(typ, {}).get("rest_up_in_body") or [0.0, 0.0, 1.0], float)
    up = up / max(float(np.linalg.norm(up)), 1e-9)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(up, z), float(up @ z)
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def standing_px(blobs, share: float = 0.1) -> int:
    """The size (pixels) of the largest unexplained region that STANDS on the support: under ``share`` of its picture
    columns end above it. A scene with one thing hidden leaves little of that (the 24 first looks of the panel case on
    the pod, 2026-11-16: at most 3,900 px); a view that does not match the stored camera leaves the table and what
    stands on it unexplained (a moved camera, another lens: 23,000 to 148,000 px in each of 20 such first looks).
    Regions without the look's ``hang_share`` field do not count."""
    return max([int(b.get("n_px") or 0) for b in blobs or []
                if b.get("hang_share") is not None and float(b["hang_share"]) < float(share)] or [0])


def occluders(blobs, hang_m: float = HANG_MIN_M, min_px: int = HANG_MIN_PX, tall: dict | None = None,
              share: float = HANG_SHARE) -> list:
    """The look's unexplained regions that hang in front of the scene, largest first. ``tall`` (off by default;
    ``{"height_m", "px"}``): a region at least that high and that large counts too, whatever its lowest part reads
    (for a depth that does not show where a hanging thing ends). ``share``: the least ``hang_share`` of a region
    joined to something standing."""
    out = []
    for b in blobs or []:
        if b.get("uv") is None:
            continue
        n = int(b.get("n_px") or 0)
        hangs = b.get("h_min") is not None and float(b["h_min"]) >= hang_m and n >= min_px
        # a region joined in the picture to something standing (the target's own top showing over the panel's edge):
        # most of its columns still end above the support, and its hanging part is large
        part = float(b.get("hang_share") or 0.0) >= float(share) and int(b.get("hang_px") or 0) >= min_px \
            and b.get("hang_uv") is not None
        is_tall = bool(tall) and float(b.get("height") or 0.0) >= float(tall.get("height_m", 0.25)) \
            and n >= int(tall.get("px", 15000))
        if hangs or part or is_tall:
            # the middle of what hangs when the look gives it (the joined thing left out), else the region's own
            out.append({**b, "uv": b["hang_uv"]} if (part and not hangs and b.get("hang_uv") is not None) else b)
    return sorted(out, key=lambda b: -int(b["n_px"]))


def ray_at(K, T, uv, z: float):
    """The world point at height ``z`` on the camera's ray through pixel ``uv`` (None when the ray does not come
    down to that height in front of the camera)."""
    K, T = np.asarray(K, float), np.asarray(T, float)
    ray = T[:3, :3] @ np.array([(float(uv[0]) + 0.5 - K[0, 2]) / K[0, 0], (float(uv[1]) + 0.5 - K[1, 2]) / K[1, 1], 1.0])
    c = T[:3, 3]
    if abs(float(ray[2])) < 1e-9:
        return None
    s = (float(z) - float(c[2])) / float(ray[2])
    return None if s <= 0 else c + s * ray


def _bodies(lib, typ: str, R, t) -> dict:
    tdoc = lib.export_types[typ]
    bp = part_body_poses(tdoc, {tdoc["root_body"]: (np.asarray(R, float), np.asarray(t, float))}, {})
    return {r: {"pos": [float(x) for x in tt], "quat_wxyz": mat_to_quat(RR)} for r, (RR, tt) in bp.items()}


def table_z(lib, xy) -> float | None:
    """The top of the workcell's support under ``xy``: its highest box there, else its floor plane."""
    best = None
    for a in (getattr(lib, "workcell", None) or {}).get("arena") or []:
        if a.get("type") == "plane":
            z = float(a["point"][2])
        else:
            c = np.asarray(a["centre"], float)
            e = np.abs(np.asarray(a.get("R", np.eye(3)), float)) @ np.asarray(a["half"], float)
            if not np.all(np.abs(np.asarray(xy, float)[:2] - c[:2]) <= e[:2]):
                continue
            z = float(c[2] + e[2])
        best = z if best is None else max(best, z)
    return best


def support_under(lib, percepts, xy, skip_types=()) -> float | None:
    """What the target would stand on at ``xy``: the highest top there of a reported object (not of ``skip_types``),
    else the table."""
    from .ground import top_under

    best = table_z(lib, xy)
    for p in percepts or []:
        if p.get("type") in skip_types or not p.get("bodies"):
            continue
        try:
            z = top_under(lib, p, xy)
        except Exception:
            z = None
        if z is not None and (best is None or z > best):
            best = float(z)
    return best


def behind_pose(lib, typ: str, occ: dict, cam: dict, percepts=(), aim: str = "origin", on_z: float | None = None):
    """(R, t, record) of ``typ``'s model at rest behind the middle of the hanging region ``occ``, or None.

    The ray through the region's middle is followed down to the height of the model's own origin (``aim``
    "origin") or of the middle of its box ("box") when it rests on the support there. The support is the table or the
    reported object under that point (``on_z`` gives it outright); it is found in a few rounds, since a higher
    support moves the point along the slanted ray."""
    from .ground import aabb

    R = rest_R(lib, typ)
    lo, hi = aabb(lib, {"type": typ, "bodies": _bodies(lib, typ, R, np.zeros(3))})
    rise = -float(lo[2]) if aim == "origin" else 0.5 * float(hi[2] - lo[2])  # the aimed point above the support
    mid = 0.5 * (np.asarray(lo, float) + np.asarray(hi, float))
    z_s = on_z if on_z is not None else table_z(lib, occ.get("xy") or (0.0, 0.0))
    if z_s is None:
        return None
    hit = None
    for _ in range(4):
        hit = ray_at(cam["K"], cam["T"], occ["uv"], z_s + rise)
        if hit is None:
            return None
        if on_z is not None:
            break
        z_new = support_under(lib, percepts, hit[:2], skip_types=(typ,))
        if z_new is None or abs(z_new - z_s) < 1e-4:
            break
        z_s = z_new
    t = np.array([hit[0], hit[1], z_s - float(lo[2])])
    if aim == "box":  # the box's middle on the ray: the origin is offset by the box's own middle
        t[:2] = hit[:2] - mid[:2]
    return R, t, {"uv": [float(x) for x in occ["uv"]], "n_px": int(occ.get("n_px") or 0), "h_min": float(occ["h_min"]),
                  "support_z": round(float(z_s), 4), "xy": [round(float(x), 4) for x in t[:2]], "aim": aim}


def behind_percept(lib, typ: str, occ: dict, cam: dict, pid: str, percepts=(), aim: str = "origin",
                   on_z: float | None = None):
    """(percept, record) of the hidden target as ``behind_pose`` places it, or (None, None). The percept has no
    ``gid``: the server cannot track what the camera does not see."""
    got = behind_pose(lib, typ, occ, cam, percepts, aim, on_z)
    if got is None:
        return None, None
    R, t, rec = got
    return ({"id": pid, "type": typ, "gid": None, "score": 0.0, "n_px": 0, "fit": {"n_agree": 0, "inlier": 0.0},
             "source": "behind", "joints": {}, "bodies": _bodies(lib, typ, R, t)}, rec)
