"""Finding an object in the wrist camera's colour picture, taken from above the support (2026-11-16).

The wrist camera gives a colour picture and nothing else: no depth. Looking straight down at a support whose height
the robot knows, that is enough to place a thing ACROSS the support: a pixel's ray meets the plane at the thing's top
in one point. This module is that measurement:

* ``camera``: the wrist camera's intrinsics and pose from the hand's own reading and the camera's fixed place in the
  hand (``WRIST``: measured once on training tasks, as the main camera's stored pose was);
* ``find``: the picture is cut to a coarse grid; the support's colour is the picture's most common one; what differs
  from it, and is neither the robot's own fingers nor an object the main camera already placed (their boxes drawn
  into the picture), is cut into connected regions; the region nearest the expected point is the thing. Its middle
  and long axis are carried down the rays to the plane at the thing's top.

Policy side: numpy only, no simulator, no file. Nothing here is specific to one case; it is the robot's second eye
for whatever the main camera cannot see (behind something, or under the hand).
"""

from __future__ import annotations

import math

import numpy as np

# the wrist camera in the hand's frame (the grip site and the reported hand orientation), OpenGL axes: measured on
# training tasks with ``probes/clear_facts.py`` (the same in all eleven tasks probed, to the fourth decimal)
WRIST = {"fovy_deg": 75.0, "t_hand_cam": (0.05, 0.0, -0.097),
         "R_hand_cam": ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))}
GRID = 256
OPTS = {"min_dist": 38.0, "k_mad": 7.0, "min_area_m2": 2.5e-4, "max_area_ratio": 6.0, "known_grow_px": 4,
        "self_grow_px": 5, "close": 2, "open": 1}


def camera(eef_pos, R_hand, width: int, wrist: dict = WRIST):
    """(K, T_world_from_camera in OpenCV axes) of the wrist camera for a picture ``width`` px across."""
    R_hand = np.asarray(R_hand, float)
    Rc = R_hand @ np.asarray(wrist["R_hand_cam"], float)
    T = np.eye(4)
    T[:3, :3] = Rc @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = np.asarray(eef_pos, float) + R_hand @ np.asarray(wrist["t_hand_cam"], float)
    f = 0.5 * width / math.tan(math.radians(float(wrist["fovy_deg"])) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, width / 2.0], [0.0, 0.0, 1.0]]), T


def to_plane(K, T, uv, z: float):
    """World points (n, 3) where the rays of pixels ``uv`` (n, 2) meet the horizontal plane at height ``z``."""
    uv = np.atleast_2d(np.asarray(uv, float))
    rays = np.column_stack([(uv[:, 0] + 0.5 - K[0, 2]) / K[0, 0], (uv[:, 1] + 0.5 - K[1, 2]) / K[1, 1],
                            np.ones(len(uv))]) @ np.asarray(T, float)[:3, :3].T
    c = np.asarray(T, float)[:3, 3]
    s = (float(z) - c[2]) / np.where(np.abs(rays[:, 2]) < 1e-9, 1e-9, rays[:, 2])
    return c + rays * s[:, None]


def project(K, T, X):
    """Pixels (n, 2) and depths (n,) of world points ``X`` (n, 3)."""
    Xc = (np.atleast_2d(np.asarray(X, float)) - np.asarray(T, float)[:3, 3]) @ np.asarray(T, float)[:3, :3]
    z = np.where(np.abs(Xc[:, 2]) < 1e-9, 1e-9, Xc[:, 2])
    return np.column_stack([K[0, 0] * Xc[:, 0] / z + K[0, 2] - 0.5, K[1, 1] * Xc[:, 1] / z + K[1, 2] - 0.5]), Xc[:, 2]


def _hull(pts):
    """The convex hull of 2D points, counter-clockwise (Andrew's chain)."""
    pts = sorted(set((round(float(x), 3), round(float(y), 3)) for x, y in pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def footprint_area(boxes, cell: float = 0.002) -> float:
    """The area seen from above of a set of oriented boxes ((centre, R, half) each): the union of their footprints on
    a grid of ``cell`` metres. (A bowl's outline is 0.78 of its bounding box, and so is what a picture from above
    shows of it.)"""
    hulls = []
    for c, R, h in boxes:
        c, R, h = np.asarray(c, float), np.asarray(R, float), np.asarray(h, float)
        cs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]) * h
        hl = _hull((c + cs @ R.T)[:, :2])
        if len(hl) >= 3:
            hulls.append(np.asarray(hl, float))
    if not hulls:
        return 0.0
    lo = np.min([h.min(0) for h in hulls], axis=0) - cell
    hi = np.max([h.max(0) for h in hulls], axis=0) + cell
    X, Y = np.meshgrid(np.arange(lo[0], hi[0], cell) + 0.5 * cell, np.arange(lo[1], hi[1], cell) + 0.5 * cell)
    m = np.zeros(X.shape, bool)
    for h in hulls:
        inside = np.ones(X.shape, bool)
        for i in range(len(h)):
            a, b = h[i], h[(i + 1) % len(h)]
            inside &= (b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0]) >= 0.0
        m |= inside
    return float(m.sum()) * cell * cell


def draw_boxes(corners_list, K, T, n: int, scale: float, grow: int = 0) -> np.ndarray:
    """An (n, n) mask of the picture's coarse grid covered by boxes given as their world corners ((8, 3) each): each
    box's corners in front of the camera, their convex hull filled, grown by ``grow`` grid cells."""
    mask = np.zeros((n, n), bool)
    yy, xx = np.mgrid[0:n, 0:n]
    for C in corners_list:
        uv, z = project(K, T, C)
        if int((z > 0.02).sum()) < 3:
            continue
        h = _hull((uv[z > 0.02] + 0.5) / scale - 0.5)
        if len(h) < 3:
            continue
        hx, hy = np.array([p[0] for p in h]), np.array([p[1] for p in h])
        if hx.max() < -grow or hy.max() < -grow or hx.min() > n + grow or hy.min() > n + grow:
            continue
        inside = np.ones((n, n), bool)
        for i in range(len(h)):
            ax, ay, bx, by = hx[i], hy[i], hx[(i + 1) % len(h)], hy[(i + 1) % len(h)]
            ln = math.hypot(bx - ax, by - ay) or 1.0
            inside &= ((bx - ax) * (yy - ay) - (by - ay) * (xx - ax)) / ln >= -float(grow)
        mask |= inside
    return mask


def _shift_or(m: np.ndarray, grow: int) -> np.ndarray:
    out = m.copy()
    for _ in range(int(grow)):
        o = out.copy()
        o[1:] |= out[:-1]
        o[:-1] |= out[1:]
        o[:, 1:] |= out[:, :-1]
        o[:, :-1] |= out[:, 1:]
        out = o
    return out


def _shift_and(m: np.ndarray, shrink: int) -> np.ndarray:
    out = m.copy()
    for _ in range(int(shrink)):
        o = out.copy()
        o[1:] &= out[:-1]
        o[:-1] &= out[1:]
        o[:, 1:] &= out[:, :-1]
        o[:, :-1] &= out[:, 1:]
        out = o
    return out


def _label(mask: np.ndarray):
    """Connected regions (4-neighbours) of a small mask: (labels, count), labels from 1."""
    lab = np.zeros(mask.shape, np.int32)
    n = 0
    h, w = mask.shape
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0]:
            continue
        n += 1
        stack = [(int(y0), int(x0))]
        lab[y0, x0] = n
        while stack:
            y, x = stack.pop()
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n
                    stack.append((yy, xx))
    return lab, n


def find(rgb, K, T, top_z: float, self_boxes=(), known_boxes=(), hint_xy=None, expect_area_m2: float | None = None,
         opts: dict | None = None) -> dict | None:
    """The thing nearest ``hint_xy`` (else the largest) in a wrist picture ``rgb`` (H, W, 3 uint8, square) taken with
    camera (K, T): its middle ``xy`` on the horizontal plane at ``top_z`` (the caller gives the thing's middle height:
    off-centre, a standing thing's top and foot fall to either side of where it stands, and cancel there), its long
    axis ``yaw``, its area on that plane and half sizes along and across the axis, and the candidates' count. None
    when nothing but support, known objects and the robot itself is in view.

    ``self_boxes``, ``known_boxes``: world corners ((8, 3) each) of the robot's own gripper and of the objects
    already placed by the main camera; what they cover in the picture is not looked at."""
    o = {**OPTS, **(opts or {})}
    a = np.asarray(rgb)
    H, W = a.shape[:2]
    n = min(GRID, H)
    s = H // n
    small = a[:n * s, :n * s].astype(np.float32).reshape(n, s, n, s, 3).mean(axis=(1, 3))
    scale = float(s)
    hide = draw_boxes(self_boxes, K, T, n, scale, int(o["self_grow_px"]))
    hide |= draw_boxes(known_boxes, K, T, n, scale, int(o["known_grow_px"]))
    free = ~hide
    if int(free.sum()) < 0.2 * n * n:
        return None
    base = np.median(small[free], axis=0)  # the support's colour: the most common one in what is left
    dist = np.abs(small - base).sum(axis=2)
    d_free = dist[free]
    med = float(np.median(d_free))
    mad = float(np.median(np.abs(d_free - med)))
    thr = max(float(o["min_dist"]), med + float(o["k_mad"]) * 1.4826 * mad)
    m = (dist > thr) & free
    m = _shift_or(_shift_and(m, int(o["open"])), int(o["open"]))  # drop specks
    k = int(o["close"])  # close small gaps (a label's print, a bowl's shaded inside)
    pad = np.pad(m, k)
    m = _shift_and(_shift_or(pad, k), k)[k:-k, k:-k] & free if k else m
    lab, k = _label(m)
    if k == 0:
        return None
    cands = []
    for i in range(1, k + 1):
        v, u = np.nonzero(lab == i)
        uv = np.column_stack([(u + 0.5) * scale - 0.5, (v + 0.5) * scale - 0.5])
        P = to_plane(K, T, uv, top_z)[:, :2]
        # the area of one grid cell on the plane, from the cell's own corners (the plane is seen slanted off-centre)
        cell = to_plane(K, T, [[uv[:, 0].mean(), uv[:, 1].mean()], [uv[:, 0].mean() + scale, uv[:, 1].mean()],
                               [uv[:, 0].mean(), uv[:, 1].mean() + scale]], top_z)[:, :2]
        area = float(len(u)) * abs(float(np.cross(cell[1] - cell[0], cell[2] - cell[0])))
        if area < float(o["min_area_m2"]):
            continue
        ctr = P.mean(0)
        ev, evec = np.linalg.eigh(np.cov((P - ctr).T)) if len(P) > 2 else (np.array([0.0, 0.0]), np.eye(2))
        ax = evec[:, int(np.argmax(ev))]
        ax = ax if ax[0] >= 0 else -ax
        ay = np.array([-ax[1], ax[0]])
        sa, sb = (P - ctr) @ ax, (P - ctr) @ ay
        mid = ctr + ax * 0.5 * (float(sa.max()) + float(sa.min())) + ay * 0.5 * (float(sb.max()) + float(sb.min()))
        edge = bool(u.min() == 0 or v.min() == 0 or u.max() == n - 1 or v.max() == n - 1)
        cands.append({"xy": [float(mid[0]), float(mid[1])], "mean_xy": [float(ctr[0]), float(ctr[1])],
                      "yaw": float(math.atan2(ax[1], ax[0])), "area_m2": area,
                      "half": [0.5 * float(sa.max() - sa.min()), 0.5 * float(sb.max() - sb.min())],
                      "n_cells": int(len(u)), "edge": edge,
                      "rgb": [int(x) for x in np.median(small[v, u], axis=0)]})
    if expect_area_m2:
        ok = [c for c in cands if c["area_m2"] <= float(o["max_area_ratio"]) * float(expect_area_m2)]
        cands = ok or cands
    if not cands:
        return None
    if hint_xy is not None and o.get("prefer") == "size" and expect_area_m2:
        # (2026-11-16, ``relation_look``; off unless asked: among the things within ``within_m`` of the expected point,
        # the one whose size is nearest the named object's, and none when nothing is that near)
        near = [c for c in cands
                if float(np.linalg.norm(np.asarray(c["xy"]) - np.asarray(hint_xy, float)[:2])) <= float(o.get("within_m", 0.3))]
        if not near:
            return None
        near.sort(key=lambda c: abs(math.log(max(c["area_m2"], 1e-6) / float(expect_area_m2))))
        # ``ranked``: the things in that order, for a caller that asks a second viewpoint which of them stands there
        return {**near[0], "n_cands": len(cands), "thr": round(thr, 1), "base_rgb": [int(x) for x in base],
                "ranked": [{k: c[k] for k in ("xy", "yaw", "area_m2", "half", "n_cells", "edge", "rgb")} for c in near[:4]]}
    elif hint_xy is not None:
        best = min(cands, key=lambda c: float(np.linalg.norm(np.asarray(c["xy"]) - np.asarray(hint_xy, float)[:2])))
    else:
        best = max(cands, key=lambda c: c["area_m2"])
    return {**best, "n_cands": len(cands), "thr": round(thr, 1), "base_rgb": [int(x) for x in base]}
