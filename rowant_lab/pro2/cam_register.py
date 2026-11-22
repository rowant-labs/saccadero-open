"""A camera that moves in the middle of an episode, read from the scene itself (stream PATH, round 3, second form).

M2 (perceived poses, grounded instruction): perception in the loop. TRUTH-FREE: robot side. It reads the robot's own
colour pictures and its depth network's output; no server request, nothing of the simulator.

The colour line's camera is a stored constant, checked once against the arm during the robot's first motion
(``arm_calib``). A camera that moves later (2026-11-18: one kind of practice episode moves it once, by 3.3 or 15.8 cm,
when the hand first nears its target) shifts everything a later look sees. The arm cannot show such a move in time: it
needs three frames of its own motion that all lie after it (``arm_cam_robot.again``, tried and reverted). The scene
can: almost all of it stands still, so one picture from before the move and the look's own picture after it give the
camera's change.

``arm_cam_robot["register"]`` (a dict of ``REGISTER`` overrides; off unless given):

* **The reference.** Once the check of the first motion has ended, ``arm_calib`` arms this and takes one frame through
  the camera tool. Its corner features (ORB) are kept with their places in the world, lifted through that frame's own
  depth at the camera the robot then assumes.
* **Every later front-camera frame** passes through ``look`` before the camera tool returns it:
  - its features are matched to the reference's. Median shift of the matches at most ``max_px``: the camera stands
    where it stood, nothing changes ("same").
  - More: the matched features, lifted through THIS frame's depth, are fitted rigidly onto their reference places
    (three-point samples, ``tol_m``, a fixed seed). The answer is the camera at which this frame's depth puts the
    still scene where the reference's depth put it. It is accepted with at least ``min_inliers`` agreeing matches,
    at least ``min_share`` of those that have depth, and a camera ``min_shift_m`` or ``min_turn_deg`` to
    ``max_shift_m`` and ``max_turn_deg`` from the reference's ("moved"). Otherwise the frame is used as it came
    ("refused").
  - A frame that matches the last registered picture keeps that camera ("as registered"); one that matches the
    reference again gets the reference's camera back ("back").

Three forms (``use``):

* ``"depth"``: as above. The colour line's network carries the stored camera's geometry in its depth, so this is
  the camera that makes the later depth agree with the earlier depth on the table. Measured on the Mac
  (2026-11-19): it lies 11.4 cm from the true camera after a 15.8 cm move and 2.2 cm after a 3.3 cm one, and still
  things then read 4 cm from where they stood.
* ``"pixels"``: the classical pose fit, from the reference's places and this frame's pixels alone. It finds the
  true camera (2 mm off in the same two episodes), and is computed and recorded in every form. But the frame's
  depth is still the network's, which does not fit the true camera.
* ``"scene"``: the classical fit's camera, and the frame's depth corrected to fit it. The still features' true
  depth in the new view is known (their reference places seen from the fitted camera); the network's depth at
  their pixels differs from it by a bias that is, for a plane, exactly a plane in inverse depth over the picture
  (``depth_plane``). That plane is added to the whole frame's inverse depth (``corrected``), for this frame and,
  fitted again on each, for every later frame the camera stays where it is.

**A thing that moves is not a camera that moves** (2026-11-19, after the first pod run of this): in
two types that add a movable body (D03, S03) the camera was "found moved" in 6 of 476 practice episodes while it
stood still (the camera then used lay 1 to 14 cm from the true one). The strongest corners of a picture crowd onto
its most textured thing; when that thing moves or is carried, most matches move with it, and one rigid body's
motion fits a camera's as well as a scene's does. Two guards, both on by default:

* ``grid``: the corners are taken evenly over the picture (at most ``3 features / grid^2`` of each of ``grid`` x
  ``grid`` cells, the strongest first), so no one thing holds most of them.
* The cells vote. A cell with at least ``cell_min`` matches has moved when its matches' median shift is over
  ``max_px``. The camera is taken as moved only when at least ``cells_moved`` of those cells moved, and at least
  ``cells_min`` cells voted. Fewer: things moved in a scene that stands still, or too little of the picture has
  anything to say, and nothing changes ("part moved").
"""

from __future__ import annotations

import time

import numpy as np

REGISTER = {
    "features": 3000,  # corners kept per picture
    "ratio": 0.75,  # a match counts when its best distance is under this share of its second best
    "max_px": 2.0,  # the camera stands still when the matches' median shift is at most this (pixels of the frame)
    "min_matches": 60,  # with fewer matches there is no answer and the frame is used as it came
    "tol_m": 0.02,  # the rigid fit: a match agrees within this
    "iters": 400,
    "min_inliers": 40,
    "min_share": 0.35,
    "min_shift_m": 0.015,  # nearer than both to the reference's camera: within the fit's precision, nothing changes
    "min_turn_deg": 0.5,
    "max_shift_m": 0.5,  # sanity, as the arm's fit has it
    "max_turn_deg": 35.0,
    "max_depth_m": 4.0,
    "px_tol": 3.0,  # the classical fit's reprojection tolerance (pixels)
    "use": "depth",  # or "pixels", or "scene" (see above)
    "max_rms_mm": 25.0,  # "scene": the corrected depth must agree with the still features' true depth this well
    "grid": 8,  # corners are taken evenly over grid x grid cells of the picture (0: the strongest anywhere)
    "cell_min": 3,  # a cell votes when it holds this many matches
    "cells_moved": 0.8,  # the camera moved only when this share of the voting cells moved,
    "cells_min": 8,  # and at least this many cells voted. Measured on the Mac (2026-11-19, the
    # benchmark's 1,024 px pictures): about 28 of the 64 cells hold corners at all; with a still camera 21 to 29
    # vote and 2 to 9 of them move (the arm, a thing carried); after a camera's move 13 to 17 vote and all move
    "seed": 20261119,
}


def features(rgb: np.ndarray, n: int, grid: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """ORB corners of a colour picture: their pixels (x, y) and descriptors. With ``grid`` the corners are taken
    evenly: of four times as many found, at most ``3 n / grid^2`` (rounded up) of each of ``grid`` x ``grid``
    cells of the picture are kept, the strongest first (a picture's corners lie in under half of its cells, so a
    cap of ``n / grid^2`` kept under half of ``n``)."""
    import cv2

    g = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    none = (np.zeros((0, 2)), np.zeros((0, 32), np.uint8))
    if int(grid) <= 0:
        kp, des = cv2.ORB_create(nfeatures=int(n)).detectAndCompute(g, None)
    else:
        orb = cv2.ORB_create(nfeatures=4 * int(n))
        kp = orb.detect(g, None)
        if not len(kp):
            return none
        h, w = g.shape[:2]
        cap, cells = -(-3 * int(n) // (int(grid) ** 2)), {}
        for k in sorted(kp, key=lambda k: (-k.response, k.pt[1], k.pt[0])):  # (the order is fixed: equal responses)
            c = (min(int(k.pt[1] * grid / h), grid - 1), min(int(k.pt[0] * grid / w), grid - 1))
            if len(cells.setdefault(c, [])) < cap:
                cells[c].append(k)
        kp, des = orb.compute(g, [k for c in sorted(cells) for k in cells[c]])
    if des is None or not len(kp):
        return none
    return np.array([k.pt for k in kp], dtype=np.float64), des


def cell_vote(xy: np.ndarray, shift: np.ndarray, shape: tuple, grid: int, cell_min: int, max_px: float) -> tuple[int, int]:
    """(voting cells, cells that moved): the matches by the cell of the picture they lie in; a cell with at least
    ``cell_min`` of them votes, and has moved when their median shift is over ``max_px``."""
    g = max(1, int(grid))
    h, w = shape[:2]
    r = np.minimum((xy[:, 1] * g / h).astype(np.int64), g - 1)
    c = np.minimum((xy[:, 0] * g / w).astype(np.int64), g - 1)
    idx = r * g + c
    n = moved = 0
    for i in np.unique(idx):
        d = shift[idx == i]
        if len(d) >= int(cell_min):
            n += 1
            moved += bool(float(np.median(d)) > float(max_px))
    return n, moved


def match(des_a: np.ndarray, des_b: np.ndarray, ratio: float) -> np.ndarray:
    """Index pairs (a, b): b is a's nearest by a clear margin, and no b is used twice (the nearer a keeps it)."""
    import cv2

    if len(des_a) < 2 or len(des_b) < 2:
        return np.zeros((0, 2), np.int64)
    best: dict = {}
    for m in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_a, des_b, k=2):
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            q = best.get(m[0].trainIdx)
            if q is None or m[0].distance < q[1]:
                best[m[0].trainIdx] = (m[0].queryIdx, m[0].distance)
    return np.array(sorted((a, b) for b, (a, _d) in best.items()), dtype=np.int64).reshape(-1, 2)


def lift(fr: dict, uv: np.ndarray, max_depth_m: float) -> tuple[np.ndarray, np.ndarray]:
    """The pixels' points in the camera's frame through the frame's own depth (the nearest pixel's), and which have
    one."""
    d = np.asarray(fr["depth_u16"])
    K = np.asarray(fr["K"], float)
    u = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, d.shape[1] - 1)
    v = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, d.shape[0] - 1)
    z = d[v, u].astype(np.float64) * float(fr["depth_scale_mm"]) * 1e-3
    ok = (z > 0) & (z < float(max_depth_m))
    return np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0] * z, (uv[:, 1] - K[1, 2]) / K[1, 1] * z, z], 1), ok


def kabsch(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t with B nearest to A R^T + t (rows are points)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cb - R @ ca


def rigid(A: np.ndarray, B: np.ndarray, tol: float, iters: int, seed: int):
    """The rigid motion that puts most of A within ``tol`` of B: three-point samples, then least squares over the
    agreeing rows until they stop changing. (R, t, agreeing) or None."""
    n = len(A)
    if n < 3:
        return None
    rng = np.random.default_rng(int(seed))
    best = None
    for _ in range(int(iters)):
        i = rng.choice(n, 3, replace=False)
        a = A[i]
        if np.linalg.norm(np.cross(a[1] - a[0], a[2] - a[0])) < 1e-4:  # three points nearly on one line
            continue
        R, t = kabsch(a, B[i])
        inl = np.linalg.norm(A @ R.T + t - B, axis=1) < tol
        if best is None or int(inl.sum()) > int(best.sum()):
            best = inl
    if best is None or int(best.sum()) < 3:
        return None
    for _ in range(5):
        R, t = kabsch(A[best], B[best])
        inl = np.linalg.norm(A @ R.T + t - B, axis=1) < tol
        if int(inl.sum()) < 3:
            return None
        if bool((inl == best).all()):
            break
        best = inl
    R, t = kabsch(A[best], B[best])
    return R, t, best


def pose_diff(Ta: np.ndarray, Tb: np.ndarray) -> tuple[float, float]:
    """How far two cameras lie apart: (m, deg)."""
    R = Ta[:3, :3].T @ Tb[:3, :3]
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), float(np.degrees(np.arccos(c)))


def _T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def classical(world: np.ndarray, uv: np.ndarray, K: np.ndarray, px_tol: float, iters: int, seed: int):
    """The camera from world points and their pixels alone (the classical pose fit, with samples). (T, agreeing) or
    None."""
    import cv2

    if len(world) < 6:
        return None
    cv2.setRNGSeed(int(seed) % (2 ** 31 - 1))
    ok, rvec, tvec, inl = cv2.solvePnPRansac(np.ascontiguousarray(world, dtype=np.float64),
                                             np.ascontiguousarray(uv, dtype=np.float64), np.asarray(K, np.float64),
                                             None, iterationsCount=int(iters), reprojectionError=float(px_tol),
                                             flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < 6:
        return None
    R_cw = cv2.Rodrigues(rvec)[0]
    return _T(R_cw.T, -R_cw.T @ tvec.ravel()), np.asarray(inl).ravel().astype(np.int64)


def depth_plane(z_true: np.ndarray, z_net: np.ndarray, un: np.ndarray, vn: np.ndarray, rounds: int = 8, k: float = 2.5):
    """a, b, c with 1/z_true nearest to 1/z_net + a + b un + c vn (``un``, ``vn``: the pixels' rays, x/z and y/z):
    least squares, rows farther than ``k`` spreads from the middle left out and the fit repeated. For points of
    one plane the difference of two cameras' inverse depths is exactly such a plane. ((a, b, c), rows used) or
    None."""
    if len(z_true) < 6:
        return None
    d = 1.0 / z_true - 1.0 / z_net
    M = np.stack([np.ones_like(un), un, vn], 1)
    use = np.ones(len(d), bool)
    x = None
    for _ in range(int(rounds)):
        if int(use.sum()) < 6:
            return None
        x = np.linalg.lstsq(M[use], d[use], rcond=None)[0]
        res = d - M @ x
        mid = float(np.median(res[use]))  # (the first fit is pulled by the stray rows: the others sit off zero)
        spread = max(1.4826 * float(np.median(np.abs(res[use] - mid))), 1e-5)
        nxt = np.abs(res - mid) <= k * spread
        if bool((nxt == use).all()):
            break
        use = nxt
    return x, use


def corrected(fr: dict, plane) -> np.ndarray:
    """The frame's depth with the plane added to its inverse depth (``depth_plane``); a pixel that had no depth, or
    that would lie beyond what the frame's numbers hold, has none."""
    d = np.asarray(fr["depth_u16"])
    K, sc = np.asarray(fr["K"], float), float(fr["depth_scale_mm"]) * 1e-3
    un = (np.arange(d.shape[1], dtype=np.float64) - K[0, 2]) / K[0, 0]
    vn = (np.arange(d.shape[0], dtype=np.float64) - K[1, 2]) / K[1, 1]
    inv = 1.0 / np.maximum(d.astype(np.float64) * sc, 1e-9) + float(plane[0]) + float(plane[1]) * un[None, :] \
        + float(plane[2]) * vn[:, None]
    ok = (d > 0) & (inv > 1.0 / (65535 * sc))
    out = np.zeros(d.shape, np.uint16)
    out[ok] = np.clip(np.rint(1.0 / inv[ok] / sc), 1, 65535).astype(np.uint16)
    return out


class Register:
    def __init__(self, o: dict | None = None):
        self.o = {**REGISTER, **dict(o or {})}
        if self.o["use"] not in ("depth", "pixels", "scene"):
            raise ValueError(f"register.use must be depth, pixels or scene, not {self.o['use']!r}")
        self.armed = False
        self.ref = None  # the reference: features, their places in the world, the camera
        self.cur = None  # the last registered picture: features, the camera
        self.last = None  # the last accepted fit's cameras (the evaluator's record compares them with the true one)
        self.rec: dict = {"looks": []}
        self.s = 0.0

    def arm(self) -> None:
        """The next front-camera frame is the reference."""
        self.armed = True

    def _shift(self, a: dict, xy: np.ndarray, des: np.ndarray, shape: tuple):
        """The matches to a kept picture, their median shift (None: too few matches), and the cells' vote."""
        o = self.o
        m = match(a["des"], des, float(o["ratio"]))
        if len(m) < int(o["min_matches"]):
            return m, None, (0, 0)
        d = np.linalg.norm(a["xy"][m[:, 0]] - xy[m[:, 1]], axis=1)
        return m, float(np.median(d)), cell_vote(xy[m[:, 1]], d, shape, int(o["grid"]) or 1, int(o["cell_min"]),
                                                 float(o["max_px"]))

    def _whole(self, px: float, vote: tuple) -> bool:
        """The picture as a whole has shifted: the median says so, and so do most of the cells that vote."""
        o = self.o
        need = min(int(o["cells_min"]), max(1, int(o["grid"])) ** 2) if int(o["grid"]) > 0 else 1
        return px > float(o["max_px"]) and vote[0] >= need and vote[1] >= float(o["cells_moved"]) * vote[0]

    def _plane(self, fr: dict, world: np.ndarray, uv: np.ndarray, T: np.ndarray):
        """``scene``: the plane in inverse depth that makes this frame's depth fit the still points ``world`` seen at
        ``uv`` from the camera ``T``. (plane, record) or None."""
        K = np.asarray(fr["K"], float)
        z_true = ((world - T[:3, 3]) @ T[:3, :3])[:, 2]
        _pc, ok = lift(fr, uv, float(self.o["max_depth_m"]))
        ok &= z_true > 0.05
        if int(ok.sum()) < int(self.o["min_inliers"]):
            return None
        z_true, uv = z_true[ok], uv[ok]
        z_net = _pc[ok][:, 2]
        un, vn = (uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1]
        got = depth_plane(z_true, z_net, un, vn)
        if got is None:
            return None
        x, use = got
        z_fit = 1.0 / (1.0 / z_net + x[0] + x[1] * un + x[2] * vn)
        rec = {"n": int(use.sum()), "of": int(len(use)),
               "rms_mm_before": round(float(np.sqrt(np.mean((z_net[use] - z_true[use]) ** 2))) * 1e3, 2),
               "rms_mm": round(float(np.sqrt(np.mean((z_fit[use] - z_true[use]) ** 2))) * 1e3, 2),
               "plane": [round(float(v), 6) for v in x]}
        if rec["n"] < int(self.o["min_inliers"]) or rec["rms_mm"] > float(self.o["max_rms_mm"]):
            return None
        return x, rec

    def look(self, fr: dict):
        """Called by the camera tool with every front-camera frame before it is returned. None: the frame as it is.
        Else a dict: ``T``, the camera every later frame is to use (absent: as it was), and ``depth_u16``, this
        frame's depth corrected (``scene`` alone; absent: as it came)."""
        if not self.armed:
            return None
        a = time.perf_counter()
        try:
            return self._look(fr)
        except Exception as exc:  # a registration that fails is no registration: the look is used as it came
            self.rec.setdefault("errors", []).append({"at": int(fr.get("step", -1)),
                                                      "error": f"{type(exc).__name__}: {str(exc)[:120]}"})
            return None
        finally:
            self.s += time.perf_counter() - a
            self.rec["s"] = round(self.s, 3)

    def _look(self, fr: dict):
        o = self.o
        xy, des = features(fr["rgb"], int(o["features"]), int(o["grid"]))
        shape = np.asarray(fr["rgb"]).shape
        T0 = np.asarray(fr["T_world_from_camera"], float)
        if self.ref is None:
            pc, ok = lift(fr, xy, float(o["max_depth_m"]))
            self.ref = {"xy": xy, "des": des, "world": pc @ T0[:3, :3].T + T0[:3, 3], "ok": ok, "T": T0.copy()}
            self.rec["reference"] = {"at": int(fr.get("step", -1)), "features": int(len(xy)), "with_depth": int(ok.sum())}
            return None
        doc = {"at": int(fr.get("step", -1)), "features": int(len(xy))}
        self.rec["looks"].append(doc)
        if self.cur is not None:
            m, px, vote = self._shift(self.cur, xy, des, shape)
            if px is not None and not self._whole(px, vote):
                # (the camera stands where the last fit put it; things may have moved before it)
                doc.update(status="as registered", matches=int(len(m)), px=round(px, 2), cells=list(vote))
                if self.cur.get("plane") is None:
                    return None
                # ``scene``: this frame's own depth is corrected, by a plane fitted on it where enough of the
                # reference is still in view from the registered camera, else by the registered frame's plane
                plane, T = self.cur["plane"], self.cur["T"]
                mr = match(self.ref["des"], des, float(o["ratio"]))
                mr = mr[self.ref["ok"][mr[:, 0]]] if len(mr) else mr
                if len(mr) >= int(o["min_inliers"]):
                    w, uv = self.ref["world"][mr[:, 0]], xy[mr[:, 1]]
                    K = np.asarray(fr["K"], float)
                    pc = (w - T[:3, 3]) @ T[:3, :3]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        e = np.hypot(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2] - uv[:, 0],
                                     K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2] - uv[:, 1])
                    near = (pc[:, 2] > 0.05) & (e <= float(o["px_tol"]))
                    got = self._plane(fr, w[near], uv[near], T) if int(near.sum()) >= int(o["min_inliers"]) else None
                    if got is not None:
                        plane, doc["scene"] = got
                return {"depth_u16": corrected(fr, plane)}
        m, px, vote = self._shift(self.ref, xy, des, shape)
        doc["matches"] = int(len(m))
        if px is None:
            doc["status"] = "too few matches"
            return None
        doc.update(px=round(px, 2), cells=list(vote))
        if not self._whole(px, vote):
            if self.cur is None:
                # nothing changes, whether nothing moved or only a part of the scene did
                doc["status"] = "same" if px <= float(o["max_px"]) and vote[1] == 0 else "part moved"
                return None
            if px <= float(o["max_px"]):
                self.cur = None
                doc["status"] = "back"
                return {"T": self.ref["T"].copy()}
            doc["status"] = "part moved"  # (neither the registered picture nor the reference: as it was)
            return None
        ref = self.ref
        pc, ok = lift(fr, xy[m[:, 1]], float(o["max_depth_m"]))
        both = ok & ref["ok"][m[:, 0]]
        A, B = pc[both], ref["world"][m[:, 0]][both]
        doc["with_depth"] = int(both.sum())
        seed = int(o["seed"])
        fit = rigid(A, B, float(o["tol_m"]), int(o["iters"]), seed) if len(A) >= int(o["min_inliers"]) else None
        Td = None
        if fit is not None:
            R, t, inl = fit
            Td = _T(R, t)
            doc["depth"] = {"inliers": int(inl.sum()), "share": round(float(inl.mean()), 3),
                            "rms_mm": round(float(np.sqrt(np.mean(np.sum((A[inl] @ R.T + t - B[inl]) ** 2, 1)))) * 1e3, 2),
                            "from_reference": [round(x, 4) for x in pose_diff(ref["T"], Td)]}
        rok = ref["ok"][m[:, 0]]
        cl = classical(ref["world"][m[:, 0]][rok], xy[m[:, 1]][rok], np.asarray(fr["K"], float), float(o["px_tol"]),
                       int(o["iters"]), seed)
        Tp, pin = None, None
        if cl is not None:
            Tp, pin = cl
            doc["pixels"] = {"inliers": int(len(pin)), "share": round(len(pin) / max(1, int(rok.sum())), 3),
                             "from_reference": [round(x, 4) for x in pose_diff(ref["T"], Tp)]}
            if Td is not None:
                doc["pixels"]["from_depth"] = [round(x, 4) for x in pose_diff(Td, Tp)]
        T, d = (Td, doc.get("depth")) if o["use"] == "depth" else (Tp, doc.get("pixels"))
        plane = None
        if o["use"] == "scene" and Tp is not None:
            got = self._plane(fr, ref["world"][m[:, 0]][rok][pin], xy[m[:, 1]][rok][pin], Tp)
            if got is None:
                doc.update(status="refused", why="the depth does not fit")
                return None
            plane, doc["scene"] = got
        if T is None:
            doc.update(status="refused", why="no fit")
            return None
        if d["inliers"] < int(o["min_inliers"]) or d["share"] < float(o["min_share"]):
            doc.update(status="refused", why="too few agree")
            return None
        shift, turn = d["from_reference"]
        if shift > float(o["max_shift_m"]) or turn > float(o["max_turn_deg"]):
            doc.update(status="refused", why="too far")
            return None
        if shift < float(o["min_shift_m"]) and turn < float(o["min_turn_deg"]):
            doc.update(status="refused", why="no change in the camera")
            return None
        self.cur = {"xy": xy, "des": des, "T": T.copy(), "plane": plane}
        self.last = {"T_depth": Td, "T_pixels": Tp, "used": o["use"]}
        doc["status"] = "moved"
        return {"T": T.copy()} if plane is None else {"T": T.copy(), "depth_u16": corrected(fr, plane)}
