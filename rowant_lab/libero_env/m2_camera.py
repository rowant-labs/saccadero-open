"""M2 camera tool (runner side; plan camera). Renders RGB-D on demand through robosuite's own offscreen render.

M2 (perceived poses, grounded instruction): perception in the loop. ``make_camera_tool(env_)`` returns a closure
``look(camera, width, step) -> frame dict`` with exactly the keys ``FRAME_KEYS``:

* ``rgb`` uint8 (H, W, 3) and ``depth_u16`` uint16 (H, W), in OpenCV row order (row 0 = top: robosuite renders in
  OpenGL order and the tool flips both up-down first);
* depth in metres along the optical axis by robosuite 1.4.0's ``get_real_depth_map`` formula (``near = znear x
  extent``, ``far = zfar x extent``, ``d = near / (1 - z (1 - near / far))``), encoded in 0.1 mm units; the far plane
  (``z >= 1 - 1e-7``) and anything beyond 6.5 m is 0 (no reading);
* ``K`` (square pixels, principal point (W / 2, H / 2), focal ``(H / 2) / tan(fovy / 2)``, as robosuite's
  ``get_camera_intrinsic_matrix``) and ``T_world_from_camera`` = ``[cam_xmat | cam_xpos] diag(1, -1, -1, 1)`` (MuJoCo
  camera axes to OpenCV axes, as ``get_camera_extrinsic_matrix``), read at the render.

It touches only ``sim.render(width, height, camera_name=..., depth=True)`` (never segmentation),
``model.camera_name2id``, ``model.cam_fovy``, ``data.cam_xpos``, ``data.cam_xmat``, ``model.stat.extent`` and
``model.vis.map.znear`` / ``zfar`` (an AST test whitelists these chains, alias-proof: the only name bound to a
simulator object is ``s = env_.sim``, and every chain from it must be one of the declared leaves). Images stay in
memory; nothing is saved.

Runtime output check (plan truth_guard.camera_tool; verifier minor M9): every frame is checked by ``check_frame``
before it is returned: its keys are exactly ``FRAME_KEYS``; ``rgb`` uint8 (W, W, 3) and ``depth_u16`` uint16 (W, W),
C-contiguous; ``depth_scale_mm`` 0.1; ``K`` float64 (3, 3) with the declared intrinsics; ``T_world_from_camera``
float64 (4, 4), finite, a rigid transform; ``width`` = ``height`` = the requested width; ``step`` an int. A frame that
fails raises ``CameraOutputError``. The first check's details are kept in ``look.first_check`` and every log row
says it was checked.
"""

from __future__ import annotations

import math
import time

import numpy as np

FRAME_KEYS = ("camera", "rgb", "depth_u16", "depth_scale_mm", "K", "T_world_from_camera", "width", "height", "step")
DEPTH_SCALE_MM = 0.1
MAX_DEPTH_M = 6.5
FAR_Z = 1.0 - 1e-7
CV_FROM_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def real_depth(zbuf: np.ndarray, extent: float, znear: float, zfar: float) -> np.ndarray:
    near = znear * extent
    far = zfar * extent
    return near / (1.0 - zbuf * (1.0 - near / far))


def encode_depth(depth_m: np.ndarray, zbuf: np.ndarray) -> np.ndarray:
    d = np.rint(depth_m / (DEPTH_SCALE_MM * 1e-3))
    bad = (zbuf >= FAR_Z) | ~np.isfinite(depth_m) | (depth_m > MAX_DEPTH_M) | (depth_m <= 0)
    d[bad] = 0
    return np.clip(d, 0, 65535).astype(np.uint16)


def intrinsics(fovy_deg: float, width: int, height: int) -> np.ndarray:
    f = 0.5 * height / math.tan(math.radians(fovy_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


class CameraOutputError(RuntimeError):
    """A camera frame that is not exactly the declared output."""


def check_frame(fr: dict, width: int) -> dict:
    """The runtime output check of one frame (raises ``CameraOutputError``); returns its details."""
    bad = []
    keys = tuple(fr)
    if set(keys) != set(FRAME_KEYS) or len(keys) != len(FRAME_KEYS):
        bad.append(f"keys {sorted(keys)} != {sorted(FRAME_KEYS)}")
    W = int(width)
    rgb, dep, K, T = fr.get("rgb"), fr.get("depth_u16"), fr.get("K"), fr.get("T_world_from_camera")
    if not (isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.shape == (W, W, 3)
            and rgb.flags["C_CONTIGUOUS"]):
        bad.append("rgb is not uint8 (W, W, 3) C-contiguous")
    if not (isinstance(dep, np.ndarray) and dep.dtype == np.uint16 and dep.shape == (W, W)
            and dep.flags["C_CONTIGUOUS"]):
        bad.append("depth_u16 is not uint16 (W, W) C-contiguous")
    if fr.get("depth_scale_mm") != DEPTH_SCALE_MM:
        bad.append("depth_scale_mm is not 0.1")
    if not (isinstance(K, np.ndarray) and K.dtype == np.float64 and K.shape == (3, 3)
            and K[0, 1] == 0.0 and K[1, 0] == 0.0 and K[2, 0] == 0.0 and K[2, 1] == 0.0 and K[2, 2] == 1.0
            and K[0, 0] == K[1, 1] and K[0, 2] == W / 2.0 and K[1, 2] == W / 2.0):
        bad.append("K is not the declared float64 intrinsics")
    if not (isinstance(T, np.ndarray) and T.dtype == np.float64 and T.shape == (4, 4) and np.all(np.isfinite(T))
            and np.allclose(T[3], [0.0, 0.0, 0.0, 1.0])
            and np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-6)):
        bad.append("T_world_from_camera is not a finite float64 rigid transform")
    if not (type(fr.get("width")) is int and type(fr.get("height")) is int and fr["width"] == W and fr["height"] == W):
        bad.append("width/height differ from the request")
    if type(fr.get("step")) is not int:
        bad.append("step is not an int")
    if not isinstance(fr.get("camera"), str):
        bad.append("camera is not a string")
    if bad:
        raise CameraOutputError("; ".join(bad))
    return {"keys": list(keys), "rgb": [list(rgb.shape), rgb.dtype.name], "depth_u16": [list(dep.shape), dep.dtype.name],
            "K": [list(K.shape), K.dtype.name], "T_world_from_camera": [list(T.shape), T.dtype.name],
            "depth_scale_mm": fr["depth_scale_mm"], "width": fr["width"], "height": fr["height"]}


def make_camera_tool(env_):
    """The tool reads ``env_.sim`` at every call (LIBERO's hard reset builds a new simulator at each reset)."""
    log: list = []

    def look(camera: str, width: int, step: int) -> dict:
        t0 = time.perf_counter()
        s = env_.sim
        extent = float(s.model.stat.extent)
        znear = float(s.model.vis.map.znear)
        zfar = float(s.model.vis.map.zfar)
        cid = s.model.camera_name2id(camera)
        fovy = float(s.model.cam_fovy[cid])
        rgb, zb = s.render(width, width, camera_name=camera, depth=True)
        rgb = np.ascontiguousarray(rgb[::-1])
        zb = np.ascontiguousarray(zb[::-1]).astype(np.float64)
        dm = real_depth(zb, extent, znear, zfar)
        T = np.eye(4)
        T[:3, :3] = np.asarray(s.data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
        T[:3, 3] = np.asarray(s.data.cam_xpos[cid], dtype=np.float64)
        T = T @ CV_FROM_GL
        out = {"camera": camera, "rgb": rgb.astype(np.uint8), "depth_u16": encode_depth(dm, zb),
               "depth_scale_mm": DEPTH_SCALE_MM, "K": intrinsics(fovy, width, width), "T_world_from_camera": T,
               "width": int(width), "height": int(width), "step": int(step)}
        chk = check_frame(out, int(width))
        if look.first_check is None:
            look.first_check = chk
        log.append({"camera": camera, "width": int(width), "step": int(step),
                    "render_s": round(time.perf_counter() - t0, 4), "checked": True})
        return out

    look.log = log
    look.first_check = None
    return look
