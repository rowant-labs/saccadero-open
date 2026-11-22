"""TRUTH-FREE: one RGB-D frame as the fast layer takes it, and its camera in the screens' ``camera.json`` form.

The screens' functions (``e6/plane_crop``, ``e0c_check_timing.frontend``, ``e0f_refine.refine``) read a camera
dict with ``fx, fy, cx, cy, width, height``, ``depth.unit_m`` (metres per raw depth unit) and
``T_world_from_camera`` (4 x 4). ``Frame.camera()`` builds it from BOP's inputs: ``K`` (no skew), ``depth_scale``
(BOP: raw value x depth_scale = mm, so ``unit_m = depth_scale / 1000``) and an optional camera-to-world transform
(identity by default: the world frame is the camera frame, and a pose is then directly BOP's model-to-camera pose).

Raw depth 0 means no reading (and 65535 is treated as invalid, as in the screens).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TRUTH_FREE = True


@dataclass(frozen=True)
class Frame:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) uint16 raw depth values
    K: np.ndarray  # (3, 3) intrinsics
    depth_scale: float  # mm per raw depth unit (BOP's depth_scale)
    T_world_from_camera: np.ndarray | None = None  # (4, 4); None = identity
    _camera: dict = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be uint8 HxWx3, got {rgb.dtype} {rgb.shape}")
        if depth.dtype != np.uint16 or depth.shape != rgb.shape[:2]:
            raise ValueError(f"depth must be uint16 HxW matching rgb, got {depth.dtype} {depth.shape}")
        K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        if abs(K[0, 1]) > 1e-9 or abs(K[1, 0]) > 1e-9 or abs(K[2, 2] - 1) > 1e-12:
            raise ValueError("K must be a pinhole matrix without skew")
        if not (float(self.depth_scale) > 0):
            raise ValueError("depth_scale must be positive")
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "_camera", camera_dict(K, self.depth_scale, rgb.shape[1], rgb.shape[0],
                                                        self.T_world_from_camera))

    @classmethod
    def from_camera_json(cls, rgb, depth, camera: dict) -> "Frame":
        """A frame from a screens ``camera.json`` dict (the E6 renders: 0.1 mm depth units, a world frame)."""
        K = np.array([[camera["fx"], 0.0, camera["cx"]], [0.0, camera["fy"], camera["cy"]], [0.0, 0.0, 1.0]])
        f = cls(rgb=rgb, depth=depth, K=K, depth_scale=float(camera["depth"]["unit_m"]) * 1000.0,
                T_world_from_camera=np.asarray(camera["T_world_from_camera"], float))
        object.__setattr__(f, "_camera", camera)  # the exact dict the screens read (bit-equal parity)
        return f

    def camera(self) -> dict:
        return self._camera

    @property
    def T(self) -> np.ndarray:
        return np.asarray(self._camera["T_world_from_camera"], float)

    def valid_depth(self) -> np.ndarray:
        return valid_depth(self.depth)

    def to_camera(self, R_world: np.ndarray, t_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A world-frame pose (model -> world) as a camera-frame pose (model -> camera), t in metres."""
        T = self.T
        Rcw = T[:3, :3].T
        return Rcw @ np.asarray(R_world, float), Rcw @ (np.asarray(t_world, float) - T[:3, 3])


def camera_dict(K, depth_scale: float, width: int, height: int, T_world_from_camera=None) -> dict:
    """The screens' camera dict of a frame (``Frame.camera()``), from its intrinsics, depth scale and size."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    T = np.eye(4) if T_world_from_camera is None else np.asarray(T_world_from_camera, float)
    return {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "width": int(width), "height": int(height), "depth": {"unit_m": float(depth_scale) / 1000.0},
            "T_world_from_camera": T.tolist()}


def valid_depth(depth: np.ndarray) -> np.ndarray:
    """Pixels with a depth reading (``Frame.valid_depth``)."""
    d = np.asarray(depth)
    return (d != 0) & (d != np.iinfo(np.uint16).max)


def depth_from_mm(depth_mm: np.ndarray, depth_scale: float) -> np.ndarray:
    """Raw uint16 depth from a float depth in mm (BOP reader output) and the image's depth_scale."""
    raw = np.rint(np.asarray(depth_mm, dtype=np.float64) / float(depth_scale))
    if raw.min() < 0 or raw.max() > np.iinfo(np.uint16).max:
        raise ValueError("depth outside the uint16 range")
    back = raw * float(depth_scale)
    if np.max(np.abs(back - np.asarray(depth_mm, dtype=np.float64))) > 1e-3:
        raise ValueError("depth is not an integer multiple of depth_scale")
    return raw.astype(np.uint16)
