"""Two small camera helpers: a rotation matrix as a quaternion, and a camera frame that looks at a point.

Used by the frame maker of the depth nets' training data (``colour_frames.random_camera``). Their code is unchanged
from the module they were written in, which is not part of this release.
"""

from __future__ import annotations

import math

import numpy as np


def _quat_wxyz(R: np.ndarray) -> list:
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    return [float(x) for x in (q if q[0] >= 0 else -q)]


def look_at_mujoco(pos: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """MuJoCo camera frame (x right, y up, looking along -z) at ``pos`` looking at ``target``: columns x, y, z."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f = f / np.linalg.norm(f)
    x = np.cross(f, np.asarray(up, float))
    x = x / np.linalg.norm(x)
    z = -f
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)
