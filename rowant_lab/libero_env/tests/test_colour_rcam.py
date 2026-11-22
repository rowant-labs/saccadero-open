"""Item 2b: drawn cameras for the depth net's frames, the colour-only net input, and validation split by camera."""

import math
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rowant_lab.pro2 import colour_frames as CF  # noqa: E402
from rowant_lab.pro2 import colour_net as CN  # noqa: E402


def _stored_like():
    """An OpenCV-frame camera like the stored table cameras: 1.6 m up, looking down and along -x."""
    from rowant_lab.pro2.camera_math import look_at_mujoco

    pos, target = np.array([0.66, 0.0, 1.61]), np.array([-0.22, 0.0, 0.90])
    R = look_at_mujoco(pos, target) @ np.diag([1.0, -1.0, -1.0])  # MuJoCo camera frame to OpenCV
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, pos
    return T


def test_random_camera_is_seeded_and_in_range():
    from rowant_lab.pro2.camera_math import look_at_mujoco  # noqa: F401  (the helper it builds on)

    T = _stored_like()
    o = CF.RANDOM_CAMERA
    a = CF.random_camera(T, random.Random("x|1"))
    b = CF.random_camera(T, random.Random("x|1"))
    assert a == b
    pivot = T[:3, 3] + o["pivot_m"] * T[:3, 2]
    r0 = np.linalg.norm((T[:3, 3] - pivot)[:2])
    for k in range(300):
        c = CF.random_camera(T, random.Random(f"x|{k}"))
        assert o["fovy_deg"][0] <= c["fovy_deg"] <= o["fovy_deg"][1]
        p = np.asarray(c["pos"])
        # height: the turn keeps it, then at most height_m up or down
        assert abs(p[2] - T[2, 3]) <= o["height_m"] + 1e-9
        # horizontal distance to the pivot's axis: the turned radius, moved by at most the shift
        assert abs(np.linalg.norm((p - pivot)[:2]) - r0) <= math.sqrt(2) * o["shift_m"] + 1e-9
        # the optical axis passes the pivot within the aim jitter
        q = np.asarray(c["quat_wxyz"])
        w, x, y, z = q
        Rm = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                       [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                       [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        fwd = -Rm[:, 2]  # MuJoCo cameras look along -z
        v = pivot - p
        miss = np.linalg.norm(v - (v @ fwd) * fwd)
        assert miss <= math.sqrt(3) * o["aim_m"] + 1e-6
        assert abs(Rm[2, 0]) < 1e-9  # no roll: the camera's x axis stays horizontal


def test_frame_names_mark_drawn_frames():
    it = {"split": "train", "suite": "libero_goal", "task": 3, "case": "S01", "state": 4, "chain": [[0] * 7]}
    own, drawn = CF.frame_name(it, 12), CF.frame_name(it, 12, True)
    assert own.endswith("c4_0012.npz") and drawn.endswith("c4_0012x.npz")
    assert not CF.drawn_frame(own) and CF.drawn_frame(drawn)
    assert CF.frame_seed(it, 12) == "libero_goal|3|S01|4|c|12"


def test_first_form_input_unchanged_and_colour_only_input():
    rgb = (np.random.default_rng(1).random((64, 96, 3)) * 255).astype(np.uint8)
    x5 = CN.to_input(rgb, 8, 16, (128, 128))
    assert x5.shape == (5, 64, 96)
    assert np.array_equal(x5[3:], CN.pixel_channels(128, 128, 8, 16, 64, 96))
    x3 = CN.to_input(rgb, 8, 16, (128, 128), pos=False)
    assert x3.shape == (3, 64, 96) and np.array_equal(x3, x5[:3])


def test_colour_only_net_runs_tiled_and_reloads():
    torch.manual_seed(0)
    m = CN.DepthUNet(base=4, stages=3, blocks=1, cap=4, pos=False).eval()
    assert m.first.in_channels == 3 and m.config["pos"] is False
    assert "pos" not in CN.DepthUNet(base=4, stages=3, blocks=1, cap=4).config  # first form: config as before
    rgb = (np.random.default_rng(2).random((128, 128, 3)) * 255).astype(np.uint8)
    d = CN.tiled_depth(m, rgb, torch.device("cpu"), crop=64, stride=32)
    assert d.shape == (128, 128) and np.isfinite(d).all()
    m2 = CN.DepthUNet(**m.config)
    m2.load_state_dict(m.state_dict())
    assert np.allclose(CN.tiled_depth(m2.eval(), rgb, torch.device("cpu"), crop=64, stride=32), d)


def _write_frame(path, drawn: bool, rng):
    H = W = 128
    f = 0.5 * W / math.tan(math.radians(45.0) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])  # looking straight down from 1 m
    T[2, 3] = 1.0
    depth = np.full((H, W), 1.0)
    depth[40:70, 40:70] = 0.9  # a 10 cm block on the floor
    np.savez_compressed(path, rgb=(rng.random((H, W, 3)) * 255).astype(np.uint8),
                        depth=np.rint(depth / 1e-4).astype(np.uint16), K=K, T=T, rc=np.uint8(drawn))


def test_training_frames_and_split_validation(tmp_path):
    from rowant_lab.pro2 import colour_train as CT

    rng = np.random.default_rng(3)
    files = []
    for i, drawn in enumerate((False, False, True, True)):
        p = tmp_path / f"r0_{i:04d}{'x' if drawn else ''}.npz"
        _write_frame(p, drawn, rng)
        files.append(str(p))
    x, y, obj = CT.Frames(files, 64, 0, pos=False)[0]
    assert x.shape == (3, 64, 64) and y.shape == (1, 64, 64)
    x5, _, _ = CT.Frames(files, 64, 0)[0]
    assert x5.shape == (5, 64, 64)
    m = CN.DepthUNet(base=4, stages=3, blocks=1, cap=4, pos=False)
    v = CT.validate(m, files, torch.device("cpu"), n=2, crop=64)
    assert v["frames"] == 2 and v["drawn"]["frames"] == 2
    assert "blob_offset" in v and "blob_offset" in v["drawn"]
    k = CT.selection_key(v)
    assert k == pytest.approx((v["blob_offset"]["p90_mm"] + v["drawn"]["blob_offset"]["p90_mm"]) / 2, abs=1e-3)
    assert CT.selection_key({"blob_offset": {"p90_mm": 5.0}}) == 5.0


def test_ray_channels_follow_the_intrinsics():
    K = np.array([[100.0, 0, 64.0], [0, 100.0, 48.0], [0, 0, 1]])
    r = CN.ray_channels(K, 8, 16, 32, 40)
    assert r.shape == (2, 32, 40) and r.dtype == np.float32
    assert r[0, 0, 0] == pytest.approx((16 + 0.5 - 64.0) / 100.0)
    assert r[1, 0, 0] == pytest.approx((8 + 0.5 - 48.0) / 100.0)
    assert r[0, 5, 39] == pytest.approx((16 + 39 + 0.5 - 64.0) / 100.0)
    rgb = (np.random.default_rng(4).random((32, 40, 3)) * 255).astype(np.uint8)
    x = CN.to_input(rgb, 8, 16, (96, 128), K=K)
    assert x.shape == (5, 32, 40) and np.array_equal(x[3:], r) and np.array_equal(x[:3], CN.to_input(rgb, pos=False))


def test_rays_net_needs_k_and_the_first_form_ignores_it():
    torch.manual_seed(0)
    m = CN.DepthUNet(base=4, stages=3, blocks=1, cap=4, rays=True).eval()
    assert m.config["rays"] is True and m.first.in_channels == 5 and m.kind == "rays"
    rgb = (np.random.default_rng(5).random((128, 128, 3)) * 255).astype(np.uint8)
    K = np.array([[150.0, 0, 64.0], [0, 150.0, 64.0], [0, 0, 1]])
    with pytest.raises(ValueError):
        CN.tiled_depth(m, rgb, torch.device("cpu"), crop=64, stride=32)
    d = CN.tiled_depth(m, rgb, torch.device("cpu"), crop=64, stride=32, K=K)
    d2 = CN.tiled_depth(m, rgb, torch.device("cpu"), crop=64, stride=32, K=K * np.array([[1.1, 1, 1], [1, 1.1, 1], [1, 1, 1]]))
    assert d.shape == (128, 128) and not np.allclose(d, d2)  # the rays reach the net
    first = CN.DepthUNet(base=4, stages=3, blocks=1, cap=4).eval()
    assert first.kind == "pixel"
    a = CN.tiled_depth(first, rgb, torch.device("cpu"), crop=64, stride=32)
    b = CN.tiled_depth(first, rgb, torch.device("cpu"), crop=64, stride=32, K=K)
    assert np.array_equal(a, b)
    m2 = CN.DepthUNet(**m.config)
    assert m2.kind == "rays"


def test_training_frames_with_rays(tmp_path):
    from rowant_lab.pro2 import colour_train as CT

    rng = np.random.default_rng(6)
    p = tmp_path / "r0_0000x.npz"
    _write_frame(p, True, rng)
    x, _y, _o = CT.Frames([str(p)], 64, 0, kind="rays")[0]
    assert x.shape == (5, 64, 64)
    z = np.load(p)
    # the crop's ray channels are the frame's own rays at the crop's offset
    r = x[3:].numpy()
    left = int(round(r[0, 0, 0] * z["K"][0, 0] + z["K"][0, 2] - 0.5))
    top = int(round(r[1, 0, 0] * z["K"][1, 1] + z["K"][1, 2] - 0.5))
    assert np.allclose(r, CN.ray_channels(z["K"], top, left, 64, 64), atol=1e-6)
    m = CN.DepthUNet(base=4, stages=3, blocks=1, cap=4, rays=True)
    v = CT.validate(m, [str(p)] * 2, torch.device("cpu"), n=2, crop=64)
    assert "drawn" in v
