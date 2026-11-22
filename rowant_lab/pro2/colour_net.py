"""Depth from one colour image, for the colour-only line: a small U-Net trained only on our own renders.

Trained from scratch (no pretrained weights) on frames of Pro2.0's training tasks rendered with the simulator
(``colour_frames.py``, ``colour_train.py``). Input: the 1024 px colour image plus two channels with the normalized
pixel position, so a fixed camera's geometry is learnable. Output: depth along the optical axis, in metres.

Item 2b (2026-11-13), for frames from many cameras: a net built with ``rays=True`` (``config["rays"]``) takes, beside the
colour image, each pixel's ray direction from the camera's intrinsics K, ``((u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy)``:
the lens and where the pixel sits in the image, but not the camera's pose, so its depth must follow the image. The
caller gives K (``DepthNet.depth_m(rgb, K)``). A net built with ``pos=False`` takes the colour image alone (tried first;
without any position it cannot tell a pixel's ray, so it stays centimetres off). Checkpoints with neither key are the first
form (pixel position), and they run exactly as before.

This module holds the model and the inference path only (torch and numpy). The robot's camera tool loads it
(``colour_line.py``). The trainer imports it too.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

DEPTH_MID_M, DEPTH_SPAN_M = 1.3, 1.0  # the net's output 0 is 1.3 m; 1 unit is 1 m
DEPTH_SCALE_MM = 0.1  # the depth encoding the perception server reads
IN_CHANNELS = 5  # RGB and two position channels: pixel position (first form) or ray direction (``rays``); 3 with ``pos`` False
TILE_CROP, TILE_STRIDE, TILE_EDGE = 640, 384, 64  # four tiles cover 1024 px; see ``tiled_depth``


def groups(ch: int) -> int:
    return min(8, ch // 4)


class Block(nn.Module):
    def __init__(self, a: int, b: int):
        super().__init__()
        self.n1 = nn.GroupNorm(groups(a), a)
        self.c1 = nn.Conv2d(a, b, 3, padding=1)
        self.n2 = nn.GroupNorm(groups(b), b)
        self.c2 = nn.Conv2d(b, b, 3, padding=1)
        self.skip = nn.Identity() if a == b else nn.Conv2d(a, b, 1)

    def forward(self, x):
        y = self.c1(F.silu(self.n1(x)))
        y = self.c2(F.silu(self.n2(y)))
        return y + self.skip(x)


class DepthUNet(nn.Module):
    def __init__(self, base: int = 16, stages: int = 6, blocks: int = 2, cap: int = 16, pos: bool = True,
                 rays: bool = False):
        super().__init__()
        ch = [base * min(2 ** k, cap) for k in range(stages)]
        self.config = {"base": base, "stages": stages, "blocks": blocks, "cap": cap}
        if rays:
            self.config["rays"] = True
        elif not pos:
            self.config["pos"] = False  # absent for the first form, whose checkpoints stay byte-identical in config
        self.first = nn.Conv2d(IN_CHANNELS if (pos or rays) else 3, ch[0], 3, padding=1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for k in range(stages):
            self.enc.append(nn.Sequential(*[Block(ch[k], ch[k]) for _ in range(blocks)]))
            if k < stages - 1:
                self.down.append(nn.Conv2d(ch[k], ch[k + 1], 3, stride=2, padding=1))
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for k in range(stages - 2, -1, -1):
            self.up.append(nn.Conv2d(ch[k + 1], ch[k], 1))
            self.dec.append(nn.Sequential(Block(2 * ch[k], ch[k]), *[Block(ch[k], ch[k]) for _ in range(blocks - 1)]))
        self.out_n = nn.GroupNorm(groups(ch[0]), ch[0])
        self.out = nn.Conv2d(ch[0], 1, 3, padding=1)

    @property
    def kind(self) -> str:
        return input_kind(self.config)

    def forward(self, x):
        """``x``: (B, 5, H, W), or (B, 3, H, W) with ``pos`` False. Returns depth in metres, (B, 1, H, W). H and W are
        multiples of 32."""
        h = self.first(x)
        skips = []
        for k, e in enumerate(self.enc):
            h = e(h)
            if k < len(self.down):
                skips.append(h)
                h = self.down[k](h)
        for u, d, s in zip(self.up, self.dec, reversed(skips)):
            h = u(F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False))
            h = d(torch.cat([h, s], 1))
        return DEPTH_MID_M + DEPTH_SPAN_M * self.out(F.silu(self.out_n(h)))


def pixel_channels(full_h: int, full_w: int, top: int = 0, left: int = 0, h: int | None = None,
                   w: int | None = None) -> np.ndarray:
    """(2, h, w) float32: the normalized position (-1 to 1) of each pixel in the full frame, optionally a crop of it."""
    h = full_h if h is None else h
    w = full_w if w is None else w
    ys = (2.0 * (np.arange(top, top + h, dtype=np.float32) + 0.5) / full_h) - 1.0
    xs = (2.0 * (np.arange(left, left + w, dtype=np.float32) + 0.5) / full_w) - 1.0
    return np.stack([np.broadcast_to(xs[None, :], (h, w)), np.broadcast_to(ys[:, None], (h, w))], 0)


def input_kind(config: dict) -> str:
    """``"pixel"`` (the first form), ``"rays"`` or ``"colour"``."""
    if config.get("rays"):
        return "rays"
    return "pixel" if config.get("pos", True) else "colour"


def ray_channels(K: np.ndarray, top: int, left: int, h: int, w: int) -> np.ndarray:
    """(2, h, w) float32: each pixel's ray direction (x / z, y / z) at intrinsics ``K``, for the crop at (top, left)."""
    K = np.asarray(K, dtype=np.float64)
    xs = ((np.arange(left, left + w, dtype=np.float64) + 0.5 - K[0, 2]) / K[0, 0]).astype(np.float32)
    ys = ((np.arange(top, top + h, dtype=np.float64) + 0.5 - K[1, 2]) / K[1, 1]).astype(np.float32)
    return np.stack([np.broadcast_to(xs[None, :], (h, w)), np.broadcast_to(ys[:, None], (h, w))], 0)


def to_input(rgb: np.ndarray, top: int = 0, left: int = 0, full: tuple | None = None, pos: bool = True,
             K: np.ndarray | None = None) -> np.ndarray:
    """(5, H, W) float32 from a uint8 (H, W, 3) image or crop; ``full`` is the (H, W) of the frame it was cut from.
    With ``K`` (a rays net), the two extra channels are the ray directions instead of the pixel position. With ``pos``
    False and no ``K``, (3, H, W): the colour channels alone."""
    H, W = rgb.shape[:2]
    fh, fw = (H, W) if full is None else full
    x = (np.moveaxis(rgb, -1, 0).astype(np.float32) / 255.0) - 0.5
    if K is not None:
        return np.concatenate([x, ray_channels(K, top, left, H, W)], 0)
    if not pos:
        return x
    return np.concatenate([x, pixel_channels(fh, fw, top, left, H, W)], 0)


def tile_weight(crop: int = TILE_CROP, edge: int = TILE_EDGE) -> np.ndarray:
    """(crop, crop) float32 blending weight: 1 in the middle, falling towards the border of the tile."""
    r = np.clip(np.minimum(np.arange(crop) + 0.5, crop - np.arange(crop) - 0.5) / edge, 0.02, 1.0)
    return np.minimum.outer(r, r).astype(np.float32)


def tile_starts(size: int, crop: int = TILE_CROP, stride: int = TILE_STRIDE) -> list:
    return [0] if size <= crop else list(range(0, size - crop, stride)) + [size - crop]


@torch.no_grad()
def tiled_depth(model: nn.Module, rgb: np.ndarray, device, crop: int = TILE_CROP, stride: int = TILE_STRIDE,
                batch: int = 4, K: np.ndarray | None = None) -> np.ndarray:
    """(H, W) float32 metres for a whole uint8 image, from overlapping tiles of the size the net was trained on.

    The net's GroupNorm layers normalize over what is in view, so a larger image shifts every answer. On full 1024 px
    frames a net trained on 640 px crops was 50 to 80 mm off on flat table; with tiles it is under 1 mm there.
    """
    H, W = rgb.shape[:2]
    kind = input_kind(getattr(model, "config", {}))
    pos = kind != "colour"
    if kind == "rays":
        if K is None:
            raise ValueError("a rays net needs the camera's intrinsics K")
    else:
        K = None  # the other forms take no K
    if min(H, W) < crop:  # smaller than a training crop: one pass (H and W must be multiples of 32)
        return model(torch.from_numpy(to_input(rgb, pos=pos, K=K)[None]).to(device))[0, 0].cpu().numpy()
    tiles = [(t, l) for t in tile_starts(H, crop, stride) for l in tile_starts(W, crop, stride)]
    weight = torch.from_numpy(tile_weight(crop)).to(device)
    num, den = torch.zeros((H, W), device=device), torch.zeros((H, W), device=device)
    for i in range(0, len(tiles), batch):
        part = tiles[i:i + batch]
        x = np.stack([to_input(rgb[t:t + crop, l:l + crop], t, l, (H, W), pos=pos, K=K) for t, l in part])
        p = model(torch.from_numpy(x).to(device))[:, 0]
        for j, (t, l) in enumerate(part):
            num[t:t + crop, l:l + crop] += p[j] * weight
            den[t:t + crop, l:l + crop] += weight
    return (num / den).cpu().numpy()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class DepthNet:
    """Loaded weights and one forward pass. Deterministic on the CPU for a fixed thread count."""

    def __init__(self, path, threads: int = 2, device: str = "cpu"):
        self.path = Path(path)
        self.sha256 = sha256_file(self.path)
        ck = torch.load(self.path, map_location="cpu", weights_only=True)
        self.model = DepthUNet(**ck["config"])
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        self.crop = int(ck.get("crop", TILE_CROP))
        self.device = torch.device(device)
        self.model.to(self.device)
        if self.device.type == "cpu":
            torch.set_num_threads(int(threads))

    def depth_m(self, rgb: np.ndarray, K: np.ndarray | None = None) -> np.ndarray:
        """(H, W) float32 metres from a uint8 (H, W, 3) image in OpenCV row order (row 0 = top). ``K``: the camera's
        intrinsics at this image's size, needed by a rays net and ignored by the others."""
        return tiled_depth(self.model, rgb, self.device, crop=self.crop, K=K)

    def depth_u16(self, rgb: np.ndarray, K: np.ndarray | None = None) -> np.ndarray:
        return np.clip(np.rint(self.depth_m(rgb, K) / (DEPTH_SCALE_MM * 1e-3)), 1, 65535).astype(np.uint16)
