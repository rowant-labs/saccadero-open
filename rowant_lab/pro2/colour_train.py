"""Train the colour-only depth net on the frames of ``colour_frames.py`` (GPU pod; a venv with CUDA torch and numpy).

    python rowant_lab/pro2/colour_train.py --frames DIR/frames --out DIR/net --hours 3 [--workers 10]

Trains from scratch. Training frames: base tasks other than 7 and 9. Validation: the judge tasks' frames (7 and 9),
scored at the full 1024 px: table pixels, pixels on objects, and each object blob's own error. Saves ``last.pt`` and
``best.pt`` (lowest blob offset error) and appends to ``log.jsonl``.

``--rays`` (item 2b): beside the colour image, each pixel's ray direction from the frame's own intrinsics
(``DepthUNet(rays=True)``). ``--no-pos``: the colour image alone (tried first). When the frames include drawn
cameras (``colour_frames.py plan --random-camera``), validation scores the scene's own cameras at the top level, as
before, and the drawn ones under ``drawn``; ``best.pt`` then goes by the mean of the two blob-offset p90s.

One net for more scenes (2026-11-15, the long suite's tables; all off unless asked for). A frame's KEY is its folder
under ``train/`` or ``val/``: a suite, or a long-suite table.

* ``--init NET``: start from another net's weights (same form) with a fresh optimizer and this run's own schedule:
  continuing the first net on old and new frames together. ``--resume`` (this run's own ``last.pt`` after a restart)
  wins over it.
* ``--warmup-steps N``: the learning rate rises linearly over the first N steps of a run that starts from ``--init``
  or from scratch (a fresh optimizer's first steps would otherwise jolt the loaded weights).
* ``--balance``: every epoch draws the same number of crops from each key, so a table with few frames is learnt as
  fast as a suite with many (a key with fewer frames than its share is repeated, each time with its own crop).
* ``--by-key``: validation per key (``--val-frames`` of each); the log's ``val`` is ``{"by_key", "worst"}`` and
  ``best.pt`` goes by the worst key's blob-offset p90.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.pro2.colour_frames import drawn_frame  # noqa: E402
from rowant_lab.pro2.colour_net import TILE_CROP, DepthUNet, input_kind, pixel_channels, ray_channels, sha256_file, tiled_depth  # noqa: E402

SCALE = 1e-4  # frames store depth in 0.1 mm


def above_table(depth_m: np.ndarray, K: np.ndarray, T: np.ndarray, margin_m: float = 0.008) -> np.ndarray:
    """Pixels well above the most common world height of the near scene (the table or floor): objects and the arm."""
    H, W = depth_m.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth_m
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    P = np.stack([x, y, z], -1)
    w = (P @ T[:3, :3].T)[..., 2] + T[2, 3]
    near = (z > 0.1) & (z < 2.2)
    hist, edges = np.histogram(w[near], bins=np.arange(-0.2, 1.6, 0.002))
    table = edges[int(hist.argmax())] + 0.001
    return (w > table + margin_m) & (z > 0)


def photometric(x: np.ndarray, rng: random.Random) -> np.ndarray:
    """x: float32 (H, W, 3) in [0, 255]. Image noise, darkening and exposure changes of the kinds Pro2.0 applies."""
    r = rng.random()
    if r < 0.08:
        g = np.random.default_rng(rng.randrange(1 << 30)).normal(0.0, rng.uniform(6.0, 14.0), x.shape)
        x = x * rng.uniform(0.75, 1.0) + g
    elif r < 0.2:
        x = x * rng.uniform(0.4, 1.0)
    elif r < 0.5:
        x = x * rng.uniform(0.85, 1.15)
    return np.clip(x, 0, 255)


class Frames(Dataset):
    def __init__(self, files: list, crop: int, seed: int, pos: bool = True, kind: str | None = None):
        self.files, self.crop, self.seed = files, crop, seed
        self.kind = kind or ("pixel" if pos else "colour")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        for _try in range(5):
            try:
                z = np.load(self.files[i])
                rgb, dep = z["rgb"], z["depth"]
                K, T = z["K"], z["T"]
                break
            except Exception:  # a truncated file: take another frame
                i = (i + 17) % len(self.files)
        H, W = dep.shape
        rng = random.Random((self.seed, i, torch.initial_seed()).__hash__())
        d = dep.astype(np.float32) * SCALE
        obj = above_table(d, K, T)
        top, left = rng.randrange(0, H - self.crop + 1), rng.randrange(0, W - self.crop + 1)
        rgb = rgb[top: top + self.crop, left: left + self.crop].astype(np.float32)
        d = d[top: top + self.crop, left: left + self.crop]
        obj = obj[top: top + self.crop, left: left + self.crop]
        rgb = photometric(rgb, rng)
        x = np.moveaxis(rgb, -1, 0) / 255.0 - 0.5
        if self.kind == "rays":
            x = np.concatenate([x, ray_channels(K, top, left, self.crop, self.crop)], 0)
        elif self.kind == "pixel":
            x = np.concatenate([x, pixel_channels(H, W, top, left, self.crop, self.crop)], 0)
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(d)[None], torch.from_numpy(obj.astype(np.float32))[None]


def loss_fn(pred, target, obj, obj_weight: float = 6.0):
    valid = (target > 0).float()
    w = valid * (1.0 + (obj_weight - 1.0) * obj)
    l1 = (torch.abs(pred - target) * w).sum() / w.sum().clamp(min=1.0)
    g = 0.0
    p, t, v = pred, target, valid
    for s in range(4):
        dpx, dtx = p[..., 1:] - p[..., :-1], t[..., 1:] - t[..., :-1]
        dpy, dty = p[..., 1:, :] - p[..., :-1, :], t[..., 1:, :] - t[..., :-1, :]
        mx, my = v[..., 1:] * v[..., :-1], v[..., 1:, :] * v[..., :-1, :]
        g = g + (torch.abs(dpx - dtx) * mx).sum() / mx.sum().clamp(min=1.0)
        g = g + (torch.abs(dpy - dty) * my).sum() / my.sum().clamp(min=1.0)
        p, t, v = F.avg_pool2d(p, 2), F.avg_pool2d(t, 2), F.max_pool2d(v, 2)
    return l1 + 0.5 * g, l1


@torch.no_grad()
def validate(model, files: list, device, n: int = 80, crop: int = TILE_CROP) -> dict:
    """Depth errors on the held-back tasks' frames at 1024 px, run the way the robot runs the net (tiles at the crop size).
    Frames from the scene's own cameras at the top level (as before); frames from drawn cameras, if any, under ``drawn``
    (``n`` of each)."""
    own = [f for f in files if not drawn_frame(f)]
    drawn = [f for f in files if drawn_frame(f)]
    model.eval()
    out = _scores(model, own, device, n, crop)
    if drawn:
        out["drawn"] = _scores(model, drawn, device, n, crop)
    model.train()
    return out


def _scores(model, files: list, device, n: int, crop: int) -> dict:
    from scipy import ndimage

    rows = {"table": [], "obj": [], "blob_mae": [], "blob_offset": []}
    picks = files[:: max(1, len(files) // n)][:n]
    for fp in picks:
        try:
            z = np.load(fp)
            rgb, dep = z["rgb"], z["depth"]
            K, T = z["K"], z["T"]
        except Exception:  # a truncated file
            continue
        d = dep.astype(np.float32) * SCALE
        p = tiled_depth(model, rgb, device, crop=crop, K=K)
        obj = above_table(d, K, T)
        table = ~obj & (d > 0)
        e = p - d
        rows["table"].append(np.abs(e[table])[::37])
        if obj.sum():
            rows["obj"].append(np.abs(e[obj])[::7])
        lab, nb = ndimage.label(ndimage.binary_opening(obj, iterations=2))
        for b in range(1, nb + 1):
            m = lab == b
            if 400 <= m.sum() <= 60000:  # object-sized blobs (the arm is larger)
                rows["blob_mae"].append(float(np.abs(e[m]).mean()))
                rows["blob_offset"].append(float(np.abs(e[m].mean())))
    out = {}
    for k, v in rows.items():
        a = (np.concatenate(v) if v else np.zeros(0)) if k in ("table", "obj") else np.asarray(v)
        if a.size:
            out[k] = {"median_mm": round(float(np.median(a)) * 1e3, 2), "p90_mm": round(float(np.percentile(a, 90)) * 1e3, 2),
                      "n": int(a.size)}
    if "obj" in out:
        a = np.concatenate(rows["obj"])
        out["obj"]["within_10mm"] = round(float(np.mean(a <= 0.01)), 3)
        out["obj"]["within_5mm"] = round(float(np.mean(a <= 0.005)), 3)
    out["frames"] = len(picks)
    return out


def selection_key(v: dict):
    """The blob-offset p90 (mm) that picks ``best.pt``: the own cameras', or with drawn frames the mean of both."""
    k = (v.get("blob_offset") or {}).get("p90_mm")
    d = ((v.get("drawn") or {}).get("blob_offset") or {}).get("p90_mm")
    if k is None or d is None:
        return k
    return round((float(k) + float(d)) / 2.0, 3)


def save_atomic(ck: dict, path: Path) -> None:
    """A restart during the write must not cost the previous checkpoint."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ck, tmp)
    os.replace(tmp, path)


def key_of(path, root) -> str:
    """A frame's key: its first folder under ``root`` (``frames/train`` or ``frames/val``): a suite or a table."""
    return Path(path).relative_to(root).parts[0]


def group_by_key(files: list, root) -> dict:
    """``{key: [file, ...]}`` in the files' order."""
    out: dict = {}
    for f in files:
        out.setdefault(key_of(f, root), []).append(f)
    return out


def balanced_sample(groups: dict, n: int, rng: random.Random) -> list:
    """``n`` files, the same number from every key (the first keys get the remainder). A key with fewer files than its
    share is repeated: whole passes, then a random part. Each occurrence gets its own random crop (``Frames``)."""
    keys = sorted(k for k in groups if groups[k])
    if not keys:
        return []
    per, extra = divmod(int(n), len(keys))
    out: list = []
    for i, k in enumerate(keys):
        want, f = per + (1 if i < extra else 0), groups[k]
        out += f * (want // len(f)) + rng.sample(f, want % len(f))
    return out


def validate_by_key(model, groups: dict, device, n: int = 80, crop: int = TILE_CROP) -> dict:
    """``validate`` for each key's files (``n`` of each)."""
    return {k: validate(model, groups[k], device, n, crop) for k in sorted(groups)}


def worst_key(by_key: dict):
    """``(key, value)``: the key whose selection value (``selection_key``: blob-offset p90, mm) is largest; a key with
    no object blob in its frames has no value and is passed over. ``(None, None)`` when no key has one."""
    vals = {k: selection_key(v) for k, v in by_key.items()}
    vals = {k: v for k, v in vals.items() if v is not None}
    if not vals:
        return None, None
    k = max(sorted(vals), key=lambda x: vals[x])
    return k, vals[k]


def warmup_factor(step: int, warmup_steps: int) -> float:
    """The learning rate's factor at optimizer step ``step`` (from 0) of a run with a linear warm-up."""
    return 1.0 if warmup_steps <= 0 or step >= warmup_steps else (step + 1) / float(warmup_steps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--crop", type=int, default=640)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--steps-per-epoch", type=int, default=1500)
    ap.add_argument("--val-frames", type=int, default=80)
    ap.add_argument("--save-every-s", type=float, default=300.0, help="time between checkpoints inside an epoch")
    ap.add_argument("--resume", type=Path, help="a last.pt: continue it (the planned hours count time already trained)")
    ap.add_argument("--no-pos", action="store_true", help="the colour image alone, no pixel-position channels (item 2b)")
    ap.add_argument("--rays", action="store_true", help="ray directions from each frame's K instead of pixel position (item 2b)")
    ap.add_argument("--init", type=Path, help="start from this net's weights (same form); fresh optimizer and schedule")
    ap.add_argument("--warmup-steps", type=int, default=0, help="linear learning-rate warm-up of a run that is not resumed")
    ap.add_argument("--balance", action="store_true", help="every epoch draws the same number of crops from each key")
    ap.add_argument("--by-key", action="store_true", help="validate each key apart (--val-frames of each); best.pt by the worst")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    train = sorted(str(p) for p in (a.frames / "train").rglob("*.npz"))
    val = sorted(str(p) for p in (a.frames / "val").rglob("*.npz"))
    if not train or not val:
        print(json.dumps({"train": len(train), "val": len(val)}))
        return 2
    print(json.dumps({"train_frames": len(train), "val_frames": len(val)}), flush=True)
    groups = group_by_key(train, a.frames / "train") if a.balance else None
    val_groups = group_by_key(val, a.frames / "val") if a.by_key else None
    if a.balance or a.by_key:
        print(json.dumps({"train_by_key": {k: len(v) for k, v in sorted(group_by_key(train, a.frames / "train").items())},
                          "val_by_key": {k: len(v) for k, v in sorted(group_by_key(val, a.frames / "val").items())}}),
              flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthUNet(base=a.base, pos=not a.no_pos, rays=a.rays).to(device)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, start_epoch, done_s = None, 0, 0.0
    init = None  # --init: the net this run started from (kept in its checkpoints)
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["state"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        best = ck.get("best")
        done_s = float(ck.get("elapsed_s", 0.0))
        init = ck.get("init")
        print(json.dumps({"resumed": str(a.resume), "epoch": start_epoch, "done_h": round(done_s / 3600, 2), "best": best}),
              flush=True)
    elif a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=True)
        if dict(ck["config"]) != dict(model.config):
            print(json.dumps({"error": "the --init net has another form", "init": ck["config"], "this": model.config}))
            return 2
        model.load_state_dict(ck["state"])
        init = {"path": str(a.init), "sha256": sha256_file(a.init), "epoch": ck.get("epoch")}
        print(json.dumps({"init": init}), flush=True)
    warm = int(a.warmup_steps) if not a.resume else 0  # a resumed run's optimizer is already running
    gstep = 0
    t_start = time.time() - done_s  # the planned hours and the learning-rate curve span restarts
    t_end = t_start + a.hours * 3600.0
    epoch = start_epoch
    log = open(a.out / "log.jsonl", "a", buffering=1)

    def checkpoint(at_epoch: int, scores: dict | None) -> dict:
        return {"config": model.config, "state": {k: p.cpu() for k, p in model.state_dict().items()}, "epoch": at_epoch,
                "val": scores, "best": best, "elapsed_s": time.time() - t_start, "crop": a.crop,
                **({"init": init} if init else {})}

    def save_last(at_epoch: int, scores: dict | None) -> None:
        save_atomic({**checkpoint(at_epoch, scores), "opt": opt.state_dict()}, a.out / "last.pt")

    last_save = time.time()
    while time.time() < t_end:
        # one epoch of ``steps-per-epoch`` random crops, learning rate on a cosine over the planned time
        if groups is not None:  # --balance: the same number of crops from every key
            files = balanced_sample(groups, a.steps_per_epoch * a.batch, random.Random(epoch))
        else:
            files = random.Random(epoch).sample(train, min(len(train), a.steps_per_epoch * a.batch))
        dl = DataLoader(Frames(files, a.crop, epoch, kind=input_kind(model.config)), batch_size=a.batch, shuffle=True,
                        num_workers=a.workers,
                        pin_memory=device.type == "cuda", drop_last=True, persistent_workers=False)
        model.train()
        tot, n, t0 = 0.0, 0, time.time()
        for x, y, obj in dl:
            frac = min(1.0, (time.time() - t_start) / (a.hours * 3600.0))
            lr = a.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * frac)))
            if warm:
                lr *= warmup_factor(gstep, warm)
                gstep += 1
            for pg in opt.param_groups:
                pg["lr"] = lr
            x, y, obj = x.to(device, non_blocking=True), y.to(device, non_blocking=True), obj.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                pred = model(x)
                loss, l1 = loss_fn(pred.float(), y, obj)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            tot += float(l1)
            n += 1
            if time.time() - last_save >= a.save_every_s:
                save_last(epoch, None)
                last_save = time.time()
        if val_groups is not None:  # --by-key: each suite and table apart; best.pt by the worst of them
            by = validate_by_key(model, val_groups, device, a.val_frames, a.crop)
            wk, key = worst_key(by)
            v = {"by_key": by, "worst": {"key": wk, "blob_offset_p90_mm": key}}
        else:
            v = validate(model, val, device, a.val_frames, a.crop)
            key = selection_key(v)
        row = {"epoch": epoch, "lr": round(lr, 6), "train_l1_mm": round(1e3 * tot / max(n, 1), 2), "steps": n,
               "epoch_s": round(time.time() - t0), "elapsed_h": round((time.time() - t_start) / 3600, 2), "val": v}
        log.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if key is not None and (best is None or key < best):
            best = key
            save_atomic(checkpoint(epoch, v), a.out / "best.pt")
        save_last(epoch, v)
        last_save = time.time()
        epoch += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
