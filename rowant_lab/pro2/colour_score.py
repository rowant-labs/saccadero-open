"""Score a trained depth net on the held-back tasks' frames, run the way the robot runs it (GPU pod).

    python rowant_lab/pro2/colour_score.py --frames DIR/frames --net DIR/net/last.pt --out DIR/net/score.json

The pod driver's gate reads the result before it lets a net into the perception server.

``--by-key`` (2026-11-15, the long suite's tables; off unless asked for): one score per KEY, the frames' folder under
``val/`` (a suite, or a long-suite table), ``--val-frames`` of each, under ``by_key`` instead of ``val``. A suite's
validation frames are its tasks 7 and 9. A table's are its held-back task (7 or 9) and, from the start state kept out
of training, the plain version of its test task (the study table has only that). A key whose validation holds more
than one task also gets each task's own score (``by_task``, ``--val-frames`` of each): a long-suite task is its own
scene. ``colour_long.py accept`` reads two such files (the new net's and the current net's, on the same frames).
``--skip-if-current`` leaves a score file alone when it is already this net's with these options (the long driver's
resume).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.pro2.colour_net import DepthNet, sha256_file  # noqa: E402
from rowant_lab.pro2.colour_train import group_by_key, validate, validate_by_key  # noqa: E402


def is_current(out: Path, net_sha256: str, val_frames: int, by_key: bool) -> bool:
    """The score file at ``out`` is this net's, with these options."""
    try:
        row = json.loads(Path(out).read_text())
    except (OSError, ValueError):
        return False
    return (row.get("net_sha256") == net_sha256 and row.get("val_frames") == int(val_frames)
            and ("by_key" in row) == bool(by_key))


def task_of(path, root) -> int:
    """A frame's base task: the folder under its key."""
    return int(Path(path).relative_to(root).parts[1])


def tasks_by_key(groups: dict, root) -> dict:
    """``{key: [task, ...]}``: the base tasks whose frames a key's validation holds (the folder under the key)."""
    return {k: sorted({task_of(f, root) for f in files}) for k, files in groups.items()}


def score_by_key(model, val: list, root, device, n: int, crop: int) -> dict:
    """``{key: scores}`` with the key's tasks and file count, and ``by_task`` where the key holds several tasks."""
    groups = group_by_key(val, root)
    tasks = tasks_by_key(groups, root)
    by = validate_by_key(model, groups, device, n, crop)
    out = {}
    for k, v in by.items():
        out[k] = {**v, "tasks": tasks[k], "files": len(groups[k])}
        if len(tasks[k]) > 1:
            out[k]["by_task"] = {str(t): validate(model, [f for f in groups[k] if task_of(f, root) == t], device, n, crop)
                                 for t in tasks[k]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--net", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-frames", type=int, default=80)
    ap.add_argument("--by-key", action="store_true", help="one score per suite or table (the folder under val/)")
    ap.add_argument("--skip-if-current", action="store_true", help="do nothing when --out already scores this net this way")
    a = ap.parse_args()
    if a.skip_if_current and is_current(a.out, sha256_file(a.net), a.val_frames, a.by_key):
        print(json.dumps({"current": str(a.out)}), flush=True)
        return 0
    val = sorted(str(p) for p in (a.frames / "val").rglob("*.npz"))
    if not val:
        print(json.dumps({"val": 0}))
        return 2
    net = DepthNet(a.net, device="cuda" if torch.cuda.is_available() else "cpu")
    row = {"net": str(a.net), "net_sha256": net.sha256, "crop": net.crop, "val_frames": a.val_frames}
    if a.by_key:
        row["by_key"] = score_by_key(net.model, val, a.frames / "val", net.device, a.val_frames, net.crop)
    else:
        row["val"] = validate(net.model, val, net.device, a.val_frames, net.crop)
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(row, indent=1) + "\n")
    os.replace(tmp, a.out)
    print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
