"""Candidate v10: candidate v9 with the unknown-object learner and the objects-in-the-way options.

    python3 -m rowant_lab.pro2.candidate_v10            # the parameters, as the confirming pair ran them
    python3 -m rowant_lab.pro2.candidate_v10 --check    # their sha256 against the frozen one

**What it is.** v9 (``long_dev.bundle`` on the colour configuration; what the registered run of 2026-11-17 scored)
and two additions, each of keys v9 does not name:

1. the learner in its registered form (the second world's registration, W2-LEARN-E; both keep rules passed
   2026-11-19): a thing with no model is looked at from above and from the wrist, learned in the episode and
   handled from what was learned; a thing the look takes for one of the robot's own types goes to the robot's own
   plan;
2. the objects-in-the-way options without the fixed mask (arm 4 of a pod run, 2026-11-19): a look for the way
   when the carry begins, one look at a stall an attempt, the empty hand's way, the detour, and the front camera
   read from the still scene (``cam_register``, with its two guards).

**How it was confirmed**: against v9 on one RTX 4090 pod, 2026-11-19, by five lines fixed
before the run. The 476 practice episodes 411 -> 415; the 73 slab episodes 43 -> 55; the camera-move type 14 -> 16
of 16; the second world's development part 43 -> 80 of 100, its new objects 11 -> 45 of 60; no row with an error.
All of it is practice cases and the second world. It has not run on the benchmark's test tasks.

**What is frozen.** ``run_params()`` is, byte for byte, the text those runs were given (``PARAMS_SHA256``; the
runs' ``params.json``). v9's own bundle is not touched: this module only adds keys, and refuses to replace one.

One thing to know before changing anything here: the learner's server key ``self_front_m`` also acts on the looks
for the way (they leave the robot out in the picture, and the key leaves in what reads more than 5 cm nearer than
the robot's model). Against the same options without it, on two pods of one kind, that cost two practice episodes
and won four slab episodes. It is part of what was confirmed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rowant_lab.pro2 import long_dev as LD  # noqa: E402

NET_POD = "/workspace/out/colour-long/net/last.pt"  # (``scored.NET_POD``: the retrained depth net on a pod)
PARAMS_SHA256 = "f20dd85d24bb2065b5a7cf0117abac7edac9a7f1543fb61b167beca523c82fd6"  # of ``run_params()``

# -- 1. the learner, as registered (W2-LEARN-E). Robot side:
LEARN_TOP = {
    "learn_in_episode": True,  # the learner, in the episode itself
    "learn_looks": "fast",  # the looks are read sooner (``learn_skill.FAST``)
    "learn_top_first": {},  # the look from above first (the server's part is below)
    "learn_wrist_stereo": {"dt_lag_ms": 0.0},  # the wrist camera's depth from the hand's own motion
    "learn_rank_known": {"margin": 0.1},
    "learn_keep": {"times": 1},  # a plan made again after a caught arm goes on with the learned thing, once
    # the learned thing gets one attempt; when that ends with at least 71 steps left, the robot's own plan follows
    "learn_way_back": {"attempts": 1, "min_left": 71},
    "learn_late": {"min_left": 209},  # with fewer steps left the learner does not start
    # the look asks the known types (``learn_skill.ASK``). The third form's keys: a thing read as one of the robot's
    # own types, or near one by either set of floors, goes to the robot's own plan; acting from above on such a
    # reading needs ``own_top_min``
    "learn_ask_known": {"max_things": 3, "max_types": 8, "move_max_m": 0.03, "near_m": 0.15, "next_need": 91,
                        "rejected": False, "own": True,
                        "own_near": {"any": [{"inlier": 0.93, "seen": 0.45, "cover": 0.85},
                                             {"inlier": 0.93, "seen": 0.65, "cover": 0.6}]},
                        "own_top_min": {"inlier": 0.93, "seen": 0.78, "cover": 0.95}},
    # when the learner has gone on to the robot's own plan and that plan has ended, the arm's camera check gets
    # its cue (without it the check never ran once the learner had started)
    "learn_calib_cue": True,
}
# Server side:
LEARN_SERVER = {
    "learn_in_episode": True,
    "learn_top_first": {"agent": "carve", "self_px": 10},
    "learn_wrist_stereo": {"threads": 2},
    "learn_rank_known": {"margin": 0.1},
    "learn_rank_floor": {"min_vox": 500},
    "learn_check_own": True,
    "learn_ask_known": {"cover_min": 0.95, "inlier_min": 0.93, "seen_min": 0.55},
    # a pixel the robot's model covers stays in a look that leaves the robot out in the picture when the frame
    # reads more than this nearer than the model (the learner's first look; the looks for the way as well)
    "self_front_m": 0.05,
}

# -- 2. objects in the way, without the fixed mask (no ``full_self_px``: with it the first look is another one and
# the learner starts on known objects)
WAY_AGENT = {
    "look_cross": True,  # a look for the way when the carry begins
    "look_stall": 1,  # one look at a stall an attempt
    "look_start": False,
    "look_route_only": False,
    "route_empty": True,  # the empty hand's way is planned too
    "transit_around": True,
    "detour_seen": True,
}
WAY_TOP = {"arm_cam_robot": {"register": {"use": "scene"}}}  # the camera read from the still scene


def _add(d: dict, more: dict, where: str) -> None:
    for k, v in more.items():
        if k in d:
            raise ValueError(f"candidate v10 adds keys and replaces none: {where}{k} is there already")
        d[k] = copy.deepcopy(v)


def bundle_v10(p: dict) -> dict:
    """``p`` (``long_dev.colour_long_params``) with v9's options and v10's additions. ``p`` is not changed."""
    q = LD.bundle(p)  # (a deep copy of ``p`` with v9's options; its ``m1``, ``m1.agent`` and ``server`` are its own)
    _add(q, LEARN_TOP, "")
    _add(q, WAY_TOP, "")
    _add(q["m1"]["agent"], WAY_AGENT, "m1.agent.")
    _add(q["server"], LEARN_SERVER, "server.")
    return q


def run_params(net: str = NET_POD) -> str:
    """The run parameters as one string, in the form the confirming pair's folders hold them."""
    return json.dumps(bundle_v10(LD.colour_long_params(net)), indent=1, sort_keys=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default=NET_POD)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = run_params(a.net)
    if not a.check:
        print(text)
        return 0
    sha = hashlib.sha256(text.encode()).hexdigest()
    print(json.dumps({"params_sha256": sha, "frozen": PARAMS_SHA256, "as_frozen": sha == PARAMS_SHA256}))
    return 0 if sha == PARAMS_SHA256 else 1


if __name__ == "__main__":
    raise SystemExit(main())
