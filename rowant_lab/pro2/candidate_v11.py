"""Candidate v11: candidate v10 with the learner's wider start.

    python3 -m rowant_lab.pro2.candidate_v11            # the parameters, as the confirming pod ran them
    python3 -m rowant_lab.pro2.candidate_v11 --check    # their sha256 against the frozen one

**What it is.** v10 (``candidate_v10``: v9, the registered learner, the objects-in-the-way options) and two keys v10
does not name. As registered the learner starts only on "pick the X and place it ..." with no object of X's type
accepted, which is three of the benchmark's eight test tasks. With these keys it also starts (six of the eight):

1. on "put the X on the Y" where no object of X's type was accepted, AFTER the robot's own attempts: the learner
   follows two failed attempts, a plan that ended with every attempt failed, or no plan at all
   (``learn_after_fail``). With known objects the robot solves these tasks itself, nearly always at its first
   attempt, and the learner in that attempt's place lost practice episodes (measured on a pod);
2. where two objects of the asked type stand in the scene, one was accepted and that one clearly fails the
   instruction's relation, so the other is the one asked for: with the look first, as the registered start
   (the Spatial suite's 220 steps leave no room for an attempt before the learner).

Not on a step of a several-step instruction (measured: it lost a practice episode and won no table scene), and not
where an instruction asks for both objects of a type.

**How it was confirmed**: against v10 on one RTX 4090 pod, 2026-11-19 to 20, by five lines
and a rule between two forms, all fixed before the run. The 476 practice episodes 420 -> 421; our 53 table
scenes with new objects 1 -> 19; the second world's development part 80 and 80 of 100; the eight test tasks'
plain scenes 40 of 40; no row with an error. The same pod ran the form with the first start alone (421, and 11
of 53); both met all five lines and the rule took the form with both starts.
All of it is practice cases, our own table scenes with new objects, the second world's development part and the
test tasks' plain scenes. It has not run on a shifted test episode.

**What is frozen.** ``run_params()`` is, byte for byte, the text those runs were given (``PARAMS_SHA256``; the
runs' ``params.json``). v10's and v9's bundles are not touched: this module only adds keys, and refuses to replace
one.
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

from rowant_lab.pro2 import candidate_v10 as V10  # noqa: E402
from rowant_lab.pro2 import long_dev as LD  # noqa: E402

NET_POD = V10.NET_POD
PARAMS_SHA256 = "d03ce124911cdb8e09b626e856b918131775ac67a1b8e54286bc1f3a781f2f94"  # of ``run_params()``

# The learner's other starts (``learn_skill.FORMS``; a start not named here is on):
FORMS = {"steps": False, "own_first": ["put"]}
# "The robot's own attempts first" for the starts ``FORMS["own_first"]`` names: the learner follows two failed
# attempts, a plan that ended with every attempt failed, or no plan at all, when at least 190 steps are left after
# the hand has gone back (at most 20 steps) for the new look
AFTER_FAIL = {"retreat": 20, "min_left": 190, "attempts": 2}
ADDED = {"learn_forms": FORMS, "learn_after_fail": AFTER_FAIL}


def bundle_v11(p: dict) -> dict:
    """``p`` (``long_dev.colour_long_params``) with v10's bundle and v11's two keys. ``p`` is not changed."""
    q = V10.bundle_v10(p)
    for k, v in ADDED.items():
        if k in q:
            raise ValueError(f"candidate v11 adds keys and replaces none: {k} is there already")
        q[k] = copy.deepcopy(v)
    return q


def run_params(net: str = NET_POD) -> str:
    """The run parameters as one string, in the form the confirming pod's folders hold them."""
    return json.dumps(bundle_v11(LD.colour_long_params(net)), indent=1, sort_keys=True)


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
