"""The wrist-stereo probe (``pro2/probes/wrist_stereo_check.py``, development only, evaluator side) is not part of
this release. What stays is the check that the runner touches it only behind its option."""

from __future__ import annotations

import json
from pathlib import Path

from rowant_lab.libero_env import m3_stereo1_match as SM
from rowant_lab.pro2 import dev_episode as DE
from rowant_lab.pro2 import machine as MC


# ------------------------------------------------------------------------------------- the runner's off path
def test_the_runner_reaches_the_probe_only_behind_its_option_and_v9_does_not_name_it():
    src = Path(DE.__file__).read_text()
    assert src.count('params.get("probe_wrist_stereo") is not None') == 1
    assert src.count("wprobe = None") == 1 and src.count("if wprobe is not None:") == 3
    head = src.split('if params.get("probe_wrist_stereo") is not None:')[0]
    assert "wrist_stereo_check" not in head.replace("probes/wrist_stereo_check.py", "")  # imported nowhere else
    from rowant_lab.pro2 import long_dev as LD

    b = LD.bundle(LD.colour_long_params(MC.REGISTERED["net"]))
    assert "probe_wrist_stereo" not in json.dumps(b)
    assert SM.PS_W[1024] == (9, 13)
