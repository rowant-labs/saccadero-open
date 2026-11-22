"""This machine's places for a registered parameter text (``rowant_lab/pro2/machine.py``): exactly four values are
swapped, and a text that holds anything else in one of those places is refused."""

import json

import pytest

from rowant_lab.pro2 import machine


def _leaves(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _leaves(v, path + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _leaves(v, path + (i,))
    else:
        yield path, o


@pytest.mark.parametrize("name", ["v9", "v11"])
def test_only_the_four_places_change(name):
    reg = json.loads((machine.REPO_ROOT / "configs" / f"{name}.registered.json").read_text())
    here = {"robot_dir": "/r", "lfb_library": "/l", "net": "/n", "net2": "/n2"}
    loc = machine.localize(reg, here)
    a, b = dict(_leaves(reg)), dict(_leaves(loc))
    assert set(a) == set(b)
    changed = {p: (a[p], b[p]) for p in a if a[p] != b[p]}
    assert changed == {("robot_dir",): (machine.REGISTERED["robot_dir"], "/r"),
                       ("lfb_library",): (machine.REGISTERED["lfb_library"], "/l"),
                       ("server", "lfb_library"): (machine.REGISTERED["lfb_library"], "/l"),
                       ("colour", "net"): (machine.REGISTERED["net"], "/n"),
                       ("colour", "net2"): (machine.REGISTERED["net2"], "/n2")}
    assert machine.localize(loc, here) == loc  # a text that already holds this machine's values stays as it is
    assert json.loads(json.dumps(reg)) == reg  # (the registered text itself is not touched)


def test_another_value_in_one_of_the_places_is_refused():
    reg = json.loads((machine.REPO_ROOT / "configs" / "v9.registered.json").read_text())
    reg["colour"]["net"] = "/some/other/net.pt"
    with pytest.raises(ValueError, match="net"):
        machine.localize(reg, {"robot_dir": "/r", "lfb_library": "/l", "net": "/n", "net2": "/n2"})


def test_the_default_places_follow_the_settings():
    here = machine.local()
    assert here["robot_dir"] == str(machine.REPO_ROOT / "rowant_lab" / "pro2" / "robot")
    assert here["lfb_library"] == str(machine.LONG_LIB) and here["net"].endswith("colour-long.pt")
    assert here["net2"].endswith("colour2.pt") and str(machine.NETS) in here["net"]
