"""Candidate v10's frozen bundle (``rowant_lab/pro2/candidate_v10.py``): it is the text its confirming pair ran
(2026-11-19), it only adds to v9, and v9's own bundle is still what was scored."""

import hashlib
import json

import pytest

from rowant_lab.pro2 import candidate_v10 as V10
from rowant_lab.pro2 import long_dev as LD

V9_SHA256 = "bd0a0cc77c0f02006b5f9244ce723aa9c8e7c05f8a0dd9b8648f2e083170523d"  # ``SCORED.lock.json``
V10_SHA256 = "f20dd85d24bb2065b5a7cf0117abac7edac9a7f1543fb61b167beca523c82fd6"  # the pair's ``params.json``


def _leaves(d: dict, pre: str = "") -> dict:
    """{dotted key: value} of every value that is no dict with keys (an empty dict is a value)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(_leaves(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def test_v10_is_byte_for_byte_the_text_its_confirming_pair_ran():
    text = V10.run_params()
    assert V10.PARAMS_SHA256 == V10_SHA256
    assert hashlib.sha256(text.encode()).hexdigest() == V10_SHA256
    assert V10.main(["--check"]) == 0
    # another depth net is another text, and the check says so
    assert hashlib.sha256(V10.run_params("other.pt").encode()).hexdigest() != V10_SHA256
    assert V10.main(["--check", "--net", "other.pt"]) == 1


def test_v9s_bundle_is_still_what_was_scored_and_v10_changes_no_value_of_it():
    from rowant_lab.pro2 import scored as S

    assert V10.NET_POD == S.NET_POD
    t9 = S.run_params()
    assert hashlib.sha256(t9.encode()).hexdigest() == V9_SHA256
    assert json.loads(S.LOCK.read_text())["params_sha256"] == V9_SHA256  # the scored run's lock
    a, b = _leaves(json.loads(t9)), _leaves(json.loads(V10.run_params()))
    assert all(k in b and b[k] == v for k, v in a.items())  # every value of v9, as it was
    assert len(b) - len(a) == 40


def test_what_v10_adds_is_the_registered_learner_and_the_way_options_without_the_mask():
    v9 = LD.bundle(LD.colour_long_params(V10.NET_POD))
    v10 = V10.bundle_v10(LD.colour_long_params(V10.NET_POD))
    a, b = _leaves(v9), _leaves(v10)
    added = {k: b[k] for k in set(b) - set(a)}
    want = {**_leaves(V10.LEARN_TOP), **_leaves(V10.WAY_TOP), **_leaves(V10.WAY_AGENT, "m1.agent."),
            **_leaves(V10.LEARN_SERVER, "server.")}
    assert added == want and len(want) == 40
    # the top-level keys v10 names are new ones, and so are its keys in the agent's and the server's dicts
    assert not (set(V10.LEARN_TOP) | set(V10.WAY_TOP)) & set(v9)
    assert not set(V10.WAY_AGENT) & set(v9["m1"]["agent"]) and not set(V10.LEARN_SERVER) & set(v9["server"])
    # the objects-in-the-way options: the seven keys of the arm without the fixed mask, and the camera read from
    # the still scene
    assert V10.WAY_AGENT == {"look_cross": True, "look_stall": 1, "look_start": False, "look_route_only": False,
                             "route_empty": True, "transit_around": True, "detour_seen": True}
    assert v10["arm_cam_robot"] == {"register": {"use": "scene"}}
    text = json.dumps(v10)
    assert "full_self_px" not in text and "learn_after_fail" not in text  # no mask; not "the own attempt first"
    # the options are off unless given: the plain configuration names none of them
    plain = _leaves(LD.colour_long_params(V10.NET_POD))
    assert not set(want) & set(plain)


def test_the_learners_keys_are_the_skills_own_numbers():
    """(The comparison with the registered third forms' modules, ``world2_learn_c`` and ``world2_learn_e``, is not
    here: those modules are not part of this release.)"""
    from rowant_lab.pro2 import learn_skill as LS

    ask = V10.LEARN_TOP["learn_ask_known"]
    assert V10.LEARN_TOP["learn_calib_cue"] is True
    assert V10.LEARN_SERVER["self_front_m"] == 0.05
    # the rest of the look's keys are the skill's own numbers, written out
    assert {k: ask[k] for k in LS.ASK} == LS.ASK
    assert ask["own_near"]["any"] == [LS.OWN_NEAR, LS.OWN_NEAR_SEEN] and ask["own_top_min"] == LS.OWN_TOP_MIN
    assert V10.LEARN_TOP["learn_way_back"] == {k: LS.WAY_BACK[k] for k in ("attempts", "min_left")}
    assert V10.LEARN_TOP["learn_late"] == LS.LATE


def test_bundle_v10_leaves_its_argument_alone_and_replaces_no_key():
    p = LD.colour_long_params(V10.NET_POD)
    before = json.dumps(p)
    q = V10.bundle_v10(p)
    assert json.dumps(p) == before and q is not p
    q["server"]["self_front_m"] = 0.0  # (the result is its own: neither the argument nor the module's dicts follow)
    q["learn_ask_known"]["own_near"]["any"].clear()
    assert V10.LEARN_SERVER["self_front_m"] == 0.05 and len(V10.LEARN_TOP["learn_ask_known"]["own_near"]["any"]) == 2
    assert json.dumps(p) == before
    for where, key in (("top", "learn_in_episode"), ("agent", "look_cross"), ("server", "self_front_m")):
        p2 = LD.colour_long_params(V10.NET_POD)
        if where == "top":
            p2[key] = False
        elif where == "agent":
            p2["m1"]["agent"][key] = False
        else:
            p2["server"][key] = 0.02
        with pytest.raises(ValueError, match="adds keys and replaces none"):
            V10.bundle_v10(p2)
