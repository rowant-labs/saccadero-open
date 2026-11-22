"""Candidate v11's frozen bundle (``rowant_lab/pro2/candidate_v11.py``): it is the text its confirming pod ran
(2026-11-19 to 20), it only adds to v10, and v10's and v9's bundles are still what they were."""

import hashlib
import json

import pytest

from rowant_lab.pro2 import candidate_v10 as V10
from rowant_lab.pro2 import candidate_v11 as V11
from rowant_lab.pro2 import long_dev as LD

V10_SHA256 = "f20dd85d24bb2065b5a7cf0117abac7edac9a7f1543fb61b167beca523c82fd6"  # v10's confirming pair
V11_SHA256 = "d03ce124911cdb8e09b626e856b918131775ac67a1b8e54286bc1f3a781f2f94"  # the pod's ``params.json``


def _leaves(d: dict, pre: str = "") -> dict:
    """{dotted key: value} of every value that is no dict with keys (an empty dict is a value)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(_leaves(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def test_v11_is_byte_for_byte_the_text_its_confirming_pod_ran():
    text = V11.run_params()
    assert V11.PARAMS_SHA256 == V11_SHA256
    assert hashlib.sha256(text.encode()).hexdigest() == V11_SHA256
    assert V11.main(["--check"]) == 0
    # another depth net is another text, and the check says so
    assert hashlib.sha256(V11.run_params("other.pt").encode()).hexdigest() != V11_SHA256
    assert V11.main(["--check", "--net", "other.pt"]) == 1


def test_v10_is_still_what_was_confirmed_and_v11_changes_no_value_of_it():
    t10 = V10.run_params()
    assert V10.PARAMS_SHA256 == V10_SHA256 and hashlib.sha256(t10.encode()).hexdigest() == V10_SHA256
    assert V11.NET_POD == V10.NET_POD
    a, b = _leaves(json.loads(t10)), _leaves(json.loads(V11.run_params()))
    assert all(k in b and b[k] == v for k, v in a.items())  # every value of v10, as it was
    added = {k: b[k] for k in set(b) - set(a)}
    assert added == _leaves(V11.ADDED) and set(V11.ADDED) == {"learn_forms", "learn_after_fail"}
    # neither key is one v10 or the plain configuration names; the hint is in no candidate
    v10 = json.loads(t10)
    assert not set(V11.ADDED) & set(v10) and not set(V11.ADDED) & set(v10["server"])
    assert not set(V11.ADDED) & set(LD.colour_long_params(V11.NET_POD))
    assert "learn_hint" not in V11.run_params()


def test_what_the_two_keys_ask_of_the_learner():
    from rowant_lab.pro2 import learn_skill as LS

    p = json.loads(V11.run_params())
    forms = {**LS.FORMS, **p["learn_forms"]}
    assert forms["put"] is True and forms["steps"] is False and forms["relation"] is True
    assert forms["own_first"] == ["put"]
    put = {"why": "put", "step": 0, "steps": 1, "form": "put", "relation": None}
    rel = {**put, "why": "relation", "form": "pick", "relation": "next_to"}
    # the "put" start waits for the robot's own attempts: two failed ones, a retreat of 20 steps, 190 steps left
    o = LS.own_first(p, put)
    assert (o["attempts"], o["retreat"], o["min_left"]) == (2, 20, 190)
    assert o["after"] == LS.AFTER_FAIL["after"] and o["ended"] == LS.AFTER_FAIL["ended"]
    # the registered start and the relation's look first, as they were measured
    assert LS.own_first(p, None) is None and LS.own_first(p, rel) is None
    # the registered learner's keys are v10's, untouched
    assert {k: p[k] for k in V10.LEARN_TOP} == V10.LEARN_TOP


def test_bundle_v11_leaves_its_argument_alone_and_replaces_no_key():
    p = LD.colour_long_params(V11.NET_POD)
    before = json.dumps(p)
    q = V11.bundle_v11(p)
    assert json.dumps(p) == before and q is not p
    q["learn_forms"]["own_first"].append("relation")  # (the result is its own: the module's dicts do not follow)
    q["learn_after_fail"]["attempts"] = 1
    assert V11.FORMS["own_first"] == ["put"] and V11.AFTER_FAIL["attempts"] == 2
    for key in ("learn_forms", "learn_after_fail"):
        p2 = LD.colour_long_params(V11.NET_POD)
        p2[key] = {}
        with pytest.raises(ValueError, match="adds keys and replaces none"):
            V11.bundle_v11(p2)
