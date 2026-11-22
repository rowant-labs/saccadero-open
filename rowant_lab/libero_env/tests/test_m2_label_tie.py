"""``dedup_label_tie`` (Pro2.0 colour line): two parts of one shape at one spot. Pure ordering logic; no simulator."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rowant_lab.libero_env import m2_perceive as P


def cand(gid: str, x: float, n_agree: int, inlier: float, n_px: int) -> dict:
    return {"gid": gid, "t": np.array([x, 0.0, 0.0]), "R": np.eye(3), "fit": {"n_agree": n_agree, "inlier": inlier},
            "n_px": n_px}


def ordered(cands: list, tie: float = 0.1, ratio: float = 1.25) -> list:
    me = SimpleNamespace(p={"dup_frac": 0.5})
    wc = SimpleNamespace(diam={"1": 0.1, "2": 0.1, "3": 0.1})
    return [c["gid"] for c in P.Perceiver._label_tie_order(me, wc, cands, tie, ratio)]


def test_off_by_default():
    assert P.PARAMS["dedup_label_tie"] is None


def test_tied_fit_goes_to_the_larger_colour_mask():
    # the measured case: the wrong label fits a little better and has about half the pixels
    assert ordered([cand("1", 0.0, 3144, 0.993, 1712), cand("2", 0.001, 3084, 0.995, 3488)]) == ["2", "1"]


def test_a_clear_fit_winner_stays_first():
    assert ordered([cand("1", 0.0, 3144, 0.993, 1712), cand("2", 0.001, 2000, 0.995, 3488)]) == ["1", "2"]


def test_similar_masks_stay_in_fit_order():
    assert ordered([cand("1", 0.0, 3144, 0.993, 3000), cand("2", 0.001, 3084, 0.995, 3488)]) == ["1", "2"]


def test_another_spot_is_not_a_rival():
    assert ordered([cand("1", 0.0, 3144, 0.993, 1712), cand("2", 0.3, 3084, 0.995, 3488)]) == ["1", "2"]


def test_same_type_is_not_a_rival():
    assert ordered([cand("1", 0.0, 3144, 0.993, 1712), cand("1", 0.001, 3084, 0.995, 3488)]) == ["1", "1"]


def test_largest_mask_of_several_rivals_wins():
    cs = [cand("1", 0.0, 3144, 0.993, 1000), cand("2", 0.001, 3084, 0.995, 2000), cand("3", 0.002, 3084, 0.99, 3000)]
    assert ordered(cs) == ["3", "2", "1"]
