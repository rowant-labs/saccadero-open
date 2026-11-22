"""The record-only watch on the pose fitter's visibility buffer (``m2_perceive.watch_visible``, 2026-11-18).

``refine.visible`` is a frozen M2 file. On one pod host it raised an IndexError that it cannot raise from inputs that
agree with themselves. The watch writes the numbers into the error and raises it again; a call that does not raise is
the frozen function's own call.
"""
import inspect
import json

import numpy as np
import pytest

from rowant_lab.hybrid_frontier.screens.e0f_refine import refine as RF
from rowant_lab.libero_env import m2_perceive as MPV


class Cam:
    """A 64 px camera at the origin looking along +z whose height reads as 4096 on the reads listed in ``tall``: two
    reads of one picture's size that disagree, which is the only way the buffer's index can run out."""

    project = RF.CameraGeom.project

    def __init__(self, tall=()):
        self.fx = self.fy = 500.0
        self.cx = self.cy = 32.0
        self.W = 64
        self.R_wc, self.c_w = np.eye(3), np.zeros(3)
        self.reads, self.tall = 0, set(tall)

    @property
    def H(self):
        self.reads += 1
        return 4096 if self.reads in self.tall else 64


class Fr:
    def __init__(self, cam):
        self.cam = cam

    def height(self, Xw):
        return np.ones(len(Xw))


def points():
    """200 points one metre ahead: half inside the 64 px picture, half below its lower edge (rows 70 to 300)."""
    rng = np.random.default_rng(3)
    u = rng.uniform(1, 63, 200)
    v = np.concatenate([rng.uniform(1, 63, 100), rng.uniform(70, 300, 100)])
    X = np.stack([(u - 32.0) / 500.0, (v - 32.0) / 500.0, np.ones(200)], axis=1)
    return X, np.tile([0.0, 0.0, -1.0], (200, 1))


@pytest.fixture
def watched():
    frozen = RF.visible
    MPV.watch_visible()
    yield frozen
    RF.visible = frozen


def test_a_call_that_does_not_raise_returns_what_the_frozen_function_returns(watched):
    X, N = points()
    assert RF.visible is not watched and RF.visible._watched
    a, b = RF.visible(X, N, Fr(Cam()), bin_px=3), watched(X, N, Fr(Cam()), bin_px=3)
    assert a.dtype == b.dtype and np.array_equal(a, b) and 0 < int(a.sum()) <= 100


def test_installing_the_watch_twice_wraps_once(watched):
    first = RF.visible
    MPV.watch_visible()
    assert RF.visible is first


def test_an_index_error_that_does_not_repeat_says_so_in_its_own_text(watched):
    X, N = points()
    with pytest.raises(IndexError) as ei:
        RF.visible(X, N, Fr(Cam(tall={1})), bin_px=3)  # the mask reads a tall picture, the buffer a 64 px one
    text = str(ei.value)
    head, note = text.split(" | visible ", 1)
    note = json.loads(note)
    assert head.startswith("index ") and "out of bounds" in head
    assert note["WH"] == ["64", "64"] and note["bin"] == "3" and note["nb"] == 22 * 22 and note["n"] == 200
    assert note["bad"] == 0 and note["again"] is False and note["uvzb"] == [] and note["nan"] == 0
    assert note["in"] == 100 and note["dt"] == ["float64", "float64"] and note["np"] == np.__version__
    assert len("IndexError: " + text) < 500  # a row keeps 500 characters of a perception error


def test_an_index_error_that_repeats_names_the_points(watched):
    X, N = points()
    with pytest.raises(IndexError) as ei:
        RF.visible(X, N, Fr(Cam(tall={1, 3, 5})), bin_px=3)
    note = json.loads(str(ei.value).split(" | visible ", 1)[1])
    assert note["again"] is True and note["bad"] == 100 and len(note["uvzb"]) == 3
    u, v, z, b = note["uvzb"][0]
    assert 0 <= u < 64 and 70 <= v < 300 and z == 1.0 and b >= note["nb"]
    assert len("IndexError: " + str(ei.value)) < 500


def test_a_failing_record_never_hides_the_error(watched, monkeypatch):
    def broken(*_a, **_k):
        raise RuntimeError("no numbers")

    monkeypatch.setattr(MPV, "_visible_numbers", broken)
    X, N = points()
    with pytest.raises(IndexError) as ei:
        RF.visible(X, N, Fr(Cam(tall={1})), bin_px=3)
    assert json.loads(str(ei.value).split(" | visible ", 1)[1]) == {"note_failed": "RuntimeError: no numbers"}


def test_the_server_installs_the_watch_before_it_builds_its_perceiver():
    src = inspect.getsource(MPV.main)
    assert 0 < src.index("watch_visible()") < src.index("per = Perceiver(")
