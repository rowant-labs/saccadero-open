"""Fixture tests: the pivot shift that keeps a fixture's pivot in place, and the fixture switches' defaults. Monty
venv, numpy only; no simulator, no truth file, no image.

M2 (perceived poses, grounded instruction): perception in the loop.
"""

from __future__ import annotations

import math



def test_pivot_shift_keeps_the_pivot_fixed_and_is_zero_without_one():
    from rowant_lab.libero_env import m2_segment as SEG

    assert SEG.pivot_shift(None, 0.3, 1.2) == (0.0, 0.0)
    pv = (0.11, -0.02)
    for y0, y1 in ((0.0, math.pi), (0.4, 0.4 + math.radians(45)), (2.0, -1.0)):
        da, db = SEG.pivot_shift(pv, y0, y1)

        def at(yaw, a, b):  # the pivot's plane position for a pose at (a, b, yaw)
            return (a + math.cos(yaw) * pv[0] - math.sin(yaw) * pv[1], b + math.sin(yaw) * pv[0] + math.cos(yaw) * pv[1])

        p0 = at(y0, 0.5, -0.3)
        p1 = at(y1, 0.5 + da, -0.3 + db)
        assert abs(p0[0] - p1[0]) < 1e-12 and abs(p0[1] - p1[1]) < 1e-12


def test_fixture_switch_defaults_are_a50s():
    """A50: the A48 settings are the defaults (self-occlusion visibility and the yaw-mode choice on); every pivot
    switch and A44's whole-assembly choice stay off (they were read on the private set and made things worse)."""
    from rowant_lab.libero_env import m2_perceive as PC
    from rowant_lab.libero_env import m2_segment as SEG

    on = {"vis_self_occlusion", "asm_select_modes"}
    off = {"pivot_centroid", "asm_select", "pivot_jointed_wide", "pivot_variant_wide"}
    for k in on:
        assert SEG.PARAMS[k] is True and PC.PARAMS["support"][k] is True, k
    for k in off:
        assert SEG.PARAMS[k] is False and PC.PARAMS["support"][k] is False, k
    assert SEG.PARAMS["asm_mode_deg"] == 20.0
