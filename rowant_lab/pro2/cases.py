"""Pro2.0 cases the depth line does not test (imported by the runner, the summary and the loop's keep rule).

The depth line's looks read the live simulator, so a case that acts only on the protocol's camera images, or on the
camera itself, never reaches the robot: S15 (camera pose), S16 (image geometry), S19 (image noise), D07 (camera
moves), D08 (image dropout), D09 (exposure), D10 (image freeze). Cases that also act on the proprioception DO reach
it and are tested: D11 (modality freeze, default all), D12 and S20 (observation delay). Corrected 2026-11-07 after the
first baseline (D11 and D12 were first listed here); summaries recompute the flag from the case, not the stored row.
"""

BYPASSED = ("S15", "S16", "S19", "D07", "D08", "D09", "D10")


def bypassed(case: str) -> bool:
    return case in BYPASSED
