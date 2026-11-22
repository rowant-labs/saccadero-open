"""Screen E-0c: a vectorised one-frame CHECK and its timing (the direction note, sections 3.2 P2, 3.7 and 6 E-0c).

A numpy / scipy dev screen on E6's 201 frozen dev renders (E5a-G dev cases). No Monty episode, no MuJoCo, no
training, no download. It reuses screen E-0d's camera features, graph export and reference scorer by import.

Modules (TRUTH-FREE unless stated):

* ``kernel``      -- Monty's one-observation evidence for all (patch, guess) pairs of one frame at once.
* ``frontend``    -- the vectorised P0 front end: 256 patch centres, TLS normal and hsv (Monty's maths).
* ``proposal``    -- the truth-free pose proposal: rest classes from the object's own learned graph, gravity from
                     the plane fit, yaw grid + xy ICP + free-space score -> 3 poses per HPG top-3 object.

The screen's pipeline, equivalence check, timing protocol, runner and evaluator-only scorer are not part of this
release.
"""
