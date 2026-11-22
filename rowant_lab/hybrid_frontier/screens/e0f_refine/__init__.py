"""Screen E-0f: pose refinement from one depth frame (the direction note, sections 3.5 and 6; follows screens E-0c
and E-0b).

Question. E-0c's fast path picks an object and a pose from one 0.5 m camera frame (median ADD-S 2.6 mm). E-0b found
that grasp frames need about 1 mm and 2 deg, and suction about 1-3 mm and 2-5 deg. Can a fast, truth-free refinement
against the same frame's depth reach that, and how long does it take?

A numpy / scipy dev screen on E6's 201 frozen dev renders. No Monty episode, no MuJoCo, no training, no download.
E-0c's decisions (``decisions-dev.jsonl``) give the start; E-0c's and E-0b's modules are imported read-only.

Modules (TRUTH-FREE unless stated):

* ``spec``       -- every constant of the refinement, the arms and the readings, declared before any case was scored.
* ``models``     -- the model point sets the refinement registers to: Monty's U77 learned graph (nodes and stored
                    normals) and the truth-free fused view-finder cloud of F1-77 (the 1 mm sensor prior), with
                    normals fitted to the cloud itself. Loaded only after ``e9/isolation.require``.
* ``refine``     -- the refinement: back-projected blob depth points (plane-and-blob mask) against the chosen
                    object's model; point-to-plane ICP with Huber weights, a distance gate, a normal-compatibility
                    test and a visibility filter (model points facing the camera, not self-occluded, above the
                    table). Truth-free fit quality (inlier share, free-space share).

The screen's timing protocol, runner and evaluator-only score and report modules are not part of this release.
"""
