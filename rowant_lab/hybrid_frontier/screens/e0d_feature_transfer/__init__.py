"""Screen E-0d: do Monty's surface features transfer from its 2.5 cm surface patch to a 0.5 m camera?

The direction note, section 5.4 item 2 and section 6 E-0d. A numpy
dev screen on the stored E6 dev renders (E5a-G dev cases); no Monty episode, no MuJoCo, no training. Decides
nothing registered; it informs whether the fast layer's CHECK (P2) can reuse Monty's evidence on camera data.

Modules:

* ``isolation``  -- a stand-in for the check that ran before any model.pt was loaded (it inspects nothing here).
* ``graphs``     -- export one memory's graphs (positions and the feature columns Monty's evidence reads) to npz,
                    in its own process, after the isolation check; load the npz.
* ``camera``     -- TRUTH-FREE: read a stored render, find the blob (``e6/plane_crop``), back-project, sample patch
                    centres and compute Monty-style patch features with Monty's own functions.
* ``evidence``   -- Monty's own ``DefaultHypothesesDisplacer._calculate_evidence_for_new_locations`` driven
                    through a small read-only graph-memory shim.

The evaluator-only placement module and the runner are not part of this release.
"""
