"""rowant_lab.fastlayer: the one-frame fast layer (plane fit and crop -> proposal -> Monty-evidence CHECK -> refine
and decide) as a library, built from the E-0c / E-0f / E-0b / E-0e / E9-0 screens code (imported unchanged; the few
copies are listed in ``provenance.py``). TRUTH-FREE: nothing in this package reads a pose, a mask or a score of any
dataset; the BOP runner reads its inputs through ``rowant_lab.bop.inputs`` and a given-detections file.

Modules: ``frame`` (the input frame), ``memory`` (a graph memory: CHECK kernel, proposal shapes, refinement
clouds), ``onboard`` (BOP mesh -> Monty-format graph memory), ``wide`` (widened clutter-aware proposal),
``refine_data`` (refinement data and clutter fit quality), ``layer`` (the library interface and presets).

Step 3: ``symmetry`` (near-symmetries from the mesh at onboarding and the flip families), ``appearance`` (the
pluggable appearance scorer and the decision rules). The runners, the detections reader, the split and
``provenance`` are not part of this release.
"""
