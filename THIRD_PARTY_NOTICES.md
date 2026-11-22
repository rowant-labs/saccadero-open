# Third-party notices

This repository's own code is under the MIT licence (`LICENSE`). It contains small pieces that follow other
projects' code, and it is used together with software and assets that are not part of it. This file says which.

## Code in this repository that follows other projects

### Monty (Thousand Brains Project)

MIT licence. Copyright 2025 Thousand Brains Project. Copyright 2021-2024 Numenta Inc.

This repository contains no file of Monty, and neither registered configuration runs any Monty code. Some of our
functions are re-implementations in numpy of Monty functions, written to give the same numbers:

- `rowant_lab/fastlayer/onboard.py`: the tangent basis and principal curvatures of a surface patch and the signed
  logarithm of a curvature (after Monty's `sensor_processing.py`);
- `rowant_lab/hybrid_frontier/screens/e0c_check_timing/frontend.py`: surface normals by total least squares (after
  Monty's `sensor_processing.py`);
- `rowant_lab/hybrid_frontier/screens/e0c_check_timing/kernel.py` and its copy `rowant_lab/fastlayer/check.py`: the
  evidence update for a pose hypothesis (after Monty's `hypotheses_displacer.py` and `graph_matching_utils.py`);
- `rowant_lab/libero_env/m2_onboard.py` uses the first of these; `rowant_lab/fastlayer/scorers/monty_colour.py` uses
  the form of Monty's feature evidence, and
  `rowant_lab/hybrid_frontier/screens/e0d_feature_transfer/evidence.py` carries Monty's default tolerances and
  weights as constants.

One function, `rowant_lab/libero_env/lfb_build.monty_build`, calls Monty directly (`tbp.monty`). It is used by
neither registered configuration and needs Monty installed.

Monty: https://github.com/thousandbrainsproject/tbp.monty

### robosuite

MIT licence. Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab.

- `rowant_lab/libero_env/m2_camera.py`: three camera formulas (metric depth from the depth buffer, the intrinsic
  matrix, the camera pose) are the expressions of robosuite 1.4.0's `utils/camera_utils.py`.
- `rowant_lab/libero_env/evidence-m3/stereo1/panda_hand_model.json`: the collision and visual meshes of the Panda
  hand and fingers, taken from the robot model that robosuite distributes (which, as far as we know, derives
  from the manufacturer's robot description). The second configuration's wrist-camera code needs it.

robosuite: https://github.com/ARISE-Initiative/robosuite

### LIBERO

MIT licence. Copyright (c) 2023 Lifelong Robot Learning.

Our code reads LIBERO's task files and object descriptions and follows their layout where it has to build or check
one. No file of LIBERO is in this repository.

LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO

### The MIT licence text (the same for the three projects above)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Used with this repository, not part of it

### LIBERO-Pro2.0 (the benchmark)

Nothing of the benchmark is in this repository: not its package, not its task files, not its assets. You download
it yourself, and this code uses it from your download. Its terms are its own:

- its simulator is a modified LIBERO and carries LIBERO's MIT licence;
- its evaluation service code and its task files carried no licence grant when we looked (2026-11-22);
- its object and fixture assets come from several sources with their own terms (among them MIT, Apache-2.0,
  CC BY 4.0 and LGPL-2.1); the notices are inside the download.

The object libraries that `scripts/build_libraries.py` builds on your machine are made of those assets. They stay
on your machine; please do not redistribute them without checking the assets' terms.

If you use the benchmark, cite its papers (arXiv:2610.24587 and arXiv:2510.03827) as its data card asks.

### The two depth nets (release files of this repository)

Trained by us, with no pretrained weights, on pictures we rendered in the benchmark's simulator of its scenes and
objects, with the simulator's depth as the target. They contain no copy of those pictures or assets. We release
them under this repository's MIT licence as far as that is ours to give; the assets they were trained on keep
their own terms.

### Python packages

The environments in `requirements/` install, among others: numpy, scipy, scikit-image (BSD-3-Clause); PyTorch
(BSD-3-Clause); MuJoCo (Apache-2.0); robosuite (MIT); OpenCV's Python wheels (MIT wrapper, Apache-2.0 library,
bundled third-party parts under their own terms); Pillow (MIT-CMU); numba and llvmlite (BSD-2-Clause). None of them
is redistributed here.
