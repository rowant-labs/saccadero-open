# Code map

What each folder is, how an episode runs, and why some names are odd.

## How an episode runs

Two processes, in two Python environments.

- **The worker** (`rowant_lab/pro2/dev_run.py`, one per processor you give it) builds the episode with the
  benchmark's own environment class from your download, runs our policy on it, and writes one row per episode. It
  sees what the benchmark hands a policy: colour pictures and the arm's own readings. Depth comes from our depth
  nets (`colour_net.py`, `colour_line.py`).
- **The perception server** (`rowant_lab/libero_env/m2_perceive.py`, one per worker, started by the worker) gets a
  picture with depth and answers where the known objects are, by fitting stored object models to it. It has no
  simulator and no neural network, and it runs under a hook that refuses to read or write outside its own folders.

They talk over a local socket (`m2_bridge.py`) and check each other's versions and shared code when they connect.

A third environment runs no episode. It is the worker's with MuJoCo 3.3.2 instead of 3.2.3, and the library
builder (`scripts/build_libraries.py`) uses it for its simulator steps, because the libraries of both scored runs
were made with that version (`requirements/export.txt`).

## Folders

| Folder | What it holds |
|---|---|
| `rowant_lab/pro2/robot/` | **The policy**: reading the instruction by rules (`ground.py`, `goal.py`, `sequence.py`), looking (`look.py`, `search.py`), grasp choice (`grasp.py`), motion (`agent.py`, `follow.py`), drawers (`drawer.py`), acting through delayed readings (`delay.py`), the episode's steps (`episode.py`). The worker loads this folder by path. |
| `rowant_lab/pro2/` | The runner (`dev_run.py`, `dev_episode.py`), the colour line and the depth net, the arm-based camera and joint checks (`arm_calib.py`, `cam_register.py`), the learner's worker side (`learn_skill.py`), the two configurations (`long_dev.py` builds v9, `candidate_v11.py` v11), scoring (`scored.py`), the practice tools, the depth nets' training code (`colour_train.py`, `colour_frames.py`, ...), the builders of the Goal and long-suite libraries, and this machine's settings (`machine.py`). |
| `rowant_lab/libero_env/` | The perception server and what it needs (`m2_perceive.py`, `m2_segment.py`, `m2_library.py`, `m2_kin.py`, `arm_camera.py`), the learner's server side (`lfb_*.py`), wrist-camera stereo (`m3_stereo1_*.py`), the builders of the first libraries (`export_geom.py`, `m2_export*.py`, `m2_libv*.py`, `m2_onboard.py`), and the tests. |
| `rowant_lab/fastlayer/` | The pose-fitting layer the server is built on: candidate poses, a check of each against the picture, colour evidence, symmetry. |
| `rowant_lab/hybrid_frontier/` | Older building blocks the fast layer uses: the check's kernel, the feature front end, point-cloud helpers. |
| `scripts/` | What you run: set-up, library builder, machine check, runner, episode lists, compare, score. |
| `configs/` | The two registered parameter texts, byte for byte, the practice list, and the paths example. |
| `docs/` | Results, disclosures, the two registrations, the records, and how the code was written. |

## Why some names are odd

The code was carved out of a larger research repository and keeps its names, because renaming the package or its
folders would change strings that feed random seeds, strings stored inside the built libraries, and the checksums
of the registered parameters.

| Name | What it means |
|---|---|
| `rowant_lab` | the package; named after the lab |
| `libero_env` | everything first written for the original LIBERO, before the Pro2.0 work |
| `m1`, `m2`, `m3` | three earlier stages: hands only with true poses (m1), perception in the loop (m2), depth and learning (m3). `m2_*` files are the second stage's and are still the perception core |
| `pro2` | the LIBERO-Pro2.0 work |
| `lfb` | the learner: builds a model of an object the program does not know from a few looks |
| `fastlayer`, `hybrid_frontier`, `screens`, `e0c`, `e0d`, `e0f`, `e6`, `e9` | earlier experiments on object recognition whose code the perception server still uses |
| "the Monty venv", `MONTY_VENV_PY`, `monty_colour` | the perception server's environment and one of its scorers. The names date from when the project was a fork of Monty; no Monty code runs here (`THIRD_PARTY_NOTICES.md`) |
| `venv-libero` | the worker's environment |
| `export-v3`, `onboard-v3`, `policy-v3` | generations of the object libraries |
| `long`, `libero_long_kitchen`, ... | the long suite (LIBERO-10) and its three tables, each handled as its own workcell |
| the depth line, the colour line | two forms of the program: one read the simulator's depth (development only, not in this release), the other reads colour pictures (both registered configurations) |
| world 2, `w2` | a second test world of our own, not in this release |
| labels in comments such as `A58`, `m2f:guard`, `W2-JOINTS`, `LFB-LIB` | references into our private working record: amendments, review rounds and registered checks |
| dates in comments | when a measurement was made; the numbers beside them are development measurements on training tasks |

## Switches that are off and not available

Both registered configurations leave these off. Their modules are not part of this release, so turning one on
ends in an import error: the earlier program's own policy (no `robot_dir`), `monty_arm: "minus"` in the server's
parameters, the semi-global stereo method of the wrist camera, a development probe (`probe_wrist_stereo`), and
episodes of our own second world. `rowant_lab/libero_env/lfb_build.monty_build` needs Monty installed, and the
scorer class `RenderCompareTuned` needs a file of tuned values that is not here.

The practice tools (`practice.py`, `subset.py`, `eight.py`, `board_average.py`) are the ones we used in
development. Some of their commands read run folders of ours by name; `scripts/episodes.py` is the way to make the
practice list and the plain list from your own download.

## Tests

    uv pip install --python "$ROWANT_SERVER_PYTHON" -r requirements/tests.txt
    PYTHONPATH=$PWD "$ROWANT_SERVER_PYTHON" -P -m pytest rowant_lab/libero_env/tests -q -n 4

From a fresh clone most run and the rest skip for want of a built library; with `ROWANT_DATA` pointing at built
libraries they run too. They include guards that read the policy's source and fail if it reaches for the
simulator's state or names a file of truth (`test_m2_truth_guard.py`).
