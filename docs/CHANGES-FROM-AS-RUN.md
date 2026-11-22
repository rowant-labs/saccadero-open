# What differs from the code as it ran

This repository was carved out of a larger private one. It is not the tree the two scored runs were made from, and this page says exactly how it differs. Every file is listed at the end with its checksum as it ran and as it is here.

## Two things to know first

- **The two runs ran two states of the code.** Every row of the first run records one checksum of the policy package and every row of the second another; 19 files changed between them (the learner, the camera registration, parts of the agent, the episode runner and the perception server). This repository holds the later state, the second attempt's. v11 adds parameters to v9 and changes none of v9's values, and on the practice list v9 solved 411 of 476 on both states. Whether v9 sends the same commands on both states was checked on one machine; the result is under "How we checked" below.

- **The registered parameters are untouched.** `configs/v9.registered.json` and `configs/v11.registered.json` are the registered texts byte for byte, and the code here builds the same two texts (a test checks both checksums against the locks). They name four places as they were on our machines. `rowant_lab/pro2/machine.py` swaps those four values for yours when a run starts, so the parameters a row of your run records differ from ours in those four values and in nothing else. One data file, `rowant_lab/pro2/colour_adapter_params.json`, is published unchanged for the same reason and still carries paths of our machines.

## What was changed, by kind

The tree has 182 files under `rowant_lab/`: 67 identical to the files as they ran, 47 that differ only in comments and docstrings, 64 with a change to code, and 4 new. "Only in comments" is checked by machine: the two files parse to the same program once docstrings are dropped.

### 1. Fixed machine paths became settings

About sixty lines named folders on our machines. They now come from `rowant_lab/settings.py` (one data folder, `ROWANT_DATA`) and `rowant_lab/pro2/machine.py` (the benchmark's download, the two interpreters, the depth nets). The layout under the data folder is the one we used. No value that the program computes with was changed.

### 2. A guard that inspected our machines is gone

As run, every program that loads an object memory first listed the machine's processes (and, in one file, its network names) and refused to start while another experiment of ours was running. That guard is not published. `rowant_lab/hybrid_frontier/e9/isolation.py` and `screens/e0d_feature_transfer/isolation.py` are stand-ins with the same entry points that inspect nothing. The perception server's own file-access hook used to allow the guard's four system commands; it now allows no subprocess at all. The same hook allowed temporary files only under a macOS folder (our Linux machines set `TMPDIR` inside it); it now also allows the machine's own temporary folder.

### 3. Code the runs never called was cut where it needed modules that are not published

- `rowant_lab/libero_env/m2_run.py`: the earlier robot's runner, cut to the five names the Pro2.0 runner uses (start and stop of the perception server, the hello check, one constant). Their code is unchanged except that the server's `PYTHONPATH` no longer names a `src` folder, which this repository does not have.
- `rowant_lab/libero_env/m2_episode.py`: cut to the two names in use. Its imports are kept on purpose: the worker sends the server the checksums of the shared modules it has loaded, and this module is what loaded them. (We found this the hard way: with the imports gone, the server refused the worker.)
- Three modules that were loaded as run but never called are not here: the earlier robot's look tool and its two evaluators (`m2_agent.py`, `m2_evaluator.py`, `evaluator.py`). The worker therefore has four fewer modules loaded than as run (those three and `m1_paths.py`) and one more (`settings.py`).
- `rowant_lab/fastlayer/segment.py` lost its own command-line runner; `screens/e0d_feature_transfer/camera.py` and `evidence.py` lost the functions that called Monty.
- Switched-off options inside functions that do run were left exactly as they were, even where the module they would import is not published (`docs/CODE-MAP.md` lists them).

### 4. The scoring module checks checksums, not our private history

As run, `rowant_lab/pro2/scored.py` refused to touch the test split unless the code was at the registered commit of the private repository. Here, `verify` compares the run parameters, the two depth nets and the benchmark's archive with the lock by checksum; the commands that touch the test split ask for `--test-split`; the command that wrote a lock is gone; and the score gained three more readings of Overall beside the registered one. The registered formula's code is unchanged, and it gives our stored numbers from our rows.

### 5. The depth line is not here

An earlier form of the program read the simulator's depth instead of estimating it. It was used in development only and is not part of this release. `long_dev.long_params()` and `practice.run_params("depth")` now say so instead of building its parameters. The colour line's parameters are built as before.

### 6. Small things

- A private branch name that was a default value in `m3_stereo1_audit.py` (used by a command the runs never called) is now `origin/HEAD`.
- Two camera helpers moved into `rowant_lab/pro2/camera_math.py` from a module that is not published.
- The practice tools read the folder of finished runs from the settings instead of from a second-world module.
- The two environments' version pins are `requirements/worker.txt` and `requirements/server.txt` (the worker's with MuJoCo 3.2.3, as on the machines of both runs).

### 7. The library builders

`scripts/build_libraries.py` and `scripts/compare_libraries.py` are new. The libraries of the scored runs were built in four stages over a month, the first from the original LIBERO release; the script replays those stages in order from the benchmark's download alone. For that, the builder modules were changed in five places:

- a scene's start state is read through LIBERO's own API (`export_geom.start_state`) instead of from arrays we kept privately (`export_geom.py`, `m2_export.py`, `m2_export3.py`);
- the first library's files are checked against that library's own manifests instead of a private record (`m2_libv3.py`). This changes one block, `sources`, in three manifest files; no reader uses it;
- the Goal suite's base task file is found through LIBERO's API instead of the service's manifest (`goal_lib.py`);
- the long suite's build no longer needs an original LIBERO checkout beside it; its record of that comparison is then empty (`long_lib.py`);
- `long_lib.py` gained a command that writes LIBERO's configuration.

**How close the rebuilt libraries are.** On the machine the scored libraries were built on (a Mac, MuJoCo 3.3.2), the rebuilt run-time set of 306 files has 198 byte-identical files and 107 with the same content: every numeric array, mesh, grid and template is identical bit for bit, and what differs is path strings, time stamps, timings and the checksums that follow from those. One file differs in content: a manifest's `sources` block, as said above. Whether Linux with the pinned MuJoCo 3.2.3 gives the same arrays is measured in the proof run below.

`docs/records/scored-runtime-libraries.digest.json` holds the checksums of our libraries (no content), so `python scripts/compare_libraries.py docs/records/scored-runtime-libraries.digest.json "$ROWANT_DATA/runs"` tells you how yours compare.

One thing to know: the scored libraries were exported with MuJoCo 3.3.2 on a Mac, while the runs' workers used MuJoCo 3.2.3 on Linux, the version this repository pins.

### 8. Comments

Comments and docstrings that named a person, a private working document, a rented machine or a cost were edited or removed: 245 lines in 56 files in one pass, and a few more by hand. Usage examples now say `<server python>` and `<worker python>` where they named paths on our machines. Dated technical notes and the internal labels of our working record were kept (`docs/CODE-MAP.md` explains the labels). Every file touched this way was compared by machine with its version before the edit: no executable code changed.

### 9. Tests

54 test files were carried, with 794 of their 835 test functions. 41 were removed because they tested modules that are not published or read private files; a few lost single assertions for the same reason. Tests that need built libraries now skip when there are none. From a fresh clone about 790 pass and about 180 skip; with built libraries about 965 pass.

Two guard tests police what the policy side may import and name. Their lists were adapted to the tree and nothing else: files that are not here left the lists; `rowant_lab.settings` is allowed as an import of `m2_paths.py`; two imports that no longer exist were removed from the allow-list (stricter); the planted leaks the guard must catch are now planted in `m2_ground.py`; and one name check is skipped for a module that is not published (`m2_mminus`), while its import stays under the allow-list.

## How we checked that the carve-out acts the same

- **Imports.** Every import of our own package inside this tree resolves inside it, apart from the switched-off options listed in `docs/CODE-MAP.md`.
- **Registered parameters.** Built by this code, both texts have the checksums registered before the runs (a test).
- **Scores.** The scorer here gives our stored numbers from the rows of both runs.
- **Tests.** With built libraries 970 pass and 7 skip; from a fresh clone 796 pass and 181 skip.
- **The same commands on one machine.** The same episodes were played from the private tree and from this one on one machine, and the commands sent to the robot were compared step by step. The results, and those of the proof run on one rented RTX 4090 machine (the private tree against a clean clone of this repository, the scored libraries against rebuilt ones, the first run's code against this code), are added by the commit that follows this one.

## Every file

`identical`: the same bytes as when the second attempt ran. `comments`: differs only in comments and docstrings. `code`: see the sections above; the last column names what changed at the file's top level. `new`: written for this release. Checksums are sha256, first 12 characters.

| File | Class | As run | Here | Changed |
|---|---|---|---|---|
| `__init__.py` | comments | 140d930dc3d1 | d883013f29b5 |  |
| `settings.py` | new |  | 9f31f1e83e08 |  |
| `pro2/arm_calib.py` | identical | cf3075b8bfe5 | cf3075b8bfe5 |  |
| `pro2/board_average.py` | code | 989f91bfab83 | c8d82328c0f2 | added EVAL |
| `pro2/cam_register.py` | comments | e41527b3b6cf | 2d85114ed76e |  |
| `pro2/camera_math.py` | new |  | c32a00260ed1 |  |
| `pro2/candidate_v10.py` | comments | 77bf634246bb | 015a1d334b5d |  |
| `pro2/candidate_v11.py` | comments | df4bc1a894ed | 2a26c3aae4bb |  |
| `pro2/cases.py` | identical | 6462e9d32be3 | 6462e9d32be3 |  |
| `pro2/colour_adapter_params.json` | identical | 1e346a05c076 | 1e346a05c076 |  |
| `pro2/colour_cameras.json` | identical | 850ef9b6ab7a | 850ef9b6ab7a |  |
| `pro2/colour_cameras_long.json` | identical | 7aa6cd3e5b3d | 7aa6cd3e5b3d |  |
| `pro2/colour_frames.py` | code | 6a3465ad1709 | 9be673e83fc7 | changed COLD, PAYLOAD, random_camera |
| `pro2/colour_line.py` | identical | 3152560c9773 | 3152560c9773 |  |
| `pro2/colour_long.py` | code | bbaca7ea8ac9 | 65b1283763f3 | changed COLD, PAYLOAD |
| `pro2/colour_net.py` | identical | 4f7dfa351cf2 | 4f7dfa351cf2 |  |
| `pro2/colour_score.py` | identical | dc1b56f1a421 | dc1b56f1a421 |  |
| `pro2/colour_sweep.py` | identical | 9d315095e5b0 | 9d315095e5b0 |  |
| `pro2/colour_train.py` | identical | a55f734658db | a55f734658db |  |
| `pro2/dev_episode.py` | comments | 211ad3256c76 | e28913d6334d |  |
| `pro2/dev_run.py` | code | 1f62df3aa091 | 07d8ba0acaa2 | changed PAYLOAD |
| `pro2/eight.py` | code | d137f9c50439 | 9d989825cf16 | added EVAL; changed ROOT |
| `pro2/goal_lib.py` | code | a28308075ff6 | 92091aca29f7 | changed G, M1LIB, M2LIB, PAYLOAD, base_task_file |
| `pro2/goal_paths.py` | comments | e5c60b832cfa | 1c365fb3c59e |  |
| `pro2/learn_skill.py` | comments | a39e0b88966c | 289b9d30178c |  |
| `pro2/long_dev.py` | code | 777d4d381362 | 1084a1870e11 | changed ROOT, long_params |
| `pro2/long_lib.py` | code | 279d2ae520bb | 1d0e297b2094 | added cmd_config; changed LL, ORIG_BDDL, cmd_export, main |
| `pro2/long_paths.py` | identical | dfd6e73f2cc2 | dfd6e73f2cc2 |  |
| `pro2/machine.py` | new |  | 535e209f82dd |  |
| `pro2/plain.py` | code | 4ea2db870cb1 | 09378500da84 | added stmt if __package__ in (None, ''):     sys.path.append(str(Path(_; changed CFG, PAYLOAD |
| `pro2/pod_selfcheck.py` | comments | 56b3913a8e71 | 92a7a67ca7e4 |  |
| `pro2/practice.py` | code | 9a9dc2fdd46f | 7f77f5c6bc0a | added ADAPTER, EVAL; changed DEVRUNS, ROOT, run_params |
| `pro2/scored.py` | code | 71d24858ffb6 | 6e2b8b195b2e | 6 removed; added PLAN_KEYS, RECORDS, _test_split_asked, cmd_verify…; changed BUNDLE, COLD, LOCK, LOCKS, PAYLOAD… |
| `pro2/split_guard.py` | code | 0a18a36c1795 | 9dc44c257549 | changed PLAIN_ROOT |
| `pro2/subset.py` | code | ea0c19d27412 | 75f2b1f68029 | added EVAL; changed ROOT |
| `pro2/wrist_stereo.py` | comments | 2417237a7d97 | c23c9f03545f |  |
| `pro2/probes/kin_same.py` | comments | d4d08615c59a | 5c028a6b13e6 |  |
| `pro2/robot/__init__.py` | comments | 3bb75dbae75d | b774875913cb |  |
| `pro2/robot/agent.py` | comments | fd90ea5c5e99 | 47b912e960ba |  |
| `pro2/robot/behind.py` | comments | 2e12e97b3860 | c090067bd4a0 |  |
| `pro2/robot/delay.py` | comments | a3f847b1a9e7 | f211bce6e681 |  |
| `pro2/robot/drawer.py` | comments | 834592d6c54c | 1939dafa66e8 |  |
| `pro2/robot/episode.py` | identical | c6254b9015b3 | c6254b9015b3 |  |
| `pro2/robot/follow.py` | identical | f954cf16db87 | f954cf16db87 |  |
| `pro2/robot/goal.py` | identical | 06487dc73a41 | 06487dc73a41 |  |
| `pro2/robot/grasp.py` | comments | 7edbeca551dc | e7a4a47c9c77 |  |
| `pro2/robot/ground.py` | comments | 1c1711f4c541 | 768c9e4764eb |  |
| `pro2/robot/look.py` | comments | e4cd9d2b2cff | 2866fff2b178 |  |
| `pro2/robot/search.py` | comments | 6e5b69c8ac64 | 2da0388fcc08 |  |
| `pro2/robot/seen.py` | identical | 32c3286a3e79 | 32c3286a3e79 |  |
| `pro2/robot/sequence.py` | identical | 5cb6820627ee | 5cb6820627ee |  |
| `pro2/robot/sized.py` | identical | 2dc81d86f501 | 2dc81d86f501 |  |
| `pro2/robot/wrist2d.py` | identical | 59f4501c9875 | 59f4501c9875 |  |
| `libero_env/__init__.py` | comments | 4aa1b4eeb874 | 2e17eb07efc3 |  |
| `libero_env/arm_camera.py` | comments | 90fa52bdf9d3 | 335c56ce299f |  |
| `libero_env/build_records.py` | comments | 440adea6fabf | 38eb7b104221 |  |
| `libero_env/export_geom.py` | code | fa0fecb8cf1b | 799e85caad83 | added start_state; changed export_task, main |
| `libero_env/grasp.py` | comments | 7edbeca551dc | bf3198a42845 |  |
| `libero_env/lfb_build.py` | comments | a4d661dc22c6 | f61dbe5f480a |  |
| `libero_env/lfb_episode.py` | comments | cabc7c17dc9f | c966e0540e64 |  |
| `libero_env/lfb_learn.py` | code | 758bcb50706f | 56a52cdf9fd4 | changed PARAMS_FILE |
| `libero_env/lfb_library.py` | comments | b491adef8401 | 1d9477deb2cc |  |
| `libero_env/lfb_paths.py` | code | a27b522ce2eb | 0c425caa2410 | changed ROOT |
| `libero_env/m1_paths.py` | code | a55afb77d5d2 | ed2cc85401a3 | changed COLD |
| `libero_env/m2_audit.py` | identical | b6f5fcd34de9 | b6f5fcd34de9 |  |
| `libero_env/m2_bridge.py` | identical | 6e91c7ccaa1f | 6e91c7ccaa1f |  |
| `libero_env/m2_camera.py` | identical | 392801114e2f | 392801114e2f |  |
| `libero_env/m2_episode.py` | code | 0c4eed07a5e4 | 74e3fe110722 | 6 removed |
| `libero_env/m2_export.py` | code | 47ddaed62ed8 | 3847d41a73b1 | changed main |
| `libero_env/m2_export3.py` | code | b748d44e65e7 | c7fefef9e16a | changed V2_SRC, V3_OUT, cmd_probe |
| `libero_env/m2_ground.py` | identical | a54fa63f8067 | a54fa63f8067 |  |
| `libero_env/m2_integrity.py` | identical | 98a6b9582dfc | 98a6b9582dfc |  |
| `libero_env/m2_kin.py` | identical | 0c3d7903501f | 0c3d7903501f |  |
| `libero_env/m2_library.py` | comments | bb6ad575cd21 | d27ff4cc8aed |  |
| `libero_env/m2_libv2.py` | comments | 13dc718cbdc3 | 604c349799b9 |  |
| `libero_env/m2_libv3.py` | code | 245ee6720c79 | a62413977395 | added m1_hashes; changed build |
| `libero_env/m2_onboard.py` | comments | 3c153cc500f7 | 0c92fb6ce03f |  |
| `libero_env/m2_paths.py` | code | 5a83b0454b01 | ecd52ee5ba49 | changed LIBRARY, M1_LIBRARY |
| `libero_env/m2_perceive.py` | code | 2a5de95c7b9e | 8ff8c6de5509 | changed GUARD_COMMANDS, audit_roots, guard_command |
| `libero_env/m2_records.py` | identical | 7df1632618a8 | 7df1632618a8 |  |
| `libero_env/m2_rpaths.py` | code | 98cb1f3e8b94 | 3109b06e6741 | changed COLD, LIBERO_VENV_PY, MONTY_VENV_PY, SCRATCH |
| `libero_env/m2_run.py` | code | a34d604c4130 | 7279e29659a8 | 37 removed; changed start_server |
| `libero_env/m2_segment.py` | comments | 4cabda34acd5 | 5c0fa5d9b949 |  |
| `libero_env/m3_stereo1_audit.py` | code | 44bca063e888 | ba54177c37b7 | changed M3_RUNS, ORIGIN_REF |
| `libero_env/m3_stereo1_handmodel.py` | comments | 5b71afd1389c | 9cbf64ae6b1c |  |
| `libero_env/m3_stereo1_match.py` | comments | 598f693bb63d | 662ceb418e06 |  |
| `libero_env/m3_stereo1_poses.py` | comments | 92daf7cb780d | 4412b829f5ad |  |
| `libero_env/tools.py` | identical | 2447d744ccb0 | 2447d744ccb0 |  |
| `libero_env/evidence-m3/stereo1/panda_hand_model.json` | identical | be9b435d422e | be9b435d422e |  |
| `libero_env/tests/__init__.py` | identical | 028b5483c9aa | 028b5483c9aa |  |
| `libero_env/tests/test_appear_stand.py` | identical | 68896d3f70ff | 68896d3f70ff |  |
| `libero_env/tests/test_arm_calib.py` | identical | e2b2e498d416 | e2b2e498d416 |  |
| `libero_env/tests/test_arm_camera.py` | identical | 2f488fa52766 | 2f488fa52766 |  |
| `libero_env/tests/test_arm_joints.py` | identical | e49b63b08e13 | e49b63b08e13 |  |
| `libero_env/tests/test_arm_readings_fit.py` | identical | 4ed611507106 | 4ed611507106 |  |
| `libero_env/tests/test_cam_register.py` | identical | dccfb3355dd2 | dccfb3355dd2 |  |
| `libero_env/tests/test_candidate_v10.py` | code | b02671905bad | 8ec0239d5288 | 1 removed; added test_the_learners_keys_are_the_skills_own_numbers |
| `libero_env/tests/test_candidate_v11.py` | comments | bc919e7a6f23 | acdcd0df2e0f |  |
| `libero_env/tests/test_clear.py` | identical | c5ac39d6d4c8 | c5ac39d6d4c8 |  |
| `libero_env/tests/test_close_look.py` | identical | e944742082c4 | e944742082c4 |  |
| `libero_env/tests/test_colour_long.py` | code | 4116b8e67173 | 0b0fc8301950 | changed PRO2, test_colour_line_parameters_for_the_long_suite, test_the_build_record_gives_the_cells_the_runner_knows, test_the_bundle_is_the_candidate |
| `libero_env/tests/test_colour_rcam.py` | code | aa187b0a0644 | d0bed29056f2 | changed _stored_like, test_random_camera_is_seeded_and_in_range |
| `libero_env/tests/test_cover_pick.py` | identical | 2c8de79d68ee | 2c8de79d68ee |  |
| `libero_env/tests/test_delay.py` | identical | 1fa0b6dde052 | 1fa0b6dde052 |  |
| `libero_env/tests/test_dev_run_reap.py` | identical | 1d8b7fca1625 | 1d8b7fca1625 |  |
| `libero_env/tests/test_drawer_open.py` | code | 5ea6dff03387 | 6d89e4482596 | changed _goal |
| `libero_env/tests/test_fast_open.py` | identical | 241c0a41e490 | 241c0a41e490 |  |
| `libero_env/tests/test_grasp_checks.py` | code | d120ccde0304 | d3b4a77c4378 | 1 removed |
| `libero_env/tests/test_lag_options.py` | code | 033e6723ad46 | effb33785400 | changed _episode |
| `libero_env/tests/test_learn_after_fail.py` | code | 119ffa55b965 | d4d1691793e7 | changed test_the_runner_looks_again_only_with_the_option_and_never_before_another_reason |
| `libero_env/tests/test_learn_ask_known.py` | code | f935b5c4dc1f | cde0b42bff0e | 1 removed; changed test_the_audit_declares_the_two_request_keys_and_the_options_are_off_by_default, test_the_camera_checks_cue_shows_the_robots_own_en |
| `libero_env/tests/test_learn_forms.py` | identical | 1d112249ac0b | 1d112249ac0b |  |
| `libero_env/tests/test_learn_way_back.py` | code | 35a7cd53b57f | c3137f224df6 | 2 removed; changed test_the_runner_asks_to_keep_the_learned_thing_where_it_plans_again, test_the_runner_goes_back_only_with_the_option_and_never_befor |
| `libero_env/tests/test_lfb_episode.py` | comments | 317818952fcb | 292c6e9598b8 |  |
| `libero_env/tests/test_long2_options.py` | code | 7a6c76c5a06a | ff0f2f46de7a | changed _kitchen, _two_pots |
| `libero_env/tests/test_long_paths.py` | code | 6cd47f4376bd | 98ff93dfb110 | changed TREE, test_each_cell_loads_from_the_built_tree_with_its_panda |
| `libero_env/tests/test_m1_truth_guard.py` | code | 3cc429f26bdd | c54c69159774 | changed ALLOWED_IMPORTS, POLICY_FILES |
| `libero_env/tests/test_m1_units.py` | code | 05587c00c0b4 | ca70e6f6056e | 8 removed |
| `libero_env/tests/test_m2_fixtures.py` | code | 6a79c6f9d64a | 5fb518f72206 | 6 removed |
| `libero_env/tests/test_m2_label_tie.py` | identical | 587c52367943 | 587c52367943 |  |
| `libero_env/tests/test_m2_library.py` | code | 8a945411972f | 443e4e6f890a | 2 removed; changed need, test_v2_manifest_matches_every_file_and_names_no_value |
| `libero_env/tests/test_m2_runtime.py` | code | adcb1580be9d | 4ec759d4aae1 | 13 removed; changed test_numba_cache_is_pinned, test_server_audit_hook_refuses_truth_reads, test_server_hello_check_refuses_mismatches |
| `libero_env/tests/test_m2_segment.py` | identical | c6222d8676a5 | c6222d8676a5 |  |
| `libero_env/tests/test_m2_truth_guard.py` | code | e76f9afabcc4 | 858832434801 | added NOT_IN_RELEASE, PLANT_IN, _absent; changed POLICY_FILES, POLICY_IMPORTS, POLICY_IMPORTS_PER_FILE, _planted, perception_violations… |
| `libero_env/tests/test_m2_units.py` | comments | 9e6097dead9e | 56d75b253f6d |  |
| `libero_env/tests/test_m3_stereo1_match.py` | code | a087b54789e5 | 8a9550d5e4d3 | 2 removed; changed _child_env, test_ast_import_allowlist, test_ast_no_c_level_io_no_spawn_no_banned_constants, test_g0_grid_numbers, test_g0_peak_rss_ |
| `libero_env/tests/test_m3_stereo1_poses.py` | code | 61d8d524bc38 | 0ae8f7c466f0 | 1 removed; added needs_git, _git_checkout, test_k_true_is_the_wrist_cameras_intrinsics; changed run_cli, test_cli_pt_poses_pick_settled_po_end_to_end, |
| `libero_env/tests/test_machine.py` | new |  | f6844e05f14b |  |
| `libero_env/tests/test_path_route.py` | code | 859fd6b70e32 | 136e889792da | changed pytestmark |
| `libero_env/tests/test_place_evidence.py` | comments | f4bf0a94693e | 0964ad849834 |  |
| `libero_env/tests/test_place_measured.py` | code | bac5411f8fd8 | 9fb71883f5f7 | changed test_a_blocked_carry_is_noticed_in_a_dozen_steps_looks_again_and_goes_over_what_it_now_sees, test_a_carry_crosses_above_what_the_look_saw_stan |
| `libero_env/tests/test_pod_selfcheck.py` | identical | ec56c3d6b6f5 | ec56c3d6b6f5 |  |
| `libero_env/tests/test_pro2_goal_paths.py` | code | 791edd0d19ff | ba9fffea24ec | 3 removed; added test_m2_paths_names_no_goal |
| `libero_env/tests/test_pro2_long_lib.py` | code | 220cc6c69443 | 273b1ff93eff | changed test_libero_config_names_the_simulators_own_task_files_and_start_states |
| `libero_env/tests/test_pro2_steps.py` | identical | 574d05803f6b | 574d05803f6b |  |
| `libero_env/tests/test_pro2_steps_run.py` | code | 36b6d3cdda62 | 427794944b82 | changed LIB_ROOT, pytestmark |
| `libero_env/tests/test_redetect.py` | identical | 77e62039acd2 | 77e62039acd2 |  |
| `libero_env/tests/test_scored.py` | code | b30168f136db | b0f7bfc97051 | 5 removed; added _sha, test_a_missing_lock_changed_parameters_or_another_net_are_refused, test_both_locks_hold_the_parameters_this_code_builds, test_t |
| `libero_env/tests/test_sequence.py` | code | d5536e955274 | 707d4f1c978b | changed _goal_scene |
| `libero_env/tests/test_server_trace.py` | code | 80a64d761950 | 8a7f43970bfc | changed TEXT |
| `libero_env/tests/test_sized.py` | identical | 195b4972e3e8 | 195b4972e3e8 |  |
| `libero_env/tests/test_twin_prior.py` | identical | a6d1f5e22fea | a6d1f5e22fea |  |
| `libero_env/tests/test_visible_watch.py` | identical | 2e01b2570ee3 | 2e01b2570ee3 |  |
| `libero_env/tests/test_wrist_stereo.py` | code | 4e9d011a4f48 | 519c68bee9e7 | 3 removed; changed test_the_options_are_off_by_default_and_v9_does_not_name_them |
| `libero_env/tests/test_wrist_stereo_probe.py` | code | f75fe83fe704 | 9b59e3c8b12e | 23 removed; changed test_the_runner_reaches_the_probe_only_behind_its_option_and_v9_does_not_name_it |
| `hybrid_frontier/__init__.py` | comments | f4690ab67615 | 84737e45df05 |  |
| `hybrid_frontier/contracts/__init__.py` | comments | 70c044af10bd | 847863ce4f70 |  |
| `hybrid_frontier/contracts/prior_shape_v1.py` | identical | 48dbab5cf745 | 48dbab5cf745 |  |
| `hybrid_frontier/contracts/sha.py` | identical | d32fccae8f71 | d32fccae8f71 |  |
| `hybrid_frontier/screens/__init__.py` | comments | f4690ab67615 | 84737e45df05 |  |
| `hybrid_frontier/screens/e0c_check_timing/__init__.py` | comments | b5e50b939a4b | 0dc151d3b926 |  |
| `hybrid_frontier/screens/e0c_check_timing/frontend.py` | identical | fa8971b93b07 | fa8971b93b07 |  |
| `hybrid_frontier/screens/e0c_check_timing/kernel.py` | identical | 0fbab913b9df | 0fbab913b9df |  |
| `hybrid_frontier/screens/e0c_check_timing/proposal.py` | identical | 169ac6adfa7e | 169ac6adfa7e |  |
| `hybrid_frontier/screens/e0d_feature_transfer/__init__.py` | comments | 625282578b9e | 9d53c0fb5147 |  |
| `hybrid_frontier/screens/e0d_feature_transfer/camera.py` | code | 888610a69910 | ee6d466b5b3d | 5 removed |
| `hybrid_frontier/screens/e0d_feature_transfer/evidence.py` | code | be8d57009fad | 4520b34642d4 | 3 removed |
| `hybrid_frontier/screens/e0d_feature_transfer/graphs.py` | code | 04164bdf1252 | e7cd1349cef2 | changed MEMORIES |
| `hybrid_frontier/screens/e0d_feature_transfer/isolation.py` | code | ba709696df7e | 53690605cd54 | 5 removed; changed hits, require |
| `hybrid_frontier/screens/e0f_refine/__init__.py` | comments | ff1c7e5cf7cb | 2c95d87cd39d |  |
| `hybrid_frontier/screens/e0f_refine/models.py` | code | afbd06b815e1 | f02cda469966 | changed H |
| `hybrid_frontier/screens/e0f_refine/refine.py` | identical | 9a8622cad8eb | 9a8622cad8eb |  |
| `hybrid_frontier/screens/e0f_refine/spec.py` | identical | 49da359bdc01 | 49da359bdc01 |  |
| `hybrid_frontier/e6/__init__.py` | comments | 66c84d0ea17c | 2ebed1d2ad82 |  |
| `hybrid_frontier/e6/plane_crop.py` | identical | 81a5c72842f4 | 81a5c72842f4 |  |
| `hybrid_frontier/e6/png_clean.py` | identical | 4b5d6dc2cd80 | 4b5d6dc2cd80 |  |
| `hybrid_frontier/e9/__init__.py` | comments | 5b9fc28bca4e | 70cd9e796b82 |  |
| `hybrid_frontier/e9/isolation.py` | code | a1f2fd8a03b3 | 902ad66acd9c | 53 removed; changed ISOLATION_FORMAT, public, require, scan |
| `fastlayer/__init__.py` | comments | 6f8cdfe64843 | 8f7f2b359b40 |  |
| `fastlayer/appearance.py` | identical | 8da32c6fa7a5 | 8da32c6fa7a5 |  |
| `fastlayer/check.py` | identical | 16fd1b88719d | 16fd1b88719d |  |
| `fastlayer/frame.py` | identical | 3ddf868e7919 | 3ddf868e7919 |  |
| `fastlayer/icp.py` | identical | 88e2cd2b03bd | 88e2cd2b03bd |  |
| `fastlayer/layer.py` | comments | aea8ced7da31 | a716d42000e6 |  |
| `fastlayer/memory.py` | identical | 28c2b0c8bc01 | 28c2b0c8bc01 |  |
| `fastlayer/onboard.py` | identical | fa2b16db37e5 | fa2b16db37e5 |  |
| `fastlayer/refine_data.py` | identical | d2c87ba7a966 | d2c87ba7a966 |  |
| `fastlayer/segment.py` | code | 0fbfcc9abe83 | 4b45f72e22de | 8 removed; changed DEFAULT_CACHE, DEFAULT_COLOUR, DEFAULT_MEMORY |
| `fastlayer/symmetry.py` | identical | 37cd176f0089 | 37cd176f0089 |  |
| `fastlayer/wide.py` | code | 334c227d2960 | 71cf48154ca4 | changed _RUNS |
| `fastlayer/scorers/monty_colour.py` | code | fc7c369b7c1a | dadd7fb7c49f | changed DEFAULT_COLOUR_NPZ |
| `fastlayer/scorers/render_compare.py` | identical | 0e6ffbc2815d | 0e6ffbc2815d |  |
