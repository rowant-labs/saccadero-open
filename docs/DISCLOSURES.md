# Disclosures

Everything we know that a reader should weigh before quoting a number from this repository. The first section
is the short list; the rest is the full one. Where a statement rests on something only we can see, it says so.

## Before you quote the number

1. **Not the kind of system the benchmark was made for.** LIBERO-Pro2.0 is, by its own title, a benchmark for
   evaluating the robustness of embodied foundation models, and the entries on its board are learned models. This
   is a program of geometry and rules, written for the benchmark's tasks. Its score shows how far such a program
   gets on the same episodes. It does not measure the robustness of a general model.
2. **Simulation only.** Both runs are episodes in the benchmark's simulator. Nothing here has been run on a robot.
3. **Developed on the training split's perturbations.** The program practised the training tasks under 41 of the
   42 perturbation types. The models on the public board were, as far as we can tell, not trained on them.
4. **It uses the benchmark's object models.** Object shapes and fixture joints are built from the benchmark's own
   asset files. A learned policy does not get those.
5. **Not submitted, not a placement.** Both runs fall between the best (81.7) and the second (75.0) of the 20
   models on the public board as we read it on 2026-11-17. That is where the number sits beside the board, not a
   rank among its models. Nobody at the benchmark has checked it.
6. **Two runs, always together.** The second is a second attempt on a test split that was no longer unseen. The
   two are level.
7. **The headline depends on the formula.** 81.1 by the weighting the board's own averages follow, 81.2 by the
   formula we registered before the runs. `docs/RESULTS.md` gives four readings.
8. **Our own runner and scorer.** The benchmark's evaluation service was not used. We step the benchmark's
   environment class inside our own process.
9. **Pictures at 1024 pixels.** The protocol allows 16 to 1024 and defaults to 256. No registration or lock
   recorded this setting; it was set in the script that started the runs.
10. **One perturbation type never reached the program** (D21, the instruction reworded during the episode), and it
    scored 100.
11. **Stored knowledge of the benchmark's scenes** is written into the code and the parameters: see below.

## How it was developed

- The plain versions of the eight test tasks were developed on throughout, and were run at the scored start state
  under the scored seeds before each run. Plain episodes are not in Overall. Runs on the original LIBERO in October
  used the plain versions of all Object and Spatial tasks.
- Development was aimed at the eight test tasks by their public names: which skills they need was read from those
  names, and practice episodes were built to exercise those skills in training tasks' scenes with our own
  instructions.
- No perturbed test episode was planned, rendered or run before the first registration was frozen. Four development
  episodes on 2026-11-17 came close to that line: one test task's instruction was run in a training task's scene
  under two perturbation types, twice each. No test file was opened; they were stopped and decided nothing.
- The blind type S06 was played blind in both runs: its case files were never opened or run in development. It
  counts in Overall (12.5, then 39.5). Before the second attempt we had read, in a training task's scene file, that
  S06 replaces the object by "same class or unseen instance".
- Start state 8 of the training tasks was the tuning set. The scored start state (0) of the training tasks was also
  run during development, with seeds 142, 143 and 160 to 173.
- The first configuration is one bundle of options registered together, not one change tested at a time. Its
  parameter text has 34 top-level entries, 61 entries for the agent and 137 values in all. Not every option was
  shown to help on its own.
- A change entered a candidate after a paired run on one machine, under a rule that rejects a change only when it
  loses more episodes than chance allows. That rule lets neutral changes in.
- The lag options are not silent outside lag cases: about one otherwise untouched episode in ninety ran differently.

## What is stored knowledge of the benchmark

- **Object models and fixture joints** come from the benchmark's asset files (the object libraries that
  `scripts/build_libraries.py` builds from your download). Building them compiles the plain scenes of every task of
  three suites and one of the fourth, the plain scenes of test tasks among them, and reads each object's shape,
  joints and resting pose from the compiled scene.
- **Each suite's and each table's camera** is stored (`rowant_lab/pro2/colour_cameras.json`,
  `colour_cameras_long.json`). They were measured in the simulator on plain tasks, some of them plain test tasks.
  (A note inside the first file once called four of those "training" tasks; they are plain versions of test tasks.)
- **One cabinet's remembered place**, each suite's list of fixtures to search for, and the routes of the model
  search are in the parameters.
- **Skills and instruction rules** are written in code: grasping, placing, opening a drawer, sweeping a cover
  aside, doing two steps in order, and a rule-based reader of the benchmark's instruction sentences. There is no
  language model.
- **The depth nets were trained on renders of the benchmark's scenes**, the eight plain test scenes among them.
- So "nothing benchmark-specific" would be false. What is general is the way it works: it sees by geometry, plans
  by geometry and checks its own work.

## The second attempt

- It is labelled a second attempt wherever it appears. The test split was no longer unseen, and the first run's
  totals by type and by task steered the work. No test episode's row, picture or record was read for development;
  we read counts.
- Its main addition, a learner for objects the program does not know, was widened knowing what S06 does and what
  it had scored, from the eight test tasks' public instructions. It was measured only on training tasks and on our
  own scenes, whose numbers are not benchmark numbers.
- Its form was chosen on one machine's rows and confirmed on another machine with the same episodes: a second
  measurement, not a fresh sample. One form was turned down first. One confirming line was looser than first set
  (0.05 for 0.01). The step before it met two of its five lines exactly at the line.
- One of its options was measured on 12 practice episodes (won one, lost one).
- It lost about 140 episodes in eight type-and-task cells where the first run had done well. One cause shows in the
  rows' counts (the learner starting on objects the program already knew); a second is a guess we did not look
  into. Practice on the training split had shown neither.
- Three of its rows ended in an error of our perception server and count as failures.
- Our forecasts before each run missed: 65 to 79 before the first, 81.5 to 83.5 before the second.

## How the runs differ from the benchmark's official path

- **Runner.** The benchmark provides an evaluation service: a policy answers over HTTP, a scheduler starts a fresh
  process per episode, and results are exported and signed. We did not use it. Our runner loads the benchmark's
  environment class from the download into its own process and steps it. Our list of episodes equals the
  benchmark planner's (checked both times).
- **Scorer.** Ours (`rowant_lab/pro2/scored.py`). See `docs/RESULTS.md` for how its formula differs from the board's
  weighting.
- **The instruction is read once.** The benchmark's runner hands the instruction to the policy at every step; ours
  reads it at the start of the episode. Type D21, which rewords it during the episode, therefore changed nothing
  for our program.
- **Refused episodes.** The simulator refused to build 14 episodes in the first run and 17 in the second (S06, one
  Object task) after one retry with the same seed. They are counted and left out as registered. The benchmark's
  service retries in a fresh process; ours retries in the same one. Whether the refusals are the benchmark's or an
  effect of our runner was never tested.
- **Versions.** Python 3.11 with newer numpy and torch, against the benchmark's reference of Python 3.8. In the
  environment the episodes ran in, MuJoCo (3.2.3) and robosuite match the reference. The object libraries were built
  with MuJoCo 3.3.2, on another machine (see Repeatability).
- **Parity with the reference environment was measured once**, on 2026-11-07: 40 chains of actions from an earlier
  form of the program that read the simulator's depth, Object and Spatial only; the same outcome in 40 of 40. It was
  never measured for the configurations that were scored, nor for the Goal and long suites.
- **Per-type comparisons** with the board take our D20 and D21 as the board's D19 and D20, by order. Board numbers
  are as printed, to one decimal.

## Repeatability

- Outcomes depend on the GPU type: the same 80 plain episodes gave 76 to 77 solved on an RTX 4090 and 69 on an
  A6000. Both runs used rented RTX 4090 machines.
- Two RTX 4090 machines with host processors of different families (an EPYC and a Ryzen) do not send identical
  commands: 43 of 60 plain episodes identical with all 60 the same outcome; on the practice list 211 of 476
  identical with the same total. Each run used machines of two processor families. A rerun should come close and
  will not match episode for episode.
- Two machines of one family did agree. A clean clone of this repository with our libraries, on two rented RTX 4090
  machines (EPYC 7763 and EPYC 7452), sent the same commands in 166 of 168 episode runs, and on each machine all
  120 practice episode runs had the recorded outcome and step count.
- On one machine, two runs of one configuration differed in commands in 5 of 476 episodes and in no outcome. (A
  registration said such a run "repeats exactly"; that was written before this was measured.)
- No interval for Overall was measured.
- The first run's four machines were not machine-checked: the check (`scripts/machine_check.py`) dates from the
  day after, when one rented machine was found to compute wrongly without any error. The first run's rows show no
  sign of that fault. The second attempt's machines passed the check twice.
- The exact runs need the two registered depth nets. A retrained net would give slightly different numbers, and
  retraining cannot rebuild our two files (`docs/DEPTH-NETS.md`).
- **The object libraries depend on the machine that builds them, and the outcomes depend on the libraries.** Ours
  were built in stages on an Apple-silicon Mac, the first stage from the original LIBERO release, with MuJoCo 3.3.2
  (the runs' workers used MuJoCo 3.2.3 on Linux). This repository rebuilds them from the download. Built on an
  Apple-silicon Mac they are ours, every array bit for bit on two Macs. Built on Linux x86-64 with the same package
  versions, 87 of the 306 files differ: in the last digits of the mesh vertices to begin with, and visibly in some
  of what is built on them (the match templates, one object's stored grasps). The program is sensitive to that. Of
  180 practice episodes the outcome differed in 9 with v9 (155 solved, 156 recorded) and in 7 with v11 (157
  solved, 158 recorded), and about half of the episodes took another number of steps. The plain test tasks were
  solved either way, 24 of 24. We did not run the test split with Linux-built libraries, so the Overall they would
  give is not known. `docs/CHANGES-FROM-AS-RUN.md` has the detail.

## What cannot be checked from outside

- That each registration was fixed before its run rests on timestamps in a private repository. The lock files
  published here carry the checksums that were fixed; the commit they name is private.
- A registration said that every episode that was run again would be listed in the report. The reports give counts
  only.
- The code published here is carved out of a larger private repository and differs from the code as it ran in the
  ways listed in `docs/CHANGES-FROM-AS-RUN.md`. The two runs also ran two different states of the code (19 files
  changed between them); this repository holds the later one.
- The costs we quote for the runs are rental rate times time, not billed totals.
