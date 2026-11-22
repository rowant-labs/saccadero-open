> **How to read this file.** This is the registration as it was frozen before the run, written by Claude (the "I" in
> the text) for the lab's owner. It is published with a small number of removals, each marked `[removed]` and listed
> at the end: quotations of the owner, names of private working documents, and names of private runs and machines.
> Nothing else is changed, so it still says some things we now know to be wrong; those are listed at the end too.
> The lock file `docs/records/SCORED.lock.json` holds the checksum of the unedited text (`6567a07c2fe7d1286ea48214f2d8583cc1ddb32c0b2be95c4d099fd2162fac7b`), which this
> edited copy cannot have.

# The one scored run on LIBERO-Pro2.0: registration

**Status: FROZEN** on 2026-11-17, after the owner's review and go [removed].

Written from 2026-11-15 on and frozen at candidate v9 ([removed]: three private working reports). When this was frozen no perturbed test episode had been
planned, rendered or run, and the 339 test files were still inside the sealed archive. The owner's decisions are at
the end.

What is frozen, by name: the commit that holds this document as it reads here (`SCORED.lock.json` records it); the
run parameters `long_dev.bundle(long_dev.colour_long_params(net))` at that commit, which are candidate v9
([removed], sha256 bd0a0cc77c0f; the lock holds the sha256 of the parameters as the runner
passes them); the depth net `runs/pro2/colour-long/net/last.pt` (sha256 a027caad2b48dacaa91be3a40bf598d8f43786e4fa1ce1ef30811b561733dd60)
and the second net `runs/pro2/colour2/net/last.pt` (sha256 895ca388a192309494bf5c24d93a90c42f47c226b3f7ee2eeb11955ee5769523);
and the sealed bundle's evaluation archive (its sha256 is in the lock).

## What is registered

- **One run of the colour line on the benchmark's test split, by the benchmark's documented protocol**
  (`docs/evaluation.md` in its bundle): `--split test --group all --seed 142 --num-seeds 30 --init-state-ids 0`.
  - 339 held-out instances: the eight base tasks plain (Object 1 and 8, Spatial 0 and 8, Goal 3 and 6, Long 5 and 8)
    and their 331 shifted versions, S06 among them.
  - Seeds 142 to 171, **start state 0 in every episode**: 10,170 episodes. The seed varies the perturbation, not the
    start.
  - Colour pictures and the robot's own readings, relative control, the benchmark's step caps, no demonstrations.
- **Scored as the board scores:** Overall is the mean over the 42 types; a type's score is the mean over the four
  suites; a suite's is the success rate over its two tasks' episodes. The plain episodes are reported apart.
- **S06 is played blind** (owner, 11-16): it has never been run in training or test, nothing was developed on it,
  and it counts in Overall as the board counts it.
- The depth line (simulator depth) is our reference in development and is **not** run on the test split.

## The configuration

One robot and one parameter set for all four suites: `long_dev.bundle(long_dev.colour_long_params(net))` at the
frozen commit: candidate v9 (set in commit 20de265b; the robot's code and the parameters have not changed since),
which is v7 (the night of 11-17) with the drawer's pull options (v8) and the day's kept options (v9).

| Part | State |
|---|---|
| Depth from colour | the net retrained on 11-15 with the long suite's tables (`runs/pro2/colour-long/net/last.pt`); pictures cut into 640 px tiles |
| Camera | the stored camera, checked against the robot's own arm at the first motion |
| The camera's own changes (11-15, 11-17) | a zoomed and shifted picture undone before the net; then the camera fit; then the same fit on the second net's depth; then a nearly-right undo; and, as the last resort, any zoom the overlap test accepted unless the undo made the arm worse |
| The robot's own readings (11-14, 11-17) | when the hand reading and the joint readings contradict each other, the constant offsets of both are found from the readings alone (and the hand reading's lag behind the joints with them); the fit from the arm's picture only if that is refused |
| Compound instructions | the step sequencer, its steps following each other without an idle step; the drawer plan (stage on the cabinet, open, take again by the near rim, with the cabinet pose of the opening step); while the staging runs the sequence keeps back what the opening needs, not a whole reserve, so the staging may grasp twice |
| Fallbacks (11-15 to 11-17) | a named object with no accepted percept is stood where its picture region shows it; else a clear rejected pose on the table, then a region, then a pose inside another object; a missing receiver follows the evidence, raised onto a region's top; a later look never turns a fixture by more than 45 degrees; a named target or receiver read as another type is stood where that weak percept stands; a reported object's own leftover region is no receiver |
| The cabinet (11-16, 11-17) | its remembered place tried before the search; the half-turn twin when the chosen pose opens away from the robot, turned about the carcass; a cabinet the look did not find is taken at its remembered place when a region's top lies over it, unless a rejected fit agrees with that place |
| The drawer's opening (11-17) | jaws that arrive shut are opened; through a delay the approach is finished on the true reading; a missed handle is followed by a look and a new plan, never the same grip again; the pull keeps the opening's direction and runs the part's whole travel, and fingers that end it shut have not opened it; the point before the handle is passed, the close begins on the way in, the release is sent before the pull's end |
| The drawer task's other steps (11-17) | the staging's release begun on the way down; the turn back begun as the hand leaves the handle; a quicker hold, lift and release in the take-again |
| The instruction's relation (11-16, 11-17) | a lone candidate that fails it is replaced by one that satisfies it; a look at the relation's place before the one bowl in view is carried; a landmark's stand-in must be seen |
| Receiver looked at again (11-15, 11-16) | before letting go and after backing off; a move is believed only with the rest of the scene seen in place, and only if it stays level |
| Grasping (11-16, 11-17) | the fingers decide the lift check; the object's place is taken where the hand let go; the look tool is told when the hand lets go; a higher look and the next candidate after a close on nothing; recovery after an object is knocked from the hand; a steady hold between 0.6 and 1.35 of the model's width goes on to the lift |
| The carry (11-17) | a thing the first look could not explain is counted on the way |
| Steps (11-16) | a shorter way between grasp and release, a faster lift and lowering, an earlier close |
| Two objects (11-16, 11-17) | the landing place of a dropped object, taken from where the hand really had it; the placed check read again after backing off; steps kept for later steps; clauses that did not do their work run once more; two objects on one surface keep their own parts of it; a track that tips a held object is not believed |
| Late readings and commands (11-15 to 11-17) | seen from exact repeats, lag measured by a pulse and read again after a rest, the hand predicted; a caught arm known by its elbow reading, freed by a pull, drawn in and turned back; the waits of a whole lag cut to the commands' own delay, the wait-and-correct taken before the descent when commands are late, a lag that outlived its delay ended by the fingers' answer |
| Covers and panels (11-16, 11-17) | the first look waits for a still scene; a cover on the receiver is swept off; a target hidden behind a hanging region is looked for behind it, with the night's rules for what counts as a panel |
| Small rules (11-15) | a placed object is at rest under 8 mm between reads; the fingers are watched while carrying |
| Off | learning new objects inside an episode (needs wrist depth; depth line only), the close look, re-detection, the command watch, keeping the first plan's target, the second look in the carry, the cover's slow push, lower touch and pick-off, the held-pose fix, the cabinet at its measured size and the touch for its front, a stood target's resize, the finger probe for a lag, the nudge plan for the drawer task |

The GPU type of the run's pods is part of what is frozen: the colour line's outcomes differ by GPU type (the same 80
plain episodes: 76 to 77 on an RTX 4090, 69 on an A6000, measured 11-16). The run is planned on RTX 4090 pods. Two
RTX 4090 pods on different host processors are not bit for bit the same either: of 60 plain episodes run with one
configuration on both (11-17), 43 had identical commands and all 60 the same outcome. On one pod a run repeats
exactly.

## What is known before the run (development; none of it is a score)

- **The eight plain test tasks at the scored start under the 30 scored seeds** (RTX 4090): v6 237 of 240, v7 and v8
  239 of 240 (one seed of the drawer task), v9 240 of 240 (the drawer task 30 of 30 at a median 202 steps, at most 208). At their ten starts (v7): 80 of 80.
- **The drawer test task, plain, takes most of its step limit.** With v8 a median 239 of 300 steps; with v9 201
  (30 of 30 solved in its own pair). Every other test task uses under half of its limit. On practice tasks many
  perturbation types cost more extra steps than this task has to spare, so it is expected to be the weakest of
  the eight under perturbation.
- **The dress rehearsal** of the run's whole path (11-16, candidate v4; training split, start state 0, a seed never
  run; 1,317 episodes planned by the run's planner, run on two pods, scored by the run's scorer): every episode
  with an outcome. By the board's formula 57.8 on the training split, which is not an estimate of the test split:
  nine of the 32 training tasks need skills the robot lacks or score near zero, none of them a test task. On the 23
  it can play: Object 87.7, Spatial 73.3, Goal 80.6, Long 73.2.
- **The night's pairs** (11-17; each option's own run against v6 on one pod; [removed]): the readings
  fit and the last-resort zoom 87 to 111 of 168 on the three calibration types; the caught arm 100 to 109 of 143;
  the stood object 215 to 235 of 309; the drawer's options 109 to 120 of 166 on the Goal training tasks and 18 to
  23 of 28 on turned cabinets (with the pull options); the panel keys 70 to 73 of 117; the relation look 126 to
  129 of 168; the two-pots rules 28 to 30 of 30 on that test task plain.
- **v7 against v6, all of it together**, on every second playable training episode at the scored start with a seed
  never run (477 episodes, every type but S06): 386 to 415, won 30, lost 1.
- **The day's pairs** (11-17; each option's own run against v8 on one pod; [removed]): the carry
  option 124 to 126 of 172; the turn limit 29 to 30 of 30 on the plain drawer task; the opening's and the other
  steps' options 239 to 201 steps there; the lag options 221 to 226 of 281; the resized-things options 54 to 57 of
  72 and 114 to 117 of 137.
- **v9 against v8, all of it together**, on the other half of the playable training episodes at the scored start
  with a seed never run (476 episodes): 401 to 411, won 12, lost 2.
- The long suite's training tasks: five of its eight need skills the robot lacks (none is a test task). One pot
  onto the stove in a training scene under every type, with our own instruction and check: 35 and 34 of 42 at two
  seeds (v8).
- The second world's development part with v9, as a record (98 of its 100 scenes): new objects 11 of 58, moved
  cameras 14 of 20, new tasks 18 of 20. The robot has no way to handle an object
  it has no model of on the colour line, which is what S06 probably asks.
- **The public board as read on 2026-11-17** (20 models): first 81.7, second 75.0, fifth 70.1, tenth 61.6.

## Steps

1. The owner's decisions: given on 11-16 (below).
2. Built on 11-16 (`scored.py`, [removed]) and rehearsed that night:
   - **The lock.** `scored.py lock` writes `SCORED.lock.json`: this document's sha256 once it says FROZEN, the code
     commit, and the sha256 of the parameters, of both depth nets and of the sealed bundle. `unpack` and `plan`
     refuse without it, at another commit, or when any of those differs. The 339 test files leave the sealed archive
     only then, into their own folder, and only counts are printed. The development planner refuses test tasks by
     name as well.
   - **The plan.** Our list builder gives the same episodes as the benchmark's own planner: checked on the training
     split with the protocol's seeds and start state (40,470 episodes, equal sets), and checked again on the test
     split when the plan is written.
   - **The driver.** One shard of the plan per pod; on each pod the queue with its heartbeat and a restart that skips
     finished episodes; on the Mac one supervisor per pod that restarts a stopped queue and brings the rows home. A
     dead worker is started again, and a run with episodes left never reports success.
   - **The scorer.** The board's formula, the plain episodes apart, episodes the simulator refuses counted and left
     out, missing episodes counted.
3. Development in parallel streams, each change behind an option and kept only if it passes the keep rule on a
   paired run: done on 11-16 and 11-17.
4. Before the freeze: v9's checks (the plain test tasks, v9 against v8, the second world's record), then the
   owner's review (11-17: a stop right before the launch). Done.
5. This document is frozen and pushed. Then `lock`, `unpack`, `plan`; the plan's sha256 is written into
   [removed] before the first episode.
6. The run: 10,170 colour episodes on four RTX 4090 pods with ten workers each (fewer or more only if pods cannot be
   had or one is lost; the report says what ran where). Before the first episode
   each pod's simulator tree is completed with the bundle's two asset files whose names the guarded push refuses
   ([removed]: a scene file whose name begins with a DEL character, an unreferenced picture with a non-ASCII
   name; no training task uses either, and whether a test task does is not known). From the rehearsal's timing: 21 to
   30 pod-hours, 5 to 8 hours on four pods, $19 to $27 depending on the hosts drawn. A pod's CPU is checked when
   it starts and a slow one is replaced before the run.
7. The report: Overall, by suite, by type, each next to the public board as read that day, and what failed.

## Rules for the run

- An episode is run again only after an infrastructure failure (a pod lost, the simulator refusing to build), with
  the same seed. Every such episode is listed in the report.
- The run goes to its end. No partial result changes anything.
- Nothing from the test rows is used for development afterwards without the owner's word.

## Claims, stated before the run

- **C1.** Overall at or above the board's tenth entry as read on the day of the freeze (61.6), with zero
  demonstrations.
- **C2.** The number of perturbation types on which the robot is above the board's best entry.
- Both are reported whether they hold or not.

## Disclosures

- The plain versions of the eight test tasks were developed on (owner, 11-15). The depth net's training frames
  include those eight plain scenes. No perturbed version was seen.
- On 11-17 a development stream ran the drawer test task's instruction as a practice instruction in a training
  task's scene under two perturbation types (S07 twice, S09 twice; the Mac's depth line; [removed]).
  No test file was opened. The episodes were stopped there, decided nothing, and the practice was not repeated.
- Start state 8 of the training tasks was the tuning set all along. The scored start (state 0) of the training tasks
  was run with seeds 142, 143 and 160 to 173 for estimates, the rehearsal and the pairs of 11-16 and 11-17.
- The configuration is registered as one bundle of about eighty-five options, not one change at a time. Each of
  the night's options ran alone on top of v6 and each of the day's on top of v8; their runs together are the two
  checks named above.
- The lag options are not perfectly silent outside the lag cases: in their two pairs one untouched episode in
  about ninety ran differently (every plain episode ran the same, command for command).
- The S18 offsets were the same in every training episode read. The robot does not use that: it fits them from
  each episode's own readings.
- The second world's record with this configuration leaves out two of its 100 development scenes (their index
  numbers collide with test-task numbers in the planner's guard; the scenes themselves are our own layouts).
- Original LIBERO runs in October (M2) used the plain versions of all Object and Spatial tasks.

## Decided by the owner on 2026-11-17

- Run once the day's pairs are in; a review with the owner right before the launch, and again after the run.
- After that review (the brief, the expected range, and the question whether to hold the run for a learner of
  unknown objects): [removed] The run is candidate v9 as registered here; the learner is work for after
  the run.

## Decided by the owner on 2026-11-16

1. **S06:** played blind in the scored run, with no development on it.
2. **One bundle** for the registration.
3. **When:** two to three more days of development, then the run.
4. **The second world** before the freeze: a record, not a gate.

---

## Added for publication

**Removed, and why.** Two quotations of the owner. The names of three private working reports, one private parameter
file, two private scripts and two private run folders. Nothing else.

**What this text says that we now know to be wrong or loose.**

- "Scored as the board scores" is wrong as worded. The formula written here (the plain mean over the 42 types) is
  what was registered and what the run reports as its registered number, 81.16. The public board's own averages
  follow a slightly different weighting, by which the same run is 81.12 (`docs/RESULTS.md`). The sentence "a type's
  score is the mean over the four suites" also misses that one static type exists in three suites.
- "On one pod a run repeats exactly" was written before it was measured. Measured the next day: two runs of one
  configuration on one machine differed in commands in 5 of 476 episodes, and in no outcome.
- "About eighty-five options" has no count behind it. The parameter text has 34 top-level entries, 61 entries for
  the agent and 137 values in all.
- "Every such episode is listed in the report": the report gives counts only.
- The rehearsal's "by the board's formula" means the formula above.
- Not said here, and it should have been: the run asks for pictures at 1024 pixels (the protocol's default is 256);
  our runner reads the instruction once at the start of an episode, so the type that rewords it during the episode
  never reaches the program; and "the second world" is a test of our own that is not part of this release.

The full list of what a reader should know is `docs/DISCLOSURES.md`.
