> **How to read this file.** This is the registration as it was frozen before the run, written by Claude (the "I" in
> the text) for the lab's owner. It is published with a small number of removals, each marked `[removed]` and listed
> at the end: quotations of the owner, names of private working documents, and names of private runs and machines.
> Nothing else is changed, so it still says some things we now know to be wrong; those are listed at the end too.
> The lock file `docs/records/SCORED2.lock.json` holds the checksum of the unedited text (`272cf0095844e4d5f452ea106def74719a1950f876e8dc79cf41e5a6627751b9`), which this
> edited copy cannot have.

# The second attempt on LIBERO-Pro2.0: registration

**Status: FROZEN** on 2026-11-20, on the owner's go [removed]. When this was frozen nothing of the
test split had been touched since the first run: no test episode planned or run again, and no row of the first
run opened for development.

**What this is.** A second run of the benchmark's test split, by the same protocol as the first, with candidate v11.
The first run (2026-11-17, candidate v9, Overall 81.2; `REGISTRATION-scored.md`, [removed]) stays the
registered result. This one is a **second attempt and is labelled so wherever it is reported**, with the first
run's number beside it. The test split is no longer unseen: we know the first run's totals by type and by task.

What is frozen, by name, once this says FROZEN: the commit that holds this document as it then reads
(`SCORED2.lock.json` records it); the run parameters `candidate_v11.run_params()` at that commit, which are
candidate v11 (sha256 `d03ce124911cdb8e09b626e856b918131775ac67a1b8e54286bc1f3a781f2f94`, byte for byte the text its
confirming pod ran on 2026-11-19 and 20); the two depth nets of the first run, unchanged
(`runs/pro2/colour-long/net/last.pt`, sha256 a027caad2b48…; `runs/pro2/colour2/net/last.pt`, sha256 895ca388a192…);
the sealed bundle's evaluation archive; and the first run's own lock (its sha256 is in the second lock).

## What is registered

- **The same run as the first, with another configuration:** `--split test --group all --seed 142 --num-seeds 30
  --init-state-ids 0`. 339 held-out instances (the eight base tasks plain and their 331 shifted versions), seeds 142
  to 171, start state 0: 10,170 episodes. The plan is the first run's plan, byte for byte (its sha256 is checked
  when it is written again).
- Colour pictures and the robot's own readings, relative control, the benchmark's step caps, no demonstrations.
- **Scored as the board scores**, by the first run's scorer: Overall is the mean over the 42 types. The plain
  episodes are reported apart.
- **S06 is played blind again.** No S06 file has been opened or run, in practice or since the first run. What we
  know of it is in the disclosures.
- RTX 4090 pods, as the first run. Every pod passes the machine check (`pod_selfcheck.py`) before the upload and
  again after set-up; a pod that fails either is ended and replaced before it runs an episode. The slow kind of
  host is turned away at the door. Each pod's processor is recorded.
- Before its part of the plan each pod runs a ten-episode smoke with the frozen parameters, on training tasks
  and our own scenes only. It starts its part only if the smoke's gate passes: every row home, no error, the
  learner starting where it should. The smoke's rows are no part of any score.

## The configuration: candidate v11

v9 as it was scored, and three additions, each of keys the configuration before it does not name
(`rowant_lab/pro2/candidate_v10.py`, `candidate_v11.py`; no value of v9 is changed):

| Part | What it adds | Where it was measured |
|---|---|---|
| The learner | an object with no accepted model is looked at from above and from the wrist, learned in the episode and handled from what was learned; a thing the look takes for one of the robot's own types goes to the robot's own plan | three registered runs on our own second world ([removed]; the third, W2-LEARN-E, passed both keep rules on 11-19) |
| Objects in the way | a look for the way when the carry begins, one look at a stall, the empty hand's way, the detour, the front camera read from the still scene with two guards | paired runs on practice episodes ([removed]) |
| The learner's wider start (new in v11) | the learner as registered starts only on "pick the X and place it ..."; now also on "put the X on the Y" after the robot's own attempts have failed, and where two objects of the asked type stand in the scene and the accepted one is clearly not the one asked for | our own table scenes with new objects and the practice list, one pod, lines fixed beforehand ([removed]) |

Everything else is v9: the robot's other code paths, both depth nets and the object libraries are the first run's.

With v9 there is no learner. With v11 it can start in six of the eight test tasks: Object 1 and 8, Long 5, Goal 6,
Spatial 0 and 8. Not in Goal 3 (two steps) and not in Long 8 (the instruction asks for both objects of a type).

## What is known before the run (development; none of it is a score)

- **v11 against v10 on one pod** (11-20; five lines and a rule between two forms, all fixed beforehand, all met;
  [removed]): 420 to 421 of 476 practice episodes (89.0 to 89.6 by the board's formula on
  that list); our 53 table scenes with new objects 1 to 19; our second world's development part 80 and 80 of 100;
  the eight test tasks' plain scenes 40 of 40.
- **v10 against v9, on two pods:** 411 to 415 of the 476 practice episodes on its confirming pod (87.6 to 88.1 by
  the formula), and 411 to 420 on a pod of another kind (87.6 to 89.0; read after the fact, not a line of that
  pod). Episodes with a slab in the way 43 to 55 of 73; the camera-move type 14 to 16 of 16; the second world's
  new objects 11 to 45 of 60.
- **v11 against v9:** 411 to 421 of the 476 practice episodes, won 13, lost 3 (87.6 to 89.6 by the formula). The
  two ran on two pods of one kind; v10 ran on both and gave the same outcome in all 476 episodes.
- **The learner alone, registered** (W2-LEARN-E): practice 411 and 411; held-back new objects 5 to 38 of 60 (that
  part's second read).
- **The first run by type** (the types this run is most likely to move): S06 12.5, D05 22.9, S05 43.3, D07 90.8.
- **The first run's S06 by task** (from its report): 30 of 226, all 30 on Long 8. v11's learner does not start
  there.
- **The public board** as read on 11-19 (20 models, unchanged since the first run): first 81.7, second 75.0.

### The pre-flight and what followed (11-19 and 11-20)

1. **New objects in the table scenes.** Until 11-19 the learner had acted on a learned object only in scenes of
   one kind (the Object suite's floor and basket). [removed] puts three of the second world's
   development objects into the plain scenes of 18 training tasks of the Spatial, Goal and Long suites (53
   scenes; each can be solved, checked by hand in the simulator).
   - 11-19, 16:50: with v10 the learner started in none of them. It started only on instructions of the form
     "pick the X and place it ...", so in at most three of the eight test tasks. The owner: [removed].
   - 11-19 evening, pod [removed]: three new starts with the look first. Table scenes 1 to 17 of 53; practice
     list 420 to 417 of 476. By its line that form was turned down.
   - 11-19 night to 11-20, pod [removed]: the form in v11, and the same form without the two-of-a-type start, each
     against v10. Both met all five lines; the rule fixed beforehand took the form with both starts. Table
     scenes 1 to 19 of 53: the "put" scenes 0 to 10 of 12, the two-of-a-type scenes 0 to 8 of 23, the two-step
     scenes 1 and 1 of 18 (no start there).
2. **The eight plain test tasks** at the scored start under five of the scored seeds (40 episodes): 40 of 40
   with v11 and 40 of 40 with v10, on the same pod. The learner started in none. v9 had 240 of 240 under all
   thirty seeds in the first run.

## Claims, stated before the run

- **C1.** Overall is not lower than the first run's 81.2.
- **C2.** The blind type, S06, scores above the first run's 12.5. This is the learner's first test on objects that
  are not our own.
- **C3.** The two types with a thing in the carry's way (S05 and D05) score above the first run's mean of the two
  (33.1).
- All three are reported whether they hold or not, each type beside the first run's number and the public board's
  best as read on the day of the run.

**What I expect, written before the run** (11-20): Overall between 81.5 and 83.5, most likely about 82.5.
- The objects in the way: about half a point to a point, if practice carries over (on practice the two slab
  types went from 17 to 23 of 31 episodes on one pod, and the slab list from 43 to 55 of 73 on another).
- The blind type: about half a point to a point. Of its 226 episodes the first run solved the 30 of Long 8. If
  the learner does on the benchmark's replaced objects about what it does on ours, it adds 40 to 90 episodes in
  the other tasks, which would put S06 between 30 and 55. That "if" is the weakest part of this guess: S06's
  objects are not ours, and the learner has never run in the scene of Long 5.
- What the three additions cost on the other types has not been measured on test tasks. On practice v11 against
  v9 won 13 episodes and lost 3.
- A number under 81.2 is possible and would be reported the same way.

## Rules for the run

- An episode is run again only after an infrastructure failure (a pod lost, the simulator refusing to build), with
  the same seed. Every such episode is listed in the report.
- The run goes to its end. No partial result changes anything.
- **Both runs are reported together, always:** the first as the registered run, this one as the second attempt.
  Neither replaces the other, whichever is higher.
- Nothing from the test rows of either run is used for development afterwards without the owner's word.
- No third attempt without a new registration and the owner's word.

## Disclosures

Every disclosure of the first registration still holds. Added for this attempt:

- **This is a second attempt.** The first run's totals by type and by task were read for its report. They are part
  of why the work since then went where it did: the learner (the blind type scored 12.5, and unknown objects were
  the weakness named in the pre-run brief) and the objects in the way (22.9 and 43.3). No test episode's row,
  picture or record was read for development. Development ran on the training tasks under perturbation, on our
  own second world and on our own table scenes.
- **S06.** Its case files have never been opened or run. On 11-19, while building the pre-flight scenes, I read a
  training task's plain scene file, which lists every perturbation of the catalogue; its S06 entry says the task's
  object is replaced by "same class or unseen instance" (an asset swap). Before that our notes said only that S06
  "probably swaps in" an object with no model. The table scenes were designed before that reading and use our
  own second-world objects, which are not of the replaced object's class.
- **The learner's wider start is development done knowing what S06 does and what it scored.** It was built
  because the pre-flight showed the learner could start in three of the eight test tasks. Which starts to build
  was read from the instructions of the eight test tasks, which are public. They were measured on training tasks
  and our own objects only. On 11-20, for the expectation above, I read in the first run's report where its
  solved S06 episodes were (all on Long 8); the starts had been built and measured before that.
- **One form was measured and turned down before this one** (the look first on all the new starts; it lost three
  practice episodes). The form in v11 was chosen from that pod's rows and then confirmed on another pod of the
  same kind. The practice list and the table scenes are the same episodes on both pods, so the confirming pod is
  a second measurement of a choice made on the first, not a fresh sample.
- **The confirming pod's second line was looser than the earlier pod's:** a sign test under 0.05 where the
  earlier pod's line was 0.01, so that the form with one start (12 scenes) could meet it. Both forms came in far
  under 0.01 (3.8e-06 and 0.00098).
- **The two-of-a-type start on known objects:** it acted in 12 practice episodes, won one and lost one, the same
  two on both pods. On the test split it can act in Spatial 0 and 8 under every type. What it does to known
  objects there is not measured beyond those 12.
- **The table scenes are our own test:** three objects, 18 training tasks, one start state. Their numbers are not
  benchmark numbers, and the form was chosen on them.
- **The plain test scenes** were run with v10 and v11 at five of the scored seeds before the freeze (they are not
  in Overall). The plain versions of the eight test tasks were developed on throughout, as the first registration
  says.
- **v10's confirming pair met two of its five lines exactly at the line** (four more practice episodes won than
  lost, where four was the line; three looks with a new thing after a camera move, where three was the most
  allowed). On a second pod of another kind it was 411 to 420, read after the fact.
- **One interplay inside v10 was measured only once:** the learner's server setting also acts on the looks for the
  way. Against the same options without the learner, on another pod of the same kind, it cost two practice
  episodes and won four slab episodes.
- **The second world is our own test.** Its new-object scenes are of one kind, and its numbers are not benchmark
  numbers.
- **Rented machines.** One machine rented on 11-18 divided wrongly on one core with no error. Every pod is now
  tested before it runs anything. The four pods of the first run were never tested this way; its rows show no
  sign of the fault.
- **A pod's processor changes the commands.** v9 on an EPYC and on a Ryzen host gives the same commands in 211 of
  476 practice episodes and the same total. v10 solved 415 of the practice list on a Ryzen pod and 420 on two
  EPYC pods. The report says what ran where.

## Steps

1. This text to the owner.
2. On the owner's go: this document says FROZEN and is committed and pushed; `scored.py lock --attempt 2` writes
   `SCORED2.lock.json`; a tag marks the frozen state.
3. `scored.py unpack --attempt 2` and `plan --attempt 2` (the first run's plan, checked byte for byte), then
   `shards --attempt 2`.
4. The run: four RTX 4090 pods with ten workers each, the machine check on each, the supervisors of the first run.
   From the first run's cost ($17.40) and v11's time an episode on practice (a quarter more than v9's): about six
   hours on four pods, about $22.
5. The report: Overall, by suite, by type and by task, each beside the first run and the public board.

## Decided by the owner

- 2026-11-19: [removed] (the owner asked whether to run now or to improve something first). My
  answer: go, after the pre-flight.
- 2026-11-19, about 17:00, after the pre-flight's finding: [removed] (fix first). The learner got the other forms of task
  before the run (candidate v11, confirmed 2026-11-20).
- 2026-11-20, about 10:30: [removed] The go on this text. Just before it the owner asked whether a
  small language model reading the instructions would help the learner. My answer: reading is not where it
  loses, and v11 runs with no language model in the control loop, as the first run did.

---

## Added for publication

**Removed, and why.** Five quotations of the owner (one is replaced by a plain statement of what was asked). The
names of two private working reports, one private registration, one private tool and two rented machines. Nothing
else.

**What this text says that we now know to be wrong or loose.**

- "Scored as the board scores": see the note at the end of the first registration. By the registered formula the
  two runs are 81.16 and 81.17; by the board's own weighting both are 81.12.
- Claim C1 ("not lower than the first run's 81.2") holds on the registered formula and on the board's weighting, by
  0.015 and 0.002. It does not hold per episode, or when the episodes the simulator refused are counted as failures.
  "Level" is the right word.
- "Every such episode is listed in the report": the report gives counts only.
- "that part's second read" counts only this learner's reads. The held-back part of our own second world had been
  run and read at nine milestones in all. That world, its registration and its table scenes are tests of our own
  and are not part of this release; their numbers are not benchmark numbers.
- "by the board's formula on that list" means the registered formula, on the practice list.
- The expectation written here (81.5 to 83.5) missed: the result was 81.17.

The full list of what a reader should know is `docs/DISCLOSURES.md`.
