# saccadero-open

A robot program for the [LIBERO-Pro2.0](https://arxiv.org/abs/2610.24587) simulation benchmark, with the records of
two registered runs on the benchmark's test split. It is the open part of Saccadero, the robotics work of Rowant
Labs.

**What it is.** A program that controls a simulated robot arm from colour pictures and the arm's own readings. It
was written without demonstrations, it has no policy network, and no language model sits in its control loop. The
only neural networks are two small nets that estimate depth from a colour picture. Everything else is geometry and
rules: it fits stored object models to what it sees, reads the instruction with rules, plans grasps and motions by
geometry, and checks its own work as it goes.

**What it is not.** It has never run on a real robot. It is not a general policy: it uses the benchmark's own
object models, and a good deal of knowledge about the benchmark's scenes is written into it. It contains no code
of the [Monty](https://github.com/thousandbrainsproject/tbp.monty) project (the project it grew out of), though a
few functions follow Monty's mathematics and are credited in `THIRD_PARTY_NOTICES.md`.

## The result, with its labels

Two runs of the benchmark's test split (10,170 episodes each), each registered before it started.

| | First run (2026-11-17) | Second attempt (2026-11-20) |
|---|---|---|
| Configuration | v9 | v11 |
| **Overall, by the public board's own weighting** | **81.1** | **81.1** |
| Overall, by the formula registered before the runs | 81.2 | 81.2 |
| Plain (unperturbed) tasks, not in Overall | 240 of 240 | 240 of 240 |

- **Developed on the training split's perturbations.** The program practised the training tasks under 41 of the 42
  perturbation types. As far as we can tell, the models on the public board were not trained on them.
- **Uses the benchmark's object models.** Object shapes and fixture joints come from the benchmark's asset files.
- **Simulation only.**
- **Not a board placement.** Nothing was submitted and nobody at the benchmark has checked these numbers. Beside
  the board as we read it on 2026-11-17 (20 models, best 81.7, second 75.0) either run would stand second.
- **Both runs are always reported together.** The second is a second attempt on a test split that was no longer
  unseen. The two are level; the second is not an improvement.

[`docs/RESULTS.md`](docs/RESULTS.md) has every number: four readings of Overall, each perturbation type, each task.

## Before you quote the number

1. It was run with **our own runner and scorer**, not the benchmark's evaluation service.
2. It asked for **pictures at 1024 pixels**. The protocol allows that; its default is 256.
3. **One perturbation type never reached the program.** It rewords the instruction during the episode, and our
   runner reads the instruction once. That type scored 100.
4. The **plain versions of the eight test tasks were developed on**, and the depth nets saw those eight scenes.
5. One type (S06) was **played blind**. The second attempt was built knowing what that type does and what it had
   scored.
6. The simulator **refused to build** 14 and 17 episodes. They are left out as registered; counted as failures the
   second attempt is 81.0 by the registered formula.
7. **Outcomes differ by GPU type and by host processor.** No error bars were measured.
8. The runs **cannot be checked from outside** in one respect: that each plan was fixed before its run rests on a
   private repository's timestamps.

The full list is [`docs/DISCLOSURES.md`](docs/DISCLOSURES.md). The two registrations, as frozen, are
[`docs/REGISTRATION-1.md`](docs/REGISTRATION-1.md) and [`docs/REGISTRATION-2.md`](docs/REGISTRATION-2.md).

## How it works

- **It sees by geometry.** A perception server holds a model of each object type and fits those models to the
  picture and its depth: where each object is and how it is turned. Depth comes from the two depth nets.
- **It reads the instruction with rules,** not with a language model: which object, which place, one step or two.
- **It plans by geometry:** a grasp from the object's model, a path for the hand, a place to let go.
- **It checks its own work.** Did the fingers close on something, did the object move with the hand, is it at
  rest where it should be. If not, it looks again and tries another way.
- **It measures what it cannot trust.** From the first motions of its own arm it checks the camera's place, the
  offsets in its joint readings and the delay of its commands, and corrects for them.
- **v11 adds a learner.** An object it has no model of is looked at from a few sides, and a model is built on the
  spot.

What is stored knowledge of the benchmark, rather than seen or worked out, is listed in the disclosures: object
models, each scene's camera, one cabinet's place, the lists of fixtures to search for, and the skills themselves.

## Where it loses

The blind type (12.5, then 39.5 of 100), four more types under 50, and the Goal suite's drawer task. The second
attempt gained on the blind type and lost about as much elsewhere. See `docs/RESULTS.md`.

## Reproduce

**What you need.** Linux, one NVIDIA RTX 4090 (other GPU types give somewhat different outcomes), 16 or more
processors, about 5 GB of memory and up to 2.6 GB of GPU memory per worker (the scripts start eight), about 20 GB
of disk. The benchmark's evaluation download, which you
get from the benchmark yourself. Our two depth nets, which are release files of this repository.

**Get the benchmark first.** Download and prepare it as its data card says
(`hf download zhouxueyang/LIBERO-Pro2.0-Eval --repo-type dataset ...`, then its `scripts/prepare_evaluation.py`).
That leaves a folder `LIBERO-Pro-evaluation` holding `simulator-source` and `service-source`: this is
`ROWANT_BENCHMARK`. The folder you downloaded into, which holds `evaluation.zip`, is `ROWANT_BUNDLE`. We used the
dataset at revision `d1933228` (benchmark source commit `4046dc47`); the archive's checksum is in the lock files. You
do not need the benchmark's own Conda environment or its service.

```bash
cp configs/paths.example.env configs/paths.env      # edit it: where the download and the data folder are
source configs/paths.env
bash scripts/setup_env.sh                           # the two Python environments, with exact versions
source "$ROWANT_DATA/env.sh"
"$ROWANT_DATA/venvs/server/bin/python" scripts/machine_check.py   # does every processor compute the same?
python3 scripts/build_libraries.py                  # the object libraries, from your download (about an hour)
python3 scripts/compare_libraries.py docs/records/scored-runtime-libraries.digest.json "$ROWANT_DATA/runs"   # yours against ours
bash scripts/reproduce_plain8.sh                    # the eight test tasks, plain: 24 episodes per configuration
bash scripts/reproduce_practice.sh                  # 60 training-split episodes, compared with our outcomes
```

The full test split is `scripts/reproduce_test_split.sh v9 --test-split` (with ten workers about 13 to 20 hours on
one such machine, 19 to 27 for v11; the script's default of eight is safer for a 24 GB card and about a quarter
slower). It asks for the flag because it touches the test split. **Please develop on the training
split**: everything else here does.

**What "reproduce" means here.** The parameters, the plan, the benchmark's archive and the two nets are the same by
checksum (`python -m rowant_lab.pro2.scored verify`). The scorer gives our published numbers from our published
rows. A fresh run on an RTX 4090 lands close to ours. It does not mean identical rows: outcomes differ by GPU type
and host processor, and the object libraries you build are close to ours but not byte for byte the same
(`docs/CHANGES-FROM-AS-RUN.md`).

**Release files.** The two depth nets (`colour-long.pt`, `colour2.pt`; their checksums are in
`docs/DEPTH-NETS.md`). The rows of both runs and their plan will be added as release files.

## What is in this repository, and what is not

`docs/CODE-MAP.md` walks through the folders and explains the odd names. In short: `rowant_lab/pro2/robot` is the
policy, `rowant_lab/libero_env` and `rowant_lab/fastlayer` are perception, `scripts/` is what you run.

Not here, on purpose:

- **The benchmark.** Not its package, not its task files, not its assets. This code uses your download.
- **The object libraries.** They are made of the benchmark's assets; you build them.
- **Everything else in the private repository this was carved from:** other research lines, development probes,
  an earlier form of the program that read the simulator's depth, a second test world of our own, the tooling for
  rented machines, working notes, and the history.
- **The self-writing loop's code** (see the next section).

## How this code was written

By Claude, Anthropic's AI model, in Claude Code sessions under one person's direction. In the private repository
all but one of the 304 commits that touch these files carry a Claude co-author line (Claude Fable 5.2 on 210,
Claude Opus 5.5 on 91, Claude Fable 5.1 on 2).

For three days in November an automated loop also wrote code: Claude Opus 5.5 read the program's failures on
training episodes and proposed changes, and a change was kept if a paired run did not lose. It ran 54 rounds and
kept 14. What it kept is **6% of today's policy code**. It would be wrong to say the robot wrote its own code.

More than half of the code predates the work on this benchmark. [`docs/HOW-IT-WAS-BUILT.md`](docs/HOW-IT-WAS-BUILT.md)
has the numbers and what we do not claim.

## What differs from the code as it ran

This is a carve-out, not the repository the runs were made from. Fixed machine paths became settings; a guard
that inspected our own machines was replaced by a stand-in that does nothing; code the runs never called was cut
where it needed unpublished modules; the lock check compares checksums instead of private commits; comments that
named people, private documents and machines were edited. The two runs also ran two states of the code, and this
is the later one. [`docs/CHANGES-FROM-AS-RUN.md`](docs/CHANGES-FROM-AS-RUN.md) lists every file with its checksum
before and after, and the test we ran to show that the carve-out acts the same.

## Licences and credits

Our code: MIT (`LICENSE`), copyright Rowant Labs. Credits and other projects' terms: `THIRD_PARTY_NOTICES.md`
(Monty, robosuite, LIBERO, the benchmark and its assets).

If you use the benchmark, cite its papers (arXiv:2610.24587 and arXiv:2510.03827). To cite this repository, see
`CITATION.cff`.

## Questions and reports

Open an issue. If you find an error in a number or a statement here, we want to know, and we will correct it in
the open.
