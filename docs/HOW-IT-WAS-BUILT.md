# How this code was written

The short answer: by Claude, Anthropic's AI model, working in Claude Code sessions under one person's direction,
plus a three-day automated loop that wrote a small part of it. The numbers below come from the private
repository's history and from the loop's own records, and they were checked by a second pass before publication.

## Who wrote what

- **The code was written by Claude models.** In the private repository, 304 commits (merges aside) touch the files
  this repository was carved from. All but one carry a Claude co-author line: Claude Fable 5.2 on 210, Claude
  Opus 5.5 on 91, Claude Fable 5.1 on 2. The one without is a one-line file from the repository's set-up. By
  lines in those files, Opus 5.5 wrote 73% and Fable 5.2 26%.
- **The person behind Rowant Labs** set the direction and the budgets, made the downloads, set the protocol, and
  gave the word for each registration and each run. The record does not show them writing code.
- **This repository** was carved out of the private one by Claude Fable 5.2 in a Claude Code session. Its commits
  carry the co-author line of the model that made them.

## It was not built in two weeks

58% of the lines date from 8 October to 6 November 2026, before the work on this benchmark began on 7 November. The
perception side (finding known objects and their poses in a picture) was built for earlier stages of the project,
on the original LIBERO and on other data. The two scored runs were on 17 and 20 November.

## The self-writing loop, and its real share

From 7 to 9 November an automated loop ran: Claude Opus 5.5, called through the API, read the failures of the
current program on training episodes, proposed a change to the policy code, and the change was kept if a paired run
did not lose.

- 54 rounds, 14 kept, 43.99 US dollars of API use.
- What it kept is 789 of the policy package's 12,627 lines today: **6%**. Another 14% is the package the loop
  started from. The remaining 79% was written afterwards, in sessions.
- It ran on a form of the program that read the simulator's depth. It never ran the colour configurations that
  were scored.

So "the robot wrote its own code" would be wrong, and so would "the loop wrote the policy". The loop's code is not
part of this release.

## After the loop

- From 16 to 19 November, parallel Claude agents each worked on one failure pattern in its own copy of the code. A
  change entered a candidate after a paired run on one machine, under a rule that rejects a change only when it
  loses more episodes than chance allows. Neutral changes could get in under that rule.
- The first configuration (v9) is the bundle of those options, registered on 17 November. The second (v11) adds a
  learner for objects the program does not know, registered on 20 November.

## What we do not claim

- Not that every option was shown to help on its own.
- Not that nothing in it is specific to the benchmark (`docs/DISCLOSURES.md` lists what is).
- Not that the only learned part is one net: the only neural networks are two small depth nets
  (`docs/DEPTH-NETS.md`); many constants were tuned by hand or by paired runs.
- Not that no person ever wrote a line. The record shows what is said above and no more.
