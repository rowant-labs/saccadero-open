"""Compound instructions as a sequence of steps (the long-task suite; off unless ``params["sequence"]``).

The long-task suite's instructions ask for two things at once: "put both the A and the B in the C", "put the A on the
B and put the C on the D", "put the A in the B and close it", "turn on the B and put the A on it", "put both moka pots
on the stove". ``split`` turns one into clauses the robot's single-step forms already read (``ground.parse``'s pick
form, ``goal.parse_put``'s put form, ``drawer``'s open form). ``SequenceAgent`` runs them one after another inside the
episode's one step budget, each step planned from the scene as the robot sees it when the step starts (the look tool's
merged percepts), never choosing as a target an object an earlier step already placed.

Splitting rules (lower case, punctuation stripped):
- at `` and `` followed by a verb that starts a clause (put, place, pick, close, open, turn, push, stack, move), except
  ``and place it`` (the pick form's own second half);
- ``both the A and the B <rest>`` -> ``the A <rest>`` and ``the B <rest>``; ``both <plural> <rest>`` -> the singular
  twice ("both moka pots" -> "the moka pot", two steps, each choosing an object not yet placed);
- ``it`` in a later clause names the previous clause's last noun phrase (its place if it has one, else its object):
  "put the bowl in the bottom drawer of the cabinet and close it" -> "close the bottom drawer of the cabinet".
A clause with no verb of its own ("and the B") is never a split point, so "yellow and white mug" and "between the A and
the B" stay whole.
"""

from __future__ import annotations

import re

from .ground import words

VERBS = ("put", "place", "pick", "close", "open", "turn", "push", "stack", "move")
NOOP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
RESERVE_STEPS = 110  # kept for each later step when a step's budget is set (one pick-and-place is about 115 steps)
MIN_STEP = 60  # a step with fewer steps left than this is not started
DONE_REASONS = ("placed", "opened", "closed", "swept", "cleared")  # how a step that did its work finishes


PLACE_VERBS = ("put", "place", "set", "drop", "move")
FILLER_HEAD = re.compile(r"^(?:(?:please|kindly|now|could you|can you|would you|i want you to|you should) )+")
FILLER_TAIL = re.compile(r"(?: please| thanks| thank you| now)+$")


def _norm(text: str) -> str:
    t = " ".join(words(text))
    return FILLER_TAIL.sub("", FILLER_HEAD.sub("", t))


def _split_verbs(text: str) -> list[str]:
    """Split at ``and [then] <verb>`` or ``then <verb>``. Never where a place verb is followed by ``it`` right after a
    pick clause ("pick the A ... and place it on the B", "grab the A then put it in the B"): that is one
    pick-and-place, and the pick form reads it whole."""
    parts, cur = [], []
    ws = text.split()
    i = 0
    while i < len(ws):
        j = None
        if cur and ws[i] == "and":
            j = i + 2 if i + 1 < len(ws) and ws[i + 1] == "then" else i + 1
        elif cur and ws[i] == "then":
            j = i + 1
        if j is not None and j < len(ws) and ws[j] in VERBS:
            pick_half = (ws[j] in PLACE_VERBS and j + 1 < len(ws) and ws[j + 1] == "it"
                         and cur[0] in ("pick", "grab", "take", "get", "lift"))
            if not pick_half:
                parts.append(" ".join(cur))
                cur = []
                i = j
                continue
        cur.append(ws[i])
        i += 1
    if cur:
        parts.append(" ".join(cur))
    return parts


def _singular(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


BOTH_PAIR = re.compile(r"^(?P<verb>\w+(?: up)?) both the (?P<a>.+?) and the (?P<b>.+?) (?P<rest>(?:in|into|on|onto|inside|on top of) .+)$")
BOTH_PLURAL = re.compile(r"^(?P<verb>\w+(?: up)?) both (?P<np>[a-z ]+?) (?P<rest>(?:in|into|on|onto|inside|on top of) .+)$")


def _expand_both(clause: str) -> list[str]:
    m = BOTH_PAIR.match(clause)
    if m:
        v, rest = m.group("verb"), m.group("rest")
        return [f"{v} the {m.group('a')} {rest}", f"{v} the {m.group('b')} {rest}"]
    m = BOTH_PLURAL.match(clause)
    if m:
        np = m.group("np").split()
        np[-1] = _singular(np[-1])
        one = f"{m.group('verb')} the {' '.join(np)} {m.group('rest')}"
        return [one, one]
    return [clause]


PLACE_NP = re.compile(r"\b(?:in|into|on|onto|inside|on top of|of) (?P<np>the [a-z ]+)$")
OBJ_NP = re.compile(r"^(?:\w+)(?: on| off| up)? (?P<np>the [a-z ]+?)(?: (?:in|into|on|onto|inside|on top of|to) .*)?$")


def _last_np(clause: str) -> str | None:
    """The clause's last noun phrase: its place ("... in the bottom drawer of the cabinet" -> "the bottom drawer of the
    cabinet"), else its object ("turn on the stove" -> "the stove")."""
    m = re.search(r"\b(?:in|into|on|onto|inside|on top of) (?P<np>the .+)$", clause)
    if m and not re.match(r"^\w+ (?:on|off) the [a-z ]+$", clause):  # "turn on the stove": "on" is the verb's
        return m.group("np")
    m = OBJ_NP.match(clause)
    return m.group("np") if m else None


INSIDE_END = re.compile(r"^(?P<head>.+?) (?:inside|in there|into there|within)$")


def split(instruction: str) -> list[str]:
    """The instruction's steps (one item when it is not compound)."""
    text = _norm(instruction)
    clauses = []
    for c in _split_verbs(text):
        clauses.extend(_expand_both(c))
    out = []
    for c in clauses:
        if out:
            np = _last_np(out[-1])
            m = INSIDE_END.match(c)
            if np and m:  # "... and put the A inside": inside what the previous clause named
                c = f"{m.group('head')} in {np}"
            elif np and re.search(r"\bit\b", c) and not re.search(r"\bplace it\b", c):
                c = re.sub(r"\bit\b", np, c)
        out.append(c)
    return out


def _is_idle(a) -> bool:
    """An agent's idle action: no movement asked (the gripper command may be either)."""
    try:
        return len(a) >= 6 and all(abs(float(x)) < 1e-12 for x in a[:6])
    except Exception:
        return False


class SequenceAgent:
    """Runs the steps in order. ``planner(clause, cap, exclude)`` returns ``plan_episode``'s dict for one step (its
    agent drives the robot until it finishes). A step that cannot be planned is recorded and skipped."""

    def __init__(self, steps: list[str], planner, cap: int, reserve: int = RESERVE_STEPS, min_step: int = MIN_STEP,
                 retry: bool = False, chain: bool = False):
        self.steps, self.planner, self.cap = list(steps), planner, int(cap)
        self.reserve, self.min_step = int(reserve), int(min_step)
        # 2026-11-17, off by default: the step that follows begins in the same control step in which the one before
        # it ended. (As built, a step's last action is its agent's idle one, sent because its generator had ended:
        # one step of standing still at every hand-over, two in the drawer task.)
        self.chain = bool(chain)
        # 2026-11-16, off by default: when the last step has ended and steps are left, the clauses that did not do
        # their work are run once more, in their order, each planned from the scene as it is then (the benchmark
        # ends an episode the moment its task is done, so an episode still running has something left to do)
        self.retry = bool(retry)
        self._retried = False
        self.k = -1
        self.t = 0
        self.cur = None
        self.cur_target = None
        self.placed: list = []  # percept ids an earlier step placed (never a later step's target)
        self.records: list = []
        self.attempts: list = []  # every step's attempts and events, each tagged with its step
        self.events: list = []
        self.finished = False
        self.finish_reason = None
        self._next()

    def _need(self, step) -> int:
        """Steps kept back for a later step: its own estimate (a planned step says ``need``), else the reserve."""
        return int(step.get("need", self.reserve)) if isinstance(step, dict) else self.reserve

    def _next(self) -> None:
        while self.k + 1 < len(self.steps):
            self.k += 1
            left = self.cap - self.t
            step = self.steps[self.k]
            step_cap = left - sum(self._need(x) for x in self.steps[self.k + 1:])
            rec = {"step": self.k, "clause": step.get("clause") if isinstance(step, dict) else step, "t_start": self.t,
                   "cap": step_cap}
            if isinstance(step, dict) and step.get("retry_of") is not None:
                rec["retry_of"] = int(step["retry_of"])
            self.records.append(rec)
            if step_cap < (int(step.get("min", self.min_step)) if isinstance(step, dict) else self.min_step):
                rec["skipped"] = "no budget"
                continue
            req = step.get("requires") if isinstance(step, dict) else None
            if req is not None:  # a planned step that only makes sense after another one succeeded
                done = (self.records[req].get("summary") or {}).get("finish_reason") if req < len(self.records) else None
                if done not in DONE_REASONS:
                    rec["skipped"] = "the step it needs did not finish"
                    continue
            try:
                planned = self.planner(self.steps[self.k], step_cap, tuple(self.placed)) or {}
            except Exception as ex:  # a step that cannot be planned is skipped, not a crash
                planned = {"agent": None, "grounding": {"status": "fail", "reason": f"plan error: {str(ex)[:80]}"}}
            rec["parsed"] = planned.get("parsed")
            rec["grounding"] = planned.get("grounding")
            if planned.get("agent") is None:
                rec["skipped"] = "no plan"
                continue
            self.cur = planned["agent"]
            self.cur_target = (planned.get("grounding") or {}).get("target")
            try:  # for the agent's ``c3_measured``: whether the episode goes on after this step whatever it does
                self.cur.more_follow = self.k + 1 < len(self.steps)
            except Exception:
                pass
            return
        if self.retry and not self._retried:
            self._retried = True
            again = [i for i, r in enumerate(self.records) if self._retriable(self.steps[i], r)]
            if again and self.cap - self.t >= self.min_step:
                self.steps += [{"kind": "clause", "retry_of": i, **(self.steps[i] if isinstance(self.steps[i], dict)
                                                                     else {"clause": self.steps[i]})} for i in again]
                return self._next()
        self.cur = None
        self.finished = True
        self.finish_reason = "steps done" if any("summary" in r for r in self.records) else "no step planned"

    @staticmethod
    def _retriable(step, rec: dict) -> bool:
        """A clause of the instruction (not a step planned around another one) that ended without doing its work,
        or could not be planned; never one skipped for want of steps."""
        if isinstance(step, dict) and step.get("kind") != "clause":
            return False
        if rec.get("skipped") == "no budget":
            return False
        return (rec.get("summary") or {}).get("finish_reason") not in DONE_REASONS

    def act(self, obs) -> list:
        if self.cur is None:
            return list(NOOP)
        a = self.cur.act(obs)
        self.t += 1
        if getattr(self.cur, "finished", False):
            summ = self.cur.summary() if hasattr(self.cur, "summary") else {}
            rec = self.records[-1]
            rec["summary"] = {k: summ.get(k) for k in ("steps", "finish_reason")}
            rec["t_end"] = self.t
            self.attempts += [{**a_, "step": self.k} for a_ in (summ.get("attempts") or []) if isinstance(a_, dict)]
            self.events += [{**e_, "step": self.k} for e_ in (summ.get("events") or []) if isinstance(e_, dict)]
            if summ.get("finish_reason") == "placed" and self.cur_target is not None:
                self.placed.append(self.cur_target)
            self._next()
            if self.chain and self.cur is not None and _is_idle(a):
                # the next step's first action instead of the ended step's idle one (the same observation: the
                # step was planned on it a moment ago). The record: the new step began one control step earlier
                # than its ``t_start`` says, and the ended step's count holds the idle step that was not sent.
                try:
                    a2 = self.cur.act(obs)
                except Exception:
                    a2 = None
                if a2 is not None:
                    a = a2
                    self.records[-1]["chained"] = True
                    if getattr(self.cur, "finished", False):  # a step that ended on its first action: as any other
                        return self._after_chain(a)
        return a

    def _after_chain(self, a):
        summ = self.cur.summary() if hasattr(self.cur, "summary") else {}
        rec = self.records[-1]
        rec["summary"] = {k: summ.get(k) for k in ("steps", "finish_reason")}
        rec["t_end"] = self.t
        self.attempts += [{**a_, "step": self.k} for a_ in (summ.get("attempts") or []) if isinstance(a_, dict)]
        self.events += [{**e_, "step": self.k} for e_ in (summ.get("events") or []) if isinstance(e_, dict)]
        if summ.get("finish_reason") == "placed" and self.cur_target is not None:
            self.placed.append(self.cur_target)
        self._next()
        return a

    def summary(self) -> dict:
        recs = list(self.records)
        if self.cur is not None and recs and "summary" not in recs[-1]:  # the step still running at the cap
            summ = self.cur.summary() if hasattr(self.cur, "summary") else {}
            recs[-1] = {**recs[-1], "summary": {k: summ.get(k) for k in ("steps", "finish_reason")}, "open": True}
            att = [{**a_, "step": self.k} for a_ in (summ.get("attempts") or []) if isinstance(a_, dict)]
            ev = [{**e_, "step": self.k} for e_ in (summ.get("events") or []) if isinstance(e_, dict)]
        else:
            att, ev = [], []
        return {"steps": self.t, "finish_reason": self.finish_reason, "sequence": recs,
                "attempts": self.attempts + att, "events": self.events + ev}
