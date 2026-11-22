"""M2 grounding (plan grounding): a rule parser over the instruction plus geometric relations over PERCEIVED poses.

M2 (perceived poses, grounded instruction): perception in the loop. No LLM, VLM or SigLIP. Policy side: it sees the
instruction string, the onboarded type names, the per-type models (primitives and body-frame sites, through
``m2_library``) and the percepts' poses. It never sees the BDDL, the goal, a region of the task or any true pose.

Parsing (``parse``): lower-case, punctuation stripped; split at ``and place it`` into a pick clause and a place
clause. The pick clause is ``pick [up] the <NP> [<relation>]``; the target NP is the longest prefix of the words
after ``the`` that names a library type by the name rule. The place clause is ``(in|on) the <NP>`` and gives the
predicate and the place type. Relations: ``between the A and the B``, ``next to the A``, ``on the A``,
``in the <part> of the A``, ``in the A``, ``from <A> cent(er|re)``, ``at the A``. Anything that does not parse is
``grounding_fail``.

Name rule (``match_type``): type names read as words (underscores are spaces); one trailing ``s`` is stripped;
a phrase matches a type when they share a content word, scored by the Jaccard overlap of the word sets; the unique
best type wins and a tie fails. ``box``, ``layer``, ``drawer`` and the articles are not content words. ``table``
resolves to the workcell's table.

Selection (``ground``) over the percepts: ``on``, ``in`` are binary windows (exactly one candidate -> it; several ->
the best graded score; none -> the best graded score, flagged ``relation_unsatisfied``); ``next to``, ``between`` and
the table centre are graded (argmin, flagged ``ambiguous`` when the two best costs differ by under 1 cm).
"""

from __future__ import annotations

import re

import numpy as np

STOP = {"the", "a", "an", "of", "it", "to"}
GENERIC = {"box", "layer", "drawer", "level"}
CONST = {
    "on_dz_m": (-0.01, 0.03),
    "on_xy_m": 0.01,
    "in_grow_m": 0.01,
    "ambiguous_m": 0.01,
}


def words(text: str) -> list[str]:
    return [w for w in re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).split() if w]


def _content(ws) -> set:
    out = set()
    for w in ws:
        if w in STOP or w in GENERIC:
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


def match_type(phrase, types) -> tuple[str | None, str]:
    """(type, reason): the unique type whose name words CONTAIN every content word of ``phrase``, scored by the
    share of the type's own words that the phrase names (so ``ramekin`` picks the ramekin and ``bbq sauce`` does not
    match ``tomato_sauce``). A tie fails."""
    pw = _content(words(phrase) if isinstance(phrase, str) else phrase)
    if not pw:
        return None, "no content word"
    best = []
    for t in types:
        tw = _content(t.split("_"))
        if not pw <= tw:
            continue
        best.append((len(pw & tw) / len(tw), t))
    if not best:
        return None, f"no type contains {sorted(pw)}"
    best.sort(reverse=True)
    if len(best) > 1 and abs(best[0][0] - best[1][0]) < 1e-9:
        return None, f"tie between {best[0][1]} and {best[1][1]}"
    return best[0][1], "ok"


REL_PATTERNS = (
    ("between", re.compile(r"\bbetween the (?P<a>.+?) and the (?P<b>.+)$")),
    ("next_to", re.compile(r"\bnext to the (?P<a>.+)$")),
    ("in_part", re.compile(r"\bin the (?P<part>[a-z ]+?) (?:layer|level|drawer) of the (?P<a>.+)$")),
    ("on", re.compile(r"\bon the (?P<a>.+)$")),
    ("in", re.compile(r"\bin the (?P<a>.+)$")),
    ("table_centre", re.compile(r"\bfrom (?P<a>[a-z ]*?) ?cent(?:er|re)$")),
    ("at", re.compile(r"\bat the (?P<a>.+)$")),
)
PICK = re.compile(r"^pick(?: up)? the (?P<rest>.+)$")
PLACE = re.compile(r"^place it (?P<pred>in|on) the (?P<np>.+)$")


def parse(instruction: str, types) -> dict:
    """{status, target_type, relation, landmarks, part, predicate, place_type} or {status: 'fail', reason}."""
    text = " ".join(words(instruction))
    if " and place it " not in text:
        return {"status": "fail", "reason": "no place clause"}
    pick_s, place_s = text.split(" and place it ", 1)
    mp = PICK.match(pick_s)
    if not mp:
        return {"status": "fail", "reason": "pick clause"}
    rest = mp.group("rest")
    ws = rest.split()
    target_type, tail = None, ""
    for n in range(len(ws), 0, -1):
        t, _why = match_type(" ".join(ws[:n]), types)
        if t is not None:
            target_type, tail = t, " ".join(ws[n:])
            break
    if target_type is None:
        return {"status": "fail", "reason": f"no target type in {rest!r}"}
    rel, lands, part = None, [], None
    if tail:
        for name, pat in REL_PATTERNS:
            m = pat.search(tail)
            if not m:
                continue
            gd = m.groupdict()
            if name == "table_centre":
                rel, lands = "table_centre", ["table"]
            else:
                ls = [gd.get("a"), gd.get("b")]
                lands = [match_type(x, types)[0] or ("table" if "table" in x else None) for x in ls if x]
                if any(x is None for x in lands):
                    return {"status": "fail", "reason": f"landmark in {tail!r}"}
                rel = "on" if name == "at" else ("in" if name == "in_part" else name)
                part = gd.get("part")
            break
        if rel is None:
            return {"status": "fail", "reason": f"relation {tail!r}"}
    mq = PLACE.match("place it " + place_s if not place_s.startswith("place it") else place_s)
    if not mq:
        return {"status": "fail", "reason": "place clause"}
    place_type, why = match_type(mq.group("np"), types)
    if place_type is None:
        return {"status": "fail", "reason": f"place type: {why}"}
    return {"status": "ok", "target_type": target_type, "relation": rel, "landmarks": lands, "part": part,
            "predicate": mq.group("pred"), "place_type": place_type}


# ------------------------------------------------------------------------------------- geometry over percepts
def _boxes(lib, p: dict):
    """(centres, Rs, halves) of every collision primitive of a percept, in the world."""
    out = []
    tdoc = lib.m1_type(p["type"])
    for rel, b in tdoc["bodies"].items():
        if rel not in p["bodies"]:
            continue
        R = lib.quat_mat(p["bodies"][rel]["quat_wxyz"])
        t = np.asarray(p["bodies"][rel]["pos"], float)
        for pr in b["primitives"]:
            out.append((t + R @ np.asarray(pr["centre"], float), R @ np.asarray(pr["R"], float),
                        np.asarray(pr["half"], float)))
    return out


def aabb(lib, p: dict):
    bs = _boxes(lib, p)
    los, his = [], []
    for c, R, h in bs:
        e = np.abs(R) @ h
        los.append(c - e)
        his.append(c + e)
    return np.min(los, 0), np.max(his, 0)


def top_under(lib, p: dict, xy) -> float | None:
    """The highest primitive top of ``p`` whose footprint covers ``xy``."""
    best = None
    for c, R, h in _boxes(lib, p):
        e = np.abs(R) @ h
        if np.all(np.abs(np.asarray(xy) - c[:2]) <= e[:2]):
            best = max(best, c[2] + e[2]) if best is not None else c[2] + e[2]
    return best


def in_footprint(lib, p: dict, xy, slack: float = 0.0) -> bool:
    for c, R, h in _boxes(lib, p):
        e = np.abs(R) @ h
        if np.all(np.abs(np.asarray(xy) - c[:2]) <= e[:2] + slack):
            return True
    return False


def rel_on(lib, cand: dict, land: dict) -> tuple[bool, float]:
    lo, _hi = aabb(lib, cand)
    xy = 0.5 * (lo[:2] + aabb(lib, cand)[1][:2])
    top = top_under(lib, land, xy)
    if top is None:
        return False, 1e3
    dz = lo[2] - top
    lo_t, hi_t = CONST["on_dz_m"]
    ok = (lo_t <= dz <= hi_t) and in_footprint(lib, land, xy, CONST["on_xy_m"])
    cost = abs(dz - 0.5 * (lo_t + hi_t)) + (0.0 if in_footprint(lib, land, xy, CONST["on_xy_m"]) else 0.05)
    return ok, float(cost)


def rel_in(lib, cand: dict, land: dict, part: str | None) -> tuple[bool, float]:
    """``cand``'s centre inside the landmark's region box of that part (grown by 1 cm)."""
    site = lib.site_for(land, part)
    if site is None:
        return False, 1e3
    c, R, half = site
    lo, hi = aabb(lib, cand)
    mid = 0.5 * (lo + hi)
    d = np.abs(R.T @ (mid - c)) - (half + CONST["in_grow_m"])
    return bool(np.all(d <= 0)), float(np.maximum(d, 0).sum() + 0.1 * max(float(d.max()), 0.0))


def rel_next_to(lib, cand: dict, land: dict) -> float:
    a, b = aabb(lib, cand), aabb(lib, land)
    return float(np.linalg.norm(0.5 * (a[0][:2] + a[1][:2]) - 0.5 * (b[0][:2] + b[1][:2])))


def rel_between(lib, cand: dict, la: dict, lb: dict) -> float:
    a, b = aabb(lib, la), aabb(lib, lb)
    pa, pb = 0.5 * (a[0][:2] + a[1][:2]), 0.5 * (b[0][:2] + b[1][:2])
    c = aabb(lib, cand)
    x = 0.5 * (c[0][:2] + c[1][:2])
    ab = pb - pa
    n = float(np.linalg.norm(ab))
    if n < 1e-6:
        return float(np.linalg.norm(x - pa))
    s = float(np.clip((x - pa) @ ab / (n * n), 0.0, 1.0))
    return float(np.linalg.norm(x - (pa + s * ab)))


def ground(parsed: dict, percepts: list, lib, workcell: dict) -> dict:
    """{status, target, place, flags, reason} over the percept list."""
    flags = []
    cands = [p for p in percepts if p["type"] == parsed["target_type"]]
    if not cands:
        return {"status": "fail", "reason": f"no percept of {parsed['target_type']}", "flags": flags}
    places = [p for p in percepts if p["type"] == parsed["place_type"]]
    if not places:
        return {"status": "fail", "reason": f"no percept of place {parsed['place_type']}", "flags": flags}
    place = max(places, key=lambda p: p.get("fit", {}).get("n_agree", 0))
    if len(places) > 1:
        flags.append("several_place_percepts")
    rel = parsed["relation"]
    if len(cands) == 1 or rel is None:
        if len(cands) > 1:
            flags.append("no_relation_several_candidates")
        tgt = max(cands, key=lambda p: p.get("fit", {}).get("n_agree", 0))
        return {"status": "ok", "target": tgt["id"], "place": place["id"], "flags": flags, "relation": rel}
    lands = []
    for lt in parsed["landmarks"]:
        if lt == "table":
            lands.append({"table": True})
            continue
        ps = [p for p in percepts if p["type"] == lt]
        if not ps:
            flags.append(f"landmark_missing:{lt}")
            lands.append(None)
        else:
            lands.append(max(ps, key=lambda p: p.get("fit", {}).get("n_agree", 0)))
    if any(x is None for x in lands):
        tgt = max(cands, key=lambda p: p.get("fit", {}).get("n_agree", 0))
        return {"status": "ok", "target": tgt["id"], "place": place["id"], "flags": flags + ["landmark_missing"],
                "relation": rel}
    costs, sat = [], []
    for c in cands:
        if rel == "on":
            ok, cost = rel_on(lib, c, lands[0])
        elif rel == "in":
            ok, cost = rel_in(lib, c, lands[0], parsed["part"])
        elif rel == "next_to":
            cost, ok = rel_next_to(lib, c, lands[0]), None
        elif rel == "between":
            cost, ok = rel_between(lib, c, lands[0], lands[1]), None
        elif rel == "table_centre":
            tb = [a for a in workcell["arena"] if a["type"] == "box"]
            ctr = np.asarray(tb[0]["centre"], float)[:2] if tb else np.zeros(2)
            a = aabb(lib, c)
            cost, ok = float(np.linalg.norm(0.5 * (a[0][:2] + a[1][:2]) - ctr)), None
        else:
            return {"status": "fail", "reason": f"relation {rel}", "flags": flags}
        costs.append(cost)
        sat.append(ok)
    if rel in ("on", "in"):
        hits = [i for i, s in enumerate(sat) if s]
        if len(hits) == 1:
            i = hits[0]
        elif len(hits) > 1:
            i = min(hits, key=lambda k: costs[k])
            flags.append("several_satisfy")
        else:
            i = int(np.argmin(costs))
            flags.append("relation_unsatisfied")
    else:
        i = int(np.argmin(costs))
        o = sorted(costs)
        if len(o) > 1 and o[1] - o[0] < CONST["ambiguous_m"]:
            flags.append("ambiguous")
    return {"status": "ok", "target": cands[i]["id"], "place": place["id"], "flags": flags, "relation": rel,
            "costs": [round(float(x), 4) for x in costs]}
