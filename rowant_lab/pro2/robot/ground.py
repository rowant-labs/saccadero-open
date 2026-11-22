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

STOP = {"the", "a", "an", "of", "it", "to", "and"}  # "and": "the yellow and white mug" names one object
GENERIC = {"box", "layer", "drawer", "level"}
CONST = {
    "on_dz_m": (-0.01, 0.03),
    "on_xy_m": 0.01,
    "in_grow_m": 0.01,
    "ambiguous_m": 0.01,
    # ``relation_holds`` (2026-11-16; used only by ``params["relation_single"]``): whether ONE candidate stands in the
    # instruction's relation at all, with room to spare. Measured on the colour line's rows (start state 8, and the
    # plain test tasks): "next to" reads 0.12-0.18 m for the named object and 0.26-0.42 m for the other one; "between"
    # 0.002-0.02 m off the line at 0.41-0.65 along it, against 0.10-0.14 m at 0.87-1.0; the table's centre 0.076 m
    # against 0.305 m
    "next_max_m": 0.22,
    "between_max_m": 0.06,
    "between_s": (0.15, 0.85),
    "centre_max_m": 0.18,
    "loose_on_dz_m": (-0.03, 0.06),
    "loose_xy_m": 0.03,
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
    share of the type's own words that the phrase names (so ``<obj>`` picks the <obj> and ``<obj> sauce`` does not
    match ``<obj>_sauce``). A tie fails."""
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
PLACE_PART = re.compile(r"^(?P<part>[a-z]+) (?:drawer|compartment|layer|level|shelf|section) of the (?P<np>.+)$")


def parse(instruction: str, types) -> dict:
    """{status, target_type, relation, landmarks, part, predicate, place_type} or {status: 'fail', reason}."""
    text = " ".join(words(instruction))
    # tolerate rewordings: leading filler before the pick verb ("please ...", "could you ..."), synonyms of the
    # verbs, "then" between the clauses and trailing filler after the place NP
    text = re.sub(r"\b(?:grab|take|get|lift)(?= (?:up )?the\b)", "pick", text)
    text = re.sub(r"\band (?:then )?(?:put|set|drop|move|place) it\b", "and place it", text)
    text = re.sub(r"\bthen (?:put|set|drop|move|place) it\b", "and place it", text)
    m0 = re.search(r"\bpick(?: up)? the\b", text)
    if m0 and m0.start() > 0:
        text = text[m0.start():]
    text = re.sub(r"(?: please| thanks| thank you)+$", "", text)
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
    np_s, place_part = mq.group("np"), None
    mpi = PLACE_PART.match(np_s)
    if mpi:  # "in the back compartment of the caddy": the part word names one of the place's sites
        place_part, np_s = mpi.group("part"), mpi.group("np")
    place_type, why = match_type(np_s, types)
    if place_type is None:
        return {"status": "fail", "reason": f"place type: {why}"}
    return {"status": "ok", "target_type": target_type, "relation": rel, "landmarks": lands, "part": part,
            "predicate": mq.group("pred"), "place_type": place_type, **({"place_part": place_part} if place_part else {})}


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


def between_at(lib, cand: dict, la: dict, lb: dict) -> tuple[float, float]:
    """(distance from the line through the two landmarks' centres, position along it: 0 at ``la``, 1 at ``lb``)."""
    a, b = aabb(lib, la), aabb(lib, lb)
    pa, pb = 0.5 * (a[0][:2] + a[1][:2]), 0.5 * (b[0][:2] + b[1][:2])
    c = aabb(lib, cand)
    x = 0.5 * (c[0][:2] + c[1][:2])
    ab = pb - pa
    n = float(np.linalg.norm(ab))
    if n < 1e-6:
        return float(np.linalg.norm(x - pa)), 0.5
    s = float((x - pa) @ ab / (n * n))
    return float(np.linalg.norm(x - (pa + s * ab))), s


def landmarks_of(parsed: dict, percepts: list, extra=()) -> list:
    """One percept per landmark of the instruction's relation (the best fitted of its type; from ``extra`` when the
    percept list has none of that type; ``{"table": True}`` for the table; None when there is none at all)."""
    out = []
    for lt in parsed.get("landmarks") or []:
        if lt == "table":
            out.append({"table": True})
            continue
        ps = [p for p in percepts if p["type"] == lt] or [p for p in extra if p["type"] == lt]
        out.append(max(ps, key=lambda p: p.get("fit", {}).get("n_agree", 0)) if ps else None)
    return out


def relation_holds(parsed: dict, cand: dict, percepts: list, lib, workcell: dict, extra_landmarks=()) -> bool | None:
    """Whether ``cand`` alone stands in the instruction's relation, judged with room to spare (``CONST``'s limits):
    True, False, or None when it cannot be told (no relation, a landmark with no percept, an unknown relation).
    ``ground`` compares candidates with each other; with one candidate there is nothing to compare, and this is the
    question that remains."""
    rel = parsed.get("relation")
    if rel is None:
        return None
    lands = landmarks_of(parsed, percepts, extra_landmarks)
    if not lands or any(x is None for x in lands):
        return None
    try:
        if rel == "on":
            lo, hi = aabb(lib, cand)
            xy = 0.5 * (lo[:2] + hi[:2])
            if not in_footprint(lib, lands[0], xy, CONST["loose_xy_m"]):
                return False
            top = top_under(lib, lands[0], xy)
            if top is None:  # over its edge, within the slack: the landmark's highest top stands in
                top = float(aabb(lib, lands[0])[1][2])
            lo_t, hi_t = CONST["loose_on_dz_m"]
            return bool(lo_t <= float(lo[2]) - float(top) <= hi_t)
        if rel == "in":
            site = lib.site_for(lands[0], parsed.get("part"))
            if site is None:
                return None
            c, R, half = site
            lo, hi = aabb(lib, cand)
            d = np.abs(R.T @ (0.5 * (lo + hi) - c)) - (half + CONST["loose_xy_m"])
            return bool(np.all(d <= 0))
        if rel == "next_to":
            return bool(rel_next_to(lib, cand, lands[0]) <= CONST["next_max_m"])
        if rel == "between":
            d, s = between_at(lib, cand, lands[0], lands[1])
            lo_s, hi_s = CONST["between_s"]
            return bool(d <= CONST["between_max_m"] and lo_s <= s <= hi_s)
        if rel == "table_centre":
            tb = [a for a in workcell["arena"] if a["type"] == "box"]
            ctr = np.asarray(tb[0]["centre"], float)[:2] if tb else np.zeros(2)
            a = aabb(lib, cand)
            return bool(float(np.linalg.norm(0.5 * (a[0][:2] + a[1][:2]) - ctr)) <= CONST["centre_max_m"])
    except Exception:
        return None
    return None


def relation_point(parsed: dict, percepts: list, lib, workcell: dict, extra_landmarks=()) -> dict | None:
    """Where the instruction's relation puts the named object, for a look from above (``params["relation_look"]``):
    ``{"xy", "support_z", "relation"}``, or None when it cannot be told (no relation, a landmark with no percept).
    On a thing: over its middle, on its top. In a part of a thing: the part's region, on its floor. Next to a thing:
    over the thing's middle (the wrist camera 27 cm up sees 28 cm to each side), on the table beside it. Between two
    things: halfway. From the table's centre: the middle of the table."""
    rel = parsed.get("relation")
    if rel is None:
        return None
    lands = landmarks_of(parsed, percepts, extra_landmarks)
    if not lands or any(x is None for x in lands):
        return None
    try:
        from .behind import support_under, table_z

        def mid(p):
            lo, hi = aabb(lib, p)
            return 0.5 * (lo[:2] + hi[:2]), float(hi[2])

        if rel == "table_centre":
            tb = [a for a in workcell["arena"] if a["type"] == "box"]
            if not tb:
                return None
            xy = np.asarray(tb[0]["centre"], float)[:2]
            z, within = table_z(lib, xy), CONST["centre_max_m"]
        elif rel == "on":
            xy, top = mid(lands[0])
            z = top_under(lib, lands[0], xy)
            z = top if z is None else float(z)
            lo, hi = aabb(lib, lands[0])
            within = 0.5 * float(np.max(hi[:2] - lo[:2])) + CONST["loose_xy_m"]
        elif rel == "in":
            site = lib.site_for(lands[0], parsed.get("part"))
            if site is None:
                return None
            c, _R, half = site
            xy, z = np.asarray(c, float)[:2], float(c[2] - half[2])
            within = float(np.max(half[:2])) + CONST["loose_xy_m"]
        elif rel == "next_to":
            xy, _top = mid(lands[0])
            z, within = table_z(lib, xy), CONST["next_max_m"]
        elif rel == "between":
            (a, _ta), (b, _tb) = mid(lands[0]), mid(lands[1])
            xy = 0.5 * (a + b)
            z = support_under(lib, [q for q in percepts if q is not lands[0] and q is not lands[1]], xy,
                              skip_types=(parsed.get("target_type"),))
            lo_s, hi_s = CONST["between_s"]
            within = float(np.hypot(0.5 * (hi_s - lo_s) * float(np.linalg.norm(b - a)), CONST["between_max_m"]))
        else:
            return None
        if z is None:
            return None
        # ``within_m``: how far from the point the named object's middle can be and still stand in the relation
        return {"xy": [float(xy[0]), float(xy[1])], "support_z": float(z), "relation": rel, "within_m": float(within)}
    except Exception:
        return None


# The world direction a side word means, from the robot's point of view (it faces +x, so its left is +y). Which way an
# instruction's "left" points is a development question for the long-task suite (``CONST["side_sign"]`` flips it).
SIDE_DIR = {"left": np.array([0.0, 1.0]), "right": np.array([0.0, -1.0])}
CONST["side_sign"] = 1.0


def pick_side(lib, places: list, side: str) -> dict:
    """The place percept furthest toward ``side`` (by its root body's xy)."""
    d = SIDE_DIR[side] * float(CONST["side_sign"])

    def along(p):
        root = lib.export_types[p["type"]]["root_body"]
        return float(np.asarray(p["bodies"][root]["pos"], float)[:2] @ d)

    return max(places, key=along)


def is_type(p: dict, typ) -> bool:
    """A percept of ``typ``: by its own type, or a thing taken as it is seen and named that (``seen.seen_percept``)."""
    return p.get("type") == typ or (p.get("named") is not None and p.get("named") == typ)


NEXT_GAP_M = 0.06  # "next to": the two footprints at most this far apart (plain scenes: touching to 1 cm; others 9 cm up)
BETWEEN_M = 0.06   # "between": at most this far from the line between the two (plain scenes: under 2 cm; others 10 cm up)
ON_XY_M, ON_DZ_M, IN_OUT_M = 0.03, 0.05, 0.03  # "on", "in": clearly off the landmark's top, clearly outside its region
CENTRE_M = 0.20    # "from table center": within this of the table's middle (one plain scene: 8 cm; the other bowl 30 cm)


def single_fails(parsed: dict, cand: dict, percepts: list, lib) -> bool:
    """Whether the ONLY percept of the target's type clearly fails the instruction's relation: it is then another
    object of that type, and the target itself is not in view (2026-11-16: with the target
    hidden, the robot carried the other bowl in 8 of 8 episodes). False whenever it cannot be told (no relation, a
    landmark not in view)."""
    rel = parsed.get("relation")
    if rel == "table_centre":
        # "from table center": clearly elsewhere when further than CENTRE_M from the middle of the workcell's table
        try:
            tb = [a for a in (getattr(lib, "workcell", None) or {}).get("arena") or [] if a.get("type") == "box"]
            if not tb:
                return False
            a = aabb(lib, cand)
            return float(np.linalg.norm(0.5 * (a[0][:2] + a[1][:2]) - np.asarray(tb[0]["centre"], float)[:2])) > CENTRE_M
        except Exception:
            return False
    if rel not in ("on", "in", "next_to", "between"):
        return False
    lands = []
    for lt in parsed.get("landmarks") or []:
        ps = [p for p in percepts if is_type(p, lt)]
        if lt == "table" or not ps:
            return False
        lands.append(max(ps, key=lambda p: p.get("fit", {}).get("n_agree", 0)))
    try:
        if rel == "on":
            lo, hi = aabb(lib, cand)
            xy = 0.5 * (lo[:2] + hi[:2])
            top = top_under(lib, lands[0], xy)
            if top is None:
                return not in_footprint(lib, lands[0], xy, ON_XY_M)
            return abs(float(lo[2] - top) - 0.01) > ON_DZ_M
        if rel == "in":
            site = lib.site_for(lands[0], parsed.get("part"))
            if site is None:
                return False
            c, R, half = site
            lo, hi = aabb(lib, cand)
            d = np.abs(R.T @ (0.5 * (lo + hi) - c)) - half
            return bool(float(d.max()) > IN_OUT_M)
        if rel == "next_to":
            a, b = aabb(lib, cand), aabb(lib, lands[0])
            pa, pb = 0.5 * (a[0][:2] + a[1][:2]), 0.5 * (b[0][:2] + b[1][:2])
            v = pb - pa
            n = float(np.linalg.norm(v))
            if n < 1e-6:
                return False
            u = np.abs(v / n)
            gap = n - float(u @ (0.5 * (a[1][:2] - a[0][:2]))) - float(u @ (0.5 * (b[1][:2] - b[0][:2])))
            return gap > NEXT_GAP_M
        return rel_between(lib, cand, lands[0], lands[1]) > BETWEEN_M
    except Exception:
        return False


def single_untold(parsed: dict, percepts: list) -> bool:
    """Whether the ONLY percept of the target's type cannot be told to be the one the instruction means: its
    relation names a landmark of which the look has no percept. (2026-11-16, the panel case on the pod: where the
    panel hid the target together with the thing it stood on or by, the other bowl was carried in 4 of 8 episodes;
    ``single_fails`` says nothing there, for want of the landmark.)"""
    if parsed.get("relation") not in ("on", "in", "next_to", "between"):
        return False
    if sum(1 for p in percepts if is_type(p, parsed.get("target_type"))) != 1:
        return False
    return any(lt != "table" and not any(is_type(p, lt) for p in percepts) for lt in parsed.get("landmarks") or [])


def ground(parsed: dict, percepts: list, lib, workcell: dict, strict_single: bool = False,
           extra_landmarks=()) -> dict:
    """{status, target, place, flags, reason} over the percept list. ``strict_single`` (off by default): a single
    candidate that clearly fails the instruction's relation is not taken; the reply is then a failure marked
    ``hidden`` (the caller may look for the target where the camera does not see). ``extra_landmarks`` (off unless
    given): percepts that stand in for a landmark type the list has none of (``look.relation_candidates``); they are
    never a target or a place."""
    flags = []
    cands = [p for p in percepts if is_type(p, parsed["target_type"])]
    if not cands:
        return {"status": "fail", "reason": f"no percept of {parsed['target_type']}", "flags": flags}
    places = [p for p in percepts if is_type(p, parsed["place_type"])]
    if not places:
        return {"status": "fail", "reason": f"no percept of place {parsed['place_type']}", "flags": flags}
    place = max(places, key=lambda p: p.get("fit", {}).get("n_agree", 0))
    if len(places) > 1 and parsed.get("place_side"):
        place = pick_side(lib, places, parsed["place_side"])
        flags.append(f"place_side_{parsed['place_side']}")
    elif len(places) > 1:
        flags.append("several_place_percepts")
    rel = parsed["relation"]
    if strict_single and len(cands) == 1 and single_fails(parsed, cands[0], percepts, lib):
        return {"status": "fail", "reason": f"the only {parsed['target_type']} in view fails '{rel}'",
                "flags": flags + ["single_fails_relation"], "hidden": True, "only": cands[0]["id"],
                "place": place["id"], "relation": rel}
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
        ps = [p for p in percepts if is_type(p, lt)]
        if not ps and extra_landmarks:
            ps = [p for p in extra_landmarks if p["type"] == lt]
            if ps:
                flags.append(f"landmark_fallback:{lt}")
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
