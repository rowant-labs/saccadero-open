"""Goal-form instructions: ``put the <X> on|in [the top of] the <Y>``.

Parsed into the same dict as ``ground.parse`` (relation None), plus ``place_part`` (a word naming a region of the
place, e.g. ``top``). A place phrase that names no type but says ``drawer`` resolves to the types whose parts move
on slide joints (the caller chooses the perceived one). The place region for ``on`` a fixture is chosen from the
type's sites on its rigid bodies (the part word first, else the highest).
"""

from __future__ import annotations

import re

from .ground import match_type, words

PUT = re.compile(r"^(?:put|place|set|stack|drop|move) the (?P<rest>.+)$")
PREP = re.compile(r"^(?P<pred>on|onto|in|into|inside|on top of|atop) (?P<np>.+)$")
# "to the right of the plate": on the table beside a landmark (the long-task suite; ``episode._beside_site``)
BESIDE = re.compile(r"^to the (?P<side>left|right) of (?P<np>.+)$")
PART_OF = re.compile(r"^(?:the )?(?P<part>top|top side|top surface|surface) of (?P<np>.+)$")
# "the bottom drawer of the cabinet", "the back compartment of the caddy": the part word names one of the place's sites
PART_IN = re.compile(r"^(?:the )?(?P<part>[a-z]+) (?:drawer|compartment|layer|level|shelf|section) of (?P<np>.+)$")
DRAWER_WORDS = {"drawer", "drawers", "cabinet", "cabinets", "dresser"}
PART_ONLY = re.compile(r"^(?:the )?(?P<part>[a-z]+) (?P<kind>drawer|compartment|layer|level|shelf|section)$")
SIDES = ("left", "right")  # a side word before the place phrase picks one of several places


def slide_types(lib) -> list:
    out = []
    try:
        for t, d in lib.export_types.items():
            parts = d.get("parts") or {}
            if any((pd or {}).get("joint") and (pd["joint"].get("type") == "slide") for pd in parts.values()):
                out.append(t)
    except Exception:
        return []
    return sorted(out)


def _place_type(np_words, types, lib):
    t, why = match_type(" ".join(np_words), types)
    if t is not None:
        return [t], "ok"
    if any(w in DRAWER_WORDS for w in np_words):
        st = slide_types(lib)
        if st:
            return st, "drawer"
    return [], why


def parse_put(instruction: str, lib) -> dict:
    types = sorted(lib.types)
    text = " ".join(words(instruction))
    text = re.sub(r"(?: please| thanks| thank you)+$", "", text)
    m0 = re.search(r"\b(?:put|place|set|stack|drop|move) the\b", text)
    if not m0:
        return {"status": "fail", "reason": "no put verb"}
    text = text[m0.start():]
    mp = PUT.match(text)
    if not mp:
        return {"status": "fail", "reason": "put clause"}
    ws = mp.group("rest").split()
    target_type, tail = None, []
    for n in range(len(ws) - 1, 0, -1):
        if not (PREP.match(" ".join(ws[n:])) or BESIDE.match(" ".join(ws[n:]))):
            continue
        t, _why = match_type(" ".join(ws[:n]), types)
        if t is not None:
            target_type, tail = t, ws[n:]
            break
    if target_type is None:
        return {"status": "fail", "reason": "no target type"}
    mb = BESIDE.match(" ".join(tail))
    if mb:
        npw = mb.group("np").split()
        npw = npw[1:] if npw and npw[0] == "the" else npw
        t, why = match_type(" ".join(npw), types)
        if t is None:
            return {"status": "fail", "reason": f"landmark: {why}"}
        return {"status": "ok", "target_type": target_type, "relation": None, "landmarks": [], "part": None,
                "predicate": "on", "place_type": t, "place_types": [t], "place_part": None, "form": "put",
                "place_beside": mb.group("side")}
    mq = PREP.match(" ".join(tail))
    pred_w, np_s = mq.group("pred"), mq.group("np")
    pred = "in" if pred_w in ("in", "into", "inside") else "on"
    part = "top" if pred_w in ("on top of", "atop") else None
    mpo = PART_OF.match(np_s)
    mpi = PART_IN.match(np_s) if not mpo else None
    if mpo:
        part = mpo.group("part").split()[0]
        np_s = mpo.group("np")
    elif mpi:
        part = mpi.group("part")
        np_s = mpi.group("np")
    else:
        mpn = PART_ONLY.match(np_s)
        if mpn:  # "the top drawer": the part word, and the kind word as the place phrase
            part, np_s = mpn.group("part"), mpn.group("kind")
    npw = np_s.split()
    if npw and npw[0] == "the":
        npw = npw[1:]
    side = None
    if len(npw) > 1 and npw[0] in SIDES:  # "the left plate": which of several places (``ground.pick_side``)
        side, npw = npw[0], npw[1:]
    cands, why = _place_type(npw, types, lib)
    if not cands:
        return {"status": "fail", "reason": f"place type: {why}"}
    return {"status": "ok", "target_type": target_type, "relation": None, "landmarks": [], "part": None,
            "predicate": pred, "place_type": cands[0], "place_types": cands, "place_part": part, "form": "put",
            **({"place_side": side} if side else {})}


def _rigid_bodies(tdoc) -> set:
    out = set()
    for pd in (tdoc.get("parts") or {}).values():
        if pd and not pd.get("joint"):
            out.update(pd.get("bodies") or [])
    if not out and tdoc.get("root_body"):
        out.add(tdoc["root_body"])
    return out


def _body_z(tdoc, body) -> float:
    z, seen = 0.0, 0
    bodies = tdoc.get("bodies") or {}
    while body in bodies and seen < 20:
        b = bodies[body]
        bp = b.get("body_pos") or [0, 0, 0]
        z += float(bp[2]) if len(bp) > 2 else 0.0
        body = b.get("parent")
        seen += 1
    return z


def on_site(lib, ptype: str, part: str | None):
    """The site name to place ON for a fixture type, or None (place on the object itself)."""
    tdoc = lib.export_types.get(ptype) or {}
    sites = tdoc.get("sites") or {}
    if not sites:
        return None
    rigid = _rigid_bodies(tdoc)
    cand = [(n, s) for n, s in sites.items() if s.get("body") in rigid]
    if part:
        named = [(n, s) for n, s in cand if part in n.split("_")]
        if named:
            cand = named
    if not cand:
        return None

    def z_of(item):
        n, s = item
        pos = s.get("pos") or [0, 0, 0]
        return _body_z(tdoc, s.get("body")) + float(pos[2])

    return max(cand, key=z_of)[0]
