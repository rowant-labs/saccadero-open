"""What the robot does in one episode, decided from the instruction. The runner calls ``plan_episode`` once, after the
episode's first full look.

It returns ``{"agent": ..., "parsed": ..., "grounding": ...}``. ``parsed`` and ``grounding`` go into the episode's record.
``agent`` is None (the robot then sends no-ops for the whole episode) or any object with:
* ``act(obs)`` -> 7 numbers, each step (``obs`` is the step's proprioception);
* ``finished``: True when it is done (the runner then sends no-ops to the step cap);
* ``summary()`` -> a dict with ``steps``, ``finish_reason``, ``attempts`` and ``events``, for the record.

Skills: pick-and-place (``OracleAgent``), reached through the parser in ``ground.py`` or the goal form
``put the X on|in [the top of] the Y`` (``goal.py``); ``on`` a fixture places in the fixture's chosen region.
"""

from __future__ import annotations

from . import make_params
from .agent import OracleAgent
from .ground import ground, parse, single_untold


def _put_parse(instruction, tool, lib):
    try:
        from .goal import parse_put
        parsed = parse_put(instruction, lib)
    except Exception as ex:
        return {"status": "fail", "reason": f"put parse error: {str(ex)[:80]}"}
    if parsed.get("status") == "ok" and len(parsed.get("place_types") or []) > 1:
        seen = {p.get("type") for p in tool.percepts}
        hit = [t for t in parsed["place_types"] if t in seen]
        if hit:
            parsed["place_type"] = hit[0]
    return parsed


def plan_episode(instruction: str, tool, lib, cap: int, params: dict) -> dict:
    if params.get("fixture_no_flip") and hasattr(tool, "no_flip"):
        tool.no_flip = True  # a later look never turns a fixture around in place (``look._flipped``)
        if params.get("fixture_turn_max_deg") and hasattr(tool, "turn_max_deg"):
            # 2026-11-17, off unless set: nor by more than this many degrees (a quarter-turn fit)
            tool.turn_max_deg = float(params["fixture_turn_max_deg"])
    if params.get("relook_strict") and hasattr(tool, "relook_strict"):
        tool.relook_strict = True  # a receiver's believed move stays level and off other objects (``look._move_refused``)
    if params.get("fixture_face_in") and hasattr(tool, "face_in"):
        tool.face_in_on = True  # a fixture's opening parts face the workspace (``look.face_in``)
        if params.get("fixture_face_in") == "carcass" and hasattr(tool, "face_in_mode"):
            tool.face_in_mode = "carcass"  # 2026-11-16: turned about the carcass's middle, no tracker
        tool.face_in()
    if params.get("fixture_memory") and hasattr(tool, "fixture_memory"):
        # 2026-11-17, off by default: a named fixture the look did not find is taken at its remembered place when
        # the look shows something standing there (``look._from_memory``); {workcell: {type: {"pos", "yaw_deg"}}}
        tool.fixture_memory = dict((params["fixture_memory"] or {}).get(getattr(tool, "suite", None)) or {})
    if params.get("sequence"):  # the long-task suite: a compound instruction runs as steps (``sequence.py``)
        from .sequence import SequenceAgent, split

        steps = split(instruction)
        if len(steps) > 1:
            if params.get("keep_target"):
                # ``keep_target`` acts within one step only: an instruction of one step, planned again. In a
                # sequence two steps can share their clause ("put both moka pots on the stove" is twice "put the
                # moka pot on the stove"), and the second step took the first step's remembered pot, which stood on
                # the stove already (the two-pots scenario: one pot of two placed). The steps plan without it.
                params = {k: v for k, v in params.items() if k != "keep_target"}
            if params.get("stage_sweep"):
                steps = _overhang_rewrite(steps, tool, lib, clear=params.get("stage_sweep") != "stage",
                                          open_need=int(params.get("overhang_open_need") or 0))
            steps = _shared_places(steps, lib)
            if hasattr(tool, "budget"):  # the look budget is per step (and one step's worth for the second pass)
                tool.budget = int(tool.budget) * (len(steps) + (1 if params.get("sequence_retry") else 0))
            calls = []
            memo = {}  # what a planned step leaves for a later one (``_plan_special``)

            def planner(clause, step_cap, exclude):
                calls.append(clause)
                for pid, part, ag in memo.get("opens") or []:  # a part an earlier step pulled open stays open
                    if getattr(ag, "finish_reason", None) == "opened" and hasattr(tool, "believe_open"):
                        tool.believe_open(pid, part)
                if len(calls) > 1 and hasattr(tool, "full_look") and tool.n_full + tool.n_track < tool.budget:
                    # a later step plans on the scene as it is now (a drawer the last step opened, an object it
                    # placed): one merged full look, which keeps every percept's id. An object an earlier step
                    # moved aside may be found further off than a look's usual match.
                    try:
                        if memo.get("moved"):
                            tool.full_look(merge=True, free=tuple(memo["moved"]))
                        else:
                            tool.full_look(merge=True)
                    except Exception:
                        pass  # the earlier scene stands
                if exclude and getattr(tool, "place_check", None) and hasattr(tool, "place_base"):
                    tool.place_base()  # the place check's "before" depth, again, with the earlier steps' objects in it
                if isinstance(clause, dict) and clause.get("kind") == "clause" and not clause.get("as_written"):
                    # a clause that shares its place
                    return _plan_one(clause["clause"], tool, lib, step_cap, params, exclude, clause.get("share"))
                if isinstance(clause, dict) and clause.get("kind") != "clause":  # a planned step (``_overhang_rewrite``)
                    return _plan_special(clause, tool, lib, step_cap, params, memo)
                if isinstance(clause, dict):  # the instruction's own clause, carrying only its step estimate
                    clause = clause["clause"]
                planned = _plan_one(clause, tool, lib, step_cap, params, exclude)
                g = (planned or {}).get("grounding") or {}
                if g.get("predicate") == "open" and g.get("part") and planned.get("agent") is not None:
                    memo.setdefault("opens", []).append((g.get("target"), g["part"], planned["agent"]))
                    # the fixture as this step was planned on it, its part still shut (kept for ``take_by_open``)
                    import copy

                    fxp = next((q for q in getattr(tool, "percepts", None) or [] if q.get("id") == g.get("target")), None)
                    if fxp is not None:
                        memo.setdefault("open_fx", {})[(g.get("target"), g["part"])] = copy.deepcopy(fxp)
                    # (for the agent's ``take_turn_early``; read by nothing else) the pose the staging let go in:
                    # the take-again that follows this opening goes back to it
                    rel = getattr((memo.get("stage") or {}).get("agent"), "released", None)
                    if isinstance(rel, dict) and rel.get("R") is not None:
                        try:
                            planned["agent"].next_R = rel["R"]
                        except Exception:
                            pass
                return planned

            # ``sequence_reserve`` (2026-11-16, off by default): the steps kept back for each later step, when set
            # (the built-in 110 is what one pick and place takes at its quickest; a second object into a basket
            # took 101 to 119 on the Mac, a second pot onto the stove about 128)
            ag = SequenceAgent(steps, planner, cap, **({"reserve": int(params["sequence_reserve"])}
                                                       if params.get("sequence_reserve") else {}),
                               **({"retry": True} if params.get("sequence_retry") else {}),
                               **({"chain": True} if params.get("sequence_chain") else {}))
            first = ag.records[-1] if ag.records else {}
            # the record only: when no step is planned, why each one was not (for want of steps, a step it needs,
            # or its own planner's reason)
            why = [{"clause": str(r.get("clause"))[:60], "cap": r.get("cap"), "skipped": r.get("skipped"),
                    "reason": (r.get("grounding") or {}).get("reason")} for r in ag.records] if ag.finished else None
            return {"agent": ag if not ag.finished else None,
                    "parsed": {"status": "ok", "steps": [x.get("clause") if isinstance(x, dict) else x for x in steps]},
                    "grounding": {"status": "ok" if not ag.finished else "fail",
                                  "reason": None if not ag.finished else "no step planned",
                                  "first_step": first.get("grounding"), **({"steps": why} if why else {})}}
    return _plan_one(instruction, tool, lib, cap, params, ())


def named_targets(instruction: str, tool, lib, params: dict) -> list:
    """What the instruction asks the robot to move, read as ``plan_episode`` reads it (2026-11-19; for the learner's
    ``learn_forms`` and for the record). Nothing is planned and the tool is left as it is.

    One entry for every step that picks an object up, in the steps' order: ``step`` and ``steps`` (its index and the
    number of steps: 1 unless the sequencer is on and splits the instruction), ``clause``, ``form`` (``pick``: the
    pick-and-place form of ``ground.parse``; ``put``: the goal form of ``goal.parse_put``), ``type`` (the object's
    type), ``relation``, ``have`` (how many percepts of that type the tool holds now) and ``single_fails`` (with
    exactly one such percept: whether it clearly fails the instruction's relation, ``ground.single_fails``; else
    None). Steps that pick nothing up (open, close, turn on) have no entry."""
    from .ground import is_type, single_fails

    steps = [instruction]
    if params.get("sequence"):
        from .sequence import split

        s = split(instruction)
        if len(s) > 1:
            steps = s
    types = sorted(lib.types)
    out = []
    for i, clause in enumerate(steps):
        parsed, form = parse(clause, types), "pick"
        if parsed.get("status") != "ok":
            parsed, form = _put_parse(clause, tool, lib), "put"
        typ = parsed.get("target_type") if parsed.get("status") == "ok" else None
        if not typ:
            continue
        have = [q for q in tool.percepts if is_type(q, typ)]
        fails = None
        if len(have) == 1:
            try:
                fails = bool(single_fails(parsed, have[0], tool.percepts, lib))
            except Exception:
                fails = None
        out.append({"step": i, "steps": len(steps), "clause": clause, "form": form, "type": typ,
                    "relation": parsed.get("relation"), "have": len(have), "single_fails": fails})
    return out


def choose_cell(instruction: str, cells: dict) -> tuple:
    """(cell, reason): of the workcells the runner offers (``{key: [type, ...]}``; a suite whose tasks use several
    tables, e.g. the long-task suite), the one whose types name every object the instruction names. Each content word
    run that names a type counts (``ground.match_type`` over that cell's types). Ties go to the cell naming the most
    words, then the smaller type list; ``None`` when no cell names anything."""
    from .ground import _content, match_type, words

    ws = [w for w in words(instruction)]
    scored = []
    for key, types in sorted(cells.items()):
        types = sorted(types)
        hits, i = 0, 0
        while i < len(ws):
            found = 0
            for n in range(min(4, len(ws) - i), 0, -1):  # the longest phrase from here that names a type
                if _content(ws[i:i + n]) and match_type(" ".join(ws[i:i + n]), types)[0] is not None:
                    found = n
                    break
            hits += 1 if found else 0
            i += max(found, 1)
        scored.append((hits, -len(types), key))
    if not scored or max(scored)[0] == 0:
        return None, "no cell names the instruction's objects"
    scored.sort(reverse=True)
    return scored[0][2], ("ok" if len(scored) == 1 or scored[1][0] < scored[0][0] else "tie")


BESIDE_CLEAR_M = 0.07  # from the landmark's edge to the point's centre
BESIDE_HALF = (0.04, 0.04, 0.01)  # the point's region (half sizes, m)


def _beside_site(lib, scene: dict, land: dict, side: str):
    """A world-frame region on the table, past the landmark's edge on ``side`` (``ground.SIDE_DIR``)."""
    import numpy as np

    from .ground import CONST, SIDE_DIR, aabb

    d = SIDE_DIR[side] * float(CONST["side_sign"])
    lo, hi = aabb(lib, land)
    c = 0.5 * (np.asarray(lo, float) + np.asarray(hi, float))
    reach = float(np.abs(d) @ (0.5 * (np.asarray(hi, float) - np.asarray(lo, float)))[:2])
    xy = c[:2] + d * (reach + BESIDE_CLEAR_M)
    top = None
    for a in scene.get("arena") or []:
        if a.get("type") == "plane":
            continue
        Rb, h = np.asarray(a["R"], float), np.asarray(a["half"], float)
        ext = np.abs(Rb) @ h
        ac = np.asarray(a["centre"], float)
        if np.all(np.abs(xy - ac[:2]) <= ext[:2]):
            top = max(top if top is not None else -np.inf, float(ac[2] + ext[2]))
    if top is None:
        return None
    return {"parent_instance": None, "frame_body": "world", "pos": [float(xy[0]), float(xy[1]), top + BESIDE_HALF[2]],
            "R": np.eye(3).tolist(), "size": list(BESIDE_HALF)}


# -- an object that the opened part will cover (Goal's "open the top drawer and put the bowl inside": the open drawer
#    hangs 10 cm over the bowl, and no top-down grasp fits under it). The plan: set the object on the fixture's top,
#    by its front edge; open the part; slide the object off the top into it.
STAGE_BACK_M = 0.0  # the staged object's front edge, behind the top site's front edge (itself inside the top's own)
HAND_REACH_M = 0.05  # a top-down hand reaches this far past an object's middle
STAGE_NEED, SWEEP_NEED = 125, 72  # steps kept for these steps when the budget is shared
TAKE_PAST_M = 0.01  # the object is let go this far past the opened region's middle (away from the fixture's face)
SWEEP_PAST_M = 0.07  # the pushing hand ends this far past the top's front edge (the object, ahead of it, tips off)


def _best(percepts: list, typ: str):
    c = [q for q in percepts if q.get("type") == typ]
    return max(c, key=lambda q: float((q.get("fit") or {}).get("n_agree") or 0.0)) if c else None


CLEAR_NEED = 60  # steps kept for the clearing step
CLEAR_M = 0.05  # the cleared object's middle ends this far outside where the part will be (half a hand's thin side + 2 cm)


def _table_box(lib):
    for a in (getattr(lib, "workcell", None) or {}).get("arena") or []:
        if a.get("type") == "box":
            import numpy as np

            c = np.asarray(a["centre"], float)
            return c, np.abs(np.asarray(a.get("R", np.eye(3)), float)) @ np.asarray(a["half"], float)
    return None


def _overhang_rewrite(steps: list, tool, lib, clear: bool = True, open_need: int = 0) -> list:
    """[open the P of F, put the X in the P of F] with X where the opened P will be. With ``clear``, and X open at
    the top with a free way out: [drag X clear, open the P of F, put the X in the P of F] (the last as written: an
    ordinary pick and place, with steps to spare). Otherwise [stage X on F's top, open the P of F, take X again and
    carry it into P]. Anything else comes back unchanged.

    ``open_need`` (``overhang_open_need``, 2026-11-17, 0: as before): in the staged plan, the steps kept back for
    the opening step while the staging runs. As built the opening is an ordinary clause and the sequence keeps its
    whole reserve for it (130 with the bundle), so the staging's own cap is 300 - 130 - 72 = 96 steps (94 on the
    Mac) for a step that takes 86 to 93 when nothing goes wrong: a second grasp, or any perturbation that costs
    it ten steps, ends it for want of steps with sixty left in the episode. The opening itself takes about 80."""
    if len(steps) != 2 or not all(isinstance(x, str) for x in steps):
        return steps
    try:
        import numpy as np

        from .drawer import open_footprint, parse_open
        from .goal import on_site, parse_put
        from .ground import aabb

        po = parse_open(steps[0], lib, [q.get("type") for q in tool.percepts])
        pp = parse_put(steps[1], lib)
        if po.get("status") != "ok" or pp.get("status") != "ok" or pp.get("predicate") != "in":
            return steps
        typ = po["target_type"]
        if typ not in (pp.get("place_types") or [pp.get("place_type")]):
            return steps
        if _best(tool.percepts, typ) is None and hasattr(tool, "on_parsed"):
            tool.on_parsed({**po, "place_type": None})
            if hasattr(tool, "face_in"):
                tool.face_in()
        fx, tg = _best(tool.percepts, typ), _best(tool.percepts, pp["target_type"])
        if fx is None or tg is None:
            return steps
        tdoc = lib.export_types[typ]
        site = _in_site(lib, fx, pp.get("place_part"))
        if site is None or tdoc["sites"][site]["body"] not in tdoc["parts"][po["part"]]["bodies"]:
            return steps  # the put names another part than the opened one
        if on_site(lib, typ, "top") is None:
            return steps
        fp = open_footprint(lib, fx, po["part"])
        if fp is None:
            return steps
        lo, hi = aabb(lib, tg)
        under = bool(np.all(lo[:2] - HAND_REACH_M < fp[1][:2]) and np.all(hi[:2] + HAND_REACH_M > fp[0][:2]))
        gap = float(fp[0][2] - hi[2])
        if not under or not 0.0 < gap < 0.14:
            return steps
        common = {"target": tg["id"], "fixture": fx["id"], "part": po["part"], "site": site}
        if clear:
            from .drawer import cavity, clear_move

            cav = cavity(lib, tg)
            if cav is not None and clear_move(lib, tool.percepts, tg, fp, CLEAR_M, _table_box(lib),
                                              hand_z=cav[0] + 0.06) is not None:
                return [{**common, "kind": "clear", "need": CLEAR_NEED, "min": 40, "clause": f"clear, for: {steps[0]}"},
                        steps[0], steps[1]]
        opening = {"kind": "clause", "as_written": True, "clause": steps[0], "need": int(open_need)} \
            if open_need > 0 else steps[0]
        return [{**common, "kind": "stage", "need": STAGE_NEED, "clause": f"stage, for: {steps[1]}"}, opening,
                {**common, "kind": "take", "need": SWEEP_NEED, "min": 45, "requires": 1,
                 "clause": f"take again, for: {steps[1]}"}]
    except Exception:
        return steps


def _site_world(lib, percept: dict, site: str, poses: dict | None = None):
    """(centre, half extents) of one of the percept's sites in the world (``poses``: body poses to use instead)."""
    import numpy as np

    sd = lib.export_types[percept["type"]]["sites"][site]
    if poses is not None:
        Rb, tb = (np.asarray(x, float) for x in poses[sd["body"]])
    else:
        b = percept["bodies"][sd["body"]]
        Rb, tb = lib.quat_mat(b["quat_wxyz"]), np.asarray(b["pos"], float)
    return tb + Rb @ np.asarray(sd["pos"], float), np.abs(Rb @ np.asarray(sd["R"], float)) @ np.asarray(sd["size"], float)


def _percept_moved(lib, a: dict, b: dict) -> dict:
    """The record only: how far percept ``b``'s root body lies from ``a``'s (m, across) and how far it is turned
    about the vertical (deg)."""
    import numpy as np

    try:
        root = lib.export_types[a["type"]]["root_body"]
        pa, pb = (np.asarray(x["bodies"][root]["pos"], float) for x in (a, b))
        Ra, Rb = (np.asarray(lib.quat_mat(x["bodies"][root]["quat_wxyz"]), float) for x in (a, b))
        dR = Rb @ Ra.T
        return {"shift_m": round(float(np.linalg.norm((pb - pa)[:2])), 4),
                "turn_deg": round(float(np.degrees(np.arctan2(dR[1, 0], dR[0, 0]))), 1)}
    except Exception:
        return {}


def _pull_short(memo: dict, fixture, part) -> float:
    """How far short of its end the opening step left the part (m, 0 to 0.02): its pull is the part's travel plus
    1.5 cm (``handle_plan``), and the step records how far the hand came."""
    for pid, prt, ag in reversed(memo.get("opens") or []):
        if pid == fixture and prt == part and getattr(ag, "finish_reason", None) == "opened":
            try:
                att = [a for a in ag.attempts if a.get("skill") == "open" and a.get("outcome") == "opened"][-1]
                return float(min(0.02, max(0.0, float(att["pull"]) - 0.015 - float(att["moved"]))))
            except (IndexError, KeyError, TypeError, ValueError):
                return 0.0
    return 0.0


def _plan_special(step: dict, tool, lib, cap: int, params: dict, memo: dict | None = None) -> dict:
    """A planned step of ``_overhang_rewrite``: ``stage`` (set the object on the fixture's top, by its front edge),
    ``take`` (take it again where the hand let go and carry it into the opened part) or ``sweep`` (slide it off the
    top into the opened part: the fallback when the staging left no release pose). ``memo`` carries the staging
    step's agent and numbers to the later step."""
    import numpy as np

    from . import make_params
    from .drawer import RetakeAgent, SweepAgent, handle_plan, open_poses
    from .goal import on_site
    from .ground import aabb

    memo = memo if memo is not None else {}
    fx = next((q for q in tool.percepts if q["id"] == step["fixture"]), None)
    tg = next((q for q in tool.percepts if q["id"] == step["target"]), None)
    parsed = {"status": "ok", "form": step["kind"], "target_type": (tg or {}).get("type"), "predicate": step["kind"]}
    if step["kind"] == "clear":
        from .drawer import DragAgent, cavity, clear_move, open_footprint

        if fx is None or tg is None:
            return {"agent": None, "parsed": parsed, "grounding": {"status": "fail", "reason": "percept gone"}}
        fpr = open_footprint(lib, fx, step["part"])
        cav = cavity(lib, tg)
        mv = clear_move(lib, tool.percepts, tg, fpr, CLEAR_M, _table_box(lib),
                        hand_z=cav[0] + 0.06) if (fpr is not None and cav) else None
        if mv is None:
            return {"agent": None, "parsed": parsed, "grounding": {"status": "fail", "reason": "no free way out"}}
        root = lib.export_types[tg["type"]]["root_body"]
        c = np.asarray(tg["bodies"][root]["pos"], float)
        ap, pp, fp = make_params(params.get("m1"))
        tool.target = tg["id"]
        spec = {"predicate": "clear", "target": tg["id"], "place": None, "place_kind": None, "place_instance": None,
                "instruction": step["clause"], "part": step["part"]}
        agent = DragAgent(lib, lib.scene_from_percepts(tool.percepts), spec, tool, cap, ap, pp, fp)
        agent.hp = {"p_in": [float(c[0]), float(c[1]), cav[0] + 0.02], "move": mv["move"], "safe_z": cav[1] + 0.05}
        memo.setdefault("moved", []).append(tg["id"])
        return {"agent": agent, "parsed": parsed,
                "grounding": {"status": "ok", "target": tg["id"], "place": None,
                              "move": [round(float(x), 4) for x in mv["move"]], "floor": round(cav[0], 4)}}
    st = memo.get("stage") or {}
    rel = getattr(st.get("agent"), "released", None)
    if step["kind"] == "take" and fx is not None and rel is not None:
        # ``take_by_open`` (2026-11-16, off by default): when the look taken since has the fixture TURNED from how
        # the opening step was planned on it (by ``take_open_turn_deg`` or more: 20 degrees unless set, 0: always),
        # where the opened part is, is taken from the fixture as that step had it: the drawer came out along that
        # pose's own direction, so the hand has verified which way it faces, and a fixture does not turn. With its
        # drawer out and the object on its top the cabinet was fitted a quarter turn off and 4 to 6 cm aside (colour
        # line, the plain drawer task at the scored start, eight runs: 7 of 192 planned take-agains, in every arm,
        # none solved): the carry went 16 cm sideways instead of 13 cm into the drawer, the step ended "placed" and
        # the object lay beside the cabinet. (A half turn was already refused, ``fixture_no_flip``.)
        # Moved but not turned, the look since is kept, as before: in those rows the two looks agree within 5 mm in
        # 163 of the 192 and within 1 cm in 176, and of the 5 that differ by 1.5 to 3 cm (all solved by the look
        # since) 4 are the first look's fault (the cabinet 3 cm too far back in it). What the look made of the
        # fixture is recorded either way (``fx_moved``).
        fx_plan, how = fx, {}
        snap = (memo.get("open_fx") or {}).get((step["fixture"], step["part"]))
        for pid_, part_, ag_ in memo.get("opens") or []:  # ``open_relook``: the fixture as the opening ended up on it
            if pid_ == step["fixture"] and part_ == step["part"] and getattr(ag_, "fx_used", None) is not None:
                snap = ag_.fx_used
        if snap is not None:
            how["fx_moved"] = _percept_moved(lib, snap, fx)
            turn = how["fx_moved"].get("turn_deg")
            if params.get("take_by_open") and turn is not None \
                    and abs(float(turn)) >= float(params.get("take_open_turn_deg", 20.0)):
                fx_plan, how["by_open"] = snap, True
        hp = handle_plan(lib, fx_plan, step["part"])
        if hp is None:
            return {"agent": None, "parsed": parsed, "grounding": {"status": "fail", "reason": "no handle"}}
        d = np.asarray(hp["d"], float)
        c_reg, _e = _site_world(lib, fx_plan, step["site"], poses=open_poses(lib, fx_plan, step["part"]))
        goal = c_reg[:2] + d[:2] * TAKE_PAST_M
        place = next((a.get("place") for a in reversed(st["agent"].attempts) if isinstance(a.get("place"), dict)), {})
        start = np.asarray(st["xy"], float)
        if how.get("by_open"):
            # the part is out as far as the hand pulled it (with ``pull_end_early`` up to 5 mm short of its end)
            short = _pull_short(memo, step["fixture"], step["part"])
            if short:
                goal, how["short_m"] = goal - d[:2] * short, round(short, 4)
            # and the object stands where the hand let it go, which is not quite where the staging aimed: the hand's
            # point less the grasp's own reach from the object's origin (the look before the grasp and the model,
            # not the look at the lifted object, which on the colour line reads centimetres off); without that
            # record, the stage point moved by as much as the hand ended off its planned point (that release is
            # made within 1 cm)
            try:
                if rel.get("hold_xy") is not None:
                    c_obj = np.asarray(rel["p"], float)[:2] - np.asarray(rel["hold_xy"], float)
                    if float(np.linalg.norm(c_obj - start)) <= 0.04:
                        how["staged_off"], start = [round(float(x), 4) for x in c_obj - start], c_obj
                else:
                    off = np.asarray(rel["p"], float)[:2] - np.asarray(place["xy"], float)[:2]
                    if float(np.linalg.norm(off)) <= 0.02:
                        start, how["staged_off"] = start + off, [round(float(x), 4) for x in off]
            except (KeyError, TypeError, ValueError):
                pass
        move = goal - start
        # ``take_near_rim`` (2026-11-16, off by default): an object the staging held on its far side (seen along
        # the way the part opens: the asked side was blocked where it was picked) is taken again on its near side,
        # at the same height and with the hand turned the same way: the point opposite, across the object's own
        # axis, when the object is the same turned half a turn. The far side lies 9 cm deeper over the fixture, at
        # the end of the arm's reach: colour line, the plain drawer task at the scored start, eight runs of 30:
        # staged by the near rim 133 of 140 done, by the far rim 27 of 75 (the reach down refused or the arm
        # caught, the bowl let go on the cabinet's edge, the steps run out)
        p_take = [float(x) for x in rel["p"]]
        hold = rel.get("hold_xy")
        if params.get("take_near_rim") and hold is not None and rel.get("half_turn"):
            hold = np.asarray(hold, float)
            if float(hold @ d[:2]) < -0.02:
                p_take[0], p_take[1] = p_take[0] - 2.0 * float(hold[0]), p_take[1] - 2.0 * float(hold[1])
                how["near_rim"] = [round(p_take[0], 4), round(p_take[1], 4)]
        surface = float(place.get("surface", rel["p"][2] - 0.045))
        drop = float(np.clip((rel["p"][2] - surface) - (rel["frame_z"] - st["lo_z"]), 0.0, 0.04))
        ap, pp, fp = make_params(params.get("m1"))
        # the agent's own target is the fixture (as when opening): the object's percept may be gone, and is not needed
        spec = {"predicate": "take", "target": fx["id"], "object": step["target"], "place": None, "place_kind": None,
                "place_instance": None, "instruction": step["clause"], "part": step["part"]}
        agent = RetakeAgent(lib, lib.scene_from_percepts(tool.percepts), spec, tool, cap, ap, pp, fp)
        agent.hp = {"p": p_take, "R": rel["R"], "drop": drop, "move": [float(move[0]), float(move[1])]}
        return {"agent": agent, "parsed": parsed,
                "grounding": {"status": "ok", "target": step["target"], "place": fx["id"],
                              "move": [round(float(x), 4) for x in move], "drop": round(drop, 4), **how}}
    if fx is None or tg is None:
        return {"agent": None, "parsed": parsed, "grounding": {"status": "fail", "reason": "percept gone"}}
    hp = handle_plan(lib, fx, step["part"])
    top = on_site(lib, fx["type"], "top")
    if hp is None or top is None:
        return {"agent": None, "parsed": parsed, "grounding": {"status": "fail", "reason": "no handle or top"}}
    d = np.asarray(hp["d"], float)
    c_top, e_top = _site_world(lib, fx, top)
    front = float(c_top[:2] @ d[:2]) + float(np.abs(d[:2]) @ e_top[:2])  # the top's front edge along d
    lo, hi = aabb(lib, tg)
    r_obj = 0.5 * float(np.abs(d[:2]) @ (hi - lo)[:2])
    line = np.asarray(hp["c"], float)[:2]  # the part's middle line (its handle)
    ap, pp, fp = make_params(params.get("m1"))
    scene = lib.scene_from_percepts(tool.percepts)
    if step["kind"] == "stage":
        xy = line + d[:2] * (front - r_obj - STAGE_BACK_M - float(line @ d[:2]))
        scene["sites"]["stage_point"] = {"parent_instance": None, "frame_body": "world",
                                         "pos": [float(xy[0]), float(xy[1]), float(c_top[2]) + 0.005],
                                         "R": np.eye(3).tolist(), "size": [0.03, 0.03, 0.005]}
        tool.target = tg["id"]
        # the hand takes the object on the side the part opens to: clear of the fixture when it picks, nearer when it
        # sets down, and where the later step takes it again
        spec = {"predicate": "on", "target": tg["id"], "place": "stage_point", "place_kind": "region",
                "place_instance": fx["id"], "instruction": step["clause"], "grasp_side": [float(d[0]), float(d[1])]}
        from dataclasses import replace

        from .agent import OracleAgent

        ap = replace(ap, trust_place=True)  # one step of the plan: released is placed
        agent = OracleAgent(lib, scene, spec, tool, cap, ap, pp, fp)
        memo["stage"] = {"agent": agent, "xy": [float(xy[0]), float(xy[1])], "lo_z": float(lo[2])}
        return {"agent": agent, "parsed": parsed,
                "grounding": {"status": "ok", "target": tg["id"], "place": fx["id"],
                              "stage": [round(float(x), 4) for x in xy]}}
    # sweep: from behind the object, along d, until the hand reaches the top's front edge
    root = lib.export_types[tg["type"]]["root_body"]
    p_t = np.asarray(tg["bodies"][root]["pos"], float)
    c_reg, _e = _site_world(lib, fx, step["site"], poses=open_poses(lib, fx, step["part"]))
    want = float((c_reg[:2] - p_t[:2]) @ d[:2])  # how far the object should travel
    start_xy = p_t[:2] - d[:2] * (r_obj + 0.03)
    # the hand (the object rests against its part-open fingers, its middle ahead of them) ends just past the top's
    # front edge, where the object's middle is well past it and it tips off; never past the region's middle
    end = min(front + SWEEP_PAST_M, float(c_reg[:2] @ d[:2]))
    dist = end - float(start_xy @ d[:2])
    if dist < 0.03:
        return {"agent": None, "parsed": parsed, "grounding": {"status": "fail", "reason": "nothing to sweep"}}
    spec = {"predicate": "sweep", "target": tg["id"], "place": None, "place_kind": None, "place_instance": None,
            "instruction": step["clause"], "part": step["part"]}
    agent = SweepAgent(lib, scene, spec, tool, cap, ap, pp, fp)
    e = np.array([-d[1], d[0]])
    agent.hp = {"d": [float(d[0]), float(d[1]), 0.0], "start": [float(start_xy[0]), float(start_xy[1]), float(lo[2]) + 0.03],
                "dist": dist, "safe_z": float(hi[2]) + 0.04, "obj": [float(p_t[0]), float(p_t[1])], "r": r_obj,
                "width": float(np.abs(e) @ (hi - lo)[:2])}
    return {"agent": agent, "parsed": parsed,
            "grounding": {"status": "ok", "target": tg["id"], "place": None, "want": round(want, 4),
                          "dist": round(dist, 4)}}


def _in_site(lib, place_p: dict, part):
    """The site to place IN: the library's choice, or, when a part word names several sites (a top drawer's region and
    the fixture's top side), the one that is a containing region."""
    s = lib.site_name_for(place_p, part)
    if s is not None or not part:
        return s
    sites = (lib.export_types.get(place_p["type"]) or {}).get("sites") or {}
    hit = [n for n in sorted(sites) if part in n.split("_") and "region" in n.split("_")]
    return hit[0] if len(hit) == 1 else None


def _shared_places(steps: list, lib) -> list:
    """Clauses that put different things in or on the same place ("both the A and the B in the C", "both As on the
    C") are marked with their turn, so each release takes its own part of the region (``agent._share_offset``)."""
    from .goal import parse_put

    keys = []
    for st in steps:
        k = None
        if isinstance(st, str):
            try:
                pp = parse_put(st, lib)
            except Exception:
                pp = {}
            if pp.get("status") == "ok" and not pp.get("place_beside") and not pp.get("place_side"):
                k = (pp.get("place_type"), pp.get("place_part"), pp.get("predicate"))
        keys.append(k)
    out = []
    for i, st in enumerate(steps):
        n = sum(1 for k in keys if k is not None and k == keys[i])
        if keys[i] is not None and n > 1:
            turn = sum(1 for k in keys[:i] if k == keys[i])
            out.append({"kind": "clause", "clause": st, "share": [turn, n]})
        else:
            out.append(st)
    return out


def _look_then(instruction: str, tool, lib, cap: int, params: dict, exclude, share, parsed: dict, point: dict,
               why: str | None = None, after: dict | None = None, kind: str = "look_behind",
               look_opts: dict | None = None) -> dict:
    """The target is not in view and something hangs in front of the scene: first a look behind it with the wrist
    camera (``search.LookAgent``), then the instruction as planned from what that look found (or, when it found
    nothing, as it would have been planned without it). ``after``, ``kind``, ``look_opts`` (``relation_look``): the
    parameters the second step is planned with, the step's name and the look step's own options, when the look is
    for another reason."""
    from . import make_params
    from .search import LookAgent
    from .sequence import SequenceAgent

    bh = params.get("behind_occluder")
    if after is None:
        after = {**params, "behind_occluder": {**(bh if isinstance(bh, dict) else {}), "mode": "done",
                                               **({"why": why} if why else {})}}
    _ap, _pp, fp = make_params(params.get("m1"))

    def planner(step, step_cap, _placed):
        if isinstance(step, dict):
            return {"agent": LookAgent(tool, parsed["target_type"], point, step_cap, fp,
                                       look_opts if look_opts is not None else
                                       (bh if isinstance(bh, dict) else {}).get("look")),
                    "parsed": {"status": "ok", "form": kind, "target_type": parsed["target_type"]},
                    "grounding": {"status": "ok", "target": None, "place": None,
                                  "point": [round(float(x), 4) for x in point["xy"]]}}
        if hasattr(tool, "full_look") and tool.n_full + tool.n_track < tool.budget:
            try:  # the hand has moved since the first look: the scene again, as the main camera has it now
                tool.full_look(merge=True)
            except Exception:
                pass
        return _plan_one(step, tool, lib, step_cap, after, exclude, share)

    said = "look behind" if kind == "look_behind" else "look where it should be"
    ag = SequenceAgent([{"kind": kind, "clause": f"{said}, for: {instruction}", "min": 40}, instruction],
                       planner, cap)
    return {"agent": ag if not ag.finished else None, "parsed": parsed,
            "grounding": {"status": "ok" if not ag.finished else "fail", "reason": None if not ag.finished else "no step planned",
                          "flags": [kind], "point": [round(float(x), 4) for x in point["xy"]]}}


LOOK_SUPPORT_OVER_TABLE_M = 0.06  # ``relation_look``: no look at a place that stands higher than this above the table


def _relation_look(instruction: str, tool, lib, cap: int, params: dict, exclude, share, parsed: dict, percepts: list,
                   stand_ins) -> dict | None:
    """``params["relation_look"]`` (2026-11-16, off by default): the one object of the named kind in view does not
    stand in the instruction's relation at all, and nothing else the first look saw does (``relation_candidates``
    added no candidate). The named object is then one the main camera did not make out (recoloured, reshaped, drawn
    at another size, half hidden); carrying the other one cannot be right. Before choosing, the robot looks from
    above at the place the relation gives (``ground.relation_point``) with the wrist camera's colour picture
    (``_look_then``, ``search.LookAgent``): a thing found there that no percept explains becomes a second candidate,
    and the relation chooses between the two. Nothing found: the plan is as it was without the look.
    Returns the two-step plan, or None when the look is not called for (or was already taken)."""
    from .ground import is_type, landmarks_of, relation_holds, relation_point

    rl = params.get("relation_look")
    opts = rl if isinstance(rl, dict) else {}
    if opts.get("mode") == "done" or not hasattr(tool, "wrist_find") or not parsed.get("relation"):
        return None
    if getattr(tool, "relation_target", None) is not None:
        return None  # a candidate that stands in the relation has joined already
    typ = parsed.get("target_type")
    cands = [q for q in percepts if is_type(q, typ)]
    if len(cands) != 1 or relation_holds(parsed, cands[0], percepts, lib, lib.workcell, stand_ins) is not False:
        return None
    point = relation_point(parsed, percepts, lib, lib.workcell, stand_ins)
    if point is None:
        return None
    if parsed["relation"] in ("on", "in"):
        # on or in a fixture (reported, or a stand-in for one): a reported fixture's own box is left out of the wrist
        # picture with all that stands on it, a fixture's top is no plain support to tell a thing from by its colour,
        # and 27 cm above a cabinet's top is out of the arm's reach. No look there.
        land = (landmarks_of(parsed, percepts, stand_ins) or [None])[0]
        if isinstance(land, dict) and (lib.export_types.get(land.get("type")) or {}).get("kind") == "fixture":
            return None
    try:
        # nor at a place that stands high (a box set on the cabinet): the hand looks from 27 cm above the support,
        # and over the far half of the table it reaches 1.23 m, a third of a metre above the table top
        from .behind import table_z

        tz = table_z(lib, point["xy"])
        if tz is not None and float(point["support_z"]) > float(tz) + LOOK_SUPPORT_OVER_TABLE_M:
            return None
    except Exception:
        return None
    if getattr(tool, "table_bound", False) and hasattr(tool, "_over_table") \
            and not tool._over_table([point["xy"][0], point["xy"][1], point["support_z"]]):
        return None
    after = {**params, "relation_look": {**opts, "mode": "done"}}
    bh = params.get("behind_occluder")
    if bh is not None and bh is not False:  # (one look an episode: no look behind something after this one)
        after["behind_occluder"] = {**(bh if isinstance(bh, dict) else {}), "mode": "done"}
    if hasattr(tool, "wrist_opts"):
        # among the things the wrist picture shows where the named object can stand, the one of its size (a slab
        # beside a hidden bowl is as near the landmark as the bowl is); nothing that near: nothing found
        tool.wrist_opts = {"prefer": "size", "within_m": float(point["within_m"]) + 0.03, **(opts.get("wrist") or {})}
    out = _look_then(instruction, tool, lib, cap, params, exclude, share, parsed, point, after=after, kind="look_at",
                     look_opts={"confirm": True, **(opts.get("look") or {})})
    if out.get("agent") is None:
        if hasattr(tool, "wrist_opts"):
            tool.wrist_opts = None  # no look was planned: the tool's wrist pictures stay as they were
        return None
    out["grounding"]["relation"] = parsed["relation"]
    return out


KEEP_SAME_M = 0.05   # ``keep_target``: a candidate this near the remembered target's place is the target
KEEP_OTHER_M = 0.05  # and one this near (across) a remembered other object of its type is that other object


def _keep_target(instruction: str, tool, lib, parsed: dict, grounding: dict, exclude=()) -> dict:
    """``params["keep_target"]`` (2026-11-16, off by default): the object an instruction names does not change during
    an episode. When the same instruction is planned again (the runner looks and plans again once a delay is
    measured or the arm was freed; a step is tried again), the target is the object the first plan chose, not what
    the relation picks in the scene as it is now: the robot's own hand hides or has moved things since (start
    state 0, one episode: planned again at step 33 with the hand over the scene, the landmark was not in the look,
    and the other bowl was carried). In the new look's percepts of the target's type:
    * one at the remembered target's place is the target;
    * else those standing where the remembered OTHER objects of the type stood are those objects; exactly one
      candidate that is none of them is the target, where it lies now (knocked, dropped);
    * else, with only the others in view, the target is not in view: its remembered percept joins the scene and is
      the target (under the hand it is tracked from there, or the hand closes on nothing and the search goes on).
    Returns the grounding, changed only when the choice changes (flag ``kept_target``)."""
    import numpy as np

    from .ground import is_type

    memo = tool.prior.get(instruction)
    typ = parsed.get("target_type")
    if not memo or memo.get("type") != typ or typ not in lib.export_types:
        return grounding
    root = lib.export_types[typ]["root_body"]

    def pos(q):
        return np.asarray(q["bodies"][root]["pos"], float)

    try:
        cands = [q for q in tool.percepts if is_type(q, typ) and root in (q.get("bodies") or {})
                 and not (exclude and q["id"] in exclude)]
        T = np.asarray(memo["pos"], float)
        same = [q for q in cands if float(np.linalg.norm(pos(q) - T)) <= KEEP_SAME_M]
        if same:
            tgt = min(same, key=lambda q: float(np.linalg.norm(pos(q) - T)))
        else:
            others = [np.asarray(o, float) for o in memo.get("others") or []]
            unknown = [q for q in cands if all(float(np.linalg.norm((pos(q) - o)[:2])) > KEEP_OTHER_M for o in others)]
            if len(unknown) == 1:
                tgt = unknown[0]
            elif not unknown:
                ids = {q.get("id") for q in tool.percepts}
                pid = next(f"m{k}" for k in range(1, 100) if f"m{k}" not in ids)
                tgt = {**memo["percept"], "id": pid, "source": "remembered"}
                tool.percepts = list(tool.percepts) + [tgt]
                tool.fallbacks.append({"role": "target", "type": typ, "id": pid, "source": "remembered",
                                       "held": bool(memo.get("held"))})
            else:
                return grounding
    except Exception:
        return grounding
    if tgt["id"] == grounding.get("target"):
        return grounding
    return {**grounding, "target": tgt["id"], "flags": list(grounding.get("flags") or []) + ["kept_target"]}


AT_PLACE_M = 0.01  # a target whose origin lies inside its place region grown by this is at its place already


def _at_place(lib, scene: dict, percepts: list, spec: dict) -> bool:
    """Whether the step's target already lies in or on the step's place region, as the percepts have it (the agent's
    own placed test, ``OracleAgent._c3``, without its shrink)."""
    import numpy as np

    from .agent import pose_of

    try:
        site = scene["sites"][spec["place"]]
        poses = lib.poses_from_percepts(percepts)
        Rb, tb = pose_of(poses, site["frame_body"])
        c = np.asarray(tb, float) + np.asarray(Rb, float) @ np.asarray(site["pos"], float)
        half = np.abs(np.asarray(Rb, float) @ np.asarray(site["R"], float) @ np.asarray(site["size"], float))
        _R, pos = pose_of(poses, scene["instances"][spec["target"]]["root_body_name"])
        pos = np.asarray(pos, float)
    except Exception:
        return False
    if not bool(np.all(np.abs(pos[:2] - c[:2]) <= half[:2] + AT_PLACE_M)):
        return False
    if spec.get("predicate") == "on":
        return bool((c[2] - half[2] - 0.03) < pos[2] < (c[2] + half[2] + 0.25))
    return bool((c[2] - half[2] - 0.01) < pos[2] < (c[2] + half[2]))


def _plan_one(instruction: str, tool, lib, cap: int, params: dict, exclude=(), share=None) -> dict:
    """One step: the instruction's single-step forms (pick, put, open). ``exclude``: percept ids an earlier step of a
    sequence placed; they are not chosen as this step's target."""
    parsed = parse(instruction, sorted(lib.types))
    if parsed["status"] != "ok":
        alt = _put_parse(instruction, tool, lib)
        if alt.get("status") != "ok":
            try:
                from .drawer import plan_open
                op = plan_open(instruction, tool, lib, cap, params)
            except Exception:
                op = None
            if op is not None:
                return op
            if exclude is not None and params.get("sequence"):  # the step sequencer's "close the X drawer" clauses
                try:
                    from .drawer import plan_close
                    op = plan_close(instruction, tool, lib, cap, params)
                except Exception:
                    op = None
                if op is not None:
                    return op
            return {"agent": None, "parsed": parsed, "grounding": {"status": "fail", "reason": parsed.get("reason")}}
        parsed = alt
    if params.get("fallback_on_table") and hasattr(tool, "table_bound"):
        tool.table_bound = True  # the long-task suite: a fallback percept must lie over the table
    if params.get("blob_rest_up") and hasattr(tool, "blob_rest_up"):
        tool.blob_rest_up = True  # 2026-11-16, off by default: a model set on an unexplained region stands as it rests
    if params.get("seen_target") and hasattr(tool, "seen_target"):
        tool.seen_target = params["seen_target"]  # 2026-11-16, off by default: an unexplained target as it is seen
    bh = params.get("behind_occluder")  # 2026-11-16, off by default: a target not in view is looked for behind a
    if bh is not None and bh is not False and hasattr(tool, "add_behind"):  # thing that hangs in front of the scene
        tool.behind = dict(bh) if isinstance(bh, dict) else {}
    fs = params.get("fallback_stand")  # 2026-11-16 night, off by default: a named target with no percept is stood
    if fs is not None and fs is not False and hasattr(tool, "stand"):  # under an unexplained region's top layer
        from .look import STAND

        tool.stand = {**STAND, **(dict(fs) if isinstance(fs, dict) else {})}
    if params.get("fallback_top") and hasattr(tool, "fallback_top"):
        tool.fallback_top = True  # a named type with no percept follows the evidence (``look._from_evidence``)
    mr = params.get("misread_target")  # 2026-11-17, off by default: the named target read as another type
    if mr is not None and mr is not False and hasattr(tool, "misread"):
        from .look import MISREAD

        tool.misread = {**MISREAD, **(dict(mr) if isinstance(mr, dict) else {})}
    if params.get("fallback_own") and hasattr(tool, "fallback_own"):
        tool.fallback_own = True  # 2026-11-17, off by default: a reported object's own leftover is no receiver
    if params.get("fallback_blob_clear") and hasattr(tool, "blob_clear_first"):
        tool.blob_clear_first = True  # among blobs, one clear of the reported objects first (2026-11-16, off by default)
    if params.get("fallback_blob_on_table") and hasattr(tool, "blob_on_table"):
        tool.blob_on_table = True  # a percept built from a blob never stands lower than the table top under it
    if hasattr(tool, "on_parsed"):  # may add percepts, knowing what the instruction names
        tool.on_parsed(parsed)
        if hasattr(tool, "face_in"):
            tool.face_in()  # a fallback percept of a fixture too

    def seen():
        return tool.percepts if not exclude else [
            q for q in tool.percepts if q["id"] not in exclude or q["type"] != parsed.get("target_type")]

    stand_ins = ()
    if params.get("relation_single") and hasattr(tool, "relation_candidates"):
        # the relation asked of a single candidate, and stand-ins for landmarks with no percept
        if params.get("relation_standin_strict") and hasattr(tool, "standin_strict"):
            tool.standin_strict = True
        stand_ins = tuple(tool.relation_candidates(parsed) or ())
    percepts = seen()
    if stand_ins and params.get("relation_standin_strict"):
        # 2026-11-16, off by default: a landmark's stand-in is a rejected pose, and it can be wrong. Among several
        # candidates it is believed only when it singles one of them out (exactly one stands in the relation to
        # it); else the choice is made as with the landmark missing. (Both starts, with stand-ins: 1 success in 7
        # episodes; truth: two carried the other bowl of two reported ones, two took a rejected pose over the
        # reported bowl that was the instructed one. With the landmark simply missing: 37 of 42.)
        try:
            from .ground import is_type, relation_holds

            cs = [q for q in percepts if is_type(q, parsed.get("target_type"))]
            if len(cs) >= 2 and sum(relation_holds(parsed, q, percepts, lib, lib.workcell, stand_ins) is True
                                    for q in cs) != 1:
                stand_ins = ()
        except Exception:
            stand_ins = ()
    if params.get("relation_in_region") and parsed.get("relation") == "in" and parsed.get("part"):
        # 2026-11-16, off by default: "in the top layer of the cabinet" names the cabinet's top REGION. The part word
        # alone matches two of its sites (the region inside and the side on top), the library then names neither,
        # and with no region the relation cannot be told: both bowls cost the same and the first one listed was
        # taken (the wrong one in 2 of 18 episodes of that task at start state 8, right in the others by its place
        # in the list). The part is read as "<part> region" when that names a site and the part alone does not.
        try:
            from .ground import landmarks_of

            land = (landmarks_of(parsed, percepts, stand_ins) or [None])[0]
            if land and "bodies" in land and lib.site_for(land, parsed["part"]) is None \
                    and lib.site_for(land, f"{parsed['part']} region") is not None:
                parsed = {**parsed, "part": f"{parsed['part']} region"}
        except Exception:
            pass
    strict = getattr(tool, "behind", None) is not None
    mode = (tool.behind or {}).get("mode", "look") if strict else None
    extra = {"extra_landmarks": stand_ins} if stand_ins else {}
    grounding = ground(parsed, percepts, lib, lib.workcell, strict_single=True, **extra) if strict else \
        ground(parsed, percepts, lib, lib.workcell, **extra)
    # ``behind_occluder["untold"]`` (2026-11-16, off by default): the one candidate in view cannot be told to be the
    # one meant, because the relation's landmark is not in view either (``ground.single_untold``)
    untold = bool(mode == "look" and (tool.behind or {}).get("untold") and grounding.get("status") == "ok"
                  and single_untold(parsed, percepts))
    if mode == "look" and parsed.get("target_type") and (grounding.get("hidden") or untold or not any(
            q.get("type") == parsed["target_type"] or q.get("named") == parsed["target_type"] for q in percepts)):
        # the target is not in view: none of its type, or only one that is not the one the instruction means. With
        # something hanging in front of the scene, the robot first looks behind it with the wrist camera
        point = tool.behind_point(parsed["target_type"])
        if point is not None:
            return _look_then(instruction, tool, lib, cap, params, exclude, share, parsed, point,
                              **({"why": "untold"} if untold else {}))
    if strict and grounding.get("hidden"):
        # no look was possible. Mode "ray": the target is placed behind the middle of what hangs there, unseen, and
        # the choice is made again between the two; otherwise the one in view is taken as before
        got = tool.add_behind(parsed["target_type"]) if mode == "ray" else None
        percepts = seen()
        grounding = ground(parsed, percepts, lib, lib.workcell, **extra)
        grounding.setdefault("flags", []).append("behind" if got is not None else "single_fails_relation")
    rl = params.get("relation_look")  # 2026-11-16, off by default: see ``_relation_look``
    if rl is not None and rl is not False and grounding.get("status") == "ok":
        looked = _relation_look(instruction, tool, lib, cap, params, exclude, share, parsed, percepts, stand_ins)
        if looked is not None:
            return looked
    added = getattr(tool, "relation_target", None) if params.get("relation_single") else None
    if added is not None and grounding.get("status") == "ok" and grounding.get("target") != added:
        # the candidate that joined stands in the relation and the reported one does not: it is the target, also
        # when the binary windows of ``ground`` call it a tie
        grounding = {**grounding, "target": added, "flags": list(grounding.get("flags") or []) + ["relation_single"]}
    if mode == "done" and (tool.behind or {}).get("why") == "untold" and grounding.get("status") == "ok":
        # the look behind was taken because the one in view could not be told: a thing of that type found there, with
        # the landmark still not in view, is the one meant (the relation cannot choose, and ``ground`` would take the
        # stronger fit, which is the one the main camera sees)
        found = next((q["id"] for q in percepts if q.get("source") == "wrist"
                      and q.get("type") == parsed.get("target_type")), None)
        if found is not None and grounding.get("target") != found and single_untold(
                parsed, [q for q in percepts if q["id"] != found]):
            grounding = {**grounding, "target": found, "flags": list(grounding.get("flags") or []) + ["behind_untold"]}
    if grounding.get("status") == "fail" or grounding.get("target") is None:
        return {"agent": None, "parsed": parsed, "grounding": grounding}
    if params.get("keep_target") and isinstance(getattr(tool, "prior", None), dict):
        grounding = _keep_target(instruction, tool, lib, parsed, grounding, exclude)
    if grounding["target"] == grounding.get("place"):
        return {"agent": None, "parsed": parsed, "grounding": {**grounding, "status": "fail", "reason": "same"}}
    tool.target = grounding["target"]
    if params.get("keep_target") and isinstance(getattr(tool, "prior", None), dict) and hasattr(tool, "remember"):
        memo = tool.remember()
        if memo is not None and instruction not in tool.prior:
            tool.prior[instruction] = memo  # the first choice for this instruction is the one that is kept
    scene = lib.scene_from_percepts(tool.percepts)
    place_p = next(p for p in tool.percepts if p["id"] == grounding["place"])
    # "in the bottom drawer of the cabinet", "in the back compartment of the caddy": the part word names the site
    site = _in_site(lib, place_p, parsed.get("place_part")) if parsed["predicate"] == "in" else None
    if parsed.get("form") == "put" and parsed["predicate"] == "on":
        try:
            from .goal import on_site
            tdoc = lib.export_types.get(place_p["type"]) or {}
            if tdoc.get("kind") == "fixture":
                s = on_site(lib, place_p["type"], parsed.get("place_part"))
                if s and f"{place_p['id']}_{s}" in (scene.get("sites") or {}):
                    site = s
        except Exception:
            site = None
    spec = {"predicate": parsed["predicate"], "target": grounding["target"],
            "place": f"{place_p['id']}_{site}" if site else place_p["id"],
            "place_kind": "region" if site else "object", "place_instance": place_p["id"], "instruction": instruction}
    if share and site:  # several objects asked into this region: this one's turn
        spec["place_share"] = [int(share[0]), int(share[1])]
        if params.get("shared_skip_at_place") and _at_place(lib, scene, tool.percepts, spec):
            # 2026-11-16, off by default: of several objects asked to one place, one that lies there already is not
            # the one to move ("both moka pots on the stove": a step counts as placed only when its own check
            # passes, so after a first step that ended any other way the second step chose by the strongest fit,
            # which can be the pot the first step had set down). Another object of the type is taken instead;
            # with none left the step has nothing to do.
            tgt = grounding["target"]
            if any(q["type"] == parsed.get("target_type") and q["id"] != tgt for q in percepts):
                alt = _plan_one(instruction, tool, lib, cap, params, tuple(exclude) + (tgt,), share)
                if alt.get("agent") is not None or (alt.get("grounding") or {}).get("reason") == "already at its place":
                    return alt
            return {"agent": None, "parsed": parsed,
                    "grounding": {**grounding, "status": "fail", "reason": "already at its place"}}
    if parsed.get("place_beside"):  # "to the right of the plate": a point on the table beside the landmark
        bs = _beside_site(lib, scene, place_p, parsed["place_beside"])
        if bs is None:
            return {"agent": None, "parsed": parsed, "grounding": {**grounding, "status": "fail", "reason": "no table"}}
        scene["sites"]["beside_point"] = bs
        spec.update(place="beside_point", place_kind="region", place_instance=None)
    ap, pp, fp = make_params(params.get("m1"))
    return {"agent": OracleAgent(lib, scene, spec, tool, cap, ap, pp, fp), "parsed": parsed, "grounding": grounding}
