"""2026-11-16 night (stream APPEAR): ``params["fallback_stand"]``, off by default. A named target the look did not
report is stood where an unexplained region shows it (``look._stand_candidates``), not at the region's own middle."""

import numpy as np

from rowant_lab.libero_env.m2_kin import mat_to_quat
from rowant_lab.libero_env.tests.test_close_look import _cam, _lib
from rowant_lab.pro2.robot import look as LK

# type -> the boxes of its model in the body frame: (centre, half)
SHAPES = {"can": [((0.0, 0.0, 0.0), (0.03, 0.038, 0.03))],                                    # rests with its y axis up
          "bottle": [((0.0, 0.0, 0.05), (0.03, 0.02, 0.05)), ((0.0, 0.0, 0.12), (0.012, 0.012, 0.02))],  # body, cap
          "box": [((0.0, 0.0, 0.01), (0.04, 0.02, 0.01))],
          "basket": [((0.0, 0.0, 0.07), (0.10, 0.08, 0.07))],
          "cabinet": [((0.0, 0.0, 0.10), (0.10, 0.10, 0.10))]}
REST_UP = {"can": [0.0, 1.0, 0.0]}
TABLE = {"arena": [{"type": "box", "centre": [0.0, 0.0, -0.05], "R": np.eye(3).tolist(), "half": [0.5, 0.6, 0.05]}]}
CAM_AT = np.array([0.9, 0.0, 0.65])


def _camera():
    """A camera at ``CAM_AT`` looking at the table's middle: x right, y down, z forward (``behind.ray_at``'s)."""
    z = -CAM_AT / np.linalg.norm(CAM_AT)
    x = np.cross(z, np.array([0.0, 0.0, 1.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.stack([x, y, z], axis=1), CAM_AT
    f = 512.0 / np.tan(np.radians(22.5))
    return {"K": np.array([[f, 0.0, 512.0], [0.0, f, 512.0], [0.0, 0.0, 1.0]]), "T": T}


def _p(pid, typ, pos, **kw):
    return {"id": pid, "type": typ, "gid": "g", "joints": {}, "fit": {"n_agree": 1000, "inlier": 0.9},
            "source": "texture", "bodies": {"main": {"pos": list(pos), "quat_wxyz": mat_to_quat(np.eye(3))}}, **kw}


def _rej(pid, typ, pos, inlier=0.8, n_agree=800, tilt=0.0):
    return {**_p(pid, typ, pos), "fit": {"n_agree": n_agree, "inlier": inlier}, "pos": list(pos), "source": "region",
            "why": {"accepted": False, "upright_err_deg": tilt}}


def _blob(xy, half, height, top_xy, top_half, n_px=5000, base=0.0, uv_box=None, hang=0.0, yaw=0.0, top_yaw=0.0):
    b = {"xy": list(xy), "half_xy": list(half), "yaw": yaw, "z_top": base + height + 0.01, "height": height,
         "n_px": n_px, "hang_share": hang,
         "top": {"xy": list(top_xy), "half_xy": list(top_half), "yaw": top_yaw, "z": base + height, "n_px": 900}}
    if uv_box is not None:
        b["uv_box"] = list(uv_box)
    return b


def _tool(percepts=(), rejected=(), blobs=(), stand=True, cam=False):
    lib = _lib()
    base = lib.export_types["can"]
    lib.export_types = {t: dict(base) for t in SHAPES}
    lib.m1_type = lambda typ: {"kind": "fixture" if typ == "cabinet" else "object", "bodies": {"main": {"primitives": [
        {"centre": list(c), "R": np.eye(3).tolist(), "half": list(h)} for c, h in SHAPES[typ]]}}}
    lib.types = {t: {"rest_up_in_body": up} for t, up in REST_UP.items()}
    lib.workcell = TABLE
    t = LK.PerceptionTool(lib, None, _cam, "libero_object", "ep")
    t.percepts = list(percepts)
    t.extra = {"rejected": list(rejected), "blobs": list(blobs)}
    t.table_bound = t.blob_on_table = True
    if stand:
        t.stand = {**LK.STAND, **(stand if isinstance(stand, dict) else {})}
    if cam:
        t.cams = {"agentview": _camera()}
    return t


def _target(t):
    fb = [f for f in t.fallbacks if f.get("role") == "target" and f.get("id")]
    p = next(x for x in t.percepts if x["id"] == fb[-1]["id"])
    lo, hi = t._box_of(p)
    return fb[-1], 0.5 * (lo + hi), hi - lo


def _outline_of(typ, xy, yaw=0.0):
    """The pixel box of ``typ`` standing at ``xy`` as the test camera sees it."""
    t = _tool(cam=True)
    R, lo, hi, tlo, thi = t._stand_model(typ, yaw)
    pos = np.array([xy[0] - 0.5 * (tlo[0] + thi[0]), xy[1] - 0.5 * (tlo[1] + thi[1]), -lo[2]])
    cam = _camera()
    sg = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
    from rowant_lab.pro2.robot.ground import _boxes

    P = np.concatenate([c + (sg * h) @ np.asarray(Rp).T for c, Rp, h in
                        _boxes(t.lib, {"type": typ, "bodies": t._build(typ, R, pos)})])
    Pc = (P - cam["T"][:3, 3]) @ cam["T"][:3, :3]
    u = cam["K"][0, 0] * Pc[:, 0] / Pc[:, 2] + 512.0 - 0.5
    v = cam["K"][1, 1] * Pc[:, 1] / Pc[:, 2] + 512.0 - 0.5
    return [u.min(), v.min(), u.max(), v.max()]


# a can 7.6 cm tall standing at (-0.12, -0.24); its region's middle lies 5.5 cm behind it, along the camera's ray
CAN_AT = (-0.12, -0.24)
SKIRT = dict(xy=(-0.175, -0.25), half=(0.073, 0.079), height=0.079, top_xy=(-0.116, -0.243), top_half=(0.032, 0.029))


def test_off_by_default_the_model_goes_to_the_regions_middle_as_before():
    t = _tool(blobs=[_blob(**SKIRT)], stand=False)
    assert t.stand is None
    t.on_parsed({"target_type": "can"})
    fb, mid, _size = _target(t)
    assert fb["source"] == "blob" and "stand" not in fb and np.allclose(mid[:2], SKIRT["xy"], atol=1e-6)


def test_the_model_is_stood_as_it_rests_under_the_regions_top_layer():
    t = _tool(blobs=[_blob(**SKIRT)])
    t.on_parsed({"target_type": "can"})
    fb, mid, size = _target(t)
    assert fb["source"] == "blob_stand" and fb["stand"]["by"] == "layer" and fb["stand"]["region"] == 0
    assert np.allclose(mid[:2], SKIRT["top_xy"], atol=1e-6)  # no camera: no slide
    assert abs(size[2] - 0.076) < 1e-6 and abs(mid[2] - 0.038) < 1e-6  # its y axis up, its base on the table
    assert "stand" not in next(x for x in t.percepts if x["id"] == fb["id"])  # the record only


def test_the_outline_places_it_where_the_depth_was_read_too_far():
    box = _outline_of("can", CAN_AT)
    C = CAM_AT
    P = np.array([CAN_AT[0], CAN_AT[1], 0.076])
    far = C + (P - C) * 1.025  # the whole thing read 2.5% (3 cm) too far: lower and further off, along the ray
    b = _blob(xy=(-0.2, -0.26), half=(0.09, 0.05), height=float(far[2]), top_xy=far[:2], top_half=(0.04, 0.03),
              uv_box=box)
    t = _tool(blobs=[b], cam=True)
    t.on_parsed({"target_type": "can"})
    fb, mid, _size = _target(t)
    assert fb["stand"]["by"] == "outline" and fb["stand"]["iou"] > 0.75
    assert np.linalg.norm(mid[:2] - np.array(CAN_AT)) < 0.01 < np.linalg.norm(far[:2] - np.array(CAN_AT))
    t = _tool(blobs=[{k: v for k, v in b.items() if k != "uv_box"}], cam=True)  # no pixel box: the layer, slid
    t.on_parsed({"target_type": "can"})
    fb, mid, _size = _target(t)
    assert fb["stand"]["by"] == "layer" and fb["stand"]["slid_m"] > 0.015
    assert np.linalg.norm(mid[:2] - np.array(CAN_AT)) < 0.01


def test_a_region_of_another_height_or_size_is_not_stood_on():
    tall = _blob(xy=(0.1, 0.1), half=(0.05, 0.05), height=0.15, top_xy=(0.1, 0.1), top_half=(0.03, 0.03))
    t = _tool(blobs=[tall])
    assert t._stand_candidates("can", set()) == []
    t.on_parsed({"target_type": "can"})
    assert _target(t)[0]["source"] == "blob"  # the older fallback
    wide = _blob(xy=(0.1, 0.1), half=(0.1, 0.1), height=0.076, top_xy=(0.1, 0.1), top_half=(0.09, 0.09))
    assert _tool(blobs=[wide])._stand_candidates("can", set()) == []
    flat = _blob(xy=(0.1, 0.1), half=(0.05, 0.03), height=0.026, top_xy=(0.1, 0.1), top_half=(0.04, 0.02))
    c = _tool(blobs=[flat])._stand_candidates("box", set())  # 6 mm high on a 2 cm box: within a centimetre
    assert len(c) == 1 and c[0][2]["stand"]["height_ratio"] == 1.3


def test_a_bottle_is_matched_by_its_cap_and_a_long_top_turns_the_model():
    cap = _blob(xy=(-0.05, 0.0), half=(0.08, 0.03), height=0.14, top_xy=(0.02, 0.0), top_half=(0.013, 0.012))
    t = _tool(blobs=[cap])
    t.on_parsed({"target_type": "bottle"})
    fb, mid, size = _target(t)
    assert fb["source"] == "blob_stand" and np.allclose(mid[:2], (0.02, 0.0), atol=1e-6) and abs(size[2] - 0.14) < 1e-6
    long_top = _blob(xy=(0.0, 0.05), half=(0.06, 0.05), height=0.02, top_xy=(0.0, 0.1), top_half=(0.04, 0.02),
                     top_yaw=0.5 * np.pi)
    t = _tool(blobs=[long_top])
    t.on_parsed({"target_type": "box"})
    fb, mid, size = _target(t)
    assert abs(fb["stand"]["yaw"] - 0.5 * np.pi) < 1e-3 and np.allclose(size[:2], (0.04, 0.08), atol=1e-6)


def test_the_older_fallbacks_pose_stays_when_it_lies_in_the_region_and_goes_when_it_lies_elsewhere():
    b = _blob(**SKIRT)
    t = _tool(rejected=[_rej("p100", "can", (-0.15, -0.245, 0.04), tilt=85.0)], blobs=[b])  # in the region, lying
    t.on_parsed({"target_type": "can"})
    assert _target(t)[0]["source"] == "rejected:region"
    t = _tool(rejected=[_rej("p100", "can", (0.2, 0.3, 0.04))], blobs=[b])  # a pose 60 cm away
    t.on_parsed({"target_type": "can"})
    fb, mid, _size = _target(t)
    assert fb["source"] == "blob_stand" and np.allclose(mid[:2], SKIRT["top_xy"], atol=1e-6)
    t = _tool(rejected=[_rej("p100", "can", (0.2, 0.3, 0.04))], blobs=[b], stand=False)
    t.on_parsed({"target_type": "can"})
    assert _target(t)[0]["source"] == "rejected:region"  # without the option: as before


def test_a_named_region_comes_first_and_one_inside_another_objects_footprint_last():
    a = _blob(xy=(0.2, 0.2), half=(0.04, 0.04), height=0.076, top_xy=(0.2, 0.2), top_half=(0.031, 0.031))
    b = _blob(xy=(-0.2, 0.2), half=(0.06, 0.05), height=0.07, top_xy=(-0.2, 0.2), top_half=(0.036, 0.027))
    t = _tool(rejected=[_rej("p100", "can", (-0.21, 0.2, 0.04), inlier=0.3)], blobs=[a, b])  # too weak to be used,
    c = t._stand_candidates("can", set())                                                     # but it names b
    assert [x[1] for x in c] == [1, 0] and c[0][2]["stand"]["named"] and not c[1][2]["stand"]["named"]
    t = _tool(percepts=[_p("p0", "basket", (0.2, 0.2, 0.0))], blobs=[a, b])
    c = t._stand_candidates("can", set())
    assert [x[1] for x in c] == [1, 0] and c[1][2]["stand"]["inside"]


def test_a_hanging_region_is_not_stood_on_and_a_fixture_never():
    hang = _blob(xy=(0.1, 0.1), half=(0.04, 0.04), height=0.076, top_xy=(0.1, 0.1), top_half=(0.03, 0.03), hang=0.8)
    assert _tool(blobs=[hang])._stand_candidates("can", set()) == []
    fx = _blob(xy=(0.1, 0.1), half=(0.1, 0.1), height=0.2, top_xy=(0.1, 0.1), top_half=(0.1, 0.1))
    assert _tool(blobs=[fx])._stand_candidates("cabinet", set()) == []


def test_the_target_read_as_a_lower_thing_of_another_type_takes_that_things_place():
    # a bottle 14 cm tall was reported as a 2 cm box; its upper part is left as a region that hangs over the box
    left = _blob(xy=(0.0, 0.0), half=(0.02, 0.02), height=0.14, top_xy=(0.0, 0.0), top_half=(0.012, 0.012), hang=1.0)
    t = _tool(percepts=[_p("p0", "box", (0.0, 0.0, 0.0)), _p("p1", "basket", (0.0, 0.3, 0.0))], blobs=[left])
    t.on_parsed({"target_type": "bottle", "place_type": "basket"})
    fb, mid, _size = _target(t)
    assert fb["source"] == "blob_stand" and fb["stand"]["inside"]
    assert [p["id"] for p in t.percepts if p["type"] == "box"] == []  # it was the bottle
    assert {"role": "dropped", "ids": ["p0"], "why": "the target read as another type"} in t.fallbacks
    # the instruction's own receiver is never dropped: a box standing in the basket
    inb = _blob(xy=(0.0, 0.3), half=(0.05, 0.03), height=0.16, top_xy=(0.0, 0.3), top_half=(0.04, 0.02), hang=1.0)
    t = _tool(percepts=[_p("p1", "basket", (0.0, 0.3, 0.0))], blobs=[inb], stand={"height": (0.5, 9.0)})
    t.on_parsed({"target_type": "box", "place_type": "basket"})
    assert [p["id"] for p in t.percepts if p["type"] == "basket"] == ["p1"]


def test_after_a_close_on_nothing_the_next_place_is_another_region_then_the_older_fallbacks():
    a = _blob(xy=(0.2, 0.2), half=(0.04, 0.04), height=0.076, top_xy=(0.2, 0.2), top_half=(0.031, 0.031))
    b = _blob(xy=(-0.2, 0.2), half=(0.06, 0.05), height=0.07, top_xy=(-0.2, 0.2), top_half=(0.036, 0.027))
    t = _tool(blobs=[a, b])
    t.on_parsed({"target_type": "can"})
    fb, mid, _size = _target(t)
    assert fb["stand"]["region"] == 0
    t.target = fb["id"]
    out = t.refute_target()
    assert out["source"] == "blob_stand" and t.fallbacks[-1]["stand"]["region"] == 1 and t.fallbacks[-1]["refuted"] == 1
    assert np.allclose(_target(t)[1][:2], (-0.2, 0.2), atol=1e-6)
    out = t.refute_target()  # both stood places refuted: the older order (a region's own middle that is not refuted)
    assert out is None or out["source"] != "blob_stand"


def test_episode_sets_the_option_from_the_run_parameters():
    import inspect

    from rowant_lab.pro2.robot import episode as EP

    src = inspect.getsource(EP)
    assert 'params.get("fallback_stand")' in src and "tool.stand = {**STAND" in src


# ---- the agent's side: a stood model held across its other side (``c1_stand``)
def test_a_stood_target_counts_as_unmeasured_until_a_look_measures_it():
    cap = _blob(xy=(-0.05, 0.0), half=(0.08, 0.03), height=0.14, top_xy=(0.02, 0.0), top_half=(0.013, 0.012))
    t = _tool(blobs=[cap])
    t.on_parsed({"target_type": "bottle"})
    t.target = _target(t)[0]["id"]
    assert t.is_stood()
    t.percepts = [{**p, "source": "track"} if p["id"] == t.target else p for p in t.percepts]
    assert not t.is_stood()  # tracked: its pose is a measured one
    t2 = _tool(blobs=[_blob(**SKIRT)], stand=False)
    t2.on_parsed({"target_type": "can"})
    t2.target = _target(t2)[0]["id"]
    assert not t2.is_stood()  # the older fallback is no stood model


def test_the_grasp_check_takes_a_steady_hold_on_the_stood_models_other_side():
    from types import SimpleNamespace

    from rowant_lab.pro2.robot import agent as AG

    cap = _blob(xy=(-0.05, 0.0), half=(0.08, 0.03), height=0.14, top_xy=(0.02, 0.0), top_half=(0.013, 0.012))
    t = _tool(blobs=[cap])
    t.on_parsed({"target_type": "bottle"})
    t.target = _target(t)[0]["id"]
    ap = AG.AgentParams(c1_stand=True)
    ag = SimpleNamespace(ap=ap, oracle=t, lib=t.lib, target=t.target)
    side = AG.OracleAgent._stood_other_side(ag, 0.04)  # the record's width is the bottle's 4 cm side
    assert abs(side - 0.06) < 1e-6  # its other side across
    assert abs(AG.OracleAgent._stood_other_side(ag, 0.06) - 0.04) < 1e-6
    ag.ap = AG.AgentParams()  # off by default
    assert AG.OracleAgent._stood_other_side(ag, 0.04) is None
    ag.ap = ap
    t.percepts = [{**p, "source": "track"} if p["id"] == t.target else p for p in t.percepts]
    assert AG.OracleAgent._stood_other_side(ag, 0.04) is None  # measured since: the record's width stands


# ---- ``track``: the stood model takes its type's model id and may be tracked, within a limit
def _tracked(move, stand):
    from rowant_lab.libero_env.tests.test_close_look import Client

    cap = _blob(xy=(-0.05, 0.0), half=(0.08, 0.03), height=0.14, top_xy=(0.02, 0.0), top_half=(0.013, 0.012))
    weak = {**_rej("p100", "bottle", (0.4, 0.4, 0.0), inlier=0.2), "gid": "g7"}  # a pose too weak to use: its model id
    t = _tool(rejected=[weak], blobs=[cap], stand=stand)
    t.client = Client(move=move)
    t.obs = {"robot0_joint_pos": np.zeros(7), "robot0_gripper_qpos": np.array([0.04, -0.04]),
             "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0])}
    t.on_parsed({"target_type": "bottle"})
    fb, mid, _size = _target(t)
    t.target = fb["id"]
    t.track_look()
    return t, mid, _target(t)[1], next(p for p in t.percepts if p["id"] == t.target)


def test_a_stood_model_is_tracked_only_with_the_option_and_only_a_short_way():
    t, before, after, p = _tracked((0.008, -0.006, 0.0), {"track": True})
    assert p["gid"] == "g7" and p["source"] == "track" and t.looks[-1]["stood_off_m"] == 0.01
    assert np.allclose(after[:2] - before[:2], (0.008, -0.006), atol=1e-6) and not t.is_stood()
    t, before, after, p = _tracked((0.03, 0.0, 0.0), {"track": True})  # 3 cm: the stood place stands
    assert p["source"] == "blob_stand" and np.allclose(after, before) and t.looks[-1]["stood_off_m"] == 0.03
    assert t.is_stood() and len(t.client.calls) == 1
    t, before, after, p = _tracked((0.008, -0.006, 0.0), True)  # without ``track``: no model id, no request
    assert p["gid"] is None and t.client.calls == [] and t.looks[-1]["status"] == "untrackable"
