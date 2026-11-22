"""In-episode learning, server side (Monty venv; the second world's amendment W2-LEARN-EP). Off unless the
perception server's params carry ``learn_in_episode``: its folder is then ``<server cwd>/learn``, which the server's
audit hook lets it write and read.

M2 (perceived poses, grounded instruction): perception in the loop. TRUTH-FREE: perception-server side; it reads the
frames, proprioception and percepts the robot sends, the run's library and the Panda model, nothing else.

ST-1's learning chain (``world2_learn.chain_steps``), moved inside one episode. Two requests, both keyed by the
episode's opaque token:

* ``learn_plan``: what to learn and where to look at it from.
  1. **The leftover** (``lfb_learn.leftover``'s rule, on the episode's first agentview frame, which the robot sends
     back): world points above the support plane, minus those within ``EXPLAIN_M`` of the accepted percepts' model
     nodes. The largest connected component (at least ``MIN_VOX`` voxels) is the new object. Here the accepted percepts
     are the robot's own first look, made with its normal memory: the instruction's type had no accepted percept, so
     nothing explains the new object (ST-1 removed that type from memory instead; the effect is the same).
  2. **The looks**, at LF-b's 0.30 m standoff: the object's two flanks (``FLANK_AZ``, azimuth -90 and +90 deg at
     45 deg elevation, LF-b's tries: 15 deg azimuth steps, then 70 deg), and one look from the robot's side between
     them (azimuth 0 at ``MID_EL``: 70 deg, else 80, else straight above). The far side is the agentview's, which
     faces the robot. Azimuth 0 points from the object to the robot's base. The camera only tilts from its start view
     (no twist about its axis), because LIBERO's OSC controller tracks the whole orientation and jams the wrist at a
     joint limit when asked to spin it. Each pose is solved by damped least squares on the server's Panda model (the
     model the self-filter uses), with the wrist camera's mount measured from a wrist frame taken at the current pose.
     A pose is refused when IK does not converge (2 mm, 1 deg), a joint ends within ``LIMIT_MARGIN`` of its range, a
     point of the moving arm comes within ``CLEAR_M`` of an observed scene point, or within ``CLEAR_PLANE_M`` of the
     support plane. ST-1 placed the arm without motion and refused poses by the simulator's contacts; this check uses
     only what the robot saw.
  The reply gives each look as the hand pose the robot's controller tracks (the proprioception's ``eef`` frame), in
  visiting order (the flank nearer the camera first), with waypoints over the object between looks (``WAYPOINT_DEG``),
  each solved and checked the same way.
* ``learn``: the wrist frames the robot took at those looks ->
  1. masks (``lfb_learn.cmd_masks``' rule) on those frames and on the wrist frame from before any motion;
  2. a TSDF model (``TSDF_MAIN``: LF-b's tuned winner, 2 mm, truncation 3) over the masked wrist frames and the
     agentview frame with the leftover's own pixels as its mask;
  3. ST-1's F7 entry: the observed model, shrunk by 0.92 (at most twice) when it yields no grasp record;
  4. a variant library in the learn folder: the episode's library with the instruction's type replaced by the model;
  5. this episode's memory (a workcell on the variant) and F7's server settings: the type texture-first with 8 peaks,
     colour yaw sweep 15 deg (r 0.6). Both last until the next ``begin_episode``, which also deletes the variant tree.

Every learned model is the episode's own: nothing carries over to another episode.

**The top-down look first** (2026-11-17; the server option ``learn_top_first``, off by default; the robot asks
with ``stage``). The colour line's first look cannot place a thing it has no model of: its depth net smears it
along the main camera's line of sight (a median 6.6 cm off and three times too long on world 2's development
scenes). So ``learn_plan`` has two stages:

* ``stage: "top"``: the leftover as before, and ONE look from above (90 degrees, else 80, else 70 from the robot's
  side) over the part of the leftover nearest the main camera, where the thing itself stands;
* ``stage: "flanks"`` with that look's frame (``top``): the thing is found again in the frame's own depth (the
  largest unexplained thing above the support within ``refind_m`` of the first leftover's footprint; the support's
  height is read from the frame itself, ``plane: "wrist"``), and the ring of looks is planned around it (the two
  flanks and the look from the robot's side, ``mid``). The first leftover's points leave the clearance cloud and
  the wrist frame's points join it.

``learn`` then builds from the top look and the flank looks; the main camera's frame is left out of the model.

**Wrist frames without depth** (2026-11-17; the server option ``learn_wrist_stereo``, off by default; the robot's
side is ``rowant_lab/pro2/wrist_stereo.py``): a wrist frame may come with colour, its camera pose and ``partners``
(earlier wrist pictures with their poses) and an all-zero depth. ``wrist_depth`` then makes its depth before the
masks: K0-stereo's plane sweep (``m3_stereo1_match``, imported, not edited; its pick per width), the hand and
fingers left out by K0's raster, and every pixel the matcher does not read left as no reading (0). Nothing is
filled in. A frame with no partner keeps no depth and joins neither the masks nor the model.

**Check before replacing** (amendment W2-LEARN-EP2; the server option ``learn_check_own``, off by default): in
``learn``, after the masks and before any model is built, ``check_own`` tests the robot's own model of the type against
the looks (the type upright on the support, its yaw and position searched over all the looks at once; the server's own
acceptance floor over the pooled looks, and the model covering the object's observed points). When it explains them,
nothing is built and the reply is ``status: own`` with the type's root pose in the world; the robot then acts with its
own model at that pose.
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np

from rowant_lab.libero_env import lfb_paths as LP

FLANK_AZ = (-90.0, 90.0)  # the object's two flanks at LF-b's 45 deg; the far side is the agentview's (it faces the robot)
MID_EL = (70.0, 80.0, 90.0)  # and one look between the flanks from the robot's side, as low as reachable (90 = above)
LIMIT_MARGIN = 0.10  # rad: a planned pose keeps every joint this far inside its range (the OSC jams at a limit)
CLEAR_M = 0.015
CLEAR_PLANE_M = 0.010
CHECK_BODIES = ("robot0_link3", "robot0_link4", "robot0_link5", "robot0_link6", "robot0_link7", "robot0_right_hand",
                "gripper0_right_gripper", "gripper0_leftfinger", "gripper0_rightfinger")
IK_ITERS, IK_POS_TOL, IK_ANG_TOL_DEG, IK_DAMP, FD_EPS = 150, 0.002, 1.0, 0.05, 1e-5  # lfb_capture's
WAYPOINT_DEG = 30.0
TSDF_MAIN = {"vox_mm": 2.0, "trunc_k": 3.0}
F7_SHRINK, F7_TRIES = 0.92, 3
F7_SUPPORT = {"peaks": 8, "colour_sweep_deg": 15, "colour_sweep_min_r": 0.6}
HAND = "robot0_right_hand"
N_ARM = 7
# check before replacing (W2-LEARN-EP2, server option ``learn_check_own``): the type's own model, upright on the support,
# searched over the looks; it explains them when the server's own acceptance floor holds over all the looks pooled
# (``accept_inlier_region``, ``accept_min_agree``) and the model covers the object's observed points just as well
CHECK_OWN = {"yaw_step_deg": 10.0, "xy_m": 0.03, "xy_step_m": 0.01,  # coarse: yaw, in-plane offsets, on the support
             "fine_yaw_deg": 10.0, "fine_yaw_step_deg": 2.0, "fine_xy_m": 0.01, "fine_xy_step_m": 0.0025,
             "dz_m": (-0.005, -0.0025, 0.0, 0.0025, 0.005),  # a visual mesh need not touch its support exactly
             "nodes": 1500, "cover_m": 0.01, "cover_min": 0.85}


STEREO_CONFIG = {1024: "PS-w13-loose", 512: "PS-w7-strict"}  # K0-stereo's picks (K0S-PICK)
STEREO = {"threads": 4}
_HAND: dict = {}


def learn_dir() -> Path:
    return Path.cwd() / "learn"


# ------------------------------------------------------------------------------------------------ what to learn
RANK_KNOWN = {"cover": 0.6, "margin": None, "size_share": 0.25, "size_vox": 300}
# ``mid``: the look from the robot's side stays in the second plan (without it and without the main camera's frame
# two sides of the thing are seen only at a slant, and the model came out 2 mm wider where the gripper closes: one
# Mac scene, 2026-11-17, enough to fail the grasp check by 1 mm)
# ``agent``: what the main camera's frame gives the model after a top look. ``"none"``; or ``"carve"``: its pixels
# around the thing (the support, known objects: depth the net knows) still show which space is empty, the thing's own
# pixels, grown by ``carve_grow_px``, give nothing (their depth is the smear).
TOP_FIRST = {"near_share": 0.25, "aim_up_m": 0.05, "refind_m": 0.06, "plane": "wrist", "plane_band_m": 0.03, "mid": True,
             "agent": "none", "carve_grow_px": 2, "self_px": None}
# ``self_px`` (2026-11-18, off unless given): the robot's own pixels in the first look's frame, found in the PICTURE
# (its model at the measured joints drawn with the frame's camera, grown by this many pixels of that frame), give no
# points: none for the leftover and none for the clearance cloud. The server's own filter is in space (1 cm from the
# model), and the colour line's depth of the hand is off by more. Measured on the Mac's colour line (six of the 17
# new-object scenes where the pod of 11-18 planned no look from above): 311 to 331 points more than 20 cm above the
# support in five scenes and 20,334 in the sixth, all in the robot's own pixels; twelve of the thirteen refused
# tries came within 1 to 3 mm of one of them ("scene"), 0.24 to 0.30 m up. With 10 pixels (the 4 mm samples are
# about 5 pixels apart at 1024) none is left and the look straight down was planned in all six. In the sixth the
# "thing" itself was the whole arm, seen when the learner started at a plan made again in the middle of an episode.
TOP_EL = (90.0, 80.0, 70.0)  # the first look: straight above, else as steep as reachable from the robot's side
# Ask the known types (2026-11-18, server option ``learn_ask_known``, off unless given; with ``learn_top_first``). The
# thing the learner goes for is often one the robot has a model of and the first look did not accept: the basket (7 of
# 57 scenes of round B) or another object (3). After the look from above the thing's depth is good, so before the
# ring of looks is spent on it the types the robot names (``msg["ask"]["types"]``: the place's type when it has no
# percept, the types of accepted percepts near the thing, the types of the first look's rejected poses) are fitted to
# it there: upright on the support, yaw and
# position searched as ``check_own`` does, the position out to the type's own half width (a large thing is seen in
# part). A type explains the thing when its model does not fill space the look saw through (``inlier_min``), enough
# of what the look could see of it agrees (``seen_min``), and the thing's own points lie on it (``cover_min``). The
# reply is then ``status: known`` with the type's pose; the robot keeps it as a percept and asks for the next thing
# (``msg["next"]``: the things tried before are left out). ``act`` false: the fits are made and reported, nothing
# else changes (the measurement the acceptance numbers were fixed on).
ASK_KNOWN = {"act": True, "yaw_step_deg": 10.0, "xy_step_m": 0.01, "xy_min_m": 0.03, "xy_max_m": 0.12,
             "fine_yaw_deg": 10.0, "fine_yaw_step_deg": 2.0, "fine_xy_m": 0.01, "fine_xy_step_m": 0.0025,
             "dz_m": (-0.005, -0.0025, 0.0, 0.0025, 0.005), "nodes": 1500, "cover_m": 0.01,
             "inlier_min": None, "seen_min": None, "cover_min": None}  # None: no type explains anything
# The floor for jumping the queue (2026-11-18, server option ``learn_rank_floor``, off unless given; with
# ``learn_rank_known``): an uncovered thing under ``min_vox`` voxels does not go before a covered larger one. Round
# B's pod rows: four wrong things of 75 to 342 voxels went first that way; the right things had 980 or more, but for
# one of 91 in a scene that failed anyway.
RANK_FLOOR = {"min_vox": 500}


def rank_components(sizes: list, covers: list, cover_min: float, min_vox: int, margin: float | None = None,
                    size_share: float = 0.25, size_vox: int = 300, floor: int | None = None) -> list:
    """The order in which unexplained things are considered for learning (``learn_rank_known``): first those large
    enough to learn that no rejected pose of a known type covers (``covers[i]`` under ``cover_min``), largest first;
    then the rest in their old order. With nothing covered the order is the old one (largest first).

    ``margin`` (off unless given; chosen on three development scenes, 2026-11-17): when every thing is covered, a
    new object that resembles a known type can be covered too (0.63 and 0.69 seen) while the missed known object is
    covered more (0.85). Then, among the things at least ``size_share`` of the largest (or ``size_vox`` voxels, when
    the largest is something as big as the basket: one colour scene, the basket unaccepted and covered 0.97), the
    least covered goes first if the largest is covered by at least ``margin`` more.

    ``floor`` (``learn_rank_floor``, off unless given): only a thing of at least that many voxels goes before the
    others, by either rule; a smaller thing keeps its place by size."""
    free = [i for i in range(len(sizes)) if sizes[i] >= max(min_vox, int(floor or 0)) and covers[i] < cover_min]
    if not free and margin is not None and sizes:
        big = [i for i in range(len(sizes))
               if sizes[i] >= max(min_vox, min(size_share * sizes[0], size_vox), int(floor or 0))]
        j = min(big, key=lambda i: (covers[i], i)) if big else 0
        if j != 0 and covers[0] - covers[j] >= margin:
            free = [j]
    return free + [i for i in range(len(sizes)) if i not in free]


# The place the robot would have guessed (2026-11-19, server option ``learn_hint``, off unless given, and only when
# the robot sends ``hint``): the unexplained thing nearest that place goes first. As built, the largest unexplained
# thing was taken, which is the right one where everything else has a model (the second world, the Object suite). In
# the table scenes it was not: a pan with no model (larger than the object asked for) and a strip of table edge the
# arm cannot reach were taken on the Mac.
HINT = {"within_m": 0.06, "top": 8}


def hint_order(V, comps: list, xy, within_m: float, top: int, min_vox: int) -> tuple:
    """(the order of the first ``top`` things, each one's level distance from ``xy``): the thing of at least
    ``min_vox`` voxels whose nearest voxel lies within ``within_m`` of the place goes first (the nearest such), the
    others keep their order. With none that near, the order is unchanged."""
    xy = np.asarray(xy, float)[:2]
    n = min(int(top), len(comps))
    d = [float(np.min(np.linalg.norm(np.asarray(V)[comps[i]][:, :2] - xy, axis=1))) for i in range(n)]
    near = [i for i in range(n) if d[i] <= float(within_m) and len(comps[i]) >= int(min_vox)]
    if not near:
        return list(range(n)), d
    first = min(near, key=lambda i: (d[i], i))
    return [first] + [i for i in range(n) if i != first], d


# ------------------------------------------------------------------------------------------------ wrist stereo
def preload_stereo() -> None:
    """K0's matcher and its hand model, loaded once (the server calls this before its audit hook is installed)."""
    from rowant_lab.libero_env import m3_stereo1_match as SM

    if "model" not in _HAND:
        _HAND["model"] = SM.load_hand_model(SM.REPO / SM.HAND_MODEL_REL)


def wrist_depth(fr: dict, o: dict | None = None) -> dict:
    """Depth for a wrist frame that came without it (``fr["partners"]``): K0's plane sweep on the frame and its
    partners, written into ``fr["depth_u16"]`` (0 = no reading; never filled in). Returns the look's numbers. A frame
    with no ``partners`` key is left as it is."""
    if "partners" not in fr:
        return {"status": "has_depth"}
    import cv2

    from rowant_lab.libero_env import m3_stereo1_match as SM

    o = {**STEREO, **(o or {})}
    W = int(fr["width"])
    parts = fr["partners"]
    row = {"n_partners": len(parts), "b_eff_mm": [round(float(p.get("b_eff_m", 0.0)) * 1e3, 1) for p in parts]}
    fr["depth_u16"] = np.zeros((W, W), np.uint16)
    if not parts:
        row["status"] = "no_partner"
        return row
    preload_stereo()
    cv2.setNumThreads(int(o["threads"]))

    def v(d, tok):
        return {"view_id": tok, "token": tok, "rgb": np.ascontiguousarray(d["rgb"], dtype=np.uint8),
                "gripper_qpos": np.asarray(d["gripper_qpos"], float).reshape(-1),
                "T": np.asarray(d["T_world_from_camera"], float), "K": np.asarray(d["K"], float)}

    ld = {"ref": v(fr, "ref"), "partners": [v(p, f"p{i}") for i, p in enumerate(parts)],
          "X": np.asarray(fr["X"], float), "select": "all", "look": "learner", "width": W}
    try:
        depth, info = SM.run_look(ld, SM.select_views(ld), SM.parse_config("PS", str(o.get("config") or STEREO_CONFIG[W]), W),
                                  _HAND["model"])
    except SM.LookFailure as exc:
        row.update(status=str(exc.status), reason=str(exc.reason)[:120])
        return row
    fr["depth_u16"] = SM.encode_map(depth)
    row.update(status="ok", read_share=round(float(np.isfinite(depth).mean()), 4), s=round(float(info["wall_s"]), 2),
               hand_px=int(info.get("hand_px", 0)))
    return row


# ------------------------------------------------------------------------------------------------ kinematics
def _quat_xyzw_to_mat(q) -> np.ndarray:
    from rowant_lab.libero_env import m2_kin as KIN

    x, y, z, w = (float(v) for v in q)
    return KIN.quat_to_mat([w, x, y, z])


def _T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.asarray(R, float), np.asarray(t, float)
    return T


def _inv(T) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return _T(R.T, -R.T @ t)


def _rotvec(R) -> np.ndarray:
    """Axis times angle of a rotation matrix (robust near 0 and pi)."""
    R = np.asarray(R, float)
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    if math.pi - ang < 1e-6:
        w, V = np.linalg.eigh((R + R.T) / 2.0)
        return V[:, int(np.argmax(w))] * ang
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(ang))
    return v * ang


class Arm:
    """The server's Panda model (``wc.panda``) with the wrist camera's mount and the eef frame measured once."""

    def __init__(self, wc, joint_pos, gripper_qpos, eef_pos, eef_quat_xyzw, T_world_cam):
        self.panda, self.gq = wc.panda, np.asarray(gripper_qpos, float)
        self.q0 = np.asarray(joint_pos, float)
        rng = []
        for n in self.panda.order:
            for jt in self.panda.bodies[n]["joints"]:
                if jt.get("proprio") and jt["proprio"][0] == "robot0_joint_pos":
                    rng.append((jt["proprio"][1], jt["range"]))
        self.range = np.array([r for _i, r in sorted(rng)], float)
        Th = self.hand(self.q0)
        self.T_hand_cam = _inv(Th) @ np.asarray(T_world_cam, float)
        self.T_hand_eef = _inv(Th) @ _T(_quat_xyzw_to_mat(eef_quat_xyzw), eef_pos)
        self.R_cam0 = np.asarray(T_world_cam, float)[:3, :3].copy()  # the look orientations tilt this, never twist
        pts = wc.robot_pts
        self.pts = {b: pts[b] for b in CHECK_BODIES if b in pts}

    def poses(self, q) -> dict:
        return self.panda.fk(q, self.gq)

    def hand(self, q) -> np.ndarray:
        R, t = self.poses(q)[HAND]
        return _T(R, t)

    def camera(self, q) -> tuple[np.ndarray, np.ndarray]:
        """(centre, optical axis); the frames' camera is OpenCV's (z forward)."""
        T = self.hand(q) @ self.T_hand_cam
        return T[:3, 3].copy(), T[:3, 2].copy()

    def eef(self, q) -> tuple[list, list]:
        from rowant_lab.libero_env import m2_kin as KIN

        T = self.hand(q) @ self.T_hand_eef
        return [float(x) for x in T[:3, 3]], [float(x) for x in KIN.mat_to_quat(T[:3, :3])]

    def robot_points(self, q) -> np.ndarray:
        P = self.poses(q)
        return np.concatenate([self.pts[b] @ P[b][0].T + P[b][1] for b in self.pts])

    def cam_rot(self, q) -> np.ndarray:
        return (self.hand(q) @ self.T_hand_cam)[:3, :3]

    def goal_rot(self, d_goal) -> np.ndarray:
        """The camera rotation looking along ``d_goal``: the start view tilted by the smallest rotation (no twist).
        Roll is fixed so the controller, which tracks the whole orientation, is never asked to spin the wrist."""
        a = self.R_cam0[:, 2]
        d = np.asarray(d_goal, float) / np.linalg.norm(d_goal)
        v, c = np.cross(a, d), float(a @ d)
        if np.linalg.norm(v) < 1e-9:
            return self.R_cam0.copy()
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return (np.eye(3) + vx + vx @ vx / (1.0 + c)) @ self.R_cam0

    def _err(self, q, c_goal, R_goal) -> np.ndarray:
        T = self.hand(q) @ self.T_hand_cam
        return np.concatenate([c_goal - T[:3, 3], _rotvec(R_goal @ T[:3, :3].T)])

    def ik(self, c_goal, d_goal, q_seed) -> dict:
        """Damped least squares on the camera's centre and full rotation (``goal_rot``: roll fixed); a solution must
        keep every joint ``LIMIT_MARGIN`` inside its range."""
        R_goal = self.goal_rot(d_goal)
        lo, hi = self.range[:, 0], self.range[:, 1]
        q = np.asarray(q_seed, float).copy()
        pos_e = ang_e = float("nan")
        for it in range(IK_ITERS):
            e = self._err(q, c_goal, R_goal)
            pos_e, ang_e = float(np.linalg.norm(e[:3])), math.degrees(float(np.linalg.norm(e[3:])))
            if pos_e <= IK_POS_TOL and ang_e <= IK_ANG_TOL_DEG:
                inside = bool(np.all(q >= lo + LIMIT_MARGIN) and np.all(q <= hi - LIMIT_MARGIN))
                return {"ok": inside, "q": q, "iters": it, "pos_err_m": pos_e, "ang_err_deg": ang_e,
                        "limit": not inside}
            J = np.zeros((6, N_ARM))
            for j in range(N_ARM):
                dq = q.copy()
                dq[j] += FD_EPS
                J[:, j] = (e - self._err(dq, c_goal, R_goal)) / FD_EPS
            step = J.T @ np.linalg.solve(J @ J.T + IK_DAMP ** 2 * np.eye(6), e)
            q = np.clip(q + step, lo, hi)
        return {"ok": False, "q": q, "iters": IK_ITERS, "pos_err_m": pos_e, "ang_err_deg": ang_e, "limit": False}


def unit(base_xy, centre, az_deg: float, el_deg: float) -> np.ndarray:
    """The unit direction from the object's centre to a camera at (azimuth, elevation); azimuth 0 points to the base."""
    v = np.asarray(base_xy, float) - np.asarray(centre, float)[:2]
    a, e = math.atan2(v[1], v[0]) + math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def goal(centre, u, r: float = LP.STANDOFF_M):
    """Camera centre at ``r`` along ``u`` and the unit look direction back at the centre."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    return np.asarray(centre, float) + r * u, -u


def slerp(u0, u1, f: float) -> np.ndarray:
    u0, u1 = np.asarray(u0, float), np.asarray(u1, float)
    w = math.acos(max(-1.0, min(1.0, float(u0 @ u1))))
    if w < 1e-6:
        return u0.copy()
    return (math.sin((1 - f) * w) * u0 + math.sin(f * w) * u1) / math.sin(w)


class Clearance:
    """A planned pose's moving arm against what the first look saw (robot-free points above the plane)."""

    def __init__(self, scene_pts: np.ndarray, plane: dict):
        from scipy.spatial import cKDTree

        self.tree = cKDTree(scene_pts) if len(scene_pts) else None
        n = np.asarray(plane["normal_w"], float)
        self.n = n / np.linalg.norm(n) * (1.0 if n[2] >= 0 else -1.0)
        self.h = float(plane["height_w"])

    def check(self, pts: np.ndarray) -> dict:
        above = float(((pts - np.array([0.0, 0.0, self.h])) @ self.n).min())
        near, at, n_near = np.inf, None, 0
        if self.tree is not None:
            d, i = self.tree.query(pts, k=1, distance_upper_bound=CLEAR_M)
            near = float(d.min())
            if np.isfinite(near):  # where the scene point is that the arm would come too close to (2026-11-18)
                at = [round(float(x), 3) for x in self.tree.data[int(i[int(np.argmin(d))])]]
                n_near = int(np.isfinite(d).sum())
        out = {"ok": above >= CLEAR_PLANE_M and not np.isfinite(near), "above_plane_m": round(above, 4),
               "near_scene_m": None if not np.isfinite(near) else round(near, 4)}
        if at is not None:
            out.update(near_at=at, n_near=n_near)
        return out


def _solve(arm, clear, c_goal, d_goal, seeds) -> dict:
    r = {"ok": False, "iters": 0, "pos_err_m": float("nan"), "ang_err_deg": float("nan")}
    for q0 in seeds:
        r = arm.ik(c_goal, d_goal, q0)
        if r["ok"]:
            break
    out = {"ik_ok": bool(r["ok"]), "iters": int(r["iters"]), "pos_err_mm": round(float(r["pos_err_m"]) * 1e3, 3),
           "ang_err_deg": round(float(r["ang_err_deg"]), 3), "at_limit": bool(r.get("limit"))}
    if r["ok"]:
        cl = clear.check(arm.robot_points(r["q"]))
        out.update(clear=cl, ok=bool(cl["ok"]), q=np.asarray(r["q"], float))
        # why (2026-11-18): "ok"; "scene": the arm within CLEAR_M of a point the first look saw; "plane": too low
        out["why"] = "ok" if cl["ok"] else "scene" if cl.get("near_scene_m") is not None else "plane"
    else:
        out["ok"] = False
        out["why"] = "limit" if r.get("limit") else "reach"  # a joint at its limit's margin; or no pose found
    return out


def self_depth(wc, fr: dict, joint_pos, gripper_qpos, grow_px: int) -> tuple[np.ndarray, np.ndarray]:
    """How far from the camera the robot's own model stands in each of the frame's pixels: metres along the camera's
    axis, the nearest of the samples drawn there, inf where none is drawn. Returned grown by ``grow_px`` exactly as
    ``self_pixels`` grows its mask (a grown pixel takes the nearest of the model's depths around it), and not grown."""
    from scipy.ndimage import generate_binary_structure, minimum_filter

    K, T = np.asarray(fr["K"], float), np.asarray(fr["T_world_from_camera"], float)
    H, W = np.asarray(fr["depth_u16"]).shape
    poses = wc.panda.fk(joint_pos, gripper_qpos)
    pts = np.concatenate([wc.robot_pts[b] @ poses[b][0].T + poses[b][1] for b in wc.robot_pts])
    Xc = (pts - T[:3, 3]) @ T[:3, :3]
    Xc = Xc[Xc[:, 2] > 0.05]
    u = np.floor(K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2]).astype(np.int64)
    v = np.floor(K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]).astype(np.int64)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    z0 = np.full((H, W), np.inf)
    np.minimum.at(z0, (v[ok], u[ok]), Xc[ok, 2])
    zg, cross = z0, generate_binary_structure(2, 1)
    for _ in range(max(0, int(grow_px))):
        zg = minimum_filter(zg, footprint=cross, mode="constant", cval=np.inf)
    return zg, z0


def self_pixels(wc, fr: dict, joint_pos, gripper_qpos, grow_px: int, front_m=None) -> np.ndarray:
    """The frame's pixels the robot's own model covers at the measured joints: its surface samples drawn with the
    frame's camera, each grown by ``grow_px`` (the samples lie a few pixels apart, and the camera is the assumed
    one).

    ``front_m`` (2026-11-18; None, the default, is the mask as it always was): a covered pixel is the robot's own
    only where the frame's depth is not NEARER than the model drawn there by more than ``front_m``. The model is
    drawn whole, the parts a table or a thing hides included, so without this the mask takes whatever stands in
    front of the robot out of the look (a paired pod run: about 179,000 pixels of the table and what is on
    it in the two table suites, against about 8,000 in the others; 411 to 275 of 476 episodes)."""
    from scipy.ndimage import binary_dilation

    if front_m is not None:
        zg, _ = self_depth(wc, fr, joint_pos, gripper_qpos, grow_px)
        depth = np.asarray(fr["depth_u16"])
        z = depth.astype(np.float64) * float(fr["depth_scale_mm"]) * 1e-3
        return np.isfinite(zg) & ~((depth > 0) & (z < zg - float(front_m)))
    K, T = np.asarray(fr["K"], float), np.asarray(fr["T_world_from_camera"], float)
    H, W = np.asarray(fr["depth_u16"]).shape
    poses = wc.panda.fk(joint_pos, gripper_qpos)
    pts = np.concatenate([wc.robot_pts[b] @ poses[b][0].T + poses[b][1] for b in wc.robot_pts])
    Xc = (pts - T[:3, 3]) @ T[:3, :3]
    Xc = Xc[Xc[:, 2] > 0.05]
    u = np.floor(K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2]).astype(np.int64)
    v = np.floor(K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]).astype(np.int64)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    m = np.zeros((H, W), bool)
    m[v[ok], u[ok]] = True
    return binary_dilation(m, iterations=int(grow_px)) if int(grow_px) > 0 else m


def near_point(C: np.ndarray, cam_xy, share: float) -> np.ndarray:
    """The middle (x, y) of the ``share`` of the points ``C`` nearest the main camera, horizontally: where a thing
    smeared along that camera's line of sight really stands (its smear runs away from the camera)."""
    d = np.linalg.norm(C[:, :2] - np.asarray(cam_xy, float)[:2], axis=1)
    k = max(1, int(round(len(C) * float(share))))
    return C[np.argsort(d, kind="stable")[:k], :2].mean(0)


def plan_top(arm, clear, centre, base_xy) -> dict:
    """One look from above ``centre`` (``TOP_EL``), solved and checked like every look."""
    row = {"look": 0, "kind": "top", "az_nominal_deg": 0.0, "tries": []}
    for el in TOP_EL:
        u = unit(base_xy, centre, 0.0, el)
        c_goal, d_goal = goal(centre, u)
        s_ = _solve(arm, clear, c_goal, d_goal, [arm.q0])
        row["tries"].append({"el_deg": el, "az_deg": 0.0, **{k_: v for k_, v in s_.items() if k_ != "q"}})
        if s_["ok"]:
            row.update(az_used_deg=0.0, el_used_deg=el, u=u, q=s_["q"], waypoints=[])
            break
    return row


def wrist_plane(Pw: np.ndarray, ok: np.ndarray, plane: dict, band: float) -> dict:
    """The support's height as the wrist frame itself shows it: the most common height of its points within
    ``band`` of the first look's plane (1 mm bins, then the mean within 2 mm). The first look's plane is kept when
    it is not level or the frame shows too little of the support."""
    n = np.asarray(plane["normal_w"], float)
    n = n / np.linalg.norm(n)
    z = Pw[..., 2][ok]
    z = z[np.abs(z - float(plane["height_w"])) <= band]
    if abs(n[2]) < 0.996 or len(z) < 500:
        return plane
    k = np.floor(z / 0.001).astype(np.int64)
    vals, cnt = np.unique(k, return_counts=True)
    c = (float(vals[int(np.argmax(cnt))]) + 0.5) * 0.001
    return {**plane, "normal_w": [0.0, 0.0, 1.0], "height_w": float(z[np.abs(z - c) <= 0.002].mean()),
            "first_look_height_w": float(plane["height_w"])}


def plan_looks(arm, clear, centre, base_xy, mid: bool = True) -> list:
    """The two flank looks (``FLANK_AZ``, with LF-b's tries: 15 deg azimuth steps, then 70 deg elevation) and the middle
    look between them (``MID_EL``). The flank nearer the current camera goes first; consecutive looks are joined over the top by
    waypoints every ``WAYPOINT_DEG`` on their great circle. IK is seeded from the latest planned pose first."""
    c_now, _ax = arm.camera(arm.q0)
    rows, seeds = [], [arm.q0]
    flanks = []
    for i, nominal in enumerate(FLANK_AZ):
        row = {"look": i, "kind": "flank", "az_nominal_deg": nominal, "tries": []}
        tries = [(LP.ELEVATION_DEG, k) for k in LP.AZ_TRIES] + [(LP.ELEVATION_FALLBACK_DEG, k) for k in LP.AZ_TRIES]
        for el, k in tries:
            az = nominal + k * LP.AZ_STEP_DEG
            u = unit(base_xy, centre, az, el)
            c_goal, d_goal = goal(centre, u)
            s_ = _solve(arm, clear, c_goal, d_goal, [arm.q0])
            row["tries"].append({"el_deg": el, "az_deg": az, **{k_: v for k_, v in s_.items() if k_ != "q"}})
            if s_["ok"]:
                row.update(az_used_deg=az, el_used_deg=el, u=u, q=s_["q"], dist=float(np.linalg.norm(c_goal - c_now)))
                break
        flanks.append(row)
    ok_flanks = sorted([r for r in flanks if "q" in r], key=lambda r: r["dist"])
    top = {"look": 2, "kind": "mid", "az_nominal_deg": 0.0, "tries": []}
    for el in (MID_EL if mid else ()):  # the object's robot side, which neither flank faces; straight above when lower is out of reach
        u_mid = unit(base_xy, centre, 0.0, el)
        c_goal, d_goal = goal(centre, u_mid)
        s_ = _solve(arm, clear, c_goal, d_goal, [r["q"] for r in ok_flanks] + [arm.q0])
        top["tries"].append({"el_deg": el, "az_deg": 0.0, **{k_: v for k_, v in s_.items() if k_ != "q"}})
        if s_["ok"]:
            top.update(az_used_deg=0.0, el_used_deg=el, u=u_mid, q=s_["q"])
            break
    order = ok_flanks[:1] + ([top] if "q" in top else []) + ok_flanks[1:]
    prev = None
    for r in order:
        r["waypoints"] = []
        if prev is not None:
            w = math.degrees(math.acos(max(-1.0, min(1.0, float(prev["u"] @ r["u"])))))
            n = max(0, int(math.ceil(w / WAYPOINT_DEG)) - 1)
            q_seed = prev["q"]
            for j in range(1, n + 1):
                u = slerp(prev["u"], r["u"], j / (n + 1))
                c_goal, d_goal = goal(centre, u)
                s_ = _solve(arm, clear, c_goal, d_goal, [q_seed, r["q"]])
                if s_["ok"]:
                    r["waypoints"].append({"el_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, float(u[2]))))), 2),
                                           "q": s_["q"]})
                    q_seed = s_["q"]
        prev = r
    rest = [x for x in flanks + ([top] if mid else []) if not any(x is y for y in order)]
    for r in rest:
        r["waypoints"] = []
    return order + rest


def _pose_doc(arm, q) -> dict:
    p, quat = arm.eef(q)
    c, ax = arm.camera(q)
    return {"eef_pos": p, "eef_quat_wxyz": quat, "cam_centre": [float(x) for x in c],
            "cam_axis": [float(x) for x in ax]}


# ------------------------------------------------------------------------------------------------ the requests
def refind(per, msg: dict) -> dict:
    """``learn_plan``, stage ``flanks`` (``learn_top_first``): the thing found again in the top look's own depth,
    and the two flank looks around it."""
    from scipy.spatial import cKDTree

    from rowant_lab.libero_env import lfb_learn as LL

    t0 = time.perf_counter()
    suite, ep = msg["suite"], msg["episode"]
    st = per.learn_state.get(ep)
    if st is None or st["suite"] != suite or st.get("stage") != "top":
        return {"ok": True, "status": "no_plan"}
    o = {**TOP_FIRST, **(per.p.get("learn_top_first") or {})}
    wc = per.cell(suite)
    w = msg["top"]
    fr, jp, gq = w["cameras"][0], w["joint_pos"], w["gripper_qpos"]
    out = {"ok": True, "status": None, "stage": "flanks"}
    if per.p.get("learn_wrist_stereo") is not None:
        out["stereo"] = wrist_depth(fr, per.p["learn_wrist_stereo"])
    lo0 = st["leftover"]
    Pw, ok = LL.world_points(per, wc, fr, jp, gq)
    plane = wrist_plane(Pw, ok, lo0["plane"], float(o["plane_band_m"])) if o["plane"] == "wrist" else lo0["plane"]
    keep = LL.above_plane(Pw, ok, plane)
    P = Pw[keep]
    if len(P):
        near, _ = cKDTree(st["voxels"][:, :2]).query(P[:, :2], k=1, distance_upper_bound=float(o["refind_m"]))
        P = P[np.isfinite(near)]
        P = P[~LL.explained(wc, st["percepts"], P)]
    V, comps = LL.components(P)
    lo = {"plane": plane, "n_vox": int(len(V)), "comp_sizes": [int(len(c)) for c in comps[:5]],
          "first": {k: lo0.get(k) for k in ("centre_w", "bbox_min", "bbox_max", "n_vox_object")}}
    if not comps or len(comps[0]) < LL.MIN_VOX:
        out.update(status="no_refind", leftover=lo, s=round(time.perf_counter() - t0, 3))
        return out
    C = V[comps[0]]
    lo.update(centre_w=[float(x) for x in C.mean(0)], bbox_min=[float(x) for x in C.min(0)],
              bbox_max=[float(x) for x in C.max(0)], n_vox_object=int(len(C)),
              runner_up_vox=int(len(comps[1])) if len(comps) > 1 else 0,
              shift_m=round(float(np.linalg.norm(C.mean(0)[:2] - np.asarray(lo0["centre_w"], float)[:2])), 4))
    ak = per.p.get("learn_ask_known")
    if ak is not None and (msg.get("ask") or {}).get("types"):
        ak = {**ASK_KNOWN, **ak}
        z = np.asarray(fr["depth_u16"]).astype(np.float64) * float(fr["depth_scale_mm"]) * 1e-3
        obs = [(np.where(z > 0, z, np.nan), np.asarray(fr["K"], float), np.asarray(fr["T_world_from_camera"], float))]
        fits = [fit_known(per, wc, str(t), C, obs, plane, ak) for t in msg["ask"]["types"]]
        out["ask"] = [{k: v for k, v in r.items() if k != "R_w"} for r in fits]  # (``t_w``: the robot compares it
        # with where the first look had placed an object of that type)
        best = max((r for r in fits if r.get("explains")), key=lambda r: r["n_agree"], default=None)
        if best is not None and ak["act"]:
            out.update(status="known", leftover=lo, s=round(time.perf_counter() - t0, 3),
                       known={k: best[k] for k in ("type", "gid", "R_w", "t_w", "inlier", "seen", "cover", "n_agree")})
            return out
    a = time.perf_counter()
    arm = Arm(wc, jp, gq, w["eef_pos"], w["eef_quat_xyzw"], fr["T_world_from_camera"])
    # the clearance cloud: what the first look saw, less its (smeared) leftover, plus what this frame shows there
    S = st["scene"]
    if len(S):
        d, _ = cKDTree(st["voxels"]).query(S, k=1, distance_upper_bound=LL.R_PAD_M)
        S = S[~np.isfinite(d)]
    looks = plan_looks(arm, Clearance(np.concatenate([S, P]) if len(S) else P, plane), lo["centre_w"], wc.base[:2],
                       mid=bool(o["mid"]))
    rows = []
    for r in looks:
        row = {k: v for k, v in r.items() if k not in ("q", "waypoints", "u", "dist")}
        if "q" in r:
            row.update(_pose_doc(arm, r["q"]))
            row["waypoints"] = [{"el_deg": wp["el_deg"], **_pose_doc(arm, wp["q"])} for wp in r.get("waypoints", [])]
        rows.append(row)
    st.update(stage="flanks", leftover=lo, voxels=C, top=(fr, jp, gq), no_agent=True)
    if o["agent"] == "carve":
        from scipy.ndimage import binary_dilation

        al = st["agent_look"]
        grow = int(o["carve_grow_px"])
        st["agent_carve"] = {**al, "carve_only": True,
                             "mask": binary_dilation(al["mask"], iterations=grow) if grow > 0 else al["mask"]}
    out.update(status="ok", leftover=lo, looks=rows, n_planned=sum("eef_pos" in r for r in rows),
               times={"refind_s": round(a - t0, 3), "plan_s": round(time.perf_counter() - a, 3)})
    return out


def learn_plan(per, msg: dict) -> dict:
    from rowant_lab.libero_env import lfb_learn as LL

    top_first = per.p.get("learn_top_first") is not None and msg.get("stage") is not None
    if top_first and msg["stage"] == "flanks":
        return refind(per, msg)
    t0 = time.perf_counter()
    suite, ep = msg["suite"], msg["episode"]
    wc = per.cell(suite)
    fr, jp, gq = msg["cameras"][0], msg["joint_pos"], msg["gripper_qpos"]
    percepts = [{"type": p["type"], "gid": p.get("gid"), "bodies": p["bodies"]} for p in msg["percepts"]]
    plane = msg.get("plane")
    out = {"ok": True, "status": None}
    if plane is None:
        out["status"] = "no_plane"
        return out
    Pw, ok = LL.world_points(per, wc, fr, jp, gq)
    n_self = None
    if top_first and per.p["learn_top_first"].get("self_px") is not None:  # the robot's own pixels give no points
        own = self_pixels(wc, fr, jp, gq, int(per.p["learn_top_first"]["self_px"]), front_m=per.p.get("self_front_m"))
        n_self = int((ok & own & LL.above_plane(Pw, ok, plane)).sum())
        ok = ok & ~own
        Pw[own] = np.nan
    keep = LL.above_plane(Pw, ok, plane)
    P = Pw[keep]
    ex = LL.explained(wc, percepts, P)
    U = P[~ex]
    tried = []
    if top_first and msg.get("next"):  # ``learn_ask_known``: the next thing; the things tried before are left out
        prev = per.learn_state.get(ep)
        if prev is not None and prev.get("suite") == suite and prev.get("first_voxels") is not None:
            tried = list(prev.get("tried") or []) + [prev["first_voxels"]]
            U = drop_tried(U, tried, LL.LINK_M)
    V, comps = LL.components(U)
    lo = {"plane": plane, "n_above": int(keep.sum()), "n_explained": int(ex.sum()), "n_vox": int(len(V)),
          "comp_sizes": [int(len(c)) for c in comps[:5]]}
    if tried:
        lo["n_tried"] = len(tried)
    if n_self is not None:
        lo["self_px_points"] = n_self  # points above the support that lay in the robot's own pixels
    rk = per.p.get("learn_rank_known")
    if rk is not None and msg.get("rejected") and len(comps) > 1:
        # 2026-11-17, off by default: an unexplained thing that a REJECTED pose of a known type covers is most likely
        # that known object, missed by the first look, and not the new one: it is considered last. The robot sends
        # the rejected poses of the types it holds no accepted percept of (never the instruction's own type).
        rk = {**RANK_KNOWN, **rk}
        top = comps[:5]
        covers = [max((float(LL.explained(wc, [q], V[c]).mean()) for q in msg["rejected"]), default=0.0) for c in top]
        fl = per.p.get("learn_rank_floor")
        floor = None if fl is None else int({**RANK_FLOOR, **fl}["min_vox"])
        order = rank_components([len(c) for c in top], covers, float(rk["cover"]), LL.MIN_VOX,
                                None if rk["margin"] is None else float(rk["margin"]), float(rk["size_share"]),
                                int(rk["size_vox"]), floor)
        comps = [top[i] for i in order] + comps[5:]
        lo["rank_known"] = {"cover": [round(c, 3) for c in covers], "order": order, "n_rejected": len(msg["rejected"])}
        if floor is not None:
            lo["rank_known"]["floor"] = floor
        lo["comp_sizes"] = [int(len(c)) for c in comps[:5]]
    hn = per.p.get("learn_hint")
    if hn is not None and hn is not False and isinstance(msg.get("hint"), dict) and msg["hint"].get("xy") is not None \
            and len(comps) > 1:
        # 2026-11-19, off unless given (``HINT``): the thing at the place the robot's own plan would have taken the
        # instruction's object to be goes first
        hn = {**HINT, **(hn if isinstance(hn, dict) else {})}
        order, dist = hint_order(V, comps, msg["hint"]["xy"], float(hn["within_m"]), int(hn["top"]), LL.MIN_VOX)
        comps = [comps[i] for i in order] + comps[len(order):]
        lo["hint"] = {"xy": [round(float(x), 4) for x in msg["hint"]["xy"][:2]], "dist_m": [round(x, 3) for x in dist],
                      "order": order, "within_m": float(hn["within_m"])}
        lo["comp_sizes"] = [int(len(c)) for c in comps[:5]]
    if not comps or len(comps[0]) < LL.MIN_VOX:
        out.update(status="no_leftover", leftover=lo, s=round(time.perf_counter() - t0, 3))
        return out
    C = V[comps[0]]
    lo.update(centre_w=[float(x) for x in C.mean(0)], bbox_min=[float(x) for x in C.min(0)],
              bbox_max=[float(x) for x in C.max(0)], n_vox_object=int(len(C)),
              runner_up_vox=int(len(comps[1])) if len(comps) > 1 else 0)
    # the agentview's own mask of the new object (its far side, which no wrist look reaches): the leftover's pixels
    idx_un = np.flatnonzero(keep.ravel())[~ex]
    vk = np.floor(P[~ex] / LL.VOX_M).astype(np.int64)
    ck = {tuple(x) for x in np.floor(C / LL.VOX_M).astype(np.int64)}
    am = np.zeros(keep.size, bool)
    am[idx_un[np.array([tuple(x) in ck for x in vk], bool)]] = True
    z_a = np.asarray(fr["depth_u16"]).astype(np.float64) * float(fr["depth_scale_mm"]) * 1e-3
    agent_look = {"rgb": np.asarray(fr["rgb"]), "z": z_a, "K": np.asarray(fr["K"], float),
                  "T": np.asarray(fr["T_world_from_camera"], float), "mask": am.reshape(keep.shape) & (z_a > 0)}
    lo["agent_mask_px"] = int(agent_look["mask"].sum())
    a = time.perf_counter()
    w = msg["wrist"]
    arm = Arm(wc, w["joint_pos"], w["gripper_qpos"], w["eef_pos"], w["eef_quat_xyzw"],
              w["cameras"][0]["T_world_from_camera"])
    clear = Clearance(P, plane)
    hP = (P - np.array([0.0, 0.0, float(plane["height_w"])])) @ clear.n if len(P) else np.zeros(0)
    lo["scene"] = {"n": int(len(P)), "n_over_20cm": int((hP > 0.20).sum()), "n_over_25cm": int((hP > 0.25).sum())}
    if top_first:  # stage "top": one look from above the part of the leftover nearest the main camera
        o = {**TOP_FIRST, **per.p["learn_top_first"]}
        xy = near_point(C, np.asarray(fr["T_world_from_camera"], float)[:3, 3], float(o["near_share"]))
        aim = [float(xy[0]), float(xy[1]), float(plane["height_w"]) + float(o["aim_up_m"])]
        lo["aim_w"] = [round(x, 4) for x in aim]
        looks = [plan_top(arm, clear, aim, wc.base[:2])]
    else:
        looks = plan_looks(arm, clear, lo["centre_w"], wc.base[:2])
    rows = []
    for r in looks:
        row = {k: v for k, v in r.items() if k not in ("q", "waypoints", "u", "dist")}
        if "q" in r:
            row.update(_pose_doc(arm, r["q"]))
            row["waypoints"] = [{"el_deg": wp["el_deg"], **_pose_doc(arm, wp["q"])} for wp in r.get("waypoints", [])]
        rows.append(row)
    per.learn_state[ep] = {"suite": suite, "leftover": lo, "voxels": C, "percepts": percepts, "agent_look": agent_look,
                           "wrist0": (w["cameras"][0], w["joint_pos"], w["gripper_qpos"])}
    if top_first:
        per.learn_state[ep].update(stage="top", scene=P, first_voxels=C, tried=tried)
        out["stage"] = "top"
    out.update(status="ok", leftover=lo, looks=rows, n_planned=sum("eef_pos" in r for r in rows),
               mount={"hand_cam_t": [float(x) for x in arm.T_hand_cam[:3, 3]]},
               times={"leftover_s": round(a - t0, 3), "plan_s": round(time.perf_counter() - a, 3)})
    return out


def masks(per, wc, st: dict, frames: list) -> tuple[list, list]:
    """``lfb_learn.cmd_masks`` on frames in memory: (looks for the builders, per-look counts)."""
    from scipy.spatial import cKDTree

    from rowant_lab.libero_env import lfb_learn as LL

    lo = st["leftover"]
    ftree = cKDTree(st["voxels"][:, :2])
    top = float(lo["bbox_max"][2]) + LL.Z_PAD_M
    lks, rows = [], []
    for i, (fr, jp, gq) in enumerate(frames):
        Pw, ok = LL.world_points(per, wc, fr, jp, gq)
        keep = LL.above_plane(Pw, ok, lo["plane"], LL.MASK_BAND_M) & (np.nan_to_num(Pw[..., 2], nan=1e9) < top)
        idx = np.flatnonzero(keep.ravel())
        P = Pw.reshape(-1, 3)[idx]
        near, _ = ftree.query(P[:, :2], k=1, distance_upper_bound=LL.R_PAD_M)
        sel = np.isfinite(near) & ~LL.explained(wc, st["percepts"], P)
        idx, P = idx[sel], P[sel]
        V, comps = LL.components(P)
        m = np.zeros(keep.size, bool)
        if comps:
            vk = np.floor(P / LL.VOX_M).astype(np.int64)
            ck = {tuple(x) for x in np.floor(V[comps[0]] / LL.VOX_M).astype(np.int64)}
            m[idx[np.array([tuple(x) in ck for x in vk], bool)]] = True
        Pm = Pw.reshape(-1, 3)[m]
        m = m.reshape(keep.shape)
        z = np.asarray(fr["depth_u16"]).astype(np.float64) * float(fr["depth_scale_mm"]) * 1e-3
        rows.append({"look": i, "n_px": int(m.sum()), "comp_sizes": [int(len(c)) for c in comps[:3]],
                     "mm_min": [round(float(x) * 1e3, 1) for x in Pm.min(0)] if len(Pm) else None,
                     "mm_max": [round(float(x) * 1e3, 1) for x in Pm.max(0)] if len(Pm) else None})
        if m.sum():
            lks.append({"rgb": np.asarray(fr["rgb"]), "z": z, "K": np.asarray(fr["K"], float),
                        "T": np.asarray(fr["T_world_from_camera"], float), "mask": m & (z > 0)})
    return lks, rows


def build_model(lks, lo: dict, label: str, out: Path) -> dict:
    """TSDF model (``lfb_build.build``'s tsdf branch) then ST-1's F7 entry (``st_pod.cmd_materialize``, observed mode
    with auto-shrink). Writes the model folder ``build_learned`` reads; the TSDF mesh stays in ``out/tsdf``."""
    from rowant_lab.fastlayer import onboard as OB
    from rowant_lab.libero_env import lfb_build as LB_
    from rowant_lab.libero_env import m2_onboard as OBM

    t0 = time.perf_counter()
    o = LB_.origin(lo)
    (out / "tsdf").mkdir(parents=True, exist_ok=False)
    m = LB_.tsdf_build(lks, o, TSDF_MAIN)
    LB_.write_ply(out / "tsdf" / "render.ply", m["V"], m["F"], m["N"], m["RGB"])
    pos, x, _diag = OBM.build_object_grid("1", out / "tsdf" / "render.ply", None, OB.GRID_M)
    seen = LB_.observed(lks, o, pos - OB.P_TRAIN, x[:, 0:3], float(TSDF_MAIN["vox_mm"]) * 1e-3)
    pos, x = pos[seen], x[seen]
    info = {"origin_w": [float(v) for v in o], "n_looks": len(lks), "n_points": m["n_points"], "grid": m["grid"],
            "n_nodes_unobserved": int((~seen).sum()), "tsdf_s": round(time.perf_counter() - t0, 3)}
    if not len(pos):
        info["status"] = "no_nodes"
        return info
    P = pos - OB.P_TRAIN
    c = np.array([*P[:, :2].mean(0), 0.0])  # scale about the footprint centre on the support plane (z = 0)
    hs = OB.FEATURE_MAPPING["hsv"][0]
    s, d, pos_s = 1.0, None, pos
    for _ in range(F7_TRIES):
        pos_s = (P - c) * s + c + OB.P_TRAIN
        d = LB_.derive(pos_s - OB.P_TRAIN, x[:, hs:hs + 3], label)
        if int(d["records"].get("n_records", 0)):
            break
        s = round(s * F7_SHRINK, 4)
    np.savez(out / "nodes.npz", pos=np.ascontiguousarray(pos_s), x=np.ascontiguousarray(x))
    LB_.write_ply(out / "surface.ply", d["V"], d["F"], d["N"], d["RGB"])
    (out / "m1type.json").write_text(json.dumps(d["m1type"], indent=1, sort_keys=True))
    (out / "records.json").write_text(json.dumps(d["records"], sort_keys=True) + "\n")
    info.update(status="ok", n_nodes=int(len(pos_s)), scale_used=s, n_boxes=d["n_boxes"],
                n_records=int(d["records"].get("n_records", 0)), extent_mm=d["extent_mm"],
                model_s=round(time.perf_counter() - t0, 3))
    (out / "build.json").write_text(json.dumps(info, indent=1))
    return info


# ------------------------------------------------------------------------------------------------ check before replacing
def _lift(lk: dict) -> np.ndarray:
    """A look's masked pixels as world points (``lfb_learn.world_points``' pixel-centre rule)."""
    z, K, T = lk["z"], np.asarray(lk["K"], float), np.asarray(lk["T"], float)
    v, u = np.nonzero(lk["mask"])
    zz = z[v, u]
    Pc = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * zz, (v + 0.5 - K[1, 2]) / K[1, 1] * zz, zz], 1)
    return Pc @ T[:3, :3].T + T[:3, 3]


def _agreement(Xw, Nw, obs: list, p: dict, soft: bool = False) -> tuple[float, float]:
    """Posed model nodes (world) against every look's depth, summed: (agreeing, seen through). Per look this is the
    server's ``fit_check`` rule: nodes facing the camera, agreeing within ``agree_m``, seen through (observed farther
    than the surface) beyond ``see_through_m``. ``soft`` weighs agreement with ``upright_search``'s Epanechnikov kernel."""
    na, nt = 0.0, 0.0
    am, tm, fc = float(p["agree_m"]), float(p["see_through_m"]), float(p["facing_cos"])
    for z, K, T in obs:
        R, t = T[:3, :3], T[:3, 3]
        X = (Xw - t) @ R  # camera frame
        N = Nw @ R
        vis = (-(N * X).sum(1) / np.maximum(np.linalg.norm(X, axis=1), 1e-9) > fc) & (X[:, 2] > 0.05)
        X = X[vis]
        H, W = z.shape
        u = np.rint(K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2] - 0.5).astype(int)
        v = np.rint(K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2] - 0.5).astype(int)
        ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        zo = z[v[ins], u[ins]]
        has = np.isfinite(zo)
        dz = np.where(has, zo - X[ins, 2], 0.0)
        if soft:
            na += float(np.where(has, np.clip(1.0 - (dz / am) ** 2, 0.0, 1.0), 0.0).sum())
        else:
            na += float((has & (np.abs(dz) <= am)).sum())
        nt += float((has & (dz > tm)).sum())
    return na, nt


def _in_plane(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 0.1:
        a = np.cross(n, [0.0, 1.0, 0.0])
    ex = a / np.linalg.norm(a)
    return ex, np.cross(n, ex)


def _align(a, b) -> np.ndarray:
    """The smallest rotation taking direction ``a`` onto direction ``b``."""
    from rowant_lab.libero_env import m2_kin as KIN

    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    ax = np.cross(a, b)
    s, c = float(np.linalg.norm(ax)), float(a @ b)
    if s < 1e-9:
        return np.eye(3) if c > 0 else KIN.axis_angle(_in_plane(a)[0], math.pi)
    return KIN.axis_angle(ax / s, math.atan2(s, c))


def check_own(per, wc, label: str, lks: list, plane: dict) -> dict:
    """Check before replacing: does the robot's own model of ``label`` explain what the looks saw of the object?

    The model stands upright on the support plane (its rest-up axis along the plane normal), and its yaw and in-plane
    position are searched around the object's observed centre: coarse (``yaw_step_deg``, ``xy_step_m``, hard counts),
    then fine (soft agreement), then a small height offset. The score is ``upright_search``'s (agreeing minus twice
    the seen-through nodes), over all the looks at once. It explains the object when, at the best pose:
    * the server's acceptance floor holds over the pooled looks (``accept_inlier_region``, ``accept_min_agree``): the
      model does not fill space the cameras saw through, and enough of it agrees;
    * and, the other way round, at least ``cover_min`` of the object's observed points lie within ``cover_m`` of the
      model: the object does not reach beyond the model.
    Returns the root body's world pose when it does."""
    from scipy.spatial import cKDTree

    from rowant_lab.libero_env import m2_kin as KIN

    t0 = time.perf_counter()
    cfg, p = CHECK_OWN, per.p
    tdoc = wc.types.get(label)
    if tdoc is None or (label, tdoc["root_body"]) not in wc.gid:
        return {"own": False, "status": "no_type"}
    g = str(wc.gid[(label, tdoc["root_body"])])
    nodes, nrm = (np.asarray(a, float) for a in wc.nodes[g])
    up_b = np.asarray(wc.rest_up[g], float)
    up_b = up_b / np.linalg.norm(up_b)
    n = np.asarray(plane["normal_w"], float)
    n = n / np.linalg.norm(n) * (1.0 if n[2] >= 0 else -1.0)
    p0 = np.array([0.0, 0.0, float(plane["height_w"])])
    obj = np.concatenate([_lift(lk) for lk in lks]) if lks else np.zeros((0, 3))
    obj = obj[np.isfinite(obj).all(1)]
    if len(obj) < 50:
        return {"own": False, "status": "no_points", "n_obj": int(len(obj))}
    m = obj.mean(0)
    q0 = m - float((m - p0) @ n) * n  # the object's observed centre, dropped onto the support
    ex, ey = _in_plane(n)
    s = nodes @ up_b
    c = 0.5 * (nodes.min(0) + nodes.max(0))
    c_bot = c - (float(c @ up_b) - float(s.min())) * up_b  # the model's footprint centre on its resting face
    R0 = _align(up_b, n)
    sel = np.linspace(0, len(nodes) - 1, min(len(nodes), int(cfg["nodes"]))).astype(int)
    Xb, Nb = nodes[sel] - c_bot, nrm[sel]
    obs = [(np.where(lk["z"] > 0, lk["z"], np.nan), np.asarray(lk["K"], float), np.asarray(lk["T"], float))
           for lk in lks]
    rot = {}

    def pose(yaw: float, o, dz: float):
        if yaw not in rot:
            rot[yaw] = KIN.axis_angle(n, yaw) @ R0
        return rot[yaw], q0 + o[0] * ex + o[1] * ey + dz * n

    def score(yaw: float, o, dz: float, soft: bool) -> float:
        R, t = pose(yaw, o, dz)
        a, th = _agreement(Xb @ R.T + t, Nb @ R.T, obs, p, soft)
        return a - 2.0 * th

    def grid(r: float, st_: float) -> list:
        k = np.arange(-r, r + 1e-9, st_)
        return [(float(x), float(y)) for x in k for y in k]

    yaws = [float(y) for y in np.radians(np.arange(0.0, 360.0, cfg["yaw_step_deg"]))]
    _s, y0, o0 = max(((score(y, o, 0.0, False), y, o) for y in yaws for o in grid(cfg["xy_m"], cfg["xy_step_m"])),
                     key=lambda r: r[0])
    fy = [y0 + float(d) for d in np.radians(np.arange(-cfg["fine_yaw_deg"], cfg["fine_yaw_deg"] + 1e-9,
                                                      cfg["fine_yaw_step_deg"]))]
    _s, y1, o1 = max(((score(y, (o0[0] + dx, o0[1] + dy), 0.0, True), y, (o0[0] + dx, o0[1] + dy))
                      for y in fy for dx, dy in grid(cfg["fine_xy_m"], cfg["fine_xy_step_m"])), key=lambda r: r[0])
    dz1 = max(cfg["dz_m"], key=lambda d: score(y1, o1, d, True))
    R, t = pose(y1, o1, dz1)
    na, nt = _agreement(Xb @ R.T + t, Nb @ R.T, obs, p, False)
    inlier = na / max(na + nt, 1.0)
    d, _ = cKDTree((nodes - c_bot) @ R.T + t).query(obj, k=1, distance_upper_bound=float(cfg["cover_m"]))
    cover = float(np.isfinite(d).mean())
    own = bool(inlier >= float(p["accept_inlier_region"]) and na >= float(p["accept_min_agree"])
               and cover >= float(cfg["cover_min"]))
    return {"own": own, "status": "own" if own else "learn", "gid": g, "n_agree": int(na), "n_through": int(nt),
            "inlier": round(inlier, 4), "cover": round(cover, 4), "n_obj": int(len(obj)), "n_looks": len(lks),
            "yaw_deg": round(math.degrees(y1) % 360.0, 1), "offset_m": [round(o1[0], 4), round(o1[1], 4)],
            "dz_m": dz1, "R_w": [[float(x) for x in r] for r in R], "t_w": [float(x) for x in t - R @ c_bot],
            "s": round(time.perf_counter() - t0, 3)}


def drop_tried(U: np.ndarray, tried: list, link_m: float) -> np.ndarray:
    """The unexplained points ``U`` less those of the things tried before (their first-look voxels): a point within
    the components' own linking distance of a tried thing's voxel belongs to that thing."""
    from scipy.spatial import cKDTree

    if not len(U) or not tried:
        return U
    d, _ = cKDTree(np.concatenate(tried)).query(U, k=1, distance_upper_bound=float(link_m))
    return U[~np.isfinite(d)]


def fit_known(per, wc, label: str, obj: np.ndarray, obs: list, plane: dict, cfg: dict) -> dict:
    """``learn_ask_known``: does the robot's model of the known type ``label`` explain the thing ``obj`` (its points
    in the world) as the look(s) ``obs`` saw it? ``check_own``'s search, for one look and any type: the model upright
    on the support, its yaw and in-plane position searched around the thing's centre out to the model's own half
    width (between ``xy_min_m`` and ``xy_max_m``: of a large thing a part is seen, and its centre is not the
    model's), coarse then fine, then a small height offset. At the best pose:
    * ``inlier``: agreeing nodes over agreeing plus seen-through ones (the model must not fill space seen through);
    * ``seen``: agreeing nodes over the nodes the look could have agreed with (facing the camera, in the picture,
      with a reading);
    * ``cover``: the share of the thing's points within ``cover_m`` of the model.
    ``explains`` when all three reach their floors (``inlier_min``, ``seen_min``, ``cover_min``; a None refuses)."""
    from scipy.spatial import cKDTree

    from rowant_lab.libero_env import m2_kin as KIN

    t0 = time.perf_counter()
    p = per.p
    tdoc = wc.types.get(label)
    if tdoc is None or (label, tdoc["root_body"]) not in wc.gid:
        return {"type": label, "explains": False, "status": "no_type"}
    g = str(wc.gid[(label, tdoc["root_body"])])
    nodes, nrm = (np.asarray(a, float) for a in wc.nodes[g])
    up_b = np.asarray(wc.rest_up[g], float)
    up_b = up_b / np.linalg.norm(up_b)
    n = np.asarray(plane["normal_w"], float)
    n = n / np.linalg.norm(n) * (1.0 if n[2] >= 0 else -1.0)
    p0 = np.array([0.0, 0.0, float(plane["height_w"])])
    obj = np.asarray(obj, float)
    obj = obj[np.isfinite(obj).all(1)]
    if len(obj) < 20:
        return {"type": label, "explains": False, "status": "no_points", "n_obj": int(len(obj))}
    m = obj.mean(0)
    q0 = m - float((m - p0) @ n) * n
    ex, ey = _in_plane(n)
    s = nodes @ up_b
    c = 0.5 * (nodes.min(0) + nodes.max(0))
    c_bot = c - (float(c @ up_b) - float(s.min())) * up_b
    R0 = _align(up_b, n)
    flat = (nodes - c_bot) - np.outer((nodes - c_bot) @ up_b, up_b)  # the model's footprint about its centre
    half = float(np.linalg.norm(flat, axis=1).max())
    r_xy = float(np.clip(half, cfg["xy_min_m"], cfg["xy_max_m"]))
    sel = np.linspace(0, len(nodes) - 1, min(len(nodes), int(cfg["nodes"]))).astype(int)
    Xb, Nb = nodes[sel] - c_bot, nrm[sel]
    rot = {}

    def pose(yaw: float, o, dz: float):
        if yaw not in rot:
            rot[yaw] = KIN.axis_angle(n, yaw) @ R0
        return rot[yaw], q0 + o[0] * ex + o[1] * ey + dz * n

    def score(yaw: float, o, dz: float, soft: bool) -> float:
        R, t = pose(yaw, o, dz)
        a, th = _agreement(Xb @ R.T + t, Nb @ R.T, obs, p, soft)
        return a - 2.0 * th

    def grid(r: float, st_: float) -> list:
        k = np.arange(-r, r + 1e-9, st_)
        return [(float(x), float(y)) for x in k for y in k if x * x + y * y <= r * r + 1e-12]

    yaws = [float(y) for y in np.radians(np.arange(0.0, 360.0, cfg["yaw_step_deg"]))]
    coarse = grid(r_xy, float(cfg["xy_step_m"]))
    _s, y0, o0 = max(((score(y, o, 0.0, False), y, o) for y in yaws for o in coarse), key=lambda r: r[0])
    fy = [y0 + float(d) for d in np.radians(np.arange(-cfg["fine_yaw_deg"], cfg["fine_yaw_deg"] + 1e-9,
                                                      cfg["fine_yaw_step_deg"]))]
    fine = grid(float(cfg["fine_xy_m"]) * math.sqrt(2.0), float(cfg["fine_xy_step_m"]))
    _s, y1, o1 = max(((score(y, (o0[0] + dx, o0[1] + dy), 0.0, True), y, (o0[0] + dx, o0[1] + dy))
                      for y in fy for dx, dy in fine), key=lambda r: r[0])
    dz1 = max(cfg["dz_m"], key=lambda d: score(y1, o1, d, True))
    R, t = pose(y1, o1, dz1)
    na, nt = _agreement(Xb @ R.T + t, Nb @ R.T, obs, p, False)
    seen_n, _ = _agreement(Xb @ R.T + t, Nb @ R.T, obs, {**p, "agree_m": 1e9}, False)  # nodes with a reading at all
    inlier = na / max(na + nt, 1.0)
    seen = na / max(seen_n, 1.0)
    d, _ = cKDTree((nodes - c_bot) @ R.T + t).query(obj, k=1, distance_upper_bound=float(cfg["cover_m"]))
    cover = float(np.isfinite(d).mean())
    floors = (cfg.get("inlier_min"), cfg.get("seen_min"), cfg.get("cover_min"))
    explains = bool(all(x is not None for x in floors) and inlier >= float(floors[0]) and seen >= float(floors[1])
                    and cover >= float(floors[2]))
    return {"type": label, "explains": explains, "status": "fit", "gid": g, "n_agree": int(na), "n_through": int(nt),
            "n_seen": int(seen_n), "inlier": round(inlier, 4), "seen": round(seen, 4), "cover": round(cover, 4),
            "n_obj": int(len(obj)), "half_m": round(half, 4), "search_m": round(r_xy, 4), "n_poses": len(yaws) * len(coarse),
            "yaw_deg": round(math.degrees(y1) % 360.0, 1), "offset_m": [round(o1[0], 4), round(o1[1], 4)], "dz_m": dz1,
            "R_w": [[float(x) for x in r] for r in R], "t_w": [float(x) for x in t - R @ c_bot],
            "s": round(time.perf_counter() - t0, 3)}


def learn(per, msg: dict) -> dict:
    from rowant_lab.libero_env import lfb_library as LLIB
    from rowant_lab.libero_env import m2_library as LB
    from rowant_lab.libero_env import m2_segment as SEG

    t0 = time.perf_counter()
    suite, ep, label = msg["suite"], msg["episode"], msg["label"]
    st = per.learn_state.get(ep)
    if st is None or st["suite"] != suite:
        return {"ok": True, "status": "no_plan"}
    wc = per.cell(suite)
    frames = [(fr, j["joint_pos"], j["gripper_qpos"]) for fr, j in zip(msg["cameras"], msg["joints"])]
    # the wrist frame from before any motion joins them (it adds pixels only when the object is in its view); after
    # a top look (``learn_top_first``) that look does, with the depth it already has
    first = st["top"] if st.get("top") is not None else st["wrist0"]
    srows = None
    if per.p.get("learn_wrist_stereo") is not None:  # wrist frames that came without depth: the plane sweep's
        srows = [wrist_depth(fr, per.p["learn_wrist_stereo"]) if st.get("top") is None or k else {"status": "top"}
                 for k, (fr, _jp, _gq) in enumerate([first] + frames)]
    stereo_s = round(time.perf_counter() - t0, 3)
    lks, mrows = masks(per, wc, st, [first] + frames)
    times = {"masks_s": round(time.perf_counter() - t0 - stereo_s, 3)}
    out = {"ok": True, "masks": mrows, "n_looks": len(lks)}
    if srows is not None:
        out["stereo"] = srows
        times["stereo_s"] = stereo_s
    if lks and st["agent_look"]["mask"].any() and not st.get("no_agent"):  # the agentview frame joins the wrist
        lks = [st["agent_look"]] + lks  # looks in the TSDF (not after a top look: its depth of this thing is a guess)
    # (after a top look it may still carve the model, ``agent: "carve"``; it is no look for the check below)
    carve = [st["agent_carve"]] if st.get("no_agent") and st.get("agent_carve") is not None else []
    if not lks:
        out.update(status="no_masks", times=times)
        return out
    if per.p.get("learn_check_own"):  # W2-LEARN-EP2: check before replacing; nothing is built when the model explains it
        a = time.perf_counter()
        chk = check_own(per, wc, label, lks, st["leftover"]["plane"])
        times["check_s"] = round(time.perf_counter() - a, 3)
        out["check"] = {k: v for k, v in chk.items() if k not in ("R_w", "t_w")}
        if chk.get("own"):
            out.update(status="own", own={k: chk[k] for k in ("R_w", "t_w", "gid", "n_agree", "inlier", "cover")},
                       times=times)
            return out
    root = learn_dir()
    mdir = root / f"model-{ep}"
    a = time.perf_counter()
    info = build_model(lks + carve, st["leftover"], label, mdir)
    times["model_s"] = round(time.perf_counter() - a, 3)
    out["model"] = {k: v for k, v in info.items() if k != "status"}
    if info.get("status") != "ok":
        out.update(status=info.get("status"), times=times)
        return out
    a = time.perf_counter()
    base = per.p.get("lfb_library")
    lib_root = root / f"lib-{ep}"
    var = LLIB.build_learned(suite, label, mdir, f"ie-{ep}", root=lib_root, base=base)
    times["variant_s"] = round(time.perf_counter() - a, 3)
    a = time.perf_counter()
    p_ep = copy.deepcopy(per.p)
    p_ep["lfb_library"] = str(lib_root)
    routes = p_ep.setdefault("routes", {})
    rt = routes.setdefault(suite, {"search": [], "texture_first": [], "fallback": []})
    rt["search"] = sorted({*(rt.get("search") or []), label})
    rt["texture_first"] = sorted({*(rt.get("texture_first") or []), label})
    p_ep["support"] = {**SEG.PARAMS, **(per.p.get("support") or {}), **F7_SUPPORT}
    _ob, pl_, m1_ = LB.lfb_dirs(str(lib_root))
    reader = LB.LibraryReader(pl_, m1_, allow_panda=True, extra_suites=p_ep.get("extra_suites"))
    Workcell = type(wc)  # the server's own workcell class (this module does not import the server)
    per.ep_scope[ep] = {"p": p_ep, "cells": {suite: Workcell(suite, p_ep, reader)}, "lib_root": lib_root}
    times["workcell_s"] = round(time.perf_counter() - a, 3)
    times["total_s"] = round(time.perf_counter() - t0, 3)
    out.update(status="ok", variant=str(lib_root), policy_lib=str(pl_), m1_library=str(m1_), gid=var.get("gid"),
               times=times)
    return out


def _remove_tree(root: Path) -> None:
    """Bottom-up removal by absolute paths (``shutil.rmtree`` works through directory handles with relative names,
    which the server's audit hook would resolve against its working folder and refuse)."""
    import os

    for d, dirs, files in os.walk(root, topdown=False):
        for f in files:
            os.remove(os.path.join(d, f))
        for x in dirs:
            os.rmdir(os.path.join(d, x))
    os.rmdir(root)


def end_episode(per) -> None:
    """At ``begin_episode``: the previous episodes' variant trees go (their model folders stay, as records)."""
    for sc in per.ep_scope.values():
        root = Path(sc["lib_root"])
        if root.is_dir() and root.parent == learn_dir():
            _remove_tree(root)
    per.ep_scope.clear()
    per.learn_state.clear()
