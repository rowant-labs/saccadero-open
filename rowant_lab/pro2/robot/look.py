"""M2 policy-side agent pieces (plan modules.m2_agent): the perception-backed pose tool.

M2 (perceived poses, grounded instruction): perception in the loop. ``PerceptionTool`` has M1's ``OracleTool``
interface (``tool() -> {body: {pos, quat_wxyz}}``), so M1's frozen ``OracleAgent`` drives it unchanged: every call is
a look. The first call returns the L0 percepts; each later call is a TRACK look of the grounded target (the other
percepts are static and keep their L0 pose: a declared amendment), predicted static or, while the gripper is closed
around it, rigidly with the hand from proprioception. When the gripper opens, one MERGED full look re-measures the
scene and keeps the percept ids and body names (association). Over the look budget, or when a track fails, the
prediction is returned. At the release (amendment A16) the target's stored pose is first moved to the hand
prediction at that moment, so the merged look can associate the placed object (it had kept the lift-time track, so C3
read a stale pose in every episode of one frozen run), and the association is nearest-pair first.

This module is policy side: it holds no simulator handle, reads no file and imports nothing of the runner, the
evaluator, LIBERO, robosuite or MuJoCo (the truth guard enforces it). The frames come from the camera tool through
the bridge client the runner passes in. ``obs`` is set by the runner each step to ``tools.filter_obs`` output (the
proprioception whitelist only; verifier B2), and ``episode`` is the runner's opaque per-episode token (amendment
A11), never the suite, task and state. The types it reads are ``lib.export_types`` (kinematics and sites); M1's agent
reads ``lib.types`` (M1's primitives).
"""

from __future__ import annotations

import math
import time

import numpy as np

from rowant_lab.libero_env import m2_paths as MP
from .grasp import quat_xyzw_to_mat
from rowant_lab.libero_env.m2_kin import mat_to_quat, part_body_poses

LOOK_BUDGET = 24
WRIST = "robot0_eye_in_hand"  # item 3's close looks (``close_look``)
CLOSE_LOOK_BASE = 1000  # close looks' own look indices (the server seeds each look by its index)
CLOSED_MARGIN_M = 0.005  # the jaw gap this much below the fully open gap counts as closed on something
REASSOC_M = 0.10  # a merged full look matches a percept to the new percept of its type within this
ROUTE_STILL_M = 0.02  # ``look_route``: things that stand still read further off than this (median): not believed
RELOOK_MAX_M = 0.20  # a receiver looked at again may have moved up to this far (measured jumps: 7.5 and 15 cm)
STILL_M = 0.02  # seen within this of where it was = seen where it was (the re-look's own threshold for "moved")
BLOB_NEAR_M = 0.02  # a blob this near another reported object's footprint is taken as that object's leftover
FACE_JUMP_M = 0.08  # ``face_in``: the tracker's pose of a turned fixture is taken within this of where it was put
RELOOK_DZ_M = 0.03  # ``relook_strict``: a receiver moves across its surface; a new place this much higher or lower is not it
MEMORY_NEAR_M = 0.06  # ``fixture_memory``: a region's top must lie this near the remembered carcass's middle
MEMORY_LIFT_M = 0.02  # ``fixture_memory``: a top this far above the model's is the fixture standing that much higher
# ``fixture_memory``, the entry's ``size`` (2026-11-17, a dict of overrides of these; off unless given): a remembered
# fixture whose region tops out at ANOTHER height than its model's stands there at another size (``sized.py``)
MEMORY_SIZE = {"same_m": 0.015,   # a top within this of the model's: the fixture as stored (as before)
               "lift_m": 0.075,   # a top this far above it or more: the fixture as stored, standing on something
               #                    (measured: on a 10 cm block 0.095 to 0.10; drawn a quarter taller 0.048 to 0.056)
               "snap": 0.08,      # the top layer's long side within this share of the model's: as wide as stored
               "lo": 0.6, "hi": 1.5}  # no factor outside these: then the look does not show this fixture resized
FALLBACK_OVER_M = 0.03  # ``fallback_top``: a rejected pose explains its place when no unexplained region tops it by more
FALLBACK_RAISE_M = 0.06  # ``fallback_top``: a region this much above a rejected pose's top is what the object stands on
#                          (the blocks under a receiver are 7 to 16 cm; an enlarged bowl topped a false plate pose by 4)
FALLBACK_WHOLE_H = (0.6, 1.5)  # ``fallback_top``: a region's height over the model's, for a fit as a whole
FALLBACK_WHOLE_COST = 0.9  # ``fallback_top``: a region fits a type as a whole below this cost (a plate drawn 0.8 times: 0.43)
FALLBACK_SIZE = (0.6, 1.6)  # ``fallback_top``: a region's (or its top layer's) size over the model's, per axis
STANDIN_SEEN = 0.25  # ``standin_strict``: the least share of a stand-in's visible model points the look agrees with
STANDIN_SHARE = 0.75  # ``standin_strict``: and of its points with nothing nearer in front (the server accepts at 0.85)
HELD_NEAR_M = 0.12
REJ_NEAR_M = 0.12  # a target dropped at the pick: a rejected pose of its type this near the pick location (xy)
REFUTE_M = 0.05  # ``refute_target``: a candidate this near (xy) a place where the hand closed on nothing is that place
# 2026-11-16 night, ``params["fallback_stand"]`` (a dict of overrides of these; off by default): a named TARGET the
# look did not report is stood where an unexplained region shows it (``_stand_candidates``).
STAND = {"top_m": 0.015,          # the model's own top layer: what of it reaches this near its highest point
         "height": (0.78, 1.25),  # the region's top layer over the model's height, both from the surface under the
         "height_abs_m": 0.01,    # region; or within this of it (a flat box read 4 mm high is 1.24 times its height)
         "min_px": 300,           # the region's least size
         "iou": 0.6,              # by the outline: the stood model's box in the picture over the region's, at least
         "size": (0.5, 1.8),      # by the layer: the region's top layer over the model's top layer, per axis across
         "slide": True,           # by the layer: its middle slid along the camera's line of sight to the model's height
         "slide_max_m": 0.06,     # (never further across than this: then the region is something else)
         "long": 1.25,            # a layer and a model top both this much longer than wide: the model is turned onto it
         "near_m": 0.02,          # a region is named when a rejected pose of the type lies this near its box
         "same_m": 0.05,          # the older fallback's pose this near the stood place is the same thing: it stays
         "drop_m": 0.03,          # a reported object of another type this near the stood target, and lower, was the
         "drop_dz_m": 0.01,       # target read as that type: it leaves the scene
         "track": False,          # the stood model takes its type's model id from the look, so the server can
         "track_max_m": 0.02,     # track it; a track that moves it further across than this is not taken
         "hang": 0.3}             # a region ending above the support in more of its columns than this share hangs:
#                                   nothing stands there (unless it sticks out of a reported object: see drop_m)
# ``fallback_stand``'s key ``resize`` (2026-11-17, a dict of overrides of these; off unless given): a named target whose
# region stands at ANOTHER height than its model (outside ``STAND["height"]``) is stood there with the model drawn
# to that height: taller or lower by the region's own measure, as wide as stored (``sized.py``). The height is what a
# look reads of a standing thing to a centimetre; its width is left to the fingers (the agent's ``c1_soft``).
STAND_SIZE = {"lo": 0.6, "hi": 1.5,   # the region's top layer over the model's height: no factor outside these
              "min_m": 0.015,         # and at least this far from the model's own height (a flat box read 4 mm high)
              "min_px": 500}          # the region's least size


# ``params["misread_target"]`` (2026-11-17, a dict of overrides of these; off unless given): the named target the look
# did not report, read as ANOTHER type where it stands (``PerceptionTool._misread``)
MISREAD = {"weak": 0.65,          # the other type's fit: the share of its visible model points the look agrees with
           "over_m": 0.012,       # a region's top layer this far above that percept's top, on its footprint
           "height": (0.5, 1.3),  # that layer over the target model's height, from the percept's base
           "size": (0.6, 1.6),    # the percept's size across over the target model's, per axis
           "near_m": 0.02,        # the footprint is grown by this for the layer's middle
           "resize": True}        # the target drawn to what was measured (only a type with no percept at all)


class PerceptionTool:
    """``tool()`` -> {body: {pos, quat_wxyz}} of every percept body; each call is a look."""

    def __init__(self, lib, client, cam_tool, suite: str, episode: str, budget: int = LOOK_BUDGET,
                 held_near_m: float = HELD_NEAR_M, place_check: dict | None = None):
        self.lib, self.client, self.cam, self.suite, self.episode = lib, client, cam_tool, suite, episode
        self.budget = budget
        self.held_near_m = float(held_near_m)  # ST-1 F4: a learned model's origin is at its base, not its middle
        self.place_check = place_check  # ST-1 H8: {rise_m, min_px}; None keeps C3 as frozen
        self.calls = 0
        self.call_steps: list = []
        self.t_now = 0
        self.percepts: list = []
        self.target = None
        self.obs = None
        self.looks: list = []
        self.n_full = self.n_track = self.n_predicted = 0
        # record only (2026-11-18): the pixels each full look took as the robot's own, and, where the look left the
        # robot out in the picture too (``self_px``), how many of them only the picture found: [n] or [n, n_picture]
        self.own_px: list = []
        self.own_dz: list = []
        self.n_close = 0  # item 3: close looks (outside the budget; the agent caps them)
        self.redetect = False  # item 5 (set by the agent's ``redetect``): a refused track, hand open -> a full look
        self.table_bound = False  # long-task suite (``params["fallback_on_table"]``): fallbacks only over the table
        # 2026-11-16 (``params["fallback_blob_clear"]``, with ``table_bound``; off by default): among unexplained blobs,
        # one clear of every reported object's footprint comes before one inside a footprint (that object's leftover)
        self.blob_clear_first = False
        # 2026-11-16 (``params["fallback_blob_on_table"]``; off by default): a percept built from a blob never stands
        # lower than the table top under it. The blob's own base is the server's support plane, which in the long
        # suite's living-room cell lies 1.1 cm under the table top (the table box's middle): a flat box built from
        # its blob sat that far inside the table and every side grasp met the table (two cases of one task, the
        # first object: 174 and 26 of 292 grasp records)
        self.blob_on_table = False
        self.place_free = ()  # ids a merged look may move further than REASSOC_M (the agent's ``relook_place``)
        self.static_check = True  # ``relook_place`` believes a move only with the rest of the scene seen in place
        # 2026-11-16, off by default (``params["relook_strict"]``): a receiver's believed move must stay at its height
        # and may not land on another known object (``_move_refused``)
        self.relook_strict = False
        # 2026-11-16, off by default (``params["fallback_top"]``): a named type with no
        # reported percept follows the evidence (``_from_evidence``) before the older fallbacks
        self.fallback_top = False
        # 2026-11-17, off by default (``params["fallback_own"]``, with ``fallback_top``): a region that is a reported
        # object's own leftover is never the stand-in for a receiver (``_from_evidence``, ``_own_leftover``)
        self.fallback_own = False
        self.landmark_fb: list = []  # ``relation_candidates``: stand-ins for landmark types with no percept
        self.wrist_opts = None  # ``relation_look``: options of the wrist picture's search for its look (``wrist2d.find``)
        # ``params["look_blind"]`` (development only, 2026-11-16; off unless given): {"type", "xy", "within_m"}: every
        # full look leaves out what it reports of that type within that distance of that place, as if the main camera
        # did not make the object out. For trying, on a plain scene, what the robot does when the named object is
        # not in the look (the look at the relation's place); never set in a scored or compared run.
        self.blind = None
        # ``params["relation_standin_strict"]``: a landmark's stand-in must be seen in the look (``_standin_seen``),
        # and under a reported object of the target's type be as tall as that object stands high (``_standin_fits``)
        self.standin_strict = False
        # ``params["keep_target"]``: {instruction: what ``remember`` returned} of an earlier plan in this episode (the
        # runner hands it to the new tool when it looks and plans again)
        self.prior: dict = {}
        self.relation_target = None  # ``relation_candidates``: the id of the candidate it added, when it added one
        self._assoc_all = None
        self.face_in_on = False  # ``face_in``: a fixture's opening parts face the workspace (the episode planner sets it)
        self._hand0 = None       # where the hand was when the episode's first plan was made
        self.face_in_mode = None  # "carcass" (2026-11-16): the half turn is about the carcass's middle, no tracker
        # 2026-11-17, off unless the planner sets it (``params["fixture_memory"]``, this workcell's part of it:
        # {type: {"pos", "yaw_deg"}}): a named fixture with no accepted percept is taken at its remembered place when
        # an unexplained region's top lies over it (``_from_memory``), before a rejected fit or a blob.
        self.fixture_memory = None
        self.n_faced = 0
        self.no_flip = False  # ``params["fixture_no_flip"]``: a merged look never turns a fixture around in place
        # ``params["fixture_turn_max_deg"]`` (2026-11-17, off unless set): with ``no_flip``, the turn between two
        # looks beyond which the new fit of a fixture is refused (120 as built: a mirror fit). At 45 a quarter turn
        # is refused too: the plain drawer test task's one loss in 30 seeds was the look taken with the hand over
        # the cabinet reading it a quarter turn round, after the first look had it right
        self.turn_max_deg = None
        self.turn_kept: list = []  # the record: the turns (deg) of the fits refused so
        self.blob_rest_up = False  # ``params["blob_rest_up"]``: a model set on a region stands the way it rests
        self.seen_target = False  # ``params["seen_target"]``: a target known only as an unexplained region is grasped
        self.seen_shrink_m = 0.002  # as it is seen (``seen.py``), each side of the region taken in by this
        # 2026-11-16 night (``params["fallback_stand"]``, a dict of ``STAND`` overrides; off by default): a named
        # target with no percept is stood under the top layer of an unexplained region before the older fallbacks
        # (``_stand_candidates``); after a close on nothing such a place is also the next candidate
        self.stand = None
        # 2026-11-17 (``params["misread_target"]``, a dict of ``MISREAD`` overrides; off by default): a named target
        # with no percept is looked for under a weak percept of another type (``_misread``)
        self.misread = None
        self.behind = None  # ``params["behind_occluder"]`` (its options): a target not in view is looked for behind a
        self.n_behind = 0   # thing that hangs in front of the scene (``add_behind``)
        self.behind_refused = None  # the record: why what hangs there was not taken for a panel (``_occluders``)
        self.cams: dict = {}  # each camera's K and pose in the last frame sent (the robot's own belief of them)
        self.n_flip_kept = 0
        self.opened: list = []  # (percept id, part) the robot itself opened: held open in every later scene
        self.n_redetect = 0
        self.budget_exhausted = False
        self.bridge_s = 0.0
        self.render_s = 0.0
        self.server_s = 0.0
        self._held = None  # (R_rel, t_rel) of the target in the hand frame
        self._release_pending = False  # ``release_here``: the hand let go; the next look with it open is the merged one
        self._refuted: list = []  # ``refute_target``: places (xy) where the hand closed on nothing at a fallback pose
        self.rigid_predict = False  # the agent's ``held_pose_fix``: the in-hand prediction by ``_carried``
        self.n_refuted = 0
        self.n_release_here = 0
        self.n_rehold = 0
        # 2026-11-16: how far the held target will fall when the hand opens (the agent sets it just before a release
        # above a surface, ``AgentParams.drop_predict``); used once, at the look that follows the opening
        self.drop_m = 0.0
        self.n_drop = 0
        # 2026-11-16 (night; the agent's ``drop_rigid``): the fall starts from where the hand has the object now
        # (``_carried``), not from ``_predict``'s pose, which counts the hand's move twice after a refused track
        self.drop_rigid = False
        # 2026-11-16 (the agent's ``held_jump_max_m``; 0: off): while the hand holds the target, a track further than
        # this from the pose the hand's own motion gives it is not taken (the object does not jump inside the hand)
        self.held_jump_max_m = 0.0
        self.n_held_kept = 0
        # 2026-11-16 (night; the agent's ``held_tilt_max_deg``; 0: off): nor a track that tips it this many degrees
        # (its upright axis; a turn about the vertical, a box squaring up between the jaws, is left alone)
        self.held_tilt_max_deg = 0.0
        self.extra = None  # the first full look's unreported evidence (kept, rejected, blobs)
        self.extra_last = None  # the latest full look's (read by ``tall_unexplained`` alone: the scene as it is now)
        self.fallbacks: list = []
        self._pick_xy = None  # the target's measured xy when the hand first closed on it
        self._opening = False  # the agent said it is letting go (``hand_opened``): the next look counts the hand as open
        self._lost_xy = None  # where the hand was when the target left it during a carry (``hand_opened(lost=True)``)
        self.n_told_open = 0
        self.target_missing = False  # the look after such a loss found no pose of the target (the agent looks again)
        self._stood = None  # the target's bodies as they stood when the hand closed on it (``hand_opened``'s last resort)
        self.rejected_used = 0
        fj = lib.robot["finger_joints"]
        q0 = [v["qpos"] for v in fj.values()]
        self._open_gap = lib.jaw.gap_from_qpos(q0[0], q0[1])

    # -- helpers
    def _hand(self):
        p = np.asarray(self.obs["robot0_eef_pos"], float)
        from .grasp import quat_xyzw_to_mat

        return p, quat_xyzw_to_mat(self.obs["robot0_eef_quat"])

    def _closed(self) -> bool:
        """The jaw gap is below the fully open gap (M1's jaw model and the exported open finger qpos)."""
        q = self.obs["robot0_gripper_qpos"]
        gap = self.lib.jaw.gap_from_qpos(q[0], q[1])
        return gap < self._open_gap - CLOSED_MARGIN_M

    def _frame(self, camera: str):
        a = time.perf_counter()
        fr = self.cam(camera, MP.CAMERAS[camera], self.t_now)
        self.render_s += time.perf_counter() - a
        self.cams[camera] = {"K": np.asarray(fr["K"], float), "T": np.asarray(fr["T_world_from_camera"], float)}
        return {k: fr[k] for k in ("camera", "rgb", "depth_u16", "depth_scale_mm", "K", "T_world_from_camera",
                                   "width", "height", "step")}

    def _msg(self, camera: str, look: int | None = None) -> dict:
        return {"suite": self.suite, "episode": self.episode, "look": self.n_full + self.n_track if look is None else look,
                "cameras": [self._frame(camera)],
                "joint_pos": np.asarray(self.obs["robot0_joint_pos"], float),
                "gripper_qpos": np.asarray(self.obs["robot0_gripper_qpos"], float)}

    def place_base(self) -> None:
        """ST-1 H8: the server keeps this look's depth as the scene before anything was placed. Outside the budget."""
        self.client.call({"op": "look", "mode": "place", "base": True, **self._msg("agentview")})

    def place_look(self, region: dict) -> dict:
        """ST-1 H8: what rose inside the container's box since ``place_base``. Outside the budget."""
        rep = self.client.call({"op": "look", "mode": "place", "rise_m": self.place_check["rise_m"],
                                "region": {k: [float(x) for x in region[k]] for k in ("centre", "half")},
                                **self._msg("agentview")})
        self.looks.append({"t": self.t_now, "mode": "place", **rep["place"]})
        return rep["place"]

    def hand_opened(self, lost: bool = False) -> None:
        """The agent is opening the hand after a grasp, a lift or a carry that failed (2026-11-16; called only with
        the agent's ``tell_open``). The fingers need about 13 steps to read fully open, and until then this tool
        counted the target as still held: a look taken sooner (a failed attempt opens for 6 steps and goes on)
        returned the target riding in the empty hand, and the next grasp was planned at the hand, in the air. On the
        colour line's training episodes of 2026-11-15 (449 on tasks the robot has skills for), 40
        held such a grasp right after a failed grasp or lift and 25 of those were lost; after a carry the fingers
        had lost (``carry_watch``), 13 of 13 next grasps were in the air. From here the next look
        takes the hand as open, whatever the gap reads: the merged full look that follows a release. ``lost``: the
        target left the hand during a carry; where the hand is now is where it fell from (the look's rejected poses
        are searched there, as they are at the pick after a failed grasp)."""
        self._opening = True
        self.n_told_open += 1
        if lost and self.obs is not None:
            self._lost_xy = np.asarray(self.obs["robot0_eef_pos"], float)[:2].copy()

    def _rejected_near(self, rep: dict, ref_xy):
        """The server's best rejected percept of the target's type within REJ_NEAR_M (xy) of ``ref_xy``."""
        try:
            tgt = next((x for x in self.percepts if x["id"] == self.target), None)
            if tgt is None:
                return None
            typ = tgt["type"]
            root = self.lib.export_types[typ]["root_body"]
            best, key = None, None
            for q in ((rep.get("extra") or {}).get("rejected") or []):
                if q.get("type") != typ or root not in (q.get("bodies") or {}):
                    continue
                fit = q.get("fit") or {}
                inl = float(fit.get("inlier") or 0.0)
                if inl < 0.5:
                    continue
                pos = np.asarray(q["bodies"][root]["pos"], float)
                if float(np.linalg.norm(pos[:2] - np.asarray(ref_xy, float)[:2])) > REJ_NEAR_M:
                    continue
                k = float(fit.get("n_agree") or 0.0) * inl
                if key is None or k > key:
                    best, key = q, k
            return best
        except Exception:
            return None

    def _blinded(self, rep: dict) -> dict:
        """``self.blind``: the look's reply without the percepts (and rejected poses) of one type near one place."""
        try:
            typ, xy, r = self.blind["type"], np.asarray(self.blind["xy"], float)[:2], float(self.blind.get("within_m", 0.06))
            root = self.lib.export_types[typ]["root_body"]

            def hit(q):
                at = q.get("pos") if q.get("pos") is not None else ((q.get("bodies") or {}).get(root) or {}).get("pos")
                return q.get("type") == typ and at is not None and float(np.linalg.norm(np.asarray(at, float)[:2] - xy)) <= r

            out = {**rep, "percepts": [q for q in rep.get("percepts") or [] if not hit(q)]}
            if isinstance(rep.get("extra"), dict) and rep["extra"].get("rejected") is not None:
                out["extra"] = {**rep["extra"], "rejected": [q for q in rep["extra"]["rejected"] if not hit(q)]}
            return out
        except Exception:
            return rep

    def full_look(self, merge: bool = False, ref_xy=None, free=(), only=None, self_px=None) -> dict:
        """A full look. With ``merge`` the episode's percept list keeps its ids and body names (the agent's scene is
        built once): each percept takes the pose of the new percept of its type nearest to its current pose, within
        ``REASSOC_M``; one that finds none keeps its pose (association, plan perception_pipeline). ``free``: ids
        that, when they find none, take the nearest fresh percept of their type no other percept claimed, up to
        ``RELOOK_MAX_M`` away (a receiver the scene moved). ``only``: the ids whose poses may change at all."""
        a = time.perf_counter()
        rep = self.client.call({"op": "look", "mode": "full", **self._msg("agentview"),
                                **({"self_px": int(self_px)} if self_px else {})})  # (``look_route`` alone sends it)
        self.bridge_s += time.perf_counter() - a
        self.server_s += float(rep.get("server_cpu_s") or 0.0)
        self.n_full += 1
        if isinstance(rep.get("self_filter"), dict):  # record only: what the server's filter took as the robot
            sf = rep["self_filter"]
            self.own_px.append([int(sf.get("robot_px") or 0)] + ([int(sf["picture_px"])] if "picture_px" in sf else [])
                               + ([int(sf["front_px"])] if "front_px" in sf else []))  # (``self_front_m``: left in)
            if "picture_dz" in sf:  # (``self_px_probe``)
                self.own_dz.append(sf["picture_dz"])
        if self.blind:
            rep = self._blinded(rep)
        self.extra_last = rep.get("extra") or {}
        assoc = assoc_all = None
        n_turn0 = len(self.turn_kept)
        if merge and self.percepts:
            # association (amendment A16): every (old, new) pair of one type within REASSOC_M, nearest first, each
            # percept and each new percept used once (the plan's Hungarian match, greedy on distance; A4's form went
            # percept by percept in list order, so an earlier percept could take a later one's nearer match)
            assoc = {}
            fresh = list(rep["percepts"])
            pairs = []
            for i, p in enumerate(self.percepts):
                root = self.lib.export_types[p["type"]]["root_body"]
                here = np.asarray(p["bodies"][root]["pos"], float)
                for j, q in enumerate(fresh):
                    if q["type"] != p["type"]:
                        continue
                    d = float(np.linalg.norm(np.asarray(q["bodies"][root]["pos"]) - here))
                    if d < REASSOC_M:
                        pairs.append((d, i, j))
            used_i, used_j, match = set(), set(), {}
            for d, i, j in sorted(pairs):
                if i in used_i or j in used_j:
                    continue
                used_i.add(i)
                used_j.add(j)
                match[i] = (j, d)
            # the grounded target that found no match within REASSOC_M: its stored pose is the in-hand prediction,
            # which is wrong whenever the object did not come with the hand (a slip, a failed lift). Take the nearest
            # fresh percept of its type that no other percept claimed, at any distance.
            for i, p in enumerate(self.percepts):
                if (p["id"] != self.target and p["id"] not in free) or i in match:
                    continue
                root = self.lib.export_types[p["type"]]["root_body"]
                here = np.asarray(p["bodies"][root]["pos"], float)
                best_j, best_d = None, None
                for j, q in enumerate(fresh):
                    if j in used_j or q["type"] != p["type"] or root not in q.get("bodies", {}):
                        continue
                    d = float(np.linalg.norm(np.asarray(q["bodies"][root]["pos"], float) - here))
                    if best_d is None or d < best_d:
                        best_j, best_d = j, d
                if best_j is not None and (p["id"] == self.target or best_d <= RELOOK_MAX_M):
                    used_i.add(i)
                    used_j.add(best_j)
                    match[i] = (best_j, best_d)
            out = []
            rej_fix = None
            if ref_xy is not None:
                rej_fix = self._rejected_near(rep, ref_xy)
            for i, p in enumerate(self.percepts):
                if i not in match:
                    if p["id"] == self.target and rej_fix is not None and rej_fix.get("type") == p["type"]:
                        # the target did not come with the hand and no accepted percept explains it (knocked over:
                        # the server rejects a tilted pose): take the server's best rejected pose of its type near
                        # where it was picked, instead of the in-hand prediction
                        out.append({**p, "bodies": rej_fix["bodies"], "joints": rej_fix.get("joints") or p.get("joints"),
                                    "fit": dict(rej_fix.get("fit") or {}),
                                    "source": f"relook_rejected:{rej_fix.get('source')}"})
                        assoc[p["id"]] = -1.0
                        self.rejected_used += 1
                        continue
                    out.append(p)
                    assoc[p["id"]] = None
                    continue
                j, d = match[i]
                best = fresh[j]
                if self.no_flip and self._flipped(p, best):
                    # a fixture does not turn around between two looks: the new fit took its mirror pose (2026-11-15:
                    # the cabinet with an object set on its top read back to front in 3 of 10 starts, and the handle
                    # was looked for behind it). The first look, taken with nothing in the way, stands.
                    out.append(p)
                    assoc[p["id"]] = None
                    self.n_flip_kept += 1
                    self.turn_kept.append(self._turn_deg(p, best))
                    continue
                out.append({**p, "bodies": best["bodies"], "joints": best.get("joints"), "fit": best.get("fit"),
                            "score": best.get("score"), "source": f"relook:{best.get('source')}"})
                assoc[p["id"]] = round(d, 4)
            assoc_all = dict(assoc)  # every percept's association, whatever ``only`` lets change
            if only is not None:  # a look for some percepts alone: the others stay as they were
                out = [q if q["id"] in only else p for q, p in zip(out, self.percepts)]
                assoc = {k: v for k, v in assoc.items() if k in only}
            self.percepts = out
        else:
            self.percepts = rep["percepts"]
            if self.extra is None:
                self.extra = rep.get("extra") or {}
        self._hold_open()
        self.looks.append({"t": self.t_now, "mode": "full", "merge": merge, "n": len(self.percepts),
                           "assoc": assoc, "times": rep.get("times")})
        if self.turn_max_deg is not None and len(self.turn_kept) > n_turn0:  # the row mark of ``fixture_turn_max_deg``
            self.looks[-1]["turn_kept"] = list(self.turn_kept[n_turn0:])
        self._assoc_all = assoc_all
        return rep

    def face_in(self) -> int:
        """A fixture's opening parts face the workspace (2026-11-15; off unless the planner turns it on). A box-like
        fixture fits its own mirror pose as well as its true one (the same volume, and the net's depth does not
        show a handle): in 2 of 10 plain starts of the drawer task the colour line read the cabinet back to front
        and the handle was looked for behind it. A cabinet stands with its drawers toward the space the hand works
        in, not toward the table's edge. So a fixture whose first opening part opens away from where the hand
        started is turned half a turn about the middle of its own boxes (which the mirror pose shares). Returns
        how many were turned this call."""
        if not self.face_in_on or self.obs is None or not self.percepts:
            return 0
        from .drawer import _jv, open_far
        from .ground import aabb

        if self._hand0 is None:
            self._hand0 = np.asarray(self.obs["robot0_eef_pos"], float)[:2].copy()
        turn = np.diag([-1.0, -1.0, 1.0])
        out, n = [], 0
        for p in self.percepts:
            tdoc = self.lib.export_types.get(p.get("type")) or {}
            parts = tdoc.get("parts") or {}
            try:
                if tdoc.get("kind") != "fixture" or not parts:
                    raise KeyError
                root = tdoc["root_body"]
                Rr = np.asarray(self.lib.quat_mat(p["bodies"][root]["quat_wxyz"]), float)
                tr = np.asarray(p["bodies"][root]["pos"], float)
                jv = _jv(p)
                part = next(k for k in sorted(parts) if (parts[k].get("joint") or {}).get("range"))
                body = parts[part]["bodies"][0]
                A = part_body_poses(tdoc, {root: (Rr, tr)}, jv)
                B = part_body_poses(tdoc, {root: (Rr, tr)}, {**jv, part: open_far(tdoc, part)})
                d = (np.asarray(B[body][1], float) - np.asarray(A[body][1], float))[:2]
                if float(np.linalg.norm(d)) < 0.02:  # it is open already, or the part hardly travels
                    raise KeyError
                if self.face_in_mode == "carcass":
                    # 2026-11-16: the mirror fit shares the CARCASS with the true pose, not the whole fixture's box
                    # (the handles stand out in front): turned about the box's middle the pose stayed 3.4 cm off
                    # along the fixture's depth, which is what sank this rule on 11-15; turned about the carcass's
                    # middle, seven back-to-front fits of plain starts land 0.6 mm from the truth. No tracker (it
                    # left the pose where it was put).
                    from .drawer import carcass_centre

                    from .drawer import FACE_MIN_COS

                    cc = carcass_centre(self.lib, p)
                    if cc is None:
                        raise KeyError
                    c = np.asarray(cc, float)
                    v = self._hand0 - c
                    if float(d @ v) > -FACE_MIN_COS * float(np.linalg.norm(d)) * float(np.linalg.norm(v)):
                        raise KeyError  # it opens toward the workspace, or across it (which says nothing)
                else:
                    lo, hi = aabb(self.lib, p)
                    c = 0.5 * (np.asarray(lo, float) + np.asarray(hi, float))[:2]
                if float(d @ (self._hand0 - c)) >= 0.0:
                    raise KeyError  # it opens toward the workspace already
                R2, t2 = turn @ Rr, np.array([2.0 * c[0] - tr[0], 2.0 * c[1] - tr[1], tr[2]])
                how = "faced"
                if self.face_in_mode != "carcass" and p.get("gid") is not None \
                        and self.n_full + self.n_track < self.budget:
                    # the mirror fit is free to sit a few centimetres off along the fixture's front-to-back line
                    # (2026-11-15: 3.4 cm on the cabinet, whose top and near side it still explains). Turned the
                    # right way round, the server's tracker settles the pose on what the picture shows; its answer
                    # is taken when it stays near and still faces in.
                    a_ = time.perf_counter()
                    rep = self.client.call({"op": "look", "mode": "track", **self._msg("agentview"), "priors": {
                        p["id"]: {"gid": p["gid"], "R_w": R2.tolist(), "t_w": [float(x) for x in t2]}}})
                    self.bridge_s += time.perf_counter() - a_
                    self.server_s += float(rep.get("server_cpu_s") or 0.0)
                    self.n_track += 1
                    tk = (rep.get("tracks") or {}).get(p["id"]) or {}
                    self.looks.append({"t": self.t_now, "mode": "track", "status": tk.get("status"), "face_in": True,
                                       "n_px": tk.get("n_px"), "jump_m": tk.get("jump_m")})
                    if tk.get("status") == "tracked":
                        R3, t3 = np.asarray(tk["R_w"], float), np.asarray(tk["t_w"], float)
                        d3 = R3 @ (R2.T @ np.array([-d[0], -d[1], 0.0]))  # where the part opens to at the tracked pose
                        if float(np.linalg.norm(t3 - t2)) <= FACE_JUMP_M and float(d3[:2] @ (self._hand0 - c)) > 0.0:
                            R2, t2, how = R3, t3, "faced+tracked"
                bp = part_body_poses(tdoc, {root: (R2, t2)}, jv)
                out.append({**p, "bodies": {b_: {"pos": [float(x) for x in t], "quat_wxyz": mat_to_quat(R)}
                                            for b_, (R, t) in bp.items()},
                            "source": f"{p.get('source')}+{how}"})
                n += 1
            except (KeyError, IndexError, TypeError, ValueError, StopIteration):
                out.append(p)
        if n:
            self.percepts = out
            self.n_faced += n
            self._hold_open()
        return n

    def believe_open(self, pid, part: str) -> None:
        """The robot has just pulled ``part`` of percept ``pid`` open to its end: the scene shows it open from now on,
        whatever a later look makes of it (2026-11-15: a later step plans its release into the opened part)."""
        if (pid, part) not in self.opened:
            self.opened.append((pid, part))
        self._hold_open()

    def _hold_open(self) -> None:
        if not self.opened:
            return
        from .drawer import open_far, open_poses

        out = []
        for p in self.percepts:
            parts = [part for pid, part in self.opened if pid == p["id"]]
            if not parts:
                out.append(p)
                continue
            try:
                tdoc = self.lib.export_types[p["type"]]
                q = {**p, "bodies": dict(p["bodies"]), "joints": dict(p.get("joints") or {})}
                for part in parts:
                    P = open_poses(self.lib, q, part)
                    for b in tdoc["parts"][part]["bodies"]:
                        if b in P:
                            q["bodies"][b] = {"pos": [float(x) for x in P[b][1]], "quat_wxyz": mat_to_quat(P[b][0])}
                    q["joints"][part] = open_far(tdoc, part)
                out.append(q)
            except Exception:
                out.append(p)
        self.percepts = out

    def _carried(self, p: dict) -> dict:
        """The held target where the hand has it now: its root at the hand-relative pose recorded when it was taken
        (or last tracked), every body moved with it. ``_predict``'s own arithmetic assumes the percept still stores
        the poses of that moment, but a refused track stores the PREDICTED pose and leaves the record as it was: from
        then on the hand's move since the record is counted twice (2026-11-16: after a lift whose look was refused,
        the usual answer for a lifted object on the colour line, the target set down again 1 cm above its place was
        stored 11 cm above it, and the next grasp closed on air there; Mac rows: grasps at 104, then 252 mm)."""
        ph, Rh = self._hand()
        R_rel, t_rel = self._held
        root = self.lib.export_types[p["type"]]["root_body"]
        Rb0 = np.asarray(self.lib.quat_mat(p["bodies"][root]["quat_wxyz"]), float)
        tb0 = np.asarray(p["bodies"][root]["pos"], float)
        dR = Rh @ np.asarray(R_rel, float) @ Rb0.T
        tn = Rh @ np.asarray(t_rel, float) + ph
        out = {k: v for k, v in p.items()}
        out["bodies"] = {rel: {"pos": [float(x) for x in dR @ (np.asarray(b["pos"], float) - tb0) + tn],
                               "quat_wxyz": mat_to_quat(dR @ np.asarray(self.lib.quat_mat(b["quat_wxyz"]), float))}
                         for rel, b in p["bodies"].items()}
        return out

    def _predict(self, p: dict) -> dict:
        """The percept's predicted pose: static, or rigid with the hand when it is the held target."""
        if self.target != p["id"] or self._held is None:
            return p
        if self.rigid_predict:
            return self._carried(p)
        ph, Rh = self._hand()
        R_rel, t_rel = self._held
        out = {k: v for k, v in p.items()}
        out["bodies"] = {}
        for rel, b in p["bodies"].items():
            Rb = self.lib.quat_mat(b["quat_wxyz"])
            tb = np.asarray(b["pos"], float)
            # the body's pose in the hand frame was recorded with the root; apply the same hand motion
            dR = Rh @ R_rel @ np.linalg.inv(self._held_R0)
            out["bodies"][rel] = {"pos": [float(x) for x in (Rh @ (R_rel @ (np.linalg.inv(self._held_R0) @
                                  (tb - self._held_t0))) + Rh @ t_rel + ph)],
                                  "quat_wxyz": mat_to_quat(dR @ Rb)}
        return out

    @staticmethod
    def _lowered(p: dict, drop_m: float) -> dict:
        """The percept with every body ``drop_m`` lower (an object that fell straight down)."""
        out = {k: v for k, v in p.items()}
        out["bodies"] = {rel: {"pos": [float(b["pos"][0]), float(b["pos"][1]), float(b["pos"][2]) - float(drop_m)],
                               "quat_wxyz": b["quat_wxyz"]} for rel, b in p["bodies"].items()}
        out["source"] = "dropped"
        return out

    def _set_held(self):
        p = next((x for x in self.percepts if x["id"] == self.target), None)
        if p is None:
            return
        ph, Rh = self._hand()
        root = self.lib.export_types[p["type"]]["root_body"]
        Rb = self.lib.quat_mat(p["bodies"][root]["quat_wxyz"])
        tb = np.asarray(p["bodies"][root]["pos"], float)
        self._held = (Rh.T @ Rb, Rh.T @ (tb - ph))
        self._held_R0, self._held_t0 = Rb, tb

    def release_here(self) -> bool:
        """The hand lets go of the target HERE (the agent calls it as it opens after a grasp that did not carry the
        target away: an empty close, a failed lift). The target's stored pose becomes the in-hand prediction now, and
        the next look with the hand open is the merged full look around that pose. Without this call the prediction
        is taken at that next look, after the open hand has risen to its pre-grasp point: a target that never left
        the table was stored 5 to 10 cm above it (2026-11-16, colour line: the merged look then often found no
        percept that near, and the next grasp closed on air where the hand had been). False when nothing was held."""
        if self._held is None:
            return False
        tgt = next((x for x in self.percepts if x["id"] == self.target), None)
        if tgt is not None:
            self.percepts = [self._carried(tgt) if x["id"] == self.target else x for x in self.percepts]
        self._held = None
        self._release_pending = True
        self.n_release_here += 1
        return True

    def _is_fallback(self, p: dict) -> bool:
        """``p`` is a named type's fallback pose (``on_parsed``: a rejected fit or an unexplained region), not yet
        replaced by a percept a later merged look reported."""
        return (any(f.get("id") == p.get("id") and f.get("role") == "target" for f in self.fallbacks)
                and not str(p.get("source") or "").startswith("relook:"))

    def refute_target(self) -> dict | None:
        """The hand has closed on nothing at the target's pose, and that pose was only a fallback (the server reported
        no percept of the named type; no later look corrects a wrong guess, and a region cannot be tracked at all):
        the object is not there. The target takes the next candidate of its type that does not stand within
        ``REFUTE_M`` of a place already refuted, in ``on_parsed``'s order. It keeps its id and body names. Returns
        what it took, or None when the target is a reported percept or no candidate is left (it then stays).
        2026-11-16: in one plain start of the book task the fallback took a small region 30 cm from the book and the
        hand closed on nothing there five times, while the book's own region was the next candidate."""
        tgt = next((x for x in self.percepts if x["id"] == self.target), None)
        if tgt is None or not self.extra or not self._is_fallback(tgt):
            return None
        typ = tgt["type"]
        root = self.lib.export_types[typ]["root_body"]
        self._refuted.append(np.asarray(tgt["bodies"][root]["pos"], float)[:2].copy())

        def ok(pos) -> bool:
            xy = np.asarray(pos, float)[:2]
            return all(float(np.linalg.norm(xy - r)) > REFUTE_M for r in self._refuted)

        order = ((("rejected", False), ("blob", None), ("rejected", True)) if self.table_bound
                 else (("rejected", None), ("blob", None)))
        new = self._from_stand(typ, set(), allow=ok) if self.stand else None
        for kind, inside in (order if new is None else ()):
            new = (self._from_rejected(typ, inside=inside, allow=ok) if kind == "rejected"
                   else self._from_blob(typ, "target", set(), allow=ok))
            if new is not None:
                break
        if new is None:
            return None
        new = {**new, "id": tgt["id"]}
        self.percepts = [new if x["id"] == self.target else x for x in self.percepts]
        self._held, self._pick_xy = None, None  # it was never in the hand
        self._release_pending = True            # and the next look with the hand open is the merged one
        self.n_refuted += 1
        out = {"source": new["source"], "n": len(self._refuted),
               "moved_m": round(float(np.linalg.norm(np.asarray(new["bodies"][root]["pos"], float)[:2]
                                                     - self._refuted[-1])), 3)}
        self.fallbacks.append({"role": "target", "type": typ, "id": tgt["id"], "source": new["source"],
                               "refuted": out["n"], **({"stand": new.pop("stand")} if "stand" in new else {})})
        return out

    def rehold(self, R_rel, t_rel) -> bool:
        """The target sits in the hand as ``(R_rel, t_rel)`` says (its root body in the hand frame; the agent's pose
        from the look at the close): its stored pose and the hand-relative transform are set from that. The agent
        calls it when its lift check went by the fingers and the look at the lifted object, which read it elsewhere
        in the hand, is not believed. False when the target is not held."""
        tgt = next((x for x in self.percepts if x["id"] == self.target), None)
        if tgt is None or self._held is None:
            return False
        self._held = (np.asarray(R_rel, float), np.asarray(t_rel, float))
        new = {**self._carried(tgt), "source": "held"}
        self.percepts = [new if x["id"] == self.target else x for x in self.percepts]
        self._set_held()
        self.n_rehold += 1
        return True

    def _hand_in_box(self, tgt: dict, ph, margin: float = 0.02) -> bool:
        """The hand centre lies inside the target's world bounding box (grown by ``margin``): a tall object whose
        origin is at its base is held by a part far above that origin (beyond ``held_near_m``)."""
        try:
            from .ground import aabb

            lo, hi = aabb(self.lib, tgt)
            ph = np.asarray(ph, float)
            return bool(np.all(ph >= np.asarray(lo, float) - margin) and np.all(ph <= np.asarray(hi, float) + margin))
        except Exception:
            return False

    def _turn_deg(self, old: dict, new: dict):
        """How far ``new``'s fixture is turned from ``old``'s (deg); None when either is no fixture pose."""
        tdoc = self.lib.export_types.get(old["type"]) or {}
        if tdoc.get("kind") != "fixture":
            return None
        root = tdoc["root_body"]
        try:
            Ra = self.lib.quat_mat(old["bodies"][root]["quat_wxyz"])
            Rb = self.lib.quat_mat(new["bodies"][root]["quat_wxyz"])
        except (KeyError, TypeError):
            return None
        c = float(np.clip((np.trace(np.asarray(Ra, float).T @ np.asarray(Rb, float)) - 1.0) / 2.0, -1.0, 1.0))
        return round(float(np.degrees(np.arccos(c))), 1)

    def _flipped(self, old: dict, new: dict, min_deg: float | None = None) -> bool:
        """``new`` is ``old``'s fixture turned by more than ``min_deg`` (a mirror fit of a box-like fixture; with
        ``turn_max_deg`` set, any turn beyond it: a fixture does not turn between two looks)."""
        if min_deg is None:
            min_deg = 120.0 if self.turn_max_deg is None else float(self.turn_max_deg)
        d = self._turn_deg(old, new)
        return d is not None and d > min_deg

    def on_receiver(self, place_id) -> dict | None:
        """One full look for what stands ON the receiving object (2026-11-16 night; the server's ``extra.on``, sent
        when its ``extra_on_top_m`` is set): the box across of the thing's top layer (``xy``, ``half_xy``, ``yaw`` of
        its long side, ``z``), ``n_px``, ``inside`` and the receiver's own top ``z_obj``; None when no look is left,
        the receiver is not in this look, or nothing is on it. No percept's pose changes: the look is for the thing
        alone. The caller takes the hand out of the camera's way first (what the hand holds would be the thing)."""
        if place_id is None or self.n_full + self.n_track >= self.budget:
            return None
        old = next((x for x in self.percepts if x["id"] == place_id), None)
        if old is None:
            return None
        root = self.lib.export_types[old["type"]]["root_body"]
        here = np.asarray(old["bodies"][root]["pos"], float)
        rep = self.full_look(merge=True, only=())
        self.looks[-1]["on_receiver"] = True
        on = (rep.get("extra") or {}).get("on") or {}
        best, best_d = None, None
        for q in rep.get("percepts") or []:
            if q.get("type") != old["type"] or root not in (q.get("bodies") or {}) or str(q.get("id")) not in on:
                continue
            d = float(np.linalg.norm(np.asarray(q["bodies"][root]["pos"], float) - here))
            if d <= REASSOC_M and (best_d is None or d < best_d):
                best, best_d = on[str(q["id"])], d
        self.looks[-1]["on"] = None if best is None else {k: best.get(k) for k in ("xy", "half_xy", "yaw", "z", "n_px",
                                                                                 "inside")}
        return best

    def relook_place(self, place_id) -> dict | None:
        """One full look just before letting go, for the receiving object alone (2026-11-15: a
        receiver that moved after the first look was never measured again). Its percept takes the fresh percept of
        its type as the merged look associates them, or the nearest unclaimed one up to ``RELOOK_MAX_M`` away. Every
        other percept keeps its pose (the held target's is the hand's). Returns ``{"moved_m", "assoc"}``, or None
        when the look budget is spent or the receiver is not among the percepts."""
        if place_id is None or self.n_full + self.n_track >= self.budget:
            return None
        old = next((x for x in self.percepts if x["id"] == place_id), None)
        if old is None:
            return None
        root = self.lib.export_types[old["type"]]["root_body"]
        before = np.asarray(old["bodies"][root]["pos"], float)
        kept = list(self.percepts)
        rep = self.full_look(merge=True, free=(place_id,), only=(place_id,))
        self.looks[-1]["relook_place"] = True
        new = next(x for x in self.percepts if x["id"] == place_id)
        moved = float(np.linalg.norm(np.asarray(new["bodies"][root]["pos"], float) - before))
        out = {"moved_m": round(moved, 4), "assoc": (self.looks[-1].get("assoc") or {}).get(place_id)}
        # what this look makes of everything else (not the receiver, not the object in the hand): each one's distance
        # from where it was, for those it saw at all. A receiver is believed moved only when the look sees the
        # things that did not move where they were (2026-11-15: after the camera itself moved mid-episode, the
        # look read the receiver 8-14 cm away, the robot followed, and 5 of 10 such episodes were lost).
        others = {k: v for k, v in (getattr(self, "_assoc_all", None) or {}).items()
                  if k not in (place_id, self.target)}
        seen = sorted(float(v) for v in others.values() if v is not None and v >= 0)
        out["others"] = {"n": len(others), "seen_m": [round(v, 3) for v in seen[:6]]}

        def dists(items, pos_of):  # this look's poses of the receiver's type, as distances from where it was
            ds = []
            for q in items or []:
                try:
                    if q.get("type") == old["type"]:
                        ds.append(round(float(np.linalg.norm(np.asarray(pos_of(q), float) - before)), 3))
                except (KeyError, TypeError, ValueError):
                    continue
            return sorted(ds)[:4]

        out["fresh_m"] = dists(rep.get("percepts"), lambda q: q["bodies"][root]["pos"])
        out["rejected_m"] = dists((rep.get("extra") or {}).get("rejected"), lambda q: q["pos"])
        still = bool(out["rejected_m"]) and out["rejected_m"][0] <= STILL_M
        if out["assoc"] is None and still:
            out["still"] = True  # a fit of its type where it was, though not an accepted one: it has not moved
        if out["assoc"] is None and not still:
            # no accepted pose of the receiver's type: a fixture is often only a rejected fit (too little of it in
            # view), and so is its pose after the scene moved it. The best such fit between 2 cm and RELOOK_MAX_M
            # from where it was is taken (nearer is the same place; farther is something else). Not when a fit
            # of its type, accepted or not, is where it was.
            best, key = None, None
            for q in (rep.get("extra") or {}).get("rejected") or []:
                if q.get("type") != old["type"] or not q.get("bodies") or q.get("pos") is None:
                    continue
                d = float(np.linalg.norm(np.asarray(q["pos"], float) - before))
                fit = q.get("fit") or {}
                inl = float(fit.get("inlier") or 0.0)
                if not 0.02 < d <= RELOOK_MAX_M or inl < 0.5 or root not in q["bodies"]:
                    continue
                k = float(fit.get("n_agree") or 0.0) * inl
                if key is None or k > key:
                    best, key = q, k
            if best is not None and not (self.no_flip and self._flipped(old, best)):
                self.percepts = [({**x, "bodies": best["bodies"], "joints": best.get("joints") or x.get("joints"),
                                   "fit": dict(best.get("fit") or {}), "source": f"relook_rejected:{best.get('source')}"}
                                  if x["id"] == place_id else x) for x in self.percepts]
                self._hold_open()
                new = next(x for x in self.percepts if x["id"] == place_id)
                moved = float(np.linalg.norm(np.asarray(new["bodies"][root]["pos"], float) - before))
                out.update(moved_m=round(moved, 4), assoc=-1.0)
        if self.relook_strict and out["moved_m"] > STILL_M:
            new = next(x for x in self.percepts if x["id"] == place_id)
            why = self._move_refused(old, new, root, kept)
            out["dz_m"] = round(float(new["bodies"][root]["pos"][2]) - float(before[2]), 4)
            if why:  # not a move a receiver can make: the look read something else as it
                self.percepts = kept
                self._hold_open()
                out.update(unconfirmed_m=out["moved_m"], moved_m=0.0, refused=why)
        if out["moved_m"] > STILL_M and others and self.static_check:
            ok = bool(seen) and float(np.median(seen)) <= STILL_M
            if not ok:  # the look does not show the rest of the scene where it was: it is not believed
                self.percepts = kept
                self._hold_open()
                out.update(unconfirmed_m=out["moved_m"], moved_m=0.0)
        return out

    def _move_refused(self, old: dict, new: dict, root: str, percepts: list) -> str | None:
        """``relook_strict``: why a receiver's believed move cannot be one, or None. A receiver the scene moves goes
        across the surface it stands on (the measured jumps: 7.5 and 15 cm, level), so
        * a new place more than ``RELOOK_DZ_M`` higher or lower is another surface read as it;
        * a new place within the footprint of another known object, where the old place was not, is that object read
          as it: two things are not in one place. (2026-11-16, truth replays of two colour-line episodes: with the
          plate hidden under the loaded hand, a rejected plate fit on the second bowl 12 cm away was taken for the
          plate, and the bowl was set down on that bowl.)"""
        from .ground import in_footprint

        a = np.asarray(old["bodies"][root]["pos"], float)
        b = np.asarray(new["bodies"][root]["pos"], float)
        if abs(float(b[2] - a[2])) > RELOOK_DZ_M:
            return "height"
        for p in percepts:
            if p.get("id") in (old.get("id"), self.target) or not p.get("bodies"):
                continue
            try:
                if in_footprint(self.lib, p, b[:2]) and not in_footprint(self.lib, p, a[:2]):
                    return f"on:{p['id']}"
            except Exception:
                continue
        return None

    def remember(self) -> dict | None:
        """``params["keep_target"]``: what is known of the grounded target, to carry over to a later plan of the same
        instruction: its percept as it is believed now (in the hand: where the hand has it), and where the other
        objects of its type stand. None with no target."""
        tgt = next((x for x in self.percepts if x["id"] == self.target), None)
        if tgt is None or tgt.get("type") not in self.lib.export_types:
            return None
        root = self.lib.export_types[tgt["type"]]["root_body"]
        try:
            now = self._predict(tgt) if self.obs is not None else tgt
        except Exception:
            now = tgt
        others = [[float(x) for x in q["bodies"][root]["pos"]] for q in self.percepts
                  if q.get("type") == tgt["type"] and q["id"] != tgt["id"] and root in (q.get("bodies") or {})]
        return {"type": tgt["type"], "percept": {**now, "bodies": {r: dict(b) for r, b in now["bodies"].items()}},
                "pos": [float(x) for x in now["bodies"][root]["pos"]], "others": others,
                "held": self._held is not None, "t": int(self.t_now)}

    def target_measured(self, since: int) -> bool:
        """The agent's ``c3_measured``: whether a look taken since ``self.looks[since]`` measured the target. A merged
        full look that found a pose for it measures it (a fresh percept of its type, or the rejected fit taken near
        where it was picked), and so does a track the server accepted. A refused track, a percept the server cannot
        track, a track not believed while the hand holds the object, and a look for the receiver alone do not: after
        those the stored pose is still the prediction it was."""
        for l in self.looks[int(since):]:
            if l.get("mode") == "full" and l.get("merge") and (l.get("assoc") or {}).get(self.target) is not None:
                return True
            if l.get("mode") == "track" and l.get("status") == "tracked" and not l.get("held_kept") and not (
                    l.get("held_off_m") is not None and self.held_jump_max_m > 0.0
                    and float(l["held_off_m"]) > self.held_jump_max_m):
                return True
        return False

    def recheck_target(self) -> bool:
        """The agent's ``c3_measured``: one merged full look for the target alone, at any distance from its stored
        pose (the merge's own rule for the grounded target: the nearest fresh percept of its type that no other
        percept claims); every other percept keeps its pose. True when the look found one. False without a look
        when the budget is spent."""
        if self.target is None or self.n_full + self.n_track >= self.budget:
            return False
        self.full_look(merge=True, only=(self.target,))
        self.looks[-1]["recheck_target"] = True
        return (self.looks[-1].get("assoc") or {}).get(self.target) is not None

    def track_look(self):
        """Track the grounded target; the other percepts keep their last pose (static)."""
        tgt = next((x for x in self.percepts if x["id"] == self.target), None)
        if tgt is None:
            return
        # (a hand that is letting go holds nothing: ``release_here``'s notice, and ``hand_opened``'s)
        if self._closed() and self._held is None and not self._release_pending and not self._opening:
            ph, _Rh = self._hand()
            root = self.lib.export_types[tgt["type"]]["root_body"]
            if (float(np.linalg.norm(np.asarray(tgt["bodies"][root]["pos"]) - ph)) < self.held_near_m
                    or self._hand_in_box(tgt, ph)):
                self._pick_xy = np.asarray(tgt["bodies"][root]["pos"], float)[:2].copy()
                # (kept from the start of the hold only: a later track of the lifted object re-anchors the hold,
                # and that pose is in the air)
                self._stood = {r: dict(b) for r, b in tgt["bodies"].items()}
                self._set_held()
        elif not self._closed():
            self._held = None
        pred = self._predict(tgt)
        root = self.lib.export_types[pred["type"]]["root_body"]
        if pred.get("gid") is None:
            # a percept built from an unexplained blob: the server cannot track it; keep the prediction
            self.n_predicted += 1
            self.percepts = [pred if x["id"] == pred["id"] else x for x in self.percepts]
            self.looks.append({"t": self.t_now, "mode": "track", "status": "untrackable"})
            return
        prior = {pred["id"]: {"gid": pred["gid"], "R_w": self.lib.quat_mat(pred["bodies"][root]["quat_wxyz"]).tolist(),
                              "t_w": list(pred["bodies"][root]["pos"])}}
        a = time.perf_counter()
        rep = self.client.call({"op": "look", "mode": "track", "priors": prior, **self._msg("agentview")})
        self.bridge_s += time.perf_counter() - a
        self.server_s += float(rep.get("server_cpu_s") or 0.0)
        self.n_track += 1
        tr = rep["tracks"][pred["id"]]
        new = pred
        held_off = held_turn = held_tilt = None
        if tr["status"] == "tracked" and self._held is not None and self.held_jump_max_m > 0.0:
            held_off = float(np.linalg.norm(np.asarray(tr["t_w"], float) - np.asarray(pred["bodies"][root]["pos"], float)))
        if tr["status"] == "tracked" and self._held is not None:  # (recorded whenever the hand holds it)
            try:  # how far the track turns it from the pose the hand's motion gives it, and how far it tips it
                dR = np.asarray(tr["R_w"], float) @ np.asarray(self.lib.quat_mat(pred["bodies"][root]["quat_wxyz"]), float).T
                held_turn = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))))
                held_tilt = float(np.degrees(np.arccos(np.clip(dR[2, 2], -1.0, 1.0))))
            except Exception:
                held_turn = held_tilt = None
        turned = held_tilt is not None and self.held_tilt_max_deg > 0.0 and held_tilt > self.held_tilt_max_deg
        stood_off = None
        if tr["status"] == "tracked" and self.stand and self._held is None \
                and str(pred.get("source") or "") in ("blob_stand", "predicted") \
                and any(f.get("id") == pred["id"] and "stand" in f for f in self.fallbacks):
            # ``fallback_stand`` with ``track``: a stood model's first track; how far across it moves the model
            stood_off = float(np.linalg.norm(np.asarray(tr["t_w"], float)[:2]
                                             - np.asarray(pred["bodies"][root]["pos"], float)[:2]))
        if (held_off is not None and held_off > self.held_jump_max_m) or turned:
            # in the hand, and the track puts it elsewhere than the hand's motion does: the prediction stands
            self.n_held_kept += 1
            new = {**pred, "source": "held"}
        elif stood_off is not None and stood_off > float(self.stand.get("track_max_m", 0.02)):
            # the track pulls a stood model away from where the region's outline put it (a depth read too far
            # lies behind the thing): the stood place stands
            self.n_predicted += 1
            new = pred
        elif tr["status"] == "tracked":
            tdoc = self.lib.export_types[pred["type"]]
            bp = part_body_poses(tdoc, {tdoc["root_body"]: (np.asarray(tr["R_w"], float),
                                                            np.asarray(tr["t_w"], float))},
                                 {jp: v["q"] for jp, v in (pred.get("joints") or {}).items()})
            new = {**pred, "bodies": {r: {"pos": [float(x) for x in t], "quat_wxyz": mat_to_quat(R)}
                                      for r, (R, t) in bp.items()}, "source": "track"}
            if self._held is not None:  # re-anchor the hand-relative transform to the tracked pose
                self.percepts = [new if x["id"] == pred["id"] else x for x in self.percepts]
                self._set_held()
        else:
            self.n_predicted += 1
            new = {**pred, "source": tr["status"]}
        self.percepts = [new if x["id"] == pred["id"] else x for x in self.percepts]
        self.looks.append({"t": self.t_now, "mode": "track", "status": tr["status"], "n_px": tr.get("n_px"),
                           "jump_m": tr.get("jump_m"), "times": rep.get("times"),
                           **({"held_off_m": round(held_off, 4)} if held_off is not None else {}),
                           **({"held_turn_deg": round(held_turn, 1), "held_tilt_deg": round(held_tilt, 1)}
                              if held_turn is not None else {}),
                           **({"held_kept": True} if turned else {}),
                           **({"stood_off_m": round(stood_off, 4)} if stood_off is not None else {})})

    def close_look(self, pid: str, camera: str = WRIST, min_move_m: float = 0.0) -> dict:
        """Item 3 (2026-11-14, off unless the agent asks): track one percept in a frame of ``camera`` (the wrist camera,
        close up) from its predicted pose. Outside the look budget, with look indices of its own (``CLOSE_LOOK_BASE`` on),
        so the other looks keep theirs. Returns the record, with the root body's ``move`` (m) and ``turn_deg`` from the
        predicted pose when tracked; the percept takes the tracked pose only when the move is at least ``min_move_m``
        (``updated``). So a close look that changes nothing leaves the episode exactly as without it. A camera tool that
        cannot give this camera's depth and pose sets ``wrist_ok = False`` (the colour line)."""
        rec = {"t": self.t_now, "mode": "close", "camera": camera, "id": pid}
        p = next((x for x in self.percepts if x["id"] == pid), None)
        if camera == WRIST and getattr(self.cam, "wrist_ok", True) is False:
            rec["status"] = "no_wrist"
        elif p is None:
            rec["status"] = "no_percept"
        elif p.get("gid") is None:
            rec["status"] = "untrackable"  # a percept built from an unexplained blob
        if "status" in rec:
            self.looks.append(rec)
            return rec
        pred = self._predict(p)
        root = self.lib.export_types[pred["type"]]["root_body"]
        R0 = self.lib.quat_mat(pred["bodies"][root]["quat_wxyz"])
        t0 = np.asarray(pred["bodies"][root]["pos"], float)
        prior = {pid: {"gid": pred["gid"], "R_w": np.asarray(R0).tolist(), "t_w": [float(x) for x in t0]}}
        a = time.perf_counter()
        rep = self.client.call({"op": "look", "mode": "track", "priors": prior,
                                **self._msg(camera, look=CLOSE_LOOK_BASE + self.n_close)})
        self.bridge_s += time.perf_counter() - a
        self.server_s += float(rep.get("server_cpu_s") or 0.0)
        self.n_close += 1
        tr = rep["tracks"][pid]
        rec.update(status=tr["status"], n_px=tr.get("n_px"), jump_m=tr.get("jump_m"), times=rep.get("times"))
        if tr["status"] == "tracked":
            R1, t1 = np.asarray(tr["R_w"], float), np.asarray(tr["t_w"], float)
            dR = R1 @ np.asarray(R0, float).T
            rec["move"] = [round(float(x), 5) for x in (t1 - t0)]
            rec["turn_deg"] = round(float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))), 2)
            rec["updated"] = bool(np.linalg.norm(t1 - t0) >= float(min_move_m))
            if rec["updated"]:
                tdoc = self.lib.export_types[pred["type"]]
                bp = part_body_poses(tdoc, {tdoc["root_body"]: (R1, t1)},
                                     {jp: v["q"] for jp, v in (pred.get("joints") or {}).items()})
                new = {**pred, "bodies": {r: {"pos": [float(x) for x in t], "quat_wxyz": mat_to_quat(R)}
                                          for r, (R, t) in bp.items()}, "source": "close"}
                self.percepts = [new if x["id"] == pid else x for x in self.percepts]
                if self._held is not None and pid == self.target:
                    self._set_held()
        self.looks.append(rec)
        return rec

    # -- fallbacks for named types with no reported percept (the first full look's ``extra``)
    def _build(self, typ: str, R, t) -> dict:
        tdoc = self.lib.export_types[typ]
        bp = part_body_poses(tdoc, {tdoc["root_body"]: (np.asarray(R, float), np.asarray(t, float))}, {})
        return {r: {"pos": [float(x) for x in tt], "quat_wxyz": mat_to_quat(RR)} for r, (RR, tt) in bp.items()}

    def look_route(self, cap: int = 8, self_px: int = 0) -> dict | None:
        """2026-11-17 (stream PATH; the agent's ``look_cross``, ``look_stall`` and ``detour_seen``, off by default): one
        full look for the hand's way alone. No percept changes (``only=()``): the look replaces the latest look's
        unnamed regions (``extra_last``) and keeps them as the way's own (``extra_route``, which
        ``tall_unexplained`` reads with ``route``). Outside the look budget (the looks the robot plans and measures
        with keep their room), at most ``cap`` an episode. ``self_px`` (2026-11-18): the server leaves the robot
        out in the picture too, its model's samples grown by so many pixels. Returns the look's count, or None
        when the cap is spent or there is nothing to look with. ``shifted`` in the reply (the median): more than
        half of the things that stand still, two at least, read as moved by over ``ROUTE_STILL_M``: the camera
        has moved or the look is off, and its regions are not kept."""
        n = int(getattr(self, "n_route", 0))
        if n >= int(cap) or self.obs is None or not self.percepts:
            return None
        before = self.extra_last
        self.full_look(merge=True, only=(), self_px=int(self_px) if self_px else None)
        self.n_route = n + 1
        self.budget += 1
        self.looks[-1]["route"] = True
        out = {"n": self.n_route}
        still = sorted(float(v) for k, v in (getattr(self, "_assoc_all", None) or {}).items()
                       if k != self.target and v is not None and v >= 0)
        if len(still) >= 2 and 2 * sum(v > ROUTE_STILL_M for v in still) > len(still):
            out["shifted"] = round(still[len(still) // 2], 3)
            self.looks[-1]["shifted"] = out["shifted"]
            self.extra_last = before
        else:
            self.extra_route = self.extra_last
        return out

    def tall_unexplained(self, min_h_m: float = 0.08, clear_xy=(), clear_seg=None, held_ids=(),
                         long_max_m: float = 0.30, route: bool = False) -> list:
        """``AgentParams.transit_unexplained`` (2026-11-16, off by default): what the looks saw standing tall and could
        not name, as boxes ``(centre, R, half)`` for a carry. For each unexplained region whose top lies
        ``min_h_m`` or more above the support and that carries a top layer (``extra_blob_top_m``): the top layer's
        own footprint, from the region's lowest seen point up to its top. The region's own box is not used: it is
        the patch of table the thing hides from the camera, and on the colour line it runs on along the camera's
        rays and into the robot's arm (172 colour first looks: the top layers are 2 to 27 cm long, a slab's as
        long as the slab; 96 of 100 regions are over twice their top layer's length, up to 1.4 m). Left out:

        * a thin slice that floats (seen over less than 6 cm of height, 20 cm and more above the support): the cut
          of something taller than the look reads (the arm above the hand, a cabinet's top edge);
        * a region whose top layer lies on a reported percept's footprint and no more than 5 cm above its top (that
          percept's own top); ``held_ids``: the percepts in the hand, whose footprints do not count;
        * a top layer longer than ``2 * long_max_m`` (no thing a hand crosses at one place);
        * one whose top layer, grown by 3 cm, holds a point of ``clear_xy`` (where the carried object stood: a model
          set on a region is that region);
        * one whose top layer's middle lies within ``clear_seg``'s radius of its segment ``(a, b, radius)`` (the
          robot's own arm, from its shoulder to its wrist, in a look taken with the hand in the scene).

        The regions of the first look and of the latest one are both read (a merged look does not replace
        ``self.extra``); one thing seen by both counts once."""
        from .ground import aabb, in_footprint

        out = []
        try:
            blobs = list((self.extra or {}).get("blobs") or [])
            if self.extra_last is not None and self.extra_last is not self.extra:
                blobs += list(self.extra_last.get("blobs") or [])
            if route and getattr(self, "extra_route", None) is not None:
                # (2026-11-18, the agent's looks for the way) the latest such look alone: the scene as it is now,
                # the robot left out of it in the picture; no older look's things
                blobs = list(self.extra_route.get("blobs") or [])
            for b in blobs:
                top = b.get("top")
                if not top or int(top.get("n_px") or 0) < 300 or float(b.get("height") or 0.0) < float(min_h_m):
                    continue
                xy = np.asarray(top["xy"], float)[:2]
                if any(float(np.linalg.norm(xy - c[:2])) <= 0.05 for c, _R, _h in out):
                    continue
                z1 = float(b["z_top"])
                h_min = max(0.0, float(b.get("h_min") or 0.0))
                z0 = z1 - float(b["height"]) + h_min
                if z1 - z0 < 0.01 or (z1 - z0 < 0.06 and h_min >= 0.20):
                    continue
                own = False
                for q in self.percepts:
                    if not q.get("bodies") or q.get("id") in held_ids or not in_footprint(self.lib, q, xy, 0.02):
                        continue
                    if z1 <= float(aabb(self.lib, q)[1][2]) + 0.05:
                        own = True
                        break
                if own:
                    continue
                c, s_ = math.cos(float(top.get("yaw") or 0.0)), math.sin(float(top.get("yaw") or 0.0))
                R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
                half = np.array([max(0.01, float(top["half_xy"][0])), max(0.01, float(top["half_xy"][1])), 0.5 * (z1 - z0)])
                if float(max(half[0], half[1])) > float(long_max_m):
                    continue
                if any(bool(np.all(np.abs(R[:2, :2].T @ (np.asarray(k, float)[:2] - xy)) <= half[:2] + 0.03))
                       for k in clear_xy):
                    continue
                if clear_seg is not None:
                    a, b_, rad = np.asarray(clear_seg[0], float)[:2], np.asarray(clear_seg[1], float)[:2], float(clear_seg[2])
                    ab = b_ - a
                    f = float(np.clip((xy - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
                    if float(np.linalg.norm(xy - (a + f * ab))) <= rad:
                        continue
                out.append((np.array([xy[0], xy[1], 0.5 * (z0 + z1)]), R, half))
        except Exception:
            return []
        return out

    def _over_table(self, pos) -> bool:
        """Whether a world point lies over the workcell's table (its first box, by world extents, 5 cm margin); True
        when the workcell has no table box."""
        for a in (self.lib.workcell or {}).get("arena") or []:
            if a.get("type") == "box":
                c = np.asarray(a["centre"], float)
                e = np.abs(np.asarray(a.get("R", np.eye(3)), float)) @ np.asarray(a["half"], float)
                return bool(np.all(np.abs(np.asarray(pos, float)[:2] - c[:2]) <= e[:2] + 0.05))
        return True

    def _table_top(self, xy, below: float):
        """The highest top of a workcell box under ``xy`` that lies under ``below`` (the surface something there
        stands on), or None."""
        best = None
        for a in (self.lib.workcell or {}).get("arena") or []:
            if a.get("type") != "box":
                continue
            c = np.asarray(a["centre"], float)
            e = np.abs(np.asarray(a.get("R", np.eye(3)), float)) @ np.asarray(a["half"], float)
            top = float(c[2] + e[2])
            if bool(np.all(np.abs(np.asarray(xy, float)[:2] - c[:2]) <= e[:2])) and top < float(below) \
                    and (best is None or top > best):
                best = top
        return best

    def _inside_other(self, typ: str, pos, skip=(), margin: float = 0.0) -> bool:
        """Whether a world point lies within the footprint (grown by ``margin``) of a reported percept of another
        type (types in ``skip`` do not count)."""
        from .ground import aabb

        xy = np.asarray(pos, float)[:2]
        for p in self.percepts:
            if p.get("type") == typ or p.get("type") in skip or not p.get("bodies"):
                continue
            try:
                lo, hi = aabb(self.lib, p)
            except Exception:
                continue
            if bool(np.all(xy >= np.asarray(lo, float)[:2] - margin) and np.all(xy <= np.asarray(hi, float)[:2] + margin)):
                return True
        return False

    def _from_memory(self, typ: str):
        """``fixture_memory``: the named fixture at its REMEMBERED place (where the robot's own accepted first looks
        of this workcell found the type; the same numbers the server's ``memory`` search starts from), when the look
        reported no percept of it and an unexplained region's top layer lies within ``MEMORY_NEAR_M`` of the
        remembered carcass's middle and no more than 8 cm under the model's top. A top more than ``MEMORY_LIFT_M``
        above the model's is the fixture standing that much higher (S03: the cabinet on a 10 cm block, at its plain
        place and turn; the region's top read 1.2287 for a true 1.228). 2026-11-16, truth replays of four training
        episodes: under S03, S07, S08 and S09 the cabinet stood at its plain place and turn, no fit was accepted and
        the robot reached for rejected fits and blobs 10 to 40 cm off; a fixture the scene has moved shows its
        region elsewhere and is left to the older fallbacks. None when the look does not support it."""
        mem = (self.fixture_memory or {}).get(typ)
        tdoc = self.lib.export_types.get(typ) or {}
        if not mem or tdoc.get("kind") != "fixture":
            return None
        from .drawer import carcass_centre
        from .ground import aabb

        root = tdoc["root_body"]
        yaw = math.radians(float(mem["yaw_deg"]))
        R = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        t = np.asarray(mem["pos"], float)

        def at(tt):
            bp = part_body_poses(tdoc, {root: (R, np.asarray(tt, float))}, {})
            return {"type": typ, "gid": None, "joints": {}, "fit": {"n_agree": 0, "inlier": 0.0},
                    "bodies": {b_: {"pos": [float(x) for x in p_], "quat_wxyz": mat_to_quat(R_)}
                               for b_, (R_, p_) in bp.items()}}

        p = at(t)
        cc = carcass_centre(self.lib, p)
        if cc is None:
            return None
        z_top = float(np.asarray(aabb(self.lib, p)[1], float)[2])
        best, bd = None, None
        for b in (self.extra or {}).get("blobs") or []:
            top = b.get("top") or {}
            if top.get("xy") is None or top.get("z") is None or float(top["z"]) < z_top - 0.08:
                continue
            d = float(np.linalg.norm(np.asarray(top["xy"], float)[:2] - np.asarray(cc, float)))
            if d <= MEMORY_NEAR_M and (bd is None or d < bd):
                best, bd = b, d
        if best is None:
            return None
        dz = float(best["top"]["z"]) - z_top
        how = "memory"
        size = None
        if mem.get("size") is not None:
            # 2026-11-17 (the entry's ``size``, off unless given): the region's top at another height than the model's
            # is the fixture at another size, when it is not far enough up to be a block under it. Colour rows of the
            # opening task and of the two that set a thing on the cabinet (start state 0 and 8, four seeds): the top
            # read 1.179 to 1.187 under the case that draws it taller (1.22 to 1.25 of the model's height; taken for
            # a 5 cm block, the handle was reached for 2.6 cm high and the fingers met on nothing in every attempt),
            # 1.083 to 1.089 under the one that draws it smaller (0.79 to 0.81; reached for 2.2 cm high), 1.226 to
            # 1.233 on the 10 cm block (lifted, as before). Its long side read the model's own to 4% under the taller
            # case and on the block, so the thing drawn taller is as wide as stored; under the smaller case part of
            # the top was out of view (0.67 of the model's), and it is taken no narrower than it is lower.
            sz = {**MEMORY_SIZE, **(mem.get("size") or {})}
            if not abs(dz) <= float(sz["same_m"]) and dz < float(sz["lift_m"]):
                try:
                    from . import sized as SZ

                    lo_m, hi_m = (np.asarray(x, float) for x in aabb(self.lib, p))
                    s_z = (float(best["top"]["z"]) - float(t[2])) / max(z_top - float(t[2]), 1e-3)
                    r_x = max(float(x) for x in best["top"]["half_xy"]) / max(0.5 * float(max(hi_m[0] - lo_m[0], hi_m[1] - lo_m[1])), 1e-3)
                    s_xy = 1.0 if abs(r_x - 1.0) <= float(sz["snap"]) else (r_x if r_x > 1.0 else max(r_x, min(1.0, s_z)))
                    if all(float(sz["lo"]) <= v <= float(sz["hi"]) for v in (s_xy, s_z)):
                        size = SZ.resize_type(self.lib, typ, (s_xy, s_xy, s_z))
                        tdoc = self.lib.export_types[typ]
                        p, how = at(t), "memory+sized"
                        size = {"s": size, "top_ratio": round(s_z, 3), "long_ratio": round(r_x, 3)}
                except Exception:  # the stored size stands whatever goes wrong here
                    size = None
        if size is not None:
            pass
        elif dz > MEMORY_LIFT_M:
            p, how = at(t + np.array([0.0, 0.0, dz])), "memory+lifted"
        elif mem.get("fit_first_m") is not None:
            # 2026-11-17 (the entry's ``fit_first_m``, off unless given): a rejected fit that AGREES with the
            # remembered place (its root within that distance across and ``fit_first_deg``, 15, of the remembered
            # turn) is a fit of the fixture as it stands in this scene, and it is nearer than the memory, which only
            # knows the place to the centimetre or two by which the benchmark sets a fixture down anew. The plain
            # drawer test task at the scored seeds, 30 episodes: the one episode where the remembered place took a
            # rejected fit's place (1.7 cm apart) had its pull slip, and was solved from the rejected fit.
            try:
                r = self._from_rejected(typ)
                if r is not None and root in (r.get("bodies") or {}):
                    rp = np.asarray(r["bodies"][root]["pos"], float)
                    Rr = np.asarray(self.lib.quat_mat(r["bodies"][root]["quat_wxyz"]), float)
                    turn = abs((math.degrees(math.atan2(Rr[1, 0], Rr[0, 0])) - float(mem["yaw_deg"]) + 180.0) % 360.0 - 180.0)
                    if float(np.linalg.norm(rp[:2] - t[:2])) <= float(mem["fit_first_m"]) \
                            and turn <= float(mem.get("fit_first_deg", 15.0)):
                        return None  # the older fallbacks take that fit
            except Exception:
                pass
        p["source"] = how
        p["memory"] = {"top_off_m": round(bd, 4), "dz_m": round(dz, 4), **({"size": size} if size is not None else {})}
        return p

    def _from_rejected(self, typ: str, inside: bool | None = None, allow=None):
        """The best rejected pose of ``typ``. ``inside`` (long-task suite): False, only poses outside every other
        reported object's footprint; True, only poses inside one (another object's surface read as this type:
        2026-11-15, the book read onto the caddy in 2 of 10 starts, and no grasp was free of the caddy)."""
        best, key = None, None
        for q in (self.extra or {}).get("rejected") or []:
            if q.get("type") != typ or not q.get("bodies"):
                continue
            if self.table_bound and q.get("pos") is not None and not self._over_table(q["pos"]):
                continue  # long-task suite: a fallback off the table is a false fit, never the object
            if inside is not None and q.get("pos") is not None and self._inside_other(typ, q["pos"]) != inside:
                continue
            if allow is not None and q.get("pos") is not None and not allow(q["pos"]):
                continue  # ``refute_target``: the hand has closed on nothing there
            fit = q.get("fit") or {}
            inl = float(fit.get("inlier") or 0.0)
            if inl < 0.5:
                continue
            k = float(fit.get("n_agree") or 0.0) * inl
            if key is None or k > key:
                best, key = q, k
        if best is None:
            return None
        p = {k: v for k, v in best.items() if k not in ("why", "pos")}
        p["fit"] = dict(best.get("fit") or {})
        p.setdefault("joints", {})
        p["source"] = f"rejected:{best.get('source')}"
        return p

    def _from_blob(self, typ: str, role: str, used: set, allow=None, inside: bool | None = None):
        """The unexplained blob that best fits ``typ``'s size. ``inside`` (``blob_clear_first``): False, only blobs
        whose centre is clear of every other reported object's footprint (grown by ``BLOB_NEAR_M``); True, only blobs
        inside one. 2026-11-16, colour line, one plain start of the book task: the book had no fit, and of the
        blobs the fallback took a 448-pixel leftover beside the mug (its size fitted the book's better) instead of
        the 11,761-pixel one that was the book; five grasps closed on nothing."""
        from .ground import aabb

        blobs = (self.extra or {}).get("blobs") or []
        R0 = np.eye(3)
        if self.blob_rest_up:
            # 2026-11-16, off unless ``params["blob_rest_up"]``: the model is stood the way it rests. Six of the
            # Object suite's types rest with their body's y axis up; set on a region with their z up they lie on
            # their side (the smoke's first row: such a fallback planned "not in its rest orientation")
            from .behind import rest_R

            R0 = rest_R(self.lib, typ)
        lo, hi = aabb(self.lib, {"type": typ, "bodies": self._build(typ, R0, np.zeros(3))})
        mh = 0.5 * (hi - lo)
        m_long_x = mh[0] >= mh[1]
        m_sorted = sorted([float(mh[0]), float(mh[1])], reverse=True)
        best, bcost = None, None
        for k, b in enumerate(blobs):
            if k in used:
                continue
            if int(b.get("n_px") or 0) < 300 or float(b.get("height") or 0.0) <= 0.01:
                continue
            if allow is not None and not allow(b["xy"]):
                continue  # ``refute_target``: the hand has closed on nothing there
            if inside is not None and self._inside_other(typ, b["xy"], margin=BLOB_NEAR_M) != inside:
                continue
            bh = [float(x) for x in b["half_xy"]]
            b_sorted = sorted(bh, reverse=True)
            ok = True
            cost = 0.0
            for m, s in zip(m_sorted, b_sorted):
                r = max(s, 1e-4) / max(m, 1e-4)
                if r < 0.25 or r > 4.0:
                    ok = False
                cost += abs(np.log(r))
            if not ok:
                continue
            cost += abs(np.log(max(float(b["height"]), 1e-3) / max(float(2 * mh[2]), 1e-3)))
            if bcost is None or cost < bcost:
                best, bcost = k, cost
        if best is None:
            return None
        b = blobs[best]
        if role == "target" and self.seen_target:
            # 2026-11-16, off unless ``params["seen_target"]``: the region is taken as it is seen (``seen.py``): a box
            # of its own size as the model the hand grasps, named as the type the instruction asks for. For a thing
            # the stored model does not explain (another make of it, a reshaped one) the stored sizes are the wrong
            # ones to close the fingers on. Kept only when the seen box can be grasped at all.
            from . import seen as SN

            sp = SN.seen_percept(self.lib, b, "seen", shrink_m=float(self.seen_shrink_m), named=typ)
            if SN.n_grasps(self.lib, sp["type"]) > 0:
                used.add(best)
                return {k: v for k, v in sp.items() if k != "id"}
        bh = [float(x) for x in b["half_xy"]]
        yaw = float(b["yaw"]) + (0.0 if bh[0] >= bh[1] else 0.5 * np.pi)  # the blob's long axis
        yaw -= 0.0 if m_long_x else 0.5 * np.pi  # the model's long axis onto it
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R0
        lo, hi = aabb(self.lib, {"type": typ, "bodies": self._build(typ, R, np.zeros(3))})
        table_z = float(b["z_top"]) - float(b["height"])
        if self.blob_on_table:  # not inside the table: the table top under the blob, when the blob's base is lower
            top = self._table_top(b["xy"], float(b["z_top"]) - 0.005)
            if top is not None:
                table_z = max(table_z, top)
        z = table_z - float(lo[2])
        if role == "place":
            z = max(z, float(b["z_top"]) - float(hi[2]))
        t = np.array([float(b["xy"][0]) - 0.5 * float(lo[0] + hi[0]),
                      float(b["xy"][1]) - 0.5 * float(lo[1] + hi[1]), z])
        used.add(best)
        return {"type": typ, "gid": None, "score": 0.0, "n_px": int(b.get("n_px") or 0),
                "fit": {"n_agree": 0, "inlier": 0.0}, "source": "blob", "joints": {},
                "bodies": self._build(typ, R, t)}

    # -- 2026-11-16 night: a named target stood under an unexplained region's top layer (``fallback_stand``)
    def _stand_model(self, typ: str, yaw: float = 0.0):
        """``typ``'s model stood the way it rests and turned ``yaw`` about the vertical, its root at the origin:
        ``(R, lo, hi, top_lo_xy, top_hi_xy)``, the box of the whole and of its own top layer (the primitives that
        reach within ``top_m`` of its highest point: a bottle's cap, a can's whole lid, a flat box's whole top)."""
        from .behind import rest_R
        from .ground import _boxes

        c, s_ = float(np.cos(yaw)), float(np.sin(yaw))
        R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]) @ rest_R(self.lib, typ)
        los, his = [], []
        for ctr, Rp, half in _boxes(self.lib, {"type": typ, "bodies": self._build(typ, R, np.zeros(3))}):
            e = np.abs(Rp) @ half
            los.append(ctr - e)
            his.append(ctr + e)
        los, his = np.array(los), np.array(his)
        lo, hi = los.min(0), his.max(0)
        sel = his[:, 2] >= float(hi[2]) - float((self.stand or STAND)["top_m"])  # (``_misread`` asks without it)
        return R, lo, hi, los[sel, :2].min(0), his[sel, :2].max(0)

    def _outline_place(self, typ: str, b: dict, base: float, cam: dict, yaw: float = 0.0):
        """Where ``typ``'s model (turned ``yaw``) stands if the region's outline in the picture is its own:
        ``(xy, iou)``, the place of the model's top layer's middle and how well the stood model's box in the picture
        covers the region's (intersection over union), or None. No depth is used: the camera's ray through the
        middle of the outline's upper edge meets the model's own height over the surface at the far edge of its
        top, and the top's middle lies half the top's extent along that ray nearer the camera."""
        from .behind import ray_at
        from .ground import _boxes

        K, T = np.asarray(cam["K"], float), np.asarray(cam["T"], float)
        u0, v0, u1, v1 = (float(x) for x in b["uv_box"])
        R, lo, hi, tlo, thi = self._stand_model(typ, yaw)
        Q = ray_at(K, T, (0.5 * (u0 + u1), v0), base + float(hi[2] - lo[2]))
        if Q is None:
            return None
        h = (T[:3, 3] - Q)[:2]
        n = float(np.linalg.norm(h))
        if n < 1e-6:
            return None
        h = h / n
        # the top layer's half extent along the ray (its box is axis-aligned in the world at this yaw)
        r = 0.5 * (abs(float(h[0])) * float(thi[0] - tlo[0]) + abs(float(h[1])) * float(thi[1] - tlo[1]))
        xy = Q[:2] + h * r
        t = np.array([xy[0] - 0.5 * float(tlo[0] + thi[0]), xy[1] - 0.5 * float(tlo[1] + thi[1]), base - float(lo[2])])
        sg = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
        P = np.concatenate([ctr + (sg * half) @ np.asarray(Rp, float).T
                            for ctr, Rp, half in _boxes(self.lib, {"type": typ, "bodies": self._build(typ, R, t)})])
        Pc = (P - T[:3, 3]) @ T[:3, :3]
        if float(Pc[:, 2].min()) <= 0.05:
            return None
        u = K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2] - 0.5
        v = K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2] - 0.5
        a0, b0, a1, b1 = float(u.min()), float(v.min()), float(u.max()), float(v.max())
        inter = max(0.0, min(u1, a1) - max(u0, a0)) * max(0.0, min(v1, b1) - max(v0, b0))
        union = (u1 - u0) * (v1 - v0) + (a1 - a0) * (b1 - b0) - inter
        return xy, (inter / union if union > 0 else 0.0)

    def _stand_candidates(self, typ: str, used: set, allow=None) -> list:
        """``fallback_stand``: the unexplained regions that can be ``typ`` standing there, best first, as
        ``[(key, region index, percept)]``.

        Measured on the rehearsal's rows against the simulator's own start poses (colour line, start state 0,
        2026-11-16): a thing the look could not name (drawn in other colours, lit otherwise) is left as one region,
        and the net's depth smears part of it along the camera's line of sight down onto the table behind it. The
        region's own middle, where ``_from_blob`` stood the model, lay 5 to 13 cm behind the thing, and the fingers
        closed on nothing there. The region's OUTLINE in the picture was the thing's own to a few pixels, also where
        the net read the whole thing 3 to 4 cm too far; its TOP LAYER (the server's ``extra_blob_top_m``) lay within
        0.2 to 0.5 cm of the thing's middle where the net read its height right.

        A region counts when it is large enough and its top layer stands at about the model's height over the
        surface under it (``height``). The model is stood the way it rests, its base on that surface:
        * by the outline (the camera and the region's pixel box known): ``_outline_place``, kept when the stood
          model's box in the picture covers the region's by ``iou``;
        * else by the layer: the layer as large across as the model's own top layer (``size``), the model's top
          layer's middle under it; with ``slide`` the layer's middle is first slid along the camera's line of sight
          to the model's own height (a depth read too far puts the layer lower and further off, both along it).
        Order: regions clear of every reported object before those inside one's footprint; regions the look itself
        named (a rejected pose of the type lies in them) before the others; then the top layer nearest the model's
        in size and height."""
        st = self.stand
        if not st or typ not in self.lib.export_types:
            return []
        blobs = (self.extra or {}).get("blobs") or []
        try:
            if self.lib.m1_type(typ).get("kind") != "object":
                return []  # a fixture is never stood on a region (its own stand-ins: the cabinet's memory and twin)
            _R, lo, hi, tlo, thi = self._stand_model(typ)
        except Exception:
            return []
        height = float(hi[2] - lo[2])
        m = sorted([0.5 * float(thi[0] - tlo[0]), 0.5 * float(thi[1] - tlo[1])], reverse=True)
        foot = sorted([float(hi[0] - lo[0]), float(hi[1] - lo[1])], reverse=True)
        cam = (self.cams or {}).get("agentview")
        named = [np.asarray(q["pos"], float)[:2] for q in (self.extra or {}).get("rejected") or []
                 if q.get("type") == typ and q.get("pos") is not None]
        out = []
        for k, b in enumerate(blobs):
            tb = b.get("top")
            if k in used or not tb or int(b.get("n_px") or 0) < int(st["min_px"]):
                continue
            base = float(tb["z"]) - float(b["height"])  # the surface under the region (both are 95th percentiles)
            if self.blob_on_table:
                t_ = self._table_top(tb["xy"], float(tb["z"]) - 0.005)
                if t_ is not None:
                    base = max(base, t_)
            seen_h = float(tb["z"]) - base
            rh = seen_h / max(height, 1e-3)
            if not float(st["height"][0]) <= rh <= float(st["height"][1]) \
                    and abs(seen_h - height) > float(st["height_abs_m"]):
                continue
            sz = sorted([float(x) for x in tb["half_xy"]], reverse=True)
            fits, cost = True, abs(float(np.log(max(rh, 1e-3))))
            for mm, ss in zip(m, sz):
                r = max(ss, 1e-4) / max(mm, 1e-4)
                fits = fits and float(st["size"][0]) <= r <= float(st["size"][1])
                cost += abs(float(np.log(r)))
            xy, how, slid, yaw, iou = None, None, 0.0, 0.0, None
            bh = [float(x) for x in tb["half_xy"]]
            if max(bh) >= float(st["long"]) * max(min(bh), 1e-4) and m[0] >= float(st["long"]) * max(m[1], 1e-4):
                yaw = float(tb["yaw"]) + (0.0 if bh[0] >= bh[1] else 0.5 * np.pi)  # the layer's long axis
                yaw -= 0.0 if (thi[0] - tlo[0]) >= (thi[1] - tlo[1]) else 0.5 * np.pi  # the model top's long axis onto it
            if cam is not None and cam.get("K") is not None and b.get("uv_box"):
                # the model's turn about the vertical: the layer's when it is long (above); else, for a model longer
                # than wide across (a bottle under its round cap), the quarter turn whose outline fits the better
                yaws = [yaw]
                if yaw == 0.0 and foot[0] >= float(st["long"]) * max(foot[1], 1e-4):
                    yaws.append(0.5 * np.pi)
                o = None
                for y_ in yaws:
                    try:
                        o_ = self._outline_place(typ, b, base, cam, y_)
                    except Exception:
                        o_ = None
                    if o_ is not None and (o is None or float(o_[1]) > float(o[1])):
                        o, yaw_o = o_, y_
                if o is not None:
                    iou = float(o[1])
                    if iou >= float(st["iou"]):
                        xy, how, yaw = o[0], "outline", yaw_o
            if xy is None:
                if not fits:
                    continue
                xy, how = np.asarray(tb["xy"], float)[:2].copy(), "layer"
                if st.get("slide") and cam is not None:
                    C = np.asarray(cam["T"], float)[:3, 3]
                    P = np.array([xy[0], xy[1], float(tb["z"])])
                    z_goal = base + height
                    if C[2] - P[2] > 0.05 and C[2] - z_goal > 0.05:
                        Q = C + (P - C) * ((C[2] - z_goal) / (C[2] - P[2]))
                        slid = float(np.linalg.norm(Q[:2] - xy))
                        if slid > float(st["slide_max_m"]):
                            continue
                        xy = Q[:2]
            if allow is not None and not allow(xy):
                continue  # ``refute_target``: the hand has closed on nothing there
            if self.table_bound and not self._over_table(xy):
                continue
            R, lo_r, _hi_r, tlo_r, thi_r = self._stand_model(typ, yaw)
            t = np.array([xy[0] - 0.5 * float(tlo_r[0] + thi_r[0]), xy[1] - 0.5 * float(tlo_r[1] + thi_r[1]),
                          base - float(lo_r[2])])
            is_named = any(self._blob_has(b, q, grow=float(st["near_m"])) for q in named)
            inside = self._inside_other(typ, xy, margin=BLOB_NEAR_M)
            if not inside and b.get("hang_share") is not None and float(b["hang_share"]) > float(st["hang"]):
                continue  # it hangs (the arm, a panel, something in front of the scene): nothing stands there
            gid = None
            if st.get("track"):  # the type's own model id, as the look gave it to a rejected pose of the type
                gid = next((q.get("gid") for q in (self.extra or {}).get("rejected") or []
                            if q.get("type") == typ and q.get("gid") is not None), None)
            out.append(((bool(inside), not is_named, float(cost)), k,
                        {"type": typ, "gid": gid, "score": 0.0, "n_px": int(tb.get("n_px") or 0),
                         "fit": {"n_agree": 0, "inlier": 0.0}, "source": "blob_stand", "joints": {},
                         "bodies": self._build(typ, R, t),
                         "stand": {"region": int(k), "by": how, "cost": round(float(cost), 3),
                                   "iou": None if iou is None else round(iou, 3),
                                   "height_ratio": round(rh, 3), "slid_m": round(slid, 4), "yaw": round(yaw, 3),
                                   "named": bool(is_named), "inside": bool(inside)}}))
        return sorted(out, key=lambda x: (x[0], x[1]))

    def _from_stand(self, typ: str, used: set, allow=None):
        """The best of ``_stand_candidates`` (its region then counts as used), or None. None too when the older
        fallback's first choice, the best rejected pose of the type, lies in that region or within ``same_m`` of the
        stood place: the look fitted the type's own model to the same thing, and that pose stays as it was (it is a
        measured one; and a thing drawn at another size keeps the pose that was fitted to it, where the stored
        height misplaced the stood model by 5 to 6 cm). So the stood model replaces only a fallback that points
        somewhere else: the region's own middle, or a pose in another place."""
        try:
            cands = self._stand_candidates(typ, used, allow)
            size = None
            if not cands and (self.stand or {}).get("resize") is not None:
                size, cands = self._stand_sized(typ, used, allow)
            if not cands:
                return None
            _c, k, p = cands[0]
            if size is not None:  # the model drawn to the region's height: no fit of the stored one stands for it
                p["stand"]["size"] = size
                used.add(k)
                return p
            lo, hi = self._box_of(p)
            at = 0.5 * (lo[:2] + hi[:2])
            region = ((self.extra or {}).get("blobs") or [])[k]
            old = (self._from_rejected(typ, inside=False, allow=allow) if self.table_bound
                   else self._from_rejected(typ, allow=allow))
            if old is not None:
                oxy = np.asarray(old["bodies"][self.lib.export_types[typ]["root_body"]]["pos"], float)[:2]
                if self._blob_has(region, oxy, grow=float(self.stand["near_m"])) \
                        or float(np.linalg.norm(oxy - at)) <= float(self.stand["same_m"]):
                    return None
            used.add(k)
            return p
        except Exception:  # the older fallbacks stand whatever goes wrong in the new one
            return None

    def _stand_sized(self, typ: str, used: set, allow=None):
        """``fallback_stand``'s ``resize``: ``(factors, candidates)`` for a named target no region shows at its stored
        height. For each region whose top layer stands between ``lo`` and ``hi`` of the model's height above the
        surface under it (largest first, at most three): the type is drawn to that height in the episode's library
        (``sized.resize_type``, as wide as stored) and the regions are judged again by ``_stand_candidates``; the
        first height that gives a candidate stays. Else the stored model is put back and nothing is returned. Only
        for a type the look reported no percept of (the library holds a type once)."""
        from . import sized as SZ

        opt = {**STAND_SIZE, **((self.stand or {}).get("resize") or {})}
        try:
            if any(p.get("type") == typ for p in self.percepts) or SZ.sized(self.lib, typ) is not None:
                return None, []
            if self.lib.m1_type(typ).get("kind") != "object":
                return None, []
            _R, lo, hi, _tlo, _thi = self._stand_model(typ)
            height = float(hi[2] - lo[2])
            st = self.stand
            tried = []
            for k, b in enumerate((self.extra or {}).get("blobs") or []):
                tb = b.get("top")
                if k in used or not tb or int(b.get("n_px") or 0) < int(opt["min_px"]):
                    continue
                base = float(tb["z"]) - float(b["height"])
                if self.blob_on_table:
                    t_ = self._table_top(tb["xy"], float(tb["z"]) - 0.005)
                    if t_ is not None:
                        base = max(base, t_)
                seen_h = float(tb["z"]) - base
                rh = seen_h / max(height, 1e-3)
                if float(st["height"][0]) <= rh <= float(st["height"][1]) or abs(seen_h - height) < float(opt["min_m"]):
                    continue  # the stored height's own range: ``_stand_candidates`` has judged it
                if float(opt["lo"]) <= rh <= float(opt["hi"]):
                    tried.append((-int(b.get("n_px") or 0), round(rh, 3)))
            for _n, rh in sorted(tried)[:3]:
                s3 = SZ.resize_type(self.lib, typ, (1.0, 1.0, rh))
                cands = self._stand_candidates(typ, used, allow)
                if cands:
                    return s3, cands
                SZ.restore_type(self.lib, typ)
        except Exception:
            try:
                SZ.restore_type(self.lib, typ)
            except Exception:
                pass
        return None, []

    def is_stood(self) -> bool:
        """``fallback_stand``: the target is a model stood on a region and no look has measured it since (its turn
        about the vertical is not a measured one; the agent's ``c1_stand`` asks)."""
        tgt = next((x for x in self.percepts if x.get("id") == self.target), None)
        if tgt is None or str(tgt.get("source") or "") not in ("blob_stand", "predicted", "held"):
            return False
        last = next((f for f in reversed(self.fallbacks) if f.get("id") == tgt["id"] and f.get("role") == "target"), None)
        return bool(last) and "stand" in last

    def _box_of(self, p: dict):
        from .ground import aabb

        lo, hi = aabb(self.lib, p)
        return np.asarray(lo, float), np.asarray(hi, float)

    def _stand_drop(self, p: dict, keep_types=()) -> list:
        """``fallback_stand``: the ids of reported objects that are the stood target read as another type: a thing
        of kind object, of another type than the target's and than ``keep_types`` (what the instruction names),
        whose root stands within ``drop_m`` of the stood target's middle and whose top is at least ``drop_dz_m``
        lower (the target's own region stuck out above it: 2026-11-16, a tall bottle reported as a shorter one in 2
        of 33 such looks, a flat box reported on top of a can in 3)."""
        st = self.stand
        lo, hi = self._box_of(p)
        at = 0.5 * (lo[:2] + hi[:2])
        out = []
        for x in self.percepts:
            typ = x.get("type")
            if x is p or typ == p.get("type") or typ in keep_types or not x.get("bodies") \
                    or typ not in self.lib.export_types:
                continue
            try:
                if self.lib.m1_type(typ).get("kind") != "object":
                    continue
                xlo, xhi = self._box_of(x)
            except Exception:
                continue
            if float(np.linalg.norm(0.5 * (xlo[:2] + xhi[:2]) - at)) <= float(st["drop_m"]) \
                    and float(xhi[2]) <= float(hi[2]) - float(st["drop_dz_m"]):
                out.append(x["id"])
        return out

    def _misread(self, typ: str, named=()) -> list:
        """``misread_target``: where the named target (or receiver) stands read as another type, best first, as
        ``[(the other percept's id, a percept of the named type there)]``.

        A thing the stored model does not explain (another make of it, wider and lower) is sometimes explained
        well enough by ANOTHER stored model for the look to report that one: a wide shallow bowl as a second plate
        (the case that swaps the bowl, Goal and Spatial training tasks, every seed run on either line: the look
        agreed with 0.42 to 0.57 of the false plate's visible points, with 0.97 and more of a real plate's, and the
        bowl's rim stood out 2 to 3 cm above it as unexplained regions; the bowl was then looked for in those
        slivers, 4 to 7 cm from its middle, the fingers closed on nothing, and the false plate stood in the way of
        every other grasp). A reported percept counts when
        * it is an object of another type, and, if the instruction names its type, not the best of that type;
        * the look agrees with no more than ``weak`` of its visible points;
        * it is about as large across as the target's model (``size``);
        * a region's top layer lies on its footprint, at least ``over_m`` above its top and between ``height`` of
          the target model's height above its base.
        The target is stood there as it rests, its middle over the percept's, its base at the percept's base; with
        ``resize``, and when the look reported no percept of the target's type at all, its model is drawn to the
        percept's size across and the region's height (``sized.py``). The weakest fit first."""
        from .ground import aabb, in_footprint

        opt = getattr(self, "misread", None)
        if not opt or typ not in self.lib.export_types:
            return []
        out = []
        try:
            if self.lib.m1_type(typ).get("kind") != "object":
                return []
            _R, lo0, hi0, _tl, _th = self._stand_model(typ)
            height = float(hi0[2] - lo0[2])
            m = sorted([float(hi0[0] - lo0[0]), float(hi0[1] - lo0[1])], reverse=True)
            blobs = (self.extra or {}).get("blobs") or []

            def strength(q):
                f_ = q.get("fit") or {}
                return float(f_.get("n_agree") or 0.0) * float(f_.get("inlier") or 0.0)

            for x in self.percepts:
                xt = x.get("type")
                if xt == typ or not x.get("bodies") or xt not in self.lib.export_types:
                    continue
                if self.lib.m1_type(xt).get("kind") != "object":
                    continue
                f_ = x.get("fit") or {}
                if not f_.get("n_vis"):
                    continue
                share = float(f_.get("n_agree") or 0.0) / float(f_["n_vis"])
                if share > float(opt["weak"]):
                    continue
                same = [q for q in self.percepts if q.get("type") == xt]
                if xt in named and (len(same) < 2 or x is max(same, key=strength)):
                    continue
                xlo, xhi = (np.asarray(v, float) for v in aabb(self.lib, x))
                px = sorted([float(xhi[0] - xlo[0]), float(xhi[1] - xlo[1])], reverse=True)
                if not all(float(opt["size"][0]) <= a / max(b, 1e-4) <= float(opt["size"][1]) for a, b in zip(px, m)):
                    continue
                top = None
                for b in blobs:
                    tb = b.get("top") or {}
                    if tb.get("xy") is None or tb.get("z") is None or int(b.get("n_px") or 0) < 300:
                        continue
                    if not in_footprint(self.lib, x, np.asarray(tb["xy"], float)[:2], float(opt["near_m"])):
                        continue
                    z = float(tb["z"])
                    if z - float(xhi[2]) >= float(opt["over_m"]) and (top is None or z > top):
                        top = z
                if top is None:
                    continue
                rh = (top - float(xlo[2])) / max(height, 1e-3)
                if not float(opt["height"][0]) <= rh <= float(opt["height"][1]):
                    continue
                out.append((share, x["id"], 0.5 * (xlo[:2] + xhi[:2]), float(xlo[2]), rh, px[0] / max(m[0], 1e-4)))
        except Exception:
            return []
        res = []
        for share, pid, at, base, rh, rx in sorted(out, key=lambda v: v[0]):
            size = None
            try:
                if opt.get("resize") and not res and not any(q.get("type") == typ for q in self.percepts):
                    from . import sized as SZ

                    if SZ.sized(self.lib, typ) is None:
                        try:
                            size = SZ.resize_type(self.lib, typ, (rx, rx, rh))
                        except Exception:  # the stored model stands there (no grasp of the drawn one)
                            size = None
                R, lo, hi, _tl, _th = self._stand_model(typ)
                t = np.array([float(at[0]) - 0.5 * float(lo[0] + hi[0]), float(at[1]) - 0.5 * float(lo[1] + hi[1]),
                              base - float(lo[2])])
                res.append((pid, {"type": typ, "gid": None, "score": 0.0, "n_px": 0,
                                  "fit": {"n_agree": 0, "inlier": 0.0}, "source": "misread", "joints": {},
                                  "bodies": self._build(typ, R, t),
                                  "misread": {"as": pid, "share": round(share, 3), "height_ratio": round(rh, 3),
                                              "across_ratio": round(rx, 3), **({"size": size} if size else {})}}))
            except Exception:
                continue
        return res

    def _tall(self) -> dict | None:
        """``behind_occluder``'s optional rule for a depth that does not show where a hanging thing ends: a region at
        least ``tall_m`` high and ``tall_px`` large (15,000) counts as hanging in front. Off unless ``tall_m`` is set."""
        if not self.behind or self.behind.get("tall_m") is None:
            return None
        return {"height_m": float(self.behind["tall_m"]), "px": int(self.behind.get("tall_px", 15000))}

    def _occluders(self) -> list:
        """The look's regions that hang in front of the scene (``behind.occluders`` with this tool's options:
        ``hang_m``, ``min_px``, ``share``, ``tall_m``). With ``standing_max_px`` set (off by default): none when the
        look leaves a larger STANDING region unexplained. Then the view itself does not match the stored camera (it
        was moved, or the lens is another), what hangs in it is no panel, and the arm check will say so at the first
        motion (2026-11-16: every first look of the moved-camera and other-lens cases, none of the panel case)."""
        from . import behind as BH

        blobs = (self.extra or {}).get("blobs")
        lim = self.behind.get("standing_max_px")
        if lim is not None:
            n = BH.standing_px(blobs)
            if n > int(lim):
                self.behind_refused = {"standing_px": int(n)}
                return []
        return BH.occluders(blobs, float(self.behind.get("hang_m", BH.HANG_MIN_M)),
                            int(self.behind.get("min_px", BH.HANG_MIN_PX)), self._tall(),
                            float(self.behind.get("share", BH.HANG_SHARE)))

    def behind_point(self, typ: str) -> dict | None:
        """Where to look for a target the main camera does not see: the point where the camera's ray through the
        middle of the largest thing hanging in front of the scene comes down to ``look_rise_m`` above the support
        there (a small object's middle height). ``{"xy", "support_z", "uv", "n_px", "h_min"}``, or None when nothing
        hangs there (2026-11-16, off unless ``self.behind`` is set)."""
        if self.behind is None or typ not in self.lib.export_types:
            return None
        from . import behind as BH

        cam = self.cams.get("agentview")
        occ = self._occluders()
        if cam is None or not occ:
            return None
        rise = float(self.behind.get("look_rise_m", 0.05))
        z_s = BH.table_z(self.lib, occ[0].get("xy") or (0.0, 0.0))
        hit = None
        for _ in range(3):
            if z_s is None:
                return None
            hit = BH.ray_at(cam["K"], cam["T"], occ[0]["uv"], z_s + rise)
            if hit is None:
                return None
            z_new = BH.support_under(self.lib, self.percepts, hit[:2], skip_types=(typ,))
            if z_new is None or abs(z_new - z_s) < 1e-4:
                break
            z_s = z_new
        if self.table_bound and not self._over_table(hit):
            return None
        return {"xy": [float(hit[0]), float(hit[1])], "support_z": float(z_s), "uv": [float(x) for x in occ[0]["uv"]],
                "n_px": int(occ[0]["n_px"]), "h_min": float(occ[0]["h_min"])}

    def _wrist_rgb(self):
        """The wrist camera's colour picture now: the colour line's own (``cam.colour``), else the camera tool's."""
        a = time.perf_counter()
        try:
            get = getattr(self.cam, "colour", None)
            rgb = get(WRIST) if get is not None else self.cam(WRIST, MP.CAMERAS[WRIST], self.t_now)["rgb"]
        except Exception:
            rgb = None
        self.render_s += time.perf_counter() - a
        return None if rgb is None else np.asarray(rgb)

    def wrist_find(self, typ: str, support_z: float, hint_xy=None, opts: dict | None = None) -> dict | None:
        """One colour picture from the wrist camera, looked at from above for a thing of ``typ``'s size that the main
        camera did not place (``wrist2d.find``): everything the percepts explain and the robot's own gripper are
        left out of the picture. Returns the find with ``top_z``, or None. Outside the look budget (no server call).
        ``opts`` (``relation_look``'s confirming picture): options for this one picture, over the tool's own."""
        from . import grasp as GR
        from . import wrist2d as W2
        from .ground import _boxes, aabb

        rgb = self._wrist_rgb()
        if rgb is None or rgb.ndim != 3 or rgb.shape[0] != rgb.shape[1]:
            self.looks.append({"t": self.t_now, "mode": "wrist", "status": "no_picture"})
            return None
        ph, Rh = self._hand()
        K, T = W2.camera(ph, Rh, rgb.shape[0])
        q = self.obs["robot0_gripper_qpos"]
        half_gap = 0.5 * self.lib.jaw.gap_from_qpos(q[0], q[1])
        own = self.lib.jaw.gripper_boxes_G(max(half_gap, 0.0)).placed(GR.G_matrix(Rh[:, 1]), ph)
        known = []
        for p in self.percepts:
            if not p.get("bodies") or p.get("source") in ("behind", "wrist"):
                continue
            try:
                bx = _boxes(self.lib, p)
            except Exception:
                continue
            if bx:
                known.append(GR.Boxes([c for c, _R, _h in bx], [R for _c, R, _h in bx], [h for _c, _R, h in bx]))
        from .behind import rest_R

        lo, hi = aabb(self.lib, {"type": typ, "bodies": self._build(typ, rest_R(self.lib, typ), np.zeros(3))})
        # the plane the picture is read on: the thing's middle height. Seen from above and off-centre, a standing
        # thing's top lies further out in the picture than its foot; on the plane halfway up the two cancel
        mid_z = float(support_z) + 0.5 * float(hi[2] - lo[2])
        area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
        wopts = (self.behind or {}).get("wrist")
        if self.wrist_opts:  # ``relation_look``'s preferences for this look (None: as before)
            wopts = {**(wopts or {}), **self.wrist_opts}
        if opts:
            wopts = {**(wopts or {}), **opts}
        sized = (wopts or {}).get("prefer") == "size"
        if sized:
            # things are compared by size with the named object's outline seen from above, not with its box (a
            # bowl's is 0.78 of the box; a slab's slanted image read 0.0109 m2 where the box says 0.0115 and a
            # bowl shows 0.008 to 0.009: the Mac depth line, a training task with a slab, start state 0)
            area = self._top_area(typ) or area
        got = W2.find(rgb, K, T, mid_z, list(own.corners()), [c for b in known for c in b.corners()], hint_xy, area,
                      wopts)
        self.looks.append({"t": self.t_now, "mode": "wrist", "status": "found" if got else "nothing",
                           **({k: got[k] for k in ("n_cands", "thr", "area_m2")} if got else {}),
                           **({"expect_m2": round(area, 5), "base_rgb": got.get("base_rgb"),
                               "ranked": [[round(float(c["xy"][0]), 3), round(float(c["xy"][1]), 3),
                                           round(float(c["area_m2"]), 5), list(c["rgb"]), bool(c["edge"])]
                                          for c in got.get("ranked") or []]} if got and sized else {})})
        return None if got is None else {**got, "plane_z": mid_z, "support_z": float(support_z)}

    def _top_area(self, typ: str) -> float | None:
        """The area of ``typ``'s outline seen from above as it rests (``wrist2d.footprint_area``); None if it cannot
        be told."""
        memo = self.__dict__.setdefault("_top_areas", {})
        if typ not in memo:
            try:
                from . import wrist2d as W2
                from .behind import rest_R
                from .ground import _boxes

                a = W2.footprint_area(_boxes(self.lib, {"type": typ, "bodies": self._build(typ, rest_R(self.lib, typ),
                                                                                             np.zeros(3))}))
                memo[typ] = float(a) if a > 0.0 else None
            except Exception:
                memo[typ] = None
        return memo[typ]

    def add_found(self, typ: str, got: dict) -> dict:
        """A percept of ``typ`` where a wrist look found a thing: the stored model upright on the support there, its
        long side along the thing's (as the blob fallback sets a model on an unexplained region). No ``gid``."""
        from .behind import rest_R
        from .ground import aabb

        R0 = rest_R(self.lib, typ)  # the model as it rests (not every type's body has its z up)
        lo, hi = aabb(self.lib, {"type": typ, "bodies": self._build(typ, R0, np.zeros(3))})
        mh = 0.5 * (hi - lo)
        yaw = 0.0
        if max(mh[0], mh[1]) > 1.25 * min(mh[0], mh[1]) and got["half"][0] > 1.15 * got["half"][1]:
            yaw = float(got["yaw"]) - (0.0 if mh[0] >= mh[1] else 0.5 * np.pi)  # the model's long axis onto the thing's
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R0
        lo, hi = aabb(self.lib, {"type": typ, "bodies": self._build(typ, R, np.zeros(3))})
        t = np.array([float(got["xy"][0]) - 0.5 * float(lo[0] + hi[0]), float(got["xy"][1]) - 0.5 * float(lo[1] + hi[1]),
                      float(got["support_z"]) - float(lo[2])])
        ids = {p.get("id") for p in self.percepts}
        pid = next(f"w{k}" for k in range(1, 100) if f"w{k}" not in ids)
        p = {"id": pid, "type": typ, "gid": None, "score": 0.0, "n_px": int(got.get("n_cells") or 0),
             "fit": {"n_agree": 0, "inlier": 0.0}, "source": "wrist", "joints": {}, "bodies": self._build(typ, R, t)}
        self.percepts = [x for x in self.percepts if not (x.get("type") == typ and x.get("source") == "wrist")] + [p]
        self.n_behind += 1
        self.fallbacks.append({"role": "target", "type": typ, "id": pid, "source": "wrist",
                               "xy": [round(float(x), 4) for x in t[:2]], "yaw": round(yaw, 3)})
        return p

    def raise_target(self, dz: float) -> bool:
        """The target's stored pose ``dz`` higher (the agent's ``stall_unseen_m``: the open hand stopped on it above
        where it was set). Only a percept whose height was never seen: found by the wrist camera or set behind what
        hides it. True when it was raised."""
        tgt = next((x for x in self.percepts if x["id"] == self.target), None)
        if tgt is None or tgt.get("source") not in ("wrist", "behind") or not 0.0 < float(dz) < 0.15:
            return False
        new = {**tgt, "raised_m": round(float(tgt.get("raised_m") or 0.0) + float(dz), 4),
               "bodies": {rel: {"pos": [float(b["pos"][0]), float(b["pos"][1]), float(b["pos"][2]) + float(dz)],
                                "quat_wxyz": b["quat_wxyz"]} for rel, b in tgt["bodies"].items()}}
        self.percepts = [new if x["id"] == self.target else x for x in self.percepts]
        self.fallbacks.append({"role": "target", "type": tgt["type"], "id": tgt["id"], "source": tgt["source"],
                               "raised_m": round(float(dz), 4)})
        return True

    def add_behind(self, typ: str, on_z: float | None = None) -> dict | None:
        """The named target is in none of the first look's percepts and a thing hangs in front of the scene: a
        percept of ``typ`` behind the middle of it (``behind.py``; 2026-11-16, off unless ``self.behind`` is set).
        Returns the record, or None when nothing hangs there or the look's camera is not known."""
        if self.behind is None or typ not in self.lib.export_types:
            return None
        from . import behind as BH

        cam = self.cams.get("agentview")
        occ = self._occluders()
        if cam is None or not occ:
            return None
        ids = {p.get("id") for p in self.percepts}
        pid = next(f"b{k}" for k in range(1, 100) if f"b{k}" not in ids)
        try:
            p, rec = BH.behind_percept(self.lib, typ, occ[0], cam, pid, self.percepts,
                                       str(self.behind.get("aim", "origin")), on_z)
        except Exception:
            return None
        if p is None or (self.table_bound and not self._over_table(p["bodies"][self.lib.export_types[typ]["root_body"]]["pos"])):
            return None
        self.percepts = list(self.percepts) + [p]
        self.n_behind += 1
        self.fallbacks.append({"role": "target", "type": typ, "id": pid, "source": "behind", **rec})
        return {"id": pid, **rec}

    # -- 2026-11-16: fallbacks that follow the evidence (``fallback_top``) and the relation asked of one candidate
    def _model_box(self, typ: str, R=None):
        """(lo, hi) of the type's model with its root at the origin, turned by ``R``."""
        from .ground import aabb

        return aabb(self.lib, {"type": typ, "bodies": self._build(typ, np.eye(3) if R is None else R, np.zeros(3))})

    def _size_cost(self, typ: str, half_xy) -> float | None:
        """How far a region's size across is from the model's: the sum over its two axes (long with long) of
        |log ratio|; None when either ratio is outside ``FALLBACK_SIZE``."""
        lo, hi = self._model_box(typ)
        m = sorted([0.5 * float(hi[0] - lo[0]), 0.5 * float(hi[1] - lo[1])], reverse=True)
        b = sorted([float(half_xy[0]), float(half_xy[1])], reverse=True)
        cost = 0.0
        for mm, bb in zip(m, b):
            r = max(bb, 1e-4) / max(mm, 1e-4)
            if not FALLBACK_SIZE[0] <= r <= FALLBACK_SIZE[1]:
                return None
            cost += abs(float(np.log(r)))
        return cost

    def _rejected_usable(self, typ: str, tilted: bool = False) -> list:
        """The server's rejected poses of ``typ`` that may stand in for it, best first: upright (within 20 degrees),
        fitted (inlier 0.5 or more), over the table, and not a copy. A copy is a pose the server accepted and then
        dropped as a duplicate: it lies on the pixels of a percept it did report (in every Spatial look, a plate on
        the stove's burner). ``tilted``: the tilted ones instead (more than 20 degrees off upright)."""
        out = []
        for q in (self.extra or {}).get("rejected") or []:
            if q.get("type") != typ or not q.get("bodies") or q.get("pos") is None:
                continue
            why, fit = q.get("why") or {}, q.get("fit") or {}
            inl = float(fit.get("inlier") or 0.0)
            if why.get("accepted") or inl < 0.5 or (float(why.get("upright_err_deg") or 0.0) > 20.0) != bool(tilted):
                continue
            if not self._over_table(q["pos"]):
                continue
            out.append((float(fit.get("n_agree") or 0.0) * inl, len(out), q))
        return [q for _k, _i, q in sorted(out, key=lambda x: (-x[0], x[1]))]

    def _as_percept(self, q: dict, source: str) -> dict:
        p = {k: v for k, v in q.items() if k not in ("why", "pos")}
        p["fit"] = dict(q.get("fit") or {})
        p.setdefault("joints", {})
        p["source"] = source
        return p

    def _upright(self, q: dict, source: str) -> dict | None:
        """A tilted rejected pose stood upright where it is: its root stays, its turn about the vertical stays, the
        tilt goes. (The look's search read a reshaped or recoloured bowl as a tilted one, 2 to 5 cm from where it
        stood, in 7 of 16 wrong picks: the place is evidence, the tilt is not.) The target's own track look
        measures it again before the first grasp."""
        typ = q["type"]
        root = self.lib.export_types[typ]["root_body"]
        try:
            R = np.asarray(self.lib.quat_mat(q["bodies"][root]["quat_wxyz"]), float)
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            c, s_ = np.cos(yaw), np.sin(yaw)
            Rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
            p = self._as_percept(q, source)
            p["bodies"] = self._build(typ, Rz, np.asarray(q["bodies"][root]["pos"], float))
            return p
        except Exception:
            return None

    @staticmethod
    def _blob_has(b: dict, xy, grow: float = 0.02) -> bool:
        """``xy`` lies inside the region's box across (grown by ``grow``)."""
        c, s_ = float(np.cos(float(b["yaw"]))), float(np.sin(float(b["yaw"])))
        d = np.asarray(xy, float)[:2] - np.asarray(b["xy"], float)[:2]
        u, v = d[0] * c + d[1] * s_, -d[0] * s_ + d[1] * c
        return bool(abs(u) <= float(b["half_xy"][0]) + grow and abs(v) <= float(b["half_xy"][1]) + grow)

    def _blob_percept(self, typ: str, b: dict, top: bool) -> dict | None:
        """A percept of ``typ`` built on an unexplained region: on its top layer (``top``: the model's top at the
        layer's level, centred on the layer; never below the surface the region stands on), or on the whole region
        (its base on that surface). None when the size across does not fit the model."""
        box = b.get("top") if top else b
        if not box or self._size_cost(typ, box["half_xy"]) is None:
            return None
        lo0, hi0 = self._model_box(typ)
        bh = [float(x) for x in box["half_xy"]]
        yaw = float(box["yaw"]) + (0.0 if bh[0] >= bh[1] else 0.5 * np.pi)  # the box's long axis
        yaw -= 0.0 if (hi0[0] - lo0[0]) >= (hi0[1] - lo0[1]) else 0.5 * np.pi  # the model's long axis onto it
        c, s_ = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        lo, hi = self._model_box(typ, R)
        z = float(b["z_top"]) - float(b["height"]) - float(lo[2])  # its base on the surface under the region
        if top:
            z = max(z, float(box["z"]) - float(hi[2]))
        t = np.array([float(box["xy"][0]) - 0.5 * float(lo[0] + hi[0]),
                      float(box["xy"][1]) - 0.5 * float(lo[1] + hi[1]), z])
        return {"type": typ, "gid": None, "score": 0.0, "n_px": int(box.get("n_px") or 0),
                "fit": {"n_agree": 0, "inlier": 0.0}, "source": "blob_top" if top else "blob_whole", "joints": {},
                "bodies": self._build(typ, R, t)}

    def _own_leftover(self, b: dict) -> bool:
        """``fallback_own``: the region is what a reported object's fit left over of that object itself: its top
        layer's middle (the region's own, without one) lies on the object's footprint grown by ``BLOB_NEAR_M`` and
        its top no more than 5 cm above the object's. (Colour line, the case that draws the basket smaller, two
        Object tasks at both seeds run: the net read the target can a centimetre and a half taller than its model,
        the 670 to 930 pixels above the fit were a region of 17 by 19 cm along the camera's rays, that region
        fitted the basket "as a whole" better than the basket's own region did, the basket was stood on the can and
        no grasp of the can was free of it: four episodes of four ended with no plan.)"""
        from .ground import aabb, in_footprint

        try:
            top = b.get("top") or {}
            xy = np.asarray(top.get("xy") if top.get("xy") is not None else b["xy"], float)[:2]
            z = float(top.get("z") if top.get("z") is not None else b["z_top"])
            for q in self.percepts:
                if not q.get("bodies") or not in_footprint(self.lib, q, xy, BLOB_NEAR_M):
                    continue
                if z <= float(aabb(self.lib, q)[1][2]) + 0.05:
                    return True
        except Exception:
            return False
        return False

    def _from_evidence(self, typ: str, role: str, used: set):
        """``fallback_top``: the stand-in for a named RECEIVER type the look did not
        report, taken from what the look did see (a target keeps the older fallbacks: offline on the colour line's
        stored looks, raising a target under a taller region would have lifted five correct poses). Rejected poses
        are those that are no copy and lie clear of the reported objects. In this order:
        1. a rejected pose that explains its own place: no unexplained region tops it by more than
           ``FALLBACK_OVER_M``;
        2. an unexplained region that fits the model as a whole, across and in height, and holds no rejected pose
           (an object drawn at another size: its false pose sat on the enlarged bowl beside it);
        3. a rejected pose under a region that tops it by more than ``FALLBACK_RAISE_M`` and whose top layer has the
           model's size across: the object stands on something. The pose is raised until its top is that layer's
           level and centred on the layer (a basket on a block was read at table level, 10 cm low, and the hand
           knocked it off);
        4. any other rejected pose, as it is;
        5. with no rejected pose at all, an unexplained region whose top layer has the model's size across: the
           model is set on that layer (a plate on a block has no pose; the region's own middle was 9 to 15 cm off).
        None when nothing fits: the older fallbacks follow. Steps 3 and 5 need the server's ``extra_blob_top_m``."""
        from .ground import aabb

        blobs = (self.extra or {}).get("blobs") or []
        root = self.lib.export_types[typ]["root_body"]
        lo0, hi0 = self._model_box(typ)
        height = float(hi0[2] - lo0[2])

        def over(q):  # the tallest unexplained region over a pose: (index, how far its top is above the pose's top)
            pos = np.asarray(q["bodies"][root]["pos"], float)
            top = float(aabb(self.lib, q)[1][2])
            best = None
            for k, b in enumerate(blobs):
                if k in used or not self._blob_has(b, pos[:2]):
                    continue
                gap = float((b.get("top") or {}).get("z", b["z_top"])) - top
                if best is None or gap > best[1]:
                    best = (k, gap)
            return best

        clear = [q for q in self._rejected_usable(typ) if not self._inside_other(typ, q["pos"])]
        doubt = []
        for q in clear:
            o = over(q)
            if o is None or o[1] <= FALLBACK_OVER_M:
                return self._as_percept(q, f"rejected:{q.get('source')}")  # 1
            doubt.append((q, o))
        best, bkey = None, None
        own = {k for k, b in enumerate(blobs) if self._own_leftover(b)} if getattr(self, "fallback_own", False) else set()
        for k, b in enumerate(blobs):  # 2
            if k in used or k in own or int(b.get("n_px") or 0) < 300 or float(b.get("height") or 0.0) <= 0.01:
                continue
            if any(self._blob_has(b, np.asarray(q["pos"], float)[:2]) for q in clear):
                continue
            c = self._size_cost(typ, b["half_xy"])
            r = max(float(b["height"]), 1e-3) / max(height, 1e-3)
            if c is None or not FALLBACK_WHOLE_H[0] <= r <= FALLBACK_WHOLE_H[1]:
                continue
            c += abs(float(np.log(r)))
            if c <= FALLBACK_WHOLE_COST and (bkey is None or c < bkey):
                best, bkey = k, c
        if best is not None:
            used.add(best)
            return self._blob_percept(typ, blobs[best], top=False)
        for q, (k, gap) in doubt:  # 3
            tb = blobs[k].get("top")
            if gap <= FALLBACK_RAISE_M or not tb or self._size_cost(typ, tb["half_xy"]) is None:
                continue
            p = self._as_percept(q, f"rejected+raised:{q.get('source')}")
            lo, hi = aabb(self.lib, q)
            move = np.array([0.0, 0.0, float(tb["z"]) - float(hi[2])])
            move[:2] = np.asarray(tb["xy"], float)[:2] - 0.5 * (lo[:2] + hi[:2])
            p["bodies"] = {rel: {"pos": [float(x) for x in np.asarray(bd["pos"], float) + move],
                                 "quat_wxyz": bd["quat_wxyz"]} for rel, bd in q["bodies"].items()}
            used.add(k)
            return p
        if doubt:  # 4
            q = doubt[0][0]
            return self._as_percept(q, f"rejected:{q.get('source')}")
        best, bkey = None, None
        for k, b in enumerate(blobs):  # 5
            tb = b.get("top")
            if k in used or k in own or not tb or int(b.get("n_px") or 0) < 300:
                continue
            c = self._size_cost(typ, tb["half_xy"])
            if c is not None and (bkey is None or c < bkey):
                best, bkey = k, c
        if best is not None:
            used.add(best)
            return self._blob_percept(typ, blobs[best], top=True)
        return None

    def _standin_fits(self, lt: str, q: dict, typ) -> bool:
        """``standin_strict``: a rejected pose of landmark type ``lt`` that lies under a reported object of the
        target's type is that object's support only if the object stands as high as the landmark is tall: its base
        between 0.6 and 1.2 of the landmark's height above what is known under it (a bowl nests a little into a
        ramekin). Start states 0 and 8, one case: the cookie box under the OTHER bowl was read as a ramekin (2 cm
        where a ramekin is 4), and "the bowl on the ramekin" became that other bowl. A pose under no such object
        is not judged here."""
        from .behind import support_under
        from .ground import aabb, in_footprint

        try:
            lo, hi = self._model_box(lt)
            h = float(hi[2] - lo[2])
            xy = np.asarray(q["pos"], float)[:2]
            over = [p for p in self.percepts if p.get("type") == typ and p.get("bodies") and in_footprint(self.lib, p, xy)]
            if not over or h <= 0.0:
                return True
            below = support_under(self.lib, [p for p in self.percepts if p.get("type") != typ], xy)
            if below is None:
                return True
            return any(0.6 * h <= float(aabb(self.lib, p)[0][2]) - float(below) <= 1.2 * h for p in over)
        except Exception:
            return True

    def _standin_seen(self, q: dict) -> bool:
        """``standin_strict``: a rejected pose stands in for a landmark only when the look shows it: a quarter or more
        of the model's visible points agree with the depth picture (the server's ``vis_share``), and of the points
        with nothing nearer in front of them three quarters or more (its ``share``; the server accepts at 0.85, and a
        landmark under an object standing on it reads just under that). Both starts, every type the first look did
        not report, against the replays' truth (cookie boxes aside: the stove's burner reads as one in every look,
        and only its footprint tells): the 5 right poses read 0.33 to 0.60 and 0.83 to 1.00; of the 53 wrong ones
        (0.11 to 0.56 m from the real thing) 37 read lower on one or the other, among them every pose that lay
        hidden under something and showed next to nothing (a stove at the cabinet, a ramekin under a slab). A pose
        that carries no such numbers is not judged here."""
        why = q.get("why") or {}
        try:
            if why.get("vis_share") is not None and float(why["vis_share"]) < STANDIN_SEEN:
                return False
            if why.get("share") is not None and float(why["share"]) < STANDIN_SHARE:
                return False
        except (TypeError, ValueError):
            pass
        return True

    def relation_candidates(self, parsed: dict) -> list:
        """``params["relation_single"]`` (2026-11-16: a single candidate was taken without checking the
        instruction's relation): called after ``on_parsed``.
        * A landmark type with no percept gets a stand-in, the best rejected pose of that type that is no copy. The
          stand-ins are returned for ``ground`` and never join the percepts (a wrong one would only misplace the
          relation, not the scene the hand moves in).
        * When the look reported ONE object of the target's type and it does not stand in the relation at all
          (``ground.relation_holds``), the named object is one the look did not report: the first rejected pose or
          unexplained region of that type that does stand in the relation joins the percepts as a second candidate,
          and ``ground`` chooses. (Colour line, start state 8: 14 of 38 wrongly placed Spatial episodes carried the
          other bowl.)"""
        from . import ground as G

        self.landmark_fb = []
        self.relation_target = None
        try:
            if not isinstance(parsed, dict) or not parsed.get("relation") or not self.extra:
                return []
            have = {p.get("type") for p in self.percepts}
            for lt in parsed.get("landmarks") or []:
                if lt == "table" or lt in have or lt not in self.lib.export_types:
                    continue
                # not on another reported object (the stove's burner fits a cookie box in every Spatial look, plain
                # ones too); under a candidate of the target's type is where a landmark it stands on would be
                rs = [q for q in self._rejected_usable(lt)
                      if not self._inside_other(lt, q["pos"], skip=(parsed.get("target_type"),))]
                if self.standin_strict:
                    rs = [q for q in rs if self._standin_seen(q)
                          and self._standin_fits(lt, q, parsed.get("target_type"))]
                if rs:
                    self.landmark_fb.append(self._as_percept(rs[0], f"landmark:{rs[0].get('source')}"))
                    self.fallbacks.append({"role": "landmark", "type": lt, "source": self.landmark_fb[-1]["source"]})
            typ = parsed.get("target_type")
            cands = [p for p in self.percepts if p.get("type") == typ]
            was = [p for p in cands if p.get("relation_candidate")]
            if was:  # planned again in the same episode: the candidate that joined before is still the target
                self.relation_target = was[0]["id"]
                return self.landmark_fb
            if len(cands) != 1 or typ not in self.lib.export_types:
                return self.landmark_fb
            wc = self.lib.workcell
            if G.relation_holds(parsed, cands[0], self.percepts, self.lib, wc, self.landmark_fb) is not False:
                return self.landmark_fb
            root = self.lib.export_types[typ]["root_body"]
            here = np.asarray(cands[0]["bodies"][root]["pos"], float)
            options = [self._as_percept(q, f"rejected:{q.get('source')}") for q in self._rejected_usable(typ)
                       if float(np.linalg.norm(np.asarray(q["pos"], float)[:2] - here[:2])) >= 0.03]
            if getattr(self, "misread", None):  # ``misread_target``: the named one read as another type, first
                named_types = tuple(x for x in [parsed.get("place_type"), *(parsed.get("place_types") or []),
                                                *(parsed.get("landmarks") or [])] if x)
                mis = [q for _pid, q in self._misread(typ, named_types)]
                for q in mis:
                    q.pop("misread", None)
                options = mis + options
            for b in (self.extra or {}).get("blobs") or []:
                if int(b.get("n_px") or 0) < 300 or float(b.get("height") or 0.0) <= 0.01:
                    continue
                q = self._blob_percept(typ, b, top=True) if b.get("top") else self._blob_percept(typ, b, top=False)
                if q is not None:
                    options.append(q)
            for q in self._rejected_usable(typ, tilted=True):  # last: a tilted fit, stood upright where it is
                u = self._upright(q, f"rejected+upright:{q.get('source')}")
                if u is not None:
                    options.append(u)
            ids = {p.get("id") for p in self.percepts}
            for q in options:
                if G.relation_holds(parsed, q, self.percepts, self.lib, wc, self.landmark_fb):
                    n = 1
                    while f"r{n}" in ids:
                        n += 1
                    q["id"] = f"r{n}"
                    q["relation_candidate"] = True
                    # the same thing read twice: a reported percept of another type that stands where this
                    # candidate stands, and is the weaker of several of its own type, leaves the scene (a reshaped
                    # bowl was reported as a second plate, and would stand in the hand's way as one)
                    at = np.asarray(q["bodies"][root]["pos"], float)[:2]
                    lands = {id(x) for x in G.landmarks_of(parsed, self.percepts, self.landmark_fb) if x}
                    drop = []
                    for p in self.percepts:
                        if p.get("type") == typ or id(p) in lands or not p.get("bodies"):
                            continue
                        same = [x for x in self.percepts if x.get("type") == p.get("type")]
                        pr = self.lib.export_types[p["type"]]["root_body"]
                        if len(same) > 1 and p is not max(same, key=lambda x: (x.get("fit") or {}).get("n_agree", 0)) \
                                and float(np.linalg.norm(np.asarray(p["bodies"][pr]["pos"], float)[:2] - at)) < 0.03:
                            drop.append(p["id"])
                    if drop:
                        self.percepts = [p for p in self.percepts if p["id"] not in drop]
                        self.fallbacks.append({"role": "dropped", "ids": drop, "why": "read twice"})
                    self.percepts = list(self.percepts) + [q]
                    self.relation_target = q["id"]
                    self.fallbacks.append({"role": "target", "type": typ, "id": q["id"], "source": q["source"],
                                           "relation": parsed["relation"]})
                    break
        except Exception:
            return self.landmark_fb
        return self.landmark_fb

    def on_parsed(self, parsed: dict) -> None:
        """A named target or place type with no reported percept takes the server's best rejected pose of that type,
        else the unexplained blob that best fits its model's size."""
        try:
            if not isinstance(parsed, dict) or not self.extra:
                return
            used: set = set()
            ids = {p.get("id") for p in self.percepts}
            n = 0
            for role, key in (("target", "target_type"), ("place", "place_type")):
                typ = parsed.get(key)
                if not typ or typ not in self.lib.export_types:
                    continue
                if self.table_bound:  # a reported percept of a named type far off the table is a false fit: drop it
                    root = self.lib.export_types[typ]["root_body"]
                    keep = [p for p in self.percepts if p.get("type") != typ or root not in (p.get("bodies") or {})
                            or self._over_table(p["bodies"][root]["pos"])]
                    if len(keep) != len(self.percepts):
                        self.fallbacks.append({"role": role, "type": typ, "dropped_off_table": len(self.percepts) - len(keep)})
                        self.percepts = keep
                if any(p.get("type") == typ for p in self.percepts):
                    continue
                if role == "target" and self.behind is not None and self.behind.get("mode") == "ray" \
                        and self.add_behind(typ) is not None:
                    continue  # nothing of it is in view and something hangs in front of the scene: behind that
                if role == "target" and self.behind is not None and self.behind.get("mode", "look") == "look" \
                        and self.behind_point(typ) is not None:
                    continue  # the planner looks behind it with the wrist camera first (``episode._plan_one``)
                p = None
                if self.fixture_memory:
                    try:
                        p = self._from_memory(typ)
                    except Exception:  # the older fallbacks stand whatever goes wrong in the new one
                        p = None
                if p is None and role == "target" and self.seen_target == "first":
                    u2 = set(used)
                    p = self._from_blob(typ, role, u2)  # what is seen before a refused pose of the stored model
                    if p is not None and p.get("source") == "seen":
                        used |= u2
                    else:
                        p = None
                if p is None and self.fallback_top and role == "place":
                    try:
                        p = self._from_evidence(typ, role, used)
                    except Exception:  # the older fallbacks stand whatever goes wrong in the new one
                        p = None
                dropped = []
                if p is None and getattr(self, "misread", None):  # (a receiver the same way as a target)
                    named_types = tuple(x for x in [parsed.get("target_type"), parsed.get("place_type"),
                                                    *(parsed.get("place_types") or []),
                                                    *(parsed.get("landmarks") or [])] if x and x != typ)
                    mr = self._misread(typ, named_types)
                    if mr:
                        dropped, p = [mr[0][0]], mr[0][1]
                        self.percepts = [x for x in self.percepts if x.get("id") not in dropped]
                        self.fallbacks.append({"role": "dropped", "ids": list(dropped),
                                               "why": "the target read as another type"})
                if p is None and self.stand and role == "target":
                    named_types = tuple(x for x in [parsed.get("place_type"), *(parsed.get("place_types") or []),
                                                    *(parsed.get("landmarks") or [])] if x)
                    p = self._from_stand(typ, used)
                    try:
                        if p is not None and p.get("source") == "blob_stand":
                            dropped = self._stand_drop(p, named_types)
                    except Exception:
                        dropped = []
                    if dropped:
                        self.percepts = [x for x in self.percepts if x.get("id") not in dropped]
                        self.fallbacks.append({"role": "dropped", "ids": list(dropped),
                                               "why": "the target read as another type"})
                if p is not None:
                    pass
                elif self.table_bound:  # a pose clear of the other objects, else a blob, else a pose inside one
                    p = self._from_rejected(typ, inside=False)
                    if p is None:  # (``blob_clear_first``: a blob clear of the other objects before one inside one)
                        p = (self._from_blob(typ, role, used, inside=False) if self.blob_clear_first
                             else self._from_blob(typ, role, used))
                    if p is None:
                        p = self._from_rejected(typ, inside=True)
                    if p is None and self.blob_clear_first:
                        p = self._from_blob(typ, role, used, inside=True)
                else:
                    p = self._from_rejected(typ)
                    if p is None:
                        p = self._from_blob(typ, role, used)
                if p is None:
                    continue
                pid = p.get("id")
                while pid is None or pid in ids:
                    n += 1
                    pid = f"f{n}"
                p["id"] = pid
                ids.add(pid)
                self.percepts = list(self.percepts) + [p]
                self.fallbacks.append({"role": role, "type": typ, "id": pid, "source": p["source"],
                                       **({"stand": p.pop("stand")} if "stand" in p else {}),
                                       **({"size": p["memory"]["size"]} if (p.get("memory") or {}).get("size") else {}),
                                       **({"misread": p.pop("misread")} if "misread" in p else {})})
        except Exception:
            return

    # -- the tool
    def __call__(self) -> dict:
        self.calls += 1
        self.call_steps.append(int(self.t_now))
        self.target_missing = False
        if self.calls > 1:
            if self.n_full + self.n_track >= self.budget:
                self.budget_exhausted = True
                tgt = next((x for x in self.percepts if x["id"] == self.target), None)
                if tgt is not None:
                    self.percepts = [self._predict(tgt) if x["id"] == self.target else x for x in self.percepts]
            elif ((self._held is not None or self._release_pending) and (not self._closed() or self._opening)) \
                    or self._lost_xy is not None:
                # the gripper has just opened: the held object left the hand, so its hand-relative prediction is
                # stale. One merged full look re-measures the scene (the release and C3 reads). The target's stored
                # pose is moved to the hand prediction AT RELEASE first (amendment A16): the merged look matches each
                # percept around its stored pose, and the stored pose was the lift-time track (0.1-0.47 m away).
                # (Also when the agent asks again for a target lost in a carry and not found yet: nothing is held
                # then, and the stored pose stays as it is.)
                lost = self._lost_xy is not None
                stood = self._stood if (self._held is not None and self._opening and not lost) else None
                tgt = next((x for x in self.percepts if x["id"] == self.target), None)
                if tgt is not None:
                    pred = self._predict(tgt)
                    if self.drop_m > 0.0:
                        # let go above its place: it is where it landed, straight below (the merged look and the
                        # track that follow search around this pose; unseen, it keeps it)
                        if self.drop_rigid and self._held is not None:
                            pred = self._carried(tgt)
                        pred = self._lowered(pred, self.drop_m)
                        self.n_drop += 1
                    self.percepts = [pred if x["id"] == self.target else x for x in self.percepts]
                self.drop_m = 0.0
                self._held = None
                self._release_pending = False  # (``release_here`` had already moved the pose; ``_predict`` left it)
                ref_xy, self._pick_xy = self._pick_xy, None
                if ref_xy is not None:
                    # only an opening near the pick (a failed grasp or lift), not the release at the place
                    ph, _Rh = self._hand()
                    if float(np.linalg.norm(ph[:2] - ref_xy)) > REJ_NEAR_M:
                        ref_xy = None
                if self._lost_xy is not None:  # it fell out during a carry: a rejected pose near where it fell from
                    ref_xy = self._lost_xy
                self.full_look(merge=True, ref_xy=ref_xy, free=tuple(self.place_free))
                if self._opening:
                    self.looks[-1]["told_open"] = True
                # the target in a container is often not detected by the full look (occluded by the basket's
                # walls); it then keeps the in-hand prediction, from which it has already fallen a few cm, so the
                # first C3 read would differ from the next by the fall (amendment A20: one dev run, 11 of 100 Object C3
                # false alarms, all 'rest' failures of 1.2-3.5 cm). A track look from that prediction re-measures it
                # in the same call when the merged look did not associate it (one more look).
                found = (self.looks[-1].get("assoc") or {}).get(self.target, 0.0) is not None
                if not found and stood is not None:
                    # told the hand let go after a grasp or a lift that failed, and the full look shows nothing of
                    # the target's type (the hand is still over it): it is where it stood when the hand closed, not
                    # in the hand. The track look below measures it from there.
                    self.percepts = [({**x, "bodies": {r: dict(b) for r, b in stood.items()}, "source": "unheld"}
                                      if x["id"] == self.target else x) for x in self.percepts]
                    self.looks[-1]["unheld"] = True
                if not found and self.n_full + self.n_track < self.budget:
                    self.track_look()
                    found = self.looks[-1].get("status") == "tracked"
                if lost:  # nothing of its type where it could have fallen, and nothing where the hand had it
                    self.target_missing = not found
                    self.looks[-1]["lost"] = "missing" if self.target_missing else "found"
            else:
                self.track_look()
                if self.redetect and self._redetect_due():
                    # item 5: the target's track was refused (it is not where it was predicted, e.g. knocked by a
                    # missed grasp, or moved by the scene) with the hand open and nothing held. One merged full look
                    # re-measures the scene; the merge takes the nearest fresh percept of the target's type at any
                    # distance when none is within REASSOC_M.
                    self.n_redetect += 1
                    self.full_look(merge=True)
                    self.looks[-1]["redetect"] = True
            self._opening, self._lost_xy = False, None  # said once, for the one look that follows
        return self.lib.poses_from_percepts(self.percepts)

    def _redetect_due(self) -> bool:
        """Item 5: the last look was the target's refused track (``predicted``; not a blob percept the server cannot
        track), the hand is open and holds nothing, and the look budget has room."""
        last = self.looks[-1] if self.looks else {}
        return (last.get("mode") == "track" and last.get("status") == "predicted" and self._held is None
                and not self._closed() and self.n_full + self.n_track < self.budget)
