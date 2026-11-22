"""The camera from the robot's own arm, during its first motion (item 2; colour line, ``colour.calib`` ``"arm"``).

Runner side, policy side in what it reads: the episode's own images through the colour line's camera tool, the
proprioception and the perception server's replies. It holds no simulator handle.

The colour line reads its camera from the constants stored per suite (``colour_line.stored_cameras``). This watches the
robot's own first motion instead of adding one:

1. a frame before the first action (the camera tool's look: the episode image and the net's depth, at the assumed
   camera), then a frame each time the hand has moved at least ``min_move_m`` from where every earlier frame saw it,
   until three frames are taken or ``max_step`` steps have passed;
2. the perception server's ``arm_cam`` request, mode ``gate_fit`` (``rowant_lab/libero_env/arm_camera.py``): if the
   assumed camera explains the arm's moving pixels, nothing changes and the episode goes on exactly as without this;
   otherwise the server fits the camera from the arm;
3. an accepted fit replaces the camera tool's camera (``look.override``), and the runner looks and plans again from the
   current state with the steps left (``dev_episode._act``).

**When the robot does not move** (no plan at the start; or it gives up, finishing with a reason in ``after``, before
the camera was checked, with at least ``min_left`` steps left; never after success): the robot makes the motion itself,
once (``begin_motion``). A fresh frame where the hand is, then ``LEGS`` at full speed holding the hand's orientation, a
frame at the end of each leg (within ``leg_tol_m`` or after ``leg_max_steps``); then step 2. With the stored camera
wrong the robot often finds no plan at all, and such an episode is lost anyway, so the motion costs nothing that the
episode had.

The frames cost no motion while the robot moves by itself. A frame costs one net pass, and the request about half a
second of server time; a fit, when the gate fires, about 10 seconds.
"""

from __future__ import annotations

import time

import numpy as np

ROBOT = {"min_move_m": 0.06, "max_step": 40, "frames": 3, "leg_tol_m": 0.03, "leg_max_steps": 12,
         "after": ("no_plan", "max_attempts"), "min_left": 60}
IMAGE = {"diff": 60, "neighbours": 3, "brighter": False,  # ``arm_image``: a pixel changed when its colour is this far
         "approx_cost_m": 0.012, "approx_ratio": 0.35}
# (summed over the three channels, of 255 each) from the frames' median, and this many of its 8 neighbours changed
# too. ``brighter``: only pixels brighter than the median by that much. Measured on real frames (2026-11-15,
# ``probes/s16_fit.py``): at 60, either sign, 99.5% or more of the changed pixels of a plain picture lie on the moving
# links, and where a link newly covers a pixel only 18 to 88% change by that much (a pale arm before its own base).
# The server's fit scores by overlap, which that suits (``arm_camera.fit_image``).
# ``approx_cost_m``, ``approx_ratio``: a zoom and shift that failed their check, used all the same when nothing else
# explained the arm and they brought its cost under the first and under the second times what it was (``_approx``).


def change_masks(rgbs: list, diff: float, neighbours: int = 3, brighter: bool = False) -> list:
    """One mask per frame: the pixels whose colour differs from the frames' per-pixel median (lone pixels dropped).
    With three frames these are the pixels the arm covers in some frames and not in all; with ``brighter``, the
    pixels a pale arm covers in that frame alone."""
    A = np.stack([np.asarray(x) for x in rgbs]).astype(np.int16)
    med = np.median(A, axis=0)
    M = ((A - med[None]).sum(axis=3) > float(diff)) if brighter else (np.abs(A - med[None]).sum(axis=3) > float(diff))
    if neighbours > 0:
        P = np.pad(M.astype(np.int16), ((0, 0), (1, 1), (1, 1)))
        nb = sum(P[:, 1 + dy:P.shape[1] - 1 + dy, 1 + dx:P.shape[2] - 1 + dx]
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
        M &= nb >= int(neighbours)
    return [m for m in M]
LEGS = ((0.0, 0.10, 0.0), (0.0, 0.10, 0.10), (0.0, -0.10, 0.10))  # hand offsets from where the motion starts (m)
# 2026-11-16, ``arm_joints["readings_fit"]`` (a dict of overrides; off unless given): the readings' offsets from the
# readings alone. ``gap_mm``: on the depth line (no ``guard_mm`` there) the fit is asked for when the readings are
# this far apart. While the offsets are not yet determined the robot asks again every ``every`` steps, until step
# ``until``; at most ``max_n`` steps' readings are kept.
READINGS = {"gap_mm": 5.0, "every": 4, "until": 160, "max_n": 400}


class ArmCalib:
    def __init__(self, cam_tool, client, suite: str, episode: str, params: dict):
        self.cam, self.client, self.suite, self.episode = cam_tool, client, suite, episode
        self.o = {**ROBOT, **(params.get("arm_cam_robot") or {})}
        self.frames: list = []
        self.rgbs: list = []  # item 2b: the frames' colour images, kept only for the second try
        self.second = getattr(cam_tool, "net2_depth", None)  # the colour line's second net (``colour.net2``), or None
        # item 4 (2026-11-14, off unless ``params["arm_joints"]``): the joint readings' constant offsets from the arm.
        # On the depth line (its camera tool has no ``override``: the camera is right) the request is ``gate_joints``;
        # on the colour line ``joints`` follows a refused camera fit. An accepted fit becomes ``fix`` ({dq, dp}): the
        # runner subtracts it from every later reading (``correct``), and the robot looks and plans again. The colour
        # line's guard (``arm_joints["guard_mm"]``): the joint fit only when the hand reading and the arm model's hand at
        # the joint readings disagree by more than that (``_readings``), and then the camera is taken as right.
        self.joints = params.get("arm_joints")
        # 2026-11-16 (off unless ``arm_joints["readings_fit"]``): when the hand reading and the joint readings
        # contradict each other, the constant offsets of both that make them agree at every step so far, from the
        # readings alone (the server's ``readings_fit``; it holds the arm model). Asked before the fit from the arm's
        # picture, which it replaces when it answers: the readings agree to float precision once the right offsets
        # are off them, where the picture's fit trades one joint against another (2026-11-16, the rehearsal: 15 of 28
        # S18 episodes ended "joints refused" and none of them was solved). Motion along one line does not determine
        # the offsets: the robot then keeps its readings, leaves the camera alone and asks again as it moves.
        rf = (self.joints or {}).get("readings_fit")
        self.rfit = None if rf is None else {**READINGS, **dict(rf)}
        self.samples: list = []  # every step's readings while no fix is known (``readings_fit`` only)
        self._rfit_wait = False
        self._rfit_t = 0
        # 2026-11-15 (off unless ``params["arm_image"]``, a dict of ``IMAGE`` overrides): when the gate fires, first
        # ask whether the picture itself came shifted or zoomed (the server's ``gate_image``, from the pixels that
        # changed between the frames' colour images). An accepted answer is checked (the same frames with it undone,
        # through the net, must pass the gate) and then undone in every later look (``look.override["image"]``).
        img = params.get("arm_image")
        self.image = None if img is None or getattr(cam_tool, "image_depth", None) is None else {**IMAGE, **dict(img)}
        self.camera_line = getattr(cam_tool, "override", None) is not None
        self.fix = None
        self.hands: list = []
        self.done = False
        self.moving = False
        self.motion_done = False
        self._leg = 0
        self._leg_steps = 0
        self._p0 = self._R0 = None
        self.rec: dict = {"status": "watching", "frames_at": []}
        self.s = 0.0
        # stream PATH, round 3 (2026-11-19, off unless ``arm_cam_robot["register"]``, a dict of
        # ``cam_register.REGISTER`` overrides; colour line): once this check has ended, one frame is the reference,
        # and every later front-camera frame is registered against it inside the camera tool. A camera that moved
        # in the middle of the episode is then replaced at the first look after the move; the robot does not plan
        # again (what it saw before the move was seen through the right camera).
        rg = self.o.get("register")
        self.register = None
        if rg is not None and self.camera_line:
            from rowant_lab.pro2.cam_register import Register

            self.register = Register(rg if isinstance(rg, dict) else {})
            cam_tool.override["register"] = self.register
            self.rec["register"] = self.register.rec

    def _grab(self, obs: dict, t: int) -> None:
        a = time.perf_counter()
        fr = self.cam("agentview", 1024, t)
        self.frames.append({"depth_u16": fr["depth_u16"], "depth_scale_mm": fr["depth_scale_mm"], "K": fr["K"],
                            "T_world_from_camera": fr["T_world_from_camera"],
                            "joint_pos": np.asarray(obs["robot0_joint_pos"], float),
                            "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], float)})
        if self.second is not None or self.image is not None:
            self.rgbs.append(np.array(fr["rgb"], copy=True))
        self.hands.append(np.asarray(obs["robot0_eef_pos"], float))
        self.rec["frames_at"].append(int(t))
        self.s += time.perf_counter() - a

    def start(self, obs: dict) -> None:
        self._grab(obs, 0)
        self._sample(obs)

    def _sample(self, obs: dict) -> None:
        if self.rfit is None:
            return
        if self.rfit.get("log"):  # development only: every step's readings into the record (numbers)
            q = obs["robot0_eef_quat_xyzw"] if "robot0_eef_quat_xyzw" in obs else obs.get("robot0_eef_quat")
            self.rec.setdefault("readings_log", []).append(
                [[round(float(x), 7) for x in np.asarray(obs[k], float)] for k in
                 ("robot0_joint_pos", "robot0_eef_pos", "robot0_gripper_qpos")] + [[round(float(x), 7) for x in (q or [])]]
                + [[round(float(x), 7) for x in np.asarray(obs["robot0_joint_vel"], float)]
                   if obs.get("robot0_joint_vel") is not None else []])
        if self.fix is None and len(self.samples) < int(self.rfit["max_n"]) and (not self.done or self._rfit_wait):
            fr = {"joint_pos": np.asarray(obs["robot0_joint_pos"], float),
                  "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], float),
                  "eef_pos": np.asarray(obs["robot0_eef_pos"], float)}
            if obs.get("robot0_joint_vel") is not None:  # for the hand reading's lag behind the joint readings
                fr["joint_vel"] = np.asarray(obs["robot0_joint_vel"], float)
            self.samples.append(fr)

    def _readings_fit(self, t: int):
        """``readings_fit``: True when the offsets were found (``fix`` set); None when they are not determined yet
        (more motion is needed); False when constant offsets do not explain the readings."""
        a = time.perf_counter()
        rep = self.client.call({"op": "arm_cam", "mode": "readings_fit", "suite": self.suite,
                                "episode": self.episode, "frames": list(self.samples)})
        self.s += time.perf_counter() - a
        rf = dict(rep.get("readings_fit") or {})
        self._rfit_t = int(t)
        self.rec["readings_fit"] = {**rf, "at": int(t), "tries": int((self.rec.get("readings_fit") or {}).get("tries", 0)) + 1}
        self.rec["s"] = round(self.s, 3)
        if rf.get("accepted") and len(rf.get("delta") or []) == 7 and len(rf.get("dp") or []) == 3:
            self.fix = {"dq": np.asarray(rf["delta"], float), "dp": np.asarray(rf["dp"], float)}
            self.samples = []
            self._rfit_wait = False
            self.rec.update(status="joints fixed", by="readings", at=int(t))
            return True
        return None if rf.get("reason") == "undetermined" else False

    def _readings_route(self, t: int):
        """True: fixed. False: not determined yet, the robot asks again as it moves (the camera is left alone: the
        readings are known to contradict each other). None: not constant offsets; the caller goes on as without it."""
        r = self._readings_fit(t)
        if r:
            return True
        if r is None:
            self._rfit_wait = True
            self.rec["status"] = "readings fit waiting"
            return False
        return None

    def _readings_again(self, t: int) -> bool:
        if t - self._rfit_t < int(self.rfit["every"]):
            return False
        r = self._readings_fit(t)
        if r:
            return True
        if r is False or t >= int(self.rfit["until"]):
            self._rfit_wait = False
            self.rec["status"] = "readings fit refused" if r is False else "readings fit undetermined"
        return False

    # -- the robot's own motion, when it does not move by itself
    def can_move(self, agent, finished: bool, t: int, cap: int) -> bool:
        """The robot's own motion is due: no agent at all, or the agent gave up (``after``) before the camera was
        checked (watching found too little motion, or is still waiting), with ``min_left`` steps left. Once only."""
        if self.moving or self.motion_done or (self.done and self.rec["status"] not in
                                               ("too little motion", "readings fit waiting")):
            return False
        if cap - t < int(self.o["min_left"]):
            return False
        return agent is None or (finished and getattr(agent, "finish_reason", None) in tuple(self.o["after"]))

    def begin_motion(self, obs: dict, t: int) -> None:
        from rowant_lab.pro2.robot.grasp import quat_xyzw_to_mat

        self.moving, self.motion_done, self.done = True, True, False
        self._rfit_wait = False  # the motion's own request asks again, with the legs' readings
        self.frames, self.hands, self.rgbs = [], [], []  # the scene may have changed since the earlier frames: start afresh
        self._grab(obs, t)
        self._p0 = np.asarray(obs["robot0_eef_pos"], float)
        self._R0 = quat_xyzw_to_mat(obs["robot0_eef_quat_xyzw"] if "robot0_eef_quat_xyzw" in obs else obs["robot0_eef_quat"])
        self._leg, self._leg_steps = 0, 0
        self.rec.update(motion_at=int(t))

    def motion_action(self, obs: dict) -> list:
        from rowant_lab.pro2.robot.follow import follow
        from rowant_lab.pro2.robot.grasp import quat_xyzw_to_mat

        q = obs["robot0_eef_quat_xyzw"] if "robot0_eef_quat_xyzw" in obs else obs["robot0_eef_quat"]
        tgt = self._p0 + np.asarray(LEGS[min(self._leg, len(LEGS) - 1)])
        return follow(np.asarray(obs["robot0_eef_pos"], float), quat_xyzw_to_mat(q), tgt, self._R0, -1.0, 1.0, 1.0)

    def _motion_step(self, obs: dict, t: int) -> bool:
        """True when a leg ended and its frame was taken."""
        self._leg_steps += 1
        tgt = self._p0 + np.asarray(LEGS[self._leg])
        if float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"], float) - tgt)) > float(self.o["leg_tol_m"]) \
                and self._leg_steps < int(self.o["leg_max_steps"]):
            return False
        self._grab(obs, t)
        self._leg += 1
        self._leg_steps = 0
        return True

    def after_step(self, obs: dict, t: int) -> bool:
        """Called after each step of the episode. True once the camera was replaced (look and plan again)."""
        self._sample(obs)
        if self._rfit_wait and not self.moving:
            return self._readings_again(t)
        if self.done:
            self._arm(t)  # (``register``, off unless given) the reference picture, once
            return False
        if self.moving:
            if not self._motion_step(obs, t):
                return False
            if len(self.frames) < int(self.o["frames"]) and self._leg < len(LEGS):
                return False
            self.moving = False
            return self._request(t)
        if t > int(self.o["max_step"]):
            self.done = True
            self.rec["status"] = "too little motion"
            return False
        p = np.asarray(obs["robot0_eef_pos"], float)
        if min(float(np.linalg.norm(p - h)) for h in self.hands) < float(self.o["min_move_m"]):
            return False
        self._grab(obs, t)
        if len(self.frames) < int(self.o["frames"]):
            return False
        return self._request(t)

    def _arm(self, t: int) -> None:
        """``register``: the check has ended, so the camera tool's next front-camera frame is the reference
        (``cam_register``); it is taken here, one step after the check's answer, with whatever the check changed."""
        r = self.register
        if r is None or r.armed or self.moving or self._rfit_wait:
            return
        a = time.perf_counter()
        r.arm()
        self.cam("agentview", 1024, t)
        self.s += time.perf_counter() - a
        self.rec["s"] = round(self.s, 3)

    def correct(self, obs: dict) -> dict:
        """Item 4: the readings with the fitted offsets taken off (``obs`` unchanged when there is no fix)."""
        if self.fix is None:
            return obs
        out = dict(obs)
        out["robot0_joint_pos"] = [float(x) for x in np.asarray(obs["robot0_joint_pos"], float) - self.fix["dq"]]
        out["robot0_eef_pos"] = [float(x) for x in np.asarray(obs["robot0_eef_pos"], float) - self.fix["dp"]]
        return out

    def _readings(self, frames: list, hands: list) -> float:
        """Item 4, the colour line's guard: the largest gap (mm), over the frames, between the hand reading and the arm
        model's hand at the joint readings (server side, where the arm model is read). Proprioception only."""
        a = time.perf_counter()
        rep = self.client.call({"op": "arm_cam", "mode": "readings", "suite": self.suite, "episode": self.episode,
                                "frames": [{"joint_pos": fr["joint_pos"], "gripper_qpos": fr["gripper_qpos"],
                                            "eef_pos": np.asarray(h, float)} for fr, h in zip(frames, hands)]})
        self.s += time.perf_counter() - a
        r = rep.get("readings") or {}
        self.rec["readings"] = r
        return float(r.get("max_mm") or 0.0)

    def _joints(self, mode: str, frames: list, hands: list, t: int) -> bool:
        """Item 4: the joint fit (``mode`` ``joints`` or ``gate_joints``). True when accepted (``fix`` set)."""
        a = time.perf_counter()
        rep = self.client.call({"op": "arm_cam", "mode": mode, "suite": self.suite, "episode": self.episode,
                                "frames": frames})
        self.s += time.perf_counter() - a
        if mode == "gate_joints":
            g = rep.get("gate") or {}
            self.rec.update(gate=g, at=int(t))
            if not g.get("moved"):
                self.rec.update(status="kept" if g.get("moved") is False else "undecided", s=round(self.s, 3))
                return False
        jf = rep.get("joints") or {}
        self.rec["joints"] = {k: v for k, v in jf.items() if k != "eef_fk"}
        self.rec["s"] = round(self.s, 3)
        if not jf.get("accepted") or len(jf.get("eef_fk") or []) != len(hands):
            if mode == "gate_joints":
                self.rec["status"] = "joints refused"
            return False
        dp = np.mean([np.asarray(h, float) - np.asarray(e, float) for h, e in zip(hands, jf["eef_fk"])], axis=0)
        self.fix = {"dq": np.asarray(jf["delta"], float), "dp": dp}
        self.rec["joints"]["hand_offset_mm"] = [round(float(x) * 1000, 2) for x in dp]
        self.rec["status"] = "joints fixed"
        return True

    def _image(self, frames: list, rgbs: list, t: int):
        """``arm_image``: True when the picture's shift and zoom were found, checked and are undone from now on;
        False when the gate did not fire (the camera explains the arm); None when the gate fired and no shift and
        zoom explains it (the caller goes on to the camera fit)."""
        a = time.perf_counter()
        one = bool(self.image.get("brighter"))
        masks = change_masks(rgbs, self.image["diff"], int(self.image["neighbours"]), brighter=one)
        rep = self.client.call({"op": "arm_cam", "mode": "gate_image", "suite": self.suite, "episode": self.episode,
                                "frames": [{**fr, "mask": m, "mask_one": one} for fr, m in zip(frames, masks)]})
        self.s += time.perf_counter() - a
        g, im = rep.get("gate") or {}, rep.get("image")
        self.rec.update(gate=g, at=int(t), s=round(self.s, 3))
        if not g.get("moved"):
            self.rec["status"] = "kept" if g.get("moved") is False else "undecided"
            return False
        self.rec["image"] = dict(im or {})
        if not (im or {}).get("accepted"):
            return None
        fix = {"zoom": float(im["zoom"]), "shift_px": [float(x) for x in im["shift_px"]],
               "centre_px": [float(x) for x in im["centre_px"]], "width": int(np.asarray(rgbs[0]).shape[1])}
        a = time.perf_counter()
        frames2 = [{**fr, "depth_u16": self.cam.image_depth(rgb, fr["K"], fix)} for fr, rgb in zip(frames, rgbs)]
        rep = self.client.call({"op": "arm_cam", "mode": "gate", "suite": self.suite, "episode": self.episode,
                                "frames": frames2})
        self.s += time.perf_counter() - a
        g2 = rep.get("gate") or {}
        self.rec["image"]["gate_after"] = g2
        self.rec["s"] = round(self.s, 3)
        if g2.get("moved") is not False:
            self._img_try = (fix, g2, g)  # kept for ``_approx``, should nothing else explain the arm
            return None  # with it undone the arm is still not where the camera puts it: not this
        self.cam.override["image"] = fix
        self.rec["status"] = "image fixed"
        return True

    def _approx(self) -> bool:
        """The last resort when the gate fired and neither camera fit was accepted: the picture's zoom and shift that
        failed their check, if they still brought the arm's cost far down (2026-11-15, the first pod run of the
        overlap fit: 20 of 48 S15 and S16 episodes ended "fit refused" with the arm 6 to 14 mm off after the undo,
        from 27 to 49 mm before it; left alone, the robot works from a picture it knows to be wrong). A camera that
        was moved a little looks like a zoom and a shift too: the undo is then approximate, and still far nearer."""
        t = getattr(self, "_img_try", None)
        if t is None or self.image is None:
            return False
        fix, g2, g = t
        c2, c1 = g2.get("cost_m"), g.get("cost_m")
        by = "approximately"
        if c2 is None or c1 is None or c2 > float(self.image["approx_cost_m"]) \
                or c2 > float(self.image["approx_ratio"]) * c1:
            # 2026-11-16, ``last_resort`` (off by default): the zoom and shift are used all the same unless the undo
            # made the arm fit worse (the check could not be made, or its cost stayed under the cost before), and
            # they lie inside the search's range. The fit was accepted by its own overlap test, nothing else
            # explained the arm, and the robot otherwise works from a picture it knows to be wrong (the rehearsal:
            # 12 of 63 S15 and S16 episodes ended "fit refused" with the zoom found within 0.02 of the others', and
            # none of the 12 was solved; three noise episodes "found" a zoom at the range's end, the arm worse after)
            lr = self.image.get("last_resort")
            if not lr:
                return False
            lr = {"zoom": (0.62, 1.38), "shift": 0.24, **(lr if isinstance(lr, dict) else {})}
            if c2 is not None and c1 is not None and c2 >= c1:
                return False
            w = float(fix.get("width") or 0.0)
            if not (float(lr["zoom"][0]) <= float(fix["zoom"]) <= float(lr["zoom"][1])) \
                    or (w > 0 and max(abs(float(x)) for x in fix["shift_px"]) > float(lr["shift"]) * w):
                return False
            by = "last resort"
        self.cam.override["image"] = fix
        self.rec["status"] = "image fixed"
        self.rec["by"] = by
        return True

    def _request(self, t: int) -> bool:
        self.done = True
        frames, rgbs = self.frames, self.rgbs
        hands = list(self.hands)
        self.frames, self.rgbs = [], []  # the depth frames are not kept
        if self.joints is not None and not self.camera_line:  # item 4, the depth line: the camera is right
            if self.rfit is not None and self._readings(frames, hands) > float(self.rfit["gap_mm"]):
                r = self._readings_route(t)
                if r is not None:
                    return r
            return self._joints("gate_joints", frames, hands, t)
        after_fit = self.joints is not None  # item 4: the joint fit follows a refused camera fit
        if self.joints is not None and self.joints.get("guard_mm") is not None:
            # item 4, the colour line's guard: the joint fit only when the readings contradict each other (then the
            # camera is taken as right, as on the depth line); never after a refused camera fit. Not fixed: on to the
            # camera route exactly as without ``arm_joints``.
            after_fit = False
            if self._readings(frames, hands) > float(self.joints["guard_mm"]):
                if self.rfit is not None:
                    r = self._readings_route(t)
                    if r is not None:
                        return r
                if self._joints("gate_joints", frames, hands, t):
                    return True
                self.rec["joints_status"] = self.rec.pop("status", None)
        if self.image is not None and len(rgbs) == len(frames):
            r = self._image(frames, rgbs, t)
            if r is not None:
                return r  # fixed (True), or the gate did not fire (False); None: on to the camera fit as before
        a = time.perf_counter()
        rep = self.client.call({"op": "arm_cam", "mode": "gate_fit", "suite": self.suite, "episode": self.episode,
                                "frames": frames})
        self.s += time.perf_counter() - a
        g = rep.get("gate") or {}
        f = rep.get("fit")
        self.rec.update(gate=g, at=int(t), s=round(self.s, 3))
        if not g.get("moved"):
            self.rec["status"] = "kept" if g.get("moved") is False else "undecided"
            return False
        self.rec["fit"] = {k: v for k, v in f.items() if k != "T_world_from_camera"}
        if not f.get("accepted"):
            self.rec["status"] = "fit refused"
            if self.second is None or len(rgbs) != len(frames):
                if self._approx():
                    return True
                # item 4: the camera fit cannot explain the arm; perhaps the joint readings are off
                return self._joints("joints", frames, hands, t) if after_fit else False
            # item 2b: the second try. The same frames with the second net's depth (at each frame's assumed camera);
            # the server's gate and fit again with its ``second`` overrides. Accepted: the second net's depth from now on.
            a = time.perf_counter()
            frames2 = [{**fr, "depth_u16": self.second(rgb, fr["K"])} for fr, rgb in zip(frames, rgbs)]
            rep = self.client.call({"op": "arm_cam", "mode": "gate_fit2", "suite": self.suite, "episode": self.episode,
                                    "frames": frames2})
            self.s += time.perf_counter() - a
            g2, f = rep.get("gate") or {}, rep.get("fit")
            self.rec["second"] = {"gate": g2, "fit": {k: v for k, v in (f or {}).items() if k != "T_world_from_camera"}}
            self.rec["s"] = round(self.s, 3)
            if not g2.get("moved") or not (f or {}).get("accepted"):
                if self._approx():
                    return True
                return self._joints("joints", frames, hands, t) if after_fit else False
            self.cam.override["net"] = 2
            self.rec["by"] = "second net"
        self.cam.override["T"] = np.asarray(f["T_world_from_camera"], float)
        if f.get("focal_scale") is not None:  # item 2b: the fit's focal length too (server option focal_scales)
            self.cam.override["focal_scale"] = float(f["focal_scale"])
        self.rec["status"] = "replaced"
        return True
