"""The colour-only line's camera tool (runner side; Pro2.0 gives colour images and proprioception, no depth or camera).

``dev_episode`` uses this tool when ``params["line"] == "colour"``. The robot's looks then use the episode's own
agentview image (so every perturbation that acts on the images reaches them) instead of the simulator's render. The rest
of a frame is filled by:

* ``params["colour"]["depth"]``: ``"net"`` (the depth net of ``colour_net``, at ``params["colour"]["net"]``), or
  ``"sim"`` (development only: the simulator's depth, to check the plumbing);
* ``params["colour"]["calib"]``: ``"stored"`` (the agentview camera per suite, measured once on the training split and
  stored in ``colour_cameras.json``), ``"arm"`` (the stored camera until the robot's own arm shows it moved, then the
  camera fitted from the arm: ``arm_calib.py`` sets ``look.override["T"]``), or ``"sim"`` (development only: the
  simulator's live camera).

A run with depth ``net`` and calib ``stored`` holds no simulator handle in this tool: only the images.

The long suite's tables (``long_paths.CELLS``; ``dev_episode`` passes the table's key as ``suite``) have their stored
cameras in ``colour_cameras_long.json``, measured the same way (``colour_long.py cameras``). They are a second file
because ``colour_cameras.json``'s bytes are named in the frozen world-2 registration and ``world2.py`` builds its moved
cameras from them. ``stored_cameras`` gives both; a key in both files is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEPTH_SCALE_MM = 0.1
CAMERAS_FILE = Path(__file__).resolve().parent / "colour_cameras.json"
LONG_CAMERAS_FILE = Path(__file__).resolve().parent / "colour_cameras_long.json"  # the long suite's tables
IMAGE_KEYS = {"agentview": "agentview_image", "robot0_eye_in_hand": "robot0_eye_in_hand_image"}
_NETS: dict = {}  # one loaded net per process


def options(params: dict) -> dict:
    o = dict(params.get("colour") or {})
    o.setdefault("depth", "net")
    o.setdefault("calib", "stored")
    return o


def needs_sim(params: dict) -> bool:
    o = options(params)
    return o["depth"] == "sim" or o["calib"] == "sim"


def stored_cameras() -> dict:
    """``{key: {"fovy_deg", "T_world_from_camera"}}``: the three suites' cameras, and the long suite's tables' when
    their file is there (the suites' entries are exactly ``colour_cameras.json``'s)."""
    cams = json.loads(CAMERAS_FILE.read_text())["cameras"]
    if LONG_CAMERAS_FILE.is_file():
        more = json.loads(LONG_CAMERAS_FILE.read_text())["cameras"]
        both = sorted(set(cams) & set(more))
        if both:
            raise ValueError(f"{LONG_CAMERAS_FILE.name} repeats keys of {CAMERAS_FILE.name}: {both}")
        cams = {**cams, **more}
    return cams


def _net(o: dict):
    key = (o.get("net"), int(o.get("threads", 2)))
    if key not in _NETS:
        from rowant_lab.pro2.colour_net import DepthNet

        _NETS[key] = DepthNet(o["net"], threads=int(o.get("threads", 2)), device=o.get("device", "cpu"))
    return _NETS[key]


def _net2(o: dict):
    """Item 2b, off by default: the second depth net (``colour.net2``), used only after the arm's camera check refused
    the first net's fit and then accepted a fit on the second net's depth (``arm_calib``). Needs depth net, calib arm."""
    if not o.get("net2") or o["depth"] != "net" or o["calib"] != "arm":
        return None
    return _net({**o, "net": o["net2"]})


def describe(params: dict) -> dict:
    o = options(params)
    d = {"depth": o["depth"], "calib": o["calib"]}
    if o["depth"] == "net":
        d["net_sha256"] = _net(o).sha256
    if _net2(o) is not None:
        d["net2_sha256"] = _net2(o).sha256
    return d


def episode_images(obs: dict) -> dict:
    """The episode's images in OpenCV row order (row 0 = top): the images arrive upside down (checked on 2026-11-09
    against the simulator's render: flipped, every pixel equal)."""
    return {cam: np.ascontiguousarray(obs[k][::-1]) for cam, k in IMAGE_KEYS.items() if k in obs}


def intrinsics(fovy_deg: float, width: int) -> np.ndarray:
    f = 0.5 * width / np.tan(np.radians(float(fovy_deg)) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, width / 2.0], [0.0, 0.0, 1.0]])


def undo_image(rgb: np.ndarray, fix: dict) -> np.ndarray:
    """The picture the assumed camera would have taken, from one that came zoomed and shifted (2026-11-15,
    ``arm_calib``'s ``arm_image``): ``out[m] = rgb[zoom (m - c) + c + shift]``, bilinear; a pixel from outside the
    picture takes the nearest edge's. ``fix``: ``zoom``, ``shift_px`` (x, y), ``centre_px`` and the ``width`` they
    were measured at (scaled to this picture's)."""
    h, w = rgb.shape[:2]
    k = w / float(fix.get("width") or w)
    z = float(fix["zoom"])

    def axis(n: int, c: float, t: float):
        x = np.clip(z * (np.arange(n, dtype=np.float64) - c * k) + c * k + t * k, 0.0, n - 1.0)
        i0 = np.minimum(np.floor(x).astype(np.int64), n - 1)
        return i0, np.minimum(i0 + 1, n - 1), (x - i0).astype(np.float32)

    u0, u1, fu = axis(w, float(fix["centre_px"][0]), float(fix["shift_px"][0]))
    v0, v1, fv = axis(h, float(fix["centre_px"][1]), float(fix["shift_px"][1]))
    a = rgb.astype(np.float32)
    top = a[v0][:, u0] * (1.0 - fu)[None, :, None] + a[v0][:, u1] * fu[None, :, None]
    bot = a[v1][:, u0] * (1.0 - fu)[None, :, None] + a[v1][:, u1] * fu[None, :, None]
    out = top * (1.0 - fv)[:, None, None] + bot * fv[:, None, None]
    return np.clip(np.rint(out), 0, 255).astype(rgb.dtype)


def make_tool(images: dict, suite: str, params: dict, sim_look=None):
    """``images``: the episode's latest images, updated by the runner after every step. Returns the ``look`` the robot's
    perception tool calls (``camera``, ``width``, ``step``) with the camera tool's frame keys."""
    o = options(params)
    if needs_sim(params) and sim_look is None:
        raise ValueError("this colour-line option reads the simulator, and no simulator look was given")
    net = _net(o) if o["depth"] == "net" else None
    net2 = _net2(o)
    stored = None if o["calib"] == "sim" else stored_cameras()[suite]
    timing: list = []
    override: dict = {"T": None}  # calib "arm": the camera fitted from the arm, once it replaces the stored one
    # (and, item 2b, "focal_scale": the fitted focal length as a factor on the stored one; "net": 2 once the fit on the
    # second net's depth was accepted)

    def look(camera: str, width: int, step: int) -> dict:
        rgb = images[camera]
        if rgb.shape != (width, width, 3):
            raise ValueError(f"the episode image is {rgb.shape}, the look asks for {width} px")
        if override.get("image") is not None and camera == "agentview":
            rgb = undo_image(rgb, override["image"])  # the picture came zoomed or shifted: as the camera would take it
        sim = sim_look(camera, width, step) if sim_look is not None else None
        if stored is None:
            K, T = sim["K"], sim["T_world_from_camera"]
        else:
            K = intrinsics(stored["fovy_deg"], width)
            if override.get("focal_scale") is not None:
                K[0, 0] *= float(override["focal_scale"])
                K[1, 1] *= float(override["focal_scale"])
            T = np.asarray(stored["T_world_from_camera"] if override["T"] is None else override["T"], dtype=np.float64)
        use = net2 if (net2 is not None and override.get("net") == 2) else net
        dep = sim["depth_u16"] if use is None else use.depth_u16(rgb, K)  # K: the camera the tool assumes (rays nets)
        fr = {"camera": camera, "rgb": rgb, "depth_u16": dep, "depth_scale_mm": DEPTH_SCALE_MM, "K": K,
              "T_world_from_camera": T, "width": int(width), "height": int(width), "step": int(step)}
        reg = override.get("register")
        if reg is not None and stored is not None and camera == "agentview":
            # stream PATH, round 3 (absent unless ``arm_cam_robot.register``): a camera that moved in the middle of
            # the episode, read from the still scene (``cam_register``). None: the frame as it is. Else the camera
            # every later frame uses, and (its form ``scene``) this frame's depth corrected to fit that camera.
            got = reg.look(fr)
            if got is not None and got.get("T") is not None:
                override["T"] = np.asarray(got["T"], dtype=np.float64)
                fr["T_world_from_camera"] = override["T"]
            if got is not None and got.get("depth_u16") is not None:
                fr["depth_u16"] = got["depth_u16"]
        return fr

    def image_depth(rgb, K, fix):
        """The net's depth of a stored picture with ``fix`` undone (``arm_calib`` checks a fix before it is used)."""
        return net.depth_u16(undo_image(rgb, fix), K)

    def colour(camera: str):
        """The episode's latest colour picture of ``camera`` as it came: no depth, no camera (the robot's wrist looks
        in colour, ``robot/wrist2d.py``)."""
        return images.get(camera)

    look.colour = colour
    look.timing = timing
    look.override = override
    look.image_depth = image_depth if (net is not None and stored is not None) else None
    look.net2_depth = None if net2 is None else net2.depth_u16  # arm_calib's second try
    look.wrist_ok = False  # item 3: no depth and no pose for the wrist camera here yet (the robot's close looks skip it)
    return look
