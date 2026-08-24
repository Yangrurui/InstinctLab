"""Launch mjlab's ``ViserPlayViewer``.

That is the viewer mjlab training already uses. Both adapters call this function; an engine
that has no MuJoCo ``Simulation`` is responsible for handing this a play env that does.
The mesh helpers below are catalog checks, not a second viewer.
"""

from __future__ import annotations

import numpy as np
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from instinctlab.sim.robot_spec import RobotSpec


def _checkpoint_step(name: str) -> int:
    """Return the numeric suffix used by ``model_<step>.pt`` checkpoints."""
    stem = Path(name).stem
    try:
        return int(stem.removeprefix("model_"))
    except ValueError:
        return -1


@dataclass(frozen=True)
class VisualMesh:
    """One visual mesh, in its parent body's frame."""

    body: str
    path: Path
    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]


def visual_meshes_from_mjcf(xml_path: Path) -> tuple[VisualMesh, ...]:
    """Read group-2 mesh geoms out of an MJCF file.

    Group 2 is how this project's G1 catalog marks visual meshes; collision capsules live in
    other groups and are not drawn. A geom without a group is accepted if it names a mesh, so a
    catalog that has not labelled groups still shows something.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    meshdir = xml_path.parent
    if compiler is not None and compiler.get("meshdir"):
        meshdir = (xml_path.parent / compiler.get("meshdir")).resolve()
    assets = {
        mesh.get("name"): meshdir / mesh.get("file")
        for mesh in root.findall("asset/mesh")
        if mesh.get("name") and mesh.get("file")
    }
    visuals: list[VisualMesh] = []
    for body in root.iter("body"):
        name = body.get("name")
        if not name:
            continue
        for geom in body.findall("geom"):
            mesh_name = geom.get("mesh")
            if not mesh_name or mesh_name not in assets:
                continue
            group = geom.get("group")
            if group is not None and group != "2":
                continue
            visuals.append(
                VisualMesh(
                    body=name,
                    path=assets[mesh_name],
                    pos=_xyz(geom.get("pos")),
                    quat_wxyz=_quat(geom.get("quat")),
                )
            )
    return tuple(visuals)


def visual_asset_path(robot: RobotSpec) -> Path:
    """The catalog file that carries visual meshes.

    Prefers MJCF over URDF because the G1 catalog's XML already names visual groups.
    """
    for suffix in (".xml", ".urdf"):
        for asset in robot.assets:
            path = Path(asset.path)
            if path.suffix == suffix:
                return path
    raise FileNotFoundError(f"{robot.name} has no MJCF or URDF asset to take meshes from")


def _xyz(text: str | None) -> tuple[float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0)
    values = tuple(float(part) for part in text.split())
    if len(values) != 3:
        raise ValueError(f"expected 3 numbers for pos, got {text!r}")
    return values  # type: ignore[return-value]


def _quat(text: str | None) -> tuple[float, float, float, float]:
    if not text:
        return (1.0, 0.0, 0.0, 0.0)
    values = tuple(float(part) for part in text.split())
    if len(values) != 4:
        raise ValueError(f"expected 4 numbers for quat, got {text!r}")
    return values  # type: ignore[return-value]


def _pin_velocity_command(env: Any, vx: float, vy: float, wz: float) -> None:
    """Write a base-velocity command the current engine will actually read.

    Isaac Lab stores it on ``vel_command_b`` and will overwrite heading/standing every step;
    mjlab stores it on ``_command`` and will resample when ``_time_left`` runs out. Matching by
    attribute, not by engine name, is what keeps this module off the denylist.
    """
    manager = getattr(env.unwrapped, "command_manager", None)
    if manager is None:
        return
    terms = getattr(manager, "_terms", None)
    if terms is None:
        return
    mapping = dict(terms) if not isinstance(terms, dict) else terms
    term = mapping.get("base_velocity")
    if term is None:
        return
    if hasattr(term, "vel_command_b"):
        term.is_heading_env[:] = False
        term.is_standing_env[:] = False
        term.vel_command_b[:, 0] = vx
        term.vel_command_b[:, 1] = vy
        term.vel_command_b[:, 2] = wz
        return
    if hasattr(term, "_command"):
        term._standing[:] = False
        term._heading[:] = False
        if hasattr(term, "_world"):
            term._world[:] = False
        if hasattr(term, "_forward"):
            term._forward[:] = False
        if hasattr(term, "_time_left"):
            term._time_left[:] = 1.0e9
        term._command[:, 0] = vx
        term._command[:, 1] = vy
        term._command[:, 2] = wz


def enable_depth_image_debug_vis(target: object) -> None:
    """Turn on InstinctMJ's ``debug_vis`` for the ``depth_image`` observation.

    Training keeps it off. The observation manager deep-copies the env cfg, so play
    has to patch the live ``term_cfg.params`` the manager actually ``**`` into the
    term — writing ``env.cfg`` after construction is a no-op.
    """
    env = getattr(target, "unwrapped", target)
    manager = getattr(env, "observation_manager", None)
    names = getattr(manager, "_group_obs_term_names", None)
    cfgs = getattr(manager, "_group_obs_term_cfgs", None)
    if isinstance(names, dict) and isinstance(cfgs, dict):
        for group, term_names in names.items():
            for name, term_cfg in zip(term_names, cfgs.get(group, ()), strict=False):
                if name != "depth_image":
                    continue
                params = getattr(term_cfg, "params", None)
                if isinstance(params, dict):
                    params["debug_vis"] = True
        return
    groups = getattr(target, "observations", None)
    if groups is None:
        return
    items = groups.values() if isinstance(groups, dict) else vars(groups).values()
    for group in items:
        terms = getattr(group, "terms", None)
        mapping = (
            terms
            if isinstance(terms, dict)
            else {name: value for name, value in vars(group).items() if hasattr(value, "params")}
        )
        term = mapping.get("depth_image")
        if term is None or not isinstance(getattr(term, "params", None), dict):
            continue
        term.params["debug_vis"] = True


def enable_pose_command_debug_vis(target: object) -> None:
    """Turn on InstinctMJ's parkour PLAY pose-target markers.

    InstinctMJ's play factory sets ``commands["base_velocity"].debug_vis = True`` and
    ``patch_vis = False``. Training leaves both off. The live command term holds the cfg
    the viewer actually reads.
    """
    env = getattr(target, "unwrapped", target)
    manager = getattr(env, "command_manager", None)
    terms = getattr(manager, "_terms", None)
    mapping = None
    if isinstance(terms, dict):
        mapping = terms
    elif terms is not None:
        try:
            mapping = dict(terms)
        except (TypeError, ValueError):
            mapping = None
    if mapping is not None:
        term = mapping.get("base_velocity")
        cfg = getattr(term, "cfg", None)
        if cfg is not None:
            cfg.debug_vis = True
            if hasattr(cfg, "patch_vis"):
                cfg.patch_vis = False
            setter = getattr(term, "set_debug_vis", None)
            if callable(setter):
                setter(True)
            return
    commands = getattr(target, "commands", None)
    if commands is None:
        return
    items = (
        commands
        if isinstance(commands, dict)
        else {name: value for name, value in vars(commands).items() if not name.startswith("_")}
    )
    term = items.get("base_velocity")
    if term is None:
        return
    term.debug_vis = True
    if hasattr(term, "patch_vis"):
        term.patch_vis = False


def _import_viser():
    """Import Viser, preferring the env install over Isaac Sim's bundled websockets.

    ``AppLauncher`` prepends Kit's ``pip_prebundle``. That ``websockets`` has no
    ``asyncio.server``. If Viser was imported before bootstrap it is already the right
    copy; otherwise evict the shadowed modules and put site-packages first.
    """
    import sys
    from pathlib import Path

    if "viser" in sys.modules:
        import viser

        return viser

    import site as site_mod

    preferred = [path for path in site_mod.getsitepackages() if (Path(path) / "websockets" / "asyncio").is_dir()]
    prebundle = [path for path in sys.path if "pip_prebundle" in path]
    sys.path = preferred + [path for path in sys.path if path not in preferred and path not in prebundle] + prebundle
    for name in [module for module in sys.modules if module == "websockets" or module.startswith("websockets.")]:
        del sys.modules[name]
    import viser

    return viser


def _depth_frame_to_rgb(frame: torch.Tensor, *, scale: int = 5) -> np.ndarray:
    """Latest policy depth for one env: black=near, white=far."""
    peak = float(frame.max())
    if peak <= 0:
        height, width = frame.shape
        return np.zeros((height * scale, width * scale, 3), dtype=np.uint8)
    img = (frame * 255.0 / peak).detach().cpu().numpy().astype("uint8")
    if scale != 1:
        import cv2

        img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_AREA)
    return np.repeat(img[..., None], 3, axis=-1)


def _update_depth_panel(state: dict[str, Any]) -> None:
    frames = state.get("frames")
    handle = state.get("handle")
    env_idx = state.get("env_idx")
    if frames is None or handle is None or env_idx is None:
        return
    idx = max(0, min(int(env_idx()), frames.shape[0] - 1))
    handle.image = _depth_frame_to_rgb(frames[idx])


def play_with_viser(
    env: Any,
    policy: Callable[[Any], Any],
    robot: RobotSpec | None = None,
    *,
    port: int = 8080,
    reload_policy: Callable[[str], Callable[[Any], Any]] | None = None,
    checkpoint_dir: Path | None = None,
) -> None:
    """Run ``policy`` in ``env`` through mjlab's ``ViserPlayViewer``.

    Depth images come from InstinctMJ's ``delayed_visualizable_image`` ``debug_vis`` path.
    Viser shows the selected env's latest frame; cv2 keeps InstinctMJ's multi-env mosaic.
    """
    del robot
    from mjlab.viewer import ViserPlayViewer
    from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago

    from instinctlab.mdp.observations import set_debug_image_sink

    class _InstinctViserPlayViewer(ViserPlayViewer):
        def __init__(self, *args: Any, depth_panel_state: dict[str, Any] | None = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._depth_panel_state = depth_panel_state

        def setup(self) -> None:
            super().setup()
            if self._depth_panel_state is not None:
                self._depth_panel_state["env_idx"] = lambda: self._scene.env_idx

        def _update_env_dependent_plots(self) -> None:
            super()._update_env_dependent_plots()
            if self._depth_panel_state is not None:
                _update_depth_panel(self._depth_panel_state)

    viser = _import_viser()
    manager = None
    if reload_policy is not None and checkpoint_dir is not None:
        directory = Path(checkpoint_dir)

        def fetch_available() -> list[tuple[str, str]]:
            now = time.time()
            entries: list[tuple[str, str, int]] = []
            for file in sorted(directory.glob("model_*.pt")):
                step = _checkpoint_step(file.name)
                entries.append((file.name, format_time_ago(int(now - file.stat().st_mtime)), step))
            entries.sort(key=lambda item: item[2])
            return [(name, ago) for name, ago, _ in entries]

        available = list(directory.glob("model_*.pt"))
        current = max(available, key=lambda path: _checkpoint_step(path.name), default=None)
        if current is not None:
            manager = CheckpointManager(
                current_name=current.name,
                fetch_available=fetch_available,
                load_checkpoint=lambda name: reload_policy(str(directory / name)),
                run_name=directory.name,
            )

    server = viser.ViserServer(label="instinctlab", port=port)
    print(f"[INFO] Viser viewer: http://0.0.0.0:{port}", flush=True)
    depth_panel_state: dict[str, Any] = {"frames": None, "env_idx": lambda: 0}
    with server.gui.add_folder("Depth image"):
        depth_handle = server.gui.add_image(
            image=np.zeros((90, 160, 3), dtype=np.uint8),
            label="depth_image",
            format="jpeg",
        )
    depth_panel_state["handle"] = depth_handle

    def _show_depth(_window_name: str, frames: Any) -> None:
        depth_panel_state["frames"] = frames
        _update_depth_panel(depth_panel_state)

    set_debug_image_sink(_show_depth)
    enable_depth_image_debug_vis(env)
    enable_pose_command_debug_vis(env)
    try:
        _InstinctViserPlayViewer(
            env,
            policy,
            viser_server=server,
            checkpoint_manager=manager,
            depth_panel_state=depth_panel_state,
        ).run()
    finally:
        set_debug_image_sink(None)
        server.stop()
