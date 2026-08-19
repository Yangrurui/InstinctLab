"""Launch mjlab's ``ViserPlayViewer``.

That is the viewer mjlab training already uses. Both adapters call this function; an engine
that has no MuJoCo ``Simulation`` is responsible for handing this a play env that does.
The mesh helpers below are catalog checks, not a second viewer.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from instinctlab.sim.robot_spec import RobotSpec


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


def play_with_viser(
    env: Any,
    policy: Callable[[Any], Any],
    robot: RobotSpec | None = None,
    *,
    port: int = 8080,
    reload_policy: Callable[[str], Callable[[Any], Any]] | None = None,
    checkpoint_dir: Path | None = None,
) -> None:
    """Run ``policy`` in ``env`` through mjlab's ``ViserPlayViewer``."""
    del robot
    from mjlab.viewer import ViserPlayViewer
    from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago

    viser = _import_viser()
    manager = None
    if reload_policy is not None and checkpoint_dir is not None:
        directory = Path(checkpoint_dir)

        def fetch_available() -> list[tuple[str, str]]:
            now = time.time()
            entries: list[tuple[str, str, int]] = []
            for file in sorted(directory.glob("model_*.pt")):
                try:
                    step = int(file.stem.split("_")[1])
                except (IndexError, ValueError):
                    step = 0
                entries.append((file.name, format_time_ago(int(now - file.stat().st_mtime)), step))
            entries.sort(key=lambda item: item[2])
            return [(name, ago) for name, ago, _ in entries]

        current = next(iter(reversed(sorted(directory.glob("model_*.pt")))), None)
        if current is not None:
            manager = CheckpointManager(
                current_name=current.name,
                fetch_available=fetch_available,
                load_checkpoint=lambda name: reload_policy(str(directory / name)),
                run_name=directory.name,
            )

    server = viser.ViserServer(label="instinctlab", port=port)
    print(f"[INFO] Viser viewer: http://0.0.0.0:{port}", flush=True)
    try:
        ViserPlayViewer(env, policy, viser_server=server, checkpoint_manager=manager).run()
    finally:
        server.stop()
