#!/usr/bin/env python3
"""Build release wheels and verify isolated engine-install combinations.

The probe runs from a temporary directory, installs only built wheels, and
filters entry-point discovery to the active virtual environment. Using
``--system-site-packages`` supplies the large simulator SDKs already installed
on a development machine without letting editable InstinctLab distributions
affect discovery.

This check covers wheel contents, backend discovery, discovery-time SDK import
isolation, native asset conformance, and the engine/task contract report.
The default matrix also installs, exercises, and uninstalls a repository-external
fixture across every public asset, actuator, sensor, and terrain extension seam.
Pass ``--live-extension`` to additionally construct and step the fixture's real
native actuator on both GPU backends before uninstalling its wheel.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS = (
    "instinctlab_engine",
    "instinctlab_engine_isaacsim",
    "instinctlab_engine_mjlab",
    "instinctlab",
)
EXTENSION_FIXTURE = "instinctlab-extension-fixture"
MATRICES = {
    "core": ("instinctlab-engine-core",),
    "isaacsim": (
        "instinctlab-engine-core",
        "instinctlab-engine-isaacsim",
        "instinctlab",
    ),
    "mjlab": (
        "instinctlab-engine-core",
        "instinctlab-engine-mjlab",
        "instinctlab",
    ),
    "both": (
        "instinctlab-engine-core",
        "instinctlab-engine-isaacsim",
        "instinctlab-engine-mjlab",
        "instinctlab",
    ),
}
EXPECTED_ENGINES = {
    "core": (),
    "isaacsim": ("isaacsim",),
    "mjlab": ("mjlab",),
    "both": ("isaacsim", "mjlab"),
}

EXTENSION_PROBE = r"""
import importlib.metadata as metadata
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

engine_name = sys.argv[1]
environment_root = Path(sys.prefix).resolve()

def belongs_to_environment(distribution):
    return Path(distribution.locate_file("")).resolve().is_relative_to(environment_root)

original_entry_points = metadata.entry_points

def isolated_entry_points(*args, **kwargs):
    return [
        entry_point
        for entry_point in original_entry_points(*args, **kwargs)
        if belongs_to_environment(entry_point.dist)
    ]

metadata.entry_points = isolated_entry_points

from instinctlab_engine import adapter
from instinctlab_engine.preflight import require_preflight
from instinctlab_engine.bridge.robot import joint_effort_limits, joint_stiffness_groups
from instinctlab_engine.registry import TERRAIN_EXTENSIONS
from instinctlab_engine.sensors import NativeSensorBuildContext, native_sensor_builder
from instinctlab_engine.spec import (
    AgentSpec,
    MdpSpec,
    NativeSensorRef,
    SceneSpec,
    SimSpec,
    TaskSpec,
    TerrainSpec,
)

selected = adapter(engine_name)
robot = selected.robot_spec("fixture_bot/v1")
sensor_ref = NativeSensorRef(
    name="imu",
    kind="fixture.imu",
    attach="base",
    update_period=0.02,
    latency=0.02,
    history_length=2,
    partial_reset=True,
)
task = TaskSpec(
    task_id=f"Fixture-{engine_name}",
    robot=robot,
    scene=SceneSpec(
        terrain=TerrainSpec(kind="fixture_plane"),
        native_sensors=(sensor_ref,),
    ),
    sim=SimSpec(physics_dt=0.01, decimation=2, episode_length_s=1.0),
    mdp=MdpSpec(),
    agent=AgentSpec(runner="builtins:object"),
    engines=(engine_name,),
)
report = require_preflight(task, engine_name, selected_adapter=selected)
if report["selected_components"] != {
    "asset_id": "fixture_bot/v1",
    "actuator_model_ids": ["fixture.stateful.v1"],
    "actuator_groups": [
        {"name": "joint", "model_id": "fixture.stateful.v1"}
    ],
    "sensor_kinds": ["fixture.imu"],
    "terrain_kind": "fixture_plane",
    "sub_terrain_kinds": [],
}:
    raise AssertionError(report["selected_components"])

conformance = selected.asset_conformance("fixture_bot/v1")
if conformance["actuator_groups"] != [{
    "name": "joint",
    "model_id": "fixture.stateful.v1",
    "selectors": ["joint"],
    "joint_names": ["joint"],
}]:
    raise AssertionError(f"native joint selector lost DFS coverage: {conformance}")

# The isolated wheel probe deliberately stays SDK-free.  Its deterministic
# stand-in exercises the public runtime bridge; --live-extension exercises the
# selected SDK's real actuator class and environment state in a separate gate.
from instinctlab_extension_fixture.runtime import StatefulActuatorCfgBase

actuator = StatefulActuatorCfgBase().build(num_envs=2)
if actuator.compute([2.0, -2.0]) != [0.0, 0.0]:
    raise AssertionError("stateful actuator did not apply its one-step delay")
if actuator.compute([2.0, -2.0]) != [3.0, -3.0]:
    raise AssertionError("stateful actuator clipping/formula is wrong")
actuator.reset([1])
if actuator.compute([0.0, 0.0]) != [3.0, 0.0]:
    raise AssertionError("stateful actuator partial reset leaked across environments")

effort = torch.tensor([[2.0], [3.0]])
data_kwargs = {
    "joint_vel": torch.tensor([[4.0], [5.0]]),
    "qfrc_actuator" if engine_name == "mjlab" else "applied_torque": effort,
}
asset = SimpleNamespace(
    actuators={"joint": actuator},
    data=SimpleNamespace(**data_kwargs),
    joint_names=("joint",),
    num_joints=1,
)
env = SimpleNamespace(scene={"robot": asset})
stiffness = list(
    joint_stiffness_groups(
        env, asset, [0], requesting_term="fixture power reward"
    )
)
if stiffness != [((0,), 2.0)]:
    raise AssertionError(f"runtime stiffness contract is wrong: {stiffness}")
limits = joint_effort_limits(env, asset, [0])
torch.testing.assert_close(limits, torch.full((2, 1), 3.0))

from instinctlab.tasks.parkour.mdp.rewards import (
    applied_torque_limits_by_ratio,
    motors_power_square,
)

asset_cfg = SimpleNamespace(name="robot", joint_ids=[0])
power_term = motors_power_square(
    SimpleNamespace(params={"asset_cfg": asset_cfg}),
    env,
)
actuator.stiffness = 4.0
power = power_term(env, asset_cfg=asset_cfg)
torch.testing.assert_close(power, torch.tensor([4.0, 14.0625]))
limit_penalty = applied_torque_limits_by_ratio(
    env, asset_cfg=asset_cfg, limit_ratio=0.8
)
torch.testing.assert_close(limit_penalty, torch.tensor([0.0, 0.36]))

sensor = native_sensor_builder(engine_name, sensor_ref)(
    sensor_ref,
    NativeSensorBuildContext(
        engine=engine_name,
        robot=robot,
        sensor_period=0.02,
        profile={},
        num_envs=2,
    ),
)
if sensor.tick([1.0, 10.0], 0.02) != [0.0, 0.0]:
    raise AssertionError("sensor latency was not applied")
if sensor.tick([2.0, 20.0], 0.04) != [1.0, 10.0]:
    raise AssertionError("sensor history timing is wrong")
sensor.reset([1])
if sensor.tick([3.0, 30.0], 0.06) != [2.0, 0.0]:
    raise AssertionError("sensor partial reset leaked across environments")

terrain = TERRAIN_EXTENSIONS.terrain(engine_name, "fixture_plane")(
    task.scene.terrain, {}
)
if terrain["engine"] != engine_name:
    raise AssertionError("wrong native terrain implementation resolved")

fixture_groups = {
    item["group"]
    for item in report["providers"]
    if item["distribution"] == "instinctlab-extension-fixture"
}
expected_groups = {
    "instinctlab.assets",
    "instinctlab.actuators",
    "instinctlab.sensors",
    "instinctlab.terrains",
}
if not expected_groups <= fixture_groups:
    raise AssertionError(f"fixture provenance groups are {sorted(fixture_groups)}")

other_engine = "mjlab" if engine_name == "isaacsim" else "isaacsim"
unexpected_modules = {
    f"instinctlab_extension_fixture.{other_engine}_asset",
    f"instinctlab_extension_fixture.{other_engine}_actuator",
    f"instinctlab_extension_fixture.{other_engine}_implementation",
}
if unexpected_modules & set(sys.modules):
    raise AssertionError(
        f"unselected fixture implementation imported: {unexpected_modules & set(sys.modules)}"
    )
unselected_sdk_roots = (
    {"mjlab", "mujoco", "mujoco_warp"}
    if engine_name == "isaacsim"
    else {"isaaclab", "omni"}
)
loaded_roots = {name.split(".", 1)[0] for name in sys.modules}
if unselected_sdk_roots & loaded_roots:
    raise AssertionError(
        f"unselected SDK imported: {sorted(unselected_sdk_roots & loaded_roots)}"
    )

print(json.dumps({
    "engine": engine_name,
    "asset": report["asset"],
    "actuator_output": actuator.applied_effort,
    "power_reward": power.tolist(),
    "limit_reward": limit_penalty.tolist(),
    "sensor_timestamp": sensor.timestamp,
    "fixture_provider_groups": sorted(fixture_groups),
}, sort_keys=True))
"""

UNINSTALL_PROBE = r"""
import importlib.metadata as metadata
import sys
from pathlib import Path

environment_root = Path(sys.prefix).resolve()

def belongs_to_environment(distribution):
    return Path(distribution.locate_file("")).resolve().is_relative_to(environment_root)

original_entry_points = metadata.entry_points

def isolated_entry_points(*args, **kwargs):
    return [
        entry_point
        for entry_point in original_entry_points(*args, **kwargs)
        if belongs_to_environment(entry_point.dist)
    ]

metadata.entry_points = isolated_entry_points

from instinctlab.tasks import registry
from instinctlab_engine import adapter, names
from instinctlab_engine.actuators import ACTUATORS
from instinctlab_engine.assets import asset_packages
from instinctlab_engine.preflight import require_preflight
from instinctlab_engine.registry import TERRAIN_EXTENSIONS
from instinctlab_engine.sensors import SENSORS

if names() != ("isaacsim", "mjlab"):
    raise AssertionError(names())
if "fixture_bot" in asset_packages():
    raise AssertionError("uninstalled fixture asset is still discoverable")
for engine_name in names():
    if "fixture.stateful.v1" in ACTUATORS.registrations(engine_name):
        raise AssertionError("uninstalled fixture actuator is still discoverable")
    if "fixture.imu" in SENSORS.registrations(engine_name):
        raise AssertionError("uninstalled fixture sensor is still discoverable")
    if "fixture_plane" in TERRAIN_EXTENSIONS.terrain_kinds(engine_name):
        raise AssertionError("uninstalled fixture terrain is still discoverable")
    selected = adapter(engine_name)
    task_id = "Instinct-Velocity-Flat-G1"
    robot = selected.robot_spec(registry.asset_id(task_id))
    task = registry.spec(task_id, robot)
    require_preflight(task, engine_name, selected_adapter=selected)
print("extension uninstall preserved both built-in backends")
"""

PROBE = r"""
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

expected_distributions = set(sys.argv[1].split(","))
expected_engines = tuple(filter(None, sys.argv[2].split(",")))
environment_root = Path(sys.prefix).resolve()

def belongs_to_environment(distribution):
    return Path(distribution.locate_file("")).resolve().is_relative_to(environment_root)

original_entry_points = metadata.entry_points

def isolated_entry_points(*args, **kwargs):
    return [
        entry_point
        for entry_point in original_entry_points(*args, **kwargs)
        if belongs_to_environment(entry_point.dist)
    ]

metadata.entry_points = isolated_entry_points
installed = {
    distribution.metadata["Name"]
    for distribution in metadata.distributions()
    if belongs_to_environment(distribution)
    and distribution.metadata["Name"].lower().startswith("instinctlab")
}
if installed != expected_distributions:
    raise AssertionError(f"installed distributions {installed}, expected {expected_distributions}")

sdk_roots = {"isaaclab", "mjlab", "mujoco", "mujoco_warp", "omni"}
before = set(sys.modules)
from instinctlab_engine import adapter, names

discovered = names()
if discovered != expected_engines:
    raise AssertionError(f"discovered engines {discovered}, expected {expected_engines}")
imported_during_discovery = {
    name.split(".", 1)[0] for name in set(sys.modules) - before
} & sdk_roots
if imported_during_discovery:
    raise AssertionError(
        f"engine discovery imported simulator SDK modules: {sorted(imported_during_discovery)}"
    )

materialized = {}
for engine in expected_engines:
    selected = adapter(engine)
    from instinctlab.tasks import registry

    task_id = "Instinct-Velocity-Flat-G1"
    robot = selected.robot_spec(registry.asset_id(task_id))
    native_path = Path(robot.asset_for(engine).path)
    if not native_path.is_file():
        raise AssertionError(f"wheel omitted native asset {native_path}")
    task = registry.spec(task_id, robot)
    report = selected.contract_report(task)
    if report["missing"]:
        raise AssertionError(f"{engine} contract has missing entries: {report['missing']}")
    materialized[engine] = {
        "task_id": task.task_id,
        "asset": str(native_path),
        "terms": len(task.mdp.terms()),
    }

print(json.dumps({"engines": discovered, "materialized": materialized}, sort_keys=True))
"""


def _run(command: list[str], *, cwd: Path) -> None:
    display = [
        "<inline-python>" if "\n" in argument else argument for argument in command
    ]
    print("+", " ".join(display), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _copy_projects(root: Path) -> list[Path]:
    copied: list[Path] = []
    ignored = shutil.ignore_patterns(
        "build",
        "dist",
        "*.egg-info",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
    )
    for project in PROJECTS:
        source = REPO_ROOT / "source" / project
        destination = root / "projects" / project
        shutil.copytree(source, destination, ignore=ignored)
        copied.append(destination)
    fixture = root / "projects" / "external_extension"
    shutil.copytree(
        REPO_ROOT / "tests" / "fixtures" / "external_extension",
        fixture,
        ignore=ignored,
    )
    copied.append(fixture)
    return copied


def _build_wheels(root: Path) -> dict[str, Path]:
    wheel_dir = root / "wheels"
    wheel_dir.mkdir()
    projects = _copy_projects(root)
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            *(str(project) for project in projects),
        ],
        cwd=root,
    )
    wheels: dict[str, Path] = {}
    for wheel in wheel_dir.glob("*.whl"):
        distribution = wheel.name.split("-0.1.0-", 1)[0].replace("_", "-")
        wheels[distribution] = wheel
    expected = {*MATRICES["both"], EXTENSION_FIXTURE}
    if set(wheels) != expected:
        raise RuntimeError(
            f"built wheels {sorted(wheels)}, expected {sorted(expected)}"
        )
    return wheels


def _verify_matrix(
    root: Path,
    matrix: str,
    wheels: dict[str, Path],
) -> None:
    environment = root / "environments" / matrix
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    python = environment / "bin" / "python"
    distributions = MATRICES[matrix]
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--ignore-installed",
            "--no-deps",
            *(str(wheels[name]) for name in distributions),
        ],
        cwd=root,
    )
    _run(
        [
            str(python),
            "-I",
            "-c",
            PROBE,
            ",".join(distributions),
            ",".join(EXPECTED_ENGINES[matrix]),
        ],
        cwd=root,
    )


def _verify_extension(
    root: Path,
    wheels: dict[str, Path],
    *,
    live: bool,
    device: str,
) -> None:
    environment = root / "environments" / "extension"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    python = environment / "bin" / "python"
    distributions = (*MATRICES["both"], EXTENSION_FIXTURE)
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--ignore-installed",
            "--no-deps",
            *(str(wheels[name]) for name in distributions),
        ],
        cwd=root,
    )
    for engine_name in EXPECTED_ENGINES["both"]:
        _run(
            [str(python), "-I", "-c", EXTENSION_PROBE, engine_name],
            cwd=root,
        )
    if live:
        for engine_name in EXPECTED_ENGINES["both"]:
            command = [
                str(python),
                "-I",
                "-m",
                "instinctlab_extension_fixture.live_probe",
                "--engine",
                engine_name,
                "--device",
                device,
            ]
            if engine_name == "isaacsim":
                command.append("--headless")
            _run(command, cwd=root)
    _run(
        [
            str(python),
            "-m",
            "pip",
            "uninstall",
            "--yes",
            EXTENSION_FIXTURE,
        ],
        cwd=root,
    )
    _run([str(python), "-I", "-c", UNINSTALL_PROBE], cwd=root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        action="append",
        choices=tuple(MATRICES),
        help="Matrix to run; repeat as needed. The default runs all four.",
    )
    parser.add_argument(
        "--extension",
        action="store_true",
        help="Also install, exercise, and uninstall the external extension fixture wheel.",
    )
    parser.add_argument(
        "--live-extension",
        action="store_true",
        help=(
            "Construct and step the external wheel's native actuator on both "
            "backends; implies --extension and requires a GPU."
        ),
    )
    parser.add_argument(
        "--live-device",
        default="cuda:0",
        help="GPU device used by --live-extension (default: cuda:0).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary wheel and environment directory for inspection.",
    )
    args = parser.parse_args()

    selected = tuple(args.matrix or MATRICES)
    verify_extension = args.extension or args.live_extension or args.matrix is None
    if args.keep_temp:
        root = Path(tempfile.mkdtemp(prefix="instinctlab-wheel-matrix-"))
        print(f"Temporary files: {root}", flush=True)
        wheels = _build_wheels(root)
        for matrix in selected:
            _verify_matrix(root, matrix, wheels)
        if verify_extension:
            _verify_extension(
                root,
                wheels,
                live=args.live_extension,
                device=args.live_device,
            )
    else:
        with tempfile.TemporaryDirectory(prefix="instinctlab-wheel-matrix-") as temp:
            root = Path(temp)
            wheels = _build_wheels(root)
            for matrix in selected:
                _verify_matrix(root, matrix, wheels)
            if verify_extension:
                _verify_extension(
                    root,
                    wheels,
                    live=args.live_extension,
                    device=args.live_device,
                )

    suffix = " plus external extension" if verify_extension else ""
    print(f"Verified wheel matrices: {', '.join(selected)}{suffix}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
