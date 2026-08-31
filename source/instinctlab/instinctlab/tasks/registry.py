"""Lazy, immutable task registrations for every supported engine.

Gym registration imports native environment configuration while listing tasks.
This catalog stores only engine-neutral strings, so discovery remains cheap and
does not import a task factory, robot asset, or simulator SDK.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType
from typing import TYPE_CHECKING

from instinctlab_engine.spec import freeze_task_spec

if TYPE_CHECKING:
    from instinctlab_engine.spec import TaskSpec
    from instinctlab_engine.spec.robot import RobotSpec


@dataclass(frozen=True, slots=True)
class TaskRegistration:
    """Complete routing metadata for one public task id."""

    factory_path: str
    asset_id: str
    checkpoint_task_id: str | None = None

    def __post_init__(self) -> None:
        module, separator, attribute = self.factory_path.partition(":")
        if not separator or not module or not attribute.isidentifier():
            raise ValueError(
                f"invalid task factory path {self.factory_path!r}; expected 'module:factory'"
            )
        package, separator, variant = self.asset_id.partition("/")
        if not separator or not package.isidentifier() or not variant:
            raise ValueError(
                f"invalid task asset id {self.asset_id!r}; expected 'package/variant'"
            )


_REGISTRATIONS = {
    "Instinct-Velocity-Flat-G1": TaskRegistration(
        factory_path="instinctlab.tasks.locomotion.config.g1:flat_g1",
        asset_id="unitree_g1/popsicle_torsobase_v1",
    ),
    "Instinct-Velocity-Flat-G1-15DoF": TaskRegistration(
        factory_path="instinctlab.tasks.locomotion.config.g1_15dof:flat_g1_15dof",
        asset_id="unitree_g1/popsicle_torsobase_locked_arms_v1",
    ),
    "Instinct-Velocity-Rough-G1": TaskRegistration(
        factory_path="instinctlab.tasks.locomotion.config.g1:rough_g1",
        asset_id="unitree_g1/popsicle_torsobase_v1",
    ),
    "Instinct-Parkour-Target-G1": TaskRegistration(
        factory_path="instinctlab.tasks.parkour.config.g1:parkour_target_g1",
        asset_id="unitree_g1/popsicle_torsobase_parkour_v1",
    ),
    "Instinct-Shadowing-WholeBody-Plane-G1-v0": TaskRegistration(
        factory_path=(
            "instinctlab.tasks.shadowing.whole_body.config.g1.plane_shadowing_cfg:"
            "g1_plane_shadowing"
        ),
        asset_id="unitree_g1/popsicle_torsobase_shadowing_v1",
    ),
    "Instinct-Shadowing-WholeBody-Plane-G1-Play-v0": TaskRegistration(
        factory_path=(
            "instinctlab.tasks.shadowing.whole_body.config.g1.plane_shadowing_cfg:"
            "g1_plane_shadowing_play"
        ),
        asset_id="unitree_g1/popsicle_torsobase_shadowing_v1",
        checkpoint_task_id="Instinct-Shadowing-WholeBody-Plane-G1-v0",
    ),
    "Instinct-Perceptive-Shadowing-G1-v0": TaskRegistration(
        factory_path=(
            "instinctlab.tasks.shadowing.perceptive.config.g1.perceptive_shadowing_cfg:"
            "g1_perceptive_shadowing"
        ),
        asset_id="unitree_g1/popsicle_torsobase_shadowing_v1",
    ),
    "Instinct-Perceptive-Shadowing-G1-Play-v0": TaskRegistration(
        factory_path=(
            "instinctlab.tasks.shadowing.perceptive.config.g1.perceptive_shadowing_cfg:"
            "g1_perceptive_shadowing_play"
        ),
        asset_id="unitree_g1/popsicle_torsobase_shadowing_v1",
        checkpoint_task_id="Instinct-Perceptive-Shadowing-G1-v0",
    ),
    "Instinct-Perceptive-Shadowing-G1-OneMotion-v0": TaskRegistration(
        factory_path=(
            "instinctlab.tasks.shadowing.perceptive.config.g1.perceptive_shadowing_cfg:"
            "g1_perceptive_shadowing_one_motion"
        ),
        asset_id="unitree_g1/popsicle_torsobase_shadowing_v1",
    ),
    "Instinct-Perceptive-Shadowing-G1-OneMotion-Play-v0": TaskRegistration(
        factory_path=(
            "instinctlab.tasks.shadowing.perceptive.config.g1.perceptive_shadowing_cfg:"
            "g1_perceptive_shadowing_one_motion_play"
        ),
        asset_id="unitree_g1/popsicle_torsobase_shadowing_v1",
        checkpoint_task_id="Instinct-Perceptive-Shadowing-G1-OneMotion-v0",
    ),
    "Instinct-Perceptive-Vae-G1-v0": TaskRegistration(
        factory_path=(
            "instinctlab.tasks.shadowing.perceptive.config.g1.perceptive_vae_cfg:"
            "g1_perceptive_vae"
        ),
        asset_id="unitree_g1/popsicle_torsobase_shadowing_v1",
    ),
    "Instinct-Perceptive-Vae-G1-Play-v0": TaskRegistration(
        factory_path=(
            "instinctlab.tasks.shadowing.perceptive.config.g1.perceptive_vae_cfg:"
            "g1_perceptive_vae_play"
        ),
        asset_id="unitree_g1/popsicle_torsobase_shadowing_v1",
        checkpoint_task_id="Instinct-Perceptive-Vae-G1-v0",
    ),
    "Instinct-Perceptive-HOI-Shadowing-G1-v0": TaskRegistration(
        factory_path=(
            "instinctlab.tasks.shadowing.perceptive_hoi.config.g1.perceptive_shadowing_cfg:"
            "g1_perceptive_hoi_shadowing"
        ),
        asset_id="unitree_g1/popsicle_torsobase_shadowing_v1",
    ),
    "Instinct-Perceptive-HOI-Shadowing-G1-Play-v0": TaskRegistration(
        factory_path=(
            "instinctlab.tasks.shadowing.perceptive_hoi.config.g1.perceptive_shadowing_cfg:"
            "g1_perceptive_hoi_shadowing_play"
        ),
        asset_id="unitree_g1/popsicle_torsobase_shadowing_v1",
        checkpoint_task_id="Instinct-Perceptive-HOI-Shadowing-G1-v0",
    ),
    "Instinct-BeyondMimic-Plane-G1-v0": TaskRegistration(
        factory_path=(
            "instinctlab.tasks.shadowing.beyondmimic.config.g1.beyondmimic_plane_cfg:"
            "g1_beyondmimic_plane"
        ),
        asset_id="unitree_g1/popsicle_torsobase_shadowing_v1",
    ),
    "Instinct-BeyondMimic-Plane-G1-Play-v0": TaskRegistration(
        factory_path=(
            "instinctlab.tasks.shadowing.beyondmimic.config.g1.beyondmimic_plane_cfg:"
            "g1_beyondmimic_plane_play"
        ),
        asset_id="unitree_g1/popsicle_torsobase_shadowing_v1",
        checkpoint_task_id="Instinct-BeyondMimic-Plane-G1-v0",
    ),
}

REGISTRATIONS: Mapping[str, TaskRegistration] = MappingProxyType(_REGISTRATIONS)
"""The single immutable task catalog."""

for _task_id, _registration in REGISTRATIONS.items():
    if (
        _registration.checkpoint_task_id is not None
        and _registration.checkpoint_task_id not in REGISTRATIONS
    ):
        raise ValueError(
            f"task {_task_id!r} references unknown checkpoint task "
            f"{_registration.checkpoint_task_id!r}"
        )

# Read-only compatibility views for application code that only needs one field.
TASKS: Mapping[str, str] = MappingProxyType(
    {task_id: registration.factory_path for task_id, registration in REGISTRATIONS.items()}
)
TASK_ASSETS: Mapping[str, str] = MappingProxyType(
    {task_id: registration.asset_id for task_id, registration in REGISTRATIONS.items()}
)
PLAY_CHECKPOINT_TASKS: Mapping[str, str] = MappingProxyType(
    {
        task_id: registration.checkpoint_task_id
        for task_id, registration in REGISTRATIONS.items()
        if registration.checkpoint_task_id is not None
    }
)


def ids() -> tuple[str, ...]:
    """Every declared task id, without importing any of them."""
    return tuple(sorted(REGISTRATIONS))


def _registration(task_id: str) -> TaskRegistration:
    try:
        return REGISTRATIONS[task_id]
    except KeyError:
        raise KeyError(
            f"unknown task {task_id!r}; declared tasks are {', '.join(ids())}"
        ) from None


def checkpoint_task_id(task_id: str) -> str:
    """Return the task identity expected in a checkpoint used by ``task_id``."""
    registration = _registration(task_id)
    return registration.checkpoint_task_id or task_id


def factory(task_id: str) -> Callable[[RobotSpec], TaskSpec]:
    """Import and return the factory for ``task_id``."""
    path = _registration(task_id).factory_path
    module_path, _, attribute = path.partition(":")
    return getattr(import_module(module_path), attribute)


def asset_id(task_id: str) -> str:
    """Return the engine-neutral native asset selection for ``task_id``."""
    return _registration(task_id).asset_id


def spec(task_id: str, robot: RobotSpec) -> TaskSpec:
    """Build ``task_id`` from a robot already normalized by the selected engine."""
    expected_asset_id = asset_id(task_id)
    if robot.asset_id != expected_asset_id:
        raise ValueError(
            f"task {task_id!r} selects {expected_asset_id!r}, got robot {robot.asset_id!r}"
        )
    built = factory(task_id)(robot)
    if built.task_id != task_id:
        raise ValueError(
            f"the registry calls this task {task_id!r} but the spec calls itself {built.task_id!r}; "
            "a task known by two names cannot be compared across engines"
        )
    built.validate()
    frozen = freeze_task_spec(built)
    frozen.validate()
    return frozen


__all__ = [
    "PLAY_CHECKPOINT_TASKS",
    "REGISTRATIONS",
    "TASKS",
    "TASK_ASSETS",
    "TaskRegistration",
    "asset_id",
    "checkpoint_task_id",
    "factory",
    "ids",
    "spec",
]
