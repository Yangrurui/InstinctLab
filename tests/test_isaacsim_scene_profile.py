from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

from instinctlab_engine_isaacsim.scene import PROFILE_DEFAULTS, _spawn_overrides


@dataclass(frozen=True)
class _ArticulationProperties:
    enabled_self_collisions: bool = True
    solver_position_iteration_count: int = 8
    solver_velocity_iteration_count: int = 4

    def replace(self, **kwargs):
        return replace(self, **kwargs)


def test_self_collision_profile_reaches_converter_and_articulation() -> None:
    spawn = SimpleNamespace(articulation_props=_ArticulationProperties())
    scene = SimpleNamespace(contact_sensors=())
    profile = {**PROFILE_DEFAULTS, "self_collision": False}

    overrides = _spawn_overrides(spawn, scene, profile)

    assert overrides["self_collision"] is False
    assert overrides["articulation_props"].enabled_self_collisions is False
    assert overrides["articulation_props"].solver_position_iteration_count == 8
