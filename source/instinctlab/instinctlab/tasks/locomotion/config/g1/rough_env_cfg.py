"""Rough-ground G1 velocity tracking on the shared engine-neutral terrain recipe.

The robot, observations, rewards, events and commands are the flat task's. What
changes is the shared rough grid and a ``terrain_levels`` curriculum that walks
robots through it. Each adapter lowers the same recipe to native terrain
primitives; height scanning remains a separate sensor concern.
"""

from __future__ import annotations

from instinctlab import mdp
from instinctlab.spec import (
    ContactSensorRef,
    CurriculumTermSpec,
    SceneSpec,
    TaskSpec,
)
from instinctlab.spec.capability import Requirement
from instinctlab.tasks.locomotion.config.g1.flat_env_cfg import (
    COMMAND,
    G1FlatEnvCfg,
)
from instinctlab.tasks.terrain import rough_terrain


class G1RoughEnvCfg(G1FlatEnvCfg):
    def __init__(self) -> None:
        super().__init__()
        self.scene = SceneSpec(
            terrain=rough_terrain(),
            contact_sensors=(
                ContactSensorRef(name="contact_forces", elements=".*", track_air_time=True, history_length=3),
            ),
            env_spacing=2.5,
        )
        self.curriculum = {
            "terrain_levels": CurriculumTermSpec(
                func=mdp.terrain_levels_vel,
                params={"command_name": COMMAND},
                level=Requirement.REQUIRED,
            )
        }


def rough_g1() -> TaskSpec:
    return G1RoughEnvCfg().to_task_spec("Instinct-Velocity-Rough-G1")
