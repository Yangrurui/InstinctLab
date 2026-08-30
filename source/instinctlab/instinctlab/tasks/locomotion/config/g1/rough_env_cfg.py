"""Rough-ground G1 velocity tracking on the shared engine-neutral terrain recipe.

The robot, observations, rewards, events and commands are the flat task's. What
changes is the shared rough grid and a ``terrain_levels`` curriculum that walks
robots through it. Each adapter lowers the same recipe to native terrain
primitives; height scanning remains a separate sensor concern.
"""

from __future__ import annotations

from instinctlab.spec import (
    ContactSensorRef,
    CurriculumTermSpec,
    MdpSpec,
    SceneSpec,
    TaskSpec,
)
from instinctlab.spec.capability import Requirement
from instinctlab.spec.robot import RobotSpec
from instinctlab.tasks.locomotion.config.g1.flat_env_cfg import G1LocomotionFlatEnvCfg
from instinctlab.tasks.locomotion.mdp import curriculums
from instinctlab.tasks.terrain import rough_terrain


class G1LocomotionRoughEnvCfg(G1LocomotionFlatEnvCfg):
    def __init__(self, robot: RobotSpec) -> None:
        super().__init__(robot)
        self.scene = SceneSpec(
            terrain=rough_terrain(),
            contact_sensors=(
                ContactSensorRef(
                    name="contact_forces",
                    elements=".*",
                    track_air_time=True,
                    history_length=3,
                ),
            ),
            env_spacing=2.5,
        )
        self.curriculum = {
            "terrain_levels": CurriculumTermSpec(
                func=curriculums.terrain_levels_vel,
                params={"command_name": "base_velocity"},
                level=Requirement.REQUIRED,
            )
        }


def rough_g1(robot: RobotSpec) -> TaskSpec:
    """Convert the explicit Rough config at the task registry boundary."""
    config = G1LocomotionRoughEnvCfg(robot)
    return TaskSpec(
        task_id="Instinct-Velocity-Rough-G1",
        robot=config.robot,
        scene=config.scene,
        sim=config.sim,
        mdp=MdpSpec(
            observations=config.observations,
            actions=config.actions,
            commands=config.commands,
            rewards=config.rewards,
            terminations=config.terminations,
            events=config.events,
            curriculum=config.curriculum,
        ),
        agent=config.agent,
        engines=("isaacsim", "mjlab"),
    )
