"""Unified rough-ground velocity tracking for Adam SP 23DOF."""

from instinctlab import mdp
from instinctlab.sim.robot_spec import RobotSpec
from instinctlab.spec import CurriculumTermSpec, MdpSpec, SceneSpec, TaskSpec
from instinctlab.spec.capability import Requirement
from instinctlab.tasks.locomotion.config.adam_sp.flat_env_cfg import (
    ADAM_SP_ACTIONS,
    ADAM_SP_AGENT,
    ADAM_SP_COMMANDS,
    ADAM_SP_CONTACT_SENSORS,
    ADAM_SP_EVENTS,
    ADAM_SP_OBSERVATIONS,
    ADAM_SP_REWARDS,
    ADAM_SP_SIM,
    ADAM_SP_TERMINATIONS,
    COMMAND,
)
from instinctlab.tasks.terrain import rough_terrain


def _rough_adam_sp(robot: RobotSpec) -> TaskSpec:
    return TaskSpec(
        task_id="Instinct-Velocity-Rough-Adam-SP",
        robot=robot,
        scene=SceneSpec(
            terrain=rough_terrain(),
            contact_sensors=ADAM_SP_CONTACT_SENSORS,
            env_spacing=2.5,
        ),
        sim=ADAM_SP_SIM,
        mdp=MdpSpec(
            observations=ADAM_SP_OBSERVATIONS,
            actions=ADAM_SP_ACTIONS,
            commands=ADAM_SP_COMMANDS,
            rewards=ADAM_SP_REWARDS,
            terminations=ADAM_SP_TERMINATIONS,
            events=ADAM_SP_EVENTS,
            curriculum={
                "terrain_levels": CurriculumTermSpec(
                    func=mdp.terrain_levels_vel,
                    params={"command_name": COMMAND},
                    level=Requirement.REQUIRED,
                )
            },
        ),
        agent=ADAM_SP_AGENT,
        engines=("isaacsim", "mjlab"),
    )


def rough_adam_sp_isaacsim() -> TaskSpec:
    from instinctlab.assets.adam_sp.isaacsim import ADAM_SP_23DOF_ROBOT

    return _rough_adam_sp(ADAM_SP_23DOF_ROBOT)


def rough_adam_sp_mjlab() -> TaskSpec:
    from instinctlab.assets.adam_sp.mjlab import ADAM_SP_23DOF_ROBOT

    return _rough_adam_sp(ADAM_SP_23DOF_ROBOT)


__all__ = ["rough_adam_sp_isaacsim", "rough_adam_sp_mjlab"]
