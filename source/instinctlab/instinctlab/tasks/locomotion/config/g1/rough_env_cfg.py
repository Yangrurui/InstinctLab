"""Rough-ground G1 velocity tracking, the same MDP as the flat task on reference rough ground.

The robot, observations, rewards, events and commands are the flat task's. What changes is the
ground -- ``kind="rough"``, which each adapter fills from its own parkour reference -- and a
``terrain_levels`` curriculum that walks robots up and down that grid. Height scanning is not
here: that is a sensor, not a terrain, and belongs with the raycaster work.

Nothing here imports an engine. The two reference grids already disagree (scale, extra tile);
the declaration names the intent and does not pick a winner.
"""

from __future__ import annotations

from instinctlab import mdp
from instinctlab.assets.unitree_g1.isaacsim import make_g1_29dof_robot_spec
from instinctlab.spec import (
    ActionTermSpec,
    AgentSpec,
    ContactSensorRef,
    CurriculumTermSpec,
    DoneTermSpec,
    MdpSpec,
    SceneSpec,
    SimSpec,
    TaskSpec,
)
from instinctlab.spec.capability import Requirement
from instinctlab.tasks.locomotion.config.g1.flat_env_cfg import (
    COMMAND,
    UPPER_BODY_CONTACT,
    _action_scale,
    _canonical_joints,
    _commands,
    _events,
    _policy_observations,
    _rewards,
)
from instinctlab.tasks.locomotion.terrains import rough_terrain


def rough_g1() -> TaskSpec:
    """The rough-ground sibling of :func:`~instinctlab.tasks.locomotion.config.g1.flat_g1`."""
    robot = make_g1_29dof_robot_spec()
    joints = _canonical_joints(robot)
    return TaskSpec(
        task_id="Instinct-Velocity-Rough-G1",
        robot=robot,
        scene=SceneSpec(
            terrain=rough_terrain(),
            contact_sensors=(
                ContactSensorRef(name="contact_forces", elements=".*", track_air_time=True, history_length=3),
            ),
            env_spacing=2.5,
        ),
        sim=SimSpec(physics_dt=0.005, decimation=4, episode_length_s=20.0),
        mdp=MdpSpec(
            observations={
                "policy": _policy_observations(corrupt=True, joints=joints),
                "critic": _policy_observations(corrupt=False, joints=joints),
            },
            actions={
                "joint_pos": ActionTermSpec(
                    kind="joint_position",
                    target=joints,
                    params={"scale": _action_scale(robot), "use_default_offset": True},
                )
            },
            commands=_commands(),
            rewards={"rewards": _rewards()},
            terminations={
                "time_out": DoneTermSpec(func=mdp.time_out, time_out=True),
                "base_contact": DoneTermSpec(
                    func=mdp.illegal_contact, time_out=False, params={"sensor": UPPER_BODY_CONTACT}
                ),
            },
            events=_events(),
            curriculum={
                "terrain_levels": CurriculumTermSpec(
                    func=mdp.terrain_levels_vel,
                    params={"command_name": COMMAND},
                    level=Requirement.REQUIRED,
                )
            },
        ),
        agent=AgentSpec(runner="instinctlab.tasks.locomotion.config.g1.agents.instinct_rl_ppo_cfg:G1FlatPPORunnerCfg"),
        engines=("isaacsim", "mjlab"),
    )
