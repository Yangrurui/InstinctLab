"""Flat-ground G1 velocity tracking, declared once for every engine.

This began as a restatement of main's ``G1FlatEnvCfg`` and is now the only description of the task:
that config, its Gym ids and its MDP package were deleted when D3 was retired, so there is no second
copy to agree with and no golden dump to diff against. What the two engines produce from this file
is compared to each other instead, both structurally and term by term, while declaration checks
hold the properties this file has to have on its own.

Nothing here imports an engine, names a joint index, or spells a native config class. That is what
lets an MJLab process read it with Isaac Sim absent. Isolation is checked with both engine imports
blocked so an accidental dependency surfaces before reaching a single-engine installation.

Three of main's rewards are declared by ``kind`` rather than by function, and the distinction is
about how they are written rather than whether the task has them. Two of them, ``dof_acc_l2`` and
``dof_torques_l2``, read quantities the two engines report differently enough that a single shared
term would mean different things on each -- joint acceleration is a finite difference here and a
solver output there, and applied torque excludes passive terms here and includes them there. The
third, ``feet_slide``, multiplies foot velocity by a contact mask derived from force magnitude, and
the two engines' contact forces are not the same measurement. Each backend registers its own
reference implementation and all three are ``REQUIRED``: a task that lost a term because no single
implementation could be written would no longer be the same task, and a run that has them on one
engine and not the other is not a comparison between engines.
"""

from __future__ import annotations

import math

from instinctlab import mdp
from instinctlab.assets.unitree_g1.catalog import make_g1_29dof_robot_spec
from instinctlab.sim.robot_spec import RobotSpec
from instinctlab.spec import (
    ActionTermSpec,
    AgentSpec,
    CommandTermSpec,
    ContactSensorRef,
    DoneTermSpec,
    EntityRef,
    EventTermSpec,
    MdpSpec,
    NoiseSpec,
    ObsGroupSpec,
    ObsTermSpec,
    RewardTermSpec,
    SceneSpec,
    SimSpec,
    TaskSpec,
)
from instinctlab.spec.capability import Requirement

ROBOT = EntityRef("robot", bodies=".*")

# Contact terms take a ContactSensorRef rather than an EntityRef, and the distinction is not
# cosmetic. An EntityRef selects parts of a scene entity and lowers to the engine's own selector; a
# ContactSensorRef names a declared sensor and a subset of the elements it already tracks, which is
# what makes the same declaration work against Isaac Lab's one broad sensor and mjlab's several
# narrow ones. The scene below declares the sensor once; these select within it.
FEET_CONTACT = ContactSensorRef(name="contact_forces", elements=".*_ankle_roll_link")
UPPER_BODY_CONTACT = ContactSensorRef(
    name="contact_forces",
    elements=("torso_link", ".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*", ".*_hip_.*", ".*_knee_.*"),
)

COMMAND = "base_velocity"


def _canonical_joints(robot: RobotSpec) -> EntityRef:
    """Every joint, named explicitly, in the canonical depth-first order. Decision D1 lives here.

    A lone ``".*"`` selects the same twenty-nine joints, so it is tempting to write. It does not
    order them: ``preserve_order`` orders a selection by the *patterns* it was given, and one
    pattern leaves the entity's own order in place -- PhysX's breadth-first walk on Isaac Sim, the
    model file's order on mjlab, neither of which is the depth-first order this project declares.
    Naming the joints is what makes the joint axis mean the same thing on both engines.

    Both the action term and the two joint observations take this, because pinning one without the
    other would leave a policy whose inputs and outputs are indexed differently per engine.
    """
    return EntityRef("robot", joints=tuple(robot.joint_names), preserve_order=True)


class G1FlatPolicyObsCfg:
    def __init__(self, joints: EntityRef) -> None:
        self.base_ang_vel = ObsTermSpec(func=mdp.base_ang_vel, noise=NoiseSpec("uniform", -0.2, 0.2))
        self.projected_gravity = ObsTermSpec(func=mdp.projected_gravity, noise=NoiseSpec("uniform", -0.05, 0.05))
        self.velocity_commands = ObsTermSpec(func=mdp.generated_commands, params={"command_name": COMMAND})
        self.joint_pos = ObsTermSpec(
            func=mdp.joint_pos_rel,
            noise=NoiseSpec("uniform", -0.01, 0.01),
            params={"asset_cfg": joints},
        )
        self.joint_vel = ObsTermSpec(
            func=mdp.joint_vel,
            noise=NoiseSpec("uniform", -1.5, 1.5),
            params={"asset_cfg": joints},
        )
        self.actions = ObsTermSpec(func=mdp.last_action)

    def to_spec(self) -> ObsGroupSpec:
        return ObsGroupSpec(terms=dict(vars(self)), enable_corruption=True, concatenate_terms=False)


class G1FlatCriticObsCfg:
    def __init__(self, joints: EntityRef) -> None:
        self.base_lin_vel = ObsTermSpec(func=mdp.base_lin_vel)
        self.base_ang_vel = ObsTermSpec(func=mdp.base_ang_vel)
        self.projected_gravity = ObsTermSpec(func=mdp.projected_gravity)
        self.velocity_commands = ObsTermSpec(func=mdp.generated_commands, params={"command_name": COMMAND})
        self.joint_pos = ObsTermSpec(func=mdp.joint_pos_rel, params={"asset_cfg": joints})
        self.joint_vel = ObsTermSpec(func=mdp.joint_vel, params={"asset_cfg": joints})
        self.actions = ObsTermSpec(func=mdp.last_action)

    def to_spec(self) -> ObsGroupSpec:
        return ObsGroupSpec(terms=dict(vars(self)), enable_corruption=False, concatenate_terms=False)


class G1FlatObservationsCfg:
    def __init__(self, joints: EntityRef) -> None:
        self.policy = G1FlatPolicyObsCfg(joints)
        self.critic = G1FlatCriticObsCfg(joints)

    def to_dict(self) -> dict[str, ObsGroupSpec]:
        return {"policy": self.policy.to_spec(), "critic": self.critic.to_spec()}


class G1FlatRewardsCfg:
    def __init__(self) -> None:
        def deviation(joints: str | tuple[str, ...], weight: float) -> RewardTermSpec:
            return RewardTermSpec(
                func=mdp.joint_deviation_l1,
                weight=weight,
                params={"asset_cfg": EntityRef("robot", joints=joints)},
            )

        terms = {
        "termination_penalty": RewardTermSpec(func=mdp.is_terminated, weight=-200.0),
        "track_lin_vel_xy_exp": RewardTermSpec(
            func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"command_name": COMMAND, "std": 0.5}
        ),
        "track_ang_vel_z_exp": RewardTermSpec(
            func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": COMMAND, "std": 0.5}
        ),
        "feet_air_time": RewardTermSpec(
            func=mdp.feet_air_time_positive_biped,
            weight=1.0,
            params={"command_name": COMMAND, "sensor": FEET_CONTACT, "threshold": 0.5},
        ),
        # Stated by name rather than by function, because the two engines measure it differently.
        # A slide penalty multiplies foot velocity by a contact mask taken from force magnitude, and
        # neither factor survives translation: Isaac Lab reports the normal component of the contact
        # force where mjlab reports the whole vector, and Isaac Lab's ``body_lin_vel_w`` is the
        # centre-of-mass velocity where mjlab's is the link's. Each backend supplies its own
        # reference implementation; ``kind`` is how a task asks for that without naming one.
        "feet_slide": RewardTermSpec(
            kind="contact_slide",
            weight=-0.1,
            params={"sensor_cfg": FEET_CONTACT, "asset_cfg": EntityRef("robot", bodies=".*_ankle_roll_link")},
            level=Requirement.REQUIRED,
        ),
        "flat_orientation_l2": RewardTermSpec(func=mdp.flat_orientation_l2, weight=-1.0),
        "stand_still": RewardTermSpec(func=mdp.stand_still, weight=-0.8, params={"command_name": COMMAND}),
        "dof_pos_limits": RewardTermSpec(
            func=mdp.joint_pos_limits,
            weight=-1.0,
            params={"asset_cfg": EntityRef("robot", joints=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"))},
        ),
        "joint_deviation_hip": deviation((".*_hip_yaw_joint", ".*_hip_roll_joint"), -0.1),
        "joint_deviation_arms": deviation(
            (
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ),
            -0.1,
        ),
        "joint_deviation_torso": deviation("waist_.*", -0.1),
        "joint_deviation_knee": deviation((".*_knee_joint",), -0.05),
        "lin_vel_z_l2": RewardTermSpec(func=mdp.lin_vel_z_l2, weight=-0.1),
        "action_rate_l2": RewardTermSpec(func=mdp.action_rate_l2, weight=-0.05),
        # Also by name. Joint acceleration is a finite difference of measured velocity on one engine
        # and a solver output on the other; applied torque excludes passive terms on one and
        # includes them on the other. Both engines ship the term, so both get their own.
        "dof_acc_l2": RewardTermSpec(
            kind="joint_acc_l2",
            weight=-2.0e-7,
            params={"asset_cfg": EntityRef("robot", joints=(".*_hip_.*", ".*_knee_joint"))},
            level=Requirement.REQUIRED,
        ),
        "dof_torques_l2": RewardTermSpec(
            kind="joint_torques_l2",
            weight=-4.0e-6,
            params={"asset_cfg": EntityRef("robot", joints=(".*_hip_.*", ".*_knee_joint"))},
            level=Requirement.REQUIRED,
        ),
        }
        for name, term in terms.items():
            setattr(self, name, term)

    def to_dict(self) -> dict[str, dict[str, RewardTermSpec]]:
        return {"rewards": dict(vars(self))}


def _events() -> dict[str, EventTermSpec]:
    """Domain randomisation, stated as intent rather than as a distribution.

    ``randomize_friction`` takes no range here even though main states one. The range is in this
    engine's profile, because the two engines randomise different things: Isaac Sim assigns
    per-shape materials from a bucket pool, mjlab scales one friction per environment. A shared
    range would be a number that means something different on each side, and matching numbers there
    would be a worse lie than not matching at all.
    """
    return {
        "physics_material": EventTermSpec(kind="randomize_friction", mode="startup", target=ROBOT),
        "add_base_mass": EventTermSpec(
            kind="randomize_body_mass",
            mode="startup",
            target=EntityRef("robot", bodies="torso_link"),
            params={"add_range": (-5.0, 5.0), "operation": "add"},
        ),
        "base_external_force_torque": EventTermSpec(
            kind="apply_external_force_torque",
            mode="reset",
            target=EntityRef("robot", bodies="torso_link"),
            params={"force_range": (0.0, 0.0), "torque_range": (-0.0, 0.0)},
            level=Requirement.OPTIONAL,
        ),
        "reset_base": EventTermSpec(
            kind="reset_root_state_uniform",
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.1, 0.1),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
            },
        ),
        "reset_robot_joints": EventTermSpec(
            kind="reset_joints_by_scale",
            mode="reset",
            params={"position_range": (0.8, 1.2), "velocity_range": (-1.0, 1.0)},
        ),
        "push_robot": EventTermSpec(
            kind="push_by_setting_velocity",
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        ),
    }


def _commands() -> dict[str, CommandTermSpec]:
    """Velocity command, shared with the rough sibling so the two ranges cannot drift apart."""
    return {
        COMMAND: CommandTermSpec(
            kind="uniform_velocity",
            params={
                "entity": "robot",
                "resampling_time_range": (10.0, 10.0),
                "rel_standing_envs": 0.2,
                "rel_heading_envs": 0.5,
                "heading_command": True,
                "heading_control_stiffness": 0.5,
                "debug_vis": True,
                "lin_vel_x": (-0.5, 1.0),
                "lin_vel_y": (-0.5, 0.5),
                "ang_vel_z": (-1.5, 1.5),
                "heading": (-math.pi, math.pi),
            },
        )
    }


def _action_scale(robot: RobotSpec) -> dict[str, float]:
    """Per-joint action scale, taken from the robot catalog rather than from the engine.

    Main reaches into the Isaac Lab asset module for this, where it is computed from actuator
    effort and stiffness and keyed by the regular expressions that group joints into actuators.
    The same numbers are already on :class:`RobotSpec` per joint, so taking them from there makes
    the task engine-free at no cost. The two agree exactly for all 29 joints; what differs is the
    shape -- sixteen patterns against twenty-nine names -- which is a whitelisted difference and
    the more explicit of the two.
    """
    return {joint.name: joint.action_scale for joint in robot.joint_properties}


class G1FlatEnvCfg:
    def __init__(self) -> None:
        self.robot = make_g1_29dof_robot_spec()
        joints = _canonical_joints(self.robot)
        self.scene = SceneSpec(
            contact_sensors=(
                ContactSensorRef(name="contact_forces", elements=".*", track_air_time=True, history_length=3),
            ),
            env_spacing=2.5,
        )
        self.sim = SimSpec(physics_dt=0.005, decimation=4, episode_length_s=20.0)
        self.observations = G1FlatObservationsCfg(joints)
        self.actions = {
            "joint_pos": ActionTermSpec(
                kind="joint_position",
                target=joints,
                params={"scale": _action_scale(self.robot), "use_default_offset": True},
            )
        }
        self.commands = _commands()
        self.rewards = G1FlatRewardsCfg()
        self.terminations = {
            "time_out": DoneTermSpec(func=mdp.time_out, time_out=True),
            "base_contact": DoneTermSpec(
                func=mdp.illegal_contact,
                time_out=False,
                params={"sensor": UPPER_BODY_CONTACT},
            ),
        }
        self.events = _events()
        self.curriculum = {}

    def to_task_spec(self, task_id: str) -> TaskSpec:
        return TaskSpec(
            task_id=task_id,
            robot=self.robot,
            scene=self.scene,
            sim=self.sim,
            mdp=MdpSpec(
                observations=self.observations.to_dict(),
                actions=self.actions,
                commands=self.commands,
                rewards=self.rewards.to_dict(),
                terminations=self.terminations,
                events=self.events,
                curriculum=self.curriculum,
            ),
            agent=AgentSpec(
                runner="instinctlab.tasks.locomotion.config.g1.agents.instinct_rl_ppo_cfg:G1FlatPPORunnerCfg"
            ),
            engines=("isaacsim", "mjlab"),
        )


def flat_g1() -> TaskSpec:
    return G1FlatEnvCfg().to_task_spec("Instinct-Velocity-Flat-G1")
