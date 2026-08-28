"""Unified flat-ground velocity tracking for Adam SP 23DOF."""

import math

from instinctlab import mdp
from instinctlab.assets.adam_sp.robot import (
    ADAM_SP_23DOF_JOINT_NAMES,
    ADAM_SP_23DOF_ROBOT,
)
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

COMMAND = "base_velocity"

ROBOT_ENTITY = EntityRef("robot", bodies=".*")
ROBOT_JOINTS = EntityRef(
    "robot",
    joints=ADAM_SP_23DOF_JOINT_NAMES,
    preserve_order=True,
)
FEET_CONTACT = ContactSensorRef(
    name="contact_forces",
    elements=("left_ankle_roll_link", "right_ankle_roll_link"),
)
UPPER_BODY_CONTACT = ContactSensorRef(
    name="contact_forces",
    elements=(
        "pelvis",
        "torso_link",
        ".*_shoulder_.*",
        ".*_elbow_.*",
        ".*_wrist_.*",
        ".*_hip_.*",
        ".*_knee_.*",
    ),
)

ADAM_SP_CONTACT_SENSORS = (
    ContactSensorRef(
        name="contact_forces",
        elements=".*",
        track_air_time=True,
        history_length=3,
    ),
)

ADAM_SP_SIM = SimSpec(
    physics_dt=0.005,
    decimation=4,
    episode_length_s=20.0,
)

ADAM_SP_POLICY_OBSERVATIONS = ObsGroupSpec(
    terms={
        "base_ang_vel": ObsTermSpec(
            func=mdp.base_ang_vel,
            noise=NoiseSpec("uniform", -0.2, 0.2),
        ),
        "projected_gravity": ObsTermSpec(
            func=mdp.projected_gravity,
            noise=NoiseSpec("uniform", -0.05, 0.05),
        ),
        "velocity_commands": ObsTermSpec(
            func=mdp.generated_commands,
            params={"command_name": COMMAND},
        ),
        "joint_pos": ObsTermSpec(
            func=mdp.joint_pos_rel,
            noise=NoiseSpec("uniform", -0.01, 0.01),
            params={"asset_cfg": ROBOT_JOINTS},
        ),
        "joint_vel": ObsTermSpec(
            func=mdp.joint_vel,
            noise=NoiseSpec("uniform", -1.5, 1.5),
            params={"asset_cfg": ROBOT_JOINTS},
        ),
        "actions": ObsTermSpec(func=mdp.last_action),
    },
    enable_corruption=True,
    concatenate_terms=False,
)

ADAM_SP_CRITIC_OBSERVATIONS = ObsGroupSpec(
    terms={
        "base_lin_vel": ObsTermSpec(func=mdp.base_lin_vel),
        "base_ang_vel": ObsTermSpec(func=mdp.base_ang_vel),
        "projected_gravity": ObsTermSpec(func=mdp.projected_gravity),
        "velocity_commands": ObsTermSpec(
            func=mdp.generated_commands,
            params={"command_name": COMMAND},
        ),
        "joint_pos": ObsTermSpec(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": ROBOT_JOINTS},
        ),
        "joint_vel": ObsTermSpec(
            func=mdp.joint_vel,
            params={"asset_cfg": ROBOT_JOINTS},
        ),
        "actions": ObsTermSpec(func=mdp.last_action),
    },
    enable_corruption=False,
    concatenate_terms=False,
)

ADAM_SP_OBSERVATIONS = {
    "policy": ADAM_SP_POLICY_OBSERVATIONS,
    "critic": ADAM_SP_CRITIC_OBSERVATIONS,
}

ADAM_SP_ACTION_SCALE = {
    "left_hip_pitch_joint": 0.19166666666666668,
    "left_hip_roll_joint": 0.075,
    "left_hip_yaw_joint": 0.0875,
    "left_knee_joint": 0.19166666666666668,
    "left_ankle_pitch_joint": 0.15384615384615385,
    "left_ankle_roll_joint": 0.14285714285714285,
    "right_hip_pitch_joint": 0.19166666666666668,
    "right_hip_roll_joint": 0.075,
    "right_hip_yaw_joint": 0.0875,
    "right_knee_joint": 0.19166666666666668,
    "right_ankle_pitch_joint": 0.15384615384615385,
    "right_ankle_roll_joint": 0.14285714285714285,
    "waist_roll_joint": 0.09375,
    "waist_pitch_joint": 0.09375,
    "waist_yaw_joint": 0.09375,
    "left_shoulder_pitch_joint": 0.2708333333333333,
    "left_shoulder_roll_joint": 0.2708333333333333,
    "left_shoulder_yaw_joint": 0.2708333333333333,
    "left_elbow_joint": 0.125,
    "right_shoulder_pitch_joint": 0.2708333333333333,
    "right_shoulder_roll_joint": 0.2708333333333333,
    "right_shoulder_yaw_joint": 0.2708333333333333,
    "right_elbow_joint": 0.125,
}

ADAM_SP_ACTIONS = {
    "joint_pos": ActionTermSpec(
        kind="joint_position",
        target=ROBOT_JOINTS,
        params={
            "scale": ADAM_SP_ACTION_SCALE,
            "use_default_offset": True,
        },
    )
}

ADAM_SP_COMMANDS = {
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

ADAM_SP_REWARDS = {
    "rewards": {
        "termination_penalty": RewardTermSpec(
            func=mdp.is_terminated,
            weight=-200.0,
        ),
        "track_lin_vel_xy_exp": RewardTermSpec(
            func=mdp.track_lin_vel_xy_yaw_frame_exp,
            weight=1.0,
            params={"command_name": COMMAND, "std": 0.5},
        ),
        "track_ang_vel_z_exp": RewardTermSpec(
            func=mdp.track_ang_vel_z_world_exp,
            weight=1.0,
            params={"command_name": COMMAND, "std": 0.5},
        ),
        "feet_air_time": RewardTermSpec(
            func=mdp.feet_air_time_positive_biped,
            weight=1.0,
            params={
                "command_name": COMMAND,
                "sensor": FEET_CONTACT,
                "threshold": 0.5,
            },
        ),
        "feet_slide": RewardTermSpec(
            kind="contact_slide",
            weight=-0.1,
            params={
                "sensor_cfg": FEET_CONTACT,
                "asset_cfg": EntityRef(
                    "robot",
                    bodies=("left_ankle_roll_link", "right_ankle_roll_link"),
                ),
            },
            level=Requirement.REQUIRED,
        ),
        "flat_orientation_l2": RewardTermSpec(
            func=mdp.flat_orientation_l2,
            weight=-1.0,
        ),
        "stand_still": RewardTermSpec(
            func=mdp.stand_still,
            weight=-0.8,
            params={"command_name": COMMAND},
        ),
        "dof_pos_limits": RewardTermSpec(
            func=mdp.joint_pos_limits,
            weight=-1.0,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
                )
            },
        ),
        "joint_deviation_hip": RewardTermSpec(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(".*_hip_yaw_joint", ".*_hip_roll_joint"),
                )
            },
        ),
        "joint_deviation_arms": RewardTermSpec(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(
                        ".*_shoulder_pitch_joint",
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        ".*_elbow_joint",
                    ),
                )
            },
        ),
        "joint_deviation_torso": RewardTermSpec(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": EntityRef("robot", joints="waist_.*")},
        ),
        "joint_deviation_knee": RewardTermSpec(
            func=mdp.joint_deviation_l1,
            weight=-0.05,
            params={"asset_cfg": EntityRef("robot", joints=".*_knee_joint")},
        ),
        "lin_vel_z_l2": RewardTermSpec(
            func=mdp.lin_vel_z_l2,
            weight=-0.1,
        ),
        "action_rate_l2": RewardTermSpec(
            func=mdp.action_rate_l2,
            weight=-0.05,
        ),
        "dof_acc_l2": RewardTermSpec(
            kind="joint_acc_l2",
            weight=-2.0e-7,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(".*_hip_.*", ".*_knee_joint"),
                )
            },
            level=Requirement.REQUIRED,
        ),
        "dof_torques_l2": RewardTermSpec(
            kind="joint_torques_l2",
            weight=-4.0e-6,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(".*_hip_.*", ".*_knee_joint"),
                )
            },
            level=Requirement.REQUIRED,
        ),
    }
}

ADAM_SP_TERMINATIONS = {
    "time_out": DoneTermSpec(
        func=mdp.time_out,
        time_out=True,
    ),
    "base_contact": DoneTermSpec(
        func=mdp.illegal_contact,
        params={"sensor": UPPER_BODY_CONTACT},
        time_out=False,
    ),
}

ADAM_SP_EVENTS = {
    "physics_material": EventTermSpec(
        kind="randomize_friction",
        mode="startup",
        target=ROBOT_ENTITY,
    ),
    "add_base_mass": EventTermSpec(
        kind="randomize_body_mass",
        mode="startup",
        target=EntityRef("robot", bodies="pelvis"),
        params={"add_range": (-5.0, 5.0), "operation": "add"},
    ),
    "base_external_force_torque": EventTermSpec(
        kind="apply_external_force_torque",
        mode="reset",
        target=EntityRef("robot", bodies="pelvis"),
        params={
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
        level=Requirement.OPTIONAL,
    ),
    "reset_base": EventTermSpec(
        kind="reset_root_state_uniform",
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),
            },
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
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (-1.0, 1.0),
        },
    ),
    "push_robot": EventTermSpec(
        kind="push_by_setting_velocity",
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
            }
        },
    ),
}

ADAM_SP_AGENT = AgentSpec(
    runner=(
        "instinctlab.tasks.locomotion.config.adam_sp.agents.instinct_rl_ppo_cfg:"
        "AdamSPFlatPPORunnerCfg"
    )
)


def flat_adam_sp() -> TaskSpec:
    return TaskSpec(
        task_id="Instinct-Velocity-Flat-Adam-SP",
        robot=ADAM_SP_23DOF_ROBOT,
        scene=SceneSpec(
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
            curriculum={},
        ),
        agent=ADAM_SP_AGENT,
        engines=("isaacsim", "mjlab"),
    )


__all__ = [
    "ADAM_SP_ACTIONS",
    "ADAM_SP_AGENT",
    "ADAM_SP_COMMANDS",
    "ADAM_SP_CONTACT_SENSORS",
    "ADAM_SP_EVENTS",
    "ADAM_SP_OBSERVATIONS",
    "ADAM_SP_REWARDS",
    "ADAM_SP_SIM",
    "ADAM_SP_TERMINATIONS",
    "COMMAND",
    "flat_adam_sp",
]
