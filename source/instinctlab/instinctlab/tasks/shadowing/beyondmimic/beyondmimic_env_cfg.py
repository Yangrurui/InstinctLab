"""Engine-neutral BeyondMimic environment configuration."""

from __future__ import annotations

from instinctlab import mdp
from instinctlab.engines.assets import RobotSpec
from instinctlab.spec import (
    ActionTermSpec,
    AgentSpec,
    CommandTermSpec,
    ContactSensorRef,
    CurriculumTermSpec,
    DoneTermSpec,
    EntityRef,
    EventTermSpec,
    MdpSpec,
    MotionReferenceRef,
    NoiseSpec,
    ObsGroupSpec,
    ObsTermSpec,
    RewardTermSpec,
    SceneSpec,
    SimSpec,
    TaskSpec,
    TerrainSpec,
)
from instinctlab.spec.capability import Requirement

NON_SUPPORT_CONTACTS = (
    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
    r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
)


def make_scene(motion_reference: MotionReferenceRef, play: bool) -> SceneSpec:
    contact_sensor = ContactSensorRef(
        name="undesired_contact_forces",
        elements=".*",
        track_air_time=False,
        air_time_force_threshold=1.0,
        history_length=3,
    )
    return SceneSpec(
        terrain=TerrainSpec(kind="plane"),
        contact_sensors=(contact_sensor,),
        motion_references=(motion_reference,),
        env_spacing=2.5 if play else 4.0,
    )


def make_commands() -> dict[str, CommandTermSpec]:
    return {
        "position_ref_command": CommandTermSpec(
            kind="shadow_position_reference",
            params={
                "motion_reference": "motion_reference",
                "current_state_command": False,
                "realtime_mode": True,
                "anchor_frame": "robot",
            },
        ),
        "rotation_ref_command": CommandTermSpec(
            kind="shadow_rotation_reference",
            params={
                "motion_reference": "motion_reference",
                "current_state_command": False,
                "realtime_mode": True,
                "in_base_frame": True,
                "rotation_mode": "tannorm",
            },
        ),
        "joint_pos_ref_command": CommandTermSpec(
            kind="shadow_joint_position_reference",
            params={
                "motion_reference": "motion_reference",
                "current_state_command": False,
            },
        ),
        "joint_vel_ref_command": CommandTermSpec(
            kind="shadow_joint_velocity_reference",
            params={
                "motion_reference": "motion_reference",
                "current_state_command": False,
            },
        ),
    }


def make_actions(robot: RobotSpec) -> dict[str, ActionTermSpec]:
    joints = EntityRef("robot", joints=tuple(robot.joint_names), preserve_order=True)
    action_scale = {joint.name: joint.action_scale for joint in robot.joint_properties}
    return {
        "joint_pos": ActionTermSpec(
            kind="joint_position",
            target=joints,
            params={"scale": action_scale, "use_default_offset": True},
        )
    }


class ObservationGroupCfg:
    def __init__(
        self,
        terms: dict[str, ObsTermSpec],
        enable_corruption: bool,
    ) -> None:
        self.enable_corruption = enable_corruption
        for name, term in terms.items():
            setattr(self, name, term)

    def to_spec(self) -> ObsGroupSpec:
        terms = dict(vars(self))
        enable_corruption = terms.pop("enable_corruption")
        return ObsGroupSpec(
            terms=terms,
            enable_corruption=enable_corruption,
            concatenate_terms=False,
        )


class ObservationsCfg:
    def __init__(self, robot: RobotSpec, motion_reference: MotionReferenceRef) -> None:
        joints = EntityRef("robot", joints=tuple(robot.joint_names), preserve_order=True)
        links = EntityRef("robot", bodies=motion_reference.links, preserve_order=True)
        policy_terms = {
            "joint_pos_ref": ObsTermSpec(
                func=mdp.generated_commands,
                params={"command_name": "joint_pos_ref_command"},
            ),
            "joint_vel_ref": ObsTermSpec(
                func=mdp.generated_commands,
                params={"command_name": "joint_vel_ref_command"},
            ),
            "position_ref": ObsTermSpec(
                func=mdp.generated_commands,
                params={"command_name": "position_ref_command"},
                noise=NoiseSpec("uniform", -0.25, 0.25),
            ),
            "rotation_ref": ObsTermSpec(
                func=mdp.generated_commands,
                params={"command_name": "rotation_ref_command"},
                noise=NoiseSpec("uniform", -0.05, 0.05),
            ),
            "base_lin_vel": ObsTermSpec(kind="shadow_base_linear_velocity"),
            "projected_gravity": ObsTermSpec(
                func=mdp.projected_gravity,
                noise=NoiseSpec("uniform", -0.05, 0.05),
            ),
            "base_ang_vel": ObsTermSpec(
                func=mdp.base_ang_vel,
                noise=NoiseSpec("uniform", -0.2, 0.2),
            ),
            "joint_pos": ObsTermSpec(
                func=mdp.joint_pos_rel,
                params={"asset_cfg": joints},
                noise=NoiseSpec("uniform", -0.01, 0.01),
            ),
            "joint_vel": ObsTermSpec(
                func=mdp.joint_vel_rel,
                params={"asset_cfg": joints},
                noise=NoiseSpec("uniform", -0.5, 0.5),
            ),
            "last_action": ObsTermSpec(func=mdp.last_action),
        }
        critic_terms = {
            "joint_pos_ref": ObsTermSpec(
                func=mdp.generated_commands,
                params={"command_name": "joint_pos_ref_command"},
            ),
            "joint_vel_ref": ObsTermSpec(
                func=mdp.generated_commands,
                params={"command_name": "joint_vel_ref_command"},
            ),
            "position_ref": ObsTermSpec(
                func=mdp.generated_commands,
                params={"command_name": "position_ref_command"},
            ),
            "rotation_ref": ObsTermSpec(
                func=mdp.generated_commands,
                params={"command_name": "rotation_ref_command"},
            ),
            "link_pos": ObsTermSpec(
                kind="shadow_link_position",
                params={"motion_reference": motion_reference, "asset_cfg": links},
            ),
            "link_rot": ObsTermSpec(
                kind="shadow_link_rotation",
                params={"motion_reference": motion_reference, "asset_cfg": links},
            ),
            "base_lin_vel": ObsTermSpec(kind="shadow_base_linear_velocity"),
            "base_ang_vel": ObsTermSpec(func=mdp.base_ang_vel),
            "joint_pos": ObsTermSpec(func=mdp.joint_pos_rel, params={"asset_cfg": joints}),
            "joint_vel": ObsTermSpec(func=mdp.joint_vel_rel, params={"asset_cfg": joints}),
            "last_action": ObsTermSpec(func=mdp.last_action),
        }
        self.policy = ObservationGroupCfg(policy_terms, enable_corruption=True)
        self.critic = ObservationGroupCfg(critic_terms, enable_corruption=False)

    def to_dict(self) -> dict[str, ObsGroupSpec]:
        return {
            "policy": self.policy.to_spec(),
            "critic": self.critic.to_spec(),
        }


def make_events(play: bool) -> dict[str, EventTermSpec]:
    events = {
        "physics_material": EventTermSpec(
            kind="randomize_friction",
            mode="startup",
            target=EntityRef("robot", bodies=".*"),
            engine_params={
                "isaacsim": {
                    "static_friction_range": (0.3, 1.6),
                    "dynamic_friction_range": (0.3, 1.2),
                    "restitution_range": (0.0, 0.5),
                    "num_buckets": 64,
                },
                "mjlab": {
                    "ranges": (0.3, 1.6),
                    "operation": "abs",
                    "shared_random": True,
                },
            },
        ),
        "add_joint_default_pos": EventTermSpec(
            kind="randomize_joint_default",
            mode="startup",
            params={"range": (-0.01, 0.01)},
        ),
        "base_com": EventTermSpec(
            kind="randomize_base_com",
            mode="startup",
            target=EntityRef("robot", bodies=("torso_link",)),
            params={
                "com_range": {
                    "x": (-0.025, 0.025),
                    "y": (-0.05, 0.05),
                    "z": (-0.05, 0.05),
                }
            },
        ),
    }
    if not play:
        events["push_robot"] = EventTermSpec(
            kind="push_root_velocity",
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.2, 0.2),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                }
            },
        )
    events["match_motion_ref_with_scene"] = EventTermSpec(
        kind="shadow_match_reference_origin",
        mode="startup",
    )
    events["reset_robot"] = EventTermSpec(
        kind="shadow_reset_robot_from_reference",
        mode="reset",
        params={
            "position_offset": (0.0, 0.0, 0.0),
            "dof_vel_ratio": 1.0,
            "base_lin_vel_ratio": 1.0,
            "base_ang_vel_ratio": 1.0,
            "randomize_joint_pos_range": (-0.1, 0.1),
            "randomize_pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            "randomize_velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
        engine_params={"isaacsim": {"root_velocity_frame": "com"}},
    )
    if not play:
        events["bin_fail_counter_smoothing"] = EventTermSpec(
            kind="shadow_smooth_bin_failures",
            mode="interval",
            interval_range_s=(0.02, 0.02),
        )
    return events


class RewardsCfg:
    def __init__(self) -> None:
        contact_sensor = ContactSensorRef(
            name="undesired_contact_forces",
            elements=NON_SUPPORT_CONTACTS,
            track_air_time=False,
            air_time_force_threshold=1.0,
            history_length=3,
        )
        terms = {
            "base_position_imitation_gauss": RewardTermSpec(
                kind="shadow_base_position_gauss",
                weight=0.5,
                params={"std": 0.3},
                level=Requirement.REQUIRED,
            ),
            "base_rot_imitation_gauss": RewardTermSpec(
                kind="shadow_base_rotation_gauss",
                weight=0.5,
                params={"std": 0.4, "difference_type": "axis_angle"},
                level=Requirement.REQUIRED,
            ),
            "link_pos_imitation_gauss": RewardTermSpec(
                kind="shadow_link_position_gauss",
                weight=1.0,
                params={
                    "combine_method": "mean_prod",
                    "in_base_frame": False,
                    "in_relative_world_frame": True,
                    "std": 0.3,
                },
                level=Requirement.REQUIRED,
            ),
            "link_rot_imitation_gauss": RewardTermSpec(
                kind="shadow_link_rotation_gauss",
                weight=1.0,
                params={
                    "combine_method": "mean_prod",
                    "in_base_frame": False,
                    "in_relative_world_frame": True,
                    "std": 0.4,
                },
                level=Requirement.REQUIRED,
            ),
            "link_lin_vel_imitation_gauss": RewardTermSpec(
                kind="shadow_link_linear_velocity_gauss",
                weight=1.0,
                params={"combine_method": "mean_prod", "std": 1.0},
                level=Requirement.REQUIRED,
            ),
            "link_ang_vel_imitation_gauss": RewardTermSpec(
                kind="shadow_link_angular_velocity_gauss",
                weight=1.0,
                params={"combine_method": "mean_prod", "std": 3.14},
                level=Requirement.REQUIRED,
            ),
            "action_rate_l2": RewardTermSpec(func=mdp.action_rate_l2, weight=-0.1, level=Requirement.REQUIRED),
            "joint_limit": RewardTermSpec(func=mdp.joint_pos_limits, weight=-10.0, level=Requirement.REQUIRED),
            "undesired_contacts": RewardTermSpec(
                kind="shadow_undesired_contacts",
                weight=-0.1,
                params={"threshold": 1.0, "sensor": contact_sensor},
                level=Requirement.REQUIRED,
            ),
        }
        for name, term in terms.items():
            setattr(self, name, term)

    def to_dict(self) -> dict[str, dict[str, RewardTermSpec]]:
        return {"rewards": dict(vars(self))}


def make_terminations(motion_reference: MotionReferenceRef) -> dict[str, DoneTermSpec]:
    return {
        "time_out": DoneTermSpec(func=mdp.time_out, time_out=True),
        "base_pos_too_far": DoneTermSpec(
            kind="shadow_base_position_too_far",
            params={
                "distance_threshold": 0.25,
                "check_at_keyframe_threshold": -1,
                "print_reason": False,
                "height_only": True,
            },
        ),
        "base_pg_too_far": DoneTermSpec(
            kind="shadow_projected_gravity_too_far",
            params={
                "projected_gravity_threshold": 0.8,
                "check_at_keyframe_threshold": -1,
                "z_only": False,
                "print_reason": False,
            },
        ),
        "link_pos_too_far": DoneTermSpec(
            kind="shadow_link_position_too_far",
            params={
                "distance_threshold": 0.25,
                "in_base_frame": False,
                "check_at_keyframe_threshold": -1,
                "height_only": True,
                "print_reason": False,
            },
            target=EntityRef(
                "motion_reference",
                bodies=(
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ),
                preserve_order=True,
            ),
        ),
        "dataset_exhausted": DoneTermSpec(
            func=mdp.dataset_exhausted,
            params={"sensor": motion_reference, "reset_without_notice": False},
            time_out=True,
        ),
        "out_of_border": DoneTermSpec(
            func=mdp.terrain_out_of_bounds,
            params={"distance_buffer": 0.1},
            time_out=True,
        ),
    }


def make_curriculum(play: bool) -> dict[str, CurriculumTermSpec]:
    if play:
        return {}
    return {"beyond_adaptive_sampling": CurriculumTermSpec(kind="shadow_adaptive_sampling")}


class BeyondMimicEnvCfg:
    def __init__(
        self,
        robot: RobotSpec,
        motion_reference: MotionReferenceRef,
        play: bool,
    ) -> None:
        self.robot = robot
        self.scene = make_scene(motion_reference, play)
        self.sim = SimSpec(
            physics_dt=0.005,
            decimation=4,
            episode_length_s=10.0,
            profiles={
                "isaacsim": {
                    "use_terrain_physics_material": True,
                    "gpu_max_rigid_patch_count": 10 * 2**15,
                },
                "mjlab": {
                    "iterations": 10,
                    "ls_iterations": 20,
                    "njmax": None if play else 350,
                    "nconmax": None if play else 100,
                    "contact_sensor_maxmatch": 500 if play else 100,
                    "ccd_iterations": 80,
                },
            },
        )
        self.observations = ObservationsCfg(robot, motion_reference)
        self.actions = make_actions(robot)
        self.commands = make_commands()
        self.rewards = RewardsCfg()
        self.events = make_events(play)
        self.curriculum = make_curriculum(play)
        self.terminations = make_terminations(motion_reference)

    def to_task_spec(self, task_id: str, runner: str) -> TaskSpec:
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
                events=self.events,
                curriculum=self.curriculum,
                terminations=self.terminations,
            ),
            agent=AgentSpec(runner=runner),
            engines=("isaacsim", "mjlab"),
        )


__all__ = ["BeyondMimicEnvCfg", "RewardsCfg"]
