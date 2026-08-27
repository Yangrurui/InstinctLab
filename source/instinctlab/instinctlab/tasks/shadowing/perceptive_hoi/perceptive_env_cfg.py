"""Engine-neutral Perceptive HOI Shadowing environment configuration."""

from __future__ import annotations

from instinctlab import mdp
from instinctlab.assets.unitree_g1.catalog import G1_29DOF_LINKS
from instinctlab.sim.robot_spec import RobotSpec
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
    RayCasterRef,
    RayPatternRef,
    RewardTermSpec,
    RigidObjectRef,
    SceneSpec,
    SimSpec,
    TaskSpec,
    TerrainSpec,
)
from instinctlab.spec.capability import Requirement

PROPRIO_HISTORY_LENGTH = 8
NON_SUPPORT_CONTACTS = (
    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
    r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
)


def make_contact_sensor(elements: str = ".*") -> ContactSensorRef:
    return ContactSensorRef(
        name="contact_forces",
        elements=elements,
        track_air_time=True,
        air_time_force_threshold=1.0,
        history_length=3,
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
        "position_b_ref_command": CommandTermSpec(
            kind="shadow_position_reference",
            params={
                "motion_reference": "motion_reference",
                "current_state_command": False,
                "realtime_mode": True,
                "anchor_frame": "reference",
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


def make_policy_proprioception(joints: EntityRef) -> dict[str, ObsTermSpec]:
    return {
        "projected_gravity": ObsTermSpec(
            func=mdp.projected_gravity,
            noise=NoiseSpec("uniform", -0.05, 0.05),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "base_ang_vel": ObsTermSpec(
            func=mdp.base_ang_vel,
            noise=NoiseSpec("uniform", -0.2, 0.2),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_pos": ObsTermSpec(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": joints},
            noise=NoiseSpec("uniform", -0.01, 0.01),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_vel": ObsTermSpec(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": joints},
            noise=NoiseSpec("uniform", -0.5, 0.5),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "last_action": ObsTermSpec(
            func=mdp.last_action,
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
    }


def make_critic_proprioception(joints: EntityRef) -> dict[str, ObsTermSpec]:
    return {
        "base_ang_vel": ObsTermSpec(func=mdp.base_ang_vel, history_length=PROPRIO_HISTORY_LENGTH),
        "joint_pos": ObsTermSpec(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": joints},
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_vel": ObsTermSpec(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": joints},
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "last_action": ObsTermSpec(func=mdp.last_action, history_length=PROPRIO_HISTORY_LENGTH),
    }


def make_policy_reference_observations() -> dict[str, ObsTermSpec]:
    return {
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
            params={"command_name": "position_b_ref_command"},
            noise=NoiseSpec("uniform", -0.25, 0.25),
        ),
        "rotation_ref": ObsTermSpec(
            func=mdp.generated_commands,
            params={"command_name": "rotation_ref_command"},
            noise=NoiseSpec("uniform", -0.05, 0.05),
        ),
    }


def make_critic_reference_observations() -> dict[str, ObsTermSpec]:
    return {
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
    }


def make_camera(object_names: tuple[str, ...]) -> RayCasterRef:
    return RayCasterRef(
        name="camera",
        attach="torso_link",
        offset=(0.0487988662332928, 0.015, 0.4378029937970051),
        offset_rot=(0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0),
        offset_convention="world",
        pattern=RayPatternRef(
            kind="pinhole",
            width=48,
            height=27,
            horizontal_fov_deg=87.0,
            vertical_fov_deg=58.0,
            focal_length=1.0,
        ),
        hit=("terrain", *G1_29DOF_LINKS, *object_names),
        ray_alignment="base",
        miss="infinity",
        max_distance=1.0e6,
        min_distance=0.05,
        crop=(2, 2, 2, 2),
        update_period=1.0 / 60.0,
    )


def make_height_scanner() -> RayCasterRef:
    return RayCasterRef(
        name="height_scanner",
        mode="terrain_height",
        attach="torso_link",
        offset=(0.0, 0.0, 20.0),
        pattern=RayPatternRef(kind="grid", resolution=0.1, size=(1.6, 1.0)),
        hit="terrain",
        ray_alignment="yaw",
        miss="infinity",
        max_distance=1.0e6,
        engine_max_distances={"isaacsim": 1.0e6, "mjlab": 5.0},
        update_period=0.02,
    )


def make_scene(
    motion_reference: MotionReferenceRef,
    objects: tuple[RigidObjectRef, ...],
    play: bool,
) -> SceneSpec:
    object_names = tuple(obj.name for obj in objects)
    return SceneSpec(
        terrain=TerrainSpec(kind="plane"),
        contact_sensors=(make_contact_sensor(),),
        ray_casters=(make_camera(object_names), make_height_scanner()),
        motion_references=(motion_reference,),
        rigid_objects=objects,
        env_spacing=2.5 if play else 4.0,
    )


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
    def __init__(
        self,
        robot: RobotSpec,
        motion_reference: MotionReferenceRef,
        objects: tuple[RigidObjectRef, ...],
    ) -> None:
        joints = EntityRef("robot", joints=tuple(robot.joint_names), preserve_order=True)
        links = EntityRef("robot", bodies=motion_reference.links, preserve_order=True)
        object_names = tuple(obj.name for obj in objects)
        policy_terms = make_policy_reference_observations()
        policy_terms["depth_image"] = ObsTermSpec(
            kind="shadow_depth_image",
            params={"sensor": make_camera(object_names)},
        )
        policy_terms.update(make_policy_proprioception(joints))

        critic_terms = make_critic_reference_observations()
        del critic_terms["rotation_ref"]
        critic_terms["link_pos"] = ObsTermSpec(
            kind="shadow_link_position",
            params={"motion_reference": motion_reference, "asset_cfg": links},
        )
        critic_terms["link_rot"] = ObsTermSpec(
            kind="shadow_link_rotation",
            params={"motion_reference": motion_reference, "asset_cfg": links},
        )
        critic_terms["height_scan"] = ObsTermSpec(
            kind="shadow_height_scan",
            params={"sensor": make_height_scanner()},
        )
        critic_terms["base_lin_vel"] = ObsTermSpec(
            kind="shadow_base_linear_velocity",
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        critic_terms.update(make_critic_proprioception(joints))
        self.policy = ObservationGroupCfg(policy_terms, enable_corruption=True)
        self.critic = ObservationGroupCfg(critic_terms, enable_corruption=False)

    def to_dict(self) -> dict[str, ObsGroupSpec]:
        return {
            "policy": self.policy.to_spec(),
            "critic": self.critic.to_spec(),
        }


def make_events(play: bool) -> dict[str, EventTermSpec]:
    reset_params = {
        "position_offset": (0.0, 0.0, 0.0),
        "dof_vel_ratio": 1.0,
        "base_lin_vel_ratio": 1.0,
        "base_ang_vel_ratio": 1.0,
        "randomize_joint_pos_range": (-0.1, 0.1),
        "randomize_pose_range": {
            "x": (-0.15, 0.15),
            "y": (-0.15, 0.15),
            "z": (0.0, 0.0),
        },
        "randomize_velocity_range": {},
    }
    events = {
        "physics_material": EventTermSpec(
            kind="randomize_friction",
            mode="startup",
            target=EntityRef("robot", bodies=".*"),
            engine_params={
                "isaacsim": {
                    "static_friction_range": (1.25, 2.0),
                    "dynamic_friction_range": (1.2, 1.8),
                    "restitution_range": (0.0, 0.5),
                    "num_buckets": 64,
                    "make_consistent": True,
                },
                "mjlab": {
                    "ranges": {0: (1.25, 2.0), 1: (1.2, 1.8), 2: (0.0, 0.5)},
                    "operation": "abs",
                    "distribution": "uniform",
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
        "randomize_ray_offsets": EventTermSpec(
            kind="shadow_randomize_ray_offsets",
            mode="startup",
            params={
                "offset_pose_ranges": {
                    "x": (-0.01, 0.01),
                    "y": (-0.01, 0.01),
                    "z": (-0.01, 0.01),
                    "roll": (-0.034906585, 0.034906585),
                    "pitch": (-0.174532925, 0.174532925),
                    "yaw": (-0.034906585, 0.034906585),
                }
            },
        ),
        "randomize_actuator_gains": EventTermSpec(
            kind="randomize_actuator_gains",
            mode="startup",
            params={
                "stiffness_range": (0.8, 1.2),
                "damping_range": (0.9, 1.1),
                "operation": "scale",
            },
        ),
        "randomize_rigid_body_mass": EventTermSpec(
            kind="shadow_randomize_body_inertia",
            mode="startup",
            target=EntityRef("robot", bodies=("torso_link", ".*ankle.*", ".*wrist.*")),
            params={"add_range": (0.8, 1.2), "operation": "scale"},
        ),
        "reset_robot": EventTermSpec(
            kind="shadow_reset_robot_from_reference",
            mode="reset",
            params=reset_params,
            engine_params={"isaacsim": {"root_velocity_frame": "com"}},
        ),
        "reset_rigid_objects_state_by_reference": EventTermSpec(
            kind="shadow_reset_objects_from_reference",
            mode="reset",
            engine_params={"isaacsim": {"root_velocity_frame": "com"}},
        ),
        "update_rigid_objects_state_by_reference": EventTermSpec(
            kind="shadow_update_objects_from_reference",
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params={"invalid_object_pos": (0.0, 0.0, -1.0)},
            engine_params={"isaacsim": {"root_velocity_frame": "com"}},
        ),
    }
    if not play:
        events["bin_fail_counter_smoothing"] = EventTermSpec(
            kind="shadow_smooth_bin_failures",
            mode="interval",
            interval_range_s=(0.02, 0.02),
        )
    return events


class RewardsCfg:
    def __init__(self) -> None:
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
            "action_rate_l2": RewardTermSpec(
                func=mdp.action_rate_l2,
                weight=-0.1,
                level=Requirement.REQUIRED,
            ),
            "joint_limit": RewardTermSpec(
                func=mdp.joint_pos_limits,
                weight=-10.0,
                level=Requirement.REQUIRED,
            ),
            "undesired_contacts": RewardTermSpec(
                kind="shadow_undesired_contacts",
                weight=-0.1,
                params={
                    "threshold": 1.0,
                    "sensor": make_contact_sensor(NON_SUPPORT_CONTACTS),
                },
                level=Requirement.REQUIRED,
            ),
            "applied_torque_limits_by_ratio": RewardTermSpec(
                kind="shadow_torque_limit_ratio",
                weight=-0.05,
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
        "illegal_reset_contact": DoneTermSpec(
            kind="shadow_illegal_reset_contact",
            params={
                "sensor": make_contact_sensor(NON_SUPPORT_CONTACTS),
                "threshold": 500.0,
                "episode_length_threshold": 2,
            },
        ),
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
    }


def make_curriculum(play: bool) -> dict[str, CurriculumTermSpec]:
    if play:
        return {}
    return {"beyond_adaptive_sampling": CurriculumTermSpec(kind="shadow_adaptive_sampling")}


class PerceptiveHoiShadowingEnvCfg:
    def __init__(
        self,
        robot: RobotSpec,
        motion_reference: MotionReferenceRef,
        objects: tuple[RigidObjectRef, ...],
        play: bool,
    ) -> None:
        self.robot = robot
        self.scene = make_scene(motion_reference, objects, play)
        self.sim = SimSpec(
            physics_dt=0.005,
            decimation=4,
            episode_length_s=10.0,
            profiles={
                "isaacsim": {
                    "use_terrain_physics_material": True,
                    "gpu_max_rigid_patch_count": 10 * 2**15,
                    "gpu_max_rigid_contact_count": 2**27,
                    "gpu_collision_stack_size": 2**27,
                },
                "mjlab": {
                    "iterations": 10,
                    "ls_iterations": 20,
                    "njmax": 700,
                    "nconmax": 256,
                    "contact_sensor_maxmatch": 256,
                    "ccd_iterations": 128,
                    "jacobian": "sparse",
                },
            },
        )
        self.observations = ObservationsCfg(robot, motion_reference, objects)
        self.actions = make_actions(robot)
        self.commands = make_commands()
        self.rewards = RewardsCfg()
        self.events = make_events(play)
        self.curriculum = make_curriculum(play)
        self.terminations = make_terminations(motion_reference)
        self.play = play

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
            engine_extras={
                "isaacsim": {
                    "shadowing_family": "perceptive_hoi",
                    "play": self.play,
                    "reference_num_envs": 4096,
                },
                "mjlab": {
                    "shadowing_family": "perceptive_hoi",
                    "play": self.play,
                    "reference_num_envs": 4096,
                },
            },
        )


__all__ = ["PerceptiveHoiShadowingEnvCfg", "RewardsCfg"]
