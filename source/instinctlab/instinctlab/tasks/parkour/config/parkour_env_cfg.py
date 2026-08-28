"""Engine-neutral Parkour environment configuration shared by concrete robot tasks.

The file is intentionally read from top to bottom in the same order in which an
environment is assembled: robot and scene, commands, observations, actions,
rewards, curriculum, terminations, and events. Engine adapters only lower these
declarations; task policy does not live in an adapter.

The terrain recipe is shared and follows main on both engines: 0.05 m height-
field resolution, 20 proportion-allocated columns, and the same named tiles.
Native sensor and solver behavior remains in the corresponding engine bridge.

Both engines share the same motion-reference exhaustion behavior: resample the
reference silently and continue the robot episode.
"""

from __future__ import annotations

import math

from instinctlab import mdp
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
    Grid3dPointsRef,
    MdpSpec,
    MotionReferenceRef,
    NoiseSpec,
    ObsGroupSpec,
    ObsTermSpec,
    RayCasterRef,
    RayPatternRef,
    RewardTermSpec,
    SceneSpec,
    SimSpec,
    TaskSpec,
    VirtualObstacleRef,
    VolumePointsRef,
)
from instinctlab.spec.capability import Requirement
from instinctlab.tasks.terrain import rough_terrain

# -----------------------------------------------------------------------------
# Assets and shared names
# -----------------------------------------------------------------------------

COMMAND = "base_velocity"
AMP_HISTORY = 10

ROBOT = EntityRef("robot", bodies=".*")
FEET = EntityRef("robot", bodies=".*_ankle_roll_link")
LEGS = EntityRef("robot", joints=(".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"))

# One broad scene sensor is sliced by terms. This preserves one common scene
# declaration while each backend resolves the same body selectors natively.
FEET_CONTACT = ContactSensorRef(name="contact_forces", elements=".*_ankle_roll_link")
TORSO_CONTACT = ContactSensorRef(name="contact_forces", elements="torso_link")
UNDESIRED_CONTACT = ContactSensorRef(name="contact_forces", elements="(?!.*_ankle_roll_link).*")


# -----------------------------------------------------------------------------
# Robot, terrain, and sensors
# -----------------------------------------------------------------------------


def _canonical_joints(robot: RobotSpec) -> EntityRef:
    """All policy joints in the shared depth-first order."""
    return EntityRef("robot", joints=tuple(robot.joint_names), preserve_order=True)


PARKOUR_EDGE_CYLINDERS = VirtualObstacleRef(
    name="edges",
    kind="greedy_edge_cylinder",
    cylinder_radius=0.05,
    min_points=2,
)

_FOOT_SCAN_PATTERN = RayPatternRef(kind="grid", resolution=0.12, size=(0.12, 0.0))
LEFT_HEIGHT_SCANNER = RayCasterRef(
    name="left_height_scanner",
    attach="left_ankle_roll_link",
    mode="terrain_height",
    offset=(0.04, 0.0, 20.0),
    pattern=_FOOT_SCAN_PATTERN,
    hit="terrain",
    ray_alignment="yaw",
    miss="infinity",
    engine_max_distances={"mjlab": 10.0},
)
RIGHT_HEIGHT_SCANNER = RayCasterRef(
    name="right_height_scanner",
    attach="right_ankle_roll_link",
    mode="terrain_height",
    offset=(0.04, 0.0, 20.0),
    pattern=_FOOT_SCAN_PATTERN,
    hit="terrain",
    ray_alignment="yaw",
    miss="infinity",
    engine_max_distances={"mjlab": 10.0},
)

def _depth_camera(robot: RobotSpec) -> RayCasterRef:
    camera_links = tuple(
        name
        for name in robot.physical_body_names
        if name not in {"head_link", "left_rubber_hand", "right_rubber_hand"}
    )
    return RayCasterRef(
        name="camera",
        attach="torso_link",
        offset=(0.0487988662332928, 0.01, 0.4378029937970051),
        offset_rot=(0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0),
        offset_convention="world",
        pattern=RayPatternRef(
            kind="pinhole",
            width=64,
            height=36,
            horizontal_fov_deg=89.51,
            vertical_fov_deg=58.29,
            focal_length=1.0,
        ),
        hit=("terrain", *camera_links),
        ray_alignment="base",
        miss="infinity",
        max_distance=2.5,
        min_distance=0.1,
        crop=(18, 0, 16, 16),
        update_period=0.02,
    )

LEG_VOLUME_POINTS = VolumePointsRef(
    name="leg_volume_points",
    attach=("left_ankle_roll_link", "right_ankle_roll_link"),
    grid=Grid3dPointsRef(
        x_min=-0.025,
        x_max=0.12,
        x_num=10,
        y_min=-0.03,
        y_max=0.03,
        y_num=5,
        z_min=-0.063,
        z_max=-0.023,
        z_num=2,
    ),
)


def _scene(motion_reference: MotionReferenceRef, depth_camera: RayCasterRef) -> SceneSpec:
    return SceneSpec(
        terrain=rough_terrain(virtual_obstacles=(PARKOUR_EDGE_CYLINDERS,)),
        contact_sensors=(
            ContactSensorRef(
                name="contact_forces",
                elements=".*",
                track_air_time=True,
                history_length=3,
            ),
        ),
        ray_casters=(LEFT_HEIGHT_SCANNER, RIGHT_HEIGHT_SCANNER, depth_camera),
        motion_references=(motion_reference,),
        volume_points=(LEG_VOLUME_POINTS,),
        env_spacing=2.5,
    )


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------

_OBSTACLE_VELOCITY = {
    "lin_vel_x": (0.45, 0.8),
    "lin_vel_y": (0.0, 0.0),
    "ang_vel_z": (-1.0, 1.0),
}
SHARED_VELOCITY_RANGES = {
    "perlin_rough": {
        "lin_vel_x": (0.45, 1.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-1.0, 1.0),
    },
    "perlin_rough_stand": {
        "lin_vel_x": (0.0, 0.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (0.0, 0.0),
    },
    "square_gaps": dict(_OBSTACLE_VELOCITY),
    "pyramid_stairs": dict(_OBSTACLE_VELOCITY),
    "pyramid_stairs_high": dict(_OBSTACLE_VELOCITY),
    "pyramid_stairs_inv": dict(_OBSTACLE_VELOCITY),
    "pyramid_stairs_inv_high": dict(_OBSTACLE_VELOCITY),
    "boxes": dict(_OBSTACLE_VELOCITY),
    "mesh_boxes": dict(_OBSTACLE_VELOCITY),
    "hf_pyramid_slope_inv": dict(_OBSTACLE_VELOCITY),
}


def _commands() -> dict[str, CommandTermSpec]:
    return {
        COMMAND: CommandTermSpec(
            kind="pose_velocity",
            level=Requirement.REQUIRED,
            params={
                "entity": "robot",
                "resampling_time_range": (8.0, 12.0),
                "lin_vel_x": (0.0, 0.0),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
                "velocity_control_stiffness": 2.0,
                "heading_control_stiffness": 2.0,
                "only_positive_lin_vel_x": True,
                "rel_standing_envs": 0.05,
                "random_velocity_terrain": ["perlin_rough_stand"],
                "velocity_ranges": SHARED_VELOCITY_RANGES,
                "lin_vel_threshold": 0.0,
                "ang_vel_threshold": 0.0,
                "lin_vel_metrics_std": 0.5,
                "ang_vel_metrics_std": 0.5,
                "target_dis_threshold": 0.4,
            },
        )
    }


# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------


def _policy_observations(*, corrupt: bool, joints: EntityRef, depth_camera: RayCasterRef) -> ObsGroupSpec:
    noise = (lambda lo, hi: NoiseSpec("uniform", lo, hi)) if corrupt else (lambda lo, hi: None)
    terms = {
        "base_ang_vel": ObsTermSpec(func=mdp.base_ang_vel, noise=noise(-0.2, 0.2), scale=0.25, history_length=8),
        "projected_gravity": ObsTermSpec(func=mdp.projected_gravity, noise=noise(-0.05, 0.05), history_length=8),
        "velocity_commands": ObsTermSpec(
            func=mdp.generated_commands,
            params={"command_name": COMMAND},
            history_length=8,
        ),
        "joint_pos": ObsTermSpec(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": joints},
            noise=noise(-0.01, 0.01),
            history_length=8,
        ),
        "joint_vel": ObsTermSpec(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": joints},
            noise=noise(-0.5, 0.5),
            scale=0.05,
            history_length=8,
        ),
        "actions": ObsTermSpec(func=mdp.last_action, history_length=8),
        "depth_image": ObsTermSpec(
            func=mdp.DelayedDepthImage,
            params={
                "sensor": depth_camera,
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "history_length": 37,
                "blur_kernel_size": 3,
                "blur_sigma": 1.0,
            },
            history_length=0,
            level=Requirement.REQUIRED,
        ),
    }
    if not corrupt:
        terms = {
            "base_lin_vel": ObsTermSpec(func=mdp.base_lin_vel, history_length=8),
            **terms,
        }
    return ObsGroupSpec(terms=terms, concatenate_terms=False, enable_corruption=corrupt)


def _amp_policy_observations(joints: EntityRef) -> ObsGroupSpec:
    return ObsGroupSpec(
        terms={
            "projected_gravity": ObsTermSpec(func=mdp.projected_gravity, history_length=AMP_HISTORY),
            "joint_pos_rel": ObsTermSpec(
                func=mdp.joint_pos_rel,
                params={"asset_cfg": joints},
                history_length=AMP_HISTORY,
            ),
            "joint_vel": ObsTermSpec(
                func=mdp.joint_vel_rel,
                params={"asset_cfg": joints},
                scale=0.05,
                history_length=AMP_HISTORY,
            ),
            "base_lin_vel": ObsTermSpec(func=mdp.base_lin_vel, history_length=AMP_HISTORY),
            "base_ang_vel": ObsTermSpec(func=mdp.base_ang_vel, history_length=AMP_HISTORY),
        },
        concatenate_terms=False,
        enable_corruption=False,
    )


def _amp_reference_observations(joints: EntityRef, sensor: MotionReferenceRef) -> ObsGroupSpec:
    params = {"sensor": sensor, "asset_cfg": joints}
    return ObsGroupSpec(
        terms={
            "projected_gravity": ObsTermSpec(
                func=mdp.projected_gravity_from_reference,
                params=params,
                history_length=AMP_HISTORY,
            ),
            "joint_pos_rel": ObsTermSpec(
                func=mdp.joint_pos_rel_from_reference,
                params=params,
                history_length=AMP_HISTORY,
            ),
            "joint_vel": ObsTermSpec(
                func=mdp.joint_vel_rel_from_reference,
                params=params,
                scale=0.05,
                history_length=AMP_HISTORY,
            ),
            "base_lin_vel": ObsTermSpec(
                func=mdp.base_lin_vel_from_reference,
                params=params,
                history_length=AMP_HISTORY,
            ),
            "base_ang_vel": ObsTermSpec(
                func=mdp.base_ang_vel_from_reference,
                params=params,
                history_length=AMP_HISTORY,
            ),
        },
        concatenate_terms=False,
        enable_corruption=False,
    )


class ObservationGroupCfg:
    def __init__(self, group: ObsGroupSpec) -> None:
        self.enable_corruption = group.enable_corruption
        self.concatenate_terms = group.concatenate_terms
        for name, term in group.terms.items():
            setattr(self, name, term)

    def to_spec(self) -> ObsGroupSpec:
        terms = dict(vars(self))
        enable_corruption = terms.pop("enable_corruption")
        concatenate_terms = terms.pop("concatenate_terms")
        return ObsGroupSpec(
            terms=terms,
            enable_corruption=enable_corruption,
            concatenate_terms=concatenate_terms,
        )


class ObservationsCfg:
    def __init__(self, joints: EntityRef, motion_reference: MotionReferenceRef, depth_camera: RayCasterRef) -> None:
        self.policy = ObservationGroupCfg(
            _policy_observations(corrupt=True, joints=joints, depth_camera=depth_camera)
        )
        self.critic = ObservationGroupCfg(
            _policy_observations(corrupt=False, joints=joints, depth_camera=depth_camera)
        )
        self.amp_policy = ObservationGroupCfg(_amp_policy_observations(joints))
        self.amp_reference = ObservationGroupCfg(_amp_reference_observations(joints, motion_reference))

    def to_dict(self) -> dict[str, ObsGroupSpec]:
        return {
            "policy": self.policy.to_spec(),
            "critic": self.critic.to_spec(),
            "amp_policy": self.amp_policy.to_spec(),
            "amp_reference": self.amp_reference.to_spec(),
        }


# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------


def _action_scale(robot: RobotSpec) -> dict[str, float]:
    return {joint.name: joint.action_scale for joint in robot.joint_properties}


def _actions(robot: RobotSpec, joints: EntityRef) -> dict[str, ActionTermSpec]:
    return {
        "joint_pos": ActionTermSpec(
            kind="joint_position",
            target=joints,
            params={"scale": _action_scale(robot), "use_default_offset": True},
        )
    }


# -----------------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------------


def _velocity_limits(robot: RobotSpec) -> tuple[float, ...]:
    return tuple(joint.velocity_limit for joint in robot.joint_properties)


class G1Rewards:
    def __init__(self, joints: EntityRef, velocity_limits: tuple[float, ...]) -> None:
        terms = {
        # Task rewards
        "track_lin_vel_xy_exp": RewardTermSpec(
            func=mdp.track_lin_vel_xy_exp,
            weight=2.0,
            params={"command_name": COMMAND, "std": 0.5},
        ),
        "track_ang_vel_z_exp": RewardTermSpec(
            func=mdp.track_ang_vel_z_exp,
            weight=2.0,
            params={"command_name": COMMAND, "std": 0.5},
        ),
        "heading_error": RewardTermSpec(func=mdp.heading_error, weight=-1.0, params={"command_name": COMMAND}),
        "dont_wait": RewardTermSpec(func=mdp.dont_wait, weight=-0.5, params={"command_name": COMMAND}),
        "is_alive": RewardTermSpec(func=mdp.is_alive, weight=3.0),
        "stand_still": RewardTermSpec(
            func=mdp.stand_still_when_idle,
            weight=-0.3,
            params={"command_name": COMMAND, "offset": 4.0},
        ),
        # Regularization rewards
        "volume_points_penetration": RewardTermSpec(
            func=mdp.volume_points_penetration,
            weight=-8.0,
            params={"sensor": LEG_VOLUME_POINTS},
            level=Requirement.REQUIRED,
        ),
        "feet_air_time": RewardTermSpec(
            func=mdp.feet_air_time,
            weight=0.5,
            params={
                "command_name": COMMAND,
                "sensor": FEET_CONTACT,
                "vel_threshold": 0.15,
            },
        ),
        "feet_slide": RewardTermSpec(
            kind="contact_slide",
            weight=-0.4,
            params={"sensor_cfg": FEET_CONTACT, "asset_cfg": FEET, "threshold": 1.0},
            level=Requirement.REQUIRED,
        ),
        "joint_deviation_hip": RewardTermSpec(
            func=mdp.joint_deviation_square,
            weight=-0.5,
            params={"asset_cfg": EntityRef("robot", joints=(".*_hip_yaw_joint", ".*_hip_roll_joint"))},
        ),
        "ang_vel_xy_l2": RewardTermSpec(func=mdp.ang_vel_xy_l2, weight=-0.05),
        "dof_torques_l2": RewardTermSpec(
            kind="joint_torques_l2",
            weight=-1.5e-7,
            params={"asset_cfg": LEGS},
            level=Requirement.REQUIRED,
        ),
        "dof_acc_l2": RewardTermSpec(
            kind="joint_acc_l2",
            weight=-1.25e-7,
            params={"asset_cfg": EntityRef("robot", joints=".*")},
            level=Requirement.REQUIRED,
        ),
        "dof_vel_l2": RewardTermSpec(
            func=mdp.joint_vel_l2,
            weight=-1e-4,
            params={"asset_cfg": EntityRef("robot", joints=".*")},
        ),
        "action_rate_l2": RewardTermSpec(func=mdp.action_rate_l2, weight=-0.005),
        "flat_orientation_l2": RewardTermSpec(func=mdp.flat_orientation_l2, weight=-3.0),
        "pelvis_orientation_l2": RewardTermSpec(
            func=mdp.link_orientation,
            weight=-3.0,
            params={"asset_cfg": EntityRef("robot", bodies="pelvis")},
        ),
        "feet_flat_ori": RewardTermSpec(
            func=mdp.feet_orientation_contact,
            weight=-0.4,
            params={"sensor": FEET_CONTACT, "asset_cfg": FEET},
        ),
        "feet_at_plane": RewardTermSpec(
            func=mdp.feet_at_plane,
            weight=-0.1,
            params={
                "sensor": FEET_CONTACT,
                "left_scanner": LEFT_HEIGHT_SCANNER,
                "right_scanner": RIGHT_HEIGHT_SCANNER,
                "asset_cfg": FEET,
                "height_offset": 0.058,
            },
            level=Requirement.REQUIRED,
        ),
        "feet_close_xy": RewardTermSpec(
            func=mdp.feet_close_xy_gauss,
            weight=0.4,
            params={"threshold": 0.12, "std": math.sqrt(0.05), "asset_cfg": FEET},
        ),
        "energy": RewardTermSpec(
            kind="motors_power_square",
            weight=-5e-5,
            params={"asset_cfg": LEGS, "normalize_by_stiffness": True},
            level=Requirement.REQUIRED,
        ),
        "freeze_upper_body": RewardTermSpec(
            func=mdp.joint_deviation_l1,
            weight=-0.004,
            params={
                "asset_cfg": EntityRef(
                    "robot",
                    joints=(".*_shoulder_.*", ".*_elbow_.*", ".*_wrist.*", "waist_.*"),
                )
            },
        ),
        # Safety rewards
        "dof_pos_limits": RewardTermSpec(
            func=mdp.joint_pos_limits,
            weight=-1.0,
            params={"asset_cfg": EntityRef("robot", joints=".*")},
        ),
        "dof_vel_limits": RewardTermSpec(
            func=mdp.joint_vel_limits,
            weight=-1.0,
            params={"soft_ratio": 0.9, "limits": velocity_limits, "asset_cfg": joints},
        ),
        "torque_limits": RewardTermSpec(
            kind="applied_torque_limits_by_ratio",
            weight=-0.01,
            params={"asset_cfg": EntityRef("robot", joints=".*"), "limit_ratio": 0.8},
            level=Requirement.REQUIRED,
        ),
        "undesired_contacts": RewardTermSpec(
            kind="undesired_contacts",
            weight=-1.0,
            params={"sensor": UNDESIRED_CONTACT},
            level=Requirement.REQUIRED,
        ),
        }
        for name, term in terms.items():
            setattr(self, name, term)

    def to_dict(self) -> dict[str, dict[str, RewardTermSpec]]:
        return {"rewards": dict(vars(self))}


# -----------------------------------------------------------------------------
# Curriculum, terminations, and events
# -----------------------------------------------------------------------------


def _curriculum() -> dict[str, CurriculumTermSpec]:
    return {
        "terrain_levels": CurriculumTermSpec(
            func=mdp.tracking_exp_vel,
            params={
                "command_name": COMMAND,
                "lin_vel_threshold": (0.3, 0.6),
                "ang_vel_threshold": (0.0, 0.0),
            },
            level=Requirement.REQUIRED,
        )
    }


def _terminations(motion_reference: MotionReferenceRef) -> dict[str, DoneTermSpec]:
    return {
        "time_out": DoneTermSpec(func=mdp.time_out, time_out=True),
        "terrain_out_of_bounds": DoneTermSpec(
            func=mdp.terrain_out_of_bounds,
            time_out=True,
            params={"distance_buffer": 2.0},
        ),
        "base_contact": DoneTermSpec(kind="illegal_contact", params={"sensor": TORSO_CONTACT}),
        "bad_orientation": DoneTermSpec(func=mdp.bad_orientation, params={"limit_angle": 1.0}),
        "root_height": DoneTermSpec(
            func=mdp.root_height_below_env_origin_minimum,
            params={"minimum_height": 0.5},
        ),
        "dataset_exhausted": DoneTermSpec(
            func=mdp.dataset_exhausted,
            time_out=True,
            params={
                "sensor": motion_reference,
                "print_reason": False,
                "reset_without_notice": True,
            },
        ),
    }


def _events() -> dict[str, EventTermSpec]:
    return {
        "physics_material": EventTermSpec(
            kind="randomize_friction",
            mode="startup",
            target=ROBOT,
            params={
                "static_friction_range": (0.3, 1.6),
                "dynamic_friction_range": (0.3, 1.6),
            },
            engine_params={
                "isaacsim": {
                    "restitution_range": (0.05, 0.5),
                    "num_buckets": 64,
                    "make_consistent": True,
                }
            },
        ),
        "reset_base": EventTermSpec(
            kind="reset_root_state_uniform",
            mode="reset",
            params={
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
                "velocity_range": {
                    "x": (-0.2, 0.2),
                    "y": (-0.2, 0.2),
                    "z": (-0.2, 0.2),
                    "roll": (-0.2, 0.2),
                    "pitch": (-0.2, 0.2),
                    "yaw": (-0.2, 0.2),
                },
            },
        ),
        "register_virtual_obstacles": EventTermSpec(
            kind="register_virtual_obstacles",
            mode="startup",
            params={"sensor": LEG_VOLUME_POINTS},
            level=Requirement.REQUIRED,
        ),
        "reset_robot_joints": EventTermSpec(
            kind="reset_joints_by_offset",
            mode="reset",
            params={"position_range": (-0.15, 0.15), "velocity_range": (0.0, 0.0)},
        ),
    }


# -----------------------------------------------------------------------------
# Public factory
# -----------------------------------------------------------------------------


class ParkourEnvCfg:
    def __init__(self, robot: RobotSpec, motion_reference: MotionReferenceRef) -> None:
        joints = _canonical_joints(robot)
        depth_camera = _depth_camera(robot)
        self.robot = robot
        self.scene = _scene(motion_reference, depth_camera)
        self.sim = SimSpec(
            physics_dt=0.005,
            decimation=4,
            episode_length_s=20.0,
            profiles={
                "mjlab": {
                    "contact_sensor_maxmatch": 128,
                    "ccd_iterations": 128,
                }
            },
        )
        self.commands = _commands()
        self.observations = ObservationsCfg(joints, motion_reference, depth_camera)
        self.actions = _actions(robot, joints)
        self.rewards = G1Rewards(joints, _velocity_limits(robot))
        self.curriculum = _curriculum()
        self.terminations = _terminations(motion_reference)
        self.events = _events()

    def to_task_spec(self, task_id: str) -> TaskSpec:
        return TaskSpec(
            task_id=task_id,
            robot=self.robot,
            scene=self.scene,
            sim=self.sim,
            mdp=MdpSpec(
                commands=self.commands,
                observations=self.observations.to_dict(),
                actions=self.actions,
                rewards=self.rewards.to_dict(),
                curriculum=self.curriculum,
                terminations=self.terminations,
                events=self.events,
            ),
            agent=AgentSpec(
                runner="instinctlab.tasks.parkour.config.g1.agents.instinct_rl_amp_cfg:G1ParkourTargetPPORunnerCfg"
            ),
            engines=("isaacsim", "mjlab"),
        )


__all__ = ["ParkourEnvCfg"]
