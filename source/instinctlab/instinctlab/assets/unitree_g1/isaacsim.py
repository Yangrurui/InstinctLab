"""Isaac Lab configurations for the Unitree G1 catalog.

Simulator-neutral names and parameters live in :mod:`instinctlab.assets.unitree_g1.catalog`.
Isaac Lab is imported lazily so the module remains safe until the application has started.
"""

from __future__ import annotations

from .catalog import (
    G1_29DOF_DEFAULT_JOINT_POS,
    RESOURCE_ROOT,
    _BEYONDMIMIC_JOINT_GROUPS,
    make_g1_29dof_robot_spec,
)

_ISAAC_EXPORTS = frozenset(
    {
        "G1_29DOF_TORSOBASE_CFG",
        "G1_29DOF_TORSOBASE_CLOG_CFG",
        "G1_29DOF_TORSOBASE_POPSICLE_CFG",
        "beyondmimic_g1_29dof_actuators",
        "beyondmimic_g1_29dof_delayed_actuators",
    }
)


def _pattern_fields(patterns: tuple[str, ...]) -> dict[str, dict[str, float]]:
    import re

    joints = make_g1_29dof_robot_spec().joint_properties
    effort: dict[str, float] = {}
    velocity: dict[str, float] = {}
    stiffness: dict[str, float] = {}
    damping: dict[str, float] = {}
    armature: dict[str, float] = {}
    for pattern in patterns:
        rx = re.compile(pattern)
        matched = [joint for joint in joints if rx.fullmatch(joint.name)]
        if not matched:
            raise ValueError(f"no joints match {pattern!r}")
        head = matched[0]
        key = (head.effort_limit, head.velocity_limit, head.stiffness, head.damping, head.armature)
        if any(
            (joint.effort_limit, joint.velocity_limit, joint.stiffness, joint.damping, joint.armature) != key
            for joint in matched
        ):
            raise ValueError(f"{pattern!r} matches joints with different PD")
        effort[pattern] = head.effort_limit
        velocity[pattern] = head.velocity_limit
        stiffness[pattern] = head.stiffness
        damping[pattern] = head.damping
        armature[pattern] = head.armature
    return {
        "effort_limit_sim": effort,
        "velocity_limit_sim": velocity,
        "stiffness": stiffness,
        "damping": damping,
        "armature": armature,
    }


def _load_isaac() -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab_assets import G1_CFG

    global G1_29DOF_TORSOBASE_CFG, G1_29DOF_TORSOBASE_CLOG_CFG, G1_29DOF_TORSOBASE_POPSICLE_CFG
    global beyondmimic_g1_29dof_actuators, beyondmimic_g1_29dof_delayed_actuators

    G1_29DOF_TORSOBASE_CFG = G1_CFG.copy()
    G1_29DOF_TORSOBASE_CFG.spawn = sim_utils.UrdfFileCfg(
        asset_path=str(RESOURCE_ROOT / "urdf" / "g1_29dof_torsobase_simplified.urdf"),
        replace_cylinders_with_capsules=False,
        merge_fixed_joints=False,
        fix_base=False,
        self_collision=True,
        activate_contact_sensors=True,
    )
    G1_29DOF_TORSOBASE_CFG.spawn.joint_drive.gains.stiffness = None  # use value from the URDF file
    G1_29DOF_TORSOBASE_CFG.soft_joint_pos_limit_factor = 0.95
    G1_29DOF_TORSOBASE_CFG.actuators = {
        # NOTE: checked, delayed PD actuator has same time-lag when computing torques; and no lag when
        # the num_pushes does not reach the lag time.
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_.*_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 88.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
                "waist_yaw_joint": 88.0,
            },
            velocity_limit=60.0,
            stiffness={
                ".*_hip_yaw_joint": 90.0,
                ".*_hip_roll_joint": 90.0,
                ".*_hip_pitch_joint": 90.0,
                ".*_knee_joint": 140.0,
                "waist_roll_joint": 60.0,
                "waist_pitch_joint": 60.0,
                "waist_yaw_joint": 90.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_roll_joint": 2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint": 2.5,
                "waist_.*_joint": 2.5,
            },
            armature=0.03,
            min_delay=0,
            max_delay=1,
        ),
        "feet": DelayedPDActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=1.0,
            velocity_limit=60.0,
            armature=0.03,
            min_delay=0,
            max_delay=1,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit=25,
            velocity_limit=60.0,
            stiffness=25,
            damping={
                ".*_shoulder_.*_joint": 1.0,
                ".*_elbow_joint": 1.0,
            },
            armature=0.03,
            min_delay=0,
            max_delay=1,
        ),
        "wrist": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*wrist_roll_joint",
                ".*wrist_pitch_joint",
                ".*wrist_yaw_joint",
            ],
            effort_limit={
                ".*wrist_roll_joint": 25.0,
                ".*wrist_pitch_joint": 5.0,
                ".*wrist_yaw_joint": 5.0,
            },
            velocity_limit=25.0,
            stiffness={
                ".*wrist_roll_joint": 25.0,
                ".*wrist_pitch_joint": 5.0,
                ".*wrist_yaw_joint": 5.0,
            },
            damping={
                ".*wrist_roll_joint": 1.0,
                ".*wrist_pitch_joint": 0.5,
                ".*wrist_yaw_joint": 0.5,
            },
            armature=0.03,
            min_delay=0,
            max_delay=1,
        ),
    }
    G1_29DOF_TORSOBASE_CFG.init_state = G1_CFG.init_state.copy()
    G1_29DOF_TORSOBASE_CFG.init_state.joint_pos = {
        ".*_hip_pitch_joint": -0.20,
        ".*_knee_joint": 0.42,
        ".*_ankle_pitch_joint": -0.23,
        ".*_elbow_joint": 0.87,
        ".*_wrist_roll_joint": 0.0,
        ".*_wrist_pitch_joint": 0.0,
        ".*_wrist_yaw_joint": 0.0,
        "left_shoulder_roll_joint": 0.16,
        "left_shoulder_pitch_joint": 0.35,
        "right_shoulder_roll_joint": -0.16,
        "right_shoulder_pitch_joint": 0.35,
    }

    G1_29DOF_TORSOBASE_CLOG_CFG = G1_29DOF_TORSOBASE_CFG.copy()
    G1_29DOF_TORSOBASE_CLOG_CFG.spawn = sim_utils.UrdfFileCfg(
        asset_path=str(RESOURCE_ROOT / "urdf" / "g1_29dof_torsobase_clog.urdf"),
        replace_cylinders_with_capsules=False,
        merge_fixed_joints=False,
        fix_base=False,
        self_collision=True,
        activate_contact_sensors=True,
        collider_type="convex_decomposition",
    )
    G1_29DOF_TORSOBASE_CLOG_CFG.spawn.joint_drive.gains.stiffness = None  # use value from the URDF file

    beyondmimic_g1_29dof_actuators = {
        name: ImplicitActuatorCfg(joint_names_expr=list(patterns), **_pattern_fields(patterns))
        for name, patterns in _BEYONDMIMIC_JOINT_GROUPS.items()
    }
    beyondmimic_g1_29dof_delayed_actuators = {
        name: DelayedPDActuatorCfg(
            joint_names_expr=list(patterns),
            min_delay=0,
            max_delay=2,
            **_pattern_fields(patterns),
        )
        for name, patterns in _BEYONDMIMIC_JOINT_GROUPS.items()
    }

    robot = make_g1_29dof_robot_spec()
    isaac_asset = robot.asset_for("isaacsim")
    G1_29DOF_TORSOBASE_POPSICLE_CFG = ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            fix_base=isaac_asset.import_options["fix_base"],
            replace_cylinders_with_capsules=isaac_asset.import_options["replace_cylinders_with_capsules"],
            merge_fixed_joints=isaac_asset.import_options["merge_fixed_joints"],
            asset_path=isaac_asset.path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=robot.default_root_pos,
            joint_pos=dict(G1_29DOF_DEFAULT_JOINT_POS),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=robot.soft_joint_pos_limit_factor,
        actuators=beyondmimic_g1_29dof_actuators,
    )


def __getattr__(name: str):
    if name not in _ISAAC_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _load_isaac()
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | _ISAAC_EXPORTS)


__all__ = [
    "G1_29DOF_TORSOBASE_CFG",
    "G1_29DOF_TORSOBASE_CLOG_CFG",
    "G1_29DOF_TORSOBASE_POPSICLE_CFG",
    "beyondmimic_g1_29dof_actuators",
    "beyondmimic_g1_29dof_delayed_actuators",
]
