"""Isaac Lab assets and actuator configurations for Unitree G1.

Isaac Lab is imported lazily so importing the engine-neutral G1 package does
not start Kit.  Actuator groups are explicit here; they are never inferred by
an engine adapter from the shared robot interface.
"""

from __future__ import annotations

from . import (
    ARMATURE_4010,
    ARMATURE_5020,
    ARMATURE_7520_14,
    ARMATURE_7520_22,
    DAMPING_4010,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
    G1_29DOF_DEFAULT_JOINT_POS,
    RESOURCE_ROOT,
    STIFFNESS_4010,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
    make_g1_29dof_parkour_robot_spec,
    make_g1_29dof_robot_spec,
    make_g1_29dof_shadowing_robot_spec,
)

ARTICULATIONS = {
    "popsicle_torsobase_v1": "G1_29DOF_TORSOBASE_POPSICLE_CFG",
    "popsicle_torsobase_shadowing_v1": "G1_29DOF_TORSOBASE_POPSICLE_SHADOWING_CFG",
    "popsicle_torsobase_parkour_v1": "G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG",
}

_ISAAC_EXPORTS = frozenset(
    {
        "G1_29DOF_TORSOBASE_CFG",
        "G1_29DOF_TORSOBASE_CLOG_CFG",
        "G1_29DOF_TORSOBASE_POPSICLE_CFG",
        "G1_29DOF_TORSOBASE_POPSICLE_SHADOWING_CFG",
        "G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG",
        "beyondmimic_g1_29dof_actuators",
        "beyondmimic_g1_29dof_delayed_actuators",
    }
)
_ISAAC_LOADED = False


def _beyondmimic_actuators(actuator_cfg_type) -> dict[str, object]:
    """Build Isaac's five native actuator groups without shared-table inference."""
    return {
        "legs": actuator_cfg_type(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": actuator_cfg_type(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist": actuator_cfg_type(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": actuator_cfg_type(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": actuator_cfg_type(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    }


def _load_isaac() -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab_assets import G1_CFG

    global _ISAAC_LOADED
    if _ISAAC_LOADED:
        return

    global G1_29DOF_TORSOBASE_CFG, G1_29DOF_TORSOBASE_CLOG_CFG, G1_29DOF_TORSOBASE_POPSICLE_CFG
    global G1_29DOF_TORSOBASE_POPSICLE_SHADOWING_CFG, G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG
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

    beyondmimic_g1_29dof_actuators = _beyondmimic_actuators(ImplicitActuatorCfg)
    beyondmimic_g1_29dof_delayed_actuators = {
        name: DelayedPDActuatorCfg(
            **{
                field: getattr(cfg, field)
                for field in (
                    "joint_names_expr",
                    "effort_limit_sim",
                    "velocity_limit_sim",
                    "stiffness",
                    "damping",
                    "armature",
                )
            },
            min_delay=0,
            max_delay=2,
        )
        for name, cfg in beyondmimic_g1_29dof_actuators.items()
    }

    robot = make_g1_29dof_robot_spec()
    G1_29DOF_TORSOBASE_POPSICLE_CFG = ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            fix_base=False,
            replace_cylinders_with_capsules=True,
            merge_fixed_joints=False,
            asset_path=str(RESOURCE_ROOT / "urdf" / "g1_29dof_torsobase_popsicle.urdf"),
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

    shadowing_robot = make_g1_29dof_shadowing_robot_spec()
    shadowing_asset = shadowing_robot.asset_for("isaacsim")
    G1_29DOF_TORSOBASE_POPSICLE_SHADOWING_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG.copy()
    G1_29DOF_TORSOBASE_POPSICLE_SHADOWING_CFG.spawn = (
        G1_29DOF_TORSOBASE_POPSICLE_SHADOWING_CFG.spawn.replace(
            asset_path=shadowing_asset.path,
            merge_fixed_joints=shadowing_asset.import_options["merge_fixed_joints"],
        )
    )

    parkour_robot = make_g1_29dof_parkour_robot_spec()
    parkour_asset = parkour_robot.asset_for("isaacsim")
    G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG.copy()
    G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG.spawn = G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG.spawn.replace(
        asset_path=parkour_asset.path,
        merge_fixed_joints=parkour_asset.import_options["merge_fixed_joints"],
    )
    G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG.init_state = (
        G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG.init_state.replace(pos=parkour_robot.default_root_pos)
    )
    _ISAAC_LOADED = True


def _with_delay(actuators: dict[str, object], delay: tuple[int, int]) -> dict[str, object]:
    if delay == (0, 0):
        return actuators
    from isaaclab.actuators import DelayedPDActuatorCfg

    fields = (
        "joint_names_expr",
        "effort_limit_sim",
        "velocity_limit_sim",
        "stiffness",
        "damping",
        "armature",
    )
    return {
        name: DelayedPDActuatorCfg(
            **{field: getattr(cfg, field) for field in fields},
            min_delay=delay[0],
            max_delay=delay[1],
        )
        for name, cfg in actuators.items()
    }


def articulation(variant: str, robot) -> object:
    """Build one registered G1 variant as an Isaac Lab articulation."""
    try:
        config_name = ARTICULATIONS[variant]
    except KeyError:
        raise KeyError(f"Unknown Isaac Unitree G1 variant {variant!r}; registered: {sorted(ARTICULATIONS)}") from None
    _load_isaac()
    cfg = globals()[config_name].copy()
    asset = robot.asset_for("isaacsim")
    spawn_updates = {"asset_path": asset.path}
    for name in ("merge_fixed_joints", "fix_base", "replace_cylinders_with_capsules"):
        if name in asset.import_options:
            spawn_updates[name] = asset.import_options[name]
    cfg.spawn = cfg.spawn.replace(**spawn_updates)
    cfg.init_state = cfg.init_state.replace(pos=robot.default_root_pos)
    cfg.actuators = _with_delay(cfg.actuators, robot.actuator_delay)
    return cfg


def __getattr__(name: str):
    if name not in _ISAAC_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _load_isaac()
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | _ISAAC_EXPORTS)


__all__ = [
    "ARTICULATIONS",
    "G1_29DOF_TORSOBASE_CFG",
    "G1_29DOF_TORSOBASE_CLOG_CFG",
    "G1_29DOF_TORSOBASE_POPSICLE_CFG",
    "G1_29DOF_TORSOBASE_POPSICLE_SHADOWING_CFG",
    "G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG",
    "beyondmimic_g1_29dof_actuators",
    "beyondmimic_g1_29dof_delayed_actuators",
    "articulation",
]
