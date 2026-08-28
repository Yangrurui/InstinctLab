"""Explicit Isaac Lab configuration for the 23-DOF PND Adam SP robot."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

RESOURCE_ROOT = Path(__file__).resolve().parent.parent / "resources" / "adam_sp"
ADAM_SP_23DOF_URDF = RESOURCE_ROOT / "urdf" / "adam_sp_23_dof.urdf"

ADAM_SP_23DOF_ACTUATORS = {
    "hip_pitch_knee": ImplicitActuatorCfg(
        joint_names_expr=[
            "left_hip_pitch_joint",
            "left_knee_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
        ],
        stiffness=300.0,
        damping=7.0,
        effort_limit_sim=230.0,
        velocity_limit_sim=15.0,
        armature=0.13426,
    ),
    "hip_roll": ImplicitActuatorCfg(
        joint_names_expr=["left_hip_roll_joint", "right_hip_roll_joint"],
        stiffness=600.0,
        damping=10.0,
        effort_limit_sim=180.0,
        velocity_limit_sim=8.0,
        armature=0.281573,
    ),
    "hip_yaw": ImplicitActuatorCfg(
        joint_names_expr=["left_hip_yaw_joint", "right_hip_yaw_joint"],
        stiffness=300.0,
        damping=2.0,
        effort_limit_sim=105.0,
        velocity_limit_sim=8.0,
        armature=0.23409,
    ),
    "ankle_pitch": ImplicitActuatorCfg(
        joint_names_expr=["left_ankle_pitch_joint", "right_ankle_pitch_joint"],
        stiffness=130.0,
        damping=3.5,
        effort_limit_sim=80.0,
        velocity_limit_sim=20.0,
        armature=0.0549,
    ),
    "ankle_roll": ImplicitActuatorCfg(
        joint_names_expr=["left_ankle_roll_joint", "right_ankle_roll_joint"],
        stiffness=70.0,
        damping=2.0,
        effort_limit_sim=40.0,
        velocity_limit_sim=20.0,
        armature=0.0549,
    ),
    "waist": ImplicitActuatorCfg(
        joint_names_expr=[
            "waist_roll_joint",
            "waist_pitch_joint",
            "waist_yaw_joint",
        ],
        stiffness=400.0,
        damping=11.0,
        effort_limit_sim=150.0,
        velocity_limit_sim=8.0,
        armature=0.23409,
    ),
    "shoulder": ImplicitActuatorCfg(
        joint_names_expr=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
        ],
        stiffness=60.0,
        damping=3.0,
        effort_limit_sim=65.0,
        velocity_limit_sim=8.0,
        armature=0.01,
    ),
    "elbow": ImplicitActuatorCfg(
        joint_names_expr=["left_elbow_joint", "right_elbow_joint"],
        stiffness=60.0,
        damping=3.0,
        effort_limit_sim=30.0,
        velocity_limit_sim=8.0,
        armature=0.01,
    ),
}

ADAM_SP_23DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(ADAM_SP_23DOF_URDF),
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            "left_hip_pitch_joint": -0.32,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": -0.18,
            "left_knee_joint": 0.66,
            "left_ankle_pitch_joint": -0.39,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.32,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.18,
            "right_knee_joint": 0.66,
            "right_ankle_pitch_joint": -0.39,
            "right_ankle_roll_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.1,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -0.3,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.1,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.3,
        },
        joint_vel={
            "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.0,
            "left_ankle_pitch_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.0,
            "right_ankle_pitch_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=ADAM_SP_23DOF_ACTUATORS,
)

__all__ = ["ADAM_SP_23DOF_ACTUATORS", "ADAM_SP_23DOF_CFG", "ADAM_SP_23DOF_URDF"]
