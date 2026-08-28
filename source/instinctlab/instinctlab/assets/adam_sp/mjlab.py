"""Explicit mjlab configurations for the PND Adam SP robot."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

RESOURCE_ROOT = Path(__file__).resolve().parent.parent / "resources" / "adam_sp"
ADAM_SP_23DOF_XML = RESOURCE_ROOT / "xml" / "adam_sp_23_dof.xml"
ADAM_SP_29DOF_XML = RESOURCE_ROOT / "xml" / "adam_sp.xml"


def get_adam_sp_23dof_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(ADAM_SP_23DOF_XML))


def get_adam_sp_29dof_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(ADAM_SP_29DOF_XML))


ADAM_SP_23DOF_ACTUATORS = (
    # Hip pitch and knee.
    BuiltinPdActuatorCfg(
        target_names_expr=(
            "left_hip_pitch_joint",
            "left_knee_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
        ),
        stiffness=300.0,
        damping=7.0,
        effort_limit=230.0,
        armature=0.13426,
    ),
    # Hip roll.
    BuiltinPdActuatorCfg(
        target_names_expr=("left_hip_roll_joint", "right_hip_roll_joint"),
        stiffness=600.0,
        damping=10.0,
        effort_limit=180.0,
        armature=0.281573,
    ),
    # Hip yaw.
    BuiltinPdActuatorCfg(
        target_names_expr=("left_hip_yaw_joint", "right_hip_yaw_joint"),
        stiffness=300.0,
        damping=2.0,
        effort_limit=105.0,
        armature=0.23409,
    ),
    # Ankle pitch.
    BuiltinPdActuatorCfg(
        target_names_expr=("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
        stiffness=130.0,
        damping=3.5,
        effort_limit=80.0,
        armature=0.0549,
    ),
    # Ankle roll.
    BuiltinPdActuatorCfg(
        target_names_expr=("left_ankle_roll_joint", "right_ankle_roll_joint"),
        stiffness=70.0,
        damping=2.0,
        effort_limit=40.0,
        armature=0.0549,
    ),
    # Waist.
    BuiltinPdActuatorCfg(
        target_names_expr=(
            "waist_roll_joint",
            "waist_pitch_joint",
            "waist_yaw_joint",
        ),
        stiffness=400.0,
        damping=11.0,
        effort_limit=150.0,
        armature=0.23409,
    ),
    # Shoulders.
    BuiltinPdActuatorCfg(
        target_names_expr=(
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
        ),
        stiffness=60.0,
        damping=3.0,
        effort_limit=65.0,
        armature=0.01,
    ),
    # Elbows.
    BuiltinPdActuatorCfg(
        target_names_expr=("left_elbow_joint", "right_elbow_joint"),
        stiffness=60.0,
        damping=3.0,
        effort_limit=30.0,
        armature=0.01,
    ),
)

ADAM_SP_23DOF_INIT_STATE = EntityCfg.InitialStateCfg(
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
)

ADAM_SP_23DOF_ARTICULATION = EntityArticulationInfoCfg(
    actuators=ADAM_SP_23DOF_ACTUATORS,
    soft_joint_pos_limit_factor=0.9,
)

ADAM_SP_23DOF_CFG = EntityCfg(
    init_state=ADAM_SP_23DOF_INIT_STATE,
    spec_fn=get_adam_sp_23dof_spec,
    articulation=ADAM_SP_23DOF_ARTICULATION,
)


ADAM_SP_29DOF_ACTUATORS = (
    # Hip pitch and knee.
    BuiltinPdActuatorCfg(
        target_names_expr=(
            "left_hip_pitch_joint",
            "left_knee_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
        ),
        stiffness=300.0,
        damping=7.0,
        effort_limit=230.0,
        armature=0.13426,
    ),
    # Hip roll.
    BuiltinPdActuatorCfg(
        target_names_expr=("left_hip_roll_joint", "right_hip_roll_joint"),
        stiffness=600.0,
        damping=10.0,
        effort_limit=180.0,
        armature=0.281573,
    ),
    # Hip yaw.
    BuiltinPdActuatorCfg(
        target_names_expr=("left_hip_yaw_joint", "right_hip_yaw_joint"),
        stiffness=300.0,
        damping=2.0,
        effort_limit=105.0,
        armature=0.23409,
    ),
    # Ankle pitch.
    BuiltinPdActuatorCfg(
        target_names_expr=("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
        stiffness=130.0,
        damping=3.5,
        effort_limit=80.0,
        armature=0.1098,
    ),
    # Ankle roll.
    BuiltinPdActuatorCfg(
        target_names_expr=("left_ankle_roll_joint", "right_ankle_roll_joint"),
        stiffness=70.0,
        damping=2.0,
        effort_limit=40.0,
        armature=0.1098,
    ),
    # Waist.
    BuiltinPdActuatorCfg(
        target_names_expr=(
            "waist_roll_joint",
            "waist_pitch_joint",
            "waist_yaw_joint",
        ),
        stiffness=400.0,
        damping=11.0,
        effort_limit=150.0,
        armature=0.23409,
    ),
    # Shoulders.
    BuiltinPdActuatorCfg(
        target_names_expr=(
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
        ),
        stiffness=60.0,
        damping=3.0,
        effort_limit=65.0,
        armature=0.01,
    ),
    # Elbows.
    BuiltinPdActuatorCfg(
        target_names_expr=("left_elbow_joint", "right_elbow_joint"),
        stiffness=60.0,
        damping=3.0,
        effort_limit=30.0,
        armature=0.01,
    ),
    # Wrists.
    BuiltinPdActuatorCfg(
        target_names_expr=(
            "left_wrist_yaw_joint",
            "left_wrist_pitch_joint",
            "left_wrist_roll_joint",
            "right_wrist_yaw_joint",
            "right_wrist_pitch_joint",
            "right_wrist_roll_joint",
        ),
        stiffness=20.0,
        damping=1.0,
        effort_limit=6.4,
        armature=0.01,
    ),
)

ADAM_SP_29DOF_INIT_STATE = EntityCfg.InitialStateCfg(
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
        "left_wrist_yaw_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": -0.1,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": -0.3,
        "right_wrist_yaw_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
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
        "left_wrist_yaw_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
    },
)

ADAM_SP_29DOF_ARTICULATION = EntityArticulationInfoCfg(
    actuators=ADAM_SP_29DOF_ACTUATORS,
    soft_joint_pos_limit_factor=0.9,
)

ADAM_SP_29DOF_CFG = EntityCfg(
    init_state=ADAM_SP_29DOF_INIT_STATE,
    spec_fn=get_adam_sp_29dof_spec,
    articulation=ADAM_SP_29DOF_ARTICULATION,
)


__all__ = [
    "ADAM_SP_23DOF_ACTUATORS",
    "ADAM_SP_23DOF_CFG",
    "ADAM_SP_23DOF_XML",
    "ADAM_SP_29DOF_ACTUATORS",
    "ADAM_SP_29DOF_CFG",
    "ADAM_SP_29DOF_XML",
    "get_adam_sp_23dof_spec",
    "get_adam_sp_29dof_spec",
]
