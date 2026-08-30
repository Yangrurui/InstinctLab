"""Complete InstinctMJ-style MJLab configuration for Unitree G1.

MJCF paths, variants, canonical metadata, and actuator groups are declared
here in MJLab-owned types. The MJLab engine adapter alone converts these values
to the shared runtime ``RobotSpec``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from instinctlab_engine.name_order import resolve_name_indices

RESOURCE_ROOT = Path(__file__).resolve().parent.parent / "resources" / "unitree_g1"
INSTINCTLAB_NATIVE_ASSET_API = "0.1"


@dataclass(frozen=True)
class MjlabJointCfg:
    name: str
    default_pos: float
    stiffness: float
    damping: float
    armature: float
    effort_limit: float
    velocity_limit: float
    action_scale: float


@dataclass(frozen=True)
class MjlabRobotCfg:
    name: str
    schema_version: str
    asset_id: str
    root_body: str
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    frame_names: tuple[str, ...]
    collision_body_names: tuple[str, ...]
    joint_properties: tuple[MjlabJointCfg, ...]
    mjcf_path: str
    contact_body_aliases: dict[str, str]
    load_mode: str
    default_root_pos: tuple[float, float, float]
    default_root_quat_wxyz: tuple[float, float, float, float]
    soft_joint_pos_limit_factor: float
    actuator_delay: tuple[int, int]
    actuator_model_ids: tuple[str, ...]
    actuator_group_count: int
    length_unit: str
    angle_unit: str
    effort_unit: str

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.01017752
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
STIFFNESS_5020 = 14.25062309787429
STIFFNESS_7520_14 = 40.17923847137318
STIFFNESS_7520_22 = 99.09842777666113
STIFFNESS_4010 = 16.77832748089279
DAMPING_5020 = 0.907222843292423
DAMPING_7520_14 = 2.5578897650279457
DAMPING_7520_22 = 6.3088018534966395
DAMPING_4010 = 1.06814150219

G1_29DOF_DFS_JOINT_NAMES = (
    "waist_pitch_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
G1_29DOF_DFS_BODY_NAMES = (
    "torso_link",
    "waist_roll_link",
    "waist_yaw_link",
    "pelvis",
    "pelvis_contour_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "LL_FOOT",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "LR_FOOT",
    "imu_in_pelvis",
    "logo_link",
    "head_link",
    "imu_in_torso",
    "mid360_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
)
G1_29DOF_DFS_FRAME_NAMES = (
    "pelvis_contour_link",
    "LL_FOOT",
    "LR_FOOT",
    "imu_in_pelvis",
    "logo_link",
    "imu_in_torso",
    "mid360_link",
)
G1_29DOF_DFS_COLLISION_BODY_NAMES = (
    "torso_link",
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
G1_29DOF_DEFAULT_JOINT_POS = {
    "waist_pitch_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "left_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.312,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "right_ankle_roll_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.6,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

G1_29DOF_JOINT_PROPERTIES = (
    MjlabJointCfg(
        name="waist_pitch_joint",
        default_pos=0.0,
        stiffness=28.50124619574858,
        damping=1.814445686584846,
        armature=0.00721945,
        effort_limit=50.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    MjlabJointCfg(
        name="waist_roll_joint",
        default_pos=0.0,
        stiffness=28.50124619574858,
        damping=1.814445686584846,
        armature=0.00721945,
        effort_limit=50.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    MjlabJointCfg(
        name="waist_yaw_joint",
        default_pos=0.0,
        stiffness=40.17923847137318,
        damping=2.5578897650279457,
        armature=0.01017752,
        effort_limit=88.0,
        velocity_limit=32.0,
        action_scale=0.5475464652142303,
    ),
    MjlabJointCfg(
        name="left_hip_pitch_joint",
        default_pos=-0.312,
        stiffness=40.17923847137318,
        damping=2.5578897650279457,
        armature=0.01017752,
        effort_limit=88.0,
        velocity_limit=32.0,
        action_scale=0.5475464652142303,
    ),
    MjlabJointCfg(
        name="left_hip_roll_joint",
        default_pos=0.0,
        stiffness=99.09842777666113,
        damping=6.3088018534966395,
        armature=0.025101925,
        effort_limit=139.0,
        velocity_limit=20.0,
        action_scale=0.3506614663788243,
    ),
    MjlabJointCfg(
        name="left_hip_yaw_joint",
        default_pos=0.0,
        stiffness=40.17923847137318,
        damping=2.5578897650279457,
        armature=0.01017752,
        effort_limit=88.0,
        velocity_limit=32.0,
        action_scale=0.5475464652142303,
    ),
    MjlabJointCfg(
        name="left_knee_joint",
        default_pos=0.669,
        stiffness=99.09842777666113,
        damping=6.3088018534966395,
        armature=0.025101925,
        effort_limit=139.0,
        velocity_limit=20.0,
        action_scale=0.3506614663788243,
    ),
    MjlabJointCfg(
        name="left_ankle_pitch_joint",
        default_pos=-0.363,
        stiffness=28.50124619574858,
        damping=1.814445686584846,
        armature=0.00721945,
        effort_limit=50.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    MjlabJointCfg(
        name="left_ankle_roll_joint",
        default_pos=0.0,
        stiffness=28.50124619574858,
        damping=1.814445686584846,
        armature=0.00721945,
        effort_limit=50.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    MjlabJointCfg(
        name="right_hip_pitch_joint",
        default_pos=-0.312,
        stiffness=40.17923847137318,
        damping=2.5578897650279457,
        armature=0.01017752,
        effort_limit=88.0,
        velocity_limit=32.0,
        action_scale=0.5475464652142303,
    ),
    MjlabJointCfg(
        name="right_hip_roll_joint",
        default_pos=0.0,
        stiffness=99.09842777666113,
        damping=6.3088018534966395,
        armature=0.025101925,
        effort_limit=139.0,
        velocity_limit=20.0,
        action_scale=0.3506614663788243,
    ),
    MjlabJointCfg(
        name="right_hip_yaw_joint",
        default_pos=0.0,
        stiffness=40.17923847137318,
        damping=2.5578897650279457,
        armature=0.01017752,
        effort_limit=88.0,
        velocity_limit=32.0,
        action_scale=0.5475464652142303,
    ),
    MjlabJointCfg(
        name="right_knee_joint",
        default_pos=0.669,
        stiffness=99.09842777666113,
        damping=6.3088018534966395,
        armature=0.025101925,
        effort_limit=139.0,
        velocity_limit=20.0,
        action_scale=0.3506614663788243,
    ),
    MjlabJointCfg(
        name="right_ankle_pitch_joint",
        default_pos=-0.363,
        stiffness=28.50124619574858,
        damping=1.814445686584846,
        armature=0.00721945,
        effort_limit=50.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    MjlabJointCfg(
        name="right_ankle_roll_joint",
        default_pos=0.0,
        stiffness=28.50124619574858,
        damping=1.814445686584846,
        armature=0.00721945,
        effort_limit=50.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    MjlabJointCfg(
        name="left_shoulder_pitch_joint",
        default_pos=0.2,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    MjlabJointCfg(
        name="left_shoulder_roll_joint",
        default_pos=0.2,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    MjlabJointCfg(
        name="left_shoulder_yaw_joint",
        default_pos=0.0,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    MjlabJointCfg(
        name="left_elbow_joint",
        default_pos=0.6,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    MjlabJointCfg(
        name="left_wrist_roll_joint",
        default_pos=0.0,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    MjlabJointCfg(
        name="left_wrist_pitch_joint",
        default_pos=0.0,
        stiffness=16.77832748089279,
        damping=1.06814150219,
        armature=0.00425,
        effort_limit=5.0,
        velocity_limit=22.0,
        action_scale=0.07450087032950714,
    ),
    MjlabJointCfg(
        name="left_wrist_yaw_joint",
        default_pos=0.0,
        stiffness=16.77832748089279,
        damping=1.06814150219,
        armature=0.00425,
        effort_limit=5.0,
        velocity_limit=22.0,
        action_scale=0.07450087032950714,
    ),
    MjlabJointCfg(
        name="right_shoulder_pitch_joint",
        default_pos=0.2,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    MjlabJointCfg(
        name="right_shoulder_roll_joint",
        default_pos=-0.2,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    MjlabJointCfg(
        name="right_shoulder_yaw_joint",
        default_pos=0.0,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    MjlabJointCfg(
        name="right_elbow_joint",
        default_pos=0.6,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    MjlabJointCfg(
        name="right_wrist_roll_joint",
        default_pos=0.0,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    MjlabJointCfg(
        name="right_wrist_pitch_joint",
        default_pos=0.0,
        stiffness=16.77832748089279,
        damping=1.06814150219,
        armature=0.00425,
        effort_limit=5.0,
        velocity_limit=22.0,
        action_scale=0.07450087032950714,
    ),
    MjlabJointCfg(
        name="right_wrist_yaw_joint",
        default_pos=0.0,
        stiffness=16.77832748089279,
        damping=1.06814150219,
        armature=0.00425,
        effort_limit=5.0,
        velocity_limit=22.0,
        action_scale=0.07450087032950714,
    ),
)

_G1_MIRRORED_JOINT_NAMES = {
    "waist_pitch_joint": "waist_pitch_joint",
    "waist_roll_joint": "waist_roll_joint",
    "waist_yaw_joint": "waist_yaw_joint",
    "left_hip_pitch_joint": "right_hip_pitch_joint",
    "left_hip_roll_joint": "right_hip_roll_joint",
    "left_hip_yaw_joint": "right_hip_yaw_joint",
    "left_knee_joint": "right_knee_joint",
    "left_ankle_pitch_joint": "right_ankle_pitch_joint",
    "left_ankle_roll_joint": "right_ankle_roll_joint",
    "right_hip_pitch_joint": "left_hip_pitch_joint",
    "right_hip_roll_joint": "left_hip_roll_joint",
    "right_hip_yaw_joint": "left_hip_yaw_joint",
    "right_knee_joint": "left_knee_joint",
    "right_ankle_pitch_joint": "left_ankle_pitch_joint",
    "right_ankle_roll_joint": "left_ankle_roll_joint",
    "left_shoulder_pitch_joint": "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint": "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint": "right_shoulder_yaw_joint",
    "left_elbow_joint": "right_elbow_joint",
    "left_wrist_roll_joint": "right_wrist_roll_joint",
    "left_wrist_pitch_joint": "right_wrist_pitch_joint",
    "left_wrist_yaw_joint": "right_wrist_yaw_joint",
    "right_shoulder_pitch_joint": "left_shoulder_pitch_joint",
    "right_shoulder_roll_joint": "left_shoulder_roll_joint",
    "right_shoulder_yaw_joint": "left_shoulder_yaw_joint",
    "right_elbow_joint": "left_elbow_joint",
    "right_wrist_roll_joint": "left_wrist_roll_joint",
    "right_wrist_pitch_joint": "left_wrist_pitch_joint",
    "right_wrist_yaw_joint": "left_wrist_yaw_joint",
}
_G1_SYMMETRY_SIGNS = {
    "waist_pitch_joint": 1,
    "waist_roll_joint": -1,
    "waist_yaw_joint": -1,
    "left_hip_pitch_joint": 1,
    "left_hip_roll_joint": -1,
    "left_hip_yaw_joint": -1,
    "left_knee_joint": 1,
    "left_ankle_pitch_joint": 1,
    "left_ankle_roll_joint": -1,
    "right_hip_pitch_joint": 1,
    "right_hip_roll_joint": -1,
    "right_hip_yaw_joint": -1,
    "right_knee_joint": 1,
    "right_ankle_pitch_joint": 1,
    "right_ankle_roll_joint": -1,
    "left_shoulder_pitch_joint": 1,
    "left_shoulder_roll_joint": -1,
    "left_shoulder_yaw_joint": -1,
    "left_elbow_joint": 1,
    "left_wrist_roll_joint": -1,
    "left_wrist_pitch_joint": 1,
    "left_wrist_yaw_joint": -1,
    "right_shoulder_pitch_joint": 1,
    "right_shoulder_roll_joint": -1,
    "right_shoulder_yaw_joint": -1,
    "right_elbow_joint": 1,
    "right_wrist_roll_joint": -1,
    "right_wrist_pitch_joint": 1,
    "right_wrist_yaw_joint": -1,
}


def g1_symmetric_joint_augmentation(
    joint_names: Sequence[str] = G1_29DOF_DFS_JOINT_NAMES,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the explicit left/right mapping and sagittal sign for one joint order."""
    names = tuple(joint_names)
    mirrored_names = tuple(_G1_MIRRORED_JOINT_NAMES[name] for name in names)
    mapping = resolve_name_indices(names, mirrored_names, require_exact=True)
    reverse = tuple(_G1_SYMMETRY_SIGNS[name] for name in names)
    return mapping, reverse


G1_29DOF_CFG = MjlabRobotCfg(
    name="unitree_g1_29dof",
    schema_version="dfs_v1",
    asset_id="unitree_g1/popsicle_torsobase_v1",
    root_body="torso_link",
    joint_names=G1_29DOF_DFS_JOINT_NAMES,
    body_names=G1_29DOF_DFS_BODY_NAMES,
    frame_names=G1_29DOF_DFS_FRAME_NAMES,
    collision_body_names=G1_29DOF_DFS_COLLISION_BODY_NAMES,
    joint_properties=G1_29DOF_JOINT_PROPERTIES,
    mjcf_path=str(RESOURCE_ROOT / "xml" / "g1_29dof_torsobase_popsicle.xml"),
    contact_body_aliases={
        "LL_FOOT": "left_ankle_roll_link",
        "LR_FOOT": "right_ankle_roll_link",
    },
    load_mode="strip_visual_meshes",
    default_root_pos=(0.0, 0.0, 0.82),
    default_root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    soft_joint_pos_limit_factor=0.9,
    actuator_delay=(0, 0),
    actuator_model_ids=("mjlab.builtin_pd.v1",),
    actuator_group_count=7,
    length_unit="m",
    angle_unit="rad",
    effort_unit="N*m",
)


G1_29DOF_SHADOWING_CFG = MjlabRobotCfg(
    name="unitree_g1_29dof",
    schema_version="dfs_v1",
    asset_id="unitree_g1/popsicle_torsobase_shadowing_v1",
    root_body="torso_link",
    joint_names=G1_29DOF_DFS_JOINT_NAMES,
    body_names=G1_29DOF_DFS_BODY_NAMES,
    frame_names=G1_29DOF_DFS_FRAME_NAMES,
    collision_body_names=G1_29DOF_DFS_COLLISION_BODY_NAMES,
    joint_properties=G1_29DOF_JOINT_PROPERTIES,
    mjcf_path=str(RESOURCE_ROOT / "xml" / "g1_29dof_torsobase_popsicle.xml"),
    contact_body_aliases={
        "LL_FOOT": "left_ankle_roll_link",
        "LR_FOOT": "right_ankle_roll_link",
    },
    load_mode="strip_visual_meshes",
    default_root_pos=(0.0, 0.0, 0.82),
    default_root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    soft_joint_pos_limit_factor=0.9,
    actuator_delay=(0, 0),
    actuator_model_ids=("mjlab.builtin_pd.v1",),
    actuator_group_count=7,
    length_unit="m",
    angle_unit="rad",
    effort_unit="N*m",
)


G1_29DOF_PARKOUR_CFG = MjlabRobotCfg(
    name="unitree_g1_29dof",
    schema_version="dfs_v1",
    asset_id="unitree_g1/popsicle_torsobase_parkour_v1",
    root_body="torso_link",
    joint_names=G1_29DOF_DFS_JOINT_NAMES,
    body_names=G1_29DOF_DFS_BODY_NAMES,
    frame_names=G1_29DOF_DFS_FRAME_NAMES,
    collision_body_names=G1_29DOF_DFS_COLLISION_BODY_NAMES,
    joint_properties=G1_29DOF_JOINT_PROPERTIES,
    mjcf_path=str(
        RESOURCE_ROOT / "xml" / "g1_29dof_torsoBase_popsicle_with_shoe.xml"
    ),
    contact_body_aliases={
        "LL_FOOT": "left_ankle_roll_link",
        "LR_FOOT": "right_ankle_roll_link",
    },
    load_mode="strip_visual_meshes",
    default_root_pos=(0.0, 0.0, 0.9),
    default_root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    soft_joint_pos_limit_factor=0.9,
    actuator_delay=(0, 2),
    actuator_model_ids=("mjlab.builtin_pd.v1",),
    actuator_group_count=7,
    length_unit="m",
    angle_unit="rad",
    effort_unit="N*m",
)


NATIVE_CONFIGS = {
    "popsicle_torsobase_v1": G1_29DOF_CFG,
    "popsicle_torsobase_shadowing_v1": G1_29DOF_SHADOWING_CFG,
    "popsicle_torsobase_parkour_v1": G1_29DOF_PARKOUR_CFG,
}


def native_config(variant: str) -> MjlabRobotCfg:
    """Return the complete MJLab-native configuration for ``variant``."""
    try:
        return NATIVE_CONFIGS[variant]
    except KeyError:
        raise KeyError(
            f"Unknown MJLab Unitree G1 variant {variant!r}; registered: {sorted(NATIVE_CONFIGS)}"
        ) from None


G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping = [
    0,
    1,
    2,
    9,
    10,
    11,
    12,
    13,
    14,
    3,
    4,
    5,
    6,
    7,
    8,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
]
G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf = [
    1,
    -1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
]
beyondmimic_action_scale = {
    "waist_pitch_joint": 0.43857731392336724,
    "waist_roll_joint": 0.43857731392336724,
    "waist_yaw_joint": 0.5475464652142303,
    "left_hip_pitch_joint": 0.5475464652142303,
    "left_hip_roll_joint": 0.3506614663788243,
    "left_hip_yaw_joint": 0.5475464652142303,
    "left_knee_joint": 0.3506614663788243,
    "left_ankle_pitch_joint": 0.43857731392336724,
    "left_ankle_roll_joint": 0.43857731392336724,
    "right_hip_pitch_joint": 0.5475464652142303,
    "right_hip_roll_joint": 0.3506614663788243,
    "right_hip_yaw_joint": 0.5475464652142303,
    "right_knee_joint": 0.3506614663788243,
    "right_ankle_pitch_joint": 0.43857731392336724,
    "right_ankle_roll_joint": 0.43857731392336724,
    "left_shoulder_pitch_joint": 0.43857731392336724,
    "left_shoulder_roll_joint": 0.43857731392336724,
    "left_shoulder_yaw_joint": 0.43857731392336724,
    "left_elbow_joint": 0.43857731392336724,
    "left_wrist_roll_joint": 0.43857731392336724,
    "left_wrist_pitch_joint": 0.07450087032950714,
    "left_wrist_yaw_joint": 0.07450087032950714,
    "right_shoulder_pitch_joint": 0.43857731392336724,
    "right_shoulder_roll_joint": 0.43857731392336724,
    "right_shoulder_yaw_joint": 0.43857731392336724,
    "right_elbow_joint": 0.43857731392336724,
    "right_wrist_roll_joint": 0.43857731392336724,
    "right_wrist_pitch_joint": 0.07450087032950714,
    "right_wrist_yaw_joint": 0.07450087032950714,
}

G1_29DOF_LINKS = [
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "waist_yaw_link",
    "waist_roll_link",
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
]

ENTITIES = frozenset(
    {
        "popsicle_torsobase_v1",
        "popsicle_torsobase_shadowing_v1",
        "popsicle_torsobase_parkour_v1",
    }
)


def beyondmimic_actuator_cfgs(
    actuator_cfg_type,
) -> tuple[object, ...]:
    """Build MJLab's seven explicit non-delayed native groups."""
    return (
        actuator_cfg_type(
            target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint"),
            effort_limit=88.0,
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        actuator_cfg_type(
            target_names_expr=("waist_yaw_joint",),
            effort_limit=88.0,
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        actuator_cfg_type(
            target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
            effort_limit=139.0,
            stiffness=STIFFNESS_7520_22,
            damping=DAMPING_7520_22,
            armature=ARMATURE_7520_22,
        ),
        actuator_cfg_type(
            target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
            effort_limit=50.0,
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        actuator_cfg_type(
            target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
            effort_limit=50.0,
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        actuator_cfg_type(
            target_names_expr=(
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ),
            effort_limit=25.0,
            stiffness=STIFFNESS_5020,
            damping=DAMPING_5020,
            armature=ARMATURE_5020,
        ),
        actuator_cfg_type(
            target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
            effort_limit=5.0,
            stiffness=STIFFNESS_4010,
            damping=DAMPING_4010,
            armature=ARMATURE_4010,
        ),
    )


def beyondmimic_delayed_actuator_cfgs(
    actuator_cfg_type,
) -> tuple[object, ...]:
    """Build MJLab's seven explicit Parkour actuator groups."""
    return (
        actuator_cfg_type(
            target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint"),
            effort_limit=88.0,
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
            delay_min_lag=0,
            delay_max_lag=2,
            delay_update_period=1_000_000,
            delay_per_env_phase=False,
        ),
        actuator_cfg_type(
            target_names_expr=("waist_yaw_joint",),
            effort_limit=88.0,
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
            delay_min_lag=0,
            delay_max_lag=2,
            delay_update_period=1_000_003,
            delay_per_env_phase=False,
        ),
        actuator_cfg_type(
            target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
            effort_limit=139.0,
            stiffness=STIFFNESS_7520_22,
            damping=DAMPING_7520_22,
            armature=ARMATURE_7520_22,
            delay_min_lag=0,
            delay_max_lag=2,
            delay_update_period=1_000_000,
            delay_per_env_phase=False,
        ),
        actuator_cfg_type(
            target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
            effort_limit=50.0,
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
            delay_min_lag=0,
            delay_max_lag=2,
            delay_update_period=1_000_001,
            delay_per_env_phase=False,
        ),
        actuator_cfg_type(
            target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
            effort_limit=50.0,
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
            delay_min_lag=0,
            delay_max_lag=2,
            delay_update_period=1_000_002,
            delay_per_env_phase=False,
        ),
        actuator_cfg_type(
            target_names_expr=(
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ),
            effort_limit=25.0,
            stiffness=STIFFNESS_5020,
            damping=DAMPING_5020,
            armature=ARMATURE_5020,
            delay_min_lag=0,
            delay_max_lag=2,
            delay_update_period=1_000_004,
            delay_per_env_phase=False,
        ),
        actuator_cfg_type(
            target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
            effort_limit=5.0,
            stiffness=STIFFNESS_4010,
            damping=DAMPING_4010,
            armature=ARMATURE_4010,
            delay_min_lag=0,
            delay_max_lag=2,
            delay_update_period=1_000_004,
            delay_per_env_phase=False,
        ),
    )


ACTUATOR_CONFIGS = {
    "popsicle_torsobase_v1": beyondmimic_actuator_cfgs,
    "popsicle_torsobase_shadowing_v1": beyondmimic_actuator_cfgs,
    "popsicle_torsobase_parkour_v1": beyondmimic_delayed_actuator_cfgs,
}


def _without_visual_meshes(xml: str) -> str:
    root = ElementTree.fromstring(xml)
    for asset in root.findall("asset"):
        for mesh in tuple(asset.findall("mesh")):
            asset.remove(mesh)
    for parent in root.iter():
        for geom in tuple(parent.findall("geom")):
            if geom.get("type") == "mesh" or geom.get("mesh"):
                parent.remove(geom)
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    return ElementTree.tostring(root, encoding="unicode")


def _load_spec(path: Path, load_mode: str) -> Any:
    import mujoco

    if load_mode == "default":
        return mujoco.MjSpec.from_file(str(path))
    if load_mode == "strip_visual_meshes":
        try:
            return mujoco.MjSpec.from_file(str(path))
        except (ValueError, OSError):
            return mujoco.MjSpec.from_string(_without_visual_meshes(path.read_text()))
    raise NotImplementedError(f"Unitree G1 has no MJLab loader for {load_mode!r}")


def entity(variant: str, robot: Any, *, actuator_order=None) -> Any:
    """Build one registered G1 variant as an MJLab entity."""
    del actuator_order
    if variant not in ENTITIES:
        raise KeyError(
            f"Unknown MJLab Unitree G1 variant {variant!r}; registered: {sorted(ENTITIES)}"
        )

    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from instinctlab_engine.actuators import native_actuator_factory

    BuiltinPdActuatorCfg = native_actuator_factory(
        "mjlab", "mjlab.builtin_pd.v1"
    )

    asset = robot.asset_for("mjlab")
    path = Path(asset.path)
    if not path.is_file():
        raise FileNotFoundError(
            f"The MJLab asset for {robot.name!r} is missing: {path}"
        )
    actuator_groups = ACTUATOR_CONFIGS[variant](BuiltinPdActuatorCfg)
    from instinctlab_engine.assets import validate_native_actuator_groups

    validate_native_actuator_groups(
        robot.asset_id,
        actuator_groups,
        robot.joint_names,
        selector_field="target_names_expr",
        expected_group_count=NATIVE_CONFIGS[variant].actuator_group_count,
    )
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=robot.default_root_pos,
            rot=robot.default_root_quat_wxyz,
            joint_pos={
                joint.name: joint.default_pos for joint in robot.joint_properties
            },
            joint_vel={".*": 0.0},
        ),
        spec_fn=lambda: _load_spec(path, asset.load_mode),
        articulation=EntityArticulationInfoCfg(
            actuators=actuator_groups,
            soft_joint_pos_limit_factor=robot.soft_joint_pos_limit_factor,
        ),
    )


__all__ = [
    "ACTUATOR_CONFIGS",
    "ENTITIES",
    "NATIVE_CONFIGS",
    "beyondmimic_actuator_cfgs",
    "beyondmimic_delayed_actuator_cfgs",
    "entity",
    "native_config",
]
