"""Complete main-style Isaac Lab configuration for Unitree G1.

Isaac model paths, variants, canonical metadata, and actuator groups are
declared here in Isaac-owned types. Isaac Lab itself remains lazy so resolving
the configuration does not start Kit. The Isaac engine adapter alone converts
these values to the shared runtime ``RobotSpec``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from instinctlab_engine.name_order import resolve_name_indices

RESOURCE_ROOT = Path(__file__).resolve().parent.parent / "resources" / "unitree_g1"


@dataclass(frozen=True)
class IsaacJointCfg:
    name: str
    default_pos: float
    stiffness: float
    damping: float
    armature: float
    effort_limit: float
    velocity_limit: float
    action_scale: float


@dataclass(frozen=True)
class IsaacRobotCfg:
    name: str
    schema_version: str
    asset_id: str
    root_body: str
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    frame_names: tuple[str, ...]
    collision_body_names: tuple[str, ...]
    joint_properties: tuple[IsaacJointCfg, ...]
    urdf_path: str
    contact_body_aliases: dict[str, str]
    import_options: dict[str, object]
    default_root_pos: tuple[float, float, float]
    default_root_quat_wxyz: tuple[float, float, float, float]
    soft_joint_pos_limit_factor: float
    actuator_delay: tuple[int, int]

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
G1_29DOF_ISAAC_BFS_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "waist_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
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
    IsaacJointCfg(
        name="waist_pitch_joint",
        default_pos=0.0,
        stiffness=28.50124619574858,
        damping=1.814445686584846,
        armature=0.00721945,
        effort_limit=50.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    IsaacJointCfg(
        name="waist_roll_joint",
        default_pos=0.0,
        stiffness=28.50124619574858,
        damping=1.814445686584846,
        armature=0.00721945,
        effort_limit=50.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    IsaacJointCfg(
        name="waist_yaw_joint",
        default_pos=0.0,
        stiffness=40.17923847137318,
        damping=2.5578897650279457,
        armature=0.01017752,
        effort_limit=88.0,
        velocity_limit=32.0,
        action_scale=0.5475464652142303,
    ),
    IsaacJointCfg(
        name="left_hip_pitch_joint",
        default_pos=-0.312,
        stiffness=40.17923847137318,
        damping=2.5578897650279457,
        armature=0.01017752,
        effort_limit=88.0,
        velocity_limit=32.0,
        action_scale=0.5475464652142303,
    ),
    IsaacJointCfg(
        name="left_hip_roll_joint",
        default_pos=0.0,
        stiffness=99.09842777666113,
        damping=6.3088018534966395,
        armature=0.025101925,
        effort_limit=139.0,
        velocity_limit=20.0,
        action_scale=0.3506614663788243,
    ),
    IsaacJointCfg(
        name="left_hip_yaw_joint",
        default_pos=0.0,
        stiffness=40.17923847137318,
        damping=2.5578897650279457,
        armature=0.01017752,
        effort_limit=88.0,
        velocity_limit=32.0,
        action_scale=0.5475464652142303,
    ),
    IsaacJointCfg(
        name="left_knee_joint",
        default_pos=0.669,
        stiffness=99.09842777666113,
        damping=6.3088018534966395,
        armature=0.025101925,
        effort_limit=139.0,
        velocity_limit=20.0,
        action_scale=0.3506614663788243,
    ),
    IsaacJointCfg(
        name="left_ankle_pitch_joint",
        default_pos=-0.363,
        stiffness=28.50124619574858,
        damping=1.814445686584846,
        armature=0.00721945,
        effort_limit=50.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    IsaacJointCfg(
        name="left_ankle_roll_joint",
        default_pos=0.0,
        stiffness=28.50124619574858,
        damping=1.814445686584846,
        armature=0.00721945,
        effort_limit=50.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    IsaacJointCfg(
        name="right_hip_pitch_joint",
        default_pos=-0.312,
        stiffness=40.17923847137318,
        damping=2.5578897650279457,
        armature=0.01017752,
        effort_limit=88.0,
        velocity_limit=32.0,
        action_scale=0.5475464652142303,
    ),
    IsaacJointCfg(
        name="right_hip_roll_joint",
        default_pos=0.0,
        stiffness=99.09842777666113,
        damping=6.3088018534966395,
        armature=0.025101925,
        effort_limit=139.0,
        velocity_limit=20.0,
        action_scale=0.3506614663788243,
    ),
    IsaacJointCfg(
        name="right_hip_yaw_joint",
        default_pos=0.0,
        stiffness=40.17923847137318,
        damping=2.5578897650279457,
        armature=0.01017752,
        effort_limit=88.0,
        velocity_limit=32.0,
        action_scale=0.5475464652142303,
    ),
    IsaacJointCfg(
        name="right_knee_joint",
        default_pos=0.669,
        stiffness=99.09842777666113,
        damping=6.3088018534966395,
        armature=0.025101925,
        effort_limit=139.0,
        velocity_limit=20.0,
        action_scale=0.3506614663788243,
    ),
    IsaacJointCfg(
        name="right_ankle_pitch_joint",
        default_pos=-0.363,
        stiffness=28.50124619574858,
        damping=1.814445686584846,
        armature=0.00721945,
        effort_limit=50.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    IsaacJointCfg(
        name="right_ankle_roll_joint",
        default_pos=0.0,
        stiffness=28.50124619574858,
        damping=1.814445686584846,
        armature=0.00721945,
        effort_limit=50.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    IsaacJointCfg(
        name="left_shoulder_pitch_joint",
        default_pos=0.2,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    IsaacJointCfg(
        name="left_shoulder_roll_joint",
        default_pos=0.2,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    IsaacJointCfg(
        name="left_shoulder_yaw_joint",
        default_pos=0.0,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    IsaacJointCfg(
        name="left_elbow_joint",
        default_pos=0.6,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    IsaacJointCfg(
        name="left_wrist_roll_joint",
        default_pos=0.0,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    IsaacJointCfg(
        name="left_wrist_pitch_joint",
        default_pos=0.0,
        stiffness=16.77832748089279,
        damping=1.06814150219,
        armature=0.00425,
        effort_limit=5.0,
        velocity_limit=22.0,
        action_scale=0.07450087032950714,
    ),
    IsaacJointCfg(
        name="left_wrist_yaw_joint",
        default_pos=0.0,
        stiffness=16.77832748089279,
        damping=1.06814150219,
        armature=0.00425,
        effort_limit=5.0,
        velocity_limit=22.0,
        action_scale=0.07450087032950714,
    ),
    IsaacJointCfg(
        name="right_shoulder_pitch_joint",
        default_pos=0.2,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    IsaacJointCfg(
        name="right_shoulder_roll_joint",
        default_pos=-0.2,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    IsaacJointCfg(
        name="right_shoulder_yaw_joint",
        default_pos=0.0,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    IsaacJointCfg(
        name="right_elbow_joint",
        default_pos=0.6,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    IsaacJointCfg(
        name="right_wrist_roll_joint",
        default_pos=0.0,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        armature=0.003609725,
        effort_limit=25.0,
        velocity_limit=37.0,
        action_scale=0.43857731392336724,
    ),
    IsaacJointCfg(
        name="right_wrist_pitch_joint",
        default_pos=0.0,
        stiffness=16.77832748089279,
        damping=1.06814150219,
        armature=0.00425,
        effort_limit=5.0,
        velocity_limit=22.0,
        action_scale=0.07450087032950714,
    ),
    IsaacJointCfg(
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


G1_29DOF_CFG = IsaacRobotCfg(
    name="unitree_g1_29dof",
    schema_version="dfs_v1",
    asset_id="unitree_g1/popsicle_torsobase_v1",
    root_body="torso_link",
    joint_names=G1_29DOF_DFS_JOINT_NAMES,
    body_names=G1_29DOF_DFS_BODY_NAMES,
    frame_names=G1_29DOF_DFS_FRAME_NAMES,
    collision_body_names=G1_29DOF_DFS_COLLISION_BODY_NAMES,
    joint_properties=G1_29DOF_JOINT_PROPERTIES,
    urdf_path=str(RESOURCE_ROOT / "urdf" / "g1_29dof_torsobase_popsicle.urdf"),
    contact_body_aliases={
        "LL_FOOT": "left_ankle_roll_link",
        "LR_FOOT": "right_ankle_roll_link",
    },
    import_options={
        "prim_path": "{ENV_REGEX_NS}/Robot",
        "fix_base": False,
        "merge_fixed_joints": False,
        "replace_cylinders_with_capsules": True,
    },
    default_root_pos=(0.0, 0.0, 0.82),
    default_root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    soft_joint_pos_limit_factor=0.9,
    actuator_delay=(0, 0),
)


G1_29DOF_SHADOWING_CFG = IsaacRobotCfg(
    name="unitree_g1_29dof",
    schema_version="dfs_v1",
    asset_id="unitree_g1/popsicle_torsobase_shadowing_v1",
    root_body="torso_link",
    joint_names=G1_29DOF_DFS_JOINT_NAMES,
    body_names=G1_29DOF_DFS_BODY_NAMES,
    frame_names=G1_29DOF_DFS_FRAME_NAMES,
    collision_body_names=G1_29DOF_DFS_COLLISION_BODY_NAMES,
    joint_properties=G1_29DOF_JOINT_PROPERTIES,
    urdf_path=str(RESOURCE_ROOT / "urdf" / "g1_29dof_torsobase_popsicle.urdf"),
    contact_body_aliases={
        "LL_FOOT": "left_ankle_roll_link",
        "LR_FOOT": "right_ankle_roll_link",
    },
    import_options={
        "prim_path": "{ENV_REGEX_NS}/Robot",
        "fix_base": False,
        "merge_fixed_joints": True,
        "replace_cylinders_with_capsules": True,
    },
    default_root_pos=(0.0, 0.0, 0.82),
    default_root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    soft_joint_pos_limit_factor=0.9,
    actuator_delay=(0, 0),
)


G1_29DOF_PARKOUR_CFG = IsaacRobotCfg(
    name="unitree_g1_29dof",
    schema_version="dfs_v1",
    asset_id="unitree_g1/popsicle_torsobase_parkour_v1",
    root_body="torso_link",
    joint_names=G1_29DOF_DFS_JOINT_NAMES,
    body_names=G1_29DOF_DFS_BODY_NAMES,
    frame_names=G1_29DOF_DFS_FRAME_NAMES,
    collision_body_names=G1_29DOF_DFS_COLLISION_BODY_NAMES,
    joint_properties=G1_29DOF_JOINT_PROPERTIES,
    urdf_path=str(
        RESOURCE_ROOT / "urdf" / "g1_29dof_torsoBase_popsicle_with_shoe.urdf"
    ),
    contact_body_aliases={
        "LL_FOOT": "left_ankle_roll_link",
        "LR_FOOT": "right_ankle_roll_link",
    },
    import_options={
        "prim_path": "{ENV_REGEX_NS}/Robot",
        "fix_base": False,
        "merge_fixed_joints": True,
        "replace_cylinders_with_capsules": True,
    },
    default_root_pos=(0.0, 0.0, 0.9),
    default_root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    soft_joint_pos_limit_factor=0.9,
    actuator_delay=(0, 2),
)


NATIVE_CONFIGS = {
    "popsicle_torsobase_v1": G1_29DOF_CFG,
    "popsicle_torsobase_shadowing_v1": G1_29DOF_SHADOWING_CFG,
    "popsicle_torsobase_parkour_v1": G1_29DOF_PARKOUR_CFG,
}


def native_config(variant: str) -> IsaacRobotCfg:
    """Return the complete Isaac-native configuration for ``variant``."""
    try:
        return NATIVE_CONFIGS[variant]
    except KeyError:
        raise KeyError(
            f"Unknown Isaac Unitree G1 variant {variant!r}; registered: {sorted(NATIVE_CONFIGS)}"
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
G1_29Dof_TorsoBase_isaac_bfs_symmetric_augmentation_joint_mapping = [
    1,
    0,
    2,
    4,
    3,
    5,
    7,
    6,
    8,
    10,
    9,
    12,
    11,
    14,
    13,
    16,
    15,
    18,
    17,
    20,
    19,
    22,
    21,
    24,
    23,
    26,
    25,
    28,
    27,
]
G1_29Dof_TorsoBase_isaac_bfs_symmetric_augmentation_joint_reverse_buf = [
    1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    1,
    1,
    -1,
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


def _load_isaac() -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab_assets import G1_CFG
    from instinctlab_engine.actuators import native_actuator_factory

    DelayedPDActuatorCfg = native_actuator_factory(
        "isaacsim", "isaaclab.delayed_pd.v1"
    )
    ImplicitActuatorCfg = native_actuator_factory(
        "isaacsim", "isaaclab.implicit_pd.v1"
    )

    global _ISAAC_LOADED
    if _ISAAC_LOADED:
        return

    global \
        G1_29DOF_TORSOBASE_CFG, \
        G1_29DOF_TORSOBASE_CLOG_CFG, \
        G1_29DOF_TORSOBASE_POPSICLE_CFG
    global \
        G1_29DOF_TORSOBASE_POPSICLE_SHADOWING_CFG, \
        G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG
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
    G1_29DOF_TORSOBASE_CFG.spawn.joint_drive.gains.stiffness = (
        None  # use value from the URDF file
    )
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
    G1_29DOF_TORSOBASE_CLOG_CFG.spawn.joint_drive.gains.stiffness = (
        None  # use value from the URDF file
    )

    beyondmimic_g1_29dof_actuators = {
        "legs": ImplicitActuatorCfg(
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
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
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
    beyondmimic_g1_29dof_delayed_actuators = {
        "legs": DelayedPDActuatorCfg(
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
            min_delay=0,
            max_delay=2,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
            min_delay=0,
            max_delay=2,
        ),
        "waist": DelayedPDActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
            min_delay=0,
            max_delay=2,
        ),
        "waist_yaw": DelayedPDActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
            min_delay=0,
            max_delay=2,
        ),
        "arms": DelayedPDActuatorCfg(
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
            min_delay=0,
            max_delay=2,
        ),
    }

    robot = G1_29DOF_CFG
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
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0, damping=0
                )
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

    shadowing_robot = G1_29DOF_SHADOWING_CFG
    G1_29DOF_TORSOBASE_POPSICLE_SHADOWING_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG.copy()
    G1_29DOF_TORSOBASE_POPSICLE_SHADOWING_CFG.spawn = (
        G1_29DOF_TORSOBASE_POPSICLE_SHADOWING_CFG.spawn.replace(
            asset_path=shadowing_robot.urdf_path,
            merge_fixed_joints=shadowing_robot.import_options["merge_fixed_joints"],
        )
    )

    parkour_robot = G1_29DOF_PARKOUR_CFG
    G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG.copy()
    G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG.spawn = (
        G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG.spawn.replace(
            asset_path=parkour_robot.urdf_path,
            merge_fixed_joints=parkour_robot.import_options["merge_fixed_joints"],
        )
    )
    G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG.init_state = (
        G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG.init_state.replace(
            pos=parkour_robot.default_root_pos
        )
    )
    G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG.actuators = (
        beyondmimic_g1_29dof_delayed_actuators
    )
    _ISAAC_LOADED = True


def articulation(variant: str, robot) -> object:
    """Build one registered G1 variant as an Isaac Lab articulation."""
    try:
        config_name = ARTICULATIONS[variant]
    except KeyError:
        raise KeyError(
            f"Unknown Isaac Unitree G1 variant {variant!r}; registered: {sorted(ARTICULATIONS)}"
        ) from None
    _load_isaac()
    cfg = globals()[config_name].copy()
    asset = robot.asset_for("isaacsim")
    spawn_updates = {"asset_path": asset.path}
    for name in ("merge_fixed_joints", "fix_base", "replace_cylinders_with_capsules"):
        if name in asset.import_options:
            spawn_updates[name] = asset.import_options[name]
    cfg.spawn = cfg.spawn.replace(**spawn_updates)
    cfg.init_state = cfg.init_state.replace(pos=robot.default_root_pos)
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
    "G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG",
    "G1_29DOF_TORSOBASE_POPSICLE_SHADOWING_CFG",
    "NATIVE_CONFIGS",
    "articulation",
    "beyondmimic_g1_29dof_actuators",
    "beyondmimic_g1_29dof_delayed_actuators",
    "native_config",
]
