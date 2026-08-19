"""The engine-neutral Unitree G1 catalog: names, physical parameters, ``RobotSpec``.

One source of truth for facts neither engine owns -- the depth-first joint and body order that
decision D1 rests on, the actuator constants, and where each engine's asset file lives. Nothing here
imports a simulator, so a task can be declared, compared and checked against this robot on a machine
with no engine installed at all.

``unitree_g1`` holds the Isaac Lab ``ArticulationCfg`` views of these same values and imports them
from here. The dependency runs one way on purpose: for a while these two modules imported each
other, with the neutral catalog forwarding Isaac's names back through ``__getattr__`` because it had
taken over the module name the Isaac configs were already published under.
"""

from __future__ import annotations

from pathlib import Path

from instinctlab.sim.robot_spec import BackendAsset, JointProperties, RobotSpec

__file_dir__ = str(Path(__file__).resolve().parent)

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

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

_G1_CONTACT_BODY_ALIASES = {
    "LL_FOOT": "left_ankle_roll_link",
    "LR_FOOT": "right_ankle_roll_link",
}


def _g1_joint_properties(name: str) -> JointProperties:
    default_pos = 0.0
    if "_hip_pitch_" in name:
        default_pos = -0.312
    elif "_knee_" in name:
        default_pos = 0.669
    elif "_ankle_pitch_" in name:
        default_pos = -0.363
    elif "_elbow_" in name:
        default_pos = 0.6
    elif name in {"left_shoulder_roll_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint"}:
        default_pos = 0.2
    elif name == "right_shoulder_roll_joint":
        default_pos = -0.2

    if "_hip_roll_" in name or "_knee_" in name:
        armature, effort_limit, velocity_limit = ARMATURE_7520_22, 139.0, 20.0
    elif "_hip_pitch_" in name or "_hip_yaw_" in name or name == "waist_yaw_joint":
        armature, effort_limit, velocity_limit = ARMATURE_7520_14, 88.0, 32.0
    elif "_ankle_" in name or name in {"waist_pitch_joint", "waist_roll_joint"}:
        armature, effort_limit, velocity_limit = 2.0 * ARMATURE_5020, 50.0, 37.0
    elif "_wrist_pitch_" in name or "_wrist_yaw_" in name:
        armature, effort_limit, velocity_limit = ARMATURE_4010, 5.0, 22.0
    else:
        armature, effort_limit, velocity_limit = ARMATURE_5020, 25.0, 37.0

    stiffness = armature * NATURAL_FREQ**2
    damping = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
    return JointProperties(
        name=name,
        default_pos=default_pos,
        stiffness=stiffness,
        damping=damping,
        armature=armature,
        effort_limit=effort_limit,
        velocity_limit=velocity_limit,
        action_scale=0.25 * effort_limit / stiffness,
    )


def make_g1_29dof_robot_spec() -> RobotSpec:
    resource_root = Path(__file_dir__) / "resources" / "unitree_g1"
    spec = RobotSpec(
        name="unitree_g1_29dof",
        schema_version="dfs_v1",
        asset_id="popsicle_torsobase_v1",
        root_body="torso_link",
        joint_names=G1_29DOF_DFS_JOINT_NAMES,
        body_names=G1_29DOF_DFS_BODY_NAMES,
        frame_names=G1_29DOF_DFS_FRAME_NAMES,
        collision_body_names=G1_29DOF_DFS_COLLISION_BODY_NAMES,
        joint_properties=tuple(_g1_joint_properties(name) for name in G1_29DOF_DFS_JOINT_NAMES),
        assets=(
            BackendAsset(
                backend="isaacsim",
                path=str(resource_root / "urdf" / "g1_29dof_torsobase_popsicle.urdf"),
                checksum="5e99d930af64f3f0bfd34235239e288cc380ba62c8b01fcfcf8cc9f0996e7fa4",
                contact_body_aliases=_G1_CONTACT_BODY_ALIASES,
                import_options={
                    "prim_path": "{ENV_REGEX_NS}/Robot",
                    "fix_base": False,
                    "merge_fixed_joints": False,
                    "replace_cylinders_with_capsules": True,
                },
            ),
            BackendAsset(
                backend="mjlab",
                path=str(resource_root / "xml" / "g1_29dof_torsobase_popsicle.xml"),
                checksum="3dae19dcdc17fbb2d37db29e3dd58894cb150e9edc96696f31a34387178ee18b",
                contact_body_aliases=_G1_CONTACT_BODY_ALIASES,
                load_mode="strip_visual_meshes",
            ),
        ),
        default_root_pos=(0.0, 0.0, 0.82),
        default_root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        soft_joint_pos_limit_factor=0.9,
    )
    spec.validate()
    return spec


__all__ = [
    "ARMATURE_4010",
    "ARMATURE_5020",
    "ARMATURE_7520_14",
    "ARMATURE_7520_22",
    "DAMPING_4010",
    "DAMPING_5020",
    "DAMPING_7520_14",
    "DAMPING_7520_22",
    "DAMPING_RATIO",
    "G1_29DOF_DFS_BODY_NAMES",
    "G1_29DOF_DFS_COLLISION_BODY_NAMES",
    "G1_29DOF_DFS_FRAME_NAMES",
    "G1_29DOF_DFS_JOINT_NAMES",
    "NATURAL_FREQ",
    "STIFFNESS_4010",
    "STIFFNESS_5020",
    "STIFFNESS_7520_14",
    "STIFFNESS_7520_22",
    "make_g1_29dof_robot_spec",
]
