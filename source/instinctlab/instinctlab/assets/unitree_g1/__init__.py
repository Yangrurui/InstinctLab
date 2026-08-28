"""Unitree G1's engine-neutral task and checkpoint interface.

Native assets and actuator configurations live in :mod:`.isaacsim` and
:mod:`.mjlab`.  This package front contains only the names and values that a
task or checkpoint must agree on across engines; it is not a third native
asset configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from instinctlab.sim.robot_spec import BackendAsset, JointProperties, RobotSpec
from instinctlab.utils.name_order import resolve_name_indices

RESOURCE_ROOT = Path(__file__).resolve().parent.parent / "resources" / "unitree_g1"

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

# Isaac/PhysX's native breadth-first-like articulation order. It is not the plain
# breadth-first traversal of URDF children in file order: the importer places the
# shoulder branches before the waist branch at each of the first levels.
#
# This is diagnostic/native order only. Policy actions, observations and motion
# references use ``G1_29DOF_DFS_JOINT_NAMES`` and resolve native indices by name.
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


def _mirrored_joint_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def g1_symmetric_joint_augmentation(
    joint_names: Sequence[str] = G1_29DOF_DFS_JOINT_NAMES,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Left/right swap indices and sagittal sign flips for one joint order.

    The mapping is ``out[i] = in[mapping[i]] * reverse[i]``. Pitch (and elbow/knee) keep
    their sign; roll and yaw flip. Passing a different name list is what keeps Isaac-only
    tasks that still see PhysX BFS from sharing the DFS tables the rest of the stack uses.
    """
    names = tuple(joint_names)
    mirrored = tuple(_mirrored_joint_name(name) for name in names)
    mapping = resolve_name_indices(names, mirrored, require_exact=True)
    reverse: list[int] = []
    for name in names:
        reverse.append(-1 if "roll" in name or "yaw" in name else 1)
    return mapping, tuple(reverse)


_G1_CONTACT_BODY_ALIASES = {
    "LL_FOOT": "left_ankle_roll_link",
    "LR_FOOT": "right_ankle_roll_link",
}

# Standing pose, one number per joint, same order as ``G1_29DOF_DFS_JOINT_NAMES``.
# Zeros are written too: a missing key is a missing joint, not "leave it at zero".
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
if tuple(G1_29DOF_DEFAULT_JOINT_POS) != G1_29DOF_DFS_JOINT_NAMES:
    raise ValueError("G1_29DOF_DEFAULT_JOINT_POS must list every joint in G1_29DOF_DFS_JOINT_NAMES order")


def _g1_joint_properties(name: str) -> JointProperties:
    default_pos = G1_29DOF_DEFAULT_JOINT_POS[name]

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


def _make_g1_29dof_robot_spec(
    *,
    variant: str,
    root_height: float,
    merge_fixed_joints: bool,
    actuator_delay: tuple[int, int] = (0, 0),
    isaac_path: Path | None = None,
    mjlab_path: Path | None = None,
) -> RobotSpec:
    isaac_path = isaac_path or RESOURCE_ROOT / "urdf" / "g1_29dof_torsobase_popsicle.urdf"
    mjlab_path = mjlab_path or RESOURCE_ROOT / "xml" / "g1_29dof_torsobase_popsicle.xml"
    spec = RobotSpec(
        name="unitree_g1_29dof",
        schema_version="dfs_v1",
        asset_id=f"unitree_g1/{variant}",
        root_body="torso_link",
        joint_names=G1_29DOF_DFS_JOINT_NAMES,
        body_names=G1_29DOF_DFS_BODY_NAMES,
        frame_names=G1_29DOF_DFS_FRAME_NAMES,
        collision_body_names=G1_29DOF_DFS_COLLISION_BODY_NAMES,
        joint_properties=tuple(_g1_joint_properties(name) for name in G1_29DOF_DFS_JOINT_NAMES),
        assets=(
            BackendAsset(
                backend="isaacsim",
                path=str(isaac_path),
                contact_body_aliases=_G1_CONTACT_BODY_ALIASES,
                import_options={
                    "prim_path": "{ENV_REGEX_NS}/Robot",
                    "fix_base": False,
                    "merge_fixed_joints": merge_fixed_joints,
                    "replace_cylinders_with_capsules": True,
                },
            ),
            BackendAsset(
                backend="mjlab",
                path=str(mjlab_path),
                contact_body_aliases=_G1_CONTACT_BODY_ALIASES,
                load_mode="strip_visual_meshes",
            ),
        ),
        default_root_pos=(0.0, 0.0, root_height),
        default_root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        soft_joint_pos_limit_factor=0.9,
        actuator_delay=actuator_delay,
    )
    spec.validate()
    return spec


def make_g1_29dof_robot_spec() -> RobotSpec:
    """Build the standard engine-neutral G1 interface."""
    return _make_g1_29dof_robot_spec(
        variant="popsicle_torsobase_v1",
        root_height=0.82,
        merge_fixed_joints=False,
    )


def make_g1_29dof_shadowing_robot_spec() -> RobotSpec:
    """Build the G1 interface used by main's Shadowing tasks."""
    return _make_g1_29dof_robot_spec(
        variant="popsicle_torsobase_shadowing_v1",
        root_height=0.82,
        merge_fixed_joints=True,
    )


def make_g1_29dof_parkour_robot_spec() -> RobotSpec:
    """Build the shoe-equipped G1 interface used by Parkour."""
    return _make_g1_29dof_robot_spec(
        variant="popsicle_torsobase_parkour_v1",
        root_height=0.9,
        merge_fixed_joints=True,
        actuator_delay=(0, 2),
        isaac_path=RESOURCE_ROOT / "urdf" / "g1_29dof_torsoBase_popsicle_with_shoe.urdf",
        mjlab_path=RESOURCE_ROOT / "xml" / "g1_29dof_torsoBase_popsicle_with_shoe.xml",
    )


_G1_DFS_JOINT_MAP, _G1_DFS_JOINT_REVERSE = g1_symmetric_joint_augmentation(G1_29DOF_DFS_JOINT_NAMES)
G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping = list(_G1_DFS_JOINT_MAP)
G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf = list(_G1_DFS_JOINT_REVERSE)
_G1_BFS_JOINT_MAP, _G1_BFS_JOINT_REVERSE = g1_symmetric_joint_augmentation(G1_29DOF_ISAAC_BFS_JOINT_NAMES)
G1_29Dof_TorsoBase_isaac_bfs_symmetric_augmentation_joint_mapping = list(_G1_BFS_JOINT_MAP)
G1_29Dof_TorsoBase_isaac_bfs_symmetric_augmentation_joint_reverse_buf = list(_G1_BFS_JOINT_REVERSE)

beyondmimic_action_scale = {joint.name: joint.action_scale for joint in make_g1_29dof_robot_spec().joint_properties}

G1_29DOF_LINKS = [  # Order not guaranteed.
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

__all__ = [
    "G1_29DOF_DEFAULT_JOINT_POS",
    "G1_29DOF_DFS_BODY_NAMES",
    "G1_29DOF_DFS_COLLISION_BODY_NAMES",
    "G1_29DOF_DFS_FRAME_NAMES",
    "G1_29DOF_DFS_JOINT_NAMES",
    "G1_29DOF_ISAAC_BFS_JOINT_NAMES",
    "G1_29DOF_LINKS",
    "RESOURCE_ROOT",
    "beyondmimic_action_scale",
    "g1_symmetric_joint_augmentation",
    "make_g1_29dof_robot_spec",
    "make_g1_29dof_parkour_robot_spec",
    "make_g1_29dof_shadowing_robot_spec",
]
