"""Unitree G1: names, PD, pose, ``RobotSpec``, and Isaac ``ArticulationCfg``.

Numbers live here. ``ArticulationCfg`` is built from those numbers on first access so a
cross-engine ``TaskSpec`` can import ``make_g1_29dof_robot_spec`` without loading Isaac Lab.
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
        actuator_group=_g1_actuator_group(name),
    )


def _g1_actuator_group(name: str) -> str:
    """Which motor bus a joint hangs off, as both references declare it.

    main's ``beyondmimic_g1_29dof_actuators`` and InstinctMJ's delayed cfgs agree on this
    partition down to the membership: the whole leg on one, feet, waist, waist_yaw, arms.
    It deliberately does not follow the PD gains -- ``legs`` spans two gain sets and ``arms``
    spans two, which is exactly why gain-keyed grouping produces a different plant.
    """
    if "_hip_" in name or "_knee_" in name:
        return "legs"
    if "_ankle_" in name:
        return "feet"
    if name == "waist_yaw_joint":
        return "waist_yaw"
    if name in {"waist_pitch_joint", "waist_roll_joint"}:
        return "waist"
    return "arms"


def make_g1_29dof_robot_spec() -> RobotSpec:
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
                path=str(RESOURCE_ROOT / "urdf" / "g1_29dof_torsobase_popsicle.urdf"),
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
                path=str(RESOURCE_ROOT / "xml" / "g1_29dof_torsobase_popsicle.xml"),
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

_BEYONDMIMIC_JOINT_GROUPS = {
    "legs": (
        ".*_hip_yaw_joint",
        ".*_hip_roll_joint",
        ".*_hip_pitch_joint",
        ".*_knee_joint",
    ),
    "feet": (".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    "waist": ("waist_roll_joint", "waist_pitch_joint"),
    "waist_yaw": ("waist_yaw_joint",),
    "arms": (
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_roll_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_yaw_joint",
    ),
}

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
    "ARMATURE_4010",
    "ARMATURE_5020",
    "ARMATURE_7520_14",
    "ARMATURE_7520_22",
    "DAMPING_4010",
    "DAMPING_5020",
    "DAMPING_7520_14",
    "DAMPING_7520_22",
    "DAMPING_RATIO",
    "G1_29DOF_DEFAULT_JOINT_POS",
    "G1_29DOF_DFS_BODY_NAMES",
    "G1_29DOF_DFS_COLLISION_BODY_NAMES",
    "G1_29DOF_DFS_FRAME_NAMES",
    "G1_29DOF_DFS_JOINT_NAMES",
    "G1_29DOF_ISAAC_BFS_JOINT_NAMES",
    "G1_29DOF_LINKS",
    "G1_29DOF_TORSOBASE_CFG",
    "G1_29DOF_TORSOBASE_CLOG_CFG",
    "G1_29DOF_TORSOBASE_POPSICLE_CFG",
    "G1_29Dof_TorsoBase_isaac_bfs_symmetric_augmentation_joint_mapping",
    "G1_29Dof_TorsoBase_isaac_bfs_symmetric_augmentation_joint_reverse_buf",
    "G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping",
    "G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf",
    "NATURAL_FREQ",
    "RESOURCE_ROOT",
    "STIFFNESS_4010",
    "STIFFNESS_5020",
    "STIFFNESS_7520_14",
    "STIFFNESS_7520_22",
    "beyondmimic_action_scale",
    "beyondmimic_g1_29dof_actuators",
    "beyondmimic_g1_29dof_delayed_actuators",
    "g1_symmetric_joint_augmentation",
    "make_g1_29dof_robot_spec",
]
