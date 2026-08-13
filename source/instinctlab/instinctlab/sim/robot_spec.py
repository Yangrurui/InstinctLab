"""Canonical robot descriptions shared by all simulator backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


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


@dataclass(frozen=True)
class BackendAsset:
    backend: str
    path: str
    checksum: str | None = None


@dataclass(frozen=True)
class JointProperties:
    name: str
    default_pos: float
    stiffness: float
    damping: float
    armature: float
    effort_limit: float
    velocity_limit: float
    action_scale: float


@dataclass(frozen=True)
class RobotSpec:
    """Engine-independent physical and indexing contract for one robot."""

    name: str
    schema_version: str
    asset_id: str
    root_body: str
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    joint_properties: tuple[JointProperties, ...]
    assets: tuple[BackendAsset, ...]
    default_root_pos: tuple[float, float, float]
    default_root_quat_wxyz: tuple[float, float, float, float]
    soft_joint_pos_limit_factor: float

    def validate(self) -> None:
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("RobotSpec joint_names must be non-empty and unique")
        if not self.body_names or len(set(self.body_names)) != len(self.body_names):
            raise ValueError("RobotSpec body_names must be non-empty and unique")
        if self.root_body != self.body_names[0]:
            raise ValueError("RobotSpec root_body must be the first canonical body")
        property_names = tuple(item.name for item in self.joint_properties)
        if property_names != self.joint_names:
            raise ValueError("joint_properties must exactly follow canonical joint_names")
        asset_backends = tuple(asset.backend for asset in self.assets)
        if len(set(asset_backends)) != len(asset_backends):
            raise ValueError("RobotSpec may declare at most one asset per backend")

    def asset_for(self, backend: str) -> BackendAsset:
        for asset in self.assets:
            if asset.backend == backend:
                return asset
        raise KeyError(f"RobotSpec {self.name!r} has no asset for backend {backend!r}")

    def joint_index(self, name: str) -> int:
        return self.joint_names.index(name)

    def materialize(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        fields = (
            "default_pos",
            "stiffness",
            "damping",
            "armature",
            "effort_limit",
            "velocity_limit",
            "action_scale",
        )
        return {
            field: torch.tensor(
                [getattr(properties, field) for properties in self.joint_properties],
                device=device,
                dtype=dtype,
            )
            for field in fields
        }


def _g1_joint_properties(name: str) -> JointProperties:
    armature_5020 = 0.003609725
    armature_7520_14 = 0.010177520
    armature_7520_22 = 0.025101925
    armature_4010 = 0.00425
    natural_frequency = 10.0 * 2.0 * 3.1415926535
    damping_ratio = 2.0

    def gains(armature: float) -> tuple[float, float]:
        stiffness = armature * natural_frequency**2
        damping = 2.0 * damping_ratio * armature * natural_frequency
        return stiffness, damping

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
        armature, effort_limit, velocity_limit = armature_7520_22, 139.0, 20.0
    elif "_hip_pitch_" in name or "_hip_yaw_" in name or name == "waist_yaw_joint":
        armature, effort_limit, velocity_limit = armature_7520_14, 88.0, 32.0
    elif "_ankle_" in name or name in {"waist_pitch_joint", "waist_roll_joint"}:
        armature, effort_limit, velocity_limit = 2.0 * armature_5020, 50.0, 37.0
    elif "_wrist_pitch_" in name or "_wrist_yaw_" in name:
        armature, effort_limit, velocity_limit = armature_4010, 5.0, 22.0
    else:
        armature, effort_limit, velocity_limit = armature_5020, 25.0, 37.0

    stiffness, damping = gains(armature)
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
    package_root = Path(__file__).resolve().parents[1]
    resource_root = package_root / "assets" / "resources" / "unitree_g1"
    spec = RobotSpec(
        name="unitree_g1_29dof",
        schema_version="dfs_v1",
        asset_id="popsicle_torsobase_v1",
        root_body="torso_link",
        joint_names=G1_29DOF_DFS_JOINT_NAMES,
        body_names=G1_29DOF_DFS_BODY_NAMES,
        joint_properties=tuple(_g1_joint_properties(name) for name in G1_29DOF_DFS_JOINT_NAMES),
        assets=(
            BackendAsset(
                backend="isaacsim",
                path=str(resource_root / "urdf" / "g1_29dof_torsobase_popsicle.urdf"),
            ),
            BackendAsset(
                backend="mjlab",
                path=str(resource_root / "xml" / "g1_29dof_torsobase_popsicle.xml"),
            ),
        ),
        default_root_pos=(0.0, 0.0, 0.82),
        default_root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        soft_joint_pos_limit_factor=0.9,
    )
    spec.validate()
    return spec


__all__ = [
    "BackendAsset",
    "G1_29DOF_DFS_BODY_NAMES",
    "G1_29DOF_DFS_JOINT_NAMES",
    "JointProperties",
    "RobotSpec",
    "make_g1_29dof_robot_spec",
]
