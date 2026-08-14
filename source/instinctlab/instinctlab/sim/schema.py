"""Stable observation/action/checkpoint schema metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .control import ControlSemantics
from .robot_spec import RobotSpec


@dataclass(frozen=True)
class TensorSegment:
    name: str
    shape: tuple[int, ...]
    semantic: str

    @property
    def flat_dim(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


@dataclass(frozen=True)
class ObservationGroupSchema:
    name: str
    segments: tuple[TensorSegment, ...]

    @property
    def flat_dim(self) -> int:
        return sum(segment.flat_dim for segment in self.segments)


@dataclass(frozen=True)
class EnvSchema:
    version: str
    joint_order: str
    body_order: str
    quaternion_order: str
    observation_groups: tuple[ObservationGroupSchema, ...]
    action_segments: tuple[TensorSegment, ...]
    reward_groups: tuple[str, ...]
    control_semantics: ControlSemantics

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["control_semantics"] = self.control_semantics.value
        return payload

    @property
    def hash(self) -> str:
        serialized = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def observation_group(self, name: str) -> ObservationGroupSchema:
        for group in self.observation_groups:
            if group.name == name:
                return group
        raise KeyError(name)


@dataclass(frozen=True)
class CheckpointManifest:
    task: str
    schema: EnvSchema
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    asset_id: str
    backend: str
    backend_version: str
    seed: int
    metadata: dict[str, Any]

    @property
    def schema_hash(self) -> str:
        return self.schema.hash

    def validate_for(self, *, schema: EnvSchema, robot: RobotSpec) -> None:
        if self.schema_hash != schema.hash:
            raise ValueError(
                f"checkpoint schema mismatch: checkpoint={self.schema_hash[:12]}, runtime={schema.hash[:12]}"
            )
        if self.joint_names != robot.joint_names:
            raise ValueError("checkpoint canonical joint names do not match the runtime RobotSpec")
        if self.body_names != robot.body_names:
            raise ValueError("checkpoint canonical body names do not match the runtime RobotSpec")
        if self.asset_id != robot.asset_id:
            raise ValueError(f"checkpoint asset {self.asset_id!r} does not match runtime asset {robot.asset_id!r}")

    @staticmethod
    def validate_payload(payload: dict[str, Any], *, schema: EnvSchema, robot: RobotSpec) -> None:
        """Validate a serialized manifest against the runtime schema and robot.

        The simulator backend is intentionally *not* compared: reusing an
        Isaac-trained policy on MJLab (or vice versa) is the whole point of the
        canonical contract. Only the observation/action schema and the robot's
        canonical indexing must match for a checkpoint to be loadable.
        """
        checkpoint_hash = payload.get("schema_hash")
        if checkpoint_hash != schema.hash:
            raise ValueError(
                f"checkpoint schema mismatch: checkpoint={str(checkpoint_hash)[:12]}, runtime={schema.hash[:12]}"
            )
        if tuple(payload.get("joint_names", ())) != robot.joint_names:
            raise ValueError("checkpoint canonical joint names do not match the runtime RobotSpec")
        if tuple(payload.get("body_names", ())) != robot.body_names:
            raise ValueError("checkpoint canonical body names do not match the runtime RobotSpec")
        if payload.get("asset_id") != robot.asset_id:
            raise ValueError(
                f"checkpoint asset {payload.get('asset_id')!r} does not match runtime asset {robot.asset_id!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "schema": self.schema.as_dict(),
            "schema_hash": self.schema_hash,
            "joint_names": list(self.joint_names),
            "body_names": list(self.body_names),
            "asset_id": self.asset_id,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "seed": self.seed,
            "metadata": self.metadata,
        }


def locomotion_flat_schema(num_joints: int) -> EnvSchema:
    policy = ObservationGroupSchema(
        name="policy",
        segments=(
            TensorSegment("base_ang_vel", (3,), "root_link_ang_vel_body"),
            TensorSegment("projected_gravity", (3,), "gravity_body"),
            TensorSegment("velocity_commands", (3,), "base_velocity_command"),
            TensorSegment("joint_pos", (num_joints,), "joint_position_relative_dfs"),
            TensorSegment("joint_vel", (num_joints,), "joint_velocity_dfs"),
            TensorSegment("actions", (num_joints,), "last_raw_action_dfs"),
        ),
    )
    critic = ObservationGroupSchema(
        name="critic",
        segments=(
            TensorSegment("base_lin_vel", (3,), "root_link_lin_vel_body"),
            *policy.segments,
        ),
    )
    return EnvSchema(
        version="locomotion_flat_v1",
        joint_order="dfs_v1",
        body_order="dfs_v1",
        quaternion_order="wxyz",
        observation_groups=(policy, critic),
        action_segments=(TensorSegment("joint_pos", (num_joints,), "raw_joint_position_action_dfs"),),
        reward_groups=("default",),
        control_semantics=ControlSemantics.NATIVE_IMPLICIT_V1,
    )


__all__ = [
    "CheckpointManifest",
    "EnvSchema",
    "ObservationGroupSchema",
    "TensorSegment",
    "locomotion_flat_schema",
]
