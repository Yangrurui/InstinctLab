"""Convert the public Unitree-G1 LAFAN1 CSV files to Instinct motion clips.

The public dataset stores a pelvis-root pose followed by 29 joints in Unitree's
legs-first order.  InstinctLab's G1 asset is rooted at ``torso_link`` and its
portable motion contract uses the canonical DFS joint order.  Reversing the
three waist joints is part of changing the kinematic tree root from pelvis to
torso; it is not a task-specific policy adjustment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
from instinctlab.engines.isaacsim.assets import robot_spec

TARGET_JOINT_NAMES = robot_spec("unitree_g1/popsicle_torsobase_v1").joint_names

LAFAN1_G1_JOINT_NAMES = (
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
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
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
REVERSED_WAIST_JOINTS = (
    "waist_pitch_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
)
DEFAULT_SOURCE_REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"
DEFAULT_TARGET_URDF = (
    Path(__file__).resolve().parents[1]
    / "source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
    if torch.any(norms <= 1e-8):
        raise ValueError("LAFAN1 motion contains a zero-length root quaternion.")
    return quaternions / norms


def _make_quaternions_continuous(quaternions: torch.Tensor) -> torch.Tensor:
    """Choose equivalent quaternion signs without introducing velocity spikes."""
    result = quaternions.clone()
    for frame in range(1, result.shape[0]):
        if torch.dot(result[frame - 1], result[frame]) < 0.0:
            result[frame] = -result[frame]
    return result


def load_lafan1_csv(
    path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pelvis position, pelvis quaternion (wxyz), and source-order joints."""
    source = Path(path).expanduser().resolve()
    motion = np.loadtxt(source, delimiter=",", ndmin=2)
    expected_columns = 7 + len(LAFAN1_G1_JOINT_NAMES)
    if motion.ndim != 2 or motion.shape[0] < 2 or motion.shape[1] != expected_columns:
        raise ValueError(
            f"{source} must contain at least two frames and {expected_columns} columns; got {motion.shape}."
        )
    if not np.isfinite(motion).all():
        raise ValueError(f"{source} contains non-finite values.")
    pelvis_pos_w = torch.as_tensor(motion[:, :3], dtype=torch.float32)
    pelvis_quat_xyzw = torch.as_tensor(motion[:, 3:7], dtype=torch.float32)
    pelvis_quat_wxyz = pelvis_quat_xyzw[:, [3, 0, 1, 2]]
    pelvis_quat_wxyz = _make_quaternions_continuous(
        _normalize_quaternions(pelvis_quat_wxyz)
    )
    source_joint_pos = torch.as_tensor(motion[:, 7:], dtype=torch.float32)
    return pelvis_pos_w, pelvis_quat_wxyz, source_joint_pos


def build_target_chain(urdf_path: str | Path):
    path = Path(urdf_path).expanduser().resolve()
    with path.open() as handle:
        chain = pk.build_chain_from_urdf(handle.read())
    chain_joint_names = tuple(chain.get_joint_parameter_names())
    if set(chain_joint_names) != set(TARGET_JOINT_NAMES):
        raise ValueError(
            f"Target URDF joint inventory does not match the G1 29-DoF contract: {chain_joint_names}."
        )
    return chain


def convert_lafan1_arrays(
    pelvis_pos_w: torch.Tensor,
    pelvis_quat_w: torch.Tensor,
    source_joint_pos: torch.Tensor,
    target_chain,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert pelvis-root/source-order arrays to torso-root/canonical-DFS arrays."""
    if pelvis_pos_w.ndim != 2 or pelvis_pos_w.shape[1] != 3:
        raise ValueError(
            f"pelvis_pos_w must have shape (frames, 3), got {tuple(pelvis_pos_w.shape)}."
        )
    frame_count = pelvis_pos_w.shape[0]
    if pelvis_quat_w.shape != (frame_count, 4):
        raise ValueError(
            f"pelvis_quat_w must have shape ({frame_count}, 4), got {tuple(pelvis_quat_w.shape)}."
        )
    if source_joint_pos.shape != (frame_count, len(LAFAN1_G1_JOINT_NAMES)):
        raise ValueError(
            f"source_joint_pos must have shape ({frame_count}, {len(LAFAN1_G1_JOINT_NAMES)}), "
            f"got {tuple(source_joint_pos.shape)}."
        )

    source_indices = {name: index for index, name in enumerate(LAFAN1_G1_JOINT_NAMES)}
    canonical_joint_pos = torch.stack(
        [
            source_joint_pos[:, source_indices[name]]
            for name in TARGET_JOINT_NAMES
        ],
        dim=1,
    )
    for name in REVERSED_WAIST_JOINTS:
        canonical_joint_pos[:, TARGET_JOINT_NAMES.index(name)] *= -1.0

    canonical_indices = {
        name: index for index, name in enumerate(TARGET_JOINT_NAMES)
    }
    chain_joint_pos = torch.stack(
        [
            canonical_joint_pos[:, canonical_indices[name]]
            for name in target_chain.get_joint_parameter_names()
        ],
        dim=1,
    )
    frame_indices = target_chain.get_frame_indices("pelvis", "torso_link")
    frame_poses = target_chain.forward_kinematics(chain_joint_pos, frame_indices)
    pelvis_in_torso = frame_poses["pelvis"]
    torso_in_torso = frame_poses["torso_link"]
    torso_in_pelvis = pelvis_in_torso.inverse().compose(torso_in_torso)

    pelvis_world = pk.Transform3d(rot=pelvis_quat_w, pos=pelvis_pos_w)
    torso_world = pelvis_world.compose(torso_in_pelvis)
    torso_matrix = torso_world.get_matrix()
    torso_pos_w = torso_matrix[:, :3, 3]
    torso_quat_w = pk.matrix_to_quaternion(torso_matrix[:, :3, :3])
    torso_quat_w = _make_quaternions_continuous(_normalize_quaternions(torso_quat_w))
    return canonical_joint_pos, torso_pos_w, torso_quat_w


def convert_file(
    source: str | Path,
    target: str | Path,
    *,
    target_chain,
    framerate: float,
    overwrite: bool = False,
) -> dict[str, object]:
    source_path = Path(source).expanduser().resolve()
    target_path = Path(target).expanduser().resolve()
    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing motion clip: {target_path}"
        )
    pelvis_pos_w, pelvis_quat_w, source_joint_pos = load_lafan1_csv(source_path)
    joint_pos, base_pos_w, base_quat_w = convert_lafan1_arrays(
        pelvis_pos_w,
        pelvis_quat_w,
        source_joint_pos,
        target_chain,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_name(f".{target_path.name}.tmp.npz")
    np.savez(
        temporary_path,
        framerate=np.asarray(float(framerate)),
        joint_names=np.asarray(TARGET_JOINT_NAMES),
        joint_pos=joint_pos.cpu().numpy(),
        base_pos_w=base_pos_w.cpu().numpy(),
        base_quat_w=base_quat_w.cpu().numpy(),
    )
    os.replace(temporary_path, target_path)
    return {
        "source": source_path.name,
        "target": target_path.name,
        "frames": int(joint_pos.shape[0]),
        "source_sha256": _sha256(source_path),
        "target_sha256": _sha256(target_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Directory containing official G1 CSV files.",
    )
    parser.add_argument(
        "--tgt",
        type=Path,
        required=True,
        help="Output directory for Instinct NPZ clips.",
    )
    parser.add_argument(
        "--urdf", type=Path, default=DEFAULT_TARGET_URDF, help="Torso-root G1 URDF."
    )
    parser.add_argument(
        "--framerate", type=float, default=30.0, help="Source CSV frame rate."
    )
    parser.add_argument(
        "--source-revision",
        default=DEFAULT_SOURCE_REVISION,
        help="Hugging Face source revision.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing converted clips."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_dir = args.src.expanduser().resolve()
    target_dir = args.tgt.expanduser().resolve()
    sources = sorted(source_dir.glob("*.csv"))
    if not sources:
        raise FileNotFoundError(f"No CSV files found in {source_dir}.")
    if not np.isfinite(args.framerate) or args.framerate <= 0.0:
        raise ValueError(f"framerate must be positive, got {args.framerate!r}.")

    target_chain = build_target_chain(args.urdf)
    converted = []
    for index, source in enumerate(sources, start=1):
        target = target_dir / f"{source.stem}_retargetted.npz"
        entry = convert_file(
            source,
            target,
            target_chain=target_chain,
            framerate=args.framerate,
            overwrite=args.overwrite,
        )
        converted.append(entry)
        print(f"[{index:03d}/{len(sources):03d}] {source.name} -> {target.name}")

    manifest = {
        "source": "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset",
        "source_revision": args.source_revision,
        "source_format": "Unitree G1 pelvis-root CSV, xyzw quaternion, 30 FPS",
        "target_format": "Instinct G1 torso-root NPZ, canonical DFS joints, wxyz quaternion",
        "target_urdf": str(args.urdf.expanduser().resolve()),
        "framerate": float(args.framerate),
        "files": converted,
    }
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "conversion_manifest.json"
    temporary_manifest = target_dir / ".conversion_manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    print(f"Wrote {len(converted)} clips and {manifest_path}")


if __name__ == "__main__":
    main()
