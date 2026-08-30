"""Physical and schema checks for the public BeyondMimic data conversion."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
from instinctlab.assets.unitree_g1.isaacsim import (
    G1_29DOF_DFS_JOINT_NAMES,
    RESOURCE_ROOT,
)
from instinctlab.motion_reference import load_retargetted_clip

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/lafan1_csv_to_instinct.py"
SPEC = importlib.util.spec_from_file_location("lafan1_csv_to_instinct", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
conversion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(conversion)


def _build_chain(path: Path):
    with path.open() as handle:
        return pk.build_chain_from_urdf(handle.read())


def _chain_joint_pos(
    chain, positions: torch.Tensor, names: tuple[str, ...]
) -> torch.Tensor:
    source_indices = {name: index for index, name in enumerate(names)}
    return torch.stack(
        [
            positions[:, source_indices[name]]
            for name in chain.get_joint_parameter_names()
        ],
        dim=1,
    )


def test_pelvis_to_torso_conversion_preserves_world_link_poses() -> None:
    source_chain = _build_chain(RESOURCE_ROOT / "g1_29dof.urdf")
    target_chain = conversion.build_target_chain(
        RESOURCE_ROOT / "urdf/g1_29dof_torsobase_popsicle.urdf"
    )
    frames = 3
    source_joint_pos = torch.linspace(
        -0.23,
        0.31,
        frames * len(conversion.LAFAN1_G1_JOINT_NAMES),
        dtype=torch.float32,
    ).reshape(frames, -1)
    pelvis_pos_w = torch.tensor([[0.1, -0.2, 0.8], [0.2, -0.1, 0.82], [0.3, 0.0, 0.84]])
    pelvis_quat_w = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9987503, 0.0, 0.0, 0.0499792],
            [0.9950042, 0.0, 0.0, 0.0998334],
        ]
    )
    canonical_joint_pos, torso_pos_w, torso_quat_w = conversion.convert_lafan1_arrays(
        pelvis_pos_w,
        pelvis_quat_w,
        source_joint_pos,
        target_chain,
    )

    links = (
        "torso_link",
        "pelvis",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    source_local = source_chain.forward_kinematics(
        _chain_joint_pos(
            source_chain, source_joint_pos, conversion.LAFAN1_G1_JOINT_NAMES
        ),
        source_chain.get_frame_indices(*links),
    )
    target_local = target_chain.forward_kinematics(
        _chain_joint_pos(target_chain, canonical_joint_pos, G1_29DOF_DFS_JOINT_NAMES),
        target_chain.get_frame_indices(*links),
    )
    source_world = pk.Transform3d(rot=pelvis_quat_w, pos=pelvis_pos_w)
    target_world = pk.Transform3d(rot=torso_quat_w, pos=torso_pos_w)
    for link in links:
        source_matrix = source_world.compose(source_local[link]).get_matrix()
        target_matrix = target_world.compose(target_local[link]).get_matrix()
        # The production popsicle asset has small torso/arm translation edits relative to the
        # public visualization URDF.  Rotations remain equivalent everywhere; pelvis and leg
        # positions must also remain equivalent because they anchor the contact trajectory.
        torch.testing.assert_close(
            source_matrix[:, :3, :3],
            target_matrix[:, :3, :3],
            atol=2e-5,
            rtol=2e-5,
        )
        if link in {"pelvis", "left_ankle_roll_link", "right_ankle_roll_link"}:
            torch.testing.assert_close(
                source_matrix[:, :3, 3],
                target_matrix[:, :3, 3],
                atol=2e-5,
                rtol=2e-5,
            )


def test_converted_file_uses_the_runtime_schema_and_canonical_order(
    tmp_path: Path,
) -> None:
    frames = 4
    source = np.zeros(
        (frames, 7 + len(conversion.LAFAN1_G1_JOINT_NAMES)), dtype=np.float32
    )
    source[:, 2] = 0.8
    source[:, 6] = 1.0
    for source_index in range(len(conversion.LAFAN1_G1_JOINT_NAMES)):
        source[:, 7 + source_index] = 0.01 * (source_index + 1)
    csv_path = tmp_path / "motion.csv"
    np.savetxt(csv_path, source, delimiter=",")
    target_path = tmp_path / "motion_retargetted.npz"
    target_chain = conversion.build_target_chain(
        RESOURCE_ROOT / "urdf/g1_29dof_torsobase_popsicle.urdf"
    )
    conversion.convert_file(
        csv_path, target_path, target_chain=target_chain, framerate=30.0
    )

    clip = load_retargetted_clip(target_path)
    assert clip["framerate"] == 30.0
    assert clip["joint_names"] == G1_29DOF_DFS_JOINT_NAMES
    assert clip["joint_pos"].shape == (frames, 29)
    assert clip["base_pos_w"].shape == (frames, 3)
    assert clip["base_quat_w"].shape == (frames, 4)
    waist_pitch_source = conversion.LAFAN1_G1_JOINT_NAMES.index("waist_pitch_joint")
    waist_pitch_target = G1_29DOF_DFS_JOINT_NAMES.index("waist_pitch_joint")
    expected = -source[0, 7 + waist_pitch_source]
    torch.testing.assert_close(
        clip["joint_pos"][0, waist_pitch_target], torch.tensor(expected)
    )
