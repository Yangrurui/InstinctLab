"""Same clip, same frames, both robot descriptions. Physics is not involved.

The reference is dataset plus FK. Joint angles and the clip root come from the
file; they must match exactly. Link poses go through two kinematic parsers, so
the residual is float32 FK noise, not a different motion.

Tolerances are derived from that residual, not widened after a failure:

* Positions in the clip-root frame: 1e-6 m. Observed max |Δlink_pos_b| is
  4.8e-7 across a 500-frame scan of the parkour clip.
* Quaternion error: 2e-6 rad. Observed max is 1.75e-6, on the arm links,
  from ``pytorch_kinematics`` URDF vs MJCF. That is 1e-4 deg, not a swapped
  frame.
* Body-frame velocities: 1e-6 / clip_dt = 5e-5. Frontward differences at
  50 Hz turn a 1e-6 pose residual into 5e-5.
* World-frame quantities at tens of metres of travel are limited by float32
  ULP (7.6e-6 at 64 m). A far frame is checked against that ULP, not against
  1e-6, which float32 cannot meet at x=72.
"""

from __future__ import annotations

import os
import torch

import pytest

from instinctlab.assets.unitree_g1.isaacsim import G1_29DOF_DFS_JOINT_NAMES, make_g1_29dof_robot_spec
from instinctlab.compat.math import quat_error_magnitude
from instinctlab.engines.motion_reference import (
    build_kinematics_chain,
    chain_inventory,
    load_retargetted_clip,
    pack_motion_clip,
)
from instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import PARKOUR_MOTION_CLIP, PARKOUR_MOTION_LINKS

pytest.importorskip("pytorch_kinematics")

CLIP = os.path.expanduser(PARKOUR_MOTION_CLIP)
POS_ATOL = 1e-6
QUAT_ATOL = 2e-6
CLIP_DT = 1.0 / 50.0
VEL_ATOL = POS_ATOL / CLIP_DT
# Near-origin frames: FK residual only. Frame 8500 is the ULP case (~72 m).
NEAR_FRAMES = (0, 100, 1000, 5000, -1)
FAR_FRAME = 8500


@pytest.fixture(scope="module")
def packed():
    if not os.path.isfile(os.path.realpath(CLIP)):
        pytest.skip(f"parkour clip is not at {CLIP}")
    robot = make_g1_29dof_robot_spec()
    raw = load_retargetted_clip(CLIP, device="cpu")
    urdf = pack_motion_clip(
        raw,
        joint_names=robot.joint_names,
        link_names=PARKOUR_MOTION_LINKS,
        model_path=robot.asset_for("isaacsim").path,
        target_fps=50.0,
    )
    mjcf = pack_motion_clip(
        raw,
        joint_names=robot.joint_names,
        link_names=PARKOUR_MOTION_LINKS,
        model_path=robot.asset_for("mjlab").path,
        target_fps=50.0,
    )
    return raw, urdf, mjcf, robot


def test_the_resolved_clip_is_the_real_parkour_file(packed) -> None:
    raw, urdf, _, _ = packed
    assert os.path.isfile(raw["path"])
    assert not os.path.islink(raw["path"])
    assert raw["path"] == urdf.path
    assert raw["framerate"] == 50.0
    assert urdf.nframes == 18982


def test_the_name_map_is_not_the_identity(packed) -> None:
    raw, urdf, _, _ = packed
    waist_src = raw["joint_names"].index("waist_pitch_joint")
    hip_src = raw["joint_names"].index("left_hip_pitch_joint")
    waist_dst = urdf.joint_names.index("waist_pitch_joint")
    hip_dst = urdf.joint_names.index("left_hip_pitch_joint")
    assert (waist_src, waist_dst) == (14, 0)
    assert hip_src != hip_dst
    assert urdf.joint_index_map[waist_dst] == waist_src
    assert urdf.joint_index_map != tuple(range(len(urdf.joint_index_map)))
    torch.testing.assert_close(urdf.joint_pos[0, waist_dst], raw["joint_pos"][0, waist_src])
    assert not torch.allclose(urdf.joint_pos[0, 0], raw["joint_pos"][0, 0])


def test_both_chains_share_the_actuated_joints_and_the_declared_links(packed) -> None:
    _, urdf, mjcf, robot = packed
    isaac_chain = build_kinematics_chain(robot.asset_for("isaacsim").path)
    mjlab_chain = build_kinematics_chain(robot.asset_for("mjlab").path)
    isaac_inv = chain_inventory(isaac_chain, robot.asset_for("isaacsim").path)
    mjlab_inv = chain_inventory(mjlab_chain, robot.asset_for("mjlab").path)

    assert isaac_inv.kind == "urdf"
    assert mjlab_inv.kind == "mjcf"
    assert isaac_inv.root == "torso_link"
    assert mjlab_inv.root == "world"
    assert isaac_inv.joint_names == mjlab_inv.joint_names
    assert isaac_inv.joint_names == tuple(robot.joint_names) == G1_29DOF_DFS_JOINT_NAMES
    assert set(PARKOUR_MOTION_LINKS) <= set(isaac_inv.link_names)
    assert set(PARKOUR_MOTION_LINKS) <= set(mjlab_inv.link_names)
    only_mjcf = set(mjlab_inv.link_names) - set(isaac_inv.link_names)
    only_urdf = set(isaac_inv.link_names) - set(mjlab_inv.link_names)
    assert only_urdf == set()
    assert only_mjcf == {"world", "left_foot", "right_foot"}
    assert urdf.inventory.joint_names == mjcf.inventory.joint_names
    assert urdf.link_names == mjcf.link_names == PARKOUR_MOTION_LINKS


def _abs_max(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _quat_max(a: torch.Tensor, b: torch.Tensor) -> float:
    return quat_error_magnitude(a.reshape(-1, 4), b.reshape(-1, 4)).max().item()


def _frame_deltas(urdf, mjcf, idx: int) -> dict[str, float]:
    return {
        "joint_pos": _abs_max(urdf.joint_pos[idx], mjcf.joint_pos[idx]),
        "joint_vel": _abs_max(urdf.joint_vel[idx], mjcf.joint_vel[idx]),
        "base_pos_w": _abs_max(urdf.base_pos_w[idx], mjcf.base_pos_w[idx]),
        "base_lin_vel_w": _abs_max(urdf.base_lin_vel_w[idx], mjcf.base_lin_vel_w[idx]),
        "base_ang_vel_w": _abs_max(urdf.base_ang_vel_w[idx], mjcf.base_ang_vel_w[idx]),
        "base_quat_w": _quat_max(urdf.base_quat_w[idx], mjcf.base_quat_w[idx]),
        "link_pos_b": _abs_max(urdf.link_pos_b[idx], mjcf.link_pos_b[idx]),
        "link_pos_w": _abs_max(urdf.link_pos_w[idx], mjcf.link_pos_w[idx]),
        "link_quat_b": _quat_max(urdf.link_quat_b[idx], mjcf.link_quat_b[idx]),
        "link_quat_w": _quat_max(urdf.link_quat_w[idx], mjcf.link_quat_w[idx]),
        "link_lin_vel_b": _abs_max(urdf.link_lin_vel_b[idx], mjcf.link_lin_vel_b[idx]),
        "link_lin_vel_w": _abs_max(urdf.link_lin_vel_w[idx], mjcf.link_lin_vel_w[idx]),
        "link_ang_vel_b": _abs_max(urdf.link_ang_vel_b[idx], mjcf.link_ang_vel_b[idx]),
        "link_ang_vel_w": _abs_max(urdf.link_ang_vel_w[idx], mjcf.link_ang_vel_w[idx]),
    }


def test_the_same_frames_match_across_descriptions(packed) -> None:
    _, urdf, mjcf, _ = packed
    n = urdf.nframes
    report: list[str] = []
    worst: dict[str, float] = {}
    for raw_idx in NEAR_FRAMES:
        idx = raw_idx if raw_idx >= 0 else n + raw_idx
        deltas = _frame_deltas(urdf, mjcf, idx)
        # File-backed quantities are shared. Any non-zero is a remap or copy bug.
        for name in ("joint_pos", "joint_vel", "base_pos_w", "base_lin_vel_w", "base_ang_vel_w"):
            assert deltas[name] == 0.0, f"{name} frame {idx}: {deltas[name]}"
        assert deltas["base_quat_w"] <= QUAT_ATOL, f"base_quat_w frame {idx}: {deltas['base_quat_w']}"
        assert deltas["link_pos_b"] <= POS_ATOL, f"link_pos_b frame {idx}: {deltas['link_pos_b']}"
        assert deltas["link_pos_w"] <= POS_ATOL, f"link_pos_w frame {idx}: {deltas['link_pos_w']}"
        assert deltas["link_quat_b"] <= QUAT_ATOL, f"link_quat_b frame {idx}: {deltas['link_quat_b']}"
        assert deltas["link_quat_w"] <= QUAT_ATOL, f"link_quat_w frame {idx}: {deltas['link_quat_w']}"
        for name in ("link_lin_vel_b", "link_lin_vel_w", "link_ang_vel_b", "link_ang_vel_w"):
            assert deltas[name] <= VEL_ATOL, f"{name} frame {idx}: {deltas[name]} > {VEL_ATOL}"
        report.append(
            f"frame {idx}: joint_pos={deltas['joint_pos']:.2e} "
            f"link_pos_b={deltas['link_pos_b']:.2e} "
            f"link_quat_b={deltas['link_quat_b']:.2e} "
            f"link_lin_vel_w={deltas['link_lin_vel_w']:.2e} "
            f"link_ang_vel_b={deltas['link_ang_vel_b']:.2e}"
        )
        for key, value in deltas.items():
            worst[key] = max(worst.get(key, 0.0), value)
    print("\n".join(report))
    print("worst_near", {key: f"{value:.2e}" for key, value in worst.items()})
    print("waist_pitch frame0", float(urdf.joint_pos[0, 0]), float(mjcf.joint_pos[0, 0]))
    print("left_hip_pitch frame0", float(urdf.joint_pos[0, 3]), float(mjcf.joint_pos[0, 3]))
    print("pelvis_b frame0 urdf", urdf.link_pos_b[0, 0].tolist())
    print("pelvis_b frame0 mjcf", mjcf.link_pos_b[0, 0].tolist())
    print("base_pos_w frame0", urdf.base_pos_w[0].tolist())
    print("base_quat_w frame0", urdf.base_quat_w[0].tolist())


def test_far_travel_world_residual_is_float32_ulp(packed) -> None:
    """At x≈72 m, 1e-6 m is finer than float32. Body-frame FK stays at 1e-6."""
    _, urdf, mjcf, _ = packed
    idx = FAR_FRAME
    deltas = _frame_deltas(urdf, mjcf, idx)
    coord = float(urdf.base_pos_w[idx].abs().max())
    ulp = float(torch.finfo(torch.float32).eps * (2 ** max(0, int(coord).bit_length() - 1)))
    assert coord > 60.0
    assert deltas["joint_pos"] == 0.0
    assert deltas["link_pos_b"] <= POS_ATOL, f"link_pos_b far: {deltas['link_pos_b']}"
    assert deltas["link_quat_b"] <= QUAT_ATOL, f"link_quat_b far: {deltas['link_quat_b']}"
    assert deltas["link_pos_w"] <= 2 * ulp, f"link_pos_w far: {deltas['link_pos_w']} > 2 ULP ({ulp}) at |x|={coord}"
    assert (
        deltas["link_lin_vel_w"] <= 2 * ulp / CLIP_DT
    ), f"link_lin_vel_w far: {deltas['link_lin_vel_w']} vs ULP/dt={ulp / CLIP_DT}"
    print(
        f"frame {idx}: |base|={coord:.3f} ulp={ulp:.2e} "
        f"link_pos_b={deltas['link_pos_b']:.2e} link_pos_w={deltas['link_pos_w']:.2e} "
        f"link_lin_vel_w={deltas['link_lin_vel_w']:.2e}"
    )
