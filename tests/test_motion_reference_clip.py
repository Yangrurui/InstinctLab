"""Clip path, name remap, and exhaustion. These fail silently if they are wrong.

No simulator. The published parkour clip is legs-first; the canonical order is
depth-first. A test that only checks a joint whose index is the same in both
orders is not a remap test.
"""

from __future__ import annotations

import numpy as np
import os
import torch

import pytest

from instinctlab.assets.unitree_g1.isaacsim import G1_29DOF_DFS_JOINT_NAMES, G1_29DOF_ISAAC_BFS_JOINT_NAMES
from instinctlab.engines.motion_reference import (
    JointNameMappingError,
    envs_due_for_update,
    fill_buffers,
    index_at_time,
    interpolate_motion,
    load_retargetted_clip,
    lookahead_times,
    make_buffers,
    pack_motion_clip,
    remap_by_name,
    resolve_clip_path,
    sample_clip,
)
from instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import PARKOUR_MOTION_CLIP

CLIP = os.path.expanduser(PARKOUR_MOTION_CLIP)
ISAAC_SYMLINK = os.path.expanduser("~/Datasets/parkour_motion_without_run_retargetted.npz")


@pytest.fixture(scope="module")
def raw_clip():
    if not os.path.isfile(os.path.realpath(CLIP)):
        pytest.skip(f"parkour clip is not at {CLIP}")
    return load_retargetted_clip(CLIP, device="cpu")


def _write_raw_clip(path, **overrides) -> None:
    arrays = {
        "framerate": np.asarray(50.0),
        "joint_names": np.asarray(["a", "b"]),
        "joint_pos": np.zeros((3, 2), dtype=np.float32),
        "base_pos_w": np.zeros((3, 3), dtype=np.float32),
        "base_quat_w": np.asarray([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=np.float32),
    }
    arrays.update(overrides)
    np.savez(path, **arrays)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"framerate": np.asarray(0.0)}, "invalid framerate"),
        ({"joint_names": np.asarray(["a", "a"])}, "repeats joint names"),
        ({"joint_pos": np.zeros((3, 1), dtype=np.float32)}, "joint_pos must have shape"),
        ({"base_pos_w": np.zeros((2, 3), dtype=np.float32)}, "aligned frames"),
        ({"joint_pos": np.asarray([[0.0, 0.0], [np.nan, 0.0], [0.0, 0.0]])}, "non-finite"),
        ({"base_quat_w": np.asarray([[2.0, 0.0, 0.0, 0.0]] * 3)}, "non-unit"),
    ],
)
def test_clip_loader_rejects_malformed_motion_arrays(tmp_path, override, message) -> None:
    path = tmp_path / "bad.npz"
    _write_raw_clip(path, **override)
    with pytest.raises(ValueError, match=message):
        load_retargetted_clip(str(path))


def test_the_declared_clip_is_the_real_file_not_a_dangling_symlink() -> None:
    real = resolve_clip_path(CLIP)
    assert os.path.isfile(real)
    assert not os.path.islink(real)
    assert os.path.basename(real) == "parkour_motion_without_run_retargetted.npz"
    assert os.path.realpath(ISAAC_SYMLINK) == real


def test_the_published_clip_is_legs_first_and_not_canonical(raw_clip) -> None:
    source = raw_clip["joint_names"]
    canonical = G1_29DOF_DFS_JOINT_NAMES
    bfs = G1_29DOF_ISAAC_BFS_JOINT_NAMES
    assert source != canonical
    assert source != bfs
    assert set(source) == set(canonical) == set(bfs)
    assert source[0] == "left_hip_pitch_joint"
    assert canonical[0] == "waist_pitch_joint"
    assert source.index("waist_pitch_joint") != canonical.index("waist_pitch_joint")
    assert source.index("left_hip_pitch_joint") != canonical.index("left_hip_pitch_joint")


def test_remap_is_by_name_and_waist_pitch_moves(raw_clip) -> None:
    """waist_pitch is index 14 in the clip and 0 in the canonical order.

    A joint that sits at the same index in both orders would stay green if the
    remap silently became positional.
    """
    remapped, index_map = remap_by_name(
        raw_clip["joint_pos"], raw_clip["joint_names"], G1_29DOF_DFS_JOINT_NAMES, what="joint"
    )
    source = raw_clip["joint_names"]
    waist_src = source.index("waist_pitch_joint")
    hip_src = source.index("left_hip_pitch_joint")
    waist_dst = G1_29DOF_DFS_JOINT_NAMES.index("waist_pitch_joint")
    hip_dst = G1_29DOF_DFS_JOINT_NAMES.index("left_hip_pitch_joint")
    assert waist_src != waist_dst
    assert hip_src != hip_dst
    assert index_map[waist_dst] == waist_src
    assert index_map[hip_dst] == hip_src
    assert index_map != tuple(range(len(index_map)))
    torch.testing.assert_close(remapped[0, waist_dst], raw_clip["joint_pos"][0, waist_src])
    torch.testing.assert_close(remapped[0, hip_dst], raw_clip["joint_pos"][0, hip_src])
    # The value at canonical index 0 is waist_pitch, not the clip's legs-first [0].
    assert not torch.allclose(remapped[0, 0], raw_clip["joint_pos"][0, 0])


def test_a_missing_name_is_loud_on_either_side(raw_clip) -> None:
    broken_target = G1_29DOF_DFS_JOINT_NAMES[:-1] + ("not_a_joint",)
    with pytest.raises(JointNameMappingError, match="Missing in source"):
        remap_by_name(raw_clip["joint_pos"], raw_clip["joint_names"], broken_target, what="joint")
    broken_source = raw_clip["joint_names"][:-1] + ("not_a_joint",)
    with pytest.raises(JointNameMappingError, match="Missing in target"):
        remap_by_name(raw_clip["joint_pos"], broken_source, G1_29DOF_DFS_JOINT_NAMES, what="joint")


def test_length_equality_is_not_a_name_check(raw_clip) -> None:
    """Same count, different names: must fail. This is the silent fallback."""
    shuffled = raw_clip["joint_names"][1:] + raw_clip["joint_names"][:1]
    assert len(shuffled) == len(G1_29DOF_DFS_JOINT_NAMES)
    # remap against a target that drops one real name and invents one
    fake_target = G1_29DOF_DFS_JOINT_NAMES[:-1] + ("invented_joint",)
    assert len(fake_target) == len(raw_clip["joint_names"])
    with pytest.raises(JointNameMappingError, match="Name-based joint remap failed"):
        remap_by_name(raw_clip["joint_pos"], raw_clip["joint_names"], fake_target, what="joint")


def _tiny_clip(nframes: int = 5, fps: float = 50.0):
    """A packed clip without FK, so exhaustion can be tested without 19k frames."""
    from instinctlab.engines.motion_reference import ChainInventory, MotionClip

    n_joints, n_links = 2, 1
    joint_pos = torch.arange(nframes, dtype=torch.float32).unsqueeze(-1).expand(nframes, n_joints)
    zeros3 = torch.zeros(nframes, 3)
    zeros4 = torch.zeros(nframes, 4)
    zeros4[:, 0] = 1.0
    zeros_l3 = torch.zeros(nframes, n_links, 3)
    zeros_l4 = torch.zeros(nframes, n_links, 4)
    zeros_l4[..., 0] = 1.0
    return MotionClip(
        path="tiny",
        source_joint_names=("a", "b"),
        joint_names=("a", "b"),
        joint_index_map=(0, 1),
        link_names=("root",),
        joint_pos=joint_pos,
        joint_vel=torch.zeros_like(joint_pos),
        base_pos_w=zeros3,
        base_quat_w=zeros4,
        base_lin_vel_w=zeros3,
        base_ang_vel_w=zeros3,
        link_pos_b=zeros_l3,
        link_quat_b=zeros_l4,
        link_pos_w=zeros_l3,
        link_quat_w=zeros_l4,
        link_lin_vel_b=zeros_l3,
        link_ang_vel_b=zeros_l3,
        link_lin_vel_w=zeros_l3,
        link_ang_vel_w=zeros_l3,
        framerate=fps,
        inventory=ChainInventory("tiny", "root", ("a", "b"), ("root",), "urdf"),
    )


def test_exhaustion_freezes_the_last_frame_and_is_counted() -> None:
    clip = _tiny_clip()
    last = clip.nframes - 1
    assert clip.duration_s == pytest.approx(last / clip.framerate)
    assert clip.sampling_length_s == pytest.approx(clip.nframes / clip.framerate)
    inside = sample_clip(clip, torch.tensor([0.0]))
    assert bool(inside.validity[0])
    assert int(inside.frame_index[0]) == 0

    past = sample_clip(clip, torch.tensor([(last + 50) / clip.framerate]))
    assert not bool(past.validity[0])
    assert int(past.frame_index[0]) == last
    torch.testing.assert_close(past.joint_pos[0], clip.joint_pos[last])

    frames, valid = index_at_time(torch.tensor([clip.duration_s + 1.0]), clip.framerate, clip.nframes)
    assert not bool(valid[0])
    assert int(frames[0]) == last

    buffers = make_buffers(1, 10, 2, 1)
    times, time_to = lookahead_times(
        torch.tensor([clip.duration_s + 1.0]),
        torch.tensor([0.0]),
        10,
        0.02,
        "one_frame_interval",
    )
    fill_buffers(buffers, torch.tensor([0]), sample_clip(clip, times), time_to)
    assert int(buffers.exhausted_count[0]) == 1
    assert bool(buffers.ever_exhausted[0])
    assert not bool(buffers.validity[0].any())
    fill_buffers(buffers, torch.tensor([0]), sample_clip(clip, times), time_to)
    assert int(buffers.exhausted_count[0]) == 2
    # A later reset of the clock does not zero the counter — that is how
    # reset_without_notice hid dataset_exhausted in Episode_Termination.
    buffers.timestamp[0] = 0.0
    assert int(buffers.exhausted_count[0]) == 2


def test_clip_interpolation_matches_the_sources_half_open_timeline() -> None:
    root_pos = torch.arange(5, dtype=torch.float32).unsqueeze(-1).expand(5, 3)
    root_quat = torch.zeros(5, 4)
    root_quat[:, 0] = 1.0
    joint_pos = torch.arange(5, dtype=torch.float32).unsqueeze(-1)
    root_pos_out, root_quat_out, joint_pos_out = interpolate_motion(
        root_pos, root_quat, joint_pos, source_fps=50.0, target_fps=50.0
    )
    assert root_pos_out.shape[0] == root_quat_out.shape[0] == joint_pos_out.shape[0] == 4
    torch.testing.assert_close(root_pos_out, root_pos[:-1])
    torch.testing.assert_close(root_quat_out, root_quat[:-1])
    torch.testing.assert_close(joint_pos_out, joint_pos[:-1])


def test_the_clip_clock_is_due_once_per_update_period_not_per_physics_dt() -> None:
    """decimation=4, physics_dt=0.005: four advances, one refresh. Otherwise
    exhausted_count on mjlab would be 4× Isaac's.
    """
    n = 3
    timestamp = torch.zeros(n)
    last = torch.zeros(n)
    due_counts = []
    for _ in range(4):
        timestamp = timestamp + 0.005
        due = envs_due_for_update(timestamp, last, 0.02)
        due_counts.append(int(due.numel()))
        if due.numel():
            last[due] = timestamp[due]
    assert due_counts == [0, 0, 0, 3]
    # A later reset of one env does not make the others due.
    timestamp[0] = 0.0
    last[0] = 0.0
    timestamp = timestamp + 0.005
    due = envs_due_for_update(timestamp, last, 0.02)
    assert due.tolist() == []
