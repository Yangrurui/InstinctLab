"""AMP left-right symmetric augmentation: IR, name maps, operators, both engines.

The source integer tables are written against two different joint orders
(InstinctMJ = canonical DFS, main parkour = PhysX BFS). Pasting either table
onto the portable clip (canonical order, published npz is legs-first) is the
silent failure this file exists to catch. Maps are names; indices are resolved
in *our* order.
"""

from __future__ import annotations

import ast
import subprocess
import torch
from pathlib import Path

import pytest

from instinctlab.assets.unitree_g1.catalog import G1_29DOF_DFS_JOINT_NAMES, G1_29DOF_ISAAC_BFS_JOINT_NAMES
from instinctlab.engines.motion_reference import (
    ChainInventory,
    MotionClip,
    MotionReferenceRuntime,
    SymmetricMappingError,
    apply_symmetric_augmentation,
    augment_ang_vel_buffer,
    augment_joint_buffer,
    augment_link_pos_buffer,
    draw_symmetric_mask,
    make_buffers,
    resolve_symmetric_augmentation,
)
from instinctlab.spec.sensor import MotionReferenceRef, SymmetricAugmentationSpec
from instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import PARKOUR_MOTION_LINKS

REPO = Path(__file__).resolve().parents[1]
INSTINCTMJ_ASSET = Path("/root/InstinctMJ/src/instinct_mj/assets/unitree_g1.py")
SOURCE_LINK_MAPPING = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]


def _literal_assign(source: str, name: str) -> list:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} assignment not found")


def _instinctmj_joint_tables() -> tuple[list[int], list[int]]:
    text = INSTINCTMJ_ASSET.read_text()
    return (
        _literal_assign(text, "G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping"),
        _literal_assign(text, "G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf"),
    )


def _main_joint_tables() -> tuple[list[int], list[int]]:
    shown = subprocess.run(
        ("git", "show", "main:source/instinctlab/instinctlab/assets/unitree_g1.py"),
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return (
        _literal_assign(shown.stdout, "G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping"),
        _literal_assign(shown.stdout, "G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf"),
    )


def _name_pairs(names: tuple[str, ...], mapping: list[int] | tuple[int, ...]) -> dict[str, str]:
    return {names[i]: names[mapping[i]] for i in range(len(names))}


def _tiny_aug(**overrides) -> SymmetricAugmentationSpec:
    fields = dict(
        joint_swaps={"left_hip_pitch_joint": "right_hip_pitch_joint", "right_hip_pitch_joint": "left_hip_pitch_joint"},
        joint_signs={"left_hip_pitch_joint": 1, "right_hip_pitch_joint": 1},
        link_swaps={"left_knee_link": "right_knee_link", "right_knee_link": "left_knee_link"},
    )
    fields.update(overrides)
    return SymmetricAugmentationSpec(**fields)


def _ref(**overrides) -> MotionReferenceRef:
    fields = dict(
        name="motion_reference",
        clip="clip.npz",
        joints=("left_hip_pitch_joint", "right_hip_pitch_joint"),
        links=("left_knee_link", "right_knee_link"),
    )
    fields.update(overrides)
    return MotionReferenceRef(**fields)


def test_mirroring_is_off_unless_a_task_turns_it_on() -> None:
    sensor = _ref()
    assert sensor.symmetric_augmentation is None
    assert not hasattr(sensor, "symmetric_augmentation_joint_mapping")


def test_from_left_right_swaps_sides_and_keeps_midline() -> None:
    joints = ("waist_pitch_joint", "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint")
    links = ("pelvis", "torso_link", "left_knee_link", "right_knee_link")
    spec = SymmetricAugmentationSpec.from_left_right(joints, links)
    assert spec.joint_swaps["left_hip_pitch_joint"] == "right_hip_pitch_joint"
    assert spec.joint_swaps["right_hip_pitch_joint"] == "left_hip_pitch_joint"
    assert spec.joint_swaps["waist_pitch_joint"] == "waist_pitch_joint"
    assert spec.joint_swaps["waist_yaw_joint"] == "waist_yaw_joint"
    assert spec.link_swaps["left_knee_link"] == "right_knee_link"
    assert spec.link_swaps["pelvis"] == "pelvis"
    assert spec.link_swaps["torso_link"] == "torso_link"
    assert spec.joint_signs["waist_pitch_joint"] == 1
    assert spec.joint_signs["waist_yaw_joint"] == -1
    assert spec.joint_signs["left_hip_pitch_joint"] == 1


def test_from_left_right_flips_roll_and_yaw_only() -> None:
    joints = (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_elbow_joint",
        "right_elbow_joint",
    )
    spec = SymmetricAugmentationSpec.from_left_right(joints, ("pelvis",))
    assert spec.joint_signs["left_hip_pitch_joint"] == 1
    assert spec.joint_signs["left_knee_joint"] == 1
    assert spec.joint_signs["left_elbow_joint"] == 1
    assert spec.joint_signs["left_hip_roll_joint"] == -1
    assert spec.joint_signs["right_hip_roll_joint"] == -1


def test_a_one_way_swap_is_refused() -> None:
    with pytest.raises(ValueError, match="involution"):
        _tiny_aug(
            joint_swaps={"left_hip_pitch_joint": "right_hip_pitch_joint", "right_hip_pitch_joint": "left_knee_joint"}
        )


def test_empty_swaps_are_refused_use_none_to_disable() -> None:
    with pytest.raises(ValueError, match="empty"):
        _tiny_aug(joint_swaps={}, joint_signs={})
    with pytest.raises(ValueError, match="empty"):
        _tiny_aug(link_swaps={})


def test_signs_must_cover_exactly_the_swapped_joints() -> None:
    with pytest.raises(ValueError, match="exactly the joints"):
        _tiny_aug(joint_signs={"left_hip_pitch_joint": 1})
    with pytest.raises(ValueError, match=r"\+1 or -1"):
        _tiny_aug(
            joint_signs={"left_hip_pitch_joint": 1, "right_hip_pitch_joint": 0},
        )


def test_an_unpaired_left_name_is_loud() -> None:
    with pytest.raises(ValueError, match="not in the name list"):
        SymmetricAugmentationSpec.from_left_right(
            ("left_hip_pitch_joint", "waist_pitch_joint"),
            ("pelvis",),
        )


def test_the_sensor_refuses_a_map_whose_names_are_not_its_joints() -> None:
    with pytest.raises(ValueError, match="do not match sensor joints"):
        _ref(
            joints=("waist_pitch_joint",),
            links=("left_knee_link", "right_knee_link"),
            symmetric_augmentation=_tiny_aug(),
        )


def test_the_sensor_accepts_a_matching_name_map() -> None:
    sensor = _ref(symmetric_augmentation=_tiny_aug())
    assert sensor.symmetric_augmentation is not None
    assert sensor.symmetric_augmentation.joint_swaps["left_hip_pitch_joint"] == "right_hip_pitch_joint"


def test_resolved_dfs_indices_equal_instinctmj_table() -> None:
    """InstinctMJ's integers are DFS. Our name map, resolved in DFS, must match them."""
    mapping, reverse = _instinctmj_joint_tables()
    spec = SymmetricAugmentationSpec.from_left_right(G1_29DOF_DFS_JOINT_NAMES, PARKOUR_MOTION_LINKS)
    resolved = resolve_symmetric_augmentation(spec, G1_29DOF_DFS_JOINT_NAMES, PARKOUR_MOTION_LINKS)
    assert list(resolved.joint_mapping) == mapping
    assert list(resolved.joint_signs) == reverse
    assert list(resolved.link_mapping) == SOURCE_LINK_MAPPING
    assert len(resolved.joint_mapping) == 29
    assert len(resolved.link_mapping) == 14
    for i, name in enumerate(G1_29DOF_DFS_JOINT_NAMES):
        other = G1_29DOF_DFS_JOINT_NAMES[resolved.joint_mapping[i]]
        if name.startswith("left_"):
            assert other == "right_" + name[5:]
        elif name.startswith("right_"):
            assert other == "left_" + name[6:]
        else:
            assert other == name


def test_resolved_bfs_indices_equal_main_table() -> None:
    """Main parkour's integers are PhysX BFS. Same names, different order, different table."""
    mapping, reverse = _main_joint_tables()
    spec = SymmetricAugmentationSpec.from_left_right(G1_29DOF_ISAAC_BFS_JOINT_NAMES, PARKOUR_MOTION_LINKS)
    resolved = resolve_symmetric_augmentation(spec, G1_29DOF_ISAAC_BFS_JOINT_NAMES, PARKOUR_MOTION_LINKS)
    assert list(resolved.joint_mapping) == mapping
    assert list(resolved.joint_signs) == reverse
    assert mapping != list(_instinctmj_joint_tables()[0])


def test_pasting_instinctmj_indices_onto_bfs_swaps_the_wrong_joints() -> None:
    """The trap: same length, no exception, wrong pairing. This is why maps are names."""
    mj_mapping, _ = _instinctmj_joint_tables()
    bfs = G1_29DOF_ISAAC_BFS_JOINT_NAMES
    naive = _name_pairs(bfs, mj_mapping)
    correct = {
        name: (
            "right_" + name[5:]
            if name.startswith("left_")
            else "left_" + name[6:] if name.startswith("right_") else name
        )
        for name in bfs
    }
    assert naive != correct
    assert naive[bfs[0]] != correct[bfs[0]]
    assert bfs[0] == "left_shoulder_pitch_joint"
    # InstinctMJ[0] is identity (waist). Pasted onto BFS[0] it leaves the left shoulder un-swapped.
    assert naive[bfs[0]] == bfs[0]


def test_a_name_map_resolves_in_any_joint_order() -> None:
    """The IR is names. Building it from DFS and resolving onto BFS must equal main's table."""
    spec = SymmetricAugmentationSpec.from_left_right(G1_29DOF_DFS_JOINT_NAMES, PARKOUR_MOTION_LINKS)
    resolved = resolve_symmetric_augmentation(spec, G1_29DOF_ISAAC_BFS_JOINT_NAMES, PARKOUR_MOTION_LINKS)
    mapping, reverse = _main_joint_tables()
    assert list(resolved.joint_mapping) == mapping
    assert list(resolved.joint_signs) == reverse


def test_resolve_refuses_a_missing_joint() -> None:
    spec = SymmetricAugmentationSpec.from_left_right(G1_29DOF_DFS_JOINT_NAMES, PARKOUR_MOTION_LINKS)
    with pytest.raises(SymmetricMappingError, match="not match"):
        resolve_symmetric_augmentation(spec, G1_29DOF_DFS_JOINT_NAMES[:-1], PARKOUR_MOTION_LINKS)


def test_mirroring_twice_is_identity() -> None:
    spec = SymmetricAugmentationSpec.from_left_right(G1_29DOF_DFS_JOINT_NAMES, PARKOUR_MOTION_LINKS)
    resolved = resolve_symmetric_augmentation(spec, G1_29DOF_DFS_JOINT_NAMES, PARKOUR_MOTION_LINKS)
    buffers = make_buffers(2, 3, 29, 14)
    torch.manual_seed(1)
    buffers.joint_pos.normal_()
    buffers.joint_vel.normal_()
    buffers.base_pos_w.normal_()
    buffers.base_quat_w.normal_()
    buffers.base_lin_vel_w.normal_()
    buffers.base_ang_vel_w.normal_()
    buffers.link_pos_w.normal_()
    buffers.link_quat_w.normal_()
    buffers.link_pos_b.normal_()
    buffers.link_quat_b.normal_()
    buffers.link_lin_vel_w.normal_()
    buffers.link_ang_vel_w.normal_()
    buffers.link_lin_vel_b.normal_()
    buffers.link_ang_vel_b.normal_()
    before = {
        name: getattr(buffers, name).clone()
        for name in (
            "joint_pos",
            "joint_vel",
            "base_pos_w",
            "base_quat_w",
            "base_lin_vel_w",
            "base_ang_vel_w",
            "link_pos_w",
            "link_quat_w",
            "link_pos_b",
            "link_quat_b",
            "link_lin_vel_w",
            "link_ang_vel_w",
            "link_lin_vel_b",
            "link_ang_vel_b",
        )
    }
    env_ids = torch.arange(2)
    mask = torch.ones(2, dtype=torch.bool)
    apply_symmetric_augmentation(buffers, env_ids, mask, resolved)
    apply_symmetric_augmentation(buffers, env_ids, mask, resolved)
    max_err = 0.0
    for name, original in before.items():
        err = float((getattr(buffers, name) - original).abs().max())
        max_err = max(max_err, err)
        torch.testing.assert_close(getattr(buffers, name), original, atol=1e-6, rtol=0)
    assert max_err < 1e-6


def test_a_raised_left_leg_becomes_a_raised_right_leg() -> None:
    joints = G1_29DOF_DFS_JOINT_NAMES
    spec = SymmetricAugmentationSpec.from_left_right(joints, PARKOUR_MOTION_LINKS)
    resolved = resolve_symmetric_augmentation(spec, joints, PARKOUR_MOTION_LINKS)
    jmap = torch.tensor(resolved.joint_mapping)
    jsign = torch.tensor(resolved.joint_signs, dtype=torch.float32)
    pose = torch.zeros(len(joints))
    left = joints.index("left_hip_pitch_joint")
    right = joints.index("right_hip_pitch_joint")
    pose[left] = 0.8
    mirrored = pose.clone()
    augment_joint_buffer(mirrored, jmap, jsign)
    assert float(mirrored[right]) == pytest.approx(0.8)
    assert float(mirrored[left]) == pytest.approx(0.0)
    assert float(mirrored.norm()) == pytest.approx(float(pose.norm()))

    roll = torch.zeros(len(joints))
    left_roll = joints.index("left_hip_roll_joint")
    right_roll = joints.index("right_hip_roll_joint")
    roll[left_roll] = 0.5
    mirrored_roll = roll.clone()
    augment_joint_buffer(mirrored_roll, jmap, jsign)
    assert float(mirrored_roll[right_roll]) == pytest.approx(-0.5)
    assert float(mirrored_roll[left_roll]) == pytest.approx(0.0)


def test_a_raised_left_link_becomes_a_raised_right_link() -> None:
    spec = SymmetricAugmentationSpec.from_left_right(G1_29DOF_DFS_JOINT_NAMES, PARKOUR_MOTION_LINKS)
    resolved = resolve_symmetric_augmentation(spec, G1_29DOF_DFS_JOINT_NAMES, PARKOUR_MOTION_LINKS)
    lmap = torch.tensor(resolved.link_mapping)
    pos = torch.zeros(len(PARKOUR_MOTION_LINKS), 3)
    left = PARKOUR_MOTION_LINKS.index("left_knee_link")
    right = PARKOUR_MOTION_LINKS.index("right_knee_link")
    pos[left] = torch.tensor([0.10, 0.12, 0.40])
    pos[right] = torch.tensor([0.10, -0.12, 0.05])
    mirrored = pos.clone()
    augment_link_pos_buffer(mirrored, lmap)
    assert float(mirrored[right, 2]) == pytest.approx(0.40)
    assert float(mirrored[left, 2]) == pytest.approx(0.05)
    assert float(mirrored[right, 1]) == pytest.approx(-0.12)
    assert float(mirrored[left, 1]) == pytest.approx(0.12)


def test_ang_vel_flips_x_and_z_not_y() -> None:
    """Source docstring says y and z. Source code (and a pseudovector) flip x and z."""
    buf = torch.tensor([1.0, 2.0, 3.0])
    augment_ang_vel_buffer(buf)
    assert buf.tolist() == [-1.0, 2.0, -3.0]


def test_mask_is_near_half_under_a_fixed_seed() -> None:
    torch.manual_seed(0)
    n = 4096
    mask = torch.zeros(n, dtype=torch.bool)
    draw_symmetric_mask(mask, torch.arange(n), enabled=True)
    ratio = float(mask.float().mean())
    assert 0.47 < ratio < 0.53
    # Held until the next draw of those envs: a later draw of a subset leaves the rest.
    first = mask.clone()
    draw_symmetric_mask(mask, torch.tensor([0, 1]), enabled=True)
    assert torch.equal(mask[2:], first[2:])
    draw_symmetric_mask(mask, torch.arange(n), enabled=False)
    assert not bool(mask.any())


def test_one_mask_mirrors_every_spatial_field_or_none() -> None:
    spec = SymmetricAugmentationSpec.from_left_right(G1_29DOF_DFS_JOINT_NAMES, PARKOUR_MOTION_LINKS)
    resolved = resolve_symmetric_augmentation(spec, G1_29DOF_DFS_JOINT_NAMES, PARKOUR_MOTION_LINKS)
    buffers = make_buffers(2, 1, 29, 14)
    left = G1_29DOF_DFS_JOINT_NAMES.index("left_hip_pitch_joint")
    right = G1_29DOF_DFS_JOINT_NAMES.index("right_hip_pitch_joint")
    left_link = PARKOUR_MOTION_LINKS.index("left_knee_link")
    right_link = PARKOUR_MOTION_LINKS.index("right_knee_link")
    buffers.joint_pos[:, 0, left] = 0.8
    buffers.base_pos_w[:, 0] = torch.tensor([1.0, 0.3, 0.9])
    buffers.base_ang_vel_w[:, 0] = torch.tensor([1.0, 2.0, 3.0])
    buffers.link_pos_w[:, 0, left_link] = torch.tensor([0.1, 0.12, 0.4])
    buffers.link_lin_vel_w[:, 0, left_link] = torch.tensor([0.2, 0.5, 0.1])
    original_unmasked_joints = buffers.joint_pos[1].clone()
    original_unmasked_links = buffers.link_pos_w[1].clone()
    mask = torch.tensor([True, False])
    apply_symmetric_augmentation(buffers, torch.arange(2), mask, resolved)
    assert float(buffers.joint_pos[0, 0, right]) == pytest.approx(0.8)
    assert float(buffers.joint_pos[0, 0, left]) == pytest.approx(0.0)
    assert float(buffers.base_pos_w[0, 0, 1]) == pytest.approx(-0.3)
    assert buffers.base_ang_vel_w[0, 0].tolist() == [-1.0, 2.0, -3.0]
    assert float(buffers.link_pos_w[0, 0, right_link, 2]) == pytest.approx(0.4)
    # Link velocities: sources flip y only, they do not permute. Pin that.
    assert float(buffers.link_lin_vel_w[0, 0, left_link, 1]) == pytest.approx(-0.5)
    assert float(buffers.link_lin_vel_w[0, 0, right_link, 1]) == pytest.approx(0.0)
    torch.testing.assert_close(buffers.joint_pos[1], original_unmasked_joints)
    torch.testing.assert_close(buffers.link_pos_w[1], original_unmasked_links)
    torch.testing.assert_close(buffers.base_pos_w[1, 0], torch.tensor([1.0, 0.3, 0.9]))


def _asymmetric_clip(nframes: int = 5, fps: float = 50.0) -> MotionClip:
    """Joint 0 raised, joint 1 still. A swap is visible; equal joints would hide it."""
    n_joints, n_links = 2, 1
    joint_pos = torch.zeros(nframes, n_joints)
    joint_pos[:, 0] = 0.8
    zeros3 = torch.zeros(nframes, 3)
    zeros4 = torch.zeros(nframes, 4)
    zeros4[:, 0] = 1.0
    zeros_l3 = torch.zeros(nframes, n_links, 3)
    zeros_l4 = torch.zeros(nframes, n_links, 4)
    zeros_l4[..., 0] = 1.0
    zeros3[:, 1] = 0.3
    return MotionClip(
        path="tiny",
        source_joint_names=("left_hip_pitch_joint", "right_hip_pitch_joint"),
        joint_names=("left_hip_pitch_joint", "right_hip_pitch_joint"),
        joint_index_map=(0, 1),
        link_names=("root",),
        joint_pos=joint_pos,
        joint_vel=torch.zeros_like(joint_pos),
        base_pos_w=zeros3,
        base_quat_w=zeros4,
        base_lin_vel_w=torch.zeros(nframes, 3),
        base_ang_vel_w=torch.zeros(nframes, 3),
        link_pos_b=zeros_l3,
        link_quat_b=zeros_l4,
        link_pos_w=zeros_l3,
        link_quat_w=zeros_l4,
        link_lin_vel_b=zeros_l3,
        link_ang_vel_b=zeros_l3,
        link_lin_vel_w=zeros_l3,
        link_ang_vel_w=zeros_l3,
        framerate=fps,
        inventory=ChainInventory("tiny", "root", ("left_hip_pitch_joint", "right_hip_pitch_joint"), ("root",), "urdf"),
    )


def _runtime(num_envs: int = 64, enabled: bool = True) -> MotionReferenceRuntime:
    joints = ("left_hip_pitch_joint", "right_hip_pitch_joint")
    links = ("root",)
    aug = SymmetricAugmentationSpec.from_left_right(joints, links) if enabled else None
    ref = MotionReferenceRef(
        name="motion_reference",
        clip="tiny.npz",
        joints=joints,
        links=links,
        num_frames=1,
        start_range=(0.0, 0.0),
        symmetric_augmentation=aug,
    )
    return MotionReferenceRuntime.from_clip(ref, _asymmetric_clip(), num_envs, "cpu")


def test_refresh_at_current_time_resets_the_aiming_window() -> None:
    """Isaac supplies its sensor clock; refreshing must also advance runtime bookkeeping."""
    runtime = _runtime(num_envs=3, enabled=False)
    env_ids = torch.tensor([0, 2])
    runtime.buffers.timestamp[env_ids] = 0.08

    runtime.refresh_at_current_time(env_ids)

    torch.testing.assert_close(runtime.last_update[env_ids], runtime.buffers.timestamp[env_ids])
    assert runtime.aiming_frame_idx.tolist() == [0, 0, 0]


def test_same_seed_two_runtimes_match() -> None:
    """Both engines hold this runtime. Same seed is the cross-engine contract."""
    ids = torch.arange(64)
    left = _runtime(64)
    right = _runtime(64)
    left.reset(ids, generator=torch.Generator().manual_seed(7))
    right.reset(ids, generator=torch.Generator().manual_seed(7))
    assert torch.equal(left.mask, right.mask)
    torch.testing.assert_close(left.buffers.joint_pos, right.buffers.joint_pos)
    torch.testing.assert_close(left.buffers.base_pos_w, right.buffers.base_pos_w)
    ratio = float(left.mask.float().mean())
    assert 0.30 < ratio < 0.70
    mirrored = left.mask
    assert bool(mirrored.any()) and bool((~mirrored).any())
    assert torch.allclose(left.buffers.joint_pos[mirrored, 0, 1], torch.tensor(0.8))
    assert torch.allclose(left.buffers.joint_pos[mirrored, 0, 0], torch.tensor(0.0))
    assert torch.allclose(left.buffers.joint_pos[~mirrored, 0, 0], torch.tensor(0.8))
    assert torch.allclose(left.buffers.base_pos_w[mirrored, 0, 1], torch.tensor(-0.3))
    assert torch.allclose(left.buffers.base_pos_w[~mirrored, 0, 1], torch.tensor(0.3))


def test_environment_origin_is_applied_after_mirroring() -> None:
    runtime = _runtime(2)
    ids = torch.arange(2)
    origins = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    runtime.bind_origins(origins)
    runtime.reset(ids, generator=torch.Generator().manual_seed(3))

    clip_y = torch.where(runtime.mask, -0.3, 0.3)
    torch.testing.assert_close(runtime.buffers.base_pos_w[:, 0, 1], origins[:, 1] + clip_y)
    torch.testing.assert_close(runtime.buffers.link_pos_w[:, 0, 0], origins)

    moved_origins = origins + torch.tensor([10.0, 20.0, 30.0])
    runtime.bind_origins(moved_origins)
    torch.testing.assert_close(runtime.buffers.base_pos_w[:, 0, 1], moved_origins[:, 1] + clip_y)
    torch.testing.assert_close(runtime.buffers.link_pos_w[:, 0, 0], moved_origins)


def test_a_second_refresh_does_not_double_mirror() -> None:
    runtime = _runtime(4)
    ids = torch.arange(4)
    runtime.reset(ids, generator=torch.Generator().manual_seed(1))
    first = runtime.buffers.joint_pos.clone()
    runtime.refresh(ids)
    torch.testing.assert_close(runtime.buffers.joint_pos, first)


def test_disabled_runtime_never_mirrors() -> None:
    runtime = _runtime(16, enabled=False)
    ids = torch.arange(16)
    runtime.reset(ids, generator=torch.Generator().manual_seed(0))
    assert not bool(runtime.mask.any())
    assert torch.allclose(runtime.buffers.joint_pos[:, 0, 0], torch.tensor(0.8))
    assert torch.allclose(runtime.buffers.base_pos_w[:, 0, 1], torch.tensor(0.3))


def test_both_engines_delegate_to_the_shared_runtime() -> None:
    import instinctlab.engines

    root = Path(instinctlab.engines.__file__).parent
    for name in ("isaacsim", "mjlab"):
        source = (root / name / "motion_reference.py").read_text()
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("motion_reference"):
                imported.update(alias.name for alias in node.names)
        assert "MotionReferenceRuntime" in imported, name
        assert "MotionReferenceRuntime.create" in source, name
        assert "def augment_joint_buffer" not in source
        assert "_symmetric_augment_joint_buffer" not in source
        assert "torch.randint" not in source


def test_parkour_turns_named_mirroring_on() -> None:
    from instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import parkour_target_g1

    sensor = parkour_target_g1().scene.motion_reference("motion_reference")
    spec = sensor.symmetric_augmentation
    assert spec is not None
    assert spec.joint_swaps["left_hip_pitch_joint"] == "right_hip_pitch_joint"
    assert spec.joint_swaps["waist_pitch_joint"] == "waist_pitch_joint"
    assert spec.link_swaps["left_knee_link"] == "right_knee_link"
    assert spec.link_swaps["pelvis"] == "pelvis"
    assert spec.link_swaps["torso_link"] == "torso_link"
    resolved = resolve_symmetric_augmentation(spec, sensor.joints, sensor.links)
    mapping, reverse = _instinctmj_joint_tables()
    assert list(resolved.joint_mapping) == mapping
    assert list(resolved.joint_signs) == reverse
    assert list(resolved.link_mapping) == SOURCE_LINK_MAPPING
    assert len(resolved.joint_mapping) == 29
    assert len(resolved.link_mapping) == 14


def test_locomotion_tasks_do_not_open_mirroring() -> None:
    from instinctlab.tasks.locomotion.config.g1.flat_env_cfg import flat_g1
    from instinctlab.tasks.locomotion.config.g1.rough_env_cfg import rough_g1

    assert flat_g1().scene.motion_references == ()
    assert rough_g1().scene.motion_references == ()
