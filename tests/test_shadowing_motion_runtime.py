from __future__ import annotations

import re
import torch
import yaml
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from instinctlab.engines.motion_reference import (
    ChainInventory,
    MotionClip,
    MotionInventoryEntry,
    MotionReferenceRuntime,
    discover_motion_inventory,
    estimate_velocity,
)
from instinctlab.spec import MotionReferenceRef
from instinctlab.tasks import registry


def _ref(root: str, **changes) -> MotionReferenceRef:
    values = {
        "name": "motion_reference",
        "clip": root,
        "joints": ("a",),
        "links": ("root",),
    }
    values.update(changes)
    return MotionReferenceRef(**values)


def _clip(name: str, offset: float, nframes: int = 151) -> MotionClip:
    joint_pos = torch.arange(nframes, dtype=torch.float32).reshape(-1, 1) + offset
    zeros3 = torch.zeros(nframes, 3)
    identity = torch.zeros(nframes, 4)
    identity[:, 0] = 1.0
    link3 = torch.zeros(nframes, 1, 3)
    link4 = torch.zeros(nframes, 1, 4)
    link4[..., 0] = 1.0
    return MotionClip(
        path=name,
        source_joint_names=("a",),
        joint_names=("a",),
        joint_index_map=(0,),
        link_names=("root",),
        joint_pos=joint_pos,
        joint_vel=torch.zeros_like(joint_pos),
        base_pos_w=zeros3,
        base_quat_w=identity,
        base_lin_vel_w=zeros3,
        base_ang_vel_w=zeros3,
        link_pos_b=link3,
        link_quat_b=link4,
        link_pos_w=link3,
        link_quat_w=link4,
        link_lin_vel_b=link3,
        link_ang_vel_b=link3,
        link_lin_vel_w=link3,
        link_ang_vel_w=link3,
        framerate=50.0,
        inventory=ChainInventory(name, "root", ("a",), ("root",), "urdf"),
    )


def test_metadata_inventory_preserves_declared_order_weights_and_one_motion(
    tmp_path,
) -> None:
    for name in ("z_retargeted.npz", "a_retargetted.npz"):
        (tmp_path / name).touch()
    metadata = tmp_path / "metadata.yaml"
    metadata.write_text(
        yaml.safe_dump(
            {
                "motion_files": [
                    {"motion_file": "z_retargeted.npz", "weight": 3.0, "terrain_id": 7},
                    {
                        "motion_file": "a_retargetted.npz",
                        "weight": 1.0,
                        "terrain_id": 2,
                    },
                ]
            },
            sort_keys=False,
        )
    )
    ref = _ref(str(tmp_path), dataset_kind="terrain", metadata_yaml=str(metadata))
    inventory = discover_motion_inventory(ref)

    assert [entry.path for entry in inventory] == [
        str(tmp_path / "z_retargeted.npz"),
        str(tmp_path / "a_retargetted.npz"),
    ]
    assert [entry.weight for entry in inventory] == [3.0, 1.0]
    assert [entry.terrain_id for entry in inventory] == [7, 2]
    assert discover_motion_inventory(replace(ref, first_motion_only=True)) == inventory[:1]


def test_recursive_inventory_is_deterministic_and_filters_endings(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "z_retargetted.npz").touch()
    (nested / "a_retargeted.npz").touch()
    (nested / "ignored.npz").touch()

    paths = [entry.path for entry in discover_motion_inventory(_ref(str(tmp_path)))]
    assert paths == sorted((str(nested / "a_retargeted.npz"), str(tmp_path / "z_retargetted.npz")))


def test_multiclip_reset_is_seed_stable_and_refreshes_the_selected_clip() -> None:
    ref = _ref("unused", start_range=(0.0, 0.8), motion_bin_length_s=1.0)
    clips = (_clip("first", 0.0), _clip("second", 1000.0))
    inventory = (
        MotionInventoryEntry("first", 1.0),
        MotionInventoryEntry("second", 4.0),
    )
    left = MotionReferenceRuntime.from_clips(ref, clips, inventory, 64)
    right = MotionReferenceRuntime.from_clips(ref, clips, inventory, 64)
    env_ids = torch.arange(64)
    left.reset(env_ids, torch.Generator().manual_seed(1234))
    right.reset(env_ids, torch.Generator().manual_seed(1234))

    torch.testing.assert_close(left.buffers.motion_id, right.buffers.motion_id)
    torch.testing.assert_close(left.buffers.start_s, right.buffers.start_s)
    torch.testing.assert_close(left.buffers.joint_pos, right.buffers.joint_pos)
    expected_offset = left.buffers.motion_id.to(torch.float32) * 1000.0
    assert torch.all(left.buffers.joint_pos[:, 0, 0] >= expected_offset)
    assert torch.all(left.buffers.joint_pos[:, 0, 0] < expected_offset + clips[0].nframes)
    assert [(entry.path, entry.frames, entry.fps) for entry in left.frame_inventory] == [
        ("first", 151, 50.0),
        ("second", 151, 50.0),
    ]


def test_multiclip_refresh_uses_one_packed_device_gather(monkeypatch) -> None:
    ref = _ref("unused", num_frames=3, frame_interval_s=0.02, data_start_from="current_time")
    clips = (_clip("short", 0.0, nframes=3), _clip("long", 1000.0, nframes=5))
    runtime = MotionReferenceRuntime.from_clips(
        ref,
        clips,
        (MotionInventoryEntry("short"), MotionInventoryEntry("long")),
        2,
    )
    runtime.buffers.motion_id[:] = torch.tensor([0, 1])
    runtime.buffers.start_s.zero_()
    runtime.buffers.timestamp[:] = torch.tensor([0.04, 0.08])

    def refuse_per_clip_sampling(*_args, **_kwargs):
        raise AssertionError("object-free multi-clip refresh fell back to per-clip sampling")

    def refuse_scalar_conversion(_tensor):
        raise AssertionError("packed refresh converted device state to a Python scalar")

    with monkeypatch.context() as patch:
        patch.setattr("instinctlab.engines.motion_reference.runtime.sample_clip", refuse_per_clip_sampling)
        patch.setattr(torch.Tensor, "__bool__", refuse_scalar_conversion)
        patch.setattr(torch.Tensor, "__float__", refuse_scalar_conversion)
        patch.setattr(torch.Tensor, "__int__", refuse_scalar_conversion)
        runtime.refresh(torch.arange(2))

    assert runtime.buffers.frame_index.tolist() == [[2, 2, 2], [4, 4, 4]]
    assert runtime.buffers.validity.tolist() == [[True, False, False], [True, False, False]]
    assert runtime.buffers.joint_pos[:, 0, 0].tolist() == pytest.approx([2.0, 1004.0])


def test_terrain_motion_reset_uses_only_origins_matching_the_sampled_motion() -> None:
    ref = _ref("unused", dataset_kind="terrain", metadata_yaml="unused")
    runtime = MotionReferenceRuntime.from_clips(
        ref,
        (_clip("first", 0.0), _clip("second", 1000.0)),
        (MotionInventoryEntry("first", terrain_id=0), MotionInventoryEntry("second", terrain_id=1)),
        128,
    )
    terrain = SimpleNamespace(
        terrain_origins=torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
        subterrain_specific_cfgs=[SimpleNamespace(difficulty=0.1), SimpleNamespace(difficulty=0.75)],
    )
    runtime.match_terrain_origins(terrain)
    runtime.reset(torch.arange(128), torch.Generator().manual_seed(7))

    expected_x = runtime.buffers.motion_id.to(torch.float32) + 1.0
    torch.testing.assert_close(runtime.env_origins[:, 0], expected_x)
    torch.testing.assert_close(runtime.init_buffers.base_pos_w[:, 0, 0], expected_x)
    torch.testing.assert_close(runtime.buffers.base_pos_w[:, 0, 0], expected_x)


def test_binding_origins_translates_robot_and_valid_hoi_objects_together() -> None:
    ref = _ref("unused", scene_objects=("floorlamp", "largebox"))
    clip = _clip("subject_largebox_motion", 0.0)
    clip.object_name = "largebox"
    clip.object_pos_w = torch.full((clip.nframes, 3), 2.0)
    clip.object_quat_w = torch.zeros(clip.nframes, 4)
    clip.object_quat_w[:, 0] = 1.0
    clip.object_lin_vel_w = torch.zeros(clip.nframes, 3)
    clip.object_ang_vel_w = torch.zeros(clip.nframes, 3)
    runtime = MotionReferenceRuntime.from_clip(ref, clip, 2)
    runtime.reset(torch.arange(2), torch.Generator().manual_seed(9))
    before = {
        (buffer_name, field): getattr(buffers, field).clone()
        for buffer_name, buffers in (
            ("data", runtime.buffers),
            ("init", runtime.init_buffers),
            ("reference", runtime.reference_buffers),
        )
        for field in ("base_pos_w", "link_pos_w", "object_pos_w")
    }
    origins = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])

    runtime.bind_origins(origins)

    for buffer_name, buffers in (
        ("data", runtime.buffers),
        ("init", runtime.init_buffers),
        ("reference", runtime.reference_buffers),
    ):
        torch.testing.assert_close(
            buffers.base_pos_w,
            before[(buffer_name, "base_pos_w")] + origins.unsqueeze(1),
        )
        torch.testing.assert_close(
            buffers.link_pos_w,
            before[(buffer_name, "link_pos_w")] + origins[:, None, None, :],
        )
        # The absent floorlamp slot remains its neutral zero pose.  The largebox
        # slot follows the same environment origin as the robot reference.
        torch.testing.assert_close(
            buffers.object_pos_w[:, :, 0],
            before[(buffer_name, "object_pos_w")][:, :, 0],
        )
        torch.testing.assert_close(
            buffers.object_pos_w[:, :, 1],
            before[(buffer_name, "object_pos_w")][:, :, 1] + origins.unsqueeze(1),
        )


def test_adaptive_sampling_records_smooths_and_reweights_failed_bins() -> None:
    ref = _ref("unused", motion_bin_length_s=1.0, sampling_strategy="concat_motion_bins")
    runtime = MotionReferenceRuntime.from_clips(
        ref,
        (_clip("first", 0.0), _clip("second", 1000.0)),
        (MotionInventoryEntry("first"), MotionInventoryEntry("second")),
        4,
    )
    runtime.buffers.motion_id[:] = torch.tensor([0, 0, 1, 1])
    runtime.buffers.start_s[:] = torch.tensor([0.1, 1.1, 0.1, 1.1])
    runtime.record_failures(
        torch.arange(4),
        torch.tensor([False, True, False, True]),
        torch.zeros(4),
    )
    assert runtime.current_motion_bin_fail_counter.sum() == 2
    runtime.smooth_failures(alpha=1.0)
    metrics = runtime.update_adaptive_weights()
    assert metrics is not None
    assert runtime.motion_bin_weights.sum() == pytest.approx(1.0)
    assert runtime.motion_bin_weights[1] > runtime.motion_bin_weights[0]
    second_offset = int(runtime._bin_offsets[1])
    assert runtime.motion_bin_weights[second_offset + 1] > runtime.motion_bin_weights[second_offset]


def test_adaptive_sampling_avoids_implicit_tensor_scalar_conversions(monkeypatch) -> None:
    ref = _ref("unused", motion_bin_length_s=1.0, sampling_strategy="concat_motion_bins")
    runtime = MotionReferenceRuntime.from_clip(ref, _clip("only", 0.0), 2)

    def refuse_scalar_conversion(_tensor):
        raise AssertionError("adaptive sampling used an implicit Tensor-to-Python conversion")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", refuse_scalar_conversion)
        patch.setattr(torch.Tensor, "__float__", refuse_scalar_conversion)
        patch.setattr(torch.Tensor, "__int__", refuse_scalar_conversion)
        runtime.record_failures(torch.arange(2), torch.tensor([False, False]), torch.zeros(2))
        metrics = runtime.update_adaptive_weights()

    assert metrics is not None
    assert set(metrics) == {"sampling_entropy", "sampling_top1_prob", "sampling_top1_bin"}


def test_reset_rebuilds_history_and_last_update_without_carrying_old_frames() -> None:
    ref = _ref("unused", num_frames=3, frame_interval_s=0.02, data_start_from="current_time")
    runtime = MotionReferenceRuntime.from_clip(ref, _clip("only", 0.0), 2)
    ids = torch.arange(2)
    runtime.reset(ids, torch.Generator().manual_seed(5))
    runtime.advance(0.02)
    assert runtime.last_update.tolist() == pytest.approx([0.02, 0.02])
    runtime.buffers.joint_pos[0].fill_(-99.0)

    runtime.reset(torch.tensor([0]), torch.Generator().manual_seed(6))
    assert runtime.buffers.timestamp.tolist() == pytest.approx([0.0, 0.02])
    assert runtime.last_update.tolist() == pytest.approx([0.0, 0.02])
    assert not torch.any(runtime.buffers.joint_pos[0] == -99.0)
    assert runtime.buffers.frame_index[0].tolist() == [0, 1, 2]


def test_reference_frame_is_current_time_while_data_remains_look_ahead() -> None:
    """AMP matches main's t sample without changing the t+dt aiming window."""
    ref = _ref(
        "unused",
        num_frames=3,
        frame_interval_s=0.02,
        update_period=0.02,
        data_start_from="one_frame_interval",
        start_range=(0.0, 0.0),
    )
    runtime = MotionReferenceRuntime.from_clip(ref, _clip("only", 0.0), 1)
    runtime.reset(torch.tensor([0]), torch.Generator().manual_seed(1))

    assert runtime.reference_frame.frame_index[0].tolist() == [0]
    assert runtime.buffers.frame_index[0].tolist() == [1, 2, 3]

    runtime.advance(0.02)

    assert runtime.reference_frame.frame_index[0].tolist() == [1]
    assert runtime.buffers.frame_index[0].tolist() == [2, 3, 4]


def test_reference_frame_access_does_not_convert_device_state_to_a_python_bool(monkeypatch) -> None:
    ref = _ref("unused", data_start_from="one_frame_interval", start_range=(0.0, 0.0))
    runtime = MotionReferenceRuntime.from_clip(ref, _clip("only", 0.0), 1)
    runtime.reset(torch.tensor([0]), torch.Generator().manual_seed(1))

    def refuse_tensor_bool(_tensor) -> bool:
        raise AssertionError("reference_frame converted tensor state to a Python bool")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", refuse_tensor_bool)
        current = runtime.reference_frame

    assert current.frame_index[0].tolist() == [0]


def test_reset_state_uses_floor_index_and_height_adjustment_separate_from_history() -> None:
    ref = _ref(
        "unused",
        start_range=(0.51, 0.51),
        ensure_link_below_zero_ground=True,
        motion_start_height_offset=0.1,
    )
    clip = _clip("only", 0.0)
    clip.link_pos_w[:, 0, 2] = -0.2
    runtime = MotionReferenceRuntime.from_clip(ref, clip, 1)
    runtime.reset(torch.tensor([0]), torch.Generator().manual_seed(2))

    start_frames = runtime.buffers.start_s[0] * clip.framerate
    assert runtime.init_buffers.frame_index[0, 0] == torch.floor(start_frames)
    assert runtime.buffers.frame_index[0, 0] == torch.round(start_frames + ref.frame_interval_s * clip.framerate)
    assert runtime.init_buffers.base_pos_w[0, 0, 2] == pytest.approx(0.3)


def test_reset_height_adjustment_preserves_reference_batch_gate() -> None:
    """Both sources correct the whole selected batch when any reset pose penetrates."""
    ref = _ref(
        "unused",
        ensure_link_below_zero_ground=True,
        motion_start_height_offset=0.1,
    )
    clip = _clip("only", 0.0)
    clip.link_pos_w[0, 0, 2] = -0.2
    clip.link_pos_w[1, 0, 2] = 0.3
    runtime = MotionReferenceRuntime.from_clip(ref, clip, 2)
    runtime.buffers.start_s[:] = torch.tensor([0.0, 1.0 / clip.framerate])

    runtime.refresh_initial(torch.arange(2))

    # The penetrating pose is raised by 0.2; the airborne pose is lowered by 0.3.
    # Both then receive the configured 0.1 start offset, exactly like main/InstinctMJ.
    torch.testing.assert_close(
        runtime.init_buffers.base_pos_w[:, 0, 2],
        torch.tensor([0.3, -0.2]),
    )


def test_shadowing_motion_effective_contract_matches_both_references() -> None:
    expected = {
        "WholeBody": (0.02, 10, "frontbackward", (0.0, 0.8), "independent", 1.0),
        "Perceptive-Shadowing": (
            0.1,
            10,
            "frontbackward",
            (0.0, 0.0),
            "concat_motion_bins",
            1.0,
        ),
        "Perceptive-Vae": (
            0.1,
            10,
            "frontbackward",
            (0.0, 0.0),
            "concat_motion_bins",
            1.0,
        ),
        "HOI": (0.1, 10, "frontbackward", (0.0, 0.0), "concat_motion_bins", 1.0),
        "BeyondMimic": (0.0, 1, "frontbackward", (0.0, 0.8), "independent", 1.0),
    }
    task_ids = [
        task_id
        for task_id in registry.ids()
        if ("Shadowing" in task_id or "BeyondMimic" in task_id)
        and not task_id.endswith("Play-v0")
        and "OneMotion" not in task_id
    ]
    for task_id in task_ids:
        motion = registry.spec(task_id).scene.motion_references[0]
        key = next(name for name in expected if name in task_id)
        assert (
            motion.frame_interval_s,
            motion.num_frames,
            motion.velocity_method,
            motion.start_range,
            motion.sampling_strategy,
            motion.motion_bin_length_s,
        ) == expected[key]
        exhausted = registry.spec(task_id).mdp.terminations["dataset_exhausted"]
        assert exhausted.params["sensor"] == motion
        assert "reference_cfg" not in exhausted.params


def test_shadowing_dataset_roots_keep_the_two_reference_bindings_explicit() -> None:
    expected = {
        "Instinct-Shadowing-WholeBody-Plane-G1-v0": (
            "~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single",
            "~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single",
        ),
        "Instinct-Perceptive-Shadowing-G1-v0": (
            "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
            "~/Datasets/deep_whole_body_parkour_g1_release/20251116_50cm_kneeClimbStep1",
        ),
        "Instinct-Perceptive-Vae-G1-v0": (
            "~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251116_50cm_kneeClimbStep1",
            "~/Xyk/Datasets/20260317_50cm_kneeClimbStep1_projectInstinct",
        ),
        "Instinct-Perceptive-HOI-Shadowing-G1-v0": (
            "/localhdd/Datasets/OMOMO/retargeted",
            "~/Datasets/OMOMO/retargeted_omniretarget_instinctmj_torso_v10_object_xy_align_foot_lock",
        ),
        "Instinct-BeyondMimic-Plane-G1-v0": (
            "~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz",
            "~/Xyk/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz",
        ),
    }
    for task_id, (isaac, mjlab) in expected.items():
        motion = registry.spec(task_id).scene.motion_references[0]
        assert motion.for_engine("isaacsim").clip == isaac
        assert motion.for_engine("mjlab").clip == mjlab


def test_play_and_one_motion_apply_the_reference_sampling_overrides() -> None:
    whole_play = registry.spec("Instinct-Shadowing-WholeBody-Plane-G1-Play-v0").scene.motion_references[0]
    perceptive_play = registry.spec("Instinct-Perceptive-Shadowing-G1-Play-v0").scene.motion_references[0]
    one_motion = registry.spec("Instinct-Perceptive-Shadowing-G1-OneMotion-v0").scene.motion_references[0]

    assert (
        whole_play.start_range,
        whole_play.motion_bin_length_s,
        whole_play.sampling_strategy,
    ) == (
        (0.0, 0.8),
        1.0,
        "independent",
    )
    assert (
        perceptive_play.start_range,
        perceptive_play.motion_bin_length_s,
        perceptive_play.sampling_strategy,
    ) == (
        (0.0, 0.0),
        None,
        "independent",
    )
    assert one_motion.first_motion_only is True
    assert one_motion.motion_bin_length_s is None
    assert one_motion.sampling_strategy == "independent"


def test_frontbackward_velocity_matches_reference_endpoint_semantics() -> None:
    positions = torch.tensor([[[0.0], [2.0], [6.0], [12.0]]])
    velocity = estimate_velocity(positions, 0.5, "frontbackward")
    torch.testing.assert_close(velocity, torch.tensor([[[2.0], [6.0], [10.0], [6.0]]]))


@pytest.mark.parametrize(
    ("relative", "interval", "frames"),
    [
        ("whole_body/config/g1/plane_shadowing_cfg.py", "0.02", "10"),
        ("perceptive/config/g1/perceptive_shadowing_cfg.py", "0.1", "10"),
        ("perceptive/config/g1/perceptive_vae_cfg.py", "0.1", "10"),
        ("perceptive_hoi/config/g1/perceptive_shadowing_cfg.py", "0.1", "10"),
        ("beyondmimic/config/g1/beyondmimic_plane_cfg.py", "0.0", "1"),
    ],
)
def test_both_reference_sources_declare_the_audited_motion_timing(relative: str, interval: str, frames: str) -> None:
    roots = (
        Path("/root/InstinctLab-main/source/instinctlab/instinctlab/tasks/shadowing"),
        Path("/root/InstinctMJ/src/instinct_mj/tasks/shadowing"),
    )
    for root in roots:
        source = (root / relative).read_text()
        assert re.search(
            r'velocity_estimation_method(?:\s*:\s*str)?\s*=\s*["\']frontbackward["\']',
            source,
        )
        assert f"frame_interval_s={interval}" in source
        assert f"num_frames={frames}" in source
