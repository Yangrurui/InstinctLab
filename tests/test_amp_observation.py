"""AMP branches are the same function of state. Joint order is by name.

A discriminator that can trivially separate the two branches drives the style
reward to a constant. Training still converges. These tests feed a clip through
the policy-side builder and demand the reference branch's vector back.

Joint order is guarded by name, not by engine-vs-engine equality: a shared
wrong order stays green under a cross-engine compare.
"""

from __future__ import annotations

import math
import torch
from types import SimpleNamespace

import pytest

from instinctlab.assets.unitree_g1.isaacsim import (
    G1_29DOF_DEFAULT_JOINT_POS,
    G1_29DOF_DFS_JOINT_NAMES,
    G1_29DOF_ISAAC_BFS_JOINT_NAMES,
)
from instinctlab_engine.bridge.math import quat_apply_inverse
from instinctlab_engine.motion_reference import clip_frame, make_buffers
from instinctlab.tasks.parkour.mdp.amp import (
    AMP_TERM_ORDER,
    GRAVITY_DOWN_W,
    amp_obs_from_reference,
    amp_obs_from_robot_like,
    robot_like_from_clip,
)
from instinctlab.tasks.parkour.config.g1.agents.instinct_rl_amp_cfg import (
    G1ParkourTargetPPORunnerCfg,
)

# Elementwise tolerance for the same-function check. Synthetic float32, same ops.
SAME_FUNCTION_ATOL = 1e-6
SAME_FUNCTION_RTOL = 0.0


def _catalog_defaults() -> torch.Tensor:
    return torch.tensor(
        [G1_29DOF_DEFAULT_JOINT_POS[name] for name in G1_29DOF_DFS_JOINT_NAMES]
    )


def _tilt_quat() -> torch.Tensor:
    """90° about X so body-frame quantities are not a copy of world ones."""
    half = 0.5 * math.pi / 2.0
    return torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]])


def _clip_tensors():
    n_joints = len(G1_29DOF_DFS_JOINT_NAMES)
    quat = _tilt_quat()
    lin_w = torch.tensor([[1.25, -0.4, 0.3]])
    ang_w = torch.tensor([[0.1, -0.2, 0.5]])
    joint_pos = _catalog_defaults().unsqueeze(0) + torch.linspace(
        -0.2, 0.2, n_joints
    ).unsqueeze(0)
    joint_vel = torch.linspace(-1.0, 1.0, n_joints).unsqueeze(0)
    defaults = _catalog_defaults().unsqueeze(0)
    default_vel = torch.zeros_like(defaults)
    gravity = torch.tensor([GRAVITY_DOWN_W])
    return quat, lin_w, ang_w, joint_pos, joint_vel, defaults, default_vel, gravity


def _buffers_from_clip(quat, lin_w, ang_w, joint_pos, joint_vel):
    buffers = make_buffers(1, 10, joint_pos.shape[-1], 1, device="cpu")
    buffers.base_quat_w[:, 0] = quat
    buffers.base_lin_vel_w[:, 0] = lin_w
    buffers.base_ang_vel_w[:, 0] = ang_w
    buffers.joint_pos[:, 0] = joint_pos
    buffers.joint_vel[:, 0] = joint_vel
    return buffers


def test_amp_term_order_is_the_legacy_order() -> None:
    assert AMP_TERM_ORDER == (
        "projected_gravity",
        "joint_pos_rel",
        "joint_vel",
        "base_lin_vel",
        "base_ang_vel",
    )


def test_policy_builder_on_clip_state_matches_reference_branch() -> None:
    """Feed the clip through the policy-side builder; get the reference vector back."""
    quat, lin_w, ang_w, joint_pos, joint_vel, defaults, default_vel, gravity = (
        _clip_tensors()
    )
    like = robot_like_from_clip(
        quat, lin_w, ang_w, joint_pos, joint_vel, defaults, default_vel, gravity
    )
    policy = amp_obs_from_robot_like(like)
    reference = amp_obs_from_reference(
        _buffers_from_clip(quat, lin_w, ang_w, joint_pos, joint_vel),
        defaults,
        default_vel,
        gravity,
    )
    for name in AMP_TERM_ORDER:
        torch.testing.assert_close(
            policy[name],
            reference[name],
            atol=SAME_FUNCTION_ATOL,
            rtol=SAME_FUNCTION_RTOL,
            msg=name,
        )
    # The conversion is not a copy: a 90° roll moves gravity off −Z and body vel off world.
    assert not torch.allclose(policy["projected_gravity"], gravity, atol=1e-5)
    assert not torch.allclose(policy["base_lin_vel"], lin_w, atol=1e-5)
    torch.testing.assert_close(
        policy["projected_gravity"],
        quat_apply_inverse(quat, gravity),
        atol=SAME_FUNCTION_ATOL,
        rtol=SAME_FUNCTION_RTOL,
    )


def test_same_function_check_is_tight_enough_to_see_a_frame_swap() -> None:
    """If the reference branch forgot to rotate world velocity, this must go red."""
    quat, lin_w, ang_w, joint_pos, joint_vel, defaults, default_vel, gravity = (
        _clip_tensors()
    )
    like = robot_like_from_clip(
        quat, lin_w, ang_w, joint_pos, joint_vel, defaults, default_vel, gravity
    )
    policy = amp_obs_from_robot_like(like)
    broken = dict(policy)
    broken["base_lin_vel"] = lin_w
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            policy["base_lin_vel"],
            broken["base_lin_vel"],
            atol=SAME_FUNCTION_ATOL,
            rtol=SAME_FUNCTION_RTOL,
        )


def test_amp_joints_are_canonical_names_not_bfs_or_clip_order() -> None:
    """waist_pitch is index 0 in DFS and 14 in the published clip. A positional
    remap would put the clip's left_hip at canonical index 0.
    """
    import os

    from instinctlab_engine.motion_reference import (
        load_retargetted_clip,
        remap_by_name,
    )
    from instinctlab.engines.mjlab.assets import robot_spec
    from instinctlab.tasks import registry

    task_id = "Instinct-Parkour-Target-G1"
    task = registry.spec(task_id, robot_spec(registry.asset_id(task_id)))
    parkour_motion_clip = task.scene.motion_references[0].clip

    path = os.path.expanduser(parkour_motion_clip)
    if not os.path.isfile(os.path.realpath(path)):
        pytest.skip(f"parkour clip is not at {path}")
    raw = load_retargetted_clip(path, device="cpu")
    source = tuple(raw["joint_names"])
    canonical = G1_29DOF_DFS_JOINT_NAMES
    bfs = G1_29DOF_ISAAC_BFS_JOINT_NAMES
    assert source != canonical
    assert bfs != canonical
    remapped, index_map = remap_by_name(
        raw["joint_pos"][:1], source, canonical, what="joint"
    )
    waist_src = source.index("waist_pitch_joint")
    waist_dst = canonical.index("waist_pitch_joint")
    hip_src = source.index("left_hip_pitch_joint")
    assert waist_src != waist_dst
    assert index_map[waist_dst] == waist_src
    assert remapped[0, waist_dst].item() == pytest.approx(
        raw["joint_pos"][0, waist_src].item()
    )
    assert remapped[0, 0].item() != pytest.approx(raw["joint_pos"][0, 0].item())
    # The value at canonical 0 is waist, not the clip's legs-first hip.
    assert canonical[0] == "waist_pitch_joint"
    assert source[0] == "left_hip_pitch_joint"
    assert hip_src == 0


def test_amp_joint_rel_indexes_waist_pitch_by_name() -> None:
    """A length-only gather would subtract the wrong default from the wrong joint."""
    n = len(G1_29DOF_DFS_JOINT_NAMES)
    defaults = _catalog_defaults().unsqueeze(0)
    joint_pos = defaults.clone()
    waist = G1_29DOF_DFS_JOINT_NAMES.index("waist_pitch_joint")
    hip = G1_29DOF_DFS_JOINT_NAMES.index("left_hip_pitch_joint")
    assert waist == 0
    assert hip != 0
    joint_pos[0, waist] = defaults[0, waist] + 0.42
    like = SimpleNamespace(
        projected_gravity_b=torch.zeros(1, 3),
        joint_pos=joint_pos,
        joint_vel=torch.zeros(1, n),
        default_joint_pos=defaults,
        default_joint_vel=torch.zeros(1, n),
        root_link_lin_vel_b=torch.zeros(1, 3),
        root_link_ang_vel_b=torch.zeros(1, 3),
    )
    rel = amp_obs_from_robot_like(like)["joint_pos_rel"]
    assert rel[0, waist].item() == pytest.approx(0.42)
    assert rel[0, hip].item() == pytest.approx(0.0)
    # Positional clip order would put left_hip at 0. That must not look like waist.
    legs_first = torch.zeros(1, n)
    legs_first[0, 0] = 0.42
    like.joint_pos = defaults + legs_first
    wrong = amp_obs_from_robot_like(like)["joint_pos_rel"]
    assert wrong[0, 0].item() == pytest.approx(0.42)
    assert G1_29DOF_DFS_JOINT_NAMES[0] != "left_hip_pitch_joint"


def test_depth_encoder_component_names_resolve_against_the_declared_term() -> None:
    """A misspelled component is a KeyError at policy init, not a silent flatten."""
    from instinct_rl.modules.parallel_layer import ParallelLayer
    from instinct_rl.utils.utils import get_subobs_size

    agent = G1ParkourTargetPPORunnerCfg()
    dumped = agent.to_dict()
    encoder = dumped["policy"]["encoder_configs"]
    assert encoder["depth_encoder"]["component_names"] == ["depth_image"]
    assert encoder["depth_encoder"]["takeout_input_components"] is True

    proprio = {
        "base_ang_vel": (24,),
        "projected_gravity": (24,),
        "velocity_commands": (24,),
        "joint_pos": (232,),
        "joint_vel": (232,),
        "actions": (232,),
        "depth_image": (8, 18, 32),
    }
    layer = ParallelLayer(proprio, encoder)
    assert "depth_image" not in layer.output_segment
    assert "parallel_latent_0_depth_encoder" in layer.output_segment
    assert layer.output_segment["parallel_latent_0_depth_encoder"] == (128,)
    actor_dim = get_subobs_size(layer.output_segment)
    assert actor_dim == 768 + 128
    first = next(
        m
        for m in layer._parallel_blocks["depth_encoder"].modules()
        if isinstance(m, torch.nn.Conv2d)
    )
    assert first.in_channels == 8

    critic = {
        "base_lin_vel": (24,),
        **proprio,
    }
    critic_layer = ParallelLayer(critic, dumped["policy"]["critic_encoder_configs"])
    assert get_subobs_size(critic_layer.output_segment) == 792 + 128

    broken = {
        **encoder,
        "depth_encoder": {
            **encoder["depth_encoder"],
            "component_names": ["not_a_term"],
        },
    }
    with pytest.raises(KeyError):
        ParallelLayer(proprio, broken)


def test_depth_encoder_output_moves_when_the_image_moves() -> None:
    """An encoder that is constructed but never fed would ignore this perturbation."""
    from instinct_rl.modules.parallel_layer import ParallelLayer
    from instinct_rl.utils.utils import get_obs_slice, get_subobs_size

    proprio_width = 768
    depth = torch.rand(2, 8, 18, 32)
    flat = torch.cat([torch.randn(2, proprio_width), depth.flatten(1)], dim=1)
    segments = {
        "base_ang_vel": (24,),
        "projected_gravity": (24,),
        "velocity_commands": (24,),
        "joint_pos": (232,),
        "joint_vel": (232,),
        "actions": (232,),
        "depth_image": (8, 18, 32),
    }
    assert get_subobs_size(segments) == proprio_width + 8 * 18 * 32
    layer = ParallelLayer(
        segments, G1ParkourTargetPPORunnerCfg().to_dict()["policy"]["encoder_configs"]
    )
    layer.eval()
    with torch.no_grad():
        out = layer(flat)
        sl, _ = get_obs_slice(segments, "depth_image")
        zeroed = flat.clone()
        zeroed[:, sl] = 0
        out_zero = layer(zeroed)
        proprio_only = flat.clone()
        proprio_only[:, :proprio_width] = 0
        out_no_proprio = layer(proprio_only)
    assert out.shape[-1] == 768 + 128
    assert not torch.allclose(out, out_zero, atol=1e-5)
    # Depth still reaches the encoder when proprio is wiped; the latent slot moves.
    latent_slice = slice(proprio_width, proprio_width + 128)
    assert not torch.allclose(
        out[:, latent_slice], out_no_proprio[:, latent_slice] * 0, atol=1e-5
    )


def test_clip_frame_reads_look_ahead_zero() -> None:
    quat, lin_w, ang_w, joint_pos, joint_vel, *_ = _clip_tensors()
    buffers = _buffers_from_clip(quat, lin_w, ang_w, joint_pos, joint_vel)
    buffers.joint_pos[:, 1] = joint_pos + 3.0
    got_q, got_lin, got_ang, got_pos, got_vel = clip_frame(buffers, 0)
    torch.testing.assert_close(got_q, quat)
    torch.testing.assert_close(got_lin, lin_w)
    torch.testing.assert_close(got_ang, ang_w)
    torch.testing.assert_close(got_pos, joint_pos)
    torch.testing.assert_close(got_vel, joint_vel)
    later = clip_frame(buffers, 1)[3]
    assert not torch.allclose(later, joint_pos)
