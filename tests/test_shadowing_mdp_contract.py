"""Value-level guards for the final shared shadowing MDP and sensor layout."""

from __future__ import annotations

import inspect
import re
import sys
import torch
import trimesh
import types
import yaml
from types import SimpleNamespace

import mujoco
import pytest

from instinctlab.compat import math as math_utils
from instinctlab.engines.isaacsim import terms as isaac_terms
from instinctlab.engines.isaacsim.adapter import IsaacSimAdapter
from instinctlab.engines.mjlab import terms as mjlab_terms
from instinctlab.engines.mjlab.adapter import MjlabAdapter
from instinctlab.engines.mjlab.shadowing import randomize_default_joint_pos
from instinctlab.engines.motion_reference.buffers import fill_buffers, make_buffers
from instinctlab.engines.motion_reference.clip import MotionSample
from instinctlab.engines.shadowing_commands import _root
from instinctlab.mdp import shadowing as shadowing_mdp
from instinctlab.tasks import registry
from instinctlab.tasks.shadowing.whole_body.config.g1.plane_shadowing_cfg import MOTION_LINKS

SHADOW_IDS = tuple(
    task_id for task_id in registry.ids() if any(token in task_id for token in ("Shadowing", "Mimic", "Vae"))
)


def test_every_shadowing_term_has_a_strict_native_lowering() -> None:
    for task_id in SHADOW_IDS:
        task = registry.spec(task_id)
        for adapter in (IsaacSimAdapter(), MjlabAdapter()):
            assert adapter.contract_report(task)["missing"] == {}


def test_shadowing_termination_signatures_are_native_manager_compatible() -> None:
    for func in (
        shadowing_mdp.base_position_too_far,
        shadowing_mdp.projected_gravity_too_far,
        shadowing_mdp.link_position_too_far,
    ):
        parameters = tuple(inspect.signature(func).parameters.values())
        assert all(parameter.kind is not inspect.Parameter.VAR_KEYWORD for parameter in parameters)
        assert all(parameter.default is not inspect.Parameter.empty for parameter in parameters[1:])


def test_shadowing_train_and_play_episode_boundaries_match_both_references() -> None:
    """Keep evaluation-only reset changes out of the corresponding training task."""
    expected_terms = {
        "Instinct-Shadowing-WholeBody-Plane-G1-v0": {
            "time_out",
            "base_pos_too_far",
            "base_pg_too_far",
            "link_pos_too_far",
            "dataset_exhausted",
            "out_of_border",
        },
        "Instinct-Shadowing-WholeBody-Plane-G1-Play-v0": {
            "time_out",
            "base_pos_too_far",
            "base_pg_too_far",
            "link_pos_too_far",
            "dataset_exhausted",
            "out_of_border",
        },
        "Instinct-BeyondMimic-Plane-G1-v0": {
            "time_out",
            "base_pos_too_far",
            "base_pg_too_far",
            "link_pos_too_far",
            "dataset_exhausted",
            "out_of_border",
        },
        "Instinct-BeyondMimic-Plane-G1-Play-v0": {
            "time_out",
            "base_pos_too_far",
            "base_pg_too_far",
            "link_pos_too_far",
            "dataset_exhausted",
            "out_of_border",
        },
        "Instinct-Perceptive-Shadowing-G1-v0": {
            "time_out",
            "illegal_reset_contact",
            "base_pos_too_far",
            "base_pg_too_far",
            "link_pos_too_far",
            "dataset_exhausted",
            "out_of_border",
        },
        "Instinct-Perceptive-Shadowing-G1-Play-v0": {
            "time_out",
            "illegal_reset_contact",
            "dataset_exhausted",
            "out_of_border",
        },
        "Instinct-Perceptive-Shadowing-G1-OneMotion-v0": {
            "time_out",
            "illegal_reset_contact",
            "base_pos_too_far",
            "base_pg_too_far",
            "link_pos_too_far",
            "dataset_exhausted",
            "out_of_border",
        },
        "Instinct-Perceptive-Shadowing-G1-OneMotion-Play-v0": {
            "time_out",
            "illegal_reset_contact",
            "dataset_exhausted",
            "out_of_border",
        },
        "Instinct-Perceptive-Vae-G1-v0": {
            "time_out",
            "illegal_reset_contact",
            "base_pos_too_far",
            "base_pg_too_far",
            "link_pos_too_far",
            "dataset_exhausted",
            "out_of_border",
        },
        "Instinct-Perceptive-Vae-G1-Play-v0": {
            "time_out",
            "illegal_reset_contact",
            "dataset_exhausted",
            "out_of_border",
        },
        "Instinct-Perceptive-HOI-Shadowing-G1-v0": {
            "time_out",
            "illegal_reset_contact",
            "base_pos_too_far",
            "base_pg_too_far",
            "link_pos_too_far",
            "dataset_exhausted",
        },
        "Instinct-Perceptive-HOI-Shadowing-G1-Play-v0": {
            "time_out",
            "illegal_reset_contact",
            "dataset_exhausted",
        },
    }

    for task_id, terms in expected_terms.items():
        task = registry.spec(task_id)
        assert set(task.mdp.terminations) == terms, task_id
        assert task.sim.is_finite_horizon is False

    for task_id in expected_terms:
        expected_length = 6000.0 if task_id == "Instinct-BeyondMimic-Plane-G1-Play-v0" else 10.0
        assert registry.spec(task_id).sim.episode_length_s == expected_length, task_id


def test_shadowing_timeout_classification_is_declared_at_the_task_boundary() -> None:
    """The runner bootstraps only terms declared as timeouts; builders must not invent the flag."""
    timeout_names = {"time_out", "illegal_reset_contact", "dataset_exhausted", "out_of_border"}
    for task_id in SHADOW_IDS:
        for name, term in registry.spec(task_id).mdp.terminations.items():
            assert term.time_out is (name in timeout_names), (task_id, name)


def test_vae_hides_dataset_exhaustion_only_in_play() -> None:
    train = registry.spec("Instinct-Perceptive-Vae-G1-v0").mdp.terminations["dataset_exhausted"]
    play = registry.spec("Instinct-Perceptive-Vae-G1-Play-v0").mdp.terminations["dataset_exhausted"]
    assert train.params["reset_without_notice"] is False
    assert play.params["reset_without_notice"] is True


def test_illegal_reset_lowerings_preserve_the_declared_timeout_flag(monkeypatch) -> None:
    """A future task may classify this term differently; lowering is not task policy."""

    class NativeDone:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    monkeypatch.setattr(isaac_terms, "_import_cfgs", lambda: {"done": NativeDone})
    monkeypatch.setattr(isaac_terms, "_as_isaac_manager_term", lambda cls: cls)
    monkeypatch.setattr(mjlab_terms, "_cfgs", lambda: {"done": NativeDone})
    spec = SimpleNamespace(kind="shadow_illegal_reset_contact", time_out=False, params={})
    ctx = SimpleNamespace(params=lambda _spec: {})

    for terms in (isaac_terms, mjlab_terms):
        native = terms.TERMS.lookup("termination", "shadow_illegal_reset_contact")(spec, ctx)
        assert native.time_out is False


def test_mjlab_default_joint_randomization_honors_partial_slices_and_list_env_ids() -> None:
    default = torch.zeros(2, 4)
    offset = torch.zeros_like(default)
    asset = SimpleNamespace(data=SimpleNamespace(default_joint_pos=default))
    action = SimpleNamespace(_offset=offset)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene={"robot": asset},
        action_manager=SimpleNamespace(get_term=lambda name: action),
    )

    randomize_default_joint_pos(
        env,
        env_ids=[1],
        asset_cfg=SimpleNamespace(name="robot", joint_ids=slice(1, 3)),
        offset_distribution_params=(0.5, 0.5),
    )

    torch.testing.assert_close(default, torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]]))
    torch.testing.assert_close(offset, default)


def test_shadowing_root_accessor_prefers_explicit_link_names_and_fails_loudly() -> None:
    explicit = torch.tensor([1.0])
    legacy = torch.tensor([2.0])
    assert _root(SimpleNamespace(root_link_pos_w=explicit, root_pos_w=legacy), "root_pos_w") is explicit
    assert _root(SimpleNamespace(root_pos_w=legacy), "root_pos_w") is legacy
    with pytest.raises(AttributeError, match="root_link_pos_w"):
        _root(SimpleNamespace(), "root_pos_w")


def test_shadowing_link_terms_read_link_frames_not_com_aliases() -> None:
    """main and InstinctMJ track link origins; Isaac's legacy velocity alias is COM."""
    zeros3 = torch.zeros(1, 2, 3)
    zeros4 = torch.zeros(1, 2, 4)
    zeros4[..., 0] = 1.0
    data = SimpleNamespace(
        body_link_pos_w=zeros3,
        body_pos_w=torch.full_like(zeros3, 7.0),
        body_link_quat_w=zeros4,
        body_quat_w=torch.full_like(zeros4, 7.0),
        body_link_lin_vel_w=zeros3,
        body_lin_vel_w=torch.full_like(zeros3, 7.0),
        body_link_ang_vel_w=zeros3,
        body_ang_vel_w=torch.full_like(zeros3, 7.0),
    )
    reference = SimpleNamespace(
        data=SimpleNamespace(
            link_lin_vel_w=zeros3.unsqueeze(1),
            link_ang_vel_w=zeros3.unsqueeze(1),
        ),
        reference_frame=SimpleNamespace(
            link_lin_vel_w=zeros3.unsqueeze(1),
            link_ang_vel_w=zeros3.unsqueeze(1),
        ),
    )
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data), "motion_reference": reference})
    asset_cfg = SimpleNamespace(name="robot", body_ids=slice(None))

    torch.testing.assert_close(shadowing_mdp.link_position(env, asset_cfg, in_base_frame=False), zeros3)
    torch.testing.assert_close(
        shadowing_mdp.link_rotation(env, asset_cfg, in_base_frame=False),
        shadowing_mdp.quat_to_tan_norm(zeros4),
    )
    torch.testing.assert_close(
        shadowing_mdp.link_linear_velocity_imitation(env, asset_cfg=asset_cfg), torch.ones(1)
    )
    torch.testing.assert_close(
        shadowing_mdp.link_angular_velocity_imitation(env, asset_cfg=asset_cfg), torch.ones(1)
    )


def test_shadowing_depth_clamps_before_resizing_like_both_references(monkeypatch) -> None:
    from instinctlab.compat import sensors as compat_sensors

    raw = torch.tensor([[[[0.0], [4.0], [0.0], [4.0]]]])
    sensor_ref = SimpleNamespace(name="camera", crop=None)
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"camera": object()}))
    monkeypatch.setattr(compat_sensors, "depth_image", lambda _sensor: raw)

    processed = shadowing_mdp.depth_image(env, sensor_ref, resize_shape=(1, 2), normalization_range=(0.0, 2.0))

    torch.testing.assert_close(processed, torch.full((1, 1, 1, 2), 0.5))


def test_shadowing_depth_image_accepts_play_debug_vis(monkeypatch) -> None:
    from instinctlab.compat import sensors as compat_sensors
    from instinctlab.mdp.observations import set_debug_image_sink

    raw = torch.tensor([[[[0.0], [2.0], [0.0], [2.0]]]])
    sensor_ref = SimpleNamespace(name="camera", crop=None)
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"camera": object()}))
    monkeypatch.setattr(compat_sensors, "depth_image", lambda _sensor: raw)
    captured: dict[str, object] = {}

    def sink(window_name: str, frames) -> None:
        captured["window"] = window_name
        captured["frames"] = frames

    set_debug_image_sink(sink)
    try:
        processed = shadowing_mdp.depth_image(
            env, sensor_ref, resize_shape=(1, 2), normalization_range=(0.0, 2.0), debug_vis=True
        )
    finally:
        set_debug_image_sink(None)

    torch.testing.assert_close(processed, torch.full((1, 1, 1, 2), 0.5))
    assert captured["window"] == "depth_image"
    assert captured["frames"].shape == (1, 1, 2)


def test_shadowing_imitation_rewards_use_current_reference_frame_not_lookahead_data() -> None:
    """main evaluates imitation at t; sensor data starts at t + dt for commands."""
    zeros3 = torch.zeros(1, 2, 3)
    zeros4 = torch.zeros(1, 2, 4)
    zeros4[..., 0] = 1.0
    current = SimpleNamespace(
        base_pos_w=torch.zeros(1, 1, 3),
        base_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        link_pos_w=zeros3.unsqueeze(1),
        link_pos_b=zeros3.unsqueeze(1),
        link_quat_w=zeros4.unsqueeze(1),
        link_quat_b=zeros4.unsqueeze(1),
        link_lin_vel_w=zeros3.unsqueeze(1),
        link_ang_vel_w=zeros3.unsqueeze(1),
        validity=torch.ones(1, 1),
    )
    lookahead = SimpleNamespace(
        **{
            name: torch.full_like(value, 7.0)
            for name, value in vars(current).items()
            if name != "validity"
        },
        validity=torch.ones(1, 1),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.zeros(1, 3),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            body_link_pos_w=zeros3,
            body_link_quat_w=zeros4,
            body_link_lin_vel_w=zeros3,
            body_link_ang_vel_w=zeros3,
        )
    )
    reference = SimpleNamespace(data=lookahead, reference_frame=current)
    env = SimpleNamespace(scene={"robot": robot, "motion_reference": reference})

    for reward in (
        shadowing_mdp.base_position_imitation(env),
        shadowing_mdp.base_rotation_imitation(env),
        shadowing_mdp.link_position_imitation(env),
        shadowing_mdp.link_rotation_imitation(env),
        shadowing_mdp.link_linear_velocity_imitation(env),
        shadowing_mdp.link_angular_velocity_imitation(env),
    ):
        torch.testing.assert_close(reward, torch.ones(1))


def test_shadowing_relative_world_imitation_matches_reference_fixed_state() -> None:
    """Pin main/InstinctMJ's XY-height anchor, yaw correction, and mean-product reduction."""
    half_sqrt = 2.0**-0.5
    ref_base_quat = torch.tensor([[[half_sqrt, 0.0, 0.0, half_sqrt]]])
    robot_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    ref_link_quat = torch.tensor(
        [[[[half_sqrt, half_sqrt, 0.0, 0.0], [half_sqrt, 0.0, half_sqrt, 0.0]]]]
    )
    reference_frame = SimpleNamespace(
        base_pos_w=torch.tensor([[[10.0, 20.0, 1.0]]]),
        base_quat_w=ref_base_quat,
        link_pos_w=torch.tensor([[[[11.0, 20.0, 1.0], [10.0, 22.0, 3.0]]]]),
        link_quat_w=ref_link_quat,
    )
    expected_pos = torch.tensor([[[3.0, 5.0, 1.0], [1.0, 4.0, 3.0]]])
    expected_quat = torch.tensor([[[0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, 0.5]]])

    position_error = torch.tensor([[[0.3, 0.0, 0.0], [0.0, 0.6, 0.0]]])
    rotation_error = math_utils.quat_from_angle_axis(
        torch.tensor([0.0, 0.4]),
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    ).unsqueeze(0)
    actual_quat = math_utils.quat_mul(rotation_error, expected_quat)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            # Explicit root-link fields must win over legacy aliases on MJLab.
            root_link_pos_w=torch.tensor([[3.0, 4.0, 9.0]]),
            root_pos_w=torch.full((1, 3), -100.0),
            root_link_quat_w=robot_quat,
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            body_link_pos_w=expected_pos + position_error,
            body_link_quat_w=actual_quat,
        )
    )
    env = SimpleNamespace(
        scene={"robot": robot, "motion_reference": SimpleNamespace(reference_frame=reference_frame)}
    )
    asset_cfg = SimpleNamespace(name="robot", body_ids=slice(None))

    relative_pos, relative_quat = shadowing_mdp._relative_reference(robot, reference_frame)
    torch.testing.assert_close(relative_pos, expected_pos)
    torch.testing.assert_close(relative_quat, expected_quat)

    position_reward = shadowing_mdp.link_position_imitation(env, asset_cfg=asset_cfg, std=0.3)
    rotation_reward = shadowing_mdp.link_rotation_imitation(env, asset_cfg=asset_cfg, std=0.4)
    torch.testing.assert_close(position_reward, torch.exp(torch.tensor([-2.5])))
    torch.testing.assert_close(rotation_reward, torch.exp(torch.tensor([-0.5])))


def test_shadowing_base_rotation_imitation_matches_reference_fixed_angle() -> None:
    half_sqrt = 2.0**-0.5
    reference = SimpleNamespace(
        reference_frame=SimpleNamespace(
            base_quat_w=torch.tensor([[[half_sqrt, 0.0, 0.0, half_sqrt]]])
        )
    )
    robot = SimpleNamespace(data=SimpleNamespace(root_link_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]])))
    env = SimpleNamespace(scene={"robot": robot, "motion_reference": reference})

    reward = shadowing_mdp.base_rotation_imitation(env, std=torch.pi / 2)

    torch.testing.assert_close(reward, torch.exp(torch.tensor([-1.0])))


def test_shadowing_failure_terms_match_reference_fixed_state() -> None:
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    roll_90 = torch.tensor([2.0**-0.5, 2.0**-0.5, 0.0, 0.0])
    body_pos = torch.zeros(2, 4, 3)
    body_pos[0, 0, 2] = 10.0  # An unselected link must not end the episode.
    body_pos[1, 3, 2] = 0.3
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[100.0, 100.0, 0.2], [0.0, 0.0, 0.3]]),
            root_link_quat_w=torch.stack((identity, roll_90)),
            gravity_vec_w=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
            body_link_pos_w=body_pos,
        )
    )
    reference_data = SimpleNamespace(
        base_pos_w=torch.zeros(2, 1, 3),
        base_quat_w=identity.repeat(2, 1).unsqueeze(1),
        link_pos_w=torch.zeros(2, 1, 4, 3),
        link_pos_b=torch.zeros(2, 1, 4, 3),
    )
    env = SimpleNamespace(scene={"robot": robot, "motion_reference": SimpleNamespace(data=reference_data)})
    asset_cfg = SimpleNamespace(name="robot", body_ids=slice(None))

    assert shadowing_mdp.base_position_too_far(env, height_only=True).tolist() == [False, True]
    assert shadowing_mdp.base_position_too_far(env, height_only=False).tolist() == [True, True]
    assert shadowing_mdp.projected_gravity_too_far(env).tolist() == [False, True]
    assert shadowing_mdp.link_position_too_far(
        env,
        asset_cfg=asset_cfg,
        link_ids=(1, 3),
    ).tolist() == [False, True]


def test_shadowing_illegal_reset_contact_matches_reference_temporal_gate(monkeypatch) -> None:
    from instinctlab.compat import sensors as compat_sensors

    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        episode_length_buf=torch.tensor([1, 1]),
        scene=SimpleNamespace(sensors={"reset_contacts": object()}),
    )
    term = shadowing_mdp.IllegalResetContact(SimpleNamespace(), env)
    sensor = SimpleNamespace(name="reset_contacts")
    history = torch.zeros(2, 3, 2, 3)
    history[:, 1, 0, 0] = 501.0
    monkeypatch.setattr(compat_sensors, "contact_force_history", lambda _sensor, _ref: history)

    assert term(env, sensor, threshold=500.0, episode_length_threshold=2).tolist() == [False, False]
    env.episode_length_buf[:] = torch.tensor([2, 3])
    assert term(env, sensor, threshold=500.0, episode_length_threshold=2).tolist() == [True, False]

    term.reset([0])
    assert term.counter.tolist() == [0, 2]
    assert term(env, sensor, threshold=500.0, episode_length_threshold=2).tolist() == [False, False]


def test_reference_observation_anchor_noise_and_history_match_effective_sources() -> None:
    whole = registry.spec("Instinct-Shadowing-WholeBody-Plane-G1-v0")
    policy = whole.mdp.observations["policy"].terms
    critic = whole.mdp.observations["critic"].terms
    assert policy["position_ref"].params["command_name"] == "position_b_ref_command"
    assert critic["position_ref"].params["command_name"] == "position_ref_command"
    assert (policy["position_ref"].noise.lo, policy["position_ref"].noise.hi) == (
        -0.25,
        0.25,
    )
    assert (policy["rotation_ref"].noise.lo, policy["rotation_ref"].noise.hi) == (
        -0.05,
        0.05,
    )
    assert {term.history_length for term in policy.values()} == {0}
    assert "projected_gravity" in policy
    assert "projected_gravity" not in critic

    for task_id in (
        "Instinct-Perceptive-Shadowing-G1-v0",
        "Instinct-Perceptive-Vae-G1-v0",
        "Instinct-Perceptive-HOI-Shadowing-G1-v0",
    ):
        task = registry.spec(task_id)
        assert "projected_gravity" in task.mdp.observations["policy"].terms
        assert "projected_gravity" not in task.mdp.observations["critic"].terms
        for group in task.mdp.observations.values():
            proprio = {
                name: term
                for name, term in group.terms.items()
                if name
                in {
                    "projected_gravity",
                    "base_ang_vel",
                    "base_lin_vel",
                    "joint_pos",
                    "joint_vel",
                    "last_action",
                }
            }
            assert proprio and {term.history_length for term in proprio.values()} == {8}


def test_perceptive_motion_height_preprocessing_matches_each_reference_engine() -> None:
    motion = registry.spec("Instinct-Perceptive-Shadowing-G1-v0").scene.motion_references[0]
    isaac = motion.for_engine("isaacsim")
    mjlab = motion.for_engine("mjlab")
    assert (isaac.ensure_link_below_zero_ground, isaac.motion_start_height_offset) == (False, 0.0)
    assert (mjlab.ensure_link_below_zero_ground, mjlab.motion_start_height_offset) == (True, 0.1)


def test_mjlab_perceptive_compilation_preserves_base_linear_velocity_history() -> None:
    """A semantic lowering must retain the observation metadata carried by TaskSpec."""
    task = registry.spec("Instinct-Perceptive-Shadowing-G1-v0")
    compiled = MjlabAdapter().compile(task, num_envs=1, device="cpu", strict=True)
    declared = task.mdp.observations["critic"].terms["base_lin_vel"]
    native = compiled.env_cfg.observations["critic"].terms["base_lin_vel"]
    assert declared.history_length == 8
    assert native.history_length == declared.history_length


def test_isaac_shadow_base_linear_velocity_lowering_preserves_observation_metadata(monkeypatch) -> None:
    """Isaac needs an app for its real imports, so isolate the lowering boundary here."""

    class NativeObservation:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    fake_envs = types.ModuleType("isaaclab.envs")
    fake_envs.mdp = SimpleNamespace(base_lin_vel=object())
    monkeypatch.setitem(sys.modules, "isaaclab.envs", fake_envs)
    monkeypatch.setattr(isaac_terms, "_import_cfgs", lambda: {"obs": NativeObservation})

    spec = registry.spec("Instinct-Perceptive-Shadowing-G1-v0").mdp.observations["critic"].terms[
        "base_lin_vel"
    ]
    noise = object()
    ctx = SimpleNamespace(params=lambda _spec: {}, noise=lambda _noise: noise)
    native = isaac_terms.TERMS.lookup("observation", "shadow_base_linear_velocity")(spec, ctx)

    assert native.history_length == spec.history_length == 8
    assert native.noise is noise
    assert native.scale == spec.scale
    assert native.clip == spec.clip


def test_imitation_rewards_and_failures_match_effective_sources() -> None:
    task = registry.spec("Instinct-Perceptive-Shadowing-G1-v0")
    rewards = task.mdp.rewards["rewards"]
    assert rewards["base_position_imitation_gauss"].weight == 0.5
    assert rewards["base_position_imitation_gauss"].params == {"std": 0.3}
    assert rewards["base_rot_imitation_gauss"].params == {
        "std": 0.4,
        "difference_type": "axis_angle",
    }
    assert rewards["link_pos_imitation_gauss"].params == {
        "combine_method": "mean_prod",
        "in_base_frame": False,
        "in_relative_world_frame": True,
        "std": 0.3,
    }
    assert rewards["undesired_contacts"].weight == -0.1
    assert rewards["applied_torque_limits_by_ratio"].weight == -0.05
    assert rewards["undesired_contacts"].params["sensor"].elements != ".*"

    done = task.mdp.terminations
    assert done["base_pos_too_far"].params["distance_threshold"] == 0.25
    assert done["base_pos_too_far"].params["height_only"] is True
    assert done["base_pg_too_far"].params["projected_gravity_threshold"] == 0.8
    assert done["base_pg_too_far"].params["z_only"] is False
    assert tuple(done["link_pos_too_far"].target.bodies) == (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    assert done["illegal_reset_contact"].params["episode_length_threshold"] == 2


def test_camera_height_and_contact_sensor_contract() -> None:
    task = registry.spec("Instinct-Perceptive-Shadowing-G1-v0")
    camera = task.scene.ray_caster("camera")
    assert (camera.pattern.height, camera.pattern.width) == (27, 48)
    assert camera.crop == (2, 2, 2, 2)
    assert camera.update_period == 1.0 / 60.0
    assert camera.min_distance == 0.05
    assert camera.for_engine("isaacsim").max_distance == 1.0e6
    scanner = task.scene.ray_caster("height_scanner")
    assert scanner.mode == "terrain_height"
    assert scanner.for_engine("isaacsim").max_distance == pytest.approx(1.0e6)
    assert scanner.for_engine("mjlab").max_distance == pytest.approx(5.0)
    contact = task.scene.contact_sensors[0]
    assert contact.history_length == 3
    assert contact.track_air_time is True
    assert contact.air_time_force_threshold == 1.0
    assert contact.for_engine("isaacsim").air_time_force_threshold == 10.0
    assert contact.for_engine("mjlab").air_time_force_threshold == 1.0

    hoi = registry.spec("Instinct-Perceptive-HOI-Shadowing-G1-v0")
    hoi_camera = hoi.scene.ray_caster("camera")
    assert {obj.name for obj in hoi.scene.rigid_objects} <= set(hoi_camera.hit)


def test_isaac_height_scan_selects_the_whole_ray_sensor_without_contact_body_fields(monkeypatch) -> None:
    """A RayCasterRef has no contact ``elements``; the height term selects it by name."""
    from instinctlab.engines.isaacsim import terms as isaac_terms

    class SceneEntityCfg:
        def __init__(self, name, **kwargs):
            self.name = name
            self.kwargs = kwargs

    fake_isaaclab = types.ModuleType("isaaclab")
    fake_envs = types.ModuleType("isaaclab.envs")
    fake_mdp = types.ModuleType("isaaclab.envs.mdp")
    fake_managers = types.ModuleType("isaaclab.managers")
    fake_mdp.height_scan = object()
    fake_envs.mdp = fake_mdp
    fake_isaaclab.envs = fake_envs
    fake_managers.SceneEntityCfg = SceneEntityCfg
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.envs", fake_envs)
    monkeypatch.setitem(sys.modules, "isaaclab.envs.mdp", fake_mdp)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", fake_managers)
    monkeypatch.setattr(
        isaac_terms,
        "_import_cfgs",
        lambda: {"obs": lambda **kwargs: SimpleNamespace(**kwargs)},
    )

    term = registry.spec("Instinct-Perceptive-Shadowing-G1-v0").mdp.observations["critic"].terms["height_scan"]
    built = isaac_terms._shadow_height(term, SimpleNamespace(params=lambda value: dict(value.params)))
    selector = built.params["sensor_cfg"]
    assert selector.name == "height_scanner"
    assert selector.kwargs == {}


def test_domain_randomization_and_reset_order_match_effective_sources() -> None:
    whole = registry.spec("Instinct-Shadowing-WholeBody-Plane-G1-v0")
    events = whole.mdp.events
    assert events["match_motion_ref_with_scene"].mode == "startup"
    assert events["push_robot"].params["velocity_range"]["yaw"] == (-0.78, 0.78)
    assert events["base_com"].params["com_range"]["x"] == (-0.025, 0.025)
    assert events["reset_robot"].params["randomize_joint_pos_range"] == (-0.1, 0.1)
    assert events["physics_material"].resolved_params("isaacsim")["restitution_range"] == (0.0, 0.5)

    perceptive = registry.spec("Instinct-Perceptive-Shadowing-G1-v0")
    assert perceptive.scene.terrain.kind == "shadow_motion_matched"
    assert perceptive.mdp.events["reset_robot"].params["randomize_velocity_range"] == {}
    assert perceptive.mdp.events["physics_material"].resolved_params("mjlab")["ranges"][2] == (0.0, 0.5)


def test_isaac_perceptive_terrain_matches_main_zero_border_height(monkeypatch) -> None:
    """Do not inherit Isaac Lab's non-zero terrain border-height default."""
    from instinctlab.engines.isaacsim import scene as isaac_scene

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_isaaclab_terrains = types.ModuleType("isaaclab.terrains")
    fake_isaaclab_terrains.TerrainImporterCfg = Config
    fake_instinctlab_terrains = types.ModuleType("instinctlab.terrains")
    fake_instinctlab_terrains.TerrainImporterCfg = Config
    fake_generator_cfg = types.ModuleType("instinctlab.terrains.terrain_generator_cfg")
    fake_generator_cfg.FiledTerrainGeneratorCfg = Config
    fake_mesh_cfg = types.ModuleType("instinctlab.terrains.trimesh.mesh_terrains_cfg")
    fake_mesh_cfg.MotionMatchedTerrainCfg = Config

    monkeypatch.setitem(sys.modules, "isaaclab.terrains", fake_isaaclab_terrains)
    monkeypatch.setitem(sys.modules, "instinctlab.terrains", fake_instinctlab_terrains)
    monkeypatch.setitem(sys.modules, "instinctlab.terrains.terrain_generator_cfg", fake_generator_cfg)
    monkeypatch.setitem(sys.modules, "instinctlab.terrains.trimesh.mesh_terrains_cfg", fake_mesh_cfg)
    monkeypatch.setattr(isaac_scene, "_physics_material", lambda _spec: object())
    monkeypatch.setattr(isaac_scene, "_visual_material", lambda: object())

    terrain_spec = registry.spec("Instinct-Perceptive-Shadowing-G1-v0").scene.terrain
    terrain_cfg = isaac_scene._terrain(terrain_spec, {})

    assert terrain_cfg.terrain_generator.border_width == 0.0
    assert terrain_cfg.terrain_generator.border_height == 0.0


def test_perceptive_play_keeps_only_reference_evaluation_randomization() -> None:
    """Perceptive play keeps sensor/actuator/body DR but removes train-only state noise."""
    task_ids = (
        "Instinct-Perceptive-Shadowing-G1-Play-v0",
        "Instinct-Perceptive-Shadowing-G1-OneMotion-Play-v0",
        "Instinct-Perceptive-Vae-G1-Play-v0",
        "Instinct-Perceptive-HOI-Shadowing-G1-Play-v0",
    )
    zero_spatial_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }

    expected_events = {
        task_ids[0]: {
            "randomize_ray_offsets",
            "randomize_actuator_gains",
            "randomize_rigid_body_mass",
            "match_motion_ref_with_scene",
            "reset_robot",
        },
        task_ids[1]: {
            "randomize_ray_offsets",
            "randomize_actuator_gains",
            "randomize_rigid_body_mass",
            "match_motion_ref_with_scene",
            "reset_robot",
        },
        task_ids[2]: {
            "randomize_ray_offsets",
            "randomize_actuator_gains",
            "randomize_rigid_body_mass",
            "match_motion_ref_with_scene",
            "reset_robot",
        },
        task_ids[3]: {
            "randomize_ray_offsets",
            "randomize_actuator_gains",
            "randomize_rigid_body_mass",
            "reset_robot",
            "reset_rigid_objects_state_by_reference",
            "update_rigid_objects_state_by_reference",
        },
    }

    for task_id in task_ids:
        events = registry.spec(task_id).mdp.events
        assert set(events) == expected_events[task_id], task_id
        reset = events["reset_robot"].params
        assert reset["randomize_pose_range"] == zero_spatial_range, task_id
        assert reset["randomize_velocity_range"] == zero_spatial_range, task_id
        assert reset["randomize_joint_pos_range"] == (0.0, 0.0), task_id

    # Whole-body and BeyondMimic references intentionally retain their play-time DR.
    for task_id in (
        "Instinct-Shadowing-WholeBody-Plane-G1-Play-v0",
        "Instinct-BeyondMimic-Plane-G1-Play-v0",
    ):
        events = registry.spec(task_id).mdp.events
        assert {"physics_material", "add_joint_default_pos", "base_com"} <= events.keys(), task_id
        assert events["reset_robot"].params["randomize_joint_pos_range"] != (0.0, 0.0), task_id


def test_one_motion_without_bins_does_not_schedule_adaptive_sampling() -> None:
    one_motion = registry.spec("Instinct-Perceptive-Shadowing-G1-OneMotion-v0")
    assert one_motion.scene.motion_references[0].motion_bin_length_s is None
    assert "beyond_adaptive_sampling" not in one_motion.mdp.curriculum
    assert "bin_fail_counter_smoothing" not in one_motion.mdp.events

    binned = registry.spec("Instinct-Perceptive-Shadowing-G1-v0")
    assert binned.scene.motion_references[0].motion_bin_length_s is not None
    assert "beyond_adaptive_sampling" in binned.mdp.curriculum
    assert binned.mdp.events["bin_fail_counter_smoothing"].interval_range_s == (0.02, 0.02)


def test_hoi_play_reset_preserves_the_engine_specific_reference_offset() -> None:
    train = registry.spec("Instinct-Perceptive-HOI-Shadowing-G1-v0").mdp.events["reset_robot"]
    play = registry.spec("Instinct-Perceptive-HOI-Shadowing-G1-Play-v0").mdp.events["reset_robot"]

    assert train.resolved_params("isaacsim")["position_offset"] == (0.0, 0.0, 0.0)
    assert train.resolved_params("mjlab")["position_offset"] == (0.0, 0.0, 0.0)
    assert play.resolved_params("isaacsim")["position_offset"] == (0.0, 1.0, 2.0)
    assert play.resolved_params("mjlab")["position_offset"] == (0.0, 0.0, 0.0)


def test_mjlab_shadow_body_inertia_is_uniform_in_mass_scale() -> None:
    task = registry.spec("Instinct-Perceptive-Shadowing-G1-v0")
    compiled = MjlabAdapter().compile(task, num_envs=2, device="cpu", strict=True)
    params = compiled.env_cfg.events["randomize_rigid_body_mass"].params
    distribution = params["distribution"]
    lo, hi = params["alpha_range"]

    assert distribution.name == "uniform_mass_scale_to_alpha"
    torch.manual_seed(123)
    alpha = distribution.sample(torch.tensor([lo]), torch.tensor([hi]), (32,), "cpu")
    torch.manual_seed(123)
    expected_mass_scale = torch.rand(32) * 0.4 + 0.8
    torch.testing.assert_close(torch.exp(2.0 * alpha), expected_mass_scale)


def test_whole_body_undesired_contact_reward_excludes_support_links() -> None:
    """Both main and InstinctMJ permit ankle and wrist support contacts."""
    task = registry.spec("Instinct-Shadowing-WholeBody-Plane-G1-v0")
    sensor = task.mdp.rewards["rewards"]["undesired_contacts"].params["sensor"]
    pattern = sensor.elements[0]
    assert re.fullmatch(pattern, "pelvis")
    for support in (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ):
        assert re.fullmatch(pattern, support) is None


def test_mjlab_pd_gain_randomization_uses_the_reference_whole_robot_selector() -> None:
    task = registry.spec("Instinct-Perceptive-Shadowing-G1-v0")
    compiled = MjlabAdapter().compile(task, num_envs=2, device="cpu", strict=True)
    selector = compiled.env_cfg.events["randomize_actuator_gains"].params["asset_cfg"]
    assert selector.actuator_names is None
    assert selector.actuator_ids == slice(None)
    assert all(type(act).__name__ == "BuiltinPdActuatorCfg" for act in compiled.env_cfg.scene.entities["robot"].articulation.actuators)


def test_omomo_object_slots_are_name_mapped_and_invalid_slots_are_cleared() -> None:
    names = ("floorlamp", "largebox")
    buffers = make_buffers(2, 1, 1, 1, names)
    zeros3 = torch.zeros(2, 1, 3)
    zeros4 = torch.zeros(2, 1, 4)
    zeros4[..., 0] = 1.0
    sample = MotionSample(
        frame_index=torch.zeros(2, 1, dtype=torch.long),
        validity=torch.ones(2, 1, dtype=torch.bool),
        joint_pos=torch.zeros(2, 1, 1),
        joint_vel=torch.zeros(2, 1, 1),
        base_pos_w=zeros3.clone(),
        base_quat_w=zeros4.clone(),
        base_lin_vel_w=zeros3.clone(),
        base_ang_vel_w=zeros3.clone(),
        link_pos_b=torch.zeros(2, 1, 1, 3),
        link_quat_b=torch.cat((torch.ones(2, 1, 1, 1), torch.zeros(2, 1, 1, 3)), -1),
        link_pos_w=torch.zeros(2, 1, 1, 3),
        link_quat_w=torch.cat((torch.ones(2, 1, 1, 1), torch.zeros(2, 1, 1, 3)), -1),
        link_lin_vel_b=torch.zeros(2, 1, 1, 3),
        link_ang_vel_b=torch.zeros(2, 1, 1, 3),
        link_lin_vel_w=torch.zeros(2, 1, 1, 3),
        link_ang_vel_w=torch.zeros(2, 1, 1, 3),
        object_name="largebox",
        object_pos_w=torch.full((2, 1, 3), 2.0),
        object_quat_w=zeros4.clone(),
        object_lin_vel_w=torch.full((2, 1, 3), 3.0),
        object_ang_vel_w=torch.full((2, 1, 3), 4.0),
    )
    fill_buffers(buffers, torch.arange(2), sample, torch.zeros(2, 1))
    assert not buffers.object_validity[:, :, 0].any()
    assert buffers.object_validity[:, :, 1].all()
    torch.testing.assert_close(buffers.object_pos_w[:, :, 1], torch.full((2, 1, 3), 2.0))


def test_hoi_scene_and_motion_declare_the_same_ordered_objects() -> None:
    task = registry.spec("Instinct-Perceptive-HOI-Shadowing-G1-v0")
    names = tuple(obj.name for obj in task.scene.rigid_objects)
    assert names == tuple(task.scene.motion_references[0].scene_objects)
    assert len(names) == 6
    assert tuple(task.scene.motion_references[0].links) == MOTION_LINKS
    assert "out_of_border" not in task.mdp.terminations


def test_all_shadowing_tasks_compile_to_mjlab_without_gpu_or_dataset_io() -> None:
    adapter = MjlabAdapter()
    for task_id in SHADOW_IDS:
        compiled = adapter.compile(registry.spec(task_id), num_envs=1, device="cpu", strict=True)
        assert not compiled.resolution.skipped


def test_mjlab_motion_matched_terrain_uses_instinctmj_coacd_profile() -> None:
    adapter = MjlabAdapter()
    compiled = adapter.compile(
        registry.spec("Instinct-Perceptive-Shadowing-G1-v0"),
        num_envs=1,
        device="cpu",
        strict=True,
    )
    terrain = compiled.env_cfg.scene.terrain.terrain_generator.sub_terrains["motion_matched"]
    assert terrain.use_input_origin_frame is True
    assert terrain.collision_coacd_threshold == 0.04
    assert terrain.collision_coacd_resolution == 3000
    assert terrain.collision_coacd_max_ch_vertex == 256
    assert terrain.collision_coacd_visualize_collision_hulls is True
    assert terrain.collision_coacd_auto_align_top_surface is True
    assert terrain.collision_coacd_prewarm_all is True
    assert terrain.collision_coacd_prewarm_workers == 0


def test_mjlab_motion_matched_coacd_builds_collision_hulls(tmp_path) -> None:
    from instinctlab.engines.mjlab.motion_matched_terrain import motion_matched_terrain

    mesh_path = tmp_path / "room.stl"
    trimesh.creation.box(extents=(2.0, 2.0, 0.2)).export(mesh_path)
    metadata = tmp_path / "metadata.yaml"
    metadata.write_text(yaml.safe_dump({"terrains": [{"terrain_file": mesh_path.name}]}))
    cfg = SimpleNamespace(
        metadata_yaml=str(metadata),
        path=str(tmp_path),
        size=(4.0, 4.0),
        crop_to_size=False,
        use_input_origin_frame=True,
        collision_coacd_threshold=0.04,
        collision_coacd_resolution=100,
        collision_coacd_decimate=False,
        collision_coacd_max_ch_vertex=256,
        collision_coacd_log_level="off",
        collision_coacd_use_disk_cache=False,
        collision_coacd_cache_dirname=".coacd_cache",
        collision_coacd_prewarm_all=False,
        collision_coacd_prewarm_workers=0,
        collision_coacd_z_offset=0.0,
        collision_coacd_auto_align_top_surface=True,
        collision_coacd_auto_align_resolution=0.04,
        collision_coacd_visualize_collision_hulls=True,
        collision_coacd_geom_margin=0.0,
    )
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = motion_matched_terrain(cfg, 0.0, spec, None)

    assert output.geometries
    assert output.origin.tolist() == [2.0, 2.0, 0.0]
    assert all(geometry.geom.group == 2 for geometry in output.geometries)
    spec.compile()
