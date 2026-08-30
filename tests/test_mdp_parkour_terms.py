"""Stub-env checks for the portable parkour terms.

The static portability scan in ``test_mdp_terms.py`` already covers these functions — same
package, same denylist. This file asks whether each new formula computes the number the two
references agree on, against stubs with known values.
"""

from __future__ import annotations

import inspect
import math
import torch
from types import SimpleNamespace

import pytest
from instinctlab_engine.bridge.observation_history import clear_observation_histories_on_reset
from instinctlab_engine.spec.sensor import (
    ContactSensorRef,
    RayCasterRef,
    RayPatternRef,
    VolumePointsRef,
)
from instinctlab.tasks.parkour.mdp import (
    curriculums,
    observations,
    rewards,
    terminations,
)
from test_mdp_terms import _Cfg, _Entity, _Env, _Sensor


class _CommandTerm:
    def __init__(self, metrics):
        self.metrics = metrics


def test_dataset_exhausted_silently_resets_only_invalid_references() -> None:
    class _MotionReference:
        aiming_frame_idx = torch.tensor([0, 1, 0])

        def __init__(self):
            self.data = SimpleNamespace(
                validity=torch.tensor([[True, True], [True, False], [True, True]])
            )
            self.reset_ids = None

        def reset(self, *, env_ids):
            self.reset_ids = env_ids.clone()

    reference = _MotionReference()
    env = SimpleNamespace(
        scene=SimpleNamespace(sensors={"motion_reference": reference})
    )
    sensor = SimpleNamespace(name="motion_reference")

    result = terminations.dataset_exhausted(env, sensor, reset_without_notice=True)

    assert result.tolist() == [False, False, False]
    assert reference.reset_ids.tolist() == [1]


class _CommandManager:
    def __init__(self, commands, terms=None):
        self.active_terms = list(commands)
        self._commands = commands
        self._terms = terms or {}

    def get_command(self, name):
        return self._commands.get(name)

    def get_term(self, name):
        return self._terms[name]


class _Generator:
    def __init__(self, size=(8.0, 8.0), num_rows=10, num_cols=20, border_width=3.0):
        self.size = size
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.border_width = border_width


class _Terrain:
    def __init__(self, generator=None, levels=None):
        self.cfg = type("Cfg", (), {"terrain_generator": generator})()
        self.terrain_levels = (
            levels if levels is not None else torch.tensor([3.0, 3.0, 3.0])
        )
        self.calls: list = []

    def update_env_origins(self, env_ids, move_up, move_down):
        self.calls.append((env_ids.clone(), move_up.clone(), move_down.clone()))


def test_is_alive_is_the_complement_of_is_terminated():
    env = _Env(terminated=torch.tensor([True, False]))
    assert rewards.is_alive(env).tolist() == pytest.approx([0.0, 1.0])
    assert (
        rewards.is_alive(env) + rewards.is_terminated(env)
    ).tolist() == pytest.approx([1.0, 1.0])


def test_track_lin_vel_xy_exp_is_the_base_frame_kernel():
    robot = _Entity(
        root_link_lin_vel_b=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )
    env = _Env(
        entities={"robot": robot},
        commands={"base_velocity": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])},
    )
    reward = rewards.track_lin_vel_xy_exp(env, "base_velocity", std=0.5)
    assert reward[0].item() == pytest.approx(1.0)
    assert reward[1].item() == pytest.approx(math.exp(-1.0 / 0.25))


def test_track_ang_vel_z_exp_is_the_base_frame_yaw_kernel():
    robot = _Entity(root_link_ang_vel_b=torch.tensor([[0.0, 0.0, 0.3]]))
    env = _Env(
        entities={"robot": robot},
        commands={"base_velocity": torch.tensor([[0.0, 0.0, 0.5]])},
    )
    reward = rewards.track_ang_vel_z_exp(env, "base_velocity", std=0.5).item()
    assert reward == pytest.approx(math.exp(-(0.2**2) / 0.25))


def test_ang_vel_xy_l2_and_joint_vel_l2():
    robot = _Entity(
        root_link_ang_vel_b=torch.tensor([[0.3, 0.4, 0.5]]),
        joint_vel=torch.tensor([[1.0, 2.0, 3.0]]),
    )
    env = _Env(entities={"robot": robot})
    assert rewards.ang_vel_xy_l2(env).item() == pytest.approx(0.25)
    assert rewards.joint_vel_l2(env).item() == pytest.approx(14.0)
    assert rewards.joint_vel_l2(env, _Cfg(joint_ids=[0, 1])).item() == pytest.approx(
        5.0
    )


def test_stand_still_when_idle_subtracts_the_offset_and_uses_its_own_gate():
    robot = _Entity(
        joint_pos=torch.tensor([[1.0, 1.0]] * 2), default_joint_pos=torch.zeros(2, 2)
    )
    env = _Env(
        entities={"robot": robot},
        commands={"base_velocity": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])},
    )
    assert rewards.stand_still_when_idle(
        env, "base_velocity", offset=1.0
    ).tolist() == pytest.approx([1.0, 0.0])
    # The existing stand_still is unchanged: no offset, 0.1 gate.
    assert rewards.stand_still(env, "base_velocity").tolist() == pytest.approx(
        [2.0, 0.0]
    )


def test_heading_error_is_the_abs_yaw_command():
    env = _Env(
        commands={"base_velocity": torch.tensor([[1.0, 0.0, -0.4], [0.0, 0.0, 0.0]])}
    )
    assert rewards.heading_error(env, "base_velocity").tolist() == pytest.approx(
        [0.4, 0.0]
    )


def test_dont_wait_counts_how_far_below_the_forward_thresholds():
    robot = _Entity(
        root_link_lin_vel_b=torch.tensor(
            [[0.10, 0.0, 0.0], [-0.20, 0.0, 0.0], [-0.20, 0.0, 0.0]]
        )
    )
    env = _Env(
        entities={"robot": robot},
        commands={
            "base_velocity": torch.tensor(
                [[0.4, 0.0, 0.0], [0.4, 0.0, 0.0], [0.2, 0.0, 0.0]]
            )
        },
    )
    assert rewards.dont_wait(env, "base_velocity").tolist() == pytest.approx(
        [1.0, 3.0, 0.0]
    )


def test_parkour_feet_air_time_does_not_clamp_and_gates_on_yaw_too():
    sensor_ref = ContactSensorRef(
        name="feet", elements=(".*foot",), track_air_time=True
    )
    sensor = _Sensor(
        body_names=["left_foot", "right_foot"],
        current_air_time=torch.tensor([[9.0, 0.0], [0.3, 0.0], [0.3, 0.0]]),
        current_contact_time=torch.tensor([[0.0, 3.0], [0.0, 0.2], [0.0, 0.2]]),
    )
    env = _Env(
        sensors={"feet": sensor},
        commands={
            "base_velocity": torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.4]]
            )
        },
    )
    reward = rewards.feet_air_time(env, "base_velocity", sensor_ref, vel_threshold=0.15)
    # Row 0 is uncapped single-stance contact time. Row 1 is a near-zero command. Row 2 is yaw.
    assert reward.tolist() == pytest.approx([3.0, 0.0, 0.2])


def test_parkour_feet_air_time_optional_threshold_caps_like_the_biped_term():
    sensor_ref = ContactSensorRef(
        name="feet", elements=(".*foot",), track_air_time=True
    )
    sensor = _Sensor(
        body_names=["left_foot", "right_foot"],
        current_air_time=torch.tensor([[9.0, 0.0]]),
        current_contact_time=torch.tensor([[0.0, 3.0]]),
    )
    env = _Env(
        sensors={"feet": sensor},
        commands={"base_velocity": torch.tensor([[1.0, 0.0, 0.0]])},
    )
    assert rewards.feet_air_time(
        env, "base_velocity", sensor_ref, vel_threshold=0.15, threshold=0.5
    ).item() == pytest.approx(0.5)


def test_joint_deviation_square():
    robot = _Entity(
        joint_pos=torch.tensor([[1.0, -2.0, 0.5]]), default_joint_pos=torch.zeros(1, 3)
    )
    env = _Env(entities={"robot": robot})
    assert rewards.joint_deviation_square(env).item() == pytest.approx(1.0 + 4.0 + 0.25)
    assert rewards.joint_deviation_square(
        env, _Cfg(joint_ids=[1])
    ).item() == pytest.approx(4.0)


def test_link_orientation_is_zero_when_the_link_is_upright():
    robot = _Entity(
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        body_link_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
    )
    env = _Env(entities={"robot": robot})
    assert rewards.link_orientation(env, _Cfg(body_ids=[0])).item() == pytest.approx(
        0.0
    )


def test_link_orientation_penalises_a_90_degree_roll():
    half = math.sqrt(2.0) / 2.0
    robot = _Entity(
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        body_link_quat_w=torch.tensor([[[half, half, 0.0, 0.0]]]),
    )
    env = _Env(entities={"robot": robot})
    assert rewards.link_orientation(env, _Cfg(body_ids=[0])).item() == pytest.approx(
        1.0
    )


def test_feet_orientation_contact_is_gated_on_contact_not_force():
    half = math.sqrt(2.0) / 2.0
    sensor_ref = ContactSensorRef(
        name="feet", elements=(".*foot",), track_air_time=True
    )
    sensor = _Sensor(
        body_names=["left_foot", "right_foot"],
        current_contact_time=torch.tensor([[0.4, 0.0]]),
    )
    robot = _Entity(
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        body_link_quat_w=torch.tensor(
            [[[half, half, 0.0, 0.0], [half, half, 0.0, 0.0]]]
        ),
    )
    env = _Env(entities={"robot": robot}, sensors={"feet": sensor})
    # Both feet are rolled 90 degrees; only the left one is in contact.
    assert rewards.feet_orientation_contact(
        env, sensor_ref, _Cfg(body_ids=[0, 1])
    ).item() == pytest.approx(1.0)


def test_feet_close_xy_gauss_pays_when_the_feet_are_apart_in_y():
    robot = _Entity(
        body_link_pos_w=torch.tensor([[[0.0, 0.10, 0.0], [0.0, -0.10, 0.0]]]),
        heading_w=torch.tensor([0.0]),
    )
    env = _Env(entities={"robot": robot})
    cfg = _Cfg(body_ids=[0, 1])
    assert rewards.feet_close_xy_gauss(
        env, threshold=0.12, std=0.1, asset_cfg=cfg
    ).item() == pytest.approx(0.0)
    robot.data.body_link_pos_w = torch.tensor([[[0.0, 0.02, 0.0], [0.0, -0.02, 0.0]]])
    expected = math.exp(-0.08 / 0.01) - 1.0
    assert rewards.feet_close_xy_gauss(
        env, threshold=0.12, std=0.1, asset_cfg=cfg
    ).item() == pytest.approx(expected)


class _RaySensor:
    def __init__(self, hits):
        self.data = type("Data", (), {"ray_hits_w": torch.tensor(hits)})()


def test_feet_at_plane_penalises_a_stance_foot_above_the_scan() -> None:
    contact = ContactSensorRef(name="feet", elements=(".*foot",), track_air_time=True)
    left = RayCasterRef(name="left_height_scanner", attach="left_foot")
    right = RayCasterRef(name="right_height_scanner", attach="right_foot")
    robot = _Entity(
        body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.20], [0.0, 0.0, 0.20]]])
    )
    ground = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    env = _Env(
        entities={"robot": robot},
        sensors={
            "feet": _Sensor(
                body_names=["left_foot", "right_foot"],
                current_contact_time=torch.tensor([[0.4, 0.4]]),
            ),
            "left_height_scanner": _RaySensor(ground),
            "right_height_scanner": _RaySensor(ground),
        },
    )
    # 0.20 - 0.00 - 0.05 = 0.15 per ray, two rays, two feet.
    reward = rewards.feet_at_plane(
        env, contact, left, right, asset_cfg=_Cfg(body_ids=[0, 1]), height_offset=0.05
    )
    assert reward.item() == pytest.approx(0.60)


def test_feet_at_plane_a_miss_is_not_ground_at_zero() -> None:
    """``hit_z = inf`` clamps to zero. The old ``inf → 0`` path would pay 0.3 here."""
    contact = ContactSensorRef(name="feet", elements=(".*foot",), track_air_time=True)
    left = RayCasterRef(name="left_height_scanner", attach="left_foot")
    right = RayCasterRef(name="right_height_scanner", attach="right_foot")
    robot = _Entity(
        body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.80], [0.0, 0.0, 0.80]]])
    )
    miss = float("inf")
    gone = [[[miss, miss, miss], [miss, miss, miss]]]
    env = _Env(
        entities={"robot": robot},
        sensors={
            "feet": _Sensor(
                body_names=["left_foot", "right_foot"],
                current_contact_time=torch.tensor([[0.4, 0.4]]),
            ),
            "left_height_scanner": _RaySensor(gone),
            "right_height_scanner": _RaySensor(gone),
        },
    )
    reward = rewards.feet_at_plane(
        env, contact, left, right, asset_cfg=_Cfg(body_ids=[0, 1]), height_offset=0.058
    )
    assert reward.item() == pytest.approx(0.0)


def test_delayed_depth_image_is_oldest_to_newest_and_does_not_use_mjlab_delay() -> None:
    """37-frame buffer, skip 5, delay 0/1. ``[:, -1]`` is the latest (or latest-1)."""
    sensor = RayCasterRef(
        name="camera",
        attach="torso_link",
        pattern=RayPatternRef(kind="pinhole", width=4, height=4),
        hit=("terrain",),
        max_distance=2.5,
    )
    raw = torch.zeros(2, 4, 4, 1)

    class _Data:
        output = {"distance_to_image_plane": raw}

    class _Cam:
        data = _Data()

    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=SimpleNamespace(sensors={"camera": _Cam()}),
    )
    cfg = SimpleNamespace(
        params={
            "sensor": sensor,
            "history_skip_frames": 5,
            "num_output_frames": 8,
            "delayed_frame_ranges": (0, 1),
            "history_length": 37,
            "blur_kernel_size": 1,
            "blur_sigma": 0.0,
        }
    )

    def _fill(term, delay: int) -> torch.Tensor:
        term._delay.fill_(delay)
        last = None
        for step in range(37):
            raw.fill_(0.05 * (step + 1))
            last = term(env, sensor)
        assert last is not None
        return last

    delay0 = _fill(observations.DelayedDepthImage(cfg, env), delay=0)
    # Written depths are 0.05, 0.10, ..., 1.85. Processed = depth / 2.5.
    # delay=0 indices: [1, 6, 11, 16, 21, 26, 31, 36] → depths 0.10 … 1.85
    assert tuple(delay0.shape) == (2, 8, 4, 4)
    assert delay0[:, -1].tolist()[0][0][0] == pytest.approx(1.85 / 2.5)
    assert delay0[:, 0].tolist()[0][0][0] == pytest.approx(0.10 / 2.5)
    delay1 = _fill(observations.DelayedDepthImage(cfg, env), delay=1)
    assert delay1[:, -1].tolist()[0][0][0] == pytest.approx(1.80 / 2.5)
    assert delay1[:, 0].tolist()[0][0][0] == pytest.approx(0.05 / 2.5)


def test_delayed_depth_image_does_not_convert_priming_state_to_a_python_bool(
    monkeypatch,
) -> None:
    sensor = RayCasterRef(
        name="camera",
        attach="torso_link",
        pattern=RayPatternRef(kind="pinhole", width=4, height=4),
        hit=("terrain",),
        max_distance=2.5,
    )
    raw = torch.ones(2, 4, 4, 1)

    class _Data:
        output = {"distance_to_image_plane": raw}

    class _Cam:
        data = _Data()
        frame_sequence = 1

    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=SimpleNamespace(sensors={"camera": _Cam()}),
    )
    cfg = SimpleNamespace(
        params={
            "sensor": sensor,
            "history_skip_frames": 1,
            "num_output_frames": 1,
            "delayed_frame_ranges": (0, 0),
            "history_length": 1,
            "blur_kernel_size": 1,
            "blur_sigma": 0.0,
        }
    )
    term = observations.DelayedDepthImage(cfg, env)

    def refuse_tensor_bool(_tensor) -> bool:
        raise AssertionError(
            "DelayedDepthImage converted priming state to a Python bool"
        )

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", refuse_tensor_bool)
        depth = term(env, sensor)

    assert tuple(depth.shape) == (2, 1, 4, 4)
    assert term._primed.tolist() == [True, True]


def test_delayed_depth_image_accepts_all_configured_processing_params() -> None:
    term, env, sensor, raw = _delayed_depth_term(
        num_envs=2,
        history_length=10,
        history_skip_frames=3,
        num_output_frames=4,
        delayed_frame_ranges=(0, 0),
        resize_shape=(18, 32),
        normalization_range=(0.0, 2.0),
    )
    result = None
    for frame in range(1, 11):
        raw.fill_(0.1 * frame)
        result = term(
            env,
            sensor,
            history_skip_frames=3,
            num_output_frames=4,
            delayed_frame_ranges=(0, 0),
            history_length=10,
            blur_kernel_size=1,
            blur_sigma=0.0,
            resize_shape=(18, 32),
            normalization_range=(0.0, 2.0),
        )

    assert result is not None
    assert result.shape == (2, 4, 18, 32)
    assert torch.allclose(
        result[:, :, 0, 0],
        torch.tensor([[0.05, 0.2, 0.35, 0.5], [0.05, 0.2, 0.35, 0.5]]),
        atol=1.0e-6,
    )


def _delayed_depth_term(num_envs: int = 3, history_length: int = 37, **param_overrides):
    sensor = RayCasterRef(
        name="camera",
        attach="torso_link",
        pattern=RayPatternRef(kind="pinhole", width=4, height=4),
        hit=("terrain",),
        max_distance=2.5,
    )
    raw = torch.zeros(num_envs, 4, 4, 1)

    class _Data:
        output = {"distance_to_image_plane": raw}

    class _Cam:
        data = _Data()

    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=SimpleNamespace(sensors={"camera": _Cam()}),
    )
    params = {
        "sensor": sensor,
        "history_skip_frames": 5,
        "num_output_frames": 8,
        "delayed_frame_ranges": (0, 1),
        "history_length": history_length,
        "blur_kernel_size": 1,
        "blur_sigma": 0.0,
    }
    params.update(param_overrides)
    cfg = SimpleNamespace(params=params)
    term = observations.DelayedDepthImage(cfg, env)
    return term, env, sensor, raw


def test_delayed_depth_partial_reset_clears_only_selected_envs() -> None:
    term, env, sensor, raw = _delayed_depth_term(num_envs=3)
    raw.fill_(1.25)
    for _ in range(8):
        term(env, sensor)
    kept = term._history[0].clone()
    assert float(term._history.abs().max()) > 0.0
    write_before = term._write
    term.clear_history(env_ids=torch.tensor([1, 2], dtype=torch.int32))
    assert torch.equal(term._history[0], kept)
    assert float(term._history[1].abs().max()) == 0.0
    assert float(term._history[2].abs().max()) == 0.0
    assert term._write == write_before


def test_delayed_depth_lookup_unwraps_isaac_manager_term() -> None:
    term, env, _sensor, _raw = _delayed_depth_term(num_envs=2)
    term._history.fill_(0.5)
    wrapper = SimpleNamespace(_impl=term)
    env.observation_manager = SimpleNamespace(
        _group_obs_term_cfgs={"policy": [SimpleNamespace(func=wrapper)]}
    )

    clear_observation_histories_on_reset(env, torch.tensor([1]))
    assert float(term._history[0].min()) == 0.5
    assert float(term._history[1].abs().max()) == 0.0


def test_observation_history_reset_rejects_an_incomplete_opt_in() -> None:
    implementation = SimpleNamespace(clears_history_on_env_reset=True)
    env = SimpleNamespace(
        device="cpu",
        observation_manager=SimpleNamespace(
            _group_obs_term_cfgs={
                "policy": [SimpleNamespace(func=implementation)]
            }
        ),
    )

    with pytest.raises(TypeError, match="does not define clear_history"):
        clear_observation_histories_on_reset(env, torch.tensor([0]))


def test_delayed_depth_reset_only_resamples_delay() -> None:
    term, env, sensor, raw = _delayed_depth_term(num_envs=2)
    raw.fill_(1.25)
    term(env, sensor)
    before = term._history.clone()
    term.reset(env_ids=torch.tensor([0], dtype=torch.int32))
    assert torch.equal(term._history[0], before[0])
    assert float(term._history[0].abs().max()) > 0.0


def test_delayed_depth_full_reset_clears_every_env() -> None:
    term, env, sensor, raw = _delayed_depth_term(num_envs=2)
    raw.fill_(0.8)
    for _ in range(4):
        term(env, sensor)
    term.clear_history()
    assert float(term._history.abs().max()) == 0.0
    term.clear_history(env_ids=None)
    assert float(term._history.abs().max()) == 0.0


def test_delayed_depth_reset_empty_ids_is_a_noop() -> None:
    term, env, sensor, raw = _delayed_depth_term(num_envs=2)
    raw.fill_(0.4)
    term(env, sensor)
    before = term._history.clone()
    delay_before = term._delay.clone()
    term.reset(env_ids=torch.tensor([], dtype=torch.long))
    assert torch.equal(term._history, before)
    assert torch.equal(term._delay, delay_before)


def test_delayed_depth_reset_accepts_scalar_and_foreign_device_ids() -> None:
    term, _env, _sensor, _raw = _delayed_depth_term(num_envs=2)
    term._history[1] = 0.3
    term.clear_history(env_ids=torch.tensor(1, dtype=torch.int64))
    assert float(term._history[1].abs().max()) == 0.0
    term._history[0] = 0.2
    cpu_ids = torch.tensor([0], device="cpu", dtype=torch.int32)
    term.clear_history(env_ids=cpu_ids)
    assert float(term._history[0].abs().max()) == 0.0


def test_delayed_depth_first_push_primes_all_slots_and_second_push_rolls() -> None:
    term, env, sensor, raw = _delayed_depth_term(num_envs=2)
    raw.fill_(1.25)
    first = term(env, sensor)
    processed = 1.25 / 2.5
    assert torch.allclose(term._history, torch.full_like(term._history, processed))
    assert bool(term._primed.all())
    assert torch.allclose(first, torch.full_like(first, processed))
    raw.fill_(2.0)
    term._delay.fill_(0)
    second = term(env, sensor)
    processed_second = 2.0 / 2.5
    write_prev = (term._write - 1) % term.sensor_history_length
    assert torch.allclose(
        term._history[:, write_prev],
        torch.full_like(term._history[:, write_prev], processed_second),
    )
    other = [i for i in range(term.sensor_history_length) if i != write_prev]
    assert torch.allclose(
        term._history[:, other], torch.full_like(term._history[:, other], processed)
    )
    assert float((second[:, -1] - processed_second).abs().max()) == 0.0


def test_delayed_depth_subset_reset_does_not_prime_unreset_envs() -> None:
    term, env, sensor, raw = _delayed_depth_term(num_envs=3)
    raw.fill_(1.0)
    for _ in range(4):
        term(env, sensor)
    kept = term._history[0].clone()
    primed_before = term._primed.clone()
    write_before = term._write
    term.clear_history(env_ids=torch.tensor([1, 2]))
    assert torch.equal(term._history[0], kept)
    assert bool(term._primed[0]) == bool(primed_before[0])
    assert not bool(term._primed[1])
    assert not bool(term._primed[2])
    assert term._write == write_before
    raw.fill_(0.5)
    term._delay[1] = 0
    term._delay[2] = 1
    before_unreset = term._history[0].clone()
    out = term(env, sensor)
    processed = 0.5 / 2.5
    write_prev = (term._write - 1) % term.sensor_history_length
    assert torch.allclose(
        term._history[1], torch.full_like(term._history[1], processed)
    )
    assert torch.allclose(
        term._history[2], torch.full_like(term._history[2], processed)
    )
    assert torch.allclose(
        term._history[0, write_prev],
        torch.full((term._history.shape[2], term._history.shape[3]), processed),
    )
    other = [i for i in range(term.sensor_history_length) if i != write_prev]
    assert torch.equal(term._history[0, other], before_unreset[other])
    assert torch.allclose(out[1], torch.full_like(out[1], processed))
    assert torch.allclose(out[2], torch.full_like(out[2], processed))


def test_delayed_depth_prime_is_identical_for_delay_0_and_1() -> None:
    term, env, sensor, raw = _delayed_depth_term(num_envs=2)
    raw.fill_(1.25)
    term._delay[0] = 0
    term._delay[1] = 1
    out = term(env, sensor)
    processed = 1.25 / 2.5
    assert torch.allclose(out[0], torch.full_like(out[0], processed))
    assert torch.allclose(out[1], torch.full_like(out[1], processed))


def test_delayed_depth_reset_keeps_delay_in_declared_range() -> None:
    term, _env, _sensor, _raw = _delayed_depth_term(num_envs=8)
    lo, hi = term.delayed_frame_ranges
    term.reset(env_ids=torch.arange(8))
    assert int(term._delay.min()) >= int(lo)
    assert int(term._delay.max()) <= int(hi)


def test_delayed_depth_first_output_after_reset_has_no_old_frames() -> None:
    """After reset, the first valid frame primes all 37 slots. Skip/delay then see copies, not zeros or the old episode."""
    term, env, sensor, raw = _delayed_depth_term(num_envs=2)
    raw.fill_(2.0)
    for _ in range(37):
        term(env, sensor)
    old = term(env, sensor)
    assert float(old.abs().max()) > 0.0
    kept = term._history[1].clone()
    term.clear_history(env_ids=torch.tensor([0]))
    term._delay[0] = 0
    term._delay[1] = 0
    raw.fill_(1.25)
    first = term(env, sensor)
    processed_new = 1.25 / 2.5
    write_prev = (term._write - 1) % term.sensor_history_length
    assert first[0].shape[0] == 8
    assert torch.allclose(first[0], torch.full_like(first[0], processed_new))
    assert float((term._history[0] - processed_new).abs().max()) == 0.0
    assert torch.allclose(
        term._history[1, write_prev],
        torch.full_like(term._history[1, write_prev], processed_new),
    )
    other = [i for i in range(term.sensor_history_length) if i != write_prev]
    assert torch.equal(term._history[1, other], kept[other])
    assert float((first[1] - processed_new).abs().max()) > 0.0


def test_undesired_contacts_counts_touches_without_a_newton_threshold():
    sensor_ref = ContactSensorRef(name="body", elements=(".*",), track_air_time=True)
    sensor = _Sensor(
        body_names=["torso", "pelvis", "head"],
        current_contact_time=torch.tensor([[0.0, 0.4, 0.0], [0.2, 0.3, 0.1]]),
    )
    env = _Env(sensors={"body": sensor})
    assert rewards.undesired_contacts(env, sensor_ref).tolist() == pytest.approx(
        [1.0, 3.0]
    )
    assert "threshold" not in inspect.signature(rewards.undesired_contacts).parameters


def test_bad_orientation_uses_projected_gravity():
    upright = _Entity(projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]))
    tilted = _Entity(
        projected_gravity_b=torch.tensor([[0.0, math.sin(1.2), -math.cos(1.2)]])
    )
    assert terminations.bad_orientation(
        _Env(entities={"robot": upright}), limit_angle=1.0
    ).tolist() == [False]
    assert terminations.bad_orientation(
        _Env(entities={"robot": tilted}), limit_angle=1.0
    ).tolist() == [True]
    assert terminations.bad_orientation(
        _Env(entities={"robot": tilted}), limit_angle=1.5
    ).tolist() == [False]


def test_bad_orientation_clamps_roundoff_before_acos():
    almost_upright = _Entity(projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0000001]]))
    almost_inverted = _Entity(projected_gravity_b=torch.tensor([[0.0, 0.0, 1.0000001]]))
    assert terminations.bad_orientation(
        _Env(entities={"robot": almost_upright}), limit_angle=1.0
    ).tolist() == [False]
    assert terminations.bad_orientation(
        _Env(entities={"robot": almost_inverted}), limit_angle=1.0
    ).tolist() == [True]


def test_root_height_below_env_origin_minimum_clamps_the_origin():
    robot = _Entity(
        root_link_pos_w=torch.tensor(
            [[0.0, 0.0, 0.3], [0.0, 0.0, -0.6], [0.0, 0.0, 0.3]]
        )
    )
    env = _Env(entities={"robot": robot})
    env.scene.env_origins = torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 2.0]]
    )
    assert terminations.root_height_below_env_origin_minimum(env, 0.5).tolist() == [
        False,
        True,
        True,
    ]


def test_terrain_out_of_bounds_uses_the_whole_map():
    # size 8x8, 10 rows, 20 cols, border 3 → half extents (43, 83); buffer 2 → limits 41, 81.
    robot = _Entity(
        root_link_pos_w=torch.tensor(
            [[0.0, 0.0, 0.8], [42.0, 0.0, 0.8], [0.0, 82.0, 0.8]]
        )
    )
    env = _Env(entities={"robot": robot})
    env.scene.terrain = _Terrain(generator=_Generator())
    assert terminations.terrain_out_of_bounds(env, distance_buffer=2.0).tolist() == [
        False,
        True,
        True,
    ]


def test_terrain_out_of_bounds_matches_reference_infinite_plane():
    env = _Env()
    env.num_envs = 1
    env.device = "cpu"
    env.scene.terrain = _Terrain(generator=None)
    env.scene["robot"] = _Entity(root_link_pos_w=torch.zeros(1, 3))
    assert terminations.terrain_out_of_bounds(env, distance_buffer=2.0).tolist() == [
        False
    ]


def test_terrain_out_of_bounds_refuses_a_generator_missing_a_field():
    env = _Env()
    env.scene["robot"] = _Entity(root_link_pos_w=torch.zeros(1, 3))
    env.scene.terrain = _Terrain(generator=type("Bare", (), {"size": (8.0, 8.0)})())
    with pytest.raises(RuntimeError, match="num_rows"):
        terminations.terrain_out_of_bounds(env, distance_buffer=2.0)


def test_tracking_exp_vel_promotes_and_demotes():
    env = _Env(commands={"base_velocity": torch.zeros(3, 3)})
    env.command_manager = _CommandManager(
        {"base_velocity": torch.zeros(3, 3)},
        terms={
            "base_velocity": _CommandTerm(
                {
                    "tracking_exp_vel_xy": torch.tensor([0.7, 0.2, 0.5]),
                    "tracking_exp_vel_yaw": torch.tensor([0.6, 0.4, 0.0]),
                }
            )
        },
    )
    env.scene.terrain = _Terrain(
        generator=_Generator(), levels=torch.tensor([3.0, 3.0, 3.0])
    )
    mean = curriculums.tracking_exp_vel(env, torch.tensor([0, 1, 2]), "base_velocity")
    assert mean.item() == pytest.approx(3.0)
    _, move_up, move_down = env.scene.terrain.calls[0]
    assert move_up.tolist() == [True, False, False]
    assert move_down.tolist() == [False, True, False]


def test_tracking_exp_vel_refuses_a_plane():
    env = _Env()
    env.scene.terrain = _Terrain(generator=None)
    with pytest.raises(RuntimeError, match="no generator"):
        curriculums.tracking_exp_vel(env, torch.tensor([0]), "base_velocity")


def test_tracking_exp_vel_lists_available_metrics_when_a_key_is_missing():
    env = _Env()
    env.scene.terrain = _Terrain(generator=_Generator())
    env.command_manager = _CommandManager(
        {"base_velocity": torch.zeros(1, 3)},
        terms={
            "base_velocity": _CommandTerm({"tracking_exp_vel_xy": torch.tensor([0.5])})
        },
    )
    with pytest.raises(RuntimeError, match="tracking_exp_vel_yaw") as caught:
        curriculums.tracking_exp_vel(env, torch.tensor([0]), "base_velocity")
    assert "tracking_exp_vel_xy" in str(caught.value)


def test_joint_vel_limits_uses_declared_caps_not_engine_data():
    """Catalog limits times soft_ratio, clipped per joint to 1 rad/s, then summed."""
    robot = _Entity(joint_vel=torch.tensor([[10.0, 2.0, 20.0], [0.0, 4.6, 4.5]]))
    env = _Env(entities={"robot": robot})
    limits = (5.0, 5.0, 5.0)
    # Cap is 4.5. Row 0: clip(5.5,0,1)+0+clip(15.5,0,1) = 2. Row 1: 0 + 0.1 + 0.
    reward = rewards.joint_vel_limits(env, soft_ratio=0.9, limits=limits)
    assert reward.tolist() == pytest.approx([2.0, 0.1])
    assert rewards.joint_vel_limits(
        env, 0.9, (5.0,), _Cfg(joint_ids=[0])
    ).tolist() == pytest.approx([1.0, 0.0])
    with pytest.raises(RuntimeError, match="selected joints"):
        rewards.joint_vel_limits(env, 0.9, (5.0, 5.0))


def test_feet_close_xy_gauss_refuses_a_single_body():
    robot = _Entity(body_link_pos_w=torch.zeros(1, 1, 3), heading_w=torch.zeros(1))
    env = _Env(entities={"robot": robot})
    with pytest.raises(RuntimeError, match="two bodies"):
        rewards.feet_close_xy_gauss(env, threshold=0.12, asset_cfg=_Cfg(body_ids=[0]))


def test_volume_points_penetration_is_depth_times_speed() -> None:
    """Hand sum: one point at depth 0.06 / speed 1, one miss. Unregistered is refused."""
    from types import SimpleNamespace

    sensor_ref = VolumePointsRef(
        name="leg_volume_points", attach=("left_ankle_roll_link",)
    )
    offset = torch.tensor([[[[-0.06, 0.0, 0.0], [0.0, 0.0, 0.0]]]])
    vel = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]])
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(name="leg_volume_points", velocity="attach_link"),
        virtual_obstacles_registered=True,
        registered_cylinder_count=1,
        _virtual_obstacles={"known": SimpleNamespace(edges_pyt=torch.zeros(1, 6))},
        data=SimpleNamespace(penetration_offset=offset, points_vel_w=vel),
    )
    env = _Env(sensors={"leg_volume_points": sensor})
    assert rewards.volume_points_penetration(env, sensor_ref).item() == pytest.approx(
        (1.0 + 1e-6) * 0.06
    )
    sensor.virtual_obstacles_registered = False
    sensor._virtual_obstacles = {}
    sensor.registered_cylinder_count = 0
    with pytest.raises(RuntimeError, match="identically zero"):
        rewards.volume_points_penetration(env, sensor_ref)
