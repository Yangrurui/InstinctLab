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
from test_mdp_terms import _Cfg, _Entity, _Env, _Sensor

import instinctlab.mdp as mdp
from instinctlab.spec.sensor import ContactSensorRef, RayCasterRef, RayPatternRef, VolumePointsRef


class _CommandTerm:
    def __init__(self, metrics):
        self.metrics = metrics


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
        self.terrain_levels = levels if levels is not None else torch.tensor([3.0, 3.0, 3.0])
        self.calls: list = []

    def update_env_origins(self, env_ids, move_up, move_down):
        self.calls.append((env_ids.clone(), move_up.clone(), move_down.clone()))


def test_is_alive_is_the_complement_of_is_terminated():
    env = _Env(terminated=torch.tensor([True, False]))
    assert mdp.is_alive(env).tolist() == pytest.approx([0.0, 1.0])
    assert (mdp.is_alive(env) + mdp.is_terminated(env)).tolist() == pytest.approx([1.0, 1.0])


def test_track_lin_vel_xy_exp_is_the_base_frame_kernel():
    robot = _Entity(root_link_lin_vel_b=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    env = _Env(entities={"robot": robot}, commands={"base_velocity": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])})
    reward = mdp.track_lin_vel_xy_exp(env, "base_velocity", std=0.5)
    assert reward[0].item() == pytest.approx(1.0)
    assert reward[1].item() == pytest.approx(math.exp(-1.0 / 0.25))


def test_track_ang_vel_z_exp_is_the_base_frame_yaw_kernel():
    robot = _Entity(root_link_ang_vel_b=torch.tensor([[0.0, 0.0, 0.3]]))
    env = _Env(entities={"robot": robot}, commands={"base_velocity": torch.tensor([[0.0, 0.0, 0.5]])})
    reward = mdp.track_ang_vel_z_exp(env, "base_velocity", std=0.5).item()
    assert reward == pytest.approx(math.exp(-(0.2**2) / 0.25))


def test_ang_vel_xy_l2_and_joint_vel_l2():
    robot = _Entity(
        root_link_ang_vel_b=torch.tensor([[0.3, 0.4, 0.5]]),
        joint_vel=torch.tensor([[1.0, 2.0, 3.0]]),
    )
    env = _Env(entities={"robot": robot})
    assert mdp.ang_vel_xy_l2(env).item() == pytest.approx(0.25)
    assert mdp.joint_vel_l2(env).item() == pytest.approx(14.0)
    assert mdp.joint_vel_l2(env, _Cfg(joint_ids=[0, 1])).item() == pytest.approx(5.0)


def test_stand_still_when_idle_subtracts_the_offset_and_uses_its_own_gate():
    robot = _Entity(joint_pos=torch.tensor([[1.0, 1.0]] * 2), default_joint_pos=torch.zeros(2, 2))
    env = _Env(entities={"robot": robot}, commands={"base_velocity": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])})
    assert mdp.stand_still_when_idle(env, "base_velocity", offset=1.0).tolist() == pytest.approx([1.0, 0.0])
    # The existing stand_still is unchanged: no offset, 0.1 gate.
    assert mdp.stand_still(env, "base_velocity").tolist() == pytest.approx([2.0, 0.0])


def test_heading_error_is_the_abs_yaw_command():
    env = _Env(commands={"base_velocity": torch.tensor([[1.0, 0.0, -0.4], [0.0, 0.0, 0.0]])})
    assert mdp.heading_error(env, "base_velocity").tolist() == pytest.approx([0.4, 0.0])


def test_dont_wait_counts_how_far_below_the_forward_thresholds():
    robot = _Entity(root_link_lin_vel_b=torch.tensor([[0.10, 0.0, 0.0], [-0.20, 0.0, 0.0], [-0.20, 0.0, 0.0]]))
    env = _Env(
        entities={"robot": robot},
        commands={"base_velocity": torch.tensor([[0.4, 0.0, 0.0], [0.4, 0.0, 0.0], [0.2, 0.0, 0.0]])},
    )
    assert mdp.dont_wait(env, "base_velocity").tolist() == pytest.approx([1.0, 3.0, 0.0])


def test_parkour_feet_air_time_does_not_clamp_and_gates_on_yaw_too():
    sensor_ref = ContactSensorRef(name="feet", elements=(".*foot",), track_air_time=True)
    sensor = _Sensor(
        body_names=["left_foot", "right_foot"],
        current_air_time=torch.tensor([[9.0, 0.0], [0.3, 0.0], [0.3, 0.0]]),
        current_contact_time=torch.tensor([[0.0, 3.0], [0.0, 0.2], [0.0, 0.2]]),
    )
    env = _Env(
        sensors={"feet": sensor},
        commands={"base_velocity": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.4]])},
    )
    reward = mdp.feet_air_time(env, "base_velocity", sensor_ref, vel_threshold=0.15)
    # Row 0 is uncapped single-stance contact time. Row 1 is a near-zero command. Row 2 is yaw.
    assert reward.tolist() == pytest.approx([3.0, 0.0, 0.2])


def test_parkour_feet_air_time_optional_threshold_caps_like_the_biped_term():
    sensor_ref = ContactSensorRef(name="feet", elements=(".*foot",), track_air_time=True)
    sensor = _Sensor(
        body_names=["left_foot", "right_foot"],
        current_air_time=torch.tensor([[9.0, 0.0]]),
        current_contact_time=torch.tensor([[0.0, 3.0]]),
    )
    env = _Env(sensors={"feet": sensor}, commands={"base_velocity": torch.tensor([[1.0, 0.0, 0.0]])})
    assert mdp.feet_air_time(
        env, "base_velocity", sensor_ref, vel_threshold=0.15, threshold=0.5
    ).item() == pytest.approx(0.5)


def test_joint_deviation_square():
    robot = _Entity(joint_pos=torch.tensor([[1.0, -2.0, 0.5]]), default_joint_pos=torch.zeros(1, 3))
    env = _Env(entities={"robot": robot})
    assert mdp.joint_deviation_square(env).item() == pytest.approx(1.0 + 4.0 + 0.25)
    assert mdp.joint_deviation_square(env, _Cfg(joint_ids=[1])).item() == pytest.approx(4.0)


def test_link_orientation_is_zero_when_the_link_is_upright():
    robot = _Entity(
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        body_link_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
    )
    env = _Env(entities={"robot": robot})
    assert mdp.link_orientation(env, _Cfg(body_ids=[0])).item() == pytest.approx(0.0)


def test_link_orientation_penalises_a_90_degree_roll():
    half = math.sqrt(2.0) / 2.0
    robot = _Entity(
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        body_link_quat_w=torch.tensor([[[half, half, 0.0, 0.0]]]),
    )
    env = _Env(entities={"robot": robot})
    assert mdp.link_orientation(env, _Cfg(body_ids=[0])).item() == pytest.approx(1.0)


def test_feet_orientation_contact_is_gated_on_contact_not_force():
    half = math.sqrt(2.0) / 2.0
    sensor_ref = ContactSensorRef(name="feet", elements=(".*foot",), track_air_time=True)
    sensor = _Sensor(
        body_names=["left_foot", "right_foot"],
        current_contact_time=torch.tensor([[0.4, 0.0]]),
    )
    robot = _Entity(
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        body_link_quat_w=torch.tensor([[[half, half, 0.0, 0.0], [half, half, 0.0, 0.0]]]),
    )
    env = _Env(entities={"robot": robot}, sensors={"feet": sensor})
    # Both feet are rolled 90 degrees; only the left one is in contact.
    assert mdp.feet_orientation_contact(env, sensor_ref, _Cfg(body_ids=[0, 1])).item() == pytest.approx(1.0)


def test_feet_close_xy_gauss_pays_when_the_feet_are_apart_in_y():
    robot = _Entity(
        body_link_pos_w=torch.tensor([[[0.0, 0.10, 0.0], [0.0, -0.10, 0.0]]]),
        heading_w=torch.tensor([0.0]),
    )
    env = _Env(entities={"robot": robot})
    cfg = _Cfg(body_ids=[0, 1])
    assert mdp.feet_close_xy_gauss(env, threshold=0.12, std=0.1, asset_cfg=cfg).item() == pytest.approx(0.0)
    robot.data.body_link_pos_w = torch.tensor([[[0.0, 0.02, 0.0], [0.0, -0.02, 0.0]]])
    expected = math.exp(-0.08 / 0.01) - 1.0
    assert mdp.feet_close_xy_gauss(env, threshold=0.12, std=0.1, asset_cfg=cfg).item() == pytest.approx(expected)


class _RaySensor:
    def __init__(self, hits):
        self.data = type("Data", (), {"ray_hits_w": torch.tensor(hits)})()


def test_feet_at_plane_penalises_a_stance_foot_above_the_scan() -> None:
    contact = ContactSensorRef(name="feet", elements=(".*foot",), track_air_time=True)
    left = RayCasterRef(name="left_height_scanner", attach="left_foot")
    right = RayCasterRef(name="right_height_scanner", attach="right_foot")
    robot = _Entity(body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.20], [0.0, 0.0, 0.20]]]))
    ground = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    env = _Env(
        entities={"robot": robot},
        sensors={
            "feet": _Sensor(body_names=["left_foot", "right_foot"], current_contact_time=torch.tensor([[0.4, 0.4]])),
            "left_height_scanner": _RaySensor(ground),
            "right_height_scanner": _RaySensor(ground),
        },
    )
    # 0.20 - 0.00 - 0.05 = 0.15 per ray, two rays, two feet.
    reward = mdp.feet_at_plane(env, contact, left, right, asset_cfg=_Cfg(body_ids=[0, 1]), height_offset=0.05)
    assert reward.item() == pytest.approx(0.60)


def test_feet_at_plane_a_miss_is_not_ground_at_zero() -> None:
    """``hit_z = inf`` clamps to zero. The old ``inf → 0`` path would pay 0.3 here."""
    contact = ContactSensorRef(name="feet", elements=(".*foot",), track_air_time=True)
    left = RayCasterRef(name="left_height_scanner", attach="left_foot")
    right = RayCasterRef(name="right_height_scanner", attach="right_foot")
    robot = _Entity(body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.80], [0.0, 0.0, 0.80]]]))
    miss = float("inf")
    gone = [[[miss, miss, miss], [miss, miss, miss]]]
    env = _Env(
        entities={"robot": robot},
        sensors={
            "feet": _Sensor(body_names=["left_foot", "right_foot"], current_contact_time=torch.tensor([[0.4, 0.4]])),
            "left_height_scanner": _RaySensor(gone),
            "right_height_scanner": _RaySensor(gone),
        },
    )
    reward = mdp.feet_at_plane(env, contact, left, right, asset_cfg=_Cfg(body_ids=[0, 1]), height_offset=0.058)
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

    delay0 = _fill(mdp.DelayedDepthImage(cfg, env), delay=0)
    # Written depths are 0.05, 0.10, ..., 1.85. Processed = depth / 2.5.
    # delay=0 indices: [1, 6, 11, 16, 21, 26, 31, 36] → depths 0.10 … 1.85
    assert tuple(delay0.shape) == (2, 8, 4, 4)
    assert delay0[:, -1].tolist()[0][0][0] == pytest.approx(1.85 / 2.5)
    assert delay0[:, 0].tolist()[0][0][0] == pytest.approx(0.10 / 2.5)
    delay1 = _fill(mdp.DelayedDepthImage(cfg, env), delay=1)
    assert delay1[:, -1].tolist()[0][0][0] == pytest.approx(1.80 / 2.5)
    assert delay1[:, 0].tolist()[0][0][0] == pytest.approx(0.05 / 2.5)


def _delayed_depth_term(num_envs: int = 3, history_length: int = 37):
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
    cfg = SimpleNamespace(
        params={
            "sensor": sensor,
            "history_skip_frames": 5,
            "num_output_frames": 8,
            "delayed_frame_ranges": (0, 1),
            "history_length": history_length,
            "blur_kernel_size": 1,
            "blur_sigma": 0.0,
        }
    )
    term = mdp.DelayedDepthImage(cfg, env)
    return term, env, sensor, raw


def test_delayed_depth_partial_reset_clears_only_selected_envs() -> None:
    term, env, sensor, raw = _delayed_depth_term(num_envs=3)
    raw.fill_(1.25)
    for _ in range(8):
        term(env, sensor)
    kept = term._history[0].clone()
    assert float(term._history.abs().max()) > 0.0
    write_before = term._write
    term.reset(env_ids=torch.tensor([1, 2], dtype=torch.int32))
    assert torch.equal(term._history[0], kept)
    assert float(term._history[1].abs().max()) == 0.0
    assert float(term._history[2].abs().max()) == 0.0
    assert term._write == write_before


def test_delayed_depth_full_reset_clears_every_env() -> None:
    term, env, sensor, raw = _delayed_depth_term(num_envs=2)
    raw.fill_(0.8)
    for _ in range(4):
        term(env, sensor)
    term.reset()
    assert float(term._history.abs().max()) == 0.0
    term.reset(env_ids=None)
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
    term.reset(env_ids=torch.tensor(1, dtype=torch.int64))
    assert float(term._history[1].abs().max()) == 0.0
    term._history[0] = 0.2
    cpu_ids = torch.tensor([0], device="cpu", dtype=torch.int32)
    term.reset(env_ids=cpu_ids)
    assert float(term._history[0].abs().max()) == 0.0


def test_delayed_depth_reset_keeps_delay_in_declared_range() -> None:
    term, _env, _sensor, _raw = _delayed_depth_term(num_envs=8)
    lo, hi = term.delayed_frame_ranges
    term.reset(env_ids=torch.arange(8))
    assert int(term._delay.min()) >= int(lo)
    assert int(term._delay.max()) <= int(hi)


def test_delayed_depth_first_output_after_reset_has_no_old_frames() -> None:
    """After reset, sampled slots that are not the just-written frame are zeros, not the old episode."""
    term, env, sensor, raw = _delayed_depth_term(num_envs=2)
    raw.fill_(2.0)
    for _ in range(37):
        term(env, sensor)
    old = term(env, sensor)
    assert float(old.abs().max()) > 0.0
    term.reset(env_ids=torch.tensor([0]))
    term._delay[0] = 0
    raw.fill_(1.25)
    first = term(env, sensor)
    processed_new = 1.25 / 2.5
    assert first[0, -1, 0, 0].item() == pytest.approx(processed_new)
    assert float(first[0, :-1].abs().max()) == 0.0
    assert float(first[1].abs().max()) > 0.0


def test_undesired_contacts_counts_touches_without_a_newton_threshold():
    sensor_ref = ContactSensorRef(name="body", elements=(".*",), track_air_time=True)
    sensor = _Sensor(
        body_names=["torso", "pelvis", "head"],
        current_contact_time=torch.tensor([[0.0, 0.4, 0.0], [0.2, 0.3, 0.1]]),
    )
    env = _Env(sensors={"body": sensor})
    assert mdp.undesired_contacts(env, sensor_ref).tolist() == pytest.approx([1.0, 3.0])
    assert "threshold" not in inspect.signature(mdp.undesired_contacts).parameters


def test_bad_orientation_uses_projected_gravity():
    upright = _Entity(projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]))
    tilted = _Entity(projected_gravity_b=torch.tensor([[0.0, math.sin(1.2), -math.cos(1.2)]]))
    assert mdp.bad_orientation(_Env(entities={"robot": upright}), limit_angle=1.0).tolist() == [False]
    assert mdp.bad_orientation(_Env(entities={"robot": tilted}), limit_angle=1.0).tolist() == [True]
    assert mdp.bad_orientation(_Env(entities={"robot": tilted}), limit_angle=1.5).tolist() == [False]


def test_root_height_below_env_origin_minimum_clamps_the_origin():
    robot = _Entity(root_link_pos_w=torch.tensor([[0.0, 0.0, 0.3], [0.0, 0.0, -0.6], [0.0, 0.0, 0.3]]))
    env = _Env(entities={"robot": robot})
    env.scene.env_origins = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 2.0]])
    assert mdp.root_height_below_env_origin_minimum(env, 0.5).tolist() == [False, True, True]


def test_terrain_out_of_bounds_uses_the_whole_map():
    # size 8x8, 10 rows, 20 cols, border 3 → half extents (43, 83); buffer 2 → limits 41, 81.
    robot = _Entity(root_link_pos_w=torch.tensor([[0.0, 0.0, 0.8], [42.0, 0.0, 0.8], [0.0, 82.0, 0.8]]))
    env = _Env(entities={"robot": robot})
    env.scene.terrain = _Terrain(generator=_Generator())
    assert mdp.terrain_out_of_bounds(env, distance_buffer=2.0).tolist() == [False, True, True]


def test_terrain_out_of_bounds_refuses_a_plane():
    env = _Env()
    env.scene.terrain = _Terrain(generator=None)
    env.scene["robot"] = _Entity(root_link_pos_w=torch.zeros(1, 3))
    with pytest.raises(RuntimeError, match="no generator"):
        mdp.terrain_out_of_bounds(env, distance_buffer=2.0)


def test_terrain_out_of_bounds_refuses_a_generator_missing_a_field():
    env = _Env()
    env.scene["robot"] = _Entity(root_link_pos_w=torch.zeros(1, 3))
    env.scene.terrain = _Terrain(generator=type("Bare", (), {"size": (8.0, 8.0)})())
    with pytest.raises(RuntimeError, match="num_rows"):
        mdp.terrain_out_of_bounds(env, distance_buffer=2.0)


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
    env.scene.terrain = _Terrain(generator=_Generator(), levels=torch.tensor([3.0, 3.0, 3.0]))
    mean = mdp.tracking_exp_vel(env, torch.tensor([0, 1, 2]), "base_velocity")
    assert mean.item() == pytest.approx(3.0)
    _, move_up, move_down = env.scene.terrain.calls[0]
    assert move_up.tolist() == [True, False, False]
    assert move_down.tolist() == [False, True, False]


def test_tracking_exp_vel_refuses_a_plane():
    env = _Env()
    env.scene.terrain = _Terrain(generator=None)
    with pytest.raises(RuntimeError, match="no generator"):
        mdp.tracking_exp_vel(env, torch.tensor([0]), "base_velocity")


def test_tracking_exp_vel_lists_available_metrics_when_a_key_is_missing():
    env = _Env()
    env.scene.terrain = _Terrain(generator=_Generator())
    env.command_manager = _CommandManager(
        {"base_velocity": torch.zeros(1, 3)},
        terms={"base_velocity": _CommandTerm({"tracking_exp_vel_xy": torch.tensor([0.5])})},
    )
    with pytest.raises(RuntimeError, match="tracking_exp_vel_yaw") as caught:
        mdp.tracking_exp_vel(env, torch.tensor([0]), "base_velocity")
    assert "tracking_exp_vel_xy" in str(caught.value)


def test_joint_vel_limits_uses_declared_caps_not_engine_data():
    """Catalog limits times soft_ratio, clipped per joint to 1 rad/s, then summed."""
    robot = _Entity(joint_vel=torch.tensor([[10.0, 2.0, 20.0], [0.0, 4.6, 4.5]]))
    env = _Env(entities={"robot": robot})
    limits = (5.0, 5.0, 5.0)
    # Cap is 4.5. Row 0: clip(5.5,0,1)+0+clip(15.5,0,1) = 2. Row 1: 0 + 0.1 + 0.
    reward = mdp.joint_vel_limits(env, soft_ratio=0.9, limits=limits)
    assert reward.tolist() == pytest.approx([2.0, 0.1])
    assert mdp.joint_vel_limits(env, 0.9, (5.0,), _Cfg(joint_ids=[0])).tolist() == pytest.approx([1.0, 0.0])
    with pytest.raises(RuntimeError, match="selected joints"):
        mdp.joint_vel_limits(env, 0.9, (5.0, 5.0))


def test_feet_close_xy_gauss_refuses_a_single_body():
    robot = _Entity(body_link_pos_w=torch.zeros(1, 1, 3), heading_w=torch.zeros(1))
    env = _Env(entities={"robot": robot})
    with pytest.raises(RuntimeError, match="two bodies"):
        mdp.feet_close_xy_gauss(env, threshold=0.12, asset_cfg=_Cfg(body_ids=[0]))


def test_volume_points_penetration_is_depth_times_speed() -> None:
    """Hand sum: one point at depth 0.06 / speed 1, one miss. Unregistered is refused."""
    from types import SimpleNamespace

    sensor_ref = VolumePointsRef(name="leg_volume_points", attach=("left_ankle_roll_link",))
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
    assert mdp.volume_points_penetration(env, sensor_ref).item() == pytest.approx((1.0 + 1e-6) * 0.06)
    sensor.virtual_obstacles_registered = False
    sensor._virtual_obstacles = {}
    sensor.registered_cylinder_count = 0
    with pytest.raises(RuntimeError, match="identically zero"):
        mdp.volume_points_penetration(env, sensor_ref)
