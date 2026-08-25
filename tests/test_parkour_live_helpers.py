"""Cheap stubs for the live-check helpers, so their assertions can be mutation-tested."""

from __future__ import annotations

import torch
from types import SimpleNamespace

import pytest

from tests.parkour_live_expect import (
    CAMERA_HISTORY_FRAMES,
    CAMERA_NAME,
    CAMERA_OFFSET,
    CAMERA_OFFSET_ROT,
    CAMERA_TILT_PITCH,
    CAMERA_TILT_ROLL,
    CAMERA_TILT_YAW,
    MJLAB_CURRICULUM_COLUMNS,
    REQUIRED_COMMAND_METRICS,
    SCANNER_OFFSET,
    assert_depth_camera_miss_is_positive_infinity,
    assert_depth_camera_shape,
    assert_depth_camera_uses_base_alignment,
    assert_depth_first_policy_obs_is_primed,
    assert_foot_scanner_uses_yaw_alignment,
    assert_parkour_live_invariants,
    assert_rewards_finite_and_alive,
    scanner_origins_for_alignment,
)
from tests.parkour_obs_shapes import assert_observation_shapes_match_declaration


def test_live_invariants_pass_on_a_faithful_stub() -> None:
    n_cols = len(MJLAB_CURRICULUM_COLUMNS)
    spec = SimpleNamespace(robot=SimpleNamespace(joint_names=("hip", "knee")))
    compiled = SimpleNamespace(resolution=SimpleNamespace(skipped={}))
    command = SimpleNamespace(
        metrics={key: torch.zeros(2) for key in REQUIRED_COMMAND_METRICS},
        _column_names=list(MJLAB_CURRICULUM_COLUMNS),
    )
    action = SimpleNamespace(target_names=["hip", "knee"])
    terrain = SimpleNamespace(
        flat_patches={"target": torch.zeros(10, n_cols, 1, 3)},
        cfg=SimpleNamespace(terrain_generator=SimpleNamespace(num_cols=n_cols)),
    )
    env = SimpleNamespace(
        action_manager=SimpleNamespace(get_term=lambda _name: action),
        command_manager=SimpleNamespace(get_term=lambda _name: command),
        scene=SimpleNamespace(terrain=terrain),
    )
    assert_parkour_live_invariants(env, spec, compiled, expected_columns=MJLAB_CURRICULUM_COLUMNS)


def _obs_env(*, instant_width: int, live_width: int, history: int, flatten: bool = True):
    instant = torch.zeros(2, instant_width)
    live = torch.zeros(2, live_width)

    def func(_env, **_params):
        return instant

    term_spec = SimpleNamespace(history_length=history, params={})
    group_spec = SimpleNamespace(history_length=None, terms={"ang": term_spec})
    spec = SimpleNamespace(mdp=SimpleNamespace(observations={"policy": group_spec}))
    term_cfg = SimpleNamespace(func=func, params={}, flatten_history_dim=flatten)
    manager = SimpleNamespace(
        _group_obs_term_names={"policy": ["ang"]},
        _group_obs_term_cfgs={"policy": [term_cfg]},
        compute=lambda: {"policy": {"ang": live}},
    )
    return SimpleNamespace(observation_manager=manager), spec


def test_obs_shape_helper_accepts_term_dim_times_declared_history() -> None:
    env, spec = _obs_env(instant_width=5, live_width=20, history=4)
    assert_observation_shapes_match_declaration(env, spec)


def test_obs_shape_helper_rejects_a_dropped_history() -> None:
    env, spec = _obs_env(instant_width=5, live_width=5, history=4)
    with pytest.raises(AssertionError, match="live shape"):
        assert_observation_shapes_match_declaration(env, spec)


def test_obs_shape_helper_honors_an_explicit_group_override_of_zero() -> None:
    env, spec = _obs_env(instant_width=5, live_width=5, history=8)
    spec.mdp.observations["policy"].history_length = 0
    assert_observation_shapes_match_declaration(env, spec)


def test_obs_shape_helper_pins_concatenated_width_to_the_sum() -> None:
    env, spec = _obs_env(instant_width=5, live_width=20, history=4)
    env.observation_manager.compute = lambda: {"policy": torch.zeros(2, 20)}
    assert_observation_shapes_match_declaration(env, spec)
    env.observation_manager.compute = lambda: {"policy": torch.zeros(2, 5)}
    with pytest.raises(AssertionError, match="concatenated width"):
        assert_observation_shapes_match_declaration(env, spec)


def test_depth_first_policy_obs_helper_rejects_zeros_and_mixed_slots() -> None:
    zeros = torch.zeros(2, CAMERA_HISTORY_FRAMES, 18, 32)
    primed = torch.full((2, CAMERA_HISTORY_FRAMES, 18, 32), 0.4)
    mixed = primed.clone()
    mixed[:, 0] = 0.0

    env = SimpleNamespace(reset=lambda: ({"policy": {"depth_image": zeros}}, {}))
    with pytest.raises(AssertionError, match="all zeros"):
        assert_depth_first_policy_obs_is_primed(env)
    env.reset = lambda: ({"policy": {"depth_image": mixed}}, {})
    with pytest.raises(AssertionError, match="not the primed first frame"):
        assert_depth_first_policy_obs_is_primed(env)
    env.reset = lambda: ({"policy": {"depth_image": primed}}, {})
    assert_depth_first_policy_obs_is_primed(env)


def test_depth_shape_helper_rejects_a_dropped_frame_axis() -> None:
    raw = torch.zeros(2, 36, 64, 1)
    dropped = torch.zeros(2, 18, 32)

    class _Data:
        output = {"distance_to_image_plane": raw}

    env = SimpleNamespace(
        num_envs=2,
        scene=SimpleNamespace(sensors={CAMERA_NAME: SimpleNamespace(data=_Data())}),
        observation_manager=SimpleNamespace(compute=lambda: {"policy": {"depth_image": dropped}}),
    )
    with pytest.raises(AssertionError, match="8"):
        assert_depth_camera_shape(env)
    env.observation_manager.compute = lambda: {"policy": {"depth_image": torch.zeros(2, CAMERA_HISTORY_FRAMES, 18, 32)}}
    assert_depth_camera_shape(env)


def test_depth_shape_helper_rejects_a_terrain_only_hit_set() -> None:
    raw = torch.zeros(2, 36, 64, 1)
    image = torch.zeros(2, CAMERA_HISTORY_FRAMES, 18, 32)

    class _Data:
        output = {"distance_to_image_plane": raw}

    camera = SimpleNamespace(data=_Data(), _allowed_geom_mask=torch.tensor([True]))
    env = SimpleNamespace(
        num_envs=2,
        scene=SimpleNamespace(sensors={CAMERA_NAME: camera}),
        observation_manager=SimpleNamespace(compute=lambda: {"policy": {"depth_image": image}}),
    )
    with pytest.raises(AssertionError, match="1"):
        assert_depth_camera_shape(env)
    camera._allowed_geom_mask = torch.ones(31, dtype=torch.bool)
    assert_depth_camera_shape(env)
    del camera._allowed_geom_mask
    camera.cfg = SimpleNamespace(mesh_prim_paths=["/World/ground"])
    with pytest.raises(AssertionError, match="1"):
        assert_depth_camera_shape(env)
    camera.cfg = SimpleNamespace(mesh_prim_paths=["/World/ground", "/World/envs/env_.*/Robot/.*"])
    assert_depth_camera_shape(env)


def test_depth_miss_helper_rejects_a_finite_raw_or_a_processed_zero() -> None:
    raw = torch.full((2, 36, 64, 1), float("inf"))
    processed = torch.ones(2, 8, 18, 32)

    class _Data:
        output = {"distance_to_image_plane": raw}

    class _Robot:
        def write_root_link_pose_to_sim(self, *_a, **_k):
            return None

    class _Scene:
        def __init__(self):
            self.sensors = {CAMERA_NAME: SimpleNamespace(data=_Data(), update=lambda *_a, **_k: None)}
            self.env_origins = torch.zeros(2, 3)
            self._robot = _Robot()

        def __getitem__(self, _name):
            return self._robot

        def write_data_to_sim(self):
            return None

    env = SimpleNamespace(
        num_envs=2,
        step_dt=0.02,
        sim=None,
        scene=_Scene(),
        observation_manager=SimpleNamespace(compute=lambda: {"policy": {"depth_image": processed}}),
    )
    assert_depth_camera_miss_is_positive_infinity(env, device="cpu")
    raw[0, 0, 0] = 1.5
    with pytest.raises(AssertionError, match=r"\+inf"):
        assert_depth_camera_miss_is_positive_infinity(env, device="cpu")
    raw.fill_(float("inf"))
    raw[0, 20, 0] = 3.5
    with pytest.raises(AssertionError, match="self-hits"):
        assert_depth_camera_miss_is_positive_infinity(env, device="cpu")


def test_reward_helper_passes_when_rewards_are_finite_and_nonzero() -> None:
    reward = torch.tensor([1.0, 2.0])
    done = torch.tensor([False, False])

    def step(_actions):
        return None, reward, done, done, None

    env = SimpleNamespace(
        num_envs=2,
        action_manager=SimpleNamespace(total_action_dim=3),
        step=step,
    )
    assert_rewards_finite_and_alive(env, steps=2, device="cpu")


def _tilted_scanner_env(*, starts_alignment: str):
    """A pitched ankle plus ray starts from one alignment, for mutation checks."""
    from instinctlab.compat.math import quat_from_euler_xyz

    pos = torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]])
    quat = quat_from_euler_xyz(
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.8, 0.8]),
        torch.tensor([0.0, 0.0]),
    )
    offset = torch.tensor(SCANNER_OFFSET)
    starts = scanner_origins_for_alignment(pos, quat, offset, starts_alignment).unsqueeze(1).expand(-1, 2, -1).clone()

    class _Robot:
        joint_names = (
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
        )
        body_names = ("left_ankle_roll_link", "right_ankle_roll_link")

        def __init__(self):
            self.data = SimpleNamespace(
                default_joint_pos=torch.zeros(2, 4),
                body_link_pos_w=pos.unsqueeze(1).expand(-1, 2, -1).clone(),
                body_link_quat_w=quat.unsqueeze(1).expand(-1, 2, -1).clone(),
            )

        def write_joint_state_to_sim(self, *_a, **_k):
            return None

    class _Scene:
        def __init__(self):
            sensor = SimpleNamespace(
                _cached_world_origins=starts,
                update=lambda *_a, **_k: None,
            )
            self.sensors = {"left_height_scanner": sensor, "right_height_scanner": sensor}
            self._robot = _Robot()

        def __getitem__(self, _name):
            return self._robot

        def write_data_to_sim(self):
            return None

        def update(self, *_a, **_k):
            return None

    return SimpleNamespace(num_envs=2, step_dt=0.02, sim=None, scene=_Scene())


def test_yaw_helper_rejects_a_full_rotation_start() -> None:
    env = _tilted_scanner_env(starts_alignment="yaw")
    assert_foot_scanner_uses_yaw_alignment(env, device="cpu")
    env = _tilted_scanner_env(starts_alignment="base")
    with pytest.raises(AssertionError, match="full-R"):
        assert_foot_scanner_uses_yaw_alignment(env, device="cpu")


def _tilted_camera_env(*, pose_alignment: str):
    """A pitched torso plus a camera pose from one alignment, for mutation checks."""
    from instinctlab.compat.math import quat_from_euler_xyz
    from instinctlab.engines.ray_alignment import camera_pose_for_alignment

    pos = torch.tensor([[0.0, 0.0, 0.82], [0.0, 0.0, 0.82]])
    quat = quat_from_euler_xyz(
        torch.tensor([CAMERA_TILT_ROLL, CAMERA_TILT_ROLL]),
        torch.tensor([CAMERA_TILT_PITCH, CAMERA_TILT_PITCH]),
        torch.tensor([CAMERA_TILT_YAW, CAMERA_TILT_YAW]),
    )
    offset = torch.tensor(CAMERA_OFFSET)
    rot = torch.tensor(CAMERA_OFFSET_ROT)
    cam_pos, cam_quat = camera_pose_for_alignment(pos, quat, offset, rot, pose_alignment)

    class _Data:
        pos_w = cam_pos.clone()
        quat_w_world = cam_quat.clone()

    class _Robot:
        body_names = ("pelvis", "torso_link")
        joint_names = ("waist_yaw_joint",)

        def __init__(self):
            self.data = SimpleNamespace(
                default_joint_pos=torch.zeros(2, 1),
                body_link_pos_w=torch.stack([pos, pos], dim=1),
                body_link_quat_w=torch.stack([quat, quat], dim=1),
            )

        def write_root_link_pose_to_sim(self, *_a, **_k):
            return None

        def write_root_link_velocity_to_sim(self, *_a, **_k):
            return None

        def write_joint_state_to_sim(self, *_a, **_k):
            return None

    class _Scene:
        def __init__(self):
            sensor = SimpleNamespace(
                data=_Data(),
                cfg=SimpleNamespace(origin_offset=CAMERA_OFFSET, origin_offset_rot=CAMERA_OFFSET_ROT),
                update=lambda *_a, **_k: None,
            )
            self.sensors = {CAMERA_NAME: sensor}
            self.env_origins = torch.zeros(2, 3)
            self._robot = _Robot()

        def __getitem__(self, _name):
            return self._robot

        def write_data_to_sim(self):
            return None

        def update(self, *_a, **_k):
            return None

    return SimpleNamespace(num_envs=2, step_dt=0.02, sim=None, scene=_Scene())


def test_base_helper_rejects_a_yaw_only_camera_pose() -> None:
    env = _tilted_camera_env(pose_alignment="base")
    assert_depth_camera_uses_base_alignment(env, device="cpu")
    env = _tilted_camera_env(pose_alignment="yaw")
    with pytest.raises(AssertionError, match="yaw-only"):
        assert_depth_camera_uses_base_alignment(env, device="cpu")
