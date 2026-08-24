"""The parkour target task must construct and step on mjlab, not merely compile.

``contract_report`` only asks the registry whether a ``kind`` is registered. The
crash that shipped was a dataclass definition inside a builder body. This is the
L6 check: build the env, step it, and ask the live scene what it made.

Default ``pytest tests/`` deselects ``mjlab``-marked tests (see ``pytest.ini``).
Run on demand:

    pytest -o addopts= -m mjlab tests/test_parkour_g1_mjlab_live.py
"""

from __future__ import annotations

import torch

import pytest

from tests.parkour_live_expect import (
    MJLAB_CURRICULUM_COLUMNS,
    assert_amp_same_function,
    assert_depth_camera_miss_is_positive_infinity,
    assert_depth_camera_shape,
    assert_depth_camera_uses_base_alignment,
    assert_depth_encoder_is_fed,
    assert_depth_first_policy_obs_is_primed,
    assert_foot_scanner_miss_is_positive_infinity,
    assert_foot_scanner_sees_some_ground,
    assert_foot_scanner_shape,
    assert_known_volume_penetration,
    assert_known_volume_spin_velocity,
    assert_parkour_live_invariants,
    assert_rewards_finite_and_alive,
    assert_terrain_generated_cylinder_penetration,
    assert_volume_points_registered,
    assert_volume_points_shape,
    require_live_device,
)
from tests.parkour_obs_shapes import assert_observation_shapes_match_declaration

pytest.importorskip("mjlab")
pytestmark = pytest.mark.mjlab


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stepping mjlab needs a GPU")
def test_mjlab_parkour_target_g1_constructs_and_steps() -> None:
    device = require_live_device()

    from instinctlab.engines.mjlab import MjlabAdapter
    from instinctlab.engines.mjlab.pose_velocity import column_sub_terrain_names
    from instinctlab.tasks.parkour.config.g1 import parkour_target_g1

    spec = parkour_target_g1()
    compiled = MjlabAdapter().compile(spec, num_envs=16, device=device)
    compiled.env_cfg.seed = 12345
    env = compiled.make_env()
    try:
        env.reset()
        assert_observation_shapes_match_declaration(env, spec)
        assert_parkour_live_invariants(env, spec, compiled, expected_columns=MJLAB_CURRICULUM_COLUMNS)
        assert column_sub_terrain_names(env.scene.terrain) == list(MJLAB_CURRICULUM_COLUMNS)
        declared = env.scene.terrain.cfg.terrain_generator.num_cols
        assert declared == len(MJLAB_CURRICULUM_COLUMNS)
        assert_foot_scanner_shape(env)
        assert_foot_scanner_sees_some_ground(env)
        assert_foot_scanner_miss_is_positive_infinity(env, device=device)
        env.reset()
        assert_depth_first_policy_obs_is_primed(env)
        assert_depth_camera_shape(env)
        assert_depth_camera_uses_base_alignment(env, device=device)
        assert_depth_camera_miss_is_positive_infinity(env, device=device)
        env.reset()
        assert_amp_same_function(env, spec, device=device)
        env.reset()
        assert_volume_points_shape(env)
        assert_volume_points_registered(env)
        assert_terrain_generated_cylinder_penetration(env, device=device)
        assert_known_volume_penetration(env, device=device)
        assert_known_volume_spin_velocity(env, device=device)
        wrapper = MjlabAdapter.wrap_for_rl(env)
        dims = assert_depth_encoder_is_fed(env, spec, wrapper, compiled.agent_cfg)
        assert dims == {"policy": 896, "critic": 920}
        env.reset()
        assert_rewards_finite_and_alive(env, steps=8, device=device)
    finally:
        env.close()
