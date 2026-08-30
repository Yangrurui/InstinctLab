"""The parkour target task must construct and step on Isaac Sim, not merely compile.

Kit must start before ``torch`` is imported. Default ``pytest tests/`` deselects
``isaacsim``-marked tests. Run on demand:

    pytest -o addopts= -m isaacsim tests/test_parkour_g1_isaacsim_live.py
"""

from __future__ import annotations

import pytest

from tests.isaacsim_app import ensure_isaac_app
from tests.parkour_live_expect import ISAAC_PROPORTION_COLUMNS, resolve_live_device

pytestmark = pytest.mark.isaacsim


def test_isaacsim_parkour_target_g1_constructs_and_steps() -> None:
    device = resolve_live_device()
    pytest.importorskip("isaaclab")
    ensure_isaac_app(device=device)

    from instinctlab_engine.bridge.terrain import column_sub_terrain_names
    from instinctlab_engine_isaacsim import IsaacSimAdapter
    from tests.parkour_live_expect import (
        assert_amp_same_function,
        assert_depth_camera_miss_is_positive_infinity,
        assert_depth_camera_shape,
        assert_depth_camera_uses_base_alignment,
        assert_depth_encoder_is_fed,
        assert_foot_scanner_miss_is_positive_infinity,
        assert_foot_scanner_sees_some_ground,
        assert_foot_scanner_shape,
        assert_foot_scanner_uses_yaw_alignment,
        assert_known_volume_penetration,
        assert_known_volume_spin_velocity,
        assert_parkour_live_invariants,
        assert_policy_joint_dfs_runtime_semantics,
        assert_rewards_finite_and_alive,
        assert_terrain_generated_cylinder_penetration,
        assert_volume_points_registered,
        assert_volume_points_shape,
        require_live_device,
    )
    from tests.parkour_obs_shapes import assert_observation_shapes_match_declaration
    from tests.task_specs import task_spec

    device = require_live_device()

    spec = task_spec("Instinct-Parkour-Target-G1", "isaacsim")
    compiled = IsaacSimAdapter().compile(spec, num_envs=16, device=device)
    compiled.env_cfg.seed = 12345
    env = compiled.make_env()
    try:
        env.reset()
        assert_observation_shapes_match_declaration(env, spec)
        assert_parkour_live_invariants(env, spec, compiled, expected_columns=ISAAC_PROPORTION_COLUMNS)
        assert_policy_joint_dfs_runtime_semantics(env, spec, device=device)
        assert column_sub_terrain_names(env.scene.terrain) == list(ISAAC_PROPORTION_COLUMNS)
        declared = env.scene.terrain.cfg.terrain_generator.num_cols
        assert declared == 20
        assert_foot_scanner_shape(env)
        assert_foot_scanner_sees_some_ground(env)
        assert_foot_scanner_uses_yaw_alignment(env, device=device)
        assert_foot_scanner_miss_is_positive_infinity(env, device=device)
        env.reset()
        assert_depth_camera_shape(env)
        assert_depth_camera_uses_base_alignment(env, device=device)
        assert_depth_camera_miss_is_positive_infinity(env, device=device)
        env.reset()
        assert_amp_same_function(env, spec, device=device)
        env.reset()
        assert_volume_points_shape(env, spec)
        assert_volume_points_registered(env)
        assert_terrain_generated_cylinder_penetration(env, spec, device=device)
        assert_known_volume_penetration(env, spec, device=device)
        assert_known_volume_spin_velocity(env, device=device)
        wrapper = IsaacSimAdapter.wrap_for_rl(env)
        dims = assert_depth_encoder_is_fed(env, spec, wrapper, compiled.agent_cfg)
        assert dims == {"policy": 896, "critic": 920}
        env.reset()
        assert_rewards_finite_and_alive(env, steps=8, device=device)
    finally:
        env.close()
