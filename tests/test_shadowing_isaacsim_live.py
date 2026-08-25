"""Full shared shadowing reset/step lifecycle on Isaac Sim."""

from __future__ import annotations

import pytest

from tests.isaacsim_app import ensure_isaac_app
from tests.live_device import resolve_live_device

pytestmark = pytest.mark.isaacsim


def test_isaacsim_shadowing_short_rollout() -> None:
    device = resolve_live_device()
    pytest.importorskip("isaaclab")
    ensure_isaac_app(device=device)

    from instinctlab.engines.isaacsim import IsaacSimAdapter
    from instinctlab.shadowing_probe import collect_shadowing_rollout, shadowing_fallback_task

    task = shadowing_fallback_task()
    compiled = IsaacSimAdapter().compile(task, num_envs=2, device=device, strict=True)
    compiled.env_cfg.seed = 2026
    env = compiled.make_env()
    try:
        state = collect_shadowing_rollout(env, task, steps=4)
        assert state["joint_pos"].shape == (5, 2, 29)
        assert state["motion_pos"].shape == (5, 2, 3)
    finally:
        env.close()
