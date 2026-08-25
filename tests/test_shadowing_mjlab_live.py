"""Full shared shadowing reset/step lifecycle on MJLab."""

from __future__ import annotations

import torch

import pytest

pytest.importorskip("mjlab")
pytestmark = pytest.mark.mjlab


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stepping mjlab needs a GPU")
def test_mjlab_shadowing_short_rollout() -> None:
    from instinctlab.engines.mjlab import MjlabAdapter
    from instinctlab.shadowing_probe import collect_shadowing_rollout, shadowing_fallback_task
    from tests.live_device import resolve_live_device

    device = resolve_live_device()
    task = shadowing_fallback_task()
    compiled = MjlabAdapter().compile(task, num_envs=2, device=device, strict=True)
    compiled.env_cfg.seed = 2026
    env = compiled.make_env()
    try:
        state = collect_shadowing_rollout(env, task, steps=4)
        assert state["joint_pos"].shape == (5, 2, 29)
        assert state["motion_pos"].shape == (5, 2, 3)
    finally:
        env.close()
