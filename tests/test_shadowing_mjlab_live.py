"""Full shared shadowing reset/step lifecycle on MJLab."""

from __future__ import annotations

import torch

import pytest

pytest.importorskip("mjlab")
pytestmark = pytest.mark.mjlab


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stepping mjlab needs a GPU")
def test_mjlab_shadowing_short_rollout() -> None:
    from instinctlab_engine_mjlab import MjlabAdapter
    from instinctlab.shadowing_probe import collect_shadowing_rollout, shadowing_task_with_motion
    from tests.live_device import resolve_live_device
    from tests.shadowing_live import resolve_shadowing_motion

    device = resolve_live_device()
    task = shadowing_task_with_motion("Instinct-Shadowing-WholeBody-Plane-G1-v0", resolve_shadowing_motion())
    compiled = MjlabAdapter().compile(task, num_envs=2, device=device, strict=True)
    compiled.env_cfg.seed = 2026
    env = compiled.make_env()
    try:
        state = collect_shadowing_rollout(env, task, steps=4)
        assert state["joint_pos"].shape == (5, 2, 29)
        assert state["motion_pos"].shape == (5, 2, 3)
        torch.testing.assert_close(
            torch.from_numpy(state["joint_pos"][0]),
            torch.from_numpy(state["motion_joint_pos"][0]),
        )
    finally:
        env.close()
