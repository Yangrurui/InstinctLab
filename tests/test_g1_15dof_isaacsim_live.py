"""Construct and step the locked-arm G1 on Isaac Sim.

Run on demand with ``pytest -o addopts= -m isaacsim``.
"""

from __future__ import annotations

import pytest

from tests.isaacsim_app import ensure_isaac_app
from tests.live_device import resolve_live_device

pytestmark = pytest.mark.isaacsim


def test_isaacsim_locked_arm_g1_constructs_with_15_actions_and_steps() -> None:
    device = resolve_live_device()
    pytest.importorskip("isaaclab")
    ensure_isaac_app(device=device)

    import torch
    from instinctlab_engine_isaacsim import IsaacSimAdapter

    from tests.task_specs import task_spec

    spec = task_spec("Instinct-Velocity-Flat-G1-15DoF", "isaacsim")
    compiled = IsaacSimAdapter().compile(
        spec,
        num_envs=4,
        device=device,
        strict=True,
    )
    env = compiled.make_env()
    try:
        env.reset()
        action_term = env.action_manager.get_term("joint_pos")
        driven = list(
            getattr(
                action_term,
                "target_names",
                getattr(action_term, "_joint_names", []),
            )
        )
        robot = env.scene["robot"]

        assert env.action_manager.total_action_dim == 15
        assert driven == list(spec.robot.joint_names)
        assert len(robot.joint_names) == 15
        assert set(robot.joint_names) == set(spec.robot.joint_names)

        actions = torch.zeros((env.num_envs, 15), device=env.device)
        for _ in range(5):
            env.step(actions)
    finally:
        env.close()
