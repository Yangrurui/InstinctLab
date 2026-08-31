"""Construct and step the locked-arm G1 on MJLab.

Run on demand with ``pytest -o addopts= -m mjlab``.
"""

from __future__ import annotations

import pytest
import torch

from tests.live_device import resolve_live_device

pytest.importorskip("mjlab")
pytestmark = pytest.mark.mjlab


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stepping mjlab needs a GPU")
def test_mjlab_locked_arm_g1_constructs_with_15_actions_and_steps() -> None:
    from instinctlab_engine_mjlab import MjlabAdapter

    from tests.task_specs import task_spec

    spec = task_spec("Instinct-Velocity-Flat-G1-15DoF", "mjlab")
    compiled = MjlabAdapter().compile(
        spec,
        num_envs=4,
        device=resolve_live_device(),
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
