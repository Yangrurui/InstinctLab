"""The Isaac PhysX overflow guard must fire on a real collision-stack overflow.

Default ``pytest tests/`` deselects ``isaacsim``-marked tests. Run on demand:

    INSTINCTLAB_LIVE_DEVICE=cuda:1 pytest -o addopts= -m isaacsim tests/test_isaac_contact_budget_live.py
"""

from __future__ import annotations

import pytest

from tests.isaacsim_app import ensure_isaac_app
from tests.live_device import resolve_live_device

pytest.importorskip("isaaclab")
pytestmark = pytest.mark.isaacsim


def test_guard_fires_on_real_physx_collision_stack_overflow() -> None:
    """16 envs at 2**14 overflowed on 2026-08-20; needed was 813424 bytes.

    That is the real narrow-phase stack, not a synthetic bit: the PhysX error
    asked for the same number, USD stayed at 16384, and contact force went to 0
    while ``step`` still returned.
    """
    device = resolve_live_device()
    ensure_isaac_app(device=device)

    from instinctlab.engines.isaacsim import IsaacSimAdapter
    from instinctlab_engine.diagnostics.contact_overflow import (
        ContactOverflowError,
        check_contact_overflow,
        contact_budget_snapshot,
    )
    from tests.task_specs import task_spec

    compiled = IsaacSimAdapter().compile(
        task_spec("Instinct-Parkour-Target-G1", "isaacsim"),
        num_envs=16,
        device=device,
    )
    compiled.env_cfg.seed = 7
    compiled.env_cfg.sim.physx.gpu_collision_stack_size = 2**14
    env = compiled.make_env()
    try:
        snapshot = contact_budget_snapshot(env)
        assert snapshot is not None
        assert snapshot["engine"] == "isaacsim"
        assert snapshot["gpu_collision_stack_size"] == 2**14
        assert snapshot["gpu_mem_collision_stack_size"] > 100_000
        assert snapshot["any_overflow"] is True
        with pytest.raises(ContactOverflowError, match="PhysX construction overflow") as caught:
            check_contact_overflow(env, phase="construction")
        text = str(caught.value)
        assert "collision stack" in text
        assert str(snapshot["gpu_mem_collision_stack_size"]) in text
        assert "16384" in text
        assert caught.value.snapshot["any_overflow"] is True
    finally:
        env.close()
