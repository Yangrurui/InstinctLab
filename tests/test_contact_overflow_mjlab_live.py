"""The overflow guard must fire on a real mujoco_warp ``d.overflow`` array.

Default ``pytest tests/`` deselects ``mjlab``-marked tests. Run on demand:

    pytest -o addopts= -m mjlab tests/test_contact_overflow_mjlab_live.py
"""

from __future__ import annotations

import numpy as np
import torch

import pytest

from tests.live_device import resolve_live_device

pytest.importorskip("mjlab")
pytestmark = pytest.mark.mjlab


def _write_overflow(env, mask: int) -> None:
    import warp as wp

    overflow = env.sim.wp_data.overflow
    host = np.zeros(int(overflow.shape[0]), dtype=np.int32)
    host[0] = int(mask)
    wp.copy(overflow, wp.array(host, dtype=wp.int32, device=overflow.device))
    wp.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stepping mjlab needs a GPU")
def test_guard_fires_on_a_real_device_overflow_bit() -> None:
    """Construct the overflow on the live ``d.overflow`` the kernels write."""
    from mujoco_warp._src.types import OverflowType

    from instinctlab.engines.mjlab import MjlabAdapter
    from instinctlab_engine.diagnostics.contact_overflow import ContactOverflowError, check_contact_overflow
    from tests.task_specs import task_spec

    compiled = MjlabAdapter().compile(
        task_spec("Instinct-Parkour-Target-G1", "mjlab"),
        num_envs=4,
        device=resolve_live_device(),
    )
    compiled.env_cfg.seed = 7
    env = compiled.make_env()
    try:
        assert check_contact_overflow(env, phase="construction") is None

        _write_overflow(env, int(OverflowType.NARROWPHASE | OverflowType.NEFC))
        with pytest.raises(ContactOverflowError, match="construction overflow") as caught:
            check_contact_overflow(env, phase="construction")
        text = str(caught.value)
        assert "NARROWPHASE" in text
        assert "NEFC" in text
        assert "1 of 4 worlds" in text
        assert f"nconmax={compiled.env_cfg.sim.nconmax}" in text
        assert f"njmax={compiled.env_cfg.sim.njmax}" in text
        assert "nacon=" in text
        assert "nefc_max=" in text
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stepping mjlab needs a GPU")
def test_step_phase_reads_a_real_overflow_bit_after_physics() -> None:
    """A step from rest drops contacts; it will not set the bit. Construct one.

    nconmax=164 matches host rest ncon, but ``put_data`` only refuses ``>``,
    and the first GPU step from that pose sheds contacts (1312 → ~100). The
    kernels never set overflow. Writing the same device array they write is
    the overflow we can actually show.
    """
    from mujoco_warp._src.types import OverflowType

    from instinctlab.engines.mjlab import MjlabAdapter
    from instinctlab_engine.diagnostics.contact_overflow import (
        ContactOverflowError,
        check_contact_overflow,
        overflow_bits_set,
    )
    from tests.task_specs import task_spec

    compiled = MjlabAdapter().compile(
        task_spec("Instinct-Parkour-Target-G1", "mjlab"),
        num_envs=4,
        device=resolve_live_device(),
    )
    compiled.env_cfg.seed = 7
    env = compiled.make_env()
    try:
        actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        env.step(actions)
        assert overflow_bits_set(env) is False
        _write_overflow(env, int(OverflowType.NARROWPHASE))
        assert overflow_bits_set(env) is True
        with pytest.raises(ContactOverflowError, match="step overflow") as caught:
            check_contact_overflow(env, phase="step")
        assert "NARROWPHASE" in str(caught.value)
        assert caught.value.snapshot["any_overflow"] is True
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stepping mjlab needs a GPU")
def test_parkour_shared_terrain_budget_stays_clean_over_150_zero_action_steps() -> None:
    """Exercise the unified 0.05 terrain budget for 150 zero-action steps."""
    from instinctlab.engines.mjlab import MjlabAdapter
    from instinctlab_engine.diagnostics.contact_overflow import contact_budget_snapshot, overflow_bits_set
    from tests.task_specs import task_spec

    device = resolve_live_device()
    compiled = MjlabAdapter().compile(
        task_spec("Instinct-Parkour-Target-G1", "mjlab"),
        num_envs=16,
        device=device,
    )
    assert compiled.env_cfg.sim.nconmax == 512
    assert compiled.env_cfg.sim.njmax == 1536
    env = compiled.make_env()
    try:
        construction = contact_budget_snapshot(env)
        assert construction is not None
        assert construction["any_overflow"] is False
        actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        peak_nacon = construction["nacon"]
        peak_nefc = construction["nefc_max"]
        for _ in range(150):
            env.step(actions)
            snap = contact_budget_snapshot(env)
            assert snap is not None
            assert snap["any_overflow"] is False
            peak_nacon = max(peak_nacon, snap["nacon"])
            peak_nefc = max(peak_nefc, snap["nefc_max"])
        assert overflow_bits_set(env) is False
        # Keep meaningful headroom within the shipped per-world budgets. ``nacon`` is
        # aggregated over worlds, while ``nefc`` is already the maximum of one world.
        assert peak_nacon <= compiled.env_cfg.sim.nconmax * env.num_envs // 4
        assert peak_nefc <= compiled.env_cfg.sim.njmax // 2
    finally:
        env.close()
