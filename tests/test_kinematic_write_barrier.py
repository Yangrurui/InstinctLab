"""Guard: nothing may assign ArticulationState kinematic fields behind the backend's back.

The AST scan that used to stand here read the unified MDP and manager modules, which are gone.
The barrier they were scanned against is not: it is enforced at runtime by the state object
itself, which is what the remaining test exercises.
"""

from __future__ import annotations

import torch  # noqa: F401  -- ArticulationState.allocate needs torch imported

import pytest

from instinctlab.sim.state import ArticulationState, freeze_kinematic_fields


def test_write_barrier_blocks_item_assignment() -> None:
    state = ArticulationState.allocate(num_envs=2, num_joints=3, num_bodies=2, device="cpu")
    freeze_kinematic_fields(state)
    with pytest.raises(RuntimeError, match="backend.write_"):
        state.joint_pos[0] = 1.0
    with pytest.raises(RuntimeError, match="backend.write_"):
        state.body_pos_w[:, 0] = 0.0
    value = state.joint_pos[0, 0].item()
    assert value == 0.0
