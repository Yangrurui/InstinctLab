"""Pin known BackendMetadata tokens. Do not put these in production cfg.

The two tests that used to sit here went with the unified stack: one read the CLI of the retired
``train_unified`` / ``play_unified`` / ``profile_backend`` scripts, and one checked that the unified
environment refuses a backend whose joint-acceleration source the task did not accept. That refusal
lived in the environment's constructor, so nothing is left to raise it -- the surviving half is the
vocabulary itself, pinned below.
"""

from __future__ import annotations

from pathlib import Path

from instinctlab.backends.mjlab.simulator import MjlabBackend
from instinctlab.backends.mock import MockSimulatorBackend
from instinctlab.sim.backend import JOINT_ACC_SOURCES
from instinctlab.verify.scene import locomotion_flat_scene

_ISAAC_BACKEND = Path(__file__).resolve().parents[1] / "source/instinctlab/instinctlab/backends/isaacsim/backend.py"


def test_known_backend_joint_acc_tokens() -> None:
    assert MockSimulatorBackend.metadata.joint_acc_source == "fd_v1"
    assert MjlabBackend.metadata.joint_acc_source == "qacc_v1"
    assert 'joint_acc_source="isaaclab_lazy_fd_v1"' in _ISAAC_BACKEND.read_text()
    assert JOINT_ACC_SOURCES == frozenset({"qacc_v1", "isaaclab_lazy_fd_v1", "fd_v1"})
    assert locomotion_flat_scene(num_envs=2).requirements.accepted_joint_acc_sources == JOINT_ACC_SOURCES
    assert "joint_acc_source" not in MjlabBackend.metadata.physics
