"""Pin known BackendMetadata tokens. Do not put these in production cfg."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from instinctlab.backends.mjlab.simulator import MjlabBackend
from instinctlab.backends.mock import MockSimulatorBackend
from instinctlab.envs.unified_manager_based_rl_env import UnifiedManagerBasedRLEnv
from instinctlab.sim.backend import JOINT_ACC_SOURCES
from instinctlab.tasks.locomotion.unified_flat_env_cfg import locomotion_flat_env_cfg

_ISAAC_BACKEND = Path(__file__).resolve().parents[1] / "source/instinctlab/instinctlab/backends/isaacsim/backend.py"


def test_known_backend_joint_acc_tokens() -> None:
    assert MockSimulatorBackend.metadata.joint_acc_source == "fd_v1"
    assert MjlabBackend.metadata.joint_acc_source == "qacc_v1"
    assert 'joint_acc_source="isaaclab_lazy_fd_v1"' in _ISAAC_BACKEND.read_text()
    assert JOINT_ACC_SOURCES == frozenset({"qacc_v1", "isaaclab_lazy_fd_v1", "fd_v1"})
    assert locomotion_flat_env_cfg(num_envs=2).requirements.accepted_joint_acc_sources == JOINT_ACC_SOURCES
    assert "joint_acc_source" not in MjlabBackend.metadata.physics


def test_unified_cli_backend_choices_follow_registry() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "scripts/instinct_rl/train_unified.py",
        "scripts/instinct_rl/play_unified.py",
        "scripts/profile_backend.py",
    ):
        text = (root / rel).read_text()
        assert "choices=BACKENDS.names()" in text, rel
        assert 'choices=("isaacsim", "mjlab", "mock")' not in text


def test_accepted_joint_acc_source_rejects_unknown() -> None:
    cfg = locomotion_flat_env_cfg(num_envs=2)
    backend = MockSimulatorBackend(device="cpu")
    backend.metadata = replace(backend.metadata, joint_acc_source="not_a_source")
    with pytest.raises(RuntimeError, match="joint_acc_source"):
        UnifiedManagerBasedRLEnv(cfg, backend)
