from __future__ import annotations

import torch
from types import SimpleNamespace

import pytest

from instinctlab.backends.mjlab.viser_play import MjlabViserPlayEnv, _ViserCommandBridge, _ViserRewardBridge


class _FakeRewardManager:
    def __init__(self) -> None:
        self._episode_sums = {
            ("default", "track_lin_vel_xy"): torch.tensor([1.5, 2.5]),
            ("default", "flat_orientation"): torch.tensor([0.1, 0.2]),
        }


class _FakeVelocityTerm:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(params={"ranges": {"lin_vel_x": (-0.5, 1.0)}})
        self._standing = torch.zeros(1, dtype=torch.bool)
        self._heading = torch.zeros(1, dtype=torch.bool)
        self._time_left = torch.zeros(1)
        self._command = torch.zeros(1, 3)


class _FakeCommandManager:
    def __init__(self) -> None:
        self.term_names = ("base_velocity",)
        self._terms = (("base_velocity", _FakeVelocityTerm()),)


def test_reward_bridge_exposes_episode_sums() -> None:
    bridge = _ViserRewardBridge(_FakeRewardManager())
    terms = dict(bridge.get_active_iterable_terms(1))
    assert terms["default/track_lin_vel_xy"] == [2.5]
    assert bridge.get_visualizable_terms() == []


def test_command_bridge_lists_terms_and_noops_pause() -> None:
    bridge = _ViserCommandBridge(_FakeCommandManager())
    assert bridge.active_terms == ["base_velocity"]
    bridge.on_viewer_pause(True)
    assert bridge.apply_gui_reset(torch.tensor([0])) is False


def test_viser_play_env_requires_mjlab_backend() -> None:
    fake_env = SimpleNamespace(unwrapped=SimpleNamespace(backend=object()))
    with pytest.raises(RuntimeError, match="MJLab backend"):
        MjlabViserPlayEnv(fake_env)
