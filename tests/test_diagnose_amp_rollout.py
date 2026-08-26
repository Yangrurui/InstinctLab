from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_amp_rollout.py"


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("diagnose_amp_rollout", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deterministic_actions_are_the_policy_mean(probe) -> None:
    actor = SimpleNamespace(
        std=torch.tensor([2.0, 3.0]),
        act_inference=lambda obs: obs + 1.0,
    )
    obs = torch.tensor([[4.0, 5.0]])
    generator = torch.Generator().manual_seed(7)

    got = probe.policy_action(actor, obs, "deterministic_mean", generator)

    torch.testing.assert_close(got, torch.tensor([[5.0, 6.0]]))


def test_stochastic_actions_use_an_isolated_reproducible_generator(probe) -> None:
    actor = SimpleNamespace(
        std=torch.tensor([2.0, 3.0]),
        act_inference=lambda obs: obs + 1.0,
    )
    obs = torch.tensor([[4.0, 5.0]])
    first = torch.Generator().manual_seed(7)
    second = torch.Generator().manual_seed(7)
    torch.manual_seed(123)
    global_before = torch.random.get_rng_state()

    got = probe.policy_action(actor, obs, "stochastic_sample", first)
    expected_noise = torch.randn(obs.shape, generator=second)

    torch.testing.assert_close(got, obs + 1.0 + expected_noise * actor.std)
    assert torch.equal(torch.random.get_rng_state(), global_before)


def test_unknown_action_mode_is_rejected(probe) -> None:
    actor = SimpleNamespace(std=torch.ones(1), act_inference=lambda obs: obs)
    with pytest.raises(ValueError, match="unknown policy action mode"):
        probe.policy_action(actor, torch.zeros(1, 1), "typo", torch.Generator())
