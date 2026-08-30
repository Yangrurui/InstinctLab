from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from instinctlab_engine.rl.termination_log import (
    PortableTerminationLogger,
)


class _TerminationManager:
    active_terms = ("fell", "timed_out")

    def __init__(self, num_envs: int):
        self.values = {
            name: torch.zeros(num_envs, dtype=torch.bool) for name in self.active_terms
        }

    def get_term(self, name: str) -> torch.Tensor:
        return self.values[name]


def _logger(num_envs: int) -> tuple[PortableTerminationLogger, _TerminationManager]:
    manager = _TerminationManager(num_envs)
    env = SimpleNamespace(num_envs=num_envs, device="cpu", termination_manager=manager)
    return PortableTerminationLogger(env), manager


def _logged(extras: dict, name: str) -> float:
    return float(extras["log"][f"Episode_Termination/{name}"])


def test_native_counts_are_replaced_by_portable_last_episode_fractions() -> None:
    logger, manager = _logger(4)
    manager.values["fell"][:] = torch.tensor([True, False, False, False])
    manager.values["timed_out"][:] = torch.tensor([False, True, False, False])
    extras = {
        "log": {
            "Episode_Termination/fell": 100,
            "Episode_Termination/timed_out": 200,
        }
    }

    logger.update(extras, torch.tensor([True, True, False, False]))

    assert _logged(extras, "fell") == pytest.approx(0.25)
    assert _logged(extras, "timed_out") == pytest.approx(0.25)


def test_completed_episode_causes_persist_until_that_environment_ends_again() -> None:
    logger, manager = _logger(4)
    extras: dict = {}

    manager.values["fell"][0] = True
    logger.update(extras, torch.tensor([True, False, False, False]))

    manager.values["fell"].zero_()
    manager.values["timed_out"][1] = True
    logger.update(extras, torch.tensor([False, True, False, False]))

    assert _logged(extras, "fell") == pytest.approx(0.25)
    assert _logged(extras, "timed_out") == pytest.approx(0.25)

    manager.values["timed_out"].zero_()
    manager.values["fell"][0] = True
    manager.values["timed_out"][0] = True
    logger.update(extras, torch.tensor([True, False, False, False]))

    assert _logged(extras, "fell") == pytest.approx(0.25)
    assert _logged(extras, "timed_out") == pytest.approx(0.5)


@pytest.mark.parametrize("num_envs", [4, 400])
def test_fraction_is_independent_of_environment_count(num_envs: int) -> None:
    logger, manager = _logger(num_envs)
    split = num_envs // 2
    manager.values["fell"][:split] = True
    manager.values["timed_out"][split:] = True

    extras: dict = {}
    logger.update(extras, torch.ones(num_envs, dtype=torch.bool))

    assert _logged(extras, "fell") == pytest.approx(0.5)
    assert _logged(extras, "timed_out") == pytest.approx(0.5)


def test_rejects_a_term_mask_with_the_wrong_environment_axis() -> None:
    logger, manager = _logger(4)
    manager.values["fell"] = torch.zeros(3, dtype=torch.bool)

    with pytest.raises(
        ValueError, match=r"termination term 'fell' must have shape \(4,\)"
    ):
        logger.update({}, torch.zeros(4, dtype=torch.bool))
