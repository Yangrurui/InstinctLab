"""Training checkpoints retain the code state and final parameters that produced them."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml
from instinctlab.training import write_run_parameters

ROOT = Path(__file__).resolve().parents[1]


class _Mode(Enum):
    TRAIN = "train"


def _observation() -> None:
    pass


@dataclass
class _EnvConfig:
    seed: int
    mode: _Mode
    terms: tuple[str, ...]
    callback: object


def test_final_environment_and_agent_parameters_use_instinctmj_layout(tmp_path: Path) -> None:
    write_run_parameters(
        tmp_path,
        _EnvConfig(seed=42, mode=_Mode.TRAIN, terms=("velocity", "height"), callback=_observation),
        {"seed": 42, "max_iterations": 1000},
    )

    env_path = tmp_path / "params" / "env.yaml"
    agent_path = tmp_path / "params" / "agent.yaml"
    assert yaml.safe_load(env_path.read_text()) == {
        "seed": 42,
        "mode": "train",
        "terms": ["velocity", "height"],
        "callback": f"{__name__}:_observation",
    }
    assert yaml.safe_load(agent_path.read_text()) == {"seed": 42, "max_iterations": 1000}


def test_training_registers_git_and_writes_parameters_before_learning() -> None:
    source = (ROOT / "scripts" / "train.py").read_text()

    git_registration = source.index("runner.add_git_repo_to_log(__file__)")
    parameter_snapshot = source.index("write_run_parameters(log_dir, compiled.env_cfg, agent_config)")
    learning = source.index("runner.learn(")
    assert git_registration < learning
    assert parameter_snapshot < learning
