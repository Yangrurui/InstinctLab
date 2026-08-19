"""The PPO configuration, pinned against both reference implementations.

Written because ``flat_g1_ppo.py`` claimed in its own docstring that this file pinned every field,
and this file did not exist. The configuration had been moved out of ``config/g1/agents/`` so it
could be read without Isaac Sim, and a move is exactly when values go missing quietly: nothing
imports a hyperparameter by name, so a dropped one shows up as a slightly different learning curve
weeks later.

Two references, checked differently. Main's values are restated here as literals, because main's
copy of this file no longer exists to compare against -- the path it lived at re-exports this
module now, so a diff against it would compare the file with itself. InstinctMJ's are read from the
``agent.yaml`` its own training run wrote, which is better evidence than its source: it is what the
reference run actually trained with.
"""

from __future__ import annotations

import glob
import yaml
from pathlib import Path

import pytest

from instinctlab.tasks.locomotion.flat_g1_ppo import G1FlatPPORunnerCfg
from instinctlab.utils.configclass import class_to_dict

# Main's values, as of the commit that moved this configuration out of the Isaac-only package
# (a6d2059). Restated rather than derived so that changing the configuration means changing this
# file too, which is the point.
MAIN = {
    "policy.init_noise_std": 1.0,
    "policy.actor_hidden_dims": [256, 128, 128],
    "policy.critic_hidden_dims": [256, 128, 128],
    "policy.activation": "elu",
    "algorithm.class_name": "PPO",
    "algorithm.value_loss_coef": 1.0,
    "algorithm.use_clipped_value_loss": True,
    "algorithm.clip_param": 0.2,
    "algorithm.entropy_coef": 0.008,
    "algorithm.num_learning_epochs": 5,
    "algorithm.num_mini_batches": 4,
    "algorithm.learning_rate": 1e-3,
    "algorithm.schedule": "adaptive",
    "algorithm.gamma": 0.99,
    "algorithm.lam": 0.95,
    "algorithm.desired_kl": 0.01,
    "algorithm.max_grad_norm": 1.0,
    "num_steps_per_env": 24,
    "max_iterations": 5000,
    "save_interval": 1000,
    "log_interval": 10,
    "experiment_name": "g1_locomotion_flat",
}

REFERENCE_RUN = "/root/InstinctMJ/logs/instinct_rl/g1_locomotion_flat/*/params/agent.yaml"


def _flatten(tree: dict, prefix: str = "") -> dict:
    flat: dict = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


@pytest.fixture(scope="module")
def declared() -> dict:
    return _flatten(class_to_dict(G1FlatPPORunnerCfg()))


def test_every_hyperparameter_is_mains(declared) -> None:
    for key, value in MAIN.items():
        assert declared[key] == value, key


def test_nothing_was_dropped_in_the_move(declared) -> None:
    """A field main had that this no longer has would not fail above -- it would be absent."""
    missing = [key for key in MAIN if key not in declared]
    assert not missing, f"{missing} disappeared from the configuration"


def test_the_run_this_trains_matches_the_reference_run(declared) -> None:
    """Against the yaml InstinctMJ's own 5000-iteration run wrote, not against its source.

    The reference's runner declares many more fields than this one -- MoE, VAE, distillation and an
    AMP discriminator -- and every one of them is inert in that run. Only the fields both declare
    are compared; a field only it has is checked to be switched off, since a reference that was
    quietly doing something extra would make the two runs incomparable however well the shared
    fields agreed.
    """
    matches = sorted(glob.glob(REFERENCE_RUN))
    if not matches:
        pytest.skip("no InstinctMJ training run to compare against")
    reference = _flatten(yaml.safe_load(Path(matches[-1]).read_text()))

    shared = {key: value for key, value in reference.items() if key in declared}
    assert len(shared) >= 30, f"only {len(shared)} fields in common; the reference's format changed"
    differing = {key: (value, declared[key]) for key, value in shared.items() if declared[key] != value}
    assert not differing, f"the two runs differ: {differing}"

    extra_features = ("kl_loss", "teacher", "distill", "discriminator", "moe", "vae", "encoder")
    inert = (None, 0, 0.0, False, "", {}, [])
    switched_on = {
        key: value
        for key, value in reference.items()
        if key not in declared and any(part in key for part in extra_features) and value not in inert
    }
    assert not switched_on, f"the reference run enabled {switched_on}, so it is not the same algorithm"
