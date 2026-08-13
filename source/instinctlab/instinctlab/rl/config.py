"""Engine-neutral Instinct-RL configuration dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ActorCriticCfg:
    class_name: str = "ActorCritic"
    init_noise_std: float = 1.0
    actor_hidden_dims: tuple[int, ...] = (256, 128, 128)
    critic_hidden_dims: tuple[int, ...] = (256, 128, 128)
    activation: str = "elu"


@dataclass
class PpoAlgorithmCfg:
    class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.008
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    optimizer_class_name: str = "AdamW"
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    advantage_mixing_weights: float | tuple[float, ...] = 1.0
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    clip_min_std: float = 1.0e-12


@dataclass
class NormalizerCfg:
    class_name: str = "EmpiricalNormalization"


@dataclass
class OnPolicyRunnerCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 5000
    policy: ActorCriticCfg = field(default_factory=ActorCriticCfg)
    algorithm: PpoAlgorithmCfg = field(default_factory=PpoAlgorithmCfg)
    normalizers: dict[str, NormalizerCfg] = field(
        default_factory=lambda: {"policy": NormalizerCfg(), "critic": NormalizerCfg()}
    )
    save_interval: int = 1000
    log_interval: int = 10
    experiment_name: str = "instinctlab"
    run_name: str = ""
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    ckpt_manipulator: str | None = None
    ckpt_manipulator_kwargs: dict[str, Any] = field(default_factory=dict)
    policy_observation_group: str = "policy"
    critic_observation_group: str = "critic"
    init_at_random_ep_len: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "ActorCriticCfg",
    "NormalizerCfg",
    "OnPolicyRunnerCfg",
    "PpoAlgorithmCfg",
]
