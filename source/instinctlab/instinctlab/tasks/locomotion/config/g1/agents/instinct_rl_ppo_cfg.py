"""PPO configuration for flat-ground G1 locomotion, shared by both engines.

Lives at the same path as main's copy. The values are main's; the only edit is the import of
``configclass``, which now comes from :mod:`instinctlab.utils.configclass` so that this module can
be read without Isaac Sim on the path. ``config/g1/__init__.py`` does not import this package and
does not register a Gym id, which is what keeps the robot package itself engine-free.

The hyperparameters are still checked against both references even though the env config no longer
is. Maintenance compares main's declaration with InstinctMJ's persisted ``agent.yaml`` because the
latter records the configuration that the reference run actually trained with.
"""

from instinctlab.utils.configclass import configclass
from instinctlab.utils.wrappers.instinct_rl.rl_cfg import (
    InstinctRlActorCriticCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@configclass
class PolicyCfg(InstinctRlActorCriticCfg):
    init_noise_std = 1.0
    actor_hidden_dims = [256, 128, 128]
    critic_hidden_dims = [256, 128, 128]
    activation = "elu"


@configclass
class AlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "PPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.008
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


@configclass
class NormalizersCfg:
    policy: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()


@configclass
class G1FlatPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: PolicyCfg = PolicyCfg()
    algorithm: AlgorithmCfg = AlgorithmCfg()
    normalizers: NormalizersCfg = NormalizersCfg()

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 1000
    log_interval = 10
    experiment_name = "g1_locomotion_flat"

    load_run = None

    def __post_init__(self):
        super().__post_init__()  # type: ignore
        self.resume = self.load_run is not None
        self.run_name = ""
