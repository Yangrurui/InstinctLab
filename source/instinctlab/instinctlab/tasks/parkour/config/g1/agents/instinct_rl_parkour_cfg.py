"""AMP runner for proprioceptive G1 parkour, shared by both engines.

WasabiPPO plus a 4-expert MoE whose Conv2d depth encoder consumes the
``depth_image`` component the task declares. Hyperparameters match the legacy
Isaac ``instinct_rl_amp_cfg.py``. This file stays engine-free: vendored
``configclass``, no Isaac imports.

The discriminator lives in ``instinct_rl`` and operates on flattened tensors
from ``amp_policy`` / ``amp_reference``. It does not import an engine.

``component_names=["depth_image"]`` must resolve against the observation group
the task declares. A wired encoder that is never fed is invisible: shapes are
fine, training runs, the policy is blind. ``takeout_input_components`` is True
so the 8×18×32 image is not also flattened into the MLP.
"""

from instinctlab.utils.configclass import configclass
from instinctlab.utils.wrappers.instinct_rl.module_cfg import InstinctRlConv2dHeadCfg
from instinctlab.utils.wrappers.instinct_rl.rl_cfg import (
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@configclass
class DepthEncoderConv2dCfg(InstinctRlConv2dHeadCfg):
    output_size = 128
    channels = [4]
    kernel_sizes = [3]
    strides = [1]
    hidden_sizes = [256, 256]
    paddings = [1]
    nonlinearity = "ReLU"
    use_maxpool = True
    component_names = ["depth_image"]
    takeout_input_components = True


@configclass
class EncoderConfigs:
    depth_encoder = DepthEncoderConv2dCfg()


@configclass
class MoEPolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
    init_noise_std = 1.0
    num_moe_experts = 4
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"
    encoder_configs = EncoderConfigs()
    critic_encoder_configs = EncoderConfigs()


@configclass
class AmpAlgoCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "WasabiPPO"
    actor_state_key = "amp_policy"
    reference_state_key = "amp_reference"
    discriminator_kwargs = {
        "hidden_sizes": [1024, 512],
        "nonlinearity": "ReLU",
    }
    discriminator_reward_coef = 0.25
    discriminator_reward_type = "quad"
    discriminator_loss_func = "MSELoss"
    discriminator_gradient_penalty_coef = 5.0
    discriminator_optimizer_class_name = "AdamW"
    discriminator_weight_decay_coef = 3e-4
    discriminator_logit_weight_decay_coef = 0.04
    discriminator_optimizer_kwargs = {
        "lr": 1.0e-4,
        "betas": [0.9, 0.999],
    }
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.006
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
class G1ParkourTargetPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: MoEPolicyCfg = MoEPolicyCfg()
    algorithm: AmpAlgoCfg = AmpAlgoCfg()
    normalizers: NormalizersCfg = NormalizersCfg()

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 5000
    log_interval = 1
    experiment_name = "g1_parkour"

    load_run = None

    def __post_init__(self):
        super().__post_init__()  # type: ignore
        self.resume = self.load_run is not None
        self.run_name = ""
