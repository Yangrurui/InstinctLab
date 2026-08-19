"""Re-export of the flat G1 PPO configuration for main's Gym registration.

The definition moved to :mod:`instinctlab.tasks.locomotion.config.flat_g1_ppo`, where no engine
import stands between a caller and the hyperparameters. This module keeps the old dotted path
resolving to the same class object, so the registration below and the cross-engine task
declaration share one definition instead of two that can drift apart.
"""

from instinctlab.tasks.locomotion.config.flat_g1_ppo import AlgorithmCfg, G1FlatPPORunnerCfg, NormalizersCfg, PolicyCfg

__all__ = ["AlgorithmCfg", "G1FlatPPORunnerCfg", "NormalizersCfg", "PolicyCfg"]
