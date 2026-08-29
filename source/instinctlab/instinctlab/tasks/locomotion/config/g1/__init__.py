"""Explicit G1 locomotion configurations and registry-boundary converters."""

from .flat_env_cfg import flat_g1
from .rl_cfgs import G1_LOCOMOTION_TRAINING_CFG
from .rough_env_cfg import rough_g1

__all__ = ["G1_LOCOMOTION_TRAINING_CFG", "flat_g1", "rough_g1"]
