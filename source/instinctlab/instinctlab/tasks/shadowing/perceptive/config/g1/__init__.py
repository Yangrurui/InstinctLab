"""G1 perceptive shadowing task factories."""

from .perceptive_shadowing_cfg import (
    g1_perceptive_shadowing,
    g1_perceptive_shadowing_one_motion,
    g1_perceptive_shadowing_one_motion_play,
    g1_perceptive_shadowing_play,
)
from .perceptive_vae_cfg import g1_perceptive_vae, g1_perceptive_vae_play

__all__ = [
    "g1_perceptive_shadowing",
    "g1_perceptive_shadowing_one_motion",
    "g1_perceptive_shadowing_one_motion_play",
    "g1_perceptive_shadowing_play",
    "g1_perceptive_vae",
    "g1_perceptive_vae_play",
]
