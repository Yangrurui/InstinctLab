"""Engine-neutral G1 shadowing task declarations."""

from .beyondmimic.config.g1 import (
    g1_beyondmimic_plane,
    g1_beyondmimic_plane_play,
)
from .perceptive.config.g1 import (
    g1_perceptive_shadowing,
    g1_perceptive_shadowing_one_motion,
    g1_perceptive_shadowing_one_motion_play,
    g1_perceptive_shadowing_play,
    g1_perceptive_vae,
    g1_perceptive_vae_play,
)
from .perceptive_hoi.config.g1 import (
    g1_perceptive_hoi_shadowing,
    g1_perceptive_hoi_shadowing_play,
)
from .whole_body.config.g1 import (
    g1_plane_shadowing,
    g1_plane_shadowing_play,
)

__all__ = [
    "g1_beyondmimic_plane",
    "g1_beyondmimic_plane_play",
    "g1_perceptive_hoi_shadowing",
    "g1_perceptive_hoi_shadowing_play",
    "g1_perceptive_shadowing",
    "g1_perceptive_shadowing_one_motion",
    "g1_perceptive_shadowing_one_motion_play",
    "g1_perceptive_shadowing_play",
    "g1_perceptive_vae",
    "g1_perceptive_vae_play",
    "g1_plane_shadowing",
    "g1_plane_shadowing_play",
]
