"""Engine-neutral G1 parkour task shared by Isaac Sim and MJLab.

The package does not import either engine or the agent configuration as a side effect.
"""

from .g1_parkour_target_amp_cfg import (
    FEET_CONTACT,
    TORSO_CONTACT,
    UNDESIRED_CONTACT,
    G1ParkourEnvCfg,
    parkour_target_g1,
)

__all__ = ["FEET_CONTACT", "TORSO_CONTACT", "UNDESIRED_CONTACT", "G1ParkourEnvCfg", "parkour_target_g1"]
