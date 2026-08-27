"""Engine-neutral G1 parkour task shared by Isaac Sim and MJLab.

The package does not import either engine or the agent configuration as a side effect.
"""

from .g1_parkour_target_amp_cfg import parkour_target_g1

__all__ = ["parkour_target_g1"]
