"""Isaac-native manager configs for shared shadowing terms."""

from __future__ import annotations

from instinctlab.engines.shadowing_commands import make_shadowing_command_classes


def build_command(kind: str, params: dict):
    from isaaclab.managers import CommandTerm, CommandTermCfg
    from isaaclab.utils import configclass

    classes = make_shadowing_command_classes(CommandTerm)

    @configclass
    class ShadowingCommandCfg(CommandTermCfg):
        class_type: type = classes[kind]
        motion_reference: str = params.get("motion_reference", "motion_reference")
        entity_name: str = params.get("entity", "robot")
        resampling_time_range: tuple[float, float] = (1.0e4, 1.0e5)
        current_state_command: bool = params.get("current_state_command", False)
        realtime_mode: bool = params.get("realtime_mode", False)
        anchor_frame: str = params.get("anchor_frame", "robot")
        in_base_frame: bool = params.get("in_base_frame", True)
        rotation_mode: str = params.get("rotation_mode", "axis_angle")
        debug_vis: bool = False

    return ShadowingCommandCfg()


def mdp_function(module: str, name: str):
    """Load the retained main implementation only inside a running Isaac application."""
    import importlib

    return getattr(importlib.import_module(f"instinctlab.envs.mdp.{module}"), name)


__all__ = ["build_command", "mdp_function"]
