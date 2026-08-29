"""Isaac-native manager configs for motion-reference commands."""

from __future__ import annotations

from instinctlab.compat.motion_reference_commands import (
    make_motion_reference_command_classes,
)


def build_command(kind: str, params: dict):
    from isaaclab.managers import CommandTerm, CommandTermCfg
    from isaaclab.utils import configclass

    classes = make_motion_reference_command_classes(CommandTerm)

    @configclass
    class MotionReferenceCommandCfg(CommandTermCfg):
        class_type: type = classes[kind]
        motion_reference: str = params["motion_reference"]
        entity_name: str = params["entity"]
        resampling_time_range: tuple[float, float] = (1.0e4, 1.0e5)
        current_state_command: bool = params.get("current_state_command", False)
        realtime_mode: bool = params.get("realtime_mode", False)
        anchor_frame: str = params.get("anchor_frame", "robot")
        in_base_frame: bool = params.get("in_base_frame", True)
        rotation_mode: str = params.get("rotation_mode", "axis_angle")
        debug_vis: bool = False

    return MotionReferenceCommandCfg()


__all__ = ["build_command"]
