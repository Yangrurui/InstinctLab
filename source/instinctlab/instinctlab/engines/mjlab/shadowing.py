"""MJLab-native manager configs for shared shadowing terms."""

from __future__ import annotations

from dataclasses import dataclass

from instinctlab.engines.shadowing_commands import make_shadowing_command_classes


def build_command(kind: str, params: dict):
    from mjlab.managers import CommandTerm, CommandTermCfg

    classes = make_shadowing_command_classes(CommandTerm)

    @dataclass(kw_only=True)
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

        def build(self, env):
            return self.class_type(self, env)

    return ShadowingCommandCfg()


__all__ = ["build_command"]


def randomize_default_joint_pos(env, env_ids, asset_cfg, offset_distribution_params):
    import torch

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None or isinstance(joint_ids, slice):
        joint_ids = torch.arange(asset.data.default_joint_pos.shape[1], device=env.device)
    else:
        joint_ids = torch.as_tensor(joint_ids, dtype=torch.long, device=env.device).flatten()
    index = (env_ids[:, None], joint_ids[None, :])
    target = asset.data.default_joint_pos[index]
    noise = torch.empty_like(target).uniform_(*offset_distribution_params)
    asset.data.default_joint_pos[index] = target + noise
    action = env.action_manager.get_term("joint_pos")
    action._offset[index] = asset.data.default_joint_pos[index]


def randomize_ray_offsets(env, env_ids, sensor_name="camera", offset_pose_ranges=None):
    """MJLab sensor-frame calibration noise, sampled once per environment."""
    import torch

    sensor = env.scene.sensors[sensor_name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ranges = offset_pose_ranges or {}
    keys = ("x", "y", "z", "roll", "pitch", "yaw")
    bounds = torch.tensor([ranges.get(key, (0.0, 0.0)) for key in keys], device=env.device)
    sample = torch.rand(len(env_ids), 6, device=env.device) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    if hasattr(sensor, "set_offset_noise"):
        sensor.set_offset_noise(env_ids, sample)
    else:
        # Native ray sensors expose the compiled start/direction arrays. Refuse a
        # silent no-op when a different MJLab sensor implementation is selected.
        raise TypeError(f"sensor {sensor_name!r} does not expose set_offset_noise")


__all__ += ["randomize_default_joint_pos", "randomize_ray_offsets"]
