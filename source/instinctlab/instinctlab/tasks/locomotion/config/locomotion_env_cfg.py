"""Robot-independent velocity-tracking locomotion configuration."""

from __future__ import annotations

import math

from instinctlab_engine.spec import (
    CommandTermSpec,
    NoiseSpec,
    ObsTermSpec,
    RewardTermSpec,
    SimSpec,
)

from instinctlab.tasks.locomotion.mdp import commands, observations, rewards


class LocomotionPolicyObsCfg:
    """Policy terms that do not select a concrete robot joint or body."""

    def __init__(self) -> None:
        self.base_ang_vel = ObsTermSpec(
            func=observations.base_ang_vel,
            noise=NoiseSpec("uniform", -0.2, 0.2),
        )
        self.projected_gravity = ObsTermSpec(
            func=observations.projected_gravity,
            noise=NoiseSpec("uniform", -0.05, 0.05),
        )
        self.velocity_commands = ObsTermSpec(
            func=observations.generated_commands,
            params={"command_name": "base_velocity"},
        )


class LocomotionCriticObsCfg:
    """Critic terms that do not select a concrete robot joint or body."""

    def __init__(self) -> None:
        self.base_lin_vel = ObsTermSpec(func=observations.base_lin_vel)
        self.base_ang_vel = ObsTermSpec(func=observations.base_ang_vel)
        self.projected_gravity = ObsTermSpec(func=observations.projected_gravity)
        self.velocity_commands = ObsTermSpec(
            func=observations.generated_commands,
            params={"command_name": "base_velocity"},
        )


class LocomotionRewardsCfg:
    """Reward terms that do not select a concrete robot joint or body."""

    def __init__(self) -> None:
        self.termination_penalty = RewardTermSpec(
            func=rewards.is_terminated, weight=-200.0
        )
        self.track_lin_vel_xy_exp = RewardTermSpec(
            func=rewards.track_lin_vel_xy_yaw_frame_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.track_ang_vel_z_exp = RewardTermSpec(
            func=rewards.track_ang_vel_z_world_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.flat_orientation_l2 = RewardTermSpec(
            func=rewards.flat_orientation_l2, weight=-1.0
        )
        self.stand_still = RewardTermSpec(
            func=rewards.stand_still,
            weight=-0.8,
            params={"command_name": "base_velocity"},
        )
        self.lin_vel_z_l2 = RewardTermSpec(func=rewards.lin_vel_z_l2, weight=-0.1)
        self.action_rate_l2 = RewardTermSpec(func=rewards.action_rate_l2, weight=-0.05)


class LocomotionEnvCfg:
    """Robot-independent task values inherited once by a concrete robot config."""

    def __init__(self) -> None:
        self.sim = SimSpec(physics_dt=0.005, decimation=4, episode_length_s=20.0)
        self.commands = {
            "base_velocity": CommandTermSpec(
                func=commands.UniformVelocityCommand,
                params={
                    "entity": "robot",
                    "resampling_time_range": (10.0, 10.0),
                    "rel_standing_envs": 0.2,
                    "rel_heading_envs": 0.5,
                    "heading_command": True,
                    "heading_control_stiffness": 0.5,
                    "debug_vis": True,
                    "lin_vel_x": (-0.5, 1.0),
                    "lin_vel_y": (-0.5, 0.5),
                    "ang_vel_z": (-1.5, 1.5),
                    "heading": (-math.pi, math.pi),
                    "extended_sampling": False,
                    "rel_world_envs": 0.0,
                    "rel_forward_envs": 0.0,
                    "init_velocity_prob": 0.0,
                    "metric_velocity_anchor": "com",
                },
                engine_params={
                    "mjlab": {
                        "extended_sampling": True,
                        "rel_world_envs": 0.0,
                        "rel_forward_envs": 0.0,
                        "init_velocity_prob": 0.0,
                        "metric_velocity_anchor": "link",
                    }
                },
            )
        }
