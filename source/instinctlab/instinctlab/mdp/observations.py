"""Observation terms that run unmodified under either engine's native manager.

Each function here takes the same arguments an Isaac Lab observation term takes, including a native
``SceneEntityCfg`` that the compiler has already lowered, and is passed to the engine's own
``ObservationManager``. Nothing wraps it at runtime. That is what makes a migrated task cheap: the
term bodies below are the Isaac Lab ones, with the attribute names moved onto the hub vocabulary.

Two of those moves change a number, and both are recorded here rather than in a commit message,
because a reader comparing against the Isaac Lab original will otherwise assume a typo:

**Angular velocity moved to the link spelling for free.** Isaac Lab's ``root_ang_vel_b`` is a legacy
alias for ``root_com_ang_vel_b``, and mjlab has only ``root_link_ang_vel_b``. Reading Isaac Lab's
``ArticulationData``, ``root_link_vel_w`` is a clone of ``root_com_vel_w`` whose *linear* rows alone
receive the centre-of-mass offset correction; the angular rows are copied untouched. So the two
angular quantities are bitwise identical, and using the hub spelling costs nothing against the
golden. This is what one expects physically -- a rigid body's angular velocity does not depend on
the point it is measured about -- but it is asserted here because it was checked, not because it
was assumed.

**Linear velocity does not.** The same code path adds ``ω × R(−com_pos_b)`` to the linear rows, so
``root_link_lin_vel_b`` and Isaac Lab's ``root_lin_vel_b`` differ by exactly that term whenever the
root link's centre of mass is offset from its origin, which for a humanoid torso it is. The hub
carries the link quantity because it is the one both engines can express, so :func:`base_lin_vel`
differs from the golden by that cross product. It belongs in the difference whitelist with this
reason, and it is a critic-only observation, so it does not reach the deployed policy.
"""

from __future__ import annotations

import torch
from typing import Any

from instinctlab.compat.env import RlEnv, get_command

__all__ = [
    "base_ang_vel",
    "base_lin_vel",
    "generated_commands",
    "joint_pos_rel",
    "joint_vel",
    "joint_vel_rel",
    "last_action",
    "projected_gravity",
]


def base_ang_vel(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Root angular velocity in the body frame.

    Identical to Isaac Lab's ``base_ang_vel`` value for value; see the module docstring for why the
    link spelling costs nothing here.
    """
    asset = env.scene[_name(asset_cfg)]
    return asset.data.root_link_ang_vel_b


def base_lin_vel(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Root **link** linear velocity in the body frame.

    Not the same quantity as Isaac Lab's ``base_lin_vel``, which reads the centre-of-mass alias.
    The hub carries the link quantity because it is the one both engines express.
    """
    asset = env.scene[_name(asset_cfg)]
    return asset.data.root_link_lin_vel_b


def projected_gravity(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Gravity direction in the body frame.

    The portable attitude signal. The raw gravity vectors are not portable -- Isaac Lab normalises
    the live simulation gravity into ``GRAVITY_VEC_W`` while mjlab hardcodes ``[0, 0, -1]`` under a
    lowercase name -- but this projection is spelled and computed the same on both.
    """
    asset = env.scene[_name(asset_cfg)]
    return asset.data.projected_gravity_b


def joint_pos_rel(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Joint positions relative to their defaults."""
    asset = env.scene[_name(asset_cfg)]
    joint_ids = _joint_ids(asset_cfg)
    return asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]


def joint_vel(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Joint velocities.

    Portable as-is: both engines expose ``joint_vel`` meaning the same thing. Note that
    ``joint_acc`` next door does *not* port -- Isaac Lab finite-differences it while mjlab reads
    MuJoCo's analytic ``qacc`` -- which is why there is no acceleration term in this module.
    """
    asset = env.scene[_name(asset_cfg)]
    return asset.data.joint_vel[:, _joint_ids(asset_cfg)]


def joint_vel_rel(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Joint velocities relative to their defaults."""
    asset = env.scene[_name(asset_cfg)]
    joint_ids = _joint_ids(asset_cfg)
    return asset.data.joint_vel[:, joint_ids] - asset.data.default_joint_vel[:, joint_ids]


def last_action(env: RlEnv, action_name: str | None = None) -> torch.Tensor:
    """The previous action, either the whole vector or one term's raw input.

    The named form goes through :func:`~instinctlab.compat.env.raw_action` because the two engines
    spell the attribute differently -- ``raw_actions`` against ``raw_action``, a single character.
    """
    if action_name is None:
        return env.action_manager.action
    from instinctlab.compat.env import raw_action

    return raw_action(env, action_name)


def generated_commands(env: RlEnv, command_name: str) -> torch.Tensor:
    """The current command from the named generator.

    Goes through :func:`~instinctlab.compat.env.get_command` rather than the manager directly,
    because a missing command is a ``KeyError`` on one engine and a silent ``None`` on the other.
    """
    return get_command(env, command_name)


def _name(asset_cfg: Any) -> str:
    """Entity key from a lowered ``SceneEntityCfg``, defaulting to the robot."""
    return "robot" if asset_cfg is None else asset_cfg.name


def _joint_ids(asset_cfg: Any) -> Any:
    """Joint selection from a lowered ``SceneEntityCfg``, defaulting to all joints.

    Indices rather than names on purpose: after ``resolve()`` the engines disagree about what
    ``joint_names`` holds -- Isaac Lab leaves the patterns in place, mjlab replaces them with the
    matches -- while the index lists mean the same thing on both.
    """
    return slice(None) if asset_cfg is None else asset_cfg.joint_ids
