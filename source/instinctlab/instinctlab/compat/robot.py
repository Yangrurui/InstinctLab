"""Engine-neutral robot reads used by task-owned MDP formulas.

This module normalizes native API shape and naming only.  It deliberately preserves the quantity
chosen by each engine: Isaac uses ``applied_torque`` and body COM velocities, while MJLab uses
``qfrc_actuator`` and body-link velocities.  Thresholds, clipping, squaring and reduction remain
in the task that defines the reward or termination.

No engine SDK is imported, so task packages can depend on this boundary without depending on an
engine implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from .denylist import PortabilityError
from .env import env_engine

__all__ = [
    "body_angular_velocity_w",
    "body_linear_velocity_w",
    "joint_acceleration",
    "joint_applied_torque",
    "joint_effort_limits",
    "joint_stiffness_groups",
    "root_command_linear_velocity_b",
]


def _native_engine(env: Any, asset: Any) -> str:
    try:
        return env_engine(env)
    except PortabilityError:
        # Fixed-state tests and lightweight probes use SDK-free stand-ins.  These attributes are
        # unambiguous because the engines intentionally expose different actuator quantities.
        if getattr(asset.data, "qfrc_actuator", None) is not None:
            return "mjlab"
        if getattr(asset.data, "applied_torque", None) is not None:
            return "isaacsim"
        raise


def joint_applied_torque(env: Any, asset: Any) -> torch.Tensor:
    """Return the engine's native joint-space actuator effort ``(env, joint)``.

    This normalizes an interface, not physics.  MuJoCo's ``qfrc_actuator`` may contain terms that
    Isaac's ``applied_torque`` does not.
    """
    attribute = {
        "isaacsim": "applied_torque",
        "mjlab": "qfrc_actuator",
    }[_native_engine(env, asset)]
    value = getattr(asset.data, attribute, None)
    if value is None:
        raise PortabilityError(
            f"{type(asset).__name__} has no {attribute}; joint actuator effort is unavailable."
        )
    return value


def joint_acceleration(env: Any, asset: Any) -> torch.Tensor:
    """Return native joint acceleration without claiming numerical parity.

    The public attribute is spelled the same, but Isaac finite-differences velocity while MJLab
    exposes analytic MuJoCo ``qacc``.  Keeping the read here makes that difference explicit and
    keeps engine semantics out of task formulas.
    """
    value = getattr(asset.data, "joint_acc", None)
    if value is None:
        raise PortabilityError(
            f"{type(asset).__name__} has no joint_acc; native joint acceleration is unavailable."
        )
    return value


def body_linear_velocity_w(env: Any, asset: Any) -> torch.Tensor:
    """Return the native body velocity historically used by each slide penalty."""
    attribute = {
        "isaacsim": "body_lin_vel_w",
        "mjlab": "body_link_lin_vel_w",
    }[_native_engine(env, asset)]
    value = getattr(asset.data, attribute, None)
    if value is None:
        raise PortabilityError(
            f"{type(asset).__name__} has no {attribute}; native body linear velocity is unavailable."
        )
    return value


def body_angular_velocity_w(env: Any, asset: Any) -> torch.Tensor:
    """Return the native body angular velocity paired with :func:`body_linear_velocity_w`."""
    attribute = {
        "isaacsim": "body_ang_vel_w",
        "mjlab": "body_link_ang_vel_w",
    }[_native_engine(env, asset)]
    value = getattr(asset.data, attribute, None)
    if value is None:
        raise PortabilityError(
            f"{type(asset).__name__} has no {attribute}; native body angular velocity is unavailable."
        )
    return value


def root_command_linear_velocity_b(env: Any, asset: Any) -> torch.Tensor:
    """Return the native quantity historically used by velocity-command metrics.

    Isaac Lab's stock command measures COM linear velocity; MJLab's measures the
    root-link velocity. The command algorithm is portable, but silently changing
    this diagnostic quantity would make its episode metrics incomparable.
    """
    attribute = {
        "isaacsim": "root_com_lin_vel_b",
        "mjlab": "root_link_lin_vel_b",
    }[_native_engine(env, asset)]
    value = getattr(asset.data, attribute, None)
    if value is None:
        raise PortabilityError(
            f"{type(asset).__name__} has no {attribute}; command tracking velocity is unavailable."
        )
    return value


def _actuators(asset: Any) -> Iterable[Any]:
    actuators = asset.actuators
    return actuators.values() if isinstance(actuators, Mapping) else actuators


def _unwrap_base_actuator(actuator: Any) -> Any:
    base = actuator
    while hasattr(base, "base_actuator"):
        base = base.base_actuator
    return base


def joint_stiffness_groups(asset: Any) -> Iterable[tuple[Any, Any]]:
    """Yield native joint indices and stiffness for each position-control actuator group."""
    for actuator in _actuators(asset):
        if getattr(actuator, "transmission_type", "joint") != "joint":
            continue
        base = _unwrap_base_actuator(actuator)

        joint_ids = getattr(actuator, "joint_indices", None)
        stiffness = getattr(actuator, "stiffness", None)
        if joint_ids is not None and stiffness is not None:
            yield joint_ids, stiffness
            continue

        joint_ids = getattr(base, "target_ids", None)
        stiffness = getattr(getattr(base, "cfg", None), "stiffness", None)
        if joint_ids is not None and stiffness is not None:
            yield joint_ids, stiffness


def _is_builtin_pd_actuator(actuator: Any) -> bool:
    return any(cls.__name__ == "BuiltinPdActuator" for cls in type(actuator).__mro__)


def joint_effort_limits(env: Any, asset: Any, joint_ids: Any) -> torch.Tensor:
    """Return native effort caps for ``joint_ids`` in the requested order.

    Isaac exposes them on articulation data.  MJLab requires mapping model force ranges back
    through local articulation joint indices.  The task remains responsible for turning the
    returned quantity into a penalty.
    """
    limits = getattr(asset.data, "joint_effort_limits", None)
    if limits is not None:
        return limits[:, joint_ids]

    applied = joint_applied_torque(env, asset)
    all_limits = torch.zeros_like(applied)
    if isinstance(joint_ids, slice):
        selected = list(range(asset.num_joints))[joint_ids]
    else:
        selected = [int(index) for index in joint_ids]
    selected_names = {asset.joint_names[index] for index in selected}

    actuator_forcerange = env.sim.model.actuator_forcerange
    if actuator_forcerange.ndim == 3:
        actuator_forcerange = actuator_forcerange[0]

    for actuator in _actuators(asset):
        if getattr(actuator, "transmission_type", "joint") != "joint":
            continue
        base = _unwrap_base_actuator(actuator)
        for local_index, joint_name in enumerate(list(actuator.target_names)):
            if joint_name not in selected_names:
                continue
            joint_id = int(actuator.target_ids[local_index])
            if _is_builtin_pd_actuator(base):
                global_joint_id = int(asset.indexing.joint_ids[joint_id])
                ranges = env.sim.model.jnt_actfrcrange
                if ranges.ndim == 3:
                    effort_limit = torch.max(
                        torch.abs(ranges[:, global_joint_id]), dim=-1
                    ).values
                else:
                    effort_limit = torch.max(torch.abs(ranges[global_joint_id]))
            else:
                global_control_id = int(actuator.global_ctrl_ids[local_index])
                effort_limit = torch.max(
                    torch.abs(actuator_forcerange[global_control_id])
                )
            all_limits[:, joint_id] = torch.maximum(
                all_limits[:, joint_id], effort_limit
            )
    return all_limits[:, joint_ids]
