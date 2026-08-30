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
from typing import Any, Literal

import torch

from instinctlab_engine.actuators import ACTUATORS, EFFORT_LIMITS, STIFFNESS

from .env import env_engine
from .errors import PortabilityError

__all__ = [
    "body_angular_velocity_w",
    "body_linear_velocity_w",
    "joint_acceleration",
    "joint_applied_torque",
    "joint_effort_limits",
    "joint_stiffness_groups",
    "root_angular_velocity_b",
    "root_linear_velocity_b",
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


def root_angular_velocity_b(asset: Any) -> torch.Tensor:
    """Return root angular velocity without constructing an unused link velocity.

    Angular velocity is independent of the point on a rigid body. Isaac exposes
    a direct COM property, while MJLab exposes the equivalent root-link value.
    Prefer the direct property when available because Isaac's link-velocity
    property also computes a linear COM-to-link correction.
    """
    value = getattr(asset.data, "root_com_ang_vel_b", None)
    if value is None:
        value = getattr(asset.data, "root_link_ang_vel_b", None)
    if value is None:
        raise PortabilityError(
            f"{type(asset).__name__} exposes neither root_com_ang_vel_b nor "
            "root_link_ang_vel_b; root angular velocity is unavailable."
        )
    return value


def root_linear_velocity_b(
    asset: Any, *, anchor: Literal["com", "link"]
) -> torch.Tensor:
    """Return body-frame root linear velocity at the requested physical point.

    Unlike angular velocity, COM and link-origin linear velocities differ by a
    lever-arm term. The task must therefore state the anchor instead of asking
    this compatibility layer to select one from the engine name.
    """
    try:
        attribute = {
            "com": "root_com_lin_vel_b",
            "link": "root_link_lin_vel_b",
        }[anchor]
    except KeyError:
        raise ValueError(
            f"root linear velocity anchor must be 'com' or 'link', got {anchor!r}."
        ) from None
    value = getattr(asset.data, attribute, None)
    if value is None:
        raise PortabilityError(
            f"{type(asset).__name__} has no {attribute}; root {anchor} linear velocity is unavailable."
        )
    return value


def _actuator_groups(asset: Any) -> Iterable[tuple[str, Any]]:
    actuators = asset.actuators
    if isinstance(actuators, Mapping):
        yield from ((str(name), actuator) for name, actuator in actuators.items())
    else:
        yield from (
            (str(index), actuator) for index, actuator in enumerate(actuators)
        )


def joint_stiffness_groups(
    env: Any, asset: Any, *, requesting_term: str = "joint stiffness reader"
) -> Iterable[tuple[Any, Any]]:
    """Yield stiffness only through the matched native actuator capability adapter."""
    engine = _native_engine(env, asset)
    for native_group, actuator in _actuator_groups(asset):
        if getattr(actuator, "transmission_type", "joint") != "joint":
            continue
        _registration, adapter = ACTUATORS.runtime_adapter(
            engine,
            actuator,
            capability=STIFFNESS,
            native_group=native_group,
            requesting_term=requesting_term,
        )
        try:
            groups = adapter.stiffness_groups(actuator)
        except AttributeError:
            raise RuntimeError(
                f"Engine {engine!r} actuator adapter for group {native_group!r} "
                f"declares {STIFFNESS!r} but does not implement stiffness_groups()."
            ) from None
        try:
            iterator = iter(groups)
        except TypeError:
            raise RuntimeError(
                f"Engine {engine!r} actuator adapter for group {native_group!r} "
                "must return an iterable of (joint_ids, stiffness) pairs."
            ) from None
        for group in iterator:
            if not isinstance(group, (tuple, list)) or len(group) != 2:
                raise RuntimeError(
                    f"Engine {engine!r} actuator adapter for group {native_group!r} "
                    "must return (joint_ids, stiffness) pairs."
                )
            yield group[0], group[1]


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

    engine = _native_engine(env, asset)
    for native_group, actuator in _actuator_groups(asset):
        if getattr(actuator, "transmission_type", "joint") != "joint":
            continue
        for local_index, joint_name in enumerate(list(actuator.target_names)):
            if joint_name not in selected_names:
                continue
            joint_id = int(actuator.target_ids[local_index])
            _registration, adapter = ACTUATORS.runtime_adapter(
                engine,
                actuator,
                capability=EFFORT_LIMITS,
                native_group=native_group,
                requesting_term="joint effort limit reader",
            )
            try:
                effort_limit = adapter.effort_limit_for_joint(
                    env, asset, actuator, local_index
                )
            except AttributeError:
                raise RuntimeError(
                    f"Engine {engine!r} actuator adapter for group {native_group!r} "
                    f"declares {EFFORT_LIMITS!r} but does not implement "
                    "effort_limit_for_joint()."
                ) from None
            effort_limit = torch.as_tensor(
                effort_limit,
                device=all_limits.device,
                dtype=all_limits.dtype,
            )
            if effort_limit.ndim > 1 or (
                effort_limit.ndim == 1
                and effort_limit.shape[0] not in (1, all_limits.shape[0])
            ):
                raise RuntimeError(
                    f"Engine {engine!r} actuator adapter for group {native_group!r} "
                    "must return a scalar or one effort limit per environment."
                )
            all_limits[:, joint_id] = torch.maximum(
                all_limits[:, joint_id], effort_limit
            )
    return all_limits[:, joint_ids]
