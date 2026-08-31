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
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import torch

from instinctlab_engine.actuators import ACTUATORS, EFFORT_LIMITS, STIFFNESS

from .env import env_engine
from .errors import PortabilityError

__all__ = [
    "JointStiffnessBinding",
    "body_angular_velocity_w",
    "body_linear_velocity_w",
    "joint_acceleration",
    "joint_applied_torque",
    "joint_effort_limits",
    "joint_stiffness_groups",
    "resolve_joint_stiffness_groups",
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
        yield from ((str(index), actuator) for index, actuator in enumerate(actuators))


def _joint_index_tuple(
    value: Any, joint_count: int, *, context: str
) -> tuple[int, ...]:
    if value is None:
        indices = tuple(range(joint_count))
    elif isinstance(value, slice):
        indices = tuple(range(joint_count))[value]
    elif isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise RuntimeError(f"{context} joint ids must be one-dimensional.")
        if value.dtype not in {
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise RuntimeError(
                f"{context} joint ids must use an integer tensor dtype, got {value.dtype}."
            )
        # Actuator ownership is static. Callers resolve this helper once while a manager term is
        # initialized, never from the per-step reward path where a CUDA ``tolist()`` would force
        # device-to-host synchronization.
        indices = tuple(value.tolist())
    elif isinstance(value, Integral) and not isinstance(value, bool):
        indices = (int(value),)
    else:
        try:
            raw_indices = tuple(value)
        except TypeError:
            raise RuntimeError(f"{context} has invalid joint ids {value!r}.") from None
        if any(
            not isinstance(index, Integral) or isinstance(index, bool)
            for index in raw_indices
        ):
            raise RuntimeError(
                f"{context} joint ids must contain only integral values, got {value!r}."
            )
        indices = tuple(int(index) for index in raw_indices)
    if len(indices) != len(set(indices)):
        raise RuntimeError(f"{context} repeats joint ids {indices!r}.")
    invalid = tuple(index for index in indices if index < 0 or index >= joint_count)
    if invalid:
        raise RuntimeError(
            f"{context} has joint ids outside [0, {joint_count}): {invalid!r}."
        )
    return indices


def _actuator_joint_ids(
    actuator: Any,
    joint_count: int,
    *,
    native_group: str,
) -> tuple[int, ...]:
    ids = getattr(actuator, "target_ids", None)
    if ids is None:
        ids = getattr(actuator, "joint_indices", None)
    if ids is None:
        raise RuntimeError(
            f"Native actuator group {native_group!r} exposes neither target_ids nor "
            "joint_indices, so its joint ownership cannot be validated."
        )
    return _joint_index_tuple(
        ids,
        joint_count,
        context=f"native actuator group {native_group!r}",
    )


def _selected_stiffness(
    stiffness: Any,
    *,
    returned_ids: tuple[int, ...],
    selected_positions: tuple[int, ...],
    target_shape: tuple[int, int],
    engine: str,
    native_group: str,
) -> Any:
    tensor = torch.as_tensor(stiffness)
    selected = stiffness
    if tensor.ndim > 0 and tensor.shape[-1] == len(returned_ids):
        tensor = tensor[..., list(selected_positions)]
        if len(selected_positions) != len(returned_ids):
            selected = tensor
    try:
        broadcast_shape = torch.broadcast_shapes(tuple(tensor.shape), target_shape)
    except RuntimeError:
        broadcast_shape = None
    if broadcast_shape != target_shape:
        raise RuntimeError(
            f"Engine {engine!r} actuator adapter for group {native_group!r} returned "
            f"stiffness shape {tuple(tensor.shape)}; it must be broadcast-compatible "
            f"with selected joint shape {target_shape}."
        )
    return selected


def _stiffness_group_iterator(
    env: Any,
    asset: Any,
    actuator: Any,
    adapter: Any,
    *,
    engine: str,
    native_group: str,
) -> Iterable[tuple[Any, Any]]:
    try:
        groups = adapter.stiffness_groups(env, asset, actuator)
    except AttributeError:
        raise RuntimeError(
            f"Engine {engine!r} actuator adapter for group {native_group!r} "
            f"declares {STIFFNESS!r} but does not implement stiffness_groups()."
        ) from None
    try:
        return iter(groups)
    except TypeError:
        raise RuntimeError(
            f"Engine {engine!r} actuator adapter for group {native_group!r} "
            "must return an iterable of (joint_ids, stiffness) pairs."
        ) from None


@dataclass(frozen=True, slots=True)
class JointStiffnessBinding:
    """Static actuator ownership plus a live reader for one stiffness group."""

    joint_ids: torch.Tensor
    _env: Any
    _asset: Any
    _actuator: Any
    _adapter: Any
    _group_index: int
    _returned_ids: tuple[int, ...]
    _selected_positions: tuple[int, ...]
    _selected_position_ids: torch.Tensor | None
    _engine: str
    _native_group: str

    def _current_value(self) -> Any:
        iterator = _stiffness_group_iterator(
            self._env,
            self._asset,
            self._actuator,
            self._adapter,
            engine=self._engine,
            native_group=self._native_group,
        )
        for index, group in enumerate(iterator):
            if index != self._group_index:
                continue
            if not isinstance(group, (tuple, list)) or len(group) != 2:
                raise RuntimeError(
                    f"Engine {self._engine!r} actuator adapter for group "
                    f"{self._native_group!r} must return "
                    "(joint_ids, stiffness) pairs."
                )
            return group[1]
        raise RuntimeError(
            f"Engine {self._engine!r} actuator adapter for group "
            f"{self._native_group!r} no longer returns stiffness group "
            f"{self._group_index}."
        )

    def diagnostic_value(self) -> Any:
        """Read the current value while preserving the public diagnostic shape."""
        return _selected_stiffness(
            self._current_value(),
            returned_ids=self._returned_ids,
            selected_positions=self._selected_positions,
            target_shape=(
                int(self._asset.data.joint_vel.shape[0]),
                len(self._selected_positions),
            ),
            engine=self._engine,
            native_group=self._native_group,
        )

    def read(self, target: torch.Tensor) -> torch.Tensor:
        """Read current native stiffness, selected and cast like ``target``."""
        stiffness = torch.as_tensor(
            self._current_value(),
            device=target.device,
            dtype=target.dtype,
        )
        if (
            stiffness.ndim > 0
            and stiffness.shape[-1] == len(self._returned_ids)
            and self._selected_position_ids is not None
        ):
            stiffness = torch.index_select(
                stiffness,
                stiffness.ndim - 1,
                self._selected_position_ids,
            )
        try:
            broadcast_shape = torch.broadcast_shapes(
                tuple(stiffness.shape), tuple(target.shape)
            )
        except RuntimeError:
            broadcast_shape = None
        if broadcast_shape != tuple(target.shape):
            raise RuntimeError(
                f"Engine {self._engine!r} actuator adapter for group "
                f"{self._native_group!r} returned stiffness shape "
                f"{tuple(stiffness.shape)}; it must be broadcast-compatible "
                f"with selected joint shape {tuple(target.shape)}."
            )
        return stiffness


def _resolve_joint_stiffness_groups(
    env: Any,
    asset: Any,
    joint_ids: Any,
    *,
    requesting_term: str = "joint stiffness reader",
) -> Iterable[tuple[tuple[int, ...], JointStiffnessBinding]]:
    engine = _native_engine(env, asset)
    joint_count = int(getattr(asset, "num_joints", asset.data.joint_vel.shape[-1]))
    selected_ids = _joint_index_tuple(
        joint_ids,
        joint_count,
        context=f"term {requesting_term!r}",
    )
    selected_set = set(selected_ids)
    num_envs = int(asset.data.joint_vel.shape[0])
    device = asset.data.joint_vel.device
    all_covered: set[int] = set()
    for native_group, actuator in _actuator_groups(asset):
        if getattr(actuator, "transmission_type", "joint") != "joint":
            continue
        owning_ids = _actuator_joint_ids(
            actuator,
            joint_count,
            native_group=native_group,
        )
        required_ids = selected_set.intersection(owning_ids)
        if not required_ids:
            continue
        _registration, adapter = ACTUATORS.runtime_adapter(
            engine,
            actuator,
            capability=STIFFNESS,
            native_group=native_group,
            requesting_term=requesting_term,
        )
        iterator = _stiffness_group_iterator(
            env,
            asset,
            actuator,
            adapter,
            engine=engine,
            native_group=native_group,
        )
        covered: set[int] = set()
        for group_index, group in enumerate(iterator):
            if not isinstance(group, (tuple, list)) or len(group) != 2:
                raise RuntimeError(
                    f"Engine {engine!r} actuator adapter for group {native_group!r} "
                    "must return (joint_ids, stiffness) pairs."
                )
            returned_ids = _joint_index_tuple(
                group[0],
                joint_count,
                context=(
                    f"engine {engine!r} actuator adapter for group {native_group!r}"
                ),
            )
            outside = set(returned_ids) - set(owning_ids)
            if outside:
                raise RuntimeError(
                    f"Engine {engine!r} actuator adapter for group {native_group!r} "
                    f"returned joint ids outside its owning group: {sorted(outside)}."
                )
            relevant_positions = tuple(
                index
                for index, joint_id in enumerate(returned_ids)
                if joint_id in required_ids
            )
            if not relevant_positions:
                continue
            relevant_ids = tuple(returned_ids[index] for index in relevant_positions)
            duplicates = covered.intersection(relevant_ids)
            if duplicates:
                raise RuntimeError(
                    f"Engine {engine!r} actuator adapter for group {native_group!r} "
                    f"returned duplicate selected joint ids: {sorted(duplicates)}."
                )
            duplicates = all_covered.intersection(relevant_ids)
            if duplicates:
                raise RuntimeError(
                    f"Engine {engine!r} actuator groups return stiffness more than once "
                    f"for selected joint ids {sorted(duplicates)}."
                )
            _selected_stiffness(
                group[1],
                returned_ids=returned_ids,
                selected_positions=relevant_positions,
                target_shape=(num_envs, len(relevant_ids)),
                engine=engine,
                native_group=native_group,
            )
            covered.update(relevant_ids)
            all_covered.update(relevant_ids)
            selected_position_ids = None
            if len(relevant_positions) != len(returned_ids):
                selected_position_ids = torch.tensor(
                    relevant_positions,
                    device=device,
                    dtype=torch.long,
                )
            yield (
                relevant_ids,
                JointStiffnessBinding(
                    joint_ids=torch.tensor(
                        relevant_ids,
                        device=device,
                        dtype=torch.long,
                    ),
                    _env=env,
                    _asset=asset,
                    _actuator=actuator,
                    _adapter=adapter,
                    _group_index=group_index,
                    _returned_ids=returned_ids,
                    _selected_positions=relevant_positions,
                    _selected_position_ids=selected_position_ids,
                    _engine=engine,
                    _native_group=native_group,
                ),
            )
        missing = required_ids - covered
        if missing:
            raise RuntimeError(
                f"Engine {engine!r} actuator adapter for group {native_group!r} did "
                f"not return stiffness for selected joint ids {sorted(missing)}."
            )
    missing = selected_set - all_covered
    if missing:
        raise RuntimeError(
            f"Engine {engine!r} actuator groups do not provide stiffness for selected "
            f"joint ids {sorted(missing)} requested by {requesting_term!r}."
        )


def joint_stiffness_groups(
    env: Any,
    asset: Any,
    joint_ids: Any,
    *,
    requesting_term: str = "joint stiffness reader",
) -> Iterable[tuple[tuple[int, ...], Any]]:
    """Yield the public Python-id stiffness interface for diagnostics and probes."""
    for python_ids, binding in _resolve_joint_stiffness_groups(
        env,
        asset,
        joint_ids,
        requesting_term=requesting_term,
    ):
        yield python_ids, binding.diagnostic_value()


def resolve_joint_stiffness_groups(
    env: Any,
    asset: Any,
    joint_ids: Any,
    *,
    requesting_term: str = "joint stiffness reader",
) -> Iterable[JointStiffnessBinding]:
    """Resolve static ownership while keeping native stiffness values live."""
    for _python_ids, binding in _resolve_joint_stiffness_groups(
        env,
        asset,
        joint_ids,
        requesting_term=requesting_term,
    ):
        yield binding


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
