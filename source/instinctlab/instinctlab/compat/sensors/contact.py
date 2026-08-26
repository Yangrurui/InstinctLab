"""Portable contact timing, element selection and history layout."""

from __future__ import annotations

import inspect
import re
import torch
import weakref
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping

from instinctlab.spec.sensor import ContactSensorRef

from ..denylist import PortabilityError

_ELEMENT_NAME_ATTR = MappingProxyType({"isaacsim": "body_names", "mjlab": "primary_names"})
_FORCE_HISTORY: Mapping[str, tuple[str, bool]] = MappingProxyType(
    {
        # Attribute name and whether the element axis precedes the time axis.
        "isaacsim": ("net_forces_w_history", False),
        "mjlab": ("force_history", True),
    }
)

_NAMES: MutableMapping[Any, list[str]] = weakref.WeakKeyDictionary()
_IDS: MutableMapping[Any, dict[tuple[str, tuple[str, ...], bool], list[int]]] = weakref.WeakKeyDictionary()
_ENGINES: MutableMapping[Any, str] = weakref.WeakKeyDictionary()


def forget(sensor: Any | None = None) -> None:
    """Clear cached name resolutions; intended for tests that mutate stub sensors."""
    for cache in (_NAMES, _IDS, _ENGINES):
        if sensor is None:
            cache.clear()
        else:
            cache.pop(sensor, None)


def sensor_engine(sensor: Any) -> str:
    """Identify a native contact sensor from its element-name API."""
    try:
        return _ENGINES[sensor]
    except (KeyError, TypeError):
        pass

    missing = object()
    for engine, attribute in _ELEMENT_NAME_ATTR.items():
        # ``hasattr`` executes descriptors.  Isaac's ``body_names`` property
        # rebuilds names from the physics view, so merely asking which engine
        # owns the sensor used to pay that GPU/CPU synchronization every step.
        if inspect.getattr_static(sensor, attribute, missing) is not missing:
            try:
                _ENGINES[sensor] = engine
            except TypeError:
                pass
            return engine
    expected = " or ".join(_ELEMENT_NAME_ATTR.values())
    raise PortabilityError(f"{type(sensor).__name__} exposes neither {expected}, so its element ordering is unknown.")


def element_names(sensor: Any) -> list[str]:
    """Return names in native element-axis order, caching expensive Isaac lookups."""
    try:
        return _NAMES[sensor]
    except (KeyError, TypeError):
        pass

    names = list(getattr(sensor, _ELEMENT_NAME_ATTR[sensor_engine(sensor)]))
    try:
        _NAMES[sensor] = names
    except TypeError:
        pass
    return names


def _match(patterns: tuple[str, ...], names: list[str], preserve_order: bool) -> list[int]:
    if preserve_order:
        ordered: list[int] = []
        for pattern in patterns:
            for index, name in enumerate(names):
                if re.fullmatch(pattern, name) and index not in ordered:
                    ordered.append(index)
        return ordered
    return [index for index, name in enumerate(names) if any(re.fullmatch(pattern, name) for pattern in patterns)]


def element_ids(sensor: Any, ref: ContactSensorRef) -> list[int]:
    """Resolve a contact reference to native element indices."""
    key = (ref.name, ref.elements, ref.preserve_order)
    try:
        return _IDS[sensor][key]
    except (KeyError, TypeError):
        pass

    names = element_names(sensor)
    finder = getattr(sensor, "find_bodies", None)
    if callable(finder):
        ids, _ = finder(list(ref.elements), preserve_order=ref.preserve_order)
    else:
        ids = _match(ref.elements, names, ref.preserve_order)
    if not ids:
        raise PortabilityError(f"Contact sensor {ref.name!r} matched none of {list(ref.elements)}. It tracks {names}.")

    resolved = list(ids)
    try:
        _IDS.setdefault(sensor, {})[key] = resolved
    except TypeError:
        pass
    return resolved


def _timing(sensor: Any, ref: ContactSensorRef, attribute: str) -> torch.Tensor:
    tensor = getattr(sensor.data, attribute, None)
    if tensor is None:
        raise PortabilityError(
            f"Contact sensor {ref.name!r} has no {attribute}; set track_air_time on its declaration."
        )
    return tensor[:, element_ids(sensor, ref)]


def air_time(sensor: Any, ref: ContactSensorRef) -> torch.Tensor:
    """Current airborne duration in seconds, shaped ``(env, element)``."""
    return _timing(sensor, ref, "current_air_time")


def contact_time(sensor: Any, ref: ContactSensorRef) -> torch.Tensor:
    """Current contact duration in seconds, shaped ``(env, element)``."""
    return _timing(sensor, ref, "current_contact_time")


def in_contact(sensor: Any, ref: ContactSensorRef) -> torch.Tensor:
    """Whether each referenced element is currently in contact."""
    return contact_time(sensor, ref) > 0.0


def contact_force_history(sensor: Any, ref: ContactSensorRef) -> torch.Tensor:
    """Return force history as ``(env, time, element, 3)``, newest first.

    Only layout is normalized. Isaac reports normal force while MJLab reports a full contact force,
    so their force values remain intentionally engine-specific.
    """
    ids = element_ids(sensor, ref)
    attribute, element_axis_first = _FORCE_HISTORY[sensor_engine(sensor)]
    history = getattr(sensor.data, attribute, None)
    if history is None:
        raise PortabilityError(f"Contact sensor {ref.name!r} has no {attribute}; configure a non-zero history_length.")
    if element_axis_first:
        history = history.transpose(1, 2)
    return history[:, :, ids]
