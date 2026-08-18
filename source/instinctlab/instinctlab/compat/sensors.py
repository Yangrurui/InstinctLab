"""Reading contact sensors the same way under either engine.

This module is the opposite case to :mod:`~instinctlab.compat.entity`. Entities need no runtime
indirection, because both engines spell the state attributes the same. Contact sensors do, because
they agree on some names and shapes and disagree on others, and the disagreements are the kind that
produce plausible wrong numbers rather than errors.

What agrees
-----------
All four air/contact-time tensors carry the same names on both engines -- ``current_air_time``,
``last_air_time``, ``current_contact_time``, ``last_contact_time`` -- and all four are 2-D
``(env, element)``. Both engines also order force history newest-first. A term reading step timing
therefore ports with nothing more than index resolution.

That matters more than it looks: each engine computes these durations from *its own* notion of
contact, using its own solver's forces, inside its own sensor. The reconciliation has already
happened by the time a term sees a duration in seconds. This is the portable contact signal.

What does not agree
-------------------
**The element list has different names.** Isaac Lab exposes ``ContactSensor.body_names``, mjlab
``ContactSensor.primary_names``. Same concept: the order the second axis is indexed in.

**Force history has a different axis order.** Isaac Lab is ``(env, time, element, 3)``, mjlab is
``(env, element, time, 3)``. Both are newest-first along the time axis, so a term that slices
elements on the wrong axis gets a correctly-shaped tensor of the wrong thing whenever the history
length happens to equal the element count -- two feet and two substeps, say.

**Force does not mean the same thing, and no transposition fixes it.** Isaac Lab's
``net_forces_w`` is world-frame and *normal only*; its docstring warns explicitly that it excludes
the tangential component. mjlab's ``force`` is the full 3-D contact force, expressed in the contact
frame unless ``reduce="netforce"`` or ``global_frame=True`` moves it to world. So ``‖force‖`` is
the normal load on one engine and the total load including friction on the other, and the two
differ by however much friction is carrying at that instant. A newton threshold tuned on one engine
does not transfer. :func:`contact_force_history` returns the tensors on a common axis order and
nothing more; it does not pretend the values are comparable.

Engine detection is by duck typing on the element-name attribute, so this module imports no engine
and a term does not have to know which one it is running under.
"""

from __future__ import annotations

import torch
from typing import Any

from instinctlab.spec.sensor import ContactSensorRef

from .denylist import PortabilityError

__all__ = [
    "air_time",
    "contact_force_history",
    "contact_time",
    "element_ids",
    "element_names",
    "in_contact",
    "sensor_engine",
]

_ELEMENT_NAME_ATTR = {"isaacsim": "body_names", "mjlab": "primary_names"}


def sensor_engine(sensor: Any) -> str:
    """Which engine's contact sensor this is, decided by what it calls its element list."""
    for engine, attr in _ELEMENT_NAME_ATTR.items():
        if hasattr(sensor, attr):
            return engine
    raise PortabilityError(
        f"{type(sensor).__name__} exposes neither {' nor '.join(_ELEMENT_NAME_ATTR.values())}, so its "
        "element ordering is unknown. A new engine must be registered in compat.sensors."
    )


def element_names(sensor: Any) -> list[str]:
    """The elements this sensor tracks, ordered as the element axis is ordered."""
    return list(getattr(sensor, _ELEMENT_NAME_ATTR[sensor_engine(sensor)]))


def element_ids(sensor: Any, ref: ContactSensorRef) -> list[int]:
    """Indices on the element axis for the elements ``ref`` names.

    Resolution goes through the sensor's own ``find_bodies``/``find_*`` when it has one, because
    that is what the engine's manager would use; otherwise the patterns are matched here against
    :func:`element_names`. Either path uses the same matching semantics -- the helper behind them
    is the same code in both engines.

    Raises:
        PortabilityError: A pattern matched nothing. Silently returning an empty selection would
            turn a foot-contact reward into a constant.
    """
    names = element_names(sensor)
    finder = getattr(sensor, "find_bodies", None)
    if callable(finder):
        ids, _ = finder(list(ref.elements), preserve_order=ref.preserve_order)
    else:
        ids = _match(ref.elements, names, ref.preserve_order)
    if not ids:
        raise PortabilityError(f"Contact sensor {ref.name!r} matched none of {list(ref.elements)}. It tracks {names}.")
    return list(ids)


def _match(patterns: tuple[str, ...], names: list[str], preserve_order: bool) -> list[int]:
    """Regex match with the same ordering rule both engines apply."""
    import re

    if preserve_order:
        ordered: list[int] = []
        for pattern in patterns:
            for index, name in enumerate(names):
                if re.fullmatch(pattern, name) and index not in ordered:
                    ordered.append(index)
        return ordered
    return [index for index, name in enumerate(names) if any(re.fullmatch(p, name) for p in patterns)]


def _timing(sensor: Any, ref: ContactSensorRef, attr: str) -> torch.Tensor:
    tensor = getattr(sensor.data, attr, None)
    if tensor is None:
        raise PortabilityError(
            f"Contact sensor {ref.name!r} has no {attr}; both engines return None for it unless the "
            "sensor was configured to track air time. Set track_air_time on the reference."
        )
    return tensor[:, element_ids(sensor, ref)]


def air_time(sensor: Any, ref: ContactSensorRef) -> torch.Tensor:
    """Seconds each referenced element has been airborne. Shape ``(env, element)``."""
    return _timing(sensor, ref, "current_air_time")


def contact_time(sensor: Any, ref: ContactSensorRef) -> torch.Tensor:
    """Seconds each referenced element has been in contact. Shape ``(env, element)``."""
    return _timing(sensor, ref, "current_contact_time")


def in_contact(sensor: Any, ref: ContactSensorRef) -> torch.Tensor:
    """Whether each referenced element is touching something. Shape ``(env, element)``.

    Derived from contact duration rather than from a force threshold, so that each engine's own
    contact criterion is the one that decides -- which is the only way this comes out consistent,
    given that the two report different force quantities.
    """
    return contact_time(sensor, ref) > 0.0


def contact_force_history(sensor: Any, ref: ContactSensorRef) -> torch.Tensor:
    """Force history for the referenced elements, on a common axis order.

    Returns:
        Shape ``(env, time, element, 3)``, newest first along the time axis -- Isaac Lab's layout,
        chosen as the hub because it puts time next to the batch the way the rest of the history
        buffers in this project do. mjlab's ``(env, element, time, 3)`` is transposed to match.

    Warning:
        The *values* are not comparable across engines and this function does not make them so.
        Isaac Lab reports the world-frame normal force alone; mjlab reports the full contact force,
        in the contact frame unless the sensor was configured otherwise. Any threshold on these
        numbers has to be declared per engine, with the tolerance written down.
    """
    engine = sensor_engine(sensor)
    ids = element_ids(sensor, ref)
    if engine == "isaacsim":
        history = sensor.data.net_forces_w_history
        attr = "net_forces_w_history"
    else:
        history = sensor.data.force_history
        attr = "force_history"
    if history is None:
        raise PortabilityError(
            f"Contact sensor {ref.name!r} has no {attr}; both engines return None for it unless the "
            "sensor was given a non-zero history_length."
        )
    if engine == "mjlab":
        history = history.transpose(1, 2)  # (env, element, time, 3) -> (env, time, element, 3)
    return history[:, :, ids]
