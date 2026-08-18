"""References to contact measurements, stated without naming an engine.

The two engines take opposite approaches to contact sensing, and the difference is structural
rather than cosmetic:

* Isaac Lab declares **one broad sensor** over a prim-path pattern -- typically every body of the
  robot -- and each term slices out the bodies it cares about with a ``SceneEntityCfg``.
* mjlab declares **many narrow sensors**, each one already scoped to its elements by a ``primary``
  pattern, and terms read the whole sensor.

Neither is more correct, and a portable term cannot be written against either shape directly. So a
:class:`ContactSensorRef` says only *what is being measured* -- these elements of this entity,
optionally only against that counterpart -- and each backend decides whether that becomes a slice
of a broad sensor or a sensor of its own. This is the same move the rest of the design makes: state
the intent in the IR, let the backend pick the idiom.

What can be read back portably is narrower than it looks. See
:mod:`~instinctlab.compat.sensors`: the air/contact-time signals line up across engines, raw
contact force does not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["ContactSensorRef"]


def _normalise(patterns: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(patterns, str):
        return (patterns,)
    return tuple(patterns)


@dataclass(frozen=True)
class ContactSensorRef:
    """A contact measurement on part of an entity.

    Args:
        name: Key this sensor is registered under in the scene. Terms use it to find the sensor,
            and the backend uses it when it has to create one.
        elements: Name patterns for the elements whose contacts are tracked, e.g. the feet.
            Matched against the entity's own names by the same helper both engines use.
        entity: Key of the entity the elements belong to.
        against: Optional counterpart restriction -- ``"terrain"`` to count only contacts with the
            ground, ``None`` to count any contact. Isaac Lab expresses this with
            ``filter_prim_paths_expr`` and mjlab with a ``secondary`` match; both narrow the same
            thing, so it is stated once here.
        track_air_time: Ask the engine to accumulate air and contact durations. Both engines gate
            this behind a config flag because it costs per-step bookkeeping, and both return
            ``None`` for the corresponding tensors when it is off.
        history_length: Number of past substeps of force data to retain. ``0`` disables it. Both
            engines order the history newest-first.
        preserve_order: Whether the element order follows the patterns rather than the entity's.

    Note:
        ``elements`` are *bodies* on the Isaac Lab side, because its contact sensor attaches to
        rigid-body prims. mjlab can additionally scope a contact to a geom or to a whole subtree.
        Which of those a backend picks is its own decision, recorded in the manifest; the reference
        deliberately does not force the finer distinction, because a task that demands geom-level
        contact is not portable to Isaac Lab in the first place and should say so through the
        capability mechanism instead.
    """

    name: str
    elements: str | Sequence[str]
    entity: str = "robot"
    against: str | None = None
    track_air_time: bool = False
    history_length: int = 0
    preserve_order: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "elements", _normalise(self.elements))
        if not self.elements:
            raise ValueError(f"Contact sensor {self.name!r} was given no element patterns.")
        if self.history_length < 0:
            raise ValueError(f"Contact sensor {self.name!r} has a negative history_length.")
