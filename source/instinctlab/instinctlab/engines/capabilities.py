"""What an engine can do, as registered identifiers rather than a closed set.

A task states what it needs and an engine states what it provides; the comparison decides whether a
term is resolved, skipped or refused. The vocabulary the two sides use has to be open, because the
list of things a simulator can do is not knowable from here. This was a closed ``Enum`` for a while,
and the cost of that shape only shows up when it is too late to change cheaply: an engine with a
depth camera, a deformable solver or per-geom friction randomisation cannot say so without editing a
module in the core, which is the same tax the whole N+M structure exists to avoid.

So an identifier is a namespaced string, registered with a description of what providing it means.
Engine packages register their own on import. An unregistered identifier is refused rather than
quietly treated as unsupported -- a typo in ``provides=`` would otherwise read as an engine that
lacks the feature, and the task would skip a term it should have refused to run without.

The namespace is not decoration. ``contact.air_time`` and ``dr.friction.sliding`` say which family a
capability belongs to, so a reader of a resolution report can see that a run lost its contact timing
rather than that it lost something called ``contact_air_time``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

_REGISTRY: dict[str, str] = {}


class UnknownCapability(KeyError):
    """Raised when a capability identifier was never registered."""


def capability(identifier: str, description: str) -> str:
    """Register ``identifier`` and hand it back, so a constant can be bound to the call.

    Args:
        identifier: Namespaced id, ``family.name`` -- the family is what makes a resolution report
            readable when a capability goes missing.
        description: What an engine is claiming when it provides this. Registering without one is
            not allowed: the whole point of the identifier is that two engines agree on what it
            means, and a bare name lets them disagree silently.

    Returns:
        The identifier, so that a module constant and its registration are the same statement.
    """
    if "." not in identifier:
        raise ValueError(f"{identifier!r} needs a namespace, as in 'contact.air_time'")
    if not description.strip():
        raise ValueError(f"{identifier!r} was registered without saying what providing it means")
    existing = _REGISTRY.get(identifier)
    if existing is not None and existing != description:
        raise ValueError(f"{identifier!r} is already registered as {existing!r}")
    _REGISTRY[identifier] = description
    return identifier


def known() -> Mapping[str, str]:
    """Every registered capability, with what it means. Grows as engine packages are imported."""
    return MappingProxyType(dict(_REGISTRY))


def check_known(values: Iterable[str]) -> None:
    """Refuse identifiers nobody registered, naming the closest thing that exists."""
    unknown = sorted(set(values) - set(_REGISTRY))
    if unknown:
        raise UnknownCapability(
            f"{unknown} are not registered capabilities. An engine package registers what it can do "
            f"when it is imported; registered so far: {sorted(_REGISTRY)}"
        )


# ------------------------------------------------------------------------------------------------
# The vocabulary both current engines speak. An engine with something neither of these has registers
# it in its own package rather than here.
# ------------------------------------------------------------------------------------------------

BATCHED_SIMULATION = capability("sim.batched", "Many environments stepped together in one call.")
GPU_SIMULATION = capability("sim.gpu", "Physics runs on the GPU with state left in device memory.")
PLANE_TERRAIN = capability("terrain.plane", "An infinite ground plane.")
ROOT_STATE = capability("state.root", "Reading and writing a body's root pose.")
ROOT_VELOCITY_WRITE = capability("state.root_velocity", "Writing a root velocity, frame qualified.")
JOINT_STATE = capability("state.joint", "Reading and writing joint positions and velocities.")
BODY_STATE = capability("state.body", "Per-body poses and velocities of an articulation.")
IMPLICIT_POSITION_CONTROL = capability("control.position_implicit", "Joint position targets tracked by the solver.")
EFFORT_CONTROL = capability("control.effort", "Direct joint torque commands.")
CONTACT_ACTIVE = capability("contact.active", "Whether a body is currently touching something.")
CONTACT_HISTORY = capability("contact.history", "Contact readings kept for several past steps.")
CONTACT_AIR_TIME = capability("contact.air_time", "Durations a body has been in contact or in the air.")
CONTACT_FORCE_VECTOR = capability("contact.force_vector", "Contact force as a vector rather than a magnitude.")
DR_SLIDING_FRICTION = capability("dr.friction.sliding", "Randomising the sliding friction coefficient.")
DR_RESTITUTION = capability("dr.restitution", "Randomising restitution.")
BODY_MASS_PROPERTIES = capability("body.mass_properties", "Changing a body's mass or inertia after load.")
EXTERNAL_WRENCH = capability("body.external_wrench", "Applying an external force or torque to a body.")
HUMAN_VIEWER = capability("render.human", "An interactive viewer window.")
RGB_ARRAY = capability("render.rgb_array", "Rendering frames to arrays.")


@dataclass(frozen=True)
class CapabilitySet:
    """What one engine provides. Immutable, and validated against the registry when built."""

    values: frozenset[str]

    @classmethod
    def of(cls, values: Iterable[str]) -> CapabilitySet:
        collected = frozenset(values)
        check_known(collected)
        return cls(collected)

    def supports(self, capability: str) -> bool:
        return capability in self.values

    def require(self, required: Iterable[str], *, context: str) -> None:
        required = frozenset(required)
        check_known(required)
        missing = required.difference(self.values)
        if missing:
            raise RuntimeError(f"{context} requires unsupported engine capabilities: {', '.join(sorted(missing))}")


__all__ = [
    "BATCHED_SIMULATION",
    "BODY_MASS_PROPERTIES",
    "BODY_STATE",
    "CONTACT_ACTIVE",
    "CONTACT_AIR_TIME",
    "CONTACT_FORCE_VECTOR",
    "CONTACT_HISTORY",
    "DR_RESTITUTION",
    "DR_SLIDING_FRICTION",
    "EFFORT_CONTROL",
    "EXTERNAL_WRENCH",
    "GPU_SIMULATION",
    "HUMAN_VIEWER",
    "IMPLICIT_POSITION_CONTROL",
    "JOINT_STATE",
    "PLANE_TERRAIN",
    "RGB_ARRAY",
    "ROOT_STATE",
    "ROOT_VELOCITY_WRITE",
    "CapabilitySet",
    "UnknownCapability",
    "capability",
    "check_known",
    "known",
]
