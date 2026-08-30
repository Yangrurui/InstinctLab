"""Capability vocabulary and task requirements shared by tasks and engines.

Capabilities are open, namespaced strings. Engine plugins register what they
provide, while task terms declare both a capability and how strongly they
require it. Keeping both sides of that protocol here prevents the declaration
layer from depending on the engine implementation package.

"Skip what the engine cannot do" is only safe when the task gets to say which things it can afford
to lose. Dropping a friction randomisation costs some robustness; dropping an observation changes
the shape of the policy input, and dropping a reward changes what is being optimised while the run
still looks healthy. One level cannot cover both, so terms carry a :class:`Requirement`, and the
compiler acts on it.

The defaults follow from that, and are set on each term class rather than chosen per task:

===================  ==========  ==========================================================
family               default     why
===================  ==========  ==========================================================
observation          REQUIRED    absence changes the network's input width and meaning
action               REQUIRED    absence means the policy cannot act
termination          REQUIRED    absence changes the episode structure
command              REQUIRED    an observation term reads it
reward               OPTIONAL    losing a regulariser is survivable -- but must be recorded
event / DR           OPTIONAL    this is where engine capability actually differs
curriculum           OPTIONAL    --
===================  ==========  ==========================================================

A task overrides per term where its own judgement differs: a locomotion task that is only stable
because of one particular reward should mark that reward REQUIRED and find out at startup.

OPTIONAL does not mean silent. Every skip is recorded in the compilation's ``Resolution`` and
printed once as a table at startup, because a silently dropped reward term is a changed objective,
and the resulting policy is otherwise indistinguishable from a healthy one. ``--strict-capabilities``
promotes every OPTIONAL to REQUIRED for CI and for runs that are meant to be comparable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

_REGISTRY: dict[str, str] = {}


class UnknownCapability(KeyError):
    """Raised when a capability identifier was never registered."""


def capability(identifier: str, description: str) -> str:
    """Register one namespaced capability and return its identifier."""
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
    """Return every registered capability and its meaning."""
    return MappingProxyType(dict(_REGISTRY))


def check_known(values: Iterable[str]) -> None:
    """Reject identifiers that no provider or shared vocabulary registered."""
    unknown = sorted(set(values) - set(_REGISTRY))
    if unknown:
        raise UnknownCapability(
            f"{unknown} are not registered capabilities. An engine package registers what it can do "
            f"when it is imported; registered so far: {sorted(_REGISTRY)}"
        )


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
    """The validated capabilities provided by one engine plugin."""

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


class Requirement(str, Enum):
    """What the compiler does when the engine cannot provide a term."""

    REQUIRED = "required"
    """Fail at startup. The task is not runnable on this engine and should say so immediately."""

    OPTIONAL = "optional"
    """Skip it, record it in the resolution, and report it in the startup summary."""

    EMULATE = "emulate"
    """Substitute the adapter's registered stand-in; fall back to OPTIONAL when it has none.

    For terms whose effect can be approximated by other means -- a push event realised by writing
    root velocity where an engine has no external-wrench API, say. The substitution is recorded
    separately from a skip, because an emulated term is running *something*, and a later comparison
    between engines needs to know which.
    """


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
    "Requirement",
    "UnknownCapability",
    "capability",
    "check_known",
    "known",
]
