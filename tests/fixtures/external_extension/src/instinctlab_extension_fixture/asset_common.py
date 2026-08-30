"""SDK-free native asset values shared by the fixture's two declarations."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class NativeJoint:
    name: str
    default_pos: float
    stiffness: float
    damping: float
    armature: float
    effort_limit: float
    velocity_limit: float
    action_scale: float


JOINT = NativeJoint(
    name="joint",
    default_pos=0.0,
    stiffness=2.0,
    damping=0.1,
    armature=0.01,
    effort_limit=3.0,
    velocity_limit=4.0,
    action_scale=0.5,
)


def resource(name: str) -> str:
    return str(Path(str(files(__package__) / "resources" / name)))
