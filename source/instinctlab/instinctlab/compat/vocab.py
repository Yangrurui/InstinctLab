"""The signed hub vocabulary: what each physical quantity means, and how every engine spells it.

Portable MDP terms read robot state through attribute names. Isaac Lab and MJLab currently happen
to agree on most of those names, which makes it tempting to treat "the name both engines use" as
the definition. That is not a definition -- it is a coincidence that a third engine will not honour,
and it leaves no place to notice when an upstream rename changes meaning.

So the hub is written down here explicitly. Choosing Isaac Lab's explicit-frame spelling as the hub
spelling is a *decision* (it is the most explicit of the two), not an observation. Each engine then
declares its own spoke. Anything a portable term reads must appear in :data:`HUB`.

``documented`` on a spoke is deliberately separate from ``attr``. Both engines return ``wxyz``
quaternions, but only Isaac Lab says so in its docstrings; MJLab inherits the convention from
MuJoCo's ``qpos``/``xquat`` without restating it. That asymmetry is an implicit dependency on an
upstream convention, so it is recorded rather than smoothed over -- see ``tests/test_compat_vocab.py``,
which checks every claim in this table against the installed engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

ENGINES: tuple[str, ...] = ("isaacsim", "mjlab")
"""Engines with a spoke in this table. New engines append here and fill in every entry."""


class Frame(str, Enum):
    """Reference frame a quantity is expressed in."""

    WORLD = "world"
    """Simulation world frame."""
    BASE = "base"
    """Root link frame of the articulation."""
    NONE = "none"
    """Frame-free (joint space, scalars)."""


class Anchor(str, Enum):
    """Which point on a body the quantity is measured at.

    The link/COM distinction is the single most common source of silent cross-engine error, so it
    is a required field rather than something inferred from the name.
    """

    LINK = "link"
    """Body/link origin as authored in the asset."""
    COM = "com"
    """Body centre of mass."""
    ELEMENT = "element"
    """The element's own origin (geom, site)."""
    NA = "n/a"
    """Not anchored to a point."""


class RotationConvention(str, Enum):
    """Component order of a quaternion."""

    WXYZ = "wxyz"


CANONICAL_QUATERNION = RotationConvention.WXYZ
"""Every quaternion crossing ``spec/``, ``mdp/`` or ``compat/`` is ``(w, x, y, z)``.

``xyzw`` exists only at engine API boundaries (PhysX/USD transforms, warp meshes, motion files).
Conversions happen on the line that calls the engine and must not propagate across a function
boundary. Note that both engines ship ``convert_quat(quat, to="xyzw")`` -- the default direction
converts *away* from the hub, so the direction is always passed explicitly.
"""

_MUJOCO_QUAT_BASIS = "MuJoCo qpos/xquat is w-first; MJLab docstrings do not restate it"


@dataclass(frozen=True)
class Spoke:
    """How one engine exposes a hub entry."""

    attr: str | None
    """Attribute on the engine's articulation data object. ``None`` means the engine has no
    equivalent -- the entry stays in the hub so a task can still reference it and get a clear
    capability error instead of a silent drop."""
    documented: bool = False
    """Whether the engine's own docs pin the semantics recorded here, as opposed to us relying on
    an upstream convention. ``False`` marks an implicit dependency to re-check on engine upgrade."""
    evidence: str = ""
    """Where the semantics were read from."""

    @property
    def available(self) -> bool:
        return self.attr is not None


@dataclass(frozen=True)
class HubEntry:
    """One quantity in the hub vocabulary."""

    name: str
    """Hub spelling. Portable terms use this name."""
    unit: str
    frame: Frame
    anchor: Anchor
    doc: str
    spokes: Mapping[str, Spoke]
    rotation: RotationConvention | None = None
    """Set only for quaternions."""

    def spoke(self, engine: str) -> Spoke:
        try:
            return self.spokes[engine]
        except KeyError:
            raise KeyError(f"hub entry {self.name!r} has no spoke for engine {engine!r}") from None


def _both(attr: str, *, isaac_documented: bool = False, isaac_evidence: str = "", mjlab_evidence: str = "") -> dict:
    """Spokes for a quantity both engines spell identically."""
    return {
        "isaacsim": Spoke(
            attr,
            documented=isaac_documented,
            evidence=isaac_evidence or f"isaaclab ArticulationData.{attr}",
        ),
        "mjlab": Spoke(attr, documented=False, evidence=mjlab_evidence or f"mjlab EntityData.{attr}"),
    }


def _mjlab_only(attr: str, *, reason: str, mjlab_evidence: str = "") -> dict:
    return {
        "isaacsim": Spoke(None, evidence=reason),
        "mjlab": Spoke(attr, evidence=mjlab_evidence or f"mjlab EntityData.{attr}"),
    }


_QUAT_DOC = "isaaclab ArticulationData docstring states (w, x, y, z)"

_ENTRIES: tuple[HubEntry, ...] = (
    # --- root link pose and velocity ------------------------------------------------------------
    HubEntry(
        name="root_link_pos_w",
        unit="m",
        frame=Frame.WORLD,
        anchor=Anchor.LINK,
        doc="Root link origin position in the world frame.",
        spokes=_both("root_link_pos_w"),
    ),
    HubEntry(
        name="root_link_quat_w",
        unit="1",
        frame=Frame.WORLD,
        anchor=Anchor.LINK,
        rotation=RotationConvention.WXYZ,
        doc="Root link orientation in the world frame.",
        spokes=_both(
            "root_link_quat_w",
            isaac_documented=True,
            isaac_evidence=_QUAT_DOC,
            mjlab_evidence=_MUJOCO_QUAT_BASIS,
        ),
    ),
    HubEntry(
        name="root_link_lin_vel_w",
        unit="m/s",
        frame=Frame.WORLD,
        anchor=Anchor.LINK,
        doc="Root link linear velocity in the world frame.",
        spokes=_both("root_link_lin_vel_w"),
    ),
    HubEntry(
        name="root_link_ang_vel_w",
        unit="rad/s",
        frame=Frame.WORLD,
        anchor=Anchor.LINK,
        doc="Root link angular velocity in the world frame.",
        spokes=_both("root_link_ang_vel_w"),
    ),
    HubEntry(
        name="root_link_lin_vel_b",
        unit="m/s",
        frame=Frame.BASE,
        anchor=Anchor.LINK,
        doc="Root link linear velocity in the base frame.",
        spokes=_both("root_link_lin_vel_b"),
    ),
    HubEntry(
        name="root_link_ang_vel_b",
        unit="rad/s",
        frame=Frame.BASE,
        anchor=Anchor.LINK,
        doc="Root link angular velocity in the base frame.",
        spokes=_both("root_link_ang_vel_b"),
    ),
    # --- root centre of mass --------------------------------------------------------------------
    HubEntry(
        name="root_com_pos_w",
        unit="m",
        frame=Frame.WORLD,
        anchor=Anchor.COM,
        doc="Root centre-of-mass position in the world frame.",
        spokes=_both("root_com_pos_w"),
    ),
    HubEntry(
        name="root_com_quat_w",
        unit="1",
        frame=Frame.WORLD,
        anchor=Anchor.COM,
        rotation=RotationConvention.WXYZ,
        doc="Root centre-of-mass orientation in the world frame.",
        spokes=_both(
            "root_com_quat_w",
            isaac_documented=True,
            isaac_evidence=_QUAT_DOC,
            mjlab_evidence=_MUJOCO_QUAT_BASIS,
        ),
    ),
    HubEntry(
        name="root_com_lin_vel_w",
        unit="m/s",
        frame=Frame.WORLD,
        anchor=Anchor.COM,
        doc=(
            "Root centre-of-mass linear velocity in the world frame. This is what Isaac Lab's legacy"
            " aliases resolve to -- see instinctlab.compat.denylist.LEGACY_COM_ALIASES."
        ),
        spokes=_both("root_com_lin_vel_w"),
    ),
    # --- derived orientation --------------------------------------------------------------------
    HubEntry(
        name="projected_gravity_b",
        unit="1",
        frame=Frame.BASE,
        anchor=Anchor.LINK,
        doc="Unit gravity direction projected into the base frame.",
        spokes=_both("projected_gravity_b"),
    ),
    HubEntry(
        name="heading_w",
        unit="rad",
        frame=Frame.WORLD,
        anchor=Anchor.NA,
        doc="Yaw of the base frame about the world z axis.",
        spokes=_both("heading_w"),
    ),
    # --- joint space ----------------------------------------------------------------------------
    HubEntry(
        name="joint_pos",
        unit="rad (revolute) / m (prismatic)",
        frame=Frame.NONE,
        anchor=Anchor.NA,
        doc="Joint positions in the canonical DFS joint order.",
        spokes=_both("joint_pos"),
    ),
    HubEntry(
        name="joint_vel",
        unit="rad/s (revolute) / m/s (prismatic)",
        frame=Frame.NONE,
        anchor=Anchor.NA,
        doc="Joint velocities in the canonical DFS joint order.",
        spokes=_both("joint_vel"),
    ),
    # --- per-body -------------------------------------------------------------------------------
    HubEntry(
        name="body_link_pos_w",
        unit="m",
        frame=Frame.WORLD,
        anchor=Anchor.LINK,
        doc="Per-body link origin positions in the world frame.",
        spokes=_both("body_link_pos_w"),
    ),
    HubEntry(
        name="body_link_quat_w",
        unit="1",
        frame=Frame.WORLD,
        anchor=Anchor.LINK,
        rotation=RotationConvention.WXYZ,
        doc="Per-body link orientations in the world frame.",
        spokes=_both(
            "body_link_quat_w",
            isaac_documented=True,
            isaac_evidence=_QUAT_DOC,
            mjlab_evidence=_MUJOCO_QUAT_BASIS,
        ),
    ),
    # Per-body *velocity* is deliberately absent: the two engines derive it differently for every
    # body except the root, so it is denylisted rather than offered here. Root velocity is available
    # above; per-body velocity needs a per-engine term. See instinctlab.compat.denylist.
    # --- MuJoCo-native elements -----------------------------------------------------------------
    # Kept in the hub even though Isaac Sim has no equivalent: an mjlab project being migrated the
    # other way must be able to express these, and get a capability error rather than a silent drop.
    HubEntry(
        name="geom_pos_w",
        unit="m",
        frame=Frame.WORLD,
        anchor=Anchor.ELEMENT,
        doc="Collision/visual geom origin positions in the world frame.",
        spokes=_mjlab_only("geom_pos_w", reason="Isaac Sim has no per-geom state view"),
    ),
    HubEntry(
        name="geom_quat_w",
        unit="1",
        frame=Frame.WORLD,
        anchor=Anchor.ELEMENT,
        rotation=RotationConvention.WXYZ,
        doc="Collision/visual geom orientations in the world frame.",
        spokes=_mjlab_only(
            "geom_quat_w",
            reason="Isaac Sim has no per-geom state view",
            mjlab_evidence=_MUJOCO_QUAT_BASIS,
        ),
    ),
    HubEntry(
        name="site_pos_w",
        unit="m",
        frame=Frame.WORLD,
        anchor=Anchor.ELEMENT,
        doc="Site origin positions in the world frame.",
        spokes=_mjlab_only("site_pos_w", reason="Isaac Sim has no site concept"),
    ),
    HubEntry(
        name="site_quat_w",
        unit="1",
        frame=Frame.WORLD,
        anchor=Anchor.ELEMENT,
        rotation=RotationConvention.WXYZ,
        doc="Site orientations in the world frame.",
        spokes=_mjlab_only(
            "site_quat_w",
            reason="Isaac Sim has no site concept",
            mjlab_evidence=_MUJOCO_QUAT_BASIS,
        ),
    ),
)

HUB: Mapping[str, HubEntry] = MappingProxyType({entry.name: entry for entry in _ENTRIES})
"""The hub vocabulary, keyed by hub spelling."""


def hub_entry(name: str) -> HubEntry:
    """Look up a hub entry, refusing names that are not part of the signed vocabulary."""
    try:
        return HUB[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not in the hub vocabulary. Portable terms may only read hub quantities;"
            " add an entry to instinctlab.compat.vocab (with a spoke per engine) first."
        ) from None


def spoke_attr(name: str, engine: str) -> str:
    """Resolve a hub name to the attribute ``engine`` exposes it as.

    Raises when the engine has no equivalent, so the caller has to decide between an engine-specific
    implementation and declaring the capability optional.
    """
    spoke = hub_entry(name).spoke(engine)
    if spoke.attr is None:
        raise LookupError(f"engine {engine!r} has no equivalent of hub quantity {name!r}: {spoke.evidence}")
    return spoke.attr
