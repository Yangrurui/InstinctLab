"""Attributes whose names agree across engines while their semantics do not.

A portable term reads ``asset.data.<attr>`` and runs unchanged under both engines' native managers.
That only works while the attribute means the same thing on both sides. The entries below are the
cases where it does not: reading them from a portable term produces numbers that are plausible,
silently wrong, and very hard to attribute later. So they are refused at import/compile time and
have to be handled by a per-engine implementation instead.

Every claim here was checked against the installed engines; ``tests/test_compat_vocab.py`` re-checks
the mechanically verifiable parts (attribute existence, alias targets) so upstream renames surface
as a test failure rather than a silent semantic change.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


class PortabilityError(RuntimeError):
    """Raised when a portable term reaches for an attribute that does not port."""


@dataclass(frozen=True)
class DenylistEntry:
    """One same-name-different-meaning trap."""

    name: str
    summary: str
    per_engine: Mapping[str, str]
    resolution: str


def _entry(name: str, summary: str, isaacsim: str, mjlab: str, resolution: str) -> DenylistEntry:
    return DenylistEntry(
        name=name,
        summary=summary,
        per_engine=MappingProxyType({"isaacsim": isaacsim, "mjlab": mjlab}),
        resolution=resolution,
    )


_ENTRIES: tuple[DenylistEntry, ...] = (
    _entry(
        "joint_acc",
        "Same name, different derivation.",
        isaacsim="Finite difference of joint velocity across steps, held in ArticulationData._previous_joint_vel.",
        mjlab="MuJoCo's analytic qacc.",
        resolution=(
            "Do not use as a cross-engine signal. If a reward needs it, declare the term per-engine"
            " and state the tolerance -- the two are not comparable value by value."
        ),
    ),
    _entry(
        "applied_torque",
        "Isaac-only name with a false friend on the MJLab side.",
        isaacsim="ArticulationData.applied_torque, joint space (nv).",
        mjlab=(
            "No applied_torque. The joint-space equivalent is qfrc_actuator (nv). actuator_force"
            " looks like the match but is scalar actuator output in actuation space (nu), which is"
            " a different dimension whenever actuators are not one-per-joint."
        ),
        resolution="Read through a per-engine accessor; never map applied_torque to actuator_force.",
    ),
    _entry(
        "default_root_state",
        "Same name, velocity rows expressed about a different point.",
        isaacsim="Linear/angular velocity rows are centre-of-mass quantities.",
        mjlab="Linear/angular velocity rows are root-link quantities.",
        resolution=(
            "Reset events are per-engine anyway (engines/<name>/events.py). Keep the frame choice"
            " there and do not read this attribute from a portable term."
        ),
    ),
    _entry(
        "body_link_lin_vel_w",
        "Same name, differs for every body except the root.",
        isaacsim="Per-body centre-of-mass offset applied to each body individually.",
        mjlab="Derived using the root's subtree centre of mass.",
        resolution=(
            "For the root body use the hub entry root_link_lin_vel_w, which both engines derive the"
            " same way. Anything per-body goes through a per-engine term with a declared tolerance;"
            " the hub deliberately offers no per-body velocity."
        ),
    ),
    _entry(
        "gravity_vec_w",
        "Different spelling and different behaviour under non-default gravity.",
        isaacsim=(
            "Spelled GRAVITY_VEC_W (upper case) and normalised from the live sim gravity, so it"
            " follows a task that changes gravity."
        ),
        mjlab="Spelled gravity_vec_w (lower case) and hard-coded to [0, 0, -1] at entity build time.",
        resolution=(
            "Portable terms use projected_gravity_b, which both engines derive consistently. A task"
            " that randomises gravity must treat this as a per-engine concern."
        ),
    ),
    _entry(
        "net_forces_w",
        "Contact force that measures a different quantity on each side, in a different frame.",
        isaacsim=(
            "ContactSensorData.net_forces_w: world frame, normal component only. Its own docstring"
            " warns that the tangential contribution is excluded."
        ),
        mjlab=(
            "No net_forces_w. The nearest field is ContactData.force, which is the full 3-D contact"
            " force and sits in the contact frame unless reduce='netforce' or global_frame=True"
            " moves it to world."
        ),
        resolution=(
            "Norms of the two are not the same physical quantity -- one is normal load, the other"
            " includes friction -- so a newton threshold does not transfer. Portable terms detect"
            " contact through compat.sensors.in_contact, which defers to each engine's own contact"
            " criterion. A term that truly needs force magnitude is per-engine and must declare its"
            " threshold and tolerance per engine."
        ),
    ),
)

DENYLIST: Mapping[str, DenylistEntry] = MappingProxyType({entry.name: entry for entry in _ENTRIES})
"""Denylisted attributes, keyed by the attribute name a term would reach for."""


LEGACY_COM_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "root_lin_vel_w": "root_com_lin_vel_w",
        "root_lin_vel_b": "root_com_lin_vel_b",
        "root_ang_vel_w": "root_com_ang_vel_w",
        "root_ang_vel_b": "root_com_ang_vel_b",
        "root_vel_w": "root_com_vel_w",
        "body_lin_vel_w": "body_com_lin_vel_w",
        "body_ang_vel_w": "body_com_ang_vel_w",
        "body_vel_w": "body_com_vel_w",
        "body_lin_acc_w": "body_com_lin_acc_w",
        "body_ang_acc_w": "body_com_ang_acc_w",
        "body_acc_w": "body_com_acc_w",
        "com_pos_b": "body_com_pos_b",
        "com_quat_b": "body_com_quat_b",
    }
)
"""Isaac Lab legacy aliases that resolve to **centre-of-mass** quantities.

The dangerous subset is the one whose *name* carries no hint of it: rewriting ``root_lin_vel_b`` to
``root_link_lin_vel_b`` is the obvious move and is a different physical quantity. MJLab has no such
aliases, so nothing downstream catches it. The correct rewrite is the mapping recorded here.
Verified against Isaac Lab's own docstrings ("Same as :attr:`root_com_...`").
"""

LEGACY_LINK_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "root_pos_w": "root_link_pos_w",
        "root_quat_w": "root_link_quat_w",
        "root_pose_w": "root_link_pose_w",
        "body_pos_w": "body_link_pos_w",
        "body_quat_w": "body_link_quat_w",
        "body_pose_w": "body_link_pose_w",
    }
)
"""Isaac Lab legacy aliases that resolve to **link** quantities.

Harmless in meaning but still ambiguous to a reader, and absent from MJLab. The codemod rewrites
them to the explicit spelling so the link/COM choice is visible at every call site.
"""


def is_legacy_alias(attr: str) -> bool:
    return attr in LEGACY_COM_ALIASES or attr in LEGACY_LINK_ALIASES


def explicit_name(attr: str) -> str:
    """Rewrite a legacy Isaac Lab alias to its explicit-frame spelling."""
    if attr in LEGACY_COM_ALIASES:
        return LEGACY_COM_ALIASES[attr]
    if attr in LEGACY_LINK_ALIASES:
        return LEGACY_LINK_ALIASES[attr]
    return attr


def assert_portable(attr: str) -> None:
    """Refuse attributes that must not be read from a portable term.

    Raises:
        PortabilityError: if ``attr`` is denylisted or is a legacy alias.
    """
    entry = DENYLIST.get(attr)
    if entry is not None:
        detail = "; ".join(f"{engine}: {text}" for engine, text in entry.per_engine.items())
        raise PortabilityError(f"{attr!r} does not port across engines. {entry.summary} {detail} -> {entry.resolution}")
    if is_legacy_alias(attr):
        target = explicit_name(attr)
        anchor = "centre-of-mass" if attr in LEGACY_COM_ALIASES else "link"
        raise PortabilityError(
            f"{attr!r} is an Isaac Lab legacy alias for the {anchor} quantity {target!r} and does not"
            f" exist on other engines. Use {target!r} explicitly."
        )
