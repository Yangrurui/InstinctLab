"""Guard: capability identifiers are open to new engines and closed to typos.

This was a closed ``Enum``. The shape mattered less than it looked: a third engine with a depth
camera or a deformable solver could not say so without editing a module in the core, which is the
tax the whole N+M structure exists to remove. The replacement is a registry of namespaced strings,
and a registry only helps if both halves hold -- an engine package can add to it, and something
nobody added is refused rather than read as a feature the backend happens to lack.

The second half is the one worth testing. An unregistered identifier that is silently treated as
unsupported turns a typo in ``provides=`` into a term the task skips, and skipping is exactly what
this project has learned to distrust: the run continues, the report says a term was skipped, and
nothing says the reason was a misspelling.
"""

from __future__ import annotations

import pytest

from instinctlab.spec.capability import (
    CONTACT_AIR_TIME,
    DR_SLIDING_FRICTION,
    CapabilitySet,
    UnknownCapability,
    capability,
    check_known,
    known,
)


def test_identifiers_are_namespaced() -> None:
    """``contact.air_time`` says which family went missing; ``contact_air_time`` only says a name."""
    assert CONTACT_AIR_TIME == "contact.air_time"
    assert DR_SLIDING_FRICTION == "dr.friction.sliding"
    assert all("." in identifier for identifier in known())


def test_an_engine_can_register_something_neither_current_engine_has() -> None:
    """The point of the change: this call lives in an engine package, not in the core."""
    identifier = capability("test.deformable_solver", "A solver for deformable bodies.")
    assert identifier in known()
    assert CapabilitySet.of([identifier]).supports(identifier)


def test_registering_the_same_id_with_a_different_meaning_is_refused() -> None:
    """Two engines that mean different things by one id is the failure the description exists for."""
    capability("test.twice", "One meaning.")
    with pytest.raises(ValueError, match="already registered"):
        capability("test.twice", "A different meaning.")


@pytest.mark.parametrize(
    ("identifier", "description", "problem"),
    [
        ("noNamespace", "Something.", "namespace"),
        ("test.blank", "   ", "without saying"),
    ],
)
def test_a_registration_that_says_too_little_is_refused(identifier, description, problem) -> None:
    with pytest.raises(ValueError, match=problem):
        capability(identifier, description)


def test_an_unregistered_id_is_refused_rather_than_treated_as_unsupported() -> None:
    """A typo in provides= must not read as a backend that lacks the feature."""
    with pytest.raises(UnknownCapability):
        check_known(["contact.air_tiem"])
    with pytest.raises(UnknownCapability):
        CapabilitySet.of(["dr.friction.slidding"])
    with pytest.raises(UnknownCapability):
        CapabilitySet.of([CONTACT_AIR_TIME]).require(["contact.air_tiem"], context="a task")


def test_a_registered_id_the_backend_lacks_is_still_a_plain_refusal() -> None:
    """The other half: known but absent has to stay distinguishable from unknown."""
    with pytest.raises(RuntimeError, match="unsupported engine capabilities"):
        CapabilitySet.of([CONTACT_AIR_TIME]).require([DR_SLIDING_FRICTION], context="a task")


def test_both_engines_declare_only_registered_capabilities() -> None:
    """The registries are built at import; this is where a typo in either terms.py surfaces."""
    from instinctlab.engines import adapter, names

    for name in names():
        declared = adapter(name).capabilities().values
        assert declared, f"{name} declares no capabilities at all"
        check_known(declared)
