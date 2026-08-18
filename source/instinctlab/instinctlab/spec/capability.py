"""How strongly a task insists on something the running engine may not have.

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

from enum import Enum

__all__ = ["Requirement"]


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
