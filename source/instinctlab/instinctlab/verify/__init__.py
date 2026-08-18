"""Verification that runs off the training hot path.

Two jobs live here. :mod:`~instinctlab.verify.structure` compares a compiled environment config
against the hand-written one it is meant to reproduce, which is how a backend is accepted. The
older ``SimulatorBackend`` implementations will join it as a state-export layer for sim2sim
assertions, per P8.

Nothing here is imported by a training run.
"""

from __future__ import annotations

from .structure import Difference, compare, dump, flatten, qualname, report, unexplained, unused

__all__ = ["Difference", "compare", "dump", "flatten", "qualname", "report", "unexplained", "unused"]
