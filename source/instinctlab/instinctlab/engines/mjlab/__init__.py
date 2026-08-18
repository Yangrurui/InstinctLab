"""mjlab backend.

Importing this module imports nothing from ``mjlab``, the same property the Isaac Sim backend has
and for the same reason: the registry's keys have to exist for ``contract_report`` to answer, while
the builders' bodies must not, so a task can be checked against this engine anywhere.
"""

from .adapter import MjlabAdapter, MjlabCompileCtx
from .terms import TERMS

__all__ = ["TERMS", "MjlabAdapter", "MjlabCompileCtx"]
