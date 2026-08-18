"""Isaac Sim backend.

Importing this module imports nothing from ``isaaclab``: the registry's decorators run at import
time so the engine's capabilities are known, while every builder body defers its imports. That is
what lets a task be checked against this engine on a machine that cannot run it.
"""

from .adapter import IsaacSimAdapter, IsaacSimCompileCtx
from .terms import TERMS

__all__ = ["TERMS", "IsaacSimAdapter", "IsaacSimCompileCtx"]
