"""Backends: the only place an engine SDK is imported.

``engines/base.py``, ``registry.py`` and ``compile.py`` are the shared machinery and stay
engine-free, so a task's compilation can be checked against an engine that is not installed.
The engine-specific code lives in ``engines/<name>/`` and is imported only once an engine has been
chosen -- importing this package must never pull in Isaac Sim or MuJoCo, since the launcher has to
be able to inspect the available adapters before deciding which one to bootstrap.
"""

from __future__ import annotations

from .base import CompiledTask, EngineAdapter, Resolution, UnsupportedTerm
from .compile import CompileCtx, compile_family, compile_mdp, qualname_of
from .registry import FAMILIES, TermRegistry

__all__ = [
    "FAMILIES",
    "CompileCtx",
    "CompiledTask",
    "EngineAdapter",
    "Resolution",
    "TermRegistry",
    "UnsupportedTerm",
    "compile_family",
    "compile_mdp",
    "qualname_of",
]
