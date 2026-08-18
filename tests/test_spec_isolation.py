"""Guard: ``instinctlab.spec`` imports with no physics engine installed.

A task declared in ``spec/`` is supposed to be readable, comparable and compilable without deciding
which engine will run it. The moment anything under ``spec/`` imports an engine -- even for a type
annotation evaluated at runtime -- that property is gone, and it goes quietly, because the developer
who breaks it has both engines installed.

So the import is exercised here with the engines blocked outright. Two checks, because they fail
differently: a static one that names the offending module in the failure, and a dynamic one that
catches an engine pulled in through a chain of otherwise innocent imports.

The same applies to the shared machinery in ``engines/`` -- ``base.py``, ``registry.py`` and
``compile.py``, but not the ``engines/<name>/`` packages, which exist to import an engine. Keeping
the machinery clean is what lets the launcher inspect which adapters exist before deciding which
engine to bootstrap, and lets a task be checked against an engine that is not installed.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys

import pytest

import instinctlab.spec

_ENGINE_ROOTS = frozenset({"isaaclab", "isaacsim", "mjlab", "omni", "pxr", "carb", "mujoco", "warp", "usd"})


def _spec_modules() -> list[pathlib.Path]:
    root = pathlib.Path(instinctlab.spec.__file__).parent
    return sorted(root.rglob("*.py"))


def test_spec_package_is_not_empty() -> None:
    """Otherwise the checks below would pass by having nothing to check."""
    assert len(_spec_modules()) >= 2


@pytest.mark.parametrize("source", _spec_modules(), ids=lambda p: p.name)
def test_no_engine_appears_in_spec_imports(source: pathlib.Path) -> None:
    """Static read of every import statement, including ones inside functions."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    leaked = imported & _ENGINE_ROOTS
    assert not leaked, f"{source.name} imports {sorted(leaked)}; spec/ must compile without an engine"


def test_spec_imports_with_engines_blocked() -> None:
    """Dynamic check, which also covers engines reached transitively."""

    class _Blocker:
        def find_module(self, name, path=None):  # noqa: D102 - legacy finder protocol, enough for importlib
            return self if name.split(".")[0] in _ENGINE_ROOTS else None

        def load_module(self, name):  # pragma: no cover - only reached on regression
            raise ImportError(f"spec/ must not need {name}")

    blocker = _Blocker()
    evicted = {
        name: module
        for name, module in sys.modules.items()
        if name.split(".")[0] in _ENGINE_ROOTS or name.startswith("instinctlab.spec")
    }
    for name in evicted:
        del sys.modules[name]
    sys.meta_path.insert(0, blocker)
    try:
        module = importlib.import_module("instinctlab.spec")
        ref = module.EntityRef(bodies=".*_ankle_roll_link", preserve_order=True)
        assert ref.kinds() == {"body"}
    finally:
        sys.meta_path.remove(blocker)
        # Put the originals back. Re-importing left a second copy of every class in ``spec``, and a
        # later test comparing one of them by identity -- ``AgentSpec.resolve() is SimSpec`` --
        # would fail against a class that is by every other measure the same one.
        sys.modules.update(evicted)


def _engine_machinery() -> list[pathlib.Path]:
    """The engine-free part of ``engines/``: its top-level modules, not the per-engine packages."""
    import instinctlab.engines

    root = pathlib.Path(instinctlab.engines.__file__).parent
    return sorted(path for path in root.glob("*.py"))


def test_engine_machinery_is_not_empty() -> None:
    assert len(_engine_machinery()) >= 3


@pytest.mark.parametrize("source", _engine_machinery(), ids=lambda p: p.name)
def test_no_engine_appears_in_the_shared_machinery(source: pathlib.Path) -> None:
    """``compile.py`` reaches an engine only through ``compat``, which imports it inside a call."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    leaked = imported & _ENGINE_ROOTS
    assert not leaked, f"{source.name} imports {sorted(leaked)}; engines/ machinery must stay engine-free"
