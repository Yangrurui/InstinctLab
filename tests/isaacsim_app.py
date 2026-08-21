"""Start Isaac Sim once per process.

Kit is process-scoped. Two things hang a combined ``pytest tests -m isaacsim``
in ways that look like a dead host:

* A second ``AppLauncher`` tears the first session down (CPU ~100%, GPU 0,
  SIGTERM ignored). Live files share :func:`ensure_isaac_app`.
* A second ``make_env`` on the same Kit after another scene has run spins
  forever (CPU ~100%, GPU occupied, Kit log frozen). Overflow after the
  locomotion cells is the one we measured: 15 minutes then SIGKILL, vs 3
  minutes in a fresh process.

``tests/conftest.py`` therefore keeps the session to isaacsim-marked files
(so mjlab cannot load site-packages warp 1.16, which has no
``warp.types.array``) and runs each marked module in its own process.

Import-safe: no engine, no torch. Call this before either is imported.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

from tests.live_device import resolve_live_device

# Set in the per-module worker so the parent does not start Kit and recurse.
ISAACSIM_MODULE_WORKER_ENV = "INSTINCTLAB_ISAACSIM_MODULE_WORKER"

_APP: Any = None


def ensure_isaac_app(*, device: str | None = None) -> Any:
    """Return the process-wide AppLauncher, starting Kit on the first call only."""
    global _APP
    if _APP is not None:
        return _APP
    from isaaclab.app import AppLauncher

    device = device or resolve_live_device()
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    argv = ["--headless", "--device", device]
    previous = sys.argv
    sys.argv = [previous[0], *argv]
    try:
        _APP = AppLauncher(parser.parse_args(argv))
    finally:
        sys.argv = previous
    return _APP


def selects_isaacsim_session(markexpr: str) -> bool:
    """True when this process is running the live Isaac suite, not excluding it.

    Default ``pytest.ini`` addopts is ``not mjlab and not isaacsim``. Command
    ``-m isaacsim`` replaces that. ``not isaacsim`` anywhere means this is not
    the live session — do not start Kit, do not drop the rest of the suite.
    """
    compact = " ".join(markexpr.split())
    if "isaacsim" not in compact:
        return False
    return "not isaacsim" not in compact


def file_has_isaacsim_mark(source: Path | ast.AST | str) -> bool:
    """True for ``pytest.mark.isaacsim``, not a string that merely says isaacsim.

    ``@pytest.mark.parametrize(..., ids=["isaacsim", "mjlab"])`` is not a live
    test. Collecting those files during ``-m isaacsim`` is how mjlab's warp
    used to get into the Kit process.
    """
    tree = (
        source
        if isinstance(source, ast.AST)
        else ast.parse(source if isinstance(source, str) else Path(source).read_text())
    )
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
        ):
            if any(_is_pytest_mark_isaacsim(child) for child in ast.walk(node.value)):
                return True
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        if any(_is_pytest_mark_isaacsim(decorator) for decorator in node.decorator_list):
            return True
    return False


def _is_pytest_mark_isaacsim(node: ast.AST) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    if not isinstance(target, ast.Attribute) or target.attr != "isaacsim":
        return False
    mark = target.value
    return (
        isinstance(mark, ast.Attribute)
        and mark.attr == "mark"
        and isinstance(mark.value, ast.Name)
        and mark.value.id == "pytest"
    )


def invocation_is_single_file(config: Any) -> bool:
    """True when pytest was pointed at one test module, not ``tests/``."""
    args = [Path(arg) for arg in (getattr(config, "args", None) or ())]
    return len(args) == 1 and args[0].is_file()


def isaacsim_module_paths(items: Any) -> list[Path]:
    """Unique test-module paths in collection order."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for item in items:
        path = Path(item.path)
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths
