"""Live Isaac Sim owns the process. Default collection must not start Kit.

``pytest tests -m isaacsim`` used to fail in two process-scoped ways:

* Importing every test module loaded mjlab's site-packages warp 1.16. Isaac's
  extensions then die on ``warp.types.array`` (only in bundled 1.8.2).
* Each file constructed its own ``AppLauncher``. The second call hangs
  (CPU ~100%, GPU 0, SIGTERM ignored). Sharing one launcher in-process then
  hangs on the second ``make_env`` (overflow after locomotion: 15 minutes
  vs 3 minutes in a fresh process).

This hook collects only isaacsim-marked files and runs each module in its
own process. A single-file invocation stays in-process and starts Kit
before collection imports torch.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.isaacsim_app import (
    ISAACSIM_MODULE_WORKER_ENV,
    file_has_isaacsim_mark,
    invocation_is_single_file,
    isaacsim_module_paths,
    selects_isaacsim_session,
)

_REPO = Path(__file__).resolve().parent.parent


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    if not selects_isaacsim_session(config.option.markexpr or ""):
        return None
    path = Path(collection_path)
    if not path.is_file() or path.suffix != ".py" or not path.name.startswith("test_"):
        return None
    try:
        return not file_has_isaacsim_mark(path)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


def pytest_configure(config: pytest.Config) -> None:
    if getattr(config.option, "collectonly", False):
        return
    if not selects_isaacsim_session(config.option.markexpr or ""):
        return
    parent_of_many = not os.environ.get(ISAACSIM_MODULE_WORKER_ENV) and not invocation_is_single_file(config)
    if parent_of_many:
        return
    try:
        import isaaclab  # noqa: F401
    except ImportError:
        return
    from tests.isaacsim_app import ensure_isaac_app

    ensure_isaac_app()


def pytest_runtestloop(session: pytest.Session) -> bool | None:
    if os.environ.get(ISAACSIM_MODULE_WORKER_ENV):
        return None
    if not selects_isaacsim_session(session.config.option.markexpr or ""):
        return None
    if session.config.option.collectonly:
        return None
    modules = isaacsim_module_paths(session.items)
    if len(modules) <= 1:
        return None
    failed: list[str] = []
    env = {**os.environ, ISAACSIM_MODULE_WORKER_ENV: "1"}
    for path in modules:
        rel = path.relative_to(_REPO) if path.is_relative_to(_REPO) else path
        print(f"\n===== isaacsim module {rel} =====", flush=True)
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(path),
            "-m",
            "isaacsim",
            "-v",
            "--tb=short",
            "--no-header",
        ]
        keyword = getattr(session.config.option, "keyword", None)
        if keyword:
            cmd.extend(["-k", keyword])
        result = subprocess.run(cmd, cwd=_REPO, env=env, check=False)
        if result.returncode != 0:
            failed.append(str(rel))
    if failed:
        pytest.exit(f"isaacsim module workers failed: {failed}", returncode=1)
    return True
