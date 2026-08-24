"""Live Isaac Sim owns the process. Default collection must not start Kit.

``pytest tests -m isaacsim`` used to fail in two process-scoped ways:

* Importing every test module loaded mjlab's site-packages warp 1.16. Isaac's
  extensions then die on ``warp.types.array`` (only in bundled 1.8.2).
* Each file constructed its own ``AppLauncher``. The second call hangs
  (CPU ~100%, GPU 0, SIGTERM ignored). Sharing one launcher in-process then
  hangs on the second ``make_env`` (overflow after locomotion: 15 minutes
  vs 3 minutes in a fresh process).

This hook collects only isaacsim-marked files and runs each selected test item in its
own process. The item-level boundary is required: two tests in one module can each call
``make_env``, which is just as unsafe as two modules sharing Kit. Worker results are also
read from JUnit XML because Kit teardown can replace pytest's non-zero process status.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tests.isaacsim_app import ISAACSIM_MODULE_WORKER_ENV, file_has_isaacsim_mark, selects_isaacsim_session

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
    if not os.environ.get(ISAACSIM_MODULE_WORKER_ENV):
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
    nodeids = [item.nodeid for item in session.items]
    if not nodeids:
        return None
    failed: list[str] = []
    env = {**os.environ, ISAACSIM_MODULE_WORKER_ENV: "1"}
    for nodeid in nodeids:
        print(f"\n===== isaacsim test {nodeid} =====", flush=True)
        with tempfile.TemporaryDirectory(prefix="instinctlab-isaacsim-pytest-") as report_dir:
            report = Path(report_dir) / "report.xml"
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                nodeid,
                "-m",
                "isaacsim",
                "-v",
                "--tb=short",
                "--no-header",
                f"--junitxml={report}",
            ]
            result = subprocess.run(cmd, cwd=_REPO, env=env, check=False)
            report_failed = True
            if report.is_file():
                root = ET.parse(report).getroot()
                suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
                report_failed = not suites or any(
                    int(suite.attrib.get("failures", 0)) + int(suite.attrib.get("errors", 0)) > 0 for suite in suites
                )
            if result.returncode != 0 or report_failed:
                failed.append(nodeid)
    if failed:
        pytest.exit(f"isaacsim test workers failed: {failed}", returncode=1)
    return True
