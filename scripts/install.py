#!/usr/bin/env python3
"""Install InstinctLab together with Isaac Lab and MJLab.

Uses pip only (never uv). Sibling checkouts are reused when present:

    <workspace>/IsaacLab
    <workspace>/mjlab
    <workspace>/InstinctLab   # this repository

Example:

    python scripts/install.py
    python -m pip install -e "source/instinctlab[all]"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PINS_PATH = REPO_ROOT / "source" / "instinctlab" / "config" / "backend_pins.toml"


def _load_pins() -> dict:
    with PINS_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.check_call(command, cwd=cwd)


def _pip(*args: str) -> None:
    _run([sys.executable, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed", *args])


def _editable_location(name: str) -> Path | None:
    try:
        dist = distribution(name)
    except PackageNotFoundError:
        return None
    payload = dist.read_text("direct_url.json")
    if not payload:
        return None
    data = json.loads(payload)
    if not (data.get("dir_info") or {}).get("editable"):
        return None
    url = str(data.get("url", ""))
    if not url.startswith("file://"):
        return None
    return Path(url.removeprefix("file://")).resolve()


def _pip_editable(name: str, path: Path, *, force: bool) -> None:
    target = path.resolve()
    current = _editable_location(name)
    if not force and current == target:
        print(f"[INFO] Already installed: {name} -> {target}", flush=True)
        return
    _pip("-e", str(target))


def _git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def _git_rev_parse(repo: Path, revision: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", f"{revision}^{{commit}}"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return None


def _ensure_checkout(url: str, dest: Path, revision: str) -> None:
    if dest.exists():
        print(f"[INFO] Using existing checkout: {dest}", flush=True)
        expected = _git_rev_parse(dest, revision)
        actual = _git_head(dest)
        if expected is None:
            print(f"[WARN] {dest} does not contain revision {revision}", flush=True)
        elif not actual.startswith(expected) and not expected.startswith(actual):
            print(f"[WARN] {dest} is at {actual[:12]}, expected {revision}", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", url, str(dest)])
    _run(["git", "-C", str(dest), "checkout", "--detach", revision])


def _install_isaaclab(workspace: Path, pins: dict, *, force: bool) -> None:
    cfg = pins["isaaclab"]
    root = workspace / "IsaacLab"
    _ensure_checkout(cfg["git"], root, cfg["commit"])
    for name in cfg["packages"]:
        _pip_editable(name, root / "source" / name, force=force)


def _install_mjlab(workspace: Path, pins: dict, *, force: bool) -> None:
    cfg = pins["mjlab"]
    root = workspace / "mjlab"
    _ensure_checkout(cfg["git"], root, cfg["tag"])
    _pip_editable("mjlab", root, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=REPO_ROOT.parent,
        help="Directory that contains IsaacLab and mjlab checkouts. Default: parent of this repo.",
    )
    parser.add_argument("--skip-isaaclab", action="store_true")
    parser.add_argument("--skip-mjlab", action="store_true")
    parser.add_argument("--force", action="store_true", help="Reinstall backends even if already present.")
    args = parser.parse_args()

    pins = _load_pins()
    workspace = args.workspace.expanduser().resolve()
    print(f"[INFO] Workspace: {workspace}", flush=True)
    print(f"[INFO] Python: {sys.executable}", flush=True)

    if not args.skip_isaaclab:
        _install_isaaclab(workspace, pins, force=args.force)
    if not args.skip_mjlab:
        _install_mjlab(workspace, pins, force=args.force)

    # Refresh InstinctLab in place. --no-deps avoids re-resolving Isaac/MJLab pins.
    instinctlab_src = REPO_ROOT / "source" / "instinctlab"
    extra = ("--no-deps",) if _editable_location("instinctlab") == instinctlab_src.resolve() else ()
    _pip("-e", str(instinctlab_src), *extra)
    print("[INFO] InstinctLab, Isaac Lab, and MJLab are installed.", flush=True)


if __name__ == "__main__":
    main()
