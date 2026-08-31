#!/usr/bin/env python3
"""Install InstinctLab, its engine packages, Isaac Lab, and MJLab.

Uses pip only (never uv). Sibling checkouts are reused when present:

    <workspace>/IsaacLab
    <workspace>/mjlab
    <workspace>/InstinctLab   # this repository

Example:

    python scripts/install.py
    python -m pip install -e source/instinctlab_engine
    python -m pip install -e "source/instinctlab[all]"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
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


def _pip_editable(
    name: str,
    path: Path,
    *,
    force: bool,
    no_deps: bool = False,
) -> None:
    target = path.resolve()
    current = _editable_location(name)
    if not force and current == target:
        print(f"[INFO] Already installed: {name} -> {target}", flush=True)
        return
    args = ("--no-deps",) if no_deps else ()
    _pip("-e", str(target), *args)


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


def _git_dirty(repo: Path) -> bool:
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=normal"],
        text=True,
    )
    return bool(status.strip())


def _checkout_report(
    url: str,
    dest: Path,
    revision: str,
    *,
    allow_unverified: bool,
) -> dict:
    expected = _git_rev_parse(dest, revision)
    actual = _git_head(dest)
    dirty = _git_dirty(dest)
    problems: list[str] = []
    if expected is None:
        problems.append(f"checkout does not contain revision {revision!r}")
    elif actual != expected:
        problems.append(f"HEAD is {actual}, expected {expected} ({revision})")
    if dirty:
        problems.append("checkout has uncommitted or untracked changes")
    if problems and not allow_unverified:
        details = "; ".join(problems)
        raise RuntimeError(
            f"Refusing unverified dependency checkout {dest}: {details}. "
            "Reconcile the checkout or pass --allow-unverified-checkouts; the override "
            "will be recorded in the installation provenance."
        )
    if problems:
        print(f"[WARN] Using unverified checkout {dest}: {'; '.join(problems)}", flush=True)
    return {
        "url": url,
        "path": str(dest.resolve()),
        "requested_revision": revision,
        "expected_commit": expected,
        "actual_commit": actual,
        "dirty": dirty,
        "problems": problems,
        "override_used": bool(problems and allow_unverified),
    }


def _ensure_checkout(
    url: str,
    dest: Path,
    revision: str,
    *,
    allow_unverified: bool,
) -> dict:
    if dest.exists():
        print(f"[INFO] Using existing checkout: {dest}", flush=True)
        return _checkout_report(
            url,
            dest,
            revision,
            allow_unverified=allow_unverified,
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", url, str(dest)])
    _run(["git", "-C", str(dest), "checkout", "--detach", revision])
    return _checkout_report(
        url,
        dest,
        revision,
        allow_unverified=allow_unverified,
    )


def _install_isaaclab(
    workspace: Path,
    pins: dict,
    *,
    force: bool,
    allow_unverified: bool,
) -> dict:
    cfg = pins["isaaclab"]
    root = workspace / "IsaacLab"
    report = _ensure_checkout(
        cfg["git"],
        root,
        cfg["commit"],
        allow_unverified=allow_unverified,
    )
    for name in cfg["packages"]:
        _pip_editable(name, root / "source" / name, force=force)
    return report


def _install_mjlab(
    workspace: Path,
    pins: dict,
    *,
    force: bool,
    allow_unverified: bool,
) -> dict:
    cfg = pins["mjlab"]
    root = workspace / "mjlab"
    report = _ensure_checkout(
        cfg["git"],
        root,
        cfg["tag"],
        allow_unverified=allow_unverified,
    )
    # MJLab's declared lower bounds allow newer MJWarp/Warp releases. Those
    # releases change the contact and constraint kernels, so satisfying the
    # version range does not reproduce InstinctMJ's training plant. Install
    # the verified runtime first; the editable install then sees its broad
    # requirements as already satisfied and leaves these exact versions in
    # place.
    _pip(*cfg["runtime"])
    _pip_editable("mjlab", root, force=force)
    return report


def _write_install_provenance(
    path: Path,
    *,
    checkouts: dict[str, dict],
    allow_unverified: bool,
) -> None:
    payload = {
        "version": "instinctlab_install_provenance_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "allow_unverified_checkouts": allow_unverified,
        "checkouts": checkouts,
    }
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temporary.replace(path)
    print(f"[INFO] Wrote installation provenance: {path}", flush=True)


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
    parser.add_argument(
        "--allow-unverified-checkouts",
        action="store_true",
        help=(
            "Allow wrong-revision or dirty Isaac Lab/MJLab checkouts. The default refuses "
            "them; use of this override is recorded in the installation provenance."
        ),
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        default=Path(sys.prefix) / "share" / "instinctlab" / "install_provenance.json",
        help="Installation provenance receipt path inside the selected Python environment.",
    )
    args = parser.parse_args()

    pins = _load_pins()
    workspace = args.workspace.expanduser().resolve()
    print(f"[INFO] Workspace: {workspace}", flush=True)
    print(f"[INFO] Python: {sys.executable}", flush=True)

    checkouts: dict[str, dict] = {}
    if not args.skip_isaaclab:
        checkouts["isaaclab"] = _install_isaaclab(
            workspace,
            pins,
            force=args.force,
            allow_unverified=args.allow_unverified_checkouts,
        )
    if not args.skip_mjlab:
        checkouts["mjlab"] = _install_mjlab(
            workspace,
            pins,
            force=args.force,
            allow_unverified=args.allow_unverified_checkouts,
        )

    # Install the stable task/engine contract before the application. Keeping it
    # as a separate editable distribution exercises the same package boundary as
    # published wheels while preserving one-command development setup.
    engine_core_src = REPO_ROOT / "source" / "instinctlab_engine"
    _pip_editable(
        "instinctlab-engine-core",
        engine_core_src,
        force=args.force,
    )

    if not args.skip_isaaclab:
        _pip_editable(
            "instinctlab-engine-isaacsim",
            REPO_ROOT / "source" / "instinctlab_engine_isaacsim",
            force=args.force,
            no_deps=True,
        )
    if not args.skip_mjlab:
        _pip_editable(
            "instinctlab-engine-mjlab",
            REPO_ROOT / "source" / "instinctlab_engine_mjlab",
            force=args.force,
            no_deps=True,
        )

    # Refresh InstinctLab in place. --no-deps avoids re-resolving Isaac/MJLab pins.
    instinctlab_src = REPO_ROOT / "source" / "instinctlab"
    _pip("-e", str(instinctlab_src), "--no-deps")
    _write_install_provenance(
        args.provenance_output,
        checkouts=checkouts,
        allow_unverified=args.allow_unverified_checkouts,
    )
    print(
        "[INFO] InstinctLab Engine Core, InstinctLab, Isaac Lab, and MJLab are installed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
