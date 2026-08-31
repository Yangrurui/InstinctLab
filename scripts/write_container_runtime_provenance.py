#!/usr/bin/env python3
"""Write the clean-source receipt required from an external simulator image."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_receipt(
    name: str, repository: Path, expected: dict[str, Any]
) -> dict[str, Any]:
    repository = repository.expanduser().resolve()
    actual_commit = _git(repository, "rev-parse", "HEAD")
    dirty = bool(_git(repository, "status", "--porcelain", "--untracked-files=normal"))
    if actual_commit != expected["commit"] or dirty:
        raise RuntimeError(
            f"Refusing {name} source at {repository}: commit={actual_commit}, "
            f"dirty={dirty}; expected clean {expected['commit']}."
        )
    origin = _git(repository, "remote", "get-url", "origin")
    accepted_origins = {
        expected["url"],
        expected["url"].removesuffix(".git"),
        expected["url"].replace("https://github.com/", "git@github.com:", 1),
    }
    if origin not in accepted_origins:
        raise RuntimeError(
            f"Refusing {name} source at {repository}: origin={origin!r}, "
            f"expected {expected['url']!r}."
        )
    return {
        "url": expected["url"],
        "commit": actual_commit,
        "dirty": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--isaaclab-checkout", type=Path, required=True)
    parser.add_argument("--mjlab-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lock_path = args.lock.expanduser().resolve()
    lock = json.loads(lock_path.read_text())
    if lock.get("schema_version") != "instinctlab_container_runtime_lock_v1":
        raise ValueError(f"Unsupported container runtime lock: {lock_path}")
    checkouts = {
        "isaaclab": args.isaaclab_checkout,
        "mjlab": args.mjlab_checkout,
    }
    sources = {
        name: _source_receipt(name, checkouts[name], expected)
        for name, expected in lock["sources"].items()
    }
    receipt = {
        "schema_version": "instinctlab_external_runtime_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_lock": str(lock_path),
        "sources": sources,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(f"Wrote clean external runtime source receipt: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
