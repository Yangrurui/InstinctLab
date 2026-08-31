#!/usr/bin/env python3
"""Fail closed when the authoritative handoff still names a release blocker."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCKER = re.compile(r"\*\*P[01]\b")
CONFLICT_MARKERS = ("<<<<<<<", "=======", ">>>>>>>")


def validate_handoff(path: Path) -> None:
    text = path.read_text()
    if any(marker in text for marker in CONFLICT_MARKERS):
        raise RuntimeError(f"Handoff contains an unresolved merge marker: {path}")
    try:
        open_work = text.split("\n## Open work\n", 1)[1].split("\n## ", 1)[0]
    except IndexError as exc:
        raise RuntimeError(f"Handoff has no unique Open work section: {path}") from exc
    blockers = BLOCKER.findall(open_work)
    if blockers:
        raise RuntimeError(
            f"Handoff still declares release blockers {blockers} in {path}."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=REPO_ROOT / "HANDOFF.md")
    args = parser.parse_args()
    validate_handoff(args.path.resolve())
    print(f"Release handoff has no unresolved P0/P1 blocker: {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
