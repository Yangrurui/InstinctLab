#!/usr/bin/env python3
"""Require repository Ruff findings to stay at or below the reviewed baseline."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "quality" / "ruff_baseline.json"


def main() -> int:
    with BASELINE_PATH.open() as handle:
        baseline = json.load(handle)
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", ".", "--output-format", "json"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        sys.stderr.write(result.stderr)
        return result.returncode
    findings = json.loads(result.stdout)
    current = Counter(item["code"] for item in findings)
    allowed = baseline["maximum_by_code"]
    regressions = {
        code: {"current": count, "maximum": allowed.get(code, 0)}
        for code, count in sorted(current.items())
        if count > allowed.get(code, 0)
    }
    if len(findings) > baseline["maximum_total"]:
        regressions["TOTAL"] = {
            "current": len(findings),
            "maximum": baseline["maximum_total"],
        }
    if regressions:
        print("Ruff ratchet regressed:", json.dumps(regressions, indent=2), file=sys.stderr)
        return 1
    print(
        f"Ruff ratchet passed: {len(findings)} findings "
        f"(maximum {baseline['maximum_total']})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
