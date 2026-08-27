"""Play several recent runs through the unified player."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _timestamp(name: str) -> datetime:
    return datetime.strptime("_".join(name.split("_")[:2]), "%Y%m%d_%H%M%S")


def _runs(log_root: Path, requested: list[str], count: int) -> list[str]:
    if count == 0:
        return requested
    candidates = sorted(
        (path for path in log_root.iterdir() if path.is_dir()),
        key=lambda path: _timestamp(path.name),
    )
    return [path.name for path in candidates[-count:]]


def main(args: argparse.Namespace, extra_args: list[str]) -> None:
    log_root = Path(args.logroot or Path("logs") / args.engine / args.experiment).resolve()
    for run_name in _runs(log_root, args.runs, args.num_runs):
        print(f"Playing run: {run_name}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "scripts/play.py",
                "--engine",
                args.engine,
                "--task",
                args.task,
                "--logroot",
                str(log_root),
                "--load_run",
                run_name,
            ]
            + extra_args,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=("isaacsim", "mjlab"))
    parser.add_argument("--experiment", default="g1_shadowing")
    parser.add_argument("--task", default="Instinct-Shadowing-WholeBody-Plane-G1-v0")
    parser.add_argument("--logroot", default=None)
    parser.add_argument("-n", "--num-runs", type=int, default=0)
    parser.add_argument("--runs", nargs="*", default=[])
    parsed, remaining = parser.parse_known_args()
    main(parsed, remaining)
