#!/usr/bin/env python3
"""Validate an installed native robot asset without constructing a simulator."""

from __future__ import annotations

import argparse
import json

import instinctlab_engine


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check native asset API, resources, canonical DFS order, names, "
            "collision geometry, limits, units, and actuator model declarations."
        )
    )
    parser.add_argument("--engine", required=True, choices=instinctlab_engine.names())
    parser.add_argument(
        "--asset",
        required=True,
        help="Engine-neutral package/variant asset id.",
    )
    args = parser.parse_args()

    report = instinctlab_engine.adapter(args.engine).asset_conformance(args.asset)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
