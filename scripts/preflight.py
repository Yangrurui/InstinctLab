#!/usr/bin/env python3
"""Report task/provider compatibility before native environment construction."""

from __future__ import annotations

import argparse
import json

import instinctlab_engine
from instinctlab_engine.preflight import preflight_report


def main() -> int:
    from instinctlab.tasks import registry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=instinctlab_engine.names())
    parser.add_argument("--task", required=True, choices=registry.ids())
    parser.add_argument(
        "--allow-nonclean",
        action="store_true",
        help="Report optional/emulated/omitted terms without failing the command.",
    )
    args = parser.parse_args()
    selected = instinctlab_engine.adapter(args.engine)
    robot = selected.robot_spec(registry.asset_id(args.task))
    spec = registry.spec(args.task, robot)
    report = preflight_report(
        spec,
        args.engine,
        selected_adapter=selected,
        allow_nonclean=args.allow_nonclean,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
