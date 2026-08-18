"""Dump a hand-written Isaac Lab env config as the golden a compiled one is checked against.

The golden is what "the task keeps behaving the way it does today" means concretely. It has to be
produced by running the real config under a real Isaac Sim, because half of what makes a config
what it is happens in ``__post_init__`` -- the action scale, the sensor update period, the run name
-- and reading the source would miss all of it.

Usage::

    python scripts/dump_golden.py \\
        --cfg instinctlab.tasks.locomotion.config.g1.flat_env_cfg:G1FlatEnvCfg \\
        --out tests/parity/isaacsim.locomotion_flat.golden.json

Re-run it whenever the golden task legitimately changes, and let the diff in review show what
moved.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", required=True, help="module:ClassName of the env config to dump")
    parser.add_argument("--out", required=True, type=pathlib.Path, help="where to write the JSON")
    args = parser.parse_args()

    # Isaac Sim has to be up before anything imports isaaclab's submodules; this is the same
    # ordering constraint the isaacsim adapter's bootstrap step exists for.
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        from instinctlab.verify.structure import dump

        module_name, _, class_name = args.cfg.replace(":", ".").rpartition(".")
        cfg_cls = getattr(importlib.import_module(module_name), class_name)
        payload = {"source": args.cfg, "config": dump(cfg_cls())}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Declaration order, not sorted: observation terms are concatenated in the order they are
        # declared, so the order in this file is part of what it records. Sorting would make a
        # golden that a reordered config still matches.
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
