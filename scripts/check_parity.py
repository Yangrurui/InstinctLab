"""Compile a declared task for Isaac Sim and diff it against main's hand-written config.

This is the acceptance test for the whole Isaac Sim path. A task declared without naming an engine
has to produce the config main already ships, and every field that differs has to be a difference
someone decided to make. Differences are checked against a whitelist keyed by dotted path; anything
outside it fails.

Needs Isaac Sim, so it is a script rather than a pytest module: ``AppLauncher`` has to run before
``isaaclab`` is importable, which does not fit pytest's collection.

    python scripts/check_parity.py --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = REPO / "tests/parity/isaacsim.locomotion_flat.golden.json"
DEFAULT_WHITELIST = REPO / "tests/parity/isaacsim.locomotion_flat.whitelist.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--whitelist", type=Path, default=DEFAULT_WHITELIST)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument(
        "--construct",
        action="store_true",
        help=(
            "Also build the environment and step it. Comparing configs proves the declaration"
            " lowers correctly; stepping proves the result runs, which the terms' shapes and the"
            " sensor wiring can break without any field differing."
        ),
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    AppLauncher(args)

    from instinctlab.engines.isaacsim import IsaacSimAdapter
    from instinctlab.tasks.locomotion.flat_g1 import flat_g1
    from instinctlab.verify.structure import compare, dump, report, unexplained, unused

    golden = json.loads(args.golden.read_text())
    whitelist = json.loads(args.whitelist.read_text()) if args.whitelist.exists() else {}

    compiled = IsaacSimAdapter().compile(flat_g1(), num_envs=args.num_envs, device="cuda:0")
    print(compiled.resolution.summary_table())

    differences = compare(golden["config"], dump(compiled.env_cfg), allow=whitelist)
    remaining = unexplained(differences)
    stale = unused(differences, whitelist)
    print(report(differences))
    if remaining:
        print("\nunexplained:")
        for difference in remaining:
            print(f"  {difference}")
    if stale:
        print("\nwhitelist entries that no longer explain anything:")
        for entry in stale:
            print(f"  {entry}")

    if args.construct:
        _step(compiled)

    # Deliberately without closing the app, and via os._exit. Isaac Sim's shutdown ends the process
    # itself with a status of zero, so a status set after it never reaches the caller and the check
    # becomes incapable of failing. Output is already flushed and the process is going away anyway.
    os._exit(1 if remaining or stale else 0)


def _step(compiled) -> None:
    """Build the compiled environment and take a few steps."""
    import torch

    env = compiled.env_cls(cfg=compiled.env_cfg)
    try:
        obs, _ = env.reset()
        groups = {name: tuple(value.shape) for name, value in obs["policy"].items()}
        print(f"reset ok; policy observation groups: {groups}")
        actions = torch.zeros(env.action_space.shape, device=env.device)
        for _ in range(5):
            obs, reward, terminated, _, _ = env.step(actions)
        print(f"stepped 5x; reward mean {reward.mean().item():.4f}, {int(terminated.sum())} terminated")
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
