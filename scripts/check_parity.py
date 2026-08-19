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
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help=(
            "Environment count to compile at. Defaults to the golden's, because the count is a"
            " property of a run rather than of the declaration -- compiling at a different one"
            " reports a difference that says nothing about whether the task lowered correctly."
        ),
    )
    parser.add_argument(
        "--no-construct",
        dest="construct",
        action="store_false",
        help=(
            "Skip building and stepping the environment. On by default: comparing configs proves"
            " the declaration lowers correctly, while stepping proves the result runs, which the"
            " terms' shapes and the sensor wiring can break without any field differing. It was"
            " opt-in for a while, which meant nothing in routine use ever ran it."
        ),
    )
    parser.add_argument(
        "--no-recheck-golden",
        dest="recheck_golden",
        action="store_false",
        help=(
            "Skip re-deriving the golden from main's live config. On by default: the golden is a"
            " file, and a file cannot notice that the config it was dumped from has moved on."
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

    num_envs = args.num_envs if args.num_envs is not None else golden["config"]["scene"]["num_envs"]
    compiled = IsaacSimAdapter().compile(flat_g1(), num_envs=num_envs, device="cuda:0")
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

    stale_golden = _recheck_golden(golden, num_envs) if args.recheck_golden else []
    if stale_golden:
        print(f"\nthe golden no longer matches {golden['source']}, which it is a recording of:")
        for difference in stale_golden:
            print(f"  {difference}")
        print(f"  re-dump it with scripts/dump_golden.py --cfg {golden['source']} --out {args.golden}")
    elif args.recheck_golden:
        # Said out loud, because a check that prints nothing when it passes is indistinguishable
        # from one that did not run.
        print(f"\ngolden still matches {golden['source']} as built here.")

    if args.construct:
        _step(compiled)

    # Deliberately without closing the app, and via os._exit. Isaac Sim's shutdown ends the process
    # itself with a status of zero, so a status set after it never reaches the caller and the check
    # becomes incapable of failing. Flush first: os._exit skips stdio buffers, and when this runs
    # with its output redirected that silently drops everything printed since the last flush --
    # which is how the stepping result went missing while the check still reported success.
    sys.stdout.flush()
    os._exit(1 if remaining or stale or stale_golden else 0)


def _recheck_golden(golden: dict, num_envs: int) -> list:
    """Rebuild main's config here and now, and check the golden still describes it.

    Everything else in this script measures the compiled task against a JSON file. Nothing measured
    the file against the thing it claims to be a recording of, so main's config could move and every
    check here would keep passing -- against a ruler that had quietly stopped describing the task it
    was cut from. That is not hypothetical: main's task was once unbuildable for long enough that
    the golden was dumped from a broken state, and no check said so.
    """
    import importlib

    from instinctlab.verify.structure import compare, dump, unexplained

    module_name, _, class_name = golden["source"].replace(":", ".").rpartition(".")
    cfg = getattr(importlib.import_module(module_name), class_name)()
    # The count is a property of a run, not of the declaration, and the golden records whichever one
    # it was dumped at. Compare like for like.
    cfg.scene.num_envs = num_envs
    return unexplained(compare(golden["config"], dump(cfg), allow={}))


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
