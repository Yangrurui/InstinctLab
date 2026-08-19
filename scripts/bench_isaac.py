"""Measure where a compiled Isaac environment's step time goes.

One environment per process: two cannot be built in one Isaac Sim session, because the stage
already holds the first one's clones. ``--profile`` adds a cProfile pass over the stepping loop,
which is the part worth seeing when the symptom is a busy CPU next to an idle GPU -- that shape
says the time is going somewhere in Python rather than into the simulation. It is how the contact
sensor's per-step rebuild of ``body_names`` was found.

    python scripts/bench_isaac.py --headless --device cuda:2 --profile

It used to take ``--which main`` and build ``G1FlatEnvCfg`` for comparison. That config was deleted
with D3; the compiled environment is the only one now.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _bench(env, steps: int, profile: bool) -> float:
    import torch

    actions = torch.zeros(env.action_space.shape, device=env.device)
    env.reset()
    for _ in range(10):  # warm up: the first steps pay for lazy buffers and kernel compilation
        env.step(actions)
    torch.cuda.synchronize()

    if profile:
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()

    started = time.perf_counter()
    for _ in range(steps):
        env.step(actions)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    if profile:
        profiler.disable()
        print("\ncumulative time, top 25:")
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(25)
        print("\nown time, top 25:")
        pstats.Stats(profiler).sort_stats("tottime").print_stats(25)

    return steps * env.num_envs / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--profile", action="store_true")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    AppLauncher(args)

    from instinctlab.engines.isaacsim import IsaacSimAdapter
    from instinctlab.tasks.locomotion.config.flat_g1 import flat_g1

    compiled = IsaacSimAdapter().compile(flat_g1(), num_envs=args.num_envs, device=args.device)
    env = compiled.make_env()

    fps = _bench(env, args.steps, args.profile)
    print(f"\n{fps:,.0f} environment steps per second at {args.num_envs} envs")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
