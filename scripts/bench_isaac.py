"""Measure where an Isaac environment's step time goes.

Two environments cannot be built in one Isaac Sim process -- the stage already holds the clones --
so this builds one, chosen by ``--which``, and is run twice to compare. ``--profile`` adds a
cProfile pass over the stepping loop, which is the part worth seeing when the symptom is a busy
CPU next to an idle GPU: that shape says the time is going somewhere in Python rather than into
the simulation.

    python scripts/bench_isaac.py --which compiled --headless --device cuda:2 --profile
    python scripts/bench_isaac.py --which main --headless --device cuda:2 --profile
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
    parser.add_argument("--which", choices=("compiled", "main"), default="compiled")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--profile", action="store_true")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    AppLauncher(args)

    if args.which == "compiled":
        from instinctlab.engines.isaacsim import IsaacSimAdapter
        from instinctlab.tasks.locomotion.config.flat_g1 import flat_g1

        compiled = IsaacSimAdapter().compile(flat_g1(), num_envs=args.num_envs, device=args.device)
        env = compiled.make_env()
    else:
        from isaaclab.envs import ManagerBasedRLEnv

        from instinctlab.tasks.locomotion.config.g1.flat_env_cfg import G1FlatEnvCfg

        cfg = G1FlatEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        env = ManagerBasedRLEnv(cfg=cfg)

    fps = _bench(env, args.steps, args.profile)
    print(f"\n{args.which}: {fps:,.0f} environment steps per second at {args.num_envs} envs")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
