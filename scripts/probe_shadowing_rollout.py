"""Save a fixed-seed short shadowing rollout for one engine."""

from __future__ import annotations

import argparse
import json
import os
import sys

if sys.path and os.path.isdir(os.path.join(sys.path[0], "instinct_rl")):
    sys.path.pop(0)


def _parse():
    from instinctlab.engines import adapter, names

    chooser = argparse.ArgumentParser(add_help=False)
    chooser.add_argument("--engine", required=True, choices=names())
    selected, _ = chooser.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__, parents=[chooser])
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", required=True)
    adapter(selected.engine).add_cli_args(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    from instinctlab.engines import adapter

    engine = adapter(args.engine)
    app = engine.bootstrap(args)

    import numpy as np

    from instinctlab.shadowing_probe import collect_shadowing_rollout, shadowing_fallback_task

    task = shadowing_fallback_task()
    compiled = engine.compile(task, num_envs=args.num_envs, device=args.device, strict=True)
    compiled.env_cfg.seed = args.seed
    env = compiled.make_env()
    try:
        state = collect_shadowing_rollout(env, task, steps=args.steps)
        state["metadata"] = np.asarray(
            json.dumps(
                {
                    "engine": args.engine,
                    "seed": args.seed,
                    "steps": args.steps,
                    "device": args.device,
                    "task": task.task_id,
                    "motion": task.scene.motion_references[0].clip,
                },
                sort_keys=True,
            )
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez_compressed(args.out, **state)
        print(f"[INFO] Saved {args.engine} shadowing rollout to {args.out}")
    finally:
        env.close()
        if app is not None:
            app.close()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
