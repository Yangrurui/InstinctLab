"""Train one task on whichever engine ``--engine`` names.

The two things this file is careful about are both ordering problems.

First, an engine must be chosen before anything is imported. Isaac Sim's ``AppLauncher`` has to run
before ``isaaclab`` -- and before torch -- so ``--engine`` is parsed by itself, then the chosen
adapter contributes its own flags, then ``bootstrap`` runs, and only after that does the rest of
the program come into existence. This is why the imports below are not at the top of the file.

Second, nothing here knows which engine it got. The adapter compiles the task, names its own RL
wrapper, and reports what it did with the declaration; the training loop that follows is the same
code in both cases. Anything that reads like ``if engine == ...`` belongs in an adapter instead,
and its absence here is the property worth protecting when a third engine arrives.

Usage::

    python scripts/train.py --engine isaacsim --task Instinct-Velocity-Flat-G1 --num_envs 4096
    python scripts/train.py --engine mjlab --task Instinct-Velocity-Flat-G1 --num_envs 4096
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

# Python puts this script's directory first on the path, and this directory contains a folder named
# ``instinct_rl`` holding the legacy per-engine entry points. That folder has no ``__init__.py``, so
# it becomes a namespace package that shadows the installed ``instinct_rl`` library, and importing
# the runner fails with a name that does exist. Drop the entry rather than rename the folder, which
# would break the paths people already invoke.
if sys.path and os.path.isdir(os.path.join(sys.path[0], "instinct_rl")):
    sys.path.pop(0)


def _parse() -> tuple[argparse.Namespace, list[str]]:
    """Choose the engine first, then let it add its own flags.

    Two passes because the flags an engine wants are not knowable before the engine is. The first
    parser is deliberately incomplete and ignores what it does not recognise.
    """
    from instinctlab.engines import names

    # Required rather than defaulted: a default would have to name one engine, and a run that
    # silently picks an engine is a run whose logs cannot be trusted to say what produced them.
    chooser = argparse.ArgumentParser(add_help=False)
    chooser.add_argument("--engine", type=str, required=True, choices=names())
    chosen, _ = chooser.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__, parents=[chooser])
    parser.add_argument("--task", type=str, default="Instinct-Velocity-Flat-G1", help="Task id to train.")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Override the agent's seed.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Override the agent's iteration count.")
    parser.add_argument("--logroot", type=str, default=None, help="Override the log root, default logs/<engine>/.")
    parser.add_argument("--run_name", type=str, default="", help="Suffix appended to the run directory.")
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail instead of skipping when the engine cannot express an optional term.",
    )

    # Contributes the engine's launch flags, and ``--device``, which the engines insist on owning.
    from instinctlab.engines import adapter as _adapter

    _adapter(chosen.engine).add_cli_args(parser)
    return parser.parse_known_args()


def _log_dir(args: argparse.Namespace, experiment: str) -> str:
    root = args.logroot or os.path.abspath(os.path.join("logs", args.engine, experiment))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        stamp += f"_{args.run_name}"
    return os.path.join(root, stamp)


def main() -> None:
    args, _unknown = _parse()

    from instinctlab.engines import adapter as engine_adapter

    engine = engine_adapter(args.engine)

    # Must precede every engine import. For mjlab this does nothing, which is the point: the
    # launcher does not need to know that only one of the two engines has a runtime to start.
    app = engine.bootstrap(args)

    import torch

    from instinctlab.tasks.registry import spec as task_spec

    spec = task_spec(args.task)
    compiled = engine.compile(spec, num_envs=args.num_envs, device=args.device, strict=args.strict)

    agent_cfg = compiled.agent_cfg
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    agent_cfg.device = args.device

    log_dir = _log_dir(args, agent_cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    print(compiled.resolution.summary_table())
    manifest_path = os.path.join(log_dir, "manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump(compiled.resolution.manifest(), handle, indent=2, sort_keys=True, default=str)
    print(f"[INFO] Wrote the compilation manifest to {manifest_path}")

    torch.manual_seed(agent_cfg.seed)

    env = compiled.make_env()
    env = engine.wrap_for_rl(env)

    from instinct_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    with open(os.path.join(log_dir, "agent.json"), "w") as handle:
        json.dump(agent_cfg.to_dict(), handle, indent=2, sort_keys=True, default=str)

    print(f"[INFO] Training {args.task} on {args.engine}: {args.num_envs} envs on {args.device}, logs in {log_dir}")
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=getattr(agent_cfg, "init_at_random_ep_len", False),
    )

    env.close()
    if app is not None:
        app.close()
    # Isaac Sim's shutdown can hang on teardown after a long run; the process is done either way.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
