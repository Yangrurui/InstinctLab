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
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path


def _parse() -> argparse.Namespace:
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
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint.")
    parser.add_argument("--distributed", action="store_true", help="Enable torchrun distributed training.")
    parser.add_argument("--local-rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--load_run", type=str, default=None, help="Run directory name or regular expression to resume."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint path or filename expression to resume."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail instead of skipping when the engine cannot express an optional term.",
    )
    parser.add_argument(
        "--log_terrain_split",
        action="store_true",
        default=False,
        help=(
            "Log episode length/reward per sub-terrain name. Measured at ~2.7 ms/step "
            "(~9%% of a wrapped step at 16 envs); the bookkeeping is Python over the "
            "named types, not the overflow poll. Off by default. Overflow refusal "
            "is separate and stays on."
        ),
    )
    parser.add_argument(
        "--allow_contact_overflow",
        action="store_true",
        default=False,
        help=(
            "Do not refuse when mujoco_warp d.overflow is set, or when PhysX "
            "GPU collision-stack / patch occupancy exceeds the allocated "
            "budget. Contacts are still dropped. Sets "
            "INSTINCTLAB_ALLOW_CONTACT_OVERFLOW=1."
        ),
    )

    # Contributes the engine's launch flags, and ``--device``, which the engines insist on owning.
    from instinctlab.engines import adapter as _adapter

    _adapter(chosen.engine).add_cli_args(parser)
    return parser.parse_args()


def _log_dir(args: argparse.Namespace, experiment: str) -> str:
    root = args.logroot or os.path.abspath(os.path.join("logs", args.engine, experiment))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        stamp += f"_{args.run_name}"
    return os.path.join(root, stamp)


def _resolve_resume_checkpoint(args: argparse.Namespace, agent_cfg: object) -> Path | None:
    """Resolve the legacy runner's run/checkpoint expressions without engine imports."""
    requested = bool(args.resume or args.load_run or args.checkpoint or getattr(agent_cfg, "resume", False))
    if not requested:
        return None

    if args.checkpoint:
        explicit = Path(args.checkpoint).expanduser()
        if explicit.is_file():
            return explicit.resolve()

    root = Path(args.logroot or os.path.join("logs", args.engine, agent_cfg.experiment_name)).resolve()
    run_selector = args.load_run if args.load_run is not None else getattr(agent_cfg, "load_run", ".*")
    run_path = Path(str(run_selector)).expanduser() if run_selector else None
    if run_path is not None and run_path.is_absolute() and run_path.is_dir():
        run_root = run_path.resolve()
        run_pattern = ".*"
    else:
        run_root = root
        run_pattern = str(run_selector or ".*")

    checkpoint_selector = args.checkpoint or getattr(agent_cfg, "load_checkpoint", r"model_.*.pt")
    from instinctlab.checkpoint import latest_checkpoint, latest_run_checkpoint

    if run_path is not None and run_path.is_absolute() and run_path.is_dir():
        return latest_checkpoint(run_root, str(checkpoint_selector))
    return latest_run_checkpoint(
        run_root,
        run_pattern=run_pattern,
        checkpoint_pattern=str(checkpoint_selector),
    )


def _close_runner_writer(runner: object) -> None:
    writer = getattr(runner, "writer", None)
    if writer is not None:
        writer.close()


def _train(args, engine, distributed, resources: ExitStack) -> None:
    from instinctlab.tasks.registry import spec as task_spec

    spec = task_spec(args.task, args.engine)
    compiled = engine.compile(spec, num_envs=args.num_envs, device=args.device, strict=args.strict)

    agent_cfg = compiled.agent_cfg
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    agent_cfg.device = args.device
    if args.load_run is not None:
        agent_cfg.load_run = args.load_run
    if args.checkpoint is not None:
        agent_cfg.load_checkpoint = args.checkpoint
    agent_cfg.resume = bool(args.resume or args.load_run or args.checkpoint or agent_cfg.resume)

    resume_path = _resolve_resume_checkpoint(args, agent_cfg)
    if resume_path is not None:
        from instinctlab.checkpoint import validate_checkpoint_contract

        validate_checkpoint_contract(resume_path, spec)

    from instinctlab.training import shared_run_directory

    log_dir = shared_run_directory(_log_dir(args, agent_cfg.experiment_name), distributed)
    os.makedirs(log_dir, exist_ok=True)

    print(compiled.resolution.summary_table())
    manifest_path = os.path.join(log_dir, "manifest.json")
    from instinctlab.checkpoint import add_task_contract

    manifest = add_task_contract(compiled.resolution.manifest(), spec)
    manifest["distributed"] = {
        "enabled": distributed.enabled,
        "world_size": distributed.world_size,
        "rank_seed_rule": "agent_seed + global_rank",
        "rank_seeds": [agent_cfg.seed + rank for rank in range(distributed.world_size)],
    }
    manifest["resume_environment_state"] = "fresh reset; simulator and motion runtime are resampled"
    if distributed.is_primary:
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, default=str)
        print(f"[INFO] Wrote the compilation manifest to {manifest_path}")

    # The environment seeds itself from its own config, and both reference training scripts hand it
    # the agent's seed to do that with. Left unset it defaults to None on both engines, which means
    # no seeding at all -- runs still look fine and are simply not reproducible, and the randomised
    # mass, friction and pushes come from wherever the global RNG happened to be.
    compiled.env_cfg.seed = distributed.seed(agent_cfg.seed)

    if args.allow_contact_overflow:
        os.environ["INSTINCTLAB_ALLOW_CONTACT_OVERFLOW"] = "1"

    native_env = compiled.make_env()
    resources.callback(native_env.close)
    env = engine.wrap_for_rl(native_env)
    if args.log_terrain_split:
        from instinctlab.utils.terrain_split_log import attach_terrain_split

        env = attach_terrain_split(env)

    from instinct_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    resources.callback(_close_runner_writer, runner)
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        print(f"[INFO] Loading training checkpoint from {resume_path}")
        from instinctlab.training import load_runner_checkpoint

        load_runner_checkpoint(runner, resume_path, distributed)

    if distributed.is_primary:
        with open(os.path.join(log_dir, "agent.json"), "w") as handle:
            json.dump(agent_cfg.to_dict(), handle, indent=2, sort_keys=True, default=str)

    print(f"[INFO] Training {args.task} on {args.engine}: {args.num_envs} envs on {args.device}, logs in {log_dir}")
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=getattr(agent_cfg, "init_at_random_ep_len", False),
    )


def main() -> None:
    args = _parse()

    from instinctlab.training import distributed_run, rank_device

    distributed = distributed_run(args.distributed, args.local_rank)
    args.device = rank_device(args.device, distributed)

    from instinctlab.engines import adapter as engine_adapter

    engine = engine_adapter(args.engine)
    with ExitStack() as resources:
        # Must precede every engine import. For mjlab this does nothing, which is the point: the
        # launcher does not need to know that only one engine has an application runtime.
        app = engine.bootstrap(args)
        if app is not None:
            resources.callback(app.close)

        from instinctlab.training import destroy_process_group, initialize_process_group

        initialize_process_group(distributed)
        resources.callback(destroy_process_group, distributed)
        _train(args, engine, distributed, resources)

    # Isaac Sim's shutdown can hang on teardown after a long run; the process is done either way.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
