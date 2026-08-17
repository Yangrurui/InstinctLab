#!/usr/bin/env python3
"""Train one canonical task on a selected simulator backend."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

_MANIFEST_FILENAME = "checkpoint_manifest.json"


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return option in parser._option_string_actions


def _base_parser() -> argparse.ArgumentParser:
    from instinctlab.sim.backend import BACKENDS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS.names(), required=True)
    parser.add_argument("--task", default="Instinct-Locomotion-Flat-G1-v0")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--log-root", type=Path, default=Path("logs/instinct_rl"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint file to resume from.")
    return parser


def _parse_args():
    from instinctlab.sim.backend import BACKENDS

    parser = _base_parser()
    preliminary, _ = parser.parse_known_args()
    provider = BACKENDS.load(preliminary.backend)
    provider.add_cli_args(parser)
    # Isaac Sim's AppLauncher already registers --device; other backends do not.
    if not _parser_has_option(parser, "--device"):
        parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(), provider


def _write_manifest(run_dir, *, task_id, schema, robot, backend, seed) -> None:
    from instinctlab.sim.schema import CheckpointManifest

    manifest = CheckpointManifest(
        task=task_id,
        schema=schema,
        joint_names=robot.joint_names,
        body_names=robot.body_names,
        asset_id=robot.asset_id,
        backend=backend.metadata.name,
        backend_version=backend.metadata.engine_version,
        seed=seed,
        metadata={"schema_version": schema.version},
    )
    (run_dir / _MANIFEST_FILENAME).write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True))


def _validate_resume_manifest(resume: Path, *, schema, robot) -> None:
    from instinctlab.sim.schema import CheckpointManifest

    manifest_path = resume.expanduser().resolve().parent / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        # Older checkpoints predate manifests; nothing to validate against.
        return
    payload = json.loads(manifest_path.read_text())
    CheckpointManifest.validate_payload(payload, schema=schema, robot=robot)


def main() -> None:
    args, provider = _parse_args()
    bootstrap_context = provider.bootstrap(args)

    import torch

    from instinct_rl.runners import OnPolicyRunner

    from instinctlab.envs import UnifiedManagerBasedRLEnv
    from instinctlab.rl import InstinctRlVecEnvWrapper
    from instinctlab.tasks import TASKS

    task = TASKS.get(args.task)
    if args.backend not in task.supported_backends:
        supported = ", ".join(sorted(task.supported_backends))
        raise RuntimeError(f"task {args.task!r} does not support {args.backend!r}; supported: {supported}")

    backend = provider.create(device=args.device, bootstrap_context=bootstrap_context)
    env_cfg = task.make_env_cfg(num_envs=args.num_envs)
    env_cfg = replace(env_cfg, seed=args.seed)
    agent_cfg = task.make_agent_cfg(seed=args.seed, device=args.device, run_name=args.run_name)
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    schema = task.make_schema()
    robot = env_cfg.scene.robot
    if args.resume is not None:
        _validate_resume_manifest(args.resume, schema=schema, robot=robot)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_dir = (
        args.log_root.expanduser().resolve()
        / agent_cfg.experiment_name
        / f"{datetime.now():%Y%m%d_%H%M%S}{f'_{args.run_name}' if args.run_name else ''}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    env = InstinctRlVecEnvWrapper(
        UnifiedManagerBasedRLEnv(env_cfg, backend),
        policy_group=agent_cfg.policy_observation_group,
        critic_group=agent_cfg.critic_observation_group,
    )
    _write_manifest(
        run_dir,
        task_id=args.task,
        schema=schema,
        robot=robot,
        backend=backend,
        seed=args.seed,
    )
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(run_dir), device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)
        if args.resume is not None:
            runner.load(str(args.resume.expanduser().resolve()))
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=agent_cfg.init_at_random_ep_len,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
