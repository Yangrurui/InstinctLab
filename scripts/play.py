"""Play one checkpoint on whichever engine ``--engine`` names.

Same ordering rules as ``scripts/train.py``: choose the engine, let it add its flags, bootstrap,
then import the rest. Environment compilation belongs to the selected adapter; viewer dispatch is
an application-level extension under ``instinctlab.play``.

Isaac Sim has no native Viser backend. The playback layer runs that checkpoint in mjlab's
``ViserPlayViewer`` -- the same viewer mjlab training uses.

Usage::

    python scripts/play.py --engine isaacsim --task Instinct-Velocity-Flat-G1 --viewer viser --headless
    python scripts/play.py --engine mjlab --task Instinct-Velocity-Flat-G1 --viewer viser
    python scripts/play.py --engine mjlab --task Instinct-Parkour-Target-G1 --viewer viser --agent random
"""

from __future__ import annotations

import argparse
import json
import os
import re
from contextlib import ExitStack
from pathlib import Path


def _parse() -> argparse.Namespace:
    from instinctlab_engine import names

    chooser = argparse.ArgumentParser(add_help=False)
    chooser.add_argument("--engine", type=str, required=True, choices=names())
    chosen, _ = chooser.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__, parents=[chooser])
    parser.add_argument("--task", type=str, default="Instinct-Velocity-Flat-G1", help="Task id to play.")
    parser.add_argument("--num_envs", type=int, default=10, help="Number of environments to simulate.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a model_*.pt file.")
    parser.add_argument(
        "--agent",
        type=str,
        default="trained",
        choices=("trained", "zero", "random"),
        help="trained loads a checkpoint; zero/random skip it and ignore the observation.",
    )
    parser.add_argument(
        "--load_run", type=str, default=None, help="Run directory name under logs/<engine>/. Latest if omitted."
    )
    parser.add_argument("--logroot", type=str, default=None, help="Override the log root, default logs/<engine>/.")
    parser.add_argument(
        "--viewer",
        type=str,
        default="auto",
        choices=("auto", "native", "viser"),
        help="auto picks viser when there is no display.",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser port.")
    parser.add_argument("--export-onnx", action="store_true", help="Export the loaded policy and normalizer.")
    parser.add_argument(
        "--export-dir", type=str, default=None, help="ONNX output directory; defaults beside checkpoint."
    )
    parser.add_argument("--export-only", action="store_true", help="Exit after ONNX export without opening a viewer.")
    parser.add_argument(
        "--allow-nonclean-resolution",
        action="store_true",
        help=(
            "Allow skipped, emulated, or profile-omitted terms. Playback defaults to a clean "
            "strict compilation."
        ),
    )

    from instinctlab_engine import adapter as _adapter

    _adapter(chosen.engine).add_cli_args(parser)
    return parser.parse_args()


def _resolve_viewer(name: str) -> str:
    if name != "auto":
        return name
    return "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"


def _silence_observation_noise(env_cfg: object) -> None:
    groups = env_cfg.observations
    items = groups.values() if isinstance(groups, dict) else vars(groups).values()
    for group in items:
        if hasattr(group, "enable_corruption"):
            group.enable_corruption = False


def _resolve_checkpoint(args: argparse.Namespace, experiment: str) -> Path:
    if args.checkpoint:
        path = Path(args.checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path
    root = Path(args.logroot or os.path.join("logs", args.engine, experiment)).resolve()
    from instinctlab.checkpoint import latest_run_checkpoint

    run_pattern = re.escape(args.load_run) if args.load_run else ".*"
    return latest_run_checkpoint(root, run_pattern=run_pattern, skip_empty_runs=True)


def _validate_args(args: argparse.Namespace) -> None:
    if args.export_only and not args.export_onnx:
        raise ValueError("--export-only requires --export-onnx")
    if args.export_onnx and args.agent != "trained":
        raise ValueError("ONNX export requires --agent trained")
    if args.export_onnx and args.num_envs != 1:
        raise ValueError("ONNX export requires --num_envs 1")


def _play(args, engine, resources: ExitStack) -> None:
    from instinctlab.play.env import PlayEnv
    from instinctlab.tasks.registry import asset_id, checkpoint_task_id
    from instinctlab.tasks.registry import spec as task_spec

    robot = engine.robot_spec(asset_id(args.task))
    spec = task_spec(args.task, robot)
    from instinctlab_engine.preflight import require_preflight

    require_preflight(
        spec,
        args.engine,
        selected_adapter=engine,
        allow_nonclean=args.allow_nonclean_resolution,
    )
    compiled = engine.compile(
        spec,
        num_envs=args.num_envs,
        device=args.device,
        strict=not args.allow_nonclean_resolution,
    )
    compiled.resolution.require_clean(
        allow_nonclean=args.allow_nonclean_resolution
    )
    agent_cfg = compiled.agent_cfg
    agent_cfg.device = args.device
    agent_config = agent_cfg.to_dict()
    _silence_observation_noise(compiled.env_cfg)
    compiled.env_cfg.seed = agent_cfg.seed
    print(compiled.resolution.summary_table())

    dummy = args.agent in {"zero", "random"}
    reload_policy = None
    checkpoint_dir = None
    if dummy:
        checkpoint = None
    else:
        checkpoint = _resolve_checkpoint(args, agent_cfg.experiment_name)
        from instinctlab.checkpoint import validate_checkpoint_contract

        validate_checkpoint_contract(
            checkpoint,
            spec,
            checkpoint_task_id=checkpoint_task_id(args.task),
        )
        print(f"[INFO] Loading {checkpoint}", flush=True)

    native_env = compiled.make_env()
    resources.callback(native_env.close)
    env = engine.wrap_for_rl(native_env)
    if dummy:
        import torch

        action_shape = tuple(env.unwrapped.action_space.shape)
        device = env.unwrapped.device
        if args.agent == "zero":

            def policy(obs):
                del obs
                return torch.zeros(action_shape, device=device)

        else:

            def policy(obs):
                del obs
                return 2 * torch.rand(action_shape, device=device) - 1

        print(f"[INFO] Using {args.agent} actions (no checkpoint)", flush=True)
    else:
        assert checkpoint is not None
        from instinct_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(env, agent_config, log_dir=None, device=args.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=args.device)

        def reload_policy(path: str):
            runner.load(path)
            return runner.get_inference_policy(device=args.device)

        checkpoint_dir = checkpoint.parent
        if args.export_onnx:
            export_dir = (
                Path(args.export_dir).expanduser().resolve() if args.export_dir else checkpoint_dir / "exported"
            )
            export_dir.mkdir(parents=True, exist_ok=True)
            obs, _ = env.get_observations()
            runner.export_as_onnx(obs, str(export_dir))
            from instinctlab.checkpoint import task_contract

            with (export_dir / "export.json").open("w") as handle:
                json.dump(
                    {
                        "checkpoint": str(checkpoint),
                        "checkpoint_task_id": checkpoint_task_id(args.task),
                        "task_contract": task_contract(
                            spec, agent_config=agent_config
                        ),
                        "allow_nonclean_resolution": bool(
                            args.allow_nonclean_resolution
                        ),
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
            print(f"[INFO] Exported ONNX policy to {export_dir}", flush=True)

    if args.export_only:
        return

    viewer = _resolve_viewer(args.viewer)
    from instinctlab.play.viser import (
        enable_camera_debug_vis,
        enable_pose_command_debug_vis,
        enable_volume_points_debug_vis,
    )

    enable_pose_command_debug_vis(env)
    enable_volume_points_debug_vis(env)
    enable_camera_debug_vis(env)
    print(f"[INFO] Playing {args.task} on {args.engine} with {viewer}", flush=True)
    from instinctlab.play import play

    play(
        args.engine,
        viewer,
        PlayEnv(env),
        policy,
        robot=spec.robot,
        spec=spec,
        port=args.port,
        reload_policy=reload_policy,
        checkpoint_dir=checkpoint_dir,
        strict=not args.allow_nonclean_resolution,
    )


def main() -> int:
    args = _parse()
    _validate_args(args)

    from instinctlab_engine import adapter as engine_adapter

    engine = engine_adapter(args.engine)
    # Viser must be imported before Isaac Sim's AppLauncher prepends Kit's pip_prebundle:
    # that bundle ships an older websockets without ``asyncio.server``.
    if _resolve_viewer(args.viewer) == "viser":
        import viser  # noqa: F401

    with ExitStack() as resources:
        app = engine.bootstrap(args)
        if app is not None:
            resources.callback(app.close)
        _play(args, engine, resources)

    return engine.finalize_process(0)


if __name__ == "__main__":
    raise SystemExit(main())
