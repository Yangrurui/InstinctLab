"""Play one checkpoint on whichever engine ``--engine`` names.

Same ordering rules as ``scripts/train.py``: choose the engine, let it add its flags, bootstrap,
then import the rest. The launcher does not name an engine in its own logic, and it does not
construct a viewer -- each adapter decides how ``--viewer viser`` is implemented on that engine.

Isaac Sim has no native Viser backend. Its adapter plays the checkpoint in mjlab's
``ViserPlayViewer`` -- the same viewer mjlab training uses.

Usage::

    python scripts/play.py --engine isaacsim --task Instinct-Velocity-Flat-G1 --viewer viser --headless
    python scripts/play.py --engine mjlab --task Instinct-Velocity-Flat-G1 --viewer viser
    python scripts/play.py --engine mjlab --task Instinct-Parkour-Target-G1 --viewer viser --agent random
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

if sys.path and os.path.isdir(os.path.join(sys.path[0], "instinct_rl")):
    sys.path.pop(0)


def _parse() -> argparse.Namespace:
    from instinctlab.engines import names

    chooser = argparse.ArgumentParser(add_help=False)
    chooser.add_argument("--engine", type=str, required=True, choices=names())
    chosen, _ = chooser.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__, parents=[chooser])
    parser.add_argument("--task", type=str, default="Instinct-Velocity-Flat-G1", help="Task id to play.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
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
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail instead of skipping when the engine cannot express an optional term.",
    )

    from instinctlab.engines import adapter as _adapter

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
    if not root.is_dir():
        raise FileNotFoundError(f"no log root at {root}; pass --checkpoint")
    runs = [path for path in root.iterdir() if path.is_dir()]
    if args.load_run:
        runs = [path for path in runs if path.name == args.load_run]
    runs.sort()
    for run in reversed(runs):
        models = sorted(run.glob("model_*.pt"), key=_checkpoint_iteration)
        if models:
            return models[-1]
    raise FileNotFoundError(f"no model_*.pt under {root}")


def _checkpoint_iteration(path: Path) -> tuple[int, str]:
    """Sort checkpoints by their numeric iteration, with a stable name fallback."""
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return (int(match.group(1)) if match else -1, path.name)


def main() -> None:
    args = _parse()

    from instinctlab.engines import adapter as engine_adapter

    engine = engine_adapter(args.engine)
    # Viser must be imported before Isaac Sim's AppLauncher prepends Kit's pip_prebundle:
    # that bundle ships an older websockets without ``asyncio.server``.
    if _resolve_viewer(args.viewer) == "viser":
        import viser  # noqa: F401
    app = engine.bootstrap(args)

    from instinctlab.play.env import PlayEnv
    from instinctlab.tasks.registry import spec as task_spec

    spec = task_spec(args.task)
    compiled = engine.compile(spec, num_envs=args.num_envs, device=args.device, strict=args.strict)
    _silence_observation_noise(compiled.env_cfg)
    compiled.env_cfg.seed = compiled.agent_cfg.seed
    print(compiled.resolution.summary_table())

    env = engine.wrap_for_rl(compiled.make_env())
    dummy = args.agent in {"zero", "random"}
    reload_policy = None
    checkpoint_dir = None
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
        checkpoint = _resolve_checkpoint(args, compiled.agent_cfg.experiment_name)
        print(f"[INFO] Loading {checkpoint}", flush=True)
        from instinct_rl.runners import OnPolicyRunner

        agent_cfg = compiled.agent_cfg
        agent_cfg.device = args.device
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=args.device)

        def reload_policy(path: str):
            runner.load(path)
            return runner.get_inference_policy(device=args.device)

        checkpoint_dir = checkpoint.parent

    viewer = _resolve_viewer(args.viewer)
    from instinctlab.play.viser import enable_pose_command_debug_vis

    enable_pose_command_debug_vis(env)
    print(f"[INFO] Playing {args.task} on {args.engine} with {viewer}", flush=True)
    engine.play(
        PlayEnv(env),
        policy,
        viewer=viewer,
        robot=spec.robot,
        spec=spec,
        port=args.port,
        reload_policy=reload_policy,
        checkpoint_dir=checkpoint_dir,
        strict=args.strict,
    )
    env.close()
    if app is not None:
        app.close()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
