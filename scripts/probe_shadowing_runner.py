"""Exercise shadowing runner, checkpoint validation, reload and ONNX export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path

if sys.path and os.path.isdir(os.path.join(sys.path[0], "instinct_rl")):
    sys.path.pop(0)


def _parse():
    from instinctlab.engines import adapter, names

    chooser = argparse.ArgumentParser(add_help=False)
    chooser.add_argument("--engine", required=True, choices=names())
    selected, _ = chooser.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__, parents=[chooser])
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--task", default="Instinct-Shadowing-WholeBody-Plane-G1-v0")
    parser.add_argument("--motion", required=True, help="Motion clip used by the diagnostic task override.")
    adapter(selected.engine).add_cli_args(parser)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close_runner_writer(runner: object) -> None:
    writer = getattr(runner, "writer", None)
    if writer is not None:
        writer.close()


def _probe(args, engine, resources: ExitStack) -> None:
    import torch

    from instinct_rl.runners import OnPolicyRunner

    from instinctlab.checkpoint import add_task_contract, validate_checkpoint_contract
    from instinctlab.shadowing_probe import shadowing_task_with_motion

    task = shadowing_task_with_motion(args.task, args.motion)
    compiled = engine.compile(task, num_envs=1, device=args.device, strict=True)
    compiled.env_cfg.seed = args.seed
    agent_cfg = compiled.agent_cfg
    agent_cfg.device = args.device
    agent_cfg.num_steps_per_env = 2
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    native_env = compiled.make_env()
    resources.callback(native_env.close)
    env = engine.wrap_for_rl(native_env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    resources.callback(_close_runner_writer, runner)
    checkpoint = args.artifact_dir / "model_0.pt"
    runner.save(str(checkpoint), infos={"probe": True})
    (args.artifact_dir / "manifest.json").write_text(
        json.dumps(
            add_task_contract(compiled.resolution.manifest(), task),
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    validate_checkpoint_contract(checkpoint, task)
    runner.load(str(checkpoint))
    observations, _ = env.get_observations()
    action = runner.get_inference_policy(device=args.device)(observations)
    env.step(action)
    export_dir = args.artifact_dir / "exported"
    export_dir.mkdir(exist_ok=True)
    runner.export_as_onnx(observations, str(export_dir))
    artifacts = sorted(path for path in args.artifact_dir.rglob("*") if path.is_file())
    report = {
        "engine": args.engine,
        "seed": args.seed,
        "device": args.device,
        "task": task.task_id,
        "action_shape": list(action.shape),
        "action_finite": bool(torch.isfinite(action).all()),
        "checkpoint_iteration": int(runner.current_learning_iteration),
        "artifacts": {
            str(path.relative_to(args.artifact_dir)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifacts
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Wrote runner probe report to {args.report}")


def main() -> None:
    args = _parse()
    motion = Path(args.motion).expanduser().resolve()
    if not motion.is_file():
        raise FileNotFoundError(f"shadowing probe motion clip not found: {motion}")
    args.motion = str(motion)
    from instinctlab.engines import adapter

    engine = adapter(args.engine)
    with ExitStack() as resources:
        app = engine.bootstrap(args)
        if app is not None:
            resources.callback(app.close)
        _probe(args, engine, resources)

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
