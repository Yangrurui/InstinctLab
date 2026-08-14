#!/usr/bin/env python3
"""Play a unified-task checkpoint and optionally record an offscreen video."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

_MANIFEST_FILENAME = "checkpoint_manifest.json"


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return option in parser._option_string_actions


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("isaacsim", "mjlab", "mock"), required=True)
    parser.add_argument("--task", default="Instinct-Locomotion-Flat-G1-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint file to load.")
    parser.add_argument("--video", action="store_true", help="Record one rgb_array video and exit.")
    parser.add_argument("--video-length", type=int, default=600, help="Policy steps to record.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Video path. Defaults to <checkpoint-dir>/videos/play/<checkpoint-stem>.mp4",
    )
    parser.add_argument(
        "--command",
        default="0.8,0.0,0.0",
        help="Pinned base-velocity command as vx,vy,wz. Empty string keeps the task sampler.",
    )
    parser.add_argument(
        "--viewer",
        choices=("auto", "native", "viser"),
        default="auto",
        help="Interactive viewer. auto uses native when a display is present, otherwise viser.",
    )
    parser.add_argument("--viser-port", type=int, default=8080, help="Port for the Viser web viewer.")
    return parser


def _parse_args():
    from instinctlab.sim.backend import BACKENDS

    parser = _base_parser()
    preliminary, _ = parser.parse_known_args()
    provider = BACKENDS.load(preliminary.backend)
    provider.add_cli_args(parser)
    if not _parser_has_option(parser, "--device"):
        parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda:0"
    return args, provider


def _validate_resume_manifest(resume: Path, *, schema, robot) -> None:
    from instinctlab.sim.schema import CheckpointManifest

    manifest_path = resume.expanduser().resolve().parent / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        return
    payload = json.loads(manifest_path.read_text())
    CheckpointManifest.validate_payload(payload, schema=schema, robot=robot)


def _parse_command(raw: str) -> tuple[float, float, float] | None:
    text = raw.strip()
    if not text:
        return None
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError("--command must be vx,vy,wz")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _prepare_play_cfg(env_cfg, *, pin_command: bool):
    events = dict(env_cfg.events)
    if "push" in events:
        events["push"] = replace(events["push"], interval_range_s=(1.0e9, 1.0e9))
    if "reset_root" in events:
        events["reset_root"] = replace(
            events["reset_root"],
            params={
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        )
    if "reset_joints" in events:
        events["reset_joints"] = replace(
            events["reset_joints"],
            params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
        )
    commands = dict(env_cfg.commands)
    if pin_command and "base_velocity" in commands:
        params = dict(commands["base_velocity"].params)
        params["rel_standing_envs"] = 0.0
        params["rel_heading_envs"] = 0.0
        params["resampling_time_range"] = (1.0e9, 1.0e9)
        commands["base_velocity"] = replace(commands["base_velocity"], params=params)
    return replace(env_cfg, events=events, commands=commands)


def _velocity_command(env):
    return dict(env.unwrapped.command_manager._terms)["base_velocity"]


def _resolve_viewer(name: str) -> str:
    if name != "auto":
        return name
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return "native"
    return "viser"


def _pin_command(env, command: tuple[float, float, float]) -> None:
    term = _velocity_command(env)
    term._standing[:] = False
    term._heading[:] = False
    term._time_left[:] = 1.0e9
    term._command[:, 0] = command[0]
    term._command[:, 1] = command[1]
    term._command[:, 2] = command[2]


def _write_video(env, policy, *, output: Path, length: int, command: tuple[float, float, float] | None) -> None:
    import torch

    import imageio.v2 as imageio

    output.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(round(1.0 / env.unwrapped.step_dt)))
    writer = imageio.get_writer(
        str(output),
        fps=fps,
        codec="libx264",
        quality=8,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        obs, _ = env.get_observations()
        if command is not None:
            _pin_command(env, command)
        for _ in range(length):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            if command is not None and bool(dones.any()):
                _pin_command(env, command)
            frame = env.unwrapped.render("rgb_array")
            if frame is None:
                raise RuntimeError("backend returned no rgb_array frame")
            writer.append_data(frame)
    finally:
        writer.close()


def main() -> None:
    args, provider = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    command = _parse_command(args.command)
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
    env_cfg = _prepare_play_cfg(task.make_env_cfg(num_envs=args.num_envs), pin_command=command is not None)
    env_cfg = replace(env_cfg, seed=args.seed)
    agent_cfg = task.make_agent_cfg(seed=args.seed, device=args.device)
    _validate_resume_manifest(checkpoint, schema=task.make_schema(), robot=env_cfg.scene.robot)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    env = InstinctRlVecEnvWrapper(
        UnifiedManagerBasedRLEnv(env_cfg, backend),
        policy_group=agent_cfg.policy_observation_group,
        critic_group=agent_cfg.critic_observation_group,
    )
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[INFO] Loading checkpoint: {checkpoint}")
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        if args.video:
            output = args.output
            if output is None:
                output = checkpoint.parent / "videos" / "play" / f"{checkpoint.stem}.mp4"
            else:
                output = output.expanduser().resolve()
            print(f"[INFO] Recording {args.video_length} steps to {output}")
            _write_video(env, policy, output=output, length=args.video_length, command=command)
            print(f"[INFO] Video saved: {output}")
            return

        viewer = _resolve_viewer(args.viewer)
        if viewer == "viser":
            if args.backend != "mjlab":
                raise RuntimeError("Viser is only available with --backend mjlab")
            from instinctlab.backends.mjlab.viser_play import play_with_viser

            play_with_viser(env, policy, command=command, port=args.viser_port)
            return

        obs, _ = env.get_observations()
        if command is not None:
            _pin_command(env, command)
        while True:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            if command is not None and bool(dones.any()):
                _pin_command(env, command)
            env.unwrapped.render("human")
    finally:
        env.close()


if __name__ == "__main__":
    main()
