#!/usr/bin/env python3
"""Measure unified-env bridge cost for one simulator backend.

Isaac Sim and MJLab cannot share a process. Run them separately, for example::

    python scripts/profile_backend.py --backend mjlab --device cuda:1
    python scripts/profile_backend.py --backend isaacsim --device cuda:0 --num-envs 512
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

_INTERESTING_OPS = (
    "aten::cat",
    "aten::index_select",
    "aten::copy_",
    "aten::cross",
    "aten::linalg_vector_norm",
    "aten::linalg.vector_norm",
)


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return option in parser._option_string_actions


def _parse_args():
    from instinctlab.sim.backend import BACKENDS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("isaacsim", "mjlab", "mock"), required=True)
    parser.add_argument("--task", default="Instinct-Locomotion-Flat-G1-v0")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--log-json", default="")
    preliminary, _ = parser.parse_known_args()
    provider = BACKENDS.load(preliminary.backend)
    provider.add_cli_args(parser)
    if not _parser_has_option(parser, "--device"):
        parser.add_argument("--device", default="cuda:0" if preliminary.backend != "mock" else "cpu")
    return parser.parse_args(), provider


def _cuda_sync(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(device, fn) -> float:
    import torch

    _cuda_sync(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))
    begin = time.perf_counter()
    fn()
    return (time.perf_counter() - begin) * 1000.0


def _summarize_profiler(prof) -> dict[str, dict[str, float]]:
    events = getattr(prof, "key_averages", lambda: [])()
    summary: dict[str, dict[str, float]] = {}
    for event in events:
        key = event.key
        if key not in _INTERESTING_OPS and not any(token in key for token in ("copy", "index_select", "cat", "cross")):
            continue
        summary[key] = {
            "cpu_time_ms": float(event.cpu_time_total) / 1000.0,
            "cuda_time_ms": (
                float(getattr(event, "device_time_total", 0.0) or getattr(event, "cuda_time_total", 0.0)) / 1000.0
            ),
            "count": int(event.count),
        }
    return summary


def main() -> None:
    args, provider = _parse_args()
    bootstrap_context = provider.bootstrap(args)

    import torch

    from instinctlab.envs import UnifiedManagerBasedRLEnv
    from instinctlab.tasks import TASKS

    task = TASKS.get(args.task)
    if args.backend not in task.supported_backends and args.backend != "mock":
        supported = ", ".join(sorted(task.supported_backends))
        raise RuntimeError(f"task {args.task!r} does not support {args.backend!r}; supported: {supported}")

    backend = provider.create(device=args.device, bootstrap_context=bootstrap_context)
    env_cfg = task.make_env_cfg(num_envs=args.num_envs)
    env_cfg = replace(env_cfg, seed=args.seed)
    env = UnifiedManagerBasedRLEnv(env_cfg, backend)
    device = env.device
    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device)

    for _ in range(args.warmup):
        env.step(action)
    _cuda_sync(device)

    orig_step = env.backend.step
    orig_sync = env.backend.synchronize
    buckets = {"step": 0.0, "synchronize": 0.0, "reward": 0.0, "obs": 0.0, "policy_step": 0.0}

    def timed_backend_step() -> None:
        buckets["step"] += _time_ms(device, orig_step)

    def timed_synchronize(phase) -> None:
        from instinctlab.sim.backend import SensorReadPhase

        elapsed = _time_ms(device, lambda: orig_sync(phase))
        if phase is SensorReadPhase.POST_PHYSICS:
            buckets["synchronize"] += elapsed

    env.backend.step = timed_backend_step  # type: ignore[method-assign]
    env.backend.synchronize = timed_synchronize  # type: ignore[method-assign]

    orig_reward = env.reward_manager.compute
    orig_obs = env.observation_manager.compute
    env.reward_manager.compute = lambda dt: _record("reward", device, lambda: orig_reward(dt), buckets)  # type: ignore[method-assign]
    env.observation_manager.compute = lambda: _record("obs", device, orig_obs, buckets)  # type: ignore[method-assign]

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        for _ in range(args.steps):
            buckets["policy_step"] += _time_ms(device, lambda: env.step(action))

    env.backend.step = orig_step  # type: ignore[method-assign]
    env.backend.synchronize = orig_sync  # type: ignore[method-assign]
    env.reward_manager.compute = orig_reward  # type: ignore[method-assign]
    env.observation_manager.compute = orig_obs  # type: ignore[method-assign]

    policy_ms = buckets["policy_step"]
    sync_ms = buckets["synchronize"]
    bridge_ratio = sync_ms / policy_ms if policy_ms else 0.0
    fps = 1000.0 * args.steps / policy_ms if policy_ms else 0.0
    report = {
        "backend": args.backend,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "decimation": env.cfg.simulation.decimation,
        "device": str(device),
        "ms": {name: value / args.steps for name, value in buckets.items()},
        "bridge_ratio": bridge_ratio,
        "fps": fps,
        "ops": _summarize_profiler(prof),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.log_json:
        from pathlib import Path

        path = Path(args.log_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
    env.close()


def _record(name: str, device, fn, buckets: dict[str, float]):
    result_box: list[object] = []

    def wrapped() -> None:
        result_box.append(fn())

    buckets[name] += _time_ms(device, wrapped)
    return result_box[0]


if __name__ == "__main__":
    main()
