#!/usr/bin/env python3
"""Measure unified-env bridge cost for one simulator backend.

Isaac Sim and MJLab cannot share a process. Run them separately, for example::

    python scripts/profile_backend.py --backend mjlab --device cuda:1
    python scripts/profile_backend.py --backend isaacsim --device cuda:0 --num-envs 512

``field_ms`` splits one synchronize. ``ms`` splits one policy step, including
mjlab ``write_data_to_sim`` / ``sim_step`` / ``scene_update``. Zero-action
rollouts fall and reset; pass ``--no-reset`` for a steady-state split.
4096-env mjlab --no-reset (2026-08-17): PD write ~11.5 ms, Warp step ~9.9 ms,
obs/reward ~2.2 ms each, synchronize ~0.4 ms. Do not cut copies, formulas,
or the solver from these numbers. See MULTI_ENGINE_TRAINING.md 2.5.
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
    parser.add_argument("--backend", choices=BACKENDS.names(), required=True)
    parser.add_argument("--task", default="Instinct-Locomotion-Flat-G1-v0")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--log-json", default="")
    parser.add_argument(
        "--aten-ops",
        action="store_true",
        help="Wrap the timed loop in torch.profiler. Inflates policy_step; off by default.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Suppress episode resets so zero-action fall/reset does not dominate the breakdown.",
    )
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

    if args.no_reset:
        env._reset_idx = lambda ids: None  # type: ignore[method-assign]
        env.termination_manager.compute = lambda: (  # type: ignore[method-assign]
            torch.zeros(env.num_envs, device=device, dtype=torch.bool),
            torch.zeros(env.num_envs, device=device, dtype=torch.bool),
        )

    for _ in range(args.warmup):
        env.step(action)
    _cuda_sync(device)

    orig_step = env.backend.step
    orig_sync = env.backend.synchronize
    orig_apply = env.action_manager.apply_action
    orig_process = env.action_manager.process_action
    orig_term = env.termination_manager.compute
    orig_command = env.command_manager.compute
    orig_event = env.event_manager.apply
    orig_reset = env._reset_idx
    buckets = {
        "step": 0.0,
        "write_data_to_sim": 0.0,
        "sim_step": 0.0,
        "scene_update": 0.0,
        "synchronize": 0.0,
        "process_action": 0.0,
        "apply_action": 0.0,
        "termination": 0.0,
        "reward": 0.0,
        "command": 0.0,
        "event": 0.0,
        "reset": 0.0,
        "obs": 0.0,
        "policy_step": 0.0,
    }

    def timed_backend_step() -> None:
        buckets["step"] += _time_ms(device, orig_step)

    def timed_synchronize(phase) -> None:
        from instinctlab.sim.backend import SensorReadPhase

        elapsed = _time_ms(device, lambda: orig_sync(phase))
        if phase is SensorReadPhase.POST_PHYSICS:
            buckets["synchronize"] += elapsed

    env.backend.step = timed_backend_step  # type: ignore[method-assign]
    env.backend.synchronize = timed_synchronize  # type: ignore[method-assign]
    env.action_manager.process_action = lambda act: _record("process_action", device, lambda: orig_process(act), buckets)  # type: ignore[method-assign]
    env.action_manager.apply_action = lambda: _record("apply_action", device, orig_apply, buckets)  # type: ignore[method-assign]
    env.termination_manager.compute = lambda: _record("termination", device, orig_term, buckets)  # type: ignore[method-assign]
    env.command_manager.compute = lambda dt: _record("command", device, lambda: orig_command(dt), buckets)  # type: ignore[method-assign]
    env.event_manager.apply = lambda *a, **k: _record("event", device, lambda: orig_event(*a, **k), buckets)  # type: ignore[method-assign]
    env._reset_idx = lambda ids: _record("reset", device, lambda: orig_reset(ids), buckets)  # type: ignore[method-assign]

    orig_reward = env.reward_manager.compute
    orig_obs = env.observation_manager.compute
    env.reward_manager.compute = lambda dt: _record("reward", device, lambda: orig_reward(dt), buckets)  # type: ignore[method-assign]
    env.observation_manager.compute = lambda: _record("obs", device, orig_obs, buckets)  # type: ignore[method-assign]

    mj_scene = getattr(env.backend, "_mj_scene", None)
    mj_sim = getattr(env.backend, "_sim", None)
    orig_write = getattr(mj_scene, "write_data_to_sim", None) if mj_scene is not None else None
    orig_sim_step = getattr(mj_sim, "step", None) if mj_sim is not None else None
    orig_scene_update = getattr(mj_scene, "update", None) if mj_scene is not None else None
    if orig_write is not None:
        mj_scene.write_data_to_sim = lambda: _record("write_data_to_sim", device, orig_write, buckets)
    if orig_sim_step is not None:
        mj_sim.step = lambda: _record("sim_step", device, orig_sim_step, buckets)
    if orig_scene_update is not None:
        mj_scene.update = lambda dt: _record("scene_update", device, lambda: orig_scene_update(dt), buckets)

    prof = None
    if args.aten_ops:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        prof = torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True)
        prof.__enter__()
    try:
        for _ in range(args.steps):
            buckets["policy_step"] += _time_ms(device, lambda: env.step(action))
    finally:
        if prof is not None:
            prof.__exit__(None, None, None)

    env.backend.step = orig_step  # type: ignore[method-assign]
    env.backend.synchronize = orig_sync  # type: ignore[method-assign]
    env.action_manager.process_action = orig_process  # type: ignore[method-assign]
    env.action_manager.apply_action = orig_apply  # type: ignore[method-assign]
    env.termination_manager.compute = orig_term  # type: ignore[method-assign]
    env.command_manager.compute = orig_command  # type: ignore[method-assign]
    env.event_manager.apply = orig_event  # type: ignore[method-assign]
    env._reset_idx = orig_reset  # type: ignore[method-assign]
    env.reward_manager.compute = orig_reward  # type: ignore[method-assign]
    env.observation_manager.compute = orig_obs  # type: ignore[method-assign]
    if orig_write is not None:
        mj_scene.write_data_to_sim = orig_write
    if orig_sim_step is not None:
        mj_sim.step = orig_sim_step
    if orig_scene_update is not None:
        mj_scene.update = orig_scene_update

    policy_ms = buckets["policy_step"]
    sync_ms = buckets["synchronize"]
    bridge_ratio = sync_ms / policy_ms if policy_ms else 0.0
    fps = 1000.0 * args.steps / policy_ms if policy_ms else 0.0
    field_ms = {}
    profile_fields = getattr(env.backend, "profile_field_groups", None)
    if callable(profile_fields):
        samples = [profile_fields() for _ in range(max(1, args.steps))]
        keys = samples[0]
        field_ms = {name: sum(sample[name] for sample in samples) / len(samples) for name in keys}
    report = {
        "backend": args.backend,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "decimation": env.cfg.simulation.decimation,
        "device": str(device),
        "ms": {name: value / args.steps for name, value in buckets.items()},
        "field_ms": field_ms,
        "bridge_ratio": bridge_ratio,
        "fps": fps,
        "ops": _summarize_profiler(prof) if prof is not None else {},
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
