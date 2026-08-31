"""Benchmark the 1.0 lifecycle surface on one native engine.

This entry point selects and bootstraps the engine before importing torch or an
engine SDK. It emits a stable JSON report for construction, throughput, device
memory, full/partial reset, same-engine snapshot costs, and trace replay.

Examples::

    python scripts/benchmark_lifecycle.py --engine mjlab --num_envs 1024
    python scripts/benchmark_lifecycle.py --engine isaacsim --num_envs 4096 --headless
    python scripts/benchmark_lifecycle.py --engine mjlab --thresholds release.json
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


def _parse() -> argparse.Namespace:
    from instinctlab_engine import names

    chooser = argparse.ArgumentParser(add_help=False)
    chooser.add_argument("--engine", required=True, choices=names())
    chosen, _ = chooser.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__, parents=[chooser])
    parser.add_argument("--task", default="Instinct-Velocity-Flat-G1")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--reset_repeats", type=int, default=10)
    parser.add_argument("--snapshot_repeats", type=int, default=3)
    parser.add_argument("--trace_steps", type=int, default=5)
    parser.add_argument("--trace_observation_atol", type=float, default=1.0e-5)
    parser.add_argument("--trace_reward_atol", type=float, default=1.0e-5)
    parser.add_argument("--trace_rtol", type=float, default=1.0e-5)
    parser.add_argument("--partial_reset_fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument(
        "--allow-nonclean-resolution",
        action="store_true",
        help="Allow skipped, emulated, or omitted terms after explicit review.",
    )
    from instinctlab_engine import adapter

    adapter(chosen.engine).add_cli_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num_envs must be positive")
    if min(
        args.warmup_steps,
        args.steps,
        args.reset_repeats,
        args.snapshot_repeats,
        args.trace_steps,
    ) < 1:
        parser.error(
            "warmup, steps, reset repeats, snapshot repeats, and trace steps "
            "must be positive"
        )
    if not 0.0 < args.partial_reset_fraction <= 1.0:
        parser.error("--partial_reset_fraction must be in (0, 1]")
    replay_tolerances = (
        args.trace_observation_atol,
        args.trace_reward_atol,
        args.trace_rtol,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in replay_tolerances):
        parser.error("trace replay tolerances must be finite and non-negative")
    return args


def _measure(operation, repeats: int, synchronize) -> tuple[list[float], object]:
    samples: list[float] = []
    result = None
    for _ in range(repeats):
        synchronize()
        started = perf_counter()
        result = operation()
        synchronize()
        samples.append(perf_counter() - started)
    return samples, result


def _run(args, engine, resources: ExitStack) -> tuple[dict, int]:
    import tempfile

    import torch
    from instinctlab.tasks.registry import asset_id
    from instinctlab.tasks.registry import spec as task_spec
    from instinctlab_engine.lifecycle import (
        EnvironmentSnapshot,
        EpisodeTrace,
        duration_statistics,
        evaluate_thresholds,
        replay_trace,
    )
    from instinctlab_engine.preflight import require_preflight

    device = torch.device(args.device)
    cuda_index = (
        torch.cuda.current_device()
        if device.type == "cuda" and device.index is None
        else device.index
    )

    def synchronize() -> None:
        if cuda_index is not None:
            torch.cuda.synchronize(cuda_index)

    def memory_allocated() -> int:
        if cuda_index is None:
            return 0
        return int(torch.cuda.memory_allocated(cuda_index))

    def peak_memory() -> int:
        if cuda_index is None:
            return 0
        return int(torch.cuda.max_memory_allocated(cuda_index))

    def device_memory_used() -> int:
        if cuda_index is None:
            return 0
        free, total = torch.cuda.mem_get_info(cuda_index)
        return int(total - free)

    if cuda_index is not None:
        # MJWarp may create the primary CUDA context before PyTorch's caching
        # allocator. Peak-stat reset is invalid until PyTorch owns one allocation.
        allocator_probe = torch.empty((), device=device)
        synchronize()
        del allocator_probe

    robot = engine.robot_spec(asset_id(args.task))
    task = task_spec(args.task, robot)
    preflight = require_preflight(
        task,
        args.engine,
        selected_adapter=engine,
        allow_nonclean=args.allow_nonclean_resolution,
    )

    compile_started = perf_counter()
    compiled = engine.compile(
        task,
        num_envs=args.num_envs,
        device=args.device,
        strict=not args.allow_nonclean_resolution,
    )
    compile_s = perf_counter() - compile_started
    compiled.resolution.require_clean(
        allow_nonclean=args.allow_nonclean_resolution
    )
    compiled.env_cfg.seed = args.seed

    baseline_memory = memory_allocated()
    baseline_device_memory = device_memory_used()
    print(f"[INFO] Lifecycle benchmark device={device}, cuda_index={cuda_index}", flush=True)
    if cuda_index is not None:
        torch.cuda.reset_peak_memory_stats(cuda_index)
    synchronize()
    construction_started = perf_counter()
    native_env = compiled.make_env()
    resources.callback(native_env.close)
    synchronize()
    construction_s = perf_counter() - construction_started
    construction_peak_memory = peak_memory()
    construction_allocated_memory = memory_allocated()
    construction_device_memory = device_memory_used()
    print("[INFO] Lifecycle benchmark native environment constructed", flush=True)

    synchronize()
    wrapper_started = perf_counter()
    env = engine.wrap_for_rl(native_env)
    synchronize()
    wrapper_construction_s = perf_counter() - wrapper_started
    print("[INFO] Lifecycle benchmark RL wrapper constructed", flush=True)

    actions = torch.zeros(
        (env.num_envs, env.num_actions),
        device=env.device,
        dtype=torch.float32,
    )
    for _ in range(args.warmup_steps):
        env.step(actions)
    synchronize()
    if cuda_index is not None:
        torch.cuda.reset_peak_memory_stats(cuda_index)
    step_memory_baseline = memory_allocated()
    step_device_memory_baseline = device_memory_used()
    throughput_started = perf_counter()
    for _ in range(args.steps):
        env.step(actions)
    synchronize()
    throughput_s = perf_counter() - throughput_started
    step_peak_memory = peak_memory()
    step_device_memory = device_memory_used()
    print("[INFO] Lifecycle benchmark throughput measured", flush=True)

    full_reset_samples, _ = _measure(env.reset, args.reset_repeats, synchronize)
    partial_count = max(1, round(args.num_envs * args.partial_reset_fraction))
    partial_ids = torch.arange(partial_count, device=env.device, dtype=torch.long)
    partial_reset_samples, _ = _measure(
        lambda: native_env._reset_idx(partial_ids),
        args.reset_repeats,
        synchronize,
    )
    print("[INFO] Lifecycle benchmark resets measured", flush=True)

    lifecycle = env.lifecycle
    snapshot_capture_samples, snapshot_value = _measure(
        lifecycle.snapshot,
        args.snapshot_repeats,
        synchronize,
    )
    snapshot = snapshot_value
    if snapshot is None:
        raise RuntimeError("Lifecycle snapshot benchmark produced no snapshot.")
    snapshot_restore_samples, _ = _measure(
        lambda: lifecycle.restore(snapshot),
        args.snapshot_repeats,
        synchronize,
    )
    with tempfile.TemporaryDirectory(prefix="instinctlab-lifecycle-benchmark-") as directory:
        snapshot_path = Path(directory) / "state.snapshot.npz"
        snapshot_save_samples, _ = _measure(
            lambda: snapshot.save(snapshot_path),
            args.snapshot_repeats,
            synchronize,
        )
        snapshot_bytes = snapshot_path.stat().st_size
        snapshot_load_samples, _ = _measure(
            lambda: EnvironmentSnapshot.load(snapshot_path),
            args.snapshot_repeats,
            synchronize,
        )
        print("[INFO] Lifecycle benchmark snapshot round trip measured", flush=True)

        env.reset()
        synchronize()
        print("[INFO] Lifecycle benchmark trace reset complete", flush=True)
        trace_started = perf_counter()
        lifecycle.start_trace(
            torch.tensor([0], device=env.device, dtype=torch.long)
        )
        for _ in range(args.trace_steps):
            env.step(actions)
        trace = lifecycle.stop_trace(require_complete=False)
        synchronize()
        trace_record_s = perf_counter() - trace_started
        print("[INFO] Lifecycle benchmark trace recorded", flush=True)

        trace_path = Path(directory) / "episode.trace.npz"
        trace.save(trace_path)
        loaded_trace = EpisodeTrace.load(trace_path)
        trace_bytes = trace_path.stat().st_size
        print("[INFO] Lifecycle benchmark trace archive loaded", flush=True)
        synchronize()
        replay_started = perf_counter()
        try:
            replay = replay_trace(
                env,
                loaded_trace,
                strict=False,
                field_tolerances={
                    "observation": (
                        args.trace_observation_atol,
                        args.trace_rtol,
                    ),
                    "reward": (args.trace_reward_atol, args.trace_rtol),
                },
            )
        except Exception:
            import traceback

            traceback.print_exc()
            raise
        synchronize()
        trace_replay_s = perf_counter() - replay_started
        print("[INFO] Lifecycle benchmark trace replayed", flush=True)

    full_reset = duration_statistics(full_reset_samples)
    partial_reset = duration_statistics(partial_reset_samples)
    snapshot_capture = duration_statistics(snapshot_capture_samples)
    snapshot_restore = duration_statistics(snapshot_restore_samples)
    snapshot_save = duration_statistics(snapshot_save_samples)
    snapshot_load = duration_statistics(snapshot_load_samples)
    policy_steps_per_s = args.steps / throughput_s
    metrics = {
        "compile_ms": compile_s * 1_000.0,
        "environment_construction_ms": construction_s * 1_000.0,
        "rl_wrapper_construction_ms": wrapper_construction_s * 1_000.0,
        "throughput_policy_steps_per_s": policy_steps_per_s,
        "throughput_env_steps_per_s": policy_steps_per_s * args.num_envs,
        "throughput_physics_steps_per_s": (
            policy_steps_per_s * args.num_envs * task.sim.decimation
        ),
        "torch_peak_allocated_bytes": max(construction_peak_memory, step_peak_memory),
        "torch_construction_peak_increment_bytes": max(
            construction_peak_memory - baseline_memory,
            0,
        ),
        "torch_construction_allocated_increment_bytes": max(
            construction_allocated_memory - baseline_memory,
            0,
        ),
        "torch_step_peak_increment_bytes": max(
            step_peak_memory - step_memory_baseline,
            0,
        ),
        "device_resident_bytes_after_construction": construction_device_memory,
        "device_resident_bytes_after_steps": step_device_memory,
        "construction_device_resident_increment_bytes": max(
            construction_device_memory - baseline_device_memory,
            0,
        ),
        "step_device_resident_increment_bytes": max(
            step_device_memory - step_device_memory_baseline,
            0,
        ),
        "full_reset_median_ms": full_reset["median_ms"],
        "partial_reset_median_ms": partial_reset["median_ms"],
        "snapshot_capture_median_ms": snapshot_capture["median_ms"],
        "snapshot_restore_median_ms": snapshot_restore["median_ms"],
        "snapshot_save_median_ms": snapshot_save["median_ms"],
        "snapshot_load_median_ms": snapshot_load["median_ms"],
        "snapshot_bytes": snapshot_bytes,
        "trace_record_ms": trace_record_s * 1_000.0,
        "trace_record_policy_step_ms": trace_record_s * 1_000.0 / args.trace_steps,
        "trace_replay_ms": trace_replay_s * 1_000.0,
        "trace_replay_policy_step_ms": trace_replay_s * 1_000.0 / args.trace_steps,
        "trace_bytes": trace_bytes,
    }
    report = {
        "schema_version": "lifecycle_benchmark_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": args.engine,
        "task_id": task.task_id,
        "device": str(args.device),
        "num_envs": args.num_envs,
        "configuration": {
            "seed": args.seed,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.steps,
            "reset_repeats": args.reset_repeats,
            "snapshot_repeats": args.snapshot_repeats,
            "trace_steps": args.trace_steps,
            "trace_observation_atol": args.trace_observation_atol,
            "trace_reward_atol": args.trace_reward_atol,
            "trace_rtol": args.trace_rtol,
            "partial_reset_envs": partial_count,
            "physics_dt": task.sim.physics_dt,
            "decimation": task.sim.decimation,
        },
        "metrics": metrics,
        "series": {
            "full_reset": full_reset,
            "partial_reset": partial_reset,
            "snapshot_capture": snapshot_capture,
            "snapshot_restore": snapshot_restore,
            "snapshot_save": snapshot_save,
            "snapshot_load": snapshot_load,
        },
        "trace": {
            "selected_envs": trace.env_ids.tolist(),
            "steps": len(trace.steps),
            "complete_episode": trace.complete,
            "archive_round_trip": True,
            "replay_matched": replay.matched,
            "replay_differences": [
                {
                    "step": difference.step,
                    "field": difference.field,
                    "max_absolute_error": difference.max_absolute_error,
                    "max_index": difference.max_index,
                    "actual_at_max": difference.actual_at_max,
                    "expected_at_max": difference.expected_at_max,
                }
                for difference in replay.differences
            ],
        },
        "lifecycle": lifecycle.manifest,
        "preflight": preflight,
        "resolution": compiled.resolution.manifest(),
        "thresholds": None,
        "threshold_failures": [],
        "status": "ok" if replay.matched else "replay_failed",
    }
    exit_code = 0 if replay.matched else 3
    if args.thresholds is not None:
        thresholds = json.loads(args.thresholds.read_text())
        failures = evaluate_thresholds(report, thresholds)
        report["thresholds"] = str(args.thresholds.resolve())
        report["threshold_failures"] = list(failures)
        if failures:
            report["status"] = "threshold_failed"
            exit_code = 2
    return report, exit_code


def _output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output.expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (
        Path("logs")
        / "benchmarks"
        / f"{args.engine}_{args.task}_{args.num_envs}_{stamp}.json"
    ).resolve()


def main() -> int:
    args = _parse()
    from instinctlab_engine import adapter

    engine = adapter(args.engine)
    with ExitStack() as resources:
        app = engine.bootstrap(args)
        if app is not None:
            resources.callback(app.close)
        report, exit_code = _run(args, engine, resources)
        output = _output_path(args)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        print(json.dumps({"status": report["status"], "output": str(output)}))
        for failure in report["threshold_failures"]:
            print(f"[FAIL] {failure}")
        if exit_code:
            # Isaac's graceful application shutdown may terminate the process
            # with code zero before the adapter finalizer is reached. Preserve
            # release-gate failures after the report has been flushed to disk.
            return engine.finalize_process(exit_code)
    return engine.finalize_process(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
