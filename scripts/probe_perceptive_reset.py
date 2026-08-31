"""Evaluate one policy on a fixed Perceptive motion bin and reset-height recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path


def _parse() -> argparse.Namespace:
    from instinctlab_engine import adapter, names

    chooser = argparse.ArgumentParser(add_help=False)
    chooser.add_argument("--engine", required=True, choices=names())
    selected, _ = chooser.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__, parents=[chooser])
    parser.add_argument("--task", default="Instinct-Perceptive-Shadowing-G1-v0")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--global-bin", type=int, default=12)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensure-link-below-zero-ground", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--motion-start-height-offset", type=float, default=0.0)
    parser.add_argument("--disable-illegal-reset-contact", action="store_true")
    parser.add_argument("--isaac-self-collision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--out", type=Path, required=True)
    adapter(selected.engine).add_cli_args(parser)
    return parser.parse_args()


def _task_with_reset_semantics(task, *, ensure_link_below_zero_ground: bool, height_offset: float):
    """Replace the sole motion reference without changing the registered task."""
    if len(task.scene.motion_references) != 1:
        raise ValueError(f"expected one motion reference, got {len(task.scene.motion_references)}")
    old_motion = task.scene.motion_references[0]
    motion = replace(
        old_motion,
        ensure_link_below_zero_ground=ensure_link_below_zero_ground,
        motion_start_height_offset=height_offset,
        engine_overrides={},
    )

    def patch_term(term):
        return replace(
            term,
            params={key: motion if value is old_motion else value for key, value in term.params.items()},
        )

    observations = {
        group_name: replace(
            group,
            terms={term_name: patch_term(term) for term_name, term in group.terms.items()},
        )
        for group_name, group in task.mdp.observations.items()
    }
    rewards = {
        group_name: {
            term_name: patch_term(term) for term_name, term in group.items()
        }
        for group_name, group in task.mdp.rewards.items()
    }
    actions = {name: patch_term(term) for name, term in task.mdp.actions.items()}
    terminations = {name: patch_term(term) for name, term in task.mdp.terminations.items()}
    events = {name: patch_term(term) for name, term in task.mdp.events.items()}
    commands = {name: patch_term(term) for name, term in task.mdp.commands.items()}
    curriculum = {
        name: patch_term(term) for name, term in task.mdp.curriculum.items()
    }
    return replace(
        task,
        scene=replace(task.scene, motion_references=(motion,)),
        mdp=replace(
            task.mdp,
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            events=events,
            commands=commands,
            curriculum=curriculum,
        ),
    )


def _without_adaptive_sampling(task):
    """Keep fixed-bin weights intact across every diagnostic reset."""
    events = {
        name: term
        for name, term in task.mdp.events.items()
        if name != "bin_fail_counter_smoothing"
    }
    return replace(task, mdp=replace(task.mdp, events=events, curriculum={}))


def _without_illegal_reset_contact(task):
    terminations = dict(task.mdp.terminations)
    terminations.pop("illegal_reset_contact")
    return replace(task, mdp=replace(task.mdp, terminations=terminations))


def _with_isaac_self_collision(task, enabled: bool):
    profiles = dict(task.sim.profiles)
    profiles["isaacsim"] = {**profiles.get("isaacsim", {}), "self_collision": enabled}
    return replace(task, sim=replace(task.sim, profiles=profiles))


def _force_global_bin(runtime, global_bin: int) -> None:
    if not 0 <= global_bin < runtime.motion_bin_weights.numel():
        raise ValueError(
            f"global bin {global_bin} is outside [0, {runtime.motion_bin_weights.numel()})"
        )
    runtime.motion_bin_weights.zero_()
    runtime.motion_bin_weights[global_bin] = 1.0


def _global_bins(runtime):
    import torch

    local_bins = torch.floor(runtime.buffers.start_s / runtime.ref.motion_bin_length_s).to(torch.long)
    return runtime._bin_offsets[runtime.buffers.motion_id] + local_bins


def _sha256_tensor(value) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def _force_distribution(per_env_force) -> dict[str, float]:
    import torch

    quantiles = torch.tensor((0.0, 0.5, 0.9, 0.95, 0.99, 1.0), device=per_env_force.device)
    values = torch.quantile(per_env_force.float(), quantiles).detach().cpu().tolist()
    names = ("min", "median", "p90", "p95", "p99", "max")
    report = dict(zip(names, values, strict=True))
    report["fraction_over_500"] = float(per_env_force.gt(500.0).float().mean().cpu())
    return report


def _root_position(robot):
    for name in ("root_link_pos_w", "root_pos_w"):
        if hasattr(robot.data, name):
            return getattr(robot.data, name)
    raise AttributeError("robot data exposes neither root_link_pos_w nor root_pos_w")


def _load_runner_checkpoint(runner, checkpoint: Path, device: str) -> None:
    """Load onto the probe device even when the checkpoint records another CUDA index."""
    import torch

    if runner.cfg.get("ckpt_manipulator", False):
        raise ValueError("the reset probe does not support manipulated checkpoints")
    loaded = torch.load(checkpoint, map_location=device, weights_only=True)
    runner.alg.load_state_dict(loaded)
    for group_name, normalizer in runner.normalizers.items():
        key = f"{group_name}_normalizer_state_dict"
        if key not in loaded:
            raise KeyError(f"checkpoint has no {key!r}")
        normalizer.load_state_dict(loaded[key])
    runner.current_learning_iteration = loaded["iter"]


def _registered_task(task_id: str, engine):
    """Build a task with the selected adapter's engine-neutral robot spec."""
    from instinctlab.tasks import registry

    robot = engine.robot_spec(registry.asset_id(task_id))
    return registry.spec(task_id, robot)


def _run(args: argparse.Namespace, engine, resources: ExitStack) -> None:
    import torch
    from instinct_rl.runners import OnPolicyRunner
    from instinctlab_engine.bridge.sensors import (
        contact_force_history,
        element_ids,
        element_names,
    )
    registered = _registered_task(args.task, engine)
    task = _without_adaptive_sampling(
        _task_with_reset_semantics(
            registered,
            ensure_link_below_zero_ground=args.ensure_link_below_zero_ground,
            height_offset=args.motion_start_height_offset,
        )
    )
    if args.disable_illegal_reset_contact:
        task = _without_illegal_reset_contact(task)
    if args.isaac_self_collision is not None:
        task = _with_isaac_self_collision(task, args.isaac_self_collision)
    compiled = engine.compile(task, num_envs=args.num_envs, device=args.device, strict=True)
    compiled.env_cfg.seed = args.seed
    agent_cfg = compiled.agent_cfg
    agent_cfg.device = args.device
    if args.engine == "isaacsim" and args.isaac_self_collision is not None:
        actual = compiled.env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions
        print(f"[PROBE] Isaac articulation self-collision: {actual}", flush=True)
        if actual is not args.isaac_self_collision:
            raise RuntimeError(
                f"requested Isaac self-collision={args.isaac_self_collision}, compiled {actual}"
            )

    native_env = compiled.make_env()
    resources.callback(native_env.close)
    runtime = native_env.scene["motion_reference"]._runtime
    _force_global_bin(runtime, args.global_bin)
    print("[PROBE] Native environment ready and fixed-bin weights installed.", flush=True)

    env = engine.wrap_for_rl(native_env)
    observed_bins = _global_bins(runtime)
    print(
        f"[PROBE] Wrapped reset bins: {torch.unique(observed_bins).detach().cpu().tolist()}",
        flush=True,
    )
    if not torch.all(observed_bins == args.global_bin):
        raise RuntimeError(f"initial reset escaped fixed global bin {args.global_bin}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    _load_runner_checkpoint(runner, args.checkpoint, args.device)
    print(f"[PROBE] Loaded checkpoint {args.checkpoint}.", flush=True)
    policy = runner.get_inference_policy(device=args.device)
    observations, _ = env.get_observations()

    robot = native_env.scene["robot"]
    reference = native_env.scene["motion_reference"].reference_frame
    initial_root_z_error = _root_position(robot)[:, 2] - reference.base_pos_w[:, 0, 2]
    initial_start_s = runtime.buffers.start_s.clone()
    contact_ref = registered.mdp.terminations["illegal_reset_contact"].params["sensor"]
    contact_sensor = native_env.scene.sensors[contact_ref.name]
    selected_body_names = [
        element_names(contact_sensor)[index] for index in element_ids(contact_sensor, contact_ref)
    ]

    def contact_distribution():
        history = contact_force_history(contact_sensor, contact_ref)
        force = torch.linalg.vector_norm(history, dim=-1)
        report = _force_distribution(force.amax(dim=(1, 2)))
        per_body = force.amax(dim=1)
        body_reports = []
        for index, name in enumerate(selected_body_names):
            body_force = per_body[:, index]
            body_reports.append(
                {
                    "body": name,
                    "fraction_over_500": float(body_force.gt(500.0).float().mean().cpu()),
                    "median": float(body_force.median().cpu()),
                    "p95": float(torch.quantile(body_force.float(), 0.95).cpu()),
                    "max": float(body_force.max().cpu()),
                }
            )
        report["top_bodies_over_500"] = sorted(
            body_reports,
            key=lambda item: (item["fraction_over_500"], item["p95"]),
            reverse=True,
        )[:8]
        return report

    early_contact_force = {"after_reset": contact_distribution()}

    term_names = tuple(native_env.termination_manager.active_terms)
    term_counts = torch.zeros(len(term_names), device=args.device, dtype=torch.long)
    first_term_counts = torch.zeros_like(term_counts)
    episode_count = torch.zeros((), device=args.device, dtype=torch.long)
    episode_step_sum = torch.zeros((), device=args.device, dtype=torch.long)
    age = torch.zeros(args.num_envs, device=args.device, dtype=torch.long)
    first_done_step = torch.full_like(age, -1)
    reward_sum = torch.zeros((), device=args.device)
    reward_samples = 0
    survival_steps = (10, 25, 50, 100, 150, 200, 300, 400)
    survival = {}

    with torch.inference_mode():
        for step in range(1, args.steps + 1):
            _force_global_bin(runtime, args.global_bin)
            actions = policy(observations)
            observations, rewards, dones, _ = env.step(actions)
            if step <= 2:
                early_contact_force[f"after_step_{step}"] = contact_distribution()
            done_mask = dones.to(torch.bool)
            age += 1
            reward_sum += rewards.float().sum()
            reward_samples += rewards.numel()

            first_now = done_mask & (first_done_step < 0)
            first_done_step[first_now] = step
            for index, name in enumerate(term_names):
                term_mask = native_env.termination_manager.get_term(name).to(torch.bool) & done_mask
                term_counts[index] += term_mask.sum()
                first_term_counts[index] += (term_mask & first_now).sum()

            episode_count += done_mask.sum()
            episode_step_sum += age[done_mask].sum()
            age[done_mask] = 0
            if step in survival_steps:
                survival[str(step)] = float((first_done_step < 0).float().mean().cpu())

    never_done = first_done_step < 0
    first_lengths = torch.where(never_done, torch.full_like(first_done_step, args.steps), first_done_step)
    report = {
        "engine": args.engine,
        "task": args.task,
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "global_bin": args.global_bin,
        "ensure_link_below_zero_ground": args.ensure_link_below_zero_ground,
        "motion_start_height_offset": args.motion_start_height_offset,
        "illegal_reset_contact_enabled": not args.disable_illegal_reset_contact,
        "isaac_self_collision_override": args.isaac_self_collision,
        "initial_start_s_sha256": _sha256_tensor(initial_start_s),
        "initial_start_s_mean": float(initial_start_s.mean().cpu()),
        "initial_root_z_error_mean_m": float(initial_root_z_error.mean().cpu()),
        "initial_root_z_error_std_m": float(initial_root_z_error.std().cpu()),
        "early_non_support_contact_force_N": early_contact_force,
        "mean_reward_per_step": float((reward_sum / reward_samples).cpu()),
        "completed_episodes": int(episode_count.cpu()),
        "mean_completed_episode_steps": (
            float((episode_step_sum.float() / episode_count.clamp_min(1)).cpu())
        ),
        "mean_first_episode_steps_censored": float(first_lengths.float().mean().cpu()),
        "first_episode_never_done_fraction": float(never_done.float().mean().cpu()),
        "first_episode_survival_fraction": survival,
        "all_termination_counts": {
            name: int(value) for name, value in zip(term_names, term_counts.cpu().tolist(), strict=True)
        },
        "first_termination_counts": {
            name: int(value)
            for name, value in zip(term_names, first_term_counts.cpu().tolist(), strict=True)
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def main() -> None:
    args = _parse()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    from instinctlab_engine import adapter

    engine = adapter(args.engine)
    with ExitStack() as resources:
        app = engine.bootstrap(args)
        if app is not None:
            resources.callback(app.close)
        try:
            _run(args, engine, resources)
        except BaseException:
            error = traceback.format_exc()
            print(error, file=sys.stderr, flush=True)
            args.out.with_suffix(".error.txt").write_text(error)
            raise
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
