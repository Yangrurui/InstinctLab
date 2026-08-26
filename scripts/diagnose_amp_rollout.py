"""Measure AMP inputs and discriminator scores during a controlled rollout.

Run the same checkpoint once per engine, then compare the two JSON files.  The
probe deliberately uses the task's normal compiler and RL wrapper: observation
history, joint selection, reset handling, and discriminator packing are the
ones used by training rather than a parallel reimplementation.

The default uses the deterministic policy mean.  ``stochastic_sample`` adds
the checkpoint's learned action standard deviation with an isolated generator,
so action sampling cannot perturb environment, reset, or domain-randomization
RNG streams.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path

if sys.path and os.path.isdir(os.path.join(sys.path[0], "instinct_rl")):
    sys.path.pop(0)


class Moments:
    """Streaming scalar moments for tensors without retaining rollout data."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.square_total = 0.0
        self.absolute_total = 0.0
        self.maximum = -math.inf

    def add(self, value) -> None:
        import torch

        data = value.detach().float()
        self.count += data.numel()
        self.total += float(data.sum())
        self.square_total += float(torch.square(data).sum())
        self.absolute_total += float(data.abs().sum())
        self.maximum = max(self.maximum, float(data.max()))

    def summary(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0}
        mean = self.total / self.count
        mean_square = self.square_total / self.count
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(max(mean_square - mean * mean, 0.0)),
            "rms": math.sqrt(mean_square),
            "abs_mean": self.absolute_total / self.count,
            "max": self.maximum,
        }


def parse_args() -> argparse.Namespace:
    from instinctlab.engines import adapter, names

    chooser = argparse.ArgumentParser(add_help=False)
    chooser.add_argument("--engine", required=True, choices=names())
    selected, _ = chooser.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__, parents=[chooser])
    parser.add_argument("--task", default="Instinct-Parkour-Target-G1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy-actions",
        choices=("deterministic_mean", "stochastic_sample"),
        default="deterministic_mean",
    )
    parser.add_argument("--action-seed", type=int, default=12345)
    parser.add_argument("--strict", action="store_true")
    adapter(selected.engine).add_cli_args(parser)
    return parser.parse_args()


def silence_observation_noise(env_cfg: object) -> None:
    groups = env_cfg.observations
    for group in groups.values() if isinstance(groups, dict) else vars(groups).values():
        if hasattr(group, "enable_corruption"):
            group.enable_corruption = False


def term_slices(obs_format: dict[str, tuple[int, ...]]) -> dict[str, slice]:
    offset = 0
    slices: dict[str, slice] = {}
    for name, shape in obs_format.items():
        width = math.prod(shape)
        slices[name] = slice(offset, offset + width)
        offset += width
    return slices


def make_term_stats(names) -> dict[str, dict[str, Moments]]:
    return {
        name: {
            "actor": Moments(),
            "reference": Moments(),
            "paired_error": Moments(),
            "actor_history_delta": Moments(),
            "reference_history_delta": Moments(),
        }
        for name in names
    }


def history_delta(flat, history_length: int = 10):
    if flat.shape[1] % history_length != 0:
        raise ValueError(f"AMP term width {flat.shape[1]} is not divisible by history length {history_length}.")
    history = flat.reshape(flat.shape[0], history_length, -1)
    return history[:, 1:] - history[:, :-1]


def record_terms(stats, slices, actor, reference) -> None:
    for name, segment in slices.items():
        actor_term = actor[:, segment]
        reference_term = reference[:, segment]
        stats[name]["actor"].add(actor_term)
        stats[name]["reference"].add(reference_term)
        stats[name]["paired_error"].add(actor_term - reference_term)
        stats[name]["actor_history_delta"].add(history_delta(actor_term))
        stats[name]["reference_history_delta"].add(history_delta(reference_term))


def policy_action(actor_critic, obs, mode: str, generator):
    """Return a mean or reproducibly sampled action without touching global RNG."""
    import torch

    mean = actor_critic.act_inference(obs)
    if mode == "deterministic_mean":
        return mean
    if mode != "stochastic_sample":
        raise ValueError(f"unknown policy action mode: {mode!r}")
    noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
    return mean + noise * actor_critic.std


def run(args: argparse.Namespace, engine, resources: ExitStack, output_path: Path) -> dict:
    import torch

    from instinct_rl.runners import OnPolicyRunner

    from instinctlab.checkpoint import validate_checkpoint_contract
    from instinctlab.tasks.registry import spec as task_spec

    checkpoint = args.checkpoint.expanduser().resolve()
    spec = task_spec(args.task)
    validate_checkpoint_contract(checkpoint, spec)

    compiled = engine.compile(spec, num_envs=args.num_envs, device=args.device, strict=args.strict)
    compiled.env_cfg.seed = args.seed
    silence_observation_noise(compiled.env_cfg)
    native_env = compiled.make_env()
    resources.callback(native_env.close)
    env = engine.wrap_for_rl(native_env)
    print("[INFO] AMP probe environment is ready.", flush=True)

    agent_cfg = compiled.agent_cfg
    agent_cfg.device = args.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    print("[INFO] AMP probe runner is ready; loading checkpoint.", flush=True)
    runner.load(str(checkpoint))
    runner.eval_mode()
    action_generator = torch.Generator(device=torch.device(args.device))
    action_generator.manual_seed(args.action_seed)
    print("[INFO] AMP probe checkpoint is loaded.", flush=True)

    obs_format = env.get_obs_format()
    actor_format = obs_format[agent_cfg.algorithm.actor_state_key]
    reference_format = obs_format[agent_cfg.algorithm.reference_state_key]
    if actor_format != reference_format:
        raise RuntimeError(f"AMP actor/reference formats differ: {actor_format} != {reference_format}")
    slices = term_slices(actor_format)
    terms = make_term_stats(slices)
    discriminator_actor = Moments()
    discriminator_reference = Moments()
    discriminator_reward = Moments()
    environment_reward = Moments()
    action_moments = Moments()
    done_count = 0
    sampled_env_steps = 0

    obs, extras = env.get_observations()
    for step in range(args.steps):
        observations = extras["observations"]
        actor = observations[agent_cfg.algorithm.actor_state_key]
        reference = observations[agent_cfg.algorithm.reference_state_key]
        with torch.inference_mode():
            action = policy_action(runner.alg.actor_critic, obs, args.policy_actions, action_generator)
            actor_logit = runner.alg.discriminator(actor)
            reference_logit = runner.alg.discriminator(reference)
            style_reward = torch.clamp(1.0 - 0.25 * torch.square(actor_logit - 1.0), min=0.0)

        if step >= args.warmup_steps:
            record_terms(terms, slices, actor, reference)
            discriminator_actor.add(actor_logit)
            discriminator_reference.add(reference_logit)
            discriminator_reward.add(style_reward)
            action_moments.add(action)
            sampled_env_steps += args.num_envs

        obs, environment_step_reward, dones, extras = env.step(action.detach())
        if step >= args.warmup_steps:
            environment_reward.add(environment_step_reward)
            done_count += int(dones.sum())
    print("[INFO] AMP probe rollout is complete.", flush=True)

    print("[INFO] Summarizing AMP probe statistics.", flush=True)
    result = {
        "metadata": {
            "engine": args.engine,
            "task": args.task,
            "checkpoint": str(checkpoint),
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "observation_noise": False,
            "policy_actions": args.policy_actions,
            "action_seed": args.action_seed,
            "learned_action_std": runner.alg.actor_critic.std.detach().float().cpu().tolist(),
            "amp_format": {name: [int(size) for size in shape] for name, shape in actor_format.items()},
        },
        "discriminator": {
            "actor": discriminator_actor.summary(),
            "reference": discriminator_reference.summary(),
            "reward_raw": discriminator_reward.summary(),
            "reward_after_coef_mean": discriminator_reward.summary().get("mean", 0.0) * float(
                agent_cfg.algorithm.discriminator_reward_coef
            ),
        },
        "environment_reward": environment_reward.summary(),
        "action": action_moments.summary(),
        "done_fraction": done_count / max(sampled_env_steps, 1),
        "terms": {
            name: {metric: moments.summary() for metric, moments in metrics.items()} for name, metrics in terms.items()
        },
    }
    print("[INFO] AMP probe statistics are summarized.", flush=True)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    except BaseException as error:
        print(f"[ERROR] Could not write AMP probe to {output_path!r}: {error!r}", flush=True)
        raise
    print(f"[INFO] Wrote AMP rollout diagnosis to {output_path}", flush=True)
    return result


def main() -> None:
    args = parse_args()
    output_path = args.out
    if args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        raise ValueError("warmup-steps must be non-negative and smaller than steps")

    from instinctlab.engines import adapter

    engine = adapter(args.engine)
    with ExitStack() as resources:
        app = engine.bootstrap(args)
        if app is not None:
            resources.callback(app.close)
        run(args, engine, resources, output_path)
        print("[INFO] AMP probe summary is ready.", flush=True)


if __name__ == "__main__":
    main()
