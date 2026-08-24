"""Measure parkour contact-criterion trigger rates on one engine.

Reports the duration / ``found`` gate (portable ``in_contact``) against a 1 N
threshold on that engine's own force history. Those are the before/after
criteria for ``base_contact`` and ``undesired_contacts``. Same seed, zero
actions, modest ``num_envs`` so the two engine runs are comparable.

    python scripts/measure_parkour_contact_threshold.py --engine mjlab --device cuda:2
    python scripts/measure_parkour_contact_threshold.py --engine isaacsim --device cuda:1

Measured on mjlab (16 envs, 80 steps, seed 42, zero actions): the compiled
``undesired_contacts`` fires at 0.109 mean count/env, matching the 1 N gate
exactly and not the 0.130 of the duration gate -- the threshold is wired, not
merely declared. The criteria disagree on 0.78% of (env, element) slots; of
the slots the duration gate calls contact, 16% carry less than 1 N (median
12.7 N). So the change is real but small. ``base_contact`` is *not* covered
by this probe: with zero actions the robot does not fall, torso samples come
out ``n=0``, and the termination path stays unmeasured.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import torch
from pathlib import Path


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=("isaacsim", "mjlab"))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_known_args()[0]


def _force_gate(history: torch.Tensor, threshold: float) -> torch.Tensor:
    """``(env, element)``: max-over-history ‖F‖ exceeds ``threshold``."""
    return history.norm(dim=-1).max(dim=1).values > threshold


def _summarize(flag: torch.Tensor) -> dict[str, float]:
    total = int(flag.numel())
    hits = int(flag.sum().item())
    return {"hits": hits, "total": total, "rate": hits / total if total else 0.0}


def main() -> int:
    args = _parse()
    if args.engine == "isaacsim":
        from isaaclab.app import AppLauncher

        AppLauncher({"headless": True, "enable_cameras": False, "device": args.device})

    from instinctlab.compat import sensors as compat_sensors
    from instinctlab.tasks.parkour.config.g1 import parkour_target_g1
    from instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import TORSO_CONTACT, UNDESIRED_CONTACT

    if args.engine == "isaacsim":
        from instinctlab.engines.isaacsim import IsaacSimAdapter as Adapter
    else:
        from instinctlab.engines.mjlab import MjlabAdapter as Adapter

    spec = parkour_target_g1()
    compiled = Adapter().compile(spec, num_envs=args.num_envs, device=args.device)
    compiled.env_cfg.seed = args.seed
    env = compiled.make_env()
    env.reset()

    sensor = env.scene.sensors["contact_forces"]
    actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    threshold = 1.0
    done_cfg = env.termination_manager._term_cfgs[env.termination_manager._term_names.index("base_contact")]
    rew_cfg = env.reward_manager._term_cfgs[env.reward_manager._term_names.index("undesired_contacts")]

    base_duration = []
    base_force = []
    und_duration_count = []
    und_force_count = []
    und_duration_any = []
    und_force_any = []
    force_when_duration = []
    torso_force_when_duration = []
    compiled_base = []
    compiled_und = []

    def _record() -> None:
        torso_touch = compat_sensors.in_contact(sensor, TORSO_CONTACT)
        und_touch = compat_sensors.in_contact(sensor, UNDESIRED_CONTACT)
        torso_hist = compat_sensors.contact_force_history(sensor, TORSO_CONTACT)
        und_hist = compat_sensors.contact_force_history(sensor, UNDESIRED_CONTACT)
        torso_force = _force_gate(torso_hist, threshold)
        und_force = _force_gate(und_hist, threshold)
        base_duration.append(torso_touch.any(dim=1))
        base_force.append(torso_force.any(dim=1))
        und_duration_count.append(und_touch.sum(dim=1).float())
        und_force_count.append(und_force.sum(dim=1).float())
        und_duration_any.append(und_touch.any(dim=1))
        und_force_any.append(und_force.any(dim=1))
        force_when_duration.append(und_hist.norm(dim=-1).max(dim=1).values[und_touch])
        torso_force_when_duration.append(torso_hist.norm(dim=-1).max(dim=1).values[torso_touch])
        compiled_base.append(done_cfg.func(env, **done_cfg.params).bool())
        compiled_und.append(rew_cfg.func(env, **rew_cfg.params).float())

    for _ in range(args.steps):
        env.step(actions)
        _record()

    # Fallen drop: env.step would reset (root_height / bad_orientation). Write a
    # pitched pose and step *physics only* so both criteria see the same contacts.
    from instinctlab.compat.math import quat_from_euler_xyz

    def _physics_step() -> None:
        env.scene.write_data_to_sim()
        sim = env.sim
        stepped = getattr(sim, "step", None)
        if callable(stepped):
            try:
                stepped(render=False)
            except TypeError:
                stepped()
        elif hasattr(sim, "forward"):
            sim.forward()
        if hasattr(sim, "sense"):
            sim.sense()
        env.scene.update(float(env.physics_dt))

    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs, device=env.device)
    fallen_steps = 40
    pos = env.scene.env_origins + torch.tensor((0.0, 0.0, 0.55), device=env.device)
    quat = quat_from_euler_xyz(
        torch.zeros(env.num_envs, device=env.device),
        torch.full((env.num_envs,), 1.55, device=env.device),
        torch.zeros(env.num_envs, device=env.device),
    )
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device), env_ids=env_ids)
    q = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    for _ in range(fallen_steps):
        _physics_step()
        _record()

    env.close()

    def stack_bool(rows):
        return torch.stack(rows, dim=0)

    def force_stats(rows):
        light = torch.cat(rows) if any(t.numel() for t in rows) else torch.zeros(0)
        if light.numel() == 0:
            return {"n": 0, "frac_below_1n": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
        return {
            "n": int(light.numel()),
            "frac_below_1n": float((light < threshold).float().mean()),
            "median": float(light.median()),
            "p10": float(torch.quantile(light, 0.1)),
            "p90": float(torch.quantile(light, 0.9)),
        }

    def pack(sl):
        base_d = stack_bool(base_duration)[sl]
        base_f = stack_bool(base_force)[sl]
        und_d_any = stack_bool(und_duration_any)[sl]
        und_f_any = stack_bool(und_force_any)[sl]
        und_d_cnt = torch.stack(und_duration_count)[sl]
        und_f_cnt = torch.stack(und_force_count)[sl]
        compiled_base_t = stack_bool(compiled_base)[sl]
        compiled_und_t = torch.stack(compiled_und)[sl]
        return {
            "base_contact": {
                "duration_rate": _summarize(base_d),
                "force_1n_rate": _summarize(base_f),
                "compiled_rate": _summarize(compiled_base_t),
                "duration_and_not_force": _summarize(base_d & ~base_f),
                "force_and_not_duration": _summarize(base_f & ~base_d),
            },
            "undesired_contacts": {
                "duration_any_rate": _summarize(und_d_any),
                "force_1n_any_rate": _summarize(und_f_any),
                "duration_mean_count": float(und_d_cnt.mean()),
                "force_1n_mean_count": float(und_f_cnt.mean()),
                "compiled_mean_count": float(compiled_und_t.mean()),
                "duration_and_not_force": _summarize(und_d_any & ~und_f_any),
            },
        }

    standing = slice(0, args.steps)
    fallen = slice(args.steps, None)
    report = {
        "engine": args.engine,
        "device": args.device,
        "num_envs": args.num_envs,
        "standing_steps": args.steps,
        "fallen_steps": fallen_steps,
        "seed": args.seed,
        "threshold_n": threshold,
        "standing": pack(standing),
        "fallen": pack(fallen),
        "all": pack(slice(None)),
        "undesired_force_when_duration_n": force_stats(force_when_duration),
        "torso_force_when_duration_n": force_stats(torso_force_when_duration),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
