"""How much does reading link velocity instead of COM velocity move the reward?

``track_lin_vel_xy_exp``, ``dont_wait`` and the command metrics read the base's
planar velocity. main reads Isaac Lab's ``root_lin_vel_b``, which is a COM alias;
we and InstinctMJ read ``root_link_lin_vel_b``. They differ by ``ω × r`` with
``r`` the root body's COM offset, so the planar gap is ``‖r‖ · ‖(ω_x, ω_y)‖`` --
zero when the base is level, and not zero on parkour terrain.

Comparing the two training runs cannot separate this from policy quality,
because each run *measures* its own frame: a lower ``error_vel_xy`` on one side
may mean it tracked better or merely that it scored itself on a different
quantity. This probe takes both frames from the same rollout of the same policy,
so everything except the frame cancels.

    python scripts/probe_velocity_frame.py --checkpoint logs/.../model_700.pt --device cuda:1

Measured (256 envs, 400 steps, model_700 of the 700-iteration Isaac run): the COM
sits 0.185 m from the link origin -- above it, because ``merge_fixed_joints``
folds the torso into the root body, so the lever is more than twice the 0.076 m
the MJCF pelvis alone would suggest. The two velocities differ by 0.067 m/s on
average, and the reward still moves only 0.2% (0.5566 link, 0.5577 COM): mean
tracking error is 0.41 m/s either way, and the exp kernel at std 0.5 does not
resolve a 0.067 m/s shift on top of that. The frame is a real difference in the
quantity and a negligible one in the objective.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import torch
from pathlib import Path

def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--command", default="base_velocity")
    parser.add_argument("--std", type=float, default=None, help="Reward std; taken from the task when omitted.")
    return parser.parse_known_args()[0]


def main() -> int:
    args = _parse()

    from isaaclab.app import AppLauncher

    AppLauncher({"headless": True, "enable_cameras": True, "device": args.device})

    from instinctlab.engines.isaacsim import IsaacSimAdapter
    from instinctlab.tasks.parkour.config.g1 import parkour_target_g1

    adapter = IsaacSimAdapter()
    spec = parkour_target_g1()
    compiled = adapter.compile(spec, num_envs=args.num_envs, device=args.device)
    compiled.env_cfg.seed = args.seed

    std = args.std
    if std is None:
        std = float(spec.mdp.rewards["rewards"]["track_lin_vel_xy_exp"].params["std"])

    inner = compiled.make_env()
    env = adapter.wrap_for_rl(inner)

    from instinct_rl.runners import OnPolicyRunner

    agent_cfg = compiled.agent_cfg
    agent_cfg.device = args.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=args.device)

    obs, _ = env.reset()
    robot = inner.scene["robot"]

    link_err, com_err, link_rew, com_rew, omega, gap = [], [], [], [], [], []
    with torch.inference_mode():
        for _ in range(args.steps):
            obs, _, _, _ = env.step(policy(obs))
            command = inner.command_manager.get_command(args.command)[:, :2]
            v_link = robot.data.root_link_lin_vel_b[:, :2]
            v_com = robot.data.root_com_lin_vel_b[:, :2]
            e_link = torch.sum(torch.square(command - v_link), dim=1)
            e_com = torch.sum(torch.square(command - v_com), dim=1)
            link_err.append(e_link.sqrt().mean().item())
            com_err.append(e_com.sqrt().mean().item())
            link_rew.append(torch.exp(-e_link / std**2).mean().item())
            com_rew.append(torch.exp(-e_com / std**2).mean().item())
            omega.append(robot.data.root_ang_vel_b[:, :2].norm(dim=1).mean().item())
            gap.append((v_link - v_com).norm(dim=1).mean().item())
        com_offset = robot.data.com_pos_b[:, 0, :].mean(dim=0).cpu()
    env.close()

    def m(xs: list[float]) -> float:
        return statistics.fmean(xs)

    print("\n== velocity frame, same rollout ==")
    print(f"steps={args.steps} envs={args.num_envs} std={std}")
    print(f"  root COM offset in base frame= {com_offset.tolist()} ({com_offset.norm().item():.4f} m)")
    print(f"  mean |v_link - v_com|        = {m(gap):.4f} m/s")
    lever = float(com_offset.norm())
    print(f"  mean ||omega_xy||            = {m(omega):.4f} rad/s  (x {lever:.4f} m = {m(omega) * lever:.4f} m/s)")
    print(f"  mean tracking error, link    = {m(link_err):.4f} m/s")
    print(f"  mean tracking error, com     = {m(com_err):.4f} m/s")
    print(f"  mean reward, link frame      = {m(link_rew):.4f}")
    print(f"  mean reward, com frame       = {m(com_rew):.4f}")
    denom = m(link_rew)
    print(f"  com/link reward ratio        = {m(com_rew) / denom if denom else float('nan'):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
