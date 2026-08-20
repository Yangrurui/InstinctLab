"""Measure the foot-height scanner on one engine, from a pose written in by hand.

Construction success is not the claim. The claim is that the two engines return the
same numbers for the same physical situation. Run once per engine and read the
printout side by side:

    python scripts/probe_foot_scanner.py --engine mjlab    --device cuda:2
    python scripts/probe_foot_scanner.py --engine isaacsim --device cuda:2 --headless

Do not use the parkour rough grid for the comparison: Isaac builds Perlin meshes,
mjlab builds boxes / heightfields, and they are different ground. The cases here
are a plane (z = 0), a 1×1 pyramid-stairs tile with a fixed 0.25 m step, and a
teleport off that finite tile (a miss). A pitched-foot case on the stairs platform
guards the offset: yaw-only keeps the 20 m world-up; a full-R offset walks off
the 2.5 m platform.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import torch
from dataclasses import replace
from pathlib import Path

NUM_ENVS = 4
SEED = 12345
# Plane hits should agree to a few millimetres: both engines put z=0 under an
# upright robot. Stairs are Isaac mesh vs mjlab boxes; 1 cm is the starting
# tolerance, not a licence to widen if they disagree by a step (0.25 m).
PLANE_TOL_M = 0.005
STAIRS_TOL_M = 0.01


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=("isaacsim", "mjlab"))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--case",
        choices=("flat", "stairs", "all"),
        default="all",
        help="Isaac Kit cannot construct a second env after the first is closed; run flat and stairs separately there.",
    )
    return parser.parse_known_args()[0]


def _task(terrain, tile_name: str):
    from instinctlab.spec.capability import Requirement
    from instinctlab.tasks.parkour.config.g1 import parkour_target_g1

    del tile_name
    spec = parkour_target_g1()
    command = replace(
        spec.mdp.commands["base_velocity"],
        kind="uniform_velocity",
        params={
            "entity": "robot",
            "resampling_time_range": (1.0e9, 1.0e9),
            "lin_vel_x": (0.0, 0.0),
            "lin_vel_y": (0.0, 0.0),
            "ang_vel_z": (0.0, 0.0),
            "heading_command": True,
            "heading": (0.0, 0.0),
            "debug_vis": False,
        },
        engine_params={},
        level=Requirement.REQUIRED,
    )
    return replace(
        spec,
        scene=replace(spec.scene, terrain=terrain, env_spacing=8.0),
        mdp=replace(spec.mdp, commands={"base_velocity": command}, curriculum={}),
    )


def _one_tile(kind: str, params: dict):
    """A 1×1 curriculum grid. pose_velocity refuses a plane and a non-curriculum mix."""
    from instinctlab.spec import SubTerrainSpec, TerrainGeneratorSpec, TerrainSpec

    return _task(
        TerrainSpec(
            kind="generator",
            generator=TerrainGeneratorSpec(
                size=(8.0, 8.0),
                border_width=0.0,
                num_rows=1,
                num_cols=1,
                curriculum=True,
                max_init_level=0,
                seed=1,
                sub_terrains={kind: SubTerrainSpec(kind=kind, proportion=1.0, params=params)},
            ),
        ),
        tile_name=kind,
    )


def _flat_task():
    """Zero-noise heightfield. Both engines put the surface at world z = 0."""
    return _one_tile("random_rough", {"noise_range": (0.0, 0.0), "noise_step": 0.005})


def _stairs_task():
    return _one_tile(
        "pyramid_stairs",
        {
            "step_height_range": (0.25, 0.25),
            "step_width": 0.3,
            "platform_width": 2.5,
            "border_width": 0.0,
        },
    )


def _make_env(engine: str, spec, device: str):
    if engine == "isaacsim":
        from instinctlab.engines.isaacsim import IsaacSimAdapter as Adapter
    else:
        from instinctlab.engines.mjlab import MjlabAdapter as Adapter

    compiled = Adapter().compile(spec, num_envs=NUM_ENVS, device=device)
    compiled.env_cfg.seed = SEED
    env = compiled.make_env()
    env.reset()
    return env


def _force_refresh(env) -> None:
    if hasattr(env.sim, "sense"):
        env.sim.sense()
        env.scene.update(float(env.step_dt))
        return
    dt = max(float(getattr(env, "step_dt", 0.02)), 1.0)
    for name in ("left_height_scanner", "right_height_scanner"):
        env.scene.sensors[name].update(dt, force_recompute=True)


def _write_root(env, offset_xyz: tuple[float, float, float], quat_wxyz: tuple[float, float, float, float]) -> None:
    robot = env.scene["robot"]
    device = env.device
    env_ids = torch.arange(env.num_envs, device=device)
    pos = env.scene.env_origins + torch.tensor(offset_xyz, device=device)
    quat = torch.tensor(quat_wxyz, device=device).expand(env.num_envs, 4)
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    zero_vel = torch.zeros(env.num_envs, 6, device=device)
    robot.write_root_link_velocity_to_sim(zero_vel, env_ids=env_ids)
    default_q = robot.data.default_joint_pos
    robot.write_joint_state_to_sim(default_q, torch.zeros_like(default_q), env_ids=env_ids)
    env.scene.write_data_to_sim()
    if hasattr(env.sim, "forward"):
        env.sim.forward()
    _force_refresh(env)


def _read(env) -> dict[str, list]:
    from instinctlab.compat.sensors import ray_hits_w

    robot = env.scene["robot"]
    names = list(robot.body_names)
    left_id = names.index("left_ankle_roll_link")
    right_id = names.index("right_ankle_roll_link")
    left_hits = ray_hits_w(env.scene.sensors["left_height_scanner"])
    right_hits = ray_hits_w(env.scene.sensors["right_height_scanner"])
    return {
        "env_origins": env.scene.env_origins.detach().float().cpu().tolist(),
        "left_ankle_w": robot.data.body_link_pos_w[:, left_id].detach().float().cpu().tolist(),
        "right_ankle_w": robot.data.body_link_pos_w[:, right_id].detach().float().cpu().tolist(),
        "left_hits_w": left_hits.detach().float().cpu().tolist(),
        "right_hits_w": right_hits.detach().float().cpu().tolist(),
    }


def _fmt_hits(hits: list) -> str:
    def one(v):
        if any(not math.isfinite(x) for x in v):
            return "+inf" if all((not math.isfinite(x)) and x > 0 for x in v) else str(v)
        return f"({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})"

    return " ".join(one(ray) for ray in hits)


def _print_case(name: str, reading: dict) -> None:
    print(f"  [{name}]")
    for i, (origin, left_a, right_a, left_h, right_h) in enumerate(
        zip(
            reading["env_origins"],
            reading["left_ankle_w"],
            reading["right_ankle_w"],
            reading["left_hits_w"],
            reading["right_hits_w"],
            strict=True,
        )
    ):
        print(f"    env{i} origin_z={origin[2]:+.4f} L_ankle_z={left_a[2]:+.4f} R_ankle_z={right_a[2]:+.4f}")
        print(f"         left_hits  {_fmt_hits(left_h)}")
        print(f"         right_hits {_fmt_hits(right_h)}")


def _pitch_quat(radians: float) -> tuple[float, float, float, float]:
    half = 0.5 * radians
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def main() -> int:
    args = _parse()
    if args.engine == "isaacsim":
        from isaaclab.app import AppLauncher

        AppLauncher({"headless": True, "enable_cameras": False, "device": args.device})

    report: dict[str, object] = {
        "engine": args.engine,
        "device": args.device,
        "num_envs": NUM_ENVS,
        "plane_tol_m": PLANE_TOL_M,
        "stairs_tol_m": STAIRS_TOL_M,
        "cases": {},
    }

    print(f"engine={args.engine} device={args.device} num_envs={NUM_ENVS} case={args.case}")
    print(f"tolerances: plane {PLANE_TOL_M*1e3:.0f} mm, stairs {STAIRS_TOL_M*1e3:.0f} mm (do not widen)")

    if args.case in {"flat", "all"}:
        flat_env = _make_env(args.engine, _flat_task(), args.device)
        try:
            _write_root(flat_env, (0.0, 0.0, 0.82), (1.0, 0.0, 0.0, 0.0))
            reading = _read(flat_env)
            report["cases"]["flat"] = reading
            _print_case("flat (zero-noise heightfield, expect hit_z ≈ 0)", reading)
        finally:
            flat_env.close()
        if args.case == "all" and args.engine == "isaacsim":
            print("skipping stairs in this Isaac process: a second env.validate() recurses after close")
            if args.out is not None:
                args.out.write_text(json.dumps(report, indent=2))
                print(f"wrote {args.out}")
            sys.stdout.flush()
            os._exit(0)

    if args.case in {"stairs", "all"}:
        stairs_env = _make_env(args.engine, _stairs_task(), args.device)
        try:
            _write_root(stairs_env, (0.0, 0.0, 0.82), (1.0, 0.0, 0.0, 0.0))
            reading = _read(stairs_env)
            report["cases"]["stairs_center"] = reading
            _print_case("stairs platform center (step=0.25, Isaac mesh / mjlab box)", reading)

            _write_root(stairs_env, (1.25, 0.0, 0.82), (1.0, 0.0, 0.0, 0.0))
            reading = _read(stairs_env)
            report["cases"]["stairs_edge"] = reading
            _print_case("stairs platform +x edge (x=+1.25)", reading)

            _write_root(stairs_env, (0.0, 0.0, 1.20), _pitch_quat(0.35))
            reading = _read(stairs_env)
            report["cases"]["stairs_pitched"] = reading
            _print_case("stairs center, root pitch 0.35 rad (yaw-only offset stays over platform)", reading)

            _write_root(stairs_env, (200.0, 200.0, 1.0), (1.0, 0.0, 0.0, 0.0))
            reading = _read(stairs_env)
            report["cases"]["miss"] = reading
            _print_case("miss (origin + 200 m)", reading)
        finally:
            stairs_env.close()

    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
