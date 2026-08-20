"""Measure the parkour depth camera on one engine, from a pose written in by hand.

Construction success is not the claim. Run once per engine and read the printout
side by side:

    python scripts/probe_depth_camera.py --engine mjlab    --device cuda:2
    python scripts/probe_depth_camera.py --engine isaacsim --device cuda:2 --headless

A plane (z = 0) is the known-distance case. Expected image-plane depth is the
ray-plane intersection from the *live* camera pose, not a remembered pelvis
height. A sky-pointing pose off the mesh is the miss. Stairs are Isaac mesh vs
mjlab boxes -- report the numbers; do not widen the plane tolerance if they
disagree by a step.
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

NUM_ENVS = 2
SEED = 12345
# Both engines put the plane at z = 0. 20 mm is ~1% at a 2 m optical-axis hit
# and well under a stair riser (0.25 m). Do not widen if they disagree by more.
PLANE_TOL_M = 0.020
SELF_HIT_MARGIN_M = 0.05
CAMERA = "camera"
OPTICAL_AXIS = (18, 32)  # raw row, col (0-index); top-center of the 18×32 crop
PROBE_PIXELS = (
    ("optical_axis", 18, 32),
    ("crop_top_left", 18, 16),
    ("crop_top_right", 18, 47),
    ("crop_bot_left", 35, 16),
    ("crop_bot_right", 35, 47),
    ("crop_bot_center", 35, 32),
)


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=("isaacsim", "mjlab"))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--case", choices=("flat", "tilt", "stairs", "miss", "arms", "all"), default="all")
    return parser.parse_known_args()[0]


def _task(terrain):
    from instinctlab.spec.capability import Requirement
    from instinctlab.tasks.parkour.config.g1 import parkour_target_g1

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
        )
    )


def _flat_task():
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
    env.scene.sensors[CAMERA].update(dt, force_recompute=True)


def _write_root(
    env,
    offset_xyz: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
    joints: dict[str, float] | None = None,
) -> None:
    robot = env.scene["robot"]
    device = env.device
    env_ids = torch.arange(env.num_envs, device=device)
    pos = env.scene.env_origins + torch.tensor(offset_xyz, device=device)
    quat = torch.tensor(quat_wxyz, device=device).expand(env.num_envs, 4)
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6, device=device), env_ids=env_ids)
    q = robot.data.default_joint_pos.clone()
    if joints:
        names = list(robot.joint_names)
        for name, value in joints.items():
            q[:, names.index(name)] = value
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    env.scene.write_data_to_sim()
    if hasattr(env.sim, "forward"):
        env.sim.forward()
    _force_refresh(env)


# Listed links, not hands. On this G1, +pitch swings the arms behind the
# camera; -pitch puts the elbows into the 48° look-down cone.
_ARMS_IN_VIEW = {
    "left_shoulder_pitch_joint": -1.2,
    "right_shoulder_pitch_joint": -1.2,
    "left_elbow_joint": 0.9,
    "right_elbow_joint": 0.9,
}


def _pinhole_dirs(width: int, height: int, h_ap: float, v_ap: float, focal: float, device: str) -> torch.Tensor:
    from instinctlab.engines.mjlab.camera import _world_pinhole_rays

    _, directions = _world_pinhole_rays(
        width=width,
        height=height,
        horizontal_aperture=h_ap,
        vertical_aperture=v_ap,
        focal_length=focal,
        device=device,
    )
    return directions


# Same tilt the live helper writes. roll=0.5 / pitch=0.6 / yaw=0.2.
_TILT_ROLL = 0.5
_TILT_PITCH = 0.6
_TILT_YAW = 0.2


def _tilt_quat(device) -> tuple[float, float, float, float]:
    from instinctlab.compat.math import quat_from_euler_xyz

    quat = quat_from_euler_xyz(
        torch.tensor([_TILT_ROLL], device=device),
        torch.tensor([_TILT_PITCH], device=device),
        torch.tensor([_TILT_YAW], device=device),
    )[0]
    return tuple(float(v) for v in quat.tolist())


def _torso_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    names = list(robot.body_names)
    torso = names.index("torso_link")
    return robot.data.body_link_pos_w[0, torso].detach(), robot.data.body_link_quat_w[0, torso].detach()


def _camera_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Live camera pose. Prefer the sensor's own buffers; reconstruct if absent."""
    from instinctlab.compat.math import quat_apply, quat_mul
    from instinctlab.tasks.parkour.config.g1.target_env_cfg import DEPTH_CAMERA

    data = env.scene.sensors[CAMERA].data
    pos = getattr(data, "pos_w", None)
    quat = getattr(data, "quat_w_world", None)
    if pos is not None and quat is not None:
        return pos[0].detach(), quat[0].detach()
    torso_pos, torso_quat = _torso_pose(env)
    offset = torch.tensor(DEPTH_CAMERA.offset, device=torso_pos.device, dtype=torso_pos.dtype)
    offset_rot = torch.tensor(DEPTH_CAMERA.offset_rot, device=torso_pos.device, dtype=torso_pos.dtype)
    cam_pos = torso_pos + quat_apply(torso_quat.unsqueeze(0), offset.unsqueeze(0))[0]
    cam_quat = quat_mul(torso_quat.unsqueeze(0), offset_rot.unsqueeze(0))[0]
    return cam_pos.detach(), cam_quat.detach()


def _expected_plane_depth(
    pos: torch.Tensor, quat: torch.Tensor, dirs: torch.Tensor, z_plane: float = 0.0
) -> torch.Tensor:
    from instinctlab.compat.math import quat_apply, quat_inv

    n = dirs.shape[0]
    quat_e = quat.unsqueeze(0).expand(n, 4)
    world = quat_apply(quat_e, dirs)
    denom = world[:, 2]
    t = (z_plane - pos[2]) / denom
    valid = (t > 1e-4) & (denom.abs() > 1e-6)
    hits = pos.unsqueeze(0) + t.unsqueeze(-1) * world
    cam = quat_apply(quat_inv(quat).unsqueeze(0).expand(n, 4), hits - pos)
    depth = torch.where(valid, cam[:, 0], torch.full((n,), float("inf"), device=pos.device))
    return depth


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "+inf" if value > 0 else str(value)
    return f"{value:.4f}"


def _read(env) -> dict:
    from instinctlab.compat.sensors import depth_image
    from instinctlab.tasks.parkour.config.g1.target_env_cfg import DEPTH_CAMERA

    raw = depth_image(env.scene.sensors[CAMERA])[0, ..., 0].detach()
    pos, quat = _camera_pose(env)
    dirs = _pinhole_dirs(
        DEPTH_CAMERA.pattern.width,
        DEPTH_CAMERA.pattern.height,
        DEPTH_CAMERA.pattern.horizontal_aperture,
        DEPTH_CAMERA.pattern.vertical_aperture,
        DEPTH_CAMERA.pattern.focal_length,
        str(raw.device),
    )
    expected = _expected_plane_depth(pos, quat, dirs).view(raw.shape)
    finite = torch.isfinite(raw)
    expected_finite = torch.isfinite(expected) & (expected <= DEPTH_CAMERA.max_distance)
    self_hit = finite & expected_finite & (raw < expected - SELF_HIT_MARGIN_M)
    plane_ok = finite & expected_finite & ((raw - expected).abs() <= PLANE_TOL_M)
    pixels = {}
    for name, row, col in PROBE_PIXELS:
        pixels[name] = {
            "row": row,
            "col": col,
            "measured": float(raw[row, col]),
            "expected_plane": float(expected[row, col]),
        }
    env.observation_manager.compute()
    processed = env.observation_manager.compute()["policy"]["depth_image"][0, -1].detach()
    torso_pos, torso_quat = _torso_pose(env)
    from instinctlab.engines.ray_alignment import camera_pose_for_alignment

    offset = torch.tensor(DEPTH_CAMERA.offset, device=torso_pos.device, dtype=torso_pos.dtype)
    offset_rot = torch.tensor(DEPTH_CAMERA.offset_rot, device=torso_pos.device, dtype=torso_pos.dtype)
    base_pos, base_quat = camera_pose_for_alignment(
        torso_pos.unsqueeze(0), torso_quat.unsqueeze(0), offset, offset_rot, "base"
    )
    yaw_pos, yaw_quat = camera_pose_for_alignment(
        torso_pos.unsqueeze(0), torso_quat.unsqueeze(0), offset, offset_rot, "yaw"
    )
    expected_yaw = _expected_plane_depth(yaw_pos[0], yaw_quat[0], dirs).view(raw.shape)
    expected_base = _expected_plane_depth(base_pos[0], base_quat[0], dirs).view(raw.shape)
    return {
        "cam_pos_w": pos.float().cpu().tolist(),
        "cam_quat_w": quat.float().cpu().tolist(),
        "torso_pos_w": torso_pos.float().cpu().tolist(),
        "base_cam_pos_w": base_pos[0].float().cpu().tolist(),
        "yaw_cam_pos_w": yaw_pos[0].float().cpu().tolist(),
        "origin_slide_m": float((base_pos[0] - yaw_pos[0]).norm()),
        "origin_err_base_m": float((pos - base_pos[0]).norm()),
        "origin_err_yaw_m": float((pos - yaw_pos[0]).norm()),
        "optical_axis_yaw_plane": float(expected_yaw[OPTICAL_AXIS]),
        "optical_axis_base_plane": float(expected_base[OPTICAL_AXIS]),
        "raw_shape": list(raw.shape),
        "processed_latest_shape": list(processed.shape),
        "raw_finite_frac": float(finite.float().mean()),
        "raw_min": float(raw[finite].min()) if bool(finite.any()) else float("inf"),
        "raw_max": float(raw[finite].max()) if bool(finite.any()) else float("inf"),
        "raw_mean": float(raw[finite].mean()) if bool(finite.any()) else float("inf"),
        "raw_miss_frac": float((~finite).float().mean()),
        "plane_agree_frac": float(plane_ok.float().mean()),
        "self_hit_frac": float(self_hit.float().mean()),
        "pixels": pixels,
        "processed_latest_min": float(processed.min()),
        "processed_latest_max": float(processed.max()),
        "processed_latest_mean": float(processed.mean()),
        "optical_axis": {
            "measured": float(raw[OPTICAL_AXIS]),
            "expected_plane": float(expected[OPTICAL_AXIS]),
        },
    }


def _print_case(name: str, reading: dict) -> None:
    print(f"  [{name}]")
    pos = reading["cam_pos_w"]
    print(
        f"    cam_pos=({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}) "
        f"finite={reading['raw_finite_frac']:.3f} miss={reading['raw_miss_frac']:.3f} "
        f"plane_agree={reading['plane_agree_frac']:.3f} self_hit={reading['self_hit_frac']:.3f}"
    )
    print(
        "    raw finite min/mean/max = "
        f"{_fmt(reading['raw_min'])} / {_fmt(reading['raw_mean'])} / {_fmt(reading['raw_max'])}"
    )
    axis = reading["optical_axis"]
    delta = (
        abs(axis["measured"] - axis["expected_plane"])
        if math.isfinite(axis["measured"]) and math.isfinite(axis["expected_plane"])
        else float("inf")
    )
    print(
        f"    optical_axis measured={_fmt(axis['measured'])} "
        f"expected_plane={_fmt(axis['expected_plane'])} |Δ|={_fmt(delta)} "
        f"(tol {PLANE_TOL_M*1e3:.0f} mm)"
    )
    if "origin_slide_m" in reading:
        yaw_plane = reading["optical_axis_yaw_plane"]
        base_plane = reading["optical_axis_base_plane"]
        yaw_delta = (
            abs(axis["measured"] - yaw_plane)
            if math.isfinite(axis["measured"]) and math.isfinite(yaw_plane)
            else float("inf")
        )
        base_delta = (
            abs(axis["measured"] - base_plane)
            if math.isfinite(axis["measured"]) and math.isfinite(base_plane)
            else float("inf")
        )
        print(
            f"    alignment origin |base-yaw|={reading['origin_slide_m']:.4f} m  "
            f"|meas-base|={reading['origin_err_base_m']:.4f}  "
            f"|meas-yaw|={reading['origin_err_yaw_m']:.4f}"
        )
        print(
            f"    optical_axis base-plane={_fmt(base_plane)} |meas-base|={_fmt(base_delta)}  "
            f"yaw-only plane={_fmt(yaw_plane)} |meas-yaw|={_fmt(yaw_delta)} "
            f"(yaw gap must stay >> {PLANE_TOL_M*1e3:.0f} mm)"
        )
    for name, pix in reading["pixels"].items():
        print(
            f"      {name:16s} [{pix['row']:2d},{pix['col']:2d}]  "
            f"meas={_fmt(pix['measured'])}  plane={_fmt(pix['expected_plane'])}"
        )
    print(
        "    processed latest min/mean/max = "
        f"{reading['processed_latest_min']:.4f} / {reading['processed_latest_mean']:.4f} / "
        f"{reading['processed_latest_max']:.4f}  shape={reading['processed_latest_shape']}"
    )


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
        "cases": {},
    }
    print(f"engine={args.engine} device={args.device} num_envs={NUM_ENVS} case={args.case}")
    print(f"tolerance: plane {PLANE_TOL_M*1e3:.0f} mm (do not widen)")

    if args.case in {"flat", "tilt", "miss", "arms", "all"}:
        env = _make_env(args.engine, _flat_task(), args.device)
        try:
            if args.case in {"flat", "all"}:
                _write_root(env, (0.0, 0.0, 0.82), (1.0, 0.0, 0.0, 0.0))
                reading = _read(env)
                report["cases"]["flat"] = reading
                _print_case("flat (z=0 plane, standing, 48° look-down)", reading)
            if args.case in {"tilt", "all"}:
                _write_root(env, (0.0, 0.0, 0.82), _tilt_quat(args.device))
                reading = _read(env)
                report["cases"]["tilt"] = reading
                _print_case(
                    "tilt (root roll=0.5 pitch=0.6 yaw=0.2; base vs yaw-only)",
                    reading,
                )
            if args.case in {"arms", "all"}:
                _write_root(env, (0.0, 0.0, 0.82), (1.0, 0.0, 0.0, 0.0), _ARMS_IN_VIEW)
                reading = _read(env)
                report["cases"]["arms"] = reading
                _print_case("arms (listed elbow/shoulder meshes in the cone)", reading)
            if args.case in {"miss", "all"}:
                _write_root(env, (200.0, 200.0, 5.0), (1.0, 0.0, 0.0, 0.0))
                reading = _read(env)
                report["cases"]["miss"] = reading
                _print_case("miss (origin+200 m, z=5, identity; ground out of 2.5 m)", reading)
        finally:
            env.close()
        if args.case == "all" and args.engine == "isaacsim":
            print("skipping stairs in this Isaac process: a second env.validate() recurses after close")
            if args.out is not None:
                args.out.write_text(json.dumps(report, indent=2))
                print(f"wrote {args.out}")
            sys.stdout.flush()
            os._exit(0)

    if args.case in {"stairs", "all"}:
        env = _make_env(args.engine, _stairs_task(), args.device)
        try:
            _write_root(env, (0.0, 0.0, 1.20), (1.0, 0.0, 0.0, 0.0))
            reading = _read(env)
            report["cases"]["stairs"] = reading
            _print_case("stairs platform center (step=0.25, Isaac mesh / mjlab box)", reading)
        finally:
            env.close()

    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
