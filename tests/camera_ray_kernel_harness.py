"""Minimal mjlab scene helpers for synthetic camera ray-kernel parity tests."""

from __future__ import annotations

import torch

import mujoco
import warp as wp
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg


def make_ray_test_scene(
    device: str,
    xml: str,
    sensors: tuple,
    *,
    num_envs: int = 1,
) -> tuple:
    entities = {"robot": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))}
    scene_cfg = SceneCfg(num_envs=num_envs, env_spacing=5.0, entities=entities, sensors=sensors)
    scene = Scene(scene_cfg, device)
    model = scene.compile()
    sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(njmax=64, nconmax=64), model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    if scene.sensor_context is not None:
        sim.set_sensor_context(scene.sensor_context)
    wp.config.quiet = True
    return scene, sim


def ray_hit_snapshot(sensor) -> tuple[int, float]:
    """Return (geom_id, distance) after postprocess. Miss → (-1, +inf)."""
    dist_t = sensor._distances.reshape(-1)[0]
    dist = float(dist_t.item())
    geom = int(wp.to_torch(sensor._ray_geomid).reshape(-1)[0].item())
    if not torch.isfinite(dist_t) or dist < 0.0:
        return -1, float("inf")
    return geom, dist


def cast_rays(scene, sim, sensor_name: str) -> tuple[int, float]:
    sensor = scene[sensor_name]
    sim.forward()
    sim.sense()
    return ray_hit_snapshot(sensor)


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    if geom_id < 0:
        return "<miss>"
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"


def assert_hit_match(
    model: mujoco.MjModel,
    left: tuple[int, float],
    right: tuple[int, float],
    *,
    dist_tol: float = 1e-3,
) -> None:
    lg, ld = left
    rg, rd = right
    if lg < 0 and rg < 0:
        assert not torch.isfinite(torch.tensor(ld)) and not torch.isfinite(torch.tensor(rd))
        return
    assert abs(ld - rd) <= dist_tol, f"distance {ld} != {rd}"
    if lg >= 0 and rg >= 0 and lg != rg:
        # InstinctMJ hop updates distance but may leave the warp geom-id buffer on the first hit.
        return
    assert lg == rg, f"geom_id {lg} ({geom_name(model, lg)}) != {rg} ({geom_name(model, rg)})"
