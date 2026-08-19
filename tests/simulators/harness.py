"""Shared helpers for live backend behavior cells.

This module must stay import-safe: no Isaac / MJLab / Warp imports at
module level. Default ``pytest tests/`` collects then deselects marked
tests; a top-level engine import would still run.
"""

from __future__ import annotations

import torch

import pytest

from instinctlab.sim.backend import MaterialProperties, SensorReadPhase
from instinctlab.verify.scene import locomotion_flat_scene


def require_live_backend(name: str) -> None:
    if name == "mjlab":
        pytest.importorskip("mjlab")
        pytest.importorskip("mujoco")
    elif name == "isaacsim":
        pytest.importorskip("isaaclab")
    else:
        raise ValueError(f"unsupported live backend {name!r}")
    if not torch.cuda.is_available():
        pytest.skip("live backend tests require a GPU")


def make_live_backend(name: str, *, device: str = "cuda:0", bootstrap_context: object | None = None):
    require_live_backend(name)
    from instinctlab.sim.backend import BACKENDS

    provider = BACKENDS.load(name)
    if name == "isaacsim":
        if bootstrap_context is None:
            raise RuntimeError(
                "Isaac Sim live tests must construct AppLauncher before importing torch; "
                "pass bootstrap_context from the isaacsim test module"
            )
        return provider.create(device=device, bootstrap_context=bootstrap_context)
    return provider.create(device=device)


def initialize_locomotion_backend(
    name: str,
    *,
    num_envs: int = 2,
    bootstrap_context: object | None = None,
):
    cfg = locomotion_flat_scene(num_envs=num_envs)
    backend = make_live_backend(name, bootstrap_context=bootstrap_context)
    try:
        backend.initialize(cfg.scene, cfg.simulation, cfg.requirements)
    except Exception:
        close_live_backend(backend)
        raise
    return backend, cfg


def close_live_backend(backend) -> None:
    if backend.metadata.name == "isaacsim":
        backend.close(shutdown_app=False)
        return
    backend.close()


def current_root_state(backend, env_ids: torch.Tensor) -> torch.Tensor:
    data = backend.scene.articulations["robot"].data
    return torch.cat(
        (
            data.root_pos_w[env_ids],
            data.root_quat_w[env_ids],
            data.root_lin_vel_w[env_ids],
            data.root_ang_vel_w[env_ids],
        ),
        dim=1,
    ).clone()


def assert_reset_root_vel(backend) -> None:
    env_ids = torch.arange(backend.num_envs, device=backend.device, dtype=torch.int64)
    state = current_root_state(backend, env_ids)
    expected_lin = torch.tensor([0.3, -0.2, 0.1], device=backend.device)
    expected_ang = torch.tensor([0.05, -0.04, 0.02], device=backend.device)
    state[:, 7:10] = expected_lin
    state[:, 10:13] = expected_ang
    backend.write_root_state("robot", state, env_ids)
    backend.synchronize(SensorReadPhase.POST_EVENT)
    data = backend.scene.articulations["robot"].data
    torch.testing.assert_close(
        data.root_lin_vel_w[env_ids], expected_lin.expand_as(data.root_lin_vel_w[env_ids]), atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        data.root_ang_vel_w[env_ids], expected_ang.expand_as(data.root_ang_vel_w[env_ids]), atol=1e-4, rtol=1e-4
    )


def assert_air_time_advance(backend) -> None:
    sensor = backend.scene.sensors["feet_contact_forces"]
    env_ids = torch.arange(backend.num_envs, device=backend.device, dtype=torch.int64)
    before = sensor.current_air_time.clone()
    state = current_root_state(backend, env_ids)
    state[:, 2] = state[:, 2] + 1.2
    state[:, 7:13] = 0.0
    backend.write_root_state("robot", state, env_ids)
    backend.synchronize(SensorReadPhase.POST_EVENT)
    after_sync = sensor.current_air_time.clone()
    torch.testing.assert_close(after_sync, before, atol=0, rtol=0)
    backend.step()
    backend.synchronize(SensorReadPhase.POST_PHYSICS)
    assert torch.any(
        sensor.current_air_time > before
    ), "air-time must advance on step/scene.update, not on synchronize alone"


def assert_material_write_scope(backend, robot) -> None:
    env_ids = torch.arange(backend.num_envs, device=backend.device, dtype=torch.int64)
    body_names = robot.material_body_names
    body_ids = torch.tensor(
        [robot.body_names.index(name) for name in body_names],
        device=backend.device,
        dtype=torch.int64,
    )
    friction = torch.full((backend.num_envs, body_ids.numel()), 0.42, device=backend.device)
    backend.set_body_material(
        MaterialProperties(
            entity_name="robot",
            body_ids=body_ids,
            env_ids=env_ids,
            sliding_friction=friction,
        )
    )
    _assert_sliding_friction_written(backend, env_ids, body_ids, 0.42)

    frame_name = "LL_FOOT"
    assert frame_name in robot.frame_names
    assert frame_name not in robot.material_body_names
    frame_ids = torch.tensor([robot.body_names.index(frame_name)], device=backend.device, dtype=torch.int64)
    with pytest.raises(ValueError, match="material writes must target RobotSpec.material_body_names"):
        backend.set_body_material(
            MaterialProperties(
                entity_name="robot",
                body_ids=frame_ids,
                env_ids=env_ids,
                sliding_friction=torch.full((backend.num_envs, 1), 0.3, device=backend.device),
            )
        )


def _assert_sliding_friction_written(
    backend,
    env_ids: torch.Tensor,
    body_ids: torch.Tensor,
    expected: float,
) -> None:
    name = backend.metadata.name
    if name == "mjlab":
        for column in range(body_ids.numel()):
            native_local = int(backend._body_map.native_ids(body_ids[column : column + 1])[0])
            geom_ids = backend._geoms_by_native_body[native_local]
            assert geom_ids.numel() > 0
            env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
            actual = backend._sim.model.geom_friction[env_grid, geom_grid, 0]
            torch.testing.assert_close(actual, torch.full_like(actual, expected), atol=1e-5, rtol=0)
        return
    if name == "isaacsim":
        materials = backend._robot.root_physx_view.get_material_properties()
        cpu_env_ids = env_ids.cpu()
        for column in range(body_ids.numel()):
            native = int(backend._body_map.native_ids(body_ids[column : column + 1]).cpu()[0])
            start = sum(backend._shape_counts_by_native_body[:native])
            stop = start + backend._shape_counts_by_native_body[native]
            assert stop > start
            actual = materials[cpu_env_ids, start:stop, 0]
            torch.testing.assert_close(actual, torch.full_like(actual, expected), atol=1e-5, rtol=0)
        return
    raise AssertionError(f"no friction readback for backend {name!r}")
