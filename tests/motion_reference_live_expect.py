"""Shared live checks for the engine-neutral motion-reference lifecycle."""

from __future__ import annotations

from typing import Any


def motion_only_parkour_task() -> Any:
    """Parkour's robot and AMP terms on a plane, without camera or terrain setup."""
    from dataclasses import replace

    from instinctlab.spec.mdp import MdpSpec
    from instinctlab.spec.task import SceneSpec, TerrainSpec
    from instinctlab.tasks.parkour.config.g1 import parkour_target_g1

    task = parkour_target_g1()
    scene = SceneSpec(
        terrain=TerrainSpec(),
        motion_references=task.scene.motion_references,
        env_spacing=task.scene.env_spacing,
    )
    mdp = MdpSpec(
        observations={name: task.mdp.observations[name] for name in ("amp_policy", "amp_reference")},
        actions=task.mdp.actions,
        terminations={"time_out": task.mdp.terminations["time_out"]},
        events={name: task.mdp.events[name] for name in ("reset_base", "reset_robot_joints")},
    )
    return replace(
        task,
        task_id="Instinct-Motion-Reference-Live",
        scene=scene,
        sim=replace(task.sim, episode_length_s=1.0, profiles={}),
        mdp=mdp,
        engine_extras={},
    )


def assert_motion_clock_and_exhaustion(env: Any, task: Any, *, device: str) -> None:
    """Check reset, refresh cadence, world origins and exhaustion on a real scene."""
    import torch

    from instinctlab.mdp.terminations import dataset_exhausted

    ref = task.scene.motion_reference("motion_reference")
    sensor = env.scene.sensors[ref.name]
    runtime = sensor._runtime
    env.reset()

    data = sensor.data
    zeros = torch.zeros(env.num_envs, device=device)
    torch.testing.assert_close(data.timestamp, zeros)
    torch.testing.assert_close(runtime.last_update, zeros)
    assert sensor.aiming_frame_idx.tolist() == [0] * env.num_envs
    assert bool(data.validity[:, 0].all())

    frame_before = data.frame_index[:, 0].clone()
    action = torch.zeros(env.num_envs, len(task.robot.joint_names), device=device)
    env.step(action)
    data = sensor.data
    expected_time = torch.full_like(data.timestamp, task.sim.step_dt)
    torch.testing.assert_close(data.timestamp, expected_time)
    torch.testing.assert_close(runtime.last_update, expected_time)
    assert sensor.aiming_frame_idx.tolist() == [0] * env.num_envs
    assert torch.equal(data.frame_index[:, 0], frame_before + 1)

    env_origins = env.scene.env_origins
    clip_root = runtime.clip.base_pos_w[data.frame_index[:, 0]].clone()
    clip_root[runtime.mask, 1] *= -1.0
    torch.testing.assert_close(data.base_pos_w[:, 0], clip_root + env_origins)

    env.reset()
    env_ids = torch.arange(env.num_envs, device=device)
    runtime.buffers.start_s[:] = runtime.clip.duration_s - ref.frame_interval_s
    runtime.buffers.timestamp[:] = 0.0
    runtime.last_update[:] = 0.0
    runtime.refresh_at_current_time(env_ids)
    assert bool(runtime.buffers.validity[:, 0].all())

    env.step(action)
    _ = sensor.data
    assert not bool(runtime.buffers.validity[:, 0].any())
    exhausted = dataset_exhausted(env, ref, reset_without_notice=False)
    assert bool(exhausted.all())

    hidden = dataset_exhausted(env, ref, reset_without_notice=True)
    assert not bool(hidden.any())
    assert bool(sensor.data.validity[:, 0].all())


__all__ = ["assert_motion_clock_and_exhaustion", "motion_only_parkour_task"]
