"""Fixed-input shadowing rollout used for cross-engine runtime evidence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any


def shadowing_task_with_motion(task_id: str, clip_path: str | Path) -> Any:
    """Return a registered shadowing task with one explicit motion clip binding.

    This is a diagnostic override, not a second task registration. Keeping the original task ID
    makes reports comparable with the normal launcher, while the task-contract hash still records
    the changed dataset binding and prevents accidental checkpoint interchange.
    """
    from instinctlab.tasks import registry

    clip = Path(clip_path).expanduser().resolve()
    if not clip.is_file():
        raise FileNotFoundError(f"shadowing probe motion clip not found: {clip}")
    task = registry.spec(task_id)
    if len(task.scene.motion_references) != 1:
        raise ValueError(
            "shadowing motion override requires exactly one reference, got "
            f"{len(task.scene.motion_references)} for {task_id!r}"
        )
    old = task.scene.motion_references[0]
    motion = replace(
        old,
        clip=str(clip),
        engine_clips={"isaacsim": str(clip), "mjlab": str(clip)},
        selected_files=(),
        first_motion_only=True,
    )

    def patch_term(term):
        return replace(
            term,
            params={key: motion if value is old else value for key, value in term.params.items()},
        )

    observations = {
        name: replace(
            group,
            terms={term_name: patch_term(term) for term_name, term in group.terms.items()},
        )
        for name, group in task.mdp.observations.items()
    }
    terminations = {name: patch_term(term) for name, term in task.mdp.terminations.items()}
    return replace(
        task,
        scene=replace(task.scene, motion_references=(motion,)),
        mdp=replace(task.mdp, observations=observations, terminations=terminations),
    )


def collect_shadowing_rollout(env: Any, task: Any, *, steps: int) -> dict[str, Any]:
    """Run a fixed reference state through MDP stepping and return runtime evidence.

    The native reset is deliberately allowed to run first so the probe exercises every reset
    manager.  It is then replaced with one non-mirrored clip sample before capture; otherwise the
    engines' intentionally independent random streams would make a fixed seed look like a state
    comparison when it was only a reproducibility check within each engine.
    """
    import torch

    from instinctlab.engines.shadowing_events import reset_robot_from_reference

    env.reset()
    robot = env.scene["robot"]
    sensor = env.scene["motion_reference"]
    env_ids = torch.arange(env.num_envs, device=env.device)
    runtime = sensor._runtime
    runtime.buffers.motion_id[env_ids] = 0
    runtime.buffers.start_s[env_ids] = min(0.25, runtime.clips[0].sampling_length_s * 0.25)
    runtime.buffers.timestamp[env_ids] = 0.0
    runtime.last_update[env_ids] = 0.0
    runtime.mask[env_ids] = False
    runtime.refresh_initial(env_ids)
    runtime.refresh_at_current_time(env_ids)
    reset_robot_from_reference(env, env_ids, randomize_joint_pos_range=(0.0, 0.0))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)
    env.command_manager.reset(env_ids)
    fixed_start_s = runtime.buffers.start_s.detach().cpu().clone()
    action = torch.linspace(-0.2, 0.2, len(task.robot.joint_names), device=env.device).repeat(env.num_envs, 1)
    fields = {
        name: []
        for name in (
            "joint_pos",
            "joint_vel",
            "root_pos",
            "root_quat",
            "root_vel",
            "motion_pos",
            "reward",
            "done",
        )
    }

    def native_field(data, *names):
        for name in names:
            if hasattr(data, name):
                return getattr(data, name)
        raise AttributeError(f"robot data has none of {names}")

    def capture(reward=None, done=None):
        data = robot.data
        fields["joint_pos"].append(data.joint_pos.detach().cpu())
        fields["joint_vel"].append(data.joint_vel.detach().cpu())
        fields["root_pos"].append(native_field(data, "root_link_pos_w", "root_pos_w").detach().cpu())
        fields["root_quat"].append(native_field(data, "root_link_quat_w", "root_quat_w").detach().cpu())
        fields["root_vel"].append(native_field(data, "root_link_vel_w", "root_vel_w").detach().cpu())
        fields["motion_pos"].append(sensor.data.base_pos_w[:, 0].detach().cpu())
        fields["reward"].append(torch.zeros(env.num_envs) if reward is None else reward.detach().cpu())
        fields["done"].append(torch.zeros(env.num_envs, dtype=torch.bool) if done is None else done.detach().cpu())

    capture()
    for _ in range(steps):
        _, reward, terminated, truncated, _ = env.step(action)
        if isinstance(reward, dict):
            reward = torch.stack(tuple(reward.values()), dim=-1).sum(dim=-1)
        capture(reward, terminated | truncated)
    result = {name: torch.stack(values).numpy() for name, values in fields.items()}
    result["action"] = action.detach().cpu().numpy()
    result["motion_start_s"] = fixed_start_s.numpy()
    return result


__all__ = ["collect_shadowing_rollout", "shadowing_task_with_motion"]
