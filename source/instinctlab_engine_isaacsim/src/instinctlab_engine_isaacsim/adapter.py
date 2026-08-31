"""The Isaac Sim backend.

Everything engine-specific about running a declared task here is in this file, its term registry and
its scene builder. That is the claim the ``N + M`` structure makes, and the size of these three
files is the evidence for it: no task file, no term in ``instinctlab/mdp``, and nothing in ``spec/``
knows this engine exists.

``bootstrap`` is separate from ``compile`` for a reason that is not stylistic. ``AppLauncher`` has to
start Isaac Sim's app before ``isaaclab`` can be imported at all, so a single entry point that took
a compiled task would need types it cannot yet import.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from instinctlab_engine.base import CompiledTask, Resolution, require_supported_version
from instinctlab_engine.compile import (
    CompileCtx,
    compile_mdp,
    contract_report,
    flatten_reward_groups,
    observation_group_settings,
)
from instinctlab_engine.spec.capability import CapabilitySet
from instinctlab_engine.spec.mdp import NoiseSpec
from instinctlab_engine.spec.task import TaskSpec

from .scene import PROFILE_DEFAULTS, build_scene
from .terms import TERMS

__all__ = ["IsaacSimAdapter", "IsaacSimCompileCtx"]

_OBSERVATION_GROUP_RESERVED_NAMES = frozenset(
    {"concatenate_terms", "concatenate_dim", "enable_corruption", "history_length", "flatten_history_dim"}
)
_INTERACTIVE_SCENE_CFG_FIELDS = frozenset(
    {
        "env_spacing",
        "filter_collisions",
        "lazy_sensor_update",
        "num_envs",
        "replicate_physics",
        "clone_in_fabric",
    }
)
_SCENE_RESERVED_NAMES = frozenset(
    {
        *_INTERACTIVE_SCENE_CFG_FIELDS,
        "robot",
        "sky_light",
        "terrain",
    }
)


def _configure_sim_contact_budget(sim: Any, profile: Mapping[str, Any], terrain_kind: str, terrain_material: Any) -> None:
    """Apply task-native PhysX budgets, falling back to the generic mesh recipe."""
    mesh_terrain = terrain_kind in {"generator", "rough"}
    patch_count = profile.get("gpu_max_rigid_patch_count")
    if patch_count is not None:
        sim.physx.gpu_max_rigid_patch_count = patch_count
    elif mesh_terrain:
        sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

    contact_count = profile.get("gpu_max_rigid_contact_count")
    if contact_count is not None:
        sim.physx.gpu_max_rigid_contact_count = contact_count

    collision_stack = profile.get("gpu_collision_stack_size")
    if collision_stack is not None:
        sim.physx.gpu_collision_stack_size = collision_stack
    elif mesh_terrain:
        sim.physx.gpu_collision_stack_size = 2**29

    if profile.get("use_terrain_physics_material", False) or mesh_terrain:
        sim.physics_material = terrain_material


def _validate_observation_term_names(spec: TaskSpec) -> None:
    """Reject term names Isaac Lab interprets as observation-group settings."""
    for group_name, group in spec.mdp.observations.items():
        collisions = sorted(set(group.terms) & _OBSERVATION_GROUP_RESERVED_NAMES)
        if collisions:
            raise ValueError(
                f"Isaac Sim observation group {group_name!r} uses reserved term names {collisions}; "
                "Isaac Lab would interpret them as group settings instead of observations."
            )


def _validate_scene_names(spec: TaskSpec) -> None:
    spec.scene.validate_symbol_table(
        reserved_names=_SCENE_RESERVED_NAMES,
        reserved_context="Isaac Sim InteractiveSceneCfg fields",
    )


class IsaacSimCompileCtx(CompileCtx):
    """Compilation context carrying Isaac Lab's noise classes."""

    def noise(self, noise: NoiseSpec | None) -> Any:
        if noise is None:
            return None
        from isaaclab.utils.noise import GaussianNoiseCfg, UniformNoiseCfg

        if noise.kind == "uniform":
            return UniformNoiseCfg(n_min=noise.lo, n_max=noise.hi, operation=noise.operation)
        if noise.kind == "gaussian":
            return GaussianNoiseCfg(mean=noise.lo, std=noise.hi, operation=noise.operation)
        raise NotImplementedError(f"Isaac Sim has no noise model for kind {noise.kind!r}.")


def _container(terms: Mapping[str, Any]) -> SimpleNamespace:
    """A term container Isaac Lab's managers can use.

    A namespace rather than a plain dict, even though the managers read either. Some of them write
    back -- ``CommandManager`` sets ``debug_vis`` on the container it was handed -- and a dict has
    nowhere to put that. Attribute order follows insertion, which is what the managers walk.
    """
    return SimpleNamespace(**terms)


def _observation_groups(compiled: Mapping[str, Any]) -> Any:
    """Observation groups as Isaac Lab's manager wants them.

    The groups have to be real ``ObservationGroupCfg`` instances -- the manager type-checks them --
    while the terms inside are found by walking ``__dict__``, so assignment order is concatenation
    order. Declaration order in the task is therefore what determines the observation vector's
    layout, which is the property that lets a policy trained here be loaded there.
    """
    from isaaclab.managers import ObservationGroupCfg

    groups: dict[str, Any] = {}
    for name, group in compiled.items():
        cfg = ObservationGroupCfg()
        for field, value in observation_group_settings(group).items():
            setattr(cfg, field, value)
        for term_name, term in group["terms"].items():
            setattr(cfg, term_name, term)
        groups[name] = cfg
    return _container(groups)


def _rewards(compiled: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Reward groups flattened into the single namespace Isaac Lab's manager has.

    Repeated names are qualified by group; otherwise flattening would silently keep only the last
    term and the policy would optimize a different objective than the declaration.
    """
    return _container(flatten_reward_groups(compiled))


def _validate_reward_scaling(spec: TaskSpec) -> None:
    """Isaac Lab always integrates reward weights over the environment step.

    Unlike mjlab, the installed Isaac Lab release has no configuration switch:
    ``RewardManager.compute`` unconditionally multiplies every term by ``dt``.
    Refuse an unscaled task instead of accepting a declaration the backend cannot honor.
    """
    if not spec.sim.scale_rewards_by_dt:
        raise ValueError(
            "Isaac Lab always scales reward terms by step_dt and cannot honor "
            "SimSpec.scale_rewards_by_dt=False. Use True for a cross-engine task."
        )


class IsaacSimAdapter:
    """Compiles a :class:`TaskSpec` into an Isaac Lab ``ManagerBasedRLEnvCfg``."""

    name = "isaacsim"
    SUPPORTED_VERSIONS = ">=0.54,<0.55"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Isaac Sim's launch flags, ``--device`` and ``--headless`` among them.

        ``AppLauncher`` declares ``--device`` itself and rejects a parser that already has one, so
        the launcher leaves that flag to the adapters; both engines end up accepting the same
        spelling with the same meaning.
        """
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)

    @staticmethod
    def bootstrap(args: argparse.Namespace) -> object:
        """Start Isaac Sim. Nothing under ``isaaclab`` may be imported before this returns.

        The torch backend settings are main's, from its own training script, and they are set here
        rather than in the launcher because the two references spell the same intent differently:
        main assigns the flags, InstinctMJ calls ``configure_torch_backends()``. Both end up with
        TF32 matmul on; they differ on ``cudnn.benchmark``, which main leaves off. Reproducing a
        reference run means reproducing the stack it ran on, and TF32 matmul changes both the
        arithmetic of the policy update and its speed.
        """
        require_supported_version("isaaclab", IsaacSimAdapter.SUPPORTED_VERSIONS, engine=IsaacSimAdapter.name)

        from isaaclab.app import AppLauncher

        app = AppLauncher(args).app

        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        return app

    @staticmethod
    def wrap_for_rl(env: Any) -> Any:
        """Wrap with main's Isaac wrapper, so training reads the same tensors main's does.

        PhysX GPU overflow is checked here: construction first, then after every
        wrapped step. The log line PhysX already prints is not a failed run.
        """
        from instinctlab_engine.diagnostics.contact_overflow import (
            attach_overflow_guard,
            check_contact_overflow,
        )

        from .rl_wrapper import InstinctRlVecEnvWrapper

        check_contact_overflow(env, phase="construction")
        return InstinctRlVecEnvWrapper(attach_overflow_guard(env))

    def capabilities(self) -> CapabilitySet:
        return TERMS.capabilities()

    def robot_spec(self, asset_id: str):
        from .assets import robot_spec

        return robot_spec(asset_id)

    def asset_conformance(self, asset_id: str) -> dict[str, Any]:
        from .assets import asset_conformance

        return asset_conformance(asset_id)

    def actuator_requirements(self, spec: TaskSpec) -> dict[str, list[str]]:
        from instinctlab_engine.actuators import task_actuator_requirements

        return task_actuator_requirements(spec, TERMS)

    def rigid_object_conformance(self, ref: Any) -> dict[str, Any]:
        from .rigid_objects import rigid_object_conformance

        return rigid_object_conformance(ref)

    def profile(self, spec: TaskSpec) -> dict[str, Any]:
        """This engine's solver settings, the task's overrides applied over the defaults."""
        merged = dict(PROFILE_DEFAULTS)
        merged.update(spec.sim.profiles.get(self.name, {}))
        return merged

    def compile(self, spec: TaskSpec, *, num_envs: int, device: str, strict: bool = False) -> CompiledTask:
        spec.validate_for_engine(self.name)
        _validate_observation_term_names(spec)
        _validate_scene_names(spec)
        _validate_reward_scaling(spec)

        from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
        from isaaclab.sim import SimulationCfg

        from .env import InstinctManagerBasedRLEnv

        profile = self.profile(spec)
        resolution = Resolution(
            engine=self.name,
            task_id=spec.task_id,
            profile=profile,
            engine_extras_used=tuple(sorted(spec.engine_extras.get(self.name, {}))),
            strict=strict,
        )
        from instinctlab_engine.lifecycle.runtime import lifecycle_manifest

        resolution.lifecycle = lifecycle_manifest(spec)
        ctx = IsaacSimCompileCtx(
            engine=self.name,
            spec=spec,
            resolution=resolution,
            profile=profile,
            num_envs=num_envs,
            device=device,
            strict=strict,
        )
        mdp = compile_mdp(spec.mdp, ctx, TERMS)

        scene = build_scene(spec.scene, spec.robot, profile, num_envs=num_envs, sensor_period=spec.sim.physics_dt)
        sim = SimulationCfg(dt=spec.sim.physics_dt, render_interval=spec.sim.decimation, device=device)
        _configure_sim_contact_budget(sim, profile, spec.scene.terrain.kind, scene.terrain.physics_material)
        env_cfg = ManagerBasedRLEnvCfg(
            scene=scene,
            observations=_observation_groups(mdp["observations"]),
            actions=_container(mdp["actions"]),
            rewards=_rewards(mdp["rewards"]),
            terminations=_container(mdp["terminations"]),
            events=_container(mdp["events"]),
            commands=_container(mdp["commands"]),
            curriculum=_container(mdp["curriculum"]),
            decimation=spec.sim.decimation,
            episode_length_s=spec.sim.episode_length_s,
            is_finite_horizon=spec.sim.is_finite_horizon,
            sim=sim,
        )
        env_cls = InstinctManagerBasedRLEnv.wrap(ManagerBasedRLEnv)

        def make_env() -> Any:
            from instinctlab_engine.motion_reference import (
                bind_motion_reference_origins,
            )

            env = env_cls(cfg=env_cfg)
            bind_motion_reference_origins(env.scene, spec.scene.motion_references)
            from instinctlab_engine.lifecycle import attach_lifecycle

            attach_lifecycle(env, spec, engine=self.name)
            return env

        resolution.capture_plugin_provenance(asset_id=spec.robot.asset_id)
        return CompiledTask(
            env_cls=env_cls,
            env_cfg=env_cfg,
            resolution=resolution,
            agent_factory=lambda: spec.agent.resolve()(**spec.agent.resolved_overrides(self.name)),
            env_factory=make_env,
        )

    def contract_report(self, spec: TaskSpec) -> dict[str, Any]:
        """What this engine would and would not provide, without importing it.

        Answerable on a machine with no Isaac Sim because the registry's keys are declared at import
        time and only the builder bodies touch ``isaaclab``. That is what makes it possible to check
        every task against every engine in one CI job.
        """
        _validate_observation_term_names(spec)
        _validate_scene_names(spec)
        _validate_reward_scaling(spec)
        return contract_report(spec, engine=self.name, registry=TERMS, capabilities=self.capabilities())

    @staticmethod
    def finalize_process(exit_code: int) -> int:
        """Avoid Isaac Sim's known post-close teardown hang."""
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
