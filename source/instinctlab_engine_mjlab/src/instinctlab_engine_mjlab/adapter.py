"""The mjlab backend.

Deliberately shaped like ``engines/isaacsim/adapter.py``, because the comparison between them is
the evidence for the whole structure. What differs is small and concrete: mjlab needs no bootstrap
step, its managers take plain dicts, and its observation groups carry their terms in a ``terms``
field instead of as attributes.

The compilation targets mjlab's own ``ManagerBasedRlEnvCfg`` rather than InstinctMJ's subclass.
Per decision D3 InstinctMJ is a reference, not a dependency, and the one thing its subclass adds
that this needs -- reward groups -- is handled the same way as on Isaac Sim: the groups live in the
declaration, and both backends flatten them into the single namespace their manager has.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any, ClassVar

from instinctlab_engine.base import CompiledTask, Resolution, require_supported_version
from instinctlab_engine.compile import (
    CompileCtx,
    compile_mdp,
    contract_report,
    flatten_reward_groups,
    observation_group_settings,
    record_reward_omissions,
)
from instinctlab_engine.spec.capability import CapabilitySet
from instinctlab_engine.spec.mdp import NoiseSpec
from instinctlab_engine.spec.task import TaskSpec

from .scene import PROFILE_DEFAULTS, build_scene
from .terms import TERMS

__all__ = ["MjlabAdapter", "MjlabCompileCtx"]

_SCENE_RESERVED_NAMES = frozenset(
    {
        "entities",
        "env_spacing",
        "extent",
        "num_envs",
        "sensors",
        "spec_fn",
        "terrain",
    }
)


def _validate_scene_names(spec: TaskSpec) -> None:
    spec.scene.validate_symbol_table(
        reserved_names=_SCENE_RESERVED_NAMES,
        reserved_context="MJLab SceneCfg fields",
    )


class MjlabCompileCtx(CompileCtx):
    """Compilation context carrying mjlab's noise classes."""

    def noise(self, noise: NoiseSpec | None) -> Any:
        if noise is None:
            return None
        from mjlab.utils.noise import GaussianNoiseCfg, UniformNoiseCfg

        if noise.kind == "uniform":
            return UniformNoiseCfg(
                n_min=noise.lo, n_max=noise.hi, operation=noise.operation
            )
        if noise.kind == "gaussian":
            return GaussianNoiseCfg(
                mean=noise.lo, std=noise.hi, operation=noise.operation
            )
        raise NotImplementedError(f"mjlab has no noise model for kind {noise.kind!r}.")


def _observation_groups(compiled: Mapping[str, Any]) -> dict[str, Any]:
    """Observation groups in mjlab's shape.

    mjlab's group holds its terms in a dict field, so declaration order is the dict's insertion
    order and the group settings cannot collide with a term named after one of them. Isaac Lab keeps
    terms as attributes beside those settings and skips them by name. Same resulting vector, and the
    difference is confined to these few lines on each side.
    """
    from mjlab.managers import ObservationGroupCfg

    return {
        name: ObservationGroupCfg(
            terms=dict(group["terms"]), **observation_group_settings(group)
        )
        for name, group in compiled.items()
    }


def _rewards(
    compiled: Mapping[str, Mapping[str, Any]], omit: tuple[str, ...] = ()
) -> dict[str, Any]:
    return flatten_reward_groups(compiled, omit=omit)


def _record_pinhole_camera_semantics(profile: dict[str, Any], spec: TaskSpec) -> None:
    """Expose mjlab-only camera hit semantics beside the checkpoint manifest."""
    from .camera import pinhole_camera_effective_semantics

    semantics = {
        sensor.name: pinhole_camera_effective_semantics(sensor, profile)
        for sensor in spec.scene.ray_casters
        if sensor.pattern.kind == "pinhole"
    }
    if semantics:
        profile["pinhole_camera_semantics"] = semantics


class MjlabAdapter:
    """Compiles a :class:`TaskSpec` into an mjlab ``ManagerBasedRlEnvCfg``."""

    name = "mjlab"
    SUPPORTED_VERSIONS = "==1.5.0"
    RUNTIME_VERSIONS: ClassVar[dict[str, str]] = {
        "mujoco": "==3.10.0",
        "mujoco-warp": "==3.10.0.1",
        "warp-lang": "==1.14.0",
    }

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Only ``--device``. mjlab is an ordinary import, which is most of why it is quick to use.

        ``--device`` is contributed by the adapter rather than by the launcher because Isaac Sim's
        ``AppLauncher`` insists on declaring that flag itself and refuses a parser that already
        has one. Both engines therefore spell it the same way and mean the same thing, but each
        engine registers its own.
        """
        parser.add_argument(
            "--device",
            type=str,
            default="cuda:0",
            help="Device to simulate and learn on.",
        )

    @staticmethod
    def bootstrap(args: argparse.Namespace) -> object | None:
        """No simulator to start, but the torch backend has to match the reference stack.

        This used to return without touching torch, on the stated grounds that InstinctMJ leaves
        torch at its defaults. It does not: its training script calls
        ``mjlab.utils.torch.configure_torch_backends()``, whose default is ``allow_tf32=True``.
        The claim was wrong and a passing test pinned it, so mjlab trained with matmul TF32 off
        (the torch default) against a reference that trains with it on -- and against our own
        Isaac side, which sets it. Every matmul in the policy and discriminator update ran on
        different arithmetic than either reference, and every two-engine comparison carried that
        asymmetry.

        Calling the reference's own helper, rather than assigning the flags here, keeps this
        engine on whatever stack mjlab decides that helper means.
        """
        require_supported_version(
            "mjlab", MjlabAdapter.SUPPORTED_VERSIONS, engine=MjlabAdapter.name
        )
        for distribution, supported in MjlabAdapter.RUNTIME_VERSIONS.items():
            require_supported_version(distribution, supported, engine=MjlabAdapter.name)

        from mjlab.utils.torch import configure_torch_backends

        configure_torch_backends()
        return None

    @staticmethod
    def wrap_for_rl(env: Any) -> Any:
        """Wrap with the wrapper ported from InstinctMJ, which is how mjlab training was run."""
        from .rl_wrapper import MjlabVecEnvWrapper

        return MjlabVecEnvWrapper(env)

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
        merged = dict(PROFILE_DEFAULTS)
        merged.update(spec.sim.profiles.get(self.name, {}))
        return merged

    def compile(
        self, spec: TaskSpec, *, num_envs: int, device: str, strict: bool = False
    ) -> CompiledTask:
        spec.validate_for_engine(self.name)
        _validate_scene_names(spec)

        from mjlab.envs import ManagerBasedRlEnvCfg
        from mjlab.sim import MujocoCfg, SimulationCfg

        profile = self.profile(spec)
        _record_pinhole_camera_semantics(profile, spec)
        resolution = Resolution(
            engine=self.name,
            task_id=spec.task_id,
            profile=profile,
            engine_extras_used=tuple(sorted(spec.engine_extras.get(self.name, {}))),
            strict=strict,
        )
        from instinctlab_engine.lifecycle.runtime import lifecycle_manifest

        resolution.lifecycle = lifecycle_manifest(spec)
        ctx = MjlabCompileCtx(
            engine=self.name,
            spec=spec,
            resolution=resolution,
            profile=profile,
            num_envs=num_envs,
            device=device,
            strict=strict,
        )
        mdp = compile_mdp(spec.mdp, ctx, TERMS)
        omitted_rewards = tuple(profile.get("omit_rewards", ()))
        record_reward_omissions(resolution, mdp["rewards"], omitted_rewards)

        mj_overrides = spec.sim.profile_for(self.name)
        mujoco_kwargs: dict[str, Any] = {
            "timestep": spec.sim.physics_dt,
            "solver": profile["solver"],
            "iterations": profile["iterations"],
            "ls_iterations": profile["ls_iterations"],
            "ccd_iterations": profile["ccd_iterations"],
        }
        if "jacobian" in mj_overrides:
            mujoco_kwargs["jacobian"] = mj_overrides["jacobian"]
        sim_kwargs: dict[str, Any] = {"mujoco": MujocoCfg(**mujoco_kwargs)}
        if spec.scene.terrain.kind == "generator":
            # InstinctMJ / mjlab's own G1 rough raises both; the plane defaults underflow a
            # generated grid and drop contacts without raising.
            sim_kwargs["nconmax"] = mj_overrides.get("nconmax", 70)
            sim_kwargs["njmax"] = mj_overrides.get("njmax", profile["njmax"])
            sim_kwargs["contact_sensor_maxmatch"] = mj_overrides.get(
                "contact_sensor_maxmatch", 500
            )
        elif spec.scene.terrain.kind == "rough":
            # Per-world allocations, multiplied by the environment count.
            # ``d.nacon`` is one global counter; per-world is nacon/nworld.
            # The shared 0.05 / mesh-box recipe has 271 host contacts at rest;
            # 512 / 1536 leaves construction and step headroom without putting
            # a MuJoCo capacity into the engine-neutral terrain declaration.
            sim_kwargs["nconmax"] = mj_overrides.get("nconmax", 512)
            sim_kwargs["njmax"] = mj_overrides.get("njmax", 1536)
            sim_kwargs["contact_sensor_maxmatch"] = mj_overrides.get(
                "contact_sensor_maxmatch", 128
            )
        else:
            sim_kwargs["njmax"] = mj_overrides.get("njmax", profile["njmax"])

        # Task-native capacities take precedence for every terrain kind.  In particular,
        # motion_matched is neither the generic "generator" nor "rough" spelling;
        # previously its InstinctMJ overrides were silently discarded here.
        for field in ("nconmax", "njmax", "contact_sensor_maxmatch"):
            if field in mj_overrides:
                sim_kwargs[field] = mj_overrides[field]
        from .env import TerrainAwareRlEnv

        env_cfg = ManagerBasedRlEnvCfg(
            scene=build_scene(
                spec.scene,
                spec.robot,
                profile,
                num_envs=num_envs,
                sensor_period=spec.sim.physics_dt,
            ),
            observations=_observation_groups(mdp["observations"]),
            actions=mdp["actions"],
            rewards=_rewards(mdp["rewards"], omitted_rewards),
            terminations=mdp["terminations"],
            events=mdp["events"],
            commands=mdp["commands"],
            curriculum=mdp["curriculum"],
            decimation=spec.sim.decimation,
            episode_length_s=spec.sim.episode_length_s,
            is_finite_horizon=spec.sim.is_finite_horizon,
            scale_rewards_by_dt=spec.sim.scale_rewards_by_dt,
            sim=SimulationCfg(**sim_kwargs),
        )

        def make_env() -> Any:
            from instinctlab_engine.motion_reference import (
                bind_motion_reference_origins,
            )

            env = TerrainAwareRlEnv(cfg=env_cfg, device=device)
            bind_motion_reference_origins(env.scene, spec.scene.motion_references)
            from instinctlab_engine.lifecycle import attach_lifecycle

            attach_lifecycle(env, spec, engine=self.name)
            return env

        resolution.capture_plugin_provenance(asset_id=spec.robot.asset_id)
        return CompiledTask(
            env_factory=make_env,
            env_cls=TerrainAwareRlEnv,
            env_cfg=env_cfg,
            resolution=resolution,
            agent_factory=lambda: spec.agent.resolve()(
                **spec.agent.resolved_overrides(self.name)
            ),
        )

    def contract_report(self, spec: TaskSpec) -> dict[str, Any]:
        _validate_scene_names(spec)
        profile = self.profile(spec)
        return contract_report(
            spec,
            engine=self.name,
            registry=TERMS,
            capabilities=self.capabilities(),
            omitted_rewards=tuple(profile.get("omit_rewards", ())),
        )

    @staticmethod
    def finalize_process(exit_code: int) -> int:
        return exit_code
