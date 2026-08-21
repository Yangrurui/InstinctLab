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

from instinctlab.engines.base import CompiledTask, Resolution
from instinctlab.engines.compile import CompileCtx, compile_mdp, observation_group_settings
from instinctlab.sim.capabilities import CapabilitySet
from instinctlab.spec.mdp import NoiseSpec
from instinctlab.spec.task import TaskSpec

from .scene import PROFILE_DEFAULTS, build_scene
from .terms import TERMS

__all__ = ["IsaacSimAdapter", "IsaacSimCompileCtx"]


def _play_native(env: Any, policy: Any) -> None:
    """Step the policy in Isaac's own viewport, if the app was launched with one."""
    obs = env.get_observations()
    try:
        while True:
            result = env.step(policy(obs))
            obs = result[0]
    except KeyboardInterrupt:
        return


def _mjlab_play_env(spec: Any, *, num_envs: int, device: str, strict: bool) -> Any:
    """Compile the same task on mjlab so ``ViserPlayViewer`` has a MuJoCo ``Simulation``."""
    from instinctlab.engines import adapter as engine_adapter
    from instinctlab.play.env import PlayEnv

    other = engine_adapter("mjlab")
    compiled = other.compile(spec, num_envs=num_envs, device=device, strict=strict)
    groups = compiled.env_cfg.observations
    items = groups.values() if isinstance(groups, dict) else vars(groups).values()
    for group in items:
        if hasattr(group, "enable_corruption"):
            group.enable_corruption = False
    return PlayEnv(other.wrap_for_rl(compiled.make_env()))


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

    Groups exist in the declaration so that a task can say which rewards belong together, and
    :class:`~instinctlab.spec.mdp.MdpSpec` already guarantees the names stay unique across them.
    Flattening therefore loses the grouping and nothing else.
    """
    return _container({name: term for group in compiled.values() for name, term in group.items()})


class IsaacSimAdapter:
    """Compiles a :class:`TaskSpec` into an Isaac Lab ``ManagerBasedRLEnvCfg``."""

    name = "isaacsim"
    SUPPORTED_VERSIONS = ">=0.40,<0.50"

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
        from instinctlab.utils.contact_overflow import attach_overflow_guard, check_contact_overflow
        from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

        check_contact_overflow(env, phase="construction")
        return InstinctRlVecEnvWrapper(attach_overflow_guard(env))

    def capabilities(self) -> CapabilitySet:
        return TERMS.capabilities()

    def profile(self, spec: TaskSpec) -> dict[str, Any]:
        """This engine's solver settings, the task's overrides applied over the defaults."""
        merged = dict(PROFILE_DEFAULTS)
        merged.update(spec.sim.profiles.get(self.name, {}))
        return merged

    def compile(self, spec: TaskSpec, *, num_envs: int, device: str, strict: bool = False) -> CompiledTask:
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
        if spec.scene.terrain.kind in {"generator", "rough"}:
            # Mesh tiles create far more contact patches than a plane; Isaac Lab's own rough
            # locomotion raises this, and leaving the default silently drops contacts.
            sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
            # Main's ParkourEnvCfg.__post_init__ (parkour_env_cfg.py:924) raises this
            # from Isaac Lab PhysxCfg's default 2**26. GPU PhysX cannot grow the
            # collision stack: overflow logs a PhysX error, drops contacts, and the
            # step still succeeds. 2**29 is 512 MiB of pinned host memory.
            sim.physx.gpu_collision_stack_size = 2**29
            sim.physics_material = scene.terrain.physics_material
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
        return CompiledTask(
            env_cls=InstinctManagerBasedRLEnv.wrap(ManagerBasedRLEnv),
            env_cfg=env_cfg,
            resolution=resolution,
            agent_factory=lambda: spec.agent.resolve()(**spec.agent.resolved_overrides(self.name)),
        )

    def play(
        self,
        env: Any,
        policy: Any,
        *,
        viewer: str,
        robot: Any,
        spec: Any | None = None,
        port: int = 8080,
        reload_policy: Any | None = None,
        checkpoint_dir: Any | None = None,
        strict: bool = False,
    ) -> None:
        """Isaac has no Viser backend; ``viser`` plays the policy in mjlab's ``ViserPlayViewer``.

        That is the same viewer mjlab training uses, and the same path the earlier Isaac play
        script took. The policy is unchanged -- both engines already share the catalog joint order.
        """
        del robot
        if viewer == "viser":
            if spec is None:
                raise ValueError("viser playback needs the task spec")
            from instinctlab.play.viser import play_with_viser

            play_env = _mjlab_play_env(spec, num_envs=env.num_envs, device=str(env.device), strict=strict)
            print(
                "[INFO] Isaac Sim has no Viser backend; playing this checkpoint in mjlab's ViserPlayViewer",
                flush=True,
            )
            play_with_viser(
                play_env,
                policy,
                port=port,
                reload_policy=reload_policy,
                checkpoint_dir=checkpoint_dir,
            )
            play_env.close()
            return
        if viewer == "native":
            _play_native(env, policy)
            return
        raise ValueError(f"unsupported viewer {viewer!r}")

    def contract_report(self, spec: TaskSpec) -> dict[str, Any]:
        """What this engine would and would not provide, without importing it.

        Answerable on a machine with no Isaac Sim because the registry's keys are declared at import
        time and only the builder bodies touch ``isaaclab``. That is what makes it possible to check
        every task against every engine in one CI job.
        """
        missing: dict[str, str] = {}
        for key, term in spec.mdp.terms().items():
            family = key.split("/", 1)[0]
            if term.is_portable or TERMS.lookup(family, term.kind) is not None:
                continue
            emulated = TERMS.lookup_emulation(family, term.kind) is not None
            missing[key] = "emulated" if emulated else f"unsupported kind {term.kind!r}"
        return {
            "engine": self.name,
            "task_id": spec.task_id,
            "capabilities": sorted(self.capabilities().values),
            "missing": missing,
            "engine_extras_used": sorted(spec.engine_extras.get(self.name, {})),
        }
