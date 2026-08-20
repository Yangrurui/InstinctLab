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
from typing import Any

from instinctlab.engines.base import CompiledTask, Resolution
from instinctlab.engines.compile import CompileCtx, compile_mdp, observation_group_settings
from instinctlab.sim.capabilities import CapabilitySet
from instinctlab.spec.mdp import NoiseSpec
from instinctlab.spec.task import TaskSpec

from .scene import PROFILE_DEFAULTS, build_scene
from .terms import TERMS

__all__ = ["MjlabAdapter", "MjlabCompileCtx"]


class MjlabCompileCtx(CompileCtx):
    """Compilation context carrying mjlab's noise classes."""

    def noise(self, noise: NoiseSpec | None) -> Any:
        if noise is None:
            return None
        from mjlab.utils.noise import GaussianNoiseCfg, UniformNoiseCfg

        if noise.kind == "uniform":
            return UniformNoiseCfg(n_min=noise.lo, n_max=noise.hi, operation=noise.operation)
        if noise.kind == "gaussian":
            return GaussianNoiseCfg(mean=noise.lo, std=noise.hi, operation=noise.operation)
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
        name: ObservationGroupCfg(terms=dict(group["terms"]), **observation_group_settings(group))
        for name, group in compiled.items()
    }


def _rewards(compiled: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {name: term for group in compiled.values() for name, term in group.items()}


class MjlabAdapter:
    """Compiles a :class:`TaskSpec` into an mjlab ``ManagerBasedRlEnvCfg``."""

    name = "mjlab"
    SUPPORTED_VERSIONS = ">=0.1,<0.3"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Only ``--device``. mjlab is an ordinary import, which is most of why it is quick to use.

        ``--device`` is contributed by the adapter rather than by the launcher because Isaac Sim's
        ``AppLauncher`` insists on declaring that flag itself and refuses a parser that already
        has one. Both engines therefore spell it the same way and mean the same thing, but each
        engine registers its own.
        """
        parser.add_argument("--device", type=str, default="cuda:0", help="Device to simulate and learn on.")

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
        from mjlab.utils.torch import configure_torch_backends

        configure_torch_backends()
        return None

    @staticmethod
    def wrap_for_rl(env: Any) -> Any:
        """Wrap with the wrapper ported from InstinctMJ, which is how mjlab training was run."""
        from instinctlab.utils.wrappers.instinct_rl.mjlab_vecenv_wrapper import MjlabVecEnvWrapper

        return MjlabVecEnvWrapper(env)

    def capabilities(self) -> CapabilitySet:
        return TERMS.capabilities()

    def profile(self, spec: TaskSpec) -> dict[str, Any]:
        merged = dict(PROFILE_DEFAULTS)
        merged.update(spec.sim.profiles.get(self.name, {}))
        return merged

    def compile(self, spec: TaskSpec, *, num_envs: int, device: str, strict: bool = False) -> CompiledTask:
        from mjlab.envs import ManagerBasedRlEnvCfg
        from mjlab.sim import MujocoCfg, SimulationCfg

        profile = self.profile(spec)
        resolution = Resolution(
            engine=self.name,
            task_id=spec.task_id,
            profile=profile,
            engine_extras_used=tuple(sorted(spec.engine_extras.get(self.name, {}))),
            strict=strict,
        )
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

        sim_kwargs: dict[str, Any] = {
            "njmax": profile["njmax"],
            "mujoco": MujocoCfg(
                timestep=spec.sim.physics_dt,
                solver=profile["solver"],
                iterations=profile["iterations"],
                ls_iterations=profile["ls_iterations"],
                ccd_iterations=profile["ccd_iterations"],
            ),
        }
        if spec.scene.terrain.kind == "generator":
            # InstinctMJ / mjlab's own G1 rough raises both; the plane defaults underflow a
            # generated grid and drop contacts without raising.
            sim_kwargs["nconmax"] = 70
            sim_kwargs["contact_sensor_maxmatch"] = 500
        elif spec.scene.terrain.kind == "rough":
            # Per-world allocations, multiplied by the environment count.
            # ``d.nacon`` is one global counter; per-world is nacon/nworld.
            # Resting default pose is the measured peak, host-copied to every
            # world: host ncon=164, host/GPU nefc=691 (64 and 256 envs, and the
            # 2026-08-20 L7 run at 256). After reset / flail both drop
            # (nacon ~2/world, flail nefc_max 162 at 16–32 envs). nconmax=256
            # is 64% at rest. njmax=700 was 98.7% at rest; put_data refuses
            # njmax < host.nefc (691). 768 is ~11% headroom. InstinctMJ still
            # writes njmax=700 (10-col grid). The leftover ``generator`` path
            # above is Isaac Lab's six-tile recipe and needs its own numbers.
            sim_kwargs["nconmax"] = 256
            sim_kwargs["njmax"] = 768
            sim_kwargs["contact_sensor_maxmatch"] = 128
        from .env import TerrainAwareRlEnv

        env_cfg = ManagerBasedRlEnvCfg(
            scene=build_scene(spec.scene, spec.robot, profile, num_envs=num_envs),
            observations=_observation_groups(mdp["observations"]),
            actions=mdp["actions"],
            rewards=_rewards(mdp["rewards"]),
            terminations=mdp["terminations"],
            events=mdp["events"],
            commands=mdp["commands"],
            curriculum=mdp["curriculum"],
            decimation=spec.sim.decimation,
            episode_length_s=spec.sim.episode_length_s,
            is_finite_horizon=spec.sim.is_finite_horizon,
            sim=SimulationCfg(**sim_kwargs),
        )
        return CompiledTask(
            env_factory=lambda: TerrainAwareRlEnv(cfg=env_cfg, device=device),
            env_cls=TerrainAwareRlEnv,
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
        """mjlab already has Viser and a native MuJoCo viewer; use those."""
        del robot, spec, strict
        if viewer == "viser":
            from instinctlab.play.viser import play_with_viser

            play_with_viser(
                env,
                policy,
                port=port,
                reload_policy=reload_policy,
                checkpoint_dir=checkpoint_dir,
            )
            return
        if viewer == "native":
            from mjlab.viewer import NativeMujocoViewer

            NativeMujocoViewer(env, policy).run()
            return
        raise ValueError(f"unsupported viewer {viewer!r}")

    def contract_report(self, spec: TaskSpec) -> dict[str, Any]:
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
