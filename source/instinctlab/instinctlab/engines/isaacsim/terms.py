"""Isaac Sim's general term registry, which is also its capability matrix.

Portable families get one builder each: the term already carries its function, and the builder's
whole job is to put it in the class Isaac Lab's manager expects. Per-engine families get a builder
per semantic name, and those names are what a task declares.

Native event and randomization lowering is installed from ``event_terms.py``.
Keeping that adapter surface separate makes this file describe the manager term
families rather than becoming a catalog of physics operations.
"""

from __future__ import annotations

import inspect
from typing import Any

from instinctlab.engines.compile import joint_position_target
from instinctlab.engines.registry import TermRegistry

from .event_terms import register_event_terms

TERMS = TermRegistry("isaacsim")


def _import_cfgs() -> dict[str, Any]:
    """Isaac Lab's config classes, imported after ``AppLauncher`` has started the app."""
    from isaaclab.managers import (
        ObservationTermCfg,
        RewardTermCfg,
        TerminationTermCfg,
    )

    return {
        "obs": ObservationTermCfg,
        "reward": RewardTermCfg,
        "done": TerminationTermCfg,
    }


def _as_isaac_manager_term(cls: type) -> type:
    """Isaac Lab refuses a class term that is not a ``ManagerTermBase`` subclass.

    mjlab instantiates any class with ``(cfg=, env=)``. The delay / history buffer
    for the depth image is portable and cannot inherit Isaac's type, so this
    wrapper is the Isaac-only tax -- not a second implementation of the pipeline.
    """
    from isaaclab.managers import ManagerTermBase

    call_sig = inspect.signature(cls.__call__)

    class Wrapped(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._impl = cls(cfg, env)

        def reset(self, env_ids=None):
            return self._impl.reset(env_ids)

        def __call__(self, *args, **kwargs):
            return self._impl(*args, **kwargs)

    Wrapped.__call__.__signature__ = call_sig  # type: ignore[attr-defined]
    Wrapped.__name__ = cls.__name__
    Wrapped.__qualname__ = cls.__qualname__
    return Wrapped


@TERMS.portable("observation")
def _observation(spec, ctx):
    cfgs = _import_cfgs()
    func = spec.func
    if inspect.isclass(func):
        func = _as_isaac_manager_term(func)
    return cfgs["obs"](
        func=func,
        params=ctx.params(spec),
        noise=ctx.noise(spec.noise),
        scale=spec.scale,
        clip=spec.clip,
        history_length=spec.history_length,
    )


@TERMS.portable("reward")
def _reward(spec, ctx):
    return _import_cfgs()["reward"](
        func=spec.func, weight=spec.weight, params=ctx.params(spec)
    )


@TERMS.portable("termination")
def _termination(spec, ctx):
    func = (
        _as_isaac_manager_term(spec.func) if inspect.isclass(spec.func) else spec.func
    )
    return _import_cfgs()["done"](
        func=func, time_out=spec.time_out, params=ctx.params(spec)
    )


@TERMS.portable("curriculum")
def _curriculum(spec, ctx):
    from isaaclab.managers import CurriculumTermCfg

    return CurriculumTermCfg(func=spec.func, params=ctx.params(spec))


@TERMS.portable("command")
def _portable_command(spec, ctx):
    """Adapt a task-owned command algorithm to Isaac Lab's manager base."""
    from isaaclab.managers import CommandTerm, CommandTermCfg
    from isaaclab.utils import configclass

    params = ctx.params(spec)
    if "resampling_time_range" not in params:
        raise ValueError("A portable command must declare resampling_time_range.")
    implementation = spec.func

    class WrappedCommand(CommandTerm):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._impl = implementation(env=env, params=params)
            self.metrics = self._impl.metrics

        @property
        def command(self):
            return self._impl.command

        def _update_metrics(self):
            return self._impl._update_metrics()

        def _resample_command(self, env_ids):
            return self._impl._resample_command(env_ids)

        def _update_command(self):
            return self._impl._update_command()

        def reset(self, env_ids=None):
            reset = getattr(self._impl, "reset", None)
            if callable(reset):
                return reset(env_ids)
            return super().reset(env_ids)

        def __getattr__(self, name):
            task_command = self.__dict__.get("_impl")
            if task_command is None:
                raise AttributeError(name)
            return getattr(task_command, name)

    WrappedCommand.__name__ = implementation.__name__
    WrappedCommand.__qualname__ = implementation.__qualname__

    @configclass
    class PortableCommandCfg(CommandTermCfg):
        class_type: type = WrappedCommand

    return PortableCommandCfg(
        resampling_time_range=params["resampling_time_range"],
        debug_vis=bool(params.get("debug_vis", False)),
    )


@TERMS.action("joint_position")
def _joint_position(spec, ctx):
    """Position-target action.

    Per-engine because the two drive joints through different objects, and because this is where
    decision D1 lands: the joint names come from the canonical depth-first order with
    ``preserve_order`` set, rather than from Isaac Lab's default ``[".*"]``, which resolves in
    PhysX's own breadth-first order. The resulting difference against the golden is a whitelist
    entry, and it is the reason the two engines' action vectors mean the same thing.
    """
    from isaaclab.envs.mdp import JointPositionActionCfg

    target = joint_position_target(spec, ctx)
    entity = ctx.entity(target)
    params = ctx.params(spec)
    return JointPositionActionCfg(
        asset_name=entity.name,
        joint_names=list(entity.joint_names),
        preserve_order=target.preserve_order,
        scale=params["scale"],
        use_default_offset=params["use_default_offset"],
    )


register_event_terms(TERMS)
