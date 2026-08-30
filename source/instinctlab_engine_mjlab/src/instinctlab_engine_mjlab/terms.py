"""MJLab's general term registry, which is also its capability matrix.

Reading this beside ``engines/isaacsim/terms.py`` is the clearest statement of what the ``N + M``
structure bought. Portable terms use one wrapper per family; native terms keep only the SDK
translation that cannot live in a task.

Domain randomisation is where they differ most, and not superficially. Isaac Lab randomises through
event functions taking distribution parameters; mjlab randomises through a declarative ``dr`` module
whose primitives are named after the model fields they perturb. Friction is the sharpest case:
PhysX has a static and a dynamic coefficient plus restitution, drawn from a bucket pool, while
MuJoCo has one sliding coefficient and no per-geom restitution at all. Neither is emulating the
other; each takes its own scheme from its own profile.

Native event and randomization lowering is installed from ``event_terms.py``.
Keeping that adapter surface separate makes this file describe the manager term
families rather than becoming a catalog of MuJoCo model mutations.
"""

from __future__ import annotations

from typing import Any

from instinctlab_engine.compile import joint_position_target
from instinctlab_engine.actuators import JOINT_POSITION_COMMAND
from instinctlab_engine.registry import TermRegistry

from .event_terms import register_event_terms

TERMS = TermRegistry("mjlab")


def _cfgs() -> dict[str, Any]:
    from mjlab.managers import (
        CurriculumTermCfg,
        ObservationTermCfg,
        RewardTermCfg,
        TerminationTermCfg,
    )

    return {
        "obs": ObservationTermCfg,
        "reward": RewardTermCfg,
        "done": TerminationTermCfg,
        "curriculum": CurriculumTermCfg,
    }


@TERMS.portable("observation")
def _observation(spec, ctx):
    return _cfgs()["obs"](
        func=spec.func,
        params=ctx.params(spec),
        noise=ctx.noise(spec.noise),
        scale=spec.scale,
        clip=spec.clip,
        history_length=spec.history_length,
    )


@TERMS.portable("reward")
def _reward(spec, ctx):
    return _cfgs()["reward"](
        func=spec.func, weight=spec.weight, params=ctx.params(spec)
    )


@TERMS.portable("termination")
def _termination(spec, ctx):
    return _cfgs()["done"](
        func=spec.func, time_out=spec.time_out, params=ctx.params(spec)
    )


@TERMS.portable("curriculum")
def _curriculum(spec, ctx):
    return _cfgs()["curriculum"](func=spec.func, params=ctx.params(spec))


@TERMS.portable("command")
def _portable_command(spec, ctx):
    """Adapt a task-owned command algorithm to MJLab's manager base."""
    from dataclasses import dataclass

    from mjlab.managers import CommandTerm, CommandTermCfg

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

        def _debug_vis_impl(self, visualizer):
            draw = getattr(self._impl, "_debug_vis_impl", None)
            if callable(draw):
                draw(visualizer)

        def __getattr__(self, name):
            task_command = self.__dict__.get("_impl")
            if task_command is None:
                raise AttributeError(name)
            return getattr(task_command, name)

    WrappedCommand.__name__ = implementation.__name__
    WrappedCommand.__qualname__ = implementation.__qualname__

    @dataclass(kw_only=True)
    class PortableCommandCfg(CommandTermCfg):
        def build(self, env):
            return WrappedCommand(self, env)

    return PortableCommandCfg(
        resampling_time_range=params["resampling_time_range"],
        debug_vis=bool(params.get("debug_vis", False)),
    )


@TERMS.action(
    "joint_position",
    requires_actuator=(JOINT_POSITION_COMMAND,),
)
def _joint_position(spec, ctx):
    """Position-target action.

    mjlab drives **actuators**, Isaac Lab drives **joints**. For a robot whose actuators are built
    one per joint they name the same things, but the selector kind differs, and this is the point
    where decision D1 is applied on this engine: the canonical depth-first joint order is passed
    explicitly with ``preserve_order`` rather than relying on ``.*``, whose expansion follows the
    model file's own order.
    """
    from .actions import PreservingJointPositionActionCfg

    target = joint_position_target(spec, ctx)
    params = ctx.params(spec)
    return PreservingJointPositionActionCfg(
        entity_name=target.entity,
        actuator_names=tuple(target.joints),
        preserve_order=target.preserve_order,
        scale=params["scale"],
        use_default_offset=params["use_default_offset"],
    )


register_event_terms(TERMS)
