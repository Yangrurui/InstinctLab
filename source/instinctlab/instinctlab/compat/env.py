"""Reaching into the environment object the same way under either engine.

This module is deliberately small, and the reason is worth recording, because the natural
expectation is the opposite. The two environment classes were written independently, yet their
public surface has converged almost exactly: ``num_envs``, ``device``, ``physics_dt``, ``step_dt``,
``max_episode_length``, ``max_episode_length_s``, ``episode_length_buf``, ``scene``, ``cfg``,
``common_step_counter``, ``extras`` and all seven managers are spelled identically and mean the
same thing. ``scene[name]`` and ``scene.sensors[name]`` both work on both. A term that reads any of
those needs nothing from this module, and should not be routed through it -- indirection that buys
nothing is the mistake this project already made once with the retired ``EntityView``.

``test_compat_env.py`` pins that convergence as an executable claim rather than a comment, so an
engine upgrade that breaks it fails a test instead of quietly breaking terms.

Three things did not converge:

**An absent command manager fails differently.** mjlab substitutes a ``NullCommandManager`` when
the task declares no commands, and its ``get_command`` returns ``None`` for every name. Isaac Lab
always installs a real ``CommandManager``, so the same call raises ``KeyError``. The mjlab branch
is the dangerous one: ``None`` is not an error until something subscripts it several frames later,
and a velocity-tracking reward that silently receives ``None`` reports a shape error far from the
misconfiguration that caused it. :func:`get_command` gives both engines the same loud failure.

**The class names are spelled differently** -- ``ManagerBasedRLEnv`` against
``ManagerBasedRlEnv``. Only the capital L differs, which is exactly the kind of difference that
survives review. A ported term should annotate :class:`RlEnv` instead; :data:`ENV_TYPE_NAMES` keeps
the native spellings for the migration codemod to recognise.

**The physics timestep lives at a different config path** -- ``cfg.sim.dt`` against
``cfg.sim.mujoco.timestep``. This is a compile-time concern belonging to the engine adapters, not
a term-time one, since ``env.physics_dt`` already reads correctly on both. It is recorded here
because the config path is where the engines actually differ, and future readers will look for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .denylist import PortabilityError

if TYPE_CHECKING:
    import torch

__all__ = [
    "ENV_TYPE_NAMES",
    "PHYSICS_DT_CFG_PATH",
    "RAW_ACTION_ATTR",
    "RlEnv",
    "command_names",
    "env_engine",
    "get_command",
    "has_command",
    "raw_action",
]

ENV_TYPE_NAMES = {"isaacsim": "ManagerBasedRLEnv", "mjlab": "ManagerBasedRlEnv"}
"""Native environment class name per engine. Differs only in the capitalisation of ``RL``."""

PHYSICS_DT_CFG_PATH = {"isaacsim": ("sim", "dt"), "mjlab": ("sim", "mujoco", "timestep")}
"""Where the physics timestep lives on each engine's config. Adapters write it; terms read
``env.physics_dt``, which is spelled the same on both."""

RAW_ACTION_ATTR = {"isaacsim": "raw_actions", "mjlab": "raw_action"}
"""An action term's untransformed input, differing by one character between the engines."""

_ENGINE_BY_ROOT_PACKAGE = {"isaaclab": "isaacsim", "mjlab": "mjlab"}


@runtime_checkable
class RlEnv(Protocol):
    """The environment surface a portable term may rely on.

    Every member here is spelled and behaves the same on both engines -- that is the entry
    requirement, not an aspiration. Annotate portable terms with this instead of importing an
    engine's environment class, which would pull the whole SDK into a module that has no other
    need of it.
    """

    cfg: Any
    scene: Any
    num_envs: int
    device: str
    episode_length_buf: torch.Tensor
    common_step_counter: int
    extras: dict[str, Any]

    action_manager: Any
    command_manager: Any
    curriculum_manager: Any
    event_manager: Any
    observation_manager: Any
    reward_manager: Any
    termination_manager: Any

    @property
    def physics_dt(self) -> float: ...

    @property
    def step_dt(self) -> float: ...

    @property
    def max_episode_length(self) -> int: ...

    @property
    def max_episode_length_s(self) -> float: ...


def env_engine(env: Any) -> str:
    """Which engine this environment belongs to.

    Decided from the defining module of the environment class and its bases, so that a project's
    own subclass of either engine's environment still resolves. No engine is imported to answer
    the question.

    Raises:
        PortabilityError: The class descends from neither engine's environment.
    """
    for klass in type(env).__mro__:
        engine = _ENGINE_BY_ROOT_PACKAGE.get(getattr(klass, "__module__", "").split(".")[0])
        if engine is not None:
            return engine
    raise PortabilityError(
        f"{type(env).__name__} descends from neither {' nor '.join(_ENGINE_BY_ROOT_PACKAGE)}, so its "
        "engine is unknown. A new engine must be registered in compat.env."
    )


def command_names(env: Any) -> list[str]:
    """The commands this environment actually provides, empty when it provides none.

    Reads ``active_terms``, which both engines expose -- including mjlab's null manager, where it
    is the empty list. This is the check to use before configuring anything conditional on a
    command existing.
    """
    return list(getattr(env.command_manager, "active_terms", []))


def has_command(env: Any, name: str) -> bool:
    """Whether ``name`` is a command of this environment."""
    return name in command_names(env)


def raw_action(env: Any, action_name: str) -> torch.Tensor:
    """The named action term's untransformed input.

    Resolved by duck typing rather than by engine, since the difference is one character --
    ``raw_actions`` against ``raw_action`` -- and a term that guessed wrong would raise an
    ``AttributeError`` naming an attribute that looks correct.
    """
    term = env.action_manager.get_term(action_name)
    for attr in RAW_ACTION_ATTR.values():
        value = getattr(term, attr, None)
        if value is not None:
            return value
    raise PortabilityError(
        f"Action term {action_name!r} ({type(term).__name__}) exposes neither "
        f"{' nor '.join(RAW_ACTION_ATTR.values())}. A new engine must be registered in compat.env."
    )


def get_command(env: Any, name: str) -> torch.Tensor:
    """The command tensor named ``name``, failing the same way on either engine.

    Both engines' native calls report a missing command badly, and each badly in its own way:
    Isaac Lab raises ``KeyError(name)`` with no indication of what was available, and mjlab's null
    manager returns ``None``, which is not an error at all until the caller subscripts it. This
    raises :class:`PortabilityError` naming the command and listing what the environment has.

    Raises:
        PortabilityError: No such command, or the environment has no commands at all.
    """
    available = command_names(env)
    if name not in available:
        have = ", ".join(available) if available else "none -- the task declares no commands"
        raise PortabilityError(f"Environment has no command {name!r}. Available: {have}.")
    command = env.command_manager.get_command(name)
    if command is None:
        raise PortabilityError(
            f"Command {name!r} is listed as active but its manager returned None. The environment "
            "is inconsistent; this is an engine bug rather than a task misconfiguration."
        )
    return command
