"""What a backend must provide, and what a compilation must account for.

An engine backend does two jobs. It **bootstraps** -- Isaac Sim's ``AppLauncher`` has to run before
anything imports torch, mjlab has nothing to do -- and it **compiles** a :class:`TaskSpec` into that
engine's own environment config. Everything else about the engine stays untouched underneath;
nothing here wraps or replaces the native stack.

The part worth dwelling on is :class:`Resolution`. Skipping what an engine cannot do is only
tolerable if every skip is visible, because the failure mode is not a crash. A task whose friction
randomisation was silently dropped still trains, still converges, and produces a policy that is
simply less robust than the one its config describes -- and nothing in the logs distinguishes it
from a healthy run six weeks later when the two checkpoints are being compared. So a compilation
returns a full account of what happened to every term, printed once as a table at startup and
written beside the checkpoint.

This module imports no engine. Backends live in ``engines/<name>/`` and are the only place an
engine SDK appears.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from instinctlab.sim.capabilities import CapabilitySet
from instinctlab.spec.task import TaskSpec

__all__ = [
    "CompiledTask",
    "EngineAdapter",
    "Resolution",
    "UnsupportedTerm",
]


class UnsupportedTerm(RuntimeError):
    """A REQUIRED term has no implementation on the engine being compiled for.

    Carries the pieces separately so a caller can group these rather than printing one line per
    term, which for a task arriving from another engine is often a dozen at once.
    """

    def __init__(self, key: str, engine: str, kind: str | None, *, detail: str = ""):
        self.key, self.engine, self.kind = key, engine, kind
        named = f"term kind {kind!r}" if kind else "a portable term"
        super().__init__(f"Engine {engine!r} cannot provide {key!r} ({named}). {detail}".strip())


@dataclass
class Resolution:
    """The full account of one compilation.

    Args:
        engine: Engine compiled for.
        task_id: Task compiled.
        resolved: ``family/name`` -> the native callable that will run it.
        skipped: ``family/name`` -> why it is not running.
        emulated: ``family/name`` -> the stand-in that replaced it.
        profile: Solver settings actually used, adapter defaults included.
        engine_extras_used: Escape-hatch keys the task reached for. Non-empty means the compiled
            task is not portable, and the manifest says so.
        strict: Whether OPTIONAL was promoted to REQUIRED for this compilation.
    """

    engine: str
    task_id: str
    resolved: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    emulated: dict[str, str] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    engine_extras_used: tuple[str, ...] = ()
    strict: bool = False

    @property
    def is_clean(self) -> bool:
        """Whether every term the task declared is running its own implementation."""
        return not self.skipped and not self.emulated

    def summary_table(self) -> str:
        """One table, printed once at startup.

        A table rather than warnings: warnings scattered through a boot log get read as noise, and
        the number of them is the thing that matters. When nothing was skipped this says so in one
        line, so that the absence of a report is never mistaken for a clean compilation.
        """
        head = f"[{self.engine}] {self.task_id}: {len(self.resolved)} terms resolved"
        if self.strict:
            head += " (strict: optional terms were required)"
        if self.is_clean and not self.engine_extras_used:
            return head + ", none skipped or emulated."

        width = max((len(k) for k in (*self.skipped, *self.emulated)), default=0)
        lines = [head]
        for title, entries in (("skipped", self.skipped), ("emulated", self.emulated)):
            if entries:
                lines.append(f"  {title}:")
                lines += [f"    {key:<{width}}  {reason}" for key, reason in sorted(entries.items())]
        if self.engine_extras_used:
            lines.append(f"  engine_extras used (task is not portable): {', '.join(self.engine_extras_used)}")
        return "\n".join(lines)

    def manifest(self) -> dict[str, Any]:
        """The part of this that belongs beside the checkpoint.

        Without it, "why is this policy worse than that one" is unanswerable after the fact: the
        two runs may have been compiled from the same task file and still have optimised different
        objectives, and the difference exists only here.
        """
        return {
            "engine": self.engine,
            "task_id": self.task_id,
            "resolved": dict(sorted(self.resolved.items())),
            "skipped": dict(sorted(self.skipped.items())),
            "emulated": dict(sorted(self.emulated.items())),
            "profile": dict(sorted(self.profile.items())),
            "engine_extras_used": list(self.engine_extras_used),
            "strict": self.strict,
            "portable": self.is_clean and not self.engine_extras_used,
        }


@dataclass
class CompiledTask:
    """What a backend hands back: a native environment, ready to construct.

    ``env_cls`` and ``env_cfg`` are the engine's own types, not wrappers. The task was declared
    portably; what runs is native.
    """

    env_cls: type
    env_cfg: Any
    agent_cfg: Any
    resolution: Resolution


@runtime_checkable
class EngineAdapter(Protocol):
    """The whole of what a new engine has to implement.

    Adding an engine means one of these plus a term registry. Nothing in ``spec/``, ``mdp/`` or any
    task file changes -- that is the property the ``N + M`` structure exists to buy, and the size
    of this protocol is the evidence for whether it was bought.
    """

    name: str
    SUPPORTED_VERSIONS: str
    """PEP 440 specifier for the engine versions this adapter is known to compile against, checked
    at bootstrap. An engine that changed a manager's attribute names between minor releases will
    otherwise fail somewhere far from the cause."""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Add this engine's launch flags. Isaac Sim needs many; mjlab needs almost none."""
        ...

    @staticmethod
    def bootstrap(args: argparse.Namespace) -> object:
        """Start whatever must exist before the engine can be imported.

        Ordering is the reason this is a separate step rather than part of ``compile``: Isaac Sim's
        ``AppLauncher`` has to run before torch is imported, so it cannot happen inside a function
        that already needs torch types in its signature.
        """
        ...

    def capabilities(self) -> CapabilitySet:
        """What this engine can do, derived from its term registry rather than declared by hand.

        Hand-maintained capability lists drift from the code that implements them -- this project
        already had a backend writing restitution while advertising that it could not.
        """
        ...

    def compile(self, spec: TaskSpec, *, num_envs: int, device: str, strict: bool = False) -> CompiledTask:
        """Compile a task into this engine's native environment config."""
        ...

    def contract_report(self, spec: TaskSpec) -> dict[str, Any]:
        """What would happen to this task here, without compiling or starting the engine.

        Lets a task be checked against every engine in CI, including ones not installed on the
        machine running it.
        """
        ...
