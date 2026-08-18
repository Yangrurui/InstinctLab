"""Turning a config tree into something two engines' outputs can be compared as.

The acceptance test for a backend is that compiling a ``TaskSpec`` produces the same environment
config the project already hand-wrote, field for field, with every difference either absent or
listed in a whitelist that states why. That needs configs to be comparable, and they are not
directly: they are trees of ``@configclass`` objects holding function references, types, sentinels
and engine objects, none of which compare or print usefully.

:func:`dump` flattens such a tree into plain data -- dicts, lists, numbers, strings -- resolving
callables and types to their qualified names, so that two configs can be diffed by path. It is
deliberately reflective rather than schema-driven: a backend that silently stopped emitting a field
would pass a schema-driven comparison that only checked the fields it knew about.

The whitelist is the part that matters for keeping this honest. A diff that is absent proves the
compiler reproduces the golden; a diff that is present and explained is a decision someone made and
a reviewer saw. What must not happen is a diff that is present and tolerated, which is what a
comparison with no whitelist mechanism degenerates into within about two weeks.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = ["Difference", "compare", "dump", "flatten", "qualname", "report", "unexplained"]

_MISSING = "<MISSING>"
_MAX_DEPTH = 24


def qualname(obj: Any) -> str:
    """Dotted name of a function or class, as the report and the golden both record it."""
    module = getattr(obj, "__module__", "")
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None) or repr(obj)
    return f"{module}.{name}" if module else str(name)


def dump(value: Any, *, _depth: int = 0) -> Any:
    """Reduce a config tree to plain comparable data.

    Callables and types become their qualified names, enums their values, dataclasses and
    ``@configclass`` instances their public fields. Anything with no better representation becomes
    ``repr``, tagged so that a reader can tell it apart from a genuine string.
    """
    if _depth > _MAX_DEPTH:  # pragma: no cover - cyclic configs are not expected
        return "<max depth>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if value is dataclasses.MISSING or repr(value) == "MISSING":
        return _MISSING
    if isinstance(value, type) or callable(value):
        return qualname(value)
    if isinstance(value, Mapping):
        return {str(k): dump(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return [dump(item, _depth=_depth + 1) for item in items]
    fields = _public_fields(value)
    if fields is not None:
        return {name: dump(getattr(value, name, None), _depth=_depth + 1) for name in fields}
    return f"<repr {value!r}>"


def _public_fields(value: Any) -> list[str] | None:
    """Field names of a config-like object, or ``None`` when it is not one.

    Isaac Lab's ``@configclass`` and mjlab's dataclasses both end up as dataclasses; anything else
    carrying a ``__dict__`` is treated as a config too, since the alternative is silently
    stringifying a subtree.
    """
    if isinstance(value, type):
        return None
    # Instance dictionary first and in its own order, then any declared field that never made it
    # there. Two reasons, and both matter. Isaac Lab configs bind attributes in ``__post_init__``
    # that were never declared -- the run name on the golden task is one -- so declared fields alone
    # would miss them. And the order is load-bearing rather than cosmetic: an observation group is
    # concatenated in attribute order, which is the order the engine's manager walks, so a dump that
    # sorted the names would compare equal to a config whose observation vector is laid out
    # differently.
    names = list(vars(value)) if hasattr(value, "__dict__") else []
    if dataclasses.is_dataclass(value):
        names += [field.name for field in dataclasses.fields(value) if field.name not in names]
    names = [name for name in names if not name.startswith("_")]
    return names or None


def flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    """Nested dump into ``dotted.path -> leaf``.

    List entries are indexed by position, because for the sequences in these configs -- joint name
    patterns, observation terms -- position is meaningful and a set comparison would lose the
    ordering that the whole D1 joint-order decision is about.
    """
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(data, list):
        out = {}
        for index, value in enumerate(data):
            out.update(flatten(value, f"{prefix}[{index}]"))
        return out
    return {prefix: data}


@dataclasses.dataclass(frozen=True)
class Difference:
    """One field where the compiled config and the golden disagree."""

    path: str
    golden: Any
    actual: Any
    reason: str | None = None

    @property
    def is_explained(self) -> bool:
        return self.reason is not None

    def __str__(self) -> str:
        line = f"{self.path}: golden={self.golden!r} actual={self.actual!r}"
        return f"{line}\n    reason: {self.reason}" if self.reason else line


def compare(golden: Any, actual: Any, allow: Mapping[str, str] | None = None) -> list[Difference]:
    """Every field where ``actual`` differs from ``golden``, explained where the whitelist says so.

    Args:
        golden: Dumped reference config.
        actual: Dumped compiled config.
        allow: ``path prefix -> reason``. A prefix rather than an exact path, because one decision
            usually shows up as several leaves -- ``actions.joint_pos.joint_names`` covers all 29
            of them -- and requiring one entry per leaf would push a reviewer to stop reading them.

    Returns:
        Differences in path order, unexplained ones first, so that a failure prints what needs
        attention before what has already been agreed.
    """
    allow = dict(allow or {})
    left, right = flatten(golden), flatten(actual)
    differences: list[Difference] = []
    for path in sorted(set(left) | set(right)):
        before, after = left.get(path, _MISSING), right.get(path, _MISSING)
        if before == after:
            continue
        differences.append(Difference(path, before, after, _reason_for(path, allow)))
    return sorted(differences, key=lambda d: (d.is_explained, d.path))


def _reason_for(path: str, allow: Mapping[str, str]) -> str | None:
    """Longest matching prefix wins, so a specific entry can override a broad one."""
    matches = [p for p in allow if path == p or path.startswith(f"{p}.") or path.startswith(f"{p}[")]
    return allow[max(matches, key=len)] if matches else None


def unexplained(differences: Iterable[Difference]) -> list[Difference]:
    """The ones no whitelist entry covers. An empty list is the acceptance criterion."""
    return [d for d in differences if not d.is_explained]


def unused(differences: Iterable[Difference], allow: Mapping[str, str]) -> list[str]:
    """Whitelist entries that explained nothing, which is a failure and not housekeeping.

    An entry outlives the difference it was written for whenever the compiler improves, and from
    then on it covers a path that no longer differs -- so the next real difference at that path is
    waved through with a reason describing something else. The whitelist is only trustworthy if
    every entry is still earning its place.
    """
    paths = [difference.path for difference in differences]
    return sorted(
        entry
        for entry in allow
        if not any(path == entry or path.startswith(f"{entry}.") or path.startswith(f"{entry}[") for path in paths)
    )


def report(differences: Sequence[Difference]) -> str:
    """Human-readable summary, for a test failure message."""
    missing = unexplained(differences)
    if not differences:
        return "no differences from golden."
    lines = [f"{len(differences)} difference(s), {len(missing)} unexplained:"]
    lines += [f"  {difference}" for difference in differences]
    return "\n".join(lines)
