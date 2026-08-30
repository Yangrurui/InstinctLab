"""Lowering an :class:`~instinctlab_engine.spec.entity.EntityRef` onto each engine's selector config.

Both engines converged on a class called ``SceneEntityCfg`` with the same shape -- ``name``,
``preserve_order``, and a ``<kind>_names`` / ``<kind>_ids`` pair per selector kind -- so lowering is
a rename-free field mapping. The name resolution underneath is the same code on both sides:
``resolve_matching_names`` is byte-identical apart from its docstring, and behaves identically for
every pattern order and both settings of ``preserve_order``. None of that needs reimplementing here.

What does need handling is where the two genuinely differ.

**Selector kinds.** Only ``joint`` and ``body`` are common. Isaac Lab adds ``fixed_tendon`` and
``object_collection``; mjlab adds eight more. A reference naming a kind the target engine cannot
express is rejected here rather than dropped, because dropping it produces a task that runs and
means something else.

**What ``<kind>_names`` holds after ``resolve()``.** This is the same kind of
same-name/different-meaning trap that this compatibility layer prevents.
Isaac Lab leaves the *user's patterns* in the field (it discards the matched names) while mjlab
overwrites it with the *matched names*. So a term reading ``asset_cfg.body_names`` gets
``[".*_ankle_roll_link"]`` under one engine and ``["left_ankle_roll_link", ...]`` under the other.
Real code reads it -- Isaac Lab's own ``events.py`` joins the field back into a regex to match USD
prim paths, and this repository stores it in an observation term. :func:`resolved_names` is the
portable way to ask the question, and it happens to need no engine-specific branch at all.

This module imports no engine at module scope; each lowering imports its own engine when called.

将 :class:`~instinctlab_engine.spec.entity.EntityRef` 下降为各引擎的选择器配置。

两引擎均收敛到 ``SceneEntityCfg`` 形态——``name``、``preserve_order``、每种选择器的
``<kind>_names`` / ``<kind>_ids``——下降过程无需改名，只是字段映射。底层名解析共用
``resolve_matching_names``（除 docstring 外字节一致），对任意模式顺序与 ``preserve_order`` 行为相同，
无需在此重实现。

真正需要处理的是两引擎 genuine 不同的部分。

**选择器种类。** 仅 ``joint`` 与 ``body`` 共通。Isaac Lab 另有 ``fixed_tendon``、``object_collection``；
mjlab 另有八种。引用目标引擎无法表达的种类时在此拒绝而非丢弃——丢弃会得到能跑但语义不同的任务。

**``resolve()`` 后 ``<kind>_names`` 存什么。** 这是兼容层所防止的同名异义陷阱。
Isaac Lab 保留 *用户模式*，mjlab 覆写为 *匹配名*。读 ``asset_cfg.body_names`` 时一侧是正则、一侧是具体名。
:func:`resolved_names` 是可移植问法，且无需 per-engine 分支。

本模块在模块级不 import 引擎；每次 ``lower()`` 调用时才 import 目标引擎。
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from instinctlab_engine.spec.entity import UNIVERSAL_KINDS, EntityRef
from instinctlab_engine.plugins import _PLUGIN_LOCK, _plugin_locked

from .errors import PortabilityError

__all__ = [
    "UnsupportedSelector",
    "lower",
    "register",
    "register_packages",
    "resolved_names",
    "selector_field",
    "selector_kinds",
    "universal",
]


@dataclass(frozen=True)
class _Selectors:
    """What one engine can select, and how its config wants to be built.

    单个引擎可选择的种类，及其配置对象的构造方式。
    """

    kinds: frozenset[str]
    cfg: tuple[str, str]
    container: type


_ENGINES: dict[str, _Selectors] = {}
_PACKAGES: dict[str, str] = {}


@_plugin_locked
def register_packages(packages: Mapping[str, str]) -> None:
    """Register lazy engine packages without making compat import the engine registry."""
    _PACKAGES.update(packages)


@_plugin_locked
def _snapshot_registrations() -> tuple[dict[str, _Selectors], dict[str, str]]:
    """Internal transaction snapshot used while backend plugins are discovered."""
    return dict(_ENGINES), dict(_PACKAGES)


@_plugin_locked
def _restore_registrations(
    snapshot: tuple[dict[str, _Selectors], dict[str, str]],
) -> None:
    """Restore selector/package registrations after failed plugin discovery."""
    engines, packages = snapshot
    _ENGINES.clear()
    _ENGINES.update(engines)
    _PACKAGES.clear()
    _PACKAGES.update(packages)


@_plugin_locked
def register(engine: str, *, kinds: Iterable[str], cfg: tuple[str, str], container: type) -> None:
    """Declare what ``engine`` can select. Called by that engine's package when it is imported.

    Args:
        engine: The engine key, matching its entry in
            :data:`instinctlab_engine.ADAPTERS`.
            引擎键，与 :data:`instinctlab_engine.ADAPTERS` 中的条目一致。
        kinds: Selector kinds its ``SceneEntityCfg`` accepts. Kinds two engines spell the same are
            not assumed to mean the same thing -- Isaac Lab's ``fixed_tendon`` and mjlab's
            ``tendon`` are registered apart, because treating them as one would let a reference
            through that the target resolves against a different set of elements.
            其 ``SceneEntityCfg`` 接受的选择器种类。拼写相同不假定语义相同——
            Isaac ``fixed_tendon`` 与 mjlab ``tendon`` 分开注册。
        cfg: Module path and attribute name of the engine's selector config, imported on use so
            that this module stays importable without any engine present.
            引擎选择器配置的模块路径与属性名；用时才 import，使本模块在无引擎环境下可导入。
        container: Sequence type the engine annotates its name fields with. Isaac Lab says
            ``list[str]`` and mjlab ``tuple[str, ...]``; both accept either at runtime, but matching
            the declaration keeps the produced config indistinguishable from a hand-written one,
            which is what the golden diff compares against.
            引擎注解 name 字段用的序列类型；匹配声明使产物与手写配置 indistinguishable。

    This is a registration rather than a table in this file for the reason decision S2 gives: an
    engine whose selectors nobody here anticipated should cost a call in its own package, not an
    edit to the shared layer. The shared layer still decides what happens to a kind it has never
    heard of, which is what :class:`UnsupportedSelector` is.

    采用注册而非本文件内静态表（决策 S2）：未anticipated 的引擎应在其包内注册，而非改共享层。
    共享层仍决定未知种类的命运，即 :class:`UnsupportedSelector`。
    """
    _ENGINES[engine] = _Selectors(frozenset(kinds), cfg, container)


@_plugin_locked
def _ensure_registered() -> None:
    """Import registered adapter packages so their selector declarations run.

    Adapters do not import their SDK at module scope, so this is safe on a machine with neither
    engine installed -- which is the case this whole layer is built to keep working.

    导入 adapter 包以触发注册副作用。adapter 不在模块级 import SDK，
    故在未安装任一引擎的机器上仍安全——本层即为此设计。
    """
    # CompileCtx may be used directly by an offline contract test without a
    # launcher first calling ``instinctlab_engine.names()``. Discover the
    # SDK-free backend registrars here so selector availability never depends
    # on import or test order.
    from instinctlab_engine import names

    names()
    for engine, package in _PACKAGES.items():
        if engine not in _ENGINES:
            importlib.import_module(package)


def selector_kinds() -> Mapping[str, frozenset[str]]:
    """Selector kinds every known engine accepts, keyed by engine.

    各已知引擎接受的选择器种类，以引擎名为键。
    """
    with _PLUGIN_LOCK:
        _ensure_registered()
        return MappingProxyType(
            {engine: entry.kinds for engine, entry in _ENGINES.items()}
        )


class UnsupportedSelector(PortabilityError):
    """Raised when an engine has no selector for a kind the reference names.

    引用命名的选择器种类目标引擎不支持时抛出。
    """


def selector_field(kind: str) -> str:
    """Name of the config field carrying patterns for ``kind``.

    Both engines follow the same convention for all twelve kinds between them, so this is one
    function rather than a per-engine table.

    承载 ``kind`` 模式的配置字段名。两引擎对全部十二种 kind 约定相同，故单一函数而非 per-engine 表。
    """
    return f"{kind}_names"


def _registered(engine: str) -> _Selectors:
    with _PLUGIN_LOCK:
        _ensure_registered()
        try:
            return _ENGINES[engine]
        except KeyError:
            raise KeyError(
                f"unknown engine {engine!r}; known engines are {sorted(_ENGINES)}"
            ) from None


def lower(ref: EntityRef, engine: str) -> Any:
    """Compile ``ref`` into ``engine``'s native ``SceneEntityCfg``.

    Args:
        ref: The engine-agnostic reference. / 引擎无关引用。
        engine: Target engine key, one of :data:`SELECTOR_KINDS`. / 目标引擎键。

    Returns:
        The engine's own ``SceneEntityCfg``, ready to be handed to its manager. Resolution to
        indices happens later, inside the engine, against the real scene.
        该引擎原生 ``SceneEntityCfg``，可交给 manager；索引解析稍后在引擎内对真实场景进行。

    Raises:
        UnsupportedSelector: ``ref`` names a selector kind this engine cannot express.
        KeyError: ``engine`` is not a known engine.
    """
    entry = _registered(engine)

    missing = sorted(ref.kinds() - entry.kinds)
    if missing:
        raise UnsupportedSelector(
            f"{engine} has no selector for {missing} (entity {ref.entity!r}). "
            f"It supports {sorted(entry.kinds)}. Express this per-engine, or drop the selector in the "
            "task spec so the omission is recorded rather than inferred."
        )

    kwargs: dict[str, Any] = {"name": ref.entity, "preserve_order": ref.preserve_order}
    for kind, patterns in ref.selectors().items():
        kwargs[selector_field(kind)] = entry.container(patterns)
    module, attribute = entry.cfg
    return getattr(importlib.import_module(module), attribute)(**kwargs)


def resolved_names(entity: Any, cfg: Any, kind: str = "body") -> list[str]:
    """The names a resolved selector actually selected, in the order it selected them.

    Read this instead of ``cfg.<kind>_names``, which means different things on the two engines: Isaac
    Lab leaves the caller's patterns in place, mjlab replaces them with what matched. Going through
    the indices sidesteps the difference entirely, because the indices are the thing both engines
    agree on.

    Args:
        entity: The scene entity the config was resolved against. / 配置所解析的场景实体。
        cfg: An engine ``SceneEntityCfg``, already resolved. / 已 resolve 的引擎 ``SceneEntityCfg``。
        kind: Selector kind to read. / 要读取的选择器种类。

    Returns:
        Matched names, ordered as the selection is ordered -- which follows the patterns when
        ``preserve_order`` was set and the entity's own order otherwise.
        实际匹配到的名字，顺序与选择顺序一致。
    """
    all_names: Sequence[str] = getattr(entity, f"{kind}_names")
    ids = getattr(cfg, f"{kind}_ids")
    if isinstance(ids, slice):
        return list(all_names[ids])
    if isinstance(ids, int):
        return [all_names[ids]]
    return [all_names[index] for index in ids]


def universal(ref: EntityRef) -> bool:
    """Whether every kind on ``ref`` is one all engines can express.

    ``ref`` 上的每种选择器是否均为所有引擎可表达的种类。
    """
    return ref.kinds() <= frozenset(UNIVERSAL_KINDS)
