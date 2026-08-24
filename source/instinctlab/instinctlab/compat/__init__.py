"""Thin compatibility layer shared by the portable MDP terms and every engine adapter.

:mod:`~instinctlab.compat.vocab` is the single named source of truth for what a physical quantity
means. :mod:`~instinctlab.compat.denylist` records the attributes whose names agree across engines
while their semantics do not. :mod:`~instinctlab.compat.math` carries the tensor math both engines
already share, so that a term can do frame arithmetic without importing either one; its names are
left on the submodule rather than re-exported here, since callers read better as
``math_utils.quat_apply_inverse``. :mod:`~instinctlab.compat.entity` lowers an ``EntityRef`` onto
each engine's selector config, which is where the engines diverge far more than their data
attributes do, and :mod:`~instinctlab.compat.sensors` normalizes contact, ray and volume-point
outputs where the engines disagree on names, shape or missing-value conventions.
:mod:`~instinctlab.compat.env` is the smallest of them, because the two environment classes turned
out to agree on nearly everything a term reads; it covers the command lookup, which fails
differently on each engine, and names the environment type portably.

可移植 MDP term 与各引擎 adapter 共用的薄兼容层。

:mod:`~instinctlab.compat.vocab` 是物理量含义的署名真源。
:mod:`~instinctlab.compat.denylist` 记录跨引擎同名但语义不同的属性。
:mod:`~instinctlab.compat.math` 承载两引擎共有的张量数学，term 做坐标运算时无需 import 任一引擎；
其子模块名不在此 re-export，调用方习惯写成 ``math_utils.quat_apply_inverse``。
:mod:`~instinctlab.compat.entity` 将 ``EntityRef`` 下降为各引擎的选择器配置（引擎在此处分歧远大于数据属性）。
:mod:`~instinctlab.compat.sensors` 统一接触、射线与体积点输出中的名称、形状和缺失值约定。
:mod:`~instinctlab.compat.env` 最小：两引擎 env 类在 term 可读字段上已基本一致；仅覆盖命令查找（失败方式不同）
与环境类型的可移植命名。
"""

from __future__ import annotations

from .denylist import DENYLIST, LEGACY_COM_ALIASES, DenylistEntry, PortabilityError, assert_portable
from .entity import UnsupportedSelector, lower, resolved_names, selector_kinds
from .env import ENV_TYPE_NAMES, RlEnv, command_names, env_engine, get_command, has_command
from .vocab import (
    CANONICAL_QUATERNION,
    ENGINES,
    HUB,
    Anchor,
    Frame,
    HubEntry,
    RotationConvention,
    Spoke,
    hub_entry,
    spoke_attr,
)

__all__ = [
    "CANONICAL_QUATERNION",
    "DENYLIST",
    "ENGINES",
    "ENV_TYPE_NAMES",
    "HUB",
    "LEGACY_COM_ALIASES",
    "selector_kinds",
    "Anchor",
    "DenylistEntry",
    "Frame",
    "HubEntry",
    "PortabilityError",
    "RlEnv",
    "RotationConvention",
    "Spoke",
    "UnsupportedSelector",
    "assert_portable",
    "command_names",
    "env_engine",
    "get_command",
    "has_command",
    "hub_entry",
    "lower",
    "resolved_names",
    "spoke_attr",
]
