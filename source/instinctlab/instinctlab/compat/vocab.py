"""The signed hub vocabulary: what each physical quantity means, and how every engine spells it.

Portable MDP terms read robot state through attribute names. Isaac Lab and MJLab currently happen
to agree on most of those names, which makes it tempting to treat "the name both engines use" as
the definition. That is not a definition -- it is a coincidence that a third engine will not honour,
and it leaves no place to notice when an upstream rename changes meaning.

So the hub is written down here explicitly. Choosing Isaac Lab's explicit-frame spelling as the hub
spelling is a *decision* (it is the most explicit of the two), not an observation. Each engine then
declares its own spoke. Anything a portable term reads must appear in :data:`HUB`.

``documented`` on a spoke is deliberately separate from ``attr``. Both engines return ``wxyz``
quaternions, but only Isaac Lab says so in its docstrings; MJLab inherits the convention from
MuJoCo's ``qpos``/``xquat`` without restating it. That asymmetry is an implicit dependency on an
upstream convention, so it is recorded rather than smoothed over. Maintainer verification checks
every claim in this table against the installed engines.

署名的中枢词汇表：每个物理量的含义，以及各引擎如何拼写它。

可移植 MDP term 通过属性名读取机器人状态。Isaac Lab 与 MJLab 目前碰巧在大多数名字上一致，
容易误把「两引擎共用的名字」当作定义——那不是定义，是巧合；第三引擎不会遵守，
上游改名改变语义时也无处察觉。

因此中枢在此显式写下。选用 Isaac Lab 的显式参考系拼写是 **决策**（两者中最 explicit），不是观察。
各引擎声明自己的 spoke（辐条映射）。可移植 term 读取的任何量必须出现在 :data:`HUB` 中。

spoke 上的 ``documented`` 与 ``attr``  deliberately 分离：两引擎四元数均为 ``wxyz``，
但只有 Isaac Lab 在文档中写明；MJLab 继承 MuJoCo ``qpos``/``xquat`` 约定而未重述。
该不对称是隐式上游依赖，故记录在案而非抹平；维护阶段会对照已安装引擎检验本表每条声明。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

ENGINES: tuple[str, ...] = ("isaacsim", "mjlab")
"""Engines with a spoke in this table. New engines append here and fill in every entry.

本表中有 spoke 的引擎。新引擎在此追加并为每条 hub 项填齐 spoke。
"""


class Frame(str, Enum):
    """Reference frame a quantity is expressed in.

    物理量所在的参考系。
    """

    WORLD = "world"
    """Simulation world frame. / 仿真世界系。"""
    BASE = "base"
    """Root link frame of the articulation. /  articulation 根连杆系。"""
    NONE = "none"
    """Frame-free (joint space, scalars). / 无参考系（关节空间、标量）。"""


class Anchor(str, Enum):
    """Which point on a body the quantity is measured at.

    The link/COM distinction is the single most common source of silent cross-engine error, so it
    is a required field rather than something inferred from the name.

    物理量在 body 上的测量锚点。

    link/COM 区分是跨引擎静默错误的最常见来源，故为必填字段，不能从名字推断。
    """

    LINK = "link"
    """Body/link origin as authored in the asset. / 资产中定义的 body/link 原点。"""
    COM = "com"
    """Body centre of mass. / body 质心。"""
    ELEMENT = "element"
    """The element's own origin (geom, site). / 元素自身原点（geom、site）。"""
    NA = "n/a"
    """Not anchored to a point. / 不锚定在空间点上。"""


class RotationConvention(str, Enum):
    """Component order of a quaternion.

    四元数分量顺序。
    """

    WXYZ = "wxyz"


CANONICAL_QUATERNION = RotationConvention.WXYZ
"""Every quaternion crossing ``spec/``, ``mdp/`` or ``compat/`` is ``(w, x, y, z)``.

``xyzw`` exists only at engine API boundaries (PhysX/USD transforms, warp meshes, motion files).
Conversions happen on the line that calls the engine and must not propagate across a function
boundary. Note that both engines ship ``convert_quat(quat, to="xyzw")`` -- the default direction
converts *away* from the hub, so the direction is always passed explicitly.

穿越 ``spec/``、``mdp/``、``compat/`` 的四元数一律为 ``(w, x, y, z)``。

``xyzw`` 仅存在于引擎 API 边界（PhysX/USD 变换、warp mesh、motion 文件）。
转换发生在调用引擎的那一行，不得跨函数边界传播。两引擎的 ``convert_quat(quat, to="xyzw")`` 默认方向
是 *离开* 中枢约定，故方向必须显式传入。
"""

_MUJOCO_QUAT_BASIS = "MuJoCo qpos/xquat is w-first; MJLab docstrings do not restate it"


@dataclass(frozen=True)
class Spoke:
    """How one engine exposes a hub entry.

    单个引擎如何暴露一条 hub 项。
    """

    attr: str | None
    """Attribute on the engine's articulation data object. ``None`` means the engine has no
    equivalent -- the entry stays in the hub so a task can still reference it and get a clear
    capability error instead of a silent drop.

    引擎 articulation data 对象上的属性名。``None`` 表示该引擎无等价物——hub 仍保留该项，
    任务引用时会得到明确的能力错误而非静默丢弃。
    """
    documented: bool = False
    """Whether the engine's own docs pin the semantics recorded here, as opposed to us relying on
    an upstream convention. ``False`` marks an implicit dependency to re-check on engine upgrade.

    引擎自身文档是否钉住此处记录的语义（相对依赖上游约定）。``False`` 表示引擎升级时需复查的隐式依赖。
    """
    evidence: str = ""
    """Where the semantics were read from. / 语义依据出处。"""

    @property
    def available(self) -> bool:
        return self.attr is not None


@dataclass(frozen=True)
class HubEntry:
    """One quantity in the hub vocabulary.

    中枢词汇表中的一条物理量。
    """

    name: str
    """Hub spelling. Portable terms use this name. / 中枢拼写；可移植 term 使用此名。"""
    unit: str
    frame: Frame
    anchor: Anchor
    doc: str
    spokes: Mapping[str, Spoke]
    rotation: RotationConvention | None = None
    """Set only for quaternions. / 仅四元数项设置。"""

    def spoke(self, engine: str) -> Spoke:
        try:
            return self.spokes[engine]
        except KeyError:
            raise KeyError(f"hub entry {self.name!r} has no spoke for engine {engine!r}") from None


def _both(attr: str, *, isaac_documented: bool = False, isaac_evidence: str = "", mjlab_evidence: str = "") -> dict:
    """Spokes for a quantity both engines spell identically.

    两引擎拼写相同的量的 spoke 映射。
    """
    return {
        "isaacsim": Spoke(
            attr,
            documented=isaac_documented,
            evidence=isaac_evidence or f"isaaclab ArticulationData.{attr}",
        ),
        "mjlab": Spoke(attr, documented=False, evidence=mjlab_evidence or f"mjlab EntityData.{attr}"),
    }


def _mjlab_only(attr: str, *, reason: str, mjlab_evidence: str = "") -> dict:
    return {
        "isaacsim": Spoke(None, evidence=reason),
        "mjlab": Spoke(attr, evidence=mjlab_evidence or f"mjlab EntityData.{attr}"),
    }


_QUAT_DOC = "isaaclab ArticulationData docstring states (w, x, y, z)"

_ENTRIES: tuple[HubEntry, ...] = (
    # --- root link pose and velocity / 根连杆位姿与速度 ---
    HubEntry(
        name="root_link_pos_w",
        unit="m",
        frame=Frame.WORLD,
        anchor=Anchor.LINK,
        doc="Root link origin position in the world frame.",
        spokes=_both("root_link_pos_w"),
    ),
    HubEntry(
        name="root_link_quat_w",
        unit="1",
        frame=Frame.WORLD,
        anchor=Anchor.LINK,
        rotation=RotationConvention.WXYZ,
        doc="Root link orientation in the world frame.",
        spokes=_both(
            "root_link_quat_w",
            isaac_documented=True,
            isaac_evidence=_QUAT_DOC,
            mjlab_evidence=_MUJOCO_QUAT_BASIS,
        ),
    ),
    HubEntry(
        name="root_link_lin_vel_w",
        unit="m/s",
        frame=Frame.WORLD,
        anchor=Anchor.LINK,
        doc="Root link linear velocity in the world frame.",
        spokes=_both("root_link_lin_vel_w"),
    ),
    HubEntry(
        name="root_link_ang_vel_w",
        unit="rad/s",
        frame=Frame.WORLD,
        anchor=Anchor.LINK,
        doc="Root link angular velocity in the world frame.",
        spokes=_both("root_link_ang_vel_w"),
    ),
    HubEntry(
        name="root_link_lin_vel_b",
        unit="m/s",
        frame=Frame.BASE,
        anchor=Anchor.LINK,
        doc="Root link linear velocity in the base frame.",
        spokes=_both("root_link_lin_vel_b"),
    ),
    HubEntry(
        name="root_link_ang_vel_b",
        unit="rad/s",
        frame=Frame.BASE,
        anchor=Anchor.LINK,
        doc="Root link angular velocity in the base frame.",
        spokes=_both("root_link_ang_vel_b"),
    ),
    # --- root centre of mass / 根质心 ---
    HubEntry(
        name="root_com_pos_w",
        unit="m",
        frame=Frame.WORLD,
        anchor=Anchor.COM,
        doc="Root centre-of-mass position in the world frame.",
        spokes=_both("root_com_pos_w"),
    ),
    HubEntry(
        name="root_com_quat_w",
        unit="1",
        frame=Frame.WORLD,
        anchor=Anchor.COM,
        rotation=RotationConvention.WXYZ,
        doc="Root centre-of-mass orientation in the world frame.",
        spokes=_both(
            "root_com_quat_w",
            isaac_documented=True,
            isaac_evidence=_QUAT_DOC,
            mjlab_evidence=_MUJOCO_QUAT_BASIS,
        ),
    ),
    HubEntry(
        name="root_com_lin_vel_w",
        unit="m/s",
        frame=Frame.WORLD,
        anchor=Anchor.COM,
        doc=(
            "Root centre-of-mass linear velocity in the world frame. This is what Isaac Lab's legacy"
            " aliases resolve to -- see instinctlab.compat.denylist.LEGACY_COM_ALIASES."
        ),
        spokes=_both("root_com_lin_vel_w"),
    ),
    # --- derived orientation / 派生朝向量 ---
    HubEntry(
        name="projected_gravity_b",
        unit="1",
        frame=Frame.BASE,
        anchor=Anchor.LINK,
        doc="Unit gravity direction projected into the base frame.",
        spokes=_both("projected_gravity_b"),
    ),
    HubEntry(
        name="heading_w",
        unit="rad",
        frame=Frame.WORLD,
        anchor=Anchor.NA,
        doc="Yaw of the base frame about the world z axis.",
        spokes=_both("heading_w"),
    ),
    # --- joint space / 关节空间 ---
    HubEntry(
        name="joint_pos",
        unit="rad (revolute) / m (prismatic)",
        frame=Frame.NONE,
        anchor=Anchor.NA,
        doc="Joint positions in the canonical DFS joint order.",
        spokes=_both("joint_pos"),
    ),
    HubEntry(
        name="joint_vel",
        unit="rad/s (revolute) / m/s (prismatic)",
        frame=Frame.NONE,
        anchor=Anchor.NA,
        doc="Joint velocities in the canonical DFS joint order.",
        spokes=_both("joint_vel"),
    ),
    # --- per-body / 逐 body ---
    HubEntry(
        name="body_link_pos_w",
        unit="m",
        frame=Frame.WORLD,
        anchor=Anchor.LINK,
        doc="Per-body link origin positions in the world frame.",
        spokes=_both("body_link_pos_w"),
    ),
    HubEntry(
        name="body_link_quat_w",
        unit="1",
        frame=Frame.WORLD,
        anchor=Anchor.LINK,
        rotation=RotationConvention.WXYZ,
        doc="Per-body link orientations in the world frame.",
        spokes=_both(
            "body_link_quat_w",
            isaac_documented=True,
            isaac_evidence=_QUAT_DOC,
            mjlab_evidence=_MUJOCO_QUAT_BASIS,
        ),
    ),
    # Per-body *velocity* is deliberately absent: the two engines derive it differently for every
    # body except the root, so it is denylisted rather than offered here. Root velocity is available
    # above; per-body velocity needs a per-engine term. See instinctlab.compat.denylist.
    # 逐 body *速度* 刻意不提供：除根外各 body 两引擎推导方式不同，故列入 denylist 而非 hub。
    # 根速度见上；逐 body 速度须用 per-engine term。见 instinctlab.compat.denylist。
    # --- MuJoCo-native elements / MuJoCo 原生元素 ---
    # Kept in the hub even though Isaac Sim has no equivalent: an mjlab project being migrated the
    # other way must be able to express these, and get a capability error rather than a silent drop.
    # 虽 Isaac Sim 无等价物仍保留：反向迁移 mjlab 项目须能表达这些量，并得到能力错误而非静默丢弃。
    HubEntry(
        name="geom_pos_w",
        unit="m",
        frame=Frame.WORLD,
        anchor=Anchor.ELEMENT,
        doc="Collision/visual geom origin positions in the world frame.",
        spokes=_mjlab_only("geom_pos_w", reason="Isaac Sim has no per-geom state view"),
    ),
    HubEntry(
        name="geom_quat_w",
        unit="1",
        frame=Frame.WORLD,
        anchor=Anchor.ELEMENT,
        rotation=RotationConvention.WXYZ,
        doc="Collision/visual geom orientations in the world frame.",
        spokes=_mjlab_only(
            "geom_quat_w",
            reason="Isaac Sim has no per-geom state view",
            mjlab_evidence=_MUJOCO_QUAT_BASIS,
        ),
    ),
    HubEntry(
        name="site_pos_w",
        unit="m",
        frame=Frame.WORLD,
        anchor=Anchor.ELEMENT,
        doc="Site origin positions in the world frame.",
        spokes=_mjlab_only("site_pos_w", reason="Isaac Sim has no site concept"),
    ),
    HubEntry(
        name="site_quat_w",
        unit="1",
        frame=Frame.WORLD,
        anchor=Anchor.ELEMENT,
        rotation=RotationConvention.WXYZ,
        doc="Site orientations in the world frame.",
        spokes=_mjlab_only(
            "site_quat_w",
            reason="Isaac Sim has no site concept",
            mjlab_evidence=_MUJOCO_QUAT_BASIS,
        ),
    ),
)

HUB: Mapping[str, HubEntry] = MappingProxyType({entry.name: entry for entry in _ENTRIES})
"""The hub vocabulary, keyed by hub spelling.

中枢词汇表，以中枢拼写为键。
"""


def hub_entry(name: str) -> HubEntry:
    """Look up a hub entry, refusing names that are not part of the signed vocabulary.

    查找 hub 项；拒绝不在署名词汇表中的名字。
    """
    try:
        return HUB[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not in the hub vocabulary. Portable terms may only read hub quantities;"
            " add an entry to instinctlab.compat.vocab (with a spoke per engine) first."
        ) from None


def spoke_attr(name: str, engine: str) -> str:
    """Resolve a hub name to the attribute ``engine`` exposes it as.

    Raises when the engine has no equivalent, so the caller has to decide between an engine-specific
    implementation and declaring the capability optional.

    将 hub 名解析为 ``engine`` 暴露的属性名。

    无等价物时抛出，由调用方选择 per-engine 实现或将能力标为 OPTIONAL。
    """
    spoke = hub_entry(name).spoke(engine)
    if spoke.attr is None:
        raise LookupError(f"engine {engine!r} has no equivalent of hub quantity {name!r}: {spoke.evidence}")
    return spoke.attr
