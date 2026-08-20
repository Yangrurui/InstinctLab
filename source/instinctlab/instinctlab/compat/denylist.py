"""Attributes whose names agree across engines while their semantics do not.

A portable term reads ``asset.data.<attr>`` and runs unchanged under both engines' native managers.
That only works while the attribute means the same thing on both sides. The entries below are the
cases where it does not: reading them from a portable term produces numbers that are plausible,
silently wrong, and very hard to attribute later. So they are refused at import/compile time and
have to be handled by a per-engine implementation instead.

Every claim here was checked against the installed engines; ``tests/test_compat_vocab.py`` re-checks
the mechanically verifiable parts (attribute existence, alias targets) so upstream renames surface
as a test failure rather than a silent semantic change.

跨引擎同名但语义不同的属性清单。

可移植 term 通过 ``asset.data.<attr>`` 读取状态，在两引擎原生 manager 下应能原样运行——前提是两侧含义一致。
下列条目不满足该条件：从可移植 term 读取会得到 plausible 但静默错误的数值，事后极难归因。
因此在 import/编译期拒绝，须改用 per-engine 实现。

每条声明均已对照已安装引擎核实；``tests/test_compat_vocab.py`` 会复检可机械验证的部分（属性存在性、别名目标），
上游改名会以测试失败暴露，而非静默改变语义。

注：``_ENTRIES`` 内字符串为运行时 ``PortabilityError`` 消息，保持英文；各条目前的 ``#`` 注释为中英对照。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


class PortabilityError(RuntimeError):
    """Raised when a portable term reaches for an attribute that does not port.

    可移植 term 访问不可跨引擎的属性时抛出。
    """


@dataclass(frozen=True)
class DenylistEntry:
    """One same-name-different-meaning trap.

    一条「同名不同义」陷阱记录。
    """

    name: str
    summary: str
    per_engine: Mapping[str, str]
    resolution: str


def _entry(name: str, summary: str, isaacsim: str, mjlab: str, resolution: str) -> DenylistEntry:
    return DenylistEntry(
        name=name,
        summary=summary,
        per_engine=MappingProxyType({"isaacsim": isaacsim, "mjlab": mjlab}),
        resolution=resolution,
    )


_ENTRIES: tuple[DenylistEntry, ...] = (
    # joint_acc — same name, different derivation / 同名，推导方式不同
    _entry(
        "joint_acc",
        "Same name, different derivation.",
        isaacsim="Finite difference of joint velocity across steps, held in ArticulationData._previous_joint_vel.",
        mjlab="MuJoCo's analytic qacc.",
        resolution=(
            "Do not use as a cross-engine signal. If a reward needs it, declare the term per-engine"
            " and state the tolerance -- the two are not comparable value by value."
        ),
    ),
    # applied_torque — Isaac-only name; mjlab false friend actuator_force / 仅 Isaac 有此名；mjlab 假朋友 actuator_force
    _entry(
        "applied_torque",
        "Isaac-only name with a false friend on the MJLab side.",
        isaacsim="ArticulationData.applied_torque, joint space (nv).",
        mjlab=(
            "No applied_torque. The joint-space equivalent is qfrc_actuator (nv). actuator_force"
            " looks like the match but is scalar actuator output in actuation space (nu), which is"
            " a different dimension whenever actuators are not one-per-joint."
        ),
        resolution="Read through a per-engine accessor; never map applied_torque to actuator_force.",
    ),
    # write_root_state_to_sim — velocity rows about different points / 速度行参考点不同
    _entry(
        "write_root_state_to_sim",
        "Same name, and the velocity rows it accepts are about a different point.",
        isaacsim="Takes the root link's pose and the centre of mass's velocity.",
        mjlab="Takes the root link's pose and the root link's velocity.",
        resolution=(
            "Use the frame-qualified writers, which both engines have:"
            " write_root_link_pose_to_sim and write_root_link_velocity_to_sim, or their com"
            " counterparts. Writing the same thirteen numbers through the unqualified method puts"
            " the two robots into different states -- measured at up to 0.85 m/s of link velocity"
            " for this task's G1 -- after which every velocity-reading term disagrees for a reason"
            " that has nothing to do with the terms. scripts/probe_terms.py does it the right way."
        ),
    ),
    # default_root_state — velocity frame mismatch / 默认根状态速度参考系不一致
    _entry(
        "default_root_state",
        "Same name, velocity rows expressed about a different point.",
        isaacsim="Linear/angular velocity rows are centre-of-mass quantities.",
        mjlab="Linear/angular velocity rows are root-link quantities.",
        resolution=(
            "Reset events are per-engine anyway (engines/<name>/events.py). Keep the frame choice"
            " there and do not read this attribute from a portable term."
        ),
    ),
    # body_link_lin_vel_w — per-body velocity semantics differ / 非根 body 线速度语义不同
    _entry(
        "body_link_lin_vel_w",
        "Same name, differs for every body except the root.",
        isaacsim="Per-body centre-of-mass offset applied to each body individually.",
        mjlab="Derived using the root's subtree centre of mass.",
        resolution=(
            "For the root body use the hub entry root_link_lin_vel_w, which both engines derive the"
            " same way. Anything per-body goes through a per-engine term with a declared tolerance;"
            " the hub deliberately offers no per-body velocity."
        ),
    ),
    # gravity_vec_w — spelling and live-gravity behaviour / 拼写与非默认重力行为不同
    _entry(
        "gravity_vec_w",
        "Different spelling and different behaviour under non-default gravity.",
        isaacsim=(
            "Spelled GRAVITY_VEC_W (upper case) and normalised from the live sim gravity, so it"
            " follows a task that changes gravity."
        ),
        mjlab="Spelled gravity_vec_w (lower case) and hard-coded to [0, 0, -1] at entity build time.",
        resolution=(
            "Portable terms use projected_gravity_b, which both engines derive consistently. A task"
            " that randomises gravity must treat this as a per-engine concern."
        ),
    ),
    # net_forces_w — contact force quantity and frame differ / 接触力物理量与坐标系均不同
    _entry(
        "net_forces_w",
        "Contact force that measures a different quantity on each side, in a different frame.",
        isaacsim=(
            "ContactSensorData.net_forces_w: world frame, normal component only. Its own docstring"
            " warns that the tangential contribution is excluded."
        ),
        mjlab=(
            "No net_forces_w. The nearest field is ContactData.force, which is the full 3-D contact"
            " force and sits in the contact frame unless reduce='netforce' or global_frame=True"
            " moves it to world."
        ),
        resolution=(
            "Norms of the two are not the same physical quantity -- one is normal load, the other"
            " includes friction -- so a newton threshold does not transfer. Portable terms detect"
            " contact through compat.sensors.in_contact, which defers to each engine's own contact"
            " criterion. A term that truly needs force magnitude is per-engine and must declare its"
            " threshold and tolerance per engine."
        ),
    ),
    # points_vel_w — velocity measured at different points / 同名缓冲，速度参考点不同
    _entry(
        "points_vel_w",
        "Same buffer name, velocity about a different point on Isaac's PhysX view.",
        isaacsim=(
            "VolumePoints fills vel_w from RigidBodyView.get_velocities(): the linear row is COM."
            " Legacy sensors (default velocity='com') then do v_com + ω × (p - link_origin), which"
            " is neither the COM formula nor the link formula whenever the foot COM is offset."
        ),
        mjlab=(
            "cvel + ω × r. mjwarp cvel linear is at the free-joint subtree COM; "
            "transport from the attach body's own subtree adds ω × (pelvis − ankle)."
            " InstinctMJ transports from the foot's own subtree and so carries that"
            " lever: at |ω| = 2.83 rad/s its v_link measures (1.267, −1.301, −0.248)"
            " where the link value is (−0.035, 0, −0.0002). ω = 0 hides it entirely."
        ),
        resolution=(
            "Hub is attach-body link origin: v_link + ω × (p_w - origin_w). The new-stack Isaac"
            " sensor sets velocity='attach_link' and converts COM → link before the cross product."
            " Portable terms read compat.sensors.volume_points_vel_w, which refuses a COM sensor."
            " Do not change the legacy default — parkour_env_cfg.py still wants COM."
            " Consequence: the cross-engine task deliberately does not reproduce"
            " InstinctMJ's volume_points_penetration magnitudes, so that penalty is not"
            " a number to match against the upstream project either."
        ),
    ),
    # actuator_delay — same physics-step bounds, different default resampling / 同是物理步，默认重采样不同
    _entry(
        "actuator_delay",
        "Same physics-step inclusive bounds; Isaac holds one draw per episode, mjlab defaults to every step.",
        isaacsim=(
            "DelayedPDActuator.min_delay / max_delay are physics steps, inclusive "
            "(torch.randint low=min, high=max+1). The lag is sampled in reset and held "
            "until the next reset. compute() pushes the command every physics step."
        ),
        mjlab=(
            "BuiltinPdActuatorCfg.delay_min_lag / delay_max_lag are also physics steps, "
            "inclusive. delay_update_period defaults to 0, which resamples every physics "
            "step. InstinctMJ recovers Isaac's per-episode hold with a period larger than "
            "any episode and delay_per_env_phase=False; identical periods also fuse groups "
            "onto one shared DelayBuffer."
        ),
        resolution=(
            "Hub is physics steps, inclusive [min, max], one draw per episode per actuator "
            "group. Tasks declare RobotSpec.actuator_delay. Adapters apply the hub: Isaac "
            "DelayedPD already matches; mjlab must set a reset-only period and a distinct "
            "period per group, and must not copy delay_update_period=0. A 0–2 that meant "
            "control steps on one side and physics steps on the other would be a silent 4x "
            "lag — that is not the case here, but the resampling default is."
        ),
    ),
)

DENYLIST: Mapping[str, DenylistEntry] = MappingProxyType({entry.name: entry for entry in _ENTRIES})
"""Denylisted attributes, keyed by the attribute name a term would reach for.

黑名单属性，以 term 会访问的属性名为键。
"""


LEGACY_COM_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "root_lin_vel_w": "root_com_lin_vel_w",
        "root_lin_vel_b": "root_com_lin_vel_b",
        "root_ang_vel_w": "root_com_ang_vel_w",
        "root_ang_vel_b": "root_com_ang_vel_b",
        "root_vel_w": "root_com_vel_w",
        "body_lin_vel_w": "body_com_lin_vel_w",
        "body_ang_vel_w": "body_com_ang_vel_w",
        "body_vel_w": "body_com_vel_w",
        "body_lin_acc_w": "body_com_lin_acc_w",
        "body_ang_acc_w": "body_com_ang_acc_w",
        "body_acc_w": "body_com_acc_w",
        "com_pos_b": "body_com_pos_b",
        "com_quat_b": "body_com_quat_b",
    }
)
"""Isaac Lab legacy aliases that resolve to **centre-of-mass** quantities.

The dangerous subset is the one whose *name* carries no hint of it: rewriting ``root_lin_vel_b`` to
``root_link_lin_vel_b`` is the obvious move and is a different physical quantity. MJLab has no such
aliases, so nothing downstream catches it. The correct rewrite is the mapping recorded here.
Verified against Isaac Lab's own docstrings ("Same as :attr:`root_com_...`").

Isaac Lab 旧别名，解析为 **质心（COM）** 量。

危险子集是名字看不出 COM 的那些：把 ``root_lin_vel_b`` 改成 ``root_link_lin_vel_b`` 看似自然，实则换了物理量。
MJLab 无此类别名，下游不会报错。正确改写见本表；已与 Isaac Lab 文档（"Same as :attr:`root_com_...`"）核对。
"""


LEGACY_LINK_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "root_pos_w": "root_link_pos_w",
        "root_quat_w": "root_link_quat_w",
        "root_pose_w": "root_link_pose_w",
        "body_pos_w": "body_link_pos_w",
        "body_quat_w": "body_link_quat_w",
        "body_pose_w": "body_link_pose_w",
    }
)
"""Isaac Lab legacy aliases that resolve to **link** quantities.

Harmless in meaning but still ambiguous to a reader, and absent from MJLab. The codemod rewrites
them to the explicit spelling so the link/COM choice is visible at every call site.

Isaac Lab 旧别名，解析为 **连杆（link）** 量。

语义无害但对读者仍易混淆，且 MJLab 不存在。codemod 会改写为显式拼写，使 link/COM 选择在每个调用点可见。
"""


def is_legacy_alias(attr: str) -> bool:
    return attr in LEGACY_COM_ALIASES or attr in LEGACY_LINK_ALIASES


def explicit_name(attr: str) -> str:
    """Rewrite a legacy Isaac Lab alias to its explicit-frame spelling.

    将 Isaac Lab 旧别名改写为带显式参考系的拼写。
    """
    if attr in LEGACY_COM_ALIASES:
        return LEGACY_COM_ALIASES[attr]
    if attr in LEGACY_LINK_ALIASES:
        return LEGACY_LINK_ALIASES[attr]
    return attr


def assert_portable(attr: str) -> None:
    """Refuse attributes that must not be read from a portable term.

    拒绝不可从可移植 term 读取的属性。

    Raises:
        PortabilityError: if ``attr`` is denylisted or is a legacy alias.
            若 ``attr`` 在黑名单中或为旧别名。
    """
    entry = DENYLIST.get(attr)
    if entry is not None:
        detail = "; ".join(f"{engine}: {text}" for engine, text in entry.per_engine.items())
        raise PortabilityError(f"{attr!r} does not port across engines. {entry.summary} {detail} -> {entry.resolution}")
    if is_legacy_alias(attr):
        target = explicit_name(attr)
        anchor = "centre-of-mass" if attr in LEGACY_COM_ALIASES else "link"
        raise PortabilityError(
            f"{attr!r} is an Isaac Lab legacy alias for the {anchor} quantity {target!r} and does not"
            f" exist on other engines. Use {target!r} explicitly."
        )
