"""Reading contact sensors the same way under either engine.

This module is the opposite case to :mod:`~instinctlab.compat.entity`. Entities need no runtime
indirection, because both engines spell the state attributes the same. Contact sensors do, because
they agree on some names and shapes and disagree on others, and the disagreements are the kind that
produce plausible wrong numbers rather than errors.

What agrees
-----------
All four air/contact-time tensors carry the same names on both engines -- ``current_air_time``,
``last_air_time``, ``current_contact_time``, ``last_contact_time`` -- and all four are 2-D
``(env, element)``. Both engines also order force history newest-first. A term reading step timing
therefore ports with nothing more than index resolution.

That matters more than it looks: each engine computes these durations from *its own* notion of
contact, using its own solver's forces, inside its own sensor. The reconciliation has already
happened by the time a term sees a duration in seconds. This is the portable contact signal.

What does not agree
-------------------
**The element list has different names.** Isaac Lab exposes ``ContactSensor.body_names``, mjlab
``ContactSensor.primary_names``. Same concept: the order the second axis is indexed in.

**Force history has a different axis order.** Isaac Lab is ``(env, time, element, 3)``, mjlab is
``(env, element, time, 3)``. Both are newest-first along the time axis, so a term that slices
elements on the wrong axis gets a correctly-shaped tensor of the wrong thing whenever the history
length happens to equal the element count -- two feet and two substeps, say.

**Force does not mean the same thing, and no transposition fixes it.** Isaac Lab's
``net_forces_w`` is world-frame and *normal only*; its docstring warns explicitly that it excludes
the tangential component. mjlab's ``force`` is the full 3-D contact force, expressed in the contact
frame unless ``reduce="netforce"`` or ``global_frame=True`` moves it to world. So ``‖force‖`` is
the normal load on one engine and the total load including friction on the other, and the two
differ by however much friction is carrying at that instant. A newton threshold tuned on one engine
does not transfer. :func:`contact_force_history` returns the tensors on a common axis order and
nothing more; it does not pretend the values are comparable.

Engine detection is by duck typing on the element-name attribute, so this module imports no engine
and a term does not have to know which one it is running under.

Resolution happens once
-----------------------
Which indices a reference names is worked out on first use and remembered. This is not an
optimisation in the usual sense of the word -- it is the difference between a usable environment
and an unusable one. Isaac Lab's ``ContactSensor.body_names`` is a property that rebuilds itself
from the physics view on every access, and at four thousand environments that access costs about
seventy milliseconds. A term that resolves its feet on every evaluation therefore spends most of
the step in name lookup: measured on flat G1, three such terms accounted for 18.7 of 21.6 seconds,
and the GPU sat idle while Python enumerated prim paths.

Isaac Lab's own terms do not pay this, because ``SceneEntityCfg`` is resolved once when the manager
is built and the term afterwards holds plain indices. A ``ContactSensorRef`` is resolved by the
term rather than by the manager -- that is what lets one declaration work against Isaac Lab's one
broad sensor and mjlab's several narrow ones -- so the caching that the managers give for free has
to happen here instead.

The cache is keyed by sensor and reference, and holds sensors weakly so that a closed environment
is collectable. It is sound because a sensor's element list is fixed once the scene is built: both
engines derive it from prims that exist for the lifetime of the simulation.

以相同方式读取两引擎接触传感器。

与 :mod:`~instinctlab.compat.entity` 相反：实体状态属性拼写一致，无需运行时间接层；接触传感器则部分一致、
部分不一致，且不一致处常产出 plausible 的错误数值而非异常。

一致之处
--------
四路空中/接触时长张量名称与形状 ``(env, element)`` 均相同；力历史均为 newest-first。
可移植接触信号是 **时长（秒）**——各引擎已在传感器内部用自己的接触判据完成 reconciliation。

不一致之处
----------
**元素列表属性名不同**（``body_names`` vs ``primary_names``）。
**力历史轴序不同**（Isaac ``(env, time, element, 3)``，mjlab ``(env, element, time, 3)``）。
**力的物理含义不同**——法向 vs 全三维+摩擦；:func:`contact_force_history` 只统一轴序，不假装数值可比。

通过元素名属性 duck typing 识别引擎；本模块不 import 引擎。

解析只做一次
------------
首次使用时解析索引并缓存。Isaac ``body_names`` 每次访问重建，4096 env 下单次约 70ms——
flat G1 上三个 term 曾占 21.6s 中的 18.7s。``ContactSensorRef`` 由 term 解析而非 manager，
故缓存须在此实现。弱引用键，场景构建后元素列表固定。
"""

from __future__ import annotations

import torch
import weakref
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping

from instinctlab.spec.sensor import ContactSensorRef

from .denylist import PortabilityError

__all__ = [
    "air_time",
    "contact_force_history",
    "contact_time",
    "depth_image",
    "element_ids",
    "element_names",
    "forget",
    "in_contact",
    "ray_hits_w",
    "registered_cylinder_count",
    "require_volume_points_registered",
    "sensor_engine",
    "volume_points_penetration_offset",
    "volume_points_vel_w",
]

_ELEMENT_NAME_ATTR = {"isaacsim": "body_names", "mjlab": "primary_names"}

_FORCE_HISTORY: Mapping[str, tuple[str, bool]] = MappingProxyType(
    {
        # attribute holding the history, and whether its element axis comes before its time axis.
        # 力历史属性名，以及 element 轴是否在 time 轴之前。
        "isaacsim": ("net_forces_w_history", False),
        "mjlab": ("force_history", True),
    }
)
"""How each engine spells and shapes its contact force history.

Data rather than a branch: a third engine adds a row, and the reader of ``contact_force_history``
can see the whole cross-engine story in one place instead of tracing two ``if``s.

各引擎接触力历史的拼写与形状。以数据表而非分支表达；第三引擎增一行即可。
"""

_NAMES: MutableMapping[Any, list[str]] = weakref.WeakKeyDictionary()
_IDS: MutableMapping[Any, dict[tuple[str, tuple[str, ...], bool], list[int]]] = weakref.WeakKeyDictionary()


def forget(sensor: Any | None = None) -> None:
    """Drop remembered resolutions, for ``sensor`` or for everything.

    Only needed by tests that reuse one stub sensor while changing what it tracks. Nothing in a
    running environment changes its element list, so nothing in one calls this.

    清除缓存的解析结果（单个 sensor 或全部）。仅测试需要；运行中环境不会改元素列表。
    """
    for cache in (_NAMES, _IDS):
        if sensor is None:
            cache.clear()
        else:
            cache.pop(sensor, None)


def sensor_engine(sensor: Any) -> str:
    """Which engine's contact sensor this is, decided by what it calls its element list.

    通过元素列表属性名 duck typing 判定接触传感器所属引擎。
    """
    for engine, attr in _ELEMENT_NAME_ATTR.items():
        if hasattr(sensor, attr):
            return engine
    raise PortabilityError(
        f"{type(sensor).__name__} exposes neither {' nor '.join(_ELEMENT_NAME_ATTR.values())}, so its "
        "element ordering is unknown. A new engine must be registered in compat.sensors."
    )


def element_names(sensor: Any) -> list[str]:
    """The elements this sensor tracks, ordered as the element axis is ordered.

    Remembered per sensor: on Isaac Lab this reads a property that rebuilds the list from the
    physics view every time it is touched, which is the dominant cost of a step at scale.

    传感器跟踪的元素名，顺序与 element 轴一致。按 sensor 缓存（Isaac 每次访问重建，大规模下是主要开销）。
    """
    try:
        return _NAMES[sensor]
    except (KeyError, TypeError):
        pass
    names = list(getattr(sensor, _ELEMENT_NAME_ATTR[sensor_engine(sensor)]))
    try:
        _NAMES[sensor] = names
    except TypeError:  # sensor cannot be weak-ref'd; still works, just uncached / 无法弱引用则仍可用但不缓存
        pass
    return names


def element_ids(sensor: Any, ref: ContactSensorRef) -> list[int]:
    """Indices on the element axis for the elements ``ref`` names.

    Resolution goes through the sensor's own ``find_bodies``/``find_*`` when it has one, because
    that is what the engine's manager would use; otherwise the patterns are matched here against
    :func:`element_names`. Either path uses the same matching semantics -- the helper behind them
    is the same code in both engines. The answer is remembered; see the module docstring for why
    that is load-bearing rather than tidy.

    Raises:
        PortabilityError: A pattern matched nothing. Silently returning an empty selection would
            turn a foot-contact reward into a constant.

    ``ref`` 所命名元素在 element 轴上的索引。优先走 sensor 自带 ``find_bodies``；否则本地正则匹配。
    结果缓存；见模块文档说明为何这是负载关键而非整洁问题。

    Raises:
        PortabilityError: 模式未匹配任何元素（空选择会把足部接触奖励变成常数）。
    """
    key = (ref.name, ref.elements, ref.preserve_order)
    try:
        return _IDS[sensor][key]
    except (KeyError, TypeError):
        pass

    names = element_names(sensor)
    finder = getattr(sensor, "find_bodies", None)
    if callable(finder):
        ids, _ = finder(list(ref.elements), preserve_order=ref.preserve_order)
    else:
        ids = _match(ref.elements, names, ref.preserve_order)
    if not ids:
        raise PortabilityError(f"Contact sensor {ref.name!r} matched none of {list(ref.elements)}. It tracks {names}.")

    resolved = list(ids)
    try:
        _IDS.setdefault(sensor, {})[key] = resolved
    except TypeError:
        pass
    return resolved


def _match(patterns: tuple[str, ...], names: list[str], preserve_order: bool) -> list[int]:
    """Regex match with the same ordering rule both engines apply.

    正则匹配，排序规则与两引擎一致。
    """
    import re

    if preserve_order:
        ordered: list[int] = []
        for pattern in patterns:
            for index, name in enumerate(names):
                if re.fullmatch(pattern, name) and index not in ordered:
                    ordered.append(index)
        return ordered
    return [index for index, name in enumerate(names) if any(re.fullmatch(p, name) for p in patterns)]


def _timing(sensor: Any, ref: ContactSensorRef, attr: str) -> torch.Tensor:
    tensor = getattr(sensor.data, attr, None)
    if tensor is None:
        raise PortabilityError(
            f"Contact sensor {ref.name!r} has no {attr}; both engines return None for it unless the "
            "sensor was configured to track air time. Set track_air_time on the reference."
        )
    return tensor[:, element_ids(sensor, ref)]


def air_time(sensor: Any, ref: ContactSensorRef) -> torch.Tensor:
    """Seconds each referenced element has been airborne. Shape ``(env, element)``.

    各引用元素当前腾空时长（秒）。形状 ``(env, element)``。
    """
    return _timing(sensor, ref, "current_air_time")


def contact_time(sensor: Any, ref: ContactSensorRef) -> torch.Tensor:
    """Seconds each referenced element has been in contact. Shape ``(env, element)``.

    各引用元素当前接触时长（秒）。形状 ``(env, element)``。
    """
    return _timing(sensor, ref, "current_contact_time")


def ray_hits_w(sensor: Any) -> torch.Tensor:
    """World-frame hit positions, shape ``(env, ray, 3)``. A miss is ``+inf``.

    Isaac Lab already stores misses as ``inf`` in ``ray_hits_w``. mjlab's stock
    sensor stores ``distances < 0`` and leaves ``hit_pos_w`` at the ray origin --
    which, from a sky origin, is a number twenty metres up, and from the ankle is
    a number that looks like a foot height. Either way it is a plausible z, not an
    error. The IR's ``miss="infinity"`` is this function: both engines come out
    here with the same sentinel, and a term that then wrote ``inf → 0`` would be
    putting ground at world height zero on purpose.

    世界系射线命中点，形状 ``(env, ray, 3)``；未命中为 ``+inf``。
    mjlab 未命中时 ``hit_pos_w`` 可能是 plausible 的有限值；此处统一为 ``inf`` 哨兵。
    """
    data = sensor.data
    hits = getattr(data, "ray_hits_w", None)
    if hits is None:
        hits = getattr(data, "hit_pos_w", None)
    if hits is None:
        raise PortabilityError(
            f"{type(sensor).__name__} exposes neither ray_hits_w nor hit_pos_w, so its "
            "ray-caster output is unknown. A new engine must be registered here."
        )
    distances = getattr(data, "distances", None)
    if distances is not None:
        miss = distances < 0.0
        if bool(miss.any()):
            hits = hits.clone()
            hits[miss] = float("inf")
    return hits


def depth_image(sensor: Any) -> torch.Tensor:
    """Current distance-to-image-plane, shape ``(env, H, W, 1)``. A miss is ``+inf``.

    Both engines' camera sensors expose ``data.output['distance_to_image_plane']``.
    Isaac's ``depth_clipping_behavior="none"`` stores a miss as NaN and can also
    return a finite hit past ``max_distance`` (it casts to ``2 * max``). mjlab
    writes +inf past the image-plane far plane. Non-finite values and anything
    past the semantic far plane become +inf here so a term cannot mistake a miss
    for a wall at a finite depth.

    当前像平面距离，形状 ``(env, H, W, 1)``；未命中为 ``+inf``。
    非有限值与超出语义远平面的值在此清洗为 ``+inf``。
    """
    data = sensor.data
    output = getattr(data, "output", None)
    if not isinstance(output, dict) or "distance_to_image_plane" not in output:
        raise PortabilityError(
            f"{type(sensor).__name__} has no data.output['distance_to_image_plane']. "
            "A new engine's camera must register that key."
        )
    image = output["distance_to_image_plane"]
    if image.ndim == 3:
        image = image.unsqueeze(-1)
    if image.ndim != 4 or image.shape[-1] != 1:
        raise PortabilityError(f"depth image should be (env, H, W, 1), got {tuple(image.shape)}.")
    cfg = getattr(sensor, "cfg", None)
    far = getattr(cfg, "image_plane_max", None)
    if far is None:
        far = getattr(cfg, "max_distance", None)
    needs_inf = ~torch.isfinite(image)
    too_far = image > float(far) if far is not None else torch.zeros_like(image, dtype=torch.bool)
    if not bool(needs_inf.any()) and not bool(too_far.any()):
        return image
    cleaned = image.clone()
    cleaned[needs_inf | too_far] = float("inf")
    return cleaned


def registered_cylinder_count(sensor: Any) -> int:
    """How many edge cylinders the sensor was given. Zero is the silent-zero failure.

    传感器已注册的边缘圆柱数量。为零表示静默失效路径。
    """
    count = getattr(sensor, "registered_cylinder_count", None)
    if count is not None:
        return int(count)
    obstacles = getattr(sensor, "_virtual_obstacles", None)
    if not obstacles:
        return 0
    total = 0
    for obstacle in obstacles.values():
        edges = getattr(obstacle, "edges_pyt", None)
        if edges is None:
            cylinders = getattr(obstacle, "cylinders", None)
            if cylinders is None:
                continue
            total += int(getattr(cylinders, "num_cylinders", 0))
            continue
        total += int(edges.shape[0])
    return total


def require_volume_points_registered(sensor: Any) -> None:
    """Refuse the silent-zero path: no event, empty terrain, or generate() found nothing.

    ``penetration_offset`` is zeros when nothing is registered. The reward term
    still exists, logs 0.0, and the robot never learns to avoid edges.

    拒绝静默全零路径：无 startup 事件、空地形或 generate() 未找到边。
    未注册时 ``penetration_offset`` 全零，奖励仍记 0.0，机器人学不到避障。
    """
    name = getattr(getattr(sensor, "cfg", None), "name", None) or type(sensor).__name__
    registered = bool(getattr(sensor, "virtual_obstacles_registered", False))
    obstacles = getattr(sensor, "_virtual_obstacles", None)
    if not registered and not obstacles:
        raise RuntimeError(
            f"volume-points sensor {name!r} has no virtual obstacles registered. "
            "The startup event never ran, or the terrain importer produced an empty set. "
            "penetration_offset is identically zero and the robot will not learn to avoid edges."
        )
    count = registered_cylinder_count(sensor)
    if count <= 0:
        raise RuntimeError(
            f"volume-points sensor {name!r} is registered but has 0 cylinders. "
            "generate() found no edges; the penalty would log a clean 0.0 forever."
        )


def _require_link_point_velocity(sensor: Any) -> None:
    """Refuse a COM ``points_vel_w``. The hub is attach-body link origin.

    Isaac's PhysX view reports COM linear velocity under the same buffer name
    the mjlab sensor uses for link velocity. ω = 0 hides it; a rotating ankle
    scales the same penetration by two different speeds.

    拒绝 COM 版 ``points_vel_w``；中枢为 attach-body link 原点速度。
    ω≠0 时 COM 与 link 速度差会导致同一穿透深度对应不同惩罚。
    """
    cfg = getattr(sensor, "cfg", None)
    velocity = getattr(cfg, "velocity", None)
    if velocity is None:
        velocity = getattr(sensor, "velocity", None)
    if velocity is None or velocity == "attach_link":
        return
    name = getattr(cfg, "name", None) or type(sensor).__name__
    raise PortabilityError(
        f"volume-points sensor {name!r} has velocity={velocity!r}. "
        "Hub point speed is attach-body link origin (v_link + ω × r). "
        "A COM sensor is the denylisted points_vel_w trap; the new-stack "
        "Isaac builder must set velocity='attach_link'."
    )


def volume_points_penetration_offset(sensor: Any) -> torch.Tensor:
    """World-frame offset from the obstacle surface toward each point. Shape ``(N, B, P, 3)``.

    Both engines store this on ``data.penetration_offset``. The vector points from
    the surface to the point: moving along it increases depth. A miss is the zero
    vector. An unregistered sensor is refused here, not returned as zeros.

    世界系：从障碍面向各点的偏移，形状 ``(N, B, P, 3)``。未命中为零向量；未注册 sensor 在此拒绝而非返回零。
    """
    require_volume_points_registered(sensor)
    data = sensor.data
    offset = getattr(data, "penetration_offset", None)
    if offset is None:
        raise PortabilityError(
            f"{type(sensor).__name__} has no data.penetration_offset. "
            "A new engine's volume-points sensor must expose that tensor."
        )
    return offset


def volume_points_vel_w(sensor: Any) -> torch.Tensor:
    """World-frame point velocity. Shape ``(N, B, P, 3)``.

    ``v_link + ω × (p_w - origin_w)`` in the attach-body link frame. A COM
    velocity paired with a link origin is a different number whenever the
    foot's centre of mass is offset — the reward multiplies by this speed.

    世界系点速度，形状 ``(N, B, P, 3)``。attach-body link 原点公式 ``v_link + ω × r``。
    """
    require_volume_points_registered(sensor)
    _require_link_point_velocity(sensor)
    data = sensor.data
    vel = getattr(data, "points_vel_w", None)
    if vel is None:
        raise PortabilityError(
            f"{type(sensor).__name__} has no data.points_vel_w. "
            "A new engine's volume-points sensor must expose that tensor."
        )
    return vel


def in_contact(sensor: Any, ref: ContactSensorRef) -> torch.Tensor:
    """Whether each referenced element is touching something. Shape ``(env, element)``.

    Derived from contact duration rather than from a force threshold, so that each engine's own
    contact criterion is the one that decides -- which is the only way this comes out consistent,
    given that the two report different force quantities.

    各引用元素是否接触。由接触时长导出而非力阈值；各引擎用自己的接触判据。
    """
    return contact_time(sensor, ref) > 0.0


def contact_force_history(sensor: Any, ref: ContactSensorRef) -> torch.Tensor:
    """Force history for the referenced elements, on a common axis order.

    Returns:
        Shape ``(env, time, element, 3)``, newest first along the time axis -- Isaac Lab's layout,
        chosen as the hub because it puts time next to the batch the way the rest of the history
        buffers in this project do. mjlab's ``(env, element, time, 3)`` is transposed to match.

    Warning:
        The *values* are not comparable across engines and this function does not make them so.
        Isaac Lab reports the world-frame normal force alone; mjlab reports the full contact force,
        in the contact frame unless the sensor was configured otherwise. Any threshold on these
        numbers has to be declared per engine, with the tolerance written down.

    引用元素的力历史，统一为 ``(env, time, element, 3)``，time 轴 newest-first。

    Warning:
        **数值** 跨引擎不可比；Isaac 仅法向，mjlab 为完整接触力（默认接触系）。
    """
    ids = element_ids(sensor, ref)
    attr, element_axis_first = _FORCE_HISTORY[sensor_engine(sensor)]
    history = getattr(sensor.data, attr)
    if history is None:
        raise PortabilityError(
            f"Contact sensor {ref.name!r} has no {attr}; both engines return None for it unless the "
            "sensor was given a non-zero history_length."
        )
    if element_axis_first:
        history = history.transpose(1, 2)  # (env, element, time, 3) -> (env, time, element, 3) / 转置统一轴序
    return history[:, :, ids]
