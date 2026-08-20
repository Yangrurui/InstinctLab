"""Refuse to run when the engine dropped contacts and kept stepping.

mujoco_warp already sets ``opt.warn_overflow = True`` in ``put_model``. That
only ``wp.printf``s from the kernels. mjlab captures CUDA graphs for ``step``,
and device printf during graph replay is unreliable; even when a print lands
it is a line in a 100k-line log, not a failed run. Contacts past ``nconmax``
and constraints past ``njmax`` are dropped. Training continues on silently
wrong physics and still converges.

PhysX GPU is the same shape with a louder log. Isaac Sim 5.1 / omni.physx
107.3 still cannot grow ``gpu_collision_stack_size``. Overflow writes
``PxGpuDynamicsMemoryConfig::collisionStackSize buffer overflow detected
... Contacts have been dropped.`` to carb, then ``env.step`` returns
successfully. Measured 2026-08-20 on cuda:1: parkour, 16 envs, stack
``2**14`` (16 KiB). ``PhysicsSceneStats.gpu_mem_collision_stack_size``
reported 813424 bytes needed — the same number the error asked for —
while the USD attribute stayed 16384. Contact-sensor force went to 0.
The shipped ``2**29`` run on the same task needed 912488 bytes at 16
envs and 18319520 bytes at 256 envs (linear to ~293 MiB at 4096, above
Isaac Lab's ``2**26`` default and below main's ``2**29``).

mjlab side polls ``d.overflow`` (24 µs at 8 worlds). Isaac side polls
``IPhysxStatistics.get_physx_scene_statistics`` and compares needed vs
allocated. A guard that cannot be shown to fire on a real overflow is
not a guard: mjlab was shown on HFIELD at ``horizontal_scale=0.05``;
Isaac was shown on the 16 KiB stack above.

Not every overflow bit is a budget of ours. ``EPA_HORIZON`` is a fixed
upstream scratch buffer inside one contact pair's penetration refinement
(``MJ_MAX_EPAHORIZON = 24``), not a model option; see ``_UNTUNABLE_FLAGS``
for why it is counted rather than fatal, and what makes it fatal anyway.

Escape hatch (not the default): ``INSTINCTLAB_ALLOW_CONTACT_OVERFLOW=1``, or
``--allow_contact_overflow`` on ``scripts/train.py``.
"""

from __future__ import annotations

import numpy as np
import os
from typing import Any

__all__ = [
    "ALLOW_ENV",
    "ContactOverflowError",
    "attach_overflow_guard",
    "check_contact_overflow",
    "contact_budget_snapshot",
    "overflow_allowed",
]

ALLOW_ENV = "INSTINCTLAB_ALLOW_CONTACT_OVERFLOW"

_ALLOWED_TRUTHY = frozenset({"1", "true", "yes", "on"})
_warned_allow = False

# HFIELD overflow is a compile-time pair cap, not nconmax/njmax.
_MJ_MAX_CON_PAIR = 50

# EPA_HORIZON is the one overflow bit that is not a budget of ours. The others say we
# allocated too little and are now dropping contacts or constraints every step, which is
# both systematic and fixable by raising a number. This one says a single contact pair's
# penetration refinement ran out of a scratch buffer whose size is the upstream literal
# ``MJ_MAX_EPAHORIZON = 24`` -- not a model option, nothing on our side to raise. EPA bails
# on that pair for that step; the rest of the world is unaffected.
#
# Treating them alike cost a run: one such event, in 1 of 256 worlds, in one step out of
# ~3.3M world-steps, killed a 700-iteration job at iteration 545. A guard that aborts on a
# 3e-7 event nobody can fix teaches people to pass --allow_contact_overflow, which switches
# off the budget checks too -- the guard would have disabled itself.
#
# So it is counted rather than fatal, and only up to a point: a *rate* this high is a
# different claim from a one-off, and the ceiling is set far above anything numerical
# noise produces so that crossing it means something real.
_UNTUNABLE_FLAGS = frozenset({"EPA_HORIZON"})
_UNTUNABLE_FATAL_RATE = 0.01
_UNTUNABLE_MIN_SAMPLES = 200
_UNTUNABLE_STATE_ATTR = "_instinctlab_untunable_overflow"


class ContactOverflowError(RuntimeError):
    """The engine dropped contacts or constraints. Physics is no longer the model."""

    def __init__(self, message: str, snapshot: dict[str, Any]):
        super().__init__(message)
        self.snapshot = snapshot


def overflow_allowed() -> bool:
    return os.environ.get(ALLOW_ENV, "").strip().lower() in _ALLOWED_TRUTHY


def contact_budget_snapshot(env: Any) -> dict[str, Any] | None:
    """Read the engine's overflow / budget snapshot, or ``None`` if neither applies."""
    mjlab = _mjlab_snapshot(env)
    if mjlab is not None:
        return mjlab
    return _isaac_snapshot(env)


def overflow_bits_set(env: Any) -> bool:
    """Hot-path read: only the mujoco_warp overflow bitmask, no nacon/nefc."""
    raw = getattr(env, "unwrapped", env)
    wp_data = getattr(getattr(raw, "sim", None), "wp_data", None)
    if wp_data is None or not hasattr(wp_data, "overflow"):
        return False
    overflow = _as_numpy(wp_data.overflow)
    return bool(np.any(overflow))


def check_contact_overflow(env: Any, *, phase: str) -> dict[str, Any] | None:
    """Raise :class:`ContactOverflowError` if this env dropped contacts.

    Returns the snapshot when the env overflowed and the escape hatch is
    set; ``None`` when the env is clean or has no overflow signal. ``phase``
    is ``\"construction\"`` or ``\"step\"`` and is written into the message.
    """
    raw = getattr(env, "unwrapped", env)
    state = getattr(raw, _UNTUNABLE_STATE_ATTR, None)
    if state is None:
        state = {"steps": 0, "events": 0, "warned": False}
        try:
            setattr(raw, _UNTUNABLE_STATE_ATTR, state)
        except AttributeError:  # slotted env; the rate ceiling degrades to per-call
            pass
    state["steps"] += 1

    snapshot = None
    if overflow_bits_set(env):
        snapshot = _mjlab_snapshot(env)
    else:
        isaac = _isaac_snapshot(env)
        if isaac is not None and isaac["any_overflow"]:
            snapshot = isaac
    if snapshot is None:
        return None

    flags = set(snapshot.get("overflow_flags") or ())
    untunable_only = bool(flags) and flags <= _UNTUNABLE_FLAGS
    if untunable_only:
        state["events"] += 1
        rate = state["events"] / max(state["steps"], 1)
        snapshot = {**snapshot, "untunable_events": state["events"], "untunable_rate": rate}
        noisy = state["steps"] >= _UNTUNABLE_MIN_SAMPLES and rate > _UNTUNABLE_FATAL_RATE
        message = _format_untunable_message(snapshot, phase=phase, fatal=noisy)
        if not noisy:
            if not state["warned"]:
                state["warned"] = True
                print(f"[instinctlab] {message}", flush=True)
            return snapshot
        if overflow_allowed():
            _warn_allowed(message)
            return snapshot
        raise ContactOverflowError(message, snapshot)

    message = format_overflow_message(snapshot, phase=phase)
    if overflow_allowed():
        _warn_allowed(message)
        return snapshot
    raise ContactOverflowError(message, snapshot)


def _format_untunable_message(snapshot: dict[str, Any], *, phase: str, fatal: bool) -> str:
    flags = ", ".join(snapshot.get("overflow_flags") or ())
    events, steps_rate = snapshot["untunable_events"], snapshot["untunable_rate"]
    head = (
        f"mujoco_warp {phase} overflow: {flags} in "
        f"{snapshot.get('worlds_with_overflow')} of {snapshot.get('nworld')} worlds "
        f"({events} checked steps affected, {steps_rate:.2%} of them). "
        "EPA ran out of its horizon buffer while refining one contact pair and returned "
        "that pair unrefined; the buffer is the upstream constant MJ_MAX_EPAHORIZON=24, "
        "so no nconmax / njmax change affects it."
    )
    if not fatal:
        return head + " Rare enough to be numerical, so this is a note, not a failure; the rate is watched."
    return (
        head
        + f" That is past {_UNTUNABLE_FATAL_RATE:.0%} of steps, which is no longer a numerical "
        "one-off -- deep or degenerate penetration is now routine, and contact depths "
        f"are unreliable. To proceed anyway set {ALLOW_ENV}=1 or pass --allow_contact_overflow."
    )


def format_overflow_message(snapshot: dict[str, Any], *, phase: str) -> str:
    if snapshot.get("engine") == "isaacsim":
        return _format_isaac_message(snapshot, phase=phase)
    return _format_mjlab_message(snapshot, phase=phase)


def _format_mjlab_message(snapshot: dict[str, Any], *, phase: str) -> str:
    flags = snapshot.get("overflow_flags") or [f"bits=0x{int(snapshot.get('overflow_max', 0)):x}"]
    nworld = snapshot.get("nworld")
    n_set = snapshot.get("worlds_with_overflow")
    nconmax = snapshot.get("nconmax")
    njmax = snapshot.get("njmax")
    nacon = snapshot.get("nacon")
    nefc_max = snapshot.get("nefc_max")

    nacon_line = "nacon unavailable"
    if nacon is not None and nconmax and nworld:
        budget = int(nconmax) * int(nworld)
        nacon_line = (
            f"nconmax={nconmax} (per world), observed nacon={nacon} "
            f"({nacon / int(nworld):.1f}/world, {nacon / budget:.1%} of nconmax*nworld={budget})"
        )
    elif nconmax is not None:
        nacon_line = f"nconmax={nconmax}, observed nacon={nacon}"

    nefc_line = "nefc unavailable"
    if nefc_max is not None and njmax:
        nefc_line = f"njmax={njmax} (per world), observed nefc_max={nefc_max} ({nefc_max / int(njmax):.1%} of njmax)"

    extra = ""
    if "HFIELD" in flags:
        extra = (
            f" HFIELD is a compile-time cap (mujoco mjMAXCONPAIR={_MJ_MAX_CON_PAIR} "
            "contacts per geom pair); raising nconmax does not clear it."
        )

    return (
        f"mujoco_warp {phase} overflow: {', '.join(flags)}. "
        f"{n_set} of {nworld} worlds have overflow bits set. "
        f"{nacon_line}. {nefc_line}. "
        "Contacts past nconmax and constraints past njmax are dropped; "
        "the step succeeds and training will converge on truncated physics. "
        "Raise sim.nconmax / sim.njmax (per-world allocations, GPU memory "
        f"scales with num_envs).{extra} "
        f"To proceed anyway set {ALLOW_ENV}=1 or pass --allow_contact_overflow."
    )


def _format_isaac_message(snapshot: dict[str, Any], *, phase: str) -> str:
    reasons = snapshot.get("overflow_reasons") or ["PhysX GPU buffer"]
    stack_needed = snapshot.get("gpu_mem_collision_stack_size")
    stack_alloc = snapshot.get("gpu_collision_stack_size")
    patch_needed = snapshot.get("gpu_mem_rigid_patch_count")
    patch_alloc = snapshot.get("gpu_max_rigid_patch_count")
    stack_line = (
        f"collision stack needed {stack_needed} bytes vs allocated {stack_alloc}"
        if stack_needed is not None and stack_alloc is not None
        else "collision stack occupancy unavailable"
    )
    patch_line = (
        f"rigid patches needed {patch_needed} vs allocated {patch_alloc}"
        if patch_needed is not None and patch_alloc is not None
        else "rigid patch occupancy unavailable"
    )
    return (
        f"PhysX {phase} overflow: {', '.join(reasons)}. {stack_line}. {patch_line}. "
        "GPU PhysX cannot grow these buffers; it logs a PhysX error, drops "
        "contacts, and the step still succeeds, so training will converge on "
        "truncated physics. Raise sim.physx.gpu_collision_stack_size "
        "(main parkour uses 2**29) and/or gpu_max_rigid_patch_count "
        f"(main parkour uses 10 * 2**15). To proceed anyway set {ALLOW_ENV}=1 "
        "or pass --allow_contact_overflow."
    )


def _mjlab_snapshot(env: Any) -> dict[str, Any] | None:
    raw = getattr(env, "unwrapped", env)
    sim = getattr(raw, "sim", None)
    wp_data = getattr(sim, "wp_data", None)
    if wp_data is None or not hasattr(wp_data, "overflow"):
        return None

    overflow = _as_numpy(wp_data.overflow).astype(np.int64, copy=False)
    nworld = int(overflow.size)
    mask = int(overflow.max()) if nworld else 0
    n_set = int(np.count_nonzero(overflow))
    cfg = getattr(sim, "cfg", None)
    nconmax = getattr(cfg, "nconmax", None) if cfg is not None else None
    njmax = getattr(cfg, "njmax", None) if cfg is not None else None

    result: dict[str, Any] = {
        "available": True,
        "engine": "mjlab",
        "nworld": nworld,
        "worlds_with_overflow": n_set,
        "any_overflow": n_set > 0,
        "overflow_max": mask,
        "overflow_flags": decode_overflow(mask),
        "nconmax": nconmax,
        "njmax": njmax,
    }
    nacon = getattr(wp_data, "nacon", None)
    if nacon is not None:
        nacon_v = _as_numpy(nacon)
        result["nacon"] = int(nacon_v.reshape(-1)[0]) if nacon_v.size else None
    nefc = getattr(wp_data, "nefc", None)
    if nefc is not None:
        nefc_v = _as_numpy(nefc)
        result["nefc_max"] = int(nefc_v.max()) if nefc_v.size else None
        result["nefc_mean"] = float(nefc_v.mean()) if nefc_v.size else None
    return result


def _isaac_physx_cfg(env: Any) -> Any | None:
    raw = getattr(env, "unwrapped", env)
    sim_cfg = getattr(getattr(raw, "cfg", None), "sim", None)
    physx = getattr(sim_cfg, "physx", None)
    if physx is None or not hasattr(physx, "gpu_collision_stack_size"):
        return None
    device = str(getattr(sim_cfg, "device", "") or getattr(raw, "device", "") or "")
    if device and not device.startswith("cuda"):
        return None
    return raw


def _read_isaac_stats(raw: Any) -> Any | None:
    override = getattr(raw, "_physx_scene_stats", None)
    if override is not None:
        return override
    try:
        import omni.physx
        from omni.physx.bindings._physx import PhysicsSceneStats
        from pxr import PhysicsSchemaTools, UsdUtils
    except ImportError:
        return None
    sim = getattr(raw, "sim", None)
    stage = getattr(sim, "stage", None)
    if stage is None:
        return None
    sim_cfg = raw.cfg.sim
    stats = PhysicsSceneStats()
    try:
        ok = omni.physx.get_physx_statistics_interface().get_physx_scene_statistics(
            UsdUtils.StageCache.Get().GetId(stage).ToLongInt(),
            PhysicsSchemaTools.sdfPathToInt(sim_cfg.physics_prim_path),
            stats,
        )
    except Exception:
        return None
    return stats if ok else None


def _isaac_snapshot(env: Any) -> dict[str, Any] | None:
    raw = _isaac_physx_cfg(env)
    if raw is None:
        return None
    physx = raw.cfg.sim.physx
    stats = _read_isaac_stats(raw)
    if stats is None:
        return None
    allocated_stack = int(physx.gpu_collision_stack_size)
    allocated_patches = int(getattr(physx, "gpu_max_rigid_patch_count", 0) or 0)
    needed_stack = int(getattr(stats, "gpu_mem_collision_stack_size", 0) or 0)
    needed_patches = int(getattr(stats, "gpu_mem_rigid_patch_count", 0) or 0)
    reasons: list[str] = []
    if allocated_stack and needed_stack > allocated_stack:
        reasons.append("collision stack")
    if allocated_patches and needed_patches > allocated_patches:
        reasons.append("rigid patch count")
    return {
        "available": True,
        "engine": "isaacsim",
        "any_overflow": bool(reasons),
        "overflow_reasons": reasons,
        "gpu_collision_stack_size": allocated_stack,
        "gpu_mem_collision_stack_size": needed_stack,
        "gpu_max_rigid_patch_count": allocated_patches,
        "gpu_mem_rigid_patch_count": needed_patches,
        "gpu_mem_rigid_contact_count": int(getattr(stats, "gpu_mem_rigid_contact_count", 0) or 0),
    }


def attach_overflow_guard(env: Any) -> Any:
    """Wrap ``step`` so overflow is checked after every environment step.

    No-op when the env has neither mujoco_warp ``d.overflow`` nor an Isaac
    PhysX GPU budget. The mjlab env class also checks itself; this wrapper
    is for VecEnv stacks that replace ``step``.
    """
    raw = getattr(env, "unwrapped", env)
    wp_data = getattr(getattr(raw, "sim", None), "wp_data", None)
    if wp_data is not None and hasattr(wp_data, "overflow"):
        return OverflowGuardVecEnv(env)
    if _isaac_physx_cfg(env) is not None:
        return OverflowGuardVecEnv(env)
    return env


class OverflowGuardVecEnv:
    """Delegating VecEnv that fails after a step whose overflow bits are set."""

    def __init__(self, env: Any):
        object.__setattr__(self, "env", env)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value) -> None:
        self.env.episode_length_buf = value

    def step(self, actions):
        result = self.env.step(actions)
        check_contact_overflow(self.env, phase="step")
        return result


def decode_overflow(mask: int) -> list[str]:
    if mask == 0:
        return []
    try:
        from mujoco_warp._src.types import OverflowType
    except ImportError:
        return [f"bits=0x{mask:x}"]
    return [flag.name for flag in OverflowType if mask & int(flag)]


def _as_numpy(arr: Any) -> np.ndarray:
    if hasattr(arr, "numpy"):
        out = arr.numpy()
        return out if isinstance(out, np.ndarray) else np.asarray(out)
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _warn_allowed(message: str) -> None:
    global _warned_allow
    if _warned_allow:
        return
    _warned_allow = True
    print(f"[WARN] {ALLOW_ENV} is set; contact overflow will not stop the run.\n{message}")
