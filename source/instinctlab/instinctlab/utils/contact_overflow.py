"""Refuse to run on mujoco_warp contact or constraint overflow.

mujoco_warp already sets ``opt.warn_overflow = True`` in ``put_model``. That
only ``wp.printf``s from the kernels. mjlab captures CUDA graphs for ``step``,
and device printf during graph replay is unreliable; even when a print lands
it is a line in a 100k-line log, not a failed run. Contacts past ``nconmax``
and constraints past ``njmax`` are dropped. Training continues on silently
wrong physics and still converges.

This module polls ``d.overflow`` — one device-to-host of ``nworld`` ints.
Measured isolated cost is 24 µs per read at 8 worlds (the array is
kilobytes even at 4096). An ``env.step`` is tens of milliseconds and already
synchronises for observations, so the extra read is in the noise (~0.08% of
a 30 ms step). Cadence is therefore every environment step, not a sample:
a run that dies at iteration 900 with a nameable overflow is better than
one that finishes on dropped contacts.

The kernels set the same bits this reads. A guard that cannot be shown to
fire on those bits is not a guard, so it was shown: parkour on mjlab at
``horizontal_scale=0.05`` (16 envs) constructs clean and raises ``HFIELD``
on the first step, while the same probe at the shipped 0.07 is clean over
150 steps. That overflow is real physics being dropped, not a synthetic bit.

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


class ContactOverflowError(RuntimeError):
    """mujoco_warp dropped contacts or constraints. Physics is no longer the model."""

    def __init__(self, message: str, snapshot: dict[str, Any]):
        super().__init__(message)
        self.snapshot = snapshot


def overflow_allowed() -> bool:
    return os.environ.get(ALLOW_ENV, "").strip().lower() in _ALLOWED_TRUTHY


def contact_budget_snapshot(env: Any) -> dict[str, Any] | None:
    """Read overflow / nacon / nefc, or ``None`` if this env is not mujoco_warp."""
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


def overflow_bits_set(env: Any) -> bool:
    """Hot-path read: only the overflow bitmask, no nacon/nefc."""
    raw = getattr(env, "unwrapped", env)
    wp_data = getattr(getattr(raw, "sim", None), "wp_data", None)
    if wp_data is None or not hasattr(wp_data, "overflow"):
        return False
    overflow = _as_numpy(wp_data.overflow)
    return bool(np.any(overflow))


def check_contact_overflow(env: Any, *, phase: str) -> dict[str, Any] | None:
    """Raise :class:`ContactOverflowError` if any world has an overflow bit.

    Returns the snapshot when the env is mujoco_warp and clean; ``None`` when
    the env has no ``wp_data.overflow`` (Isaac, stubs). ``phase`` is
    ``\"construction\"`` or ``\"step\"`` and is written into the message.
    """
    if not overflow_bits_set(env):
        return None

    snapshot = contact_budget_snapshot(env)
    assert snapshot is not None
    message = format_overflow_message(snapshot, phase=phase)
    if overflow_allowed():
        _warn_allowed(message)
        return snapshot
    raise ContactOverflowError(message, snapshot)


def format_overflow_message(snapshot: dict[str, Any], *, phase: str) -> str:
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


def attach_overflow_guard(env: Any) -> Any:
    """Wrap ``step`` so overflow is checked after every environment step.

    No-op when the env has no ``wp_data.overflow``. The mjlab env class also
    checks itself; this wrapper is for VecEnv stacks that replace ``step``.
    """
    raw = getattr(env, "unwrapped", env)
    wp_data = getattr(getattr(raw, "sim", None), "wp_data", None)
    if wp_data is None or not hasattr(wp_data, "overflow"):
        return env
    return OverflowGuardVecEnv(env)


class OverflowGuardVecEnv:
    """Delegating VecEnv that fails after a step whose overflow bits are set."""

    def __init__(self, env: Any):
        object.__setattr__(self, "env", env)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

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
