"""Name-resolved left/right augmentation for motion-reference buffers."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import dataclass

from instinctlab.spec.sensor import SymmetricAugmentationSpec
from instinctlab.utils.name_order import NameOrderError, resolve_name_indices

from .buffers import MotionReferenceBuffers


class SymmetricMappingError(ValueError):
    """A name-based mirror map could not be resolved in this joint/link order."""


@dataclass(frozen=True)
class ResolvedSymmetricAugmentation:
    """Integer tables in *this* sensor's joint/link order, never a source repo's."""

    joint_mapping: tuple[int, ...]
    joint_signs: tuple[int, ...]
    link_mapping: tuple[int, ...]
    joint_names: tuple[str, ...]
    link_names: tuple[str, ...]


def resolve_symmetric_augmentation(
    spec: SymmetricAugmentationSpec,
    joint_names: Sequence[str],
    link_names: Sequence[str],
) -> ResolvedSymmetricAugmentation:
    """Turn name maps into indices for ``joint_names`` / ``link_names``.

    The source tables are integers in *their* order. InstinctMJ's G1 table is
    canonical DFS; main parkour's is PhysX BFS. Applying either list to the
    other order (or to the published legs-first npz) does not raise — it
    swaps the wrong joints. Resolution is therefore by name, then indexed
    here.
    """
    joints = tuple(joint_names)
    links = tuple(link_names)
    if set(spec.joint_swaps) != set(joints):
        raise SymmetricMappingError(
            f"symmetric_augmentation joints {sorted(spec.joint_swaps)} do not match "
            f"sensor joints {list(joints)}. An integer table from another order is "
            "not a name map."
        )
    if set(spec.link_swaps) != set(links):
        raise SymmetricMappingError(
            f"symmetric_augmentation links {sorted(spec.link_swaps)} do not match sensor links {list(links)}."
        )
    try:
        joint_mapping = resolve_name_indices(joints, tuple(spec.joint_swaps[name] for name in joints))
        link_mapping = resolve_name_indices(links, tuple(spec.link_swaps[name] for name in links))
    except NameOrderError as exc:
        raise SymmetricMappingError(f"symmetric augmentation cannot be resolved: {exc}") from exc
    return ResolvedSymmetricAugmentation(
        joint_mapping=joint_mapping,
        joint_signs=tuple(int(spec.joint_signs[name]) for name in joints),
        link_mapping=link_mapping,
        joint_names=joints,
        link_names=links,
    )


def augment_joint_buffer(buf: torch.Tensor, mapping: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """``new[i] = old[mapping[i]] * signs[i]``. In place on the last axis."""
    buf[:] = buf[..., mapping] * signs
    return buf


def augment_ang_vel_buffer(buf: torch.Tensor) -> torch.Tensor:
    """Mirror angular velocity across the x-z plane. In place.

    Angular velocity is a pseudovector: a reflection through the x-z plane
    keeps the y component and flips x and z. Both source managers do that
    (``buf[..., 0] *= -1; buf[..., 2] *= -1``). Their docstring says "y and z";
    the docstring is wrong. This function follows the code.
    """
    buf[..., 0] *= -1
    buf[..., 2] *= -1
    return buf


def augment_link_pos_buffer(buf: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """Permute left/right links, then negate y. In place. ``buf`` is ``(..., L, 3)``."""
    buf[:] = buf[..., mapping, :]
    buf[..., 1] *= -1
    return buf


def augment_link_quat_buffer(buf: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """Permute left/right links, then flip quat x and z (wxyz). In place."""
    buf[:] = buf[..., mapping, :]
    buf[..., 1] *= -1
    buf[..., 3] *= -1
    return buf


def augment_quat_buffer(buf: torch.Tensor) -> torch.Tensor:
    """Mirror a wxyz quaternion across the x-z plane. In place."""
    buf[..., 1] *= -1
    buf[..., 3] *= -1
    return buf


def _selected_envs(env_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if env_ids.numel() == 0:
        return env_ids
    return env_ids[mask[env_ids]]


def apply_symmetric_augmentation(
    buffers: MotionReferenceBuffers,
    env_ids: torch.Tensor,
    mask: torch.Tensor,
    resolved: ResolvedSymmetricAugmentation,
) -> None:
    """Mirror every spatial field of ``buffers[env_ids]`` that the sources mirror.

    The same per-env mask is used for joints, base, and links. Time stamps,
    validity, frame indices and exhaustion counters are not spatial and are
    left alone.

    Link *velocities* follow the sources: sign flip only, no ``link_mapping``
    permute. Positions and quaternions of those same links *are* permuted.
    That inconsistency is in both source managers; we match it rather than
    silently "fix" a quantity AMP observations do not currently read.
    """
    selected = _selected_envs(env_ids, mask)
    if selected.numel() == 0:
        return
    device = buffers.joint_pos.device
    dtype = buffers.joint_pos.dtype
    jmap = torch.as_tensor(resolved.joint_mapping, device=device, dtype=torch.long)
    jsign = torch.as_tensor(resolved.joint_signs, device=device, dtype=dtype)
    lmap = torch.as_tensor(resolved.link_mapping, device=device, dtype=torch.long)

    joint_pos = buffers.joint_pos[selected]
    augment_joint_buffer(joint_pos, jmap, jsign)
    buffers.joint_pos[selected] = joint_pos
    joint_vel = buffers.joint_vel[selected]
    augment_joint_buffer(joint_vel, jmap, jsign)
    buffers.joint_vel[selected] = joint_vel

    base_pos = buffers.base_pos_w[selected]
    base_pos[..., 1] *= -1
    buffers.base_pos_w[selected] = base_pos
    base_quat = buffers.base_quat_w[selected]
    augment_quat_buffer(base_quat)
    buffers.base_quat_w[selected] = base_quat
    base_lin = buffers.base_lin_vel_w[selected]
    base_lin[..., 1] *= -1
    buffers.base_lin_vel_w[selected] = base_lin
    base_ang = buffers.base_ang_vel_w[selected]
    augment_ang_vel_buffer(base_ang)
    buffers.base_ang_vel_w[selected] = base_ang

    for name in ("link_pos_w", "link_pos_b"):
        pos = getattr(buffers, name)[selected]
        augment_link_pos_buffer(pos, lmap)
        getattr(buffers, name)[selected] = pos
    for name in ("link_quat_w", "link_quat_b"):
        quat = getattr(buffers, name)[selected]
        augment_link_quat_buffer(quat, lmap)
        getattr(buffers, name)[selected] = quat

    for name in ("link_lin_vel_w", "link_lin_vel_b"):
        lin = getattr(buffers, name)[selected]
        lin[..., 1] *= -1
        getattr(buffers, name)[selected] = lin
    for name in ("link_ang_vel_w", "link_ang_vel_b"):
        ang = getattr(buffers, name)[selected]
        augment_ang_vel_buffer(ang)
        getattr(buffers, name)[selected] = ang


def draw_symmetric_mask(
    mask: torch.Tensor,
    env_ids: torch.Tensor,
    *,
    enabled: bool,
    generator: torch.Generator | None = None,
) -> None:
    """Per-env Bernoulli(1/2), held until the next reset of that env.

    Both source ``AmassMotion.reset`` implementations write
    ``torch.randint(0, 2, (len(env_ids),), dtype=torch.bool)``. Disabled means
    the mask is cleared for those envs, not left stale from a previous on/off.
    """
    if env_ids.numel() == 0:
        return
    if not enabled:
        mask[env_ids] = False
        return
    mask[env_ids] = torch.randint(
        0,
        2,
        (int(env_ids.numel()),),
        device=mask.device,
        dtype=torch.bool,
        generator=generator,
    )
