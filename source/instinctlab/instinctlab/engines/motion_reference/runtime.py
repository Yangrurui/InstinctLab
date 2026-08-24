"""Shared motion-reference state machine used by every engine sensor."""

from __future__ import annotations

import torch

from instinctlab.spec.sensor import MotionReferenceRef

from .buffers import (
    MotionReferenceBuffers,
    envs_due_for_update,
    fill_buffers,
    lookahead_times,
    make_buffers,
    translate_world_positions,
)
from .clip import MotionClip, load_retargetted_clip, pack_motion_clip, sample_clip
from .symmetry import (
    ResolvedSymmetricAugmentation,
    apply_symmetric_augmentation,
    draw_symmetric_mask,
    resolve_symmetric_augmentation,
)


class MotionReferenceRuntime:
    """Clip clock, look-ahead fill, and the per-env mirror mask.

    Both engine sensors hold one of these. Reset draws the Bernoulli mask and
    the start time; every refresh writes the raw sample then mirrors the
    masked envs. Applying after fill (not onto the previous output) is what
    stops a second update from double-mirroring.
    """

    def __init__(
        self,
        ref: MotionReferenceRef,
        clip: MotionClip,
        buffers: MotionReferenceBuffers,
        env_origins: torch.Tensor,
        last_update: torch.Tensor,
        mask: torch.Tensor,
        resolved: ResolvedSymmetricAugmentation | None,
    ) -> None:
        self.ref = ref
        self.clip = clip
        self.buffers = buffers
        self.env_origins = env_origins
        self.last_update = last_update
        self.mask = mask
        self.resolved = resolved

    @classmethod
    def from_clip(
        cls,
        ref: MotionReferenceRef,
        clip: MotionClip,
        num_envs: int,
        device: torch.device | str = "cpu",
    ) -> MotionReferenceRuntime:
        joints = tuple(ref.joints)
        links = tuple(ref.links)
        buffers = make_buffers(num_envs, ref.num_frames, len(joints), len(links), device=device)
        resolved = None
        if ref.symmetric_augmentation is not None:
            resolved = resolve_symmetric_augmentation(ref.symmetric_augmentation, joints, links)
        return cls(
            ref=ref,
            clip=clip,
            buffers=buffers,
            env_origins=torch.zeros(num_envs, 3, device=device),
            last_update=torch.zeros(num_envs, device=device),
            mask=torch.zeros(num_envs, dtype=torch.bool, device=device),
            resolved=resolved,
        )

    @classmethod
    def create(
        cls,
        ref: MotionReferenceRef,
        model_path: str,
        num_envs: int,
        device: torch.device | str = "cpu",
    ) -> MotionReferenceRuntime:
        raw = load_retargetted_clip(ref.clip, device=device)
        clip = pack_motion_clip(
            raw,
            joint_names=tuple(ref.joints),
            link_names=tuple(ref.links),
            model_path=model_path,
            velocity_method=ref.velocity_method,
            target_fps=ref.clip_target_fps,
            device=device,
        )
        return cls.from_clip(ref, clip, num_envs, device)

    @property
    def enabled(self) -> bool:
        return self.resolved is not None

    @property
    def aiming_frame_idx(self) -> torch.Tensor:
        """Current target slot, using the source managers' strict time comparison."""
        elapsed = self.buffers.timestamp - self.last_update
        aiming = torch.sum(
            torch.logical_and(
                elapsed.unsqueeze(-1) > self.buffers.time_to_target_frame,
                self.buffers.time_to_target_frame > 0.0,
            ),
            dim=-1,
        )
        aiming[aiming >= self.ref.num_frames] = -1
        return aiming

    def bind_origins(self, origins: torch.Tensor) -> None:
        if origins.shape != self.env_origins.shape:
            raise ValueError(
                f"motion-reference origins must have shape {tuple(self.env_origins.shape)}, got {tuple(origins.shape)}."
            )
        delta = origins - self.env_origins
        self.buffers.base_pos_w += delta.unsqueeze(1)
        self.buffers.link_pos_w += delta.unsqueeze(1).unsqueeze(1)
        self.env_origins = origins

    def reset(self, env_ids: torch.Tensor, generator: torch.Generator | None = None) -> None:
        lo, hi = self.ref.start_range
        span = (
            torch.rand(
                int(env_ids.numel()),
                device=self.buffers.timestamp.device,
                generator=generator,
            )
            * (hi - lo)
            + lo
        )
        self.buffers.start_s[env_ids] = span * self.clip.sampling_length_s
        self.buffers.timestamp[env_ids] = 0.0
        self.last_update[env_ids] = 0.0
        draw_symmetric_mask(self.mask, env_ids, enabled=self.enabled, generator=generator)
        self.refresh_at_current_time(env_ids)

    def refresh(self, env_ids: torch.Tensor) -> None:
        """Rebuild selected buffers without changing their clock bookkeeping."""
        times, time_to = lookahead_times(
            self.buffers.timestamp[env_ids],
            self.buffers.start_s[env_ids],
            self.ref.num_frames,
            self.ref.frame_interval_s,
            self.ref.data_start_from,
        )
        fill_buffers(
            self.buffers,
            env_ids,
            sample_clip(self.clip, times),
            time_to,
        )
        if self.resolved is not None:
            apply_symmetric_augmentation(self.buffers, env_ids, self.mask, self.resolved)
        translate_world_positions(self.buffers, env_ids, self.env_origins)

    def refresh_at_current_time(self, env_ids: torch.Tensor) -> None:
        """Rebuild selected buffers and mark their current timestamp as sampled."""
        self.refresh(env_ids)
        self.last_update[env_ids] = self.buffers.timestamp[env_ids]

    def advance(self, dt: float) -> torch.Tensor:
        self.buffers.timestamp = self.buffers.timestamp + dt
        due = envs_due_for_update(self.buffers.timestamp, self.last_update, self.ref.update_period)
        if due.numel() == 0:
            return due
        self.refresh_at_current_time(due)
        return due


def bind_motion_reference_origins(scene, references: tuple[MotionReferenceRef, ...]) -> None:
    """Bind every declared motion sensor to its live scene's environment origins."""
    for ref in references:
        scene.sensors[ref.name].bind_origins(scene.env_origins)
