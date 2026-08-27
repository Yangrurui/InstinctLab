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
from .clip import (
    MotionClip,
    load_retargetted_clip,
    pack_motion_clip,
    pack_motion_clips_for_sampling,
    sample_clip,
    sample_packed_motion_clips,
)
from .inventory import MotionFrameInventory, MotionInventoryEntry, discover_motion_inventory
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
        clips: tuple[MotionClip, ...],
        inventory: tuple[MotionInventoryEntry, ...],
        buffers: MotionReferenceBuffers,
        init_buffers: MotionReferenceBuffers,
        reference_buffers: MotionReferenceBuffers,
        env_origins: torch.Tensor,
        last_update: torch.Tensor,
        reference_timestamp: torch.Tensor,
        mask: torch.Tensor,
        resolved: ResolvedSymmetricAugmentation | None,
    ) -> None:
        self.ref = ref
        if not clips or len(clips) != len(inventory):
            raise ValueError("Motion runtime needs one non-empty inventory entry per packed clip.")
        self.clips = clips
        self.inventory = inventory
        # Compatibility for single-clip AMP users. Dataset code must use ``clips[motion_id]``.
        self.clip = clips[0]
        self.buffers = buffers
        self.init_buffers = init_buffers
        self.reference_buffers = reference_buffers
        self.env_origins = env_origins
        self.last_update = last_update
        self._reference_timestamp = reference_timestamp
        self.mask = mask
        self.resolved = resolved
        self._packed_clips = None
        if not buffers.scene_object_names and all(clip.object_name is None for clip in clips):
            self._packed_clips = pack_motion_clips_for_sampling(clips)
        self._bin_counts_host = tuple(self._num_bins(clip) for clip in clips)
        bin_offsets = [0]
        for count in self._bin_counts_host:
            bin_offsets.append(bin_offsets[-1] + count)
        self._bin_offsets_host = tuple(bin_offsets)
        self._bin_counts = torch.tensor(
            self._bin_counts_host,
            dtype=torch.long,
            device=buffers.timestamp.device,
        )
        self._bin_offsets = torch.tensor(
            self._bin_offsets_host,
            dtype=torch.long,
            device=buffers.timestamp.device,
        )
        total_bins = self._bin_offsets_host[-1]
        self.motion_bin_weights = torch.ones(total_bins, device=buffers.timestamp.device)
        self.motion_bin_fail_counter = torch.zeros_like(self.motion_bin_weights)
        self.current_motion_bin_fail_counter = torch.zeros_like(self.motion_bin_weights)
        self._motion_origins: tuple[torch.Tensor, ...] | None = None

    def _num_bins(self, clip: MotionClip) -> int:
        if self.ref.motion_bin_length_s is None:
            return 1
        bins = int(clip.sampling_length_s // self.ref.motion_bin_length_s)
        if bins < 1:
            raise ValueError(f"Motion {clip.path!r} is shorter than one {self.ref.motion_bin_length_s}s sampling bin.")
        return bins

    @classmethod
    def from_clip(
        cls,
        ref: MotionReferenceRef,
        clip: MotionClip,
        num_envs: int,
        device: torch.device | str = "cpu",
    ) -> MotionReferenceRuntime:
        return cls.from_clips(ref, (clip,), (MotionInventoryEntry(clip.path),), num_envs, device)

    @classmethod
    def from_clips(
        cls,
        ref: MotionReferenceRef,
        clips: tuple[MotionClip, ...],
        inventory: tuple[MotionInventoryEntry, ...],
        num_envs: int,
        device: torch.device | str = "cpu",
    ) -> MotionReferenceRuntime:
        joints = tuple(ref.joints)
        links = tuple(ref.links)
        buffers = make_buffers(
            num_envs,
            ref.num_frames,
            len(joints),
            len(links),
            tuple(ref.scene_objects),
            device=device,
        )
        init_buffers = make_buffers(
            num_envs,
            1,
            len(joints),
            len(links),
            tuple(ref.scene_objects),
            device=device,
        )
        reference_buffers = make_buffers(
            num_envs,
            1,
            len(joints),
            len(links),
            tuple(ref.scene_objects),
            device=device,
        )
        resolved = None
        if ref.symmetric_augmentation is not None:
            resolved = resolve_symmetric_augmentation(ref.symmetric_augmentation, joints, links)
        return cls(
            ref=ref,
            clips=clips,
            inventory=inventory,
            buffers=buffers,
            init_buffers=init_buffers,
            reference_buffers=reference_buffers,
            env_origins=torch.zeros(num_envs, 3, device=device),
            last_update=torch.zeros(num_envs, device=device),
            reference_timestamp=torch.full((num_envs,), -1.0, device=device),
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
        inventory = discover_motion_inventory(ref)
        clips = tuple(
            pack_motion_clip(
                load_retargetted_clip(entry.path, device=device),
                joint_names=tuple(ref.joints),
                link_names=tuple(ref.links),
                model_path=model_path,
                velocity_method=ref.velocity_method,
                target_fps=ref.clip_target_fps,
                device=device,
            )
            for entry in inventory
        )
        return cls.from_clips(ref, clips, inventory, num_envs, device)

    @property
    def enabled(self) -> bool:
        return self.resolved is not None

    @property
    def frame_inventory(self) -> tuple[MotionFrameInventory, ...]:
        return tuple(
            MotionFrameInventory(clip.path, clip.nframes, clip.framerate, clip.duration_s) for clip in self.clips
        )

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

    @property
    def reference_frame(self) -> MotionReferenceBuffers:
        """Motion state eagerly sampled at the current clip time for AMP."""
        return self.reference_buffers

    def bind_origins(self, origins: torch.Tensor) -> None:
        if origins.shape != self.env_origins.shape:
            raise ValueError(
                f"motion-reference origins must have shape {tuple(self.env_origins.shape)}, got {tuple(origins.shape)}."
            )
        delta = origins - self.env_origins
        env_ids = torch.arange(origins.shape[0], device=origins.device)
        for buffers in (self.buffers, self.init_buffers, self.reference_buffers):
            translate_world_positions(buffers, env_ids, delta)
        self.env_origins = origins

    def match_terrain_origins(self, terrain: object, *, max_origins_per_motion: int = 49) -> None:
        """Bind each terrain motion to origins containing its declared terrain mesh."""
        origins = torch.as_tensor(terrain.terrain_origins, device=self.env_origins.device, dtype=self.env_origins.dtype)
        cfgs = terrain.subterrain_specific_cfgs
        if cfgs is None or origins.ndim != 3 or len(cfgs) != origins.shape[0] * origins.shape[1]:
            raise ValueError("Motion-matched terrain does not expose one subterrain config per origin.")
        terrain_ids = [entry.terrain_id for entry in self.inventory]
        if any(terrain_id is None for terrain_id in terrain_ids):
            raise ValueError("A terrain motion inventory entry has no terrain_id.")
        num_terrains = max(int(terrain_id) for terrain_id in terrain_ids) + 1
        by_terrain: dict[int, list[torch.Tensor]] = {terrain_id: [] for terrain_id in range(num_terrains)}
        for row in range(origins.shape[0]):
            for col in range(origins.shape[1]):
                cfg = cfgs[row * origins.shape[1] + col]
                if cfg is None or not hasattr(cfg, "difficulty"):
                    raise ValueError(f"Motion-matched terrain cell ({row}, {col}) has no resolved difficulty.")
                terrain_id = min(max(int(float(cfg.difficulty) * num_terrains), 0), num_terrains - 1)
                by_terrain[terrain_id].append(origins[row, col])
        motion_origins = []
        for terrain_id in terrain_ids:
            available = by_terrain[int(terrain_id)]
            if not available:
                raise ValueError(f"Terrain motion {terrain_id} has no compatible scene origin.")
            stacked = torch.stack(available)
            motion_origins.append(stacked[:max_origins_per_motion])
        self._motion_origins = tuple(motion_origins)

    def _sample_motion_origins(self, env_ids: torch.Tensor, generator: torch.Generator | None) -> None:
        if self._motion_origins is None:
            return
        random = torch.rand(env_ids.numel(), device=env_ids.device, generator=generator)
        for motion_id, origins in enumerate(self._motion_origins):
            mask = self.buffers.motion_id[env_ids] == motion_id
            selected = env_ids[mask]
            if selected.numel() == 0:
                continue
            indexes = torch.floor(random[mask] * origins.shape[0]).long()
            self.env_origins[selected] = origins[indexes]

    def reset(self, env_ids: torch.Tensor, generator: torch.Generator | None = None) -> None:
        if env_ids.numel() == 0:
            return
        device = self.buffers.timestamp.device
        clip_weights = torch.tensor(
            [entry.weight for entry in self.inventory],
            dtype=torch.float32,
            device=device,
        )
        sampled_bins = torch.empty(env_ids.numel(), dtype=torch.long, device=device)
        if self.ref.sampling_strategy == "concat_motion_bins" and self.ref.motion_bin_length_s is not None:
            global_bin = torch.multinomial(
                self.motion_bin_weights,
                int(env_ids.numel()),
                replacement=True,
                generator=generator,
            )
            motion_ids = torch.bucketize(global_bin, self._bin_offsets[1:], right=True)
            sampled_bins.copy_(global_bin - self._bin_offsets[motion_ids])
        else:
            motion_ids = torch.multinomial(
                clip_weights,
                int(env_ids.numel()),
                replacement=True,
                generator=generator,
            )
            for motion_id in range(len(self.clips)):
                selected = motion_ids == motion_id
                count = int(selected.sum())
                if not count:
                    continue
                lo_bin, hi_bin = (
                    self._bin_offsets_host[motion_id],
                    self._bin_offsets_host[motion_id + 1],
                )
                sampled_bins[selected] = torch.multinomial(
                    self.motion_bin_weights[lo_bin:hi_bin],
                    count,
                    replacement=True,
                    generator=generator,
                )
        self.buffers.motion_id[env_ids] = motion_ids
        lo, hi = self.ref.start_range
        random = None
        if self.ref.sampling_strategy == "concat_motion_bins" and self.ref.motion_bin_length_s is not None:
            within = torch.rand(int(env_ids.numel()), device=device, generator=generator)
            self.buffers.start_s[env_ids] = (sampled_bins + within) * self.ref.motion_bin_length_s
        else:
            random = torch.rand(int(env_ids.numel()), device=device, generator=generator)
        for motion_id, clip in enumerate(self.clips):
            mask = self.buffers.motion_id[env_ids] == motion_id
            selected = env_ids[mask]
            if selected.numel() == 0:
                continue
            if random is None:
                continue
            selected_random = random[mask]
            if self.ref.motion_bin_length_s is None:
                self.buffers.start_s[selected] = (selected_random * (hi - lo) + lo) * clip.sampling_length_s
            else:
                bin_id = sampled_bins[mask]
                within = torch.rand(int(selected.numel()), device=selected.device, generator=generator)
                self.buffers.start_s[selected] = (bin_id + within) * self.ref.motion_bin_length_s
        self.buffers.timestamp[env_ids] = 0.0
        self.last_update[env_ids] = 0.0
        self._reference_timestamp[env_ids] = -1.0
        draw_symmetric_mask(self.mask, env_ids, enabled=self.enabled, generator=generator)
        self._sample_motion_origins(env_ids, generator)
        self.refresh_initial(env_ids)
        self.refresh_at_current_time(env_ids)

    def record_failures(self, env_ids: torch.Tensor, failed: torch.Tensor, elapsed_s: torch.Tensor) -> None:
        """Accumulate the failed endpoint bin, matching BeyondMimic's reset-time update."""
        if self.ref.motion_bin_length_s is None or env_ids.numel() == 0:
            return
        failed_envs = env_ids[failed]
        motion_ids = self.buffers.motion_id[failed_envs]
        local_bins = torch.floor(
            (self.buffers.start_s[failed_envs] + elapsed_s[failed]) / self.ref.motion_bin_length_s
        ).long()
        local_bins = torch.minimum(local_bins, self._bin_counts[motion_ids] - 1)
        global_bins = self._bin_offsets[motion_ids] + local_bins
        self.current_motion_bin_fail_counter.scatter_add_(
            0,
            global_bins,
            torch.ones_like(global_bins, dtype=self.current_motion_bin_fail_counter.dtype),
        )

    def smooth_failures(self, alpha: float = 0.001) -> None:
        self.motion_bin_fail_counter.lerp_(self.current_motion_bin_fail_counter, alpha)
        self.current_motion_bin_fail_counter.zero_()

    def update_adaptive_weights(
        self, uniform_ratio: float = 0.1, kernel_size: int = 3, decay: float = 0.8
    ) -> dict[str, float] | None:
        if self.ref.motion_bin_length_s is None:
            return None
        kernel = torch.tensor(
            [decay**i for i in range(kernel_size)],
            device=self.motion_bin_weights.device,
        )
        kernel /= kernel.sum()
        probabilities = torch.empty_like(self.motion_bin_weights)
        for motion_id, count in enumerate(self._bin_counts_host):
            lo, hi = (
                self._bin_offsets_host[motion_id],
                self._bin_offsets_host[motion_id + 1],
            )
            source = self.motion_bin_fail_counter[lo:hi] + uniform_ratio / count
            indexes = torch.arange(count, device=source.device).unsqueeze(1) + torch.arange(
                kernel_size, device=source.device
            )
            probabilities[lo:hi] = (source[indexes.clamp(max=count - 1)] * kernel).sum(1)
        if self.ref.sampling_strategy == "concat_motion_bins":
            probabilities /= probabilities.sum().clamp_min(1e-10)
        else:
            for motion_id in range(len(self.clips)):
                lo, hi = (
                    self._bin_offsets_host[motion_id],
                    self._bin_offsets_host[motion_id + 1],
                )
                probabilities[lo:hi] /= probabilities[lo:hi].sum().clamp_min(1e-10)
        self.motion_bin_weights.copy_(probabilities)
        weights = (
            probabilities
            if self.ref.sampling_strategy == "concat_motion_bins"
            else probabilities[: self._bin_counts_host[0]]
        )
        entropy_denominator = torch.log(torch.tensor(float(weights.numel()), device=weights.device))
        entropy = weights.new_zeros(()) if weights.numel() == 1 else (
            -(weights * torch.log(weights + 1e-12)).sum() / entropy_denominator
        )
        metrics = torch.stack(
            (
                entropy,
                weights.max(),
                weights.argmax().to(weights.dtype) / max(weights.numel(), 1),
            )
        )
        sampling_entropy, sampling_top1_prob, sampling_top1_bin = metrics.detach().cpu().tolist()
        return {
            "sampling_entropy": sampling_entropy,
            "sampling_top1_prob": sampling_top1_prob,
            "sampling_top1_bin": sampling_top1_bin,
        }

    def refresh_initial(self, env_ids: torch.Tensor) -> None:
        """Build the floor-indexed reset state separately from rounded look-ahead data."""
        self.init_buffers.motion_id[env_ids] = self.buffers.motion_id[env_ids]
        self.init_buffers.start_s[env_ids] = self.buffers.start_s[env_ids]
        if self._packed_clips is not None:
            motion_ids = self.buffers.motion_id[env_ids]
            frame = torch.floor(self.buffers.start_s[env_ids] * self._packed_clips.framerate)
            times = frame.unsqueeze(-1) / self._packed_clips.framerate
            fill_buffers(
                self.init_buffers,
                env_ids,
                sample_packed_motion_clips(self._packed_clips, motion_ids, times),
                torch.zeros_like(times),
            )
        else:
            for motion_id, clip in enumerate(self.clips):
                selected = env_ids[self.buffers.motion_id[env_ids] == motion_id]
                if selected.numel() == 0:
                    continue
                frame = torch.floor(self.buffers.start_s[selected] * clip.framerate)
                times = frame.unsqueeze(-1) / clip.framerate
                fill_buffers(
                    self.init_buffers,
                    selected,
                    sample_clip(clip, times),
                    torch.zeros_like(times),
                )
        if self.ref.ensure_link_below_zero_ground:
            minimum = self.init_buffers.link_pos_w[env_ids, 0, :, 2].amin(dim=-1)
            # Both source managers gate this correction once for the selected batch.  When any
            # selected pose penetrates zero, they apply ``-minimum`` to every selected pose,
            # including lowering poses whose minimum link is already above zero.  Preserve that
            # reset-time behavior without polling the device through a Python ``if``.
            correction = torch.where((minimum < 0.0).any(), -minimum, torch.zeros_like(minimum))
            self.init_buffers.base_pos_w[env_ids, 0, 2] += correction
        self.init_buffers.base_pos_w[env_ids, 0, 2] += self.ref.motion_start_height_offset
        if self.resolved is not None:
            apply_symmetric_augmentation(self.init_buffers, env_ids, self.mask, self.resolved)
        translate_world_positions(self.init_buffers, env_ids, self.env_origins)

    def refresh_reference(self, env_ids: torch.Tensor) -> None:
        """Sample a single frame at ``t`` independently of the look-ahead window."""
        self.reference_buffers.motion_id[env_ids] = self.buffers.motion_id[env_ids]
        self.reference_buffers.start_s[env_ids] = self.buffers.start_s[env_ids]
        self.reference_buffers.timestamp[env_ids] = self.buffers.timestamp[env_ids]
        if self._packed_clips is not None:
            times = self.buffers.start_s[env_ids].unsqueeze(-1) + self.buffers.timestamp[env_ids].unsqueeze(-1)
            fill_buffers(
                self.reference_buffers,
                env_ids,
                sample_packed_motion_clips(self._packed_clips, self.buffers.motion_id[env_ids], times),
                torch.zeros_like(times),
            )
        else:
            for motion_id, clip in enumerate(self.clips):
                selected = env_ids[self.buffers.motion_id[env_ids] == motion_id]
                if selected.numel() == 0:
                    continue
                times = self.buffers.start_s[selected].unsqueeze(-1) + self.buffers.timestamp[selected].unsqueeze(-1)
                fill_buffers(
                    self.reference_buffers,
                    selected,
                    sample_clip(clip, times),
                    torch.zeros_like(times),
                )
        if self.resolved is not None:
            apply_symmetric_augmentation(self.reference_buffers, env_ids, self.mask, self.resolved)
        translate_world_positions(self.reference_buffers, env_ids, self.env_origins)
        self._reference_timestamp[env_ids] = self.buffers.timestamp[env_ids]

    def refresh(self, env_ids: torch.Tensor) -> None:
        """Rebuild selected buffers without changing their clock bookkeeping."""
        if self._packed_clips is not None:
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
                sample_packed_motion_clips(self._packed_clips, self.buffers.motion_id[env_ids], times),
                time_to,
            )
        else:
            for motion_id, clip in enumerate(self.clips):
                selected = env_ids[self.buffers.motion_id[env_ids] == motion_id]
                if selected.numel() == 0:
                    continue
                times, time_to = lookahead_times(
                    self.buffers.timestamp[selected],
                    self.buffers.start_s[selected],
                    self.ref.num_frames,
                    self.ref.frame_interval_s,
                    self.ref.data_start_from,
                )
                fill_buffers(self.buffers, selected, sample_clip(clip, times), time_to)
        if self.resolved is not None:
            apply_symmetric_augmentation(self.buffers, env_ids, self.mask, self.resolved)
        translate_world_positions(self.buffers, env_ids, self.env_origins)

    def refresh_at_current_time(self, env_ids: torch.Tensor) -> None:
        """Rebuild look-ahead and AMP-current buffers at the selected timestamp."""
        self.refresh(env_ids)
        self.refresh_reference(env_ids)
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
