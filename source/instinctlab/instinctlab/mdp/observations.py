"""Observation terms that run unmodified under either engine's native manager.

Each function here takes the same arguments an Isaac Lab observation term takes, including a native
``SceneEntityCfg`` that the compiler has already lowered, and is passed to the engine's own
``ObservationManager``. Nothing wraps it at runtime. That is what makes a migrated task cheap: the
term bodies below are the Isaac Lab ones, with the attribute names moved onto the hub vocabulary.

Two of those moves change a number, and both are recorded here rather than in a commit message,
because a reader comparing against the Isaac Lab original will otherwise assume a typo:

**Angular velocity moved to the link spelling for free.** Isaac Lab's ``root_ang_vel_b`` is a legacy
alias for ``root_com_ang_vel_b``, and mjlab has only ``root_link_ang_vel_b``. Reading Isaac Lab's
``ArticulationData``, ``root_link_vel_w`` is a clone of ``root_com_vel_w`` whose *linear* rows alone
receive the centre-of-mass offset correction; the angular rows are copied untouched. So the two
angular quantities are bitwise identical, and using the hub spelling costs nothing against the
golden. This is what one expects physically -- a rigid body's angular velocity does not depend on
the point it is measured about -- but it is asserted here because it was checked, not because it
was assumed.

**Linear velocity does not.** The same code path adds ``ω × R(−com_pos_b)`` to the linear rows, so
``root_link_lin_vel_b`` and Isaac Lab's ``root_lin_vel_b`` differ by exactly that term whenever the
root link's centre of mass is offset from its origin, which for a humanoid torso it is. The hub
carries the link quantity because it is the one both engines can express, so :func:`base_lin_vel`
differs from the golden by that cross product. It belongs in the difference whitelist with this
reason, and it is a critic-only observation, so it does not reach the deployed policy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from collections.abc import Sequence
from typing import Any

from instinctlab.compat.env import RlEnv, get_command
from instinctlab.compat.sensors import depth_image
from instinctlab.spec.sensor import RayCasterRef

__all__ = [
    "DelayedDepthImage",
    "clear_delayed_depth_history",
    "delayed_depth_terms",
    "base_ang_vel",
    "base_lin_vel",
    "generated_commands",
    "joint_pos_rel",
    "joint_vel",
    "joint_vel_rel",
    "last_action",
    "projected_gravity",
]


def base_ang_vel(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Root angular velocity in the body frame.

    Identical to Isaac Lab's ``base_ang_vel`` value for value; see the module docstring for why the
    link spelling costs nothing here.
    """
    asset = env.scene[_name(asset_cfg)]
    return asset.data.root_link_ang_vel_b


def base_lin_vel(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Root **link** linear velocity in the body frame.

    Not the same quantity as Isaac Lab's ``base_lin_vel``, which reads the centre-of-mass alias.
    The hub carries the link quantity because it is the one both engines express.
    """
    asset = env.scene[_name(asset_cfg)]
    return asset.data.root_link_lin_vel_b


def projected_gravity(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Gravity direction in the body frame.

    The portable attitude signal. The raw gravity vectors are not portable -- Isaac Lab normalises
    the live simulation gravity into ``GRAVITY_VEC_W`` while mjlab hardcodes ``[0, 0, -1]`` under a
    lowercase name -- but this projection is spelled and computed the same on both.
    """
    asset = env.scene[_name(asset_cfg)]
    return asset.data.projected_gravity_b


def joint_pos_rel(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Joint positions relative to their defaults."""
    asset = env.scene[_name(asset_cfg)]
    joint_ids = _joint_ids(asset_cfg)
    return asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]


def joint_vel(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Joint velocities.

    Portable as-is: both engines expose ``joint_vel`` meaning the same thing. Note that
    ``joint_acc`` next door does *not* port -- Isaac Lab finite-differences it while mjlab reads
    MuJoCo's analytic ``qacc`` -- which is why there is no acceleration term in this module.
    """
    asset = env.scene[_name(asset_cfg)]
    return asset.data.joint_vel[:, _joint_ids(asset_cfg)]


def joint_vel_rel(env: RlEnv, asset_cfg: Any = None) -> torch.Tensor:
    """Joint velocities relative to their defaults."""
    asset = env.scene[_name(asset_cfg)]
    joint_ids = _joint_ids(asset_cfg)
    return asset.data.joint_vel[:, joint_ids] - asset.data.default_joint_vel[:, joint_ids]


def last_action(env: RlEnv, action_name: str | None = None) -> torch.Tensor:
    """The previous action, either the whole vector or one term's raw input.

    The named form goes through :func:`~instinctlab.compat.env.raw_action` because the two engines
    spell the attribute differently -- ``raw_actions`` against ``raw_action``, a single character.
    """
    if action_name is None:
        return env.action_manager.action
    from instinctlab.compat.env import raw_action

    return raw_action(env, action_name)


class DelayedDepthImage:
    """Crop, blur, normalise, history and delay of a pinhole depth image.

    The reference puts crop / blur / normalisation on the *sensor* and delay on
    ``delayed_visualizable_image``. Isaac Lab has no observation-manager delay;
    mjlab's ``delay_max_lag`` is applied *before* history and would age the
    image differently. Both therefore live here, so the two engines feed the
    policy the same pipeline.

    Raw misses stay ``+inf`` on the sensor (visible). This term replaces them
    with ``max_distance`` *before* the blur -- the same order as the reference
    clipping -- then normalises to ``[0, 1]``. A miss is therefore 1.0 in the
    observation, which is the known silent-looking value; the live tests pin
    both the raw +inf and this 1.0.

    History is oldest-to-newest with ``[:, -1]`` the latest frame, matching the
    reference buffer. Manager-level ``history_length`` on this term must stay 0:
    stacking the already-sampled 8-frame output would silently change the width.
    """

    def __init__(self, cfg: Any, env: RlEnv) -> None:
        params = cfg.params
        self.sensor_ref: RayCasterRef = params["sensor"]
        self.history_skip_frames = max(int(params.get("history_skip_frames", 5)), 1)
        self.num_output_frames = max(int(params.get("num_output_frames", 8)), 1)
        self.delayed_frame_ranges = tuple(params.get("delayed_frame_ranges", (0, 1)))
        self.sensor_history_length = int(params.get("history_length", 37))
        self.blur_kernel_size = int(params.get("blur_kernel_size", 3))
        self.blur_sigma = float(params.get("blur_sigma", 1.0))
        crop_h, crop_w = self.sensor_ref.cropped_hw()
        device = env.device
        self._history = torch.zeros(env.num_envs, self.sensor_history_length, crop_h, crop_w, device=device)
        self._write = 0
        self._delay = torch.zeros(env.num_envs, device=device, dtype=torch.long)
        self._primed = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
        self.frame_offset = torch.flip(
            torch.arange(
                0,
                self.num_output_frames * self.history_skip_frames,
                self.history_skip_frames,
                device=device,
            ),
            dims=(0,),
        )
        self._check_delay_bounds()
        self.reset()

    def _check_delay_bounds(self) -> None:
        max_delay = self.delayed_frame_ranges[1]
        needed = (self.num_output_frames - 1) * self.history_skip_frames + 1
        if needed + max_delay > self.sensor_history_length:
            raise ValueError(
                f"depth history of {self.sensor_history_length} cannot hold "
                f"{self.num_output_frames} frames skipped by {self.history_skip_frames} "
                f"with delay up to {max_delay} "
                f"({needed + max_delay} slots required)."
            )

    def clear_history(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Zero the 37-slot ring for selected envs. Called on episode reset, not on obs-term reset.

        InstinctMJ and main Isaac parkour clear the camera ``AsyncCircularBuffer`` from the sensor
        ``reset``; ``delayed_visualizable_image.reset`` only redraws delay. This term owns the ring
        on compiled/mjlab engines, so engines call this hook when the camera resets.
        """
        if env_ids is None:
            env_ids = slice(None)
        elif isinstance(env_ids, torch.Tensor):
            if env_ids.numel() == 0:
                return
            env_ids = env_ids.to(device=self._history.device, dtype=torch.long)
            if env_ids.ndim == 0:
                env_ids = env_ids.unsqueeze(0)
        self._history[env_ids] = 0
        self._primed[env_ids] = False

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Resample per-env delay only, matching ``delayed_visualizable_image.reset``."""
        if env_ids is None:
            env_ids = slice(None)
        elif isinstance(env_ids, torch.Tensor):
            if env_ids.numel() == 0:
                return
            env_ids = env_ids.to(device=self._history.device, dtype=torch.long)
            if env_ids.ndim == 0:
                env_ids = env_ids.unsqueeze(0)
        n = self._delay[env_ids].shape[0]
        lo, hi = self.delayed_frame_ranges
        self._delay[env_ids] = torch.randint(int(lo), int(hi) + 1, (n,), device=self._delay.device)

    def __call__(
        self,
        env: RlEnv,
        sensor: RayCasterRef,
        history_skip_frames: int = 5,
        num_output_frames: int = 8,
        delayed_frame_ranges: tuple[int, int] = (0, 1),
        history_length: int = 37,
        blur_kernel_size: int = 3,
        blur_sigma: float = 1.0,
    ) -> torch.Tensor:
        del history_skip_frames, num_output_frames, delayed_frame_ranges, history_length
        del blur_kernel_size, blur_sigma
        raw = depth_image(env.scene.sensors[sensor.name])
        processed = _process_depth_image(raw, sensor, self.blur_kernel_size, self.blur_sigma)
        self._history[:, self._write] = processed
        unprimed = ~self._primed
        if bool(unprimed.any()):
            self._history[unprimed] = processed[unprimed].unsqueeze(1)
            self._primed[unprimed] = True
        self._write = (self._write + 1) % self.sensor_history_length
        order = (torch.arange(self.sensor_history_length, device=self._history.device) + self._write) % (
            self.sensor_history_length
        )
        linear = self._history[:, order]
        indices = (self.sensor_history_length - self.frame_offset.unsqueeze(0) - self._delay.unsqueeze(1) - 1).to(
            torch.long
        )
        batch = torch.arange(linear.shape[0], device=linear.device).unsqueeze(1).expand_as(indices)
        return linear[batch, indices]


def _coerce_reset_env_ids(env: RlEnv, env_ids: Any | None) -> torch.Tensor | slice:
    """Normalize ``_reset_idx`` env_ids (``None``, slice, int, Sequence, tensor)."""
    if env_ids is None:
        return slice(None)
    if isinstance(env_ids, slice):
        return env_ids
    device = getattr(env, "device", "cpu")
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return env_ids.reshape(0).to(device=device, dtype=torch.long)
        return env_ids.reshape(-1).to(device=device, dtype=torch.long)
    if isinstance(env_ids, int):
        return torch.tensor([env_ids], device=device, dtype=torch.long)
    if isinstance(env_ids, Sequence) and not isinstance(env_ids, (str, bytes, bytearray)):
        return torch.tensor(list(env_ids), device=device, dtype=torch.long)
    raise TypeError(f"unsupported env_ids type {type(env_ids)!r}")


def delayed_depth_terms(env: RlEnv) -> tuple[DelayedDepthImage, ...]:
    """Return live depth terms, unwrapping Isaac's ``ManagerTermBase`` adapter."""
    manager = getattr(env, "observation_manager", None)
    if manager is None:
        return ()
    found: list[DelayedDepthImage] = []
    cfgs = getattr(manager, "_group_obs_term_cfgs", {})
    for group_cfgs in cfgs.values():
        for cfg in group_cfgs:
            term = getattr(cfg, "func", None)
            implementation = getattr(term, "_impl", term)
            if isinstance(implementation, DelayedDepthImage):
                found.append(implementation)
    return tuple(found)


def clear_delayed_depth_history(env: RlEnv, env_ids: Any | None = None) -> None:
    """Episode-reset hook: clear ``DelayedDepthImage`` rings the way a camera sensor reset would."""
    coerced = _coerce_reset_env_ids(env, env_ids)
    if isinstance(coerced, torch.Tensor) and coerced.numel() == 0:
        return
    for term in delayed_depth_terms(env):
        term.clear_history(coerced)


def _process_depth_image(image: torch.Tensor, sensor: RayCasterRef, kernel_size: int, sigma: float) -> torch.Tensor:
    """Crop → blur → clip-and-normalise. Misses become the ceiling (1.0)."""
    finite = torch.where(torch.isfinite(image), image, torch.full_like(image, sensor.max_distance))
    if sensor.crop is not None:
        top, bottom, left, right = sensor.crop
        height, width = finite.shape[1], finite.shape[2]
        finite = finite[:, top : height - bottom, left : width - right]
    plane = finite.squeeze(-1)
    if kernel_size > 1 and sigma > 0.0:
        plane = _gaussian_blur(plane, kernel_size, sigma)
    clipped = plane.clamp(0.0, sensor.max_distance)
    return clipped / sensor.max_distance


def _gaussian_blur(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - kernel_size // 2
    gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
    gauss = gauss / gauss.sum()
    kernel = (gauss[:, None] * gauss[None, :]).view(1, 1, kernel_size, kernel_size)
    pad = kernel_size // 2
    # torchvision.transforms.GaussianBlur (used by InstinctMJ) pads by
    # reflection before applying the separable Gaussian kernel.
    padded = F.pad(image.unsqueeze(1), (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(padded, kernel).squeeze(1)


def generated_commands(env: RlEnv, command_name: str) -> torch.Tensor:
    """The current command from the named generator.

    Goes through :func:`~instinctlab.compat.env.get_command` rather than the manager directly,
    because a missing command is a ``KeyError`` on one engine and a silent ``None`` on the other.
    """
    return get_command(env, command_name)


def _name(asset_cfg: Any) -> str:
    """Entity key from a lowered ``SceneEntityCfg``, defaulting to the robot."""
    return "robot" if asset_cfg is None else asset_cfg.name


def _joint_ids(asset_cfg: Any) -> Any:
    """Joint selection from a lowered ``SceneEntityCfg``, defaulting to all joints.

    Indices rather than names on purpose: after ``resolve()`` the engines disagree about what
    ``joint_names`` holds -- Isaac Lab leaves the patterns in place, mjlab replaces them with the
    matches -- while the index lists mean the same thing on both.
    """
    return slice(None) if asset_cfg is None else asset_cfg.joint_ids


def _body_ids(asset_cfg: Any) -> Any:
    """Body selection from a lowered ``SceneEntityCfg``, defaulting to all bodies.

    Same reason as :func:`_joint_ids`: after ``resolve()`` ``body_names`` is patterns on one
    engine and matches on the other. The indices agree.
    """
    return slice(None) if asset_cfg is None else asset_cfg.body_ids


def _body_index_list(asset_cfg: Any, n_bodies: int) -> list[int]:
    """Concrete body indices, so a term can ask for the first or the first two without guessing."""
    ids = _body_ids(asset_cfg)
    if isinstance(ids, slice):
        return list(range(*ids.indices(n_bodies)))
    if isinstance(ids, int):
        return [ids]
    return list(ids)
