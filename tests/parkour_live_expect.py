"""Expected live terrain tables for ``Instinct-Parkour-Target-G1``.

Import-safe: no engine, no torch. Isaac Sim live tests must be able to import
this before Kit starts.
"""

from __future__ import annotations

import os

from tests.live_device import resolve_live_device  # re-exported: existing imports resolve it here

# Declared type shares, sub-terrain declaration order. Both engines' curriculum
# generators allocate columns from these; even-split spawn then reproduces them.
PARKOUR_DECLARED_PROPORTIONS: tuple[tuple[str, float], ...] = (
    ("perlin_rough", 0.05),
    ("perlin_rough_stand", 0.05),
    ("square_gaps", 0.10),
    ("pyramid_stairs", 0.15),
    ("pyramid_stairs_high", 0.10),
    ("pyramid_stairs_inv", 0.15),
    ("pyramid_stairs_inv_high", 0.10),
    ("boxes", 0.10),
    ("dense_boxes", 0.10),
    ("hf_pyramid_slope_inv", 0.10),
)
ISAAC_NINTH_NAME = "mesh_boxes"
MJLAB_NINTH_NAME = "dense_boxes"

# Both engines, num_cols=20, Isaac cumulative-proportion formula
# (j / 20 + 0.001). Was 10 names in declaration order on mjlab (one column per
# type, ``num_cols`` ignored). Ninth tile is ``dense_boxes`` on mjlab,
# ``mesh_boxes`` on Isaac — same slot, different reference name.
MJLAB_CURRICULUM_COLUMNS: tuple[str, ...] = (
    "perlin_rough",
    "perlin_rough_stand",
    "square_gaps",
    "square_gaps",
    "pyramid_stairs",
    "pyramid_stairs",
    "pyramid_stairs",
    "pyramid_stairs_high",
    "pyramid_stairs_high",
    "pyramid_stairs_inv",
    "pyramid_stairs_inv",
    "pyramid_stairs_inv",
    "pyramid_stairs_inv_high",
    "pyramid_stairs_inv_high",
    "boxes",
    "boxes",
    "dense_boxes",
    "dense_boxes",
    "hf_pyramid_slope_inv",
    "hf_pyramid_slope_inv",
)

ISAAC_PROPORTION_COLUMNS: tuple[str, ...] = (
    "perlin_rough",
    "perlin_rough_stand",
    "square_gaps",
    "square_gaps",
    "pyramid_stairs",
    "pyramid_stairs",
    "pyramid_stairs",
    "pyramid_stairs_high",
    "pyramid_stairs_high",
    "pyramid_stairs_inv",
    "pyramid_stairs_inv",
    "pyramid_stairs_inv",
    "pyramid_stairs_inv_high",
    "pyramid_stairs_inv_high",
    "boxes",
    "boxes",
    "mesh_boxes",
    "mesh_boxes",
    "hf_pyramid_slope_inv",
    "hf_pyramid_slope_inv",
)

REQUIRED_COMMAND_METRICS: tuple[str, ...] = ("tracking_exp_vel_xy", "tracking_exp_vel_yaw")
PARKOUR_KIND_NAMES: frozenset[str] = frozenset(
    {
        "pose_velocity",
        "motors_power_square",
        "applied_torque_limits_by_ratio",
        "joint_torques_l2",
        "joint_acc_l2",
        "contact_slide",
        "reset_joints_by_offset",
        "randomize_friction",
        "joint_position",
        "reset_root_state_uniform",
        "register_virtual_obstacles",
        "illegal_contact",
        "undesired_contacts",
    }
)


def driven_joint_names(action_term) -> list[str]:
    return list(getattr(action_term, "target_names", getattr(action_term, "_joint_names", [])))


def assert_parkour_live_invariants(env, spec, compiled, *, expected_columns: tuple[str, ...]) -> None:
    """Shared L6 checks. Import this only after the engine is bootstrapped."""
    from instinctlab.engines.pose_velocity import actual_column_count

    assert compiled.resolution.skipped == {}, compiled.resolution.skipped
    driven = driven_joint_names(env.action_manager.get_term("joint_pos"))
    assert driven == list(spec.robot.joint_names)
    command = env.command_manager.get_term("base_velocity")
    missing = [key for key in REQUIRED_COMMAND_METRICS if key not in command.metrics]
    assert not missing, f"command metrics missing {missing}; have {sorted(command.metrics)}"
    names = list(getattr(command, "_column_names", ()))
    built_cols = actual_column_count(env.scene.terrain)
    declared = getattr(env.scene.terrain.cfg.terrain_generator, "num_cols", None)
    assert declared == 20, declared
    assert built_cols == len(
        expected_columns
    ), f"built {built_cols} columns, expected {len(expected_columns)} (declared num_cols={declared})"
    assert names == list(expected_columns), names


def parkour_declared_shares(*, ninth_name: str) -> dict[str, float]:
    """Declared type shares with the engine's name for the ninth tile.

    Order is the declaration order. The formula assigns columns from that
    order; moving the ninth name to the end of a dict would reallocate later
    columns onto the wrong type.
    """
    shares: dict[str, float] = {}
    for name, value in PARKOUR_DECLARED_PROPORTIONS:
        shares[ninth_name if name == MJLAB_NINTH_NAME else name] = value
    return shares


def assert_terrain_type_shares(terrain_types, column_names, declared: dict[str, float], *, num_envs: int) -> None:
    """Measured type histogram must match declared proportions, not a uniform 10%.

    Even-split across proportion-allocated columns reproduces the declaration.
    A uniform 10% each means spawn fell through to 'one env per type' or
    'even across types' and must fail this check, not have its expectation edited.
    """
    from instinctlab.engines.pose_velocity import type_share_histogram

    hist = type_share_histogram(terrain_types, column_names)
    if "" in hist:
        raise AssertionError(f"unnamed columns appeared in the spawn histogram: {hist}")
    # One extra env on a type is the even-split remainder. 512/20 = 25.6, so a
    # 3-column type can be off by at most 3/num_envs. Uniform 10% is 0.05 away
    # from a 5% type — well outside this bound. Check uniform first so that
    # failure is loud instead of looking like a single-type miss.
    tol = 3.0 / num_envs + 1e-12
    uniform = 1.0 / len(declared)
    if all(abs(hist.get(name, 0.0) - uniform) <= tol for name in declared):
        raise AssertionError(
            f"terrain_types is uniform at ~{uniform:.0%} each. That is the silent "
            f"spawn-fallback failure, not the declared shares {declared}."
        )
    for name, share in declared.items():
        measured = hist.get(name, 0.0)
        assert abs(measured - share) <= tol, (
            f"{name}: measured {measured:.4f} vs declared {share:.4f} (tol {tol:.4f}). "
            f"full histogram={ {k: round(v, 4) for k, v in hist.items()} }"
        )


SCANNER_NAMES: tuple[str, ...] = ("left_height_scanner", "right_height_scanner")
SCANNER_RAYS = 2
SCANNER_OFFSET: tuple[float, float, float] = (0.04, 0.0, 20.0)
# Large enough that 20 * sin(pitch) is metres, not millimetres. A level
# foot makes yaw-only and full-R identical -- that is the blind spot.
SCANNER_PITCH_JOINTS: tuple[str, ...] = ("left_ankle_pitch_joint", "right_ankle_pitch_joint")
SCANNER_ROLL_JOINTS: tuple[str, ...] = ("left_ankle_roll_joint", "right_ankle_roll_joint")
SCANNER_TILT_PITCH = 0.8
SCANNER_TILT_ROLL = 0.5
CAMERA_NAME = "camera"
CAMERA_RAW_HW = (36, 64)
CAMERA_CROP_HW = (18, 32)
CAMERA_HISTORY_FRAMES = 8
CAMERA_SENSOR_HISTORY = 37
# Parkour grid is 10×20 tiles of 8 m plus a 3 m border → half-extents (43, 83).
# Two hundred metres from the origin is off every engine's mesh.
SCANNER_MISS_OFFSET: tuple[float, float, float] = (200.0, 200.0, 1.0)
# High enough that a 48° look-down cannot reach z=0 within 2.5 m
# (5 / sin(48°) ≈ 6.7 m). The camera is body-fixed: rotating the root
# cannot move the robot out of the image, so a full-frame miss is
# impossible. The top rows look above the body; those are the miss band.
CAMERA_MISS_OFFSET: tuple[float, float, float] = (200.0, 200.0, 5.0)
CAMERA_MISS_SKY_ROWS = 8
# Root pose that makes base and yaw-only camera rays disagree. A level torso
# makes them identical -- that is the same blind spot the scanner had.
# roll=0.5 / pitch=0.6 / yaw=0.2 slides the ~0.44 m offset by ~0.33 m and
# moves the optical-axis plane hit by ~0.5 m (both >> 20 mm).
CAMERA_TILT_ROLL = 0.5
CAMERA_TILT_PITCH = 0.6
CAMERA_TILT_YAW = 0.2
CAMERA_TILT_ROOT_OFFSET: tuple[float, float, float] = (0.0, 0.0, 0.82)
CAMERA_OFFSET: tuple[float, float, float] = (0.0487988662332928, 0.01, 0.4378029937970051)
CAMERA_OFFSET_ROT: tuple[float, float, float, float] = (
    0.9135367613482678,
    0.004363309284746571,
    0.4067366430758002,
    0.0,
)
CAMERA_ALIGN_TOL_M = 0.020
CAMERA_YAW_SEP_M = 0.15


def assert_foot_scanner_shape(env) -> None:
    """The grid is two rays. A third ray is a silent observation-width change."""
    from instinctlab.compat.sensors import ray_hits_w

    for name in SCANNER_NAMES:
        hits = ray_hits_w(env.scene.sensors[name])
        assert tuple(hits.shape) == (env.num_envs, SCANNER_RAYS, 3), (name, tuple(hits.shape))


def require_live_device() -> str:
    """Skip if the resolved live device is missing. Does not point at cuda:0."""
    import torch

    import pytest

    device = resolve_live_device()
    if not torch.cuda.is_available():
        pytest.skip("parkour live tests need a GPU")
    index = int(device.split(":")[-1]) if device.startswith("cuda:") else 0
    if torch.cuda.device_count() <= index:
        pytest.skip(f"parkour live device {device} is not present")
    if "INSTINCTLAB_LIVE_DEVICE" not in os.environ and torch.cuda.device_count() < 3:
        pytest.skip("default live device is cuda:2; trainings occupy cuda:0 and cuda:1")
    return device


def scanner_origins_for_alignment(ankle_pos, ankle_quat, offset, alignment: str):
    """Declared ray-start of the sky scanner. ``yaw`` keeps the 20 m world-up."""
    from instinctlab.compat.math import quat_apply, yaw_quat

    rot = yaw_quat(ankle_quat) if alignment == "yaw" else ankle_quat
    shift = offset.to(dtype=ankle_pos.dtype, device=ankle_pos.device)
    if shift.shape != ankle_pos.shape:
        shift = shift.reshape(1, -1).expand_as(ankle_pos)
    return ankle_pos + quat_apply(rot, shift)


def _scanner_ray_starts(sensor):
    for attr in ("_cached_world_origins", "_ray_starts_w"):
        value = getattr(sensor, attr, None)
        if value is not None:
            return value
    return None


def assert_foot_scanner_uses_yaw_alignment(env, *, device: str) -> None:
    """A pitched/rolled ankle must not tilt the 20 m offset.

    Level-foot checks cannot see this field: yaw-only and full rotation then
    produce the same rays. Parkour ankles pitch constantly. If either engine
    applies the full R, the origin slides ``20 * sin(pitch)`` and samples the
    wrong tile -- wrong heights, no exception.
    """
    import torch

    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs, device=device)
    names = list(robot.joint_names)
    q = robot.data.default_joint_pos.clone()
    for joint in SCANNER_PITCH_JOINTS:
        q[:, names.index(joint)] = SCANNER_TILT_PITCH
    for joint in SCANNER_ROLL_JOINTS:
        q[:, names.index(joint)] = SCANNER_TILT_ROLL
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    env.scene.write_data_to_sim()
    if hasattr(env.sim, "forward"):
        env.sim.forward()
    _force_scanner_refresh(env)

    offset = torch.tensor(SCANNER_OFFSET, device=device, dtype=robot.data.body_link_pos_w.dtype)
    bodies = list(robot.body_names)
    for sensor_name, body_name in (
        ("left_height_scanner", "left_ankle_roll_link"),
        ("right_height_scanner", "right_ankle_roll_link"),
    ):
        body_id = bodies.index(body_name)
        pos = robot.data.body_link_pos_w[:, body_id]
        quat = robot.data.body_link_quat_w[:, body_id]
        yaw_origin = scanner_origins_for_alignment(pos, quat, offset, "yaw")
        full_origin = scanner_origins_for_alignment(pos, quat, offset, "base")
        slide = (yaw_origin - full_origin).norm(dim=-1)
        assert bool(
            (slide > 2.0).all()
        ), f"{sensor_name}: tilt only slides {slide.tolist()} m; yaw-only and full-R are still indistinguishable"
        starts = _scanner_ray_starts(env.scene.sensors[sensor_name])
        if starts is None:
            from instinctlab.compat.sensors import ray_hits_w

            hits = ray_hits_w(env.scene.sensors[sensor_name])
            measured = hits[:, :, :2].mean(dim=1)
            predicted = yaw_origin[:, :2]
            other = full_origin[:, :2]
        else:
            measured = starts.mean(dim=1)
            predicted = yaw_origin
            other = full_origin
        err_yaw = (measured - predicted).norm(dim=-1)
        err_full = (measured - other).norm(dim=-1)
        assert bool((err_yaw < 0.15).all()), (
            f"{sensor_name}: ray start is {err_yaw.tolist()} m from the yaw-only "
            f"origin (full-R error {err_full.tolist()})"
        )
        assert bool((err_full > 2.0).all()), (
            f"{sensor_name}: ray start matches full-R ({err_full.tolist()} m) "
            f"rather than yaw-only ({err_yaw.tolist()} m); the declared "
            "ray_alignment='yaw' is not what ran"
        )


def assert_foot_scanner_sees_some_ground(env) -> None:
    """If every ray misses, the terrain-only filter is hitting nothing."""
    import torch

    from instinctlab.compat.sensors import ray_hits_w

    finite = []
    for name in SCANNER_NAMES:
        hits = ray_hits_w(env.scene.sensors[name])
        finite.append(bool(torch.isfinite(hits[..., 2]).any()))
    assert any(finite), "every foot-scanner ray missed after reset; the caster is not hitting parkour terrain"


def _force_scanner_refresh(env) -> None:
    """Recompute after a written pose. Accessing ``.data`` is not enough.

    Isaac's ``update(force_recompute=True)`` still skips envs whose period has
    not elapsed; pass a dt at least the period. mjlab's ``sense()`` fills the
    buffers but leaves the ``.data`` cache valid, so ``scene.update`` must
    invalidate it.
    """
    if hasattr(env.sim, "sense"):
        env.sim.sense()
        env.scene.update(float(env.step_dt))
        return
    dt = max(float(getattr(env, "step_dt", 0.02)), 1.0)
    for name in SCANNER_NAMES:
        env.scene.sensors[name].update(dt, force_recompute=True)


def assert_foot_scanner_miss_is_positive_infinity(env, *, device: str) -> None:
    """A miss is ``+inf``, not world-z=0. Parkour has gaps; this path is live."""
    import torch

    from instinctlab.compat.sensors import ray_hits_w

    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs, device=device)
    pos = env.scene.env_origins + torch.tensor(SCANNER_MISS_OFFSET, device=device)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(env.num_envs, 4)
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    env.scene.write_data_to_sim()
    if hasattr(env.sim, "forward"):
        env.sim.forward()
    _force_scanner_refresh(env)
    for name in SCANNER_NAMES:
        hits = ray_hits_w(env.scene.sensors[name])
        assert torch.isposinf(
            hits
        ).all(), f"{name}: miss must be +inf on every axis, finite={hits[torch.isfinite(hits)][:12].tolist()}"


def assert_depth_camera_shape(env) -> None:
    """Raw 36×64, processed 8×18×32. A dropped axis is a silent policy-width change."""
    from instinctlab.compat.sensors import depth_image

    raw = depth_image(env.scene.sensors[CAMERA_NAME])
    assert tuple(raw.shape[1:3]) == CAMERA_RAW_HW, tuple(raw.shape)
    obs = env.observation_manager.compute()["policy"]
    image = obs["depth_image"] if isinstance(obs, dict) else None
    assert image is not None, "policy group is concatenated; depth_image must stay a separate term"
    assert tuple(image.shape[1:]) == (CAMERA_HISTORY_FRAMES, *CAMERA_CROP_HW), tuple(image.shape)
    camera = env.scene.sensors[CAMERA_NAME]
    mask = getattr(camera, "_allowed_geom_mask", None)
    if mask is not None:
        # Terrain plus listed G1 links. Terrain-only is the silent "no legs" failure.
        assert int(mask.sum()) > 10, int(mask.sum())
    else:
        meshes = getattr(getattr(camera, "cfg", None), "mesh_prim_paths", None)
        if meshes is not None:
            assert len(meshes) > 1, len(meshes)


def assert_depth_camera_miss_is_positive_infinity(env, *, device: str) -> None:
    """A miss is +inf on the raw image, not the 2.5 m normalisation ceiling.

    The camera is welded to the torso and the hit list includes the robot, so
    rotating the root cannot empty the image -- hands and feet sit ~35–40°
    off the optical axis, inside the crop. Off the mesh at z=5 the ground is
    out of range; the top rows look above the body and must be +inf. Finite
    pixels are self-hits, not a clipped miss. Processed 1.0 is pinned on an
    all-inf buffer in the cheap tests (blur would smear a mixed frame).
    """
    import torch

    from instinctlab.compat.sensors import depth_image

    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs, device=device)
    pos = env.scene.env_origins + torch.tensor(CAMERA_MISS_OFFSET, device=device)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(env.num_envs, 4)
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    env.scene.write_data_to_sim()
    if hasattr(env.sim, "forward"):
        env.sim.forward()
    _force_camera_refresh(env)
    raw = depth_image(env.scene.sensors[CAMERA_NAME])
    sky = raw[:, :CAMERA_MISS_SKY_ROWS]
    assert torch.isposinf(sky).all(), f"sky-band miss must be +inf, finite={sky[torch.isfinite(sky)][:12].tolist()}"
    finite = raw[torch.isfinite(raw)]
    if finite.numel():
        assert bool((finite > 0.0).all()) and bool(
            (finite <= 2.5).all()
        ), f"self-hits must be in (0, 2.5], got min={float(finite.min())} max={float(finite.max())}"


def _force_camera_refresh(env) -> None:
    if hasattr(env.sim, "sense"):
        env.sim.sense()
        env.scene.update(float(env.step_dt))
        return
    dt = max(float(getattr(env, "step_dt", 0.02)), 1.0)
    env.scene.sensors[CAMERA_NAME].update(dt, force_recompute=True)


def _camera_declared_offset(sensor) -> tuple[object, object]:
    """Offset the live sensor was built with. Isaac uses ``offset``; mjlab uses ``origin_offset``."""
    cfg = sensor.cfg
    if hasattr(cfg, "origin_offset"):
        return cfg.origin_offset, cfg.origin_offset_rot
    return cfg.offset.pos, cfg.offset.rot


def _camera_measured_pose(sensor):
    """Live camera pose. Missing buffers are a failure, not a chance to reconstruct."""
    data = sensor.data
    pos = getattr(data, "pos_w", None)
    quat = getattr(data, "quat_w_world", None)
    if pos is None or quat is None:
        raise AssertionError(
            "camera has no pos_w/quat_w_world; reconstructing from the torso "
            "would always match 'base' and hide a yaw-only flip"
        )
    return pos, quat


def assert_depth_camera_uses_base_alignment(env, *, device: str) -> None:
    """A pitched/rolled torso must rotate the camera, not just yaw it.

    Level-torso checks cannot see this field: ``base`` and ``yaw`` then
    produce the same rays. Parkour pitches the torso constantly. If either
    engine dropped pitch/roll, the whole depth image would slide exactly
    when the terrain matters -- no exception, a plausible-looking image.
    """
    import torch

    from instinctlab.compat.math import quat_apply, quat_from_euler_xyz
    from instinctlab.engines.ray_alignment import camera_pose_for_alignment

    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs, device=device)
    pos = env.scene.env_origins + torch.tensor(CAMERA_TILT_ROOT_OFFSET, device=device)
    quat = quat_from_euler_xyz(
        torch.full((env.num_envs,), CAMERA_TILT_ROLL, device=device),
        torch.full((env.num_envs,), CAMERA_TILT_PITCH, device=device),
        torch.full((env.num_envs,), CAMERA_TILT_YAW, device=device),
    )
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    if hasattr(robot, "write_root_link_velocity_to_sim"):
        robot.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6, device=device), env_ids=env_ids)
    q = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    env.scene.write_data_to_sim()
    if hasattr(env.sim, "forward"):
        env.sim.forward()
    _force_camera_refresh(env)

    bodies = list(robot.body_names)
    torso = bodies.index("torso_link")
    torso_pos = robot.data.body_link_pos_w[:, torso]
    torso_quat = robot.data.body_link_quat_w[:, torso]
    sensor = env.scene.sensors[CAMERA_NAME]
    offset_xyz, offset_rot = _camera_declared_offset(sensor)
    offset = torch.as_tensor(offset_xyz, device=device, dtype=torso_pos.dtype)
    rot = torch.as_tensor(offset_rot, device=device, dtype=torso_quat.dtype)
    base_pos, base_quat = camera_pose_for_alignment(torso_pos, torso_quat, offset, rot, "base")
    yaw_pos, yaw_quat = camera_pose_for_alignment(torso_pos, torso_quat, offset, rot, "yaw")
    slide = (base_pos - yaw_pos).norm(dim=-1)
    assert bool(
        (slide > CAMERA_YAW_SEP_M).all()
    ), f"camera: tilt only slides {slide.tolist()} m; yaw-only and full-R are still indistinguishable"
    measured_pos, measured_quat = _camera_measured_pose(sensor)
    err_base = (measured_pos - base_pos).norm(dim=-1)
    err_yaw = (measured_pos - yaw_pos).norm(dim=-1)
    assert bool(
        (err_base < CAMERA_ALIGN_TOL_M).all()
    ), f"camera: pose is {err_base.tolist()} m from the full-R origin (yaw-only error {err_yaw.tolist()})"
    assert bool((err_yaw > CAMERA_YAW_SEP_M).all()), (
        f"camera: pose matches yaw-only ({err_yaw.tolist()} m) rather than "
        f"full-R ({err_base.tolist()} m); the declared ray_alignment='base' "
        "is not what ran"
    )
    x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=measured_quat.dtype).expand_as(measured_pos)
    measured_axis = quat_apply(measured_quat, x)
    base_axis = quat_apply(base_quat, x)
    yaw_axis = quat_apply(yaw_quat, x)
    axis_base = 1.0 - (measured_axis * base_axis).sum(dim=-1).clamp(-1.0, 1.0)
    axis_yaw = 1.0 - (measured_axis * yaw_axis).sum(dim=-1).clamp(-1.0, 1.0)
    assert bool(
        (axis_base < 0.01).all()
    ), f"camera: optical axis is {axis_base.tolist()} from full-R (yaw-only {axis_yaw.tolist()})"
    assert bool(
        (axis_yaw > 0.05).all()
    ), f"camera: optical axis matches yaw-only ({axis_yaw.tolist()}) rather than full-R ({axis_base.tolist()})"


def assert_rewards_finite_and_alive(env, *, steps: int, device: str) -> None:
    import torch

    actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=device)
    saw_nonzero = False
    for step in range(steps):
        _obs, reward, terminated, truncated, _info = env.step(actions)
        assert torch.isfinite(reward).all(), f"non-finite reward at step {step}"
        if float(reward.abs().sum()) > 0.0:
            saw_nonzero = True
        if step == 0:
            assert not bool((terminated | truncated).all()), "every env done on the first step"
    assert saw_nonzero, "rewards were identically zero across the smoke steps"


AMP_SAME_FUNCTION_ATOL = 5e-4
AMP_SAME_FUNCTION_RTOL = 1e-4
"""Live write-roundtrip. Physics buffers are float32; 5e-4 still sees a frame swap."""


def canonical_joint_ids(robot, spec) -> list[int]:
    """Native indices of the catalog joints, gathered by name."""
    native = list(robot.joint_names)
    return [native.index(name) for name in spec.robot.joint_names]


def assert_amp_same_function(env, spec, *, device: str) -> None:
    """Write clip frame 0 onto the robot; policy AMP must match reference AMP.

    Instant terms, not manager history: history mixes earlier states.
    """
    import torch

    from instinctlab.mdp.amp import AMP_TERM_ORDER, amp_obs_from_reference, amp_obs_from_robot_like

    robot = env.scene["robot"]
    sensor = env.scene.sensors["motion_reference"]
    joint_ids = canonical_joint_ids(robot, spec)
    assert list(spec.robot.joint_names)[0] == "waist_pitch_joint"
    native = list(robot.joint_names)
    # Isaac is BFS; a name-based gather that became positional would still pass if we
    # only compared engines to each other. This asks for the catalog names.
    assert native.index("waist_pitch_joint") == joint_ids[0]

    quat = sensor.data.base_quat_w[:, 0]
    lin_w = sensor.data.base_lin_vel_w[:, 0]
    ang_w = sensor.data.base_ang_vel_w[:, 0]
    q_canon = sensor.data.joint_pos[:, 0]
    qd_canon = sensor.data.joint_vel[:, 0]
    n = env.num_envs
    env_ids = torch.arange(n, device=device)
    native_q = torch.zeros(n, len(native), device=device, dtype=q_canon.dtype)
    native_qd = torch.zeros_like(native_q)
    ids = torch.as_tensor(joint_ids, device=device)
    native_q[:, ids] = q_canon
    native_qd[:, ids] = qd_canon
    origins = env.scene.env_origins
    pose = torch.cat([sensor.data.base_pos_w[:, 0] + origins, quat], dim=-1)
    vel = torch.cat([lin_w, ang_w], dim=-1)
    robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
    robot.write_joint_state_to_sim(native_q, native_qd, env_ids=env_ids)
    env.scene.write_data_to_sim()
    if hasattr(env.sim, "forward"):
        env.sim.forward()
    if hasattr(robot, "update"):
        robot.update(env.physics_dt)

    policy = amp_obs_from_robot_like(robot.data, joint_ids)
    from instinctlab.compat.math import quat_apply

    gravity_w = quat_apply(robot.data.root_link_quat_w, robot.data.projected_gravity_b)
    reference = amp_obs_from_reference(
        sensor.data,
        robot.data.default_joint_pos[:, joint_ids],
        robot.data.default_joint_vel[:, joint_ids],
        gravity_w,
    )
    for name in AMP_TERM_ORDER:
        torch.testing.assert_close(
            policy[name],
            reference[name],
            atol=AMP_SAME_FUNCTION_ATOL,
            rtol=AMP_SAME_FUNCTION_RTOL,
            msg=f"{name}: policy vs reference after writing the clip onto the robot",
        )


def assert_depth_encoder_is_fed(env, spec, wrapper, agent_cfg) -> dict[str, int]:
    """``component_names=['depth_image']`` must resolve and the Conv2d must see 8 channels.

    Returns policy/critic MLP input widths after the encoder takeout.
    """
    import torch

    from instinct_rl.modules.parallel_layer import ParallelLayer
    from instinct_rl.utils.utils import get_obs_slice, get_subobs_size

    obs_format = wrapper.get_obs_format()
    policy_seg = obs_format["policy"]
    critic_seg = obs_format["critic"]
    assert "depth_image" in policy_seg, f"policy segments: {list(policy_seg)}"
    assert tuple(policy_seg["depth_image"]) == (8, 18, 32), policy_seg["depth_image"]
    assert tuple(critic_seg["depth_image"]) == (8, 18, 32), critic_seg["depth_image"]
    assert "depth_image" in spec.mdp.observations["policy"].terms

    dumped = agent_cfg.to_dict()
    enc = dumped["policy"]["encoder_configs"]
    assert enc["depth_encoder"]["component_names"] == ["depth_image"]
    layer = ParallelLayer(policy_seg, enc)
    assert "depth_image" not in layer.output_segment
    first = next(m for m in layer._parallel_blocks["depth_encoder"].modules() if isinstance(m, torch.nn.Conv2d))
    assert first.in_channels == 8, first.in_channels
    policy_dim = int(get_subobs_size(layer.output_segment))
    critic_layer = ParallelLayer(critic_seg, dumped["policy"]["critic_encoder_configs"])
    critic_dim = int(get_subobs_size(critic_layer.output_segment))
    assert policy_dim == 768 + 128, policy_dim
    assert critic_dim == 792 + 128, critic_dim

    obs, extras = wrapper.get_observations()
    layer = layer.to(obs.device)
    layer.eval()
    with torch.no_grad():
        out = layer(obs)
        sl, _ = get_obs_slice(policy_seg, "depth_image")
        zeroed = obs.clone()
        zeroed[:, sl] = 0
        out_zero = layer(zeroed)
    assert not torch.allclose(out, out_zero, atol=1e-5), "encoder ignored a depth wipe; it is not being fed"
    del extras
    return {"policy": policy_dim, "critic": critic_dim}


VOLUME_SENSOR_NAME = "leg_volume_points"
VOLUME_BODIES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")
VOLUME_POINTS_PER_BODY = 100
VOLUME_CYLINDER_RADIUS = 0.10
VOLUME_KNOWN_AXIS_OFFSET = 0.06
VOLUME_KNOWN_SPEED = 1.0
# Training uses r=0.05 terrain cylinders, not the synthetic r=0.10 fixture.
TERRAIN_CYLINDER_RADIUS = 0.05
TERRAIN_AXIS_OFFSET = 0.02


def _force_volume_refresh(env) -> None:
    if hasattr(env.sim, "sense"):
        env.sim.sense()
        env.scene.update(float(env.step_dt))
        return
    dt = max(float(getattr(env, "step_dt", 0.02)), 1.0)
    env.scene.sensors[VOLUME_SENSOR_NAME].update(dt, force_recompute=True)


def assert_volume_points_shape(env) -> None:
    """Two ankles, 100 local points each. A dropped body is a silent sum change."""
    from instinctlab.compat.sensors import volume_points_penetration_offset

    sensor = env.scene.sensors[VOLUME_SENSOR_NAME]
    assert list(sensor.body_names) == list(VOLUME_BODIES), sensor.body_names
    offset = volume_points_penetration_offset(sensor)
    assert tuple(offset.shape) == (env.num_envs, 2, VOLUME_POINTS_PER_BODY, 3), tuple(offset.shape)
    pattern = getattr(sensor, "_volume_points_pattern", None)
    if pattern is not None:
        from instinctlab.tasks.parkour.config.g1.target_env_cfg import LEG_VOLUME_POINTS

        declared = list(LEG_VOLUME_POINTS.grid.points())
        assert pattern.shape[0] == VOLUME_POINTS_PER_BODY
        import pytest

        assert pattern[0].tolist() == pytest.approx(declared[0])
        assert pattern[-1].tolist() == pytest.approx(declared[-1])


def summarize_virtual_obstacles(terrain, *, xy_aabb=None) -> dict[str, object]:
    """Count / centroid / length / bbox of the generated edge cylinders."""
    import torch

    obstacles = getattr(terrain, "virtual_obstacles", None)
    if not obstacles:
        raise AssertionError("terrain.virtual_obstacles is empty; generate() never ran")
    obstacle = obstacles["edges"]
    edges = obstacle.edges_pyt
    if edges.numel() == 0:
        summary = {"count": 0, "centroid": None, "length_sum": 0.0, "bbox": None, "radius": 0.05}
        if xy_aabb is not None:
            summary["tile_count"] = 0
        return summary
    mid = (edges[:, :3] + edges[:, 3:6]) * 0.5
    lengths = torch.linalg.vector_norm(edges[:, 3:6] - edges[:, :3], dim=-1)
    kept = edges
    kept_mid = mid
    kept_len = lengths
    if xy_aabb is not None:
        lo_x, lo_y, hi_x, hi_y = xy_aabb
        mask = (mid[:, 0] >= lo_x) & (mid[:, 0] <= hi_x) & (mid[:, 1] >= lo_y) & (mid[:, 1] <= hi_y)
        kept = edges[mask]
        kept_mid = mid[mask]
        kept_len = lengths[mask]
    radius = float(getattr(getattr(obstacle, "cfg", None), "cylinder_radius", 0.05))
    summary: dict[str, object] = {
        "count": int(edges.shape[0]),
        "centroid": mid.mean(dim=0).tolist(),
        "length_sum": float(lengths.sum()),
        "bbox": [
            mid.min(dim=0).values.tolist(),
            mid.max(dim=0).values.tolist(),
        ],
        "radius": radius,
    }
    if xy_aabb is not None:
        summary["tile_count"] = int(kept.shape[0])
        summary["tile_centroid"] = kept_mid.mean(dim=0).tolist() if kept.numel() else None
        summary["tile_length_sum"] = float(kept_len.sum()) if kept.numel() else 0.0
    return summary


def pyramid_stairs_tile_aabb(terrain) -> tuple[float, float, float, float]:
    """XY box of row-0 / first ``pyramid_stairs`` column. Tiles are 8 m."""
    import torch

    from instinctlab.engines.pose_velocity import column_sub_terrain_names

    origins = getattr(terrain, "terrain_origins", None)
    if origins is None:
        generator = getattr(terrain, "terrain_generator", None)
        origins = getattr(generator, "terrain_origins", None)
    if origins is None:
        raise AssertionError("terrain has no terrain_origins to place a tile AABB")
    origins = torch.as_tensor(origins)
    names = column_sub_terrain_names(terrain)
    col = names.index("pyramid_stairs")
    origin = origins[0, col]
    half = 4.0
    return (
        float(origin[0] - half),
        float(origin[1] - half),
        float(origin[0] + half),
        float(origin[1] + half),
    )


def assert_volume_points_registered(env) -> dict[str, object]:
    """Startup registration must have run. Zero cylinders is the silent-zero failure."""
    from instinctlab.compat.sensors import registered_cylinder_count, require_volume_points_registered

    sensor = env.scene.sensors[VOLUME_SENSOR_NAME]
    require_volume_points_registered(sensor)
    count = registered_cylinder_count(sensor)
    assert count > 0, count
    terrain = env.scene.terrain
    full = summarize_virtual_obstacles(terrain)
    tile = summarize_virtual_obstacles(terrain, xy_aabb=pyramid_stairs_tile_aabb(terrain))
    assert int(full["count"]) > 0, full
    print(f"[volume-points] registered {count} cylinders; terrain {full}; pyramid_stairs tile {tile}")
    spawn = _spawn_to_cylinder_distances(env)
    print(
        "[volume-points] spawn-to-cylinder min/median/max "
        f"{spawn['min']:.3f}/{spawn['median']:.3f}/{spawn['max']:.3f} m; "
        f"inside r {spawn['inside_radius']}/{env.num_envs}; "
        f"within 0.5 m {spawn['within_half_m']}/{env.num_envs}"
    )
    return {"registered": count, "terrain": full, "pyramid_stairs_tile": tile, "spawn": spawn}


def _spawn_to_cylinder_distances(env) -> dict[str, float]:
    """How far each env origin is from the nearest generated cylinder midpoint."""
    import torch

    edges = env.scene.terrain.virtual_obstacles["edges"].edges_pyt
    mid = (edges[:, :3] + edges[:, 3:6]) * 0.5
    origins = env.scene.env_origins
    dist = torch.cdist(origins, mid)
    nearest = dist.min(dim=-1).values
    return {
        "min": float(nearest.min()),
        "median": float(nearest.median()),
        "max": float(nearest.max()),
        "inside_radius": int((nearest < TERRAIN_CYLINDER_RADIUS).sum()),
        "within_half_m": int((nearest < 0.5).sum()),
    }


def _pose_robot_standing(env, *, device: str, vel_x: float = 0.0):
    import torch

    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs, device=device)
    pos = env.scene.env_origins + torch.tensor((0.0, 0.0, 0.82), device=device)
    quat = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device).expand(env.num_envs, 4)
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    vel = torch.zeros(env.num_envs, 6, device=device)
    vel[:, 0] = vel_x
    robot.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
    q = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    env.scene.write_data_to_sim()
    if hasattr(env.sim, "forward"):
        env.sim.forward()
    if hasattr(robot, "update"):
        robot.update(env.physics_dt)
    return robot, env_ids


def _left_foot_points_fk(env, robot, *, device: str):
    import torch

    from instinctlab.compat.math import quat_apply
    from instinctlab.tasks.parkour.config.g1.target_env_cfg import LEG_VOLUME_POINTS

    bodies = list(robot.body_names)
    left = bodies.index("left_ankle_roll_link")
    ankle_pos = robot.data.body_link_pos_w[:, left]
    ankle_quat = robot.data.body_link_quat_w[:, left]
    local = torch.tensor(LEG_VOLUME_POINTS.grid.points(), device=device, dtype=ankle_pos.dtype)
    return ankle_pos.unsqueeze(1) + quat_apply(
        ankle_quat.unsqueeze(1).expand(-1, local.shape[0], -1),
        local.unsqueeze(0).expand(env.num_envs, -1, -1),
    )


def assert_terrain_generated_cylinder_penetration(env, *, device: str) -> dict[str, float]:
    """Put a foot inside a *generated* terrain cylinder and check the hand numbers.

    ``assert_known_volume_penetration`` replaces the terrain set with a fixture.
    Training reads these cylinders. A green fixture test plus a 0.0 training
    log is exactly the silent-zero path if generate() and the step query disagree.
    """
    import torch

    import instinctlab.mdp as mdp
    from instinctlab.compat.sensors import volume_points_penetration_offset, volume_points_vel_w
    from instinctlab.engines.volume_points import penetration_reward
    from instinctlab.tasks.parkour.config.g1.target_env_cfg import LEG_VOLUME_POINTS

    obstacle = env.scene.terrain.virtual_obstacles["edges"]
    edges = obstacle.edges_pyt
    radius = float(getattr(getattr(obstacle, "cfg", None), "cylinder_radius", TERRAIN_CYLINDER_RADIUS))
    lengths = torch.linalg.vector_norm(edges[:, 3:6] - edges[:, :3], dim=-1)
    usable = lengths > 0.2
    assert int(usable.sum()) > 0, "generate() produced no finite-length cylinders"

    robot, env_ids = _pose_robot_standing(env, device=device, vel_x=VOLUME_KNOWN_SPEED)
    expected_points = _left_foot_points_fk(env, robot, device=device)
    sample = expected_points[:, 0]

    mid = (edges[:, :3] + edges[:, 3:6]) * 0.5
    near = torch.cdist(sample, mid[usable]).argmin(dim=-1)
    chosen = edges[usable][near]
    starts, ends = chosen[:, :3], chosen[:, 3:6]
    axis = ends - starts
    unit = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    up = torch.zeros_like(unit)
    up[:, 2] = 1.0
    perp = torch.linalg.cross(unit, up)
    fallback = torch.linalg.cross(unit, torch.tensor((1.0, 0.0, 0.0), device=device).expand_as(unit))
    perp = torch.where(torch.linalg.vector_norm(perp, dim=-1, keepdim=True) > 1e-4, perp, fallback)
    perp = perp / torch.linalg.vector_norm(perp, dim=-1, keepdim=True)
    proj = starts + (0.5 * lengths[usable][near]).unsqueeze(-1) * unit
    target = proj + TERRAIN_AXIS_OFFSET * perp

    pos = robot.data.root_link_pos_w + (target - sample)
    quat = robot.data.root_link_quat_w
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    vel = torch.zeros(env.num_envs, 6, device=device)
    vel[:, 0] = VOLUME_KNOWN_SPEED
    robot.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
    env.scene.write_data_to_sim()
    if hasattr(env.sim, "forward"):
        env.sim.forward()
    if hasattr(robot, "update"):
        robot.update(env.physics_dt)
    _force_volume_refresh(env)

    sensor = env.scene.sensors[VOLUME_SENSOR_NAME]
    measured_offset = volume_points_penetration_offset(sensor)
    sample_offset = measured_offset[:, 0, 0]
    sample_depth = torch.linalg.vector_norm(sample_offset, dim=-1)
    expected_depth = radius - TERRAIN_AXIS_OFFSET
    import pytest

    assert float(sample_depth[0]) >= expected_depth - 8e-3, (
        f"terrain-cylinder depth {float(sample_depth[0]):.4f} < hand {expected_depth:.4f}; "
        "the training path is not reading the generated set"
    )
    assert float(sample_depth[0]) <= radius + 1e-3
    assert bool((sample_depth > 0.0).any()), "no env penetrated a generated cylinder"

    hand = _hand_cylinder_offset(sensor.data.points_pos_w[0].reshape(-1, 3), starts[0], ends[0], radius).reshape_as(
        sensor.data.points_pos_w[0]
    )
    hand_sample = float(hand[0, 0].norm())
    assert hand_sample == pytest.approx(expected_depth, abs=5e-3)

    speed = torch.linalg.vector_norm(volume_points_vel_w(sensor).flatten(1, 2), dim=-1)
    reward = mdp.volume_points_penetration(env, LEG_VOLUME_POINTS)
    assert float(reward[0]) > 0.0
    hand_reward = penetration_reward(float(sample_depth[0]), float(speed[0, 0]))
    print(
        f"[volume-points] terrain-cylinder env0 depth {float(sample_depth[0]):.4f} "
        f"hand {hand_sample:.4f} reward {float(reward[0]):.4f}"
    )
    return {
        "terrain_sample_depth": float(sample_depth[0]),
        "terrain_hand_depth": hand_sample,
        "terrain_reward0": float(reward[0]),
        "terrain_hand_reward": hand_reward,
        "chosen_length": float(lengths[usable][near][0]),
    }


class _KnownCylinder:
    """One finite cylinder the live sensor can register in place of the terrain set."""

    def __init__(self, start, end, radius: float, device: str):
        import numpy as np
        import torch

        from instinctlab.utils.warp.cylinder import CylinderSpatialGrid

        self.edges_pyt = torch.stack([torch.cat([start, end])], dim=0)
        self.cfg = type("Cfg", (), {"cylinder_radius": radius})()
        self.cylinders = CylinderSpatialGrid(
            cylinders=np.array(
                [[*start.detach().cpu().tolist(), *end.detach().cpu().tolist(), radius]],
                dtype=np.float32,
            ),
            num_grid_cells=8**3,
            device=str(device),
        )

    def get_points_penetration_offset(self, points):
        return self.cylinders.get_points_penetration_offset(points)


def _hand_cylinder_offset(points, start, end, radius: float):
    """Vector form of ``cylinder_penetration_offset`` for a batch of world points."""
    import torch

    axis = end - start
    length = torch.linalg.vector_norm(axis)
    unit = axis / length
    t = ((points - start) * unit).sum(dim=-1)
    inside = (t >= 0.0) & (t <= length)
    proj = start + t.unsqueeze(-1) * unit
    delta = points - proj
    dist = torch.linalg.vector_norm(delta, dim=-1)
    hit = inside & (dist > 0.0) & (dist < radius)
    scale = torch.zeros_like(dist)
    scale[hit] = (radius - dist[hit]) / dist[hit]
    return (proj - points) * scale.unsqueeze(-1)


def assert_known_volume_penetration(env, *, device: str) -> dict[str, float]:
    """Put a known cylinder next to a known point at a known speed; check the numbers.

    ω = 0 so Isaac's COM velocity and mjlab's link velocity agree. The offset is
    compared to the published kernel, not to the other engine.
    """
    import torch

    import instinctlab.mdp as mdp
    from instinctlab.compat.sensors import volume_points_penetration_offset, volume_points_vel_w
    from instinctlab.engines.volume_points import penetration_reward
    from instinctlab.tasks.parkour.config.g1.target_env_cfg import LEG_VOLUME_POINTS

    robot, _env_ids = _pose_robot_standing(env, device=device, vel_x=VOLUME_KNOWN_SPEED)
    expected_points = _left_foot_points_fk(env, robot, device=device)

    sensor = env.scene.sensors[VOLUME_SENSOR_NAME]
    terrain_set = dict(env.scene.terrain.virtual_obstacles)

    # Register the known cylinder against the first left-foot point of env 0.
    sample = expected_points[0, 0]
    start = sample + torch.tensor((VOLUME_KNOWN_AXIS_OFFSET, -0.5, 0.0), device=device, dtype=sample.dtype)
    end = sample + torch.tensor((VOLUME_KNOWN_AXIS_OFFSET, 0.5, 0.0), device=device, dtype=sample.dtype)
    known = _KnownCylinder(start, end, VOLUME_CYLINDER_RADIUS, device)
    try:
        sensor.register_virtual_obstacles({"known": known})
        _force_volume_refresh(env)

        measured_points = sensor.data.points_pos_w[:, 0]
        point_err = (measured_points - expected_points).norm(dim=-1)
        assert bool((point_err < 0.02).all()), (
            f"left-foot cloud is {point_err.max().item():.4f} m from attach-frame FK; "
            "the local grid or the quaternion convention is wrong"
        )

        expected_offset = _hand_cylinder_offset(
            sensor.data.points_pos_w.reshape(-1, 3), start, end, VOLUME_CYLINDER_RADIUS
        ).reshape_as(sensor.data.points_pos_w)
        measured_offset = volume_points_penetration_offset(sensor)
        offset_err = (measured_offset - expected_offset).norm(dim=-1)
        assert bool(
            (offset_err < 5e-3).all()
        ), f"penetration_offset off by {offset_err.max().item():.4f} m from the hand kernel"

        sample_depth = float(expected_offset[0, 0, 0].norm())
        import pytest

        assert sample_depth == pytest.approx(VOLUME_CYLINDER_RADIUS - VOLUME_KNOWN_AXIS_OFFSET, abs=2e-3)

        measured_vel = volume_points_vel_w(sensor)
        speed = torch.linalg.vector_norm(measured_vel.flatten(1, 2), dim=-1)
        # ω=0: every point should carry the written (1, 0, 0).
        assert bool((speed[0] > 0.5).any()), speed[0].max().item()

        depth = torch.linalg.vector_norm(measured_offset.flatten(1, 2), dim=-1)
        expected_reward = torch.sum((depth > 0.0).float() * (speed + 1e-6) * depth, dim=-1)
        reward = mdp.volume_points_penetration(env, LEG_VOLUME_POINTS)
        torch.testing.assert_close(reward, expected_reward, atol=1e-4, rtol=1e-4)
        hand = penetration_reward(sample_depth, float(speed[0, 0]))
        assert float(reward[0]) > 0.0
        return {
            "sample_depth": sample_depth,
            "sample_hand_reward": hand,
            "reward0": float(reward[0]),
            "max_offset_err": float(offset_err.max()),
            "max_point_err": float(point_err.max()),
        }
    finally:
        sensor.register_virtual_obstacles(terrain_set)
        _force_volume_refresh(env)


VOLUME_SPIN_SPEED = 2.0


def _body_com_pos_w(robot, body_index: int):
    """World COM of one body.

    mjlab's ``body_com_pos_w`` multiplies ``xquat`` (nworld, nbody, 4) by
    ``body_iquat`` (1, nbody, 4) and TorchScript refuses the broadcast.
    ``xipos`` is the same quantity without that product.
    """
    xipos = getattr(getattr(robot.data, "data", None), "xipos", None)
    indexing = getattr(robot.data, "indexing", None)
    if xipos is not None and indexing is not None:
        return xipos[:, indexing.body_ids][:, body_index]
    return robot.data.body_com_pos_w[:, body_index]


def _body_link_lin_vel_w(robot, body_index: int):
    """Link-origin linear velocity of one body.

    mjwarp ``cvel`` linear is at the free-joint subtree COM. That is what
    ``body_link_lin_vel_w`` already uses; a per-body subtree would add
    ``ω × (pelvis − ankle)`` and disagree with Isaac.
    """
    return robot.data.body_link_lin_vel_w[:, body_index]


def assert_known_volume_spin_velocity(env, *, device: str) -> dict[str, float]:
    """ω ≠ 0: point speed must be v_link + ω × r, not PhysX COM linear.

    Spin the left ankle hinges. The hand value is the hub formula from the
    attach body's link state. Pairing COM linear with a link lever arm is a
    different number; that is the case the denylist exists for. A root-only
    twist cannot be the check: mjlab's write+forward does not always carry
    the free-joint orbit into a child cvel, and ω = 0 hid the COM/link split.
    """
    import torch

    import pytest

    from instinctlab.compat.sensors import volume_points_vel_w
    from instinctlab.engines.volume_points import point_velocity_from_link

    robot, env_ids = _pose_robot_standing(env, device=device, vel_x=0.0)
    names = list(robot.joint_names)
    q = robot.data.default_joint_pos.clone()
    qd = torch.zeros_like(q)
    qd[:, names.index("left_ankle_roll_joint")] = VOLUME_SPIN_SPEED
    qd[:, names.index("left_ankle_pitch_joint")] = VOLUME_SPIN_SPEED
    robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
    env.scene.write_data_to_sim()
    if hasattr(env.sim, "forward"):
        env.sim.forward()
    if hasattr(robot, "update"):
        robot.update(env.physics_dt)
    _force_volume_refresh(env)

    sensor = env.scene.sensors[VOLUME_SENSOR_NAME]
    bodies = list(robot.body_names)
    left = bodies.index("left_ankle_roll_link")
    omega = robot.data.body_link_ang_vel_w[:, left]
    assert (
        float(omega[0].norm()) > 1.0
    ), f"ankle ω is {float(omega[0].norm()):.4f}; the hinge rate never reached the foot"
    origin = robot.data.body_link_pos_w[:, left]
    com = _body_com_pos_w(robot, left)
    v_link = _body_link_lin_vel_w(robot, left)
    v_com = v_link + torch.linalg.cross(omega, com - origin, dim=-1)
    com_gap = (v_com - v_link).norm(dim=-1)
    assert (
        float(com_gap[0]) > 0.02
    ), f"foot COM offset is {float((com-origin)[0].norm()):.4f} m; ω × r_com is too small to distinguish link from COM"

    # Parent still, hinge at the attach origin: v_link is centimetres/s, not
    # ω × (pelvis − ankle) ≈ 1.8 m/s. That was the per-body-subtree silent miss.
    assert float(v_link[0].norm()) < 0.15, (
        f"v_link is {v_link[0].tolist()} (|v|={float(v_link[0].norm()):.3f}); "
        "the ankle origin should be almost still. A pelvis-length lever means "
        "cvel was transported from the wrong COM."
    )

    reported = sensor.data.vel_w[:, 0]
    assert bool(((reported - v_link).norm(dim=-1) < 5e-2).all()), (
        f"sensor vel_w off the body's link linear by {float((reported-v_link).norm(dim=-1).max()):.4f} "
        f"(COM gap {float(com_gap[0]):.4f})"
    )
    com_reader = getattr(robot.data, "body_com_lin_vel_w", None)
    if com_reader is not None:
        try:
            engine_com = com_reader[:, left]
            assert (
                float((reported[0] - engine_com[0]).norm()) > 0.02
            ), "sensor vel_w matches the engine COM linear; attach_link conversion did not run"
        except Exception:
            pass

    points = sensor.data.points_pos_w[:, 0]
    measured = volume_points_vel_w(sensor)[:, 0]
    sample_hand = point_velocity_from_link(
        tuple(v_link[0].tolist()),
        tuple(omega[0].tolist()),
        tuple(origin[0].tolist()),
        tuple(points[0, 0].tolist()),
    )
    r = points - origin.unsqueeze(1)
    hand = v_link.unsqueeze(1) + torch.linalg.cross(omega.unsqueeze(1).expand_as(r), r, dim=-1)
    assert tuple(hand[0, 0].tolist()) == pytest.approx(sample_hand, abs=2e-3)
    err = (measured - hand).norm(dim=-1)
    assert bool((err < 5e-2).all()), (
        f"point velocity off the hand link formula by {float(err.max()):.4f} m/s; "
        f"measured={measured[0, 0].tolist()} hand={list(sample_hand)}"
    )
    mixed_hand = v_com.unsqueeze(1) + torch.linalg.cross(omega.unsqueeze(1).expand_as(r), r, dim=-1)
    mixed_err = (measured - mixed_hand).norm(dim=-1)
    assert (
        float(mixed_err[0].mean()) > float(err[0].mean()) + 0.01
    ), "measured velocity matches the COM-mixed formula as well as the link formula; ω ≠ 0 is not discriminating"
    print(
        f"[volume-points] spin ω={omega[0].tolist()} |ω|={float(omega[0].norm()):.3f} "
        f"v_link={v_link[0].tolist()} sample_hand={list(sample_hand)} "
        f"measured={measured[0, 0].tolist()} "
        f"link_err={float(err[0].max()):.4f} com_mixed_err={float(mixed_err[0].max()):.4f} "
        f"com_gap={float(com_gap[0]):.4f}"
    )
    return {
        "omega": float(omega[0].norm()),
        "omega_xyz": omega[0].tolist(),
        "link_err": float(err[0].max()),
        "com_mixed_err": float(mixed_err[0].max()),
        "com_gap": float(com_gap[0]),
        "v_link": v_link[0].tolist(),
        "sample_hand": list(sample_hand),
        "sample_measured": measured[0, 0].tolist(),
    }
