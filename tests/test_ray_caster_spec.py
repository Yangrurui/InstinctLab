"""The foot-height scanner IR, and the builder choices that would otherwise be silent.

Cheap: no GPU. Isaac's builder body is reached only for the ``against`` refusal and for
the ray-caster cfg that does not import ``omni`` until ``isaaclab.sensors`` is loaded --
the against check runs first. mjlab cfg construction needs the package, not a device.
"""

from __future__ import annotations

import math
import torch
from types import SimpleNamespace

import pytest

from instinctlab.compat import sensors as compat_sensors
from instinctlab.compat.denylist import PortabilityError
from instinctlab.spec.sensor import ContactSensorRef, RayCasterRef, RayPatternRef
from instinctlab.spec.task import SceneSpec


def test_a_ray_caster_states_isaac_semantics_and_refuses_the_zero_miss() -> None:
    sensor = RayCasterRef(
        name="left_height_scanner",
        attach="left_ankle_roll_link",
        offset=(0.04, 0.0, 20.0),
        pattern=RayPatternRef(resolution=0.12, size=(0.12, 0.0)),
    )
    assert sensor.hit == "terrain"
    assert sensor.miss == "infinity"
    assert sensor.ray_alignment == "yaw"
    assert sensor.max_distance == 1e6
    with pytest.raises(ValueError, match="attach body"):
        RayCasterRef(name="x", attach="")
    with pytest.raises(ValueError, match="non-positive max_distance"):
        RayCasterRef(name="x", attach="foot", max_distance=0.0)


def test_the_grid_this_increment_uses_is_two_rays() -> None:
    """size=(0.12, 0) / resolution=0.12 is two samples on x, one on y.

    Isaac's arange and mjlab's arange use slightly different inclusive ends; for
    these numbers they agree. A third ray would be a silent observation-width change.
    """
    size_x, size_y = 0.12, 0.0
    res = 0.12
    isaac_x = torch.arange(-size_x / 2, size_x / 2 + 1.0e-9, res)
    isaac_y = torch.arange(-size_y / 2, size_y / 2 + 1.0e-9, res)
    mjlab_x = torch.arange(-size_x / 2, size_x / 2 + res * 0.5, res)
    mjlab_y = torch.arange(-size_y / 2, size_y / 2 + res * 0.5, res)
    assert isaac_x.tolist() == pytest.approx([-0.06, 0.06])
    assert mjlab_x.tolist() == pytest.approx([-0.06, 0.06])
    assert isaac_y.tolist() == pytest.approx([0.0])
    assert mjlab_y.tolist() == pytest.approx([0.0])
    assert isaac_x.numel() * isaac_y.numel() == 2


def test_isaac_refuses_against_rather_than_emitting_an_unfiltered_sensor() -> None:
    from instinctlab.engines.isaacsim.scene import _contact_sensor

    with pytest.raises(ValueError, match="cannot honor ContactSensorRef.against"):
        _contact_sensor(ContactSensorRef(name="contact_forces", elements=".*", against="terrain"))


def test_mjlab_still_passes_against_as_a_secondary_match() -> None:
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab.scene import _contact_sensor

    cfg = _contact_sensor(ContactSensorRef(name="contact_forces", elements=".*", against="terrain"))
    assert cfg.secondary is not None
    assert cfg.secondary.pattern == ("terrain",)


def test_mjlab_ray_caster_cfg_is_sky_origin_and_not_the_stock_group_mask() -> None:
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab.scene import _ray_caster

    cfg = _ray_caster(
        RayCasterRef(
            name="left_height_scanner",
            attach="left_ankle_roll_link",
            offset=(0.04, 0.0, 20.0),
        )
    )
    assert cfg.origin_offset == (0.04, 0.0, 20.0)
    assert cfg.ray_alignment == "yaw"
    assert cfg.max_distance == 1e6
    assert cfg.include_geom_groups == ()
    assert cfg.pattern.direction == (0.0, 0.0, -1.0)
    assert cfg.pattern.size == (0.12, 0.0)
    assert cfg.pattern.resolution == 0.12


def test_pinhole_yaw_is_refused_because_both_engines_ignore_it() -> None:
    """A pinhole + yaw would compile and look fine. The image would be full-R."""
    from instinctlab.engines.ray_alignment import refuse_unhonored_ray_alignment

    yaw_camera = RayCasterRef(
        name="camera",
        attach="torso_link",
        pattern=RayPatternRef(kind="pinhole", width=4, height=4),
        hit=("terrain",),
        ray_alignment="yaw",
    )
    with pytest.raises(ValueError, match="silently ignored"):
        refuse_unhonored_ray_alignment(yaw_camera)
    with pytest.raises(ValueError, match="silently ignored"):
        refuse_unhonored_ray_alignment(
            RayCasterRef(
                name="camera",
                attach="torso_link",
                pattern=RayPatternRef(kind="pinhole", width=4, height=4),
                hit=("terrain",),
                ray_alignment="world",
            )
        )
    from instinctlab.engines.isaacsim.scene import _ray_caster as isaac_ray_caster

    with pytest.raises(ValueError, match="silently ignored"):
        isaac_ray_caster(yaw_camera, sensor_period=0.02)
    from instinctlab.engines.mjlab.scene import _ray_caster as mjlab_ray_caster

    with pytest.raises(ValueError, match="silently ignored"):
        mjlab_ray_caster(yaw_camera)
    refuse_unhonored_ray_alignment(
        RayCasterRef(
            name="camera",
            attach="torso_link",
            pattern=RayPatternRef(kind="pinhole", width=4, height=4),
            hit=("terrain",),
            ray_alignment="base",
        )
    )
    refuse_unhonored_ray_alignment(
        RayCasterRef(
            name="left_height_scanner",
            attach="foot",
            offset=(0.04, 0.0, 20.0),
            pattern=RayPatternRef(kind="grid", resolution=0.12, size=(0.12, 0.0)),
            ray_alignment="yaw",
        )
    )
    refuse_unhonored_ray_alignment(
        RayCasterRef(
            name="left_height_scanner",
            attach="foot",
            offset=(0.04, 0.0, 20.0),
            pattern=RayPatternRef(kind="grid", resolution=0.12, size=(0.12, 0.0)),
            ray_alignment="base",
        )
    )


def test_mjlab_grid_passes_base_alignment_through() -> None:
    """Grid + base is honoured on both engines; do not start refusing it."""
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab.scene import _ray_caster

    cfg = _ray_caster(
        RayCasterRef(
            name="left_height_scanner",
            attach="left_ankle_roll_link",
            offset=(0.04, 0.0, 20.0),
            pattern=RayPatternRef(kind="grid", resolution=0.12, size=(0.12, 0.0)),
            ray_alignment="base",
        )
    )
    assert cfg.ray_alignment == "base"


def test_a_pitched_torso_makes_yaw_and_full_rotation_camera_poses_disagree() -> None:
    """Level-torso checks cannot see ray_alignment. This is the distinguishing pose.

    At 0.6 rad pitch the 0.44 m camera offset slides ~0.33 m under full-R
    and stays yaw-only under yaw. If a later edit dropped pitch/roll, this
    gap closes only when the torso is upright -- the silent parkour failure.
    """
    from instinctlab.compat.math import quat_from_euler_xyz
    from instinctlab.engines.ray_alignment import camera_pose_for_alignment
    from tests.parkour_live_expect import (
        CAMERA_OFFSET,
        CAMERA_OFFSET_ROT,
        CAMERA_TILT_PITCH,
        CAMERA_TILT_ROLL,
        CAMERA_TILT_YAW,
        CAMERA_YAW_SEP_M,
    )

    pos = torch.zeros(1, 3)
    quat = quat_from_euler_xyz(
        torch.tensor([CAMERA_TILT_ROLL]),
        torch.tensor([CAMERA_TILT_PITCH]),
        torch.tensor([CAMERA_TILT_YAW]),
    )
    offset = torch.tensor(CAMERA_OFFSET)
    rot = torch.tensor(CAMERA_OFFSET_ROT)
    yaw_pos, _ = camera_pose_for_alignment(pos, quat, offset, rot, "yaw")
    full_pos, _ = camera_pose_for_alignment(pos, quat, offset, rot, "base")
    slide = (yaw_pos - full_pos).norm().item()
    assert slide > CAMERA_YAW_SEP_M, slide


def test_a_pitched_foot_makes_yaw_and_full_rotation_origins_disagree() -> None:
    """Level-foot checks cannot see ray_alignment. This is the distinguishing pose.

    At 0.8 rad pitch the 20 m offset slides ~12 m under full-R and stays world-up
    under yaw. If a later edit applies the attach body's full R, this gap closes
    only when the foot is level again -- the silent parkour failure.
    """
    from instinctlab.compat.math import quat_from_euler_xyz
    from tests.parkour_live_expect import SCANNER_OFFSET, scanner_origins_for_alignment

    pos = torch.zeros(1, 3)
    quat = quat_from_euler_xyz(torch.tensor([0.5]), torch.tensor([0.8]), torch.tensor([0.2]))
    offset = torch.tensor(SCANNER_OFFSET)
    yaw = scanner_origins_for_alignment(pos, quat, offset, "yaw")
    full = scanner_origins_for_alignment(pos, quat, offset, "base")
    slide = (yaw - full).norm().item()
    assert slide > 2.0, slide
    assert abs(yaw[0, 2] - 20.0) < 0.02
    assert abs(full[0, 2] - 20.0) > 1.0


def test_ensure_warp_ray_on_device_is_a_no_op_on_cpu() -> None:
    from instinctlab.engines.mjlab.ray_device import ensure_warp_ray_on_device

    ensure_warp_ray_on_device("cpu")
    ensure_warp_ray_on_device("cpu:0")


def test_ray_hits_w_turns_a_negative_distance_into_infinity() -> None:
    """mjlab's stock miss is distance=-1 and hit_pos at the origin. That is not z=0."""

    class _Data:
        hit_pos_w = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        distances = torch.tensor([[1.5, -1.0]])

    class _Sensor:
        data = _Data()

    hits = compat_sensors.ray_hits_w(_Sensor())
    assert hits[0, 0].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert math.isinf(hits[0, 1, 0]) and hits[0, 1, 0] > 0
    assert math.isinf(hits[0, 1, 2]) and hits[0, 1, 2] > 0


def test_ray_hits_w_leaves_isaac_infinity_alone() -> None:
    class _Data:
        ray_hits_w = torch.tensor([[[0.0, 0.0, float("inf")]]])

    class _Sensor:
        data = _Data()

    hits = compat_sensors.ray_hits_w(_Sensor())
    assert math.isinf(hits[0, 0, 2])


def test_ray_hits_w_refuses_an_unknown_sensor() -> None:
    class _Sensor:
        data = object()

    with pytest.raises(PortabilityError, match="neither ray_hits_w nor hit_pos_w"):
        compat_sensors.ray_hits_w(_Sensor())


def test_scene_lists_ray_casters() -> None:
    scanner = RayCasterRef(name="left_height_scanner", attach="foot")
    scene = SceneSpec(ray_casters=(scanner,))
    assert scene.ray_caster("left_height_scanner") is scanner


def test_terrain_mask_is_the_terrain_body_not_a_group_number() -> None:
    """Group 2 on the G1 is the visual shoe. A group mask is not terrain-only."""
    mujoco = pytest.importorskip("mujoco")
    from instinctlab.engines.mjlab.raycast import _terrain_geom_mask

    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <worldbody>
            <body name="terrain">
              <geom name="ground" type="plane" size="1 1 0.01" group="0"/>
            </body>
            <body name="robot">
              <geom name="shoe_visual" type="sphere" size="0.05" group="2"/>
            </body>
          </worldbody>
        </mujoco>
        """)
    mask, groups = _terrain_geom_mask(model, "cpu")
    assert groups == (0,)
    assert bool(mask[0]) and not bool(mask[1])


def test_a_pinhole_states_intrinsics_crop_and_an_explicit_hit_list() -> None:
    sensor = RayCasterRef(
        name="camera",
        attach="torso_link",
        pattern=RayPatternRef(kind="pinhole", width=64, height=36),
        hit=("terrain", "left_ankle_roll_link"),
        ray_alignment="base",
        max_distance=2.5,
        min_distance=0.1,
        crop=(18, 0, 16, 16),
    )
    assert sensor.hits_terrain()
    assert sensor.hit_bodies() == ("left_ankle_roll_link",)
    assert sensor.cropped_hw() == (18, 32)
    assert sensor.pattern.horizontal_aperture == pytest.approx(2.0 * 1.0 * math.tan(math.radians(89.51) / 2.0))
    with pytest.raises(ValueError, match="empty hit list"):
        RayCasterRef(name="x", attach="torso", pattern=RayPatternRef(kind="pinhole"), hit=())
    with pytest.raises(ValueError, match="terrain-only"):
        RayCasterRef(name="x", attach="foot", hit=("terrain", "shoe"))
    with pytest.raises(ValueError, match="leaves a"):
        RayCasterRef(
            name="x",
            attach="torso",
            pattern=RayPatternRef(kind="pinhole", width=64, height=36),
            hit=("terrain",),
            crop=(20, 20, 0, 0),
        )


def test_depth_image_turns_nan_into_infinity() -> None:
    class _Data:
        output = {"distance_to_image_plane": torch.tensor([[[[1.0], [float("nan")]]]])}

    class _Sensor:
        data = _Data()

    image = compat_sensors.depth_image(_Sensor())
    assert image[0, 0, 0, 0] == pytest.approx(1.0)
    assert math.isinf(image[0, 0, 1, 0]) and image[0, 0, 1, 0] > 0


def test_depth_image_turns_a_past_far_plane_hit_into_infinity() -> None:
    """Isaac returns finite depths past max_distance; those are misses."""

    class _Data:
        output = {"distance_to_image_plane": torch.tensor([[[[1.5], [3.37]]]])}

    class _Sensor:
        data = _Data()
        cfg = SimpleNamespace(max_distance=2.5)

    image = compat_sensors.depth_image(_Sensor())
    assert image[0, 0, 0, 0] == pytest.approx(1.5)
    assert math.isinf(image[0, 0, 1, 0]) and image[0, 0, 1, 0] > 0


def test_mjlab_camera_cfg_is_world_convention_and_not_a_group_mask() -> None:
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab.scene import _ray_caster

    cfg = _ray_caster(
        RayCasterRef(
            name="camera",
            attach="torso_link",
            offset=(0.05, 0.01, 0.44),
            offset_rot=(0.9, 0.0, 0.4, 0.0),
            pattern=RayPatternRef(kind="pinhole"),
            hit=("terrain", "left_ankle_roll_link"),
            ray_alignment="base",
            max_distance=2.5,
            min_distance=0.1,
        )
    )
    assert cfg.origin_offset == (0.05, 0.01, 0.44)
    assert cfg.origin_offset_rot[0] == pytest.approx(0.9)
    assert cfg.max_distance == 5.0
    assert cfg.image_plane_max == 2.5
    assert cfg.min_distance == 0.1
    assert cfg.include_geom_groups is None
    assert cfg.ray_alignment == "base"
    assert cfg.pattern.width == 64
    assert cfg.pattern.height == 36


def test_camera_mask_is_named_bodies_not_a_group_number() -> None:
    """Group 2 is the visual shoe. Listing the ankle must not pull in every group-2 geom.

    A listed body with a mesh keeps the mesh and drops its collision capsule --
    Isaac's ``/visuals`` targets do the same. A capsule-only body is the fallback
    in the next test, not this one.
    """
    mujoco = pytest.importorskip("mujoco")
    from instinctlab.engines.mjlab.camera import _camera_geom_mask

    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <asset>
            <mesh name="shoe" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"/>
          </asset>
          <worldbody>
            <body name="terrain">
              <geom name="ground" type="plane" size="1 1 0.01" group="0"/>
            </body>
            <body name="left_ankle_roll_link">
              <geom name="shoe_visual" type="mesh" mesh="shoe" group="2"/>
              <geom name="shoe_collision" type="capsule" size="0.03 0.02" group="3"/>
            </body>
            <body name="head_link">
              <geom name="head_visual" type="sphere" size="0.08" group="2"/>
            </body>
          </worldbody>
        </mujoco>
        """)
    mask = _camera_geom_mask(model, bodies=("left_ankle_roll_link",), include_terrain=True, device="cpu")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]
    allowed = {names[i] for i, keep in enumerate(mask.tolist()) if keep}
    assert "ground" in allowed
    assert "shoe_visual" in allowed
    assert "shoe_collision" not in allowed
    assert "head_visual" not in allowed


def test_camera_mask_falls_back_to_every_geom_on_a_meshless_body() -> None:
    """A capsule-only listed link must stay visible rather than silently vanish."""
    mujoco = pytest.importorskip("mujoco")
    from instinctlab.engines.mjlab.camera import _camera_geom_mask

    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <worldbody>
            <body name="terrain">
              <geom name="ground" type="plane" size="1 1 0.01" group="0"/>
            </body>
            <body name="left_ankle_roll_link">
              <geom name="shoe_collision" type="capsule" size="0.03 0.02" group="3"/>
            </body>
          </worldbody>
        </mujoco>
        """)
    mask = _camera_geom_mask(model, bodies=("left_ankle_roll_link",), include_terrain=True, device="cpu")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]
    allowed = {names[i] for i, keep in enumerate(mask.tolist()) if keep}
    assert allowed == {"ground", "shoe_collision"}


def test_processed_depth_turns_infinity_into_the_normalisation_ceiling() -> None:
    from instinctlab.mdp.observations import _process_depth_image

    sensor = RayCasterRef(
        name="camera",
        attach="torso_link",
        pattern=RayPatternRef(kind="pinhole", width=4, height=4),
        hit=("terrain",),
        max_distance=2.5,
        crop=(1, 1, 1, 1),
    )
    raw = torch.full((2, 4, 4, 1), 1.25)
    raw[:, 0, 0] = float("inf")
    processed = _process_depth_image(raw, sensor, kernel_size=1, sigma=0.0)
    assert tuple(processed.shape) == (2, 2, 2)
    assert processed.max() <= 1.0 + 1e-6
    raw_miss = torch.full((1, 4, 4, 1), float("inf"))
    miss = _process_depth_image(raw_miss, sensor, kernel_size=1, sigma=0.0)
    assert torch.allclose(miss, torch.ones_like(miss))


def test_terrain_mask_refuses_a_non_terrain_geom_in_the_same_group() -> None:
    mujoco = pytest.importorskip("mujoco")
    from instinctlab.engines.mjlab.raycast import _terrain_geom_mask

    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <worldbody>
            <body name="terrain">
              <geom name="ground" type="plane" size="1 1 0.01" group="0"/>
            </body>
            <body name="robot">
              <geom name="shoe" type="sphere" size="0.05" group="0"/>
            </body>
          </worldbody>
        </mujoco>
        """)
    with pytest.raises(RuntimeError, match="non-terrain geoms share them"):
        _terrain_geom_mask(model, "cpu")


def test_portable_terms_do_not_read_ray_hits_off_sensor_data() -> None:
    """``sensor.data.ray_hits_w`` is Isaac's name; mjlab's is ``hit_pos_w``. Go through compat."""
    import ast
    from pathlib import Path

    import instinctlab.mdp as mdp

    forbidden = {"ray_hits_w", "hit_pos_w"}
    offenders: dict[str, set[str]] = {}
    for source in Path(mdp.__file__).parent.glob("*.py"):
        tree = ast.parse(source.read_text())
        for function in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]:
            for node in ast.walk(function):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "data"
                    and node.attr in forbidden
                ):
                    offenders.setdefault(node.attr, set()).add(function.name)
    assert not offenders, offenders
