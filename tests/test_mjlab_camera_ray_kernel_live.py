"""Synthetic geometry live parity: ours pinhole vs InstinctMJ GroupedRayCaster ray kernel."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mujoco
import pytest
import torch
from instinctlab.engines.mjlab.camera import (
    pinhole_camera_geom_groups,
    pinhole_camera_hop_params,
)
from instinctlab_engine.spec.sensor import RayCasterRef, RayPatternRef

from tests.camera_ray_kernel_harness import (
    assert_hit_match,
    cast_rays,
    geom_name,
    make_ray_test_scene,
)

pytest.importorskip("mjlab")
pytestmark = pytest.mark.mjlab

MIN_DIST = 0.1
MAX_DIST = 5.0
GROUPS = pinhole_camera_geom_groups()
HOP = pinhole_camera_hop_params()

INSTINCTMJ_SRC = Path(os.environ.get("INSTINCTMJ_ROOT", "/root/InstinctMJ")) / "src"


def _require_instinctmj() -> None:
    if not INSTINCTMJ_SRC.is_dir():
        pytest.skip(f"InstinctMJ not checked out at {INSTINCTMJ_SRC.parent}")
    if str(INSTINCTMJ_SRC) not in sys.path:
        sys.path.insert(0, str(INSTINCTMJ_SRC))


def _ours_sensor_cfg(name: str = "ours", *, perceptive: bool = False):
    from instinctlab.engines.mjlab.camera import pinhole_ray_caster

    ref = RayCasterRef(
        name=name,
        attach="cam",
        pattern=RayPatternRef(kind="pinhole", width=1, height=1),
        hit=("terrain",),
        ray_alignment="base",
        max_distance=MAX_DIST,
        min_distance=MIN_DIST,
    )
    profile = {
        "pinhole_cameras": {
            name: {
                "include_geom_groups": (0, 2) if perceptive else (0, 1, 2),
                "exclude_parent_body": not perceptive,
                "mesh_filter_max_hops": 24 if perceptive else 6,
                "mesh_filter_epsilon": 1.0e-4,
                "update_period": 1.0 / 60.0 if perceptive else 0.0,
            }
        }
    }
    return pinhole_ray_caster(ref, profile)


def _instinctmj_sensor_cfg(name: str = "ref", *, perceptive: bool = False):
    _require_instinctmj()
    from instinct_mj.sensors.grouped_ray_caster import GroupedRayCasterCfg
    from mjlab.sensor import GridPatternCfg, ObjRef

    kwargs = {}
    if perceptive:
        kwargs = {
            "exclude_parent_body": False,
            "mesh_filter_max_hops": 24,
        }
    return GroupedRayCasterCfg(
        name=name,
        frame=ObjRef(type="body", name="cam", entity="robot"),
        pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1, direction=(1.0, 0.0, 0.0)),
        ray_alignment="base",
        max_distance=MAX_DIST,
        min_distance=MIN_DIST,
        include_geom_groups=(0, 2) if perceptive else GROUPS,
        mesh_prim_paths=[],
        **kwargs,
    )


def _run_pair(
    xml: str,
    device: str,
    *,
    perceptive: bool = False,
) -> tuple[tuple[int, float], tuple[int, float], mujoco.MjModel]:
    ours_cfg = _ours_sensor_cfg(perceptive=perceptive)
    ref_cfg = _instinctmj_sensor_cfg(perceptive=perceptive)
    scene, sim = make_ray_test_scene(device, xml, sensors=(ours_cfg, ref_cfg))
    try:
        ours = cast_rays(scene, sim, "ours")
        ref = cast_rays(scene, sim, "ref")
        return ours, ref, sim.mj_model
    finally:
        del sim, scene


@pytest.fixture(scope="module")
def device():
    if os.environ.get("FORCE_CPU") == "1":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_hop_params_read_from_instinctmj_source() -> None:
    assert HOP["filter"] == "geom_groups_min_distance_hop"
    assert HOP["hop_max"] == 6
    assert HOP["hop_epsilon_m"] == pytest.approx(1e-4)
    assert HOP["mesh_prim_paths_enabled"] is False


def test_calibration_noise_mutates_local_rays_like_instinctmj(device) -> None:
    from instinctlab_engine.bridge.math import quat_apply, quat_from_euler_xyz, quat_mul
    from instinctlab.engines.mjlab.camera import pinhole_ray_caster

    offset = (0.2, -0.1, 0.3)
    offset_rot = (0.9393727, 0.0, 0.3428978, 0.0)
    ref = RayCasterRef(
        name="camera",
        attach="cam",
        pattern=RayPatternRef(kind="pinhole", width=1, height=1),
        hit=("terrain",),
        ray_alignment="base",
        max_distance=MAX_DIST,
        min_distance=0.0,
        offset=offset,
        offset_rot=offset_rot,
        offset_convention="world",
    )
    xml = """
    <mujoco>
      <worldbody>
        <body name="cam" pos="0.4 -0.2 0.7" quat="0.9659258 0.0 0.0 0.2588190">
          <freejoint name="free"/>
          <geom name="probe" type="sphere" size="0.005" contype="0" conaffinity="0" group="3"/>
        </body>
        <geom name="bvh_anchor" type="box" size="0.1 0.1 0.1" pos="10 0 0" group="0"/>
      </worldbody>
    </mujoco>
    """
    profile = {
        "pinhole_cameras": {
            "camera": {
                "include_geom_groups": (0, 1, 2),
                "exclude_parent_body": True,
                "mesh_filter_max_hops": 6,
                "mesh_filter_epsilon": 1.0e-4,
                "update_period": 0.0,
            }
        }
    }
    scene, sim = make_ray_test_scene(
        device, xml, sensors=(pinhole_ray_caster(ref, profile),)
    )
    try:
        sensor = scene["camera"]
        calibration = torch.tensor([[0.11, -0.07, 0.05, 0.2, -0.1, 0.3]], device=device)
        sensor.set_offset_noise(torch.tensor([0], device=device), calibration)
        sim.forward()
        sensor.prepare_rays()

        body_id = sensor._frame_infos[0][1]
        frame_pos = sim.data.xpos[:, body_id]
        frame_quat = sim.data.xquat[:, body_id]
        offset_pos = torch.tensor(offset, device=device).expand_as(frame_pos)
        offset_quat = torch.tensor(offset_rot, device=device).expand_as(frame_quat)
        camera_pos = frame_pos + quat_apply(frame_quat, offset_pos)
        camera_quat = quat_mul(frame_quat, offset_quat)
        delta_quat = quat_from_euler_xyz(*calibration[:, 3:].T)
        expected_origins = camera_pos.unsqueeze(1) + quat_apply(
            camera_quat,
            sensor._local_offsets.unsqueeze(0) + calibration[:, None, :3],
        )
        expected_directions = quat_apply(
            quat_mul(camera_quat, delta_quat),
            sensor._local_directions.unsqueeze(0),
        )

        torch.testing.assert_close(sensor._cached_world_origins, expected_origins)
        torch.testing.assert_close(sensor._cached_world_rays, expected_directions)
        torch.testing.assert_close(sensor._reported_cam_pos, camera_pos)
        torch.testing.assert_close(sensor._reported_cam_quat, camera_quat)
    finally:
        del sim, scene


def test_min_distance_hop_skips_near_allowed_geom(device) -> None:
    """First allowed hit at ~0.05 m; second at ~0.59 m. min_distance=0.1 → take far."""
    if device == "cpu":
        pytest.skip("InstinctMJ's continuation kernel returns miss on CPU; production parity is CUDA")
    xml = """
    <mujoco>
      <worldbody>
        <body name="cam" pos="0 0 0">
          <freejoint name="free"/>
          <geom name="probe" type="sphere" size="0.005" contype="0" conaffinity="0" group="3"/>
        </body>
        <geom name="near" type="box" size="0.01 0.05 0.05" pos="0.06 0 0" group="0"/>
        <geom name="far" type="box" size="0.01 0.05 0.05" pos="0.6 0 0" group="0"/>
      </worldbody>
    </mujoco>
    """
    ours, ref, model = _run_pair(xml, device)
    assert_hit_match(model, ours, ref, dist_tol=0.02)
    assert ours[1] == pytest.approx(0.59, abs=0.03)
    assert ours[1] > MIN_DIST


def test_disallowed_group_is_bvh_filtered(device) -> None:
    """Group-3 near occluder must not appear; first allowed hit is far box."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="cam" pos="0 0 0">
          <freejoint name="free"/>
          <geom name="probe" type="sphere" size="0.005" contype="0" conaffinity="0" group="2"/>
        </body>
        <geom name="blocked" type="box" size="0.01 0.05 0.05" pos="0.06 0 0" group="3"/>
        <geom name="far" type="box" size="0.01 0.05 0.05" pos="0.6 0 0" group="0"/>
      </worldbody>
    </mujoco>
    """
    ours, ref, model = _run_pair(xml, device)
    assert_hit_match(model, ours, ref, dist_tol=0.02)
    assert ours[1] == pytest.approx(0.59, abs=0.03)


def test_perceptive_group_one_occluder_is_ignored(device) -> None:
    """Perceptive explicitly narrows Parkour's (0,1,2) mask to (0,2)."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="cam" pos="0 0 0">
          <freejoint name="free"/>
          <geom name="probe" type="sphere" size="0.005" contype="0" conaffinity="0" group="3"/>
        </body>
        <geom name="group_one_occluder" type="box" size="0.01 0.05 0.05" pos="0.3 0 0" group="1"/>
        <geom name="far" type="box" size="0.01 0.05 0.05" pos="0.6 0 0" group="0"/>
      </worldbody>
    </mujoco>
    """
    ours, ref, model = _run_pair(xml, device, perceptive=True)
    assert_hit_match(model, ours, ref, dist_tol=0.02)
    assert geom_name(model, ours[0]).endswith("far")
    assert ours[1] == pytest.approx(0.59, abs=0.03)


def test_all_miss_when_only_disallowed_groups(device) -> None:
    xml = """
    <mujoco>
      <worldbody>
        <body name="cam" pos="0 0 0">
          <freejoint name="free"/>
          <geom name="probe" type="sphere" size="0.005" contype="0" conaffinity="0" group="2"/>
        </body>
        <geom name="blocked" type="box" size="0.01 0.05 0.05" pos="0.6 0 0" group="3"/>
      </worldbody>
    </mujoco>
    """
    ours, ref, model = _run_pair(xml, device)
    assert_hit_match(model, ours, ref)
    assert ours[0] < 0


def test_mutation_no_hop_would_miss_far_geom(device) -> None:
    """If min_distance hop were removed, near geom would win and this would fail."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="cam" pos="0 0 0">
          <freejoint name="free"/>
          <geom name="probe" type="sphere" size="0.005" contype="0" conaffinity="0" group="2"/>
        </body>
        <geom name="near" type="box" size="0.01 0.05 0.05" pos="0.06 0 0" group="0"/>
        <geom name="far" type="box" size="0.01 0.05 0.05" pos="0.6 0 0" group="0"/>
      </worldbody>
    </mujoco>
    """
    ours, _, _model = _run_pair(xml, device)
    assert ours[1] == pytest.approx(0.59, abs=0.03)
    assert ours[1] > MIN_DIST
