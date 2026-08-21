"""Synthetic geometry live parity: ours pinhole vs InstinctMJ GroupedRayCaster ray kernel."""

from __future__ import annotations

import os
import sys
import torch
from pathlib import Path

import mujoco
import pytest

from instinctlab.engines.mjlab.camera import pinhole_camera_geom_groups, pinhole_camera_hop_params
from instinctlab.spec.sensor import RayCasterRef, RayPatternRef
from tests.camera_ray_kernel_harness import assert_hit_match, cast_rays, geom_name, make_ray_test_scene

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


def _ours_sensor_cfg(name: str = "ours"):
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
    return pinhole_ray_caster(ref)


def _instinctmj_sensor_cfg(name: str = "ref"):
    _require_instinctmj()
    from instinct_mj.sensors.grouped_ray_caster import GroupedRayCasterCfg
    from mjlab.sensor import GridPatternCfg, ObjRef

    return GroupedRayCasterCfg(
        name=name,
        frame=ObjRef(type="body", name="cam", entity="robot"),
        pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1, direction=(1.0, 0.0, 0.0)),
        ray_alignment="base",
        max_distance=MAX_DIST,
        min_distance=MIN_DIST,
        include_geom_groups=GROUPS,
        mesh_prim_paths=[],
    )


def _run_pair(xml: str, device: str) -> tuple[tuple[int, float], tuple[int, float], mujoco.MjModel]:
    ours_cfg = _ours_sensor_cfg()
    ref_cfg = _instinctmj_sensor_cfg()
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


def test_min_distance_hop_skips_near_allowed_geom(device) -> None:
    """First allowed hit at ~0.05 m; second at ~0.59 m. min_distance=0.1 → take far."""
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
    ours, _, model = _run_pair(xml, device)
    assert ours[1] == pytest.approx(0.59, abs=0.03)
    assert ours[1] > MIN_DIST
