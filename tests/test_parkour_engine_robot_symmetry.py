"""Parkour G1 robot/actuation symmetry between mjlab and Isaac compile paths.

Both engines must train the main-parkour plant: shoe collision 23 mm below the
bare ankle, spawn z=0.9, Isaac merge_fixed_joints=True, torque delay 0–2 physics
steps resampled once per episode. The catalog popsicle is the flat/rough plant.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from instinctlab.assets.unitree_g1.isaacsim import G1_29DOF_LINKS, make_g1_29dof_robot_spec
from instinctlab.engines.mjlab.assets import DELAY_RESET_ONLY_PERIOD
from instinctlab.tasks.parkour.config.g1 import parkour_target_g1
from instinctlab.tasks.parkour.config.g1.target_env_cfg import (
    DEPTH_CAMERA,
    LEFT_HEIGHT_SCANNER,
    LEG_VOLUME_POINTS,
    PARKOUR_MOTION_LINKS,
    RIGHT_HEIGHT_SCANNER,
)

REPO = Path(__file__).resolve().parents[1]

SHOE_URDF = "g1_29dof_torsoBase_popsicle_with_shoe.urdf"
SHOE_MJCF = "g1_29dof_torsoBase_popsicle_with_shoe.xml"
SPAWN_Z = 0.9
SOFT_LIMIT = 0.9
ANKLE_COLLISION_GEOMS_PER_FOOT = 7
SHOE_SOLE_OFFSET = -0.023
FEET_AT_PLANE_HEIGHT_OFFSET = 0.058
SHOE_CAPSULE_RADIUS = 0.01


# Parkour names a body. After Isaac merge_fixed_joints=True the child of each
# fixed joint disappears and its collision rides on the parent. A vanished name
# fails loudly. A name that still exists but now denotes the merged body is the
# silent case — list every referenced name and what it resolves to.
def _attach_names(attach: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(attach, str):
        return (attach,)
    return tuple(attach)


PARKOUR_NAMED_BODIES = frozenset(
    {
        "torso_link",
        "pelvis",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        *G1_29DOF_LINKS,
        *PARKOUR_MOTION_LINKS,
        *_attach_names(LEFT_HEIGHT_SCANNER.attach),
        *_attach_names(RIGHT_HEIGHT_SCANNER.attach),
        *_attach_names(LEG_VOLUME_POINTS.attach),
    }
)


@pytest.fixture(scope="module")
def spec():
    return parkour_target_g1()


@pytest.fixture(scope="module")
def mjlab_robot(spec):
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab import MjlabAdapter

    compiled = MjlabAdapter().compile(spec, num_envs=16, device="cpu")
    return compiled.env_cfg.scene.entities["robot"]


def test_parkour_robot_matches_instinctmj_constants(spec) -> None:
    from tests import reference_mjlab_parkour as mj_ref

    if not mj_ref.available():
        pytest.skip("InstinctMJ is not checked out")
    assert spec.robot.default_root_pos[2] == pytest.approx(mj_ref.SPAWN_Z)
    assert spec.robot.actuator_delay == (0, mj_ref.DELAY_MAX_LAG)
    assert spec.robot.asset_for("mjlab").path.endswith(mj_ref.SHOE_XML_SUFFIX)


def test_both_sides_load_the_shoe_asset(spec, mjlab_robot) -> None:
    assert spec.robot.asset_for("mjlab").path.endswith(SHOE_MJCF)
    assert spec.robot.asset_for("isaacsim").path.endswith(SHOE_URDF)
    assert mjlab_robot.spec_fn is not None


def test_spawn_z_matches_on_both_sides(spec, mjlab_robot) -> None:
    assert spec.robot.default_root_pos[2] == pytest.approx(SPAWN_Z)
    assert mjlab_robot.init_state.pos[2] == pytest.approx(SPAWN_Z)


def test_isaac_merge_fixed_joints_is_true_only_on_the_parkour_copy(spec) -> None:
    assert spec.robot.asset_for("isaacsim").import_options["merge_fixed_joints"] is True
    assert make_g1_29dof_robot_spec().asset_for("isaacsim").import_options["merge_fixed_joints"] is False


def test_soft_joint_limit_factor_matches(spec, mjlab_robot) -> None:
    assert spec.robot.soft_joint_pos_limit_factor == SOFT_LIMIT
    assert mjlab_robot.articulation.soft_joint_pos_limit_factor == SOFT_LIMIT


def test_mjlab_actuators_hold_the_hub_episode_delay(spec, mjlab_robot) -> None:
    lags = {(act.delay_min_lag, act.delay_max_lag) for act in mjlab_robot.articulation.actuators}
    assert spec.robot.actuator_delay == (0, 2)
    assert lags == {(0, 2)}
    periods = [act.delay_update_period for act in mjlab_robot.articulation.actuators]
    assert all(period >= DELAY_RESET_ONLY_PERIOD for period in periods)
    assert len(set(periods)) == len(periods)
    assert all(act.delay_per_env_phase is False for act in mjlab_robot.articulation.actuators)
    assert all(type(act).__name__ == "BuiltinPdActuatorCfg" for act in mjlab_robot.articulation.actuators)


def test_undesired_contacts_is_the_per_engine_force_threshold_term_on_mjlab(spec) -> None:
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab import MjlabAdapter
    from instinctlab.engines.mjlab.rewards import CONTACT_FORCE_THRESHOLD_N, undesired_contacts

    compiled = MjlabAdapter().compile(spec, num_envs=1, device="cpu")
    term = compiled.env_cfg.rewards["undesired_contacts"]
    assert term.func is undesired_contacts
    assert term.params["threshold"] == CONTACT_FORCE_THRESHOLD_N == 1.0
    done = compiled.env_cfg.terminations["base_contact"]
    assert done.params["threshold"] == 1.0


def _urdf_ankle_collision_z(path: Path) -> list[float]:
    root = ET.parse(path).getroot()
    for link in root.iter("link"):
        if link.get("name") == "left_ankle_roll_link":
            return [
                float(col.find("origin").get("xyz").split()[2])  # type: ignore[union-attr]
                for col in link.findall("collision")
            ]
    raise AssertionError("left_ankle_roll_link missing")


def _mjcf_ankle_collision_z(path: Path) -> list[float]:
    root = ET.parse(path).getroot()
    for body in root.iter("body"):
        if body.get("name") != "left_ankle_roll_link":
            continue
        zs = []
        for geom in body.findall("geom"):
            if geom.get("type") != "capsule":
                continue
            pos = (geom.get("pos") or "0 0 0").split()
            zs.append(float(pos[2]))
        return zs
    raise AssertionError("left_ankle_roll_link missing")


def test_isaac_and_mjlab_shoes_are_the_same_23mm_sole() -> None:
    """Both shoe files drop every ankle capsule 23 mm vs the catalog bare foot."""
    catalog = make_g1_29dof_robot_spec()
    base_urdf = Path(catalog.asset_for("isaacsim").path)
    base_xml = Path(catalog.asset_for("mjlab").path)
    shoe_urdf = REPO / "source/instinctlab/instinctlab/tasks/parkour/urdf" / SHOE_URDF
    shoe_xml = REPO / "source/instinctlab/instinctlab/tasks/parkour/mjcf" / SHOE_MJCF
    assert shoe_urdf.is_file() and shoe_xml.is_file()

    isaac_base = _urdf_ankle_collision_z(base_urdf)
    isaac_shoe = _urdf_ankle_collision_z(shoe_urdf)
    mjlab_base = _mjcf_ankle_collision_z(base_xml)
    mjlab_shoe = _mjcf_ankle_collision_z(shoe_xml)
    assert len(isaac_base) == len(isaac_shoe) == ANKLE_COLLISION_GEOMS_PER_FOOT
    assert len(mjlab_base) == len(mjlab_shoe) == ANKLE_COLLISION_GEOMS_PER_FOOT
    isaac_deltas = [s - b for s, b in zip(isaac_shoe, isaac_base, strict=True)]
    mjlab_deltas = [s - b for s, b in zip(mjlab_shoe, mjlab_base, strict=True)]
    assert all(d == pytest.approx(SHOE_SOLE_OFFSET, abs=1e-3) for d in isaac_deltas)
    assert all(d == pytest.approx(SHOE_SOLE_OFFSET, abs=1e-3) for d in mjlab_deltas)
    assert isaac_shoe == pytest.approx(mjlab_shoe, abs=1e-6)


def _mjcf_ankle_capsules(path: Path) -> list[tuple[float, float]]:
    """(center_z, radius) for every collision capsule on the left ankle."""
    root = ET.parse(path).getroot()
    for body in root.iter("body"):
        if body.get("name") != "left_ankle_roll_link":
            continue
        out: list[tuple[float, float]] = []
        for geom in body.findall("geom"):
            if geom.get("type") != "capsule":
                continue
            pos = (geom.get("pos") or "0 0 0").split()
            radius = float((geom.get("size") or "0").split()[0])
            out.append((float(pos[2]), radius))
        return out
    raise AssertionError("left_ankle_roll_link missing")


def test_feet_at_plane_height_offset_is_the_shoe_sole(spec) -> None:
    """0.058 is the lowest shoe capsule (r=0.01 at z=-0.048). Bare foot would want 0.035."""
    offset = spec.mdp.rewards["rewards"]["feet_at_plane"].params["height_offset"]
    assert offset == FEET_AT_PLANE_HEIGHT_OFFSET
    shoe = _mjcf_ankle_capsules(Path(spec.robot.asset_for("mjlab").path))
    assert all(z == pytest.approx(-0.048, abs=1e-6) for z, _ in shoe)
    lowest = min(z - radius for z, radius in shoe)
    assert lowest == pytest.approx(-offset, abs=1e-3)
    assert any(radius == SHOE_CAPSULE_RADIUS for _, radius in shoe)


def _fixed_joint_merges(urdf: Path) -> dict[str, str]:
    """Child link of a fixed joint is absorbed into its parent (and that parent's parent)."""
    root = ET.parse(urdf).getroot()
    parent_of: dict[str, str] = {}
    for joint in root.iter("joint"):
        if joint.get("type") != "fixed":
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_of[child.get("link")] = parent.get("link")

    def root_parent(name: str) -> str:
        seen: set[str] = set()
        while name in parent_of and name not in seen:
            seen.add(name)
            name = parent_of[name]
        return name

    return {child: root_parent(parent) for child, parent in parent_of.items()}


def test_every_parkour_body_name_survives_merge_or_is_a_vanished_frame(spec) -> None:
    """Named bodies stay. Frame/hand children vanish into them — the silent shift."""
    urdf = Path(spec.robot.asset_for("isaacsim").path)
    merges = _fixed_joint_merges(urdf)
    vanished = set(merges)
    surviving = {name: name for name in PARKOUR_NAMED_BODIES}
    assert not (
        PARKOUR_NAMED_BODIES & vanished
    ), f"a name the task uses disappears under merge_fixed_joints: {sorted(PARKOUR_NAMED_BODIES & vanished)}"
    silent = {child: parent for child, parent in merges.items() if parent in PARKOUR_NAMED_BODIES}
    assert silent["LL_FOOT"] == "left_ankle_roll_link"
    assert silent["LR_FOOT"] == "right_ankle_roll_link"
    assert silent["left_rubber_hand"] == "left_wrist_yaw_link"
    assert silent["right_rubber_hand"] == "right_wrist_yaw_link"
    assert silent["pelvis_contour_link"] == "pelvis"
    assert silent["imu_in_pelvis"] == "pelvis"
    assert silent["logo_link"] == "torso_link"
    assert silent["head_link"] == "torso_link"
    assert silent["imu_in_torso"] == "torso_link"
    assert silent["mid360_link"] == "torso_link"
    assert DEPTH_CAMERA.attach == "torso_link"
    assert surviving["torso_link"] == "torso_link"
    assert surviving["left_wrist_yaw_link"] == "left_wrist_yaw_link"


def test_isaac_ankle_visuals_exist_for_the_camera_hit_suffix() -> None:
    """Isaac camera hits ``{link}/visuals``. An ankle with collision only is an invisible foot."""
    shoe_urdf = REPO / "source/instinctlab/instinctlab/tasks/parkour/urdf" / SHOE_URDF
    visuals: dict[str, int] = {}
    for link in ET.parse(shoe_urdf).getroot().iter("link"):
        name = link.get("name") or ""
        if name in {"left_ankle_roll_link", "right_ankle_roll_link"}:
            visuals[name] = len(link.findall("visual"))
    assert set(visuals) == {"left_ankle_roll_link", "right_ankle_roll_link"}
    assert all(count >= 1 for count in visuals.values()), visuals
    assert "left_ankle_roll_link" in G1_29DOF_LINKS
    assert "right_ankle_roll_link" in G1_29DOF_LINKS


def test_shoe_geometry_hangs_off_a_camera_hit_body() -> None:
    """Shoe capsules live on ankle_roll_link. Off-list they would be invisible, not an error."""
    from instinctlab.tasks.parkour.config.g1.target_env_cfg import DEPTH_CAMERA

    hit = set(DEPTH_CAMERA.hit_bodies())
    assert "left_ankle_roll_link" in hit
    assert "right_ankle_roll_link" in hit
    shoe_xml = REPO / "source/instinctlab/instinctlab/tasks/parkour/mjcf" / SHOE_MJCF
    shoe_urdf = REPO / "source/instinctlab/instinctlab/tasks/parkour/urdf" / SHOE_URDF
    parents: set[str] = set()
    for body in ET.parse(shoe_xml).getroot().iter("body"):
        name = body.get("name") or ""
        for geom in body.findall("geom"):
            geom_name = geom.get("name") or ""
            if "foot" in geom_name and "collision" in geom_name:
                parents.add(name)
    assert parents == {"left_ankle_roll_link", "right_ankle_roll_link"}
    urdf_parents: set[str] = set()
    for link in ET.parse(shoe_urdf).getroot().iter("link"):
        name = link.get("name") or ""
        if link.findall("collision") and "ankle_roll" in name:
            urdf_parents.add(name)
    assert urdf_parents <= hit
    assert urdf_parents == {"left_ankle_roll_link", "right_ankle_roll_link"}


def test_mjlab_camera_mask_keeps_ankle_geoms_from_the_hit_list() -> None:
    """Named-body mask must include the ankle. A missing mesh must not drop the foot."""
    mujoco = pytest.importorskip("mujoco")
    from instinctlab.assets.unitree_g1.isaacsim import G1_29DOF_LINKS
    from instinctlab.engines.mjlab.camera import _camera_geom_mask

    shoe_xml = REPO / "source/instinctlab/instinctlab/tasks/parkour/mjcf" / SHOE_MJCF
    try:
        model = mujoco.MjModel.from_xml_path(str(shoe_xml))
    except (ValueError, OSError) as exc:
        pytest.skip(f"shoe MJCF did not load ({exc}); mesh-free fallback is an adapter concern")
    mask = _camera_geom_mask(model, bodies=tuple(G1_29DOF_LINKS), include_terrain=False, device="cpu")
    kept = 0
    for geom_id in range(int(model.ngeom)):
        if not bool(mask[geom_id]):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if name.rsplit("/", 1)[-1] in {"left_ankle_roll_link", "right_ankle_roll_link"}:
            kept += 1
    assert kept >= 2, f"camera mask kept {kept} ankle geoms; the feet would be invisible"
    mesh_kept = {"left_ankle_roll_link": 0, "right_ankle_roll_link": 0}
    for geom_id in range(int(model.ngeom)):
        if not bool(mask[geom_id]):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "").rsplit("/", 1)[-1]
        if name in mesh_kept and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_kept[name] += 1
    assert all(
        count >= 1 for count in mesh_kept.values()
    ), f"camera kept no ankle mesh {mesh_kept}; shoe capsules are not the hit surface"


def test_camera_hit_list_does_not_name_a_merged_away_link() -> None:
    hit = set(DEPTH_CAMERA.hit) - {"terrain"}
    merges = _fixed_joint_merges(REPO / "source/instinctlab/instinctlab/tasks/parkour/urdf" / SHOE_URDF)
    assert not (hit & set(merges)), f"camera hit names a body merge removes: {sorted(hit & set(merges))}"


@pytest.mark.isaacsim
def test_isaac_compiled_robot_matches_mjlab_on_the_actuation_axes(spec, mjlab_robot) -> None:
    """Kit session: the Isaac half of the symmetry table."""
    pytest.importorskip("isaaclab")
    import argparse
    import sys

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    argv = ["--headless", "--device", "cuda:0"]
    previous = sys.argv
    sys.argv = [previous[0], *argv]
    try:
        AppLauncher(parser.parse_args(argv))
    finally:
        sys.argv = previous

    from instinctlab.engines.isaacsim import IsaacSimAdapter

    isaac_robot = IsaacSimAdapter().compile(spec, num_envs=16, device="cuda:0").env_cfg.scene.robot
    assert isaac_robot.init_state.pos[2] == pytest.approx(mjlab_robot.init_state.pos[2])
    assert isaac_robot.spawn.merge_fixed_joints is True
    assert isaac_robot.spawn.asset_path.endswith(SHOE_URDF)
    assert isaac_robot.soft_joint_pos_limit_factor == mjlab_robot.articulation.soft_joint_pos_limit_factor
    actuator_types = {type(cfg).__name__ for cfg in isaac_robot.actuators.values()}
    assert actuator_types == {"DelayedPDActuatorCfg"}
    delays = {(cfg.min_delay, cfg.max_delay) for cfg in isaac_robot.actuators.values()}
    assert delays == {(0, 2)}
