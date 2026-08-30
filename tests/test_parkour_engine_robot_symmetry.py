"""Parkour G1 robot/actuation symmetry between mjlab and Isaac compile paths.

Both engines must train the main-parkour plant: shoe collision 23 mm below the
bare ankle, spawn z=0.9, Isaac merge_fixed_joints=True, torque delay 0–2 physics
steps resampled once per episode. The catalog popsicle is the flat/rough plant.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from instinctlab.assets.unitree_g1.isaacsim import G1_29DOF_LINKS
from instinctlab_engine_isaacsim.assets import robot_spec as isaac_robot_spec
from instinctlab_engine_mjlab.assets import robot_spec as mjlab_robot_spec
from instinctlab.tasks.parkour.config.g1 import parkour_target_g1
from tests.g1_specs import paired_robot_spec

REPO = Path(__file__).resolve().parents[1]
BASE_ASSET_ID = "unitree_g1/popsicle_torsobase_v1"
PARKOUR_ASSET_ID = "unitree_g1/popsicle_torsobase_parkour_v1"

SHOE_URDF = "g1_29dof_torsoBase_popsicle_with_shoe.urdf"
SHOE_MJCF = "g1_29dof_torsoBase_popsicle_with_shoe.xml"
G1_RESOURCE_ROOT = REPO / "source/instinctlab/instinctlab/assets/resources/unitree_g1"
SHOE_URDF_PATH = G1_RESOURCE_ROOT / "urdf" / SHOE_URDF
SHOE_MJCF_PATH = G1_RESOURCE_ROOT / "xml" / SHOE_MJCF
SPAWN_Z = 0.9
SOFT_LIMIT = 0.9
ANKLE_COLLISION_GEOMS_PER_FOOT = 7
SHOE_SOLE_OFFSET = -0.023
FEET_AT_PLANE_HEIGHT_OFFSET = 0.058
SHOE_CAPSULE_RADIUS = 0.01
RESET_ONLY_DELAY_PERIOD = 1_000_000


# Parkour names a body. After Isaac merge_fixed_joints=True the child of each
# fixed joint disappears and its collision rides on the parent. A vanished name
# fails loudly. A name that still exists but now denotes the merged body is the
# silent case — list every referenced name and what it resolves to.
def _attach_names(attach: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(attach, str):
        return (attach,)
    return tuple(attach)


def _parkour_named_bodies(spec) -> frozenset[str]:
    return frozenset(
        {
            "torso_link",
            "pelvis",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            *G1_29DOF_LINKS,
            *spec.scene.motion_reference("motion_reference").links,
            *_attach_names(spec.scene.ray_caster("left_height_scanner").attach),
            *_attach_names(spec.scene.ray_caster("right_height_scanner").attach),
            *_attach_names(spec.scene.volume_point("leg_volume_points").attach),
        }
    )


@pytest.fixture(scope="module")
def spec():
    return parkour_target_g1(paired_robot_spec(PARKOUR_ASSET_ID))


@pytest.fixture(scope="module")
def mjlab_spec():
    return parkour_target_g1(mjlab_robot_spec(PARKOUR_ASSET_ID))


@pytest.fixture(scope="module")
def mjlab_robot(mjlab_spec):
    pytest.importorskip("mjlab")
    from instinctlab_engine_mjlab import MjlabAdapter

    compiled = MjlabAdapter().compile(mjlab_spec, num_envs=16, device="cpu")
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
    base = paired_robot_spec(BASE_ASSET_ID)
    assert base.asset_for("isaacsim").import_options["merge_fixed_joints"] is False


def test_soft_joint_limit_factor_matches(spec, mjlab_robot) -> None:
    assert spec.robot.soft_joint_pos_limit_factor == SOFT_LIMIT
    assert mjlab_robot.articulation.soft_joint_pos_limit_factor == SOFT_LIMIT


def test_mjlab_actuators_hold_the_hub_episode_delay(spec, mjlab_robot) -> None:
    actuators = mjlab_robot.articulation.actuators
    pd_actuators = tuple(act for act in actuators if type(act).__name__ == "BuiltinPdActuatorCfg")
    lags = {(act.delay_min_lag, act.delay_max_lag) for act in pd_actuators}
    assert spec.robot.actuator_delay == (0, 2)
    assert lags == {(0, 2)}
    periods = [act.delay_update_period for act in pd_actuators]
    assert all(period >= RESET_ONLY_DELAY_PERIOD for period in periods)
    # Distinct periods, once. It read as "keep mjlab from fusing groups", but fusion is the
    # only way two configs can share a lag draw, and the G1's legs need exactly that -- they
    # are one motor bus over two gain sets. What the period has to track is the bus, so the
    # count of distinct periods is the count of buses, not the count of configs.
    assert len(set(periods)) == 5
    assert len(periods) == 7
    assert all(act.delay_per_env_phase is False for act in pd_actuators)
    assert len(pd_actuators) == 7


def test_mjlab_uses_only_the_reference_pd_actuators(mjlab_robot) -> None:
    actuators = tuple(mjlab_robot.articulation.actuators)
    assert len(actuators) == 7
    assert all(type(act).__name__ == "BuiltinPdActuatorCfg" for act in actuators)


def test_undesired_contacts_uses_the_task_owned_force_threshold_term_on_mjlab(mjlab_spec) -> None:
    pytest.importorskip("mjlab")
    from instinctlab_engine_mjlab import MjlabAdapter
    from instinctlab.tasks.parkour.mdp.rewards import undesired_contacts_by_force

    compiled = MjlabAdapter().compile(mjlab_spec, num_envs=1, device="cpu")
    term = compiled.env_cfg.rewards["undesired_contacts"]
    assert term.func is undesired_contacts_by_force
    assert term.params["threshold"] == 1.0
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
    catalog = paired_robot_spec(BASE_ASSET_ID)
    base_urdf = Path(catalog.asset_for("isaacsim").path)
    base_xml = Path(catalog.asset_for("mjlab").path)
    shoe_urdf = SHOE_URDF_PATH
    shoe_xml = SHOE_MJCF_PATH
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
    named_bodies = _parkour_named_bodies(spec)
    surviving = {name: name for name in named_bodies}
    assert not (
        named_bodies & vanished
    ), f"a name the task uses disappears under merge_fixed_joints: {sorted(named_bodies & vanished)}"
    silent = {child: parent for child, parent in merges.items() if parent in named_bodies}
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
    assert spec.scene.ray_caster("camera").attach == "torso_link"
    assert surviving["torso_link"] == "torso_link"
    assert surviving["left_wrist_yaw_link"] == "left_wrist_yaw_link"


def test_isaac_ankle_visuals_exist_for_the_camera_hit_suffix() -> None:
    """Isaac camera hits ``{link}/visuals``. An ankle with collision only is an invisible foot."""
    shoe_urdf = SHOE_URDF_PATH
    visuals: dict[str, int] = {}
    for link in ET.parse(shoe_urdf).getroot().iter("link"):
        name = link.get("name") or ""
        if name in {"left_ankle_roll_link", "right_ankle_roll_link"}:
            visuals[name] = len(link.findall("visual"))
    assert set(visuals) == {"left_ankle_roll_link", "right_ankle_roll_link"}
    assert all(count >= 1 for count in visuals.values()), visuals
    assert "left_ankle_roll_link" in G1_29DOF_LINKS
    assert "right_ankle_roll_link" in G1_29DOF_LINKS


def test_shoe_geometry_hangs_off_a_camera_hit_body(spec) -> None:
    """Shoe capsules live on ankle_roll_link. Off-list they would be invisible, not an error."""
    hit = set(spec.scene.ray_caster("camera").hit_bodies())
    assert "left_ankle_roll_link" in hit
    assert "right_ankle_roll_link" in hit
    shoe_xml = SHOE_MJCF_PATH
    shoe_urdf = SHOE_URDF_PATH
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


def test_mjlab_camera_uses_geom_groups_not_body_mask() -> None:
    """Production mjlab camera hits geom groups (0,1,2), not the Isaac G1 link list."""
    mujoco = pytest.importorskip("mujoco")
    from instinctlab_engine_mjlab.camera import geom_groups_camera_mask, pinhole_camera_geom_groups

    shoe_xml = SHOE_MJCF_PATH
    try:
        model = mujoco.MjModel.from_xml_path(str(shoe_xml))
    except (ValueError, OSError) as exc:
        pytest.skip(f"shoe MJCF did not load ({exc}); mesh-free fallback is an adapter concern")
    groups = pinhole_camera_geom_groups()
    mask = geom_groups_camera_mask(model, groups, device="cpu")
    group012 = sum(1 for geom_id in range(int(model.ngeom)) if int(model.geom_group[geom_id]) in groups)
    assert int(mask.sum()) == group012
    body_mask_names = set(G1_29DOF_LINKS)
    kept_bodies: set[str] = set()
    for geom_id in range(int(model.ngeom)):
        if not bool(mask[geom_id]):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "").rsplit("/", 1)[-1]
        kept_bodies.add(name)
    assert kept_bodies - body_mask_names, "group mask must hit geoms outside the Isaac link list"


def test_camera_hit_list_does_not_name_a_merged_away_link(spec) -> None:
    hit = set(spec.scene.ray_caster("camera").hit) - {"terrain"}
    merges = _fixed_joint_merges(SHOE_URDF_PATH)
    assert not (hit & set(merges)), f"camera hit names a body merge removes: {sorted(hit & set(merges))}"


@pytest.mark.isaacsim
def test_isaac_compiled_robot_matches_mjlab_on_the_actuation_axes(spec) -> None:
    """Kit session: the Isaac half of the symmetry table.

    Against ``spec``, not against a live ``mjlab_robot``, and the difference is the whole
    reason this test used to be impossible to pass. Isaac Sim and mjlab need different warp
    builds -- Isaac's extensions want the 1.8.2 it bundles in ``extscache/omni.warp.core``,
    mujoco_warp wants site-packages 1.16.0, and ``warp.types.array`` exists only in the
    former. Whichever lands in ``sys.modules`` first wins the process, which is why
    ``pytest.ini`` says the two cannot share one. Taking ``mjlab_robot`` here imported
    mjlab's warp during fixture setup, before the test body ever reached ``AppLauncher``,
    so every Isaac extension then failed to start and the test failed on an environment
    error that had nothing to do with what it was checking.

    Comparing to ``spec`` loses nothing: the two axes it borrowed from mjlab are the spawn
    height and the soft joint limit, both of which are ``spec`` values that
    ``test_spawn_z_matches_on_both_sides`` and ``test_soft_joint_limit_factor_matches``
    already pin on the mjlab side. The symmetry holds through the declaration both engines
    read, which is the only place it can hold when they cannot be alive together.
    """
    pytest.importorskip("isaaclab")
    from tests.isaacsim_app import ensure_isaac_app
    from tests.live_device import resolve_live_device

    device = resolve_live_device()
    ensure_isaac_app(device=device)

    from instinctlab_engine_isaacsim import IsaacSimAdapter

    isaac_spec = parkour_target_g1(isaac_robot_spec(PARKOUR_ASSET_ID))
    compiled_robot = IsaacSimAdapter().compile(
        isaac_spec, num_envs=16, device=device
    ).env_cfg.scene.robot
    assert compiled_robot.init_state.pos[2] == pytest.approx(isaac_spec.robot.default_root_pos[2]) == SPAWN_Z
    assert compiled_robot.spawn.merge_fixed_joints is True
    assert isaac_robot.spawn.asset_path.endswith(SHOE_URDF)
    assert isaac_robot.soft_joint_pos_limit_factor == spec.robot.soft_joint_pos_limit_factor == SOFT_LIMIT
    actuator_types = {type(cfg).__name__ for cfg in isaac_robot.actuators.values()}
    assert actuator_types == {"DelayedPDActuatorCfg"}
    delays = {(cfg.min_delay, cfg.max_delay) for cfg in isaac_robot.actuators.values()}
    assert delays == {(0, 2)} == {spec.robot.actuator_delay}
    # One actuator config per motor bus, so a whole leg draws one lag. Isaac has no
    # fusion mechanism -- two configs are two independent draws -- so unlike mjlab the
    # bus and the config have to be the same object here.
    assert set(isaac_robot.actuators) == {name for name, _ in spec.robot.actuator_groups()}


def test_no_engine_marked_test_asks_for_the_other_engines_fixture() -> None:
    """The structural version of "Isaac and MJLab cannot share a process".

    ``pytest.ini`` states the rule and gives the per-file invocation, but nothing enforced
    it inside a file, and a test that requests a fixture from the other engine breaks the
    rule during setup -- before its own body can launch anything. Such a test cannot pass
    under any invocation, and it fails on an unrelated environment error, so it reads as
    machine trouble rather than as a broken test. This one sat red long enough to get
    written into a handoff note as "hangs on this box".

    Checked statically, on the signature, because the dynamic version would need exactly
    the process this rule forbids.
    """
    import ast

    source = Path(__file__).read_text()
    engine_of = {"mjlab_robot": "mjlab"}
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        marks = {
            ast.unparse(d).rsplit(".", 1)[-1].split("(")[0] for d in node.decorator_list if "mark" in ast.unparse(d)
        }
        engines = marks & {"mjlab", "isaacsim"}
        wanted = {engine_of[a.arg] for a in node.args.args if a.arg in engine_of}
        if engines and wanted - engines:
            offenders.append((node.name, sorted(engines), sorted(wanted)))
    assert not offenders, offenders
