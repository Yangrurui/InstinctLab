"""Parkour G1 robot/actuation symmetry between mjlab and Isaac compile paths.

Mjlab compilation is cheap (no GPU). Isaac fields that need Kit are marked ``isaacsim`` and
were measured once: init_pos=(0,0,0.82), urdf=popsicle, merge_fixed=False, ImplicitActuator only,
soft_limit=0.9 — matching mjlab on every axis below.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from instinctlab.assets.unitree_g1.isaacsim import make_g1_29dof_robot_spec
from instinctlab.tasks.parkour.config.g1 import parkour_target_g1

REPO = Path(__file__).resolve().parents[1]

POPSICLE_URDF = "g1_29dof_torsobase_popsicle.urdf"
POPSICLE_MJCF = "g1_29dof_torsobase_popsicle.xml"
SPAWN_Z = 0.82
SOFT_LIMIT = 0.9
ANKLE_COLLISION_GEOMS_PER_FOOT = 7  # left_ankle_roll_link collision primitives, base popsicle


@pytest.fixture(scope="module")
def spec():
    return parkour_target_g1()


@pytest.fixture(scope="module")
def mjlab_robot(spec):
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab import MjlabAdapter

    compiled = MjlabAdapter().compile(spec, num_envs=16, device="cpu")
    return compiled.env_cfg.scene.entities["robot"]


def test_mjlab_loads_the_catalog_popsicle_not_the_shoe_asset(spec, mjlab_robot) -> None:
    assert spec.robot.asset_for("mjlab").path.endswith(POPSICLE_MJCF)
    assert spec.robot.asset_for("isaacsim").path.endswith(POPSICLE_URDF)
    assert "with_shoe" not in spec.robot.asset_for("mjlab").path
    assert "with_shoe" not in spec.robot.asset_for("isaacsim").path


def test_spawn_z_matches_on_both_sides(spec, mjlab_robot) -> None:
    assert spec.robot.default_root_pos[2] == pytest.approx(SPAWN_Z)
    assert mjlab_robot.init_state.pos[2] == pytest.approx(SPAWN_Z)


def test_merge_fixed_joints_is_false_in_the_catalog(spec) -> None:
    assert spec.robot.asset_for("isaacsim").import_options["merge_fixed_joints"] is False


def test_soft_joint_limit_factor_matches(spec, mjlab_robot) -> None:
    assert spec.robot.soft_joint_pos_limit_factor == SOFT_LIMIT
    assert mjlab_robot.articulation.soft_joint_pos_limit_factor == SOFT_LIMIT


def test_mjlab_actuators_have_zero_torque_delay(mjlab_robot) -> None:
    lags = {(act.delay_min_lag, act.delay_max_lag) for act in mjlab_robot.articulation.actuators}
    assert lags == {(0, 0)}
    assert all(type(act).__name__ == "BuiltinPdActuatorCfg" for act in mjlab_robot.articulation.actuators)


def test_undesired_contacts_uses_the_same_portable_term_on_mjlab(spec, mjlab_robot) -> None:
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab import MjlabAdapter

    compiled = MjlabAdapter().compile(spec, num_envs=1, device="cpu")
    term = compiled.env_cfg.rewards["undesired_contacts"]
    assert term.func.__name__ == "undesired_contacts"
    assert "threshold" not in term.params


def test_base_popsicle_and_shoe_urdf_differ_by_23mm_sole_offset() -> None:
    """Quantifies why main-parkour shoe mattered; both cross-engine assets skip it."""
    import xml.etree.ElementTree as ET
    from pathlib import Path

    base = Path(make_g1_29dof_robot_spec().asset_for("isaacsim").path)
    shoe = REPO / "source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"
    assert shoe.is_file()

    def ankle_z(path: Path) -> list[float]:
        root = ET.parse(path).getroot()
        for link in root.iter("link"):
            if link.get("name") == "left_ankle_roll_link":
                return [
                    float(col.find("origin").get("xyz").split()[2])  # type: ignore[union-attr]
                    for col in link.findall("collision")
                ]
        raise AssertionError("left_ankle_roll_link missing")

    base_z = ankle_z(base)
    shoe_z = ankle_z(shoe)
    assert len(base_z) == ANKLE_COLLISION_GEOMS_PER_FOOT
    assert len(shoe_z) == ANKLE_COLLISION_GEOMS_PER_FOOT
    deltas = [s - b for s, b in zip(shoe_z, base_z, strict=True)]
    assert all(d == pytest.approx(-0.023, abs=1e-3) for d in deltas)


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
    assert isaac_robot.spawn.merge_fixed_joints is False
    assert isaac_robot.spawn.asset_path.endswith(POPSICLE_URDF)
    assert isaac_robot.soft_joint_pos_limit_factor == mjlab_robot.articulation.soft_joint_pos_limit_factor
    actuator_types = {type(cfg).__name__ for cfg in isaac_robot.actuators.values()}
    assert actuator_types == {"ImplicitActuatorCfg"}
    assert "DelayedPDActuatorCfg" not in actuator_types
