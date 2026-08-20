"""Who owns the four robot drifts flagged against main parkour.

Drifts #1–4 (shoe URDF, spawn_z, merge_fixed_joints, delayed actuators) are properties of
``make_g1_29dof_robot_spec()`` / ``G1_29DOF_TORSOBASE_POPSICLE_CFG``, not of the parkour
declaration. Commit ``601e767`` consolidated the joint table, default pose and PD into that
catalog; the cross-engine stack has always been shoeless popsicle at z=0.82 with implicit PD.

Main's *flat* locomotion used the same four numbers. Main's *parkour* added task-level overrides
(shoe URDF, z=0.9, merge_fixed_joints=True, delayed actuators) that the new parkour TaskSpec does
not restate because it reads the catalog like flat and rough do.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from instinctlab.assets.unitree_g1.isaacsim import make_g1_29dof_robot_spec
from instinctlab.tasks.locomotion.config.g1 import flat_g1
from instinctlab.tasks.locomotion.config.g1.rough_env_cfg import rough_g1
from instinctlab.tasks.parkour.config.g1 import parkour_target_g1
from tests import reference_main_parkour as main_parkour

REPO = Path(__file__).resolve().parents[1]
MAIN_FLAT = "source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py"
MAIN_ASSETS = "source/instinctlab/instinctlab/assets/unitree_g1.py"

CATALOG = {
    "urdf_suffix": "g1_29dof_torsobase_popsicle.urdf",
    "mjcf_suffix": "g1_29dof_torsobase_popsicle.xml",
    "spawn_z": 0.82,
    "merge_fixed_joints": False,
    "actuators": "beyondmimic_g1_29dof_actuators",
    "soft_joint_pos_limit_factor": 0.9,
}


def _git_show(path: str) -> str:
    shown = subprocess.run(("git", "show", f"main:{path}"), cwd=REPO, capture_output=True, text=True)
    assert shown.returncode == 0, shown.stderr
    return shown.stdout


@pytest.fixture(scope="module")
def catalog_robot():
    return make_g1_29dof_robot_spec()


@pytest.mark.parametrize(
    "factory",
    [parkour_target_g1, flat_g1, rough_g1],
    ids=["parkour", "flat", "rough"],
)
def test_every_cross_engine_g1_task_reads_the_same_catalog(factory, catalog_robot) -> None:
    task_robot = factory().robot
    assert task_robot.name == catalog_robot.name
    assert task_robot.default_root_pos == catalog_robot.default_root_pos
    assert task_robot.soft_joint_pos_limit_factor == catalog_robot.soft_joint_pos_limit_factor
    assert task_robot.asset_for("isaacsim").path.endswith(CATALOG["urdf_suffix"])
    assert task_robot.asset_for("mjlab").path.endswith(CATALOG["mjcf_suffix"])
    assert task_robot.asset_for("isaacsim").import_options["merge_fixed_joints"] is CATALOG["merge_fixed_joints"]


def test_main_flat_popsicle_already_matched_the_catalog() -> None:
    """Main flat never had shoes, delay, z=0.9 or merge_fixed=True."""
    flat = _git_show(MAIN_FLAT)
    assets = _git_show(MAIN_ASSETS)
    assert "with_shoe" not in flat
    assert "delayed" not in flat.lower()
    assert "POPSICLE" in flat
    assert "pos=(0.0, 0.0, 0.82)" in assets
    assert "merge_fixed_joints=False" in assets
    assert "actuators=beyondmimic_g1_29dof_actuators" in assets
    assert "with_shoe" not in assets


def test_main_parkour_overrides_were_task_level_not_catalog() -> None:
    """The four drifts vs main parkour are G1ParkourEnvCfg overrides, not catalog defaults."""
    overrides = main_parkour.g1_robot_overrides()
    assert overrides["shoe_urdf"].endswith("with_shoe.urdf")
    assert overrides["spawn_z"] == 0.9
    assert overrides["merge_fixed_joints"] is True
    assert overrides["uses_delayed_actuators"] is True
    assert overrides["actuators"] == "beyondmimic_g1_29dof_delayed_actuators"


def test_the_catalog_was_shoeless_before_parkour_adaptation(catalog_robot) -> None:
    assert not catalog_robot.asset_for("isaacsim").path.endswith("with_shoe.urdf")
    assert catalog_robot.default_root_pos[2] == 0.82
