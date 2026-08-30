"""Who owns the four robot plant numbers: catalog vs parkour task override.

The catalog (``make_g1_29dof_robot_spec`` / ``G1_29DOF_TORSOBASE_POPSICLE_CFG``) is the
flat/rough plant: shoeless popsicle, z=0.82, merge_fixed_joints=False, implicit PD.
Main's *flat* locomotion already matched those. Main's *parkour* overrode three of
them at the task (shoe URDF, z=0.9, merge_fixed_joints=True). The fourth, delayed
actuators, it only *appears* to override -- see the test below.

If the catalog factory itself grew those four, flat and rough would silently train
a different robot. This file pins the factory *and* asserts only parkour holds the
override copy.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from instinctlab.tasks.locomotion.config.g1 import flat_g1
from instinctlab.tasks.locomotion.config.g1.rough_env_cfg import rough_g1
from instinctlab.tasks.parkour.config.g1 import parkour_target_g1
from tests import reference_main_parkour as main_parkour
from tests.g1_specs import paired_robot_spec

REPO = Path(__file__).resolve().parents[1]
BASE_ASSET_ID = "unitree_g1/popsicle_torsobase_v1"
PARKOUR_ASSET_ID = "unitree_g1/popsicle_torsobase_parkour_v1"
MAIN_FLAT = "source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_env_cfg.py"
MAIN_ASSETS = "source/instinctlab/instinctlab/assets/unitree_g1.py"

CATALOG = {
    "urdf_suffix": "g1_29dof_torsobase_popsicle.urdf",
    "mjcf_suffix": "g1_29dof_torsobase_popsicle.xml",
    "spawn_z": 0.82,
    "merge_fixed_joints": False,
    "actuators": "beyondmimic_g1_29dof_actuators",
    "actuator_delay": (0, 0),
    "soft_joint_pos_limit_factor": 0.9,
}

PARKOUR = {
    "urdf_suffix": "g1_29dof_torsoBase_popsicle_with_shoe.urdf",
    "mjcf_suffix": "g1_29dof_torsoBase_popsicle_with_shoe.xml",
    "spawn_z": 0.9,
    "merge_fixed_joints": True,
    "actuator_delay": (0, 2),
}


def _git_show(path: str) -> str:
    shown = subprocess.run(("git", "show", f"main:{path}"), cwd=REPO, capture_output=True, text=True)
    assert shown.returncode == 0, shown.stderr
    return shown.stdout


def _assert_catalog_plant(robot) -> None:
    assert robot.default_root_pos[2] == CATALOG["spawn_z"]
    assert robot.actuator_delay == CATALOG["actuator_delay"]
    assert robot.soft_joint_pos_limit_factor == CATALOG["soft_joint_pos_limit_factor"]
    assert robot.asset_for("isaacsim").path.endswith(CATALOG["urdf_suffix"])
    assert robot.asset_for("mjlab").path.endswith(CATALOG["mjcf_suffix"])
    assert robot.asset_for("isaacsim").import_options["merge_fixed_joints"] is CATALOG["merge_fixed_joints"]
    assert "with_shoe" not in robot.asset_for("isaacsim").path
    assert "with_shoe" not in robot.asset_for("mjlab").path


@pytest.fixture(scope="module")
def catalog_robot():
    return paired_robot_spec(BASE_ASSET_ID)


@pytest.mark.parametrize("factory", [flat_g1, rough_g1], ids=["flat", "rough"])
def test_flat_and_rough_read_the_catalog_without_overrides(factory, catalog_robot) -> None:
    task_robot = factory(catalog_robot).robot
    assert task_robot.name == catalog_robot.name
    assert task_robot.default_root_pos == catalog_robot.default_root_pos
    assert task_robot.actuator_delay == catalog_robot.actuator_delay == CATALOG["actuator_delay"]
    assert task_robot.soft_joint_pos_limit_factor == catalog_robot.soft_joint_pos_limit_factor
    assert task_robot.asset_for("isaacsim").path.endswith(CATALOG["urdf_suffix"])
    assert task_robot.asset_for("mjlab").path.endswith(CATALOG["mjcf_suffix"])
    assert task_robot.asset_for("isaacsim").import_options["merge_fixed_joints"] is CATALOG["merge_fixed_joints"]


def test_the_catalog_factory_still_returns_the_popsicle(catalog_robot) -> None:
    """If this fails, the override leaked into the factory and flat/rough moved with it."""
    _assert_catalog_plant(catalog_robot)
    assert catalog_robot.default_root_pos == (0.0, 0.0, 0.82)


def test_parkour_holds_a_copy_with_the_four_main_overrides(catalog_robot) -> None:
    parkour = parkour_target_g1(paired_robot_spec(PARKOUR_ASSET_ID)).robot
    _assert_catalog_plant(catalog_robot)
    assert parkour.default_root_pos[2] == PARKOUR["spawn_z"]
    assert parkour.actuator_delay == PARKOUR["actuator_delay"]
    assert parkour.asset_for("isaacsim").path.endswith(PARKOUR["urdf_suffix"])
    assert parkour.asset_for("mjlab").path.endswith(PARKOUR["mjcf_suffix"])
    assert parkour.asset_for("isaacsim").import_options["merge_fixed_joints"] is PARKOUR["merge_fixed_joints"]
    assert parkour.name == catalog_robot.name
    assert parkour.joint_names == catalog_robot.joint_names
    assert parkour.soft_joint_pos_limit_factor == catalog_robot.soft_joint_pos_limit_factor


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
    """The overrides vs main parkour are G1ParkourEnvCfg overrides, not catalog defaults."""
    overrides = main_parkour.g1_robot_overrides()
    assert overrides["shoe_urdf"].endswith("with_shoe.urdf")
    assert overrides["spawn_z"] == 0.9
    assert overrides["merge_fixed_joints"] is True


def test_main_parkour_declares_delayed_actuators_but_does_not_run_them() -> None:
    """The delayed table is assigned and then thrown away, so main trains without delay.

    ``G1ParkourRoughEnvCfg.__post_init__`` sets ``self.scene.robot.actuators`` to the
    delayed table; the registered ``G1ParkourEnvCfg`` then calls ``apply_shoe_config()``,
    which replaces ``self.scene.robot`` with a module-level copy taken before that
    assignment. Confirmed against the ``params/env.yaml`` a real main run wrote: five
    ``ImplicitActuator`` groups, no delay fields.

    An earlier reader asked whether the delayed table was *named anywhere* in the file.
    It is -- in the discarded line -- so the answer was True and we copied a delay main
    does not have into our Isaac task.
    """
    effective = main_parkour.effective_robot_actuators()
    assert effective["declared_in_base"] is True, "the dead assignment vanished; this test lost its subject"
    assert effective["robot_symbol"] == "G1_with_shoe_CFG"
    assert effective["table"] == "beyondmimic_g1_29dof_actuators"
    assert effective["delayed"] is False
    assert main_parkour.g1_robot_overrides()["uses_delayed_actuators"] is False


_SHOE_FIRST = """
task_entry = "x"
gym.register(
    id="Instinct-Parkour-Target-Amp-G1-v0",
    kwargs={"env_cfg_entry_point": f"{task_entry}.m:G1ParkourEnvCfg"},
)
G1_CFG = copy.deepcopy(G1_29DOF_TORSOBASE_POPSICLE_CFG)
G1_with_shoe_CFG = copy.deepcopy(G1_CFG)


class G1ParkourRoughEnvCfg(ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_CFG.replace(prim_path="p")


class ShoeConfigMixin:
    def apply_shoe_config(self):
        self.scene.robot = G1_with_shoe_CFG.replace(prim_path="p")


class G1ParkourEnvCfg(G1ParkourRoughEnvCfg, ShoeConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        self.apply_shoe_config()
        self.scene.robot.actuators = beyondmimic_g1_29dof_delayed_actuators
"""


def test_the_reader_reports_delay_when_the_assignment_survives(monkeypatch) -> None:
    """The resolver must answer on override *order*, not always say "no delay".

    Same file as main except the delayed assignment comes after ``apply_shoe_config()``.
    If this still reported no delay, the reader would be a constant and the drift row
    resting on it would mean nothing.
    """
    real = main_parkour._git_show

    def fake(path: str) -> str:
        if path in (main_parkour.G1_CFG, main_parkour.G1_INIT):
            return _SHOE_FIRST
        return real(path)

    monkeypatch.setattr(main_parkour, "_git_show", fake)
    effective = main_parkour.effective_robot_actuators()
    assert effective["delayed"] is True
    assert effective["table"] == "beyondmimic_g1_29dof_delayed_actuators"


def test_the_reader_refuses_a_file_it_cannot_resolve(monkeypatch) -> None:
    """No assignment to self.scene.robot at all must raise, not report a default."""
    stripped = (
        _SHOE_FIRST.replace('self.scene.robot = G1_CFG.replace(prim_path="p")', "pass")
        .replace('self.scene.robot = G1_with_shoe_CFG.replace(prim_path="p")', "pass")
        .replace("self.scene.robot.actuators = beyondmimic_g1_29dof_delayed_actuators", "pass")
    )
    real = main_parkour._git_show
    monkeypatch.setattr(
        main_parkour,
        "_git_show",
        lambda path: stripped if path in (main_parkour.G1_CFG, main_parkour.G1_INIT) else real(path),
    )
    with pytest.raises(LookupError, match="never assigns"):
        main_parkour.effective_robot_actuators()


def test_overridden_does_not_mutate_the_catalog(catalog_robot) -> None:
    copy = catalog_robot.overridden(default_root_pos=(0.0, 0.0, 0.9), actuator_delay=(0, 2))
    assert catalog_robot.default_root_pos[2] == 0.82
    assert catalog_robot.actuator_delay == (0, 0)
    assert copy.default_root_pos[2] == 0.9
    assert copy.actuator_delay == (0, 2)


def test_overridden_rejects_a_misspelled_engine_key(catalog_robot) -> None:
    with pytest.raises(ValueError, match="isaac"):
        catalog_robot.overridden(asset_paths={"isaac": "nope.urdf"})
