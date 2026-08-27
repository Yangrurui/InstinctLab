"""The rough tasks share one engine-neutral terrain declaration."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from instinctlab.tasks.locomotion.config.g1 import rough_g1
from instinctlab.tasks.terrain import rough_terrain

REPO = Path(__file__).resolve().parent.parent
DECLARATION = REPO / "source/instinctlab/instinctlab/tasks/locomotion/config/g1/rough_env_cfg.py"
RECIPE = REPO / "source/instinctlab/instinctlab/tasks/terrain.py"
ISAAC_ROUGH = REPO / "source/instinctlab/instinctlab/engines/isaacsim/rough.py"
MJLAB_ROUGH = REPO / "source/instinctlab/instinctlab/engines/mjlab/rough.py"
MJLAB_INIT = REPO / "source/instinctlab/instinctlab/engines/mjlab/__init__.py"

TERRAIN_NAMES = (
    "perlin_rough",
    "perlin_rough_stand",
    "square_gaps",
    "pyramid_stairs",
    "pyramid_stairs_high",
    "pyramid_stairs_inv",
    "pyramid_stairs_inv_high",
    "boxes",
    "mesh_boxes",
    "hf_pyramid_slope_inv",
)


@pytest.fixture(scope="module")
def task():
    return rough_g1()


def test_the_task_carries_the_shared_rough_recipe(task) -> None:
    terrain = task.scene.terrain
    assert task.task_id == "Instinct-Velocity-Rough-G1"
    assert terrain.kind == "rough"
    assert terrain.generator is not None
    assert terrain.generator.horizontal_scale == 0.05
    assert terrain.generator.num_cols == 20
    assert tuple(terrain.generator.sub_terrains) == TERRAIN_NAMES
    assert set(task.engines) == {"isaacsim", "mjlab"}
    assert not task.engine_extras


def test_locomotion_and_parkour_use_the_same_training_recipe() -> None:
    from instinctlab.tasks.parkour.config.g1 import parkour_target_g1

    locomotion = rough_g1().scene.terrain.generator
    parkour = parkour_target_g1().scene.terrain.generator
    assert locomotion == parkour == rough_terrain().generator


def test_the_curriculum_is_required_and_names_the_command(task) -> None:
    from instinctlab.spec.capability import Requirement

    term = task.mdp.curriculum["terrain_levels"]
    assert term.func is not None
    assert term.params["command_name"] == "base_velocity"
    assert term.level is Requirement.REQUIRED


def test_the_declaration_imports_no_engine() -> None:
    for source in (DECLARATION, RECIPE):
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        forbidden = [
            name
            for name in imported
            if name.split(".")[0] in {"isaaclab", "mjlab", "omni", "mujoco", "isaacsim"}
        ]
        assert not forbidden, f"{source.name} imports {forbidden}."


def _top_level_roots(path: Path) -> set[str]:
    imported: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def test_adapter_bridge_modules_import_no_engine_at_rest() -> None:
    for source in (ISAAC_ROUGH, MJLAB_ROUGH):
        leaked = _top_level_roots(source) & {"isaaclab", "isaacsim", "mjlab", "omni", "mujoco"}
        assert not leaked, f"{source.name} imports {sorted(leaked)} at module top."


def test_the_mjlab_package_front_does_not_load_the_terrain_stack() -> None:
    text = MJLAB_INIT.read_text()
    assert "from .env" not in text
    assert "from .terrains" not in text
    assert "import .terrains" not in text


def test_both_backends_report_that_they_can_run_this_task(task) -> None:
    from instinctlab.engines.isaacsim import IsaacSimAdapter
    from instinctlab.engines.mjlab import MjlabAdapter

    for adapter in (IsaacSimAdapter(), MjlabAdapter()):
        report = adapter.contract_report(task)
        assert report["missing"] == {}, report["missing"]
        assert report["engine_extras_used"] == []


def test_mjlab_compiles_the_shared_recipe_and_rough_capacity_profile(task) -> None:
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab import MjlabAdapter
    from instinctlab.engines.mjlab.env import TerrainAwareRlEnv
    from instinctlab.engines.mjlab.terrains.mesh_terrains_cfg import PerlinMeshRandomMultiBoxTerrainCfg
    from instinctlab.engines.mjlab.terrains.terrain_generator import FiledTerrainGenerator
    from instinctlab.engines.mjlab.terrains.terrain_importer import TerrainImporter

    compiled = MjlabAdapter().compile(task, num_envs=16, device="cpu")
    terrain = compiled.env_cfg.scene.terrain
    generator = terrain.terrain_generator
    assert compiled.env_cls is TerrainAwareRlEnv
    assert terrain.terrain_type == "hacked_generator"
    assert terrain.class_type is TerrainImporter
    assert generator is not None
    assert generator.class_type is FiledTerrainGenerator
    assert terrain.max_init_terrain_level == task.scene.terrain.generator.max_init_level == 5
    assert generator.horizontal_scale == task.scene.terrain.generator.horizontal_scale == 0.05
    assert generator.num_cols == task.scene.terrain.generator.num_cols == 20
    assert tuple(generator.sub_terrains) == TERRAIN_NAMES
    assert isinstance(generator.sub_terrains["mesh_boxes"], PerlinMeshRandomMultiBoxTerrainCfg)
    assert compiled.env_cfg.sim.nconmax == 512
    assert compiled.env_cfg.sim.njmax == 1536
    assert compiled.env_cfg.sim.contact_sensor_maxmatch == 128


def test_mjlab_effective_heightfield_scale_comes_from_the_shared_recipe(task) -> None:
    """Mutation guard: the bridge must propagate the declared value into every native hfield tile."""
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab.rough import rough_generator_cfg
    from instinctlab.engines.mjlab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
    from instinctlab.engines.mjlab.terrains.terrain_generator import FiledTerrainGenerator

    declared = task.scene.terrain.generator
    mutated = replace(declared, horizontal_scale=0.08)
    native = rough_generator_cfg(mutated)
    effective = FiledTerrainGenerator(native, device="cpu").cfg
    hfields = [tile for tile in effective.sub_terrains.values() if isinstance(tile, HfTerrainBaseCfg)]
    assert hfields
    assert {tile.horizontal_scale for tile in hfields} == {0.08}


def test_mjlab_constructs_a_five_centimeter_native_heightfield(task) -> None:
    """The compiled value must determine the actual MuJoCo hfield lattice."""
    pytest.importorskip("mjlab")
    import mujoco
    import numpy as np

    from instinctlab.engines.mjlab.rough import rough_generator_cfg
    from instinctlab.engines.mjlab.terrains.terrain_generator import FiledTerrainGenerator

    declared = task.scene.terrain.generator
    native = rough_generator_cfg(declared)
    effective = FiledTerrainGenerator(native, device="cpu").cfg
    tile = effective.sub_terrains["perlin_rough"]
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = tile.function(0.5, spec, np.random.default_rng(0))
    hfield = output.geometries[0].hfield
    assert hfield is not None
    assert hfield.nrow == int(declared.size[0] / declared.horizontal_scale) == 160
    assert hfield.ncol == int(declared.size[1] / declared.horizontal_scale) == 160


def test_mjlab_mesh_boxes_build_native_collision_and_target_patches(task) -> None:
    pytest.importorskip("mjlab")
    import mujoco
    import numpy as np

    from instinctlab.engines.mjlab.rough import rough_generator_cfg
    from instinctlab.engines.mjlab.terrains.terrain_generator import FiledTerrainGenerator

    native = rough_generator_cfg(task.scene.terrain.generator)
    effective = FiledTerrainGenerator(native, device="cpu").cfg
    tile = effective.sub_terrains["mesh_boxes"]
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = tile.function(0.5, spec, np.random.default_rng(0))
    assert len(output.geometries) > 50
    assert all(geometry.geom is not None for geometry in output.geometries)
    assert output.flat_patches is not None
    assert output.flat_patches["target"].shape == (50, 3)
    assert np.all(output.flat_patches["target"][:, 2] == 0.0)
    assert getattr(output, "instinct_surface_mesh").faces.shape[0] > 0


def test_mjlab_hfield_repair_mapping_skips_mesh_only_cells(task) -> None:
    """A mesh tile between hfields must not shift the importer cfg association."""
    pytest.importorskip("mjlab")
    import mujoco
    from types import SimpleNamespace

    from instinctlab.engines.mjlab.rough import rough_generator_cfg
    from instinctlab.engines.mjlab.terrains.terrain_generator import FiledTerrainGenerator
    from instinctlab.engines.mjlab.terrains.terrain_importer import TerrainImporter

    native = rough_generator_cfg(task.scene.terrain.generator)
    native.num_rows = 1
    native.num_cols = 3
    native.sub_terrains = {
        name: native.sub_terrains[name]
        for name in ("perlin_rough", "mesh_boxes", "hf_pyramid_slope_inv")
    }
    generator = FiledTerrainGenerator(native, device="cpu")
    spec = mujoco.MjSpec()
    generator.compile(spec)
    holder = SimpleNamespace(
        _spec=spec,
        terrain_generator=generator,
        subterrain_specific_cfgs=generator.subterrain_specific_cfgs,
    )
    pairs = list(TerrainImporter._iter_hfield_geoms_with_subterrain_cfgs(holder))
    assert [type(cfg).__name__ for _, cfg in pairs] == [
        "PerlinPlaneTerrainCfg",
        "PerlinInvertedPyramidSlopedTerrainCfg",
    ]


def test_the_mjlab_importer_exposes_variant_metadata() -> None:
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab.terrains.terrain_importer_cfg import TerrainImporterCfg

    terrain = TerrainImporterCfg(terrain_type="plane").class_type(
        TerrainImporterCfg(terrain_type="plane"), device="cpu"
    )
    assert terrain.variant_metadata is None
