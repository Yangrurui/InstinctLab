"""What holds for the rough G1 declaration: intent, curriculum, and per-engine reference recipes."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from instinctlab.tasks.locomotion.config.g1 import rough_g1
from instinctlab.tasks.locomotion.terrains import rough_terrain

REPO = Path(__file__).resolve().parent.parent
DECLARATION = REPO / "source/instinctlab/instinctlab/tasks/locomotion/config/g1/rough_env_cfg.py"
RECIPE = REPO / "source/instinctlab/instinctlab/tasks/locomotion/terrains.py"
ISAAC_ROUGH = REPO / "source/instinctlab/instinctlab/engines/isaacsim/rough.py"
MJLAB_ROUGH = REPO / "source/instinctlab/instinctlab/engines/mjlab/rough.py"
MJLAB_INIT = REPO / "source/instinctlab/instinctlab/engines/mjlab/__init__.py"
INSTINCTMJ_PARKOUR = Path("/root/InstinctMJ/src/instinct_mj/tasks/parkour/config/parkour_env_cfg.py")

_ENGINE_ROOTS = frozenset({"isaaclab", "isaacsim", "mjlab", "omni", "mujoco"})

_GENERATOR_FIELDS = (
    "seed",
    "size",
    "border_width",
    "num_rows",
    "num_cols",
    "horizontal_scale",
    "vertical_scale",
    "slope_threshold",
    "curriculum",
)
_TILE_FIELDS = (
    "proportion",
    "step_width",
    "step_height_range",
    "platform_width",
    "border_width",
    "noise_scale",
    "noise_frequency",
    "num_obstacles",
    "gap_distance_range",
    "gap_depth",
    "slope_range",
    "box_height_mean",
    "generation_ratio",
    "obstacle_width_range",
    "obstacle_height_range",
    "obstacle_height_mode",
    "wall_prob",
    "wall_height",
    "wall_thickness",
)


@pytest.fixture(scope="module")
def task():
    return rough_g1()


def test_the_task_is_the_flat_mdp_on_reference_rough(task) -> None:
    assert task.task_id == "Instinct-Velocity-Rough-G1"
    assert task.scene.terrain.kind == "rough"
    assert task.scene.terrain.generator is None
    assert set(task.engines) == {"isaacsim", "mjlab"}
    assert not task.engine_extras


def test_the_declaration_does_not_name_tiles() -> None:
    """A shared tile list would force one reference to wear the other's numbers."""
    terrain = rough_terrain()
    assert terrain.kind == "rough"
    assert terrain.generator is None
    text = RECIPE.read_text()
    assert "pyramid_stairs" not in text
    assert "dense_boxes" not in text
    assert "mesh_boxes" not in text


def test_the_curriculum_is_required_and_names_the_command(task) -> None:
    term = task.mdp.curriculum["terrain_levels"]
    assert term.func is not None
    assert term.params["command_name"] == "base_velocity"
    from instinctlab.spec.capability import Requirement

    assert term.level is Requirement.REQUIRED


def test_the_declaration_imports_no_engine() -> None:
    for source in (DECLARATION, RECIPE):
        imported = set()
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        forbidden = [
            name for name in imported if name.split(".")[0] in {"isaaclab", "mjlab", "omni", "mujoco", "isaacsim"}
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


def test_adapter_recipe_modules_import_no_engine_at_rest() -> None:
    """Builders import the engine; the module being importable must not."""
    for source in (ISAAC_ROUGH, MJLAB_ROUGH):
        leaked = _top_level_roots(source) & _ENGINE_ROOTS
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


def _normalize(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _snapshot(generator: Any) -> dict[str, Any]:
    snap = {field: _normalize(getattr(generator, field)) for field in _GENERATOR_FIELDS}
    tiles: dict[str, Any] = {}
    for name, tile in generator.sub_terrains.items():
        entry = {"type": type(tile).__name__}
        for field in _TILE_FIELDS:
            if hasattr(tile, field):
                entry[field] = _normalize(getattr(tile, field))
        tiles[name] = entry
    snap["tiles"] = tiles
    return snap


def test_mjlab_compiles_instinctmj_parkour_rough(task) -> None:
    """Construction of the native config, not the simulator -- the tiles are not meshed here."""
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab import MjlabAdapter
    from instinctlab.engines.mjlab.env import TerrainAwareRlEnv
    from instinctlab.engines.mjlab.rough import rough_generator_cfg
    from instinctlab.engines.mjlab.terrains.terrain_generator import FiledTerrainGenerator
    from instinctlab.engines.mjlab.terrains.terrain_importer import TerrainImporter

    compiled = MjlabAdapter().compile(task, num_envs=16, device="cpu")
    terrain = compiled.env_cfg.scene.terrain
    assert compiled.env_cls is TerrainAwareRlEnv
    assert terrain.terrain_type == "hacked_generator"
    assert terrain.class_type is TerrainImporter
    assert terrain.terrain_generator is not None
    assert terrain.terrain_generator.class_type is FiledTerrainGenerator
    assert terrain.max_init_terrain_level == 5
    assert "terrain_levels" in compiled.env_cfg.curriculum
    assert compiled.env_cfg.sim.nconmax == 256
    assert compiled.env_cfg.sim.njmax == 768
    assert compiled.env_cfg.sim.contact_sensor_maxmatch == 128
    assert _snapshot(terrain.terrain_generator) == _snapshot(rough_generator_cfg())
    assert "dense_boxes" in terrain.terrain_generator.sub_terrains
    assert "mesh_boxes" not in terrain.terrain_generator.sub_terrains
    assert terrain.terrain_generator.horizontal_scale == 0.07


def test_the_mjlab_importer_exposes_variant_metadata() -> None:
    """The importer skips ``Entity.__init__``; Scene still reads this on every entity."""
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab.terrains.terrain_importer_cfg import TerrainImporterCfg

    terrain = TerrainImporterCfg(terrain_type="plane").class_type(
        TerrainImporterCfg(terrain_type="plane"), device="cpu"
    )
    assert terrain.variant_metadata is None


def _assigned_call(path: Path, name: str) -> ast.Call:
    assert path.is_file(), f"missing reference {path}"
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        assert isinstance(node.value, ast.Call)
        return node.value
    raise LookupError(f"{path} has no {name} assignment")


def _first_call_named(path: Path, suffix: str) -> ast.Call:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith(suffix):
            return node
    raise LookupError(f"{path} has no call ending in {suffix}")


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        if isinstance(node, ast.Dict):
            return {
                _literal(key): _literal(value)
                for key, value in zip(node.keys, node.values, strict=True)
                if key is not None
            }
        if isinstance(node, (ast.Tuple, ast.List)):
            return [_literal(element) for element in node.elts]
        return None


def _usable(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict) and any(item is None for item in value.values()):
        return False
    return True


def _compare_generator_call(ours: ast.Call, reference: ast.Call) -> None:
    """Literal fields and tile names; skip helper calls such as ``target()``."""
    ours_kw = {kw.arg: _literal(kw.value) for kw in ours.keywords if kw.arg is not None}
    ref_kw = {kw.arg: _literal(kw.value) for kw in reference.keywords if kw.arg is not None}
    for field in _GENERATOR_FIELDS:
        if not _usable(ref_kw.get(field)) or not _usable(ours_kw.get(field)):
            continue
        assert _normalize(ours_kw[field]) == _normalize(ref_kw[field]), field

    ours_tiles = next(kw.value for kw in ours.keywords if kw.arg == "sub_terrains")
    ref_tiles = next(kw.value for kw in reference.keywords if kw.arg == "sub_terrains")
    assert isinstance(ours_tiles, ast.Dict) and isinstance(ref_tiles, ast.Dict)
    ours_names = [key.value for key in ours_tiles.keys if isinstance(key, ast.Constant)]
    ref_names = [key.value for key in ref_tiles.keys if isinstance(key, ast.Constant)]
    assert ours_names == ref_names

    ours_by_name = {
        key.value: value
        for key, value in zip(ours_tiles.keys, ours_tiles.values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Call)
    }
    for key, value in zip(ref_tiles.keys, ref_tiles.values, strict=True):
        assert isinstance(key, ast.Constant) and isinstance(value, ast.Call)
        ours_tile = ours_by_name[key.value]
        assert ast.unparse(ours_tile.func).rsplit(".", 1)[-1] == ast.unparse(value.func).rsplit(".", 1)[-1]
        ours_fields = {kw.arg: _literal(kw.value) for kw in ours_tile.keywords if kw.arg is not None}
        for kw in value.keywords:
            if kw.arg is None:
                continue
            expected = _literal(kw.value)
            if not _usable(expected) or not _usable(ours_fields.get(kw.arg)):
                continue
            assert _normalize(ours_fields[kw.arg]) == _normalize(expected), (key.value, kw.arg)


def test_the_mjlab_recipe_matches_instinctmj_parkour_literals() -> None:
    """Read InstinctMJ's file; do not import the package.

    The mjlab recipe follows InstinctMJ's constants on purpose. A new
    unexplained literal difference fails here; the Isaac-vs-mjlab table lives
    in ``tests/test_rough_recipe_parity.py``.
    """
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab.rough import rough_generator_cfg

    reference = _assigned_call(INSTINCTMJ_PARKOUR, "ROUGH_TERRAINS_CFG")
    ours = rough_generator_cfg()
    kwargs = {kw.arg: _literal(kw.value) for kw in reference.keywords if kw.arg is not None}
    for field in _GENERATOR_FIELDS:
        if not _usable(kwargs.get(field)):
            continue
        assert _normalize(getattr(ours, field)) == _normalize(kwargs[field]), field

    tiles_node = next(kw.value for kw in reference.keywords if kw.arg == "sub_terrains")
    assert isinstance(tiles_node, ast.Dict)
    names = [key.value for key in tiles_node.keys if isinstance(key, ast.Constant)]
    assert list(ours.sub_terrains) == names
    for key, value in zip(tiles_node.keys, tiles_node.values, strict=True):
        assert isinstance(key, ast.Constant) and isinstance(value, ast.Call)
        tile = ours.sub_terrains[key.value]
        assert type(tile).__name__ == ast.unparse(value.func).rsplit(".", 1)[-1]
        for kw in value.keywords:
            if kw.arg is None:
                continue
            expected = _literal(kw.value)
            if not _usable(expected) or not hasattr(tile, kw.arg):
                continue
            assert _normalize(getattr(tile, kw.arg)) == _normalize(expected), (key.value, kw.arg)


def test_isaac_recipe_matches_main_parkour() -> None:
    """Ask main's parkour file, not an import that needs the USD stack."""
    parkour = REPO / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    _compare_generator_call(
        _first_call_named(ISAAC_ROUGH, "TerrainGeneratorCfg"), _assigned_call(parkour, "ROUGH_TERRAINS_CFG")
    )
