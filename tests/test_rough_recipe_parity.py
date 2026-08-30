"""The shared rough recipe follows main and both engines only lower it."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from instinctlab.tasks.terrain import rough_terrain
from instinctlab.utils.terrain_split_log import ALIGNED_TERRAINS, EXCLUDED_TERRAINS
from tests.engine_packages import ISAACSIM_ENGINE, MJLAB_ENGINE

REPO = Path(__file__).resolve().parent.parent
MAIN_PARKOUR = Path("/root/InstinctLab-main/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py")
ISAAC_BRIDGE = ISAACSIM_ENGINE / "terrain_builders.py"
MJLAB_BRIDGE = MJLAB_ENGINE / "terrain_builders.py"

TYPE_TO_KIND = {
    "PerlinPlaneTerrainCfg": "perlin_plane",
    "PerlinSquareGapTerrainCfg": "perlin_square_gap",
    "PerlinPyramidStairsTerrainCfg": "perlin_pyramid_stairs",
    "PerlinInvertedPyramidStairsTerrainCfg": "perlin_pyramid_stairs_inv",
    "PerlinDiscreteObstaclesTerrainCfg": "perlin_discrete_obstacles",
    "PerlinMeshRandomMultiBoxTerrainCfg": "perlin_random_multi_box",
    "PerlinInvertedPyramidSlopedTerrainCfg": "perlin_pyramid_slope_inv",
}


def _assigned_call(path: Path, name: str) -> ast.Call:
    assert path.is_file(), path
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            assert isinstance(node.value, ast.Call)
            return node.value
    raise LookupError(f"{path} has no {name} assignment")


def _value(node: ast.AST) -> Any:
    if isinstance(node, ast.Call):
        return {
            "_type": ast.unparse(node.func).rsplit(".", 1)[-1],
            **{keyword.arg: _value(keyword.value) for keyword in node.keywords if keyword.arg is not None},
        }
    if isinstance(node, ast.Dict):
        return {
            _value(key): _value(value)
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None
        }
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_value(element) for element in node.elts]
    return ast.literal_eval(node)


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items() if key != "_type"}
    return value


def test_shared_recipe_matches_main_effective_rough_config() -> None:
    reference = _assigned_call(MAIN_PARKOUR, "ROUGH_TERRAINS_CFG")
    reference_fields = {keyword.arg: _value(keyword.value) for keyword in reference.keywords if keyword.arg is not None}
    declared = rough_terrain().generator
    assert declared is not None

    for field in (
        "seed",
        "size",
        "border_width",
        "num_rows",
        "num_cols",
        "horizontal_scale",
        "vertical_scale",
        "slope_threshold",
        "curriculum",
    ):
        assert _plain(getattr(declared, field)) == _plain(reference_fields[field]), field

    reference_tiles: dict[str, dict[str, Any]] = reference_fields["sub_terrains"]
    assert list(declared.sub_terrains) == list(reference_tiles)
    for name, expected in reference_tiles.items():
        tile = declared.sub_terrains[name]
        assert tile.kind == TYPE_TO_KIND[expected["_type"]], name
        assert tile.proportion == expected["proportion"], name
        expected_params = {key: value for key, value in expected.items() if key not in {"_type", "proportion"}}
        if name in {"perlin_rough", "perlin_rough_stand"}:
            # Main omits this field; both native base classes default to zero.
            expected_params["border_width"] = 0.0
        assert _plain(tile.params) == _plain(expected_params), name


def test_engine_bridges_do_not_redeclare_training_recipe_literals() -> None:
    """Constants belong to tasks/terrain.py; adapter edits cannot fork training config."""
    for path in (ISAAC_BRIDGE, MJLAB_BRIDGE):
        text = path.read_text()
        assert "step_width=" not in text
        assert '"perlin_rough":' not in text
        assert '"mesh_boxes":' not in text
        assert "horizontal_scale=0.05" not in text
        assert "spec.horizontal_scale" in text


def test_recipe_resolution_is_explicitly_five_centimeters() -> None:
    generator = rough_terrain().generator
    assert generator is not None
    assert generator.horizontal_scale == 0.05


def test_terrain_split_log_tracks_the_shared_recipe() -> None:
    generator = rough_terrain().generator
    assert generator is not None
    assert ALIGNED_TERRAINS == tuple(generator.sub_terrains)
    assert EXCLUDED_TERRAINS == ()
