"""FiledTerrainGenerator honors ``num_cols`` and indexes cells by the built width.

No GPU. Imports mjlab (the generator subclasses it). Default ``pytest tests/`` still
collects this file.
"""

from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

import pytest

pytest.importorskip("mjlab")

from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg

from instinctlab.compat.terrain import curriculum_column_indices, even_column_assignment
from instinctlab.engines.mjlab.terrains.terrain_generator import FiledTerrainGenerator
from instinctlab.engines.mjlab.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
from instinctlab.spec import SubTerrainSpec, TerrainGeneratorSpec, TerrainSpec


def _cfg(*, num_rows=3, num_cols=6, curriculum=True):
    return FiledTerrainGeneratorCfg(
        seed=0,
        size=(8.0, 8.0),
        num_rows=num_rows,
        num_cols=num_cols,
        curriculum=curriculum,
        sub_terrains={
            "a": BoxFlatTerrainCfg(proportion=0.5),
            "b": BoxFlatTerrainCfg(proportion=0.5),
        },
    )


def test_curriculum_init_uses_declared_num_cols_not_type_count() -> None:
    gen = FiledTerrainGenerator(_cfg(num_cols=6), device="cpu")
    assert gen._num_cols == 6
    assert gen.terrain_origins.shape == (3, 6, 3)
    assert gen.cfg.num_cols == 6
    assert len(gen.cfg.sub_terrains) == 2


def test_get_subterrain_cfg_uses_built_width_not_cfg_num_cols() -> None:
    """The old ``row * cfg.num_cols + col`` returned another cell when those differed."""
    gen = FiledTerrainGenerator(_cfg(num_rows=2, num_cols=4, curriculum=False), device="cpu")
    assert gen._num_cols == 4
    sentinels = [SimpleNamespace(name=f"r{row}c{col}") for row in range(2) for col in range(4)]
    gen._subterrain_specific_cfgs = sentinels
    gen.cfg.num_cols = 20
    assert gen.get_subterrain_cfg(1, 3).name == "r1c3"
    old_idx = 1 * 20 + 3
    assert old_idx >= len(sentinels)
    assert gen.get_subterrain_cfg(1, 3) is not sentinels[1 * 4 + 3] or sentinels[1 * 4 + 3].name == "r1c3"
    assert gen._cell_index(1, 3) == 7
    assert old_idx == 23


def test_curriculum_compile_writes_proportion_columns_and_identical_row_difficulty() -> None:
    import mujoco

    gen = FiledTerrainGenerator(_cfg(num_rows=3, num_cols=6), device="cpu")
    spec = mujoco.MjSpec()
    gen.compile(spec)
    assigned = curriculum_column_indices([0.5, 0.5], 6)
    assert assigned == [0, 0, 0, 1, 1, 1]
    for col, type_idx in enumerate(assigned):
        for row in range(3):
            cell = gen.get_subterrain_cfg(row, col)
            assert cell is not None
            assert cell.proportion == pytest.approx([0.5, 0.5][type_idx])
            assert cell.difficulty == pytest.approx(row / 2.0)
    # Duplicate columns of the same type at one row share a difficulty.
    assert gen.get_subterrain_cfg(1, 0).difficulty == gen.get_subterrain_cfg(1, 1).difficulty
    assert gen.get_subterrain_cfg(1, 0).difficulty == pytest.approx(0.5)


def test_importer_rejects_type_level_spawn_weights() -> None:
    from instinctlab.engines.mjlab.terrains.terrain_importer import TerrainImporter
    from instinctlab.engines.mjlab.terrains.terrain_importer_cfg import TerrainImporterCfg

    importer = TerrainImporter(TerrainImporterCfg(terrain_type="plane", num_envs=8), device="cpu")
    origins = torch.zeros(2, 4, 3)
    with pytest.raises(RuntimeError, match="even-splits"):
        importer._compute_env_origins_curriculum(8, origins, proportions=np.ones(4) / 4)
    importer._compute_env_origins_curriculum(8, origins)
    assert torch.equal(importer.terrain_types, even_column_assignment(8, 4, device=importer._device))


def test_portable_generator_uses_filed_width_and_applies_terrain_friction() -> None:
    from instinctlab.engines.mjlab.scene import _terrain
    from instinctlab.engines.mjlab.terrains.terrain_generator import FiledTerrainGenerator

    generator = TerrainGeneratorSpec(
        num_rows=2,
        num_cols=4,
        curriculum=True,
        sub_terrains={
            "rough": SubTerrainSpec(
                kind="random_rough",
                params={"noise_range": (0.0, 0.01), "noise_step": 0.01},
            )
        },
    )
    cfg = _terrain(
        TerrainSpec(kind="generator", generator=generator, static_friction=0.7, dynamic_friction=0.7),
        {},
    )
    importer = cfg.class_type(cfg, device="cpu")

    assert isinstance(importer.terrain_generator, FiledTerrainGenerator)
    assert tuple(importer.terrain_origins.shape) == (2, 4, 3)
    assert all(float(geom.friction[0]) == pytest.approx(0.7) for geom in importer.spec.geoms)


def test_mjlab_refuses_terrain_material_semantics_it_cannot_represent() -> None:
    from instinctlab.engines.mjlab.scene import _terrain

    with pytest.raises(ValueError, match="one sliding-friction"):
        _terrain(TerrainSpec(static_friction=0.9, dynamic_friction=0.7), {})
    with pytest.raises(ValueError, match="cannot honor restitution"):
        _terrain(TerrainSpec(restitution=0.1), {})
