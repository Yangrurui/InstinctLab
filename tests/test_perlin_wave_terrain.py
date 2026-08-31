"""The application-owned Perlin wave kind builds real MJLab geometry."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mjlab")

from instinctlab.terrains.registration import PERLIN_WAVE_KIND, register_terrains
from instinctlab_engine.registry import TerrainExtensionRegistry
from instinctlab_engine.spec import SubTerrainSpec, TerrainGeneratorSpec


def _portable_wave() -> tuple[SubTerrainSpec, TerrainGeneratorSpec]:
    tile = SubTerrainSpec(
        kind=PERLIN_WAVE_KIND,
        params={
            "amplitude_range": (0.12, 0.36),
            "num_waves": 4,
            "border_width": 0.0,
            "perlin_cfg": {
                "noise_scale": 0.04,
                "noise_frequency": 20,
                "fractal_octaves": 2,
                "fractal_lacunarity": 2.0,
                "fractal_gain": 0.25,
                "centering": True,
            },
        },
    )
    generator = TerrainGeneratorSpec(
        seed=17,
        size=(8.0, 8.0),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        slope_threshold=1.0,
        sub_terrains={"wave": tile},
    )
    return tile, generator


def _apply_generator(native, generator: TerrainGeneratorSpec) -> None:
    native.size = generator.size
    native.horizontal_scale = generator.horizontal_scale
    native.vertical_scale = generator.vertical_scale
    native.slope_threshold = generator.slope_threshold


def test_mjlab_materializes_registered_perlin_wave_collision_geometry() -> None:
    import mujoco
    from instinctlab_engine_mjlab.terrains.height_field.hf_terrains_cfg import (
        PerlinWaveTerrainCfg,
    )

    tile, generator = _portable_wave()
    registry = TerrainExtensionRegistry(load_entry_points=False)
    register_terrains(registry)
    builder = registry.sub_terrain("mjlab", PERLIN_WAVE_KIND)
    assert builder is not None

    native = builder(tile, generator)
    assert isinstance(native, PerlinWaveTerrainCfg)
    _apply_generator(native, generator)

    np.random.seed(generator.seed)
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = native.function(0.65, spec, np.random.default_rng(generator.seed))

    assert len(output.geometries) == 1
    assert output.geometries[0].hfield is not None
    assert output.geometries[0].hfield.nrow == 160
    assert output.geometries[0].hfield.ncol == 160
    surface = output.instinct_surface_mesh
    assert surface.faces.shape == (2 * 159 * 159, 3)
    assert np.isfinite(surface.vertices).all()
    assert float(surface.bounds[1, 2] - surface.bounds[0, 2]) > 0.40
