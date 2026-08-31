"""The application-owned Perlin wave kind builds real Isaac terrain geometry.

Run on demand:

    pytest -o addopts= -m isaacsim tests/test_perlin_wave_isaacsim_live.py
"""

from __future__ import annotations

import pytest

from tests.isaacsim_app import ensure_isaac_app
from tests.live_device import resolve_live_device

pytestmark = pytest.mark.isaacsim


def test_isaacsim_materializes_registered_perlin_wave_mesh() -> None:
    ensure_isaac_app(device=resolve_live_device())

    import numpy as np
    from instinctlab.terrains.registration import PERLIN_WAVE_KIND, register_terrains
    from instinctlab_engine.registry import TerrainExtensionRegistry
    from instinctlab_engine.spec import SubTerrainSpec, TerrainGeneratorSpec
    from instinctlab_engine_isaacsim.terrains import PerlinWaveTerrainCfg

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
    registry = TerrainExtensionRegistry(load_entry_points=False)
    register_terrains(registry)
    builder = registry.sub_terrain("isaacsim", PERLIN_WAVE_KIND)
    assert builder is not None

    native = builder(tile, generator)
    assert isinstance(native, PerlinWaveTerrainCfg)
    native.size = generator.size
    native.horizontal_scale = generator.horizontal_scale
    native.vertical_scale = generator.vertical_scale
    native.slope_threshold = generator.slope_threshold

    np.random.seed(generator.seed)
    meshes, origin = native.function(0.65, native)
    assert len(meshes) == 1
    assert meshes[0].faces.shape[0] > 50_000
    assert np.isfinite(meshes[0].vertices).all()
    assert float(meshes[0].bounds[1, 2] - meshes[0].bounds[0, 2]) > 0.40
    assert np.isfinite(origin).all()
