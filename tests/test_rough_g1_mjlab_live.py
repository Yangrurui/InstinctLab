"""The rough mjlab env must construct and step, not merely compile.

``variant_metadata`` failed only after ``make_env()``. A compile-time snapshot of the cfg cannot
see that the importer skipped ``Entity.__init__``. This test builds a shrunk copy of the same
importer path play uses, then asks the live scene what generator it actually ran.
"""

from __future__ import annotations

import torch

import pytest

from tests.live_device import resolve_live_device

pytest.importorskip("mjlab")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stepping mjlab needs a GPU")
def test_mjlab_rough_constructs_the_filed_generator_and_steps() -> None:
    from instinctlab.engines.mjlab import MjlabAdapter
    from instinctlab.engines.mjlab.terrains.terrain_generator import FiledTerrainGenerator
    from instinctlab.engines.mjlab.terrains.terrain_importer import TerrainImporter
    from instinctlab.tasks.locomotion.config.g1 import rough_g1

    compiled = MjlabAdapter().compile(rough_g1(), num_envs=4, device=resolve_live_device())
    generator = compiled.env_cfg.scene.terrain.terrain_generator
    names = list(generator.sub_terrains)
    generator.sub_terrains = {name: generator.sub_terrains[name] for name in names[:2]}
    generator.num_rows = 2
    generator.num_cols = 2

    env = compiled.make_env()
    try:
        terrain = env.scene.terrain
        assert type(terrain) is TerrainImporter
        assert getattr(terrain, "_hacked_terrain_type", None) == "hacked_generator"
        assert type(terrain.terrain_generator) is FiledTerrainGenerator
        env.reset()
        actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        for _ in range(5):
            env.step(actions)
    finally:
        env.close()
