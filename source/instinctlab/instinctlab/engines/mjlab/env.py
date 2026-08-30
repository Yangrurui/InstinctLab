"""mjlab env whose Scene honors terrain ``cfg.class_type``.

mjlab's ``Scene._add_terrain`` always constructs ``TerrainEntity``, so a
``FiledTerrainGenerator`` / Instinct importer never runs -- the same hole InstinctMJ
patched with ``InstinctScene``. This module is imported only from the adapter's
``compile`` body.
"""

from __future__ import annotations

import warnings

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.terrains import TerrainEntity
from mjlab.utils.spec import non_default_option_fields

from .terrains.terrain_importer_cfg import TerrainImporterCfg

__all__ = ["TerrainAwareRlEnv", "TerrainAwareScene"]


class TerrainAwareScene(Scene):
    """``Scene`` that uses ``terrain_cfg.class_type`` when the importer declares one."""

    def _add_terrain(self) -> None:
        if self._cfg.terrain is None:
            return
        terrain_cfg = self._cfg.terrain
        terrain_cfg.num_envs = self._cfg.num_envs
        terrain_cfg.env_spacing = self._cfg.env_spacing
        if isinstance(terrain_cfg, TerrainImporterCfg):
            terrain = terrain_cfg.class_type(terrain_cfg, device=self._device)
        else:
            terrain = TerrainEntity(terrain_cfg, device=self._device)
        self._terrain = terrain
        self._entities["terrain"] = terrain
        non_default = non_default_option_fields(terrain.spec.option)
        if non_default:
            fields = ", ".join(non_default)
            warnings.warn(
                f"Terrain has non-default <option> fields ({fields}) that will not be"
                " propagated by MjSpec.attach(). Use MujocoCfg instead.",
                stacklevel=2,
            )
        frame = self._spec.worldbody.add_frame()
        self._spec.attach(terrain.spec, prefix="", frame=frame)


class TerrainAwareRlEnv(ManagerBasedRlEnv):
    """``ManagerBasedRlEnv`` that constructs :class:`TerrainAwareScene`.

    Construction and every ``step`` poll mujoco_warp ``d.overflow``. Overflow
    drops contacts without raising; a run that continues is a trained policy
    on the wrong physics. See :mod:`instinctlab.engines.diagnostics.contact_overflow`.
    """

    def __init__(
        self, cfg, device: str, render_mode: str | None = None, **kwargs
    ) -> None:
        import mjlab.envs.manager_based_rl_env as env_mod

        from instinctlab.engines.diagnostics.contact_overflow import (
            check_contact_overflow,
        )

        previous = env_mod.Scene
        env_mod.Scene = TerrainAwareScene
        try:
            super().__init__(cfg, device, render_mode=render_mode, **kwargs)
        finally:
            env_mod.Scene = previous
        check_contact_overflow(self, phase="construction")

    def step(self, action):
        result = super().step(action)
        from instinctlab.engines.diagnostics.contact_overflow import (
            check_contact_overflow,
        )

        check_contact_overflow(self, phase="step")
        return result

    def update_visualizers(self, visualizer) -> None:
        """InstinctMJ draws virtual edge cylinders through the terrain importer."""
        super().update_visualizers(visualizer)
        terrain = self.scene.terrain
        if terrain is not None and hasattr(terrain, "debug_vis"):
            terrain.debug_vis(visualizer)

    def _reset_idx(self, env_ids=None) -> None:
        from instinctlab.compat.observation_history import (
            clear_observation_histories_on_reset,
        )

        clear_observation_histories_on_reset(self, env_ids)
        super()._reset_idx(env_ids)
