"""Isaac USD lowering for portable scene relations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def with_collision_exclusions(spawn: Any, exclusions: Sequence[Any]) -> Any:
    """Apply body filtered-pair relationships to the source articulation.

    Isaac's homogeneous environment clone then reproduces these internal
    relationships for every environment.
    """
    exclusions = tuple(exclusions)
    if not exclusions:
        return spawn
    original_func = spawn.func

    def spawn_with_exclusions(
        prim_path: str,
        cfg: Any,
        translation=None,
        orientation=None,
    ):
        from pxr import Sdf, UsdPhysics

        from isaaclab.sim import find_matching_prim_paths, get_current_stage

        native_cfg = cfg.replace(func=original_func)
        prim = original_func(
            prim_path,
            native_cfg,
            translation=translation,
            orientation=orientation,
        )
        stage = get_current_stage()
        robot_paths = find_matching_prim_paths(prim_path)
        if not robot_paths:
            raise RuntimeError(
                f"Isaac collision exclusions matched no articulation at {prim_path!r}."
            )
        for robot_path in robot_paths:
            for exclusion in exclusions:
                path_a = f"{robot_path}/{exclusion.body_a}"
                path_b = f"{robot_path}/{exclusion.body_b}"
                prim_a = stage.GetPrimAtPath(path_a)
                prim_b = stage.GetPrimAtPath(path_b)
                if not prim_a.IsValid() or not prim_b.IsValid():
                    raise RuntimeError(
                        f"Isaac collision exclusion {exclusion.pair!r} did not resolve "
                        f"under {robot_path!r}."
                    )
                relation = UsdPhysics.FilteredPairsAPI.Apply(
                    prim_a
                ).CreateFilteredPairsRel()
                relation.AddTarget(Sdf.Path(path_b))
        return prim

    return spawn.replace(func=spawn_with_exclusions)


__all__ = ["with_collision_exclusions"]
