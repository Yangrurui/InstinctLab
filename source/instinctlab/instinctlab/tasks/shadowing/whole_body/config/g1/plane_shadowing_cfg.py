"""Unified G1 whole-body shadowing declarations."""

from instinctlab.tasks.shadowing.task_spec import ShadowingVariant, build_shadowing_task

TASK_ID = "Instinct-Shadowing-WholeBody-Plane-G1-v0"
PLAY_TASK_ID = "Instinct-Shadowing-WholeBody-Plane-G1-Play-v0"


def g1_plane_shadowing(*, play: bool = False):
    return build_shadowing_task(
        ShadowingVariant(task_id=PLAY_TASK_ID if play else TASK_ID, family="whole_body", play=play)
    )


def g1_plane_shadowing_play():
    return g1_plane_shadowing(play=True)


__all__ = ["g1_plane_shadowing", "g1_plane_shadowing_play"]
