"""Unified BeyondMimic G1 shadowing declaration."""

from instinctlab.tasks.shadowing.task_spec import ShadowingVariant, build_shadowing_task

TASK_ID = "Instinct-BeyondMimic-Plane-G1-v0"
PLAY_TASK_ID = "Instinct-BeyondMimic-Plane-G1-Play-v0"


def g1_beyondmimic_plane(*, play: bool = False):
    return build_shadowing_task(
        ShadowingVariant(task_id=PLAY_TASK_ID if play else TASK_ID, family="beyondmimic", play=play)
    )


def g1_beyondmimic_plane_play():
    return g1_beyondmimic_plane(play=True)


__all__ = ["g1_beyondmimic_plane", "g1_beyondmimic_plane_play"]
