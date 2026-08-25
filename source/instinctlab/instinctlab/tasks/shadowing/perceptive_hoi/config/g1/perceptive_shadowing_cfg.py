"""Unified perceptive HOI G1 shadowing declaration."""

from instinctlab.tasks.shadowing.task_spec import ShadowingVariant, build_shadowing_task

TASK_ID = "Instinct-Perceptive-HOI-Shadowing-G1-v0"
PLAY_TASK_ID = "Instinct-Perceptive-HOI-Shadowing-G1-Play-v0"


def g1_perceptive_hoi_shadowing(*, play: bool = False):
    return build_shadowing_task(
        ShadowingVariant(task_id=PLAY_TASK_ID if play else TASK_ID, family="perceptive_hoi", play=play)
    )


def g1_perceptive_hoi_shadowing_play():
    return g1_perceptive_hoi_shadowing(play=True)


__all__ = ["g1_perceptive_hoi_shadowing", "g1_perceptive_hoi_shadowing_play"]
