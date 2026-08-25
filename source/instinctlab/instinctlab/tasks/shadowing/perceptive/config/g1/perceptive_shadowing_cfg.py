"""Unified perceptive and single-motion G1 shadowing declarations."""

from instinctlab.tasks.shadowing.task_spec import ShadowingVariant, build_shadowing_task

TASK_ID = "Instinct-Perceptive-Shadowing-G1-v0"
PLAY_TASK_ID = "Instinct-Perceptive-Shadowing-G1-Play-v0"
ONE_MOTION_TASK_ID = "Instinct-Perceptive-Shadowing-G1-OneMotion-v0"
ONE_MOTION_PLAY_TASK_ID = "Instinct-Perceptive-Shadowing-G1-OneMotion-Play-v0"


def g1_perceptive_shadowing(*, play: bool = False, one_motion: bool = False):
    ids = {
        (False, False): TASK_ID,
        (True, False): PLAY_TASK_ID,
        (False, True): ONE_MOTION_TASK_ID,
        (True, True): ONE_MOTION_PLAY_TASK_ID,
    }
    return build_shadowing_task(
        ShadowingVariant(task_id=ids[(play, one_motion)], family="perceptive", play=play, one_motion=one_motion)
    )


def g1_perceptive_shadowing_play():
    return g1_perceptive_shadowing(play=True)


def g1_perceptive_shadowing_one_motion():
    return g1_perceptive_shadowing(one_motion=True)


def g1_perceptive_shadowing_one_motion_play():
    return g1_perceptive_shadowing(play=True, one_motion=True)


__all__ = [
    "g1_perceptive_shadowing",
    "g1_perceptive_shadowing_one_motion",
    "g1_perceptive_shadowing_one_motion_play",
    "g1_perceptive_shadowing_play",
]
