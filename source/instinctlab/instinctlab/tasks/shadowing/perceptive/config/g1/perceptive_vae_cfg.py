"""Unified perceptive VAE G1 shadowing declaration."""

from instinctlab.tasks.shadowing.task_spec import ShadowingVariant, build_shadowing_task

TASK_ID = "Instinct-Perceptive-Vae-G1-v0"
PLAY_TASK_ID = "Instinct-Perceptive-Vae-G1-Play-v0"


def g1_perceptive_vae(*, play: bool = False):
    return build_shadowing_task(
        ShadowingVariant(task_id=PLAY_TASK_ID if play else TASK_ID, family="perceptive_vae", play=play)
    )


def g1_perceptive_vae_play():
    return g1_perceptive_vae(play=True)


__all__ = ["g1_perceptive_vae", "g1_perceptive_vae_play"]
