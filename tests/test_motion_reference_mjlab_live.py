"""Motion-reference lifecycle on a minimal live MJLab scene."""

from __future__ import annotations

import torch

import pytest

pytest.importorskip("mjlab")
pytestmark = pytest.mark.mjlab


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stepping mjlab needs a GPU")
def test_mjlab_motion_clock_and_exhaustion() -> None:
    from instinctlab_engine_mjlab import MjlabAdapter
    from tests.live_device import resolve_live_device
    from tests.motion_reference_live_expect import assert_motion_clock_and_exhaustion, motion_only_parkour_task
    from tests.parkour_live_expect import assert_amp_same_function

    device = resolve_live_device()
    task = motion_only_parkour_task("mjlab")
    env = MjlabAdapter().compile(task, num_envs=4, device=device).make_env()
    try:
        assert_motion_clock_and_exhaustion(env, task, device=device)
        assert_amp_same_function(env, task, device=device)
    finally:
        env.close()
