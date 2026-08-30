"""Motion-reference lifecycle on a minimal live Isaac scene."""

from __future__ import annotations

import pytest

from tests.isaacsim_app import ensure_isaac_app
from tests.live_device import resolve_live_device

pytestmark = pytest.mark.isaacsim


def test_isaac_motion_clock_and_exhaustion() -> None:
    device = resolve_live_device()
    pytest.importorskip("isaaclab")
    ensure_isaac_app(device=device)

    from instinctlab_engine_isaacsim import IsaacSimAdapter
    from tests.motion_reference_live_expect import assert_motion_clock_and_exhaustion, motion_only_parkour_task
    from tests.parkour_live_expect import assert_amp_same_function

    task = motion_only_parkour_task("isaacsim")
    env = IsaacSimAdapter().compile(task, num_envs=4, device=device).make_env()
    try:
        assert_motion_clock_and_exhaustion(env, task, device=device)
        assert_amp_same_function(env, task, device=device)
    finally:
        env.close()
