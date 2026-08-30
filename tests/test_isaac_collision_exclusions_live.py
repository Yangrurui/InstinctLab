"""Isaac preserves portable collision exclusions across environment clones.

Run on demand:

    pytest -o addopts= -m isaacsim tests/test_isaac_collision_exclusions_live.py
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.isaacsim_app import ensure_isaac_app
from tests.live_device import resolve_live_device

pytestmark = pytest.mark.isaacsim


def test_filtered_pair_targets_are_remapped_for_every_clone() -> None:
    device = resolve_live_device()
    ensure_isaac_app(device=device)

    import isaaclab.sim as sim_utils
    from pxr import UsdPhysics

    from instinctlab_engine.spec import CollisionExclusionRef
    from instinctlab_engine_isaacsim import IsaacSimAdapter

    from tests.task_specs import task_spec

    exclusions = (
        CollisionExclusionRef("left_elbow_link", "left_wrist_pitch_link"),
        CollisionExclusionRef("right_elbow_link", "right_wrist_pitch_link"),
        CollisionExclusionRef("pelvis", "right_hip_roll_link"),
        CollisionExclusionRef("pelvis", "left_hip_roll_link"),
    )
    task = task_spec("Instinct-Velocity-Flat-G1", "isaacsim")
    task = replace(
        task,
        scene=replace(task.scene, collision_exclusions=exclusions),
    )
    env = IsaacSimAdapter().compile(task, num_envs=4, device=device).make_env()
    try:
        stage = sim_utils.get_current_stage()
        targets_by_source: dict[str, set[str]] = {}
        for exclusion in exclusions:
            targets_by_source.setdefault(exclusion.body_a, set()).add(exclusion.body_b)
        for env_index in range(env.num_envs):
            robot_path = f"/World/envs/env_{env_index}/Robot"
            for source_body, target_bodies in targets_by_source.items():
                source_path = f"{robot_path}/{source_body}"
                source = stage.GetPrimAtPath(source_path)
                assert source.IsValid(), source_path
                relation = UsdPhysics.FilteredPairsAPI(source).GetFilteredPairsRel()
                assert relation.IsValid(), source_path
                assert {str(path) for path in relation.GetTargets()} == {
                    f"{robot_path}/{target_body}" for target_body in target_bodies
                }
    finally:
        env.close()
