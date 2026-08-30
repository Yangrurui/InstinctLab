"""Portable scene relations are declared once and lowered by each backend."""

from __future__ import annotations

from dataclasses import dataclass, replace
import sys
from types import ModuleType, SimpleNamespace

import pytest

from instinctlab_engine.spec import CollisionExclusionRef, SceneSpec
from instinctlab_engine.spec.validation import validate_task

from tests.task_specs import task_spec


PERCEPTIVE_EXCLUSIONS = {
    ("left_elbow_link", "left_wrist_pitch_link"),
    ("right_elbow_link", "right_wrist_pitch_link"),
    ("pelvis", "right_hip_roll_link"),
    ("left_hip_roll_link", "pelvis"),
}


def test_collision_exclusions_are_unordered_and_unique() -> None:
    with pytest.raises(ValueError, match="unique unordered"):
        SceneSpec(
            collision_exclusions=(
                CollisionExclusionRef("pelvis", "left_hip_roll_link"),
                CollisionExclusionRef("left_hip_roll_link", "pelvis"),
            )
        )


def test_collision_exclusion_rejects_unknown_robot_body() -> None:
    task = task_spec("Instinct-Velocity-Flat-G1")
    scene = replace(
        task.scene,
        collision_exclusions=(
            CollisionExclusionRef("pelvis", "body_that_does_not_exist"),
        ),
    )

    with pytest.raises(ValueError, match="unknown robot bodies"):
        validate_task(replace(task, scene=scene))


@pytest.mark.parametrize(
    "task_id",
    (
        "Instinct-Perceptive-Shadowing-G1-v0",
        "Instinct-Perceptive-Shadowing-G1-Play-v0",
        "Instinct-Perceptive-Shadowing-G1-OneMotion-v0",
        "Instinct-Perceptive-Shadowing-G1-OneMotion-Play-v0",
        "Instinct-Perceptive-Vae-G1-v0",
        "Instinct-Perceptive-Vae-G1-Play-v0",
        "Instinct-Perceptive-HOI-Shadowing-G1-v0",
        "Instinct-Perceptive-HOI-Shadowing-G1-Play-v0",
    ),
)
def test_perceptive_tasks_declare_the_mjcf_collision_exclusions(task_id: str) -> None:
    declared = {exclusion.pair for exclusion in task_spec(task_id).scene.collision_exclusions}

    assert declared == PERCEPTIVE_EXCLUSIONS


@dataclass(frozen=True)
class _MjEntityCfg:
    spec_fn: object


def test_mjlab_relation_lowering_adds_only_missing_pairs() -> None:
    from instinctlab_engine_mjlab.relations import with_collision_exclusions

    existing = SimpleNamespace(bodyname1="pelvis", bodyname2="left_hip_roll_link")
    spec = SimpleNamespace(excludes=[existing])

    def add_exclude(**values):
        spec.excludes.append(SimpleNamespace(**values))

    spec.add_exclude = add_exclude
    cfg = _MjEntityCfg(spec_fn=lambda: spec)
    exclusions = (
        CollisionExclusionRef("left_hip_roll_link", "pelvis"),
        CollisionExclusionRef("left_elbow_link", "left_wrist_pitch_link"),
    )

    lowered = with_collision_exclusions(cfg, exclusions).spec_fn()

    assert {
        tuple(sorted((item.bodyname1, item.bodyname2)))
        for item in lowered.excludes
    } == {
        ("left_hip_roll_link", "pelvis"),
        ("left_elbow_link", "left_wrist_pitch_link"),
    }


class _IsaacSpawnCfg:
    def __init__(self, func):
        self.func = func

    def replace(self, **values):
        return _IsaacSpawnCfg(values.get("func", self.func))


def test_isaac_relation_lowering_applies_filtered_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    from instinctlab_engine_isaacsim.relations import with_collision_exclusions

    targets: list[str] = []

    class _Prim:
        def IsValid(self):
            return True

    class _Relation:
        def AddTarget(self, path):
            targets.append(str(path))

    class _FilteredPairs:
        @staticmethod
        def Apply(_prim):
            return SimpleNamespace(CreateFilteredPairsRel=lambda: _Relation())

    stage = SimpleNamespace(GetPrimAtPath=lambda _path: _Prim())
    pxr = ModuleType("pxr")
    pxr.Sdf = SimpleNamespace(Path=lambda value: value)
    pxr.UsdPhysics = SimpleNamespace(FilteredPairsAPI=_FilteredPairs)
    isaac_sim = ModuleType("isaaclab.sim")
    isaac_sim.find_matching_prim_paths = lambda _path: ["/World/envs/env_0/Robot"]
    isaac_sim.get_current_stage = lambda: stage
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", isaac_sim)
    original_calls = []

    def original(prim_path, cfg, translation=None, orientation=None):
        original_calls.append((prim_path, cfg, translation, orientation))
        return "robot_prim"

    lowered = with_collision_exclusions(
        _IsaacSpawnCfg(original),
        (CollisionExclusionRef("pelvis", "left_hip_roll_link"),),
    )

    assert lowered.func("/World/envs/env_.*/Robot", lowered) == "robot_prim"
    assert len(original_calls) == 1
    assert targets == ["/World/envs/env_0/Robot/left_hip_roll_link"]
