"""Unified preflight fails before resolving native implementation objects."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

import instinctlab_engine
from instinctlab_engine.actuators import (
    APPLIED_EFFORT,
    JOINT_POSITION_COMMAND,
    ActuatorRegistry,
)
from instinctlab_engine.preflight import PreflightError, preflight_report, require_preflight
from instinctlab_engine.spec import ArticulationRef, EntityRef, NativeSensorRef
from instinctlab.tasks import registry

from tests.task_specs import task_spec, with_rigid_object_fixture


REGISTERED_TASK_ENGINES = tuple(
    (task_id, engine)
    for task_id in registry.ids()
    for engine in task_spec(task_id).engines
)


@pytest.mark.parametrize("engine", ("isaacsim", "mjlab"))
def test_builtin_flat_task_passes_preflight_with_exact_selected_components(
    engine: str,
) -> None:
    task = task_spec("Instinct-Velocity-Flat-G1", engine)

    report = preflight_report(task, engine)

    assert report["status"] == "ok"
    assert report["asset"]["status"] == "ok"
    assert report["selected_components"]["asset_id"] == task.robot.asset_id
    assert len(report["selected_components"]["actuator_model_ids"]) == 1
    assert report["actuators"][0]["missing_capabilities"] == []
    assert report["requested_capabilities"]["actuator_by_term"][
        "action/joint_pos"
    ] == ["joint_position_command"]
    assert report["incompatibilities"] == []
    assert report["omissions"] == []
    provider_groups = {provider["group"] for provider in report["providers"]}
    assert {
        "instinctlab.actuators",
        "instinctlab.assets",
        "instinctlab.engines",
    } <= provider_groups


def test_generated_terrain_preflight_uses_registered_tile_kinds() -> None:
    task = task_spec("Instinct-Velocity-Rough-G1", "mjlab")

    report = require_preflight(task, "mjlab")

    assert report["selected_components"]["sub_terrain_kinds"] == [
        tile.kind for tile in task.scene.terrain.generator.sub_terrains.values()
    ]
    assert "perlin_pyramid_stairs" in report["selected_components"][
        "sub_terrain_kinds"
    ]
    assert "pyramid_stairs" not in report["selected_components"][
        "sub_terrain_kinds"
    ]


def test_preflight_accounts_for_every_articulation_asset() -> None:
    task = task_spec("Instinct-Velocity-Flat-G1", "mjlab")
    task = replace(
        task,
        scene=replace(
            task.scene,
            articulations=(
                ArticulationRef(
                    "training_partner",
                    replace(task.robot, name="training_partner"),
                ),
            ),
        ),
    )

    report = require_preflight(task, "mjlab")

    assert report["additional_articulations"][0]["name"] == "training_partner"
    assert report["additional_articulations"][0]["asset"]["status"] == "ok"
    assert report["selected_components"]["articulation_asset_ids"] == {
        "robot": task.robot.asset_id,
        "training_partner": task.robot.asset_id,
    }


@pytest.mark.parametrize(("task_id", "engine"), REGISTERED_TASK_ENGINES)
def test_every_registered_task_passes_preflight_with_local_object_fixtures(
    task_id: str,
    engine: str,
) -> None:
    task = with_rigid_object_fixture(task_spec(task_id, engine))

    report = require_preflight(task, engine)

    assert report["status"] == "ok"
    assert report["incompatibilities"] == []


def test_preflight_report_imports_neither_engine_sdk() -> None:
    code = """
import json
import sys
import instinctlab_engine
from instinctlab.tasks import registry
selected = instinctlab_engine.adapter('isaacsim')
task = registry.spec(
    'Instinct-Velocity-Flat-G1',
    selected.robot_spec(registry.asset_id('Instinct-Velocity-Flat-G1')),
)
report = instinctlab_engine.preflight_report(task, 'isaacsim', selected_adapter=selected)
print(json.dumps({'status': report['status'], 'modules': sorted(sys.modules)}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["status"] == "ok"
    assert not any(
        name.startswith(("isaaclab", "mjlab", "mujoco", "omni", "pxr"))
        for name in payload["modules"]
    )


def test_missing_native_sensor_provider_fails_before_construction() -> None:
    task = task_spec("Instinct-Velocity-Flat-G1", "mjlab")
    task = replace(
        task,
        scene=replace(
            task.scene,
            native_sensors=(
                NativeSensorRef(
                    name="external_imu",
                    kind="provider_that_is_not_installed",
                    attach="pelvis",
                ),
            ),
        ),
    )

    report = preflight_report(task, "mjlab")

    assert report["status"] == "failed"
    assert any("has no provider" in problem for problem in report["incompatibilities"])
    with pytest.raises(PreflightError, match="external_imu"):
        require_preflight(task, "mjlab")


def test_missing_object_resources_are_all_reported_before_native_config(
    tmp_path: Path,
) -> None:
    task = task_spec("Instinct-Perceptive-HOI-Shadowing-G1-v0", "mjlab")
    objects = tuple(
        replace(
            obj,
            mesh=str(tmp_path / f"missing-{obj.name}.obj"),
            engine_meshes={},
        )
        for obj in task.scene.rigid_objects
    )
    task = replace(task, scene=replace(task.scene, rigid_objects=objects))

    report = preflight_report(task, "mjlab")

    object_failures = [
        problem
        for problem in report["incompatibilities"]
        if problem.startswith("rigid object")
    ]
    assert report["status"] == "failed"
    assert len(object_failures) == len(objects) == 6
    assert all(item["exists"] is False for item in report["rigid_objects"])


class _AssetOverrideAdapter:
    def __init__(self, wrapped, model_id: str):
        self._wrapped = wrapped
        self.name = wrapped.name
        self._model_id = model_id

    def contract_report(self, spec):
        return self._wrapped.contract_report(spec)

    def asset_conformance(self, asset_id):
        report = self._wrapped.asset_conformance(asset_id)
        return {
            **report,
            "actuator_model_ids": [self._model_id],
            "actuator_groups": [
                {**group, "model_id": self._model_id}
                for group in report["actuator_groups"]
            ],
        }

    def actuator_requirements(self, spec):
        return self._wrapped.actuator_requirements(spec)

    def rigid_object_conformance(self, ref):
        return self._wrapped.rigid_object_conformance(ref)


def test_actuator_capability_mismatch_fails_without_resolving_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from instinctlab_engine import preflight as preflight_module

    registry = ActuatorRegistry(load_entry_points=False)
    registry.register(
        engine="mjlab",
        model_id="external.insufficient.v1",
        config_factory="module_that_must_not_import:Factory",
        runtime_adapter="module_that_must_not_import:RUNTIME",
        capabilities={APPLIED_EFFORT},
    )
    monkeypatch.setattr(preflight_module, "ACTUATORS", registry)
    selected = instinctlab_engine.adapter("mjlab")
    task = task_spec("Instinct-Velocity-Flat-G1", "mjlab")

    report = preflight_report(
        task,
        "mjlab",
        selected_adapter=_AssetOverrideAdapter(
            selected,
            "external.insufficient.v1",
        ),
    )

    assert report["status"] == "failed"
    assert report["actuators"][0]["missing_capabilities"] == [
        "joint_position_command"
    ]
    assert "module_that_must_not_import" not in sys.modules


class _MixedActuatorAdapter:
    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.name = wrapped.name

    def contract_report(self, spec):
        return self._wrapped.contract_report(spec)

    def asset_conformance(self, asset_id):
        report = self._wrapped.asset_conformance(asset_id)
        groups = [
            {
                **group,
                "model_id": (
                    "external.position.v1"
                    if "waist_pitch_joint" in group["joint_names"]
                    else "external.other.v1"
                ),
            }
            for group in report["actuator_groups"]
        ]
        return {
            **report,
            "actuator_model_ids": [
                "external.other.v1",
                "external.position.v1",
            ],
            "actuator_groups": groups,
        }

    def actuator_requirements(self, spec):
        return {"action/joint_pos": [JOINT_POSITION_COMMAND]}

    def rigid_object_conformance(self, ref):
        return self._wrapped.rigid_object_conformance(ref)


def test_mixed_actuator_capabilities_apply_only_to_selected_joint_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from instinctlab_engine import preflight as preflight_module

    actuator_registry = ActuatorRegistry(load_entry_points=False)
    actuator_registry.register(
        engine="mjlab",
        model_id="external.position.v1",
        config_factory="position_factory_that_must_not_import:Factory",
        capabilities={JOINT_POSITION_COMMAND},
    )
    actuator_registry.register(
        engine="mjlab",
        model_id="external.other.v1",
        config_factory="other_factory_that_must_not_import:Factory",
        capabilities=(),
    )
    monkeypatch.setattr(preflight_module, "ACTUATORS", actuator_registry)
    selected = instinctlab_engine.adapter("mjlab")
    task = task_spec("Instinct-Velocity-Flat-G1", "mjlab")
    joint_pos = replace(
        task.mdp.actions["joint_pos"],
        target=EntityRef(
            "robot",
            joints=("waist_pitch_joint",),
            preserve_order=True,
        ),
    )
    task = replace(
        task,
        mdp=replace(task.mdp, actions={"joint_pos": joint_pos}),
    )

    report = preflight_report(
        task,
        "mjlab",
        selected_adapter=_MixedActuatorAdapter(selected),
    )

    assert report["status"] == "ok"
    selected_groups = [
        group
        for group in report["actuators"]
        if group["requested_capabilities"]
    ]
    assert [(group["group"], group["model_id"]) for group in selected_groups] == [
        ("4", "external.position.v1")
    ]
    assert selected_groups[0]["requested_capabilities"] == [
        JOINT_POSITION_COMMAND
    ]
    assert "position_factory_that_must_not_import" not in sys.modules
    assert "other_factory_that_must_not_import" not in sys.modules


def test_allow_nonclean_does_not_override_hard_component_incompatibility() -> None:
    task = task_spec("Instinct-Velocity-Flat-G1", "isaacsim")
    task = replace(
        task,
        scene=replace(
            task.scene,
            native_sensors=(
                NativeSensorRef(
                    name="missing_sensor",
                    kind="not_installed",
                    attach="pelvis",
                ),
            ),
        ),
    )

    report = preflight_report(task, "isaacsim", allow_nonclean=True)

    assert report["allow_nonclean"] is True
    assert report["status"] == "failed"
