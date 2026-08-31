"""Offline checks for the Perceptive fixed-bin reset probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from instinctlab.tasks import registry

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/probe_perceptive_reset.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("probe_perceptive_reset", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_task_resolves_the_selected_adapters_robot(monkeypatch) -> None:
    probe = _load_probe()
    robot = object()
    task = object()
    calls = []

    class Engine:
        def robot_spec(self, asset_id: str):
            calls.append(("robot_spec", asset_id))
            return robot

    monkeypatch.setattr(registry, "asset_id", lambda task_id: f"asset-for:{task_id}")

    def build(task_id, selected_robot):
        calls.append(("spec", task_id, selected_robot))
        return task

    monkeypatch.setattr(registry, "spec", build)

    assert probe._registered_task("perceptive-task", Engine()) is task
    assert calls == [
        ("robot_spec", "asset-for:perceptive-task"),
        ("spec", "perceptive-task", robot),
    ]


def test_reset_semantics_replace_every_mdp_reference_to_the_motion() -> None:
    from instinctlab_engine import adapter

    probe = _load_probe()
    task_id = "Instinct-Perceptive-Shadowing-G1-v0"
    original = probe._registered_task(task_id, adapter("isaacsim"))
    old_motion = original.scene.motion_references[0]

    patched = probe._task_with_reset_semantics(
        original,
        ensure_link_below_zero_ground=False,
        height_offset=0.0,
    )

    patched.validate()
    assert patched.scene.motion_references[0] is not old_motion
    assert all(
        value is not old_motion
        for term in patched.mdp.terms().values()
        for value in term.params.values()
    )
