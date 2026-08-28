from __future__ import annotations

import importlib.util
from pathlib import Path

from instinctlab.tasks import registry


def _probe_module():
    path = Path(__file__).parents[1] / "scripts" / "probe_perceptive_reset.py"
    spec = importlib.util.spec_from_file_location("probe_perceptive_reset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reset_probe_changes_only_the_copied_motion_reference() -> None:
    module = _probe_module()
    original = registry.spec("Instinct-Perceptive-Shadowing-G1-v0")
    changed = module._task_with_reset_semantics(
        original,
        ensure_link_below_zero_ground=True,
        height_offset=0.1,
    )

    original_motion = original.scene.motion_references[0]
    changed_motion = changed.scene.motion_references[0]
    assert original_motion.ensure_link_below_zero_ground is True
    assert original_motion.motion_start_height_offset == 0.1
    assert original_motion.for_engine("isaacsim").ensure_link_below_zero_ground is False
    assert original_motion.for_engine("isaacsim").motion_start_height_offset == 0.0
    assert changed_motion.ensure_link_below_zero_ground is True
    assert changed_motion.motion_start_height_offset == 0.1
    assert changed_motion.engine_overrides == {}
    assert changed.mdp.terminations["dataset_exhausted"].params["sensor"] is changed_motion
    assert original.mdp.terminations["dataset_exhausted"].params["sensor"] is original_motion


def test_reset_probe_disables_weight_updates_for_fixed_bin_sampling() -> None:
    module = _probe_module()
    original = registry.spec("Instinct-Perceptive-Shadowing-G1-v0")
    changed = module._without_adaptive_sampling(original)

    assert "beyond_adaptive_sampling" in original.mdp.curriculum
    assert "bin_fail_counter_smoothing" in original.mdp.events
    assert changed.mdp.curriculum == {}
    assert "bin_fail_counter_smoothing" not in changed.mdp.events
    assert changed.mdp.events["reset_robot"] is original.mdp.events["reset_robot"]


def test_reset_probe_can_disable_only_the_early_contact_termination() -> None:
    module = _probe_module()
    original = registry.spec("Instinct-Perceptive-Shadowing-G1-v0")
    changed = module._without_illegal_reset_contact(original)

    assert "illegal_reset_contact" in original.mdp.terminations
    assert "illegal_reset_contact" not in changed.mdp.terminations
    assert changed.mdp.terminations["base_pos_too_far"] is original.mdp.terminations["base_pos_too_far"]


def test_reset_probe_can_override_only_isaac_self_collision() -> None:
    module = _probe_module()
    original = registry.spec("Instinct-Perceptive-Shadowing-G1-v0")
    changed = module._with_isaac_self_collision(original, False)

    assert "self_collision" not in original.sim.profiles["isaacsim"]
    assert changed.sim.profiles["isaacsim"]["self_collision"] is False
    assert changed.sim.profiles["mjlab"] == original.sim.profiles["mjlab"]
