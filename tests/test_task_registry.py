"""The engine-neutral task registry is the only production registration path."""

from pathlib import Path

from instinctlab.tasks import registry


TASK_ROOT = Path(__file__).resolve().parents[1] / "source/instinctlab/instinctlab/tasks"


def test_task_sources_do_not_register_gym_environments() -> None:
    registrations = [
        path.relative_to(TASK_ROOT)
        for path in TASK_ROOT.rglob("*.py")
        if "gym.register(" in path.read_text()
    ]
    assert registrations == []


def test_every_registered_factory_returns_its_own_task_id() -> None:
    for task_id in registry.ids():
        assert registry.spec(task_id).task_id == task_id


def test_shadowing_tasks_use_the_reference_file_layout() -> None:
    shadowing_root = TASK_ROOT / "shadowing"
    assert not (shadowing_root / "config.py").exists()

    shadowing_factories = [
        path
        for task_id, path in registry.TASKS.items()
        if "Shadowing" in task_id or "Mimic" in task_id or "Vae" in task_id
    ]
    assert all(".config.g1." in path for path in shadowing_factories)
