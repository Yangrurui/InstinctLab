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
