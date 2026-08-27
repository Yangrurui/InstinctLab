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


def test_registered_tasks_do_not_use_engine_extras() -> None:
    for task_id in registry.ids():
        assert not registry.spec(task_id).engine_extras


def test_shadowing_tasks_use_the_reference_file_layout() -> None:
    shadowing_root = TASK_ROOT / "shadowing"
    assert not (shadowing_root / "config.py").exists()

    shadowing_factories = [
        path
        for task_id, path in registry.TASKS.items()
        if "Shadowing" in task_id or "Mimic" in task_id or "Vae" in task_id
    ]
    assert all(".config.g1." in path for path in shadowing_factories)


def test_task_configs_use_env_cfg_classes_instead_of_make_task() -> None:
    sources = [path.read_text() for path in TASK_ROOT.rglob("*.py")]
    assert not [source for source in sources if "def make_task(" in source]
    assert (TASK_ROOT / "parkour/config/parkour_env_cfg.py").is_file()


def test_locomotion_and_parkour_can_change_one_reward() -> None:
    from instinctlab.tasks.locomotion.config.g1.flat_env_cfg import G1FlatEnvCfg
    from instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import (
        G1ParkourEnvCfg,
    )

    flat = G1FlatEnvCfg()
    flat.rewards.action_rate_l2 = flat.rewards.action_rate_l2.replace(weight=-0.2)
    flat_task = flat.to_task_spec("Tuned-Flat")
    fresh_flat = G1FlatEnvCfg().to_task_spec("Fresh-Flat")
    assert flat_task.mdp.rewards["rewards"]["action_rate_l2"].weight == -0.2
    assert fresh_flat.mdp.rewards["rewards"]["action_rate_l2"].weight == -0.05

    parkour = G1ParkourEnvCfg()
    parkour.rewards.action_rate_l2 = parkour.rewards.action_rate_l2.replace(weight=-0.2)
    parkour_task = parkour.to_task_spec("Tuned-Parkour")
    fresh_parkour = G1ParkourEnvCfg().to_task_spec("Fresh-Parkour")
    assert parkour_task.mdp.rewards["rewards"]["action_rate_l2"].weight == -0.2
    assert fresh_parkour.mdp.rewards["rewards"]["action_rate_l2"].weight == -0.005
