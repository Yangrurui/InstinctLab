"""The engine-neutral task registry is the only production registration path."""

from collections.abc import Mapping
from pathlib import Path

from instinctlab_engine.spec import EntityRef
from instinctlab.tasks import registry
from tests.task_specs import task_spec

TASK_ROOT = Path(__file__).resolve().parents[1] / "source/instinctlab/instinctlab/tasks"

PLAY_CHECKPOINT_PAIRS = {
    "Instinct-Shadowing-WholeBody-Plane-G1-Play-v0": "Instinct-Shadowing-WholeBody-Plane-G1-v0",
    "Instinct-Perceptive-Shadowing-G1-Play-v0": "Instinct-Perceptive-Shadowing-G1-v0",
    "Instinct-Perceptive-Shadowing-G1-OneMotion-Play-v0": (
        "Instinct-Perceptive-Shadowing-G1-OneMotion-v0"
    ),
    "Instinct-Perceptive-Vae-G1-Play-v0": "Instinct-Perceptive-Vae-G1-v0",
    "Instinct-Perceptive-HOI-Shadowing-G1-Play-v0": "Instinct-Perceptive-HOI-Shadowing-G1-v0",
    "Instinct-BeyondMimic-Plane-G1-Play-v0": "Instinct-BeyondMimic-Plane-G1-v0",
}


def test_task_sources_do_not_register_gym_environments() -> None:
    registrations = [
        path.relative_to(TASK_ROOT)
        for path in TASK_ROOT.rglob("*.py")
        if "gym.register(" in path.read_text()
    ]
    assert registrations == []


def test_every_registered_factory_returns_its_own_task_id() -> None:
    for task_id in registry.ids():
        assert task_spec(task_id).task_id == task_id


def test_play_checkpoint_pairs_are_explicit_and_policy_compatible() -> None:
    assert registry.PLAY_CHECKPOINT_TASKS == PLAY_CHECKPOINT_PAIRS
    for play_id, train_id in PLAY_CHECKPOINT_PAIRS.items():
        assert registry.checkpoint_task_id(play_id) == train_id
        assert registry.checkpoint_task_id(train_id) == train_id

        train = task_spec(train_id)
        play = task_spec(play_id)
        assert train.robot.asset_id == play.robot.asset_id
        assert train.robot.schema_version == play.robot.schema_version
        assert train.robot.joint_names == play.robot.joint_names
        assert train.mdp.actions == play.mdp.actions

        train_policy = train.mdp.observations["policy"]
        play_policy = play.mdp.observations["policy"]
        assert tuple(train_policy.terms) == tuple(play_policy.terms)
        assert train_policy.history_length == play_policy.history_length
        for term_name in train_policy.terms:
            train_term = train_policy.terms[term_name]
            play_term = play_policy.terms[term_name]
            assert (
                train_term.func,
                train_term.kind,
                train_term.target,
                train_term.scale,
                train_term.clip,
                train_term.history_length,
            ) == (
                play_term.func,
                play_term.kind,
                play_term.target,
                play_term.scale,
                play_term.clip,
                play_term.history_length,
            )


def test_registered_tasks_do_not_use_engine_extras() -> None:
    for task_id in registry.ids():
        assert not task_spec(task_id).engine_extras


def test_every_registered_policy_joint_axis_is_the_robot_canonical_order() -> None:
    """Discover every task instead of maintaining family-specific joint-order lists."""
    joint_vector_functions = {
        "instinctlab.tasks.parkour.mdp.observations.joint_pos_rel",
        "instinctlab.tasks.parkour.mdp.observations.joint_vel",
        "instinctlab.tasks.parkour.mdp.observations.joint_vel_rel",
        "instinctlab.tasks.parkour.mdp.amp.joint_pos_rel_from_reference",
        "instinctlab.tasks.parkour.mdp.amp.joint_vel_rel_from_reference",
    }
    for task_id in registry.ids():
        task = task_spec(task_id)
        canonical = tuple(task.robot.joint_names)

        for action_name, action in task.mdp.actions.items():
            if action.kind != "joint_position":
                continue
            assert action.target is not None, (task_id, action_name)
            assert action.target.joints == canonical, (task_id, action_name)
            assert action.target.preserve_order is True, (task_id, action_name)
            scale = action.params.get("scale")
            if isinstance(scale, Mapping):
                assert tuple(scale) == canonical, (task_id, action_name)

        for group_name, group in task.mdp.observations.items():
            for term_name, term in group.terms.items():
                func = term.func
                if func is None:
                    continue
                function_name = f"{func.__module__}.{func.__qualname__}"
                if function_name not in joint_vector_functions:
                    continue
                selector = term.params.get("asset_cfg")
                assert isinstance(selector, EntityRef), (task_id, group_name, term_name)
                assert selector.joints == canonical, (task_id, group_name, term_name)
                assert selector.preserve_order is True, (task_id, group_name, term_name)

        for motion in task.scene.motion_references:
            selected = set(motion.joints)
            assert tuple(motion.joints) == tuple(name for name in canonical if name in selected), task_id


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
    from instinctlab_engine_mjlab.assets import robot_spec
    from instinctlab.tasks import registry
    from instinctlab.tasks.locomotion.config.g1.flat_env_cfg import G1LocomotionFlatEnvCfg
    from instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import (
        G1ParkourEnvCfg,
    )

    flat_robot = robot_spec(registry.asset_id("Instinct-Velocity-Flat-G1"))
    flat = G1LocomotionFlatEnvCfg(flat_robot)
    flat.rewards["rewards"]["action_rate_l2"] = flat.rewards["rewards"]["action_rate_l2"].replace(weight=-0.2)
    fresh_flat = G1LocomotionFlatEnvCfg(flat_robot)
    assert flat.rewards["rewards"]["action_rate_l2"].weight == -0.2
    assert fresh_flat.rewards["rewards"]["action_rate_l2"].weight == -0.05

    parkour_robot = robot_spec(registry.asset_id("Instinct-Parkour-Target-G1"))
    parkour = G1ParkourEnvCfg(parkour_robot)
    parkour.rewards["rewards"]["action_rate_l2"] = parkour.rewards["rewards"]["action_rate_l2"].replace(weight=-0.2)
    fresh_parkour = G1ParkourEnvCfg(parkour_robot)
    assert parkour.rewards["rewards"]["action_rate_l2"].weight == -0.2
    assert fresh_parkour.rewards["rewards"]["action_rate_l2"].weight == -0.005
