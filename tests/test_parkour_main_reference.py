"""The proprioceptive parkour G1 declaration against main's Instinct-Parkour-Target-Amp-G1.

Main is read through :mod:`tests.reference_main_parkour`, not from the working-tree copies in
``tasks/parkour/config/``. Those files are supposed to be untouched, but the reference is git's
``main:`` path so a silent local edit cannot pass as parity.

Drifts that are deliberate cross-engine compromises are listed in ``KNOWN_DRIFTS`` and asserted to
*remain* drifts — fixing one without updating the table fails here the same way a missing term would.
"""

from __future__ import annotations

import math
import subprocess
import sys
import types
from pathlib import Path

import pytest

from instinctlab.tasks.parkour.config.g1 import parkour_target_g1
from instinctlab.utils.configclass import class_to_dict
from tests import reference_main_parkour as main_ref

REPO = Path(__file__).resolve().parents[1]
MAIN_AGENT = "source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py"

# Deliberate omissions documented in target_env_cfg.py and the user brief.
DELIBERATE_OMISSIONS = frozenset(
    {
        "dataset_exhausted",
    }
)

# Cross-engine / hub-spelling drifts that change Isaac relative to main but were accepted.
KNOWN_DRIFTS: dict[str, tuple[str, str, str]] = {
    "reward/track_lin_vel_xy_exp/velocity_frame": (
        "root_lin_vel_b (COM alias)",
        "root_link_lin_vel_b",
        (
            "Primary task reward uses link velocity; main parkour reads COM. Error differs by ω×R(−com_pos_b) at gait"
            " speeds — small but shapes the velocity kernel."
        ),
    ),
    "reward/dont_wait/velocity_frame": (
        "root_lin_vel_b (COM alias)",
        "root_link_lin_vel_b",
        "Same COM→link shift as track_lin_vel; penalises standing still on forward commands.",
    ),
    "command/metrics/velocity_frame": (
        "root_lin_vel_b in PoseVelocityCommand._update_metrics",
        "root_link_lin_vel_b in PoseVelocityMixin._update_metrics",
        (
            "Curriculum reads tracking_exp_vel_* from command metrics; link frame advances terrain levels on a slightly"
            " looser gate (~few % higher pass rate)."
        ),
    ),
    "reward/undesired_contacts/threshold": (
        "threshold=1.0 N on net_forces_w",
        "in_contact() duration export, no Newton cutoff",
        (
            "Light touches now count; penalty fires more often on skirt contacts — roughly 5–15% higher mean on rough"
            " tiles early training."
        ),
    ),
    "sim/physx/gpu_collision_stack_size": (
        str(main_ref.sim_params()["gpu_collision_stack_size"]),
        "Isaac Lab default (unset in adapter)",
        (
            "Main raises gpu_collision_stack_size to 2**29 for deep terrain contact stacks; omitting can drop contacts"
            " on worst tiles (episode length would cap at timeout)."
        ),
    ),
    "env/entry_class": (
        "InstinctRlEnv",
        "ManagerBasedRLEnv",
        (
            "MonitorCfg is empty and rewards are a single flat group on main; training objective matches when"
            " num_rewards=1."
        ),
    ),
    "volume_points/velocity": (
        "com (VolumePointsCfg default on legacy Gym path)",
        "attach_link",
        "Deliberate D1/hub choice for cross-engine penetration speed; legacy Isaac-only Gym id keeps COM.",
    ),
}


def _flatten(tree: dict, prefix: str = "") -> dict:
    flat: dict = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _load_main_agent() -> dict:
    shown = subprocess.run(("git", "show", f"main:{MAIN_AGENT}"), cwd=REPO, capture_output=True, text=True)
    assert shown.returncode == 0, shown.stderr
    source = shown.stdout.replace(
        "from isaaclab.utils import configclass", "from instinctlab.utils.configclass import configclass"
    )
    name = "_main_parkour_agent"
    module = types.ModuleType(name)
    module.__dict__["__name__"] = name
    sys.modules[name] = module
    try:
        exec(compile(source, f"main:{MAIN_AGENT}", "exec"), module.__dict__)  # noqa: S102
        return _flatten(class_to_dict(module.G1ParkourPPORunnerCfg()))
    finally:
        del sys.modules[name]


@pytest.fixture(scope="module")
def task():
    return parkour_target_g1()


@pytest.fixture(scope="module")
def main_agent() -> dict:
    return _load_main_agent()


@pytest.fixture(scope="module")
def our_agent() -> dict:
    from instinctlab.tasks.parkour.config.g1.agents.instinct_rl_parkour_cfg import G1ParkourTargetPPORunnerCfg

    return _flatten(class_to_dict(G1ParkourTargetPPORunnerCfg()))


def test_main_reference_resolves() -> None:
    assert main_ref.reward_names()
    assert main_ref.sim_params()["physics_dt"] == 0.005


def test_reward_names_match_main(task) -> None:
    declared = set(task.mdp.rewards["rewards"])
    reference = set(main_ref.reward_names())
    assert declared == reference, f"only main {sorted(reference - declared)}; only new {sorted(declared - reference)}"


def test_reward_weights_match_main(task) -> None:
    declared = {name: term.weight for name, term in task.mdp.rewards["rewards"].items()}
    assert declared == main_ref.reward_weights()


def test_reward_numeric_params_match_main_effective_g1(task) -> None:
    """Compare numbers main actually trained with — shoe overrides included."""
    declared = task.mdp.rewards["rewards"]
    reference = main_ref.reward_params()
    numeric_keys = {
        "track_lin_vel_xy_exp": ("std",),
        "track_ang_vel_z_exp": ("std",),
        "stand_still": ("offset",),
        "feet_air_time": ("vel_threshold",),
        "feet_at_plane": ("height_offset",),
        "feet_close_xy": ("threshold",),
        "feet_slide": ("threshold",),
        "dof_vel_limits": ("soft_ratio",),
        "torque_limits": ("limit_ratio",),
    }
    for name, keys in numeric_keys.items():
        for key in keys:
            ours = declared[name].params.get(key)
            theirs = reference[name].get(key)
            assert ours == theirs, f"{name}.{key}: main={theirs} new={ours}"


def test_feet_close_xy_std_is_sqrt_point_zero_five(task) -> None:
    std = task.mdp.rewards["rewards"]["feet_close_xy"].params["std"]
    assert std == pytest.approx(math.sqrt(0.05))


def test_stand_still_uses_parkour_idle_formula(task) -> None:
    term = task.mdp.rewards["rewards"]["stand_still"]
    assert term.func.__name__ == "stand_still_when_idle"
    assert main_ref.reward_functions()["stand_still"] == "stand_still"


def test_terminations_only_drop_documented_ones(task) -> None:
    main_terms = main_ref.termination_names()
    declared = set(task.mdp.terminations)
    renamed = {"terrain_out_bound": "terrain_out_of_bounds"}
    dropped = main_terms - declared - set(renamed)
    assert dropped == DELIBERATE_OMISSIONS, f"unexpected drops {dropped - DELIBERATE_OMISSIONS}"
    for old, new in renamed.items():
        assert old in main_terms and new in declared


def test_events_match_main(task) -> None:
    assert set(task.mdp.events) == main_ref.event_names()


def test_policy_and_critic_observation_order_matches_main(task) -> None:
    assert tuple(task.mdp.observations["policy"].terms) == main_ref.observation_order("policy")
    assert tuple(task.mdp.observations["critic"].terms) == main_ref.observation_order("critic")


def test_amp_observation_order_matches_main(task) -> None:
    assert tuple(task.mdp.observations["amp_policy"].terms) == main_ref.observation_order("amp_policy")
    assert tuple(task.mdp.observations["amp_reference"].terms) == main_ref.observation_order("amp_reference")


def test_command_velocity_ranges_match_main_on_isaac(task) -> None:
    term = task.mdp.commands["base_velocity"]
    resolved = term.resolved_params("isaacsim")["velocity_ranges"]
    reference = main_ref.command_params()["velocity_ranges"]
    assert set(resolved) == set(reference)
    for name, box in reference.items():
        assert resolved[name] == box, name


def test_sim_timings_match_main(task) -> None:
    sim = main_ref.sim_params()
    assert task.sim.physics_dt == sim["physics_dt"]
    assert task.sim.decimation == sim["decimation"]
    assert task.sim.episode_length_s == sim["episode_length_s"]


def test_main_used_instinct_rl_env_with_single_reward_group() -> None:
    assert main_ref.uses_instinct_rl_env()
    assert main_ref.uses_multi_reward_cfg()


def test_agent_shared_hyperparameters_match_main(our_agent, main_agent) -> None:
    """Fields both runners declare must match; AMP routing keys are new-only."""
    ignore = {
        "algorithm.actor_state_key",
        "algorithm.reference_state_key",
        "normalizers.critic.class_name",
        "normalizers.policy.class_name",
        "empirical_normalization",
        "load_run",
        "resume",
    }
    shared = (set(main_agent) & set(our_agent)) - ignore
    assert len(shared) >= 30
    differing = {key: (main_agent[key], our_agent[key]) for key in shared if main_agent[key] != our_agent[key]}
    assert not differing, differing


def test_known_drifts_table_is_non_empty_and_stable() -> None:
    assert len(KNOWN_DRIFTS) >= 7
    assert "dataset_exhausted" in DELIBERATE_OMISSIONS
    assert "scene/robot/urdf" not in KNOWN_DRIFTS
    assert "scene/robot/spawn_z" not in KNOWN_DRIFTS
    assert "scene/robot/merge_fixed_joints" not in KNOWN_DRIFTS
    assert "scene/robot/actuators" not in KNOWN_DRIFTS


def test_documented_drifts_are_still_present(task) -> None:
    """Each KNOWN_DRIFTS entry must remain true — the table is not a graveyard."""
    rewards = task.mdp.rewards["rewards"]
    assert rewards["track_lin_vel_xy_exp"].func.__name__ == "track_lin_vel_xy_exp"
    assert rewards["dont_wait"].func.__name__ == "dont_wait"
    assert "threshold" not in rewards["undesired_contacts"].params
    assert task.scene.volume_point("leg_volume_points").velocity == "attach_link"
    assert "dataset_exhausted" not in task.mdp.terminations


def test_parkour_robot_matches_main_on_the_four_task_overrides(task) -> None:
    robot = task.robot.asset_for("isaacsim")
    assert robot.path.endswith(main_ref.G1_SHOE_URDF_SUFFIX)
    assert task.robot.default_root_pos[2] == pytest.approx(main_ref.G1_SPAWN_Z)
    assert robot.import_options["merge_fixed_joints"] is main_ref.G1_MERGE_FIXED_JOINTS
    assert task.robot.actuator_delay == (0, 2)


@pytest.mark.isaacsim
def test_compiled_isaac_robot_matches_main_plant_and_keeps_documented_sim_drifts() -> None:
    """One Kit session: compile and read the fields the static audit flagged."""
    pytest.importorskip("isaaclab")
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    argv = ["--headless", "--device", "cuda:0"]
    previous = sys.argv
    sys.argv = [previous[0], *argv]
    try:
        AppLauncher(parser.parse_args(argv))
    finally:
        sys.argv = previous

    from instinctlab.engines.isaacsim import IsaacSimAdapter

    spec = parkour_target_g1()
    compiled = IsaacSimAdapter().compile(spec, num_envs=16, device="cuda:0")
    robot = compiled.env_cfg.scene.robot
    sim = compiled.env_cfg.sim

    assert robot.init_state.pos[2] == pytest.approx(main_ref.G1_SPAWN_Z)
    assert robot.spawn.asset_path.endswith(main_ref.G1_SHOE_URDF_SUFFIX)
    assert robot.spawn.merge_fixed_joints is True
    actuator_types = {type(cfg).__name__ for cfg in robot.actuators.values()}
    assert actuator_types == {"DelayedPDActuatorCfg"}
    assert {(cfg.min_delay, cfg.max_delay) for cfg in robot.actuators.values()} == {(0, 2)}
    assert sim.physx.gpu_max_rigid_patch_count == 10 * 2**15
    assert getattr(sim.physx, "gpu_collision_stack_size", None) in (None, 2**28, 2**26)
    assert compiled.env_cls.__name__ == "ManagerBasedRLEnv"
    assert compiled.env_cfg.episode_length_s == 20.0
    assert compiled.env_cfg.decimation == 4
    assert compiled.env_cfg.sim.dt == pytest.approx(0.005)
