"""The proprioceptive parkour G1 declaration against main's Instinct-Parkour-Target-Amp-G1.

Main is read through :mod:`tests.reference_main_parkour`, not from the working-tree copies in
``tasks/parkour/config/``. Those files are supposed to be untouched, but the reference is git's
``main:`` path so a silent local edit cannot pass as parity.

Drifts that are deliberate cross-engine compromises are listed in ``KNOWN_DRIFTS`` and asserted to
*remain* drifts — fixing one without updating the table fails here the same way a missing term would.
"""

from __future__ import annotations

import inspect
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
    "obs/joint_axis": (
        "implicit .* / unnamed SceneEntityCfg (PhysX BFS)",
        "named G1_29DOF_DFS_JOINT_NAMES + preserve_order",
        (
            "Policy joint_pos/joint_vel, last_action, the action vector, and AMP joints are a permutation "
            "of main. Same information, different layout — an MLP can learn either, but the optimization "
            "path and any checkpoint transfer change."
        ),
    ),
    "obs/base_lin_vel_frame": (
        "isaaclab.envs.mdp.base_lin_vel → root_lin_vel_b (COM)",
        "instinctlab.mdp.base_lin_vel → root_link_lin_vel_b",
        (
            "Critic and both AMP groups read link velocity. Same ω×R(−com_pos_b) shift as the reward "
            "drifts, now also in the value function and the discriminator."
        ),
    ),
    "amp/symmetric_augmentation": (
        "MotionReferenceManagerCfg BFS maps; AmassMotion.reset draws Bernoulli(0.5)",
        "clip sensor has no mirror",
        (
            "Main mirrors half of amp_reference trajectories on reset. The portable clip never does, "
            "so the discriminator's positive class is the unmirrored clip only."
        ),
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


def test_every_sim_param_parses_out_of_main() -> None:
    """A reference reader that answers ``None`` when it fails to parse pins nothing.

    ``sim_params`` used to probe for substrings in the unparsed source, and ``ast.unparse`` writes
    ``2 ** 29`` where the file writes ``2**29``. Every power of two therefore read as absent, so the
    ``gpu_collision_stack_size`` drift row reported main's value as the string ``"None"`` -- and
    passed, because a row only has to differ from ours.
    """
    params = main_ref.sim_params()
    assert set(params) == set(main_ref.SIM_PARAM_TARGETS)
    unparsed = sorted(name for name, value in params.items() if value is None)
    assert not unparsed, f"{unparsed} did not parse out of main's ParkourEnvCfg.__post_init__"
    assert params["gpu_collision_stack_size"] == 2**29
    assert params["gpu_max_rigid_patch_count"] == 10 * 2**15


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


def _declared_scale(term) -> float | None:
    return term.scale


def _declared_noise(term) -> tuple[float, float] | None:
    noise = term.noise
    if noise is None:
        return None
    return (noise.lo, noise.hi)


@pytest.mark.parametrize("group", ("policy", "critic", "amp_policy", "amp_reference"))
def test_observation_scales_match_main(task, group) -> None:
    """observation_scales() existed with no caller — extracted-but-unasserted."""
    declared = {name: _declared_scale(term) for name, term in task.mdp.observations[group].terms.items()}
    assert declared == main_ref.observation_scales(group)


@pytest.mark.parametrize("group", ("policy", "critic", "amp_policy", "amp_reference"))
def test_observation_noise_matches_main(task, group) -> None:
    declared = {name: _declared_noise(term) for name, term in task.mdp.observations[group].terms.items()}
    assert declared == main_ref.observation_noise(group)


@pytest.mark.parametrize("group", ("policy", "critic", "amp_policy", "amp_reference"))
def test_observation_history_and_clip_match_main(task, group) -> None:
    declared_hist = {name: term.history_length for name, term in task.mdp.observations[group].terms.items()}
    declared_clip = {name: term.clip for name, term in task.mdp.observations[group].terms.items()}
    assert declared_hist == main_ref.observation_history(group)
    assert declared_clip == main_ref.observation_clip(group)
    assert all(clip is None for clip in declared_clip.values())


def test_flatten_history_dim_is_isaac_default_true_on_both_sides() -> None:
    """We omit the field; Isaac Lab defaults True. Main sets True on most terms, omits on a few."""
    assert main_ref.isaac_observation_term_flatten_history_default() is True
    for group in ("policy", "critic", "amp_policy", "amp_reference"):
        flags = main_ref.observation_flatten_history(group)
        assert set(flags.values()) <= {True, None}, flags
        assert any(value is True for value in flags.values()) or group.startswith("amp")


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
    assert len(KNOWN_DRIFTS) >= 9
    assert "dataset_exhausted" in DELIBERATE_OMISSIONS
    assert "reward/undesired_contacts/threshold" not in KNOWN_DRIFTS
    assert "scene/robot/urdf" not in KNOWN_DRIFTS
    assert "scene/robot/spawn_z" not in KNOWN_DRIFTS
    assert "scene/robot/merge_fixed_joints" not in KNOWN_DRIFTS
    assert "scene/robot/actuators" not in KNOWN_DRIFTS
    assert "obs/joint_axis" in KNOWN_DRIFTS
    assert "obs/base_lin_vel_frame" in KNOWN_DRIFTS
    assert "amp/symmetric_augmentation" in KNOWN_DRIFTS


def test_documented_drifts_are_still_present(task) -> None:
    """Each KNOWN_DRIFTS entry must remain true — the table is not a graveyard."""
    from instinctlab.assets.unitree_g1.isaacsim import G1_29DOF_DFS_JOINT_NAMES, G1_29DOF_ISAAC_BFS_JOINT_NAMES
    from instinctlab.mdp.observations import base_lin_vel

    rewards = task.mdp.rewards["rewards"]
    assert rewards["track_lin_vel_xy_exp"].func.__name__ == "track_lin_vel_xy_exp"
    assert rewards["dont_wait"].func.__name__ == "dont_wait"
    assert rewards["undesired_contacts"].kind == "undesired_contacts"
    assert task.scene.volume_point("leg_volume_points").velocity == "attach_link"
    assert "dataset_exhausted" not in task.mdp.terminations
    assert tuple(task.robot.joint_names) == G1_29DOF_DFS_JOINT_NAMES
    assert G1_29DOF_DFS_JOINT_NAMES != G1_29DOF_ISAAC_BFS_JOINT_NAMES
    assert main_ref.observation_joint_names("policy", "joint_pos") is None
    assert main_ref.action_kwargs()["joint_names"] == [".*"]
    assert task.mdp.observations["critic"].terms["base_lin_vel"].func is base_lin_vel
    assert "root_link_lin_vel_b" in inspect.getsource(base_lin_vel)
    assert main_ref.observation_functions("critic")["base_lin_vel"] == "base_lin_vel"
    source = main_ref.motion_reference_source()
    assert (
        source.get("symmetric_augmentation_link_mapping") is not None or "symmetric_augmentation_link_mapping" in source
    )
    assert not hasattr(task.scene.motion_reference("motion_reference"), "symmetric_augmentation_joint_mapping")


def test_main_leaves_policy_and_action_joints_in_entity_order(task) -> None:
    """Main's implicit `.*` is PhysX BFS; we name DFS. This is KNOWN_DRIFTS['obs/joint_axis']."""
    from instinctlab.assets.unitree_g1.isaacsim import G1_29DOF_DFS_JOINT_NAMES, G1_29DOF_ISAAC_BFS_JOINT_NAMES

    assert G1_29DOF_DFS_JOINT_NAMES != G1_29DOF_ISAAC_BFS_JOINT_NAMES
    assert G1_29DOF_DFS_JOINT_NAMES[0] == "waist_pitch_joint"
    assert G1_29DOF_ISAAC_BFS_JOINT_NAMES[0] == "left_shoulder_pitch_joint"
    for group, term in (
        ("policy", "joint_pos"),
        ("policy", "joint_vel"),
        ("critic", "joint_pos"),
        ("critic", "joint_vel"),
    ):
        assert main_ref.observation_joint_names(group, term) is None, f"{group}/{term}"
    assert main_ref.observation_joint_names("amp_policy", "joint_pos_rel") is None
    assert main_ref.observation_preserve_order("amp_policy", "joint_pos_rel") is True
    action = main_ref.action_kwargs()
    assert action["joint_names"] == [".*"]
    assert action.get("preserve_order") is None
    assert action.get("use_default_offset") is True
    canonical = tuple(task.robot.joint_names)
    assert canonical == G1_29DOF_DFS_JOINT_NAMES
    for group, name in (("policy", "joint_pos"), ("amp_policy", "joint_pos_rel")):
        ref = task.mdp.observations[group].terms[name].params["asset_cfg"]
        assert tuple(ref.joints) == canonical
        assert ref.preserve_order is True
    assert tuple(task.mdp.actions["joint_pos"].target.joints) == canonical


def test_critic_and_amp_lin_vel_use_the_hub_link_spelling(task) -> None:
    """Same COM→link shift as the reward drifts, now in critic and AMP inputs."""
    from instinctlab.mdp.amp import base_lin_vel_from_reference
    from instinctlab.mdp.observations import base_lin_vel

    assert task.mdp.observations["critic"].terms["base_lin_vel"].func is base_lin_vel
    assert task.mdp.observations["amp_policy"].terms["base_lin_vel"].func is base_lin_vel
    assert task.mdp.observations["amp_reference"].terms["base_lin_vel"].func is base_lin_vel_from_reference
    assert "root_link_lin_vel_b" in inspect.getsource(base_lin_vel)
    assert main_ref.observation_functions("critic")["base_lin_vel"] == "base_lin_vel"
    assert main_ref.observation_functions("amp_policy")["base_lin_vel"] == "base_lin_vel"
    assert main_ref.observation_functions("amp_reference")["base_lin_vel"] == "base_lin_vel_reference_as_state"


def test_main_amp_mirrors_half_the_reference_and_we_do_not(task) -> None:
    source = main_ref.motion_reference_source()
    assert source["symmetric_augmentation_link_mapping"] == [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]
    assert "symmetric_augmentation_joint_mapping" in source
    assert "symmetric_augmentation_joint_reverse_buf" in source
    amass = source["amass"]
    assert "filtered_motion_selection_filepath" in amass
    assert "parkour_motion_without_run.yaml" in str(amass["filtered_motion_selection_filepath"])
    clip = task.scene.motion_reference("motion_reference")
    assert clip.clip.endswith("parkour_motion_without_run_retargetted.npz")
    assert not hasattr(clip, "symmetric_augmentation_joint_mapping")


def test_shipped_motion_yaml_selects_the_same_npz_as_our_clip(task) -> None:
    """Measured: main's yaml filter is not a multi-clip walk on the shipped dataset."""
    import yaml

    if not main_ref.SHIPPED_MOTION_YAML.is_file():
        pytest.skip(f"shipped yaml missing at {main_ref.SHIPPED_MOTION_YAML}")
    data = yaml.safe_load(main_ref.SHIPPED_MOTION_YAML.read_text())
    files = data["selected_files"]
    assert files == ["parkour_motion_without_run_retargetted.npz"]
    assert task.scene.motion_reference("motion_reference").clip.endswith(files[0])
    assert main_ref.SHIPPED_MOTION_NPZ.is_file()
    import numpy as np

    raw = np.load(main_ref.SHIPPED_MOTION_NPZ, mmap_mode="r", allow_pickle=True)
    assert int(raw["joint_pos"].shape[0]) == 18982
    assert float(np.asarray(raw["framerate"]).item()) == pytest.approx(50.0)


def test_depth_delay_params_match_main(task) -> None:
    ours = task.mdp.observations["policy"].terms["depth_image"].params
    theirs = main_ref.delayed_depth_params()
    assert ours["history_skip_frames"] == theirs["history_skip_frames"] == 5
    assert ours["num_output_frames"] == theirs["num_output_frames"] == 8
    assert tuple(ours["delayed_frame_ranges"]) == tuple(theirs["delayed_frame_ranges"]) == (0, 1)
    assert ours["history_length"] == 37
    assert theirs.get("data_type") == "distance_to_image_plane_noised_history"


def test_camera_ray_alignment_config_differs_but_grouped_camera_ignores_it(task) -> None:
    """Main writes yaw; we declare base. GroupedRayCasterCamera forces attach_yaw_only=False."""
    assert main_ref.camera_ray_alignment() == "yaw"
    assert task.scene.ray_caster("camera").ray_alignment == "base"
    cfg = Path(
        "/root/InstinctLab/source/instinctlab/instinctlab/sensors/grouped_ray_caster/grouped_ray_caster_camera_cfg.py"
    )
    assert "self.attach_yaw_only = False" in cfg.read_text()


def test_wasabi_defaults_match_our_explicit_amp_keys(our_agent, main_agent) -> None:
    import inspect

    from instinct_rl.algorithms.wasabi import WasabiAlgoMixin

    params = inspect.signature(WasabiAlgoMixin.__init__).parameters
    assert params["actor_state_key"].default == "amp_policy"
    assert params["reference_state_key"].default == "amp_reference"
    assert our_agent["algorithm.actor_state_key"] == "amp_policy"
    assert our_agent["algorithm.reference_state_key"] == "amp_reference"
    assert "algorithm.actor_state_key" not in main_agent
    assert our_agent.get("policy.encoder_configs.depth_encoder.takeout_input_components", True) is True


def test_runner_has_no_empirical_normalizer_on_either_side(our_agent, main_agent) -> None:
    assert main_agent["empirical_normalization"] is False
    assert not any(key.startswith("normalizers.") for key in our_agent)
    assert not any(key.startswith("normalizers.") for key in main_agent)


def test_train_scripts_agree_on_seed_num_envs_and_tf32() -> None:
    """CLI defaults and the seed hand-off. Resume load is main-only and unused when resume=False."""
    theirs = main_ref.train_script_facts()
    ours = Path("/root/InstinctLab/scripts/train.py").read_text()
    adapter = Path("/root/InstinctLab/source/instinctlab/instinctlab/engines/isaacsim/adapter.py").read_text()
    assert theirs["sets_env_seed_from_agent"] is True
    assert "compiled.env_cfg.seed = agent_cfg.seed" in ours
    assert theirs["num_envs_default"] is None
    assert "default=4096" in ours
    assert theirs["seed_default"] is None
    assert theirs["sets_tf32"] is True
    assert "allow_tf32 = True" in adapter
    assert theirs["calls_runner_load"] is True
    assert "runner.load(" not in ours
    assert theirs["init_at_random_ep_len"] is True
    assert "init_at_random_ep_len" in ours
    assert theirs["wrapper"] is True


def test_wrapper_setdefault_step_does_not_change_plain_ppo_rewards() -> None:
    """Isaac ManagerBasedRLEnv never writes extras['step']. AMP writes into it; plain PPO does not."""
    import torch

    extras = {"log": {}}
    extras.setdefault("step", {})
    extras.setdefault("episode", {})
    for key, value in {}.items():
        extras["step"][key] = value
    assert extras["step"] == {}
    reward = torch.zeros(4)
    assert reward.unsqueeze(1).shape == (4, 1)
    missing: dict = {}
    with pytest.raises(KeyError):
        missing["step"]["discriminator_reward"] = torch.ones(2, 1)
    missing.setdefault("step", {})
    missing["step"]["discriminator_reward"] = torch.ones(2, 1)
    assert "discriminator_reward" in missing["step"]
    assert main_ref.wrapper_sets_missing_step_dict() is True
    assert main_ref.main_wrapper_sets_missing_step_dict() is True


def test_single_reward_group_keeps_num_rewards_one() -> None:
    assert main_ref.uses_multi_reward_cfg()
    assert main_ref.uses_instinct_rl_env()
    from instinctlab.tasks.parkour.config.g1 import parkour_target_g1

    task = parkour_target_g1()
    assert list(task.mdp.rewards) == ["rewards"]


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
