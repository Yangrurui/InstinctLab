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

# Deliberate omissions documented in g1_parkour_target_amp_cfg.py and the user brief.
DELIBERATE_OMISSIONS = frozenset()

# Cross-engine / hub-spelling drifts that change Isaac relative to main but were accepted.
KNOWN_DRIFTS: dict[str, tuple[str, str, str]] = {
    "reward/volume_points_penetration/weight": (
        "weight=-4.0",
        "weight=-8.0",
        (
            "Requested follow-up experiment doubles the virtual-terrain penetration penalty for the "
            "next MJLab run. The shared declaration also records this for future Isaac runs, while the "
            "already-running baselines retain the -4.0 configuration loaded at process construction."
        ),
    ),
    "reward/track_lin_vel_xy_exp/velocity_frame": (
        "root_lin_vel_b (COM alias)",
        "root_link_lin_vel_b",
        (
            "Primary task reward uses link velocity; main parkour reads COM, and InstinctMJ reads link, so main is"
            " the odd one of the three. RULED OUT AS THE GAP'S CAUSE, by training rather than by probe: flipping"
            " the whole frame cluster to main's COM spelling (both rewards, the command metrics, and the obs) and"
            " retraining at 256/seed42/700 left 85% of this term's shortfall in place (-0.0345 -> -0.0292) and"
            " moved measured tracking error 0.2682 -> 0.2634 against main's 0.2435. Return went 0.847 -> 0.905,"
            " inside the 12% seed noise floor, so even that is not a claim. We track worse than main for some"
            " other reason; the frame is a real difference that costs little."
        ),
    ),
    "reward/dont_wait/velocity_frame": (
        "root_lin_vel_b (COM alias)",
        "root_link_lin_vel_b",
        (
            "Same COM→link shift as track_lin_vel, on the term that penalises standing still under a forward"
            " command; same 0.067 m/s difference in the quantity it thresholds. Carried 14-23% of the Isaac-vs-main"
            " gap, and the same training A/B closed only a third of it (-0.0161 -> -0.0112). Ruled out with"
            " track_lin_vel above."
        ),
    ),
    "command/metrics/velocity_frame": (
        "root_lin_vel_b in PoseVelocityCommand._update_metrics",
        "root_link_lin_vel_b in PoseVelocityMixin._update_metrics",
        (
            "Curriculum reads tracking_exp_vel_* from command metrics, so the frame reaches terrain progression."
            " Unmeasured at this scale: terrain_levels reads 0.042 for us against 0.068 on main, but those are"
            " mean levels on a ten-level ladder, so both runs are still sitting on level 0 and the 0.62x ratio is"
            " two near-zero numbers dividing. The curriculum term and its thresholds are identical on both sides"
            " (tracking_exp_vel, (0.3, 0.6), (0.0, 0.0)); only the frame feeding its gate differs."
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
    "actuation/delay": (
        "ImplicitPD, no delay (the delayed table is assigned, then discarded by apply_shoe_config)",
        "DelayedPD, 0-2 physics steps",
        (
            "Kept on purpose after measuring it. Main only appears to use delay; the registered cfg "
            "replaces self.scene.robot after setting the delayed table, and a real main run's "
            "params/env.yaml shows five ImplicitActuator groups. Dropping our delay moves dof_acc_l2 "
            "from 1.71x main to 0.98x, dof_vel_l2 1.04x→0.98x, action_rate_l2 1.02x→0.99x, and leaves "
            "reward at 0.82x either way while episode length goes 1.02x→0.93x. We keep the delay as "
            "the more realistic plant; the ~18% reward gap is something else."
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
    from instinctlab.tasks.parkour.config.g1.agents.instinct_rl_amp_cfg import G1ParkourTargetPPORunnerCfg

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
    reference = main_ref.reward_weights()
    assert declared.pop("volume_points_penetration") == -8.0
    assert reference.pop("volume_points_penetration") == -4.0
    assert declared == reference


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


def test_env_entry_class_drift_is_guarded_without_isaacsim() -> None:
    """KNOWN_DRIFTS['env/entry_class']: main InstinctRlEnv vs our ManagerBasedRLEnv compile path."""
    assert main_ref.uses_instinct_rl_env()
    isaac_adapter = (REPO / "source/instinctlab/instinctlab/engines/isaacsim/adapter.py").read_text()
    assert "InstinctManagerBasedRLEnv.wrap(ManagerBasedRLEnv)" in isaac_adapter
    assert "InstinctRlEnv" not in isaac_adapter
    train = (REPO / "scripts/train.py").read_text()
    assert "InstinctRlEnv" not in train
    assert "compiled.make_env()" in train


def test_agent_shared_hyperparameters_match_main(our_agent, main_agent) -> None:
    """Learning fields match; checkpoint and logging cadence are operational settings."""
    ignore = {
        "algorithm.actor_state_key",
        "algorithm.reference_state_key",
        "normalizers.critic.class_name",
        "normalizers.policy.class_name",
        "empirical_normalization",
        "load_run",
        "log_interval",
        "resume",
        "save_interval",
    }
    shared = (set(main_agent) & set(our_agent)) - ignore
    assert len(shared) >= 30
    differing = {key: (main_agent[key], our_agent[key]) for key in shared if main_agent[key] != our_agent[key]}
    assert not differing, differing


def test_known_drifts_table_is_non_empty_and_stable() -> None:
    assert len(KNOWN_DRIFTS) == 9
    assert "actuation/delay" in KNOWN_DRIFTS
    assert "dataset_exhausted" not in DELIBERATE_OMISSIONS
    assert "reward/undesired_contacts/threshold" not in KNOWN_DRIFTS
    assert "scene/robot/urdf" not in KNOWN_DRIFTS
    assert "scene/robot/spawn_z" not in KNOWN_DRIFTS
    assert "scene/robot/merge_fixed_joints" not in KNOWN_DRIFTS
    assert "scene/robot/actuators" not in KNOWN_DRIFTS
    assert "sim/physx/gpu_collision_stack_size" not in KNOWN_DRIFTS
    assert "amp/symmetric_augmentation" not in KNOWN_DRIFTS
    assert "obs/joint_axis" in KNOWN_DRIFTS
    assert "obs/base_lin_vel_frame" in KNOWN_DRIFTS
    assert "reward/volume_points_penetration/weight" in KNOWN_DRIFTS


def test_documented_drifts_are_still_present(task) -> None:
    """Each KNOWN_DRIFTS entry must remain true — the table is not a graveyard."""
    from instinctlab.assets.unitree_g1.catalog import G1_29DOF_DFS_JOINT_NAMES, G1_29DOF_ISAAC_BFS_JOINT_NAMES
    from instinctlab.mdp.observations import base_lin_vel

    rewards = task.mdp.rewards["rewards"]
    assert main_ref.reward_weights()["volume_points_penetration"] == -4.0
    assert rewards["volume_points_penetration"].weight == -8.0
    assert rewards["track_lin_vel_xy_exp"].func.__name__ == "track_lin_vel_xy_exp"
    assert rewards["dont_wait"].func.__name__ == "dont_wait"
    assert rewards["undesired_contacts"].kind == "undesired_contacts"

    # velocity_frame x3 -- both sides. Matching function *names* said nothing about the frame,
    # which is the only thing these three rows are about: they would have stayed green through
    # any change to either side's spelling.
    theirs = main_ref.velocity_frame_spellings()
    assert set(theirs.values()) == {"root_lin_vel_b"}, theirs
    for func in (rewards["track_lin_vel_xy_exp"].func, rewards["dont_wait"].func):
        body = inspect.getsource(func)
        assert "root_link_lin_vel_b" in body and ".root_lin_vel_b" not in body, func.__name__
    from instinctlab.engines.pose_velocity import PoseVelocityMixin

    metrics = inspect.getsource(PoseVelocityMixin._update_metrics)
    assert "root_link_lin_vel_b" in metrics and ".root_lin_vel_b" not in metrics
    assert task.scene.volume_point("leg_volume_points").velocity == "attach_link"
    # Their half of that row: PhysX COM velocity with the lever measured from the link origin,
    # so every point carries a constant ω × (origin − com) error. Same family as InstinctMJ's
    # pelvis-to-ankle lever, different arm.
    assert main_ref.volume_points_point_velocity() == {
        "velocity_from_physx_com": True,
        "lever_from_link_origin": True,
    }
    assert "dataset_exhausted" in task.mdp.terminations
    assert tuple(task.robot.joint_names) == G1_29DOF_DFS_JOINT_NAMES
    assert G1_29DOF_DFS_JOINT_NAMES != G1_29DOF_ISAAC_BFS_JOINT_NAMES
    assert main_ref.observation_joint_names("policy", "joint_pos") is None
    assert main_ref.action_kwargs()["joint_names"] == [".*"]
    assert task.mdp.observations["critic"].terms["base_lin_vel"].func is base_lin_vel
    assert "root_link_lin_vel_b" in inspect.getsource(base_lin_vel)
    # Same name on both sides, different quantity behind it -- which is the whole row, so the
    # name match alone would have been the vacuous form of this assertion.
    assert main_ref.observation_functions("critic")["base_lin_vel"] == "base_lin_vel"
    assert theirs["base_lin_vel"] == "root_lin_vel_b"
    source = main_ref.motion_reference_source()
    assert (
        source.get("symmetric_augmentation_link_mapping") is not None or "symmetric_augmentation_link_mapping" in source
    )
    assert task.scene.motion_reference("motion_reference").symmetric_augmentation is not None
    # actuation/delay — verify both sides, not just ours. "We declare a delay" would stay
    # green if main grew one too, which is precisely how this row got written backwards.
    assert task.robot.actuator_delay == (0, 2)
    assert main_ref.effective_robot_actuators()["delayed"] is False
    isaac_adapter = (
        Path(__file__).resolve().parents[1] / "source/instinctlab/instinctlab/engines/isaacsim/adapter.py"
    ).read_text()
    assert main_ref.uses_instinct_rl_env()
    assert "InstinctManagerBasedRLEnv.wrap(ManagerBasedRLEnv)" in isaac_adapter
    assert "InstinctRlEnv" not in isaac_adapter


def test_dof_vel_limits_match_mains_actuator_table(task) -> None:
    """Our literal 29 limits against the ones main's penalty reads off the robot.

    Not a drift row, a *latent* one. main computes ``soft_ratio * soft_joint_vel_limits``,
    which Isaac Lab fills from the actuator table (``velocity_limit`` unset falls back to
    ``velocity_limit_sim``). We pass the numbers in as a literal tuple, so today's agreement
    is a coincidence maintained by hand: change one actuator on main and nothing here would
    notice. The observed 3.36x on ``Episode_Reward/dof_vel_limits`` had this as its most
    plausible mechanism until these were compared and found equal, which leaves that ratio
    as noise on a term worth about -0.001.
    """
    from instinctlab.assets.unitree_g1.catalog import G1_29DOF_DFS_JOINT_NAMES

    term = task.mdp.rewards["rewards"]["dof_vel_limits"]
    ours = dict(zip(term.params["asset_cfg"].joints, term.params["limits"], strict=True))
    theirs = main_ref.actuator_joint_velocity_limits(G1_29DOF_DFS_JOINT_NAMES)
    assert ours == theirs, {j: (ours[j], theirs[j]) for j in ours if ours[j] != theirs[j]}
    assert term.params["soft_ratio"] == main_ref.reward_params()["dof_vel_limits"]["soft_ratio"] == 0.9


def test_the_actuator_limit_reader_refuses_a_table_it_cannot_expand() -> None:
    """A reader that silently drops a joint would make the comparison above vacuous.

    Both ways it can go blind are made loud: a joint no actuator group claims, and a group
    that claims nothing at all (which is what a renamed joint on their side looks like).
    """
    from instinctlab.assets.unitree_g1.catalog import G1_29DOF_DFS_JOINT_NAMES

    with pytest.raises(LookupError, match="without a velocity limit"):
        main_ref.actuator_joint_velocity_limits([*G1_29DOF_DFS_JOINT_NAMES, "no_such_joint"])
    with pytest.raises(LookupError, match="matches no joint"):
        main_ref.actuator_joint_velocity_limits(["left_knee_joint"])


def test_the_velocity_frame_reader_refuses_what_it_cannot_resolve() -> None:
    """A reader that guesses is worse than no reader: it puts a wrong value into a drift row.

    Two failure modes are made loud rather than defaulted -- a function that reads no root
    velocity at all, and one that reads two spellings, which is what a half-finished migration
    from COM to link looks like.
    """
    resolve = main_ref._velocity_spelling_in

    assert resolve("def f(a):\n    return a.data.root_link_lin_vel_b[:, 0]\n", "f") == "root_link_lin_vel_b"
    assert resolve("class C:\n    def m(self):\n        return self.r.data.root_lin_vel_b\n", "C.m") == "root_lin_vel_b"

    with pytest.raises(LookupError, match="no root velocity"):
        resolve("def f(a):\n    return a.data.joint_pos\n", "f")
    with pytest.raises(LookupError, match="root_lin_vel_b"):
        resolve("def f(a):\n    return a.data.root_lin_vel_b + a.data.root_link_lin_vel_b\n", "f")
    with pytest.raises(LookupError, match="no function"):
        resolve("def g(a):\n    return a.data.root_lin_vel_b\n", "f")


def test_main_leaves_policy_and_action_joints_in_entity_order(task) -> None:
    """Main's implicit `.*` is PhysX BFS; we name DFS. This is KNOWN_DRIFTS['obs/joint_axis']."""
    from instinctlab.assets.unitree_g1.catalog import G1_29DOF_DFS_JOINT_NAMES, G1_29DOF_ISAAC_BFS_JOINT_NAMES

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


def test_main_amp_mirrors_half_the_reference_and_so_do_we(task) -> None:
    """Main's maps are integer tables in PhysX BFS order; ours resolve by joint name.

    The assertion this replaces was ``not hasattr(clip, "symmetric_augmentation_joint_mapping")``,
    naming a field only the reference has. It stayed green after we implemented mirroring, because
    we were never going to grow the reference's spelling. That the name map resolves to main's exact
    BFS table is ``tests/test_amp_symmetric_augmentation.py::test_resolved_bfs_indices_equal_main_table``.
    """
    source = main_ref.motion_reference_source()
    assert source["symmetric_augmentation_link_mapping"] == [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]
    assert "symmetric_augmentation_joint_mapping" in source
    assert "symmetric_augmentation_joint_reverse_buf" in source
    amass = source["amass"]
    assert "filtered_motion_selection_filepath" in amass
    assert "parkour_motion_without_run.yaml" in str(amass["filtered_motion_selection_filepath"])
    clip = task.scene.motion_reference("motion_reference")
    assert clip.clip.endswith("parkour_motion_without_run_retargetted.npz")
    mirror = clip.symmetric_augmentation
    assert mirror is not None, "parkour declares no mirror augmentation; main mirrors half its resets"
    assert mirror.joint_swaps["left_hip_pitch_joint"] == "right_hip_pitch_joint"
    assert mirror.joint_swaps["waist_pitch_joint"] == "waist_pitch_joint"
    assert mirror.joint_signs["waist_roll_joint"] == -1


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
    """CLI defaults, seed hand-off, and the optional resume path."""
    theirs = main_ref.train_script_facts()
    ours = Path("/root/InstinctLab/scripts/train.py").read_text()
    adapter = Path("/root/InstinctLab/source/instinctlab/instinctlab/engines/isaacsim/adapter.py").read_text()
    assert theirs["sets_env_seed_from_agent"] is True
    assert "compiled.env_cfg.seed = distributed.seed(agent_cfg.seed)" in ours
    assert theirs["num_envs_default"] is None
    assert "default=4096" in ours
    assert theirs["seed_default"] is None
    assert theirs["sets_tf32"] is True
    assert "allow_tf32 = True" in adapter
    assert theirs["calls_runner_load"] is True
    assert "load_runner_checkpoint(runner, resume_path, distributed)" in ours
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


def test_parkour_robot_matches_main_on_the_three_task_overrides(task) -> None:
    """The plant overrides that main really applies. The delay is not one of them."""
    robot = task.robot.asset_for("isaacsim")
    assert robot.path.endswith(main_ref.G1_SHOE_URDF_SUFFIX)
    assert task.robot.default_root_pos[2] == pytest.approx(main_ref.G1_SPAWN_Z)
    assert robot.import_options["merge_fixed_joints"] is main_ref.G1_MERGE_FIXED_JOINTS
    assert main_ref.effective_robot_actuators()["delayed"] is False
    assert task.robot.actuator_delay == (0, 2), "kept on purpose; KNOWN_DRIFTS['actuation/delay']"


@pytest.mark.isaacsim
def test_compiled_isaac_robot_matches_main_plant_and_keeps_documented_sim_drifts() -> None:
    """One Kit session: compile and read the fields the static audit flagged."""
    pytest.importorskip("isaaclab")
    from tests.isaacsim_app import ensure_isaac_app
    from tests.live_device import resolve_live_device

    device = resolve_live_device()
    ensure_isaac_app(device=device)

    from instinctlab.engines.isaacsim import IsaacSimAdapter

    spec = parkour_target_g1()
    compiled = IsaacSimAdapter().compile(spec, num_envs=16, device=device)
    robot = compiled.env_cfg.scene.robot
    sim = compiled.env_cfg.sim
    camera = compiled.env_cfg.scene.camera

    assert robot.init_state.pos[2] == pytest.approx(main_ref.G1_SPAWN_Z)
    assert robot.spawn.asset_path.endswith(main_ref.G1_SHOE_URDF_SUFFIX)
    assert robot.spawn.merge_fixed_joints is True
    # Delay is our documented divergence, not main's plant: KNOWN_DRIFTS['actuation/delay'].
    actuator_types = {type(cfg).__name__ for cfg in robot.actuators.values()}
    assert actuator_types == {"DelayedPDActuatorCfg"}
    assert {(cfg.min_delay, cfg.max_delay) for cfg in robot.actuators.values()} == {(0, 2)}
    assert sim.physx.gpu_max_rigid_patch_count == 10 * 2**15
    assert sim.physx.gpu_collision_stack_size == 2**29
    assert compiled.env_cls.__name__ == "ManagerBasedRLEnv"
    assert compiled.env_cfg.episode_length_s == 20.0
    assert compiled.env_cfg.decimation == 4
    assert compiled.env_cfg.sim.dt == pytest.approx(0.005)
    assert camera.depth_clipping_behavior == "none"
