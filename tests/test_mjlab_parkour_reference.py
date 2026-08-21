"""Compiled Instinct-Parkour-Target-G1 on mjlab, checked against InstinctMJ's parkour.

InstinctMJ is read through :mod:`tests.reference_mjlab_parkour` (syntax tree, no import).
The compiled cfg is what trains: ``MjlabAdapter().compile`` on CPU.

``KNOWN_DRIFTS`` entries must remain true. A silent fix, or a new unexplained difference,
fails the same way a missing reward would. Deliberate omissions live in ``DELIBERATE``.
"""

from __future__ import annotations

import ast
import inspect
import math
import re
from pathlib import Path

import pytest

from instinctlab.tasks.parkour.config.g1 import parkour_target_g1
from instinctlab.utils.configclass import class_to_dict
from tests import reference_main_parkour as main_ref
from tests import reference_mjlab_parkour as mj_ref

pytestmark = pytest.mark.skipif(not mj_ref.available(), reason="InstinctMJ is not checked out")

DELIBERATE = {
    "termination/dataset_exhausted": (
        "InstinctMJ: dataset_exhausted with reset_without_notice=True (reports 0)",
        "absent",
        "Omitted by design; exhaustion is on the sensor (validity / exhausted_count).",
    ),
    "volume_points/velocity": (
        "foot-subtree cvel (ω × (pelvis − ankle) lever)",
        "attach_link",
        "Upstream bug; compat/denylist.py. Deliberately not reproduced.",
    ),
    "terrain/num_cols": (
        "written 20, curriculum mode builds one column per type",
        "honored: 20 columns",
        (
            "Isaac's cumulative-proportion allocation. InstinctMJ ignores num_cols in curriculum."
            " Changes the type mix: 20 columns by proportion give 50% stairs and 10% perlin against"
            " their 40% / 20%. Not the cause of our shorter episodes, and checked rather than assumed:"
            " reweighting our own per-sub-terrain lengths to their mix moves the mean 126.8 -> 122.1,"
            " i.e. 4.7 steps the WRONG WAY against the 29-56 needed to reach them. Stairs are the"
            " terrains our policy survives longest on (~137) and perlin the shortest (~90), so having"
            " more stairs helps us. Both seeds agree."
        ),
    ),
    "sim/nconmax": (
        str(mj_ref.NCONMAX),
        "256",
        "20-column grid overflowed 128 (host mj_forward ncon=164). Per-world budget, not a task term.",
    ),
    "sim/njmax": (
        "700",
        "768",
        (
            "Resting pose peaked at nefc=691 (98.7% of 700); put_data refuses njmax<691. "
            "768 is ~11% headroom. InstinctMJ still writes 700 for the 10-column grid."
        ),
    ),
    "viz": (
        "collision_debug_vis / debug_vis on",
        "omitted",
        "Play-time markers only; they do not enter the observation or the reward.",
    ),
    "scene/height_scanner/offset": (
        f"{mj_ref.SCANNER_ORIGIN_OFFSET}, max_distance={mj_ref.SCANNER_MAX_DISTANCE}",
        "(0.04, 0.0, 20.0), max_distance=1e6",
        (
            "Isaac sky-ray on both engines: origin 20 m above the ankle, miss=+inf. "
            "Verified on flat ground, step edges, and a tilted ankle. InstinctMJ starts at the ankle."
        ),
    ),
    "sim/ccd_iterations": (
        str(mj_ref.CCD_ITERATIONS),
        "500 (PROFILE_DEFAULTS)",
        (
            "8-env 20-step probe: 128-vs-128 noise already 0.0027 reward / 16 ncon; 128-vs-500 "
            "sat inside that. Per-step 31.07 vs 31.41 ms (+1%). A cost knob, not a term change."
        ),
    ),
}

# Two references disagree; we track Isaac so the Isaac engine does not silently change.
REFERENCE_DIVERGENCE: dict[str, tuple[str, str, str]] = {
    "reward/dof_vel_limits": (
        "InstinctMJ: absent; Isaac main: weight=-1.0, soft_ratio=0.9",
        "weight=-1.0, soft_ratio=0.9 (track Isaac)",
        (
            "Isaac's 26-term set includes this; InstinctMJ dropped it. Cross-engine declaration "
            "follows Isaac so a later 'fix' toward InstinctMJ would retune the Isaac side."
        ),
    ),
    "camera/hit_targets": (
        "InstinctMJ: default geom groups (0, 1, 2); Isaac main: /World/ground + G1_29DOF_LINKS",
        "terrain + G1_29DOF_LINKS by body name (track Isaac)",
        (
            "Named-body hits, as main's Isaac parkour. InstinctMJ's group mask is a "
            "reference-vs-reference split. 'Fixing' toward groups would retune the Isaac side."
        ),
    ),
    "actuation/pd": (
        "InstinctMJ: BuiltinPd implicit-integration + delay; main: ImplicitPD, no delay",
        "both engines delayed: mjlab matches InstinctMJ, Isaac deliberately does not match main",
        (
            "InstinctMJ keeps its delay through the shoe branch because that branch deepcopies the "
            "already-patched robot. Main's assigns the delayed table and then replaces the whole robot, "
            "so the registered task trains on implicit PD -- see test_g1_robot_catalog_ownership. "
            "Measured on Isaac at 700 iterations: dropping our delay moves dof_acc_l2 from 1.71x main "
            "to 0.98x and dof_vel_l2 from 1.04x to 0.98x, but leaves reward at 0.82x either way. "
            "The delay is kept because it is the more realistic plant and the episode-length match is "
            "better with it (1.02x vs 0.93x); it is not kept because main has it."
        ),
    ),
}

KNOWN_DRIFTS: dict[str, tuple[str, str, str]] = {
    # "contact/sensor" was here until the air-time clock was aligned. The row read as
    # a neutral implementation detail -- "we keep it portable, they subclass" -- but
    # mjlab clocks air time off `found`, which is set for any contact at any force,
    # while Isaac Lab, main and InstinctMJ all require 1 N. Ours was the one behaviour
    # of four, and feet_air_time was scoring a different gait on each of our engines.
    # The threshold is now ContactSensorRef.air_time_force_threshold and both backends
    # map it onto their own field; see tests/test_contact_air_time_threshold.py.
    # "actuation/delay_lag_groups" was here until the lag partition was aligned. mjlab fuses
    # actuators whose delay settings match onto one DelayBuffer, so delay_update_period -- not
    # the config split -- decides which joints draw the SAME lag, and we were deriving it from
    # a PD-gain group index. That put one leg's hip pitch/yaw on a different draw from its hip
    # roll/knee and bound the ankles to the waist: a partition neither reference has, and a
    # plant where a leg can be internally inconsistently late. The bus is now declared once on
    # the robot (JointProperties.actuator_group) and both engines wire the lag off it; see
    # test_lag_partition_matches_instinctmj.
    #
    # The row also guessed at a consequence -- "a plausible contributor to our 25-35% higher
    # root_height rate" -- and that guess was wrong. Two seeds each side at 256/700: root_height
    # 1.871 -> 2.052 (against InstinctMJ's 1.384, so 1.35x -> 1.48x) and episode length 136.5 ->
    # 122.1. Both moves are smaller than the spread between the two seeds *within* each arm
    # (post 1.91/2.20, pre 1.98/1.76), so the honest reading is no detectable effect, and
    # certainly not the closure the hypothesis predicted. The alignment is kept because the
    # partition is a physical fact both references declare and we were the odd one out on it,
    # not because it bought anything measurable. Our higher fall rate remains unexplained.
    "motion/source": (
        "AmassMotionCfg yaml filter → parkour_motion_without_run.yaml",
        "MotionReferenceRef clip=…parkour_motion_without_run_retargetted.npz",
        (
            "Spelling, not inventory: the shipped yaml resolves to exactly the one npz we load, "
            "re-checked on every run by test_shipped_motion_yaml_selects_the_same_npz rather than "
            "trusted from a dated note "
            "(selected_files=['parkour_motion_without_run_retargetted.npz'], "
            "18982 frames @ 50 Hz). Symmetric augmentation was listed here as a residual until it "
            "was implemented on both engines; what remains is the loader path alone."
        ),
    ),
}


@pytest.fixture(scope="module")
def task():
    return parkour_target_g1()


@pytest.fixture(scope="module")
def compiled(task):
    pytest.importorskip("mjlab")
    from instinctlab.engines.mjlab import MjlabAdapter

    return MjlabAdapter().compile(task, num_envs=16, device="cpu")


@pytest.fixture(scope="module")
def our_agent() -> dict:
    from instinctlab.tasks.parkour.config.g1.agents.instinct_rl_parkour_cfg import G1ParkourTargetPPORunnerCfg

    return class_to_dict(G1ParkourTargetPPORunnerCfg())


def _flatten(tree: dict, prefix: str = "") -> dict:
    flat: dict = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def test_the_reference_file_is_the_parkour_factory() -> None:
    assert mj_ref.REFERENCE.is_file()
    assert mj_ref.FACTORY in mj_ref.REFERENCE.read_text()


def test_reward_sets_differ_only_by_the_documented_extra(task) -> None:
    """Enumerate both sides. A missing or extra term does not throw; it trains a different task."""
    declared = set(task.mdp.rewards["rewards"])
    reference = set(mj_ref.reward_names())
    assert declared - reference == {"dof_vel_limits"}
    assert reference - declared == set()
    assert [n for n in mj_ref.reward_names()] == [n for n in task.mdp.rewards["rewards"] if n != "dof_vel_limits"]


def test_shared_reward_weights_match(task) -> None:
    declared = {name: term.weight for name, term in task.mdp.rewards["rewards"].items()}
    reference = mj_ref.reward_weights()
    shared = {name: declared[name] for name in reference}
    assert shared == reference
    assert declared["dof_vel_limits"] == -1.0


def test_shared_reward_numeric_params_match_effective_shoe(task) -> None:
    declared = task.mdp.rewards["rewards"]
    reference = mj_ref.reward_params()
    numeric_keys = {
        "track_lin_vel_xy_exp": ("std",),
        "track_ang_vel_z_exp": ("std",),
        "stand_still": ("offset",),
        "feet_air_time": ("vel_threshold",),
        "feet_at_plane": ("height_offset",),
        "feet_close_xy": ("threshold",),
        "feet_slide": ("threshold",),
        "torque_limits": ("limit_ratio",),
    }
    for name, keys in numeric_keys.items():
        for key in keys:
            assert declared[name].params.get(key) == reference[name].get(key), f"{name}.{key}"
    assert declared["feet_at_plane"].params["height_offset"] == mj_ref.SHOE_HEIGHT_OFFSET
    assert declared["feet_close_xy"].params["std"] == pytest.approx(math.sqrt(0.05))
    assert mj_ref.reward_functions()["stand_still"] == "stand_still"
    assert declared["stand_still"].func.__name__ == "stand_still_when_idle"


def test_terminations_only_drop_dataset_exhausted(task) -> None:
    reference = set(mj_ref.termination_names())
    declared = set(task.mdp.terminations)
    renamed = {"terrain_out_bound": "terrain_out_of_bounds"}
    dropped = reference - declared - set(renamed)
    assert dropped == {"dataset_exhausted"}
    for old, new in renamed.items():
        assert old in reference and new in declared
    assert "dataset_exhausted" not in declared


def test_event_names_match(task) -> None:
    assert set(task.mdp.events) == set(mj_ref.event_names())


def test_reset_ranges_and_friction_match(task) -> None:
    events = mj_ref.event_params()
    reset = task.mdp.events["reset_base"].params
    assert reset["pose_range"] == events["reset_base"]["pose_range"]
    assert reset["velocity_range"] == events["reset_base"]["velocity_range"]
    joints = task.mdp.events["reset_robot_joints"].params
    assert joints["position_range"] == events["reset_robot_joints"]["position_range"]
    assert joints["velocity_range"] == events["reset_robot_joints"]["velocity_range"]
    material = events["physics_material"]
    assert material["static_friction_range"] == (0.3, 1.6)
    assert material["dynamic_friction_range"] == (0.3, 1.6)
    assert material["make_consistent"] is True


def test_observation_order_noise_and_scale_match(task) -> None:
    for group in ("policy", "critic", "amp_policy", "amp_reference"):
        assert tuple(task.mdp.observations[group].terms) == mj_ref.observation_order(group), group
    policy = task.mdp.observations["policy"].terms
    noise = mj_ref.observation_noise("policy")
    for name, (lo, hi) in noise.items():
        assert policy[name].noise is not None, name
        assert (policy[name].noise.lo, policy[name].noise.hi) == (lo, hi)
    for group in ("policy", "critic", "amp_policy", "amp_reference"):
        scales = mj_ref.observation_scales(group)
        for name, scale in scales.items():
            ours = task.mdp.observations[group].terms[name].scale
            if scale is None:
                assert ours in (None, 1.0), f"{group}.{name}"
            else:
                assert ours == scale, f"{group}.{name}"


def test_command_velocity_boxes_match_including_dense_boxes(task) -> None:
    reference = mj_ref.command_params()
    resolved = task.mdp.commands["base_velocity"].resolved_params("mjlab")
    assert resolved["resampling_time_range"] == reference["resampling_time_range"]
    assert resolved["velocity_control_stiffness"] == reference["velocity_control_stiffness"]
    assert resolved["heading_control_stiffness"] == reference["heading_control_stiffness"]
    assert resolved["only_positive_lin_vel_x"] is True
    assert resolved["rel_standing_envs"] == 0.05
    assert resolved["lin_vel_threshold"] == 0.0
    assert resolved["ang_vel_threshold"] == 0.0
    assert resolved["target_dis_threshold"] == 0.4
    assert set(resolved["velocity_ranges"]) == set(reference["velocity_ranges"])
    for name, box in reference["velocity_ranges"].items():
        assert resolved["velocity_ranges"][name] == box, name


def test_sim_timings_match_and_solver_diffs_are_the_documented_ones(task, compiled) -> None:
    overrides = mj_ref.sim_overrides()
    assert task.sim.physics_dt == 0.005
    assert task.sim.decimation == 4
    assert task.sim.episode_length_s == overrides["episode_length_s"] == 20.0
    assert compiled.env_cfg.sim.njmax == 768
    assert overrides["njmax"] == 700
    assert compiled.env_cfg.sim.contact_sensor_maxmatch == overrides["contact_sensor_maxmatch"] == 128
    assert compiled.env_cfg.sim.mujoco.iterations == overrides["iterations"] == 10
    assert compiled.env_cfg.sim.mujoco.ls_iterations == overrides["ls_iterations"] == 20
    assert compiled.env_cfg.sim.nconmax == 256
    assert overrides["nconmax"] == mj_ref.NCONMAX
    assert compiled.env_cfg.sim.mujoco.ccd_iterations == 500
    assert overrides["ccd_iterations"] == mj_ref.CCD_ITERATIONS
    assert overrides["init_pos"][2] == mj_ref.SPAWN_Z


def test_parkour_robot_matches_instinctmj_on_spawn_delay_shoe(task, compiled) -> None:
    robot = compiled.env_cfg.scene.entities["robot"]
    assert robot.init_state.pos[2] == pytest.approx(mj_ref.SPAWN_Z)
    assert mj_ref.sim_overrides()["init_pos"][2] == pytest.approx(mj_ref.SPAWN_Z)
    lags = {
        (getattr(act, "delay_min_lag", 0), getattr(act, "delay_max_lag", 0)) for act in robot.articulation.actuators
    }
    assert lags == {(0, mj_ref.DELAY_MAX_LAG)}
    assert mj_ref.delayed_actuator_lags() == (0, mj_ref.DELAY_MAX_LAG)
    asset = task.robot.asset_for("mjlab").path
    assert asset.endswith(mj_ref.SHOE_XML_SUFFIX)
    assert mj_ref.shoe_effective()["xml_suffix"] == mj_ref.SHOE_XML_SUFFIX


def test_compiled_scanners_volume_contact_and_camera_are_the_documented_drifts(task, compiled) -> None:
    robot = compiled.env_cfg.scene.entities["robot"]
    assert robot.init_state.pos[2] == pytest.approx(mj_ref.SPAWN_Z)
    lags = {
        (getattr(act, "delay_min_lag", 0), getattr(act, "delay_max_lag", 0)) for act in robot.articulation.actuators
    }
    assert lags == {(0, mj_ref.DELAY_MAX_LAG)}
    asset = task.robot.asset_for("mjlab").path
    assert asset.endswith(mj_ref.SHOE_XML_SUFFIX)
    assert mj_ref.shoe_effective()["xml_suffix"] == mj_ref.SHOE_XML_SUFFIX
    sensors = {sensor.name: sensor for sensor in compiled.env_cfg.scene.sensors}
    for name in ("left_height_scanner", "right_height_scanner"):
        assert sensors[name].origin_offset == (0.04, 0.0, 20.0)
        assert sensors[name].max_distance == pytest.approx(1e6)
    reference_sensors = mj_ref.sensor_cfgs()
    assert reference_sensors["left_height_scanner"]["max_distance"] == mj_ref.SCANNER_MAX_DISTANCE
    assert "origin_offset" not in reference_sensors["left_height_scanner"]
    volume = sensors["leg_volume_points"]
    assert volume.velocity == "attach_link"
    grid = task.scene.volume_point("leg_volume_points").grid
    assert (grid.z_min, grid.z_max) == mj_ref.SHOE_VOLUME_Z
    assert mj_ref.shoe_effective()["volume_z"] == mj_ref.SHOE_VOLUME_Z
    contact = sensors["contact_forces"]
    assert "found" in contact.fields
    # Both sides now clock air time off net force at the same threshold. Asserted as an
    # equality rather than as two separate literals, so a change on either side shows up
    # here instead of only in whichever half someone remembered to update.
    assert reference_sensors["contact_forces"]["cfg_class"] == "ForceThresholdContactSensorCfg"
    assert contact.force_threshold == reference_sensors["contact_forces"]["force_threshold"] == 1.0
    assert "found" not in reference_sensors["contact_forces"]["fields"]
    camera = sensors["camera"]
    assert camera.include_geom_groups is None
    assert "camera" in reference_sensors
    assert reference_sensors["camera"]["cfg_class"] == "NoisyGroupedRayCasterCameraCfg"
    assert reference_sensors["camera"]["max_distance"] == 2.5
    assert compiled.env_cfg.scene.env_spacing == 2.5


def test_compiled_friction_is_the_union_instinctmj_states(compiled) -> None:
    params = compiled.env_cfg.events["physics_material"].params
    assert params["ranges"] == (0.3, 1.6)
    assert params["shared_random"] is True


def test_compiled_action_scale_matches_beyondmimic_formula(task, compiled) -> None:
    scale = compiled.env_cfg.actions["joint_pos"].scale
    expected = {joint.name: joint.action_scale for joint in task.robot.joint_properties}
    assert scale == expected
    hip = expected["left_hip_pitch_joint"]
    assert hip == pytest.approx(0.25 * 88.0 / task.robot.joint_properties[3].stiffness)


def test_terrain_recipe_constants_match_instinctmj(compiled) -> None:
    """Read from their file, not transcribed from it.

    This asserted hand-copied literals until 2026-08-20, which meant a change on the InstinctMJ
    side could not fail it: the test would keep agreeing with a number nobody had rechecked while
    its name went on claiming a match.
    """
    theirs = mj_ref.terrain_recipe()
    gen = compiled.env_cfg.scene.terrain.terrain_generator

    assert gen.horizontal_scale == theirs["horizontal_scale"] == 0.07
    assert gen.num_cols == theirs["num_cols"] == 20
    assert gen.num_rows == theirs["num_rows"] == 10
    assert gen.curriculum is theirs["curriculum"] is True
    assert tuple(gen.size) == tuple(theirs["size"])
    assert gen.border_width == theirs["border_width"]
    assert gen.vertical_scale == theirs["vertical_scale"]

    # Every sub-terrain they declare must exist here, with the same proportion -- the column
    # allocation is derived from those, so a silent change reshuffles which terrain a velocity
    # box lands on.
    assert set(gen.sub_terrains) == set(theirs["sub_terrains"])
    for name, params in theirs["sub_terrains"].items():
        ours = gen.sub_terrains[name]
        assert ours.proportion == params["proportion"], name
        for field in ("step_width", "border_width", "platform_width", "step_height_range"):
            if field in params:
                assert getattr(ours, field) == params[field], f"{name}.{field}"


def test_the_terrain_reader_would_notice_a_change_on_their_side() -> None:
    """The recipe comparison is only worth anything if the reader can disagree.

    It reads their file at call time and caches; the fields the test compares must actually come
    from there, so a reader that quietly returned an empty recipe would make every comparison
    vacuously true.
    """
    theirs = mj_ref.terrain_recipe()
    assert set(theirs) >= {"num_cols", "num_rows", "horizontal_scale", "curriculum", "sub_terrains"}
    assert len(theirs["sub_terrains"]) == 10
    assert all(isinstance(params, dict) and params for params in theirs["sub_terrains"].values())
    assert sum(params["proportion"] for params in theirs["sub_terrains"].values()) == pytest.approx(1.0)

    broken = dict(theirs, num_cols=999)
    assert broken["num_cols"] != mj_ref.terrain_recipe()["num_cols"], "the cache must not be shared by reference"


def test_motion_source_is_the_documented_drift(task) -> None:
    ours = task.scene.motion_references[0].clip
    assert ours.endswith("parkour_motion_without_run_retargetted.npz")
    source = mj_ref.motion_source()
    assert "parkour_motion_reference" in source["dataset_dir"]
    assert "parkour_motion_without_run.yaml" in (source["filter"] or "")


def test_shipped_motion_yaml_selects_the_same_npz(task) -> None:
    """Quantify motion/source: the yaml filter is not a multi-clip AMASS walk on shipped data."""
    files = mj_ref.motion_filter_files()
    assert len(files) == 1
    assert files[0].endswith("parkour_motion_without_run_retargetted.npz")
    assert task.scene.motion_references[0].clip.endswith(files[0])
    assert mj_ref.SHIPPED_MOTION_NPZ.is_file()


def test_amp_symmetric_augmentation_matches_instinctmj(task) -> None:
    """We mirror too now. This was a drift row until the reference was read rather than assumed.

    The row it replaces asserted ``not hasattr(clip, "symmetric_augmentation_joint_mapping")`` --
    a field name we never adopted, because ours resolves by joint *name* rather than by the
    reference's integer table. That assertion would keep passing after we implemented mirroring,
    which is why it is spelled against the declaration here instead.
    """
    source = mj_ref.motion_source()
    assert source["symmetric_augmentation_link_mapping"] == [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]
    assert source["symmetric_augmentation_joint_mapping"] is not None
    assert source["symmetric_augmentation_joint_reverse_buf"] is not None

    mirror = task.scene.motion_references[0].symmetric_augmentation
    assert mirror is not None, "parkour declares no mirror augmentation; InstinctMJ mirrors half its resets"
    assert mirror.joint_swaps["left_hip_pitch_joint"] == "right_hip_pitch_joint"
    assert mirror.joint_swaps["waist_pitch_joint"] == "waist_pitch_joint"
    assert mirror.joint_signs["waist_roll_joint"] == -1
    assert mirror.joint_signs["waist_pitch_joint"] == 1
    assert task.scene.motion_references[0].clip.endswith(".npz")


def test_train_pipeline_configures_torch_like_instinctmj() -> None:
    """Both call the same helper. This was a drift until the reference was read rather than assumed.

    The behavioural half -- that the flags actually flip -- is in
    ``tests/test_train_entry.py::test_mjlab_bootstrap_actually_turns_tf32_matmul_on``.
    """
    import inspect

    assert mj_ref.train_script_calls_configure_torch_backends()
    from instinctlab.engines.mjlab import adapter as mj_adapter

    assert "configure_torch_backends" in inspect.getsource(mj_adapter.MjlabAdapter.bootstrap)


def test_ppo_runner_fields_match_instinctmj(our_agent) -> None:
    reference = mj_ref.agent_fields()
    shared_algo = (
        "class_name",
        "discriminator_reward_coef",
        "discriminator_reward_type",
        "discriminator_loss_func",
        "discriminator_gradient_penalty_coef",
        "discriminator_weight_decay_coef",
        "discriminator_logit_weight_decay_coef",
        "value_loss_coef",
        "use_clipped_value_loss",
        "clip_param",
        "entropy_coef",
        "num_learning_epochs",
        "num_mini_batches",
        "learning_rate",
        "schedule",
        "gamma",
        "lam",
        "desired_kl",
        "max_grad_norm",
    )
    for key in shared_algo:
        assert our_agent["algorithm"][key] == reference[f"AmpAlgoCfg.{key}"], key
    assert our_agent["num_steps_per_env"] == reference["G1ParkourPPORunnerCfg.num_steps_per_env"]
    assert our_agent.get("normalizers") in ({}, None)
    assert reference["G1ParkourPPORunnerCfg.empirical_normalization"] is False


def test_agent_normalizers_are_empty_like_both_references(our_agent) -> None:
    reference = mj_ref.agent_fields()
    assert reference["AmpAlgoCfg.class_name"] == "WasabiPPO"
    assert reference["AmpAlgoCfg.discriminator_reward_coef"] == 0.25
    assert reference["AmpAlgoCfg.entropy_coef"] == 0.006
    assert reference["MoEPolicyCfg.num_moe_experts"] == 4
    assert reference["G1ParkourPPORunnerCfg.num_steps_per_env"] == 24
    assert reference["G1ParkourPPORunnerCfg.max_iterations"] == 30000
    assert reference["G1ParkourPPORunnerCfg.empirical_normalization"] is False
    assert our_agent["algorithm"]["class_name"] == "WasabiPPO"
    assert our_agent["algorithm"]["discriminator_reward_coef"] == 0.25
    assert our_agent["algorithm"]["entropy_coef"] == 0.006
    assert our_agent["policy"]["num_moe_experts"] == 4
    assert our_agent["num_steps_per_env"] == 24
    assert our_agent.get("normalizers") in ({}, None)
    assert "policy" not in (our_agent.get("normalizers") or {})
    assert our_agent["policy"]["encoder_configs"]["depth_encoder"]["takeout_input_components"] is True
    assert our_agent["algorithm"]["actor_state_key"] == "amp_policy"
    assert our_agent["algorithm"]["reference_state_key"] == "amp_reference"


def test_instinct_rl_normalizer_cfg_default_is_a_running_zscore_not_identity() -> None:
    """``InstinctRlNormalizerCfg()`` is EmpiricalNormalization. Re-adding it is behavioural."""
    import torch

    from instinct_rl.modules import build_normalizer

    from instinctlab.utils.wrappers.instinct_rl.rl_cfg import InstinctRlNormalizerCfg

    dumped = class_to_dict(InstinctRlNormalizerCfg())
    assert dumped == {"class_name": "EmpiricalNormalization"}
    norm = build_normalizer(input_shape=4, normalizer_class_name=dumped["class_name"], normalizer_kwargs={})
    sample = torch.tensor([[2.0, 4.0, 6.0, 8.0], [4.0, 8.0, 12.0, 16.0]])
    out = norm(sample)
    assert not torch.allclose(out, sample)
    assert (out - sample).abs().max().item() == pytest.approx(15.002493858337402, abs=1e-5)


def test_known_drifts_and_deliberate_tables_are_not_empty() -> None:
    assert len(KNOWN_DRIFTS) == 1
    assert len(DELIBERATE) == 8
    assert len(REFERENCE_DIVERGENCE) == 3
    assert "agent/normalizers" not in KNOWN_DRIFTS
    assert "scene/height_scanner/offset" not in KNOWN_DRIFTS
    assert "reward/dof_vel_limits" not in KNOWN_DRIFTS
    assert "camera/hit_targets" not in KNOWN_DRIFTS
    assert "contact/threshold" not in KNOWN_DRIFTS
    assert "sim/ccd_iterations" not in KNOWN_DRIFTS
    assert "scene/robot/spawn_z" not in KNOWN_DRIFTS
    assert "scene/robot/actuators/delay" not in KNOWN_DRIFTS
    assert "scene/robot/shoe" not in KNOWN_DRIFTS
    assert "amp/symmetric_augmentation" not in KNOWN_DRIFTS
    for table in (KNOWN_DRIFTS, DELIBERATE, REFERENCE_DIVERGENCE):
        for path, (theirs, ours, reason) in table.items():
            assert theirs != ours, path
            assert len(reason) > 40, path


def test_documented_drifts_are_still_present(task, compiled) -> None:
    """Each KNOWN_DRIFTS row must still describe a real difference."""
    from tests import reference_mjlab_parkour as mj_ref

    clip = task.scene.motion_references[0].clip
    assert clip.endswith("parkour_motion_without_run_retargetted.npz")
    assert clip.endswith(".npz")
    source = mj_ref.motion_source()
    assert "parkour_motion_without_run.yaml" in (source.get("filter") or "")
    assert "parkour_motion_without_run_retargetted.npz" not in (source.get("filter") or "")
    assert source["symmetric_augmentation_link_mapping"] is not None


def _lag_partition(robot) -> set[frozenset[str]]:
    """Which joints share one lag draw, read off the wiring rather than the declaration.

    Goes through ``_torque_delay_kwargs`` because that is what decides fusion: mjlab merges
    actuators whose delay settings match, so two configs split for gain reasons land on one
    buffer iff they were handed the same period. Reading ``actuator_group`` directly would
    assert the declaration against itself and stay green if the wiring stopped using it.
    """
    from instinctlab.engines.mjlab.assets import _torque_delay_kwargs, grouped_actuators

    by_period: dict[int, set[str]] = {}
    for names, _, group in grouped_actuators(robot.joint_properties):
        period = _torque_delay_kwargs(robot, group)["delay_update_period"]
        by_period.setdefault(int(period), set()).update(names)
    return {frozenset(members) for members in by_period.values()}


def test_lag_partition_matches_instinctmj(task) -> None:
    """The motor buses that share an actuation lag, joint for joint, against theirs.

    Their side is patterns and ours is names, so theirs is expanded against our joint list --
    which also catches a rename on either side, since an unmatched pattern leaves a joint out
    and the partitions stop being equal.
    """
    from tests import reference_mjlab_parkour as mj_ref

    joints = list(task.robot.joint_names)
    theirs = {
        frozenset(name for name in joints if any(re.fullmatch(pattern, name) for pattern in group))
        for group in mj_ref.delayed_actuator_lag_groups()
    }
    assert frozenset() not in theirs, "a reference pattern group matched no joint of ours"
    assert set().union(*theirs) == set(joints), sorted(set(joints) - set().union(*theirs))
    assert _lag_partition(task.robot) == theirs, {
        "ours": sorted(sorted(g) for g in _lag_partition(task.robot)),
        "theirs": sorted(sorted(g) for g in theirs),
    }
    # The property the alignment was for: no leg spans two draws, on either side.
    leg = {j for j in joints if "_hip_" in j or "_knee_" in j}
    assert sum(1 for g in theirs if g & leg) == 1
    assert sum(1 for g in _lag_partition(task.robot) if g & leg) == 1


def test_isaac_and_mjlab_agree_on_the_motor_buses() -> None:
    """One physical fact, two places that state it -- pinned to each other.

    Isaac builds its actuators from ``_BEYONDMIMIC_JOINT_GROUPS`` patterns and mjlab wires the
    lag off ``JointProperties.actuator_group``. Both already match InstinctMJ today; nothing
    stops one from being edited alone, and the result would be two engines with different lag
    correlation and no failing test.
    """
    from instinctlab.assets.unitree_g1.isaacsim import _BEYONDMIMIC_JOINT_GROUPS, make_g1_29dof_robot_spec

    robot = make_g1_29dof_robot_spec()
    isaac = {
        name: frozenset(j for j in robot.joint_names if any(re.fullmatch(p, j) for p in patterns))
        for name, patterns in _BEYONDMIMIC_JOINT_GROUPS.items()
    }
    ours = {name: frozenset(members) for name, members in robot.actuator_groups()}
    assert ours == isaac, {k: (sorted(ours.get(k, ())), sorted(isaac.get(k, ()))) for k in ours.keys() | isaac.keys()}


def test_depth_history_reset_matches_instinctmj_sensor_buffer() -> None:
    """Ours now zeros the 37-slot ring the way InstinctMJ zeros the camera buffer.

    InstinctMJ's obs term only redraws delay; the sensor ``AsyncCircularBuffer.reset``
    is what actually clears history. This test reads those files (missing → fail)
    and then checks our term, which owns the ring, does the same clear.
    """
    facts = mj_ref.depth_history_reset()
    assert facts["camera_reset_calls_reset_history_buffers"] is True, facts
    assert facts["history_buffers_reset_calls_buffer_reset"] is True, facts
    assert facts["buffer_reset_zeros_buffer"] is True, facts
    assert facts["buffer_reset_zeros_num_pushes"] is True, facts
    assert facts["obs_term_reset_clears_history"] is False, facts
    assert facts["obs_term_reset_resamples_delay"] is True, facts
    assert facts["history_owner"] == "sensor_AsyncCircularBuffer"
    assert facts["camera_source"].endswith("noisy_grouped_raycaster_camera.py")
    assert Path(facts["camera_source"]).is_file()
    assert Path(facts["buffer_reset_source"]).is_file()

    import torch
    from types import SimpleNamespace

    from instinctlab.mdp.observations import DelayedDepthImage
    from instinctlab.spec.sensor import RayCasterRef, RayPatternRef

    sensor = RayCasterRef(
        name="camera",
        attach="torso_link",
        pattern=RayPatternRef(kind="pinhole", width=2, height=2),
        hit=("terrain",),
        max_distance=2.5,
    )
    raw = torch.ones(2, 2, 2, 1)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=SimpleNamespace(
            sensors={"camera": SimpleNamespace(data=SimpleNamespace(output={"distance_to_image_plane": raw}))}
        ),
    )
    cfg = SimpleNamespace(
        params={
            "sensor": sensor,
            "history_skip_frames": 5,
            "num_output_frames": 8,
            "delayed_frame_ranges": (0, 1),
            "history_length": 37,
            "blur_kernel_size": 1,
            "blur_sigma": 0.0,
        }
    )
    term = DelayedDepthImage(cfg, env)
    raw.fill_(1.0)
    for _ in range(4):
        term(env, sensor)
    term.reset(env_ids=torch.tensor([0]))
    assert float(term._history[0].abs().max()) == 0.0
    assert float(term._history[1].abs().max()) > 0.0


def test_every_extractor_has_a_caller() -> None:
    """An extractor with no caller is the defect that hid contact fields last time."""
    public = [
        name
        for name, value in inspect.getmembers(mj_ref, inspect.isfunction)
        if not name.startswith("_") and value.__module__ == mj_ref.__name__
    ]
    source = Path(__file__).read_text()
    unused = [name for name in public if f"mj_ref.{name}" not in source and name != "available"]
    assert not unused, f"extractors with no caller: {unused}"
    assert ast.parse(source)


def test_reference_divergence_dof_vel_limits_tracks_isaac(task) -> None:
    assert "dof_vel_limits" in task.mdp.rewards["rewards"]
    assert task.mdp.rewards["rewards"]["dof_vel_limits"].weight == -1.0
    assert "dof_vel_limits" not in mj_ref.reward_names()
    assert "dof_vel_limits" in main_ref.reward_names()
    assert main_ref.reward_weights()["dof_vel_limits"] == -1.0


def test_reference_divergence_camera_hit_tracks_isaac(task, compiled) -> None:
    from instinctlab.assets.unitree_g1.isaacsim import G1_29DOF_LINKS

    camera = {s.name: s for s in compiled.env_cfg.scene.sensors}["camera"]
    theirs = mj_ref.camera_include_geom_groups()
    assert theirs == (0, 1, 2), theirs
    assert "include_geom_groups" not in mj_ref.sensor_cfgs()["camera"]
    assert camera.include_geom_groups is None
    assert task.scene.ray_caster("camera").hit_bodies() == tuple(G1_29DOF_LINKS)
    assert mj_ref.sensor_cfgs()["camera"]["cfg_class"] == "NoisyGroupedRayCasterCameraCfg"
    assert theirs != tuple(G1_29DOF_LINKS)


def test_reference_divergence_pd_tracks_each_reference(task, compiled) -> None:
    robot = compiled.env_cfg.scene.entities["robot"]
    assert all(type(act).__name__ == "BuiltinPdActuatorCfg" for act in robot.articulation.actuators)
    assert mj_ref.delayed_actuator_lags() == (0, mj_ref.DELAY_MAX_LAG)
    assert task.robot.actuator_delay == (0, 2)
    assert main_ref.effective_robot_actuators()["delayed"] is False


def test_deliberate_rows_are_still_present(task, compiled) -> None:
    """Each row is a *difference*, so both halves have to be asserted.

    Pinning only our side lets a row go stale in silence: if InstinctMJ moved to ``nconmax=256``
    tomorrow the entry would stop describing anything, and a test that reads
    ``compiled.env_cfg.sim.nconmax == 256`` would go on passing. That is the same shape as the
    reference-reader failures in SKILL.md -- an assertion written against ourselves while treating
    the reference as a constant nobody rechecks.
    """
    theirs_sim = mj_ref.sim_overrides()
    theirs_terrain = mj_ref.terrain_recipe()
    theirs_sensors = mj_ref.sensor_cfgs()

    assert "dataset_exhausted" not in task.mdp.terminations
    assert "dataset_exhausted" in mj_ref.termination_names()

    assert task.scene.volume_point("leg_volume_points").velocity == "attach_link"
    # Their VolumePointsCfg has no frame selector at all -- the foot-subtree cvel is wired into the
    # sensor, which is why this is a code drift rather than a config one.
    assert "velocity" not in theirs_sensors["leg_volume_points"]

    # They write 20 columns too; the drift is that their curriculum path builds one column per
    # sub-terrain regardless, and ours honours the declaration.
    assert compiled.env_cfg.scene.terrain.terrain_generator.num_cols == theirs_terrain["num_cols"] == 20
    column_maps = mj_ref.terrain_column_maps()
    assert column_maps["declared_num_cols"] == 20
    assert column_maps["instinctmj_built_num_cols"] == len(column_maps["sub_terrain_names"])
    assert column_maps["ours_built_num_cols"] == 20
    assert column_maps["instinctmj_allocation"] == "one_column_per_type"
    assert column_maps["ours_allocation"] == "isaac_cumulative_proportion"
    assert column_maps["instinctmj_column_to_name"] != column_maps["ours_column_to_name"]
    assert column_maps["instinctmj_column_to_name"].count("pyramid_stairs") == 1
    assert column_maps["ours_column_to_name"].count("pyramid_stairs") == 3

    assert compiled.env_cfg.sim.nconmax == 256
    assert theirs_sim["nconmax"] == 128
    assert compiled.env_cfg.sim.njmax == 768
    assert theirs_sim["njmax"] == 700
    assert compiled.env_cfg.sim.mujoco.ccd_iterations == 500
    assert theirs_sim["ccd_iterations"] == 128

    scanners = {s.name: s for s in compiled.env_cfg.scene.sensors}
    assert scanners["left_height_scanner"].origin_offset == (0.04, 0.0, 20.0)
    assert scanners["left_height_scanner"].max_distance == pytest.approx(1e6)
    assert "origin_offset" not in theirs_sensors["left_height_scanner"]
    assert theirs_sensors["left_height_scanner"]["max_distance"] == 10.0

    # viz: theirs draws collision debug geometry on the terrain, ours omits it. Their *sensor*
    # debug_vis is off in training and only turned on in the play branch, so the row is about
    # the terrain importer, not the sensors.
    assert mj_ref.terrain_importer()["collision_debug_vis"] is True
    assert all(cfg.get("debug_vis") in (False, None) for cfg in theirs_sensors.values())


def test_the_prose_counts_the_drift_table() -> None:
    """The literals that claim a drift count must still be counting this table."""
    import re

    source = Path(__file__).read_text()
    counts = {int(match) for match in re.findall(r"len\(KNOWN_DRIFTS\) == (\d+)", source)}
    assert counts == {len(KNOWN_DRIFTS)} == {1}
