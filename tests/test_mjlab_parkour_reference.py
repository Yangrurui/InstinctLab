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
        "Isaac's cumulative-proportion allocation. InstinctMJ ignores num_cols in curriculum.",
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
    "contact/sensor": (
        "ForceThresholdContactSensorCfg force_threshold=1.0 N (air-time also 1 N)",
        "ContactSensorCfg fields=(found, force); air-time from found",
        (
            "feet_air_time stays portable. Isaac's sensor already gates air-time at 1 N "
            "(ContactSensorCfg.force_threshold default). base_contact / undesired_contacts "
            "are per-engine 1 N terms against each engine's own force quantity."
        ),
    ),
    "motion/source": (
        "AmassMotionCfg yaml filter → parkour_motion_without_run.yaml",
        "MotionReferenceRef clip=…parkour_motion_without_run_retargetted.npz",
        (
            "Shipped yaml lists exactly one file, the same retargetted npz we load "
            "(verified 2026-08-20: selected_files=['parkour_motion_without_run_retargetted.npz'], "
            "18982 frames @ 50 Hz). Residual AMP drift is symmetric augmentation and loader path, "
            "not a different clip inventory."
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
    assert reference_sensors["contact_forces"]["cfg_class"] == "ForceThresholdContactSensorCfg"
    assert reference_sensors["contact_forces"]["force_threshold"] == 1.0
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
    gen = compiled.env_cfg.scene.terrain.terrain_generator
    assert gen.horizontal_scale == 0.07
    assert gen.num_cols == 20
    assert gen.curriculum is True
    assert gen.sub_terrains["pyramid_stairs"].step_width == 0.35
    assert gen.sub_terrains["pyramid_stairs_high"].step_width == 1.54
    assert gen.sub_terrains["perlin_rough"].border_width == 1.0
    assert gen.sub_terrains["boxes"].border_width == 1.0


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
    assert len(KNOWN_DRIFTS) == 2
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

    scanners = {s.name: s for s in compiled.env_cfg.scene.sensors}
    assert "found" in scanners["contact_forces"].fields
    assert mj_ref.sensor_cfgs()["contact_forces"]["cfg_class"] == "ForceThresholdContactSensorCfg"
    assert task.scene.motion_references[0].clip.endswith(".npz")
    assert mj_ref.motion_source()["symmetric_augmentation_link_mapping"] is not None


def test_reference_divergence_dof_vel_limits_tracks_isaac(task) -> None:
    assert "dof_vel_limits" in task.mdp.rewards["rewards"]
    assert task.mdp.rewards["rewards"]["dof_vel_limits"].weight == -1.0
    assert "dof_vel_limits" not in mj_ref.reward_names()
    assert "dof_vel_limits" in main_ref.reward_names()
    assert main_ref.reward_weights()["dof_vel_limits"] == -1.0


def test_reference_divergence_camera_hit_tracks_isaac(task, compiled) -> None:
    from instinctlab.assets.unitree_g1.isaacsim import G1_29DOF_LINKS

    camera = {s.name: s for s in compiled.env_cfg.scene.sensors}["camera"]
    assert camera.include_geom_groups is None
    assert task.scene.ray_caster("camera").hit_bodies() == tuple(G1_29DOF_LINKS)
    assert mj_ref.sensor_cfgs()["camera"]["cfg_class"] == "NoisyGroupedRayCasterCameraCfg"


def test_reference_divergence_pd_tracks_each_reference(task, compiled) -> None:
    robot = compiled.env_cfg.scene.entities["robot"]
    assert all(type(act).__name__ == "BuiltinPdActuatorCfg" for act in robot.articulation.actuators)
    assert task.robot.actuator_delay == (0, 2)


def test_deliberate_rows_are_still_present(task, compiled) -> None:
    assert "dataset_exhausted" not in task.mdp.terminations
    assert task.scene.volume_point("leg_volume_points").velocity == "attach_link"
    assert compiled.env_cfg.scene.terrain.terrain_generator.num_cols == 20
    assert compiled.env_cfg.sim.nconmax == 256
    assert compiled.env_cfg.sim.njmax == 768
    scanners = {s.name: s for s in compiled.env_cfg.scene.sensors}
    assert scanners["left_height_scanner"].origin_offset == (0.04, 0.0, 20.0)
    assert scanners["left_height_scanner"].max_distance == pytest.approx(1e6)
    assert compiled.env_cfg.sim.mujoco.ccd_iterations == 500


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


def test_the_prose_counts_the_drift_table() -> None:
    """The literals that claim a drift count must still be counting this table."""
    import re

    source = Path(__file__).read_text()
    counts = {int(match) for match in re.findall(r"len\(KNOWN_DRIFTS\) == (\d+)", source)}
    assert counts == {len(KNOWN_DRIFTS)} == {2}
