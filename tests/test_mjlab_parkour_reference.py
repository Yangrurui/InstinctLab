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
    "viz": (
        "collision_debug_vis / debug_vis on",
        "omitted",
        "Play-time markers only; they do not enter the observation or the reward.",
    ),
}

KNOWN_DRIFTS: dict[str, tuple[str, str, str]] = {
    "reward/dof_vel_limits": (
        "absent",
        "weight=-1.0, soft_ratio=0.9",
        (
            "Copied from Isaac parkour. Extra velocity-limit penalty on every step; late-training "
            "cost is small unless joints saturate, then it reshapes the action distribution."
        ),
    ),
    "scene/robot/spawn_z": (
        str(mj_ref.SPAWN_Z),
        "0.82",
        "8 cm lower root spawn. First-contact and fall-recovery change for ~0.4 s; can shorten early episodes.",
    ),
    "scene/robot/actuators/delay": (
        f"delay_min_lag=0, delay_max_lag={mj_ref.DELAY_MAX_LAG} (per episode)",
        "delay_max_lag=0",
        (
            "InstinctMJ and Isaac both delay torque 0–2 steps. Removing lag makes the plant easier "
            "(would lengthen episodes vs InstinctMJ, not shorten them)."
        ),
    ),
    "scene/robot/shoe": (
        mj_ref.SHOE_XML_SUFFIX,
        "g1_29dof_torsobase_popsicle.xml (no shoe)",
        (
            "Volume-point z and feet_at_plane offset 0.058 were tuned for the shoe sole. Same numbers "
            "on a bare ankle change stance height and penetration volume."
        ),
    ),
    "contact/threshold": (
        "ForceThresholdContactSensorCfg force_threshold=1.0 N; illegal/undesired also 1 N",
        "in_contact() from contact duration / found; no Newton cutoff",
        (
            "A light touch now counts. base_contact and undesired_contacts fire more often — "
            "the shape that once left illegal_contact dead when found was missing, in reverse."
        ),
    ),
    "scene/height_scanner/offset": (
        f"{mj_ref.SCANNER_ORIGIN_OFFSET}, max_distance={mj_ref.SCANNER_MAX_DISTANCE}",
        "(0.04, 0.0, 20.0), max_distance=1e6",
        (
            "Isaac-style sky rays. InstinctMJ starts at the ankle and clips at 10 m. feet_at_plane "
            "hit_z differs on slopes and near walls."
        ),
    ),
    "sim/ccd_iterations": (
        str(mj_ref.CCD_ITERATIONS),
        "500 (PROFILE_DEFAULTS)",
        "Parkour override is 128. More CCD iterations change contact timing, not the MDP terms.",
    ),
    "agent/normalizers": (
        "empirical_normalization=False, empty normalizers",
        "EmpiricalNormalization on policy and critic",
        (
            "Neither InstinctMJ nor the legacy Isaac runner normalises. Running means rescale the "
            "768-d proprioception the MoE sees; learning dynamics diverge from both references."
        ),
    ),
    "motion/source": (
        "AMASS directory + parkour_motion_without_run.yaml",
        "single npz parkour_motion_without_run_retargetted.npz",
        "AMP discriminator trains on a different reference distribution.",
    ),
    "camera/hit_targets": (
        "default geom groups (0, 1, 2); no named-body list",
        "terrain + G1_29DOF_LINKS by body name",
        (
            "Group 2 is the visual shoe, group 3 the collision capsule. Named-body hits change "
            "self-occlusion in the depth image the Conv2d encoder consumes."
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
    assert compiled.env_cfg.sim.njmax == overrides["njmax"] == 700
    assert compiled.env_cfg.sim.contact_sensor_maxmatch == overrides["contact_sensor_maxmatch"] == 128
    assert compiled.env_cfg.sim.mujoco.iterations == overrides["iterations"] == 10
    assert compiled.env_cfg.sim.mujoco.ls_iterations == overrides["ls_iterations"] == 20
    assert compiled.env_cfg.sim.nconmax == 256
    assert overrides["nconmax"] == mj_ref.NCONMAX
    assert compiled.env_cfg.sim.mujoco.ccd_iterations == 500
    assert overrides["ccd_iterations"] == mj_ref.CCD_ITERATIONS
    assert overrides["init_pos"][2] == mj_ref.SPAWN_Z


def test_compiled_spawn_delay_shoe_and_scanners_are_the_documented_drifts(task, compiled) -> None:
    robot = compiled.env_cfg.scene.entities["robot"]
    assert robot.init_state.pos[2] == pytest.approx(0.82)
    assert mj_ref.sim_overrides()["init_pos"][2] == pytest.approx(0.9)
    lags = [getattr(act, "delay_max_lag", 0) for act in robot.articulation.actuators]
    assert max(lags) == 0
    assert mj_ref.delayed_actuator_lags() == (0, mj_ref.DELAY_MAX_LAG)
    asset = task.robot.asset_for("mjlab").path
    assert asset.endswith("g1_29dof_torsobase_popsicle.xml")
    assert mj_ref.SHOE_XML_SUFFIX not in asset
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


def test_agent_shared_hyperparameters_match_except_documented_normalizers(our_agent) -> None:
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
    assert "policy" in our_agent["normalizers"]
    assert "critic" in our_agent["normalizers"]
    assert our_agent["normalizers"]["policy"]["class_name"] == "EmpiricalNormalization"
    assert our_agent["policy"]["encoder_configs"]["depth_encoder"]["takeout_input_components"] is True
    assert our_agent["algorithm"]["actor_state_key"] == "amp_policy"
    assert our_agent["algorithm"]["reference_state_key"] == "amp_reference"


def test_known_drifts_and_deliberate_tables_are_not_empty() -> None:
    assert len(KNOWN_DRIFTS) == 10
    assert len(DELIBERATE) == 5
    for table in (KNOWN_DRIFTS, DELIBERATE):
        for path, (theirs, ours, reason) in table.items():
            assert theirs != ours, path
            assert len(reason) > 40, path


def test_documented_drifts_are_still_present(task, compiled) -> None:
    """Each KNOWN_DRIFTS row must still describe a real difference."""
    rewards = task.mdp.rewards["rewards"]
    assert "dof_vel_limits" in rewards
    assert "dof_vel_limits" not in mj_ref.reward_names()
    assert task.robot.default_root_pos[2] == pytest.approx(0.82)
    robot = compiled.env_cfg.scene.entities["robot"]
    assert max(getattr(act, "delay_max_lag", 0) for act in robot.articulation.actuators) == 0
    assert not task.robot.asset_for("mjlab").path.endswith(mj_ref.SHOE_XML_SUFFIX)
    assert "threshold" not in rewards["undesired_contacts"].params
    scanners = {s.name: s for s in compiled.env_cfg.scene.sensors}
    assert scanners["left_height_scanner"].origin_offset == (0.04, 0.0, 20.0)
    assert compiled.env_cfg.sim.mujoco.ccd_iterations == 500
    from instinctlab.tasks.parkour.config.g1.agents.instinct_rl_parkour_cfg import G1ParkourTargetPPORunnerCfg

    agent = G1ParkourTargetPPORunnerCfg()
    assert "policy" in class_to_dict(agent)["normalizers"]
    assert task.scene.motion_references[0].clip.endswith(".npz")
    assert scanners["camera"].include_geom_groups is None


def test_deliberate_rows_are_still_present(task, compiled) -> None:
    assert "dataset_exhausted" not in task.mdp.terminations
    assert task.scene.volume_point("leg_volume_points").velocity == "attach_link"
    assert compiled.env_cfg.scene.terrain.terrain_generator.num_cols == 20
    assert compiled.env_cfg.sim.nconmax == 256


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
    """The literals that claim ten drifts must still be counting this table."""
    import re

    source = Path(__file__).read_text()
    counts = {int(match) for match in re.findall(r"len\(KNOWN_DRIFTS\) == (\d+)", source)}
    assert counts == {len(KNOWN_DRIFTS)} == {10}
