"""The Viser player has to work from catalog files and vocab attributes, not from an engine."""

from __future__ import annotations

import inspect
import torch
from types import SimpleNamespace

from instinctlab.assets.unitree_g1.isaacsim import make_g1_29dof_robot_spec
from instinctlab.play import viser as play_viser
from instinctlab.play.viser import _pin_velocity_command, visual_asset_path, visual_meshes_from_mjcf


def test_the_g1_catalog_names_a_visual_mesh_for_every_moving_link() -> None:
    robot = make_g1_29dof_robot_spec()
    meshes = visual_meshes_from_mjcf(visual_asset_path(robot))
    bodies = {mesh.body for mesh in meshes}
    assert "torso_link" in bodies
    assert "left_ankle_roll_link" in bodies
    assert "right_wrist_yaw_link" in bodies
    assert all(mesh.path.is_file() for mesh in meshes)
    assert len(meshes) >= 20


def test_velocity_command_pins_isaac_lab_fields() -> None:
    term = SimpleNamespace(
        vel_command_b=torch.zeros(2, 3),
        is_heading_env=torch.ones(2, dtype=torch.bool),
        is_standing_env=torch.ones(2, dtype=torch.bool),
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(command_manager=SimpleNamespace(_terms={"base_velocity": term})))
    _pin_velocity_command(env, 0.8, -0.1, 0.3)
    assert not bool(term.is_heading_env.any())
    assert not bool(term.is_standing_env.any())
    assert torch.equal(term.vel_command_b[0], torch.tensor([0.8, -0.1, 0.3]))


def test_velocity_command_pins_mjlab_fields() -> None:
    term = SimpleNamespace(
        _command=torch.zeros(1, 3),
        _standing=torch.ones(1, dtype=torch.bool),
        _heading=torch.ones(1, dtype=torch.bool),
        _time_left=torch.zeros(1),
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(command_manager=SimpleNamespace(_terms=(("base_velocity", term),))))
    _pin_velocity_command(env, 0.4, 0.0, -0.2)
    assert not bool(term._standing.any())
    assert float(term._time_left[0]) > 1.0e6
    assert torch.equal(term._command[0], torch.tensor([0.4, 0.0, -0.2]))


def test_play_uses_mjlab_viser_play_viewer() -> None:
    """Isaac play used to ship a second mesh-streaming viewer; both engines share this one."""
    source = inspect.getsource(play_viser.play_with_viser)
    assert "ViserPlayViewer" in source
    assert "set_debug_image_sink" in source
    assert "enable_pose_command_debug_vis" in source
    assert "add_mesh_trimesh" not in source
    assert "_DepthViserPlayViewer" not in source


def test_play_enables_instinctmj_depth_debug_vis() -> None:
    term = SimpleNamespace(params={"sensor": "camera", "debug_vis": False})
    env_cfg = SimpleNamespace(observations={"policy": SimpleNamespace(terms={"depth_image": term})})
    play_viser.enable_depth_image_debug_vis(env_cfg)
    assert term.params["debug_vis"] is True


def test_play_patches_debug_vis_on_the_live_observation_manager() -> None:
    """The manager deep-copies env cfg; writing env.cfg after make_env does not reach compute()."""
    term = SimpleNamespace(params={"sensor": "camera"})
    manager = SimpleNamespace(
        _group_obs_term_names={"policy": ["joint_pos", "depth_image"]},
        _group_obs_term_cfgs={"policy": [SimpleNamespace(params={}), term]},
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(observation_manager=manager, cfg=SimpleNamespace(observations={})))
    play_viser.enable_depth_image_debug_vis(env)
    assert term.params["debug_vis"] is True


def test_play_turns_on_depth_debug_vis_after_the_env_exists() -> None:
    """make_env() computes observations; cv2.imshow during that reset aborts without a display."""
    from pathlib import Path

    play = (Path(__file__).resolve().parents[1] / "scripts" / "play.py").read_text()
    assert play.index("compiled.make_env()") < play.index("enable_depth_image_debug_vis(")
    assert play.index("compiled.make_env()") < play.index("enable_pose_command_debug_vis(")


def test_play_enables_instinctmj_pose_command_debug_vis() -> None:
    term = SimpleNamespace(debug_vis=False, patch_vis=True)
    env_cfg = SimpleNamespace(commands={"base_velocity": term})
    play_viser.enable_pose_command_debug_vis(env_cfg)
    assert term.debug_vis is True
    assert term.patch_vis is False


def test_play_patches_pose_command_debug_vis_on_the_live_term() -> None:
    cfg = SimpleNamespace(debug_vis=False, patch_vis=True)
    term = SimpleNamespace(cfg=cfg)
    env = SimpleNamespace(unwrapped=SimpleNamespace(command_manager=SimpleNamespace(_terms={"base_velocity": term})))
    play_viser.enable_pose_command_debug_vis(env)
    assert cfg.debug_vis is True
    assert cfg.patch_vis is False
