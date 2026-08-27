"""Sensor settings read directly from main and InstinctMJ.

These checks cover values that can compile and train while still changing the signal: contact
clock thresholds and the native geom mask, parent exclusion, hop limit, and refresh clock of the
MJLab Perceptive cameras. The reference repositories are parsed rather than imported.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch
from instinctlab.engines.mjlab.scene import _build_ray_caster
from instinctlab.tasks import registry

MAIN_SHADOWING = Path("/root/InstinctLab-main/source/instinctlab/instinctlab/tasks/shadowing")
MJ_SHADOWING = Path("/root/InstinctMJ/src/instinct_mj/tasks/shadowing")
MJ_GROUPED_RAY = Path("/root/InstinctMJ/src/instinct_mj/sensors/grouped_ray_caster")


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _value(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_value(item) for item in node.elts)
    if isinstance(node, ast.List):
        return [_value(item) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_value(node.operand)
    if isinstance(node, ast.BinOp):
        left, right = _value(node.left), _value(node.right)
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
    if isinstance(node, ast.Call) and _call_name(node) == "int" and len(node.args) == 1:
        return int(_value(node.args[0]))
    raise LookupError(f"unsupported reference expression: {ast.unparse(node)}")


def _calls(path: Path, class_name: str) -> list[dict[str, ast.AST]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == class_name:
            result.append({item.arg: item.value for item in node.keywords if item.arg is not None})
    assert result, f"{path} has no {class_name} call"
    return result


def _one_setting(path: Path, class_name: str, field: str):
    values = {_value(call[field]) for call in _calls(path, class_name) if field in call}
    assert len(values) == 1, f"{path}: {class_name}.{field} has values {values}"
    return values.pop()


def _class_setting(path: Path, class_name: str, field: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if (
                isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and item.target.id == field
                and item.value is not None
            ):
                return _value(item.value)
    raise LookupError(f"{path}: {class_name}.{field} not found")


CAMERA_CASES = (
    (
        "Instinct-Perceptive-Shadowing-G1-v0",
        MJ_SHADOWING / "perceptive/perceptive_env_cfg.py",
    ),
    (
        "Instinct-Perceptive-Shadowing-G1-Play-v0",
        MJ_SHADOWING / "perceptive/perceptive_env_cfg.py",
    ),
    (
        "Instinct-Perceptive-Shadowing-G1-OneMotion-v0",
        MJ_SHADOWING / "perceptive/perceptive_env_cfg.py",
    ),
    (
        "Instinct-Perceptive-Shadowing-G1-OneMotion-Play-v0",
        MJ_SHADOWING / "perceptive/perceptive_env_cfg.py",
    ),
    (
        "Instinct-Perceptive-Vae-G1-v0",
        MJ_SHADOWING / "perceptive/perceptive_env_cfg.py",
    ),
    (
        "Instinct-Perceptive-Vae-G1-Play-v0",
        MJ_SHADOWING / "perceptive/perceptive_env_cfg.py",
    ),
    (
        "Instinct-Perceptive-HOI-Shadowing-G1-v0",
        MJ_SHADOWING / "perceptive_hoi/perceptive_env_cfg.py",
    ),
    (
        "Instinct-Perceptive-HOI-Shadowing-G1-Play-v0",
        MJ_SHADOWING / "perceptive_hoi/perceptive_env_cfg.py",
    ),
)


@pytest.mark.skipif(not MJ_GROUPED_RAY.is_dir(), reason="InstinctMJ sensor source is unavailable")
def test_mjlab_parkour_camera_defaults_match_instinctmj_without_a_runtime_source_dependency() -> None:
    task = registry.spec("Instinct-Parkour-Target-G1")
    camera = _build_ray_caster(task.scene.ray_caster("camera"), task.sim.profile_for("mjlab"))
    ray_cfg = MJ_GROUPED_RAY / "grouped_ray_caster_cfg.py"
    camera_cfg = MJ_GROUPED_RAY / "grouped_ray_caster_camera_cfg.py"

    assert camera.mesh_filter_max_hops == _class_setting(
        ray_cfg, "GroupedRayCasterCfg", "mesh_filter_max_hops"
    )
    assert camera.mesh_filter_epsilon == _class_setting(ray_cfg, "GroupedRayCasterCfg", "mesh_filter_epsilon")
    assert camera.update_period == _class_setting(camera_cfg, "GroupedRayCasterCameraCfg", "update_period")


@pytest.mark.skipif(not MJ_SHADOWING.is_dir(), reason="InstinctMJ reference checkout is unavailable")
@pytest.mark.parametrize(("task_id", "reference_path"), CAMERA_CASES)
def test_mjlab_perceptive_camera_native_settings_match_instinctmj(task_id: str, reference_path: Path) -> None:
    task = registry.spec(task_id)
    camera_ref = task.scene.ray_caster("camera")
    camera = _build_ray_caster(camera_ref, task.sim.profile_for("mjlab"))

    assert camera.include_geom_groups == _one_setting(
        reference_path, "NoisyGroupedRayCasterCameraCfg", "include_geom_groups"
    )
    assert camera.exclude_parent_body is _one_setting(
        reference_path, "NoisyGroupedRayCasterCameraCfg", "exclude_parent_body"
    )
    assert camera.mesh_filter_max_hops == _one_setting(
        reference_path, "NoisyGroupedRayCasterCameraCfg", "mesh_filter_max_hops"
    )
    assert camera.update_period == pytest.approx(
        _one_setting(reference_path, "NoisyGroupedRayCasterCameraCfg", "update_period")
    )


@pytest.mark.skipif(
    not MAIN_SHADOWING.is_dir() or not MJ_SHADOWING.is_dir(),
    reason="shadowing reference checkouts are unavailable",
)
@pytest.mark.parametrize(
    ("task_id", "relative_path"),
    (
        ("Instinct-Perceptive-Shadowing-G1-v0", "perceptive/perceptive_env_cfg.py"),
        ("Instinct-Perceptive-Shadowing-G1-Play-v0", "perceptive/perceptive_env_cfg.py"),
        ("Instinct-Perceptive-HOI-Shadowing-G1-v0", "perceptive_hoi/perceptive_env_cfg.py"),
        ("Instinct-Perceptive-HOI-Shadowing-G1-Play-v0", "perceptive_hoi/perceptive_env_cfg.py"),
    ),
)
def test_perceptive_contact_clock_threshold_matches_each_reference(task_id: str, relative_path: str) -> None:
    contact = registry.spec(task_id).scene.contact_sensors[0]
    main_threshold = _one_setting(MAIN_SHADOWING / relative_path, "ContactSensorCfg", "force_threshold")
    mj_threshold = _one_setting(
        MJ_SHADOWING / relative_path,
        "ForceThresholdContactSensorCfg",
        "force_threshold",
    )

    assert contact.for_engine("isaacsim").air_time_force_threshold == main_threshold == 10.0
    assert contact.for_engine("mjlab").air_time_force_threshold == mj_threshold == 1.0


@pytest.mark.skipif(not MJ_SHADOWING.is_dir(), reason="InstinctMJ reference checkout is unavailable")
def test_mjlab_perceptive_camera_refresh_clock_matches_instinctmj() -> None:
    """Fixed-time probe: 1/60 s refreshes on the fourth 5 ms tick and retains stale poses before it."""
    task = registry.spec("Instinct-Perceptive-Shadowing-G1-v0")
    camera_ref = task.scene.ray_caster("camera")
    cfg = _build_ray_caster(camera_ref, task.sim.profile_for("mjlab"))
    sensor = cfg.build()

    period = _one_setting(
        MJ_SHADOWING / "perceptive/perceptive_env_cfg.py",
        "NoisyGroupedRayCasterCameraCfg",
        "update_period",
    )
    sensor._update_period_s = period
    sensor._elapsed_since_refresh = torch.full((2,), period)
    sensor._refresh_mask = torch.ones(2, dtype=torch.bool)
    sensor._reported_cam_pos = torch.zeros(2, 3)
    sensor._reported_cam_quat = torch.zeros(2, 4)
    sensor.frame_sequence = 0

    first_pos = torch.ones(2, 3)
    first_quat = torch.ones(2, 4)
    sensor._refresh_reported_pose(first_pos, first_quat)
    assert sensor._refresh_mask.tolist() == [True, True]
    assert torch.equal(sensor._reported_cam_pos, first_pos)

    next_pos = torch.full((2, 3), 2.0)
    next_quat = torch.full((2, 4), 2.0)
    for _ in range(3):
        sensor.update(0.005)
    sensor._refresh_reported_pose(next_pos, next_quat)
    assert sensor._refresh_mask.tolist() == [False, False]
    assert torch.equal(sensor._reported_cam_pos, first_pos)

    sensor.update(0.005)
    sensor._refresh_reported_pose(next_pos, next_quat)
    assert sensor._refresh_mask.tolist() == [True, True]
    assert torch.equal(sensor._reported_cam_pos, next_pos)
    assert torch.allclose(
        sensor._elapsed_since_refresh,
        torch.full((2,), math.fmod(0.02, period)),
        atol=1.0e-7,
    )
