"""Native robot onboarding is explicit, SDK-free, and repository-external."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from instinctlab_engine.assets import (
    AssetRegistry,
    NativeActuatorGroup,
    NativeAssetContractError,
    native_asset_conformance_report,
    native_asset_definition,
    validate_native_actuator_groups,
)


@dataclass(frozen=True)
class _Joint:
    name: str = "joint"
    default_pos: float = 0.0
    stiffness: float = 10.0
    damping: float = 1.0
    armature: float = 0.01
    effort_limit: float = 20.0
    velocity_limit: float = 30.0
    action_scale: float = 0.5


@dataclass(frozen=True)
class _NativeConfig:
    resource: str
    asset_id: str = "external_robot/standard"
    name: str = "external_robot"
    schema_version: str = "dfs_v1"
    root_body: str = "root"
    joint_names: tuple[str, ...] = ("joint",)
    body_names: tuple[str, ...] = ("root", "link")
    frame_names: tuple[str, ...] = ("link",)
    collision_body_names: tuple[str, ...] = ("root", "link")
    joint_properties: tuple[_Joint, ...] = (_Joint(),)
    contact_body_aliases: tuple = ()
    default_root_pos: tuple[float, float, float] = (0.0, 0.0, 0.5)
    default_root_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    soft_joint_pos_limit_factor: float = 0.9
    actuator_delay: tuple[int, int] = (0, 0)
    actuator_model_ids: tuple[str, ...] = ("mjlab.builtin_pd.v1",)
    actuator_group_count: int = 1
    actuator_groups: tuple[NativeActuatorGroup, ...] = (
        NativeActuatorGroup(
            name="0",
            model_id="mjlab.builtin_pd.v1",
            selectors=("joint",),
        ),
    )
    length_unit: str = "m"
    angle_unit: str = "rad"
    effort_unit: str = "N*m"


def _external_module(path: Path) -> ModuleType:
    module = ModuleType("external_robot_assets.mjlab")
    module.INSTINCTLAB_NATIVE_ASSET_API = "0.1"
    config = _NativeConfig(str(path))
    module.native_config = lambda variant: config
    module.entity = lambda variant, robot, **kwargs: (variant, robot, kwargs)
    return module


def _external_registry(module: ModuleType) -> AssetRegistry:
    registry = AssetRegistry(load_entry_points=False)
    registry.register("external_robot", lambda engine, variant: (module, variant))
    return registry


def test_external_asset_resolves_and_conforms_without_a_backend_edit(
    tmp_path: Path,
) -> None:
    resource = tmp_path / "robot.xml"
    resource.write_text(
        """<mujoco><worldbody><body name="root"><geom type="sphere" size="0.1"/>"
        "<body name="link"><joint name="joint"/><geom type="sphere" size="0.1"/>"
        "</body></body></worldbody></mujoco>"""
    )
    module = _external_module(resource)

    definition = native_asset_definition(
        "external_robot/standard",
        "mjlab",
        builder_name="entity",
        resource_field="resource",
        resource_kind="mjcf",
        registry=_external_registry(module),
    )
    report = native_asset_conformance_report(definition)

    assert report["status"] == "ok"
    assert report["canonical_order"] == "dfs"
    assert report["actuator_model_ids"] == ["mjlab.builtin_pd.v1"]
    assert definition.module is module


def test_native_asset_contract_rejects_wrong_api_before_construction(
    tmp_path: Path,
) -> None:
    resource = tmp_path / "robot.xml"
    resource.write_text("<mujoco><worldbody/></mujoco>")
    module = _external_module(resource)
    module.INSTINCTLAB_NATIVE_ASSET_API = "9.0"

    with pytest.raises(NativeAssetContractError, match="required '0.1'"):
        native_asset_definition(
            "external_robot/standard",
            "mjlab",
            builder_name="entity",
            resource_field="resource",
            resource_kind="mjcf",
            registry=_external_registry(module),
        )


def test_native_asset_contract_rejects_noncanonical_joint_order(
    tmp_path: Path,
) -> None:
    resource = tmp_path / "robot.xml"
    resource.write_text(
        """<mujoco><worldbody><body name="root"><geom type="sphere" size="0.1"/>"
        "<body name="link"><joint name="other"/><geom type="sphere" size="0.1"/>"
        "</body></body></worldbody></mujoco>"""
    )
    module = _external_module(resource)

    with pytest.raises(NativeAssetContractError, match="resource DFS order"):
        native_asset_definition(
            "external_robot/standard",
            "mjlab",
            builder_name="entity",
            resource_field="resource",
            resource_kind="mjcf",
            registry=_external_registry(module),
        )


@pytest.mark.parametrize("engine", ("isaacsim", "mjlab"))
def test_builtin_g1_asset_passes_sdk_free_conformance(engine: str) -> None:
    from instinctlab_engine import adapter

    report = adapter(engine).asset_conformance("unitree_g1/popsicle_torsobase_v1")

    assert report["status"] == "ok"
    assert report["joint_count"] == 29
    assert report["units"] == {"length": "m", "angle": "rad", "effort": "N*m"}
    assert len(report["actuator_groups"]) == report["actuator_group_count"]
    assert {
        joint
        for group in report["actuator_groups"]
        for joint in group["joint_names"]
    } == set(
        adapter(engine)
        .robot_spec("unitree_g1/popsicle_torsobase_v1")
        .joint_names
    )


@dataclass
class _NativeActuatorCfg:
    target_names_expr: tuple[str, ...]
    instinctlab_model_id: str


@pytest.mark.parametrize(
    ("model_id", "selectors", "message"),
    (
        ("wrong.model", ("joint",), "was built by model"),
        ("mjlab.builtin_pd.v1", ("other",), "constructed selectors"),
    ),
)
def test_native_actuator_groups_reject_identity_or_selector_drift(
    model_id: str,
    selectors: tuple[str, ...],
    message: str,
) -> None:
    expected = (
        NativeActuatorGroup(
            name="0",
            model_id="mjlab.builtin_pd.v1",
            selectors=("joint",),
        ),
    )
    native = (_NativeActuatorCfg(selectors, model_id),)

    with pytest.raises(NativeAssetContractError, match=message):
        validate_native_actuator_groups(
            "external_robot/standard",
            native,
            ("joint",),
            selector_field="target_names_expr",
            expected_groups=expected,
        )


def test_asset_conformance_does_not_import_torch_or_an_engine_sdk() -> None:
    source = """
import json
import sys
from instinctlab_engine import adapter
reports = {
    engine: adapter(engine).asset_conformance('unitree_g1/popsicle_torsobase_v1')
    for engine in ('isaacsim', 'mjlab')
}
loaded = sorted(
    name for name in sys.modules
    if name == 'torch' or name.startswith('torch.')
    or name == 'isaaclab' or name.startswith('isaaclab.')
    or name == 'mjlab' or name.startswith('mjlab.')
)
print(json.dumps({'reports': reports, 'loaded': loaded}))
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[1],
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["loaded"] == []
    assert {engine: report["status"] for engine, report in payload["reports"].items()} == {
        "isaacsim": "ok",
        "mjlab": "ok",
    }
