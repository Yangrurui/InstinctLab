"""Native G1 selection belongs to the asset layer, not the Isaac adapter."""

from __future__ import annotations

import inspect

from instinctlab.assets.unitree_g1 import isaacsim
from instinctlab_engine.assets import native_asset_module
from instinctlab_engine_isaacsim import assets

ASSET_IDS = (
    "unitree_g1/popsicle_torsobase_v1",
    "unitree_g1/popsicle_torsobase_shadowing_v1",
    "unitree_g1/popsicle_torsobase_parkour_v1",
    "unitree_g1/popsicle_torsobase_locked_arms_v1",
)


def test_each_g1_task_variant_is_registered_inside_the_isaac_asset_module() -> None:
    expected = {
        "popsicle_torsobase_v1": "G1_29DOF_TORSOBASE_POPSICLE_CFG",
        "popsicle_torsobase_shadowing_v1": "G1_29DOF_TORSOBASE_POPSICLE_SHADOWING_CFG",
        "popsicle_torsobase_parkour_v1": "G1_29DOF_TORSOBASE_POPSICLE_PARKOUR_CFG",
        "popsicle_torsobase_locked_arms_v1": (
            "G1_15DOF_TORSOBASE_POPSICLE_LOCKED_ARMS_CFG"
        ),
    }
    assert isaacsim.ARTICULATIONS == expected


def test_generic_loader_resolves_variants_without_a_g1_dependency_in_engine() -> None:
    for asset_id in ASSET_IDS:
        module, variant = native_asset_module(asset_id, "isaacsim")
        assert module is isaacsim
        assert variant in isaacsim.ARTICULATIONS

    source = inspect.getsource(assets)
    assert "unitree_g1" not in source
    assert "G1_" not in source
    assert not hasattr(assets, "apply_robot_spec")
    assert not hasattr(assets, "delayed_actuators")
