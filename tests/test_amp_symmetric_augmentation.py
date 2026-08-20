"""AMP left-right symmetric augmentation: IR, name maps, operators, both engines.

The source integer tables are written against two different joint orders
(InstinctMJ = canonical DFS, main parkour = PhysX BFS). Pasting either table
onto the portable clip (canonical order, published npz is legs-first) is the
silent failure this file exists to catch. Maps are names; indices are resolved
in *our* order.
"""

from __future__ import annotations

import pytest

from instinctlab.spec.sensor import MotionReferenceRef, SymmetricAugmentationSpec


def _tiny_aug(**overrides) -> SymmetricAugmentationSpec:
    fields = dict(
        joint_swaps={"left_hip_pitch_joint": "right_hip_pitch_joint", "right_hip_pitch_joint": "left_hip_pitch_joint"},
        joint_signs={"left_hip_pitch_joint": 1, "right_hip_pitch_joint": 1},
        link_swaps={"left_knee_link": "right_knee_link", "right_knee_link": "left_knee_link"},
    )
    fields.update(overrides)
    return SymmetricAugmentationSpec(**fields)


def _ref(**overrides) -> MotionReferenceRef:
    fields = dict(
        name="motion_reference",
        clip="clip.npz",
        joints=("left_hip_pitch_joint", "right_hip_pitch_joint"),
        links=("left_knee_link", "right_knee_link"),
    )
    fields.update(overrides)
    return MotionReferenceRef(**fields)


def test_mirroring_is_off_unless_a_task_turns_it_on() -> None:
    sensor = _ref()
    assert sensor.symmetric_augmentation is None
    assert not hasattr(sensor, "symmetric_augmentation_joint_mapping")


def test_from_left_right_swaps_sides_and_keeps_midline() -> None:
    joints = ("waist_pitch_joint", "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint")
    links = ("pelvis", "torso_link", "left_knee_link", "right_knee_link")
    spec = SymmetricAugmentationSpec.from_left_right(joints, links)
    assert spec.joint_swaps["left_hip_pitch_joint"] == "right_hip_pitch_joint"
    assert spec.joint_swaps["right_hip_pitch_joint"] == "left_hip_pitch_joint"
    assert spec.joint_swaps["waist_pitch_joint"] == "waist_pitch_joint"
    assert spec.joint_swaps["waist_yaw_joint"] == "waist_yaw_joint"
    assert spec.link_swaps["left_knee_link"] == "right_knee_link"
    assert spec.link_swaps["pelvis"] == "pelvis"
    assert spec.link_swaps["torso_link"] == "torso_link"
    assert spec.joint_signs["waist_pitch_joint"] == 1
    assert spec.joint_signs["waist_yaw_joint"] == -1
    assert spec.joint_signs["left_hip_pitch_joint"] == 1


def test_from_left_right_flips_roll_and_yaw_only() -> None:
    joints = (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_elbow_joint",
        "right_elbow_joint",
    )
    spec = SymmetricAugmentationSpec.from_left_right(joints, ("pelvis",))
    assert spec.joint_signs["left_hip_pitch_joint"] == 1
    assert spec.joint_signs["left_knee_joint"] == 1
    assert spec.joint_signs["left_elbow_joint"] == 1
    assert spec.joint_signs["left_hip_roll_joint"] == -1
    assert spec.joint_signs["right_hip_roll_joint"] == -1


def test_a_one_way_swap_is_refused() -> None:
    with pytest.raises(ValueError, match="involution"):
        _tiny_aug(
            joint_swaps={"left_hip_pitch_joint": "right_hip_pitch_joint", "right_hip_pitch_joint": "left_knee_joint"}
        )


def test_empty_swaps_are_refused_use_none_to_disable() -> None:
    with pytest.raises(ValueError, match="empty"):
        _tiny_aug(joint_swaps={}, joint_signs={})
    with pytest.raises(ValueError, match="empty"):
        _tiny_aug(link_swaps={})


def test_signs_must_cover_exactly_the_swapped_joints() -> None:
    with pytest.raises(ValueError, match="exactly the joints"):
        _tiny_aug(joint_signs={"left_hip_pitch_joint": 1})
    with pytest.raises(ValueError, match=r"\+1 or -1"):
        _tiny_aug(
            joint_signs={"left_hip_pitch_joint": 1, "right_hip_pitch_joint": 0},
        )


def test_an_unpaired_left_name_is_loud() -> None:
    with pytest.raises(ValueError, match="not in the name list"):
        SymmetricAugmentationSpec.from_left_right(
            ("left_hip_pitch_joint", "waist_pitch_joint"),
            ("pelvis",),
        )


def test_the_sensor_refuses_a_map_whose_names_are_not_its_joints() -> None:
    with pytest.raises(ValueError, match="do not match sensor joints"):
        _ref(
            joints=("waist_pitch_joint",),
            links=("left_knee_link", "right_knee_link"),
            symmetric_augmentation=_tiny_aug(),
        )


def test_the_sensor_accepts_a_matching_name_map() -> None:
    sensor = _ref(symmetric_augmentation=_tiny_aug())
    assert sensor.symmetric_augmentation is not None
    assert sensor.symmetric_augmentation.joint_swaps["left_hip_pitch_joint"] == "right_hip_pitch_joint"
