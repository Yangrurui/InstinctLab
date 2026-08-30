"""The motion-reference IR: what it states and what it refuses.

Cheap: no GPU, no clip, no kinematics. A silent restart, an xyzw quaternion, or a
positional joint fallback would all look like a working declaration.
"""

from __future__ import annotations

import pytest

from instinctlab.spec.sensor import MotionReferenceRef


def _ref(**overrides) -> MotionReferenceRef:
    fields = dict(
        name="motion_reference",
        clip="clip.npz",
        joints=("hip", "knee"),
        links=("root", "foot"),
    )
    fields.update(overrides)
    return MotionReferenceRef(**fields)


def test_the_ir_states_the_frames_and_the_exhaustion_policy() -> None:
    sensor = _ref()
    assert sensor.quaternion == "wxyz"
    assert sensor.exhaustion == "freeze_last_and_flag"
    assert sensor.velocity_method == "frontward"
    assert sensor.data_start_from == "one_frame_interval"
    assert tuple(sensor.joints) == ("hip", "knee")
    assert tuple(sensor.links) == ("root", "foot")


def test_xyzw_is_refused() -> None:
    with pytest.raises(ValueError, match="wxyz"):
        _ref(quaternion="xyzw")  # type: ignore[arg-type]


def test_silent_restart_is_refused() -> None:
    with pytest.raises(ValueError, match="silent restart"):
        _ref(exhaustion="restart")  # type: ignore[arg-type]


def test_empty_names_and_duplicate_joints_fail() -> None:
    with pytest.raises(ValueError, match="no clip"):
        _ref(clip="")
    with pytest.raises(ValueError, match="no joint"):
        _ref(joints=())
    with pytest.raises(ValueError, match="no link"):
        _ref(links=())
    with pytest.raises(ValueError, match="repeats a joint"):
        _ref(joints=("hip", "hip"))
    with pytest.raises(ValueError, match="repeats a link"):
        _ref(links=("foot", "foot"))


def test_start_range_is_a_fraction_of_clip_duration() -> None:
    with pytest.raises(ValueError, match="start_range"):
        _ref(start_range=(-0.1, 0.5))
    with pytest.raises(ValueError, match="start_range"):
        _ref(start_range=(0.8, 0.2))
    with pytest.raises(ValueError, match="start_range"):
        _ref(start_range=(0.0, 1.1))
    assert _ref(start_range=(0.0, 0.9)).start_range == (0.0, 0.9)
