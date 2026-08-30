"""Guard: the contact-sensor claims hold, and the reader normalises what actually differs.

``compat.sensors`` asserts three things about the engines: that the four air/contact-time tensors
share names and rank, that force history is laid out with time and element axes swapped between
them, and that the force values are not comparable. The first two decide whether a portable term
computes the right number; the third decides whether it should exist at all.

Those claims are checked against the installed engines below -- mjlab's ``ContactData`` by
introspection, Isaac Lab's ``ContactSensorData`` with ``ast``, since importing it needs ``carb``.
The reader is then exercised against stubs whose shapes come from those same declarations, which is
as close to the real thing as it gets without a simulator.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib
import re
import torch

import pytest

from instinctlab.compat.errors import PortabilityError
from instinctlab.compat.sensors import (
    air_time,
    contact_force_history,
    contact_time,
    element_ids,
    element_names,
    forget,
    in_contact,
    sensor_engine,
)
from instinctlab.spec.sensor import ContactSensorRef

_TIMING_ATTRS = ("current_air_time", "last_air_time", "current_contact_time", "last_contact_time")
_SHAPE = re.compile(r"Shape is \(([^)]*)\)")


def _isaac_contact_fields() -> dict[str, str]:
    """``ContactSensorData`` member -> declared shape, without importing ``carb``."""
    isaaclab = pytest.importorskip("isaaclab")
    source = pathlib.Path(isaaclab.__file__).parent / "sensors/contact_sensor/contact_sensor_data.py"
    if not source.is_file():  # pragma: no cover - upstream layout change
        pytest.skip(f"Isaac Lab contact sensor data not found at {source}")
    class_def = next(
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.ClassDef) and node.name == "ContactSensorData"
    )
    fields: dict[str, str] = {}
    body = class_def.body
    for index, node in enumerate(body):
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
            continue
        doc = ""
        following = body[index + 1] if index + 1 < len(body) else None
        if isinstance(following, ast.Expr) and isinstance(following.value, ast.Constant):
            doc = " ".join(str(following.value.value).split())
        match = _SHAPE.search(doc)
        fields[node.target.id] = match.group(1) if match else ""
    return fields


def _mjlab_contact_fields() -> dict[str, str]:
    """``ContactData`` field -> the shape stated in its attribute docstring."""
    module = pytest.importorskip("mjlab.sensor.contact_sensor")
    source = pathlib.Path(module.__file__).read_text()
    fields: dict[str, str] = {}
    for field in dataclasses.fields(module.ContactData):
        match = re.search(rf"\n\s+{field.name}:[^\n]*\n\s+\"\"\"\s*(\[[^\]]*\])", source)
        fields[field.name] = match.group(1) if match else ""
    return fields


"""
The claims about the engines.
"""


@pytest.mark.parametrize("attr", _TIMING_ATTRS)
def test_timing_tensors_share_a_name_on_both_engines(attr: str) -> None:
    """The convergence the portable step-timing terms rest on."""
    assert attr in _isaac_contact_fields()
    assert attr in _mjlab_contact_fields()


@pytest.mark.parametrize("attr", _TIMING_ATTRS)
def test_timing_tensors_are_two_dimensional_on_both_engines(attr: str) -> None:
    """Both are (env, element), so one indexing expression works for both."""
    assert _isaac_contact_fields()[attr].count(",") == 1, "Isaac Lab timing tensor is no longer (N, B)"
    assert _mjlab_contact_fields()[attr].count(",") == 1, "mjlab timing tensor is no longer [B, P]"


def test_force_history_axis_order_still_differs() -> None:
    """The transposition in contact_force_history exists only because of this."""
    isaac = _isaac_contact_fields()["net_forces_w_history"].replace(" ", "")
    mjlab = _mjlab_contact_fields()["force_history"].replace(" ", "")
    assert isaac == "N,T,B,3", f"Isaac Lab history layout changed to ({isaac}); revisit the transpose"
    assert mjlab == "[B,N,H,3]", f"mjlab history layout changed to {mjlab}; revisit the transpose"


def test_engines_disagree_on_the_force_attribute_name() -> None:
    """If they ever converge, the branch in contact_force_history can go."""
    isaac, mjlab = _isaac_contact_fields(), _mjlab_contact_fields()
    assert "net_forces_w_history" in isaac and "net_forces_w_history" not in mjlab
    assert "force_history" in mjlab and "force_history" not in isaac


def test_isaac_still_documents_net_forces_as_normal_only() -> None:
    """The basis for the denylist entry: the two engines measure different things."""
    isaaclab = pytest.importorskip("isaaclab")
    source = pathlib.Path(isaaclab.__file__).parent / "sensors/contact_sensor/contact_sensor_data.py"
    text = " ".join(source.read_text().split())
    assert "sum of the normal contact forces" in text
    assert "which also includes the tangential forces" in text


"""
Stubs shaped like the real data classes.
"""


class _IsaacData:
    def __init__(self, envs: int, elements: int, history: int) -> None:
        self.current_air_time = torch.arange(envs * elements, dtype=torch.float32).reshape(envs, elements)
        self.current_contact_time = torch.zeros(envs, elements)
        self.last_air_time = torch.zeros(envs, elements)
        self.last_contact_time = torch.zeros(envs, elements)
        # (N, T, B, 3)
        self.net_forces_w_history = torch.arange(envs * history * elements * 3, dtype=torch.float32).reshape(
            envs, history, elements, 3
        )


class _IsaacSensor:
    def __init__(self, envs: int = 2, history: int = 3) -> None:
        self.body_names = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link"]
        self.data = _IsaacData(envs, len(self.body_names), history)

    def find_bodies(self, name_keys, preserve_order: bool = False):
        ids = [i for i, n in enumerate(self.body_names) if any(re.fullmatch(k, n) for k in name_keys)]
        if preserve_order:
            ids = [i for k in name_keys for i, n in enumerate(self.body_names) if re.fullmatch(k, n)]
        return ids, [self.body_names[i] for i in ids]


class _MjlabData:
    def __init__(self, envs: int, elements: int, history: int) -> None:
        self.current_air_time = torch.arange(envs * elements, dtype=torch.float32).reshape(envs, elements)
        self.current_contact_time = torch.zeros(envs, elements)
        self.last_air_time = torch.zeros(envs, elements)
        self.last_contact_time = torch.zeros(envs, elements)
        # [B, N, H, 3] -- element and time swapped relative to Isaac Lab
        self.force_history = torch.zeros(envs, elements, history, 3)


class _MjlabSensor:
    """mjlab's sensor has no find_bodies; patterns are matched against primary_names."""

    def __init__(self, envs: int = 2, history: int = 3) -> None:
        self.primary_names = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link"]
        self.data = _MjlabData(envs, len(self.primary_names), history)


ANKLES = ContactSensorRef(name="feet", elements=(".*_ankle_roll_link",), track_air_time=True, history_length=3)


def test_engine_is_detected_from_the_element_attribute() -> None:
    assert sensor_engine(_IsaacSensor()) == "isaacsim"
    assert sensor_engine(_MjlabSensor()) == "mjlab"


def test_unknown_sensor_is_refused_rather_than_guessed() -> None:
    with pytest.raises(PortabilityError, match="element ordering is unknown"):
        sensor_engine(object())


@pytest.mark.parametrize("sensor", [_IsaacSensor(), _MjlabSensor()], ids=["isaacsim", "mjlab"])
def test_element_names_and_ids_agree_across_engines(sensor) -> None:
    assert element_names(sensor) == ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link"]
    assert element_ids(sensor, ANKLES) == [1, 2]


@pytest.mark.parametrize("sensor", [_IsaacSensor(), _MjlabSensor()], ids=["isaacsim", "mjlab"])
def test_air_time_selects_the_referenced_elements(sensor) -> None:
    """Same call, same answer, on sensors that name their element list differently."""
    assert torch.equal(air_time(sensor, ANKLES), torch.tensor([[1.0, 2.0], [4.0, 5.0]]))


@pytest.mark.parametrize("sensor", [_IsaacSensor(), _MjlabSensor()], ids=["isaacsim", "mjlab"])
def test_preserve_order_follows_the_patterns(sensor) -> None:
    ref = ContactSensorRef(name="feet", elements=("right_ankle_roll_link", "pelvis"), preserve_order=True)
    assert element_ids(sensor, ref) == [2, 0]


@pytest.mark.parametrize("sensor", [_IsaacSensor(), _MjlabSensor()], ids=["isaacsim", "mjlab"])
def test_a_pattern_that_matches_nothing_is_an_error(sensor) -> None:
    """Otherwise a foot-contact reward silently becomes a constant."""
    ref = ContactSensorRef(name="feet", elements=("left_wrist",))
    with pytest.raises(PortabilityError, match="matched none of"):
        element_ids(sensor, ref)


@pytest.mark.parametrize("sensor", [_IsaacSensor(), _MjlabSensor()], ids=["isaacsim", "mjlab"])
def test_in_contact_uses_each_engines_own_criterion(sensor) -> None:
    sensor.data.current_contact_time[:, 1] = 0.02
    contact = in_contact(sensor, ANKLES)
    assert contact.dtype == torch.bool
    assert contact[:, 0].all() and not contact[:, 1].any()


@pytest.mark.parametrize("sensor", [_IsaacSensor(), _MjlabSensor()], ids=["isaacsim", "mjlab"])
def test_missing_timing_tensor_says_which_flag_to_set(sensor) -> None:
    sensor.data.current_air_time = None
    with pytest.raises(PortabilityError, match="track_air_time"):
        air_time(sensor, ANKLES)


"""
The axis-order normalisation, which is the part that fails quietly.
"""


def test_force_history_comes_back_time_major_from_both_engines() -> None:
    isaac = contact_force_history(_IsaacSensor(envs=2, history=3), ANKLES)
    mjlab = contact_force_history(_MjlabSensor(envs=2, history=3), ANKLES)
    assert isaac.shape == (2, 3, 2, 3), "hub layout is (env, time, element, 3)"
    assert mjlab.shape == isaac.shape


def test_mjlab_history_is_transposed_not_reinterpreted() -> None:
    """Values must follow the axes, so mark one cell and check where it lands."""
    sensor = _MjlabSensor(envs=1, history=4)
    element, step = 2, 3  # right ankle, oldest retained substep
    sensor.data.force_history[0, element, step] = torch.tensor([7.0, 8.0, 9.0])
    out = contact_force_history(sensor, ANKLES)
    assert torch.equal(out[0, step, 1], torch.tensor([7.0, 8.0, 9.0]))


def test_square_history_would_hide_a_missing_transpose() -> None:
    """Element count equal to history length is the case a shape assertion cannot catch.

    Two feet and two substeps is an ordinary configuration, and with the axes swapped the tensor is
    still (env, 2, 2, 3). Only the values say which is which.
    """
    sensor = _MjlabSensor(envs=1, history=2)
    sensor.data.force_history[0, 1, 0] = torch.tensor([1.0, 0.0, 0.0])  # left ankle, newest
    out = contact_force_history(sensor, ANKLES)
    assert out.shape == (1, 2, 2, 3)
    assert torch.equal(out[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert not out[0, 1].any(), "older substep should be empty; axes are still swapped"


@pytest.mark.parametrize("sensor", [_IsaacSensor(), _MjlabSensor()], ids=["isaacsim", "mjlab"])
def test_missing_history_says_which_setting_to_change(sensor) -> None:
    attr = "net_forces_w_history" if sensor_engine(sensor) == "isaacsim" else "force_history"
    setattr(sensor.data, attr, None)
    with pytest.raises(PortabilityError, match="history_length"):
        contact_force_history(sensor, ANKLES)


"""
ContactSensorRef.
"""


def test_bare_string_elements_are_accepted() -> None:
    assert ContactSensorRef(name="feet", elements=".*_ankle").elements == (".*_ankle",)


def test_reference_without_elements_is_refused() -> None:
    with pytest.raises(ValueError, match="no element patterns"):
        ContactSensorRef(name="feet", elements=())


def test_negative_history_is_refused() -> None:
    with pytest.raises(ValueError, match="negative history_length"):
        ContactSensorRef(name="feet", elements=".*", history_length=-1)


"""
Resolution happens once.

Not a performance nicety. Isaac Lab's ``ContactSensor.body_names`` rebuilds itself from the physics
view on every access, costing about seventy milliseconds at four thousand environments, so a term
that resolves its feet per evaluation spends the step enumerating prim paths with the GPU idle --
measured at 18.7 of 21.6 seconds on flat G1, and a tenfold slowdown of the whole environment. The
counting stub below is what stops that returning quietly.
"""


class _CountingSensor:
    """An Isaac-shaped sensor that records how often its element list is read.

    Standalone rather than a subclass of :class:`_IsaacSensor`, whose constructor reads
    ``body_names`` before a counter could exist.
    """

    def __init__(self, names: list[str] | None = None) -> None:
        self.reads = 0
        self._names = names or ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link"]
        self.data = _IsaacData(2, len(self._names), 3)

    @property
    def body_names(self) -> list[str]:
        self.reads += 1
        return self._names

    def find_bodies(self, name_keys, preserve_order: bool = False):
        names = self.body_names
        ids = [i for i, n in enumerate(names) if any(re.fullmatch(k, n) for k in name_keys)]
        if preserve_order:
            ids = [i for k in name_keys for i, n in enumerate(names) if re.fullmatch(k, n)]
        return ids, [names[i] for i in ids]


def test_the_element_list_is_read_once_however_often_it_is_asked_for() -> None:
    """Stated as growth rather than a fixed count: engine detection probes the attribute too, and
    what matters is that the cost is paid on the first call and never again."""
    sensor = _CountingSensor()
    forget(sensor)
    first = element_names(sensor)
    settled = sensor.reads
    for _ in range(20):
        element_names(sensor)
    assert sensor.reads == settled, f"read the element list {sensor.reads - settled} more times; a step pays that"
    assert element_names(sensor) == first


def test_indices_are_resolved_once_per_reference() -> None:
    sensor = _CountingSensor()
    forget(sensor)
    expected = element_ids(sensor, ANKLES)
    reads_after_first = sensor.reads
    for _ in range(20):
        assert element_ids(sensor, ANKLES) == expected
    assert sensor.reads == reads_after_first, "re-resolved a reference that had already been resolved"


def test_cached_force_history_does_not_probe_body_names_each_step() -> None:
    """Engine detection must not execute Isaac's expensive ``body_names`` property."""
    sensor = _CountingSensor()
    forget(sensor)
    contact_force_history(sensor, ANKLES)
    reads_after_first = sensor.reads
    for _ in range(20):
        contact_force_history(sensor, ANKLES)
    assert sensor.reads == reads_after_first, "engine detection re-read body_names on the contact hot path"


def test_two_references_against_one_sensor_do_not_collide() -> None:
    """The cache is keyed by reference as well as by sensor; one entry per sensor would alias."""
    sensor = _CountingSensor()
    forget(sensor)
    pelvis = ContactSensorRef(name="feet", elements="pelvis")
    assert element_ids(sensor, ANKLES) != element_ids(sensor, pelvis)
    assert element_ids(sensor, pelvis) == [0]


def test_two_sensors_are_remembered_separately() -> None:
    """Keyed by identity, so a second sensor tracking other elements is not served the first's."""
    first = _CountingSensor()
    second = _CountingSensor(["pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "extra_link"])
    forget()
    assert element_ids(first, ANKLES) == [1, 2]
    assert element_ids(second, ContactSensorRef(name="extra", elements="extra_link")) == [3]


def test_caching_does_not_change_what_is_resolved() -> None:
    """Mutation check: the cached answer is the answer the uncached path gives."""
    sensor = _IsaacSensor()
    forget(sensor)
    uncached = element_ids(sensor, ANKLES)
    forget(sensor)
    assert element_ids(sensor, ANKLES) == uncached
    assert element_ids(sensor, ANKLES) == uncached
