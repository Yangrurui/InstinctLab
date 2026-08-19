"""Guard: the selector table and the lowering match what the engines actually accept.

``compat.entity`` claims which selector kinds each engine can express and lowers an ``EntityRef``
onto the engine's own ``SceneEntityCfg``. Both claims decay silently -- an engine gains a selector
kind and the table quietly under-reports it, or renames a field and the lowering produces a config
that constructs fine and selects nothing.

The two engines are checked differently for the usual reason: mjlab's ``SceneEntityCfg`` imports
standalone, Isaac Lab's is a ``configclass`` and pulls in ``omni``, so its fields are read with
``ast``. That is enough to verify the table and the field names the lowering targets; constructing
an Isaac config is skipped unless a full install is present.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib

import pytest

from instinctlab.compat.entity import (
    UnsupportedSelector,
    lower,
    resolved_names,
    selector_field,
    selector_kinds,
    universal,
)
from instinctlab.spec.entity import UNIVERSAL_KINDS, EntityRef


def _isaac_cfg_fields() -> list[str]:
    """Field names on Isaac Lab's ``SceneEntityCfg``, without importing ``omni``."""
    isaaclab = pytest.importorskip("isaaclab")
    source = pathlib.Path(isaaclab.__file__).parent / "managers/scene_entity_cfg.py"
    if not source.is_file():  # pragma: no cover - upstream layout change
        pytest.skip(f"Isaac Lab scene entity cfg not found at {source}")
    class_def = next(
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.ClassDef) and node.name == "SceneEntityCfg"
    )
    return [n.target.id for n in class_def.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]


def _mjlab_cfg_fields() -> list[str]:
    module = pytest.importorskip("mjlab.managers.scene_entity_config")
    return [f.name for f in dataclasses.fields(module.SceneEntityCfg)]


def _engine_fields(engine: str) -> list[str]:
    return _isaac_cfg_fields() if engine == "isaacsim" else _mjlab_cfg_fields()


def test_every_engine_registers_its_selectors() -> None:
    """The registry is populated by importing engine packages, and the tests below parametrize on it.

    Which is a trap worth naming: if registration stopped happening, ``sorted(selector_kinds())``
    would be empty, every parametrized test in this file would collect zero cases, and the run would
    be green. A guard that the registry is not empty is the difference between those tests checking
    the engines and checking nothing.
    """
    from instinctlab.engines import ADAPTERS

    # Both being empty satisfies the equality below while leaving the same zero cases collected, so
    # the count is asserted first. This project ships two engines and the parametrized tests are
    # worth having only while there are engines to run them against.
    assert len(ADAPTERS) >= 2, f"only {sorted(ADAPTERS)} adapters are known; the comparison below would be vacuous"
    assert set(selector_kinds()) == set(ADAPTERS), (
        f"registered engines {sorted(selector_kinds())} are not the known ones {sorted(ADAPTERS)}; "
        "an engine package registers its selectors when it is imported"
    )


@pytest.mark.parametrize("engine", sorted(selector_kinds()))
def test_selector_table_matches_engine(engine: str) -> None:
    """The table lists exactly the kinds the engine's config declares -- no more, no fewer."""
    declared = {name[: -len("_names")] for name in _engine_fields(engine) if name.endswith("_names")}
    assert declared == set(selector_kinds()[engine]), (
        f"{engine} selector kinds drifted. Config declares {sorted(declared)}, "
        f"table says {sorted(selector_kinds()[engine])}."
    )


@pytest.mark.parametrize("engine", sorted(selector_kinds()))
def test_every_kind_has_a_names_and_ids_field(engine: str) -> None:
    """The ``<kind>_names`` / ``<kind>_ids`` convention holds, which is what the lowering assumes."""
    fields = set(_engine_fields(engine))
    for kind in selector_kinds()[engine]:
        assert selector_field(kind) in fields
        assert f"{kind}_ids" in fields


@pytest.mark.parametrize("engine", sorted(selector_kinds()))
def test_engines_agree_on_the_non_selector_fields(engine: str) -> None:
    """``name`` and ``preserve_order`` are the only other fields, so lowering can pass them blindly."""
    others = {f for f in _engine_fields(engine) if not f.endswith(("_names", "_ids"))}
    assert others == {"name", "preserve_order"}


def test_universal_kinds_are_the_intersection() -> None:
    """``UNIVERSAL_KINDS`` is derived from the engines, not asserted independently of them."""
    intersection = frozenset.intersection(*selector_kinds().values())
    assert intersection == frozenset(UNIVERSAL_KINDS)


def test_engines_overlap_on_almost_nothing() -> None:
    """The premise behind an open selector set: agreement is the exception, not the rule."""
    isaac, mjlab = selector_kinds()["isaacsim"], selector_kinds()["mjlab"]
    assert len(isaac & mjlab) == 2
    assert isaac - mjlab == {"fixed_tendon", "object_collection"}
    assert len(mjlab - isaac) == 8
    # Related names that are deliberately not unified; unifying them would resolve a reference
    # against a different set of elements on the far side.
    assert "fixed_tendon" in isaac and "fixed_tendon" not in mjlab
    assert "tendon" in mjlab and "tendon" not in isaac


"""
EntityRef itself.
"""


def test_bare_string_is_accepted_like_the_engines_do() -> None:
    ref = EntityRef(bodies=".*_ankle_roll_link")
    assert ref.bodies == (".*_ankle_roll_link",)
    assert ref.selectors() == {"body": (".*_ankle_roll_link",)}


def test_selectors_merges_named_and_open_kinds() -> None:
    ref = EntityRef(entity="robot", joints=("hip",), other={"geom": "left_foot", "site": ("imu",)})
    assert ref.selectors() == {"joint": ("hip",), "geom": ("left_foot",), "site": ("imu",)}
    assert ref.kinds() == {"joint", "geom", "site"}


def test_universal_kinds_go_through_their_own_fields() -> None:
    """Two ways to say the same thing would compare unequal, so one of them is refused."""
    with pytest.raises(ValueError, match="own field"):
        EntityRef(other={"body": ("pelvis",)})


def test_empty_selector_is_refused() -> None:
    """An empty pattern list selects everything on one engine and nothing on another; say which."""
    with pytest.raises(ValueError, match="no patterns"):
        EntityRef(other={"geom": ()})


def test_equal_references_compare_equal_regardless_of_mapping_order() -> None:
    a = EntityRef(other={"geom": ("a",), "site": ("b",)})
    b = EntityRef(other={"site": ("b",), "geom": ("a",)})
    assert a == b


def test_universal_helper() -> None:
    assert universal(EntityRef(joints=".*", bodies="pelvis"))
    assert not universal(EntityRef(other={"geom": ("foot",)}))


"""
Lowering.
"""


def test_lower_to_mjlab_produces_a_usable_config() -> None:
    module = pytest.importorskip("mjlab.managers.scene_entity_config")
    ref = EntityRef(entity="robot", bodies=(".*_ankle_roll_link",), other={"geom": ("left_foot",)}, preserve_order=True)
    cfg = lower(ref, "mjlab")
    assert isinstance(cfg, module.SceneEntityCfg)
    assert cfg.name == "robot"
    assert cfg.preserve_order is True
    assert cfg.body_names == (".*_ankle_roll_link",)
    assert cfg.geom_names == ("left_foot",)
    assert cfg.joint_names is None


def test_lower_leaves_unnamed_kinds_at_their_defaults() -> None:
    """Only the kinds the reference names are set, so the rest keep the engine's own default."""
    module = pytest.importorskip("mjlab.managers.scene_entity_config")
    cfg = lower(EntityRef(entity="robot", joints=("hip",)), "mjlab")
    defaults = {f.name: f for f in dataclasses.fields(module.SceneEntityCfg)}
    for kind in selector_kinds()["mjlab"] - {"joint"}:
        assert getattr(cfg, selector_field(kind)) is defaults[selector_field(kind)].default


@pytest.mark.parametrize("engine", sorted(selector_kinds()))
def test_lowering_targets_fields_the_engine_declares(engine: str) -> None:
    """Every field the lowering would set exists on the engine's config.

    This is the part of the Isaac Lab lowering that can be checked without a USD runtime: not that
    the object constructs, but that no field name is invented.
    """
    fields = set(_engine_fields(engine))
    for kind in selector_kinds()[engine]:
        assert selector_field(kind) in fields
    assert {"name", "preserve_order"} <= fields


def test_lower_refuses_a_kind_the_engine_cannot_express() -> None:
    """The mjlab-to-Isaac direction, which is where this matters."""
    ref = EntityRef(entity="robot", other={"geom": ("left_foot_collision",)})
    with pytest.raises(UnsupportedSelector, match="geom") as excinfo:
        lower(ref, "isaacsim")
    # The error has to say what the engine *can* do, or the reader has to go read the table.
    assert "joint" in str(excinfo.value)


def test_lower_reports_every_missing_kind_at_once() -> None:
    ref = EntityRef(entity="robot", other={"geom": ("a",), "site": ("b",)})
    with pytest.raises(UnsupportedSelector) as excinfo:
        lower(ref, "isaacsim")
    assert "geom" in str(excinfo.value) and "site" in str(excinfo.value)


def test_lower_rejects_an_unknown_engine() -> None:
    with pytest.raises(KeyError, match="unknown engine"):
        lower(EntityRef(), "bullet")


def test_lower_uses_each_engines_declared_container_type() -> None:
    """Isaac Lab annotates ``list[str]``, mjlab ``tuple[str, ...]``; produce what each declares."""
    pytest.importorskip("mjlab.managers.scene_entity_config")
    assert isinstance(lower(EntityRef(bodies=("a", "b")), "mjlab").body_names, tuple)


"""
The post-resolve normalisation.
"""


class _FakeEntity:
    """Stands in for an articulation; both engines expose ``<kind>_names`` this way."""

    body_names = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link"]
    joint_names = ["hip", "knee", "ankle"]


class _FakeCfg:
    def __init__(self, **ids: object) -> None:
        self.__dict__.update(ids)


def test_resolved_names_reads_through_indices_not_patterns() -> None:
    """The whole point: the answer does not depend on which engine filled the config in."""
    isaac_style = _FakeCfg(body_names=[".*_ankle_roll_link"], body_ids=[1, 2])
    mjlab_style = _FakeCfg(body_names=("left_ankle_roll_link", "right_ankle_roll_link"), body_ids=[1, 2])
    expected = ["left_ankle_roll_link", "right_ankle_roll_link"]
    assert resolved_names(_FakeEntity(), isaac_style, "body") == expected
    assert resolved_names(_FakeEntity(), mjlab_style, "body") == expected
    # Reading the field directly is what gives two different answers.
    assert isaac_style.body_names != list(mjlab_style.body_names)


def test_resolved_names_handles_the_slice_optimisation() -> None:
    """Both engines collapse a full in-order selection to ``slice(None)``."""
    cfg = _FakeCfg(body_ids=slice(None))
    assert resolved_names(_FakeEntity(), cfg, "body") == _FakeEntity.body_names


def test_resolved_names_preserves_selection_order() -> None:
    """With ``preserve_order`` the indices are out of entity order; the names must follow them."""
    cfg = _FakeCfg(body_ids=[2, 0])
    assert resolved_names(_FakeEntity(), cfg, "body") == ["right_ankle_roll_link", "pelvis"]


def test_resolved_names_accepts_a_scalar_index() -> None:
    assert resolved_names(_FakeEntity(), _FakeCfg(joint_ids=1), "joint") == ["knee"]


def test_resolved_names_defaults_to_bodies() -> None:
    assert resolved_names(_FakeEntity(), _FakeCfg(body_ids=[0])) == ["pelvis"]
