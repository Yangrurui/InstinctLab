"""Guard: every claim in the hub vocabulary holds against the installed engines.

``instinctlab.compat.vocab`` and ``instinctlab.compat.denylist`` assert things about Isaac Lab and
MJLab -- which attribute carries a quantity, which legacy alias resolves to a centre-of-mass value,
which engine documents its quaternion order. Written as prose those claims rot silently across an
engine upgrade, and a portable term keeps reading an attribute whose meaning moved underneath it.

So the tables are checked here instead of trusted. Neither engine's runtime is needed: MJLab's
``EntityData`` imports standalone, and Isaac Lab's ``ArticulationData`` is read with ``ast`` because
importing ``isaaclab.assets`` pulls in ``omni`` and therefore a running Isaac Sim app.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from instinctlab.compat import (
    CANONICAL_QUATERNION,
    DENYLIST,
    ENGINES,
    HUB,
    LEGACY_COM_ALIASES,
    PortabilityError,
    RotationConvention,
    assert_portable,
    hub_entry,
    spoke_attr,
)
from instinctlab.compat.denylist import LEGACY_LINK_ALIASES, explicit_name

_SAME_AS = re.compile(r"^Same as :attr:`([A-Za-z0-9_]+)`")

_REPO = pathlib.Path(__file__).resolve().parents[1]

# Three documents stated three different sizes for the same table -- five, six and seven -- because
# nothing tied the prose to the list. Each entry here is a place that commits to a count out loud.
_COUNTED_IN_PROSE = (
    (".cursor/rules/multi-engine-training.mdc", r"(\d+) 个同名不同义的陷阱", "DENYLIST"),
    ("CROSS_ENGINE_DESIGN.md", r"denylist：(\d+) 个同名不同义", "DENYLIST"),
    ("CROSS_ENGINE_DESIGN.md", r"(\d+) 项 denylist", "DENYLIST"),
    (".cursor/rules/multi-engine-training.mdc", r"legacy 别名共 (\d+) 个", "ALIASES"),
    ("CROSS_ENGINE_DESIGN.md", r"legacy 别名是 (\d+) 个", "ALIASES"),
)


def _isaac_data_members() -> dict[str, str]:
    """Map ``ArticulationData`` member name -> docstring, without importing ``omni``."""
    isaaclab = pytest.importorskip("isaaclab")
    source = pathlib.Path(isaaclab.__file__).parent / "assets/articulation/articulation_data.py"
    if not source.is_file():  # pragma: no cover - upstream layout change
        pytest.skip(f"Isaac Lab articulation data not found at {source}")

    class_def = next(
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.ClassDef) and node.name == "ArticulationData"
    )
    members: dict[str, str] = {}
    for node in class_def.body:
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(dec, ast.Name) and dec.id == "property" for dec in node.decorator_list
        ):
            members[node.name] = ast.get_docstring(node) or ""
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            members.setdefault(node.target.id, "")
    return members


def _mjlab_data_members() -> set[str]:
    """Names exposed by ``EntityData``, including plain dataclass fields.

    ``hasattr`` alone misses annotated fields that carry no class-level default, which is how
    ``gravity_vec_w`` and ``default_root_state`` are declared.
    """
    data = pytest.importorskip("mjlab.entity.data")
    entity_data = data.EntityData
    names = {name for name in dir(entity_data) if not name.startswith("_")}
    for klass in entity_data.__mro__:
        names.update(getattr(klass, "__annotations__", {}))
    return {name for name in names if not name.startswith("_")}


def _isaac_aliases(members: dict[str, str]) -> dict[str, str]:
    """Aliases Isaac Lab itself documents as ``Same as :attr:`target`."""
    aliases: dict[str, str] = {}
    for name, doc in members.items():
        match = _SAME_AS.match(doc)
        if match:
            aliases[name] = match.group(1)
    return aliases


# --- hub table holds up against both engines ----------------------------------------------------


def test_every_hub_entry_covers_every_engine() -> None:
    for entry in HUB.values():
        assert set(entry.spokes) == set(ENGINES), f"{entry.name} is missing a spoke"


def test_isaacsim_spokes_exist() -> None:
    members = _isaac_data_members()
    for entry in HUB.values():
        spoke = entry.spoke("isaacsim")
        if spoke.attr is None:
            continue
        assert spoke.attr in members, f"hub {entry.name!r} claims Isaac Lab exposes {spoke.attr!r}"


def test_mjlab_spokes_exist() -> None:
    members = _mjlab_data_members()
    for entry in HUB.values():
        spoke = entry.spoke("mjlab")
        if spoke.attr is None:
            continue
        assert spoke.attr in members, f"hub {entry.name!r} claims MJLab exposes {spoke.attr!r}"


def test_engine_specific_entries_really_are_engine_specific() -> None:
    """An entry marked unavailable must actually be absent, or the hub is understating coverage."""
    members = _isaac_data_members()
    unavailable = [entry for entry in HUB.values() if not entry.spoke("isaacsim").available]
    assert unavailable, "expected the hub to carry at least one MuJoCo-native quantity"
    for entry in unavailable:
        assert entry.name not in members, f"{entry.name!r} is marked Isaac-unavailable but exists upstream"
        assert entry.spoke("isaacsim").evidence, f"{entry.name!r} must say why the engine has no equivalent"


def test_spoke_attr_reports_missing_engine_support() -> None:
    with pytest.raises(LookupError, match="no equivalent"):
        spoke_attr("site_quat_w", "isaacsim")
    assert spoke_attr("site_quat_w", "mjlab") == "site_quat_w"


def test_hub_entry_rejects_names_outside_the_vocabulary() -> None:
    with pytest.raises(KeyError, match="not in the hub vocabulary"):
        hub_entry("root_lin_vel_w")


# --- D8: quaternion order -----------------------------------------------------------------------


def test_every_quaternion_entry_is_wxyz() -> None:
    quats = [entry for entry in HUB.values() if entry.name.endswith("quat_w")]
    assert quats, "expected quaternion entries in the hub"
    for entry in quats:
        assert entry.rotation is CANONICAL_QUATERNION is RotationConvention.WXYZ, entry.name


def test_isaac_documents_wxyz_where_the_hub_says_it_does() -> None:
    members = _isaac_data_members()
    for entry in HUB.values():
        spoke = entry.spoke("isaacsim")
        if entry.rotation is None or not spoke.available or not spoke.documented:
            continue
        assert "(w, x, y, z)" in members[spoke.attr], f"Isaac Lab no longer documents {spoke.attr} as wxyz"


def test_mjlab_quaternion_order_stays_an_undocumented_dependency() -> None:
    """MJLab inherits w-first from MuJoCo without restating it, so the hub must not claim otherwise.

    If MJLab ever documents the convention this fails, and the right fix is to flip ``documented``
    and point the evidence at MJLab rather than at MuJoCo.
    """
    for entry in HUB.values():
        spoke = entry.spoke("mjlab")
        if entry.rotation is None or not spoke.available:
            continue
        assert not spoke.documented, f"{entry.name}: MJLab spoke claims to be documented"
        assert "MuJoCo" in spoke.evidence, f"{entry.name}: evidence must name the upstream convention"


# --- legacy aliases -----------------------------------------------------------------------------


def test_legacy_alias_tables_match_isaac_lab_exactly() -> None:
    """Both directions: nothing we list is invented, nothing upstream lists is missed."""
    members = _isaac_data_members()
    upstream = _isaac_aliases(members)
    assert upstream, "expected Isaac Lab to document legacy aliases"

    recorded = {**LEGACY_COM_ALIASES, **LEGACY_LINK_ALIASES}
    assert recorded == upstream

    for alias, target in upstream.items():
        expected_com = "_com_" in target
        assert (alias in LEGACY_COM_ALIASES) is expected_com, f"{alias} -> {target} is filed under the wrong anchor"


def test_the_expensive_aliases_are_the_ones_that_hide_their_anchor() -> None:
    """The traps are COM aliases whose own name says nothing about COM."""
    traps = {alias for alias in LEGACY_COM_ALIASES if "com" not in alias}
    assert "root_lin_vel_b" in traps, "the alias most likely to be mis-rewritten dropped out of the table"
    for alias in traps:
        assert explicit_name(alias).startswith(("root_com_", "body_com_"))


def test_legacy_aliases_are_absent_from_mjlab() -> None:
    members = _mjlab_data_members()
    for alias in (*LEGACY_COM_ALIASES, *LEGACY_LINK_ALIASES):
        assert alias not in members, f"MJLab grew {alias!r}; the alias table needs revisiting"


# --- denylist -----------------------------------------------------------------------------------


@pytest.mark.parametrize(("relative", "pattern", "table"), _COUNTED_IN_PROSE)
def test_the_prose_counts_the_table_it_describes(relative: str, pattern: str, table: str) -> None:
    sizes = {"DENYLIST": len(DENYLIST), "ALIASES": len(LEGACY_COM_ALIASES) + len(LEGACY_LINK_ALIASES)}
    stated = re.search(pattern, (_REPO / relative).read_text())
    assert stated is not None, f"{relative} no longer says how big {table} is; the regex needs updating"
    assert int(stated.group(1)) == sizes[table], (
        f"{relative} says {table} has {stated.group(1)} entries, but it has {sizes[table]}. "
        "A reader trusts the number, so fix the prose rather than this test."
    )


def test_denylisted_attributes_are_not_reachable_through_the_hub() -> None:
    for name in DENYLIST:
        assert name not in HUB, f"{name!r} is denylisted but also offered as a portable quantity"


def test_denylist_covers_every_engine_with_a_resolution() -> None:
    for entry in DENYLIST.values():
        assert set(entry.per_engine) == set(ENGINES), entry.name
        assert entry.resolution, f"{entry.name} must say what to do instead"


def test_applied_torque_has_no_mjlab_counterpart_but_its_false_friend_exists() -> None:
    """Guards the denylist claim that ``actuator_force`` is the wrong mapping for ``applied_torque``."""
    isaac = _isaac_data_members()
    mjlab = _mjlab_data_members()

    assert "applied_torque" in isaac
    assert "applied_torque" not in mjlab
    assert "qfrc_actuator" in mjlab, "the documented joint-space equivalent disappeared"
    assert "actuator_force" in mjlab, "the false friend disappeared; the denylist entry can be simplified"


def test_gravity_vector_spelling_differs_between_engines() -> None:
    """Guards the denylist claim that there is no shared ``gravity_vec_w``."""
    isaac = _isaac_data_members()
    mjlab = _mjlab_data_members()

    assert "gravity_vec_w" not in isaac, "Isaac Lab spells it GRAVITY_VEC_W and derives it from sim gravity"
    assert "gravity_vec_w" in mjlab
    assert "projected_gravity_b" in isaac and "projected_gravity_b" in mjlab, "the portable alternative must exist"


def test_assert_portable_refuses_traps_and_accepts_hub_names() -> None:
    with pytest.raises(PortabilityError, match="does not port"):
        assert_portable("joint_acc")
    with pytest.raises(PortabilityError, match="legacy alias"):
        assert_portable("root_lin_vel_b")
    for name in HUB:
        assert_portable(name)


def _isaac_articulation_methods() -> set[str]:
    """Method names on ``Articulation``, read from source so ``omni`` is not needed."""
    isaaclab = pytest.importorskip("isaaclab")
    source = pathlib.Path(isaaclab.__file__).parent / "assets/articulation/articulation.py"
    if not source.is_file():  # pragma: no cover - upstream layout change
        pytest.skip(f"Isaac Lab articulation not found at {source}")
    class_def = next(
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.ClassDef) and node.name == "Articulation"
    )
    return {node.name for node in class_def.body if isinstance(node, ast.FunctionDef)}


def test_the_frame_qualified_root_writers_exist_on_both_engines() -> None:
    """The denylist tells callers to use these instead of the ambiguous one; they have to be there.

    ``write_root_state_to_sim`` takes a centre-of-mass velocity on Isaac Lab and a link velocity on
    mjlab, so the advice is to say which frame is meant. If either engine renames the qualified
    writers, the advice becomes wrong, and it should fail here rather than in a training run whose
    resets are quietly a different distribution.
    """
    entry = DENYLIST["write_root_state_to_sim"]
    required = {"write_root_link_pose_to_sim", "write_root_link_velocity_to_sim"}

    isaac = _isaac_articulation_methods()
    assert required <= isaac, sorted(required - isaac)
    assert "write_root_state_to_sim" in isaac, "the ambiguous method the entry warns about is gone"

    entity = pytest.importorskip("mjlab.entity").Entity
    mjlab_methods = {name for name in dir(entity) if not name.startswith("__")}
    assert required <= mjlab_methods, sorted(required - mjlab_methods)
    assert "write_root_state_to_sim" in mjlab_methods

    for method in sorted(required):
        assert method in entry.resolution, f"the entry should name {method} as the way out"
