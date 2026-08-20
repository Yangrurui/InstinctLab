"""The two rough recipes must not drift in silence.

``engines/isaacsim/rough.py`` and ``engines/mjlab/rough.py`` are the same parkour
grid written twice. Nothing used to compare them, so a copied constant could
diverge on one side and every test would stay green: no exception, no failed
assertion, training still converges. That is how ``step_width`` 0.30 vs 0.35
and ``horizontal_scale`` 0.05 vs 0.07 shipped.

mjlab's recipe now deliberately follows InstinctMJ, so five constant groups
differ from Isaac on purpose. Those live in ``KNOWN_DIFFS`` — tile, field,
Isaac value, mjlab value — and each row is asserted to still be present. A
dead entry fails the same way a new unexplained diff does. Slot 9, Isaac's
``use_cache``, and mjlab's ``add_lights`` stay on the structural allow-list.

Why AST, not constructed cfg objects: Isaac's module imports ``isaaclab`` /
``instinctlab.terrains`` inside the builder, and those pull ``pxr``. The default
suite (``pytest tests/``) must not start Kit. Drift is introduced by people
editing literals, not by people running a simulator, so a guard that only
runs in a live test is the weaker one.

The comparison expands ``perlin()`` / ``target()`` / ``target_center()`` /
``**_walls()`` so a helper that changed on one side still fails. Numbers are
normalized (``3`` == ``3.0``).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
ISAAC_ROUGH = REPO / "source/instinctlab/instinctlab/engines/isaacsim/rough.py"
MJLAB_ROUGH = REPO / "source/instinctlab/instinctlab/engines/mjlab/rough.py"
ISAAC_HF_BASE = Path("/root/IsaacLab/source/isaaclab/isaaclab/terrains/height_field/hf_terrains_cfg.py")
MJLAB_HF_BASE = REPO / "source/instinctlab/instinctlab/engines/mjlab/terrains/height_field/hf_terrains_cfg.py"

HELPERS = frozenset({"_walls", "perlin", "target", "target_center"})
SHARED_GENERATOR = (
    "seed",
    "size",
    "border_width",
    "num_rows",
    "num_cols",
    "horizontal_scale",
    "vertical_scale",
    "slope_threshold",
    "curriculum",
)
ISAAC_ONLY_GENERATOR = frozenset({"use_cache"})
MJLAB_ONLY_GENERATOR = frozenset({"add_lights"})
SLOT9 = 8
ISAAC_SLOT9_NAME = "mesh_boxes"
MJLAB_SLOT9_NAME = "dense_boxes"
ISAAC_SLOT9_TYPE = "PerlinMeshRandomMultiBoxTerrainCfg"
MJLAB_SLOT9_TYPE = "PerlinDiscreteObstaclesTerrainCfg"
# Fields both slot-9 types actually declare and that we still want to keep in
# lockstep. Type-specific fields (box_* vs num_obstacles / perlin_cfg / …)
# are the structural gap and are not compared.
SLOT9_SHARED_FIELDS = (
    "proportion",
    "platform_width",
    "wall_prob",
    "wall_height",
    "wall_thickness",
    "flat_patch_sampling",
)

ABSENT = object()
# Five intended differences, one row per (tile, field) so a half-aligned pair
# is a dead entry rather than a still-green group. Generator-level rows use
# tile=None. Isaac's omitted plane ``border_width`` is ABSENT, not 0.0.
KNOWN_DIFFS: tuple[tuple[str | None, str, Any, Any], ...] = (
    (None, "horizontal_scale", 0.05, 0.07),
    ("perlin_rough", "border_width", ABSENT, 1.0),
    ("perlin_rough_stand", "border_width", ABSENT, 1.0),
    ("pyramid_stairs", "step_width", 0.3, 0.35),
    ("pyramid_stairs_inv", "step_width", 0.3, 0.35),
    ("pyramid_stairs_high", "step_width", 1.5, 1.54),
    ("pyramid_stairs_inv_high", "step_width", 1.5, 1.54),
    ("boxes", "border_width", 0.0, 1.0),
)


def _helpers(tree: ast.Module) -> dict[str, ast.AST]:
    found: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in HELPERS:
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return) and stmt.value is not None:
                    found[node.name] = stmt.value
                    break
    missing = HELPERS - found.keys()
    assert not missing, f"recipe helpers missing {sorted(missing)}"
    return found


def _generator_call(tree: ast.Module) -> ast.Call:
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "rough_generator_cfg":
            continue
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                if ast.unparse(stmt.value.func).endswith("TerrainGeneratorCfg"):
                    return stmt.value
    raise LookupError("rough_generator_cfg does not return a TerrainGeneratorCfg call")


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def _normalize(value: Any) -> Any:
    if value is ABSENT:
        return ABSENT
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return tuple(_normalize(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    return value


def _resolve(node: ast.AST, helpers: dict[str, ast.AST]) -> Any:
    if isinstance(node, ast.Call):
        short = ast.unparse(node.func).rsplit(".", 1)[-1]
        if short in helpers:
            return _resolve(helpers[short], helpers)
        return {"_type": short, **_call_kwargs(node, helpers)}
    if isinstance(node, ast.Dict):
        return {
            _resolve(key, helpers) if key is not None else None: _resolve(value, helpers)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.Name) and node.id in helpers:
        return _resolve(helpers[node.id], helpers)
    literal = _literal(node)
    if literal is not None:
        return _normalize(literal)
    raise AssertionError(f"recipe value is not a literal or known helper: {ast.unparse(node)}")


def _call_kwargs(call: ast.Call, helpers: dict[str, ast.AST]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            expanded = _resolve(keyword.value, helpers)
            assert isinstance(expanded, dict) and "_type" not in expanded, expanded
            out.update(expanded)
            continue
        out[keyword.arg] = _resolve(keyword.value, helpers)
    return out


def _load(path: Path) -> tuple[str, dict[str, Any]]:
    tree = ast.parse(path.read_text())
    call = _generator_call(tree)
    return ast.unparse(call.func).rsplit(".", 1)[-1], _call_kwargs(call, _helpers(tree))


def _class_attr_default(path: Path, class_name: str, attr: str) -> Any:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for stmt in node.body:
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == attr
                and stmt.value is not None
            ):
                return _normalize(_literal(stmt.value))
    raise LookupError(f"{path} has no {class_name}.{attr} default")


def _field(mapping: dict[str, Any], name: str) -> Any:
    return mapping[name] if name in mapping else ABSENT


def _fmt(value: Any) -> Any:
    return "<absent>" if value is ABSENT else value


def test_isaac_perlin_plane_border_width_default_is_zero() -> None:
    """Isaac omits the kwarg; the effective value is the class default, not 0.0-looking."""
    assert ISAAC_HF_BASE.is_file(), ISAAC_HF_BASE
    assert _class_attr_default(ISAAC_HF_BASE, "HfTerrainBaseCfg", "border_width") == 0.0


def test_mjlab_perlin_plane_border_width_default_is_zero() -> None:
    assert _class_attr_default(MJLAB_HF_BASE, "HfTerrainBaseCfg", "border_width") == 0.0


def test_the_two_rough_recipes_match_except_the_allow_list() -> None:
    isaac_cls, isaac = _load(ISAAC_ROUGH)
    mjlab_cls, mjlab = _load(MJLAB_ROUGH)

    assert isaac_cls == "TerrainGeneratorCfg"
    assert mjlab_cls == "FiledTerrainGeneratorCfg"

    isaac_only = (set(isaac) - {"sub_terrains"}) - set(SHARED_GENERATOR)
    mjlab_only = (set(mjlab) - {"sub_terrains"}) - set(SHARED_GENERATOR)
    assert isaac_only == ISAAC_ONLY_GENERATOR, isaac_only
    assert mjlab_only == MJLAB_ONLY_GENERATOR, mjlab_only
    assert isaac["use_cache"] is False
    assert mjlab["add_lights"] is True

    known = {(tile, field): (_normalize(left), _normalize(right)) for tile, field, left, right in KNOWN_DIFFS}
    seen: set[tuple[str | None, str]] = set()

    for field in SHARED_GENERATOR:
        key = (None, field)
        if key in known:
            expect_isaac, expect_mjlab = known[key]
            assert _field(isaac, field) == expect_isaac and _field(mjlab, field) == expect_mjlab, (
                field,
                _fmt(_field(isaac, field)),
                _fmt(_field(mjlab, field)),
            )
            seen.add(key)
            continue
        assert field in isaac and field in mjlab, field
        assert isaac[field] == mjlab[field], (field, isaac[field], mjlab[field])

    isaac_tiles: dict[str, Any] = isaac["sub_terrains"]
    mjlab_tiles: dict[str, Any] = mjlab["sub_terrains"]
    isaac_names = list(isaac_tiles)
    mjlab_names = list(mjlab_tiles)
    assert len(isaac_names) == len(mjlab_names) == 10
    assert isaac_names[SLOT9] == ISAAC_SLOT9_NAME
    assert mjlab_names[SLOT9] == MJLAB_SLOT9_NAME
    aligned_names = [name for i, name in enumerate(isaac_names) if i != SLOT9]
    assert [name for i, name in enumerate(mjlab_names) if i != SLOT9] == aligned_names

    unexplained: list[tuple[Any, ...]] = []
    for index, (isaac_name, mjlab_name) in enumerate(zip(isaac_names, mjlab_names, strict=True)):
        isaac_tile = isaac_tiles[isaac_name]
        mjlab_tile = mjlab_tiles[mjlab_name]
        if index == SLOT9:
            assert isaac_tile["_type"] == ISAAC_SLOT9_TYPE
            assert mjlab_tile["_type"] == MJLAB_SLOT9_TYPE
            for field in SLOT9_SHARED_FIELDS:
                assert isaac_tile[field] == mjlab_tile[field], (isaac_name, mjlab_name, field)
            continue
        assert isaac_name == mjlab_name
        assert isaac_tile["_type"] == mjlab_tile["_type"], (isaac_name, isaac_tile["_type"], mjlab_tile["_type"])
        isaac_fields = {key: value for key, value in isaac_tile.items() if key != "_type"}
        mjlab_fields = {key: value for key, value in mjlab_tile.items() if key != "_type"}
        fields = set(isaac_fields) | set(mjlab_fields)
        for field in sorted(fields):
            key = (isaac_name, field)
            left = _field(isaac_fields, field)
            right = _field(mjlab_fields, field)
            if key in known:
                expect_isaac, expect_mjlab = known[key]
                assert left == expect_isaac and right == expect_mjlab, (
                    isaac_name,
                    field,
                    _fmt(left),
                    _fmt(right),
                    _fmt(expect_isaac),
                    _fmt(expect_mjlab),
                )
                seen.add(key)
                continue
            if left != right:
                unexplained.append((isaac_name, field, _fmt(left), _fmt(right)))

    assert not unexplained, f"unexplained recipe diffs: {unexplained}"
    assert seen == set(known), f"known-diff table drifted: missing {set(known) - seen}, unexpected {seen - set(known)}"
