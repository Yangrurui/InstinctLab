"""Guard: the vendored math in ``instinctlab_engine.bridge.math`` still equals both engines'.

Isaac Lab owns the original ``utils/math.py`` and mjlab carries a copy of it. ``compat.math`` is a
third copy, taken because importing either engine's version drags the engine in -- Isaac Lab's is
not even importable without a USD runtime. A copy that nobody checks is a copy that drifts, and the
failure mode is silent: a portable term keeps computing, just not what the native task computes.

Source text cannot be compared, since this repository's black profile reflows the vendored code and
the two engines already disagree cosmetically. Values can, and they agree exactly -- so these tests
assert bitwise equality in float64 rather than a tolerance, which is what makes an upstream change
to any of these formulas impossible to miss.

Neither engine's runtime is needed. mjlab's math module imports standalone; Isaac Lab's is loaded
by file path to bypass ``isaaclab.utils.__init__``, which imports ``pxr``.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pathlib
import sys
import torch
import types
from collections.abc import Callable

import pytest

from instinctlab_engine.bridge import math as compat_math

# Vendored from these two, so both are the reference. Names are the public surface of compat.math
# minus the layout converters, which no engine exposes in this form.
_VENDORED = [
    "axis_angle_from_quat",
    "combine_frame_transforms",
    "copysign",
    "euler_xyz_from_quat",
    "matrix_from_quat",
    "normalize",
    "quat_apply",
    "quat_apply_inverse",
    "quat_box_minus",
    "quat_conjugate",
    "quat_error_magnitude",
    "quat_from_angle_axis",
    "quat_from_euler_xyz",
    "quat_from_matrix",
    "quat_inv",
    "quat_mul",
    "quat_unique",
    "subtract_frame_transforms",
    "transform_points",
    "wrap_to_pi",
    "yaw_quat",
]


def _isaac_math() -> types.ModuleType:
    """Load Isaac Lab's ``utils/math.py`` directly; the package ``__init__`` needs ``pxr``."""
    isaaclab = pytest.importorskip("isaaclab")
    source = pathlib.Path(isaaclab.__file__).parent / "utils/math.py"
    if not source.is_file():  # pragma: no cover - upstream layout change
        pytest.skip(f"Isaac Lab math not found at {source}")
    spec = importlib.util.spec_from_file_location("_isaac_math_under_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mjlab_math() -> types.ModuleType:
    return pytest.importorskip("mjlab.utils.lab_api.math")


@pytest.fixture(scope="module")
def sample() -> dict[str, torch.Tensor]:
    """Deterministic float64 inputs, random bulk plus the cases where formulas tend to diverge."""
    gen = torch.Generator().manual_seed(20260818)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    # Near-identity and near-pi rotations exercise the Taylor branch of axis_angle_from_quat and the
    # argmax branch of quat_from_matrix; a bare random batch reaches neither.
    tiny = torch.tensor([1.0, 1e-9, 0.0, 0.0], dtype=torch.float64)
    half_turn = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    negative_w = torch.tensor([-1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    edge = torch.stack([tiny, half_turn, identity, negative_w])
    quat = torch.cat([torch.nn.functional.normalize(rand(128, 4), dim=-1), edge])

    # Odd multiples of pi are the documented boundary of wrap_to_pi.
    angle = torch.cat(
        [
            rand(128) * 10.0,
            torch.tensor([torch.pi, -torch.pi, 3 * torch.pi, -3 * torch.pi, 0.0], dtype=torch.float64),
        ]
    )
    n = quat.shape[0]
    return {
        "quat": quat,
        "quat_other": torch.roll(quat, 1, dims=0),
        "vec": rand(n, 3),
        "vec_other": rand(n, 3),
        "angle": angle,
        "axis": rand(angle.shape[0], 3),
        "points": rand(8, 16, 3),
        "matrix": compat_math.matrix_from_quat(quat),
    }


# Argument builders keyed by function name. Kept explicit so a signature change upstream surfaces
# as a failure here rather than as a silently skipped comparison.
_ARGS: dict[str, Callable[[dict[str, torch.Tensor]], tuple]] = {
    "axis_angle_from_quat": lambda s: (s["quat"],),
    "combine_frame_transforms": lambda s: (s["vec"], s["quat"], s["vec_other"], s["quat_other"]),
    "copysign": lambda s: (1.5, s["angle"]),
    "euler_xyz_from_quat": lambda s: (s["quat"],),
    "matrix_from_quat": lambda s: (s["quat"],),
    "normalize": lambda s: (s["vec"],),
    "quat_apply": lambda s: (s["quat"], s["vec"]),
    "quat_apply_inverse": lambda s: (s["quat"], s["vec"]),
    "quat_box_minus": lambda s: (s["quat"], s["quat_other"]),
    "quat_conjugate": lambda s: (s["quat"],),
    "quat_error_magnitude": lambda s: (s["quat"], s["quat_other"]),
    "quat_from_angle_axis": lambda s: (s["angle"], s["axis"]),
    "quat_from_euler_xyz": lambda s: (s["angle"], s["angle"] * 0.5, s["angle"] * -0.25),
    "quat_from_matrix": lambda s: (s["matrix"],),
    "quat_inv": lambda s: (s["quat"],),
    "quat_mul": lambda s: (s["quat"], s["quat_other"]),
    "quat_unique": lambda s: (s["quat"],),
    "subtract_frame_transforms": lambda s: (s["vec"], s["quat"], s["vec_other"], s["quat_other"]),
    "transform_points": lambda s: (s["points"], s["vec"][:8], s["quat"][:8]),
    "wrap_to_pi": lambda s: (s["angle"],),
    "yaw_quat": lambda s: (s["quat"],),
}


def _flatten(result) -> torch.Tensor:
    if isinstance(result, torch.Tensor):
        return result.reshape(-1)
    return torch.cat([item.reshape(-1) for item in result])


@pytest.mark.parametrize("name", _VENDORED)
@pytest.mark.parametrize("engine", ["isaaclab", "mjlab"])
def test_vendored_function_matches_engine(name: str, engine: str, sample: dict[str, torch.Tensor]) -> None:
    """Every vendored function reproduces the engine's output exactly."""
    reference = _isaac_math() if engine == "isaaclab" else _mjlab_math()
    assert hasattr(reference, name), f"{engine} no longer defines {name}; compat.math must be revisited"

    args = _ARGS[name](sample)
    ours = _flatten(getattr(compat_math, name)(*args))
    theirs = _flatten(getattr(reference, name)(*args))

    assert ours.shape == theirs.shape, f"{name}: shape {tuple(ours.shape)} != {tuple(theirs.shape)}"
    assert torch.equal(ours, theirs), f"{name}: max deviation from {engine} is {(ours - theirs).abs().max().item():.3e}"


@pytest.mark.parametrize("name", _VENDORED)
def test_vendored_function_matches_in_float32(name: str, sample: dict[str, torch.Tensor]) -> None:
    """Same equality holds at the precision training actually runs in."""
    reference = _mjlab_math()
    single = {key: value.float() if value.is_floating_point() else value for key, value in sample.items()}
    args = _ARGS[name](single)
    assert torch.equal(_flatten(getattr(compat_math, name)(*args)), _flatten(getattr(reference, name)(*args)))


def test_sample_uniform_matches_engines() -> None:
    """Seeded draws agree, so randomised events reproduce across the two implementations."""
    for reference in (_isaac_math(), _mjlab_math()):
        torch.manual_seed(7)
        ours = compat_math.sample_uniform(-2.0, 3.0, (64, 3), "cpu")
        torch.manual_seed(7)
        theirs = reference.sample_uniform(-2.0, 3.0, (64, 3), "cpu")
        assert torch.equal(ours, theirs)


def test_module_imports_no_engine() -> None:
    """compat.math stays engine-free, which is the entire reason it exists."""
    tree = ast.parse(pathlib.Path(compat_math.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert not imported & {"isaaclab", "mjlab", "omni", "pxr", "mujoco", "warp"}, f"engine import leaked: {imported}"


def test_public_surface_is_declared() -> None:
    """``__all__`` matches what the module actually defines, so nothing leaks in unannounced."""
    public = {name for name in vars(compat_math) if not name.startswith("_") and callable(getattr(compat_math, name))}
    assert public - {"torch"} == set(compat_math.__all__)


"""
Decision D8: quaternions are (w, x, y, z) everywhere above the engine boundary.
"""


def test_layout_converters_have_no_default_direction() -> None:
    """Neither converter can be called without naming the direction, unlike ``convert_quat``."""
    assert not hasattr(compat_math, "convert_quat"), "convert_quat defaults to to='xyzw' and must not be vendored"
    for name in ("quat_wxyz_to_xyzw", "quat_xyzw_to_wxyz"):
        assert name in compat_math.__all__
        with pytest.raises(TypeError):
            getattr(compat_math, name)()  # direction lives in the name; the tensor is required


def test_layout_converters_match_convert_quat(sample: dict[str, torch.Tensor]) -> None:
    """The explicit converters agree with the engines' ``convert_quat`` given an explicit ``to=``."""
    quat = sample["quat"]
    for reference in (_isaac_math(), _mjlab_math()):
        assert torch.equal(compat_math.quat_wxyz_to_xyzw(quat), reference.convert_quat(quat, to="xyzw"))
        assert torch.equal(compat_math.quat_xyzw_to_wxyz(quat), reference.convert_quat(quat, to="wxyz"))


def test_layout_converters_round_trip(sample: dict[str, torch.Tensor]) -> None:
    quat = sample["quat"]
    assert torch.equal(compat_math.quat_xyzw_to_wxyz(compat_math.quat_wxyz_to_xyzw(quat)), quat)


@pytest.mark.parametrize("bad_shape", [(4, 3), (8,), (2, 5)])
def test_layout_converters_reject_non_quaternions(bad_shape: tuple[int, ...]) -> None:
    for converter in (compat_math.quat_wxyz_to_xyzw, compat_math.quat_xyzw_to_wxyz):
        with pytest.raises(ValueError, match=r"\(\.\.\., 4\)"):
            converter(torch.zeros(bad_shape))


def test_vendored_docstrings_state_wxyz() -> None:
    """Each vendored quaternion function documents the order, so callers never have to guess."""
    for name in _VENDORED:
        doc = getattr(compat_math, name).__doc__ or ""
        if "quat" not in name and "quat" not in doc.lower():
            continue
        assert "(w, x, y, z)" in doc, f"{name} handles quaternions without stating the (w, x, y, z) order"


def test_utils_math_runs_without_an_engine() -> None:
    """``instinctlab_engine.math`` was the first module ported off the engine; keep it that way.

    It reaches ``compat.math`` for seven functions and is otherwise pure torch, so it now imports
    with no simulator present. The six ``torch.jit.script`` functions are the fragile part: script
    compilation follows calls into the vendored module, so a change there can break them even
    though eager execution is fine.
    """
    blocked = {"isaaclab", "omni", "pxr"}

    class _Blocker:
        def find_module(self, name, path=None):  # noqa: D102 - legacy finder protocol, enough for importlib
            return self if name.split(".")[0] in blocked else None

        def load_module(self, name):  # pragma: no cover - only reached on regression
            raise ImportError(f"{name} must not be needed here")

    blocker = _Blocker()
    for name in [name for name in sys.modules if name.split(".")[0] in blocked]:
        del sys.modules[name]
    sys.modules.pop("instinctlab_engine.math", None)
    sys.meta_path.insert(0, blocker)
    try:
        module = importlib.import_module("instinctlab_engine.math")
        scripted = [name for name, value in vars(module).items() if isinstance(value, torch.jit.ScriptFunction)]
        assert len(scripted) == 6, f"expected six scripted functions, found {scripted}"
        quat = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
        assert module.tan_norm_to_quat(module.quat_to_tan_norm(quat)).shape == quat.shape
    finally:
        sys.meta_path.remove(blocker)


"""
Migration rules that this module encodes by omission.
"""


_KNOWN_REWRITES = frozenset(
    {"_sqrt_positive_part", "quat_from_matrix", "apply_delta_pose", "rigid_body_twist_transform"}
)


def _module_functions(source: pathlib.Path) -> dict[str, str]:
    """Every module-level function, as source text with the docstring dropped."""
    functions = {}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.FunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            functions[node.name] = ast.unparse(ast.Module(body=body, type_ignores=[]))
    return functions


def test_the_two_engines_copies_have_only_the_known_rewrites_between_them() -> None:
    """How far apart the two upstreams have drifted, asserted rather than described.

    The design quotes a count of how many of the shared functions are character-identical and names
    the handful that are not. That sentence was written from a one-off measurement and went stale
    without anything noticing -- one of the named functions had since become identical. A number
    that appears in prose and nowhere in a test is a number that is true on the day it is written.

    Only the set matters here, not the count: an upstream that rewrites a formula shows up as a new
    name, and one that reconciles a rewrite shows up as a missing one. Whether the rewrites are
    still numerically equivalent is what the value comparisons above are for.
    """
    isaac = _module_functions(pathlib.Path(_isaac_math().__file__))
    mjlab = _module_functions(pathlib.Path(_mjlab_math().__file__))

    shared = set(isaac) & set(mjlab)
    rewritten = {name for name in shared if isaac[name] != mjlab[name]}

    assert rewritten == _KNOWN_REWRITES, (
        f"the two vendored copies now differ in {sorted(rewritten)}, not {sorted(_KNOWN_REWRITES)}. "
        "A new name means an upstream rewrote a formula and compat.math may be following the other "
        "one; a missing name means the rewrite was reconciled and the design's prose is stale."
    )


def test_deprecated_rotate_aliases_are_not_vendored() -> None:
    """``quat_rotate`` is Isaac-only and deprecated, so a term using it cannot run on mjlab."""
    isaac, mjlab = _isaac_math(), _mjlab_math()
    for deprecated, replacement in (("quat_rotate", "quat_apply"), ("quat_rotate_inverse", "quat_apply_inverse")):
        assert hasattr(isaac, deprecated), f"Isaac Lab dropped {deprecated}; the migration rule can retire"
        assert not hasattr(mjlab, deprecated), f"mjlab gained {deprecated}; the migration rule needs revisiting"
        assert not hasattr(compat_math, deprecated)
        assert replacement in compat_math.__all__


def test_deprecated_rotate_aliases_are_exactly_their_replacements(sample: dict[str, torch.Tensor]) -> None:
    """The rewrite is mechanical only because the outputs are identical, not merely close."""
    isaac = _isaac_math()
    quat, vec = sample["quat"], sample["vec"]
    assert torch.equal(isaac.quat_rotate(quat, vec), compat_math.quat_apply(quat, vec))
    assert torch.equal(isaac.quat_rotate_inverse(quat, vec), compat_math.quat_apply_inverse(quat, vec))
