"""Guard: the vendored ``configclass`` is Isaac Lab's, unchanged.

:mod:`instinctlab.utils.configclass` is a copy, taken so that a PPO learning rate can be read on a
machine that is training with mjlab. A copy only stays trustworthy while it stays a copy, and the
failure mode of a drifted one is quiet: a config would still construct, still convert to a dict,
and differ from main's somewhere subtle, like whether a mutable default is shared between two
instances.

Two checks, because they catch different drift. The first compares the vendored function bodies
against Isaac Lab's source, which needs no Isaac Sim because it reads the file. The second
constructs the real agent config both ways and compares the dictionaries, which does need Isaac
Sim and so is skipped without it.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import instinctlab.utils.configclass as vendored

_UPSTREAM = pathlib.Path("/root/IsaacLab/source/isaaclab/isaaclab/utils")

_COPIED = {
    "configclass.py": (
        "__dataclass_transform__",
        "configclass",
        "_add_annotation_types",
        "_process_mutable_types",
        "_custom_post_init",
        "_combined_function",
        "_class_to_dict",
        "_update_class_from_dict",
        "_replace_class_with_kwargs",
        "_copy_class",
        "_validate",
        "_skippable_class_member",
        "_return_f",
    ),
    "dict.py": ("class_to_dict", "update_class_from_dict"),
    "string.py": ("callable_to_string", "string_to_callable", "string_to_slice"),
}


def _functions(source: str) -> dict[str, str]:
    """Every module-level function in ``source``, keyed by name, dumped as normalised AST.

    Compared as AST rather than text so that reformatting by this repository's own hooks -- which
    run on the vendored file and not on Isaac Lab's -- is not mistaken for a semantic edit. Nested
    helpers are not listed separately; they are inside the dump of the function that defines them.
    """
    return {
        node.name: ast.dump(node, annotate_fields=True)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }


@pytest.fixture(scope="module")
def vendored_functions() -> dict[str, str]:
    return _functions(pathlib.Path(vendored.__file__).read_text())


@pytest.mark.skipif(not _UPSTREAM.exists(), reason="Isaac Lab source not present")
@pytest.mark.parametrize(
    ("filename", "name"),
    [(filename, name) for filename, names in _COPIED.items() for name in names],
)
def test_the_vendored_function_is_identical_to_isaac_labs(
    filename: str, name: str, vendored_functions: dict[str, str]
) -> None:
    upstream = _functions((_UPSTREAM / filename).read_text())
    if name not in upstream:
        pytest.skip(f"{name} is not in this version of {filename}")
    assert name in vendored_functions, f"{name} was dropped from the vendored copy"
    assert vendored_functions[name] == upstream[name], (
        f"{name} differs from isaaclab.utils.{filename[:-3]}. The copy exists to be identical; "
        "re-vendor it rather than editing it in place."
    )


def test_every_copied_function_is_accounted_for(vendored_functions: dict[str, str]) -> None:
    """The vendored file declares nothing beyond what it copied."""
    declared = set(vendored_functions)
    expected = {name for names in _COPIED.values() for name in names}
    extra = declared - expected
    assert not extra, f"the vendored module declares {sorted(extra)}, which came from nowhere upstream"


def test_the_vendored_module_imports_no_engine() -> None:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(pathlib.Path(vendored.__file__).read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    engines = {"isaaclab", "isaacsim", "mjlab", "omni", "pxr", "carb", "mujoco", "warp"}
    assert not (imported & engines), "the whole point of the copy is that it needs no engine"


def test_mutable_defaults_are_not_shared_between_instances() -> None:
    """The behaviour the decorator exists for, and the one a plain dataclass gets wrong."""

    @vendored.configclass
    class Cfg:
        items = [1, 2]

    first, second = Cfg(), Cfg()
    first.items.append(3)
    assert second.items == [1, 2]


def test_unannotated_attributes_become_fields() -> None:
    """A plain dataclass ignores these; every config in this repo relies on them being fields."""

    @vendored.configclass
    class Cfg:
        name = "policy"
        depth: int = 3

    assert vendored.class_to_dict(Cfg()) == {"name": "policy", "depth": 3}


@pytest.mark.skipif(not _UPSTREAM.exists(), reason="Isaac Lab source not present")
def test_the_real_agent_config_converts_identically_either_way() -> None:
    """The end-to-end claim: same declaration, same dictionary, whichever decorator built it.

    Requires a launched Isaac Sim, since building the comparison class means importing
    ``isaaclab.utils``. Without one this skips -- the AST comparison above still runs.
    """
    isaaclab_utils = pytest.importorskip("isaaclab.utils", reason="needs a launched Isaac Sim")

    from instinctlab.tasks.locomotion import flat_g1_ppo

    theirs = isaaclab_utils.configclass(type("Probe", (flat_g1_ppo.G1FlatPPORunnerCfg,), {}))
    assert theirs().to_dict() == flat_g1_ppo.G1FlatPPORunnerCfg().to_dict()
