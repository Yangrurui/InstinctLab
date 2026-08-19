"""Every module a legacy Gym registration points at has to exist.

``gym.register`` stores its ``kwargs`` as strings and imports nothing, so an entry point naming a
module that was never written registers cleanly and stays wrong until somebody asks for it. Three
already do. They cause no harm today because the repo standardised on ``instinct_rl`` and nothing
resolves the ``rsl_rl`` ones -- which is exactly why they survived: the failure needs a reader.

Resolution is static. The strings are f-strings over a module-level constant and ``agents.__name__``,
both of which can be read off the syntax tree, so this runs without Isaac Sim and covers every task
rather than the ones an installed engine happens to let us import.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SOURCE = REPO / "source/instinctlab"
TASKS = SOURCE / "instinctlab/tasks"

KNOWN_DEAD = {
    "instinctlab.tasks.shadowing.whole_body.config.g1.agents.rsl_rl_ppo_cfg",
    "instinctlab.tasks.shadowing.perceptive.config.g1.agents.rsl_rl_ppo_cfg",
    "instinctlab.tasks.shadowing.perceptive_hoi.config.g1.agents.rsl_rl_ppo_cfg",
}
"""Dangling on main too, in files this repo keeps verbatim.

Left alone rather than deleted: D3 makes these files main's, and the tasks holding them are still
Isaac-only. Naming them here is what stops a fourth from joining quietly.
"""


def _string_constants(tree: ast.Module) -> dict[str, str]:
    return {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _resolve(node: ast.expr, constants: dict[str, str], package: str) -> str | None:
    """Flatten the f-string an entry point is written as, or give up rather than guess."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if not isinstance(node, ast.JoinedStr):
        return None
    parts: list[str] = []
    for piece in node.values:
        if isinstance(piece, ast.Constant):
            parts.append(str(piece.value))
        elif isinstance(piece, ast.FormattedValue) and isinstance(piece.value, ast.Name):
            if piece.value.id not in constants:
                return None
            parts.append(constants[piece.value.id])
        elif (
            isinstance(piece, ast.FormattedValue)
            and isinstance(piece.value, ast.Attribute)
            and piece.value.attr == "__name__"
            and isinstance(piece.value.value, ast.Name)
        ):
            parts.append(f"{package}.{piece.value.value.id}")
        else:
            return None
    return "".join(parts)


def _entry_points() -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for path in sorted(TASKS.rglob("__init__.py")):
        tree = ast.parse(path.read_text())
        constants = _string_constants(tree)
        package = ".".join(path.relative_to(SOURCE).parent.parts)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "register"):
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            task_id = keywords.get("id")
            mapping = keywords.get("kwargs")
            if not isinstance(mapping, ast.Dict) or not isinstance(task_id, ast.Constant):
                continue
            for key, value in zip(mapping.keys, mapping.values):
                if not (isinstance(key, ast.Constant) and str(key.value).endswith("_cfg_entry_point")):
                    continue
                resolved = _resolve(value, constants, package)
                if resolved is not None:
                    found.append((str(task_id.value), str(key.value), resolved))
    return found


ENTRY_POINTS = _entry_points()


def test_the_scan_found_the_registrations_at_all() -> None:
    """Parametrising over a computed list means an empty list is a silent pass."""
    assert len(ENTRY_POINTS) >= 28, f"only {len(ENTRY_POINTS)} entry points parsed; the scan is no longer finding them"


@pytest.mark.parametrize(("task_id", "role", "target"), ENTRY_POINTS, ids=lambda value: str(value))
def test_every_entry_point_names_a_module_that_exists(task_id: str, role: str, target: str) -> None:
    module = target.partition(":")[0]
    path = SOURCE / pathlib.Path(*module.split("."))
    exists = path.with_suffix(".py").exists() or (path / "__init__.py").exists()
    if module in KNOWN_DEAD:
        assert not exists, f"{module} exists now; drop it from KNOWN_DEAD so the list keeps meaning something"
        pytest.skip(f"{role} of {task_id} is a recorded dead link")
    assert exists, (
        f"{task_id}'s {role} points at {module}, which is not a module. gym.register stores this as "
        "a string and imports nothing, so nothing else will tell you."
    )
