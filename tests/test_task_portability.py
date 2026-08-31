"""Task portability is measured as separate, reviewable dimensions."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

from instinctlab_engine.spec import portability_report

from tests.task_specs import task_spec

TASK_ROOT = Path(__file__).resolve().parents[1] / "source/instinctlab/instinctlab/tasks"
OVERLAY_FIELDS = {
    "engine_params",
    "profiles",
    "engine_overrides",
    "engine_extras",
}
# Flat and Rough intentionally own complete, independent concrete declarations.
# Their equivalent native friction forms therefore count once per concrete task.
REVIEWED_OVERLAY_BUDGET = 41


def _source_overlay_uses() -> list[tuple[Path, int, str]]:
    uses: list[tuple[Path, int, str]] = []
    for path in TASK_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        uses.extend(
            (path.relative_to(TASK_ROOT), node.lineno, node.arg)
            for node in ast.walk(tree)
            if isinstance(node, ast.keyword) and node.arg in OVERLAY_FIELDS
        )
    return uses


def test_task_source_overlay_budget_cannot_grow_without_architecture_review() -> None:
    uses = _source_overlay_uses()
    counts = Counter(field for _, _, field in uses)
    assert len(uses) <= REVIEWED_OVERLAY_BUDGET, (
        f"task backend-name overlays grew from the reviewed budget of "
        f"{REVIEWED_OVERLAY_BUDGET} to {len(uses)} ({dict(counts)}). "
        "Translate shared meanings into typed fields/builders, or explicitly "
        "review and update the budget for an intentional native difference."
    )
    assert counts["engine_extras"] == 0


def test_runtime_report_names_overlays_without_calling_them_native_extras() -> None:
    spec = task_spec("Instinct-Parkour-Target-G1")
    report = portability_report(spec)

    assert report["contract_portability"] == {
        "multi_engine": True,
        "declared_engines": ["isaacsim", "mjlab"],
    }
    assert report["semantic_overlays"]["count"] > 0
    assert {entry["path"] for entry in report["semantic_overlays"]["entries"]} >= {
        "sim.profiles",
        "mdp.event/physics_material.engine_params",
    }
    assert report["native_extras"] == {"count": 0, "entries": []}
    assert report["clean_resolution"]["known"] is False


def test_clean_resolution_is_independent_of_declaration_overlays() -> None:
    spec = task_spec("Instinct-Velocity-Flat-G1")
    clean = portability_report(
        spec,
        {"skipped": {}, "emulated": {}, "omitted": {}},
    )
    nonclean = portability_report(
        spec,
        {
            "skipped": {"reward/rewards/example": "missing"},
            "emulated": {},
            "omitted": {},
        },
    )

    assert clean["clean_resolution"] == {
        "known": True,
        "clean": True,
        "skipped": 0,
        "emulated": 0,
        "omitted": 0,
    }
    assert nonclean["clean_resolution"]["clean"] is False
    assert nonclean["semantic_overlays"] == clean["semantic_overlays"]
