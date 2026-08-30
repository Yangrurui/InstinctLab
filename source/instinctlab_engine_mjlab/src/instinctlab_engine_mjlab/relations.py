"""MuJoCo lowering for portable scene relations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any


def with_collision_exclusions(entity_cfg: Any, exclusions: Sequence[Any]) -> Any:
    """Ensure declared body exclusions are present in the entity MjSpec."""
    exclusions = tuple(exclusions)
    if not exclusions:
        return entity_cfg
    original_spec_fn = entity_cfg.spec_fn

    def spec_fn():
        spec = original_spec_fn()
        existing = {
            tuple(sorted((item.bodyname1, item.bodyname2)))
            for item in spec.excludes
        }
        for exclusion in exclusions:
            if exclusion.pair not in existing:
                spec.add_exclude(
                    name=f"instinctlab_exclude_{exclusion.body_a}_{exclusion.body_b}",
                    bodyname1=exclusion.body_a,
                    bodyname2=exclusion.body_b,
                )
        return spec

    return replace(entity_cfg, spec_fn=spec_fn)


__all__ = ["with_collision_exclusions"]
