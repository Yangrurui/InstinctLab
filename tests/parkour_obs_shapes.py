"""Observation widths derived from the declaration, not pasted literals.

A pasted ``768`` would stay green if the task later asked for history 4 and both
engines dropped it the same way. Instantaneous width comes from the live term
function; the multiplier comes from the spec.

Image terms are compared on the full trailing shape. Using ``shape[-1]`` on an
``(N, 8, 18, 32)`` depth history would report 32 and hide a dropped frame axis.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def declared_history_length(group_spec: Any, term_spec: Any) -> int:
    """What the observation manager will keep, given both engines' ``is not None`` test."""
    if group_spec.history_length is not None:
        return group_spec.history_length
    return term_spec.history_length


def _term_cfgs(env: Any, group_name: str) -> dict[str, Any]:
    manager = env.observation_manager
    return dict(zip(manager._group_obs_term_names[group_name], manager._group_obs_term_cfgs[group_name], strict=True))


def expected_term_shape(env: Any, group_name: str, term_name: str, group_spec: Any, term_spec: Any) -> tuple[int, ...]:
    """Trailing shape the manager should expose for one term."""
    term_cfg = _term_cfgs(env, group_name)[term_name]
    instant = term_cfg.func(env, **term_cfg.params)
    history = declared_history_length(group_spec, term_spec)
    flatten = getattr(term_cfg, "flatten_history_dim", True)
    trailing = tuple(int(dim) for dim in instant.shape[1:])
    if history > 0 and flatten:
        return (int(instant[0].numel()) * history,)
    return trailing


def expected_term_width(env: Any, group_name: str, term_name: str, group_spec: Any, term_spec: Any) -> int:
    """Flattened width. Kept for callers that only need a scalar."""
    shape = expected_term_shape(env, group_name, term_name, group_spec, term_spec)
    width = 1
    for dim in shape:
        width *= dim
    return width


def assert_observation_shapes_match_declaration(env: Any, spec: Any) -> None:
    """Pin live observation shapes to the declaration, on either engine."""
    obs = env.observation_manager.compute()
    for group_name, group_spec in spec.mdp.observations.items():
        group_obs = obs[group_name]
        expected: dict[str, tuple[int, ...]] = {}
        for term_name, term_spec in group_spec.terms.items():
            shape = expected_term_shape(env, group_name, term_name, group_spec, term_spec)
            expected[term_name] = shape
            if isinstance(group_obs, Mapping):
                actual = tuple(int(dim) for dim in group_obs[term_name].shape[1:])
                assert actual == shape, (
                    f"{group_name}/{term_name}: live shape {actual} != declared {shape} "
                    f"(history={declared_history_length(group_spec, term_spec)})"
                )
        if not isinstance(group_obs, Mapping):
            actual = int(group_obs.shape[-1])
            total = sum(int(torch_prod(shape)) for shape in expected.values())
            assert actual == total, f"{group_name}: concatenated width {actual} != {total} from {expected}"


def torch_prod(shape: tuple[int, ...]) -> int:
    width = 1
    for dim in shape:
        width *= dim
    return width
