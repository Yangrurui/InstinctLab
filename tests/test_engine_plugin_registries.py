"""Independent packages can add assets and native term lowering through entry points."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from instinctlab_engine import assets as asset_module
from instinctlab_engine import registry as registry_module
from instinctlab_engine.assets import AssetRegistry
from instinctlab_engine.registry import TermRegistry


class _EntryPoint:
    def __init__(self, name: str, value):
        self.name = name
        self._value = value
        self.loads = 0

    def load(self):
        self.loads += 1
        return self._value


def test_asset_package_is_resolved_from_an_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    native = ModuleType("external_robot_assets.isaacsim")

    def resolve(engine: str, variant: str):
        assert engine == "isaacsim"
        return native, f"native_{variant}"

    entry_point = _EntryPoint("external_robot", resolve)
    monkeypatch.setattr(
        asset_module.metadata,
        "entry_points",
        lambda *, group: [entry_point] if group == "instinctlab.assets" else [],
    )

    assets = AssetRegistry()
    assert assets.native_module("external_robot/standard", "isaacsim") == (
        native,
        "native_standard",
    )
    assert assets.packages() == ("external_robot",)
    assert entry_point.loads == 1


def test_asset_registry_reports_an_uninstalled_package() -> None:
    assets = AssetRegistry(load_entry_points=False)
    with pytest.raises(KeyError, match="unknown asset package 'missing'"):
        assets.native_module("missing/standard", "mjlab")


def test_term_extensions_load_only_for_the_selected_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def register_isaac(terms: TermRegistry) -> None:
        calls.append(terms.engine)

        @terms.event("external_randomization", provides=("dr.external",))
        def build(spec, ctx):
            return spec, ctx

    isaac = _EntryPoint("isaacsim.external_randomization", register_isaac)
    mjlab = _EntryPoint(
        "mjlab.external_randomization",
        lambda _terms: pytest.fail("the unselected engine extension was imported"),
    )
    monkeypatch.setattr(
        registry_module.metadata,
        "entry_points",
        lambda *, group: [mjlab, isaac] if group == "instinctlab.engine_terms" else [],
    )

    terms = TermRegistry("isaacsim")
    assert terms.kinds("event") == frozenset({"external_randomization"})
    assert terms.provides() == {"event/external_randomization": ("dr.external",)}
    assert calls == ["isaacsim"]
    assert isaac.loads == 1
    assert mjlab.loads == 0


def test_application_declares_the_g1_asset_entry_point() -> None:
    setup_source = (
        Path(__file__).resolve().parents[1] / "source/instinctlab/setup.py"
    ).read_text()
    assert '"instinctlab.assets"' in setup_source
    assert "unitree_g1 = instinctlab.assets.unitree_g1.interface:native_module" in setup_source
