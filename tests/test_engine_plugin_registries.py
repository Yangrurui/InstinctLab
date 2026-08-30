"""Independent packages can add assets and native term lowering through entry points."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import instinctlab_engine as engine_module
import pytest
from instinctlab_engine import assets as asset_module
from instinctlab_engine import plugins as plugin_module
from instinctlab_engine import registry as registry_module
from instinctlab_engine.assets import AssetRegistry
from instinctlab_engine.base import Resolution
from instinctlab_engine.bridge import entity as entity_module
from instinctlab_engine.plugins import PluginDiscoveryError, plugin_provenance
from instinctlab_engine.registry import TermRegistry, TerrainExtensionRegistry


class _Distribution:
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version


class _EntryPoint:
    def __init__(
        self,
        name: str,
        value,
        *,
        distribution: str = "test-plugin",
        version: str = "1.0",
        target: str = "test_plugin:register",
    ):
        self.name = name
        self._value = value
        self.loads = 0
        self.dist = _Distribution(distribution, version)
        self.value = target

    def load(self):
        self.loads += 1
        return self._value


@pytest.fixture(autouse=True)
def _restore_plugin_provenance():
    snapshot = plugin_module._snapshot_provenance()
    yield
    plugin_module._restore_provenance(snapshot)


def test_asset_package_is_resolved_from_an_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_broken_term_plugin_is_rolled_back_and_failure_is_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def register_broken(terms: TermRegistry) -> None:
        @terms.event("partial_registration")
        def build(spec, ctx):
            return spec, ctx

        raise RuntimeError("registrar exploded")

    entry_point = _EntryPoint(
        "isaacsim.broken",
        register_broken,
        distribution="broken-randomizers",
        version="2.3",
    )
    monkeypatch.setattr(
        registry_module.metadata,
        "entry_points",
        lambda *, group: [entry_point] if group == "instinctlab.engine_terms" else [],
    )

    terms = TermRegistry("isaacsim")
    with pytest.raises(PluginDiscoveryError, match="broken-randomizers") as first:
        terms.kinds("event")
    assert terms._builders == {}
    assert terms._provides == {}

    with pytest.raises(PluginDiscoveryError) as second:
        terms.kinds("event")
    assert second.value is first.value
    assert entry_point.loads == 1


def test_broken_terrain_plugin_is_rolled_back_and_failure_is_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def register_broken(terrains: TerrainExtensionRegistry) -> None:
        terrains.register_terrain("isaacsim", "partial", lambda spec, profile: spec)
        raise RuntimeError("terrain registrar exploded")

    entry_point = _EntryPoint(
        "broken_terrain",
        register_broken,
        distribution="broken-terrains",
    )
    monkeypatch.setattr(
        registry_module.metadata,
        "entry_points",
        lambda *, group: [entry_point] if group == "instinctlab.terrains" else [],
    )

    terrains = TerrainExtensionRegistry()
    with pytest.raises(PluginDiscoveryError, match="broken-terrains") as first:
        terrains.terrain_kinds("isaacsim")
    assert terrains._terrains == {}
    assert terrains._origins == {}

    with pytest.raises(PluginDiscoveryError) as second:
        terrains.terrain_kinds("isaacsim")
    assert second.value is first.value
    assert entry_point.loads == 1


def test_broken_engine_plugin_rolls_back_every_shared_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def register_alpha() -> None:
        engine_module.register_adapter("atomic_alpha", "fake_alpha.adapter:Adapter")
        engine_module.register_terrain(
            "atomic_alpha", "flat", lambda spec, profile: spec
        )

    def register_beta() -> None:
        engine_module.register_adapter("atomic_beta", "fake_beta.adapter:Adapter")
        raise RuntimeError("backend registrar exploded")

    alpha = _EntryPoint(
        "atomic_alpha", register_alpha, distribution="atomic-alpha-backend"
    )
    beta = _EntryPoint("atomic_beta", register_beta, distribution="atomic-beta-backend")

    adapter_snapshot = dict(engine_module.ADAPTERS)
    source_snapshot = dict(engine_module._ADAPTER_SOURCES)
    loaded_snapshot = engine_module._engine_entry_points_loaded
    error_snapshot = engine_module._engine_entry_point_error
    entity_snapshot = entity_module._snapshot_registrations()
    terrain_snapshot = registry_module.TERRAIN_EXTENSIONS._snapshot()
    provenance_snapshot = plugin_module._snapshot_provenance()
    monkeypatch.setattr(
        engine_module.metadata,
        "entry_points",
        lambda *, group: [beta, alpha] if group == "instinctlab.engines" else [],
    )
    engine_module._engine_entry_points_loaded = False
    engine_module._engine_entry_point_error = None

    try:
        with pytest.raises(PluginDiscoveryError, match="atomic-beta-backend") as first:
            engine_module.names()
        assert engine_module.ADAPTERS == adapter_snapshot
        assert engine_module._ADAPTER_SOURCES == source_snapshot
        assert registry_module.TERRAIN_EXTENSIONS._snapshot() == terrain_snapshot
        assert entity_module._snapshot_registrations() == entity_snapshot

        with pytest.raises(PluginDiscoveryError) as second:
            engine_module.names()
        assert second.value is first.value
        assert alpha.loads == 1
        assert beta.loads == 1
    finally:
        engine_module.ADAPTERS.clear()
        engine_module.ADAPTERS.update(adapter_snapshot)
        engine_module._ADAPTER_SOURCES.clear()
        engine_module._ADAPTER_SOURCES.update(source_snapshot)
        engine_module._engine_entry_points_loaded = loaded_snapshot
        engine_module._engine_entry_point_error = error_snapshot
        engine_module._active_engine_plugin = None
        entity_module._restore_registrations(entity_snapshot)
        registry_module.TERRAIN_EXTENSIONS._restore(terrain_snapshot)
        plugin_module._restore_provenance(provenance_snapshot)


def test_duplicate_asset_plugins_report_both_distributions_and_roll_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_resolver = lambda engine, variant: (ModuleType(engine), variant)
    second_resolver = lambda engine, variant: (ModuleType(engine), variant)
    first = _EntryPoint(
        "duplicate_robot",
        first_resolver,
        distribution="first-assets",
    )
    second = _EntryPoint(
        "duplicate_robot",
        second_resolver,
        distribution="second-assets",
    )
    monkeypatch.setattr(
        asset_module.metadata,
        "entry_points",
        lambda *, group: [first, second] if group == "instinctlab.assets" else [],
    )

    assets = AssetRegistry()
    with pytest.raises(PluginDiscoveryError) as error:
        assets.packages()
    assert "first-assets" in str(error.value)
    assert "second-assets" in str(error.value)
    assert assets._resolvers == {}
    assert assets._resolver_sources == {}


def test_unsupported_plugin_core_api_is_rejected_without_partial_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def resolve(engine: str, variant: str):
        return ModuleType(engine), variant

    resolve.instinctlab_engine_api = ">=9,<10"
    entry_point = _EntryPoint(
        "future_robot",
        resolve,
        distribution="future-assets",
        version="9.1",
    )
    monkeypatch.setattr(
        asset_module.metadata,
        "entry_points",
        lambda *, group: [entry_point] if group == "instinctlab.assets" else [],
    )

    assets = AssetRegistry()
    with pytest.raises(PluginDiscoveryError) as error:
        assets.packages()
    message = str(error.value)
    assert "future-assets" in message
    assert "9.1" in message
    assert ">=9,<10" in message
    assert assets._resolvers == {}


def test_two_term_extensions_for_one_backend_are_composed_and_attributed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def register_alpha(terms: TermRegistry) -> None:
        @terms.event("alpha")
        def build_alpha(spec, ctx):
            return spec, ctx

    def register_beta(terms: TermRegistry) -> None:
        @terms.event("beta")
        def build_beta(spec, ctx):
            return spec, ctx

    alpha = _EntryPoint(
        "isaacsim.alpha", register_alpha, distribution="alpha-randomizers"
    )
    beta = _EntryPoint("isaacsim.beta", register_beta, distribution="beta-randomizers")
    monkeypatch.setattr(
        registry_module.metadata,
        "entry_points",
        lambda *, group: [beta, alpha] if group == "instinctlab.engine_terms" else [],
    )

    terms = TermRegistry("isaacsim")
    assert terms.lookup("event", "alpha") is not None
    assert terms.lookup("event", "beta") is not None
    manifest = Resolution(engine="isaacsim", task_id="test").manifest()
    assert [entry["distribution"] for entry in manifest["plugins"]] == [
        "alpha-randomizers",
        "beta-randomizers",
    ]
    assert manifest["plugins"][0]["registered_keys"] == ["isaacsim:kind:event:alpha"]
    assert manifest["plugins"][1]["registered_keys"] == ["isaacsim:kind:event:beta"]


def test_unused_plugin_is_not_recorded_as_affecting_a_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def register_terms(terms: TermRegistry) -> None:
        @terms.event("used")
        def build_used(spec, ctx):
            return spec, ctx

        @terms.event("unused")
        def build_unused(spec, ctx):
            return spec, ctx

    entry_point = _EntryPoint("mjlab.extra", register_terms)
    monkeypatch.setattr(
        registry_module.metadata,
        "entry_points",
        lambda *, group: [entry_point] if group == "instinctlab.engine_terms" else [],
    )

    terms = TermRegistry("mjlab")
    assert terms.lookup("event", "used") is not None
    provenance = plugin_provenance(engine="mjlab")
    assert len(provenance) == 1
    assert provenance[0]["registered_keys"] == [
        "mjlab:kind:event:unused",
        "mjlab:kind:event:used",
    ]


def test_application_declares_the_g1_asset_entry_point() -> None:
    setup_source = (
        Path(__file__).resolve().parents[1] / "source/instinctlab/setup.py"
    ).read_text()
    assert '"instinctlab.assets"' in setup_source
    assert (
        "unitree_g1 = instinctlab.assets.unitree_g1.interface:native_module"
        in setup_source
    )
