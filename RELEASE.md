# Release policy and checklist

InstinctLab publishes four coordinated distributions:
`instinctlab-engine-core`, `instinctlab-engine-isaacsim`,
`instinctlab-engine-mjlab`, and `instinctlab`. All four use the same semantic
version. Backend and application dependencies pin that exact coordinated
version; a release never mixes versions from different source revisions.

## Public API and plugin compatibility

Package versions follow semantic versioning. The public surface consists of
documented imports, `TaskSpec` and `RobotSpec` schemas, engine adapter methods,
entry-point groups, provider call contracts, report/file schema versions, and
the train/play command-line interface.

`ENGINE_CORE_API_VERSION` governs all `instinctlab.*` plugin entry points.
External providers must declare their supported core API range. The native
asset boundary is additionally governed by `NATIVE_ASSET_API_VERSION`. These
API versions use `major.minor` and match the coordinated package release's
major/minor version. A breaking provider or schema change increments the major
version. A backward-compatible extension increments the minor version. Patch
releases do not change a public call or data contract.

A public API is deprecated with a `DeprecationWarning`, migration text, and a
release-note entry for at least one coordinated minor release before removal.
Silent compatibility fallbacks are not allowed. Versioned manifests,
preflight reports, snapshots, and traces remain fail-closed; a format is
removed only after a newer reader and an explicit migration path have shipped.
Experimental modules must be named as such and are not covered by this policy.

## Required gates

Every release candidate must satisfy all of the following on one source commit:

1. The SDK-free pull-request suite, Python 3.11 Pyright check, release metadata
   check, and Ruff ratchet pass.
2. Core-only, Isaac-only, MJLab-only, and dual-backend isolated wheel matrices
   pass, including external extension install/exercise/uninstall.
3. The self-hosted GPU workflow passes the real Isaac Sim and MJLab external
   extension probes and the selected live backend checks.
4. The dependency installer accepts clean pinned Isaac Lab and MJLab checkouts
   without an override, and its provenance receipt is archived.
5. Full tests, task declaration/preflight checks, asset conformance, and any
   change-specific fixed-state, temporal, contact, or capacity probes pass.
6. Release hardware has reviewed lifecycle benchmark threshold documents; both
   engine reports pass those thresholds at the intended environment count.
7. `HANDOFF.md` contains no unresolved P0/P1 release blocker and records the
   exact accepted evidence, datasets, external revisions, and known limits.

## Build and publication

Use Python 3.11 in a clean checkout. Do not build from a dirty tree.

```bash
python scripts/check_release.py --expected-version VERSION
python scripts/build_release.py --expected-version VERSION
python -m twine upload dist/release/*.whl dist/release/*.tar.gz
```

`build_release.py` refuses a non-empty output directory, builds a wheel and
source distribution for each package from clean temporary source copies, uses
an isolated pinned build/twine tool environment, runs `twine check`, and writes
artifact sizes and SHA-256 digests to `dist/release/SHA256SUMS.json`. Install a
current `twine` only for the final upload command. Archive that file,
the workflow URLs, wheel-matrix logs, GPU reports, installer receipt, and
lifecycle threshold reports with the release. Publish only after a maintainer
verifies the tag points at the tested commit and the index contains all four
coordinated distributions.

After publication, create a fresh Python 3.11 environment, install the four
artifacts from the index without editable sibling state, rerun the isolated
wheel probes, and record the result in `HANDOFF.md`. If any post-publication
probe fails, stop promotion and publish a coordinated corrective release; do
not replace artifacts under an existing version.
