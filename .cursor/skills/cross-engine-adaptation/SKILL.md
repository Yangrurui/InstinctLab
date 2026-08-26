---
name: cross-engine-adaptation
description: Audit or change InstinctLab's unified Isaac Sim/MJLab tasks, shared TaskSpec/MDP/motion/sensor layers, engine adapters, reference parity, or simulator stack. Use for cross-engine bugs, task migration, AMP/motion alignment, joint ordering, actuator/contact parity, engine upgrades, and training-curve divergence. Do not use for ordinary task tuning that does not question engine semantics.
---

# Cross-engine adaptation

Read [`AGENTS.md`](../../../AGENTS.md) first. Read [`HANDOFF.md`](../../../HANDOFF.md)
when the work depends on current branches, dependency versions, datasets,
active experiments, or unresolved findings. Those files own project state; do
not duplicate it here.

## Establish the comparison contract

Before editing, state which relation is being tested:

- Isaac implementation against the effective main runtime.
- MJLab implementation against the effective InstinctMJ runtime.
- Both engines against the shared `TaskSpec` interface.
- Native physics behavior across PhysX and MuJoCo.

These are different claims. The first three can be aligned; the fourth often
cannot. Read executed configuration and runtime output, not merely assignments
that may later be overwritten.

## Audit from boundaries inward

Use this order unless evidence points to a narrower fault:

1. Environment and numeric stack: exact package versions, Torch flags, seed,
   device, physics dt, decimation, solver/contact settings.
2. Assets and actuation: model files, joint/body inventory, defaults, limits,
   gains, armature, effort, motor-bus delay grouping, action scaling.
3. State boundaries: canonical DFS order versus native order, quaternion
   convention, velocity point/frame, history order, reset writes.
4. Sensors and motion: element matching, contact fields, update clocks, cache
   invalidation, motion selection, mirroring, exhaustion, current/reference
   frame semantics.
5. MDP and algorithm: observation order/scale/noise, reward and termination
   timing, runner config, discriminator inputs and learned outputs.
6. Full rollout and training: episode length, termination mix, action noise,
   per-term reward, per-terrain behavior, overflow, then total return.

Never infer an earlier-layer cause from a later-layer curve when the earlier
quantity can be measured directly.

## Non-negotiable invariants

- Shared code does not import an engine SDK at module import time.
- `RobotSpec.joint_names` is canonical DFS. Resolve and gather by name at every
  native boundary; `preserve_order=True` with a lone `.*` does not pin order.
- A velocity name must identify both the moving point and coordinate frame.
- Contact presence uses the portable contact contract, not cross-engine force
  magnitude thresholds. Native contact force tensors are not equivalent.
- Actuator delay correlation follows declared motor groups, not coincidental PD
  gain equality.
- AMP actor and reference use the same term builder, order, defaults, scale,
  and history. Motion look-ahead `data` may begin at `t+dt`; AMP must use the
  dedicated current-time `reference_frame` at `t`.
- Keep reference-specific settings in engine adapters when main and InstinctMJ
  disagree. Do not force a shared compromise into `train.py`.
- Do not treat equal seed as equal random assignment across engines unless the
  tested subsystem uses an explicit isolated generator.

## Evidence ladder

Use the cheapest layer that can answer the question, then add temporal evidence
when the property lives on a timeline:

1. Static declaration and import isolation.
2. Reference extraction from the actual upstream checkout.
3. Fixed-state term comparison without stepping.
4. One-step plant probes with identical written state/control.
5. Multi-step contact, delay, sensor, reset, and motion lifecycle probes.
6. Small live construction and rollout in separate engine processes.
7. Matched short training, followed by production-scale checks where required.

Compare episode length and termination behavior, not only reward. Normalize
episode-summed metrics before interpreting ratios. A discriminator is learned
per run, so its reward/logit scale is not an absolute cross-run ruler; replaying
the same actor and discriminator in both engines is the relevant isolation.

For detailed blind spots and probe selection, read
[`verification.md`](verification.md). For recurring silent-failure patterns,
read [`silent-failures.md`](silent-failures.md). When adding a third engine or
upgrading an engine contract, also read [`new-engine.md`](new-engine.md).

## Completion

- Add a regression test that fails under the old behavior and asserts the
  physical/semantic effect, not merely a config flag.
- Run focused tests, then the shared suite appropriate to the touched layer.
- Record exact versions and measured values for physics-sensitive conclusions.
- Keep diagnostics separate from production behavior where practical.
- Update `HANDOFF.md` only for durable state changes or new unresolved risks.
- Commit an independently reversible, verified increment. Do not stop live
  training unless the user explicitly authorized it.
