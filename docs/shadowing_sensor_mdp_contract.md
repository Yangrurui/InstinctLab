# Shadowing sensor and MDP contract

This record describes the effective configuration compiled from the shared shadowing `TaskSpec`.
The Isaac reference is `/root/InstinctLab-main` at `ba28d3d`; the MJLab reference is
`/root/InstinctMJ` at `4ed2b32`.

## Reference-matched behavior

- The perceptive camera renders 27×48 rays at 87°×58°, updates at 60 Hz, rejects hits within
  0.05 m, crops two pixels per edge, and supplies 18×32 depth normalized over 0–2 m. VAE history
  is ten sensor frames, sampled every third frame into four outputs. History advances only when
  the native camera publishes a new frame and is cleared per environment on reset.
- The height scanner has the reference 187-ray layout and 20 ms update. Its native maximum range
  remains 1e6 m on Isaac and 5 m on MJLab.
- Contact layout is declared by body name. The non-support selector excludes both ankle-roll and
  both wrist-yaw links while preserving the native sensor's element order. Contact/air time uses
  the 1 N threshold; illegal reset contact uses 500 N for the first two episode steps.
- Commands sample once per episode from the shared motion runtime. Policy position references use
  the base-relative anchor command; critic references use world position. BeyondMimic has no
  anchor command and therefore uses the world position reference on both sides.
- Reset uses the separately floor-indexed motion endpoint and writes canonical joint state once.
  Motion history is rebuilt, `last_update` is synchronized, exhaustion freezes and flags the last
  sample, and HOI object slots are mapped by object name and cleared when invalid.
- Failed motion bins are accumulated at episode reset, exponentially smoothed every environment
  step with alpha 0.001, forward-smoothed with the 3-tap `[1, .8, .64]` kernel, mixed with 10%
  uniform mass, and normalized either per motion or over concatenated motion bins as declared.
- MJLab perceptive terrain uses the InstinctMJ CoACD profile: threshold 0.04, resolution 3000,
  maximum convex-hull vertex budget 256, no decimation, disk cache, zero margin/offset, top-surface
  alignment, and visible collision hulls. `coacd==1.0.7` is part of the MJLab install extra.
- HOI declares the six reference rigid objects in stable order and includes them in Isaac camera
  hit targets. MJLab sees the same objects through its native geom-group ray filter.

## Intentional engine differences

These quantities match their respective reference rather than pretending to be numerically
portable:

- Isaac contact penalties threshold PhysX per-body net force; MJLab thresholds MuJoCo contact
  force after its native matching/reduction. Friction cone and solver contact construction differ,
  so identical thresholds do not imply identical trigger sets near the boundary.
- Isaac torque-limit penalties read PhysX `applied_torque`; MJLab reads MuJoCo
  `qfrc_actuator`. Gear/transmission and constraint accounting make these values non-equivalent.
- Isaac's native shadowing critic base velocity keeps main's root-COM accessor. MJLab keeps
  InstinctMJ's root-link accessor. The shared declaration selects an engine semantic term rather
  than silently substituting one frame for the other.
- Isaac joint acceleration is the backend's finite-difference quantity; MJLab exposes MuJoCo's
  analytic `qacc`. Shadowing does not currently reward either, but they must not be used as golden
  cross-engine equalities.
- PhysX and MuJoCo solver iteration, friction, restitution, armature, actuator force, and delay-bus
  models remain native. Their configuration is tested against the corresponding reference, while
  short cross-engine rollouts are treated as diagnostic envelopes rather than exact trajectories.

## Evidence boundary

Static contract/configuration tests, synthetic motion value tests, reset/exhaustion tests, sensor
layout tests, and CPU compilation cover the behavior above. The referenced production motion and
terrain datasets are not present on this machine, so dataset-inventory equality, CoACD cache
content, camera images over those meshes, and long-horizon policy performance require validation
where those datasets are mounted. Chaotic physics trajectories are not claimed to match pointwise
between engines.
