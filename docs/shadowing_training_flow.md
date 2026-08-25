# Shadowing training, resume, play and export

The unified launchers consume the same registered `TaskSpec` for training and playback.  The
selected engine validates and lowers that declaration, constructs its native scene and managers,
and exposes the same canonical observation/action schema to Instinct-RL.  There is no
shadowing-specific launcher or alternate play configuration.

## Lifecycle contract

The exercised path is:

1. resolve the registered task and validate its shared declaration;
2. compile native scene, sensors, managers and the engine-neutral agent configuration;
3. construct and reset the environment, including motion and command sampling;
4. build policy and critic observations in declaration order;
5. map canonical policy actions into native joint order, step physics and refresh sensors;
6. compute rewards and terminations, then reset completed environments;
7. wrap the environment once for Instinct-RL and construct `OnPolicyRunner`;
8. write a checkpoint beside a compilation manifest containing the task contract;
9. validate that contract before resume/play, restore runner state, and export the same policy.

`scripts/play.py` disables observation noise after compilation but does not replace sensors,
actuators, observations or the task declaration. ONNX export is available only for a trained
checkpoint, validates the adjacent contract before tensor loading, exports the policy normalizer,
and writes `export.json` with the source checkpoint and task contract.

## Resume and distributed behavior

Every distributed rank maps `cuda` to its `LOCAL_RANK`, receives seed `base_seed + global_rank`,
and shares rank zero's timestamped log directory.  The manifest records the world size and all
rank seeds.  Resume restores model/noise state, optimizer moments, observation normalizers and the
completed iteration on every rank; relying on the upstream rank-zero-only loader would leave
optimizer state inconsistent.

Neither reference checkpoint contains simulator, episode, command RNG or motion-runtime buffers.
Resume therefore creates a fresh environment and resamples motion/commands.  The manifest records
this explicitly as `resume_environment_state`; exact continuation of a physical episode is not
claimed.  Process-group initialization and shutdown are owned by the generic training launcher,
and manifest/agent writes are rank-zero-only.

## Live evidence

The production shadow datasets configured by the references are absent on this host.  Live probes
therefore change only the dataset binding to the available parkour reference clip and retain the
whole-body scene and MDP.  GPU 2 remained occupied by the pre-existing parkour training process;
all probes used GPU 1.

The following artifacts were generated with seed 2026:

- `evidence/shadowing_rollout_isaacsim_seed2026.npz` and
  `evidence/shadowing_rollout_mjlab_seed2026.npz`: two environments, native reset followed by a
  forced non-mirrored clip at 0.25 s, identical 29-dimensional action, and four physics steps;
- `evidence/shadowing_rollout_comparison_seed2026.json`: value-level comparison of the captured
  joint/root/reference/reward/done tensors;
- `evidence/shadowing_runner_isaacsim_seed2026.json` and
  `evidence/shadowing_runner_mjlab_seed2026.json`: one-environment runner construction,
  checkpoint save, contract validation, reload, finite inference action, native step, and ONNX plus
  policy-normalizer export. Large checkpoints/ONNX files remain temporary; reports record their
  sizes and SHA-256 hashes.

At the forced initial state, action, motion start, joint position/velocity, root position,
quaternion and motion position match exactly. Root velocity differs by at most 1.79e-7. Once
physics advances, PhysX and MuJoCo trajectories intentionally diverge: in this four-step probe the
Isaac reference-distance termination resets after step one while MJLab resets after step four.
States following the first auto-reset are not comparable and the evidence report does not present
them as portable values.

## Remaining validation boundary

Static and live smoke evidence establishes wiring and short-horizon runtime behavior, not learned
policy equivalence. Production dataset inventory, production CoACD terrain caches, camera values on
those meshes, multi-node communication under real load, long-run checkpoint reproducibility,
training stability, return distributions and final policy quality still require runs with the
reference datasets and normal training duration. PhysX contact/applied-torque/COM quantities and
MuJoCo contact/`qfrc_actuator`/root-link quantities remain engine-native as documented in
`shadowing_sensor_mdp_contract.md`.
