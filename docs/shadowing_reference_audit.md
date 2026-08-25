# Shadowing reference audit

This audit is the input contract for the unified shadowing rewrite.  It was made from the
effective registration/factory chains in the two sibling checkouts, not from this branch's Git
history:

- Isaac/main: `/root/InstinctLab-main` at `ba28d3d2655b15a19b729476a630937a19610a3b`.
- MJLab/InstinctMJ: `/root/InstinctMJ` at `4ed2b32f8719ff9fc138708341031e935afda0d2`.

InstinctMJ had unrelated local edits in `manager_based_rl_env.py` and `scripts/instinct_rl/play.py`
when audited.  Neither file is used as a configuration reference.  InstinctLab's worktree was
clean at the start of the audit.  GPU 2 was occupied by PID 322939 running the existing 4096-env
parkour training job and is excluded from shadowing tests.

## Inventory and effective entry points

The legacy tree contains four task families: `whole_body`, `perceptive`, `perceptive_hoi`, and
`beyondmimic`.  Each family contains a base env configuration, G1 final overrides, Gym registration
and an Instinct-RL agent.  Perceptive also contains the VAE policy/configuration.  The Isaac-only
launch surface is `tasks/shadowing/play.py`, `cli_args.py`, and `grid_search.sh`; generic
`scripts/instinct_rl/play.py` and `scripts/multi_play.py` also reach the legacy Gym registrations.

Code shared outside the task tree is spread over:

- `envs/mdp/commands/shadowing_command.py`;
- `envs/mdp/rewards/{shadowing_command,motion_reference}.py`;
- `envs/mdp/{events,observations,curriculums,terminations}/motion_reference.py`;
- `motion_reference/` (manager, buffers, file loaders and generators);
- shadowing monitor terms in `monitors/monitors.py`.

The final entry point is the G1 class/factory, not the base class.  Main registers its classes with
Gym.  InstinctMJ calls `register_instinct_task`, lazily runs its G1 factory and passes `play` into
the factory where applicable.  Both references expose these ten common IDs:

- `Instinct-Shadowing-WholeBody-Plane-G1-v0` and its `-Play-v0` form;
- `Instinct-Perceptive-Shadowing-G1-v0` and its `-Play-v0` form;
- `Instinct-Perceptive-Vae-G1-v0` and its `-Play-v0` form;
- `Instinct-Perceptive-HOI-Shadowing-G1-v0` and its `-Play-v0` form;
- `Instinct-BeyondMimic-Plane-G1-v0` and its `-Play-v0` form.

InstinctMJ additionally registers the train/play pair
`Instinct-Perceptive-Shadowing-G1-OneMotion[-Play]-v0`.  The unified registry must include it;
there is no Isaac/main registration to preserve for that pair.

## Effective whole-body baseline

Constructing InstinctMJ's registered train factory (with a harmless import stub for the optional
`coacd` terrain dependency) gives the following effective configuration:

- 2048 environments, spacing 4.0, decimation 4, 10 s episodes, physics step 0.005 s;
- MuJoCo solver iterations 10, line-search iterations 20, `njmax=1200`, unbounded `nconmax`;
- five reference commands in this order: world/base position, reference-frame position, rotation,
  joint position, joint velocity;
- policy order: joint position reference, joint velocity reference, position reference, rotation
  reference, projected gravity, base angular velocity, joint position, joint velocity, last action;
- critic adds link position/rotation and base linear velocity before the common proprioception;
- nine rewards, seven events, one adaptive-sampling curriculum, six terminations;
- effective sensors: undesired contact, self collision and motion reference.

Main's registered final G1 train class retains 4096 environments and the Isaac robot/config
overrides.  Its base classes alone are therefore not a valid golden.  Play subclasses reduce to one
environment, add a reference robot/debug visualization, disable pushes/bin smoothing/adaptive
sampling, and modify motion start/bin settings.

## Reference differences that must remain explicit

These are genuine reference differences, not presumed adapter equivalences:

| Area | Isaac/main | MJLab/InstinctMJ |
|---|---|---|
| Registration | Gym class entry points | lazy task factories; also OneMotion IDs |
| Train population | whole-body final class uses 4096 envs | whole-body factory uses 2048 envs |
| Robot asset | USD/URDF via Isaac articulation | G1 MJCF entity |
| Joint native order | PhysX breadth-first | MJCF/actuator natural order |
| Actuation | Isaac implicit/delayed actuator classes as selected by the final G1 override | `BuiltinPdActuatorCfg`, MJ gains/armature/effort and delay groups |
| Contact | PhysX body contact sensor; main base sensor declares force threshold 10 N | body/geom match sensors; threshold-contact sensor uses 1 N where configured |
| Dynamics observables | Isaac applied torque, body COM velocity and PhysX joint acceleration | MuJoCo `qfrc_actuator`, link-frame velocity and MJ acceleration |
| Motion asset path | main's local `~/Datasets/...` selections | effective whole-body path is `~/Xyk/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single` |
| Whole-body motion velocity | final main buffer uses `frontbackward` | final MJ buffer uses `frontbackward` |
| Scene/rendering | USD lights/materials and tiled ray sensors | MJCF textures/materials and MJLab ray sensors |
| Object interaction | USD rigid objects and PhysX contacts | MJCF entities/geoms and MuJoCo contacts |

The shared declaration may select different engine profiles for these rows.  It must not claim
value-level portability for contact impulses, applied torque/`qfrc_actuator`, joint acceleration,
or COM-versus-link quantities.  Each engine implementation must instead be tested against its own
reference.

## Rewrite boundary

The legacy `tasks/shadowing/**/{*_env_cfg.py,mdp}` copies, Gym registration modules, task-local
play/CLI scripts and agent duplication are rewrite inputs, not stable APIs.  The supported public
surface after the rewrite is the engine-neutral task registry plus generic train/play/export
frontends.  Stable task IDs and checkpoint observation/action contracts are retained; legacy Python
module paths are not retained with forwarding wrappers.

The following evidence is deliberately deferred to later phases: exact canonical/native joint
permutations, motion frame inventory and fixed-seed values, sensor tensor axes, compiled MDP term
parameters/order, live reset/rollout behavior, and checkpoint/resume/export behavior.  Those claims
require the shared declaration and both compiled engines to exist first.
