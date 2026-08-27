# InstinctLab agent guide

Read [HANDOFF.md](HANDOFF.md) before changing code or starting a run. It is the
authoritative record for server migration, external repositories, datasets,
live experiments, and unfinished work. `README.md` is the user-facing setup
guide. Historical plans and audit narratives are intentionally not live docs;
use Git history when their provenance is needed.

## Architecture

- Every active task is a single engine-neutral `TaskSpec` registered in
  `source/instinctlab/instinctlab/tasks/registry.py`.
- Keep dependency directions explicit. Engines must not import or depend on one
  another. Task declarations must not import engine implementations, and engine
  implementations must not contain task-specific policy. Connect both sides
  through small, engine-neutral interfaces in the shared layer.
- Keep each task's concrete configuration together in one obvious task config
  file whenever practical. Avoid inheritance chains and repeated overrides that
  make the final value of a parameter difficult to find.
- Use `TaskSpec` only to describe the stable, engine-neutral task contract and
  component selection. Do not put concrete reward weights, training settings,
  physics values, or engine-specific parameters in `TaskSpec`; those belong in
  the task config or the relevant engine builder.
- `scripts/train.py` and `scripts/play.py` select `isaacsim` or `mjlab` before
  importing an engine SDK. Do not add engine branches to these shared entry
  points.
- Shared declarations, tensor math, and MDP terms live in `spec/`, `sim/`,
  `compat/`, and `mdp/`. Engine SDK imports belong inside builders under
  `engines/<name>/`; importing the shared layer must not require either SDK.
- `RobotSpec.joint_names` is the canonical DFS order. Always resolve joints by
  explicit names with `preserve_order=True`. Isaac's native articulation order
  is BFS; MJLab is naturally DFS. Never write a canonical vector positionally
  into a native articulation.
- AMP policy and reference terms share one builder and one ordered schema. The
  motion sensor exposes look-ahead `data` beginning at `t + dt`, while
  `reference_frame` is a separate current-time `t` sample used by AMP, matching
  main.
- Isaac behavior is checked against `/root/InstinctLab-main`; MJLab behavior is
  checked against `/root/InstinctMJ`. Preserve intentional solver, contact,
  actuator, and sensor differences. Cross-engine equality is required at the
  declared interface, not for native physics internals.

## Working rules

- Preserve unrelated user changes and inspect `git status` before editing.
- Write for algorithm engineers who may not have extensive software-engineering
  training. Prefer direct control flow, familiar Python, descriptive names, and
  small single-purpose functions over clever abstractions or advanced syntax.
- Fix problems at the responsible interface or implementation. Do not use monkey
  patches, scattered special cases, temporary branches, or task-specific checks
  in shared code to bypass an architectural problem. Put unavoidable
  compatibility behavior in `compat/` with its reason and scope made explicit.
- Do not stop, restart, or signal a training process unless the user explicitly
  asks. Before starting a run, inspect `pgrep -af scripts/train.py` and
  `nvidia-smi`; never overwrite an active run's log directory.
- Use `/root/miniconda3/envs/env_isaaclab/bin/python`. Set
  `PYTHONPATH=source/instinctlab` for direct pytest invocations when needed.
- Use `rg` for searches and `apply_patch` for hand edits. Do not use broad
  destructive Git commands in a dirty worktree.
- Commit one independently verified change at a time. Keep diagnostics and
  production behavior in separate commits when practical.
- A green construction smoke test is not parity evidence. Cross-engine bugs in
  this repository are commonly silent: training runs and rewards rise while a
  term, timer, ordering, or reference frame is wrong.

## Verification

Choose checks according to the change; do not claim more than they prove.

```bash
PYTHONPATH=source/instinctlab /root/miniconda3/envs/env_isaaclab/bin/python -m pytest -q tests
python scripts/check_mjlab.py
PYTHONPATH=source/instinctlab /root/miniconda3/envs/env_isaaclab/bin/python -m pytest -q tests/test_parkour_g1_declaration.py -k contract
```

For term or physics changes, add fixed-state and temporal probes. Compare
episode length and termination behavior as well as reward. Run production-scale
contact/constraint checks after terrain, collision, or solver changes. The
cross-engine workflow and known silent failures are in
`.cursor/skills/cross-engine-adaptation/`.

## Documentation policy

- Update `HANDOFF.md` when dependencies, data locations, task status, active
  baselines, or unresolved risks change.
- Keep `AGENTS.md` short and operational. Put reusable cross-engine procedures
  in the skill, not here.
- Do not add one document per audit phase. Fold durable conclusions into the
  handoff, code comments, tests, or the skill; Git history retains the narrative.
