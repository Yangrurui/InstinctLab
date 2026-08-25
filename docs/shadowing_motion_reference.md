# Shadowing motion-reference contract

Shadowing uses the common motion-reference runtime for both engines. The declaration records a
dataset root per reference engine because the checked-out projects do not point at the same local
data. Directory inventories include only `retargetted.npz` and `retargeted.npz`; terrain datasets
take their declared order, weights and terrain ids from `metadata.yaml`; BeyondMimic selects
`sprint1_subject2_retargetted.npz`. OneMotion selects the first effective metadata entry. Recursive
inventories are sorted so rank assignment and seeded sampling do not depend on filesystem order.

All families load `wxyz` root quaternions, remap joint arrays by name to the shared DFS order,
interpolate onto the 50 Hz half-open timeline and compute joint, root and link velocities with the
references' `frontbackward` finite difference. Whole-body uses a 20 ms, 10-frame current-time
horizon. Perceptive, VAE and HOI use a 100 ms, 10-frame current-time horizon. BeyondMimic uses only
the current frame. Symmetric augmentation is disabled in every effective shadowing config.

The runtime stores clip id, start time, timestamp, last update, look-ahead history and a separate
floor-indexed reset state. Look-ahead indexing uses the references' rounded frame selection. Invalid
future samples freeze at the last frame and set validity false; `dataset_exhausted` consumes that
flag instead of silently restarting. Reset rebuilds the selected environments' endpoint and history
and does not reset the lifetime exhaustion counter.

Dataset roots intentionally differ by engine and are fingerprinted in the TaskSpec contract. In
particular, main's perceptive config still contains the literal placeholder
`{AbsolutePathOfYourDataDirectory}`, whereas InstinctMJ names a local dataset. The configured roots
are absent on this machine, so their real file counts, frame counts and fixed-seed values cannot be
certified here. Synthetic multi-clip tests cover the loader and state machine; live value parity
requires mounting the exact referenced datasets.
