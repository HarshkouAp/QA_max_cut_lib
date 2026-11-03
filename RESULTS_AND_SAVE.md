# Results and save

`result`, direct attributes and `last_run` describe one independent run.
A later `solve()` creates a new record; it does not mutate previously
retained snapshots. `save` is opt-in and uses no-overwrite semantics.

`save_original_data` sidecars are separate from `atts`; arrays retain
their dtype/shape and integer graph labels remain integer. Checkpoints
contain resumable state and are not a substitute for a scientific
result record.
