# Git stage 15

Commit `c048016` contains Stage 15: `DistributedLSRQAOACPU`, independent
mean-field episodes, fresh final ZZ extraction, equal-episode mean
aggregation, carry/cold transport, complete anchor repair, outer reducer
commit and a compact outer checkpoint at `initialized`/`reduction_committed`.

Limitations: episodes execute sequentially; mid-episode nested resume, fault
injection, replay receipts, the 16-trial series and GPU modes are not claimed.
