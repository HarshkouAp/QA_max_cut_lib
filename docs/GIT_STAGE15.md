# Git stage 15

Коммит `c048016` содержит Stage 15: `DistributedLSRQAOACPU`, independent
mean-field episodes, fresh final ZZ extraction, equal-episode mean
aggregation, carry/cold transport, complete anchor repair, outer reducer
commit and compact outer checkpoint at `initialized`/`reduction_committed`.

Ограничения: episodes исполняются последовательно; mid-episode nested
resume, fault injection, replay receipts, 16-trial series и GPU modes не
заявлены.
