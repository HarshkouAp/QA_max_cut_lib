# Проверка 0.1.0a10

Дата: 2 ноября 2026 года.

## Фактическая база

- Python 3.12.14, Windows 10; CPU 4 физических / 8 логических ядер;
- NumPy 2.5.4, SciPy 1.18.1, CVXPY 1.9.3, SCS 3.3.1,
  `dwave-samplers 1.8.0`, `dimod 0.12.23`;
- GPU-профиль: NVIDIA GeForce RTX 3050 Laptop, CC 8.6, 4096 MiB,
  CuPy 14.2.0, драйвер 591.74;
- wheel `0.1.0a10` установлен в `.venv` и `.venv-wheelcheck` вне `src`.
  `examples/checkpoint_resume.py` и CLI `tools.checkpoints` запущены из
  установленного wheel.

## Обязательные проверки

```text
CPU: 151 passed, 1 skipped, 1 warning
GPU: 10 passed
Sphinx -W: успешно
wheel: QA_max_cut_lib-0.1.0a10-py3-none-any.whl
sdist: QA_max_cut_lib-0.1.0a10.tar.gz
```

## A. QAOA/ADAM pause и resume

Контрольный p=1, `max_steps=4`, `M_train=32`, `M_readout=64`, `B=704`.

| Точка | iterations | known training | held readout | available |
|---|---:|---:|---:|---:|
| pause после 2 | 2 | 320 | 64 | 320 |
| resumed completed | 4 | 640 | 0 | 0 |

После паузы `status="paused"`, `readout_completed=false`; `inspect_checkpoint`
показывает `phase="training"` и `next_operation={"optimizer_iteration": 3}`.
`restore(path).resume()` в новом объекте дал:
- тот же canonical `run_id`;
- одинаковые `optimal_angles`, полное назначение, optimizer history и
  `objective_evaluations=20`;
- финальный `shots_used=704`.

Непрерывный a10 и resumed a10 совпали для p=1/p=2, exact и shots, при
cvar/mean и при early gradient stop. `best_evaluation_id` сохраняется отдельно.
Проверка выполнена на реальном `QAOACPU`, не на JSON round-trip.

## B. Durable journal и потерянная попытка

После pause (known=320, held=64) в журнал добавлен BEGIN для третьей
итерации с upper=160 без COMMIT.

```text
known_spent=544
uncertain_upper_charge=160
total_charged=704
readout_shots=64
optimizer_steps=3
status=stopped
```

Четвёртая итерация не помещается. Повторное reconciliation после RESOLVE не
удваивает 160. Число 704 — консервативный charge, не доказанное физическое
число shots неизвестной попытки. Readout token резервируется один раз; повтор
completed generation не делает новые измерения.

## C. Readout и max_steps

- `max_steps=0`: ноль optimizer steps, initial angles сохраняются.
- `gradient_tolerance`: phase `training_done`, update не выполняется повторно.
- checkpoint после `readout_done`: `resume()` не запускает solver и не
  добавляет generation.
- новый `solve()` на восстановленном объекте имеет новый `run_id`.

## D–G. Формат, блокировка, повреждение

- `json_npy_v1`: manifest + state.json + `theta/moment/variance/best_theta`
  `.npy`; SHA-256, dtype/shape/size проверяются до использования.
- `latest.json` обновляется после immutable generation; старые generation
  остаются.
- Подмена/порча нового payload даёт `CheckpointCorrupt`; старая generation
  читается.
- Второй writer одного managed run получает `CheckpointLocked`; после release
  lock можно взять снова.
- Копия generation tree в другой каталог восстанавливается; путь
  `checkpoint.path` пересчитывается от фактического дерева.

## Capability matrix

| Компонент | a10 |
|---|---|
| `QAOACPU` | полный codec: ADAM iteration, training_done, readout_done |
| `QAOAGPU` | отклоняется; нужен переносимый CuPy sampler protocol |
| `RQAOACPU`, `RQAOAGPU` | отклоняется; source/reduction codec не реализован |
| `LSRQAOACPU`, `LSRQAOAGPU` | отклоняется; child task codec не реализован |
| `QAOAInQAOACPU`, `QAOAInQAOAGPU` | отклоняется; level/local-pattern codec не реализован |
| `ExactCPU`, `GW`, `SA`, `Refiner`, `Refined` | отклоняются явно |
| `ExperimentRunner` | completed reuse работает; paused child codec не заявлен |

## Stage 09 повторно проверен на a10

```text
small series: 56 planned / 56 valid / 0 failed; 2 exact refs
profile: 84 points, 84 warmups, 252 measured, 336 calls
concurrent batch: 4 valid, 0 failed
mixed CPU/GPU: 3 valid, qaoa_gpu completed, gpu_synchronized=true
```

## Известные ограничения

- RQAOA/LS/hierarchy/Exact/GW/SA/refiner resume не реализованы; включение
  checkpoint для них отклоняется.
- GPU checkpoint не заявлен; CuPy random state не сериализуется как NumPy.
- Нет distributed ownership и exactly-once физических измерений.
- Нет incremental/dedup generations; retention только `keep_all`.
- Campaign paused child codec не реализован; завершённые trials
  переиспользуются.
- Checkpoint I/O и recovery rebuild не входят в active compute time.
