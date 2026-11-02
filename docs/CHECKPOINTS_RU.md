# Контрольные точки и продолжение (0.1.0a10)

## Режимы

| Вход | Семантика |
|---|---|
| `solver.solve(graph)` | новый независимый запуск; новый `run_id`, новый prefix |
| `solver.solve(graph, control=RunControl(...))` | cooperative pause на поддержанной границе |
| `restore(path).resume()` | продолжение того же canonical run |
| `restore(path).solve(graph)` | новый запуск; restore сам по себе не продолжает |
| crash recovery | последняя published generation + durable journal; без обещания отсутствия повторов |

## Целевой пример

```python
from QA_max_cut_lib import RunControl, inspect_checkpoint, restore, tools

config = tools.load_json("configs/solvers/qaoa_cpu_checkpoint.json")
solver = tools.SolverFactory.from_config(config)
spins = solver.solve(graph, control=RunControl(pause_after={"optimizer_iterations": 2}))
assert solver.result["status"] == "paused"
path = solver.last_checkpoint.path
summary = inspect_checkpoint(path)
continued = restore(path)
final_spins = continued.resume()
```

`inspect_checkpoint` читает manifest, schema, generation, phase, next operation,
array headers/checksums и journal cutoff, не создавая solver и не поднимая
CUDA-контекст.

## Capability matrix

В a11 реализованы переносимые codecs для `QAOACPU` и `laboratory.warm_start_qaoa_cpu`:

| Компонент | a10 boundary | Статус |
|---|---|---|
| `QAOACPU` / `qaoa_cpu` | completed ADAM iteration, `training_done`, `readout_done` | реализован |
| `WarmStartQAOACPU` / `laboratory.warm_start_qaoa_cpu` | те же границы; frozen source/q/mixer | реализован |
| `QAOAGPU` / `qaoa_gpu` | требуется переносимый CuPy sampler protocol | отклоняется |
| `RQAOACPU`, `RQAOAGPU` | source/reduction codec | отклоняется |
| `LSRQAOACPU`, `LSRQAOAGPU` | parent/child task codec | отклоняется |
| `QAOAInQAOACPU`, `QAOAInQAOAGPU` | atomic level/local-pattern codec | отклоняется |
| `ExactCPU` | batch enumeration codec | отклоняется |
| `GoemansWilliamsonCPU` | SDP opaque call / hyperplane state | отклоняется |
| `SimulatedAnnealingCPU` | read-boundary state | отклоняется |
| `ExactBlockRefiner`, `RefinedSolver` | block/base child state | отклоняется |
| `ExperimentRunner` | campaign child links | частично: completed reuse; paused child codec не заявлен |

Отказ явный: включение `checkpoint.enabled` для компонента без codec даёт
`ValueError`/`CheckpointIncompatible`, а не тихую частичную сериализацию.
Новый неизвестный плагин без codec также отклоняется.

## RunState QAOACPU и warm-start

Сохраняются:

- `run_id`, generation/parent, component id, package version;
- graph snapshot, node order, content/instance fingerprints;
- resolved config, limits, checkpoint policy;
- `theta`, `m`, `v`, `best_theta`, `best_loss`, evaluation IDs;
- `evaluations`, `optimizer_steps`, `training_shots`;
- optimizer histories и objective history в пределах collect;
- полный `numpy.random.Generator` state (PCG64 JSON state);
- `budget_total`, `budget_used`, readout reserved/result;
- phase, `next_iteration`, control consumed flags;
- durable journal cutoff.

Не сохраняются live Python objects, statevector при восстановимости из angles,
CUDA pointers, CuPy RNG objects, pools и locks.

## Границы и точность

Поддержана граница полной ADAM-итерации: все `4p+1` central-difference
оценок завершены, затем решение о convergence, затем атомарное обновление
`m/v/theta`. В checkpoint не попадает смесь нового `m` и старой `theta`.
`max_steps` — предел всего запуска: после двух итераций из четырёх остаются
две, а не ещё четыре.

`best_theta` может быть точкой gradient probe, а не последней центральной
точкой; это сохраняется отдельно. При `gradient_tolerance` update не
выполняется, phase становится `training_done`, а не «повторить итерацию».

Проверка A:

| Граница | iterations | accepted training shots | held readout | available |
|---|---:|---:|---:|---:|
| до обучения | 0 | 0 | 64 | 640 |
| pause после 2 | 2 | 320 | 64 | 320 |
| training_done | 4 | 640 | 64 | 0 |
| completed | 4 | 640 | 0 | 0 (spent 704) |

Final `best_of_shots` резерв M=64 существует один раз. На resume token
восстанавливается; повторное чтение completed generation не делает новые
измерения. Если readout начался и результат потерян, повтор требует нового
ресурса; он не маскируется как successful readout.

## Формат и публикация

Формат `json_npy_v1`: строгий `manifest.json` + `state.json` + числовые
`.npy` sidecars (`allow_pickle=False`). Публикация:

1. staging directory рядом с generation на той же filesystem;
2. payload файлы, flush/fsync;
3. manifest с SHA-256, размерами, dtype/shape;
4. atomic rename staging → `gen_000NNN`;
5. atomic `latest.json`, затем `latest-index` принадлежит manager.

Старые generations не изменяются. Ошибка записи staging/manifest оставляет
предыдущую generation. `latest.json` не является единственным источником:
loader проверяет manifest и checksums. `os.replace` не объявляется
универсальной транзакцией на любой network filesystem; supported local
durability — `local_fsync` на проверенной локальной ФС.

## Журнал ресурсов

`ResourceJournal` хранит append-only BEGIN/COMMIT/RESOLVE с fsync. BEGIN без
COMMIT/RESOLVE при recovery списывается консервативно по upper bound и
закрывается записью RESOLVE; повторное reconciliation не списывает ту же
операцию второй раз. Одновременно parent/child reservations не складываются:
ledger имеет одного владельца.

Пример B: pause после 2 итераций (320 known, 64 held), затем crash с BEGIN
160 без результата. Restore даёт known=320, uncertain=160, held=64,
available=160. После повторного выполнения третьей итерации и readout:
known_spent=544, uncertain=160, total=704; четвёртая итерация не помещается.
`704` — консервативно зачтенный расход, а не доказанное число физических
измерений неизвестной попытки.

## Ограничения a10

- RQAOA/LS/hierarchy/Exact/GW/SA/refiner checkpoint не реализованы;
- GPU checkpoint отсутствует: CuPy random state нельзя сериализовать как
  NumPy state; нужен отдельный operation-addressed sampler;
- нет distributed ownership и exactly-once физических измерений;
- нет incremental/dedup generations и retention, кроме `keep_all`;
- campaign хранит completed reuse, но paused child codec не заявлен;
- checkpoint I/O time и recovery rebuild учитываются отдельно от active
  compute; wall-time budget с checkpoint overhead проверяется отдельно.
