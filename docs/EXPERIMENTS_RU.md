# Серии экспериментов и профилирование параллелизма

Версия **0.1.0a9** добавляет общий слой экспериментов. Он не переносит циклы
солверов: `ExperimentRunner` и `ParallelismProfiler` вызывают существующие
реализации через `SolverFactory` и общий `execute_trial`.

## Быстрый пример

```python
from QA_max_cut_lib import tools

spec = tools.load_json("experiments/benchmark_small.json")
plan = tools.plan_experiments(spec)
print(plan.summary)          # graphs=2 cases=28 trials=56 references=2

series = tools.ExperimentRunner(spec).run()
print(series.summary)
print(series.analysis["by_solver"])

profile_config = tools.load_json("tools/parallelism_sweep.json")
profile = tools.profile_parallelism(spec, config=profile_config)
print(profile.recommended_num_treads)
```

`plan_experiments` только разворачивает план: рецепты графов, solver-сетки,
seed, эталоны и ресурсные верхние оценки. Он не вызывает solver и без
включённого `save` не создаёт файлов.

## BenchmarkSpec

`BenchmarkSpec` проверяет закрытые поля верхнего уровня и строит план в
следующем порядке:

1. разворачивает `graph_cases` по явным осям `grid`;
2. независимо разворачивает `solver_cases` по их осям и overrides;
3. проверяет каждую пару «граф × разрешённый solver config» фактической
   `_validate` существующего класса;
4. строит декартово произведение graph instance × solver config × trial seed;
5. назначает стабильные `case_id` и `trial_id` через SHA-256 канонического JSON;
6. добавляет reference jobs по выбранной политике.

Неизвестное поле, повторяющаяся или пересекающаяся ось, отсутствующий путь
grid, дубликат graph instance и превышение `planner_limits` блокируют план.
Несовместимая отдельная комбинация попадает в `plan.rejected` и не запускается.

Оси — только явные `{"path": [...], "values": [...]}`. Непрерывный интервал
сам по себе не является осью: без конечного списка значений он не превращается
в «проверить все значения».

## Идентификаторы

- `graph_content_id` — канонические вершины, рёбра, точные веса, изоляты и
  версия схемы; порядок вставки не влияет.
- `graph_instance_id` — content плюс `node_order`; поэтому statevector нельзя
  молча переиспользовать на перестановке.
- `case_id` — graph instance и научная конфигурация без `seed`, `num_treads`,
  output/save и profiled CPU-бюджета.
- `trial_id` — case, algorithm seed, role, repetition, `num_treads`,
  measurement protocol и версия реализации.
- `attempt_id` — отдельная попытка выполнения; retry не скрывается.
- `reference_id` — граф, метод/настройки эталона, seed и смысл результата.

Порядок словарных ключей и порядок завершения процессов не меняют
семантические id. Python `hash()`, PID и номер строки CSV не используются.

## Режимы исполнения

`execution.mode`:

- `isolated` — один измеряемый solver trial активен в каждый момент;
- `concurrent_batch` — независимые trials выполняются пулом процессов
  (`backend="process"`, `spawn`) или потоками; число активных trials и
  `cpu_slots` ограничены явно. `ProcessPoolExecutor` требует обычного
  импортируемого script с guard `if __name__ == "__main__":`; в Jupyter
  используйте импортируемый CLI-подпроцесс.

`num_treads` означает CPU-бюджет одного солвера. Если реализация
последовательная, запись получает `requested_num_treads` больше
`effective_num_treads` и `capability_limited=true`; это не подменяет backend
на другой алгоритм и не выдаёт фиктивное ускорение.

## Эталоны

`ReferencePolicy` поддерживает:

| kind | смысл |
|---|---|
| `exact_max_cut` | доказанный полный максимум после `exhaustive_search` |
| `exact_min_cut` | отдельный запуск `ExactCPU` на графе с весами `-w` |
| `gw_rounded_cut` | эвристический допустимый GW-разрез |
| `best_known_cut` | внешний источник с `graph_content_id`, датой и provenance |
| `numerical_sdp_objective` | численное значение SDP без сертификации |

`exact_min_cut` получается как `C_min(G) = -C_max(-G)`. Для `q_range` нужны
оба доказанных экстремума; на постоянной цели возвращается `null` с причиной.
GW-значение не объявляется строгой SDP-границей, а `sdp_upper_bound` остаётся
`null` без dual certificate.

## Параллельный профиль

`ParallelismProfiler` берёт тот же BenchmarkSpec, один выбранный seed и все
допустимые case-ы. Ось `num_treads_values` добавляет profiler. Для каждого
case × candidate выполняются warmups и repetitions; warmup не используется
как timing sample. Порядок кандидатов внутри repetition может перемешиваться
отдельным RNG.

Результат содержит:

- `plan.result` с полным покрытием;
- все warmup/measured records с `requested/effective` ресурсами и work
  counters;
- `case_curves`, `aggregate`, `per_solver`;
- Amdahl fit и Karp–Flatt `f_eff` (без обрезания отрицательных/больших
  значений);
- paired-block bootstrap, если он явно запрошен;
- `recommended_num_treads` по полному набору case-ов, с `raw_argmin` и
  `search_complete` отдельно.

Неполный candidate исключается из глобальной рекомендации; его время не
заменяется нулём. `best_observed_complete` не называется оптимумом всех
запрошенных значений.

## Сохранение

`save` использует общую схему `{path, filename, atts, save_original_data,
makedir, savedir}`. Координатор — единственный писатель. Каждый trial/reference
публикуется отдельным JSON с `export_complete=true`; короткое имя проверяется
по полному id. Итоговый manifest не перезаписывается: при повторе создаётся
`*_summary_v2.json`. Без `save` папки и файлы не создаются.

`ExperimentRunner(..., reuse_completed=True)` переиспользует только завершённые
записи с совпавшими trial id, measurement protocol и environment id. Reuse
не является новым timing repeat.

## Ограничения a9

- checkpoint/resume внутри ADAM, редукций и иерархии не входит;
- concurrent profiling на одном GPU не входит;
- profile на этой сборке исполняет изолированный полный grid; поле
  `execution_mode` сохраняется, но конкурентный профиль требует следующего
  расширения;
- для строгого ограничения нативных BLAS/OpenMP нужен `threadpoolctl` и
  проверка фактической библиотеки; одни env vars не считаются доказательством;
- полный GPU-профиль ограничен доступной VRAM и проверяется отдельно.
