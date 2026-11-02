# Laboratory: каталог и warm-start QAOA

Дата: 2 ноября 2026 года. Версия: `0.1.0a11`.

## 1. Что перенесено

В этой версии реализован **один** самостоятельный исследовательский вариант:

```text
variant_id: warm_start_product_qaoa_cpu_v1
family:     warm_start_qaoa
type:       laboratory.warm_start_qaoa_cpu
class:      QA_max_cut_lib.laboratory.WarmStartQAOACPU
```

Это CPU-реализация полного statevector. GPU-вариант, warm-start RQAOA,
LS/иерархия, поля/окружение и MI-стратегии не объявляются готовыми.

## 2. Инвентаризация архива

Инвентаризация читает байты и AST, не импортирует и не исполняет старые
модули.

```text
archive sha256: b4c684a5d4da469adc25bd99a405e402892d1a2524465e4c97ab6229c71a4aed
Python members: 89
parser:         Python 3.12.14
Diplom 28, ENV 3, Generators 3, Laboratory 22,
Other 2, RQAOA_exp 2, Solvers 18, Tools 11
```

`Diplom/Mk_5.py` и `Laboratory/LS_RQAOA_fast_5.py` имеют одинаковый
`ast.dump(..., include_attributes=False)`, но разные raw SHA-256. Это
**provisional AST equivalence**, не доказательство одинаковой полной
stochastic trajectory. Исходные пути и оба hash сохранены.

File records и variant cards находятся в:

```text
src/QA_max_cut_lib/laboratory/data/inventory.json
src/QA_max_cut_lib/laboratory/data/catalog.json
src/QA_max_cut_lib/laboratory/data/inventory-v1.schema.json
src/QA_max_cut_lib/laboratory/data/catalog-v1.schema.json
```

`source_to_variant` покрывает все 89 записей. Неразобранные семейства
остаются в `deferred_variant_ids` с причиной и backlog; они не запускают
базовый `QAOACPU` под чужим именем.

## 3. Математика warm-start

Для дискретного seed:

.. math::

   q_i = \frac{1-s_i^{(0)}}2,\qquad
   \widetilde q_i=\operatorname{clip}(q_i,\varepsilon,1-\varepsilon).

Для `provided_probabilities` вход уже является вероятностью bit=1 и
подвергается только clip; аффинное преобразование не применяется.
Исходные `q` вне `[0,1]`, NaN/Inf отвергаются до clip.

.. math::

   \alpha_i=2\arcsin\sqrt{\widetilde q_i},\qquad
   |\phi_0\rangle=\bigotimes_i
   \left(\sqrt{1-\widetilde q_i}|0\rangle+
   \sqrt{\widetilde q_i}|1\rangle\right).

`alpha` — фиксированные preparation angles, не trainable gamma/beta.

Два mixer:

.. math::

   U_i^{aligned}(\beta)=R_Y(\alpha_i)R_Z(-2\beta)R_Y(-\alpha_i)
   =\cos\beta I+i\sin\beta(\cos\alpha_i Z+\sin\alpha_i X),

.. math::

   U_i^{modified}(\beta)=R_Y(-\alpha_i)R_Z(-2\beta)R_Y(\alpha_i)
   =\cos\beta I+i\sin\beta(\cos\alpha_i Z-\sin\alpha_i X).

Mixer `x` — обычный `exp(-i beta X)`. При `epsilon=1/2` modified совпадает
с обычным QAOACPU при том же beta, aligned — при `-beta`.

## 4. Источники

| Source | Смысл | Стоимость |
|---|---|---|
| `provided_spins` | Полное назначение внешних меток, значения строго ±1 | external/unaccounted |
| `provided_probabilities` | Полные q=P(bit=1), конечные, `[0,1]` | external/unaccounted |
| `classical_solver` | Явный CPU baseline `gw_cpu`, `simulated_annealing_cpu` или `exact_cpu` | `source_cost` |

Для `provided_*` entries проверяются на полное покрытие текущего графа:
неизвестная, пропущенная или повторяющаяся вершина — ошибка. Изоляты
покрываются так же. Источник не запускается в конструкторе или import;
каждое новое `solve()` получает данные для своего графа. Циклический
источник (`laboratory.*` внутри `classical_solver`) отклоняется.

Classical child возвращает полное назначение, проверенное на исходном
графе. SDP-матрица GW сама по себе не превращается в q.

## 5. Запуск и бюджет

Целевой конфиг:

```bash
python -m QA_max_cut_lib.solvers.warm_start_qaoa_cpu \
  --graph configs/graphs/warm_start_4.json \
  --config configs/solvers/warm_start_qaoa_cpu_provided.json
```

При p=1, `max_steps=3`, `M_train=32`, `M_readout=64`:

```text
training = 3 * 5 * 32 = 480
readout  = 64
total    = 544 shots
```

Pause после одной итерации: `known_spent=160`, `readout_reserved=64`,
`available=320`. После resume завершённый run имеет 544, а не второй
независимый лимит. `source_cost` classical preprocessing не маскируется
как quantum shots.

## 6. Результат

`RunRecord` сохраняет общие поля и добавляет:

- `variant_id`, `warm_start_data`, `mixer_id`;
- `source_type`, `source_cut`, `source_run`, `source_cost`;
- `initial_expected_cut`, `initial_expected_energy`;
- обычные `solution`, `cut_value`, `optimal_angles`, `expected_cut`,
  `resources`, `checkpoint_info`.

`source_cut` (дискретный seed) и `initial_expected_cut` (product state)
разные числа. `initial_expected_cut` не считается конечным cut.

## 7. Checkpoint

`laboratory.warm_start_qaoa_cpu` поддерживает те же границы, что
`QAOACPU`: завершённая ADAM-итерация, `training_done`, `readout_done`.
В checkpoint замораживаются source result/q/epsilon/mixer, node_order,
theta/m/v, RNG, budget и readout token. `restore` создаёт объект через
фабрику; `resume` продолжает canonical run. `solve()` всегда новый run.

Strict resume отклоняет изменённые `epsilon`, source или mixer как новую
научную задачу. Classical source после restore не решается повторно.

## 8. Каталог CLI

```bash
python -m QA_max_cut_lib.laboratory list
python -m QA_max_cut_lib.laboratory show --id warm_start_product_qaoa_cpu_v1
python -m QA_max_cut_lib.laboratory list --implemented-only
```

`list/show` не импортирует CuPy/CVXPY и не запускает solver.

## 9. Малая серия H

`experiments/benchmark_warm_start.json`:

```text
3 graph instances × 2 algorithm seeds × 7 configurations = 42 runs
7 configs = base + {epsilon 0.25,0.50} × {x, warm_aligned, rounded_modified}
quantum shots = 42 × 544 = 22848
```

Это контролируемая серия с равным бюджетом. Она не доказывает, что
warm-start всегда лучше base/GW/SA. Exact reference считается отдельно.

## 10. Ограничения

- Только CPU full statevector; 2^N память и время.
- GPU warm-start не реализован; chi-симметрия неприменима.
- Нет analytic p=1, RQAOA, LS, hierarchy, fields/environment.
- Нет multiple warm starts/агрегации для рекурсии.
- Нет автоматического выбора источника по всем GW rounding.
- Нет обещания улучшения качества; endpoint-инварианты не являются
  доказательством качества конечного ADAM.
