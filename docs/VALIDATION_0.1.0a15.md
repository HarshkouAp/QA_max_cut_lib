# Проверка 0.1.0a15

Дата: 2 ноября 2026 года.

## База

- `QA_max_cut_lib 0.1.0a15`, Python 3.12.14, NumPy 2.5.4.
- CPU: `189 passed, 1 skipped, 1 warning`.
- Sphinx `-W`: успешно.
- Wheel установлен в `.venv` и `.venv-wheelcheck`; примеры/CLI запускаются
  вне дерева `src`.
- Notebook 16 выполнен установленным ядром.

## C — fresh final correlation

Два узла 10,30, ребро веса 2; один block; `max_sweeps=0`, exact режимы,
θ=(-π/4, π/8). Fresh extraction даёт `c=-1.0`, reducer выбирает
`sigma=-1`; после lift полный cut равен 2. Damped/model q не использован
для выбора.

## Основной six-node run

Greedy partition max_size=3, один episode на parent step, `finish_at=4`,
`max_reductions=2`:

```text
reductions=2
solution_source=exact_finisher
status=completed
final full cut пересчитан на исходном graph
```

Fresh tables содержат episode/block/pairs/mode/shots и реальные c.

## Transport / reducer

Для fixture A после удаления 80 и сохранения 30 с `sigma=-1`:

```text
new edge (10,30) = -1
b = 8
carried mean retained labels = (0.2,-0.4,0.6)
product q(10,30) = -0.08
```

`needs_repartition=True`; старые block q/θ не переносятся.

## Checkpoint

Outer codec сохраняет reduction state, parent means, budget, completed
episodes/history, `logical_updates` и phase. `restore()` + `resume()`
проверены на C и six-node run. Mid-episode nested codec, fault injection
и replay receipt в этой версии **не заявлены**.

## Not run

- 16-trial series, реальный parallel episode execution, P=1/2/4 profile;
- multi-round / all_pairs production cases;
- strict require_full/anchor production sizes;
- crash injection I и cross-process restore;
- GPU/chi/analytic/warm-start/overlap/nonlinear fields.
