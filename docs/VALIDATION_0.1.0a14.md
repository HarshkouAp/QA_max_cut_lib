# Проверка 0.1.0a14

Дата: 2 ноября 2026 года.

## База

- `QA_max_cut_lib 0.1.0a14`, Python 3.12.14, NumPy 2.5.4.
- CPU: `184 passed, 1 skipped, 1 warning`.
- Sphinx `-W`: успешно.
- Wheel установлен в `.venv` и `.venv-wheelcheck`; примеры/CLI запускаются
  вне дерева `src`.
- GPU: не перезапускался; field/mean-field GPU modes не заявлены.

## A — model cut и округление

Два singleton-блока, ребро веса 2, means `(0.2,-0.4)`:

```text
model mean cut = 1.08
rounded solution = (+1,-1)
full cut = 2
```

Сумма двух local affected means равна `2.16`; это не global cut.

## B — damping

Старые `(m,q)=((1,1),1)`, candidate `z=(-1,1)`, `r=-1`, `lambda=1/2`
дают `m'=(0,1)`, `q'=0` и pair probabilities `(1/2,0,1/2,0)`. Сырой `q=-1`
с уже сглаженным `m` даёт отрицательную pair probability и отклоняется.

## D — applied angles

Один спин, `h=1`, `theta_old=(0,0)`, candidate `(-pi/4,pi/4)`,
`kappa=1/4`: applied `(-pi/16,pi/16)` даёт
`z=-(2-sqrt(2))/4 ≈ -0.1464466094`.

## Budget / status

- `B=47`, 3 обязательных moment jobs по 16: sweep не начинается,
  `moment_reserve_unavailable`, 0 shots.
- `B=48`: все три job выполняют только moments, `shots_used=48`,
  `training_shots=0`.
- target Jacobi 3 блока × 3 sweep × (`2×5×8+16`)=864 shots; реальный
  прогон использовал 864 и 9 logical jobs.
- `max_sweeps=0`: 0 local jobs, sign-rounded result и полный cut.
- empty graph: `{}` без quantum работы.

## Checkpoint

Реализован compact checkpoint на границе полного sweep:
`initialized` и `gauss_seidel_sweep_commit`/`jacobi_sweep_commit`.
`restore(path)` + `resume()` сохраняет m/q/theta, budget, RNG, histories
и не повторяет уже завершённые sweeps. Проверка выполнена в том же
процессе; отдельный cross-process test и fault injection в этой версии
**не заявлены**. Mid-sweep child checkpoint и replay receipt также не
заявлены.

## Not run

- overlapping blocks, adaptive damping, nonlinear/correlation-weighted
  fields, GPU/chi/analytic path;
- полный Jacobi multiprocessing pool с memory-limited resident jobs;
- GS/Jacobi completion permutations на production-размерах;
- 16-trial comparative series и profile P=1/2/4;
- fault-injection I до/после BEGIN/receipt/commit.
