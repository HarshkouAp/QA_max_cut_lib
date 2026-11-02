# Условный QAOA и LS-RQAOA с фиксированными внешними спинами

Версия `0.1.0a13` добавляет один явный тип локальной задачи: внешние спины
`r_T` **фиксированы**, а активный блок `S` обучается полным CPU statevector.
Это conditioning, а не маргинализация большого QAOA и не mean-field обмен.

## Локальная задача

Для текущего графа `G_t`, активного блока `S` и фиксированного `r_T`:

\[
A_S(x;r_T)=C_{SS}(x)+C_{ST}(x,r_T),
\]

\[
E_{\mathrm{aff}}(x;r_T)=-A_S
=c_{\mathrm{aff}}+\sum_{i\in S}h_i x_i+
\sum_{\{i,k\}\subset S}J_{ik}x_ix_k,
\quad J_{ik}=\frac{w_{ik}}2,\quad
h_i=\frac12\sum_{j\in T}w_{ij}r_j.
\]

Константа `c_aff`, `C_TT` и накопленный `b_t` хранятся отдельно. Для
`original_graph` активная строка оценивается как `b_t + C_TT + A_S`.

`h` не кодируется самопетлёй и не превращается в атрибут `WeightedGraph`.
Локальные поля не передаются в глобальный reducer: они возникают из
выбранного окружения. Глобальный граф остаётся zero-field Max-Cut графом.

## Публичные маршруты

```python
from QA_max_cut_lib.laboratory import ConditionalQAOACPU, EnvironmentLSRQAOACPU
```

- `laboratory.conditional_qaoa_cpu` — один блок, полное завершение
  фиксированными внешними спинами;
- `laboratory.environment_ls_rqaoa_cpu` — LS controller с frozen reference
  на шаг и `retain_survivor`.

## Что не входит

Итеративный обмен `〈Z〉`, soft means, correlation-weighted fields,
нелинейные F2-правила, GPU/chi/analytic-p=1 и warm-start остаются
отдельными cards/backlog. `update="refresh_all"` означает, что после
редукции локальные задачи строятся заново; старые корреляции не
маскируются под cache hit.

Checkpoint/resume для новой LS recipe в этой версии не заявлен: конфиг с
включённым `checkpoint` отклоняется. Standalone также не поддерживает
checkpoint. Это видимое ограничение, а не «тихое» продолжение.

## Ограничения памяти

Полный CPU statevector имеет размер `2^{|S|}`; по умолчанию
`max_active_vertices=18`, жёсткий предел — 24. Parent graph может быть
большим, но каждый quantum task ограничен `|S|`. GPU-путь для поля не
объявлен.
