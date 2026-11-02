# Проверка 0.1.0a13

Дата: 2 ноября 2026 года.

## База

- `QA_max_cut_lib 0.1.0a13`, Python 3.12.14, NumPy 2.5.4.
- CPU: `173 passed, 1 skipped, 1 warning`.
- Sphinx `-W`: успешно.
- Wheel/sdist проверены установкой вне дерева `src`.
- GPU: старая матрица не перезапускалась; GPU-поля в этом этапе **не заявлены**.

## A — поля и константы

Граф: узлы `(-7,10,30,80,101)`, рёбра
`(10,30)=2`, `(10,80)=3`, `(30,80)=-4`, `(30,101)=5`, `(80,101)=-2`,
`S=(10,30)`, `r_80=+1`, `r_101=-1`, `r_-7=-1`.

```text
J(10,30)=1
h=(1.5,-4.5)
c_aff=-3
C_TT=-2
A_S=[5,10,-2,-1]
C_full=[3,8,-4,-3]
```

Таблица проверена независимым полным перебором cuts на исходном графе.

## B — shifted threshold

Для `b=7` original scores `[10,15,3,4]`; при равномерном активном
распределении и threshold `A=12` вероятность равна `0.25`.
Проверка с `cut_offset=7` вернула `0.25`; забывание `C_TT` дало бы `0.5`.

## C — один кубит

Активен `10`, внешний `r_80=+1`, ребро веса 2. При
`gamma=-pi/4`, `beta=pi/4` активный спин равен `-1`, полный cut равен 2;
фиксация активного спина ради zero-field symmetry дала бы неверный cut 0.

## D — full-graph limit

При `S=V` поля нулевые, `C_TT=0`; вероятности совпали с обычным
`QAOACPU` при тех же углах с ошибкой `0.0` на проверенном графе.
Отдельно реализован pinned embedding в тестовой математике.

## F — реальная редукция

До редукции `S={10}`: `h=2.5`, `c_aff=-2.5`.
После удаления 80, сохранения 30 при `sigma=-1`: `b=8`, новое ребро
`(10,30)=-1`, reference `r_30=+1`; `h=-0.5`, `c_aff=0.5`, original
scores `[8,7]`. `retain_survivor` удаляет только запись 80; lift задаёт
`r_80=-r_30`.

## Standalone и LS evidence

- `examples/conditional_qaoa.py`: 384 shots, полный cut 8, affected 10.
- `examples/environment_ls.py`: 1 reduction, 96 shots, `exact_finisher`,
  полный cut проверен на исходном графе.
- `tests/test_conditional.py`: 7 тестов, включая полный graph limit,
  threshold, singleton и `retain_survivor`.

## Не проверено

- checkpoint/resume новой LS recipe: не реализовано;
- GPU field backend, soft/mean-field exchange, correlation-weighted fields:
  `not_run`/deferred;
- большие parent graphs, несколько local tasks и crash fault injection:
  не заявляются пройденными;
- RAR-контейнер в рабочем дереве отсутствует; provenance — по
  извлечённому архиву этапа 11.
