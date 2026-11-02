# Проверка сборки 0.1.0a3

**Версия:** `QA_max_cut_lib 0.1.0a3`, 2 ноября 2026 года.  
**Назначение:** базовый CPU RQAOA на точном ядре редукций этапа 02.

## Проверенное состояние

- Реализованы `RQAOACPU`, `QAOAStatevectorCorrelationSource`,
  `MaxAbsCorrelationSelector`, `ExactCPUFinisher`.
- Работают точный и шотовый режимы, общий `ShotBudget`, `edges`/`all_pairs`,
  `restart`/`previous_step`, точный остаток и `all_plus_reduced_fallback`.
- `reduction_history`, `parameter_trajectory`, `local_optimizer_history`,
  `correlation_tables` и `final_reduction_state` подключаются отдельными
  флагами.
- Добавлены графики весов, энергий, всех `binomial(2p,2)` траекторий и
  автономный 3D HTML.
- Полный набор проверок на локальной базе: **80 passed, 1 warning**.
- Wheel `0.1.0a3` собирался и устанавливался в чистое окружение; ноутбук 04
  исполнялся; CLI `rqaoa_cpu` выполнял реальный запуск.

## Известные ограничения

- Источник использует полный вектор состояния и ограничен его размером.
- Аналитический `p=1`, GPU, LS-RQAOA, QAOA-in-QAOA и Laboratory ещё не
  перенесены.
- `num_treads=1`; checkpoint/resume не реализован.
- RQAOA не гарантирует глобальный оптимум.

Финальная документация и экспорт архива этапа 03 не были завершены до перехода
к этапу 04; локальный код и тесты этой базы сохранены и использованы в
`0.1.0a4`.
