# Git stage 13

Коммит `e1958a2` содержит Stage 13: общий `ConditionalCutProblem`, полный CPU
statevector для активного блока, `ConditionalQAOACPU`,
`EnvironmentLSRQAOACPU`, Laboratory-маршруты, source cards, конфигурации,
примеры, тесты, документацию и ноутбук 14.

Ограничения: checkpoint/resume новой LS recipe не реализован; GPU-поля,
soft/mean-field exchange и correlation-weighted fields не заявлены.
Старые LS/QAOA-контракты сохранены; общий `LSRQAOACPU` получил только
neutral hooks `_fallback_solution` и `_on_reduction`.
