from pathlib import Path

p=Path('README.md'); s=p.read_text(encoding='utf-8')
if '[WARM_START_RQAOA_RU.md]' not in s:
    s=s.replace('* [LABORATORY_RU.md](docs/LABORATORY_RU.md)', '* [LABORATORY_RU.md](docs/LABORATORY_RU.md)\n* [WARM_START_RQAOA_RU.md](docs/WARM_START_RQAOA_RU.md)')
p.write_text(s,encoding='utf-8')

p=Path('docs/API.md'); s=p.read_text(encoding='utf-8').replace('# API версии 0.1.0a11','# API версии 0.1.0a12')
if '## Warm-start ensemble RQAOA' not in s:
    s += '''\n\n## Warm-start ensemble RQAOA\n\n`WarmStartRQAOACPU` (`type="laboratory.warm_start_rqaoa_cpu"`) использует\n`correlation_source.type="warm_start_ensemble_cpu"`. Source принимает\n`candidate_source` `gw_rounds_cpu` или `provided_by_graph`, дискретные\nстарты, `warm_start`, objective/estimation/optimizer/initialization и\n`correlations` с `aggregation="uniform_components"`.\n\n```python\nfrom QA_max_cut_lib.laboratory import WarmStartRQAOACPU\nsolver = WarmStartRQAOACPU(config)\nsolution = solver.solve(graph)\nprint(solver.warm_start_history)\n```\n\nРезультат дополнительно содержит `warm_start_history`, `variant_id`,\n`aggregate_statevector=None` и статус `not_applicable_statistical_mixture`.\n`source_cost` classical preprocessing находится в `resources.source_cost`.\nCheckpoint/resume для этой сборки в a12 не заявлен.\n'''
p.write_text(s,encoding='utf-8')

p=Path('docs/index.rst'); s=p.read_text(encoding='utf-8')
s=s.replace('QA_max_cut_lib 0.1.0a11','QA_max_cut_lib 0.1.0a12').replace('Одиннадцатый этап','Двенадцатый этап').replace('warm-start CPU.', 'warm-start CPU и warm-start RQAOA.')
if 'WARM_START_RQAOA_RU.md' not in s:
    s=s.replace('* :download:`Проверка этапа 11 <VALIDATION_0.1.0a11.md>`', '* :download:`Проверка этапа 11 <VALIDATION_0.1.0a11.md>`\n* :download:`Warm-start RQAOA <WARM_START_RQAOA_RU.md>`')
p.write_text(s,encoding='utf-8')

p=Path('docs/MIGRATION_STATUS.md'); s=p.read_text(encoding='utf-8').replace('# Статус 0.1.0a11','# Статус 0.1.0a12')
s=s.replace('| Laboratory | 89 file records', '| Laboratory | 89 file records; `WarmStartRQAOACPU` ensemble assembly | 89 file records')
p.write_text(s,encoding='utf-8')

p=Path('docs/ARCHITECTURE_MAP_RU.md'); s=p.read_text(encoding='utf-8')
s=s.replace('| Laboratory | `laboratory.catalog`, `solvers.warm_start`, `solvers.warm_start_qaoa_cpu` |', '| Laboratory | `laboratory.catalog`, `solvers.warm_start`, `solvers.warm_start_qaoa_cpu`, `solvers.warm_start_ensemble`, `solvers.warm_start_rqaoa_cpu` |')
p.write_text(s,encoding='utf-8')
print('docs updated')
