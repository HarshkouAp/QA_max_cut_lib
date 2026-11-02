"""Build and execute the Stage 04 analytic p=1 notebook."""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

root = Path(r"C:/Users/Alexey/PycharmProjects/QA_max_cut_lib")
path = root / "notebooks" / "05_analytic_p1_rqaoa.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md(
    """# Аналитические корреляции p=1 и RQAOA

**Версия 0.1.0a4.** Ноутбук проверяет аналитический вычислитель среднего
разреза и физических ZZ-корреляций для одного слоя QAOA. Вектор состояния
`2**N` не создаётся. Источник подключается к тому же `RQAOACPU`, который
использовался в ноутбуке 04; рекурсивный координатор и точный редуктор не
заменяются.

Поддерживаются взвешенный Max-Cut без полей, стандартное состояние `|+>`,
X-миксер, `p=1` и критерий Mean Cut. Полное распределение, шоты, CVaR, Gibbs,
Best-of-K и Threshold Probability этим источником не поддерживаются.
"""
)

code(
    """from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from copy import deepcopy
from itertools import combinations
import json
import math
import time

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

import QA_max_cut_lib as qa
from QA_max_cut_lib import (
    AnalyticP1Evaluator, AnalyticP1CorrelationSource, ExactCPU,
    GraphReducer, SolverFactory, WeightedGraph, tools,
)
from QA_max_cut_lib.core.budget import ShotBudget
from QA_max_cut_lib.solvers.qaoa_state import cut_diagonal, state_at_angles

root = Path.cwd()
if not (root / 'configs').is_dir():
    root = root.parent
assert (root / 'configs').is_dir(), 'Open the notebook from the project or notebooks folder.'
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:6]
output_dir = root / 'notebooks' / '_outputs' / ('analytic_p1_' + stamp)
output_dir.mkdir(parents=True, exist_ok=True)
print('QA_max_cut_lib', qa.__version__)
print('Output:', output_dir)
"""
)

md(
    """## 1. Формула и знаки

Для `E=-C` и `J=w/2` корреляция имеет вид

$$c_{uv}=\\frac12\\sin(\\gamma w_{uv})(P_{u\\setminus v}+P_{v\\setminus u})\\sin(4\\beta)
+\\frac12(P^-_{uv}-P^+_{uv})\\sin^2(2\\beta).$$

Первое слагаемое положительно при положительном весе и положительном
`sin(gamma*w)`. Для одного ребра при `gamma=pi/(2|w|)` и `beta=-pi/8`
получается `c=-sign(w)`. Для несмежной пары `w=0`, но общие соседи могут дать
ненулевое второе слагаемое.
"""
)

code(
    """graph = WeightedGraph(
    [(10, 20, 1.5), (20, 30, -0.5), (10, 30, 2.0),
     (30, 40, 1.0), (40, 50, 2.5), (10, 50, -1.0)],
    nodes=[10, 20, 30, 40, 50, 101],
)
reference = ExactCPU()
reference.solve(graph, report=0, logs=0)
figure = tools.plot_graph_info(graph, {'style': {'text_engine': 'mathtext', 'figsize': [10, 6]}})
display(figure)
plt.close(figure)
print('Exact cut:', reference.cut_value)
"""
)

md(
    """## 2. Независимая проверка при фиксированных углах

Сравним аналитические корреляции с полным вектором состояния того же пакета.
Это проверка формулы, а не проверка оптимизатора: углы заданы вручную.
"""
)

code(
    """evaluator = AnalyticP1Evaluator(graph, pair_batch_size=3)
gamma_test, beta_test = 0.37, -0.61
probabilities = state_at_angles(cut_diagonal(graph), [gamma_test, beta_test], 1)[1]
n = len(graph)
indices = np.arange(1 << n, dtype=np.uint64)
bits = ((indices[:, None] >> np.arange(n, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(float)
signs = 1.0 - 2.0 * bits
state_matrix = (signs * probabilities[:, None]).T @ signs
analytic_values = evaluator.correlations(gamma_test, beta_test, pair_domain='all_pairs')
error = max(
    abs(value - state_matrix[evaluator.node_order.index(u), evaluator.node_order.index(v)])
    for (u, v), value in analytic_values.items()
)
print('All pairs:', len(analytic_values))
print('Max correlation error:', error)
print('Analytic mean energy:', evaluator.expected_energy(gamma_test, beta_test))
print('Statevector mean energy:', float(-(cut_diagonal(graph) @ probabilities)))
assert error < 1e-12
assert math.isclose(
    evaluator.expected_energy(gamma_test, beta_test),
    float(-(cut_diagonal(graph) @ probabilities)),
    abs_tol=1e-12,
)
"""
)

md(
    """## 3. Одно ребро и несмежная пара

Одно ребро проверяет знак фазы и ограничения. Путь `u-k-v` проверяет, что
нулевой вес не означает нулевую корреляцию: при `gamma=pi/2`, `beta=pi/4`
общий сосед даёт `c_uv=1`.
"""
)

code(
    """for weight in (0.5, 1.0, -2.0):
    edge_graph = WeightedGraph([(0, 1, weight)], nodes=[0, 1])
    edge_eval = AnalyticP1Evaluator(edge_graph)
    value = edge_eval.correlations(math.pi / (2 * abs(weight)), -math.pi / 8)[(0, 1)]
    print('w=', weight, 'c=', value, 'E=', edge_eval.expected_energy(
        math.pi / (2 * abs(weight)), -math.pi / 8))
    assert math.isclose(value, -math.copysign(1.0, weight), abs_tol=1e-12)

path_graph = WeightedGraph([(0, 1, 1.0), (1, 2, 1.0)], nodes=[0, 1, 2])
path_eval = AnalyticP1Evaluator(path_graph)
path_c = path_eval.correlations(math.pi / 2, math.pi / 4, pair_domain='all_pairs')[(0, 2)]
print('Nonedge c(0,2):', path_c)
assert math.isclose(path_c, 1.0, abs_tol=1e-12)
"""
)

md(
    """## 4. Точная минимизация beta

`E(gamma,beta)=F0+Fs sin(4beta)+Fc cos(4beta)`, поэтому минимум равен
`F0-hypot(Fs,Fc)`, а `beta_*` вычисляется через `atan2`. Сравним с плотной
сеткой beta; сетка не должна дать меньшую энергию.
"""
)

code(
    """profile = evaluator.energy_min_beta(0.7)
beta_grid = np.linspace(-math.pi / 2, math.pi / 2, 801)
grid_energies = [evaluator.expected_energy(0.7, float(beta)) for beta in beta_grid]
print('beta*:', profile['beta_star'])
print('F0, Fs, Fc, R:', profile['F0'], profile['Fs'], profile['Fc'], profile['R'])
print('minimum:', profile['energy_min'])
print('dense-grid minimum:', min(grid_energies))
assert min(grid_energies) >= profile['energy_min'] - 1e-12
assert math.isclose(
    evaluator.expected_energy(0.7, profile['beta_star']),
    profile['energy_min'],
    abs_tol=1e-12,
)
"""
)

md(
    """## 5. Поиск gamma и источник корреляций

Источник перебирает явную сетку gamma, в каждой точке исключает beta и
возвращает корреляции только при выбранных углах. Шоты не расходуются.
"""
)

code(
    """source_config = tools.load_json(root / 'configs/solvers/rqaoa_analytic_p1_cpu.json')['correlation_source']
source_config['optimizer']['parameters']['points'] = 21
state = GraphReducer().start(graph)
source = AnalyticP1CorrelationSource(source_config)
source_result = source.estimate(state, budget=ShotBudget(0), pair_domain='all_pairs')
print('Angles:', source_result.angles)
print('Covered pairs:', len(source_result.values))
print('Profile points:', len(source_result.training['gamma_profile']['gamma_values']))
print('Resources:', source_result.resources)
assert source_result.resources['training_shots'] == 0
assert source_result.resources['correlation_shots'] == 0
assert len(source_result.values) == len(list(combinations(sorted(graph.nodes), 2)))
"""
)

md(
    """## 6. RQAOA с аналитическим источником

Меняется только конфигурация источника. Цикл `RQAOACPU`, точный редуктор,
точный остаток и восстановление остаются прежними. Общий лимит шотов равен
нулю.
"""
)

code(
    """config = tools.load_json(root / 'configs/solvers/rqaoa_analytic_p1_cpu.json')
config.pop('save', None)
config['parameters'] = {'finish_at': 3, 'max_reductions': None}
config['correlation_source']['optimizer']['parameters']['points'] = 21
config['output'] = {'report': 0, 'logs': 0, 'color': 'never'}
algorithm = SolverFactory.from_config(config)
solution = algorithm.solve(graph)
print('Solution:', solution)
print('Cut:', algorithm.cut_value, 'Energy:', algorithm.solution_energy)
print('Reductions:', algorithm.reductions_completed, 'Source:', algorithm.solution_source)
print('Shots:', algorithm.resources['shots_used'])
print('Resources:', algorithm.resources)
assert set(solution) == set(graph.nodes)
assert algorithm.resources['shots_used'] == 0
assert math.isclose(algorithm.cut_value, graph.cut_value(solution), abs_tol=1e-12)
assert math.isclose(algorithm.solution_energy, -algorithm.cut_value, abs_tol=1e-12)
assert math.isclose(algorithm.cut_value, algorithm.cut_offset + algorithm.reduced_cut_value)
"""
)

md(
    """## 7. Таблица редукций и профиль gamma

Строка профиля относится к графу перед соответствующим слиянием. `A`, `B` и
`E0` — коэффициенты аналитической энергии; `gamma_profile` не является
историей ADAM.
"""
)

code(
    """rows = []
for event in algorithm.reduction_history:
    rows.append({
        'step': event['step'],
        'removed': event['removed_node'],
        'retained': event['retained_node'],
        'sign': event['sign'],
        'weight': event['selected_weight'],
        'correlation': event['physical_correlation'],
        'fixed_energy': event['fixed_energy'],
        'expected_fixed_energy': event['expected_fixed_energy'],
        'cut_offset': event['cut_offset'],
        'ising_constant': event['ising_constant'],
    })
from pandas import DataFrame
display(DataFrame(rows))

profile_figure = tools.plot_analytic_gamma_profile(algorithm, {
    'selection': {'reduction_step': 1},
    'style': {'text_engine': 'mathtext', 'figsize': [10, 7]},
})
display(profile_figure)
plt.close(profile_figure)
"""
)

md(
    """## 8. Прежние графики и автономный HTML

Энергии, веса и траектории читаются из той же записи. Построение не запускает
сканер gamma и не расходует шоты.
"""
)

code(
    """weights_figure = tools.plot_reduction_weights(algorithm, {'style': {'text_engine': 'mathtext', 'figsize': [10, 6]}})
display(weights_figure)
plt.close(weights_figure)
energy_figure = tools.plot_reduction_energy(algorithm, {'style': {'text_engine': 'mathtext', 'figsize': [12, 8]}})
display(energy_figure)
plt.close(energy_figure)
trajectory_figure = tools.plot_parameter_trajectories(algorithm, {'style': {'text_engine': 'mathtext', 'figsize': [8, 6]}})
display(trajectory_figure)
plt.close(trajectory_figure)
trajectory_3d = tools.plot_parameter_trajectories_3d(algorithm, {
    'save': {
        'path': str(output_dir),
        'filename': 'analytic_trajectories',
        'format': 'html',
        'makedir': True,
        'savedir': True,
        'metadata': True,
    }
})
print('3D traces:', len(trajectory_3d.data))
print('HTML:', trajectory_3d.layout.meta['qa_primary'])
"""
)

md(
    """## 9. Сохранение и повторный анализ

Сохраняется корневой JSON с выбранными атрибутами. Профиль входит в
`local_optimizer_history`; после загрузки его можно строить без нового расчёта.
"""
)

code(
    """files = tools.save_result(algorithm, {
    'path': str(output_dir / 'saved'),
    'filename': 'analytic_rqaoa',
    'atts': ['solution', 'cut_value', 'cut_offset', 'ising_constant',
             'reduction_history', 'parameter_trajectory', 'local_optimizer_history'],
    'savedir': True,
    'makedir': True,
})
loaded = tools.load_result(files.primary)
print('Saved:', files.primary)
assert loaded.solution == algorithm.solution
assert loaded.local_optimizer_history[0]['profile_type'] == 'analytic_gamma_grid'
loaded_figure = tools.plot_analytic_gamma_profile(loaded, {
    'style': {'text_engine': 'mathtext', 'figsize': [10, 7]},
})
display(loaded_figure)
plt.close(loaded_figure)
"""
)

md(
    """## 10. Задача больше statevector-предела

Ниже 30 вершин. Полный вектор имел бы `2**30` амплитуд; аналитический источник
работает с плотной матрицей 30x30 и пакетами пар. Запускается короткий полный
RQAOA до остатка из трёх вершин. Это не заявление о 1000-вершинной полной
цепочке.
"""
)

code(
    """rng = np.random.default_rng(42)
large_nodes = [100 + 2 * i for i in range(30)]
large_edges = []
for u, v in combinations(large_nodes, 2):
    if rng.random() < 0.18:
        large_edges.append((u, v, float(rng.choice([-1.5, -0.5, 0.75, 1.25]))))
large_graph = WeightedGraph(large_edges, nodes=large_nodes)
large_config = deepcopy(config)
large_config['parameters'] = {'finish_at': 3, 'max_reductions': None}
large_config['correlation_source']['optimizer']['parameters']['points'] = 9
large_config['collect'] = {
    'reduction_history': False,
    'parameter_trajectory': False,
    'local_optimizer_history': False,
    'correlation_tables': False,
    'final_reduction_state': False,
}
started = time.perf_counter()
large = SolverFactory.from_config(large_config)
large_solution = large.solve(large_graph)
elapsed = time.perf_counter() - started
print('N:', len(large_graph), 'edges:', len(large_graph.edges))
print('Time:', elapsed, 's')
print('Status:', large.status, 'Reductions:', large.reductions_completed)
print('Cut:', large.cut_value, 'Shots:', large.resources['shots_used'])
assert set(large_solution) == set(large_graph.nodes)
assert large.resources['shots_used'] == 0
assert math.isclose(large.cut_value, large_graph.cut_value(large_solution), abs_tol=1e-12)
"""
)

md(
    """## 11. Итог и ограничения

Подтверждены: формула корреляций, средняя энергия, точная beta-минимизация,
конечная gamma-сетка, покрытие `all_pairs`, подключение к существующему RQAOA,
нулевой расход шотов, сохранение и повторная визуализация. Аналитический
источник не предоставляет полное распределение, шоты, CVaR/Gibbs/Best-of-K/
Threshold Probability, поля, `p>1` или GPU. RQAOA остаётся эвристикой; точный
остаток не доказывает глобальный оптимум исходной задачи.
"""
)

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "QA_max_cut_lib",
    "language": "python",
    "name": "qa-max-cut-lib",
}
nb["metadata"]["language_info"] = {"name": "python"}

client = NotebookClient(
    nb,
    timeout=900,
    kernel_name="qa-max-cut-lib",
    resources={"metadata": {"path": str(root)}},
    allow_errors=False,
)
client.execute()
nbf.write(nb, path)
print("written", path, "cells", len(cells))
