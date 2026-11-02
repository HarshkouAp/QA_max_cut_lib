"""Add the Stage 04 analytic example to the existing guides and execute them."""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

root = Path(r"C:/Users/Alexey/PycharmProjects/QA_max_cut_lib")

marked = """## Аналитический p=1

Источник `qaoa_analytic_p1_cpu` вычисляет средний разрез и физические
ZZ-корреляции без вектора состояния `2**N`. Он поддерживает только Mean Cut,
`p=1` и точные моменты; шоты и остальные критерии остаются у statevector
источника. Полный пример с профилем gamma и задачей из 30 вершин находится в
[05_analytic_p1_rqaoa.ipynb](05_analytic_p1_rqaoa.ipynb).
"""

snippet = """from QA_max_cut_lib import SolverFactory, WeightedGraph, tools
import math as _math

analytic_graph = WeightedGraph(
    [(10, 20, 1.5), (20, 30, -0.5), (10, 30, 2.0)],
    nodes=[10, 20, 30, 101],
)
analytic_config = tools.load_json(root / 'configs/solvers/rqaoa_analytic_p1_cpu.json')
analytic_config.pop('save', None)
analytic_config['parameters'] = {'finish_at': 2, 'max_reductions': None}
analytic_config['correlation_source']['optimizer']['parameters']['points'] = 9
analytic_config['output'] = {'report': 0, 'logs': 0, 'color': 'never'}
analytic = SolverFactory.from_config(analytic_config)
analytic_solution = analytic.solve(analytic_graph)
assert analytic.resources['shots_used'] == 0
assert _math.isclose(analytic.cut_value, analytic_graph.cut_value(analytic_solution))
print('Analytic RQAOA cut:', analytic.cut_value,
      'reductions:', analytic.reductions_completed,
      'shots:', analytic.resources['shots_used'])
"""

for filename in ("01_quickstart.ipynb", "02_current_api_guide.ipynb"):
    path = root / "notebooks" / filename
    nb = nbf.read(path, as_version=4)
    first = nb.cells[0]
    first.source = first.source.replace("0.1.0a3", "0.1.0a4")
    first.source = first.source.replace(
        "Добавлены RQAOA, его графики и учебный ноутбук 04; ExactCPU и QAOACPU сохранены.",
        "Добавлены RQAOA, аналитический p=1 CPU и учебные ноутбуки 04–05; ExactCPU и QAOACPU сохранены.",
    )
    first.source = first.source.replace(
        "включая базовый RQAOA.",
        "включая базовый RQAOA и аналитический p=1.",
    )
    # Add the new example at the end; the earlier cells remain the main guide.
    nb.cells.append(nbf.v4.new_markdown_cell(marked))
    nb.cells.append(nbf.v4.new_code_cell(snippet))
    for cell in nb.cells:
        if cell.cell_type == "markdown" and "python -m QA_max_cut_lib.solvers.rqaoa_cpu" in cell.source:
            cell.source = cell.source.replace(
                "python -m QA_max_cut_lib.solvers.rqaoa_cpu --graph configs/graphs/complete_signed_7.json --config configs/solvers/rqaoa_cpu.json --report 2 --logs 1",
                "python -m QA_max_cut_lib.solvers.rqaoa_cpu --graph configs/graphs/complete_signed_7.json --config configs/solvers/rqaoa_analytic_p1_cpu.json --report 2 --logs 1",
            )
    nbf.write(nb, path)
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="qa-max-cut-lib",
        resources={"metadata": {"path": str(root)}},
        allow_errors=False,
    )
    client.execute()
    nbf.write(nb, path)
    print("executed", path, "cells", len(nb.cells))
