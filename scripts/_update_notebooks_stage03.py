"""Update Quickstart and full guide for Stage 03, then execute them."""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

root = Path(r"C:/Users/Alexey/PycharmProjects/QA_max_cut_lib")

rqaoa_md = """## RQAOA на CPU

Базовый `RQAOACPU` обучает QAOA на текущем графе, получает физические
корреляции, выбирает пару и применяет точное ограничение. Остаток решает
`ExactCPU`; исходные спины восстанавливаются.

Полный разбор с таблицей редукций, шотовым режимом, остановкой по бюджету и
сохранением находится в [04_rqaoa_cpu.ipynb](04_rqaoa_cpu.ipynb). Полный вектор
состояния ограничивает размер исходного графа; ориентир 1000 вершин к этой
реализации не относится.
"""

rqaoa_code = """from QA_max_cut_lib import SolverFactory, WeightedGraph, tools
import math as _math

rqaoa_graph = WeightedGraph(
    [(10, 20, 1.5), (20, 30, -0.5), (10, 30, 2.0)],
    nodes=[10, 20, 30, 101],
)
rqaoa_config = tools.load_json(root / 'configs/solvers/rqaoa_cpu.json')
rqaoa_config['parameters'] = {'finish_at': 2, 'max_reductions': None}
rqaoa_config['correlation_source']['optimizer']['parameters']['max_steps'] = 2
rqaoa_config['output'] = {'report': 0, 'logs': 0, 'color': 'never'}
rqaoa = SolverFactory.from_config(rqaoa_config)
rqaoa_solution = rqaoa.solve(rqaoa_graph)
assert set(rqaoa_solution) == set(rqaoa_graph.nodes)
assert _math.isclose(rqaoa.cut_value, rqaoa_graph.cut_value(rqaoa_solution))
print('RQAOA cut:', rqaoa.cut_value,
      'reductions:', rqaoa.reductions_completed,
      'source:', rqaoa.solution_source)
"""

for filename, version_note in [
    ("01_quickstart.ipynb", "Версия **0.1.0a3**, третий CPU-этап. Добавлены RQAOA, его графики и учебный ноутбук 04; ExactCPU и QAOACPU сохранены."),
    ("02_current_api_guide.ipynb", "Версия **0.1.0a3**. Этот ноутбук охватывает реализованный функционал трёх CPU-этапов, включая базовый RQAOA. Полный целевой набор возможностей содержится в согласованной архитектуре и статусе переноса."),
]:
    path = root / "notebooks" / filename
    nb = nbf.read(path, as_version=4)
    first = nb.cells[0]
    source = first.source
    if "0.1.0a2" in source:
        source = source.replace("0.1.0a2", "0.1.0a3")
    # Replace the descriptive first paragraph fragment for clarity.
    lines = source.splitlines()
    if lines:
        for index, line in enumerate(lines):
            if line.startswith("Версия **0.1.0a3**"):
                lines[index] = version_note
                break
        first.source = "\n".join(lines)
    nb.cells.append(nbf.v4.new_markdown_cell(rqaoa_md))
    nb.cells.append(nbf.v4.new_code_cell(rqaoa_code))

    # Correct stale statements in the existing final markdown.
    for cell in nb.cells:
        if cell.cell_type == "markdown" and "Для следующих этапов предусмотрены" in cell.source:
            cell.source = cell.source.replace(
                "Для следующих этапов предусмотрены заменяемые компоненты LS/RQAOA,",
                "Базовый RQAOA уже реализован; для следующих этапов предусмотрены аналитический p=1, LS-RQAOA,",
            )
        if cell.cell_type == "markdown" and "В этой версии реализованы ExactCPU и QAOACPU." in cell.source:
            cell.source = cell.source.replace(
                "В этой версии реализованы ExactCPU и QAOACPU. LS, RQAOA, GPU, Laboratory, общий бенчмарк и контрольные точки ещё предстоит перенести; этот ноутбук не имитирует их работу.",
                "В этой версии реализованы ExactCPU, QAOACPU и RQAOACPU. Аналитический p=1, LS, GPU, Laboratory, общий бенчмарк и контрольные точки ещё предстоит перенести; этот ноутбук не имитирует их работу.",
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
