Справочник Python
=================

Граф и результат
----------------

.. autoclass:: QA_max_cut_lib.core.graph.WeightedGraph
   :members:

.. autoclass:: QA_max_cut_lib.core.records.RunRecord
   :members:

.. autofunction:: QA_max_cut_lib.core.records.load_result

.. autofunction:: QA_max_cut_lib.core.saving.save_result

Фабрики и генераторы
--------------------

.. autoclass:: QA_max_cut_lib.core.factory.SolverFactory
   :members:

.. autoclass:: QA_max_cut_lib.generators.graphs.GraphFactory
   :members:

.. autofunction:: QA_max_cut_lib.generators.graphs.partition

Солверы
-------

.. autoclass:: QA_max_cut_lib.solvers.base.BaseSolver
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.exact_cpu.ExactCPU

.. autoclass:: QA_max_cut_lib.solvers.qaoa_cpu.QAOACPU

.. autofunction:: QA_max_cut_lib.solvers.objectives.validate_objective

.. autofunction:: QA_max_cut_lib.solvers.objectives.objective_value

.. autofunction:: QA_max_cut_lib.solvers.qaoa_cpu.state_at_angles

Инструменты
-----------

.. autofunction:: QA_max_cut_lib.tools.numerics.correlation_matrix

.. autofunction:: QA_max_cut_lib.tools.numerics.quality_score

.. autofunction:: QA_max_cut_lib.tools.landscape.scan_landscape

.. automodule:: QA_max_cut_lib.tools.plotting
   :members: plot_graph, plot_graph_solution, plot_graph_info, plot_probabilities, plot_optimization, plot_landscape
