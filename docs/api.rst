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

.. autoclass:: QA_max_cut_lib.generators.subgraphs.RandomPartitionBuilder
   :members: build

.. autoclass:: QA_max_cut_lib.generators.subgraphs.GreedyAbsWeightPartitionBuilder
   :members: build

.. autofunction:: QA_max_cut_lib.generators.subgraphs.build_subgraph_builder

Солверы
-------

.. autoclass:: QA_max_cut_lib.solvers.base.BaseSolver
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.exact_cpu.ExactCPU

.. autoclass:: QA_max_cut_lib.solvers.qaoa_cpu.QAOACPU

.. autoclass:: QA_max_cut_lib.solvers.rqaoa_cpu.RQAOACPU
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.ls_rqaoa_cpu.LSRQAOACPU
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.ls_components.MeanCorrelationAggregator
   :members: aggregate, make_table

.. autoclass:: QA_max_cut_lib.solvers.ls_components.ObservedMaxAbsCorrelationSelector
   :members: select

.. autoclass:: QA_max_cut_lib.solvers.components.QAOAStatevectorCorrelationSource
   :members: estimate

.. autoclass:: QA_max_cut_lib.solvers.components.MaxAbsCorrelationSelector
   :members: select

.. autoclass:: QA_max_cut_lib.solvers.components.ExactCPUFinisher
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.components.CorrelationResult

.. autoclass:: QA_max_cut_lib.solvers.components.SelectionResult

.. autoclass:: QA_max_cut_lib.solvers.analytic_p1.AnalyticP1Evaluator
   :members: correlations, correlation_matrix, energy_min_beta, expected_energy, expected_cut, scan_gamma

.. autoclass:: QA_max_cut_lib.solvers.analytic_p1.AnalyticP1CorrelationSource
   :members: estimate

.. autofunction:: QA_max_cut_lib.solvers.objectives.validate_objective

.. autofunction:: QA_max_cut_lib.solvers.objectives.objective_value

.. autofunction:: QA_max_cut_lib.solvers.qaoa_cpu.state_at_angles

Редукции
--------

.. autoclass:: QA_max_cut_lib.core.reduction.IsingProblem
   :members:

.. autoclass:: QA_max_cut_lib.core.reduction.SpinConstraint
   :members:

.. autoclass:: QA_max_cut_lib.core.reduction.ReductionState
   :members:

.. autoclass:: QA_max_cut_lib.core.reduction.GraphReducer
   :members:

Инструменты
-----------

.. autofunction:: QA_max_cut_lib.tools.numerics.correlation_matrix

.. autofunction:: QA_max_cut_lib.tools.numerics.quality_score

.. autofunction:: QA_max_cut_lib.tools.landscape.scan_landscape

.. automodule:: QA_max_cut_lib.tools.plotting
   :members: plot_graph, plot_graph_solution, plot_graph_info, plot_probabilities, plot_optimization, plot_landscape

.. automodule:: QA_max_cut_lib.tools.reduction_plots
   :members: plot_reduction_weights, plot_reduction_energy, plot_parameter_trajectories, plot_parameter_trajectories_3d, plot_analytic_gamma_profile, plot_ls_coverage
