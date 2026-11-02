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

.. autoclass:: QA_max_cut_lib.solvers.warm_start_qaoa_cpu.WarmStartQAOACPU

.. autoclass:: QA_max_cut_lib.solvers.warm_start_rqaoa_cpu.WarmStartRQAOACPU

.. autoclass:: QA_max_cut_lib.solvers.warm_start_ensemble.WarmStartEnsembleCorrelationSource

.. autoclass:: QA_max_cut_lib.solvers.conditional_qaoa_cpu.ConditionalQAOACPU
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.conditional.ConditionalCutProblem

.. autoclass:: QA_max_cut_lib.solvers.environment_ls_rqaoa_cpu.EnvironmentLSRQAOACPU
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.block_mean_field_qaoa_cpu.BlockMeanFieldQAOACPU
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.block_mean_field.MeanFieldBlockProblem

.. autoclass:: QA_max_cut_lib.solvers.distributed_ls.DistributedLSRQAOACPU
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.mi_observer.QAOAWithMIObserverCPU
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.mi_observer.RQAOAWithMIObserverCPU
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.incremental_ls.IncrementalLSRQAOACPU
   :members: solve

.. automodule:: QA_max_cut_lib.tools.mutual_information
   :members: reduced_density_matrix, von_neumann_entropy, quantum_mutual_information, classical_from_probabilities, classical_from_counts, classical_from_moments, map_summary, hop_distances

.. automodule:: QA_max_cut_lib.tools.mi_plots
   :members: plot_mutual_information_map, plot_mutual_information_frames, plot_mutual_information_distance

.. autoclass:: QA_max_cut_lib.solvers.rqaoa_cpu.RQAOACPU
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.ls_rqaoa_cpu.LSRQAOACPU
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.qaoa_gpu.QAOAGPU
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.qaoa_gpu.QAOAGPUCorrelationSource
   :members: estimate

.. autoclass:: QA_max_cut_lib.solvers.rqaoa_gpu.RQAOAGPU

.. autoclass:: QA_max_cut_lib.solvers.ls_rqaoa_gpu.LSRQAOAGPU

.. autoclass:: QA_max_cut_lib.solvers.qaoa_in_qaoa_cpu.QAOAInQAOACPU
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.qaoa_in_qaoa_gpu.QAOAInQAOAGPU
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.goemans_williamson_cpu.GoemansWilliamsonCPU
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.simulated_annealing_cpu.SimulatedAnnealingCPU
   :members: solve

.. autoclass:: QA_max_cut_lib.solvers.exact_block_refiner.ExactBlockRefiner
   :members: refine

.. autoclass:: QA_max_cut_lib.solvers.refined.RefinedSolver
   :members: solve

.. automodule:: QA_max_cut_lib.laboratory.catalog
   :members: load_catalog, load_inventory, list_variants, get_variant, get_file, source_to_variant, catalog_summary

.. automodule:: QA_max_cut_lib.solvers.warm_start
   :members: validate_warm_start_section, prepare_warm_start, product_state, state_at_warm_angles, mixer_unitary

.. automodule:: QA_max_cut_lib.tools.classical_plots
   :members: analyze_solution, plot_gw_convergence, plot_sa_reads, plot_refinement, plot_classical_comparison

.. automodule:: QA_max_cut_lib.tools.hierarchy_plots
   :members: plot_hierarchy_overview, plot_hierarchy_parameters, plot_hierarchy_parameters_3d, plot_hierarchy_tree

.. automodule:: QA_max_cut_lib.solvers.gpu_state
   :members: gpu_available, gpu_device_info, cut_diagonal_gpu, state_at_angles_gpu, restore_full_gpu, correlations_gpu, objective_value_gpu, sample_full_indices_gpu, adjoint_mean_cut_gradient_gpu

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

Эксперименты и профилирование
------------------------------

.. autoclass:: QA_max_cut_lib.core.experiment_spec.BenchmarkSpec
   :members: plan

.. autoclass:: QA_max_cut_lib.core.experiment_spec.ExperimentPlan

.. autoclass:: QA_max_cut_lib.core.runner.ExperimentRunner
   :members: run

.. autoclass:: QA_max_cut_lib.core.runner.ExperimentSeries

.. autoclass:: QA_max_cut_lib.core.references.ReferencePolicy
   :members: jobs_for, run_job, run_all

.. autoclass:: QA_max_cut_lib.core.profile.ParallelismProfiler
   :members: run

.. autoclass:: QA_max_cut_lib.core.profile.ProfileSpec

.. autofunction:: QA_max_cut_lib.core.profile.fit_amdahl

.. autofunction:: QA_max_cut_lib.core.profile.karp_flatt

.. autofunction:: QA_max_cut_lib.core.profile.recommend_total_curve

.. automodule:: QA_max_cut_lib.tools.experiment_plots
   :members: plot_series_coverage, plot_series_quality, plot_parallelism, parallelism_html

Checkpoints
-----------

.. autoclass:: QA_max_cut_lib.core.checkpoint.RunControl
   :members: request_pause, request_cancel, should_pause, snapshot

.. autoclass:: QA_max_cut_lib.core.checkpoint.CheckpointManager
   :members: write_generation, load, list_generations

.. autofunction:: QA_max_cut_lib.core.checkpoint.inspect_checkpoint

.. autofunction:: QA_max_cut_lib.core.checkpoint.restore

.. autoclass:: QA_max_cut_lib.core.checkpoint.ResourceJournal
   :members: begin, commit, resolve, reconcile

.. automodule:: QA_max_cut_lib.core.checkpoint_capabilities
   :members: CAPABILITIES, capability, capabilities
