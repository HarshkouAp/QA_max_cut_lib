# QA_max_cut_lib: requirements and architecture design

**Version:** 0.10, 1 November 2026.  
**Stage:** design based on code analysis and requirements discussion.

**Package name and Python import:** QA_max_cut_lib. Letter case and underscores are preserved.

The document records the adopted decisions and proposes the library structure. The "Agreed requirements" section reflects the user's answers. Class names, specific JSON fields, and the technical rules of the following sections are an implementation design for further refinement.

Source — [RQAOA_main.rar](https://chatgpt.com/library?file_id=libfile_6ad1141395e88191a0b4aaa32e78b1c5&account_id=personal). The source paths below are relative to the RQAOA_main directory inside the archive.

**SHA-256 of the source archive:** b4c684a5d4da469adc25bd99a405e402892d1a2524465e4c97ab6229c71a4aed

A static analysis of the structure of 89 Python files, their dependencies, and repeated methods was performed; the main families were studied in more detail. The volume is 25,548 lines. Numerical results, accuracy, and performance of the source algorithms were not verified in this analysis.

## Agreed requirements

| No. | Requirement |
|---|---|
| R01 | Four main sections: Solvers, Tools, Generators, Laboratory. The architecture uses OOP. |
| R02 | Composite algorithms are assembled from replaceable components. For LS-RQAOA, the correlation source, subgraph builder, reduction rule, and residual solver are selected independently. |
| R03 | One solver instance accepts different graphs: solver.solve(graph). |
| R04 | The solver stores additional results of the last run. The history of several runs is collected by a separate experiment object. |
| R05 | Each ordinary solve() starts a new run. Previous angles, solutions, and continuation of computations are used explicitly. Suitable working buffers may be reused. |
| R06 | A given seed is applied anew on each new run. |
| R07 | The graph object is mutable. |
| R08 | External vertex numbers are integers with possible gaps. Subgraphs and solutions keep the original numbers. |
| R09 | The graph JSON contains the generation recipe: generator type, parameters, and seeds. |
| R10 | The reference point for the nearest experiments is complete graphs up to approximately 1,000 vertices. |
| R11 | All graphs are weighted. Unit weights represent the ordinary unweighted case. |
| R12 | CPU and GPU implementations are separate algorithms. |
| R13 | The normal result of a solver is a dictionary original_vertex_number → +1 or −1. Additional results are available separately. |
| R14 | For reduction, existing edges or arbitrary pairs are allowed — by algorithm setting. |
| R15 | QAOA supports obtaining the solution as the most probable state or the best of M shots. |
| R16 | The user input is a Max-Cut problem on a weighted graph. Local fields are formed by the algorithms themselves during computation. |
| R17 | During QAOA training, either the exact objective estimate or a shot-based estimate is selected. |
| R18 | Mean Cut, CVaR, Gibbs, Best-of-K, and Threshold Probability are included with configurable parameters. |
| R19 | Laboratory preserves all substantively distinct experimental versions from the archive. |
| R20 | Old scripts can be adapted to the new unified interface. |
| R21 | New components are added to the library source code. |
| R22 | A common mechanism is needed for running experiment series and collecting results. |
| R23 | A series runs on one machine. CPU/GPU is selected for each solver; mixed series are allowed. |
| R24 | A common shot budget per run and configurable limits for individual stages, for example the number of optimization steps, are needed. |
| R25 | The total shot budget is strict: exceeding it is prohibited. |
| R26 | By default, the main results are saved; detailed data is enabled through configuration. |
| R27 | Continuation of an interrupted algorithm from a checkpoint is supported. |
| R28 | References: brute force for small problems, GW for medium problems, comparison of method energies and the best known solution for large problems. |
| R29 | Tools needs a ready-made computation of QAOA landscapes over a grid and their visualization. |
| R30 | Generators creates source graphs, subgraphs, and partitions. QAOA initialization remains among solver components. |
| R31 | Each full solver module contains standalone execution via `if __name__ == "__main__":`, including CPU/GPU implementations and Laboratory. |
| R32 | Tools builds probability distribution histograms: flat and as columns on a coordinate plane, including for large distributions. |
| R33 | Landscapes are displayed as flat heat maps and interactive three-dimensional surfaces with export to HTML. |
| R34 | For RQAOA and its composite variants, Tools builds parameter trajectories over reduction steps: on a plane and as interactive three-dimensional HTML. |
| R35 | After solve(), algorithm.result provides JSON with all final result attributes, metadata, and accumulated data of the last run. |
| R36 | Each result attribute is also available directly as algorithm.attribute_x. Direct access and result describe the same data. |
| R37 | The code is maximally transparent. Detailed comments and docstrings follow ASD-STE100 and explain the algorithm, the mathematical meaning, and the reason for each non-trivial transformation. |
| R38 | The automatic final report has an independent level argument: 0 — off, 1 — brief, 2 — detailed. |
| R39 | Logs have an independent level argument: 0 — off, 1 — brief, 2 — detailed. Logs and report have a single neat layout and color differentiation. |
| R40 | Tools builds optimization curves from accumulated data; the corresponding functionality from the base algorithms is moved into a common tool. For the LS family, the main progress plot is replaced by the reduction energy plots specified in R42–R43. |
| R41 | Tools supports simple rendering of a weighted graph, a graph with a solution, and a graph with basic problem information. |
| R42 | For the LS family, the main progress plot shows the fixed energy in reduced edges over reduction steps. |
| R43 | For LS, the energy notation E = −C is fixed. The contribution of the selected edge after fixing is −w(1−σ)/2; its mathematical expectation over the correlation is −w(1−c)/2. Per-step values and cumulative sums are plotted without normalization. |
| R44 | For LS and RQAOA, trajectories of optimized parameters are built at each reduction step. For p layers, all unordered pairs of the 2p parameters are included; flat plots and interactive HTML with the reduction step number on the Z axis are available. |
| R45 | Plots are needed of the weight of the selected reduced edge versus the reduction number and of the mean absolute edge weight versus the reduction number. |
| R46 | The package is accompanied by detailed executable tutorial notebooks on the use of the main capabilities. |
| R47 | All plots have a unified LaTeX style and LaTeX font, including labels, legends, scales, and interactive HTML. |
| R48 | The parameters of each plot are passed to the corresponding tool through a JSON configuration. |
| R49 | All algorithms and tools use the optional save section in the configuration. If it is absent, automatic export of the corresponding results is not performed. |
| R50 | For graphics, save sets path, format, DPI, filename, metadata, makedir, and, if needed, savedir. The main file is named filename.format. With metadata = true, the date and time, the algorithm, and the run and problem parameters are shown at the bottom of the plot. |
| R51 | The algorithm result is saved to JSON with the date and time, all used solver and graph parameters, the computation time, the number of stages, a detailed report, and the attributes selected via atts. |
| R52 | Attributes from save_original_data are additionally saved in their own formats with the base name filename_attributename; for a NumPy array, .npy is used. |
| R53 | makedir permits creation of a missing path. With savedir = true, a folder filename is created inside path, and the entire saved set is placed in it. |
| R54 | Overwrite protection is mandatory for all files and folders of the saved set, including additional arrays and plot collections. |
| R55 | A separate executable quick-start notebook is needed with the main examples of running, configuring, visualizing, and saving. |
| R56 | A separate detailed notebook is needed that describes all package functionality. Thematic notebooks complement it. |
| R57 | Separate dependency sets are defined for CPU and NVIDIA GPU. CPU installation and use do not require a GPU or CUDA. |
| R58 | Each algorithm and function is documented in the repository in NumPy/SciPy style: a strict physical and mathematical formulation, all arguments, results, and attributes. |
| R59 | The common argument num_treads controls CPU parallelism; it is described in signatures, JSON, and documentation. |
| R60 | Parallel execution of independent local tasks is implemented, especially LS subgraphs and independent QAOA-in-QAOA tasks, while preserving algorithm dependencies. |
| R61 | Tools contains a CPU parallelism profiler before the benchmark. It accepts the same task and solver specification as the common benchmark. |
| R62 | The profiler expands all admissible combinations of the given parameter grid and uses one selected seed. The thread-count recommendation is optimized for the whole range of configurations. |
| R63 | Curves of time, speed, and speedup versus the number of cores/threads are built; efficiency, parallelizable fractions, and the serial-fraction percentage are estimated with the model and the quality of the estimate indicated. |
| R64 | Seaborn is preferred for statistical 2D visualization. The styling of Another_pars.py serves as the user reference for the common theme while preserving the LaTeX font and JSON configuration. |
| R65 | The package and Python import name is QA_max_cut_lib. This name with the specified letter case is used in examples, the source layout, and run commands. |

## What the source code shows

| Directory | Python files | Observed role |
|---|---:|---|
| Solvers | 18 | Main solvers, as well as Benchmark and verification scripts |
| Laboratory | 22 | Experimental algorithms, subgraph construction, and diagnostics |
| Diplom | 28 | Historical MK series, QAOA variants with fields, and QAOA-in-QAOA |
| Tools | 11 | Partitions, metrics, generators, and MI-QAOA algorithms |
| Generators | 3 | Generation of graphs and problem sets, nonlocality study |
| ENV | 3 | Additional implementations of QAOA, GW, and generators |
| Other | 2 | Warm-start QAOA variants and experimental scenarios |
| RQAOA_exp | 2 | An additional QAOA/RQAOA pair |
| **Total** | **89** | |

Several architectural approaches are already present in the sources. RecursiveQAOA inherits from QAOA and modifies its internal state. LargeScaleRQAOA creates local QAOA and ClassicSolver inside its methods. QAOA-in-QAOA organizes a hierarchy of local tasks; one of its implementations uses global variables.

These differences require a common interface of computational capabilities, not a common large parent class with all arrays and methods.

In Solvers/LS_RQAOA.py, specific classes of QAOA, the dense subgraph builder, and the exact residual method are fixed. In Diplom/Mk_8_3.py, the correlation source is already replaced by QAOA-in-QAOA. In Diplom/Mk_5_2.py, subgraphs are updated around affected vertices after reduction. These are examples of independent extension points.

The duplication is substantial: the same convergence visualization method occurs in 13 files; definitions named _graph_reduction occur 32 times. These numbers characterize the code structure. A match of a method or class name does not by itself prove that algorithms are equivalent: dependencies, initialization rules, and the data used may differ.

### Substantive findings of the static audit

| Source | Observation | Consequence for migration |
|---|---|---|
| Solvers/QAOA.py, QAOA_GPU.py | Half of the symmetric state is stored; the sum of the stored probabilities is 1/2 | The state representation and its normalization must be explicitly known to common functions |
| Solvers/QAOA.py:mean_map | Correlations are computed over half of the probability mass without a factor of 2 | Verify and fix the physical normalization of correlators during migration |
| Solvers/QAOA.py:361–362 | result is used as a flag of matching one exact solution taking global inversion into account | Free result for the common JSON contract. Store the match with the reference in a separate field; absence of such a match does not prove suboptimality |
| Solvers/WS_QAOA.py and variants with fields | Use the full state vector | Unifying arrays requires accounting for the representation, not the same formula over raw probs |
| Solvers/WS_QAOA.py:64, 87–116 | The initial solution is selected using the exact brute-force optimum; the passed initial_state is replaced by an internal choice | Keep such a strategy as explicitly named initialization that uses the exact reference; implement the passing of an initial solution separately |
| Solvers/AnalyticalQAOA.py | Returns angles and expectation, is limited to unit weights, and does not receive a spin dictionary | Present its computational part as an evaluator; a full solver needs a separate stage for obtaining the solution |
| Solvers/AnalyticalRQAOA.py | Already contains the weighted p=1 formula, correlations, the reduction constant, and restoration of original numbers | Use as the source of standalone analytical and reduction components |
| Solvers/QAOA_GPU.py | NumPy and CuPy are switched within one class | Register separate CPU/GPU implementations in the new API |
| Solvers/FineTuning.py:83 | In both branches of the spin transformation, '1' is written | Fix the output-solution error and verify the improvement on the original graph |
| Solvers/Dwave_aneal.py:31 | When minimizing the Ising energy, J = −w/2 is set | For E = −C, J = +w/2 and a corresponding constant are required; the current sign defines a different objective |
| Solvers/Dwave_aneal.py:79 | The found sample remains a local variable | Bring the solution output to the common contract |
| Solvers/Dwave_aneal.py | SimulatedAnnealingSampler is used | In this directory this is classical simulated annealing, not a run on a quantum device |
| Solvers/GoemansWilliamson.py | Vertex numbers are used as SDP matrix indices, and rounding indices as vertex labels | Add an explicit mapping of external numbers to internal indices |
| Tools/GraphDecomposer.py, PartitionEfficiency.py | Mutual import | Separate partition generation and evaluation |
| Tools/MutInfo.py | A full density matrix is created via an outer product; there are demonstration computations at import | Isolate diagnostics, remove the demonstration from import; compute small reduced matrices without the full density matrix |
| Several directories | Ambiguous imports such as `from QAOA import QAOA`, global warning and plotting settings | Switch to package imports and explicit settings |

For the remark about annealing, it is sufficient to expand the sign convention:

$$
-C(s)=-\frac12\sum_{\{i,j\}}w_{ij}
+\frac12\sum_{\{i,j\}}w_{ij}s_i s_j.
$$

Therefore the pair interaction coefficient when minimizing −C is positive with respect to w. A historical file name does not define the problem being computed.

## Main principle of the proposed architecture

Separate **the algorithm description**, **the state of one run**, and **the computational components**.

A solver instance stores the configuration, the components, suitable working buffers, and the results of the last run. On solve(graph), a new run state is created. On explicit restoration, the state is loaded from a checkpoint.

This separation reconciles the reuse of one solver, the independence of new runs, memory savings, and continuation of interrupted computations.

## Distribution of responsibilities between modules

The placement of functionality is proposed below. The four public sections are preserved; a small internal Core contains common types and execution rules.

| Section | Responsibility |
|---|---|
| Solvers | Full Max-Cut algorithms; their own CPU/GPU computational implementations; optimization and initialization components |
| Generators | Source weighted graphs, weight distributions as part of their recipes, subgraphs, and partitions |
| Tools: numerical functions | Probabilities, samples, correlators, representation transformations, and other reusable operations |
| Tools: analysis | Quality metrics, series statistics, partition diagnostics, mutual information |
| Tools: visualization | Weighted graphs and solutions, probability histograms, optimization and reduction curves, heat maps, all pairwise LS/RQAOA parameter trajectories, interactive HTML, and the unified LaTeX style |
| Tools: result presentation | Common report and log formatters, detail levels, and color theme |
| Tools: saving | Common result exporter and plots per save, source-data serializers, metadata, and overwrite protection |
| Tools: research | ExperimentRunner, LandscapeScanner, and ParallelismProfiler for selecting CPU resources before the benchmark |
| Laboratory | Experimental algorithms and components; named assemblies that correspond to historical versions |
| Core, internal | Graph and index mappings, configurations, contracts, results, run state, budget, randomness, CPU resources, and checkpoints |

Classes are used for entities with state, configuration, or replaceable behavior. Simple mathematical transformations may remain functions. OOP does not require wrapping every formula in a class.

The numerical functions of Tools must have light dependencies and must not import the runner, the visualization, or specific experimental solvers. Analysis receives results through common types. The runner receives a component factory through a separate assembly layer.

Laboratory may use Solvers, Generators, and common functions. Base solvers use common interfaces; importing them must not require Laboratory. The common Core types do not depend on specific algorithms. Initial package loading remains light, and additional dependencies are loaded when the corresponding functionality is accessed.

### Main objects

| Object / interface | Purpose |
|---|---|
| WeightedGraph | Mutable source graph with explicit weights and stable external numbers |
| MaxCutSolver | Common interface solve(graph) returning a complete spin dictionary |
| RunRecord | Independent record of one run; the single data source for result and direct result attributes |
| RunState | Current algorithm state sufficient for continuation |
| RunContext | Services of one run: budget, randomness, data collection, and checkpoints |
| ParallelContext | CPU budget of one run, executor of local tasks, and nested-pool policy |
| QAOAEvaluatorCPU / QAOAEvaluatorGPU | Computations at given angles without necessarily running optimization |
| CorrelationProvider | Obtaining correlation estimates together with the coverage area and estimate information |
| SubgraphGenerator | Building selected subgraphs or a partition |
| PairSelector | Selecting a pair and a constraint sign from allowed and available estimates |
| GraphReducer | Applying a constraint, updating the problem and the restoration information |
| MaxCutSolver as residual method | Solving the remaining graph after reductions |
| Refiner | Improving a passed complete solution, for example by local correction |
| ExperimentRunner | Run series, separate history, references, and saving |
| BenchmarkSpec / trial plan | Common input of the benchmark and profiler: graphs, solvers, parameter grids, seeds, limits, and measurement mode |
| ParallelismProfiler | Scaling measurements, estimation of the effective serial fraction, and recommendation of the CPU thread count for the whole plan |
| LandscapeScanner | Grid traversal and map collection at fixed parameters; the result is passed to visualization facilities |
| DistributionData | Distribution over states or measurement frequencies with bit order, normalization, and representation method |
| ParameterTrace | Parameters and selected characteristics over reduction steps with binding to the problem and the local call |
| ReductionHistory | Weights of selected pairs, physical correlations, constraint signs, energy contributions, reduction constant, and working-graph statistics over steps |
| OptimizationHistory | History of optimizer steps and objective estimates with explicit counters and stages |
| Tools visualization facilities | Rendering of saved distributions, maps, and trajectories; static images and interactive HTML |
| Tools output facilities | Formatting of events and the final record; consistent colors, tables, and labels |
| SaveManager / data serializers | Common handling of save, attribute selection, file-name plan, and saving without replacing existing data |

These are object roles. The final class names and signatures of auxiliary methods are refined during implementation.

## Dependencies and installation profiles

Separate CPU and NVIDIA installation profiles are mandatory. The main package import and all CPU scenarios work without CuPy, CUDA, and the NVIDIA driver. The GPU module checks dependencies when the GPU implementation is accessed and reports the specific reason for unavailability. Selecting GPU does not automatically become a CPU algorithm.

The composition of dependencies is drafted below. This is a list for forming installation files and checking environments; it should not be presented as an already tested set of pinned versions of the new package.

| Group | Packages | Purpose |
|---|---|---|
| Common computing and data | numpy, scipy, networkx, pandas | Arrays, numerical methods, weighted graphs, and result tables |
| Configuration and output | jsonschema, rich | JSON validation and unified reports, logs, and progress |
| Visualization | matplotlib, seaborn, plotly | Statistical 2D plots with Seaborn/Matplotlib and interactive views |
| CPU parallelism and diagnostics | joblib, threadpoolctl, psutil | Task executors, native-pool control, and resource information |
| Classical methods | cvxpy, scs, dwave-samplers, dimod | SDP implementation of GW and classical simulated annealing |
| Additional partitions | python-louvain | Preservation of the partition functionality that uses community_louvain |
| Notebooks | jupyterlab, ipykernel, nbformat, nbclient | Running, editing, and checking tutorial notebooks |
| Reference build | sphinx, numpydoc, pydata-sphinx-theme, myst-nb | API documentation, mathematical sections, and notebook materials |

The full CPU profile includes common computing, configuration, output, visualization, CPU diagnostics, and supported CPU methods. The notebook and documentation-build groups are installed separately through requirements/notebooks.in and requirements/docs.in with the composition from the table. The GPU profile adds the selected CuPy to the CPU capabilities. The installation format is proposed as extras in pyproject.toml and as separate source requirement lists.

Draft requirements/cpu.in:

```text
numpy
scipy
networkx
pandas
jsonschema
rich
matplotlib
seaborn
plotly
joblib
threadpoolctl
psutil
cvxpy
scs
dwave-samplers
dimod
python-louvain
```

For NVIDIA, one of two profiles is selected:

| File | Addition to the CPU profile |
|---|---|
| requirements/gpu-nvidia-cu12.in | cupy-cuda12x[ctk] |
| requirements/gpu-nvidia-cu13.in | cupy-cuda13x[ctk] |

Each such file includes cpu.in and exactly one CuPy variant. According to the [official CuPy instructions](https://docs.cupy.dev/en/stable/install.html), the wheel variant depends on CUDA, and the ctk extra installs the required CUDA components into the environment; a compatible NVIDIA driver is required separately. When a system CUDA Toolkit is used, a profile without ctk is allowed with explicitly verified compatibility. Both CuPy variants should not be installed in one environment.

The profile is selected by GPU, driver, CUDA, and the supported Python version. CUDA 13 is not a universal replacement for CUDA 12 for older GPUs: [NVIDIA indicates the removal of Maxwell, Pascal, and Volta support](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html). Installation checks the availability of the device and the compilation of the kernel in use, not only a successful `import cupy`.

SCS is proposed as the open SDP backend for the first CPU profile of GW; the specific backend, tolerances, and solution status are part of the configuration and the result. Other backends may be separate dependencies. The library's own CMA-ES from the archive does not by itself require the external cma package. Unused imports of the sources are not transferred to dependencies automatically; for example, the import of dimension from SymPy in GraphGenerator.py is not used.

The LaTeX engine and the required fonts are system dependencies of the selected styling mode. They are described separately from the Python packages. HTML materials receive the required resources at export according to the previously defined self-containment rules.

For the release, verified lock files and a Python/OS/CPU/NVIDIA matrix are created. Experiment results store the versions of the package, NumPy/SciPy, the BLAS/OpenMP implementation in use, and, for GPU, CuPy, CUDA, and the driver. Compatibility and exact version constraints are declared based on the results of checking installation, algorithm execution, notebooks, and documentation builds.

## Graph and solution contract

External vertex numbers are preserved when subgraphs are built and in the output dictionaries. A separate number → index mapping is built for computations. For example, external numbers 2, 7, 15 correspond to internal indices 0, 1, 2.

The proposed initial domain is a simple undirected graph with finite real weights, including negative and zero weights. A structure with self-loops or multiple edges should be checked when it is converted to the input type. This is a technical proposal based on assumptions of the current implementations.

Each edge has an explicit weight. For the unit case, the recipe uses a constant weight of 1. The graph may store a sparse or a dense representation; the choice of representation does not change the mathematical problem.

The graph is updated through methods that maintain its version and index mapping. For a cache, the version number alone is insufficient: two different objects may have the same version. The key must account for the identity of the graph and its current state. Long-term records need a stable content identifier.

The solver stores the contents of the passed graph. Reductions are performed on a separate working problem. The run record fixes the state of the source graph to which the answer refers.

The solution dictionary contains each source vertex exactly once; the values are ordinary integers +1 or −1. Renumbering of internal arrays, removal of isolated vertices, and intermediate merges must be fully accounted for during restoration.

Common quality convention:

$$
C_G(s)=\frac12\sum_{\{u,v\}\in E}w_{uv}(1-s_us_v).
$$

C_G is maximized. Separate fields are proposed in the common result record:

| Field | Value |
|---|---|
| solution | Complete dictionary of source numbers and spins |
| cut_value | C_G(solution), computed on the source graph |
| solution_energy | −C_G(solution) |
| expected_cut | Mathematical expectation of the cut, when available |
| objective_value | Value of the selected training criterion |
| optimizer_loss | Value passed to the minimizing optimizer |

The objective_value of Threshold Probability is dimensionless; for Mean Cut it expresses the mean cut weight. They cannot be combined into one unspecified energy column. The modulus of the energy value does not replace the sign convention.

For quantum evolution, it is proposed to keep the convention of the main QAOA from the archive: H = −Ĉ and phase exp(−iγH). Analytical formulas that use +Ĉ are brought to it by an explicit sign transformation of γ. The Hamiltonian representation and the training criterion are separate concepts: changing CVaR to Gibbs changes the criterion, not automatically the phase operator.

Internal local problems may have the form c + Σ h_i s_i + Σ J_ij s_i s_j. They are stored separately from the source graph. The local value of such a functional must not be reported as the cut of the source graph.

## Run, result, and continuation

Proposed user-facing form:

```python
solver = SolverFactory.from_config(solver_config)

solution_a = solver.solve(graph_a)
result_a = solver.result

solution_b = solver.solve(graph_b)
result_b = solver.result
```

Each solve() creates a new RunState and re-initializes randomness with the given seed. Prepared buffers are reused when they fit the shape, data type, and current problem.

RunRecord is a separate record with its own solution dictionary. A subsequent solve() does not modify previously obtained solutions and saved records. For large arrays included in the record, data ownership must be defined: a reference to a rewritable working GPU buffer is not a saved result.

One solver instance performs one active run. For parallel jobs, the runner uses separate executor instances. This corresponds to the design of the current GPU code with shared working buffers inside an instance.

The main RunRecord contains the implementation identifier, the resolved configuration, the graph identifier, the seed, the solution, its quality, the status, the time, and the counters. Details are connected through collect. The state required to continue the algorithm belongs to RunState and may be saved independently of the user diagnostic settings.

### algorithm.result and direct attributes

Mandatory user contract: solve() still returns a dictionary source_vertex_number → ±1, and algorithm.result provides the complete outcome of the last run. The proposed form in Python is an ordinary JSON-compatible dict. It can be passed directly to json.dump() or json.dumps(); conversion to a string is not required to access individual fields. This clarifies the form of the Python API; it does not change the JSON requirement.

Example of access:

```python
import json

solution = algorithm.solve(graph, report=1, logs=0)
result = algorithm.result

cut_value = algorithm.cut_value
history = algorithm.optimizer_history

result_text = json.dumps(result, ensure_ascii=False, allow_nan=False)
```

Each public result attribute is represented by a named field of result. For example, result["cut_value"] and algorithm.cut_value give the same value; optimizer_history is available in both ways. Nested metadata may be separate fields, for example resources, timing, and config. For attributes with a special JSON representation, a documented reversible transformation is used.

RunRecord serves as the single source of the final data. Common typed properties and additional fields of a specific algorithm read this record. It is not allowed to assemble the JSON and the attribute set independently from two mutable copies. last_run may be kept as access to the RunRecord itself; the main user point of full export is result. New Laboratory fields are registered together with the name, type, description, units, and serialization rule. Service API names, for example solve and result, cannot be names of additional fields.

The full result includes the implementation identifier and version, the result schema, the run identifier, configurations and seeds, information about the graph that was actually solved, the status and reason for termination, the solution and available quality estimates, the final parameters, the time, resource counters, and all collected histories, distributions, correlations, and additional algorithm results. For nested algorithms, their origin and connection to a stage, reduction, or subgraph are preserved. Scientific result attributes must not disappear from the export because of a non-standard name or a large size.

"All accumulated data" means all data that were saved in the result of the given run. collect still controls the acquisition and saving of detailed diagnostics. Fields that were not collected or are not applicable to the method have an explicit status, for example not_collected or not_applicable, and a null value where the schema provides the field. If a public result attribute is available after solve(), its value is included in result. The export must not silently replace a saved array with a brief description or a reference.

The separate save.atts setting explicitly selects a subset of attributes for file export; this does not change the completeness of algorithm.result. The save_original_data list independently selects additional files in their source formats. The rules of selective saving are described in the common protocol section.

The solver object itself, working buffers, random number generators, GPU descriptors, and open files are not result attributes. Internal fields are used for them. The continuation state is stored in a checkpoint: result is the complete scientific outcome of a run, not a snapshot of all executable Python objects. A change in output styling does not change the composition of this record.

Proposed serialization rules:

| Value in Python | JSON representation |
|---|---|
| Ordinary and NumPy/CuPy scalars | Standard JSON numbers, strings, booleans, or null |
| Numerical arrays | Values, shape, data type, and explicitly specified axis order; without losing elements |
| Complex numbers and arrays | Separate real and imaginary parts with a description of the representation |
| Solution with integer vertex numbers | Object with JSON string keys; the schema specifies the restoration of integer numbers |
| Mathematically undefined indicator | null and a separate reason for unavailability; not NaN or Infinity |

In algorithm.solution and in the return value of solve(), the keys remain integers. The result loader restores the original integer numbers from JSON, including negative numbers and gaps. Transfer of saved GPU data to CPU and creation of the JSON representation are performed at an explicit request for the result; this is not a hidden part of every iteration. Reading result does not start additional measurements or optimization.

Before the first run, result has the value None. Declared result attributes also have the value None before data are obtained; an unknown attribute name raises the ordinary AttributeError. A new solve() creates a new record without mixing it with the previous one. On a failed run, the available partial record receives the corresponding status; previous results are not presented as new ones. The obtained result dict is an independent snapshot: a subsequent solve() and modification of this dict do not change the already issued snapshots or the internal results. Direct result properties are proposed to be made read-only.

Example of explicit restoration:

```python
restored = SolverFactory.from_checkpoint(checkpoint_path)
solution = restored.resume()
result = restored.result
```

Restoration uses the saved state of the random number generators. Re-initializing the seed during continuation would lead to a different sequence of computations.

## Common save protocol

The save section is the single configuration point for automatic export of the results of Solvers, Laboratory, and Tools. If save is absent, an algorithm or tool performs its work and returns the result without automatic saving of files and without creating export folders. The collect settings control data collection, output controls console output and display, and save controls writing to disk. Explicitly enabled checkpoints remain a separate checkpoint mechanism; they should not be created as a side effect of ordinary export.

An algorithm passes the record of its run to the exporter. For a tool, a record of its work is formed: tool name and version, configuration, input data or run identifiers, execution time, available attributes, and created artifacts. This record does not change the usual way of returning a plot or a numerical result. The common SaveManager uses the producer record and serializers by data type. A new algorithm or tool does not need its own file-export implementation.

### Common path parameters

| Field | Meaning |
|---|---|
| path | Destination directory; a relative path is resolved relative to an explicitly defined working directory |
| filename | Main name without an extension and without path components |
| makedir | Allow creation of a missing path and its parent directories |
| savedir | Create a separate folder filename inside path and place the whole set in it |

The proposed default values for makedir and savedir are false. If path is absent and makedir = false, saving fails with an error and writes no files. If path exists, savedir = true allows creation of the child folder filename even when makedir = false: the first flag applies to the base path, the second to the folder of the set.

With savedir = false, the result file is written as path/filename.json, and the image as path/filename.format. With savedir = true, these are path/filename/filename.json and path/filename/filename.format respectively. Additional files receive the same destination directory. filename and attribute names are checked before writing so that the resulting names are unambiguous and do not change the destination directory.

### Saving graphics

The tool configuration contains a block:

```json
{
  "save": {
    "path": "results/figures",
    "format": "png",
    "DPI": 300,
    "filename": "ls_reduction_energy",
    "metadata": true,
    "makedir": true,
    "savedir": true
  }
}
```

In the example, the file results/figures/ls_reduction_energy/ls_reduction_energy.png is created. The field preserves the user's spelling DPI. png, jpeg, and pdf are supported; interactive views support html, and vector export may also include svg. In the user example, pdg is treated as a typo for pdf. For formats that the specific renderer does not support, a clear error is issued before writing.

DPI sets the density of raster export. For vector formats, it applies to raster elements if they are present; geometry and fonts must not be deliberately rasterized for the sake of DPI. For HTML, the parameter does not change the interactive data; the scene size is set by the plot settings. These differences are noted in the allowed export configuration.

With metadata = true, a readable block is placed inside the exported image at the bottom: the run date and time with time zone, the algorithm name and version, the computational parameters actually used, the graph parameters and seeds, and the selected data slice. For a tool result, its name and parameters are indicated; if the input was an algorithm run, that context is also saved. The export date may be indicated separately from the date of the original computation.

The caption uses the common LaTeX style. Space is reserved for it, the text wraps, and the final image size is adjusted so that it does not cover the axes or cut off lines. The problem description includes the parameters and identifier of the graph that was actually solved; large arrays of its edges are not expanded into the caption instead of the plot. When several runs are compared, the source of each series remains distinguishable. With metadata = false, this block is not added.

The metadata flag controls the visible caption on the plot. It does not mean automatic creation of a separate JSON file and does not disable the mandatory JSON metadata of the algorithm result. If atts or save_original_data are additionally set for the visualizer, the common exporter also creates the JSON record of the tool and the selected source data according to the rules of the next section.

One save block sets one format. For another format, the export of the finished result can be explicitly repeated with another save without recomputing the scientific data. A collection of plots is saved as one multi-page PDF or one self-contained HTML if the format allows it. For several raster images, deterministic suffixes filename_page_001.png or filename_<panel_id>.png are used. The plan of all names is available before writing; plots cannot overwrite each other.

### Saving the result of an algorithm or numerical tool

Example configuration fragment:

```json
{
  "collect": {
    "optimizer_history": true,
    "probabilities": true,
    "statevector": true
  },
  "save": {
    "path": "results/experiments",
    "atts": [
      "solution",
      "cut_value",
      "solution_energy",
      "optimal_angles",
      "optimizer_history"
    ],
    "filename": "qaoa_run_001",
    "save_original_data": ["probabilities", "statevector"],
    "makedir": true,
    "savedir": true
  }
}
```

The main file is results/experiments/qaoa_run_001/qaoa_run_001.json. It contains the schema version, producer type and version, run identifier, start and end date and time, save date, the solver and component parameters actually used, graph parameters and identifier, seeds, computational resource, computation time, numbers of stages and operations, shot counters, status, reason for termination, and a detailed structured report on the results. Stage names are refined by method: optimizer steps, reductions, local calls, and the residual solution are not mixed into one unspecified counter.

Mandatory metadata and the report are supplemented with values from atts. Proposed rule: if atts is not specified, all accumulated public result attributes are saved; atts = [] saves metadata and the report without an additional set of scientific attributes. A full array explicitly included in atts is serialized according to the common JSON schema without silent reduction to a description.

The detailed report in JSON is saved independently of report and logs. For example, report = 0 does not prohibit a file report when save is present. The report is structured and contains no terminal color codes. Computation time is separated from file export time. The composition of the selected fields and the save configuration are fixed so that selective writing is not presented as a complete snapshot of all results.

Attribute names are checked against the common result interface. An unknown name is a configuration error. For a declared but unavailable JSON indicator, null and the reason for unavailability are saved according to the common schema. A request for a source file for data that were not collected or have no serializer ends with a clear error. save does not start additional measurements, optimization, or collection of disabled data; the required collect settings are specified in advance.

### Source data in their own formats

The atts and save_original_data lists are independent. An attribute from atts is saved inside the main JSON. An attribute from save_original_data is saved as a separate file of its type; the attribute name, relative file name, format, and restoration information are additionally written for it in the main JSON. If a name is in both lists, both representations are created. If it is only in save_original_data, large values are not duplicated inside the JSON.

For a NumPy array, .npy is used: the file has the name filename_attributename.npy and preserves the numeric type and shape of the array. In the example, qaoa_run_001_probabilities.npy and qaoa_run_001_statevector.npy are created. A complex array is saved as a complex array; to restore it from the source file, there is no need to combine the textual real and imaginary parts of JSON.

For a saved CuPy array, an explicit transfer of data to CPU and writing of a compatible .npy array are performed while preserving the numeric type, shape, and values. GPU descriptors and the working device context are not source scientific data for such export. Other structures use a registered serializer that declares the extension, the saved properties, and the method of reading back. The absence of a serializer does not lead to a silent replacement of data with a string.

The filename prefix is preserved also with savedir = true. Relative references in the main JSON allow the folder of the set to be moved as a whole. All additional files have the same overwrite protection guarantees as the main result.

Technical references — [the .npy format and numpy.save](https://numpy.org/doc/stable/reference/generated/numpy.save.html) and [image export parameters](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html). Specific serializers and renderers are selected during implementation of the common interface.

### Protection of existing data and completeness of the set

If the target file exists, saving is rejected with a list of conflicting names. With savedir = true, an existing folder filename is also considered a conflict, even if it is empty. Automatic renaming, adding time to the name, or overwriting are not performed: the user explicitly selects a new filename.

The check covers the whole set: the main JSON, images, source arrays, and collection pages. A conflict of one additional file must not leave new final files that look like a successfully saved complete set. Where possible, the configuration, the required serializers, and the already existing names are checked before expensive computations.

A single preliminary check of path existence is insufficient when two runs work simultaneously. The implementation requires exclusive reservation of names and publication without replacing an existing object. Files are first formed in a service area; the completeness of the set is recorded only after all parts are written successfully. An incomplete set is marked explicitly, and completed user source data are not deleted during cleanup of a failed export. Ordinary image or array writing functions must not bypass this protection on their own.

An export error is reported as a saving error and does not turn completed computations into a new run. The obtained scientific results remain available in memory for a repeated explicit export. Resuming an experiment is a separate mode: already completed records are read and remain unchanged, while new results receive new names. If the final file of a previous run already exists, the new outcome does not replace it.

## Standalone launch of each solver

**Mandatory requirement R31:** each full solver module contains an `if __name__ == "__main__":` block that runs this solver on a demonstration problem. This applies to separate CPU/GPU implementations and Laboratory solvers.

Such a launch uses the ordinary solve(graph) interface, common resource accounting, and the standard result record. The demonstration must use a small problem and a limited configuration suitable for the specific method. For a composite algorithm, the example parameters must engage its main stages.

It is proposed to move the common example preparation and argument handling into a helper function. A short entry tied to its own class remains at the end of the specific module:

```python
# Example module: solvers/qaoa_cpu.py.
# The helper name is an example.

def main(argv=None):
    from QA_max_cut_lib.tools.demo import run_solver_demo
    return run_solver_demo(QAOACPU, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
```

The common helper creates a weighted graph, forms the configuration of exactly the passed class, and calls solve(graph). The output of the spin dictionary, the cut weight, and the main outcomes uses the common report and obeys the report and logs levels. The example passes these levels explicitly and does not print a second report on top of the solver report.

The main proposed launch method:

```bash
python -m QA_max_cut_lib.solvers.qaoa_cpu
```

After package installation, direct execution of the file is also supported, including from an IDE. Correct package imports are used for this. An ordinary import of the module provides classes and functions without launching the demonstration, computations, or plotting.

Additionally, it is proposed to support passing a JSON graph recipe and solver configuration through command-line arguments. The exact argument names belong to the API design.

For a historical variant represented by a named configuration, a standalone example with the same main block is needed that selects the corresponding configuration. The common launch logic is reused.

## Transparency of scientific code and ASD-STE100

A mandatory requirement is that the code must explain the implemented algorithm. Comments and docstrings are written in technical English according to ASD-STE100. The standard defines language rules and a controlled dictionary; it permits domain-specific technical nouns and verbs. The source of the rules is the [official ASD-STE100 description](https://www.asd-ste100.org/about_STE.html) and the [official Issue 9 text](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf). The specific edition for the project should be fixed during implementation.

For the project, a unified glossary of mathematical and software terms is proposed. One term is used for one concept; in particular, cut value, signed energy, objective value, optimizer loss, optimizer step, and objective evaluation are distinguished. Formulas, variable notation, and source vertex numbers preserve precision regardless of language simplification. Simplified English must not replace the scientific explanation with general phrases.

The description of each algorithm includes the mathematical formulation, the sign convention, the scope of applicability, the sequence of stages, and the output data. A function description defines the purpose, the meaning of the arguments, the shape and order of arrays, the normalization, the result, and the essential constraints. Before each non-trivial transformation, a comment explains the reason for its use and the property it preserves: for example, why the weight is updated after spins are fixed, or why a factor of 2 appears when converting from the reduced state.

Example of a normalization explanation:

```python
# The stored amplitude represents two spin states.
# These states have the same probability.
# Multiply by two to get the probability of the pair.
pair_probability = 2.0 * abs(amplitude) ** 2
```

The text is built from short direct sentences. The active voice is used for actions, and terms remain consistent with the glossary. This applies to docstrings and to comments inside Python, CPU, and GPU kernels. The docstring block format is selected separately: ASD-STE100 regulates the language of explanations, not Python syntax.

Optimized sections additionally explain the relation to the source formula, the data layout, the symmetry used, the precision, and the reason for the chosen computation method. Approximations, heuristics, state truncation, and reuse of estimates are named explicitly. The source of a formula or research modification is indicated where it determines behavior. Names, function decomposition, and explicit parameters must allow the code to be matched to the algorithm description without searching for hidden switches.

During the migration review, not only the operation of the method but also the completeness of its explanations is assessed. Automated style checking helps to detect omissions; the mathematical correctness of the explanations and the compliance of the text with the language rules are checked separately. A large number of comments does not by itself confirm compliance with ASD-STE100.

## API reference and mathematical documentation

Each public algorithm, component, generator, and tool receives a reference page in the repository. NumPyDoc docstring format is used for functions, methods, and classes; the HTML reference is built by Sphinx from the same sources. The structure follows the [numpydoc guide](https://numpydoc.readthedocs.io/en/latest/format.html), and the language of comments and docstrings follows the previously adopted ASD-STE100. The structure format and the language rules complement each other.

| Section | Mandatory content |
|---|---|
| Summary | Purpose of the object and the problem solved |
| Parameters | Each argument: type, shape, default value, allowed values, units, and meaning |
| Attributes | Each available result attribute: type, shape, meaning, time of appearance, and availability conditions |
| Returns | Returned objects and their exact contracts |
| Raises | Configuration, capability-compatibility, resource, and saving errors |
| Notes | Physical and mathematical formulation, formulas, and computational assumptions |
| References | Sources of formulas, the algorithm, and research modifications |
| Examples / See Also | Reproducible examples and related methods |

In the mathematical description of an algorithm, the input problem, the variables, the Hamiltonian and the E = −C convention, the state and operators used, the training criterion, the sequence of transformations, the stopping conditions, and the answer restoration are defined. Preserved invariants are given for the transformations. Exact and approximate stages, admissibility of weights and fields, symmetries and normalization, memory costs, computational complexity under explicitly stated assumptions, and the implemented parallelism capabilities are indicated.

For each scientific function, the definition of the computed quantity and its relation to the code are given. For example, it is not enough to name an array energies: the description must distinguish the solution energy, the expected energy, the training criterion, and the reduction constant. For correlators, the physical normalization, vertex order, coverage, and currency are indicated. For CPU/GPU, the numerical precision, data layout, and explicit transfers between devices are described.

Theoretical guarantees are formulated together with the conditions of applicability. For example, the standard GW guarantee for non-negative weights cannot automatically be declared a guarantee for mixed signs. The complexity and parallelizability of a specific QAOA-in-QAOA assembly depend on the task tree and data exchange. The result of local angle optimization is not called a proven global optimum.

All common and specific arguments are documented, including seed, collect, report, logs, save, checkpoint, and num_treads. Separate tables of all fields and default values exist for nested configurations. Attributes and result fields are linked to the unified result registry; adding a new Laboratory field requires a description, units, serialization, and an example of use.

For each substantive version, there is a card: source file, new name, difference from the base algorithm, replaced components, modification formulas, and checks. Repeated rationales can be described by a common page with exact references to the differences. Internal functions with non-trivial transformations also receive docstrings and explanations, even if they are not included in the public index.

The build checks the correspondence of signatures and Parameters, the presence of Attributes descriptions, mathematical references, and the operability of examples. Documentation of GPU objects is built without creating a CUDA context at import. Completeness is tracked by the matrix "public object → API page → mathematical description → example in the guide".

## Report, logs, and output styling

The report describes the outcome of computations after solve() completes; logs describe the course of execution. They are controlled independently by two arguments, report and logs. A unified interface solve(graph, report=..., logs=...) is proposed, the same for Solvers and Laboratory. Default values can be set in the configuration, and explicit arguments of a specific call take priority. The technical proposal for ordinary use of the library is both levels equal to 0 by default; standalone examples select the levels explicitly.

| Level | report: final report | logs: course of computations |
|---|---|---|
| 0 | Do not output the automatic report | Do not output progress logs |
| 1 | Brief: algorithm and resource, graph size, status, solution quality, time, and shot consumption | Major stages and compact progress: optimization, reduction, completion |
| 2 | Detailed: configuration, available result characteristics, stages, stopping reasons, resource statistics, and reference information | Detailed: optimizer steps, objective estimates, reductions, local tasks, and counters with stage identifiers |

Both levels accept only 0, 1, or 2. For example, report=2, logs=0 gives a detailed outcome without intermediate output; report=0, logs=2 gives a detailed course without an automatic final report. Disabling output does not suppress exceptions and does not turn a failed execution into a successful one.

The collection of diagnostics via collect is independent of the output levels. logs=0 does not disable optimizer_history if its collection is enabled. report=2 does not start computation of a missing reference, does not perform new shots, and does not retroactively enable a history that was not collected. The output uses already available information and explicitly marks unavailable indicators. The detailed report summarizes large arrays for human reading; the full result still preserves their values.

All methods use the common formatting mechanism of Tools. The color theme distinguishes stage names, main numbers, successful completion, limit stop, and error. Color supplements the textual status. Successful completion of computations does not mean proven optimality; an optimum mark requires an appropriate basis. The existing color differentiation of the sources should be moved into the common theme instead of independent ANSI sequences in each solver.

The styling includes consistent headings, indentation for nested calls, table alignment, units of quantities, and reasonable numerical precision. A color setting is proposed with the values auto, always, and never. In auto mode, color is used with suitable terminal output; text export remains readable without control codes. JSON and numerical data are always saved without color codes.

A composite solver controls the common output through RunContext. Child components pass structured events with stage identifiers, and the common formatter decides what to show at the given level. After the parent solve() completes, one common final report is output; the detailed level may include information about local calls without duplicating many independent reports.

Changing the output levels and the color theme must not change pair selection, the random sequence, the number of steps, or shot consumption. Printed text is formed from structured events; saved events, if their collection is enabled, are also included in result. The measured execution time may differ because of output cost.

## QAOA as a set of components

For QAOA, the state/observable evaluator, the training criterion, the method of its estimation, the optimizer, initialization, and answer acquisition are separated. Some capabilities depend on the implementation and are checked at assembly.

```text
flowchart TD
    W["Working problem"] --> E["QAOA evaluator"]
    A["Angles"] --> E
    E --> D["Observables or sample"]
    D --> J["Objective function"]
    J --> O["Optimizer"]
    O --> A
    D --> K["Correlations"]
    K --> R["RQAOA reduction"]
    R --> W
    D --> M["Solution acquisition"]
    M --> S["Spin dictionary"]
```

An angle-optimization object may be used as a correlation source without necessarily obtaining a discrete local answer. This saves final measurements in composite algorithms that only need correlations at this stage. A full public solver still keeps the contract of returning a spin dictionary.

### Training criteria

All criteria are defined to maximize quality. The transition to a minimizing optimizer is specified explicitly.

| Criterion | Design parameters | Rule |
|---|---|---|
| Mean Cut | — | Mathematical expectation of C |
| CVaR | alpha, 0 < alpha ≤ 1 | Mean of the upper alpha tail of the quality distribution; alpha > 0 |
| Gibbs | eta > 0 | log E[exp(eta·C)] / eta |
| Best-of-K | Integer K ≥ 1 | E[max(C₁,…,C_K)] for independent measurements |
| Threshold Probability | Finite threshold A | P(C ≥ A) |

For CVaR, the partial mass at the tail boundary must be taken into account. For Gibbs, a numerically stable exponential estimate is needed. K in Best-of-K is a criterion parameter; actual shots are accounted for by the measurements created.

In shots mode, the criteria are estimated from the empirical distribution. When several criteria are analyzed together, one sample can be reused: measurements are counted once, and the computation of several statistics does not create new shots.

### Evaluator capabilities

The evaluator reports what it supports: the number of layers, weighted problems, the full vector or analytical moments, fields, gradients, sampling, access to probabilities, and estimation of arbitrary pairs.

The presence of an analytical mean at p=1 does not mean that the full distribution, CVaR, or the most probable string is available. Such combinations are checked before computations start. The weighted formulas from AnalyticalRQAOA can be isolated into a standalone analytical component.

CPU and GPU have separate implementations and names, for example qaoa_cpu and qaoa_gpu. Common contracts, checks, and mathematical definitions are reused. Selecting GPU must not imperceptibly turn into execution of a CPU algorithm.

A complete graph of 1,000 vertices belongs to the scale of composite methods. The subgraph size for state simulation is limited by the specific implementation and the available memory.

### State representation

Common functions receive information about the representation: vertex and bit order, full or symmetrically reduced state, normalization, and data layout.

In the reduced representation used in the archive, one stored amplitude represents a pair of globally inverted strings. The mass of the pair is 2|ψ|². During sampling, the pair and one of its two representatives are selected. For pair ZZ correlators, both strings have the same contribution.

Warm start and problems with local fields may violate the symmetry in use. The applicability of the reduced representation is determined by the properties of the state and the evolution.

Intermediate GPU arrays are proposed to be kept on the device. Transfer of large data to CPU is performed explicitly on request, analysis, or saving. The main spin dictionary and scalar outcomes are returned as ordinary Python values.

## Composite algorithms and experimental versions

LS-RQAOA contains a subgraph builder, a correlation source, a rule for combining estimates, pair selection, a reducer, and a residual solver. The services of one run are passed down the component tree through RunContext.

Example of the logic of one step:

1. Determine the current local problems.
2. Obtain or update correlation estimates.
3. Combine the estimates taking their coverage and the selected strategy into account.
4. Select an admissible pair and constraint sign.
5. Apply the reduction, update the restoration data and the complete available solution.
6. Record the required diagnostics and a consistent checkpoint.

The hierarchical QAOA-in-QAOA uses a similar approach to local solvers and level state. FineTuning is conveniently represented as an improvement component and a "base solver + improvement" wrapper.

A small modification receives a separate component or a named configuration. A change in the sequence of stages, the reduction method, or the process state may require a standalone algorithm class.

The card of a historical variant must fix the source, subgraph construction, initialization, evaluator, correlation rule, pair selection, reduction, completion, and additional corrections. Differences in caching are also recorded if they affect the estimates used.

### Correlations and coverage area

The source returns values together with information about which pairs were estimated and to which problem the estimates belong. The internal representation may be a dictionary or a matrix with a mask.

The following must be distinguished:

- pairs allowed by the algorithm;
- pairs for which estimates were obtained;
- currency of the estimates relative to the current state;
- the numerical value of the correlation.

The value 0 does not denote the absence of a computation. In a dense representation, an availability mask or an explicit designation of a missing estimate is needed. Pair selection is performed in the admissible domain taking the available data and the selected currency policy into account.

The "any pairs" domain allows non-adjacent vertices. The coverage of local subgraphs determines which of these pairs are actually estimated. If the algorithm necessarily requires full coverage, it is checked separately.

The sign of the fixed constraint is always +1 or −1. For a zero estimate, a definite selection rule is needed; direct use of sign(0) is not allowed for a spin constraint.

### Technical cache and estimate-update strategy

Buffers, index tables, and unchanged parts of the Hamiltonian can be reused while preserving the result of the computation.

Keeping old correlations after dependent inputs have changed is a separate algorithm decision. This may be a meaningful heuristic, but it must have a name, parameters, and currency diagnostics.

Therefore it is proposed to separate the technical cache and CorrelationUpdateStrategy. The technical cache checks all essential dependencies. The update strategy may deliberately select subgraphs, recompute part of the estimates, or use accumulated values.

For local problems, the dependencies may include not only the weights of internal edges but also fields, the neighborhood, the initial state, the circuit parameters, and the estimation method.

### Reduction invariant

Under the constraint s_u = σs_v, the interaction weights with the remaining vertices are updated by the rule:

$$
w'_{vk}=w_{vk}+\sigma w_{uk}.
$$

For the cut value, a constant addition is required. When u is removed, it increases by:

$$
\Delta b=\frac{1-\sigma}{2}\sum_{k\in N(u)}w_{uk}.
$$

The sum includes the edge u–v if it exists. For a non-adjacent pair, its contribution is zero.

The reducer preserves the invariant:

$$
C_{\mathrm{original}}(s\mid s_u=\sigma s_v)
=b+C_{\mathrm{reduced}}(s').
$$

For a sequence of reductions, the accumulated constant and the restoration constraints are stored. The final cut is additionally computed directly on the source graph.

## CPU parallelism and num_treads

The name of the public argument is preserved exactly as num_treads, according to the user requirement. It sets the maximum budget of simultaneously working computational CPU threads of one run. The proposed default value is 1; positive integer values are allowed. The field is in the common solver configuration and can be explicitly overridden by an argument of solve(). The requested value and the policy actually applied are recorded in the resolved configuration.

num_treads refers to CPU and does not set the number of CUDA threads in a GPU kernel block. One logical CPU thread is also not declared equal to one physical core. The metadata contain the available physical and logical CPUs, environment constraints, and, when used, core affinity.

ParallelContext manages local executors, native BLAS/OpenMP threads, and nested calls. The budget is not multiplied at each recursion level. For example, with eight CPU slots, eight single-threaded tasks are allowed, or another distribution with the same limit; eight tasks with eight native threads each is not execution within this budget.

The implementation describes parallel regions and sequential barriers separately. The metadata indicate the backend, the number of executors, the native limits, and the resources actually used. If a specific implementation remains sequential, this is explicitly reflected in its capabilities, documentation, and profiler report; one added argument is not considered an implemented speedup.

### LS and QAOA-in-QAOA

For LS, independent subgraphs of one reduction step are solved in parallel. All tasks of this stage receive a consistent snapshot of the working problem. After the local computations complete, the coordinator combines the estimates, selects a pair, and performs the reduction. The common graph is not modified simultaneously by several local solvers.

For QAOA-in-QAOA, an explicit dependency graph of local tasks is built. Independent branches or tasks of one admissible level are executed in parallel; dependent stages wait for the required results. If a historical version updates the neighborhood sequentially, the transition to synchronous exchange may change the algorithm and is formalized as a separate strategy, not as a hidden consequence of num_treads.

Each executor owns a separate instance of the local solver and its buffers. Executors and suitable buffers are reused between tasks and steps. Processes or threads are selected taking the executable code and the cost of data transfer into account. For an ordinary GIL build of CPython, pure Python computations require a separate process strategy; native operations may permit threads. For notebooks, worker functions are declared in importable package modules.

The randomness of a local task is determined by the root seed and a stable task identifier, not by the executor number or completion order. Aggregation, tie-breaking rules, and shot distribution are set deterministically by the coordinator. The strict shot budget applies to the whole task tree. A change in the number of threads preserves the mathematical definition of the method and is checked for consistency of results within the declared precision.

### Native pools and concurrent execution

When subgraphs are launched, the internal parallelism of NumPy/SciPy and third-party solvers is taken into account. Control of native pools is implemented at the level of the whole execution context and executor initialization. [threadpoolctl](https://github.com/joblib/threadpoolctl) serves as the proposed mechanism for inspecting and limiting supported BLAS/OpenMP pools. Parallel child tasks do not change each other's global settings on every call.

The parallel-trial mode of ExperimentRunner also consumes the CPU budget of the machine. The number of simultaneously executed trials and the num_treads of each solver are coordinated by a common scheduler. The admissible distribution is checked taking memory, available CPUs, and nested parallelism into account; it is saved in the result.

## Budgets and solution acquisition

The configuration distinguishes the total shot limit and the limits of individual stages: the number of optimization steps, local tasks per step, and other operations of the specific method.

The number of optimization steps, function evaluations, and used shots are different counters. One batch call of the evaluator may estimate several sets of angles; one gradient evaluation may require several measurement samples.

ShotBudget belongs to the whole run. Nested components use the common accounting. Measurements for initialization, training, correlations, additional diagnostics, and the final answer are included in the budget when they are actually performed.

One created sample can be used for several diagonal statistics. Such reuse does not increase shot consumption. Classical random rounding of GW is not considered quantum shots; its volume belongs to separate algorithm limits.

Before a measurement operation starts, the component checks the available budget. Measurements that exceed the remainder are not started. For a composite operation, for example the full set of measurements for one gradient step, a safe boundary and the volume requested in advance should be defined.

For the "best of M final shots" mode, it is proposed to reserve M before training. If the mandatory final measurements do not fit into the total budget, the configuration is rejected before the run starts. Training consumes the available remainder.

Normal completion of QAOA respects the selected readout at the angles produced by the optimizer:

| Mode | Answer acquisition |
|---|---|
| most_probable | The string with the maximum probability in the selected final state |
| best_of_shots | The maximum cut among the M strings actually obtained |

A composite method must support a complete admissible solution available when it stops because of the budget. The record indicates the reason for stopping and the source of the returned solution. Such completion is not proof of optimality.

Configuration errors, an unsupported combination of components, and execution failures should be distinguished from a regular stop by limit. ExperimentRunner saves information about a failed trial separately from a valid result.

## Checkpoints

A checkpoint contains the computation state sufficient for continuing the same run. The generation recipe and the initial seed are not sufficient for this.

| Data group | What is saved |
|---|---|
| Identity | Implementation identifier, format and code version, resolved configuration |
| Source problem | Actual graph and information about its origin, source numbers, and data order |
| Current problem | Working graph, fields and constants, restoration constraints, hierarchy levels |
| Control | Process stage, indices of steps and local tasks |
| Optimization | Angles, state of ADAM/CMA-ES/another method, and the required unfinished work |
| Randomness | States of all generators in use |
| Resources | Spent shots, reservations, stage counters |
| Solution | Available complete solution and its quality |
| Estimates and caches | Data that affect continuation, including old estimates under the corresponding strategy |

GPU pointers, CUDA streams, and executable graphs are restored as working resources. The necessary numerical data and a description for their preparation are saved.

Checkpoints are created at consistent stage boundaries. Their writing must be atomic so that an interrupted write does not destroy the last correct checkpoint. The periodicity is set by a setting for the corresponding algorithms.

The graph in a checkpoint stores the actual state after modifications. This is a separate computation-state format; the user graph JSON remains the generation recipe.

During continuation, the compatibility of the implementation, format, and configuration is checked. The shot counter is restored from the saved value. The absence of restoration support in an enabled variant is an unfinished implementation of the requirement, not a silent restart.

## Experiment series and analysis

ExperimentRunner receives graph recipes, solver configurations, trial seeds, reference settings, data collection, and saving. It forms independent run records on one machine.

One problem set can be passed to CPU and GPU implementations. The record retains the exact implementation name and the computation parameters. A solver instance may be reused sequentially inside its executor.

With save enabled, completed trials are saved as they are executed through the common exporter; without save, the history is collected in memory. The runner uses separate names based on trial identifiers and does not rewrite published results. When a series is resumed, completed records are skipped, and interrupted ones continue from explicitly configured checkpoints. The trial identifier must take the actual graph, the implementation configuration, and the seed into account, not only the row number.

References are organized by a separate policy: exact brute force for small problems, GW for medium problems, and comparison of methods/best known solution for large problems. Numerical boundaries are proposed to be set in the configuration and selected by measured cost.

A reference record contains the type, the cut value, the available solution, and the origin. The GW result and the best known solution remain heuristic reference points if there is no proof of optimality. A reference run has its own cost record.

The common result format is now fixed: JSON with metadata, report, and selected attributes, supplemented by source files from save_original_data. Checkpoint formats are defined separately. The series configuration agrees with the same save protocol used by individual solvers and tools.

For the execution mode, sequential launch for measuring individual methods and a parallel mode for accelerating the whole series are proposed. This is a technical proposal: the mode and the actual resource contention must be reflected in the experiment data.

### ParallelismProfiler before the benchmark

The tool accepts the same BenchmarkSpec that ExperimentRunner uses: graph recipes and parameters, solver types, fixed parameters and grids, seeds, budgets, and measurement rules. A separate profiling configuration sets num_treads candidates, the choice of one seed, warmups, repetitions, CPU placement mode, statistics, and saving. The expansion of trials is performed by the common scheduler so that the profiler and the main benchmark check one set of tasks.

Designed interface:

```python
benchmark_config = load_json("experiments/benchmark.json")
sweep_config = load_json("tools/parallelism_sweep.json")

profile = tools.profile_parallelism(benchmark_config, config=sweep_config)

plot_config = load_json("plots/parallelism.json")
figure = tools.plot_parallelism(profile, config=plot_config)
```

The profiler returns a full tool result with result and direct access to the described result fields. Measurements can be analyzed again from the saved JSON without new trial runs. Plotting can be enabled in the profiler configuration through the common visualizer.

#### Full parameter range and one seed

All parameters with finite given lists are expanded into a Cartesian product. Conditional spaces take component compatibility into account: for example, the parameters of a specific criterion apply to that criterion. Skipped incompatible combinations and the reasons are recorded. A continuous interval requires an explicit finite grid; "all configurations" does not mean enumerating an infinite set of real values.

The num_treads axis is set by the sweep itself, and repetitions over seeds are replaced by one selected seed from the plan. If there is no explicit choice, the first seed in the given list is proposed. Fixed recipe seeds and other graph parameters are resolved by the common scheduler. To compare thread counts, each configuration uses the same actual graph and the same algorithm setting.

By default, a random subset of combinations is not executed. A plan is available first: admissible configurations, the selected seed, the number of measurements, thread candidates, and resource constraints. If a given candidate did not pass all required configurations because of an error or insufficient memory, it does not receive a recommendation for the full range. Errors are not hidden by averaging over the remaining easier tasks.

#### Measurement methodology

A baseline point num_treads = 1 is mandatory. For each configuration and each candidate, warmups and several measurements are performed with repeated initialization of the same seed. Warmups are excluded from the time but included in the total profiling cost. Initial arrays, parameters, limits, and the buffer-reuse policy agree with the main benchmark.

The measurement loop of the benchmark and the profiler is common. It explicitly specifies whether solver creation, buffer preparation, and executor shutdown are included in the measured time. Output and export of trial runs are controlled by a separate explicit measurement policy and must not accidentally write files under the names of the main benchmark. The profiler's own save stores its outcome. The initial and actually applied settings are part of the measurement record.

The main mode isolates one algorithm run from other trials of the series. For parallel execution of the whole series, a separate mode for measuring the full batch with the same scheduler is provided; in it, the admissible resources of external trials and internal tasks are optimized jointly. These modes have separate results and labels.

Real steps, evaluations, local tasks, shots, status, and solution quality are recorded. If a change in the number of threads substantially changed the amount of computation, the point is marked as incomparable for estimating pure scaling. Parts of the algorithm must not be hiddenly disabled for the sake of a smooth curve.

Available physical cores, logical CPUs, affinity, and environment constraints are taken into account. If the platform supports it, placement by physical cores and with SMT can be compared separately. The number of executors and native threads is fixed at each point. The measurement order can be shuffled by a separate reproducible profiler generator; it does not change the randomness of the algorithm.

For a GPU algorithm, num_treads investigates only the CPU part and the effect of its execution on the overall run. CUDA blocks are not configured by this argument. GPU time is measured with correct synchronization; warmup and compilation are accounted for explicitly by the benchmark policy. The basis for the methodology is the [CuPy performance measurement recommendations](https://docs.cupy.dev/en/stable/user_guide/performance.html).

#### Curves and a recommendation for the full range

For each configuration q, all repeated measurements and their spread are saved. The representative time T_q(n) is proposed to be determined by the median. Time, execution speed 1/T_q(n), speedup, and efficiency are computed:

$$
S_q(n)=\frac{T_q(1)}{T_q(n)},\qquad
\eta_q(n)=\frac{S_q(n)}{n}.
$$

For sequential execution of the set, the target curve has the form:

$$
\mathcal T(n)=\sum_{q\in Q}m_q T_q(n),\qquad
n^*=\arg\min_{n\in\mathcal N_{\mathrm{feasible}}}\mathcal T(n).
$$

Q contains all admissible parameter and task configurations. The coefficient m_q sets the number of occurrences of the configuration in the benchmark plan; with an equal number of runs, m_q = 1 can be used. The final recommendation minimizes the time of the whole range, not the time of one selected configuration. Separate profiles are built for each solver implementation, and a general summary of the plan is also available.

For concurrent execution, the target curve is the measured time of the whole batch with the given scheduler. The sum of isolated times cannot be presented as the time of a parallel series. Recommendations include the measurement mode and the resources actually available. With statistically indistinguishable minima, the smaller resource can be explicitly selected by a configured rule.

The visualization shows T(n), speed or S(n), efficiency, individual configurations, and the summary curve. A heat map "configuration × number of threads", the measurement spread, and a plot of the effective serial fraction are available. All plots use the common LaTeX style, JSON configuration, and save.

#### Parallelizability coefficients

The basic Amdahl model defines:

$$
T(n)\approx T(1)\left(f_s+\frac{1-f_s}{n}\right),
\qquad 0\le f_s\le1.
$$

From a consistent set of measurements, f_s, the parallelizable fraction 1−f_s, and the percentages 100f_s and 100(1−f_s) are estimated. For individual points, the effective Karp–Flatt fraction is additionally computed:

$$
f_{\mathrm{eff}}(n)=
\frac{1/S(n)-1/n}{1-1/n},\qquad n>1.
$$

Fit residuals, uncertainty intervals, and model applicability are saved. The effective fraction includes the influence of exchange and overhead; it cannot be considered an exact percentage of the source code that is fundamentally not parallelized. Values outside [0,1] under noise, slowdown, or superlinear speedup are marked and not silently clipped.

If necessary, a separate overhead model is used, for example T(n) = a + b/n + h(n−1) with non-negative coefficients. It and the basic fit have different names. With insufficient data or poor agreement, the percentage is not declared a reliable estimate; the measured curves and the time minimum remain available.

Sources: [G. Amdahl, 1967](https://doi.org/10.1145/1465482.1465560) and [A. Karp, H. Flatt, 1990](https://www.cse.iitd.ac.in/~rijurekha/col380_2024/class3_0_karp_flatt_orig.pdf). For the GPU profile, the estimate means the fraction of time not accelerated by the selected CPU control and does not claim that GPU computations are sequential.

#### Profiler configuration and result

Example separate JSON:

```json
{
  "schema_version": 1,
  "num_treads_values": [1, 2, 4, 8],
  "parameter_coverage": "full_grid",
  "seed_index": 0,
  "warmup_runs": 1,
  "repetitions": 3,
  "execution_mode": "isolated",
  "timing_scope": "benchmark",
  "recommendation": {
    "criterion": "total_time"
  },
  "models": ["amdahl", "karp_flatt"],
  "save": {
    "path": "results/parallelism",
    "atts": [
      "expanded_plan",
      "measurements",
      "per_solver",
      "aggregate",
      "recommended_num_treads"
    ],
    "filename": "thread_sweep",
    "save_original_data": [],
    "makedir": true,
    "savedir": true
  }
}
```

Numbers and special field names are demonstrational; candidates are checked against the machine resources. The result contains the full expanded plan, the selected seed, the source measurements, CPU/GPU and native-pool information, the curves of each configuration and implementation, summary curves, fits, recommendations, errors, and applicability constraints. Metadata and selected attributes are saved to results/parallelism/thread_sweep/thread_sweep.json according to the common protocol.

### LandscapeScanner

The scanner accepts a graph, an evaluator, the selected parameter axes, the other fixed parameters, the grid, and the requested characteristics. For p=1 these are usually γ and β; for greater depth, a section over two parameters can be selected.

The result contains coordinates, map values, criterion parameters, estimation mode, seeds, and resource consumption. In shots mode, several criteria can be estimated from a common sample at each point.

Visualization works on the saved results. For additional maps of probabilities, correlations, and derivatives, the required computational capabilities are indicated. Unavailable or not yet computed points are marked explicitly.

It is proposed to use the common budget and checkpoint services for continuing a large scan. Computation and rendering are separated so that styling can be changed without recomputing the maps.

## Visualization in Tools

Visualization is an independent part of Tools with common settings for axes, labels, color scales, and export. It accepts saved data of one run or a series. Building a plot does not by itself start a solver, optimization, grid scanning, or additional measurements. Obtaining missing data is formalized as a separate explicit computation with its own resource accounting.

### Tool input and plot configuration

The proposed main input of analysis tools after computations is the solver object itself. The same tool accepts algorithm.result or a saved run record. A common adapter obtains the current result record and checks for the required fields. When a solver is passed, the required record data are read; serialization of all arrays to JSON is not required for each plot. The obtained plot fixes a specific run and does not change after the next solve().

Graph rendering also accepts WeightedGraph directly, and the variant with a solution accepts a graph and a spin dictionary. When a solver is passed, the graph snapshot of the run to which the solution refers is used. A subsequent modification of the user graph must not change already saved analysis results. For standalone rendering from JSON, a saved weighted graph snapshot is required; one generation recipe does not replace the problem that was actually solved.

All parameters of rendering, data selection, and export are passed through config — a JSON-compatible dictionary loaded from JSON. The input scientific data and the styling configuration have different roles. For example:

```python
plot_config = load_json("plots/reduction_trajectories.json")
figures = tools.plot_reduction_trajectories(algorithm, config=plot_config)

graph_config = load_json("plots/graph_with_solution.json")
figure = tools.plot_graph_solution(algorithm, config=graph_config)
```

The final method names are proposed. The common configuration schema contains the plot type, the selected indicators and data slices, axes, style, sizes, panel arrangement, labels, legend, and export settings. Special sections define the graph layout, layers, subgraphs, pairwise combinations, snapshot time, and the method of accumulating values. The configuration is resolved relative to one common theme; the final parameters and the run identifier are saved together with the export. The tools return a plot object or a named collection of objects for several panels.

### Unified LaTeX style

The common theme extends to all types of plots: graphs, probabilities, maps, surfaces, time curves, trajectories, and series statistics. The proposed base typeface is Computer Modern with a consistent math font set. Font sizes, line widths, axis ticks, grid, margins, palette, and unit notation are set centrally and overridden through the JSON of a specific tool. The styling is applied locally, without a global change of settings at library import.

The user reference is [Another_pars.py](https://chatgpt.com/library?file_id=libfile_51de58d2341881918758827aed783313&account_id=personal). For statistical 2D views, Seaborn is selected with the common Matplotlib Figure/Axes foundation. For graphs and special primitives, suitable Matplotlib operations with the same theme are used. The interactive 3D renderer receives a common set of colors, fonts, and notation separately.

The following styling profile is extracted from the file:

| Setting | Value in the reference |
|---|---|
| Export background | White |
| Text | Serif typeface; in the common theme refined to Computer Modern with a math font set |
| Image size | 14 × 8 inches |
| Axis labels | 19 pt |
| Tick labels | 16 pt |
| Legend | 15 pt; title 16 pt; frame enabled |
| Grid | By the vertical quantity, alpha = 0.3 |
| Export | 250 DPI, tight bounds, 0.1 inch outer padding |
| Distribution comparison | Seaborn boxplot, width = 0.8, separate groups and explicit series order |

This is the initial profile for setting up and scaling, not a mandatory size of every panel or every HTML view. The metric, data grouping, and computation of the ratio to the reference are set by a separate analysis tool. The order and colors of series are tied to stable method identifiers.

The Seaborn style and context are set inside the local plotting and export context. Default settings, global suppression of warnings, and CSV reading at import from the research script are not transferred into library modules. The reference sets the appearance; the tools receive computational data through the agreed interface. The implementation basis is the [official Seaborn styling guide](https://seaborn.pydata.org/tutorial/aesthetics.html).

The LaTeX style includes ordinary text, numbers, Greek letters, indices, legends, and color scales. For static export, a LaTeX typesetting mode is proposed; its dependencies and Cyrillic support are checked when the mode is selected. Switching to another typesetting mechanism is performed explicitly and saved in the styling metadata. User LaTeX text and ordinary labels are distinguished in the schema so that service characters do not disrupt typesetting.

HTML preserves the same typeface and notation system. For a self-contained file, the required font and math resources are included in the export. The renderer's ability to show LaTeX notation on three-dimensional axes is checked separately; the choice of libraries must satisfy this requirement. A font_family entry in JSON alone is not sufficient to confirm the correct display of all labels.

Technical references: [LaTeX typesetting in Matplotlib](https://matplotlib.org/stable/users/explain/text/usetex.html) and [math typesetting in Plotly](https://plotly.com/python/LaTeX/). Seaborn/Matplotlib are fixed as the foundation of statistical 2D tools; the specific mechanism of math labels of the interactive 3D renderer is checked during implementation.

Example JSON for all pairwise trajectories:

```json
{
  "schema_version": 1,
  "type": "reduction_trajectories",
  "selection": {
    "parameters": "all_angles",
    "pairs": "all_unordered",
    "subgraphs": "separate"
  },
  "views": ["2d", "3d"],
  "axes": {
    "z": "reduction_step"
  },
  "style": {
    "theme": "latex_scientific",
    "font_family": "Computer Modern",
    "font_size_pt": 11,
    "line_width_pt": 1.2,
    "grid": true
  },
  "layout": {
    "panels_per_page": 6
  },
  "output": {
    "show": false
  },
  "save": {
    "path": "figures",
    "format": "html",
    "DPI": 300,
    "filename": "reduction_trajectories",
    "metadata": true,
    "makedir": true,
    "savedir": true
  }
}
```

Numbers and special plot field names are demonstrational; the common save fields are agreed separately. The panels_per_page setting splits a large collection into pages while preserving all requested combinations. This JSON configures the analysis of already obtained data and does not change the reduction algorithm. The example saves a self-contained figures/reduction_trajectories/reduction_trajectories.html with all selected views. For a separate static export, save with the required format is used.

Proposed interface composition:

| Data | Flat view | Three-dimensional view |
|---|---|---|
| Weighted graph | Vertices, edges, weights, solution, and a basic information block | An additional interactive view is allowed by configuration |
| Probability distribution | State horizontally, probability vertically | Columns on a coordinate plane; height is the state probability |
| QAOA landscape | Heat map over two selected parameters | Interactive surface: two parameters and the value of the selected characteristic |
| LS/RQAOA reduction parameters | All pairwise parameter trajectories and individual parameters versus the reduction number | For each pair (θ_a, θ_b), an interactive trajectory with Z = reduction step number |
| Optimization history of a direct solver | Objective, energy, or quality value over steps, evaluations, shots, or time | An interactive view may use the common HTML mechanism; the mandatory basic form is flat curves |
| LS reduction progress | Fixed and weighted energy contributions, per step and cumulative | An additional interactive view uses the same data record |
| Reduction weights | Weight of the selected edge and mean absolute weight in the working graph versus the reduction number | An additional interactive view uses the same data record |

Interactive 3D landscapes and trajectories are saved to separate HTML files with rotation, zoom, and tooltips. A self-contained HTML with embedded data and the required scripts is proposed, openable without a server or network connection. For probability columns on a plane, the same mechanism can provide an interactive view. Static Seaborn/Matplotlib images and interactive versions are built from the same data and use the common theme; the 3D layer is separated from the statistical 2D API.

### Rendering graphs and solutions

Three ready modes are provided: a simple weighted graph; a graph with a solution; a graph with a basic information block. They can be combined: for example, show the solution and the summary on one image. The configuration controls vertex coordinates, the layout algorithm and its seed, source-number labels, weights, vertex sizes, edge thickness, and colors.

When the solution is shown, the two spin values are distinguished by vertex color, and cut edges are highlighted by a separate visual feature. The sign and modulus of the weight can be encoded additionally and are explained by the legend. A negative edge crossing the cut retains its negative contribution to quality. The solution dictionary is checked against the source vertices. The solution view uses the same coordinates as the simple view if one layout is passed.

The information block contains the number of vertices and edges, density, weight statistics, the number of components, and isolated vertices. For a solution, the cut value and part sizes are added. Additional indicators are enabled by configuration and computed only on explicit request; rendering does not start a search for an optimal solution for the sake of an informational caption.

For dense graphs, the JSON sets label visibility, transparency, a filter of displayed edges, and a limit on their number. Any filtering is indicated on the image together with the number of shown and actual edges. Display does not change the source problem. The snapshot of the source graph and the snapshot of the graph after the selected reduction are chosen explicitly; the intermediate graph is available if the corresponding data exist.

### Probability distribution histograms

The main distribution refers to bit strings or spin assignments of the selected problem. A histogram of cuts or energies is a separate grouping: several states may have the same value of the functional. The grouping type and the target quantity are explicitly indicated in the settings and labels.

The flat histogram supports the order of states by their index and sorting by probability. In the proposed column view on a plane, the bit string is divided into two blocks: the index of the first sets x, the second sets y, and the probability sets the column height. The user can select the split of bits. With an odd number of bits, the base is rectangular. The vertex order, bit order, and inverse mapping of coordinates to state are preserved; the tooltip shows the source state.

The data source may contain exact probabilities or empirical frequencies over M shots. These modes are labeled separately. The common transformations of Tools take normalization and the full or reduced state representation into account. For globally inverted pairs, individual states or the pair mass can be explicitly selected; they cannot be mixed on one plot under one label.

For large distributions, selection of the most probable K states, a specified subset, and grouping are provided. The plot indicates the mode, the shown probability, and the remaining mass when it is known. If the data source does not allow determination of the remaining mass, it is marked as unknown. Discarding states must not automatically renormalize the shown part to one. Frequencies from measurements distinguish unobserved states from states with a proven zero exact probability.

Placement on a plane improves readability but does not eliminate the exponential number of states. Tools must not create an array of 2^n elements only for rendering sparse frequencies or selected K states. The reference point of a source graph up to 1,000 vertices does not mean that its full distribution is available. For a composite solver, the saved distribution of a specific local problem or the distribution of final solutions over repeated runs can be visualized; the origin is mandatory.

### Heat maps and interactive landscapes

For p=1, the usual axes are γ and β; the color of the heat map or the height of the surface reflects the selected indicator. For p>1, the user selects two parameters and fixes the others, obtaining a section. The map stores the definition of the indicator, for example expected_cut, optimizer_loss, or CVaR with a given alpha, as well as the sign convention, estimation mode, and units of the quantity.

Rendering uses the coordinates and values of LandscapeScanner. Unavailable and not yet computed points are displayed as gaps. Interpolation, if enabled, is marked explicitly and does not replace measured points. The heat map and the 3D surface have consistent axis order and value scale. Grid data are saved separately from styling so that images and HTML can be rebuilt.

### Optimization curves

The common Tools tool builds curves from OptimizationHistory in algorithm.optimizer_history, algorithm.result, or a saved run record. The plot_convergence functionality from the base algorithms is moved here. In particular, the objective-change lines, layer-optimization boundaries, and reference horizontal lines available in the sources are preserved. If a convenient plot_convergence() method remains in the solver, it only passes data to the common tool.

For the LS family, the main progress plot shows the fixed energy of reductions and its weighted variant, described below. A change of indicator is reflected in titles and axes. Histories of local optimizers remain available by a separate request with selection of the reduction and subgraph; they are not presented as the general course of the LS algorithm.

The user selects the vertical indicator: optimizer_loss, the original objective_value, expected_cut, the quality of the available discrete solution, or another collected characteristic. The current and best achieved values are shown separately taking the optimization direction into account. Horizontally, the optimizer step, the number of objective evaluations, the accumulated number of shots, or the time from the start of the run can be selected if the corresponding coordinates were collected.

The optimizer step and the objective function call are distinguished. Central differences, line search, and batch computations may produce several evaluations per step. The history indicates optimizer_step, evaluation_id, the purpose of the evaluation, the stage, the layer, and the local task identifier. The tool does not label all objective calls as optimization steps. For recursive methods and several subgraphs, a separate local history can be selected; unrelated local optimizations are not merged into one curve without an explicit choice of the merging method.

Signed marks are provided for layer-optimization boundaries, restarts, and reductions. The reference line appears only if a suitable reference exists and the displayed quantity matches: an exact cut cannot automatically be drawn as the value of CVaR, Gibbs, or Threshold Probability. Comparison of several solvers uses consistent metrics and units; different criteria can be shown in separate panels.

Curves are built from saved values without additional objective evaluations. Smoothing is off by default; if it is selected, the method and processing parameters are shown. The tool returns a plot object and supports explicit saving or display. It does not open an automatic window during solve(). If the history was not collected or the method has no optimization stage, a clear message about data availability is output.

### Energy and weights in LS/RQAOA reductions

ReductionHistory saves at each step t the identifier and version of the working problem before the reduction, the selected pair (u_t, v_t), its current weight w_t, the physical correlation c_t = ⟨Z_u Z_v⟩, the constraint sign σ_t, and the problem statistics after the reduction. For a missing edge of the selected pair, w_t = 0; this is an admissible case when arbitrary pairs are selected. The current weight may differ from the weight of the source edge because previous reductions merge interactions.

The physical correlation and the estimate for pair selection are different fields. In the source LS_RQAOA.py, the estimate matrix is additionally scaled by weights. Such an estimate is not required to lie in the range [−1, 1] and is not used as a probability. The record separately includes the source correlator, the rule for combining local estimates, the currency, and the final selection_score.

For these plots, the common convention E = −C is fixed. The JSON selects the fixed contribution or its mathematical expectation, individual steps or the cumulative sum, the local series, and the axis labels. The convention is saved in the metadata, and the tool checks its consistency with the result. The sum of contributions should not be automatically divided by the number of steps or by the sum of correlations. If an average indicator is needed, it is a separate explicitly defined metric.

For the edge selected at step t, the following quantities are used:

$$
e_t^{\mathrm{fixed}}=-\frac{w_t}{2}(1-\sigma_t),
\qquad
\overline e_t=-\frac{w_t}{2}(1-c_t).
$$

The first quantity is the contribution of the selected edge after its spin product σ_t is fixed; the second is the mathematical expectation of the energy of this edge in the state before the constraint is introduced, from which c_t was obtained. In both formulas, w_t is the edge weight of the current working problem before the corresponding reduction.

The cumulative curves are defined uniquely:

$$
E_{\mathrm{fixed}}(t)
=-\frac12\sum_{k=1}^{t}w_k(1-\sigma_k),
\qquad
\overline E_{\mathrm{fixed}}(t)
=-\frac12\sum_{k=1}^{t}w_k(1-c_k).
$$

The second curve is a consistent definition of the requested "mean weighted fixed energy": it sums the expected energy contributions of the selected edges before fixing. fixed_energy and expected_fixed_energy are proposed for the field names, and the accumulation method is set separately. The factor 1/2 and the constant part −w_k/2 are included in the value; they are not discarded when plotting.

The physical correlation c_t is the mean spin product, not a probability. In particular:

$$
P(Z_uZ_v=\sigma_t)=\frac{1+\sigma_t c_t}{2}.
$$

The computation of the expected energy uses c_t directly in the formula −w_t(1−c_t)/2. Additional multiplication of the already fixed contribution by the probability of the selected constraint is not part of this metric. The correlation must refer to the indicated state and problem; for a deliberately outdated estimate, its origin is marked.

The main curves show the contribution of each step or cumulative sums over the selected reduction events. Since the working problem changes, the cumulative sum of selected edges describes this event log; it is not automatically declared the total energy of the source problem. Values may be non-monotonic, in particular with mixed weight signs.

The exact accumulated reduction constant b_t and its increment Δb_t from the invariant C_original = b_t + C_reduced are saved separately. In the energy convention, the constant part is −b_t. The increment depends on all affected edges of the removed vertex and in the general case does not coincide with the contribution of one selected edge. This plot has a separate name; it does not replace the requested statistics of selected edges.

For the weight of the reduced edge, a curve of w_t versus t is built with the weight before the corresponding reduction. For the mean absolute weight, the following is built:

$$
\mu_{|w|}(t)=\frac{1}{|\mathcal E_t|}
\sum_{e\in\mathcal E_t}|w_e^{(t)}|.
$$

Here \(\mathcal E_t\) is an explicitly specified domain of edges of the working graph. The proposed default mode is non-zero edges after step t; the JSON can also select all existing edges or all pairs if a different statistic is needed. Missing edges do not enter the denominator of the non-zero-edge mode. With an empty domain, the value is unavailable rather than arbitrarily equal to zero. The source graph corresponds to a separate point t = 0.

The weights of selected pairs, the number of edges, the sum of weight moduli, and the reduction constant can be collected compactly, without a snapshot of each full graph. This scalar trace is sufficient for the energy curves. The plots of weights, correlations, and the constant are also useful for RQAOA; for LS, the energy curves become the main representation of progress.

### LS/RQAOA reduction parameter trajectories

Reduction parameters include the angles of the local QAOA and the selected scalar characteristics of the step available to the specific method: the correlation of the selected pair, the value of the selection criterion, and other named parameters. A unified ParameterTrace is proposed, which the solver fills through the common data collection mechanism. It stores the reduction number, the identifier and version of the working problem, the number of remaining vertices, the identifiers of the local call and subgraph, the source numbers of the selected pair, the constraint sign, and the available parameters. Angles are recorded as separate named coordinates γ_l and β_l with the layer and convention indicated.

At each step, the angle set θ_t = (γ_0,…,γ_{p−1}, β_0,…,β_{p−1}) found by the optimizer is saved. The word "optimal" refers to the obtained optimization result and does not mean a proven global optimum. For p layers, the mandatory full set contains all unordered pairs of different parameters:

$$
N_{\mathrm{pairs}}=\binom{2p}{2}=p(2p-1).
$$

For p = 1 this is one trajectory, for p = 2 it is six, for p = 3 it is fifteen. γ–γ, β–β, and γ–β are included, including parameters of different layers. Permutations of one pair are not duplicated. The flat version shows θ_a versus θ_b with the order of reductions indicated. For each such pair, an interactive 3D trajectory (θ_a, θ_b, t) is built, where the Z axis necessarily sets the reduction step number.

Additionally, γ_l and β_l versus the reduction number and other selected characteristics are available. The full set of pairs can be output as a panel grid, pages, or an HTML collection with pair switching. The interface decomposition must not silently exclude combinations. Tooltips show the step, problem size, selected vertex pair, parameters, layer, and local data source. When the circuit depth changes, missing parameters produce gaps, not artificial zero angles. The trajectory of successive reductions and the history of optimizer iterations within one reduction are different data; the internal history is collected by a separate setting.

In LS-RQAOA, one reduction may have several local optimizations. They are saved with subgraph identifiers and displayed as separate series or a selected slice; averaging their angles requires an explicit definition. If the builder updates subgraphs, the origin record allows changed problems to be distinguished. If the method reuses previous angles or correlations, the trace indicates the source and the step of data acquisition. The trajectory does not connect unrelated local calls as one continuous sequence.

If the correlation source does not use QAOA angles, the corresponding coordinates are absent. For such a method, trajectories of other collected characteristics are available. Local values of the functional refer to changing problems; their labels preserve this binding and information about the reduction constant when it is needed for the selected quantity. Angles are displayed as saved by default; unwrapping of periodic coordinates is possible only with an explicitly specified correct periodicity, without a universal assumption for arbitrary weights.

### Data collection for visualization

It is proposed to enable the parameter trace with a separate collect.parameter_history flag, independently of the full optimizer_history and probability snapshots. Parameters, layers, local tasks, and the save frequency can be selected. This preserves the agreed principle: main outcomes are available by default, detailed diagnostics are enabled by configuration. Intermediate distributions require separate snapshot settings; saving one final distribution does not mean saving the distributions of all reductions.

The collection of ReductionHistory is set through collect.reduction_history. For the new curves, the record includes the weight of the selected edge before the reduction, the physical correlation, the selection score, the constraint sign, the defined energy contributions, the number of edges and the sum of weight moduli after the reduction, the constant and its increment. A full parameter trace for all steps requires a save frequency of 1 and all required layers. With less frequent saving, the plot explicitly shows the points actually saved.

When a plot is requested, Tools checks which data are present. If they were not collected, a clear message with the required setting is output. Saved traces after continuation from a checkpoint preserve identifiers and step order without duplicate points. The data required by the algorithm for continuation are still saved independently of the user visualization settings.

## Design of JSON configurations

The examples illustrate the structure. Type and field names are proposed for the new library; they are not an already existing API. The numerical values here are demonstrational.

Each document has schema_version. The component type is selected from the internal library catalog. At assembly, the parameters and the compatibility of computational capabilities are checked. A new implementation is added to the sources, the schema, and the internal catalog.

### Graph recipe

A graph of 12 vertices and 66 edges with independently specified randomness of topology and weights:

```json
{
  "schema_version": 1,
  "type": "gnm",
  "parameters": {
    "n_vertices": 12,
    "n_edges": 66
  },
  "seed": 1000,
  "weights": {
    "type": "signed_discrete",
    "parameters": {
      "values": [-3, -2, -1, 1, 2, 3]
    },
    "seed": 1001
  }
}
```

Unit weights are specified as weights with type = constant and value = 1.0. In the finished graph, all weights remain explicit.

### QAOA GPU configuration

```json
{
  "schema_version": 1,
  "type": "qaoa_gpu",
  "seed": 42,
  "num_treads": 1,
  "parameters": {
    "layers": 1,
    "device": 0,
    "dtype": "complex64"
  },
  "objective": {
    "type": "cvar",
    "parameters": {
      "alpha": 0.1
    }
  },
  "estimation": {
    "mode": "shots",
    "shots_per_evaluation": 512
  },
  "optimizer": {
    "type": "adam",
    "parameters": {
      "max_steps": 50,
      "learning_rate": 0.05
    },
    "gradient": {
      "type": "central_difference",
      "parameters": {
        "step": 0.01
      }
    }
  },
  "initialization": {
    "type": "fixed_angles",
    "parameters": {
      "values": [0.5, 0.3]
    }
  },
  "readout": {
    "mode": "best_of_shots",
    "shots": 2048
  },
  "limits": {
    "shots_total": 200000
  },
  "collect": {
    "optimizer_history": true,
    "statevector": false,
    "probabilities": false,
    "correlations": false
  },
  "output": {
    "report": 1,
    "logs": 0,
    "color": "auto"
  },
  "checkpoint": {
    "enabled": true,
    "every_optimizer_steps": 10
  }
}
```

For p layers, the angle order is proposed to be fixed as γ₀,…,γₚ₋₁, β₀,…,βₚ₋₁. Changing readout to most_probable does not change the selected training criterion. Changing estimation to exact does not change the method of obtaining the final solution.

Initialization, the optimization method, and the gradient computation method have their own parameters. The variants available in the archive are preserved with verification of support by the corresponding evaluator.

The output section describes only automatic output. Its report and logs can be overridden by arguments of solve() without changing collect. In the example, the optimization history is collected although logs are off; it can be obtained through solver.optimizer_history and solver.result and used for a convergence curve.

### Composite algorithm fragment

In the example, the correlation source is analytical. It shows that a composite method does not have to depend on the state vector of the local QAOA:

```json
{
  "schema_version": 1,
  "type": "ls_rqaoa_cpu",
  "seed": 42,
  "num_treads": 4,
  "parameters": {
    "residual_vertices": 8,
    "pair_domain": "all_pairs"
  },
  "components": {
    "subgraphs": {
      "type": "dense_cover",
      "parameters": {
        "max_vertices": 16,
        "max_subgraphs_per_step": 5
      }
    },
    "correlations": {
      "type": "analytical_p1_correlations_cpu",
      "parameters": {
        "gamma_samples": 1000,
        "gamma_bounds": [-3.141592653589793, 3.141592653589793]
      }
    },
    "correlation_update": {
      "type": "recompute_current_cover"
    },
    "pair_selection": {
      "type": "max_absolute_correlation"
    },
    "reduction": {
      "type": "spin_contraction"
    },
    "residual_solver": {
      "type": "exact_cpu",
      "parameters": {}
    }
  },
  "collect": {
    "reduction_history": true,
    "parameter_history": true
  },
  "checkpoint": {
    "enabled": true,
    "every_reduction_steps": 1
  }
}
```

The γ window in the example sets the search domain. For arbitrary real weights, the universal periodicity of the unit case cannot be transferred automatically.

The variant with measurements uses another registered correlation source and the common strict shot budget. Historical versions receive configurations that correspond to their actual rules, including weighting, combination, and updating of correlations.

### Assembly of objects in Python

```python
graph_recipe = load_json("graphs/small_12.json")
solver_config = load_json("solvers/qaoa_gpu.json")

graph = GraphFactory.from_recipe(graph_recipe)
solver = SolverFactory.from_config(solver_config)

solution = solver.solve(graph)
result = solver.result

graph.set_weight(2, 5, weight=-3.5)
modified_solution = solver.solve(graph)
modified_record = solver.last_run
```

The file names demonstrate the intended separation of configurations. Factories create objects, and graph methods track changes. The previous solution and record preserve their content.

## Tutorial notebooks

Detailed tutorial notebooks are part of the package and are a separate user-facing result of the implementation. They explain mathematical concepts and the interface design sequentially: the goal of the action, the required data, the code, the expected result, and its interpretation. Markdown explanations are written in English; comments and docstrings in the code follow ASD-STE100.

Two main files are mandatory:

| Notebook | Purpose and mandatory examples |
|---|---|
| notebooks/00_quickstart.ipynb | Selection of the CPU/NVIDIA environment, a small weighted graph, configuration and launch, solution/result/attributes, modification of the main settings, graph with solution, probabilities and progress, JSON and .npy, saving an image with metadata |
| notebooks/01_full_guide.ipynb | A sequential guide through all sections Solvers, Tools, Generators, and Laboratory, with mathematical explanations and executable examples of all public capabilities |

The main quick-start path works on CPU on a small reproducible problem. The GPU example is provided separately and explicitly checks the availability of the NVIDIA profile. The cells show the use of tools both with a solver object and with a saved result. Saving uses an explicitly selected new filename; repeated export with the same name demonstrates protection rather than deleting previous data automatically.

The full guide covers the object lifecycle and repeated solve(), graphs and components, solver families, criteria and readout, resources and num_treads, visualization and JSON styles, save and source formats, checkpoints, benchmarks, the parallelism profile, and research modifications. A detailed argument reference is available by links, but the guide is not replaced by a single list of links. Completeness is verified against the public API registry.

Thematic additions to the two main notebooks:

| Notebook | Content |
|---|---|
| 01. First problem | Weighted graph, configuration, exact CPU solver, spin dictionary, result, direct attributes, report and logs |
| 02. Graphs and generators | Recipes, seeds, weights, integer numbers with gaps, editing, subgraphs, and rendering a graph with a solution |
| 03. QAOA | State, angles, layers, normalization, training, two modes of the final solution, and the difference between CPU/GPU implementations |
| 04. Objective functions and resources | Mean Cut, CVaR, Gibbs, Best-of-K, Threshold Probability, exact and sampled estimates, strict shot budget |
| 05. Visualization | Common JSON style, graphs, histograms, optimization curves, landscape grids, heat maps, and self-contained HTML |
| 06. RQAOA | Spin constraints, weight changes, reduction constant, solution restoration, weights, and all pairwise angle trajectories |
| 07. LS-RQAOA | Component replacement, local problems, correlation origin, reduction energy curves, and the difference between their definitions |
| 08. Series and saving | Common runner, CPU/GPU in one series, references, comparison of results, JSON, and restoration from a checkpoint |
| 09. Laboratory | Reproduction of historical assemblies and an example of a small modification by replacing one component |

Each notebook uses small reproducible problems and runs top to bottom without hidden state from previous runs. The required dependencies, computational resource, approximate cost, and data collection settings are indicated. GPU examples are marked explicitly; a CPU example must not require an installed GPU stack. For the main scenarios, the expected invariants are included: completeness of the solution, its cut on the source graph, consistency of result, and compliance with the budget.

The tutorial results and figures are obtained from the cell code with the same JSON configurations that are shown to the reader. The notebooks are checked by execution in the provided environments after the corresponding modules are implemented. At the stage of this specification, their program is fixed; ready executable notebooks have not yet been created.

Training on saving includes the absence of side-effect writes without save, the choice of atts and save_original_data, the metadata caption on the plot, the difference between makedir and savedir, moving the folder of the set, and handling a conflict of existing names without losing the previous result.

Both main notebooks are checked by running from a clean kernel in the provided environment. The dependencies and availability of external resources are described at the beginning. The CPU path must not require importing CuPy. For expensive and GPU scenarios, the execution conditions are indicated; the absence of a device is not masked by running another implementation. The saved results of the cells must correspond to the delivered code and API version.

## Source migration

| Source family | Proposed direction |
|---|---|
| ClassicSolver, GoemansWilliamson, Dwave_aneal | Unified classical CPU solvers with a verified quality convention |
| QAOA.py and QAOA_GPU.py | Separate CPU/GPU evaluators and solvers; take into account the CPU permutation circuit and the NumPy Walsh implementation from the GPU file |
| AnalyticalQAOA and AnalyticalRQAOA | Analytical p=1 evaluator, correlation source, standalone recursive solver |
| RQAOA, LS_RQAOA, D_RQAOA, DLS_RQAOA | Composite methods with explicit state and configurable components |
| WS_QAOA, WS_RQAOA, Other/WS_* | Variants of state preparation, mixer, initialization, and recursive control |
| QAOA2 and Diplom/QAOA2_* | Hierarchical algorithm and components of information exchange between levels |
| Diplom/MK_*, Mk_* and Laboratory/LS_* | Cards of historical methods, separate components, and named assemblies |
| Laboratory/DistQAOA, Distributed_QAOA_CF, LS-DistQAOA, QAOA_F* | Variants with neighborhood, fields, and exchange of estimates |
| FineTuning | Improvement component and composite wrapper solver |
| Tools/MI_QAOA, MI_LS_RQAOA | Experimental algorithms; move common diagnostics into Tools |
| GraphDecomposer, Graph_partition, Graph_part_interception, Densest_sub, NodeExtender, EdgeExtender, GraphScanning | Generators for subgraphs/partitions; move evaluation and rendering into Tools |
| GraphGenerator, maxcut_benchmark_generator, ProblemGenerator, Hebbian_generator, WPE_generator | Generators of source problems and their metadata |
| PartitionEfficiency, MutInfo, diagnostic parts of Nonlocality_Gen | Tools |
| AutoTest, Diplom/Benchmark | Common ExperimentRunner |
| ENV, RQAOA_exp, verification and demonstration parts | Match with the main implementations; preserve substantive differences and move examples into separate scenarios |

Renaming does not define algorithmic identity. Before similar sources are merged, dependencies and behavior are checked. Errors in output, energy sign, normalization, and indexing are fixed as errors; substantive research variants are preserved.

For each historical variant, a correspondence record is needed: source file → new name → components → known fixes → check. A variant is considered migrated after its computations, solution output, budget, and resumption have been checked.

## Checks required during implementation

This is a verification plan for the future implementation, not a report of tests already performed.

- The cut weight is checked by independent direct summation over edges. Mixed signs, source numbers with gaps, and global spin inversion are taken into account.
- Reductions are checked by enumerating the remaining assignments on small graphs: the source cut under the constraint must equal the reduced cut plus the constant. Non-adjacent pairs and both constraint signs are included.
- The exact solver is compared with independent simple brute force on small problems.
- CPU/GPU evaluators and the analytical p=1 formula are compared at the same angles and a common Hamiltonian convention. The normalization of probabilities and the correlators are checked separately.
- Objective functions are checked on distributions with known answers: CVaR at alpha = 1 coincides with the mean, Best-of-K at K = 1 coincides with the mean, Threshold Probability correctly accounts for the boundary C = A.
- The sampler is checked on normalized simple distributions and on the restoration of global-inversion pairs.
- The strict budget is checked on nested calls, gradients, a common sample of several statistics, and the reserve of final measurements.
- The checkpoint is checked by comparing continued and continuous execution under identical conditions, including counters and randomness states.
- Solver reuse is checked on graph change, weight modification, and independence of previously saved solutions.
- For each historical variant, it is checked that the selected components reflect its substantive differences.
- Each solver module is checked by standalone launch on a small problem; ordinary import must not start computations. The obtained dictionary is checked for completeness and admissible spins.
- Visualization of distributions is checked on a small known distribution: normalization, correspondence of states to coordinates, mass of symmetric pairs, frequencies, and preservation of the unshown mass when K states are selected.
- The heat map and 3D surface are checked on one grid with known values and gaps to detect axis permutation and incorrect interpretation of the target quantity.
- Trajectories are checked on several reductions and local subgraphs, including reuse of estimates and continuation from a checkpoint: series, origin, and step numbers must be preserved.
- HTML is checked by opening without a network connection: the surface and trajectory are available for rotation, zoom, and tooltips. Re-rendering saved data must not call the solver or consume shots.
- The completeness of result is checked against every public result attribute, including additional Laboratory fields, nested histories, and arrays. The independence of an already issued snapshot from a new solve() is checked.
- JSON is checked by strict serialization and reading back: integer vertex numbers, NumPy/CuPy types, complex arrays, and unavailable quantities are restored according to the schema without loss of saved data.
- All nine combinations of report and logs are checked on a short run. A disabled channel does not print its text; a change of detail and color preserves the solution, computational counters, and randomness states under the same data collection configuration.
- Optimization curves are checked on a history with several objective evaluations per step, layer boundaries, and separate local tasks. The axes, improvement direction, and reference line must correspond to the selected quantity.
- For migrated algorithms, the explanations of mathematical transformations, the unified glossary, and the compliance of comments and docstrings with the adopted ASD-STE100 edition are checked.
- Graph rendering is checked on negative weights, numbers with gaps, isolated vertices, and a solution with a known cut. Filtering and the use of the snapshot of the required run must be explicitly reflected.
- Energy curves are checked with independent values for w, c, and σ. The probability range (1 + σc)/2, the difference between the physical correlator and the selection score, and the difference between the selected-edge contribution and the reduction constant are checked.
- For the mean absolute weight, the selected edge domain, the moment before/after reduction, and the empty domain are checked. For a non-existent selected pair, the weight is zero, but the reduction constant may change.
- For p = 1, 2, 3, the number of pairwise trajectories 1, 6, 15 is checked; in 3D the reduction number is always on the Z axis. Local series and missing parameters are not mixed.
- The unified LaTeX style is checked on static images and self-contained HTML, including Cyrillic, formulas, indices, numbers, and legends. The application of JSON parameters and the absence of global style changes at import are checked.
- Ready tutorial notebooks are checked by sequential execution with a clean state in the declared CPU or GPU environment.
- Without save, no export files or folders must appear. Independence from collect, report, and logs and explicitly enabled checkpoints are checked separately.
- Saving is checked for combinations of makedir and savedir with an existing and a missing path. The names of the main JSON, images, and source files must match the resolved plan.
- The independent lists atts and save_original_data, their intersection, empty atts, unknown names, and the absence of required data are checked. NumPy/CuPy arrays after reading .npy must preserve values, shape, and numeric type.
- The metadata caption is checked for completeness of context, date and time zone, readability, LaTeX style, and absence of clipping; with false it is absent. For HTML, self-contained opening is checked.
- The conflict of the main file, the conflict of one additional array, an existing folder of the set, and simultaneous saving of two runs under one name are checked. The contents of previously existing objects must remain unchanged.
- On a write failure, the explicit status of the incomplete set and the preservation of the scientific result in memory are checked. The selective JSON must explicitly describe the composition of the export and correctly link to additional files after the folder is moved.
- The CPU profile is checked in an environment without CuPy and CUDA; NVIDIA profiles are checked separately against the declared GPU/CUDA/driver matrix. A successful import does not replace the check of computations and kernel compilation.
- Execution of 00_quickstart.ipynb and 01_full_guide.ipynb from a clean kernel and the correspondence of the guide coverage to the public API are checked. The reference is built and checks the description of arguments and attributes.
- For LS and QAOA-in-QAOA, the independence of local executors, compliance with dependencies, the common CPU budget, the absence of multiplication of nested pools, and consistency of results with different num_treads are checked.
- The profiler and the benchmark must expand one set of admissible configurations at the selected seed. The full Cartesian product, conditional parameters, and the recording of all skips are checked.
- The computation of speedup, efficiency, and the Amdahl and Karp–Flatt coefficients is checked on data with known values. n = 1, poor fit, superlinearity, slowdown, and different amounts of actual work are checked.
- The global recommendation is checked on a set of configurations with different local optima: the selected number of threads must correspond to the minimum of the target curve of the whole set. A candidate with unfinished mandatory configurations is not considered a complete profile.
- For the concurrent benchmark, measurement of the full batch is checked, and for GPU, synchronization and separate designation of the applicability domain of the CPU model are checked.
- For the common theme, correspondence to the Another_pars.py profile, explicit font selection, locality of Seaborn settings, and identical series notation in different tools are checked. The invalid legend loc = "lower" value should not be transferred from the reference; the legend position is set by a correct specific position.

## What must be clarified during implementation

The main user requirements are fixed. The names of part of the API and directories, special sections of the JSON schemas, the implementation of serializers and checkpoint formats, planning details and overhead models, exact compatible dependency versions, and reference selection thresholds remain technical proposals. The common save fields, the name num_treads, the main JSON export, and the .npy format for source arrays are agreed. The specific installation files, reference pages, ready notebooks, and profiler are created and checked during implementation of the corresponding package parts; this version of the document fixes their composition and contracts.

The first verifiable implementation fragment is WeightedGraph, the configurations, the MaxCutSolver contract, RunRecord/RunState, the budget, and the exact CPU solver. It can be used to check numbering, the quality convention, run independence, and continuation before the quantum and experimental families are migrated.

## Full inventory of Python sources

The following list was generated automatically from the archive. It lists the source files and classes; the final semantic classification of historical variants is refined during migration. Three archive file names (Diplom/QAOA2_CE.py, Diplom/QAOA_CF.py, Diplom/QAOA_CFR.py) contain a Cyrillic letter that resembles C in their original spelling; the table gives the Latin spelling to keep this document English-only.

| File | Lines | Top-level classes |
|---|---:|---|
| Diplom/Bench_mod.py | 14 | mod |
| Diplom/Benchmark.py | 166 | Benchmark |
| Diplom/MK_1.py | 216 | LargeScaleRQAOA_mk1 |
| Diplom/MK_2.py | 244 | LargeScaleRQAOA_mk2 |
| Diplom/MK_2_1.py | 239 | LargeScaleRQAOA_mk2v1 |
| Diplom/MK_2_2.py | 239 | LargeScaleRQAOA_mk2v2 |
| Diplom/MK_3.py | 251 | LargeScaleRQAOA_mk3 |
| Diplom/MK_3_1.py | 248 | LargeScaleRQAOA_mk3 |
| Diplom/MK_6_1.py | 266 | LargeScaleRQAOA_mk6v1 |
| Diplom/MK_6_2.py | 280 | LargeScaleRQAOA_mk6v2 |
| Diplom/MK_7_1.py | 268 | LargeScaleRQAOA_mk7v1 |
| Diplom/MK_7_2.py | 280 | LargeScaleRQAOA_mk7v2 |
| Diplom/Mk_4.py | 277 | LargeScaleRQAOA_mk4 |
| Diplom/Mk_5.py | 344 | LargeScaleRQAOA |
| Diplom/Mk_5_1.py | 283 | LargeScaleRQAOA_mk5 |
| Diplom/Mk_5_2.py | 306 | LargeScaleRQAOA_mk5 |
| Diplom/Mk_5_3(8_3).py | 310 | LargeScaleRQAOA_mk5 |
| Diplom/Mk_8.py | 205 | LargeScaleRQAOA_mk5 |
| Diplom/Mk_8_1.py | 257 | LargeScaleRQAOA_mk8v1 |
| Diplom/Mk_8_2.py | 261 | LargeScaleRQAOA_mk8v1 |
| Diplom/Mk_8_3.py | 287 | LargeScaleRQAOA_mk8v1 |
| Diplom/QAOA2_CE_fixed.py | 451 | QAOAinQAOA |
| Diplom/QAOA2_Dip.py | 156 | QAOAinQAOA |
| Diplom/QAOA2_CE.py | 283 | QAOAinQAOA |
| Diplom/QAOA_DF.py | 508 | QAOA |
| Diplom/QAOA_F.py | 343 | QAOA |
| Diplom/QAOA_CF.py | 491 | QAOA |
| Diplom/QAOA_CFR.py | 495 | QAOA |
| ENV/GoemansWilliamson.py | 93 | GoemansWilliamsonMaxcut |
| ENV/ProblemGenerator.py | 118 | ProblemGenerator, HebbianMaxCutGenerator, WishartMaxCutGenerator |
| ENV/QAOA.py | 348 | QAOA |
| Generators/GraphGenerator.py | 224 | Functions / script |
| Generators/Nonlocality_Gen.py | 120 | Functions / script |
| Generators/maxcut_benchmark_generator.py | 368 | Functions / script |
| Laboratory/CorrelationExtractor.py | 353 | LargeScaleRQAOA |
| Laboratory/DistQAOA.py | 283 | LocalQAOA, DistributedQAOA |
| Laboratory/Distributed_QAOA_CF.py | 425 | QAOA_CF |
| Laboratory/EdgeExtender.py | 225 | Functions / script |
| Laboratory/GraphDecomposer_light.py | 759 | FastConfig, FastDenseSubgraphGenerator |
| Laboratory/GraphScanning.py | 100 | GraphScanner |
| Laboratory/LS-DistQAOA.py | 325 | LocalBlockQAOA, DistLSRQAOA |
| Laboratory/LS_QAOA_field.py | 256 | LargeScaleRQAOAF |
| Laboratory/LS_RQAOA_exp.py | 233 | LargeScaleRQAOA |
| Laboratory/LS_RQAOA_fast_1.py | 362 | LargeScaleRQAOA_1 |
| Laboratory/LS_RQAOA_fast_2.py | 342 | LargeScaleRQAOA_2 |
| Laboratory/LS_RQAOA_fast_3.py | 342 | LargeScaleRQAOA_3 |
| Laboratory/LS_RQAOA_fast_4.py | 334 | LargeScaleRQAOA_1 |
| Laboratory/LS_RQAOA_fast_5.py | 347 | LargeScaleRQAOA |
| Laboratory/NodeExtender.py | 198 | Functions / script |
| Laboratory/QAOA_ADAM.py | 521 | QAOA |
| Laboratory/QAOA_F.py | 349 | QAOA |
| Laboratory/QAOA_F2.py | 372 | QAOA |
| Laboratory/QAOA_F_M.py | 549 | QAOA |
| Laboratory/QAOA_F_M_D.py | 107 | DistQAOA |
| Laboratory/correct_mean.py | 347 | LargeScaleRQAOA_1 |
| Laboratory/test_partition.py | 40 | Functions / script |
| Other/WS_QAOA_correct.py | 428 | WSQAOA |
| Other/WS_QAOA_no_GW.py | 513 | QAOA |
| RQAOA_exp/QAOA.py | 491 | QAOA |
| RQAOA_exp/RQAOA.py | 325 | RecursiveQAOA |
| Solvers/AnalyticalQAOA.py | 255 | AnalyticalQAOA |
| Solvers/AnalyticalRQAOA.py | 447 | AnalyticalRQAOA |
| Solvers/AutoTest.py | 227 | Benchmark |
| Solvers/ClassicSolver.py | 58 | ClassicSolver |
| Solvers/DLS_RQAOA.py | 250 | DiscLargeScaleRQAOA |
| Solvers/D_QAOA.py | 299 | D_QAOA |
| Solvers/D_RQAOA.py | 210 | RecursiveQAOA |
| Solvers/Dwave_aneal.py | 111 | DwaveAnneal |
| Solvers/FT_test.py | 48 | Test |
| Solvers/FineTuning.py | 115 | FineTuning |
| Solvers/GoemansWilliamson.py | 112 | GoemansWilliamsonMaxcut |
| Solvers/LS_RQAOA.py | 283 | LargeScaleRQAOA |
| Solvers/QAOA.py | 470 | QAOA |
| Solvers/QAOA2.py | 192 | QAOAinQAOA |
| Solvers/QAOA_GPU.py | 730 | QAOA |
| Solvers/RQAOA.py | 326 | RecursiveQAOA |
| Solvers/WS_QAOA.py | 393 | WSQAOA |
| Solvers/WS_RQAOA.py | 287 | WSRecursiveQAOA |
| Tools/Densest_sub.py | 167 | Functions / script |
| Tools/GraphDecomposer.py | 335 | SubgraphConfig, GraphUtils, OptimizedDenseSubgraphGenerator |
| Tools/Graph_part_interception.py | 363 | LazyMaxHeap |
| Tools/Graph_partition.py | 171 | Functions / script |
| Tools/Hebbian_generator.py | 69 | MaxCut_definite_generator |
| Tools/MI_LS_RQAOA.py | 428 | MI_RQAOA |
| Tools/MI_QAOA.py | 529 | MutualInfoQAOA |
| Tools/MutInfo.py | 96 | MutualInformationMap |
| Tools/PartitionEfficiency.py | 253 | PartitionEfficiency |
| Tools/ProblemGenerator.py | 113 | ProblemGenerator, HebbianMaxCutGenerator, WishartMaxCutGenerator |
| Tools/WPE_generator.py | 46 | WishartMaxCutGenerator |
