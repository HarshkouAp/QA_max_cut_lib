QA_max_cut_lib 0.1.0rc2
========================

Weighted Max-Cut on small and medium graphs: exact enumeration, QAOA, RQAOA,
local search, classical baselines, generators, and reproducible experiments.

This documentation is organized as a reader path, not as a stage log. Start with
the quickstart, then learn the conventions, then choose a method. Reference pages
contain exact types and defaults; historical acceptance records are kept in the
archive.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   QUICKSTART
   CONCEPTS
   CHOOSE_ALGORITHM

.. toctree::
   :maxdepth: 2
   :caption: Scientific model

   weighted_max_cut
   api
   ALL_DOCS

Guides
------

* :download:`Documentation authoring rules <DOCUMENTATION_STYLE.md>`
* :download:`API reference: fields, types, and exceptions <API.md>`
* :download:`Mathematical conventions and numerical limits <MATHEMATICS.md>`
* :download:`Installation for CPU and NVIDIA profiles <INSTALLATION.md>`
* :download:`Configuration and solver lifecycle <../CONFIGURATION.md>`
* :download:`Saving and restoring results <../RESULTS_AND_SAVE.md>`
* :download:`Troubleshooting <../TROUBLESHOOTING.md>`
* :download:`Checkpoint and resume <CHECKPOINTS.md>`
* :download:`Experiment series and parallelism <EXPERIMENTS.md>`
* :download:`NVIDIA GPU QAOA <GPU.md>`
* :download:`Warm-start RQAOA <WARM_START_RQAOA.md>`
* :download:`Laboratory catalog and warm start <LABORATORY.md>`
* :download:`Laboratory controllers <LABORATORY_CONTROLLERS.md>`
* :download:`Conditional fields and environment LS <CONDITIONAL.md>`
* :download:`Block mean-field QAOA <BLOCK_MEAN_FIELD.md>`
* :download:`Distributed LS-RQAOA <DISTRIBUTED_LS.md>`
* :download:`Mutual-information diagnostics <MUTUAL_INFORMATION.md>`
* :download:`Incremental LS-RQAOA <INCREMENTAL_LS.md>`
* :download:`Generators and typed plans <GENERATORS_PLANS.md>`
* :download:`Architecture map <ARCHITECTURE_MAP.md>`
* :download:`Migration status <MIGRATION_STATUS.md>`
* :download:`Project environment <PROJECT_SETUP.md>`
* :download:`Public release checklist <PUBLISHING.md>`
* :download:`Release readiness and R19 <../RELEASE_READINESS.md>`
* :download:`Source closure inventory <../SOURCE_CLOSURE.json>`
* :download:`Acceptance evidence archive <ARCHIVE.md>`

Notebooks
---------

Executable notebooks are in the repository's ``notebooks/`` directory. HTML
previews are in ``docs/notebook_previews/``. The quickstart and full guide are
the primary entry points; specialized notebooks cover reductions, RQAOA,
analytic p=1, LS-RQAOA, GPU, QAOA-in-QAOA, classical baselines, experiments,
checkpoints, Laboratory, mutual information, incremental LS, and generators.

Energy convention
-----------------

The library reports

.. math::

   E(s)=-C(s)=\sum_{(i,j)\in E}\frac{w_{ij}}2(s_i s_j-1).

Lower energy means a better cut. The constant term is retained. The optimization
criterion and the physical expectation can differ for CVaR, Gibbs, best-of-K,
and threshold-probability objectives.
