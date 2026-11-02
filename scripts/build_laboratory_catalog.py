"""Build the versioned variant catalog from reviewed source evidence."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "QA_max_cut_lib" / "laboratory" / "data"
ARCHIVE_SHA = "b4c684a5d4da469adc25bd99a405e402892d1a2524465e4c97ab6229c71a4aed"


def member(path, role, sha, note=None):
    item = {"path": path, "sha256": sha, "role": role}
    if note:
        item["note"] = note
    return item


variants = [
    {
        "schema_version": 1,
        "variant_id": "warm_start_product_qaoa_cpu_v1",
        "display_name": "Warm-start product QAOA (CPU, full state)",
        "family": "warm_start_qaoa",
        "source_archive_sha256": ARCHIVE_SHA,
        "source_members": [
            member(
                "Other/WS_QAOA_no_GW.py",
                "explicit_start_product_state_and_mixers",
                "71d41c7f33bb3ba3fe8bc2b6e7dc8c336f7d9eb2ca37b29ce21a6246dbf2f877",
            ),
            member(
                "Other/WS_QAOA_correct.py",
                "classical_start_and_mixer_comparison",
                "ccfa1802dff370b0df830a0cebec40fabab5dfea043db3b5bac7a2a33e22e148",
            ),
        ],
        "role": "standalone_solver",
        "algorithm_signature": {
            "initial_state": "clipped_probability_product",
            "cost_convention": "energy_equals_negative_cut",
            "mixer_options": ["warm_aligned", "rounded_modified", "x"],
            "parameter_order": "gammas_then_betas",
            "preprocessing": "explicit_source",
            "controller": "fixed_depth_shared_adam",
            "bit_order": "little_endian_node_order",
            "regularization": "clip_q_to_epsilon_and_one_minus_epsilon",
        },
        "status": {
            "source_review": "partially_reviewed",
            "scientific_specification": "specified",
            "implementation": "implemented",
            "cpu_validation": "passed",
            "gpu_validation": "not_supported",
            "resume_validation": "passed_on_declared_boundaries",
        },
        "historical_relation": "corrected_ansatz_port",
        "declared_differences": [
            "explicit_node_index_and_spin_bit_mapping",
            "shared_adam_instead_of_legacy_layerwise_lbfgsb",
            "checked_gw_factorization_when_gw_source_is_selected",
            "no_implicit_exact_search_for_starting_solution",
            "only_supplied_data_and_classical_cpu_sources_in_a11",
        ],
        "backlog": [],
    },
    {
        "schema_version": 1,
        "variant_id": "warm_start_ensemble_rqaoa_cpu_v1",
        "display_name": "Warm-start ensemble RQAOA (CPU, full state)",
        "family": "warm_start_rqaoa",
        "source_archive_sha256": ARCHIVE_SHA,
        "source_members": [
            member(
                "Solvers/WS_RQAOA.py",
                "historical_recursive_source",
                "e02a7beb0c39807eb382dae5a0f8ea4cd47bd8c9b015625282dcbff9f3c9f4b9",
            ),
            member(
                "Solvers/WS_QAOA.py",
                "historical_warm_ansatz_source",
                "7beaba04c7f37e9bcc0393273c4ff8ec2f292ed27ecf295aae14d78497a5c2e5",
            ),
        ],
        "role": "standalone_solver",
        "algorithm_signature": {
            "initial_state": "ensemble_of_clipped_product_states",
            "candidate_source": ["gw_rounds_cpu", "provided_by_graph"],
            "mixer_options": ["warm_aligned", "rounded_modified", "x"],
            "aggregation": "uniform_components",
            "controller": "shared_recursive_rqaoa",
            "bit_order": "little_endian_node_order",
        },
        "status": {
            "source_review": "partially_reviewed",
            "scientific_specification": "specified",
            "implementation": "implemented",
            "cpu_validation": "passed",
            "gpu_validation": "not_supported",
            "resume_validation": "not_supported",
        },
        "historical_relation": "component_recomposition",
        "declared_differences": [
            "explicit_node_index_and_spin_bit_mapping",
            "shared_adam_instead_of_legacy_layerwise_lbfgsb",
            "bounded_gw_rounding_batch_and_global_flip_dedup",
            "multiple_independent_warm_children_with_equal_weights",
            "graph_reducer_and_exact_finisher_contract",
            "one_root_shot_budget_instead_of_per_child_limits",
        ],
        "backlog": ["Resume codec is not claimed in a12."],
    },
    {
        "schema_version": 1,
        "variant_id": "legacy_ls_rqaoa_fast_alias",
        "display_name": "Legacy fast LS-RQAOA alias pair (source only)",
        "family": "laboratory_ls_rqaoa",
        "source_archive_sha256": ARCHIVE_SHA,
        "source_members": [
            member("Diplom/Mk_5.py", "source_alias_primary", "34a3dd3ba5346583456d281646a5272cad7506dae09bbb0bb0f90948b780fe71"),
            member("Laboratory/LS_RQAOA_fast_5.py", "source_alias_secondary", "91006f32df8dc04e705e834551a0219f37ae1134253d58816dae2ce3d69c2b5c"),
        ],
        "role": "duplicate_alias",
        "algorithm_signature": None,
        "status": {
            "source_review": "partially_reviewed",
            "scientific_specification": "hypothesis",
            "implementation": "not_implemented",
            "cpu_validation": "not_run",
            "gpu_validation": "not_supported",
            "resume_validation": "not_supported",
        },
        "historical_relation": "source_alias",
        "equivalence": {
            "kind": "provisional_ast_equivalence",
            "evidence": "ast.dump(..., include_attributes=False) is identical; text diff reports three commented main-scenario lines",
            "limitations": "import resolution and runtime behaviour were not executed; not a trajectory equivalence proof",
        },
        "declared_differences": ["three_commented_main_scenario_lines"],
        "backlog": ["Resolve imports", "Define one component recipe", "Validate a CPU trajectory before porting."],
    },
    {
        "schema_version": 1,
        "variant_id": "warm_start_rqaoa",
        "display_name": "Warm-start recursive QAOA",
        "family": "warm_start_rqaoa",
        "source_archive_sha256": ARCHIVE_SHA,
        "source_members": [
            member("Solvers/WS_RQAOA.py", "legacy_recursive_source", "e02a7beb0c39807eb382dae5a0f8ea4cd47bd8c9b015625282dcbff9f3c9f4b9"),
        ],
        "role": "deferred_family",
        "algorithm_signature": None,
        "status": {
            "source_review": "inventoried",
            "scientific_specification": "hypothesis",
            "implementation": "not_implemented",
            "cpu_validation": "not_run",
            "gpu_validation": "not_supported",
            "resume_validation": "not_supported",
        },
        "historical_relation": "distinct_strategy",
        "declared_differences": ["global_graph_constructor_argument", "hardcoded_parameters", "post_filter_n_nodes_change", "uniform_init_after_filtering"],
        "backlog": [
            "Define start-source policy on the current reduced graph.",
            "Decide how multiple preparations and correlations are aggregated.",
            "Share one root budget across recursive steps.",
            "Do not inherit the old WS_RQAOA constructor.",
        ],
    },
    {
        "schema_version": 1,
        "variant_id": "laboratory_ls_families",
        "display_name": "Laboratory LS and distributed QAOA families",
        "family": "laboratory_ls",
        "source_archive_sha256": ARCHIVE_SHA,
        "source_members": [
            member("Laboratory/", "directory_family", "directory"),
        ],
        "role": "deferred_family",
        "algorithm_signature": None,
        "status": {
            "source_review": "inventoried",
            "scientific_specification": "hypothesis",
            "implementation": "not_implemented",
            "cpu_validation": "not_run",
            "gpu_validation": "not_supported",
            "resume_validation": "not_supported",
        },
        "historical_relation": "distinct_strategy",
        "declared_differences": ["multiple_graph_builders", "field_and_environment_modes", "intentional_stale_correlations"],
        "backlog": ["Review each Laboratory file", "Separate components from demos", "Define graph/field semantics before porting."],
    },
    {
        "schema_version": 1,
        "variant_id": "mutual_information_diagnostics",
        "display_name": "Mutual-information and selector diagnostics",
        "family": "diagnostics",
        "source_archive_sha256": ARCHIVE_SHA,
        "source_members": [
            member("Tools/MI_LS_RQAOA.py", "diagnostic", "8ada89f744624b40add553b2f125186026194795ac4f395e4c43d81fdefc6685"),
            member("Tools/MI_QAOA.py", "diagnostic", "e3eee87591c2a8b3131c022c5e4d5eff1868d95adb7f7814198e04ce266ea170"),
            member("Tools/MutInfo.py", "diagnostic", "c0ab49ef8538e7bb4b5fe0523dc6c26d0511826074efd84185b031d8e668950a"),
        ],
        "role": "diagnostic_tool",
        "algorithm_signature": None,
        "status": {
            "source_review": "inventoried",
            "scientific_specification": "hypothesis",
            "implementation": "not_implemented",
            "cpu_validation": "not_run",
            "gpu_validation": "not_supported",
            "resume_validation": "not_supported",
        },
        "historical_relation": "distinct_strategy",
        "declared_differences": ["diagnostic_not_solver"],
        "backlog": ["Separate diagnostics from selectors", "Do not expose a diagnostic as a solver."],
    },
    {
        "schema_version": 1,
        "variant_id": "deferred_environment_and_generators",
        "display_name": "Environment, generators and legacy solver sources",
        "family": "environment_and_generators",
        "source_archive_sha256": ARCHIVE_SHA,
        "source_members": [
            member("ENV/", "environment_family", "directory"),
            member("Generators/", "generator_family", "directory"),
            member("Solvers/", "legacy_solver_family", "directory"),
            member("RQAOA_exp/", "research_family", "directory"),
            member("Diplom/", "historical_family", "directory"),
            member("Tools/", "tools_family", "directory"),
        ],
        "role": "deferred_family",
        "algorithm_signature": None,
        "status": {
            "source_review": "inventoried",
            "scientific_specification": "hypothesis",
            "implementation": "not_implemented",
            "cpu_validation": "not_run",
            "gpu_validation": "not_supported",
            "resume_validation": "not_supported",
        },
        "historical_relation": "unresolved",
        "declared_differences": ["not_yet_audited"],
        "backlog": ["Review by family", "Keep unresolved files visible", "Do not claim transfer from folder presence."],
    },
]

inventory = json.loads((DATA / "inventory.json").read_text(encoding="utf-8"))

catalog = {
    "schema_version": 1,
    "archive_sha256": ARCHIVE_SHA,
    "generated_from": "scripts/build_laboratory_catalog.py",
    "variants": variants,
    "implemented_variant_ids": ["warm_start_product_qaoa_cpu_v1", "warm_start_ensemble_rqaoa_cpu_v1"],
    "deferred_variant_ids": [
        item["variant_id"]
        for item in variants
        if item["variant_id"] not in ("warm_start_product_qaoa_cpu_v1", "warm_start_ensemble_rqaoa_cpu_v1")
    ],
    "source_to_variant": {
        record["path"]: record["variant_id"] for record in inventory["files"]
    },
}

DATA.mkdir(parents=True, exist_ok=True)
(DATA / "catalog.json").write_text(
    json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print("wrote", DATA / "catalog.json", "variants", len(variants))
