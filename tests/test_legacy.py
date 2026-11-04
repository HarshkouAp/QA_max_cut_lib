"""R19 compatibility tests for the reviewed historical variants."""

from __future__ import annotations

import math

import numpy as np
import pytest

from QA_max_cut_lib import WeightedGraph
from QA_max_cut_lib.laboratory import legacy


def graph():
    return WeightedGraph(
        edges=[(10, 20, 1.0), (10, 30, 2.0), (20, 30, 3.0)],
        nodes=[10, 20, 30, 101],
    )


@pytest.mark.parametrize(
    "adapter",
    [
        legacy.LSRQAOAExp,
        legacy.LSRQAOAFast1,
        legacy.LSRQAOAFast2,
        legacy.LSRQAOAFast3,
        legacy.LSRQAOAFast4,
        legacy.MK6V1,
        legacy.MK6V2,
        legacy.MK7V1,
        legacy.MK7V2,
    ],
)
def test_laboratory_adapters_solve(adapter):
    solver = adapter()
    solution = solver.solve(graph())
    assert set(solution) == {10, 20, 30, 101}
    assert solver.result["status"] in ("completed", "stopped")


@pytest.mark.parametrize("adapter", [legacy.QAOAAdam, legacy.QAOAF2, legacy.QAOAFM])
def test_qaoa_adapters_solve(adapter):
    solver = adapter()
    solution = solver.solve(graph())
    assert set(solution) == {10, 20, 30, 101}
    assert solver.result["status"] == "completed"


def test_corrected_mean_map_and_correlations():
    states = np.array([[1, -1, 1], [-1, 1, 1]], dtype=float)
    means = legacy.CorrectedMeanMap([10, 20, 30])(states, [0.5, 0.5])
    assert means == {10: 0.0, 20: 0.0, 30: 1.0}
    corr = legacy.CorrelationExtractor([10, 20, 30]).from_samples(states)
    assert corr[(10, 20)] == -1.0
    vector = np.zeros(8, dtype=complex)
    vector[0] = 1.0
    result = legacy.CorrelationExtractor([10, 20, 30]).from_statevector(vector)
    assert all(math.isclose(value, 1.0) for value in result.values())


def test_edge_decomposer_scanner_and_partition():
    graph_value = graph()
    extended = legacy.EdgeExtender(3).around(graph_value, (10, 20))
    assert set(extended.nodes) == {10, 20, 30}
    blocks = legacy.GraphDecomposerLight(3, seed=17).decompose(graph_value)
    assert sum(len(block.nodes) for block in blocks) >= 4
    scan = legacy.GraphScanner(3, seed=17).scan(graph_value)
    assert scan["cut"] >= 0
    report = legacy.partition_smoke_test(graph_value, 3)
    assert report["complete"] is True
    assert set(report["covered"]) == {10, 20, 30, 101}


def test_source_alias_manifest_covers_all_r19_paths():
    assert len(legacy.SOURCE_ALIASES) == 18
    assert set(legacy.SOURCE_ALIASES) == {
        "Diplom/MK_6_1.py",
        "Diplom/MK_6_2.py",
        "Diplom/MK_7_1.py",
        "Diplom/MK_7_2.py",
        "Laboratory/correct_mean.py",
        "Laboratory/CorrelationExtractor.py",
        "Laboratory/EdgeExtender.py",
        "Laboratory/GraphDecomposer_light.py",
        "Laboratory/GraphScanning.py",
        "Laboratory/LS_RQAOA_exp.py",
        "Laboratory/LS_RQAOA_fast_1.py",
        "Laboratory/LS_RQAOA_fast_2.py",
        "Laboratory/LS_RQAOA_fast_3.py",
        "Laboratory/LS_RQAOA_fast_4.py",
        "Laboratory/QAOA_ADAM.py",
        "Laboratory/QAOA_F2.py",
        "Laboratory/QAOA_F_M.py",
        "Laboratory/test_partition.py",
    }
