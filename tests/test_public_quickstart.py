"""Guard the published quickstart numbers."""

from math import isclose

from examples.public_quickstart import compare


def test_public_quickstart_numbers():
    graph, solution, mean_cut, mass_by_cut = compare()
    assert graph.cut_value(solution) == 3.5
    assert isclose(mean_cut, 3.044365, abs_tol=5e-7)
    assert isclose(mass_by_cut[3.5], 0.805755, abs_tol=5e-7)
