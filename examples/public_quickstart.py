"""Compare exact enumeration and QAOA on a signed weighted graph."""

import argparse
from collections import defaultdict
from math import isclose
from pathlib import Path

from QA_max_cut_lib import ExactCPU, QAOACPU, WeightedGraph, __version__


def compare():
    graph = WeightedGraph(
        edges=[(10, 30, 1.5), (10, 80, 2.0), (30, 80, -0.5)],
        nodes=[10, 30, 80, 101],
    )
    exact = ExactCPU()
    exact.solve(graph)

    qaoa = QAOACPU({
        "seed": 42,
        "parameters": {"layers": 1},
        "objective": {"type": "mean_cut"},
        "estimation": {"mode": "exact"},
        "initialization": {
            "type": "fixed_angles",
            "parameters": {"values": [-0.8, 0.4]},
        },
        "optimizer": {"parameters": {"max_steps": 24}},
        "readout": {"mode": "best_of_shots", "shots": 128},
        "limits": {"shots_total": 128},
        "collect": {"probabilities": True},
    })
    solution = qaoa.solve(graph)

    # Index k uses bit j for the node at position j in node_order.
    mass_by_cut = defaultdict(float)
    for index, probability in enumerate(qaoa.probabilities):
        spins = {
            node: 1 - 2 * ((index >> position) & 1)
            for position, node in enumerate(qaoa.node_order)
        }
        mass_by_cut[graph.cut_value(spins)] += float(probability)

    assert isclose(sum(mass_by_cut.values()), 1.0, abs_tol=1e-12)
    reconstructed_mean = sum(cut * mass for cut, mass in mass_by_cut.items())
    assert isclose(reconstructed_mean, qaoa.expected_cut, abs_tol=1e-12)
    assert exact.cut_value == 3.5 and exact.optimal_count == 4
    assert graph.cut_value(solution) == qaoa.cut_value <= exact.cut_value
    assert qaoa.resources["shots_used"] == 128

    print(f"QA_max_cut_lib {__version__}")
    print(f"Exact optimum                    {exact.cut_value:.6f}")
    print(f"Expected QAOA cut               {qaoa.expected_cut:.6f}")
    print(f"Best cut in 128 simulated shots {qaoa.cut_value:.6f}")
    print(f"Final-state optimal mass        {mass_by_cut[exact.cut_value]:.6f}")
    print(f"Readout spins                   {solution}")
    return graph, solution, qaoa.expected_cut, dict(sorted(mass_by_cut.items()))


def save_figure(path, graph, solution, mean_cut, mass_by_cut):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    text = "#14283c"
    muted = "#526479"
    teal = "#007e76"
    orange = "#bb500d"
    positions = {10: (0.14, 0.63), 30: (0.74, 0.87), 80: (0.74, 0.30), 101: (0.20, 0.20)}
    weight_labels = {(10, 30): (0.39, 0.81), (10, 80): (0.39, 0.39), (30, 80): (0.85, 0.59)}

    with plt.rc_context({"font.family": "DejaVu Sans", "text.color": text, "svg.fonttype": "path"}):
        fig, (network, histogram) = plt.subplots(
            1, 2, figsize=(10.8, 4.0), gridspec_kw={"width_ratios": [1, 1.28]},
        )
        fig.patch.set_facecolor("#f4f8fb")
        fig.subplots_adjust(left=0.035, right=0.955, top=0.71, bottom=0.18, wspace=0.24)
        fig.text(0.052, 0.91, "Weighted Max-Cut on four vertices", fontsize=19, weight="bold")
        fig.text(0.052, 0.83, "One negative edge, one isolated vertex, depth-one QAOA", fontsize=10.5, color=muted)

        for u, v, weight in graph.edges:
            x0, y0 = positions[u]
            x1, y1 = positions[v]
            network.plot(
                [x0, x1], [y0, y1], color=teal if weight > 0 else orange,
                linestyle="-" if weight > 0 else "--", linewidth=2.2, zorder=1,
            )
            x, y = weight_labels[(u, v)]
            network.text(x, y, f"{weight:+g}", fontsize=10, ha="center", va="center", color=muted)

        for node, (x, y) in positions.items():
            network.scatter(x, y, s=660, color=teal if solution[node] == 1 else text, zorder=2)
            network.text(x, y, str(node), color="white", ha="center", va="center", fontsize=10, zorder=3)
        network.set(xlim=(-0.02, 1), ylim=(0.02, 1))
        network.set_axis_off()
        network.set_title("Partition returned by readout", loc="left", fontsize=12, weight="bold", pad=10)
        network.text(0.05, -0.04, "Node colors show the two sides of the cut", fontsize=9, color=muted)

        cuts = list(mass_by_cut)
        histogram.set_facecolor("#f4f8fb")
        histogram.bar(cuts, mass_by_cut.values(), width=0.18,
                      color=[teal if cut == max(cuts) else "#819cb3" for cut in cuts], zorder=2)
        histogram.axvline(mean_cut, linestyle="--", color=text, linewidth=1.35, zorder=3)
        histogram.text(mean_cut - 0.08, 0.91, f"Mean {mean_cut:.3f}", ha="right", fontsize=9.5,
                       transform=histogram.get_xaxis_transform(),
                       bbox={"facecolor": "#f4f8fb", "edgecolor": "none", "pad": 1})
        histogram.set(xlim=(-0.28, 3.79), ylim=(0, 1.05), xticks=cuts, xlabel="Cut value")
        histogram.set_yticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
        histogram.tick_params(length=0, colors=muted)
        histogram.grid(axis="y", color="#dbe6ef", zorder=0)
        for spine in histogram.spines.values():
            spine.set_visible(False)
        histogram.set_title("Exact final-state probabilities", loc="left", fontsize=12, weight="bold", pad=10)

        fig.savefig(path, dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, help="Save the graph and probability chart.")
    args = parser.parse_args()
    result = compare()
    if args.figure is not None:
        save_figure(args.figure, *result)
