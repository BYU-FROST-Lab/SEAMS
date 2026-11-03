"""Compare SLAM factor graphs and Hessian sparsity after marginalization."""

import argparse
import pickle
from pathlib import Path

import gtsam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import common.fgHelpers as fgtools


S = gtsam.Symbol
workspaceDir = Path(__file__).parents[1]
fullSLAMDir = workspaceDir / "results/marginalizationEvaluation/nc/fullSLAM__quad-medium__quad-easy"
denseMarginalDir = workspaceDir / "results/marginalizationEvaluation/nc/dense__quad-medium__quad-easy"
sparseMarginalDir = workspaceDir / "results/marginalizationEvaluation/nc/approx__quad-medium__quad-easy"

# Keep the figure sized for a two-column paper and use embedded vector fonts.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 0.8,
})

MAX_LOOP_DISTANCE_METERS = 20.0
METHODS = [
    ("Full SLAM", fullSLAMDir),
    ("Dense Marg.", denseMarginalDir),
    ("Sparse Marg.", sparseMarginalDir),
]


def _find_graph_directory(base_dir: Path) -> Path:
    """Resolve a run directory or its base to the folder containing serialized data."""
    if (base_dir / "graph.gtsam").is_file() and (base_dir / "values.gtsam").is_file():
        return base_dir

    candidates = sorted(
        directory
        for directory in base_dir.glob("*")
        if (directory / "graph.gtsam").is_file() and (directory / "values.gtsam").is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No graph.gtsam and values.gtsam found under {base_dir}")
    raise RuntimeError(f"Multiple serialized graph runs found under {base_dir}: {candidates}")


def _remove_long_same_session_loops(
    graph: gtsam.NonlinearFactorGraph,
    values: gtsam.Values,
    max_distance: float = MAX_LOOP_DISTANCE_METERS,
) -> tuple[gtsam.NonlinearFactorGraph, int]:
    """Drop nonconsecutive same-session pose constraints longer than the distance limit."""
    remove_indices: list[int] = []
    for index in range(graph.size()):
        factor = graph.at(index)
        if not isinstance(factor, gtsam.BetweenFactorPose3):
            continue

        key1, key2 = factor.keys()[0], factor.keys()[1]
        symbol1, symbol2 = S(key1), S(key2)
        # Consecutive poses form the odometry chain, not a loop closure.
        if symbol1.chr() != symbol2.chr() or abs(symbol1.index() - symbol2.index()) <= 1:
            continue

        pose1, pose2 = values.atPose3(key1), values.atPose3(key2)
        distance = np.linalg.norm(pose1.translation() - pose2.translation())
        if distance > max_distance:
            remove_indices.append(index)

    # Remove backwards so factor indices remain valid while the graph is edited.
    for index in reversed(remove_indices):
        graph.remove(index)
    return graph, len(remove_indices)


def _load_graph_and_values(base_dir: Path) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values, int]:
    """Load a saved run and apply the same-session loop-closure distance filter."""
    data_dir = _find_graph_directory(base_dir)
    graph = fgtools.deserializeGraphFromFile(data_dir / "graph.gtsam")
    values = fgtools.deserializeValuesFromFile(data_dir / "values.gtsam")
    graph, removed_count = _remove_long_same_session_loops(graph, values)
    print(f"Loaded {data_dir}; removed {removed_count} same-session closures over {MAX_LOOP_DISTANCE_METERS:g} m")
    return graph, values, removed_count


def _load_optimization_times(data_dir: Path) -> np.ndarray:
    """Load per-update optimization durations from seconds and return milliseconds."""
    timing_path = data_dir / "timeToOptimize.txt"
    if not timing_path.is_file():
        raise FileNotFoundError(f"No optimization timing data found at {timing_path}")

    times_seconds = np.atleast_1d(np.loadtxt(timing_path, dtype=float))
    if times_seconds.size == 0 or not np.all(np.isfinite(times_seconds)):
        raise ValueError(f"Optimization timing data is empty or invalid: {timing_path}")
    return times_seconds * 1e3


def create_figure(save_path: Path | None = None):
    """Build and optionally save the three-method graph/Hessian comparison."""
    # Wide panels keep the pose trajectories legible beside the sparsity plots.
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.2), constrained_layout=False)

    for row_index, (row_title, data_dir) in enumerate(METHODS):
        graph, values, _ = _load_graph_and_values(data_dir)
        graph_ax, hessian_ax = axes[row_index]

        # Color pose symbols consistently across methods; helper colors distinguish
        # sequential, loop-closure, and marginalized linear-container factors.
        fgtools.plot_graph(
            graph,
            values,
            ax=graph_ax,
            title="",
            changeColorBySymbols=True,
            symbolColors={ord("A"): "tab:orange", ord("B"): "tab:green"},
            symbolLabels={ord("A"): "Prior Session", ord("B"): "Current Session"},
            addKeyLabels=False,
            skipUnaryFactors=False,
        )
        # graph_ax.set_title("Factor graph", pad=4)
        graph_ax.legend().remove()
        graph_ax.set_xlabel("x (m)")
        graph_ax.set_ylabel("y (m)")
        graph_ax.grid(False)
        # graph_ax.grid(color="0.88", linestyle="--", linewidth=0.45)
        # graph_ax.tick_params(direction="out", length=2.5, width=0.6)

        # Linearize each nonlinear graph at its saved estimate and plot its Hessian.
        fgtools.plot_hessian(graph, values, ax=hessian_ax)
        # hessian_ax.set_title("Hessian sparsity", pad=4)
        hessian_ax.set_xlabel("Variable dimension")
        hessian_ax.set_ylabel("Variable dimension")
        # hessian_ax.tick_params(direction="out", length=2.5, width=0.6)

        # Row labels keep method names clear without repeating them in each panel.
        graph_ax.text(
            -0.2,
            0.5,
            row_title,
            rotation=90,
            va="center",
            ha="center",
            transform=graph_ax.transAxes,
            fontsize=9,
            fontweight="bold",
        )

        if row_index == 0:
            # graph_ax.set_title("Factor graph", pad=4, fontweight="bold")
            # hessian_ax.set_title("Hessian sparsity", pad=4, fontweight="bold")
            graph_ax.text(
                        0.5,
                        1.15,
                        "Factor Graph",
                        va="center",
                        ha="center",
                        transform=graph_ax.transAxes,
                        fontsize=9,
                        fontweight="bold",
                    )
            hessian_ax.text(
                        0.5,
                        1.15,
                        "Information Sparsity",
                        va="center",
                        ha="center",
                        transform=hessian_ax.transAxes,
                        fontsize=9,
                        fontweight="bold",
                    )

    # Keep one compact legend for the shared factor-graph encodings.
    handles = [
        Line2D([], [], marker="o", linestyle="", color="tab:orange", label="Prior Session"),
        Line2D([], [], marker="o", linestyle="", color="tab:green", label="Current Session"),
        Line2D([0, 0.5, 1], [0, 0, 0], color="tab:orange", marker="s",
            markevery=[1], markerfacecolor="tab:orange",
            markeredgecolor="tab:orange", linewidth=1.1, markersize=3,
            label="Odometry Factor"),
        Line2D([0, 0.5, 1], [0, 0, 0], color="tab:olive", marker="s",
            markevery=[1], markerfacecolor="tab:olive",
            markeredgecolor="tab:olive", linewidth=1.1, markersize=3,
            label="Loop Closure"),
        Line2D([0, 0.5, 1], [0, 0, 0], color="tab:purple", marker="s",
            markevery=[1], markerfacecolor="tab:purple",
            markeredgecolor="tab:purple", linewidth=1.1, markersize=3,
            label="Marginal Factor"),
    ]
    
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.))
    fig.tight_layout(rect=(0.04, 0.03, 1, 1))
    # fig.subplots_adjust(left=0.13, right=0.99, top=0.98, bottom=0.09, hspace=0.34, wspace=0.24)

    if save_path is not None:
        fig.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return fig, axes

def create_table(save_path: Path | None = None):
    """Build a LaTeX-ready summary of graph size, optimization time, and Hessian sparsity."""
    rows = []
    for method, base_dir in METHODS:
        # Use the same resolved saved run and filtered graph as the comparison figure.
        data_dir = _find_graph_directory(base_dir)
        graph, values, _ = _load_graph_and_values(base_dir)
        hessian = graph.linearize(values).hessian()[0]
        total_entries = int(np.prod(hessian.shape))
        nonzero_entries = int(np.count_nonzero(abs(hessian) > 1e-9))
        zero_entries = total_entries - nonzero_entries
        density_percent = 100.0 * nonzero_entries / total_entries if total_entries else 0.0

        # The saved times are seconds; convert them to milliseconds for the table.
        optimization_times_ms = _load_optimization_times(data_dir)
        rows.append({
            "Method": method,
            "Pose nodes": len(values.keys()),
            "Optimization time (ms)": (
                f"{np.mean(optimization_times_ms):.2f} $\\pm$ "
                f"{np.std(optimization_times_ms):.2f}"
            ),
            "Hessian density (\\%)": f"{density_percent:.2f}",
        })

    table = pd.DataFrame(rows)

    # Print the table to the console for quick inspection.
    print(table)
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_latex(save_path, index=False, escape=False, column_format="lccc")
        print(f"Saved LaTeX table to {save_path}")
    return table


def plot_optimization_times(save_path: Path | None = None):
    """Plot every saved optimization duration in one separate comparison figure."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for method, base_dir in METHODS:
        # Find the same run folder used for graph loading so timings stay aligned.
        data_dir = _find_graph_directory(base_dir)
        times_ms = _load_optimization_times(data_dir)
        ax.plot(np.arange(1, len(times_ms) + 1), times_ms, linewidth=1.0, label=method)

    ax.set_title("Optimization time per update")
    ax.set_xlabel("Update")
    ax.set_ylabel("Optimization time (ms)")
    ax.grid(True, color="0.88", linestyle="--", linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        print(f"Saved optimization-time plot to {save_path}")
    return fig, ax


def plot_covariance_determinants(save_path: Path | None = None):
    """Plot log determinants of current-pose covariance, translation, and rotation."""
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.0), sharex=True)
    covariance_blocks = [
        ("Full 6D covariance", None),
        ("Translation covariance", slice(3, 6)),
        ("Rotation covariance", slice(0, 3)),
    ]

    for method, base_dir in METHODS:
        # MultiSessionRun writes this pickle in the configured method directory.
        covariance_path = base_dir / "cov_of_currentPoses.pkl"
        if not covariance_path.is_file():
            raise FileNotFoundError(f"No current-pose covariance data found at {covariance_path}")
        with covariance_path.open("rb") as covariance_file:
            covariances = np.asarray(pickle.load(covariance_file), dtype=float)

        if covariances.ndim != 3 or covariances.shape[1:] != (6, 6):
            raise ValueError(
                f"Expected a sequence of 6x6 covariance matrices in {covariance_path}; "
                f"found shape {covariances.shape}"
            )

        for ax, (title, block) in zip(axes, covariance_blocks):
            # slogdet avoids overflow/underflow; covariance ordering is rotation then translation.
            block_covariances = covariances if block is None else covariances[:, block, block]
            signs, log_determinants = np.linalg.slogdet(block_covariances)
            log_determinants = np.where(signs > 0, log_determinants, np.nan)
            ax.plot(np.arange(1, len(log_determinants) + 1), log_determinants, label=method)
            ax.set_title(title, loc="left")
            ax.grid(True, color="0.88", linestyle="--", linewidth=0.5)

    axes[0].legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Current-pose update")
    fig.supylabel("Log determinant")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        print(f"Saved covariance-determinant plot to {save_path}")
    return fig, axes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-times",
        action="store_true",
        help="Also save a separate plot containing all optimization-time series.",
    )
    args = parser.parse_args()

    # output_dir = Path(__file__).parent
    output_dir = Path().home() / "Pictures/MastersThesis"
    # create_figure(output_dir / "dense_marginal_plot.png")
    # create_table(output_dir / "dense_marginal_table.tex")
    # plot_covariance_determinants(output_dir / "dense_marginal_covariance_determinants.png")
    # plot_optimization_times(output_dir / "dense_marginal_optimization_times.png")
    create_figure()
    create_table()
    plot_covariance_determinants()
    plot_optimization_times()
    # if args.plot_times:
    plt.show()
