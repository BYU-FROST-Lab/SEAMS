from error_evaluations import *
# from error_evaluations import _print_result

import re
import copy
import pandas as pd
import gtsam
import common.fgHelpers as fgtools
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from natsort import natsort_keygen

S = gtsam.Symbol
DEBUG = False
LOC2MAP = True

FULL_WIDTH = 5.0
SINGLE_HEIGHT = 3.5
DOUBLE_HEIGHT = 5.2
TRIPLE_HEIGHT = 6.5

pgf_with_rc_fonts = {
    # Match paper typography (IEEE uses Times / Times New Roman)
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    
    # Enable LaTeX math rendering
    "text.usetex": False,  # Set to True if you have full TeX distribution installed
    "pdf.fonttype": 42,    # Embed TrueType fonts so text renders as scalable vectors
    "ps.fonttype": 42,

    # Font sizes relative to IEEE standard body text (10 pt) / figure captions (8 pt)
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,

    # Line weights and spacing
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    
    # Tight margins so axes labels aren't clipped
    "figure.autolayout": True,
}

plt.rcParams.update(pgf_with_rc_fonts)

@dataclass
class BoundaryResults(ErrorResultBase):
    """Information for boundary nodes for a session."""

    est_name: str
    boundaryNode: int
    detMargCovBoundaryNode: float
    numConstrainNodes: int
    numInterSessionLoopClosures: int

    # For plotting
    graph: gtsam.NonlinearFactorGraph
    values: gtsam.Values
    isam: gtsam.ISAM2
    transitionNodes: list[int]



    def stats_row(self) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            "estimate": self.est_name,
            # "boundaryNode": S(self.boundaryNode).string(),
            "detMargCovBoundaryNode": self.detMargCovBoundaryNode,
            "numConstrainNodes": self.numConstrainNodes,
            "numInterSessionLoopClosures": self.numInterSessionLoopClosures
        }
        return row


def _path_to_parent_parent(file: str | Path) -> str:
    return str(Path(file).parents[2].name)

def _find_dataset_from_path(file: str | Path) -> str:
    """Return the dataset name from a path, e.g. 'nc' from '/path/to/nc/mission/est.g2o'."""
    path = Path(file)
    for parent in path.parents:
        if parent.name in ["nc", "urban", "nclt"]:
            return parent.name
    raise ValueError(f"Could not find dataset name in path: {file}")


def _to_final_table(evo_table: pd.DataFrame, boundary_table: pd.DataFrame):
    # Merge the two tables based on the 'estimate' column
    print(evo_table.head())
    print(boundary_table.head())
    evo_table = evo_table[['estimate', 'n_associated', 'ape_rmse']].copy()
    evo_table.rename(columns={'ape_rmse': 'rmse'}, inplace=True)
    merged_table = pd.merge(evo_table, boundary_table, on='estimate', how='outer')

    merged_table['estimate'] = merged_table['estimate'].apply(_path_to_parent_parent)
    return merged_table.sort_values(by='estimate', key=natsort_keygen())

def get_cov_det(cov: np.ndarray):
    return np.linalg.det(cov)

def get_boundary_info_from_session(est_files: list[Path], isLoc2Map: bool) -> list[BoundaryResults]:
    results = []
    for est_file in est_files:
        # Assume session information is in same directory as estimate file
        sessionDir = est_file.parent
        graph = fgtools.deserializeGraphFromFile(sessionDir / "graph.gtsam")
        values = fgtools.deserializeValuesFromFile(sessionDir / "values.gtsam")
        isam = gtsam.ISAM2()
        isam.update(graph, values)

        transitionNodesFile = Path(sessionDir / "boundaryKeysForPlotting.txt")
        if transitionNodesFile.is_file():
            transitionKeys = [int(key) for key in transitionNodesFile.read_text().splitlines()]
            if isLoc2Map:
                boundaryNode = max(transitionKeys) + 1
            else:
                boundaryNode = min(transitionKeys) - 1

        else:
            transitionKeys = []
            keys = list(values.keys())
            lastSessionChr = S(keys[-1]).chr()
            for k in keys:
                if S(k).chr() == lastSessionChr:
                    boundaryNode = k
                    break

        if est_file.parents[1].name == "block1__block0":
            boundaryNode = S('B', 156).key() # All are the same for urban block1
            print(f"Using mode change key '{S(boundaryNode).string()}' for mission 'block1__block0'")
        elif est_file.parents[1].name == "block2__block0":
            boundaryNode = S('B', 194).key() # All are the same for urban block2
            print(f"Using mode change key '{S(boundaryNode).string()}' for mission 'block2__block0'")
        else:
            print(f"Mode change key: {S(boundaryNode).string()}")

        margCov = isam.marginalCovariance(boundaryNode)
        detMargCov = get_cov_det(margCov)

        numInterSessionLoopClosures = 0
        for i in range(graph.size()):
            f = graph.at(i)
            if f is None or isinstance(f, gtsam.LinearContainerFactor):
                continue
            symbolChrs = set([S(key).chr() for key in f.keys()])
            if len(symbolChrs) > 1:
                numInterSessionLoopClosures += 1

        if isLoc2Map:
            numInterSessionLoopClosures -= 1
            if numInterSessionLoopClosures < 0:
                numInterSessionLoopClosures = 0

        results.append(BoundaryResults(
            est_name=str(est_file),
            boundaryNode=boundaryNode,
            detMargCovBoundaryNode=detMargCov,
            numConstrainNodes=len(transitionKeys),
            graph=graph,
            values=values,
            isam=isam,
            transitionNodes=transitionKeys,
            numInterSessionLoopClosures=numInterSessionLoopClosures,
        ))

    return results

def plot_boundary_info(boundary_results: list[BoundaryResults], saveFilePath: Path | None, isLoc2Map: bool):
    rc = int(np.ceil(np.sqrt(len(boundary_results))))
    print(f"Number of results: {len(boundary_results)}")
    print(f"Number of subplots: {rc} x {rc}")
    fig, axes = plt.subplots(rc, rc, figsize=(FULL_WIDTH, DOUBLE_HEIGHT))
    axes = axes.flatten() if rc > 1 else [axes]

    if not isLoc2Map:
        boundary_results = [boundary_results[0]] + list(reversed(boundary_results[1:]))

    for i in range(len(boundary_results)):
        result = boundary_results[i]
        est_file = result.est_name
        graph = result.graph
        values = result.values
        isam = result.isam
        boundaryNode = result.boundaryNode
        transitionKeys = result.transitionNodes

        ax = axes[i]
        fgtools.plot_graph(graph, values, ax, title="", changeColorBySymbols=True, addKeyLabels=False,
                           symbolColors={ord('A'): 'tab:orange', ord('B'): 'tab:green'},
                           symbolLabels={ord('A'): 'Prior', ord('B'): 'Current'}, skipUnaryFactors=True)
        fgtools.plot_points(transitionKeys, values, "tab:red", ax=ax)
        fgtools.plot_points([boundaryNode], values, "tab:purple", ax=ax)
        # fgtools.plot_graph_covariances([boundaryNode], isam, values, ax=ax, edgecolor="tab:purple")

        ax.legend().remove()
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)
        splitName = _path_to_parent_parent(est_file).split("_")
        if splitName is not None and len(splitName) > 1:
            title = splitName[-1]
            if isLoc2Map:
                ax.set_title(fr"$\tau_{{L2M}}={float(title) * .01 if splitName[1] != "inf" else "inf"}$")
            else:
                ax.set_title(fr"$\tau_{{M2L}}=-{float(title) * .01 if splitName[1] != "inf" else "inf"}$")

            if isLoc2Map and Path(est_file).parents[1].name == "block1__block0":
                ax.set_xlim(-170, -60)
                ax.set_ylim(-200, -90)
            elif isLoc2Map and Path(est_file).parents[1].name == "park__quad-easy":
                ax.set_xlim(10, 50)
                ax.set_ylim(-30, 10)
            elif not isLoc2Map and Path(est_file).parents[1].name == "block2__block0":
                ax.set_xlim(-160, -70)
                ax.set_ylim(-45, 44)

            # Color border of plots
            for spine in ax.spines.values():
                spine.set_color("blue")
                spine.set_linewidth(1.5)
        else:
            ax.set_title("Full SLAM")
            currentSessionValues = [values.atPose3(k) for k in values.keys() if chr(S(k).chr()) == 'B']
            ax.scatter(currentSessionValues[0].x(), currentSessionValues[0].y(), color="gold", marker='*', label="Start")
            ax.legend(loc="lower center")

    # Make a rectangle in the first subplot showing where the other subplots are zoomed in.
    if len(boundary_results) > 1:
        other_axes = axes[1:len(boundary_results)]
        x_limits = [limit for ax in other_axes for limit in ax.get_xlim()]
        y_limits = [limit for ax in other_axes for limit in ax.get_ylim()]
        x_min, x_max = min(x_limits), max(x_limits)
        y_min, y_max = min(y_limits), max(y_limits)
        axes[0].add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor="blue",
                linewidth=1.5,
            )
        )

    fig.supxlabel("x (m)", y=0.08)
    # fig.supxlabel("x (m)", y = 0.5)
    fig.supylabel("y (m)")

    # Build one legend for the whole figure so it stays outside the subplots.
    # Matplotlib fills multi-column legends column-by-column, so this order
    # renders Prior, Current, and Loop closure on the first row.
    nodeLegendSize = 7
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Prior Session",
               markerfacecolor="tab:orange", markersize=nodeLegendSize),
        Line2D([0], [0], marker="o", color="w", label="Transition Nodes",
               markerfacecolor="tab:red", markersize=nodeLegendSize),
        Line2D([0], [0], marker="o", color="w", label="Current Session",
               markerfacecolor="tab:green", markersize=nodeLegendSize),
        Line2D([0], [0], marker="o", color="w", label="Boundary Node",
               markerfacecolor="tab:purple", markersize=nodeLegendSize),
        # Represent loop-closure factors with a yellow line and centered square.
        Line2D([0, 0.5, 1], [0, 0, 0], color="tab:olive", marker="s",
               markevery=[1], markerfacecolor="tab:olive",
               markeredgecolor="tab:olive", linewidth=1.2, markersize=3,
               label="Loop closure factor"),
        # Represent marginal factors with a purple line and centered square.
        Line2D([0, 0.5, 1], [0, 0, 0], color="tab:purple", marker="s",
                markevery=[1], markerfacecolor="tab:purple",
                markeredgecolor="tab:purple", linewidth=1.2, markersize=3,
                label="Marginal factor"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.), ncol=3, frameon=False)
            #    bbox_to_anchor=(0.5, 0.04), ncol=4, frameon=False)

    # Reserve space at the bottom for the figure-level legend and x-axis label.
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    # fig.tight_layout()

    if saveFilePath is not None:
        transitionTypeString = "Loc2Map" if isLoc2Map else "Map2Loc"
        if saveFilePath.is_dir():
            print(f"Saving {transitionTypeString} Boundary plot to {saveFilePath / f'{transitionTypeString} Boundary.pdf'}")
            plt.savefig(saveFilePath / f"{transitionTypeString} Boundary - {_find_dataset_from_path(est_file)}.pdf", format="pdf", dpi=300)
        else:
            print(f"Error saving {transitionTypeString} Boundary plot: {saveFilePath} is not a directory")



def _get_beta(est_name: str, isLoc2Map: bool) -> float | None:
    """Return beta threshold for a `Loc2Map_<N>` or `Map2Loc_<N>` run (N / 100), or None if not a thresholded Loc2Map or Map2Loc run."""
    if isLoc2Map:
        match = re.fullmatch(r"Loc2Map_(\d+)", _path_to_parent_parent(est_name))
        sign = 1
    else:
        match = re.fullmatch(r"Map2Loc_(\d+)", _path_to_parent_parent(est_name))
        sign = -1
    return sign * int(match.group(1)) / 100 if match else None


def plot_threshold_sensitivity(evo_table: pd.DataFrame, boundary_results: list[BoundaryResults],
                               saveFilePath: Path | None, isLoc2Map: bool):
    """Three subplots vs. beta threshold: (a) RMSE, (b) transition nodes retained,
    (c) determinant of the boundary node's marginal covariance."""
    rmse_by_est = dict(zip(evo_table["estimate"], evo_table["ape_rmse"]))

    points = []
    for result in boundary_results:
        beta = _get_beta(result.est_name, isLoc2Map)
        if beta is None or result.est_name not in rmse_by_est:
            continue
        points.append((beta, rmse_by_est[result.est_name], result.numConstrainNodes,
                       result.detMargCovBoundaryNode))
    points.sort()

    if not points:
        print(f"No {"Loc2Map" if isLoc2Map else "Map2Loc"}_<N> runs found; skipping threshold sensitivity plot.")
        return

    betas, rmses, numNodes, dets = zip(*points)
    print(f" betas: {betas}")

    fig, (ax_err, ax_det, ax_nodes) = plt.subplots(1, 3, figsize=(FULL_WIDTH * 1.5, SINGLE_HEIGHT*0.8))

    ax_err.plot(betas, rmses, marker="o", color="tab:blue")
    ax_err.set_xlabel(r"$\tau_{L2M}$" if isLoc2Map else r"$\tau_{M2L}$")
    ax_err.set_ylabel("ATE RMSE (m)")
    ax_err.set_title("Accuracy")
    ax_err.grid(True)

    ax_det.plot(betas, dets, marker="^", color="tab:purple")
    ax_det.set_xlabel(r"$\tau_{L2M}$" if isLoc2Map else r"$\tau_{M2L}$")
    ax_det.set_ylabel(r"$\det(\Sigma_{\mathrm{boundary}})$")
    ax_det.set_title("Uncertainty")
    ax_det.set_yscale("log")
    ax_det.grid(True)

    ax_nodes.plot(betas, numNodes, marker="s", color="tab:red")
    ax_nodes.set_xlabel(r"$\tau_{L2M}$" if isLoc2Map else r"$\tau_{M2L}$")
    ax_nodes.set_ylabel("Transition Nodes Retained")
    ax_nodes.set_title("Graph Cost")
    ax_nodes.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_nodes.grid(True)

    fig.tight_layout()

    if saveFilePath is not None:
        transitionTypeString = "Loc2Map" if isLoc2Map else "Map2Loc"
        if saveFilePath.is_dir():
            outFile = saveFilePath / f"{transitionTypeString} Threshold Sensitivity - {_find_dataset_from_path(boundary_results[0].est_name)}.pdf"
            print(f"Saving threshold sensitivity plot to {outFile}")
            fig.savefig(outFile, format="pdf", dpi=300)
        else:
            print(f"Error saving threshold sensitivity plot: {saveFilePath} is not a directory")



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    plane = Plane(args.project_to_plane) if args.project_to_plane else None
    est_files = resolve_estimate_files(args.est)
    if not args.silent:
        print(SEP)
        print(f"Estimate files ({len(est_files)}):")
        for path in est_files:
            print(f"  {path}")

    isLoc2Map = (_path_to_parent_parent(est_files[0])[:7] == "Loc2Map")

    evo_results = evaluate(
        args.ref,
        est_files,
        merge_refs=True if args.merge_refs else None,
        compute_ape="ape" in args.metrics,
        compute_rpe="rpe" in args.metrics,
        pose_relation=POSE_RELATION_MAP[args.pose_relation],
        align=args.align,
        align_origin=args.align_origin,
        correct_scale=args.correct_scale,
        n_to_align=args.n_to_align,
        t_max_diff=args.t_max_diff,
        t_offset=args.t_offset,
        t_start=args.t_start,
        t_end=args.t_end,
        # t_start=1778705028. if not isLoc2Map and est_files[0].parents[1].name == "block2__block0" else args.t_start,
        # t_end=1778705028. if not isLoc2Map and est_files[0].parents[1].name == "block2__block0" else args.t_end,
        rpe_delta=args.delta,
        rpe_delta_unit=DELTA_UNIT_MAP[args.delta_unit],
        rpe_delta_tol=args.delta_tol,
        rpe_all_pairs=args.all_pairs,
        project_to_plane=plane,
    )

    # if not args.silent:
    #     for pair in evo_results:
    #         _print_result(pair)

    evo_table = results_to_table(evo_results)

    boundary_results = get_boundary_info_from_session(est_files, isLoc2Map)

    boundary_table = results_to_table(boundary_results)

    table = _to_final_table(evo_table, boundary_table)
    print(SEP)
    print("Summary: Terminal")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2e}" if x < 1e-3 else f"{x:.2f}"))

    print(SEP)
    print("Summary: For LaTex")
    print(
        "\n".join(
            " & ".join(f"{x:.2e}" if isinstance(x, float) and x < 1e-3 
                       else f"{x:.2f}" if isinstance(x, float) 
                       else str(x) for x in row) + " \\\\"
            # " & ".join(str(x) for x in row)
            for row in table.to_numpy()
        )
    )

    if args.save_table:
        table_path = Path(args.save_table)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_path, index=False)
        print(f"Wrote summary table to {table_path}")

    if args.save_results:
        out_dir = Path(args.save_results)
        for pair in evo_results:
            save_pair_results(pair, out_dir)
        print(f"Wrote evo result zips to {out_dir}")

    if args.plot or args.save_plot:
        plot_boundary_info(boundary_results, Path(args.save_plot) if args.save_plot else None, isLoc2Map)
        plot_threshold_sensitivity(evo_table, boundary_results, Path(args.save_plot) if args.save_plot else None, isLoc2Map)
        # plot_results(
        #     evo_results,
        #     plot_mode=args.plot_mode,
        #     show=args.plot,
        #     save_path=args.save_plot,
        # )
        # if args.save_plot:
        #     print(f"Wrote plots to {args.save_plot}")
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
