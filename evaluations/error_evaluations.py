#!/usr/bin/env python3
"""Evaluate TUM-format trajectories against ground truth with evo.

Computes Absolute Pose Error (APE) and Relative Pose Error (RPE) for one or
more estimated trajectories given one or more TUM reference files.

Pairing rules:
    * one reference, N estimates  -> each estimate vs that reference
    * N references, N estimates   -> pair by index
    * N references, one estimate  -> merge references by timestamp (multi-session GT)

Example:
    python -m evaluations.error_evaluations \\
        --ref /path/to/gt-nc-quad-easy.csv \\
        --est results/nc/quad-medium__quad-easy \\
        --align --plot
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from evo.core import metrics, sync, trajectory
from evo.core.result import Result
from evo.core.trajectory import Plane, PoseTrajectory3D
from evo.main_ape import ape as evo_ape
from evo.main_rpe import rpe as evo_rpe
from evo.tools import file_interface, log as evo_log

logger = logging.getLogger(__name__)

SEP = "-" * 80

POSE_RELATION_MAP: dict[str, metrics.PoseRelation] = {
    "full": metrics.PoseRelation.full_transformation,
    "trans_part": metrics.PoseRelation.translation_part,
    "rot_part": metrics.PoseRelation.rotation_part,
    "angle_deg": metrics.PoseRelation.rotation_angle_deg,
    "angle_rad": metrics.PoseRelation.rotation_angle_rad,
    "point_distance": metrics.PoseRelation.point_distance,
    "point_distance_error_ratio": metrics.PoseRelation.point_distance_error_ratio,
}

DELTA_UNIT_MAP: dict[str, metrics.Unit] = {
    "f": metrics.Unit.frames,
    "d": metrics.Unit.degrees,
    "r": metrics.Unit.radians,
    "m": metrics.Unit.meters,
}


@dataclass
class PairEvaluation:
    """APE/RPE results for one reference–estimate pair."""

    ref_name: str
    est_name: str
    n_associated: int
    ape: Result | None = None
    rpe: Result | None = None
    traj_ref: PoseTrajectory3D | None = field(default=None, repr=False)
    traj_est: PoseTrajectory3D | None = field(default=None, repr=False)

    def stats_row(self) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            "reference": self.ref_name,
            "estimate": self.est_name,
            "n_associated": self.n_associated,
        }
        if self.ape is not None:
            for key, val in self.ape.stats.items():
                row[f"ape_{key}"] = val
        if self.rpe is not None:
            for key, val in self.rpe.stats.items():
                row[f"rpe_{key}"] = val
        return row


def load_tum_trajectory(path: str | Path) -> PoseTrajectory3D:
    """Load a TUM trajectory (`timestamp tx ty tz qx qy qz qw`).

    Falls back to comma-separated rows if space-delimited parsing fails.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    try:
        return file_interface.read_tum_trajectory_file(str(path))
    except file_interface.FileInterfaceException:
        raw = file_interface.csv_read_matrix(str(path), delim=",", comment_str="#")
        if not raw or len(raw[0]) != 8:
            raise
        try:
            mat = np.array(raw, dtype=float)
        except ValueError as exc:
            raise file_interface.FileInterfaceException(
                f"Could not parse TUM file {path}"
            ) from exc
        stamps = mat[:, 0]
        xyz = mat[:, 1:4]
        quat_wxyz = np.roll(mat[:, 4:], 1, axis=1)
        logger.debug("Loaded %d poses from comma-delimited TUM file %s", len(stamps), path)
        return PoseTrajectory3D(xyz, quat_wxyz, stamps)


def merge_references(trajs: list[PoseTrajectory3D]) -> PoseTrajectory3D:
    """Merge multiple timestamped trajectories into one, sorted by time."""
    if len(trajs) == 1:
        return trajs[0]
    return trajectory.merge(trajs)


def unique_path_labels(paths: list[Path]) -> list[str]:
    """Return labels that distinguish paths even when filenames collide."""
    labels = [p.name for p in paths]
    if len(set(labels)) == len(labels):
        return labels
    labels = [f"{p.parent.name}/{p.name}" for p in paths]
    if len(set(labels)) == len(labels):
        return labels
    return [str(p) for p in paths]


TUM_ESTIMATE_NAME = "odom_tum.txt"


def resolve_estimate_files(est_args: list[str | Path]) -> list[Path]:
    """Expand estimate arguments into TUM files.

    A path may be a TUM file or a directory. Directories are searched for
    ``odom_tum.txt`` in the directory itself and in subfolders (e.g. policy
    folders such as ``INLIER_RATIO/odom_tum.txt``).
    """
    found: list[Path] = []
    seen: set[Path] = set()

    for raw in est_args:
        path = Path(raw)
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            direct = path / TUM_ESTIMATE_NAME
            nested = sorted(
                p for p in path.glob(f"*/{TUM_ESTIMATE_NAME}") if p.is_file()
            )
            candidates = ([direct] if direct.is_file() else []) + nested
            if not candidates:
                candidates = sorted(p for p in path.rglob(TUM_ESTIMATE_NAME) if p.is_file())
            if not candidates:
                raise FileNotFoundError(
                    f"No {TUM_ESTIMATE_NAME} files found under directory {path}"
                )
        else:
            raise FileNotFoundError(f"Estimate path not found: {path}")

        for candidate in candidates:
            key = candidate.resolve()
            if key in seen:
                continue
            seen.add(key)
            found.append(candidate)

    if not found:
        raise FileNotFoundError("No estimate TUM files found.")
    return found


def _associate(
    traj_ref: PoseTrajectory3D,
    traj_est: PoseTrajectory3D,
    t_max_diff: float,
    t_offset: float,
    ref_name: str,
    est_name: str,
    t_start: float | None = None,
    t_end: float | None = None,
) -> tuple[PoseTrajectory3D, PoseTrajectory3D]:
    traj_ref = copy.deepcopy(traj_ref)
    traj_est = copy.deepcopy(traj_est)
    if t_start is not None or t_end is not None:
        traj_ref.reduce_to_time_range(t_start, t_end)
    traj_ref, traj_est = sync.associate_trajectories(
        traj_ref,
        traj_est,
        max_diff=t_max_diff,
        offset_2=t_offset,
        first_name=ref_name,
        snd_name=est_name,
    )
    return traj_ref, traj_est


def evaluate_pair(
    traj_ref: PoseTrajectory3D,
    traj_est: PoseTrajectory3D,
    ref_name: str,
    est_name: str,
    *,
    compute_ape: bool = True,
    compute_rpe: bool = True,
    pose_relation: metrics.PoseRelation = metrics.PoseRelation.translation_part,
    align: bool = False,
    align_origin: bool = False,
    correct_scale: bool = False,
    n_to_align: int = -1,
    t_max_diff: float = 0.1,
    t_offset: float = 0.0,
    t_start: float | None = None,
    t_end: float | None = None,
    rpe_delta: float = 1.0,
    rpe_delta_unit: metrics.Unit = metrics.Unit.frames,
    rpe_delta_tol: float = 0.1,
    rpe_all_pairs: bool = False,
    project_to_plane: Plane | None = None,
) -> PairEvaluation:
    """Associate timestamps and compute APE and/or RPE for one pair."""
    traj_ref, traj_est = _associate(
        traj_ref,
        traj_est,
        t_max_diff=t_max_diff,
        t_offset=t_offset,
        ref_name=ref_name,
        est_name=est_name,
        t_start=t_start,
        t_end=t_end,
    )

    result = PairEvaluation(
        ref_name=ref_name,
        est_name=est_name,
        n_associated=traj_est.num_poses,
        traj_ref=traj_ref,
        traj_est=traj_est,
    )

    if compute_ape:
        result.ape = evo_ape(
            traj_ref=copy.deepcopy(traj_ref),
            traj_est=copy.deepcopy(traj_est),
            pose_relation=pose_relation,
            align=align,
            correct_scale=correct_scale,
            n_to_align=n_to_align,
            align_origin=align_origin,
            ref_name=ref_name,
            est_name=est_name,
            project_to_plane=project_to_plane,
        )

    if compute_rpe:
        result.rpe = evo_rpe(
            traj_ref=copy.deepcopy(traj_ref),
            traj_est=copy.deepcopy(traj_est),
            pose_relation=pose_relation,
            delta=rpe_delta,
            delta_unit=rpe_delta_unit,
            rel_delta_tol=rpe_delta_tol,
            all_pairs=rpe_all_pairs,
            align=align,
            correct_scale=correct_scale,
            n_to_align=n_to_align,
            align_origin=align_origin,
            ref_name=ref_name,
            est_name=est_name,
            support_loop=True,
            project_to_plane=project_to_plane,
        )

    return result


def evaluate(
    ref_files: list[str | Path],
    est_files: list[str | Path],
    *,
    merge_refs: bool | None = None,
    **kwargs,
) -> list[PairEvaluation]:
    """Evaluate one or more TUM estimates against one or more TUM references.

    ``est_files`` may be TUM files or session directories; directories are
    expanded to every ``odom_tum.txt`` found in policy subfolders.
    """
    ref_paths = [Path(p) for p in ref_files]
    est_paths = resolve_estimate_files(est_files)
    if not ref_paths or not est_paths:
        raise ValueError("At least one --ref and one --est file are required.")

    auto_merge = len(ref_paths) > 1 and len(est_paths) == 1
    do_merge = auto_merge if merge_refs is None else merge_refs
    est_labels = unique_path_labels(est_paths)
    ref_labels = unique_path_labels(ref_paths)

    results: list[PairEvaluation] = []
    if do_merge:
        merged_ref = merge_references([load_tum_trajectory(p) for p in ref_paths])
        ref_label = "+".join(p.name for p in ref_paths)
        for est_path, est_label in zip(est_paths, est_labels):
            logger.info("%s\nEvaluating %s  vs  %s", SEP, ref_label, est_label)
            results.append(
                evaluate_pair(
                    merged_ref,
                    load_tum_trajectory(est_path),
                    ref_label,
                    est_label,
                    **kwargs,
                )
            )
        return results

    if len(ref_paths) == 1:
        traj_ref = load_tum_trajectory(ref_paths[0])
        ref_label = ref_labels[0]
        for est_path, est_label in zip(est_paths, est_labels):
            logger.info("%s\nEvaluating %s  vs  %s", SEP, ref_label, est_label)
            results.append(
                evaluate_pair(
                    traj_ref,
                    load_tum_trajectory(est_path),
                    ref_label,
                    est_label,
                    **kwargs,
                )
            )
        return results

    if len(ref_paths) != len(est_paths):
        raise ValueError(
            f"Got {len(ref_paths)} reference(s) and {len(est_paths)} estimate(s). "
            "Provide equal counts, a single reference, or pass --merge-refs."
        )

    for ref_path, est_path, ref_label, est_label in zip(
        ref_paths, est_paths, ref_labels, est_labels
    ):
        logger.info("%s\nEvaluating %s  vs  %s", SEP, ref_label, est_label)
        results.append(
            evaluate_pair(
                load_tum_trajectory(ref_path),
                load_tum_trajectory(est_path),
                ref_label,
                est_label,
                **kwargs,
            )
        )
    return results


def results_to_table(results: list[PairEvaluation]) -> pd.DataFrame:
    return pd.DataFrame([r.stats_row() for r in results])


def _print_result(pair: PairEvaluation) -> None:
    print(SEP)
    print(f"reference : {pair.ref_name}")
    print(f"estimate  : {pair.est_name}")
    print(f"associated poses: {pair.n_associated}")
    if pair.ape is not None:
        print()
        print(pair.ape.pretty_str())
    if pair.rpe is not None:
        print()
        print(pair.rpe.pretty_str())


def plot_results(
    results: list[PairEvaluation],
    plot_mode: str = "xy",
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """Plot error time series and trajectories, following evo's PlotCollection style."""
    from evo.tools import plot
    import matplotlib.pyplot as plt

    plot_collection = plot.PlotCollection("Trajectory error evaluation")
    mode = plot.PlotMode(plot_mode)

    fig_traj = plt.figure(figsize=(8, 8))
    ax_traj = plot.prepare_axis(fig_traj, mode)
    plotted_refs: set[str] = set()
    colormap_used = False

    for i, pair in enumerate(results):
        if pair.ape is not None:
            traj_ref = pair.ape.trajectories[pair.ref_name]
            traj_est = pair.ape.trajectories[pair.est_name]
            error = pair.ape.np_arrays["error_array"]
            stats = pair.ape.stats
            label = str(pair.ape.info.get("label", "APE"))
            title = str(pair.ape.info.get("title", "APE"))
        elif pair.rpe is not None:
            traj_ref = pair.rpe.trajectories[pair.ref_name]
            traj_est = pair.rpe.trajectories[pair.est_name]
            error = pair.rpe.np_arrays["error_array"]
            stats = pair.rpe.stats
            label = str(pair.rpe.info.get("label", "RPE"))
            title = str(pair.rpe.info.get("title", "RPE"))
        else:
            continue

        if pair.ref_name not in plotted_refs:
            plot.traj(ax_traj, mode, traj_ref, "--", "gray", f"ref: {pair.ref_name}")
            plotted_refs.add(pair.ref_name)

        fig_err = plt.figure(figsize=(8, 4))
        seconds = None
        if pair.ape is not None and "seconds_from_start" in pair.ape.np_arrays:
            seconds = pair.ape.np_arrays["seconds_from_start"]
        elif pair.rpe is not None and "seconds_from_start" in pair.rpe.np_arrays:
            seconds = pair.rpe.np_arrays["seconds_from_start"]
        plot.error_array(
            fig_err.gca(),
            error,
            x_array=seconds,
            statistics={k: v for k, v in stats.items() if k not in ("min", "max", "sse")},
            name=label,
            title=f"{pair.est_name}\n{title}",
            xlabel="$t$ (s)" if seconds is not None else "index",
        )
        fig_key = "error_" + pair.est_name.replace("/", "_")
        plot_collection.add_figure(fig_key, fig_err)

        if len(results) == 1:
            plot.traj_colormap(
                ax_traj,
                traj_est,
                error,
                mode,
                min_map=stats["min"],
                max_map=stats["max"],
                title="error mapped onto trajectory",
            )
            colormap_used = True
        else:
            plot.traj(ax_traj, mode, traj_est, "-", f"C{i}", pair.est_name)

    if not colormap_used:
        ax_traj.set_title("trajectories")
        ax_traj.legend()

    plot_collection.add_figure("trajectories", fig_traj)

    if save_path is not None:
        plot_collection.export(str(save_path), confirm_overwrite=False)
    if show:
        plot_collection.show()
    else:
        plot_collection.close()


def save_pair_results(pair: PairEvaluation, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stem = pair.est_name.replace("/", "_")
    if stem.endswith(".txt") or stem.endswith(".csv"):
        stem = Path(stem).stem
    if pair.ape is not None:
        file_interface.save_res_file(
            str(directory / f"{stem}_ape.zip"), pair.ape, confirm_overwrite=False
        )
    if pair.rpe is not None:
        file_interface.save_res_file(
            str(directory / f"{stem}_rpe.zip"), pair.rpe, confirm_overwrite=False
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TUM trajectories against ground truth using evo (APE / RPE).",
    )
    parser.add_argument(
        "--ref",
        nargs="+",
        required=True,
        help="Ground-truth TUM file(s). Multiple files are merged when a single estimate is given.",
    )
    parser.add_argument(
        "--est",
        nargs="+",
        required=True,
        help="Estimated TUM file(s) and/or session directories. A directory is "
             "searched for odom_tum.txt in the folder itself and in policy "
             "subfolders (e.g. results/nc/quad-medium__quad-easy/).",
    )
    parser.add_argument(
        "--merge-refs",
        action="store_true",
        help="Merge all reference files by timestamp before associating (default when N refs and 1 est).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["ape", "rpe"],
        choices=["ape", "rpe"],
        help="Which metrics to compute (default: ape rpe).",
    )

    algo = parser.add_argument_group("algorithm options")
    algo.add_argument(
        "-r",
        "--pose_relation",
        default="trans_part",
        choices=list(POSE_RELATION_MAP),
        help="Pose relation used by APE/RPE (default: trans_part).",
    )
    align = algo.add_mutually_exclusive_group()
    align.add_argument(
        "-a",
        "--align",
        action="store_true",
        help="SE(3) Umeyama alignment before computing errors.",
    )
    align.add_argument(
        "--align_origin",
        action="store_true",
        help="Align the estimate origin to the reference origin.",
    )
    algo.add_argument(
        "-s",
        "--correct_scale",
        action="store_true",
        help="Correct scale with Umeyama's method (Sim(3) if combined with --align).",
    )
    algo.add_argument(
        "--n_to_align",
        type=int,
        default=-1,
        help="Number of poses from the start used for Umeyama alignment (default: all).",
    )
    algo.add_argument(
        "--project_to_plane",
        choices=["xy", "xz", "yz"],
        default=None,
        help="Project trajectories to a 2D plane after alignment.",
    )
    algo.add_argument(
        "--t_max_diff",
        type=float,
        default=0.1,
        help="Max timestamp difference (s) for association (default: 0.1).",
    )
    algo.add_argument(
        "--t_offset",
        type=float,
        default=0.0,
        help="Constant timestamp offset applied to the estimate during association.",
    )
    algo.add_argument("--t_start", type=float, default=None, help="Only use reference timestamps >= this time.")
    algo.add_argument("--t_end", type=float, default=None, help="Only use reference timestamps <= this time.")
    algo.add_argument("--delta", type=float, default=1.0, help="RPE pose-pair delta (default: 1).")
    algo.add_argument(
        "--delta_unit",
        default="f",
        choices=list(DELTA_UNIT_MAP),
        help="RPE delta unit: f=frames, m=meters, d=degrees, r=radians (default: f).",
    )
    algo.add_argument("--delta_tol", type=float, default=0.1, help="Relative RPE delta tolerance.")
    algo.add_argument("--all_pairs", action="store_true", help="Use all RPE pairs instead of consecutive pairs.")

    output = parser.add_argument_group("output options")
    output.add_argument("-p", "--plot", action="store_true", help="Show matplotlib plots.")
    output.add_argument(
        "--plot_mode",
        default="xy",
        choices=["xy", "xz", "yx", "yz", "zx", "zy", "xyz"],
        help="Trajectory plot projection (default: xy).",
    )
    output.add_argument("--save_plot", default=None, help="Path to export plots (e.g. errors.pdf).")
    output.add_argument("--save_results", default=None, help="Directory to write evo .zip result files.")
    output.add_argument("--save_table", default=None, help="CSV path for a summary statistics table.")
    output.add_argument("-v", "--verbose", action="store_true", help="Verbose evo logging.")
    output.add_argument("--debug", action="store_true", help="Debug logging.")
    output.add_argument("--silent", action="store_true", help="Suppress metric printout (table still printed).")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evo_log.configure_logging(
        verbose=args.verbose or args.debug,
        silent=args.silent or not (args.verbose or args.debug),
        debug=args.debug,
    )

    plane = Plane(args.project_to_plane) if args.project_to_plane else None
    est_files = resolve_estimate_files(args.est)
    if not args.silent:
        print(SEP)
        print(f"Estimate files ({len(est_files)}):")
        for path in est_files:
            print(f"  {path}")

    results = evaluate(
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
        rpe_delta=args.delta,
        rpe_delta_unit=DELTA_UNIT_MAP[args.delta_unit],
        rpe_delta_tol=args.delta_tol,
        rpe_all_pairs=args.all_pairs,
        project_to_plane=plane,
    )

    if not args.silent:
        for pair in results:
            _print_result(pair)

    table = results_to_table(results)
    print(SEP)
    print("Summary")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.save_table:
        table_path = Path(args.save_table)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_path, index=False)
        print(f"Wrote summary table to {table_path}")

    if args.save_results:
        out_dir = Path(args.save_results)
        for pair in results:
            save_pair_results(pair, out_dir)
        print(f"Wrote evo result zips to {out_dir}")

    if args.plot or args.save_plot:
        plot_results(
            results,
            plot_mode=args.plot_mode,
            show=args.plot,
            save_path=args.save_plot,
        )
        if args.save_plot:
            print(f"Wrote plots to {args.save_plot}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
