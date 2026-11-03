#!/usr/bin/env python3
"""Evaluate per-policy resource usage for saved Blocklio sessions.

Computes, per policy folder found under the given session directory:

* Average optimization time (ms) and standard deviation
  from ``timeToOptimize.txt``
* Average time to calculate metrics and decision state (ms) and
  standard deviation from ``timeToCalcMetricsAndDecision.txt``, reported
  as three views: combined total, localization-only (lower k-means cluster),
  and mapping-only (upper k-means cluster)
* Total number of nodes in the final factor graph (from ``values.gtsam``)
* Final factor-graph memory size (MB) from ``graph.gtsam`` and
  ``values.gtsam`` (loaded and re-serialized to measure byte length)
* Final point-cloud memory size (MB) from the global scans store (only the keys
  that appear in ``values.gtsam`` are loaded and summed)

Example
-------
    python -m evaluations.resource_evaluations \\
        --dir results/nc/quad-medium__quad-easy \\
        --save_table resources.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEP = "-" * 80

# ---------------------------------------------------------------------------
# Sentinel file that indicates a directory is a policy output folder.
# A directory is treated as a policy folder when it contains at least one of
# these files.
# ---------------------------------------------------------------------------
_POLICY_SENTINEL_FILES = ("graph.gtsam", "values.gtsam", "odom_tum.txt")

# Default global scans store (see Params.scansRootDir)
DEFAULT_SCANS_DIR = Path(__file__).resolve().parents[1] / "results" / "scans"

# Both spellings of the metrics/decision timing file have been observed.
_METRIC_TIMING_CANDIDATES = (
    "timeToCalcMetricsAndDecision.txt",
    "timeToCalcMetricsAndDecisionState.txt",
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PolicyResources:
    """Resource measurements for a single policy folder."""

    policy_name: str
    path: Path

    opt_times_ms: np.ndarray | None = None        # per-update optimization time
    metric_times_ms: np.ndarray | None = None     # per-update metric+decision time (all)
    metric_loc_times_ms: np.ndarray | None = None # lower k-means cluster (localization)
    metric_map_times_ms: np.ndarray | None = None # upper k-means cluster (mapping)
    graph_num_nodes: int | None = None
    graph_memory_MB: float | None = None
    pcd_memory_MB: float | None = None

    def stats_row(self, split_metric: bool = False) -> dict:
        def _mean(arr): return float(np.mean(arr)) if arr is not None and len(arr) else float("nan")
        def _std(arr):  return float(np.std(arr))  if arr is not None and len(arr) else float("nan")
        def _n(arr):    return int(len(arr))         if arr is not None else 0

        row: dict = {
            "policy":           self.policy_name,
            "opt_mean_ms":      _mean(self.opt_times_ms),
            "opt_std_ms":       _std(self.opt_times_ms),
            "n_opt":            _n(self.opt_times_ms),
            "metric_mean_ms":   _mean(self.metric_times_ms),
            "metric_std_ms":    _std(self.metric_times_ms),
            "n_metric":         _n(self.metric_times_ms),
        }
        if split_metric:
            row.update({
                "metric_loc_mean_ms": _mean(self.metric_loc_times_ms),
                "metric_loc_std_ms":  _std(self.metric_loc_times_ms),
                "n_metric_loc":       _n(self.metric_loc_times_ms),
                "metric_map_mean_ms": _mean(self.metric_map_times_ms),
                "metric_map_std_ms":  _std(self.metric_map_times_ms),
                "n_metric_map":       _n(self.metric_map_times_ms),
            })
        row.update({
            "graph_num_nodes":  self.graph_num_nodes if self.graph_num_nodes is not None else float("nan"),
            "graph_memory_MB":  self.graph_memory_MB if self.graph_memory_MB is not None else float("nan"),
            "pcd_memory_MB":    self.pcd_memory_MB   if self.pcd_memory_MB   is not None else float("nan"),
        })
        return row


# ---------------------------------------------------------------------------
# Policy folder discovery
# ---------------------------------------------------------------------------

def _is_policy_dir(path: Path) -> bool:
    """Return True if *path* looks like a saved policy output folder."""
    return path.is_dir() and any((path / f).is_file() for f in _POLICY_SENTINEL_FILES)


def resolve_policy_dirs(dir_args: list[str | Path]) -> list[Path]:
    """Expand the ``--dir`` arguments into a deduplicated list of policy dirs.

    Rules (mirrors ``resolve_estimate_files`` in ``error_evaluations.py``):

    1. If the given path is already a policy folder, use it directly.
    2. Otherwise search immediate sub-directories for policy folders.
    3. Fall back to a recursive search if step 2 finds nothing.

    Raises ``FileNotFoundError`` if no policy folders are found at all.
    """
    found: list[Path] = []
    seen: set[Path] = set()

    for raw in dir_args:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if _is_policy_dir(path):
            candidates = [path]
        elif path.is_dir():
            # Immediate sub-directories first (the common case)
            candidates = sorted(p for p in path.iterdir() if _is_policy_dir(p))
            if not candidates:
                # Recursive fallback
                candidates = sorted(
                    p for p in path.rglob("*") if _is_policy_dir(p)
                )
            if not candidates:
                raise FileNotFoundError(
                    f"No policy output folders found under {path}"
                )
        else:
            raise FileNotFoundError(f"Not a directory: {path}")

        for c in candidates:
            key = c.resolve()
            if key in seen:
                continue
            seen.add(key)
            found.append(c)

    if not found:
        raise FileNotFoundError("No policy output folders found.")
    return found


def unique_policy_labels(paths: list[Path]) -> list[str]:
    """Return display labels that are unique even when folder names collide."""
    labels = [p.name for p in paths]
    if len(set(labels)) == len(labels):
        return labels
    labels = [f"{p.parent.name}/{p.name}" for p in paths]
    if len(set(labels)) == len(labels):
        return labels
    return [str(p) for p in paths]


# ---------------------------------------------------------------------------
# Bimodal split (localization vs. mapping)
# ---------------------------------------------------------------------------

def _split_bimodal_1d(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Partition a 1-D array into a lower and upper cluster via k-means (k=2).

    Centroids are initialised at the sample min and max and iterated until
    convergence (at most 200 steps).  The cluster with the smaller centroid is
    returned first (localization / lower), the larger one second
    (mapping / upper).

    Returns ``(lower, upper)`` — either may be empty if all samples collapse
    into one cluster.
    """
    if len(data) < 2:
        return data.copy(), np.array([], dtype=data.dtype)

    c1, c2 = float(data.min()), float(data.max())
    for _ in range(200):
        # Each sample belongs to the closer centroid; ties go to c1 (lower).
        in_c1 = np.abs(data - c1) <= np.abs(data - c2)
        new_c1 = float(data[in_c1].mean())  if in_c1.any()  else c1
        new_c2 = float(data[~in_c1].mean()) if (~in_c1).any() else c2
        if np.isclose(c1, new_c1) and np.isclose(c2, new_c2):
            break
        c1, c2 = new_c1, new_c2

    lower, upper = data[in_c1], data[~in_c1]
    # Guarantee lower centroid ≤ upper centroid (swap if initialisation
    # happened to converge the other way round).
    if len(lower) and len(upper) and lower.mean() > upper.mean():
        lower, upper = upper, lower
    return lower, upper


# ---------------------------------------------------------------------------
# Per-metric loaders
# ---------------------------------------------------------------------------

def _load_timing_ms(path: Path) -> np.ndarray | None:
    """Load a timing file (one float in seconds per line) and convert to ms."""
    if not path.is_file():
        return None
    try:
        data = np.loadtxt(str(path))
        if data.ndim == 0:
            data = data.reshape(1)
        return data * 1e3
    except Exception as exc:
        warnings.warn(f"Could not load timing file {path}: {exc}")
        return None


def _load_graph_info(policy_dir: Path) -> tuple[float | None, int | None]:
    """Deserialize graph+values; return ``(memory_MB, num_nodes)``.

    *memory_MB* is the combined UTF-8 byte length of the re-serialized graph
    and values strings divided by 1e6.  *num_nodes* is the number of keys in
    ``values.gtsam``.  Both are ``None`` on failure.
    """
    graph_file  = policy_dir / "graph.gtsam"
    values_file = policy_dir / "values.gtsam"

    if not graph_file.is_file() or not values_file.is_file():
        warnings.warn(f"Missing graph.gtsam or values.gtsam in {policy_dir}")
        return None, None

    try:
        import common.fgHelpers as fgtools  # workspace must be on sys.path
        graph  = fgtools.deserializeGraphFromFile(str(graph_file))
        values = fgtools.deserializeValuesFromFile(str(values_file))
        graph_bytes  = len(graph.serialize().encode("utf-8"))
        values_bytes = len(values.serialize().encode("utf-8"))
        memory_MB  = (graph_bytes + values_bytes) / 1e6
        num_nodes  = len(list(values.keys()))
        return memory_MB, num_nodes
    except Exception as exc:
        warnings.warn(f"Could not load graph/values for {policy_dir}: {exc}")
        return None, None


def _load_pcd_memory_MB(policy_dir: Path, scans_dir: Path = DEFAULT_SCANS_DIR) -> float | None:
    """Load the point clouds belonging to this policy and return their memory in MB.

    Scans are loaded like ``SessionManager.LoadSession``: from the global scans
    store ``{scans_dir}/{dataset}/{mission}/{timestamp_us}.pcd``, falling back
    to a legacy ``{policy}/scans/`` or ``{policy.parent}/scans/`` directory.
    Only keys that are present in ``values.gtsam`` are loaded.
    """
    values_file = policy_dir / "values.gtsam"
    timestamps_file = policy_dir / "timestamps.txt"

    try:
        import common.fgHelpers as fgtools  # workspace must be on sys.path

        timestamps = fgtools.loadTimestampsWithKeys(timestamps_file)
        if values_file.is_file():
            values = fgtools.deserializeValuesFromFile(str(values_file))
            keys   = list(values.keys())
        else:
            keys = list(timestamps.keys())  # load all scans in the session

        dataset = fgtools.readSessionDataset(policy_dir)
        if dataset is None:
            warnings.warn(f"No dataset found in metadata.yaml for policy {policy_dir}")
            return None

        pcds = fgtools.loadSessionPCDs(policy_dir, keys, timestamps, scans_dir, dataset)

        total_bytes = 0
        for pcd in pcds.values():
            pts = np.asarray(pcd.points)
            total_bytes += pts.nbytes
            if pcd.has_colors():
                total_bytes += np.asarray(pcd.colors).nbytes
            if pcd.has_normals():
                total_bytes += np.asarray(pcd.normals).nbytes

        return total_bytes / 1e6
    except Exception as exc:
        warnings.warn(f"Could not measure PCD memory for {policy_dir}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(policy_dir: Path, label: str, scans_dir: Path = DEFAULT_SCANS_DIR) -> PolicyResources:
    """Load all resource metrics for a single policy folder."""
    res = PolicyResources(policy_name=label, path=policy_dir)

    # Optimization timing
    res.opt_times_ms = _load_timing_ms(policy_dir / "timeToOptimize.txt")
    if res.opt_times_ms is None:
        logger.warning("No timeToOptimize.txt found in %s", policy_dir)
    elif res.policy_name == "RELATIVE_COVARIANCE":
        res.opt_times_ms = res.opt_times_ms[res.opt_times_ms < 1e3] # Only keep the ones that are less than 1 second. There were some weird outliers.

    # Metric + decision timing (try both spellings)
    for candidate in _METRIC_TIMING_CANDIDATES:
        arr = _load_timing_ms(policy_dir / candidate)
        if arr is not None:
            res.metric_times_ms = arr
            break
    if res.metric_times_ms is None:
        logger.warning("No metric timing file found in %s", policy_dir)

    # Split metric timing into localization (lower) and mapping (upper) clusters.
    if res.metric_times_ms is not None and len(res.metric_times_ms) >= 2:
        res.metric_loc_times_ms, res.metric_map_times_ms = _split_bimodal_1d(
            res.metric_times_ms
        )

    # Factor-graph memory and node count (single deserialization)
    res.graph_memory_MB, res.graph_num_nodes = _load_graph_info(policy_dir)

    # Point-cloud memory
    res.pcd_memory_MB = _load_pcd_memory_MB(policy_dir, scans_dir)

    return res


def evaluate(dir_args: list[str | Path], scans_dir: Path = DEFAULT_SCANS_DIR) -> list[PolicyResources]:
    """Evaluate all policy folders discovered under ``dir_args``."""
    policy_dirs = resolve_policy_dirs(dir_args)
    labels = unique_policy_labels(policy_dirs)
    results = []
    for policy_dir, label in zip(policy_dirs, labels):
        logger.info("%s\nEvaluating policy: %s  (%s)", SEP, label, policy_dir)
        results.append(evaluate_policy(policy_dir, label, scans_dir))
    return results


def results_to_table(results: list[PolicyResources], split_metric: bool = False) -> pd.DataFrame:
    return pd.DataFrame([r.stats_row(split_metric=split_metric) for r in results])


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_result(res: PolicyResources, split_metric: bool = False) -> None:
    print(SEP)
    print(f"policy : {res.policy_name}")
    print(f"path   : {res.path}")

    if res.opt_times_ms is not None and len(res.opt_times_ms):
        print(f"  optimization time : mean={np.mean(res.opt_times_ms):.3f} ms, "
              f"std={np.std(res.opt_times_ms):.3f} ms, "
              f"n={len(res.opt_times_ms)}")
    else:
        print("  optimization time : N/A")

    if res.metric_times_ms is not None and len(res.metric_times_ms):
        label = "(total)" if split_metric else "      "
        print(f"  metric+decision {label}: mean={np.mean(res.metric_times_ms):.3f} ms, "
              f"std={np.std(res.metric_times_ms):.3f} ms, "
              f"n={len(res.metric_times_ms)}")
        if split_metric:
            loc = res.metric_loc_times_ms
            if loc is not None and len(loc):
                print(f"  metric+decision (loc)  : mean={np.mean(loc):.3f} ms, "
                      f"std={np.std(loc):.3f} ms, "
                      f"n={len(loc)}")
            map_ = res.metric_map_times_ms
            if map_ is not None and len(map_):
                print(f"  metric+decision (map)  : mean={np.mean(map_):.3f} ms, "
                      f"std={np.std(map_):.3f} ms, "
                      f"n={len(map_)}")
    else:
        print("  metric+decision        : N/A")

    if res.graph_num_nodes is not None:
        print(f"  graph nodes       : {res.graph_num_nodes}")
    else:
        print("  graph nodes       : N/A")

    if res.graph_memory_MB is not None:
        print(f"  graph memory      : {res.graph_memory_MB:.3f} MB")
    else:
        print("  graph memory      : N/A")

    if res.pcd_memory_MB is not None:
        print(f"  point-cloud memory: {res.pcd_memory_MB:.3f} MB")
    else:
        print("  point-cloud memory: N/A")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate per-policy resource usage for saved Blocklio sessions.\n"
            "Reports optimization timing, metric/decision timing, factor-graph "
            "memory, and point-cloud memory for each policy folder found."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        nargs="+",
        required=True,
        metavar="PATH",
        help=(
            "Session directory (or policy directory) to evaluate. "
            "A session directory is searched for immediate policy sub-folders "
            "(e.g. INLIER_RATIO/, DISTANCE/) then falls back to a recursive search."
        ),
    )
    parser.add_argument(
        "--scans_dir",
        default=str(DEFAULT_SCANS_DIR),
        metavar="PATH",
        help="Global scans directory laid out as <scans_dir>/<dataset>/<mission>/<timestamp_us>.pcd "
             f"(default: {DEFAULT_SCANS_DIR}).",
    )
    parser.add_argument(
        "--save_table",
        default=None,
        metavar="CSV",
        help="Optional path to write the summary table as a CSV file.",
    )
    parser.add_argument(
        "--split_metric",
        action="store_true",
        help=(
            "Split metric+decision timing into localization (lower) and mapping "
            "(upper) clusters via k-means and report them separately. "
            "Adds loc/map columns to the table and detail printout."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Suppress the per-policy detail printout (summary table is still printed).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    try:
        policy_dirs = resolve_policy_dirs(args.dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not args.silent:
        print(SEP)
        print(f"Policy folders ({len(policy_dirs)}):")
        for p in policy_dirs:
            print(f"  {p}")

    results = evaluate(args.dir, Path(args.scans_dir))

    if not args.silent:
        for res in results:
            _print_result(res, split_metric=args.split_metric)

    table = results_to_table(results, split_metric=args.split_metric)
    print(SEP)
    print("Summary")
    print(
        table.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
            na_rep="N/A",
        )
    )

    if args.save_table:
        table_path = Path(args.save_table)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_path, index=False)
        print(f"Wrote summary table to {table_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
